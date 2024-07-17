import torch
import numpy as np
import os
import abc
from plyfile import PlyData, PlyElement
from .gaussian_model import GaussianModel
from enum import Enum
from utils.spatialsplit_utils import spatial_split, spatial_merge, split_by_blkids, spatial_split_octree


class Attribute(Enum):
    scaling = 'scaling'
    rotation = 'rotation'
    features_dc = 'features_dc'
    features_rest = 'features_rest'
    opacity = 'opacity'

    def __str__(self):
        return self.value


def save_vq_ply(path, xyz_rests, xyz_blkids, xyz_tile_size, scale, rotation, opacities, f_dc, f_rest):
    np.savez(
        os.path.join(os.path.dirname(path), "tilesinfo.npz"),
        tile_size=xyz_tile_size, blkids=xyz_blkids)
    normals = np.zeros_like(xyz_rests)
    dtype_full = [
        ('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
        ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
        ('scale', 'i4'),
        ('rot', 'i4'),
        ('opacity', 'i4'),
        ('f_dc', 'i4'),
        ('f_rest', 'i4')]

    elements = np.empty(xyz_rests.shape[0], dtype=dtype_full)
    attributes = np.concatenate((xyz_rests, normals, scale, rotation, opacities, f_dc, f_rest), axis=1)
    elements[:] = list(map(tuple, attributes))
    el = PlyElement.describe(elements, 'vertex')
    PlyData([el]).write(path)


def load_vq_ply(path):
    plydata = PlyData.read(path)

    xyz_rests = np.stack((np.asarray(plydata.elements[0]["x"]),
                          np.asarray(plydata.elements[0]["y"]),
                          np.asarray(plydata.elements[0]["z"])),  axis=1)
    tileinfo = np.load(os.path.join(os.path.dirname(path), "tilesinfo.npz"))
    xyz_blkids, xyz_tile_size = tileinfo['blkids'], tileinfo['tile_size']
    scale = np.asarray(plydata.elements[0]["scale"])
    rotation = np.asarray(plydata.elements[0]["rot"])
    opacities = np.asarray(plydata.elements[0]["opacity"])
    f_dc = np.asarray(plydata.elements[0]["f_dc"])
    f_rest = np.asarray(plydata.elements[0]["f_rest"])
    return xyz_rests, xyz_blkids, xyz_tile_size, scale, rotation, opacities, f_dc, f_rest


class VQGaussianModel(GaussianModel, metaclass=abc.ABCMeta):

    def _get_filename(self, method, log2_clusters: int, attr: Attribute, i=0):
        data = getattr(self, "_" + str(attr))
        if data.ndim <= 2:
            name = f"{method}_{log2_clusters}_{attr}"
        elif data.ndim == 3:
            if data.shape[1] == 1:
                name = f"{method}_{log2_clusters}_{attr}"
            else:
                name = f"{method}_{log2_clusters}_{attr}_{i}"
        else:
            raise ValueError("Not supported")
        return name

    def _get_name(self, method, attr: Attribute, i=0):
        data = getattr(self, "_" + str(attr))
        if data.ndim <= 2:
            name = f"{method}_{attr}"
        elif data.ndim == 3:
            if data.shape[1] == 1:
                name = f"{method}_{attr}"
            else:
                name = f"{method}_{attr}_{i}"
        else:
            raise ValueError("Not supported")
        return name

    def get_data(self, attr: Attribute, i=0):
        data = getattr(self, "_" + str(attr)).detach()
        if data.ndim <= 2:
            kdata = data
        elif data.ndim == 3:
            if data.shape[1] == 1:
                kdata = data[:, 0, ...]
            else:
                kdata = data[:, i, ...]
        else:
            raise ValueError("Not supported")
        return kdata

    @abc.abstractmethod
    def build_codebook(self, log2_clusters: int, attr: Attribute, i=0):
        pass

    @abc.abstractmethod
    def save_codebook(self, dirpath, attr: Attribute, i=0):
        pass

    @abc.abstractmethod
    def load_codebook(self, dirpath, log2_clusters: int, attr: Attribute, i=0):
        pass

    def load_codebooks(self, dirpath,
                       log2_clusters_scaling,
                       log2_clusters_rotation,
                       log2_clusters_features_dc,
                       log2_clusters_features_rest,
                       log2_clusters_opacity):
        self.load_codebook(dirpath, log2_clusters_scaling, Attribute.scaling)
        self.load_codebook(dirpath, log2_clusters_rotation, Attribute.rotation)
        self.load_codebook(dirpath, log2_clusters_features_dc, Attribute.features_dc)
        self.load_codebook(dirpath, log2_clusters_features_rest, Attribute.features_rest)
        self.load_codebook(dirpath, log2_clusters_opacity, Attribute.opacity)

    @abc.abstractmethod
    def quantize(self, attr: Attribute, i=0):
        pass

    def set_data(self, attr: Attribute, kdata, i=0):
        data = getattr(self, "_" + str(attr))
        data.requires_grad_(False)
        if data.ndim <= 2:
            data[...] = kdata
        elif data.ndim == 3:
            if data.shape[1] == 1:
                data[:, 0, ...] = kdata
            else:
                data[:, i, ...] = kdata
        else:
            raise ValueError("Not supported")
        data.requires_grad_(True)

    xyz_tile_size = np.array([1., 1., 1.])

    def save_vq_ply(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)

        xyz_blkids, xyz_rests = spatial_split(self._xyz.detach(), self.xyz_tile_size)
        save_vq_ply(
            path=path,
            xyz_rests=xyz_rests.cpu().numpy(),
            xyz_blkids=xyz_blkids.cpu().numpy(),
            xyz_tile_size=self.xyz_tile_size,
            f_dc=self.quantize(Attribute.features_dc).detach().unsqueeze(-1).cpu().numpy(),
            f_rest=self.quantize(Attribute.features_rest).detach().unsqueeze(-1).cpu().numpy(),
            opacities=self.quantize(Attribute.opacity).detach().unsqueeze(-1).cpu().numpy(),
            scale=self.quantize(Attribute.scaling).detach().unsqueeze(-1).cpu().numpy(),
            rotation=self.quantize(Attribute.rotation).detach().unsqueeze(-1).cpu().numpy(),
        )

    def save_vq_split_ply(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)

        blkids, xyz_rests = spatial_split_octree(self._xyz.detach())
        distinct_blkids, xyz_blks, f_dc_blks, f_rest_blks, opacities_blks, scale_blks, rotation_blks = split_by_blkids(
            blkids,
            xyz_rests,
            self.quantize(Attribute.features_dc).detach(),
            self.quantize(Attribute.features_rest).detach(),
            self.quantize(Attribute.opacity).detach(),
            self.quantize(Attribute.scaling).detach(),
            self.quantize(Attribute.rotation).detach())
        pass

    @abc.abstractmethod
    def dequantize(self, attr: Attribute, quant, i=0):
        pass

    def test(self, attr: Attribute, i=0):
        self.dequantize(attr, self.quantize(attr, i))

    def load_vq_ply(self, path):
        xyz_rests, xyz_blkids, xyz_tile_size, scale, rotation, opacities, f_dc, f_rest = load_vq_ply(path)
        xyz_rests = torch.FloatTensor(xyz_rests).to(self._xyz.device)
        xyz_blkids = torch.from_numpy(xyz_blkids).to(self._xyz.device)
        xyz = spatial_merge(xyz_blkids, xyz_rests, xyz_tile_size)
        self._xyz.requires_grad_(False)
        mean = torch.abs(self._xyz).mean(dim=0).cpu().numpy()
        mean_dequantized = torch.abs(xyz).mean(dim=0).cpu().numpy()
        loss = torch.abs(xyz - self._xyz).mean(dim=0).cpu().numpy()
        print(f"merge tiles.")
        print(f"xyz loss:       {loss}")
        print(f"xyz rel loss:   {loss / mean_dequantized}")
        print(f"xyz mean:       {mean_dequantized}")
        print(f"xyz mean shift: {mean - mean_dequantized}")
        self._xyz[...] = xyz
        self._xyz.requires_grad_(True)

        self.dequantize(Attribute.scaling, scale)
        self.dequantize(Attribute.rotation, rotation)
        self.dequantize(Attribute.opacity, opacities)
        self.dequantize(Attribute.features_dc, f_dc)
        self.dequantize(Attribute.features_rest, f_rest)

    def test_all(self):
        self.test(Attribute.scaling)
        self.test(Attribute.rotation)
        self.test(Attribute.opacity)
        self.test(Attribute.features_dc)
        self.test(Attribute.features_rest)
