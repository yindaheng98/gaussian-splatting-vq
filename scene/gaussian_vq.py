import torch
import numpy as np
import os
import abc
from plyfile import PlyData, PlyElement
from .gaussian_model import GaussianModel
from enum import Enum
import shutil
import re
from utils.spatialsplit_utils import spatial_split, spatial_merge, split_by_blkids, spatial_split_octree


class Attribute(Enum):
    scaling = 'scaling'
    rotation = 'rotation'
    features_dc = 'features_dc'
    features_rest = 'features_rest'
    opacity = 'opacity'

    def __str__(self):
        return self.value


def save_vq_ply(path, xyz, scale, rotation, opacities, f_dc, f_rest):
    normals = np.zeros_like(xyz)
    dtype_full = [
        ('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
        ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
        ('scale', 'i4'),
        ('rot', 'i4'),
        ('opacity', 'i4'),
        ('f_dc', 'i4'),
        ('f_rest', 'i4')]

    elements = np.empty(xyz.shape[0], dtype=dtype_full)
    attributes = np.concatenate((xyz, normals, np.stack((scale, rotation, opacities, f_dc, f_rest), axis=-1)), axis=1)
    elements[:] = list(map(tuple, attributes))
    el = PlyElement.describe(elements, 'vertex')
    PlyData([el]).write(path)


def save_vq_ply_with_tileinfo(path, xyz_rests, xyz_blkids, xyz_tile_size, scale, rotation, opacities, f_dc, f_rest):
    np.savez(
        os.path.join(os.path.dirname(path), "tilesinfo.npz"),
        tile_size=xyz_tile_size, blkids=xyz_blkids)
    save_vq_ply(
        path=path,
        xyz=xyz_rests,
        scale=scale,
        rotation=rotation,
        opacities=opacities,
        f_dc=f_dc,
        f_rest=f_rest,
    )


def load_vq_ply(path):
    plydata = PlyData.read(path)

    xyz_rests = np.stack((np.asarray(plydata.elements[0]["x"]),
                          np.asarray(plydata.elements[0]["y"]),
                          np.asarray(plydata.elements[0]["z"])),  axis=1)
    scale = np.asarray(plydata.elements[0]["scale"])
    rotation = np.asarray(plydata.elements[0]["rot"])
    opacities = np.asarray(plydata.elements[0]["opacity"])
    f_dc = np.asarray(plydata.elements[0]["f_dc"])
    f_rest = np.asarray(plydata.elements[0]["f_rest"])
    return xyz_rests, scale, rotation, opacities, f_dc, f_rest


def load_vq_ply_with_tileinfo(path):
    tileinfo = np.load(os.path.join(os.path.dirname(path), "tilesinfo.npz"))
    xyz_blkids, xyz_tile_size = tileinfo['blkids'], tileinfo['tile_size']
    xyz_rests, scale, rotation, opacities, f_dc, f_rest = load_vq_ply(path)
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

    def _get_features_rest(self, i=0):
        features_rest = self._features_rest.detach()
        match i:
            case 0:
                return features_rest[:, 0:3, ...].reshape(features_rest.shape[0], -1)
            case 1:
                return features_rest[:, 3:8, ...].reshape(features_rest.shape[0], -1)
            case 2:
                return features_rest[:, 8:15, ...].reshape(features_rest.shape[0], -1)

    def get_data(self, attr: Attribute, i=0):
        if attr == Attribute.features_rest:
            return self._get_features_rest(i=i)
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

    def _set_features_rest(self, kdata, i=0):
        with torch.no_grad():
            features_rest = self._features_rest
            match i:
                case 0:
                    features_rest[:, 0:3, ...] = kdata.reshape(features_rest.shape[0], 3, -1)
                case 1:
                    features_rest[:, 3:8, ...] = kdata.reshape(features_rest.shape[0], 5, -1)
                case 2:
                    features_rest[:, 8:15, ...] = kdata.reshape(features_rest.shape[0], 7, -1)

    def set_data(self, attr: Attribute, kdata, i=0):
        if attr == Attribute.features_rest:
            return self._set_features_rest(kdata=kdata, i=i)
        data = getattr(self, "_" + str(attr))
        with torch.no_grad():
            if data.ndim <= 2:
                data[...] = kdata
            elif data.ndim == 3:
                if data.shape[1] == 1:
                    data[:, 0, ...] = kdata
                else:
                    data[:, i, ...] = kdata
            else:
                raise ValueError("Not supported")

    xyz_tile_size = np.array([4, 4, 4])

    def save_vq_ply(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)

        xyz_blkids, xyz_rests = spatial_split(self._xyz.detach(), self.xyz_tile_size)
        save_vq_ply_with_tileinfo(
            path=path,
            xyz_rests=xyz_rests.cpu().numpy(),
            xyz_blkids=xyz_blkids.cpu().numpy(),
            xyz_tile_size=self.xyz_tile_size,
            f_dc=self.quantize(Attribute.features_dc).detach().cpu().numpy(),
            f_rest=self.quantize(Attribute.features_rest).detach().cpu().numpy(),
            opacities=self.quantize(Attribute.opacity).detach().cpu().numpy(),
            scale=self.quantize(Attribute.scaling).detach().cpu().numpy(),
            rotation=self.quantize(Attribute.rotation).detach().cpu().numpy(),
        )

    def save_vq_split_ply(self, path):
        os.makedirs(path, exist_ok=True)
        shutil.rmtree(path)
        os.makedirs(path, exist_ok=True)

        blkids, blksizes, xyz_rests = spatial_split_octree(self._xyz.detach())
        blkidsizes = torch.concat([blkids, blksizes], dim=1)
        distinct_blkidsizes, item_idx, xyz_rest_blks, f_dc_blks, f_rest_blks, opacities_blks, scale_blks, rotation_blks = split_by_blkids(
            blkidsizes,
            xyz_rests,
            self.quantize(Attribute.features_dc).detach(),
            self.quantize(Attribute.features_rest).detach(),
            self.quantize(Attribute.opacity).detach(),
            self.quantize(Attribute.scaling).detach(),
            self.quantize(Attribute.rotation).detach())
        for i in range(distinct_blkidsizes.shape[0]):
            blkid = distinct_blkidsizes[i, :blkids.shape[1]]
            blksize = distinct_blkidsizes[i, blkids.shape[1]:]
            blkid_str = "_".join([str(_) for _ in blkid.cpu().numpy().tolist()])
            blksize_str = "_".join([str(_) for _ in blksize.cpu().numpy().tolist()])
            name = f"size_{blksize_str}_id_{blkid_str}"
            save_vq_ply(
                os.path.join(path, name + ".ply"),
                xyz=xyz_rest_blks[i].cpu().numpy(),
                f_dc=f_dc_blks[i].cpu().numpy(),
                f_rest=f_rest_blks[i].cpu().numpy(),
                opacities=opacities_blks[i].cpu().numpy(),
                scale=scale_blks[i].cpu().numpy(),
                rotation=rotation_blks[i].cpu().numpy(),
            )
            np.savez_compressed(os.path.join(path, name + ".npz"), idx=item_idx[i].cpu().numpy())

    @abc.abstractmethod
    def dequantize(self, attr: Attribute, quant, i=0):
        pass

    def test(self, attr: Attribute, i=0):
        self.dequantize(attr, self.quantize(attr, i))

    def load_vq_ply(self, path):
        xyz_rests, xyz_blkids, xyz_tile_size, scale, rotation, opacities, f_dc, f_rest = load_vq_ply_with_tileinfo(path)
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

    def load_vq_split_ply(self, path):
        xyz_blks, f_dc_blks, f_rest_blks, opacities_blks, scale_blks, rotation_blks = [], [], [], [], [], []
        item_idx_blks = []
        for entry in os.scandir(path):
            find = re.findall(r"^size_([-0-9]+)_([-0-9]+)_([-0-9]+)_id_([-0-9]+)_([-0-9]+)_([-0-9]+).ply$", entry.name)
            if not len(find) == 1 or not len(find[0]) == 6:
                continue
            tile_size = [int(i) for i in find[0][0:3]]
            blkid = [int(i) for i in find[0][3:6]]
            xyz_rests, scale, rotation, opacities, f_dc, f_rest = load_vq_ply(entry.path)
            xyz_rests = torch.FloatTensor(xyz_rests).to(self._xyz.device)
            xyz_blkids = torch.from_numpy(np.asarray([blkid])).to(self._xyz.device)
            xyz = spatial_merge(xyz_blkids, xyz_rests, tile_size)
            xyz_blks.append(xyz)
            f_dc_blks.append(f_dc)
            f_rest_blks.append(f_rest)
            opacities_blks.append(opacities)
            scale_blks.append(scale)
            rotation_blks.append(rotation)
            item_idx_blks.append(np.load(entry.path[:-4] + ".npz")["idx"])

        item_idx = np.concatenate(item_idx_blks, axis=0)
        reverse_idx = item_idx.argsort()

        xyz = torch.concat(xyz_blks, dim=0)[reverse_idx, ...]
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

        f_dc = np.concatenate(f_dc_blks, axis=0)[reverse_idx]
        f_rest = np.concatenate(f_rest_blks, axis=0)[reverse_idx]
        opacities = np.concatenate(opacities_blks, axis=0)[reverse_idx]
        scale = np.concatenate(scale_blks, axis=0)[reverse_idx]
        rotation = np.concatenate(rotation_blks, axis=0)[reverse_idx]

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
