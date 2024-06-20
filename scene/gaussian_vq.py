import torch
import numpy as np
import os
import abc
from plyfile import PlyData, PlyElement
from .gaussian_model import GaussianModel
from enum import Enum


class Attribute(Enum):
    scaling = 'scaling'
    rotation = 'rotation'
    features_dc = 'features_dc'
    features_rest = 'features_rest'
    opacity = 'opacity'

    def __str__(self):
        return self.value


class VQGaussianModel(GaussianModel, metaclass=abc.ABCMeta):

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

    def save_vq_ply(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)

        xyz = self._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = self.quantize(Attribute.features_dc).detach().unsqueeze(-1).cpu().numpy()
        f_rest = self.quantize(Attribute.features_rest).detach().unsqueeze(-1).cpu().numpy()
        opacities = self.quantize(Attribute.opacity).detach().unsqueeze(-1).cpu().numpy()
        scale = self.quantize(Attribute.scaling).detach().unsqueeze(-1).cpu().numpy()
        rotation = self.quantize(Attribute.rotation).detach().unsqueeze(-1).cpu().numpy()

        dtype_full = [
            ('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
            ('scale', 'i4'),
            ('rot', 'i4'),
            ('opacity', 'i4'),
            ('f_dc', 'i4'),
            ('f_rest', 'i4')]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, normals, scale, rotation, opacities, f_dc, f_rest), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)

    @abc.abstractmethod
    def dequantize(self, attr: Attribute, quant, i=0):
        pass

    def test(self, attr: Attribute, i=0):
        self.dequantize(attr, self.quantize(attr, i))

    def load_vq_ply(self, path):
        plydata = PlyData.read(path)

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        self._xyz.requires_grad_(False)
        self._xyz[...] = torch.FloatTensor(xyz)
        self._xyz.requires_grad_(True)

        scaling = np.asarray(plydata.elements[0]["scale"])
        rotation = np.asarray(plydata.elements[0]["rot"])
        opacity = np.asarray(plydata.elements[0]["opacity"])
        features_dc = np.asarray(plydata.elements[0]["f_dc"])
        features_rest = np.asarray(plydata.elements[0]["f_rest"])
        self.dequantize(Attribute.scaling, scaling)
        self.dequantize(Attribute.rotation, rotation)
        self.dequantize(Attribute.opacity, opacity)
        self.dequantize(Attribute.features_dc, features_dc)
        self.dequantize(Attribute.features_rest, features_rest)

    def test_all(self):
        self.test(Attribute.scaling)
        self.test(Attribute.rotation)
        self.test(Attribute.opacity)
        self.test(Attribute.features_dc)
        self.test(Attribute.features_rest)
