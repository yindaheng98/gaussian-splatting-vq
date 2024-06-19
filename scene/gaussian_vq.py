import torch
import numpy as np
import os
import re
import abc
from sklearn.cluster import MiniBatchKMeans as KMeans
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
    method = None

    def get_filename(self, log2_clusters: int, attr: Attribute, i=0):
        data = getattr(self, "_" + str(attr))
        if data.ndim <= 2:
            name = f"{self.method}_{log2_clusters}_{attr}"
        elif data.ndim == 3:
            if data.shape[1] == 1:
                name = f"{self.method}_{log2_clusters}_{attr}"
            else:
                name = f"{self.method}_{log2_clusters}_{attr}_{i}"
        else:
            raise ValueError("Not supported")
        return name

    def get_name(self, attr: Attribute, i=0):
        data = getattr(self, "_" + str(attr))
        if data.ndim <= 2:
            name = f"{self.method}_{attr}"
        elif data.ndim == 3:
            if data.shape[1] == 1:
                name = f"{self.method}_{attr}"
            else:
                name = f"{self.method}_{attr}_{i}"
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
    def get_log2_clusters(self, attr: Attribute, i=0):
        pass

    def save_codebook(self, dirpath, attr: Attribute, i=0):
        os.makedirs(dirpath, exist_ok=True)
        log2_clusters = self.get_log2_clusters(attr, i)
        path = os.path.join(dirpath, self.get_filename(log2_clusters, attr, i) + ".npz")
        kmeans = getattr(self, self.get_name(attr, i))
        print(f"save codebook {path}.")
        np.savez(path, codebook=kmeans.cpu().numpy())

    def load_codebook(self, dirpath, log2_clusters: int, attr: Attribute, i=0):
        path = os.path.join(dirpath, self.get_filename(log2_clusters, attr, i) + ".npz")
        data = self.get_data(attr, i)
        print(f"load codebook {path}.")
        kmeans = torch.FloatTensor(np.load(path)["codebook"]).to(data.device)
        setattr(self, self.get_name(attr, i), kmeans)

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

    def construct_list_of_vq_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz', "scale", "rot", "opacity", "f_dc", "f_rest"]
        return l

    def save_vq_ply(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)

        xyz = self._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = self.quantize(Attribute.features_dc).detach().unsqueeze(-1).cpu().numpy()
        f_rest = self.quantize(Attribute.features_rest).detach().unsqueeze(-1).cpu().numpy()
        opacities = self.quantize(Attribute.opacity).detach().unsqueeze(-1).cpu().numpy()
        scale = self.quantize(Attribute.scaling).detach().unsqueeze(-1).cpu().numpy()
        rotation = self.quantize(Attribute.rotation).detach().unsqueeze(-1).cpu().numpy()

        dtype_full = [(attribute, 'i4') for attribute in self.construct_list_of_vq_attributes()]

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


class KMeansGaussianModel(VQGaussianModel):
    method = "kmeans"
    quant_batch = 4096

    @staticmethod
    def kmeans_batch(log2_clusters, datasize):
        return datasize // 10

    def build_codebook(self, log2_clusters: int, attr: Attribute, i=0):
        data = self.get_data(attr, i).detach()
        kmeans = KMeans(n_clusters=2**log2_clusters, init='random', random_state=0,
                        n_init="auto", verbose=1, batch_size=self.kmeans_batch(log2_clusters, data.shape[0]))
        print(f"{log2_clusters} bit Kmeans {self.get_name(attr, i)}. shape: {data.shape}")
        kmeans.fit(data.cpu())
        setattr(self, self.get_name(attr, i), torch.FloatTensor(kmeans.cluster_centers_).to(data.device))
        setattr(self, f"log2_clusters_{self.get_name(attr, i)}", log2_clusters)

    def get_log2_clusters(self, attr: Attribute, i=0):
        return getattr(self, f"log2_clusters_{self.get_name(attr, i)}")

    def quantize(self, attr: Attribute, i=0):
        kmeans = getattr(self, self.get_name(attr, i))
        data = self.get_data(attr, i).detach()
        print(f"quantize by {self.get_name(attr, i)}. shape: {data.shape}")
        quantized = torch.zeros(data.shape[0], dtype=torch.int32, device=data.device)
        for i in range(0, data.shape[0], self.quant_batch):
            step = self.quant_batch if i+self.quant_batch < data.shape[0] else i+self.quant_batch - data.shape[0]
            dist = torch.norm(data[i:i+step, ...].unsqueeze(1) - kmeans.unsqueeze(0), p=2, dim=2)
            quantized[i:i+step] = dist.argmin(dim=1)
        return quantized

    def dequantize(self, attr: Attribute, quant, i=0):
        kmeans = getattr(self, self.get_name(attr, i))
        data = self.get_data(attr, i).detach()
        dequantized = kmeans[quant]
        mean = torch.abs(data).mean(dim=0).cpu().numpy()
        mean_dequantized = torch.abs(dequantized).mean(dim=0).cpu().numpy()
        loss = torch.abs(dequantized - data).mean(dim=0).cpu().numpy()
        self.set_data(attr, dequantized, i)
        print(f"dequantized by {self.get_name(attr, i)}.")
        print(f"dequantized loss:       {loss}")
        print(f"dequantized rel loss:   {loss / mean_dequantized}")
        print(f"dequantized mean:       {mean_dequantized}")
        print(f"dequantize  mean shift: {mean - mean_dequantized}")
