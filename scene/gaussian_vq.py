import torch
import numpy as np
import os
import re
import abc
from sklearn.cluster import MiniBatchKMeans as KMeans
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

    @abc.abstractmethod
    def dequantize(self, attr: Attribute, quant, i=0):
        pass

    def load_and_test(self, dirpath, log2_clusters: int, attr: Attribute, i=0):
        self.load_codebook(dirpath, log2_clusters, attr, i)
        self.dequantize(attr, self.quantize(attr, i))

    def load_and_test_all(self, dirpath,
                          log2_clusters_scaling,
                          log2_clusters_rotation,
                          log2_clusters_features_dc,
                          log2_clusters_features_rest,
                          log2_clusters_opacity):
        self.load_and_test(dirpath, log2_clusters_scaling, Attribute.scaling)
        self.load_and_test(dirpath, log2_clusters_rotation, Attribute.rotation)
        self.load_and_test(dirpath, log2_clusters_features_dc, Attribute.features_dc)
        self.load_and_test(dirpath, log2_clusters_features_rest, Attribute.features_rest)
        self.load_and_test(dirpath, log2_clusters_opacity, Attribute.opacity)


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
        print(f"dequantized mean:       {mean_dequantized}")
        print(f"dequantize  mean shift: {mean - mean_dequantized}")
