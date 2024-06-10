import torch
import pickle
import os
from sklearn.cluster import KMeans
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


class VQGaussianModel(GaussianModel):

    def kmeans_name(self, attr: Attribute, k=0):
        data = getattr(self, "_" + str(attr))
        if data.ndim <= 2:
            name = f"kmeans_{attr}"
        elif data.ndim == 3:
            if data.shape[1] == 1:
                name = f"kmeans_{attr}"
            else:
                name = f"kmeans_{attr}_{k}"
        else:
            raise ValueError("Not supported")
        return name

    def kmeans_data(self, attr: Attribute, k=0):
        data = getattr(self, "_" + str(attr)).detach()
        if data.ndim <= 2:
            kdata = data
        elif data.ndim == 3:
            if data.shape[1] == 1:
                kdata = data[:, 0, ...]
            else:
                kdata = data[:, k, ...]
        else:
            raise ValueError("Not supported")
        return kdata

    def kmeans(self, attr: Attribute, log2_clusters: int, k=0):
        kmeans = KMeans(n_clusters=2**log2_clusters, random_state=0, n_init="auto")
        data = self.kmeans_data(attr, k)
        print(f"{log2_clusters} bit Kmeans {self.kmeans_name(attr, k)}. shape: {data.shape}")
        kmeans.fit(data.cpu())
        setattr(self, self.kmeans_name(attr, k), kmeans)

    def save_kmeans(self, dirpath, attr: Attribute, k=0):
        path = os.path.join(dirpath, self.kmeans_name(attr, k) + ".pkl")
        kmeans = getattr(self, self.kmeans_name(attr, k))
        print(f"save {path}.")
        with open(path, "wb") as f:
            pickle.dump(kmeans, f)

    def quantize(self, attr: Attribute, k=0):
        kmeans = getattr(self, self.kmeans_name(attr, k))
        data = self.kmeans_data(attr, k)
        print(f"quantize by {self.kmeans_name(attr, k)}. shape: {data.shape}")
        return kmeans.predict(data.cpu())

    def dequantize(self, attr: Attribute, quant, k=0):
        kmeans = getattr(self, self.kmeans_name(attr, k))
        data = getattr(self, "_" + str(attr))
        print(f"dequantize by {self.kmeans_name(attr, k)}.")
        data.requires_grad_(False)
        if data.ndim <= 2:
            data[...] = torch.tensor(kmeans.cluster_centers_[quant], dtype=data.dtype, device=data.device)
        elif data.ndim == 3:
            if data.shape[1] == 1:
                data[:, 0, ...] = torch.tensor(kmeans.cluster_centers_[quant], dtype=data.dtype, device=data.device)
            else:
                data[:, k, ...] = torch.tensor(kmeans.cluster_centers_[quant], dtype=data.dtype, device=data.device)
        else:
            raise ValueError("Not supported")
        data.requires_grad_(True)

    def quantize_test(self, attr: Attribute, log2_clusters: int, k=0):
        self.kmeans(attr, log2_clusters)
        self.dequantize(attr, self.quantize(attr, k))

    def quantize_test_all(self,
                          log2_clusters_scaling,
                          log2_clusters_rotation,
                          log2_clusters_features_dc,
                          log2_clusters_features_rest,
                          log2_clusters_opacity):
        self.quantize_test(Attribute.scaling, log2_clusters_scaling)
        self.quantize_test(Attribute.rotation, log2_clusters_rotation)
        self.quantize_test(Attribute.features_dc, log2_clusters_features_dc)
        self.quantize_test(Attribute.features_rest, log2_clusters_features_rest)
        self._features_rest.requires_grad_(False)
        self._features_rest[:, 1:, ...] = 0
        self._features_rest.requires_grad_(True)
        self.quantize_test(Attribute.opacity, log2_clusters_opacity)
