import torch
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

    def kmeans(self, attr: Attribute, log2_clusters: int, k=0):
        kmeans = KMeans(n_clusters=2**log2_clusters, random_state=0, n_init="auto")
        data = getattr(self, "_" + str(attr)).detach()
        if data.ndim <= 2:
            print(f"{log2_clusters} bit Kmeans for quantization of {attr}. shape: {data.shape}")
            kmeans.fit(data.cpu())
            setattr(self, f"kmeans_{attr}", kmeans)
        elif data.ndim == 3:
            if data.shape[1] == 1:
                print(f"{log2_clusters} bit Kmeans for quantization of {attr}. shape: {data.shape}")
                kmeans.fit(data[:, 0, ...].cpu())
                setattr(self, f"kmeans_{attr}", kmeans)
            else:
                print(f"{log2_clusters} bit Kmeans for quantization of {attr} no.{k}. shape: {data.shape}")
                kmeans.fit(data[:, k, ...].cpu())
                setattr(self, f"kmeans_{attr}_{k}", kmeans)
        else:
            raise ValueError("Not supported")

    def quantize(self, attr: Attribute, k=0):
        print(f"quantize {attr}")
        data = getattr(self, "_" + str(attr)).detach()
        if data.ndim <= 2:
            kmeans = getattr(self, f"kmeans_{attr}")
            quant = kmeans.predict(data.cpu())
        elif data.ndim == 3:
            if data.shape[1] == 1:
                kmeans = getattr(self, f"kmeans_{attr}")
                quant = kmeans.predict(data[:, 0, ...].cpu())
            else:
                kmeans = getattr(self, f"kmeans_{attr}_{k}")
                quant = kmeans.predict(data[:, k, ...].cpu())
        else:
            raise ValueError("Not supported")
        return quant

    def dequantize(self, attr: Attribute, quant, k=0):
        print(f"dequantize {attr}")
        data = getattr(self, "_" + str(attr))
        data.requires_grad_(False)
        if data.ndim <= 2:
            kmeans = getattr(self, f"kmeans_{attr}")
            data[...] = torch.tensor(kmeans.cluster_centers_[quant], dtype=data.dtype, device=data.device)
        elif data.ndim == 3:
            if data.shape[1] == 1:
                kmeans = getattr(self, f"kmeans_{attr}")
                data[:, 0, ...] = torch.tensor(kmeans.cluster_centers_[quant], dtype=data.dtype, device=data.device)
            else:
                kmeans = getattr(self, f"kmeans_{attr}_{k}")
                data[:, k, ...] = torch.tensor(kmeans.cluster_centers_[quant], dtype=data.dtype, device=data.device)
        else:
            raise ValueError("Not supported")
        data.requires_grad_(True)

    def quantize_test(self, attr: Attribute, log2_clusters: int):
        self.kmeans(attr, log2_clusters)
        self.dequantize(attr, self.quantize(attr))

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
        self.quantize_test(Attribute.opacity, log2_clusters_opacity)
