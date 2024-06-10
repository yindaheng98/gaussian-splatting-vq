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

    def kmeans(self, attr: Attribute, log2_clusters: int):
        kmeans = KMeans(n_clusters=2**log2_clusters, random_state=0, n_init="auto")
        data = getattr(self, "_" + str(attr)).detach()
        print(f"{log2_clusters} bit Kmeans for quantization of {attr}. shape: {data.shape}")
        kmeans.fit(data.cpu())
        setattr(self, "kmeans_" + str(attr), kmeans)

    def quantize(self, attr: Attribute):
        print(f"quantize {attr}")
        kmeans = getattr(self, "kmeans_" + str(attr))
        data = getattr(self, "_" + str(attr)).detach()
        return kmeans.predict(data.cpu())

    def dequantize(self, attr: Attribute, quant):
        print(f"dequantize {attr}")
        kmeans = getattr(self, "kmeans_" + str(attr))
        data = getattr(self, "_" + str(attr))
        data.requires_grad_(False)
        data[...] = torch.tensor(kmeans.cluster_centers_[quant], dtype=data.dtype, device=data.device)
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
