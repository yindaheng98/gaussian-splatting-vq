import os
import torch
from sklearn.cluster import KMeans
from .gaussian_model import GaussianModel


def kmeans_fit(log2_clusters, data: torch.Tensor):
    kmeans = KMeans(n_clusters=2**log2_clusters, random_state=0, n_init="auto")
    kmeans.fit(data.cpu())
    return kmeans


def kmeans_predict(kmeans: KMeans, data: torch.Tensor):
    quant = kmeans.predict(data.cpu())
    return torch.tensor(kmeans.cluster_centers_[quant], dtype=data.dtype, device=data.device)


class VQGaussianModel(GaussianModel):
    def requires_grad_(self, mode: bool):
        print("xyz", self._xyz.shape)
        print("scaling", self._scaling.shape)
        self._scaling.requires_grad_(mode)
        print("rotation", self._rotation.shape)
        self._rotation.requires_grad_(mode)
        print("features_dc", self._features_dc.shape)
        self._features_dc.requires_grad_(mode)
        print("features_rest", self._features_rest.shape)
        self._features_rest.requires_grad_(mode)
        print("opacity", self._opacity.shape)
        self._opacity.requires_grad_(mode)

    def VectorQuant(self,
                    log2_clusters_scaling,
                    log2_clusters_rotation,
                    log2_clusters_features_dc,
                    log2_clusters_features_rest,
                    log2_clusters_opacity):
        self.requires_grad_(False)
        kmeans_scaling = kmeans_fit(log2_clusters_scaling, self._scaling)
        self._scaling[...] = kmeans_predict(kmeans_scaling, self._scaling)
        kmeans_rotation = kmeans_fit(log2_clusters_rotation, self._rotation)
        self._rotation[...] = kmeans_predict(kmeans_rotation, self._rotation)
        kmeans_features_dc = kmeans_fit(log2_clusters_features_dc, self._features_dc[:, 0, :])
        self._features_dc[:, 0, :] = kmeans_predict(kmeans_features_dc, self._features_dc[:, 0, :])
        self._features_rest[...] = 0
        kmeans_opacity = kmeans_fit(log2_clusters_opacity, self._opacity)
        self._opacity[...] = kmeans_predict(kmeans_opacity, self._opacity)
        self.requires_grad_(True)
