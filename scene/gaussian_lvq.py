import torch
import tqdm
import math
from .gaussian_kmeans import KMeansGaussianModel, Attribute


def knn(kmeans, n=8, batch=4096):
    """Get k neighbor points for each point."""
    neighbors_idx = torch.zeros(kmeans.shape[0], n, dtype=torch.int32, device="cuda")
    dists = torch.zeros(kmeans.shape[0], n, dtype=torch.float32, device="cuda")
    progress_bar = tqdm.tqdm(range(kmeans.shape[0]), desc="Init KMeans center K nearest")
    for i in range(math.ceil(kmeans.shape[0]/batch)):
        dist = torch.norm(kmeans[i*batch:i*batch+batch, ...].unsqueeze(-2) - kmeans, p=2, dim=-1)
        knn = dist.topk(n + 1, largest=False)
        dists[i*batch:i*batch+batch, ...] = knn.values[:, 1:]
        neighbors_idx[i*batch:i*batch+batch, ...] = knn.indices[:, 1:]
        progress_bar.update(min(i*batch+batch, kmeans.shape[0])-i*batch)
    return neighbors_idx, dists


def vdistance(a: torch.Tensor, b: torch.Tensor):
    """Virtual distance for quant tree. Just the avg distance from center to points after merge."""
    ab = torch.cat([a, b], dim=0)
    center = ab.mean(dim=0, keepdim=True)
    dist = torch.norm(ab-center, dim=1, p=2).mean()
    return dist


def distance_init(kmeans, neighbors_idx, data, quant):
    """Calculate the virtual distance between the point and their k neighbor points"""
    dists = torch.zeros(neighbors_idx.shape, dtype=torch.float32, device=data.device)
    for center_idx in tqdm.tqdm(range(kmeans.shape[0]), desc="Init KMeans center distance"):
        center_data = data[quant == center_idx, ...]
        for i, neighbor_idx in enumerate(neighbors_idx[center_idx, ...]):
            neighbor_data = data[quant == neighbor_idx]
            dists[center_idx, i] = vdistance(center_data, neighbor_data)
    return dists


class LayeredKMeansGaussianModel(KMeansGaussianModel):
    dirpath = ''

    def build_codebook(self, log2_clusters: int, attr: Attribute, i=0):
        super().load_codebook(self.dirpath, log2_clusters, attr, i)
        kmeans = getattr(self, super().get_name(attr, i))
        data = self.get_data(attr, i).detach()
        quant = super().quantize(attr, i)
        indxs, kdists = knn(kmeans)
        dists = distance_init(kmeans, indxs, data, quant)
        raise NotImplementedError("build_codebook not implemented")
