import torch
import tqdm
import math
from .gaussian_kmeans import KMeansGaussianModel, Attribute


def knn_init(kmeans, n=8, batch=4096):
    """Get k neighbor points for each point."""
    neighbors_idx = torch.zeros(kmeans.shape[0], n, dtype=torch.int32, device="cuda")
    progress_bar = tqdm.tqdm(range(kmeans.shape[0]), desc="Init KMeans center K nearest")
    dists = torch.zeros(kmeans.shape[0], n, dtype=torch.float32, device="cuda")
    for i in range(math.ceil(kmeans.shape[0]/batch)):
        dist = torch.norm(kmeans[i*batch:i*batch+batch, ...].unsqueeze(-2) - kmeans, p=2, dim=-1)
        knn = dist.topk(n + 1, largest=False)
        dists[i*batch:i*batch+batch, ...] = knn.values[:, 1:]
        neighbors_idx[i*batch:i*batch+batch, ...] = knn.indices[:, 1:]
        progress_bar.update(min(i*batch+batch, kmeans.shape[0])-i*batch)
    return neighbors_idx, dists


def marged_cluster(a: torch.Tensor, b: torch.Tensor):
    ab = torch.cat([a, b], dim=0)
    center = ab.mean(dim=0, keepdim=True)
    return ab, center


def vdistance(a: torch.Tensor, b: torch.Tensor):
    """Virtual distance for quant tree. Just the avg distance from center to points after merge."""
    _, center = marged_cluster(a, b)
    dist = torch.norm(torch.cat([a, b], dim=0)-center, dim=1, p=2).mean()
    return dist


class LayeredKMeans:
    def __init__(self, kmeans):
        self.kmeans = kmeans
        self.neighbors_idx, self.kdists = knn_init(self.kmeans)

    def clusters_init(self, data, quant):
        """Split the data into clusters"""
        clusters = [None]*self.kmeans.shape[0]
        for i in tqdm.tqdm(range(self.kmeans.shape[0]), desc="Init KMeans clusters"):
            clusters[i] = data[quant == i, ...]
        return clusters

    def distance_init(self, clusters):
        """Calculate the virtual distance between the point and their k neighbor points"""
        dists = torch.zeros(self.neighbors_idx.shape, dtype=torch.float32)
        for center_idx in tqdm.tqdm(range(self.kmeans.shape[0]), desc="Init KMeans center distance"):
            center_data = clusters[center_idx]
            for i, neighbor_idx in enumerate(self.neighbors_idx[center_idx, ...]):
                neighbor_data = clusters[neighbor_idx]
                dists[center_idx, i] = vdistance(center_data, neighbor_data)
        return dists

    def merge(self, clusters, dists, tmp_kmeans):
        """Merge the clusters and dists, and expand self.layerized_kmeans"""
        min_pos = (dists == dists.min()).nonzero()[0]
        min_idx, min_neighbors_idx = min_pos[0], self.neighbors_idx[*min_pos]
        # concat to get new cluster and center
        new_cluster, new_center = marged_cluster(clusters[min_idx], clusters[min_neighbors_idx])
        # add new cluster
        new_center_idx = len(clusters)
        clusters.append(None)
        clusters[new_center_idx] = new_cluster
        # delete old clusters
        clusters[min_idx] = clusters[min_neighbors_idx] = None
        # add new center
        tmp_kmeans = torch.cat([tmp_kmeans, new_center], dim=0)
        # TODO: delete old center
        print(min_idx, min_neighbors_idx)
        return tmp_kmeans

    def fit(self, data, quant):
        clusters = self.clusters_init(data, quant)
        dists = self.distance_init(clusters)
        self.layerized_kmeans = self.kmeans.clone()
        tmp_kmeans = self.kmeans.clone()
        tmp_kmeans = self.merge(clusters, dists, tmp_kmeans)
        raise NotImplementedError("fit not implemented")


class LayeredKMeansGaussianModel(KMeansGaussianModel):
    dirpath = ''

    def build_codebook(self, log2_clusters: int, attr: Attribute, i=0):
        super().load_codebook(self.dirpath, log2_clusters, attr, i)

        kmeans = getattr(self, super().get_name(attr, i))
        lkmeans = LayeredKMeans(kmeans)
        data = self.get_data(attr, i).detach()
        quant = super().quantize(attr, i)
        lkmeans.fit(data, quant)
        # TODO: save LayeredKMeans
        raise NotImplementedError("build_codebook not implemented")
