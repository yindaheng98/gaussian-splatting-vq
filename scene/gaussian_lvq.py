import os
import numpy as np
import json
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


def merged_cluster(a: torch.Tensor, b: torch.Tensor):
    ab = torch.cat([a, b], dim=0)
    center = ab.mean(dim=0, keepdim=True)
    return ab, center


def vdistance(a: torch.Tensor, b: torch.Tensor):
    """Virtual distance for quant tree. Just the avg distance from center to points after merge."""
    _, center = merged_cluster(a, b)
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

    def merge(self, clusters, dists, tmp_kmeans, tmp_neighbors_idx):
        """Merge the clusters and dists, and expand self.layerized_kmeans"""
        min_pos = (dists == dists.min()).nonzero()[0]
        min_idx, min_neighbors_idx = min_pos[0].item(), tmp_neighbors_idx[*min_pos].item()
        # concat to get new cluster and center
        new_cluster, new_center = merged_cluster(clusters[min_idx], clusters[min_neighbors_idx])
        # add new cluster
        new_center_idx = len(clusters)
        clusters.append(None)
        clusters[new_center_idx] = new_cluster
        # delete old clusters
        clusters[min_idx] = clusters[min_neighbors_idx] = None
        # add new center
        tmp_kmeans = torch.cat([tmp_kmeans, new_center], dim=0)
        self.layerized_kmeans = torch.cat([self.layerized_kmeans, new_center], dim=0)  # save result
        self.tree.append((min_idx, min_neighbors_idx))  # save result
        # delete old center
        tmp_kmeans[min_idx, ...] = torch.inf
        tmp_kmeans[min_neighbors_idx, ...] = torch.inf
        # which center should be updated
        should_update_idx = torch.logical_or(
            tmp_neighbors_idx == min_idx,
            tmp_neighbors_idx == min_neighbors_idx
        ).any(dim=1).nonzero()[..., 0]
        should_update_idx = should_update_idx[torch.logical_and(
            should_update_idx != min_idx, should_update_idx != min_neighbors_idx
        )]
        should_update_idx = torch.cat([
            should_update_idx,
            torch.tensor([new_center_idx], dtype=should_update_idx.dtype, device=should_update_idx.device)
        ], dim=0)
        # compute new neighbors idx
        new_kdist = torch.norm(tmp_kmeans[should_update_idx, ...].unsqueeze(-2) - tmp_kmeans, p=2, dim=-1)
        new_knn = new_kdist.topk(dists.shape[1] + 1, largest=False)
        # update neighbors idx
        tmp_neighbors_idx = torch.cat([tmp_neighbors_idx, torch.zeros_like(tmp_neighbors_idx[0:1, ...])], dim=0)
        tmp_neighbors_idx[should_update_idx, ...] = new_knn.indices[:, 1:].type(torch.int32)
        # update neighbors distance
        dists = torch.cat([dists, torch.zeros_like(dists[0:1, ...])], dim=0)
        for center_idx in tqdm.tqdm(should_update_idx, desc="Updating distance", position=1, leave=False):
            center_data = clusters[center_idx]
            for i, neighbor_idx in enumerate(tmp_neighbors_idx[center_idx, ...]):
                neighbor_data = clusters[neighbor_idx]
                if center_data is not None and neighbor_data is not None:
                    dists[center_idx, i] = vdistance(center_data, neighbor_data)
                else:
                    dists[center_idx, i] = torch.inf
        dists[min_idx, ...] = torch.inf
        dists[min_neighbors_idx, ...] = torch.inf
        return clusters, dists, tmp_kmeans, tmp_neighbors_idx

    def fit(self, data, quant, final_clusters):
        clusters = self.clusters_init(data, quant)
        dists = self.distance_init(clusters).to(data.device)
        self.layerized_kmeans = self.kmeans.clone()
        self.tree = []
        tmp_kmeans = self.kmeans.clone()
        tmp_neighbors_idx = self.neighbors_idx.clone()
        for _ in tqdm.tqdm(range(len(clusters)-final_clusters), desc="Merging clusters", position=0, leave=False):
            clusters, dists, tmp_kmeans, tmp_neighbors_idx = self.merge(clusters, dists, tmp_kmeans, tmp_neighbors_idx)
        pass


class LayeredKMeansGaussianModel(KMeansGaussianModel):
    dirpath = ''
    final_clusters = 2**6

    def lkmeans_filename(self, log2_clusters: int, attr: Attribute, i=0):
        return self._get_filename("lkmeans", log2_clusters, attr, i)

    def lkmeans_varname(self, attr: Attribute, i=0):
        return self._get_name("lkmeans", attr, i)

    def lkmeans_treename(self, attr: Attribute, i=0):
        return self._get_name("lkmeans_tree", attr, i)

    def build_codebook(self, log2_clusters: int, attr: Attribute, i=0):
        super().load_codebook(self.dirpath, log2_clusters, attr, i)
        kmeans = getattr(self, self.kmeans_varname(attr, i))
        lkmeans = LayeredKMeans(kmeans)
        data = self.get_data(attr, i).detach()
        quant = super().quantize(attr, i)
        lkmeans.fit(data, quant, self.final_clusters)
        setattr(self, self.lkmeans_varname(attr, i), lkmeans.layerized_kmeans)
        setattr(self, self.lkmeans_treename(attr, i), lkmeans.tree)
        setattr(self, f"log2_clusters_{self.kmeans_varname(attr, i)}", log2_clusters)

    def save_codebook(self, dirpath, attr: Attribute, i=0):
        super().save_codebook(dirpath, attr, i)
        os.makedirs(dirpath, exist_ok=True)
        log2_clusters = getattr(self, f"log2_clusters_{self.kmeans_varname(attr, i)}")
        path = os.path.join(dirpath, self.lkmeans_filename(log2_clusters, attr, i) + ".npz")
        lkmeans = getattr(self, self.lkmeans_varname(attr, i))
        print(f"save layerized codebook {path}.")
        np.savez(path, codebook=lkmeans.cpu().numpy())
        tree_path = path = os.path.join(dirpath, self.lkmeans_filename(log2_clusters, attr, i) + ".json")
        tree = getattr(self, self.lkmeans_treename(attr, i))
        with open(tree_path, "w", encoding='utf8') as f:
            json.dump(tree, f, indent=2)
