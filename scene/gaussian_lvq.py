import torch
from tqdm import tqdm
from typing import List
from .gaussian_kmeans import KMeansGaussianModel, Attribute


def distance(a: torch.Tensor, b: torch.Tensor):
    ab = torch.cat([a, b], dim=0)
    center = ab.mean(dim=0, keepdim=True)
    dist = torch.norm(ab-center, dim=1, p=2).mean()
    return dist


def fill_distances(dists: torch.Tensor, datalist: List[torch.Tensor]):
    n = len(datalist)
    pbar = tqdm(desc="Calculate distances", total=dists.shape[0])
    for i in range(n):
        for j in range(i+1, n):
            dists[(n-1+n-i)*i//2+j] = distance(datalist[i], datalist[j])
            pbar.update(1)


class LayeredKMeansGaussianModel(KMeansGaussianModel):
    dirpath = ''

    def build_codebook(self, log2_clusters: int, attr: Attribute, i=0):
        super().load_codebook(self.dirpath, log2_clusters, attr, i)
        kmeans = getattr(self, super().get_name(attr, i))
        data = self.get_data(attr, i).detach()
        quant = super().quantize(attr, i)
        dists = torch.zeros((kmeans.shape[0]-1)*kmeans.shape[0]//2, dtype=torch.float32, device='cpu')
        fill_distances(dists, [data[quant == i, ...] for i in range(kmeans.shape[0])])
        raise NotImplementedError("build_codebook not implemented")
