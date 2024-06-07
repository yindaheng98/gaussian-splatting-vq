import torch
import tqdm
import math
from utils.loss_utils import l2_loss
from .gaussian_model import GaussianModel


def simple_knn(xyz: torch.Tensor, n=20, batch=256):
    indxs = torch.zeros(xyz.shape[0], n, dtype=torch.int32)
    dists = torch.zeros(xyz.shape[0], n, dtype=torch.float32)
    progress_bar = tqdm.tqdm(range(xyz.shape[0]), desc="Init Gaussians K nearest")
    for i in range(math.ceil(xyz.shape[0]/batch)):
        dist = torch.norm(xyz[i*batch:i*batch+batch, ...].unsqueeze(1) - xyz, dim=2, p=None)
        knn = dist.topk(n + 1, largest=False)
        dists[i*batch:i*batch+batch, ...] = knn.values[:, 1:]
        indxs[i*batch:i*batch+batch, ...] = knn.indices[:, 1:]
        progress_bar.update(min(i*batch+batch, xyz.shape[0])-i*batch)
    return indxs, dists


class GaussianModelIncremental(GaussianModel):
    def __init__(self, sh_degree: int):
        super().__init__(sh_degree=sh_degree)
        self.sh_degree = sh_degree

    def load_last_ply(self, path):
        last_gaussian = GaussianModel(sh_degree=self.sh_degree)
        last_gaussian.load_ply(path)
        self._xyz_last = last_gaussian._xyz.detach()
        self.neighbor_indices, dists = simple_knn(self._xyz_last)

    def load_ply(self, path):
        super().load_ply(path)
        self.load_last_ply(path)

    def incremental_reg(self):
        return l2_loss(self._xyz, self._xyz_last)
