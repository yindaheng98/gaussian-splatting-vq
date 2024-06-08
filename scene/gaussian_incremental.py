import torch
import tqdm
import math
from .gaussian_model import GaussianModel


def simple_knn(xyz: torch.Tensor, n=20, batch=256):
    indxs = torch.zeros(xyz.shape[0], n, dtype=torch.int32, device="cuda")
    dists = torch.zeros(xyz.shape[0], n, dtype=torch.float32, device="cuda")
    progress_bar = tqdm.tqdm(range(xyz.shape[0]), desc="Init Gaussians K nearest")
    for i in range(math.ceil(xyz.shape[0]/batch)):
        dist = torch.norm(xyz[i*batch:i*batch+batch, ...].unsqueeze(-2) - xyz, p=2, dim=-1)
        knn = dist.topk(n + 1, largest=False)
        dists[i*batch:i*batch+batch, ...] = knn.values[:, 1:]
        indxs[i*batch:i*batch+batch, ...] = knn.indices[:, 1:]
        progress_bar.update(min(i*batch+batch, xyz.shape[0])-i*batch)
    return indxs, dists


def weighted_l2_loss(x, y, w):
    return torch.sqrt(((x - y) ** 2) * w + 1e-20).mean()


class GaussianModelIncremental(GaussianModel):
    def __init__(self, sh_degree: int):
        super().__init__(sh_degree=sh_degree)
        self.sh_degree = sh_degree

    def load_last_ply(self, path):
        last_gaussian = GaussianModel(sh_degree=self.sh_degree)
        last_gaussian.load_ply(path)
        self._xyz_last = last_gaussian._xyz.detach()
        self.neighbor_indices, dists = simple_knn(self._xyz_last)
        self.neighbor_weights = torch.exp(-dists)

        # pre-compute values
        self.neighbor_relative_dists = dists

    def load_ply(self, path):
        super().load_ply(path)
        self.load_last_ply(path)

    def incremental_reg(self):
        neighbor_relative_dists = torch.norm(
            self._xyz.unsqueeze(-2) - self._xyz_last[self.neighbor_indices],
            p=2, dim=-1)
        loss_relative_dists = weighted_l2_loss(
            neighbor_relative_dists, self.neighbor_relative_dists, self.neighbor_weights)
        return loss_relative_dists
