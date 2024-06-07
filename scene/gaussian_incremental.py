import torch
from .gaussian_model import GaussianModel


def l2_loss(x, y):
    return torch.abs(x - y).mean()


class GaussianModelIncremental(GaussianModel):
    def __init__(self, sh_degree: int):
        super().__init__(sh_degree=sh_degree)
        self.last_gaussian = GaussianModel(sh_degree=sh_degree)

    def load_last_ply(self, path):
        self.last_gaussian.load_ply(path)
        self.last_gaussian._xyz = self._xyz.requires_grad_(False)

    def load_ply(self, path):
        super().load_ply(path)
        self.load_last_ply(path)

    def incremental_reg(self):
        return l2_loss(self._xyz, self.last_gaussian._xyz)
