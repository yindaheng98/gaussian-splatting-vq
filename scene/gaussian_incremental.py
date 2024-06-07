from utils.loss_utils import l2_loss
from .gaussian_model import GaussianModel


class GaussianModelIncremental(GaussianModel):
    def __init__(self, sh_degree: int):
        super().__init__(sh_degree=sh_degree)
        self.sh_degree = sh_degree

    def load_last_ply(self, path):
        last_gaussian = GaussianModel(sh_degree=self.sh_degree)
        last_gaussian.load_ply(path)
        self._xyz_last = last_gaussian._xyz.detach()

    def load_ply(self, path):
        super().load_ply(path)
        self.load_last_ply(path)

    def incremental_reg(self):
        return l2_loss(self._xyz, self._xyz_last)
