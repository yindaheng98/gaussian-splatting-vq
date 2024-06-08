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


def quaternion_mult(q1, q2):
    w1, x1, y1, z1 = q1.T
    w2, x2, y2, z2 = q2.T
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    return torch.stack([w, x, y, z]).T


def quaternion2rotation(q):
    norm = torch.sqrt(q[:, 0] * q[:, 0] + q[:, 1] * q[:, 1] + q[:, 2] * q[:, 2] + q[:, 3] * q[:, 3])
    q = q / norm[:, None]
    rot = torch.zeros((q.size(0), 3, 3), device='cuda')
    r = q[:, 0]
    x = q[:, 1]
    y = q[:, 2]
    z = q[:, 3]
    rot[:, 0, 0] = 1 - 2 * (y * y + z * z)
    rot[:, 0, 1] = 2 * (x * y - r * z)
    rot[:, 0, 2] = 2 * (x * z + r * y)
    rot[:, 1, 0] = 2 * (x * y + r * z)
    rot[:, 1, 1] = 1 - 2 * (x * x + z * z)
    rot[:, 1, 2] = 2 * (y * z - r * x)
    rot[:, 2, 0] = 2 * (x * z - r * y)
    rot[:, 2, 1] = 2 * (y * z + r * x)
    rot[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return rot


def weighted_l2_loss(x, y, w):
    return torch.sqrt((torch.flatten(x - y, start_dim=2).sum(-1) ** 2) * w + 1e-20).mean()


class GaussianModelIncremental(GaussianModel):
    def __init__(self, sh_degree: int):
        super().__init__(sh_degree=sh_degree)
        self.sh_degree = sh_degree

    def load_last_ply(self, path):
        last_gaussian = GaussianModel(sh_degree=self.sh_degree)
        last_gaussian.load_ply(path)

        _xyz_last = last_gaussian._xyz.detach()

        # pre-compute values
        self.neighbor_indices, dists = simple_knn(_xyz_last)
        self.neighbor_weights = torch.exp(-dists)
        self.neighbor_relative_dists_last = dists
        self.neighbor_relative_offsets_last = _xyz_last[self.neighbor_indices] - _xyz_last.unsqueeze(-2)
        # pre-compute values
        self.rotation_inv_last = last_gaussian._rotation.detach()
        self.rotation_inv_last[:, 1:] = -1 * self.rotation_inv_last[:, 1:]

    def load_ply(self, path):
        super().load_ply(path)
        self.load_last_ply(path)

    def incremental_reg(self):
        loss = {}
        rel_rotation = quaternion2rotation(quaternion_mult(self._rotation, self.rotation_inv_last))
        loss['rotation'] = weighted_l2_loss(
            rel_rotation.unsqueeze(-3),
            rel_rotation[self.neighbor_indices],
            self.neighbor_weights
        )

        neighbor_relative_offsets = self._xyz[self.neighbor_indices] - self._xyz.unsqueeze(-2)
        loss['rigidity'] = weighted_l2_loss(
            (rel_rotation.transpose(2, 1).unsqueeze(1) @ neighbor_relative_offsets.unsqueeze(-1)).squeeze(-1),
            self.neighbor_relative_offsets_last,
            self.neighbor_weights)

        neighbor_relative_dists = torch.norm(neighbor_relative_offsets, p=2, dim=-1)
        loss['isometry'] = weighted_l2_loss(
            neighbor_relative_dists.unsqueeze(-1),
            self.neighbor_relative_dists_last.unsqueeze(-1),
            self.neighbor_weights)

        return loss
