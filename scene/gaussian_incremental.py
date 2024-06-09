import torch
import tqdm
import math
from .gaussian_model import GaussianModel


def simple_knn(xyz: torch.Tensor, n=8, batch=256):
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


def quaternion_to_matrix(quaternions: torch.Tensor) -> torch.Tensor:
    # From pytorch3d.transforms.rotation_conversions
    """
    Convert rotations given as quaternions to rotation matrices.

    Args:
        quaternions: quaternions with real part first,
            as tensor of shape (..., 4).

    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """
    r, i, j, k = torch.unbind(quaternions, -1)
    # pyre-fixme[58]: `/` is not supported for operand types `float` and `Tensor`.
    two_s = 2.0 / (quaternions * quaternions).sum(-1)

    o = torch.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        -1,
    )
    return o.reshape(quaternions.shape[:-1] + (3, 3))


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
        self.neighbor_offsets_last = _xyz_last[self.neighbor_indices] - _xyz_last.unsqueeze(-2)
        # pre-compute values
        self.rotation_matrix_last = quaternion_to_matrix(last_gaussian._rotation.detach())
        self.rotation_matrix_inv_last = self.rotation_matrix_last.transpose(2, 1)
        self.neighbor_offsets_point_coord_last = (
            self.rotation_matrix_inv_last.unsqueeze(1) @ self.neighbor_offsets_last.unsqueeze(-1)
        ).squeeze(-1)
        self.stretch_coef = torch.sign(self.neighbor_offsets_point_coord_last)
        self.neighbor_scalings = last_gaussian._scaling.detach()

    def load_ply(self, path):
        super().load_ply(path)
        self.load_last_ply(path)

    def incremental_reg(self):
        loss = {}
        rotation_matrix = quaternion_to_matrix(self._rotation)
        relative_rotation_matrix = rotation_matrix @ self.rotation_matrix_inv_last
        loss['rotation'] = weighted_l2_loss(
            relative_rotation_matrix.unsqueeze(-3),
            relative_rotation_matrix[self.neighbor_indices],
            self.neighbor_weights
        )

        neighbor_offsets = self._xyz[self.neighbor_indices] - self._xyz.unsqueeze(-2)
        neighbor_offsets_point_coord = (
            rotation_matrix.transpose(2, 1).unsqueeze(1) @
            neighbor_offsets.unsqueeze(-1)
        ).squeeze(-1)
        loss['rigidity'] = weighted_l2_loss(
            neighbor_offsets_point_coord,
            self.neighbor_offsets_point_coord_last,
            self.neighbor_weights)

        neighbor_relative_dists = torch.norm(neighbor_offsets, p=2, dim=-1)
        loss['isometry'] = weighted_l2_loss(
            neighbor_relative_dists.unsqueeze(-1),
            self.neighbor_relative_dists_last.unsqueeze(-1),
            self.neighbor_weights)

        relative_scaling = self._scaling / self.neighbor_scalings
        neighbor_relative_scaling = relative_scaling[self.neighbor_indices]
        loss['stretch'] = weighted_l2_loss(
            relative_scaling.unsqueeze(1),
            neighbor_relative_scaling,
            self.neighbor_weights)

        return loss
