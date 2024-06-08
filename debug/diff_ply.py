import argparse
import os
import torch
from scene import GaussianModel
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument("--src", type=str, required=True, help="The source ply.")
parser.add_argument("--dst", type=str, required=True, help="The destination ply.")


def diff(a, b):
    s = min(a.shape[0], b.shape[0])
    return a[:s, ...].detach() - b[:s, ...].detach()


def clamp(p):
    return torch.clamp(p, max=p.topk(p.shape[0]//100).values[-1])


if __name__ == "__main__":
    args = parser.parse_args()
    gaussians_src = GaussianModel(sh_degree=3)
    gaussians_src.load_ply(os.path.join(args.src, "point_cloud.ply"))
    gaussians_dst = GaussianModel(sh_degree=3)
    gaussians_dst.load_ply(os.path.join(args.dst, "point_cloud.ply"))
    print(gaussians_src, gaussians_dst)

    fig = plt.figure(figsize=(12, 6))
    ax = fig.subplots(nrows=3)

    diff_xyz = diff(gaussians_src.get_xyz, gaussians_dst.get_xyz)
    dist_xyz = torch.norm(diff_xyz, p=2, dim=-1)
    dist_xyz_clamp = clamp(dist_xyz)
    ax[0].hist(dist_xyz_clamp.detach().cpu().numpy(), bins=100)

    diff_rotation = diff(gaussians_src.get_rotation, gaussians_dst.get_rotation)
    diff_rotation_max = torch.abs(diff_rotation).max(dim=-1).values
    diff_rotation_clamp = clamp(diff_rotation_max)
    ax[1].hist(diff_rotation_clamp.detach().cpu().numpy(), bins=100)

    diff_scaling = diff(gaussians_src.get_scaling, gaussians_dst.get_scaling)
    diff_scaling_max = torch.abs(diff_scaling).max(dim=-1).values
    diff_scaling_clamp = clamp(diff_scaling_max)
    ax[2].hist(diff_scaling_clamp.detach().cpu().numpy(), bins=100)

    plt.show()
