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


def clamp_diff(a, b):
    d = diff(a, b)
    d_max = torch.abs(d).max(dim=-1).values
    d_clamp = clamp(d_max)
    return d_clamp


if __name__ == "__main__":
    args = parser.parse_args()
    gaussians_src = GaussianModel(sh_degree=3)
    gaussians_src.load_ply(os.path.join(args.src, "point_cloud.ply"))
    gaussians_dst = GaussianModel(sh_degree=3)
    gaussians_dst.load_ply(os.path.join(args.dst, "point_cloud.ply"))
    print(gaussians_src, gaussians_dst)

    fig = plt.figure(figsize=(12, 8))
    ax = fig.subplots(nrows=4)

    diff_xyz = clamp_diff(gaussians_src._xyz, gaussians_dst._xyz)
    ax[0].hist(diff_xyz.detach().cpu().numpy(), bins=100)
    ax[0].set_ylabel("xyz")

    diff_rotation = clamp_diff(gaussians_src._rotation, gaussians_dst._rotation)
    ax[1].hist(diff_rotation.detach().cpu().numpy(), bins=100)
    ax[1].set_ylabel("rotation")

    diff_scaling = clamp_diff(gaussians_src._scaling, gaussians_dst._scaling)
    ax[2].hist(diff_scaling.detach().cpu().numpy(), bins=100)
    ax[2].set_ylabel("scaling")

    diff_features_dc = clamp_diff(gaussians_src._features_dc.squeeze(1), gaussians_dst._features_dc.squeeze(1))
    ax[3].hist(diff_features_dc.detach().cpu().numpy(), bins=100)
    ax[3].set_ylabel("features_dc")

    plt.show()
