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


if __name__ == "__main__":
    args = parser.parse_args()
    gaussians_src = GaussianModel(sh_degree=3)
    gaussians_src.load_ply(os.path.join(args.src, "point_cloud.ply"))
    gaussians_dst = GaussianModel(sh_degree=3)
    gaussians_dst.load_ply(os.path.join(args.dst, "point_cloud.ply"))
    print(gaussians_src, gaussians_dst)
    diff_xyz = diff(gaussians_src.get_xyz, gaussians_dst.get_xyz)
    dist_xyz = torch.norm(diff_xyz, p=2, dim=-1)
    dist_xyz_clamp = torch.clamp(dist_xyz, max=dist_xyz.topk(dist_xyz.shape[0]//100).values[-1])
    fig = plt.figure(figsize=(12, 2))
    ax = fig.subplots()
    ax.hist(dist_xyz_clamp.detach().cpu().numpy(), bins=100)
    plt.show()
