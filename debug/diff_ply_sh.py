import argparse
import os
import torch
from scene import GaussianModel
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument("--src", type=str, required=True, help="The source ply.")
parser.add_argument("--dst", type=str, required=True, help="The destination ply.")
parser.add_argument("--show", action="store_true")


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
    ax = fig.subplots(nrows=5, ncols=3)
    for i in range(5):
        for j in range(3):
            k = i*3+j
            diff_sh = clamp_diff(gaussians_src._features_rest[:, k, ...], gaussians_dst._features_rest[:, k, ...])
            ax[i][j].hist(diff_sh.detach().cpu().numpy(), bins=100)
            ax[i][j].set_ylabel(f"SH{k+1}")
    fig.tight_layout()
    if args.show:
        plt.show()
    else:
        fig.savefig("diff_ply_sh.png")
