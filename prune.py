import cv2
import numpy as np
import json
import torch
import torch.nn.functional as F
import argparse
import os
from gaussian_renderer import GaussianModel


parser = argparse.ArgumentParser()
parser.add_argument("--read", type=str, required=True, help="Load from which ply.")
parser.add_argument("--sh-degree", type=int, default=3)
parser.add_argument("--target", type=int, default=200000, help="Target points.")
parser.add_argument("--save", type=str, required=True, help="Save to which ply.")

if __name__ == "__main__":
    args = parser.parse_args()
    gaussians = GaussianModel(args.sh_degree)
    gaussians.load_ply(args.read)
    scaling_sum = torch.sum(gaussians._scaling.detach(), dim=1)
    opacity = gaussians.get_opacity.detach().squeeze(-1)
    score = torch.log(opacity)+scaling_sum
    topk = torch.topk(score, args.target)
    top = topk.values[-1].item()
    visible = score > top
    print(visible.shape[0], "->", visible.sum())
    gaussians._xyz = gaussians._xyz[visible, ...]
    gaussians._features_dc = gaussians._features_dc[visible, ...]
    gaussians._features_rest = gaussians._features_rest[visible, ...]
    gaussians._scaling = gaussians._scaling[visible, ...]
    gaussians._rotation = gaussians._rotation[visible, ...]
    gaussians._opacity = gaussians._opacity[visible, ...]
    gaussians.max_radii2D = gaussians.max_radii2D[visible, ...]
    gaussians.save_ply(args.save)
