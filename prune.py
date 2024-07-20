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
    scaling = gaussians.get_scaling.detach()
    print(scaling)
