import argparse
import os
from scene import GaussianModel

parser = argparse.ArgumentParser()
parser.add_argument("--src", type=str, required=True, help="The source bson.")
parser.add_argument("--dst", type=str, required=True, help="The destination bson.")

if __name__ == "__main__":
    args = parser.parse_args()
    gaussians = GaussianModel(sh_degree=3)
    gaussians.load_ply(os.path.join(args.src, "point_cloud.ply"))
    os.makedirs(args.dst, exist_ok=True)
    gaussians.save_ply(os.path.join(args.dst, "point_cloud.ply"))
