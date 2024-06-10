import os
import argparse
from scene.gaussian_vq import VQGaussianModel, Attribute

parser = argparse.ArgumentParser()
parser.add_argument("--src", type=str, required=True, help="The source dir.")
parser.add_argument("--dst", type=str, required=True, help="The destination dir.")
parser.add_argument("--iteration", type=int, required=True, help="The source iteration.")
parser.add_argument("--log2-clusters", type=int, required=True, help="Qualtize to how many clusters.")
parser.add_argument("--attribute", type=Attribute, choices=list(Attribute),
                    help="Which attribute do you want to quantize.")
parser.add_argument("--index", type=int, default=0)


if __name__ == "__main__":
    args = parser.parse_args()
    target_reldir = os.path.join("point_cloud", f"iteration_{args.iteration}")
    target_relpath = os.path.join(target_reldir, "point_cloud.ply")
    gaussians = VQGaussianModel(sh_degree=3)
    gaussians.load_ply(os.path.join(args.src, target_relpath))
    gaussians.kmeans(args.attribute, args.log2_clusters, args.index)
    gaussians.save_kmeans(os.path.join(args.dst, target_reldir), args.attribute, args.index)
