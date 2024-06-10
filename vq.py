import os
import argparse
from scene.gaussian_vq import VQGaussianModel

parser = argparse.ArgumentParser()
parser.add_argument("--src", type=str, required=True, help="The source bson.")
parser.add_argument("--dst", type=str, required=True, help="The destination bson.")
parser.add_argument("--log2-clusters", type=int, required=True, help="Qualtize to how many clusters.")
parser.add_argument("--log2-clusters-scaling", type=int, default=0, help="Qualtize from which layer.")
parser.add_argument("--log2-clusters-rotation", type=int, default=0, help="Qualtize from which layer.")
parser.add_argument("--log2-clusters-features_dc", type=int, default=0, help="Qualtize from which layer.")
parser.add_argument("--log2-clusters-features_rest", type=int, default=0, help="Qualtize from which layer.")
parser.add_argument("--log2-clusters-opacity", type=int, default=0, help="Qualtize from which layer.")


if __name__ == "__main__":
    args = parser.parse_args()
    gaussians = VQGaussianModel(sh_degree=3)
    gaussians.load_ply(os.path.join(args.src, "point_cloud.ply"))
    gaussians.VectorQuant(
        args.log2_clusters_scaling or args.log2_clusters,
        args.log2_clusters_rotation or args.log2_clusters,
        args.log2_clusters_features_dc or args.log2_clusters,
        args.log2_clusters_features_rest or args.log2_clusters,
        args.log2_clusters_opacity or args.log2_clusters)
    os.makedirs(args.dst, exist_ok=True)
    gaussians.save_ply(os.path.join(args.dst, "point_cloud.ply"))
