import os
import argparse
import shutil
from scene.gaussian_vq import VQGaussianModel

parser = argparse.ArgumentParser()
parser.add_argument("--src", type=str, required=True, help="The source dir.")
parser.add_argument("--dst", type=str, required=True, help="The destination dir.")
parser.add_argument("--iteration", type=int, required=True, help="The source iteration.")
parser.add_argument("--log2-clusters", type=int, required=True, help="Qualtize to how many clusters.")
parser.add_argument("--log2-clusters-scaling", type=int, default=0, help="Qualtize from which layer.")
parser.add_argument("--log2-clusters-rotation", type=int, default=0, help="Qualtize from which layer.")
parser.add_argument("--log2-clusters-features_dc", type=int, default=0, help="Qualtize from which layer.")
parser.add_argument("--log2-clusters-features_rest", type=int, default=0, help="Qualtize from which layer.")
parser.add_argument("--log2-clusters-opacity", type=int, default=0, help="Qualtize from which layer.")


if __name__ == "__main__":
    args = parser.parse_args()
    target_relpath = os.path.join("point_cloud", f"iteration_{args.iteration}", "point_cloud.ply")
    os.makedirs(os.path.dirname(os.path.join(args.dst, target_relpath)), exist_ok=True)
    shutil.copy2(os.path.join(args.src, "cameras.json"), os.path.join(args.dst, "cameras.json"))
    shutil.copy2(os.path.join(args.src, "cfg_args"), os.path.join(args.dst, "cfg_args"))
    gaussians = VQGaussianModel(sh_degree=3)
    gaussians.load_ply(os.path.join(args.src, target_relpath))
    gaussians.quantize_test_all(
        args.log2_clusters_scaling or args.log2_clusters,
        args.log2_clusters_rotation or args.log2_clusters,
        args.log2_clusters_features_dc or args.log2_clusters,
        args.log2_clusters_features_rest or args.log2_clusters,
        args.log2_clusters_opacity or args.log2_clusters)
    gaussians.save_ply(os.path.join(args.dst, target_relpath))
