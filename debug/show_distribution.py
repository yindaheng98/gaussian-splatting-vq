import argparse
import os
import numpy as np
from scene.gaussian_lvq import LayeredKMeansGaussianModel, Attribute
import open3d as o3d

parser = argparse.ArgumentParser()
parser.add_argument("--src", type=str, required=True, help="The source ply.")
parser.add_argument("--save", type=str, required=True, help="Where to save the codebook.")
parser.add_argument("--iteration", type=int, required=True, help="The source iteration.")
parser.add_argument("--log2-clusters", type=int, required=True, help="Qualtize to how many clusters."),
parser.add_argument("--attribute", type=Attribute, choices=list(Attribute),
                    help="Which attribute do you want to quantize."),
parser.add_argument("--index", type=int, default=0),


if __name__ == "__main__":
    args = parser.parse_args()
    target_reldir = os.path.join("point_cloud", f"iteration_{args.iteration}")
    target_relpath = os.path.join(target_reldir, "point_cloud.ply")
    gaussians = LayeredKMeansGaussianModel(sh_degree=3)
    gaussians.load_ply(os.path.join(args.src, target_relpath))
    gaussians.dirpath = os.path.join(args.save, target_reldir)
    gaussians.load_codebook(os.path.join(args.save, target_reldir), args.log2_clusters, args.attribute, args.index)

    data = gaussians.get_data(args.attribute, args.index).detach().cpu().numpy()
    kmeans = getattr(gaussians, gaussians.kmeans_varname(args.attribute, args.index)).detach().cpu().numpy()

    pcd_src = o3d.geometry.PointCloud()
    pcd_src.points = o3d.utility.Vector3dVector(data)
    pcd_src.colors = o3d.utility.Vector3dVector(data)
    pcd_dst = o3d.geometry.PointCloud()
    pcd_dst.points = o3d.utility.Vector3dVector(kmeans)
    pcd_dst.colors = o3d.utility.Vector3dVector(np.array([[0, 0, 0]]*kmeans.shape[0]))
    o3d.visualization.draw_geometries([pcd_src, pcd_dst])
