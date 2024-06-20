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
    pcd_data = o3d.geometry.PointCloud()
    pcd_data.points = o3d.utility.Vector3dVector(data)
    pcd_data.colors = o3d.utility.Vector3dVector(data)

    kmeans = getattr(gaussians, gaussians.kmeans_varname(args.attribute, args.index)).detach().cpu().numpy()
    pcd_kmeans = o3d.geometry.PointCloud()
    pcd_kmeans.points = o3d.utility.Vector3dVector(kmeans)
    pcd_kmeans.colors = o3d.utility.Vector3dVector(np.array([[0, 0, 0]]*kmeans.shape[0]))

    lkmeans = getattr(gaussians, gaussians.lkmeans_varname(args.attribute, args.index)).detach().cpu().numpy()
    tree = getattr(gaussians, gaussians.lkmeans_treename(args.attribute, args.index))
    leaf_n = kmeans.shape[0]
    lines = [(i+leaf_n, t[0]) for i, t in enumerate(tree)] + [(i+leaf_n, t[1]) for i, t in enumerate(tree)]
    line_set_lkmeans = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(lkmeans),
        lines=o3d.utility.Vector2iVector(lines),
    )

    geometries = [
        pcd_data,
        pcd_kmeans,
        line_set_lkmeans
    ]
    o3d.visualization.draw_geometries(geometries)
