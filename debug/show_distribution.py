import argparse
import os
import numpy as np
from scene import GaussianModel
import open3d as o3d

parser = argparse.ArgumentParser()
parser.add_argument("--src", type=str, required=True, help="The source ply.")
parser.add_argument("--dst", type=str, required=True, help="The destination ply.")


if __name__ == "__main__":
    args = parser.parse_args()
    gaussians_src = GaussianModel(sh_degree=3)
    gaussians_src.load_ply(os.path.join(args.src, "point_cloud.ply"))
    gaussians_dst = GaussianModel(sh_degree=3)
    gaussians_dst.load_ply(os.path.join(args.dst, "point_cloud.ply"))
    data_src = gaussians_src.get_features[:, 0, ...].detach().squeeze(1).cpu().numpy()
    data_dst = gaussians_dst.get_features[:, 0, ...].detach().squeeze(1).cpu().numpy()

    pcd_src = o3d.geometry.PointCloud()
    pcd_src.points = o3d.utility.Vector3dVector(data_src)
    pcd_src.colors = o3d.utility.Vector3dVector(data_src)
    pcd_dst = o3d.geometry.PointCloud()
    pcd_dst.points = o3d.utility.Vector3dVector(data_dst)
    pcd_dst.colors = o3d.utility.Vector3dVector(np.array([[1, 1, 1]]*data_dst.shape[0]))
    o3d.visualization.draw_geometries([pcd_src, pcd_dst])
