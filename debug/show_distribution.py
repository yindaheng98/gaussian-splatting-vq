import argparse
import os
import torch
from scene import GaussianModel
import open3d as o3d

parser = argparse.ArgumentParser()
parser.add_argument("--src", type=str, required=True, help="The source ply.")


if __name__ == "__main__":
    args = parser.parse_args()
    gaussians = GaussianModel(sh_degree=3)
    gaussians.load_ply(os.path.join(args.src, "point_cloud.ply"))
    
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(gaussians._features_dc.detach().squeeze(1).cpu().numpy())
    pcd.colors = o3d.utility.Vector3dVector(gaussians._features_dc.detach().squeeze(1).cpu().numpy())
    o3d.visualization.draw_geometries([pcd])
