import open3d as o3d
print("Load a ply point cloud, print it, and render it")
pcd = o3d.io.read_point_cloud("data/coffee_martini/frame1/sparse/0/points3D.ply")
o3d.visualization.draw_geometries([pcd])
