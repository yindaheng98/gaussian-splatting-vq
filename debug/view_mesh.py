import open3d as o3d
print("Load a ply mesh and render it")
pcd = o3d.io.read_triangle_mesh("data/coffee_martini/frame1/colorful-delaunay.ply")
o3d.visualization.draw_geometries([pcd])
