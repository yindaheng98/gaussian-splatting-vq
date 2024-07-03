import open3d as o3d
import matplotlib.pyplot as plt
import cv2
import numpy as np

print("Read Redwood dataset")
redwood_rgbd = o3d.data.SampleRedwoodRGBDImages()
color_raw = cv2.imread("output/coffee_martini/frame1/train_interp/ours_30000/renders/00000.png")
depth_raw = np.load("output/coffee_martini/frame1/train_interp/ours_30000/depth/00000.npz")["depth"][0, ...]
# depth_raw = (depth_raw * 65536).astype(np.uint16)
# color_raw = o3d.io.read_image(redwood_rgbd.color_paths[0])
# depth_raw = o3d.io.read_image(redwood_rgbd.depth_paths[0])
rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(o3d.geometry.Image(color_raw), o3d.geometry.Image(depth_raw))
print(rgbd_image)

plt.subplot(1, 2, 1)
plt.title('Redwood grayscale image')
plt.imshow(rgbd_image.color)
plt.subplot(1, 2, 2)
plt.title('Redwood depth image')
plt.imshow(rgbd_image.depth)
plt.show()

pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
    rgbd_image,
    o3d.camera.PinholeCameraIntrinsic(
        o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault))
# Flip it, otherwise the pointcloud will be upside down
pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
o3d.visualization.draw_geometries([pcd])