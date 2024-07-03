import open3d as o3d
import matplotlib.pyplot as plt
import cv2
import numpy as np
import json
import torch

print("Read Redwood dataset")
redwood_rgbd = o3d.data.SampleRedwoodRGBDImages()
color_raw = cv2.imread("output/coffee_martini/frame1/train_interp/ours_30000/renders/00000.png")
depth_raw = np.load("output/coffee_martini/frame1/train_interp/ours_30000/depth/00000.npz")["depth"][0, ...]
# depth_raw = (depth_raw * 65536).astype(np.uint16)
# color_raw = o3d.io.read_image(redwood_rgbd.color_paths[0])
# depth_raw = o3d.io.read_image(redwood_rgbd.depth_paths[0])
rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
    o3d.geometry.Image(color_raw), o3d.geometry.Image(depth_raw))
print(rgbd_image)

plt.subplot(1, 2, 1)
plt.title('Redwood grayscale image')
plt.imshow(rgbd_image.color)
plt.subplot(1, 2, 2)
plt.title('Redwood depth image')
plt.imshow(rgbd_image.depth)
plt.show()


with open("output/coffee_martini/frame1/train_interp/ours_30000/depth/00000.camera.json", "r") as f:
    camera = json.load(f)
height, width = camera["height"], camera["width"]
pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
    rgbd_image,
    o3d.camera.PinholeCameraIntrinsic(
        width=width, height=height,
        fx=camera["fx"], fy=camera["fy"],
        cx=camera["width"]/2, cy=camera["height"]/2
    ))
o3d.visualization.draw_geometries([pcd])

R = torch.tensor(camera["rotation"])
t = torch.tensor(camera["position"])
K = torch.tensor([
    [camera["fx"], 0, camera["width"]/2],
    [0, camera["fy"], camera["height"]/2],
    [0, 0, 1]
])
uv = torch.ones(color_raw.shape, dtype=torch.float32)
uv[..., 0] = torch.arange(0, width, dtype=torch.float32).unsqueeze(0).expand(height, -1)
uv[..., 1] = torch.arange(0, height, dtype=torch.float32).unsqueeze(1).expand(-1, width)
depth = torch.from_numpy(depth_raw)
xyz_camera = torch.inverse(K) @ uv.reshape(-1, 3).T * depth.reshape(-1)
# xyz_camera = torch.from_numpy(np.asarray(pcd.points, dtype=np.float32)).T*1000
xyz_world = torch.inverse(R) @ (xyz_camera - t.unsqueeze(1))
xyz = xyz_world.T.cpu().numpy()

colors = color_raw.reshape(-1, 3).astype(np.float32)/255
idx = np.abs(xyz).sum(axis=-1) < 1000
xyz = xyz[idx, ...]
colors = colors[idx, ...]
pcd.points = o3d.utility.Vector3dVector(xyz)
pcd.colors = o3d.utility.Vector3dVector(colors[:xyz.shape[0], ...])
o3d.visualization.draw_geometries([pcd])
