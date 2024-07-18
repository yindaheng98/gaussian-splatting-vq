import os
import argparse
import torch
from typing import NamedTuple
from scene.camera_dataset import CameraPoseDataset, Pose
from gaussian_renderer import render, GaussianModel
from arguments import PipelineParams
from scene.cameras import Camera as View
from utils.system_utils import searchForMaxIteration
from utils.graphics_utils import fov2focal
from warping import reconstrucion, projection, warp

parser = argparse.ArgumentParser()
parser.add_argument("--cameras", type=str, required=True, help="Path to the camera pose.")
parser.add_argument("--fovx", type=float, default=fov2focal(876.90487504022, 1600), help="Camera fov x axis.")
parser.add_argument("--fovy", type=float, default=fov2focal(875.13605328243, 1200), help="Camera fov y axis.")
parser.add_argument("--width", type=float, default=1600, help="Camera width.")
parser.add_argument("--height", type=float, default=1200, help="Camera height.")
parser.add_argument("--buffersize", type=int, default=10, help="Camera buffer size.")
parser.add_argument("--fps", type=int, default=30, help="Playback fps.")
parser.add_argument("--history-size", type=int, default=15)
parser.add_argument("--prediction-stride", type=int, default=5)
parser.add_argument("--video", type=str, required=True)
parser.add_argument("--sh-degree", type=int, default=3)
parser.add_argument("--max-frame", type=int, default=30)


class Camera(NamedTuple):
    pose: Pose
    fovx: float
    fovy: float
    width: int
    height: int


def predict_camera(pose_history, fovx: float, fovy: float, width: int, height: int):
    return Camera(
        pose=Pose(
            timestamp=pose_history["history_timestamp"][-1],
            R=pose_history["R"][-1, ...],
            T=pose_history["T"][-1, ...]),
        fovx=fovx, fovy=fovy, width=width, height=height)


def render_frame(camera: Camera, gaussians: GaussianModel, pipeline: PipelineParams):
    view = View(colmap_id="", R=camera.pose.R.cpu().numpy(), T=camera.pose.T.cpu().numpy(),
                FoVx=camera.fovx, FoVy=camera.fovy,
                image=None, gt_alpha_mask=None,
                image_name="", uid="",
                data_device="cuda",
                image_width=camera.width, image_height=camera.height)
    with torch.no_grad():
        background = torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda")
        render_pkg = render(view, gaussians, pipeline, background)
        rendering, depth = render_pkg["render"], render_pkg["depth"]
        return rendering, depth


def warping_frame(camera: Camera, depth, camera_ref: Camera, color_ref):
    K = torch.tensor([
        [camera.fovx, 0, camera.width/2],
        [0, camera.fovy, camera.height/2],
        [0, 0, 1]
    ])
    R_c2w, T_c2w = camera.pose.R, camera.pose.T
    xyz = reconstrucion(K, R_c2w, T_c2w, depth)
    K_r = torch.tensor([
        [camera.fovx, 0, camera.width/2],
        [0, camera.fovy, camera.height/2],
        [0, 0, 1]
    ])
    K_r = torch.tensor([
        [camera_ref.fovx, 0, camera_ref.width/2],
        [0, camera_ref.fovy, camera_ref.height/2],
        [0, 0, 1]
    ])
    R_r, t_r = camera_ref.pose.R, camera_ref.pose.T
    uv, z = projection(K_r, R_r, t_r, xyz)
    warped = warp(uv, color_ref, z)  # wrap it
    return warped


def show3images(distorted_image, reference_image, warpedref_image):
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(12, 4))
    axs = fig.subplots(ncols=3, nrows=1)
    axs[0].set_title('distorted image')
    axs[0].imshow(distorted_image.clamp(0, 1).permute(1, 2, 0).cpu().numpy())
    axs[1].set_title('reference image')
    axs[1].imshow(reference_image.clamp(0, 1).permute(1, 2, 0).cpu().numpy())
    axs[2].set_title('warped image')
    axs[2].imshow(warpedref_image.clamp(0, 1).permute(1, 2, 0).cpu().numpy())
    fig.tight_layout(pad=5)
    plt.show()


videoout = None


def save2video(distorted_image, warpedref_image):
    import cv2
    frame = torch.concat((distorted_image, warpedref_image), dim=1).permute(1, 2, 0)
    frame_uint8 = (frame[..., [2, 1, 0]].clamp(0, 1) * 255).type(torch.uint8).cpu().numpy()
    height, width, _ = frame_uint8.shape
    global videoout
    if videoout is None:
        videoout = cv2.VideoWriter('output/run.mp4', -1, 60, (width, height))
        videoout.write((frame.clamp(0, 1) * 255).type(torch.uint8).cpu().numpy())


def save2images(distorted_image, warpedref_image, n_frame, n_render, folder="output/run"):
    os.makedirs(folder, exist_ok=True)
    import cv2
    frame = torch.concat((distorted_image, warpedref_image), dim=1).permute(1, 2, 0)
    frame_uint8 = (frame[..., [2, 1, 0]].clamp(0, 1) * 255).type(torch.uint8).cpu().numpy()
    cv2.imwrite(os.path.join(folder, f"frame{n_frame}_{n_render}.png"), frame_uint8)


if __name__ == "__main__":
    torch.device("cuda").__enter__()
    pipeline = PipelineParams(parser)
    args = parser.parse_args()
    pose_dataset = CameraPoseDataset(args.cameras, history_size=args.history_size, prediction_stride=args.prediction_stride)
    frame_stride = 1/args.fps
    last_timestamp = None
    last_frame = None
    n_frame = 0
    n_render = 0
    server_gaussians = GaussianModel(args.sh_degree)
    for pose_history, pose_groundtruth in pose_dataset:
        timestamp = pose_history["timestamp"]
        if last_timestamp is None or timestamp - last_timestamp > frame_stride:
            n_frame = n_frame + 1
            last_timestamp = timestamp
            n_render = 0
            if n_frame > args.max_frame:
                break
            print(f"{timestamp:.4f}", "frame", n_frame, "loading")
            prediction_camera = predict_camera(pose_history, fovx=args.fovx, fovy=args.fovy, width=args.width, height=args.height)
            frame_folder = os.path.join(args.video, f"frame{n_frame}", "point_cloud")
            n_iter = searchForMaxIteration(frame_folder)
            frame_ply = os.path.join(frame_folder, f"iteration_{n_iter}", "point_cloud.ply")
            server_gaussians.load_ply(path=frame_ply)
            reference_image, _ = render_frame(prediction_camera, server_gaussians, pipeline)
        n_render += 1
        print(f"{timestamp:.4f}", "frame", n_frame, "rendering", n_render)
        groundtruth_camera = Camera(
            pose=Pose(
                timestamp=timestamp,
                R=pose_groundtruth["R"],
                T=pose_groundtruth["T"]),
            fovx=args.fovx, fovy=args.fovy, width=args.width, height=args.height)
        distorted_image, depth = render_frame(groundtruth_camera, server_gaussians, pipeline)
        warpedref_image = warping_frame(groundtruth_camera, depth[0, ...], prediction_camera, reference_image.permute(1, 2, 0)).permute(2, 0, 1)
        # show3images(distorted_image, reference_image, warpedref_image)  # debug
        # save2video(distorted_image, warpedref_image)  # debug
        save2images(distorted_image, warpedref_image, n_frame, n_render)  # debug

    if videoout is not None:
        videoout.release()

# TODO: 渲染预测视角处的低清图
# TODO: 读取带宽数据
# TODO: 决策量化级别, 预测视角内塞满带宽
# TODO: 按照量化级别执行反量化
# TODO: 渲染色彩失真图
# TODO: 色彩恢复
# TODO: 测质量
