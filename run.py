import argparse
from typing import NamedTuple
from scene.camera_dataset import CameraPoseDataset, Pose
from gaussian_renderer import render

parser = argparse.ArgumentParser()
parser.add_argument("--viewport", type=str, required=True, help="Path to the camera pose.")
parser.add_argument("--fovx", type=float, default=888, help="Camera fov x axis.")
parser.add_argument("--fovy", type=float, default=888, help="Camera fov y axis.")
parser.add_argument("--width", type=float, default=1600, help="Camera width.")
parser.add_argument("--height", type=float, default=1200, help="Camera height.")
parser.add_argument("--buffersize", type=int, default=10, help="Camera buffer size.")
parser.add_argument("--fps", type=int, default=30, help="Playback fps.")
parser.add_argument("--history-size", type=int, default=15)
parser.add_argument("--prediction-stride", type=int, default=5)


class Camera(NamedTuple):
    pose: Pose
    fovx: float
    fovy: float
    width: int
    height: int


def predict_viewport(pose_history, fovx: float, fovy: float, width: int, height: int):
    return Camera(
        pose=Pose(
            timestamp=pose_history["timestamp"][-1],
            R=pose_history["R"][-1, ...],
            T=pose_history["T"][-1, ...]),
        fovx=fovx, fovy=fovy, width=width, height=height)


if __name__ == "__main__":
    args = parser.parse_args()
    pose_dataset = CameraPoseDataset(args.viewport, history_size=args.history_size, prediction_stride=args.prediction_stride)
    frame_stride = 1/args.fps
    last_timestamp = None
    frame = 1
    for pose_history, pose_groundtruth in pose_dataset:
        timestamp = pose_history["timestamp"]
        if last_timestamp is not None and timestamp - last_timestamp < frame_stride:
            continue
        prediction_viewport = predict_viewport(pose_history, fovx=args.fovx, fovy=args.fovy, width=args.width, height=args.height)
        print(prediction_viewport)
        last_timestamp = timestamp
        frame = frame + 1


# TODO: 渲染预测视角处的低清图
# TODO: 读取带宽数据
# TODO: 决策量化级别, 预测视角内塞满带宽
# TODO: 按照量化级别执行反量化
# TODO: 渲染色彩失真图
# TODO: 色彩恢复
# TODO: 测质量
