import argparse
from typing import NamedTuple
from scene.camera_dataset import CameraPoseDataset, Pose

parser = argparse.ArgumentParser()
parser.add_argument("--viewport", type=str, required=True, help="Path to the camera pose.")
parser.add_argument("--fovx", type=float, default=888, help="Camera fov x axis.")
parser.add_argument("--fovy", type=float, default=888, help="Camera fov y axis.")
parser.add_argument("--width", type=float, default=1600, help="Camera width.")
parser.add_argument("--height", type=float, default=1200, help="Camera height.")
parser.add_argument("--buffersize", type=int, default=10, help="Camera buffer size.")


class Camera(NamedTuple):
    pose: Pose
    fovx: float
    fovy: float
    width: int
    height: int


def predict_viewport(pose_history, fovx: float, fovy: float, width: int, height: int):
    return Camera(pose=pose_history[-1], fovx=fovx, fovy=fovy, width=width, height=height)


if __name__ == "__main__":
    args = parser.parse_args()
    camerapose_dataset = CameraPoseDataset(args.viewport)
    for camera in camerapose_dataset:
        print(camera)


# TODO: 渲染预测视角处的低清图
# TODO: 读取带宽数据
# TODO: 决策量化级别, 预测视角内塞满带宽
# TODO: 按照量化级别执行反量化
# TODO: 渲染色彩失真图
# TODO: 色彩恢复
# TODO: 测质量
