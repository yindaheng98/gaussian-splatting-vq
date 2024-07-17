import argparse
import numpy as np
import re

import torch
# from utils.camera_utils import quaternion_to_matrix # circular import? why?
from typing import NamedTuple


def quaternion_to_matrix(quaternions: torch.Tensor) -> torch.Tensor:
    # From pytorch3d.transforms.rotation_conversions
    """
    Convert rotations given as quaternions to rotation matrices.

    Args:
        quaternions: quaternions with real part first,
            as tensor of shape (..., 4).

    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """
    r, i, j, k = torch.unbind(quaternions, -1)
    # pyre-fixme[58]: `/` is not supported for operand types `float` and `Tensor`.
    two_s = 2.0 / (quaternions * quaternions).sum(-1)

    o = torch.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        -1,
    )
    return o.reshape(quaternions.shape[:-1] + (3, 3))


parser = argparse.ArgumentParser()
parser.add_argument("--viewport", type=str, required=True, help="Path to the camera pose.")
parser.add_argument("--fovx", type=float, default=888, help="Camera fov x axis.")
parser.add_argument("--fovy", type=float, default=888, help="Camera fov y axis.")
parser.add_argument("--width", type=float, default=1600, help="Camera width.")
parser.add_argument("--height", type=float, default=1200, help="Camera height.")
# TODO: 读取预测视角数据
# TODO: 渲染预测视角处的低清图
# TODO: 读取带宽数据
# TODO: 决策量化级别, 预测视角内塞满带宽
# TODO: 按照量化级别执行反量化
# TODO: 渲染色彩失真图
# TODO: 色彩恢复
# TODO: 测质量


class ReaderIter:
    def __init__(self, cameras):
        self.i = 0
        self.cameras = cameras

    def __next__(self):
        if self.i < len(self.cameras):
            return self.cameras[self.i]
        raise StopIteration


class Pose(NamedTuple):
    timestamp: float
    R: torch.Tensor
    T: torch.Tensor


class PoseReader:
    regex = re.compile(r"^([0-9.]+), \(([-0-9.]+), ([-0-9.]+), ([-0-9.]+)\), \(([-0-9.]+), ([-0-9.]+), ([-0-9.]+), ([-0-9.]+)\)")

    def __init__(self, path):
        self.path = path

    def __iter__(self):
        cameras = []
        with open(self.path, 'r') as f:
            ts, Ts, quaternions = [], [], []
            for line in f.readlines():
                find = re.findall(self.regex, line)[0]
                find = [float(f) for f in find]
                t, T, quaternion = find[0], find[1:4], find[4:8]
                ts.append(t)
                Ts.append(T)
                quaternions.append(quaternion)
            Ts = torch.tensor(Ts)
            Rs = quaternion_to_matrix(torch.tensor(quaternions))
            return ReaderIter([Pose(timestamp=t, R=Rs[i, ...], T=Ts[i, ...]) for i, t in enumerate(ts)])


class Camera(NamedTuple):
    P: Pose
    FovY: np.array
    FovX: np.array
    image: np.array
    width: int
    height: int


def predict_viewport(history):
    return history[-1]


if __name__ == "__main__":
    args = parser.parse_args()
    camera_reader = PoseReader(args.viewport)
    for camera in camera_reader:
        print(camera)
