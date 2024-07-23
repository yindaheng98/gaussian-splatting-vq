import re
import torch
from torch.utils.data import Dataset
from typing import NamedTuple
from utils.camera_utils import quaternion_to_matrix


class Pose(NamedTuple):
    timestamp: float
    R: torch.Tensor
    T: torch.Tensor


class CameraPoseDataset(Dataset):
    regex = re.compile(r"^([0-9.]+), \(([-0-9.]+), ([-0-9.]+), ([-0-9.]+)\), \(([-0-9.]+), ([-0-9.]+), ([-0-9.]+), ([-0-9.]+)\)")

    def __init__(self, path, history_size=8, prediction_stride=6, prediction_length=3):
        self.history_size = history_size
        self.prediction_stride = prediction_stride
        self.prediction_length = prediction_length
        self.item_size = self.history_size + self.prediction_stride + self.prediction_length
        with open(path, 'r') as f:
            ts, Ts, quaternions = [], [], []
            for line in f.readlines():
                find = re.findall(self.regex, line)[0]
                find = [float(f) for f in find]
                t, T, quaternion = find[0], find[1:4], find[4:8]
                ts.append(t)
                Ts.append(T)
                quaternions.append(quaternion)
            Ts = torch.tensor(Ts)
            Ts[..., 0] *= -1  # for Unity data (inverse x)
            Ts[..., 2] *= -1  # for Unity data (inverse z)
            Rs = quaternion_to_matrix(torch.tensor(quaternions)[..., [3, 0, 1, 2]])  # for Unity data (real first)
            Rs[:, [[False, True, False], [True, False, True], [False, True, False]]] *= -1  # for Unity data (inverse xz)
            # Rs = quaternion_to_matrix(torch.tensor(quaternions))
            self.poses = [Pose(timestamp=t, R=Rs[i, ...], T=Ts[i, ...]) for i, t in enumerate(ts)]

    def __len__(self):
        return (len(self.poses) - self.history_size - self.prediction_stride) // self.prediction_length

    def __getitem__(self, idx):
        prediction_idx = self.history_size+self.prediction_stride+idx*self.prediction_length
        groundtruth = self.poses[prediction_idx:prediction_idx+self.prediction_length]
        history_idx = prediction_idx-self.history_size-self.prediction_stride
        history = self.poses[history_idx:history_idx+self.history_size]
        history_timestamp = torch.tensor([h.timestamp for h in history])
        history_R = torch.stack([h.R for h in history], dim=0)
        history_T = torch.stack([h.T for h in history], dim=0)
        groundtruth_timestamp = torch.tensor([h.timestamp for h in groundtruth])
        groundtruth_R = torch.stack([h.R for h in groundtruth], dim=0)
        groundtruth_T = torch.stack([h.T for h in groundtruth], dim=0)
        pred_timestamp = [h.timestamp for h in self.poses[prediction_idx-self.prediction_stride:prediction_idx+self.prediction_length]]
        history_ = {"timestamp": history_timestamp, "R": history_R, "T": history_T, "pred_timestamp": pred_timestamp}
        groundtruth_ = {"timestamp": groundtruth_timestamp, "R": groundtruth_R, "T": groundtruth_T}
        return history_, groundtruth_
