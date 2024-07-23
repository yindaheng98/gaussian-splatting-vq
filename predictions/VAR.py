import torch
import statsmodels.api as sm
from statsmodels.tsa.api import VAR
from statsmodels.tsa.vector_ar.var_model import VARResults
from scene.camera_dataset import CameraPoseDataset, Pose
from utils.camera_utils import matrix_to_quaternion, quaternion_to_matrix
from .base import Prediction


class VARPrediction(Prediction):
    def __init__(self, path, history_size=100):
        super().__init__()
        pose_dataset = CameraPoseDataset(path, history_size=history_size, prediction_stride=1, prediction_length=1)
        pose_history, pose_groundtruth = pose_dataset[0]
        R = pose_history['R']
        T = pose_history['T']
        Q = matrix_to_quaternion(R)
        t = pose_history['timestamp'].cpu().numpy()
        data = torch.concat((Q, T), dim=-1).cpu().numpy()
        self.model = VAR(endog=data).fit(2)

    def predict(self, pose_history, prediction_stride, prediction_length):
        R = pose_history['R']
        T = pose_history['T']
        Q = matrix_to_quaternion(R)
        data = torch.concat((Q, T), dim=-1).cpu().numpy()
        pred = self.model.forecast(data, steps=prediction_stride + prediction_length)
        Q = pred[prediction_stride:, :4]
        T = torch.tensor(pred[prediction_stride:, 4:], device=R.device, dtype=R.dtype)
        R = quaternion_to_matrix(torch.tensor(Q, device=R.device, dtype=R.dtype))
        return {'R': R, 'T': T}
