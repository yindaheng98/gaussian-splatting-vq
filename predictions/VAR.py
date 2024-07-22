import torch
import statsmodels.api as sm
from statsmodels.tsa.api import VAR
from statsmodels.tsa.vector_ar.var_model import VARResults
from scene.camera_dataset import CameraPoseDataset, Pose
from utils.camera_utils import matrix_to_quaternion, quaternion_to_matrix


def get_var_model(path, history_size=100):
    pose_dataset = CameraPoseDataset(path, history_size=history_size, prediction_stride=1, prediction_length=1)
    pose_history, pose_groundtruth = pose_dataset[0]
    R = pose_history['R']
    T = pose_history['T']
    Q = matrix_to_quaternion(R)
    t = pose_history['timestamp'].cpu().numpy()
    data = torch.concat((Q, T), dim=-1).cpu().numpy()
    model = VAR(endog=data)
    return model.fit(2)


def predict_var_model(model: VARResults, pose_history, prediction_stride, prediction_length):
    R = pose_history['R']
    T = pose_history['T']
    Q = matrix_to_quaternion(R)
    data = torch.concat((Q, T), dim=-1).cpu().numpy()
    pred = model.forecast(data, steps=prediction_stride + prediction_length)
    Q = pred[prediction_stride:, :4]
    T = torch.tensor(pred[prediction_stride:, 4:], device=R.device, dtype=R.dtype)
    R = quaternion_to_matrix(torch.tensor(Q, device=R.device, dtype=R.dtype))
    return {'R': R, 'T': T}
