from .base import PredictionFoV
from sklearn.linear_model import LinearRegression
import numpy as np
from utils.camera_utils import quaternion_to_axis_angle
import torch


class LinearPredictionFoV(PredictionFoV):
    def __init__(self) -> None:
        super().__init__()
        data = np.genfromtxt("fov.txt", delimiter=",")
        y, X = data[..., :2], quaternion_to_axis_angle(torch.tensor(data[..., 2:])).cpu().numpy()
        model = LinearRegression()
        model.fit(X, y)
        self.model = model

    def predict(self, speed):
        return self.model.predict(quaternion_to_axis_angle(speed.unsqueeze(0)).cpu().numpy())[0, ...]


if __name__ == "__main__":
    data = np.genfromtxt("fov.txt", delimiter=",")
    y, X = data[..., :2], quaternion_to_axis_angle(torch.tensor(data[..., 2:])).cpu().numpy()
    model = LinearRegression()
    model.fit(X, y)
    y_ = model.predict(X)
    print(y_ - y)
