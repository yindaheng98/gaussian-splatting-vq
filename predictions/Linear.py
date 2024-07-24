from .base import PredictionFoV
from sklearn.linear_model import LinearRegression
import numpy as np


class LinearPredictionFoV(PredictionFoV):
    def __init__(self) -> None:
        super().__init__()
        data = np.genfromtxt("fov.txt", delimiter=",")
        y, X = data[..., :2], data[..., 2:]
        model = LinearRegression()
        model.fit(X, y)
        self.model = model

    def predict(self, speed):
        return self.model.predict(speed.unsqueeze(0).cpu().numpy())[0, ...]
