import abc


class Prediction(metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def predict(self, pose_history, prediction_stride, prediction_length):
        pass
