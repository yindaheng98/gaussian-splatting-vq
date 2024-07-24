from .VAR import VARPrediction
from .LSTM import LSTMPrediction
prediction_dict = {
    "VAR": VARPrediction,
    "LSTM": LSTMPrediction
}
