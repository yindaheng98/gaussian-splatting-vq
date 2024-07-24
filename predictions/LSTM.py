import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from .base import Prediction
from utils.camera_utils import matrix_to_quaternion, quaternion_to_matrix


class LSTM(nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim):
        super(LSTM, self).__init__()
        self.hidden_dim = hidden_dim

        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.lstm = nn.LSTM(input_dim, hidden_dim)

        # The linear layer that maps from hidden state space to tag space
        self.hidden2tag = nn.Linear(hidden_dim, output_dim)

    def forward(self, sentence, hidden=None):
        lstm_out, hidden = self.lstm(sentence.view(len(sentence), 1, -1), hidden)
        tag_space = self.hidden2tag(lstm_out.view(len(sentence), -1))
        tag_scores = F.log_softmax(tag_space, dim=1)
        return tag_scores


def prepare_sequence(seq, to_ix):
    idxs = [to_ix[w] for w in seq]
    return torch.tensor(idxs, dtype=torch.long)


class LSTMPrediction(Prediction):

    def __init__(self, path):
        model = LSTM(input_dim=7, hidden_dim=16, output_dim=7)
        self.lstm = model

    def predict(self, pose_history, prediction_stride, prediction_length):
        R = pose_history['R']
        T = pose_history['T']
        Q = matrix_to_quaternion(R)
        data = torch.concat((Q, T), dim=-1)
        for _ in range(prediction_stride + prediction_length):
            output = self.lstm(data)
            data = torch.concat((data, output[-1:]), dim=0)
        Q = data[-prediction_length:, :4]
        T = data[-prediction_length:, 4:]
        R = quaternion_to_matrix(Q)
        return {'R': R, 'T': T}
