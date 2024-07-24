import torch
import torch.nn as nn
from .base import Prediction
from .VAR import VARPrediction
from utils.camera_utils import matrix_to_quaternion, quaternion_to_matrix
import torch.nn.init as weight_init


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
        return tag_space, hidden


class LSTMEncDec(nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim):
        super(LSTMEncDec, self).__init__()
        self.lstm_enc = LSTM(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim)
        self.lstm_dec = LSTM(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim)

    def forward(self, sentence, length):
        lstm_out, hidden = self.lstm_enc(sentence)
        data = torch.zeros_like(lstm_out[-1:])
        for _ in range(length):
            output, hidden = self.lstm_dec(data, hidden)
            data = torch.concat((data, output[-1:]), dim=0)
        return data[-length:]


def prepare_sequence(seq, to_ix):
    idxs = [to_ix[w] for w in seq]
    return torch.tensor(idxs, dtype=torch.long)


class LSTMPrediction(Prediction):

    def __init__(self, path):
        self.lstm = LSTMEncDec(input_dim=7, hidden_dim=16, output_dim=7)
        for _, param in self.lstm.named_parameters():
            weight_init.trunc_normal_(param, 0, 1e-8, 0, 0)
        self.var = VARPrediction(path)

    def predict(self, pose_history, prediction_stride, prediction_length):
        var_pred = self.var.predict(pose_history, prediction_stride, prediction_length)
        R = pose_history['R']
        T = pose_history['T']
        Q = matrix_to_quaternion(R)
        data = torch.concat((Q, T), dim=-1)
        data = self.lstm(data, prediction_stride + prediction_length)
        Q = data[-prediction_length:, :4]
        T = data[-prediction_length:, 4:]
        Q[..., 0] += 1.
        R = quaternion_to_matrix(Q)
        return {'R': R @ var_pred['R'], 'T': T * 10 + var_pred['T']}
