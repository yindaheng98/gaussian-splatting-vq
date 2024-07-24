import torch
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.api import VAR
from statsmodels.tsa.vector_ar.var_model import VARResults
from scene.camera_dataset import CameraPoseDataset, Pose
from utils.camera_utils import matrix_to_quaternion, quaternion_to_matrix
from .base import Prediction
from typing import NamedTuple
import sys
import os
sys.path.append("predictions/Autoformer")
if "predictions/Autoformer" in sys.path:
    from args import parser
    from utils2.timefeatures import time_features
    from models import Informer, Autoformer, Transformer, Reformer
model_dict = {
    'Autoformer': Autoformer,
    'Transformer': Transformer,
    'Informer': Informer,
    'Reformer': Reformer,
}


class TransformerPrediction(Prediction):
    fps = 90

    def __init__(self, path):
        super().__init__()
        args = parser.parse_args(args=[
            "--is_training",
            "0",
            "--train_epochs",
            "1000",
            "--patience",
            "10",
            "--root_path",
            "./dataset/camera/",
            "--data_path",
            "test.txt",
            "--model_id",
            "ECL_96_96",
            "--model",
            "Informer",
            "--freq",
            "h",
            "--data",
            "camera",
            "--features",
            "M",
            "--seq_len",
            "96",
            "--label_len",
            "48",
            "--pred_len",
            "96",
            "--e_layers",
            "2",
            "--d_layers",
            "1",
            "--factor",
            "3",
            "--enc_in",
            "7",
            "--dec_in",
            "7",
            "--c_out",
            "7",
            "--des",
            "'Exp'",
            "--itr",
            "1",
        ])
        self.model = model_dict[args.model].Model(args).float()
        ii = 0
        setting = '{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_dt{}_{}_{}'.format(args.model_id,
                                                                                                      args.model,
                                                                                                      args.data,
                                                                                                      args.features,
                                                                                                      args.seq_len,
                                                                                                      args.label_len,
                                                                                                      args.pred_len,
                                                                                                      args.d_model,
                                                                                                      args.n_heads,
                                                                                                      args.e_layers,
                                                                                                      args.d_layers,
                                                                                                      args.d_ff,
                                                                                                      args.factor,
                                                                                                      args.embed,
                                                                                                      args.distil,
                                                                                                      args.des, ii)

        self.model.load_state_dict(torch.load(os.path.join('./predictions/Autoformer/checkpoints/' + setting, 'checkpoint.pth')))
        self.freq = args.freq
        self.args = args
        self.seq_len = args.seq_len
        self.label_len = args.label_len
        self.pred_len = args.pred_len
        self.model.eval()

    def predict(self, pose_history, prediction_stride, prediction_length):
        R = pose_history['R']
        T = pose_history['T']
        Q = matrix_to_quaternion(R)
        data = torch.concat((T/10, Q), dim=-1)
        data = torch.concat((data, torch.zeros(size=(prediction_stride + prediction_length, data.shape[1]), dtype=data.dtype, device=data.device)), dim=0)
        ts = torch.concat((pose_history['timestamp'], pose_history['pred_timestamp']), dim=0)
        data_stamp = time_features(pd.to_datetime(ts.cpu().numpy() * self.fps, unit=self.freq), freq=self.freq)
        data_stamp = torch.tensor(data_stamp.transpose(1, 0))

        batch_x = data.float().to('cuda')
        batch_y = data.float().to('cuda')
        batch_x_mark = data_stamp.float().to('cuda')
        batch_y_mark = data_stamp.float().to('cuda')
        self.model.pred_len = prediction_length + prediction_stride
        self.model.label_len = R.shape[0]
        self.args.pred_len = prediction_length + prediction_stride
        self.args.label_len = R.shape[0]
        pred = self._predict(batch_x, batch_y, batch_x_mark, batch_y_mark).squeeze(0).detach().cpu().numpy()
        Q = pred[prediction_stride:, 3:]
        T = torch.tensor(pred[prediction_stride:, :3], device=R.device, dtype=R.dtype)*10
        R = quaternion_to_matrix(torch.tensor(Q, device=R.device, dtype=R.dtype))
        return {'R': R, 'T': T}

    def _predict(self, batch_x, batch_y, batch_x_mark, batch_y_mark):
        # encoder - decoder

        def _run_model():
            outputs = self.model(batch_x.unsqueeze(0), batch_x_mark.unsqueeze(0), batch_y.unsqueeze(0), batch_y_mark.unsqueeze(0))
            if self.args.output_attention:
                outputs = outputs[0]
            return outputs

        if self.args.use_amp:
            with torch.cuda.amp.autocast():
                outputs = _run_model()
        else:
            outputs = _run_model()

        return outputs
