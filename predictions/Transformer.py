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
    def __init__(self, path):
        super().__init__()
        args = parser.parse_args(args=[
            "--is_training",
            "1",
            "--root_path",
            "./dataset/camera/",
            "--data_path",
            "test.txt",
            "--model_id",
            "ECL_96_96",
            "--model",
            "Autoformer",
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
