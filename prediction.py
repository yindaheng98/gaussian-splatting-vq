import os
import argparse
import torch
import pandas
import numpy as np
from predictions.VAR import get_var_model, predict_var_model
from typing import NamedTuple, List
from scene.camera_dataset import CameraPoseDataset, Pose
from gaussian_renderer import render, GaussianModel
from scene.gaussian_vq import VQGaussianModel, Attribute
from scene.gaussian_kmeans import KMeansGaussianModel
from arguments import PipelineParams
from scene.cameras import Camera as View
from utils.system_utils import searchForMaxIteration
from warping import fromJSON, reconstrucion, projection, warp

parser = argparse.ArgumentParser()
parser.add_argument("--cameras", type=str, required=True, help="Path to the camera pose.")
parser.add_argument("--fovx", type=float, default=1.4773902348773813, help="Camera fov x axis.")
parser.add_argument("--fovy", type=float, default=1.2005465997792715, help="Camera fov y axis.")
parser.add_argument("--width", type=float, default=1600, help="Camera width.")
parser.add_argument("--height", type=float, default=1200, help="Camera height.")
parser.add_argument("--fps", type=int, default=30, help="Playback fps.")
parser.add_argument("--history-size", type=int, default=15)
parser.add_argument("--prediction-stride", type=int, default=5)
parser.add_argument("--prediction-length", type=int, default=5)
parser.add_argument("--video", type=str, required=True)
parser.add_argument("--sh-degree", type=int, default=3)
parser.add_argument("--max-frame", type=int, default=30)

if __name__ == "__main__":
    torch.device("cuda").__enter__()
    pipeline = PipelineParams(parser)
    args = parser.parse_args()
    pose_dataset = CameraPoseDataset(args.cameras, history_size=args.history_size, prediction_stride=args.prediction_stride, prediction_length=args.prediction_length)
    frame_stride = 1/args.fps
    last_frame = None
    server_gaussians = GaussianModel(args.sh_degree)
    model = get_var_model(args.cameras)
    for i, (pose_history, pose_groundtruth) in enumerate(pose_dataset):
        n_frame = i + 1
        timestamp = pose_history["timestamp"][0].item()
        if n_frame > args.max_frame:
            break
        print(f"{timestamp:.4f}", "frame", n_frame, "loading")
        pose_prediction = predict_var_model(model, pose_history, prediction_stride=args.prediction_stride, prediction_length=args.prediction_length)
        print(pose_prediction)
