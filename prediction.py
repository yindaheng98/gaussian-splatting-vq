import argparse
import torch
from predictions.VAR import VARPrediction
from predictions.LSTM import LSTMPrediction
from scene.camera_dataset import CameraPoseDataset
from arguments import PipelineParams
import torch.optim as optim

parser = argparse.ArgumentParser()
parser.add_argument("--cameras", type=str, required=True, help="Path to the camera pose.")
parser.add_argument("--fovx", type=float, default=1.4773902348773813, help="Camera fov x axis.")
parser.add_argument("--fovy", type=float, default=1.2005465997792715, help="Camera fov y axis.")
parser.add_argument("--width", type=float, default=1600, help="Camera width.")
parser.add_argument("--height", type=float, default=1200, help="Camera height.")
parser.add_argument("--fps", type=int, default=30, help="Playback fps.")
parser.add_argument("--history-size", type=int, default=15)
parser.add_argument("--prediction-stride", type=int, default=9)
parser.add_argument("--prediction-length", type=int, default=3)
parser.add_argument("--save", type=str, required=True)

if __name__ == "__main__":
    torch.device("cuda").__enter__()
    pipeline = PipelineParams(parser)
    args = parser.parse_args()
    pose_dataset = CameraPoseDataset(args.cameras, history_size=args.history_size, prediction_stride=args.prediction_stride, prediction_length=args.prediction_length)
    frame_stride = 1/args.fps
    last_frame = None
    model = LSTMPrediction(args.cameras)
    teacher = VARPrediction(args.cameras)
    optimizer = optim.SGD(model.lstm.parameters(), lr=0.1)
    for i in range(len(pose_dataset)):
        pose_history, pose_groundtruth = pose_dataset[i]
        n_frame = i + 1
        timestamp = pose_history["timestamp"][0].item()
        model.lstm.zero_grad()
        pose_prediction = model.predict(pose_history, prediction_stride=args.prediction_stride, prediction_length=args.prediction_length)
        mse_R = torch.sqrt(((pose_groundtruth['R'] - pose_prediction['R'])**2).mean())
        mse_T = torch.sqrt(((pose_groundtruth['T'] - pose_prediction['T'])**2).mean())
        loss = mse_R + mse_T/10
        loss.backward()
        teacher_pose_prediction = teacher.predict(pose_history, prediction_stride=args.prediction_stride, prediction_length=args.prediction_length)
        tmse_R = torch.sqrt(((pose_groundtruth['R'] - teacher_pose_prediction['R'])**2).mean())
        tmse_T = torch.sqrt(((pose_groundtruth['T'] - teacher_pose_prediction['T'])**2).mean())
        print(f"{timestamp:.4f}", "frame", n_frame, "loading", 'R mse', mse_R.item(), 'T mse', mse_T.item(), 'teacher R mse', tmse_R.item(), 'T mse', tmse_T.item())
        optimizer.step()
    model.save(args.save)
    teacher.save(args.save + ".teacher.pickle")
