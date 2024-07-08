import argparse
import torch
import numpy as np
from plyfile import PlyData, PlyElement
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--delaunay", type=str, required=True, help="path to the delaunay point cloud")
parser.add_argument("--reference", type=str, required=True, help="path to the reference point cloud")
parser.add_argument("--save", type=str, required=True, help="path to the reference point cloud")
parser.add_argument("--batch", type=int, default=1024)


def read_ply(path):
    plydata = PlyData.read(path)
    vertices = plydata['vertex']
    positions = torch.tensor(np.vstack([vertices['x'], vertices['y'], vertices['z']])).T
    colors = torch.tensor(np.vstack([vertices['red'], vertices['green'], vertices['blue']])).T
    return positions, colors


def read_delaunay(path):
    plydata = PlyData.read(path)
    vertices = plydata['vertex']
    positions = torch.tensor(np.vstack([vertices['x'], vertices['y'], vertices['z']])).T
    return positions


def get_color(pos, pos_reference, color_reference):
    n_points = pos.shape[0]
    color = torch.zeros(size=(n_points, color_reference.shape[1]), dtype=color_reference.dtype)
    pbar = tqdm(desc="Processing points", total=n_points)
    for i in range(0, n_points, args.batch):
        step = args.batch if i+args.batch < n_points else (n_points-i)
        dist = torch.norm(pos[i:i+step, ...].unsqueeze(1) - pos_reference.unsqueeze(0), p=2, dim=2)
        idx = dist.argmin(dim=1)
        color[i:i+step] = color_reference[idx, ...]
        pbar.update(step)
    return color


def save_ply(xyz, color, path):
    dtype_full = [(attr, 'float32') for attr in ['x', 'y', 'z']]
    dtype_full += [(attr, 'uint8') for attr in ['red', 'green', 'blue']]
    elements = np.empty(xyz.shape[0], dtype=dtype_full)
    attributes = np.concatenate((xyz, color), axis=1)
    elements[:] = list(map(tuple, attributes))
    el = PlyElement.describe(elements, 'vertex')
    PlyData([el]).write(path)


if __name__ == "__main__":
    args = parser.parse_args()
    with torch.device(device="cuda"):
        pos_delaunay = read_delaunay(args.delaunay)
        pos_reference, color_reference = read_ply(args.reference)
        color_delaunay = get_color(pos_delaunay, pos_reference, color_reference)
        save_ply(pos_delaunay.cpu().numpy(), color_delaunay.cpu().numpy(), args.save)
