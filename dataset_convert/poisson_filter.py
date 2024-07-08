import argparse
import numpy as np
from plyfile import PlyData
import open3d as o3d

parser = argparse.ArgumentParser()
parser.add_argument("--poisson", type=str, required=True, help="path to the delaunay point cloud")
parser.add_argument("--reference", type=str, required=True, help="path to the reference point cloud")
parser.add_argument("--save", type=str, required=True, help="path to the reference point cloud")
parser.add_argument("--threshold", type=float, default=0.2)


def read_ply(path):
    plydata = PlyData.read(path)
    vertices = plydata['vertex']
    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T
    return positions, colors


def append_ply(xyz, color, src_path, dst_path):
    dtype_full = [(attr, 'float32') for attr in ['x', 'y', 'z']]
    dtype_full += [(attr, 'uint8') for attr in ['red', 'green', 'blue']]
    elements = np.empty(xyz.shape[0], dtype=dtype_full)
    attributes = np.concatenate((xyz, color), axis=1)
    elements[:] = list(map(tuple, attributes))

    plydata = PlyData.read(src_path)
    plydata['vertex'].data = np.r_[plydata['vertex'].data, elements]
    plydata.write(dst_path)


if __name__ == "__main__":
    args = parser.parse_args()

    # Load mesh and convert to open3d.t.geometry.TriangleMesh
    mesh = o3d.io.read_triangle_mesh(args.reference)
    mesh = o3d.t.geometry.TriangleMesh.from_legacy(mesh)

    # Create a scene and add the triangle mesh
    scene = o3d.t.geometry.RaycastingScene()
    _ = scene.add_triangles(mesh)  # we do not need the geometry ID for mesh

    pos_reference, color_reference = read_ply(args.poisson)
    pos_reference_o3d = o3d.core.Tensor(pos_reference, dtype=o3d.core.Dtype.Float32)
    unsigned_distance_o3d = scene.compute_distance(pos_reference_o3d)
    unsigned_distance = unsigned_distance_o3d.numpy()
    filter_index = unsigned_distance < args.threshold
    pos_filtered = pos_reference[filter_index, ...]
    color_filtered = color_reference[filter_index, ...]
    append_ply(pos_filtered, color_filtered, args.reference, args.save)
