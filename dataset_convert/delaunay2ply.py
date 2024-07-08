import argparse
from plyfile import PlyData, PlyElement

parser = argparse.ArgumentParser()
parser.add_argument("--delaunay", type=str, required=True, help="path to the delaunay point cloud")
parser.add_argument("--reference", type=str, required=True, help="path to the reference point cloud")

if __name__ == "__main__":
    args = parser.parse_args()
    data_delaunay = PlyData.read(args.delaunay)
    data_reference = PlyData.read(args.reference)
    print(data_delaunay.elements[0])
    print(data_reference.elements[0])
