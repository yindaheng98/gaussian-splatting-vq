import argparse
import numpy as np
from plyfile import PlyData


def ply2obj(ply_path, obj_path):
    ply = PlyData.read(ply_path)

    with open(obj_path, 'w') as f:
        f.write("# OBJ file\n")
        xyz = np.stack((np.asarray(ply.elements[0]["x"]),
                        np.asarray(ply.elements[0]["y"]),
                        np.asarray(ply.elements[0]["z"])),  axis=1)
        for v in xyz:
            f.write("v %.6f %.6f %.6f \n" % tuple(v))


parser = argparse.ArgumentParser()
parser.add_argument("--src", type=str, required=True, help="The source bson.")
parser.add_argument("--dst", type=str, required=True, help="The destination bson.")

if __name__ == "__main__":
    args = parser.parse_args()
    ply2obj(args.src, args.dst)
