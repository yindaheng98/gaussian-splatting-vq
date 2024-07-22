import cv2
import argparse
import os


parser = argparse.ArgumentParser()
parser.add_argument("--read", type=str, required=True, help="Read which image.")
parser.add_argument("--like", type=str, required=True, help="Like which image.")
parser.add_argument("--save", type=str, required=True, help="Save to where.")
parser.add_argument("--scale", type=float, required=True)

if __name__ == "__main__":
    args = parser.parse_args()
    read = cv2.imread(args.read)
    downsample = cv2.resize(read, (0, 0), fx=args.scale, fy=args.scale)
    like = cv2.imread(args.like)
    h, w, c = like.shape
    save = cv2.resize(downsample, (w, h))
    os.makedirs(os.path.dirname(args.save), exist_ok=True)
    cv2.imwrite(args.save, save)
