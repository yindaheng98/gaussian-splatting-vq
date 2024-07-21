import cv2
import argparse
import os
import re


parser = argparse.ArgumentParser()
parser.add_argument("--images", type=str, required=True, help="Images.")
parser.add_argument("--gt", type=str, required=True, help="GT images.")
parser.add_argument("--match", type=str, default=r"[0-9]+.png")

if __name__ == "__main__":
    args = parser.parse_args()
    for entry in os.scandir(args.images):
        if not re.match(args.match, entry.name):
            continue
        lr_path = entry.path
        gt_path = os.path.join(args.gt, entry.name)
        lr = cv2.imread(lr_path)
        gt = cv2.imread(gt_path)
        print(lr_path, gt_path)
