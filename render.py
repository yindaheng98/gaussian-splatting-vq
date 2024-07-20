#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
from scene import Scene
import os
import json
from tqdm import tqdm
import numpy as np
from os import makedirs
from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
import cv2
# import tifffile

def depth_colormap(depth):
    depth_valid = depth[depth<depth.max()]
    depth_max = torch.topk(depth_valid, depth_valid.shape[0]//10).values[-1]
    depth_min = depth_valid.min()
    depth_preview = torch.clamp((depth-depth_min)/(depth_max-depth_min), 0, 1)
    depth_colored = cv2.applyColorMap((depth_preview[0,...]*255).type(torch.uint8).cpu().numpy(), cv2.COLORMAP_JET)
    return depth_colored

def render_set(model_path, name, iteration, views, gaussians, pipeline, background):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")

    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)

    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        render_pkg = render(view, gaussians, pipeline, background)
        rendering, depth = render_pkg["render"], render_pkg["depth"]
        if hasattr(view, "original_image") and view.original_image is not None:
            gt = view.original_image[0:3, :, :]
            torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
        cv2.imwrite(os.path.join(render_path, '{0:05d}'.format(idx) + ".depth.png"), depth_colormap(depth))
        np.savez_compressed(os.path.join(render_path, '{0:05d}'.format(idx) + ".depth.npz"), depth=depth.cpu().numpy())
        # tifffile.imsave(os.path.join(render_path, '{0:05d}'.format(idx) + ".depth.tif"), depth.cpu().numpy()) # another method
        with open(os.path.join(render_path, '{0:05d}'.format(idx) + ".camera.json"), "w", encoding="utf8") as f:
            json.dump(view.toJSON(idx), f, indent=2)

def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train : bool, render_train_interp : bool, skip_test : bool, args):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        if not skip_train:
             render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background)

        if render_train_interp:
             render_set(dataset.model_path, "train_interp", scene.loaded_iter, scene.getTrainInterpCameras(fovx=args.forcefovx, fovy=args.forcefovy, width=args.forcewidth, height=args.forceheight), gaussians, pipeline, background)

        if not skip_test:
             render_set(dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, background)

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--render_train_interp", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--forcefovx", type=float, default=0)
    parser.add_argument("--forcefovy", type=float, default=0)
    parser.add_argument("--forcewidth", type=int, default=0)
    parser.add_argument("--forceheight", type=int, default=0)
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.render_train_interp, args.skip_test, args)