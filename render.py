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

import cv2
import numpy as np
import torch
from scene import Scene
import os
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel

def render_set(model_path, name, iteration, views, gaussians, pipeline, background):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")
    ralpha_path = os.path.join(model_path, name, "ours_{}".format(iteration), "ralpha")
    rdepth_path = os.path.join(model_path, name, "ours_{}".format(iteration), "rdepth")
    rnormal_path = os.path.join(model_path, name, "ours_{}".format(iteration), "rnormal")

    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)
    makedirs(ralpha_path, exist_ok=True)
    makedirs(rdepth_path, exist_ok=True)
    makedirs(rnormal_path, exist_ok=True)

    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        render_pkg = render([view], gaussians, pipeline, background)
        rendering = render_pkg["render"]
        gt = view.original_image[0:3, :, :]
        
        torchvision.utils.save_image(rendering[0], os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))
        
        if render_pkg.get("ralpha") is not None and torch.numel(render_pkg.get("ralpha")) > 0:
            torchvision.utils.save_image(render_pkg.get("ralpha")[0], os.path.join(ralpha_path, '{0:05d}'.format(idx) + ".png"))
        
        if render_pkg.get("rdepth") is not None and torch.numel(render_pkg.get("rdepth")) > 0:
            # to numpy
            depth_map_np = render_pkg.get("rdepth")[0, 0].cpu().numpy()
            mask = depth_map_np == 0
            depth_map_np[mask] == np.nan
            depth_map_np[mask] = np.nanmin(depth_map_np) - 1

            # normalized
            depth_map_np = (depth_map_np - np.min(depth_map_np)) / (np.max(depth_map_np) - np.min(depth_map_np))
            depth_map_np = (depth_map_np * 255).astype(np.uint8)

            # color
            color_map = cv2.applyColorMap(depth_map_np, cv2.COLORMAP_JET)
            color_map_rgb = cv2.cvtColor(color_map, cv2.COLOR_BGR2RGB)
            color_map_rgb *= ~mask[..., np.newaxis]
            color_map_tensor = torch.from_numpy(color_map_rgb).permute(2, 0, 1) / 255.0
            
            torchvision.utils.save_image(color_map_tensor, os.path.join(rdepth_path, '{0:05d}'.format(idx) + ".png"))

        if render_pkg.get("rnormal") is not None and torch.numel(render_pkg.get("rnormal")) > 0:
            mask = torch.isclose(render_pkg.get("rnormal")[0], torch.tensor(0.0), atol=1e-7).all(dim=0)
            normal = (-render_pkg.get("rnormal")[0] * 0.5 + 0.5) * (~mask).float()
            torchvision.utils.save_image(normal, os.path.join(rnormal_path, '{0:05d}'.format(idx) + ".png"))

def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        print(f"Point Number: {gaussians.get_xyz.shape[0]}...")
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        if not skip_train:
             render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background)

        if not skip_test:
             render_set(dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, background)

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test)