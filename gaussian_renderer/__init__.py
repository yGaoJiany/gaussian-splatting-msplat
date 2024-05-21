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
import math
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from scene.gaussian_model import GaussianModel
from utils.graphics_utils import fov2focal
from utils.sh_utils import eval_sh

import msplat as ms

def render(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, override_color = None):
    """
    Render the scene with msplat.
    
    Background tensor (bg_color) must be on GPU!
    """
    
    # print("Hello, rendering with msplat.")
    
    # tranform 3dgs to msplat
    position = pc.get_xyz
    fovx = viewpoint_camera.FoVx
    fovy = viewpoint_camera.FoVy
    width = int(viewpoint_camera.image_width)
    height = int(viewpoint_camera.image_height)
    
    fx = fov2focal(fovx, width)
    fy = fov2focal(fovy, height)
    cx = float(width) / 2
    cy = float(height) / 2
    
    intrinsic_params = torch.tensor([fx, fy, cx, cy]).cuda().float()
    # extrinsic_matrix = viewpoint_camera.world_view_transform
    # extrinsic_matrix = extrinsic_matrix[:3, :]
    # camera_center = viewpoint_camera.camera_center
    extrinsic_matrix, camera_center = viewpoint_camera.get_extrinsic_camcenter()
    
    opacity = pc.get_opacity
    shs = pc.get_features.permute(0, 2, 1)
    scaling = pc.get_scaling
    rotation = pc.get_rotation

    # project points and perform culling
    (uv, depth) = ms.project_point(
        position,
        intrinsic_params,
        extrinsic_matrix,
        width, height)

    visible = depth != 0

    # compute sh if not None
    direction = (position -
                camera_center.repeat(position.shape[0], 1))
    direction = direction / direction.norm(dim=1, keepdim=True)
    
    sh2rgb = ms.compute_sh(shs, direction, visible)
    rgb = torch.clamp_min(sh2rgb + 0.5, 0.0)
    
    # compute cov3d
    cov3d = ms.compute_cov3d(scaling, rotation, visible)

    # ewa project
    (conic, radius, tiles_touched) = ms.ewa_project(
        position,
        cov3d,
        intrinsic_params,
        extrinsic_matrix,
        uv,
        width,
        height,
        visible
    )

    # sort
    (gaussian_ids_sorted, tile_range) = ms.sort_gaussian(
        uv, depth, width, height, radius, tiles_touched
    )
    
    # render
    ndc = torch.zeros_like(uv, requires_grad=True)
    try:
        ndc.retain_grad()
    except:
        raise ValueError("ndc does not have grad")

    # alpha blending
    render = ms.alpha_blending(
        uv, conic, opacity, rgb,
        gaussian_ids_sorted, tile_range, bg_color[0].item(), width, height, ndc
    )
    
    return {"render": render,
            "viewspace_points": ndc,
            "visibility_filter" : radius > 0,
            "radii": radius}
    

# def render(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, override_color = None):
#     """
#     Render the scene. 
    
#     Background tensor (bg_color) must be on GPU!
#     """
 
#     # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
#     screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
#     try:
#         screenspace_points.retain_grad()
#     except:
#         pass

#     # Set up rasterization configuration
#     tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
#     tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

#     raster_settings = GaussianRasterizationSettings(
#         image_height=int(viewpoint_camera.image_height),
#         image_width=int(viewpoint_camera.image_width),
#         tanfovx=tanfovx,
#         tanfovy=tanfovy,
#         bg=bg_color,
#         scale_modifier=scaling_modifier,
#         viewmatrix=viewpoint_camera.world_view_transform,
#         projmatrix=viewpoint_camera.full_proj_transform,
#         sh_degree=pc.active_sh_degree,
#         campos=viewpoint_camera.camera_center,
#         prefiltered=False,
#         debug=pipe.debug
#     )

#     rasterizer = GaussianRasterizer(raster_settings=raster_settings)

#     means3D = pc.get_xyz
#     means2D = screenspace_points
#     opacity = pc.get_opacity

#     # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
#     # scaling / rotation by the rasterizer.
#     scales = None
#     rotations = None
#     cov3D_precomp = None
#     if pipe.compute_cov3D_python:
#         cov3D_precomp = pc.get_covariance(scaling_modifier)
#     else:
#         scales = pc.get_scaling
#         rotations = pc.get_rotation

#     # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
#     # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
#     shs = None
#     colors_precomp = None
#     if override_color is None:
#         if pipe.convert_SHs_python:
#             shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
#             dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1))
#             dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
#             sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
#             colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
#         else:
#             shs = pc.get_features
#     else:
#         colors_precomp = override_color

#     # Rasterize visible Gaussians to image, obtain their radii (on screen). 
#     rendered_image, radii = rasterizer(
#         means3D = means3D,
#         means2D = means2D,
#         shs = shs,
#         colors_precomp = colors_precomp,
#         opacities = opacity,
#         scales = scales,
#         rotations = rotations,
#         cov3D_precomp = cov3D_precomp)

#     # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
#     # They will be excluded from value updates used in the splitting criteria.
#     return {"render": rendered_image,
#             "viewspace_points": screenspace_points,
#             "visibility_filter" : radii > 0,
#             "radii": radii}
