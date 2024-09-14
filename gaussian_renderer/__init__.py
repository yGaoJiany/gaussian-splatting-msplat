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
from scene.gaussian_model import GaussianModel
from utils.graphics_utils import fov2focal

import msplat as mp
import msplat.types as mptype
import msplat.functional as mpf

import m_splat


def render(batch_camera: list, pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, override_color = None):
    """
    Render the scene with msplat2.
    """

    # configs
    camera_type = mptype.CAMERA_ORTHOGRAPHIC if pipe.is_ortho_cam else mptype.CAMERA_PERSPECTIVE
    sort_method = mptype.SORT_RADIX if pipe.sort_method == "radix" else mptype.SORT_MERGE

    inter_method = None
    inter_method = mptype.TILE_INTER_AABB if pipe.inter_method == "aabb" else inter_method
    inter_method = mptype.TILE_INTER_OBB if pipe.inter_method == "obb" else inter_method
    inter_method = mptype.TILE_INTER_CON_OBB if pipe.inter_method == "cobb" else inter_method

    render_mode = mptype.RenderType(0)
    render_type = pipe.render_mode.split(", ")
    if "dmtr" in render_type:
        render_mode = render_mode | mptype.RENDERTYPE_DMTR
    if "depth" in render_type:
        render_mode = render_mode | mptype.RENDERTYPE_DEPTH
    if "normal" in render_type:
        render_mode = render_mode | mptype.RENDERTYPE_NORMAL
    if "alpha" in render_type:
        render_mode = render_mode | mptype.RENDERTYPE_ALPHA
    if "aux" in render_type:
        render_mode = render_mode | mptype.RENDERTYPE_AUXILIARY

    # get gaussian properties 
    position = pc.get_xyz
    opacity = pc.get_opacity
    # [P, 0:(active_sh_degree+1)^2, C] -> [P, C, 0:(active_sh_degree+1)^2]
    shs = pc.get_features[:, 0:(pc.active_sh_degree+1)**2, :].permute(0, 2, 1) 
    scaling = pc.get_scaling * scaling_modifier
    rotation = pc.get_rotation

    # deal with camera in a batch
    intrinsic_params = []
    extrinsic_matrix = []
    camera_center = []
    for b in range(len(batch_camera)):
        camera = batch_camera[b]

        fovx = camera.FoVx
        fovy = camera.FoVy
        width = int(camera.image_width)
        height = int(camera.image_height)
        
        fx = fov2focal(fovx, width)
        fy = fov2focal(fovy, height)
        cx = float(width) / 2
        cy = float(height) / 2
    
        intr = torch.tensor([fx, fy, cx, cy]).cuda().float()
        extr = camera.world_view_transform.transpose(0, 1)
        extr = extr[:3, :]
        center = camera.camera_center

        intrinsic_params.append(intr)
        extrinsic_matrix.append(extr)
        camera_center.append(center)
    
    intrinsic_params = torch.stack(intrinsic_params, dim=0)        # [B, 4]
    extrinsic_matrix = torch.stack(extrinsic_matrix, dim=0)        # [B, 3, 4]
    camera_center = torch.stack(camera_center, dim=0)              # [B, 3]

    # msplat1
    # uv, depth = m_splat.project_point(position, intrinsic_params[0], extrinsic_matrix[0], width, height, 0)
    # visible = depth != 0
    # uvd = torch.concat([uv, depth], dim=-1).unsqueeze(0)
    # visible = visible.unsqueeze(0)

    # cov3d = m_splat.compute_cov3d(scaling, rotation, visible=visible[0])
    # conic, radius, tiles = m_splat.ewa_project(position, cov3d, intrinsic_params[0], extrinsic_matrix[0], uv, width, height, visible[0])
    # conic = conic.unsqueeze(0)
    # radius = radius.unsqueeze(0)
    # tiles = tiles.unsqueeze(0)

    # project points and perform culling
    with torch.no_grad():
        uvd = mpf.project_point(position, intrinsic_params, extrinsic_matrix, cam_type=camera_type)
        visible = torch.logical_and(uvd[..., 2:] >= 0, uvd[..., 0:1] < 1.3 * width)
        visible = torch.logical_and(visible, uvd[..., 0:1] > -0.3 * width)
        visible = torch.logical_and(visible, uvd[..., 1:2] < 1.3 * height)
        visible = torch.logical_and(visible, uvd[..., 1:2] > -0.3 * height)

    # ewa project
    uvd, conic = mpf.ewa_project(
        position, 
        scaling, 
        rotation, 
        intrinsic_params, 
        extrinsic_matrix,
        cam_type=camera_type, 
        visible=visible)
    
    # evaluate sh
    direction = (position[None] - camera_center[:, None, :])      # [B, N, 3]
    direction = direction / direction.norm(dim=2, keepdim=True)
    
    # set is_standard to false for alignment of the original 3DGS
    sh = mp.SphericalHarmonics(shs, is_standard=False) 
    sh2rgb = sh.eval(direction, visible=visible)
    rgb = torch.clamp_min(sh2rgb + 0.5, 0.0)

    # sh2rgb = m_splat.compute_sh(shs, direction[0], visible)
    # rgb = torch.clamp_min(sh2rgb + 0.5, 0.0)
    # rgb = rgb.unsqueeze(0)

    # sort
    key, index, radius = mpf.compute_gaussian_key(
        uvd, conic, (height, width), inter_type=inter_method, sort_method=sort_method)
    tile_range = mpf.compute_tile_range(key, (height, width))

    # index, tile_range = m_splat.sort_gaussian(uvd[0, :, :2], uvd[0, :, -1:], width, height, radius[0], tiles[0])
    # tile_range = tile_range.unsqueeze(0)
    
    # render
    ndc = torch.zeros((uvd.shape[0], uvd.shape[1], 4), device=uvd.device, requires_grad=True)
    try:
        ndc.retain_grad()
    except:
        raise ValueError("ndc does not have grad")

    # alpha blending
    rfeat, ralpha, rdepth, rnormal, raux = mpf.alpha_blending(
        uvd, 
        conic, 
        intrinsic_params, 
        extrinsic_matrix, 
        position, 
        rotation, 
        scaling, 
        opacity, 
        rgb,
        index, 
        tile_range, 
        (height, width), 
        background = bg_color,
        ndc=ndc,
        homo_grad=pipe.homo_grad,
        cam_type=camera_type,
        mode=render_mode
    )
    # rfeat = m_splat.alpha_blending(uvd[0, :, :2], conic[0], opacity, rgb[0], index, tile_range[0], 0.0, width, height, ndc[0, :, :2])
    # rfeat = rfeat.unsqueeze(0)

    return {"render": rfeat,
            "viewspace_points": ndc,
            "visibility_filter" : radius > 0, 
            "radii": radius}
#             # "ralpha": ralpha,
#             # "rdepth": rdepth,
#             # "rnormal": rnormal,
#             # "raux": raux,

# def render(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, override_color = None):
#     """
#     Render the scene with msplat.
    
#     Background tensor (bg_color) must be on GPU!
#     """
    
#     # print("Hello, rendering with msplat.")
    
#     viewpoint_camera = viewpoint_camera[0]

#     # tranform 3dgs to msplat
#     position = pc.get_xyz
#     fovx = viewpoint_camera.FoVx
#     fovy = viewpoint_camera.FoVy
#     width = int(viewpoint_camera.image_width)
#     height = int(viewpoint_camera.image_height)
    
#     fx = fov2focal(fovx, width)
#     fy = fov2focal(fovy, height)
#     cx = float(width) / 2
#     cy = float(height) / 2
    
#     intrinsic_params = torch.tensor([fx, fy, cx, cy]).cuda().float()
#     extrinsic_matrix = viewpoint_camera.world_view_transform.transpose(0, 1)
#     extrinsic_matrix = extrinsic_matrix[:3, :]
#     camera_center = viewpoint_camera.camera_center
    
#     opacity = pc.get_opacity
#     shs = pc.get_features[:, 0:(pc.active_sh_degree+1)**2, :].permute(0, 2, 1) 
#     scaling = pc.get_scaling
#     rotation = pc.get_rotation

#     # project points and perform culling
#     with torch.profiler.record_function("project_point"):
#         (uv, depth) = m_splat.project_point(
#             position,
#             intrinsic_params,
#             extrinsic_matrix,
#             width, height)

#     visible = depth != 0

#     # compute sh if not None
#     direction = (position - camera_center.repeat(position.shape[0], 1))
#     direction = direction / direction.norm(dim=1, keepdim=True)
    
#     with torch.profiler.record_function("compute_sh"):
#         sh2rgb = m_splat.compute_sh(shs, direction, visible)
#     rgb = torch.clamp_min(sh2rgb + 0.5, 0.0)
    
#     # compute cov3d
#     with torch.profiler.record_function("compute_cov3d"):
#         cov3d = m_splat.compute_cov3d(scaling, rotation, visible)

#     # ewa project
#     with torch.profiler.record_function("ewa_project"):
#         (conic, radius, tiles_touched) = m_splat.ewa_project(
#             position,
#             cov3d,
#             intrinsic_params,
#             extrinsic_matrix,
#             uv,
#             width,
#             height,
#             visible
#         )

#     # sort
#     with torch.profiler.record_function("sort_gaussian"):
#         (gaussian_ids_sorted, tile_range) = m_splat.sort_gaussian(
#             uv, depth, width, height, radius, tiles_touched
#         )
    
#     # render
#     ndc = torch.zeros((1, uv.shape[0], 4), device=uv.device, requires_grad=True)
#     try:
#         ndc.retain_grad()
#     except:
#         raise ValueError("ndc does not have grad")

#     # alpha blending
#     with torch.profiler.record_function("alpha_blending"):
#         render = m_splat.alpha_blending(
#             uv, conic, opacity, rgb,
#             gaussian_ids_sorted, tile_range, bg_color[0].item(), width, height, ndc[0, :, :2]
#         )
    
#     return {"render": render.unsqueeze(0),
#             "viewspace_points": ndc,
#             "visibility_filter" : radius.unsqueeze(0) > 0,
#             "radii": radius.unsqueeze(0)}