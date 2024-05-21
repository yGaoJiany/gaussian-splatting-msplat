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
from torch import nn
import numpy as np
from utils.graphics_utils import getWorld2View2, getProjectionMatrix

import roma

class Camera(nn.Module):
    def __init__(self, colmap_id, Q, T, FoVx, FoVy, image, gt_alpha_mask,
                 image_name, uid,
                 trans=np.array([0.0, 0.0, 0.0]), scale=1.0, data_device = "cuda"
                 ):
        super(Camera, self).__init__()

        self.uid = uid
        self.colmap_id = colmap_id
        self.init_Q = torch.tensor(Q, dtype=torch.float32, device="cuda")
        self.Q = nn.Parameter(self.init_Q.requires_grad_(True))
        self.init_T = torch.tensor(T, dtype=torch.float32, device="cuda")
        self.T =  nn.Parameter(self.init_T.requires_grad_(True))
        # self.R = R
        # self.T = T
        self.FoVx = FoVx
        self.FoVy = FoVy
        self.image_name = image_name

        try:
            self.data_device = torch.device(data_device)
        except Exception as e:
            print(e)
            print(f"[Warning] Custom device {data_device} failed, fallback to default cuda device" )
            self.data_device = torch.device("cuda")

        self.original_image = image.clamp(0.0, 1.0).to(self.data_device)
        self.image_width = self.original_image.shape[2]
        self.image_height = self.original_image.shape[1]

        if gt_alpha_mask is not None:
            self.original_image *= gt_alpha_mask.to(self.data_device)
        else:
            self.original_image *= torch.ones((1, self.image_height, self.image_width), device=self.data_device)

        self.zfar = 100.0
        self.znear = 0.01

        self.trans = trans
        self.scale = scale
        
        # self.optimizer = torch.optim.Adam(self.parameters(), lr=0.0001)
        
    def get_extrinsic_camcenter(self):
        R = roma.unitquat_to_rotmat(self.Q)
        Rt = torch.zeros((4, 4), dtype=torch.float32).to(self.Q.device)
        Rt[:3, :3] = R
        Rt[:3, 3] = self.T
        Rt[3, 3] = 1.0

        extrinsic_matrix = Rt[:3, :]
        world_view_transform = Rt.transpose(0, 1)
        camera_center = world_view_transform.inverse()[3, :3]
        return extrinsic_matrix, camera_center

class MiniCam:
    def __init__(self, width, height, fovy, fovx, znear, zfar, world_view_transform, full_proj_transform):
        self.image_width = width
        self.image_height = height    
        self.FoVy = fovy
        self.FoVx = fovx
        self.znear = znear
        self.zfar = zfar
        self.world_view_transform = world_view_transform
        self.full_proj_transform = full_proj_transform
        view_inv = torch.inverse(self.world_view_transform)
        self.camera_center = view_inv[3][:3]

