# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING
from torch.nn.functional import normalize

from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import subtract_frame_transforms

from pxr import Usd, UsdGeom

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

# Aux functions for observations
def quat_to_rot_matrix(quat: torch.Tensor) -> torch.Tensor:
    """Convert (N, 4) quaternion to (N, 3, 3) rotation matrices."""
    q = normalize(quat, dim=-1)  # (N, 4)
    w, x, y, z = q.unbind(-1)
    B = quat.shape[0]
    R = torch.zeros(B, 3, 3, device=quat.device)

    R[:, 0, 0] = 1 - 2 * (y**2 + z**2)
    R[:, 0, 1] = 2 * (x*y - z*w)
    R[:, 0, 2] = 2 * (x*z + y*w)

    R[:, 1, 0] = 2 * (x*y + z*w)
    R[:, 1, 1] = 1 - 2 * (x**2 + z**2)
    R[:, 1, 2] = 2 * (y*z - x*w)

    R[:, 2, 0] = 2 * (x*z - y*w)
    R[:, 2, 1] = 2 * (y*z + x*w)
    R[:, 2, 2] = 1 - 2 * (x**2 + y**2)
    return R

def compute_walls(box_pos, box_rot, L, W, H, t):
    """Return world centers & sizes of the 4 walls for each env."""
    N = box_pos.shape[0]
    R = quat_to_rot_matrix(box_rot)  # (N, 3, 3)

    # local_centers = torch.tensor([
    #     [0,  W/2 - t/2, H/2],   # front
    #     [0, -W/2 + t/2, H/2],   # back
    #     [-L/2 + t/2, 0, H/2],   # left
    #     [ L/2 - t/2, 0, H/2],   # right
    # ], device=box_pos.device)  # (4, 3)

    # sizes = torch.tensor([
    #     [L, t, H],
    #     [L, t, H],
    #     [t, W, H],
    #     [t, W, H],
    # ], device=box_pos.device)  # (4, 3)

    # Adapted for asset where X=W, Y=H, Z=L
    local_centers = torch.tensor([
        [ 0.0,        H/2.0,  L/2.0 - t/2.0],   # front  (+Z)
        [ 0.0,        H/2.0, -L/2.0 + t/2.0],   # back   (-Z)
        [-W/2.0 + t/2.0, H/2.0,  0.0      ],   # left   (-X)
        [ W/2.0 - t/2.0, H/2.0,  0.0      ],   # right  (+X)
    ], device=box_pos.device)  # (4,3)

    # sizes: (size_x, size_y, size_z) per wall in local coords
    sizes = torch.tensor([
        [ W, H,  t],    # front  : spans width (X), height (Y), thin in Z
        [ W, H,  t],    # back
        [ t, H,  L],    # left   : thin in X, spans height and length
        [ t, H,  L],    # right
    ], device=box_pos.device)  # (4,3)

    # Broadcast local_centers across batch
    local_centers = local_centers.unsqueeze(0).expand(N, -1, -1)  # (N, 4, 3)
    sizes = sizes.unsqueeze(0).expand(N, -1, -1)                  # (N, 4, 3)

    # Transform to world
    world_centers = torch.bmm(local_centers, R.transpose(1, 2)) + box_pos.unsqueeze(1)  # (N, 4, 3)

    return world_centers, sizes, box_rot

def quat_to_z_axis(quat: torch.Tensor) -> torch.Tensor:
    """
    Convert quaternion (x, y, z, w) to the object's local z-axis (0, 0, 1) in world frame.
    quat: (..., 4) tensor
    returns: (..., 3) tensor
    """
    x, y, z, w = quat.unbind(-1)

    # Rotation matrix elements (row-major)
    R00 = 1 - 2 * (y*y + z*z)
    R01 = 2 * (x*y - z*w)
    R02 = 2 * (x*z + y*w)

    R10 = 2 * (x*y + z*w)
    R11 = 1 - 2 * (x*x + z*z)
    R12 = 2 * (y*z - x*w)

    R20 = 2 * (x*z - y*w)
    R21 = 2 * (y*z + x*w)
    R22 = 1 - 2 * (x*x + y*y)

    # Local z-axis is [0, 0, 1] → world = third column of R
    z_axis = torch.stack([R02, R12, R22], dim=-1)

    # Normalize just in case
    return normalize(z_axis, dim=-1)

def quat_to_yaw_cos_sin(quat: torch.Tensor) -> torch.Tensor:
    """
    Convert quaternion (x, y, z, w) into [cos(yaw), sin(yaw)].
    Assumes yaw is rotation around z-axis (object upright).
    quat: (..., 4) tensor
    returns: (..., 2) tensor
    """
    x, y, z, w = quat.unbind(-1)

    # rotation matrix element for yaw
    # yaw = atan2(2*(w*z + x*y), 1 - 2*(y*y + z*z))
    sin_yaw = 2 * (w * z + x * y)
    cos_yaw = 1 - 2 * (y * y + z * z)

    return torch.stack([cos_yaw, sin_yaw], dim=-1)

def quat_to_6d(quat: torch.Tensor) -> torch.Tensor:
    """
    Convert quaternion (x, y, z, w) into a 6D rotation representation
    (first two columns of the rotation matrix).
    
    Args:
        quat: (..., 4) tensor of quaternions
    Returns:
        (..., 6) tensor
    """
    x, y, z, w = quat.unbind(-1)

    # rotation matrix elements
    R = torch.stack([
        1 - 2 * (y*y + z*z),   2 * (x*y - z*w),     2 * (x*z + y*w),
        2 * (x*y + z*w),       1 - 2 * (x*x + z*z), 2 * (y*z - x*w),
        2 * (x*z - y*w),       2 * (y*z + x*w),     1 - 2 * (x*x + y*y)
    ], dim=-1).reshape(*quat.shape[:-1], 3, 3)

    # take first two columns (6D rep)
    r1 = normalize(R[..., 0], dim=-1)
    r2 = normalize(R[..., 1], dim=-1)

    return torch.cat([r1, r2], dim=-1)

# ### Observation functions

def object_position_in_robot_root_frame(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """The position of the object in the robot's root frame."""
    robot: RigidObject = env.scene[robot_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]
    
    # object_pos_w = object.data.root_pos_w[:, :3]
    # object_pos_b, _ = subtract_frame_transforms(
    #     robot.data.root_state_w[:, :3], robot.data.root_state_w[:, 3:7], object_pos_w
    # )
    # # print(object_pos_b)
    # return object_pos_b
    
    # World states
    robot_pos_w = robot.data.root_state_w[:, :3]
    robot_quat_w = robot.data.root_state_w[:, 3:7]
    object_pos_w = object.data.root_state_w[:, :3]
    object_quat_w = object.data.root_state_w[:, 3:7]

    # Transform object into robot frame
    object_pos_b, object_quat_b = subtract_frame_transforms(
        robot_pos_w, robot_quat_w, object_pos_w, object_quat_w
    )

    # Return [pos, quat]
    #z_axis_rotation = quat_to_z_axis(object_quat_b)  # (N, 3)
    #z_axis_rotation = quat_to_yaw_cos_sin(object_quat_b)  # (N, 2)
    quat_6d = quat_to_6d(object_quat_b)  # (N, 6)
    obj_pos_ori = torch.cat([object_pos_b, quat_6d], dim=-1)
    #print("obj pos ori: ", obj_pos_ori)
    return obj_pos_ori

def box_walls_positions_in_robot_frame(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("box"),
) -> torch.Tensor:
    """The positions of the 4 walls of the box in the robot's root frame."""
    robot: RigidObject = env.scene[robot_cfg.name]
    box: RigidObject = env.scene[object_cfg.name]

    box_pos = box.data.root_pos_w[:, :3]     # (N, 3)
    box_rot = box.data.root_quat_w[:, :4]    # (N, 4)

    # Box dimensions (fixed)
    #L, W, H, t = 0.48, 0.28, 0.15, 0.02
    L, W, H, t = 0.3974, 0.2968, 0.2574, 0.02

    # Get wall centers in world frame
    wall_centers, wall_sizes, _ = compute_walls(box_pos, box_rot, L, W, H, t)  # (N, 4, 3)

    # Flatten walls across envs: (N, 12)
    wall_centers = wall_centers.reshape(env.num_envs, -1)  

    # Robot frame (N, 3) -> expand to (N*4, 3)
    robot_pos = robot.data.root_state_w[:, :3].repeat_interleave(4, dim=0)
    robot_rot = robot.data.root_state_w[:, 3:7].repeat_interleave(4, dim=0)

    # Convert to robot frame
    wall_pos_b, _ = subtract_frame_transforms(
        robot_pos,    # robot positions (N, 3)
        robot_rot,   # robot quaternions (N, 4)
        wall_centers.view(env.num_envs, 4, 3).reshape(-1, 3),  # (N*4, 3)
    )

    #change box rotation to be in robot frame as well
    _, box_rot_b = subtract_frame_transforms(
        robot.data.root_state_w[:, :3],   # robot pos (N, 3)
        robot.data.root_state_w[:, 3:7],  # robot rot (N, 4)
        box.data.root_state_w[:, :3],     # box pos (dummy, not needed)
        box_rot,                          # box rot (N, 4)
    )

    wall_pos_b   = wall_pos_b.view(env.num_envs, -1)    # (N, 12)
    wall_sizes_b = wall_sizes.view(env.num_envs, -1)    # (N, 12)
    box_rot_b = box_rot_b.view(env.num_envs, -1)

    # final obs
    obs_walls = torch.cat([wall_pos_b, wall_sizes_b, box_rot_b], dim=-1)
 
    # wall_pos_b is now (N*4, 3). Reshape back to (N, 12).
    #print("box_pos: ", box_pos)
    #print("box_rot: ", box_rot)
    #print("wall_centers: ", wall_pos_b.view(env.num_envs, -1))
    return obs_walls
    #return torch.zeros(env.num_envs, wall_pos_b.numel() // env.num_envs, device=wall_pos_b.device)   # (N, 12)

def end_effector_pos_ori(
    env: ManagerBasedRLEnv,
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """The position and orientation (quaternion) of the end-effector in the world frame."""
    ee_frame: SceneEntityCfg = env.scene[ee_frame_cfg.name]

    ee_pos = ee_frame.data.target_pos_w[..., 0, :]      # (N, 3)
    ee_quat = ee_frame.data.target_quat_w[..., 0, :]     # (N, 4)

    #obs_ee = torch.cat([ee_pos, ee_quat], dim=-1)           # (N, 7)

    # Define finger offsets in gripper local frame (center points of fingers)
    finger_sep = 0.14 / 2.0
    finger_offsets = torch.tensor([
        [0.0, -finger_sep, -0.03],   # left finger center
        [0.0,  finger_sep, -0.03],   # right finger center
    ], device=ee_pos.device, dtype=ee_pos.dtype)  # (2, 3)

    # Convert quaternion to rotation matrix
    def quat_to_rotmat(q):
        # q: (..., 4) with (x, y, z, w)
        x, y, z, w = q.unbind(-1)
        B = q.shape[0]
        rot = torch.empty(B, 3, 3, device=q.device, dtype=q.dtype)
        rot[:, 0, 0] = 1 - 2 * (y*y + z*z)
        rot[:, 0, 1] = 2 * (x*y - z*w)
        rot[:, 0, 2] = 2 * (x*z + y*w)
        rot[:, 1, 0] = 2 * (x*y + z*w)
        rot[:, 1, 1] = 1 - 2 * (x*x + z*z)
        rot[:, 1, 2] = 2 * (y*z - x*w)
        rot[:, 2, 0] = 2 * (x*z - y*w)
        rot[:, 2, 1] = 2 * (y*z + x*w)
        rot[:, 2, 2] = 1 - 2 * (x*x + y*y)
        return rot

    rot = quat_to_rotmat(ee_quat)   # (N, 3, 3)

    # Transform local finger offsets into world space
    finger_pos = torch.einsum("bij,kj->bki", rot, finger_offsets) + ee_pos.unsqueeze(1)
    # finger_pos: (N, 2, 3)

    # Flatten finger positions into observation
    obs_fingers = finger_pos.reshape(ee_pos.shape[0], -1)  # (N, 6)

    # Concatenate: ee position + quat + fingers
    obs = torch.cat([ee_pos, ee_quat, obs_fingers], dim=-1)  # (N, 13)

    return obs



def get_object_sizes(env: ManagerBasedRLEnv, object_cfg: SceneEntityCfg) -> torch.Tensor:
    
    # return cached (and ensure it's on the correct device)
    if hasattr(env, "_cached_object_sizes"):
        sizes = env._cached_object_sizes
        if sizes.device != env.device:
            sizes = sizes.to(env.device)
            env._cached_object_sizes = sizes
        #print("using cached object sizes: ", sizes)
        return sizes
    
    stage = env.scene.stage
    bbox_cache = UsdGeom.BBoxCache(0, ["default"])
    sizes_list = []
    gripper_span = 0.14  # approx max span of gripper in meters

    for i in range(env.num_envs):
        prim_path = f"{env.scene.env_ns}/env_{i}/Object"   # same "Object" as in prim_path
        prim = stage.GetPrimAtPath(prim_path)
        bbox = bbox_cache.ComputeLocalBound(prim)
        min_pt, max_pt = bbox.GetRange().GetMin(), bbox.GetRange().GetMax()
        raw_sizes = [
            (max_pt[0] - min_pt[0])/gripper_span,   # normalize by gripper span
            (max_pt[1] - min_pt[1])/gripper_span,
            (max_pt[2] - min_pt[2])/gripper_span,
        ]

        # clamp each value into [0, 1]
        clamped_sizes = [min(1.0, s) for s in raw_sizes]

        sizes_list.append(clamped_sizes)
        #print(f"Object size (env {i}): raw={raw_sizes}, clamped={clamped_sizes}")

    #print("create cache of sizes ##############################")
    sizes = torch.tensor(sizes_list, device=env.device)
    env._cached_object_sizes = sizes
    return sizes