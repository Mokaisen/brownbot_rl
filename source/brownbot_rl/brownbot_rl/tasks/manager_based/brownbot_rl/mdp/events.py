# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Common functions that can be used to activate certain events for the lift task.

The functions can be passed to the :class:`omni.isaac.lab.managers.EventTermCfg` object to enable
the event introduced by the function.
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import quat_mul, quat_from_euler_xyz

from pxr import UsdGeom

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

def quat_inv(q):
    # q: (...,4) in (x,y,z,w)
    return torch.cat([-q[..., :3], q[..., 3:].clone()], dim=-1)

def normalize(q):
    return q / q.norm(dim=-1, keepdim=True)

def quat_to_euler_deg(q):
    # q: (...,4) in (x,y,z,w)
    x, y, z, w = q.unbind(-1)
    t0 = 2.0 * (w * x + y * z)
    t1 = 1.0 - 2.0 * (x * x + y * y)
    roll = torch.atan2(t0, t1)

    t2 = 2.0 * (w * y - z * x)
    t2 = torch.clamp(t2, -1.0, 1.0)
    pitch = torch.asin(t2)

    t3 = 2.0 * (w * z + x * y)
    t4 = 1.0 - 2.0 * (y * y + z * z)
    yaw = torch.atan2(t3, t4)

    return torch.stack([roll, pitch, yaw], dim=-1) * (180.0 / torch.pi)

def reset_object_2_based_on_object(env: ManagerBasedRLEnv,
                                   env_ids: torch.Tensor, 
                                   offset: tuple[float, float, float] = (0.0, 0.0, -0.02),
                                   object_to_spawn: SceneEntityCfg = SceneEntityCfg("box"),
                                   object_to_follow: SceneEntityCfg = SceneEntityCfg("object"),
                                   rotation_offset: tuple[float, float, float] = (1.57, 0.0, 0.0)):
    """Reset the position of object_2 based on the position of object with an offset."""    
    # grab assets
    asset_spawn = env.scene[object_to_spawn.name]
    asset_follow = env.scene[object_to_follow.name]

    # get positions and orientations of the object to follow
    obj_pos = asset_follow.data.root_pos_w[env_ids]
    obj_quat = asset_follow.data.root_quat_w[env_ids]

    #original spawn orientation
    asset_spawn_quat = asset_spawn.data.root_quat_w[env_ids]

    # # obj_quat coming from the object you follow (N,4)
    # print("obj_quat[0] (x,y,z,w):", obj_quat[0].cpu().numpy())
    # # What the spawn asset currently reports (its current root rotation)
    # print("asset_spawn root_quat_w before write (x,y,z,w):", asset_spawn.data.root_quat_w[env_ids[0]].cpu().numpy())

    # print("obj_quat euler deg:", quat_to_euler_deg(obj_quat)[0].cpu().numpy())
    # print("asset_spawn euler deg:", quat_to_euler_deg(asset_spawn.data.root_quat_w[env_ids])[0].cpu().numpy())
    

    # apply offset
    offset_tensor = torch.tensor(offset, device=obj_pos.device)
    new_pos = obj_pos + offset_tensor

    # create correction quaternion from Euler (in radians)
    rot_offset_quat = quat_from_euler_xyz(
        torch.tensor(rotation_offset[0], device=obj_pos.device),
        torch.tensor(rotation_offset[1], device=obj_pos.device),
        torch.tensor(rotation_offset[2], device=obj_pos.device),
    ) # shape (4,)

    rot_offset_quat = rot_offset_quat.repeat(obj_quat.shape[0], 1)  # broadcast to (N, 4)

    # apply correction quaternion
    new_quat = quat_mul(obj_quat, rot_offset_quat)

    # zero velocities
    obj_lin_vel = torch.zeros_like(obj_pos)
    obj_ang_vel = torch.zeros_like(obj_pos)

    # build pose and velocity tensors
    new_pose = torch.cat([new_pos, asset_spawn_quat], dim=-1)              # (N, 7)
    new_vel = torch.cat([obj_lin_vel, obj_ang_vel], dim=-1)        # (N, 6)

    # write into sim
    asset_spawn.write_root_pose_to_sim(new_pose, env_ids=env_ids)
    asset_spawn.write_root_velocity_to_sim(new_vel, env_ids=env_ids)

    #print("obj_pos: ", obj_pos[0].cpu().numpy())
    #print("obj_pos event: ", obj_pos)
    #print("box pos event: ", new_pose)
    #print("asset_spawn pos: ", asset_spawn.data.root_pos_w[env_ids[0]].cpu().numpy())

def cache_object_sizes(env: ManagerBasedRLEnv, env_ids: torch.Tensor):
    """Cache per-env object sizes at reset."""

    # return cached (and ensure it's on the correct device)
    if hasattr(env, "_cached_object_sizes"):
        return 

    stage = env.scene.stage
    bbox_cache = UsdGeom.BBoxCache(0, ["default"])
    sizes = []
    gripper_span = 0.14  # approx max span of gripper in meters

    for i in range(env.num_envs):
        prim_path = f"{env.scene.env_ns}/env_{i}/Object"   # must match your prim_path
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

        sizes.append(clamped_sizes)
        #print(f"Object size (env {i}): clamped={clamped_sizes}")

    env._cached_object_sizes = torch.tensor(sizes, device=env.device)