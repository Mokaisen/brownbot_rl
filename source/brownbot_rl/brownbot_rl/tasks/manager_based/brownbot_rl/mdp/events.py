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

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

def reset_object_2_based_on_object(env: ManagerBasedRLEnv,
                                   env_ids: torch.Tensor, 
                                   offset: tuple[float, float, float] = (0.0, 0.0, -0.05),
                                   object_to_spawn: SceneEntityCfg = SceneEntityCfg("box"),
                                   object_to_follow: SceneEntityCfg = SceneEntityCfg("object")):
    """Reset the position of object_2 based on the position of object with an offset."""    
    # grab assets
    asset_spawn = env.scene[object_to_spawn.name]
    asset_follow = env.scene[object_to_follow.name]

    # get positions and orientations of the object to follow
    obj_pos = asset_follow.data.root_pos_w[env_ids]
    obj_quat = asset_follow.data.root_quat_w[env_ids]

    # apply offset
    offset_tensor = torch.tensor(offset, device=obj_pos.device)
    new_pos = obj_pos + offset_tensor

    # zero velocities
    obj_lin_vel = torch.zeros_like(obj_pos)
    obj_ang_vel = torch.zeros_like(obj_pos)

    # build pose and velocity tensors
    new_pose = torch.cat([new_pos, obj_quat], dim=-1)              # (N, 7)
    new_vel = torch.cat([obj_lin_vel, obj_ang_vel], dim=-1)        # (N, 6)

    # write into sim
    asset_spawn.write_root_pose_to_sim(new_pose, env_ids=env_ids)
    asset_spawn.write_root_velocity_to_sim(new_vel, env_ids=env_ids)
