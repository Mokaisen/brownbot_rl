# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Common functions that can be used to activate certain terminations for the lift task.

The functions can be passed to the :class:`omni.isaac.lab.managers.TerminationTermCfg` object to enable
the termination introduced by the function.
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import combine_frame_transforms
from isaaclab.sensors import ContactSensor

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def object_reached_goal(
    env: ManagerBasedRLEnv,
    command_name: str = "object_pose",
    threshold: float = 0.02,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Termination condition for the object reaching the goal position.

    Args:
        env: The environment.
        command_name: The name of the command that is used to control the object.
        threshold: The threshold for the object to reach the goal position. Defaults to 0.02.
        robot_cfg: The robot configuration. Defaults to SceneEntityCfg("robot").
        object_cfg: The object configuration. Defaults to SceneEntityCfg("object").

    """
    # extract the used quantities (to enable type-hinting)
    robot: RigidObject = env.scene[robot_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]
    command = env.command_manager.get_command(command_name)
    # compute the desired position in the world frame
    des_pos_b = command[:, :3]
    des_pos_w, _ = combine_frame_transforms(robot.data.root_state_w[:, :3], robot.data.root_state_w[:, 3:7], des_pos_b)
    # distance of the end-effector to the object: (num_envs,)
    distance = torch.norm(des_pos_w - object.data.root_pos_w[:, :3], dim=1)

    # rewarded if the object is lifted above the threshold
    return distance < threshold

def joint_velocity_exceeded(env: ManagerBasedRLEnv, velocity_threshold: float, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Terminate when any joint velocity exceeds the threshold."""
    robot: RigidObject = env.scene[asset_cfg.name]
    joint_velocities = robot.data.joint_vel # Shape: (num_envs, num_joints)
    
    #print(joint_velocities)

    mask = torch.any(torch.abs(joint_velocities) > velocity_threshold, dim=1)

    # if mask.any():
    #     print("Velocity termination: ", joint_velocities[mask])

    return mask

def action_rate_exceeded(env: ManagerBasedRLEnv, action_threshold: float) -> torch.Tensor:
    """Terminate when any joint velocity exceeds the threshold."""
    #robot: RigidObject = env.scene[asset_cfg.name]
    action = env.action_manager.action # Shape: (num_envs, num_joints)
    prev_action = env.action_manager.prev_action
    action_rate = torch.abs(action - prev_action)

    mask = torch.any(action_rate > action_threshold, dim=1)  # shape (num_envs,)

    # if mask.any():
    #     print("action rate termination: ", action_rate[mask])

    return mask

def terminate_on_robot_box_collision(
    env: ManagerBasedRLEnv,
    sensor_gripper_1: str = "contact_sensor_gripper_base_link",
    sensor_gripper_2: str = "contact_sensor_gripper_left_outer_finger",
    sensor_gripper_3: str = "contact_sensor_gripper_left_inner_finger",
    sensor_gripper_4: str = "contact_sensor_gripper_right_outer_finger",
    sensor_gripper_5: str = "contact_sensor_gripper_right_inner_finger",
    force_threshold: float = 1.0,
) -> torch.Tensor:
    """Terminate the episode if the robot collides with the box (force above threshold)."""

    sensor_1: ContactSensor = env.scene[sensor_gripper_1]
    sensor_2: ContactSensor = env.scene[sensor_gripper_2]
    sensor_3: ContactSensor = env.scene[sensor_gripper_3]
    sensor_4: ContactSensor = env.scene[sensor_gripper_4]
    sensor_5: ContactSensor = env.scene[sensor_gripper_5]

    # each is (num_envs, 3)
    force_1 = sensor_1.data.net_forces_w.squeeze()
    force_2 = sensor_2.data.net_forces_w.squeeze()
    force_3 = sensor_3.data.net_forces_w.squeeze()
    force_4 = sensor_4.data.net_forces_w.squeeze()
    force_5 = sensor_5.data.net_forces_w.squeeze()

    # norms (num_envs,)
    force_1_norm = torch.norm(force_1, dim=-1)
    force_2_norm = torch.norm(force_2, dim=-1)
    force_3_norm = torch.norm(force_3, dim=-1)
    force_4_norm = torch.norm(force_4, dim=-1)
    force_5_norm = torch.norm(force_5, dim=-1)

    total_force = (force_1_norm + force_2_norm + force_3_norm + force_4_norm + force_5_norm) / 100.0

    # termination mask: True where force exceeds threshold
    terminated = total_force > force_threshold

    return terminated