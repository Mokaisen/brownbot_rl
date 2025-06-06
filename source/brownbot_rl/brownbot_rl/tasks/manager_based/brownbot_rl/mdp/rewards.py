# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import RigidObject
from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import wrap_to_pi
from isaaclab.sensors import FrameTransformer
from isaaclab.utils.math import combine_frame_transforms
from isaaclab.sensors import ContactSensor

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def object_is_lifted(
    env: ManagerBasedRLEnv, minimal_height: float, object_cfg: SceneEntityCfg = SceneEntityCfg("object")
) -> torch.Tensor:
    """Reward the agent for lifting the object above the minimal height."""
    object: RigidObject = env.scene[object_cfg.name]
    return torch.where(object.data.root_pos_w[:, 2] > minimal_height, 1.0, 0.0)


def object_ee_distance(
    env: ManagerBasedRLEnv,
    std: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Reward the agent for reaching the object using tanh-kernel."""
    # extract the used quantities (to enable type-hinting)
    object: RigidObject = env.scene[object_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    # Target object position: (num_envs, 3)
    cube_pos_w = object.data.root_pos_w
    # End-effector position: (num_envs, 3)
    ee_w = ee_frame.data.target_pos_w[..., 0, :]
    # Distance of the end-effector to the object: (num_envs,)
    object_ee_distance = torch.norm(cube_pos_w - ee_w, dim=1)
    # print(f"################### Object ee distance {object_ee_distance}:")

    # End-effector velocity (num_envs, 3)
    # ee_velocity = ee_frame.data.target_lin_vel_w[..., 0, :]
    # ee_speed = torch.norm(ee_velocity, dim=1)  # Speed scalar (num_envs,)

    # based reward with smooth decay
    # reward_distance = torch.exp(- (object_ee_distance / std) )
    reward_distance = torch.exp(-(object_ee_distance / 0.2))
    # print(f"object ee distance: {object_ee_distance}")

    reward_distance += (object_ee_distance < 0.27) * 1.5
    reward_distance += (object_ee_distance < 0.15) * 3.0
    reward_distance += (object_ee_distance < 0.08) * 5.0
    reward_distance += (object_ee_distance < 0.02) * 8.0
    #print(f"REWARD ee distance: {reward_distance}")

    # Velocity penalty when close to the object
    # velocity_penalty = (object_ee_distance < 0.40) * (ee_speed * -0.5)
    # reward_distance += velocity_penalty

    return reward_distance 

def object_goal_distance(
    env: ManagerBasedRLEnv,
    std: float,
    minimal_height: float,
    command_name: str,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Reward the agent for tracking the goal pose using tanh-kernel."""
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
    return (object.data.root_pos_w[:, 2] > minimal_height) * (1 - torch.tanh(distance / std))

def object_goal_distance_smooth(
    env: ManagerBasedRLEnv,
    std: float,
    command_name: str,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Reward the agent based on how close the object is to the goal position using smooth exponential + bonus thresholds."""
    object: RigidObject = env.scene[object_cfg.name]
    command = env.command_manager.get_command(command_name)
    robot: RigidObject = env.scene[robot_cfg.name]

    # Final desired goal position in world frame (x, y, z)
    goal_pos_b = command[:, :3]  # shape: (num_envs, 3)
    goal_pos_w, _ = combine_frame_transforms(
        robot.data.root_state_w[:, :3],  # robot position
        robot.data.root_state_w[:, 3:7],  # robot orientation (quaternion)
        goal_pos_b
    )

    # Current object position in world frame
    object_pos_w = object.data.root_pos_w[:, :3]

    # Euclidean distance from object to goal: (num_envs,)
    object_goal_dist = torch.norm(object_pos_w - goal_pos_w, dim=1)
    #print(f"object goal distance: {object_goal_dist}")

    # Smooth reward based on distance
    reward = torch.exp(-(object_goal_dist / std))
    #print(f"reward: {reward}")

    # Add bonus reward tiers for being very close to the goal
    reward += (object_goal_dist < 0.25) * 1.0
    reward += (object_goal_dist < 0.15) * 2.0
    reward += (object_goal_dist < 0.08) * 4.0
    reward += (object_goal_dist < 0.02) * 6.0
    #print(f"reward weighted: {reward}")

    return reward

def penalize_closing_when_far(
    env: ManagerBasedRLEnv,
    min_distance: float = 0.10,  # threshold distance to consider "near"
    gripper_action_name: str = "gripper_action",
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Penalize the agent for closing the gripper when far from the object."""
    # get object and EE positions
    object: RigidObject = env.scene[object_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    cube_pos = object.data.root_pos_w  # (num_envs, 3)
    ee_pos = ee_frame.data.target_pos_w[..., 0, :]  # (num_envs, 3)

    # compute distance
    distance = torch.norm(cube_pos - ee_pos, dim=1)  # (num_envs,)

    # get the class of gripper action from the environment
    #gripper_term = env.action_manager.get_term(gripper_action_name)
    #print(type(gripper_term))

    # get gripper action (assume binary action in shape (num_envs, 1))
    gripper_action = env.action_manager.get_term(gripper_action_name).processed_actions.squeeze(-1)
    #print("gripper action: ", gripper_action)

    # Define threshold for "closing" — adjust if needed
    is_closing = gripper_action > 0.5
    is_far = distance > min_distance

    # Penalize if the agent is trying to close the gripper while far from the object
    penalty = is_closing & is_far  # boolean mask
    
    penalty_res = penalty.float() * -1.0  # Apply -1.0 penalty where condition is met
    #print("penalty_res: ", penalty_res)

    return penalty_res

def penalize_opening_when_near(
    env: ManagerBasedRLEnv,
    min_distance: float = 0.05,  # threshold distance to consider "near"
    gripper_action_name: str = "gripper_action",
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Penalize the agent for opening the gripper when near from the object."""
    # get object and EE positions
    object: RigidObject = env.scene[object_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    cube_pos = object.data.root_pos_w  # (num_envs, 3)
    ee_pos = ee_frame.data.target_pos_w[..., 0, :]  # (num_envs, 3)

    # compute distance
    distance = torch.norm(cube_pos - ee_pos, dim=1)  # (num_envs,)
    is_near = distance < min_distance  # (num_envs,)
    # get the class of gripper action from the environment
    #gripper_term = env.action_manager.get_term(gripper_action_name)
    #print(type(gripper_term))

    # get gripper action (assume binary action in shape (num_envs, 1))
    gripper_action = env.action_manager.get_term(gripper_action_name).processed_actions.squeeze(-1)
    #print("gripper action: ", gripper_action)

    # previous action 
    total_prev_action = env.action_manager.prev_action
    gripper_action_prev = total_prev_action[:,-1]

    # detect opening transition 
    was_closed = gripper_action_prev < 0
    is_opening = gripper_action < 0.5
    opening_transition = was_closed & is_opening

    # Apply penalty only when near the object
    penalty_mask = opening_transition & is_near
    penalty = penalty_mask.float() * -1.0

    # print("gripper_action_prev:", gripper_action_prev)
    # print("gripper_action:", gripper_action)
    # print("penalty:", penalty)
    # print("###########################")

    # # Define threshold for "closing" — adjust if needed
    # is_opening = gripper_action < 0.5
    # is_far = distance < min_distance

    # # Penalize if the agent is trying to close the gripper while far from the object
    # penalty = is_opening & is_far  # boolean mask

    # penalty_res = penalty.float() * -1.0  # Apply -1.0 penalty where condition is met
    # #print("penalty_res: ", penalty_res)

    return penalty

def reward_closing_when_near(
    env: ManagerBasedRLEnv,
    min_distance: float = 0.05,
    gripper_action_name: str = "gripper_action",
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    object = env.scene[object_cfg.name]
    ee = env.scene[ee_frame_cfg.name]

    distance = torch.norm(object.data.root_pos_w - ee.data.target_pos_w[..., 0, :], dim=1)
    gripper_action = env.action_manager.get_term(gripper_action_name).processed_actions.squeeze(-1)

    is_closing = gripper_action > 0.5
    is_near = distance < min_distance

    reward = is_closing & is_near
    return reward.float() * 1.0

def penalty_for_being_far(
    env: ManagerBasedRLEnv,
    threshold: float = 1.0,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Penalize the agent if it is too far from the object."""
    object = env.scene[object_cfg.name]
    ee = env.scene[ee_frame_cfg.name]
    dists = torch.norm(object.data.root_pos_w - ee.data.target_pos_w[..., 0, :], dim=1)
    
    return -torch.clamp((dists - threshold), min=0.0)

def reward_double_contact_on_grasp(
    env: ManagerBasedRLEnv,
    contact_threshold: float = 0.1,
    left_sensor_name: str = "contact_forces_LF",
    right_sensor_name: str = "contact_forces_RF",
) -> torch.Tensor:
    """Reward 1.0 for single finger contact, 2.0 for both fingers in contact with the object."""
    left_contact: ContactSensor = env.scene[left_sensor_name]
    right_contact: ContactSensor = env.scene[right_sensor_name]

    left_force = left_contact.data.force_matrix_w.squeeze()   # (num_envs,)
    right_force = right_contact.data.force_matrix_w.squeeze()   # (num_envs,)
    left_force_norm = torch.norm(left_force, dim=-1)
    right_force_norm = torch.norm(right_force, dim=-1)

    #print("left_force_norm: ", left_force_norm)
    #print("right_force_norm: ", right_force_norm)

    # # Compute contact status per finger (True if force norm > threshold)
    # left_has_contact = left_force_norm > contact_threshold  # shape: [num_envs]
    # right_has_contact = right_force_norm > contact_threshold

    #print("left_has_contact: ", left_has_contact)
    #print("right_has_contact: ", right_has_contact)

    # both_contact = left_has_contact & right_has_contact
    # single_contact = (left_has_contact ^ right_has_contact)  # XOR: only one finger has contact

    # reward = torch.zeros_like(left_has_contact, dtype=torch.float32)
    # reward[single_contact] = 1.0
    # reward[both_contact] = 2.0
    #reward = reward.unsqueeze(-1)

    #print("reward shape: ", reward.shape)
    #print("reward: ", reward)

    # reward = left_force_norm + right_force_norm
    reward = torch.clamp(left_force_norm + right_force_norm, max=3.0)
    #print("reward: ", reward)

    return reward  # (num_envs,)

def gripper_action_rate_l2(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Penalize the rate of change of the gripper action using L2 squared kernel."""
    # Assuming gripper action is last dimension
    gripper_curr = env.action_manager.action[..., -1]
    gripper_prev = env.action_manager.prev_action[..., -1]
    reward = torch.square(gripper_curr - gripper_prev)
    #print(f"penalty action rate: {reward}")
    return reward