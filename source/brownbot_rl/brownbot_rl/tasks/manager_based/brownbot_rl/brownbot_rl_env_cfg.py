# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import MISSING
import math

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, DeformableObjectCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass

from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import FrameTransformerCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg, UsdFileCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from isaaclab.sensors import ContactSensorCfg

from . import mdp

##
# Pre-defined configs
##

from isaaclab_assets.robots.cartpole import CARTPOLE_CFG  # isort:skip


##
# Scene definition
##


@configclass
class BrownbotRlSceneCfg(InteractiveSceneCfg):
    """Configuration for a cart-pole scene."""

    # robot
    robot: ArticulationCfg = MISSING
    # end-effector sensor: will be populated by agent env cfg
    ee_frame: FrameTransformerCfg = MISSING
    # target object: will be populated by agent env cfg
    object: RigidObjectCfg | DeformableObjectCfg = MISSING

    # Add contact sensors to the fingers of the gripper
    contact_forces_LF: ContactSensorCfg = MISSING
    contact_forces_RF: ContactSensorCfg = MISSING

    # Table
    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0.5, 0, 0], rot=[0.707, 0, 0, 0.707]),
        spawn=UsdFileCfg(usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/SeattleLabTable/table_instanceable.usd"),
    )

    # plane
    plane = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0, 0, -1.05]),
        spawn=GroundPlaneCfg(),
    )

    # lights
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )


##
# MDP settings
##

@configclass
class CommandsCfg:
    """Command terms for the MDP."""

    object_pose = mdp.UniformPoseCommandCfg(
        asset_name="robot",
        body_name=MISSING,  # will be set by agent env cfg
        resampling_time_range=(5.0, 5.0),
        debug_vis=True,
        ranges=mdp.UniformPoseCommandCfg.Ranges(
            pos_x=(0.4, 0.6), pos_y=(-0.25, 0.25), pos_z=(0.25, 0.5), roll=(1.57, 1.57), pitch=(-1.57, -1.57), yaw=(1.57,1.57)
        ),
    )

@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    # will be set by agent env cfg
    arm_action: mdp.JointPositionActionCfg | mdp.DifferentialInverseKinematicsActionCfg = MISSING
    gripper_action: mdp.BinaryJointPositionActionCfg = MISSING


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        object_position = ObsTerm(func=mdp.object_position_in_robot_root_frame)
        target_object_position = ObsTerm(func=mdp.generated_commands, params={"command_name": "object_pose"})
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """Configuration for events."""

    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")

    reset_object_position = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.1, 0.1), "y": (-0.25, 0.25), "z": (0.0, 0.0)},
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("object", body_names="Object"),
        },
    )


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    reaching_object = RewTerm(func=mdp.object_ee_distance, params={"std": 1.0}, weight=3.0)

    # Penalize closing when far from the object
    penalize_closing_far = RewTerm(
        func=mdp.penalize_closing_when_far,
        params={"min_distance": 0.09, "gripper_action_name": "gripper_action"},
        weight=7.0,  # this weight will multiply the internal -1.0 returned by the function
    )

    # Penalize opening when near from the object
    # penalize_opening_near = RewTerm(
    #     func=mdp.penalize_opening_when_near,
    #     params={"min_distance": 0.056, "gripper_action_name": "gripper_action"},
    #     weight=7.0,  # this weight will multiply the internal -1.0 returned by the function
    # )

    # penalize action rate only in the gripper
    penalty_action_rate_gripper = RewTerm(
        func=mdp.gripper_action_rate_l2,
        weight=-9.0
    )

    # Reward for closing the gripper near the object
    reward_closing_near = RewTerm(
        func=mdp.reward_closing_when_near,
        params={"min_distance": 0.056, "gripper_action_name": "gripper_action"},
        weight=10.0,
    )

    # Reward for having contact with the object and the gripper
    double_contact_reward = RewTerm(
        func=mdp.reward_double_contact_on_grasp,
        params={
            "contact_threshold": 0.1,
            "left_sensor_name": "contact_forces_LF",
            "right_sensor_name": "contact_forces_RF",
        },
        weight=10.0  # adjust this weight based on reward scaling
    )

    being_far_penalty = RewTerm(func=mdp.penalty_for_being_far,
                                params={"threshold": 0.29},
                                weight=7.0)  # Weight is applied inside the function

    lifting_object = RewTerm(func=mdp.object_is_lifted, 
                             params={"minimal_height": 0.04}, 
                             weight=10.0)

    # object_goal_tracking = RewTerm(
    #     func=mdp.object_goal_distance,
    #     params={"std": 0.3, "minimal_height": 0.04, "command_name": "object_pose"},
    #     weight=30.0,
    # )

    # object_goal_tracking_fine_grained = RewTerm(
    #     func=mdp.object_goal_distance,
    #     params={"std": 0.05, "minimal_height": 0.04, "command_name": "object_pose"},
    #     weight=30.0,
    # )

    object_goal_smooth = RewTerm(
        func=mdp.object_goal_distance_smooth,
        params={"std":0.2, "command_name": "object_pose", "minimal_height": 0.04},
        weight=10.0
    )

    # action penalty
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-1e-1) #-1e-1

    joint_vel = RewTerm(
        func=mdp.joint_vel_l2,
        weight=-1e-1,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    object_dropping = DoneTerm(
        func=mdp.root_height_below_minimum, params={"minimum_height": -0.05, "asset_cfg": SceneEntityCfg("object")}
    )

    excessive_velocity = DoneTerm(
        func=mdp.joint_velocity_exceeded, params={"velocity_threshold": 120.0, "asset_cfg": SceneEntityCfg("robot")}
    )

@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""

    # action_rate = CurrTerm(
    #     func=mdp.modify_reward_weight, params={"term_name": "action_rate", "weight": -1e-1, "num_steps": 10000}
    # )

    # joint_vel = CurrTerm(
    #     func=mdp.modify_reward_weight, params={"term_name": "joint_vel", "weight": -1e-1, "num_steps": 10000}
    # )

    reward_closing_near = CurrTerm(
        func=mdp.modify_reward_weight,
        params={ "term_name": "reward_closing_near", "weight": 10.0, "num_steps": 20000},
    )


##
# Environment configuration
##


@configclass
class BrownbotRlEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the lifting environment."""

    # Scene settings
    # scene: ObjectTableSceneCfg = ObjectTableSceneCfg(num_envs=4096, env_spacing=2.5)
    scene: BrownbotRlSceneCfg = BrownbotRlSceneCfg(num_envs=2, env_spacing=2.5)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    # curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.decimation = 2
        self.episode_length_s = 5.0
        # simulation settings
        self.sim.dt = 0.01  # 100Hz  0.01
        self.sim.render_interval = self.decimation

        #self.sim.physx.bounce_threshold_velocity = 0.2
        self.sim.physx.bounce_threshold_velocity = 0.01
        self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 1024 * 1024 * 4
        self.sim.physx.gpu_total_aggregate_pairs_capacity = 16 * 1024
        self.sim.physx.friction_correlation_distance = 0.00625

        # added for the contact sensors to handle 4096 environments
        # the error recommends at least 170648
        self.sim.physx.gpu_max_rigid_patch_count = 200000