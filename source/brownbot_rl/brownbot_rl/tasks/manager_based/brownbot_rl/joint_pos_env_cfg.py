# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.assets import RigidObjectCfg
from isaaclab.sensors import FrameTransformerCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg, ArticulationRootPropertiesCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from isaaclab_tasks.manager_based.manipulation.lift import mdp
from isaaclab_tasks.manager_based.manipulation.lift.lift_env_cfg import LiftEnvCfg

from isaaclab.sensors import ContactSensorCfg

##
# Pre-defined configs
##
from isaaclab.markers.config import FRAME_MARKER_CFG  # isort: skip

from .brownbot_rl_env_cfg import BrownbotRlEnvCfg
from .brownbot import BROWNBOT05_CFG

@configclass
class BrownbotCubeLiftEnvCfg(BrownbotRlEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # Set Brownbot as robot
        self.scene.robot = BROWNBOT05_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        # Set actions for the specific robot type (brownbot)
        self.actions.arm_action = mdp.JointPositionActionCfg(
            asset_name="robot", 
            joint_names=["shoulder_pan_joint",
                         "shoulder_lift_joint",
                         "elbow_joint",
                         "wrist_1_joint",
                         "wrist_2_joint",
                         "wrist_3_joint"], 
            scale=0.2, use_default_offset=True
        )
        self.actions.gripper_action = mdp.BinaryJointPositionActionCfg(
            asset_name="robot",
            joint_names=["finger_joint"],
            open_command_expr={"finger_joint": 0.0}, #0.0
            close_command_expr={"finger_joint": 0.57}, #0.53
        )
        # Set the body name for the end effector
        self.commands.object_pose.body_name = "robotiq_base_link"

        # Set Cube as object
        self.scene.object = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Object",
            init_state=RigidObjectCfg.InitialStateCfg(pos=[0.5, 0, 0.055], rot=[1, 0, 0, 0]),
            spawn=UsdFileCfg(
                usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/DexCube/dex_cube_instanceable.usd",
                scale=(0.8, 0.8, 0.8),
                rigid_props=RigidBodyPropertiesCfg(
                    solver_position_iteration_count=16,
                    solver_velocity_iteration_count=1,
                    max_angular_velocity=1000.0,
                    max_linear_velocity=1000.0,
                    max_depenetration_velocity=5.0,
                    disable_gravity=False,
                ),
            ),
        )

        # Set Box as object
        self.scene.box = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Box",
            init_state=RigidObjectCfg.InitialStateCfg(pos=[0.5, 0, 0.055], rot=[1, 0, 0, 0]),
            spawn=UsdFileCfg(
                usd_path=f"{ISAAC_NUCLEUS_DIR}/IsaacLab/Mimic/nut_pour_task/nut_pour_assets/sorting_bin_blue.usd",
                #usd_path=f"/isaac-sim/workspaces/isaac_sim_scene/bin_mod_4.usd",
                scale=(1.5, 2.0, 3.0),
                activate_contact_sensors=True,
                rigid_props=RigidBodyPropertiesCfg(
                    # disable_gravity=False,
                    # retain_accelerations=False,
                    # linear_damping=0.0,
                    # angular_damping=0.0,
                    # max_linear_velocity=1000.0,
                    # max_angular_velocity=1000.0,
                    # max_depenetration_velocity=1.0,
                    solver_position_iteration_count=4,
                    solver_velocity_iteration_count=0,
                    max_angular_velocity=1000.0,
                    max_linear_velocity=1000.0,
                    max_depenetration_velocity=5.0,
                    disable_gravity=False,
                ),
                # articulation_props=ArticulationRootPropertiesCfg(
                #     enabled_self_collisions=True,
                #     solver_position_iteration_count=4,
                #     solver_velocity_iteration_count=0,
                #     fix_root_link=True,
                # ),
            ),
        )

        # Listens to the required transforms
        marker_cfg = FRAME_MARKER_CFG.copy()
        marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
        marker_cfg.prim_path = "/Visuals/FrameTransformer"
        
        self.scene.object_frame = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Object",
            debug_vis=False,
            visualizer_cfg=marker_cfg,
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Object",
                    name="object",
                    offset=OffsetCfg(pos=[0.0, 0.0, 0.0]),
                ),
            ],
        )

        self.scene.ee_frame = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Robot/ur5/base_link",
            debug_vis=False,
            visualizer_cfg=marker_cfg,
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/ur5/Robotiq_2F_140_physics_edit/robotiq_base_link",
                    name="end_effector",
                    offset=OffsetCfg(
                        pos=[0.0, 0.0, 0.2134],
                    ),
                ),
            ],
        )

        # add contact sensor to the gripper of the robot
        self.scene.contact_forces_LF = ContactSensorCfg(
            prim_path="{ENV_REGEX_NS}/Robot/ur5/Robotiq_2F_140_physics_edit/left_inner_pad", #left_inner_pad
            update_period=0.0,
            history_length=6,
            debug_vis=False,
            filter_prim_paths_expr=["{ENV_REGEX_NS}/Object"],
        )

        self.scene.contact_forces_RF = ContactSensorCfg(
            prim_path="{ENV_REGEX_NS}/Robot/ur5/Robotiq_2F_140_physics_edit/right_inner_pad", #right_inner_pad
            update_period=0.0,
            history_length=6,
            debug_vis=False,
            filter_prim_paths_expr=["{ENV_REGEX_NS}/Object"],
        )

        # add contact sensors to all the robot links to detect collisions with obstacles
        # self.scene.contact_sensor_shoulder = ContactSensorCfg(
        #     prim_path="{ENV_REGEX_NS}/Robot/ur5/shoulder_link",
        #     update_period=0.0,
        #     history_length=6,
        #     debug_vis=False,
        #     filter_prim_paths_expr=["{ENV_REGEX_NS}/Box"]
        # )
        # self.scene.contact_sensor_upper_arm = ContactSensorCfg(
        #     prim_path="{ENV_REGEX_NS}/Robot/ur5/upper_arm_link",
        #     update_period=0.0,
        #     history_length=6,
        #     debug_vis=False,
        #     filter_prim_paths_expr=["{ENV_REGEX_NS}/Box"]
        # )
        # self.scene.contact_sensor_forearm = ContactSensorCfg(
        #     prim_path="{ENV_REGEX_NS}/Robot/ur5/forearm_link",
        #     update_period=0.0,
        #     history_length=6,
        #     debug_vis=False,
        #     filter_prim_paths_expr=["{ENV_REGEX_NS}/Box"]
        # )
        # self.scene.contact_sensor_wrist_1 = ContactSensorCfg(
        #     prim_path="{ENV_REGEX_NS}/Robot/ur5/wrist_1_link",
        #     update_period=0.0,
        #     history_length=6,
        #     debug_vis=False,
        #     filter_prim_paths_expr=["{ENV_REGEX_NS}/Box"]
        # )
        # self.scene.contact_sensor_wrist_2 = ContactSensorCfg(
        #     prim_path="{ENV_REGEX_NS}/Robot/ur5/wrist_2_link",
        #     update_period=0.0,
        #     history_length=6,
        #     debug_vis=False,
        #     filter_prim_paths_expr=["{ENV_REGEX_NS}/Box"]
        # )
        # self.scene.contact_sensor_wrist_3 = ContactSensorCfg(
        #     prim_path="{ENV_REGEX_NS}/Robot/ur5/wrist_3_link",
        #     update_period=0.0,
        #     history_length=6,
        #     debug_vis=False,
        #     filter_prim_paths_expr=["{ENV_REGEX_NS}/Box"]
        # )
        self.scene.contact_sensor_gripper_base_link = ContactSensorCfg(
            prim_path="{ENV_REGEX_NS}/Robot/ur5/Robotiq_2F_140_physics_edit/robotiq_base_link",
            update_period=0.0,
            history_length=6,
            debug_vis=False,
            # filter_prim_paths_expr=["{ENV_REGEX_NS}/Box"]
        )
        self.scene.contact_sensor_gripper_left_outer_finger = ContactSensorCfg(
            prim_path="{ENV_REGEX_NS}/Robot/ur5/Robotiq_2F_140_physics_edit/left_outer_finger",
            update_period=0.0,
            history_length=6,
            debug_vis=False,
            # filter_prim_paths_expr=["{ENV_REGEX_NS}/Box"]
        )
        self.scene.contact_sensor_gripper_left_inner_finger = ContactSensorCfg(
            prim_path="{ENV_REGEX_NS}/Robot/ur5/Robotiq_2F_140_physics_edit/left_inner_finger",
            update_period=0.0,
            history_length=6,
            debug_vis=False,
            # filter_prim_paths_expr=["{ENV_REGEX_NS}/Box"]
        )
        self.scene.contact_sensor_gripper_right_outer_finger = ContactSensorCfg(
            prim_path="{ENV_REGEX_NS}/Robot/ur5/Robotiq_2F_140_physics_edit/right_outer_finger",
            update_period=0.0,
            history_length=6,
            debug_vis=False,
            # filter_prim_paths_expr=["{ENV_REGEX_NS}/Box"]
        )
        self.scene.contact_sensor_gripper_right_inner_finger = ContactSensorCfg(
            prim_path="{ENV_REGEX_NS}/Robot/ur5/Robotiq_2F_140_physics_edit/right_inner_finger",
            update_period=0.0,
            history_length=6,
            debug_vis=False,
            # filter_prim_paths_expr=["{ENV_REGEX_NS}/Box"]
        )

@configclass
class BrownbotCubeLiftEnvCfg_PLAY(BrownbotCubeLiftEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # disable randomization for play
        self.observations.policy.enable_corruption = False
