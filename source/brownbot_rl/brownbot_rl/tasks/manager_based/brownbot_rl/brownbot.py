# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the Universal Robots of brownbotics.

Reference: https://github.com/ros-industrial/universal_robot
"""

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR

##
# Configuration
# test private setting in github
##


BROWNBOT05_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"/isaac-sim/workspaces/isaac_sim_scene/brownbot_07.usd",
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=5.0,
            rigid_body_enabled=True,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
        ),  
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,  # Default to False, adjust if needed
            solver_position_iteration_count=12, # 8
            solver_velocity_iteration_count=4, # 0
        ),
        activate_contact_sensors=True,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        joint_pos={
            "shoulder_pan_joint": 0.0, #-1.312
            "shoulder_lift_joint": -1.712,
            "elbow_joint": 1.312,
            "wrist_1_joint": -1.712,
            "wrist_2_joint": -1.712,
            "wrist_3_joint": 0.0,
            "finger_joint": 0.0,
        },
    ),
    actuators={
        "arm_00": ImplicitActuatorCfg(
            joint_names_expr=["shoulder_pan_joint"],
            velocity_limit=0.02, #1.220
            effort_limit=150.0,
            stiffness=261.7,
            damping=26.17,
        ),
        "arm_01": ImplicitActuatorCfg(
            joint_names_expr=["shoulder_lift_joint"],
            velocity_limit=0.02,
            effort_limit=150.0,
            stiffness=261.0,
            damping=26.1,
        ),
        "arm_02": ImplicitActuatorCfg(
            joint_names_expr=["elbow_joint"],
            velocity_limit=0.02,
            effort_limit=150.0,
            stiffness=261.0,
            damping=26.0,
        ),
        "arm_03": ImplicitActuatorCfg(
            joint_names_expr=["wrist_1_joint"],
            velocity_limit=0.02,
            effort_limit=28.0,
            stiffness=261.0,
            damping=26.0,
        ),
        "arm_04": ImplicitActuatorCfg(
            joint_names_expr=["wrist_2_joint"],
            velocity_limit=0.02,
            effort_limit=28.0,
            stiffness=261.0,
            damping=26.0,
        ),
        "arm_05": ImplicitActuatorCfg(
            joint_names_expr=["wrist_3_joint"],
            velocity_limit=0.02,
            effort_limit=28.0,
            stiffness=261.0,
            damping=26.0,
        ),
        "arm_06": ImplicitActuatorCfg(
            joint_names_expr=["finger_joint"],
            velocity_limit=0.2, #1.220
            effort_limit=10.0, #10
            stiffness=30.125, #0.1125
            damping=0.3, #0.001
        ),
        "arm_07": ImplicitActuatorCfg(
            joint_names_expr=["left_outer_finger_joint"],
            velocity_limit=0.2, #1.220
            effort_limit=3.0, #1.0
            stiffness=0.5, #0.02
            damping=0.1, #0.001
        ),
        "arm_08": ImplicitActuatorCfg(
            joint_names_expr=["right_outer_finger_joint"],
            velocity_limit=0.2, #1.220
            effort_limit=3.0, #1.0
            stiffness=0.5, #0.02
            damping=0.1, #0.001
        ),
        "arm_09": ImplicitActuatorCfg(
            joint_names_expr=["left_inner_finger_joint"],
            velocity_limit=0.2, #1.220
            effort_limit=3.0, #1.0
            stiffness=0.5, #0.002
            damping=0.1, #0.0001
        ),
        "arm_10": ImplicitActuatorCfg(
            joint_names_expr=["right_inner_finger_joint"],
            velocity_limit=0.2, #1.220
            effort_limit=3.0, #1.0
            stiffness=0.5, #0.002
            damping=0.1, #0.0001
        ),
        "arm_12": ImplicitActuatorCfg(
            joint_names_expr=["left_inner_finger_pad_joint"],
            velocity_limit=0.2, #1.220
            effort_limit=3.0, #1.0
            stiffness=0.5, #0.002
            damping=0.1, #0.0001
        ),
        "arm_13": ImplicitActuatorCfg(
            joint_names_expr=["right_inner_finger_pad_joint"],
            velocity_limit=0.2, #1.220
            effort_limit=3.0, #1.0
            stiffness=0.5, #0.002
            damping=0.1, #0.0001
        ),
    },
    # Using default soft limits
    soft_joint_pos_limit_factor=1.0,
)
"""Configuration of UR-10 arm using implicit actuator models."""