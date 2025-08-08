"""A collection of utility functions for fetchign/processing actions.
"""
import numpy as np
import torch

def get_robot_control_mode(robot_name, policy_name):
    """Overwrite simpler_env.utils.env.env_builder.get_robot_control_mode,
    since printing the control mode is annoying.
    """
    if "google_robot_static" in robot_name:
        control_mode = (
            "arm_pd_ee_delta_pose_align_interpolate_by_planner_gripper_pd_joint_target_delta_pos_interpolate_by_planner"
        )
    elif "widowx" in robot_name:
        control_mode = "arm_pd_ee_target_delta_pose_align2_gripper_pd_joint_pos"
        # control_mode = 'arm_pd_ee_delta_pose_align2_gripper_pd_joint_pos'
    else:
        raise NotImplementedError()
    return control_mode


def get_pose_world(env, cand_action):
    """A helper function that computes the predicted pose in world frame.

    Args:
        env: A gym environment.
        cand_action: A dictionary containing the candidate action.

    Returns:
        pose_world: A numpy array representing the pose in world frame.
    """
    controller = env.unwrapped.agent.controllers[
        get_robot_control_mode(env.unwrapped.robot_uid, "")
    ].controllers['arm']
    prev_ee_pose_at_base = controller._target_pose
    pose = controller.compute_target_pose(
        prev_ee_pose_at_base,
        np.concatenate([
            cand_action["world_vector"],
            cand_action["rot_axangle"],
            cand_action["gripper"]
        ])
    )
    pose_world = controller.articulation.pose * pose
    
    return pose_world

def get_pose_base(env, cand_action):
    """A helper function that computes the predicted pose in base frame.

    Args:
        env: A gym environment.
        cand_action: A dictionary containing the candidate action.

    Returns:
        pose_world: A numpy array representing the pose in world frame.
    """
    controller = env.unwrapped.agent.controllers[
        "arm_pd_ee_target_delta_pose_align2_gripper_pd_joint_pos"
    ].controllers['arm']

    prev_ee_pose_at_base = controller._target_pose

    pose = controller.compute_target_pose(
        prev_ee_pose_at_base,
        cand_action
    )

    return pose

