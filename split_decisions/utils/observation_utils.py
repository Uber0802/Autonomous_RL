"""A collection of utility functions for fetchign/processing observations.
"""
from collections import OrderedDict
from copy import deepcopy

import numpy as np
from gymnasium import spaces

from mani_skill2_real2sim.utils.wrappers.observation import (
    BaseGymObservationWrapper,
    merge_dict_spaces
)


def get_default_obs_camera_name(env):
    # TODO: should not assume env.unwrapped.robot_uid
    # obtain image from observation dictionary returned by ManiSkill2 environment
    if "google_robot" in env.unwrapped.robot_uid:
        camera_name = "overhead_camera"
    elif "widowx" in env.unwrapped.robot_uid:
        camera_name = "3rd_view_camera"
    else:
        raise NotImplementedError()
    
    return camera_name


def world_to_screen(env, obs_camera_name, positions):
    """Convert 3D positions in world frame to 2D positions in screen frame.

    Args:
        env: Gym environment object.
        obs_camera_name: A string indicates the camera name
        positions: A np array of shape [N, 3/4], indicating 3D positions in
            world frame, where sapien uses OpenGL world frame.
    """    
    if positions.shape[-1] == 3:
        positions = np.concatenate(
            [positions, np.ones_like(positions[..., :1])], axis=-1
        )
    camera_params = env.unwrapped._cameras[obs_camera_name].get_params()
    cam2world = camera_params["cam2world_gl"]

    camera_3d = positions @ (np.linalg.inv(cam2world)).T
    camera_3d = camera_3d[..., :3] / camera_3d[..., 3]
    pixel_2d = camera_to_screen(
        env, obs_camera_name, camera_3d.reshape(1, 3))
    
    return pixel_2d


def camera_to_screen(env, obs_camera_name, positions):
    """Convert 3D positions in camera frame to 2D positions in screen frame.

    Args:
        env: Gym environment object.
        obs_camera_name: A string indicates the camera name
        positions: A np array of shape [N, 3], indicating 3D positions in
            camera frame, where sapien uses OpenGL camera frame.
    """
    camera = env.unwrapped._cameras[obs_camera_name].camera

    # Sapien use OpenGL projection matrix
    cx, cy = camera.cx, camera.cy
    w, h = camera.width, camera.height
    near, far = camera.near, camera.far
    fx, fy = camera.fx, camera.fy
    opengl_mtx = np.array([
        [2 * fx / w, 0.0, (w - 2 * cx) / w, 0.0],
        [0.0, -2 * fy / h, (h - 2 * cy) / h, 0.0],
        [0.0, 0.0, (-far - near) / (far - near), -2.0 * far * near / (far - near)],
        [0.0, 0.0, -1.0, 0.0]
    ])

    # Convert 3D positions to 2D positions
    positions = np.concatenate(
        [positions, np.ones((positions.shape[0], 1))], axis=1)

    clip_points = positions @ opengl_mtx.T
    ndc_points = clip_points / clip_points[:, 3]

    viewport_points = (
        (ndc_points + 1.0) / 2.0 * np.array([w, h, 1.0, 1.0]).reshape(1, 4)
    )

    return viewport_points[:, :2]


class PointCloudObservationWrapper(BaseGymObservationWrapper):
    """ Turn camera frame's 2D position information to world frame in 'xyzw'. """
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = deepcopy(env.observation_space)
        self.update_observation_space(self.observation_space)
        self._buffer = {}

    @staticmethod
    def update_observation_space(space: spaces.Dict):
        image_space: spaces.Dict = space.spaces.pop("image")
        space.spaces.pop("camera_param")
        pcd_space = OrderedDict()
        for cam_uid in image_space:
            cam_image_space = image_space[cam_uid]
            cam_pcd_space = OrderedDict()
            h, w = cam_image_space["Position"].shape[:2]
            cam_pcd_space["xyzw"] = spaces.Box(
                low=-np.inf, high=np.inf, shape=(h * w, 4), dtype=np.float32
            )
            if "Color" in cam_image_space.spaces:
                cam_pcd_space["rgb"] = spaces.Box(
                    low=0, high=255, shape=(h * w, 3), dtype=np.uint8
                )
            if "Segmentation" in cam_image_space.spaces:
                cam_pcd_space["Segmentation"] = spaces.Box(
                    low=0, high=(2 ** 32 - 1), shape=(h * w, 4), dtype=np.uint32
                )
            pcd_space[cam_uid] = spaces.Dict(cam_pcd_space)
        pcd_space = merge_dict_spaces(pcd_space.values())
        space.spaces["pointcloud"] = pcd_space

    def observation(self, observation: dict):
        image_obs = observation["image"]
        camera_params = observation["camera_param"]
        for cam_uid, images in image_obs.items():
            position = images["Position"]
            rgb = images["Color"]
            rgb_h, rgb_w = rgb.shape[:2]
            position[..., 3] = position[..., 2] < 0
            cam2world = camera_params[cam_uid]["cam2world_gl"]
            xyzw = position.reshape(-1, 4) @ cam2world.T
            xyzw = xyzw.reshape(rgb_h, rgb_w, 4)
            images["xyzw"] = xyzw
        return observation

