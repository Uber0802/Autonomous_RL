import os
import itertools
from pathlib import Path
import cv2
import numpy as np
import sapien
import torch
import torch.nn.functional as F
from sapien.physx import PhysxMaterial
from transforms3d.euler import euler2quat

from mani_skill.envs.sapien_env import BaseEnv
from .base_env import BRIDGE_DATASET_ASSET_PATH, \
    WidowX250SBridgeDatasetFlatTable
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import common, io_utils, sapien_utils
from mani_skill.utils.geometry import rotation_conversions
from mani_skill.utils.structs.actor import Actor
from mani_skill.utils.structs.pose import Pose
from mani_skill.utils.structs.types import SimConfig
from mani_skill.utils.registration import register_env

CARROT_DATASET_DIR = Path(__file__).resolve().parents[2] / "ManiSkill" / "mani_skill" / "assets" / "carrot"


def masks_to_boxes_pytorch(masks):
    b, H, W = masks.shape
    boxes = []
    for i in range(b):
        pos = masks[i].nonzero(as_tuple=False)  # [N, 2]
        if pos.shape[0] == 0:
            boxes.append(torch.tensor([0, 0, 0, 0], dtype=torch.long, device=masks.device))
        else:
            ymin, xmin = pos.min(dim=0)[0]
            ymax, xmax = pos.max(dim=0)[0]
            boxes.append(torch.stack([xmin, ymin, xmax, ymax]))
    return torch.stack(boxes, dim=0)  # [b, 4]


class BasePickPlace(BaseEnv):
    """Base Digital Twin environment for digital twins of the BridgeData v2"""

    SUPPORTED_OBS_MODES = ["rgb+segmentation"]
    SUPPORTED_REWARD_MODES = ["none"]

    obj_static_friction = 1.0
    obj_dynamic_friction = 1.0

    rgb_camera_name: str = "3rd_view_camera"
    rgb_overlay_mode: str = "background"  # 'background' or 'object' or 'debug' or combinations of them

    overlay_images_numpy: list[np.ndarray]
    overlay_textures_numpy: list[np.ndarray]
    overlay_mix_numpy: list[float]
    overlay_images: torch.Tensor
    overlay_textures: torch.Tensor
    overlay_mix: torch.Tensor
    model_db_carrot: dict[str, dict]
    model_db_plate: dict[str, dict]
    carrot_names: list[str]
    plate_names: list[str]
    select_carrot_ids: torch.Tensor
    select_plate_ids: torch.Tensor
    select_overlay_ids: torch.Tensor
    select_pos_ids: torch.Tensor
    select_quat_ids: torch.Tensor

    initial_qpos: np.ndarray
    initial_robot_pos: sapien.Pose
    safe_robot_pos: sapien.Pose

    def __init__(self, **kwargs):
        # random pose
        self._generate_init_pose()

        # widowx
        self.initial_qpos = np.array([
            -0.01840777, 0.0398835, 0.22242722,
            -0.00460194, 1.36524296, 0.00153398,
            0.037, 0.037,
        ])
        self.initial_robot_pos = sapien.Pose([0.147, 0.028, 0.870], q=[0, 0, 0, 1])
        self.safe_robot_pos = sapien.Pose([0.147, 0.028, 1.870], q=[0, 0, 0, 1])

        # stats
        self.extra_stats = dict()

        super().__init__(
            robot_uids=WidowX250SBridgeDatasetFlatTable,
            **kwargs
        )

    def _generate_init_pose(self):
        raise NotImplementedError

    @property
    def _default_sim_config(self):
        return SimConfig(sim_freq=500, control_freq=5, spacing=20)

    def _build_actor_helper(self, name: str, path: Path, density: float, scale: float, pose: Pose):
        """helper function to build actors by ID directly and auto configure physical materials"""
        physical_material = PhysxMaterial(
            static_friction=self.obj_static_friction,
            dynamic_friction=self.obj_dynamic_friction,
            restitution=0.0,
        )
        builder = self.scene.create_actor_builder()

        collision_file = str(path / "collision.obj")
        builder.add_multiple_convex_collisions_from_file(
            filename=collision_file,
            scale=[scale] * 3,
            material=physical_material,
            density=density,
        )

        visual_file = str(path / "textured.obj")
        if not os.path.exists(visual_file):
            visual_file = str(path / "textured.dae")
            if not os.path.exists(visual_file):
                visual_file = str(path / "textured.glb")
        builder.add_visual_from_file(filename=visual_file, scale=[scale] * 3)

        builder.initial_pose = pose
        actor = builder.build(name=name)
        return actor

    def _load_agent(self, options: dict):
        super()._load_agent(
            options, sapien.Pose(p=[0.127, 0.060, 0.85], q=[0, 0, 0, 1])
        )

    def _load_scene(self, options: dict):
        # original SIMPLER envs always do this? except for open drawer task
        for i in range(self.num_envs):
            sapien_utils.set_articulation_render_material(
                self.agent.robot._objs[i], specular=0.9, roughness=0.3
            )

        # load background
        builder = self.scene.create_actor_builder()  # Warning should be dissmissed, for we set the initial pose below -> actor.set_pose
        scene_pose = sapien.Pose(q=[0.707, 0.707, 0, 0])
        scene_offset = np.array([-2.0634, -2.8313, 0.0])

        scene_file = str(BRIDGE_DATASET_ASSET_PATH / "stages/bridge_table_1_v1.glb")

        builder.add_nonconvex_collision_from_file(scene_file, pose=scene_pose)
        builder.add_visual_from_file(scene_file, pose=scene_pose)
        builder.initial_pose = sapien.Pose(-scene_offset)
        builder.build_static(name="arena")

        # models
        self.model_bbox_sizes = {}

        # carrot
        self.objs_carrot: dict[str, Actor] = {}

        for idx, name in enumerate(self.model_db_carrot):
            model_path = CARROT_DATASET_DIR / "more_carrot" / name
            density = self.model_db_carrot[name].get("density", 1000)
            scale_list = self.model_db_carrot[name].get("scale", [1.0])
            bbox = self.model_db_carrot[name]["bbox"]

            scale = self.np_random.choice(scale_list)
            pose = Pose.create_from_pq(torch.tensor([1.0, 0.3 * idx, 1.0]))
            self.objs_carrot[name] = self._build_actor_helper(name, model_path, density, scale, pose)

            bbox_size = np.array(bbox["max"]) - np.array(bbox["min"])  # [3]
            self.model_bbox_sizes[name] = common.to_tensor(bbox_size * scale, device=self.device)  # [3]

        # plate
        self.objs_plate: dict[str, Actor] = {}

        for idx, name in enumerate(self.model_db_plate):
            model_path = CARROT_DATASET_DIR / "more_plate" / name
            density = self.model_db_plate[name].get("density", 1000)
            scale_list = self.model_db_plate[name].get("scale", [1.0])
            bbox = self.model_db_plate[name]["bbox"]

            scale = self.np_random.choice(scale_list)
            pose = Pose.create_from_pq(torch.tensor([2.0, 0.3 * idx, 1.0]))
            self.objs_plate[name] = self._build_actor_helper(name, model_path, density, scale, pose)

            bbox_size = np.array(bbox["max"]) - np.array(bbox["min"])  # [3]
            self.model_bbox_sizes[name] = common.to_tensor(bbox_size * scale, device=self.device)  # [3]

    def _load_lighting(self, options: dict):
        self.scene.set_ambient_light([0.3, 0.3, 0.3])
        self.scene.add_directional_light(
            [0, 0, -1],
            [2.2, 2.2, 2.2],
            shadow=False,
            shadow_scale=5,
            shadow_map_size=2048,
        )
        self.scene.add_directional_light([-1, -0.5, -1], [0.7, 0.7, 0.7])
        self.scene.add_directional_light([1, 1, -1], [0.7, 0.7, 0.7])

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        self._initialize_episode_pre(env_idx, options)
        raise NotImplementedError

    def _initialize_episode_pre(self, env_idx: torch.Tensor, options: dict):
        raise NotImplementedError

    def _settle(self, t=0.5):
        """run the simulation for some steps to help settle the objects"""
        if self.gpu_sim_enabled:
            self.scene._gpu_apply_all()

        sim_steps = int(self.sim_freq * t / self.control_freq)
        for _ in range(sim_steps):
            self.scene.step()

        if self.gpu_sim_enabled:
            self.scene._gpu_fetch_all()

    def reset_grasp_stats(self):
        self.consecutive_grasp.zero_()
        self.episode_stats["is_src_obj_grasped"].zero_()
        self.episode_stats["consecutive_grasp"].zero_()
        print("reset Grasp Stats")

    def evaluate(self, success_require_src_completely_on_target=True):
        xy_flag_required_offset = 0.01
        z_flag_required_offset = 0.05
        netforce_flag_required_offset = 0.03

        b = self.num_envs

        # actor
        if not hasattr(self, "select_carrot_ids"):
            # Fallback during initial reset if not yet set by set_current_task or _initialize_episode
            self.select_carrot_ids = getattr(self, "select_carrot1_ids", torch.zeros((b,), dtype=torch.long, device=self.device))
        if not hasattr(self, "select_plate_ids"):
            self.select_plate_ids = getattr(self, "select_plate1_ids", torch.zeros((b,), dtype=torch.long, device=self.device))

        select_carrot = [self.carrot_names[idx] for idx in self.select_carrot_ids]
        select_plate = [self.plate_names[idx] for idx in self.select_plate_ids]
        
        carrot_actor = [self.objs_carrot[n] for n in select_carrot]
        plate_actor = [self.objs_plate[n] for n in select_plate]

        carrot_p = torch.stack([a.pose.p[idx] for idx, a in enumerate(carrot_actor)])  # [b, 3]
        carrot_q = torch.stack([a.pose.q[idx] for idx, a in enumerate(carrot_actor)])  # [b, 4]
        plate_p = torch.stack([a.pose.p[idx] for idx, a in enumerate(plate_actor)])  # [b, 3]
        plate_q = torch.stack([a.pose.q[idx] for idx, a in enumerate(plate_actor)])  # [b, 4]

        # optional second plate
        src_on_target2 = torch.zeros((b,), dtype=torch.bool, device=self.device)
        if hasattr(self, "select_extra2_ids"):
            select_plate2 = [self.plate_names[idx] for idx in self.select_extra2_ids]
            plate2_actor = [self.objs_plate[n] for n in select_plate2]
            plate2_p = torch.stack([a.pose.p[idx] for idx, a in enumerate(plate2_actor)])  # [b, 3]
            plate2_q = torch.stack([a.pose.q[idx] for idx, a in enumerate(plate2_actor)])  # [b, 4]
            
            pos_tgt2 = plate2_p
            offset2 = carrot_p - pos_tgt2
            xy_flag2 = (
                torch.linalg.norm(offset2[:, :2], dim=1)
                <= self.plate_bbox_world.max(dim=1).values / 2 + xy_flag_required_offset
            )
            z_flag2 = (offset2[:, 2] > 0) & (
                offset2[:, 2] - self.plate_bbox_world[:, 2] / 2 - self.carrot_bbox_world[:, 2] / 2
                <= z_flag_required_offset
            )
            src_on_target2 = xy_flag2 & z_flag2

        # whether moved the correct object
        # source_obj_xy_move_dist = torch.linalg.norm(
        #     self.episode_source_obj_xyz_after_settle[:, :2] - source_obj_pose.p[:, :2],
        #     dim=1,
        # )
        # other_obj_xy_move_dist = []
        # for obj_name in self.objs.keys():
        #     obj = self.objs[obj_name]
        #     obj_xyz_after_settle = self.episode_obj_xyzs_after_settle[obj_name]
        #     if obj.name == self.source_obj_name:
        #         continue
        #     other_obj_xy_move_dist.append(
        #         torch.linalg.norm(
        #             obj_xyz_after_settle[:, :2] - obj.pose.p[:, :2], dim=1
        #         )
        #     )

        # moved_correct_obj = (source_obj_xy_move_dist > 0.03) and (
        #     all([x < source_obj_xy_move_dist for x in other_obj_xy_move_dist])
        # )
        # moved_wrong_obj = any([x > 0.03 for x in other_obj_xy_move_dist]) and any(
        #     [x > source_obj_xy_move_dist for x in other_obj_xy_move_dist]
        # )
        # moved_correct_obj = False
        # moved_wrong_obj = False

        # whether the source object is grasped

        is_src_obj_grasped = torch.zeros((b,), dtype=torch.bool, device=self.device)  # [b]

        for idx, name in enumerate(self.model_db_carrot):
            is_select = self.select_carrot_ids == idx  # [b]
            grasped = self.agent.is_grasping(self.objs_carrot[name])  # [b]
            is_src_obj_grasped = torch.where(is_select, grasped, is_src_obj_grasped)  # [b]

        # if is_src_obj_grasped:
        self.consecutive_grasp += is_src_obj_grasped
        self.consecutive_grasp[is_src_obj_grasped == 0] = 0
        consecutive_grasp = self.consecutive_grasp >= 5
        #print(f"is_src_obj_grasped: {is_src_obj_grasped.squeeze().cpu().tolist()}")
        #print(f"consecutive_grasp: {self.consecutive_grasp.squeeze().cpu().tolist()}")
        #g = self.episode_stats["is_src_obj_grasped"]
        #print(f"info is_src_obj_grasped: {g.squeeze().cpu().tolist()}")
        #g = self.episode_stats["consecutive_grasp"]
        #print(f"info consecutive_grasp: {g.squeeze().cpu().tolist()}")

        # whether the source object is on the target object based on bounding box position
        tgt_obj_half_length_bbox = getattr(self, "plate_bbox_world", torch.zeros((b, 3), device=self.device)) / 2
        src_obj_half_length_bbox = getattr(self, "carrot_bbox_world", torch.zeros((b, 3), device=self.device)) / 2

        pos_src = carrot_p
        pos_tgt = plate_p
        offset = pos_src - pos_tgt
        xy_flag = (
                torch.linalg.norm(offset[:, :2], dim=1)
                <= tgt_obj_half_length_bbox.max(dim=1).values + xy_flag_required_offset
        )
        z_flag = (offset[:, 2] > 0) & (
                offset[:, 2] - tgt_obj_half_length_bbox[:, 2] - src_obj_half_length_bbox[:, 2]
                <= z_flag_required_offset
        )
        src_on_target = xy_flag & z_flag
        # src_on_target = False

        if success_require_src_completely_on_target:
            # whether the source object is on the target object based on contact information
            net_forces = torch.zeros((b,), dtype=torch.float32, device=self.device)  # [b]
            for idx in range(self.num_envs):
                force = self.scene.get_pairwise_contact_forces(
                    self.objs_carrot[select_carrot[idx]],
                    self.objs_plate[select_plate[idx]],
                )[idx]
                force = torch.linalg.norm(force)
                net_forces[idx] = force

            src_on_target = src_on_target & (net_forces > netforce_flag_required_offset)

        success = src_on_target

        # Track if object on other receptable (handled above)

        # prepare dist
        gripper_p = (self.agent.finger1_link.pose.p + self.agent.finger2_link.pose.p) / 2  # [b, 3]
        gripper_q = (self.agent.finger1_link.pose.q + self.agent.finger2_link.pose.q) / 2  # [b, 4]
        gripper_carrot_dist = torch.linalg.norm(gripper_p - carrot_p, dim=1)  # [b, 3]
        gripper_plate_dist = torch.linalg.norm(gripper_p - plate_p, dim=1)  # [b, 3]
        carrot_plate_dist = torch.linalg.norm(carrot_p - plate_p, dim=1)  # [b, 3]

        # Track if object on table
        src_not_on_any_plate = (~src_on_target) & (~src_on_target2)
        src_not_grasped = ~is_src_obj_grasped
        src_not_on_floor = carrot_p[:, 2] >= 0.7
        src_on_table = src_not_on_any_plate & src_not_grasped & src_not_on_floor

        # self.episode_stats["moved_correct_obj"] = moved_correct_obj
        # self.episode_stats["moved_wrong_obj"] = moved_wrong_obj
        self.episode_stats["src_on_target"] = src_on_target
        self.episode_stats["src_on_target2"] = src_on_target2
        self.episode_stats["src_on_table"] = src_on_table
        self.episode_stats["is_src_obj_grasped"] = self.episode_stats["is_src_obj_grasped"] | is_src_obj_grasped
        self.episode_stats["consecutive_grasp"] = self.episode_stats["consecutive_grasp"] | consecutive_grasp
        self.episode_stats["gripper_carrot_dist"] = gripper_carrot_dist
        self.episode_stats["gripper_plate_dist"] = gripper_plate_dist
        self.episode_stats["carrot_plate_dist"] = carrot_plate_dist

        self.extra_stats["extra_pos_carrot"] = carrot_p
        self.extra_stats["extra_q_carrot"] = carrot_q
        self.extra_stats["extra_pos_plate"] = plate_p
        self.extra_stats["extra_q_plate"] = plate_q
        self.extra_stats["extra_pos_gripper"] = gripper_p
        self.extra_stats["extra_q_gripper"] = gripper_q

        return dict(**self.episode_stats, success=success)

    def is_final_subtask(self):
        # whether the current subtask is the final one, only meaningful for long-horizon tasks
        return True

    def get_carrot_actors(self):
        select_carrot = [self.carrot_names[idx] for idx in self.select_carrot_ids]
        return [self.objs_carrot[n] for n in select_carrot]

    def get_plate_actors(self):
        select_plate = [self.plate_names[idx] for idx in self.select_plate_ids]
        return [self.objs_plate[n] for n in select_plate]

    def get_extra_plate_actors(self):
        if hasattr(self, "select_extra2_ids"):
            select_plate2 = [self.plate_names[idx] for idx in self.select_extra2_ids]
            return [self.objs_plate[n] for n in select_plate2]
        return self.get_plate_actors() # Fallback

    def get_obj_pos(self):
        return self.extra_stats["extra_pos_carrot"]

    def get_recep_pos(self):    
        return self.extra_stats["extra_pos_plate"]

    def reset_unsuitable_envs(self, env_idx=[]):
        xyz_min = torch.tensor([-0.235, -0.075, 0.92], device=self.device)
        xyz_max = torch.tensor([-0.085,  0.075, 0.95], device=self.device)
        select_carrot = [self.carrot_names[idx] for idx in self.select_carrot_ids]
        select_plate = [self.plate_names[idx] for idx in self.select_plate_ids]
        carrot_actor = [self.objs_carrot[n] for n in select_carrot]
        plate_actor = [self.objs_plate[n] for n in select_plate]

        # Get unsuitable object, recep idx
        obj_pos = self.get_obj_pos()
        recep_pos = self.get_recep_pos()
        obj_z = obj_pos[:, 2]
        recep_z = recep_pos[:, 2]

        # Find indices where obj_z < 0.7
        obj_low_z_mask = obj_z < 0.7
        obj_low_z_indices = torch.nonzero(obj_low_z_mask, as_tuple=False).squeeze()

        # Convert to list of ints
        if obj_low_z_indices.ndim == 0:
            obj_low_z_list = [obj_low_z_indices.item()]
        else:
            obj_low_z_list = obj_low_z_indices.tolist()

        # Find indices where recep_z < 0.7
        recep_low_z_mask = recep_z < 0.7
        recep_low_z_indices = torch.nonzero(recep_low_z_mask, as_tuple=False).squeeze()

        # Convert to list of ints
        if recep_low_z_indices.ndim == 0:
            recep_low_z_list = [recep_low_z_indices.item()]
        else:
            recep_low_z_list = recep_low_z_indices.tolist()

        # Generate position & quant samples
        lc = 16
        lo = len(self.overlay_images_numpy)
        l1 = len(self.xyz_configs)
        l2 = len(self.quat_configs)
        ltt = lc * 1 * 16 * lo * l1 * l2
        indices = np.random.choice(ltt, self.num_envs)
        xyz_indices = (indices//l2) %l1
        xyz_sample = torch.tensor(self.xyz_configs[xyz_indices], device=self.device)
        quant_indices = indices % l2
        quant_sample = torch.tensor(self.quat_configs[quant_indices], device=self.device)

        # loop over plate
        for idx, a in enumerate(plate_actor):
            if idx in recep_low_z_list:
                pos = xyz_sample[idx]
                quant = quant_sample[idx]
                prev_mask = a.scene._reset_mask.clone()
                a.scene._reset_mask[:] = False
                a.scene._reset_mask[idx] = True
                # set the pose in batched format
                a.set_pose(Pose.create_from_pq(p=pos[2], q=quant[1]))
                a.scene._reset_mask = prev_mask

        # loop over carrot
        for idx, a in enumerate(carrot_actor):
            if idx in obj_low_z_list:
                pos = xyz_sample[idx]
                quant = quant_sample[idx]
                prev_mask = a.scene._reset_mask.clone()
                a.scene._reset_mask[:] = False
                a.scene._reset_mask[idx] = True
                # set the pose in batched format
                a.set_pose(Pose.create_from_pq(p=pos[0], q=quant[1]))
                a.scene._reset_mask = prev_mask

        self._settle(0.5)
        print(f"Reset Unsuitable. Obj: {obj_low_z_list}, Recep: {recep_low_z_list}")
        reset_env_count = len(set(obj_low_z_list) | set(recep_low_z_list))
        return reset_env_count

    def get_language_instruction(self):
        select_carrot = [self.carrot_names[idx] for idx in self.select_carrot_ids]
        select_plate = [self.plate_names[idx] for idx in self.select_plate_ids]

        instruct = []
        for idx in range(self.num_envs):
            carrot_name = self.model_db_carrot[select_carrot[idx]]["name"]
            plate_name = self.model_db_plate[select_plate[idx]]["name"]
            instruct.append(f"put {carrot_name} on {plate_name}")
        return instruct

    def _after_reconfigure(self, options: dict):
        target_object_actor_ids = [
            x._objs[0].per_scene_id
            for x in self.scene.actors.values()
            if x.name not in ["ground", "goal_site", "", "arena"]
        ]
        self.target_object_actor_ids = torch.tensor(
            target_object_actor_ids, dtype=torch.int16, device=self.device
        )
        # get the robot link ids
        robot_links = self.agent.robot.get_links()
        self.robot_link_ids = torch.tensor(
            [x._objs[0].entity.per_scene_id for x in robot_links],
            dtype=torch.int16,
            device=self.device,
        )

    def _green_sceen_rgb(self, rgb, segmentation, overlay_img, overlay_texture, overlay_mix):
        """returns green screened RGB data given a batch of RGB and segmentation images and one overlay image"""
        actor_seg = segmentation[..., 0]
        # mask = torch.ones_like(actor_seg, device=actor_seg.device)
        if actor_seg.device != self.robot_link_ids.device:
            # if using CPU simulation, the device of the robot_link_ids and target_object_actor_ids will be CPU first
            # but for most users who use the sapien_cuda render backend image data will be on the GPU.
            self.robot_link_ids = self.robot_link_ids.to(actor_seg.device)
            self.target_object_actor_ids = self.target_object_actor_ids.to(actor_seg.device)

        mask = torch.isin(actor_seg, torch.concat([self.robot_link_ids, self.target_object_actor_ids]))
        mask = (~mask).to(torch.float32)  # [b, H, W]
        # m = torch.isin(actor_seg, self.robot_link_ids) # "object" mode

        mask = mask.unsqueeze(-1)  # [b, H, W, 1]
        # mix = overlay_mix.unsqueeze(1).unsqueeze(1).unsqueeze(1)  # [b, 1, 1, 1]

        # perform overlay on the RGB observation image
        assert rgb.shape == overlay_img.shape
        assert rgb.shape == overlay_texture.shape

        rgb = rgb.to(torch.float32)  # [b, H, W, 3]

        rgb_ret = overlay_img * mask  # [b, H, W, 3]
        rgb_ret += rgb * (1 - mask)  # [b, H, W, 3]

        rgb_ret = torch.clamp(rgb_ret, 0, 255)  # [b, H, W, 3]
        rgb_ret = rgb_ret.to(torch.uint8)  # [b, H, W, 3]

        # rgb = rgb * (1 - mask) + overlay_img * mask
        # rgb = rgb * 0.5 + overlay_img * 0.5 # "debug" mode

        return rgb_ret

    def get_obs(self, info: dict = None):
        obs = super().get_obs(info)

        # "greenscreen" process
        if self.obs_mode_struct.visual.rgb and self.obs_mode_struct.visual.segmentation and self.overlay_images_numpy:
            # get the actor ids of objects to manipulate; note that objects here are not articulated
            camera_name = self.rgb_camera_name
            assert "segmentation" in obs["sensor_data"][camera_name].keys()

            overlay_img = self.overlay_images.to(obs["sensor_data"][camera_name]["rgb"].device)
            overlay_texture = self.overlay_textures.to(obs["sensor_data"][camera_name]["rgb"].device)
            overlay_mix = self.overlay_mix.to(obs["sensor_data"][camera_name]["rgb"].device)

            green_screened_rgb = self._green_sceen_rgb(
                obs["sensor_data"][camera_name]["rgb"],
                obs["sensor_data"][camera_name]["segmentation"],
                overlay_img,
                overlay_texture,
                overlay_mix
            )
            obs["sensor_data"][camera_name]["rgb"] = green_screened_rgb
        return obs

    # widowx
    @property
    def _default_human_render_camera_configs(self):
        sapien_utils.look_at([0.6, 0.7, 0.6], [0.0, 0.0, 0.35])
        return CameraConfig(
            "render_camera",
            pose=sapien.Pose(
                [0.00, -0.16, 0.336], [0.909182, -0.0819809, 0.347277, 0.214629]
            ),
            width=512,
            height=512,
            intrinsic=np.array(
                [[623.588, 0, 319.501], [0, 623.588, 239.545], [0, 0, 1]]
            ),
            near=0.01,
            far=100,
            mount=self.agent.robot.links_map["base_link"],
        )   

class BaseMultiPickPlace(BasePickPlace):
    def __init__(self, **kwargs):
        self._prep_init()

        super().__init__(**kwargs)

    def _prep_init(self):
        # models
        self.model_db_carrot: dict[str, dict] = io_utils.load_json(
            CARROT_DATASET_DIR / "more_carrot" / "model_db.json"
        )
        assert len(self.model_db_carrot) == 25

        self.model_db_plate: dict[str, dict] = io_utils.load_json(
            CARROT_DATASET_DIR / "more_plate" / "model_db.json"
        )
        only_plate_name = list(self.model_db_plate.keys())[0]
        self.model_db_plate = {k: v for k, v in self.model_db_plate.items() if k == only_plate_name}
        assert len(self.model_db_plate) == 1

        # random configs
        self.carrot_names = list(self.model_db_carrot.keys())
        self.plate_names = list(self.model_db_plate.keys())

        # rgb overlay
        model_db_table = io_utils.load_json(
            CARROT_DATASET_DIR / "more_table" / "model_db.json"
        )

        img_fd = CARROT_DATASET_DIR / "more_table" / "imgs"
        texture_fd = CARROT_DATASET_DIR / "more_table" / "textures"
        self.overlay_images_numpy = [
            cv2.resize(cv2.cvtColor(cv2.imread(str(img_fd / k)), cv2.COLOR_BGR2RGB), (640, 480))
            for k in model_db_table  # [H, W, 3]
        ]  # (B) [H, W, 3]
        self.overlay_textures_numpy = [
            cv2.resize(cv2.cvtColor(cv2.imread(str(texture_fd / v["texture"])), cv2.COLOR_BGR2RGB), (640, 480))
            for v in model_db_table.values()  # [H, W, 3]
        ]  # (B) [H, W, 3]
        self.overlay_mix_numpy = [
            v["mix"] for v in model_db_table.values()  # []
        ]
        assert len(self.overlay_images_numpy) == 21
        assert len(self.overlay_textures_numpy) == 21
        assert len(self.overlay_mix_numpy) == 21

    def set_current_task(self, object: list[str], receptacle: list[str]):
        raise NotImplementedError

    def object_name(self):
        """
        Get all object names in all env.
        return list[list[str]]
        """
        raise NotImplementedError

    def receptacle_name(self):
        """
        Get all receptacle names in all env.
        return list[list[str]]
        """
        raise NotImplementedError

    def object_id(self):
        """
        Get all object ids in all env.
        """
        raise NotImplementedError

    def receptacle_id(self):
        """
        Get all receptacle ids in all env.
        """
        raise NotImplementedError

    def task_pool(self):
        """
        Get task pool in all env.
        return list[list[str]]
        """
        objects = self.object_name()
        receptacles = self.receptacle_name()

        tasks_all_env = []
        for idx in range(self.num_envs):
            tasks_cur_env = []
            
            for pair in itertools.product(objects[idx], receptacles[idx]):
                tasks_cur_env.append(f"put {pair[0]} on {pair[1]}")
            
            tasks_all_env.append(tasks_cur_env)
        
        return tasks_all_env



@register_env("TwoObjectOneReceptacle-v1", max_episode_steps=80, asset_download_ids=["bridge_v2_real2sim"])
class TwoObjectOneReceptacle(BaseMultiPickPlace):
    select_extra_ids: torch.Tensor
    select_carrot1_ids: torch.Tensor
    select_carrot2_ids: torch.Tensor

    def get_carrot_actors(self):
        select_carrot = [self.carrot_names[idx] for idx in self.select_carrot_ids]
        return [self.objs_carrot[n] for n in select_carrot]

    def get_plate_actors(self):
        select_plate = [self.plate_names[idx] for idx in self.select_plate_ids]
        return [self.objs_plate[n] for n in select_plate]

    def get_extra_plate_actors(self):
        return self.get_plate_actors() # Only one plate

    def get_carrot_pose(self):
        p = torch.zeros((self.num_envs, 3), device=self.device)
        for name, actor in self.objs_carrot.items():
            mask = (self.select_carrot_ids == self.carrot_names.index(name))
            if mask.any(): p[mask] = actor.pose.p[mask]
        return p

    def get_plate_pose(self):
        p = torch.zeros((self.num_envs, 3), device=self.device)
        for name, actor in self.objs_plate.items():
            mask = (self.select_plate_ids == self.plate_names.index(name))
            if mask.any(): p[mask] = actor.pose.p[mask]
        return p

    def get_extra_plate_pose(self):
        return self.get_plate_pose()
    
    POSE_PRESET = "TwoObjectOneReceptacle"
    NUM_OBJECTS = 2
    NUM_RECEPTACLES = 1
    DEFAULT_OBJ_INDICES = [7, 2]     # 1-based
    DEFAULT_PLATE_INDICES = [1]      # 1-based

    def _generate_init_pose(self):
        from .suite import generate_pose_configs, POSE_PRESETS, QUAT_CONFIGS
        params = POSE_PRESETS[self.POSE_PRESET]
        self.xyz_configs = generate_pose_configs(**params)
        self.quat_configs = QUAT_CONFIGS.copy()
        print(f"xyz_configs: {self.xyz_configs.shape}")
        print(f"quat_configs: {self.quat_configs.shape}")

    def _initialize_episode_pre(self, env_idx: torch.Tensor, options: dict):
        # NOTE: this part of code is not GPU parallelized
        b = len(env_idx)
        assert b == self.num_envs

        self._generate_init_pose()

        lc = 16
        lo = len(self.overlay_images_numpy)
        l1 = len(self.xyz_configs)
        l2 = len(self.quat_configs)
        lp = 1
        le = 16
        ltt = lc * lp * le * lo * l1 * l2

        # Use one shared episode ID for all environments
        if "episode_id" in options:
            episode_id = options["episode_id"]
        else:
            single_id = torch.randint(low=0, high=ltt, size=(1,), device=self.device).item()
            episode_id = torch.full((b,), single_id, device=self.device)
            episode_id = episode_id.reshape(b)
            episode_id = episode_id % ltt

        obj1_index = options.get("obj1_index", self.DEFAULT_OBJ_INDICES[0]) - 1
        obj2_index = options.get("obj2_index", self.DEFAULT_OBJ_INDICES[1]) - 1
        self.select_carrot1_ids = torch.full((b,), obj1_index, device=self.device)
        self.select_carrot2_ids = torch.full((b,), obj2_index, device=self.device)

        plate1_index = options.get("plate1_index", self.DEFAULT_PLATE_INDICES[0]) - 1
        self.select_plate_ids = torch.full((b,), plate1_index, device=self.device)

        self.select_overlay_ids = (episode_id // (l1 * l2)) % lo
        self.select_pos_ids = (episode_id // l2) % l1
        self.select_quat_ids = episode_id % l2

    def set_current_task(self, object: list[str], receptacle: list[str]):
        select_carrot1 = [self.carrot_names[idx] for idx in self.select_carrot1_ids]
        select_carrot2 = [self.carrot_names[idx] for idx in self.select_carrot2_ids]
        select_plate = [self.plate_names[idx] for idx in self.select_plate_ids]

        new_select_carrot_ids = []
        new_select_plate_ids = []
        for idx in range(self.num_envs):
            if object[idx] == self.model_db_carrot[select_carrot1[idx]]["name"]:
                new_select_carrot_ids.append(self.select_carrot1_ids[idx])
            elif object[idx] == self.model_db_carrot[select_carrot2[idx]]["name"]:
                new_select_carrot_ids.append(self.select_carrot2_ids[idx])
            else:
                raise ValueError(f"{object[idx]} does not exist!")

            if receptacle[idx] == self.model_db_plate[select_plate[idx]]["name"]:
                new_select_plate_ids.append(self.select_plate_ids[idx])
            else:
                raise ValueError(f"{receptacle[idx]} does not exist!")

        self.select_carrot_ids = torch.stack(new_select_carrot_ids)
        self.select_plate_ids = torch.stack(new_select_plate_ids)

        # for motion planning capability
        select_carrot = [self.carrot_names[idx] for idx in self.select_carrot_ids]
        select_plate = [self.plate_names[idx] for idx in self.select_plate_ids]
        carrot_actor = [self.objs_carrot[n] for n in select_carrot]
        plate_actor = [self.objs_plate[n] for n in select_plate]
        self.source_obj_name = select_carrot
        self.target_obj_name = select_plate
        self.objs = {
            self.source_obj_name[0]: carrot_actor[0],
            self.target_obj_name[0]: plate_actor[0]
        }

    def object_name(self):
        """
        Get all object names in env.
        """
        select_carrot1 = [self.carrot_names[idx] for idx in self.select_carrot1_ids]
        select_carrot2 = [self.carrot_names[idx] for idx in self.select_carrot2_ids]

        name_in_all_envs = []
        for idx in range(self.num_envs):
            name_in_cur_envs = []
            name_in_cur_envs.append(f"{self.model_db_carrot[select_carrot1[idx]]['name']}")
            name_in_cur_envs.append(f"{self.model_db_carrot[select_carrot2[idx]]['name']}")
            name_in_all_envs.append(name_in_cur_envs)

        return name_in_all_envs


    def receptacle_name(self):
        """
        Get all receptacle names in env.
        """
        select_plate = [self.plate_names[idx] for idx in self.select_plate_ids]

        name_in_all_envs = []
        for idx in range(self.num_envs):
            name_in_cur_envs = []
            name_in_cur_envs.append(f"{self.model_db_plate[select_plate[idx]]['name']}")
            name_in_all_envs.append(name_in_cur_envs)

        return name_in_all_envs

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        self._initialize_episode_pre(env_idx, options)

        # Initial set carrot1 as task object, carrot 2 as extra object
        self.select_carrot_ids = self.select_carrot1_ids
        self.select_extra_ids = self.select_carrot2_ids

        b = self.num_envs

        # rgb overlay
        sensor = self._sensor_configs[self.rgb_camera_name]
        assert sensor.width == 640
        assert sensor.height == 480
        overlay_images = np.stack([self.overlay_images_numpy[idx] for idx in self.select_overlay_ids])
        self.overlay_images = torch.tensor(overlay_images, device=self.device)  # [b, H, W, 3]
        overlay_textures = np.stack([self.overlay_textures_numpy[idx] for idx in self.select_overlay_ids])
        self.overlay_textures = torch.tensor(overlay_textures, device=self.device)  # [b, H, W, 3]
        overlay_mix = np.array([self.overlay_mix_numpy[idx] for idx in self.select_overlay_ids])
        self.overlay_mix = torch.tensor(overlay_mix, device=self.device)  # [b]

        # xyz and quat
        xyz_configs = torch.tensor(self.xyz_configs, device=self.device)
        quat_configs = torch.tensor(self.quat_configs, device=self.device)

        select_carrot = [self.carrot_names[idx] for idx in self.select_carrot_ids]
        select_plate = [self.plate_names[idx] for idx in self.select_plate_ids]
        select_extra = [self.carrot_names[idx] for idx in self.select_extra_ids]
        carrot_actor = [self.objs_carrot[n] for n in select_carrot]
        plate_actor = [self.objs_plate[n] for n in select_plate]
        extra_actor = [self.objs_carrot[n] for n in select_extra]

        # for motion planning capability
        self.source_obj_name = select_carrot
        self.target_obj_name = select_plate
        self.objs = {
            self.source_obj_name[0]: carrot_actor[0],
            self.target_obj_name[0]: plate_actor[0]
        }

        # set pose for robot
        self.agent.robot.set_pose(self.safe_robot_pos)
        # self._settle(0.5)

        # set pose for objs
        for idx, name in enumerate(self.model_db_carrot):
            p_reset = torch.tensor([1.0, 0.3 * idx, 1.0], device=self.device).reshape(1, -1).repeat(b, 1)  # [b, 3]
            is_select = self.select_carrot_ids == idx  # [b]
            p_select = xyz_configs[self.select_pos_ids, 0].reshape(b, 3)  # [b, 3]
            is_select_extra = self.select_extra_ids == idx  # [b]
            p_select_extra = xyz_configs[self.select_pos_ids, 2].reshape(b, 3)  # [b, 3]
            p = torch.where(is_select.unsqueeze(1).repeat(1, 3), p_select, p_reset)  # [b, 3]
            p = torch.where(is_select_extra.unsqueeze(1).repeat(1, 3), p_select_extra, p)  # [b, 3]

            q_reset = torch.tensor([0, 0, 0, 1], device=self.device).reshape(1, -1).repeat(b, 1)  # [b, 4]
            q_select = quat_configs[self.select_quat_ids, 0].reshape(b, 4)  # [b, 4]
            q = torch.where(is_select.unsqueeze(1).repeat(1, 4), q_select, q_reset)  # [b, 4]
            q = torch.where(is_select_extra.unsqueeze(1).repeat(1, 4), q_select, q)  # [b, 4]

            self.objs_carrot[name].set_pose(Pose.create_from_pq(p=p, q=q))

        for idx, name in enumerate(self.model_db_plate):
            is_select = self.select_plate_ids == idx  # [b]
            p_reset = torch.tensor([2.0, 0.3 * idx, 1.0], device=self.device).reshape(1, -1).repeat(b, 1)  # [b, 3]
            p_select = xyz_configs[self.select_pos_ids, 1].reshape(b, 3)  # [b, 3]
            p = torch.where(is_select.unsqueeze(1).repeat(1, 3), p_select, p_reset)  # [b, 3]

            q_reset = torch.tensor([0, 0, 0, 1], device=self.device).reshape(1, -1).repeat(b, 1)  # [b, 4]
            q_select = quat_configs[self.select_quat_ids, 1].reshape(b, 4)  # [b, 4]
            q = torch.where(is_select.unsqueeze(1).repeat(1, 4), q_select, q_reset)  # [b, 4]

            self.objs_plate[name].set_pose(Pose.create_from_pq(p=p, q=q))

        self._settle(0.5)

        # Some objects need longer time to settle
        c_lin = torch.stack([a.linear_velocity[i] for i, a in enumerate(carrot_actor)])
        c_ang = torch.stack([a.angular_velocity[i] for i, a in enumerate(carrot_actor)])
        p_lin = torch.stack([a.linear_velocity[i] for i, a in enumerate(plate_actor)])
        p_ang = torch.stack([a.angular_velocity[i] for i, a in enumerate(plate_actor)])
        e_lin = torch.stack([a.linear_velocity[i] for i, a in enumerate(extra_actor)])
        e_ang = torch.stack([a.angular_velocity[i] for i, a in enumerate(extra_actor)])

        lin_vel = torch.linalg.norm(c_lin) + torch.linalg.norm(p_lin) + torch.linalg.norm(e_lin)
        ang_vel = torch.linalg.norm(c_ang) + torch.linalg.norm(p_ang) + torch.linalg.norm(e_ang)

        if lin_vel > 1e-3 or ang_vel > 1e-2:
            self._settle(6)

        # measured values for bridge dataset
        self.agent.robot.set_pose(self.initial_robot_pos)
        self.agent.reset(init_qpos=self.initial_qpos)

        # figure out object bounding boxes after settling. This is used to determine if an object is near the target object
        self.carrot_q_after_settle = torch.stack([a.pose.q[idx] for idx, a in enumerate(carrot_actor)])  # [b, 4]
        self.plate_q_after_settle = torch.stack([a.pose.q[idx] for idx, a in enumerate(plate_actor)])  # [b, 4]
        corner_signs = torch.tensor([
            [-1, -1, -1], [-1, -1, 1], [-1, 1, -1], [-1, 1, 1],
            [1, -1, -1], [1, -1, 1], [1, 1, -1], [1, 1, 1]
        ], device=self.device)

        # carrot
        carrot_bbox_world = torch.stack([self.model_bbox_sizes[n] for n in select_carrot])  # [b, 3]
        c_bbox_half = carrot_bbox_world / 2  # [b, 3]
        c_bbox_corners = c_bbox_half[:, None, :] * corner_signs[None, :, :]  # [b, 8, 3]

        c_q_matrix = rotation_conversions.quaternion_to_matrix(self.carrot_q_after_settle)  # [b, 3, 3]
        c_bbox_corners_rot = torch.matmul(c_bbox_corners, c_q_matrix.transpose(1, 2))  # [b, 8, 3]
        c_rotated_bbox_size = c_bbox_corners_rot.max(dim=1).values - c_bbox_corners_rot.min(dim=1).values  # [b, 3]
        self.carrot_bbox_world = c_rotated_bbox_size  # [b, 3]

        # plate
        plate_bbox_world = torch.stack([self.model_bbox_sizes[n] for n in select_plate])  # [b, 3]
        p_bbox_half = plate_bbox_world / 2  # [b, 3]
        p_bbox_corners = p_bbox_half[:, None, :] * corner_signs[None, :, :]  # [b, 8, 3]

        p_q_matrix = rotation_conversions.quaternion_to_matrix(self.plate_q_after_settle)  # [b, 3, 3]
        p_bbox_corners_rot = torch.matmul(p_bbox_corners, p_q_matrix.transpose(1, 2))  # [b, 8, 3]
        p_rotated_bbox_size = p_bbox_corners_rot.max(dim=1).values - p_bbox_corners_rot.min(dim=1).values  # [b, 3]
        self.plate_bbox_world = p_rotated_bbox_size  # [b, 3]

        # stats to track
        self.consecutive_grasp = torch.zeros((b,), dtype=torch.int32, device=self.device)
        self.episode_stats = dict(
            # all_obj_keep_height=torch.zeros((b,), dtype=torch.bool),
            # moved_correct_obj=torch.zeros((b,), dtype=torch.bool),
            # moved_wrong_obj=torch.zeros((b,), dtype=torch.bool),
            # near_tgt_obj=torch.zeros((b,), dtype=torch.bool),
            is_src_obj_grasped=torch.zeros((b,), dtype=torch.bool, device=self.device),
            # is_closest_to_tgt=torch.zeros((b,), dtype=torch.bool),
            consecutive_grasp=torch.zeros((b,), dtype=torch.bool, device=self.device),
            src_on_target=torch.zeros((b,), dtype=torch.bool, device=self.device),

            gripper_carrot_dist=torch.zeros((b,), dtype=torch.float32, device=self.device),
            gripper_plate_dist=torch.zeros((b,), dtype=torch.float32, device=self.device),
            carrot_plate_dist=torch.zeros((b,), dtype=torch.float32, device=self.device),
        )


@register_env("OneObjectTwoReceptacle-v1", max_episode_steps=80, asset_download_ids=["bridge_v2_real2sim"])
class OneObjectTwoReceptacle(BaseMultiPickPlace):
    select_extra_ids: torch.Tensor
    select_plate1_ids: torch.Tensor
    select_plate2_ids: torch.Tensor

    def get_carrot_actors(self):
        select_carrot = [self.carrot_names[idx] for idx in self.select_carrot_ids]
        return [self.objs_carrot[n] for n in select_carrot]

    def get_plate_actors(self):
        select_plate = [self.plate_names[idx] for idx in self.select_plate_ids]
        return [self.objs_plate[n] for n in select_plate]

    def get_extra_plate_actors(self):
        select_extra = [self.plate_names[idx] for idx in self.select_extra_ids]
        return [self.objs_plate[n] for n in select_extra]

    def get_carrot_pose(self):
        p = torch.zeros((self.num_envs, 3), device=self.device)
        for name, actor in self.objs_carrot.items():
            mask = (self.select_carrot_ids == self.carrot_names.index(name))
            if mask.any(): p[mask] = actor.pose.p[mask]
        return p

    def get_plate_pose(self):
        p = torch.zeros((self.num_envs, 3), device=self.device)
        for name, actor in self.objs_plate.items():
            mask = (self.select_plate_ids == self.plate_names.index(name))
            if mask.any(): p[mask] = actor.pose.p[mask]
        return p

    def get_extra_plate_pose(self):
        p = torch.zeros((self.num_envs, 3), device=self.device)
        for name, actor in self.objs_plate.items():
            mask = (self.select_extra_ids == self.plate_names.index(name))
            if mask.any(): p[mask] = actor.pose.p[mask]
        return p

    def _prep_init(self):
        # models
        self.model_db_carrot: dict[str, dict] = io_utils.load_json(
            CARROT_DATASET_DIR / "more_carrot" / "model_db.json"
        )
        assert len(self.model_db_carrot) == 25

        self.model_db_plate: dict[str, dict] = io_utils.load_json(
            CARROT_DATASET_DIR / "more_plate" / "model_db.json"
        )
        #assert len(self.model_db_plate) == 17

        # random configs
        self.carrot_names = list(self.model_db_carrot.keys())
        self.plate_names = list(self.model_db_plate.keys())

        # rgb overlay
        model_db_table = io_utils.load_json(
            CARROT_DATASET_DIR / "more_table" / "model_db.json"
        )

        img_fd = CARROT_DATASET_DIR / "more_table" / "imgs"
        texture_fd = CARROT_DATASET_DIR / "more_table" / "textures"
        self.overlay_images_numpy = [
            cv2.resize(cv2.cvtColor(cv2.imread(str(img_fd / k)), cv2.COLOR_BGR2RGB), (640, 480))
            for k in model_db_table  # [H, W, 3]
        ]  # (B) [H, W, 3]
        self.overlay_textures_numpy = [
            cv2.resize(cv2.cvtColor(cv2.imread(str(texture_fd / v["texture"])), cv2.COLOR_BGR2RGB), (640, 480))
            for v in model_db_table.values()  # [H, W, 3]
        ]  # (B) [H, W, 3]
        self.overlay_mix_numpy = [
            v["mix"] for v in model_db_table.values()  # []
        ]
        assert len(self.overlay_images_numpy) == 21
        assert len(self.overlay_textures_numpy) == 21
        assert len(self.overlay_mix_numpy) == 21

    POSE_PRESET = "OneObjectTwoReceptacle"
    NUM_OBJECTS = 1
    NUM_RECEPTACLES = 2
    DEFAULT_OBJ_INDICES = [7]        # 1-based
    DEFAULT_PLATE_INDICES = [1, 2]   # 1-based

    def _generate_init_pose(self):
        from .suite import generate_pose_configs, POSE_PRESETS, QUAT_CONFIGS
        params = POSE_PRESETS[self.POSE_PRESET]
        self.xyz_configs = generate_pose_configs(**params)
        self.quat_configs = QUAT_CONFIGS.copy()
        print(f"xyz_configs: {self.xyz_configs.shape}")
        print(f"quat_configs: {self.quat_configs.shape}")

    def _initialize_episode_pre(self, env_idx: torch.Tensor, options: dict):
        # NOTE: this part of code is not GPU parallelized
        b = len(env_idx)
        assert b == self.num_envs

        self._generate_init_pose()

        lc = 16
        lo = len(self.overlay_images_numpy)
        l1 = len(self.xyz_configs)
        l2 = len(self.quat_configs)
        lp = 1
        le = 16
        ltt = lc * lp * le * lo * l1 * l2

        # Use one shared episode ID for all environments
        if "episode_id" in options:
            episode_id = options["episode_id"]
        else:
            single_id = torch.randint(low=0, high=ltt, size=(1,), device=self.device).item()
            episode_id = torch.full((b,), single_id, device=self.device)
            episode_id = episode_id.reshape(b)
            episode_id = episode_id % ltt

        obj1_index = options.get("obj1_index", self.DEFAULT_OBJ_INDICES[0]) - 1
        self.select_carrot_ids = torch.full((b,), obj1_index, device=self.device)

        plate1_index = options.get("plate1_index", self.DEFAULT_PLATE_INDICES[0]) - 1
        plate2_index = options.get("plate2_index", self.DEFAULT_PLATE_INDICES[1]) - 1
        self.select_plate1_ids = torch.full((b,), plate1_index, device=self.device)
        self.select_plate2_ids = torch.full((b,), plate2_index, device=self.device)

        self.select_overlay_ids = (episode_id // (l1 * l2)) % lo
        self.select_pos_ids = (episode_id // l2) % l1
        self.select_quat_ids = episode_id % l2

    def set_current_task(self, object: list[str], receptacle: list[str]):
        select_carrot = [self.carrot_names[idx] for idx in self.select_carrot_ids]
        select_plate1 = [self.plate_names[idx] for idx in self.select_plate1_ids]
        select_plate2 = [self.plate_names[idx] for idx in self.select_plate2_ids]

        new_select_carrot_ids = []
        new_select_plate_ids = []
        for idx in range(self.num_envs):
            if object[idx] == self.model_db_carrot[select_carrot[idx]]["name"]:
                new_select_carrot_ids.append(self.select_carrot_ids[idx])
            else:
                raise ValueError(f"{object[idx]} does not exist!")

            if receptacle[idx] == self.model_db_plate[select_plate1[idx]]["name"]:
                new_select_plate_ids.append(self.select_plate1_ids[idx])
            elif receptacle[idx] == self.model_db_plate[select_plate2[idx]]["name"]:
                new_select_plate_ids.append(self.select_plate2_ids[idx])
            else:
                raise ValueError(f"{receptacle[idx]} does not exist!")

        self.select_carrot_ids = torch.stack(new_select_carrot_ids)
        self.select_plate_ids = torch.stack(new_select_plate_ids)

        # for motion planning capability
        select_carrot = [self.carrot_names[idx] for idx in self.select_carrot_ids]
        select_plate = [self.plate_names[idx] for idx in self.select_plate_ids]
        carrot_actor = [self.objs_carrot[n] for n in select_carrot]
        plate_actor = [self.objs_plate[n] for n in select_plate]
        self.source_obj_name = select_carrot
        self.target_obj_name = select_plate
        self.objs = {
            self.source_obj_name[0]: carrot_actor[0],
            self.target_obj_name[0]: plate_actor[0]
        }

    def object_name(self):
        """
        Get all object names in env.
        """
        select_carrot = [self.carrot_names[idx] for idx in self.select_carrot_ids]

        name_in_all_envs = []
        for idx in range(self.num_envs):
            name_in_cur_envs = []
            name_in_cur_envs.append(f"{self.model_db_carrot[select_carrot[idx]]['name']}")
            name_in_all_envs.append(name_in_cur_envs)

        return name_in_all_envs


    def receptacle_name(self):
        """
        Get all receptacle names in env.
        """
        select_plate1 = [self.plate_names[idx] for idx in self.select_plate1_ids]
        select_plate2 = [self.plate_names[idx] for idx in self.select_plate2_ids]

        name_in_all_envs = []
        for idx in range(self.num_envs):
            name_in_cur_envs = []
            name_in_cur_envs.append(f"{self.model_db_plate[select_plate1[idx]]['name']}")
            name_in_cur_envs.append(f"{self.model_db_plate[select_plate2[idx]]['name']}")
            name_in_all_envs.append(name_in_cur_envs)

        return name_in_all_envs

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        self._initialize_episode_pre(env_idx, options)

        # Initial set plate 1 as task receptacle, plate 2 as extra receptacle
        self.select_plate_ids = self.select_plate1_ids
        self.select_extra_ids = self.select_plate2_ids

        b = self.num_envs

        # rgb overlay
        sensor = self._sensor_configs[self.rgb_camera_name]
        assert sensor.width == 640
        assert sensor.height == 480
        overlay_images = np.stack([self.overlay_images_numpy[idx] for idx in self.select_overlay_ids])
        self.overlay_images = torch.tensor(overlay_images, device=self.device)  # [b, H, W, 3]
        overlay_textures = np.stack([self.overlay_textures_numpy[idx] for idx in self.select_overlay_ids])
        self.overlay_textures = torch.tensor(overlay_textures, device=self.device)  # [b, H, W, 3]
        overlay_mix = np.array([self.overlay_mix_numpy[idx] for idx in self.select_overlay_ids])
        self.overlay_mix = torch.tensor(overlay_mix, device=self.device)  # [b]

        # xyz and quat
        xyz_configs = torch.tensor(self.xyz_configs, device=self.device)
        quat_configs = torch.tensor(self.quat_configs, device=self.device)

        select_carrot = [self.carrot_names[idx] for idx in self.select_carrot_ids]
        select_plate = [self.plate_names[idx] for idx in self.select_plate_ids]
        select_extra = [self.plate_names[idx] for idx in self.select_extra_ids]
        carrot_actor = [self.objs_carrot[n] for n in select_carrot]
        plate_actor = [self.objs_plate[n] for n in select_plate]
        extra_actor = [self.objs_plate[n] for n in select_extra]

        # for motion planning capability
        self.source_obj_name = select_carrot
        self.target_obj_name = select_plate
        self.objs = {
            self.source_obj_name[0]: carrot_actor[0],
            self.target_obj_name[0]: plate_actor[0]
        }

        # set pose for robot
        self.agent.robot.set_pose(self.safe_robot_pos)
        # self._settle(0.5)

        # set pose for objs
        for idx, name in enumerate(self.model_db_carrot):
            p_reset = torch.tensor([1.0, 0.3 * idx, 1.0], device=self.device).reshape(1, -1).repeat(b, 1)  # [b, 3]
            is_select = self.select_carrot_ids == idx  # [b]
            p_select = xyz_configs[self.select_pos_ids, 0].reshape(b, 3)  # [b, 3]
            p = torch.where(is_select.unsqueeze(1).repeat(1, 3), p_select, p_reset)  # [b, 3]

            q_reset = torch.tensor([0, 0, 0, 1], device=self.device).reshape(1, -1).repeat(b, 1)  # [b, 4]
            q_select = quat_configs[self.select_quat_ids, 0].reshape(b, 4)  # [b, 4]
            q = torch.where(is_select.unsqueeze(1).repeat(1, 4), q_select, q_reset)  # [b, 4]

            self.objs_carrot[name].set_pose(Pose.create_from_pq(p=p, q=q))

        for idx, name in enumerate(self.model_db_plate):
            p_reset = torch.tensor([2.0, 0.3 * idx, 1.0], device=self.device).reshape(1, -1).repeat(b, 1)  # [b, 3]
            is_select = self.select_plate_ids == idx  # [b]
            p_select = xyz_configs[self.select_pos_ids, 1].reshape(b, 3)  # [b, 3]
            is_select_extra = self.select_extra_ids == idx  # [b]
            p_select_extra = xyz_configs[self.select_pos_ids, 2].reshape(b, 3)  # [b, 3]
            p = torch.where(is_select.unsqueeze(1).repeat(1, 3), p_select, p_reset)  # [b, 3]
            p = torch.where(is_select_extra.unsqueeze(1).repeat(1, 3), p_select_extra, p)  # [b, 3]

            q_reset = torch.tensor([0, 0, 0, 1], device=self.device).reshape(1, -1).repeat(b, 1)  # [b, 4]
            q_select = quat_configs[self.select_quat_ids, 1].reshape(b, 4)  # [b, 4]
            q = torch.where(is_select.unsqueeze(1).repeat(1, 4), q_select, q_reset)  # [b, 4]
            q = torch.where(is_select_extra.unsqueeze(1).repeat(1, 4), q_select, q)

            self.objs_plate[name].set_pose(Pose.create_from_pq(p=p, q=q))

        self._settle(0.5)

        # Some objects need longer time to settle
        c_lin = torch.stack([a.linear_velocity[i] for i, a in enumerate(carrot_actor)])
        c_ang = torch.stack([a.angular_velocity[i] for i, a in enumerate(carrot_actor)])
        p_lin = torch.stack([a.linear_velocity[i] for i, a in enumerate(plate_actor)])
        p_ang = torch.stack([a.angular_velocity[i] for i, a in enumerate(plate_actor)])
        e_lin = torch.stack([a.linear_velocity[i] for i, a in enumerate(extra_actor)])
        e_ang = torch.stack([a.angular_velocity[i] for i, a in enumerate(extra_actor)])

        lin_vel = torch.linalg.norm(c_lin) + torch.linalg.norm(p_lin) + torch.linalg.norm(e_lin)
        ang_vel = torch.linalg.norm(c_ang) + torch.linalg.norm(p_ang) + torch.linalg.norm(e_ang)

        if lin_vel > 1e-3 or ang_vel > 1e-2:
            self._settle(6)

        # measured values for bridge dataset
        self.agent.robot.set_pose(self.initial_robot_pos)
        self.agent.reset(init_qpos=self.initial_qpos)

        # figure out object bounding boxes after settling. This is used to determine if an object is near the target object
        self.carrot_q_after_settle = torch.stack([a.pose.q[idx] for idx, a in enumerate(carrot_actor)])  # [b, 4]
        self.plate_q_after_settle = torch.stack([a.pose.q[idx] for idx, a in enumerate(plate_actor)])  # [b, 4]
        corner_signs = torch.tensor([
            [-1, -1, -1], [-1, -1, 1], [-1, 1, -1], [-1, 1, 1],
            [1, -1, -1], [1, -1, 1], [1, 1, -1], [1, 1, 1]
        ], device=self.device)

        # carrot
        carrot_bbox_world = torch.stack([self.model_bbox_sizes[n] for n in select_carrot])  # [b, 3]
        c_bbox_half = carrot_bbox_world / 2  # [b, 3]
        c_bbox_corners = c_bbox_half[:, None, :] * corner_signs[None, :, :]  # [b, 8, 3]

        c_q_matrix = rotation_conversions.quaternion_to_matrix(self.carrot_q_after_settle)  # [b, 3, 3]
        c_bbox_corners_rot = torch.matmul(c_bbox_corners, c_q_matrix.transpose(1, 2))  # [b, 8, 3]
        c_rotated_bbox_size = c_bbox_corners_rot.max(dim=1).values - c_bbox_corners_rot.min(dim=1).values  # [b, 3]
        self.carrot_bbox_world = c_rotated_bbox_size  # [b, 3]

        # plate
        plate_bbox_world = torch.stack([self.model_bbox_sizes[n] for n in select_plate])  # [b, 3]
        p_bbox_half = plate_bbox_world / 2  # [b, 3]
        p_bbox_corners = p_bbox_half[:, None, :] * corner_signs[None, :, :]  # [b, 8, 3]

        p_q_matrix = rotation_conversions.quaternion_to_matrix(self.plate_q_after_settle)  # [b, 3, 3]
        p_bbox_corners_rot = torch.matmul(p_bbox_corners, p_q_matrix.transpose(1, 2))  # [b, 8, 3]
        p_rotated_bbox_size = p_bbox_corners_rot.max(dim=1).values - p_bbox_corners_rot.min(dim=1).values  # [b, 3]
        self.plate_bbox_world = p_rotated_bbox_size  # [b, 3]

        # stats to track
        self.consecutive_grasp = torch.zeros((b,), dtype=torch.int32, device=self.device)
        self.episode_stats = dict(
            # all_obj_keep_height=torch.zeros((b,), dtype=torch.bool),
            # moved_correct_obj=torch.zeros((b,), dtype=torch.bool),
            # moved_wrong_obj=torch.zeros((b,), dtype=torch.bool),
            # near_tgt_obj=torch.zeros((b,), dtype=torch.bool),
            is_src_obj_grasped=torch.zeros((b,), dtype=torch.bool, device=self.device),
            # is_closest_to_tgt=torch.zeros((b,), dtype=torch.bool),
            consecutive_grasp=torch.zeros((b,), dtype=torch.bool, device=self.device),
            src_on_target=torch.zeros((b,), dtype=torch.bool, device=self.device),

            gripper_carrot_dist=torch.zeros((b,), dtype=torch.float32, device=self.device),
            gripper_plate_dist=torch.zeros((b,), dtype=torch.float32, device=self.device),
            carrot_plate_dist=torch.zeros((b,), dtype=torch.float32, device=self.device),
        )



class GenericNxMPickPlace(BaseMultiPickPlace):
    """Generalized base for arbitrary N objects × M receptacles.

    Subclasses only need to set class attributes:
        POSE_PRESET, NUM_OBJECTS, NUM_RECEPTACLES,
        DEFAULT_OBJ_INDICES, DEFAULT_PLATE_INDICES

    Optional:
        POSE_PRESET_OOD: str — OOD variant preset name
        SLOT_ORDER: list[int] — maps generic slot index to physical preset slot.
            Default: [0, 1, ..., N+M-1] (objects first, then receptacles).
            Override for envs where the preset uses a different slot ordering.
    """
    POSE_PRESET: str
    POSE_PRESET_OOD: str = ""
    NUM_OBJECTS: int
    NUM_RECEPTACLES: int
    DEFAULT_OBJ_INDICES: list
    DEFAULT_PLATE_INDICES: list
    SLOT_ORDER: list = None  # None = identity mapping

    def _prep_init(self):
        """Override to keep all plates (not filtered to 1), matching TwoObjectTwoReceptacle."""
        self.model_db_carrot = io_utils.load_json(CARROT_DATASET_DIR / "more_carrot" / "model_db.json")
        assert len(self.model_db_carrot) == 25
        self.model_db_plate = io_utils.load_json(CARROT_DATASET_DIR / "more_plate" / "model_db.json")
        self.carrot_names = list(self.model_db_carrot.keys())
        self.plate_names = list(self.model_db_plate.keys())

        model_db_table = io_utils.load_json(CARROT_DATASET_DIR / "more_table" / "model_db.json")
        img_fd = CARROT_DATASET_DIR / "more_table" / "imgs"
        texture_fd = CARROT_DATASET_DIR / "more_table" / "textures"
        self.overlay_images_numpy = [
            cv2.resize(cv2.cvtColor(cv2.imread(str(img_fd / k)), cv2.COLOR_BGR2RGB), (640, 480))
            for k in model_db_table
        ]
        self.overlay_textures_numpy = [
            cv2.resize(cv2.cvtColor(cv2.imread(str(texture_fd / v["texture"])), cv2.COLOR_BGR2RGB), (640, 480))
            for v in model_db_table.values()
        ]
        self.overlay_mix_numpy = [v["mix"] for v in model_db_table.values()]

    def _generate_init_pose(self):
        from .suite import generate_pose_configs, POSE_PRESETS, QUAT_CONFIGS
        params = POSE_PRESETS[self.POSE_PRESET]
        self.xyz_configs = generate_pose_configs(**params)
        self.quat_configs = QUAT_CONFIGS.copy()
        print(f"xyz_configs: {self.xyz_configs.shape}")
        print(f"quat_configs: {self.quat_configs.shape}")

    def _generate_OOD_init_pose(self):
        if not self.POSE_PRESET_OOD:
            self._generate_init_pose()
            return
        from .suite import generate_pose_configs, POSE_PRESETS, QUAT_CONFIGS
        params = POSE_PRESETS[self.POSE_PRESET_OOD]
        self.xyz_configs = generate_pose_configs(**params)
        self.quat_configs = QUAT_CONFIGS.copy()
        print(f"xyz_configs: {self.xyz_configs.shape}")
        print(f"quat_configs: {self.quat_configs.shape}")

    def get_carrot_actors(self):
        select_carrot = [self.carrot_names[idx] for idx in self.select_carrot_ids]
        return [self.objs_carrot[n] for n in select_carrot]

    def get_plate_actors(self):
        select_plate = [self.plate_names[idx] for idx in self.select_plate_ids]
        return [self.objs_plate[n] for n in select_plate]

    def get_extra_plate_actors(self):
        if self.NUM_RECEPTACLES > 1:
            sel = [self.plate_names[idx] for idx in self._all_plate_ids[1]]
            return [self.objs_plate[n] for n in sel]
        return self.get_plate_actors()

    def get_carrot_pose(self):
        p = torch.zeros((self.num_envs, 3), device=self.device)
        for name, actor in self.objs_carrot.items():
            mask = (self.select_carrot_ids == self.carrot_names.index(name))
            if mask.any(): p[mask] = actor.pose.p[mask]
        return p

    def get_plate_pose(self):
        p = torch.zeros((self.num_envs, 3), device=self.device)
        for name, actor in self.objs_plate.items():
            mask = (self.select_plate_ids == self.plate_names.index(name))
            if mask.any(): p[mask] = actor.pose.p[mask]
        return p

    def get_extra_plate_pose(self):
        if self.NUM_RECEPTACLES > 1:
            p = torch.zeros((self.num_envs, 3), device=self.device)
            for name, actor in self.objs_plate.items():
                mask = (self._all_plate_ids[1] == self.plate_names.index(name))
                if mask.any(): p[mask] = actor.pose.p[mask]
            return p
        return self.get_plate_pose()

    def _initialize_episode_pre(self, env_idx, options):
        b = len(env_idx)
        assert b == self.num_envs

        obj_set = options.get("obj_set", "rand")
        if obj_set == "rand_ood":
            self._generate_OOD_init_pose()
        else:
            self._generate_init_pose()

        lc = 16
        lo = len(self.overlay_images_numpy)
        l1 = len(self.xyz_configs)
        l2 = len(self.quat_configs)
        ltt = lc * 16 * lo * l1 * l2

        if "episode_id" in options:
            episode_id = options["episode_id"]
        else:
            single_id = torch.randint(low=0, high=ltt, size=(1,), device=self.device).item()
            episode_id = torch.full((b,), single_id, device=self.device) % ltt

        self._all_carrot_ids = []
        for i in range(self.NUM_OBJECTS):
            idx = options.get(f"obj{i+1}_index", self.DEFAULT_OBJ_INDICES[i]) - 1
            ids = torch.full((b,), idx, device=self.device)
            setattr(self, f"select_carrot{i+1}_ids", ids)
            self._all_carrot_ids.append(ids)

        self._all_plate_ids = []
        for i in range(self.NUM_RECEPTACLES):
            idx = options.get(f"plate{i+1}_index", self.DEFAULT_PLATE_INDICES[i]) - 1
            ids = torch.full((b,), idx, device=self.device)
            setattr(self, f"select_plate{i+1}_ids", ids)
            self._all_plate_ids.append(ids)

        self.select_overlay_ids = (episode_id // (l1 * l2)) % lo
        self.select_pos_ids = (episode_id // l2) % l1
        self.select_quat_ids = episode_id % l2
        if obj_set != "fixed":
            if obj_set != "rand_8":
                rand_id = torch.randint(low=0, high=ltt, size=(b,), device=self.device)
            else:
                rand_id = torch.randint(low=0, high=ltt, size=(b // 8,), device=self.device)
                rand_id = rand_id.repeat(8)
            self.select_pos_ids = (rand_id // l2) % l1
            self.select_quat_ids = rand_id % l2

    def _slot(self, logical_idx):
        """Maps a generic logical slot index to the physical preset slot index."""
        if self.SLOT_ORDER is not None:
            return self.SLOT_ORDER[logical_idx]
        return logical_idx

    def set_current_task(self, object, receptacle):
        new_carrot_ids = []
        for env_idx in range(self.num_envs):
            found = False
            for i in range(self.NUM_OBJECTS):
                cid = self._all_carrot_ids[i][env_idx]
                if object[env_idx] == self.model_db_carrot[self.carrot_names[cid]]["name"]:
                    new_carrot_ids.append(cid)
                    found = True
                    break
            if not found:
                raise ValueError(f"{object[env_idx]} not in available objects")

        new_plate_ids = []
        for env_idx in range(self.num_envs):
            found = False
            for i in range(self.NUM_RECEPTACLES):
                pid = self._all_plate_ids[i][env_idx]
                if receptacle[env_idx] == self.model_db_plate[self.plate_names[pid]]["name"]:
                    new_plate_ids.append(pid)
                    found = True
                    break
            if not found:
                raise ValueError(f"{receptacle[env_idx]} not in available receptacles")

        self.select_carrot_ids = torch.stack(new_carrot_ids)
        self.select_plate_ids = torch.stack(new_plate_ids)

        select_carrot = [self.carrot_names[idx] for idx in self.select_carrot_ids]
        select_plate = [self.plate_names[idx] for idx in self.select_plate_ids]
        self.source_obj_name = select_carrot
        self.target_obj_name = select_plate
        self.objs = {
            self.source_obj_name[0]: self.objs_carrot[select_carrot[0]],
            self.target_obj_name[0]: self.objs_plate[select_plate[0]]
        }

    def object_name(self):
        result = []
        for env_idx in range(self.num_envs):
            names = [self.model_db_carrot[self.carrot_names[self._all_carrot_ids[i][env_idx]]]["name"]
                     for i in range(self.NUM_OBJECTS)]
            result.append(names)
        return result

    def receptacle_name(self):
        result = []
        for env_idx in range(self.num_envs):
            names = [self.model_db_plate[self.plate_names[self._all_plate_ids[i][env_idx]]]["name"]
                     for i in range(self.NUM_RECEPTACLES)]
            result.append(names)
        return result

    def _initialize_episode(self, env_idx, options):
        self._initialize_episode_pre(env_idx, options)

        self.select_carrot_ids = self._all_carrot_ids[0]
        self.select_plate_ids = self._all_plate_ids[0]

        b = self.num_envs

        # RGB overlay
        sensor = self._sensor_configs[self.rgb_camera_name]
        assert sensor.width == 640 and sensor.height == 480
        self.overlay_images = torch.tensor(
            np.stack([self.overlay_images_numpy[idx] for idx in self.select_overlay_ids]), device=self.device)
        self.overlay_textures = torch.tensor(
            np.stack([self.overlay_textures_numpy[idx] for idx in self.select_overlay_ids]), device=self.device)
        self.overlay_mix = torch.tensor(
            np.array([self.overlay_mix_numpy[idx] for idx in self.select_overlay_ids]), device=self.device)

        xyz_configs = torch.tensor(self.xyz_configs, device=self.device)
        quat_configs = torch.tensor(self.quat_configs, device=self.device)

        select_carrot = [self.carrot_names[idx] for idx in self.select_carrot_ids]
        select_plate = [self.plate_names[idx] for idx in self.select_plate_ids]
        carrot_actor = [self.objs_carrot[n] for n in select_carrot]
        plate_actor = [self.objs_plate[n] for n in select_plate]

        self.source_obj_name = select_carrot
        self.target_obj_name = select_plate
        self.objs = {select_carrot[0]: carrot_actor[0], select_plate[0]: plate_actor[0]}

        self.agent.robot.set_pose(self.safe_robot_pos)

        # Place all carrots (logical slots 0..NUM_OBJECTS-1)
        for db_idx, name in enumerate(self.model_db_carrot):
            p_reset = torch.tensor([1.0, 0.3 * db_idx, 1.0], device=self.device).reshape(1, -1).repeat(b, 1)
            q_reset = torch.tensor([0, 0, 0, 1], device=self.device).reshape(1, -1).repeat(b, 1).float()
            p, q = p_reset, q_reset
            for slot in range(self.NUM_OBJECTS):
                is_this = self._all_carrot_ids[slot] == db_idx
                p_slot = xyz_configs[self.select_pos_ids, self._slot(slot)].reshape(b, 3)
                q_slot = quat_configs[self.select_quat_ids, 0].reshape(b, 4)
                p = torch.where(is_this.unsqueeze(1).expand_as(p), p_slot, p)
                q = torch.where(is_this.unsqueeze(1).expand_as(q), q_slot, q)
            self.objs_carrot[name].set_pose(Pose.create_from_pq(p=p, q=q))

        # Place all plates (logical slots NUM_OBJECTS..NUM_OBJECTS+NUM_RECEPTACLES-1)
        for db_idx, name in enumerate(self.model_db_plate):
            p_reset = torch.tensor([2.0, 0.3 * db_idx, 1.0], device=self.device).reshape(1, -1).repeat(b, 1)
            q_reset = torch.tensor([0, 0, 0, 1], device=self.device).reshape(1, -1).repeat(b, 1).float()
            p, q = p_reset, q_reset
            for slot in range(self.NUM_RECEPTACLES):
                is_this = self._all_plate_ids[slot] == db_idx
                p_slot = xyz_configs[self.select_pos_ids, self._slot(self.NUM_OBJECTS + slot)].reshape(b, 3)
                q_slot = quat_configs[self.select_quat_ids, 1].reshape(b, 4)
                p = torch.where(is_this.unsqueeze(1).expand_as(p), p_slot, p)
                q = torch.where(is_this.unsqueeze(1).expand_as(q), q_slot, q)
            self.objs_plate[name].set_pose(Pose.create_from_pq(p=p, q=q))

        self._settle(0.5)

        # Settle check
        lin_vel = torch.tensor(0.0, device=self.device)
        ang_vel = torch.tensor(0.0, device=self.device)
        for slot_ids_list, db in [(self._all_carrot_ids, self.objs_carrot),
                                   (self._all_plate_ids, self.objs_plate)]:
            db_names = list(db.keys())
            for slot_ids in slot_ids_list:
                names = [db_names[idx] for idx in slot_ids]
                actors = [db[n] for n in names]
                lin_vel = lin_vel + torch.linalg.norm(torch.stack([a.linear_velocity[i] for i, a in enumerate(actors)]))
                ang_vel = ang_vel + torch.linalg.norm(torch.stack([a.angular_velocity[i] for i, a in enumerate(actors)]))
        if lin_vel > 1e-3 or ang_vel > 1e-2:
            self._settle(6)

        self.agent.robot.set_pose(self.initial_robot_pos)
        self.agent.reset(init_qpos=self.initial_qpos)

        # Bounding boxes
        self.carrot_q_after_settle = torch.stack([a.pose.q[idx] for idx, a in enumerate(carrot_actor)])
        self.plate_q_after_settle = torch.stack([a.pose.q[idx] for idx, a in enumerate(plate_actor)])
        corner_signs = torch.tensor([
            [-1,-1,-1],[-1,-1,1],[-1,1,-1],[-1,1,1],[1,-1,-1],[1,-1,1],[1,1,-1],[1,1,1]
        ], device=self.device)

        carrot_bbox = torch.stack([self.model_bbox_sizes[n] for n in select_carrot])
        c_corners = (carrot_bbox / 2)[:, None, :] * corner_signs[None, :, :]
        c_rot = torch.matmul(c_corners, rotation_conversions.quaternion_to_matrix(self.carrot_q_after_settle).transpose(1, 2))
        self.carrot_bbox_world = c_rot.max(dim=1).values - c_rot.min(dim=1).values

        plate_bbox = torch.stack([self.model_bbox_sizes[n] for n in select_plate])
        p_corners = (plate_bbox / 2)[:, None, :] * corner_signs[None, :, :]
        p_rot = torch.matmul(p_corners, rotation_conversions.quaternion_to_matrix(self.plate_q_after_settle).transpose(1, 2))
        self.plate_bbox_world = p_rot.max(dim=1).values - p_rot.min(dim=1).values

        self.consecutive_grasp = torch.zeros((b,), dtype=torch.int32, device=self.device)
        self.episode_stats = dict(
            is_src_obj_grasped=torch.zeros((b,), dtype=torch.bool, device=self.device),
            consecutive_grasp=torch.zeros((b,), dtype=torch.bool, device=self.device),
            src_on_target=torch.zeros((b,), dtype=torch.bool, device=self.device),
            src_on_target2=torch.zeros((b,), dtype=torch.bool, device=self.device),
            src_on_table=torch.ones((b,), dtype=torch.bool, device=self.device),
            gripper_carrot_dist=torch.zeros((b,), dtype=torch.float32, device=self.device),
            gripper_plate_dist=torch.zeros((b,), dtype=torch.float32, device=self.device),
            carrot_plate_dist=torch.zeros((b,), dtype=torch.float32, device=self.device),
        )



@register_env("TwoObjectTwoReceptacle-v1", max_episode_steps=80, asset_download_ids=["bridge_v2_real2sim"])
class TwoObjectTwoReceptacle(GenericNxMPickPlace):
    POSE_PRESET = "TwoObjectTwoReceptacle"
    POSE_PRESET_OOD = "TwoObjectTwoReceptacle_OOD"
    NUM_OBJECTS = 2
    NUM_RECEPTACLES = 2
    DEFAULT_OBJ_INDICES = [7, 2]
    DEFAULT_PLATE_INDICES = [1, 2]



@register_env("ThreeObjectThreeReceptacle-v1", max_episode_steps=80, asset_download_ids=["bridge_v2_real2sim"])
class ThreeObjectThreeReceptacle(GenericNxMPickPlace):
    POSE_PRESET = "ThreeObjectThreeReceptacle"
    POSE_PRESET_OOD = "ThreeObjectThreeReceptacle_OOD"
    NUM_OBJECTS = 3
    NUM_RECEPTACLES = 3
    DEFAULT_OBJ_INDICES = [7, 2, 10]
    DEFAULT_PLATE_INDICES = [1, 2, 3]



@register_env("ThreeObjectOneReceptacle-v1", max_episode_steps=80, asset_download_ids=["bridge_v2_real2sim"])
class ThreeObjectOneReceptacle(GenericNxMPickPlace):
    POSE_PRESET = "ThreeObjectOneReceptacle"
    NUM_OBJECTS = 3
    NUM_RECEPTACLES = 1
    DEFAULT_OBJ_INDICES = [7, 2, 10]
    DEFAULT_PLATE_INDICES = [1]


@register_env("OneObjectThreeReceptacle-v1", max_episode_steps=80, asset_download_ids=["bridge_v2_real2sim"])
class OneObjectThreeReceptacle(GenericNxMPickPlace):
    POSE_PRESET = "OneObjectThreeReceptacle"
    NUM_OBJECTS = 1
    NUM_RECEPTACLES = 3
    DEFAULT_OBJ_INDICES = [7]
    DEFAULT_PLATE_INDICES = [1, 2, 3]


@register_env("ThreeObjectTwoReceptacle-v1", max_episode_steps=80, asset_download_ids=["bridge_v2_real2sim"])
class ThreeObjectTwoReceptacle(GenericNxMPickPlace):
    POSE_PRESET = "ThreeObjectTwoReceptacle"
    NUM_OBJECTS = 3
    NUM_RECEPTACLES = 2
    DEFAULT_OBJ_INDICES = [7, 2, 10]
    DEFAULT_PLATE_INDICES = [1, 2]


@register_env("TwoObjectThreeReceptacle-v1", max_episode_steps=80, asset_download_ids=["bridge_v2_real2sim"])
class TwoObjectThreeReceptacle(GenericNxMPickPlace):
    POSE_PRESET = "TwoObjectThreeReceptacle"
    NUM_OBJECTS = 2
    NUM_RECEPTACLES = 3
    DEFAULT_OBJ_INDICES = [7, 2]
    DEFAULT_PLATE_INDICES = [1, 2, 3]
