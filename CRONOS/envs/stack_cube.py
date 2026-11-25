"""CRONOS — StackCubeBridge-v1: stack one coloured cube on another.

Same stage as ``PickPlaceNxM-v1`` (Bridge flat table, WidowX 250S, the
3rd-view camera the VLAs were trained on, green-screened with one of the 21
``more_table`` backgrounds), but the objects are 3 cm cubes instead of
carrots / plates.

Scene
-----
* One actor per entry of ``CUBE_COLORS`` is built in every env. Each env shows
  ``N`` of them (``cube_colors`` / ``cube{i}_index`` options); the rest are
  parked off the table, the same way ``PickPlaceNxM`` hides unused carrots.
* Cubes are procedural boxes (``CUBE_HALF_SIZE`` = 1.5 cm, the size of the
  Bridge ``green_cube_3cm`` / ``yellow_cube_3cm`` assets), so any colour in
  the table can be used.
* Initial xy comes from ``suite.generate_pose_configs`` on the 6x6 grid around
  the same workspace centre as the pick-place presets; yaw from ``CUBE_YAWS``.

Task
----
``set_current_task(objects, receptacles)`` takes colour names per env:
``objects[i]`` is the cube to pick (source), ``receptacles[i]`` the cube to
stack on (target). Instruction: ``"stack the <src> block on the <tgt> block"``
— the wording of SimplerEnv's ``StackGreenCubeOnYellowCubeBakedTexInScene-v1``
(Bridge eval), which both OpenVLA and SpatialVLA have been evaluated on.

``evaluate()`` returns the keys ``RewardShaper.compute_reward`` already reads
(``is_src_obj_grasped``, ``consecutive_grasp``, ``success``, ``src_on_table``)
plus the stacking-specific ones used by ``compute_dense_reward`` (ManiSkill
``StackCube-v1``'s staged reward, adapted to the WidowX gripper).

How this env is wired into training (task scheduler, LSR backward goal, HSR
rules, eval plan) is NOT decided yet; nothing outside this file imports it.
"""

from __future__ import annotations

import itertools
from typing import Dict, List, Optional, Sequence

import cv2
import numpy as np
import sapien
import torch
from transforms3d.euler import euler2quat

from mani_skill.utils import io_utils
from mani_skill.utils.registration import register_env
from mani_skill.utils.structs.actor import Actor
from mani_skill.utils.structs.pose import Pose
from mani_skill.utils.structs.types import SceneConfig, SimConfig

from .bridge_multi import BasePickPlace, CARROT_DATASET_DIR
from .base_env import BRIDGE_DATASET_ASSET_PATH
from .suite import GRID_POS_UNIT, generate_pose_configs

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# name -> RGBA. The name is what appears in the language instruction.
CUBE_COLORS: Dict[str, Sequence[float]] = {
    "green": (0.10, 0.60, 0.20, 1.0),
    "yellow": (0.95, 0.80, 0.10, 1.0),
    "red": (0.80, 0.10, 0.10, 1.0),
    "blue": (0.10, 0.25, 0.80, 1.0),
    "orange": (0.95, 0.45, 0.05, 1.0),
    "purple": (0.50, 0.15, 0.60, 1.0),
}
CUBE_NAMES: List[str] = list(CUBE_COLORS)

CUBE_HALF_SIZE = 0.015          # 3 cm cube, same as the Bridge cube assets
CUBE_DENSITY = 1000.0           # -> 27 g
TABLE_TOP_Z = 0.8695            # measured: a settled 3 cm cube's centre is at 0.8845
SPAWN_Z = TABLE_TOP_Z + CUBE_HALF_SIZE + 0.002   # dropped 2 mm, then settled

# A cube is 90°-symmetric about z, so yaws in [0, 90°) cover every look.
CUBE_YAWS = (0.0, np.pi / 8, np.pi / 4, 3 * np.pi / 8)

# Initial layout: per-N preset on the standard 6x6 grid, same workspace
# centre as the pick-place presets. 0.06 m spacing (2 cube widths) leaves room
# for the open gripper (~7.4 cm) between cubes.
#   in-domain   (obj_set != "rand_ood"): half extent (0.075, 0.075), as
#               TwoObjectTwoReceptacle
#   out-of-domain (obj_set == "rand_ood", main.py's OOD eval): (0.11, 0.15),
#               as TwoObjectTwoReceptacle_OOD — the same grid stretched, so
#               most OOD placements fall outside or between the ID grid points.
ID_HALF_EXTENT = (0.075, 0.075)
OOD_HALF_EXTENT = (0.11, 0.15)


def _pose_preset(n: int, half_extent) -> dict:
    return dict(
        slot_heights=[SPAWN_Z] * n,
        half_edge_length=half_extent,
        spacing_matrix=[[0.06 if i != j else 0.0 for j in range(n)] for i in range(n)],
        grid=GRID_POS_UNIT,
    )


STACK_POSE_PRESETS = {n: _pose_preset(n, ID_HALF_EXTENT) for n in (2, 3, 4)}
STACK_POSE_PRESETS_OOD = {n: _pose_preset(n, OOD_HALF_EXTENT) for n in (2, 3, 4)}
DEFAULT_CUBES = {2: ["green", "yellow"], 3: ["green", "yellow", "red"],
                 4: ["green", "yellow", "red", "blue"]}

# Hidden cubes rest on a static shelf off the table (out of camera view). With
# nothing under them they would free-fall for the whole run.
PARK_X = 3.0
PARK_Z = 1.0 + CUBE_HALF_SIZE + 0.001


def stack_instruction(src: str, tgt: str) -> str:
    return f"stack the {src} block on the {tgt} block"


# ---------------------------------------------------------------------------
# CronosWrapper hooks — everything the wrapper needs to know about this env
# lives here, so wrapper.py only dispatches on ENV_ID.
# ---------------------------------------------------------------------------

ENV_ID = "StackCubeBridge-v1"


def gym_kwargs(args) -> dict:
    """Constructor kwargs from main.py Args; rejects modes this env lacks.

    Every YAML `obj` / `recep` entry is a cube, so N = env_n + env_m.
    HSR (`--reset-unsuitable`) needs `get_obj_pos` / `reset_unsuitable_envs`
    and LSR (`--enable-backward`) needs `_all_plate_ids`; both are carrot /
    plate machinery not implemented here. Fail before the sim is built
    instead of at the first segment boundary.
    """
    missing = [flag for flag, on in (("--reset-unsuitable (HSR)", getattr(args, "reset_unsuitable", False)),
                                     ("--enable-backward (LSR)", getattr(args, "enable_backward", False)))
               if on]
    if missing:
        raise NotImplementedError(f"{ENV_ID} does not support {', '.join(missing)} yet; "
                                  f"use reset mode `normal` (EER on/off both work).")
    return {"N": getattr(args, "env_n", 1) + getattr(args, "env_m", 1)}


def seed_reset_options(args) -> dict:
    """Options for the wrapper's seeding reset, which runs before main.py
    hands it the YAML groups. The global obj*/plate* Args are carrot / plate
    indices and mean nothing here: use cube colours 1..N instead (every later
    reset takes the YAML group's colours)."""
    env_n, env_m = getattr(args, "env_n", 1), getattr(args, "env_m", 1)
    opts = {f"obj{i + 1}_index": i + 1 for i in range(env_n)}
    opts.update({f"plate{j + 1}_index": env_n + j + 1 for j in range(env_m)})
    return opts


# ---------------------------------------------------------------------------
# Env
# ---------------------------------------------------------------------------

@register_env(ENV_ID, max_episode_steps=80,
              asset_download_ids=["bridge_v2_real2sim"])
class StackCubeBridge(BasePickPlace):
    """``N`` coloured cubes per env (N in 2..4); stack ``src`` on ``tgt``.

    Usage::

        gym.make("StackCubeBridge-v1", num_envs=64, N=2,
                 obs_mode="rgb+segmentation", sim_backend="gpu")

    Reset options (all optional):
        cube_colors        list[str] of length N, same for every env
        cube{i}_index      int or per-env LongTensor, 0-based into CUBE_NAMES
                           (overrides cube_colors for slot i)
        obj{i}_index / plate{j}_index
                           what CronosWrapper sends: 1-based into CUBE_NAMES,
                           objects first, then receptacles (= the YAML group's
                           `obj` / `recep`). Their count must equal N.
        layout_ids         per-env int64 (envs/rng_streams.py): layout + yaw
        select_overlay_ids per-env int: background image
        obj_set            "rand_ood" -> out-of-domain layout table (wider
                           grid); "fixed" -> layout from episode_id; else
                           in-domain random
    After a reset the task is slot 0 on slot 1 until `set_current_task`.
    """

    def __init__(self, *args, N: int = 2, **kwargs):
        if N not in STACK_POSE_PRESETS:
            raise ValueError(f"StackCubeBridge-v1: N must be one of "
                             f"{sorted(STACK_POSE_PRESETS)}, got {N}")
        self.NUM_CUBES = N
        self._n_obj_slots = 1           # slots [0, n) are "objects", the rest "receptacles"
        self._scene_spec_arg = kwargs.pop("scene_spec", None)
        # Name tables in the shape main.py / envs.config read to resolve a YAML
        # group's 1-based `obj` / `recep` indices (build_obj_recep_name_maps):
        # both index CUBE_NAMES, so `obj: [1]`, `recep: [2]` = green on yellow.
        self.model_db_carrot = {n: {"name": n} for n in CUBE_NAMES}
        self.model_db_plate = self.model_db_carrot
        self.carrot_names = self.plate_names = CUBE_NAMES
        self._load_overlays()
        super().__init__(*args, **kwargs)

    # --- construction -----------------------------------------------------

    @property
    def _default_sim_config(self):
        # Same rates as PickPlaceNxM, more PhysX solver iterations (defaults
        # 15 / 1). With the defaults ~40% of cube-on-cube stacks kept rocking
        # (|v| > 1 cm/s after 1.2 s, GPU sim, 64 envs); with 30 / 4, 1/64.
        return SimConfig(sim_freq=500, control_freq=5, spacing=20,
                         scene_config=SceneConfig(solver_position_iterations=30,
                                                  solver_velocity_iterations=4))

    def _load_overlays(self):
        # Same 21 table backgrounds as PickPlaceNxM (GenericNxMPickPlace._prep_init).
        model_db_table = io_utils.load_json(CARROT_DATASET_DIR / "more_table" / "model_db.json")
        img_fd = CARROT_DATASET_DIR / "more_table" / "imgs"
        texture_fd = CARROT_DATASET_DIR / "more_table" / "textures"
        rd = lambda p: cv2.resize(cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB), (640, 480))
        self.overlay_images_numpy = [rd(img_fd / k) for k in model_db_table]
        self.overlay_textures_numpy = [rd(texture_fd / v["texture"]) for v in model_db_table.values()]
        self.overlay_mix_numpy = [v["mix"] for v in model_db_table.values()]

    def _generate_init_pose(self, domain: str = "id"):
        """Select the ID or OOD layout table (each built once and cached:
        N=4 enumerates 36^4 grid combos)."""
        cache = self.__dict__.setdefault("_pose_cache", {})
        if domain not in cache:
            preset = (STACK_POSE_PRESETS_OOD if domain == "ood" else STACK_POSE_PRESETS)[self.NUM_CUBES]
            cache[domain] = generate_pose_configs(**preset)
        self.pose_domain = domain
        self.xyz_configs = cache[domain]
        self.quat_configs = np.stack([euler2quat(0, 0, y) for y in CUBE_YAWS])  # [Q, 4] wxyz

    def _load_scene(self, options: dict):
        from mani_skill.utils import sapien_utils
        for i in range(self.num_envs):
            sapien_utils.set_articulation_render_material(
                self.agent.robot._objs[i], specular=0.9, roughness=0.3)

        builder = self.scene.create_actor_builder()
        scene_pose = sapien.Pose(q=[0.707, 0.707, 0, 0])
        scene_offset = np.array([-2.0634, -2.8313, 0.0])
        scene_file = str(BRIDGE_DATASET_ASSET_PATH / "stages/bridge_table_1_v1.glb")
        builder.add_nonconvex_collision_from_file(scene_file, pose=scene_pose)
        builder.add_visual_from_file(scene_file, pose=scene_pose)
        builder.initial_pose = sapien.Pose(-scene_offset)
        builder.build_static(name="arena")

        material = sapien.physx.PhysxMaterial(
            static_friction=self.obj_static_friction,
            dynamic_friction=self.obj_dynamic_friction,
            restitution=0.0,
        )
        h = [CUBE_HALF_SIZE] * 3
        self.objs_cube: Dict[str, Actor] = {}
        for k, (name, rgba) in enumerate(CUBE_COLORS.items()):
            b = self.scene.create_actor_builder()
            b.add_box_collision(half_size=h, material=material, density=CUBE_DENSITY)
            b.add_box_visual(half_size=h, material=sapien.render.RenderMaterial(
                base_color=list(rgba), roughness=0.6, metallic=0.0))
            b.initial_pose = sapien.Pose([PARK_X, 0.1 * k, PARK_Z])
            self.objs_cube[name] = b.build(name=f"cube_{name}")

        shelf = self.scene.create_actor_builder()
        shelf.add_box_collision(half_size=[0.2, 0.1 * len(CUBE_COLORS) + 0.1, 0.05])
        shelf.initial_pose = sapien.Pose([PARK_X, 0.05 * len(CUBE_COLORS), 0.95])  # top at z = 1.0
        shelf.build_static(name="park_shelf")

    # --- episode ------------------------------------------------------------

    def _cube_ids_from_options(self, options, b) -> List[torch.Tensor]:
        if "obj1_index" in options:
            return self._cube_ids_from_wrapper(options, b)
        self._n_obj_slots = 1
        colors = options.get("cube_colors", DEFAULT_CUBES[self.NUM_CUBES])
        if len(colors) != self.NUM_CUBES:
            raise ValueError(f"cube_colors needs {self.NUM_CUBES} names, got {colors}")
        ids = []
        for i in range(self.NUM_CUBES):
            val = options.get(f"cube{i}_index", CUBE_NAMES.index(colors[i]))
            if isinstance(val, torch.Tensor):
                t = val.reshape(b).long().to(self.device)
            else:
                t = torch.full((b,), int(val), dtype=torch.long, device=self.device)
            ids.append(t)
        stacked = torch.stack(ids, 1)                     # [b, N]
        if (stacked.sort(1).values.diff(dim=1) == 0).any():
            raise ValueError("the same cube colour is used twice in one env")
        return ids

    def _cube_ids_from_wrapper(self, options, b) -> List[torch.Tensor]:
        """CronosWrapper path: obj1.., plate1.. indices (1-based, 0 = hidden)."""
        def collect(prefix):
            vals, i = [], 1
            while f"{prefix}{i}_index" in options:
                vals.append(options[f"{prefix}{i}_index"]); i += 1
            return vals
        objs, receps = collect("obj"), collect("plate")
        if len(objs) + len(receps) != self.NUM_CUBES:
            raise ValueError(
                f"StackCubeBridge-v1(N={self.NUM_CUBES}) got {len(objs)} obj + "
                f"{len(receps)} recep slots. Use a YAML config with `groups:` whose "
                f"len(obj) + len(recep) == N (the wrapper derives N from it).")
        ids = []
        for v in objs + receps:
            t = (v.reshape(b).long().to(self.device) if isinstance(v, torch.Tensor)
                 else torch.full((b,), int(v), dtype=torch.long, device=self.device)) - 1
            if (t < 0).any() or (t >= len(CUBE_NAMES)).any():
                raise ValueError(f"cube index out of range 1..{len(CUBE_NAMES)} "
                                 f"(0 = hidden slot is not supported): {(t + 1).unique().tolist()}")
            ids.append(t)
        if (torch.stack(ids, 1).sort(1).values.diff(dim=1) == 0).any():
            raise ValueError("the same cube colour is used twice in one env")
        self._n_obj_slots = len(objs)
        return ids

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        b = len(env_idx)
        assert b == self.num_envs, "StackCubeBridge resets all envs at once"
        obj_set = options.get("obj_set", "rand")
        self._generate_init_pose("ood" if obj_set == "rand_ood" else "id")

        n_layout, n_yaw, n_bg = len(self.xyz_configs), len(self.quat_configs), len(self.overlay_images_numpy)
        total = n_layout * n_yaw * n_bg
        if "layout_ids" in options and obj_set != "fixed":
            rand_id = torch.as_tensor(options["layout_ids"], dtype=torch.long,
                                      device=self.device).reshape(b) % total
        elif obj_set == "fixed" and "episode_id" in options:
            rand_id = torch.as_tensor(options["episode_id"], device=self.device).reshape(b).long() % total
        else:
            rand_id = torch.randint(0, total, (b,), device=self.device)
        self.last_layout_rand_ids = rand_id
        self.select_pos_ids = (rand_id // n_yaw) % n_layout
        self.select_quat_ids = rand_id % n_yaw
        if "select_overlay_ids" in options:
            self.select_overlay_ids = torch.as_tensor(options["select_overlay_ids"],
                                                      device=self.device).reshape(b) % n_bg
        else:
            self.select_overlay_ids = (rand_id // (n_layout * n_yaw)) % n_bg

        self._all_cube_ids = self._cube_ids_from_options(options, b)
        self.select_src_ids = self._all_cube_ids[0]
        self.select_tgt_ids = self._all_cube_ids[1]

        # Background overlay
        sel = self.select_overlay_ids.tolist()
        self.overlay_images = torch.tensor(np.stack([self.overlay_images_numpy[i] for i in sel]), device=self.device)
        self.overlay_textures = torch.tensor(np.stack([self.overlay_textures_numpy[i] for i in sel]), device=self.device)
        self.overlay_mix = torch.tensor(np.array([self.overlay_mix_numpy[i] for i in sel]), device=self.device)

        # Place cubes: shown ones at their slot, the rest parked.
        xyz = torch.tensor(self.xyz_configs, device=self.device, dtype=torch.float32)
        quat = torch.tensor(self.quat_configs, device=self.device, dtype=torch.float32)
        self.agent.robot.set_pose(self.safe_robot_pos)
        for k, name in enumerate(CUBE_NAMES):
            p = torch.tensor([PARK_X, 0.1 * k, PARK_Z], device=self.device).repeat(b, 1)
            q = torch.tensor([1.0, 0, 0, 0], device=self.device).repeat(b, 1)
            for slot, ids in enumerate(self._all_cube_ids):
                m = (ids == k).unsqueeze(1)
                p = torch.where(m, xyz[self.select_pos_ids, slot], p)
                q = torch.where(m, quat[self.select_quat_ids], q)
            self.objs_cube[name].set_pose(Pose.create_from_pq(p=p, q=q))

        self._settle(0.5)

        self.agent.robot.set_pose(self.initial_robot_pos)
        self.agent.reset(init_qpos=self.initial_qpos)
        # Same GPU sync as GenericNxMPickPlace: the first rendered frame must
        # show the robot at its initial pose, not at safe_robot_pos.
        if self.gpu_sim_enabled:
            self.scene._gpu_apply_all()
            self.scene.px.gpu_update_articulation_kinematics()
            self.scene._gpu_fetch_all()

        self.consecutive_grasp = torch.zeros((b,), dtype=torch.int32, device=self.device)
        z = lambda dt=torch.bool: torch.zeros((b,), dtype=dt, device=self.device)
        self.episode_stats = dict(
            is_src_obj_grasped=z(), consecutive_grasp=z(), src_on_target=z(),
            src_on_table=torch.ones((b,), dtype=torch.bool, device=self.device),
            gripper_src_dist=z(torch.float32), src_tgt_dist=z(torch.float32),
        )

    # --- task ---------------------------------------------------------------

    def object_name(self) -> List[List[str]]:
        """Colours present in each env (any of them can be the source)."""
        ids = torch.stack(self._all_cube_ids, 1).tolist()
        return [[CUBE_NAMES[i] for i in row] for row in ids]

    receptacle_name = object_name   # any present cube can be the target

    def task_pool(self) -> List[List[str]]:
        return [[stack_instruction(a, c) for a, c in itertools.permutations(names, 2)]
                for names in self.object_name()]

    def set_current_task(self, object: List[str], receptacle: List[str]):
        present = self.object_name()
        for e, (s, t) in enumerate(zip(object, receptacle)):
            if s == t or s not in present[e] or t not in present[e]:
                raise ValueError(f"env {e}: cannot stack {s!r} on {t!r}; present {present[e]}")
        self.select_src_ids = torch.tensor([CUBE_NAMES.index(s) for s in object], device=self.device)
        self.select_tgt_ids = torch.tensor([CUBE_NAMES.index(t) for t in receptacle], device=self.device)

    def get_language_instruction(self) -> List[str]:
        return [stack_instruction(CUBE_NAMES[s], CUBE_NAMES[t])
                for s, t in zip(self.select_src_ids.tolist(), self.select_tgt_ids.tolist())]

    def get_all_slot_poses(self):
        """Per-slot poses for main.py's segment_pose.csv (same layout as
        GenericNxMPickPlace.get_all_slot_poses): object slots, receptacle
        slots, gripper."""
        out = {"obj": [], "recep": [], "obj_names": [], "recep_names": []}
        for slot, ids in enumerate(self._all_cube_ids):
            key = "obj" if slot < self._n_obj_slots else "recep"
            out[key].append((self._gather("p", ids), self._gather("q", ids)))
            out[f"{key}_names"].append([CUBE_NAMES[i] for i in ids.tolist()])
        g_p = (self.agent.finger1_link.pose.p + self.agent.finger2_link.pose.p) / 2
        g_q = (self.agent.finger1_link.pose.q + self.agent.finger2_link.pose.q) / 2
        out["gripper"] = (g_p, g_q)
        return out

    # --- evaluation -----------------------------------------------------------

    def _gather(self, field: str, ids: torch.Tensor) -> torch.Tensor:
        """Per-env value of `field` (pose.p, linear_velocity, ...) of cube ids[e]."""
        def get(a):
            if field in ("p", "q"):
                return getattr(a.pose, field)
            return getattr(a, field)
        allv = torch.stack([get(self.objs_cube[n]) for n in CUBE_NAMES])   # [C, b, d]
        return allv[ids, torch.arange(self.num_envs, device=self.device)]

    def _pair_contact_force(self, src_ids, tgt_ids) -> torch.Tensor:
        """|contact force| between each env's src and tgt cube, one physx query per pair in use."""
        out = torch.zeros(self.num_envs, device=self.device)
        pairs = torch.stack([src_ids, tgt_ids], 1)
        for s, t in torch.unique(pairs, dim=0).tolist():
            f = torch.linalg.norm(self.scene.get_pairwise_contact_forces(
                self.objs_cube[CUBE_NAMES[s]], self.objs_cube[CUBE_NAMES[t]]), dim=1)
            m = (src_ids == s) & (tgt_ids == t)
            out = torch.where(m, f, out)
        return out

    def _is_grasping_src(self) -> torch.Tensor:
        g = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        for k, name in enumerate(CUBE_NAMES):
            m = self.select_src_ids == k
            if m.any():
                g = torch.where(m, self.agent.is_grasping(self.objs_cube[name]), g)
        return g

    def evaluate(self):
        src_p = self._gather("p", self.select_src_ids)
        tgt_p = self._gather("p", self.select_tgt_ids)
        offset = src_p - tgt_p
        h = CUBE_HALF_SIZE

        # On-top test. xy: within the cube's half-diagonal (+3 mm), as
        # SimplerEnv's stack task (base_env._evaluate). z: resting one cube
        # height above, ±5 mm (ManiSkill StackCube-v1). Plus physical contact
        # (> 0.05 N, SimplerEnv), so a cube hovering in the gripper is not "on".
        xy_flag = torch.linalg.norm(offset[:, :2], dim=1) <= np.sqrt(2) * h + 0.003
        z_flag = torch.abs(offset[:, 2] - 2 * h) <= 0.005
        contact = self._pair_contact_force(self.select_src_ids, self.select_tgt_ids) > 0.05
        src_on_target = xy_flag & z_flag & contact

        is_grasped = self._is_grasping_src()
        self.consecutive_grasp += is_grasped
        self.consecutive_grasp[~is_grasped] = 0
        consecutive_grasp = self.consecutive_grasp >= 5

        lin_v = torch.linalg.norm(self._gather("linear_velocity", self.select_src_ids), dim=1)
        ang_v = torch.linalg.norm(self._gather("angular_velocity", self.select_src_ids), dim=1)
        src_static = (lin_v <= 1e-2) & (ang_v <= 0.5)   # ManiSkill's GPU-tolerant thresholds

        gripper_p = (self.agent.finger1_link.pose.p + self.agent.finger2_link.pose.p) / 2
        # Bottom face resting on the table (not on another cube, not dropped off).
        src_on_table = (~is_grasped) & (torch.abs(src_p[:, 2] - (TABLE_TOP_Z + h)) <= 0.01)

        self.episode_stats["src_on_target"] = src_on_target
        self.episode_stats["src_on_table"] = src_on_table
        self.episode_stats["is_src_obj_grasped"] = self.episode_stats["is_src_obj_grasped"] | is_grasped
        self.episode_stats["consecutive_grasp"] = self.episode_stats["consecutive_grasp"] | consecutive_grasp
        self.episode_stats["gripper_src_dist"] = torch.linalg.norm(gripper_p - src_p, dim=1)
        self.episode_stats["src_tgt_dist"] = torch.linalg.norm(offset, dim=1)

        self.extra_stats.update(extra_pos_src=src_p, extra_pos_tgt=tgt_p, extra_pos_gripper=gripper_p)

        return dict(
            **self.episode_stats,
            is_src_grasped_now=is_grasped,
            src_static=src_static,
            # SimplerEnv-comparable: on top with contact (may still be held).
            success=src_on_target,
            # ManiSkill StackCube-v1: on top, released and at rest.
            success_strict=src_on_target & (~is_grasped) & src_static,
        )

    def reset_grasp_stats(self):
        self.consecutive_grasp.zero_()
        self.episode_stats["is_src_obj_grasped"].zero_()
        self.episode_stats["consecutive_grasp"].zero_()

    # --- reward -------------------------------------------------------------

    def compute_dense_reward(self, obs=None, action=None, info: Optional[dict] = None):
        """ManiSkill StackCube-v1 staged dense reward, max 8, for the WidowX.

        reach   2 * (1 - tanh(5 |gripper - src|))                      [0, 2]
        grasped 4 + (1 - tanh(5 |src - (tgt + 2h z)|))                 [4, 5]
        on top  6 + (gripper openness + (1 - tanh(10 v + w))) / 2      [6, 7]
        success 8   (success_strict: on top, released, static)

        NOT the training reward: CronosWrapper uses envs/reward.py
        RewardShaper on `success`. Kept for comparison. Its top stage keys on
        `success_strict`, whose "static" test is noisy on the GPU sim (a
        released stack can keep rocking above 1 cm/s): in a scripted
        pick-and-stack, 11/16 envs stacked (`success`) but only 4/16 were
        `success_strict` 1.2 s after release.

        Differences from ManiSkill: the gripper is the finger midpoint (no
        `tcp` link on this WidowX), and openness is normalised by the WidowX
        finger range (0.015..0.037 m per finger) instead of Panda's.
        """
        info = info if info is not None else self.evaluate()
        src_p = self._gather("p", self.select_src_ids)
        tgt_p = self._gather("p", self.select_tgt_ids)
        gripper_p = (self.agent.finger1_link.pose.p + self.agent.finger2_link.pose.p) / 2

        reward = 2 * (1 - torch.tanh(5 * torch.linalg.norm(gripper_p - src_p, dim=1)))

        goal = tgt_p + torch.tensor([0, 0, 2 * CUBE_HALF_SIZE], device=self.device)
        place = 1 - torch.tanh(5 * torch.linalg.norm(goal - src_p, dim=1))
        grasped = info["is_src_grasped_now"]
        reward = torch.where(grasped, 4 + place, reward)

        finger = self.agent.robot.get_qpos()[:, -2:].sum(1)           # 0.030 closed .. 0.074 open
        openness = ((finger - 0.030) / (0.074 - 0.030)).clamp(0, 1)
        openness = torch.where(grasped, openness, torch.ones_like(openness))
        v = torch.linalg.norm(self._gather("linear_velocity", self.select_src_ids), dim=1)
        w = torch.linalg.norm(self._gather("angular_velocity", self.select_src_ids), dim=1)
        static = 1 - torch.tanh(10 * v + w)
        reward = torch.where(info["src_on_target"], 6 + (openness + static) / 2, reward)

        reward = torch.where(info["success_strict"], torch.full_like(reward, 8.0), reward)
        return reward

    def compute_normalized_dense_reward(self, obs=None, action=None, info=None):
        return self.compute_dense_reward(obs, action, info) / 8.0
