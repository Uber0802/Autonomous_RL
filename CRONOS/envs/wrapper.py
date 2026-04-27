import gymnasium as gym
import numpy as np
import torch
from .reset import ResetStrategy
from .reward import RewardShaper

class CronosWrapper:
    """Unified environment wrapper for CRONOS, integrating decoupled modules."""

    def __init__(self, args, unnorm_state, task_suite, device=None, task_scheduler=None,
                 group_specs=None):
        self.args = args
        self.unnorm_state = unnorm_state
        self.num_envs = args.num_envs
        self.device = device if device is not None else torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # Core Env
        env_config = dict(
            id=args.env_id,
            num_envs=args.num_envs,
            obs_mode="rgb+segmentation",
            control_mode="arm_pd_ee_target_delta_pose_align2_gripper_pd_joint_pos",
            sim_backend="gpu",
            sim_config={"sim_freq": 500, "control_freq": 5},
            max_episode_steps=args.segment_len,
            sensor_configs={"shader_pack": "default"},
        )
        # V0.2 M2 Phase B: PickPlaceNxM-v1 takes (N, M) construction kwargs.
        # gym.make forwards extra kwargs to the env constructor, so any future
        # parametric env can opt in by adding the same Args fields.
        if args.env_id == "PickPlaceNxM-v1":
            env_config["N"] = getattr(args, "env_n", 2)
            env_config["M"] = getattr(args, "env_m", 1)
        # V0.2 M4: pass named scene spec to env constructor
        scene_name = getattr(args, "scene", "")
        if scene_name:
            from .scenes import get_scene
            env_config["scene_spec"] = get_scene(scene_name)
        self.env = gym.make(**env_config)

        import random
        random.seed(self.args.seed)
        self.rand_episode_id = random.randint(0, 1000)

        # V0.3 M1: per-group object/receptacle indices → per-env tensors
        self.group_specs = group_specs  # List[GroupSpec] or None
        self._build_per_env_indices()

        # AutoRL compatibility: Explicitly seed the env with an initial reset.
        # Pass obj_set + obj/plate indices so the task pool matches training.
        options = self._build_options()
        options["episode_id"] = torch.full((self.num_envs,), self.rand_episode_id, dtype=torch.long, device=self.device)
        self.env.reset(seed=[self.args.seed * 1000 + i for i in range(self.args.num_envs)], options=options)
        
        # Integrated Modules
        self.suite = task_suite
        self.scheduler = task_scheduler
        self.reset_strategy = ResetStrategy(
            self.env,
            self.num_envs,
            self.device,
            detector=getattr(args, "unsuitable_detector", "low_z"),
        )
        self.reward_shaper = RewardShaper(self.num_envs, self.device)
        
        # Binning for action processing
        bins = np.linspace(-1, 1, 256)
        self.bin_centers = (bins[:-1] + bins[1:]) / 2.0

    def _build_per_env_indices(self):
        """V0.3 M1: Build per-env object/plate index tensors from group_specs.

        When group_specs is provided and groups have per-group obj/recep,
        creates per-env tensors. Otherwise falls back to global args scalars.
        """
        if not self.group_specs or len(self.group_specs) <= 1:
            # Single group or no group specs — use global scalar args (V0.2 path)
            self._per_env_obj = None
            self._per_env_plate = None
            return

        from envs.config import get_group_starts
        group_starts = get_group_starts(self.group_specs)
        env_n = getattr(self.args, "env_n", 2)
        env_m = getattr(self.args, "env_m", 1)

        # Per-env obj indices (1-based, 0 = hidden), shape (num_slots, num_envs)
        self._per_env_obj = []
        for slot in range(env_n):
            t = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
            for gi, g in enumerate(self.group_specs):
                start, end = group_starts[gi], group_starts[gi + 1]
                if slot < len(g.obj):
                    t[start:end] = g.obj[slot]
                else:
                    t[start:end] = 0  # V0.3 M2: hidden slot marker (0 → -1 after -1 in env)
            self._per_env_obj.append(t)

        # Per-env plate indices (1-based, 0 = hidden), shape (num_slots, num_envs)
        self._per_env_plate = []
        for slot in range(env_m):
            t = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
            for gi, g in enumerate(self.group_specs):
                start, end = group_starts[gi], group_starts[gi + 1]
                if slot < len(g.recep):
                    t[start:end] = g.recep[slot]
                else:
                    t[start:end] = 0  # V0.3 M2: hidden slot marker
            self._per_env_plate.append(t)

        # Per-env overlay indices (0-based) — only when ALL groups specify int background
        all_custom_bg = all(isinstance(g.background, int) for g in self.group_specs)
        if all_custom_bg:
            self._per_env_overlay = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
            for gi, g in enumerate(self.group_specs):
                start, end = group_starts[gi], group_starts[gi + 1]
                self._per_env_overlay[start:end] = g.background
        else:
            self._per_env_overlay = None

    def _build_options(self, obj_set_override=None, group_idx_override=None):
        """Build the env reset options dict with per-env or scalar indices.

        Args:
            obj_set_override: Override obj_set (e.g. "rand_ood" for OOD eval).
            group_idx_override: If set, use this group's obj/recep for ALL envs
                (scalar broadcast). Used during eval to ensure all envs have
                the correct physical objects for the group being evaluated.
        """
        options = {
            "obj_set": obj_set_override if obj_set_override else self.args.obj_set,
        }
        if group_idx_override is not None and self.group_specs:
            # V0.3 eval path: broadcast one group's indices to all envs
            g = self.group_specs[group_idx_override]
            env_n = getattr(self.args, "env_n", 2)
            env_m = getattr(self.args, "env_m", 1)
            for i in range(env_n):
                options[f"obj{i+1}_index"] = g.obj[i] if i < len(g.obj) else 0  # 0 = hidden
            for i in range(env_m):
                options[f"plate{i+1}_index"] = g.recep[i] if i < len(g.recep) else 0
            if isinstance(g.background, int):
                options["select_overlay_ids"] = torch.full(
                    (self.num_envs,), g.background, dtype=torch.long, device=self.device)
        elif self._per_env_obj is not None:
            # V0.3 training path: per-env tensors
            for i, t in enumerate(self._per_env_obj):
                options[f"obj{i+1}_index"] = t
            for i, t in enumerate(self._per_env_plate):
                options[f"plate{i+1}_index"] = t
            if self._per_env_overlay is not None:
                options["select_overlay_ids"] = self._per_env_overlay
        else:
            # V0.2: global scalars
            options["obj1_index"] = self.args.obj1_index
            options["obj2_index"] = self.args.obj2_index
            options["obj3_index"] = self.args.obj3_index
            options["plate1_index"] = self.args.plate1_index
            options["plate2_index"] = self.args.plate2_index
            options["plate3_index"] = self.args.plate3_index
        return options

    def _process_action(self, raw_actions: torch.Tensor) -> torch.Tensor:
        """Processes raw action tokens into executable continuous actions."""
        pact_token = raw_actions.cpu().numpy()
        dact = np.clip(32000 - pact_token - 1, a_min=0, a_max=254)
        normalized_actions = np.asarray([self.bin_centers[da] for da in dact])

        action_norm_stats = self.unnorm_state
        mask = np.asarray(action_norm_stats.get("mask", np.ones_like(action_norm_stats["q01"], dtype=bool))).reshape(1, -1)
        action_high = np.array(action_norm_stats["q99"]).reshape(1, -1)
        action_low = np.array(action_norm_stats["q01"]).reshape(1, -1)
        
        raw_action_np = np.where(
            mask,
            0.5 * (normalized_actions + 1) * (action_high - action_low) + action_low,
            normalized_actions,
        )

        action = torch.cat([
            torch.tensor(raw_action_np[:, :3]), # world_vector
            torch.tensor(raw_action_np[:, 3:6]), # rotation_delta
            2.0 * (torch.tensor(raw_action_np[:, 6:7]) > 0.5) - 1.0 # gripper
        ], dim=1).to(self.device)
        
        return action

    def step(self, raw_action):
        """Step matching AutoRL: use env's native info for reward and episode reporting."""
        action = self._process_action(raw_action)
        obs, _reward, _terminated, truncated, info = self.env.step(action)

        obs_image = obs["sensor_data"]["3rd_view_camera"]["rgb"].to(torch.uint8)
        truncated = truncated.reshape(-1, 1)

        # Reward from env's native info (AutoRL: self.get_reward(info))
        reward = self.reward_shaper.compute_reward(info)

        # Episode info at truncation: instantaneous values (matching AutoRL exactly)
        if truncated.any():
            info["episode"] = {}
            for k in ["is_src_obj_grasped", "consecutive_grasp", "success"]:
                v = [info[k][idx].item() for idx in range(self.num_envs)]
                info["episode"][k] = v

        return obs_image, reward, truncated, info

    def reset(self, same_init=True, obj_set_override=None, group_idx_override=None,
              skip_scheduler=False, **kwargs):
        """Unified reset matching AutoRL: env.reset → set_task → zero reward_old."""
        options = self._build_options(obj_set_override=obj_set_override,
                                      group_idx_override=group_idx_override)
        if same_init:
            options["episode_id"] = torch.full((self.num_envs,), getattr(self, 'rand_episode_id', self.args.seed), dtype=torch.long, device=self.device)

        obs, info = self.env.reset(options=options)

        # Set tasks from scheduler if available.
        # Skip when: eval with group_idx_override, or caller will set tasks explicitly.
        if self.scheduler and group_idx_override is None and not skip_scheduler:
            objs, receps = self.scheduler.get_next_tasks()
            self.env.unwrapped.set_current_task(objs, receps)

        # AutoRL alignment: only zero reward_old on reset
        self.reward_shaper.reward_old.zero_()
        obs_image = obs["sensor_data"]["3rd_view_camera"]["rgb"].to(torch.uint8)
        instruction = self.get_language_instructions()

        return obs_image, instruction, info

    def get_language_instructions(self):
        """Generates instructions based on current objects and receptacles."""
        default_instructions = self.env.unwrapped.get_language_instruction()
        instructions = []
        for i, instr in enumerate(default_instructions):
            if self.reward_shaper.backward[i]:
                # Non-episodic backward task logic: "put [object] on table"
                # Extract object name robustly from "put [object] on [receptacle]"
                if " on " in instr:
                    obj_name = instr.split("put ")[1].split(" on ")[0]
                else:
                    obj_name = "object"
                instructions.append(f"put {obj_name} on table")
            else:
                instructions.append(instr)
        return instructions

    def set_forward(self):
        """Sets all environments to forward mode.

        AutoRL alignment: only zero reward_old, not consecutive_grasp.
        """
        self.reward_shaper.set_backward_mask(torch.zeros(self.num_envs, dtype=torch.bool, device=self.device))
        self.reward_shaper.reward_old.zero_()

    def set_backward(self):
        """Sets all environments to backward mode.

        AutoRL alignment: only zero reward_old, not consecutive_grasp.
        """
        self.reward_shaper.set_backward_mask(torch.ones(self.num_envs, dtype=torch.bool, device=self.device))
        self.reward_shaper.reward_old.zero_()

    def reset_robot(self):
        """Performs a partial reset (robot only) for non-episodic transitions."""
        self.reset_strategy.reset_robot()
        return self.get_obs_image()

    def get_obs_image(self):
        """Directly retrieves the current observation image."""
        info = self.env.unwrapped.get_info()
        obs = self.env.unwrapped.get_obs(info)
        return obs["sensor_data"]["3rd_view_camera"]["rgb"].to(torch.uint8)

    def reset_unsuitable_envs(self):
        """Resets only the unsuitable environments."""
        self.reset_strategy.reset_unsuitable_envs()
        return self.get_obs_image()

    def set_task(self, objects, receptacles):
        """Manually sets the current task for all environments.

        AutoRL alignment: only zero reward_old on task switch.
        Do NOT reset consecutive_grasp — it persists across task boundaries.
        """
        self.env.unwrapped.set_current_task(objects, receptacles)
        self.reward_shaper.reward_old.zero_()

    def get_task_pool(self):
        """Exposes the internal environment's task pool."""
        return self.env.unwrapped.task_pool()[0]

    def set_scheduler(self, scheduler):
        """Assigns a TaskScheduler to the wrapper."""
        self.scheduler = scheduler

    def set_group_specs(self, group_specs):
        """V0.3 M1: Set per-group specs and rebuild per-env index tensors."""
        self.group_specs = group_specs
        self._build_per_env_indices()

    def get_obs_instruct_info(self):
        """Returns current obs, instructions, and info without resetting the env."""
        obs = self.get_obs_image()
        instruct = self.get_language_instructions()
        info = self.env.unwrapped.get_info()
        return obs, instruct, info
