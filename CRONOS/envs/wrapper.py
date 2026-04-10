import gymnasium as gym
import numpy as np
import torch
from .reset import ResetStrategy
from .reward import RewardShaper

class CronosWrapper:
    """Unified environment wrapper for CRONOS, integrating decoupled modules."""
    
    def __init__(self, args, unnorm_state, task_suite, device=None, task_scheduler=None):
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
        self.env = gym.make(**env_config)
        
        import random
        random.seed(self.args.seed)
        self.rand_episode_id = random.randint(0, 1000)
        
        # AutoRL compatibility: Explicitly seed the env with an initial reset.
        # Pass obj_set so the task pool built from this reset matches training.
        options = {
            "obj_set": self.args.obj_set,
            "episode_id": torch.full((self.num_envs,), self.rand_episode_id, dtype=torch.long, device=self.device),
        }
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

    def reset(self, same_init=True, obj_set_override=None, **kwargs):
        """Unified reset matching AutoRL: env.reset → set_task → zero reward_old."""
        options = {
            "obj_set": obj_set_override if obj_set_override else self.args.obj_set,
        }
        if same_init:
            options["episode_id"] = torch.full((self.num_envs,), getattr(self, 'rand_episode_id', self.args.seed), dtype=torch.long, device=self.device)

        obs, info = self.env.reset(options=options)

        # Set tasks from scheduler if available
        if self.scheduler:
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

    def get_obs_instruct_info(self):
        """Returns current obs, instructions, and info without resetting the env."""
        obs = self.get_obs_image()
        instruct = self.get_language_instructions()
        info = self.env.unwrapped.get_info()
        return obs, instruct, info
