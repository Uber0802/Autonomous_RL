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
        self.env = gym.make(**env_config)
        
        import random
        random.seed(self.args.seed)
        self.rand_episode_id = random.randint(0, 1000)
        
        # AutoRL compatibility: Explicitly seed the env with an initial reset
        options = {}
        options["episode_id"] = torch.full((self.num_envs,), self.rand_episode_id, dtype=torch.long, device=self.device)
        self.env.reset(seed=[self.args.seed * 1000 + i for i in range(self.args.num_envs)], options=options)
        
        # Integrated Modules
        self.suite = task_suite
        self.scheduler = task_scheduler
        self.reset_strategy = ResetStrategy(self.env, self.num_envs, self.device)
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
        """Unified step function using decoupled reward shaping."""
        action = self._process_action(raw_action)
        obs, _, _, truncated, info = self.env.step(action)
        
        obs_image = obs["sensor_data"]["3rd_view_camera"]["rgb"].to(torch.uint8)
        
        # Centralized info computation in RewardShaper
        info = self.reward_shaper.compute_info(self.env)
        reward = self.reward_shaper.compute_reward(info)
        
        truncated_reshaped = truncated.reshape(-1, 1)
        if truncated_reshaped.any():
            info["episode"] = {
                "success": info["success"].float().tolist(),
                "is_src_obj_grasped": self.reward_shaper.is_src_obj_grasped_ever.float().tolist(),
                "consecutive_grasp": self.reward_shaper.consecutive_grasp_ever.float().tolist(),
                "src_on_table": info["src_on_table"].float().tolist(),
                "dist_tcp_to_obj": info["gripper_carrot_dist"].tolist(),
                "dist_obj_to_recep": info["dist_obj_to_recep"].tolist()
            }
            
        return obs_image, reward, truncated_reshaped, info

    def reset(self, same_init=True, obj_set_override=None, **kwargs):
        """Unified reset function using suite and reset strategy."""
        options = {
            "obj_set": obj_set_override if obj_set_override else self.args.obj_set,
            "obj1_index": self.args.obj1_index,
            "obj2_index": self.args.obj2_index,
            "obj3_index": getattr(self.args, "obj3_index", 3),
            "plate1_index": self.args.plate1_index,
            "plate2_index": self.args.plate2_index,
            "plate3_index": getattr(self.args, "plate3_index", 3),
        }
        if same_init:
            options["episode_id"] = torch.full((self.num_envs,), getattr(self, 'rand_episode_id', self.args.seed), dtype=torch.long, device=self.device)
            
        obs, info = self.env.reset(options=options)
        
        # Set tasks from scheduler if available
        if self.scheduler:
            objs, receps = self.scheduler.get_next_tasks()
            self.env.unwrapped.set_current_task(objs, receps)
        
        self.reward_shaper.reset_reward_stats()
        # Compute rich info from reward_shaper so all keys exist at step 0
        info = self.reward_shaper.compute_info(self.env)
        obs_image = obs["sensor_data"]["3rd_view_camera"]["rgb"].to(torch.uint8)
        
        return obs_image, self.get_language_instructions(), info

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
        """Sets all environments to forward mode."""
        self.reward_shaper.set_backward_mask(torch.zeros(self.num_envs, dtype=torch.bool, device=self.device))
        self.reward_shaper.reset_reward_stats()

    def set_backward(self):
        """Sets all environments to backward mode."""
        self.reward_shaper.set_backward_mask(torch.ones(self.num_envs, dtype=torch.bool, device=self.device))
        self.reward_shaper.reset_reward_stats()

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
        """Manually sets the current task for all environments."""
        self.env.unwrapped.set_current_task(objects, receptacles)
        self.reward_shaper.reset_reward_stats()

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
        info = self.reward_shaper.compute_info(self.env)
        return obs, instruct, info
