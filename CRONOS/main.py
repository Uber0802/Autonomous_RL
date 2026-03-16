import os
os.environ["TOKENIZERS_PARALLELISM"] = "false" # Silence fork warnings
import logging

# Suppress ManiSkill agent registration warnings
class ManiSkillFilter(logging.Filter):
    def filter(self, record):
        return "is already registered" not in record.getMessage() and "Overriding registered agent" not in record.getMessage()

logging.getLogger("mani_skill").addFilter(ManiSkillFilter())

import atexit
import shutil
import itertools
import torch
import tyro
import wandb
from tqdm import tqdm
from pathlib import Path

from envs.wrapper import CronosWrapper
from envs.suite import TaskSuite
from envs.scheduler import TaskScheduler
import envs.bridge_multi  # Trigger environment registration
from training.ppo import CronosPPO
from training.buffer import CronosReplayBuffer
from mani_skill.utils.visualization.misc import images_to_video, put_info_on_image

# Standard Args from train_ms3_ppo.py
from dataclasses import dataclass
from typing import Annotated

@dataclass
class Args:
    env_id: str = "TwoObjectOneReceptacle-v1"
    seed: int = 0
    name: str = "CRONOS-PPO"
    obj_set: str = "rand"
    obj1_index: int = 7
    obj2_index: int = 2
    obj3_index: int = 3
    plate1_index: int = 1
    plate2_index: int = 2
    plate3_index: int = 3
    num_envs: int = 64
    episode_len: int = 80
    training_len: int = 320
    instruction_switch_interval: int = 80
    training_interval: int = 160
    enable_backward: bool = False
    backward_interval: int = 1
    reset_unsuitable: bool = False
    reset_robot: bool = True
    eval_interval: int = 4
    buffer_gamma: float = 0.99
    buffer_lambda: float = 0.95
    buffer_minibatch: int = 8
    buffer_inferbatch: int = 32
    alg_ppo_epoch: int = 1
    alg_gradient_accum: int = 1
    vla_grad_norm: float = 10.0
    alg_entropy_coef: float = 0.0
    vla_path: str = "openvla/openvla-7b"
    vla_unnorm_key: str = "bridge_orig"
    vla_load_path: str = ""
    vla_lora_rank: int = 32
    vla_lr: float = 1e-4
    vla_vhlr: float = 3e-3
    vla_optim_beta1: float = 0.9
    vla_optim_beta2: float = 0.999
    vla_temperature: float = 1.0
    vla_temperature_eval: float = 0.6
    wandb: bool = True
    wandb_dir: str = ""
    record_video: bool = False
    video_dir: str = "videos"
    num_groups: int = 0  # 0 means dynamically scale with available tasks
    max_episodes: int = 32
    vla_checkpoint_interval: int = 8
    vla_save_path: str = "checkpoints"
    resume_iteration: int = 0

class CronosRunner:
    """Coordinates the CRONOS training workflow."""
    
    def __init__(self, args):
        self.args = args
        self.iteration = args.resume_iteration
        
        # Initialize WandB
        run = wandb.init(
            project="CRONOS",
            name=args.name,
            config=args.__dict__,
            mode="online" if args.wandb else "offline"
        )
        self.glob_dir = Path(run.dir).parent / "glob"
        self.glob_dir.mkdir(parents=True, exist_ok=True)
        
        # Device Management (AutoRL style)
        device_id = 0
        device_id_other = 1 if torch.cuda.device_count() > 1 else 0
        self.device = torch.device(f"cuda:{device_id}")
        
        # Initialize Modular Components
        self.suite = TaskSuite()
        self.env = CronosWrapper(args, None, self.suite, device=self.device)
        
        # Initialize Policy
        from simpler_env.policies.openvla.openvla_train import OpenVLAPolicy
        self.policy = OpenVLAPolicy(args, device_id=device_id_other)
        self.env.unnorm_state = self.policy.vla.get_action_stats(args.vla_unnorm_key)
        
        # Get task pool from env after a synchronized reset and initialize scheduler
        self.env.reset() 
        task_pool = self.env.get_task_pool()
        self.scheduler = TaskScheduler(
            task_pool=task_pool,
            mode="sequential",
            num_envs=args.num_envs
        )
        self.env.set_scheduler(self.scheduler)
        
        self.ppo = CronosPPO(args, self.policy)
        self.buffer = CronosReplayBuffer(args)
        
        # Register cleanup for memmap files
        atexit.register(self.buffer.cleanup)
        
        if args.record_video:
            self.video_frames = [[] for _ in range(args.num_envs)]

    @torch.no_grad()
    def _get_action(self, obs, instruct, deterministic=False):
        """Processes observations in micro-batches to save activation VRAM."""
        total_batch = obs.shape[0]
        values, actions, logprobs = [], [], []

        for i in range(0, total_batch, self.args.buffer_inferbatch):
            obs_batch = obs[i:i + self.args.buffer_inferbatch]
            instruct_batch = instruct[i:i + self.args.buffer_inferbatch]
            
            val, act, logp = self.policy.get_action(
                {"image": obs_batch, "task_description": instruct_batch}, 
                deterministic=deterministic
            )
            values.append(val)
            actions.append(act)
            logprobs.append(logp)

        return torch.cat(values, dim=0), torch.cat(actions, dim=0), torch.cat(logprobs, dim=0)

    def eval(self, iteration, task_idx, obj_set, envs_object, envs_recep, prefix="ood"):
        """Runs a sequential robust evaluation sequence akin to eval.py out of distribution"""
        self.policy.prep_rollout()
        
        # Reset specific targets across all envs for evaluation segment
        self.env.set_task(envs_object, envs_recep)
        obs, instruct, info = self.env.reset(obj_set_override=obj_set)
        print(f"Evaluating: {instruct[0]}")
        
        episode_success = torch.zeros(self.args.num_envs, dtype=torch.bool, device=self.device)
        
        if self.args.record_video:
            video_frames = [[] for _ in range(self.args.num_envs)]
            for i in range(self.args.num_envs):
                video_frames[i].append(obs[i].cpu().numpy().copy())
        
        for _ in tqdm(range(self.args.episode_len), desc="eval", leave=False):
            with torch.no_grad():
                val, action, logp = self._get_action(obs, instruct, deterministic=True)
                
            obs, reward, truncated, env_info = self.env.step(action)
            episode_success |= env_info["success"]
            
            if self.args.record_video:
                for i in range(self.args.num_envs):
                    video_frames[i].append(obs[i].cpu().numpy().copy())
                    
        if self.args.record_video:
            render_dir = self.glob_dir / f"eval_{prefix}_ep{iteration}_task{task_idx}"
            render_dir.mkdir(parents=True, exist_ok=True)
            obj = envs_object[0]
            recep = envs_recep[0]
            for i in range(self.args.num_envs):
                success_val = int(episode_success[i].item())
                video_name = f"video_{i}-{obj}_{recep}-s_{success_val}"
                images_to_video(video_frames[i], str(render_dir), video_name, fps=10, verbose=False)
                
        return episode_success.float().mean().item()

    def run_rollout(self):
        """Executes a non-episodic rollout with instruction switching and partial resets."""
        self.policy.prep_rollout()
        obs, instruct, _ = self.env.reset()
        self.buffer.warmup(obs, instruct)
        
        self.video_frames = [[] for _ in range(self.args.num_envs)]
        
        # Determine number of active environmental groups dynamically
        self.num_groups = self.args.num_groups if self.args.num_groups > 0 else len(self.scheduler.task_pool)
        
        forward_count = 1
        segment_id = 0
        
        # Track current tasks for logging/naming
        active_groups = self.args.num_groups if self.args.num_groups > 0 else None
        objs, receps = self.scheduler.get_next_tasks(active_groups)
        self.last_info = {}
        
        for step_idx in tqdm(range(self.args.training_len), desc="Rollout", leave=False):
            # 1. Action Prediction (Vectorized/Micro-batched)
            with torch.no_grad():
                value, action, logprob = self._get_action(obs, instruct, deterministic=False)

            # 2. Environment Step
            next_obs, reward, truncated, info = self.env.step(action)
            # 3. Buffer Storage
            self.buffer.insert(next_obs, action, logprob, value, reward, 1.0 - truncated.float())
            
            # 4. Optional Video Recording
            if self.args.record_video:
                for i in range(self.args.num_envs):
                    self.video_frames[i].append(obs[i].cpu().numpy().copy())

            # 5. Segment & Task Switching Logic
            if (step_idx + 1) % self.args.instruction_switch_interval == 0:
                # Save video segment before switch
                if self.args.record_video:
                    self.save_video_segment(iteration=self.iteration, segment_id=segment_id)
                    segment_id += 1

                # Task Switching (AutoRL Continual Learning Logic)
                if self.args.enable_backward and forward_count >= self.args.backward_interval:
                    self.env.set_backward()
                    forward_count = 0
                else:
                    self.scheduler.update_index()
                    self.env.set_forward()
                    active_groups = self.args.num_groups if self.args.num_groups > 0 else None
                    objs, receps = self.scheduler.get_next_tasks(active_groups)
                    self.env.set_task(objs, receps)
                    forward_count += 1
                
                # Resets for non-episodic continuity
                if self.args.reset_unsuitable:
                    next_obs = self.env.reset_unsuitable_envs()
                elif self.args.reset_robot:
                    next_obs = self.env.reset_robot()

                # Prepare for NEW segment instructions
                instruct = self.env.get_language_instructions()
                with torch.no_grad():
                    next_value, _, _ = self._get_action(next_obs, instruct, deterministic=False)
                self.buffer.end_segment(next_value)
                
                # GPU Memory Management
                import gc
                gc.collect()
                torch.cuda.empty_cache()
                
                if step_idx + 1 < self.args.training_len:
                    self.buffer.warmup(next_obs, instruct)
            
            obs = next_obs

    def train(self):
        """Main training loop coordinating rollouts and PPO updates."""
        for iteration in range(self.args.resume_iteration, self.args.max_episodes):
            self.iteration = iteration
            
            # 1. Collect Experience
            self.run_rollout()
            
            # 2. Optimization Step
            self.buffer.compute_gae()
            self.policy.prep_training()
            
            train_results = []
            for _ in range(self.args.alg_ppo_epoch):
                train_results.extend(self.ppo.train_epoch(self.buffer))
            
            # 3. Logging & Checkpointing
            if train_results:
                wandb.log(train_results[-1], step=iteration)
            
            # 4. In-Training Evaluation
            if (iteration + 1) % self.args.eval_interval == 0 or (iteration + 1) == self.args.max_episodes:
                print(f"Running periodic evaluation at iteration {iteration}...")
                task_pool = self.env.get_task_pool()
                for task_idx, task in enumerate(task_pool):
                    obj, recep = TaskScheduler._extract_obj_recep(task)
                    
                    # 1. In-Domain Evaluation ("rand")
                    success_rate_id = self.eval(iteration, task_idx, "rand", [obj]*self.args.num_envs, [recep]*self.args.num_envs, prefix="in_domain")
                    wandb.log({f"eval_in_domain/{task}_success": success_rate_id}, step=iteration)
                    
                    # 2. Out-Of-Domain Evaluation ("rand_ood")
                    success_rate_ood = self.eval(iteration, task_idx, "rand_ood", [obj]*self.args.num_envs, [recep]*self.args.num_envs, prefix="ood")
                    wandb.log({f"eval_ood/{task}_success": success_rate_ood}, step=iteration)
            
            if (iteration + 1) % self.args.vla_checkpoint_interval == 0:
                ckpt_path = self.glob_dir / f"steps_{iteration + 1:04d}"
                self.policy.save(ckpt_path)
                print(f"Checkpoint saved: {ckpt_path}")
                
            self.buffer.reset()

    def save_video_segment(self, iteration, segment_id):
        """Saves current video buffer and clears it."""
        render_dir = self.glob_dir / f"rollout_ep{iteration}_seg{segment_id}"
        render_dir.mkdir(parents=True, exist_ok=True)
        
        for i in range(self.args.num_envs):
            frames = self.video_frames[i]
            if len(frames) > 0:
                images_to_video(frames, str(render_dir), f"env{i}", fps=10, verbose=False)
                self.video_frames[i] = []

def main():
    args = tyro.cli(Args)
    runner = CronosRunner(args)
    
    try:
        runner.train()
    finally:
        # Robust cleanup on exit or interrupt
        if hasattr(runner, "buffer"):
            runner.buffer.cleanup()

if __name__ == "__main__":
    main()
