import os
os.environ["TOKENIZERS_PARALLELISM"] = "false" # Silence fork warnings
import torch
import tyro
import wandb
import numpy as np
import itertools
import random
from tqdm import tqdm
from pathlib import Path
from dataclasses import dataclass

from envs.wrapper import CronosWrapper
from envs.suite import TaskSuite
from envs.scheduler import TaskScheduler
import envs.bridge_multi  # Trigger environment registration
from training.ppo import CronosPPO
from training.buffer import CronosReplayBuffer
from mani_skill.utils.visualization.misc import images_to_video

@dataclass
class EvalArgs:
    env_id: str = "TwoObjectOneReceptacle-v1"
    seed: int = 0
    name: str = "CRONOS-Eval"
    obj_set: str = "rand"
    vla_path: str = "openvla/openvla-7b"
    vla_load_path: str = "" # Full path to the checkpoint directory (e.g., checkpoints/checkpoint-8)
    vla_unnorm_key: str = "bridge_orig"
    obj1_index: int = 7
    obj2_index: int = 2
    obj3_index: int = 3
    plate1_index: int = 1
    plate2_index: int = 2
    plate3_index: int = 3
    num_envs: int = 8
    episode_len: int = 80
    eval_sequences: int = 5 # Number of sequences to evaluate (1 training + 4 random)
    wandb: bool = True
    record_video: bool = True
    video_dir: str = "eval_videos"
    vla_temperature_eval: float = 0.0
    vla_temperature: float = 1.0
    buffer_inferbatch: int = 32
    vla_lora_rank: int = 32
    vla_lr: float = 1e-4
    vla_vhlr: float = 3e-3
    vla_optim_beta1: float = 0.9
    vla_optim_beta2: float = 0.999
    alg_entropy_coef: float = 0.0

class CronosEvaluator:
    """Handles evaluation of saved CRONOS checkpoints matching AutoRL logic."""
    
    def __init__(self, args):
        self.args = args
        
        # Device Management (AutoRL style)
        device_id = 0
        device_id_other = 1 if torch.cuda.device_count() > 1 else 0
        self.device = torch.device(f"cuda:{device_id}")
        
        # Initialize WandB
        run = wandb.init(
            project="CRONOS-Eval",
            name=args.name,
            config=args.__dict__,
            mode="online" if args.wandb else "offline"
        )
        self.glob_dir = Path(run.dir).parent / "glob"
        self.glob_dir.mkdir(parents=True, exist_ok=True)
            
        # Initialize Modular Components
        self.suite = TaskSuite()
        self.env = CronosWrapper(args, None, self.suite, device=self.device)
            
        # Initialize Policy
        from simpler_env.policies.openvla.openvla_train import OpenVLAPolicy
        self.policy = OpenVLAPolicy(args, device_id=device_id_other)
        self.policy.prep_rollout()
        self.env.unnorm_state = self.policy.vla.get_action_stats(args.vla_unnorm_key)
        
    @torch.no_grad()
    def _get_action(self, obs, instruct, deterministic=True):
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
        
    def evaluate(self):
        """Evaluates 5 task sequences (training + 4 random permutations)."""
        # Ensure env is initialized and task pool is populated
        self.env.reset()
        task_pool = self.env.get_task_pool() # List of task strings
        print(f"Task Pool: {task_pool}")
        
        # 1. Generate sequences
        perms = list(itertools.permutations(task_pool))
        training_seq = perms.pop(0) # Assumes first perm is training order
        
        random.seed(self.args.seed)
        selected_perms = random.sample(perms, self.args.eval_sequences - 1)
        all_sequences = [training_seq] + selected_perms
        
        print(f"Starting sequential evaluation across {len(all_sequences)} sequences...")
        
        for seq_idx, task_list in enumerate(all_sequences):
            print(f"Sequence {seq_idx}: {task_list}")
            
            for task_idx, task_str in enumerate(task_list):
                obj, recep = TaskScheduler._extract_obj_recep(task_str)
                
                # Only reset on the first task of the sequence
                reset = (task_idx == 0)
                
                success_rate = self.run_task_episode(seq_idx, task_idx, obj, recep, reset)
                wandb.log({
                    f"eval/seq{seq_idx}_task{task_idx}_success": success_rate,
                    "eval/task_name": task_str
                })

    def run_task_episode(self, seq_idx, task_idx, obj, recep, reset):
        """Runs a single task episode, potentially without robot reset."""
        if reset:
            obs, instruct, _ = self.env.reset()
        else:
            self.env.set_task([obj] * self.args.num_envs, [recep] * self.args.num_envs)
            obs = self.env.get_obs_image()
            instruct = self.env.get_language_instructions()
            
        print(f"Rendering Seq {seq_idx} Task {task_idx}: {instruct[0]}")
        
        video_frames = [[] for _ in range(self.args.num_envs)]
        episode_success = torch.zeros(self.args.num_envs, dtype=torch.bool, device=self.device)
        
        for step_idx in tqdm(range(self.args.episode_len), desc=f"Seq{seq_idx}-Task{task_idx}", leave=False):
            # 1. Action Prediction
            with torch.no_grad():
                _, action, _ = self._get_action(obs, instruct, deterministic=True)
            
            # 2. Environment Step
            obs, reward, truncated, info = self.env.step(action)
            episode_success |= info["success"]
            
            # 3. Optional Video Recording
            if self.args.record_video:
                for i in range(self.args.num_envs):
                    video_frames[i].append(obs[i].cpu().numpy().copy())
            
        # 4. Save Final Evaluation Videos
        if self.args.record_video:
            render_dir = self.glob_dir / f"seq{seq_idx}_task{task_idx}"
            render_dir.mkdir(parents=True, exist_ok=True)
            for i in range(self.args.num_envs):
                success_val = int(episode_success[i].item())
                video_name = f"video_{i}-{obj}_{recep}-s_{success_val}"
                images_to_video(video_frames[i], str(render_dir), video_name, fps=10, verbose=False)
                
        return episode_success.float().mean().item()

def main():
    args = tyro.cli(EvalArgs)
    evaluator = CronosEvaluator(args)
    evaluator.evaluate()

if __name__ == "__main__":
    main()
