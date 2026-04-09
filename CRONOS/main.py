import logging
logging.getLogger("mani_skill").setLevel(logging.ERROR)

import gc
import sys
import atexit
import random
import itertools
import numpy as np
import torch
import tyro
import wandb
from tqdm import tqdm
from pathlib import Path
from dataclasses import dataclass
from collections import defaultdict
import pprint

from envs.wrapper import CronosWrapper
from envs.suite import TaskSuite
from envs.scheduler import TaskScheduler
import envs.bridge_multi  # Trigger environment registration
from training.ppo import CronosPPO
from training.buffer import CronosReplayBuffer
from mani_skill.utils.visualization.misc import images_to_video

@dataclass
class Args:
    # --- Run ---
    name: str = "CRONOS-PPO"
    seed: int = 0
    resume_episode: int = 0

    # --- Environment ---
    env_id: str = "TwoObjectOneReceptacle-v1"
    num_envs: int = 64
    obj_set: str = "rand"
    obj1_index: int = 7
    obj2_index: int = 2
    obj3_index: int = 3
    plate1_index: int = 1
    plate2_index: int = 2
    plate3_index: int = 3

    # --- Training lengths ---
    segment_len: int = 80           # ManiSkill horizon — one robot execution segment
    episode_len: int = 320          # full rollout between hard resets (non-episodic)
    task_len: int = 80              # steps between task switches (usually = segment_len)
    ppo_update_len: int = 160       # steps accumulated before one PPO update

    # --- Stopping conditions ---
    max_episodes: int = 32
    max_steps: int = 0              # 0 = derive from max_episodes
    max_reset: int = 8192

    # --- Non-episodic reset ---
    reset_robot: bool = True
    reset_unsuitable: bool = False
    enable_backward: bool = False
    backward_interval: int = 1
    num_groups: int = 0             # 0 = dynamically scale with available tasks

    # --- PPO / buffer ---
    alg_ppo_epoch: int = 1
    alg_gradient_accum: int = 20
    alg_entropy_coef: float = 0.0
    buffer_gamma: float = 0.99
    buffer_lambda: float = 0.95
    buffer_minibatch: int = 8
    buffer_inferbatch: int = 32

    # --- VLA model ---
    vla_path: str = "openvla/openvla-7b"
    vla_load_path: str = ""
    vla_unnorm_key: str = "bridge_orig"
    vla_lora_rank: int = 32
    vla_lr: float = 1e-4
    vla_vhlr: float = 3e-3
    vla_grad_norm: float = 10.0
    vla_optim_beta1: float = 0.9
    vla_optim_beta2: float = 0.999
    vla_temperature: float = 1.0
    vla_temperature_eval: float = 0.6

    # --- Evaluation ---
    eval_interval: int = 4
    eval_single: bool = False
    eval_sequential: bool = False
    eval_sequences: int = 5         # permutation sequences (1 training + N-1 random)
    vla_checkpoint_interval: int = 8

    # --- Logging ---
    wandb: bool = True
    wandb_dir: str = ""
    record_video: bool = True
    ppo_log: str = "ppo_log.txt"            # per-minibatch PPO progress (in glob_dir)
    eval_report: str = "eval_report.txt"    # per-eval success rate summary (in glob_dir)
    log_file: str = "run.log"               # tee of stdout for the whole run (in glob_dir); empty string disables
    debug_rollout: bool = False             # per-step [ROLLOUT] dump during iteration 0 (verbose)

class CronosRunner:
    """Coordinates the CRONOS training workflow."""
    
    def __init__(self, args):
        self.args = args
        self.iteration = args.resume_episode
        self._validate_args()

        # Set seed for deterministic alignment (matching AutoRL exactly)
        np.random.seed(args.seed)
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        
        # Initialize WandB
        wandb_kwargs = dict(
            project="CRONOS",
            name=args.name,
            config=args.__dict__,
            mode="online" if args.wandb else "offline",
        )
        if args.wandb_dir:
            wandb_kwargs["dir"] = args.wandb_dir
        run = wandb.init(**wandb_kwargs)
        self.glob_dir = Path(run.dir).parent / "glob"
        self.glob_dir.mkdir(parents=True, exist_ok=True)

        # Tee stdout to a log file so the terminal stream is minimal and the full
        # run record lives on disk. tqdm/wandb still write to stderr untouched.
        if args.log_file:
            log_path = self.glob_dir / args.log_file
            log_fp = open(log_path, "a", buffering=1)
            class _Tee:
                def __init__(self, *streams): self.streams = streams
                def write(self, s):
                    for st in self.streams: st.write(s)
                def flush(self):
                    for st in self.streams: st.flush()
            sys.stdout = _Tee(sys.__stdout__, log_fp)


        # Device Management (AutoRL style)
        device_id = 0
        device_id_other = 1 if torch.cuda.device_count() > 1 else 0
        self.device = torch.device(f"cuda:{device_id}")
        
        # Initialize Policy FIRST (matching AutoRL order: policy before env)
        from simpler_env.policies.openvla.openvla_train import OpenVLAPolicy
        self.policy = OpenVLAPolicy(args, device_id=device_id_other)
        self.ppo = CronosPPO(args, self.policy)

        # Initialize Env AFTER policy (matching AutoRL order)
        unnorm_state = self.policy.vla.get_action_stats(args.vla_unnorm_key)
        self.suite = TaskSuite()
        self.env = CronosWrapper(args, unnorm_state, self.suite, device=self.device)

        # Get task pool from env (CronosWrapper.__init__ already performs a seeded reset)
        task_pool = self.env.get_task_pool()
        self.scheduler = TaskScheduler(
            task_pool=task_pool,
            mode="sequential",
            num_envs=args.num_envs
        )
        self.env.set_scheduler(self.scheduler)

        self.buffer = CronosReplayBuffer(args)

        # Register cleanup for memmap files
        atexit.register(self.buffer.cleanup)

        # Reset tracking — restore from checkpoint if available
        self.hard_reset_count = 0
        self.soft_reset_count = 0
        if args.vla_load_path:
            self._restore_training_state(args.vla_load_path)
            # Sync strategy counter so it continues from checkpoint value
            self.env.reset_strategy.reset_unsuitable_count = self.soft_reset_count

        if args.record_video:
            self.video_frames = [[] for _ in range(args.num_envs)]

    def _restore_training_state(self, load_path):
        """Restores training progress from checkpoint (backward-compatible with old checkpoints)."""
        state_path = Path(load_path) / "training_state.pt"
        if not state_path.exists():
            return
        state = torch.load(state_path, map_location="cpu", weights_only=False)
        if "resume_episode" in state:
            self.args.resume_episode = state["resume_episode"]
            self.iteration = self.args.resume_episode
            self.hard_reset_count = state.get("hard_reset_count", 0)
            self.soft_reset_count = state.get("soft_reset_count", 0)
            prior_steps = self.args.resume_episode * self.args.episode_len * self.args.num_envs
            prior_resets = self.hard_reset_count + self.soft_reset_count
            print(f"Checkpoint loaded: episode={self.args.resume_episode}, "
                  f"total_steps={prior_steps}, total_resets={prior_resets}")

    def _validate_args(self):
        args = self.args
        assert args.segment_len > 0, "segment_len must be positive"
        assert args.num_envs > 0, "num_envs must be positive"
        assert args.max_episodes > 0, "max_episodes must be positive"
        assert args.episode_len % args.segment_len == 0, \
            f"episode_len ({args.episode_len}) must be divisible by segment_len ({args.segment_len})"
        assert args.task_len % args.segment_len == 0, \
            f"task_len ({args.task_len}) must be divisible by segment_len ({args.segment_len})"
        assert args.episode_len % args.task_len == 0, \
            f"episode_len ({args.episode_len}) must be divisible by task_len ({args.task_len})"
        assert args.ppo_update_len % args.task_len == 0, \
            f"ppo_update_len ({args.ppo_update_len}) must be divisible by task_len ({args.task_len})"

    def _write_eval_report(self, header, results):
        """Appends eval results to the eval report file."""
        report_path = str(self.glob_dir / self.args.eval_report)
        with open(report_path, "a") as f:
            f.write(header)
            for task_name, stats in results:
                success = stats.get("success", 0.0)
                grasp = stats.get("consecutive_grasp", 0.0)
                obj_grasped = stats.get("is_src_obj_grasped", 0.0)
                f.write(f"  {task_name:<45s} success: {success:.4f}  grasp: {grasp:.4f}  obj_grasped: {obj_grasped:.4f}\n")
            f.write("\n")

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

    def eval(self, iteration, task_idx, obj_set, envs_object, envs_recep, prefix="eval", reset=True):
        """Evaluates one task episode, matching AutoRL's eval() behavior."""
        self.policy.prep_rollout()

        if reset:
            obs, _, info = self.env.reset(obj_set_override=obj_set)
            # Re-apply eval task: wrapper.reset() may override via scheduler
            self.env.set_task(envs_object, envs_recep)
            instruct = self.env.get_language_instructions()
        else:
            self.env.set_task(envs_object, envs_recep)
            obs, instruct, info = self.env.get_obs_instruct_info()

        print("Evaluating:", instruct[0])

        env_infos = defaultdict(list)

        if self.args.record_video:
            video_frames = [[] for _ in range(self.args.num_envs)]
            for i in range(self.args.num_envs):
                video_frames[i].append(obs[i].cpu().numpy().copy())

        for _ in tqdm(range(self.args.segment_len), desc="eval", leave=False):
            with torch.no_grad():
                val, action, logp = self._get_action(obs, instruct, deterministic=True)

            obs, reward, truncated, env_info = self.env.step(action)

            # Collect terminal episode stats (fires when any env truncates)
            if "episode" in env_info:
                for k, v in env_info["episode"].items():
                    env_infos[k] += v  # v is a list of per-env values

            if self.args.record_video:
                for i in range(self.args.num_envs):
                    video_frames[i].append(obs[i].cpu().numpy().copy())

        env_stats = {k: float(np.mean(v)) for k, v in env_infos.items()}
        print(pprint.pformat({k: round(v, 4) for k, v in env_stats.items()}))
        print()

        if self.args.record_video:
            render_dir = self.glob_dir / f"eval_{prefix}_ep{iteration + 1}_task{task_idx + 1}"
            render_dir.mkdir(parents=True, exist_ok=True)
            obj = envs_object[0]
            recep = envs_recep[0]
            for i in range(self.args.num_envs):
                success_val = int(env_infos["success"][i]) if "success" in env_infos and i < len(env_infos["success"]) else 0
                video_name = f"video_{i}-{obj}_{recep}-s_{success_val}"
                images_to_video(video_frames[i], str(render_dir), video_name, fps=10, verbose=False)

        return env_stats

    def _run_ppo_update(self, ppo_log_path):
        """Runs one PPO update on the current buffer, then resets it."""
        self.buffer.compute_gae()
        self.policy.prep_training()
        train_results = []
        for _ in range(self.args.alg_ppo_epoch):
            train_results.extend(self.ppo.train_epoch(self.buffer, log_path=ppo_log_path))
        self.buffer.reset()
        self.policy.prep_rollout()
        return train_results

    def run_rollout(self, ppo_log_path=None):
        """Executes a non-episodic rollout with instruction switching, partial resets,
        and mid-rollout PPO training every ppo_update_len steps (matching AutoRL)."""
        self.policy.prep_rollout()
        obs, instruct, _ = self.env.reset()
        self.buffer.warmup(obs, instruct)
        print(f"[INIT] obs_mean: {obs.float().mean().item():.6f}, instruction: {instruct[0][:40]}...", flush=True)

        self.video_frames = [[] for _ in range(self.args.num_envs)]

        # Determine number of active environmental groups dynamically
        self.num_groups = self.args.num_groups if self.args.num_groups > 0 else len(self.scheduler.task_pool)

        forward_count = 1
        segment_id = 0
        steps_since_ppo = 0
        all_train_results = []

        # Track current tasks for logging/naming
        active_groups = self.args.num_groups if self.args.num_groups > 0 else None
        objs, receps = self.scheduler.get_next_tasks(active_groups)
        self.last_info = {}

        for step_idx in tqdm(range(self.args.episode_len), desc="Rollout", leave=False):
            # 1. Action Prediction (Vectorized/Micro-batched)
            with torch.no_grad():
                value, action, logprob = self._get_action(obs, instruct, deterministic=False)

            # 2. Environment Step
            next_obs, reward, truncated, info = self.env.step(action)

            if self.args.debug_rollout and self.iteration == 0:
                print(f"[ROLLOUT] step: {step_idx + 1}, reward: {reward.mean().item():.6f}, value: {value.mean().item():.6f}, logprob: {logprob.mean().item():.6f}, action_mean: {action.float().mean().item():.6f}, obs_mean: {obs.float().mean().item():.6f}", flush=True)

            # 3. Buffer Storage
            self.buffer.insert(next_obs, action, logprob, value, reward, 1.0 - truncated.float())

            # 4. Optional Video Recording
            if self.args.record_video:
                for i in range(self.args.num_envs):
                    self.video_frames[i].append(obs[i].cpu().numpy().copy())

            # 5. Segment & Task Switching Logic
            if (step_idx + 1) % self.args.task_len == 0:
                # Save video segment before switch
                if self.args.record_video:
                    self.save_video_segment(iteration=self.iteration, segment_id=segment_id)
                    segment_id += 1

                # End current buffer segment
                instruct_next = self.env.get_language_instructions()  # still old task
                with torch.no_grad():
                    next_value, _, _ = self._get_action(next_obs, instruct_next, deterministic=False)
                self.buffer.end_segment(next_value)
                steps_since_ppo += self.args.task_len

                # 6. Mid-rollout PPO update (matching AutoRL's training_interval)
                if steps_since_ppo >= self.args.ppo_update_len and step_idx > 0:
                    results = self._run_ppo_update(ppo_log_path)
                    all_train_results.extend(results)
                    steps_since_ppo = 0

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
                    self.soft_reset_count = self.env.reset_strategy.reset_unsuitable_count
                    print(f"Total unsuitable resets: {self.soft_reset_count}")
                elif self.args.reset_robot:
                    next_obs = self.env.reset_robot()

                # Prepare for NEW segment instructions
                instruct = self.env.get_language_instructions()
                group_size = self.args.num_envs // self.num_groups
                group_instrs = [instruct[g * group_size] for g in range(self.num_groups)]
                print(f"Step {step_idx + 1}: switch instruction to  {' '.join(group_instrs)}")

                # GPU Memory Management
                gc.collect()
                torch.cuda.empty_cache()

                if step_idx + 1 < self.args.episode_len:
                    self.buffer.warmup(next_obs, instruct)

            obs = next_obs

        # Final PPO update on remaining buffer (if any data left)
        if self.buffer.num_env > 0:
            results = self._run_ppo_update(ppo_log_path)
            all_train_results.extend(results)

        return all_train_results

    def train(self):
        """Main training loop coordinating rollouts and PPO updates."""
        start_episode = self.args.resume_episode
        end_episode = start_episode + self.args.max_episodes

        # Derive max_steps from absolute endpoint if not explicitly set
        if self.args.max_steps <= 0:
            self.args.max_steps = end_episode * self.args.episode_len * self.args.num_envs

        print(f"Training {self.args.max_episodes} episodes ({start_episode + 1} → {end_episode}), "
              f"max_steps={self.args.max_steps}, max_reset={self.args.max_reset}")

        for iteration in range(start_episode, end_episode):
            self.iteration = iteration
            episode = iteration + 1  # 1-based absolute

            # 1. Collect Experience + Mid-rollout PPO
            ppo_log_path = str(self.glob_dir / self.args.ppo_log)
            total_steps = episode * self.args.episode_len * self.args.num_envs
            with open(ppo_log_path, "a") as f:
                f.write(f"step : {total_steps}\n")

            train_results = self.run_rollout(ppo_log_path=ppo_log_path)
            self.hard_reset_count += self.args.num_envs
            total_resets = self.hard_reset_count + self.soft_reset_count
            print(f"Episode {episode}: total_steps={total_steps}, total_resets={total_resets}")

            # 3. Logging (dual-axis: both episode and total_steps)
            if train_results:
                wandb.log({
                    **train_results[-1],
                    "episode": episode,
                    "total_steps": total_steps,
                    "total_resets": total_resets,
                })

            # 4. Stopping conditions
            exceed_step_limit = total_steps >= self.args.max_steps
            exceed_reset_limit = total_resets > self.args.max_reset
            should_stop = exceed_step_limit or exceed_reset_limit

            # 5. In-Training Evaluation
            should_eval = (iteration % self.args.eval_interval == self.args.eval_interval - 1
                           or iteration == end_episode - 1
                           or should_stop)

            if should_eval:
                task_pool = self.env.get_task_pool()
                eval_log = {"episode": episode, "total_steps": total_steps, "total_resets": total_resets}
                report_header = (
                    f"{'=' * 50}\n"
                    f"Episode {episode} | Steps {total_steps} | Resets {total_resets}\n"
                    f"{'=' * 50}\n"
                )

                # In-domain eval: same object set as training, randomized scene layout
                print(f"Evaluating In-Domain at {total_steps}")
                in_domain_results = []
                for task_idx, task in enumerate(task_pool):
                    obj, recep = TaskScheduler._extract_obj_recep(task)
                    sval_stats = self.eval(iteration, task_idx, self.args.obj_set, [obj]*self.args.num_envs, [recep]*self.args.num_envs, prefix=f"in_domain_{obj}_{recep}")
                    eval_log.update({f"eval_in_domain_put_{obj}_in_{recep}/{k}": v for k, v in sval_stats.items()})
                    in_domain_results.append((task, sval_stats))

                # Out-of-domain eval: different object set, novel scene layout
                print(f"Evaluating Out-of-Domain at {total_steps}")
                ood_results = []
                for task_idx, task in enumerate(task_pool):
                    obj, recep = TaskScheduler._extract_obj_recep(task)
                    sval_stats = self.eval(iteration, task_idx, "rand_ood", [obj]*self.args.num_envs, [recep]*self.args.num_envs, prefix=f"out_of_domain_{obj}_{recep}")
                    eval_log.update({f"eval_out_of_domain_put_{obj}_in_{recep}/{k}": v for k, v in sval_stats.items()})
                    ood_results.append((task, sval_stats))

                wandb.log(eval_log)
                self._write_eval_report(report_header + "\nIn-Domain Evaluation:\n", in_domain_results)
                self._write_eval_report("Out-of-Domain Evaluation:\n", ood_results)

            # 6. Checkpoint (includes training progress for resume)
            if (iteration + 1) % self.args.vla_checkpoint_interval == 0 or should_stop:
                ckpt_path = self.glob_dir / f"episode_{episode:04d}"
                self.policy.save(ckpt_path, extra_state={
                    "resume_episode": episode,
                    "hard_reset_count": self.hard_reset_count,
                    "soft_reset_count": self.soft_reset_count,
                })
                print(f"Checkpoint saved at episode {episode} (steps={total_steps}): {ckpt_path}")

            self.buffer.reset()

            # 7. Break if limit hit (after completing this episode fully)
            if should_stop:
                reason = "step limit" if exceed_step_limit else "reset limit"
                print(f"Stopping ({reason}): steps={total_steps}/{self.args.max_steps}, resets={total_resets}/{self.args.max_reset}")
                break

        # Final summary
        final_episode = self.iteration + 1
        final_steps = final_episode * self.args.episode_len * self.args.num_envs
        final_resets = self.hard_reset_count + self.soft_reset_count
        print(f"Training complete: episode={final_episode}, total_steps={final_steps}, total_resets={final_resets}")

    def save_video_segment(self, iteration, segment_id):
        """Saves current video buffer and clears it."""
        render_dir = self.glob_dir / f"rollout_ep{iteration + 1}_seg{segment_id + 1}"
        render_dir.mkdir(parents=True, exist_ok=True)
        
        for i in range(self.args.num_envs):
            frames = self.video_frames[i]
            if len(frames) > 0:
                images_to_video(frames, str(render_dir), f"env{i}", fps=10, verbose=False)
                self.video_frames[i] = []

def main():
    args = tyro.cli(Args)
    runner = CronosRunner(args)
    
    if args.eval_single:
        # Standard Single/All Task Evaluation
        task_pool = runner.env.get_task_pool()
        results = []
        for task_idx, task in enumerate(task_pool):
            obj, recep = TaskScheduler._extract_obj_recep(task)
            sval_stats = runner.eval(0, task_idx, args.obj_set, [obj]*args.num_envs, [recep]*args.num_envs, prefix="eval")
            results.append((task, sval_stats))
        runner._write_eval_report("Single Task Evaluation:\n", results)

    elif args.eval_sequential:
        # Multi-sequence permutation eval
        import random as _random
        runner.env.reset()
        task_pool = runner.env.get_task_pool()
        print(f"Task Pool: {task_pool}")

        perms = list(itertools.permutations(task_pool))
        training_seq = perms.pop(0)  # first permutation = training order

        _random.seed(args.seed)
        selected_perms = _random.sample(perms, min(args.eval_sequences - 1, len(perms)))
        all_sequences = [training_seq] + selected_perms

        print(f"Running Sequential Evaluation across {len(all_sequences)} sequences...")
        for seq_idx, task_list in enumerate(all_sequences):
            print(f"Sequence {seq_idx + 1}: {task_list}")
            seq_results = []
            for task_idx, task_str in enumerate(task_list):
                obj, recep = TaskScheduler._extract_obj_recep(task_str)
                reset = (task_idx == 0)
                sval_stats = runner.eval(seq_idx, task_idx, args.obj_set, [obj]*args.num_envs, [recep]*args.num_envs, prefix=f"seq{seq_idx}_task{task_idx}", reset=reset)
                wandb.log({f"eval/seq{seq_idx}_task{task_idx}_{k}": v for k, v in sval_stats.items()})
                seq_results.append((task_str, sval_stats))
            runner._write_eval_report(f"Sequence {seq_idx + 1}:\n", seq_results)

    else:
        runner.train()

if __name__ == "__main__":
    main()
