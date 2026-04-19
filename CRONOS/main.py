import logging
logging.getLogger("mani_skill").setLevel(logging.ERROR)

import gc
import sys
import signal
import atexit
import json
import random
import itertools
import datetime
import subprocess
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
from training.metrics import SuccessRecorder
from mani_skill.utils.visualization.misc import images_to_video

@dataclass
class Args:
    # --- Run ---
    name: str = "CRONOS-PPO"
    seed: int = 0
    resume_episode: int = 0

    # --- Environment ---
    # V0.2 M2 Phase B: PickPlaceNxM-v1 replaces the 8 legacy registered IDs.
    # Pick the (N, M) shape via --env_n / --env_m (defaults to (2, 1), the
    # V0.1 baseline shape from train.sh).
    env_id: str = "PickPlaceNxM-v1"
    env_n: int = 2  # number of objects (carrots)
    env_m: int = 2  # number of receptacles (plates)
    num_envs: int = 64
    obj_set: str = "rand"
    obj1_index: int = 7             # 1-based object indices (passed to env via options)
    obj2_index: int = 2
    obj3_index: int = 10
    plate1_index: int = 1           # 1-based plate indices
    plate2_index: int = 2
    plate3_index: int = 3
    scene: str = ""                 # V0.2 M4: named scene (empty = default lighting/overlay)

    # --- Training lengths ---
    segment_len: int = 80           # ManiSkill horizon — one robot execution segment
    episode_len: int = 320          # full rollout between hard resets (non-episodic)
    task_len: int = 80              # steps between task switches (usually = segment_len)
    ppo_update_len: int = 160       # steps accumulated before one PPO update

    # --- Stopping conditions (all RELATIVE to this run, not cumulative) ---
    max_episodes: int = 32          # how many episodes to run in THIS invocation
    max_steps: int = 0              # env-step budget for THIS run; 0 = derive from max_episodes
    max_reset: int = 8192           # reset budget for THIS run; added to prior resets from checkpoint

    # --- Task control (V0.2 M3) ---
    config_path: str = ""           # YAML experiment config (overrides below when set)
    task_order: str = "sequential"  # sequential | pure_random | sequence_random
    task_filter: str = ""           # comma-separated indices or task strings; empty = all

    # --- Non-episodic reset ---
    reset_mode: str = "per_episode"  # per_episode | none (V0.2 M5)
    reset_robot: bool = True
    reset_unsuitable: bool = False
    unsuitable_detector: str = "low_z"
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
    resume_from: str = ""           # V0.2 M5: path to checkpoint dir (auto-loads config + weights)

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
        # V0.2 M5: --resume_from auto-sets vla_load_path from checkpoint dir
        if args.resume_from:
            ckpt = Path(args.resume_from)
            if not ckpt.exists():
                raise FileNotFoundError(f"--resume_from path does not exist: {ckpt}")
            args.vla_load_path = str(ckpt)
            # Load scheduler state if available
            sched_path = ckpt / "scheduler_state.json"
            if sched_path.exists():
                self._resume_scheduler_state = json.loads(sched_path.read_text())
            print(f"[RESUME] from {ckpt}, vla_load_path set")

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
        # X-axis: all metrics keyed on total_steps (environment steps)
        wandb.define_metric("total_steps")
        wandb.define_metric("eval_*", step_metric="total_steps")
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

        # V0.2 M0: SuccessRecorder — dual-axis CSVs + wandb 5-chart panel + counters.json
        self.recorder = SuccessRecorder(self.glob_dir)

        # V0.2 M1: pre-init config dump — written BEFORE any SAPIEN scene is
        # constructed, so a crash during env init still leaves an on-disk
        # snapshot of exactly what was attempted.
        self._dump_run_config()

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

        # V0.2 M3: load YAML config if provided, merge with CLI overrides
        yaml_config = None
        if args.config_path:
            from envs.config import load_cronos_config
            yaml_config = load_cronos_config(args.config_path)
            # YAML env params → Args (CLI overrides YAML)
            for field_name in ("env_n", "env_m", "num_envs",
                               "obj1_index", "obj2_index", "obj3_index",
                               "plate1_index", "plate2_index", "plate3_index",
                               "scene", "task_order"):
                yaml_val = getattr(yaml_config, field_name, None)
                if yaml_val is not None and not self._cli_provided(field_name):
                    setattr(args, field_name, yaml_val)

        # Parse task_filter from CLI string or YAML
        task_filter = None
        if args.task_filter:
            parts = [p.strip() for p in args.task_filter.split(",")]
            task_filter = []
            for p in parts:
                try:
                    task_filter.append(int(p))
                except ValueError:
                    task_filter.append(p)
        elif yaml_config and yaml_config.task_filter:
            task_filter = yaml_config.task_filter

        # Get task pool from env (CronosWrapper.__init__ already performs a seeded reset)
        task_pool = self.env.get_task_pool()

        # V0.2: resolve symbolic refs (obj1, recep2, ...) in YAML groups + task_filter
        from envs.config import resolve_symbolic_task, has_symbolic_refs
        from envs.scheduler import GroupState
        obj_names, recep_names = self._build_symbolic_map(task_pool)

        # Build per-group GroupState from YAML or auto-generate from flat task_pool
        group_states = []
        if yaml_config and yaml_config.groups:
            # New per-group format
            for g in yaml_config.groups:
                # Build per-group symbolic maps from this group's obj/recep lists
                g_obj_names = {f"obj{i+1}": obj_names.get(f"obj{i+1}", f"obj{i+1}")
                                for i in range(len(g.obj))}
                g_recep_names = {f"recep{i+1}": recep_names.get(f"recep{i+1}", f"recep{i+1}")
                                  for i in range(len(g.recep))}
                # Resolve symbolic refs in task_sequence and eval_tasks
                resolved_seq = [
                    resolve_symbolic_task(t, g_obj_names, g_recep_names) if has_symbolic_refs(t) else t
                    for t in g.task_sequence
                ]
                resolved_eval = [
                    resolve_symbolic_task(t, g_obj_names, g_recep_names) if has_symbolic_refs(t) else t
                    for t in g.eval_tasks
                ]
                group_states.append(GroupState.from_sequence(
                    name=g.name,
                    task_sequence=resolved_seq,
                    eval_tasks=resolved_eval,
                ))
            self.scheduler = TaskScheduler(
                group_states=group_states,
                mode=args.task_order,
                num_envs=args.num_envs,
            )
        else:
            # Legacy flat path: use env-provided task pool with optional filter
            if task_filter:
                task_filter = [
                    resolve_symbolic_task(t, obj_names, recep_names)
                    if isinstance(t, str) and has_symbolic_refs(t) else t
                    for t in task_filter
                ]
            self.scheduler = TaskScheduler.from_flat_pool(
                task_pool=task_pool,
                mode=args.task_order,
                num_envs=args.num_envs,
                task_filter=task_filter,
            )
        self.env.set_scheduler(self.scheduler)

        # V0.2 M5: restore scheduler cursor from checkpoint if available
        if hasattr(self, '_resume_scheduler_state'):
            self.scheduler.load_state(self._resume_scheduler_state)
            print(f"[SCHEDULER] restored cursor from checkpoint")
        print(f"[SCHEDULER] mode={args.task_order}, pool={self.scheduler.task_pool}")

        self.buffer = CronosReplayBuffer(args)

        # V0.2 M1: deterministic mmap cleanup on SIGINT. atexit alone races
        # against still-open mmap fds at interpreter shutdown, producing the
        # ENOTEMPTY silly-rename error seen in V0.1.
        atexit.register(self.buffer.cleanup)
        self._install_sigint_handler()

        # Reset tracking — restore from checkpoint if available
        self.hard_reset_count = 0
        self.soft_reset_count = 0
        if args.vla_load_path:
            self._restore_training_state(args.vla_load_path)
            # Sync strategy counter so it continues from checkpoint value
            self.env.reset_strategy.reset_unsuitable_count = self.soft_reset_count

        if args.record_video:
            self.video_frames = [[] for _ in range(args.num_envs)]

    def _dump_run_config(self):
        """V0.2 M1: write glob_dir/run_config.{json,yaml} before any SAPIEN
        init. Contents: resolved Args + seed + start_time + git_rev (best-
        effort) + cronos_version. The snapshot lands *before* the policy/env
        are constructed, so even a crash during env init leaves a diff-able
        record of exactly what the user attempted."""
        cfg = dict(self.args.__dict__)
        cfg["start_time"] = datetime.datetime.utcnow().isoformat() + "Z"
        cfg["cronos_version"] = "V0.2"
        try:
            rev = subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                stderr=subprocess.DEVNULL,
                cwd=str(Path(__file__).resolve().parent),
            ).decode().strip()
            cfg["git_rev"] = rev
        except Exception:
            cfg["git_rev"] = None

        # Coerce anything non-JSON-native (Path, etc.) via default=str.
        (self.glob_dir / "run_config.json").write_text(
            json.dumps(cfg, indent=2, default=str, sort_keys=False) + "\n"
        )
        try:
            import yaml  # type: ignore
            (self.glob_dir / "run_config.yaml").write_text(
                yaml.safe_dump(cfg, sort_keys=False)
            )
        except Exception:
            # yaml is a soft dep — the .json file is the authoritative snapshot.
            pass

    def _install_sigint_handler(self):
        """Ensure buffer.cleanup() runs synchronously on Ctrl-C before the
        interpreter tears down (atexit runs too late relative to the mmap
        fd release window on NFS)."""
        prev = signal.getsignal(signal.SIGINT)

        def handler(signum, frame):
            try:
                self.buffer.cleanup()
            except Exception as e:
                print(f"[SIGINT] buffer cleanup error (non-fatal): {e}")
            # Chain to the previous handler (tyro / default) so Ctrl-C still
            # terminates the process as expected.
            if callable(prev) and prev not in (signal.SIG_DFL, signal.SIG_IGN):
                try:
                    prev(signum, frame)
                    return
                except Exception:
                    pass
            raise KeyboardInterrupt

        try:
            signal.signal(signal.SIGINT, handler)
        except ValueError:
            # Not in main thread — skip silently.
            pass

    @staticmethod
    def _build_symbolic_map(task_pool):
        """Build obj1→name and recep1→name maps from the env's task pool.

        The task pool has strings like "put ketchup bottle on yellow_plate".
        We extract unique objects and receptacles in the order they first appear,
        and assign them symbolic names obj1, obj2, ..., recep1, recep2, ...
        """
        from envs.scheduler import TaskScheduler
        objs_seen, receps_seen = [], []
        for task in task_pool:
            obj, recep = TaskScheduler._extract_obj_recep(task)
            if obj and obj not in objs_seen:
                objs_seen.append(obj)
            if recep and recep not in receps_seen:
                receps_seen.append(recep)
        obj_map = {f"obj{i+1}": name for i, name in enumerate(objs_seen)}
        recep_map = {f"recep{i+1}": name for i, name in enumerate(receps_seen)}
        return obj_map, recep_map

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
            # total_steps from checkpoint (correct even if episode_len changes between runs).
            # Fallback: recompute from resume_episode × current episode_len (old checkpoints).
            self.prior_total_steps = state.get(
                "total_steps",
                self.args.resume_episode * self.args.episode_len * self.args.num_envs
            )
            prior_resets = self.hard_reset_count + self.soft_reset_count
            print(f"Checkpoint loaded: episode={self.args.resume_episode}, "
                  f"total_steps={self.prior_total_steps}, total_resets={prior_resets}")

    @staticmethod
    def _cli_provided(field_name):
        """Check if a CLI flag was explicitly passed (heuristic: present in sys.argv)."""
        import sys
        return any(f"--{field_name}" in arg for arg in sys.argv)

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
        assert args.reset_mode in ("per_episode", "none"), \
            f"reset_mode must be 'per_episode' or 'none', got '{args.reset_mode}'"
        if args.reset_mode == "none":
            assert args.reset_unsuitable, \
                "reset_mode=none requires reset_unsuitable=True (otherwise stuck envs never recover)"

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

    def run_rollout(self, ppo_log_path=None, episode=0, episode_base_steps=0, episode_base_resets=0):
        """Executes a non-episodic rollout with instruction switching, partial resets,
        and mid-rollout PPO training every ppo_update_len steps (matching AutoRL)."""
        self.policy.prep_rollout()

        # V0.2 M5: reset_mode=none skips env.reset() after the first episode.
        # The sim continues from the live state; only reset_unsuitable + task
        # switching drive state changes.
        if self.args.reset_mode == "none" and hasattr(self, '_no_reset_obs'):
            obs = self._no_reset_obs
            instruct = self.env.get_language_instructions()
        else:
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
        info = {}

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

        # V0.2 M5: stash live obs for reset_mode=none so next episode skips env.reset
        self._no_reset_obs = obs

        return all_train_results

    def train(self):
        """Main training loop coordinating rollouts and PPO updates."""
        start_episode = self.args.resume_episode
        end_episode = start_episode + self.args.max_episodes

        # All stopping conditions are RELATIVE (per-run budget).
        # Convert to absolute ceilings by adding the checkpoint's prior totals.
        # prior_total_steps is loaded from checkpoint (survives episode_len changes);
        # falls back to recomputing if no checkpoint was loaded.
        prior_steps = getattr(self, 'prior_total_steps',
                              start_episode * self.args.episode_len * self.args.num_envs)
        prior_resets = self.hard_reset_count + self.soft_reset_count

        # max_steps: 0 = derive from max_episodes (default)
        if self.args.max_steps <= 0:
            abs_max_steps = end_episode * self.args.episode_len * self.args.num_envs
        else:
            abs_max_steps = prior_steps + self.args.max_steps

        # max_reset: relative budget for THIS run
        abs_max_reset = prior_resets + self.args.max_reset

        print(f"Training {self.args.max_episodes} episodes ({start_episode + 1} → {end_episode}), "
              f"max_steps={abs_max_steps} (prior={prior_steps} + budget={self.args.max_steps}), "
              f"max_reset={abs_max_reset} (prior={prior_resets} + budget={self.args.max_reset})")

        for iteration in range(start_episode, end_episode):
            self.iteration = iteration
            episode = iteration + 1  # 1-based absolute

            # 1. Collect Experience + Mid-rollout PPO
            ppo_log_path = str(self.glob_dir / self.args.ppo_log)
            # total_steps = prior (from checkpoint) + steps generated in THIS run.
            # This is correct even when episode_len changes between runs.
            episodes_this_run = iteration - start_episode
            total_steps = prior_steps + (episodes_this_run + 1) * self.args.episode_len * self.args.num_envs
            episode_base_steps = prior_steps + episodes_this_run * self.args.episode_len * self.args.num_envs
            episode_base_resets = self.hard_reset_count + self.soft_reset_count
            with open(ppo_log_path, "a") as f:
                f.write(f"step : {total_steps}\n")

            train_results = self.run_rollout(
                ppo_log_path=ppo_log_path,
                episode=episode,
                episode_base_steps=episode_base_steps,
                episode_base_resets=episode_base_resets,
            )
            # Only count hard resets when env.reset() actually fires
            if self.args.reset_mode != "none" or iteration == start_episode:
                self.hard_reset_count += self.args.num_envs
            total_resets = self.hard_reset_count + self.soft_reset_count
            print(f"Episode {episode}: total_steps={total_steps}, total_resets={total_resets}")

            # V0.2 M0: durable counters sidecar for resume continuity
            self.recorder.write_counters(
                episode=episode,
                total_steps=total_steps,
                total_resets=total_resets,
            )

            # 3. Logging (dual-axis: both episode and total_steps)
            if train_results:
                wandb.log({
                    **train_results[-1],
                    "episode": episode,
                    "total_steps": total_steps,
                    "total_resets": total_resets,
                }, step=total_steps)

            # 4. Stopping conditions (checked against absolute ceilings)
            exceed_step_limit = total_steps >= abs_max_steps
            exceed_reset_limit = total_resets > abs_max_reset
            should_stop = exceed_step_limit or exceed_reset_limit

            # 5. In-Training Evaluation
            should_eval = (iteration % self.args.eval_interval == self.args.eval_interval - 1
                           or iteration == end_episode - 1
                           or should_stop)

            if should_eval:
                # V0.2: per-group eval tasks. Union all groups' eval_tasks (dedup
                # preserving order) — all envs share the same physical objects
                # so each unique task is evaluated once across all num_envs envs.
                eval_log = {"episode": episode, "total_steps": total_steps, "total_resets": total_resets}
                report_header = (
                    f"{'=' * 50}\n"
                    f"Episode {episode} | Steps {total_steps} | Resets {total_resets}\n"
                    f"{'=' * 50}\n"
                )

                # Collect per-(task, group) eval assignments, deduplicating task strings
                # but preserving group attribution for CSV logging.
                per_group_evals = self.scheduler.get_eval_tasks_per_group()
                # Build ordered list of (group_name, task) pairs, deduplicating repeats
                seen_tasks = set()
                eval_pairs = []  # [(group_name, task), ...]
                for g_name, g_eval_tasks in per_group_evals:
                    for t in g_eval_tasks:
                        if t not in seen_tasks:
                            seen_tasks.add(t)
                            eval_pairs.append((g_name, t))

                # In-domain eval
                print(f"Evaluating In-Domain at {total_steps}")
                in_domain_results = []
                for task_idx, (group_name, task) in enumerate(eval_pairs):
                    obj, recep = TaskScheduler._extract_obj_recep(task)
                    sval_stats = self.eval(iteration, task_idx, self.args.obj_set, [obj]*self.args.num_envs, [recep]*self.args.num_envs, prefix=f"in_domain_{obj}_{recep}")
                    scalars = self.recorder.log_eval(
                        episode=episode,
                        total_steps=total_steps,
                        total_resets=total_resets,
                        eval_kind="in_domain",
                        group=group_name,
                        task=task,
                        scene="default",
                        n_envs=self.args.num_envs,
                        success=sval_stats.get("success", 0.0),
                        grasp=sval_stats.get("consecutive_grasp", 0.0),
                        obj_grasped=sval_stats.get("is_src_obj_grasped", 0.0),
                    )
                    eval_log.update(scalars)
                    in_domain_results.append((task, sval_stats))
                eval_log.update(self.recorder.build_wandb_eval_panel("in_domain"))

                # Out-of-domain eval
                print(f"Evaluating Out-of-Domain at {total_steps}")
                ood_results = []
                for task_idx, (group_name, task) in enumerate(eval_pairs):
                    obj, recep = TaskScheduler._extract_obj_recep(task)
                    sval_stats = self.eval(iteration, task_idx, "rand_ood", [obj]*self.args.num_envs, [recep]*self.args.num_envs, prefix=f"out_of_domain_{obj}_{recep}")
                    scalars = self.recorder.log_eval(
                        episode=episode,
                        total_steps=total_steps,
                        total_resets=total_resets,
                        eval_kind="out_of_domain",
                        group=group_name,
                        task=task,
                        scene="default",
                        n_envs=self.args.num_envs,
                        success=sval_stats.get("success", 0.0),
                        grasp=sval_stats.get("consecutive_grasp", 0.0),
                        obj_grasped=sval_stats.get("is_src_obj_grasped", 0.0),
                    )
                    eval_log.update(scalars)
                    ood_results.append((task, sval_stats))
                eval_log.update(self.recorder.build_wandb_eval_panel("out_of_domain"))

                wandb.log(eval_log, step=total_steps)
                self._write_eval_report(report_header + "\nIn-Domain Evaluation:\n", in_domain_results)
                self._write_eval_report("Out-of-Domain Evaluation:\n", ood_results)

            # 6. Checkpoint (includes training progress for resume)
            if (iteration + 1) % self.args.vla_checkpoint_interval == 0 or should_stop:
                ckpt_path = self.glob_dir / f"episode_{episode:04d}"
                self.policy.save(ckpt_path, extra_state={
                    "resume_episode": episode,
                    "total_steps": total_steps,
                    "hard_reset_count": self.hard_reset_count,
                    "soft_reset_count": self.soft_reset_count,
                })
                # V0.2 M5: per-checkpoint config + scheduler state
                import shutil, json as _json
                src_cfg = self.glob_dir / "run_config.yaml"
                if src_cfg.exists():
                    shutil.copy2(src_cfg, ckpt_path / "run_config.yaml")
                sched_state = self.scheduler.get_state()
                sched_state["task_pool"] = self.scheduler.task_pool  # for human inspection
                (ckpt_path / "scheduler_state.json").write_text(
                    _json.dumps(sched_state, indent=2) + "\n"
                )
                print(f"Checkpoint saved at episode {episode} (steps={total_steps}): {ckpt_path}")

            self.buffer.reset()

            # 7. Break if limit hit (after completing this episode fully)
            if should_stop:
                reason = "step limit" if exceed_step_limit else "reset limit"
                print(f"Stopping ({reason}): steps={total_steps}/{abs_max_steps}, resets={total_resets}/{abs_max_reset}")
                break

        # Final summary
        final_episode = self.iteration + 1
        episodes_done = final_episode - start_episode
        final_steps = prior_steps + episodes_done * self.args.episode_len * self.args.num_envs
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
        task_pool = runner.scheduler.task_pool
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
        task_pool = runner.scheduler.task_pool
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
                seq_total_steps = (seq_idx + 1) * args.episode_len * args.num_envs
                scalars = runner.recorder.log_eval(
                    episode=seq_idx + 1,
                    total_steps=seq_total_steps,
                    total_resets=0,
                    eval_kind=f"sequential_seq{seq_idx}",
                    task=task_str,
                    scene="default",
                    n_envs=args.num_envs,
                    success=sval_stats.get("success", 0.0),
                    grasp=sval_stats.get("consecutive_grasp", 0.0),
                    obj_grasped=sval_stats.get("is_src_obj_grasped", 0.0),
                )
                panel = runner.recorder.build_wandb_eval_panel(f"sequential_seq{seq_idx}")
                wandb.log({**scalars, **panel}, step=seq_total_steps)
                seq_results.append((task_str, sval_stats))
            runner._write_eval_report(f"Sequence {seq_idx + 1}:\n", seq_results)

    else:
        runner.train()

if __name__ == "__main__":
    main()
