import logging
logging.getLogger("mani_skill").setLevel(logging.ERROR)

import gc
import os
import sys
import signal
import atexit
import json
import random
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

from run_paths import prepare_wandb_dir, verify_run_dir
from envs.wrapper import CronosWrapper
from envs.suite import TaskSuite
from envs.scheduler import TaskScheduler, build_eval_sequences
import envs.bridge_multi  # Trigger environment registration
from training.ppo import CronosPPO, aggregate_train_results
from training.grpo import CronosGRPO
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
    # PickPlaceNxM-v1 — pick the (N, M) shape via --env_n / --env_m.
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
    scene: str = ""                 # named scene (empty = default lighting/overlay)

    # --- Training lengths ---
    segment_len: int = 80           # ManiSkill horizon — one robot execution segment
    episode_len: int = 320          # full rollout between hard resets (non-episodic)
    task_len: int = 80              # steps between task switches (usually = segment_len)
    ppo_update_len: int = 160       # steps accumulated before one PPO update

    # --- Stopping conditions (all RELATIVE to this run, not cumulative) ---
    max_episodes: int = 32          # how many episodes to run in THIS invocation
    max_steps: int = 0              # env-step budget for THIS run; 0 = derive from max_episodes
    max_reset: int = 8192           # reset budget for THIS run; added to prior resets from checkpoint

    # --- Task control ---
    config_path: str = ""           # YAML experiment config (overrides below when set)
    task_order: str = "sequential"  # sequential | pure_random | sequence_random
    task_filter: str = ""           # comma-separated indices or task strings; empty = all

    # --- Non-episodic reset ---
    reset_mode: str = "per_episode"  # per_episode | none
    reset_robot: bool = True
    reset_unsuitable: bool = False
    unsuitable_detector: str = "low_z"
    # HSR respawn scope.
    #   "per_env"   (default) — for any env where the detector flagged either
    #                obj or recep, respawn BOTH actors of that env. Other envs
    #                untouched. Matches the "reset this env's task" semantic.
    #   "per_actor"          — respawn only the specific actor (obj OR recep)
    #                that tripped. Other actors of the same env stay put.
    #   "all"                — respawn every actor of every env whenever any
    #                env trips. Visually uniform across the batch.
    # YAML can override via `unsuitable_detector.reset_scope`.
    hsr_reset_scope: str = "per_env"
    # Dump the full pose state (every object slot, every receptacle slot, and the
    # gripper — position + quaternion) at the END of every segment, before any
    # HSR/LSR reset, to glob/segment_pose.csv. On by default: the record is
    # cheap (num_envs x (N+M+1) rows per segment) and is the only source for
    # where objects actually came to rest. Disable with --no-record-segment-pose.
    record_segment_pose: bool = True
    # Deprecated alias for --record-segment-pose. The old flag wrote
    # end_of_segment_xyz.csv with only the task-selected pair and only xyz;
    # it now enables the superset recorder. Kept so existing launch scripts
    # keep working.
    record_end_of_segment_xyz: bool = False
    enable_backward: bool = False
    backward_interval: int = 1
    num_groups: int = 0             # 0 = dynamically scale with available tasks

    # --- Algorithm ---
    # `ppo` (actor-critic, GAE) or `grpo` (critic-free, group-normalized
    # reward-to-go). Names follow AutoRL/RL4VLA so configs transfer.
    alg_name: str = "ppo"                # ppo | grpo
    # AutoRL's `alg_grpo_fix`: compute the reward statistics from NON-ZERO
    # rewards only. The shaped reward is a potential difference, so most steps
    # are exactly 0; including them would drag the mean to ~0 and make the std
    # a measure of sparsity rather than of policy spread.
    alg_grpo_fix: bool = True
    # What counts as one GRPO group. Three nesting levels, widest to narrowest.
    # ALL of them are scoped to a single segment — the group is (segment, block)
    # — so the ladder is a clean nesting. Sizes in brackets are for
    # `four_group_sequential_2x2` (64 envs = 4 YAML groups x 16 envs, 4 unique
    # tasks per group).
    #   "batch" [64] — one group per segment: every env, no further grouping.
    #                  This is AutoRL's statistic: its `train_grpo` runs on a
    #                  buffer holding exactly one segment of `num_envs`
    #                  trajectories, so `compute_returns_grpo` pools precisely
    #                  these 64. Verified bit-identical.
    #   "scene" [16] — one group per (segment, YAML group). Envs compared share
    #                  the same objects, receptacles and background overlay, but
    #                  may be running different tasks within that scene.
    #   "task"  [4]  — one group per (segment, fan-out sub-block). Only envs that
    #                  ran the SAME (object, receptacle) in the SAME segment are
    #                  compared. Narrowest and most apples-to-apples, but the
    #                  group can get small enough that its rewards are often all
    #                  identical — see the startup warning and
    #                  `grpo_adv_zero_frac`.
    # NOTE: "scene" here means a YAML `groups:` entry. It is unrelated to
    # `--scene` (the named lighting/overlay spec in `envs/scenes.py`) and to the
    # `scene` column in `eval_success.csv`.
    grpo_group_scope: str = "batch"      # batch | scene | task
    # What to divide the group-centred rewards by.
    #   "group"  — that group's own std (AutoRL / textbook GRPO).
    #   "global" — the std over the whole update. Groups are centred
    #              independently but weighted equally, which removes the
    #              difficulty bias per-group std introduces while keeping the
    #              gradient scale stable.
    #   "none"   — centring only (Dr. GRPO style).
    # Under `grpo_group_scope=batch` the group IS the whole update, so "group"
    # and "global" are the same thing. See `doc/grpo_autorl.md` §"是否需要 /std".
    grpo_std_scope: str = "group"        # group | global | none

    # --- PPO / buffer ---
    alg_ppo_epoch: int = 1
    alg_gradient_accum: int = 20    # phaseO-5 attempted 20→4 (mb 8→40 + LM-trunk checkpointing);
                                    # parity GREEN but 0.72× rollback speedup — checkpointing recompute
                                    # tax outweighed the per-minibatch overhead savings. Reverted.
    alg_entropy_coef: float = 0.0
    buffer_gamma: float = 0.99
    buffer_lambda: float = 0.95
    buffer_minibatch: int = 8       # phaseO-5 attempted 8→40; reverted (slower under checkpointing)
    buffer_inferbatch: int = 32     # phaseO-1 attempted 32→64; reverted (parity FAIL + no speedup at fixture)

    # --- VLA model ---
    # `policy` selects which adapter to instantiate. The default (`openvla`)
    # uses OpenVLA; `spatialvla` swaps in the SpatialVLA adapter from
    # `simpler_env.policies.spatialvla.spatialvla_train`. The default `vla_path`
    # / `vla_unnorm_key` follow the OpenVLA path; YAML training configs
    # (e.g. `configs/spatialvla_2x2_train.yaml`) override them to SpatialVLA's
    # `IPEC-COMMUNITY/spatialvla-4b-224-sft-bridge` + the `bridge_orig/1.0.0` key.
    policy: str = "openvla"          # "openvla" | "spatialvla"
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
    # 2 eval episodes per round: with num_envs=64 fan-out across 4 tasks, each
    # episode contributes 16 trials/task → 2 episodes = 32 trials/task per mid-
    # training eval, above the McNemar gate's 24-trial floor.
    num_eval_episode: int = 4
    eval_at_start: bool = False     # run eval before first training episode
    eval_single: bool = False
    eval_sequential: bool = False
    eval_sequences: int = 5         # permutation sequences (1 training + N-1 random)
    vla_checkpoint_interval: int = 8
    resume_from: str = ""           # path to checkpoint dir (auto-loads config + weights)

    # --- Logging ---
    wandb: bool = True
    wandb_dir: str = ""
    record_video: bool = True
    ppo_log: str = "ppo_log.txt"            # per-minibatch PPO progress (in glob_dir)
    eval_report: str = "eval_report.txt"    # per-eval success rate summary (in glob_dir)
    log_file: str = "run.log"               # tee of stdout for the whole run (in the wandb files/ dir, NOT glob/); empty string disables

class CronosRunner:
    """Coordinates the CRONOS training workflow."""
    
    def __init__(self, args):
        self.args = args
        # --resume_from auto-sets vla_load_path from checkpoint dir
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
        # Create + validate the root dir BEFORE wandb.init. Passing a relative,
        # not-yet-existing path (what scripts/train.sh does) makes wandb fall
        # back to the system temp dir with only a termwarn — see run_paths.py.
        wandb_root = prepare_wandb_dir(args.wandb_dir)
        if wandb_root:
            wandb_kwargs["dir"] = wandb_root
        run = wandb.init(**wandb_kwargs)
        verify_run_dir(run.dir, wandb_root)
        # X-axis: all metrics keyed on total_steps (environment steps)
        wandb.define_metric("total_steps")
        wandb.define_metric("eval_*", step_metric="total_steps")
        # `glob/` is a SIBLING of `files/`, not a child — so wandb does NOT
        # upload it. Everything durable (CSVs, checkpoints, videos) lives here
        # and is local-only; only `files/` is synced to the wandb cloud.
        self.glob_dir = Path(run.dir).parent / "glob"
        self.glob_dir.mkdir(parents=True, exist_ok=True)
        self.files_dir = Path(run.dir)  # wandb files/ dir (auto-synced)

        # Tee stdout to a log file in wandb files/ dir (auto-synced by wandb).
        # tqdm/wandb still write to stderr untouched.
        if args.log_file:
            log_path = self.files_dir / args.log_file
            log_fp = open(log_path, "a", buffering=1)
            class _Tee:
                def __init__(self, *streams): self.streams = streams
                def write(self, s):
                    for st in self.streams: st.write(s)
                def flush(self):
                    for st in self.streams: st.flush()
            sys.stdout = _Tee(sys.__stdout__, log_fp)

        # SuccessRecorder — dual-axis CSVs + wandb 5-chart panel + counters.json
        self.recorder = SuccessRecorder(self.glob_dir)

        # pre-init config dump — written BEFORE any SAPIEN scene is
        # constructed, so a crash during env init still leaves an on-disk
        # snapshot of exactly what was attempted.
        self._dump_run_config()

        # config_history.jsonl — append initial or resume entry.
        # Load previous config from checkpoint's run_config.json for diff.
        prev_config = None
        if args.resume_from:
            prev_config_path = Path(args.resume_from).parent / "run_config.json"
            if prev_config_path.exists():
                try:
                    prev_config = json.loads(prev_config_path.read_text())
                except Exception:
                    pass
        history_path = self.glob_dir / "config_history.jsonl"
        if not history_path.exists():
            self._append_config_history("initial", episode=args.resume_episode)
        if args.resume_from and prev_config is not None:
            self._append_config_history("resume", episode=args.resume_episode,
                                         prev_config=prev_config)

        # Device Management (AutoRL style)
        device_id = 0
        device_id_other = 1 if torch.cuda.device_count() > 1 else 0
        self.device = torch.device(f"cuda:{device_id}")
        
        # load YAML config BEFORE env creation (env_n/env_m must be
        # set first). The following this BEFORE policy creation so a YAML-pinned
        # `policy` / `vla_path` / `vla_unnorm_key` takes effect on the
        # adapter constructor (rather than after the wrong policy is already
        # loaded).
        self.yaml_config = None
        if args.config_path:
            from envs.config import load_cronos_config
            self.yaml_config = load_cronos_config(args.config_path)
            # YAML env params → Args (CLI overrides YAML). The following the
            # `policy` switch and per-policy VLA fields so a training config can
            # pin them without re-stating them on every command line.
            for field_name in ("env_n", "env_m", "num_envs",
                               "obj1_index", "obj2_index", "obj3_index",
                               "plate1_index", "plate2_index", "plate3_index",
                               "scene", "task_order",
                               "policy", "vla_path", "vla_unnorm_key",
                               "vla_lr", "vla_vhlr"):
                yaml_val = getattr(self.yaml_config, field_name, None)
                if yaml_val is not None and not self._cli_provided(field_name):
                    setattr(args, field_name, yaml_val)
            # env_n/env_m = max across groups (mixed N/M support)
            if self.yaml_config.groups and not self._cli_provided("env_n"):
                args.env_n = max(len(g.obj) for g in self.yaml_config.groups)
            if self.yaml_config.groups and not self._cli_provided("env_m"):
                args.env_m = max(len(g.recep) for g in self.yaml_config.groups)

        # Initialize Policy (matching AutoRL order: policy before env).
        # `--policy` switch: pick the adapter at construction time. Each
        # adapter exposes the same surface (`get_action`, `evaluate_actions`,
        # `get_value`, `prep_rollout/training`, `save/load`) and an
        # `act_token_len` attribute used to size the replay buffer.
        if args.policy == "spatialvla":
            from simpler_env.policies.spatialvla.spatialvla_train import SpatialVLAPolicy as _Policy
        elif args.policy == "openvla":
            from simpler_env.policies.openvla.openvla_train import OpenVLAPolicy as _Policy
        else:
            raise ValueError(
                f"--policy='{args.policy}' is not supported; expected one of "
                "'openvla' or 'spatialvla'."
            )
        self.policy = _Policy(args, device_id=device_id_other)
        # PPO or GRPO — same constructor signature, same `train_epoch` contract
        # (list of per-minibatch dicts), so `_run_ppo_update` and
        # `aggregate_train_results` are shared.
        if args.alg_name == "grpo":
            self.trainer = CronosGRPO(args, self.policy)
        else:
            self.trainer = CronosPPO(args, self.policy)

        # Initialize Env AFTER config + policy (env_n/env_m now correct)
        unnorm_state = self.policy.vla.get_action_stats(args.vla_unnorm_key)
        self.suite = TaskSuite()
        # forward the YAML `unsuitable_detector` block (if present)
        # so the wrapper can build a parametric workspace-AABB detector.
        det_cfg = getattr(self.yaml_config, "unsuitable_detector", None) if self.yaml_config else None
        # action_tokenizer routing (mirrors eval_only.py:174 / wrapper.py:188-216):
        # OpenVLA decodes via the wrapper's built-in 256-bin `bin_centers` table
        # (`action_tokenizer=None`); SpatialVLA passes its live
        # `SpatialActionTokenizer` so the wrapper's `_process_action`
        # SpatialVLA branch decodes the 3-token sequence per
        # `processing_spatialvla.py:247-250`. The following the runtime switch.
        wrapper_action_tokenizer = None
        if args.policy == "spatialvla":
            wrapper_action_tokenizer = self.policy.vla.action_tokenizer
        self.env = CronosWrapper(args, unnorm_state, self.suite, device=self.device,
                                 unsuitable_detector_cfg=det_cfg,
                                 action_tokenizer=wrapper_action_tokenizer)

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
        elif self.yaml_config and self.yaml_config.task_filter:
            task_filter = self.yaml_config.task_filter

        # Get task pool from env (CronosWrapper.__init__ already performs a seeded reset)
        task_pool = self.env.get_task_pool()

        # resolve symbolic refs (obj1, recep2, ...) in YAML groups + task_filter
        from envs.config import resolve_symbolic_task, has_symbolic_refs, build_obj_recep_name_maps
        from envs.scheduler import GroupState
        obj_names, recep_names = self._build_symbolic_map(task_pool)

        # Build per-group GroupState from YAML or auto-generate from flat task_pool
        group_states = []
        if self.yaml_config and self.yaml_config.groups:
            # access model_db for per-group symbolic resolution
            env_unwrapped = self.env.env.unwrapped
            model_db_carrot = env_unwrapped.model_db_carrot
            model_db_plate = env_unwrapped.model_db_plate

            # New per-group format
            for g in self.yaml_config.groups:
                # per-group symbolic maps from this group's own obj/recep indices
                g_obj_names, g_recep_names = build_obj_recep_name_maps(
                    g.obj, g.recep, model_db_carrot, model_db_plate)
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
                    num_envs=g.num_envs,
                ))
            self.scheduler = TaskScheduler(
                group_states=group_states,
                mode=args.task_order,
                num_envs=args.num_envs,
                fan_out=self.yaml_config.fan_out,
            )
            # pass group specs to wrapper for per-env object indices
            self.env.set_group_specs(self.yaml_config.groups)
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

        # restore scheduler cursor from checkpoint if available
        if hasattr(self, '_resume_scheduler_state'):
            self.scheduler.load_state(self._resume_scheduler_state)
            print(f"[SCHEDULER] restored cursor from checkpoint")
        print(f"[SCHEDULER] mode={args.task_order}, pool={self.scheduler.task_pool}")

        # GRPO grouping: per-env sub-block id, fixed for the whole run.
        self._build_grpo_env_blocks()

        # buffer width comes from the policy, NOT from
        # `get_action_dim` (= continuous DoF = 7 for both backends, drives the
        # wrapper decoder). For OpenVLA tokens==DoF==7; for SpatialVLA
        # tokens=3 / DoF=7. Reading `act_token_len` off the live policy keeps
        # them aligned without re-deriving from `vla_path`.
        self.buffer = CronosReplayBuffer(args, act_dim=self.policy.act_token_len)

        # deterministic mmap cleanup on SIGINT. atexit alone races
        # against still-open mmap fds at interpreter shutdown, producing the
        # ENOTEMPTY silly-rename error
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

    def _build_grpo_env_blocks(self):
        """Map each env to a block id, once per scope, fixed for the whole run.

        The scheduler lays every YAML group's envs out contiguously and, with
        `fan_out=True`, splits each group into one sub-block per unique task
        (`envs/scheduler.py:297-311`). Config validation V22/V23 guarantees the
        split is even, so the three nested levels are:

            batch block     ->  all envs (a single block)
            scene block g   ->  envs [starts[g], starts[g+1])
            task block g,t  ->  envs [starts[g] + t*sub, starts[g] + (t+1)*sub)
                                sub = groups[g].num_envs // len(task_pool)

        Block membership is fixed for the run; what rotates each segment is
        WHICH task a given sub-block runs (`_fan_out_offsets`). That is why the
        buffer-side key combines the block id with the segment index — see
        `_grpo_group_ids`.

        With `fan_out=False`, or a single-task group, the task block degenerates
        to the scene block (all the group's envs share one task anyway).

        Sets `self._grpo_blocks`: {scope: (per-env id array, block count)}.
        """
        scene_blocks, n_scene = [], 0
        task_blocks, n_task = [], 0
        for gs in self.scheduler.group_states:
            scene_blocks.extend([n_scene] * gs.num_envs)
            n_scene += 1

            n_tasks = len(gs.task_pool)
            if self.scheduler.fan_out and n_tasks > 1:
                sub = gs.num_envs // n_tasks
                for t in range(n_tasks):
                    task_blocks.extend([n_task + t] * sub)
                # Remainder envs (only reachable if V23 was bypassed) join the
                # last block rather than forming a ragged one.
                remainder = gs.num_envs - sub * n_tasks
                if remainder > 0:
                    task_blocks.extend([n_task + n_tasks - 1] * remainder)
                n_task += n_tasks
            else:
                task_blocks.extend([n_task] * gs.num_envs)
                n_task += 1

        ne = self.args.num_envs
        # Pad any env not covered by a group into a final catch-all block.
        for blocks, n in ((scene_blocks, n_scene), (task_blocks, n_task)):
            if len(blocks) < ne:
                blocks.extend([n] * (ne - len(blocks)))

        # Block count is read back off the array so the catch-all pad block, if
        # one was needed, is counted exactly once.
        self._grpo_blocks = {"batch": (np.zeros(ne, dtype=np.int64), 1)}
        for scope, blocks in (("scene", scene_blocks), ("task", task_blocks)):
            arr = np.asarray(blocks[:ne], dtype=np.int64)
            self._grpo_blocks[scope] = (arr, int(arr.max()) + 1)

        self._report_grpo_grouping()

    def _report_grpo_grouping(self):
        """Print the selected grouping and warn when it is too narrow to help."""
        scope = self.args.grpo_group_scope
        if self.args.alg_name != "grpo":
            return
        arr, n_blocks = self._grpo_blocks[scope]
        sizes = np.bincount(arr, minlength=n_blocks)
        print(f"[GRPO] group_scope={scope}: {n_blocks} blocks/segment, "
              f"sizes={sizes.tolist()}, std_scope={self.args.grpo_std_scope}")
        smallest = int(sizes.min())
        if smallest < 8:
            # A group whose rewards all come out identical contributes zero
            # gradient, and that gets rapidly more likely as the group shrinks.
            # `grpo_adv_zero_frac` in wandb measures the real rate.
            print(f"[WARN] smallest GRPO group has {smallest} envs. With few envs "
                  f"per group the reward spread is often zero, which makes the "
                  f"whole group's advantage zero — watch `grpo_adv_zero_frac`. "
                  f"Raise per-group num_envs, widen --grpo-group-scope, or accept "
                  f"the sparser signal.")
            if self.args.grpo_std_scope == "group":
                print(f"[WARN] --grpo-std-scope group divides by each group's own "
                      f"std, which at this group size is both noisy and acts as a "
                      f"per-group weight (imbalanced groups get amplified). "
                      f"--grpo-std-scope global keeps the scale but drops the bias.")

    def _grpo_group_ids(self, n_slots):
        """Per-slot group id for `buffer.compute_grpo_returns`.

        Buffer slot `k` is segment `k // num_envs`, env `k % num_envs`. All three
        scopes key on **(segment, block)**, never on block alone:

        - `task` must, because `TaskScheduler.update_index` rotates
          `_fan_out_offsets` every segment, so the same sub-block runs a
          different task in the next segment and pooling across segments would
          compare unlike tasks.
        - `scene` does not strictly have to — a scene block runs the same *set*
          of tasks every segment — but does anyway, for consistency and because
          segment N+1 starts from segment N's end state under non-episodic
          continuity, so the two are not drawn from the same state distribution.
        - `batch` is one block, so its group is the segment itself: `num_envs`
          trajectories. That is exactly what AutoRL normalizes over — its
          `train_grpo` runs on a buffer holding a single segment — so pooling
          the whole `ppo_update_len` window instead would silently change the
          statistic.
        """
        env_block, n_blocks = self._grpo_blocks[self.args.grpo_group_scope]
        ne = self.args.num_envs
        idx = np.arange(n_slots, dtype=np.int64)
        return (idx // ne) * n_blocks + env_block[idx % ne]

    def _dump_run_config(self):
        """M1: write glob_dir/run_config.{json,yaml} before any SAPIEN
        init. Contents: resolved Args + seed + start_time + git_rev (best-
        effort) + cronos_version. The snapshot lands *before* the policy/env
        are constructed, so even a crash during env init leaves a diff-able
        record of exactly what the user attempted."""
        cfg = dict(self.args.__dict__)
        cfg["start_time"] = datetime.datetime.utcnow().isoformat() + "Z"
        cfg["cronos_version"] = "V0.3"
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

    def _append_config_history(self, event: str, episode: int = 0,
                                prev_config: dict = None, reason: str = ""):
        """M5: Append an entry to config_history.jsonl.

        Args:
            event: "initial" or "resume"
            episode: current episode count at time of event
            prev_config: previous config dict (for diff on resume)
            reason: optional human reason for the config change
        """
        import datetime as _dt
        history_path = self.glob_dir / "config_history.jsonl"

        entry = {
            "episode": episode,
            "timestamp": _dt.datetime.utcnow().isoformat() + "Z",
            "event": event,
        }

        current_cfg = dict(self.args.__dict__)
        # Coerce non-serializable values
        for k, v in current_cfg.items():
            if isinstance(v, Path):
                current_cfg[k] = str(v)

        if event == "initial":
            entry["config"] = current_cfg
        elif event == "resume" and prev_config is not None:
            # Compute diff: keys where values changed
            diff = {}
            all_keys = set(current_cfg.keys()) | set(prev_config.keys())
            for k in all_keys:
                old_v = prev_config.get(k)
                new_v = current_cfg.get(k)
                if old_v != new_v:
                    diff[k] = [old_v, new_v]
            entry["diff"] = diff
            if reason:
                entry["reason"] = reason
        else:
            entry["config"] = current_cfg

        with open(history_path, "a") as f:
            f.write(json.dumps(entry, default=str) + "\n")

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
        """Check if a CLI flag was explicitly passed (heuristic: present in sys.argv).

        Handles both --field_name and --field-name variants (tyro uses dashes).
        """
        import sys
        dash_name = field_name.replace("_", "-")
        return any(f"--{field_name}" in arg or f"--{dash_name}" in arg for arg in sys.argv)

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
        assert args.alg_name in ("ppo", "grpo"), \
            f"alg_name must be 'ppo' or 'grpo', got '{args.alg_name}'"
        assert args.grpo_group_scope in ("batch", "scene", "task"), \
            f"grpo_group_scope must be 'batch', 'scene' or 'task', got '{args.grpo_group_scope}'"
        assert args.grpo_std_scope in ("group", "global", "none"), \
            f"grpo_std_scope must be 'group', 'global' or 'none', got '{args.grpo_std_scope}'"
        if args.reset_mode == "none" and not args.reset_unsuitable:
            # user opts in to non-episodic without HSR
            # (object respawn). The original constraint was that under
            # `reset_mode=none` there's no env.reset() to recover envs
            # whose objects fell off the table or wedged into impossible
            # poses — `reset_unsuitable=True` was the only safety valve.
            # Relaxed to a warning so the experimental non-episodic spec is allowed;
            # stuck envs will accumulate over training and may degrade the
            # rollout's effective sample diversity. Caller has been told.
            print("[WARN] reset_mode=none + reset_unsuitable=False — stuck "
                  "envs will not recover via HSR; training may degrade if "
                  "objects accumulate in impossible states.")

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

    @torch.no_grad()
    def eval(self, iteration, task_idx, obj_set, object, receptacle, prefix="eval", reset=True, group_idx=0):
        """— Standalone evaluation (port of AutoRL `render`).

        All envs run the SAME (object[i], receptacle[i]) — typically the caller broadcasts
        a single task as `[object_str]*num_envs / [receptacle_str]*num_envs`. Runs one
        `segment_len`-step rollout with deterministic policy, records per-env videos,
        and returns aggregated stats. **No fan-out** — that's `eval_all_groups`.

        This is the eval mode used by `--eval-single`, `--eval-sequential`, and
        `eval_only.py`; `eval_all_groups` (fan-out rotation) is reserved for
        training-time eval (`--eval-at-start` and the periodic eval inside `train()`).

        Args:
            iteration: epoch / sequence index (logging only)
            task_idx: task index within iteration (logging + video name)
            obj_set: physics randomization set (e.g. "rand", "rand_ood")
            object: list[str] of length num_envs — object names
            receptacle: list[str] of length num_envs — receptacle names
            prefix: video sub-dir name under `glob/eval_videos/`
            reset: True → env.reset (start of a sequence); False → continue from
                   current state (chained sequence steps, AutoRL-aligned)
            group_idx: which YAML group's physical objects to broadcast to all envs
                       (default 0; multi-group sequential eval is work)
        """
        if reset:
            self.policy.prep_rollout()
            obs, _, _ = self.env.reset(
                obj_set_override=obj_set,
                group_idx_override=group_idx,
                skip_scheduler=True,
            )
            self.env.set_task(object, receptacle)
        else:
            self.env.set_task(object, receptacle)
            # Reopen the measurement window on a continued segment — see
            # `CronosWrapper.begin_segment`. Without it `_elapsed_steps` keeps
            # the env truncated from step 1, turning `success` into a
            # time-average and letting grasp flags carry over across tasks.
            # Scene state is untouched, so chained-sequence continuity holds.
            self.env.begin_segment()
            obs = self.env.get_obs_image()
        instruction = self.env.get_language_instructions()

        print(f"  [{prefix}] {object[0]} -> {receptacle[0]}")

        record = self.args.record_video
        if record:
            video_frames = [[] for _ in range(self.args.num_envs)]

        env_infos = defaultdict(list)
        for _ in tqdm(range(self.args.segment_len), desc=f"eval {prefix}", leave=False):
            if record:
                for env_i in range(self.args.num_envs):
                    video_frames[env_i].append(obs[env_i].cpu().numpy().copy())
            val, action, logp = self._get_action(obs, instruction, deterministic=True)
            obs, reward, truncated, env_info = self.env.step(action)
            if "episode" in env_info:
                for k, v in env_info["episode"].items():
                    env_infos[k] += v

        if record:
            for env_i in range(self.args.num_envs):
                video_frames[env_i].append(obs[env_i].cpu().numpy().copy())
            render_dir = self.glob_dir / "eval_videos" / prefix
            render_dir.mkdir(parents=True, exist_ok=True)
            successes = env_infos.get("success", [0] * self.args.num_envs)
            for i in range(self.args.num_envs):
                s = int(successes[i]) if i < len(successes) else 0
                obj_safe = str(object[i]).replace(" ", "_")
                rec_safe = str(receptacle[i]).replace(" ", "_")
                images_to_video(
                    video_frames[i], str(render_dir),
                    f"task{task_idx}-env{i}-{obj_safe}_{rec_safe}-s{s}",
                    fps=10, verbose=False,
                )

        return {k: float(np.mean(v)) for k, v in env_infos.items() if v}

    def eval_all_groups(self, iteration, obj_set, prefix="eval"):
        """M4: Evaluate all groups using per-env rotation.

        Assignment formula:
            task_for_env_i_at_episode_e = eval_tasks[(i % n_eval_tasks + e) % n_eval_tasks]
        where i is the env index within its group.

        Each env resets between episodes. Results are accumulated per (group, task)
        and averaged at the end.

        Returns: list of (g_idx, group_name, task, n_samples, stats_dict) tuples.
        """
        self.policy.prep_rollout()

        per_group_evals = self.scheduler.get_eval_tasks_per_group()
        n_groups = len(per_group_evals)
        num_eps = self.args.num_eval_episode

        # Pre-extract (obj, recep) for each group's eval tasks + per-group sizes
        group_eval_info = []  # [(g_name, g_size, [(task_str, obj, recep), ...]), ...]
        group_starts = []     # cumulative env start indices
        offset = 0
        for g_name, g_size, g_tasks in per_group_evals:
            tasks_info = []
            for task in g_tasks:
                obj, recep = TaskScheduler._extract_obj_recep(task)
                tasks_info.append((task, obj, recep))
            group_eval_info.append((g_name, g_size, tasks_info))
            group_starts.append(offset)
            offset += g_size

        # Accumulate results: (g_name, task_str) -> {"success": [0,1,1,...], ...}
        accum = {}
        for g_name, g_size, tasks_info in group_eval_info:
            for task_str, _, _ in tasks_info:
                accum[(g_name, task_str)] = defaultdict(list)

        # Log rotation table on first call
        print(f"  Eval rotation: {num_eps} episodes, {n_groups} groups")
        for g_idx, (g_name, g_size, tasks_info) in enumerate(group_eval_info):
            n_tasks = len(tasks_info)
            print(f"  [{g_name}] {n_tasks} eval tasks, {g_size} envs "
                  f"({g_size // n_tasks} envs/task/ep, "
                  f"{num_eps * g_size // n_tasks} samples/task total)")
            for t_idx, (task_str, _, _) in enumerate(tasks_info):
                print(f"    T{t_idx}: {task_str}")

        for ep in range(num_eps):
            # Build per-env task assignment for this episode using rotation formula
            envs_obj = []
            envs_recep = []
            env_task_map = [None] * self.args.num_envs  # env_idx -> (g_idx, task_str)

            for g_idx, (g_name, g_size, tasks_info) in enumerate(group_eval_info):
                n_tasks = len(tasks_info)
                g_start = group_starts[g_idx]
                for local_i in range(g_size):
                    t_idx = (local_i % n_tasks + ep) % n_tasks
                    task_str, obj, recep = tasks_info[t_idx]
                    envs_obj.append(obj)
                    envs_recep.append(recep)
                    env_task_map[g_start + local_i] = (g_idx, g_name, task_str)

            # Pad remainder envs (if total per-group sizes < num_envs)
            while len(envs_obj) < self.args.num_envs:
                envs_obj.append(envs_obj[-1])
                envs_recep.append(envs_recep[-1])

            # Reset env and assign tasks
            obs, _, _ = self.env.reset(obj_set_override=obj_set, skip_scheduler=True)
            self.env.set_task(envs_obj, envs_recep)
            instruct = self.env.get_language_instructions()

            # Record video for all envs across ALL eval episodes
            record_this_ep = self.args.record_video
            total_group_envs_count = sum(gs for _, gs, _ in group_eval_info)
            if record_this_ep:
                video_frames = {i: [] for i in range(total_group_envs_count)}

            # Rollout one episode
            env_infos = defaultdict(list)
            for _ in tqdm(range(self.args.segment_len),
                          desc=f"eval {prefix} ep{ep+1}/{num_eps}", leave=False):
                with torch.no_grad():
                    val, action, logp = self._get_action(obs, instruct, deterministic=True)
                if record_this_ep:
                    for env_i in range(total_group_envs_count):
                        video_frames[env_i].append(obs[env_i].cpu().numpy().copy())
                obs, reward, truncated, env_info = self.env.step(action)
                if "episode" in env_info:
                    for k, v in env_info["episode"].items():
                        env_infos[k] += v

            # Save eval videos: eval_videos/ep{outer_ep}/{prefix}/eval_ep{ep+1}/env{i}.mp4
            if record_this_ep:
                # append the final post-step obs so each eval video has
                # T+1 frames (matches AutoRL `render` and the standalone-eval
                # `runner.eval`). Otherwise the recorded video ends at the input
                # to the last action and the natural end-state is missing.
                for env_i in range(total_group_envs_count):
                    video_frames[env_i].append(obs[env_i].cpu().numpy().copy())
                eval_video_dir = (self.glob_dir / "eval_videos"
                                  / f"ep{iteration + 1}" / prefix / f"eval_ep{ep + 1}")
                eval_video_dir.mkdir(parents=True, exist_ok=True)
                for env_i, frames in video_frames.items():
                    if frames:
                        images_to_video(frames, str(eval_video_dir),
                                        f"env{env_i}", fps=10, verbose=False)

            # Distribute per-env results to per-(group, task) accumulators
            total_group_envs = sum(gs for _, gs, _ in group_eval_info)
            for env_idx in range(min(len(env_task_map), total_group_envs)):
                entry = env_task_map[env_idx]
                if entry is None:
                    continue
                _, g_name_r, task_str = entry
                for k, vals in env_infos.items():
                    if env_idx < len(vals):
                        accum[(g_name_r, task_str)][k].append(vals[env_idx])

        # Compute mean stats per (group, task) and build results
        results = []
        for g_idx, (g_name, g_size, tasks_info) in enumerate(group_eval_info):
            n_tasks = len(tasks_info)
            n_samples = num_eps * g_size // n_tasks
            for task_str, _, _ in tasks_info:
                key = (g_name, task_str)
                stats = {k: float(np.mean(v)) for k, v in accum[key].items()}
                actual_samples = len(accum[key].get("success", []))
                print(f"  [{g_name}] {task_str} "
                      f"({actual_samples} samples): "
                      f"success={stats.get('success', 0.0):.4f} "
                      f"grasp={stats.get('consecutive_grasp', 0.0):.4f}")
                results.append((g_idx, g_name, task_str, n_samples, stats))

        return results

    def _flush_rollout_rows(self):
        """Backfill GAE-derived columns onto pending rows, then write them out.

        Must run after `buffer.compute_gae()` and before `buffer.reset()` — the
        arrays are only valid in that window.

        The buffer lays segments out along its env axis: `end_segment()` advances
        `curr_env` by `num_envs` per segment, so buffer slot `k` is segment
        `k // num_envs`, env `k % num_envs`. `_pending_rollout_rows` is appended
        in exactly that order (segment-major, env-minor), so row index == slot
        index and no key matching is needed. Rows beyond the buffer's filled
        width keep empty GAE columns rather than borrowing another segment's.
        """
        rows = getattr(self, "_pending_rollout_rows", None)
        if not rows:
            return
        n_slots = self.buffer.num_env
        if n_slots > 0:
            # Mean over the segment's timesteps, per buffer slot.
            returns = self.buffer.returns[:, :n_slots, 0].mean(axis=0)
            values = self.buffer.value_preds[:-1, :n_slots, 0].mean(axis=0)
            advs = self.buffer.advantages[:, :n_slots, 0].mean(axis=0)
            for k, row in enumerate(rows):
                if k >= n_slots:
                    break
                row["return_gae"] = float(returns[k])
                row["value_mean"] = float(values[k])
                row["advantage_mean"] = float(advs[k])
        self.recorder.log_rollout_segments(rows)
        self._pending_rollout_rows = []

    def _run_ppo_update(self, ppo_log_path):
        """Runs one policy update on the current buffer, then resets it.

        Name kept as `_run_ppo_update` under GRPO too: `tools/bench_rollout.py`
        monkey-patches this method by name to time the update phase.
        """
        # The only algorithm-dependent step: how `returns` / `advantages` are
        # filled. Everything downstream (row flush, minibatch generator,
        # train_epoch contract, result aggregation) is shared.
        if self.args.alg_name == "grpo":
            self.buffer.compute_grpo_returns(
                group_ids=self._grpo_group_ids(self.buffer.num_env),
                fix=self.args.alg_grpo_fix,
                std_scope=self.args.grpo_std_scope,
            )
        else:
            self.buffer.compute_gae()
        # Drain the rollout rows for the segments this update covers while the
        # returns/advantages arrays are still live (buffer.reset() invalidates
        # them below).
        self._flush_rollout_rows()
        self.policy.prep_training()
        train_results = []
        for _ in range(self.args.alg_ppo_epoch):
            train_results.extend(self.trainer.train_epoch(self.buffer, log_path=ppo_log_path))
        self.buffer.reset()
        self.policy.prep_rollout()
        return train_results

    def run_rollout(self, ppo_log_path=None, episode=0, episode_base_steps=0, episode_base_resets=0):
        """Executes a non-episodic rollout with instruction switching, partial resets,
        and mid-rollout PPO training every ppo_update_len steps (matching AutoRL)."""
        self.policy.prep_rollout()

        # reset_mode=none skips env.reset() after the first episode.
        # The sim continues from the live state; only reset_unsuitable + task
        # switching drive state changes.
        if self.args.reset_mode == "none" and hasattr(self, '_no_reset_obs'):
            obs = self._no_reset_obs
            instruct = self.env.get_language_instructions()
        else:
            obs, instruct, _ = self.env.reset()
        self.buffer.warmup(obs, instruct)

        # Record video for all envs
        self._video_envs = list(range(self.args.num_envs))
        self.video_frames = {i: [] for i in self._video_envs}

        # with fan_out, each YAML group is internally split by the scheduler.
        # num_groups here = total sub-groups across all YAML groups (for logging only).
        if self.scheduler.fan_out:
            self.num_groups = sum(
                len(gs.task_pool) for gs in self.scheduler.group_states)
        else:
            self.num_groups = self.args.num_groups if self.args.num_groups > 0 else len(self.scheduler.task_pool)

        forward_count = 1
        segment_id = 0
        steps_since_ppo = 0
        all_train_results = []

        # Track current tasks for logging/naming
        objs, receps = self.scheduler.get_next_tasks()
        self.last_info = {}
        info = {}

        # per-task rollout metrics accumulator. Keyed by
        # `(obj, recep)` — the task PAIR that was active DURING the
        # just-completed segment (read BEFORE the scheduler advances, see the
        # capture point below). The wrapper writes `info["episode"]` only at
        # truncation (wrapper.py:249-253), which fires exactly at the
        # `task_len` boundary, so each segment contributes one batch of
        # per-env values per metric. Logged as `rollout/<task>/{success,
        # consecutive_grasp, is_src_obj_grasped}` by `train()` after this
        # rollout returns.
        self._rollout_per_task = {}        # {(obj, recep): {metric: [floats]}}

        # Per-env, per-segment rollout record (rollout_success.csv). Same
        # granularity as eval's per-trial rows, taken at rollout time.
        #
        # `reward_sum` is the plain sum of the shaped reward the policy actually
        # received. Note `RewardShaper` emits a potential *difference*, so this
        # telescopes to (potential at segment end - potential at segment start)
        # — it is the PPO training signal, not a conventional return, which is
        # why the discounted sum is tracked separately below.
        seg_reward_sum = torch.zeros(self.args.num_envs, device=self.device)
        seg_disc_return = torch.zeros(self.args.num_envs, device=self.device)
        seg_step = 0                        # steps elapsed inside current segment
        # Rows wait here until the PPO update computes GAE for their segment;
        # `_run_ppo_update` fills in return_gae/value/advantage and flushes.
        self._pending_rollout_rows = []

        # env index -> YAML group name, so each row can be attributed to the
        # group whose objects/background that env is running.
        if self.yaml_config and self.yaml_config.groups:
            from envs.config import get_group_starts
            _starts = get_group_starts(self.yaml_config.groups)
            env_group = []
            for gi, g in enumerate(self.yaml_config.groups):
                env_group.extend([g.name] * (_starts[gi + 1] - _starts[gi]))
            env_group += ["default"] * max(0, self.args.num_envs - len(env_group))
        else:
            env_group = ["default"] * self.args.num_envs

        for step_idx in tqdm(range(self.args.episode_len), desc="Rollout", leave=False):
            # 1. Action Prediction (Vectorized/Micro-batched)
            with torch.no_grad():
                value, action, logprob = self._get_action(obs, instruct, deterministic=False)

            # 2. Environment Step
            next_obs, reward, truncated, info = self.env.step(action)

            # 3. Buffer Storage
            self.buffer.insert(next_obs, action, logprob, value, reward, 1.0 - truncated.float())

            # 3b. Per-env reward accumulation for rollout_success.csv. Both sums
            # run over the same r_t the buffer just stored, so they can be read
            # against `return_gae` without re-deriving anything.
            r_flat = reward.reshape(-1).to(seg_reward_sum.dtype)
            seg_reward_sum += r_flat
            seg_disc_return += (self.args.buffer_gamma ** seg_step) * r_flat
            seg_step += 1

            # 4. Video Recording — M1: append POST-step obs only, matching
            # AutoRL train_ms3_ppo.py:580-581. Recording pre-step `obs` at step 0
            # of seg N+1 would capture the direct return of reset_unsuitable_envs/
            # reset_robot, whose camera buffer is stale relative to the masked
            # set_pose. By using next_obs (post-env.step), every recorded frame
            # is at least one physics tick past any reset, so the camera buffer
            # is always refreshed. T frames per training segment.
            if self.args.record_video:
                for i in self._video_envs:
                    self.video_frames[i].append(next_obs[i].cpu().numpy().copy())

            # 5. Segment & Task Switching Logic
            if (step_idx + 1) % self.args.task_len == 0:
                # capture per-task rollout outcomes BEFORE the
                # scheduler advances. `info["episode"]` is populated by the
                # wrapper at truncation (`wrapper.py:249-253`), which fires at
                # this exact `task_len` boundary. Attributing to `(objs[i],
                # receps[i])` here — the still-current pair for the env that
                # just completed — preserves the task→outcome mapping; doing
                # it AFTER `scheduler.get_next_tasks()` advances mislabels
                # every segment's metrics by one task.
                ep_info = info.get("episode", {}) if isinstance(info, dict) else {}
                if ep_info:
                    for env_i in range(self.args.num_envs):
                        if env_i >= len(objs) or env_i >= len(receps):
                            continue
                        key = (objs[env_i], receps[env_i])
                        slot = self._rollout_per_task.setdefault(
                            key, {"success": [], "consecutive_grasp": [], "is_src_obj_grasped": []}
                        )
                        for metric in ("success", "consecutive_grasp", "is_src_obj_grasped"):
                            vals = ep_info.get(metric)
                            if vals is not None and env_i < len(vals):
                                slot[metric].append(float(vals[env_i]))

                # per-env row for rollout_success.csv, captured at the same
                # point and from the same `ep_info` as the per-task aggregate
                # above — so `success` here is the segment-terminal value, the
                # identical definition eval reports. `return_gae` is left unset;
                # `_run_ppo_update` backfills it once GAE runs for this segment.
                seg_total_steps = episode_base_steps + (step_idx + 1) * self.args.num_envs
                backward_mask = self.env.reward_shaper.backward.reshape(-1).tolist()
                r_sum = seg_reward_sum.tolist()
                d_ret = seg_disc_return.tolist()

                def _metric(name, env_i):
                    """Per-env terminal value, or '' when the env didn't report."""
                    vals = ep_info.get(name)
                    return float(vals[env_i]) if vals and env_i < len(vals) else ""

                for env_i in range(self.args.num_envs):
                    obj_i = objs[env_i] if env_i < len(objs) else ""
                    rec_i = receps[env_i] if env_i < len(receps) else ""
                    self._pending_rollout_rows.append({
                        "episode": episode,
                        "segment": segment_id + 1,
                        "total_steps": seg_total_steps,
                        # Live total, not base+delta: `soft_reset_count` is
                        # already cumulative across the run, so adding the
                        # episode base would count prior soft resets twice.
                        "total_resets": self.hard_reset_count + self.soft_reset_count,
                        "env_idx": env_i,
                        "group": env_group[env_i] if env_i < len(env_group) else "default",
                        "task": f"put {obj_i} on {rec_i}",
                        "obj": obj_i,
                        "recep": rec_i,
                        # In LSR / noep the policy alternates forward and backward
                        # goals, but `success` always reflects the FORWARD
                        # predicate. Without this column a backward segment's 0
                        # is indistinguishable from a failed forward one.
                        "direction": "backward" if (env_i < len(backward_mask)
                                                    and backward_mask[env_i]) else "forward",
                        "success": _metric("success", env_i),
                        "consecutive_grasp": _metric("consecutive_grasp", env_i),
                        "is_src_obj_grasped": _metric("is_src_obj_grasped", env_i),
                        "reward_sum": r_sum[env_i],
                        "return_discounted": d_ret[env_i],
                        "return_gae": "",
                        "value_mean": "",
                        "advantage_mean": "",
                    })
                seg_reward_sum.zero_()
                seg_disc_return.zero_()
                seg_step = 0

                # save the just-completed segment (T post-step frames)
                # BEFORE any reset runs. The reset+_settle then executes in
                # physics but contributes no recorded frame — the next segment's
                # first frame will be one env.step past the reset, with a
                # refreshed camera buffer.
                if self.args.record_video:
                    self.save_video_segment(iteration=self.iteration, segment_id=segment_id)

                # dump the full pose state (every object + receptacle slot and
                # the gripper, position + quaternion) BEFORE the HSR/LSR reset
                # overwrites it, so the coordinates are the steady state the
                # policy produced rather than a post-respawn placement.
                if self.args.record_segment_pose or self.args.record_end_of_segment_xyz:
                    self._record_segment_pose(episode, segment_id, seg_total_steps)

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
                    objs, receps = self.scheduler.get_next_tasks()
                    self.env.set_task(objs, receps)
                    forward_count += 1

                # Resets for non-episodic continuity. M3 (HSR Bug 1):
                # two independent ifs, matching AutoRL train_ms3_ppo.py:637-643.
                # HSR (object respawn) and LSR (robot reset) are now truly
                # orthogonal; HSR-only no longer silently also resets the robot.
                if self.args.reset_unsuitable:
                    next_obs = self.env.reset_unsuitable_envs()
                    self.soft_reset_count = self.env.reset_strategy.reset_unsuitable_count
                    print(f"Total unsuitable resets: {self.soft_reset_count}")
                if self.args.reset_robot:
                    next_obs = self.env.reset_robot()
                else:
                    # EER off. `reset_robot()` is the ONLY thing that zeroes
                    # `_elapsed_steps` on the training path, and `truncated`
                    # feeds the buffer's masks — left at the time limit the env
                    # reports truncated on every step, masks go to 0, and GAE
                    # degenerates to `returns = reward` with no bootstrapping.
                    # `begin_segment()` reopens the accounting window without
                    # touching the robot, which is the whole point of turning
                    # EER off.
                    self.env.begin_segment()

                # Prepare for NEW segment instructions
                instruct = self.env.get_language_instructions()
                if self.yaml_config and self.yaml_config.groups:
                    from envs.config import get_group_starts
                    g_starts = get_group_starts(self.yaml_config.groups)
                    n_yaml_groups = len(self.yaml_config.groups)
                else:
                    n_yaml_groups = 1
                    g_starts = [0]
                group_instrs = [instruct[g_starts[g]] for g in range(n_yaml_groups)]
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

        # stash live obs for reset_mode=none so next episode skips env.reset
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

        # eval before first training episode (e.g. to measure pretrained checkpoint)
        if self.args.eval_at_start:
            total_steps_0 = prior_steps
            total_resets_0 = prior_resets
            eval_log = {"episode": start_episode, "total_steps": total_steps_0, "total_resets": total_resets_0}
            report_header = (
                f"{'=' * 50}\n"
                f"Eval-at-start | Steps {total_steps_0} | Resets {total_resets_0}\n"
                f"{'=' * 50}\n"
            )

            def _run_eval_start(obj_set, kind_prefix, kind_label):
                print(f"Evaluating {kind_label} at start (steps={total_steps_0}, "
                      f"{self.args.num_eval_episode} episodes)")
                results_raw = self.eval_all_groups(start_episode, obj_set, prefix=kind_prefix)
                results = []
                for g_idx, g_name, task, n_samples, stats in results_raw:
                    scalars = self.recorder.log_eval(
                        episode=start_episode,
                        total_steps=total_steps_0,
                        total_resets=total_resets_0,
                        eval_kind=kind_label,
                        group=g_name,
                        task=task,
                        scene="default",
                        n_envs=n_samples,
                        success=stats.get("success", 0.0),
                        grasp=stats.get("consecutive_grasp", 0.0),
                        obj_grasped=stats.get("is_src_obj_grasped", 0.0),
                    )
                    eval_log.update(scalars)
                    results.append((task, stats))
                return results

            in_domain_results = _run_eval_start(self.args.obj_set, "in_domain", "in_domain")
            eval_log.update(self.recorder.build_wandb_eval_panel("in_domain"))
            ood_results = _run_eval_start("rand_ood", "out_of_domain", "out_of_domain")
            eval_log.update(self.recorder.build_wandb_eval_panel("out_of_domain"))
            wandb.log(eval_log, step=total_steps_0)
            self._write_eval_report(report_header + "\nIn-Domain Evaluation:\n", in_domain_results)
            self._write_eval_report("Out-of-Domain Evaluation:\n", ood_results)
            print("Eval-at-start complete.\n")

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

            # durable counters sidecar for resume continuity
            self.recorder.write_counters(
                episode=episode,
                total_steps=total_steps,
                total_resets=total_resets,
            )

            # 3. Logging (dual-axis: both episode and total_steps).
            # aggregate the update's minibatch list instead of
            # logging `train_results[-1]` (the last minibatch only, dominated
            # by post-step drift). `_run_ppo_update` rebuilds `train_results`
            # per update, so the list is exactly this episode's minibatches.
            # `g2_logp_gap` is read from `train_results[0]` (the first
            # minibatch — rollout θ unchanged, gap ≈ 0; persistent non-zero
            # is a wiring bug — see `training/ppo.aggregate_train_results`).
            if train_results:
                agg = aggregate_train_results(train_results)
                log_payload = {
                    **agg,
                    "episode": episode,
                    "total_steps": total_steps,
                    "total_resets": total_resets,
                }
                # per-task rollout metrics, attributed at segment
                # boundary BEFORE the scheduler advanced (`run_rollout`
                # capture). Each (obj, recep) pair gets one wandb scalar per
                # metric per episode, averaged across all envs that ran that
                # pair this episode. Keys: `rollout/<obj>_<recep>/{success,
                # consecutive_grasp, is_src_obj_grasped}` — slashes in the
                # task pair are replaced with `_` to keep wandb's panel tree
                # readable.
                for (obj, recep), metrics in (self._rollout_per_task or {}).items():
                    task_slug = f"{obj}_{recep}".replace(" ", "_").replace("/", "_")
                    for metric_name, vals in metrics.items():
                        if vals:
                            log_payload[f"rollout/{task_slug}/{metric_name}"] = (
                                float(sum(vals) / len(vals))
                            )
                wandb.log(log_payload, step=total_steps)

            # 4. Stopping conditions (checked against absolute ceilings)
            exceed_step_limit = total_steps >= abs_max_steps
            exceed_reset_limit = total_resets > abs_max_reset
            should_stop = exceed_step_limit or exceed_reset_limit

            # 5. In-Training Evaluation
            should_eval = (iteration % self.args.eval_interval == self.args.eval_interval - 1
                           or iteration == end_episode - 1
                           or should_stop)

            if should_eval:
                # per-env rotation eval — each env rotates through
                # eval tasks across num_eval_episode episodes.
                # Formula: eval_tasks[(i % n_eval_tasks + ep) % n_eval_tasks]
                eval_log = {"episode": episode, "total_steps": total_steps, "total_resets": total_resets}
                report_header = (
                    f"{'=' * 50}\n"
                    f"Episode {episode} | Steps {total_steps} | Resets {total_resets}\n"
                    f"{'=' * 50}\n"
                )

                # in non-episodic mode, eval's env.reset would re-randomize
                # object poses / robot pose and break training continuity. Snapshot
                # the live scene state and restore it after eval. Skipped for
                # `reset_mode=per_episode` (next iteration's env.reset would clobber
                # the live state anyway) and skipped on the very first iteration
                # (no `_no_reset_obs` yet → no continuity to preserve).
                saved_env_state = None
                if (self.args.reset_mode == "none"
                        and hasattr(self, '_no_reset_obs')):
                    saved_env_state = self.env.get_env_state()

                def _run_eval_pass(obj_set, kind_prefix, kind_label):
                    """All groups × all tasks simultaneously, num_eval_episode episodes."""
                    print(f"Evaluating {kind_label} at {total_steps} "
                          f"({self.args.num_eval_episode} episodes)")
                    results_raw = self.eval_all_groups(iteration, obj_set, prefix=kind_prefix)
                    results = []
                    for g_idx, g_name, task, n_samples, stats in results_raw:
                        scalars = self.recorder.log_eval(
                            episode=episode,
                            total_steps=total_steps,
                            total_resets=total_resets,
                            eval_kind=kind_label,
                            group=g_name,
                            task=task,
                            scene="default",
                            n_envs=n_samples,
                            success=stats.get("success", 0.0),
                            grasp=stats.get("consecutive_grasp", 0.0),
                            obj_grasped=stats.get("is_src_obj_grasped", 0.0),
                        )
                        eval_log.update(scalars)
                        results.append((task, stats))
                    return results

                # In-domain eval
                in_domain_results = _run_eval_pass(self.args.obj_set, "in_domain", "in_domain")
                eval_log.update(self.recorder.build_wandb_eval_panel("in_domain"))

                # Out-of-domain eval
                ood_results = _run_eval_pass("rand_ood", "out_of_domain", "out_of_domain")
                eval_log.update(self.recorder.build_wandb_eval_panel("out_of_domain"))

                wandb.log(eval_log, step=total_steps)
                self._write_eval_report(report_header + "\nIn-Domain Evaluation:\n", in_domain_results)
                self._write_eval_report("Out-of-Domain Evaluation:\n", ood_results)

                # Trends dashboard refresh (per-eval): regenerate the
                # 4-panel `trends.png` in the run's glob dir after each eval
                # point so a browser hitting the wandb run dir sees the live
                # success/grasp curves + PPO health (approx_kl, clip_fraction,
                # value explained_var) without leaving wandb. Errors are
                # swallowed — the dashboard is a convenience, not load-
                # bearing (the McNemar gate reads `eval_success.csv` directly).
                try:
                    self._refresh_trends_image()
                except Exception as e:
                    print(f"[trends] refresh failed (non-fatal): {type(e).__name__}: {e}")

                # restore training scene state if we snapshotted above.
                # set_env_state restores poses but not the env's per-env task
                # tensors, so re-apply the current training task via set_task,
                # then refresh the cached obs that the next rollout will read.
                if saved_env_state is not None:
                    self.env.set_env_state(saved_env_state)
                    objs, receps = self.scheduler.get_next_tasks()
                    self.env.set_task(objs, receps)
                    self._no_reset_obs = self.env.get_obs_image()

            # 6. Checkpoint (includes training progress for resume)
            if (iteration + 1) % self.args.vla_checkpoint_interval == 0 or should_stop:
                ckpt_path = self.glob_dir / f"episode_{episode:04d}"
                self.policy.save(ckpt_path, extra_state={
                    "resume_episode": episode,
                    "total_steps": total_steps,
                    "hard_reset_count": self.hard_reset_count,
                    "soft_reset_count": self.soft_reset_count,
                })
                # per-checkpoint config + scheduler state
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

    def _record_segment_pose(self, episode, segment_id, total_steps):
        """Append the full scene pose state to glob/segment_pose.csv.

        One row per (env, actor) at the end of every segment, where "actor"
        covers **every** object slot and **every** receptacle slot plus the
        gripper — not just the pair the current task happens to select. Pose is
        the full `pq`: position xyz + quaternion wxyz.

        This supersedes the earlier `end_of_segment_xyz.csv`, which recorded only
        the task-selected pair and only xyz. That one existed to anchor
        `workspace_aabb` bounds from observed steady-state positions (and as a
        reference for the non-episodic restore path); this one is a general
        record of where everything ended up, so orientation and the untouched
        distractor objects are both needed.

        Called BEFORE the HSR/LSR resets of this boundary, so the coordinates
        are the steady state the policy actually produced rather than a
        post-respawn position.

        `episode` already arrives 1-based from `train()` (`iteration + 1`) while
        `segment_id` is 0-based; only the latter is incremented here. The old
        `end_of_segment_xyz.csv` incremented both and so labelled every row one
        episode ahead of the matching `train_videos/rollout_epX_segY` directory.

        Hidden slots (a group declaring fewer objects than the batch-wide N)
        write NaN, so every segment contributes a fixed row count and the file
        pivots cleanly.
        """
        unwrapped = self.env.env.unwrapped
        if not hasattr(unwrapped, "get_all_slot_poses"):
            # The predecessor read `get_obj_pos()`, which lives on the base env,
            # so this flag used to work anywhere. Per-slot coverage is built on
            # `_all_carrot_ids` / `_all_plate_ids` and narrows that to the NxM
            # family. Fail loudly at the first segment boundary rather than
            # letting a run that was explicitly asked to record produce nothing.
            raise AttributeError(
                f"--record-segment-pose requires an env exposing "
                f"get_all_slot_poses() (GenericNxMPickPlace and subclasses, "
                f"e.g. PickPlaceNxM-v1); got {type(unwrapped).__name__}."
            )
        poses = unwrapped.get_all_slot_poses()
        try:
            instr = self.env.get_language_instructions()
        except Exception:
            instr = [""] * self.args.num_envs

        csv_path = self.glob_dir / "segment_pose.csv"
        write_hdr = not csv_path.exists()
        ep_1, seg_1 = episode, segment_id + 1

        # Materialize on CPU once per segment rather than per row — these are
        # GPU tensors and a per-row .item() would sync 64x(N+M) times.
        def _np(pair):
            p, q = pair
            return p.detach().cpu().numpy(), q.detach().cpu().numpy()

        rows = []
        for kind, entries, name_lists in (
            ("obj", poses["obj"], poses["obj_names"]),
            ("recep", poses["recep"], poses["recep_names"]),
        ):
            for slot, (pair, names) in enumerate(zip(entries, name_lists)):
                p, q = _np(pair)
                rows.append((kind, slot, names, p, q))
        g_p, g_q = _np(poses["gripper"])
        rows.append(("gripper", 0, [""] * self.args.num_envs, g_p, g_q))

        with open(csv_path, "a") as f:
            if write_hdr:
                f.write("episode,segment,total_steps,env,actor_kind,slot,"
                        "model_name,task,px,py,pz,qw,qx,qy,qz\n")
            for i in range(self.args.num_envs):
                # Quote task/model in case they contain commas (they don't today
                # but defensive — CSV-safe).
                task_str = str(instr[i]).replace('"', '""') if i < len(instr) else ""
                for kind, slot, names, p, q in rows:
                    model = str(names[i]).replace('"', '""') if i < len(names) else ""
                    f.write(f"{ep_1},{seg_1},{total_steps},{i},{kind},{slot},"
                            f"\"{model}\",\"{task_str}\","
                            f"{p[i,0]:.6f},{p[i,1]:.6f},{p[i,2]:.6f},"
                            f"{q[i,0]:.6f},{q[i,1]:.6f},{q[i,2]:.6f},{q[i,3]:.6f}\n")

    def _refresh_trends_image(self):
        """re-render the trends dashboards after each
        eval point.

        Writes TWO images:
          1. `trends.png`           — 4-panel overview (task perf + Policy
                                       drift / LoRA trust region / Value-head
                                       explained_var).
          2. `trends_per_task.png`  — per-(obj, recep) breakdown of the
                                       task-performance panel: one sub-panel
                                       per task with the same rollout +
                                       eval ID/OOD lines.

        Both go to `<glob_dir>/...` (live, browsable from wandb) and to
        `reports/figures/<date>_<phase>-{trends,trends-per_task}.png` (the
        report-fig convention). Failures are swallowed at the caller — the
        dashboard is convenience, not load-bearing (the McNemar gate reads
        `eval_success.csv` directly).
        """
        cronos_root = Path(__file__).resolve().parent           # Autonomous_RL/CRONOS
        sys.path.insert(0, str(cronos_root))
        from tools.plot_run_trends import render as _render_trends
        from tools.plot_run_trends import render_per_task as _render_trends_per_task

        # Total episodes the user launched — for the title's "live @ ep<N>/<total>".
        max_eps = self.args.resume_episode + self.args.max_episodes
        # Out path matches the report convention `<date>_<phase>-<slug>.png`.
        # The phase tag is derived from `args.name`: `P5_*` → P5, `P6_*` → P6.
        phase = "P5" if "P5" in self.args.name.upper() else (
            "P6" if "P6" in self.args.name.upper() else "PPO"
        )
        date = datetime.datetime.utcnow().strftime("%Y-%m-%d")
        report_figs = (cronos_root.parent.parent / "reports" / "figures").resolve()
        out_path = report_figs / f"{date}_{phase}-trends.png"
        out_path_pt = report_figs / f"{date}_{phase}-trends-per_task.png"

        wandb_entity = os.environ.get("WANDB_ENTITY", wandb.run.entity if wandb.run else None)
        wandb_project = os.environ.get("WANDB_PROJECT", wandb.run.project if wandb.run else None)
        run_id = wandb.run.id if wandb.run else None

        # Resume support: if this training was started via --resume-from
        # the resume ckpt's glob dir is two levels up from the ckpt path
        # (`.../<glob>/episode_<N>` → `<glob>` is the prior wandb run's
        # glob). Pull the prior wandb run id from `run_config.yaml` /
        # `.json` written by the prior runner so the trends image chains
        # both runs' wandb histories (per-task rollout series, train
        # scalars) AND merges both `eval_success.csv` files.
        extra_run_ids = []
        extra_eval_csvs = []
        resume_from = getattr(self.args, "resume_from", "")
        if resume_from:
            prior_glob = Path(resume_from).parent       # `.../<glob>/`
            prior_eval_csv = prior_glob / "eval_success.csv"
            if prior_eval_csv.exists():
                extra_eval_csvs.append(str(prior_eval_csv))
            # Prior wandb run id: glob's parent's dir name = `run-<ts>-<id>`.
            prior_run_dir = prior_glob.parent           # `.../wandb/run-<ts>-<id>`
            try:
                prior_run_id = prior_run_dir.name.split("-")[-1]
                if prior_run_id and prior_run_id != run_id:
                    extra_run_ids.append(prior_run_id)
            except Exception:
                pass

        _render_trends(self.glob_dir, max_eps, out_path,
                       entity=wandb_entity, project=wandb_project, run_id=run_id,
                       extra_run_ids=extra_run_ids, extra_eval_csvs=extra_eval_csvs)
        _render_trends_per_task(self.glob_dir, max_eps, out_path_pt,
                                entity=wandb_entity, project=wandb_project, run_id=run_id,
                                extra_run_ids=extra_run_ids, extra_eval_csvs=extra_eval_csvs)

    def save_video_segment(self, iteration, segment_id):
        """Saves current video buffer to glob/train_videos/ (first env per group)."""
        render_dir = self.glob_dir / "train_videos" / f"rollout_ep{iteration + 1}_seg{segment_id + 1}"
        render_dir.mkdir(parents=True, exist_ok=True)

        for i in self._video_envs:
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
        runner.env.reset()
        task_pool = runner.scheduler.task_pool
        print(f"Task Pool: {task_pool}")

        # Shared with eval_only.py — sequence 0 is the training order, the rest
        # are distinct random permutations. Same draw as the previous inline
        # `random.sample(list(permutations(pool))[1:], k)`, minus the factorial
        # materialization for large pools.
        all_sequences = build_eval_sequences(task_pool, args.eval_sequences, args.seed)

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
