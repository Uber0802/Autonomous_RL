"""CRONOS V0.3 — Standalone evaluation script.

Loads a checkpoint and runs eval_all_groups() with per-env rotation.
No training rollout, no PPO, no replay buffer allocation.

Usage:
    python eval_only.py \
        --config-path configs/two_group_2x2.yaml \
        --vla-load-path /path/to/checkpoint/episode_0128 \
        --num-envs 16 --num-eval-episode 4 \
        --record-video
"""

import logging
logging.getLogger("mani_skill").setLevel(logging.ERROR)

import itertools
import json
import random
import sys
import numpy as np
import torch
import tyro
import wandb
from pathlib import Path
from dataclasses import dataclass
from collections import defaultdict
from tqdm import tqdm

from mani_skill.utils.visualization.misc import images_to_video

from envs.wrapper import CronosWrapper
from envs.suite import TaskSuite
from envs.scheduler import TaskScheduler
import envs.bridge_multi  # Trigger environment registration
from training.metrics import SuccessRecorder


@dataclass
class EvalArgs:
    # --- Run ---
    name: str = "CRONOS-Eval"
    seed: int = 0

    # --- Environment ---
    env_id: str = "PickPlaceNxM-v1"
    env_n: int = 2
    env_m: int = 2
    num_envs: int = 64
    obj_set: str = "rand"
    obj1_index: int = 7
    obj2_index: int = 2
    obj3_index: int = 10
    plate1_index: int = 1
    plate2_index: int = 2
    plate3_index: int = 3
    scene: str = ""

    # --- Eval control ---
    segment_len: int = 80
    num_eval_episode: int = 4               # legacy / unused in single+sequential modes
    eval_mode: str = "sequential"           # "sequential" (AutoRL default) | "single"
    eval_sequences: int = 5                 # sequential mode: training_seq + (N-1) random perms
    config_path: str = ""
    task_order: str = "sequential"
    task_filter: str = ""
    num_groups: int = 0

    # --- VLA model ---
    vla_path: str = "openvla/openvla-7b"
    vla_load_path: str = ""
    vla_unnorm_key: str = "bridge_orig"
    vla_lora_rank: int = 32
    vla_temperature_eval: float = 0.6

    # --- Logging ---
    wandb: bool = False
    wandb_dir: str = ""
    record_video: bool = True
    log_file: str = "eval.log"
    eval_report: str = "eval_report.txt"

    # --- Inference ---
    buffer_inferbatch: int = 32


class EvalRunner:
    """Lightweight runner for eval-only mode. No PPO, no buffer."""

    def __init__(self, args: EvalArgs):
        self.args = args

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
        wandb.define_metric("total_steps")
        wandb.define_metric("eval_*", step_metric="total_steps")
        self.glob_dir = Path(run.dir).parent / "glob"
        self.glob_dir.mkdir(parents=True, exist_ok=True)
        self.files_dir = Path(run.dir)

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

        self.recorder = SuccessRecorder(self.glob_dir)

        # Device
        device_id = 0
        device_id_other = 1 if torch.cuda.device_count() > 1 else 0
        self.device = torch.device(f"cuda:{device_id}")

        # Policy (no PPO needed)
        from simpler_env.policies.openvla.openvla_train import OpenVLAPolicy
        # Build a minimal args object that OpenVLAPolicy expects
        self.policy = OpenVLAPolicy(self._policy_args(), device_id=device_id_other)

        # Config (load BEFORE env creation so env_n/env_m/num_envs are correct).
        # V0.3: num_envs is authoritative from the YAML (sum of per-group num_envs);
        # the CLI default is only a fallback when no config is provided.
        yaml_config = None
        if args.config_path:
            from envs.config import load_cronos_config
            yaml_config = load_cronos_config(args.config_path)
            for field_name in ("env_n", "env_m", "num_envs",
                               "obj1_index", "obj2_index", "obj3_index",
                               "plate1_index", "plate2_index", "plate3_index",
                               "scene", "task_order"):
                yaml_val = getattr(yaml_config, field_name, None)
                if yaml_val is not None:
                    setattr(args, field_name, yaml_val)
            if yaml_config.groups:
                args.env_n = max(len(g.obj) for g in yaml_config.groups)
                args.env_m = max(len(g.recep) for g in yaml_config.groups)

        # Environment (created after config so env_n/env_m are correct)
        unnorm_state = self.policy.vla.get_action_stats(args.vla_unnorm_key)
        self.suite = TaskSuite()
        self.env = CronosWrapper(self._wrapper_args(), unnorm_state, self.suite, device=self.device)

        task_pool = self.env.get_task_pool()
        from envs.config import resolve_symbolic_task, has_symbolic_refs, build_obj_recep_name_maps
        from envs.scheduler import GroupState

        group_states = []
        if yaml_config and yaml_config.groups:
            env_unwrapped = self.env.env.unwrapped
            model_db_carrot = env_unwrapped.model_db_carrot
            model_db_plate = env_unwrapped.model_db_plate

            for g in yaml_config.groups:
                g_obj_names, g_recep_names = build_obj_recep_name_maps(
                    g.obj, g.recep, model_db_carrot, model_db_plate)
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
                fan_out=yaml_config.fan_out,
            )
            self.env.set_group_specs(yaml_config.groups)
        else:
            self.scheduler = TaskScheduler.from_flat_pool(
                task_pool=task_pool,
                mode=args.task_order,
                num_envs=args.num_envs,
            )
        self.env.set_scheduler(self.scheduler)
        print(f"[SCHEDULER] mode={args.task_order}, pool={self.scheduler.task_pool}")

        # Dump config
        cfg = dict(args.__dict__)
        (self.glob_dir / "run_config.json").write_text(json.dumps(cfg, indent=2, default=str) + "\n")

    def _policy_args(self):
        """Build a namespace that OpenVLAPolicy.__init__ expects."""
        a = self.args
        class _PA:
            pass
        pa = _PA()
        for k, v in a.__dict__.items():
            setattr(pa, k, v)
        # Training-related defaults that policy init reads
        pa.vla_lr = 1e-4
        pa.vla_vhlr = 3e-3
        pa.vla_grad_norm = 10.0
        pa.vla_optim_beta1 = 0.9
        pa.vla_optim_beta2 = 0.999
        pa.vla_temperature = 1.0
        pa.alg_gradient_accum = 1
        return pa

    def _wrapper_args(self):
        """Build a namespace that CronosWrapper.__init__ expects."""
        a = self.args
        class _WA:
            pass
        wa = _WA()
        for k, v in a.__dict__.items():
            setattr(wa, k, v)
        # Wrapper expects these training-related fields
        wa.episode_len = a.segment_len
        wa.task_len = a.segment_len
        wa.reset_mode = "per_episode"
        wa.reset_robot = True
        wa.reset_unsuitable = False
        wa.unsuitable_detector = "low_z"
        wa.enable_backward = False
        wa.backward_interval = 1
        return wa

    @torch.no_grad()
    def _get_action(self, obs, instruct):
        total_batch = obs.shape[0]
        values, actions, logprobs = [], [], []
        for i in range(0, total_batch, self.args.buffer_inferbatch):
            obs_batch = obs[i:i + self.args.buffer_inferbatch]
            instruct_batch = instruct[i:i + self.args.buffer_inferbatch]
            val, act, logp = self.policy.get_action(
                {"image": obs_batch, "task_description": instruct_batch},
                deterministic=True
            )
            values.append(val)
            actions.append(act)
            logprobs.append(logp)
        return torch.cat(values, 0), torch.cat(actions, 0), torch.cat(logprobs, 0)

    def _write_eval_report(self, header, results):
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
    def eval(self, iteration, task_idx, obj_set, object, receptacle, prefix="eval", reset=True, group_idx=0):
        """V0.3.1 — Standalone evaluation (port of AutoRL `render`).

        Mirrors `CronosRunner.eval` in main.py. All envs run the SAME (object[i],
        receptacle[i]) — the caller broadcasts a single task. No fan-out. This is
        the eval mode for `eval_only.py`; `eval_all_groups` (fan-out rotation) is
        reserved for training-time eval inside `main.py:train()` and is intentionally
        absent here.
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
            val, action, logp = self._get_action(obs, instruction)
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

    def _build_sequences(self, task_pool):
        """Return list of task orderings to evaluate.

        Sequential mode: [training_seq] + (N-1) random distinct permutations.
        Single mode: one one-task "sequence" per task in the pool.
        """
        if self.args.eval_mode == "single":
            return [[t] for t in task_pool]

        all_perms = list(itertools.permutations(task_pool))
        training_seq = tuple(task_pool)
        other_perms = [p for p in all_perms if p != training_seq]
        n_random = max(0, self.args.eval_sequences - 1)
        random.seed(self.args.seed)
        sampled = random.sample(other_perms, min(n_random, len(other_perms)))
        return [list(training_seq)] + [list(p) for p in sampled]

    def run(self):
        """Run AutoRL-style eval (sequential default) for in-domain + out-of-domain."""
        task_pool = list(self.scheduler.task_pool)
        sequences = self._build_sequences(task_pool)
        group_name = (self.scheduler.group_states[0].name
                      if getattr(self.scheduler, "group_states", None)
                      else "default")

        print(f"\nEval mode: {self.args.eval_mode}, "
              f"{len(sequences)} sequence(s), num_envs={self.args.num_envs}")
        for i, seq in enumerate(sequences):
            tag = "training" if (i == 0 and self.args.eval_mode == "sequential") else "task" if self.args.eval_mode == "single" else "random"
            print(f"  seq{i} ({tag}): {seq}")

        eval_log = {"episode": 0, "total_steps": 0, "total_resets": 0}

        for kind_label, obj_set in [("in_domain", self.args.obj_set),
                                     ("out_of_domain", "rand_ood")]:
            print(f"\nEvaluating {kind_label}")
            report_results = []
            for seq_idx, sequence in enumerate(sequences):
                for task_idx, task_str in enumerate(sequence):
                    obj, recep = TaskScheduler._extract_obj_recep(task_str)
                    reset = (task_idx == 0)  # only at sequence start
                    stats = self.eval(
                        iteration=seq_idx,
                        task_idx=task_idx,
                        obj_set=obj_set,
                        object=[obj] * self.args.num_envs,
                        receptacle=[recep] * self.args.num_envs,
                        prefix=f"{kind_label}_seq{seq_idx}_task{task_idx}",
                        reset=reset,
                    )
                    print(f"    seq{seq_idx} task{task_idx}: {task_str} "
                          f"success={stats.get('success', 0.0):.4f} "
                          f"grasp={stats.get('consecutive_grasp', 0.0):.4f}")
                    scalars = self.recorder.log_eval(
                        episode=0,
                        total_steps=seq_idx,
                        total_resets=0,
                        eval_kind=kind_label,
                        group=group_name,
                        task=task_str,
                        scene="default",
                        n_envs=self.args.num_envs,
                        success=stats.get("success", 0.0),
                        grasp=stats.get("consecutive_grasp", 0.0),
                        obj_grasped=stats.get("is_src_obj_grasped", 0.0),
                    )
                    eval_log.update(scalars)
                    report_results.append((f"seq{seq_idx}_task{task_idx}: {task_str}", stats))
            eval_log.update(self.recorder.build_wandb_eval_panel(kind_label))
            self._write_eval_report(f"{kind_label.replace('_', ' ').title()} Evaluation:\n",
                                    report_results)

        wandb.log(eval_log, step=0)
        print("\nEval complete. Results saved to:")
        print(f"  CSV:    {self.glob_dir / 'eval_success.csv'}")
        print(f"  Report: {self.glob_dir / self.args.eval_report}")
        if self.args.record_video:
            print(f"  Videos: {self.glob_dir / 'eval_videos'}")


def main():
    args = tyro.cli(EvalArgs)
    runner = EvalRunner(args)
    runner.run()


if __name__ == "__main__":
    main()
