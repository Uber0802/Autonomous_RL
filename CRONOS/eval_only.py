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
    num_eval_episode: int = 4
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

        # Config (load BEFORE env creation so env_n/env_m are correct)
        yaml_config = None
        if args.config_path:
            from envs.config import load_cronos_config
            yaml_config = load_cronos_config(args.config_path)
            # Apply config fields but NOT num_envs (CLI wins for eval flexibility)
            for field_name in ("env_n", "env_m",
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
        wa.debug_rollout = False
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

    def eval_all_groups(self, obj_set, prefix="eval"):
        """Eval all groups using per-env rotation. Same logic as main.py."""
        from tqdm import tqdm
        from mani_skill.utils.visualization.misc import images_to_video

        self.policy.prep_rollout()

        per_group_evals = self.scheduler.get_eval_tasks_per_group()
        n_groups = len(per_group_evals)
        num_eps = self.args.num_eval_episode

        group_eval_info = []
        group_starts = []
        offset = 0
        for g_name, g_size, g_tasks in per_group_evals:
            tasks_info = []
            for task in g_tasks:
                obj, recep = TaskScheduler._extract_obj_recep(task)
                tasks_info.append((task, obj, recep))
            group_eval_info.append((g_name, g_size, tasks_info))
            group_starts.append(offset)
            offset += g_size

        accum = {}
        for g_name, g_size, tasks_info in group_eval_info:
            for task_str, _, _ in tasks_info:
                accum[(g_name, task_str)] = defaultdict(list)

        print(f"  Eval rotation: {num_eps} episodes, {n_groups} groups")
        for g_idx, (g_name, g_size, tasks_info) in enumerate(group_eval_info):
            n_tasks = len(tasks_info)
            print(f"  [{g_name}] {n_tasks} eval tasks, {g_size} envs "
                  f"({g_size // n_tasks} envs/task/ep, "
                  f"{num_eps * g_size // n_tasks} samples/task total)")
            for t_idx, (task_str, _, _) in enumerate(tasks_info):
                print(f"    T{t_idx}: {task_str}")

        for ep in range(num_eps):
            envs_obj = []
            envs_recep = []
            env_task_map = [None] * self.args.num_envs

            for g_idx, (g_name, g_size, tasks_info) in enumerate(group_eval_info):
                n_tasks = len(tasks_info)
                g_start = group_starts[g_idx]
                for local_i in range(g_size):
                    t_idx = (local_i % n_tasks + ep) % n_tasks
                    task_str, obj, recep = tasks_info[t_idx]
                    envs_obj.append(obj)
                    envs_recep.append(recep)
                    env_task_map[g_start + local_i] = (g_idx, g_name, task_str)

            while len(envs_obj) < self.args.num_envs:
                envs_obj.append(envs_obj[-1])
                envs_recep.append(envs_recep[-1])

            obs, _, _ = self.env.reset(obj_set_override=obj_set, skip_scheduler=True)
            self.env.set_task(envs_obj, envs_recep)
            instruct = self.env.get_language_instructions()

            # Record video for all envs (first episode only)
            record_this_ep = self.args.record_video and ep == 0
            total_group_envs_count = sum(gs for _, gs, _ in group_eval_info)
            if record_this_ep:
                video_frames = {i: [] for i in range(total_group_envs_count)}

            env_infos = defaultdict(list)
            for _ in tqdm(range(self.args.segment_len),
                          desc=f"eval {prefix} ep{ep+1}/{num_eps}", leave=False):
                val, action, logp = self._get_action(obs, instruct)
                if record_this_ep:
                    for env_i in range(total_group_envs_count):
                        video_frames[env_i].append(obs[env_i].cpu().numpy().copy())
                obs, reward, truncated, env_info = self.env.step(action)
                if "episode" in env_info:
                    for k, v in env_info["episode"].items():
                        env_infos[k] += v

            if record_this_ep:
                eval_video_dir = self.glob_dir / "eval_videos" / prefix
                eval_video_dir.mkdir(parents=True, exist_ok=True)
                for env_i, frames in video_frames.items():
                    if frames:
                        entry = env_task_map[env_i]
                        g_name_v = entry[1] if entry else f"g{env_i}"
                        task_v = entry[2].replace(" ", "_") if entry else "unknown"
                        images_to_video(frames, str(eval_video_dir),
                                        f"{g_name_v}_{task_v}_env{env_i}", fps=10, verbose=False)

            total_group_envs = sum(gs for _, gs, _ in group_eval_info)
            for env_idx in range(min(len(env_task_map), total_group_envs)):
                entry = env_task_map[env_idx]
                if entry is None:
                    continue
                _, g_name_r, task_str = entry
                for k, vals in env_infos.items():
                    if env_idx < len(vals):
                        accum[(g_name_r, task_str)][k].append(vals[env_idx])

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

    def run(self):
        """Run in-domain and out-of-domain eval, log to wandb + CSV."""
        eval_log = {"episode": 0, "total_steps": 0, "total_resets": 0}

        for kind_label, obj_set in [("in_domain", self.args.obj_set),
                                     ("out_of_domain", "rand_ood")]:
            print(f"\nEvaluating {kind_label} ({self.args.num_eval_episode} episodes)")
            results_raw = self.eval_all_groups(obj_set, prefix=kind_label)
            report_results = []
            for g_idx, g_name, task, n_samples, stats in results_raw:
                scalars = self.recorder.log_eval(
                    episode=0,
                    total_steps=0,
                    total_resets=0,
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
                report_results.append((task, stats))
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
