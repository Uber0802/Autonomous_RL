"""CRONOS — Standalone evaluation script.

Loads a checkpoint and runs single-task or sequential eval over every scene
(YAML group) of the config. No training rollout, no PPO, no replay buffer.

The environment comes from the checkpoint's training config file (the snapshot
next to the checkpoint, or the recorded `config_path`) unless `--config-path`
names one; either way it must define the same scenes as training
(`evaluation/provenance.py`). Settings come from, in order of precedence: CLI
flags > that config's `eval:` block > defaults
(`evaluation/plan.py::EVAL_SETTING_SPEC`). The resolved values,
and where each came from, are written to `glob/eval_plan.json` before any
rollout, together with every sequence and the RNG record.

Usage:
    python eval_only.py \
        --vla-load-path /path/to/glob/episode_0128 \
        --eval-scene-schedule parallel          # all 24 orders by default
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

from run_paths import prepare_wandb_dir, verify_run_dir
from oom_report import is_oom, write_oom_report
from envs.wrapper import CronosWrapper
from envs.suite import TaskSuite
from envs.scheduler import TaskScheduler
import envs.bridge_multi  # Trigger environment registration
from evaluation.plan import build_plan, build_scenes, checkpoint_progress, plan_fingerprint, resolve_eval_settings
from evaluation.provenance import (check_against_training, resolve_config_path, resolve_policy_args,
                                   scene_definition)
from evaluation.sequential import SequentialEvaluator


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
    # Every field in this block marked [key] can also be set in the config's
    # `eval:` block. An explicit CLI flag wins over the YAML; the YAML wins over
    # the default shown here. See evaluation/plan.py::EVAL_SETTING_SPEC.
    segment_len: int = 80
    num_eval_episode: int = 4               # legacy / unused in single+sequential modes
    eval_mode: str = "sequential"           # [mode] sequential (AutoRL render_seq) | single (AutoRL render, blocks run tasks side by side)
    eval_pose_sets: int = 1                 # [pose_sets] sets of start poses; each set = every cycle once (4 tasks: 6 rounds)
    eval_rounds: str = "all"                # [rounds] all | 3 | 3-5 | 3- | 0,6-11 — global round numbers
    eval_layout_slots: int = -1             # [layout_slots] start poses per order block; -1 = one per env, 1 = a single pose
    eval_sequence_seed: int = -1            # [sequence_seed] cycle-order stream; -1 = --seed
    eval_layout_seed: int = -1              # [layout_seed] start-pose stream; -1 = --seed
    eval_policy_seed: int = -1              # [policy_seed] per-unit action-sampling reseed; -1 = --seed
    eval_resume: str = ""                   # glob dir of an interrupted eval: skip its done units, rerun the rest there
    eval_scene_schedule: str = "parallel"   # [scene_schedule] parallel (all scenes in one batch) | serial
    eval_domains: str = "in_domain,out_of_domain"  # [domains] comma-separated subset
    video_envs_per_block: int = -1          # [video_envs_per_block] first k envs of each order block; -1 = all
    record_eval_pose: bool = True           # [record_pose] eval_segment_pose.csv (training segment_pose columns)
    eval_pose_phase: str = "both"           # [pose_phase] start | end | both
    config_path: str = ""                   # default: the checkpoint's training config
    allow_config_mismatch: bool = False     # evaluate on a config whose scenes differ from training
    task_order: str = "sequential"
    task_filter: str = ""
    num_groups: int = 0

    # --- VLA model ---
    # policy / vla_path / vla_unnorm_key / vla_temperature_eval / vla_lora_rank are
    # resolved per field: explicit CLI flag > the checkpoint's training run_config >
    # the config YAML > per-policy defaults (openvla: bridge_orig, temperature 0.6;
    # spatialvla: IPEC-COMMUNITY/spatialvla-4b-224-sft-bridge, bridge_orig/1.0.0,
    # temperature 0.0 — as scripts/train.sh). The defaults written here only
    # matter when a flag is passed. See evaluation/provenance.py.
    policy: str = "openvla"                 # {"openvla", "spatialvla"} — picks the policy class
    vla_path: str = "openvla/openvla-7b"
    vla_load_path: str = ""
    vla_unnorm_key: str = "bridge_orig"
    vla_lora_rank: int = 32
    vla_temperature_eval: float = 0.6
    action_chunk: int = 1                   # SpatialVLA chunk(K) open-loop deployment; K=1 = single-step
    eval_ood: bool = True                   # legacy alias: --no-eval-ood == --eval-domains in_domain

    # --- Logging ---
    wandb: bool = False
    wandb_dir: str = ""
    record_video: bool = True               # [record_video]
    log_file: str = "eval.log"              # tee of stdout (in the wandb files/ dir, NOT glob/)
    eval_report: str = "eval_report.txt"

    # --- Inference ---
    buffer_inferbatch: int = 32  # phaseO-1 attempted 32→64; reverted (parity FAIL + no speedup at fixture)


class EvalRunner:
    """Lightweight runner for eval-only mode. No PPO, no buffer."""

    def __init__(self, args: EvalArgs):
        self.args = args
        self.oom_context = {"phase": "init"}      # for oom_report.py

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
        # Create + validate BEFORE wandb.init — see run_paths.py. Without this a
        # relative/absent --wandb-dir sends the whole eval to the system temp dir.
        wandb_root = prepare_wandb_dir(args.wandb_dir)
        if wandb_root:
            wandb_kwargs["dir"] = wandb_root
        run = wandb.init(**wandb_kwargs)
        verify_run_dir(run.dir, wandb_root)
        wandb.define_metric("total_steps")
        wandb.define_metric("eval_*", step_metric="total_steps")
        # `glob/` is a sibling of `files/` and is NOT synced to wandb.
        self.glob_dir = Path(run.dir).parent / "glob"
        if args.eval_resume:
            # Results continue in the interrupted run's glob; this wandb run only
            # holds the resumed session's log.
            self.glob_dir = Path(args.eval_resume).resolve()
            if not (self.glob_dir / "eval_plan.json").exists():
                raise ValueError(f"--eval-resume {args.eval_resume}: no eval_plan.json there")
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

        # Device
        device_id = 0
        device_id_other = 1 if torch.cuda.device_count() > 1 else 0
        self.device = torch.device(f"cuda:{device_id}")

        # Config (load BEFORE policy and env creation: a wrong or mismatched
        # config should fail before a 7B model is loaded, and env_n/env_m/num_envs
        # must be correct before the env is built).
        # num_envs is authoritative from the YAML (sum of per-group num_envs);
        # the CLI default is only a fallback when no config is provided.
        from envs.config import load_cronos_config
        cronos_root = Path(__file__).resolve().parent
        args.config_path, self.provenance = resolve_config_path(args.config_path, args.vla_load_path, cronos_root)
        print(f"[eval] config: {args.config_path} ({self.provenance['config_source']})")
        yaml_config = load_cronos_config(args.config_path)
        check_against_training(yaml_config, self.provenance, load_cronos_config, args.allow_config_mismatch)
        print(f"[eval] training config check: {self.provenance['check']}")
        # Policy settings must be known before the policy is built.
        self.provenance["policy_args"] = resolve_policy_args(args, sys.argv[1:], args.vla_load_path, yaml_config)
        pa = self.provenance["policy_args"]
        print("[eval] policy: " + ", ".join(f"{k}={v!r} ({pa['sources'][k]})" for k, v in pa["values"].items()))
        for w in pa["warnings"]:
            print(f"[eval] WARNING: {w}")
        import shutil as _shutil
        if not (args.eval_resume and (self.glob_dir / "experiment_config.yaml").exists()):
            _shutil.copy2(args.config_path, self.glob_dir / "experiment_config.yaml")
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

        self.yaml_config = yaml_config

        # Policy (no PPO needed). `--policy` picks the class; `_policy_args`
        # passes through the same minimal namespace to either constructor —
        # both policies expose the same rollout surface (get_action, get_action_stats).
        if args.policy == "openvla":
            from simpler_env.policies.openvla.openvla_train import OpenVLAPolicy as _PolicyCls
        elif args.policy == "spatialvla":
            from simpler_env.policies.spatialvla.spatialvla_train import SpatialVLAPolicy as _PolicyCls
        else:
            raise ValueError(f"Unknown --policy {args.policy!r}; expected one of 'openvla', 'spatialvla'.")
        self.policy = _PolicyCls(self._policy_args(), device_id=device_id_other)


        # Environment (created after config so env_n/env_m are correct)
        unnorm_state = self.policy.vla.get_action_stats(args.vla_unnorm_key)
        self.suite = TaskSuite()
        # forward YAML `unsuitable_detector` block for parametric AABB detector.
        det_cfg = getattr(yaml_config, "unsuitable_detector", None) if yaml_config else None
        # SpatialVLA path: pass `processor.action_tokenizer` so the wrapper's
        # `_process_action` decodes the 3-id [B, 3] output into a 7-DoF action
        # before the shared q01/q99 unnorm. OpenVLA leaves it `None` and the
        # wrapper uses the legacy `bin_centers` path.
        action_tokenizer = (self.policy.processor.action_tokenizer
                            if args.policy == "spatialvla" else None)
        self.env = CronosWrapper(self._wrapper_args(), unnorm_state, self.suite,
                                 device=self.device, unsuitable_detector_cfg=det_cfg,
                                 action_tokenizer=action_tokenizer)

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
        name = "run_config.json"
        if args.eval_resume and (self.glob_dir / name).exists():
            import datetime as _dt
            name = f"run_config_resume_{_dt.datetime.utcnow().strftime('%Y%m%dT%H%M%S')}.json"
        (self.glob_dir / name).write_text(json.dumps(cfg, indent=2, default=str) + "\n")

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

    @torch.no_grad()
    def _get_action_chunk(self, obs, instruct, chunk):
        """SpatialVLA chunk(K) open-loop inference (E-4 co-gate, K>1 path only).

        Returns `action_ids[num_envs, K, ACTION_LEN]` where `ACTION_LEN=3` for
        SpatialVLA — the caller then steps the env K times, one [num_envs, 3]
        slice per step, before re-inferring. OpenVLA does not implement
        `get_action_chunk`; `--action-chunk K>1` is SpatialVLA-only.
        """
        total_batch = obs.shape[0]
        outs = []
        for i in range(0, total_batch, self.args.buffer_inferbatch):
            obs_batch = obs[i:i + self.args.buffer_inferbatch]
            instruct_batch = instruct[i:i + self.args.buffer_inferbatch]
            ids = self.policy.get_action_chunk(
                {"image": obs_batch, "task_description": instruct_batch},
                chunk=chunk,
            )  # [B_i, 3*K]
            outs.append(ids)
        flat = torch.cat(outs, 0)  # [num_envs, 3*K]
        action_len = flat.shape[1] // chunk
        return flat.view(flat.shape[0], chunk, action_len)

    def _build_scenes(self):
        """One EvalScene per YAML group (or a single "default" scene)."""
        return build_scenes(
            self.scheduler.group_states, self.args.num_envs,
            env_n=self.args.env_n, env_m=self.args.env_m,
            group_specs=self.yaml_config.groups if (self.yaml_config and self.yaml_config.groups) else None,
            fan_out=self.scheduler.fan_out,
        )

    def run(self):
        """Standalone eval: resolve settings → build plan → execute → write records."""
        a = self.args
        settings = resolve_eval_settings(a, self.yaml_config.eval if self.yaml_config else None, sys.argv[1:])
        plan = build_plan(self._build_scenes(), settings, num_envs=a.num_envs,
                          segment_len=a.segment_len, seed=a.seed)
        plan.provenance = self.provenance
        plan.fingerprint = plan_fingerprint(plan, dict(
            checkpoint=str(Path(a.vla_load_path).resolve()) if a.vla_load_path else "",
            policy=a.policy, vla_path=a.vla_path, vla_unnorm_key=a.vla_unnorm_key, vla_lora_rank=a.vla_lora_rank,
            vla_temperature_eval=a.vla_temperature_eval, action_chunk=a.action_chunk,
            buffer_inferbatch=a.buffer_inferbatch, env_id=a.env_id, obj_set=a.obj_set,
            scene_definition=scene_definition(self.yaml_config),
        ))
        episode, total_steps, progress_src = checkpoint_progress(a.vla_load_path)
        print(f"[eval] checkpoint progress: episode={episode} total_steps={total_steps} ({progress_src})")

        evaluator = SequentialEvaluator(
            plan=plan, env=self.env, glob_dir=self.glob_dir,
            act_fn=lambda obs, instr: self._get_action(obs, instr)[1],
            chunk_fn=self._get_action_chunk if a.action_chunk > 1 else None,
            action_chunk=a.action_chunk,
            prep_rollout=self.policy.prep_rollout, obj_set=a.obj_set,
            episode=episode, total_steps=total_steps, report_name=a.eval_report,
            resume=bool(a.eval_resume),
            oom_context=self.oom_context,
        )
        self.oom_context["phase"] = "standalone_eval"
        result = evaluator.run()
        if result["wandb"]:
            wandb.log(result["wandb"], step=0)

        print("\nEval complete. Results saved to:")
        for name in ("eval_plan.json", "eval_status.json", "eval_per_trial.csv", "eval_sequence_summary.csv",
                     "eval_coverage.csv", "eval_layouts.csv", "eval_segment_pose.csv",
                     "eval_success.csv", a.eval_report):
            path = self.glob_dir / name
            if path.exists():
                print(f"  {path}")
        if settings.record_video:
            print(f"  {self.glob_dir / 'eval_videos'}")

def main():
    args = tyro.cli(EvalArgs)
    runner = None
    try:
        runner = EvalRunner(args)
        runner.run()
    except BaseException as e:
        if is_oom(e):
            out_dir = getattr(runner, "glob_dir", None) or (Path(args.wandb_dir) if args.wandb_dir else None)
            write_oom_report(e, out_dir, getattr(runner, "oom_context", {"phase": "init"}), args=args)
        raise


if __name__ == "__main__":
    main()
