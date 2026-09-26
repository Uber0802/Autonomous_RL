"""Measure rollout throughput and GPU memory for one (policy, package stack).

Why this exists
---------------
The README documents peak memory per conda env (~40 GB on the lightweight
OpenVLA stack, ~55 GB once transformers/peft are upgraded to serve SpatialVLA)
but nothing about speed, and the memory figures were never broken down by
phase. That makes "which package made this slower/fatter" unanswerable, and it
is the question the four-env split exists to work around. This script produces
the missing numbers in a form that is comparable across envs and against
AutoRL.

It drives the REAL training path — `CronosRunner.run_rollout`, including the
mid-rollout PPO updates — rather than a simplified copy, so the numbers reflect
what training actually costs and cannot drift away from it as the code changes.
Timing is attached by wrapping the phase entry points, not by reimplementing
them.

What it reports
---------------
* throughput: env-steps/sec and wall-clock per segment
* phase breakdown: policy inference / env.step / buffer insert / PPO update
* GPU peak memory: allocated and reserved, per device, split rollout vs update
* a package fingerprint (torch, transformers, peft, tokenizers, CUDA, GPU) so a
  result can be attributed to a stack

Usage
-----
    python tools/bench_rollout.py \
        --config-path configs/one_group_seq_random_2x2.yaml \
        --policy openvla --vla-path openvla/openvla-7b \
        --vla-unnorm-key bridge_orig \
        --segment-len 80 --episode-len 80 --task-len 80 --ppo-update-len 80 \
        --bench-episodes 2 --bench-warmup-episodes 1 \
        --bench-out reports/bench/openvla_cronos_env.json

Comparing stacks: run the identical command under each conda env and diff the
JSONs. Keep `--seed`, the config, and every length flag fixed — throughput
depends on `num_envs`, `segment_len` and `buffer_inferbatch`.

Note on phase timing: GPU work is asynchronous, so the per-phase numbers are
taken with `torch.cuda.synchronize()` at each boundary. That makes the
breakdown correct but inflates the total slightly versus an unsynchronized run.
`--bench-sync false` drops the syncs: the total is then the honest wall clock,
but phase attribution smears into whichever call next forces a sync. Compare
totals across stacks with the same setting.
"""

from __future__ import annotations

import json
import os
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

# Run from the CRONOS package root (tools/ is a subdir of it).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import tyro

from main import Args, CronosRunner


@dataclass
class BenchArgs(Args):
    """Training args plus benchmark controls.

    Inherits every training flag so the benchmark is launched exactly like the
    run it is meant to characterize — no separate, drifting flag surface.
    """
    bench_episodes: int = 2            # measured episodes
    bench_warmup_episodes: int = 1     # discarded: CUDA context, autotune, cache warmup
    bench_out: str = ""                # extra JSON destination
    bench_sync: bool = True            # cuda.synchronize() at phase boundaries


class PhaseTimer:
    """Accumulates wall time and call counts per named phase."""

    def __init__(self, sync: bool):
        self.sync = sync and torch.cuda.is_available()
        self.totals = defaultdict(float)
        self.counts = defaultdict(int)
        self.enabled = True

    def _barrier(self):
        # Without this, an async kernel launch returns immediately and its cost
        # lands on whatever later call happens to synchronize.
        if self.sync:
            torch.cuda.synchronize()

    def wrap(self, obj, attr, label):
        """Wrap `obj.attr` so each call adds to `label`. Returns the original."""
        original = getattr(obj, attr)

        def timed(*a, **kw):
            if not self.enabled:
                return original(*a, **kw)
            self._barrier()
            t0 = time.perf_counter()
            try:
                return original(*a, **kw)
            finally:
                self._barrier()
                self.totals[label] += time.perf_counter() - t0
                self.counts[label] += 1

        setattr(obj, attr, timed)
        return original

    def reset(self):
        self.totals.clear()
        self.counts.clear()


def _package_fingerprint() -> dict:
    """Versions that plausibly move speed or memory, plus the hardware."""
    def _v(mod):
        try:
            return __import__(mod).__version__
        except Exception:
            return None

    fp = {
        "python": sys.version.split()[0],
        "conda_env": os.environ.get("CONDA_DEFAULT_ENV"),
        "torch": _v("torch"),
        "torch_cuda": getattr(torch.version, "cuda", None),
        "transformers": _v("transformers"),
        "peft": _v("peft"),
        "tokenizers": _v("tokenizers"),
        "numpy": _v("numpy"),
        "gymnasium": _v("gymnasium"),
        "cuda_available": torch.cuda.is_available(),
        "gpus": [],
        "env_vars": {
            k: os.environ.get(k)
            for k in ("PYTORCH_CUDA_ALLOC_CONF", "CUDA_VISIBLE_DEVICES",
                      "XLA_PYTHON_CLIENT_PREALLOCATE")
        },
    }
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            fp["gpus"].append({
                "index": i,
                "name": props.name,
                "total_gb": round(props.total_memory / 1024 ** 3, 2),
                "capability": f"{props.major}.{props.minor}",
            })
    return fp


def _peak_memory() -> dict:
    """Per-device peak allocated/reserved since the last reset, in GB."""
    if not torch.cuda.is_available():
        return {}
    out = {}
    for i in range(torch.cuda.device_count()):
        out[f"cuda:{i}"] = {
            "allocated_gb": round(torch.cuda.max_memory_allocated(i) / 1024 ** 3, 3),
            "reserved_gb": round(torch.cuda.max_memory_reserved(i) / 1024 ** 3, 3),
        }
    return out


def _reset_peak_memory() -> None:
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            torch.cuda.reset_peak_memory_stats(i)


def main() -> int:
    args = tyro.cli(BenchArgs)

    # The benchmark drives the rollout itself; keep the runner from also
    # evaluating or checkpointing, which would pollute both timing and memory.
    args.max_episodes = max(1, args.bench_warmup_episodes + args.bench_episodes)
    args.eval_at_start = False
    args.eval_single = False
    args.eval_sequential = False
    args.wandb = False

    steps_per_episode = args.episode_len * args.num_envs
    segments_per_episode = args.episode_len // args.task_len

    print("=" * 72)
    print("CRONOS rollout benchmark")
    print("=" * 72)
    fingerprint = _package_fingerprint()
    for k in ("conda_env", "torch", "torch_cuda", "transformers", "peft", "tokenizers"):
        print(f"  {k:<14s} {fingerprint[k]}")
    for g in fingerprint["gpus"]:
        print(f"  gpu[{g['index']}]        {g['name']} ({g['total_gb']} GB, sm_{g['capability'].replace('.', '')})")
    print(f"  policy         {args.policy} ({args.vla_path})")
    print(f"  num_envs       {args.num_envs}")
    print(f"  segment_len    {args.segment_len}  task_len {args.task_len}  "
          f"episode_len {args.episode_len}  ppo_update_len {args.ppo_update_len}")
    print(f"  warmup/measure {args.bench_warmup_episodes}/{args.bench_episodes} episode(s)")
    print()

    t_build = time.perf_counter()
    runner = CronosRunner(args)
    build_s = time.perf_counter() - t_build
    print(f"[bench] setup (model load + env build): {build_s:.1f}s")

    timer = PhaseTimer(sync=args.bench_sync)
    # Phase boundaries. `_get_action` covers VLA inference including the
    # micro-batching loop; `env.step` covers action decode + simulation +
    # observation fetch; `buffer.insert` covers the memmap write of the frame.
    timer.wrap(runner, "_get_action", "policy_inference")
    timer.wrap(runner.env, "step", "env_step")
    timer.wrap(runner.buffer, "insert", "buffer_insert")
    timer.wrap(runner, "_run_ppo_update", "ppo_update")

    ppo_log = str(runner.glob_dir / "bench_ppo_log.txt")

    # Warmup: first pass pays CUDA context creation, cuDNN/cuBLAS autotuning and
    # allocator growth. Timed separately and discarded.
    if args.bench_warmup_episodes > 0:
        print(f"[bench] warmup ({args.bench_warmup_episodes} episode(s))...")
        t0 = time.perf_counter()
        for ep in range(args.bench_warmup_episodes):
            runner.run_rollout(ppo_log_path=ppo_log, episode=ep + 1,
                               episode_base_steps=0, episode_base_resets=0)
        print(f"[bench] warmup done in {time.perf_counter() - t0:.1f}s (discarded)")
        timer.reset()

    _reset_peak_memory()
    print(f"[bench] measuring ({args.bench_episodes} episode(s))...")
    t0 = time.perf_counter()
    for ep in range(args.bench_episodes):
        runner.run_rollout(
            ppo_log_path=ppo_log,
            episode=args.bench_warmup_episodes + ep + 1,
            episode_base_steps=(args.bench_warmup_episodes + ep) * steps_per_episode,
            episode_base_resets=0,
        )
    wall_s = time.perf_counter() - t0

    total_env_steps = args.bench_episodes * steps_per_episode
    total_segments = args.bench_episodes * segments_per_episode
    # env-steps counts each parallel env separately (num_envs per tick), which is
    # the unit the README's step budgets and `total_steps` axis use.
    result = {
        "fingerprint": fingerprint,
        "config": {
            "policy": args.policy,
            "vla_path": args.vla_path,
            "config_path": args.config_path,
            "seed": args.seed,
            "num_envs": args.num_envs,
            "segment_len": args.segment_len,
            "task_len": args.task_len,
            "episode_len": args.episode_len,
            "ppo_update_len": args.ppo_update_len,
            "buffer_inferbatch": args.buffer_inferbatch,
            "buffer_minibatch": args.buffer_minibatch,
            "alg_gradient_accum": args.alg_gradient_accum,
            "record_video": args.record_video,
            "record_segment_pose": args.record_segment_pose,
            "bench_episodes": args.bench_episodes,
            "bench_warmup_episodes": args.bench_warmup_episodes,
            "bench_sync": args.bench_sync,
        },
        "setup_seconds": round(build_s, 2),
        "wall_seconds": round(wall_s, 3),
        "env_steps": total_env_steps,
        "segments": total_segments,
        "env_steps_per_sec": round(total_env_steps / wall_s, 2) if wall_s else None,
        "seconds_per_segment": round(wall_s / total_segments, 3) if total_segments else None,
        "phases": {
            label: {
                "seconds": round(secs, 3),
                "calls": timer.counts[label],
                "pct_of_wall": round(100.0 * secs / wall_s, 1) if wall_s else None,
                "ms_per_call": round(1000.0 * secs / timer.counts[label], 2)
                               if timer.counts[label] else None,
            }
            for label, secs in sorted(timer.totals.items(), key=lambda kv: -kv[1])
        },
        "peak_memory": _peak_memory(),
    }
    # Whatever the phases do not cover: video encoding, CSV writes, scheduler,
    # gc/empty_cache at segment boundaries. A large residual is itself a finding.
    accounted = sum(timer.totals.values())
    result["phases_unaccounted"] = {
        "seconds": round(wall_s - accounted, 3),
        "pct_of_wall": round(100.0 * (wall_s - accounted) / wall_s, 1) if wall_s else None,
        "note": "video encode + CSV writes + scheduler + gc/empty_cache",
    }

    print()
    print("=" * 72)
    print(f"  wall                {wall_s:.2f}s for {total_env_steps} env-steps "
          f"({total_segments} segments)")
    print(f"  throughput          {result['env_steps_per_sec']} env-steps/sec")
    print(f"  per segment         {result['seconds_per_segment']}s")
    print()
    print(f"  {'phase':<20s} {'seconds':>10s} {'% wall':>8s} {'ms/call':>10s} {'calls':>8s}")
    print("  " + "-" * 60)
    for label, p in result["phases"].items():
        print(f"  {label:<20s} {p['seconds']:>10.2f} {p['pct_of_wall']:>8.1f} "
              f"{(p['ms_per_call'] or 0):>10.2f} {p['calls']:>8d}")
    u = result["phases_unaccounted"]
    print(f"  {'(unaccounted)':<20s} {u['seconds']:>10.2f} {u['pct_of_wall']:>8.1f}")
    print()
    for dev, m in result["peak_memory"].items():
        print(f"  peak {dev:<12s} allocated {m['allocated_gb']:>7.2f} GB   "
              f"reserved {m['reserved_gb']:>7.2f} GB")
    print("=" * 72)

    out_paths = [runner.glob_dir / "bench_rollout.json"]
    if args.bench_out:
        out_paths.append(Path(args.bench_out))
    for p in out_paths:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(result, indent=2, default=str) + "\n")
        print(f"[bench] wrote {p}")

    runner.buffer.cleanup()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
