"""CRONOS — per-segment head/tail frames across training horizons.

Runs one checkpoint over the **same N segments** under two or more values of
``--episode-len`` (the ``T`` in a run tag: `scripts/train.sh` sets ``T80`` to
``--episode-len 80``, ``T1280`` to ``--episode-len 1280``, with ``--task-len 80``
throughout), and writes the head and tail frame of every segment.

The default — ``--horizons 80,1280 --segments 16`` — is 2 x 16 x 2 = **64
frames**, and isolates exactly one variable. A segment is 80 steps and one task
in both columns; what changes is how often ``env.reset()`` interrupts them:

::

    T80     [seg0][seg1][seg2] ... [seg15]      16 episodes, 16 hard resets
    T1280   [seg0  seg1  seg2  ...  seg15]       1 episode,   1 hard reset

Under ``T80`` every segment starts from a freshly drawn scene. Under ``T1280``
segment *k* starts from wherever the policy left segment *k-1*, which is the
start-state drift `doc/reset_modes.md` describes — here it is visible frame by
frame instead of inferred from a curve.

**The task sequence is identical across runs, and the tool proves it.** Every
run walks the scheduler one task per segment from a fresh start, so segment *k*
carries the same instruction everywhere. The per-segment instruction is
recorded, printed, and diffed across the runs; a mismatch is reported as loudly
as it would ruin the comparison. Segment 0 additionally starts from the same
``env.reset()`` seed in every run, so its head frame is shared — the check
prints that pixel diff too.

**``--repeats N`` re-runs a horizon from the same initial scene and the same
task order, varying only what the policy does.** The two are held apart by
seeding the global torch RNG twice per episode: once before ``env.reset()`` from
``--seed`` (the scene), once after it from ``--action-seed`` plus the repeat
index (the sampling). So a set of repeats answers "how much of the drift is this
policy, and how much is this one rollout" — the spread across runs is the
sampling variance of the same starting condition. It requires
``--vla-temperature > 0``: greedy decoding is a function of the observation
alone, so with the scene held fixed every repeat would be byte-identical, and
the tool refuses rather than burning the GPU on duplicates. It also checks the
other way afterwards, reporting the first segment at which each repeat's tail
diverges from run 0's; "IDENTICAL trajectory" there means the extra runs bought
nothing.

**A checkpoint is required.** Unlike `tools/render_episode_boundaries.py`, whose
point survives with the arm held still, this figure is *about* what the policy
leaves behind, so there is nothing to show without one. ``--ckpt`` is normally
the only policy flag needed: the family, base model, unnorm key and LoRA rank
come from the checkpoint's ``run_config.yaml``, or — for a checkpoint copied out
of its run directory — from PEFT's ``adapter_config.json``. Sampling is greedy
so the horizons stay comparable.

EER is on by default (`main.py`'s ``--reset-robot``), so the robot returns home
at every segment boundary; ``--no-eer`` turns it off. HSR and LSR stay off.

Usage::

    cd V0.93/CRONOS
    CUDA_VISIBLE_DEVICES=4 python tools/render_segment_frames.py \\
        --ckpt /path/to/checkpoint_dir

    # four T1280 rollouts from one initial scene, differing only in sampling:
    CUDA_VISIBLE_DEVICES=4 python tools/render_segment_frames.py \\
        --ckpt /path/to/checkpoint_dir --horizons 1280 --repeats 4 \\
        --vla-temperature 0.6

    # a cheap shape check before committing an hour of GPU:
    CUDA_VISIBLE_DEVICES=4 python tools/render_segment_frames.py \\
        --ckpt /path/to/checkpoint_dir --segments 2 --task-len 10

Budget ~1.05 s per policy step with OpenVLA-7B, plus ~5 min to load the model;
~16 GB VRAM. The 64-frame default is 2560 steps, ~50 min.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

# Shared with the episode-boundary figure: checkpoint resolution, the namespace
# the policy and wrapper constructors read, and the scheduler. Imported rather
# than copied — a second transcription of `resolve_ckpt` is a second thing to
# keep in sync with `main.py`'s checkpoint layout.
from render_episode_boundaries import (  # noqa: E402
    build_policy_args, build_scheduler, label_sheet, resolve_ckpt)

CRONOS_DIR = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = CRONOS_DIR / "configs" / "spatialvla_2x2_train.yaml"
DEFAULT_OUT_DIR = CRONOS_DIR / "figures" / "segment_frames"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0],
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--ckpt", type=Path, required=True,
                   help="checkpoint directory (holding adapter_model.safetensors)")
    p.add_argument("--horizons", default="80,1280",
                   help="comma-separated --episode-len values — the `T` in a run "
                        "tag. Each must be a multiple of --task-len")
    p.add_argument("--segments", type=int, default=16,
                   help="how many segments to run under each horizon")
    p.add_argument("--task-len", type=int, default=80,
                   help="steps per segment (main.py --task-len; 80 in every "
                        "train.sh horizon)")
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG,
                   help="CRONOS YAML config supplying the scene and task pool")
    p.add_argument("--group", type=int, default=0,
                   help="which YAML group to instantiate")
    p.add_argument("--seed", type=int, default=0, help="run seed")
    p.add_argument("--obj-set", default="rand",
                   help="physics randomization set; 'rand' matches training")
    p.add_argument("--episode-id", type=int, default=0,
                   help="pins the background overlay")
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR,
                   help="directory for the per-segment PNGs")
    p.add_argument("--sheet", type=Path, default=None,
                   help="also write a contact sheet here (default: "
                        "<out-dir>/../segment_frames.png)")
    p.add_argument("--no-sheet", action="store_true",
                   help="write only the individual frames")
    p.add_argument("--no-eer", action="store_true",
                   help="disable EER (main.py --no-reset-robot)")
    p.add_argument("--vla-family", choices=("spatialvla", "openvla"), default=None,
                   help="override the checkpoint's policy family")
    p.add_argument("--vla-path", default=None, help="override the base model")
    p.add_argument("--vla-unnorm-key", default=None, help="override the unnorm key")
    p.add_argument("--vla-lora-rank", type=int, default=None, help="override the LoRA rank")
    p.add_argument("--vla-temperature", type=float, default=0.0,
                   help="sampling temperature. 0 = greedy, which is what keeps the "
                        "horizons comparable; --repeats needs it nonzero")
    p.add_argument("--repeats", type=int, default=1,
                   help="run each horizon this many times from the SAME initial "
                        "scene and the SAME task order, letting only the policy's "
                        "sampling differ. Needs --vla-temperature > 0")
    p.add_argument("--action-seed", type=int, default=1000,
                   help="base seed for the policy's sampling; repeat r uses "
                        "--action-seed + 10007*r. Separate from --seed so the "
                        "scene can be held fixed while the actions vary")
    args = p.parse_args()
    if args.repeats < 1:
        raise SystemExit("--repeats must be >= 1")
    if args.repeats > 1 and args.vla_temperature == 0.0:
        raise SystemExit(
            "--repeats > 1 with --vla-temperature 0 would produce identical runs: "
            "greedy decoding is a function of the observation alone, and the "
            "initial scene and task order are deliberately held fixed across "
            "repeats. Pass a sampling temperature (eval uses 0.6).")
    # `build_policy_args` is shared with render_episode_boundaries and reads
    # these two; this tool drives segments directly and has no separate notion
    # of an episode length beyond the horizon under test.
    args.episode_len = args.task_len
    args.policy = "vla"
    return args


def _tag(horizon: int, repeat: int) -> str:
    """How a (horizon, repeat) is named in the checks."""
    return f"T{horizon} run {repeat}"


def run_horizon(args, horizon: int, policy, policy_args, cfg,
                task_log: list, repeat: int = 0
                ) -> list[tuple[np.ndarray, np.ndarray]]:
    """`--segments` segments under one `--episode-len`; (head, tail) per segment.

    The loop is `main.py::run_rollout`'s, unrolled over segments instead of
    steps: an `env.reset()` whenever the segment index crosses an episode
    boundary, then task_len policy steps, then the boundary block — advance the
    scheduler, EER — in that order. HSR and LSR are off, so those are the only
    two things that touch the scene.

    `repeat` indexes a re-run of the same horizon. Every repeat draws the same
    scene and walks the same task order; only the policy's sampling differs.
    That split is enforced by seeding the global torch RNG twice per episode —
    once before `env.reset()` from the scene seed, once after it from the
    action seed — so the scene is a function of `--seed` alone and the actions
    of `--action-seed` and `repeat` alone.
    """
    import torch

    import envs.bridge_multi  # noqa: F401 — registers PickPlaceNxM-v1
    from envs.suite import TaskSuite
    from envs.wrapper import CronosWrapper

    segs_per_episode = horizon // args.task_len
    unnorm_state = policy.vla.get_action_stats(policy_args.vla_unnorm_key)
    action_tokenizer = (policy.processor.action_tokenizer
                        if policy_args.policy == "spatialvla" else None)
    env = CronosWrapper(policy_args, unnorm_state, TaskSuite(),
                        device=torch.device("cuda:0"),
                        action_tokenizer=action_tokenizer)
    env.set_group_specs(cfg.groups)
    env.set_scheduler(build_scheduler(cfg, env, num_envs=1))
    policy.prep_rollout()

    pairs: list[tuple[np.ndarray, np.ndarray]] = []
    obs = instruct = None
    for seg in range(args.segments):
        episode = seg // segs_per_episode
        if seg % segs_per_episode == 0:
            # `CronosWrapper.reset` takes no seed (wrapper.py:265, matching
            # training, where object pose comes off the ambient torch RNG).
            # Seeding here instead makes every horizon's episode *e* draw the
            # same scene, so segment 0 is shared across horizons and the two
            # runs differ only in how often this line executes.
            torch.manual_seed(args.seed + episode)
            torch.cuda.manual_seed_all(args.seed + episode)
            obs, instruct, _ = env.reset()
            # Re-seed for the sampling that follows. Without this the action
            # stream would restart from the scene seed at every episode
            # boundary, which is fine for one greedy run but makes every
            # repeat identical — the opposite of the point.
            action_seed = args.action_seed + 10007 * repeat + episode
            torch.manual_seed(action_seed)
            torch.cuda.manual_seed_all(action_seed)
        head = obs[0].cpu().numpy()
        task_log.append(instruct[0])

        for _ in range(args.task_len):
            with torch.no_grad():
                _, action, _ = policy.get_action(
                    {"image": obs, "task_description": instruct}, deterministic=True)
            obs, _, _, _ = env.step(action)
        # Tail read before the boundary block, so the arm is where the policy
        # left it rather than back at its home pose.
        pairs.append((head, obs[0].cpu().numpy()))

        env.scheduler.update_index()
        env.set_forward()
        env.set_task(*env.scheduler.get_next_tasks())
        if not args.no_eer:
            obs = env.reset_robot()
        instruct = env.get_language_instructions()
        print(f"  [T{horizon} run {repeat}] seg {seg:2d} (episode {episode}) "
              f"done — {task_log[-1]}")

    env.env.close()
    return pairs


def main() -> None:
    args = parse_args()
    horizons = [int(t) for t in args.horizons.split(",") if t.strip()]
    if not horizons:
        raise SystemExit("--horizons is empty")
    for h in horizons:
        if h % args.task_len != 0:
            raise SystemExit(f"--horizons entry {h} is not a multiple of "
                             f"--task-len {args.task_len} (main.py asserts the same)")

    import torch
    if not torch.cuda.is_available():
        raise SystemExit(
            "no CUDA device visible. PickPlaceNxM-v1 requires the GPU PhysX backend, "
            "and the policy needs somewhere to live. Set CUDA_VISIBLE_DEVICES.")

    from envs.config import load_cronos_config

    config = load_cronos_config(args.config)
    if not 0 <= args.group < len(config.groups):
        raise SystemExit(f"--group {args.group} out of range (config has {len(config.groups)})")
    group = config.groups[args.group]
    print(f"[config] {args.config.name} group '{group.name}': "
          f"obj={group.obj} recep={group.recep}")
    n_runs = len(horizons) * args.repeats
    print(f"[plan] horizons {horizons} x {args.repeats} repeat(s) x "
          f"{args.segments} segments x {args.task_len} steps = "
          f"{n_runs * args.segments * 2} frames, "
          f"{n_runs * args.segments * args.task_len} policy steps")

    spec = resolve_ckpt(args)
    policy_args = build_policy_args(args, spec, group)
    if spec["policy"] == "spatialvla":
        from simpler_env.policies.spatialvla.spatialvla_train import (
            SpatialVLAPolicy as _PolicyCls)
    else:
        from simpler_env.policies.openvla.openvla_train import (
            OpenVLAPolicy as _PolicyCls)
    t0 = time.time()
    policy = _PolicyCls(policy_args, device_id=0)
    print(f"[policy] {spec['policy']} from {spec['vla_path']} + LoRA "
          f"{args.ckpt.resolve().name}, loaded in {time.time() - t0:.1f}s, "
          f"{torch.cuda.memory_allocated() / 2 ** 30:.1f} GiB")

    # One group, one env — see render_episode_boundaries.main for why a
    # multi-group config cannot be handed to a single-env wrapper as-is.
    import dataclasses
    config_one = dataclasses.replace(
        config, groups=[dataclasses.replace(group, num_envs=1)])

    # Keyed by (horizon, repeat). With --repeats 1 that is one entry per
    # horizon and the output names stay short.
    results, task_logs = {}, {}
    for horizon in horizons:
        segs_per_ep = horizon // args.task_len
        print(f"[run] T{horizon}: --episode-len {horizon} → {segs_per_ep} "
              f"segment(s) per episode, "
              f"{-(-args.segments // segs_per_ep)} episode(s) over "
              f"{args.segments} segments, x{args.repeats} repeat(s)")
        for r in range(args.repeats):
            t0 = time.time()
            task_logs[horizon, r] = []
            results[horizon, r] = run_horizon(args, horizon, policy, policy_args,
                                              config_one, task_logs[horizon, r],
                                              repeat=r)
            print(f"[run] T{horizon} run {r} took {time.time() - t0:.1f}s")

    # --- checks -------------------------------------------------------------
    # The horizons are only comparable segment-by-segment if they ran the same
    # tasks in the same order. Print the sequence and diff it rather than
    # asserting it silently.
    keys = list(results)
    ref_key = keys[0]
    ref = task_logs[ref_key]
    print(f"[tasks] {len(ref)} segments:")
    for i, t in enumerate(ref):
        print(f"         seg {i:2d}  {t}")
    for k in keys[1:]:
        if task_logs[k] == ref:
            print(f"[check] task sequence {_tag(*k)} vs {_tag(*ref_key)}: identical")
        else:
            print(f"[check] task sequence {_tag(*k)} vs {_tag(*ref_key)}: DIFFERS — "
                  f"these runs are NOT comparable segment by segment")
            for i, (a, b) in enumerate(zip(ref, task_logs[k])):
                if a != b:
                    print(f"         first divergence at seg {i}: {a!r} vs {b!r}")
                    break

    # Segment 0 opens on the same seeded reset in every run, so its head frame
    # is a shared baseline. If it is not, the sim is not reproducible across
    # wrapper instances and nothing downstream should be read as an effect of
    # the horizon or of the sampling.
    for k in keys[1:]:
        d = int(np.abs(results[k][0][0].astype(np.int16)
                       - results[ref_key][0][0].astype(np.int16)).max())
        print(f"[check] seg 00 head, {_tag(*k)} vs {_tag(*ref_key)}: "
              + ("identical" if d == 0 else f"DIFFERS (max |diff| = {d}) — this "
                 "run did not start from the same scene"))

    # Repeats are only worth the GPU time if they actually diverge. Report the
    # first segment whose tail differs from run 0's, per horizon; "no
    # divergence" means the sampling did not change anything and the extra runs
    # are duplicates.
    if args.repeats > 1:
        for horizon in horizons:
            base = results[horizon, 0]
            for r in range(1, args.repeats):
                other = results[horizon, r]
                first = next((i for i in range(args.segments)
                              if not np.array_equal(base[i][1], other[i][1])), None)
                if first is None:
                    print(f"[check] T{horizon} run {r} vs run 0: IDENTICAL "
                          f"trajectory — the sampling changed nothing")
                else:
                    d = int(np.abs(base[-1][1].astype(np.int16)
                                   - other[-1][1].astype(np.int16)).max())
                    print(f"[check] T{horizon} run {r} vs run 0: diverges from "
                          f"seg {first:02d}; final tail max |diff| = {d}")

    # --- outputs ------------------------------------------------------------
    args.out_dir.mkdir(parents=True, exist_ok=True)
    import imageio.v3 as iio

    n = 0
    for (h, r), pairs in results.items():
        # Zero-padded horizon so a directory listing sorts T0080 before T1280
        # rather than the other way round. The run index appears only when
        # there is more than one, so a single-run directory keeps short names.
        run_part = f"run{r:02d}__" if args.repeats > 1 else ""
        for seg, (head, tail) in enumerate(pairs):
            for phase, frame in (("head", head), ("tail", tail)):
                iio.imwrite(
                    args.out_dir / f"T{h:04d}__{run_part}seg{seg:02d}__{phase}.png",
                    frame)
                n += 1
    print(f"[frames] wrote {n} PNGs to {args.out_dir}")

    if not args.no_sheet:
        sheet_path = args.sheet or args.out_dir.parent / f"{args.out_dir.name}.png"
        rows, labels = [], []
        for (h, r), pairs in results.items():
            run_part = f" run {r}" if args.repeats > 1 else ""
            rows.append([p[0] for p in pairs])
            labels.append(f"T{h}{run_part} — head\n--episode-len {h}")
            rows.append([p[1] for p in pairs])
            labels.append(f"T{h}{run_part} — tail\n{h // args.task_len} seg/episode")
        cols = [f"seg {i:02d}" for i in range(args.segments)]
        run_name = next((q.name for q in args.ckpt.resolve().parents
                         if q.name.startswith("CRONOS-")), args.ckpt.resolve().name)
        repeats_part = (f" · {args.repeats} repeats from action seed "
                        f"{args.action_seed}" if args.repeats > 1 else "")
        caption = (f"{spec['policy']} · {run_name} · temperature "
                   f"{args.vla_temperature} · scene seed {args.seed}"
                   f"{repeats_part} · {args.config.name} group '{group.name}' · "
                   f"EER {'off' if args.no_eer else 'on'}")
        sheet = label_sheet(rows, labels, cols, caption=caption)
        sheet_path.parent.mkdir(parents=True, exist_ok=True)
        iio.imwrite(sheet_path, sheet)
        print(f"[sheet] {sheet.shape[1]}x{sheet.shape[0]} px → {sheet_path}")


if __name__ == "__main__":
    main()
