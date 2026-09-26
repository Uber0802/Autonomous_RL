"""CRONOS — episode-boundary figure: episodic vs non-episodic reset.

Renders the **head and tail frame of the first two episodes** of one single
environment (``num_envs=1``), once with the episodic reset and once without, and
tiles the eight frames into a 2x4 sheet:

::

                 ep1 head     ep1 tail     ep2 head     ep2 tail
    Episodic     env.reset()  policy end   env.reset()  policy end
    Non-episodic env.reset()  policy end   = ep1 tail   policy end

That is the whole content of ``--reset-mode`` (see
[`doc/reset_modes.md`](../doc/reset_modes.md)): under ``per_episode`` every
episode starts from a fresh ``env.reset()`` — which, with the training default
``obj_set=rand``, re-draws object position and rotation — while under ``none``
episode 2 inherits the live scene, so its **head is bit-identical to episode 1's
tail**. Column 3 of the two rows is the figure's entire point; the two rows'
columns 1-2 are identical by construction and are there as the shared baseline.

The scene is the one-group ``configs/spatialvla_2x2_train.yaml`` — one group
named ``default``, ketchup bottle + kitchen shovel over yellow plate + cloth.
``four_group_sequential_2x2.yaml``'s group 0 declares the same two objects and
the same two receptacles, so the two configs render the same scene; the one-group
file is the default because it is also the config a ``--policy vla`` checkpoint
is meant to be shown on.

**Three things can drive the arm.**

``--policy zero`` (the default) commands it to hold. A figure about *reset
semantics* does not depend on which policy produced the tail state, and with the
arm still, the only thing that moves the scene is the reset machinery itself:
the episodic row's episode 2 is visibly a fresh draw, the non-episodic row's four
frames are one continuous scene, and nothing else is in the way.

``--policy vla --ckpt <glob/episode_NNNN>`` loads a real checkpoint and lets the
trained policy produce the tail states. This is the version to use when the
question is what the *policy* leaves behind for the next episode to start from —
the drift `doc/reset_modes.md` warns about. The checkpoint's own
``run_config.yaml`` supplies the policy family, base model, unnorm key and LoRA
rank, so ``--ckpt`` is normally the only flag needed; ``--vla-*`` override any of
them. Sampling is greedy by default (``--vla-temperature 0``) because the two
rows must share episode 1.

``--policy random`` is the illustration-only option; see below.

Only the ``vla`` path goes through `envs/wrapper.py::CronosWrapper`, because a
VLA emits action *tokens* and the wrapper's ``_process_action`` is what decodes
them — a second copy of that decode here would be the most likely thing in this
file to drift out of sync. It therefore reproduces `main.py::run_rollout`
closely: same wrapper, same scheduler-driven task switch at every ``--task-len``
boundary, same EER. The ``zero`` / ``random`` paths drive ``PickPlaceNxM-v1``
through raw ``gym.make`` the way `tools/render_background_catalog.py` does, and
reuse `envs/reset.py::ResetStrategy.reset_robot` for EER. Neither path imports
`main.py`.

``--policy random`` drives the arm with seeded white noise instead, so the tail
frames show a scene the arm has actually disturbed. Treat it as illustration,
not as a stand-in for a trained policy: it is a lottery, and a bad draw sweeps
the objects off the table or parks the gripper in front of the camera, which
looks like a finding and is not one. Note also that the controller is
``pd_ee_target_delta_pose`` — it integrates the *target* pose, so temporally
correlated noise (``--action-momentum`` near 1) random-walks the target out of
the workspace within one segment. White noise (the default 0.0) stays near home.

Both modes consume the *same* action sequence from the same seed, so episode 1
is identical in both rows and the only difference the sheet can show is the
reset itself. The script verifies this and prints the pixel diff of the shared
columns.

For the real thing — a tail state produced by the policy under test — the
frames already exist in a run's ``glob/train_videos/rollout_ep<N>_seg<M>/``.
This script is for the case where no such run exists for the mode you want to
show, or where the two modes must share episode 1 exactly.

**GPU.** A GPU is required. Without a policy it is only a rasterizer plus PhysX
device — one env, ~2.7 GB VRAM; ``--policy vla`` adds the model, ~9 GB peak for
SpatialVLA-4B at bf16. It cannot be made GPU-free — ``sim_backend=cpu``
dies in `envs/bridge_multi.py::_initialize_episode`, which calls
``scene._gpu_apply_all()`` unconditionally, and SAPIEN's software-Vulkan fallback
(lavapipe) is missing ``VK_KHR_external_semaphore_fd``, which SAPIEN requires.
Point ``CUDA_VISIBLE_DEVICES`` at an idle card and it will not disturb training.

Usage::

    cd V0.93/CRONOS
    CUDA_VISIBLE_DEVICES=5 python tools/render_episode_boundaries.py
    # quick shape check (2 x 80 steps instead of 2 x 320):
    CUDA_VISIBLE_DEVICES=5 python tools/render_episode_boundaries.py --episode-len 80
    # arm actually disturbs the scene:
    CUDA_VISIBLE_DEVICES=5 python tools/render_episode_boundaries.py --policy random \
        --out figures/episode_boundaries_random.png
    # tail taken after EER instead of at the last policy step:
    CUDA_VISIBLE_DEVICES=5 python tools/render_episode_boundaries.py --tail-phase post_eer
    # a trained checkpoint drives the arm:
    CUDA_VISIBLE_DEVICES=5 python tools/render_episode_boundaries.py --policy vla \
        --ckpt <run>/wandb/<run-id>/glob/episode_0016 \
        --out figures/episode_boundaries_vla.png

At the default 2 x 320 steps on an idle card: ~8 min and ~2.7 GB for the
policy-free modes, ~14 min and ~9 GB for ``--policy vla``.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

CRONOS_DIR = Path(__file__).resolve().parents[1]
# One group, named `default`, obj [7, 2] / recep [1, 2] — the single-scene
# config. `four_group_sequential_2x2.yaml`'s group 0 carries the same two
# objects and the same two receptacles, so switching between them does not
# change a pixel of this figure; the one-group file is the default because it
# is also the config the `--policy vla` checkpoints are meant to be shown on.
DEFAULT_CONFIG = CRONOS_DIR / "configs" / "spatialvla_2x2_train.yaml"
DEFAULT_OUT = CRONOS_DIR / "figures" / "episode_boundaries.png"

# Row order of the sheet. `label` is what gets drawn; `reset_mode` is the
# `main.py --reset-mode` value the row reproduces.
MODES = (
    ("Episodic", "per_episode"),
    ("Non-Episodic", "none"),
)
COL_LABELS = ("Episode 1 — head", "Episode 1 — tail",
              "Episode 2 — head", "Episode 2 — tail")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0],
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG,
                   help="CRONOS YAML config; supplies N, M and the obj/recep indices")
    p.add_argument("--group", type=int, default=0,
                   help="which YAML group's objects/receptacles to instantiate")
    p.add_argument("--out", type=Path, default=DEFAULT_OUT, help="output PNG")
    p.add_argument("--episode-len", type=int, default=320,
                   help="control steps per episode (main.py --episode-len)")
    p.add_argument("--task-len", type=int, default=80,
                   help="steps between segment boundaries (main.py --task-len)")
    p.add_argument("--seed", type=int, default=0, help="run seed")
    p.add_argument("--obj-set", default="rand",
                   help="physics randomization set; 'rand' matches training, "
                        "'fixed' pins pose to --episode-id and makes the "
                        "episodic row's ep2 head identical to its ep1 head")
    p.add_argument("--episode-id", type=int, default=0,
                   help="pins the background overlay (and, with --obj-set fixed, the pose)")
    p.add_argument("--policy", choices=("zero", "random", "vla"), default="zero",
                   help="what drives the arm; see module docstring. 'vla' needs --ckpt")
    p.add_argument("--ckpt", type=Path, default=None,
                   help="--policy vla: a checkpoint directory written by "
                        "`main.py` (`glob/episode_NNNN/`, holding "
                        "adapter_model.safetensors + run_config.yaml). Everything "
                        "else about the policy is read out of its run_config.yaml")
    p.add_argument("--vla-family", choices=("spatialvla", "openvla"), default=None,
                   help="override the checkpoint's `policy` field")
    p.add_argument("--vla-path", default=None,
                   help="override the checkpoint's `vla_path` (the base model)")
    p.add_argument("--vla-unnorm-key", default=None,
                   help="override the checkpoint's `vla_unnorm_key`")
    p.add_argument("--vla-lora-rank", type=int, default=None,
                   help="override the checkpoint's `vla_lora_rank`")
    p.add_argument("--vla-temperature", type=float, default=0.0,
                   help="sampling temperature. 0 = greedy, which is what makes the "
                        "two rows' episode 1 identical; eval uses 0.6")
    p.add_argument("--action-scale", type=float, default=0.5,
                   help="--policy random: amplitude of the action, in [0, 1]")
    p.add_argument("--action-momentum", type=float, default=0.0,
                   help="--policy random: low-pass coefficient. 0 = white noise, which "
                        "keeps the accumulated EE target near home; raising it toward 1 "
                        "random-walks the target out of the workspace")
    p.add_argument("--tail-phase", choices=("pre_eer", "post_eer"), default="pre_eer",
                   help="'pre_eer': tail = the last policy step, arm where the policy "
                        "left it. 'post_eer': tail = after the episode's final segment "
                        "boundary (EER robot reset + settle) — i.e. literally the state "
                        "`main.py` stashes in `_no_reset_obs` and that the non-episodic "
                        "row's ep2 head then inherits")
    p.add_argument("--no-eer", action="store_true",
                   help="disable EER (main.py --no-reset-robot): the robot is not "
                        "returned home at segment boundaries")
    p.add_argument("--no-tiles", action="store_true",
                   help="skip writing the eight individual frames")
    p.add_argument("--no-labels", action="store_true",
                   help="butt the frames together with no captions, like the other "
                        "catalog figures")
    return p.parse_args()


def make_actions(args, n_steps: int, action_dim: int) -> np.ndarray:
    """The action sequence both modes replay, from `--seed` alone.

    White noise by default. The tempting alternative — low-pass the noise so the
    arm sweeps rather than jitters — backfires here: the controller integrates
    *target* EE deltas, so correlated actions random-walk the target far out of
    the workspace inside a single segment, and the tail frame ends up being the
    gripper pressed against the camera. An AR(1) filter is still available via
    `--action-momentum` for when a slower sweep is wanted.
    """
    if args.policy == "zero":
        return np.zeros((n_steps, action_dim), dtype=np.float32)
    rng = np.random.default_rng(args.seed)
    out = np.zeros((n_steps, action_dim), dtype=np.float32)
    a = np.zeros(action_dim, dtype=np.float32)
    m = float(np.clip(args.action_momentum, 0.0, 0.99))
    # The AR(1) filter shrinks the input's variance by (1-m)/(1+m); undo exactly
    # that, so `--action-scale` means the same amplitude at any momentum instead
    # of collapsing to zero as the walk gets smoother.
    gain = np.sqrt((1.0 + m) / (1.0 - m)) if m < 1.0 else 1.0
    for t in range(n_steps):
        a = m * a + (1.0 - m) * rng.uniform(-1.0, 1.0, size=action_dim).astype(np.float32)
        out[t] = np.clip(a * gain * args.action_scale, -1.0, 1.0)
    return out


def rgb_of(env) -> np.ndarray:
    """The 3rd-person 640x480 frame, as uint8 HWC — the sensor training records."""
    import torch
    obs = env.unwrapped.get_obs()
    return obs["sensor_data"]["3rd_view_camera"]["rgb"][0].to(torch.uint8).cpu().numpy()


def run_mode(args, reset_mode: str, env_kwargs: dict, options: dict,
             actions: np.ndarray) -> list[np.ndarray]:
    """Two episodes under one reset mode; returns [ep1 head, ep1 tail, ep2 head, ep2 tail].

    The two modes differ in exactly one line — whether episode 2 opens with an
    ``env.reset()`` — which mirrors `main.py::run_rollout`'s
    ``if reset_mode == "none" and hasattr(self, '_no_reset_obs')`` branch.
    """
    import gymnasium as gym
    import torch

    import envs.bridge_multi  # noqa: F401 — registers PickPlaceNxM-v1
    from envs.reset import ResetStrategy

    env = gym.make(**env_kwargs)
    device = env.unwrapped.device
    strategy = ResetStrategy(env, num_envs=1, device=device)
    opts = dict(options)
    opts["episode_id"] = torch.full((1,), args.episode_id, dtype=torch.long, device=device)
    action_t = torch.zeros((1, actions.shape[1]), dtype=torch.float32, device=device)

    frames: list[np.ndarray] = []
    for ep in range(2):
        # Episode start. Seeding every reset explicitly (rather than letting
        # `seed=None` fall through to the ambient torch RNG, as training does)
        # is what makes both rows' episode 1 identical and the whole sheet
        # reproducible from `--seed`; the per-episode offset keeps episodic
        # ep2 a genuinely fresh draw instead of a replay of ep1.
        if ep == 0 or reset_mode == "per_episode":
            env.reset(seed=[args.seed * 1000 + ep], options=opts)
        frames.append(rgb_of(env))

        for step_idx in range(args.episode_len):
            action_t[0] = torch.from_numpy(actions[ep * args.episode_len + step_idx]).to(device)
            env.step(action_t)
            at_boundary = (step_idx + 1) % args.task_len == 0
            last_step = step_idx + 1 == args.episode_len
            if last_step and args.tail_phase == "pre_eer":
                # Tail read before the boundary block, so the arm is where the
                # policy left it. `post_eer` instead falls through and reads
                # after EER, matching `_no_reset_obs`.
                frames.append(rgb_of(env))
            if at_boundary and not args.no_eer:
                strategy.reset_robot()
            if last_step and args.tail_phase == "post_eer":
                frames.append(rgb_of(env))
        print(f"  [{reset_mode}] episode {ep + 1} done "
              f"({args.episode_len} steps, {args.episode_len // args.task_len} segments)")

    env.close()
    return frames


def resolve_ckpt(args) -> dict:
    """Read the policy spec out of a checkpoint's `run_config.yaml`.

    `main.py` copies the run's config next to every checkpoint it writes, so the
    four fields needed to reconstitute the policy — family, base model, unnorm
    key, LoRA rank — are already on disk. Reading them beats making the caller
    retype them, and beats guessing: a `vla_path` that disagrees with the
    adapter loads without error and produces a silently wrong policy.
    """
    ckpt = args.ckpt
    if ckpt is None:
        raise SystemExit("--policy vla needs --ckpt <checkpoint dir>")
    ckpt = ckpt.resolve()
    if not (ckpt / "adapter_model.safetensors").exists():
        raise SystemExit(f"{ckpt} has no adapter_model.safetensors — not a "
                         f"checkpoint directory written by main.py")
    spec = {"policy": None, "vla_path": None,
            "vla_unnorm_key": None, "vla_lora_rank": None}
    cfg_path = ckpt / "run_config.yaml"
    if cfg_path.exists():
        import yaml
        saved = yaml.safe_load(cfg_path.read_text()) or {}
        for k in spec:
            if saved.get(k) is not None:
                spec[k] = saved[k]
        print(f"[ckpt] run_config.yaml: policy={spec['policy']} "
              f"base={spec['vla_path']} unnorm={spec['vla_unnorm_key']} "
              f"lora_rank={spec['vla_lora_rank']}")
    else:
        # A checkpoint copied out of its run directory loses `run_config.yaml`.
        # PEFT's own `adapter_config.json` still carries three of the four
        # fields — base model, rank, and (via the wrapped class name) the policy
        # family — so read those rather than making the caller retype them or,
        # worse, defaulting to a family that silently mismatches the adapter.
        import json
        ac_path = ckpt / "adapter_config.json"
        ac = json.loads(ac_path.read_text()) if ac_path.exists() else {}
        base_cls = (ac.get("auto_mapping") or {}).get("base_model_class", "")
        spec["vla_path"] = ac.get("base_model_name_or_path")
        spec["vla_lora_rank"] = ac.get("r")
        if "SpatialVLA" in base_cls:
            spec["policy"] = "spatialvla"
        elif "OpenVLA" in base_cls:
            spec["policy"] = "openvla"
        # The two families spell the same statistics differently:
        # `openvla_train` looks up "bridge_orig", `spatialvla_train`'s processor
        # publishes "bridge_orig/1.0.0". Getting this wrong is a KeyError at
        # `get_action_stats`, not a wrong number, so the guess is safe to make.
        spec["vla_unnorm_key"] = ("bridge_orig/1.0.0"
                                  if spec["policy"] == "spatialvla" else "bridge_orig")
        print(f"[ckpt] no run_config.yaml — read adapter_config.json instead: "
              f"policy={spec['policy']} (from {base_cls or 'unknown class'}) "
              f"base={spec['vla_path']} unnorm={spec['vla_unnorm_key']} "
              f"lora_rank={spec['vla_lora_rank']}")
    for flag, key in (("vla_family", "policy"), ("vla_path", "vla_path"),
                      ("vla_unnorm_key", "vla_unnorm_key"),
                      ("vla_lora_rank", "vla_lora_rank")):
        override = getattr(args, flag)
        if override is not None:
            print(f"[ckpt] --{flag.replace('_', '-')} overrides {key}: "
                  f"{spec[key]!r} -> {override!r}")
            spec[key] = override
    for key, flag in (("policy", "--vla-family"), ("vla_path", "--vla-path"),
                      ("vla_unnorm_key", "--vla-unnorm-key"),
                      ("vla_lora_rank", "--vla-lora-rank")):
        if not spec[key]:
            raise SystemExit(
                f"could not determine `{key}` for {ckpt}: neither run_config.yaml "
                f"nor adapter_config.json supplied it. Pass {flag}.")
    spec["vla_lora_rank"] = int(spec["vla_lora_rank"])
    spec["vla_load_path"] = str(ckpt)
    return spec


def build_policy_args(args, spec: dict, group):
    """The namespace `SpatialVLAPolicy` / `OpenVLAPolicy` and `CronosWrapper` read.

    Both take a single `args` object and pull attributes off it, so one
    namespace serves both — the same trick `eval_only.py::_policy_args` /
    `_wrapper_args` use. Every field here is either a checkpoint fact, one of
    this tool's flags, or a training-only knob that the constructors read but
    that cannot matter with no training step (`vla_lr`, the Adam betas).
    """
    class _NS:
        pass
    ns = _NS()
    for k, v in dict(
        # --- env: one env, the group's objects, this tool's horizon ---
        env_id="PickPlaceNxM-v1", num_envs=1,
        env_n=len(group.obj), env_m=len(group.recep), scene="",
        seed=args.seed, obj_set=args.obj_set,
        obj1_index=7, obj2_index=2, obj3_index=10,
        plate1_index=1, plate2_index=2, plate3_index=3,
        segment_len=args.task_len, episode_len=args.episode_len,
        task_len=args.task_len,
        # The wrapper only reads these to size its own bookkeeping; this tool
        # drives the boundaries itself, exactly as `main.py::run_rollout` does.
        reset_mode="per_episode", reset_robot=not args.no_eer,
        reset_unsuitable=False, unsuitable_detector="low_z",
        enable_backward=False, backward_interval=1,
        task_order="sequential",
        # --- policy ---
        policy=spec["policy"], vla_path=spec["vla_path"],
        vla_load_path=spec["vla_load_path"],
        vla_unnorm_key=spec["vla_unnorm_key"],
        vla_lora_rank=int(spec["vla_lora_rank"]),
        vla_temperature_eval=args.vla_temperature, vla_temperature=1.0,
        vla_lr=1e-4, vla_vhlr=3e-3, vla_grad_norm=10.0,
        vla_optim_beta1=0.9, vla_optim_beta2=0.999, alg_gradient_accum=1,
    ).items():
        setattr(ns, k, v)
    return ns


def build_scheduler(cfg, env, num_envs: int):
    """One `TaskScheduler` over the config's groups, sized for this tool.

    Mirrors `eval_only.py`'s construction, with two changes forced by
    `num_envs=1`: each `GroupState` is built with `num_envs=1`, and `fan_out` is
    forced off. Fanning out would split one env across the group's four tasks
    (`sub = num_envs // n_tasks == 0`) and strand it; with fan-out off the single
    env walks the task sequence one task per segment, which is what a single-env
    figure should show anyway.
    """
    from envs.config import (build_obj_recep_name_maps, has_symbolic_refs,
                             resolve_symbolic_task)
    from envs.scheduler import GroupState, TaskScheduler

    uw = env.env.unwrapped
    states = []
    for g in cfg.groups:
        obj_names, recep_names = build_obj_recep_name_maps(
            g.obj, g.recep, uw.model_db_carrot, uw.model_db_plate)

        def _resolve(tasks):
            return [resolve_symbolic_task(t, obj_names, recep_names)
                    if has_symbolic_refs(t) else t for t in tasks]

        states.append(GroupState.from_sequence(
            name=g.name, task_sequence=_resolve(g.task_sequence),
            eval_tasks=_resolve(g.eval_tasks), num_envs=num_envs))
    return TaskScheduler(group_states=states, mode=cfg.task_order or "sequential",
                         num_envs=num_envs, fan_out=False)


def run_mode_vla(args, reset_mode: str, policy, policy_args, cfg,
                 task_log: list) -> list[np.ndarray]:
    """Two episodes under one reset mode, driven by a loaded checkpoint.

    Unlike the `zero` / `random` path this goes through `CronosWrapper`, because
    a VLA emits action *tokens* and `wrapper._process_action` is what turns them
    into the 7-DoF command — reimplementing that decode here would be a second
    copy of the one thing most likely to drift. The consequence is that this
    path reproduces `main.py::run_rollout` closely: same wrapper, same
    scheduler-driven task switch at each `--task-len` boundary, same EER.
    """
    import torch

    import envs.bridge_multi  # noqa: F401 — registers PickPlaceNxM-v1
    from envs.suite import TaskSuite
    from envs.wrapper import CronosWrapper

    unnorm_state = policy.vla.get_action_stats(policy_args.vla_unnorm_key)
    action_tokenizer = (policy.processor.action_tokenizer
                        if policy_args.policy == "spatialvla" else None)
    env = CronosWrapper(policy_args, unnorm_state, TaskSuite(),
                        device=torch.device("cuda:0"),
                        action_tokenizer=action_tokenizer)
    env.set_group_specs(cfg.groups)
    env.set_scheduler(build_scheduler(cfg, env, num_envs=1))
    policy.prep_rollout()

    frames: list[np.ndarray] = []
    for ep in range(2):
        if ep == 0 or reset_mode == "per_episode":
            # `CronosWrapper.reset` takes no seed — it pins only the overlay, and
            # lets object pose come from the ambient torch RNG (wrapper.py:265,
            # matching training). Two modes run back to back in one process
            # would therefore start episode 1 from different draws, so seed the
            # global RNG here instead: same effect, same place in the sequence,
            # and episode 2 still gets a genuinely different draw.
            torch.manual_seed(args.seed + ep)
            torch.cuda.manual_seed_all(args.seed + ep)
            obs, instruct, _ = env.reset()
        else:
            obs, instruct = env.get_obs_image(), env.get_language_instructions()
        frames.append(obs[0].cpu().numpy())
        task_log.append(instruct[0])

        for step_idx in range(args.episode_len):
            with torch.no_grad():
                _, action, _ = policy.get_action(
                    {"image": obs, "task_description": instruct}, deterministic=True)
            obs, _, _, _ = env.step(action)
            at_boundary = (step_idx + 1) % args.task_len == 0
            last_step = step_idx + 1 == args.episode_len
            if last_step and args.tail_phase == "pre_eer":
                frames.append(obs[0].cpu().numpy())
            if at_boundary:
                # The segment boundary, in `main.py::run_rollout`'s order:
                # advance the task, then EER. HSR and LSR stay off.
                env.scheduler.update_index()
                env.set_forward()
                env.set_task(*env.scheduler.get_next_tasks())
                if not args.no_eer:
                    obs = env.reset_robot()
                instruct = env.get_language_instructions()
                # One entry per segment the policy actually ran, in order. Two
                # runs that are supposed to be comparable must produce the same
                # list; `main()` prints it and diffs the rows.
                if not last_step:
                    task_log.append(instruct[0])
            if last_step and args.tail_phase == "post_eer":
                frames.append(obs[0].cpu().numpy())
        print(f"  [{reset_mode}] episode {ep + 1} done "
              f"({args.episode_len} steps, {args.episode_len // args.task_len} "
              f"segments)")

    env.env.close()
    return frames


def label_sheet(rows: list[list[np.ndarray]], row_labels, col_labels,
                caption: str = "") -> np.ndarray:
    """Grid with a caption row on top and a mode label down the left."""
    from PIL import Image, ImageDraw, ImageFont
    from matplotlib import font_manager

    h, w = rows[0][0].shape[:2]
    n_rows, n_cols = len(rows), len(rows[0])
    pad, gap = 10, 8
    head_h, side_w = 34, 200

    font_path = font_manager.findfont(font_manager.FontProperties(family="DejaVu Sans"))
    font = ImageFont.truetype(font_path, 22)
    font_small = ImageFont.truetype(font_path, 21)
    font_mono = ImageFont.truetype(font_path, 15)

    foot_h = 30 if caption else 0
    sheet_w = side_w + n_cols * w + (n_cols - 1) * gap + 2 * pad
    sheet_h = head_h + n_rows * h + (n_rows - 1) * gap + foot_h + 2 * pad
    img = Image.new("RGB", (sheet_w, sheet_h), (255, 255, 255))
    draw = ImageDraw.Draw(img)

    for c, text in enumerate(col_labels):
        x = pad + side_w + c * (w + gap) + w // 2
        draw.text((x, pad + head_h // 2), text, font=font, fill=(0, 0, 0), anchor="mm")
    for r, (text, frames) in enumerate(zip(row_labels, rows)):
        y = pad + head_h + r * (h + gap)
        # Drawn line by line: PIL rejects a vertical anchor on multiline text.
        lines = text.split("\n")
        for li, line in enumerate(lines):
            ly = y + h // 2 + (li - (len(lines) - 1) / 2) * 26
            draw.text((pad + side_w // 2, ly), line,
                      font=font_small if li == 0 else font_mono,
                      fill=(0, 0, 0) if li == 0 else (110, 110, 110), anchor="mm")
        for c, frame in enumerate(frames):
            img.paste(Image.fromarray(frame), (pad + side_w + c * (w + gap), y))
    if caption:
        # Provenance under the grid: which policy and which checkpoint produced
        # these tail states. A figure of policy behaviour that does not name the
        # policy is not much of a figure.
        draw.text((pad + side_w, sheet_h - pad - foot_h // 2), caption,
                  font=font_mono, fill=(110, 110, 110), anchor="lm")
    return np.asarray(img)


def bare_sheet(rows: list[list[np.ndarray]]) -> np.ndarray:
    h, w = rows[0][0].shape[:2]
    sheet = np.full((len(rows) * h, len(rows[0]) * w, 3), 255, dtype=np.uint8)
    for r, frames in enumerate(rows):
        for c, frame in enumerate(frames):
            sheet[r * h:(r + 1) * h, c * w:(c + 1) * w] = frame
    return sheet


def main() -> None:
    args = parse_args()
    if args.episode_len % args.task_len != 0:
        raise SystemExit(f"--episode-len {args.episode_len} must be divisible by "
                         f"--task-len {args.task_len} (main.py asserts the same)")

    import torch
    if not torch.cuda.is_available():
        raise SystemExit(
            "no CUDA device visible. PickPlaceNxM-v1 requires the GPU PhysX backend "
            "(`envs/bridge_multi.py::_initialize_episode` calls `scene._gpu_apply_all()`), "
            "and SAPIEN needs a Vulkan device it can share semaphores with. Set "
            "CUDA_VISIBLE_DEVICES to an idle card — one env needs ~2.7 GB and no policy.")

    from envs.config import load_cronos_config

    config = load_cronos_config(args.config)
    if not 0 <= args.group < len(config.groups):
        raise SystemExit(f"--group {args.group} out of range (config has {len(config.groups)})")
    group = config.groups[args.group]
    n_obj, n_recep = len(group.obj), len(group.recep)
    print(f"[config] {args.config.name} group '{group.name}': "
          f"obj={group.obj} recep={group.recep} (N={n_obj}, M={n_recep})")

    env_kwargs = dict(
        id="PickPlaceNxM-v1",
        num_envs=1,
        N=n_obj,
        M=n_recep,
        obs_mode="rgb+segmentation",
        control_mode="arm_pd_ee_target_delta_pose_align2_gripper_pd_joint_pos",
        sim_backend="gpu",
        sim_config={"sim_freq": 500, "control_freq": 5},
        max_episode_steps=args.task_len,
        sensor_configs={"shader_pack": "default"},
    )
    options = {"obj_set": args.obj_set}
    for i in range(n_obj):
        options[f"obj{i + 1}_index"] = group.obj[i]
    for i in range(n_recep):
        options[f"plate{i + 1}_index"] = group.recep[i]

    caption = ""
    if args.policy == "vla":
        # The checkpoint is loaded ONCE and both rows share it: nothing about
        # the policy changes between modes, and a second load would cost another
        # ~50 s and another copy of the weights for no gain.
        spec = resolve_ckpt(args)
        policy_args = build_policy_args(args, spec, group)
        # Narrow the config to the one selected group, resized to this tool's
        # single env. `CronosWrapper._build_per_env_indices` slices per-env
        # tensors by each group's declared `num_envs`, so a four-group config
        # left intact would hand env 0 group 0's objects and silently drop the
        # rest — and the scheduler would be built over tasks no env can run.
        import dataclasses
        config_vla = dataclasses.replace(
            config, groups=[dataclasses.replace(group, num_envs=1)])
        if spec["policy"] == "spatialvla":
            from simpler_env.policies.spatialvla.spatialvla_train import (
                SpatialVLAPolicy as _PolicyCls)
        else:
            from simpler_env.policies.openvla.openvla_train import (
                OpenVLAPolicy as _PolicyCls)
        t0 = time.time()
        policy = _PolicyCls(policy_args, device_id=0)
        print(f"[policy] {spec['policy']} from {spec['vla_path']} + LoRA "
              f"{args.ckpt.name}, loaded in {time.time() - t0:.1f}s, "
              f"{torch.cuda.memory_allocated() / 2 ** 30:.1f} GiB")
        task_logs = {}

        def run_one(reset_mode):
            task_logs[reset_mode] = []
            return run_mode_vla(args, reset_mode, policy, policy_args,
                                config_vla, task_logs[reset_mode])
        run_name = next((q.name for q in args.ckpt.resolve().parents
                         if q.name.startswith("CRONOS-")), args.ckpt.parent.name)
        caption = (f"{spec['policy']} · {run_name}/{args.ckpt.name} · "
                   f"temperature {args.vla_temperature} · seed {args.seed} · "
                   f"{args.config.name} group '{group.name}'")
    else:
        # 7-D for the align2 EE controller.
        actions = make_actions(args, 2 * args.episode_len, 7)
        print(f"[policy] {args.policy} (scale={args.action_scale}, "
              f"momentum={args.action_momentum}); |a| mean {np.abs(actions).mean():.3f}")
        task_logs = {}
        run_one = lambda reset_mode: run_mode(  # noqa: E731
            args, reset_mode, env_kwargs, options, actions)

    rows, labels = [], []
    for label, reset_mode in MODES:
        t0 = time.time()
        print(f"[run] {label} (--reset-mode {reset_mode})")
        frames = run_one(reset_mode)
        assert len(frames) == 4, len(frames)
        rows.append(frames)
        labels.append(f"{label}\n--reset-mode {reset_mode}")
        print(f"[run] {label} took {time.time() - t0:.1f}s")

    # The two rows share episode 1 by construction. If they do not, the sheet's
    # ep2 comparison is confounded by sim nondeterminism, so say so loudly
    # rather than letting the figure be read as a reset effect.
    for c, name in ((0, "ep1 head"), (1, "ep1 tail")):
        d = int(np.abs(rows[0][c].astype(np.int16) - rows[1][c].astype(np.int16)).max())
        status = "identical" if d == 0 else f"DIFFERS (max |diff| = {d})"
        print(f"[check] {name} across the two rows: {status}")
        if d != 0:
            print("        the two rows' episode 1 should be bit-identical — same seed, "
                  "same actions. A nonzero diff means the sim is not reproducible "
                  "across env instances here; do not read column 3 as a pure reset effect.")
            if args.policy == "vla" and args.vla_temperature != 0.0:
                print("        --vla-temperature is nonzero, so the policy is SAMPLING; "
                      "that alone explains the divergence. Use 0 for a comparable sheet.")
    d_noep = int(np.abs(rows[1][1].astype(np.int16) - rows[1][2].astype(np.int16)).max())
    print(f"[check] non-episodic ep1 tail vs ep2 head: max |diff| = {d_noep}"
          + ("  (inherited unchanged, as expected)" if d_noep == 0 else
             "  (nonzero — expected only with --tail-phase pre_eer, where the "
             "boundary's EER + settle runs between the two frames)"))

    # The task sequence the policy was actually given, segment by segment.
    # Printed in full because comparing two horizons (`--episode-len 80` vs
    # `1280`) is only meaningful if both walked the same tasks in the same
    # order; the shorter run's list must be a prefix of the longer one's.
    if task_logs:
        for reset_mode, log in task_logs.items():
            print(f"[tasks] {reset_mode}: {len(log)} segments")
            for i, t in enumerate(log):
                print(f"         seg {i:2d}  {t}")
        seqs = list(task_logs.values())
        same = all(seq == seqs[0] for seq in seqs)
        print(f"[check] task sequence across the two rows: "
              f"{'identical' if same else 'DIFFERS — the rows are not comparable'}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    import imageio.v3 as iio

    if not args.no_tiles:
        tile_dir = args.out.parent / f"{args.out.stem}_tiles"
        tile_dir.mkdir(parents=True, exist_ok=True)
        slugs = ("ep1_head", "ep1_tail", "ep2_head", "ep2_tail")
        for (label, reset_mode), frames in zip(MODES, rows):
            for slug, frame in zip(slugs, frames):
                iio.imwrite(tile_dir / f"{reset_mode}__{slug}.png", frame)
        print(f"[tiles] wrote {sum(len(r) for r in rows)} PNGs to {tile_dir}")

    sheet = (bare_sheet(rows) if args.no_labels
             else label_sheet(rows, labels, COL_LABELS, caption=caption))
    iio.imwrite(args.out, sheet)
    print(f"[sheet] {sheet.shape[1]}x{sheet.shape[0]} px")
    print(f"[out] {args.out}")


if __name__ == "__main__":
    main()
