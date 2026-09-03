"""Per-segment (per-80-step) object position distribution, from `segment_pose.csv`.

`segment_pose.csv` has one row per (episode, segment, phase, env, actor), written
at every `task_len` boundary — 80 steps by default — so "per-80" is the file's
native granularity.

A boundary is not one instant, which is what the `phase` column records:

    start   the state the segment BEGINS from — after that boundary's HSR
            respawn and EER `reset_robot()` (whose `_settle(0.5)` also nudges
            objects), and after the full `env.reset()` at an episode boundary.
            This is the initial-state distribution the forward policy faces,
            and what `--backward-goal` (perturbation) is meant to widen.
    end     the steady state the policy produced, before any of those resets.
            Anchor `workspace_aabb` bounds from this one.

`--phase` defaults to `start`. They are not interchangeable: `--reset-robot` is
on by default in every reset mode, so the gripper always differs between them
and the objects often do.

A run recorded before the `phase` split holds only `end` rows, and `--phase
start` then rebuilds them: the start of segment s is the recorded end of s-1
wherever the boundary provably did not move anything, and only the boundaries
that really are an `env.reset()` draw are synthesized. See `rebuild_start_rows`
— it also explains when the rebuild is not available and the tool falls back to
synthesizing every start.

    python tools/plot_segment_positions.py --run-dir <RUN_OUT_DIR>/wandb/run-*/glob

Layout: **one PNG per figure, never a grid.** Each figure is a single xy scatter
of one `actor_kind` (`obj` or `recep`) for one experiment, so a run of the tool
writes `<...>_obj.png` and `<...>_recep.png`, and in `--config` mode one such
pair per group. Overlaying kinds or groups in one image made every panel small
and forced a shared colour scale onto distributions that are read one at a time;
the xy view range is still shared across every figure a single invocation
writes, which is what actually makes them comparable.

    obj     the objects the task asks to move
    recep   the receptacles they are moved onto

The scatter is coloured by episode, so drift over training is visible.

`--step-range` selects which part of the run to draw, and its default
(`DEFAULT_STEP_RANGE`) is **not the whole run** — a longer run is cropped unless
the range is widened, so pass `--step-range all` for everything. What it kept is
always reported on stderr. Recorded and rebuilt rows are treated the same by it.

Notes on the data
-----------------
- Hidden slots (a YAML group declaring fewer objects than the batch-wide N)
  write NaN, deliberately, so the row count per segment stays fixed. They are
  dropped here.
- The gripper is recorded in `segment_pose.csv` but is not plotted: with EER on
  every `phase=start` gripper row is the identical homed pose, so its panel was
  a single dot. `summarize()` still reports its extent on stderr.
- `actor_kind` covers **every** object and receptacle slot, not just the pair
  the current task selected — so distractor objects the policy was supposed to
  leave alone are included. Use `--slot` / `--model` to narrow.
- There is no `group` column. Filter by `--model` (the per-env model name) or
  `--task` instead; under fan-out, slot 0 is a different model in different envs.
- `--forward-only` joins against `rollout_success.csv` on (episode, segment,
  env) and keeps only forward segments. Worth using under a reset mode that
  includes LSR (`LSR`, `HSR+LSR`, `noep+LSR`), where half the segment ends are
  reset-goal states and would otherwise be mixed in. Without LSR — bare `noep`
  included — every row is already `forward`, so it is a no-op.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from functools import lru_cache
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from plot_common import (concat_chain, default_colors, load_plot_config,  # noqa: E402
                         read_run_config)

# `envs/unsuitable.py::LowZDetector.z_threshold` — the height below which HSR
# treats an actor as fallen. Reported by `summarize()`; there is no pz figure.
LOW_Z_THRESHOLD = 0.7

# Only the task-relevant kinds are plotted. `phase=start` pz is the preset's
# fixed `slot_heights`, so a height histogram of the initial-state distribution
# is one bar by construction and the pz question is answered by the
# below-threshold counts `summarize()` prints. The gripper's xy is likewise a
# single pinned pose under EER.
_KIND_ORDER = ("obj", "recep")

# (N, M) -> (POSE_PRESET, SLOT_ORDER). Mirrors `_NxM_PRESETS` in
# `envs/bridge_multi.py`, which cannot be imported here because it pulls in
# ManiSkill/SAPIEN. Only the two fields the synthetic reconstruction needs are
# copied; the pose tables themselves come from `envs/suite.py` so there is a
# single source of truth for the geometry.
_NXM_PRESET = {
    (2, 1): ("TwoObjectOneReceptacle", [0, 2, 1]),
    (1, 2): ("OneObjectTwoReceptacle", None),
    (2, 2): ("TwoObjectTwoReceptacle", None),
    (3, 3): ("ThreeObjectThreeReceptacle", None),
    (3, 1): ("ThreeObjectOneReceptacle", None),
    (1, 3): ("OneObjectThreeReceptacle", None),
    (3, 2): ("ThreeObjectTwoReceptacle", None),
    (2, 3): ("TwoObjectThreeReceptacle", None),
}


_VALID_KINDS = _KIND_ORDER


def parse_actor_kinds(value) -> list:
    """`--actor-kind` / config `actor_kind` -> an ordered list of kinds.

    Accepts "all", one kind, or a comma-separated list ("obj,recep"). A list is
    also accepted from the config, where JSON can express it directly.
    """
    if value is None or value == "all":
        return list(_VALID_KINDS)
    parts = value if isinstance(value, list) else [v.strip() for v in str(value).split(",")]
    parts = [p for p in parts if p]
    if "all" in parts:
        return list(_VALID_KINDS)
    if "gripper" in parts:
        raise SystemExit("actor kind 'gripper' is no longer plotted (its pose is "
                         "pinned by EER); choose from ['obj', 'recep'] or 'all'")
    bad = [p for p in parts if p not in _VALID_KINDS]
    if bad:
        raise SystemExit(f"unknown actor kind(s) {bad}; choose from "
                         f"{list(_VALID_KINDS)} or 'all', comma-separated")
    # De-duplicate while keeping _KIND_ORDER for a stable figure order.
    return [k for k in _VALID_KINDS if k in parts]


# `--step-range` default: one training segment's step budget (see
# `scripts/train.sh`'s horizon table — the 'a' segments each run 655,360 env
# steps, and every horizon's `total_steps` advances 5120 per boundary). A run
# longer than this is CROPPED by default, so `apply_filters` always reports what
# the range kept.
DEFAULT_STEP_RANGE = "0:163840"


def parse_step_range(value):
    """`--step-range` / config `step_range` -> (lo, hi), or None for no filter.

    Accepts `LO:HI`, an open end on either side (`:HI`, `LO:`), and `all`.
    """
    if value is None:
        value = DEFAULT_STEP_RANGE
    text = str(value).strip()
    if text.lower() in ("all", "none", ""):
        return None
    if ":" not in text:
        raise SystemExit(f"--step-range must look like LO:HI (or 'all'), got {text!r}")
    lo_s, hi_s = text.split(":", 1)
    try:
        lo = float(lo_s) if lo_s.strip() else float("-inf")
        hi = float(hi_s) if hi_s.strip() else float("inf")
    except ValueError:
        raise SystemExit(f"--step-range bounds must be numbers, got {text!r}")
    if lo > hi:
        raise SystemExit(f"--step-range LO must not exceed HI, got {text!r}")
    return lo, hi


def _slug(text: str) -> str:
    """A group label as a filename fragment (labels carry spaces and '+')."""
    s = re.sub(r"[^0-9A-Za-z._-]+", "-", str(text)).strip("-._")
    return s or "group"


def unique_slugs(labels) -> dict:
    """label -> distinct filename fragment. Two labels can slugify the same
    way ("noep +PTB" and "noep-PTB"), and the second figure would silently
    overwrite the first, so collisions get a numeric suffix."""
    out, used = {}, set()
    for label in labels:
        base = _slug(label)
        slug, i = base, 2
        while slug in used:
            slug, i = f"{base}-{i}", i + 1
        used.add(slug)
        out[label] = slug
    return out


def out_variant(base: Path, *parts: str) -> Path:
    """`fig.png` + ("noep", "obj") -> `fig_noep_obj.png`."""
    base = Path(base)
    return base.with_name("_".join([base.stem, *parts]) + (base.suffix or ".png"))


def load_pose(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(
            f"{csv_path} not found. It is written only when --record-segment-pose "
            f"is on (it is on by default; --no-record-segment-pose disables it)."
        )
    df = pd.read_csv(csv_path)
    for col in ("px", "py", "pz"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    before = len(df)
    df = df.dropna(subset=["px", "py", "pz"])
    hidden = before - len(df)
    if hidden:
        print(f"[pose] dropped {hidden} hidden-slot rows (NaN by design)", file=sys.stderr)
    if "phase" not in df.columns:
        # CSVs written before the phase split recorded end-of-segment only.
        df["phase"] = "end"
    return df


@lru_cache(maxsize=None)
def pose_configs(preset: str):
    """`xyz_configs` for a preset, built once per process.

    `generate_pose_configs` brute-forces `itertools.product` over a 36-point grid
    — 36^4 = 1.7M candidate layouts for the 2x2 preset, ~9 s. It was being
    rebuilt once per run directory, so a config listing six runs spent a minute
    recomputing an identical table. It depends only on the preset name, so cache
    on that.
    """
    from envs.suite import POSE_PRESETS, generate_pose_configs
    return generate_pose_configs(**POSE_PRESETS[preset])


def workspace_extent(run_dirs):
    """The env's own sampling region, read from the pose preset.

    Far better than inferring a view range from the data: `xyz_configs` IS the
    set of positions `_initialize_episode` can draw, so its extent is the
    workspace by definition — deterministic, independent of how many actors
    escaped, and identical across runs of the same (N, M).

    Returns `(xlim, ylim, (z_lo, z_hi))` for the union over `run_dirs`, or None
    when the preset cannot be determined (no `run_config.json`, unsupported
    (N, M), or `envs.suite` not importable).
    """
    presets = set()
    for run_dir in {Path(d) for d in run_dirs}:
        rc = read_run_config(run_dir)
        if not rc:
            continue
        key = (int(rc.get("env_n", 2)), int(rc.get("env_m", 2)))
        if key in _NXM_PRESET:
            presets.add(_NXM_PRESET[key][0])
    boxes = []
    for name in presets:
        try:
            xyz = pose_configs(name)
        except ImportError:
            return None
        boxes.append((xyz[..., 0].min(), xyz[..., 0].max(),
                      xyz[..., 1].min(), xyz[..., 1].max(),
                      xyz[..., 2].min(), xyz[..., 2].max()))
    if not boxes:
        return None
    a = np.array(boxes, dtype=float)
    return ((float(a[:, 0].min()), float(a[:, 1].max())),
            (float(a[:, 2].min()), float(a[:, 3].max())),
            (float(a[:, 4].min()), float(a[:, 5].max())))


def _scale_box(lo: float, hi: float, scale: float):
    """Grow an interval about its centre by `scale` (1.0 = unchanged)."""
    mid, half = (hi + lo) / 2.0, (hi - lo) / 2.0
    return (mid - half * scale, mid + half * scale)


def _robust_range(v, k: float = 6.0, pad: float = 0.01, enabled: bool = True):
    """A view range for `v` that a few runaway actors cannot blow out.

    Uses median ± k·MAD rather than a quantile. A quantile needs to be told what
    fraction is bad — `--clip-quantile 0.999` removes 0.1%, so 2% of runaways
    survive and the axis is destroyed anyway — whereas MAD estimates the spread
    of the *bulk* and is unaffected by how many outliers there are.

    Never clips tighter than the data: the result is intersected with the true
    min/max, so a well-behaved column keeps its exact range and nothing is
    clipped when there is nothing to clip.

    MAD is 0 when half the points are identical (the homed gripper). Fall back
    to a small window around the median instead of the full range, which one
    outlier would otherwise stretch to infinity.
    """
    v = np.asarray(v, dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return (-pad, pad)
    lo_d, hi_d = float(v.min()), float(v.max())
    if not enabled:
        return (lo_d - pad, hi_d + pad)
    med = float(np.median(v))
    mad = float(np.median(np.abs(v - med)))
    half = k * mad if mad > 0 else pad
    lo, hi = max(med - half, lo_d), min(med + half, hi_d)
    if hi - lo < 2 * pad:
        lo, hi = med - pad, med + pad
    return (lo - pad, hi + pad)


def _shared_limits(df: pd.DataFrame, pad: float = 0.01, robust: bool = True,
                   ws=None, scale: float = 3.0):
    """xy limits shared by every figure one invocation writes, robust to
    runaway actors.

    Shared, because otherwise each figure auto-scales to its own spread and a
    tight cluster looks like a wide one — and because a degenerate one (a kind
    whose pose is pinned) would zoom into millimetres of float noise. This is
    what keeps separate PNGs comparable now that nothing is drawn side by side.

    Robust, because actors do escape. The `low_z` detector only tests height, so
    an object flung sideways at table height is never flagged and never
    respawned; one that misses the table keeps falling. Both produce coordinates
    orders of magnitude outside the 0.15 x 0.15 m workspace, and on a shared axis
    one such point compresses every real point into a pixel.

    Nothing is dropped from the data or from any count — only the VIEW is
    bounded, and `_report_offscreen` states how many points it excludes.
    """
    if ws is not None:
        (x0, x1), (y0, y1), _ = ws
        return _scale_box(x0, x1, scale), _scale_box(y0, y1, scale)
    return (_robust_range(df["px"], pad=pad, enabled=robust),
            _robust_range(df["py"], pad=pad, enabled=robust))


def _report_offscreen(df: pd.DataFrame, xlim, ylim, label: str = "") -> int:
    """Count and announce points the clipped view cannot show."""
    off = (~df["px"].between(*xlim) | ~df["py"].between(*ylim)).sum()
    if off:
        tag = f"{label}: " if label else ""
        print(f"[pose] {tag}{off}/{len(df)} points ({off / len(df):.2%}) lie outside "
              f"the plotted range and are not drawn — px {df['px'].min():.3f}…"
              f"{df['px'].max():.3f}, py {df['py'].min():.3f}…{df['py'].max():.3f}. "
              f"Use --no-clip to include them.", file=sys.stderr)
    return int(off)


def synth_start_poses(run_dir: Path, n_draws: int, seed: int = 0,
                      segment: int = 1, episodes=None,
                      total_steps=None) -> pd.DataFrame:
    """Reconstruct the *distribution* of segment-start poses for an old run.

    Runs recorded before the `phase` split hold end-of-segment rows only, and the
    start poses cannot be recovered: `_initialize_episode_pre` draws them with
    `torch.randint` on the global CUDA generator, which the VLA's action sampling
    also consumes, and HSR draws with `np.random.choice`, which the PPO minibatch
    shuffle also consumes. Neither index is logged, so a same-seed replay would
    have to reproduce the entire training bit-for-bit.

    What *is* recoverable is the distribution those draws came from. The sampler
    is uniform over a deterministic table — `xyz_configs`, built by
    `envs/suite.py::generate_pose_configs` from the (N, M) preset with no
    randomness at all — so drawing uniformly from that same table reproduces the
    initial-state distribution exactly. Only the per-env identities are lost.

    `n_draws` is taken from the run's own reset count, so the synthetic cloud has
    the same sample size the real run would have produced: a T80 run (128
    episodes x 64 envs) draws far more than a T2560 one (4 x 64), and the plots
    show that difference in density instead of hiding it.

    **This is only the right reconstruction for a boundary that really is a
    fresh draw** — i.e. an `env.reset()`. `rebuild_start_rows` decides which
    boundaries those are and calls this for those alone; every other segment
    start is carried over from the recorded `end` instead. See that function.

    Rows come back tagged `phase="start"`, `synthetic=True`, and `segment` set
    by the caller: 1 when these are episode-first starts, -1 when the run's
    structure could not be determined and every start is being synthesized.

    `episodes` / `total_steps` are per-draw stamps, length `n_draws`. Passing
    them makes the synthetic rows selectable by `--step-range`, `--episode-range`
    and `--last-episodes` like any other row; without them the rows carry -1 and
    those filters treat them as undated (see `apply_filters`).
    """
    try:
        pose_configs(_NXM_PRESET[(2, 2)][0])   # probe importability
    except ImportError as e:
        raise SystemExit(
            f"synthetic reconstruction needs `envs.suite` importable (numpy + "
            f"transforms3d; no GPU stack required). Run from the CRONOS "
            f"directory. Underlying error: {e}"
        )

    rc = read_run_config(run_dir) or {}
    n_obj, n_rec = int(rc.get("env_n", 2)), int(rc.get("env_m", 2))
    if (n_obj, n_rec) not in _NXM_PRESET:
        raise SystemExit(f"{run_dir}: unsupported (N={n_obj}, M={n_rec}) for synthesis")
    preset, slot_order = _NXM_PRESET[(n_obj, n_rec)]
    xyz = pose_configs(preset)                               # (Ncfg, N+M, 3)

    def physical(logical: int) -> int:
        return logical if slot_order is None else slot_order[logical]

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(xyz), size=n_draws)             # uniform, as the env does
    rows = []
    for kind, count, base in (("obj", n_obj, 0), ("recep", n_rec, n_obj)):
        for logical in range(count):
            p = xyz[idx, physical(base + logical)]            # (n_draws, 3)
            rows.append(pd.DataFrame({
                # Without stamps `episode`/`total_steps` stay -1: the draws are
                # exchangeable, so no row can be attributed to a moment in the
                # run. With stamps each draw belongs to a known boundary.
                "episode": -1 if episodes is None else episodes,
                "segment": segment, "phase": "start",
                "total_steps": -1 if total_steps is None else total_steps,
                "env": np.arange(n_draws) % max(1, int(rc.get("num_envs", 64))),
                "actor_kind": kind, "slot": logical,
                "model_name": "", "task": "",
                "px": p[:, 0], "py": p[:, 1], "pz": p[:, 2],
                "synthetic": True,
            }))
    out = pd.concat(rows, ignore_index=True)
    print(f"[synth] {run_dir.name}: {len(xyz)} configs x uniform, {n_draws} draws "
          f"-> {len(out)} rows ({preset})", file=sys.stderr)
    return out


def reset_count(run_dir: Path) -> int:
    """The run's total reset count — one per per-env fresh pose draw.

    `hard_reset_count` advances by `num_envs` per episode (every env is
    re-randomized by `env.reset()`) and `soft_reset_count` by the number of envs
    HSR respawned, so the sum is exactly how many independent draws from
    `xyz_configs` the run made. Read from `counters.json`, falling back to the
    last `total_resets` in `rollout_success.csv`.
    """
    counters = Path(run_dir) / "counters.json"
    if counters.exists():
        try:
            return int(json.loads(counters.read_text())["total_resets"])
        except Exception:
            pass
    roll = Path(run_dir) / "rollout_success.csv"
    if roll.exists():
        r = pd.read_csv(roll, usecols=["total_resets"])
        if len(r):
            return int(pd.to_numeric(r["total_resets"], errors="coerce").max())
    raise SystemExit(f"{run_dir}: cannot determine the reset count "
                     f"(no counters.json, no rollout_success.csv)")


def _segments_per_episode(rc: dict):
    """`episode_len / task_len` — how many segment boundaries an episode holds.

    1 means every segment start is also an episode start, so every one of them
    follows a full `env.reset()`. That is the T80 case, and the only shape for
    which synthesizing *all* starts is right.
    """
    try:
        ep, task = int(rc["episode_len"]), int(rc["task_len"])
    except (KeyError, TypeError, ValueError):
        return None
    return max(1, ep // task) if task > 0 else None


def _carryover_is_exact(rc: dict):
    """Whether the end of one segment IS the start of the next, exactly.

    Between the `phase="end"` record and the next segment's first step the train
    loop runs exactly two things that can move an actor, and nothing else:

        --reset-unsuitable   HSR respawns the flagged envs to a fresh draw
        --reset-robot        EER's `reset_robot()`, whose `_settle(0.5)` also
                             lets objects drift

    With both off, the boundary only reassigns the task and no `env.step` runs,
    so the poses are unchanged and the carry-over is exact rather than merely
    close. LSR is irrelevant here: `set_backward_goals` swaps the goal, not the
    poses.

    Returns `(exact, reason_if_not)`.
    """
    if rc.get("reset_unsuitable"):
        return False, "reset_unsuitable=True — HSR respawns flagged envs at the boundary"
    if rc.get("reset_robot"):
        return False, "reset_robot=True — EER's _settle(0.5) nudges objects"
    return True, ""


def rebuild_start_rows(run_dir: Path, df: pd.DataFrame, args) -> pd.DataFrame:
    """`phase=start` rows for a run that only recorded `phase=end`.

    Only the boundaries that really are a fresh `xyz_configs` draw need
    synthesizing. Every other segment start is a pose the run already recorded,
    one segment earlier — `segment` in the CSV is 1-based and the recorder logs
    the start of segment s with that same s, so the start of s is the end of
    s-1. Synthesizing all of them (which this tool used to do) throws away
    those measurements and replaces them with an idealized 16-site lattice; the
    further into an episode a segment sits the more wrong that is. Measured on
    Q1's T320, the fraction of end poses still within 1 mm of a spawn site falls
    26.9% -> 22.2% -> 18.1% -> 15.1% across the four segments of an episode.

    Which boundaries are fresh draws depends on the reset mode:

        reset_mode=per_episode   segment 1 of every episode (`env.reset()`)
        reset_mode=none          only the very first boundary of the run

    Returns the rows to append to `df`. Empty when the run's structure cannot be
    determined or the carry-over would not be exact — the caller then falls back
    to synthesizing everything, which is the previous behaviour.
    """
    rc = read_run_config(run_dir)
    if not rc:
        print(f"[start] {run_dir.name}: no run_config.json, cannot tell which "
              f"boundaries were env.reset() — synthesizing every start",
              file=sys.stderr)
        return pd.DataFrame()

    segs = _segments_per_episode(rc)
    if segs is None:
        print(f"[start] {run_dir.name}: run_config.json has no episode_len/"
              f"task_len — synthesizing every start", file=sys.stderr)
        return pd.DataFrame()

    exact, why = _carryover_is_exact(rc)
    if not exact:
        print(f"[start] {run_dir.name}: {why}; the previous segment's end is no "
              f"longer the next one's start — synthesizing every start",
              file=sys.stderr)
        return pd.DataFrame()

    end = df[df["phase"] == "end"]
    if end.empty:
        return pd.DataFrame()

    per_episode = str(rc.get("reset_mode", "per_episode")) != "none"
    # segs == 1 (T80) needs no special case: every boundary is then an episode
    # first, `nxt % segs != 0` is false for all of them, and the code below
    # correctly derives nothing and synthesizes every start — which for that
    # shape is the right answer, not a fallback. Going through the same path
    # also stamps those draws with their episode and step.

    # Flatten (episode, segment) to one boundary ordinal so both reset modes are
    # handled by the same "the start of ordinal o+1 is the end of ordinal o".
    eps = sorted(end["episode"].unique())
    ep_idx = {e: i for i, e in enumerate(eps)}
    inv = {i: e for e, i in ep_idx.items()}
    n_bnd = len(eps) * segs

    end = end.copy()
    end["_ord"] = end["episode"].map(ep_idx) * segs + (end["segment"] - 1)
    nxt = end["_ord"] + 1
    keep = nxt < n_bnd                      # the run's last end feeds no start
    if per_episode:
        keep &= (nxt % segs != 0)           # episode-first starts are fresh draws
    derived = end[keep].copy()
    derived["_ord"] = derived["_ord"] + 1
    derived["episode"] = (derived["_ord"] // segs).map(inv)
    derived["segment"] = (derived["_ord"] % segs) + 1
    derived["phase"] = "start"
    derived["synthetic"] = False            # a recorded pose, re-labelled
    derived = derived.drop(columns=["_ord"])

    # Every boundary that is NOT derived is a fresh draw and still needs one.
    # Stamp each draw with the boundary it stands for so it filters like real
    # data: a boundary's start is the previous boundary's end, i.e. one
    # boundary's worth of steps earlier (`task_len * num_envs` — every boundary
    # advances `total_steps` by exactly that).
    n_envs = max(1, int(rc.get("num_envs", 64)))
    step_per_bnd = int(rc.get("task_len", 80)) * n_envs
    firsts = (end[end["segment"] == 1][["episode", "total_steps"]]
              .drop_duplicates().sort_values("episode"))
    if not per_episode:
        firsts = firsts.head(1)        # only the run's very first boundary
    ep_stamp = np.repeat(firsts["episode"].to_numpy(), n_envs)
    ts_stamp = np.repeat(firsts["total_steps"].to_numpy() - step_per_bnd, n_envs)
    n_fresh = len(firsts)

    print(f"[start] {run_dir.name}: {segs} segments/episode, "
          f"reset_mode={'per_episode' if per_episode else 'none'} -> "
          f"{len(derived)} start rows carried over from the recorded ends, "
          f"{n_fresh}/{n_bnd} boundaries still need a draw", file=sys.stderr)

    if args.no_synth:
        print(f"[start] {run_dir.name}: --no-synth, so the {n_fresh} "
              f"env.reset() boundaries are omitted", file=sys.stderr)
        return derived
    synth = synth_start_poses(run_dir, n_fresh * n_envs, args.synth_seed,
                              segment=1, episodes=ep_stamp, total_steps=ts_stamp)
    return pd.concat([derived, synth], ignore_index=True)


def ensure_start_rows(run_dir: Path, df: pd.DataFrame, args) -> pd.DataFrame:
    """Append rebuilt `phase=start` rows when `df` has none and they are wanted.

    Shared by the single-run and `--config` paths. It used to live only in the
    group loader, so `--run-dir --phase start` on a pre-`phase` run failed with
    "no rows with phase='start'" instead of rebuilding them.
    """
    if args.phase not in ("start", "all"):
        return df
    if (df["phase"] == "start").any():
        return df
    extra = rebuild_start_rows(run_dir, df, args)
    if extra.empty and args.no_synth:
        print(f"[warn] {run_dir.name}: no phase=start rows, nothing could be "
              f"carried over, and --no-synth given", file=sys.stderr)
    elif extra.empty:
        extra = synth_start_poses(run_dir, reset_count(run_dir),
                                  args.synth_seed, segment=-1)
    return pd.concat([df, extra], ignore_index=True) if not extra.empty else df


def load_group_poses(run_dirs, args) -> pd.DataFrame:
    """Load one group's runs, rebuilding `start` rows where they are missing."""
    frames = []
    for run_dir in run_dirs:
        run_dir = Path(run_dir)
        df = load_pose(run_dir / "segment_pose.csv")
        df["synthetic"] = False
        frames.append(ensure_start_rows(run_dir, df, args))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def apply_filters(df: pd.DataFrame, args, csv_path: Path) -> pd.DataFrame:
    if args.phase != "all":
        sel = df[df["phase"] == args.phase]
        if sel.empty:
            avail = sorted(df["phase"].unique())
            raise SystemExit(
                f"no rows with phase={args.phase!r}; present: {avail}. The run may "
                f"have used --segment-pose-phase to record only one side."
            )
        df = sel
    kinds = parse_actor_kinds(args.actor_kind)
    if set(kinds) != set(_VALID_KINDS):
        df = df[df["actor_kind"].isin(kinds)]
    if args.slot is not None:
        df = df[df["slot"] == args.slot]
    if args.model:
        df = df[df["model_name"].astype(str).str.contains(args.model, case=False, na=False)]
    if args.task:
        df = df[df["task"].astype(str).str.contains(args.task, case=False, na=False)]
    if args.segment is not None:
        df = df[df["segment"] == args.segment]
    rng = parse_step_range(args.step_range)
    if rng is not None:
        lo, hi = rng
        # Rows with total_steps < 0 are undated — synthetic draws made without a
        # known boundary (see `synth_start_poses`). They are kept rather than
        # silently dropped, because dropping them would empty the figure for a
        # run whose structure could not be reconstructed.
        dated = df["total_steps"] >= 0
        before = len(df)
        df = df[~dated | (df["total_steps"].between(lo, hi))]
        undated = int((~dated).sum())
        note = f", {undated} undated rows kept" if undated else ""
        print(f"[pose] --step-range {lo:g}:{hi:g} kept {len(df)}/{before} rows"
              f"{note}", file=sys.stderr)
        if len(df) < before - undated:
            print(f"[pose] {before - undated - (len(df) - undated)} rows lie "
                  f"outside that step range — widen it or pass "
                  f"--step-range all to use the whole run", file=sys.stderr)
    if args.episode_range:
        try:
            lo, hi = (int(v) for v in args.episode_range.split(":"))
        except ValueError:
            raise SystemExit("--episode-range must look like 5:20")
        df = df[(df["episode"] >= lo) & (df["episode"] <= hi)]
    if args.last_episodes:
        cutoff = df["episode"].max() - args.last_episodes + 1
        df = df[df["episode"] >= cutoff]
    if args.forward_only:
        df = _keep_forward(df, csv_path)
    if df.empty:
        raise SystemExit("no rows left after filtering")
    return df


def _keep_forward(df: pd.DataFrame, pose_csv: Path) -> pd.DataFrame:
    """Drop reset-goal segments by joining on rollout_success.csv.

    `segment_pose.csv` carries no `direction`, but the two files share the
    (episode, segment, env) key — `rollout_success.csv`'s `env_idx` is
    `segment_pose.csv`'s `env`.
    """
    roll = pose_csv.with_name("rollout_success.csv")
    if not roll.exists():
        print(f"[pose] --forward-only: {roll.name} not found, keeping all segments",
              file=sys.stderr)
        return df
    r = pd.read_csv(roll, usecols=lambda c: c in
                    {"episode", "segment", "env_idx", "direction"})
    if "direction" not in r.columns:
        return df
    fwd = (r[r["direction"] == "forward"][["episode", "segment", "env_idx"]]
           .drop_duplicates().rename(columns={"env_idx": "env"}))
    before = len(df)
    df = df.merge(fwd, on=["episode", "segment", "env"], how="inner")
    print(f"[pose] --forward-only kept {len(df)}/{before} rows", file=sys.stderr)
    return df


def _new_panel():
    """One figure, one xy scatter. Square, because the axes are equal-aspect."""
    return plt.subplots(figsize=(5.8, 5.8))


def report_panel(label: str, sub: pd.DataFrame) -> None:
    """What the figure no longer says, on stderr.

    The panel used to carry the point count, the synthetic share and the
    below-`low_z` fraction in its title. The figures are captioned wherever they
    are used, so the title is gone — but the numbers are still the ones you need
    to read the cloud, so they are printed instead of dropped.
    """
    below = int((sub["pz"] < LOW_Z_THRESHOLD).sum())
    n_synth = int(sub["synthetic"].sum())
    synth = f", {n_synth} synthetic" if n_synth else ""
    print(f"[pose] {label}: n={len(sub)}{synth}, {below} below "
          f"low_z={LOW_Z_THRESHOLD} ({below / max(1, len(sub)):.1%})",
          file=sys.stderr)


def _finish_panel(ax, *, xlim, ylim, workspace) -> None:
    if workspace:
        x0, x1, y0, y1 = workspace
        ax.add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False,
                                   edgecolor="crimson", linestyle="--",
                                   linewidth=1.2, label="workspace"))
    # No title and no legend. There is one actor kind and one experiment per
    # figure, so a legend labels the only series on the axes, and the caption
    # belongs to whatever document uses the figure. `report_panel` prints the
    # counts. The workspace rectangle keeps its label only so a reader who
    # enables it can still tell what the dashed box is — see below.
    ax.set_xlabel("px")
    ax.set_ylabel("py")
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(alpha=0.25)
    if workspace:
        ax.legend(loc="upper right", fontsize=7)


def _save_panel(fig, out_path: Path) -> Path:
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _announce_view(ws, scale: float, xlim, ylim) -> None:
    if ws is not None:
        print(f"[pose] view from the pose preset's workspace x{scale:g}: "
              f"px {xlim[0]:.3f}…{xlim[1]:.3f}  py {ylim[0]:.3f}…{ylim[1]:.3f}",
              file=sys.stderr)


def render(df: pd.DataFrame, out_base: Path, *, hexbin: bool, workspace,
           label: str = "", robust: bool = True, ws=None,
           scale: float = 3.0) -> list:
    """One PNG per actor kind: `<out_base stem>_obj.png`, `..._recep.png`."""
    kinds = [k for k in _KIND_ORDER if k in set(df["actor_kind"])]
    if not kinds:
        raise SystemExit(f"no rows for the plotted kinds {list(_KIND_ORDER)}; "
                         f"present: {sorted(set(df['actor_kind']))}")
    df = df[df["actor_kind"].isin(kinds)]

    ep_lo, ep_hi = int(df["episode"].min()), int(df["episode"].max())
    # One view range for every figure this call writes: separate figures are
    # only comparable if the same cluster comes out the same size in each.
    xlim, ylim = _shared_limits(df, robust=robust, ws=ws, scale=scale)
    _announce_view(ws, scale, xlim, ylim)

    written = []
    for kind in kinds:
        sub = df[df["actor_kind"] == kind]
        _report_offscreen(sub, xlim, ylim, kind)
        fig, ax = _new_panel()
        if hexbin:
            hb = ax.hexbin(sub["px"], sub["py"], gridsize=45, cmap="viridis",
                           mincnt=1, linewidths=0, extent=(*xlim, *ylim))
            fig.colorbar(hb, ax=ax, label="count", shrink=0.85)
        else:
            sc = ax.scatter(sub["px"], sub["py"], c=sub["episode"], cmap="viridis",
                            s=5, alpha=0.45, linewidths=0,
                            vmin=ep_lo, vmax=max(ep_hi, ep_lo + 1))
            if ep_hi > ep_lo:
                fig.colorbar(sc, ax=ax, label="episode", shrink=0.85)
        report_panel(f"{label} {kind}".strip(), sub)
        # A single distinct xy means the pose is pinned rather than sparsely
        # sampled. Say so; a lone dot on a clipped axis is otherwise easy to
        # misread as missing data — and the figure no longer has a title to
        # carry the warning.
        spread = max(sub["px"].max() - sub["px"].min(),
                     sub["py"].max() - sub["py"].min())
        if spread < 1e-9:
            print(f"[pose] {kind}: every point is the SAME xy — a pinned pose, "
                  f"not missing data", file=sys.stderr)
        _finish_panel(ax, xlim=xlim, ylim=ylim, workspace=workspace)
        written.append(_save_panel(fig, out_variant(out_base, kind)))
    return written


def render_groups(cfg, out_base: Path, *, args) -> list:
    """One PNG per (config group × actor kind).

    Each experiment gets its own figure rather than a column of a grid: a group
    is read on its own — "where did this condition's objects end up" — and the
    grid made every cloud small while forcing obj and recep, which sit in
    different parts of the table, to share an axis. The view range is still
    computed across every figure written here, so they remain comparable.
    """
    robust = not args.no_clip
    scale = args.workspace_scale
    ws = None if args.no_clip else workspace_extent(
        [d for g in cfg.groups for ch in g.chains for d in ch])
    colors = default_colors(len(cfg.groups))
    slugs = unique_slugs([g.label for g in cfg.groups])

    panels = []   # (label, color, kind, rows)
    for gi, group in enumerate(cfg.groups):
        runs = [d for chain in group.chains for d in chain]
        df = load_group_poses(runs, args)
        if df.empty:
            print(f"[warn] group '{group.label}' produced no rows", file=sys.stderr)
            continue
        df = apply_filters(df, args, Path(runs[0]) / "segment_pose.csv")
        for kind in [k for k in _KIND_ORDER if k in set(df["actor_kind"])]:
            panels.append((group.label, colors[gi], kind,
                           df[df["actor_kind"] == kind]))
    if not panels:
        raise SystemExit("no group produced any rows")

    xlim, ylim = _shared_limits(pd.concat([p[3] for p in panels]),
                                robust=robust, ws=ws, scale=scale)
    _announce_view(ws, scale, xlim, ylim)

    written = []
    for label, color, kind, sub in panels:
        _report_offscreen(sub, xlim, ylim, f"{label} / {kind}")
        report_panel(f"{label} / {kind}", sub)
        fig, ax = _new_panel()
        if args.hexbin:
            hb = ax.hexbin(sub["px"], sub["py"], gridsize=45, cmap="viridis",
                           mincnt=1, linewidths=0, extent=(*xlim, *ylim))
            fig.colorbar(hb, ax=ax, label="count", shrink=0.85)
        else:
            # One style for both. The synthetic rows are no longer a stand-in
            # for the whole run: `rebuild_start_rows` synthesizes only the
            # boundaries that really are an `env.reset()` draw, and for those the
            # uniform draw over `xyz_configs` IS the initial-state distribution
            # — the same quantity the recorded rows carry. Drawing them as black
            # crosses made a legitimate part of the distribution read as an
            # annotation. The synthetic share is reported by `report_panel`.
            ax.scatter(sub["px"], sub["py"], s=6, alpha=0.35, linewidths=0,
                       color=color)
        _finish_panel(ax, xlim=xlim, ylim=ylim, workspace=args.workspace)
        out = out_variant(out_base, slugs[label], kind)
        written.append(_save_panel(fig, out))
    return written


def summarize(df: pd.DataFrame) -> None:
    n_seg = df[["episode", "segment"]].drop_duplicates().shape[0]
    print(f"[pose] {len(df)} rows, {n_seg} segments, {df['env'].nunique()} envs, "
          f"episodes {int(df['episode'].min())}..{int(df['episode'].max())}, "
          f"phase={sorted(df['phase'].unique())}", file=sys.stderr)
    for kind, sub in df.groupby("actor_kind"):
        below = int((sub["pz"] < LOW_Z_THRESHOLD).sum())
        print(f"[pose] {kind:<8s} n={len(sub):<7d} "
              f"px [{sub['px'].min():+.3f},{sub['px'].max():+.3f}] "
              f"py [{sub['py'].min():+.3f},{sub['py'].max():+.3f}] "
              f"pz [{sub['pz'].min():+.3f},{sub['pz'].max():+.3f}] "
              f"below_low_z={below}", file=sys.stderr)


def main():
    p = argparse.ArgumentParser(
        "plot_segment_positions",
        description="Per-segment (per-80-step) actor position distribution from segment_pose.csv",
    )
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--run-dir", help="the run's glob dir (…/wandb/run-<ts>-<id>/glob)")
    src.add_argument("--csv", help="path to segment_pose.csv directly")
    src.add_argument("--config", help="JSON describing several groups of runs "
                                      "(see tools/plot_common.py); one column per group")
    p.add_argument("--out", default=None,
                   help="output PNG BASE path; each figure appends its own "
                        "suffix, e.g. `--out fig.png` writes fig_obj.png and "
                        "fig_recep.png (in --config mode, fig_<group>_<kind>.png). "
                        "Default: <run-dir>/segment_positions.png, or "
                        "<out_dir>/<name>_segment_positions.png in --config mode")
    p.add_argument("--no-synth", action="store_true",
                   help="for a run that recorded no phase=start rows, omit the "
                        "env.reset() boundaries instead of drawing them from the "
                        "same `xyz_configs` table the env samples. The starts "
                        "carried over from the recorded ends are still used, so "
                        "this gives measured data only — at the cost of dropping "
                        "segment 1 of every episode")
    p.add_argument("--synth-seed", type=int, default=0,
                   help="seed for the synthetic draw (reproducible plots)")
    p.add_argument("--phase", default=None, choices=["start", "end", "all"],
                   help="which side of the segment boundary. 'start' (default) is "
                        "the state each segment BEGINS from — after that boundary's "
                        "HSR/EER resets, and after env.reset() at an episode "
                        "boundary — i.e. the initial-state distribution the policy "
                        "faces. 'end' is the steady state the policy produced, "
                        "before any reset; use it to anchor workspace_aabb bounds.")
    p.add_argument("--actor-kind", default=None,
                   help="'all' (default = obj,recep), one of them, or a "
                        "comma-separated list. Each kind gets its own PNG. The "
                        "gripper is not plotted. Can also be set in the config "
                        "as `actor_kind`.")
    p.add_argument("--slot", type=int, default=None, help="keep only this logical slot")
    p.add_argument("--model", default=None, help="substring match on model_name")
    p.add_argument("--task", default=None, help="substring match on the task string")
    p.add_argument("--segment", type=int, default=None,
                   help="keep only this 1-based segment index within each episode")
    p.add_argument("--step-range", default=None, metavar="LO:HI",
                   help=f"keep only boundaries whose `total_steps` falls in this "
                        f"range, so one training segment of a resumed run can be "
                        f"plotted on its own. Default {DEFAULT_STEP_RANGE} (one "
                        f"segment's step budget); either side may be left open "
                        f"(':HI', 'LO:') and 'all' disables the filter. Can also "
                        f"be set in the config as `step_range`. What it kept is "
                        f"always reported on stderr.")
    p.add_argument("--episode-range", default=None, metavar="LO:HI")
    p.add_argument("--last-episodes", type=int, default=None,
                   help="keep only the last N episodes")
    p.add_argument("--forward-only", action="store_true",
                   help="drop reset-goal segments by joining rollout_success.csv")
    p.add_argument("--workspace-scale", type=float, default=None,
                   help="view size as a multiple of the env's own sampling "
                        "region (the extent of `xyz_configs`, read from the run's "
                        "(N, M) preset). 1.0 = exactly the region objects spawn "
                        "in; the default leaves room for actors the policy pushed "
                        "outside it.")
    p.add_argument("--no-clip", action="store_true",
                   help="plot the full coordinate range. By default the view is "
                        "bounded to median +/- 6*MAD so a few escaped actors "
                        "(px=10, pz=-1400) cannot compress every real point into "
                        "one pixel; nothing is dropped from the data or the "
                        "counts, and the number of points outside the view is "
                        "reported on stderr.")
    p.add_argument("--hexbin", action="store_true",
                   help="density hexbin instead of an episode-coloured scatter")
    p.add_argument("--workspace", default=None, metavar="X0,X1,Y0,Y1",
                   help="overlay a workspace rectangle (e.g. the workspace_aabb "
                        "bounds you are validating). The bounds are negative, so "
                        "use the '=' form or argparse reads them as a flag: "
                        "--workspace=-0.235,-0.085,-0.075,0.075")
    args = p.parse_args()

    if args.workspace:
        try:
            args.workspace = tuple(float(v) for v in args.workspace.split(","))
            if len(args.workspace) != 4:
                raise ValueError
        except ValueError:
            raise SystemExit("--workspace must be four floats: X0,X1,Y0,Y1")
    workspace = args.workspace

    if args.config:
        cfg = load_plot_config(args.config)
        # Config supplies defaults; anything given on the CLI wins.
        args.actor_kind = cfg.option("actor_kind", args.actor_kind, "all")
        args.phase = cfg.option("phase", args.phase, "start")
        args.workspace_scale = float(cfg.option("workspace_scale",
                                                args.workspace_scale, 3.0))
        args.step_range = cfg.option("step_range", args.step_range,
                                     DEFAULT_STEP_RANGE)
        out = Path(args.out) if args.out else cfg.out_dir / f"{cfg.name}_segment_positions.png"
        for path in render_groups(cfg, out, args=args):
            print(f"[ok] wrote {path}", file=sys.stderr)
        return

    if args.phase is None:
        args.phase = "start"
    if args.workspace_scale is None:
        args.workspace_scale = 3.0
    csv_path = Path(args.csv) if args.csv else Path(args.run_dir) / "segment_pose.csv"
    ws = None if args.no_clip else workspace_extent([csv_path.parent])
    df = load_pose(csv_path)
    df["synthetic"] = False
    df = ensure_start_rows(csv_path.parent, df, args)
    df = apply_filters(df, args, csv_path)
    summarize(df)

    out = Path(args.out) if args.out else csv_path.with_name("segment_positions.png")
    label = {"start": "segment-start", "end": "segment-end", "all": "segment-boundary"}[args.phase]
    for path in render(df, out, hexbin=args.hexbin, workspace=workspace,
                       label=label, robust=not args.no_clip, ws=ws,
                       scale=args.workspace_scale):
        print(f"[ok] wrote {path}", file=sys.stderr)


if __name__ == "__main__":
    main()
