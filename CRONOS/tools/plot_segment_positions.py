"""Per-segment (per-80-step) object position distribution, from `segment_pose.csv`.

`segment_pose.csv` has one row per (episode, segment, phase, env, actor), written
at every `task_len` boundary — 80 steps by default — so "per-80" is the file's
native granularity.

A boundary is not one instant, which is what the `phase` column records:

    start   the state the segment BEGINS from — after that boundary's HSR
            respawn and EER `reset_robot()`, and after the full `env.reset()`
            at an episode boundary.
            This is the initial-state distribution the forward policy faces,
            and what `--backward-goal` (perturbation) is meant to widen.
    end     the steady state the policy produced, before any of those resets.
            Anchor `workspace_aabb` bounds from this one.

`--phase` defaults to `start`. They are not interchangeable: `--reset-robot` is
on by default in every reset mode, so the gripper always differs between them,
and HSR respawns move the objects. `reset_robot()` itself leaves the objects
where they are.

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

Points that land close together are merged (`--density`, `--bin-size`): the
spawn lattice stacks thousands of poses on one xy, and a plain scatter draws
that stack as one dot the size of a lone escaped object. `emphasis` (default)
keeps the old scatter and adds a larger, darker marker on every cell holding at
least `--dense-min` points, so the figure reads as before and only the stacks
(spawn / reset sites) stand out; `size` draws one marker per cell with its area
proportional to the count; `shade` fills the cell from light to dark on a log
scale; `scatter` is the old unmerged look, coloured by episode for a single run.

`--color-by item` (objects only, one colour per object x step range) and
`--color-by scene` (per scene: one figure per object and receptacle plus the
whole scene, one colour per step range) take the actor order from the scene
config's `obj:` / `recep:` lists — see `scene_actor_table`.

`--step-range` takes several comma-separated ranges in `--config` mode
(`0:163840,163841:327680`): one figure per range, all on one view and one
count scale, with `_steps<LO>-<HI>` in the filename. One count scale is shared by every figure
an invocation writes, like the view range. There are no px / py axis labels.

`--per-task` writes one figure per (task, kind) instead — only the task's own
object and receptacle, matched by model name — so with `--step-range` each
task's distribution over one stretch of training can be read on its own.

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
from plot_common import (NoData, default_colors, legend_pt,  # noqa: E402
                         load_plot_config, out_variant, read_run_config,
                         read_table, slugify, unique_slugs, warn)

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


def split_step_ranges(value) -> list:
    """`0:163840,163841:327680` -> `["0:163840", "163841:327680"]`.

    Each piece is validated by `parse_step_range`. A list is also accepted from
    the config. Several ranges are drawn as separate figures on ONE shared view
    and count scale, which is what makes the stretches comparable.
    """
    if value is None:
        value = DEFAULT_STEP_RANGE
    parts = value if isinstance(value, list) else str(value).split(",")
    parts = [str(v).strip() for v in parts if str(v).strip()]
    for part in parts:
        parse_step_range(part)
    return parts or [DEFAULT_STEP_RANGE]


def step_range_tag(value: str) -> str:
    """A step range as a filename fragment: `0:163840` -> `steps0-163840`."""
    rng = parse_step_range(value)
    if rng is None:
        return "steps-all"
    fmt = lambda v: "" if not np.isfinite(v) else f"{v:g}"
    return f"steps{fmt(rng[0])}-{fmt(rng[1])}"


def load_pose(csv_path: Path, *, required: bool = True) -> pd.DataFrame:
    """Read one `segment_pose.csv`.

    `required=False` is the `--config` path: every "nothing here" case raises
    `NoData` for the group loop to warn about and skip, so one run recorded with
    `--no-record-segment-pose` does not cost the other groups their figures.
    """
    if not required:
        df = read_table(csv_path, what="segment_pose.csv",
                        required_cols=("episode", "segment", "env",
                                       "actor_kind", "px", "py", "pz"))
    elif not csv_path.exists():
        raise FileNotFoundError(
            f"{csv_path} not found. It is written only when --record-segment-pose "
            f"is on (it is on by default; --no-record-segment-pose disables it)."
        )
    else:
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


def reset_count(run_dir: Path, *, required: bool = True):
    """The run's total reset count — one per per-env fresh pose draw.

    `hard_reset_count` advances by `num_envs` per episode (every env is
    re-randomized by `env.reset()`) and `soft_reset_count` by the number of envs
    HSR respawned, so the sum is exactly how many independent draws from
    `xyz_configs` the run made. Read from `counters.json`, falling back to the
    last `total_resets` in `rollout_success.csv`.

    `required=False` returns None instead of exiting, for the `--config` path
    where a run that cannot be reconstructed is skipped rather than fatal.
    """
    counters = Path(run_dir) / "counters.json"
    if counters.exists():
        try:
            return int(json.loads(counters.read_text())["total_resets"])
        except Exception:
            pass
    roll = Path(run_dir) / "rollout_success.csv"
    if roll.exists():
        try:
            r = pd.read_csv(roll, usecols=["total_resets"])
        except (ValueError, pd.errors.EmptyDataError):
            r = pd.DataFrame()               # empty, or predates the column
        if len(r):
            total = pd.to_numeric(r["total_resets"], errors="coerce").max()
            if pd.notna(total):
                return int(total)
    if required:
        raise SystemExit(f"{run_dir}: cannot determine the reset count "
                         f"(no counters.json, no rollout_success.csv)")
    return None


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

    Between the `phase="end"` record and the next segment's first step the only
    thing that moves an object is `--reset-unsuitable` (HSR respawns the flagged
    envs to a fresh draw). `--reset-robot` (EER) re-homes the robot only; the
    objects stay where the segment left them, so it does not break the
    carry-over. LSR is irrelevant too: `set_backward_goals` swaps the goal, not
    the poses.

    Returns `(exact, reason_if_not)`.
    """
    if rc.get("reset_unsuitable"):
        return False, "reset_unsuitable=True — HSR respawns flagged envs at the boundary"
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


def ensure_start_rows(run_dir: Path, df: pd.DataFrame, args, *,
                      required: bool = True) -> pd.DataFrame:
    """Append rebuilt `phase=start` rows when `df` has none and they are wanted.

    Shared by the single-run and `--config` paths. It used to live only in the
    group loader, so `--run-dir --phase start` on a pre-`phase` run failed with
    "no rows with phase='start'" instead of rebuilding them.
    """
    if args.phase not in ("start", "all"):
        return df
    if (df["phase"] == "start").any():
        return df
    try:
        extra = rebuild_start_rows(run_dir, df, args)
    except SystemExit as e:
        if required:
            raise
        warn(f"{run_dir.name}: start rows could not be rebuilt ({e})")
        return df
    if extra.empty and args.no_synth:
        print(f"[warn] {run_dir.name}: no phase=start rows, nothing could be "
              f"carried over, and --no-synth given", file=sys.stderr)
    elif extra.empty:
        n_resets = reset_count(run_dir, required=required)
        if n_resets is None:
            # --config path: no counters.json and no usable rollout CSV, so the
            # number of fresh draws is unknown and nothing can be synthesized.
            # The run keeps whatever real rows it has instead of taking the
            # whole figure set down with it.
            warn(f"{run_dir.name}: no phase=start rows and the reset count is "
                 f"unknown — cannot rebuild them, keeping the recorded rows")
            return df
        try:
            extra = synth_start_poses(run_dir, n_resets, args.synth_seed,
                                      segment=-1)
        except SystemExit as e:
            if required:
                raise
            warn(f"{run_dir.name}: start rows could not be synthesized ({e})")
            return df
    return pd.concat([df, extra], ignore_index=True) if not extra.empty else df


def load_group_poses(run_dirs, args, *, label: str = "") -> pd.DataFrame:
    """Load one group's runs, rebuilding `start` rows where they are missing.

    A run with no usable `segment_pose.csv` — absent, empty, or predating a
    column — is named on stderr and skipped; the group is built from whatever
    is left. `render_groups` drops the group only when nothing is.
    """
    frames = []
    for run_dir in run_dirs:
        run_dir = Path(run_dir)
        try:
            df = load_pose(run_dir / "segment_pose.csv", required=False)
        except NoData as e:
            warn(f"group '{label}': {e}" if label else str(e))
            continue
        df["synthetic"] = False
        df = ensure_start_rows(run_dir, df, args, required=False)
        if getattr(args, "color_by", "none") in ("item", "scene"):
            df = annotate_scene_items(df, run_dir)
        frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def apply_filters(df: pd.DataFrame, args, csv_path: Path, *,
                  required: bool = True) -> pd.DataFrame:
    """Apply every CLI filter in turn.

    `required=False` returns an empty frame instead of exiting when a filter
    leaves nothing — the `--config` path, where one group filtering itself out
    must not take the other groups' figures with it.
    """
    def _empty(msg: str) -> pd.DataFrame:
        if required:
            raise SystemExit(msg)
        warn(msg)
        return df.iloc[0:0]

    if args.phase != "all":
        sel = df[df["phase"] == args.phase]
        if sel.empty:
            avail = sorted(df["phase"].unique())
            return _empty(
                f"no rows with phase={args.phase!r}; present: {avail}. The run may "
                f"have used --segment-pose-phase to record only one side.")
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
        return _empty("no rows left after filtering")
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
    # No px / py axis labels either: every figure is the same top-down table
    # view, so the labels only repeat what the caption says. Ticks stay, so
    # coordinates can still be read off.
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(alpha=0.25)
    if workspace:
        ax.legend(loc="upper right", fontsize=legend_pt(-2))


_DENSITY_MODES = ("emphasis", "size", "shade", "scatter")
DEFAULT_DENSITY = "emphasis"
# Merge radius in metres, per mode. `size` / `emphasis` draw a marker per cell,
# so a fine grid still reads; `shade` fills the cell itself, and 5 mm cells are
# specks at the default x3 workspace view.
DEFAULT_BIN_SIZE = {"emphasis": 0.005, "size": 0.005, "shade": 0.01,
                    "scatter": 0.005}
# `emphasis`: a cell needs at least this many points to be drawn enlarged. Below
# it the plain scatter already shows the points faithfully. `None` = automatic:
# DENSE_FRACTION of the figure's points, at least DENSE_FLOOR. A fixed count does
# not fit both a whole-run figure and a per-task one holding 1/16 of its points.
# On Q2 (8192 points/figure, 5 mm cells) the automatic 50 marks ~15 obj and ~30
# recep cells — the spawn / reset sites — out of 1300 and 400 occupied.
DEFAULT_DENSE_MIN = None
DENSE_FRACTION = 0.006
DENSE_FLOOR = 5


def resolve_dense_min(dense_min, n_points: int) -> int:
    if dense_min is not None:
        return max(2, int(dense_min))
    return max(DENSE_FLOOR, int(np.ceil(DENSE_FRACTION * n_points)))


def bin_points(px, py, bin_size: float) -> pd.DataFrame:
    """Merge neighbouring points into one entry per `bin_size` grid cell.

    Returns one row per occupied cell: `x`, `y` (the centroid of the points in
    it, not the cell centre, so a tight cluster is drawn where it really is)
    and `count`. The spawn lattice puts thousands of points on the exact same
    xy, and a plain scatter draws them as one dot the same size as a lone
    escaped object — the count is what the figure has to show.
    """
    d = pd.DataFrame({"x": np.asarray(px, dtype=float),
                      "y": np.asarray(py, dtype=float)})
    d["ix"] = np.floor(d["x"] / bin_size).astype(np.int64)
    d["iy"] = np.floor(d["y"] / bin_size).astype(np.int64)
    return (d.groupby(["ix", "iy"])
             .agg(x=("x", "mean"), y=("y", "mean"), count=("x", "size"))
             .reset_index(drop=True))


def _light_cmap(color):
    """Near-white -> `color`, so a group keeps its hue in shade mode."""
    from matplotlib.colors import LinearSegmentedColormap, to_rgb
    return LinearSegmentedColormap.from_list("shade", [(0.93, 0.93, 0.93),
                                                       to_rgb(color)])


def _emphasis_style(color, c_max: int, dense_min: int):
    """count -> (marker area, RGBA) for `emphasis` mode.

    Area grows with sqrt(count): a stack of 800 must stand out without covering
    the workspace. Colour darkens from `color` towards black and turns opaque
    on a log scale between `dense_min` and the densest cell, so "denser" reads
    as both bigger and darker.
    """
    from matplotlib.colors import to_rgb
    base = np.array(to_rgb(color))
    span = max(np.log(max(c_max, dense_min + 1) / dense_min), 1e-9)

    def style(counts):
        counts = np.asarray(counts, dtype=float)
        t = np.clip(np.log(counts / dense_min) / span, 0.0, 1.0)
        sizes = np.minimum(220.0, 6.0 * np.sqrt(counts))
        rgb = base[None, :] * (1.0 - 0.55 * t[:, None])
        alpha = 0.6 + 0.35 * t
        return sizes, np.column_stack([rgb, alpha])
    return style


def emphasis_legend(ax, *, color, c_max: int, dense_min: int,
                    bin_size: float) -> None:
    """The `emphasis` size / shade key, bottom left, as a separate artist so a
    second (colour) legend can sit next to it."""
    style = _emphasis_style(color, c_max, dense_min)
    refs = sorted({dense_min,
                   max(dense_min, int(round(np.sqrt(dense_min * c_max)))),
                   max(dense_min, c_max)})
    handles = []
    for v in refs:
        sv, cv = style(np.array([v]))
        handles.append(ax.scatter([], [], s=sv, c=cv, linewidths=0.4,
                                  edgecolors="white"))
    ax.add_artist(ax.legend(handles, [f"{v}" for v in refs],
                            title=f"points / {bin_size * 1000:g} mm",
                            loc="lower left", fontsize=legend_pt(-3),
                            title_fontsize=legend_pt(-3),
                            labelspacing=1.3, borderpad=0.8, framealpha=0.8))


def cell_count_max(frames, *, xlim, ylim, bin_size: float) -> int:
    """Largest per-cell count over several figures' rows, inside the view.

    Passed to `_draw_cloud` as `count_max` so every figure one invocation
    writes shares one size / shade scale — otherwise each is normalised to its
    own densest cell, and a task with 8 points in its fullest cell draws that
    cell as large as one with 80.
    """
    best = 1
    for sub in frames:
        cells = bin_points(sub["px"], sub["py"], bin_size)
        cells = cells[cells["x"].between(*xlim) & cells["y"].between(*ylim)]
        if len(cells):
            best = max(best, int(cells["count"].max()))
    return best


def _draw_cloud(fig, ax, sub: pd.DataFrame, *, xlim, ylim, density: str,
                bin_size: float, hexbin: bool, color="tab:blue",
                episode_range=None, count_max=None,
                dense_min=DEFAULT_DENSE_MIN, count_legend: bool = True) -> None:
    """The distribution itself, in one of the density encodings.

    emphasis every point exactly as `scatter` draws it, plus one enlarged,
             darker marker on each `bin_size` cell holding >= `dense_min`
             points — the overall look is unchanged and only the stacks (the
             spawn / reset sites an untouched object never left) stand out
    size     one marker per `bin_size` cell, AREA proportional to the count
    shade    one filled cell per `bin_size` cell, shade from light to `color`
             on a log scale (counts on the lattice sites are 1000x the rest)
    scatter  every point, as before: coloured by episode when `episode_range`
             is given, else in `color`
    """
    if hexbin:
        hb = ax.hexbin(sub["px"], sub["py"], gridsize=45, cmap="viridis",
                       mincnt=1, linewidths=0, extent=(*xlim, *ylim))
        fig.colorbar(hb, ax=ax, label="count", shrink=0.85)
        return
    if density == "scatter":
        if episode_range is not None:
            ep_lo, ep_hi = episode_range
            sc = ax.scatter(sub["px"], sub["py"], c=sub["episode"], cmap="viridis",
                            s=5, alpha=0.45, linewidths=0,
                            vmin=ep_lo, vmax=max(ep_hi, ep_lo + 1))
            if ep_hi > ep_lo:
                fig.colorbar(sc, ax=ax, label="episode", shrink=0.85)
        else:
            ax.scatter(sub["px"], sub["py"], s=6, alpha=0.35, linewidths=0,
                       color=color)
        return

    from matplotlib.colors import LogNorm
    cells = bin_points(sub["px"], sub["py"], bin_size)
    # Only the cells inside the view: an escaped object 200 m away must not set
    # the top of the scale for the cells that are actually drawn.
    cells = cells[cells["x"].between(*xlim) & cells["y"].between(*ylim)]
    if cells.empty:
        return
    c_max = int(count_max or cells["count"].max())
    if density == "emphasis":
        dense_min = resolve_dense_min(dense_min, len(sub))
        ax.scatter(sub["px"], sub["py"], s=6, alpha=0.35, linewidths=0,
                   color=color, zorder=2)
        dense = cells[cells["count"] >= dense_min]
        if dense.empty:
            return
        style = _emphasis_style(color, c_max, dense_min)
        # Densest last, so a big stack is never hidden under a smaller one.
        dense = dense.sort_values("count")
        s_, c_ = style(dense["count"].to_numpy())
        ax.scatter(dense["x"], dense["y"], s=s_, c=c_, linewidths=0.4,
                   edgecolors="white", zorder=3)
        if count_legend:
            emphasis_legend(ax, color=color, c_max=c_max, dense_min=dense_min,
                            bin_size=bin_size)
        return
    if density == "size":
        # Area, not radius, is proportional to the count: that is what the eye
        # compares. The smallest marker stays visible.
        s_max = 260.0
        sizes = np.maximum(4.0, s_max * cells["count"] / c_max)
        ax.scatter(cells["x"], cells["y"], s=sizes, color=color, alpha=0.55,
                   linewidths=0.4, edgecolors="white")
        refs = sorted({1, max(1, c_max // 10), c_max})
        handles = [ax.scatter([], [], s=max(4.0, s_max * v / c_max), color=color,
                              alpha=0.55, linewidths=0.4, edgecolors="white")
                   for v in refs]
        ax.add_artist(ax.legend(handles, [str(v) for v in refs], title="count",
                                loc="lower left", fontsize=legend_pt(-3),
                                title_fontsize=legend_pt(-3),
                                labelspacing=1.2, borderpad=0.8, framealpha=0.8))
    else:
        x_edges = np.arange(np.floor(xlim[0] / bin_size),
                            np.ceil(xlim[1] / bin_size) + 1) * bin_size
        y_edges = np.arange(np.floor(ylim[0] / bin_size),
                            np.ceil(ylim[1] / bin_size) + 1) * bin_size
        h, _, _ = np.histogram2d(sub["px"], sub["py"], bins=(x_edges, y_edges))
        h = np.ma.masked_equal(h.T, 0)
        mesh = ax.pcolormesh(x_edges, y_edges, h, cmap=_light_cmap(color),
                             norm=LogNorm(vmin=1, vmax=max(2, c_max)))
        fig.colorbar(mesh, ax=ax, label="count", shrink=0.85)


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


def _model_core(model_name: str) -> str:
    """`007_ketchup bottle_1` -> `ketchup bottle`, `001_plate_simpler` -> `plate`."""
    import re
    s = re.sub(r"^\d+_", "", str(model_name))
    s = re.sub(r"_(\d+|simpler)$", "", s)
    return s.replace("_", " ").strip().lower()


def task_actor_rows(sub: pd.DataFrame, task: str) -> pd.DataFrame:
    """The rows of `sub` that are the task's OWN object / receptacle.

    `segment_pose.csv` records every slot for every env, so a task's rows also
    hold the distractor object and the other receptacle — which the task never
    touches, and which would make every task's figure look alike. The actor is
    matched by name: the model's core name (`ketchup bottle`, `plate`) must be
    a phrase of the task string (`put ketchup bottle on yellow_plate`), on the
    object side of " on " for `obj` rows and the receptacle side for `recep`.

    Returns `sub` unchanged, with a warning, when no slot matches — better an
    unfiltered figure than a missing one.
    """
    text = str(task).replace("_", " ").lower()
    obj_part, _, recep_part = text.removeprefix("put ").partition(" on ")
    side = {"obj": f" {obj_part} ", "recep": f" {recep_part} "}
    cores = sub["model_name"].map(_model_core)
    hay = sub["actor_kind"].map(side).fillna("")
    keep = np.array([f" {c} " in h or (c and c in h) for c, h in zip(cores, hay)])
    if not keep.any():
        warn(f"task {task!r}: no model_name matches the task string — keeping "
             f"every slot (models: {sorted(sub['model_name'].unique())})")
        return sub
    return sub[keep]


def split_by_task(df: pd.DataFrame, *, task_actor_only: bool = True):
    """`[(task, rows)]` for `--per-task`, in a stable (sorted) task order.

    Rows without a task — synthetic `phase=start` draws, which stand for a
    fresh env.reset() and belong to no task — cannot be attributed and are
    dropped with a note.
    """
    tasks = df["task"].fillna("").astype(str)
    untasked = int((tasks == "").sum())
    if untasked:
        print(f"[pose] --per-task: {untasked} rows carry no task (synthetic "
              f"start draws) and are left out; --phase end uses recorded rows "
              f"only", file=sys.stderr)
    out = []
    for task in sorted(t for t in tasks.unique() if t):
        rows = df[tasks == task]
        out.append((task, task_actor_rows(rows, task) if task_actor_only else rows))
    return out


def render(df: pd.DataFrame, out_base: Path, *, hexbin: bool, workspace,
           label: str = "", robust: bool = True, ws=None,
           scale: float = 3.0, density: str = DEFAULT_DENSITY,
           bin_size: float = DEFAULT_BIN_SIZE[DEFAULT_DENSITY], per_task: bool = False,
           task_actor_only: bool = True,
           dense_min=DEFAULT_DENSE_MIN) -> list:
    """One PNG per actor kind: `<out_base stem>_obj.png`, `..._recep.png`.

    `per_task` writes one PNG per (task, kind) instead, into
    `<out_base stem>_per_task/`."""
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

    if per_task:
        panels = [(t, kind, rows[rows["actor_kind"] == kind])
                  for t, rows in split_by_task(df, task_actor_only=task_actor_only)
                  for kind in kinds]
        task_dir = out_base.with_name(f"{out_base.stem}_per_task") / out_base.name
        slugs = unique_slugs(sorted({t for t, _, _ in panels}))
        c_max = cell_count_max([p[2] for p in panels], xlim=xlim, ylim=ylim,
                               bin_size=bin_size)
        written = []
        for task, kind, sub in panels:
            if sub.empty:
                continue
            _report_offscreen(sub, xlim, ylim, f"{task} / {kind}")
            report_panel(f"{task} / {kind}", sub)
            fig, ax = _new_panel()
            _draw_cloud(fig, ax, sub, xlim=xlim, ylim=ylim, density=density,
                        bin_size=bin_size, hexbin=hexbin, count_max=c_max,
                        dense_min=dense_min)
            _finish_panel(ax, xlim=xlim, ylim=ylim, workspace=workspace)
            written.append(_save_panel(fig, out_variant(task_dir, slugs[task], kind)))
        if not written:
            raise SystemExit("--per-task: no task has rows to plot")
        return written

    c_max = cell_count_max([df[df["actor_kind"] == k] for k in kinds],
                           xlim=xlim, ylim=ylim, bin_size=bin_size)
    written = []
    for kind in kinds:
        sub = df[df["actor_kind"] == kind]
        _report_offscreen(sub, xlim, ylim, kind)
        fig, ax = _new_panel()
        _draw_cloud(fig, ax, sub, xlim=xlim, ylim=ylim, density=density,
                    bin_size=bin_size, hexbin=hexbin, episode_range=(ep_lo, ep_hi),
                    count_max=c_max, dense_min=dense_min)
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


# ---------------------------------------------------------------------------
# --color-by item / scene: which scene and which of its actors each row is,
# from the scene config
# ---------------------------------------------------------------------------

_CRONOS_ROOT = Path(__file__).resolve().parents[1]
# The tables `envs/bridge_multi.py::_prep_init` indexes: YAML `obj: [7, 2]` /
# `recep: [1, 2]` are 1-based into their key order, and the keys are what
# `segment_pose.csv` records as `model_name`.
_ASSET_DIR = _CRONOS_ROOT.parent / "ManiSkill" / "mani_skill" / "assets" / "carrot"
_MODEL_DB = {"obj": _ASSET_DIR / "more_carrot" / "model_db.json",
             "recep": _ASSET_DIR / "more_plate" / "model_db.json"}


def _read_scene_groups(path: Path) -> list:
    """The `groups:` list of a scene config: [{name, num_envs, obj, recep}, ...].

    Uses PyYAML when it is installed. The plot environment does not otherwise
    need it (`scripts/requirements_plot.txt`), so without it a line parser
    reads just those four keys, which is all the actor order needs.
    """
    import re
    text = path.read_text()
    try:
        import yaml
    except ImportError:
        yaml = None
    if yaml is not None:
        return (yaml.safe_load(text) or {}).get("groups") or []
    groups, inside = [], False
    for line in text.splitlines():
        s = line.split("#", 1)[0].rstrip()
        if not s.strip():
            continue
        if re.match(r"^groups\s*:", s):
            inside = True
            continue
        if inside and re.match(r"^\S", s):
            inside = False
        if not inside:
            continue
        m = re.match(r"^\s*-\s*name\s*:\s*[\"']?([^\"']*)[\"']?\s*$", s)
        if m:
            groups.append({"name": m.group(1).strip()})
            continue
        m = re.match(r"^\s*(num_envs|obj|recep)\s*:\s*(.+)$", s)
        if m and groups:
            key, val = m.groups()
            nums = [int(v) for v in re.findall(r"-?\d+", val)]
            groups[-1][key] = nums[0] if key == "num_envs" else nums
    return groups


def scene_config_path(run_dir: Path):
    """The run's scene config: its own `experiment_config.yaml` snapshot, else
    `run_config.json`'s `config_path` as it is in this checkout now."""
    snap = Path(run_dir) / "experiment_config.yaml"
    if snap.exists():
        return snap, True
    rc = read_run_config(run_dir) or {}
    raw = rc.get("config_path")
    if not raw:
        return None, False
    for cand in (Path(raw), _CRONOS_ROOT / raw):
        if cand.exists():
            return cand, False
    return None, False


def _model_index_maps() -> dict:
    """kind -> {model_name: 1-based index in its model table}, or None for a
    kind whose table is not on disk (ManiSkill assets not downloaded)."""
    out = {}
    for kind, db in _MODEL_DB.items():
        out[kind] = ({name: i for i, name in enumerate(json.loads(db.read_text()), 1)}
                     if db.exists() else None)
    return out


def _model_number(name: str):
    """`007_ketchup bottle_1` -> 7. Every key of both shipped tables starts
    with its own 1-based position, which is the fallback when a table is
    missing."""
    import re
    m = re.match(r"^(\d+)_", str(name))
    return int(m.group(1)) if m else None


def scene_actor_table(run_dir: Path):
    """`(env, actor_kind, model_idx) -> (scene, item)` from the scene config.

    `scene` is the YAML group name; `item` is the actor's 1-based position in
    that group's `obj:` / `recep:` list — "object 1" of group_A is its first
    `obj:` entry, whatever slot the env put it in. `model_idx` is the value
    written in that list. Env ranges follow the groups' `num_envs` in file
    order, as `envs/config.py::get_group_starts` lays them out. Returns None
    (with a warning) when the config cannot be found or read.
    """
    path, is_snapshot = scene_config_path(run_dir)
    if path is None:
        rc = read_run_config(run_dir) or {}
        warn(f"{Path(run_dir).name}: scene config not found — no "
             f"experiment_config.yaml in the run dir, and config_path="
             f"{rc.get('config_path')!r} exists neither as given nor under "
             f"{_CRONOS_ROOT}")
        return None
    rows, start = [], 0
    for gi, g in enumerate(_read_scene_groups(path)):
        n = int(g.get("num_envs") or 0)
        if n <= 0 or not g.get("obj"):
            warn(f"{path}: group {g.get('name')!r} lacks num_envs / obj — "
                 f"cannot place its envs")
            return None
        scene = str(g.get("name") or f"scene{gi}")
        for kind in ("obj", "recep"):
            for item, idx in enumerate(g.get(kind) or [], start=1):
                rows += [(env, kind, int(idx), scene, item)
                         for env in range(start, start + n)]
        start += n
    if not rows:
        warn(f"{path}: no `groups:` found")
        return None
    rc = read_run_config(run_dir) or {}
    if rc.get("num_envs") and int(rc["num_envs"]) != start:
        warn(f"{path}: groups cover {start} envs, the run had {rc['num_envs']}")
    note = "" if is_snapshot else (" (the file as it is now — the run left no "
                                   "experiment_config.yaml snapshot)")
    print(f"[scene] {Path(run_dir).name}: scenes and actor order from {path}{note}",
          file=sys.stderr)
    return pd.DataFrame(rows, columns=["env", "actor_kind", "model_idx",
                                       "scene", "item"])


def env_groups_from_rollout(run_dir: Path):
    """env -> YAML group name, from the run's own `rollout_success.csv`, which
    records the group of every env. The fallback when the config is gone."""
    path = Path(run_dir) / "rollout_success.csv"
    try:
        r = pd.read_csv(path, usecols=["env_idx", "group"])
    except (FileNotFoundError, ValueError, pd.errors.EmptyDataError):
        return None
    r = r.dropna().drop_duplicates()
    if r.empty or r["env_idx"].duplicated().any():
        return None
    return r.rename(columns={"env_idx": "env", "group": "scene"})


def annotate_scene_items(df: pd.DataFrame, run_dir: Path) -> pd.DataFrame:
    """Add `scene` and `item` to the obj / recep rows.

    Preferred source is the scene config (`scene_actor_table`). When it cannot
    be read, the scene comes from `rollout_success.csv` and the item order from
    the slot index — which is the config order for the shipped presets, but is
    not checked against it. Whatever is still unknown is labelled `unknown`,
    and every fallback is announced on stderr.
    """
    name = Path(run_dir).name
    maps = _model_index_maps()
    df = df.copy()
    idx = pd.Series(np.nan, index=df.index)
    for kind, table in maps.items():
        sel = df["actor_kind"] == kind
        if table is None:
            warn(f"{name}: {_MODEL_DB[kind]} not found — matching {kind} models "
                 f"to the config by their number prefix")
            idx[sel] = df.loc[sel, "model_name"].map(_model_number)
        else:
            idx[sel] = df.loc[sel, "model_name"].map(table)
    df["model_idx"] = idx

    table = scene_actor_table(run_dir)
    if table is not None:
        out = df.merge(table, on=["env", "actor_kind", "model_idx"], how="left")
    else:
        out = df.assign(scene=np.nan, item=np.nan)

    groups = env_groups_from_rollout(run_dir)
    miss_scene = out["scene"].isna() & out["actor_kind"].isin(_KIND_ORDER)
    if miss_scene.any():
        if groups is not None:
            out = out.merge(groups.rename(columns={"scene": "_scene_rs"}),
                            on="env", how="left")
            out["scene"] = out["scene"].fillna(out.pop("_scene_rs"))
            warn(f"{name}: {int(miss_scene.sum())} rows took their scene from "
                 f"rollout_success.csv and their object order from the slot index "
                 f"(not checked against the scene config)")
        else:
            warn(f"{name}: {int(miss_scene.sum())} rows have no scene — neither "
                 f"the scene config nor rollout_success.csv names it; they are "
                 f"drawn as scene 'unknown'")
    out["scene"] = out["scene"].fillna("unknown")
    out["item"] = out["item"].fillna(out["slot"] + 1).astype(int)
    return out.drop(columns=["model_idx"])


def step_range_label(value: str) -> str:
    rng = parse_step_range(value)
    if rng is None:
        return "all steps"
    fmt = lambda v: "" if not np.isfinite(v) else f"{v:,.0f}"
    return f"steps {fmt(rng[0])}–{fmt(rng[1])}"


def render_groups_by_item(cfg, out_base: Path, *, args) -> list:
    """`--color-by item`: objects only, one colour per (object, step range).

    One figure per group holds every step range and every object of the
    scene: with two ranges and two objects that is four colours, ordered
    object 1 / range 1, object 1 / range 2, object 2 / range 1, ... Object
    numbers come from the scene config (`scene_actor_table`), so "object 1" is the
    first `obj:` entry of each group even when groups are pooled. `--per-task`
    gives each task its own figure; a task moves one object, so it shows that
    object's colours only.
    """
    import copy
    if args.hexbin or args.density == "shade":
        raise SystemExit("--color-by item needs a point density mode "
                         "(emphasis, size or scatter), not shade / hexbin")
    ranges = split_step_ranges(args.step_range)
    n_r = len(ranges)
    r_labels = [step_range_label(r) for r in ranges]
    palette = default_colors(10)
    color_of = lambda item, ri: palette[((item - 1) * n_r + ri) % len(palette)]
    slugs = unique_slugs([g.label for g in cfg.groups])

    figs = []   # (group label, task or None, [(item, range idx, rows)])
    for group in cfg.groups:
        runs = [d for chain in group.chains for d in chain]
        loaded = load_group_poses(runs, args, label=group.label)
        if loaded.empty:
            warn(f"group '{group.label}' produced no rows")
            continue
        loaded = loaded[loaded["actor_kind"] == "obj"]
        per_range = []
        for rng_text in ranges:
            r_args = copy.copy(args)
            r_args.step_range = rng_text
            per_range.append(apply_filters(loaded, r_args,
                                           Path(runs[0]) / "segment_pose.csv",
                                           required=False))
        if all(d.empty for d in per_range):
            warn(f"group '{group.label}': every row was filtered out")
            continue
        if args.per_task:
            by_task = [dict(split_by_task(d, task_actor_only=not args.all_slots))
                       if len(d) else {} for d in per_range]
            for task in sorted(set().union(*by_task)):
                layers = [(int(item), ri, rows)
                          for ri, tasks in enumerate(by_task) if task in tasks
                          for item, rows in tasks[task].groupby("item")]
                figs.append((group.label, task, sorted(layers, key=lambda l: l[:2])))
            continue
        layers = [(int(item), ri, rows)
                  for ri, d in enumerate(per_range) if len(d)
                  for item, rows in d.groupby("item")]
        figs.append((group.label, None, sorted(layers, key=lambda l: l[:2])))
    if not figs:
        raise SystemExit("[pose] --color-by item: no group produced obj rows")

    every = [rows for _, _, layers in figs for _, _, rows in layers]
    robust = not args.no_clip
    ws = None if args.no_clip else workspace_extent(
        [d for g in cfg.groups for ch in g.chains for d in ch])
    xlim, ylim = _shared_limits(pd.concat(every), robust=robust, ws=ws,
                                scale=args.workspace_scale)
    _announce_view(ws, args.workspace_scale, xlim, ylim)
    c_max = cell_count_max(every, xlim=xlim, ylim=ylim, bin_size=args.bin_size)
    task_slugs = unique_slugs(sorted({t for _, t, _ in figs if t}))

    from matplotlib.lines import Line2D
    written = []
    for label, task, layers in figs:
        # One threshold per figure, from ALL its points, so "enlarged" means the
        # same count for every colour in it and the same as in the single-colour
        # figure of the same data.
        # Per step range, since one range is what the single-colour figure holds.
        n_ranges = len({ri for _, ri, _ in layers})
        dense_min = resolve_dense_min(
            args.dense_min, sum(len(rows) for _, _, rows in layers) // n_ranges)
        fig, ax = _new_panel()
        handles = []
        for item, ri, rows in layers:
            tag = " / ".join(x for x in (label, r_labels[ri], task, f"obj{item}") if x)
            _report_offscreen(rows, xlim, ylim, tag)
            report_panel(tag, rows)
            color = color_of(item, ri)
            _draw_cloud(fig, ax, rows, xlim=xlim, ylim=ylim, density=args.density,
                        bin_size=args.bin_size, hexbin=False, color=color,
                        count_max=c_max, dense_min=dense_min, count_legend=False)
            names = sorted({_model_core(m) for m in rows["model_name"]})
            what = names[0] if len(names) == 1 else f"object {item}"
            handles.append(Line2D([], [], ls="", marker="o", ms=6, color=color,
                                  label=f"{what} · {r_labels[ri]}"))
        if args.density == "emphasis":
            emphasis_legend(ax, color="0.45", c_max=c_max, dense_min=dense_min,
                            bin_size=args.bin_size)
        # Above the axes: four colours spread over the whole cloud, so no
        # corner inside it is free. The axes' own legend (not add_artist), so
        # `bbox_inches="tight"` keeps it in the saved image.
        ax.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 1.01),
                  ncol=2, fontsize=legend_pt(-2), frameon=False,
                  handletextpad=0.2, columnspacing=1.0)
        _finish_panel(ax, xlim=xlim, ylim=ylim, workspace=args.workspace)
        if task:
            task_dir = out_base.with_name(
                f"{out_base.stem}_{slugs[label]}_by_item_per_task")
            out = out_variant(task_dir / out_base.name, task_slugs[task], "obj")
        else:
            out = out_variant(out_base, slugs[label], "obj_by_item")
        written.append(_save_panel(fig, out))
    return written


def render_scenes(cfg, out_base: Path, *, args) -> list:
    """`--color-by scene`: per scene, one figure per actor plus one of them all.

    For every scene (YAML group) of every config group, in the scene config's
    order:

        1..N      object 1..N          (`obj:` list)
        N+1..N+M  receptacle 1..M      (`recep:` list)
        last      every actor of the scene together

    so a 2x2 scene gives 5 figures. Each figure has one colour per step range
    (`--step-range A,B` -> two colours), which makes it a before/after of the
    same actor. View and count scale are shared by every figure written.
    """
    import copy
    if args.hexbin or args.density == "shade":
        raise SystemExit("--color-by scene needs a point density mode "
                         "(emphasis, size or scatter), not shade / hexbin")
    ranges = split_step_ranges(args.step_range)
    r_labels = [step_range_label(r) for r in ranges]
    palette = default_colors(max(2, len(ranges)))
    slugs = unique_slugs([g.label for g in cfg.groups])

    figs = []   # (group label, scene, fig no., name, [(range idx, rows)])
    for group in cfg.groups:
        runs = [d for chain in group.chains for d in chain]
        loaded = load_group_poses(runs, args, label=group.label)
        if loaded.empty:
            warn(f"group '{group.label}' produced no rows")
            continue
        loaded = loaded[loaded["actor_kind"].isin(_KIND_ORDER)]
        per_range = []
        for rng_text in ranges:
            r_args = copy.copy(args)
            r_args.step_range = rng_text
            r_args.actor_kind = "all"
            per_range.append(apply_filters(loaded, r_args,
                                           Path(runs[0]) / "segment_pose.csv",
                                           required=False))
        scenes = list(dict.fromkeys(loaded["scene"]))   # config order
        for scene in scenes:
            actors = (loaded[loaded["scene"] == scene][["actor_kind", "item", "model_name"]]
                      .drop_duplicates())
            actors = actors.assign(k=actors["actor_kind"].map(_KIND_ORDER.index))
            n = 0
            for (kind, item), names in actors.groupby(["k", "item"], sort=True):
                kind = _KIND_ORDER[kind]
                n += 1
                cores = sorted({_model_core(m) for m in names["model_name"]})
                name = f"{'obj' if kind == 'obj' else 'recep'}{item}-{'+'.join(cores)}"
                layers = [(ri, d[(d["scene"] == scene) & (d["actor_kind"] == kind)
                                 & (d["item"] == item)])
                          for ri, d in enumerate(per_range) if len(d)]
                figs.append((group.label, scene, n, name,
                             [(ri, rows) for ri, rows in layers if len(rows)]))
            layers = [(ri, d[d["scene"] == scene])
                      for ri, d in enumerate(per_range) if len(d)]
            figs.append((group.label, scene, n + 1, "all",
                         [(ri, rows) for ri, rows in layers if len(rows)]))
    figs = [f for f in figs if f[4]]
    if not figs:
        raise SystemExit("[pose] --color-by scene: nothing to plot")

    every = [rows for *_, layers in figs for _, rows in layers]
    ws = None if args.no_clip else workspace_extent(
        [d for g in cfg.groups for ch in g.chains for d in ch])
    xlim, ylim = _shared_limits(pd.concat(every), robust=not args.no_clip,
                                ws=ws, scale=args.workspace_scale)
    _announce_view(ws, args.workspace_scale, xlim, ylim)
    c_max = cell_count_max(every, xlim=xlim, ylim=ylim, bin_size=args.bin_size)

    from matplotlib.lines import Line2D
    written = []
    for label, scene, n, name, layers in figs:
        dense_min = resolve_dense_min(
            args.dense_min, sum(len(r) for _, r in layers) // len(layers))
        fig, ax = _new_panel()
        handles = []
        what = "all actors" if name == "all" else name.split("-", 1)[1]
        for ri, rows in layers:
            tag = f"{label} / {scene} / {name} / {r_labels[ri]}"
            _report_offscreen(rows, xlim, ylim, tag)
            report_panel(tag, rows)
            _draw_cloud(fig, ax, rows, xlim=xlim, ylim=ylim, density=args.density,
                        bin_size=args.bin_size, hexbin=False,
                        color=palette[ri % len(palette)], count_max=c_max,
                        dense_min=dense_min, count_legend=False)
            handles.append(Line2D([], [], ls="", marker="o", ms=6,
                                  color=palette[ri % len(palette)],
                                  label=f"{what} · {r_labels[ri]}"))
        if args.density == "emphasis":
            emphasis_legend(ax, color="0.45", c_max=c_max, dense_min=dense_min,
                            bin_size=args.bin_size)
        ax.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 1.01),
                  ncol=len(handles), fontsize=legend_pt(-2), frameon=False,
                  handletextpad=0.2, columnspacing=1.0)
        _finish_panel(ax, xlim=xlim, ylim=ylim, workspace=args.workspace)
        scene_dir = out_base.with_name(f"{out_base.stem}_{slugs[label]}_by_scene")
        out = scene_dir / f"{slugify(scene)}_{n}_{slugify(name)}.png"
        written.append(_save_panel(fig, out))
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

    import copy
    ranges = split_step_ranges(args.step_range)
    # Only tagged when there is something to tell apart, so a single-range run
    # keeps its old filenames.
    range_tags = ({r: step_range_tag(r) for r in ranges} if len(ranges) > 1
                  else {ranges[0]: None})

    panels = []   # (label, color, kind, rows, task or None, range tag or None)
    for gi, group in enumerate(cfg.groups):
        runs = [d for chain in group.chains for d in chain]
        loaded = load_group_poses(runs, args, label=group.label)
        if loaded.empty:
            warn(f"group '{group.label}' produced no rows")
            continue
        for rng_text, rtag in range_tags.items():
            r_args = copy.copy(args)
            r_args.step_range = rng_text
            df = apply_filters(loaded, r_args, Path(runs[0]) / "segment_pose.csv",
                               required=False)
            if df.empty:
                warn(f"group '{group.label}' step range {rng_text}: every row "
                     f"was filtered out")
                continue
            kinds = [k for k in _KIND_ORDER if k in set(df["actor_kind"])]
            if args.per_task:
                for task, rows in split_by_task(df, task_actor_only=not args.all_slots):
                    for kind in kinds:
                        sub = rows[rows["actor_kind"] == kind]
                        if len(sub):
                            panels.append((group.label, colors[gi], kind, sub,
                                           task, rtag))
                continue
            for kind in kinds:
                panels.append((group.label, colors[gi], kind,
                               df[df["actor_kind"] == kind], None, rtag))
    if not panels:
        raise SystemExit(
            "[pose] no group produced any rows — nothing to plot.\n"
            "  The [warn] lines above name the groups. Check that each `runs`\n"
            "  entry points at a run's glob/ dir containing segment_pose.csv\n"
            "  (a run launched with --no-record-segment-pose has none), and that\n"
            "  --phase / --actor-kind / --step-range keep something.")

    xlim, ylim = _shared_limits(pd.concat([p[3] for p in panels]),
                                robust=robust, ws=ws, scale=scale)
    _announce_view(ws, scale, xlim, ylim)

    task_slugs = unique_slugs(sorted({p[4] for p in panels if p[4]}))
    c_max = cell_count_max([p[3] for p in panels], xlim=xlim, ylim=ylim,
                           bin_size=args.bin_size)
    written = []
    for label, color, kind, sub, task, rtag in panels:
        tag = " / ".join(x for x in (label, rtag, task, kind) if x)
        _report_offscreen(sub, xlim, ylim, tag)
        report_panel(tag, sub)
        fig, ax = _new_panel()
        # One style for recorded and synthetic rows. The synthetic rows are no
        # longer a stand-in for the whole run: `rebuild_start_rows` synthesizes
        # only the boundaries that really are an `env.reset()` draw, and for
        # those the uniform draw over `xyz_configs` IS the initial-state
        # distribution — the same quantity the recorded rows carry. The
        # synthetic share is reported by `report_panel`.
        _draw_cloud(fig, ax, sub, xlim=xlim, ylim=ylim, density=args.density,
                    bin_size=args.bin_size, hexbin=args.hexbin, color=color,
                    count_max=c_max, dense_min=args.dense_min)
        _finish_panel(ax, xlim=xlim, ylim=ylim, workspace=args.workspace)
        parts = [slugs[label]] + ([rtag] if rtag else [])
        if task:
            # A directory per (group, range): 16 tasks x 2 kinds would bury
            # the group-level figures if they all shared one folder.
            task_dir = out_base.with_name(f"{out_base.stem}_{'_'.join(parts)}_per_task")
            out = out_variant(task_dir / out_base.name, task_slugs[task], kind)
        else:
            out = out_variant(out_base, *parts, kind)
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
                   help="viridis hexbin (overrides --density)")
    p.add_argument("--density", default=None, choices=list(_DENSITY_MODES),
                   help=f"how points that land close together are drawn. "
                        f"'emphasis' (default): every point as in 'scatter', "
                        f"plus an enlarged, darker marker on each --bin-size "
                        f"cell holding >= --dense-min points. 'size': "
                        f"neighbours within --bin-size merge into one marker "
                        f"whose area is the count. 'shade': one "
                        f"cell per --bin-size square, darker = more points (log "
                        f"scale). 'scatter': every point, unmerged (the old look). "
                        f"Config key `density`.")
    p.add_argument("--bin-size", type=float, default=None,
                   help=f"merge radius for --density size/shade, in metres "
                        f"(default: size {DEFAULT_BIN_SIZE['size']}, shade "
                        f"{DEFAULT_BIN_SIZE['shade']}). Config key `bin_size`.")
    p.add_argument("--per-task", action="store_true", default=None,
                   help="one PNG per (task, actor kind) instead of one per kind, "
                        "into a `..._per_task/` directory; combine with "
                        "--step-range to see each task's distribution over one "
                        "stretch of training. Only the task's own object / "
                        "receptacle is drawn (see --all-slots). Rows without a "
                        "task (synthetic start draws) are left out, so on a run "
                        "recorded before the phase split use --phase end. "
                        "Config key `per_task`.")
    p.add_argument("--dense-min", type=int, default=None,
                   help=f"--density emphasis: a --bin-size cell needs at least "
                        f"this many points to be enlarged. Default: automatic, "
                        f"{DENSE_FRACTION * 100:.1f}%% of the figure's points and at "
                        f"least {DENSE_FLOOR}. Lower it to mark more stacks. "
                        f"Config key `dense_min`.")
    p.add_argument("--color-by", default=None, choices=["none", "item", "scene"],
                   help="--config mode. 'item': draw objects only, one colour per "
                        "(object, step range) — object 1 / range 1, object 1 / "
                        "range 2, object 2 / range 1, ... — with the object "
                        "order taken from the scene config's `obj:` list. One "
                        "figure per group (<...>_<group>_obj_by_item.png). "
                        "'scene': for every scene, one figure per object and "
                        "receptacle plus one of the whole scene, one colour per "
                        "step range (<...>_<group>_by_scene/). "
                        "Config key `color_by`.")
    p.add_argument("--all-slots", action="store_true",
                   help="--per-task: keep every slot of the envs running the "
                        "task, distractors included")
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
        args.density = cfg.option("density", args.density, DEFAULT_DENSITY)
        args.bin_size = float(cfg.option("bin_size", args.bin_size,
                                         DEFAULT_BIN_SIZE[args.density]))
        args.per_task = bool(cfg.option("per_task", args.per_task, False))
        args.dense_min = cfg.option("dense_min", args.dense_min, DEFAULT_DENSE_MIN)
        args.color_by = cfg.option("color_by", args.color_by, "none")
        out =Path(args.out) if args.out else cfg.out_dir / f"{cfg.name}_segment_positions.png"
        renderer = {"item": render_groups_by_item,
                    "scene": render_scenes}.get(args.color_by, render_groups)
        for path in renderer(cfg, out, args=args):
            print(f"[ok] wrote {path}", file=sys.stderr)
        return

    if args.phase is None:
        args.phase = "start"
    if args.workspace_scale is None:
        args.workspace_scale = 3.0
    args.density = args.density or DEFAULT_DENSITY
    if args.bin_size is None:
        args.bin_size = DEFAULT_BIN_SIZE[args.density]
    args.per_task = bool(args.per_task)
    if args.color_by not in (None, "none"):
        raise SystemExit("--color-by item needs --config (it colours by step "
                         "range and object across one shared figure)")
    if args.step_range and "," in str(args.step_range):
        raise SystemExit("several --step-range values are only supported with "
                         "--config, where they share one view and count scale")
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
                       scale=args.workspace_scale, density=args.density,
                       bin_size=args.bin_size, per_task=args.per_task,
                       task_actor_only=not args.all_slots,
                       dense_min=args.dense_min):
        print(f"[ok] wrote {path}", file=sys.stderr)


if __name__ == "__main__":
    main()
