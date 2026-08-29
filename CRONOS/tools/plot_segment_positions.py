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

    python tools/plot_segment_positions.py --run-dir <RUN_OUT_DIR>/wandb/run-*/glob

Layout: one column per `actor_kind` present (`obj`, `recep`, `gripper`).

    row 1   xy scatter, coloured by episode, so drift over training is visible
    row 2   pz histogram, with the `low_z` detector threshold marked — points
            left of it are what HSR would flag as fallen off the table

Notes on the data
-----------------
- Hidden slots (a YAML group declaring fewer objects than the batch-wide N)
  write NaN, deliberately, so the row count per segment stays fixed. They are
  dropped here.
- `actor_kind` covers **every** object and receptacle slot, not just the pair
  the current task selected — so distractor objects the policy was supposed to
  leave alone are included. Use `--slot` / `--model` to narrow.
- There is no `group` column. Filter by `--model` (the per-env model name) or
  `--task` instead; under fan-out, slot 0 is a different model in different envs.
- `--forward-only` joins against `rollout_success.csv` on (episode, segment,
  env) and keeps only forward segments. Worth using under LSR / noep, where half
  the segment ends are reset-goal states and would otherwise be mixed in.
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
from plot_common import (concat_chain, default_colors, load_plot_config,  # noqa: E402
                         read_run_config)

# `envs/unsuitable.py::LowZDetector.z_threshold` — the height below which HSR
# treats an actor as fallen. Drawn as a reference line, not applied as a filter.
LOW_Z_THRESHOLD = 0.7
_KIND_ORDER = ("obj", "recep", "gripper")

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


_VALID_KINDS = ("obj", "recep", "gripper")


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
    bad = [p for p in parts if p not in _VALID_KINDS]
    if bad:
        raise SystemExit(f"unknown actor kind(s) {bad}; choose from "
                         f"{list(_VALID_KINDS)} or 'all', comma-separated")
    # De-duplicate while keeping _KIND_ORDER for a stable panel order.
    return [k for k in _VALID_KINDS if k in parts]


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


def _z_bins(z, n_bins: int = 60, ref: float = None, robust: bool = True,
            ws=None, scale: float = 3.0):
    """Histogram edges for `pz` that survive a constant column.

    `phase=start` poses come straight out of `xyz_configs`, whose z is the
    preset's fixed `slot_heights` (1.0 for objects, 0.95 for receptacles). So for
    a single `actor_kind` the start-side `pz` is genuinely one value, and
    `np.linspace(c, c, n)` returns n identical edges — every bin has zero width
    and the histogram renders completely empty, which reads as missing data
    rather than as zero spread.

    `ref` is the reference line the caller will draw (the `low_z` threshold).
    It is not binned, but it stretches the axis, so the degenerate case sizes its
    single bar against that span — otherwise the bar is drawn correctly and is
    still one pixel wide next to a threshold 0.3 m away.
    """
    z = np.asarray(z, dtype=float)
    if ws is not None:
        lo, hi = _scale_box(*ws[2], scale)
    else:
        lo, hi = _robust_range(z, pad=0.0, enabled=robust)
    if not (np.isfinite(lo) and np.isfinite(hi)):
        return np.linspace(0.0, 1.0, n_bins)
    if ref is not None:
        # Always show the threshold line's neighbourhood: "how many are below
        # low_z" is the question this panel exists to answer.
        lo, hi = min(lo, ref), max(hi, ref)
    if hi - lo < 1e-9:
        span = max(abs(hi - ref), 0.05) if ref is not None else 0.05
        half = span * 0.02
        return np.array([lo - half, hi + half])
    return np.linspace(lo, hi, n_bins)


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
    """xy limits shared by every panel of a figure, robust to runaway actors.

    Shared, because otherwise each panel auto-scales to its own spread and a
    tight cluster looks like a wide one — and because a degenerate panel (with
    EER on, every `phase=start` gripper row is the identical homed pose) would
    zoom into millimetres of float noise.

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


def _report_offscreen(df: pd.DataFrame, xlim, ylim, zbins, label: str = "") -> int:
    """Count and announce points the clipped view cannot show."""
    off = (~df["px"].between(*xlim) | ~df["py"].between(*ylim)
           | ~df["pz"].between(zbins[0], zbins[-1])).sum()
    if off:
        tag = f"{label}: " if label else ""
        print(f"[pose] {tag}{off}/{len(df)} points ({off / len(df):.2%}) lie outside "
              f"the plotted range and are not drawn — px {df['px'].min():.3f}…"
              f"{df['px'].max():.3f}, py {df['py'].min():.3f}…{df['py'].max():.3f}, "
              f"pz {df['pz'].min():.3f}…{df['pz'].max():.3f}. Use --no-clip to "
              f"include them.", file=sys.stderr)
    return int(off)


def synth_start_poses(run_dir: Path, n_draws: int, seed: int = 0) -> pd.DataFrame:
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

    Rows come back tagged `phase="start"`, `synthetic=True`.
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
                "episode": -1, "segment": -1, "phase": "start", "total_steps": -1,
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


def load_group_poses(run_dirs, args) -> pd.DataFrame:
    """Load one group's runs, synthesizing `start` rows where they are missing."""
    frames = []
    for run_dir in run_dirs:
        run_dir = Path(run_dir)
        df = load_pose(run_dir / "segment_pose.csv")
        df["synthetic"] = False
        has_start = (df["phase"] == "start").any()
        if args.phase in ("start", "all") and not has_start:
            if args.no_synth:
                print(f"[warn] {run_dir.name}: no phase=start rows and --no-synth "
                      f"given; skipping", file=sys.stderr)
            else:
                df = pd.concat(
                    [df, synth_start_poses(run_dir, reset_count(run_dir), args.synth_seed)],
                    ignore_index=True)
        frames.append(df)
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


def render(df: pd.DataFrame, out_path: Path, *, hexbin: bool, workspace, title: str,
           robust: bool = True, ws=None, scale: float = 3.0) -> Path:
    kinds = [k for k in _KIND_ORDER if k in set(df["actor_kind"])]
    kinds += sorted(set(df["actor_kind"]) - set(_KIND_ORDER))
    n = len(kinds)
    fig, axes = plt.subplots(2, n, figsize=(4.6 * n, 8.4), squeeze=False)

    ep_lo, ep_hi = int(df["episode"].min()), int(df["episode"].max())
    xlim, ylim = _shared_limits(df, robust=robust, ws=ws, scale=scale)
    zbins = _z_bins(df["pz"], ref=LOW_Z_THRESHOLD, robust=robust, ws=ws, scale=scale)
    if ws is not None:
        print(f"[pose] view from the pose preset's workspace x{scale:g}: "
              f"px {xlim[0]:.3f}…{xlim[1]:.3f}  py {ylim[0]:.3f}…{ylim[1]:.3f}",
              file=sys.stderr)
    _report_offscreen(df, xlim, ylim, zbins)

    for col, kind in enumerate(kinds):
        sub = df[df["actor_kind"] == kind]
        ax = axes[0][col]

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

        if workspace:
            x0, x1, y0, y1 = workspace
            ax.add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False,
                                       edgecolor="crimson", linestyle="--",
                                       linewidth=1.2, label="workspace"))
            ax.legend(loc="upper right", fontsize=7)

        # A single distinct xy means the pose is pinned rather than sparsely
        # sampled — the homed gripper. Say so; a lone dot on a shared axis is
        # otherwise easy to misread as missing data.
        spread = max(sub["px"].max() - sub["px"].min(),
                     sub["py"].max() - sub["py"].min())
        pinned = "  (fixed pose)" if spread < 1e-9 else ""
        ax.set_title(f"{kind}  (n={len(sub)}){pinned}")
        ax.set_xlabel("px")
        ax.set_ylabel("py" if col == 0 else "")
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(alpha=0.25)

        axz = axes[1][col]
        axz.hist(sub["pz"], bins=zbins, color="tab:blue", alpha=0.8)
        axz.axvline(LOW_Z_THRESHOLD, color="crimson", linestyle="--", linewidth=1.2,
                    label=f"low_z = {LOW_Z_THRESHOLD}")
        below = int((sub["pz"] < LOW_Z_THRESHOLD).sum())
        axz.set_title(f"{kind} pz — {below}/{len(sub)} below threshold "
                      f"({below / max(1, len(sub)):.1%})")
        axz.set_xlabel("pz")
        axz.set_ylabel("count" if col == 0 else "")
        axz.legend(fontsize=7)
        axz.grid(alpha=0.25)

    fig.suptitle(title, fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return out_path


def render_groups(cfg, out_path: Path, *, args) -> Path:
    kinds_sel = parse_actor_kinds(args.actor_kind)
    robust = not args.no_clip
    scale = args.workspace_scale
    ws = None if args.no_clip else workspace_extent(
        [d for g in cfg.groups for ch in g.chains for d in ch])
    """One column per config group: xy scatter on top, pz histogram below."""
    frames = []
    for group in cfg.groups:
        runs = [d for chain in group.chains for d in chain]
        df = load_group_poses(runs, args)
        if df.empty:
            print(f"[warn] group '{group.label}' produced no rows", file=sys.stderr)
            continue
        df = apply_filters(df, args, Path(runs[0]) / "segment_pose.csv")
        # One panel per (group, actor_kind), never a pooled cloud: obj / recep /
        # gripper sit at different heights, so mixing them makes the xy scatter
        # ambiguous and the pz histogram meaningless.
        for kind in [k for k in _KIND_ORDER if k in set(df["actor_kind"])]:
            sub = df[df["actor_kind"] == kind].copy()
            sub["__group"] = (group.label if len(kinds_sel) == 1
                              else f"{group.label}\n{kind}")
            frames.append(sub)
    if not frames:
        raise SystemExit("no group produced any rows")

    n = len(frames)
    fig, axes = plt.subplots(2, n, figsize=(4.8 * n, 8.6), squeeze=False)
    # A shared range makes the columns visually comparable — the whole point of
    # putting them side by side.
    allx = pd.concat(frames)
    xlim, ylim = _shared_limits(allx, robust=robust, ws=ws, scale=scale)
    zbins = _z_bins(allx["pz"], ref=LOW_Z_THRESHOLD, robust=robust, ws=ws, scale=scale)
    if ws is not None:
        print(f"[pose] view from the pose preset's workspace x{scale:g}: "
              f"px {xlim[0]:.3f}…{xlim[1]:.3f}  py {ylim[0]:.3f}…{ylim[1]:.3f}",
              file=sys.stderr)
    for sub in frames:
        _report_offscreen(sub, xlim, ylim, zbins, sub["__group"].iloc[0])

    for col, sub in enumerate(frames):
        label = sub["__group"].iloc[0]
        real = sub[~sub["synthetic"]]
        synth = sub[sub["synthetic"]]
        ax = axes[0][col]
        if args.hexbin:
            hb = ax.hexbin(sub["px"], sub["py"], gridsize=45, cmap="viridis",
                           mincnt=1, linewidths=0, extent=(*xlim, *ylim))
            fig.colorbar(hb, ax=ax, label="count", shrink=0.85)
        else:
            # Real and reconstructed points are drawn distinctly and never
            # merged into one cloud: the synthetic set is a uniform draw over the
            # discrete `xyz_configs` table, so it looks nothing like a recorded
            # distribution and pooling them silently would misrepresent both.
            if len(real):
                ax.scatter(real["px"], real["py"], s=6, alpha=0.35, linewidths=0,
                           color=default_colors(n)[col], label=f"recorded ({len(real)})")
            if len(synth):
                ax.scatter(synth["px"], synth["py"], s=26, alpha=0.75, marker="x",
                           linewidths=0.9, color="black",
                           label=f"synthetic ({len(synth)})")
            if len(real) and len(synth):
                ax.legend(loc="upper right", fontsize=7)
        if args.workspace:
            x0, x1, y0, y1 = args.workspace
            ax.add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False,
                                       edgecolor="crimson", linestyle="--", linewidth=1.2))
        tag = f"  [{len(synth)}/{len(sub)} synthetic]" if len(synth) else ""
        ax.set_title(f"{label}{tag}\nn={len(sub)}", fontsize=10)
        ax.set_xlim(*xlim); ax.set_ylim(*ylim)
        ax.set_xlabel("px"); ax.set_ylabel("py" if col == 0 else "")
        ax.set_aspect("equal", adjustable="box")
        ax.grid(alpha=0.25)

        axz = axes[1][col]
        axz.hist(sub["pz"], bins=zbins, color=default_colors(n)[col], alpha=0.85)
        axz.axvline(LOW_Z_THRESHOLD, color="crimson", linestyle="--", linewidth=1.2,
                    label=f"low_z = {LOW_Z_THRESHOLD}")
        below = int((sub["pz"] < LOW_Z_THRESHOLD).sum())
        axz.set_title(f"pz — {below}/{len(sub)} below ({below / max(1, len(sub)):.1%})",
                      fontsize=10)
        axz.set_xlabel("pz"); axz.set_ylabel("count" if col == 0 else "")
        axz.legend(fontsize=7); axz.grid(alpha=0.25)

    ks = parse_actor_kinds(args.actor_kind)
    kind = "all actors" if len(ks) == len(_VALID_KINDS) else ",".join(ks)
    fig.suptitle(f"{cfg.name} — segment-{args.phase} positions ({kind})", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return out_path


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
                   help="output PNG (default: <run-dir>/segment_positions.png, or "
                        "<out_dir>/<name>_segment_positions.png in --config mode)")
    p.add_argument("--no-synth", action="store_true",
                   help="--config mode: skip runs that lack phase=start rows instead "
                        "of reconstructing their start distribution by drawing "
                        "uniformly from the same `xyz_configs` table the env samples, "
                        "as many times as the run actually reset")
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
                   help="'all' (default), one kind, or a comma-separated list "
                        "such as obj,recep. In --config mode each kind gets its "
                        "own column so its pz histogram stays meaningful. Can "
                        "also be set in the config as `actor_kind`.")
    p.add_argument("--slot", type=int, default=None, help="keep only this logical slot")
    p.add_argument("--model", default=None, help="substring match on model_name")
    p.add_argument("--task", default=None, help="substring match on the task string")
    p.add_argument("--segment", type=int, default=None,
                   help="keep only this 1-based segment index within each episode")
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
        out = Path(args.out) if args.out else cfg.out_dir / f"{cfg.name}_segment_positions.png"
        render_groups(cfg, out, args=args)
        print(f"[ok] wrote {out}", file=sys.stderr)
        return

    if args.phase is None:
        args.phase = "start"
    if args.workspace_scale is None:
        args.workspace_scale = 3.0
    csv_path = Path(args.csv) if args.csv else Path(args.run_dir) / "segment_pose.csv"
    ws = None if args.no_clip else workspace_extent([csv_path.parent])
    df = load_pose(csv_path)
    df["synthetic"] = False
    df = apply_filters(df, args, csv_path)
    summarize(df)

    out = Path(args.out) if args.out else csv_path.with_name("segment_positions.png")
    label = {"start": "segment-start", "end": "segment-end", "all": "segment-boundary"}[args.phase]
    render(df, out, hexbin=args.hexbin, workspace=workspace,
           title=f"{label} positions — {csv_path.parent}",
           robust=not args.no_clip, ws=ws, scale=args.workspace_scale)
    print(f"[ok] wrote {out}", file=sys.stderr)


if __name__ == "__main__":
    main()
