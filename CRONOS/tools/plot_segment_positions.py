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


def _z_bins(z, n_bins: int = 60, ref: float = None):
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
    lo, hi = float(np.min(z)), float(np.max(z))
    if not (np.isfinite(lo) and np.isfinite(hi)):
        return np.linspace(0.0, 1.0, n_bins)
    if hi - lo < 1e-9:
        span = max(abs(hi - ref), 0.05) if ref is not None else 0.05
        half = span * 0.02
        return np.array([lo - half, hi + half])
    return np.linspace(lo, hi, n_bins)


def _shared_limits(df: pd.DataFrame, pad: float = 0.01):
    """xy limits shared by every panel of a figure.

    Two reasons, both visible without it:
      - panels become comparable; otherwise each auto-scales to its own spread
        and a tight cluster looks like a wide one.
      - a degenerate panel stays readable. With EER on, every `phase=start`
        gripper row is the identical homed pose (`initial_robot_pos` +
        `initial_qpos` are constants), and an auto-scaled axis zooms into ~2 mm
        of float noise.
    """
    return ((df["px"].min() - pad, df["px"].max() + pad),
            (df["py"].min() - pad, df["py"].max() + pad))


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
        from envs.suite import POSE_PRESETS, generate_pose_configs
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
    xyz = generate_pose_configs(**POSE_PRESETS[preset])      # (Ncfg, N+M, 3)

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
    if args.actor_kind != "all":
        df = df[df["actor_kind"] == args.actor_kind]
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


def render(df: pd.DataFrame, out_path: Path, *, hexbin: bool, workspace, title: str) -> Path:
    kinds = [k for k in _KIND_ORDER if k in set(df["actor_kind"])]
    kinds += sorted(set(df["actor_kind"]) - set(_KIND_ORDER))
    n = len(kinds)
    fig, axes = plt.subplots(2, n, figsize=(4.6 * n, 8.4), squeeze=False)

    ep_lo, ep_hi = int(df["episode"].min()), int(df["episode"].max())
    xlim, ylim = _shared_limits(df)
    zbins = _z_bins(df["pz"], ref=LOW_Z_THRESHOLD)

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
    """One column per config group: xy scatter on top, pz histogram below."""
    frames = []
    for group in cfg.groups:
        runs = [d for chain in group.chains for d in chain]
        df = load_group_poses(runs, args)
        if df.empty:
            print(f"[warn] group '{group.label}' produced no rows", file=sys.stderr)
            continue
        df = apply_filters(df, args, Path(runs[0]) / "segment_pose.csv")
        df["__group"] = group.label
        frames.append(df)
    if not frames:
        raise SystemExit("no group produced any rows")

    n = len(frames)
    fig, axes = plt.subplots(2, n, figsize=(4.8 * n, 8.6), squeeze=False)
    # A shared range makes the columns visually comparable — the whole point of
    # putting them side by side.
    allx = pd.concat(frames)
    xlim, ylim = _shared_limits(allx)
    zbins = _z_bins(allx["pz"], ref=LOW_Z_THRESHOLD)

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

    kind = args.actor_kind if args.actor_kind != "all" else "all actors"
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
    p.add_argument("--phase", default="start", choices=["start", "end", "all"],
                   help="which side of the segment boundary. 'start' (default) is "
                        "the state each segment BEGINS from — after that boundary's "
                        "HSR/EER resets, and after env.reset() at an episode "
                        "boundary — i.e. the initial-state distribution the policy "
                        "faces. 'end' is the steady state the policy produced, "
                        "before any reset; use it to anchor workspace_aabb bounds.")
    p.add_argument("--actor-kind", default="all",
                   choices=["all", "obj", "recep", "gripper"])
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
        out = Path(args.out) if args.out else cfg.out_dir / f"{cfg.name}_segment_positions.png"
        render_groups(cfg, out, args=args)
        print(f"[ok] wrote {out}", file=sys.stderr)
        return

    csv_path = Path(args.csv) if args.csv else Path(args.run_dir) / "segment_pose.csv"
    df = load_pose(csv_path)
    df["synthetic"] = False
    df = apply_filters(df, args, csv_path)
    summarize(df)

    out = Path(args.out) if args.out else csv_path.with_name("segment_positions.png")
    label = {"start": "segment-start", "end": "segment-end", "all": "segment-boundary"}[args.phase]
    render(df, out, hexbin=args.hexbin, workspace=workspace,
           title=f"{label} positions — {csv_path.parent}")
    print(f"[ok] wrote {out}", file=sys.stderr)


if __name__ == "__main__":
    main()
