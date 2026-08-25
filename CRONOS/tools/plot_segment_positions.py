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
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# `envs/unsuitable.py::LowZDetector.z_threshold` — the height below which HSR
# treats an actor as fallen. Drawn as a reference line, not applied as a filter.
LOW_Z_THRESHOLD = 0.7
_KIND_ORDER = ("obj", "recep", "gripper")


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
    for col, kind in enumerate(kinds):
        sub = df[df["actor_kind"] == kind]
        ax = axes[0][col]

        if hexbin:
            hb = ax.hexbin(sub["px"], sub["py"], gridsize=45, cmap="viridis",
                           mincnt=1, linewidths=0)
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

        ax.set_title(f"{kind}  (n={len(sub)})")
        ax.set_xlabel("px")
        ax.set_ylabel("py" if col == 0 else "")
        ax.set_aspect("equal", adjustable="datalim")
        ax.grid(alpha=0.25)

        axz = axes[1][col]
        axz.hist(sub["pz"], bins=60, color="tab:blue", alpha=0.8)
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
    p.add_argument("--out", default=None,
                   help="output PNG (default: <run-dir>/segment_positions.png)")
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

    csv_path = Path(args.csv) if args.csv else Path(args.run_dir) / "segment_pose.csv"
    df = apply_filters(load_pose(csv_path), args, csv_path)
    summarize(df)

    workspace = None
    if args.workspace:
        try:
            workspace = tuple(float(v) for v in args.workspace.split(","))
            if len(workspace) != 4:
                raise ValueError
        except ValueError:
            raise SystemExit("--workspace must be four floats: X0,X1,Y0,Y1")

    out = Path(args.out) if args.out else csv_path.with_name("segment_positions.png")
    label = {"start": "segment-start", "end": "segment-end", "all": "segment-boundary"}[args.phase]
    render(df, out, hexbin=args.hexbin, workspace=workspace,
           title=f"{label} positions — {csv_path.parent}")
    print(f"[ok] wrote {out}", file=sys.stderr)


if __name__ == "__main__":
    main()
