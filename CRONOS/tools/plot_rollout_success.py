"""Per-segment (per-80-step) rollout success rate, straight from `rollout_success.csv`.

`rollout_success.csv` has one row per (episode, segment, env) and is written at
every `task_len` boundary — 80 steps by default — so a "per-80" curve is the
file's native granularity: no resampling, no interpolation. Each plotted point is
the mean over the `num_envs` rows of one segment.

    python tools/plot_rollout_success.py --run-dir <RUN_OUT_DIR>/wandb/run-*/glob

Why the `direction` filter defaults to `forward`
------------------------------------------------
Under a reset mode that includes LSR — `LSR`, `HSR+LSR`, `noep+LSR` — the
segments alternate between the forward task and a reset goal, and the env's
`success` predicate means something different in each (see
`doc/data_schemas.md`). Modes without LSR, bare `noep` included, log every row
as `forward`, so the default filter below costs them nothing:

    forward         success = the scheduler's task was completed
    backward        goal is "put X on table" but success still scores the
                    FORWARD pair, so it is 0 by construction
    backward_recep  the target receptacle was swapped, so success means the
                    object reached THAT receptacle

Averaging them together produces a ~50% collapse that is a pure artifact of the
alternation. `--direction forward` is therefore the default; `--direction all`
plots each direction as its own series so the alternation is visible rather than
silently folded in.

Comparing several experiments
-----------------------------
`--config plot_runs.json` plots one curve per group, aggregating that group's
series into a mean ± spread band. A series may be a **resume chain** — several
run dirs stitched into one continuous line. See `tools/plot_common.py` for the
schema.

    python tools/plot_rollout_success.py --config tools/plot_runs_example.json

Companion to `tools/plot_run_trends.py`, which plots the *eval* points and the
PPO health scalars. This one is the training-side view and needs no wandb access
— it reads only the local CSV.
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

sys.path.insert(0, str(Path(__file__).resolve().parent))
from plot_common import (X_LABEL, concat_chain, default_colors,  # noqa: E402
                         load_plot_config, new_curve_figure, plot_group_curve,
                         prepend_origin, save_curve_figure, style_curve_axes)

# Columns whose empty-string cells mean "this env did not report at this
# boundary" rather than zero. `training/metrics.py` writes "" for those.
_METRIC_COLS = ("success", "consecutive_grasp", "is_src_obj_grasped")


def load_rollout(csv_path: Path) -> pd.DataFrame:
    """Read `rollout_success.csv` and coerce the metric columns to float.

    Empty cells become NaN and are excluded from the means (rather than being
    read as 0, which would silently depress every curve).
    """
    if not csv_path.exists():
        raise FileNotFoundError(
            f"{csv_path} not found. It is written by the training rollout; an "
            f"eval-only run does not produce one."
        )
    df = pd.read_csv(csv_path)
    for col in _METRIC_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if "direction" not in df.columns:
        # Pre-LSR CSVs predate the column; everything in them is forward.
        df["direction"] = "forward"
    df["direction"] = df["direction"].fillna("forward")
    # Monotonic segment counter across episodes, for --x-axis segment. Built
    # from the (episode, segment) pairs actually present so a resumed run whose
    # episode numbering starts mid-way still yields a dense axis.
    keys = df[["episode", "segment"]].drop_duplicates().sort_values(["episode", "segment"])
    keys["seg_index"] = np.arange(1, len(keys) + 1)
    return df.merge(keys, on=["episode", "segment"], how="left")


def segment_means(df: pd.DataFrame, x_key: str, extra_group=()) -> pd.DataFrame:
    """Collapse per-env rows into one row per segment (per group, if asked).

    `n_envs` is carried through so a caller can spot segments where only part of
    the batch reported.
    """
    by = ["episode", "segment", "seg_index", "total_steps", *extra_group]
    agg = {c: "mean" for c in _METRIC_COLS if c in df.columns}
    out = df.groupby(by, dropna=False).agg(**{
        **{c: (c, "mean") for c in agg},
        "n_envs": ("env_idx", "count"),
    }).reset_index()
    x = "seg_index" if x_key == "segment" else x_key
    return out.sort_values(x)


def _x_values(frame: pd.DataFrame, x_key: str) -> np.ndarray:
    return frame["seg_index" if x_key == "segment" else x_key].to_numpy()


def _plot_series(ax, frame, x_key, col, label, color, smooth, raw_alpha=0.25):
    """Raw per-segment points plus a rolling mean over `smooth` segments."""
    if col not in frame.columns:
        return
    x, y = _x_values(frame, x_key), frame[col].to_numpy(dtype=float)
    ok = ~np.isnan(y)
    if not ok.any():
        return
    ax.plot(x[ok], y[ok], marker=".", markersize=3, linewidth=0.7,
            alpha=raw_alpha, color=color)
    if smooth > 1 and ok.sum() >= smooth:
        ma = pd.Series(y[ok]).rolling(smooth, min_periods=1).mean().to_numpy()
        ax.plot(x[ok], ma, linewidth=2.0, color=color, label=f"{label} (MA{smooth})")
    else:
        ax.plot(x[ok], y[ok], linewidth=1.4, color=color, label=label)


def render(df: pd.DataFrame, out_path: Path, *, direction: str, by: str,
           x_key: str, smooth: int, title: str) -> Path:
    if direction != "all":
        sel = df[df["direction"] == direction]
        if sel.empty:
            avail = sorted(df["direction"].unique())
            raise SystemExit(
                f"no rows with direction={direction!r}; present: {avail}. "
                f"Use --direction all to plot every direction as its own series."
            )
        df = sel

    n_panels = 1 if by == "none" else 2
    fig, axes = plt.subplots(n_panels, 1, figsize=(11, 4.2 * n_panels), squeeze=False)
    ax0 = axes[0][0]

    if direction == "all":
        # One series per direction, so the LSR alternation is explicit.
        cmap = plt.get_cmap("tab10")
        for i, d in enumerate(sorted(df["direction"].unique())):
            frame = segment_means(df[df["direction"] == d], x_key)
            _plot_series(ax0, frame, x_key, "success", f"success [{d}]", cmap(i), smooth)
        ax0.set_title("per-segment success by direction")
    else:
        frame = segment_means(df, x_key)
        _plot_series(ax0, frame, x_key, "success", "success", "tab:blue", smooth)
        _plot_series(ax0, frame, x_key, "consecutive_grasp", "grasp", "tab:orange", smooth)
        _plot_series(ax0, frame, x_key, "is_src_obj_grasped", "obj_grasped",
                     "tab:green", smooth, raw_alpha=0.12)
        # The success-vs-grasp gap is the placement-collapse diagnostic: a
        # policy that grasps reliably but never places shows a wide gap.
        ax0.set_title(f"per-segment success / grasp  (direction={direction})")

    ax0.set_ylabel("rate")
    ax0.set_ylim(-0.02, 1.02)
    ax0.grid(alpha=0.3)
    ax0.legend(loc="upper left", fontsize=8)

    if by != "none":
        ax1 = axes[1][0]
        if by not in df.columns:
            raise SystemExit(f"--by {by}: column not in the CSV "
                             f"(have: {sorted(df.columns)})")
        cats = sorted(df[by].dropna().unique())
        cmap = plt.get_cmap("tab20" if len(cats) > 10 else "tab10")
        for i, cat in enumerate(cats):
            frame = segment_means(df[df[by] == cat], x_key)
            _plot_series(ax1, frame, x_key, "success", str(cat), cmap(i % cmap.N),
                         smooth, raw_alpha=0.15)
        ax1.set_ylabel("success rate")
        ax1.set_ylim(-0.02, 1.02)
        ax1.grid(alpha=0.3)
        ax1.legend(loc="upper left", fontsize=7, ncol=2)
        ax1.set_title(f"per-segment success by {by}")

    axes[-1][0].set_xlabel(X_LABEL[x_key])
    fig.suptitle(title, fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return out_path


def render_groups(cfg, out_path: Path, *, direction: str, x_key: str,
                  smooth: int, metric: str) -> Path:
    """One curve per config group; mean ± spread across that group's series."""
    fig, ax = new_curve_figure()
    colors = default_colors(len(cfg.groups))
    x_max = 0.0
    drawn = 0

    for gi, group in enumerate(cfg.groups):
        series = []
        for chain in group.chains:
            frames = []
            for run_dir in chain:
                df = load_rollout(Path(run_dir) / "rollout_success.csv")
                if direction != "all":
                    df = df[df["direction"] == direction]
                if len(df):
                    frames.append(segment_means(df, "total_steps"))
            merged = concat_chain(frames, "total_steps")
            if len(merged):
                series.append(merged)
        if not series:
            print(f"[warn] group '{group.label}' produced no rows", file=sys.stderr)
            continue

        # Align the series on their shared x values. Same-config seeds land on
        # identical `total_steps`, so an inner align is exact; anything a series
        # is missing simply does not contribute to that x.
        wide = pd.concat(
            [s.set_index("total_steps")[metric].rename(k) for k, s in enumerate(series)],
            axis=1,
        ).sort_index()
        if smooth > 1:
            wide = wide.rolling(smooth, min_periods=1).mean()
        mean = wide.mean(axis=1)
        std = wide.std(axis=1)
        x = mean.index.to_numpy()

        n = wide.shape[1]
        print(f"[group] {group.label:<34s} {n} series, {len(x)} x-points, "
              f"final {metric}={mean.to_numpy()[-1]:.4f}", file=sys.stderr)

        # The first boundary is 80 steps in, so without this the curve starts
        # hanging in mid-air; every panel of both success-rate tools begins at
        # the untrained policy's (0, 0) instead.
        x, mean_y, std_y = prepend_origin(x, mean.to_numpy(), std.to_numpy())
        plot_group_curve(ax, x, mean_y, std_y, color=colors[gi],
                         label=group.label, n_series=n)
        x_max = max(x_max, float(x.max()) if len(x) else 0.0)
        drawn += 1

    if not drawn:
        # Without this the figure saves as an empty pair of axes, which reads as
        # "the policy scored zero" rather than "no input was found".
        raise SystemExit(
            f"[rollout] no group produced a curve — nothing to plot.\n"
            f"  The [warn] lines above name the groups. Check that each `runs`\n"
            f"  entry points at a run's glob/ dir containing rollout_success.csv\n"
            f"  (an eval-only run has none), and that direction={direction!r} and\n"
            f"  metric={metric!r} exist in it.")

    style_curve_axes(ax, x_axis="total_steps", y_label=metric, x_max=x_max)
    # No title: direction / smoothing / band meaning are settings, not findings,
    # and they are already on stderr with the per-group series counts.
    smooth_note = f", MA{smooth}" if smooth > 1 else ""
    print(f"[rollout] per-segment {metric} (direction={direction}{smooth_note}); "
          f"band = ±1 std across series", file=sys.stderr)
    return save_curve_figure(fig, out_path)


def summarize(df: pd.DataFrame, direction: str) -> None:
    sel = df if direction == "all" else df[df["direction"] == direction]
    n_seg = sel[["episode", "segment"]].drop_duplicates().shape[0]
    print(f"[rollout] {len(sel)} rows, {n_seg} segments, "
          f"{sel['env_idx'].nunique()} envs, episodes "
          f"{int(sel['episode'].min())}..{int(sel['episode'].max())}", file=sys.stderr)
    counts = df["direction"].value_counts().to_dict()
    print(f"[rollout] rows by direction: {counts}", file=sys.stderr)
    for col in _METRIC_COLS:
        if col in sel.columns and sel[col].notna().any():
            print(f"[rollout] mean {col:<20s} = {sel[col].mean():.4f}", file=sys.stderr)


def main():
    p = argparse.ArgumentParser(
        "plot_rollout_success",
        description="Per-segment (per-80-step) rollout success rate from rollout_success.csv",
    )
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--run-dir", help="the run's glob dir (…/wandb/run-<ts>-<id>/glob)")
    src.add_argument("--csv", help="path to rollout_success.csv directly")
    src.add_argument("--config", help="JSON describing several groups of runs "
                                      "(see tools/plot_common.py); one curve per group")
    p.add_argument("--out", default=None,
                   help="output PNG (default: <run-dir>/rollout_success.png, or "
                        "<out_dir>/<name>_rollout_success.png in --config mode)")
    p.add_argument("--metric", default=None, choices=list(_METRIC_COLS),
                   help="--config mode: which column to curve")
    p.add_argument("--direction", default=None,
                   choices=["forward", "backward", "backward_recep", "all"],
                   help="which segments to plot. Default 'forward' — reset segments "
                        "score `success` against a different goal, see the module "
                        "docstring. 'all' draws one series per direction.")
    p.add_argument("--by", default=None, choices=["none", "task", "group", "obj", "recep"],
                   help="add a second panel broken down by this column")
    p.add_argument("--x-axis", default="total_steps",
                   choices=["total_steps", "segment", "episode"])
    p.add_argument("--smooth", type=int, default=None,
                   help="rolling-mean window in segments (1 disables)")
    args = p.parse_args()

    if args.config:
        cfg = load_plot_config(args.config)
        args.direction = cfg.option("direction", args.direction, "forward")
        args.by = cfg.option("by", args.by, "none")
        args.metric = cfg.option("metric", args.metric, "success")
        args.smooth = int(cfg.option("smooth", args.smooth, 5))
        out = Path(args.out) if args.out else cfg.out_dir / f"{cfg.name}_rollout_success.png"
        render_groups(cfg, out, direction=args.direction, x_key="total_steps",
                      smooth=args.smooth, metric=args.metric)
        print(f"[ok] wrote {out}", file=sys.stderr)
        return

    args.direction = args.direction or "forward"
    args.by = args.by or "none"
    args.metric = args.metric or "success"
    args.smooth = 5 if args.smooth is None else args.smooth
    csv_path = Path(args.csv) if args.csv else Path(args.run_dir) / "rollout_success.csv"
    df = load_rollout(csv_path)
    summarize(df, args.direction)

    out = Path(args.out) if args.out else csv_path.with_name("rollout_success.png")
    render(df, out, direction=args.direction, by=args.by, x_key=args.x_axis,
           smooth=args.smooth, title=f"rollout success — {csv_path.parent}")
    print(f"[ok] wrote {out}", file=sys.stderr)


if __name__ == "__main__":
    main()
