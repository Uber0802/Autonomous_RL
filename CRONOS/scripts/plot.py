#!/usr/bin/env python3
"""V0.4 success-curve aggregator + plotter — CSV-native replacement for V0.1's plot.py.

V0.1 plot.py read pre-pivoted .xlsx files (sheets: ID Curve(EP), OOD Curve(EP),
ID Curve(Reset), OOD Curve(Reset)) where each sheet had one column per task.
V0.4 instead reads `eval_success.csv` directly — the canonical long-form per-run
log written by main.py's SuccessRecorder. Schema:

    episode, total_steps, total_resets, eval_kind, group, task, scene, n_envs,
    success, grasp, obj_grasped

Workflow
--------
1. Edit `plot_config.json` to list run groups (label -> [csv_paths]).
2. Run: `python scripts/plot.py --config plot_config.json`.
3. Outputs land under `<out_dir>/`:
   - `aggregated.csv` (long-form mean + std per group, eval_kind, x_axis, x_value)
   - `summary.csv` (final-value mean ± std per group × eval_kind)
   - 4 main PNGs: `<name>_<eval_kind>_<x_axis>.png`
   - 2 gap PNGs: `<name>_gap_<eval_kind>.png` (success vs grasp overlaid — the
     V0.3 seed-4 placement-collapse diagnostic from V0.4 M2)

Adding a new run = append its `eval_success.csv` path to the right group's
`csv_paths` list, then rerun. No code changes.

Requires: pandas, numpy, matplotlib. No scipy dependency (uses np.interp).
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class GroupSpec:
    label: str
    # Each entry is ONE seed's data. Can be:
    #   - a string (path to a single eval_success.csv), OR
    #   - a list of strings, treated as a resume chain that is concatenated
    #     end-to-end (parent first, child last). Duplicate (total_steps,
    #     eval_kind, group, task) keep the LATER (downstream) row's value.
    # Example chain (T320 parent + T1280 child resumed from its ep_0032):
    #   ["/path/to/T320_run/glob/eval_success.csv",
    #    "/path/to/T1280_run/glob/eval_success.csv"]
    csv_paths: List
    color: Optional[str] = None
    # Optional: only keep these CRONOS groups (e.g. ["group_A", "group_B"]).
    # If None, all CRONOS groups in the CSV are aggregated together.
    cronos_group_filter: Optional[List[str]] = None
    # Optional: only keep these tasks. Default None = all tasks.
    task_filter: Optional[List[str]] = None


@dataclass
class PlotConfig:
    out_dir: str
    name: str = "v04"
    groups: List[GroupSpec] = field(default_factory=list)
    end_steps: Optional[float] = None       # crop x-axis (total_steps)
    end_resets: Optional[float] = None      # crop x-axis (total_resets)
    smoothing_window: int = 5
    n_interp_points: int = 500
    figsize: Tuple[float, float] = (8.0, 8.0)
    eval_kinds: Tuple[str, ...] = ("in_domain", "out_of_domain")
    x_axes: Tuple[str, ...] = ("total_steps", "total_resets")
    # Default color palette (cycled through when group spec doesn't set color).
    default_colors: Tuple[str, ...] = (
        "#97bb8f", "#5a5a5a", "#02bfbf", "#c109c1",
        "#d62728", "#c1c10b", "#6666cc",
    )


def load_config(path: str) -> PlotConfig:
    raw = json.loads(Path(path).read_text())
    groups = []
    for g in raw.get("groups", []):
        groups.append(GroupSpec(
            label=g["label"],
            csv_paths=g["csv_paths"],
            color=g.get("color"),
            cronos_group_filter=g.get("cronos_group_filter"),
            task_filter=g.get("task_filter"),
        ))
    cfg = PlotConfig(
        out_dir=raw["out_dir"],
        name=raw.get("name", "v04"),
        groups=groups,
        end_steps=raw.get("end_steps"),
        end_resets=raw.get("end_resets"),
        smoothing_window=int(raw.get("smoothing_window", 5)),
        n_interp_points=int(raw.get("n_interp_points", 500)),
        figsize=tuple(raw.get("figsize", [8.0, 8.0])),
        eval_kinds=tuple(raw.get("eval_kinds", ("in_domain", "out_of_domain"))),
        x_axes=tuple(raw.get("x_axes", ("total_steps", "total_resets"))),
    )
    if not cfg.groups:
        raise ValueError("config has no groups")
    for g in cfg.groups:
        if not g.csv_paths:
            raise ValueError(f"group '{g.label}' has no csv_paths")
    return cfg


# ---------------------------------------------------------------------------
# Data layer
# ---------------------------------------------------------------------------


def load_run_csv(csv_path: str,
                 cronos_group_filter: Optional[List[str]] = None,
                 task_filter: Optional[List[str]] = None) -> pd.DataFrame:
    """Load one eval_success.csv and apply optional filters.

    Returns a long-form DataFrame with the original columns. Missing/non-CSV
    paths raise FileNotFoundError so configuration errors surface loudly.
    """
    p = Path(csv_path)
    if not p.is_file():
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    df = pd.read_csv(p)
    expected = {"episode", "total_steps", "total_resets", "eval_kind",
                "group", "task", "success", "grasp"}
    missing = expected - set(df.columns)
    if missing:
        raise ValueError(f"{csv_path} is missing required columns: {sorted(missing)}")
    if cronos_group_filter:
        df = df[df["group"].isin(cronos_group_filter)].copy()
    if task_filter:
        df = df[df["task"].isin(task_filter)].copy()
    return df


def load_seed(seed_entry, cronos_group_filter: Optional[List[str]] = None,
              task_filter: Optional[List[str]] = None) -> pd.DataFrame:
    """Load one seed's data. `seed_entry` is either a str (single CSV) or a
    list of str (resume chain: parent CSV first, downstream CSV last).

    For a chain, the CSVs are loaded in order and concatenated. Duplicate
    `(total_steps, eval_kind, group, task)` rows keep the LATER (downstream)
    value — that's the right behavior when a child run re-evals at the resume
    point and you'd rather trust the child's measurement.

    Returns one long-form DataFrame spanning the whole chain.
    """
    if isinstance(seed_entry, str):
        return load_run_csv(seed_entry, cronos_group_filter, task_filter)
    if not isinstance(seed_entry, list) or not all(isinstance(p, str) for p in seed_entry):
        raise TypeError(f"csv_paths entry must be a str or list[str], got {type(seed_entry).__name__}")
    if not seed_entry:
        raise ValueError("csv_paths chain entry is empty")
    dfs = [load_run_csv(p, cronos_group_filter, task_filter) for p in seed_entry]
    chained = pd.concat(dfs, ignore_index=True)
    # Dedupe by the eval-round key; keep the LATER segment's row when both
    # report the same (total_steps, eval_kind, group, task) — the child run's
    # measurement is the source of truth at the resume boundary.
    dedup_keys = ["total_steps", "eval_kind", "group", "task"]
    chained = (chained
               .sort_values("total_steps", kind="mergesort")
               .drop_duplicates(subset=dedup_keys, keep="last")
               .reset_index(drop=True))
    return chained


def per_run_series(df: pd.DataFrame, eval_kind: str, x_axis: str,
                   metric: str = "success") -> pd.DataFrame:
    """Reduce one run's CSV to (x_axis -> mean_over_tasks_and_groups).

    Each eval round writes one row per (group, task) pair. We:
      1. Filter to eval_kind.
      2. Group by the eval-round key (episode, x_axis) and take the mean of
         `metric` over (group, task) — this matches the "Average across tasks"
         line in V0.1 plot.py's `avg=True` mode.
      3. Return a single-column frame indexed by `x_axis` with `metric` values.

    If the run never logged this `eval_kind`, returns an empty frame.
    """
    sub = df[df["eval_kind"] == eval_kind]
    if sub.empty:
        return pd.DataFrame(columns=[x_axis, metric])
    means = (sub
             .groupby(["episode", x_axis], as_index=False)[metric]
             .mean()
             .sort_values(x_axis))
    return means[[x_axis, metric]].reset_index(drop=True)


def per_run_series_per_task(df: pd.DataFrame, eval_kind: str, x_axis: str,
                            metric: str = "success") -> pd.DataFrame:
    """Same as per_run_series but keep tasks as columns (pivot wide).

    Columns: x_axis + one column per (group, task) combination.
    """
    sub = df[df["eval_kind"] == eval_kind]
    if sub.empty:
        return pd.DataFrame(columns=[x_axis])
    sub = sub.assign(task_label=sub["group"] + " :: " + sub["task"])
    pv = (sub
          .pivot_table(index=[x_axis], columns="task_label",
                       values=metric, aggfunc="mean")
          .reset_index()
          .sort_values(x_axis))
    return pv


def interpolate_runs_to_grid(series_list: List[pd.DataFrame], x_axis: str,
                              metric: str, n_points: int,
                              x_clip: Optional[float] = None) -> Tuple[np.ndarray, np.ndarray]:
    """Interpolate each run's `metric` onto a common x grid.

    series_list: list of single-metric frames from `per_run_series`.
    Returns (x_grid, stacked) where stacked has shape (N_runs, n_points).
    Runs that are entirely empty are dropped.
    """
    usable = [s for s in series_list if not s.empty]
    if not usable:
        return np.empty((0,)), np.empty((0, n_points))
    # Prepend a (0, 0) start point so the curves all anchor at the origin
    # (matches V0.1 plot.py's `np.concatenate(([0], ...))`).
    prepped = []
    for s in usable:
        x = np.concatenate(([0.0], s[x_axis].to_numpy(dtype=float)))
        y = np.concatenate(([0.0], s[metric].to_numpy(dtype=float)))
        prepped.append((x, y))
    # Common grid: min of left edges (== 0) to min of right edges.
    x_min = 0.0
    x_max = min(x.max() for x, _ in prepped)
    if x_clip is not None:
        x_max = min(x_max, float(x_clip))
    grid = np.linspace(x_min, x_max, n_points)
    rows = []
    for x, y in prepped:
        # Clip y to finite values only (np.interp handles monotone x already).
        mask = np.isfinite(y)
        rows.append(np.interp(grid, x[mask], y[mask],
                              left=y[mask][0], right=y[mask][-1]))
    return grid, np.stack(rows, axis=0)


def moving_average(y: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return y
    y = np.asarray(y, dtype=float)
    out = np.full_like(y, np.nan, dtype=float)
    half = window // 2
    for i in range(len(y)):
        lo = max(0, i - half)
        hi = min(len(y), i + half + 1)
        seg = y[lo:hi]
        seg = seg[np.isfinite(seg)]
        if seg.size > 0:
            out[i] = seg.mean()
    return out


# ---------------------------------------------------------------------------
# Aggregation across groups → single long-form table
# ---------------------------------------------------------------------------


def aggregate_all(cfg: PlotConfig) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Build the long-form `aggregated` table + the `summary` table.

    `aggregated` columns: group, eval_kind, x_axis, x_value, metric, mean, std, n_runs
    `summary` columns:    group, eval_kind, metric, final_mean, final_std, n_runs
    """
    long_rows = []
    summary_rows = []
    metrics = ("success", "grasp")
    for spec in cfg.groups:
        # Load seeds (each may be a single CSV or a resume chain).
        try:
            dfs = [load_seed(entry, spec.cronos_group_filter, spec.task_filter)
                   for entry in spec.csv_paths]
        except (FileNotFoundError, TypeError, ValueError) as e:
            print(f"  [WARN] {spec.label}: {e}")
            continue
        for eval_kind in cfg.eval_kinds:
            for x_axis in cfg.x_axes:
                x_clip = (cfg.end_steps if x_axis == "total_steps"
                          else cfg.end_resets if x_axis == "total_resets"
                          else None)
                for metric in metrics:
                    series_list = [per_run_series(d, eval_kind, x_axis, metric) for d in dfs]
                    grid, stacked = interpolate_runs_to_grid(
                        series_list, x_axis, metric,
                        n_points=cfg.n_interp_points, x_clip=x_clip)
                    if stacked.size == 0:
                        continue
                    mean = stacked.mean(axis=0)
                    std = stacked.std(axis=0, ddof=0)
                    n_runs = stacked.shape[0]
                    for x_v, m_v, s_v in zip(grid, mean, std):
                        long_rows.append({
                            "group": spec.label, "eval_kind": eval_kind,
                            "x_axis": x_axis, "x_value": float(x_v),
                            "metric": metric, "mean": float(m_v),
                            "std": float(s_v), "n_runs": n_runs,
                        })
                    summary_rows.append({
                        "group": spec.label, "eval_kind": eval_kind,
                        "x_axis": x_axis, "metric": metric,
                        "final_x": float(grid[-1]),
                        "final_mean": float(mean[-1]),
                        "final_std": float(std[-1]),
                        "n_runs": n_runs,
                    })
    return pd.DataFrame(long_rows), pd.DataFrame(summary_rows)


# ---------------------------------------------------------------------------
# Plot layer
# ---------------------------------------------------------------------------


def _format_axis(ax, x_axis: str, y_label: str = "Success Rate") -> None:
    ax.set_ylim(0, 1.0)
    if x_axis == "total_steps":
        ax.set_xlabel("Interaction Step", fontsize=18, fontweight="bold")
    elif x_axis == "total_resets":
        ax.set_xlabel("Number of Reset", fontsize=18, fontweight="bold")
    else:
        ax.set_xlabel(x_axis, fontsize=18, fontweight="bold")
    ax.set_ylabel(y_label, fontsize=18, fontweight="bold")
    ax.minorticks_on()
    ax.grid(which="major", alpha=0.4)
    ax.grid(which="minor", alpha=0.2)
    ax.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
    ax.tick_params(axis="both", labelsize=14)


def plot_main_panel(long_df: pd.DataFrame, eval_kind: str, x_axis: str,
                     cfg: PlotConfig, out_path: Path) -> None:
    """One PNG per (eval_kind × x_axis). Overlays mean ± std envelopes for each
    config group on the same axes (one line per group). Matches V0.1 plot.py's
    `avg=True` mode."""
    sub = long_df[(long_df["eval_kind"] == eval_kind) &
                  (long_df["x_axis"] == x_axis) &
                  (long_df["metric"] == "success")]
    if sub.empty:
        return
    fig, ax = plt.subplots(figsize=cfg.figsize)
    used_labels = []
    x_max = 0.0
    for i, spec in enumerate(cfg.groups):
        g_sub = sub[sub["group"] == spec.label].sort_values("x_value")
        if g_sub.empty:
            continue
        color = spec.color or cfg.default_colors[i % len(cfg.default_colors)]
        x = g_sub["x_value"].to_numpy()
        m = moving_average(g_sub["mean"].to_numpy(), cfg.smoothing_window)
        s = moving_average(g_sub["std"].to_numpy(), cfg.smoothing_window)
        ax.plot(x, m, label=spec.label, color=color, lw=2, zorder=3)
        ax.plot(x, m - s, color=color, ls="--", lw=1, alpha=0.4, zorder=2)
        ax.plot(x, m + s, color=color, ls="--", lw=1, alpha=0.4, zorder=2)
        ax.fill_between(x, m - s, m + s, color=color, alpha=0.2, zorder=1)
        used_labels.append((color, spec.label))
        x_max = max(x_max, float(x.max()) if len(x) else 0.0)
    ax.set_xlim(0, x_max)
    _format_axis(ax, x_axis)
    handles = [Patch(facecolor=c, label=l) for c, l in used_labels]
    ax.legend(handles=handles, prop={"weight": "bold", "size": 14},
              loc="lower right", ncol=1,
              handleheight=0.7, handlelength=0.7, handletextpad=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_gap_panel(long_df: pd.DataFrame, eval_kind: str,
                    cfg: PlotConfig, out_path: Path) -> None:
    """V0.4 M2 diagnostic: success vs grasp overlaid. A persistent gap (grasp
    high, success low) is the V0.3 seed-4 placement-collapse signature."""
    sub = long_df[(long_df["eval_kind"] == eval_kind) &
                  (long_df["x_axis"] == "total_steps")]
    if sub.empty:
        return
    fig, ax = plt.subplots(figsize=cfg.figsize)
    used_labels = []
    x_max = 0.0
    for i, spec in enumerate(cfg.groups):
        color = spec.color or cfg.default_colors[i % len(cfg.default_colors)]
        for metric, ls, alpha in (("success", "-", 1.0), ("grasp", ":", 0.7)):
            g_sub = sub[(sub["group"] == spec.label) & (sub["metric"] == metric)] \
                .sort_values("x_value")
            if g_sub.empty:
                continue
            x = g_sub["x_value"].to_numpy()
            m = moving_average(g_sub["mean"].to_numpy(), cfg.smoothing_window)
            ax.plot(x, m, label=f"{spec.label} ({metric})",
                    color=color, ls=ls, lw=2, alpha=alpha, zorder=3)
            x_max = max(x_max, float(x.max()) if len(x) else 0.0)
        used_labels.append((color, spec.label))
    ax.set_xlim(0, x_max)
    _format_axis(ax, "total_steps", y_label="Success / Grasp")
    ax.legend(prop={"weight": "bold", "size": 11},
              loc="lower right", ncol=1)
    ax.set_title(f"Success vs Grasp ({eval_kind}) — gap = placement collapse early warning",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", required=True, help="Path to plot config JSON")
    args = ap.parse_args()

    cfg = load_config(args.config)
    out_dir = Path(cfg.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[plot] config={args.config}")
    print(f"[plot] out_dir={out_dir}")
    for spec in cfg.groups:
        # Each entry in csv_paths may be a string (1 CSV) or list (chain).
        seed_status = []
        for entry in spec.csv_paths:
            if isinstance(entry, str):
                seed_status.append("1" if Path(entry).is_file() else "0")
            else:
                ok = sum(Path(p).is_file() for p in entry)
                seed_status.append(f"{ok}/{len(entry)}")
        chain_note = "(chain)" if any(isinstance(e, list) for e in spec.csv_paths) else ""
        print(f"  - {spec.label:50s}  seeds={len(spec.csv_paths)}  segs/seed=[{','.join(seed_status)}]  {chain_note}")

    print("[plot] aggregating...")
    long_df, summary_df = aggregate_all(cfg)
    long_df.to_csv(out_dir / "aggregated.csv", index=False)
    summary_df.to_csv(out_dir / "summary.csv", index=False)
    print(f"[plot] wrote aggregated.csv ({len(long_df)} rows)")
    print(f"[plot] wrote summary.csv ({len(summary_df)} rows)")

    # Pretty-print final values per group / eval_kind.
    if not summary_df.empty:
        print("\n[final values @ rightmost eval]")
        view = (summary_df[summary_df["x_axis"] == "total_steps"]
                .pivot_table(index=["group", "eval_kind"],
                             columns="metric",
                             values=["final_mean", "final_std", "n_runs"]))
        with pd.option_context("display.width", 200, "display.precision", 3):
            print(view)

    print("\n[plot] plotting...")
    for eval_kind in cfg.eval_kinds:
        for x_axis in cfg.x_axes:
            png = out_dir / f"{cfg.name}_{eval_kind}_{x_axis}.png"
            plot_main_panel(long_df, eval_kind, x_axis, cfg, png)
            print(f"  wrote {png.name}")
        gap_png = out_dir / f"{cfg.name}_gap_{eval_kind}.png"
        plot_gap_panel(long_df, eval_kind, cfg, gap_png)
        print(f"  wrote {gap_png.name}")

    print("[plot] done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
