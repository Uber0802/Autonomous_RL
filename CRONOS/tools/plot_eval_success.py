#!/usr/bin/env python3
"""Success-curve aggregator + plotter — reads long-form `eval_success.csv` files.

Input CSV schema (written by `main.py`'s `SuccessRecorder`):

    episode, total_steps, total_resets, eval_kind, group, task, scene, n_envs,
    success, grasp, obj_grasped

Workflow
--------
1. Edit `plot_config.json` to list run groups (label -> [csv_paths]).
2. Run: `python tools/plot_eval_success.py --config tools/plot_config.json`.
3. Outputs land under `<out_dir>/`, all prefixed `<name>_` so configs can share
   a directory (`out_dir` empty/absent = the config file's own directory):
   - `<name>_aggregated.csv` (long-form mean + std per group, eval_kind, x_axis, x_value)
   - `<name>_summary.csv` (final-value mean ± std per group × eval_kind)
   - 4 main PNGs: `<name>_<eval_kind>_<x_axis>.png`
   - 2 gap PNGs: `<name>_gap_<eval_kind>.png` (success vs grasp overlaid)

Adding a new run = append its `eval_success.csv` path to the right group's
`csv_paths` list, then rerun. No code changes.

The figures are drawn to the parameters in `tools/plot_common.py`, shared with
`tools/plot_rollout_success.py`: the two tools measure the same rate at
different sampling points (eval rounds vs 80-step boundaries) and are read side
by side, so they use one look, one x label vocabulary and one (0, 0) anchor
rather than each inventing its own.

Requires: pandas, numpy, matplotlib.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from plot_common import (CURVE_FIGSIZE, default_colors,  # noqa: E402
                         new_curve_figure, plot_group_curve, prepend_origin,
                         resolve_out_dir, save_curve_figure, style_curve_axes)


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
    # Shared with plot_rollout_success.py; a group's `color` still overrides the
    # per-group colour, and `figsize` the panel size.
    figsize: Tuple[float, float] = CURVE_FIGSIZE
    eval_kinds: Tuple[str, ...] = ("in_domain", "out_of_domain")
    x_axes: Tuple[str, ...] = ("total_steps", "total_resets")


def _runs_to_csv_paths(entries: List) -> List:
    """Map the shared `runs` key (glob dirs) onto this script's `csv_paths`.

    `runs` is the format `tools/plot_common.py` uses, and it points at a run's
    `glob/` DIRECTORY rather than at one CSV inside it. A directory is the more
    useful unit — it holds `eval_success.csv`, `rollout_success.csv` and
    `segment_pose.csv` — so one config can drive this script and the two
    per-segment plot tools. Here each run dir simply resolves to its
    `eval_success.csv`; nesting (a resume chain) is preserved.
    """
    out = []
    for e in entries:
        if isinstance(e, str):
            out.append(str(Path(e) / "eval_success.csv"))
        elif isinstance(e, list):
            out.append([str(Path(x) / "eval_success.csv") for x in e])
        else:
            raise ValueError(
                f"`runs` entries must be a run-dir string or a list of them "
                f"(a resume chain), got {type(e).__name__}")
    return out


def load_config(path: str) -> PlotConfig:
    raw = json.loads(Path(path).read_text())
    groups = []
    for g in raw.get("groups", []):
        # `runs` (glob dirs, shared with the per-segment tools) or the original
        # `csv_paths` (direct eval_success.csv paths). Both accepted so a single
        # config file drives every plot tool; `runs` is the preferred spelling.
        if "runs" in g and "csv_paths" in g:
            raise ValueError(
                f"group '{g.get('label')}' sets both `runs` and `csv_paths`; "
                f"use one (prefer `runs`, which points at the glob dir)")
        if "runs" in g:
            csv_paths = _runs_to_csv_paths(g["runs"])
        else:
            csv_paths = g["csv_paths"]
        groups.append(GroupSpec(
            label=g["label"],
            csv_paths=csv_paths,
            color=g.get("color"),
            cronos_group_filter=g.get("cronos_group_filter"),
            task_filter=g.get("task_filter"),
        ))
    cfg = PlotConfig(
        out_dir=raw.get("out_dir") or "",
        name=raw.get("name", "v04"),
        groups=groups,
        end_steps=raw.get("end_steps"),
        end_resets=raw.get("end_resets"),
        smoothing_window=int(raw.get("smoothing_window", 5)),
        n_interp_points=int(raw.get("n_interp_points", 500)),
        figsize=tuple(raw.get("figsize", CURVE_FIGSIZE)),
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
         `metric` over (group, task) — an "average across tasks" line.
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
    # Prepend a (0, 0) start point so the curves all anchor at the origin.
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
        # Apply BOTH end_steps and end_resets crops at the row level so every
        # x_axis view sees the same eval-point subset. Without this, a run
        # cropped to end_steps on the step-axis would still contribute its
        # post-crop rows to the reset-axis view (its post-crop total_steps
        # rows still have valid total_resets), producing curves that no longer
        # look like re-scaled versions of each other.
        def _row_crop(d):
            if cfg.end_steps is not None:
                d = d[d["total_steps"] <= float(cfg.end_steps)]
            if cfg.end_resets is not None:
                d = d[d["total_resets"] <= float(cfg.end_resets)]
            return d
        dfs = [_row_crop(d) for d in dfs]
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


def _group_color(cfg: PlotConfig, index: int, palette) -> object:
    """A group's colour: its own `color` if the config set one, else the shared
    palette `plot_rollout_success.py` also draws from."""
    return cfg.groups[index].color or palette[index]


def plot_main_panel(long_df: pd.DataFrame, eval_kind: str, x_axis: str,
                     cfg: PlotConfig, out_path: Path) -> None:
    """One PNG per (eval_kind × x_axis). Overlays mean ± std envelopes for each
    config group on the same axes (one line per group)."""
    sub = long_df[(long_df["eval_kind"] == eval_kind) &
                  (long_df["x_axis"] == x_axis) &
                  (long_df["metric"] == "success")]
    if sub.empty:
        return
    fig, ax = new_curve_figure(cfg.figsize)
    palette = default_colors(len(cfg.groups))
    x_max = 0.0
    for i, spec in enumerate(cfg.groups):
        g_sub = sub[sub["group"] == spec.label].sort_values("x_value")
        if g_sub.empty:
            continue
        x = g_sub["x_value"].to_numpy()
        m = moving_average(g_sub["mean"].to_numpy(), cfg.smoothing_window)
        s = moving_average(g_sub["std"].to_numpy(), cfg.smoothing_window)
        n = int(g_sub["n_runs"].iloc[0])
        x, m, s = prepend_origin(x, m, s)
        plot_group_curve(ax, x, m, s, color=_group_color(cfg, i, palette),
                         label=spec.label, n_series=n)
        x_max = max(x_max, float(x.max()) if len(x) else 0.0)
    style_curve_axes(ax, x_axis=x_axis, y_label="success", x_max=x_max)
    ax.set_title(f"eval success ({eval_kind}, MA{cfg.smoothing_window}); "
                 f"band = ±1 std across series")
    save_curve_figure(fig, out_path, suptitle=cfg.name)


def plot_gap_panel(long_df: pd.DataFrame, eval_kind: str,
                    cfg: PlotConfig, out_path: Path) -> None:
    """Success vs grasp overlaid. A persistent gap (grasp high, success low)
    is the placement-collapse signature."""
    sub = long_df[(long_df["eval_kind"] == eval_kind) &
                  (long_df["x_axis"] == "total_steps")]
    if sub.empty:
        return
    fig, ax = new_curve_figure(cfg.figsize)
    palette = default_colors(len(cfg.groups))
    x_max = 0.0
    for i, spec in enumerate(cfg.groups):
        color = _group_color(cfg, i, palette)
        for metric, ls in (("success", "-"), ("grasp", ":")):
            g_sub = sub[(sub["group"] == spec.label) & (sub["metric"] == metric)] \
                .sort_values("x_value")
            if g_sub.empty:
                continue
            x = g_sub["x_value"].to_numpy()
            m = moving_average(g_sub["mean"].to_numpy(), cfg.smoothing_window)
            x, m = prepend_origin(x, m)
            # No band here: two overlaid metrics per group already crowd the
            # panel, and the gap between the lines is what this figure is for.
            ax.plot(x, m, label=f"{spec.label} ({metric})", color=color, ls=ls,
                    linewidth=2.0)
            x_max = max(x_max, float(x.max()) if len(x) else 0.0)
    style_curve_axes(ax, x_axis="total_steps", y_label="success / grasp",
                     x_max=x_max)
    ax.set_title(f"eval success vs grasp ({eval_kind}, MA{cfg.smoothing_window}); "
                 f"gap = placement-collapse early warning")
    save_curve_figure(fig, out_path, suptitle=cfg.name)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", required=True, help="Path to plot config JSON")
    args = ap.parse_args()

    cfg = load_config(args.config)
    out_dir = resolve_out_dir(cfg.out_dir, args.config)
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
    # `<name>_` prefixed like every PNG below: without it two configs sharing an
    # out_dir silently overwrite each other's aggregates.
    agg_csv = out_dir / f"{cfg.name}_aggregated.csv"
    sum_csv = out_dir / f"{cfg.name}_summary.csv"
    long_df.to_csv(agg_csv, index=False)
    summary_df.to_csv(sum_csv, index=False)
    print(f"[plot] wrote {agg_csv.name} ({len(long_df)} rows)")
    print(f"[plot] wrote {sum_csv.name} ({len(summary_df)} rows)")

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
