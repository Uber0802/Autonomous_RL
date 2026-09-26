#!/usr/bin/env python3
"""Success-curve aggregator + plotter — reads long-form `eval_success.csv` files.

Input CSV schema (written by `main.py`'s `SuccessRecorder`):

    episode, total_steps, total_resets, eval_kind, group, task, scene, n_envs,
    success, grasp, obj_grasped

Workflow
--------
1. Edit `plot_config.json` to list run groups (label -> [csv_paths]).
2. Run: `python plotting/plot_eval_success.py --config plotting/configs/plot_config.json`.
3. Outputs land under `<out_dir>/`, all prefixed `<name>_` so configs can share
   a directory (`out_dir` empty/absent = the config file's own directory):
   - `<name>_aggregated.csv` (long-form mean + std per group, eval_kind, x_axis, x_value)
   - `<name>_summary.csv` (final-value mean ± std per group × eval_kind)
   - 4 main PNGs: `<name>_<eval_kind>_<x_axis>.png`
   - 2 gap PNGs: `<name>_gap_<eval_kind>.png` (success vs grasp overlaid)

Adding a new run = append its `eval_success.csv` path to the right group's
`csv_paths` list, then rerun. No code changes.

The figures are drawn to the parameters in `plotting/plot_common.py`, shared with
`plotting/plot_rollout_success.py`: the two tools measure the same rate at
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
from plot_common import (AXIS_LABEL_PT, CURVE_FIGSIZE,  # noqa: E402
                         HORIZON_RULE, NoData, annotate_horizon_rule,
                         gap_mask, metric_axis_label,
                         default_colors, sample_step, starts_at_origin,
                         new_curve_figure, plot_group_curve, prepend_origin,
                         read_table, resolve_out_dir, save_curve_figure,
                         style_curve_axes, warn)


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
    # Vertical rules marking an event inside the run — the curriculum switch is
    # what this exists for. One list per x_axis, each entry a bare position or
    # a {"x": ..., "label": ...} pair whose label is annotated onto the rule:
    #   "horizon_lines": {"total_steps": [{"x": 655360,
    #                                      "label": "T = 320 -> 2560"}],
    #                     "total_resets": [{"x": 2048, "label": "..."}]}
    # drawn with the same heavy grey rule plot_rollout_success.py puts at a
    # horizon change, so the two tools mark the same event the same way.
    horizon_lines: Optional[Dict[str, List[float]]] = None
    # Legend styling, optionally per x_axis:
    #   "legend": {"fontsize": 20,
    #              "total_steps":  {"framealpha": 0.0, "reverse": true},
    #              "total_resets": {"loc": "lower right"}}
    # Keys other than `reverse` go straight to `ax.legend`.
    legend: Dict[str, object] = field(default_factory=dict)
    # Axis type sizes (pt); absent/null = plot_common's AXIS_LABEL_PT /
    # TICK_LABEL_PT:
    #   "axis": {"label_fontsize": 18, "tick_fontsize": 15}
    axis: Dict[str, object] = field(default_factory=dict)
    # Success-rate y bounds of the main panels. Shared lower/upper, then a
    # per-eval_kind block, then a per-x_axis block inside it, each laid over
    # the one before; a null / absent end keeps the bound fitted to the data.
    # The gap panels (which also draw grasp) keep the fitted bound.
    #   "sr_ylim": {"lower": 0.0,
    #               "in_domain": {"total_steps":  {"upper": 0.95},
    #                             "total_resets": {"upper": 0.3}},
    #               "out_of_domain": {"upper": 0.2}}
    sr_ylim: Dict[str, object] = field(default_factory=dict)
    # Plot-box width:height, e.g. "4:3" (default), "16:9", or a number (= w/h).
    aspect_ratio: Optional[object] = None


def _runs_to_csv_paths(entries: List) -> List:
    """Map the shared `runs` key (glob dirs) onto this script's `csv_paths`.

    `runs` is the format `plotting/plot_common.py` uses, and it points at a run's
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


def _parse_horizon_lines(raw) -> Optional[Dict[str, List[float]]]:
    """`horizon_lines` from the config: {x_axis: [x, ...]}, validated here so a
    typo fails at load rather than silently drawing nothing."""
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise ValueError("`horizon_lines` must be a mapping of x_axis -> list "
                         "of x positions, e.g. {\"total_steps\": [655360]}")
    out = {}
    for axis, values in raw.items():
        if axis not in ("total_steps", "total_resets"):
            raise ValueError(f"`horizon_lines` key {axis!r}: expected "
                             f"'total_steps' or 'total_resets'")
        if isinstance(values, (int, float, dict)):
            values = [values]
        entries = []
        for v in values:
            if isinstance(v, dict):
                if "x" not in v:
                    raise ValueError("`horizon_lines` entry needs an `x`")
                entries.append((float(v["x"]), v.get("label")))
            else:
                entries.append((float(v), None))
        out[axis] = entries
    return out


def draw_horizon_lines(ax, cfg: "PlotConfig", x_axis: str) -> None:
    """Rules for this panel's axis; nothing drawn when the config set none.

    A labelled entry also gets an arrow pointing at the rule, placed high in
    the panel and offset to the right — where a curve that falls after the
    switch leaves room, and clear of the legend in the upper left.
    """
    for x, label in (cfg.horizon_lines or {}).get(x_axis, []):
        ax.axvline(x, **HORIZON_RULE)
        if not label:
            continue
        annotate_horizon_rule(ax, x, label)


def legend_options(cfg: "PlotConfig", x_axis: str):
    """(kwargs for `ax.legend`, reverse?) for this panel: the config's shared
    settings with its per-axis block laid over them."""
    shared = {k: v for k, v in (cfg.legend or {}).items()
              if k not in ("total_steps", "total_resets")}
    kw = {**shared, **((cfg.legend or {}).get(x_axis) or {})}
    reverse = bool(kw.pop("reverse", False))
    return kw, reverse


def parse_aspect(raw) -> Optional[float]:
    """`aspect_ratio` ("16:9" or w/h as a number) -> box height/width."""
    if raw is None:
        return None
    if isinstance(raw, str):
        w, sep, h = raw.partition(":")
        if not sep:
            raise ValueError(f"`aspect_ratio` {raw!r}: expected \"W:H\", e.g. \"16:9\"")
        w, h = float(w), float(h)
    else:
        w, h = float(raw), 1.0
    if w <= 0 or h <= 0:
        raise ValueError(f"`aspect_ratio` {raw!r} must be positive")
    return h / w


def axis_options(cfg: "PlotConfig") -> dict:
    """`label_pt` / `tick_pt` / `box_aspect` kwargs for `style_curve_axes`
    from `axis` and `aspect_ratio`."""
    a = cfg.axis or {}
    return {"label_pt": a.get("label_fontsize"),
            "tick_pt": a.get("tick_fontsize"),
            "box_aspect": parse_aspect(cfg.aspect_ratio)}


_SR_ENDS = ("lower", "upper")
_EVAL_KINDS = ("in_domain", "out_of_domain")
_X_AXES = ("total_steps", "total_resets")


def sr_range(cfg: "PlotConfig", eval_kind: str, x_axis: str):
    """(lower, upper) success-rate bounds for one main panel: `sr_ylim`'s
    shared ends, overlaid by its `eval_kind` block, overlaid by that block's
    `x_axis` block."""
    raw = cfg.sr_ylim or {}
    kind = raw.get(eval_kind) or {}
    kw = {k: raw[k] for k in _SR_ENDS if k in raw}
    kw.update({k: kind[k] for k in _SR_ENDS if k in kind})
    kw.update(kind.get(x_axis) or {})
    return kw.get("lower"), kw.get("upper")


def _check_style(raw: dict) -> None:
    """Typos in `axis` / `sr_ylim` fail at load instead of drawing defaults."""
    bad = set(raw.get("axis") or {}) - {"label_fontsize", "tick_fontsize"}
    if bad:
        raise ValueError(f"`axis` has unknown keys {sorted(bad)}; valid: "
                         f"label_fontsize, tick_fontsize")

    def ends_only(block, where):
        if not isinstance(block, dict) or set(block) - set(_SR_ENDS):
            raise ValueError(f"`sr_ylim` {where}: expected only lower / upper")

    for k, v in (raw.get("sr_ylim") or {}).items():
        if k in _SR_ENDS:
            continue
        if k not in _EVAL_KINDS or not isinstance(v, dict):
            raise ValueError(f"`sr_ylim` key {k!r}: expected lower / upper, or "
                             f"an in_domain / out_of_domain block")
        for kk, vv in v.items():
            if kk in _SR_ENDS:
                continue
            if kk not in _X_AXES:
                raise ValueError(f"`sr_ylim.{k}` key {kk!r}: expected lower / "
                                 f"upper, or a total_steps / total_resets block")
            ends_only(vv, f"{k}.{kk}")


def _figsize(raw: dict) -> Tuple[float, float]:
    """`figsize` if the config set one; otherwise the default height with the
    width following `aspect_ratio`. A fixed 8x6 figure cannot widen, so a
    16:9 box inside it only gets shorter while the type stays the same size."""
    if raw.get("figsize"):
        return tuple(raw["figsize"])
    box = parse_aspect(raw.get("aspect_ratio"))
    if box is None:
        return CURVE_FIGSIZE
    w, h = CURVE_FIGSIZE
    # Scale the default width by how much wider the box is than the default
    # 4:3 box, so the margins around it keep their proportion.
    from plot_common import CURVE_BOX_ASPECT
    return (w * CURVE_BOX_ASPECT / box, h)


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
        figsize=_figsize(raw),
        eval_kinds=tuple(raw.get("eval_kinds", ("in_domain", "out_of_domain"))),
        x_axes=tuple(raw.get("x_axes", ("total_steps", "total_resets"))),
        horizon_lines=_parse_horizon_lines(raw.get("horizon_lines")),
        legend=raw.get("legend") or {},
        axis=raw.get("axis") or {},
        sr_ylim=raw.get("sr_ylim") or {},
        aspect_ratio=raw.get("aspect_ratio"),
    )
    parse_aspect(cfg.aspect_ratio)
    _check_style(raw)
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

    Returns a long-form DataFrame with the original columns.

    Every "there is nothing here" case — absent, zero bytes, header-only, or an
    older file without a column — raises `NoData`, which `aggregate_all` catches
    per group: one such run is named on stderr and skipped rather than costing
    every other group its figure. Only a config that yields NO group at all is
    fatal, because that is a configuration error rather than a missing file.
    """
    df = read_table(csv_path, what="eval_success.csv",
                    required_cols=("episode", "total_steps", "total_resets",
                                   "eval_kind", "group", "task", "success",
                                   "grasp"))
    if cronos_group_filter:
        df = df[df["group"].isin(cronos_group_filter)].copy()
    if task_filter:
        df = df[df["task"].isin(task_filter)].copy()
    # This file's own eval cadence on each x axis, carried per row so a resume
    # chain whose legs eval at different intervals (T320 every 4 episodes,
    # T2560 every episode) judges holes against the right one.
    for x_axis in ("total_steps", "total_resets"):
        step = sample_step(df[x_axis].to_numpy(dtype=float)) if len(df) else None
        df[f"_step_{x_axis}"] = np.nan if step is None else step
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

    Each row also carries its `total_steps` and that leg's step cadence as
    `_ref_x` / `_ref_step`: eval rounds are scheduled on steps, so that is the
    axis a missing round is judged on, whatever axis is plotted.
    """
    sub = df[df["eval_kind"] == eval_kind]
    step_col = f"_step_{x_axis}"
    cols = [x_axis, metric, step_col, "_ref_x", "_ref_step"]
    if sub.empty:
        return pd.DataFrame(columns=cols)
    for c in (step_col, "_step_total_steps"):
        if c not in sub.columns:
            sub = sub.assign(**{c: np.nan})
    keys = ["episode", x_axis] if x_axis == "total_steps" \
        else ["episode", x_axis, "total_steps"]
    means = (sub
             .groupby(keys, as_index=False)
             .agg(**{metric: (metric, "mean"), step_col: (step_col, "first"),
                     "_ref_step": ("_step_total_steps", "first")})
             .assign(_ref_x=lambda d: d["total_steps"])
             .sort_values("_ref_x"))
    return means[cols].reset_index(drop=True)


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
                              x_end: Optional[float] = None) -> Tuple[np.ndarray, np.ndarray]:
    """Interpolate each run's `metric` onto a common x grid.

    The grid spans the runs' full data (0 to the shortest run's last eval)
    whatever the crop is, so cropping never changes the sampling density or
    what the smoothing window covers. `x_end` (the crop) only adds one extra
    grid point exactly at the crop, so a cropped curve ends on it instead of
    at the nearest grid point or eval before it.

    series_list: list of single-metric frames from `per_run_series`.
    Returns (x_grid, stacked) where stacked has shape (N_runs, n_points).
    Runs that are entirely empty are dropped.

    A grid point a run did not measure is NaN in that run's row, never a
    value bridged from its neighbours: before its first eval when its
    beginning is missing (only a run that starts at its first eval round is
    anchored at (0, 0)), and inside any hole `gap_mask` finds — judged against
    each leg's own eval interval. Both are judged on `total_steps` (`_ref_x`),
    the axis evals are scheduled on: resets per eval round are not constant
    (HSR-only's grow 9 -> 27 -> 105 -> ... per round), so a normal round on
    the reset axis would read as a hole and the curve would break. Callers
    average with nan-aware reductions, so such a point is
    the mean of the runs that have it, and NaN (a break in the line) when none
    do.
    """
    usable = [s for s in series_list if not s.empty]
    if not usable:
        return np.empty((0,)), np.empty((0, n_points))
    prepped = []
    for s in usable:
        x = s[x_axis].to_numpy(dtype=float)
        y = s[metric].to_numpy(dtype=float)
        step_col = f"_step_{x_axis}"
        step = (s[step_col].to_numpy(dtype=float) if step_col in s.columns
                else np.full(x.size, np.nan))
        if "_ref_x" in s.columns:
            rx = s["_ref_x"].to_numpy(dtype=float)
            rstep = s["_ref_step"].to_numpy(dtype=float)
        else:
            rx, rstep = x, step
        mask = np.isfinite(x) & np.isfinite(y) & np.isfinite(rx)
        x, y, rx, rstep = x[mask], y[mask], rx[mask], rstep[mask]
        if x.size == 0:
            continue
        if x[0] > 0.0 and starts_at_origin(rx, rstep):
            x = np.concatenate(([0.0], x))
            y = np.concatenate(([0.0], y))
            rx = np.concatenate(([0.0], rx))
            rstep = np.concatenate((rstep[:1], rstep))
        prepped.append((x, y, rx, rstep))
    if not prepped:
        return np.empty((0,)), np.empty((0, n_points))
    # Common grid: 0 to min of right edges, plus the crop point.
    x_min = 0.0
    x_max = min(x.max() for x, _, _, _ in prepped)
    grid = np.linspace(x_min, x_max, n_points)
    if x_end is not None and x_min < x_end < x_max:
        grid = np.union1d(grid, [float(x_end)])
    rows = []
    for x, y, rx, rstep in prepped:
        row = np.interp(grid, x, y)
        row[(grid < x[0]) | (grid > x[-1])] = np.nan
        if x.size > 1:
            # Grid points strictly inside a hole between two evals (a hole
            # found on the step axis, located on the plotted one).
            holes = gap_mask(rx, rstep)
            i = np.clip(np.searchsorted(x, grid, side="right"), 1, x.size - 1)
            inside = (grid > x[i - 1]) & (grid < x[i])
            row[holes[i - 1] & inside] = np.nan
        rows.append(row)
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
        if not np.isfinite(y[i]):
            continue            # a hole stays a hole; neighbours do not fill it
        seg = y[lo:hi]
        seg = seg[np.isfinite(seg)]
        out[i] = seg.mean()
    return out


# ---------------------------------------------------------------------------
# Aggregation across groups → single long-form table
# ---------------------------------------------------------------------------


# Columns `aggregate_all` produces. Named so the empty case can still carry
# them: `pd.DataFrame([])` has no columns at all, so every downstream
# `long_df["eval_kind"]` raised `KeyError: 'eval_kind'` from deep inside pandas
# instead of saying that nothing was aggregated.
_AGG_COLUMNS = ("group", "eval_kind", "x_axis", "x_value",
                "metric", "mean", "std", "n_runs", "_x_end")
_SUMMARY_COLUMNS = ("group", "eval_kind", "x_axis", "metric",
                    "final_x", "final_mean", "final_std", "n_runs")


def crop_end(cfg: PlotConfig, dfs: List[pd.DataFrame], eval_kind: str,
             x_axis: str) -> Optional[float]:
    """Where this group's `x_axis` panel ends, or None for "all the data".

    The axis's own crop (`end_steps` on total_steps, `end_resets` on
    total_resets) wins. Without one, the other axis's crop carries over,
    converted per run through that run's own steps <-> resets pairs
    (interpolated, so it lands exactly on the crop, not on the last eval
    before it) and taking the smallest over runs so every seed still covers
    the panel.
    """
    own = {"total_steps": cfg.end_steps, "total_resets": cfg.end_resets}
    if own.get(x_axis) is not None:
        return float(own[x_axis])
    other = "total_resets" if x_axis == "total_steps" else "total_steps"
    if x_axis not in own or own[other] is None:
        return None
    ends = []
    for d in dfs:
        sub = d[d["eval_kind"] == eval_kind]
        if sub.empty:
            continue
        pairs = (sub.groupby("episode")[[other, x_axis]].first()
                 .sort_values(other))
        xo = np.concatenate(([0.0], pairs[other].to_numpy(dtype=float)))
        xa = np.concatenate(([0.0], pairs[x_axis].to_numpy(dtype=float)))
        ends.append(float(np.interp(float(own[other]), xo, xa)))
    return min(ends) if ends else None


def aggregate_all(cfg: PlotConfig) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Build the long-form `aggregated` table + the `summary` table.

    `aggregated` columns: group, eval_kind, x_axis, x_value, metric, mean, std, n_runs
    `summary` columns:    group, eval_kind, x_axis, metric, final_x, final_mean,
                          final_std, n_runs

    Both frames come back with those columns even when no group yielded a
    single row, so a caller can test `.empty` instead of tripping over a
    missing column.
    """
    long_rows = []
    summary_rows = []
    metrics = ("success", "grasp")
    for spec in cfg.groups:
        # Load seeds (each may be a single CSV or a resume chain).
        try:
            dfs = [load_seed(entry, spec.cronos_group_filter, spec.task_filter)
                   for entry in spec.csv_paths]
        except (NoData, FileNotFoundError, TypeError, ValueError) as e:
            print(f"  [WARN] {spec.label}: {e}")
            continue
        if all(d.empty for d in dfs):
            print(f"  [WARN] {spec.label}: every CSV loaded but held no rows "
                  f"after cronos_group_filter / task_filter")
            continue
        rows_before = len(long_rows)
        # No rows are dropped for the crop: every eval point is sampled, and
        # the crop only decides where each panel's curve ends (`crop_end`).
        for eval_kind in cfg.eval_kinds:
            for x_axis in cfg.x_axes:
                x_end = crop_end(cfg, dfs, eval_kind, x_axis)
                for metric in metrics:
                    series_list = [per_run_series(d, eval_kind, x_axis, metric) for d in dfs]
                    grid, stacked = interpolate_runs_to_grid(
                        series_list, x_axis, metric,
                        n_points=cfg.n_interp_points, x_end=x_end)
                    if stacked.size == 0:
                        continue
                    # NaN = that run did not measure this x; the mean is over
                    # the runs that did, and NaN where none did.
                    have = np.isfinite(stacked)
                    counts = have.sum(axis=0)
                    with np.errstate(invalid="ignore", divide="ignore"):
                        total = np.where(have, stacked, 0.0).sum(axis=0)
                        mean = np.where(counts > 0, total / np.maximum(counts, 1), np.nan)
                        sq = np.where(have, (stacked - mean) ** 2, 0.0).sum(axis=0)
                        std = np.where(counts > 0, np.sqrt(sq / np.maximum(counts, 1)), np.nan)
                    for x_v, m_v, s_v, c in zip(grid, mean, std, counts):
                        long_rows.append({
                            "group": spec.label, "eval_kind": eval_kind,
                            "x_axis": x_axis, "x_value": float(x_v),
                            "metric": metric, "mean": float(m_v),
                            "std": float(s_v), "n_runs": int(c),
                            "_x_end": np.inf if x_end is None else x_end,
                        })
                    in_crop = grid <= (np.inf if x_end is None else x_end)
                    last = np.flatnonzero((counts > 0) & in_crop)
                    if last.size == 0:
                        continue
                    last = last[-1]
                    summary_rows.append({
                        "group": spec.label, "eval_kind": eval_kind,
                        "x_axis": x_axis, "metric": metric,
                        "final_x": float(grid[last]),
                        "final_mean": float(mean[last]),
                        "final_std": float(std[last]),
                        "n_runs": int(counts[last]),
                    })
        if len(long_rows) == rows_before:
            # The CSVs loaded but nothing survived. Report the group's own
            # contents so the cause is visible without opening the files: it is
            # almost always an `eval_kind` that is not in `cfg.eval_kinds`, or a
            # filter / crop that removed every row.
            kinds = sorted({k for d in dfs for k in d["eval_kind"].unique()})
            print(f"  [WARN] {spec.label}: loaded {sum(len(d) for d in dfs)} rows "
                  f"but produced no curve. eval_kind present: {kinds or '(none)'}; "
                  f"config wants: {list(cfg.eval_kinds)}. Also check "
                  f"cronos_group_filter / task_filter / end_steps / end_resets.")
    return (pd.DataFrame(long_rows, columns=list(_AGG_COLUMNS)),
            pd.DataFrame(summary_rows, columns=list(_SUMMARY_COLUMNS)))


# ---------------------------------------------------------------------------
# Plot layer
# ---------------------------------------------------------------------------


def _group_color(cfg: PlotConfig, index: int, palette) -> object:
    """A group's colour: its own `color` if the config set one, else the shared
    palette `plot_rollout_success.py` also draws from."""
    return cfg.groups[index].color or palette[index]


def plot_main_panel(long_df: pd.DataFrame, eval_kind: str, x_axis: str,
                     cfg: PlotConfig, out_path: Path):
    """One PNG per (eval_kind × x_axis). Overlays mean ± std envelopes for each
    config group on the same axes (one line per group). Returns the SR bounds
    drawn, (lower, upper), or False when no figure was written."""
    sub = long_df[(long_df["eval_kind"] == eval_kind) &
                  (long_df["x_axis"] == x_axis) &
                  (long_df["metric"] == "success")]
    if sub.empty:
        # Returning False rather than writing an empty axes: a blank panel reads
        # as "every group scored zero", and `main` used to announce it as
        # written even though no file was ever saved.
        warn(f"{eval_kind} / {x_axis}: no group has success rows — no figure")
        return False
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
        # Smoothed over the full range first, THEN cut, so the crop does not
        # change the values near its edge either.
        keep = x <= g_sub["_x_end"].to_numpy()
        x, m, s = x[keep], m[keep], s[keep]
        n = int(g_sub["n_runs"][keep].max())
        x, m, s = prepend_origin(x, m, s)
        plot_group_curve(ax, x, m, s, color=_group_color(cfg, i, palette),
                         label=spec.label, n_series=n)
        x_max = max(x_max, float(x.max()) if len(x) else 0.0)
    draw_horizon_lines(ax, cfg, x_axis)
    legend_kw, legend_reverse = legend_options(cfg, x_axis)
    style_curve_axes(ax, x_axis=x_axis, y_label=metric_axis_label("success"),
                     x_max=x_max, legend_kw=legend_kw,
                     legend_reverse=legend_reverse, **axis_options(cfg),
                     y_range=sr_range(cfg, eval_kind, x_axis))
    # The visible SR bounds (padding removed), so `sr_ylim` can be filled in
    # from what the fitted bound gave.
    lo, hi = ax.get_ylim()
    pad = (hi - lo) / 1.04 * 0.02
    y_shown = (lo + pad, hi - pad)
    # No title / suptitle: eval_kind and x_axis are already in the filename, and
    # the smoothing window and band meaning are settings rather than findings.
    save_curve_figure(fig, out_path)
    return y_shown


def plot_gap_panel(long_df: pd.DataFrame, eval_kind: str,
                    cfg: PlotConfig, out_path: Path) -> bool:
    """Success vs grasp overlaid. A persistent gap (grasp high, success low)
    is the placement-collapse signature."""
    sub = long_df[(long_df["eval_kind"] == eval_kind) &
                  (long_df["x_axis"] == "total_steps")]
    if sub.empty:
        warn(f"gap / {eval_kind}: no group has rows on total_steps — no figure")
        return False
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
            keep = x <= g_sub["_x_end"].to_numpy()
            x, m = prepend_origin(x[keep], m[keep])
            # No band here: two overlaid metrics per group already crowd the
            # panel, and the gap between the lines is what this figure is for.
            ax.plot(x, m, label=f"{spec.label} ({metric})", color=color, ls=ls,
                    linewidth=2.0)
            x_max = max(x_max, float(x.max()) if len(x) else 0.0)
    draw_horizon_lines(ax, cfg, "total_steps")
    legend_kw, legend_reverse = legend_options(cfg, "total_steps")
    style_curve_axes(ax, x_axis="total_steps", y_label="success / grasp rate",
                     legend_kw=legend_kw, legend_reverse=legend_reverse,
                     x_max=x_max, **axis_options(cfg))
    save_curve_figure(fig, out_path)
    return True


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
    if long_df.empty:
        # Stop here rather than at the first `long_df["eval_kind"]` in the plot
        # layer: every group already printed a [WARN] naming its own reason, and
        # continuing would only bury those behind a pandas KeyError.
        raise SystemExit(
            "[plot] nothing was aggregated — no figure can be drawn.\n"
            "  The per-group [WARN] lines above give the reason. The usual ones:\n"
            "    * a `runs`/`csv_paths` path does not exist (the segs/seed line\n"
            "      above shows 0 for those), or points at a run dir instead of\n"
            "      its glob/ subdirectory\n"
            "    * the run is eval-only / too early and eval_success.csv has no\n"
            "      rows yet\n"
            "    * `eval_kinds` in the config does not match the eval_kind values\n"
            "      in the CSV\n"
            "    * cronos_group_filter / task_filter matches nothing\n"
            "    * end_steps / end_resets crop away every eval point")
    # `<name>_` prefixed like every PNG below: without it two configs sharing an
    # out_dir silently overwrite each other's aggregates.
    agg_csv = out_dir / f"{cfg.name}_aggregated.csv"
    sum_csv = out_dir / f"{cfg.name}_summary.csv"
    (long_df[long_df["x_value"] <= long_df["_x_end"]]
     .drop(columns="_x_end").to_csv(agg_csv, index=False))
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
            y_shown = plot_main_panel(long_df, eval_kind, x_axis, cfg, png)
            if y_shown:
                print(f"  wrote {png.name}  (SR y {y_shown[0]:.3g} .. {y_shown[1]:.3g})")
        gap_png = out_dir / f"{cfg.name}_gap_{eval_kind}.png"
        if plot_gap_panel(long_df, eval_kind, cfg, gap_png):
            print(f"  wrote {gap_png.name}")

    print("[plot] done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
