"""Shared config loading for the per-run plot tools.

One JSON describes several experiment groups; each group holds several runs, and
a run may itself be a **resume chain** — a list of run dirs that are stitched
into a single continuous series. Same shape as `plotting/configs/plot_config.json`'s
`csv_paths`, so the two config styles stay recognisable.

    {
      "out_dir": "reports/figures/2026-08-26",
      "name": "perturb_ablation",
      "groups": [
        {
          "label": "noep baseline",
          "runs": [
            "/data/runs/A/wandb/run-.../glob",
            ["/data/runs/B-parent/.../glob", "/data/runs/B-child/.../glob"]
          ]
        },
        { "label": "noep + PTBmixed", "runs": ["/data/runs/C/.../glob"] }
      ]
    }

`runs` entries:
  - a string  -> one run, one series
  - a list    -> a resume chain, concatenated in order into ONE series

Within a group each entry is a separate series (typically a seed); the plot
tools aggregate them into a mean ± spread band. Across groups you get one curve
or one panel each.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import pandas as pd


# Keys that only `plot_eval_success.py` acts on. Accepted and ignored here so
# ONE config file drives all three plot tools instead of each needing its own.
_PLOT_PY_TOP_KEYS = {
    "smoothing_window", "n_interp_points", "end_steps", "end_resets",
    "figsize", "eval_kinds", "x_axes", "horizon_lines", "legend",
    "axis", "sr_ylim", "aspect_ratio",
}
_PLOT_PY_GROUP_KEYS = {"color", "cronos_group_filter", "task_filter"}

# Tool options that may be set in the config instead of on the command line, so
# a comparison is reproducible from one file. Each is the snake_case form of the
# matching CLI flag, and the CLI always wins when both are given. Keys a given
# tool does not understand are simply ignored, exactly like the eval-SR keys
# above — one config can carry settings for all three tools.
TOOL_OPTION_KEYS = {
    # plot_segment_positions.py
    "actor_kind", "phase", "workspace_scale", "step_range", "density",
    "bin_size", "per_task", "dense_min", "color_by",
    # plot_sequence_eval.py: label -> one training glob dir per series, used
    # when a checkpoint's own directory carries no `eval_success.csv` (a
    # checkpoint copied into a bundle).
    "final_eval_runs",
    # plot_rollout_success.py (`metric` is also plot_sequence_eval.py's)
    "direction", "by", "metric", "smooth", "per_group", "reset_split",
    "rollout_style", "rollout_figsize", "rollout_ylim",
    # plot_sequence_eval.py
    "seq_kind",
}


@dataclass
class Group:
    label: str
    # One entry per series; each entry is the ordered list of run dirs that make
    # up that series (length 1 for a run that was not resumed).
    chains: List[List[Path]] = field(default_factory=list)


@dataclass
class PlotConfig:
    name: str
    out_dir: Path
    groups: List[Group] = field(default_factory=list)
    # Tool options carried in the config (see TOOL_OPTION_KEYS). Read via
    # `option()`, which lets the CLI override.
    options: dict = field(default_factory=dict)

    def option(self, key, cli_value, default=None):
        """CLI value if given, else the config's, else `default`."""
        if cli_value is not None:
            return cli_value
        return self.options.get(key, default)


def resolve_out_dir(raw_value, config_path) -> Path:
    """Where a config's outputs go — one rule for every plot tool.

    - set        -> resolved to an absolute path, so the result does not depend
                    on the directory the tool happened to be launched from.
    - empty/absent -> **next to the config file**, not the CWD. The eval-SR
                    tool used `Path("").resolve()`, i.e. the CWD, and the
                    shipped `plot_config.json` ships `"out_dir": ""` — so
                    figures landed wherever the shell was, which is neither
                    predictable nor discoverable. The config's own directory is
                    both.

    Every caller writes `<out_dir>/<name>_*`, so all three tools' figures for
    one comparison land together and several configs can share one directory
    without colliding.
    """
    config_path = Path(config_path)
    if raw_value:
        return Path(raw_value).expanduser().resolve()
    return config_path.resolve().parent


def load_plot_config(path) -> PlotConfig:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"config not found: {path}")
    raw = json.loads(path.read_text())
    if not isinstance(raw, dict):
        raise ValueError(f"config must be a JSON object, got {type(raw).__name__}")

    known = {"out_dir", "name", "groups"}
    # JSON has no comment syntax, so any key starting with "_" is treated as one
    # and ignored — that is what `plotting/configs/plot_runs_example.json` uses to carry
    # its own documentation.
    unknown = ({k for k in raw if not k.startswith("_")}
               - known - _PLOT_PY_TOP_KEYS - TOOL_OPTION_KEYS)
    if unknown:
        raise ValueError(f"unknown config keys: {sorted(unknown)}; valid: "
                         f"{sorted(known | _PLOT_PY_TOP_KEYS | TOOL_OPTION_KEYS)} "
                         f"(keys starting with '_' are ignored as comments)")

    groups = []
    for i, g in enumerate(raw.get("groups", [])):
        if not isinstance(g, dict):
            raise ValueError(f"groups[{i}] must be an object")
        g_unknown = set(g) - {"label", "runs", "csv_paths"} - _PLOT_PY_GROUP_KEYS
        if g_unknown:
            raise ValueError(f"groups[{i}] has unknown keys: {sorted(g_unknown)}")
        label = g.get("label") or f"group_{i}"
        if "runs" in g and "csv_paths" in g:
            raise ValueError(
                f"groups[{i}] ('{label}') sets both `runs` and `csv_paths`; use "
                f"one (prefer `runs`, which points at the glob dir)")
        # `csv_paths` is `plot_eval_success.py`'s original spelling: paths to
        # individual `eval_success.csv` files. A CSV only names one of the three
        # files these tools read, so it is resolved back to its containing glob
        # dir — which is what `runs` states directly, and why `runs` is the
        # preferred form.
        key = "runs" if "runs" in g else "csv_paths"
        to_dir = (lambda s: Path(s)) if key == "runs" else (lambda s: Path(s).parent)
        entries = g.get(key, [])
        if not entries:
            raise ValueError(f"groups[{i}] ('{label}') has no {key}")
        chains = []
        for j, entry in enumerate(entries):
            if isinstance(entry, str):
                chains.append([to_dir(entry)])
            elif isinstance(entry, list) and all(isinstance(x, str) for x in entry):
                if not entry:
                    raise ValueError(f"groups[{i}].{key}[{j}] is an empty chain")
                chains.append([to_dir(x) for x in entry])
            else:
                raise ValueError(
                    f"groups[{i}].{key}[{j}] must be a path string or a list of "
                    f"them (a resume chain), got {type(entry).__name__}")
        groups.append(Group(label=label, chains=chains))

    if not groups:
        raise ValueError("config has no groups")
    return PlotConfig(name=raw.get("name", path.stem),
                      out_dir=resolve_out_dir(raw.get("out_dir"), path),
                      groups=groups,
                      options={k: raw[k] for k in TOOL_OPTION_KEYS if k in raw})


def concat_chain(frames: List[pd.DataFrame], x_col: str = "total_steps") -> pd.DataFrame:
    """Stitch a resume chain into one series.

    A resumed run restates the parent's counters, so the child's x values can
    overlap the parent's. At the seam the CHILD wins — it is the run that
    actually produced those steps under the resumed configuration. Same rule
    `plot_eval_success.py` applies to its `csv_paths` chains.
    """
    frames = [f for f in frames if f is not None and len(f)]
    if not frames:
        return pd.DataFrame()
    if len(frames) == 1:
        return frames[0]
    kept = []
    for idx, f in enumerate(frames):
        later = frames[idx + 1:]
        if later:
            # Drop rows this run contributed that a later run re-covers.
            floor = min(g[x_col].min() for g in later if len(g))
            f = f[f[x_col] < floor]
        kept.append(f)
    return pd.concat(kept, ignore_index=True)


def default_colors(n: int):
    import matplotlib.pyplot as plt
    cmap = plt.get_cmap("tab10" if n <= 10 else "tab20")
    return [cmap(i % cmap.N) for i in range(n)]


# ---------------------------------------------------------------------------
# Shared look for the two success-rate figures
# ---------------------------------------------------------------------------
#
# `plot_eval_success.py` (eval SR) and `plot_rollout_success.py` (rollout SR)
# measure the same quantity on the same x axis, one at eval points and one at
# segment boundaries, so they are read side by side and must be drawn to the
# same parameters. These are `plot_rollout_success.py`'s, which is the version
# both now use; the one thing added on top is that every curve is extended back
# to the origin (see `prepend_origin`).

# ---------------------------------------------------------------------------
# Legend type size — ONE knob for every figure in plotting/
# ---------------------------------------------------------------------------
#
# Every legend in every tool is written as `legend_pt(n)`, n notches below this
# base, so raising `LEGEND_PT` grows all of them together and keeps their
# relative sizes: the main curve legends sit at the base, legends inside a
# crowded panel or a small grid subplot sit one to three notches under it.
# Change this number and re-run the tools; nothing else needs touching.
# Axis dressing: one notch up from matplotlib's 10pt defaults, so the units
# stay readable when a figure is shrunk into a column.
# What a metric is called on an axis. Every tool draws the same quantity —
# the fraction of trials that succeeded — so it reads the same everywhere,
# `success_chained` included: the chaining is the figure's subject, not a
# different unit.
METRIC_AXIS_LABEL = {
    "success": "success rate",
    "success_chained": "success rate",
    "grasp": "grasp rate",
    "obj_grasped": "object-grasped rate",
}


def metric_axis_label(metric: str) -> str:
    """The axis name for `metric`; unknown metrics keep their own spelling."""
    return METRIC_AXIS_LABEL.get(metric, str(metric).replace("_", " "))


AXIS_LABEL_PT = 18
TICK_LABEL_PT = 15
LEGEND_PT = 16
LEGEND_MIN_PT = 7          # below this the labels stop being readable in print


def legend_pt(steps: int = 0) -> int:
    """`LEGEND_PT` moved `steps` notches down the scale (negative = smaller)."""
    return max(LEGEND_MIN_PT, LEGEND_PT + steps)


CURVE_FIGSIZE = (8.0, 6.0)
CURVE_DPI = 120
# The clipping range for the ±std bands. Not the axes limit: that is fitted to
# the data by `curve_ylim` below, so a run that never passes 0.2 is not drawn as
# a flat line along the bottom fifth of an empty panel.
CURVE_YLIM = (-0.02, 1.02)
# Fitted upper bound = the highest plotted value (band tops included) plus
# `HEADROOM` of it, rounded up to a whole `STEP` so the ticks stay round, and
# never below `MIN_TOP` — zooming in past a quarter of the range makes a low
# curve look like a high one, which is the opposite of the point.
CURVE_Y_HEADROOM = 0.30
CURVE_Y_STEP = 0.05
CURVE_Y_MIN_TOP = 0.25
CURVE_LINEWIDTH = 2.0
CURVE_BAND_ALPHA = 0.16
CURVE_GRID_ALPHA = 0.3
# Placement only — the type size comes from `legend_pt` via `curve_legend`.
CURVE_LEGEND = {"loc": "upper left"}
# ...unless the legend is put OUTSIDE the box, which is the only placement that
# cannot cover a curve. `bbox_inches="tight"` in `save_curve_figure` grows the
# saved image to include it, so nothing is clipped.
CURVE_LEGEND_OUTSIDE = {"loc": "upper left", "bbox_to_anchor": (1.02, 1.0),
                        "borderaxespad": 0.0}
# A legend is only readable while it stays out of the curve's way: a
# reset-split panel can carry ten entries, and ten full-size lines reach halfway
# down a panel whose y bound is now fitted to the data. So past a few entries
# the type steps back down — `(at most this many entries, notches below
# LEGEND_PT)`, with anything longer three notches down.
CURVE_LEGEND_STEPS = ((6, 0), (10, -2))
# The plot box's height:width ratio is forced rather than left to `figsize`:
# the figure's own margins depend on how wide the tick labels come out, so a
# 4:3 `figsize` gives a visibly non-4:3 box, and a different x range changes it
# again. These are the two shapes every curve figure comes in — landscape 4:3
# by default, 16:9 where the x range is the point.
CURVE_BOX_ASPECT = 3 / 4
# The rollout-success curves run over millions of environment steps and are
# read left-to-right — where the resets fall, how each inter-reset piece rises
# and drops. 4:3 squeezes the sixteen pieces of a T2560 run together; 16:9
# gives them room.
FLAT_FIGSIZE = (12.0, 7.0)
FLAT_BOX_ASPECT = 9 / 16

# The one axis-label vocabulary every curve tool uses. Steps are shown in
# millions (`steps_in_millions`) rather than with a "1e6" offset in the corner.
X_LABEL = {
    "total_steps": "environment steps (M)",
    "total_resets": "number of resets",
    "segment": "segment index (80 steps each)",
    "episode": "episode",
}


STEPS_UNIT = 1e6


def steps_in_millions(ax) -> None:
    """Tick labels of a raw-step x axis in millions, to match X_LABEL."""
    from matplotlib.ticker import FuncFormatter
    ax.xaxis.set_major_formatter(
        FuncFormatter(lambda v, _: f"{v / STEPS_UNIT:g}"))


def new_curve_figure(figsize=None):
    import matplotlib.pyplot as plt
    return plt.subplots(figsize=tuple(figsize or CURVE_FIGSIZE))


# A spacing wider than this many typical sample steps is a hole in the data
# (missing segments / eval rounds), not the sampling interval.
GAP_FACTOR = 1.5


def sample_step(x):
    """The series' typical x spacing (median of the positive differences), or
    None when it has fewer than two distinct points."""
    import numpy as np
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    d = np.diff(np.unique(x))
    return float(np.median(d)) if d.size else None


def _local_steps(x, step):
    """`step` as an array aligned with `x`: a scalar is broadcast, an array
    (one run's own cadence per point — a resume chain changes cadence between
    legs) has its NaNs filled with the series-wide estimate."""
    import numpy as np
    x = np.asarray(x, dtype=float)
    if step is None or np.ndim(step) == 0:
        step = sample_step(x) if step is None else step
        return np.full(x.size, np.nan if step is None else float(step))
    step = np.asarray(step, dtype=float)
    fallback = sample_step(x)
    return np.where(np.isfinite(step), step,
                    np.nan if fallback is None else fallback)


def gap_mask(x, step=None):
    """Boolean per spacing of `x` (length `x.size - 1`): True where the spacing
    is a hole, i.e. wider than `GAP_FACTOR` × the cadence of the runs on both
    sides of it (the larger of the two, so a chain that switches from a
    coarse to a fine cadence is not read as holes)."""
    import numpy as np
    x = np.asarray(x, dtype=float)
    if x.size < 2:
        return np.zeros(0, dtype=bool)
    st = _local_steps(x, step)
    with np.errstate(invalid="ignore"):
        thr = GAP_FACTOR * np.fmax(st[:-1], st[1:])
        return np.diff(x) > thr          # NaN threshold -> False


def starts_at_origin(x, step=None) -> bool:
    """Whether the series begins at its first sampling point, so that (0, 0)
    is the measurement just before it rather than across a hole.

    A series whose first point sits more than `GAP_FACTOR` steps from 0 lost its
    beginning (a resumed child without its parent, early segments missing), and
    joining it to the origin would draw a ramp nobody measured. A single-point
    series has no step to judge by and keeps the anchor.
    """
    import numpy as np
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        return False
    if x[0] <= 0.0:
        return True
    first = _local_steps(x, step)[0]
    return not np.isfinite(first) or x[0] <= GAP_FACTOR * first


def prepend_origin(x, mean, std=None, step=None):
    """Extend a curve back to (0, 0).

    Both figures start from an untrained policy, so the origin is a measured
    fact and not an extrapolation — but neither file records it. `eval_success.
    csv`'s first row is the first eval round and `rollout_success.csv`'s is the
    first 80-step boundary, so a curve drawn from the data alone begins hanging
    in mid-air at whatever success rate that first point happened to hit, and
    two runs whose first point sits at different x are not comparable at the
    left edge. Anchoring at (0, 0) makes the left edge mean the same thing in
    every panel of both tools.

    Nothing is prepended when the series already starts at or before 0, or
    when its beginning is missing (see `starts_at_origin`): the curve then
    starts at its first real point instead of being joined to the origin.
    `step` overrides the spacing estimated from `x` (scalar, or per point).
    """
    import numpy as np
    x = np.asarray(x, dtype=float)
    mean = np.asarray(mean, dtype=float)
    std = None if std is None else np.asarray(std, dtype=float)
    if x.size == 0 or x[0] <= 0.0 or not starts_at_origin(x, step):
        return (x, mean) if std is None else (x, mean, std)
    x = np.concatenate(([0.0], x))
    mean = np.concatenate(([0.0], mean))
    if std is None:
        return x, mean
    return x, mean, np.concatenate(([0.0], std))


def break_gaps(x, *ys, step=None):
    """Insert a NaN point inside every hole of `x`, so a line is not drawn
    across missing samples. Returns `(x, *ys)` with the same number of arrays.

    A hole is what `gap_mask` says: a spacing wider than `GAP_FACTOR` × the
    cadence around it (`step`: scalar or per point; estimated from `x` when
    None). Arrays passed as None stay None.
    """
    import numpy as np
    x = np.asarray(x, dtype=float)
    if x.size < 2:
        return (x, *ys)
    gaps = np.flatnonzero(gap_mask(x, step)) + 1
    if gaps.size == 0:
        return (x, *ys)
    mids = 0.5 * (x[gaps - 1] + x[gaps])
    out = [np.insert(x, gaps, mids)]
    for y in ys:
        out.append(None if y is None else
                   np.insert(np.asarray(y, dtype=float), gaps, np.nan))
    return tuple(out)


def plot_group_curve(ax, x, mean, std=None, *, color, label, n_series=None):
    """One group's mean curve plus its ±1 std band, drawn identically in both
    tools. The band is omitted for a single series, where std is 0 everywhere
    and a zero-width ribbon only suggests a spread that was never measured.

    `n_series` only decides whether to draw the band; it is not put in the
    legend. Both tools print the series count per group on stderr, so the
    figure stays a figure.
    """
    import numpy as np
    ax.plot(x, mean, linewidth=CURVE_LINEWIDTH, color=color, label=label)
    if std is not None and (n_series is None or n_series > 1):
        lo = np.clip(np.asarray(mean) - std, CURVE_YLIM[0], 1.0)
        hi = np.clip(np.asarray(mean) + std, 0.0, 1.0)
        ax.fill_between(x, lo, hi, color=color, alpha=CURVE_BAND_ALPHA, linewidth=0)


def curve_legend(n_entries: int = 1, *, outside: bool = False) -> dict:
    """`CURVE_LEGEND` with the type size fitted to how many entries it holds.

    `outside=True` parks it beside the box instead of inside the upper-left
    corner, for a panel whose curve reaches into that corner.
    """
    steps = -3
    for limit, notches in CURVE_LEGEND_STEPS:
        if n_entries <= limit:
            steps = notches
            break
    size = legend_pt(steps)
    base = CURVE_LEGEND_OUTSIDE if outside else CURVE_LEGEND
    return {**base, "fontsize": size, "title_fontsize": size}


def axes_peak(ax) -> float:
    """The highest y value actually drawn on `ax`: lines and band tops.

    Rules drawn with `axvline` are skipped — their y data is (0, 1) in axes
    coordinates, not success, and taking it for data would pin every fitted
    limit at 1.0. They are identified by their transform, which is blended
    rather than `transData`.
    """
    import numpy as np
    peak = 0.0
    for line in ax.get_lines():
        if line.get_transform() is not ax.transData:
            continue
        y = np.asarray(line.get_ydata(), dtype=float)
        if y.size and np.isfinite(y).any():
            peak = max(peak, float(np.nanmax(y)))
    for coll in ax.collections:                 # fill_between bands
        if coll.get_transform() is not ax.transData:
            continue
        for path in coll.get_paths():
            v = path.vertices
            if v.size and np.isfinite(v[:, 1]).any():
                peak = max(peak, float(np.nanmax(v[:, 1])))
    return peak


def curve_ylim(peak: float):
    """`(bottom, top)` for a success axis whose data tops out at `peak`.

    A success rate is bounded at 1, but a run that never clears 0.2 spends four
    fifths of a full-range panel on whitespace, and the shape that the figure
    exists to show is squashed into the bottom strip. The bound is therefore
    fitted to the data — rounded up to a whole `CURVE_Y_STEP` so the ticks stay
    round, floored at `CURVE_Y_MIN_TOP` so a near-zero curve is not magnified
    into a dramatic one, and padded by the same 2% of the range at both ends
    that the old fixed `(-0.02, 1.02)` gave the unit interval.
    """
    import math
    if not (peak == peak) or peak <= 0:          # NaN or nothing drawn
        top = CURVE_Y_MIN_TOP
    else:
        top = math.ceil(peak * (1.0 + CURVE_Y_HEADROOM) / CURVE_Y_STEP) * CURVE_Y_STEP
        top = min(1.0, max(CURVE_Y_MIN_TOP, top))
    pad = 0.02 * top
    return (-pad, top + pad)


def style_curve_axes(ax, *, x_axis: str, y_label: str, x_max=None,
                     y_max=None, legend=True, box_aspect=None,
                     legend_outside=False, legend_kw=None,
                     legend_reverse=False, label_pt=None, tick_pt=None,
                     y_range=None):
    """Shared axis dressing. Call it AFTER every curve is drawn: the y bound is
    fitted to what is on the axes.

    `y_max` overrides the measured peak (for a figure whose scale must match
    another's); `legend=False` drops the legend, which is what a panel holding a
    single unsplit curve wants — its one entry would only repeat the filename.
    `box_aspect` overrides `CURVE_BOX_ASPECT` (`FLAT_BOX_ASPECT` for a curve
    read along its x range), and `legend_outside` moves the legend clear of the
    box. `legend_kw` is passed to `ax.legend` on top of the fitted defaults
    (`loc`, `fontsize`, `framealpha`, ...) and `legend_reverse` flips the entry
    order, for a panel read bottom-up.

    `label_pt` / `tick_pt` override `AXIS_LABEL_PT` / `TICK_LABEL_PT`.
    `y_range` = `(lower, upper)` fixes the y bounds instead of fitting them;
    either end may be None to keep the fitted value. The same 2% of the range is
    padded onto both ends as the fitted bound gets, so a curve lying on the
    bound is not half clipped.
    """
    ax.set_xlabel(X_LABEL.get(x_axis, x_axis),
                  fontsize=AXIS_LABEL_PT if label_pt is None else label_pt)
    if x_axis == "total_steps":
        steps_in_millions(ax)
    ax.set_ylabel(y_label,
                  fontsize=AXIS_LABEL_PT if label_pt is None else label_pt)
    ax.tick_params(labelsize=TICK_LABEL_PT if tick_pt is None else tick_pt)
    if tick_pt is not None:          # the "1e6" multiplier follows the ticks
        ax.xaxis.get_offset_text().set_fontsize(tick_pt)
        ax.yaxis.get_offset_text().set_fontsize(tick_pt)
    lo, hi = curve_ylim(axes_peak(ax) if y_max is None else float(y_max))
    lower, upper = y_range or (None, None)
    if lower is not None or upper is not None:
        lower = 0.0 if lower is None else float(lower)
        upper = hi / 1.02 if upper is None else float(upper)   # fitted top
        if upper <= lower:
            raise ValueError(f"y range upper {upper} <= lower {lower}")
        pad = 0.02 * (upper - lower)
        lo, hi = lower - pad, upper + pad
    ax.set_ylim(lo, hi)
    # Left edge pinned to 0 so the anchored origin is actually visible.
    ax.set_xlim(0.0, None if not x_max else float(x_max))
    ax.grid(alpha=CURVE_GRID_ALPHA)
    ax.set_box_aspect(CURVE_BOX_ASPECT if box_aspect is None else box_aspect)
    if legend:
        handles, labels = ax.get_legend_handles_labels()
        kw = curve_legend(len(handles), outside=legend_outside)
        kw.update(legend_kw or {})
        if legend_reverse:
            handles, labels = handles[::-1], labels[::-1]
        ax.legend(handles, labels, **kw)


def save_curve_figure(fig, out_path) -> Path:
    """Write the figure. No suptitle: these go into documents that caption them
    themselves, and a baked-in run name is wrong the moment the figure is
    reused. The run identity lives in the filename."""
    import matplotlib.pyplot as plt
    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=CURVE_DPI, bbox_inches="tight")
    plt.close(fig)
    return out_path


def read_run_config(run_dir: Path) -> Optional[dict]:
    """Load a run's `run_config.json` (env_n / env_m / num_envs / obj_set / ...).

    Returns None when absent — an older run, or a glob dir assembled by hand.
    """
    p = Path(run_dir) / "run_config.json"
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Missing / unusable input
# ---------------------------------------------------------------------------
#
# The tools are pointed at whole directories of runs, and a run legitimately
# lacks a file: an eval-only run writes no `rollout_success.csv`, a run started
# with `--no-record-segment-pose` writes no `segment_pose.csv`, and a run that
# is still in its first episode has files that exist but are empty. Older runs
# additionally predate columns the current code expects.
#
# In `--config` mode one such run must not abort the whole figure set, so the
# loaders raise `NoData` and every group loop catches it, warns, and moves on.
# A tool only fails when NOTHING could be plotted — which is a configuration
# error, not a missing file.
#
# The single-run paths (`--run-dir` / `--csv`) still fail loudly: there the user
# named one specific file, so "not found" is the answer to their question.


class NoData(Exception):
    """This input holds nothing plottable. Caught per group in --config mode."""


def warn(msg: str) -> None:
    print(f"[warn] {msg}", file=sys.stderr)


def read_table(path, *, what: str, required_cols=()) -> pd.DataFrame:
    """Read one CSV, raising `NoData` for every "there is nothing here" case.

    Covers all four in one place: the file is absent, it is zero bytes, it holds
    only a header, or it predates a column the caller needs. Each raises with a
    message naming the file, so the [warn] line a caller prints is enough to act
    on without opening anything.
    """
    path = Path(path)
    if not path.exists():
        raise NoData(f"{path}: no {what}")
    try:
        df = pd.read_csv(path)
    except pd.errors.EmptyDataError:
        raise NoData(f"{path}: {what} is empty (0 bytes) — the run wrote no rows yet")
    except Exception as e:                      # malformed / truncated mid-write
        raise NoData(f"{path}: {what} could not be parsed ({type(e).__name__}: {e})")
    if df.empty:
        raise NoData(f"{path}: {what} has a header but no rows")
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise NoData(f"{path}: {what} is missing column(s) {missing} "
                     f"(have: {sorted(df.columns)}) — written by an older version?")
    return df


# ---------------------------------------------------------------------------
# Output naming, shared by the per-group figures
# ---------------------------------------------------------------------------


def slugify(text: str) -> str:
    """A group label as a filename fragment (labels carry spaces and '+')."""
    import re
    s = re.sub(r"[^0-9A-Za-z._-]+", "-", str(text)).strip("-._")
    return s or "group"


def unique_slugs(labels) -> dict:
    """label -> distinct filename fragment. Two labels can slugify the same
    way ("noep +PTB" and "noep-PTB"), and the second figure would silently
    overwrite the first, so collisions get a numeric suffix."""
    out, used = {}, set()
    for label in labels:
        base = slugify(label)
        slug, i = base, 2
        while slug in used:
            slug, i = f"{base}-{i}", i + 1
        used.add(slug)
        out[label] = slug
    return out


def out_variant(base, *parts: str) -> Path:
    """`fig.png` + ("noep", "obj") -> `fig_noep_obj.png`."""
    base = Path(base)
    return base.with_name("_".join([base.stem, *parts]) + (base.suffix or ".png"))


# ---------------------------------------------------------------------------
# Reset-segmented curves (per-group figures)
# ---------------------------------------------------------------------------
#
# A training curve is not one continuous experiment: at every reset the batch is
# re-randomized, so the success rate the policy achieves *within* one inter-reset
# stretch is a different quantity from the trend across them. Drawing the whole
# thing as one line hides that structure — a within-stretch climb followed by a
# drop at the reset reads as noise.
#
# So the per-group figures split each curve at its own reset boundaries and give
# each piece its own colour. A per-group figure holds exactly one group and its
# legend title names it, so the hue is not needed for group identity there and
# is spent on the reset index instead.
#
# It is only worth doing when a stretch has enough points to show a shape. At
# `segment_len = 80` an episode holds `episode_len / 80` segments, so T1280 is
# the first horizon with a usable 16; T320's 4 and T80's 1 are not curves.

RESET_SPLIT_MIN_EPISODE_LEN = 1280
# Independent of the horizon: HSR fires soft resets at segment boundaries, so a
# long-horizon run can still come out chopped into 2-point fragments. Below this
# median piece length the split is noise and the curve is drawn whole.
RESET_SPLIT_MIN_PIECE = 4


def piece_colors(n: int):
    """`n` visibly distinct colours, one per reset piece.

    Categorical, not a ramp through one hue. A same-hue ramp keeps the tie to
    the group's colour in the all-group figure, but its adjacent steps differ
    only in lightness, and past three or four pieces that is not enough to tell
    them apart at a glance — which is the whole point of splitting the curve.

    Nothing is lost by spending the hue here: a per-group figure holds exactly
    one group, and the legend's title names it, so the hue is free to carry the
    reset index instead. The first four entries are the maximally-separated
    head of `tab10` (blue, orange, green, red).

    Greys are left out: `SHORT_HORIZON_COLOR` marks the unsplit short-horizon
    stretch of a curriculum chain, and a piece must not be mistaken for it.
    """
    import matplotlib.pyplot as plt
    cmap = plt.get_cmap("tab10" if n <= 9 else "tab20")
    palette = [c for c in (cmap(i) for i in range(cmap.N))
               if max(c[:3]) - min(c[:3]) > 0.05]
    return [palette[i % len(palette)] for i in range(n)]


# The part of a curve whose runs are too short-horizon to split (the T320 leg
# of a T320 -> T2560 curriculum) is drawn whole in this colour.
SHORT_HORIZON_COLOR = "0.45"


def reset_pieces(resets):
    """Split an aligned cumulative-reset array into inter-reset index ranges.

    Returns half-open `(start, stop)` index pairs. A boundary is any point where
    the cumulative count went up — which is what "a reset happened just before
    this segment" means in `rollout_success.csv`, where `total_resets` is stamped
    at the segment's end.
    """
    import numpy as np
    r = np.asarray(resets, dtype=float)
    if r.size == 0:
        return []
    bounds = [0]
    last = r[0]
    for i in range(1, r.size):
        # NaN (an older CSV without the column, a gap in a resume chain, or a
        # `break_gaps` hole) is not a boundary: an unknown reset count is not
        # evidence of a reset. Across it, the next known count is compared with
        # the last known one.
        if not np.isfinite(r[i]):
            continue
        if np.isfinite(last) and r[i] > last:
            bounds.append(i)
        last = r[i]
    bounds.append(r.size)
    return [(bounds[i], bounds[i + 1]) for i in range(len(bounds) - 1)]


def piece_labels(x, resets, pieces, *, max_labelled: int = 10):
    """Legend text per reset piece, or None for the pieces left unlabelled.

    With more pieces than a legend can carry, only the first and last are
    named — the shade ramp already says which is which, and 30 legend entries
    would cover the panel they describe.
    """
    import numpy as np
    r = np.asarray(resets, dtype=float)
    names = []
    for lo, _ in pieces:
        n = r[lo] if lo < r.size and np.isfinite(r[lo]) else None
        names.append("before any reset" if n == 0 else
                     "resets unknown" if n is None else f"after {int(n)} resets")
    if len(pieces) <= max_labelled:
        return names
    return [names[0]] + [None] * (len(pieces) - 2) + [names[-1]]


def plot_reset_segmented_curve(ax, x, mean, std=None, resets=None, *,
                               pieces=None, labels=None, n_series=None,
                               mark_resets=True, colors=None):
    """One group's curve, split at its resets, a distinct colour per piece.

    Same line width, band and clipping as `plot_group_curve` — only the colour
    varies along the curve — so a per-group figure and the main figure are read
    to the same scale.

    Pieces are drawn joined: each starts at its predecessor's last point, so the
    line is continuous and the colour change alone marks the reset. The dotted
    rule sits between the two points, which is where the reset actually happened.

    `colors` gives each piece its colour (default: `piece_colors`).
    """
    import numpy as np
    x = np.asarray(x, dtype=float)
    mean = np.asarray(mean, dtype=float)
    std = None if std is None else np.asarray(std, dtype=float)
    if pieces is None:
        pieces = reset_pieces(resets if resets is not None else np.zeros_like(x))
    if labels is None:
        labels = [None] * len(pieces)
    shades = list(colors) if colors is not None else piece_colors(len(pieces))
    band = std is not None and (n_series is None or n_series > 1)
    for i, (lo, hi) in enumerate(pieces):
        start = lo - 1 if i else lo          # join to the previous piece
        xs, ms = x[start:hi], mean[start:hi]
        if xs.size == 0:
            continue
        ax.plot(xs, ms, linewidth=CURVE_LINEWIDTH, color=shades[i],
                label=labels[i], solid_capstyle="round")
        if band:
            ss = std[start:hi]
            ax.fill_between(xs, np.clip(ms - ss, CURVE_YLIM[0], 1.0),
                            np.clip(ms + ss, 0.0, 1.0),
                            color=shades[i], alpha=CURVE_BAND_ALPHA, linewidth=0)
        if mark_resets and i and np.isfinite(x[lo - 1]):
            # Darker than the grid it sits on: the rule marks an event, and at
            # the grid's own weight it read as one more gridline.
            ax.axvline(0.5 * (x[lo - 1] + x[lo]), color="0.45", linewidth=0.9,
                       linestyle=":", zorder=0)


# ---------------------------------------------------------------------------
# The curriculum switch
# ---------------------------------------------------------------------------
#
# A resume chain may change the horizon mid-curve — a T320 -> T2560 curriculum
# (CL) is one run continued under a longer episode. That switch is a different
# kind of event from a reset: a reset re-randomizes the batch, the switch
# changes what the task IS, so everything left of it and everything right of it
# were measured under different conditions. It gets the heaviest mark on the
# figure and it is drawn whether or not the curve is split.
#
# Heaviest, not darkest: weight carries the distinction from the dotted reset
# rules, so the tone can stay grey. At full black a 2pt rule is the loudest
# thing in the panel and reads as a border cutting the figure in two rather
# than as an event inside it.

HORIZON_RULE = {"color": "0.35", "linewidth": 2.2, "zorder": 2.5}


# The label a horizon rule carries, above the panel with an arrow down to the
# rule: inside the box it lands on the curves or on the legend.
HORIZON_LABEL_DY = 12.0


def annotate_horizon_rule(ax, x: float, text: str) -> None:
    """`text` above the panel, centred on the rule at `x`, arrow pointing down."""
    ax.annotate(text, xy=(x, 1.0), xycoords=("data", "axes fraction"),
                xytext=(0.0, HORIZON_LABEL_DY), textcoords="offset points",
                ha="center", va="bottom", fontsize=AXIS_LABEL_PT - 2,
                color=HORIZON_RULE["color"], annotation_clip=False,
                arrowprops=dict(arrowstyle="->", color=HORIZON_RULE["color"],
                                linewidth=1.4, shrinkA=2.0, shrinkB=0.0))


def horizon_switch_label(old: float, new: float) -> str:
    """`T = 320 -> 2560`, the two episode lengths the switch ran between."""
    fmt = lambda v: f"{v:g}"
    return f"T = {fmt(old)} \u2192 {fmt(new)}"


def horizon_changes(x, horizons):
    """x positions where `horizons` steps to a new value.

    Each entry is `(x, old_horizon, new_horizon)`. The x is the midpoint
    between the last point of the old horizon and the first
    of the new one, which is where the switch actually happened — the same
    convention as the reset rule. NaNs (gap points, or a run that did not record
    its `episode_len`) carry the last known horizon forward rather than counting
    as a change of their own.
    """
    import numpy as np
    x = np.asarray(x, dtype=float)
    if horizons is None:
        return []
    h = np.asarray(horizons, dtype=float)
    if h.size != x.size or h.size < 2:
        return []
    out, last, last_i = [], None, None
    for i in range(h.size):
        if not np.isfinite(h[i]):
            continue
        if last is not None and h[i] != last and last_i is not None:
            a, b = x[last_i], x[i]
            if np.isfinite(a) and np.isfinite(b):
                out.append((0.5 * (a + b), last, h[i]))
        last, last_i = h[i], i
    return out


def mark_horizon_changes(ax, x, horizons, *, label: bool = True) -> list:
    """Draw the rule at every horizon switch, labelled with the two episode
    lengths it ran between. Returns the x positions."""
    changes = horizon_changes(x, horizons)
    xs = []
    for xc, old, new in changes:
        ax.axvline(xc, **HORIZON_RULE)
        if label:
            annotate_horizon_rule(ax, xc, horizon_switch_label(old, new))
        xs.append(xc)
    return xs
