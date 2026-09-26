"""Sequence-eval success by task position and by task, as bars, from
`eval_per_trial.csv`.

A sequential eval round runs every task slot in a row without resetting, so the
same task is met at position 1, 2, 3 and 4 (`task_idx` 0..3) across the round's
orders — see `doc/eval_sequential.md` §3. This tool asks the question those
files are laid out for: **does the success rate depend on where in the sequence
the task sits?** — and, with the positions pooled away, the plain one next to
it: **how often does each task succeed at all?**

    python tools/plot_sequence_eval.py --run-dir <EVAL_OUT_DIR>/wandb/run-*/glob
    python tools/plot_sequence_eval.py --config tools/plot_sequence_example.json

Figures (one PNG each, never a grid — the convention of the other plot tools),
one set per metric and round kind (`seen` = the orders training ran,
`unseen` = the untrained ones; recorded as `training` / `random`):

    <name>_seq_position_<metric>_<kind>_in_domain.png       pooled over tasks,
    <name>_seq_position_<metric>_<kind>_out_of_domain.png   one per domain
    <name>_seq_position_<metric>_<kind>_per_task/…_<task>.png
                                                    one per task, both domains

So the default on a two-domain eval writes **eight** pooled figures —
{in-domain, out-of-domain} × {seen, unseen} × {success, success_chained} —
plus the per-task folders. `--seq-kind pooled` goes back to one set per metric
with the round kinds averaged together; naming one kind keeps only it.

x = position in the sequence, one bar per group at each position. A pooled
figure holds one domain and its bars are solid; a per-task figure holds both
domains in the same bar, aligned at the same x: **out-of-domain is the solid
bar, in-domain the diagonally hatched bar drawn over it**, in the group's colour (the hatch a darker shade,
so it stays visible over the solid part). In-domain usually scores higher and
shows as hatching above the solid bar. Bars are the mean over a group's series
(seeds); no spread is drawn or written.

`<name>_seq_position.csv` carries every plotted number (`metric` and `seq_kind`
columns tell the figures apart; `seq_kind` keeps the recorded `training` /
`random` spelling that the figures call seen / unseen), plus `n_trials`.

The second series pools the positions away instead, to answer the other
question: **how well is each task done, wherever in the round it sits?**

    <name>_seq_task_<metric>_<kind>.png                   x = task, both domains
    <name>_seq_task_<metric>_<kind>_per_scene/…_<scene>.png     with --per-scene

x = task, one bar per group at each, over every trial of that task whatever its
position — the task's overall rate in these rounds, order and all. Both domains
share a bar exactly as the per-task position figures do: out-of-domain solid,
in-domain hatched over it. `--per-scene` adds one figure per scene (`group` in
`eval_per_trial.csv`, the YAML scene the env belongs to) holding that scene's
tasks; the pooled-over-scenes figure is written either way, and is not the mean
of the per-scene ones but the mean over their trials, so a scene with more envs
weighs more. `<name>_seq_task.csv` carries these numbers, the pooled rows under
`scene = __all__`.

Metrics (`--metric`, config `metric`; one or several, default
`success success_chained` — both ways of scoring a sequence):

    success           each task judged on its own — the position effect
    success_chained   cumulative AND along the round: position k succeeds only if
                      positions 1..k all did, so once a task fails every later
                      one counts as failed; it can only fall with position.
                      Derived from `success` when the file predates the column.
    grasp, obj_grasped  latched within the task slot

`--seq-kind` (config `seq_kind`) decides what happens to the round kinds:
`each` (default) gives the **seen** rounds (`training`: the rotations training
ran) and the **unseen** ones (`random`: untrained cycles) their own figures —
which is what answers whether an untrained order costs more — `pooled` averages
them into one set (the full 24-order design), and naming one kind (`seen` /
`unseen`, or the recorded `training` / `random`) keeps only it.

The average figure pools trials, not task means: with every task at every
position equally often (a complete design) the two are identical, and on an
incomplete eval pooling weights each task by what was actually run.

Overlapping runs: when the shards of one series both hold the same trial
(same domain, pass, round, slot and env), the trial's value is the mean over
the shards that ran it; a trial only one shard ran counts as that shard's
value. Across series (seeds) every bar is likewise the mean over the series
that ran that (task, position) — or that task, in the second series — and a
series that did not is left out, not counted as zero.

Missing input follows the other tools: in `--config` mode a run without a usable
`eval_per_trial.csv` is warned about and skipped; `--run-dir` fails loudly.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import json
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from plot_common import (AXIS_LABEL_PT, CURVE_BOX_ASPECT,  # noqa: E402
                         CURVE_DPI, CURVE_GRID_ALPHA, Group, NoData,
                         PlotConfig, TICK_LABEL_PT, default_colors,
                         load_plot_config, metric_axis_label, out_variant,
                         read_table, unique_slugs, warn)

_METRICS = ("success", "success_chained", "grasp", "obj_grasped")
_SEQ_KINDS = ("training", "random")
# `--seq-kind pooled`: every round kind in one figure set, the pre-split
# behaviour. Also the label those pooled rows carry.
POOLED = "pooled"
# What a round kind is called in figures and filenames. `eval_per_trial.csv`
# (and this tool's own CSV) keep the recorded `training` / `random`; the
# figures say what those mean — the orders training saw, and orders it did not.
SEQ_LABEL = {"training": "seen", "random": "unseen", POOLED: "all"}
SEQ_LEGEND = {"training": "seen sequence", "random": "unseen sequence",
              POOLED: "all sequences"}
# Accepted spellings of a round kind on the command line / in the config.
SEQ_ALIASES = {"seen": "training", "unseen": "random", "all": POOLED}
_DOMAINS = ("in_domain", "out_of_domain")
_DOMAIN_LABEL = {"in_domain": "in-domain", "out_of_domain": "out-of-domain"}
_DEFAULT_METRICS = ("success", "success_chained")
# One trial. `pass_label` matters: in the serial schedule every scene reuses env
# indices 0..n-1 in its own pass. `task` / `seq_kind` keep shards of different
# plans from being averaged into one trial.
_TRIAL_KEY = ["eval_kind", "pass_label", "seq_idx", "task_idx", "env_idx",
              "task", "seq_kind"]
_ROUND_KEY = ["eval_kind", "pass_label", "seq_idx", "env_idx"]
ALL_TASKS = "__all__"
ALL_SCENES = "__all__"

SEQ_LEGEND_PT = 15
BAR_FIGSIZE = (6.4, 4.8)
BAR_YLIM = (0.0, 1.02)
BAR_EDGE_WIDTH = 1.8
IN_DOMAIN_HATCH = "///"
# Task names are whole phrases, so the per-task figures lean their tick labels.
TASK_TICK_PT = TICK_LABEL_PT - 3   # 16 two-line task ticks on one axis
# The task family is one figure: plain success, every round kind pooled.
TASK_METRIC = "success"
# A task axis carries 16 of those ticks, each two lines wide: the panel is
# widened and flattened for them, where the position axis (4 ticks) is not.
TASK_FIGSIZE = (13.0, 4.8)
TASK_BOX_ASPECT = 0.30
TASK_X_ROTATE = 20.0


def task_tick_label(task: str) -> str:
    """`put X on Y` -> "X\nY": the object over its receptacle.

    Upright two-line ticks, not one leaning line — a slanted sentence is read
    by turning the page, and the two halves are what actually differ between
    the tasks. Anything that does not parse is left as it is.
    """
    t = str(task).strip()
    if t.lower().startswith("put "):
        t = t[4:]
    obj, sep, recep = t.partition(" on ")
    return f"{obj}\n{recep}" if sep else t


def hatch_color(color, factor: float = 0.55):
    """A darker shade of `color` for the in-domain hatch, so it stays visible
    where it crosses the same group's solid out-of-domain bar."""
    from matplotlib.colors import to_rgb
    return tuple(c * factor for c in to_rgb(color))


def load_trials(run_dir: Path, *, required: bool) -> pd.DataFrame:
    """One run's `eval_per_trial.csv`, with `eval_kind` / `seq_kind` ensured.

    `eval_kind` and `seq_kind` were appended after the first ten columns; a file
    that predates them has the domain only in `prefix`
    (`{eval_kind}_seq{round}_task{slot}`) and no round kind, which is then
    reported as `unknown` so `--seq-kind all` still plots it.
    """
    path = Path(run_dir) / "eval_per_trial.csv"
    if required and not path.exists():
        raise FileNotFoundError(
            f"{path} not found. It is written by the standalone eval "
            f"(eval_only.py / main.py --eval-sequential), not by training.")
    df = read_table(path, what="eval_per_trial.csv",
                    required_cols=("seq_idx", "task_idx", "task", "env_idx"))
    if "eval_kind" not in df.columns:
        if "prefix" not in df.columns:
            raise NoData(f"{path}: neither `eval_kind` nor `prefix` — cannot "
                         f"tell in-domain from out-of-domain")
        df["eval_kind"] = np.where(
            df["prefix"].astype(str).str.startswith("out_of_domain"),
            "out_of_domain", "in_domain")
    if "seq_kind" not in df.columns:
        df["seq_kind"] = "unknown"
    # `group` in this file is the SCENE (the YAML group an env belongs to);
    # `group` in a plot config is an experiment series. Only one of the two may
    # travel under that name inside this tool, so the scene is renamed here and
    # `group` means the config's group everywhere below.
    if "group" not in df.columns:
        df["group"] = ""
    df["scene"] = (df["group"].fillna("").astype(str)
                   .replace("", "unknown"))
    if "pass_label" not in df.columns:
        df["pass_label"] = ""
    df["pass_label"] = df["pass_label"].fillna("").astype(str)
    for col in _METRICS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if "success_chained" not in df.columns and "success" in df.columns:
        df["success_chained"] = chain_success(df)
    return df


def chain_success(df: pd.DataFrame) -> pd.Series:
    """Cumulative AND of `success` along `task_idx` within one round of one env
    (the live eval's `chain *= success`): once a task fails, every later task of
    that round counts as failed.

    A missing `success` is "not measured", never a failure: it is NaN itself,
    and so is every later task of the round — unless an earlier task already
    failed, in which case they are a known 0.
    """
    d = df.sort_values("task_idx", kind="stable")
    keys = [d[k] for k in _ROUND_KEY]
    s = d["success"]
    prod = s.fillna(1.0).groupby(keys, sort=False, dropna=False).cumprod()
    unknown = (s.isna().astype(int)
                .groupby(keys, sort=False, dropna=False).cummax().astype(bool))
    chained = prod.where(~unknown | (prod == 0.0))
    return chained.reindex(df.index)


def load_series(chain, *, label: str, required: bool) -> pd.DataFrame:
    """A series = one run, or several shards of one eval stitched together.

    A trial that several shards hold is the mean of their values (per metric,
    over the shards that have it); a trial only one shard holds keeps its value.
    Each chained value is taken from its own shard's round before averaging, so
    the mean of `success_chained` is still "cleared every task up to here".
    """
    frames = []
    for run_dir in chain:
        try:
            frames.append(load_trials(Path(run_dir), required=required))
        except NoData as e:
            if required:
                raise SystemExit(str(e))
            warn(f"group '{label}': {e}")
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    if not df.duplicated(subset=_TRIAL_KEY).any():
        return df
    metrics = [m for m in _METRICS if m in df.columns]
    overlap = df.duplicated(subset=_TRIAL_KEY, keep=False)
    # mean() skips NaN only: a 0 (failed) counts, a missing value does not.
    # `scene` is a property of the trial, not a measurement: the shards agree on
    # it (it follows env_idx within a pass), so the first one's is kept.
    merged = (df.groupby(_TRIAL_KEY, as_index=False, sort=False, dropna=False)
                .agg({**{m: "mean" for m in metrics}, "scene": "first"}))
    n_overlap = int(df[overlap].drop_duplicates(subset=_TRIAL_KEY).shape[0])
    print(f"[series] group '{label}': {n_overlap} trials held by more than one "
          f"run ({int(overlap.sum())} rows) — averaged", file=sys.stderr)
    return merged


_KEY = ["seq_kind", "eval_kind", "task", "position"]
_TASK_KEY = ["seq_kind", "eval_kind", "scene", "task"]


def position_rates(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Per-series rates: (seq_kind, eval_kind, task, position) -> mean, plus
    the pooled all-task row under `task = ALL_TASKS`."""
    d = df.dropna(subset=[metric]).assign(position=lambda x: x["task_idx"] + 1)
    per_task = (d.groupby(_KEY)[metric]
                 .agg(rate="mean", n_trials="size").reset_index())
    pooled = (d.groupby(["seq_kind", "eval_kind", "position"])[metric]
               .agg(rate="mean", n_trials="size").reset_index()
               .assign(task=ALL_TASKS))
    return pd.concat([pooled, per_task], ignore_index=True)


def task_rates(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Per-series rates with the position pooled away:
    (seq_kind, eval_kind, scene, task) -> mean, plus the all-scene row under
    `scene = ALL_SCENES`.

    Every trial of the task counts once, whatever slot it sat in, so the
    all-scene row is the rate over that series' trials — not the mean of the
    per-scene rates, which would give a 4-env scene the weight of a 16-env one.
    """
    d = df.dropna(subset=[metric])
    per_scene = (d.groupby(_TASK_KEY)[metric]
                  .agg(rate="mean", n_trials="size").reset_index())
    pooled = (d.groupby(["seq_kind", "eval_kind", "task"])[metric]
               .agg(rate="mean", n_trials="size").reset_index()
               .assign(scene=ALL_SCENES))
    return pd.concat([pooled, per_scene], ignore_index=True)


def load_groups(cfg: PlotConfig, *, seq_kind: str, required: bool) -> dict:
    """Every group's trials: label -> [(chain, df), ...], one entry per series.

    Read once and handed to each `collect` call, so the two figure series are
    built from the same rows rather than from two reads of the same files.

    `seq_kind` selects the rounds: one of `_SEQ_KINDS` keeps only those rounds,
    `POOLED` pools every kind into one set of figures, and `each` (the default)
    keeps them apart, so a round kind is never averaged with another.
    """
    series = {g.label: [] for g in cfg.groups}
    for group in cfg.groups:
        for chain in group.chains:
            df = load_series(chain, label=group.label, required=required)
            if df.empty:
                continue
            if seq_kind == POOLED:
                df = df.assign(seq_kind=POOLED)
            elif seq_kind != "each":
                kept = df[df["seq_kind"] == seq_kind]
                if kept.empty:
                    warn(f"group '{group.label}': {chain[-1]} has no "
                         f"seq_kind={seq_kind!r} rows (present: "
                         f"{sorted(df['seq_kind'].unique())})")
                    continue
                df = kept
            series[group.label].append((chain, df))
    return series


def collect(cfg: PlotConfig, series: dict, *, metrics, rates_fn, key,
            pooled) -> pd.DataFrame:
    """Every group's rates, aggregated over its series, for every metric.

    Columns: metric, group, `key`, mean, n_series, n_trials. A key with no
    trial in any series has no row at all — "no data" is never written as a 0.

    `rates_fn` is what a series' rows are reduced to (`position_rates` or
    `task_rates`) and `key` its keys; `pooled` names the (column, value) those
    rates carry their pooled row under, which is what the per-group line counts
    trials over.
    """
    pooled_col, pooled_val = pooled
    rows = []
    for metric in metrics:
        for group in cfg.groups:
            agg = aggregate_group(group.label, series[group.label], metric,
                                  rates_fn=rates_fn, key=key)
            if agg is None:
                continue
            rows.append(agg.assign(metric=metric, group=group.label))
            print(f"[group] {metric:<16s} {group.label:<34s} "
                  f"{int(agg['series_total'].iloc[0])} series, "
                  f"{int(agg.loc[agg[pooled_col] == pooled_val, 'n_trials'].sum())} trials",
                  file=sys.stderr)
    if not rows:
        raise SystemExit(
            "[seq] no group produced any rows — nothing to plot.\n"
            "  The [warn] lines above name the groups. Each `runs` entry must be a\n"
            "  standalone eval's glob/ dir holding eval_per_trial.csv (training\n"
            "  runs have none), and --seq-kind / --metric must exist in it.")
    cols = ["metric", "group", *key, "mean", "n_series", "n_trials"]
    return pd.concat(rows, ignore_index=True)[cols]


def aggregate_group(label: str, series, metric: str, *, rates_fn, key):
    """One group's per-`key` mean, or None when no series has a value for
    `metric`.

    Each bar is the mean over the series that ran it; a series that did not
    contributes nothing (not a 0), and a series whose trials all failed
    contributes its 0.
    """
    per_series = []
    for chain, df in series:
        if metric not in df.columns:
            warn(f"group '{label}': {chain[-1]} has no {metric!r} column")
            continue
        rates = rates_fn(df, metric)
        if rates.empty:
            warn(f"group '{label}': {chain[-1]} has no non-empty {metric!r} value")
            continue
        per_series.append(rates)
    if not per_series:
        warn(f"group '{label}' produced no {metric!r} rows")
        return None
    stacked = pd.concat([s.assign(series=i) for i, s in enumerate(per_series)])
    agg = (stacked.groupby(key)
                      .agg(mean=("rate", "mean"),
                           n_series=("series", "nunique"),
                           n_trials=("n_trials", "sum"))
                      .reset_index())
    return agg.assign(series_total=len(per_series))


def render_bars(table: pd.DataFrame, color_of: dict, out_path: Path, *,
                metric: str, x_col: str = "position",
                x_label: str = "position in sequence", x_rotate: float = 0.0,
                x_tick_fmt=str, x_tick_pt: float = TICK_LABEL_PT,
                figsize=BAR_FIGSIZE, box_aspect: float = CURVE_BOX_ASPECT,
                legend_outside: bool = False):
    """One bar chart: x = `x_col` (the position in the round, or the task), one
    bar per group at each of its values.

    Both domains of `table` go in the same bar (out-of-domain solid, in-domain
    hatched on top); a one-domain `table` draws solid bars, since nothing has
    to be told apart inside the bar.

    `table` must hold at most one row per (group, eval_kind, x) — the caller
    picks the task, or the scene, the figure is about.

    `color_of` maps every group to its colour (so a group keeps its hue across
    figures); a group with no row in `table` gets no bar slot and no legend
    entry. Returns None, writing nothing, when `table` is empty.
    """
    present = set(table.dropna(subset=["mean"])["group"])
    groups = [g for g in color_of if g in present]
    if not groups:
        return None
    table = table[table["group"].isin(groups)]
    colors = [color_of[g] for g in groups]
    domains = [d for d in _DOMAINS if d in set(table["eval_kind"])]
    xvals = sorted(table[x_col].unique())
    n_bars = len(groups)
    width = 0.8 / max(1, n_bars)
    fig, ax = plt.subplots(figsize=figsize)
    x0 = np.arange(len(xvals), dtype=float)

    for gi, (group, color) in enumerate(zip(groups, colors)):
        for domain in domains:
            sub = (table[(table["group"] == group) & (table["eval_kind"] == domain)]
                   .set_index(x_col).reindex(xvals))
            offset = (gi - (n_bars - 1) / 2) * width
            # The hatch only has to tell the two domains apart where they
            # share a bar; a one-domain figure draws its bars solid.
            hatched = domain == "in_domain" and len(domains) > 1
            # An x this (group, domain) never ran gets no bar rather
            # than a zero-height one, which would read as "always failed".
            ran = sub["mean"].notna().to_numpy()
            if not ran.any():
                continue
            xs = (x0 + offset)[ran]
            ys = sub["mean"].to_numpy(dtype=float)[ran]
            if hatched:
                ax.bar(xs, ys, width=width * 0.92, facecolor="none",
                       edgecolor=hatch_color(color), hatch=IN_DOMAIN_HATCH,
                       linewidth=BAR_EDGE_WIDTH, zorder=3)
            else:
                ax.bar(xs, ys, width=width * 0.92, color=color, linewidth=0,
                       zorder=2)
            # A measured 0 is a flat bar the axis hides; mark it so it reads
            # as "always failed", distinct from an absent bar (not run).
            zero = ys == 0.0
            if zero.any():
                ax.plot(xs[zero], ys[zero], linestyle="none", marker="_",
                        markersize=max(4.0, 260 * width * 0.92 / len(xvals)),
                        markeredgewidth=2.4, zorder=5, clip_on=False,
                        color=hatch_color(color) if hatched else color)

    ax.set_xticks(x0, [x_tick_fmt(v) for v in xvals])
    if x_rotate:
        # Task names are sentences ("put carrot on plate"): upright they
        # overlap, so they lean and end under their own tick.
        plt.setp(ax.get_xticklabels(), rotation=x_rotate, ha="right",
                 rotation_mode="anchor")
    ax.set_xlabel(x_label, fontsize=AXIS_LABEL_PT)
    ax.set_ylabel(metric_axis_label(metric), fontsize=AXIS_LABEL_PT)
    ax.tick_params(labelsize=TICK_LABEL_PT)
    ax.tick_params(axis="x", labelsize=x_tick_pt)
    ax.set_ylim(*BAR_YLIM)
    ax.grid(axis="y", alpha=CURVE_GRID_ALPHA, zorder=0)
    ax.set_axisbelow(True)
    # Same 4:3 box as the curve figures — `figsize` alone does not give one,
    # since the margins depend on how wide the tick labels come out.
    ax.set_box_aspect(box_aspect)

    # Two keys: colour = group (omitted for a single group, where it would label
    # the only hue), fill = domain (omitted on a one-domain figure, whose legend
    # title already names it — unless nothing else would be in the legend).
    from matplotlib.patches import Patch
    group_handles = ([Patch(facecolor=c, edgecolor=c, label=g)
                      for g, c in zip(groups, colors)] if len(groups) > 1 else [])
    domain_handles = [Patch(facecolor="none", edgecolor="0.3", hatch=IN_DOMAIN_HATCH,
                            linewidth=BAR_EDGE_WIDTH, label=_DOMAIN_LABEL[d])
                      if d == "in_domain" else
                      Patch(facecolor="0.55", linewidth=0, label=_DOMAIN_LABEL[d])
                      for d in domains]
    handles = group_handles + (domain_handles
                               if len(domains) > 1 or not group_handles else [])
    # Inside the axes, upper right: success falls with position, so the right
    # of the panel is the empty corner. `legend_outside` moves it clear of the
    # box instead, for a panel whose bars reach the top right (a scene's four
    # tasks). No title — the figure is named by its filename and its caption.
    # A notch under the shared `legend_pt`: the key sits inside the panel and
    # carries up to four group labels plus the domain fills, so the full 15pt
    # covered the bars it sits over — but 8pt was too small to read.
    if legend_outside:
        ax.legend(handles=handles, loc="upper left",
                  bbox_to_anchor=(1.01, 1.0), borderaxespad=0.0,
                  fontsize=SEQ_LEGEND_PT, frameon=False)
    else:
        ax.legend(handles=handles, loc="upper right", fontsize=SEQ_LEGEND_PT,
                  frameon=False)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=CURVE_DPI, bbox_inches="tight")
    plt.close(fig)
    return out_path


def group_colors(table: pd.DataFrame, cfg: PlotConfig) -> dict:
    """group -> colour, assigned once over every group with any data, so a
    group keeps its hue in every figure even where another group is skipped.
    Both figure series carry the same groups, so both get the same map."""
    groups = [g.label for g in cfg.groups if g.label in set(table["group"])]
    return dict(zip(groups, default_colors(len(groups))))


def figure_kinds(table: pd.DataFrame) -> list:
    """The round kinds in `table`, in the order their figures are written."""
    kinds = [k for k in (*_SEQ_KINDS, POOLED) if k in set(table["seq_kind"])]
    return kinds + sorted(set(table["seq_kind"]) - set(kinds))   # e.g. 'unknown'


def render_all(table: pd.DataFrame, cfg: PlotConfig, *, metrics,
               per_task: bool) -> list:
    """Every figure. Per metric and round kind (`seq_kind`):

        both domains in one bar, pooled     <name>_seq_position_<metric>_<kind>_both.png
        one per domain, pooled over tasks   <name>_seq_position_<metric>_<kind>_<domain>.png
        one per task, both domains in it    <name>_seq_position_<metric>_<kind>_per_task/...

    The pooled figures answer "does this domain lose success with position",
    which is read one domain at a time; the per-task ones carry both domains in
    one bar, so a task's in-domain / out-of-domain gap is read without paging
    between figures.
    """
    color_of = group_colors(table, cfg)
    tasks = sorted(t for t in table["task"].unique() if t != ALL_TASKS)
    slugs = unique_slugs(tasks)
    kinds = figure_kinds(table)

    written = []
    for metric in metrics:
        for kind in kinds:
            kt = table[(table["metric"] == metric) & (table["seq_kind"] == kind)]
            if kt.empty:
                continue
            tag = SEQ_LABEL.get(kind, kind)
            stem = f"{cfg.name}_seq_position_{metric}_{tag}"
            pooled = kt[kt["task"] == ALL_TASKS]
            # Both domains in one bar: the in-domain / out-of-domain gap is
            # read here, the per-domain panels below without the second set of
            # bars halving the width.
            written.append(render_bars(
                pooled, color_of, cfg.out_dir / f"{stem}_both.png",
                metric=metric))
            for domain in _DOMAINS:
                dt = pooled[pooled["eval_kind"] == domain]
                if dt.empty:
                    continue
                written.append(render_bars(
                    dt, color_of, cfg.out_dir / f"{stem}_{domain}.png",
                    metric=metric))
            if per_task:
                task_dir = cfg.out_dir / f"{stem}_per_task" / f"{stem}.png"
                for task in tasks:
                    # The task goes in the legend title: no suptitle, as in the
                    # other tools, but a folder of 16 bar charts must still say
                    # which is which.
                    written.append(render_bars(
                        kt[kt["task"] == task], color_of,
                        out_variant(task_dir, slugs[task]),
                        metric=metric))
    return [w for w in written if w is not None]


# ---------------------------------------------------------------------------
# The checkpoint's own last training eval
# ---------------------------------------------------------------------------


def final_eval_source(run_dir: Path, override=None):
    """(training glob dir, episode) whose eval produced this checkpoint.

    The standalone eval records which checkpoint it loaded; that checkpoint
    sits inside the training run's `glob/`, next to the `eval_success.csv`
    holding the rotation eval of every episode — including its own. A
    checkpoint COPIED somewhere else (a checkpoint bundle) has no such
    neighbour, which is what `final_eval_runs` in the config is for.
    """
    rc = Path(run_dir) / "run_config.json"
    if not rc.exists():
        return None, None
    load = str(json.loads(rc.read_text()).get("vla_load_path") or "").rstrip("/")
    m = re.search(r"episode_(\d+)$", load)
    if not m:
        return None, None
    episode = int(m.group(1))
    home = Path(override) if override else Path(load).parent
    return (home if (home / "eval_success.csv").exists() else None), episode


def final_eval_table(cfg: PlotConfig, *, required: bool) -> pd.DataFrame:
    """Per (group, scene, task, domain): the rate the checkpoint's own last
    training eval measured, averaged over the group's series.

    One row per bar of the `_final_eval_task` figures. Empty when no series
    could be resolved — the caller then writes no figure.
    """
    overrides = cfg.option("final_eval_runs", None, {}) or {}
    rows = []
    for group in cfg.groups:
        over = overrides.get(group.label) or []
        for i, chain in enumerate(group.chains):
            run_dir = Path(chain[-1])
            home, episode = final_eval_source(
                run_dir, over[i] if i < len(over) else None)
            if home is None:
                warn(f"group '{group.label}': {run_dir} — no training "
                     f"eval_success.csv for its checkpoint; set "
                     f"`final_eval_runs` to point at one")
                continue
            df = read_table(home / "eval_success.csv", what="eval_success.csv",
                            required_cols=("episode", "eval_kind", "group",
                                           "task", "success"))
            df = df[df["episode"].astype(int) == episode]
            if df.empty:
                warn(f"group '{group.label}': {home}/eval_success.csv has no "
                     f"episode {episode}")
                continue
            rows.append(df.assign(series=i, label=group.label))
    if not rows:
        if required:
            raise SystemExit("[seq] no series resolved a training eval")
        return pd.DataFrame(columns=["group", "eval_kind", "scene", "task",
                                     "mean", "std", "n_series"])
    df = pd.concat(rows, ignore_index=True)
    per_series = (df.groupby(["label", "eval_kind", "group", "task", "series"],
                             as_index=False)["success"].mean())
    agg = (per_series.groupby(["label", "eval_kind", "group", "task"])
           .agg(mean=("success", "mean"),
                std=("success", lambda v: v.std(ddof=1) if len(v) > 1 else np.nan),
                n_series=("series", "nunique")).reset_index())
    return agg.rename(columns={"label": "group", "group": "scene"})


def render_final_eval(table: pd.DataFrame, cfg: PlotConfig, color_of: dict, *,
                      per_scene: bool) -> list:
    """The `_final_eval_task` figures: x = task, both domains in one bar.

        every scene's tasks      <name>_final_eval_task.png
        one scene each           <name>_final_eval_task_per_scene/...
    """
    if table.empty:
        return []
    written = [render_bars(
        table, color_of, cfg.out_dir / f"{cfg.name}_final_eval_task.png",
        metric="success", x_col="task", x_label="task",
        x_tick_fmt=task_tick_label, x_tick_pt=TASK_TICK_PT,
        figsize=TASK_FIGSIZE, box_aspect=TASK_BOX_ASPECT)]
    if per_scene:
        scenes = sorted(table["scene"].unique())
        slugs = unique_slugs(scenes)
        scene_dir = (cfg.out_dir / f"{cfg.name}_final_eval_task_per_scene"
                     / f"{cfg.name}_final_eval_task.png")
        for scene in scenes:
            written.append(render_bars(
                table[table["scene"] == scene], color_of,
                out_variant(scene_dir, slugs[scene]), metric="success",
                x_col="task", x_label="task", x_tick_fmt=task_tick_label,
                legend_outside=True))
    return [w for w in written if w is not None]


def render_task_all(table: pd.DataFrame, cfg: PlotConfig, *,
                    per_scene: bool = True) -> list:
    """The position-free figures — ONE of them, plus a scene breakdown:

        x = task, both domains in one bar   <name>_seq_task.png
        the same, one scene each            <name>_seq_task_per_scene/...

    Which task is hard does not depend on where in the round it was met, so
    this family pools every round kind and shows plain `success` only: the
    seen / unseen split and the chained metric are questions about position,
    and they are answered by the `_seq_position` figures.
    """
    color_of = group_colors(table, cfg)
    scenes = sorted(s for s in table["scene"].unique() if s != ALL_SCENES)
    slugs = unique_slugs(scenes)

    written = []
    for metric in (TASK_METRIC,):
        for kind in ("",):
            kt = table[table["metric"] == metric]
            if kt.empty:
                continue
            stem = f"{cfg.name}_seq_task"
            written.append(render_bars(
                kt[kt["scene"] == ALL_SCENES], color_of,
                cfg.out_dir / f"{stem}.png", metric=metric,
                x_col="task", x_label="task",
                x_tick_fmt=task_tick_label, x_tick_pt=TASK_TICK_PT,
                figsize=TASK_FIGSIZE, box_aspect=TASK_BOX_ASPECT))
            if per_scene:
                scene_dir = cfg.out_dir / f"{stem}_per_scene" / f"{stem}.png"
                for scene in scenes:
                    # One scene's four tasks, same style as the pooled figure.
                    written.append(render_bars(
                        kt[kt["scene"] == scene], color_of,
                        out_variant(scene_dir, slugs[scene]), metric=metric,
                        x_col="task", x_label="task",
                        x_tick_fmt=task_tick_label, x_tick_pt=TASK_TICK_PT,
                        legend_outside=True))
    return [w for w in written if w is not None]


def report(table: pd.DataFrame) -> None:
    """The average figure's numbers on stderr, one line per (group, domain)."""
    avg = table[table["task"] == ALL_TASKS]
    for (metric, kind, group, domain), sub in avg.groupby(
            ["metric", "seq_kind", "group", "eval_kind"], sort=False):
        cells = "  ".join(f"p{int(r.position)}={r.mean:.3f}"
                          for r in sub.sort_values("position").itertuples())
        print(f"[seq] {metric:<16s} {SEQ_LABEL.get(kind, kind):<7s} {group} / "
              f"{_DOMAIN_LABEL.get(domain, domain):<13s} {cells}", file=sys.stderr)


def report_task(table: pd.DataFrame) -> None:
    """The pooled-over-scenes task rates on stderr, one line per
    (group, domain)."""
    avg = table[table["scene"] == ALL_SCENES]
    for (metric, kind, group, domain), sub in avg.groupby(
            ["metric", "seq_kind", "group", "eval_kind"], sort=False):
        cells = "  ".join(f"{r.task}={r.mean:.3f}"
                          for r in sub.sort_values("task").itertuples())
        print(f"[task] {metric:<16s} {SEQ_LABEL.get(kind, kind):<7s} {group} / "
              f"{_DOMAIN_LABEL.get(domain, domain):<13s} {cells}", file=sys.stderr)


def main():
    p = argparse.ArgumentParser(
        "plot_sequence_eval",
        description="Sequence-eval success by task position, and by task "
                    "whatever its position (bars), from eval_per_trial.csv",
    )
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--run-dir", help="the eval's glob dir, holding eval_per_trial.csv")
    src.add_argument("--config", help="JSON describing several groups of eval runs "
                                      "(see tools/plot_common.py); one bar colour per group")
    p.add_argument("--out-dir", default=None,
                   help="where the figures go (default: the run dir, or the "
                        "config's out_dir)")
    p.add_argument("--name", default=None,
                   help="filename prefix (default: 'eval', or the config's name)")
    p.add_argument("--metric", default=None, nargs="+", choices=list(_METRICS),
                   help="one or more; default: success success_chained (each "
                        "task on its own, and failed-once-failed-after)")
    p.add_argument("--seq-kind", default=None,
                   choices=["each", POOLED, *_SEQ_KINDS, *SEQ_ALIASES],
                   help="which rounds to use: 'each' (default) gives the seen "
                        "(training) and unseen (random) orders their own "
                        "figures, 'pooled' puts them in one set, or name one "
                        "kind — 'seen'/'training' or 'unseen'/'random'")
    p.add_argument("--no-per-task", dest="per_task", action="store_false",
                   help="write only the average position figure")
    p.add_argument("--no-final-eval", dest="final_eval", action="store_false",
                   help="skip the figures of the checkpoint's own last "
                        "training eval (single-task success, per scene)")
    p.add_argument("--no-per-scene", dest="per_scene", action="store_false",
                   help="write only the pooled task figure, without the "
                        "per-scene breakdown (`group` in eval_per_trial.csv)")
    args = p.parse_args()

    if args.config:
        cfg = load_plot_config(args.config)
        required = False
    else:
        run_dir = Path(args.run_dir)
        cfg = PlotConfig(name="eval", out_dir=run_dir,
                         groups=[Group(label=run_dir.parent.name or "eval",
                                       chains=[[run_dir]])])
        required = True
    if args.out_dir:
        cfg.out_dir = Path(args.out_dir).expanduser().resolve()
    if args.name:
        cfg.name = args.name
    metrics = cfg.option("metric", args.metric, list(_DEFAULT_METRICS))
    if isinstance(metrics, str):
        metrics = [metrics]
    metrics = list(dict.fromkeys(metrics))
    bad = [m for m in metrics if m not in _METRICS]
    if bad or not metrics:
        # `metric` is shared with plot_rollout_success.py, whose columns differ.
        raise SystemExit(f"metric {bad or metrics!r} is not in eval_per_trial.csv; "
                         f"choose from {list(_METRICS)} (pass --metric to "
                         f"override the config)")
    seq_kind = cfg.option("seq_kind", args.seq_kind, "each")
    # 'seen' / 'unseen' are what the figures say; 'all' was the pre-split
    # spelling of 'pooled'.
    seq_kind = SEQ_ALIASES.get(seq_kind, seq_kind)
    if seq_kind not in ("each", POOLED, *_SEQ_KINDS):
        raise SystemExit(f"seq_kind {seq_kind!r} is not one of "
                         f"{['each', POOLED, *_SEQ_KINDS]}")

    # One read of the eval files feeds both figure series: success by position,
    # and success by task with the position pooled away.
    series = load_groups(cfg, seq_kind=seq_kind, required=required)
    table = collect(cfg, series, metrics=metrics, rates_fn=position_rates,
                    key=_KEY, pooled=("task", ALL_TASKS))
    # The task family pools the round kinds: which task is hard is not a
    # question about where in the round it was met.
    task_series = {label: [(chain, df.assign(seq_kind=POOLED))
                           for chain, df in entries]
                   for label, entries in series.items()}
    task_table = collect(cfg, task_series, metrics=[TASK_METRIC],
                         rates_fn=task_rates, key=_TASK_KEY,
                         pooled=("scene", ALL_SCENES))
    print(f"[seq] metrics={metrics}, seq_kind={seq_kind}; bar = mean across "
          f"series", file=sys.stderr)
    report(table)
    report_task(task_table)
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    for path, t in ((cfg.out_dir / f"{cfg.name}_seq_position.csv", table),
                    (cfg.out_dir / f"{cfg.name}_seq_task.csv", task_table)):
        t.to_csv(path, index=False)
        print(f"[ok] wrote {path}", file=sys.stderr)
    written = (render_all(table, cfg, metrics=metrics, per_task=args.per_task)
               + render_task_all(task_table, cfg, per_scene=args.per_scene))
    if args.final_eval:
        # The single-task rates the checkpoint was last measured at in
        # training: the same tasks, evaluated one at a time rather than in a
        # sequence, so the sequence cost is read against them.
        final = final_eval_table(cfg, required=required)
        if not final.empty:
            path = cfg.out_dir / f"{cfg.name}_final_eval_task.csv"
            final.to_csv(path, index=False)
            print(f"[ok] wrote {path}", file=sys.stderr)
            written += render_final_eval(final, cfg, group_colors(table, cfg),
                                         per_scene=True)
    for path in written:
        print(f"[ok] wrote {path}", file=sys.stderr)


if __name__ == "__main__":
    main()
