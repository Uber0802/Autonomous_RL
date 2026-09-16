"""Sequence-eval success by task position, as bars, from `eval_per_trial.csv`.

A sequential eval round runs every task slot in a row without resetting, so the
same task is met at position 1, 2, 3 and 4 (`task_idx` 0..3) across the round's
orders — see `doc/eval_sequential.md` §3. This tool asks the one question those
files are laid out for: **does the success rate depend on where in the sequence
the task sits?**

    python tools/plot_sequence_eval.py --run-dir <EVAL_OUT_DIR>/wandb/run-*/glob
    python tools/plot_sequence_eval.py --config tools/plot_runs_example.json

Figures (one PNG each, never a grid — the convention of the other plot tools):

    <name>_seq_position.png                         average over every task
    <name>_seq_position_per_task/<name>_seq_position_<task>.png
                                                    one per task

x = position in the sequence, one bar per (group, domain) at each position.
**In-domain bars are filled, out-of-domain bars hollow**, in the group's colour,
so a group keeps one hue and the domain is read from the fill alone. Bars are
the mean over a group's series (seeds); the error bar is ±1 std across them and
is drawn only when a group has more than one series.

`<name>_seq_position.csv` carries every plotted number, plus `n_trials`.

Metric (`--metric`, config `metric`):

    success           each task judged on its own (default) — the position effect
    success_chained   cumulative AND along the round: position k succeeds only if
                      positions 1..k all did, so it can only fall with position
    grasp, obj_grasped  latched within the task slot

`--seq-kind` (config `seq_kind`) keeps `training` rounds (the rotations training
ran), `random` rounds (untrained cycles) or `all` (default). Pooling them is the
full 24-order design; the split answers whether an untrained order costs more.

The average figure pools trials, not task means: with every task at every
position equally often (a complete design) the two are identical, and on an
incomplete eval pooling weights each task by what was actually run.

Missing input follows the other tools: in `--config` mode a run without a usable
`eval_per_trial.csv` is warned about and skipped; `--run-dir` fails loudly.
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
from plot_common import (CURVE_DPI, CURVE_GRID_ALPHA, Group, NoData,  # noqa: E402
                         PlotConfig, default_colors, load_plot_config,
                         out_variant, read_table, unique_slugs, warn)

_METRICS = ("success", "success_chained", "grasp", "obj_grasped")
_DOMAINS = ("in_domain", "out_of_domain")
_DOMAIN_LABEL = {"in_domain": "in-domain", "out_of_domain": "out-of-domain"}
_TRIAL_KEY = ["eval_kind", "seq_idx", "task_idx", "env_idx"]
ALL_TASKS = "__all__"

BAR_FIGSIZE = (6.4, 4.8)
BAR_YLIM = (0.0, 1.02)
BAR_EDGE_WIDTH = 1.6


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
    for col in _METRICS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def load_series(chain, *, label: str, required: bool) -> pd.DataFrame:
    """A series = one run, or several shards of one eval stitched together.

    A unit that two shards both hold keeps the later shard's rows, the same
    "later wins" rule as a training resume chain; `tools/rebuild_eval_outputs.py`
    refuses that case outright, so it only arises for hand-assembled chains.
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
    return df.drop_duplicates(subset=_TRIAL_KEY, keep="last")


def position_rates(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Per-series rates: (eval_kind, task, position) -> mean, plus the pooled
    all-task row under `task = ALL_TASKS`."""
    d = df.dropna(subset=[metric]).assign(position=lambda x: x["task_idx"] + 1)
    per_task = (d.groupby(["eval_kind", "task", "position"])[metric]
                 .agg(rate="mean", n_trials="size").reset_index())
    pooled = (d.groupby(["eval_kind", "position"])[metric]
               .agg(rate="mean", n_trials="size").reset_index()
               .assign(task=ALL_TASKS))
    return pd.concat([pooled, per_task], ignore_index=True)


def collect(cfg: PlotConfig, *, metric: str, seq_kind: str,
            required: bool) -> pd.DataFrame:
    """Every group's rates, aggregated over its series.

    Columns: group, eval_kind, task, position, mean, std, n_series, n_trials.
    """
    rows = []
    for group in cfg.groups:
        per_series = []
        for chain in group.chains:
            df = load_series(chain, label=group.label, required=required)
            if df.empty:
                continue
            if metric not in df.columns:
                warn(f"group '{group.label}': {chain[-1]} has no {metric!r} column")
                continue
            if seq_kind != "all":
                kept = df[df["seq_kind"] == seq_kind]
                if kept.empty:
                    warn(f"group '{group.label}': {chain[-1]} has no "
                         f"seq_kind={seq_kind!r} rows (present: "
                         f"{sorted(df['seq_kind'].unique())})")
                    continue
                df = kept
            per_series.append(position_rates(df, metric))
        if not per_series:
            warn(f"group '{group.label}' produced no rows")
            continue
        stacked = pd.concat([s.assign(series=i) for i, s in enumerate(per_series)])
        agg = (stacked.groupby(["eval_kind", "task", "position"])
                      .agg(mean=("rate", "mean"),
                           # ddof=0 would report a spread of 0 for one series;
                           # NaN is what "not measured" is, and draws no bar.
                           std=("rate", lambda v: v.std(ddof=1) if len(v) > 1 else np.nan),
                           n_series=("series", "nunique"),
                           n_trials=("n_trials", "sum"))
                      .reset_index())
        rows.append(agg.assign(group=group.label))
        print(f"[group] {group.label:<34s} {len(per_series)} series, "
              f"{int(agg.loc[agg['task'] == ALL_TASKS, 'n_trials'].sum())} trials",
              file=sys.stderr)
    if not rows:
        raise SystemExit(
            "[seq] no group produced any rows — nothing to plot.\n"
            "  The [warn] lines above name the groups. Each `runs` entry must be a\n"
            "  standalone eval's glob/ dir holding eval_per_trial.csv (training\n"
            "  runs have none), and --seq-kind / --metric must exist in it.")
    cols = ["group", "eval_kind", "task", "position", "mean", "std",
            "n_series", "n_trials"]
    return pd.concat(rows, ignore_index=True)[cols]


def render_bars(table: pd.DataFrame, groups, colors, out_path: Path, *,
                metric: str, legend_title=None) -> Path:
    """One bar chart: x = position, one bar per (group, domain) at each."""
    domains = [d for d in _DOMAINS if d in set(table["eval_kind"])]
    positions = sorted(table["position"].unique())
    n_bars = len(groups) * len(domains)
    width = 0.8 / max(1, n_bars)
    fig, ax = plt.subplots(figsize=BAR_FIGSIZE)
    x0 = np.arange(len(positions), dtype=float)

    for gi, (group, color) in enumerate(zip(groups, colors)):
        for di, domain in enumerate(domains):
            sub = (table[(table["group"] == group) & (table["eval_kind"] == domain)]
                   .set_index("position").reindex(positions))
            offset = (gi * len(domains) + di - (n_bars - 1) / 2) * width
            filled = domain == "in_domain"
            # A position this (group, domain) never ran gets no bar rather
            # than a zero-height one, which would read as "always failed".
            ran = sub["mean"].notna().to_numpy()
            yerr = sub["std"].to_numpy(dtype=float)[ran]
            ax.bar((x0 + offset)[ran], sub["mean"].to_numpy(dtype=float)[ran],
                   width=width * 0.92,
                   color=color if filled else "white", edgecolor=color,
                   linewidth=BAR_EDGE_WIDTH, zorder=2,
                   yerr=None if np.isnan(yerr).all() else np.nan_to_num(yerr),
                   error_kw={"ecolor": "0.25", "elinewidth": 1.0, "capsize": 2.5})

    ax.set_xticks(x0, [str(p) for p in positions])
    ax.set_xlabel("position in sequence")
    ax.set_ylabel(metric.replace("_", " "))
    ax.set_ylim(*BAR_YLIM)
    ax.grid(axis="y", alpha=CURVE_GRID_ALPHA, zorder=0)
    ax.set_axisbelow(True)

    # Two keys: colour = group (omitted for a single group, where it would label
    # the only hue), fill = domain.
    from matplotlib.patches import Patch
    handles = [Patch(facecolor="0.45", edgecolor="0.45", linewidth=BAR_EDGE_WIDTH,
                     label=_DOMAIN_LABEL[d]) if d == "in_domain" else
               Patch(facecolor="white", edgecolor="0.45", linewidth=BAR_EDGE_WIDTH,
                     label=_DOMAIN_LABEL[d]) for d in domains]
    if len(groups) > 1:
        handles = [Patch(facecolor=c, edgecolor=c, label=g)
                   for g, c in zip(groups, colors)] + handles
    # Outside the axes: bars reach 1.0, and no corner is reliably empty.
    ax.legend(handles=handles, title=legend_title, loc="upper left",
              bbox_to_anchor=(1.01, 1.0), fontsize=8, title_fontsize=8,
              frameon=False)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=CURVE_DPI, bbox_inches="tight")
    plt.close(fig)
    return out_path


def render_all(table: pd.DataFrame, cfg: PlotConfig, *, metric: str,
               per_task: bool) -> list:
    groups = [g.label for g in cfg.groups if g.label in set(table["group"])]
    colors = default_colors(len(groups))
    out_base = cfg.out_dir / f"{cfg.name}_seq_position.png"

    written = [render_bars(table[table["task"] == ALL_TASKS], groups, colors,
                           out_base, metric=metric)]
    if per_task:
        tasks = sorted(t for t in table["task"].unique() if t != ALL_TASKS)
        slugs = unique_slugs(tasks)
        task_dir = out_base.with_name(f"{out_base.stem}_per_task") / out_base.name
        for task in tasks:
            # The task goes in the legend title: no suptitle, as in the other
            # tools, but a folder of 16 bar charts must still say which is which.
            written.append(render_bars(table[table["task"] == task], groups,
                                       colors, out_variant(task_dir, slugs[task]),
                                       metric=metric, legend_title=task))
    return written


def report(table: pd.DataFrame) -> None:
    """The average figure's numbers on stderr, one line per (group, domain)."""
    avg = table[table["task"] == ALL_TASKS]
    for (group, domain), sub in avg.groupby(["group", "eval_kind"], sort=False):
        cells = "  ".join(f"p{int(r.position)}={r.mean:.3f}"
                          for r in sub.sort_values("position").itertuples())
        print(f"[seq] {group} / {_DOMAIN_LABEL.get(domain, domain):<13s} {cells}",
              file=sys.stderr)


def main():
    p = argparse.ArgumentParser(
        "plot_sequence_eval",
        description="Sequence-eval success by task position (bars) from eval_per_trial.csv",
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
    p.add_argument("--metric", default=None, choices=list(_METRICS),
                   help="default success (each task judged on its own)")
    p.add_argument("--seq-kind", default=None, choices=["all", "training", "random"],
                   help="which rounds to use (default all)")
    p.add_argument("--no-per-task", dest="per_task", action="store_false",
                   help="write only the average figure")
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
    metric = cfg.option("metric", args.metric, "success")
    if metric not in _METRICS:
        # `metric` is shared with plot_rollout_success.py, whose columns differ.
        raise SystemExit(f"metric {metric!r} is not in eval_per_trial.csv; "
                         f"choose from {list(_METRICS)} (pass --metric to "
                         f"override the config)")
    seq_kind = cfg.option("seq_kind", args.seq_kind, "all")

    table = collect(cfg, metric=metric, seq_kind=seq_kind, required=required)
    print(f"[seq] metric={metric}, seq_kind={seq_kind}; error bar = ±1 std "
          f"across series", file=sys.stderr)
    report(table)
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = cfg.out_dir / f"{cfg.name}_seq_position.csv"
    table.to_csv(csv_path, index=False)
    print(f"[ok] wrote {csv_path}", file=sys.stderr)
    for path in render_all(table, cfg, metric=metric, per_task=args.per_task):
        print(f"[ok] wrote {path}", file=sys.stderr)


if __name__ == "__main__":
    main()
