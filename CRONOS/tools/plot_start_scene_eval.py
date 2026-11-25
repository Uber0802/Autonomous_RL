#!/usr/bin/env python
"""Bar plot of start-scene sequential-eval success, checkpoints side by side.

Reads one `eval_per_trial.csv` (written by `eval_only.py`) per checkpoint and
draws two panels: success rate per group (+ overall) and success rate per task
position within the sequence (+ overall). Bars are grouped per checkpoint so
the same label lands next to itself for every checkpoint.

Usage:
    python tools/plot_start_scene_eval.py \
        --run "T80 ep576=eval_runs/T80/chaos/wandb/offline-run-*/glob/eval_per_trial.csv" \
        --run "T320 ep144=eval_runs/T320/chaos/wandb/offline-run-*/glob/eval_per_trial.csv" \
        --title "Chaos start scene" --out figures/start_scene_eval/chaos.png
"""

import argparse
import csv
import glob
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Categorical slots 1..4 of the reference palette (fixed order, never cycled).
SERIES_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
INK = "#1f1f1f"
INK_MUTED = "#6b6b6b"
GRID = "#e6e6e6"


def load_trials(path_glob, domain=""):
    """Rows of one eval domain (`eval_kind`). domain="" = the only domain in the files."""
    paths = sorted(glob.glob(path_glob))
    if not paths:
        raise FileNotFoundError(f"no file matches {path_glob!r}")
    all_rows = []
    for p in paths:
        with open(p) as f:
            all_rows.extend(csv.DictReader(f))
    kinds = sorted({r["eval_kind"] for r in all_rows})
    if not domain:
        if len(kinds) != 1:
            raise ValueError(f"{path_glob}: several eval domains {kinds}; pick one with --domain")
        domain = kinds[0]
    rows = []
    for row in all_rows:
        if row["eval_kind"] != domain:
            continue
        rows.append({
            "group": row.get("group", "default"),
            "task_idx": int(row["task_idx"]),
            "seq_idx": int(row["seq_idx"]),
            "success": float(row["success"]),
        })
    return rows


def rate(rows):
    if not rows:
        return np.nan, np.nan, 0
    s = np.array([r["success"] for r in rows])
    p = s.mean()
    se = np.sqrt(p * (1 - p) / len(s)) if len(s) > 1 else 0.0
    return p, se, len(s)


def summarize(rows, key):
    by = defaultdict(list)
    for r in rows:
        by[r[key]].append(r)
    keys = sorted(by.keys())
    out = {k: rate(by[k]) for k in keys}
    out["overall"] = rate(rows)
    return keys + ["overall"], out


def draw_panel(ax, categories, per_run, labels, title, xlabel):
    n_runs = len(per_run)
    width = 0.8 / n_runs
    x = np.arange(len(categories))
    for ri, (run_label, stats) in enumerate(zip(labels, per_run)):
        vals = np.array([stats.get(c, (np.nan, np.nan, 0))[0] for c in categories])
        errs = np.array([stats.get(c, (np.nan, np.nan, 0))[1] for c in categories])
        offs = (ri - (n_runs - 1) / 2) * width
        bars = ax.bar(x + offs, vals, width * 0.92, color=SERIES_COLORS[ri],
                      label=run_label, yerr=errs, error_kw=dict(ecolor=INK_MUTED, lw=1, capsize=2),
                      linewidth=0)
        for b, v, e in zip(bars, vals, errs):
            if np.isfinite(v):
                ax.text(b.get_x() + b.get_width() / 2, v + (e if np.isfinite(e) else 0) + 0.015, f"{v:.2f}",
                        ha="center", va="bottom", fontsize=8, color=INK)
    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=9, color=INK)
    ax.set_ylim(0, 1.05)
    ax.set_yticks(np.linspace(0, 1, 6))
    ax.set_ylabel("success rate", color=INK)
    ax.set_xlabel(xlabel, color=INK_MUTED)
    ax.set_title(title, loc="left", fontsize=11, color=INK)
    ax.yaxis.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK_MUTED)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", action="append", required=True,
                    help="LABEL=GLOB_TO_eval_per_trial.csv (repeat once per checkpoint)")
    ap.add_argument("--title", default="Sequential eval")
    ap.add_argument("--domain", default="", help="eval domain to plot (default: the only one in each file)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    labels, per_group, per_pos, counts = [], [], [], []
    group_cats = pos_cats = None
    for spec in args.run:
        label, path_glob = spec.split("=", 1)
        rows = load_trials(path_glob, args.domain)
        cats_g, stats_g = summarize(rows, "group")
        cats_p, stats_p = summarize(rows, "task_idx")
        group_cats = sorted((set(group_cats or []) | set(cats_g)) - {"overall"}) + ["overall"]
        pos_cats = sorted((set(pos_cats or []) | set(cats_p)) - {"overall"}) + ["overall"]
        labels.append(label)
        per_group.append(stats_g)
        per_pos.append(stats_p)
        counts.append(len(rows))

    pos_names = [f"task {c}" if c != "overall" else "overall" for c in pos_cats]
    per_pos_named = [{(f"task {k}" if k != "overall" else "overall"): v for k, v in s.items()} for s in per_pos]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    draw_panel(axes[0], group_cats, per_group, labels, "by group", "group (all tasks, all sequences)")
    draw_panel(axes[1], pos_names, per_pos_named, labels, "by position in sequence",
               "task position (0 = first task after the start-scene perturbation)")
    handles, lbls = axes[0].get_legend_handles_labels()
    fig.legend(handles, lbls, loc="upper right", ncol=len(lbls), frameon=False, fontsize=9)
    n_txt = ", ".join(f"{l}: n={c}" for l, c in zip(labels, counts))
    fig.suptitle(args.title, x=0.01, ha="left", fontsize=13, color=INK, fontweight="bold")
    fig.text(0.01, 0.905, f"in-domain trials per checkpoint — {n_txt}; error bars = 1 s.e.",
             fontsize=8.5, color=INK_MUTED)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
