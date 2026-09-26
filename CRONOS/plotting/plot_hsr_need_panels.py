"""HSR-need detail: obj / recep / env counts per segment (panels) and per episode (tables).

Counts at every segment end how many task (scope=task) or all (scope=all)
obj / recep actors are below z=0.7, and how many envs have at least one — see
`hsr_need_common.py` for the definitions and how they were validated.

    python plotting/plot_hsr_need_panels.py --config history/paper_plot_configs/hsr_need_config.json

Writes to the config's `out_dir`:
  <name>_hsr_need_per_segment.png        2 x 3 panels: scope (task / all) x
                                         (obj, recep, obj OR recep envs);
                                         series mean, band = series min-max
  <name>_hsr_need_per_episode_<stat>.png episode table, stat in mean / end / max / sum
  <name>_hsr_need_per_episode_table.csv  those tables (series mean) in one CSV
  <name>_hsr_need_per_segment.csv, <name>_hsr_need_per_episode.csv

An "episode" is a different length for different horizons, so the tables only
hold groups with the SAME segments-per-episode as the first group (others are
named on stderr); `--table-groups` picks them explicitly.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from hsr_need_common import load_groups, per_episode, write_tables  # noqa: E402
from plot_common import default_colors, load_plot_config, warn  # noqa: E402

SCOPES = [("task", "task-relevant"), ("all", "all obj / recep")]
PANEL_METRICS = [("obj_low", "obj fallen", "obj"), ("recep_low", "recep fallen", "recep"),
                 ("env_flag", "obj OR recep (envs HSR would reset)", "obj|recep")]
STATS = {"mean": ("mean_per_seg", "mean per segment"), "end": ("end", "end of episode"),
         "max": ("max", "max over the episode"), "sum": ("sum", "sum over the episode")}


def panels(seg, cfg, out: Path):
    colors = dict(zip([g.label for g in cfg.groups], default_colors(len(cfg.groups))))
    x_end = seg.total_steps.max() / 1e6
    fig, axes = plt.subplots(2, 3, figsize=(17, 8.2), sharex=True)
    for r, (scope, scope_lbl) in enumerate(SCOPES):
        for c, (m, m_lbl, _) in enumerate(PANEL_METRICS):
            ax = axes[r, c]
            ymax = 0
            for g in cfg.groups:
                d = seg[(seg.group == g.label) & (seg.scope == scope)]
                if d.empty:
                    continue
                per = d.pivot(index="global_segment", columns="series", values=m).sort_index()
                x = d.groupby("global_segment").total_steps.first().loc[per.index] / 1e6
                ax.fill_between(x, per.min(axis=1), per.max(axis=1), color=colors[g.label],
                                alpha=0.18, lw=0)
                ax.plot(x, per.mean(axis=1), color=colors[g.label], lw=1.2, label=g.label)
                ymax = max(ymax, int(d.n_envs.max()) * (2 if scope == "all" and m != "env_flag" else 1))
            ax.set_ylim(0, ymax * 1.03)
            ax.set_xlim(0, x_end)
            ax.set_title(f"{scope_lbl}: {m_lbl}", fontsize=11)
            ax.grid(axis="y", color="0.9", lw=0.6)
            ax.spines[["top", "right"]].set_visible(False)
            if c == 0:
                ax.set_ylabel(f"count at segment end (max {ymax})")
            if r == 1:
                ax.set_xlabel("environment steps (M)")
    h, lbl = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, lbl, loc="lower center", ncol=len(h), fontsize=10, frameon=False,
               bbox_to_anchor=(0.5, 0.0), title="line = series mean, band = series min-max",
               title_fontsize=9)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"[hsr] wrote {out}", file=sys.stderr)


def table_groups(seg, cfg, requested):
    if requested:
        labels = [s.strip() for s in requested.split(",") if s.strip()]
        missing = set(labels) - set(seg.group)
        if missing:
            raise SystemExit(f"--table-groups: unknown group(s) {sorted(missing)}")
        return labels
    spe = seg.groupby("group").seg_per_ep.first()
    first = spe[cfg.groups[0].label]
    keep = [g.label for g in cfg.groups if spe.get(g.label) == first]
    skipped = [g.label for g in cfg.groups if g.label not in keep and g.label in spe]
    if skipped:
        warn(f"episode tables leave out {skipped}: their episodes are not {first} segments long "
             f"(use --table-groups to choose)")
    return keep


def tables(seg, cfg, out_dir: Path, groups, n_eps):
    ep = per_episode(seg)
    mean = ep.groupby(["group", "scope", "episode"]).mean(numeric_only=True).reset_index()
    spe = int(seg[seg.group == groups[0]].seg_per_ep.iloc[0])
    steps_per_ep = float(seg.total_steps.iloc[0]) / float(seg.global_segment.iloc[0]) * spe
    colors = default_colors(len(cfg.groups))
    tint = {g.label: colors[i] for i, g in enumerate(cfg.groups)}
    long = []
    for stat, (suffix, stat_lbl) in STATS.items():
        cols, header = {}, []
        for scope, _ in SCOPES:
            for m, _, short in PANEL_METRICS:
                for g in groups:
                    s = mean[(mean.group == g) & (mean.scope == scope)].set_index("episode")[f"{m}_{suffix}"]
                    cols[f"{scope} | {short} | {g}"] = s.reindex(range(1, n_eps + 1))
                    header.append((scope, short, g))
        t = pd.DataFrame(cols)
        t.index.name = "episode"
        t.insert(0, "steps_end_M", [round(e * steps_per_ep / 1e6, 2) for e in t.index])
        long.append(t.assign(stat=stat))

        cell = [[str(e), f"{t.loc[e, 'steps_end_M']:.2f}"] +
                ["—" if pd.isna(v) else f"{v:.1f}" for v in t.loc[e].values[1:]] for e in t.index]
        fig, ax = plt.subplots(figsize=(max(10, 2 + 1.25 * len(header)), 1.6 + 0.3 * n_eps))
        ax.axis("off")
        tb = ax.table(cellText=cell, colLabels=["ep", "steps(M)"] + [f"{a}\n{b}\n{c}" for a, b, c in header],
                      loc="center", cellLoc="right")
        tb.auto_set_font_size(False)
        tb.set_fontsize(8.5)
        tb.scale(1, 1.35)
        for (i, j), c in tb.get_celld().items():
            c.set_edgecolor("0.85")
            if i == 0:
                c.set_height(c.get_height() * 2.6)
                c.set_text_props(fontweight="bold", ha="center")
                if j >= 2:
                    c.set_facecolor((*tint[header[j - 2][2]][:3], 0.18))
            elif i % 2 == 0:
                c.set_facecolor("0.965")
        ax.set_title(f"HSR position at segment end, per episode: {stat_lbl} (mean over series); "
                     f"— = episode not trained", fontsize=10.5)
        fig.tight_layout()
        out = out_dir / f"{cfg.name}_hsr_need_per_episode_{stat}.png"
        fig.savefig(out, dpi=150)
        plt.close(fig)
        print(f"[hsr] wrote {out}", file=sys.stderr)
    pd.concat(long).to_csv(out_dir / f"{cfg.name}_hsr_need_per_episode_table.csv")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--config", required=True, help="plot config JSON (schema: plotting/plot_common.py)")
    p.add_argument("--step-range", default=None, help="LO:HI env steps; overrides config `step_range`")
    p.add_argument("--table-episodes", type=int, default=18, help="rows in the episode tables (default 18)")
    p.add_argument("--table-groups", default=None,
                   help="comma-separated group labels for the tables (default: groups with the "
                        "first group's episode length)")
    args = p.parse_args()

    cfg = load_plot_config(args.config)
    seg = load_groups(cfg, cfg.option("step_range", args.step_range, "all"))
    write_tables(seg, cfg.out_dir, cfg.name)
    panels(seg, cfg, cfg.out_dir / f"{cfg.name}_hsr_need_per_segment.png")
    tables(seg, cfg, cfg.out_dir, table_groups(seg, cfg, args.table_groups), args.table_episodes)
    return 0


if __name__ == "__main__":
    sys.exit(main())
