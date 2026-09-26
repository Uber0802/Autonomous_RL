"""Envs needing HSR over training — one curve per group, drawn like the eval-SR curves.

y = envs whose current task obj or recep is below z=0.7 at a segment's end,
i.e. the envs HSR's `low_z` detector would reset (see `hsr_need_common.py`).
For runs WITHOUT HSR the count only falls at an episode reset, so every
episode is drawn from 0 at the previous episode's last step (the reset
point), then one point per segment end.

    python plotting/plot_hsr_envs_curve.py --config history/paper_plot_configs/hsr_need_config.json

Writes to the config's `out_dir`:
  <name>_hsr_envs.png             series mean, band = +-1 std across series
  <name>_hsr_envs_seed<k>.png     one figure per series position k (--per-seed)
  <name>_hsr_envs_all*.png        the same with `--scope all` / `both`
                                  (any obj / recep slot, distractors included)
  <name>_hsr_need_per_segment.csv, <name>_hsr_need_per_episode.csv
Config `step_range` ("LO:HI", env steps) trims every series, e.g. "0:2949120".
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from hsr_need_common import load_groups, write_tables  # noqa: E402
from plot_common import (CURVE_BAND_ALPHA, CURVE_LINEWIDTH, FLAT_BOX_ASPECT,  # noqa: E402
                         FLAT_FIGSIZE, default_colors, load_plot_config, new_curve_figure,
                         save_curve_figure, style_curve_axes)

Y_LABEL = {"task": "envs needing HSR", "all": "envs with any obj / recep fallen"}


def sawtooth(sub, series):
    """x, y (n_points x n_series) with a 0 at every episode's reset point."""
    per = sub[sub.series.isin(series)].pivot(index="global_segment", columns="series",
                                             values="env_flag").sort_index().dropna()
    spe = int(sub.seg_per_ep.iloc[0])
    steps_per_seg = float(sub.total_steps.iloc[0]) / float(sub.global_segment.iloc[0])
    xs, ys = [], []
    for ep in sorted({(g - 1) // spe + 1 for g in per.index}):
        block = per.loc[(ep - 1) * spe + 1: ep * spe]
        xs.append((ep - 1) * spe * steps_per_seg)
        ys.append(np.zeros(per.shape[1]))
        xs.extend(block.index * steps_per_seg)
        ys.extend(block.to_numpy())
    return np.asarray(xs, float), np.vstack(ys)


def draw(seg, cfg, out_dir, scope, series_sel, tag):
    n_envs = int(seg.n_envs.max())
    fig, ax = new_curve_figure(FLAT_FIGSIZE)
    colors = default_colors(len(cfg.groups))
    x_max, drawn = 0.0, 0
    for g, color in zip(cfg.groups, colors):
        sub = seg[(seg.group == g.label) & (seg.scope == scope)]
        series = [s for s in sorted(sub.series.unique()) if series_sel is None or s == series_sel]
        if not series:
            continue
        x, y = sawtooth(sub, series)
        m, s = y.mean(axis=1), y.std(axis=1)
        ax.plot(x, m, linewidth=CURVE_LINEWIDTH, color=color, label=g.label)
        if y.shape[1] > 1:
            ax.fill_between(x, np.clip(m - s, 0, n_envs), np.clip(m + s, 0, n_envs),
                            color=color, alpha=CURVE_BAND_ALPHA, linewidth=0)
        x_max, drawn = max(x_max, x.max()), drawn + 1
        seeds = sorted(sub[sub.series.isin(series)].seed.unique())
        print(f"[hsr] {tag or 'mean'} {scope}: {g.label}: series {series} (seed {seeds}), "
              f"{len(x)} points to {x.max():,.0f} steps", file=sys.stderr)
    if not drawn:
        return
    style_curve_axes(ax, x_axis="total_steps", y_label=f"{Y_LABEL[scope]} (of {n_envs})",
                     x_max=x_max, box_aspect=FLAT_BOX_ASPECT, legend_outside=True)
    ax.set_ylim(-0.02 * n_envs, 1.02 * n_envs)          # counts, not a 0-1 rate
    ax.set_yticks(range(0, n_envs + 1, max(1, n_envs // 4)))
    out = out_dir / f"{cfg.name}_hsr_envs{'' if scope == 'task' else '_all'}{tag}.png"
    save_curve_figure(fig, out)
    print(f"[hsr] wrote {out}", file=sys.stderr)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--config", required=True, help="plot config JSON (schema: plotting/plot_common.py)")
    p.add_argument("--scope", choices=["task", "all", "both"], default="both",
                   help="task = what HSR checks (default figure); all = every slot")
    p.add_argument("--step-range", default=None, help="LO:HI env steps; overrides config `step_range`")
    ps = p.add_mutually_exclusive_group()
    ps.add_argument("--per-seed", dest="per_seed", action="store_true", default=True,
                    help="also write one figure per series position (default)")
    ps.add_argument("--no-per-seed", dest="per_seed", action="store_false")
    args = p.parse_args()

    cfg = load_plot_config(args.config)
    seg = load_groups(cfg, cfg.option("step_range", args.step_range, "all"))
    write_tables(seg, cfg.out_dir, cfg.name)
    scopes = ["task", "all"] if args.scope == "both" else [args.scope]
    n_series = int(seg.series.max()) + 1
    for scope in scopes:
        draw(seg, cfg, cfg.out_dir, scope, None, "")
        if args.per_seed and n_series > 1:
            for k in range(n_series):
                draw(seg, cfg, cfg.out_dir, scope, k, f"_seed{k}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
