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
`--config plot_runs.json` aggregates each group's series into a mean ± spread
band. A series may be a **resume chain** — several run dirs stitched into one
continuous line. See `tools/plot_common.py` for the schema.

    python tools/plot_rollout_success.py --config tools/plot_runs_example.json

It writes a figure set, not one figure:

    <name>_rollout_success.png            all groups, one hue each, drawn whole
    <name>_rollout_success_<group>.png    one group, split at ITS OWN resets

The per-group figures are the reason for the split. A training curve is not one
continuous experiment: every reset re-randomizes the batch, so the rate the
policy reaches *within* an inter-reset stretch and the trend *across* stretches
are different quantities, and one unbroken line hides the difference — a
within-stretch climb followed by the reset's drop reads as noise. Each piece
gets its own colour — categorical, not a ramp, so four pieces are unmistakable
and eight are still separable — and a dotted rule marks where the reset fell.
The hue is free to carry the reset index because a per-group figure holds one
group and the legend's title names it.

The split stays out of the main figure on purpose: there the hue is what tells
the conditions apart, and spending it on the reset index would leave nothing to
identify a group by.

Only for T1280 and longer (`--no-reset-split` / `--no-per-group` to opt out).
At `segment_len = 80` an episode holds `episode_len / 80` segments, so T1280 is
the first horizon whose 16 make a shape; T320's 4 and T80's 1 do not, and those
runs are drawn whole with the reason on stderr. A resume chain is only as
splittable as its coarsest leg. A run whose resets fire faster than the horizon
implies — HSR soft-resets at segment boundaries — is caught by a second guard on
the measured piece length.

Missing data is skipped, not fatal
----------------------------------
An eval-only run has no `rollout_success.csv`, a run in its first episode has an
empty one, and an older run may predate a column. In `--config` mode each of
those is warned about by name and that run is skipped, so one such run does not
cost every other group its figure; the tool fails only when NO group produced a
curve. `--run-dir` / `--csv` still fail loudly — there you named one file, so
"not found" is the answer to the question you asked.

Companion to `tools/plot_run_trends.py`, which plots the *eval* points and the
PPO health scalars. This one is the training-side view and needs no wandb access
— it reads only the local CSV.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from plot_common import (CURVE_LEGEND, RESET_SPLIT_MIN_EPISODE_LEN,  # noqa: E402
                         RESET_SPLIT_MIN_PIECE, X_LABEL, NoData, concat_chain,
                         default_colors, load_plot_config, new_curve_figure,
                         out_variant, piece_labels, plot_group_curve,
                         plot_reset_segmented_curve, prepend_origin,
                         read_run_config, read_table, reset_pieces,
                         save_curve_figure, style_curve_axes, unique_slugs,
                         warn)

# Columns whose empty-string cells mean "this env did not report at this
# boundary" rather than zero. `training/metrics.py` writes "" for those.
_METRIC_COLS = ("success", "consecutive_grasp", "is_src_obj_grasped")


def load_rollout(csv_path: Path, *, required: bool = True) -> pd.DataFrame:
    """Read `rollout_success.csv` and coerce the metric columns to float.

    Empty cells become NaN and are excluded from the means (rather than being
    read as 0, which would silently depress every curve).

    `required=False` is the `--config` path: a missing, empty or too-old file
    raises `NoData` for the caller to warn about and skip, so one eval-only run
    in a config does not cost every other group its figure. The single-run path
    keeps `required=True`, where "not found" is the answer to the question the
    user asked.
    """
    if not required:
        # `episode`/`segment`/`total_steps` are the plot's x axis and the
        # grouping key; without them there is no curve to draw, so they are
        # required rather than back-filled.
        return _coerce_rollout(read_table(
            csv_path, what="rollout_success.csv",
            required_cols=("episode", "segment", "total_steps", "env_idx")))
    if not csv_path.exists():
        raise FileNotFoundError(
            f"{csv_path} not found. It is written by the training rollout; an "
            f"eval-only run does not produce one."
        )
    return _coerce_rollout(pd.read_csv(csv_path))


def _coerce_rollout(df: pd.DataFrame) -> pd.DataFrame:
    for col in _METRIC_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if "direction" not in df.columns:
        # Pre-LSR CSVs predate the column; everything in them is forward.
        df["direction"] = "forward"
    df["direction"] = df["direction"].fillna("forward")
    if "total_resets" not in df.columns:
        # Predates the column. NaN rather than 0: an unknown reset count must
        # not read as "no reset happened", which would fake one giant piece.
        df["total_resets"] = np.nan
    df["total_resets"] = pd.to_numeric(df["total_resets"], errors="coerce")
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
    # `total_resets` joins the key rather than the aggregates: it is constant
    # within a (episode, segment), and the per-group figures split the curve on
    # it, so it has to survive this collapse.
    by = ["episode", "segment", "seg_index", "total_steps", "total_resets",
          *extra_group]
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


# ---------------------------------------------------------------------------
# --config mode: one main figure over all groups + one figure per group
# ---------------------------------------------------------------------------
#
# Two different questions, so two different figures rather than one compromise:
#
#   <name>_rollout_success.png            every group, one hue each, drawn whole.
#                                         "Which condition ends up higher?"
#   <name>_rollout_success_<group>.png    one group, split at its own resets and
#                                         shaded light -> dark.
#                                         "What happens between resets?"
#
# The reset split lives only in the per-group figures. In the main figure the
# hue already carries group identity, and re-using it for the reset index would
# leave nothing to tell the conditions apart.


@dataclass
class GroupCurve:
    """One group's aggregated series: mean ± std over its seeds, on `total_steps`."""
    label: str
    x: np.ndarray
    mean: np.ndarray
    std: np.ndarray
    # Cumulative reset count aligned to `x`; NaN where no run recorded one.
    resets: np.ndarray
    n_series: int
    # Shortest `episode_len` among the group's runs, or None if no run said.
    # Shortest, not mean: a resume chain that spans T320 -> T2560 is only as
    # splittable as its coarsest leg.
    episode_len: Optional[float]


def _episode_len(run_dir: Path, frame: pd.DataFrame) -> Optional[float]:
    """The run's `episode_len`, from `run_config.json` or measured from the CSV.

    The measured fallback is what keeps pre-`run_config.json` runs usable:
    segments-per-episode is visible in the CSV's own (episode, segment) pairs,
    and multiplying by the segment length recovers the horizon.
    """
    rc = read_run_config(run_dir) or {}
    try:
        return float(rc["episode_len"])
    except (KeyError, TypeError, ValueError):
        pass
    if frame.empty or "episode" not in frame.columns:
        return None
    segs = frame.groupby("episode")["segment"].nunique()
    if segs.empty:
        return None
    try:
        seg_len = float(rc.get("task_len") or rc.get("segment_len") or 80)
    except (TypeError, ValueError):
        seg_len = 80.0
    return float(segs.median()) * seg_len


def collect_group(group, *, direction: str, metric: str,
                  smooth: int) -> Optional[GroupCurve]:
    """Load one group's runs and aggregate its seeds into a single curve.

    Every reason a run contributes nothing — no file, an empty file, an older
    file without the column, or no rows left after the direction filter — is
    warned about by name and skipped. A group survives on whatever runs are
    left; it is dropped only when none are.
    """
    series, ep_lens = [], []
    for chain in group.chains:
        frames = []
        for run_dir in chain:
            run_dir = Path(run_dir)
            try:
                df = load_rollout(run_dir / "rollout_success.csv", required=False)
            except NoData as e:
                warn(f"group '{group.label}': {e}")
                continue
            if metric not in df.columns:
                warn(f"group '{group.label}': {run_dir} has no {metric!r} column "
                     f"(have: {sorted(c for c in _METRIC_COLS if c in df.columns)})")
                continue
            if direction != "all":
                sel = df[df["direction"] == direction]
                if sel.empty:
                    warn(f"group '{group.label}': {run_dir} has no "
                         f"direction={direction!r} rows "
                         f"(present: {sorted(df['direction'].unique())})")
                    continue
                df = sel
            frame = segment_means(df, "total_steps")
            if frame.empty:
                continue
            frames.append(frame)
            ep_lens.append(_episode_len(run_dir, frame))
        merged = concat_chain(frames, "total_steps")
        if len(merged):
            series.append(merged)

    if not series:
        warn(f"group '{group.label}' produced no rows")
        return None

    # Align the series on their shared x values. Same-config seeds land on
    # identical `total_steps`, so the align is exact; anything a series is
    # missing simply does not contribute to that x.
    wide = pd.concat(
        [s.set_index("total_steps")[metric].rename(k) for k, s in enumerate(series)],
        axis=1,
    ).sort_index()
    # Same alignment for the reset counter. Median across seeds: same-config
    # seeds reset in lockstep, and the median ignores a seed that stopped early
    # rather than letting it drag the boundary.
    res = pd.concat(
        [s.set_index("total_steps")["total_resets"].rename(k)
         for k, s in enumerate(series)],
        axis=1,
    ).sort_index().median(axis=1).reindex(wide.index)
    if smooth > 1:
        wide = wide.rolling(smooth, min_periods=1).mean()
    mean, std = wide.mean(axis=1), wide.std(axis=1)
    n = wide.shape[1]

    x, mean_y, std_y = prepend_origin(mean.index.to_numpy(),
                                      mean.to_numpy(), std.to_numpy())
    resets = res.to_numpy(dtype=float)
    if x.size == resets.size + 1:
        # `prepend_origin` added the (0, 0) anchor. It belongs to the FIRST
        # reset piece, so it inherits that piece's count — prepending a 0 here
        # would invent a reset boundary at the very first recorded segment for
        # any run that starts mid-way (a resume).
        resets = np.concatenate(([resets[0] if resets.size else np.nan], resets))

    known = [e for e in ep_lens if e is not None]
    return GroupCurve(label=group.label, x=x, mean=mean_y, std=std_y,
                      resets=resets, n_series=n,
                      episode_len=min(known) if known else None)


def reset_split_plan(curve: GroupCurve):
    """`(pieces, reason_it_was_declined)` — exactly one of the two is meaningful.

    Declining is the normal outcome for a short-horizon run and is reported, not
    silently applied: a T320 curve drawn whole next to a split T2560 one is only
    readable if the figure says which it is.
    """
    pieces = reset_pieces(curve.resets)
    if len(pieces) < 2:
        return None, "no reset boundary inside the plotted range"
    if curve.episode_len is not None and curve.episode_len < RESET_SPLIT_MIN_EPISODE_LEN:
        return None, (f"episode_len={curve.episode_len:g} < "
                      f"{RESET_SPLIT_MIN_EPISODE_LEN} (T{RESET_SPLIT_MIN_EPISODE_LEN}+ only)")
    if curve.episode_len is None:
        warn(f"group '{curve.label}': episode_len unknown (no run_config.json and "
             f"the CSV gave no segments-per-episode); splitting on piece length alone")
    median_piece = float(np.median([hi - lo for lo, hi in pieces]))
    if median_piece < RESET_SPLIT_MIN_PIECE:
        return None, (f"resets every {median_piece:g} segments on average, under "
                      f"{RESET_SPLIT_MIN_PIECE} — the pieces would be shorter than "
                      f"the trend inside them")
    return pieces, None


def render_main(curves, colors, out_path: Path, *, metric: str) -> Path:
    """Every group on one axes, one hue each, each curve drawn whole."""
    fig, ax = new_curve_figure()
    x_max = 0.0
    for curve, color in zip(curves, colors):
        plot_group_curve(ax, curve.x, curve.mean, curve.std, color=color,
                         label=curve.label, n_series=curve.n_series)
        x_max = max(x_max, float(curve.x.max()) if curve.x.size else 0.0)
    style_curve_axes(ax, x_axis="total_steps", y_label=metric, x_max=x_max)
    return save_curve_figure(fig, out_path)


def render_group_panel(curve: GroupCurve, color, out_path: Path, *,
                       metric: str, split: bool) -> Path:
    """One group on its own axes, split at its resets when that is meaningful.

    When the curve is split the group's label goes in the legend's TITLE, since
    the pieces occupy the entries and the panel would otherwise not say which
    condition it shows (`save_curve_figure` deliberately writes no suptitle —
    the run identity lives in the filename). When it is drawn whole the label is
    the single entry instead; carrying both would print it twice.
    """
    fig, ax = new_curve_figure()
    pieces, declined = reset_split_plan(curve) if split else (None, "--no-reset-split")
    if pieces:
        plot_reset_segmented_curve(
            ax, curve.x, curve.mean, curve.std, pieces=pieces,
            labels=piece_labels(curve.x, curve.resets, pieces),
            n_series=curve.n_series)
        print(f"[rollout] {curve.label}: split into {len(pieces)} inter-reset "
              f"pieces (episode_len="
              f"{'?' if curve.episode_len is None else f'{curve.episode_len:g}'})",
              file=sys.stderr)
    else:
        plot_group_curve(ax, curve.x, curve.mean, curve.std, color=color,
                         label=curve.label, n_series=curve.n_series)
        print(f"[rollout] {curve.label}: drawn whole — {declined}", file=sys.stderr)
    x_max = float(curve.x.max()) if curve.x.size else 0.0
    style_curve_axes(ax, x_axis="total_steps", y_label=metric, x_max=x_max)
    if pieces:
        ax.legend(title=curve.label, **CURVE_LEGEND)
    return save_curve_figure(fig, out_path)


def render_groups(cfg, out_path: Path, *, direction: str, x_key: str,
                  smooth: int, metric: str, per_group: bool = True,
                  reset_split: bool = True) -> list:
    """The whole `--config` figure set. Returns the paths written."""
    curves = []
    for group in cfg.groups:
        curve = collect_group(group, direction=direction, metric=metric,
                              smooth=smooth)
        if curve is not None:
            curves.append(curve)

    if not curves:
        # Without this the figure saves as an empty pair of axes, which reads as
        # "the policy scored zero" rather than "no input was found".
        raise SystemExit(
            f"[rollout] no group produced a curve — nothing to plot.\n"
            f"  The [warn] lines above name the groups. Check that each `runs`\n"
            f"  entry points at a run's glob/ dir containing rollout_success.csv\n"
            f"  (an eval-only run has none), and that direction={direction!r} and\n"
            f"  metric={metric!r} exist in it.")

    for curve in curves:
        print(f"[group] {curve.label:<34s} {curve.n_series} series, "
              f"{curve.x.size} x-points, final {metric}={curve.mean[-1]:.4f}",
              file=sys.stderr)
    smooth_note = f", MA{smooth}" if smooth > 1 else ""
    print(f"[rollout] per-segment {metric} (direction={direction}{smooth_note}); "
          f"band = ±1 std across series", file=sys.stderr)

    # Colours are assigned over the groups that SURVIVED, so a skipped group
    # does not leave a hole in the palette — but the same hue is then used for a
    # group in both figures, which is the property that makes them comparable.
    colors = default_colors(len(curves))
    # No title on the main figure: direction / smoothing / band meaning are
    # settings, not findings, and they are already on stderr above.
    written = [render_main(curves, colors, out_path, metric=metric)]

    if per_group:
        slugs = unique_slugs([c.label for c in curves])
        for curve, color in zip(curves, colors):
            written.append(render_group_panel(
                curve, color, out_variant(out_path, slugs[curve.label]),
                metric=metric, split=reset_split))
    return written


def summarize(df: pd.DataFrame, direction: str) -> None:
    sel = df if direction == "all" else df[df["direction"] == direction]
    if sel.empty:
        print(f"[rollout] no rows with direction={direction!r}", file=sys.stderr)
        return
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
    pg = p.add_mutually_exclusive_group()
    pg.add_argument("--per-group", dest="per_group", action="store_true",
                    default=None,
                    help="--config mode: also write one figure per group "
                         "(<name>_rollout_success_<group>.png). On by default.")
    pg.add_argument("--no-per-group", dest="per_group", action="store_false",
                    help="write only the all-group main figure")
    rs = p.add_mutually_exclusive_group()
    rs.add_argument("--reset-split", dest="reset_split", action="store_true",
                    default=None,
                    help=f"split each per-group curve at its reset boundaries "
                         f"and shade the pieces light->dark. On by default, and "
                         f"applied only to runs with episode_len >= "
                         f"{RESET_SPLIT_MIN_EPISODE_LEN} (T{RESET_SPLIT_MIN_EPISODE_LEN}+); "
                         f"a shorter run is drawn whole and says so on stderr.")
    rs.add_argument("--no-reset-split", dest="reset_split", action="store_false",
                    help="draw every per-group curve whole")
    args = p.parse_args()

    if args.config:
        cfg = load_plot_config(args.config)
        args.direction = cfg.option("direction", args.direction, "forward")
        args.by = cfg.option("by", args.by, "none")
        args.metric = cfg.option("metric", args.metric, "success")
        args.smooth = int(cfg.option("smooth", args.smooth, 5))
        args.per_group = bool(cfg.option("per_group", args.per_group, True))
        args.reset_split = bool(cfg.option("reset_split", args.reset_split, True))
        out = Path(args.out) if args.out else cfg.out_dir / f"{cfg.name}_rollout_success.png"
        for path in render_groups(cfg, out, direction=args.direction,
                                  x_key="total_steps", smooth=args.smooth,
                                  metric=args.metric, per_group=args.per_group,
                                  reset_split=args.reset_split):
            print(f"[ok] wrote {path}", file=sys.stderr)
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
