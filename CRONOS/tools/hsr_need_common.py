"""Shared loader for the HSR-need tools: how many envs / actors sit in an HSR
position at the END of every segment, computed from `segment_pose.csv`.

Used by `plot_hsr_envs_curve.py` and `plot_hsr_need_panels.py`. Both take the
same `--config` as the other plot tools (schema: `plot_common.py`): each group
is one condition, each `runs` entry one series (a seed), a list entry a resume
chain.

What is counted, per (series, episode, segment), from the `phase=end` rows —
recorded BEFORE that boundary's HSR respawn / EER reset:

  scope=task  only the obj + recep of the task that segment ran. This is
              exactly what HSR's `low_z` detector reads (`get_obj_pos()` /
              `get_recep_pos()` hold the last `evaluate()` of the segment, i.e.
              the task that just ran, not the one switched to next). Checked
              against the `Reset Unsuitable. envs: [...]` lines of four T2560
              HSR runs: 512/512 segments give the identical env set.
  scope=all   every obj / recep slot in the env (distractors included).

  obj_low / recep_low   actors with z < LOW_Z_THRESHOLD (0.7)
  actor_sum             obj_low + recep_low
  env_flag              envs with at least one low actor = envs HSR would reset

Resume chains: a later run wins every episode it covers (a resumed run that
re-runs a crashed parent's last episode supersedes the partial one). Episodes
that are still incomplete after stitching are dropped with a warning.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from plot_common import NoData, read_run_config, read_table, warn  # noqa: E402
from plot_segment_positions import LOW_Z_THRESHOLD, _model_core, parse_step_range  # noqa: E402

METRICS = ("obj_low", "recep_low", "actor_sum", "env_flag")
SCOPES = ("task", "all")
_POSE_COLS = ("episode", "segment", "env", "actor_kind", "model_name", "task", "pz")


def _run_shape(run_dir: Path) -> dict:
    rc = read_run_config(run_dir) or {}
    try:
        seg_len = int(rc.get("segment_len") or rc["task_len"])
        seg_per_ep = max(1, int(rc["episode_len"]) // seg_len)
        n_envs = int(rc["num_envs"])
    except (KeyError, TypeError, ValueError):
        raise NoData(f"{run_dir}: run_config.json lacks episode_len / segment_len / num_envs")
    return {"seg_len": seg_len, "seg_per_ep": seg_per_ep, "n_envs": n_envs,
            "seed": rc.get("seed"), "hsr": bool(rc.get("reset_unsuitable")),
            "eer": bool(rc.get("reset_robot", True))}


def _end_rows(run_dir: Path) -> pd.DataFrame:
    df = read_table(run_dir / "segment_pose.csv", what="segment_pose.csv", required_cols=_POSE_COLS)
    if "phase" not in df.columns:                  # pre-phase CSVs recorded end only
        df["phase"] = "end"
    df = df[(df["phase"] == "end") & df["actor_kind"].isin(("obj", "recep"))].copy()
    df["pz"] = pd.to_numeric(df["pz"], errors="coerce")
    return df.dropna(subset=["pz"])


def _task_mask(df: pd.DataFrame) -> np.ndarray:
    """True on the rows that are their env's task obj / recep (same phrase match
    as `plot_segment_positions.task_actor_rows`: `plate` matches `yellow_plate`)."""
    text = df["task"].astype(str).str.replace("_", " ").str.lower().str.removeprefix("put ")
    sides = text.str.partition(" on ")
    hay = np.where(df["actor_kind"] == "obj", sides[0], sides[2])
    cores = df["model_name"].map(_model_core)
    return np.array([bool(c) and c in h for c, h in zip(cores, hay)])


def _count(df: pd.DataFrame, where: str) -> pd.DataFrame:
    """One row per (episode, segment, scope) with the four METRICS."""
    df = df.assign(low=df["pz"] < LOW_Z_THRESHOLD, is_task=_task_mask(df))
    per_env_task = df[df.is_task].groupby(["episode", "segment", "env", "actor_kind"]).size()
    bad = per_env_task[per_env_task != 1]
    if len(bad):
        warn(f"{where}: {len(bad)} (segment, env, kind) cells do not match exactly one task "
             f"actor (e.g. {bad.index[0]}) — task-scope counts there are unreliable")
    out = []
    for scope, sub in (("task", df[df.is_task]), ("all", df)):
        g = sub.groupby(["episode", "segment", "env", "actor_kind"])["low"].sum().unstack("actor_kind", fill_value=0)
        g = g.reindex(columns=["obj", "recep"], fill_value=0)
        e = g.assign(any=(g["obj"] + g["recep"]) > 0).groupby(["episode", "segment"])
        r = pd.DataFrame({"obj_low": e["obj"].sum(), "recep_low": e["recep"].sum(),
                          "env_flag": e["any"].sum(), "n_envs": e.size()})
        r["actor_sum"] = r["obj_low"] + r["recep_low"]
        out.append(r.reset_index().assign(scope=scope))
    return pd.concat(out, ignore_index=True)


def load_series(chain, *, label: str) -> pd.DataFrame:
    """Per-segment HSR-need rows for one series (one seed's resume chain)."""
    kept, shape = {}, None
    for run_dir in chain:
        run_dir = Path(run_dir)
        try:
            s = _run_shape(run_dir)
            counts = _count(_end_rows(run_dir), f"{label} / {run_dir}")
        except NoData as e:
            warn(f"{label}: {e} — skipped")
            continue
        if s["hsr"]:
            warn(f"{label}: {run_dir} ran WITH HSR — its counts are per-segment respawns, "
                 f"not an accumulating backlog")
        if shape and (shape["seg_per_ep"], shape["seg_len"], shape["n_envs"]) != \
                (s["seg_per_ep"], s["seg_len"], s["n_envs"]):
            warn(f"{label}: {run_dir} changes the episode shape mid-chain "
                 f"({shape['seg_per_ep']}x{shape['seg_len']} -> {s['seg_per_ep']}x{s['seg_len']})")
        shape = shape or s
        for ep, rows in counts.groupby("episode"):
            kept[int(ep)] = rows.assign(source=str(run_dir))      # later run wins
    if not kept:
        raise NoData(f"{label}: no run in the chain produced rows")
    spe = shape["seg_per_ep"]
    df = pd.concat(kept.values(), ignore_index=True)
    n_seg = df[df.scope == "task"].groupby("episode")["segment"].nunique()
    partial = n_seg[n_seg != spe]
    if len(partial):
        warn(f"{label}: dropping incomplete episode(s) {partial.index.tolist()} "
             f"({partial.tolist()} of {spe} segments)")
        df = df[~df.episode.isin(partial.index)]
    df["seg_per_ep"] = spe
    df["global_segment"] = (df["episode"] - 1) * spe + df["segment"]
    df["total_steps"] = df["global_segment"] * shape["seg_len"] * shape["n_envs"]
    df["seed"] = shape["seed"]
    return df


def load_groups(cfg, step_range=None) -> pd.DataFrame:
    """All groups of a PlotConfig -> one long per-segment frame.

    `series` is the entry's position in the group's `runs` (0, 1, ...);
    `seed` is what its run_config says. `step_range` ("LO:HI") keeps only the
    segments whose END step falls inside it.
    """
    rng = parse_step_range(step_range if step_range is not None else "all")
    frames = []
    for g in cfg.groups:
        for i, chain in enumerate(g.chains):
            try:
                df = load_series(chain, label=f"{g.label}[{i}]")
            except NoData as e:
                warn(str(e))
                continue
            if rng is not None:
                df = df[(df.total_steps >= rng[0]) & (df.total_steps <= rng[1])]
            frames.append(df.assign(group=g.label, series=i))
    if not frames:
        raise SystemExit("no group produced any segment_pose rows")
    cols = ["group", "series", "seed", "scope", "episode", "segment", "seg_per_ep",
            "global_segment", "total_steps", "n_envs", *METRICS, "source"]
    return pd.concat(frames, ignore_index=True)[cols].sort_values(
        ["group", "series", "scope", "global_segment"], kind="stable").reset_index(drop=True)


def per_episode(seg: pd.DataFrame) -> pd.DataFrame:
    """Per (group, series, scope, episode): mean / sum / max / end-of-episode of
    every metric over that episode's segments."""
    keys = ["group", "series", "seed", "scope", "episode"]
    seg = seg.sort_values([*keys, "segment"])
    g = seg.groupby(keys, sort=False)
    out = g.agg(n_segments=("segment", "size"), total_steps_end=("total_steps", "max"))
    for m in METRICS:
        out[f"{m}_mean_per_seg"] = g[m].mean().round(3)
        out[f"{m}_sum"] = g[m].sum()
        out[f"{m}_max"] = g[m].max()
        out[f"{m}_end"] = g[m].last()
    return out.reset_index()


def write_tables(seg: pd.DataFrame, out_dir: Path, name: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    seg.to_csv(out_dir / f"{name}_hsr_need_per_segment.csv", index=False)
    per_episode(seg).to_csv(out_dir / f"{name}_hsr_need_per_episode.csv", index=False)
    print(f"[hsr] wrote {name}_hsr_need_per_segment.csv, {name}_hsr_need_per_episode.csv "
          f"to {out_dir}", file=sys.stderr)
