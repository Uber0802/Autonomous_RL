"""Render the 4-panel `trends.png` dashboard for a PPO training run.

Layout:

  ┌──────────────────────────────────────────────┬──────────────────────────────────┐
  │ Task performance: success & grasp             │ Policy drift  mean(approx_kl)    │
  │  rollout succ (5-ep MA)  + raw faint          │  per-ep mean(approx_kl) over     │
  │  rollout grasp (5-ep MA) + raw faint          │  the update's minibatches        │
  │  eval ID succ (per eval point)                │                                  │
  │  eval OOD succ (per eval point)               │                                  │
  ├──────────────────────────────────────────────┼──────────────────────────────────┤
  │ LoRA trust region  clip_fraction / episode    │ Value head  explained_var        │
  │  per-ep mean(clip_fraction) — the fraction   │  per-ep mean(1 - Var(ret-val)/    │
  │  of minibatch ratios outside the PPO clip    │  Var(ret)) over the update's     │
  │  band; the "trust region pulse"               │  minibatches                     │
  └──────────────────────────────────────────────┴──────────────────────────────────┘

Data sources:
  - `<glob_dir>/eval_success.csv`  — per-eval-point per-task success/grasp
    (read locally; no network).
  - wandb cloud history (via the public `wandb.Api()`) — per-episode
    aggregated train metrics + the `rollout/<task>/*` per-task panels.

The script is idempotent and side-effect-only-on-disk: it does not modify
the running training process and can be invoked at any time (between eval
points, after the run ends, etc.). Calling it during an eval may capture a
partially-written `eval_success.csv`; that's harmless — the next call refreshes.

Usage:
    python tools/plot_run_trends.py \\
        --run-dir <path>/wandb/run-<ts>-<id>/files/glob \\
        --max-episodes <N> \\
        --out <out-path>.png

The `--out` PNG is written; a copy also lands in `<run-dir>/trends.png` so
each run's dir has a live in-place dashboard.
"""
import argparse
import csv
import os
import sys
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _moving_average(xs, window: int):
    """5-ep MA with edge handling (shorter window near the start)."""
    xs = np.asarray(xs, dtype=float)
    n = xs.shape[0]
    out = np.full(n, np.nan)
    for i in range(n):
        lo = max(0, i - window + 1)
        chunk = xs[lo : i + 1]
        chunk = chunk[~np.isnan(chunk)]
        if chunk.size:
            out[i] = chunk.mean()
    return out


def _read_eval_csv(eval_csv: Path):
    """Group eval_success.csv rows by (kind, episode).

    Columns (`SuccessRecorder.log_eval`):
      0 episode | 1 total_steps | 2 total_resets | 3 eval_kind | 4 group
      5 task    | 6 scene       | 7 n_envs       | 8 success   | 9 grasp
     10 obj_grasped
    """
    by_kind = {"in_domain": {}, "out_of_domain": {}}
    if not eval_csv.exists():
        return by_kind
    with open(eval_csv) as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) < 11:
                continue
            try:
                ep = int(row[0])
            except ValueError:
                continue
            kind = row[3]
            if kind not in by_kind:
                continue
            try:
                succ = float(row[8])
            except ValueError:
                continue
            by_kind[kind].setdefault(ep, []).append(succ)
    return by_kind


def _read_eval_csv_avg(eval_csv: Path, x_key: str = "episode"):
    """Like _read_eval_csv but keyed on the chosen x-axis (episode or
    total_steps) instead of always episode, averaging success across tasks
    at each x. Built on top of _read_eval_csv_per_task for the column choice."""
    per_task = _read_eval_csv_per_task(eval_csv, x_key=x_key)
    by_kind = {"in_domain": {}, "out_of_domain": {}}
    for kind, tasks in per_task.items():
        for task, x_to_metrics in tasks.items():
            for x, metrics in x_to_metrics.items():
                by_kind[kind].setdefault(x, []).append(metrics["success"])
    return by_kind


def _read_eval_csv_per_task(eval_csv: Path, x_key: str = "episode"):
    """Same as _read_eval_csv but keyed by (kind, task) → {x: {success, grasp, obj_grasped}}.

    `x_key` selects which CSV column drives the x-axis:
      - `"episode"` → column 0 (the iteration count, 1, 2, …).
      - `"total_steps"` → column 1 (cumulative env-steps, the unit the
        rollout series uses on the same axis when `render_per_task` is
        called with x_key="total_steps").
    """
    col = 1 if x_key == "total_steps" else 0
    by = {"in_domain": {}, "out_of_domain": {}}
    if not eval_csv.exists():
        return by
    with open(eval_csv) as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) < 11:
                continue
            try:
                x = int(row[col])
                kind = row[3]
                task = row[5]
                succ = float(row[8])
                grasp = float(row[9])
                obj_grasped = float(row[10])
            except (ValueError, IndexError):
                continue
            if kind not in by:
                continue
            by[kind].setdefault(task, {})[x] = {
                "success": succ, "consecutive_grasp": grasp, "is_src_obj_grasped": obj_grasped,
            }
    return by


def _wandb_history(entity: str, project: str, run_id: str):
    """Return (run_name, list[dict]) from wandb cloud. Falls back to an
    empty list (with a warning) if wandb auth is missing / network is down."""
    try:
        import wandb
    except ImportError:
        print("[warn] wandb not installed — skipping cloud history", file=sys.stderr)
        return ("?", [])
    api = wandb.Api()
    full = f"{entity}/{project}/{run_id}"
    try:
        run = api.run(full)
    except Exception as e:
        print(f"[warn] could not fetch wandb run {full}: {e}", file=sys.stderr)
        return ("?", [])
    # rollout/<task>/* keys are wildcards; pull them all via samples=10000
    # (wandb.history returns a generator of dicts when pandas=False).
    history = list(run.history(samples=10000, keys=None, pandas=False))
    return (run.name, history)


def _wandb_history_chain(entity: str, project: str, run_ids):
    """Concatenate wandb histories across multiple runs in order.

    Used when a training run is RESUMED (`--resume-from`) — each resume
    spawns a fresh wandb run, so a 32→128-ep extension lives in TWO runs
    (the original ep 1-32 + the +96-ep extension ep 33-128). Passing both
    run ids in order yields one combined history sorted by `episode`.

    The display name is taken from the FIRST run id (typically the
    fresh-from-base run); episodes from later runs are appended as-is.
    """
    run_ids = [r for r in run_ids if r]
    if not run_ids:
        return ("?", [])
    primary_name, combined = _wandb_history(entity, project, run_ids[0])
    for rid in run_ids[1:]:
        _, extra = _wandb_history(entity, project, rid)
        combined.extend(extra)
    # Stable-sort by `episode` so MA / line plots advance left-to-right
    # even if the calling order was reversed.
    combined.sort(key=lambda h: h.get("episode") if h.get("episode") is not None else -1)
    return (primary_name, combined)


def _per_episode_rollout_avgs(history, metric_suffix: str, x_key: str = "episode"):
    """For each x (episode or total_steps), average `rollout/<task>/{metric_suffix}`
    over the tasks that reported a value at that x.

    `metric_suffix`: 'success' | 'consecutive_grasp' | 'is_src_obj_grasped'.
    `x_key`: 'episode' | 'total_steps' — which wandb history field drives the x-axis.
    Returns (xs_sorted, avg_values).
    """
    out = {}
    for h in history:
        x = h.get(x_key)
        if x is None:
            continue
        vals = []
        for k, v in h.items():
            if k.startswith("rollout/") and k.endswith(f"/{metric_suffix}") and v is not None:
                vals.append(float(v))
        if vals:
            out.setdefault(int(x), []).extend(vals)
    xs = sorted(out.keys())
    avgs = [float(np.mean(out[x])) for x in xs]
    return xs, avgs


def _per_task_rollout_series(history, metric_suffix: str, x_key: str = "episode"):
    """Return {task_slug: (xs, values)} for the per-task rollout metric.

    `x_key` selects the x-axis: `"episode"` (1, 2, …, max_episodes) or
    `"total_steps"` (cumulative env-steps = `num_envs × episode_len × N`,
    the unit `wandb.log(...)` ships as `total_steps` from main.py:1095).

    Mirrors the `rollout/<task_slug>/<metric>` keys main.py writes at the
    wandb.log site (slashes/spaces in task names replaced with `_`).
    """
    out = {}                                              # task_slug -> {x: value}
    for h in history:
        x = h.get(x_key)
        if x is None:
            continue
        for k, v in h.items():
            if v is None:
                continue
            if k.startswith("rollout/") and k.endswith(f"/{metric_suffix}"):
                slug = k[len("rollout/") : -len(f"/{metric_suffix}")]
                out.setdefault(slug, {})[int(x)] = float(v)
    return {slug: (sorted(d.keys()), [d[k] for k in sorted(d.keys())])
            for slug, d in out.items()}


def _ep_to_step_map(history):
    """Map episode → total_steps from the wandb history (used to convert
    eval_success.csv rows from episode-keyed to step-keyed)."""
    out = {}
    for h in history:
        ep = h.get("episode")
        ts = h.get("total_steps")
        if ep is not None and ts is not None:
            out[int(ep)] = int(ts)
    return out


def _per_episode_scalar(history, key: str, x_key: str = "episode"):
    """Time series for a single non-rollout key like `approx_kl`, against
    either `episode` or `total_steps`."""
    out = {}
    for h in history:
        x = h.get(x_key)
        if x is None:
            continue
        v = h.get(key)
        if v is None:
            continue
        out[int(x)] = float(v)
    xs = sorted(out.keys())
    vals = [out[x] for x in xs]
    return xs, vals


def render(run_dir: Path, max_episodes: int, out_path: Path,
           entity: str, project: str, run_id: str | None,
           extra_run_ids: list | None = None,
           extra_eval_csvs: list | None = None,
           x_key: str = "episode"):
    
    eval_csv = run_dir / "eval_success.csv"
    by_kind = _read_eval_csv_avg(eval_csv, x_key=x_key)   # was: _read_eval_csv(eval_csv)

    # Resume support: each `--resume-from` spawns a fresh wandb run + glob
    # dir. Pass the prior run's id(s) via `extra_run_ids` and prior glob
    # `eval_success.csv` path(s) via `extra_eval_csvs` so the chained
    # history covers ep 1 … ep<max>.
    for extra in (extra_eval_csvs or []):
        extra_path = Path(extra)
        if not extra_path.exists():
            continue
        extra_by_kind = _read_eval_csv_avg(extra_path, x_key=x_key)
        for kind, eps in extra_by_kind.items():
            for ep, vals in eps.items():
                by_kind.setdefault(kind, {}).setdefault(ep, []).extend(vals)

    if run_id is None:
        # `<glob_dir> = wandb/run-<ts>-<id>/glob` → run_id = `<id>`.
        # Walk up one level (to `run-<ts>-<id>`) and take the last
        # hyphen-separated component.
        try:
            run_id = run_dir.parent.name.split("-")[-1]
        except Exception:
            run_id = "?"

    if extra_run_ids:
        run_name, history = _wandb_history_chain(entity, project,
                                                  list(extra_run_ids) + [run_id])
    else:
        run_name, history = _wandb_history(entity, project, run_id)

    # Per-episode rollout averages.
    roll_succ_eps, roll_succ = _per_episode_rollout_avgs(history, "success", x_key=x_key)
    roll_grasp_eps, roll_grasp = _per_episode_rollout_avgs(history, "consecutive_grasp", x_key=x_key)

    # Per-episode aggregated train scalars.
    kl_eps, kl_vals = _per_episode_scalar(history, "approx_kl", x_key=x_key)
    cf_eps, cf_vals = _per_episode_scalar(history, "clip_fraction", x_key=x_key)
    ev_eps, ev_vals = _per_episode_scalar(history, "value_explained_variance", x_key=x_key)

    # Eval per-point averages (across the 4 tasks).
    def _ep_avg(by_ep):
        eps = sorted(by_ep.keys())
        return eps, [float(np.mean(by_ep[e])) for e in eps]

    id_eps, id_succ = _ep_avg(by_kind["in_domain"])
    ood_eps, ood_succ = _ep_avg(by_kind["out_of_domain"])

    # Headline episode count for the title.
    current_ep = max(
        [m for m in (
            (roll_succ_eps[-1] if roll_succ_eps else None),
            (id_eps[-1] if id_eps else None),
        ) if m is not None],
        default=0,
    )

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    xlabel = "episode" if x_key == "episode" else "total_steps"
    title = f"Run {run_name} — live @ ep{current_ep}/{max_episodes}"
    fig.suptitle(title, fontsize=13, fontweight="bold")

    # --- Panel (0, 0): Task performance ----------------------------------
    ax = axes[0, 0]
    if roll_succ_eps:
        eps_np = np.array(roll_succ_eps)
        succ_np = np.array(roll_succ)
        ax.plot(eps_np, succ_np, color="steelblue", alpha=0.3, linewidth=1)
        ax.plot(eps_np, _moving_average(succ_np, 5), color="steelblue",
                linewidth=2, label="rollout succ (5-ep MA)")
    if roll_grasp_eps:
        eps_np = np.array(roll_grasp_eps)
        grasp_np = np.array(roll_grasp)
        ax.plot(eps_np, grasp_np, color="seagreen", alpha=0.3, linewidth=1)
        ax.plot(eps_np, _moving_average(grasp_np, 5), color="seagreen",
                linewidth=2, label="rollout grasp (5-ep MA)")
    if id_eps:
        ax.plot(id_eps, id_succ, color="darkorange", marker="o", markersize=5,
                linewidth=1.5, label="eval ID succ")
    if ood_eps:
        ax.plot(ood_eps, ood_succ, color="crimson", marker="o", markersize=5,
                linewidth=1.5, label="eval OOD succ")
    ax.set_xlabel(xlabel)
    ax.set_title("Task performance: success & grasp")
    ax.set_ylim(0, 1)
    if any([roll_succ_eps, id_eps, ood_eps]):
        ax.legend(loc="best", fontsize=9)
    ax.grid(True, alpha=0.3)

    # --- Panel (0, 1): Policy drift = approx_kl --------------------------
    ax = axes[0, 1]
    if kl_eps:
        ax.plot(kl_eps, kl_vals, color="purple", marker="o", markersize=3, linewidth=1.2)
    ax.set_xlabel(xlabel)
    ax.set_title("Policy drift  mean(approx_kl) per update")
    ax.grid(True, alpha=0.3)

    # --- Panel (1, 0): LoRA trust region = clip_fraction ----------------
    ax = axes[1, 0]
    if cf_eps:
        ax.plot(cf_eps, cf_vals, color="red", marker="o", markersize=3, linewidth=1.2)
    ax.set_xlabel(xlabel)
    ax.set_title("LoRA trust region  clip_fraction / update")
    ax.grid(True, alpha=0.3)

    # --- Panel (1, 1): Value head explained_var --------------------------
    ax = axes[1, 1]
    if ev_eps:
        ax.plot(ev_eps, ev_vals, color="teal", marker="o", markersize=3, linewidth=1.2)
    ax.axhline(0.0, color="gray", linestyle=":", alpha=0.5)
    ax.set_xlabel(xlabel)
    ax.set_title("Value head  explained_var (per-ep mean)")
    ax.grid(True, alpha=0.3)

    fig.tight_layout(rect=[0, 0, 1, 0.97])

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    # Drop a live copy in the run dir so users browsing wandb don't have to
    # leave the run.
    live_copy = run_dir / "trends.png"
    fig.savefig(live_copy, dpi=110, bbox_inches="tight")
    plt.close(fig)

    print(f"[ok] wrote {out_path} (and live copy {live_copy})", file=sys.stderr)
    return str(out_path)


def _slug_to_task_label(slug: str) -> str:
    """Slug `ketchup_bottle_yellow_plate` → readable label
    `put ketchup bottle on yellow_plate`. Best-effort: the rollout slug
    is `{obj}_{recep}` with spaces→`_`, so we split heuristically by
    looking for the longest `recep` suffix that matches the known eval
    `task` strings. Falls back to a slug-with-spaces if no match."""
    pretty = slug.replace("_", " ")
    return pretty


def render_per_task(run_dir: Path, max_episodes: int, out_path: Path,
                    entity: str, project: str, run_id: str | None,
                    extra_run_ids: list | None = None,
                    extra_eval_csvs: list | None = None):
    """Per-task breakdown of `Task performance: success & grasp`.

    Companion to `render(...)` — one panel per (obj, recep) pair the
    rollout sees (the 4 panels of the Minimal single-group 2x2 scene), each
    showing the same 4 lines as the main dashboard's task-performance panel:
    rollout succ + grasp (5-ep MA + raw faint) and eval ID + OOD success.

    X-axis is **rollout steps** (`total_steps` from `wandb.log(...)` at
    main.py:1095 — cumulative env-steps = `num_envs × episode_len × N`)
    so each tick corresponds to a fixed amount of environment interaction
    regardless of `num_envs`/`episode_len` configuration changes.

    Resume support: see `render(...)` — pass prior wandb run ids and
    glob `eval_success.csv` paths via `extra_run_ids` / `extra_eval_csvs`
    so the chained x-axis covers ep 1 … ep<max>.
    """
    if run_id is None:
        try:
            run_id = run_dir.parent.name.split("-")[-1]
        except Exception:
            run_id = "?"

    if extra_run_ids:
        run_name, history = _wandb_history_chain(entity, project,
                                                  list(extra_run_ids) + [run_id])
    else:
        run_name, history = _wandb_history(entity, project, run_id)

    # X-axis = rollout steps. Pull rollout series keyed on `total_steps`
    # directly from the wandb history.
    succ_series = _per_task_rollout_series(history, "success", x_key="total_steps")
    grasp_series = _per_task_rollout_series(history, "consecutive_grasp", x_key="total_steps")

    eval_csv = run_dir / "eval_success.csv"
    eval_pt = _read_eval_csv_per_task(eval_csv, x_key="total_steps")  # {kind: {task: {steps: {metric: val}}}}
    # Merge prior glob's per-task eval points (resume support).
    for extra in (extra_eval_csvs or []):
        extra_path = Path(extra)
        if not extra_path.exists():
            continue
        extra_eval = _read_eval_csv_per_task(extra_path, x_key="total_steps")
        for kind, tasks in extra_eval.items():
            for task, step_to_metrics in tasks.items():
                eval_pt.setdefault(kind, {}).setdefault(task, {}).update(step_to_metrics)

    # Cover every task seen either in the rollout (slug form) OR in
    # eval_success.csv (raw `put obj on recep` form). Match by checking that
    # each word of the rollout slug appears in the eval task name.
    eval_tasks = sorted({t
                         for kind in eval_pt
                         for t in eval_pt[kind].keys()})
    rollout_slugs = sorted(set(succ_series.keys()) | set(grasp_series.keys()))

    def _slug_eval_match(slug: str) -> str | None:
        words = slug.split("_")
        for t in eval_tasks:
            if all(w in t for w in words):
                return t
        return None

    # The panel ordering is anchored to eval_tasks if present (more stable
    # across runs); otherwise fall back to rollout-slug order.
    if eval_tasks:
        task_panels = []
        for t in eval_tasks:
            words = t.replace("put ", "").replace(" on ", "_").split()
            slug_guess = "_".join(words)
            # Find a matching rollout slug if present.
            slug = next((s for s in rollout_slugs if all(w in s for w in t.split() if w not in ("put", "on"))),
                        slug_guess)
            task_panels.append((t, slug))
    else:
        task_panels = [(_slug_to_task_label(s), s) for s in rollout_slugs]

    n_panels = max(len(task_panels), 1)
    # 2-column layout — extra rows if more than 4 tasks (e.g., 3x3 scene).
    ncols = 2 if n_panels <= 4 else 3
    nrows = (n_panels + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(7 * ncols, 4.5 * nrows),
                             squeeze=False)

    # Headline counter for the title — show the latest episode AND the
    # cumulative rollout-step count it corresponds to. We can recover the
    # latest episode from the episode→total_steps mapping (history is
    # already on hand).
    ep_to_steps = _ep_to_step_map(history)
    current_steps = max(
        [m for m in (
            (max(eval_pt["in_domain"].get(eval_tasks[0], {0: None}).keys())
             if eval_tasks and eval_pt["in_domain"].get(eval_tasks[0]) else None),
            (max(x for xs, _ in succ_series.values() for x in xs)
             if succ_series else None),
        ) if m is not None],
        default=0,
    )
    # Find the episode at-or-below current_steps for the title header.
    current_ep = max((e for e, s in ep_to_steps.items() if s <= current_steps), default=0)
    fig.suptitle(
        f"Run {run_name} — per-task task-performance breakdown — "
        f"live @ ep{current_ep}/{max_episodes}  ({current_steps:,} total_steps)",
        fontsize=13, fontweight="bold",
    )

    for idx, (label, slug) in enumerate(task_panels):
        ax = axes[idx // ncols][idx % ncols]

        xs_s, succ_vals = succ_series.get(slug, ([], []))
        if xs_s:
            ax.plot(xs_s, succ_vals, color="steelblue", alpha=0.3, linewidth=1)
            ax.plot(xs_s, _moving_average(succ_vals, 5), color="steelblue",
                    linewidth=2, label="rollout succ (5-ep MA)")
        xs_g, grasp_vals = grasp_series.get(slug, ([], []))
        if xs_g:
            ax.plot(xs_g, grasp_vals, color="seagreen", alpha=0.3, linewidth=1)
            ax.plot(xs_g, _moving_average(grasp_vals, 5), color="seagreen",
                    linewidth=2, label="rollout grasp (5-ep MA)")

        # Eval ID + OOD success per eval point for this task.
        for kind, color, marker_label in (
            ("in_domain", "darkorange", "eval ID succ"),
            ("out_of_domain", "crimson", "eval OOD succ"),
        ):
            step_to_metrics = eval_pt[kind].get(label, {})
            if step_to_metrics:
                xs = sorted(step_to_metrics.keys())
                ys = [step_to_metrics[s]["success"] for s in xs]
                ax.plot(xs, ys, color=color, marker="o", markersize=5,
                        linewidth=1.5, label=marker_label)

        ax.set_title(label, fontsize=11)
        ax.set_xlabel("total_steps  (num_envs × episode_len, cumulative)")
        ax.set_ylim(0, 1)
        # Thousand-separator x ticks so cumulative step counts stay
        # readable as they grow into the 100K+ range.
        ax.xaxis.set_major_formatter(
            matplotlib.ticker.FuncFormatter(lambda x, _p: f"{int(x):,}")
        )
        if idx == 0:
            ax.legend(loc="best", fontsize=8)
        ax.grid(True, alpha=0.3)

    # Hide any trailing unused subplots.
    for idx in range(len(task_panels), nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)

    fig.tight_layout(rect=[0, 0, 1, 0.96])

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    live_copy = run_dir / "trends_per_task.png"
    fig.savefig(live_copy, dpi=110, bbox_inches="tight")
    plt.close(fig)

    print(f"[ok] wrote {out_path} (and live copy {live_copy})", file=sys.stderr)
    return str(out_path)


def main():
    p = argparse.ArgumentParser("plot_run_trends")
    p.add_argument("--run-dir", required=True,
                   help="path to the run's glob_dir (e.g. wandb/run-.../files/glob)")
    p.add_argument("--max-episodes", type=int, default=32,
                   help="--max-episodes used to launch the training run (for the title)")
    p.add_argument("--out", required=True, help="output PNG path for the 4-panel dashboard")
    p.add_argument("--out-per-task", default=None,
                   help="output PNG path for the per-task breakdown (default: alongside --out)")
    p.add_argument("--entity", default=os.environ.get("WANDB_ENTITY",
                                                      "b09501048-national-taiwan-university"))
    p.add_argument("--project", default=os.environ.get("WANDB_PROJECT", "CRONOS"))
    p.add_argument("--run-id", default=None, help="wandb run id (derived from --run-dir if omitted)")
    p.add_argument("--prior-run-id", action="append", default=[],
                   help="Prior wandb run id (repeatable, ordered chronologically). Use when "
                        "the current run was started via --resume-from so the chained history "
                        "covers the full ep 1…ep<max> trajectory.")
    p.add_argument("--prior-eval-csv", action="append", default=[],
                   help="Prior glob `eval_success.csv` path (repeatable). Same use as "
                        "--prior-run-id but for the local eval-point series.")
    p.add_argument("--x-axis", choices=["episode", "total_steps"], default="episode",
                   help="x-axis unit for the MAIN 4-panel dashboard (trends.png). "
                        "The per-task breakdown always uses total_steps regardless "
                        "of this flag.")
    args = p.parse_args()

    out = Path(args.out)
    render(Path(args.run_dir), args.max_episodes, out,
           args.entity, args.project, args.run_id,
           extra_run_ids=args.prior_run_id, extra_eval_csvs=args.prior_eval_csv,
           x_key=args.x_axis)
    # Auto-derive the per-task path next to --out: `<stem>-per_task.png`.
    out_pt = Path(args.out_per_task) if args.out_per_task else (
        out.with_name(out.stem + "-per_task" + out.suffix)
    )
    render_per_task(Path(args.run_dir), args.max_episodes, out_pt,
                    args.entity, args.project, args.run_id,
                    extra_run_ids=args.prior_run_id, extra_eval_csvs=args.prior_eval_csv)


if __name__ == "__main__":
    main()
