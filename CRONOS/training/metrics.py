"""CRONOS — metrics.

SuccessRecorder owns `eval_success.csv` plus the `counters.json` sidecar,
and builds the wandb eval panel payload (per-task scalars + overlay + mean).

Design notes:
- stdlib csv only, no pandas.
- Both CSVs carry **dual x-axes** (`total_steps` AND `total_resets`) so plots
  can be reconstructed against either without re-running.
- ID and OOD eval rows live in the *same* `eval_success.csv`, distinguished by
  the `eval_kind` column. `eval_kind` examples: `in_domain`, `out_of_domain`,
  `sequential_seq0`, ...
- `scene` column is always present. In M0..M3 it is a constant sentinel
  (`"default"`); M4 wires real scene names into it as a drop-in data change.
- An in-memory `eval_history` buffer keyed by `(eval_kind, task)` feeds the
  wandb `line_series` overlay. On instantiation, if `eval_success.csv` already
  exists (relaunched from the same glob_dir) the buffer is rebuilt from it, so
  resumed runs don't lose overlay curves.
"""

from __future__ import annotations

import csv
import json
import os
import tempfile
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

try:
    import wandb
except Exception:  # wandb is optional; metrics still work for CSVs
    wandb = None  # type: ignore


EVAL_FIELDS: Tuple[str, ...] = (
    "episode",
    "total_steps",
    "total_resets",
    "eval_kind",
    "group",
    "task",
    "scene",
    "n_envs",
    "success",
    "grasp",
    "obj_grasped",
)

# Per-env, per-segment rollout record — the training-time counterpart of
# `eval_success.csv`. One row per (episode, segment, env). Deliberately shares
# the dual x-axis columns (`total_steps` / `total_resets`) so rollout and eval
# curves can be plotted against the same axis without a join.
#
# `success` / `consecutive_grasp` / `is_src_obj_grasped` use the *identical*
# definition as eval: the value at the segment's final step, with the grasp
# flags latched within that segment only. Training gets this for free because
# `reset_robot()` clears `_elapsed_steps` and the grasp latches every segment;
# sequential eval needs `CronosWrapper.begin_segment()` to match.
#
# The three value columns answer different questions and are all kept:
#   reward_sum        — Σ r_t, the raw shaped signal PPO consumed. Because
#                       `RewardShaper` emits a potential *difference*, this
#                       telescopes to (potential_end - potential_start); useful
#                       for reconciling against the buffer, misleading alone.
#   return_discounted — Σ γ^t r_t over the same rewards. Plain Monte-Carlo, no
#                       bootstrap: "what did this segment earn".
#   return_gae        — the buffer's GAE return (advantage + value), i.e. what
#                       the critic actually regresses onto. Pairs with
#                       `value_mean` / `advantage_mean` and with the
#                       `value_explained_variance` PPO scalar.
ROLLOUT_FIELDS: Tuple[str, ...] = (
    "episode",
    "segment",
    "total_steps",
    "total_resets",
    "env_idx",
    "group",
    "task",
    "obj",
    "recep",
    "direction",
    "success",
    "consecutive_grasp",
    "is_src_obj_grasped",
    "reward_sum",
    "return_discounted",
    "return_gae",
    "value_mean",
    "advantage_mean",
)


def _slugify(s: str) -> str:
    """wandb-friendly task key: lowercase, spaces → underscores."""
    return s.strip().lower().replace(" ", "_")


def _atomic_write_text(path: Path, text: str) -> None:
    """Write `text` to `path` atomically (tmp in same dir, then rename)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".tmp_", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w") as f:
            f.write(text)
        os.replace(tmp, path)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


class SuccessRecorder:
    """Append-only CSV logger + in-memory eval history for wandb overlay."""

    def __init__(self, glob_dir: Path | str) -> None:
        self.glob_dir = Path(glob_dir)
        self.glob_dir.mkdir(parents=True, exist_ok=True)

        self.eval_csv_path = self.glob_dir / "eval_success.csv"
        self.rollout_csv_path = self.glob_dir / "rollout_success.csv"
        self.counters_path = self.glob_dir / "counters.json"

        self._ensure_header(self.eval_csv_path, EVAL_FIELDS)
        self._ensure_header(self.rollout_csv_path, ROLLOUT_FIELDS)

        # eval_history[eval_kind][task] -> list of (total_steps, success)
        self.eval_history: Dict[str, Dict[str, List[Tuple[int, float]]]] = {}
        self._rebuild_eval_history()

    # ---------- CSV helpers ----------

    @staticmethod
    def _ensure_header(path: Path, fields: Iterable[str]) -> None:
        path = Path(path)
        if path.exists() and path.stat().st_size > 0:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", newline="") as f:
            csv.writer(f).writerow(list(fields))

    @staticmethod
    def _append_row(path: Path, fields: Iterable[str], row: Dict[str, object]) -> None:
        with open(path, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(fields))
            w.writerow({k: row.get(k, "") for k in fields})
            f.flush()
            try:
                os.fsync(f.fileno())
            except OSError:
                pass  # fsync may fail on some tmpfs; not critical

    def log_rollout_segments(self, rows: List[Dict[str, object]]) -> None:
        """Append per-env rollout rows for one or more completed segments.

        Batched rather than row-at-a-time: a segment produces `num_envs` rows at
        once and the caller flushes a whole PPO update's worth, so one open/
        fsync per flush instead of 64+ keeps this off the rollout's critical
        path. Unknown keys are dropped and missing ones written empty, so adding
        a column to `ROLLOUT_FIELDS` never breaks an existing caller.
        """
        if not rows:
            return
        with open(self.rollout_csv_path, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(ROLLOUT_FIELDS))
            for row in rows:
                w.writerow({k: row.get(k, "") for k in ROLLOUT_FIELDS})
            f.flush()
            try:
                os.fsync(f.fileno())
            except OSError:
                pass

    def _rebuild_eval_history(self) -> None:
        """Populate `eval_history` from `eval_success.csv` if present."""
        if not self.eval_csv_path.exists():
            return
        try:
            with open(self.eval_csv_path, "r", newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    kind = row.get("eval_kind") or ""
                    task = row.get("task") or ""
                    try:
                        step = int(row.get("total_steps") or 0)
                        success = float(row.get("success") or 0.0)
                    except ValueError:
                        continue
                    self.eval_history.setdefault(kind, {}).setdefault(task, []).append(
                        (step, success)
                    )
        except Exception:
            # Corrupt file → leave history empty, don't crash.
            pass

    # ---------- Public API: eval ----------

    def log_eval(
        self,
        *,
        episode: int,
        total_steps: int,
        total_resets: int,
        eval_kind: str,
        group: str = "",
        task: str,
        scene: str,
        n_envs: int,
        success: float,
        grasp: float,
        obj_grasped: float,
    ) -> Dict[str, float]:
        """Append an eval row and update eval_history. Returns per-task scalars
        to dual-log to wandb (the namespaced scalars in charts 1–4 of the
        5-chart panel). The overlay + mean are produced separately via
        `build_wandb_eval_panel(eval_kind)` once all tasks of that kind have
        been logged for this eval call."""
        row = {
            "episode": episode,
            "total_steps": total_steps,
            "total_resets": total_resets,
            "eval_kind": eval_kind,
            "group": group,
            "task": task,
            "scene": scene,
            "n_envs": n_envs,
            "success": round(float(success), 6),
            "grasp": round(float(grasp), 6),
            "obj_grasped": round(float(obj_grasped), 6),
        }
        self._append_row(self.eval_csv_path, EVAL_FIELDS, row)

        # Update in-memory history for the wandb line_series overlay.
        self.eval_history.setdefault(eval_kind, {}).setdefault(task, []).append(
            (int(total_steps), float(success))
        )

        prefix = f"eval_{eval_kind}/{task}"
        return {
            f"{prefix}_success": float(success),
            f"{prefix}_grasp": float(grasp),
            f"{prefix}_obj_grasped": float(obj_grasped),
        }

    # ---------- wandb panel builder ----------

    def build_wandb_eval_panel(self, eval_kind: str) -> Dict[str, object]:
        """Build the overlay chart + mean_success scalar for one eval_kind.

        Returns a dict suitable for merging into a `wandb.log(..., step=total_steps)`
        payload. Safe when `wandb` is unavailable — returns only the mean scalar
        and skips the `line_series` object.
        """
        hist = self.eval_history.get(eval_kind, {})
        if not hist:
            return {}

        # Align all tasks to the same x-axis (union of steps, sorted).
        all_steps = sorted({s for curve in hist.values() for s, _ in curve})
        if not all_steps:
            return {}

        tasks = sorted(hist.keys())
        xs: List[List[int]] = []
        ys: List[List[float]] = []
        for task in tasks:
            per_task = dict(hist[task])  # {step: success} (last value wins if dup)
            xs.append(all_steps)
            # Forward-fill: for steps before the first measurement, use nan.
            filled: List[float] = []
            last_val: Optional[float] = None
            for step in all_steps:
                if step in per_task:
                    last_val = per_task[step]
                filled.append(last_val if last_val is not None else float("nan"))
            ys.append(filled)

        payload: Dict[str, object] = {}

        # Chart 5: overlaid line series.
        if wandb is not None:
            try:
                payload[f"eval_{eval_kind}/all_tasks"] = wandb.plot.line_series(
                    xs=xs,
                    ys=ys,
                    keys=tasks,
                    title=f"eval_{eval_kind} — all tasks",
                    xname="total_steps",
                )
            except Exception:
                # Fall through with only the mean scalar if line_series fails.
                pass

        # Chart 6: mean_success at the *latest* step of this eval kind.
        latest_step = all_steps[-1]
        per_task_at_latest = []
        for task in tasks:
            curve = hist[task]
            # Take the most recent value for this task at or before latest_step.
            last = None
            for s, v in curve:
                if s <= latest_step:
                    last = v
            if last is not None:
                per_task_at_latest.append(last)
        if per_task_at_latest:
            payload[f"eval_{eval_kind}/mean_success"] = sum(per_task_at_latest) / len(
                per_task_at_latest
            )

        return payload

    # ---------- counters.json ----------

    def write_counters(self, *, episode: int, total_steps: int, total_resets: int) -> None:
        import datetime

        data = {
            "episode": int(episode),
            "total_steps": int(total_steps),
            "total_resets": int(total_resets),
            "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
        }
        _atomic_write_text(self.counters_path, json.dumps(data, indent=2) + "\n")

    @staticmethod
    def load_counters(glob_dir: Path | str) -> Optional[Dict[str, object]]:
        path = Path(glob_dir) / "counters.json"
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text())
        except Exception:
            return None
