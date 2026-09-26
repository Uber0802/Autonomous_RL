"""Standalone-eval record files: schemas, writers, readers, aggregations. Stdlib only.

Source of truth (appended per unit as the eval runs):

    eval_per_trial.csv          one row per (domain, round, task slot, env)
    eval_layouts.csv            one row per (domain, round, env) reset: slot, key, ids applied
    eval_segment_pose.csv       training `segment_pose.csv` columns + eval keys

Derived (rebuilt whole, atomically, by `evaluation.outputs.rebuild_outputs`):

    eval_sequence_summary.csv   aggregates; `seq_kind` separates training/random/single
    eval_coverage.csv           planned vs actual trials / resets / layouts
    eval_report.txt             human-readable
    eval_status.json            complete?, done and missing units
    eval_success.csv            SuccessRecorder schema — only when the eval is complete

`SegmentPoseWriter` is also what `main.py` uses for the training-time
`segment_pose.csv`, so the two files share one formatter.

Why derived files are rebuilt rather than appended: see doc/rng_and_io_notes.md.
"""

from __future__ import annotations

import csv
import io
import os
from collections import OrderedDict
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

PER_TRIAL = "eval_per_trial.csv"
LAYOUTS = "eval_layouts.csv"
POSES = "eval_segment_pose.csv"
SUMMARY = "eval_sequence_summary.csv"
COVERAGE = "eval_coverage.csv"
STATUS = "eval_status.json"
SUCCESS = "eval_success.csv"

# The first ten columns are the pre-existing schema, in the pre-existing order,
# so `plotting/mcnemar_pair.py` and `plotting/parse_autorl_eval.py` output stay
# compatible. Everything after `prefix` is new.
PER_TRIAL_FIELDS: Tuple[str, ...] = (
    "seq_idx", "task_idx", "obj_set", "task", "env_idx",
    "success", "success_chained", "grasp", "obj_grasped", "prefix",
    "eval_kind", "seq_kind", "group", "obj", "recep", "order",
    "pose_set", "cycle_idx", "pass_label", "episode", "total_steps",
)

SUMMARY_FIELDS: Tuple[str, ...] = (
    "level", "eval_kind", "seq_kind", "group", "pose_set", "seq_idx", "order", "task_idx", "task",
    "n_trials", "success", "success_chained", "grasp", "obj_grasped",
)

COVERAGE_FIELDS: Tuple[str, ...] = (
    "eval_kind", "seq_kind", "group", "pass_label", "envs", "rounds", "pose_sets", "orders",
    "tasks_per_round", "distinct_tasks", "trials", "trials_per_task", "resets", "layouts",
    "actual_trials", "actual_resets", "actual_layouts", "padded_envs", "match",
)

LAYOUT_FIELDS: Tuple[str, ...] = (
    "eval_kind", "seq_idx", "seq_kind", "pose_set", "pass_label", "group", "env_idx", "scene_env_idx",
    "order", "layout_slot", "layout_key", "layout_id", "rand_id", "pos_id", "quat_id", "overlay_id",
)

# Training `segment_pose.csv` header, verbatim. Eval appends EVAL_POSE_EXTRA.
SEGMENT_POSE_FIELDS: Tuple[str, ...] = (
    "episode", "segment", "phase", "total_steps", "env", "actor_kind", "slot",
    "model_name", "task", "px", "py", "pz", "qw", "qx", "qy", "qz",
)
EVAL_POSE_EXTRA: Tuple[str, ...] = ("eval_kind", "seq_kind", "seq_idx", "pose_set", "task_idx",
                                    "group", "obj_set", "pass_label")

_FLOATS = ("success", "success_chained", "grasp", "obj_grasped")
_INTS = ("seq_idx", "task_idx", "env_idx", "pose_set", "cycle_idx")


# ---------------------------------------------------------------------------
# File primitives
# ---------------------------------------------------------------------------

def append_rows(path: Path, fields: Sequence[str], rows: Iterable[dict]) -> None:
    """Append, writing the header for a new file; refuse a file with another header."""
    path = Path(path)
    new = not path.exists() or path.stat().st_size == 0
    if not new:
        with open(path, newline="") as f:
            header = next(csv.reader(f), [])
        if tuple(header) != tuple(fields):
            raise RuntimeError(f"{path} exists with a different header {header}; "
                               f"expected {list(fields)}. Write to a fresh run directory.")
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(fields))
        if new:
            w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fields})
        f.flush()
        os.fsync(f.fileno())


def atomic_write_text(path: Path, text: str) -> None:
    path = Path(path)
    tmp = path.with_name(f".{path.name}.tmp")
    with open(tmp, "w", newline="") as f:
        f.write(text)
        f.flush()
        os.fsync(f.fileno())
    tmp.replace(path)


def atomic_write_csv(path: Path, fields: Sequence[str], rows: Iterable[dict]) -> None:
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=list(fields))
    w.writeheader()
    for r in rows:
        w.writerow({k: r.get(k, "") for k in fields})
    atomic_write_text(path, buf.getvalue())


def read_rows(path: Path) -> List[dict]:
    path = Path(path)
    if not path.exists() or path.stat().st_size == 0:
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def filter_rows_in_place(path: Path, keep: Callable[[dict], bool]) -> int:
    """Drop rows for which `keep` is false, keeping the other lines byte for byte.

    Used to cut a crashed unit's partial rows before resuming. Returns rows dropped.
    """
    path = Path(path)
    if not path.exists() or path.stat().st_size == 0:
        return 0
    with open(path, newline="") as f:
        lines = f.read().splitlines(keepends=True)
    header = next(csv.reader([lines[0]]))
    kept, dropped = [lines[0]], 0
    for line in lines[1:]:
        values = next(csv.reader([line]))
        if keep(dict(zip(header, values))):
            kept.append(line)
        else:
            dropped += 1
    if dropped:
        atomic_write_text(path, "".join(kept))
    return dropped


def typed_trials(rows: Iterable[dict]) -> List[dict]:
    out = []
    for r in rows:
        r = dict(r)
        for k in _FLOATS:
            r[k] = float(r[k])
        for k in _INTS:
            if r.get(k, "") != "":
                r[k] = int(r[k])
        out.append(r)
    return out


def format_trial(r: dict) -> dict:
    return {**r, **{k: f"{float(r[k]):.4f}" for k in _FLOATS}}


# ---------------------------------------------------------------------------
# Aggregations (pure functions of rows)
# ---------------------------------------------------------------------------

def _group(rows: List[dict], keys: Tuple[str, ...]) -> "OrderedDict":
    out: "OrderedDict" = OrderedDict()
    for r in rows:
        out.setdefault(tuple(r[k] for k in keys), []).append(r)
    return out


def _mean(xs: Sequence[float]) -> float:
    return sum(xs) / len(xs) if xs else float("nan")


def stats(rows: List[dict]) -> dict:
    return dict(n_trials=len(rows), **{k: _mean([r[k] for r in rows]) for k in _FLOATS})


def summary_rows(trials: List[dict]) -> List[dict]:
    """Aggregates, always split by (eval_kind, seq_kind).

    level=slot      (group, seq_idx, order, task_idx)  one task slot of one order of one round
    level=order     (group, seq_idx, order)            one order of one round
    level=pose_set  (group, pose_set)                  a scene within one set of start poses
    level=task      (group, task)                      a task pooled over rounds of this kind
    level=scene     (group)                            a scene pooled over tasks and rounds
    level=kind      ()                                 all scenes

    `success_chained` above level=slot averages over the positions pooled; read
    it per position at level=slot, or at the last position of an order.
    """
    levels = (
        ("slot", ("eval_kind", "seq_kind", "group", "seq_idx", "order", "task_idx")),
        ("order", ("eval_kind", "seq_kind", "group", "seq_idx", "order")),
        ("pose_set", ("eval_kind", "seq_kind", "group", "pose_set")),
        ("task", ("eval_kind", "seq_kind", "group", "task")),
        ("scene", ("eval_kind", "seq_kind", "group")),
        ("kind", ("eval_kind", "seq_kind")),
    )
    rows = []
    for level, keys in levels:
        for key, grp in _group(trials, keys).items():
            row = dict(zip(keys, key))
            row["level"] = level
            if level == "slot":
                row["task"] = grp[0]["task"]
            if level in ("slot", "order"):
                row["pose_set"] = grp[0]["pose_set"]
            row.update(stats(grp))
            rows.append(row)
    return rows


def actual_counts(trials: List[dict], layouts: List[dict]) -> Dict[Tuple[str, str, str], dict]:
    """Counted trials, resets and distinct start poses per (eval_kind, seq_kind, group)."""
    out: Dict[Tuple[str, str, str], dict] = {}

    def slot(r):
        return out.setdefault((r["eval_kind"], r["seq_kind"], r["group"]), dict(trials=0, resets=0, poses=set()))

    for r in trials:
        slot(r)["trials"] += 1
    for r in layouts:
        a = slot(r)
        a["resets"] += 1
        a["poses"].add((str(r["pose_set"]), str(r["layout_slot"])))
    return out


def coverage_rows(planned: List[dict], trials: List[dict], layouts: List[dict]) -> List[dict]:
    actual = actual_counts(trials, layouts)
    out = []
    for p in planned:
        a = actual.get((p["eval_kind"], p["seq_kind"], p["group"]), dict(trials=0, resets=0, poses=set()))
        out.append({**p, "actual_trials": a["trials"], "actual_resets": a["resets"],
                    "actual_layouts": len(a["poses"]), "padded_envs": 0,
                    "match": int(a["trials"] == p["trials"] and a["resets"] == p["resets"]
                                 and len(a["poses"]) == p["layouts"])})
    return out


def render_report(trials: List[dict], header: str = "") -> str:
    """Training rounds, random rounds and single-task rounds in separate sections."""
    kind_title = {"training": "training rounds", "random": "random rounds", "single": "single-task rounds"}
    lines = [header] if header else []
    for (eval_kind, seq_kind), kind_rows in _group(trials, ("eval_kind", "seq_kind")).items():
        rounds = sorted({r["seq_idx"] for r in kind_rows})
        sets = sorted({r["pose_set"] for r in kind_rows})
        lines.append(f"=== {eval_kind} · {kind_title.get(seq_kind, seq_kind)} "
                     f"(rounds {', '.join(map(str, rounds))}; pose sets {', '.join(map(str, sets))}) ===")
        for (group,), g_rows in _group(kind_rows, ("group",)).items():
            st = stats(g_rows)
            lines.append(f"  scene {group}: trials={st['n_trials']}  success={st['success']:.4f}  "
                         f"chained={st['success_chained']:.4f}  grasp={st['grasp']:.4f}")
            if len(sets) > 1:
                lines.append("    by pose set")
                for (ps,), p_rows in sorted(_group(g_rows, ("pose_set",)).items()):
                    s_ = stats(p_rows)
                    lines.append(f"      set{ps}  n={s_['n_trials']:<5} success={s_['success']:.4f}  "
                                 f"chained={s_['success_chained']:.4f}")
            if seq_kind != "single":
                lines.append("    by order (chained = cleared the whole round)")
                for (seq_idx, order), o_rows in _group(g_rows, ("seq_idx", "order")).items():
                    last = max(r["task_idx"] for r in o_rows)
                    o_ = stats(o_rows)
                    l_ = stats([r for r in o_rows if r["task_idx"] == last])
                    lines.append(f"      round{seq_idx:<3} {order:<6} n={l_['n_trials']:<4} "
                                 f"success={o_['success']:.4f}  chained={l_['success_chained']:.4f}")
                lines.append("    by position (chained = cleared every task up to here)")
                for (task_idx,), p_rows in sorted(_group(g_rows, ("task_idx",)).items()):
                    p_ = stats(p_rows)
                    tasks = sorted({r["task"] for r in p_rows})
                    label = tasks[0] if len(tasks) == 1 else f"{len(tasks)} tasks"
                    lines.append(f"      pos{task_idx}  n={p_['n_trials']:<5} success={p_['success']:.4f}  "
                                 f"chained={p_['success_chained']:.4f}  [{label}]")
            lines.append("    by task (independent success)")
            for (task,), t_rows in _group(g_rows, ("task",)).items():
                t_ = stats(t_rows)
                lines.append(f"      {task:<45s} n={t_['n_trials']:<5} success={t_['success']:.4f}  "
                             f"grasp={t_['grasp']:.4f}  obj_grasped={t_['obj_grasped']:.4f}")
        lines.append("")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# segment_pose.csv (training) / eval_segment_pose.csv (eval)
# ---------------------------------------------------------------------------

def _to_numpy(t):
    return t.detach().cpu().numpy()


class SegmentPoseWriter:
    """Formats `get_all_slot_poses()` output into segment-pose CSV rows.

    With no `extra_fields` the output is byte-identical to the writer that used
    to live inline in `main.py::_record_segment_pose`. Eval passes
    `EVAL_POSE_EXTRA`, appended after `qz`, so a reader that selects training
    columns by name reads both files the same way.
    """

    def __init__(self, csv_path: Path, extra_fields: Sequence[str] = ()):
        self.csv_path = Path(csv_path)
        self.extra_fields = tuple(extra_fields)

    def write(self, poses: dict, instructions: Sequence[str], num_envs: int, *,
              episode: int, segment: int, phase: str, total_steps: int,
              extra_values: Optional[Sequence[Sequence[object]]] = None,
              to_numpy=_to_numpy) -> None:
        """`segment` is 1-based as written. `extra_values[k][env]` fills extra field k."""
        if self.extra_fields and (extra_values is None or len(extra_values) != len(self.extra_fields)):
            raise ValueError(f"expected {len(self.extra_fields)} extra columns {self.extra_fields}")
        write_hdr = not self.csv_path.exists()

        # Materialize on CPU once per boundary rather than per row — these are
        # GPU tensors and a per-row .item() would sync num_envs x (N+M) times.
        rows = []
        for kind, entries, name_lists in (
            ("obj", poses["obj"], poses["obj_names"]),
            ("recep", poses["recep"], poses["recep_names"]),
        ):
            for slot, (pair, names) in enumerate(zip(entries, name_lists)):
                p, q = pair
                rows.append((kind, slot, names, to_numpy(p), to_numpy(q)))
        g_p, g_q = poses["gripper"]
        rows.append(("gripper", 0, [""] * num_envs, to_numpy(g_p), to_numpy(g_q)))

        with open(self.csv_path, "a") as f:
            if write_hdr:
                f.write(",".join(SEGMENT_POSE_FIELDS + self.extra_fields) + "\n")
            for i in range(num_envs):
                # Quoted in case a task/model name ever contains a comma.
                task_str = str(instructions[i]).replace('"', '""') if i < len(instructions) else ""
                extra = ""
                if self.extra_fields:
                    extra = "".join(",\"" + str(col[i]).replace('"', '""') + "\"" for col in extra_values)
                for kind, slot, names, p, q in rows:
                    model = str(names[i]).replace('"', '""') if i < len(names) else ""
                    f.write(f"{episode},{segment},{phase},{total_steps},{i},{kind},{slot},"
                            f"\"{model}\",\"{task_str}\","
                            f"{p[i,0]:.6f},{p[i,1]:.6f},{p[i,2]:.6f},"
                            f"{q[i,0]:.6f},{q[i,1]:.6f},{q[i,2]:.6f},{q[i,3]:.6f}{extra}\n")
