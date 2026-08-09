"""Rebuild a correct per-trial eval baseline from an AutoRL run's artifacts.

Why this exists
---------------
AutoRL's `render()` reports its aggregate stats through `env_infos`, which is
appended to on every step where `truncated` is set. In `render_seq()` only the
first task of a sequence resets the env, so from the second task onward
`_elapsed_steps` is already at `max_episode_steps` and the env reports truncated
on *every* step. `env_infos["success"]` therefore holds `ep_len * num_envs`
samples instead of `num_envs`, and `np.mean` over it silently answers "what
fraction of timesteps was the object resting on the plate" instead of "did this
trial succeed". Since `success = src_on_target` is instantaneous (unlike the
grasp flags, which latch with `|=`), that systematically under-reports every
task after the first. The printed log line and `stats.yaml["stats"]` both come
from this path, as does `stats.yaml["last_info"]`, which indexes the
over-long list by env id and so reads the *first* step's snapshot.

The per-env terminal value survives elsewhere, untouched: when writing videos,
`render()` walks its own per-step `datas[i]["info"]` buffer and takes
`infos[-1]["success"]` — the last step, for env `i` specifically — and bakes it
into the filename:

    glob/{seq}-{task}/video_{env}-{obj}_{recep}-s_{0|1}.mp4

So the numbers we want were written to disk all along. This script recovers
them by pairing those filenames with `stats.yaml["instruction"]` (the per-env
task string), and emits the same schema as CRONOS's `eval_per_trial.csv` so
both sides feed the same `mcnemar_pair.py` / `plot.py`.

AutoRL is read-only here: this only reads files, and only under the directory
you point it at.

What cannot be recovered
------------------------
`is_src_obj_grasped` / `consecutive_grasp` are latched with `|=` and only
`reset_grasp_stats()` clears them — which `render_seq` never calls — so in
AutoRL they additionally carry over between tasks of a sequence. Their per-step
values were never written to disk (they live only in the in-memory `datas`
buffer, which is discarded), and both `stats.yaml` fields that hold them are
computed through the polluted path. The grasp columns below are therefore
emitted empty. Grasp comparisons against AutoRL are only sound for
`--only_render` (single-task) runs, where every task resets and the aggregate is
correct as-is.

Usage
-----
    python tools/parse_autorl_eval.py \
        --glob-dir /path/to/AutoRL/SimplerEnv/wandb/run-<id>/glob \
        --obj-set rand \
        --out reports/autorl_baseline_per_trial.csv
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from collections import defaultdict
from pathlib import Path

try:
    import yaml
except ImportError:  # pragma: no cover - yaml ships with the CRONOS envs
    yaml = None

# `render()` writes `video_{env}-{obj}_{recep}-s_{success}`, and
# `images_to_video` appends ".mp4" after replacing spaces with underscores. The
# middle chunk is greedy-free so a receptacle name containing "-s_" cannot eat
# the success marker; anchoring on the final "-s_<digits>.mp4" makes the split
# unambiguous regardless of underscores or hyphens in the model names.
_VIDEO_RE = re.compile(r"^video_(\d+)-(.+)-s_(\d+)\.mp4$")

# `render_seq` names its dirs "{sequence}-{task}"; `--only_render` uses "0-{i}".
# Anything else (e.g. the training loop's "vis_{epoch}_{obj_set}") is skipped.
_DIR_RE = re.compile(r"^(\d+)-(\d+)$")

FIELDS = (
    "seq_idx",
    "task_idx",
    "obj_set",
    "task",
    "env_idx",
    "success",
    "success_chained",
    "grasp",
    "obj_grasped",
    "prefix",
)


def _read_instructions(stats_path: Path) -> dict:
    """Return {env_idx: task string} from a render() stats.yaml, or {}.

    Only the `instruction` block is read. `stats` and `last_info` in the same
    file come from the polluted aggregation path and are deliberately ignored.
    """
    if yaml is None or not stats_path.exists():
        return {}
    try:
        data = yaml.safe_load(stats_path.read_text()) or {}
    except Exception as e:
        print(f"[warn] unreadable {stats_path}: {e}", file=sys.stderr)
        return {}
    instr = data.get("instruction") or {}
    if not isinstance(instr, dict):
        return {}
    out = {}
    for k, v in instr.items():
        try:
            out[int(k)] = str(v)
        except (TypeError, ValueError):
            continue
    return out


def parse_glob_dir(glob_dir: Path, obj_set: str) -> list:
    """Scan one AutoRL glob dir and return per-trial rows, chained score filled.

    Rows are ordered by (seq_idx, task_idx, env_idx) so the cumulative AND that
    produces `success_chained` sees each sequence's tasks in execution order.
    """
    if not glob_dir.is_dir():
        raise NotADirectoryError(f"not a directory: {glob_dir}")

    # (seq_idx, task_idx) -> {env_idx: (success, task_string)}
    trials: dict = defaultdict(dict)
    skipped_dirs = []

    for sub in sorted(glob_dir.iterdir()):
        if not sub.is_dir():
            continue
        m = _DIR_RE.match(sub.name)
        if not m:
            skipped_dirs.append(sub.name)
            continue
        seq_idx, task_idx = int(m.group(1)), int(m.group(2))
        instructions = _read_instructions(sub / "stats.yaml")

        for video in sorted(sub.glob("*.mp4")):
            vm = _VIDEO_RE.match(video.name)
            if not vm:
                print(f"[warn] unparsable video name, skipped: {video}", file=sys.stderr)
                continue
            env_idx = int(vm.group(1))
            success = float(int(vm.group(3)))
            # The task string from stats.yaml is authoritative; the filename's
            # middle chunk is "{obj}_{recep}" with spaces already collapsed to
            # underscores, so it cannot be split back apart reliably.
            task = instructions.get(env_idx)
            if task is None:
                task = vm.group(2).replace("_", " ")
            trials[(seq_idx, task_idx)][env_idx] = (success, task)

    if skipped_dirs:
        print(f"[info] skipped {len(skipped_dirs)} non-eval dir(s): "
              f"{', '.join(skipped_dirs[:5])}"
              f"{' ...' if len(skipped_dirs) > 5 else ''}", file=sys.stderr)

    if not trials:
        raise RuntimeError(
            f"no '<seq>-<task>' eval directories with videos found under {glob_dir}. "
            "This tool needs the mp4 files render() writes; a run whose videos "
            "were deleted keeps only the polluted aggregate in stats.yaml, and "
            "the correct per-trial values cannot be recovered from it — rerun "
            "AutoRL's eval to regenerate them."
        )

    # Cumulative AND per (sequence, env) across increasing task_idx.
    chain: dict = defaultdict(lambda: 1.0)
    rows = []
    for seq_idx, task_idx in sorted(trials.keys()):
        per_env = trials[(seq_idx, task_idx)]
        for env_idx in sorted(per_env):
            success, task = per_env[env_idx]
            key = (seq_idx, env_idx)
            if task_idx == 0:
                chain[key] = 1.0
            chain[key] = chain[key] * success
            rows.append({
                "seq_idx": seq_idx,
                "task_idx": task_idx,
                "obj_set": obj_set,
                "task": task,
                "env_idx": env_idx,
                "success": f"{success:.4f}",
                "success_chained": f"{chain[key]:.4f}",
                # Not recoverable for sequential runs — see module docstring.
                "grasp": "",
                "obj_grasped": "",
                "prefix": f"autorl_seq{seq_idx}_task{task_idx}",
            })

    # A ragged env count means some videos are missing; the chained score would
    # silently skip a task for those envs, so say so rather than average over it.
    counts = {k: len(v) for k, v in trials.items()}
    if len(set(counts.values())) > 1:
        print(f"[warn] uneven env counts across eval dirs: "
              f"{sorted(set(counts.values()))}. Some videos are missing; "
              f"success_chained for the affected envs skips those tasks.",
              file=sys.stderr)

    return rows


def summarize(rows: list) -> None:
    """Print per-(seq, task) means for both scoring semantics."""
    agg = defaultdict(lambda: {"success": [], "success_chained": []})
    for r in rows:
        key = (r["seq_idx"], r["task_idx"], r["task"])
        agg[key]["success"].append(float(r["success"]))
        agg[key]["success_chained"].append(float(r["success_chained"]))

    print(f"\n{'seq':>4} {'task':>5} {'n':>4}  {'success(B)':>10} {'chained(A)':>10}  instruction")
    print("-" * 78)
    for (seq_idx, task_idx, task) in sorted(agg.keys()):
        v = agg[(seq_idx, task_idx, task)]
        n = len(v["success"])
        b = sum(v["success"]) / n if n else 0.0
        a = sum(v["success_chained"]) / n if n else 0.0
        print(f"{seq_idx:>4} {task_idx:>5} {n:>4}  {b:>10.4f} {a:>10.4f}  {task}")


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Rebuild per-trial eval results from an AutoRL run's videos "
                    "(read-only; AutoRL is never modified)."
    )
    ap.add_argument("--glob-dir", required=True, type=Path,
                    help="AutoRL run's glob/ directory (contains <seq>-<task>/ dirs)")
    ap.add_argument("--obj-set", default="rand",
                    help="obj_set label the run was launched with; AutoRL does not "
                         "record it in the artifacts (default: rand)")
    ap.add_argument("--out", type=Path, default=None,
                    help="output CSV path (default: <glob-dir>/eval_per_trial_recovered.csv)")
    ap.add_argument("--quiet", action="store_true", help="skip the summary table")
    args = ap.parse_args()

    rows = parse_glob_dir(args.glob_dir, args.obj_set)
    out = args.out or (args.glob_dir / "eval_per_trial_recovered.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(FIELDS))
        w.writeheader()
        w.writerows(rows)

    if not args.quiet:
        summarize(rows)
    print(f"\nWrote {len(rows)} trial rows -> {out}")
    print("Columns match CRONOS eval_per_trial.csv; grasp/obj_grasped are empty "
          "(not recoverable for sequential runs — see module docstring).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
