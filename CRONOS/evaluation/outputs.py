"""Derived eval outputs, unit completion, resume and merge. Stdlib only.

One function, `rebuild_outputs`, produces every aggregate file from the source
rows (`eval_per_trial.csv`, `eval_layouts.csv`). The live eval calls it at the
end; `--eval-resume` calls it after finishing the remaining units;
`tools/rebuild_eval_outputs.py` calls it to repair a run or merge shards. So the
three paths cannot disagree about what "complete" means or produce different
aggregates from the same rows.

Completeness is per unit = (eval_kind, pass_label, round). A unit is done when
the per-trial file holds exactly `num_envs × task slots` rows for it. A unit with
fewer rows was cut short by a crash; its rows are dropped before resuming because
chained success needs the whole round. More rows than that means two runs wrote
the same unit, which is an error.

`eval_success.csv` is written only when every planned unit is done, coverage
matches, AND the planned rounds are the whole design (every round of every pose
set, for each selected domain). A shard (`rounds: 0-2`) is complete for its own
selection but gets no `eval_success.csv` until merged. Otherwise the file is
removed (if present) and `eval_status.json` says why — an absent file means "not a
complete eval", never "complete but empty".
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from evaluation import records as R
from evaluation.plan import coverage_from_units, fingerprint_differences, format_coverage_table, unit_key

UnitKey = Tuple[str, str, int]


def planned_units(plans: Sequence[dict]) -> Dict[UnitKey, dict]:
    """Union of the units of one or more plan dicts (shards of one eval)."""
    out: Dict[UnitKey, dict] = {}
    for plan in plans:
        for u in plan["units"]:
            out[unit_key(u)] = u
    return out


def unit_row_counts(trials: Iterable[dict]) -> Dict[UnitKey, int]:
    counts: Dict[UnitKey, int] = {}
    for r in trials:
        k = unit_key(r)
        counts[k] = counts.get(k, 0) + 1
    return counts


def completed_units(trials: Iterable[dict], units: Dict[UnitKey, dict]) -> Tuple[set, set, Dict[UnitKey, int]]:
    """(done, partial, overfull) against the planned trial count of each unit."""
    counts = unit_row_counts(trials)
    done, partial, overfull = set(), set(), {}
    for k, n in counts.items():
        expected = units[k]["trials"] if k in units else None
        if expected is None:
            overfull[k] = n            # rows for a unit no plan asked for
        elif n == expected:
            done.add(k)
        elif n < expected:
            partial.add(k)
        else:
            overfull[k] = n
    return done, partial, overfull


def truncate_to_units(glob_dir: Path, keep: set) -> Dict[str, int]:
    """Drop every source row whose unit is not in `keep`. Returns rows dropped per file."""
    glob_dir = Path(glob_dir)

    def keep_row(row):
        try:
            return unit_key(row) in keep
        except (KeyError, ValueError):
            return False

    return {name: R.filter_rows_in_place(glob_dir / name, keep_row)
            for name in (R.PER_TRIAL, R.LAYOUTS, R.POSES)}


def check_same_eval(plans: Sequence[dict]) -> None:
    base = plans[0]["fingerprint"]
    for other in plans[1:]:
        if other["fingerprint"].get("hash") != base.get("hash"):
            diffs = fingerprint_differences(base.get("body", {}), other["fingerprint"].get("body", {}))
            raise ValueError("eval runs are not shards of the same eval (fingerprints differ):\n  " +
                             "\n  ".join(diffs or ["hash differs"]))


def write_status(glob_dir: Path, plans: Sequence[dict], trials: List[dict], extra: Optional[dict] = None) -> dict:
    units = planned_units(plans)
    done, partial, overfull = completed_units(trials, units)
    missing = sorted(set(units) - done)
    domains = sorted({k[0] for k in units})
    # Design size per pass: the largest any input declares. pose_sets is not in
    # the fingerprint, so a pose_sets=2 shard extends a pose_sets=1 run's design.
    design: Dict[str, int] = {}
    for plan in plans:
        for label, total in plan.get("design", {}).items():
            design[label] = max(design.get(label, 0), int(total))
    not_selected = sorted((d, label, r) for d in domains for label, total in design.items()
                          for r in range(total) if (d, label, r) not in units)
    status = dict(
        complete=not missing and not overfull,
        full_design=not not_selected,
        rounds_not_selected=len(not_selected),
        planned_units=len(units), done_units=len(done),
        missing=[list(k) for k in missing],
        partial=[list(k) for k in sorted(partial)],
        overfull={"|".join(map(str, k)): n for k, n in overfull.items()},
        fingerprint=plans[0]["fingerprint"].get("hash"),
        updated=datetime.now(timezone.utc).isoformat(),
        **(extra or {}),
    )
    R.atomic_write_text(Path(glob_dir) / R.STATUS, json.dumps(status, indent=2) + "\n")
    return status


def _eval_success_rows(summary: List[dict], plans: Sequence[dict]) -> List[dict]:
    backgrounds = {sc["name"]: sc["background"] for plan in plans for sc in plan["scenes"]}
    progress = plans[0].get("checkpoint_progress", {})
    rows = []
    for r in summary:
        if r["level"] != "task":
            continue
        rows.append(dict(
            episode=progress.get("episode", 0), total_steps=progress.get("total_steps", 0), total_resets=0,
            eval_kind=f"{r['eval_kind']}_{r['seq_kind']}", group=r["group"], task=r["task"],
            scene=backgrounds.get(r["group"], "default"), n_envs=r["n_trials"],
            success=round(r["success"], 6), grasp=round(r["grasp"], 6), obj_grasped=round(r["obj_grasped"], 6),
        ))
    return rows


def wandb_payload(summary: List[dict]) -> dict:
    out = {}
    for r in summary:
        base = f"eval_{r['eval_kind']}_{r['seq_kind']}"
        if r["level"] == "task":
            out[f"{base}/{r['task']}_success"] = r["success"]
            out[f"{base}/{r['task']}_grasp"] = r["grasp"]
            out[f"{base}/{r['task']}_obj_grasped"] = r["obj_grasped"]
        elif r["level"] == "scene":
            out[f"{base}/{r['group']}/mean_success"] = r["success"]
            out[f"{base}/{r['group']}/mean_success_chained"] = r["success_chained"]
        elif r["level"] == "kind":
            out[f"{base}/all_scenes_success"] = r["success"]
            out[f"{base}/all_scenes_success_chained"] = r["success_chained"]
    return out


def refuse_training_dir(glob_dir: Path) -> None:
    """Standalone eval owns `eval_success.csv` in its directory and may delete it.
    A training run's glob holds training-time eval rows under the same name."""
    if (Path(glob_dir) / "rollout_success.csv").exists():
        raise ValueError(f"{glob_dir} is a training run directory (has rollout_success.csv); "
                         f"standalone eval outputs must go to their own directory")


def rebuild_outputs(glob_dir: Path, plans: Sequence[dict], report_name: str = "eval_report.txt",
                    header: str = "") -> dict:
    """Rebuild every derived file in `glob_dir` from its source rows.

    `plans` are the plan dicts whose units this directory is meant to hold (one
    for a normal or resumed run; several for a merge). Returns
    {status, summary, coverage, wandb} — `wandb` is empty unless complete.
    """
    glob_dir = Path(glob_dir)
    refuse_training_dir(glob_dir)
    check_same_eval(plans)
    trials = R.typed_trials(R.read_rows(glob_dir / R.PER_TRIAL))
    layouts = R.read_rows(glob_dir / R.LAYOUTS)
    units = planned_units(plans)

    summary = R.summary_rows(trials)
    R.atomic_write_csv(glob_dir / R.SUMMARY, R.SUMMARY_FIELDS,
                       ({**r, **{k: f"{r[k]:.6f}" for k in ("success", "success_chained", "grasp", "obj_grasped")}}
                        for r in summary))
    coverage = R.coverage_rows(coverage_from_units(list(units.values())), trials, layouts)
    R.atomic_write_csv(glob_dir / R.COVERAGE, R.COVERAGE_FIELDS, coverage)

    status = write_status(glob_dir, plans, trials,
                          extra=dict(coverage_match=all(c["match"] for c in coverage)))
    complete = status["complete"] and status["coverage_match"]
    full = complete and status["full_design"]
    if full:
        state = "COMPLETE"
    elif complete:
        state = (f"COMPLETE SELECTION, NOT FULL DESIGN ({status['rounds_not_selected']} unit(s) of the "
                 f"design not selected — merge shards for eval_success.csv)")
    else:
        state = f"INCOMPLETE ({status['done_units']}/{status['planned_units']} units)"
    report = (header + f"status: {state}\n\n" + R.render_report(trials) +
              format_coverage_table(coverage, title="actual coverage") + "\n")
    R.atomic_write_text(glob_dir / report_name, report)

    success_path = glob_dir / R.SUCCESS
    if full:
        from training.metrics import EVAL_FIELDS
        R.atomic_write_csv(success_path, EVAL_FIELDS, _eval_success_rows(summary, plans))
    elif success_path.exists():
        success_path.unlink()
    return dict(status=status, summary=summary, coverage=coverage, report=report,
                wandb=wandb_payload(summary) if full else {})
