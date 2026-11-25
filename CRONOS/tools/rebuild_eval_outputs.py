"""Rebuild standalone-eval outputs from per-trial rows; merge shards of one eval.

Repair one run (e.g. its process died during the final aggregation):

    python tools/rebuild_eval_outputs.py <glob>

Merge shards run with different `rounds` / `domains` selections into one
directory:

    python tools/rebuild_eval_outputs.py --out <merged_glob> <glob_a> <glob_b> ...

Checks, in order, failing loudly:
  - every input has eval_plan.json and all fingerprints are equal (same checkpoint,
    config scenes, seeds, sampling settings — only the selection may differ);
  - no unit appears complete in two inputs.
A unit that is partial in its input (crashed shard) is left out of the merge and
reported as missing. The merged directory gets the concatenated source CSVs
(per-trial, layouts, poses — complete units only), `eval_plan_merged.json`
listing the source plans, and every derived file from
`evaluation.outputs.rebuild_outputs`. `eval_success.csv` appears only if the
union of the inputs' units is complete and covers every round of the design. Videos are not copied.

Runs from the CRONOS directory (imports `evaluation`).
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluation import outputs as O          # noqa: E402
from evaluation import records as R          # noqa: E402
from evaluation.plan import unit_key         # noqa: E402


def load_plan(glob_dir: Path) -> dict:
    path = glob_dir / "eval_plan.json"
    if not path.exists():
        raise SystemExit(f"{glob_dir}: no eval_plan.json")
    return json.loads(path.read_text())


def merge(out: Path, inputs, report_name: str) -> dict:
    plans = [load_plan(g) for g in inputs]
    O.check_same_eval(plans)
    out.mkdir(parents=True, exist_ok=True)
    O.refuse_training_dir(out)
    if any((out / name).exists() for name in (R.PER_TRIAL, R.LAYOUTS, R.POSES)):
        raise SystemExit(f"{out} already has source rows; merge into an empty directory")

    owner = {}
    keep_per_input = []
    for g, plan in zip(inputs, plans):
        units = O.planned_units([plan])
        done, partial, overfull = O.completed_units(R.read_rows(g / R.PER_TRIAL), units)
        if overfull:
            raise SystemExit(f"{g}: units with more rows than planned {overfull}")
        for k in done:
            if k in owner:
                raise SystemExit(f"unit {k} is complete in both {owner[k]} and {g}")
            owner[k] = g
        if partial:
            print(f"[merge] {g}: partial units left out {sorted(partial)}", file=sys.stderr)
        keep_per_input.append(done)

    for name, fields in ((R.PER_TRIAL, R.PER_TRIAL_FIELDS), (R.LAYOUTS, R.LAYOUT_FIELDS)):
        for g, keep in zip(inputs, keep_per_input):
            rows = [r for r in R.read_rows(g / name) if unit_key(r) in keep]
            if rows:
                R.append_rows(out / name, fields, rows)
    # Pose rows: copy lines verbatim (training formatting), one header.
    header, lines = None, []
    for g, keep in zip(inputs, keep_per_input):
        path = g / R.POSES
        if not path.exists():
            continue
        with open(path, newline="") as f:
            content = f.read().splitlines(keepends=True)
        if not content:
            continue
        if header is None:
            header = content[0]
        elif content[0] != header:
            raise SystemExit(f"{path}: pose CSV header differs from the first input")
        import csv
        cols = next(csv.reader([content[0]]))
        for line in content[1:]:
            row = dict(zip(cols, next(csv.reader([line]))))
            if unit_key(row) in keep:
                lines.append(line)
    if header is not None:
        R.atomic_write_text(out / R.POSES, header + "".join(lines))

    merged = dict(sources=[str(g) for g in inputs], plans=plans)
    R.atomic_write_text(out / "eval_plan_merged.json", json.dumps(merged, indent=2) + "\n")
    # Units and fingerprint of the merge: the first plan's body, union of units.
    union = dict(plans[0], units=list(O.planned_units(plans).values()))
    R.atomic_write_text(out / "eval_plan.json", json.dumps(union, indent=2) + "\n")
    return O.rebuild_outputs(out, plans, report_name=report_name,
                             header=f"Merged standalone eval from {len(inputs)} run(s)\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("globs", nargs="+", type=Path)
    ap.add_argument("--out", type=Path, default=None, help="merge into this directory")
    ap.add_argument("--report-name", default="eval_report.txt")
    args = ap.parse_args()

    if args.out is None:
        if len(args.globs) != 1:
            raise SystemExit("several inputs need --out <merged_dir>")
        g = args.globs[0]
        result = O.rebuild_outputs(g, [load_plan(g)], report_name=args.report_name,
                                   header="Rebuilt standalone eval outputs\n")
    else:
        result = merge(args.out, args.globs, args.report_name)
    st = result["status"]
    print(f"complete={st['complete']} coverage_match={st['coverage_match']} full_design={st['full_design']} "
          f"units {st['done_units']}/{st['planned_units']}")
    if st["missing"]:
        print(f"missing units: {st['missing']}")
    sys.exit(0 if st["complete"] and st["coverage_match"] and st["full_design"] else 1)


if __name__ == "__main__":
    main()
