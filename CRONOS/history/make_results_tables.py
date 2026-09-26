"""Render the paper-experiment result tables for doc/results/paper_experiments.md.

Usage (from CRONOS/):
    python history/make_results_tables.py <summary_snapshot_dir> <seq_eval_snapshot_dir> > /tmp/tables.md

<summary_snapshot_dir> holds Q1/..Q6/ with `*_summary.csv` written by
plotting/plot_eval_success.py; <seq_eval_snapshot_dir> holds Q6_seq_opVLA/ and
Q6_seq_SpVLA/ written by plotting/plot_sequence_eval.py. The output is pasted
into the doc between the TABLES markers.
"""
import csv
import sys
from collections import OrderedDict
from pathlib import Path

SUMMARIES = [
    ("Q1", "OpenVLA", "Q1/Q1_opVLA_summary.csv"),
    ("Q1", "SpatialVLA", "Q1/Q1_SpVLA_summary.csv"),
    ("Q2", "SpatialVLA", "Q2/Q2_summary.csv"),
    ("Q3", "SpatialVLA", "Q3/Q3_summary.csv"),
    ("Q4", "SpatialVLA", "Q4/Q4_summary.csv"),
    ("Q5", "SpatialVLA", "Q5/Q5_summary.csv"),
    ("Q6", "OpenVLA", "Q6/Q6_opVLA_summary.csv"),
    ("Q6", "SpatialVLA", "Q6/Q6_SpVLA_summary.csv"),
]
SEQ = [("OpenVLA", "Q6_seq_opVLA"), ("SpatialVLA", "Q6_seq_SpVLA")]


def label(g):
    return " ".join(g.replace("w\\o", "w/o").replace("w\\", "w/").split())


def pm(mean, std):
    return f"{float(mean):.3f} ± {float(std):.3f}"


def summary_table(path):
    rows = [r for r in csv.DictReader(open(path)) if r["x_axis"] == "total_steps"]
    by = OrderedDict()
    for r in rows:
        by.setdefault(r["group"], {})[(r["eval_kind"], r["metric"])] = r
    out = ["| Group | Env steps (M) | ID success | ID grasp | OOD success | OOD grasp | Seeds |",
           "|---|---:|---:|---:|---:|---:|---:|"]
    for g, d in by.items():
        any_r = next(iter(d.values()))
        cells = []
        for k in [("in_domain", "success"), ("in_domain", "grasp"),
                  ("out_of_domain", "success"), ("out_of_domain", "grasp")]:
            r = d.get(k)
            cells.append(pm(r["final_mean"], r["final_std"]) if r else "—")
        steps = float(any_r["final_x"]) / 1e6
        out.append(f"| {label(g)} | {steps:.2f} | " + " | ".join(cells) + f" | {any_r['n_runs']} |")
    return "\n".join(out)


def seq_table(path, metric):
    rows = [r for r in csv.DictReader(open(path))
            if r["metric"] == metric and r["task"] == "__all__"]
    groups = list(OrderedDict.fromkeys(r["group"] for r in rows))
    out = ["| Group | Rounds | Domain | Pos 1 | Pos 2 | Pos 3 | Pos 4 |",
           "|---|---|---|---:|---:|---:|---:|"]
    for g in groups:
        for kind, kname in [("training", "seen"), ("random", "unseen")]:
            for dom, dname in [("in_domain", "ID"), ("out_of_domain", "OOD")]:
                v = {r["position"]: float(r["mean"]) for r in rows
                     if r["group"] == g and r["seq_kind"] == kind and r["eval_kind"] == dom}
                if not v:
                    continue
                cells = [f"{v[p]:.3f}" if p in v else "—" for p in "1234"]
                out.append(f"| {label(g)} | {kname} | {dname} | " + " | ".join(cells) + " |")
    return "\n".join(out)


def main(summ_dir, seq_dir):
    summ_dir, seq_dir = Path(summ_dir), Path(seq_dir)
    for q, vla, rel in SUMMARIES:
        print(f"#### {q} — {vla}\n")
        print(summary_table(summ_dir / rel) + "\n")
    for vla, d in SEQ:
        for metric in ["success", "success_chained"]:
            print(f"#### Sequential eval — {vla}, `{metric}` by task position\n")
            print(seq_table(seq_dir / d / "seq_eval_seq_position.csv", metric) + "\n")


if __name__ == "__main__":
    main(*sys.argv[1:3])
