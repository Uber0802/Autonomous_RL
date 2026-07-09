"""Compute the pooled paired-McNemar gate.

Reads two `eval_per_trial.csv` files (zero-shot baseline + post-RL eval at
the SAME `--seed`) and computes the paired McNemar statistic on
the pooled discordant pairs across all tasks of one `eval_kind`
(`in_domain` for the plan's primary gate; `out_of_domain` for context).

Each row of the input CSV is one trial:
    seq_idx,task_idx,obj_set,task,env_idx,success,grasp,obj_grasped,prefix

Pairing key: `(eval_kind, task, env_idx)`. Under the same `--seed`,
the episodic `eval_only.py` reset is seed-determined and policy-
independent (wrapper.py:42-44), so two runs at the same seed yield byte-
identical inits per (task, env_idx) — no `episode_id` is needed in the key.

McNemar test:
  - b = pairs where baseline=0, post-RL=1  (gains)
  - c = pairs where baseline=1, post-RL=0  (losses)
  - χ² = (|b - c| - 1)² / (b + c)          (Edwards continuity correction)
  - one-sided p for "post-RL > baseline":   1 - F_χ²(χ², df=1) / 2 if b > c,
                                             else 1 - (1 - F_χ²(χ², df=1) / 2)

The plan's gate rejects "no improvement" at α=0.05 one-sided.
"""
import argparse
import csv
import math
import sys
from pathlib import Path


def read_per_trial(path: Path):
    """Return {(eval_kind, task, env_idx): {success, grasp, obj_grasped}}."""
    by = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            kind = "in_domain" if row["prefix"].startswith("in_domain") else \
                   ("out_of_domain" if row["prefix"].startswith("out_of_domain") else None)
            if kind is None:
                continue
            key = (kind, row["task"], int(row["env_idx"]))
            by[key] = {
                "success": float(row["success"]),
                "grasp": float(row["grasp"]),
                "obj_grasped": float(row["obj_grasped"]),
            }
    return by


def chi2_one_sided_p(chi2: float, b: int, c: int) -> float:
    """One-sided p-value for `b > c` from the McNemar χ²(df=1) statistic.

    Under H0, b ~ Bin(b+c, 0.5). The two-sided McNemar p is symmetric in
    (b, c); the one-sided p for the "improvement" alternative is half the
    two-sided p when b > c, and `1 - half_two_sided_p` when b < c (i.e.,
    the direction is wrong).

    We use the closed-form χ²(1) survival function instead of pulling in
    scipy (the spatialvla_cronos env doesn't ship it): for χ²_1,
    survival(x) = erfc(sqrt(x / 2)).
    """
    if (b + c) == 0:
        return 1.0
    p_two_sided = math.erfc(math.sqrt(chi2 / 2.0))
    if b > c:
        return p_two_sided / 2.0
    else:
        return 1.0 - p_two_sided / 2.0


def mcnemar(baseline, post_rl, kind: str, metric: str = "success"):
    """Pooled McNemar over all tasks of one eval_kind, computed on `metric`."""
    pairs_total = 0
    a = b = c = d = 0                                       # confusion-matrix counts
    per_task = {}                                           # task -> (baseline_succ, post_succ, n)

    for key, base_row in baseline.items():
        if key[0] != kind:
            continue
        post_row = post_rl.get(key)
        if post_row is None:
            continue
        pairs_total += 1
        bv = int(round(base_row[metric]))
        pv = int(round(post_row[metric]))
        if   bv == 0 and pv == 0: d += 1
        elif bv == 0 and pv == 1: b += 1
        elif bv == 1 and pv == 0: c += 1
        elif bv == 1 and pv == 1: a += 1
        # Per-task aggregates for the insight table.
        task = key[1]
        slot = per_task.setdefault(task, [0, 0, 0])
        slot[0] += bv
        slot[1] += pv
        slot[2] += 1

    # McNemar χ² with Edwards continuity correction.
    if (b + c) == 0:
        chi2 = 0.0
    else:
        chi2 = (abs(b - c) - 1) ** 2 / (b + c) if (b + c) >= 1 else float("nan")
    p_one = chi2_one_sided_p(chi2, b, c)
    return {
        "kind": kind,
        "metric": metric,
        "pairs_total": pairs_total,
        "agree_zero_zero": d,
        "agree_one_one": a,
        "discordant_b_gain": b,
        "discordant_c_loss": c,
        "chi2": chi2,
        "p_one_sided": p_one,
        "reject_at_0_05": (p_one < 0.05),
        "baseline_mean": sum(per_task[t][0] / per_task[t][2] for t in per_task) / len(per_task) if per_task else 0.0,
        "post_mean":     sum(per_task[t][1] / per_task[t][2] for t in per_task) / len(per_task) if per_task else 0.0,
        "per_task":      per_task,
    }


def main():
    p = argparse.ArgumentParser("mcnemar_pair")
    p.add_argument("--baseline", required=True, type=Path,
                   help="zero-shot baseline eval_per_trial.csv")
    p.add_argument("--post-rl", required=True, type=Path,
                   help="post-RL eval_per_trial.csv (SAME `--seed`, post-PPO checkpoint)")
    p.add_argument("--kinds", default="in_domain,out_of_domain",
                   help="comma-separated eval kinds to test")
    p.add_argument("--metric", default="success",
                   choices=["success", "grasp", "obj_grasped"],
                   help="binary outcome to McNemar on (default: success)")
    args = p.parse_args()

    base = read_per_trial(args.baseline)
    post = read_per_trial(args.post_rl)

    print(f"baseline: {args.baseline}  rows={len(base)}")
    print(f"post-RL : {args.post_rl}  rows={len(post)}")
    print()

    for kind in args.kinds.split(","):
        r = mcnemar(base, post, kind, metric=args.metric)
        print(f"### {kind}  ({args.metric})")
        print(f"  pairs            : {r['pairs_total']}")
        print(f"  baseline mean    : {r['baseline_mean']:.4f}  (per-task avg)")
        print(f"  post-RL mean     : {r['post_mean']:.4f}  (per-task avg)")
        print(f"  confusion (a,b,c,d) = (1→1, 0→1 gains, 1→0 losses, 0→0)")
        print(f"                    a={r['agree_one_one']}  b={r['discordant_b_gain']}  c={r['discordant_c_loss']}  d={r['agree_zero_zero']}")
        print(f"  McNemar χ²      = {r['chi2']:.4f}  (df=1, Edwards continuity)")
        print(f"  one-sided p     = {r['p_one_sided']:.6f}  (alt: post-RL > baseline)")
        print(f"  reject H0@α=0.05 : {'YES (learns)' if r['reject_at_0_05'] else 'NO'}")
        print()
        # Per-task insight (not the gate; the gate is the pooled test above).
        print(f"  per-task (baseline | post-RL | n):")
        for task in sorted(r["per_task"].keys()):
            bs, ps, n = r["per_task"][task]
            print(f"    {task:<45s}  {bs/n*100:6.2f}%  →  {ps/n*100:6.2f}%   ({n} trials)")
        print()


if __name__ == "__main__":
    main()
