# Paper experiments (Q1–Q6)

Index of the documentation: [`../README.md`](../README.md).

Final numbers of the experiment series used for the paper, one section per
question. Detailed learning curves, rollout curves and position plots are
produced from the same runs by the plotting tools (not in this release).

## How to read these tables

- **Source.** Each table is the `*_summary.csv` written by
  `plotting/plot_eval_success.py` from the runs' `eval_success.csv` (training-time
  eval, per-env rotation over each scene's 4 tasks; see the top-level README,
  *Eval Design*). Values are the **last eval point** of each group on the
  `total_steps` axis, mean ± std across seeds; "Seeds" is the number of runs
  averaged. The sequential-eval tables come from standalone eval
  (`eval_only.py`, [`../eval_sequential.md`](../eval_sequential.md)) of the Q6
  final checkpoints.
- **Snapshot.** Summaries from the 2026-09-26 figure run; sequential eval from
  2026-09-25. Regenerate the tables with `history/make_results_tables.py` (local).
- **Common setup.** `configs/four_group_sequential_2x2.yaml` (4 scenes × 16 envs,
  4 tasks each), `segment_len = 80`, PPO unless stated. `T` is the episode length
  (`episode_len`) in env steps; T = 80 resets every segment, T = 2560 every 32.
- **Abbreviations.** ID / OOD — in-domain / out-of-domain eval. EER — end-effector
  reset at every segment boundary. CL — curriculum, T = 320 first and T = 2560
  after (the resume chain switches horizon). HSR / LSR / PER — see
  [`../reset_modes.md`](../reset_modes.md).
- **Group labels** are the labels of the plot configs. Where a setting below is
  marked *(inferred)* it was reconstructed from the run names, not read from the
  runs' `run_config.json` — confirm before quoting.
- **Seeds.** Two seeds per group (one for SpatialVLA T = 80 in Q6), so the ± is a
  two-point spread, not a confidence interval.
- **Code versions.** The runs span V0.4.x – V0.95; check
  [`bug_reports.md`](bug_reports.md) before mixing them with newer runs.

## Summary of findings

| Q | Question | Answer (at the step budget of the table) |
|---|---|---|
| Q1 | Does a longer horizon hurt at a fixed step budget? | Yes, sharply, for both VLAs: ID success T = 80 ≈ 0.64–0.66, T = 320 ≈ 0.25–0.36, T = 2560 ≈ 0.01. |
| Q2 | Does resetting the end effector (EER) help? | Yes: T = 320 ID success 0.25 → 0.53; T = 2560 0.01 → 0.11. |
| Q3 | Which reset strategy works at a long horizon? | HSR is best (0.65 ID success), far above periodic reset (0.14); LSR is unstable (0.13 ± 0.12); perturbation fails (0.01); HSR without episodic resets is high-variance (0.42 ± 0.31). |
| Q4 | Does the curriculum help at T = 2560? | Only together with EER: CL + EER 0.40 vs EER alone 0.14; without EER both are ≈ 0. |
| Q5 | GRPO vs PPO? | GRPO collapses to ≈ 0 at both horizons — see [`grpo.md`](grpo.md). |
| Q6 | Where does each setting end up with a longer budget (2.95 M steps)? | T = 80 and T = 320 + EER reach ≈ 0.77–0.83 ID success; T = 2560 + EER stays low (≤ 0.04); CL lifts T = 2560 to 0.31–0.40. Sequential eval: success drops with the task's position in the round, and chained success at position 4 is ≤ 0.24 for every group. |

---

## Q1 — Horizon

Episode length T ∈ {80, 320, 2560}, `normal` reset, EER off *(inferred: these
runs are the "w/o EER" arm of Q2)*, 0.66 M env steps.

**OpenVLA**

| Group | Env steps (M) | ID success | ID grasp | OOD success | OOD grasp | Seeds |
|---|---:|---:|---:|---:|---:|---:|
| T=80 | 0.66 | 0.658 ± 0.033 | 0.840 ± 0.023 | 0.469 ± 0.043 | 0.748 ± 0.029 | 2 |
| T=320 | 0.66 | 0.361 ± 0.072 | 0.650 ± 0.029 | 0.223 ± 0.039 | 0.564 ± 0.010 | 2 |
| T=2560 | 0.66 | 0.004 ± 0.004 | 0.033 ± 0.006 | 0.002 ± 0.002 | 0.012 ± 0.008 | 2 |

**SpatialVLA**

| Group | Env steps (M) | ID success | ID grasp | OOD success | OOD grasp | Seeds |
|---|---:|---:|---:|---:|---:|---:|
| T=80 | 0.66 | 0.637 ± 0.023 | 0.816 ± 0.039 | 0.514 ± 0.029 | 0.758 ± 0.020 | 2 |
| T=320 | 0.66 | 0.252 ± 0.014 | 0.537 ± 0.006 | 0.178 ± 0.018 | 0.428 ± 0.045 | 2 |
| T=2560 | 0.66 | 0.008 ± 0.004 | 0.105 ± 0.043 | 0.002 ± 0.002 | 0.086 ± 0.023 | 2 |

Grasp falls with the horizon as well as success, so the drop is not only a
placement problem: at T = 2560 the policy rarely even grasps.

## Q2 — End-effector reset (EER)

SpatialVLA, T ∈ {320, 2560}, with and without EER, 0.66 M env steps.

| Group | Env steps (M) | ID success | ID grasp | OOD success | OOD grasp | Seeds |
|---|---:|---:|---:|---:|---:|---:|
| T=320 w/o EER | 0.66 | 0.252 ± 0.014 | 0.537 ± 0.006 | 0.178 ± 0.018 | 0.428 ± 0.045 | 2 |
| T=320 w/ EER | 0.66 | 0.525 ± 0.092 | 0.740 ± 0.033 | 0.453 ± 0.074 | 0.664 ± 0.031 | 2 |
| T=2560 w/o EER | 0.66 | 0.008 ± 0.004 | 0.105 ± 0.043 | 0.002 ± 0.002 | 0.086 ± 0.023 | 2 |
| T=2560 w/ EER | 0.66 | 0.113 ± 0.047 | 0.404 ± 0.104 | 0.102 ± 0.059 | 0.332 ± 0.098 | 2 |

## Q3 — Reset strategy at a long horizon

SpatialVLA, 1.31 M env steps. Groups *(inferred from run names)*: **PR** — the
T = 2560 + EER run with the `normal` (periodic, per-episode) reset, the same run as
Q4's "EER-only"; **HSR**, **LSR**, **PER** (perturbation of the LSR reset goal) and
**HSR-only** (`noep`: HSR with no episodic reset) — the corresponding `train.sh`
reset modes.

| Group | Env steps (M) | ID success | ID grasp | OOD success | OOD grasp | Seeds |
|---|---:|---:|---:|---:|---:|---:|
| PR | 1.31 | 0.139 ± 0.072 | 0.551 ± 0.074 | 0.084 ± 0.033 | 0.439 ± 0.014 | 2 |
| HSR | 1.31 | 0.648 ± 0.020 | 0.814 ± 0.018 | 0.572 ± 0.041 | 0.760 ± 0.053 | 2 |
| LSR | 1.31 | 0.127 ± 0.123 | 0.344 ± 0.289 | 0.094 ± 0.090 | 0.254 ± 0.227 | 2 |
| PER | 1.31 | 0.010 ± 0.002 | 0.197 ± 0.049 | 0.004 ± 0.004 | 0.170 ± 0.088 | 2 |
| HSR-only | 1.31 | 0.416 ± 0.314 | 0.850 ± 0.029 | 0.354 ± 0.279 | 0.799 ± 0.045 | 2 |

HSR-only's grasp is the highest (0.85) while its success varies widely between
seeds: with no episodic reset, start states drift toward already-satisfied tasks,
which makes its rollout metrics optimistic ([`../reset_modes.md`](../reset_modes.md)).

## Q4 — Curriculum × EER at T = 2560

SpatialVLA, 1.31 M env steps, a 2 × 2 of curriculum (CL, T = 320 → 2560) and EER.

| Group | Env steps (M) | ID success | ID grasp | OOD success | OOD grasp | Seeds |
|---|---:|---:|---:|---:|---:|---:|
| noCL, noEER | 1.31 | 0.004 ± 0.000 | 0.018 ± 0.018 | 0.004 ± 0.004 | 0.014 ± 0.006 | 2 |
| CL-only | 1.31 | 0.006 ± 0.002 | 0.059 ± 0.012 | 0.002 ± 0.002 | 0.020 ± 0.012 | 2 |
| EER-only | 1.31 | 0.139 ± 0.072 | 0.551 ± 0.074 | 0.084 ± 0.033 | 0.439 ± 0.014 | 2 |
| CL+EER | 1.31 | 0.398 ± 0.125 | 0.820 ± 0.012 | 0.324 ± 0.043 | 0.771 ± 0.025 | 2 |

## Q5 — GRPO vs PPO

SpatialVLA, T ∈ {80, 2560}, 0.66 M env steps. The PPO rows are the Q1 T = 80 run
and the Q2 T = 2560 + EER run at the same budget.

| Group | Env steps (M) | ID success | ID grasp | OOD success | OOD grasp | Seeds |
|---|---:|---:|---:|---:|---:|---:|
| GRPO T=80 | 0.66 | 0.002 ± 0.002 | 0.018 ± 0.018 | 0.000 ± 0.000 | 0.008 ± 0.008 | 2 |
| PPO T=80 | 0.66 | 0.637 ± 0.023 | 0.816 ± 0.039 | 0.514 ± 0.029 | 0.758 ± 0.020 | 2 |
| GRPO T=2560 | 0.66 | 0.000 ± 0.000 | 0.012 ± 0.008 | 0.000 ± 0.000 | 0.008 ± 0.000 | 2 |
| PPO T=2560 | 0.66 | 0.113 ± 0.047 | 0.404 ± 0.104 | 0.102 ± 0.059 | 0.332 ± 0.098 | 2 |

## Q6 — Longer budget and sequential eval

Both VLAs, 2.95 M env steps. T = 80, T = 320 + EER, T = 2560 + EER, and
T = 2560 + CL + EER.

**OpenVLA**

| Group | Env steps (M) | ID success | ID grasp | OOD success | OOD grasp | Seeds |
|---|---:|---:|---:|---:|---:|---:|
| T=80 | 2.95 | 0.771 ± 0.049 | 0.904 ± 0.010 | 0.646 ± 0.064 | 0.855 ± 0.016 | 2 |
| T=320 w/ EER | 2.95 | 0.777 ± 0.035 | 0.855 ± 0.012 | 0.650 ± 0.033 | 0.814 ± 0.057 | 2 |
| T=2560 w/ EER | 2.95 | 0.004 ± 0.000 | 0.000 ± 0.000 | 0.002 ± 0.002 | 0.000 ± 0.000 | 2 |
| T=2560 w/ CL, EER | 2.95 | 0.312 ± 0.070 | 0.494 ± 0.076 | 0.172 ± 0.035 | 0.461 ± 0.133 | 2 |

**SpatialVLA**

| Group | Env steps (M) | ID success | ID grasp | OOD success | OOD grasp | Seeds |
|---|---:|---:|---:|---:|---:|---:|
| T=80 | 2.95 | 0.832 ± 0.000 | 0.953 ± 0.000 | 0.699 ± 0.000 | 0.855 ± 0.000 | 1 |
| T=320 w/ EER | 2.95 | 0.771 ± 0.061 | 0.902 ± 0.016 | 0.689 ± 0.084 | 0.832 ± 0.012 | 2 |
| T=2560 w/ EER | 2.95 | 0.039 ± 0.020 | 0.299 ± 0.104 | 0.018 ± 0.014 | 0.277 ± 0.113 | 2 |
| T=2560 w/ CL, EER | 2.95 | 0.402 ± 0.262 | 0.801 ± 0.086 | 0.299 ± 0.244 | 0.736 ± 0.084 | 2 |

### Sequential eval of the Q6 checkpoints

Standalone sequential eval: each round resets once and then runs all 4 tasks of
the scene back to back. **seen** rounds use the 4 training rotations, **unseen**
rounds the other orderings. `success` scores each task on its own;
`success_chained` is the AND along the round (a task counts only if every earlier
task of the round also succeeded). Pooled over all tasks and scenes.

**OpenVLA — `success`**

| Group | Rounds | Domain | Pos 1 | Pos 2 | Pos 3 | Pos 4 |
|---|---|---|---:|---:|---:|---:|
| T=80 | seen | ID | 0.781 | 0.562 | 0.367 | 0.281 |
| T=80 | seen | OOD | 0.633 | 0.492 | 0.258 | 0.195 |
| T=80 | unseen | ID | 0.784 | 0.503 | 0.381 | 0.284 |
| T=80 | unseen | OOD | 0.662 | 0.384 | 0.275 | 0.261 |
| T=320 w/ EER | seen | ID | 0.656 | 0.594 | 0.430 | 0.328 |
| T=320 w/ EER | seen | OOD | 0.594 | 0.383 | 0.266 | 0.195 |
| T=320 w/ EER | unseen | ID | 0.758 | 0.536 | 0.392 | 0.322 |
| T=320 w/ EER | unseen | OOD | 0.647 | 0.386 | 0.236 | 0.167 |
| T=2560 w/ EER | seen | ID | 0.016 | 0.000 | 0.000 | 0.000 |
| T=2560 w/ EER | seen | OOD | 0.000 | 0.000 | 0.000 | 0.000 |
| T=2560 w/ EER | unseen | ID | 0.003 | 0.000 | 0.002 | 0.002 |
| T=2560 w/ EER | unseen | OOD | 0.002 | 0.002 | 0.002 | 0.003 |
| T=2560 w/ CL, EER | seen | ID | 0.305 | 0.172 | 0.117 | 0.055 |
| T=2560 w/ CL, EER | seen | OOD | 0.141 | 0.055 | 0.016 | 0.031 |
| T=2560 w/ CL, EER | unseen | ID | 0.269 | 0.127 | 0.069 | 0.041 |
| T=2560 w/ CL, EER | unseen | OOD | 0.150 | 0.047 | 0.039 | 0.019 |

**OpenVLA — `success_chained`**

| Group | Rounds | Domain | Pos 1 | Pos 2 | Pos 3 | Pos 4 |
|---|---|---|---:|---:|---:|---:|
| T=80 | seen | ID | 0.781 | 0.445 | 0.180 | 0.055 |
| T=80 | seen | OOD | 0.633 | 0.375 | 0.148 | 0.062 |
| T=80 | unseen | ID | 0.784 | 0.409 | 0.141 | 0.042 |
| T=80 | unseen | OOD | 0.662 | 0.281 | 0.094 | 0.042 |
| T=320 w/ EER | seen | ID | 0.656 | 0.469 | 0.266 | 0.102 |
| T=320 w/ EER | seen | OOD | 0.594 | 0.320 | 0.156 | 0.070 |
| T=320 w/ EER | unseen | ID | 0.758 | 0.433 | 0.202 | 0.083 |
| T=320 w/ EER | unseen | OOD | 0.647 | 0.289 | 0.105 | 0.033 |
| T=2560 w/ EER | seen | ID | 0.016 | 0.000 | 0.000 | 0.000 |
| T=2560 w/ EER | seen | OOD | 0.000 | 0.000 | 0.000 | 0.000 |
| T=2560 w/ EER | unseen | ID | 0.003 | 0.000 | 0.000 | 0.000 |
| T=2560 w/ EER | unseen | OOD | 0.002 | 0.000 | 0.000 | 0.000 |
| T=2560 w/ CL, EER | seen | ID | 0.305 | 0.062 | 0.016 | 0.000 |
| T=2560 w/ CL, EER | seen | OOD | 0.141 | 0.031 | 0.000 | 0.000 |
| T=2560 w/ CL, EER | unseen | ID | 0.269 | 0.056 | 0.009 | 0.002 |
| T=2560 w/ CL, EER | unseen | OOD | 0.150 | 0.011 | 0.003 | 0.000 |

**SpatialVLA — `success`**

| Group | Rounds | Domain | Pos 1 | Pos 2 | Pos 3 | Pos 4 |
|---|---|---|---:|---:|---:|---:|
| T=80 | seen | ID | 0.812 | 0.781 | 0.500 | 0.484 |
| T=80 | seen | OOD | 0.812 | 0.578 | 0.406 | 0.297 |
| T=80 | unseen | ID | 0.875 | 0.644 | 0.487 | 0.422 |
| T=80 | unseen | OOD | 0.756 | 0.506 | 0.372 | 0.284 |
| T=320 w/ EER | seen | ID | 0.773 | 0.570 | 0.406 | 0.336 |
| T=320 w/ EER | seen | OOD | 0.680 | 0.477 | 0.367 | 0.266 |
| T=320 w/ EER | unseen | ID | 0.739 | 0.403 | 0.295 | 0.283 |
| T=320 w/ EER | unseen | OOD | 0.694 | 0.427 | 0.309 | 0.231 |
| T=2560 w/ EER | seen | ID | 0.039 | 0.008 | 0.000 | 0.000 |
| T=2560 w/ EER | seen | OOD | 0.047 | 0.008 | 0.000 | 0.000 |
| T=2560 w/ EER | unseen | ID | 0.037 | 0.009 | 0.005 | 0.003 |
| T=2560 w/ EER | unseen | OOD | 0.025 | 0.000 | 0.002 | 0.003 |
| T=2560 w/ CL, EER | seen | ID | 0.328 | 0.281 | 0.266 | 0.250 |
| T=2560 w/ CL, EER | seen | OOD | 0.391 | 0.211 | 0.172 | 0.109 |
| T=2560 w/ CL, EER | unseen | ID | 0.367 | 0.252 | 0.227 | 0.225 |
| T=2560 w/ CL, EER | unseen | OOD | 0.370 | 0.195 | 0.180 | 0.156 |

**SpatialVLA — `success_chained`**

| Group | Rounds | Domain | Pos 1 | Pos 2 | Pos 3 | Pos 4 |
|---|---|---|---:|---:|---:|---:|
| T=80 | seen | ID | 0.812 | 0.641 | 0.406 | 0.234 |
| T=80 | seen | OOD | 0.812 | 0.500 | 0.281 | 0.031 |
| T=80 | unseen | ID | 0.875 | 0.562 | 0.247 | 0.125 |
| T=80 | unseen | OOD | 0.756 | 0.400 | 0.122 | 0.047 |
| T=320 w/ EER | seen | ID | 0.773 | 0.492 | 0.305 | 0.211 |
| T=320 w/ EER | seen | OOD | 0.680 | 0.398 | 0.234 | 0.148 |
| T=320 w/ EER | unseen | ID | 0.739 | 0.347 | 0.161 | 0.108 |
| T=320 w/ EER | unseen | OOD | 0.694 | 0.353 | 0.162 | 0.080 |
| T=2560 w/ EER | seen | ID | 0.039 | 0.000 | 0.000 | 0.000 |
| T=2560 w/ EER | seen | OOD | 0.047 | 0.000 | 0.000 | 0.000 |
| T=2560 w/ EER | unseen | ID | 0.037 | 0.000 | 0.000 | 0.000 |
| T=2560 w/ EER | unseen | OOD | 0.025 | 0.000 | 0.000 | 0.000 |
| T=2560 w/ CL, EER | seen | ID | 0.328 | 0.164 | 0.125 | 0.062 |
| T=2560 w/ CL, EER | seen | OOD | 0.391 | 0.172 | 0.062 | 0.031 |
| T=2560 w/ CL, EER | unseen | ID | 0.367 | 0.166 | 0.075 | 0.036 |
| T=2560 w/ CL, EER | unseen | OOD | 0.370 | 0.150 | 0.069 | 0.030 |

Per-task success falls with position even though the tasks are the same — later
tasks start from a scene the policy has already changed. Seen and unseen orderings agree
at position 1 and differ by at most ~0.17 later, with no consistent advantage for the
training order.
