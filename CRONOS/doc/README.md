# CRONOS documentation

All CRONOS documentation lives in this directory. The top-level
[`README.md`](../../README.md) says *how to run things*; the documents here say
*why a thing works the way it does*, what changed between versions, and what
their numbers do and do not mean.

Current version: **V0.99** (early release) — see [`../version.py`](../version.py).

## Changelog

| Document | Contents |
|---|---|
| [`CHANGELOG.md`](CHANGELOG.md) | Every version from the initial refactor to V0.99, newest first, reconstructed from the git history. Entries marked **[numbers-affected]** changed results collected by earlier versions; **Report:** entries point at the report that explains the defect. |

## Design references

Describe the current tree. A change that makes one of them wrong updates it in
the same commit.

| Document | Read it when |
|---|---|
| [`reset_modes.md`](reset_modes.md) | You are choosing `train.sh`'s reset argument, comparing two reset modes, or wondering why a `-noep-` run directory does not mean what its name suggests. Gives the four orthogonal flags, what each mode preset expands to, the `RUN_TAG` rename and what it invalidates, and the start-state drift that makes bare `noep` metrics optimistic. |
| [`eval_sequential.md`](eval_sequential.md) | You are running standalone eval (`eval_only.py`, `scripts/eval.sh`) or reading its outputs. Gives the `eval:` config block and CLI precedence, what a round is, the RNG contract that makes round N identical across runs, how many trials each scene gets, and where padding does and does not happen. |
| [`data_schemas.md`](data_schemas.md) | You are reading a CSV out of a run's `glob/` and need to know what a column means — especially `direction`, `phase`, and the three value columns, all of which change meaning with the reset mode or the algorithm. |
| [`rng_and_io_notes.md`](rng_and_io_notes.md) | You are changing how training (or eval) seeds, draws layouts, writes CSVs or resumes. Lists every RNG coupling and file-writing failure found while making eval reproducible, how each was fixed in eval, and a checklist of what training still lacks. |

## Reports (bug / failure analyses)

Point-in-time investigations. Each names the version it was written against,
what it found, and which previously collected numbers it invalidates. They are
kept as a record; the fixes they led to are listed in the changelog.

| Report | Kind | Summary |
|---|---|---|
| [`reports/eval_audit.md`](reports/eval_audit.md) | Bug report (fixed, V0.91) | Sequential eval never reset ManiSkill's `_elapsed_steps`, so every task after the first reported `truncated` on every step: aggregate `success` became a time-average and grasp flags leaked between tasks. Fixed by `CronosWrapper.begin_segment()`. Sequential-eval numbers from before the fix are not comparable. |
| [`reports/grpo_autorl.md`](reports/grpo_autorl.md) | Implementation review (V0.93) | Read-only review of AutoRL's GRPO path, where CRONOS matches it bit-for-bit, and the grouping / `std` options. §4.1, §5 and §6 are corrected by `grpo_failure.md` §5. |
| [`reports/grpo_failure.md`](reports/grpo_failure.md) | Failure report (open, V0.95) | Why `--alg-name grpo` collapses: normalization is per step with no group baseline, and `alg_grpo_fix` never penalizes inaction, so "stop interacting" is the stable optimum. Shared bit-for-bit with AutoRL, whose GRPO path was never run. The recommended fix (§6) is not yet implemented — **treat GRPO as experimental in V0.99**. |

## What is not in the release

Plotting and statistics tools, per-experiment plot configs, figure renderers and
their outputs live in `CRONOS/analysis/`, which is git-ignored: they are kept in
local checkouts and are not part of the release. Documents that mention
`analysis/<tool>.py` (for example `analysis/mcnemar_pair.py`,
`analysis/parse_autorl_eval.py`, `analysis/plot_segment_positions.py`) refer to
those local tools.

Everything needed to train, evaluate, merge eval shards and check checkpoints is
in the release: `main.py`, `eval_only.py`, `envs/`, `training/`, `evaluation/`,
`configs/`, `scripts/`, `tests/`, and `tools/` (`rebuild_eval_outputs.py`,
`check_ckpt_compat.py`, `bench_rollout.py`, and `plot_run_trends.py`, the live
dashboard `main.py` refreshes after each eval).

## Conventions these documents follow

**Claims are marked with how they were checked.** "Verified", "measured" and
"reproduced" mean a script was run and its output pasted in; each such section
says how to re-run it. Anything established only by reading code says so, and
end-to-end confirmation on hardware is called out separately where it matters.

**Behaviour changes name what they invalidate.** When a fix changes numbers, the
document says which previously-collected data stays comparable and which does
not, rather than leaving it to be discovered later.

**Deliberate deviations from the papers or from AutoRL are recorded as such**, so
a difference is never mistaken for a bug — and so the reverse is also true.

Paths such as `$AUTORL_ROOT` and `$RL4VLA_ROOT` stand for a local checkout of the
corresponding upstream repository.
