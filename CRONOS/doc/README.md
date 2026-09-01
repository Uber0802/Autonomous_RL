# CRONOS design documents

Reference material that is too long or too load-bearing for the top-level
[`README.md`](../../README.md). The README says *how to run things*; these say
*why a thing works the way it does*, and what its numbers do and do not mean.

| Document | Read it when |
|---|---|
| [`data_schemas.md`](data_schemas.md) | You are reading a CSV out of a run's `glob/` and need to know what a column means — especially `direction`, `phase`, and the three value columns, all of which change meaning with the reset mode or the algorithm. |
| [`eval_audit.md`](eval_audit.md) | You are comparing eval numbers across CRONOS versions or against AutoRL. Documents an accounting defect in sequential eval, its fix, and which historical numbers stayed comparable. |
| [`grpo_autorl.md`](grpo_autorl.md) | You are running or interpreting `--alg-name grpo`. Reviews AutoRL's implementation, records where CRONOS matches it bit-for-bit and where it deliberately does not, and gives the grouping / `std` trade-off with measured numbers. |
| [`reset_modes.md`](reset_modes.md) | You are choosing `train.sh`'s reset argument, comparing two reset modes, or wondering why a `-noep-` run directory does not mean what its name suggests. Gives the four orthogonal flags, what each mode preset expands to, the `RUN_TAG` rename and what it invalidates, and the start-state drift that makes bare `noep` metrics optimistic. |

## Conventions these documents follow

**Claims are marked with how they were checked.** "Verified", "measured" and
"reproduced" mean a script was run and its output pasted in; each such section
says how to re-run it. Anything established only by reading code says so — most
of this work was done on a machine with no GPU large enough to load a VLA, so
end-to-end confirmation on hardware is called out separately where it matters.

**Behaviour changes name what they invalidate.** When a fix changes numbers, the
document says which previously-collected data stays comparable and which does
not, rather than leaving it to be discovered later.

**Deliberate deviations from the papers or from AutoRL are recorded as such**, so
a difference is never mistaken for a bug — and so the reverse is also true.

## Version

The code version stamped into every run's `run_config.json` lives in
[`../version.py`](../version.py). These documents are not separately versioned:
they describe the current tree, and a change that makes one of them wrong is
supposed to update it in the same commit.
