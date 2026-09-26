# CRONOS documentation

Internal documentation (not released). The top-level
[`README.md`](../../README.md) covers only *how to use the codebase*; the
documents here record what changed, why things work the way they do, and what
the experiments found.

Current version: **V0.99** (early release) — see [`../version.py`](../version.py).

## Changelog

| Document | Contents |
|---|---|
| [`CHANGELOG.md`](CHANGELOG.md) | Every version from the initial refactor to V0.99, newest first. **[numbers-affected]** marks changes that alter results of earlier versions. |

## Results

| Document | Contents |
|---|---|
| [`results/paper_experiments.md`](results/paper_experiments.md) | Final numbers of the paper experiments Q1–Q6 (horizon, EER, reset strategy, curriculum, GRPO vs PPO, long-budget comparison and sequential eval), with a one-table summary of findings. |
| [`results/grpo.md`](results/grpo.md) | GRPO: review of AutoRL's implementation (V0.93) and the analysis of why it collapses (V0.95). **GRPO is experimental in V0.99.** |
| [`results/bug_reports.md`](results/bug_reports.md) | Every defect whose fix changed earlier numbers, with what each invalidates, and the full sequential-eval accounting audit. Read before comparing runs across versions. |

## Design references

Describe the current tree; a change that makes one of them wrong updates it in the
same commit.

| Document | Read it when |
|---|---|
| [`reset_modes.md`](reset_modes.md) | You are choosing `train.sh`'s reset argument or comparing two reset modes: the four orthogonal flags, what each mode expands to, `RUN_TAG`s, and the start-state drift that makes bare `noep` metrics optimistic. |
| [`eval_sequential.md`](eval_sequential.md) | You are running standalone eval or reading its outputs: the `eval:` block, rounds and pose sets, the RNG contract, coverage, resume and shards. |
| [`data_schemas.md`](data_schemas.md) | You are reading a CSV out of a run's `glob/` and need to know what a column means. |
| [`rng_and_io_notes.md`](rng_and_io_notes.md) | You are changing how training or eval seeds, draws layouts, writes CSVs or resumes; includes the checklist of what training still lacks. |
| [`environments.md`](environments.md) | You want to know why there are four conda envs, what is bit-exact to what, and how checkpoints move between LM stacks. |

## Repository layout: what is and is not released

| Path | In V0.99 | Contents |
|---|---|---|
| `main.py`, `eval_only.py`, `envs/`, `training/`, `evaluation/`, `run_paths.py`, `version.py` | yes | training and evaluation |
| `configs/` | yes | sample experiment configs |
| `scripts/` | yes | `train.sh`, `eval.sh`, plotting requirements |
| `tools/` | yes | eval-shard merging, checkpoint compatibility check, rollout benchmark, live dashboard |
| `tests/` | yes | CPU tests (`python -m pytest tests/ -q`) |
| `doc/` | never (ignored) | this directory — internal documentation; the release documents itself through the top-level README |
| `plotting/` | **not yet** (ignored) | plotting and statistics tools and their README; planned for a later release. Documents that mention `plotting/<tool>.py` refer to these. |
| `history/` | never (ignored) | debug output, backups, launch queues and per-experiment plot configs with machine-specific paths |

## Conventions these documents follow

**Claims are marked with how they were checked.** "Verified", "measured" and
"reproduced" mean a script was run and its output pasted in; anything established
only by reading code says so, and end-to-end confirmation on hardware is called
out separately where it matters.

**Behaviour changes name what they invalidate.** When a fix changes numbers, the
document says which previously collected data stays comparable and which does not.

**Deliberate deviations from the papers or from AutoRL are recorded as such**, so a
difference is never mistaken for a bug — and the reverse.

Paths such as `$AUTORL_ROOT` and `$RL4VLA_ROOT` stand for a local checkout of the
corresponding upstream repository.
