# CRONOS changelog

Newest first. Index of the documentation: [`README.md`](README.md).

Reconstructed from the git history at the V0.99 release: the commits carry only a
version label, so every entry below was written from the diff. Conventions:

- **[numbers-affected]** — the change alters results produced by earlier versions.
  Check these before comparing runs across versions.
- **Report:** — a bug report, failure report or design document was added; see
  [`README.md`](README.md) for the full list.
- *Analysis tools* — plotting / statistics tools that were moved out of the release
  at V0.99 (see [the repository layout](README.md#repository-layout-what-is-and-is-not-released)).
  Their history is kept here but condensed.

**Version stamp caveat.** `version.py` is what `run_config.json` records. It was a
hard-coded `"V0.3"` until V0.93c, and then stayed `"V0.93"` through V0.95. Runs
stamped `V0.3` may come from any version up to V0.93b; runs stamped `V0.93` may come
from V0.93c–V0.95. V0.99 is the first stamp that matches the code.

---

## V0.99 — 2026-09-26 (early release)

**Changed**
- Version set to `V0.99` in `version.py` (code and config format) and in the
  `cronos_version` annotation of every sample config.
- Documentation consolidated under `CRONOS/doc/`: this changelog; design references
  plus a new `environments.md`; experiment results in `doc/results/`
  (`paper_experiments.md` — Q1–Q6 result tables; `grpo.md` — merged GRPO review and
  failure analysis; `bug_reports.md` — every numbers-affecting defect plus the
  sequential-eval audit). The top-level README now covers usage only.
- Plotting / statistics tools (`plot_*.py`, `mcnemar_pair.py`,
  `parse_autorl_eval.py`, `hsr_need_common.py`, `render_*.py`,
  `extract_run_frames.py`) and their example configs moved from `tools/` to
  `CRONOS/plotting/`, ignored until they are released. Debug output, backups, launch
  queues and per-experiment plot configs moved to `CRONOS/history/` (never released).
  `tools/` keeps `rebuild_eval_outputs.py`, `check_ckpt_compat.py`,
  `bench_rollout.py` and `plot_run_trends.py`.
- `tools/plot_run_trends.py` no longer imports `plot_common.py`.
- `tests/test_eval_plan.py`: the McNemar test is skipped when `plotting/` is absent.

**Removed**
- Machine- and user-specific paths, the hard-coded wandb entity default in
  `plot_run_trends.py` (now `$WANDB_ENTITY`, else the wandb default entity), and
  internal repository names in comments.

**Known issues**
- GRPO collapses — see [`results/grpo.md`](results/grpo.md). Use PPO.
- Training RNG is not yet isolated — see [`rng_and_io_notes.md`](rng_and_io_notes.md).

---

## V0.95 — 2026-09-26 (`255d86b`)

*Core:* no changes.

**Report:** [`results/grpo.md`](results/grpo.md) — why
`--alg-name grpo` collapses. Normalization is per step with no group baseline;
`alg_grpo_fix=True` leaves all-zero ("idle") trajectories at advantage 0, and
mean-centering cancels the +0.1 grasp signal, so the objective ranks
success > inaction > attempt-and-fail. Grasp falls in all 7 GRPO runs and rises in
7/8 PPO controls; 98.8 % of samples carry zero advantage. The math is bit-identical
to AutoRL, whose GRPO path was never run. Corrects three claims in
`grpo_autorl.md`. The proposed trajectory-level fix is not implemented; **no
GRPO-vs-PPO numbers should be cited**.

*Analysis tools:* HSR-need tools (`plot_hsr_envs_curve.py`,
`plot_hsr_need_panels.py`, `hsr_need_common.py`); GPU figure renderers
(`render_episode_boundaries.py`, `render_segment_frames.py`,
`render_asset_catalog.py`, `render_background_catalog.py`) and
`extract_run_frames.py`, documented in `figures/README.md`; paper-style rollout
figures; `plot_eval_success.py` config keys `horizon_lines`, `legend`, `axis`,
`sr_ylim`, `aspect_ratio`. **[numbers-affected]** `plot_eval_success.py` crops no
longer drop rows, so aggregate final mean / std can differ from V0.94p.
Experiment plot configs (`scripts/Q*.json`) and tool backups were committed here;
moved to `history/` in V0.99.

## V0.94l – V0.94p — 2026-09-20 … 2026-09-21

*Analysis tools only:* figure styling — fitted y-axis, curriculum-switch rule,
one global legend size (`LEGEND_PT`), 4:3 / 16:9 box shapes, legends outside the
box. No core changes.

## V0.94i – V0.94k — 2026-09-19 … 2026-09-20

*Analysis tools only:* `plot_sequence_eval.py` splits training ("seen") and random
("unseen") rounds into separate figures, per domain, and adds a success-by-task
figure family (`--per-scene`).

## V0.94h — 2026-09-17 (`0d84e4f`)

*Analysis tools:* **[numbers-affected]** eval-curve gap detection uses each resume
leg's own eval cadence, so T320 → T2560 chains are no longer read as full of holes.

## V0.94f — 2026-09-17 (`3363cec`)

**Report:** `eval_sequential.md` gains what `--seed` does and does not control in
eval; `rng_and_io_notes.md` corrects the object-scale note (unseeded, but every
scale list is `[1.0]`, so no effect).

*Analysis tools:* **[numbers-affected]** `plot_eval_success.py` leaves a run's missing
evals as NaN instead of interpolating, and computes mean / std / `n_runs` per x
point over the runs present. **[numbers-affected]** `plot_segment_positions.py` no
longer synthesizes start poses because of EER alone.

## V0.94e — 2026-09-17 (`54c6231`)

*Analysis tools:* **[numbers-affected]** `plot_sequence_eval.py` averages a trial
held by more than one shard instead of "last wins"; `--metric` takes several
metrics.

## V0.94d — 2026-09-17 (`c6ff916`)

**Added**
- `evaluation/provenance.py::resolve_policy_args`: standalone eval takes `policy`,
  `vla_path`, `vla_unnorm_key`, `vla_temperature_eval` and `vla_lora_rank` from the
  checkpoint's training `run_config` (CLI > run_config > config YAML > per-policy
  defaults matching `train.sh`). Resolved values and sources go into
  `eval_plan.json` and the fingerprint. `main.py --eval-single/--eval-sequential`
  uses the same resolution.

**Changed**
- **[numbers-affected]** SpatialVLA checkpoints are evaluated by default with
  `bridge_orig/1.0.0` and greedy decoding (temperature 0.0) instead of 0.6.
- `scripts/eval.sh` no longer hard-codes the OpenVLA path and unnorm key.

## V0.94b — 2026-09-16 (`f007a48`)

**Fixed**
- Checkpoints load across LM stacks: `tf447` (peft 0.14) writes `LoraConfig` fields
  that `tf440` (peft 0.11.1) rejects. `SimplerEnv/.../policies/peft_compat.py`
  drops a field only when it holds its "off" value and raises otherwise; both VLA
  policies load through it (`tests/test_peft_compat.py`).

**Added**
- `tools/check_ckpt_compat.py`: read-only preflight check (`--target tf440|tf447`).

## V0.94a – V0.94c — 2026-09-16

*Analysis tools only:* new `plot_sequence_eval.py` (success by task position);
`plot_segment_positions.py` density modes, `--per-task`, multiple `--step-range`,
`--color-by item|scene`.

## V0.94 — 2026-09-15 (`36b7a8a`)

**Added**
- `evaluation/` package (`plan`, `records`, `outputs`, `provenance`, `sequential`),
  shared by `eval_only.py` and `main.py --eval-single/--eval-sequential`.
- Standalone eval covers every scene of the checkpoint's **training config**, split
  into 4 order blocks. A pose set is 6 rounds: round 6k is the training rotations,
  6k+1…6k+5 the untrained cycles. Rounds are selected by global number.
- Named RNG streams (`envs/rng_streams.py`) for layouts (`layout_ids` reset option),
  orders and per-unit policy reseeding: any subset of rounds reproduces those rounds
  of a full run.
- `eval:` block in the training YAML with matching CLI flags (CLI wins).
- `--eval-resume`, round shards, and `tools/rebuild_eval_outputs.py` to merge them.
- Outputs `eval_plan.json`, `eval_status.json`, `eval_layouts.csv`,
  `eval_segment_pose.csv`, `eval_sequence_summary.csv`, `eval_coverage.csv`.
- Training snapshots `experiment_config.yaml` into `glob/` and every checkpoint;
  eval checks its scenes against it (`--allow-config-mismatch` to override).
- CPU tests `tests/test_eval_plan.py`, `tests/test_sequential_eval.py`.

**Changed**
- **[numbers-affected]** Standalone eval is not comparable with pre-V0.94 standalone
  eval: orders, layouts, per-scene env ranges and the sampling seed all changed.
  `eval_kind` is now `<domain>_<seq_kind>`; `eval_success.csv` is written only for
  a complete eval covering the full design.
- `eval_per_trial.csv` gains `eval_kind`, `seq_kind`, `group`, `order`, `pose_set`;
  `seq_idx` is the global round.
- `scripts/eval.sh`: config defaults to the checkpoint's training config (`-`),
  extra flags are passed through. Eval refuses to pad envs.

**Fixed**
- Multi-group standalone eval broadcast group 0's objects to every env and crashed on
  other groups' tasks.
- Eval aggregate lost when the coverage check failed; header-only
  `eval_success.csv` after a crash; duplicate aggregate rows on relaunch.

**Removed**
- `--eval-sequences`, `--eval-training-sequence`; YAML keys `num_sequences`,
  `include_training_sequence`, `training_orders`, `random_orders`, `layout_rng`;
  `CronosRunner.eval`. `configs/eval/*` are legacy.

**Report:** [`eval_sequential.md`](eval_sequential.md) (standalone-eval design) and
[`rng_and_io_notes.md`](rng_and_io_notes.md) (every RNG coupling and file-writing
failure found, with a checklist of what training still lacks).

---

## V0.93k — 2026-09-09 (`bcd7720`)

*Analysis tools only:* `--config` mode writes one figure per group, split at the
group's resets (T1280+); a run with no data is skipped with a warning instead of
failing the whole figure set.

## V0.93h – V0.93j — 2026-09-03

*Analysis tools only:* diagnostics instead of crashes / empty figures when nothing
aggregates; `plot_segment_positions.py --step-range` and start-pose reconstruction
for runs recorded before the `phase` split; titles removed from figures.

## V0.93g — 2026-09-02 (`4b8c508`)

**Changed**
- **Breaking — reset-mode meaning.** `noep` now means HSR + `--reset-mode none`
  **without** LSR; the old meaning is the new `noep+LSR`. `HSR+LSR` is the canonical
  spelling (`LSR+HSR` still accepted). `RUN_TAG` fragments renamed
  (`LSR+HSR`→`HSRLSR`, `noep`→`HSRnoep`, `noep+LSR`→`HSRLSRnoep`), so no new run
  collides with an old one. Old `-noep-` directories are `noep+LSR` runs.
- `main.py` warns at startup for bare `noep`: start states drift toward
  already-satisfied tasks, which makes rollout success optimistic.

**Report:** [`reset_modes.md`](reset_modes.md) — the four orthogonal flags, what each
mode expands to, what the rename invalidates.

## V0.93f — 2026-09-01 (`a9cb1ce`)

**Report:** `doc/README.md` created as the documentation index.

*Analysis tools:* `scripts/plot.py` → `tools/plot_eval_success.py`; all plot tools
share one config format and look.

## V0.93c – V0.93e — 2026-08-29

**Added**
- `version.py`, stamped into `run_config.json` (was a literal `"V0.3"`).
- `envs/suite.py` defers its ManiSkill import so the pose tables load without a GPU
  stack.

*Analysis tools:* `--config` mode for comparing several runs (resume chains stitched,
child wins on overlap), shared `plot_common.py`, synthetic start poses for pre-`phase`
runs, tool options settable from the config.

## V0.93b — 2026-08-26 (`143261f`)

**Added**
- `segment_pose.csv` records both sides of every boundary: `phase=end` before
  HSR/EER, `phase=start` after them (and after `env.reset()`).
  `--segment-pose-phase start|end|both` (default `both`).

*Analysis tools:* first versions of `plot_rollout_success.py` and
`plot_segment_positions.py`.

## V0.93a — 2026-08-19 (`079e797`)

**Added**
- Perturbation through the LSR reset goal: `--backward-goal table|recep|mixed`,
  `--backward-recep-prob`; `train.sh` 9th argument `perturb`. Uses its own RNG, so
  `table` (the default) is byte-identical to earlier versions.
- `rollout_success.csv` `direction` gains `backward_recep`.

**Changed**
- **[numbers-affected]** wandb per-task rollout scalars: reset segments now go to
  `rollout_reset/<task>/…`, so `rollout/<task>/*` of LSR runs no longer includes the
  backward zeros.

## V0.93 — 2026-08-16 (`e65d33f`)

**Added**
- GRPO: `--alg-name ppo|grpo`, `--alg-grpo-fix`, `--grpo-group-scope
  batch|scene|task`, `--grpo-std-scope group|global|none`; `training/grpo.py`;
  `CronosReplayBuffer.compute_grpo_returns()` (`batch` bit-identical to AutoRL);
  `train.sh` 8th argument `algo`. *Later found to collapse — see V0.95.*
- `run_paths.py`: the run directory is created as an absolute path before
  `wandb.init`, and the run fails if wandb falls back to `$TMPDIR`. `RUN_OUT_DIR`
  override in `train.sh` / `eval.sh`.

**Fixed**
- A relative, not-yet-existing `--wandb-dir` (what `train.sh` passed) could silently
  put every CSV, checkpoint and video under `/tmp`.

**Report:** [`results/grpo.md`](results/grpo.md) — review of AutoRL's
GRPO and CRONOS's grouping / std choices.

---

## V0.92 — 2026-08-10 (`e45cdb8`)

**Added**
- `train.sh` 7th argument `eer` = `on|off` (End-Effector Reset); `off` adds
  `--no-reset-robot` and a `-noEER` tag.

**Changed**
- `--record-segment-pose` is on by default.

**Fixed**
- **[numbers-affected]** With `--no-reset-robot`, training never reset
  `_elapsed_steps`: every later step reported truncated, masks went to 0 and GAE
  degenerated to `returns = reward`. The train loop now calls `begin_segment()`.

## V0.91a — 2026-08-10 (`594b054`)

**Fixed**
- An eval-only run left an empty `rollout_success.csv`.
- `mcnemar_pair.py` crashed on empty cells (e.g. a recovered AutoRL baseline).

## V0.91 — 2026-08-10 (`7ecc981`)

**Added**
- `rollout_success.csv` — one row per (episode, segment, env): success / grasp at
  segment end (same definition as eval), `direction`, `reward_sum`,
  `return_discounted`, `return_gae`, `value_mean`, `advantage_mean`.
- `segment_pose.csv` (`--record-segment-pose`), replacing `end_of_segment_xyz.csv`.
- `eval_per_trial.csv` gains `success_chained` next to the independent `success`.
- `tools/bench_rollout.py`; `tools/parse_autorl_eval.py` (analysis tool since V0.99).
- `envs/scheduler.py::build_eval_sequences`, shared by both eval entry points.

**Changed**
- Conda envs renamed `cronos_<tf447|tf440>_<cu121|cu128>`.

**Fixed**
- **[numbers-affected]** Sequential eval: every task after the first ran permanently
  truncated, so `success` became a time-average and grasp flags carried over.
  `CronosWrapper.begin_segment()` now reopens the measurement window.
  **Sequential-eval numbers from before this fix are not comparable.**
- `end_of_segment_xyz.csv` labelled every row one episode ahead of its video.

**Report:** [`results/bug_reports.md`](results/bug_reports.md) — the sequential-eval
accounting defect, its fix, and which historical numbers stay comparable.
[`data_schemas.md`](data_schemas.md) — CSV column specs.

---

## V0.9a — 2026-07-09 (`b8ab6a8`)

**Changed**
- Conda envs renamed `cronos_env`, `cronos_env_blackwell`, `cronos_env_lite`,
  `cronos_env_lite_blackwell`; install logic unchanged.

## V0.9 — 2026-07-09 (`b308bf4`)

**Changed**
- Comment / docstring cleanup across `envs/`, `main.py`, `eval_only.py`,
  `training/` and the SimplerEnv adapters (internal milestone tags removed); no
  functional change.

**Fixed**
- **[numbers-affected]** `scripts/plot.py`: step / reset crops applied at row level
  before aggregation.

---

## V0.4.2 – V0.4.2b — 2026-06-25

**Added**
- `setup.sh [policy] [gpu]`: `openvla_v01` lightweight OpenVLA-only stack
  (torch 2.2.0, transformers 4.40.1, peft 0.11.1) for 48 GB Ada GPUs, bit-exact to
  V0.1; `blackwell` variant on torch 2.7.0+cu128; post-install sanity check;
  protobuf / tensorflow-metadata pins.
- `plot_run_trends.py --x-axis episode|total_steps`.

**Changed**
- **[numbers-affected]** SpatialVLA preset in `train.sh` evaluates with
  `--vla-temperature-eval 0.0` (deterministic).

## V0.4.1 – V0.4.1a — 2026-06-17

**Added**
- SpatialVLA integration: `--policy openvla|spatialvla` (also YAML), 3-token action
  decoding in `CronosWrapper`, buffer sized by `act_token_len`; vendored
  `SpatialVLA/` and a SimplerEnv SpatialVLA adapter. `train.sh` 6th argument `vla`.
- `eval_only.py --action-chunk`, `--eval-ood/--no-eval-ood`, `eval_per_trial.csv`.
- `training/ppo.py` per-minibatch diagnostics (`approx_kl`, `clip_fraction`,
  explained variance, grad norms); per-task rollout metrics in wandb.
- Live dashboard `tools/plot_run_trends.py`, refreshed after each eval.
- Configs `spatialvla_2x2_smoke.yaml`, `spatialvla_2x2_train.yaml`,
  `eval/one_group_2x2_eval24.yaml`, `eval/one_group_carrot_1x1.yaml`.

**Changed**
- **[numbers-affected]** wandb training scalars are aggregated over all minibatches of
  an update (previously the last minibatch only).
- `setup.sh` moved to torch 2.7.0+cu128 / transformers 4.47 / peft 0.14.
- `RUN_TAG` carries the VLA (`CRONOS-<vla>-…`).

## V0.4c — 2026-06-14 (`7a5394d`)

**Changed**
- **[numbers-affected]** `train.sh` reset modes redefined: `LSR` = backward-policy
  learning (`--enable-backward --backward-interval 1`), no longer a robot reset;
  `HSR` = `--reset-unsuitable` with the gripper reset on. Earlier LSR / HSR runs used
  different semantics.

**Fixed**
- **[numbers-affected]** HSR `MAX_RESET` budget is `max_ep × segs_per_ep × num_envs`
  (was ~3× too small at T1280, stopping runs early).

## V0.4b — 2026-06-11 (`5df78a3`)

**Changed**
- `LSR+HSR` and `noep` add backward learning (superseded by V0.4c).

## V0.4a — 2026-06-02 (`a58ff91`)

**Added**
- `WorkspaceAABBDetector` (`unsuitable_detector` YAML block); detectors report
  per-env reasons.
- `--hsr-reset-scope per_env|per_actor|all` (default `per_env`).
- `configs/four_group_sequential_2x2.yaml` (4 groups × 16 envs).

**Changed**
- **[numbers-affected]** Default HSR scope is `per_env` (every actor of a flagged env
  respawns); the V0.3.1 behaviour is `per_actor`.
- Training videos record post-step frames only.

**Fixed**
- **[numbers-affected]** HSR-only no longer silently resets the robot (HSR and robot
  reset are now independent).

---

## V0.3a – V0.3b — 2026-05-07

**Added**
- AutoRL-style standalone eval (`--eval-mode sequential|single`, `--eval-sequences`);
  scene snapshot / restore around mid-training eval when `reset_mode=none`;
  per-training eval configs under `configs/eval/`; T2560 horizon.

**Changed**
- README moved to the repository root.

**Fixed**
- **[numbers-affected]** HSR respawn used hard-coded slots and un-rotated carrot
  quaternions; the first observation showed the robot at a stale pose (missing
  CPU→GPU sync); mid-training eval clobbered non-episodic training state.

**Removed**
- `--debug-rollout` and assorted debug prints.

## V0.3 – V0.3.2 — 2026-04-27 … 2026-05-04

**Added**
- Per-group `num_envs`, objects, receptacles and background in one run; mixed N/M.
- Sub-group fan-out (`fan_out: true`), matching AutoRL's multi-task gradient mixing.
- Per-env rotation eval (`--num-eval-episode`, `--eval-at-start`).
- `eval_only.py`, `scripts/train.sh`, `scripts/eval.sh`; config validation V22–V25;
  example configs.

**Changed**
- 3x3 pose preset uses a larger workspace (V0.3.2); eval videos cover every env.
- README object / receptacle tables corrected to the real model names (V0.3.1).

**Fixed**
- Single-group configs, config-before-env ordering and log indexing (V0.3e–g).

## V0.2-a – V0.2-f — 2026-04-10 … 2026-04-20

**Added**
- YAML experiment config (`envs/config.py`) with symbolic tasks and validation;
  scheduler with `sequential` / `pure_random` / `sequence_random`; scene registry;
  `--reset_mode none`; `--resume_from` with scheduler state.
- Single parametric `PickPlaceNxM-v1` env; pluggable unsuitable-env detectors.
- `SuccessRecorder` (`eval_success.csv`, `counters.json`), `run_config` snapshot,
  SIGINT cleanup of the memory-mapped buffer.

**Changed**
- `RewardShaper` uses the env's native `evaluate()` (matches AutoRL `get_reward()`).
- Episode init matches AutoRL `TwoObjectTwoReceptacle` for every shape.

**Fixed**
- mmap buffer cleanup on NFS (`ENOTEMPTY`).

## Pre-release — 2026-03-11 … 2026-04-09

- **Init** (`2b4256e`): modular CRONOS on RL4VLA / AutoRL — multi-object pick-and-place
  envs, `ResetStrategy`, `RewardShaper`, `TaskSuite`, `TaskScheduler`,
  `CronosWrapper`, non-episodic PPO in `main.py`, memory-mapped GAE buffer.
- **Refactoring / alignment checkpoints** (2026-03-16 … 03-23): seeding and init
  order aligned with AutoRL; `segment_len` / `episode_len` / `task_len` /
  `ppo_update_len`; `max_steps` / `max_reset` stops; resume from
  `training_state.pt`; `suite.py` pose presets reproducing AutoRL layouts.
- **Stable 0.0.1** (2026-04-02): `GenericNxMPickPlace`; PPO updates mid-rollout every
  `ppo_update_len` steps (AutoRL `training_interval`).
- **Hotfix "PPOLog"** (2026-04-09): `--log_file`, resume-split test runs.
