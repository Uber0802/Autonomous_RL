# CSV output schemas

Describes the current tree; the code version is in [`../version.py`](../version.py).
Index of these documents: [`README.md`](README.md).

Every file lands in the run's `glob/` directory, written with stdlib `csv`.
Training files are append-only and carry both x-axes (`total_steps` and
`total_resets`). Standalone eval has append-only *source* files and *derived*
files that are rebuilt whole (see `eval_sequential.md` §6).

| File | One row per | Written by | Enabled by |
|---|---|---|---|
| `rollout_success.csv` | (episode, segment, env) | training rollout | always |
| `segment_pose.csv` | (episode, segment, phase, env, actor) | training rollout | always (`--no-record-segment-pose` to disable) |
| `eval_success.csv` | (eval point, group, task) | training eval (append) + standalone eval (derived) | always; standalone: complete evals only |
| `eval_per_trial.csv` | (domain, round, task slot, env) | standalone eval (source) | always |
| `eval_layouts.csv` | (domain, round, env) | standalone eval (source) | always |
| `eval_segment_pose.csv` | (domain, round, task slot, phase, env, actor) | standalone eval (source) | `eval.record_pose` (default on) |
| `eval_sequence_summary.csv` | (level, domain, seq_kind, …) | standalone eval (derived) | always |
| `eval_coverage.csv` | (domain, seq_kind, scene) | standalone eval (derived) | always |
| `eval_status.json` | — | standalone eval (derived) | always |

"Standalone eval" is `eval_only.py` and `main.py --eval-single/--eval-sequential`;
its design, settings and RNG contract are in [`eval_sequential.md`](eval_sequential.md).

---

## `rollout_success.csv`

Per-env outcome and value signal for each completed 80-step segment — the
training-time counterpart of `eval_per_trial.csv`. Captured at the segment
boundary **before** the scheduler advances, so the task attribution is the pair
that was actually running.

| Column | Meaning |
|---|---|
| `episode` | 1-based, matches `train_videos/rollout_ep<N>_seg<M>/` |
| `segment` | 1-based within the episode |
| `total_steps` | cumulative env-steps at segment end (`num_envs` per tick) |
| `total_resets` | cumulative hard + soft resets at segment end |
| `env_idx` | parallel env index, `[0, num_envs)` |
| `group` | YAML group this env belongs to |
| `task` / `obj` / `recep` | the task this env ran during the segment |
| `direction` | `forward`, `backward` or `backward_recep` — see below |
| `success` | segment-terminal success, 0/1 |
| `consecutive_grasp` | latched within this segment, 0/1 |
| `is_src_obj_grasped` | latched within this segment, 0/1 |
| `reward_sum` | Σ rₜ over the segment |
| `return_discounted` | Σ γᵗ rₜ over the same rewards |
| `return_gae` | buffer's GAE return, mean over the segment's steps |
| `value_mean` | critic prediction, mean over the segment's steps |
| `advantage_mean` | normalized advantage, mean over the segment's steps |

### `success` is the same definition eval uses

Terminal value at the segment's last step, with grasp flags latched within that
segment only. Training gets this because `reset_robot()` clears `_elapsed_steps`
and the grasp latches every segment; sequential eval now matches via
`CronosWrapper.begin_segment()`. So `rollout_success.csv` and
`eval_per_trial.csv` can be compared column to column — same measurement, taken
at different points in the loop.

### `direction` is load-bearing whenever LSR is on

With `--enable-backward` — reset modes `LSR`, `HSR+LSR` and `noep+LSR`, see
[`reset_modes.md`](reset_modes.md) — segments alternate between the forward goal
and a **reset goal**. Three values, because `success` means something different
in each:

| `direction` | Goal | What `success` measures |
|---|---|---|
| `forward` | the scheduler's `put X on Y` | the task was completed |
| `backward` | `put X on table` | **0 by construction** — the env still scores the forward pair |
| `backward_recep` | `put X on <another receptacle>` | the object reached **that** receptacle |

`backward_recep` only appears under `--backward-goal recep|mixed` (the
perturbation option). It works by swapping the env's target receptacle, so the
env's own `success` predicate and its language instruction both follow — which
is why it is scored by the forward reward branch rather than by `src_on_table`.
The receptacle is always chosen different from the forward task's; an env with
only one receptacle falls back to `backward`.

Filter `direction == 'forward'` for a success curve comparable to eval. Without
this column a reset segment's 0 is indistinguishable from a failed forward
segment, and the aggregate reads as a ~50% collapse that is purely an artifact
of the alternation.

Pre-perturbation runs only ever emit `forward` / `backward`, and those two
labels are unchanged, so old and new CSVs stay directly comparable.

### wandb per-task scalars are bucketed by direction

Forward segments populate `rollout/<task>/{success, consecutive_grasp,
is_src_obj_grasped}`; reset segments populate `rollout_reset/<task>/...`
instead. Mixing them corrupts the key in both directions — a to-table segment
contributes a structural 0 to the forward pair, and a to-receptacle segment
contributes a genuine placement against a task the forward policy was never
asked to perform. **This changes `rollout/<task>/*` values for existing LSR
runs**: they previously included the backward segments' zeros.

### The three value columns

They answer different questions and disagree by construction:

- **`reward_sum`** is the raw shaped signal PPO consumed. `RewardShaper` emits a
  potential *difference* (`reward - reward_old`), so summing telescopes to
  `potential(end) - potential(start)`, roughly `[-1.2, 1.2]`. Useful for
  reconciling against the buffer; misleading if read as a return.
- **`return_discounted`** is `Σ γᵗ rₜ` over the same rewards — plain Monte Carlo,
  no bootstrap. The interpretable "what did this segment earn".
- **`return_gae`** is what the critic actually regresses onto (advantage +
  value). Pairs with `value_mean`, `advantage_mean`, and the
  `value_explained_variance` PPO scalar.

### Under `--alg-name grpo` these three columns change meaning

GRPO has no critic, so the same columns carry different quantities. The header
does not change — check `run_config.json`'s `alg_name` before comparing runs.

| Column | under PPO | under GRPO |
|---|---|---|
| `return_gae` | GAE return (`advantage + value`) | group-normalized, **undiscounted** reward-to-go — `buffer_gamma` / `buffer_lambda` are unused |
| `advantage_mean` | GAE advantage, normalized over the whole update | identical to `return_gae` (GRPO has no baseline) |
| `value_mean` | the critic's prediction | still the value head's output, but **untrained** — no gradient reaches it, so this is a frozen readout, not a critic |

`reward_sum` and `return_discounted` are computed in the rollout loop and are
unaffected by the algorithm choice, so they stay comparable across PPO and GRPO
runs. `value_explained_variance` is absent from GRPO's wandb payload entirely;
`grpo_adv_zero_frac` appears instead — the fraction of minibatch samples whose
advantage is exactly zero, i.e. how often a group's rewards came out all
identical and contributed no gradient. See [`results/grpo.md`](results/grpo.md).

### Timing of the GAE columns

`success` and the reward sums are known at the segment boundary; `return_gae`,
`value_mean` and `advantage_mean` only exist after `buffer.compute_gae()`, which
runs once per PPO update (every `ppo_update_len` steps). Rows are therefore
buffered in memory and flushed by `_run_ppo_update` once GAE is available.

The mapping needs no key matching: the buffer lays segments out along its env
axis (`end_segment()` advances `curr_env` by `num_envs`), so buffer slot `k` is
segment `k // num_envs`, env `k % num_envs` — exactly the order rows were
appended. Verified against `CronosReplayBuffer` with per-(segment, env)
distinguishable rewards.

---

## `segment_pose.csv`

Full pose state of the manipulable scene at **both sides** of every segment
boundary — see `phase` below. On by default; disable with `--no-record-segment-pose`. (`--record-end-of-segment-xyz` is
kept as a deprecated alias that also enables it.)

| Column | Meaning |
|---|---|
| `episode`, `segment` | 1-based, matching `train_videos/rollout_ep<N>_seg<M>/` |
| `phase` | `start` or `end` — which side of the boundary, see below |
| `total_steps` | cumulative env-steps at the boundary |
| `env` | parallel env index |
| `actor_kind` | `obj`, `recep`, or `gripper` |
| `slot` | logical slot index within its kind (0-based); `0` for gripper |
| `model_name` | the model occupying that slot **in this env** |
| `task` | the language instruction active for this env |
| `px,py,pz` | position |
| `qw,qx,qy,qz` | orientation quaternion |

Rows per segment: `num_envs × (N + M + 1)`.

### Why every slot, not just the task pair

`get_obj_pos()` / `get_recep_pos()` read `extra_stats`, which only covers the
pair the current task selected. Distractor objects — the ones the policy is
supposed to leave alone — never appeared, so "did the arm knock something off
the table" was unanswerable. `get_all_slot_poses()` walks `_all_carrot_ids` /
`_all_plate_ids` and gathers every slot.

### Why `model_name` is per row

Under fan-out, slot 0 is a different model in different envs (each group brings
its own objects). Without the per-env model name a coordinate row cannot be
attributed to an actual object.

### Hidden slots are NaN, not 0

When a group declares fewer objects than the batch-wide `N`, the extra slots are
hidden (`select` index `-1`). Those write `NaN`. A zero would be
indistinguishable from an actor genuinely at the origin and would silently
become a real data point; `NaN` keeps the row count fixed per segment so the
file pivots cleanly while excluding itself from means.

### `phase` — a boundary is not one instant

Between the end of segment N and the start of segment N+1 sit that boundary's
resets. Both sides are recorded, because they answer different questions:

| `phase` | Recorded | Use it for |
|---|---|---|
| `end` | **before** HSR's respawn and EER's `reset_robot()` | the steady state the policy produced — what `workspace_aabb` bounds should be anchored from |
| `start` | **after** them, and after the full `env.reset()` at an episode boundary | the initial-state distribution each segment actually begins from — what `--backward-goal` (perturbation) is meant to widen |

They are not interchangeable. `--reset-robot` is on by default in *every* reset
mode, so the gripper always differs between the two, and `reset_robot()`'s
`_settle(0.5)` nudges objects as well. Under `--reset-unsuitable` any respawned
actor differs outright.

Row counts per episode: `K` segments produce `K` `start` rows and `K` `end` rows
per (env, actor). The first `start` is taken before the step loop, right after
the episode's `env.reset()`; the last boundary writes no `start` because there is
no segment `K+1` — the same guard `buffer.warmup` uses.

`--segment-pose-phase start|end|both` selects which are written (default `both`).
CSVs written before this column existed contain `end` rows only;
`plotting/plot_segment_positions.py` backfills `phase="end"` when the column is
absent.

### Note on the predecessor

This supersedes `end_of_segment_xyz.csv` (task-selected pair only, xyz only).
That file also labelled every row one episode ahead of the matching
`train_videos/` directory: `episode` already arrives 1-based from `train()` and
was incremented again. Fixed here — only `segment` is incremented.

---

## `eval_per_trial.csv`

Source of truth for standalone eval: one row per (domain, round, task slot, env).
`plotting/mcnemar_pair.py` pairs on `(eval_kind, task, group, seq_idx, task_idx, env_idx)`.
A unit's rows — (eval_kind, pass_label, seq_idx) — are appended together after its
last task slot.

| Column | Meaning |
|---|---|
| `seq_idx` | global round number (see `eval_sequential.md` §3) |
| `task_idx` | task slot within the round |
| `obj_set` | `rand` (in-domain) or `rand_ood` |
| `task` | resolved task string |
| `env_idx` | parallel env index |
| `success` | **B, independent** — this task judged on its own |
| `success_chained` | **A, chained** — cumulative AND along `task_idx` within (domain, round, env) |
| `grasp`, `obj_grasped` | latched within this segment |
| `prefix` | `{eval_kind}_seq{round}_task{slot}` |
| `eval_kind` | `in_domain` / `out_of_domain` |
| `seq_kind` | `training` / `random` / `single` — filter on this to keep training rounds apart |
| `group` | scene (YAML group) |
| `obj`, `recep` | the task's object and receptacle |
| `order` | the order this env runs in this round, as training-order letters (`ABCD`, `BCDA`, …) |
| `pose_set` | set of start poses (`round // rounds_per_set`) |
| `cycle_idx` | 0 = training cycle, 1.. = random cycles; 0 in single mode |
| `pass_label` | `all_scenes` (parallel) or the scene name (serial) |
| `episode`, `total_steps` | checkpoint progress, 0 if unknown |

The first ten columns keep their old names and order. Row count per unit is
exactly `num_envs × task slots`; the eval refuses to continue otherwise.

### AutoRL side

`plotting/parse_autorl_eval.py` emits this same schema from an AutoRL run's video
filenames, with `grasp` / `obj_grasped` left empty (not recoverable — see the
audit doc). Both sides then feed the same `mcnemar_pair.py` and `plot.py`.

`mcnemar_pair.py` treats an empty cell as "not available": pairs missing the
requested metric on either side are skipped and counted, and it says so instead
of scoring them as zeros. So `--metric success` against a recovered AutoRL
baseline works, and `--metric grasp` reports zero comparable pairs rather than
producing a confident-looking result from nothing.

Known gap (read, not fixed): `parse_autorl_eval.py` writes `prefix =
autorl_seq…`, while `mcnemar_pair.py` derives the domain from a prefix starting
with `in_domain` / `out_of_domain`, so recovered AutoRL rows are currently skipped.
AutoRL orders and start states also differ from the round design, so only pooled
rates are comparable anyway.

## `eval_layouts.csv`

One row per env per unit reset.

| Column | Meaning |
|---|---|
| `eval_kind`, `seq_idx`, `seq_kind`, `pose_set`, `pass_label`, `group` | which reset |
| `env_idx`, `scene_env_idx` | batch index, and index within the scene's env range |
| `order` | order this env runs in the round |
| `layout_slot` | slot inside the env's block (mod `layout_slots`); with `pose_set`, the start pose is a function of this alone |
| `layout_key` | RNG stream key, `layout\|seed=0\|domain=in_domain\|set=0` |
| `layout_id` | the drawn 62-bit id |
| `rand_id`, `pos_id`, `quat_id`, `overlay_id` | what the env applied: `rand_id = layout_id % ltt`, `pos_id` indexes `xyz_configs`, `quat_id` indexes `quat_configs` |

Two runs with the same seed produce identical files; within a file every
`(eval_kind, pose_set, layout_slot)` has one `(pos_id, quat_id)`.

## `eval_segment_pose.csv`

Exactly the `segment_pose.csv` columns, same formatting (one shared writer,
`evaluation/records.py::SegmentPoseWriter`), with `episode = seq_idx + 1` and
`segment = task_idx + 1`, followed by `eval_kind, seq_kind, seq_idx, pose_set,
task_idx, group, obj_set, pass_label`. `phase=start` is recorded after the reset
(slot 0) or after `set_task` + `begin_segment` (later slots; the scene is not
touched, so it equals the previous slot's `end`); `phase=end` after the last
step. `total_steps` is the checkpoint's, so plot with `--step-range all`, and
filter on `eval_kind` — both domains reuse `episode` numbers.

## `eval_sequence_summary.csv`

Derived; rebuilt whole from `eval_per_trial.csv`.

| Column | Meaning |
|---|---|
| `level` | `slot` (group, round, order, slot) · `order` (group, round, order) · `pose_set` (group, pose set) · `task` (group, task) · `scene` (group) · `kind` (all scenes) |
| `eval_kind`, `seq_kind` | always part of the key, so training and random rounds never pool |
| `group`, `pose_set`, `seq_idx`, `order`, `task_idx`, `task` | filled as far as the level defines them |
| `n_trials` | trials pooled |
| `success`, `success_chained`, `grasp`, `obj_grasped` | means |

`success_chained` above `level=slot` averages over the positions pooled.

## `eval_coverage.csv`

Derived. Planned (`envs, rounds, pose_sets, orders, tasks_per_round,
distinct_tasks, trials, trials_per_task, resets, layouts`) and counted
(`actual_trials, actual_resets, actual_layouts`) per (eval_kind, seq_kind, group),
with `padded_envs` (always 0 — eval does not pad) and `match`.

## `eval_status.json`

Derived, also refreshed after every unit: `complete` (selected units done),
`full_design` / `rounds_not_selected` (whether the selection covers every round of
every pose set), `coverage_match`, `planned_units`, `done_units`, `missing` /
`partial` unit keys, `overfull` (units with too many rows), `fingerprint`,
`updated`.

## `eval_success.csv`

Aggregate per (eval point, group, task). Training-time eval appends
`eval_kind` `in_domain` / `out_of_domain`. Standalone eval **rebuilds** it with
`eval_kind = <domain>_<seq_kind>` (`in_domain_training`, `in_domain_random`,
`out_of_domain_single`, …), the scene name in `group` and the background label in
`scene` — and writes it **only when the eval is complete and covers the full
design**; an incomplete eval or a round shard has no `eval_success.csv` (see
`eval_status.json`). The older `sequential_seq<N>`
kinds are no longer written. Consumed by `scripts/plot.py` and
`tools/plot_run_trends.py`.
