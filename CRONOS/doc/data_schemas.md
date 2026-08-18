# CSV output schemas

**Version:** V0.91

Every file lands in the run's `glob/` directory. All of them are append-only,
written with stdlib `csv`, and carry both x-axes (`total_steps` and
`total_resets`) so any curve can be plotted against either without a re-run.

| File | One row per | Written by | Enabled by |
|---|---|---|---|
| `rollout_success.csv` | (episode, segment, env) | training rollout | always |
| `segment_pose.csv` | (episode, segment, env, actor) | training rollout | always (`--no-record-segment-pose` to disable) |
| `eval_success.csv` | (eval point, group, task) | training eval + `eval_only.py` | always |
| `eval_per_trial.csv` | (sequence, task, env) | `eval_only.py` | always |

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
| `direction` | `forward` or `backward` — see below |
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

### `direction` is load-bearing under LSR / noep

With `--enable-backward`, segments alternate between the forward goal and a
**reset goal**. Three values, because `success` means something different in
each:

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
identical and contributed no gradient. See [`grpo_autorl.md`](grpo_autorl.md).

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

Full pose state of the manipulable scene at the end of every segment. On by
default; disable with `--no-record-segment-pose`. (`--record-end-of-segment-xyz` is
kept as a deprecated alias that also enables it.)

| Column | Meaning |
|---|---|
| `episode`, `segment` | 1-based, matching `train_videos/rollout_ep<N>_seg<M>/` |
| `total_steps` | cumulative env-steps at segment end |
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

### Recorded before the resets

The dump runs before HSR/LSR fire at that boundary, so coordinates are the
steady state the policy produced, not a post-respawn placement. This is what
makes the file usable for deriving `workspace_aabb` bounds.

### Note on the predecessor

This supersedes `end_of_segment_xyz.csv` (task-selected pair only, xyz only).
That file also labelled every row one episode ahead of the matching
`train_videos/` directory: `episode` already arrives 1-based from `train()` and
was incremented again. Fixed here — only `segment` is incremented.

---

## `eval_per_trial.csv`

Per-env outcome for each (sequence, task) in a standalone eval. This is the file
`tools/mcnemar_pair.py` pairs on, with key `(eval_kind, task, env_idx)`.

| Column | Meaning |
|---|---|
| `seq_idx` | ordering index; 0 is the training order |
| `task_idx` | position within the ordering |
| `obj_set` | `rand` (in-domain) or `rand_ood` |
| `task` | resolved task string |
| `env_idx` | parallel env index |
| `success` | **B, independent** — this task judged on its own |
| `success_chained` | **A, chained** — cumulative AND along `task_idx` |
| `grasp`, `obj_grasped` | latched within this segment |
| `prefix` | `{kind}_seq{i}_task{j}`, matches the video subdirectory |

Both scoring semantics are always emitted; see `doc/eval_audit.md §3`.

Row count must be exactly `sequences × tasks × num_envs`. A larger count means a
segment reported on more than its final step — `eval_only.py` prints a warning
when it sees this.

### AutoRL side

`tools/parse_autorl_eval.py` emits this same schema from an AutoRL run's video
filenames, with `grasp` / `obj_grasped` left empty (not recoverable — see the
audit doc). Both sides then feed the same `mcnemar_pair.py` and `plot.py`.

`mcnemar_pair.py` treats an empty cell as "not available": pairs missing the
requested metric on either side are skipped and counted, and it says so instead
of scoring them as zeros. So `--metric success` against a recovered AutoRL
baseline works, and `--metric grasp` reports zero comparable pairs rather than
producing a confident-looking result from nothing.

---

## `eval_success.csv`

Unchanged. Aggregate per (eval point, group, task), with `eval_kind`
distinguishing `in_domain` / `out_of_domain` / `sequential_seq<N>`. Consumed by
`scripts/plot.py` and `tools/plot_run_trends.py`.
