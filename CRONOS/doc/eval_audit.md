# Eval / sequential-eval audit

**Version:** V0.91

Scope: `main.py` (training-time eval, `--eval-single`, `--eval-sequential`) and
`eval_only.py`, compared against `AutoRL/SimplerEnv/simpler_env/train_ms3_ppo.py`.
AutoRL was read only; nothing in it was modified.

Status: the defect below was found by reading the code and confirmed at every
link in the chain. It has **not** been reproduced on hardware — this machine has
no GPU large enough to load either VLA. Reproducing it is a one-line check on
the training machine (see *Verifying on hardware*).

---

## 1. The defect

**Sequential eval reports the wrong numbers for every task after the first in a
sequence.** Three independent facts combine:

1. `TimeLimitWrapper.step` (`ManiSkill/mani_skill/utils/registration.py:160-168`)
   recomputes `truncated = elapsed_steps >= max_episode_steps` on *every* step
   and never auto-resets. It is a standing comparison, not an edge trigger.
2. `_elapsed_steps` is zeroed in exactly two places: `env.reset()`
   (`sapien_env.py:841`) and `ResetStrategy.reset_robot()` (`envs/reset.py:57`).
   `set_task()` does not touch it.
3. Sequential eval runs `reset = (task_idx == 0)`, so from the second task on
   only `set_task()` is called.

With `segment_len = 80`:

| | `elapsed_steps` on entry | during the segment | `truncated` |
|---|---|---|---|
| task 0 (`reset=True`) | 0 | 1…80 | true on the last step only |
| task ≥ 1 (`reset=False`) | 80 | 81…160 | **true on every step** |

`CronosWrapper.step` emits `info["episode"]` whenever `truncated.any()`, and the
caller accumulates with `env_infos[k] += v`. So a continued segment appends
`segment_len * num_envs` samples where the contract is `num_envs`.

### Consequences

**`success` is under-reported.** `success = src_on_target` (`bridge_multi.py:347`)
is instantaneous and does not latch, so averaging over all timesteps answers
"what fraction of the segment was the object resting on the plate", not "did the
trial succeed". A trial completed at step 60 of 80 scores ≈0.25; completed at
step 79 it scores ≈0.0125.

**Grasp metrics are over-reported.** `is_src_obj_grasped` and `consecutive_grasp`
latch with `|=` (`bridge_multi.py:367-368`) and are cleared only by
`reset_grasp_stats()`, which sequential eval never called. They therefore
carried over from earlier tasks in the sequence and saturated toward 1.0
independent of performance on the current task.

**`eval_per_trial.csv` held wrong values.** The writer indexes `successes[env_i]`
for `env_i` in `[0, num_envs)`, assuming one sample per env. Against the
over-long list those indices land on the *first* truncation event — step 1 of
the task, before the arm has moved — so the column was effectively all zeros for
`task_idx ≥ 1`. `tools/mcnemar_pair.py` reads this file, so the paired
significance test was consuming corrupted data.

### Not affected

Verified by inspection of each path:

- **Training rollout.** `--reset-robot` defaults on in every reset mode and runs
  at each segment boundary, zeroing `_elapsed_steps` and the grasp latches.
- **Training-time rotation eval** (`eval_all_groups`). Calls `env.reset()` at the
  start of every eval episode and runs exactly `segment_len` steps.
- **`--eval-single` / `--eval-mode single`.** Every task resets.

Only the sequential path, `task_idx ≥ 1`, was wrong.

### Relationship to AutoRL

`AutoRL/SimplerEnv/simpler_env/env/simpler_wrapper.py:151-169` is line-for-line
identical in the relevant block, and `render_seq` uses the same
`reset = (j == 0)`. So the success and grasp distortions are **inherited from
AutoRL, not introduced by the refactor**. The `eval_per_trial.csv` corruption is
CRONOS-specific: that export is new, and it landed on the same defect.

AutoRL's affected surface is exactly `--only_render_seq`; its `--only_render`
and its in-training `eval()` both reset per task and are correct.

---

## 2. The fix

`CronosWrapper.begin_segment()` (`envs/wrapper.py`), called on the `reset=False`
branch of both `CronosRunner.eval` and `EvalRunner.eval`. It clears exactly two
things:

- `_elapsed_steps` → `truncated` fires once, at the end of the segment
- `reset_grasp_stats()` → grasp latches accumulate within this segment only

It does **not** touch poses, velocities or contact state, so the no-reset
continuity that sequential eval exists to measure is preserved. This is the same
bookkeeping `reset_robot()` performs during training, minus the physical reset —
so eval and training now compute `success` and grasp identically, and
`rollout_success.csv` and `eval_per_trial.csv` can be compared column to column.

`eval_only.py` also warns if a segment yields more than `num_envs` terminal
samples, so a future regression surfaces instead of silently skewing the data.

### One residual difference, by design

The robot is not reset between tasks, so a gripper still holding the object from
the previous task re-latches `is_src_obj_grasped` on the first `evaluate()` of
the new segment. That is the true state under continuity, not leakage — but it
means grasp metrics in sequential eval are not directly comparable to
single-task eval, where every trial starts from an open gripper.

---

## 3. Scoring semantics: A and B

Both are emitted; neither is a mode switch. One eval run answers both questions,
and no re-run is needed to change the lens.

| Column | Semantics | Definition |
|---|---|---|
| `success` | **B — independent** | This task judged on its own, regardless of earlier tasks in the sequence. AutoRL's semantics, correctly computed. |
| `success_chained` | **A — chained** | Cumulative AND along `task_idx` within one `(obj_set, seq_idx, env_idx)`. Once an env fails, every later task in that sequence scores 0 for it. |

```
success_chained[j] = success[0] AND success[1] AND ... AND success[j]
```

B measures single-task capability. A measures how far into a sequence the policy
survives, and is deliberately order-sensitive: the same task set under different
permutations yields different A values, which is itself informative about which
task orderings are hard.

They coincide at `task_idx == 0`.

---

## 4. Recovering a correct AutoRL baseline

AutoRL's aggregate numbers — the printed log line, `stats.yaml["stats"]`, and
`stats.yaml["last_info"]` — all flow through the affected path. But `render()`
writes video filenames from a *separate* buffer: it collects per-step, per-env
`info` into `datas[i]["info"]` and takes `infos[-1]["success"]`, the terminal
value for that specific env, then bakes it into the name:

```
glob/{seq}-{task}/video_{env}-{obj}_{recep}-s_{0|1}.mp4
```

`tools/parse_autorl_eval.py` reads those filenames plus
`stats.yaml["instruction"]` (for the task string) and emits a CSV with the same
schema as `eval_per_trial.csv`, including both A and B columns. Read-only; it
never writes inside AutoRL.

```bash
python tools/parse_autorl_eval.py \
    --glob-dir /path/to/AutoRL/SimplerEnv/wandb/run-<id>/glob \
    --obj-set rand --out reports/autorl_baseline_per_trial.csv
```

**Requires the mp4 files to still exist.** A run whose videos were deleted keeps
only the polluted aggregate; its correct per-trial values are unrecoverable
without rerunning AutoRL's eval (rerunning is not modifying — `render()` always
writes videos).

### What cannot be recovered

Grasp metrics. They carry both distortions (time-average *and* cross-task latch
carry-over), the per-step values were never persisted — they live only in the
in-memory `datas` buffer — and both `stats.yaml` fields that hold them are
computed through the polluted path. Grasp comparisons against AutoRL are sound
only for `--only_render` (single-task) runs.

---

## 5. Secondary finding: permutation enumeration

`eval_only.py` and `main.py` each materialized `list(itertools.permutations(pool))`
before sampling from it. Fine for the 4-task 2×2 configs (24 orderings); for
`one_group_sequential_3x3` (9 tasks) that is 362,880 tuples built to draw 4.

Both now call `envs/scheduler.py::build_eval_sequences`, which enumerates only
below a threshold and otherwise draws distinct shuffles by rejection sampling.
This also removes a duplicated implementation — the two call sites had
hand-rolled the same logic separately.

Verified: for a 4-task pool the new function returns byte-identical orderings to
both previous implementations across seeds 0-4 and `eval_sequences` ∈ {1,3,5,24},
so existing 2×2 runs stay reproducible. For 9 tasks it returns in 0.09 ms versus
116 ms of pure enumeration.

---

## 6. Impact on existing results

Sequential-eval numbers produced before this change are not comparable to
numbers produced after:

- `success` rises (no longer diluted across timesteps)
- grasp metrics fall (no longer inherited across tasks)
- `eval_per_trial.csv` becomes usable at all for `task_idx ≥ 1`

Single-task eval (`--eval-single`, `--eval-mode single`) and all training-time
metrics are unchanged, and remain directly comparable to prior runs and to
AutoRL.

The recommended baseline path is to rebuild the AutoRL side with
`tools/parse_autorl_eval.py` so both sides use the corrected definition, rather
than preserving a matched-but-wrong metric.

---

## 7. Verifying on hardware

The chain is deterministic and cheap to confirm on any GPU that can load a
policy. In `eval_only.py::eval`, at the top of the `reset=False` branch:

```python
print("elapsed_steps on entry:", self.env.env.unwrapped._elapsed_steps[:4])
```

Before the fix this prints `[80, 80, 80, 80]` for `task_idx ≥ 1`; after, `[0,0,0,0]`.

End to end, run `--eval-mode sequential --eval-sequences 2` and check
`eval_per_trial.csv`: the row count must be exactly
`sequences × tasks × num_envs`, and `task_idx ≥ 1` rows must no longer be
uniformly zero.
