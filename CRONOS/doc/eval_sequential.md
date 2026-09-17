# Standalone eval: scenes, rounds, pose sets, resume

Describes the current tree; the code version is in [`../version.py`](../version.py).
Index of these documents: [`README.md`](README.md).

Scope: `eval_only.py`, `main.py --eval-single / --eval-sequential`, the
`evaluation/` package and `tools/rebuild_eval_outputs.py`. Training-time rotation
eval (`main.py::eval_all_groups`) is not covered. RNG and file-writing lessons
that training should adopt are in [`rng_and_io_notes.md`](rng_and_io_notes.md).

Status: plan, RNG streams, settings, records, outputs, resume and merge are covered
by `tests/test_eval_plan.py` and `tests/test_sequential_eval.py` (fake env, CPU
torch). **Not yet run on hardware**; see §8.

---

## 1. Summary of the design

- **Environment = the training config file** (§2a), including its `eval:` block.
- **Every scene** (YAML group) runs on its own env range, split into 4 **order
  blocks** exactly as fan-out training splits it.
- **Rounds are the only selector.** A round is one reset plus every task slot; its
  number fully determines its orders, start poses and sampling seed (§3).
- **4 tasks: 6 rounds per pose set = all 24 orderings.** Round `6k` is a training
  round (ABCD BCDA CDAB DABC), rounds `6k+1..6k+5` run the other 5 cycles.
- **Same start poses** for every round, order and scene of a pose set; each pose
  set brings new poses (§4).
- **Resume and shard** by round; completeness is per unit, `eval_success.csv`
  exists only for a complete eval (§6).

AutoRL `render_seq` semantics are kept: reset at the start of a round, switch tasks
without resetting, `begin_segment()` reopening the measurement window
([`eval_audit.md`](eval_audit.md)).

## 2. Settings

Precedence per field: **explicit CLI flag > YAML `eval:` block > default.** A CLI
flag counts only if it is on the command line (including `--no-…`).

| YAML `eval:` key | CLI flag | Default | Meaning |
|---|---|---|---|
| `mode` | `--eval-mode` | `sequential` | `sequential`: rounds of task sequences. `single`: one task slot per round, blocks run A/B/C/D side by side |
| `pose_sets` | `--eval-pose-sets` | `1` | sets of start poses; the eval has `pose_sets × R` rounds (R = 6 for 4 tasks, 1 in single mode) |
| `rounds` | `--eval-rounds` | `all` | global round numbers: `all`, `3`, `3-5`, `3-`, `0,6-11`. `all` is refused when R > 24 (e.g. 9 tasks) |
| `sequence_seed` | `--eval-sequence-seed` | `-1` (= `--seed`) | cycle-order stream |
| `layout_seed` | `--eval-layout-seed` | `-1` | start-pose stream |
| `policy_seed` | `--eval-policy-seed` | `-1` | per-unit reseed of action sampling |
| `layout_slots` | `--eval-layout-slots` | `-1` | start poses per block; `-1` = one per env of the block, `1` = one pose |
| `scene_schedule` | `--eval-scene-schedule` | `parallel` | `parallel`: all scenes in one batch. `serial`: one scene × all envs per pass |
| `domains` | `--eval-domains` | `in_domain,out_of_domain` | subset; `--no-eval-ood` = `in_domain` |
| `record_video` | `--record-video` | `true` | mp4s |
| `video_envs_per_block` | `--video-envs-per-block` | `-1` | first K envs of each block (1 = one video per order per round) |
| `record_pose` | `--record-eval-pose` | `true` | `eval_segment_pose.csv` |
| `pose_phase` | `--eval-pose-phase` | `both` | `start` / `end` / `both` |
| — | `--policy`, `--vla-path`, `--vla-unnorm-key`, `--vla-temperature-eval`, `--vla-lora-rank` | from the checkpoint (§2b) | override the training policy settings |
| — | `--eval-resume <glob>` | — | continue an interrupted eval in its directory (§6) |
| — | `--allow-config-mismatch` | false | evaluate a config whose scenes differ from training (§2a) |

Removed keys are rejected with a pointer: `num_sequences` and
`include_training_sequence` (use `rounds`), `training_orders` (always rotations),
`random_orders` (always cycles), `layout_rng` (always seeded).

`eval:` keys are validated when the config loads (V30); checks that need resolved
tasks run when the plan is built, before the first action is sampled.

### 2a. Environment config = the training config file

Resolution (`evaluation/provenance.py`): `--config-path` if given, else the
`experiment_config.yaml` snapshot `main.py` writes into `glob/` and every
`episode_XXXX/`, else the `config_path` in the checkpoint's `run_config`.
`scripts/eval.sh <ckpt>` (config empty or `-`) takes this path.

The chosen file's scene definition — `fan_out, scene, task_order, env_n/env_m`,
legacy index keys, per group `name, num_envs, obj, recep, table, background,
task_sequence, eval_tasks` — is compared with the training reference. A difference
stops eval before the model loads. For old checkpoints without a snapshot, the
comparison is against the recorded file as it is now, and the plan says so. The
reduced-env configs under `configs/eval/` differ in `num_envs` and need
`--allow-config-mismatch`.

### 2b. Policy settings = the training run's (OpenVLA or SpatialVLA)

`evaluation/provenance.py::resolve_policy_args` sets `policy, vla_path,
vla_unnorm_key, vla_temperature_eval, vla_lora_rank` before the model loads,
per field: explicit CLI flag > the checkpoint's training `run_config` > the config
YAML (`policy`, `vla_path`, `vla_unnorm_key`) > per-policy defaults mirroring
`scripts/train.sh`:

| policy | vla_path | vla_unnorm_key | vla_temperature_eval |
|---|---|---|---|
| `openvla` | `openvla/openvla-7b` | `bridge_orig` | 0.6 |
| `spatialvla` | `IPEC-COMMUNITY/spatialvla-4b-224-sft-bridge` | `bridge_orig/1.0.0` | 0.0 (greedy) |

So `scripts/eval.sh <ckpt>` evaluates a SpatialVLA checkpoint with no extra flags.
The run_config is used only when its policy is the one being evaluated (a
run_config without `policy` is an OpenVLA run). A CLI value that differs from
training, a policy mismatch, or a checkpoint without run_config is printed as a
warning and recorded in `eval_plan.json → provenance.policy_args`, and every
resolved value is part of the fingerprint. Note that
`configs/spatialvla_2x2_train.yaml` does not set `vla_temperature_eval`, so runs
launched from it without `train.sh` trained with main.py's 0.6 — the run_config
records which. `main.py --eval-single/--eval-sequential` applies the same
resolution.

`--action-chunk K` (SpatialVLA open-loop chunks) is an eval-only choice and is
not taken from training.

## 3. Rounds

### 3a. Orders

Fan-out training splits each group into one sub-block per task and gives sub-block
t `pool[(t + offset) % n]`, with `offset` advancing per forward segment — so the
batch runs the cyclic rotations **ABCD, BCDA, CDAB, DABC** (letters = positions in
the training order) side by side. Those four are one **cycle**. n tasks have
(n−1)! cycles × n rotations = n! orderings; for 4 tasks 6 × 4 = 24.

Round r, with R = (n−1)! rounds per pose set:

```
pose_set  = r // R
cycle_idx = r %  R
cycle_idx 0      training round: block t runs the rotation starting at task t
cycle_idx c ≥ 1  random round: block t runs rotation t of random_cycles(n, sequence_seed)[c − 1]
```

Every pose set uses the same cycle order, so pose sets × 24 orders are fully
crossed. Within a round every task is at every position exactly once per scene.

Why the same cycle order in every pose set (decided; alternative was reshuffling
per set with a `set` field in the cycle key):
- Rounds carry no state (reset, frozen policy, per-round sampling seed), so a
  cycle's position inside a set is only a label. With full sets both designs
  evaluate every (pose set, cycle) pair once and give identical aggregates.
- They differ only for partial selections such as `rounds: 0-2,6-8`: the same
  order keeps pose-set comparisons on identical cycles (pose effect only);
  reshuffling would reach more cycles but confound set with cycle and unbalance
  the crossing. With more than 4 tasks, where partial selections are the norm,
  reshuffling would spread coverage — revisit then.
- Analysis note: of the 5 non-training cycles, 4 share exactly one transition with
  the training cycle and one (the reverse, A→D→C→B) shares none, so "random"
  rounds are not all-untrained transitions.
`test_training_round_matches_scheduler_fan_out` drives the real scheduler and
checks round 0 gives every env its training task sequence.

**Single mode**: R = 1, round r = pose set r; the blocks run tasks A, B, C, D (one
task slot) side by side — AutoRL `render` semantics with the same block structure.

Caveats:
- Exact env ↔ rotation pairing holds for T320/T1280/T2560 without LSR; otherwise
  only the set of rotations matches (see `rng_and_io_notes.md` A7).
- A scene whose training did not rotate (`fan_out: false`, one training task, eval
  tasks ≠ trained pool) is not split in sequential mode: all its envs run the
  round's first order; the plan printout says why. Single mode always splits.
- A scene's env count must divide by n; eval refuses instead of padding.
- A 2-task scene has one cycle: R = 1.

### 3b. What makes a round reproducible

For a fixed seed a round is a function of its number, whatever the checkpoint,
the selection, the scene schedule, or what ran before it:

| | stream key | tested |
|---|---|---|
| orders | `cycle\|seed\|n_tasks`, prefix-stable shuffle | selecting `3-4,9` reproduces those rounds of a full run |
| start poses | `layout\|seed\|domain\|set`, slot j = j-th draw | layouts identical across different policies; every slot maps to one (pos_id, quat_id) |
| action sampling | `policy\|seed\|domain\|round\|pass`, torch reseeded per unit | crash + resume is byte-identical to an uninterrupted run with an action-dependent outcome |

Implemented in [`../envs/rng_streams.py`](../envs/rng_streams.py). The env accepts
`options["layout_ids"]` and otherwise keeps its `torch.randint` draw, so training
is unaffected.

**What `--seed` does in eval.** It is only the default for the three stream seeds
(`sequence_seed`, `layout_seed`, `policy_seed`). Its other uses have no effect on
results: the global `random`/`numpy`/`torch` seeding at startup (the loop reseeds
torch per round and draws nothing else), `rand_episode_id` (overlay index is 0 for
any value ≤ 1000), the wrapper's `reset(seed=seed*1000+i)` (reseeds an episode RNG
this env does not use), and the policy constructor's value-head/LoRA init (both
replaced by the checkpoint). It is **not** taken from the checkpoint, and matching
the training seed has no meaning: eval draws from its own streams, so even the same
number gives different start states than training had. Keep one eval seed for
every checkpoint you want to compare — they then see identical rounds; use
`pose_sets` (or a different `layout_seed`) for more start states.
`scripts/eval.sh` passes `--seed 0`; a later `--seed N` among the extra flags
wins (tyro keeps the last occurrence), or set the three `--eval-*-seed` flags.

## 4. Start poses

An env's start pose depends only on its **pose set** and its **slot** inside its
block: `slot = (env − scene_start) mod block_size [mod layout_slots]`.
With 16-env scenes (block = 4 envs) each pose set has 4 poses, shared by every
round, order and scene of the set; the next set draws 4 new ones. `serial` blocks
are 16 envs, and their slots 0–3 are the parallel poses.

Trade-off: per scene and domain there are `4 × pose_sets` distinct start layouts
(16 × pose_sets with serial). Order effects are measured on identical starts;
breadth over layouts comes from `pose_sets`.

Recorded: `eval_plan.json → rng` (keys, ids, per-env slot map, per-unit torch
seeds) and `eval_layouts.csv` (per reset and env: order, slot, id, applied
pose/quat/overlay ids). Settling on GPU is not bit-deterministic; realised poses
are in `eval_segment_pose.csv`.

## 5. Coverage

Per domain, per scene: `trials = envs × task slots` summed over rounds,
`per_task = trials / tasks`, `resets = envs × rounds`, `layouts = pose sets × slots`.

`four_group_sequential_2x2`, per pose set:

| schedule | seq_kind | rounds | orders | trials | per task | per (order, slot) | resets | layouts | env-steps (all scenes, both domains) |
|---|---|---|---|---|---|---|---|---|---|
| parallel | training | 1 | 4 | 64 | 16 | 4 | 16 | 4 | 40,960 |
| parallel | random | 5 | 20 | 320 | 80 | 4 | 80 | 4 | 204,800 |
| serial | training | 1 | 4 | 256 | 64 | 16 | 64 | 16 | 163,840 |
| serial | random | 5 | 20 | 1,280 | 320 | 16 | 320 | 16 | 819,200 |

Multiply by `pose_sets`. One full parallel pose set is 245,760 env-steps.

## 6. Units, resume, shards, completeness

A **unit** is `(domain, pass, round)`. Each unit writes its per-trial and layout
rows together after its last task slot, then updates `eval_status.json`; pose rows
are written as they happen.

**Resume** (`--eval-resume <glob>`, e.g. after a crash in round 3):
1. the command must have the same fingerprint and the same `rounds`/`domains` as
   the original run. The fingerprint covers the checkpoint, config scenes, seeds,
   num_envs, segment_len, policy/sampling settings (`vla_path`, unnorm key, LoRA
   rank, `vla_temperature_eval`, `action_chunk`, `buffer_inferbatch`), `env_id`,
   `obj_set`, and every `eval:` setting **except** `rounds`, `domains`,
   `pose_sets` (design size only — round r does not depend on it) and
   `record_video` / `video_envs_per_block` (videos do not touch results, so a
   resume may switch them). `record_pose` / `pose_phase` are included, so the pose
   CSV cannot cover some units and not others;
2. a unit is done iff its per-trial rows = `num_envs × task slots`; a partial unit's
   rows are removed from every source file (chained success needs the whole
   round); more rows than planned is an error;
3. remaining units run in order; `eval_plan.json → resumes` records what happened.

**Shards**: run different `rounds` (and/or `domains`) into separate directories —
e.g. on several GPUs — then
`python tools/rebuild_eval_outputs.py --out <merged> <glob_a> <glob_b>`. Shards must
share a fingerprint; a unit complete in two shards is an error; partial units are
left out and reported. To *add* rounds to a finished eval, run them as a new shard
and merge — including a new pose set: a finished `pose_sets: 1` eval plus a
`--eval-pose-sets 2 --eval-rounds 6-11` run merge into a complete 12-round eval
(the design size is the largest `pose_sets` among the inputs).

**Completeness**: `evaluation/outputs.py::rebuild_outputs` is the only writer of
derived files and runs at the end of every path (normal, resume, merge, repair
with `tools/rebuild_eval_outputs.py <glob>`):

| File | Written when |
|---|---|
| `eval_status.json` | always (and after every unit): `complete` (selected units done), `full_design` (selection = every round of every pose set), done/partial/missing units |
| `eval_sequence_summary.csv`, `eval_coverage.csv`, `eval_report.txt` | always, rebuilt whole; the report starts with `status: COMPLETE` / `INCOMPLETE (k/n units)` |
| `eval_success.csv` | **only** when every planned unit is done, coverage matches and the selection is the full design; otherwise removed. A shard therefore has none until merged |
| wandb scalars | same condition |

## 7. Outputs

All in the eval run's `glob/`. Columns in [`data_schemas.md`](data_schemas.md).

| File | Content |
|---|---|
| `eval_plan.json` | settings + sources, fingerprint, provenance, scenes, every selected round's orders/tasks per block, units, planned coverage, RNG record, checkpoint progress, resume history |
| `eval_status.json` | completeness (§6) |
| `experiment_config.yaml` | copy of the config used |
| `eval_per_trial.csv` | per (domain, round, task slot, env); old 10 columns first, then `eval_kind, seq_kind, group, obj, recep, order, pose_set, cycle_idx, pass_label, episode, total_steps` |
| `eval_layouts.csv` | per reset and env |
| `eval_segment_pose.csv` | training `segment_pose.csv` columns + `eval_kind, seq_kind, seq_idx, pose_set, task_idx, group, obj_set, pass_label` |
| `eval_sequence_summary.csv` | levels slot / order / pose_set / task / scene / kind, always keyed by `seq_kind` |
| `eval_coverage.csv` | planned vs actual trials, resets, layouts |
| `eval_report.txt` | status, then per domain: training rounds, random rounds; per scene by pose set, order, position, task |
| `eval_success.csv` | `eval_kind = <domain>_<seq_kind>`, complete evals only |
| `eval_videos/<domain>/<seq_kind>/round<N>/task<M>/<group>-<order>-env<i>-…-s<0/1>.mp4` | videos |

## 8. Verifying on hardware

```bash
cd Benchmark/CRONOS
python -m pytest tests/ -q        # CPU only; needs pyyaml, numpy, torch, tqdm

bash scripts/eval.sh <glob>/episode_XXXX - <cuda> 4 --eval-rounds 0-1 --video-envs-per-block 1
```

1. The plan printout shows the config source, `training config check: match`, four
   scenes, `round 0 (set 0, training): ABCD BCDA CDAB DABC`, block = 4 envs,
   4 start poses.
2. `eval_status.json` complete; `eval_coverage.csv` all `match = 1`;
   `eval_success.csv` present.
3. Kill a run during round 1, rerun with `--eval-resume <glob>` and the same
   flags; `eval_layouts.csv` must equal an uninterrupted run's, and per-trial
   results should match up to GPU physics non-determinism.
4. Same command on a second checkpoint: `eval_layouts.csv` byte-identical.

## 9. Differences from AutoRL

Re-audited against `AutoRL/SimplerEnv/simpler_env/train_ms3_ppo.py` (`render`,
`render_seq`, `run`), `simpler_wrapper.py`, `pick_place_multi.py`
(`TwoObjectTwoReceptacle`) and AutoRL's `openvla`. Marks: **[local]** checked by a
script on this machine, **[read]** by reading code.

### 9a. Identical

| Item | Check |
|---|---|
| `success`, `is_src_obj_grasped`, `consecutive_grasp` formulas in `evaluate()` | AST diff: only the second-receptacle block and init guards differ (see 9c) **[local]** |
| In-domain pose table (432 configs), OOD pose table (106,032), quaternion table | numeric equality **[local]** |
| Layout distribution: uniform over `ltt`, then `pos = (id // l2) % l1`, `quat = id % l2` | same formula; ids uniform **[read]** + distribution test **[test]** |
| Background overlay in a single-group config | `rand_episode_id ≤ 1000 < l1·l2 = 1728`, so overlay index 0 in both **[read]** |
| Env construction seeds `seed*1000+i`, `rand_episode_id`, sim 500 Hz / control 5 Hz, 80-step segments | **[read]** |
| Action decoding (256-bin centers, q01/q99 unnorm, gripper `2·(x>0.5)−1`) | **[read]** |
| Prompt, left padding, `predict_action_batch`, temperature 0.6 sampling, 32-env inference micro-batches | diff identical except removed `cuda.synchronize()` **[local]** |
| Training task order and fan-out: AutoRL `run()` gives block i `task_list[(task_id+i)%4]`, `task_id` advancing per forward switch and carried across episodes | same as `TaskScheduler` **[read]** |
| Round/sequence mechanics: reset at the start, no env or robot reset between tasks, first obs from `reset`, later obs from `get_obs_image` | **[read]** |

### 9b. Deliberate differences

| # | AutoRL `render_seq` / `render` | CRONOS | Effect on comparing numbers |
|---|---|---|---|
| 1 | `_elapsed_steps` not reset between tasks: from task 2 on, aggregate `success` is a time-average (video filenames hold the correct terminal value) | `begin_segment()`: terminal value | AutoRL aggregates are wrong; rebuild with `tools/parse_autorl_eval.py` |
| 2 | grasp latches carry over across tasks | cleared per task slot | grasp metrics not comparable for task slot ≥ 1 |
| 3 | every env runs one order per sequence | 4 blocks run 4 orders per round; round 0 = the rotations AutoRL's own *training* ran | compare pooled rates, not per-sequence |
| 4 | 4 random permutations sampled from the 23 non-identity orders; can include training rotations (seeds 0 and 1 draw `BCDA`) **[local]** | 5 untrained cycles = the 20 non-rotation orders, each once | "random" means untrained only in CRONOS |
| 5 | 5 sequences × 64 envs, fresh layouts per sequence from the CUDA RNG (checkpoint-dependent) | 6 rounds per pose set, 4 fixed start poses per scene per set, seeded | different trial populations; more layouts in AutoRL |
| 6 | no reseed; sampling depends on everything before | torch reseeded per round | statistically equivalent |
| 7 | `render_seq` evaluates only `--obj_set` (in-domain) | both domains by default | set `domains: [in_domain]` to match |
| 8 | single scene | every YAML group | — |
| 9 | `render` (single task): a reset per task, all 64 envs on it | one reset per round, blocks run A/B/C/D side by side | 64 vs 16 envs per task per pass |
| 10 | first frame after reset shows the robot at the stale pre-settle pose (no GPU sync after `agent.reset`) | GPU state pushed before the first render (pre-existing CRONOS fix) | the first action of every round sees a different image |
| 11 | aggregate `stats.yaml` + videos | per-trial CSV, `success_chained`, status, `eval_success.csv` only when complete | — |

### 9c. Found during the audit, not an eval difference

AutoRL sets `select_extra2_ids = plate slot 2` at episode init, so `src_on_target2`
marks "object on plate slot 2" and `src_on_table` excludes both plates. CRONOS
never sets `select_extra2_ids`: `src_on_target2` is always False and `src_on_table`
is true for an object resting on the non-target plate. `success` and grasp do not
use it, so eval is unaffected; the LSR backward reward
(`src_on_table & is_src_obj_grasped`, `envs/reward.py`) is. **[read]**, not fixed.

### 9d. Known gap

`tools/parse_autorl_eval.py` writes `prefix = autorl_seq…`, which
`tools/mcnemar_pair.py` does not map to a domain, so recovered AutoRL rows are
skipped. With 9b #3–#5 only pooled rates are comparable in any case.
