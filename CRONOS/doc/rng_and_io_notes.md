# RNG and file-writing notes (from the standalone-eval rework)

Describes the current tree; the code version is in [`../version.py`](../version.py).
Index of these documents: [`README.md`](README.md).

Scope: problems found while making standalone eval reproducible and resumable
([`eval_sequential.md`](eval_sequential.md)). Eval has been fixed; **training has
not** and still has most of the RNG issues below. This document is the checklist
for bringing training onto the same footing.

How each claim was checked is marked:
**[test]** covered by `tests/` (CPU, fake env), **[local]** a script run on this
machine, **[read]** established by reading code only. Nothing here was run with a
real VLA and ManiSkill on a GPU.

---

## Part A — RNG

### A1. Object layouts share a generator with action sampling

`GenericNxMPickPlace._initialize_episode_pre` draws each env's pose/quaternion
index with `torch.randint(..., device=cuda)` — the global CUDA generator. Action
sampling (`predict_action_batch`, temperature > 0) draws from the same generator.
**[read]**

Consequences:
- The layout an env gets at reset N depends on how many actions were sampled
  before it: a different checkpoint, `vla_temperature`, `buffer_inferbatch`, number
  of envs, or an extra eval pass shifts every later layout.
- Two training runs that differ only in something policy-side do not face the
  same start states, so start-state effects and policy effects are confounded.
- Start states cannot be reconstructed after the fact. README already notes this
  for `segment_pose.csv` `phase=start`: the draws were never logged.

Eval fix: the env accepts `options["layout_ids"]` (per-env 62-bit ids; `% ltt`
gives the same uniform distribution) and eval passes ids from
`envs/rng_streams.py`. Without the option the env keeps `torch.randint`. **[test]**

Training fix (V0.99 hotfix): `main.py` passes `scene_ids` keyed
`scene|seed|kind=episode|episode` at every episode reset,
`scene|seed|kind=eval|point|domain|round` at every training-time eval reset
(`point` = training episodes completed, so the at-start eval and the eval after
the next episode never share layouts, and a resumed run's at-start eval matches
the uninterrupted run's),
and `scene|seed|kind=respawn|episode|segment` to HSR (`respawn_ids`, replacing
`np.random.choice`). The task schedule has its own `task|seed` generator.
`--legacy-rng` restores the old draws. **[test]** for the streams and the
scheduler; the env-side `respawn_ids` path **[read]**.

### A2. Hidden reseeds in constructors

Global generators are reseeded in places a reader would not look for them:

| Where | What | Effect |
|---|---|---|
| `OpenVLAPolicy.__init__` (`simpler_env/policies/openvla/openvla_train.py`) | `torch.manual_seed(seed)` + `cuda.manual_seed_all(seed)` before the value-head init, again before LoRA init | Everything drawn from torch before the policy is built is replayed afterwards. **[read]** |
| `CronosWrapper.__init__` (`envs/wrapper.py`) | `random.seed(seed)` then `random.randint(0, 1000)` for `rand_episode_id` | Resets Python's global `random`, which the scheduler (`random.shuffle`, `random.choices`) uses afterwards. **[read]** |
| `main.py` / `eval_only.py` start | `random.seed`, `np.random.seed`, `torch.manual_seed` | The intended seeding, but A2's other rows override part of it. **[read]** |

**Not** a leak, despite appearances: ManiSkill's `BaseEnv.reset(seed=...)` calls
`torch.manual_seed(episode_seed)` only inside `torch.random.fork_rng()` during
reconfiguration, so the global torch state is restored. **[read]**
(An earlier message in this work claimed the wrapper's seeded reset reseeds torch
to `seed*1000`; that was wrong.)

For training: reseed only at explicit, documented points. Constructors that need
deterministic init should use a local `torch.Generator` or `fork_rng`.

### A3. One global generator, many consumers

| Generator | Consumers | Coupling |
|---|---|---|
| numpy global | HSR respawn `np.random.choice` (`envs/bridge_multi.py::reset_unsuitable_envs`), PPO minibatch `np.random.permutation` (`training/buffer.py`) | Turning HSR on changes minibatch order, and vice versa. **[read]** |
| Python `random` | scheduler `sequence_random` / `pure_random` draws, `rand_episode_id` | Any new consumer shifts the task schedule. **[read]** |
| torch CUDA | layout draw (A1), action sampling | See A1. **[read]** |

Good existing pattern: `main.py`'s `_backward_rng` for perturbation is a
dedicated `random.Random`, so enabling perturbation does not move other streams.

For training: one dedicated generator per consumer, or better, named streams (A5).

### A4. Samplers that are not prefix-stable

`random.sample(population, k)` picks different elements for different `k`, so
"sequence N" changed when the number of sequences changed. **[local]** Eval now uses
a one-time shuffle of the candidate list (`rng_streams.random_orders`,
`random_cycles`): item N is the same for any count > N. **[test]**

The same trap applies to anything drawn "k at a time" in training (e.g. choosing
which envs to perturb). Draw element by element from a keyed stream instead.

### A5. Named streams (`envs/rng_streams.py`)

Every draw comes from `random.Random(sha256(key))` with a readable key such as
`layout|seed=0|domain=in_domain|set=1`. The value depends only on its key, the key
is the record, and adding consumers never shifts existing draws. Stdlib only, no
dependence on training. **[test]**

Keys in use: `layout|seed|domain|set`, `cycle|seed|n_tasks`,
`sequence|seed|n_tasks|exclude`, `policy|seed|domain|round|pass`.

### A6. Per-unit policy reseed makes resume exact

Eval reseeds torch from `policy|seed|domain|round|pass` at the start of every unit.
With it, running rounds 3–5 alone reproduces rounds 3–5 of a full run, and a
crashed-then-resumed eval is byte-identical to an uninterrupted one — the test
uses an outcome that depends on the sampled actions. **[test]** Physical
simulation on GPU is not bit-deterministic (A8), so on hardware expect equality up
to that.

For training resume: reseed at a checkpointable boundary (episode or PPO update)
from `(seed, episode)`, and restore every dedicated generator from the checkpoint.

### A7. Other randomness worth knowing about

- **Eval "deterministic" still samples.** `get_action(..., deterministic=True)`
  uses `vla_temperature_eval = 0.6`, not argmax. Eval outcomes are stochastic.
  **[read]**
- **Object scale is drawn from an unseeded generator — harmless today.**
  `_load_scene` calls `self.np_random.choice(scale_list)` once per model at
  construction. ManiSkill does not define `np_random`, so this is gymnasium's lazy
  property, seeded from OS entropy on first use — not by `--seed` and not by
  ManiSkill's `2022 + i` seeds. No model in `more_carrot` / `more_plate`
  `model_db.json` (CRONOS or AutoRL) has a `scale` list, so every draw is from
  `[1.0]` and the result is fixed. **[local]** Adding a multi-scale model would make
  object size differ between processes; seed that draw explicitly first. (An
  earlier version of this note said the fixed construction seeds applied; they do
  not.) Construction is the only reconfigure: `reconfiguration_freq` is 0, so the
  wrapper's `reset(seed=[seed*1000+i])` reseeds only the episode RNG, which this
  env does not use. **[read]**
- **Overlay/background is not random** in the shipped configs: an int `background`,
  or `background: default` derived from the fixed `rand_episode_id`. **[read]**
- **Fan-out offset carries across episodes.** `TaskScheduler._fan_out_offsets`
  advances once per forward segment and is not reset per episode, so which
  sub-block starts at which task depends on segments per episode and on LSR's
  backward segments. Exact env ↔ rotation pairing holds for T320/T1280/T2560
  without LSR. **[read]**; the T320 case **[test]**.
- **AutoRL parity trap.** AutoRL's `render_seq` samples permutations from the
  global `random` *after* the wrapper's `random.randint`; a fresh
  `random.Random(seed)` matched for seeds 0 and 2 but not 1. **[local]**

### A8. What cannot be made bit-exact

GPU PhysX settling after placement, and cuBLAS/attention kernels across torch
builds (README's env matrix). Record realised state (`eval_segment_pose.csv`,
`segment_pose.csv`) rather than assuming it from the ids.

---

## Part B — File writing

### B1. Aggregates written last went missing

Old eval wrote `eval_success.csv` rows only at the very end, and the end-of-run
coverage check raised **before** that write: a run with complete per-trial data
could end with no aggregate. **[read]**

Fix: aggregates are *derived* — `evaluation/outputs.py::rebuild_outputs` rebuilds
summary, coverage, report, status and `eval_success.csv` from the per-trial rows,
and is the only writer of those files. **[test]**

### B2. An empty file that looks like a result

`SuccessRecorder.__init__` writes the `eval_success.csv` header immediately, so a
crashed run left a header-only file — indistinguishable from "ran, produced
nothing". **[read]**

Fix: standalone eval no longer creates the recorder. `eval_success.csv` exists
**only when the eval is complete and covers the full design**; otherwise
`eval_status.json` lists done, partial and missing units, and a stale file is
deleted. **[test]**

A related trap caught while testing: a round *shard* is "complete" relative to its
own selection. Letting it write the aggregate would produce a plausible-looking
file covering 3 of 6 rounds. "Complete" has to be measured against the full
design, not against what the command asked for. **[test]**

### B3. Append-only aggregates duplicate on relaunch

`SuccessRecorder` appends and rebuilds its in-memory history from an existing CSV.
Rerunning the final aggregation into the same directory duplicated rows and
wandb overlay points. **[read]** Fix: B1's rebuild rewrites whole files. **[test]**

### B4. Silent padding of missing values

The old per-trial writer used `successes[env_i] if env_i < len(successes) else 0.0`
and, on the continued-segment path, indexed the wrong timestep
([`results/bug_reports.md`](results/bug_reports.md)). Fix: exactly one terminal report per segment,
on its last step, with `num_envs` values — anything else raises. **[test]**

### B5. Consumer keys narrower than the data

`plotting/mcnemar_pair.py` keyed trials by `(eval_kind, task, env_idx)`; with several
rounds every round overwrote the previous one. Fix: key includes group, round and
task slot, and duplicates raise. **[test]**

General rule: every row carries its full unit key (`eval_kind, pass_label,
seq_idx, pose_set, task_idx, env_idx`), including rows in files that "obviously"
belong to one domain — `eval_segment_pose.csv` reuses `episode` numbers across
domains and would otherwise be ambiguous.

### B6. Recording a path instead of the content

`run_config` stored only `config_path`. A config edited after training silently
changed what "the training config" meant. Fix: `main.py` copies the file to
`glob/experiment_config.yaml` and into every checkpoint; eval compares scene
definitions against it. **[test]**

### B7. Column reuse

Old standalone eval wrote the sequence index into `eval_success.csv`'s
`total_steps`, which plotting tools read as a training-step axis. Fix: separate
columns; `total_steps` is the checkpoint's. **[read]**

### B8. Crash consistency for resumable runs

The unit of completion must be explicit:
- per-trial rows of a unit are written together after its last slot, so a crash
  mid-unit leaves none; pose rows are written as they happen;
- on resume, a unit is done iff it has exactly the planned row count; partial
  units' rows are dropped from every source file; more rows than planned raise;
- a fingerprint (checkpoint, config scenes, seeds, sampling settings) must match
  before appending; the selection (`rounds`, `domains`) must match too;
- the fingerprint must contain exactly what changes a unit's result: an earlier
  version also hashed `pose_sets` (design size) and video settings, which
  needlessly blocked adding a pose set later and turning videos off on resume;
- shards with the same fingerprint merge (`tools/rebuild_eval_outputs.py`), and a
  unit complete in two shards raises. **[test]**

### B9. Mechanics

- Derived files: write to a temp file, `fsync`, rename. **[test]**
- Appends to an existing CSV check the header first. **[test]**
- A directory holding `rollout_success.csv` is refused as a standalone-eval
  output, because `eval_success.csv` there belongs to training. **[test]**
- Videos: one device→host copy per step instead of one per env, written by a
  bounded thread pool whose exceptions surface. **[test]** (fake writer)
- wandb: output dir created and verified before `wandb.init` (`run_paths.py`),
  otherwise it silently falls back to `$TMPDIR`. (Pre-existing fix.) **[read]**

---

## Checklist for training

| # | Item | Status in training |
|---|---|---|
| 1 | Env layouts from keyed streams (`layout_ids`), keys logged | done in V0.99 (episode and training-time eval resets); keys are derivable from (seed, episode), not logged |
| 2 | HSR respawn draws from its own keyed stream | done in V0.99 |
| 3 | PPO minibatch shuffle on a dedicated generator | not done, but no longer coupled to HSR (HSR left the numpy global in V0.99) |
| 4 | No reseeds inside constructors (policy init) | not done (A2); since V0.99 they can no longer move a scene or the task order |
| 5 | Scheduler draws from a dedicated stream | done in V0.99 (state in `scheduler_state.json`) |
| 6 | Per-episode (or per-update) reseed of action sampling; generator states in checkpoints | not done (A6) |
| 7 | Rows carry full unit keys; one writer per derived file; derived files rebuilt, atomic | partial: `counters.json` is atomic; `eval_success.csv` is append-only (B1–B3) |
| 8 | `rollout_success.csv` rows are buffered until the PPO update that computes GAE — a crash loses rows since the last update, and a resume writes a new run dir; define the unit and truncate/stitch explicitly | not done; plot tools stitch resume chains with "child wins" **[read]** |
| 9 | Config content snapshotted with every checkpoint | done (B6) |
| 10 | Fingerprint checked on resume | not done |
