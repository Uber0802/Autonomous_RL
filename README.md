# CRONOS

CRONOS is a refactored robotic manipulation training benchmark designed for **non-episodic reinforcement learning** and **multi-task training**. It is built on the `RL4VLA` backbone with optimized modules from `AutoRL`.

## Features
- **YAML-driven experiment configs** — per-group objects, receptacles, task sequences, eval tasks, and fan-out settings in a single file.
- **Sub-group fan-out** — all tasks in a group run simultaneously on different env sub-groups, matching AutoRL's default multi-task gradient mixing.
- **Per-env rotation eval** — each env rotates through eval tasks across episodes; `num_eval_episode` controls sample count per task.
- **Per-group objects/backgrounds** — different groups can have different physical objects and visual overlays in the same training run.
- **Dual VLA support** — `--policy openvla|spatialvla` switches between OpenVLA-7B and SpatialVLA-4B adapters (7-token vs 3-token action sequences).
- **PPO or GRPO** — `--alg-name grpo` swaps the critic-free path in; grouping is selectable at three nesting levels (`batch` / `scene` / `task`), with `batch` verified bit-identical to AutoRL's `compute_returns_grpo`.
- **Orthogonal reset dimensions** — LSR (learned reset policy), HSR (respawn fallen actors), EER (gripper re-home), and Perturbation (the reset goal is sometimes a *different receptacle* instead of the table), each toggled independently.
- **Modular Environment** — decoupled `reset_strategy`, `reward_shaping`, `task_suite`, and `task_scheduler`.
- **Efficient Rollouts** — multi-task execution with GPU-parallelized ManiSkill environments, memory-mapped replay buffers.
- **Analysis-ready outputs** — per-segment CSVs recording both sides of every segment boundary, plus plot tools that read them directly and compare several runs from one config.

## Installation

The following steps assume a Linux environment with NVIDIA GPUs and Conda installed.

### 1. Get the repository

Clone (or download) and move into the repository directory:

```bash
git clone <repo-url>
cd <repo-directory>
```

### 2. Create the conda environment

Pick one of four envs depending on which policies you need and which GPU class you have:

2×2 matrix (LM stack × GPU class). Names are `cronos_<lm-stack>_<torch-channel>`:
`tf447` = transformers 4.47 / peft 0.14 (serves both VLAs), `tf440` = transformers
4.40.1 / peft 0.11.1 (OpenVLA only); `cu121` = Ampere/Ada/Hopper, `cu128` = Blackwell.

> Renamed from `cronos_env` / `cronos_env_blackwell` / `cronos_env_lite` /
> `cronos_env_lite_blackwell`, which did not say which stack they carried. Nothing
> reads the env name programmatically, so existing envs keep working — rename with
> `conda rename -n cronos_env cronos_tf447_cu121` when convenient.

| Env | `setup.sh` args | Stack | Policies | GPU class | OpenVLA-7B PPO peak | When to use |
|---|---|---|---|---|---|---|
| `cronos_tf447_cu121` | `setup.sh all` | `torch==2.5.1+cu121` + `transformers==4.47.0` + `peft==0.14.0` + `tokenizers==0.21.0` | OpenVLA **+** SpatialVLA | Ampere / Ada / Hopper (sm_80…sm_90) | ~55 GB | Hopper / A100 (both VLAs); Ada (SpatialVLA only — OpenVLA OOMs 48 GB) |
| **`cronos_tf447_cu128`** | `setup.sh all blackwell` | `torch==2.7.0+cu128` + `transformers==4.47.0` + `peft==0.14.0` + `tokenizers==0.21.0` | OpenVLA **+** SpatialVLA | Blackwell (sm_75…sm_120) | ~55 GB | Blackwell (both VLAs; 96 GB has plenty of room) |
| `cronos_tf440_cu121` | `setup.sh openvla_v01` | `torch==2.2.0+cu121` + `transformers==4.40.1` + `peft==0.11.1` + `tokenizers==0.19.1` | OpenVLA only | Ampere / Ada / Hopper (sm_50…sm_90) | ~40 GB | OpenVLA-only on Ada (48 GB) — fits where V0.4 OOMs; **bit-exact** to V0.1 baseline runs |
| `cronos_tf440_cu128` | `setup.sh openvla_v01 blackwell` | `torch==2.7.0+cu128` + `transformers==4.40.1` + `peft==0.11.1` + `tokenizers==0.19.1` | OpenVLA only | Blackwell (sm_75…sm_120) | ~45 GB | OpenVLA-only on Blackwell with V0.1 transformers ABI (not bit-exact to V0.1 cu121 — cu128 changes cuBLAS/attention kernels — but transformers/peft surface identical) |

```bash
# Pick the env that matches your GPU + policy needs, e.g.:
conda create -n cronos_tf447_cu128 -y python=3.10        # Blackwell + both VLAs
conda activate cronos_tf447_cu128

# or one of:
#   cronos_tf447_cu121 — both VLAs on Hopper/A100 (SpatialVLA-only on Ada)
#   cronos_tf440_cu121 — OpenVLA-only, Ada-friendly, V0.1-bit-exact
#   cronos_tf440_cu128 — OpenVLA-only on Blackwell, V0.1 transformers ABI
```

> **Bit-exact note:** Only `cronos_tf440_cu121` reproduces V0.1 baseline PPO logs bit-for-bit. `cronos_tf440_cu128` upgrades torch (no V0.1-era cu128 wheels exist), so it shares V0.1's transformers/peft *ABI* but not its exact cuBLAS/attention kernels. The dual-VLA (`tf447`) envs upgrade both torch and transformers, so their PPO logs drift ~10⁻² from V0.1 in the first 1000 minibatches and converge to <0.2% by PPO step 100 — algorithmically correct, numerically different. For bit-exact ablations, run the baseline arm in the **same env** as the test arm. Multi-seed mean±std comparisons are unaffected (drift ≪ seed-to-seed variance).

### 3. Run the setup script

```bash
cd CRONOS
chmod +x *.sh

# setup.sh [policy] [gpu]
#   [policy]: openvla | spatialvla | all | openvla_v01    (default: all)
#   [gpu]:    default | blackwell                          (default: default)

# Four canonical invocations (one per 2x2 env above):
./setup.sh all                       # → cronos_tf447_cu121 (cu121, Ada/Hopper, both VLAs)
./setup.sh all blackwell             # → cronos_tf447_cu128 (cu128, Blackwell, both VLAs)
./setup.sh openvla_v01               # → cronos_tf440_cu121 (cu121, Ada/Hopper, OpenVLA, V0.1-bit-exact)
./setup.sh openvla_v01 blackwell     # → cronos_tf440_cu128 (cu128, Blackwell, OpenVLA)

# Subset modes (dual-VLA stack with only one VLA pillar installed):
./setup.sh openvla                   # dual-VLA, OpenVLA pillar only (skips SpatialVLA)
./setup.sh spatialvla blackwell      # dual-VLA, SpatialVLA pillar only, Blackwell
```

> **Memory budget — pick the right env for your GPU.** The dual-VLA stack lifts OpenVLA-7B PPO peak memory from ~40 GB → ~55 GB (`transformers==4.47` HybridCache + `peft==0.14` fast path + newer torch caching), which **does not fit on Ada-class GPUs (48 GB)**. If you only need OpenVLA on Ada, the lightweight stack (`torch==2.2.0+cu121` + `transformers==4.40.1`, ~40 GB peak) still fits 1 OpenVLA-7B PPO on a 48 GB Ada. See [Lightweight env](#6-optional-lightweight-openvla-only-env-for-ada-class-gpus).
>
> The `tf440` split is a workaround for that regression, not a design goal. `tools/bench_rollout.py` measures throughput and peak memory per stack with a phase breakdown, so the ~15 GB can be attributed and ideally removed — at which point `tf440` retires and the matrix collapses to one env per torch channel. The memory figures in the table above are historical reports, not `bench_rollout.py` output; re-measure on your own hardware before relying on them.

`setup.sh` installs CRONOS plus its sibling pillars (`SimplerEnv`, `ManiSkill`, `openvla`, `SpatialVLA`), which must already be present in the same parent directory as `CRONOS/`. The script `cd`s to its own directory before each editable install, so the resolved paths are unambiguous regardless of the caller's `cwd`. A post-install Python sanity check verifies `torch.cuda`, `tensorflow_datasets`, `OpenVLAPolicy.act_token_len`, and (when present) `SpatialVLAPolicy`.

**Hotfix** for the `runtime_version` ImportError (protobuf 4.x vs 5.x mismatch on `import tensorflow_datasets`):
```bash
pip install "tensorflow-metadata<1.21" "protobuf>=3.20,<5"
```
(setup.sh pins these permanently so this won't recur on fresh envs.)

### 4. (Optional) Ubuntu 22.04 prerequisite

If the Vulkan driver is missing on Ubuntu 22.04 (SAPIEN will warn about a missing ICD file at startup), install:

```bash
sudo apt-get update
sudo apt-get install -y libglvnd-dev
```

### 5. (Optional) Blackwell GPU support

Blackwell cards (RTX PRO 6000, RTX 5090, B200) ship `sm_120` SASS, which only `+cu128` torch wheels include. Pass `blackwell` as `setup.sh`'s 2nd arg to install the Blackwell-compatible variant of either LM stack:

```bash
./setup.sh all blackwell             # cronos_tf447_cu128 — both VLAs
./setup.sh openvla_v01 blackwell     # cronos_tf440_cu128 — OpenVLA-only, V0.1 transformers ABI
```

Both variants pin `torch==2.7.0+cu128` (the lowest stable cu128 build with sm_120). What differs is the LM stack:

| Env | Transformers/peft/tokenizers | OpenVLA peak | Bit-exact to V0.1 cu121? |
|---|---|---|---|
| `cronos_tf447_cu128` | V0.4 (4.47 / 0.14 / 0.21) | ~55 GB | No — also drifts from V0.1 |
| `cronos_tf440_cu128` | V0.1 (4.40.1 / 0.11.1 / 0.19.1) | ~45 GB | No — cu128 changes cuBLAS/attention kernels, but transformers ABI matches V0.1 |

**No torch build is simultaneously V0.1-era *and* Blackwell-compatible.** The cu128 channel does not ship `torch==2.2.0` (cu128 wheels start at torch 2.7), and torch 2.2.0+cu121 has no `sm_120` SASS or PTX. Bit-exact V0.1 baseline replication is therefore Ada/Hopper-only by physics of GPU release dates.

*Reproducibility note:* PTX is JIT-compiled on first CUDA op on a new arch, so SASS may differ slightly between a Blackwell run and a Hopper run even within the same env. Training curves on Blackwell are *statistically* equivalent to Hopper, not bit-exact.

### 6. (Optional) Lightweight OpenVLA-only env for Ada-class GPUs

For Ada-class GPUs (L40S, RTX 6000 Ada, A6000 — 48 GB), the dual-VLA stack's ~55 GB OpenVLA-7B PPO peak does **not** fit. The lightweight `openvla_v01` mode pins the lightweight stack (`transformers==4.40.1` + `peft==0.11.1` + `tokenizers==0.19.1`) and — on `cu121` — pins `torch==2.2.0` to match V0.1 exactly. OpenVLA-7B PPO peak stays at ~40 GB, fitting one PPO on a 48 GB Ada with headroom.

```bash
conda create -n cronos_tf440_cu121 -y python=3.10
conda activate cronos_tf440_cu121
cd Benchmark/CRONOS
./setup.sh openvla_v01
```

Tradeoffs:
- ✅ Fits on Ada (48 GB) — restores parity with V0.1's running memory profile.
- ✅ Numerically bit-exact against V0.1 baseline runs (same cuBLAS GEMM tile order + attention kernels).
- ❌ Cannot run `--policy spatialvla` — transformers ≥ 4.43 needed for the `HybridCache` import in SpatialVLA's `model/modeling_gemma2.py`. `setup.sh openvla_v01` skips the `../SpatialVLA` editable install entirely; the policy's lazy import in [main.py:270-271](CRONOS/main.py#L270-L271) and [eval_only.py:140](CRONOS/eval_only.py#L140) is gated by `--policy spatialvla` so it never fires under OpenVLA-only runs.
- ❌ Will not run on Blackwell as-is — pass `blackwell` as the 2nd arg to install the Blackwell variant: `./setup.sh openvla_v01 blackwell` produces `cronos_tf440_cu128` (lightweight stack on `torch==2.7.0+cu128`; loses cu121 bit-exactness but keeps V0.1 transformers ABI).

How `setup.sh` picks the install: the 1st positional arg picks the LM stack + which sibling pillars get installed, the 2nd picks the torch wheel channel. See the header comment in `setup.sh` for the full pin rationale and the 4-env recommended workflows.

## Quick Start

### Training
```bash
# 9 positional args
bash scripts/train.sh <mode> [seed] [cuda] [reset] [config] [vla] [eer] [algo] [perturb]
#                      │      │      │      │       │        │     │     │      └─ off (default) | recep | mixed
#                      │      │      │      │       │        │     │     └─ ppo (default) | grpo | grpo-scene | grpo-task
#                      │      │      │      │       │        │     └─ on (default) | off — End-Effector Reset
#                      │      │      │      │       │        └─ openvla (default) | spatialvla
#                      │      │      │      │       └─ YAML config filename (default: four_group_sequential_2x2)
#                      │      │      │      └─ normal | LSR | HSR | LSR+HSR | noep (default: normal)
#                      │      │      └─ GPU id (default: 3)
#                      │      └─ seed (default: 0)
#                      └─ horizon tag: t80a..t2560c (12 horizons × 3 segment-len variants)
```

Output directory: defaults to `./$RUN_TAG`, created before launch and passed as an
**absolute** `--wandb-dir`. Override with `RUN_OUT_DIR=/data/runs/my-run`. Passing a
relative or not-yet-existing directory used to make wandb silently redirect the whole
run — every CSV, checkpoint and video — into `$TMPDIR`; `run_paths.py` now creates and
validates the directory up front and fails loudly if wandb ignores it.

Examples:
```bash
# OpenVLA, T320 segment 'a', seed 0, GPU 3, normal reset
bash scripts/train.sh t320a 0 3 normal four_group_sequential_2x2

# SpatialVLA, T1280 segment 'b', seed 1, GPU 2, non-episodic (LSR+HSR + reset_mode=none)
bash scripts/train.sh t1280b 1 2 noep four_group_sequential_2x2 spatialvla

# Same, but with the end effector never repositioned between segments
bash scripts/train.sh t1280b 1 2 noep four_group_sequential_2x2 spatialvla off

# GRPO instead of PPO — AutoRL-compatible grouping (one group per segment)
bash scripts/train.sh t320a 0 3 normal four_group_sequential_2x2 openvla on grpo

# GRPO grouped per scene (segment × YAML group — same objects/receptacles/background)
bash scripts/train.sh t320a 0 3 normal four_group_sequential_2x2 openvla on grpo-scene

# GRPO grouped per task (segment × fan-out sub-block — narrowest, most apples-to-apples)
bash scripts/train.sh t320a 0 3 normal four_group_sequential_2x2 openvla on grpo-task

# Perturbation: the LSR reset goal is sometimes "put X on the OTHER receptacle"
# instead of always "put X on table" — widens the forward policy's start states
bash scripts/train.sh t320a 0 3 noep four_group_sequential_2x2 openvla on ppo mixed

# Perturbation composes with GRPO (orthogonal dimensions)
bash scripts/train.sh t320a 0 3 noep four_group_sequential_2x2 openvla on grpo-task recep

# Any GRPO mode with the std term overridden (tagged, so it lands in its own dir)
GRPO_STD_SCOPE=none bash scripts/train.sh t320a 0 3 normal four_group_sequential_2x2 openvla on grpo-task
```

`RUN_TAG` carries the VLA tag (`CRONOS-openvla-<config>-<horizon>-<reset>-seed<N>`), so OpenVLA and SpatialVLA runs land in separate output dirs.

**Reset-mode legend:**

| mode | CLI flags added | Meaning |
|---|---|---|
| `normal` | (nothing) | hard `env.reset()` every episode |
| `LSR` | `--enable-backward --backward-interval 1` | learn the backward policy (put X back) alternating with forward task switches |
| `HSR` | `--reset-unsuitable` | respawn fallen / out-of-workspace actors at every task boundary |
| `LSR+HSR` | LSR + HSR | backward learning + soft respawn |
| `noep` | LSR+HSR + `--reset-mode none` | non-episodic continuity (no inter-episode hard reset) |

**Perturbation** — the 9th positional arg, orthogonal to the reset modes but requiring one that includes LSR:

| perturb | CLI flags added | LSR reset goal |
|---|---|---|
| `off` (default) | (nothing) | always `put X on table` |
| `recep` | `--backward-goal recep` | always another receptacle, chosen != the forward task's |
| `mixed` | `--backward-goal mixed --backward-recep-prob P` | per-env draw between the two; `P` via `PERTURB_RECEP_PROB` |

Both goals reuse tasks that already exist in the pool — the receptacle variant is
literally an existing `put <obj> on <recep>` pair — so there is no new task string
and no new reward term. Swapping the env's target receptacle makes its own
`success` predicate and language instruction follow, which is why that variant is
scored by the *forward* reward branch rather than by `src_on_table`. Motivation:
a reset policy that always returns the object to one canonical state keeps the
forward policy's start-state distribution narrow (arXiv:2004.12570 §4.1).
`off` emits no flag, no tag, and does not draw from the RNG, so it is numerically
identical to before the option existed.

**EER (End-Effector Reset)** — the 7th positional arg, orthogonal to all five reset modes above:

| eer | CLI flag added | Meaning |
|---|---|---|
| `on` (default) | (nothing — `--reset-robot` is already the `main.py` default) | gripper returns to its initial pose at every segment boundary, in every reset mode |
| `off` | `--no-reset-robot` | fully continuous arm — nothing repositions the end effector between segments |

`eer=off` appends `-noEER` to `RUN_TAG` so it lands in its own output directory;
`eer=on` emits a command line byte-identical to before the option existed, so
prior runs, resume paths and wandb dirs are unaffected.

> `reset_robot()` is also the only thing that zeroes ManiSkill's `_elapsed_steps`
> on the training path, and `truncated` feeds the PPO buffer's masks. With EER
> off, the training loop calls `CronosWrapper.begin_segment()` instead, which
> reopens the accounting window without touching the arm — otherwise every step
> after the first segment would report truncated, masks would go to zero and GAE
> would degenerate to `returns = reward` with no bootstrapping. This is the same
> mechanism described in [`doc/eval_audit.md`](CRONOS/doc/eval_audit.md).

Key training flags:
| Flag | Default | Description |
|---|---|---|
| `--config-path` | required | YAML experiment config |
| `--policy` | `openvla` | `openvla` or `spatialvla` |
| `--alg-name` | `ppo` | `ppo` (actor-critic + GAE) or `grpo` (critic-free) |
| `--grpo-group-scope` | `batch` | GRPO only. What counts as one group: all three are per-segment. `batch` (whole segment — the statistic AutoRL uses, bit-identical) \| `scene` (segment × YAML group) \| `task` (segment × fan-out sub-block). Sizes for `four_group_sequential_2x2`: 64 / 16 / 4 |
| `--grpo-std-scope` | `group` | GRPO only. Divide group-centred rewards by `group` / `global` std, or `none`. See [`doc/grpo_autorl.md` §9](CRONOS/doc/grpo_autorl.md) |
| `--alg-grpo-fix` | on | GRPO only. Compute reward statistics from non-zero rewards only (AutoRL's `alg_grpo_fix`) |
| `--wandb-dir` | `""` | Run output root. Created and validated before `wandb.init`; a run that cannot land here fails instead of silently going to `$TMPDIR` |
| `--num-envs` | 64 | Total parallel environments |
| `--segment-len` | 80 | Steps per segment (AutoRL: 80) |
| `--ppo-update-len` | 80 | Steps between PPO updates |
| `--eval-interval` | 4 | Eval every N episodes |
| `--num-eval-episode` | 4 | Episodes per eval round |
| `--eval-at-start` | false | Run eval before first training episode |
| `--enable-backward` / `--backward-interval N` | off | LSR — backward policy alternating with forward at step interval N |
| `--reset-unsuitable` | off | HSR — respawn fallen/out-of-workspace actors at task boundary |
| `--hsr-reset-scope` | `per_env` | `per_env` (full-env reset of flagged envs) \| `per_actor` (single-actor) \| `all` |
| `--unsuitable-detector` | `low_z` | `low_z` (`z < 0.7`) or `workspace` (configurable xyz AABB via YAML) |
| `--reset-mode` | `per_episode` | `per_episode` \| `none` (non-episodic) |
| `--reset-robot` / `--no-reset-robot` | on | EER — return the gripper to its initial pose at every segment boundary |
| `--backward-goal` | `table` | LSR reset goal (perturbation). `table` = "put X on table" (unchanged) \| `recep` = another receptacle, != the forward task's \| `mixed` = per-env draw. Requires `--enable-backward` |
| `--backward-recep-prob` | 0.5 | `mixed` only: P(receptacle variant) per env per reset segment |
| `--segment-pose-phase` | `both` | `start` (state each segment begins from, after that boundary's resets) \| `end` (steady state the policy produced, before them) \| `both` |
| `--record-segment-pose` | **on** | Dump every object/receptacle slot + gripper pose (position + quaternion) at each segment end to `glob/segment_pose.csv`; disable with `--no-record-segment-pose` |

### Training-time outputs

Written to the run's `glob/` on every run; full column specs in
[`doc/data_schemas.md`](CRONOS/doc/data_schemas.md).

| File | Contents |
|---|---|
| `rollout_success.csv` | one row per (episode, segment, env): `success` / grasp at segment end, `reward_sum`, `return_discounted`, `return_gae`, `value_mean`, `advantage_mean`, plus a `direction` column marking forward vs backward segments |
| `segment_pose.csv` | one row per (episode, segment, env, actor) with full `pq`; only with `--record-segment-pose` |
| `eval_success.csv` | aggregate per (eval point, group, task) |

`rollout_success.csv` uses the **same** `success` definition as eval — the value
at the segment's final step — so rollout and eval curves are directly
comparable; they differ only in when they are sampled. Under LSR/noep, filter
`direction == 'forward'` before aggregating: the env's success predicate is
always the forward one, so backward segments score 0 by construction.

### Evaluation (standalone)

`eval_only.py` runs **AutoRL-style broadcast eval** by default: every env in the batch runs the same `(object, receptacle)` for one `segment_len` rollout, no fan-out. Two modes:

- `--eval-mode sequential` (default): runs `--eval-sequences N` orderings of the eval tasks. The first ordering is the training task order (sequence 0, AutoRL `eval_training_seq=True` convention); subsequent orderings are random permutations. Within each ordering, tasks switch one-by-one without env reset (matches AutoRL `render_seq`).
- `--eval-mode single`: runs each task in `eval_tasks` once independently, all envs on the same task per pass (matches AutoRL `render`).

**Note:** the fan-out rotation eval (per-env rotation) is not used for standalone eval — it lives only inside `train()` for training-time eval. Use `--eval-at-start` if you want the rotation eval against a checkpoint loaded by main.py.

**Wrapper script** (recommended — sets the right env vars):
```bash
# Args: <checkpoint_dir> [config] [cuda] [num_eval_episode]
bash scripts/eval.sh /path/to/glob/episode_0128
bash scripts/eval.sh /path/to/glob/episode_0128 configs/two_group_sequential_2x2.yaml 0,1 4
```

**Direct invocation**:
```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0,1 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python eval_only.py \
  --name CRONOS-eval --seed 0 \
  --env-id PickPlaceNxM-v1 \
  --vla-path openvla/openvla-7b --vla-unnorm-key bridge_orig \
  --config-path configs/two_group_sequential_2x2.yaml \
  --vla-load-path /path/to/glob/episode_0128 \
  --segment-len 80 --num-eval-episode 4 \
  --record-video
```

`--num-envs` is **not** needed on the CLI — it's derived from per-group `num_envs` in the config.

**Outputs** (under a fresh `wandb/offline-run-<timestamp>-<id>/glob/`):

| File | Contents |
|---|---|
| `eval_success.csv` | one row per (group, task, eval_kind) with `success`, `grasp`, `obj_grasped` |
| `eval_per_trial.csv` | one row per (sequence, task, env) — the pairing key `tools/mcnemar_pair.py` needs. Carries both scoring semantics: `success` (independent) and `success_chained` (cumulative AND along the sequence) |
| `eval_report.txt` | human-readable per-eval summary (rotation table + per-task means, both semantics) |
| `eval_videos/{kind}/eval_ep{M}/env{i}.mp4` | one mp4 per env per eval episode (if `--record-video`) |
| `run_config.json` / `run_config.yaml` | exact args + resolved YAML at run time |

Common eval flags:
| Flag | Default | Description |
|---|---|---|
| `--config-path` | required | YAML experiment config (typically one from `configs/eval/`) |
| `--vla-load-path` | required | Checkpoint dir (the `episode_XXXX/` from a training run, or `TestCheckpoint/seed0/`) |
| `--eval-mode` | `sequential` | `sequential` (AutoRL `render_seq`) or `single` (AutoRL `render`) |
| `--eval-sequences` | 5 | Sequential mode: training-order + N-1 random permutations |
| `--segment-len` | 80 | Steps per task rollout |
| `--record-video` | true | Write mp4s under `glob/eval_videos/{prefix}/` |
| `--vla-temperature-eval` | 0.6 | Sampling temperature for the policy |

### Visualization

CRONOS ships two complementary plotting tools — one per-run live dashboard, one cross-run aggregator.

#### Per-run live dashboard — `tools/plot_run_trends.py`

Refreshes a 4-panel `trends.png` directly inside a running training run's `glob/` dir. Pulls per-episode aggregates (`approx_kl`, `clip_fraction`, `explained_var`) from wandb cloud history and per-eval-point success/grasp from the local `eval_success.csv`. Read-only on the training process; safe to call mid-run between eval points.

```bash
# Render trends.png for one in-progress run; also drops a copy at <run-dir>/trends.png
python tools/plot_run_trends.py \
  --run-dir wandb/run-20260618_103000-abcd1234/files/glob \
  --max-episodes 32 \
  --out reports/figures/2026-06-18_t320a-trends.png
```

Layout (each panel is one PPO health signal):

| Panel | Metric | What to watch |
|---|---|---|
| (0,0) Task performance | rollout success/grasp (5-ep MA) + eval ID/OOD per eval point | task-side learning curve |
| (0,1) Policy drift | per-ep mean(`approx_kl`) | should stay ≈ const; spikes ⇒ unstable ratio |
| (1,0) LoRA trust region | per-ep mean(`clip_fraction`) | "trust region pulse" — fraction of minibatches outside the PPO clip band |
| (1,1) Value head | per-ep mean(`explained_var`) | critic quality; flat 0 ⇒ value head not learning |

For **resumed** runs (`--resume-from`), pass each prior wandb run id and prior `eval_success.csv` path so the dashboard covers ep 1…ep<max> in one image:

```bash
python tools/plot_run_trends.py \
  --run-dir wandb/run-<current>/files/glob \
  --max-episodes 32 --out reports/figures/<date>_trends.png \
  --prior-run-id <parent_run_id> --prior-eval-csv /path/to/parent/glob/eval_success.csv \
  --prior-run-id <grandparent_run_id> --prior-eval-csv /path/to/grandparent/glob/eval_success.csv
```

(Auto-renders a sibling `<stem>-per_task.png` per-task breakdown next to `--out`.)

#### Per-segment training curves — `tools/plot_rollout_success.py`

Per-80-step success rate straight from the run's `rollout_success.csv`. That file
holds one row per (episode, segment, env) written at every `task_len` boundary,
so "per-80" is its native granularity — each point is the mean over one
segment's `num_envs` rows, with no resampling. Needs no wandb access.

```bash
python tools/plot_rollout_success.py --run-dir <RUN_OUT_DIR>/wandb/run-*/glob
```

| Flag | Default | Description |
|---|---|---|
| `--config` | — | JSON with several groups of runs; one curve per group, mean ± 1 std band across that group's series. See [Comparing runs](#comparing-several-runs) |
| `--direction` | `forward` | `forward` \| `backward` \| `backward_recep` \| `all`. Reset segments score `success` against a different goal, so mixing them in reads as a ~50% collapse that is pure alternation artifact — see [`doc/data_schemas.md`](CRONOS/doc/data_schemas.md). `all` draws one series per direction. |
| `--by` | `none` | Add a second panel split by `task` \| `group` \| `obj` \| `recep` |
| `--x-axis` | `total_steps` | `total_steps` \| `segment` \| `episode` |
| `--smooth` | 5 | Rolling-mean window in segments (1 disables) |

The top panel overlays success, `consecutive_grasp` and `is_src_obj_grasped`;
the success-vs-grasp gap is the placement-collapse diagnostic. Empty cells (an
env that did not report at a boundary) are read as NaN and excluded from the
means rather than as zeros.

#### Per-segment position scatter — `tools/plot_segment_positions.py`

Actor positions from `segment_pose.csv`. Each boundary is recorded twice — once
**before** its HSR/EER resets (`phase=end`, the steady state the policy produced)
and once **after** (`phase=start`, the initial state the next segment begins
from, and after a full `env.reset()` at an episode boundary). `--phase` defaults
to `start`, which is the distribution the forward policy actually faces.

```bash
python tools/plot_segment_positions.py --run-dir <RUN_OUT_DIR>/wandb/run-*/glob
```

One column per `actor_kind` (`obj` / `recep` / `gripper`); row 1 is an xy scatter
coloured by episode (so drift over training is visible), row 2 is a `pz`
histogram with the `low_z = 0.7` detector threshold marked and the fraction below
it in the panel title — a direct read on how often objects are ending up off the
table.

| Flag | Description |
|---|---|
| `--config` | JSON with several groups of runs; one column per group. See [Comparing runs](#comparing-several-runs) |
| `--phase` | `start` (default) — the state each segment *begins* from, after that boundary's HSR/EER resets and after `env.reset()` at an episode boundary. `end` — the steady state the policy produced, before them. `all` — both |
| `--actor-kind` / `--slot` / `--model` / `--task` | Narrow to one actor class, logical slot, model-name substring, or task substring |
| `--segment` / `--episode-range LO:HI` / `--last-episodes N` | Narrow in time |
| `--forward-only` | Join `rollout_success.csv` on (episode, segment, env) and keep only forward segments — worth using under LSR / noep, where half the segment ends are reset-goal states |
| `--hexbin` | Density hexbin instead of the episode-coloured scatter |
| `--workspace=X0,X1,Y0,Y1` | Overlay a rectangle, e.g. `workspace_aabb` bounds being validated. Use the `=` form — the bounds are negative and argparse would read them as a flag |

Hidden slots (a group declaring fewer objects than the batch-wide N) are written
as NaN by design and are dropped, with the count reported.

#### Comparing several runs — `--config`

Both per-segment tools take a `--config` JSON describing several experiment
groups. A group is one curve (success) or one column (positions); each entry in
its `runs` list is one series, typically a seed. **A `runs` entry that is itself
a list is a resume chain** — those run dirs are stitched into one continuous
series, with the child winning at any overlapping `total_steps`.

```bash
python tools/plot_rollout_success.py   --config scripts/plot_runs_example.json
python tools/plot_segment_positions.py --config scripts/plot_runs_example.json --actor-kind obj
```

```json
{
  "out_dir": "reports/figures/2026-08-26",
  "name": "perturb_ablation",
  "groups": [
    { "label": "noep baseline",
      "runs": [ ["/data/runs/T320-seed0/.../glob", "/data/runs/T1280-seed0/.../glob"],
                "/data/runs/T320-seed1/.../glob" ] },
    { "label": "noep + PTBmixed",
      "runs": [ "/data/runs/T320-PTBmixed0.5-seed0/.../glob" ] }
  ]
}
```

Schema and full docs in [`tools/plot_common.py`](CRONOS/tools/plot_common.py);
a ready-to-edit copy is [`scripts/plot_runs_example.json`](CRONOS/scripts/plot_runs_example.json).
Top-level keys starting with `_` are ignored, so the example carries its own notes.

##### Runs recorded before the `phase` split

An older run has no `phase=start` rows, and they cannot be recovered — the env
draws initial poses with `torch.randint` on the global CUDA generator (which the
VLA's action sampling also consumes) and HSR with `np.random.choice` (which the
PPO minibatch shuffle also consumes), and neither index is logged. A same-seed
replay would have to reproduce the whole training bit-for-bit.

The *distribution* those draws came from is recoverable, though: the sampler is
uniform over `xyz_configs`, a deterministic table `envs/suite.py` builds from the
(N, M) preset with no randomness. `plot_segment_positions.py` therefore
reconstructs the start cloud by drawing uniformly from that same table **as many
times as the run actually reset** — `total_resets` counts exactly one draw per
per-env respawn, so a T80 run (128 episodes × 64 envs = 8,192 draws) and a T2560
one (4 × 64 = 256) produce clouds of the right relative density instead of a
misleading uniform one.

Synthetic points are drawn as black `×` and counted separately in the panel
title; they are never merged into the recorded cloud. `--no-synth` skips such
runs instead, `--synth-seed` makes the draw reproducible.

For the shipped 2×2 preset that table is 432 ordered configs = 18 distinct
four-point geometries × 4! slot permutations, occupying just 16 xy positions
(a 4×4 corner sub-grid) — the workspace is 0.15 × 0.15 m and the spacing
constraint is 0.12 m, so the points are pushed into the corners.

#### Cross-run aggregator — `scripts/plot.py`

Reads multiple `eval_success.csv` files (one per seed × config × condition), aggregates mean ± std, and writes 4 main PNGs (ID/OOD × Steps/Resets) plus 2 gap PNGs (success vs grasp).

```bash
# 1. Edit scripts/plot_config.json — list run groups (label → list of CSV paths)
# 2. Run the aggregator
python scripts/plot.py --config scripts/plot_config.json
```

`plot_config.json` schema (one entry per logical comparison curve):

```json
{
  "out_dir": "reports/aggregated/2026-06-18",
  "name": "four_group_T320_vs_T1280",
  "groups": [
    {
      "label": "T320 normal (3 seeds)",
      "csv_paths": [
        "/path/to/CRONOS-openvla-…-T320-normal-seed0/glob/eval_success.csv",
        "/path/to/CRONOS-openvla-…-T320-normal-seed1/glob/eval_success.csv",
        "/path/to/CRONOS-openvla-…-T320-normal-seed2/glob/eval_success.csv"
      ]
    },
    {
      "label": "T1280 noep (resumed from T320 seed1)",
      "csv_paths": [
        [
          "/path/to/CRONOS-openvla-…-T320-normal-seed1/glob/eval_success.csv",
          "/path/to/CRONOS-openvla-…-T1280-noep-seed1/glob/eval_success.csv"
        ]
      ]
    }
  ]
}
```

A `csv_paths` entry that is a **list** (chain) is the **resume chain**: parent CSV + child CSV; the aggregator dedupes overlapping `(total_steps, eval_kind, group, task)` rows and keeps the child at the seam. Adding a new run = append a path; no code changes.

Outputs:

| File | Contents |
|---|---|
| `aggregated.csv` | long-form per-group/eval_kind/x_axis mean ± std |
| `summary.csv` | final-value mean ± std at the rightmost eval per group × eval_kind |
| `<name>_<eval_kind>_<x_axis>.png` | 4 main curves (ID/OOD × total_steps/total_resets) |
| `<name>_gap_<eval_kind>.png` | success-vs-grasp overlay (placement-collapse diagnostic) |

`plot.py` requires `pandas`, `numpy`, `matplotlib`; pinned versions are in `scripts/requirements_plot.txt` and pulled in by `setup.sh` automatically.

### Training config → eval config mapping

Each training config under `configs/` has a matching eval config under `configs/eval/` with smaller `num_envs` for faster eval. Eval configs use `task_order: sequence_random` so eval orderings match AutoRL's random-permutation default.

| Training config | Eval config | Notes |
|---|---|---|
| `one_group_half_train_2x2.yaml` | `eval/one_group_half_train_2x2.yaml` | Generalization: trained on 2 of 4 tasks, eval on all 4 |
| `one_group_pure_random_2x2.yaml` | `eval/one_group_2x2.yaml` | Default 2x2 single-group eval |
| `one_group_seq_random_2x2.yaml` | `eval/one_group_2x2.yaml` | Same eval as pure_random — neither has a canonical training order |
| `one_group_sequential_3x3.yaml` | `eval/one_group_3x3.yaml` | 9 tasks; sequence 0 = auto-generated NxM order from training config |
| `two_group_sequential_2x2.yaml` | `eval/two_group_2x2.yaml` | Multi-group standalone eval — see caveat in the file |

## YAML Config Format

Configs live in `configs/`. A single YAML file fully describes an experiment.

### Top-level fields

| Field | Type | Default | Description |
|---|---|---|---|
| `cronos_version` | string | — | Human annotation only. The loader accepts the key but never reads or validates it, so it is not a compatibility gate. The *code* version is stamped into each run's `run_config.json` from [`version.py`](CRONOS/version.py). |
| `task_order` | string | `sequential` | `sequential`, `pure_random`, or `sequence_random` |
| `fan_out` | bool | `true` | All tasks run simultaneously within each group (AutoRL default) |

### Per-group fields

Each entry in the `groups:` list defines one group:

| Field | Type | Required | Description |
|---|---|---|---|
| `name` | string | yes | Group identifier (used in logs and CSV) |
| `num_envs` | int | yes | Number of parallel envs for this group |
| `obj` | list[int] | yes | 1-based object indices |
| `recep` | list[int] | yes | 1-based receptacle indices |
| `table` | string | no | Stage mesh (default: `flat_table`) |
| `background` | int/string | no | Overlay index (int) or `"default"` (episode-based) |
| `task_sequence` | list[str] | no | Training task list using symbolic refs (`obj1`, `recep2`). Auto-generated if omitted. |
| `eval_tasks` | list[str] | no | Eval task list. Defaults to unique tasks from `task_sequence`. |

### Object index reference (1-based)

| Index | Name | Index | Name | Index | Name |
|---|---|---|---|---|---|
| 1 | carrot | 10 | toy bear | 19 | nonstop can |
| 2 | kitchen shovel | 11 | fast food cup | 20 | potato |
| 3 | bread | 12 | plant | 21 | baguette |
| 4 | plastic bottle | 13 | banana | 22 | champagne glass |
| 5 | 7up can | 14 | hamburger | 23 | kitchen spoon |
| 6 | zuchinni | 15 | golf ball | 24 | onion |
| 7 | ketchup bottle | 16 | BBQ sauce | 25 | cup |
| 8 | watering can | 17 | travel cup | | |
| 9 | pipe | 18 | pepper | | |

### Receptacle index reference (1-based)

| Index | Name | Index | Name | Index | Name |
|---|---|---|---|---|---|
| 1 | yellow plate | 7 | tomato slice | 13 | cutting board |
| 2 | cloth | 8 | pizza | 14 | chess board |
| 3 | carpet | 9 | flat bowl | 15 | manhole cover |
| 4 | newspaper | 10 | gramophone disk | 16 | envelope |
| 5 | sheet metal | 11 | frying pan | 17 | notepad |
| 6 | drawing tablet | 12 | mouse pad | | |

### Symbolic task format

Tasks use `put obj{N} on recep{M}` where N/M are 1-based indices into the group's `obj`/`recep` lists. Resolved at load time to real names (e.g. `put ketchup bottle on yellow_plate`).

### Divisibility constraints

The config loader validates:

| Rule | Constraint | Purpose |
|---|---|---|
| V22 | `group_envs >= n_unique_train_tasks` | Enough envs for fan-out |
| V23 | `group_envs % n_unique_train_tasks == 0` | Even split for training |
| V24 | `group_envs % n_eval_tasks == 0` | Even split for eval rotation |
| V25 | `n_eval_tasks % n_unique_train_tasks == 0` | Eval aligns with train sub-groups |

### Example configs

| Config | Description |
|---|---|
| `two_group_sequential_2x2.yaml` | 2 groups, different obj/recep/background, sequential |
| `one_group_sequential_3x3.yaml` | 1 group, 3x3 (9 tasks), sequential |
| `one_group_half_train_2x2.yaml` | 1 group, 2 train tasks, 4 eval tasks (generalization test) |
| `one_group_seq_random_2x2.yaml` | 1 group, sequence_random order |
| `one_group_pure_random_2x2.yaml` | 1 group, pure_random order |

### Minimal single-group config

```yaml
cronos_version: V0.4
task_order: sequential

groups:
  - name: "default"
    num_envs: 64
    obj: [7, 2]
    recep: [1, 2]
    # task_sequence and eval_tasks auto-generated: all 4 NxM combinations
```

### Two-group config with different objects and receptacles

```yaml
cronos_version: V0.4
task_order: sequential

groups:
  - name: "group_A"
    num_envs: 32
    obj: [7, 2]              # ketchup_bottle, kitchen_shovel
    recep: [1, 2]            # yellow_plate, cloth
    background: 0
    task_sequence:
      - "put obj1 on recep1"
      - "put obj1 on recep2"
      - "put obj2 on recep1"
      - "put obj2 on recep2"

  - name: "group_B"
    num_envs: 32
    obj: [3, 5]              # bread, 7up_can
    recep: [4, 3]            # newspaper, carpet
    background: 1
    task_sequence:
      - "put obj1 on recep1"
      - "put obj1 on recep2"
      - "put obj2 on recep1"
      - "put obj2 on recep2"
```

## Eval Design

CRONOS has two distinct eval semantics for two different use-cases.

### Training-time eval — per-env rotation

Used inside `main.py train()` (eval-at-start + periodic mid-training eval). Fan-out across envs: each env independently rotates through the eval tasks so all eval tasks are scored every eval round without idle envs.

Per-env rotation formula:
```
task_for_env_i_at_episode_e = eval_tasks[(i % n_eval_tasks + e) % n_eval_tasks]
```

Samples per task = `num_eval_episode * group_envs / n_eval_tasks`.

Divisibility constraints (validated at config load):
- `group_envs % n_eval_tasks == 0`
- `n_eval_tasks % n_train_tasks == 0`

In non-episodic mode (`reset_mode=none`), the training scene state is snapshotted before each mid-training eval and restored after, so eval's `env.reset` calls don't break the live simulation continuity that non-episodic training relies on.

### Standalone eval — AutoRL-style broadcast

Used by `eval_only.py` and `main.py --eval-single` / `--eval-sequential`. All envs run the same `(object, receptacle)` for one `segment_len`-step rollout (no fan-out). Two modes, configurable via `--eval-mode`:

| Mode | Behavior | AutoRL analog |
|---|---|---|
| `sequential` (default) | `--eval-sequences N` orderings of the eval tasks; sequence 0 is the training task order, sequences 1..N-1 are random permutations. Tasks within an ordering switch one-by-one without env reset. | `render_seq(eval_training_seq=True)` |
| `single` | Iterate `eval_tasks` once; all envs run the same task per pass. | `render` |

This is the eval mode used by `scripts/eval.sh` and the per-training eval configs under `configs/eval/`.

#### Sequential scoring: independent vs chained

A sequential eval emits **both** semantics per trial, so one run answers both
questions and no re-run is needed to switch lens:

| Column in `eval_per_trial.csv` | Semantics |
|---|---|
| `success` | **independent** — this task judged on its own, whatever happened earlier in the sequence. AutoRL's semantics. |
| `success_chained` | **chained** — cumulative AND along `task_idx` within one `(obj_set, sequence, env)`. Once an env fails a task, every later task in that sequence scores 0 for it. |

Independent measures single-task capability; chained measures how far into a
sequence the policy survives, and is deliberately order-sensitive — the same
task set under different permutations gives different chained values. They
coincide at `task_idx == 0`.

#### Accounting fix (affects numbers from prior runs)

Sequential eval deliberately does not reset between tasks. That left ManiSkill's
`_elapsed_steps` at the time limit, so from the second task onward the env
reported `truncated` on *every* step — the aggregate `success` silently became a
time-average instead of a terminal value (under-reporting), grasp flags carried
over between tasks (over-reporting), and `eval_per_trial.csv` sampled the wrong
timestep entirely. `CronosWrapper.begin_segment()` now reopens the measurement
window at each task boundary without touching the scene, so eval and training
compute `success` and grasp identically.

**Sequential-eval numbers from before this change are not comparable to numbers
after it.** Single-task eval and all training-time metrics are unaffected. The
same defect exists upstream in AutoRL's `--only_render_seq`; a correct AutoRL
baseline can be rebuilt from its own artifacts with `tools/parse_autorl_eval.py`
without modifying AutoRL. Full analysis in
[`doc/eval_audit.md`](CRONOS/doc/eval_audit.md).

## Tools

| Tool | Purpose |
|---|---|
| `tools/plot_run_trends.py` | Per-run live 4-panel dashboard (see [Visualization](#visualization)) |
| `tools/plot_rollout_success.py` | Per-segment (per-80-step) rollout success rate from `rollout_success.csv` |
| `tools/plot_segment_positions.py` | Per-segment actor position distribution from `segment_pose.csv` |
| `scripts/plot.py` | Cross-run aggregator over `eval_success.csv` files |
| `tools/mcnemar_pair.py` | Paired McNemar gate over `eval_per_trial.csv` |
| `tools/parse_autorl_eval.py` | Rebuild a correct per-trial baseline from an AutoRL run's video filenames (read-only; AutoRL is never modified) |
| `tools/bench_rollout.py` | Rollout throughput + GPU peak memory per package stack, with a phase breakdown (inference / env.step / buffer / PPO update) |

`bench_rollout.py` exists to make the four-env split answerable rather than
permanent: it attributes the OpenVLA memory regression (~40 GB → ~55 GB after the
transformers/peft upgrade needed by SpatialVLA) to a phase and a package set. Run
the identical command under each env and diff the JSONs — keep the seed, config
and all length flags fixed, since throughput depends on `num_envs`,
`segment_len` and `buffer_inferbatch`.

```bash
python tools/bench_rollout.py \
    --config-path configs/one_group_seq_random_2x2.yaml \
    --policy openvla --vla-path openvla/openvla-7b --vla-unnorm-key bridge_orig \
    --segment-len 80 --episode-len 80 --task-len 80 --ppo-update-len 80 \
    --bench-episodes 2 --bench-warmup-episodes 1 \
    --bench-out reports/bench/openvla_tf447_cu121.json
```

## Architecture
- `envs/` — Environment wrapper, config loader, task scheduler, bridge_multi env
- `training/` — PPO and GRPO algorithms, replay buffer, metrics/CSV recorders
- `main.py` — Training entry point (train + eval)
- `eval_only.py` — Standalone eval script (no training)
- `run_paths.py` — Run output directory resolution, shared by both entry points
- `version.py` — Single source for the version stamped into every `run_config.json`
- `configs/` — YAML training configs
- `configs/eval/` — Per-training eval configs (smaller `num_envs`, `task_order: sequence_random`)
- `scripts/` — Shell scripts for training and eval
- `tools/` — Analysis, plotting, benchmarking and AutoRL-interop utilities
- `doc/` — Design documents, indexed by [`doc/README.md`](CRONOS/doc/README.md): [`eval_audit.md`](CRONOS/doc/eval_audit.md) (eval semantics + the accounting fix), [`data_schemas.md`](CRONOS/doc/data_schemas.md) (CSV column specs), [`grpo_autorl.md`](CRONOS/doc/grpo_autorl.md) (AutoRL GRPO review + CRONOS's grouping / std choices)

