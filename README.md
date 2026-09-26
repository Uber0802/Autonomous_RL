# CRONOS

> **V0.99 — early release.** What changed in each version: [`CRONOS/doc/CHANGELOG.md`](CRONOS/doc/CHANGELOG.md). All other documentation: [`CRONOS/doc/`](CRONOS/doc/README.md).

CRONOS is a refactored robotic manipulation training benchmark designed for **non-episodic reinforcement learning** and **multi-task training**. It is built on the `RL4VLA` backbone with optimized modules from `AutoRL`.

## Features
- **YAML-driven experiment configs** — per-group objects, receptacles, task sequences, eval tasks, and fan-out settings in a single file.
- **Sub-group fan-out** — all tasks in a group run simultaneously on different env sub-groups, matching AutoRL's default multi-task gradient mixing.
- **Per-env rotation eval** — each env rotates through eval tasks across episodes; `num_eval_episode` controls sample count per task.
- **Per-group objects/backgrounds** — different groups can have different physical objects and visual overlays in the same training run.
- **Dual VLA support** — `--policy openvla|spatialvla` switches between OpenVLA-7B and SpatialVLA-4B adapters (7-token vs 3-token action sequences).
- **PPO or GRPO** — `--alg-name grpo` swaps the critic-free path in (**experimental** — it currently collapses, see [`grpo_failure.md`](CRONOS/doc/reports/grpo_failure.md)); grouping is selectable at three nesting levels (`batch` / `scene` / `task`), with `batch` verified bit-identical to AutoRL's `compute_returns_grpo`.
- **Orthogonal reset dimensions** — LSR (learned reset policy), HSR (respawn fallen actors), EER (gripper re-home), and Perturbation (the reset goal is sometimes a *different receptacle* instead of the table), each toggled independently.
- **Modular Environment** — decoupled `reset_strategy`, `reward_shaping`, `task_suite`, and `task_scheduler`.
- **Efficient Rollouts** — multi-task execution with GPU-parallelized ManiSkill environments, memory-mapped replay buffers.
- **Analysis-ready outputs** — per-segment CSVs recording both sides of every segment boundary, documented column by column in [`CRONOS/doc/data_schemas.md`](CRONOS/doc/data_schemas.md).

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
- ❌ Cannot run `--policy spatialvla` — transformers ≥ 4.43 needed for the `HybridCache` import in SpatialVLA's `model/modeling_gemma2.py`. `setup.sh openvla_v01` skips the `../SpatialVLA` editable install entirely; the policy's lazy import in [main.py:385](CRONOS/main.py#L385) and [eval_only.py:210](CRONOS/eval_only.py#L210) is gated by `--policy spatialvla` so it never fires under OpenVLA-only runs.
- ❌ Will not run on Blackwell as-is — pass `blackwell` as the 2nd arg to install the Blackwell variant: `./setup.sh openvla_v01 blackwell` produces `cronos_tf440_cu128` (lightweight stack on `torch==2.7.0+cu128`; loses cu121 bit-exactness but keeps V0.1 transformers ABI).

How `setup.sh` picks the install: the 1st positional arg picks the LM stack + which sibling pillars get installed, the 2nd picks the torch wheel channel. See the header comment in `setup.sh` for the full pin rationale and the 4-env recommended workflows.

### 7. Checkpoint portability across envs

A checkpoint is written by whichever env trained it, and the two LM stacks do not
serialize the LoRA adapter the same way. **`tf447` (peft 0.14) writes three
`LoraConfig` fields that `tf440` (peft 0.11.1) has no field for** — `eva_config`,
`exclude_modules`, `lora_bias`. `LoraConfig` is a dataclass, so peft 0.11.1 does
not ignore the extras; `PeftModel.from_pretrained` dies on the constructor:

```
TypeError: LoraConfig.__init__() got an unexpected keyword argument 'eva_config'
```

This cannot be pinned away. peft 0.14 needs `transformers>=4.43` (for
`EncoderDecoderCache`), which breaks OpenVLA's `transformers<4.43` pin, and peft
0.11.1 is the newest release that works against transformers 4.40.1 — so it is
handled at the load site instead, in
[`SimplerEnv/simpler_env/policies/peft_compat.py`](SimplerEnv/simpler_env/policies/peft_compat.py).
`load_peft_adapter` filters out fields the *installed* peft cannot represent
before building the config, and both VLA pillars load through it.

The filter is behaviour-neutral, and enforces that rather than assuming it: a key
is dropped **only** when it holds the value that means "feature off"
(`eva_config: null`, `exclude_modules: null`, `lora_bias: false`) — which is what
`tf447` checkpoints actually contain, since nobody turned those features on. A key
that is genuinely set, or one no peft release in the table accounts for, raises
instead of being silently discarded. The reverse direction (`tf440` checkpoint,
`tf447` reader) needs no repair and is the identity transform.

Audit a checkpoint tree before committing to a long eval, without loading a model
(or peft, or torch):

```bash
python tools/check_ckpt_compat.py /path/to/runs                 # default target: tf440, the strict reader
python tools/check_ckpt_compat.py /path/to/runs --target tf447
python tools/check_ckpt_compat.py /path/to/runs --verbose       # one line per checkpoint
```

It exits non-zero if any checkpoint cannot be loaded by the target stack, so it
also works as a preflight step in a shell script.

The LoRA *weights* (`adapter_model.safetensors`) have the same layout in both peft
releases and need no translation. `training_state.pt` is a torch pickle, and its
own cross-version hazard is torch 2.6 flipping the `torch.load` default to
`weights_only=True`; every load site passes `weights_only=False` explicitly, since
CRONOS spans torch 2.2 / 2.5 / 2.7 and writes these files itself.

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
#                      │      │      │      └─ normal | LSR | HSR | HSR+LSR | noep | noep+LSR (default: normal)
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

# SpatialVLA, T1280 segment 'b', seed 1, GPU 2, non-episodic (HSR + reset_mode=none)
bash scripts/train.sh t1280b 1 2 noep four_group_sequential_2x2 spatialvla

# Same plus a learned reset policy — the mode `noep` should be compared against
bash scripts/train.sh t1280b 1 2 noep+LSR four_group_sequential_2x2 spatialvla

# Same, but with the end effector never repositioned between segments
bash scripts/train.sh t1280b 1 2 noep four_group_sequential_2x2 spatialvla off

# GRPO instead of PPO — AutoRL-compatible grouping (one group per segment)
bash scripts/train.sh t320a 0 3 normal four_group_sequential_2x2 openvla on grpo

# GRPO grouped per scene (segment × YAML group — same objects/receptacles/background)
bash scripts/train.sh t320a 0 3 normal four_group_sequential_2x2 openvla on grpo-scene

# GRPO grouped per task (segment × fan-out sub-block — narrowest, most apples-to-apples)
bash scripts/train.sh t320a 0 3 normal four_group_sequential_2x2 openvla on grpo-task

# Perturbation: the LSR reset goal is sometimes "put X on the OTHER receptacle"
# instead of always "put X on table" — widens the forward policy's start states.
# Needs a mode with LSR; plain `noep` has no reset segment to perturb.
bash scripts/train.sh t320a 0 3 noep+LSR four_group_sequential_2x2 openvla on ppo mixed

# Perturbation composes with GRPO (orthogonal dimensions)
bash scripts/train.sh t320a 0 3 noep+LSR four_group_sequential_2x2 openvla on grpo-task recep

# Any GRPO mode with the std term overridden (tagged, so it lands in its own dir)
GRPO_STD_SCOPE=none bash scripts/train.sh t320a 0 3 normal four_group_sequential_2x2 openvla on grpo-task
```

`RUN_TAG` carries the VLA tag (`CRONOS-openvla-<config>-<horizon>-<reset>-seed<N>`), so OpenVLA and SpatialVLA runs land in separate output dirs.

**Reset-mode legend** — full account in [`CRONOS/doc/reset_modes.md`](CRONOS/doc/reset_modes.md):

| mode | CLI flags added | `RUN_TAG` | Meaning |
|---|---|---|---|
| `normal` | (nothing) | `normal` | hard `env.reset()` every episode |
| `LSR` | `--enable-backward --backward-interval 1` | `LSR` | learn the backward policy (put X back) alternating with forward task switches |
| `HSR` | `--reset-unsuitable` | `HSR` | respawn fallen / out-of-workspace actors at every task boundary |
| `HSR+LSR` | HSR + LSR | `HSRLSR` | soft respawn + backward learning. `LSR+HSR` is still accepted as input |
| `noep` | HSR + `--reset-mode none` | `HSRnoep` | non-episodic continuity (no inter-episode hard reset). **No LSR** |
| `noep+LSR` | HSR + LSR + `--reset-mode none` | `HSRLSRnoep` | non-episodic continuity *with* a learned reset policy |

`noep` means "HSR but no episodic reset" and does **not** include LSR — that is
`noep+LSR`. The `HSR+LSR` / `noep` / `noep+LSR` triple is what separates "does
removing the episodic reset hurt" from "does learning a reset policy pay for it".

> ⚠️ **`noep` changed meaning, and the tags were renamed.** It used to expand to
> LSR+HSR+`--reset-mode none`, i.e. today's `noep+LSR`. Any run directory
> containing `-noep-` predates the change and is a `noep+LSR` run under the
> current definition — do not put it in the same plot-config group as a new
> `-HSRnoep-` run. The tag rename (`LSR+HSR`→`HSRLSR`, `noep`→`HSRnoep`,
> `noep+LSR`→`HSRLSRnoep`) guarantees no new run collides with a historical one.

> ⚠️ Bare `noep` has no mechanism that returns a *successfully placed* object to
> its initial state: HSR respawns only what its detector flags as fallen or out
> of bounds, and there is no `env.reset()`. Start states therefore drift toward
> already-satisfied tasks, which inflates both reward and rollout success rate.
> `main.py` warns at startup; `doc/reset_modes.md` says how to check for it.

**Perturbation** — the 9th positional arg, orthogonal to the reset modes but requiring one that includes LSR (`LSR`, `HSR+LSR`, `noep+LSR`):

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

**EER (End-Effector Reset)** — the 7th positional arg, orthogonal to every reset mode above:

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
> mechanism described in [`doc/eval_audit.md`](CRONOS/doc/reports/eval_audit.md).

Key training flags:
| Flag | Default | Description |
|---|---|---|
| `--config-path` | required | YAML experiment config |
| `--policy` | `openvla` | `openvla` or `spatialvla` |
| `--alg-name` | `ppo` | `ppo` (actor-critic + GAE) or `grpo` (critic-free) |
| `--grpo-group-scope` | `batch` | GRPO only. What counts as one group: all three are per-segment. `batch` (whole segment — the statistic AutoRL uses, bit-identical) \| `scene` (segment × YAML group) \| `task` (segment × fan-out sub-block). Sizes for `four_group_sequential_2x2`: 64 / 16 / 4 |
| `--grpo-std-scope` | `group` | GRPO only. Divide group-centred rewards by `group` / `global` std, or `none`. See [`doc/grpo_autorl.md` §9](CRONOS/doc/reports/grpo_autorl.md) |
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
comparable; they differ only in when they are sampled. Under a mode with LSR
(`LSR`, `HSR+LSR`, `noep+LSR`), filter `direction == 'forward'` before
aggregating: the env's success predicate is always the forward one, so backward
segments score 0 by construction. Modes without LSR — including bare `noep` —
log every row as `forward`, so the filter is a harmless no-op there.

### Evaluation (standalone)

`eval_only.py` (and `main.py --eval-single / --eval-sequential`, which share its
loop in `evaluation/`) evaluates a checkpoint on **every scene** (YAML group) of
the checkpoint's **training config file** — found from the checkpoint (the
`experiment_config.yaml` snapshot training writes next to it, else the
`config_path` in its `run_config`). A config passed explicitly must define the same
scenes as training, or eval stops before loading the model
(`--allow-config-mismatch` to override on purpose).
Full design: [`doc/eval_sequential.md`](CRONOS/doc/eval_sequential.md).

**Rounds.** A round is one reset followed by every task of the scene without
resets (AutoRL `render_seq`). Each scene's envs are split into 4 order blocks, as
fan-out training splits them, and each block runs its own order. With 4 tasks
there are 6 cycles × 4 rotations = 24 orderings, so a **pose set is 6 rounds**:

| round | kind | blocks run |
|---|---|---|
| 0, 6, 12, … | training | ABCD BCDA CDAB DABC — exactly the training rotations |
| 1–5, 7–11, … | random | the 4 rotations of one untrained cycle (same cycle order in every pose set) |

Round numbers are global and the **only selector** (`rounds`). Everything about a
round — orders, start poses, action-sampling seed — is a function of its number
and the seed, so any subset reproduces those rounds of a full run.

**Start poses.** Every round, order and scene of a pose set starts from the same
poses (one per block slot, 4 with 16-env scenes); round 6 starts pose set 1 with
new poses. `pose_sets: K` gives K × 24 orderings on 4K start layouts.

**Resume / shards.** After a crash, rerun the same command with
`--eval-resume <glob>`: finished rounds are skipped, a half-finished round is
rerun. Rounds can also be split across GPUs (`--eval-rounds 0-2` / `3-5`) and merged
with `tools/rebuild_eval_outputs.py --out <dir> <glob_a> <glob_b>`.
`eval_success.csv` is written only when the eval is complete *and* covers every
round of every pose set (a shard gets none until merged); `eval_status.json`
always says what is done and missing.

**Settings** live in the `eval:` block of the training config (training ignores
it); CLI flags override it:

```yaml
eval:
  mode: sequential                 # sequential | single (one task slot per round, blocks run A/B/C/D)
  pose_sets: 1                     # sets of start poses; 6 rounds each (4 tasks)
  rounds: all                      # all | 3 | 3-5 | 3- | 0,6-11
  sequence_seed: -1                # cycle order; -1 = --seed
  layout_seed: -1                  # start poses; -1 = --seed
  policy_seed: -1                  # per-round action-sampling reseed; -1 = --seed
  layout_slots: -1                 # start poses per block; -1 = one per env of the block, 1 = one pose
  scene_schedule: parallel         # parallel: all scenes in one batch | serial: one scene x all envs
  domains: [in_domain, out_of_domain]
  record_video: true
  video_envs_per_block: 1          # first K envs of each block; -1 = all
  record_pose: true                # eval_segment_pose.csv (training segment_pose.csv columns)
  pose_phase: both
```

**Wrapper script** (recommended — sets the right env vars):
```bash
# Args: <checkpoint_dir> [config|-] [cuda] [num_eval_episode] [extra eval_only.py flags...]
# config empty or "-" = the checkpoint's training config
bash scripts/eval.sh /path/to/glob/episode_0128 - 0
bash scripts/eval.sh /path/to/glob/episode_0128 - 0 4 --eval-rounds 0-5 --eval-domains in_domain
bash scripts/eval.sh /path/to/glob/episode_0128 - 0 4 --eval-resume /path/to/eval/wandb/offline-run-…/glob
```

`--num-envs` is **not** needed — it is the sum of per-group `num_envs`, and eval
refuses a batch that does not match it (no padding envs).

**OpenVLA and SpatialVLA** use the same command. `policy`, `vla_path`,
`vla_unnorm_key`, `vla_temperature_eval` and `vla_lora_rank` come from the
checkpoint's training `run_config` (then the config YAML, then defaults matching
`train.sh` — SpatialVLA: `bridge_orig/1.0.0`, greedy decoding). Pass
`--policy spatialvla` etc. after the 4th argument only to override, e.g. for a
checkpoint without a run_config. The resolved values and their sources are printed
at startup and stored in `eval_plan.json`.

**Coverage** (`four_group_sequential_2x2`, one pose set, per domain, per scene):
training round 64 trials (16 per task), random rounds 320 trials (80 per task),
4 start poses; 245,760 env-steps for all scenes and both domains. `serial` gives
4× trials and 16 poses at 4× the steps. Printed before rollout, checked after.

**Outputs** (under a fresh `wandb/offline-run-<timestamp>-<id>/glob/`; columns in [`doc/data_schemas.md`](CRONOS/doc/data_schemas.md)):

| File | Contents |
|---|---|
| `eval_plan.json` | resolved settings and their sources, fingerprint, config provenance, every round's orders per block, units, RNG keys and ids, planned coverage, resume history |
| `eval_status.json` | complete?, done / partial / missing units |
| `eval_per_trial.csv` | source: one row per (domain, round, task slot, env) — `success`, `success_chained`, grasp, `seq_kind`, `group`, `order`, `pose_set` |
| `eval_layouts.csv` | source: per reset and env, layout slot, stream key, id, applied pose/quat/overlay ids |
| `eval_segment_pose.csv` | source: poses at every task start/end, training `segment_pose.csv` columns + eval keys |
| `eval_sequence_summary.csv` / `eval_coverage.csv` / `eval_report.txt` | derived, rebuilt whole; always split by `seq_kind` |
| `eval_success.csv` | derived, `eval_kind` = `in_domain_training`, `in_domain_random`, …; **complete evals only** |
| `eval_videos/<domain>/<seq_kind>/round<N>/task<M>/…` | videos |
| `experiment_config.yaml`, `run_config.json` | config and args used |

Eval flags (the bracketed name is the `eval:` key):
| Flag | Default | Description |
|---|---|---|
| `--vla-load-path` | required | Checkpoint dir (`episode_XXXX/`) |
| `--config-path` | checkpoint's training config | must define the same scenes as training |
| `--allow-config-mismatch` | false | evaluate on a config whose scenes differ from training |
| `--eval-resume` | — | glob dir of an interrupted eval |
| `--eval-mode` [mode] | `sequential` | `sequential` or `single` |
| `--eval-pose-sets` [pose_sets] | 1 | sets of start poses |
| `--eval-rounds` [rounds] | `all` | global round selection |
| `--eval-sequence-seed` / `--eval-layout-seed` / `--eval-policy-seed` | -1 | streams; -1 = `--seed` |
| `--eval-layout-slots` [layout_slots] | -1 | start poses per block |
| `--eval-scene-schedule` [scene_schedule] | `parallel` | `parallel` or `serial` |
| `--eval-domains` [domains] | `in_domain,out_of_domain` | comma-separated; `--no-eval-ood` = `in_domain` |
| `--record-video` / `--video-envs-per-block` | true / -1 | videos, first K envs of each block |
| `--record-eval-pose` / `--eval-pose-phase` | true / `both` | pose CSV |
| `--segment-len` | 80 | Steps per task rollout |
| `--policy` / `--vla-path` / `--vla-unnorm-key` / `--vla-temperature-eval` / `--vla-lora-rank` | from the checkpoint | override the training policy settings |
| `--action-chunk` | 1 | SpatialVLA open-loop chunk length |

Removed: `--eval-sequences`, `--eval-training-sequence`, and the YAML keys
`num_sequences`, `include_training_sequence`, `training_orders`, `random_orders`,
`layout_rng` — use `rounds`.

**Note:** the per-env rotation eval lives only inside `train()` for training-time
eval. Use `--eval-at-start` for rotation eval of a checkpoint loaded by `main.py`.

### Monitoring a run

`main.py` refreshes a live dashboard in the run's `glob/` after every eval point
(`tools/plot_run_trends.py`; failures are non-fatal). It can also be run by hand.
Offline plotting and statistics tools (cross-run aggregation, per-segment curves,
position distributions, sequence-eval bars, McNemar tests) are not part of this
release; see [`CRONOS/doc/README.md`](CRONOS/doc/README.md#what-is-not-in-the-release).

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

### Training config → eval config mapping

Standalone eval now uses the **training config itself** (see [Evaluation (standalone)](#evaluation-standalone)).
The configs under `configs/eval/` are legacy reduced-env copies: they differ from
training in `num_envs`, so they only run with `--allow-config-mismatch`.

| Training config | Eval config | Notes |
|---|---|---|
| `one_group_half_train_2x2.yaml` | `eval/one_group_half_train_2x2.yaml` | Generalization: trained on 2 of 4 tasks, eval on all 4 |
| `one_group_pure_random_2x2.yaml` | `eval/one_group_2x2.yaml` | Default 2x2 single-group eval |
| `one_group_seq_random_2x2.yaml` | `eval/one_group_2x2.yaml` | Same eval as pure_random — neither has a canonical training order |
| `one_group_sequential_3x3.yaml` | `eval/one_group_3x3.yaml` | 9 tasks; sequence 0 = auto-generated NxM order from training config |
| `two_group_sequential_2x2.yaml` | `eval/two_group_2x2.yaml` | legacy; the training config itself now carries an `eval:` block |
| `four_group_sequential_2x2.yaml` | — | use the training config; every `eval:` key spelled out |

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
cronos_version: V0.99
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
cronos_version: V0.99
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

### Standalone eval — per-scene rounds

Used by `eval_only.py` and `main.py --eval-single` / `--eval-sequential`. Each
scene runs on its own env range (`scene_schedule: parallel`) or on all envs one
scene at a time (`serial`), split into 4 order blocks; each block runs its own
order for `segment_len` steps per task.

| Mode | Behavior | AutoRL analog |
|---|---|---|
| `sequential` (default) | rounds selected by number; each pose set = 6 rounds = all 24 orderings (round 6k training rotations, 6k+1..6k+5 the other cycles); every round of a pose set starts from the same poses; tasks within a round switch without env reset. Round N is reproducible per seed. | `render_seq(eval_training_seq=True)` |
| `single` | One task slot per round from a fresh reset; the blocks run tasks A/B/C/D side by side. | `render` |

Details: [`doc/eval_sequential.md`](CRONOS/doc/eval_sequential.md).

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
baseline can be rebuilt from its own artifacts (offline tool `parse_autorl_eval.py`,
not part of this release) without modifying AutoRL. Full analysis in
[`doc/eval_audit.md`](CRONOS/doc/reports/eval_audit.md).

## Tools

| Tool | Purpose |
|---|---|
| `tools/plot_run_trends.py` | Per-run live 4-panel dashboard, refreshed by `main.py` after each eval (see [Monitoring a run](#monitoring-a-run)) |
| `tools/rebuild_eval_outputs.py` | Rebuild standalone-eval derived files from per-trial rows; merge round shards of one eval |
| `tools/check_ckpt_compat.py` | Preflight: can a checkpoint tree be loaded by the target LM stack (`tf440` / `tf447`)? No GPU or model load needed |
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
- `evaluation/` — Standalone eval: plan (settings, scenes, rounds, coverage, fingerprint), records (CSV schemas, readers/writers), outputs (derived files, status, resume/merge), provenance (training config), sequential (rollout loop)
- `tests/` — CPU-only tests for eval planning, RNG streams, records, the eval loop and peft checkpoint compatibility (`python -m pytest tests/ -q`)
- `run_paths.py` — Run output directory resolution, shared by both entry points
- `version.py` — Single source for the version stamped into every `run_config.json`
- `configs/` — Sample YAML training configs
- `configs/eval/` — Legacy reduced-env eval configs (need `--allow-config-mismatch`)
- `scripts/` — `train.sh`, `eval.sh`, and the plotting requirements `setup.sh` installs
- `tools/` — Eval-shard merging, checkpoint compatibility, benchmarking, live dashboard
- `doc/` — All documentation, indexed by [`doc/README.md`](CRONOS/doc/README.md): changelog, design references, and bug / failure reports

## Version and changelog

This is **CRONOS V0.99**, an early release. The version is defined in
[`CRONOS/version.py`](CRONOS/version.py) and stamped into every run's
`run_config.json`.

Every version since the initial refactor is listed in
[`CRONOS/doc/CHANGELOG.md`](CRONOS/doc/CHANGELOG.md). Entries marked
**[numbers-affected]** changed results collected with earlier versions — check
them before comparing runs across versions. Known issues in this release:

- **GRPO is experimental.** `--alg-name grpo` degrades the SFT policy instead of
  improving it; see [`doc/reports/grpo_failure.md`](CRONOS/doc/reports/grpo_failure.md).
  Use PPO (the default) for results.
- **Training RNG is not yet isolated.** Standalone eval is reproducible per round,
  but training still shares generators between layout draws and action sampling;
  see [`doc/rng_and_io_notes.md`](CRONOS/doc/rng_and_io_notes.md).
