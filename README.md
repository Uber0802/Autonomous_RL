# CRONOS

CRONOS is a refactored robotic manipulation training benchmark designed for **non-episodic reinforcement learning** and **multi-task training**. It is built on the `RL4VLA` backbone with optimized modules from `AutoRL`.

## Features
- **YAML-driven experiment configs** — per-group objects, receptacles, task sequences, eval tasks, and fan-out settings in a single file.
- **Sub-group fan-out** (V0.3) — all tasks in a group run simultaneously on different env sub-groups, matching AutoRL's default multi-task gradient mixing.
- **Per-env rotation eval** (V0.3) — each env rotates through eval tasks across episodes; `num_eval_episode` controls sample count per task.
- **Per-group objects/backgrounds** (V0.3) — different groups can have different physical objects and visual overlays in the same training run.
- **Modular Environment**: Decoupled `reset_strategy`, `reward_shaping`, `task_suite`, and `task_scheduler`.
- **Efficient Rollouts**: Multi-task execution with GPU-parallelized ManiSkill environments.
- **Robust Training**: PPO implementation with memory-mapped replay buffers.
- **Pluggable VLA backbones** — OpenVLA (default) and [SpatialVLA](#spatialvla-policy) are both first-class, PPO-trainable policies behind the same SimplerEnv adapter seam, selected per run via a config field or CLI flag.

## Installation

The following steps assume a Linux environment with NVIDIA GPUs and Conda installed.

### 1. Get the repository

Clone (or download) and move into the repository directory:

```bash
git clone <repo-url>
cd <repo-directory>
```

**Anonymous repository (zipped distribution):**

If you received the project as a zip archive (e.g., from an anonymized paper submission link), unpack it instead:

```bash
mkdir Autonomous_RL
mv XXX.zip Autonomous_RL
cd Autonomous_RL
unzip XXX.zip
```

### 2. Create the conda environment

```bash
conda create -n cronos_env -y python=3.10
conda activate cronos_env
```

### 3. Run the setup script

```bash
cd CRONOS
chmod +x *.sh
./setup.sh
```

`setup.sh` installs CRONOS plus its sibling dependencies (`SimplerEnv`, `ManiSkill`, `openvla`, and `SpatialVLA`), which must already be present in the same parent directory as `CRONOS/`. The same script installs the SpatialVLA backbone (`pip install -e ../SpatialVLA`) and pins a Blackwell-compatible torch/transformers stack — no extra step is needed to enable SpatialVLA. See [§SpatialVLA Policy](#spatialvla-policy) for how to run it.

### 4. (Optional) Ubuntu 22.04 prerequisite

If the Vulkan driver is missing on Ubuntu 22.04 (SAPIEN will warn about a missing ICD file at startup), install:

```bash
sudo apt-get update
sudo apt-get install -y libglvnd-dev
```

### 5. (Optional) Blackwell GPU support

CRONOS has been verified on Ampere (A100), Ada (L40S), and Hopper (H100). **Blackwell cards (RTX PRO 6000, RTX 5090, B200) require a different torch build** — the stock `cronos_env` ships wheels whose SASS kernels top out at `sm_90`, so the first CUDA op on a Blackwell device crashes with:

```
RuntimeError: CUDA error: no kernel image is available for execution on the device
```

Clone the working env and upgrade torch to the `cu128` channel, which ships kernels for `sm_75 ... sm_120 + PTX` and therefore runs on **every** GPU from Turing through Blackwell in a single env:

```bash
conda create -n cronos_env_blackwell --clone cronos_env
conda activate cronos_env_blackwell
pip uninstall -y torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

If `flash-attn` or `xformers` break afterwards (they link against the torch C++ ABI):

```bash
pip install flash-attn --no-build-isolation
pip install xformers --index-url https://download.pytorch.org/whl/cu128
```

**Why a clone and not an in-place upgrade of `cronos_env`:** openvla pins transformers/flash-attn versions compiled against the original torch ABI. Cloning isolates the blast radius — if something breaks, `cronos_env` remains intact on the non-Blackwell machines. The Blackwell-specific env is named with the `_blackwell` suffix so each machine uses the matching one.

*Reproducibility note:* PTX is JIT-compiled on first CUDA op on a new arch, so SASS may differ slightly from a native build. Training curves on Blackwell are *statistically* equivalent to Hopper, not bit-exact.

## Quick Start

### Training
```bash
# V0.3 training with YAML config, fan-out, and eval-at-start
bash scripts/train.sh
```

Key training flags:
| Flag | Default | Description |
|---|---|---|
| `--config-path` | required | YAML experiment config |
| `--policy` | `openvla` | VLA backbone: `openvla` or `spatialvla` (usually pinned in the YAML — see [§SpatialVLA Policy](#spatialvla-policy)) |
| `--num-envs` | 64 | Total parallel environments |
| `--segment-len` | 80 | Steps per segment (AutoRL: 80) |
| `--ppo-update-len` | 80 | Steps between PPO updates |
| `--eval-interval` | 4 | Eval every N episodes |
| `--num-eval-episode` | 4 | Episodes per eval round |
| `--eval-at-start` | false | Run eval before first training episode |

`scripts/train.sh` runs the OpenVLA baseline; for SpatialVLA PPO use the `main.py` invocations in [§SpatialVLA Policy → Training](#training-ppo).

### Evaluation (standalone)

`eval_only.py` runs **AutoRL-style broadcast eval** by default: every env in the batch runs the same `(object, receptacle)` for one `segment_len` rollout, no fan-out. Two modes:

- `--eval-mode sequential` (default): runs `--eval-sequences N` orderings of the eval tasks. The first ordering is the training task order (sequence 0, AutoRL `eval_training_seq=True` convention); subsequent orderings are random permutations. Within each ordering, tasks switch one-by-one without env reset (matches AutoRL `render_seq`).
- `--eval-mode single`: runs each task in `eval_tasks` once independently, all envs on the same task per pass (matches AutoRL `render`).

**Note:** the V0.3 fan-out rotation eval (per-env rotation) is no longer the default for standalone eval — it lives only inside `train()` for training-time eval. Use `--eval-at-start` if you want the rotation eval against a checkpoint loaded by main.py.

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
| `eval_report.txt` | human-readable per-eval summary (rotation table + per-task means) |
| `eval_videos/{kind}/eval_ep{M}/env{i}.mp4` | one mp4 per env per eval episode (if `--record-video`) |
| `run_config.json` / `run_config.yaml` | exact args + resolved YAML at run time |

Common eval flags:
| Flag | Default | Description |
|---|---|---|
| `--config-path` | required | YAML experiment config (typically one from `configs/eval/`) |
| `--policy` | `openvla` | VLA backbone: `openvla` or `spatialvla` (pair with the matching `--vla-path` / `--vla-unnorm-key`) |
| `--vla-load-path` | required | Checkpoint dir (the `episode_XXXX/` from a training run, or `TestCheckpoint/seed0/`) |
| `--eval-mode` | `sequential` | `sequential` (AutoRL `render_seq`) or `single` (AutoRL `render`) |
| `--eval-sequences` | 5 | Sequential mode: training-order + N-1 random permutations |
| `--action-chunk` | 1 | SpatialVLA chunk(K) open-loop deployment (`K=1` = single-step; SpatialVLA-only) |
| `--segment-len` | 80 | Steps per task rollout |
| `--record-video` | true | Write mp4s under `glob/eval_videos/{prefix}/` |
| `--vla-temperature-eval` | 0.6 | Sampling temperature for the policy |

### Training config → eval config mapping

Each training config under `configs/` has a matching eval config under `configs/eval/` with smaller `num_envs` for faster eval. Eval configs use `task_order: sequence_random` so eval orderings match AutoRL's random-permutation default.

| Training config | Eval config | Notes |
|---|---|---|
| `one_group_half_train_2x2.yaml` | `eval/one_group_half_train_2x2.yaml` | Generalization: trained on 2 of 4 tasks, eval on all 4 |
| `one_group_pure_random_2x2.yaml` | `eval/one_group_2x2.yaml` | Default 2x2 single-group eval |
| `one_group_seq_random_2x2.yaml` | `eval/one_group_2x2.yaml` | Same eval as pure_random — neither has a canonical training order |
| `one_group_sequential_3x3.yaml` | `eval/one_group_3x3.yaml` | 9 tasks; sequence 0 = auto-generated NxM order from training config |
| `two_group_sequential_2x2.yaml` | `eval/two_group_2x2.yaml` | **Multi-group standalone eval is V0.4 work** — see caveat in the file |
| `spatialvla_2x2_train.yaml` | `eval/one_group_2x2_eval24.yaml` | SpatialVLA PPO ([§SpatialVLA Policy](#spatialvla-policy)); same Minimal 2×2 scene, eval at `num_envs=24` (paired McNemar baseline) |

## SpatialVLA Policy

CRONOS ships [SpatialVLA](https://github.com/SpatialVLA/SpatialVLA) as a first-class, PPO-trainable policy, wired through the **same SimplerEnv seam** as OpenVLA. The integration is non-invasive: OpenVLA remains the default everywhere, and SpatialVLA is selected per run — by a `policy: spatialvla` field in a training config, or a `--policy spatialvla` flag in standalone eval.

### Setup

No SpatialVLA-specific install step is required beyond the standard [Installation](#installation) — `setup.sh` already vendors the `SpatialVLA/` sibling and installs it editable (`pip install -e ../SpatialVLA`). Two notes:

- **Weights are pulled from HuggingFace on first use.** The canonical base checkpoint is `IPEC-COMMUNITY/spatialvla-4b-224-sft-bridge`; it downloads automatically the first time a SpatialVLA run starts (set `HF_HOME` to control the cache location). No manual download is needed.
- **Unnorm key.** SpatialVLA's action de-normalization statistics live under `bridge_orig/1.0.0` (not the OpenVLA `bridge_orig` key). The training configs pin this; for standalone eval you must pass `--vla-unnorm-key bridge_orig/1.0.0` on the CLI.

The policy adapter lives at `SimplerEnv/simpler_env/policies/spatialvla/spatialvla_train.py` (mirrors `policies/openvla/openvla_train.py`).

### Evaluation

Standalone eval runs through the same `eval_only.py` as OpenVLA, with the policy chosen by `--policy spatialvla`. The canonical zero-shot eval on the 2×2 Minimal scene — also the paired baseline for the PPO McNemar gate — uses `configs/eval/one_group_2x2_eval24.yaml` (24 envs = 24 trials/task):

```bash
safejob python eval_only.py \
    --config-path configs/eval/one_group_2x2_eval24.yaml \
    --eval-mode single --action-chunk 1 \
    --seed 0 \
    --policy spatialvla \
    --vla-path IPEC-COMMUNITY/spatialvla-4b-224-sft-bridge \
    --vla-unnorm-key bridge_orig/1.0.0 \
    --record-video
```

To evaluate a **trained checkpoint** instead of the zero-shot base, add `--vla-load-path /path/to/glob/episode_XXXX`. All other eval semantics (modes, outputs, flags) are identical to the [Evaluation](#evaluation-standalone) section above. SpatialVLA-specific flag:

| Flag | Default | Description |
|---|---|---|
| `--action-chunk` | 1 | SpatialVLA chunk(K) open-loop deployment. `K=1` = single-step (matches OpenVLA); `K>1` steps the env K times per model call. SpatialVLA-only. |

### Training (PPO)

PPO training also runs through `main.py`, with the policy pinned in the config — `configs/spatialvla_2x2_train.yaml` sets `policy: spatialvla`, the `vla_path`, the `bridge_orig/1.0.0` unnorm key, and the LoRA / value-head learning-rate split. The three regime flags (`--reset-mode`, `--episode-len`, `--ppo-update-len`) are supplied on the command line.

**Smoke test first** (verifies the full pipeline on `num_envs=8`, one episode, no wandb/video):

```bash
safejob python main.py --config-path configs/spatialvla_2x2_smoke.yaml \
    --seed 0 \
    --reset-mode per_episode --episode-len 80 --ppo-update-len 80 \
    --task-len 80 --segment-len 80 \
    --max-episodes 1 \
    --buffer-minibatch 4 --alg-gradient-accum 1 \
    --no-record-video --no-wandb \
    --name spatialvla_smoke
```

**Episodic PPO** (standard hard-reset training):

```bash
safejob python main.py \
    --config-path configs/spatialvla_2x2_train.yaml \
    --seed 0 \
    --reset-mode per_episode \
    --episode-len 80 --ppo-update-len 80 --task-len 80 --segment-len 80 \
    --max-episodes 128 \
    --name spatialvla_episodic
```

**Non-episodic PPO** (CRONOS's target regime — no episodic reset, soft state resets only):

```bash
safejob python main.py \
    --config-path configs/spatialvla_2x2_train.yaml \
    --seed 0 \
    --reset-mode none --episode-len 320 --ppo-update-len 160 \
    --task-len 80 --segment-len 80 --reset-unsuitable \
    --max-episodes 8 \
    --name spatialvla_nonepisodic
```

Both training runs start **fresh** from the sft-bridge base (no `--vla-load-path`); neither is warm-started from the other. Each is independently the PPO actor that the eval command above scores against its zero-shot baseline. Two ready-to-run launchers reproduce the recorded runs verbatim (exact flags, run dirs, and McNemar results in their header comments): `scripts/p5_fresh_128ep.sh` (episodic, 128 episodes) and `scripts/p6_perep_320_evi1.sh` (long-horizon 320-step episodes — the integration done-criterion run).

> **Note:** the OpenVLA-oriented `scripts/train.sh` / `scripts/eval.sh` wrappers default to `openvla/openvla-7b` and do **not** select the SpatialVLA policy — use the `main.py` / `eval_only.py` invocations above (with the `spatialvla_*` configs and `--policy spatialvla`) for SpatialVLA.

## YAML Config Format

Configs live in `configs/`. A single YAML file fully describes an experiment.

### Top-level fields

| Field | Type | Default | Description |
|---|---|---|---|
| `cronos_version` | string | — | Config format version (e.g. `V0.3`) |
| `task_order` | string | `sequential` | `sequential`, `pure_random`, or `sequence_random` |
| `fan_out` | bool | `true` | All tasks run simultaneously within each group (AutoRL default) |
| `policy` | string | `openvla` | VLA backbone adapter: `openvla` or `spatialvla`. Overrides the CLI default at construction time. |
| `vla_path` | string | `openvla/openvla-7b` | HF model id / local path for the chosen policy (e.g. `IPEC-COMMUNITY/spatialvla-4b-224-sft-bridge`) |
| `vla_unnorm_key` | string | `bridge_orig` | Action de-normalization stats key (SpatialVLA uses `bridge_orig/1.0.0`) |
| `vla_lr` | float | `1.0e-4` | LoRA AdamW learning rate (policy params) |
| `vla_vhlr` | float | `3.0e-3` | Value-head AdamW learning rate |

The `policy` / `vla_*` fields are optional and apply to both backbones; the `spatialvla_*` configs set them so a SpatialVLA run needs no extra CLI flags. See [§SpatialVLA Policy](#spatialvla-policy).

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
| `spatialvla_2x2_train.yaml` | SpatialVLA PPO on the Minimal 2×2 scene (`policy: spatialvla` pinned) |
| `spatialvla_2x2_smoke.yaml` | SpatialVLA pipeline smoke test (`num_envs: 8`, one episode) |

### Minimal single-group config

```yaml
cronos_version: V0.3
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
cronos_version: V0.3
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

### Training-time eval — per-env rotation (V0.3)

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

### Standalone eval — AutoRL-style broadcast (V0.3.1)

Used by `eval_only.py` and `main.py --eval-single` / `--eval-sequential`. All envs run the same `(object, receptacle)` for one `segment_len`-step rollout (no fan-out). Two modes, configurable via `--eval-mode`:

| Mode | Behavior | AutoRL analog |
|---|---|---|
| `sequential` (default) | `--eval-sequences N` orderings of the eval tasks; sequence 0 is the training task order, sequences 1..N-1 are random permutations. Tasks within an ordering switch one-by-one without env reset. | `render_seq(eval_training_seq=True)` |
| `single` | Iterate `eval_tasks` once; all envs run the same task per pass. | `render` |

This is the eval mode used by `scripts/eval.sh` and the per-training eval configs under `configs/eval/`.

## Architecture
- `envs/` — Environment wrapper, config loader, task scheduler, bridge_multi env
- `training/` — PPO algorithm and replay buffer
- `main.py` — Training entry point (train + eval); `--policy` dispatches to the OpenVLA or SpatialVLA adapter
- `eval_only.py` — Standalone eval script (no training); same `--policy` dispatch + SpatialVLA chunk(K) path
- `configs/` — YAML training configs (incl. `spatialvla_2x2_*.yaml`)
- `configs/eval/` — Per-training eval configs (smaller `num_envs`, `task_order: sequence_random`)
- `scripts/` — Shell scripts for training and eval (incl. SpatialVLA launchers `p5_fresh_128ep.sh`, `p6_perep_320_evi1.sh`)
- `doc/` — Design documents (plan, reasoning) per version

The VLA policy adapters live in the `SimplerEnv` sibling: `SimplerEnv/simpler_env/policies/openvla/openvla_train.py` and `.../policies/spatialvla/spatialvla_train.py`. The SpatialVLA backbone is vendored as the `SpatialVLA/` sibling and installed editable by `setup.sh`.

## Version History

| Version | Key additions |
|---|---|
| V0.1 | Initial CRONOS: env wrapper, PPO, basic training |
| V0.2 | YAML configs, task scheduler, eval CSV, checkpoint/resume, config validation |
| V0.3 | Per-group objects/overlay, mixed N/M, sub-group fan-out, per-env rotation eval, config_history, eval_only.py |
| V0.3.1 | Standalone-eval refactor (`runner.eval` restored, AutoRL `render` port) — `eval_only.py` and `--eval-single` / `--eval-sequential` now use broadcast eval; `eval_mode=sequential` default; per-training eval configs in `configs/eval/`; non-episodic state preserved across mid-training eval via `get_env_state` / `set_env_state`; HSR respawn now uses per-env active-slot indexing (was hardcoded to V0.1's (N=2,M=1) layout); first-frame GPU-sync fix at episode init; `T2560` horizon added to `scripts/train.sh` |
| SpatialVLA | [SpatialVLA](#spatialvla-policy) integrated as a first-class, PPO-trainable policy behind the SimplerEnv seam — `--policy {openvla,spatialvla}` dispatch in `main.py` / `eval_only.py`; SpatialVLA adapter (`policies/spatialvla/spatialvla_train.py`); 3-token action decode + value head; chunk(K) open-loop eval (`--action-chunk`); `spatialvla_2x2_{train,smoke}.yaml` configs; backbone vendored as the `SpatialVLA/` sibling and installed by `setup.sh` (Blackwell-compatible torch 2.7 / transformers 4.47 pins); reproducible PPO launchers `scripts/p5_fresh_128ep.sh` + `scripts/p6_perep_320_evi1.sh`. OpenVLA path is unchanged (default). |

