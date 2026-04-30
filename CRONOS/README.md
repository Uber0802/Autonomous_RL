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

## Installation
The following steps assume a Linux environment with NVIDIA GPUs and Conda installed.

### 1. Setup Environment
Run the provided setup script:
```bash
bash setup.sh
```

### 2. Prepare Dependencies
Ensure the following directories are in your workspace:
- `Benchmark/SimplerEnv`
- `Benchmark/ManiSkill`
- `Benchmark/openvla`

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
| `--num-envs` | 64 | Total parallel environments |
| `--segment-len` | 80 | Steps per segment (AutoRL: 80) |
| `--ppo-update-len` | 80 | Steps between PPO updates |
| `--eval-interval` | 4 | Eval every N episodes |
| `--num-eval-episode` | 4 | Episodes per eval round |
| `--eval-at-start` | false | Run eval before first training episode |

### Evaluation (standalone)
```bash
# Eval-only using eval_only.py (no training, no PPO)
bash scripts/eval.sh
```

Or directly:
```bash
python eval_only.py \
  --config-path configs/two_group_sequential_2x2.yaml \
  --num-eval-episode 4 \
  --segment-len 80 \
  --vla-path openvla/openvla-7b \
  --vla-load-path path/to/checkpoint
```

Note: `--num-envs` is no longer needed on the CLI — it's computed from per-group `num_envs` in the config.

## YAML Config Format

Configs live in `configs/`. A single YAML file fully describes an experiment.

### Top-level fields

| Field | Type | Default | Description |
|---|---|---|---|
| `cronos_version` | string | — | Config format version (e.g. `V0.3`) |
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

## Eval Design (V0.3)

Per-env rotation formula:
```
task_for_env_i_at_episode_e = eval_tasks[(i % n_eval_tasks + e) % n_eval_tasks]
```

Samples per task = `num_eval_episode * group_envs / n_eval_tasks`.

Divisibility constraints (validated at config load):
- `group_envs % n_eval_tasks == 0`
- `n_eval_tasks % n_train_tasks == 0`

## Architecture
- `envs/` — Environment wrapper, config loader, task scheduler, bridge_multi env
- `training/` — PPO algorithm and replay buffer
- `main.py` — Training entry point (train + eval)
- `eval_only.py` — Standalone eval script (no training)
- `configs/` — YAML experiment configs
- `scripts/` — Shell scripts for training and eval
- `doc/` — Design documents (plan, reasoning) per version

## Version History

| Version | Key additions |
|---|---|
| V0.1 | Initial CRONOS: env wrapper, PPO, basic training |
| V0.2 | YAML configs, task scheduler, eval CSV, checkpoint/resume, config validation |
| V0.3 | Per-group objects/overlay, mixed N/M, sub-group fan-out, per-env rotation eval, config_history, eval_only.py |

## GPU Compatibility

CRONOS has been verified on Ampere (A100), Ada (L40S), and Hopper (H100). **Blackwell cards (RTX PRO 6000, RTX 5090, B200) require a different torch build** — the stock `cronos_envV0.1` ships wheels whose SASS kernels top out at `sm_90`, so the first CUDA op on a Blackwell device crashes with:

```
RuntimeError: CUDA error: no kernel image is available for execution on the device
```

### One-time fix: `cronos_envV0.1_blackwell`

Clone the working env and upgrade torch to the `cu128` channel, which ships kernels for `sm_75 ... sm_120 + PTX` and therefore runs on **every** GPU from Turing through Blackwell in a single env:

```bash
conda create -n cronos_envV0.1_blackwell --clone cronos_envV0.1
conda activate cronos_envV0.1_blackwell
pip uninstall -y torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

If `flash-attn` or `xformers` break afterwards (they link against the torch C++ ABI):

```bash
pip install flash-attn --no-build-isolation
pip install xformers --index-url https://download.pytorch.org/whl/cu128
```

**Why a clone and not an in-place upgrade of `cronos_envV0.1`:** openvla pins transformers/flash-attn versions compiled against the original torch ABI. Cloning isolates the blast radius — if something breaks, `cronos_envV0.1` remains intact on the non-Blackwell machines. The Blackwell-specific env is named with the `_blackwell` suffix so each machine uses the matching one.

**Reproducibility note:** PTX is JIT-compiled on first CUDA op on a new arch, so SASS may differ slightly from a native build. Training curves on Blackwell are *statistically* equivalent to Hopper, not bit-exact.
