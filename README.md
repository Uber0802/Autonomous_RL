# CogACT-RL

CogACT VLM + Gaussian Action Head + PPO for robotic manipulation. Built on [CRONOS/AutoRL](../AutoRL/) by replacing OpenVLA with CogACT's vision-language backbone.

## Table of Contents
- [Overview](#overview)
- [Install](#install)
- [Quick Start](#quick-start)
- [BC Pretraining](#bc-pretraining)
- [Train](#train)
  - [Basic Configs](#basic-configs)
  - [Environment Configs](#environment-configs)
  - [Training Configs](#training-configs)
  - [CogACT-specific Configs](#cogact-specific-configs)
  - [Forward Backward](#forward-backward)
  - [FIFO Buffer](#fifo-buffer)
  - [Example](#example)
- [Evaluate](#evaluate)
- [Code Structure](#code-structure)
- [Architecture](#architecture)
- [Troubleshooting](#troubleshooting)

## Overview

This project replaces OpenVLA's discrete token-based action prediction with CogACT's VLM backbone + a Gaussian action head, enabling PPO-based RL fine-tuning.

**Key idea**: CogACT's VLM produces a 4096-dim "cognition token" encoding scene understanding. We discard CogACT's DiT diffusion head (incompatible with PPO) and attach a lightweight Gaussian policy head that outputs continuous actions with exact log-probabilities.

```
CogACT VLM (frozen + LoRA)     Gaussian Head (trainable)     Value Head (trainable)
Image + Instruction --> cognition token [4096] --+---> N(mean, std) --> action [7]
                                                  +--> MLP --> V(s) [1]
```

## Install

### Prerequisites
- Python 3.10
- CUDA 12.1+ (GPU with >= 24GB VRAM recommended)
- ~35GB disk for CogACT-Base checkpoint (downloaded on first run)

### Setup

```bash
# 1. Clone / copy CogACT_RL
cd CogACT_RL

# 2. Create conda environment
conda create -n cogact python=3.10 -y
conda activate cogact

# 3. Run setup (installs all dependencies + patches prismatic)
bash setup.sh
```

`setup.sh` handles:
- PyTorch 2.2 + CUDA 12.1
- OpenVLA package (provides Prismatic VLM backbone)
- CogACT model code
- flash-attn (compiled from source to match torch version)
- ManiSkill + SimplerEnv (simulation environments)
- Auto-patches prismatic to use bundled `llama2_local/` (avoids gated Llama-2 download)

### First-time model download

CogACT-Base (~29GB) downloads from HuggingFace on first run. If rate-limited, login first:
```bash
huggingface-cli login
```

### Verify installation

```bash
cd SimplerEnv
CUDA_VISIBLE_DEVICES=0 python -c "
from simpler_env.policies.cogact.cogact_train import CogACTPolicy
import types, torch, numpy as np
args = types.SimpleNamespace(
    vla_path='CogACT/CogACT-Base', vla_lora_rank=16, vla_load_path=None,
    vla_optim_beta1=0.9, vla_optim_beta2=0.999, vla_vhlr=3e-3, vla_lr=1e-4,
    cogact_action_model_type='DiT-B', cogact_future_window=15, bc_init_path=None,
)
policy = CogACTPolicy(args, device_id=0)
obs = {
    'image': torch.from_numpy(np.random.randint(0,255,(2,224,224,3),dtype=np.uint8)),
    'task_description': ['pick up cube', 'move block'],
}
policy.prep_rollout()
v, a, l = policy.get_action(obs, deterministic=True)
print(f'values={v.shape}, actions={a.shape}, logprobs={l.shape}')
print('Install OK')
"
```

## Quick Start

```bash
# Step 1: BC pretrain Gaussian head from DiT (~45 min)
bash bc_pretrain.sh

# Step 2: RL training with BC-initialized head
bash train_cogact.sh --bc_init_path ../bc_checkpoints/gaussian_head_init/bc_init.pt

# Or: RL training without BC (ablation -- starts from random head)
bash train_cogact.sh
```

## BC Pretraining

The Gaussian action head starts randomly initialized. Unlike OpenVLA (which has a pretrained action decoder), our head needs warmstarting before RL can get useful reward signal.

We distill CogACT's DiT action head into the Gaussian head:

```
For each observation in ManiSkill env:
  VLM --> cognition token
      |-- DiT (teacher, frozen) --> expert action
      +-- Gaussian head (student) --> predicted action
          Loss = MSE(predicted, expert)
```

### Usage

```bash
# Default (PutCarrotOnPlate, ~45 min on 1 GPU)
bash bc_pretrain.sh

# Different task
bash bc_pretrain.sh --env_id TwoObjectTwoReceptacle-v1 \
    --save_path ../bc_checkpoints/two_obj_init

# Custom parameters
bash bc_pretrain.sh \
    --num_envs 32 \
    --collect_steps 300 \
    --bc_steps 5000 \
    --bc_lr 1e-4
```

### BC Configs

| Param | Default | Description |
|-------|---------|-------------|
| `--env_id` | PutCarrotOnPlateInScene-v1 | ManiSkill environment |
| `--num_envs` | 32 | Parallel environments for data collection |
| `--collect_steps` | 300 | Steps to collect DiT demonstrations (300 x 32 = 9600 samples) |
| `--bc_steps` | 5000 | Gaussian head training steps |
| `--bc_lr` | 1e-4 | Learning rate |
| `--bc_batch_size` | 32 | Batch size |
| `--replay_size` | 20000 | Replay buffer capacity |
| `--dit_cfg_scale` | 1.5 | Classifier-free guidance scale for DiT |
| `--dit_ddim_steps` | 5 | DDIM sampling steps for DiT |
| `--save_path` | ../bc_checkpoints/gaussian_head_init | Output directory |

Output: `bc_init.pt` containing action_head + value_head state_dicts.

## Train

All training uses `train_ms3_ppo.py` via launch scripts.

### CogACT training

```bash
# Modify train_cogact.sh as needed, then:
bash train_cogact.sh
```

### OpenVLA training (original AutoRL)

```bash
# Use the original train.sh
bash train.sh
```

### Basic Configs
| Param | Default | Description |
|-------|---------|-------------|
| `--policy_type` | openvla | `openvla` or `cogact` |
| `--name` | "" | WandB run name |
| `--log` | "" | Path to log file |
| `--seed` | 0 | Random seed |
| `--no_wandb` | false | Disable WandB logging |
| `--vla_load_path` | "" | Resume from checkpoint |
| `--bc_init_path` | "" | BC-pretrained Gaussian head (CogACT only) |

### Environment Configs
| Param | Default | Description |
|-------|---------|-------------|
| `--env_id` | TwoObjectTwoReceptacle-v1 | ManiSkill environment |
| `--num_envs` | 64 | Number of parallel environments |
| `--obj_set` | rand | Scene layout: `fixed`, `rand`, `rand_8`, `rand_ood` |
| `--obj1_index` | 7 | Object index (see Object List below) |
| `--plate1_index` | 1 | Receptacle index (see Plate List below) |

### Training Configs
| Param | Default | Description |
|-------|---------|-------------|
| `--max_episodes` | 32 | Total training episodes |
| `--training_len` | 80 | Rollout steps per episode |
| `--training_interval` | 80 | Steps between VLA updates |
| `--instruction_switch_interval` | 80 | Steps between task instruction changes |
| `--alg_name` | ppo | Algorithm: `ppo` or `grpo` |
| `--alg_ppo_epoch` | 1 | PPO epochs per update |
| `--alg_gradient_accum` | 20 | Gradient accumulation steps |
| `--alg_entropy_coef` | 0.0 | Entropy bonus coefficient |
| `--buffer_minibatch` | 8 | Minibatch size for PPO |
| `--buffer_gamma` | 0.99 | Discount factor |
| `--buffer_lambda` | 0.95 | GAE lambda |

### CogACT-specific Configs
| Param | Default | Description |
|-------|---------|-------------|
| `--vla_path` | CogACT/CogACT-Base | CogACT model checkpoint |
| `--vla_lora_rank` | 32 | LoRA rank for VLM fine-tuning |
| `--vla_lr` | 1e-4 | Learning rate for LoRA + action head |
| `--vla_vhlr` | 3e-3 | Learning rate for value head |
| `--vla_unnorm_key` | bridge_orig | Dataset stats key for action denormalization |

### Forward Backward
| Param | Default | Description |
|-------|---------|-------------|
| `--enable_backward` | false | Enable forward-backward training |
| `--backward_interval` | 1 | Forward instructions between backward |

### FIFO Buffer
| Param | Default | Description |
|-------|---------|-------------|
| `--fifo_buffer` | false | Enable FIFO replay buffer |
| `--fifo_length` | 5 | Max trajectories in buffer |

### Example

CogACT with BC init, forward-backward, 1280-step episodes:
```bash
cd SimplerEnv
CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_PREALLOCATE=false \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TOKENIZERS_PARALLELISM=false \
python simpler_env/train_ms3_ppo.py \
    --policy_type cogact \
    --vla_path CogACT/CogACT-Base \
    --vla_unnorm_key bridge_orig \
    --vla_lora_rank 32 \
    --bc_init_path ../bc_checkpoints/gaussian_head_init/bc_init.pt \
    --env_id TwoObjectTwoReceptacle-v1 \
    --num_envs 64 \
    --training_len 1280 \
    --max_episodes 8 \
    --enable_backward --backward_interval 1 \
    --reset_unsuitable \
    --seed 0 \
    --name "CogACT-PPO-FB-1280" \
    --obj_set rand
```

## Evaluate

Modify `vla_load_paths` in `eval_single.sh` or `eval_seq.sh`, then:

```bash
# Single task evaluation
bash eval_single.sh

# Sequential task evaluation
bash eval_seq.sh
```

Use the same `--seed` across all evaluations to ensure consistent object placement.

## Code Structure

```
CogACT_RL/
+-- CogACT/                          # CogACT model code (local copy)
|   +-- vla/cogactvla.py             # CogACT main class + cognition extraction
|   +-- vla/load.py                  # Model loading (patched for inference-mode)
|   +-- action_model/                # DiT action head (used as BC teacher only)
+-- llama2_local/                    # Bundled Llama-2 config + tokenizer (2.3MB)
+-- ManiSkill/                       # ManiSkill simulation engine
|   +-- mani_skill/envs/tasks/       # Environment definitions
+-- SimplerEnv/simpler_env/
|   +-- train_ms3_ppo.py             # Main training loop (Runner class)
|   +-- bc_pretrain.py               # BC pretraining: DiT -> Gaussian head
|   +-- policies/
|   |   +-- cogact/
|   |   |   +-- cogact_model.py      # GaussianActionHead + ValueHead
|   |   |   +-- cogact_train.py      # CogACTPolicy + CogACTPPO
|   |   +-- openvla/
|   |       +-- openvla_train.py     # OpenVLAPolicy + OpenVLAPPO (original)
|   +-- env/simpler_wrapper.py       # Env wrapper (discrete + continuous actions)
|   +-- utils/replay_buffer.py       # Replay buffers + GAE
+-- openvla/                         # OpenVLA package (provides Prismatic backbone)
+-- setup.sh                         # Full installation script
+-- bc_pretrain.sh                   # BC pretraining launch script
+-- train_cogact.sh                  # CogACT RL training launch script
+-- train.sh                         # OpenVLA RL training launch script (original)
```

## Architecture

### CogACT vs OpenVLA in RL

| Aspect | OpenVLA (AutoRL) | CogACT (this repo) |
|--------|-----------------|---------------------|
| VLM backbone | Prismatic (SigLIP + DINOv2 + Llama-2 7B) | Same |
| Action head | LLM token prediction (discrete) | Gaussian MLP (continuous) |
| Action output | 7 token IDs -> bin centers | 7 continuous values in [-1, 1] |
| Log-prob | Softmax per token | Gaussian log-prob (exact) |
| RL algorithm | PPO | PPO (identical) |
| Gradient | LoRA on VLM + value head | LoRA on VLM + action head + value head |
| BC warmstart | Not needed (SFT'd action decoder) | DiT distillation recommended |

### Cognition Token

The core of CogACT's architecture -- a 4096-dim vector encoding scene + task understanding:

```
Image + Instruction
  -> Vision Backbone (DINOv2 + SigLIP)
  -> LLM (Llama-2 7B)
  -> hidden_states[-1]           [B, seq_len, 4096]
  -> remove vision patch tokens  [B, text_len, 4096]
  -> last valid token (EOS)      [B, 1, 4096]
  = cognition token
```

### Two Optimizers

| Optimizer | lr | Parameters | Rationale |
|-----------|-----|-----------|-----------|
| `vh_optimizer` | 3e-3 | Value head | V(s) starts random, needs fast learning |
| `vla_optimizer` | 1e-4 | LoRA + action head | Small lr to protect pretrained VLM |

## Troubleshooting

### `401 Unauthorized` when loading CogACT

`setup.sh` auto-patches this. If loading manually, see [FAQ Q1](../FAQ.md#q1).

### `flash_attn` undefined symbol error

Recompile flash-attn or disable it. See [FAQ Q2](../FAQ.md#q2).

### `ScalarType BFloat16` error

Your GPU doesn't support bf16 (V100, T4). The code auto-detects this and falls back to fp16. If you still see this error, pull the latest `cogact_train.py`.

### HuggingFace rate limit

```bash
huggingface-cli login
```

### Out of memory

Reduce `--num_envs` (e.g., 32 instead of 64) or use a GPU with more VRAM.

## Object List

| Index | Object | Index | Object |
|-------|--------|-------|--------|
| 1 | carrot | 14 | hamburger |
| 2 | kitchen shovel | 15 | golf ball |
| 3 | bread | 16 | BBQ sauce |
| 4 | plastic bottle | 17 | travel cup |
| 5 | 7up can | 18 | pepper |
| 6 | zucchini | 19 | nonstop can |
| 7 | ketchup bottle | 20 | potato |
| 8 | watering can | 21 | baguette |
| 9 | pipe | 22 | champagne glass |
| 10 | toy bear | 23 | kitchen spoon |
| 11 | fast food cup | 24 | onion |
| 12 | plant | 25 | cup |
| 13 | banana | | |

## Plate List

| Index | Plate | Index | Plate |
|-------|-------|-------|-------|
| 1 | yellow_plate | 10 | gramophone disk |
| 2 | cloth | 11 | frying pan |
| 3 | carpet | 12 | mouse pad |
| 4 | newspaper | 13 | cutting board |
| 5 | sheet metal | 14 | chess board |
| 6 | drawing tablet | 15 | manhole cover |
| 7 | tomato slice | 16 | envelope |
| 8 | pizza | 17 | notepad |
| 9 | flat bowl | 18 | black_plate |
