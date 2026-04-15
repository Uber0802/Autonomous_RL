# CRONOS

CRONOS is a refactored robotic manipulation training benchmark designed for **non-episodic reinforcement learning** and **multi-task training**. It is built on the `RL4VLA` backbone with optimized modules from `AutoRL`.

## Features
- **Modular Environment**: Decoupled `reset_strategy`, `reward_shaping`, `task_suite`, and `task_scheduler`.
- **Efficient Rollouts**: Multi-task execution with GPU-parallelized ManiSkill environments.
- **Robust Training**: PPO/GRPO implementation with memory-mapped replay buffers.

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

## Workflow

### Training
To start multi-task PPO training:
```bash
bash train.sh
```

### Evaluation
To evaluate a trained policy:
```bash
bash eval.sh
```

## GPU Compatibility

CRONOS has been verified on Ampere (A100), Ada (L40S), and Hopper (H100). **Blackwell cards (RTX PRO 6000, RTX 5090, B200) require a different torch build** — the stock `cronos_envV0.1` ships wheels whose SASS kernels top out at `sm_90`, so the first CUDA op on a Blackwell device crashes with:

```
RuntimeError: CUDA error: no kernel image is available for execution on the device
```

### One-time fix: `cronos_envV0.1_blackwell`

Clone the working env and upgrade torch to the `cu128` channel, which ships kernels for `sm_75 … sm_120 + PTX` and therefore runs on **every** GPU from Turing through Blackwell in a single env:

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

## Architecture
- `envs/`: Modural environment interaction and wrapper.
- `training/`: PPO algorithm and replay buffer logic.
- `main.py`: Central training entry point.
