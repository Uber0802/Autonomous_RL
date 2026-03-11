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

## Architecture
- `envs/`: Modural environment interaction and wrapper.
- `training/`: PPO algorithm and replay buffer logic.
- `main.py`: Central training entry point.
