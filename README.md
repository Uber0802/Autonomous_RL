# CRONOS × UniVLA

CRONOS extended with **UniVLA** as a switchable VLA backbone. Select the backend with a single flag:

```
--vla_type openvla   # original behaviour, unchanged
--vla_type univla    # UniVLA (VQ-VAE latent actions + ActionDecoder)
```

The original OpenVLA training path is unmodified. All code changes are additive.

---

## Quick Start

### Clone the repository

```bash
git clone https://github.com/Uber0802/Autonomous_RL.git -b cronos_univla
cd Autonomous_RL
```

> **Note:** Model weights are **not** included in this repository due to size constraints. Follow the [Model Weights](#model-weights) section below to download them after installation.

---

## Repository Layout

```
cronos_univla/
├── CRONOS/          # RL runner (main.py, PPO, buffer, train.sh, eval.sh, setup.sh)
├── ManiSkill/       # Simulation engine — not modified
├── SimplerEnv/      # Policy wrappers; univla_train.py lives here
│   └── simpler_env/policies/univla/univla_train.py
├── openvla/         # Prismatic + ValueHead for OpenVLA — not modified
└── UniVLA/          # Minimal UniVLA subtree (prismatic, action decoder, checkpoint)
    ├── prismatic/extern/hf/modeling_prismatic.py   # + ValueHead + UniVLAForActionPredictionWithValueHead
    ├── univla_action_decoder.py                    # ActionDecoder/Head, TF-free
    └── qwbu__univla-7b-224-sft-simpler-bridge/     # checkpoint directory
```

All training commands run from **`CRONOS/`**.

---

## Requirements

- Linux, NVIDIA GPU (Ampere/Ada/Hopper; see [GPU Compatibility](#gpu-compatibility) for Blackwell)
- Conda
- ~14 GB free disk space for UniVLA weights; ~15 GB for OpenVLA weights

---

## Installation

### 1. Create a Conda environment

```bash
conda create -n cronos-univla python=3.10 -y
conda activate cronos-univla
```

### 2. Run `setup.sh`

```bash
cd /path/to/cronos_univla/CRONOS
bash setup.sh
```

`setup.sh` installs PyTorch 2.2 (CUDA 12.1), ManiSkill, SimplerEnv, openvla, and UniVLA in editable mode. UniVLA is installed **last** so its `prismatic` package takes precedence over openvla's.

### 3. Verify the install

```bash
python -c "
import sys, pathlib
sys.path.insert(0, str(pathlib.Path('.').resolve().parent / 'UniVLA'))
from prismatic.extern.hf.modeling_prismatic import UniVLAForActionPredictionWithValueHead
print('OK:', UniVLAForActionPredictionWithValueHead)
"
```

Expected: `OK: <class '...UniVLAForActionPredictionWithValueHead'>`. If you get `ImportError`, re-run `pip install -e ../UniVLA` inside the activated environment.

---

## Model Weights

### UniVLA — `qwbu/univla-7b-224-sft-simpler-bridge`

**The checkpoint must be downloaded before running.** `UniVLAPolicy.__init__` calls `torch.load(vla_path / "action_decoder.pt")` — `torch.load` requires a local file path and cannot fetch from the HuggingFace Hub. The VLA backbone and processor use `from_pretrained` and could download on first run, but the `action_decoder.pt` load will raise `FileNotFoundError` if the checkpoint is not already local.

The checkpoint is expected at `UniVLA/qwbu__univla-7b-224-sft-simpler-bridge/` (relative to `cronos_univla/`). If `.safetensors` files are already present there, skip this step.

```bash
python -c "
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='qwbu/univla-7b-224-sft-simpler-bridge',
    local_dir='../UniVLA/qwbu__univla-7b-224-sft-simpler-bridge',
    local_dir_use_symlinks=False,
    ignore_patterns=['*.msgpack', '*.h5', 'flax_model*', 'tf_model*', 'rust_model*'],
)
print('Done.')
"
```

Verify:

```bash
ls ../UniVLA/qwbu__univla-7b-224-sft-simpler-bridge/*.safetensors | wc -l
# expected: 3
ls ../UniVLA/qwbu__univla-7b-224-sft-simpler-bridge/action_decoder.pt
# must exist
```

> If you already have `action_decoder.pt` locally at a different path, you can skip downloading the full checkpoint and pass `--univla_decoder_path /absolute/path/to/action_decoder.pt` alongside any `--vla_path` (local or HF repo ID). The VLA backbone will still be fetched via `from_pretrained` if not cached.

### OpenVLA — `openvla/openvla-7b` (needed only for `--vla_type openvla`)

The OpenVLA path in `train.sh` resolves to `CRONOS/openvla/openvla-7b/`. Download if not already present:

```bash
python -c "
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='openvla/openvla-7b',
    local_dir='openvla/openvla-7b',
    local_dir_use_symlinks=False,
    ignore_patterns=['*.msgpack', '*.h5', 'flax_model*', 'tf_model*'],
)
print('Done.')
"
```

Alternatively, symlink an existing copy:

```bash
ln -s /path/to/existing/openvla-7b openvla/openvla-7b
```

---

## Pre-flight Check (recommended)

Run this before any GPU job to catch import errors and checkpoint issues in under 30 seconds, with no GPU required:

```bash
python smoke_test_logs/preflight.py
```

Expected final line: `=== Pre-flight PASSED — all imports and checkpoint checks succeeded ===`

If any `[FAIL]` line appears, fix that issue before continuing. Common causes:

- `[FAIL] univla_action_decoder` — run `pip install -e ../UniVLA` in the active environment.
- `[FAIL] action_decoder.pt window_size` — re-check `--univla_window_size`; the correct value for `qwbu__univla-7b-224-sft-simpler-bridge` is **10**.
- `[FAIL] bridge_oxe` — the checkpoint download was incomplete; re-run the download step above.

---

## Training

All commands run from `CRONOS/`. Set `CUDA_VISIBLE_DEVICES` to the GPU you want to use.

### OpenVLA backend

```bash
bash train.sh 80      # T=80: 80-step episodes, 16 envs
bash train.sh 320     # T=320: 320-step episodes, 16 envs
bash train.sh 1280    # T=1280: 1280-step episodes, 16 envs
```

Or explicitly:

```bash
CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_PREALLOCATE=false \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
PYTHONPATH=$(pwd) python main.py \
    --name "CRONOS-OpenVLA" --seed 0 \
    --env_id PickPlaceNxM-v1 --env_n 2 --env_m 2 \
    --vla_type openvla \
    --vla_path openvla/openvla-7b \
    --vla_unnorm_key bridge_orig \
    --num_envs 16 \
    --segment_len 80 --episode_len 80 --task_len 80 --ppo_update_len 80 \
    --max_episodes 32 --eval_interval 4 --vla_checkpoint_interval 32
```

### UniVLA backend

```bash
bash train.sh univla   # T=320, 16 envs; uses HF repo ID (downloads if not cached)
```

Or with the local checkpoint directory:

```bash
CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_PREALLOCATE=false \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
PYTHONPATH=$(pwd) python main.py \
    --name "CRONOS-UniVLA" --seed 0 \
    --env_id PickPlaceNxM-v1 --env_n 2 --env_m 2 \
    --vla_type univla \
    --vla_path ../UniVLA/qwbu__univla-7b-224-sft-simpler-bridge \
    --vla_unnorm_key bridge_oxe \
    --univla_window_size 10 \
    --num_envs 16 \
    --segment_len 80 --episode_len 320 --task_len 80 --ppo_update_len 160 \
    --max_episodes 32 --eval_interval 4 --vla_checkpoint_interval 8 \
    --reset_robot
```

`--vla_path` accepts either a local directory path or a HuggingFace repo ID. When a repo ID is given, `from_pretrained` downloads the model on first run.

Checkpoints are written to `wandb/offline-run-*/glob/episode_NNNN/`. Pass `--no-wandb` to disable W&B upload (logs still written to disk).

---

## Evaluation

### OpenVLA

```bash
CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_PREALLOCATE=false \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
PYTHONPATH=$(pwd) python main.py \
    --name "CRONOS-Eval" --seed 0 \
    --env_id PickPlaceNxM-v1 --env_n 2 --env_m 2 \
    --num_envs 16 \
    --vla_type openvla \
    --vla_path openvla/openvla-7b \
    --vla_unnorm_key bridge_orig \
    --vla_load_path /path/to/checkpoint/episode_NNNN \
    --eval_sequential \
    --eval_sequences 5 \
    --no-wandb
```

### UniVLA

```bash
CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_PREALLOCATE=false \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
PYTHONPATH=$(pwd) python main.py \
    --name "CRONOS-UniVLA-Eval" --seed 0 \
    --env_id PickPlaceNxM-v1 --env_n 2 --env_m 2 \
    --num_envs 16 \
    --vla_type univla \
    --vla_path ../UniVLA/qwbu__univla-7b-224-sft-simpler-bridge \
    --vla_unnorm_key bridge_oxe \
    --univla_window_size 10 \
    --vla_load_path /path/to/checkpoint/episode_NNNN \
    --eval_sequential \
    --eval_sequences 5 \
    --no-wandb
```

`--vla_load_path` points to the `episode_NNNN/` directory written during training. To find the latest:

```bash
ls -dt wandb/offline-run-*/glob/episode_*/ | head -1
```

Add `--record_video` to save per-episode video alongside the eval report.

---

## Checkpoint Testing (`test.sh`)

`test.sh` runs twelve reproducible configs — six for OpenVLA and six for UniVLA, each across two horizons (T80, T320) split into three segments (a/b/c) — so checkpoint resumption can be verified at every stage. All commands run from `CRONOS/`.

```
Usage: bash test.sh [MODE] [SEED] [CUDA]
```

| Argument | Default | Description |
|---|---|---|
| `MODE` | `t80a` | See mode table below |
| `SEED` | `0` | Random seed |
| `CUDA` | `0` | GPU index (`CUDA_VISIBLE_DEVICES`) |

#### Modes and step counts

| Backbone | Horizon | Mode | Episodes | Steps (this) | Steps (cumul.) |
|---|---|---|---|---|---|
| OpenVLA | T80 | `t80a` | 128 | 655,360 | 655,360 |
| OpenVLA | T80 | `t80b` | 128 | 655,360 | 1,310,720 |
| OpenVLA | T80 | `t80c` | 320 | 1,638,400 | 2,949,120 |
| OpenVLA | T320 | `t320a` | 32 | 655,360 | 655,360 |
| OpenVLA | T320 | `t320b` | 32 | 655,360 | 1,310,720 |
| OpenVLA | T320 | `t320c` | 80 | 1,638,400 | 2,949,120 |
| UniVLA | T80 | `univla_t80a` | 128 | 655,360 | 655,360 |
| UniVLA | T80 | `univla_t80b` | 128 | 655,360 | 1,310,720 |
| UniVLA | T80 | `univla_t80c` | 320 | 1,638,400 | 2,949,120 |
| UniVLA | T320 | `univla_t320a` | 32 | 655,360 | 655,360 |
| UniVLA | T320 | `univla_t320b` | 32 | 655,360 | 1,310,720 |
| UniVLA | T320 | `univla_t320c` | 80 | 1,638,400 | 2,949,120 |

#### Running a full sequence

**1. Run the first segment (no checkpoint needed):**

```bash
cd CRONOS
bash test.sh t320a        0 0   # OpenVLA T320, seed 0, GPU 0
bash test.sh univla_t320a 0 0   # UniVLA  T320, seed 0, GPU 0
```

**2. Resume from the previous segment's final checkpoint:**

OpenVLA T320:
```bash
CKPT_T320=runs/CRONOS-V0.2-T320-seed0/glob/episode_0032 \
    bash test.sh t320b 0 0

CKPT_T320=runs/CRONOS-V0.2-T320-seed0/glob/episode_0064 \
    bash test.sh t320c 0 0
```

UniVLA T320:
```bash
CKPT_UNIVLA_T320=runs/CRONOS-V0.2-UniVLA-T320-seed0/glob/episode_0032 \
    bash test.sh univla_t320b 0 0

CKPT_UNIVLA_T320=runs/CRONOS-V0.2-UniVLA-T320-seed0/glob/episode_0064 \
    bash test.sh univla_t320c 0 0
```

The same pattern applies to T80 horizons using `CKPT_T80` / `CKPT_UNIVLA_T80` with checkpoints at `episode_0128` and `episode_0256`.

> **Note:** Running a `*b` or `*c` segment without the matching `CKPT_*` variable set will print an error and exit before launching any training.

---

## Key CLI Flags

| Flag | Default | Description |
|---|---|---|
| `--vla_type` | `openvla` | VLA backend: `openvla` or `univla` |
| `--vla_path` | `openvla/openvla-7b` | Local directory or HF repo ID for the base checkpoint |
| `--vla_unnorm_key` | `bridge_orig` | Action normalization key in `dataset_statistics.json`. Use `bridge_orig` for OpenVLA, `bridge_oxe` for UniVLA |
| `--univla_window_size` | `10` | ActionDecoder temporal window. Must match the checkpoint's `proj.0.weight` shape (`70 // 7 = 10` for `qwbu__univla-7b-224-sft-simpler-bridge`) |
| `--univla_decoder_path` | `""` | Path to `action_decoder.pt`. Defaults to `<vla_path>/action_decoder.pt` when empty |
| `--vla_load_path` | `""` | Resume from a LoRA checkpoint directory |
| `--num_envs` | `64` | Number of parallel environments |
| `--buffer_inferbatch` | `32` | Per-forward-pass micro-batch size during rollout. Reduce to `16` if VRAM is tight |
| `--eval_interval` | `4` | Run in-training evaluation every N episodes |
| `--vla_checkpoint_interval` | varies | Save a LoRA checkpoint every N episodes |
| `--no-wandb` | — | Disable W&B upload; logs still written to `wandb/offline-run-*/glob/` |
| `--debug_rollout` | — | Print per-step `[ROLLOUT]` lines during iteration 0 |

---

## Troubleshooting

### `ValueError: action_decoder.pt window_size=73 but args.univla_window_size=10`

This is a false positive caused by an older code path. If you see it, confirm you are on the latest `univla_train.py`. The fix is that `proj_key` must be selected with `k.startswith("proj.")`, not `"proj" in k`. Check:

```bash
grep "startswith.*proj" ../SimplerEnv/simpler_env/policies/univla/univla_train.py
```

Expected: `k for k in decoder_sd if k.startswith("proj.")`.

The correct `window_size` for `qwbu__univla-7b-224-sft-simpler-bridge` is **10**. Pass `--univla_window_size 10`.

---

### `AttributeError: 'Tensor' object has no attribute 'convert'`

`_preprocess_obs` is passing a GPU tensor to the Prismatic image processor, which expects PIL images. This was fixed in `univla_train.py`. Confirm the fix is present:

```bash
grep "PILImage.fromarray" ../SimplerEnv/simpler_env/policies/univla/univla_train.py
```

Expected: at least one match. If absent, the file is outdated.

---

### `AssertionError: Generation is only currently supported for batch size of 1!`

`UniVLAForActionPredictionWithValueHead` is not being loaded — the base class is. Check that `sys.path` has `cronos_univla/UniVLA` ahead of any `openvla/` path, and that `AutoModelForVision2Seq.register(...)` runs before `from_pretrained`. Quick diagnostic:

```bash
python -c "
import sys, pathlib
sys.path.insert(0, str(pathlib.Path('../UniVLA').resolve()))
from prismatic.extern.hf.modeling_prismatic import UniVLAForActionPredictionWithValueHead
print('OK:', UniVLAForActionPredictionWithValueHead)
"
```

If this fails with `ImportError`, run `pip install -e ../UniVLA` in the active environment.

---

### `ValueError: Generation with batch size > 1 is not currently supported!`

Same root cause as above — the base `OpenVLAForActionPrediction` was loaded instead of `UniVLAForActionPredictionWithValueHead`. See the diagnostic above.

---

### `RuntimeError: mat1 and mat2 must have the same dtype, but got Float and BFloat16`

The `value_head` was not cast to `float32` after model load. This was fixed in `univla_train.py` `__init__` and `load()`. Confirm:

```bash
grep "value_head.to(torch.float32)" ../SimplerEnv/simpler_env/policies/univla/univla_train.py
```

Expected: two matches (one in `__init__`, one in `load()`).

---

### `KeyError: 'bridge_oxe'`

The checkpoint download was incomplete. Verify:

```bash
python -c "
import json
d = json.load(open('../UniVLA/qwbu__univla-7b-224-sft-simpler-bridge/dataset_statistics.json'))
print(list(d.keys()))   # expected: ['bridge_oxe']
"
```

If `bridge_oxe` is absent, re-run the download step. The `dataset_statistics.json` file itself is small and may have been corrupted; delete it and re-download.

---

### `load_state_dict` shape mismatch on `proj.0.weight` (eval step)

`action_decoder.pt` must be loaded via `dec.net.load_state_dict(...)`, not `dec.load_state_dict(...)`. This is already handled in `UniVLAPolicy.__init__` and `load()`. If you see this error, ensure you are running the latest `univla_train.py`.

---

### `ModuleNotFoundError: No module named 'prismatic'`

UniVLA is not installed in the active environment:

```bash
pip install -e ../UniVLA
```

If the error persists, verify `_UNIVLA_ROOT` resolves correctly:

```bash
python -c "
import pathlib
p = pathlib.Path('../SimplerEnv/simpler_env/policies/univla/univla_train.py').resolve()
root = p.parents[3] / 'UniVLA'
print('UNIVLA_ROOT:', root, '| exists:', root.exists())
"
```

---

### Eval log contains `ERRORS — check log` despite a clean run

TensorFlow's C++ backend prints an INFO message containing the word "errors" to stderr. This is suppressed by `os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')` at the top of `main.py`. If you see this in custom scripts, set the variable before any import:

```python
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
```

---

### ActionDecoder temporal buffer corrupts the first steps of a new eval task

`reset_action_decoders()` must be called at the start of each evaluation task. In `main.py`, `CronosRunner.eval()` calls it immediately after `policy.prep_rollout()` when `vla_type == "univla"`. If you are using a custom eval loop, add:

```python
if args.vla_type == "univla":
    policy.reset_action_decoders()
```

at the start of each task's evaluation.

---

### CUDA OOM during training

Lower `--buffer_inferbatch` to reduce peak activation memory:

```
--buffer_inferbatch 16
```

The default is 32. The model is loaded in bfloat16; the 7B backbone itself requires ~14 GB of VRAM before activations.

---

### `[PPO LOSS]` line missing

PPO did not run. Check that the rollout completed (look for a task-switch line near the end of the log) and that no exception was raised before the PPO call. With `--num_envs 2 --buffer_minibatch 8 --segment_len 16`, `total_batches = (16×2)//8 = 4` and `[PPO LOSS]` prints at `idx == 0`.

---

## GPU Compatibility

Verified on Ampere (A100), Ada (L40S), and Hopper (H100).

**Blackwell cards (RTX PRO 6000, RTX 5090, B200)** require a torch build with `sm_100`/PTX support. The default `cu121` wheels from `setup.sh` top out at `sm_90` and crash on first CUDA op:

```
RuntimeError: CUDA error: no kernel image is available for execution on the device
```

One-time fix — upgrade torch in-place after running `setup.sh`:

```bash
pip uninstall -y torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

If `flash-attn` breaks afterwards:

```bash
pip install flash-attn --no-build-isolation
```

PTX is JIT-compiled on first use on a new architecture. Training curves on Blackwell are statistically equivalent to Hopper but not bit-exact.
