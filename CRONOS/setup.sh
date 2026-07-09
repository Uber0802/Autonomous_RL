#!/bin/bash
# setup.sh - install Python dependencies for CRONOS + one or both VLA policies.
#
# Usage:
#   bash setup.sh [policy] [gpu]
#
#   [policy]: openvla | spatialvla | all | openvla_v01    (default: all)
#   [gpu]:    default | blackwell                          (default: default)
#
# Policy axis — picks which VLA(s) to install and which LM stack to pin:
#   * all          — dual-VLA stack + both OpenVLA & SpatialVLA pillars
#   * openvla      — dual-VLA stack + OpenVLA pillar only
#   * spatialvla   — dual-VLA stack + SpatialVLA pillar only
#   * openvla_v01  — lightweight stack + OpenVLA pillar only (cannot run --policy spatialvla)
#
# GPU axis — picks the torch wheel channel + version:
#   * default      — cu121 wheels (Ampere / Ada / Hopper, sm_80…sm_90)
#                    dual-VLA stack → torch 2.5.1+cu121 (SpatialVLA pyproject pin)
#                    lightweight stack → torch 2.2.0+cu121 (OpenVLA pyproject pin)
#   * blackwell    — cu128 wheels with torch 2.7.0+cu128 (sm_75…sm_120)
#                    The cu128 channel does NOT ship torch 2.2.0, so the V0.1
#                    Blackwell variant runs the lightweight pins on top of torch 2.7+cu128
#                    (functionally V0.1-compat but NOT bit-equivalent to V0.1 cu121
#                    numerics — different cuBLAS GEMM tile order + attention kernel).
#
# Recommended workflows (4-env 2×2 matrix):
#
#   cronos_env — dual-VLA, default GPU (Ampere/Ada/Hopper):
#       conda create -n cronos_env python=3.10 -y
#       conda activate cronos_env
#       cd Benchmark/CRONOS && bash setup.sh all
#
#   cronos_env_blackwell — dual-VLA, Blackwell:
#       conda create -n cronos_env_blackwell python=3.10 -y
#       conda activate cronos_env_blackwell
#       cd Benchmark/CRONOS && bash setup.sh all blackwell
#
#   cronos_env_lite — lightweight, default GPU (Ampere/Ada/Hopper; bit-exact to V0.1 baselines):
#       conda create -n cronos_env_lite python=3.10 -y
#       conda activate cronos_env_lite
#       cd Benchmark/CRONOS && bash setup.sh openvla_v01
#
#   cronos_env_lite_blackwell — lightweight, Blackwell:
#       conda create -n cronos_env_lite_blackwell python=3.10 -y
#       conda activate cronos_env_lite_blackwell
#       cd Benchmark/CRONOS && bash setup.sh openvla_v01 blackwell
#
# The script `cd`s to its own directory so the editable installs of the sibling
# pillars (`../ManiSkill`, `../SimplerEnv`, `../openvla`, `../SpatialVLA`)
# always resolve to the tree that contains THIS setup.sh — not whatever cwd the
# caller happened to be in. Don't symlink setup.sh from elsewhere.
#
# Pin rationale:
#
#   dual-VLA stack — SpatialVLA hard-pins transformers≥4.43 (its model code
#   imports HybridCache), so the dual-VLA stack is needed any time SpatialVLA
#   is in the picture. Side effect: OpenVLA-7B PPO peak rises from ~40 GB →
#   ~55 GB, which exceeds Ada-class GPU memory (so OpenVLA on Ada needs the
#   lightweight stack).
#     * transformers==4.47.0      — SpatialVLA's model files use APIs from 4.47
#                                    (GenerationMixin layout, Gemma2 logits format).
#                                    Hard ceiling: <4.50 (GenerationMixin removed).
#     * peft==0.14.0              — required by SpatialVLA's adapter layer types.
#     * tokenizers==0.21.0        — matches transformers 4.47 wheel ABI.
#     * torch (cu121 path)        — 2.5.1, matches SpatialVLA's pyproject pin.
#     * torch (cu128 path)        — 2.7.0, lowest stable cu128 build with sm_120.
#
#   lightweight stack (openvla_v01) — OpenVLA-only, mirrors the original cronos_env
#   pins. The SpatialVLA install is skipped entirely; its lazy import in
#   `main.py:270-271` / `eval_only.py:140` is never fired under --policy openvla.
#     * transformers==4.40.1      — OpenVLA pyproject pin; below 4.43, so the
#                                    HybridCache import in SpatialVLA's
#                                    modeling_gemma2.py would fail — intentional,
#                                    since we don't install SpatialVLA.
#     * peft==0.11.1              — OpenVLA pyproject pin.
#     * tokenizers==0.19.1        — OpenVLA pyproject pin.
#     * torch (cu121 path)        — 2.2.0, matches V0.1 baseline; sm_50..sm_90 only.
#                                    Keeps OpenVLA-7B PPO bit-identical to V0.1.
#     * torch (cu128 path)        — 2.7.0; cu128 doesn't ship torch 2.2.0 (cu128
#                                    wheels start at torch 2.7). Newer torch
#                                    changes cuBLAS/attention kernels → NOT
#                                    bit-equivalent to V0.1 cu121 baseline runs,
#                                    but transformers/peft ABI still V0.1-stable.
#
# Shared across both LM stacks:
#     * tensorflow==2.15.0 + tensorflow-datasets==4.9.3 (OpenVLA pulls TFDS in for
#       get_action_stats; both stacks need this).
#     * tensorflow-metadata<1.21  — versions ≥1.21 require `protobuf>=5.26`
#                                    (`from google.protobuf import runtime_version`),
#                                    but tensorflow==2.15.0's transitive pin caps
#                                    protobuf at <5. Mismatch surfaces as `ImportError`
#                                    on any `import tensorflow_datasets`. Cap at <1.21
#                                    keeps the wheels in sync.
#
# OpenVLA's setup.py declares the V0.1-style pins (torch 2.2.0, transformers 4.40.1,
# tokenizers 0.19.1); on the dual-VLA stack pip prints a dep-conflict warning but
# the runtime API is compatible (verified end-to-end on cronos_env and by the
# post-install sanity check at the bottom of this script). On the lightweight stack
# with cu121 the pins match exactly and there is no warning.

set -e

# Anchor at this script's directory so the `../*` editable installs are
# unambiguous regardless of the caller's cwd. The earlier setup.sh assumed
# the caller had `cd Benchmark/CRONOS` first, which produced silently-wrong
# editable installs when run from V0.1/CRONOS or any sibling tree.
cd "$(dirname "$(readlink -f "$0")")"
echo "[setup.sh] Working from: $(pwd)"

POLICY=${1:-all}
GPU=${2:-default}

case $POLICY in
  openvla|spatialvla|all|openvla_v01) ;;
  *) echo "Unknown policy: $POLICY"
     echo "Valid: openvla | spatialvla | all | openvla_v01"
     exit 1
     ;;
esac
case $GPU in
  default|blackwell) ;;
  *) echo "Unknown gpu: $GPU"
     echo "Valid: default | blackwell"
     exit 1
     ;;
esac

# Pick LM pins by POLICY (V0.1 vs V0.4 stack) and torch pin by GPU class
# (default = cu121, blackwell = cu128). Pillar selection (OpenVLA, SpatialVLA)
# is by POLICY as well.

# ----- LM stack (transformers / peft / tokenizers / accelerate) ----------
case $POLICY in
  openvla_v01)
    LM_STACK="V0.1"
    LM_PINS=("transformers==4.40.1" "accelerate==0.32.1" "peft==0.11.1" "tokenizers==0.19.1")
    INSTALL_OPENVLA=1
    INSTALL_SPATIALVLA=0
    ;;
  openvla)
    LM_STACK="V0.4"
    LM_PINS=("transformers==4.47.0" "accelerate==1.0.1" "peft==0.14.0" "tokenizers==0.21.0")
    INSTALL_OPENVLA=1
    INSTALL_SPATIALVLA=0
    ;;
  spatialvla)
    LM_STACK="V0.4"
    LM_PINS=("transformers==4.47.0" "accelerate==1.0.1" "peft==0.14.0" "tokenizers==0.21.0")
    INSTALL_OPENVLA=0
    INSTALL_SPATIALVLA=1
    ;;
  all)
    LM_STACK="V0.4"
    LM_PINS=("transformers==4.47.0" "accelerate==1.0.1" "peft==0.14.0" "tokenizers==0.21.0")
    INSTALL_OPENVLA=1
    INSTALL_SPATIALVLA=1
    ;;
esac

# ----- torch wheel (depends on both POLICY and GPU) ----------------------
# cu121 channel: torch 2.2.0 (lightweight) / 2.5.1 (dual-VLA).
# cu128 channel: torch 2.7.0 for both LM stacks — cu128 wheels do not exist
# for torch 2.2 or 2.5; the lightweight blackwell variant therefore runs the
# V0.1 transformers/peft pins on top of torch 2.7 (NOT bit-equivalent to V0.1
# cu121 numerics, but transformers/peft ABI still V0.1).
if [ "$GPU" = "blackwell" ]; then
    TORCH_PKGS="torch==2.7.0 torchvision"
    TORCH_INDEX="https://download.pytorch.org/whl/cu128"
    GPU_NOTE="Blackwell (sm_120, cu128)"
elif [ "$LM_STACK" = "V0.1" ]; then
    TORCH_PKGS="torch==2.2.0 torchvision==0.17.0"
    TORCH_INDEX="https://download.pytorch.org/whl/cu121"
    GPU_NOTE="Ampere/Ada/Hopper (sm_80..sm_90, cu121)"
else  # V0.4 + default
    TORCH_PKGS="torch==2.5.1 torchvision==0.20.1"
    TORCH_INDEX="https://download.pytorch.org/whl/cu121"
    GPU_NOTE="Ampere/Ada/Hopper (sm_80..sm_90, cu121)"
fi

STACK_LABEL="LM=$LM_STACK / GPU=$GPU_NOTE / policy=$POLICY"
echo "[setup.sh] Stack: $STACK_LABEL"

# Fail fast if no conda env is active (otherwise pip silently installs into the
# system / base interpreter — easy to lose 10 minutes on a wrong target env).
if [ -z "$CONDA_DEFAULT_ENV" ] || [ "$CONDA_DEFAULT_ENV" = "base" ]; then
    # Suggested env name by the 2x2 matrix (LM stack × GPU class).
    if [ "$LM_STACK" = "V0.1" ]; then
        if [ "$GPU" = "blackwell" ]; then SUGGESTED_ENV="cronos_env_lite_blackwell"
        else                              SUGGESTED_ENV="cronos_env_lite"
        fi
    else
        if [ "$GPU" = "blackwell" ]; then SUGGESTED_ENV="cronos_env_blackwell"
        else                              SUGGESTED_ENV="cronos_env"
        fi
    fi
    echo "[setup.sh] WARNING: no non-base conda env is active."
    echo "                  Activate the target env first, e.g."
    echo "                    conda create -n $SUGGESTED_ENV python=3.10 -y"
    echo "                    conda activate $SUGGESTED_ENV"
    echo "                  Continuing in 5 s — Ctrl-C to abort."
    sleep 5
fi

echo "[setup.sh] Installing CRONOS core dependencies (policy=$POLICY, gpu=$GPU) …"

# 1. Core dependencies. Torch + LM pins picked above per POLICY × GPU.
pip install "setuptools<70.0.0"
pip install $TORCH_PKGS --index-url $TORCH_INDEX
pip install "numpy<2.0.0" \
    "${LM_PINS[@]}" \
    einops==0.8.0 scipy==1.14.1 \
    "timm>=0.9.10,<1.0.0" draccus rich \
    gymnasium==0.29.1 tyro wandb tqdm transforms3d sapien==3.0.0.b1 \
    dacite h5py tabulate trimesh imageio "imageio[ffmpeg]" \
    "mplib==0.1.1" "fast_kinematics==0.2.2" IPython \
    "pytorch_kinematics==0.7.5" pynvml
pip install tensorflow==2.15.0 tensorflow-datasets==4.9.3 tensorflow-graphics \
    "tensorflow-metadata<1.21" "protobuf>=3.20,<5"

# 2. Plotting tooling (`scripts/plot.py` needs pandas + matplotlib at pinned versions).
pip install -r scripts/requirements_plot.txt

# 3. Pillar packages used by every policy (env, manipulation simulator, shared adapters).
pip install --no-deps -e ../ManiSkill
pip install --no-deps -e ../SimplerEnv

# 4. Policy-specific backbones.
if [ "$INSTALL_OPENVLA" = "1" ]; then
    echo "[setup.sh] Installing OpenVLA backbone …"
    pip install --no-deps "dlimp @ git+https://github.com/moojink/dlimp_openvla"
    pip install --no-deps -e ../openvla
fi

if [ "$INSTALL_SPATIALVLA" = "1" ]; then
    echo "[setup.sh] Installing SpatialVLA backbone …"
    # SpatialVLA is vendored as a sibling dir; install editable so the integration
    # code in CRONOS/SimplerEnv can import `model.modeling_spatialvla` directly.
    # `--no-deps` because section 1 already covers its runtime deps, and pinning
    # torch==2.5.1+cu121 from its pyproject would defeat the Blackwell-compatible
    # torch pin chosen above.
    pip install --no-deps -e ../SpatialVLA
fi

# 5. Post-install sanity check — fails fast on the exact issues that bit us
#    before (protobuf/tensorflow_metadata mismatch; simpler_env editable pointing
#    at a pre-merge tree; missing act_token_len on OpenVLAPolicy).
echo "[setup.sh] Verifying core imports …"
python - <<'PY'
import sys

print(f"  python      {sys.version.split()[0]}")

import torch
print(f"  torch       {torch.__version__}  CUDA={torch.version.cuda}")
assert torch.cuda.is_available(), "CUDA not visible from python — bad driver / wrong CUDA env"

import tensorflow as _tf
print(f"  tensorflow  {_tf.__version__}")
import tensorflow_datasets as _tfds
print(f"  tfds        {_tfds.__version__}  (protobuf import path OK)")

import simpler_env
print(f"  simpler_env {simpler_env.__path__[0]}")
from simpler_env.policies.openvla.openvla_train import OpenVLAPolicy
assert hasattr(OpenVLAPolicy, "act_token_len"), \
    "OpenVLAPolicy has no act_token_len — simpler_env editable points at a pre-merge tree"
print(f"  OpenVLA     act_token_len={OpenVLAPolicy.act_token_len}")

# SpatialVLA check is conditional on the policy being installed. On the V0.1
# stack (openvla_v01) the spatialvla pillar is not installed, so this block
# is skipped — exactly the intended behavior.
import importlib
if importlib.util.find_spec("spatialvla") is not None:
    from simpler_env.policies.spatialvla.spatialvla_train import SpatialVLAPolicy
    print(f"  SpatialVLA  act_token_len={SpatialVLAPolicy.act_token_len}")

print("[setup.sh] All core imports OK.")
PY

echo "[setup.sh] Done (policy=$POLICY, gpu=$GPU)."
