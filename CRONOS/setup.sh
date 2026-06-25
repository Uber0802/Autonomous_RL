#!/bin/bash
# setup.sh - install Python dependencies for CRONOS + one or both VLA policies.
#
# Usage:
#   bash setup.sh                # default: install BOTH OpenVLA + SpatialVLA stacks
#                                  (V0.4 stack; needs cronos_envV0.4 + Blackwell GPU)
#   bash setup.sh all            # same as above (explicit)
#   bash setup.sh openvla        # OpenVLA-only on V0.4 stack (cu128 + transformers 4.47)
#   bash setup.sh spatialvla     # SpatialVLA-only on V0.4 stack (cu128 + transformers 4.47)
#   bash setup.sh openvla_v01    # OpenVLA-only on V0.1 stack — torch 2.2.0+cu121 +
#                                  transformers 4.40.1. Lightweight (~40 GB OpenVLA
#                                  peak, fits Ada 48 GB GPUs; bit-exact V0.1 numerics).
#                                  Cannot run --policy spatialvla.
#
# Recommended workflows:
#
#   Dual-VLA on Blackwell (one env serves both policies):
#       conda create -n cronos_envV0.4 python=3.10 -y
#       conda activate cronos_envV0.4
#       cd Benchmark/CRONOS
#       bash setup.sh all
#
#   Lightweight OpenVLA-only on Ada (or for bit-exact V0.1 baselines):
#       conda create -n cronos_envV0.1 python=3.10 -y
#       conda activate cronos_envV0.1
#       cd Benchmark/CRONOS
#       bash setup.sh openvla_v01
#
# The script `cd`s to its own directory so the editable installs of the sibling
# pillars (`../ManiSkill`, `../SimplerEnv`, `../openvla`, `../SpatialVLA`)
# always resolve to the tree that contains THIS setup.sh — not whatever cwd the
# caller happened to be in. Don't symlink setup.sh from elsewhere.
#
# Two dependency stacks — picked by POLICY:
#
#   V0.4 stack (openvla | spatialvla | all): SpatialVLA hard-pins transformers≥4.43
#   (its model code imports HybridCache), so the V0.4 stack is needed any time
#   SpatialVLA is in the picture. Side effect: OpenVLA-7B PPO peak rises from
#   ~40 GB → ~55 GB, which exceeds Ada-class GPU memory.
#     * transformers==4.47.0      — SpatialVLA's model files use APIs from 4.47
#                                    (GenerationMixin layout, Gemma2 logits format).
#                                    Hard ceiling: <4.50 (GenerationMixin removed).
#     * torch==2.7.0+cu128        — sm_120 support for Blackwell (RTX PRO 6000).
#                                    SpatialVLA's own pyproject pins torch==2.5.1+cu121,
#                                    but +cu121 wheels are sm_50..sm_90 only.
#     * peft==0.14.0              — required by SpatialVLA's adapter layer types.
#     * tokenizers==0.21.0        — matches transformers 4.47 wheel ABI.
#
#   V0.1 stack (openvla_v01): OpenVLA-only, mirrors the original cronos_env pins.
#   Fits Ada 48 GB at ~40 GB OpenVLA peak and is numerically bit-equivalent to
#   the V0.1 baseline runs (same cuBLAS GEMM tile order + attention kernels).
#   The SpatialVLA install is skipped entirely; its lazy import in
#   `main.py:270-271` / `eval_only.py:140` is never fired under --policy openvla.
#     * torch==2.2.0+cu121        — matches V0.1 baseline; sm_50..sm_90 only
#                                    (no Blackwell — clone + retarget to cu128 if you
#                                    need V0.1 on Blackwell; see README §5).
#     * transformers==4.40.1      — V0.1 pin; below 4.43, so the HybridCache import
#                                    in SpatialVLA's modeling_gemma2.py would fail
#                                    — intentional, since we don't install SpatialVLA.
#     * peft==0.11.1              — V0.1 pin; compatible with transformers 4.40.
#     * tokenizers==0.19.1        — matches OpenVLA's pyproject pin exactly.
#
# Shared across both stacks:
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
# tokenizers 0.19.1); on the V0.4 stack pip prints a dep-conflict warning but the
# runtime API is compatible (verified end-to-end on cronos_envV0.4 and by the
# post-install sanity check at the bottom of this script). On the V0.1 stack the
# pins match exactly and there is no warning.

set -e

# Anchor at this script's directory so the `../*` editable installs are
# unambiguous regardless of the caller's cwd. The earlier setup.sh assumed
# the caller had `cd Benchmark/CRONOS` first, which produced silently-wrong
# editable installs when run from V0.1/CRONOS or any sibling tree.
cd "$(dirname "$(readlink -f "$0")")"
echo "[setup.sh] Working from: $(pwd)"

POLICY=${1:-all}
case $POLICY in
  openvla|spatialvla|all|openvla_v01) ;;
  *) echo "Unknown policy: $POLICY"
     echo "Valid: openvla | spatialvla | all | openvla_v01"
     exit 1
     ;;
esac

# Pick the dependency stack and which sibling pillars to install based on POLICY.
case $POLICY in
  openvla_v01)
    STACK_LABEL="V0.1 (lightweight; OpenVLA-only, fits Ada 48 GB)"
    # Pins below match OpenVLA's own pyproject.toml exactly, which is what
    # the original V0.1 cronos_env install resolved to. Keeps OpenVLA-7B PPO
    # forward bit-identical to V0.1 baseline runs.
    TORCH_PKGS="torch==2.2.0 torchvision==0.17.0"
    TORCH_INDEX="https://download.pytorch.org/whl/cu121"
    LM_PINS=("transformers==4.40.1" "accelerate==0.32.1" "peft==0.11.1" "tokenizers==0.19.1")
    INSTALL_OPENVLA=1
    INSTALL_SPATIALVLA=0
    ;;
  openvla)
    STACK_LABEL="V0.4 (Blackwell; OpenVLA-only on the dual-VLA stack)"
    TORCH_PKGS="torch==2.7.0 torchvision"
    TORCH_INDEX="https://download.pytorch.org/whl/cu128"
    LM_PINS=("transformers==4.47.0" "accelerate==1.0.1" "peft==0.14.0" "tokenizers==0.21.0")
    INSTALL_OPENVLA=1
    INSTALL_SPATIALVLA=0
    ;;
  spatialvla)
    STACK_LABEL="V0.4 (Blackwell; SpatialVLA-only)"
    TORCH_PKGS="torch==2.7.0 torchvision"
    TORCH_INDEX="https://download.pytorch.org/whl/cu128"
    LM_PINS=("transformers==4.47.0" "accelerate==1.0.1" "peft==0.14.0" "tokenizers==0.21.0")
    INSTALL_OPENVLA=0
    INSTALL_SPATIALVLA=1
    ;;
  all)
    STACK_LABEL="V0.4 (Blackwell; OpenVLA + SpatialVLA)"
    TORCH_PKGS="torch==2.7.0 torchvision"
    TORCH_INDEX="https://download.pytorch.org/whl/cu128"
    LM_PINS=("transformers==4.47.0" "accelerate==1.0.1" "peft==0.14.0" "tokenizers==0.21.0")
    INSTALL_OPENVLA=1
    INSTALL_SPATIALVLA=1
    ;;
esac
echo "[setup.sh] Stack: $STACK_LABEL"

# Fail fast if no conda env is active (otherwise pip silently installs into the
# system / base interpreter — easy to lose 10 minutes on a wrong target env).
if [ -z "$CONDA_DEFAULT_ENV" ] || [ "$CONDA_DEFAULT_ENV" = "base" ]; then
    echo "[setup.sh] WARNING: no non-base conda env is active."
    if [ "$POLICY" = "openvla_v01" ]; then
        echo "                  Activate the target env first, e.g."
        echo "                    conda create -n cronos_envV0.1 python=3.10 -y"
        echo "                    conda activate cronos_envV0.1"
    else
        echo "                  Activate the target env first, e.g."
        echo "                    conda create -n cronos_envV0.4 python=3.10 -y"
        echo "                    conda activate cronos_envV0.4"
    fi
    echo "                  Continuing in 5 s — Ctrl-C to abort."
    sleep 5
fi

echo "[setup.sh] Installing CRONOS core dependencies (policy=$POLICY) …"

# 1. Core dependencies. Torch + LM pins picked above per POLICY.
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

echo "[setup.sh] Done (policy=$POLICY)."
