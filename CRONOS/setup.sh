#!/bin/bash
# setup.sh - install Python dependencies for CRONOS + one or both VLA policies.
#
# Usage:
#   bash setup.sh                # default: install BOTH OpenVLA + SpatialVLA stacks in
#                                  the currently active conda env (one env serves both)
#   bash setup.sh all            # same as above (explicit)
#   bash setup.sh openvla        # OpenVLA-only stack (lighter; cannot run --policy spatialvla)
#   bash setup.sh spatialvla     # SpatialVLA-only stack (cannot run --policy openvla)
#
# Recommended workflow (one env for both policies):
#   conda create -n cronos_env python=3.10 -y
#   conda activate cronos_env
#   cd Benchmark/CRONOS
#   bash setup.sh                # both OpenVLA + SpatialVLA share one env
#
# Pin rationale:
#   * transformers==4.47.0  — SpatialVLA's model files use APIs from 4.47.0
#                             (GenerationMixin layout, Gemma2 logits format).
#                             Hard ceiling: <4.50 (GenerationMixin removed).
#   * torch==2.7.0+cu128    — supports sm_120 (Blackwell, e.g. RTX PRO 6000).
#                             SpatialVLA's own pyproject pins torch==2.5.1+cu121,
#                             but +cu121 wheels are built for sm_50..sm_90 only,
#                             so generate() throws "no kernel image" on Blackwell.
#                             2.7.0+cu128 is the lowest stable build with sm_120.
#
# OpenVLA's setup.py declares older pins (torch 2.2.0, transformers 4.40.1,
# tokenizers 0.19.1); pip prints a dep-conflict warning but the runtime API
# is compatible — verified end-to-end on cronos-univla.

set -e

POLICY=${1:-all}

case $POLICY in
  openvla|spatialvla|all) ;;
  *) echo "Unknown policy: $POLICY"; echo "Valid: openvla | spatialvla | all"; exit 1 ;;
esac

echo "[setup.sh] Installing CRONOS core dependencies (policy=$POLICY) …"

# 1. Core dependencies (Blackwell-compatible torch + transformers that span both VLAs).
pip install "setuptools<70.0.0"
pip install torch==2.7.0 torchvision --index-url https://download.pytorch.org/whl/cu128
pip install "numpy<2.0.0" \
    transformers==4.47.0 accelerate==1.0.1 peft==0.14.0 \
    einops==0.8.0 tokenizers==0.21.0 scipy==1.14.1 \
    "timm>=0.9.10,<1.0.0" draccus rich \
    gymnasium==0.29.1 tyro wandb tqdm transforms3d sapien==3.0.0.b1 \
    dacite h5py tabulate trimesh imageio "imageio[ffmpeg]" \
    "mplib==0.1.1" "fast_kinematics==0.2.2" IPython \
    "pytorch_kinematics==0.7.5" pynvml
pip install tensorflow==2.15.0 tensorflow-datasets==4.9.3 tensorflow-graphics

# 2. Plotting tooling (`scripts/plot.py` needs pandas + matplotlib at pinned versions).
pip install -r scripts/requirements_plot.txt

# 3. Pillar packages used by every policy (env, manipulation simulator, shared adapters).
pip install --no-deps -e ../ManiSkill
pip install --no-deps -e ../SimplerEnv

# 4. Policy-specific backbones.
if [ "$POLICY" = "openvla" ] || [ "$POLICY" = "all" ]; then
    echo "[setup.sh] Installing OpenVLA backbone …"
    pip install --no-deps "dlimp @ git+https://github.com/moojink/dlimp_openvla"
    pip install --no-deps -e ../openvla
fi

if [ "$POLICY" = "spatialvla" ] || [ "$POLICY" = "all" ]; then
    echo "[setup.sh] Installing SpatialVLA backbone …"
    # SpatialVLA is vendored as a sibling dir; install editable so the integration
    # code in CRONOS/SimplerEnv can import `model.modeling_spatialvla` directly.
    # `--no-deps` because section 1 already covers its runtime deps, and pinning
    # torch==2.5.1+cu121 from its pyproject would defeat the Blackwell-compatible
    # torch pin chosen above.
    pip install --no-deps -e ../SpatialVLA
fi

echo "[setup.sh] Done (policy=$POLICY)."
