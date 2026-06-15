"""P-2 gate driver — `SpatialVLAPolicy` (real adapter), PPO surface authoritative re-run.

Runs the P-2 surface gates (`gate_frozen_spatial_embed`, `gate_suffix_layout`,
`gate_G3_anchor`, `gate_G2`, `gate_G3`, `gate_save_load`) through the production
adapter from `simpler_env.policies.spatialvla.spatialvla_train.SpatialVLAPolicy`.

P-1's gate run went through a `HandSuffixShim`; per NF-13, P-2 is the
*authoritative* first exercise of the real `_preprocess_obs` evaluate path.

Usage (from `Autonomous_RL/SpatialVLA/`):
    conda activate spatialvla_cronos
    CUDA_VISIBLE_DEVICES=0 PYTHONPATH=.:../SimplerEnv \\
        python -m test.test_p2_adapter \\
        --model-path IPEC-COMMUNITY/spatialvla-4b-224-sft-bridge
"""
import argparse
import shutil
import sys
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from PIL import Image


@dataclass
class _AdapterArgs:
    """Minimal namespace satisfying `SpatialVLAPolicy.__init__` expectations.

    Mirrors the subset of CRONOS `Args` fields the adapter reads at construction
    time. PPO-specific fields default to plan values (lr 1e-4 / 3e-3, LoRA r32).
    """
    vla_path: str = ""
    vla_load_path: str = ""
    vla_unnorm_key: str = "bridge_orig/1.0.0"
    seed: int = 0
    vla_temperature: float = 1.0
    vla_temperature_eval: float = 0.0
    vla_lora_rank: int = 32
    vla_lr: float = 1e-4
    vla_vhlr: float = 3e-3
    vla_optim_beta1: float = 0.9
    vla_optim_beta2: float = 0.999


def build_obs(device, image_path: Path, instructions):
    pil = Image.open(image_path).convert("RGB").resize((224, 224))
    arr = np.asarray(pil, dtype=np.uint8)
    images = np.stack([arr for _ in instructions], axis=0)
    image_t = torch.from_numpy(images).to(device=device)
    return {"image": image_t, "task_description": list(instructions)}


def _run_gate(name, fn):
    try:
        out = fn()
        msg = f"  PASS  {name}"
        if isinstance(out, float):
            msg += f"  · returned {out:.3e}"
        print(msg)
        return True
    except Exception as e:
        print(f"  FAIL  {name}  · {type(e).__name__}: {e}")
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser("P-2 PPO-surface gates (real adapter)")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--image", default="test/example.png")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save-load-dir", default="logs/p2_save_load_scratch")
    args = parser.parse_args()

    # Make the adapter importable from its SimplerEnv location.
    here = Path(__file__).resolve().parent.parent           # Autonomous_RL/SpatialVLA
    simpler_env = (here.parent / "SimplerEnv").resolve()
    sys.path.insert(0, str(simpler_env))
    sys.path.insert(0, str(here))                            # for `model.*` imports inside adapter

    from simpler_env.policies.spatialvla.spatialvla_train import SpatialVLAPolicy

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    adapter_args = _AdapterArgs(
        vla_path=args.model_path,
        seed=args.seed,
    )
    print(f"[load] SpatialVLAPolicy from {args.model_path}")
    policy = SpatialVLAPolicy(adapter_args, device_id=0)
    device = policy.tpdv["device"]

    # Build obs batch.
    image_path = Path(args.image)
    if not image_path.is_absolute():
        image_path = here / image_path
    instructions = ["pick the cup", "stack the blocks"][: args.batch_size]
    if len(instructions) < args.batch_size:
        instructions += [instructions[-1]] * (args.batch_size - len(instructions))
    obs = build_obs(device, image_path, instructions)

    from test.gates_ppo import (
        gate_geometry, gate_G2, gate_G2d_no_truncation, gate_G2d_warper_absent,
        gate_G3, gate_G3_anchor,
        gate_frozen_spatial_embed, gate_suffix_layout, gate_save_load,
    )

    # Get a real action_ids batch to drive gate_suffix_layout.
    policy.prep_rollout()
    with torch.no_grad():
        _v, action_ids, _lp = policy.get_action(obs, deterministic=True)

    print("\n[P-2 gates]")
    results = []
    # M4 — LoRA wiring sanity.
    results.append(_run_gate("gate_frozen_spatial_embed (M4)", lambda: gate_frozen_spatial_embed(policy)))
    # M3 — suffix layout on the real evaluate-side preprocess.
    results.append(_run_gate("gate_suffix_layout (M3)", lambda: gate_suffix_layout(policy, obs, action_ids)))

    # Authoritative re-run of P-1 gates through the real adapter (NF-13).
    results.append(_run_gate("gate_geometry (NF-7)",            lambda: gate_geometry(policy)))
    results.append(_run_gate("gate_G2 (authoritative)",         lambda: gate_G2(policy, obs)))
    results.append(_run_gate("gate_G2d_no_truncation",          lambda: gate_G2d_no_truncation(policy, obs)))
    results.append(_run_gate("gate_G2d_warper_absent (NF-1)",   lambda: gate_G2d_warper_absent(policy, obs)))
    results.append(_run_gate("gate_G3_anchor G3a (NF-8)",       lambda: gate_G3_anchor(policy, obs)))
    results.append(_run_gate("gate_G3 (G3 + G3b)",              lambda: gate_G3(policy, obs)))

    # M5 / FR-9 — save / load round-trip (PEFT adapter + value head + BOTH
    # optimizers + norm_stats). Runs LAST because it mutates parameters via an
    # optimizer step and then tears down + reloads the model.
    save_load_dir = Path(args.save_load_dir).resolve()
    if save_load_dir.exists():
        shutil.rmtree(save_load_dir)
    save_load_dir.mkdir(parents=True, exist_ok=True)
    results.append(_run_gate("gate_save_load (M5 / FR-9, NF-11)",
                             lambda: gate_save_load(policy, obs, save_load_dir)))

    n_pass = sum(results)
    n_total = len(results)
    print(f"\n[summary] {n_pass}/{n_total} gates passed")
    sys.exit(0 if n_pass == n_total else 1)


if __name__ == "__main__":
    main()
