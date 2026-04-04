"""
Quick sanity check: compare BC-pretrained Gaussian head vs DiT on live rollouts.

Usage:
    cd SimplerEnv
    CUDA_VISIBLE_DEVICES=0 python simpler_env/bc_eval.py \
        --bc_init_path ../bc_checkpoints/gaussian_head_init/bc_init.pt \
        --env_id PutCarrotOnPlateInScene-v1 \
        --num_episodes 3 \
        --episode_len 80
"""

import argparse
import os
import sys
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "CogACT"))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bc_init_path", type=str, required=True)
    parser.add_argument("--vla_path", type=str, default="CogACT/CogACT-Base")
    parser.add_argument("--vla_unnorm_key", type=str, default="bridge_orig")
    parser.add_argument("--env_id", type=str, default="PutCarrotOnPlateInScene-v1")
    parser.add_argument("--num_envs", type=int, default=4)
    parser.add_argument("--num_episodes", type=int, default=3)
    parser.add_argument("--episode_len", type=int, default=80)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda:0")
    half_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    print("=" * 60)
    print("BC Pretrain Sanity Check")
    print("=" * 60)

    # Load full CogACT (VLM + DiT) for comparison
    from vla.load import load_vla
    print("\nLoading CogACT (VLM + DiT)...")
    cogact = load_vla(args.vla_path, load_for_training=False)
    cogact = cogact.to(device).eval()

    # Load BC-pretrained Gaussian head
    from simpler_env.policies.cogact.cogact_model import GaussianActionHead, ValueHead
    print(f"Loading BC checkpoint: {args.bc_init_path}")
    bc_state = torch.load(args.bc_init_path, map_location=device)
    print(f"  BC training MSE: {bc_state.get('final_mse', '?')}")
    print(f"  BC args: {bc_state.get('args', {})}")

    hidden_dim = cogact.vlm.llm_backbone.llm.lm_head.in_features
    gaussian_head = GaussianActionHead(hidden_dim, 7).to(device)
    gaussian_head.load_state_dict(bc_state["action_head"])
    gaussian_head.eval()

    # Create env
    from simpler_env.env.simpler_wrapper import SimlerWrapper
    env_args = SimpleNamespace(
        env_id=args.env_id, num_envs=args.num_envs,
        episode_len=args.episode_len, obj_set="rand", seed=args.seed,
    )
    unnorm_state = cogact.norm_stats.get(args.vla_unnorm_key, {}).get("action", {})
    env = SimlerWrapper(env_args, unnorm_state=unnorm_state, continuous_actions=True)

    # Helper: extract cognition + DiT action
    from simpler_env.bc_pretrain import collect_dit_actions

    print(f"\nRunning {args.num_episodes} episodes x {args.num_envs} envs "
          f"({args.episode_len} steps each)")
    print("-" * 60)

    all_mse = []
    all_cosine = []
    all_max_err = []

    for ep in range(args.num_episodes):
        obs_img, instruction, info = env.reset(obj_set="rand")
        print(f"\nEpisode {ep+1}: {instruction[0]}")

        ep_mse = []
        ep_cosine = []

        for step in range(args.episode_len):
            with torch.no_grad():
                # Get cognition + DiT action
                cognition, dit_action = collect_dit_actions(
                    cogact, obs_img, instruction, device, half_dtype,
                    cfg_scale=1.5, ddim_steps=5,
                )

                # Get Gaussian head action (deterministic = mean)
                dist = gaussian_head(cognition.to(device))
                gaussian_action = dist.mean

                # Compare
                mse = F.mse_loss(gaussian_action, dit_action).item()
                cosine = F.cosine_similarity(
                    gaussian_action.flatten(), dit_action.flatten(), dim=0
                ).item()
                max_err = (gaussian_action - dit_action).abs().max().item()

                ep_mse.append(mse)
                ep_cosine.append(cosine)
                all_max_err.append(max_err)

            # Step env with Gaussian head actions
            try:
                obs_img, reward, done, env_info = env.step(gaussian_action.to(device))
            except Exception:
                obs_img, instruction, info = env.reset(obj_set="rand")
                break

            if step % 20 == 0:
                print(f"  Step {step:3d} | MSE: {mse:.4f} | "
                      f"Cosine: {cosine:.3f} | MaxErr: {max_err:.3f}")
                if step == 0:
                    print(f"    DiT:      {dit_action[0].cpu().numpy().round(3)}")
                    print(f"    Gaussian: {gaussian_action[0].cpu().numpy().round(3)}")

        avg_mse = np.mean(ep_mse)
        avg_cos = np.mean(ep_cosine)
        all_mse.extend(ep_mse)
        all_cosine.extend(ep_cosine)
        print(f"  Episode avg | MSE: {avg_mse:.4f} | Cosine: {avg_cos:.3f}")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total steps evaluated: {len(all_mse)}")
    print(f"MSE:    mean={np.mean(all_mse):.4f}  std={np.std(all_mse):.4f}")
    print(f"Cosine: mean={np.mean(all_cosine):.3f}  std={np.std(all_cosine):.3f}")
    print(f"MaxErr: mean={np.mean(all_max_err):.3f}  max={np.max(all_max_err):.3f}")

    # Verdict
    mean_mse = np.mean(all_mse)
    mean_cos = np.mean(all_cosine)
    print()
    if mean_mse < 0.01 and mean_cos > 0.95:
        print("EXCELLENT: Gaussian head closely matches DiT.")
    elif mean_mse < 0.05 and mean_cos > 0.8:
        print("GOOD: Gaussian head reasonably approximates DiT. Fine for RL init.")
    elif mean_mse < 0.1 and mean_cos > 0.5:
        print("FAIR: Noticeable deviation from DiT. RL should still work but may need more BC steps.")
    else:
        print("POOR: Large deviation from DiT. Consider increasing --collect_steps and --bc_steps.")


if __name__ == "__main__":
    main()
