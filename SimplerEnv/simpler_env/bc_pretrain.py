"""
BC Pretraining: Distill CogACT DiT action head → Gaussian action head.

The CogACT model has a pretrained DiT that generates high-quality actions from
cognition tokens. We use it as a teacher to warmstart the Gaussian action head
before RL training. This gives the policy a reasonable starting point instead
of random actions.

Usage:
    cd SimplerEnv
    CUDA_VISIBLE_DEVICES=0 python simpler_env/bc_pretrain.py \
        --vla_path CogACT/CogACT-Base \
        --env_id PutCarrotOnPlateInScene-v1 \
        --num_envs 16 \
        --bc_steps 2000 \
        --bc_lr 3e-4 \
        --bc_batch_size 16 \
        --save_path ../bc_checkpoints/gaussian_head_init \
        --seed 42
"""

import argparse
import os
import sys
import time
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from tqdm import tqdm

# Add parent paths
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "CogACT"))

from vla.load import load_vla
from simpler_env.policies.cogact.cogact_model import GaussianActionHead, ValueHead


def parse_args():
    parser = argparse.ArgumentParser(description="BC pretrain Gaussian head from DiT")
    parser.add_argument("--vla_path", type=str, default="CogACT/CogACT-Base")
    parser.add_argument("--env_id", type=str, default="PutCarrotOnPlateInScene-v1")
    parser.add_argument("--num_envs", type=int, default=16,
                        help="Number of parallel envs for collecting observations")
    parser.add_argument("--bc_steps", type=int, default=2000,
                        help="Number of BC training steps")
    parser.add_argument("--bc_lr", type=float, default=3e-4)
    parser.add_argument("--bc_batch_size", type=int, default=16,
                        help="Batch size for BC training (samples from replay)")
    parser.add_argument("--replay_size", type=int, default=5000,
                        help="Size of replay buffer for BC data")
    parser.add_argument("--collect_steps", type=int, default=50,
                        help="Env steps to collect before BC training starts")
    parser.add_argument("--dit_cfg_scale", type=float, default=1.5,
                        help="Classifier-free guidance scale for DiT sampling")
    parser.add_argument("--dit_ddim_steps", type=int, default=5,
                        help="DDIM sampling steps for DiT")
    parser.add_argument("--save_path", type=str, default="../bc_checkpoints/gaussian_head_init")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log_interval", type=int, default=100)
    parser.add_argument("--vla_unnorm_key", type=str, default="bridge_orig")
    return parser.parse_args()


class BCReplayBuffer:
    """Simple replay buffer storing (cognition_token, dit_action) pairs."""

    def __init__(self, max_size: int, cognition_dim: int = 4096, action_dim: int = 7):
        self.max_size = max_size
        self.cognition = np.zeros((max_size, cognition_dim), dtype=np.float32)
        self.actions = np.zeros((max_size, action_dim), dtype=np.float32)
        self.size = 0
        self.ptr = 0

    def add(self, cognition: np.ndarray, action: np.ndarray):
        """Add batch of (cognition, action) pairs."""
        B = cognition.shape[0]
        for i in range(B):
            self.cognition[self.ptr] = cognition[i]
            self.actions[self.ptr] = action[i]
            self.ptr = (self.ptr + 1) % self.max_size
            self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size: int):
        idx = np.random.randint(0, self.size, size=batch_size)
        return (
            torch.tensor(self.cognition[idx], dtype=torch.float32),
            torch.tensor(self.actions[idx], dtype=torch.float32),
        )


def collect_dit_actions(cogact_model, obs_img, instruction, device, half_dtype,
                        cfg_scale=1.5, ddim_steps=5):
    """
    Generate DiT teacher actions for a batch of observations.
    Args:
        obs_img: [B, H, W, 3] uint8 tensor or numpy array
    Returns: cognition [B, 4096], actions [B, 7] (normalized [-1, 1])
    """
    from PIL import Image as PILImage
    from torch.nn.utils.rnn import pad_sequence

    B = obs_img.shape[0]

    # Preprocess observations (same as CogACTPolicy._preprocess_obs)
    tokenizer = cogact_model.vlm.llm_backbone.tokenizer
    image_transform = cogact_model.vlm.vision_backbone.get_image_transform()

    input_ids_list = []
    pixel_values_list = []

    for i in range(B):
        prompt_builder = cogact_model.vlm.get_prompt_builder()
        prompt_builder.add_turn(
            role="human",
            message=f"What action should the robot take to {instruction[i].lower()}?",
        )
        prompt_text = prompt_builder.get_prompt()

        single_ids = tokenizer(prompt_text, truncation=True, return_tensors="pt").input_ids.squeeze(0)
        single_ids = torch.cat([single_ids, torch.tensor([29871, 2], dtype=torch.long)])
        input_ids_list.append(single_ids.to(device))

        img_np = obs_img[i].cpu().numpy() if isinstance(obs_img[i], torch.Tensor) else obs_img[i]
        pil_img = PILImage.fromarray(img_np)
        pixel_values_list.append(image_transform(pil_img))

    pad_token_id = tokenizer.pad_token_id
    input_ids = pad_sequence(input_ids_list, batch_first=True, padding_value=pad_token_id)
    input_ids = input_ids[:, :tokenizer.model_max_length].to(device)
    attention_mask = input_ids.ne(pad_token_id).to(device)

    if isinstance(pixel_values_list[0], dict):
        pixel_values = {
            k: torch.stack([pv[k] for pv in pixel_values_list]).to(device)
            for k in pixel_values_list[0]
        }
    else:
        pixel_values = torch.stack(pixel_values_list).to(device)

    # Extract cognition token via VLM forward
    with torch.no_grad(), torch.autocast("cuda", dtype=half_dtype):
        output = cogact_model.vlm(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            output_hidden_states=True,
            return_dict=True,
        )

    # Extract cognition feature
    if cogact_model.vlm.vision_backbone.featurizer is not None:
        num_patch = cogact_model.vlm.vision_backbone.featurizer.patch_embed.num_patches
    elif hasattr(cogact_model.vlm.vision_backbone, 'siglip_featurizer'):
        num_patch = cogact_model.vlm.vision_backbone.siglip_featurizer.patch_embed.num_patches
    else:
        num_patch = 0

    last_hidden = output.hidden_states[-1][:, num_patch:]
    cumsum = attention_mask.cumsum(dim=1)
    last_idx = (cumsum == cumsum.max(dim=1, keepdim=True)[0]).float().argmax(dim=1)
    expanded = last_idx.unsqueeze(-1).expand(-1, last_hidden.size(-1))
    cognition = last_hidden.gather(1, expanded.unsqueeze(1)).squeeze(1)  # [B, 4096]

    # Generate actions via DiT (teacher)
    model_dtype = next(cogact_model.action_model.net.parameters()).dtype
    cog_for_dit = cognition.unsqueeze(1).to(model_dtype)  # [B, 1, 4096]

    future_window = cogact_model.future_action_window_size
    action_dim = cogact_model.action_model.in_channels
    noise = torch.randn(B, future_window + 1, action_dim, device=device, dtype=model_dtype)

    using_cfg = cfg_scale > 1.0
    if using_cfg:
        noise = torch.cat([noise, noise], 0)
        uncond = cogact_model.action_model.net.z_embedder.uncondition.unsqueeze(0).expand(B, 1, -1)
        z = torch.cat([cog_for_dit, uncond], 0)
        model_kwargs = dict(z=z, cfg_scale=cfg_scale)
        sample_fn = cogact_model.action_model.net.forward_with_cfg
    else:
        model_kwargs = dict(z=cog_for_dit)
        sample_fn = cogact_model.action_model.net.forward

    with torch.no_grad():
        if cogact_model.action_model.ddim_diffusion is None:
            cogact_model.action_model.create_ddim(ddim_step=ddim_steps)
        samples = cogact_model.action_model.ddim_diffusion.ddim_sample_loop(
            sample_fn, noise.shape, noise,
            clip_denoised=False,
            model_kwargs=model_kwargs,
            progress=False,
            device=device,
            eta=0.0,
        )

    if using_cfg:
        samples, _ = samples.chunk(2, dim=0)

    # Take only the first action (current timestep), clipped to [-1, 1]
    dit_actions = samples[:, 0, :].float().clamp(-1, 1)  # [B, 7]

    return cognition.float().detach(), dit_actions.detach()


def main():
    args = parse_args()

    # Seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda:0")
    half_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    print(f"=== BC Pretraining: DiT → Gaussian Head ===")
    print(f"Model: {args.vla_path}")
    print(f"Env: {args.env_id}, num_envs: {args.num_envs}")
    print(f"BC steps: {args.bc_steps}, lr: {args.bc_lr}, batch_size: {args.bc_batch_size}")
    print(f"Half dtype: {half_dtype}")

    # Load full CogACT model (VLM + DiT)
    print("\nLoading CogACT model (VLM + DiT)...")
    cogact = load_vla(args.vla_path, load_for_training=False)  # inference mode for DiT
    cogact = cogact.to(device)
    cogact.eval()
    print(f"Model loaded. Norm stats keys: {list(cogact.norm_stats.keys())[:3]}")

    # Create Gaussian head (student) and value head
    hidden_dim = cogact.vlm.llm_backbone.llm.lm_head.in_features  # 4096
    action_head = GaussianActionHead(hidden_dim, 7).to(device)
    value_head = ValueHead(hidden_dim).to(device)
    optimizer = Adam(list(action_head.parameters()) + list(value_head.parameters()),
                     lr=args.bc_lr)

    # Create env
    print(f"\nCreating env: {args.env_id} x {args.num_envs}...")
    from types import SimpleNamespace
    from simpler_env.env.simpler_wrapper import SimlerWrapper
    env_args = SimpleNamespace(
        env_id=args.env_id,
        num_envs=args.num_envs,
        episode_len=80,
        obj_set="rand",
        seed=args.seed,
    )
    unnorm_state = cogact.norm_stats.get(args.vla_unnorm_key, {}).get("action", {})
    env = SimlerWrapper(env_args, unnorm_state=unnorm_state, continuous_actions=True)

    # Reset env
    obs_img, instruction, info = env.reset(obj_set=env_args.obj_set)
    print(f"Env reset. Instructions: {instruction[:2]}")

    # Create replay buffer
    replay = BCReplayBuffer(args.replay_size, cognition_dim=hidden_dim, action_dim=7)

    # Phase 1: Collect initial data
    print(f"\n--- Phase 1: Collecting {args.collect_steps} steps of DiT data ---")
    for step in tqdm(range(args.collect_steps), desc="Collecting"):
        cognition, dit_actions = collect_dit_actions(
            cogact, obs_img, instruction, device, half_dtype,
            cfg_scale=args.dit_cfg_scale, ddim_steps=args.dit_ddim_steps,
        )
        replay.add(cognition.cpu().numpy(), dit_actions.cpu().numpy())

        # Step env with DiT actions → robot moves, giving diverse arm states
        action_tensor = dit_actions.to(device)
        try:
            obs_img, reward, done, env_info = env.step(action_tensor)
        except Exception:
            obs_img, instruction, info = env.reset(obj_set=env_args.obj_set)

        # Reset every ~1 episode (80 steps) for new scene layout
        # Within each episode, the DiT drives the robot through a full trajectory,
        # giving diverse robot states (reaching, grasping, lifting, placing)
        if (step + 1) % 80 == 0:
            obs_img, instruction, info = env.reset(obj_set=env_args.obj_set)

    print(f"Replay buffer: {replay.size} samples")

    # Phase 2: BC training
    print(f"\n--- Phase 2: BC Training ({args.bc_steps} steps) ---")
    action_head.train()
    value_head.train()

    losses = []
    for step in range(args.bc_steps):
        cog_batch, action_batch = replay.sample(args.bc_batch_size)
        cog_batch = cog_batch.to(device)
        action_batch = action_batch.to(device)

        # Forward through Gaussian head
        dist = action_head(cog_batch)
        predicted_mean = dist.mean

        # BC loss: NLL (trains both mean and std jointly)
        # log_prob per-dim then mean over dims and batch — balanced with MSE scale
        nll_loss = -dist.log_prob(action_batch).mean()
        mse_loss = F.mse_loss(predicted_mean, action_batch)

        # Value head regularization: predict zero (no reward signal yet)
        value_pred = value_head(cog_batch)
        value_loss = F.mse_loss(value_pred, torch.zeros_like(value_pred))

        # NLL is the primary loss (trains both mean and std).
        # MSE is a stabilizer (prevents mean from drifting if std collapses).
        loss = nll_loss + 0.5 * mse_loss + 0.1 * value_loss

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(action_head.parameters()) + list(value_head.parameters()), 1.0
        )
        optimizer.step()

        losses.append(mse_loss.item())

        if (step + 1) % args.log_interval == 0:
            avg_loss = np.mean(losses[-args.log_interval:])
            std_pred = dist.scale.mean().item()
            log_std_val = action_head.log_std.data.mean().item()
            print(f"Step {step+1}/{args.bc_steps} | MSE: {avg_loss:.6f} | "
                  f"NLL: {nll_loss.item():.4f} | std: {std_pred:.4f} | "
                  f"log_std: {log_std_val:.3f}")

        # Collect more data periodically to keep replay diverse
        if (step + 1) % 200 == 0:
            action_head.eval()
            with torch.no_grad():
                cognition, dit_actions = collect_dit_actions(
                    cogact, obs_img, instruction, device, half_dtype,
                    cfg_scale=args.dit_cfg_scale, ddim_steps=args.dit_ddim_steps,
                )
                replay.add(cognition.cpu().numpy(), dit_actions.cpu().numpy())
                try:
                    obs_img, reward, done, env_info = env.step(dit_actions.to(device))
                except Exception:
                    obs_img, instruction, info = env.reset(obj_set=env_args.obj_set)
            action_head.train()

    # Save
    os.makedirs(args.save_path, exist_ok=True)
    torch.save({
        "action_head": action_head.state_dict(),
        "value_head": value_head.state_dict(),
        "args": vars(args),
        "final_mse": np.mean(losses[-100:]),
    }, os.path.join(args.save_path, "bc_init.pt"))

    print(f"\n=== BC Pretraining Complete ===")
    print(f"Final MSE (last 100): {np.mean(losses[-100:]):.6f}")
    print(f"Saved to: {args.save_path}/bc_init.pt")

    # Quick eval: compare Gaussian vs DiT on a few samples
    print(f"\n--- Quick comparison: Gaussian vs DiT ---")
    action_head.eval()
    with torch.no_grad():
        cog_eval, dit_eval = replay.sample(8)
        cog_eval = cog_eval.to(device)
        dit_eval = dit_eval.to(device)
        gaussian_eval = action_head(cog_eval).mean
        mse = F.mse_loss(gaussian_eval, dit_eval).item()
        max_err = (gaussian_eval - dit_eval).abs().max().item()
        print(f"Eval MSE: {mse:.6f} | Max error: {max_err:.4f}")
        print(f"DiT sample:      {dit_eval[0].cpu().numpy().round(3)}")
        print(f"Gaussian sample:  {gaussian_eval[0].cpu().numpy().round(3)}")

    if hasattr(env, 'close'):
        env.close()


if __name__ == "__main__":
    main()
