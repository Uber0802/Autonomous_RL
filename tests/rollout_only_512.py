"""
Inference-only rollout for the Emu3 path at 512² image resolution.

Bypasses PPO training (which would OOM on a 48 GB card at 512²) and just
runs 20 SimplerEnv steps with a fresh policy, saving an mp4 per env.
Used to verify that 512² gives a stronger zero-shot rollout than 256².
"""

import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "SimplerEnv"))
sys.path.insert(0, str(REPO_ROOT / "UniVLA" / "reference" / "Emu3"))

from mani_skill.utils.visualization.misc import images_to_video

from simpler_env.env.simpler_wrapper import SimlerWrapper
from simpler_env.policies.univla.univla_train import UniVLAPolicy


def main():
    args = SimpleNamespace(
        # vla
        vla_path=str(REPO_ROOT / "checkpoints/univla-emu3-raw/UNIVLA_SIMPLER_BRIDGE_VIDEO_BS128_20K"),
        vision_vq_path=str(REPO_ROOT / "checkpoints/emu3-vision-tokenizer"),
        fast_tokenizer_path=str(REPO_ROOT / "checkpoints/fast-bridge-t5-s50"),
        vla_unnorm_key="bridge_robot",
        vla_load_path="",
        vla_lora_rank=32,
        vla_vhlr=1e-5, vla_lr=1e-5,
        vla_temperature=1.0, vla_temperature_eval=0.0,
        vla_optim_beta1=0.9, vla_optim_beta2=0.999,
        vla_image_pixels=262144,  # 512² → ~4800 vision tokens, training-time grid
        # env
        env_id="TwoObjectTwoReceptacle-v1",
        num_envs=2,
        episode_len=20,
        seed=0,
        obj_set="rand",
        obj1_index=7, obj2_index=2, plate1_index=1, plate2_index=2,
        use_same_init=True,
        reset_unsuitable=False,
        few_position=False,
        random_task_order=False,
    )

    out_dir = REPO_ROOT.parent / "reference_emu3_512"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"=== Loading UniVLAPolicy at 512² ({args.vla_image_pixels} pixels) ===")
    policy = UniVLAPolicy(args, device_id=0)
    policy.prep_rollout()

    # The wrapper needs unnorm_state in OpenVLA-style {q01, q99, ...}
    unnorm_state = policy.get_action_stats()
    print(f"  unnorm_state keys: {list(unnorm_state.keys()) if unnorm_state else None}")

    print("\n=== Initializing SimlerWrapper ===")
    env = SimlerWrapper(args, unnorm_state, last_vocab_idx=151642)

    object_list = ["ketchup_bottle", "ketchup_bottle"]
    receptacle_list = ["yellow_plate", "cloth"]
    obs_img, instruction, info = env.reset(
        obj_set="rand", same_init=True,
        object=object_list, receptacle=receptacle_list,
    )
    print(f"  obs_img shape: {tuple(obs_img.shape)}")
    print(f"  instruction[0]: {instruction[0]}")
    print(f"  instruction[1]: {instruction[1]}")

    frames = [[] for _ in range(args.num_envs)]
    actions_log = [[] for _ in range(args.num_envs)]

    print("\n=== Rolling out 20 deterministic steps ===")
    for step in range(args.episode_len):
        obs = dict(image=obs_img, task_description=instruction)
        with torch.no_grad():
            values, cont_action, padded, lp = policy.get_action(obs, deterministic=True)
        obs_img_new, reward, done, env_info = env.step_continuous(cont_action)

        for i in range(args.num_envs):
            frames[i].append(obs_img[i].cpu().numpy())
            actions_log[i].append(cont_action[i].cpu().numpy().tolist())

        if step % 5 == 0:
            print(f"  step {step:2}: env0 z={cont_action[0,2].item():+.4f} g={cont_action[0,6].item():+.3f}  "
                  f"env1 z={cont_action[1,2].item():+.4f} g={cont_action[1,6].item():+.3f}")
        obs_img = obs_img_new

    # Append final frame
    for i in range(args.num_envs):
        frames[i].append(obs_img[i].cpu().numpy())

    print("\n=== Saving videos ===")
    for i in range(args.num_envs):
        name = f"reference_emu3_512_env{i}_{object_list[i]}_{receptacle_list[i]}"
        images_to_video(frames[i], str(out_dir), name, fps=10, verbose=False)
        print(f"  saved {out_dir}/{name}.mp4 ({len(frames[i])} frames)")

    print("\n=== Action trajectory summary ===")
    for i in range(args.num_envs):
        arr = np.asarray(actions_log[i])
        print(f"  env{i} ({object_list[i]} → {receptacle_list[i]}):")
        print(f"    z mean: {arr[:, 2].mean():+.5f}  range: [{arr[:, 2].min():+.5f}, {arr[:, 2].max():+.5f}]")
        print(f"    gripper: {[round(g, 3) for g in arr[:, 6]]}")


if __name__ == "__main__":
    main()
