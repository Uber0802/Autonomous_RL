import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import numpy as np
import tyro
import torch
from main import Args

def inspect_memmap(mmap_dir):
    args = tyro.cli(Args, args=[])
    ep_len = args.episode_len
    num_envs = args.num_envs
    max_envs = args.num_envs * args.training_len // args.episode_len
    
    act_path = os.path.join(mmap_dir, "actions.dat")
    rew_path = os.path.join(mmap_dir, "rewards.dat")
    val_path = os.path.join(mmap_dir, "value_preds.dat")
    
    print(f"Loading from {mmap_dir}")
    print(f"Expected shape: ({ep_len}, {max_envs}, ...)")
    
    if not os.path.exists(act_path):
        print("Traces not found.")
        return
        
    act_size = os.path.getsize(act_path)
    # actions -> np.int32 (4 bytes) * 7 per step = 28 bytes per step per env
    max_envs = act_size // (ep_len * 7 * 4)
    print(f"Dynamically calculated max_envs: {max_envs}")
        
    actions = np.memmap(act_path, dtype=np.int32, mode='r', shape=(ep_len, max_envs, 7))
    rewards = np.memmap(rew_path, dtype=np.float32, mode='r', shape=(ep_len, max_envs, 1))
    values_size = os.path.getsize(val_path)
    # value_preds -> ep_len+1 
    values = np.memmap(val_path, dtype=np.float32, mode='r', shape=(ep_len+1, max_envs, 1))
    
    # We want to see action distributions across the segment (how many 0s vs other bins?)
    print("Action distribution per step across all processed envs:")
    for step in range(0, min(ep_len, 20)):
        # Inspect env 0 across steps
        acts = actions[step, 0]
        rew = rewards[step, 0, 0]
        val = values[step, 0, 0]
        print(f"Step {step:02d} | Env 0 Acts: {acts} | Rew: {rew:.4f} | Val: {val:.4f}")
        
    print("\nAction variance across the entire buffer section:")
    total_valid = 0
    # Let's check how many steps have identical non-moving actions (e.g. 128 is center)
    for env in range(max_envs):
        if np.any(actions[:, env] != 0): # non-empty env
            total_valid += 1
            unique_acts = np.unique(actions[:, env], axis=0)
            if env < 5 or env > max_envs - 5: # log first and last few
                print(f"Env {env:02d} has {len(unique_acts)} unique action vectors over {ep_len} steps.")
                
    print(f"Found {total_valid} populated trajectories out of {max_envs}.")

if __name__ == "__main__":
    # Example usage with relative path to a mmap directory
    # inspect_memmap("cronos_mmap_PID")
    pass
