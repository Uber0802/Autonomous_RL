import os
import numpy as np
import torch
import random


class SuccessOfflineBuffer:
    def __init__(self, save_dir, buffer_minibatch=64):
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)

        self.trajectories = []  # In-memory store (optional)
        self.buffer_minibatch = buffer_minibatch

    def save_trajectory(self, obs, actions, instructions, filename=None):
        """
        Save a successful trajectory to disk.
        obs: np.array of shape [T, *obs_dim]
        actions: np.array of shape [T, act_dim]
        instructions: list of strings of length T (optional)
        filename: custom file name (optional)
        """
        traj = {
            'obs': obs,
            'actions': actions,
            'instructions': instructions,
        }

        if filename is None:
            filename = f"traj_{len(os.listdir(self.save_dir))}.npz"
        path = os.path.join(self.save_dir, filename)

        np.savez_compressed(path, **traj)
        print(f"Trajectory saved to {path}")

    def load_trajectories(self):
        """
        Load all successful trajectories from disk into memory.
        """
        self.trajectories = []

        files = sorted(f for f in os.listdir(self.save_dir) if f.endswith('.npz'))
        for file in files:
            path = os.path.join(self.save_dir, file)
            data = np.load(path, allow_pickle=True)

            traj = {
                'obs': data['obs'],
                'actions': data['actions'],
            }
            if 'instructions' in data:
                traj['instructions'] = data['instructions'].tolist()
            else:
                traj['instructions'] = ["" for _ in range(len(data['obs']))]

            self.trajectories.append(traj)

        print(f"Loaded {len(self.trajectories)} trajectories from {self.save_dir}")

    def feed_forward_generator(self, batch_count):
        # Combine all data
        all_obs = []
        all_actions = []
        all_instructions = []
    
        for traj in self.trajectories:
            all_obs.append(traj['obs'])           # [T, *obs_dim]
            all_actions.append(traj['actions'])   # [T, act_dim]
            all_instructions.extend(traj['instructions'])  # [T]
    
            obs = np.concatenate(all_obs, axis=0)         # [N, *obs_dim]
            actions = np.concatenate(all_actions, axis=0) # [N, act_dim]
            instructions = all_instructions               # list of N strings
    
            dataset_size = len(obs)
            total_samples = batch_count * self.buffer_minibatch
    
            if dataset_size >= total_samples:
                # enough samples, sample without replacement
                indices = torch.randperm(dataset_size).numpy()[:total_samples]
            else:
                # not enough samples, sample with replacement
                indices = np.random.choice(dataset_size, total_samples, replace=True)
    
            for i in range(batch_count):
                start = i * self.buffer_minibatch
                end = start + self.buffer_minibatch
                batch_indices = indices[start:end]
    
                obs_batch = obs[batch_indices]
                actions_batch = actions[batch_indices]
                instructions_batch = [instructions[j] for j in batch_indices]
    
                yield obs_batch, instructions_batch, actions_batch