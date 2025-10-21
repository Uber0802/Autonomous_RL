import os
import numpy as np
import torch
import random
import gc


class SuccessOfflineBuffer:
    def __init__(self, save_dir, buffer_minibatch=4, max_traj=64):
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)

        self.trajectories = []  # In-memory store (optional)
        self.buffer_minibatch = buffer_minibatch
        self.max_traj = max_traj

    def clear_trajectories(self):
        """
        Clear trajectories from memory and force garbage collection.
        """
        self.trajectories = None
        gc.collect()  # Force garbage collection

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
        Load all successful trajectories from disk into memory, preallocating space for obs, actions, and instructions.
        """
        files = sorted(f for f in os.listdir(self.save_dir) if f.endswith('.npz'))
        total_files = len(files)
        if self.max_traj < total_files:
            files = random.sample(files, self.max_traj)

        num_traj = len(files)

        # Load one sample to determine shapes
        sample_data = np.load(os.path.join(self.save_dir, files[0]), allow_pickle=True)
        T = len(sample_data['obs'])  # time steps
        obs_shape = sample_data['obs'].shape[1:]  # shape without time
        action_shape = sample_data['actions'].shape[1:]  # shape without time

        # Preallocate arrays
        obs_array = np.zeros((num_traj, T, *obs_shape), dtype=sample_data['obs'].dtype)
        action_array = np.zeros((num_traj, T, *action_shape), dtype=sample_data['actions'].dtype)
        instructions_array = np.empty((num_traj, T), dtype=object)  # Assuming instructions are strings

        for i, file in enumerate(files):
            path = os.path.join(self.save_dir, file)
            data = np.load(path, allow_pickle=True)

            obs_array[i] = data['obs']
            action_array[i] = data['actions']
            instructions_array[i] = data['instructions']  # Already assumed present

        # Store as a single dict or split if needed
        self.trajectories = {
            'obs': obs_array,
            'actions': action_array,
            'instructions': instructions_array,
        }

        print(f"Loaded {num_traj} from {self.save_dir} with {total_files} trajectories.")


    def feed_forward_generator(self, batch_count):
        """
        Yield batches of (obs, instructions, actions) from preloaded trajectories.
        Each batch has self.buffer_minibatch samples.
        """

        # Unpack and flatten trajectory data
        obs = self.trajectories['obs'].reshape(-1, *self.trajectories['obs'].shape[2:])         # [N, *obs_dim]
        actions = self.trajectories['actions'].reshape(-1, *self.trajectories['actions'].shape[2:])  # [N, act_dim]
        instructions = self.trajectories['instructions'].reshape(-1)  # [N], still dtype=object (strings)

        dataset_size = len(obs)
        total_samples = batch_count * self.buffer_minibatch

        # Decide whether to sample with or without replacement
        if dataset_size >= total_samples:
            indices = torch.randperm(dataset_size).numpy()[:total_samples]
        else:
            indices = np.random.choice(dataset_size, total_samples, replace=True)

        for i in range(batch_count):
            start = i * self.buffer_minibatch
            end = start + self.buffer_minibatch
            batch_indices = indices[start:end]

            obs_batch = obs[batch_indices]
            actions_batch = actions[batch_indices]
            instructions_batch = instructions[batch_indices].tolist()  # convert to list of strings

            yield obs_batch, instructions_batch, actions_batch

        del obs, actions, instructions
        gc.collect()
        self.clear_trajectories()
