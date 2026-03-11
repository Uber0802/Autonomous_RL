import numpy as np
import torch
import os
import gc

def create_memmap(filename, shape, dtype):
    """Utility to create memory-mapped files for large buffers."""
    pid = os.getpid()
    buffer_dir = os.path.join(os.getcwd(), f"cronos_mmap_{pid}")
    os.makedirs(buffer_dir, exist_ok=True)
    filepath = os.path.join(buffer_dir, filename)
    if os.path.exists(filepath):
        os.remove(filepath)
    return np.memmap(filepath, dtype=dtype, mode='w+', shape=shape)

class CronosReplayBuffer:
    """Efficient memory-mapped replay buffer for PPO training."""
    
    def __init__(self, args, obs_dim=(480, 640, 3), act_dim=7):
        self.ep_len = args.episode_len
        self.gamma = args.buffer_gamma
        self.gae_lambda = args.buffer_lambda
        self.minibatch_size = args.buffer_minibatch
        
        self.max_envs = args.num_envs * args.training_len // args.episode_len
        self.num_env = 0
        self.curr_env = 0
        self.step = 0
        
        # Preallocate using memmap
        self.obs = create_memmap('obs.dat', (self.ep_len + 1, self.max_envs, *obs_dim), np.uint8)
        self.value_preds = create_memmap('value_preds.dat', (self.ep_len + 1, self.max_envs, 1), np.float32)
        self.returns = create_memmap('returns.dat', (self.ep_len, self.max_envs, 1), np.float32)
        self.actions = create_memmap('actions.dat', (self.ep_len, self.max_envs, act_dim), np.int32)
        self.action_log_probs = create_memmap('action_log_probs.dat', (self.ep_len, self.max_envs, 1), np.float32)
        self.rewards = create_memmap('rewards.dat', (self.ep_len, self.max_envs, 1), np.float32)
        
        # CRITICAL FIX: Masks must default to 1.0 (non-terminal). Memmap defaults to 0.0, causing instant advantage collapse.
        self.masks = create_memmap('masks.dat', (self.ep_len + 1, self.max_envs, 1), np.float32)
        self.masks[:] = 1.0
        
        self.advantages = create_memmap('advantages.dat', (self.ep_len, self.max_envs, 1), np.float32)
        self.instructions = [""] * self.max_envs

    def _to_numpy(self, x):
        """Helper to convert tensors or arrays to numpy safely, handling BFloat16."""
        if isinstance(x, torch.Tensor):
            if x.dtype == torch.bfloat16:
                x = x.float()
            return x.detach().cpu().numpy()
        return np.array(x)

    def warmup(self, obs, instruction):
        """Initializes the buffer with the starting observation and instruction for current segment."""
        b = obs.shape[0]
        start, end = self.curr_env, self.curr_env + b
        
        self.obs[0, start:end] = self._to_numpy(obs)
        self.instructions[start:end] = instruction if isinstance(instruction, list) else [instruction] * b
        self.masks[0, start:end] = 1.0
        self.step = 0

    def insert(self, obs, actions, action_log_probs, value_preds, rewards, masks):
        """Inserts a single step of data for the current segment."""
        b = obs.shape[0]
        start, end = self.curr_env, self.curr_env + b
        
        self.obs[self.step + 1, start:end] = self._to_numpy(obs)
        self.actions[self.step, start:end] = self._to_numpy(actions)
        self.action_log_probs[self.step, start:end] = self._to_numpy(action_log_probs).reshape(b, 1)
        self.value_preds[self.step, start:end] = self._to_numpy(value_preds).reshape(b, 1)
        self.rewards[self.step, start:end] = self._to_numpy(rewards).reshape(b, 1)
        self.masks[self.step + 1, start:end] = self._to_numpy(masks).reshape(b, 1)

        self.step = (self.step + 1) % self.ep_len

    def end_segment(self, next_value):
        """Finalizes the current segment and prepares for the next one."""
        b = next_value.shape[0] if isinstance(next_value, (torch.Tensor, np.ndarray)) else len(next_value)
        start, end = self.curr_env, self.curr_env + b
        
        self.value_preds[-1, start:end] = self._to_numpy(next_value).reshape(b, 1)
        self.curr_env += b
        self.num_env = self.curr_env
        self.step = 0

    def compute_gae(self):
        """Computes Generalized Advantage Estimation (GAE) with normalization."""
        gae = 0
        for step in reversed(range(self.ep_len)):
            # Using slice to handle only valid envs
            masks_next = self.masks[step + 1, :self.num_env]
            v_next = self.value_preds[step + 1, :self.num_env]
            v_curr = self.value_preds[step, :self.num_env]
            rew = self.rewards[step, :self.num_env]
            
            delta = rew + self.gamma * v_next * masks_next - v_curr
            gae = delta + self.gamma * self.gae_lambda * masks_next * gae
            self.returns[step, :self.num_env] = gae + v_curr
            
        adv = self.returns[:, :self.num_env] - self.value_preds[:-1, :self.num_env]
        self.advantages[:, :self.num_env] = (adv - adv.mean()) / (adv.std() + 1e-5)

    def feed_forward_generator(self):
        """Yields minibatches for PPO training with optimized indexing."""
        batch_size = self.ep_len * self.num_env
        num_minibatches = batch_size // self.minibatch_size
        indices = np.random.permutation(batch_size)
        
        # Flattened views for sampling, sliced by valid envs
        obs_flat = self.obs[:-1, :self.num_env].reshape(-1, *self.obs.shape[2:])
        act_flat = self.actions[:, :self.num_env].reshape(-1, self.actions.shape[-1])
        logp_flat = self.action_log_probs[:, :self.num_env].reshape(-1, 1)
        ret_flat = self.returns[:, :self.num_env].reshape(-1, 1)
        adv_flat = self.advantages[:, :self.num_env].reshape(-1, 1)
        val_flat = self.value_preds[:-1, :self.num_env].reshape(-1, 1)
        
        for i in range(num_minibatches):
            idx = indices[i * self.minibatch_size : (i + 1) * self.minibatch_size]
            yield {
                "obs": obs_flat[idx],
                "instruct": [self.instructions[j % self.num_env] for j in idx],
                "actions": act_flat[idx],
                "values": val_flat[idx],
                "returns": ret_flat[idx],
                "logprobs": logp_flat[idx],
                "advantages": adv_flat[idx]
            }

    def reset(self):
        """Clears the buffer status."""
        self.num_env = self.curr_env = self.step = 0
        gc.collect()

    def cleanup(self):
        """Manually closes and removes memmap files directory."""
        self.obs = self.value_preds = self.returns = self.actions = None
        self.action_log_probs = self.rewards = self.masks = self.advantages = None
        gc.collect()
        
        buffer_dir = os.path.join(os.getcwd(), f"cronos_mmap_{os.getpid()}")
        if os.path.exists(buffer_dir):
            import shutil
            try:
                shutil.rmtree(buffer_dir)
                print(f"Cleaned up {buffer_dir}")
            except Exception as e:
                print(f"Cleanup error: {e}")
