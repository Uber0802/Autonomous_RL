import torch
import numpy as np
import gc

class SeparatedReplayBuffer(object):
    def __init__(self, all_args, obs_dim, act_dim):
        self.ep_len = all_args.episode_len
        self.num_env = all_args.num_envs
        self.gamma = all_args.buffer_gamma
        self.gae_lambda = all_args.buffer_lambda
        self.buffer_minibatch = all_args.buffer_minibatch
        self.alg_grpo_fix = all_args.alg_grpo_fix

        self.obs = np.zeros((self.ep_len + 1, self.num_env, *obs_dim), dtype=np.uint8)
        self.instruction = [""] * self.num_env
        self.value_preds = np.zeros((self.ep_len + 1, self.num_env, 1), dtype=np.float32)
        self.returns = np.zeros((self.ep_len, self.num_env, 1), dtype=np.float32)
        self.actions = np.zeros((self.ep_len, self.num_env, act_dim), dtype=np.int32)
        self.action_log_probs = np.zeros((self.ep_len, self.num_env, act_dim), dtype=np.float32)
        self.rewards = np.zeros((self.ep_len, self.num_env, 1), dtype=np.float32)
        self.masks = np.ones((self.ep_len + 1, self.num_env, 1), dtype=np.float32)

        self.advantages = np.zeros((self.ep_len, self.num_env, 1), dtype=np.float32)

        self.step = 0

    def cat_buffer(self, buffer: 'SeparatedReplayBuffer'):
        assert self.ep_len == buffer.ep_len, "Episode lengths must match"
        assert self.obs.shape[0] == buffer.obs.shape[0], "Obs time dimension mismatch"
        assert self.obs.shape[2:] == buffer.obs.shape[2:], "Obs shape mismatch"

        def print_shape_diff(name, before, after):
            print(f"{name}: {before} -> {after}")

        # Print shapes before
        print("Before concatenation:")
        print(f"obs: {self.obs.shape}, buffer.obs: {buffer.obs.shape}")
        print(f"value_preds: {self.value_preds.shape}, buffer: {buffer.value_preds.shape}")
        print(f"returns: {self.returns.shape}, buffer: {buffer.returns.shape}")
        print(f"actions: {self.actions.shape}, buffer: {buffer.actions.shape}")
        print(f"action_log_probs: {self.action_log_probs.shape}, buffer: {buffer.action_log_probs.shape}")
        print(f"rewards: {self.rewards.shape}, buffer: {buffer.rewards.shape}")
        print(f"masks: {self.masks.shape}, buffer: {buffer.masks.shape}")
        print(f"advantages: {self.advantages.shape}, buffer: {buffer.advantages.shape}")

        self.obs = np.concatenate([self.obs, buffer.obs], axis=1)
        self.instruction += buffer.instruction  # List of strings
        self.value_preds = np.concatenate([self.value_preds, buffer.value_preds], axis=1)
        self.returns = np.concatenate([self.returns, buffer.returns], axis=1)
        self.actions = np.concatenate([self.actions, buffer.actions], axis=1)
        self.action_log_probs = np.concatenate([self.action_log_probs, buffer.action_log_probs], axis=1)
        self.rewards = np.concatenate([self.rewards, buffer.rewards], axis=1)
        self.masks = np.concatenate([self.masks, buffer.masks], axis=1)
        self.advantages = np.concatenate([self.advantages, buffer.advantages], axis=1)

        print("After concatenation:")
        print_shape_diff("obs", buffer.obs.shape, self.obs.shape)
        print_shape_diff("value_preds", buffer.value_preds.shape, self.value_preds.shape)
        print_shape_diff("returns", buffer.returns.shape, self.returns.shape)
        print_shape_diff("actions", buffer.actions.shape, self.actions.shape)
        print_shape_diff("action_log_probs", buffer.action_log_probs.shape, self.action_log_probs.shape)
        print_shape_diff("rewards", buffer.rewards.shape, self.rewards.shape)
        print_shape_diff("masks", buffer.masks.shape, self.masks.shape)
        print_shape_diff("advantages", buffer.advantages.shape, self.advantages.shape)

        self.num_env += buffer.num_env
        if self.step != buffer.step:
            print(f"[Warning] Step mismatch: self.step={self.step}, buffer.step={buffer.step}. Using self.step.")

    def insert(self, obs, actions, action_log_probs, value_preds, rewards, masks):
        self.obs[self.step + 1] = obs.copy()
        self.actions[self.step] = actions.copy()
        self.action_log_probs[self.step] = action_log_probs.copy()
        self.value_preds[self.step] = value_preds.copy()
        self.rewards[self.step] = rewards.copy()
        self.masks[self.step + 1] = masks.copy()

        self.step = (self.step + 1) % self.ep_len

    def warmup(self, obs, instruction):
        self.obs[0] = obs
        self.instruction = instruction
        self.masks[0] = 1.0

        self.step = 0

    def endup(self, next_value):
        self.value_preds[-1] = next_value

    def compute_returns_ppo(self):
        gae = 0
        for step in reversed(range(self.rewards.shape[0])):
            vt1 = self.value_preds[step + 1]
            vt = self.value_preds[step]

            delta = self.rewards[step] + self.gamma * vt1 * self.masks[step + 1] - vt
            gae = delta + self.gamma * self.gae_lambda * self.masks[step + 1] * gae
            self.returns[step] = gae + vt

        # calc adv
        advantages = self.returns - self.value_preds[:-1]
        mean_advantages = advantages.mean()
        std_advantages = advantages.std()
        self.advantages = (advantages - mean_advantages) / (std_advantages + 1e-5)

    def compute_returns_grpo(self):
        if self.alg_grpo_fix:
            rewards_valid = self.rewards[self.rewards != 0]
            rewards_norm = self.rewards.copy()
            rewards_norm[rewards_norm != 0] -= rewards_valid.mean()
            rewards_norm[rewards_norm != 0] /= (rewards_valid.std() + 1e-5)
        else:
            rewards_norm = (self.rewards - self.rewards.mean()) / (self.rewards.std() + 1e-5)

        returns = 0
        for step in reversed(range(self.rewards.shape[0])):
            returns = rewards_norm[step] + self.masks[step + 1] * returns
            self.returns[step] = returns

        # calc adv
        self.advantages = self.returns.copy()

    def get_minibatch_count(self):
        episode_length, n_rollout_threads = self.rewards.shape[:2]
        batch_size = episode_length * n_rollout_threads

        if self.buffer_minibatch < 0:
            num_mini_batch = 1
        else:
            assert batch_size % self.buffer_minibatch == 0
            num_mini_batch = batch_size // self.buffer_minibatch

        return num_mini_batch

    def feed_forward_generator(self):
        episode_length, n_rollout_threads = self.rewards.shape[:2]
        batch_size = episode_length * n_rollout_threads

        if self.buffer_minibatch < 0:
            num_mini_batch = 1
        else:
            assert batch_size % self.buffer_minibatch == 0
            num_mini_batch = batch_size // self.buffer_minibatch

        rand = torch.randperm(batch_size).numpy()
        sampler = [rand[i * self.buffer_minibatch:(i + 1) * self.buffer_minibatch] for i in range(num_mini_batch)]

        obs = self.obs[:-1].reshape(-1, *self.obs.shape[2:])
        actions = self.actions.reshape(-1, self.actions.shape[-1])
        value_preds = self.value_preds[:-1].reshape(-1, 1)
        returns = self.returns.reshape(-1, 1)
        masks = self.masks[:-1].reshape(-1, 1)
        action_logits = self.action_log_probs.reshape(-1, self.action_log_probs.shape[-1])
        advantages = self.advantages.reshape(-1, 1)

        for indices in sampler:
            # obs size [T+1 N Dim]-->[T N Dim]-->[T*N,Dim]-->[index,Dim]
            obs_batch = obs[indices]
            actions_batch = actions[indices]
            value_preds_batch = value_preds[indices]
            return_batch = returns[indices]
            masks_batch = masks[indices]
            old_action_logits_batch = action_logits[indices]
            adv_targ = advantages[indices]

            # instruct
            instruct_indices = indices % n_rollout_threads
            instruct_batch = [self.instruction[i] for i in instruct_indices]

            yield (obs_batch, instruct_batch, actions_batch, value_preds_batch, return_batch, masks_batch,
                   old_action_logits_batch, adv_targ)

    def update_instruction(self, new_instruction):
        """
        將目前 buffer 的 instruction 替換為新的 instruction 列表。

        Args:
            new_instruction (List[str]): 長度為 num_env 的新指令列表
        """
        assert isinstance(new_instruction, list)
        assert len(new_instruction) == self.num_env
        self.instruction = new_instruction

def create_memmap(filename, shape, dtype):
    if os.path.exists(filename):
        os.remove(filename)  # overwrite existing file
    return np.memmap(filename, dtype=dtype, mode='w+', shape=shape)

class PreallocReplayBuffer(SeparatedReplayBuffer):
    def __init__(self, all_args, obs_dim, act_dim):
        self.ep_len = all_args.episode_len
        self.num_env = 0
        self.gamma = all_args.buffer_gamma
        self.gae_lambda = all_args.buffer_lambda
        self.buffer_minibatch = all_args.buffer_minibatch
        self.alg_grpo_fix = all_args.alg_grpo_fix

        max_num_envs = all_args.num_envs * all_args.training_interval // all_args.episode_len
        self.curr_env = 0

        # Preallocate huge buffers
        self.obs = create_memmap('obs.dat', (self.ep_len + 1, max_num_envs, *obs_dim), np.uint8)
        self.value_preds = create_memmap('value_preds.dat', (self.ep_len + 1, max_num_envs, 1), np.float32)
        self.returns = create_memmap('returns.dat', (self.ep_len, max_num_envs, 1), np.float32)
        self.actions = create_memmap('actions.dat', (self.ep_len, max_num_envs, act_dim), np.int32)
        self.action_log_probs = create_memmap('action_log_probs.dat', (self.ep_len, max_num_envs, act_dim), np.float32)
        self.rewards = create_memmap('rewards.dat', (self.ep_len, max_num_envs, 1), np.float32)
        self.masks = create_memmap('masks.dat', (self.ep_len + 1, max_num_envs, 1), np.float32)
        self.advantages = create_memmap('advantages.dat', (self.ep_len, max_num_envs, 1), np.float32)

        self.instruction = [""] * max_num_envs

        self.step = 0
        self.max_num_envs = max_num_envs

    def cat_buffer(self, buffer: 'SeparatedReplayBuffer'):
        if self.curr_env + buffer.num_env > self.max_num_envs:
            raise MemoryError(f"Buffer overflow: {self.curr_env + buffer.num_env} > {self.max_num_envs}")

        start, end = self.curr_env, self.curr_env + buffer.num_env

        self.obs[:, start:end] = buffer.obs
        self.instruction[start:end] = buffer.instruction
        self.value_preds[:, start:end] = buffer.value_preds
        self.returns[:, start:end] = buffer.returns
        self.actions[:, start:end] = buffer.actions
        self.action_log_probs[:, start:end] = buffer.action_log_probs
        self.rewards[:, start:end] = buffer.rewards
        self.masks[:, start:end] = buffer.masks
        self.advantages[:, start:end] = buffer.advantages

        self.curr_env = end
        self.num_env += buffer.num_env
        print(f"Num Envs In Prealloc: {self.num_env}.")

        if self.step != buffer.step:
            print(f"[Warning] Step mismatch: self.step={self.step}, buffer.step={buffer.step}")
        
        del buffer
        gc.collect()

        
    def reset(self):
        self.step = 0
        self.curr_env = 0
        self.num_env = 0
        print("Reset PreallocReplayBuffer")