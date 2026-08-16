import numpy as np
import torch
import os
import gc
import shutil
import time


def _buffer_dir() -> str:
    return os.path.join(os.getcwd(), f"cronos_mmap_{os.getpid()}")


def create_memmap(filename, shape, dtype):
    """Utility to create memory-mapped files for large buffers."""
    buffer_dir = _buffer_dir()
    os.makedirs(buffer_dir, exist_ok=True)
    filepath = os.path.join(buffer_dir, filename)
    if os.path.exists(filepath):
        os.remove(filepath)
    return np.memmap(filepath, dtype=dtype, mode='w+', shape=shape)


def _close_mmap(arr) -> None:
    """Release the underlying mmap on a numpy.memmap (idempotent, best-effort)."""
    if arr is None:
        return
    try:
        inner = getattr(arr, "_mmap", None)
        if inner is not None:
            inner.close()
    except Exception:
        pass


def _rmtree_retry(path: str, attempts: int = 3, delay: float = 0.5) -> None:
    """Remove `path` with retry. Survives residual silly-rename entries that
    may linger a few hundred ms on NFS after the last fd is released."""
    for i in range(attempts):
        if not os.path.exists(path):
            return
        try:
            shutil.rmtree(path, ignore_errors=(i == 0))
            if not os.path.exists(path):
                return
        except Exception:
            pass
        time.sleep(delay)
    try:
        shutil.rmtree(path, ignore_errors=True)
    except Exception:
        pass

class CronosReplayBuffer:
    """Efficient memory-mapped replay buffer for PPO training."""
    
    def __init__(self, args, obs_dim=(480, 640, 3), act_dim=7):
        self.ep_len = args.segment_len
        self.gamma = args.buffer_gamma
        self.gae_lambda = args.buffer_lambda
        self.minibatch_size = args.buffer_minibatch
        
        self.max_envs = args.num_envs * args.episode_len // args.segment_len
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
        self._closed = False

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

    def compute_grpo_returns(self, group_ids=None, fix=True, std_scope="group"):
        """GRPO returns/advantages — the critic-free counterpart of `compute_gae`.

        Port of AutoRL `SeparatedReplayBuffer.compute_returns_grpo`
        (`AutoRL/SimplerEnv/simpler_env/utils/replay_buffer.py:107-122`), with
        the grouping and the std term made explicit instead of hard-wired.

        The normalization is **step-level**, not trajectory-level: individual
        `r_t` entries are centred/scaled, then accumulated into an *undiscounted*
        reward-to-go. So `buffer_gamma` and `buffer_lambda` are unused here, and
        the advantage varies along a trajectory. This matches AutoRL exactly.

        Args:
            group_ids: per-slot group id, length `num_env` (the filled slot
                count), or None for one global group. Slot `k` is segment
                `k // num_envs`, env `k % num_envs`, so the caller decides
                whether "same group" means same task, same scene (YAML group),
                or the whole batch — see `CronosRunner._grpo_group_ids`. None
                reproduces AutoRL, which does no grouping.
            fix: AutoRL's `alg_grpo_fix`. True → compute the statistics from the
                **non-zero** rewards only and leave the zeros untouched. The
                shaped reward is a potential *difference* (`envs/reward.py`), so
                most steps are exactly 0; including them would drag the mean to
                ~0 and inflate the std by the sparsity ratio rather than by
                anything about the policy.
            std_scope: what to divide the centred rewards by.
                ``"group"``  — that group's own std (AutoRL / textbook GRPO).
                ``"global"`` — the std over the whole update, so groups are
                               centred independently but weighted equally.
                ``"none"``   — no division (centring only).
                See `doc/grpo_autorl.md` for the trade-off; the short version is
                that per-group std doubles as a per-group *weight*, which is
                harmless when the group is the whole batch and distorting when
                the group is a handful of envs.

        Writes `returns` and `advantages` in place over `[:, :num_env]`; shapes
        are unchanged, so `feed_forward_generator` needs no changes.
        """
        if std_scope not in ("group", "global", "none"):
            raise ValueError(
                f"std_scope must be 'group', 'global' or 'none', got {std_scope!r}")

        n = self.num_env
        if n == 0:
            return

        rewards = np.asarray(self.rewards[:, :n])          # (ep_len, n, 1)
        norm = rewards.copy()

        # Entries that participate in the statistics AND get rescaled.
        stat_mask = (rewards != 0) if fix else np.ones_like(rewards, dtype=bool)

        # Global scale, used by std_scope="global". Computed over the same
        # entries the per-group pass will touch, so the two agree on what counts.
        all_vals = rewards[stat_mask]
        global_std = all_vals.std() if all_vals.size else np.float32(0.0)

        if group_ids is None:
            col_sets = [np.arange(n)]
        else:
            gid = np.asarray(group_ids)[:n]
            if gid.shape[0] != n:
                raise ValueError(
                    f"group_ids has {gid.shape[0]} entries but the buffer holds "
                    f"{n} slots")
            col_sets = [np.nonzero(gid == g)[0] for g in np.unique(gid)]

        for cols in col_sets:
            block = norm[:, cols]                          # fancy index → copy
            mask = stat_mask[:, cols]
            vals = rewards[:, cols][mask]
            # A group whose rewards are all exactly zero has nothing to compare.
            # Leave it at zero (no gradient) rather than dividing by an empty
            # mean/std, which is a nan and a RuntimeWarning. AutoRL only ever
            # takes the whole batch so it never hits this; per-task groups do,
            # routinely, early in training.
            if vals.size == 0:
                continue
            # In-place, in this order, to stay bit-identical to AutoRL's
            # `rn[rn != 0] -= mean` / `/= std` on a float32 array.
            sub = block[mask]
            sub -= vals.mean()
            if std_scope == "group":
                sub /= (vals.std() + 1e-5)
            elif std_scope == "global":
                sub /= (global_std + 1e-5)
            block[mask] = sub
            norm[:, cols] = block

        # Undiscounted, mask-terminated reward-to-go (AutoRL: no gamma).
        acc = 0
        for step in reversed(range(self.ep_len)):
            acc = norm[step] + self.masks[step + 1, :n] * acc
            self.returns[step, :n] = acc

        # GRPO has no baseline: the advantage IS the normalized return.
        self.advantages[:, :n] = self.returns[:, :n]

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

    def close(self):
        """Synchronously release every underlying mmap. Idempotent.

        Calling `_mmap.close()` on each np.memmap before rmtree is what
        prevents the `ENOTEMPTY` silly-rename race on NFS: GC-driven release
        can lag the rmtree call by tens of milliseconds, which is enough
        for NFS to leave residual `.nfs*` entries behind.
        """
        if self._closed:
            return
        for name in (
            "obs", "value_preds", "returns", "actions",
            "action_log_probs", "rewards", "masks", "advantages",
        ):
            _close_mmap(getattr(self, name, None))
            setattr(self, name, None)
        gc.collect()
        self._closed = True

    def cleanup(self):
        """Manually closes mmaps and removes the buffer directory with retry."""
        self.close()
        buffer_dir = _buffer_dir()
        if os.path.exists(buffer_dir):
            _rmtree_retry(buffer_dir)
            if not os.path.exists(buffer_dir):
                print(f"Cleaned up {buffer_dir}")
            else:
                print(f"Cleanup warning: {buffer_dir} not fully removed")
