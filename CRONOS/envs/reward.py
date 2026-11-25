import torch


# Per-env goal mode. The forward goal and the two backward (LSR reset) goals.
#
# MODE_BACK_TABLE is the historical LSR goal: "put X on table", scored by
# `src_on_table`. MODE_BACK_RECEP is the perturbation variant: instead of always
# returning the object to the one canonical table state, the reset policy
# sometimes places it on a DIFFERENT receptacle than the forward task's. That
# widens the state distribution the forward policy starts its next segment from
# — the point of the perturbation controller in arXiv:2004.12570 §4.1, reached
# through the reset policy rather than through a separate high-entropy actor.
#
# MODE_BACK_RECEP deliberately shares the FORWARD reward branch: the caller
# swaps the env's target receptacle via `set_current_task`, so the env's own
# `success` predicate already means "object is on the other receptacle", and
# `get_language_instruction` already emits "put X on <other receptacle>". No new
# reward term and no new task string are needed — only the goal selection.
MODE_FORWARD = 0
MODE_BACK_TABLE = 1
MODE_BACK_RECEP = 2

VALID_MODES = (MODE_FORWARD, MODE_BACK_TABLE, MODE_BACK_RECEP)


class RewardShaper:
    """Computes potential-based reward diff, matching AutoRL's get_reward() exactly.

    The reward uses info fields from the env's native evaluate() — NOT a
    reimplemented compute_info(). This ensures the success threshold, grasp
    detection, and consecutive_grasp counter are all the env's authoritative
    values (xy_flag=0.01, env-side consecutive_grasp counter, etc.).
    """

    def __init__(self, num_envs=64, device="cuda"):
        self.num_envs = num_envs
        self.device = device
        self.reward_old = torch.zeros(self.num_envs, 1, dtype=torch.float32, device=self.device)
        # Per-env goal mode (see the constants above). Replaces the previous
        # bool `backward` tensor, which is now derived from it so every existing
        # reader keeps working.
        self.mode = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

    @property
    def backward(self):
        """Envs whose goal is "put X on table".

        Deliberately excludes MODE_BACK_RECEP: that mode's reward and its
        language instruction are the forward ones, evaluated against a swapped
        target receptacle. Consumers of this flag (`compute_reward`'s branch and
        `CronosWrapper.get_language_instructions`' rewrite) must NOT fire for it.
        """
        return self.mode == MODE_BACK_TABLE

    @property
    def is_reset_segment(self):
        """Envs running either backward goal — for logging and bookkeeping only."""
        return self.mode != MODE_FORWARD

    def set_backward_mask(self, mask):
        """Sets which environments are in 'backward to table' mode.

        Kept for the two-state callers (`set_forward` / `set_backward`); the
        three-state entry point is `set_mode`.
        """
        mask = mask.to(torch.bool)
        self.mode = torch.where(
            mask,
            torch.full_like(self.mode, MODE_BACK_TABLE),
            torch.zeros_like(self.mode),
        )

    def set_mode(self, mode):
        """Sets the per-env goal mode directly (long tensor of VALID_MODES)."""
        mode = mode.to(dtype=torch.long, device=self.device).reshape(-1)
        assert mode.shape[0] == self.num_envs, \
            f"mode must have {self.num_envs} entries, got {mode.shape[0]}"
        self.mode = mode

    def compute_reward(self, info):
        """Computes potential-based reward diff matching AutoRL's get_reward().

        Uses env's native info dict (from env.step → evaluate()). The fields
        ``is_src_obj_grasped``, ``consecutive_grasp``, ``success``, and
        ``src_on_table`` are all produced by the env's evaluate() method with
        the authoritative xy_flag=0.01 threshold.

        Two branches, not three: MODE_BACK_RECEP takes the ``success`` branch
        because its target receptacle has already been swapped in the env, so
        ``info["success"]`` *is* "object on the other receptacle".
        """
        backward_mask = self.backward.reshape(-1, 1)

        reward = torch.zeros(self.num_envs, 1, dtype=torch.float32, device=self.device)

        reward += info["is_src_obj_grasped"].reshape(-1, 1) * 0.1
        reward += info["consecutive_grasp"].reshape(-1, 1) * 0.1

        reward += torch.where(
            backward_mask,
            (info["src_on_table"].reshape(-1, 1) & info["is_src_obj_grasped"].reshape(-1, 1)) * 1.0,
            (info["success"].reshape(-1, 1) & info["is_src_obj_grasped"].reshape(-1, 1)) * 1.0,
        )

        reward_diff = reward - self.reward_old
        self.reward_old = reward

        return reward_diff
