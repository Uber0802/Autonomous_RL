import torch


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
        self.backward = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

    def set_backward_mask(self, mask):
        """Sets which environments are in 'backward' mode."""
        self.backward = mask

    def compute_reward(self, info):
        """Computes potential-based reward diff matching AutoRL's get_reward().

        Uses env's native info dict (from env.step → evaluate()). The fields
        ``is_src_obj_grasped``, ``consecutive_grasp``, ``success``, and
        ``src_on_table`` are all produced by the env's evaluate() method with
        the authoritative xy_flag=0.01 threshold.
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
