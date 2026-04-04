"""
cogact_model.py

GaussianActionHead, ValueHead, and CogACTForRL — CogACT VLM backbone
with Gaussian action head and value head for PPO-compatible RL training.
"""

import torch
import torch.nn as nn
from torch.distributions import Normal


class GaussianActionHead(nn.Module):
    """
    Maps cognition token [B, 4096] to a diagonal Gaussian over 7-dim actions.
    Outputs mean and log_std; provides log_prob() and entropy().
    """

    def __init__(self, hidden_dim: int = 4096, action_dim: int = 7,
                 init_log_std: float = -2.0):
        super().__init__()
        self.action_dim = action_dim
        self.mean_net = nn.Sequential(
            nn.Linear(hidden_dim, 512),
            nn.Tanh(),
            nn.Linear(512, action_dim),
        )
        self.log_std = nn.Parameter(torch.full((action_dim,), init_log_std))

    def forward(self, cognition: torch.Tensor) -> Normal:
        """
        Args:
            cognition: [B, hidden_dim] — pooled cognition token
        Returns:
            Normal distribution over [B, action_dim]
        """
        mean = self.mean_net(cognition)  # [B, action_dim]
        # Clamp log_std to prevent collapse (<-4 → std<0.018) or explosion (>0 → std>1)
        log_std = self.log_std.clamp(min=-4.0, max=0.0)
        std = log_std.exp().expand_as(mean)
        return Normal(mean, std)

    def get_action(self, cognition: torch.Tensor, deterministic: bool = False):
        """
        Returns:
            action: [B, action_dim]
            log_prob: [B, 1]
        """
        dist = self.forward(cognition)
        if deterministic:
            action = dist.mean
        else:
            action = dist.rsample()
        log_prob = dist.log_prob(action).sum(dim=-1, keepdim=True)  # [B, 1]
        return action, log_prob

    def evaluate(self, cognition: torch.Tensor, action: torch.Tensor):
        """
        Returns:
            log_prob: [B, 1]
            entropy: [B, 1]
        """
        dist = self.forward(cognition)
        log_prob = dist.log_prob(action).sum(dim=-1, keepdim=True)  # [B, 1]
        entropy = dist.entropy().sum(dim=-1, keepdim=True)  # [B, 1]
        return log_prob, entropy


class ValueHead(nn.Module):
    """
    Maps cognition token [B, 4096] to scalar value V(s).
    Same architecture as AutoRL's value head but standalone.
    """

    def __init__(self, hidden_dim: int = 4096):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, 512),
            nn.Tanh(),
            nn.Linear(512, 1),
        )

    def forward(self, cognition: torch.Tensor) -> torch.Tensor:
        """
        Args:
            cognition: [B, hidden_dim]
        Returns:
            value: [B, 1]
        """
        return self.net(cognition)
