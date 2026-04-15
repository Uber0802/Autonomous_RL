"""
Standalone ActionDecoder module for CRONOS × UniVLA integration.

Extracted from:
  UniVLA/experiments/robot/simpler-bridge/policies/univla/univla_model.py

Only ActionDecoderHead and ActionDecoder are included.
All TensorFlow, OpenCV, matplotlib, and transforms3d imports are removed.
UniVLABridgeInference, get_vla, get_processor, and crop_and_resize are NOT included.

The MAPBlock dependency is satisfied by the prismatic subtree already present in
cronos_univla/UniVLA/prismatic/models/policy/transformer_utils.py.
"""

import numpy as np
import torch
import torch.nn as nn

from prismatic.models.policy.transformer_utils import MAPBlock


class ActionDecoderHead(torch.nn.Module):
    def __init__(self, window_size: int = 5, hidden_dim: int = 512):
        super().__init__()
        self.latent_action_pool = MAPBlock(n_latents=1, vis_dim=4096, embed_dim=hidden_dim, n_heads=hidden_dim // 64)
        self.visual_pool = MAPBlock(n_latents=1, vis_dim=4096, embed_dim=hidden_dim, n_heads=hidden_dim // 64)

        self.proj = nn.Sequential(
            nn.Linear(hidden_dim, 7 * window_size),
            nn.Tanh(),
        )

    def forward(self, latent_action_tokens, visual_embed):
        latent_action_tokens = latent_action_tokens[:, -4:]
        visual_embed = self.visual_pool(visual_embed)
        action = self.proj(self.latent_action_pool(latent_action_tokens, init_embed=visual_embed))

        return action


class ActionDecoder(nn.Module):
    def __init__(self, window_size=10):
        super().__init__()
        self.net = ActionDecoderHead(window_size=window_size)

        self.temporal_size = window_size
        self.temporal_mask = torch.flip(
            torch.triu(torch.ones(self.temporal_size, self.temporal_size, dtype=torch.bool)), dims=[1]
        ).numpy()

        self.action_buffer = np.zeros((self.temporal_mask.shape[0], self.temporal_mask.shape[0], 7))
        self.action_buffer_mask = np.zeros((self.temporal_mask.shape[0], self.temporal_mask.shape[0]), dtype=np.bool_)

        # Action chunking with temporal aggregation
        balancing_factor = 0.1
        self.temporal_weights = np.array(
            [np.exp(-1 * balancing_factor * i) for i in range(self.temporal_size)]
        )[:, None]

    def reset(self):
        self.action_buffer = np.zeros((self.temporal_mask.shape[0], self.temporal_mask.shape[0], 7))
        self.action_buffer_mask = np.zeros((self.temporal_mask.shape[0], self.temporal_mask.shape[0]), dtype=np.bool_)

    def forward(self, latent_actions, visual_embed, mask, action_low, action_high):
        # Forward action decoder
        pred_action = self.net(latent_actions.to(torch.float), visual_embed.to(torch.float)).reshape(
            -1, self.temporal_size, 7
        )
        pred_action = np.array(pred_action.tolist())

        # Shift action buffer
        self.action_buffer[1:, :, :] = self.action_buffer[:-1, :, :]
        self.action_buffer_mask[1:, :] = self.action_buffer_mask[:-1, :]
        self.action_buffer[:, :-1, :] = self.action_buffer[:, 1:, :]
        self.action_buffer_mask[:, :-1] = self.action_buffer_mask[:, 1:]
        self.action_buffer_mask = self.action_buffer_mask * self.temporal_mask

        # Add to action buffer
        self.action_buffer[0] = pred_action
        self.action_buffer_mask[0] = np.array([True] * self.temporal_mask.shape[0], dtype=np.bool_)

        # Ensemble temporally to predict action
        action_prediction = np.sum(
            self.action_buffer[:, 0, :] * self.action_buffer_mask[:, 0:1] * self.temporal_weights, axis=0
        ) / np.sum(self.action_buffer_mask[:, 0:1] * self.temporal_weights)
        action_prediction = np.where(
            mask,
            0.5 * (action_prediction + 1) * (action_high - action_low) + action_low,
            action_prediction,
        )

        return action_prediction
