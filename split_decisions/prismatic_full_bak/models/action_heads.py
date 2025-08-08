"""Implementations of various action heads, which serve as alternatives to VLM sequential token prediction."""

import math

import numpy as np
import torch
import torch.nn as nn
from ..vla.constants import ACTION_DIM, ACTION_TOKEN_BEGIN_IDX, IGNORE_INDEX, NUM_ACTIONS_CHUNK, PROPRIO_DIM, STOP_INDEX


class SinusoidalPositionalEncoding(nn.Module):
    """
    Sine- and cosine-based positional encoding that produces embeddings of a batch of timesteps.

    For example, at train time, the input might be a batch of 32 randomly sampled diffusion timesteps -> shape (32,)
    Then the output would be a batch of 32 timestep embeddings -> shape (32, D)

    Adapted from: https://github.com/real-stanford/diffusion_policy/blob/main/diffusion_policy/model/diffusion/positional_embedding.py
    """

    def __init__(self, dim):
        super().__init__()
        self.dim = dim  # dimensionality of the positional encoding

    def forward(self, x):
        # x: (batch_size,)
        device = x.device
        assert self.dim % 2 == 0, f"# dimensions must be even but got {self.dim}"
        half_dim = self.dim // 2
        exponent = torch.arange(half_dim, device=device) * -math.log(10000) / (half_dim - 1)  # shape: (D/2,)
        emb = torch.exp(exponent)  # shape: (D/2,)
        emb = x[:, None] * emb[None, :]  # shape: (batch_size, 1) * (1, D/2) -> (batch_size, D/2)
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)  # shape: (batch_size, D)
        return emb


class MLPResNetBlock(nn.Module):
    """One MLP ResNet block with a residual connection."""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.ffn = nn.Sequential(  # feedforward network, similar to the ones in Transformers
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.ReLU(),
        )

    def forward(self, x):
        # x: (batch_size, hidden_dim)
        # We follow the module ordering of "Pre-Layer Normalization" feedforward networks in Transformers as
        # described here: https://arxiv.org/pdf/2002.04745.pdf
        identity = x
        x = self.ffn(x)
        x = x + identity
        return x


class MLPResNet(nn.Module):
    """MLP with residual connection blocks."""
    def __init__(self, num_blocks, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.layer_norm1 = nn.LayerNorm(input_dim)
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.mlp_resnet_blocks = nn.ModuleList()
        for _ in range(num_blocks):
            self.mlp_resnet_blocks.append(MLPResNetBlock(dim=hidden_dim))
        self.layer_norm2 = nn.LayerNorm(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # x: (batch_size, input_dim)
        x = self.layer_norm1(x)  # shape: (batch_size, input_dim)
        x = self.fc1(x)  # shape: (batch_size, hidden_dim)
        x = self.relu(x)  # shape: (batch_size, hidden_dim)
        for block in self.mlp_resnet_blocks:
            x = block(x)  # shape: (batch_size, hidden_dim)
        x = self.layer_norm2(x)  # shape: (batch_size, hidden_dim)
        x = self.fc2(x)  # shape: (batch_size, output_dim)
        return x


class L1RegressionActionHead(nn.Module):
    def __init__(self, input_dim=4096, hidden_dim=4096, action_dim=7):
        super().__init__()
        self.action_dim = action_dim
        self.model = MLPResNet(
            num_blocks=2,
            input_dim=input_dim * 7,
            hidden_dim=hidden_dim,
            output_dim=action_dim
        )

    def predict_action(self, x):
        # x: (batch_size, hidden_dim)
        return torch.tanh(self.model(x))    # → [batch_size, action_dim]


# class MSEActionHead(nn.Module):
#     def __init__(self, input_dim=4096, hidden_dim=4096, action_dim=7):
#         super().__init__()
#         self.net = nn.Sequential(
#             nn.Linear(input_dim, hidden_dim),
#             nn.ReLU(),
#             nn.Linear(hidden_dim, hidden_dim // 2), 
#             nn.ReLU(),
#             nn.Linear(hidden_dim // 2, action_dim),
#         )
#     def predict_action(self, x: torch.Tensor) -> torch.Tensor:
#         return self.net(x)

# class MSEActionHead(nn.Module):
#     def __init__(
#         self, 
#         input_dim: int = 4096,     # 保持與原參數名一致
#         hidden_dim: int = 4096,    # 保留參數但不使用（用於接口兼容）
#         action_dim: int = 7        # 必須保留的關鍵參數
#     ):
#         super().__init__()
#         # 實際使用更深的網絡結構
#         self.mlp = nn.Sequential(
#             nn.Linear(input_dim, 2048),
#             nn.GELU(),
#             nn.LayerNorm(2048),     # 添加歸一化層
#             nn.Linear(2048, 2048),
#             nn.GELU(),
#             nn.LayerNorm(2048),
#             nn.Linear(2048, 2048),
#             nn.GELU(),
#             nn.LayerNorm(2048),
#             nn.Linear(2048, action_dim)
#         )
    
#     def predict_action(self, x: torch.Tensor) -> torch.Tensor:
#         """保持與原方法名一致的重要接口"""
#         return self.mlp(x)

# import torch
# import torch.nn as nn



# class MSEActionHead(nn.Module):
#     def __init__(self,
#                  input_dim:  int = 4096,
#                  hidden_dim: int = 8192,
#                  action_dim: int = 7):
#         super().__init__()
#         self.mlp = nn.Sequential(
#             # layer1: 4096 → 8192
#             nn.Linear(input_dim, hidden_dim), nn.GELU(),
#             # layer2: 8192 → 8192
#             nn.Linear(hidden_dim, hidden_dim), nn.GELU(),
#             # layer3
#             nn.Linear(hidden_dim, hidden_dim), nn.GELU(),
#             # layer4
#             nn.Linear(hidden_dim, hidden_dim), nn.GELU(),
#             # layer5
#             nn.Linear(hidden_dim, hidden_dim), nn.GELU(),
#             # layer6 最後收斂到 action_dim
#             nn.Linear(hidden_dim, action_dim)
#         )
#         # **完全不加 Dropout / LayerNorm / BatchNorm**

#     def predict_action(self, x: torch.Tensor) -> torch.Tensor:
#         return self.mlp(x)

import torch
import torch.nn as nn
import torch.nn.functional as F  # 新增這行

# class MSEActionHead(nn.Module):
#     def __init__(self,
#                  input_dim:  int = 4096,
#                  hidden_dim: int = 16384,
#                  action_dim: int = 7):
#         super().__init__()
#         # 首層線性投影
#         self.input_lin = nn.Linear(input_dim, hidden_dim)
#         # 由 10 個 Block 組成，每個 block 內部兩層 + 殘差
#         blocks = []
#         for _ in range(10):
#             blocks.append(nn.Sequential(
#                 nn.Linear(hidden_dim, hidden_dim),
#                 nn.GELU(),
#                 nn.Linear(hidden_dim, hidden_dim),
#             ))
#         self.blocks = nn.ModuleList(blocks)
#         # 最後投影到 action_dim
#         self.output_lin = nn.Linear(hidden_dim, action_dim)
#         # 完全不加 Dropout / LayerNorm / BatchNorm

#     def predict_action(self, x: torch.Tensor) -> torch.Tensor:
#         # x: (B, input_dim)
#         h = self.input_lin(x)            # (B, hidden_dim)
#         for block in self.blocks:
#             h = h + block(h)             # 殘差連接
#             h = F.gelu(h)                # ← 改成這行
#         return self.output_lin(h)        # (B, action_dim)

import torch
import torch.nn as nn
import torch.nn.functional as F

class MSEActionHead(nn.Module):
    def __init__(self,
                 input_dim=4096,
                 hidden_dim=8192,       # 加大 hidden size
                 action_dim=7,
                 n_blocks=16,            # 增加 block 數量
                 resid_scale=0.2):       # 殘差縮放係數
        super().__init__()
        self.input_lin = nn.utils.spectral_norm(
            nn.Linear(input_dim, hidden_dim)
        )
        self.blocks = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, hidden_dim),
            )
            for _ in range(n_blocks)
        ])
        self.resid_scale = resid_scale
        self.final_norm = nn.LayerNorm(hidden_dim)
        self.output_lin = nn.Linear(hidden_dim, action_dim)

        # 初始化
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))
                if m.bias is not None:
                    fan_in, _ = nn.init._calculate_fan_in_and_fan_out(m.weight)
                    bound = 1 / math.sqrt(fan_in)
                    nn.init.uniform_(m.bias, -bound, bound)

    def predict_action(self, x):
        h = self.input_lin(x)
        for block in self.blocks:
            delta = block(h)
            h = h + self.resid_scale * delta    # 放小比例防爆
            h = F.gelu(h)
        h = self.final_norm(h)
        return self.output_lin(h)



# class MSEActionHead(nn.Module):
#     def __init__(self, input_dim=4096, hidden_dim=4096, action_dim=7):
#         super().__init__()
#         self.input_lin  = nn.Linear(input_dim, hidden_dim)
#         self.blocks     = nn.ModuleList([
#             nn.Sequential(
#                 nn.Linear(hidden_dim, hidden_dim),
#                 nn.GELU(),
#                 nn.Linear(hidden_dim, hidden_dim),
#             )
#             for _ in range(4)   # 深度減到 4 塊
#         ])
#         self.output_lin = nn.Linear(hidden_dim, action_dim)
#         # 直接把 x→action 的捷徑
#         self.skip       = nn.Linear(input_dim, action_dim, bias=False)
#     def predict_action(self, x):
#         h = F.gelu(self.input_lin(x))
#         for block in self.blocks:
#             h = h + block(h)
#             h = F.gelu(h)
#         return self.output_lin(h) + self.skip(x)





class NoisePredictionModel(nn.Module):
    """
    Diffusion noise prediction model that takes an observation embedding (which fuses the
    noisy action, diffusion timestep, and image-language observation embeddings) and
    outputs a noise prediction.
    """

    def __init__(
        self,
        transformer_hidden_dim,  # Transformer hidden embedding size
        hidden_dim,  # MLP hidden size
        action_dim=7,  # action dimensionality
    ):
        super().__init__()
        self.mlp_resnet = MLPResNet(
            num_blocks=2,
            input_dim=transformer_hidden_dim,
            hidden_dim=hidden_dim,
            output_dim=action_dim,
        )

    def forward(
        self,
        obs,
    ):
        # obs: observation embeddings to condition the generation on
        # - shape: (batch_size, chunk_len, rearranged_hidden_dim=action_dim*hidden_dim)
        #
        # output: predicted noise
        # - shape: (batch_size, action_dim)
        output = self.mlp_resnet(obs)
        return output

