"""
z_fin — finance/macro temporal encoder (research plan sec. 5.1: "2-3 layer small
TCN/GRU -> 32-d, the most robust branch"). Input is the M1 finance feature
series over a lookback; output is a 32-d embedding.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .shipping_encoder import TemporalTCN


class FinanceTCNEncoder(nn.Module):
    """(B, L, F_fin) -> z_fin (B, d_out) via a small causal TCN."""

    def __init__(self, f_in: int, d_model: int = 32, d_out: int = 32,
                 tcn_layers: int = 2, kernel: int = 3, dropout: float = 0.1):
        super().__init__()
        self.proj = nn.Linear(f_in, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.tcn = TemporalTCN(d_model, tcn_layers, kernel, dropout)
        self.head = nn.Sequential(nn.Linear(d_model, d_out), nn.ReLU())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x (B, L, F_fin)
        x = torch.nan_to_num(x)
        h = self.norm(self.proj(x))          # (B, L, d)
        h = h.transpose(1, 2)                # (B, d, L)
        h = self.tcn(h)                      # (B, d) last step
        return self.head(h)                  # (B, d_out)
