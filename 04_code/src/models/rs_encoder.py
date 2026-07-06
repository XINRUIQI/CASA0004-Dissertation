"""
z_rs — remote-sensing encoder over FROZEN Prithvi-EO-2.0 image embeddings
(research plan sec. 5.1: frozen EO FM -> temporal attention over months +
site attention over the 11 AOIs -> 32-d). The Prithvi backbone is NOT fine-tuned
(embeddings are precomputed in s2_prithvi_emb_meanpool.npy); this module only
learns the light temporal/site attention + projection.

Input : rs (B, L, S, 1024) as-of aligned Prithvi embeddings, rs_mask (B, L, S)
        in {0,1} (1 = an embedding was already available that week/site).
Output: z_rs (B, d_out) + site_att (B, S) (RQ3 which AOI the RS branch weights).
"""

from __future__ import annotations

import torch
import torch.nn as nn


class RSPrithviEncoder(nn.Module):
    def __init__(self, emb_dim: int = 1024, n_sites: int = 11, d_model: int = 64,
                 d_out: int = 32, dropout: float = 0.1):
        super().__init__()
        self.n_sites = n_sites
        self.proj = nn.Linear(emb_dim, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)
        self.temporal_q = nn.Parameter(torch.randn(d_model) * 0.02)
        self.site_q = nn.Parameter(torch.randn(d_model) * 0.02)
        self.head = nn.Sequential(nn.Linear(d_model, d_out), nn.ReLU())

    def forward(self, rs: torch.Tensor, mask: torch.Tensor):
        # rs (B,L,S,emb); mask (B,L,S)
        rs = torch.nan_to_num(rs)
        h = self.drop(self.norm(self.proj(rs)))        # (B,L,S,d)
        h = h.permute(0, 2, 1, 3)                       # (B,S,L,d)
        m = mask.permute(0, 2, 1)                       # (B,S,L)

        # Temporal attention per site over the lookback (masked by availability).
        t_score = (h * self.temporal_q).sum(-1)        # (B,S,L)
        t_score = t_score.masked_fill(m == 0, float("-inf"))
        t_alpha = torch.nan_to_num(torch.softmax(t_score, dim=-1))
        site_emb = (t_alpha.unsqueeze(-1) * h).sum(2)  # (B,S,d)

        # Site attention over the 11 AOIs (masked by any-valid-in-window).
        site_valid = (m.sum(-1) > 0).float()           # (B,S)
        s_score = (site_emb * self.site_q).sum(-1)     # (B,S)
        s_score = s_score.masked_fill(site_valid == 0, float("-inf"))
        s_alpha = torch.nan_to_num(torch.softmax(s_score, dim=-1))
        z = (s_alpha.unsqueeze(-1) * site_emb).sum(1)  # (B,d)
        # site_emb (B,S,d_model) are per-AOI tokens for cross-modal attention.
        return self.head(z), s_alpha, site_emb
