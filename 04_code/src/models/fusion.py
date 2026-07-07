"""
Modality-aware fusion + regression head for the representation-level baseline
(research plan sec. 5.2). Supports three configs on a common interface so the
rolling backtest can compare them under the identical protocol:

  mode="ship"    z_ship only            (shipping representation, RQ1)
  mode="fin"     z_fin only             (finance deep baseline)
  mode="fusion"  gated(z_fin, z_ship)   (finance + shipping, RQ2 representation arm)

Output is the predicted next-week log return r_hat (B,), plus an interpretability
dict (site attention over the 17 nodes; modality gate weights for fusion) for
RQ3. The single training loss is the regression loss on r_{t+1} (deep_rolling).
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .finance_encoder import FinanceTCNEncoder
from .rs_encoder import RSPrithviEncoder
from .shipping_encoder import ShippingGraphEncoder


class GatedFusion(nn.Module):
    """Softmax gate over modality embeddings -> convex combination.

    alpha = softmax(MLP([z_1..z_m]));  z = sum_i alpha_i * z_i.
    The gate weights are returned as the RQ3 modality-level explanation.
    """

    def __init__(self, n_mod: int, d: int, hidden: int = 32):
        super().__init__()
        self.n_mod = n_mod
        self.gate = nn.Sequential(
            nn.Linear(n_mod * d, hidden), nn.ReLU(), nn.Linear(hidden, n_mod))

    def forward(self, zs: list[torch.Tensor]):
        cat = torch.cat(zs, dim=-1)                 # (B, n_mod*d)
        alpha = torch.softmax(self.gate(cat), dim=-1)   # (B, n_mod)
        z = sum(alpha[:, i:i + 1] * zs[i] for i in range(self.n_mod))
        return z, alpha


class ConcatFusion(nn.Module):
    """Encoder-concatenation fusion: [z_1..z_m] -> MLP -> d.

    The middle rung of the RQ2 fusion ladder (Baltrusaitis et al., 2019 [P101];
    the concat baseline in Gated Multimodal Units [P096]). Modalities are still
    encoded separately (unlike the flat Ridge/XGB early fusion), but mixed by a
    fixed MLP with NO per-sample gating or cross-modal attention. Any gain of
    GatedFusion / CrossModalAttentionFusion over this arm is therefore
    attributable to the fusion mechanism, not merely to per-modality encoding.
    Returns (z, None) to keep the (embedding, weights) interface of GatedFusion.
    """

    def __init__(self, n_mod: int, d: int):
        super().__init__()
        self.n_mod = n_mod
        self.proj = nn.Sequential(nn.Linear(n_mod * d, d), nn.ReLU())

    def forward(self, zs: list[torch.Tensor]):
        z = self.proj(torch.cat(zs, dim=-1))        # (B, d)
        return z, None


class CrossModalAttentionFusion(nn.Module):
    """Finance z_fin as Query attending over RS/shipping node tokens (research
    plan sec 5.2 option 3): z_fused = LN(z_fin + gamma * CrossAttn(z_fin, H_kv)).
    The token-attention weights are returned for RQ3 (which node/lane the
    financial state attends to)."""

    def __init__(self, d: int, token_dim: int, n_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.q = nn.Linear(d, d)
        self.kv_proj = nn.Linear(token_dim, d)
        self.attn = nn.MultiheadAttention(d, n_heads, dropout=dropout, batch_first=True)
        self.gamma = nn.Parameter(torch.tensor(0.1))
        self.norm = nn.LayerNorm(d)

    def forward(self, z_fin: torch.Tensor, kv_tokens: torch.Tensor):
        q = self.q(z_fin).unsqueeze(1)               # (B,1,d)
        kv = self.kv_proj(kv_tokens)                 # (B,M,d)
        ctx, attw = self.attn(q, kv, kv, need_weights=True)
        z = self.norm(z_fin + self.gamma * ctx.squeeze(1))
        return z, attw.squeeze(1)                    # (B,d), (B,M)


class DeepForecastModel(nn.Module):
    """Modality encoders (subset of fin/rs/ship) + fusion (gated | xattn) +
    regression head -> r_hat, with optional sample-level modality dropout.

    `modalities` is an ordered subset of ("fin", "rs", "ship"). fusion_type:
      gated  softmax gate over modality embeddings (default, main model)
      xattn  finance-as-query cross-attention over RS/shipping node tokens
    modality_dropout (train only) randomly drops modalities for robustness /
    missing-modality training (ModDrop).
    """

    def __init__(self, modalities: list[str], f_aoi: int, f_choke: int,
                 f_fin: int, rs_emb_dim: int = 1024, n_sites: int = 11,
                 d: int = 32, dropout: float = 0.1, gat_layers: int = 2,
                 tcn_layers: int = 2, fusion_type: str = "gated",
                 modality_dropout: float = 0.0, token_dim: int = 64):
        super().__init__()
        for m in modalities:
            assert m in ("fin", "rs", "ship"), f"bad modality {m}"
        self.modalities = list(modalities)
        self.fusion_type = fusion_type
        self.modality_dropout = modality_dropout
        if "ship" in modalities:
            self.ship = ShippingGraphEncoder(f_aoi, f_choke, d_out=d, dropout=dropout,
                                             gat_layers=gat_layers, tcn_layers=tcn_layers)
        if "fin" in modalities:
            self.fin = FinanceTCNEncoder(f_fin, d_out=d, dropout=dropout,
                                         tcn_layers=tcn_layers)
        if "rs" in modalities:
            self.rs = RSPrithviEncoder(rs_emb_dim, n_sites, d_out=d, dropout=dropout)
        if len(modalities) > 1:
            if fusion_type == "gated":
                self.fuse = GatedFusion(len(modalities), d)
            elif fusion_type == "concat":
                self.fuse = ConcatFusion(len(modalities), d)
            elif fusion_type == "xattn":
                assert "fin" in modalities, "xattn needs finance as query"
                self.xattn = CrossModalAttentionFusion(d, token_dim, dropout=dropout)
            else:
                raise ValueError(f"bad fusion_type {fusion_type}")
        self.head = nn.Sequential(
            nn.Linear(d, d), nn.ReLU(), nn.Dropout(dropout), nn.Linear(d, 1))

    def _modality_dropout(self, zs: list[torch.Tensor]) -> list[torch.Tensor]:
        """Sample-level: each sample drops each modality w.p. p, keep >= 1."""
        B, M = zs[0].shape[0], len(zs)
        keep = (torch.rand(B, M, device=zs[0].device) > self.modality_dropout).float()
        none = keep.sum(1) == 0
        if none.any():
            keep[none, 0] = 1.0
        return [zs[i] * keep[:, i:i + 1] for i in range(M)]

    def forward(self, aoi=None, choke=None, adj=None, fin=None,
                rs=None, rs_mask=None):
        info: dict = {}
        zs, toks = [], {}
        for m in self.modalities:
            if m == "fin":
                zs.append(self.fin(fin))
            elif m == "ship":
                z, sa, tk = self.ship(aoi, choke, adj)
                zs.append(z); info["ship_site_att"] = sa; toks["ship"] = tk
            elif m == "rs":
                z, sa, tk = self.rs(rs, rs_mask)
                zs.append(z); info["rs_site_att"] = sa; toks["rs"] = tk

        drop = self.training and self.modality_dropout > 0 and len(zs) > 1
        if len(zs) == 1:
            z = zs[0]
        elif self.fusion_type == "gated":
            zin = self._modality_dropout(zs) if drop else zs
            z, gate = self.fuse(zin)
            info["gate"] = gate; info["gate_order"] = self.modalities
        elif self.fusion_type == "concat":
            zin = self._modality_dropout(zs) if drop else zs
            z, _ = self.fuse(zin)
        else:  # xattn — finance queries RS/shipping node tokens
            z_fin = zs[self.modalities.index("fin")]
            kv_list = [toks[m] for m in self.modalities if m in ("rs", "ship")]
            if drop and len(kv_list) > 1:
                kv_list = [t for t in kv_list
                           if torch.rand(1).item() > self.modality_dropout] or kv_list[:1]
            z, xaw = self.xattn(z_fin, torch.cat(kv_list, dim=1))
            info["xattn_weights"] = xaw
        r_hat = self.head(z).squeeze(-1)             # (B,)
        return r_hat, info
