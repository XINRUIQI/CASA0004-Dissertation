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


class DeepForecastModel(nn.Module):
    """Modality encoders (subset of fin/rs/ship) + gated fusion + head -> r_hat.

    `modalities` is an ordered subset of ("fin", "rs", "ship"); the gate weights
    (when >1 modality) follow that order. Single-modality configs skip fusion.
    """

    def __init__(self, modalities: list[str], f_aoi: int, f_choke: int,
                 f_fin: int, rs_emb_dim: int = 1024, n_sites: int = 11,
                 d: int = 32, dropout: float = 0.1):
        super().__init__()
        for m in modalities:
            assert m in ("fin", "rs", "ship"), f"bad modality {m}"
        self.modalities = list(modalities)
        if "ship" in modalities:
            self.ship = ShippingGraphEncoder(f_aoi, f_choke, d_out=d, dropout=dropout)
        if "fin" in modalities:
            self.fin = FinanceTCNEncoder(f_fin, d_out=d, dropout=dropout)
        if "rs" in modalities:
            self.rs = RSPrithviEncoder(rs_emb_dim, n_sites, d_out=d, dropout=dropout)
        if len(modalities) > 1:
            self.fuse = GatedFusion(len(modalities), d)
        self.head = nn.Sequential(
            nn.Linear(d, d), nn.ReLU(), nn.Dropout(dropout), nn.Linear(d, 1))

    def forward(self, aoi=None, choke=None, adj=None, fin=None,
                rs=None, rs_mask=None):
        info: dict = {}
        zs = []
        for m in self.modalities:
            if m == "fin":
                zs.append(self.fin(fin))
            elif m == "ship":
                z, sa = self.ship(aoi, choke, adj)
                zs.append(z); info["ship_site_att"] = sa
            elif m == "rs":
                z, sa = self.rs(rs, rs_mask)
                zs.append(z); info["rs_site_att"] = sa
        if len(zs) == 1:
            z = zs[0]
        else:
            z, gate = self.fuse(zs)
            info["gate"] = gate                     # (B, n_mod), order = modalities
            info["gate_order"] = self.modalities
        r_hat = self.head(z).squeeze(-1)            # (B,)
        return r_hat, info
