"""
z_ship — heterogeneous dynamic-graph + temporal encoder for the shipping
modality (innovation-layer branch of the Modality-Aware Spatio-Temporal Fusion
Network; research plan sec. 5.1: "1-2 layer GAT + small TCN -> 32-d").

Input  : the 17-node graph bundle from build_m3_graph17.py
           aoi_feat   (B, L, 11, F_aoi)   AOI node features over a lookback L
           choke_feat (B, L,  6, F_choke) chokepoint node features
           adj        (B, L, 17, 17)      per-week adjacency (O-D + static)
Output : z_ship (B, 32)  +  site_attention (B, 17)  (RQ3 interpretability)

Design
------
* Heterogeneous input: AOI (F_aoi) and chokepoint (F_choke) live in different
  feature spaces -> node-type-specific linear projection to a shared d_model,
  plus a learned node-type embedding. This is the standard "type-specific input
  encoder + shared message passing" recipe for small heterogeneous graphs.
* Spatial: 1-2 dense multi-head graph-attention layers over the 17 nodes. The
  adjacency is symmetrised and self-looped for message passing; we use a dense
  (17x17) masked attention rather than torch_geometric sparse ops because the
  graph is tiny and the adjacency is dynamic per (batch, week) — dense is
  simpler and keeps the temporal batch clean. (Swap in tgnn.GATConv if desired.)
* Temporal: a causal 1-D TCN over the lookback L, per node.
* Pooling: additive attention over the 17 nodes -> graph embedding; the weights
  are returned as a site/lane importance signal.

The module is self-contained (pure torch). Running it as a script executes a
smoke test on the real m3_graph17_tensors.npz.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------------------------------------------------------------------
# Layers
# ----------------------------------------------------------------------------
class DenseGATLayer(nn.Module):
    """Multi-head graph attention on a small dense graph with a boolean mask.

    Optionally biases the attention logits by a per-edge weight (log O-D flow):
    heavier shipping lanes get a higher attention prior instead of the edge
    strength being discarded by the boolean adjacency mask (P1-4). The gain
    `edge_scale` is learned, so the model can down-weight the prior if unhelpful.
    """

    def __init__(self, d_in: int, d_out: int, heads: int = 4,
                 dropout: float = 0.1, leaky: float = 0.2):
        super().__init__()
        assert d_out % heads == 0, "d_out must be divisible by heads"
        self.heads = heads
        self.dh = d_out // heads
        self.W = nn.Linear(d_in, d_out, bias=False)
        self.a_src = nn.Parameter(torch.empty(heads, self.dh))
        self.a_dst = nn.Parameter(torch.empty(heads, self.dh))
        self.edge_scale = nn.Parameter(torch.tensor(1.0))   # learned edge-flow gain
        self.leaky = nn.LeakyReLU(leaky)
        self.drop = nn.Dropout(dropout)
        nn.init.xavier_uniform_(self.W.weight)
        nn.init.xavier_uniform_(self.a_src)
        nn.init.xavier_uniform_(self.a_dst)

    def forward(self, x: torch.Tensor, mask: torch.Tensor,
                edge_w: "torch.Tensor | None" = None):
        # x (M, N, d_in); mask (M, N, N) bool, mask[i, j] = i attends to j.
        # edge_w (M, N, N) optional continuous edge weight (broadcast over heads).
        M, N, _ = x.shape
        h = self.W(x).view(M, N, self.heads, self.dh)
        src = (h * self.a_src).sum(-1)               # (M, N, H)
        dst = (h * self.a_dst).sum(-1)               # (M, N, H)
        e = self.leaky(src.unsqueeze(2) + dst.unsqueeze(1))   # (M, N, N, H)
        if edge_w is not None:
            e = e + (self.edge_scale * edge_w).unsqueeze(-1)  # O-D flow prior
        e = e.masked_fill(~mask.unsqueeze(-1), float("-inf"))
        alpha = torch.softmax(e, dim=2)              # over neighbours j
        alpha = torch.nan_to_num(alpha)              # guard fully-masked rows
        alpha = self.drop(alpha)
        out = torch.einsum("mijh,mjhd->mihd", alpha, h).reshape(M, N, -1)
        return out, alpha


class TemporalTCN(nn.Module):
    """Causal 1-D convolution stack with *adaptive* dilation (P1-3 + sweep evidence).

    Layer i uses dilation 2**i ONLY when the lookback is longer than the plain
    (dilation-1) receptive field 1 + layers*(kernel-1); otherwise it falls back
    to dense dilation-1 conv. Dilation helps long windows (lookback 8/12: lifts
    the receptive field 5->7 and stops long-lookback sweeps looking artificially
    worse) but slightly hurts a short lookback=4 window (skill +0.14% -> -0.07%),
    where dense sampling is better. Pass `lookback` so the choice is made
    automatically per window; lookback=None keeps dilation on. Returns last step.
    """

    def __init__(self, d: int, layers: int = 2, kernel: int = 3, dropout: float = 0.1,
                 lookback: "int | None" = None):
        super().__init__()
        self.kernel = kernel
        plain_rf = 1 + layers * (kernel - 1)          # receptive field w/o dilation
        if lookback is not None and lookback <= plain_rf:
            self.dilations = [1] * layers             # short window -> dense conv
        else:
            self.dilations = [2 ** i for i in range(layers)]
        self.convs = nn.ModuleList(
            [nn.Conv1d(d, d, kernel, dilation=dl) for dl in self.dilations])
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x (M, d, L) -> causal dilated conv (left pad) -> last timestep (M, d)
        for conv, dl in zip(self.convs, self.dilations):
            res = x
            pad = (self.kernel - 1) * dl
            x = F.relu(conv(F.pad(x, (pad, 0))))
            x = self.drop(x)
            x = x + res
        return x[:, :, -1]


class ShippingGraphEncoder(nn.Module):
    """17-node heterogeneous graph + temporal encoder -> z_ship (d_out)."""

    def __init__(self, f_aoi: int, f_choke: int, n_aoi: int = 11, n_choke: int = 6,
                 d_model: int = 64, d_out: int = 32, gat_layers: int = 2,
                 heads: int = 4, tcn_layers: int = 2, dropout: float = 0.1,
                 lookback: "int | None" = None):
        super().__init__()
        self.n_aoi, self.n_choke = n_aoi, n_choke
        self.N = n_aoi + n_choke
        self.aoi_proj = nn.Linear(f_aoi, d_model)
        self.choke_proj = nn.Linear(f_choke, d_model)
        self.type_emb = nn.Embedding(2, d_model)     # 0 = AOI, 1 = chokepoint
        self.in_norm = nn.LayerNorm(d_model)
        self.gats = nn.ModuleList(
            [DenseGATLayer(d_model, d_model, heads, dropout) for _ in range(gat_layers)])
        self.gat_norm = nn.LayerNorm(d_model)
        self.tcn = TemporalTCN(d_model, tcn_layers, dropout=dropout, lookback=lookback)
        self.pool_score = nn.Linear(d_model, 1)
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(d_model, d_out))
        self.register_buffer("node_types",
                             torch.tensor([0] * n_aoi + [1] * n_choke))

    def forward(self, aoi_feat: torch.Tensor, choke_feat: torch.Tensor,
                adj: torch.Tensor):
        # aoi_feat (B,L,Na,Fa); choke_feat (B,L,Nc,Fc); adj (B,L,N,N)
        B, L, Na, _ = aoi_feat.shape
        N = self.N
        aoi_feat = torch.nan_to_num(aoi_feat)
        choke_feat = torch.nan_to_num(choke_feat)

        h = torch.cat([self.aoi_proj(aoi_feat), self.choke_proj(choke_feat)], dim=2)
        h = h + self.type_emb(self.node_types)[None, None, :, :]
        h = self.in_norm(h)                          # (B,L,N,d)

        # Symmetrise + self-loop -> boolean attention mask.
        adj_sym = adj + adj.transpose(-1, -2)
        eye = torch.eye(N, device=adj.device, dtype=adj.dtype)
        mask = (adj_sym + eye[None, None]) > 0       # (B,L,N,N)
        # Continuous edge weight (log O-D flow) as an attention prior (P1-4).
        edge_w = torch.log1p(adj_sym.clamp(min=0))   # (B,L,N,N)

        # Spatial GAT over each (batch, week) slice.
        hM = h.reshape(B * L, N, -1)
        mM = mask.reshape(B * L, N, N)
        eM = edge_w.reshape(B * L, N, N)
        last_alpha = None
        for gat in self.gats:
            out, last_alpha = gat(hM, mM, eM)
            hM = self.gat_norm(F.elu(out) + hM)      # residual + norm
        h = hM.reshape(B, L, N, -1)

        # Temporal TCN per node: (B,N,L,d) -> (B*N, d, L) -> (B,N,d)
        d = h.shape[-1]
        ht = h.permute(0, 2, 3, 1).reshape(B * N, d, L)
        ht = self.tcn(ht).reshape(B, N, d)

        # Additive attention pool over nodes.
        score = self.pool_score(ht).squeeze(-1)      # (B,N)
        site_att = torch.softmax(score, dim=1)
        z = (site_att.unsqueeze(-1) * ht).sum(1)     # (B,d)
        z = self.head(z)                             # (B,d_out)
        # ht (B,N,d_model) are the per-node tokens for cross-modal attention.
        return z, site_att, ht


# ----------------------------------------------------------------------------
# Data helper + smoke test
# ----------------------------------------------------------------------------
def _zscore_time(x: np.ndarray) -> np.ndarray:
    """Per-(node,feature) z-score over time, NaN-aware. NOTE: global here for
    the smoke test only — real training MUST use a past-only expanding scaler."""
    m = np.nanmean(x, axis=0, keepdims=True)
    s = np.nanstd(x, axis=0, keepdims=True)
    s = np.where(s < 1e-6, 1.0, s)
    return np.nan_to_num((x - m) / s)


def load_graph17_windows(npz_path: Path, lookback: int = 4, stride: int = 1,
                         standardize: bool = True):
    """Load the 17-node bundle and cut rolling lookback windows.
    Returns (aoi (W,L,11,Fa), choke (W,L,6,Fc), adj (W,L,17,17), end_weeks)."""
    d = np.load(npz_path, allow_pickle=True)
    aoi = d["aoi_features"].astype("float32")
    choke = d["choke_features"].astype("float32")
    adj = d["adjacency"].astype("float32")
    weeks = [str(w) for w in d["weeks"]]
    if standardize:
        aoi, choke = _zscore_time(aoi), _zscore_time(choke)
    T = aoi.shape[0]
    A, C, Ad, ew = [], [], [], []
    for t in range(lookback - 1, T, stride):
        sl = slice(t - lookback + 1, t + 1)
        A.append(aoi[sl]); C.append(choke[sl]); Ad.append(adj[sl]); ew.append(weeks[t])
    return (torch.from_numpy(np.stack(A)), torch.from_numpy(np.stack(C)),
            torch.from_numpy(np.stack(Ad)), ew)


def _smoke() -> None:
    root = Path(__file__).resolve().parents[3]
    npz = root / "data" / "processed" / "M3" / "outputs" / "m3_graph17_tensors.npz"
    A, C, Ad, ew = load_graph17_windows(npz, lookback=4)
    print(f"windows: aoi{tuple(A.shape)} choke{tuple(C.shape)} adj{tuple(Ad.shape)}")
    print(f"end weeks: {ew[0]} .. {ew[-1]} ({len(ew)} windows)")

    torch.manual_seed(0)
    enc = ShippingGraphEncoder(f_aoi=A.shape[-1], f_choke=C.shape[-1],
                               d_model=64, d_out=32)
    n_params = sum(p.numel() for p in enc.parameters())
    print(f"encoder params: {n_params:,}")

    enc.eval()
    with torch.no_grad():
        B = min(32, A.shape[0])
        z, site_att, _tok = enc(A[:B], C[:B], Ad[:B])
    print(f"z_ship: {tuple(z.shape)}  finite={torch.isfinite(z).all().item()}  "
          f"mean={z.mean():.3f} std={z.std():.3f}")
    print(f"site_att: {tuple(site_att.shape)}  row-sum={site_att[0].sum():.3f}")

    node_ids = [str(x) for x in np.load(npz, allow_pickle=True)["node_ids"]]
    top = torch.topk(site_att.mean(0), 5)
    print("top-5 attended nodes (mean over batch):")
    for w, i in zip(top.values.tolist(), top.indices.tolist()):
        print(f"    {node_ids[i]:8s} {w:.3f}")

    # Gradient sanity: one backward step on a dummy regression target.
    enc.train()
    opt = torch.optim.Adam(enc.parameters(), lr=1e-3)
    z, _, _ = enc(A[:B], C[:B], Ad[:B])
    loss = (z.sum(dim=1) ** 2).mean()               # dummy scalar objective
    opt.zero_grad(); loss.backward(); opt.step()
    gnorm = torch.sqrt(sum((p.grad ** 2).sum() for p in enc.parameters()
                           if p.grad is not None))
    print(f"backward OK  dummy_loss={loss.item():.4f}  grad_norm={gnorm.item():.4f}")
    print("SMOKE OK")


if __name__ == "__main__":
    _smoke()
