"""
Spatio-Temporal Graph Neural Network for oil price prediction.
Ablation across M1–M4 layers × 3 prediction targets.
Pure-PyTorch GCN (no torch_geometric dependency).
"""
from __future__ import annotations

import copy

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler

from config import (
    SEED, OUT_DIR, LOOKBACK, TARGETS,
    GNN_HIDDEN, GNN_LAYERS, GNN_DROPOUT,
    GNN_EPOCHS, GNN_PATIENCE, GNN_LR, GNN_BATCH,
    AOI_NODES, CHOKEPOINT_EDGES, M1_VARS, M2_RS_ADD, M3_SHIP_ADD, M5_GDELT_ADD,
    TRAIN_RATIO, VAL_RATIO,
)
from data_loader import load_weekly
from evaluation import direction_metrics, regression_metrics, results_row, save_results

LAYERS = ["M1", "M2", "M3", "M4", "M5"]
TARGET_KEYS = ["direction", "volatility", "price"]

N_NODES = len(AOI_NODES)
NODE_IDX = {n: i for i, n in enumerate(AOI_NODES)}

NTL_NODE_COL: dict[str, str] = {}
for _col in M2_RS_ADD:
    for _node in AOI_NODES:
        if _col.endswith(_node):
            NTL_NODE_COL[_node] = _col
            break

CHOKEPOINT_SHIP_COL: dict[str, str] = {}
for _col in M3_SHIP_ADD:
    if "capacity_tanker" in _col:
        for _cp in ("hormuz", "suez", "malacca"):
            if _cp in _col:
                CHOKEPOINT_SHIP_COL[_cp] = _col
                break


# ── Model ─────────────────────────────────────────────────────────

class GraphConvLayer(nn.Module):
    """H' = σ(D⁻¹ A H W)"""

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        support = torch.matmul(x, self.weight)
        output = torch.matmul(adj, support)
        return torch.relu(output)


class STGNN(nn.Module):
    """GCN (spatial) → GRU (temporal) → FC head."""

    def __init__(self, n_nodes: int, node_feat_size: int, hidden_size: int,
                 n_gcn_layers: int, dropout: float, output_size: int):
        super().__init__()
        self.gcn_layers = nn.ModuleList([
            GraphConvLayer(node_feat_size if i == 0 else hidden_size, hidden_size)
            for i in range(n_gcn_layers)
        ])
        self.gru = nn.GRU(n_nodes * hidden_size, hidden_size, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x_seq: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        batch, seq_len, n_nodes, _ = x_seq.shape
        temporal = []
        for t in range(seq_len):
            h = x_seq[:, t, :, :]
            for gcn in self.gcn_layers:
                h = self.dropout(gcn(h, adj))
            temporal.append(h.reshape(batch, -1))
        temporal = torch.stack(temporal, dim=1)
        _, h_n = self.gru(temporal)
        return self.fc(h_n[-1])


# ── Graph construction ────────────────────────────────────────────

def _build_adjacency(df: pd.DataFrame, layer: str) -> np.ndarray:
    """
    Build D⁻¹A adjacency. For M3/M4 edges through chokepoints with
    shipping data are weighted by mean capacity (scaled to [0.5, 1.5]).
    """
    adj = np.eye(N_NODES, dtype=np.float32)
    use_shipping = layer in ("M3", "M4", "M5")

    cp_weights: dict[str, float] = {}
    if use_shipping:
        for cp, col in CHOKEPOINT_SHIP_COL.items():
            if col in df.columns:
                cp_weights[cp] = df[col].mean()
        if cp_weights:
            vals = np.array(list(cp_weights.values()))
            vmin, vmax = vals.min(), vals.max()
            for cp in cp_weights:
                cp_weights[cp] = 0.5 + (cp_weights[cp] - vmin) / (vmax - vmin + 1e-8)

    for src, dst, cp in CHOKEPOINT_EDGES:
        i, j = NODE_IDX[src], NODE_IDX[dst]
        w = cp_weights.get(cp, 1.0) if use_shipping else 1.0
        adj[i, j] = w
        adj[j, i] = w

    deg = adj.sum(axis=1, keepdims=True)
    return adj / deg


# ── Data preparation ──────────────────────────────────────────────

def prepare_stgnn_data(
    df: pd.DataFrame, layer: str, target_key: str, lookback: int = LOOKBACK,
) -> tuple | None:
    """
    Returns (X_train, X_val, X_test, y_train, y_val, y_test, adj, test_idx)
    where X shape is (N, lookback, N_NODES, node_feat_dim), or None if
    data is insufficient.
    """
    target_col = TARGETS[target_key]
    use_ntl = layer in ("M2", "M4", "M5")
    use_gdelt = layer == "M5"

    fin_cols = [c for c in M1_VARS if c in df.columns]
    ntl_cols: list[str] = []
    ntl_node_indices: list[int] = []
    if use_ntl:
        for node in AOI_NODES:
            if node in NTL_NODE_COL and NTL_NODE_COL[node] in df.columns:
                ntl_cols.append(NTL_NODE_COL[node])
                ntl_node_indices.append(NODE_IDX[node])

    gdelt_cols: list[str] = []
    if use_gdelt:
        gdelt_cols = [c for c in M5_GDELT_ADD if c in df.columns]

    all_feat_cols = fin_cols + ntl_cols + gdelt_cols
    sub = df[all_feat_cols + [target_col]].dropna()

    if len(sub) < lookback + 30:
        return None

    scaler = StandardScaler()
    scaled = scaler.fit_transform(sub[all_feat_cols])
    targets_arr = sub[target_col].values
    dates = sub.index

    adj = _build_adjacency(sub, layer)

    n_fin = len(fin_cols)
    n_gdelt = len(gdelt_cols)
    node_feat_dim = n_fin + (1 if use_ntl else 0) + n_gdelt
    T = len(scaled)

    node_features = np.zeros((T, N_NODES, node_feat_dim), dtype=np.float32)
    node_features[:, :, :n_fin] = scaled[:, :n_fin][:, np.newaxis, :]
    if use_ntl:
        for k, node_i in enumerate(ntl_node_indices):
            node_features[:, node_i, n_fin] = scaled[:, n_fin + k]
    if use_gdelt:
        gdelt_offset = n_fin + len(ntl_cols)
        gdelt_data = scaled[:, gdelt_offset:gdelt_offset + n_gdelt]
        ntl_feat_offset = n_fin + (1 if use_ntl else 0)
        node_features[:, :, ntl_feat_offset:ntl_feat_offset + n_gdelt] = \
            gdelt_data[:, np.newaxis, :]

    X_all, y_all, idx_all = [], [], []
    for i in range(lookback, T):
        X_all.append(node_features[i - lookback:i])
        y_all.append(targets_arr[i])
        idx_all.append(dates[i])

    X_all = np.array(X_all, dtype=np.float32)
    y_all = np.array(y_all, dtype=np.float32)
    idx_all = pd.DatetimeIndex(idx_all)

    n = len(X_all)
    n_train = int(n * TRAIN_RATIO)
    n_val = int(n * VAL_RATIO)

    return (
        X_all[:n_train], X_all[n_train:n_train + n_val], X_all[n_train + n_val:],
        y_all[:n_train], y_all[n_train:n_train + n_val], y_all[n_train + n_val:],
        adj, idx_all[n_train + n_val:],
    )


# ── Training helpers ──────────────────────────────────────────────

def _make_loader(X: np.ndarray, y: np.ndarray, batch_size: int,
                 shuffle: bool, is_cls: bool) -> DataLoader:
    X_t = torch.tensor(X, dtype=torch.float32)
    if is_cls:
        y_t = torch.tensor(y.astype(int) + 1, dtype=torch.long)  # {-1,0,1} → {0,1,2}
    else:
        y_t = torch.tensor(y, dtype=torch.float32)
    return DataLoader(TensorDataset(X_t, y_t), batch_size=batch_size, shuffle=shuffle)


def train_stgnn(model: STGNN, adj_t: torch.Tensor,
                train_loader: DataLoader, val_loader: DataLoader,
                criterion, optimizer, epochs: int, patience: int,
                device: torch.device, is_cls: bool = False) -> STGNN:
    best_val_loss = float("inf")
    best_state = None
    wait = 0

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            out = model(xb, adj_t)
            if not is_cls:
                out = out.squeeze(-1)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(xb)
        train_loss /= len(train_loader.dataset)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                out = model(xb, adj_t)
                if not is_cls:
                    out = out.squeeze(-1)
                val_loss += criterion(out, yb).item() * len(xb)
        val_loss /= len(val_loader.dataset)

        if epoch % 25 == 0 or epoch == 1:
            print(f"    epoch {epoch:3d}  train_loss={train_loss:.5f}  val_loss={val_loss:.5f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                print(f"    early stop at epoch {epoch}  best_val_loss={best_val_loss:.5f}")
                break

    model.load_state_dict(best_state)
    return model


def predict_stgnn(model: STGNN, X: np.ndarray, adj_t: torch.Tensor,
                  device: torch.device, is_cls: bool) -> np.ndarray:
    model.eval()
    X_t = torch.tensor(X, dtype=torch.float32).to(device)
    with torch.no_grad():
        out = model(X_t, adj_t)
    if is_cls:
        return out.argmax(dim=1).cpu().numpy() - 1  # {0,1,2} → {-1,0,1}
    return out.squeeze(-1).cpu().numpy()


# ── Main driver ───────────────────────────────────────────────────

def run_stgnn():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    df = load_weekly()
    rows: list[dict] = []

    for layer in LAYERS:
        for target_key in TARGET_KEYS:
            print(f"\n{'=' * 60}")
            print(f"  ST-GNN  Layer={layer}  Target={target_key}")
            print(f"{'=' * 60}")

            result = prepare_stgnn_data(df, layer, target_key)
            if result is None:
                print("    [skip] insufficient data")
                continue

            X_tr, X_v, X_te, y_tr, y_v, y_te, adj, test_idx = result

            if len(X_tr) < 30:
                print(f"    [skip] insufficient training data ({len(X_tr)} samples)")
                continue

            is_cls = (target_key == "direction")
            output_size = 3 if is_cls else 1
            node_feat_dim = X_tr.shape[3]

            adj_t = torch.tensor(adj, dtype=torch.float32).to(device)
            train_loader = _make_loader(X_tr, y_tr, GNN_BATCH, shuffle=True, is_cls=is_cls)
            val_loader = _make_loader(X_v, y_v, GNN_BATCH, shuffle=False, is_cls=is_cls)

            model = STGNN(N_NODES, node_feat_dim, GNN_HIDDEN, GNN_LAYERS,
                          GNN_DROPOUT, output_size).to(device)
            criterion = nn.CrossEntropyLoss() if is_cls else nn.MSELoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=GNN_LR)

            model = train_stgnn(model, adj_t, train_loader, val_loader,
                                criterion, optimizer, GNN_EPOCHS, GNN_PATIENCE,
                                device, is_cls=is_cls)

            y_pred = predict_stgnn(model, X_te, adj_t, device, is_cls)
            metrics = (direction_metrics(y_te, y_pred) if is_cls
                       else regression_metrics(y_te, y_pred))
            print(f"    {metrics}")

            rows.append(results_row("ST-GNN", layer, target_key, metrics))

    save_results(rows, OUT_DIR / "stgnn_results.csv")
    print("\nDone — ST-GNN results saved.")


if __name__ == "__main__":
    run_stgnn()
