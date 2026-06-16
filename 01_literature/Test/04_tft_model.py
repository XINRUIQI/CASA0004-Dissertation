"""
Simplified Temporal Fusion Transformer (TFT) for oil price prediction.
Key components: Variable Selection Network + GRU encoder + multi-head attention + gated output.
Designed for small sample size (~1043 weeks).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from config import (
    SEED, OUT_DIR, LOOKBACK, LAYER_FEATURES,
    TFT_HIDDEN, TFT_ATTENTION_HEADS, TFT_DROPOUT,
    TFT_EPOCHS, TFT_PATIENCE, TFT_LR, TFT_BATCH,
)
from data_loader import prepare_sequences
from evaluation import direction_metrics, regression_metrics, results_row, save_results

torch.manual_seed(SEED)
np.random.seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── Model Components ──────────────────────────────────────────────

class GatedResidualNetwork(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int, dropout: float):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.elu = nn.ELU()
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.gate_fc = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(output_size)
        self.skip = nn.Linear(input_size, output_size) if input_size != output_size else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x if self.skip is None else self.skip(x)
        h = self.elu(self.fc1(x))
        h = self.dropout(h)
        out = self.fc2(h)
        gate = self.sigmoid(self.gate_fc(h))
        gated = gate * out
        return self.layer_norm(gated + residual)


class VariableSelectionNetwork(nn.Module):
    def __init__(self, n_features: int, hidden_size: int, dropout: float):
        super().__init__()
        self.n_features = n_features
        self.hidden_size = hidden_size
        self.flattened_grn = GatedResidualNetwork(n_features, hidden_size, n_features, dropout)
        self.softmax = nn.Softmax(dim=-1)
        self.per_var_grns = nn.ModuleList([
            GatedResidualNetwork(1, hidden_size, hidden_size, dropout)
            for _ in range(n_features)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, n_features)
        batch, seq_len, _ = x.shape

        weights = self.softmax(self.flattened_grn(x))  # (batch, seq_len, n_features)

        var_outputs = []
        for i in range(self.n_features):
            var_i = x[:, :, i:i+1]  # (batch, seq_len, 1)
            var_outputs.append(self.per_var_grns[i](var_i))  # (batch, seq_len, hidden)

        var_outputs = torch.stack(var_outputs, dim=-1)  # (batch, seq_len, hidden, n_features)
        weights_expanded = weights.unsqueeze(2)  # (batch, seq_len, 1, n_features)
        selected = (var_outputs * weights_expanded).sum(dim=-1)  # (batch, seq_len, hidden)
        return selected

    def get_variable_weights(self, x: torch.Tensor) -> np.ndarray:
        """Extract average variable importance weights across samples and timesteps."""
        with torch.no_grad():
            weights = self.softmax(self.flattened_grn(x))  # (batch, seq_len, n_features)
            return weights.mean(dim=(0, 1)).cpu().numpy()


class SimplifiedTFT(nn.Module):
    def __init__(self, n_features: int, hidden_size: int, n_heads: int,
                 dropout: float, output_size: int):
        super().__init__()
        self.vsn = VariableSelectionNetwork(n_features, hidden_size, dropout)
        self.encoder = nn.GRU(hidden_size, hidden_size, batch_first=True)
        self.attention = nn.MultiheadAttention(
            hidden_size, n_heads, dropout=dropout, batch_first=True
        )
        self.grn_output = GatedResidualNetwork(hidden_size, hidden_size, hidden_size, dropout)
        self.fc_out = nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor):
        selected = self.vsn(x)
        encoded, _ = self.encoder(selected)
        attn_out, attn_weights = self.attention(encoded, encoded, encoded)
        last_step = self.grn_output(attn_out[:, -1, :])
        return self.fc_out(last_step), attn_weights


# ── Training ──────────────────────────────────────────────────────

def train_model(model, train_loader, val_loader, criterion, target_key,
                epochs=TFT_EPOCHS, patience=TFT_PATIENCE, lr=TFT_LR):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )

    best_val_loss = float("inf")
    best_state = None
    wait = 0

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            out, _ = model(X_batch)
            if target_key == "direction":
                loss = criterion(out, y_batch.long())
            else:
                loss = criterion(out.squeeze(-1), y_batch)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item() * X_batch.size(0)
        train_loss /= len(train_loader.dataset)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                out, _ = model(X_batch)
                if target_key == "direction":
                    loss = criterion(out, y_batch.long())
                else:
                    loss = criterion(out.squeeze(-1), y_batch)
                val_loss += loss.item() * X_batch.size(0)
        val_loss /= len(val_loader.dataset)
        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break

    model.load_state_dict(best_state)
    return model


def predict(model, X: np.ndarray, target_key: str):
    model.eval()
    X_t = torch.tensor(X, dtype=torch.float32).to(device)
    all_preds = []
    all_attn = []

    with torch.no_grad():
        for i in range(0, len(X_t), TFT_BATCH):
            batch = X_t[i:i + TFT_BATCH]
            out, attn_w = model(batch)
            if target_key == "direction":
                preds = out.argmax(dim=-1).cpu().numpy()
            else:
                preds = out.squeeze(-1).cpu().numpy()
            all_preds.append(preds)
            all_attn.append(attn_w.cpu().numpy())

    preds = np.concatenate(all_preds)
    attn_weights = np.concatenate(all_attn, axis=0)

    if target_key == "direction":
        preds = preds - 1  # map {0,1,2} back to {-1,0,1}

    return preds, attn_weights


# ── Visualization ─────────────────────────────────────────────────

def plot_attention_heatmap(attn_weights: np.ndarray, target: str, save_path):
    avg_attn = attn_weights.mean(axis=0)  # average over samples → (seq, seq)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(avg_attn, cmap="YlOrRd", ax=ax, square=True,
                xticklabels=range(1, avg_attn.shape[1] + 1),
                yticklabels=range(1, avg_attn.shape[0] + 1))
    ax.set_xlabel("Key position (week)")
    ax.set_ylabel("Query position (week)")
    ax.set_title(f"TFT Attention Weights – M4 – {target}")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_variable_importance(model, X_sample: np.ndarray, feat_names: list[str],
                             target: str, save_path):
    X_t = torch.tensor(X_sample, dtype=torch.float32).to(device)
    importance = model.vsn.get_variable_weights(X_t)

    sorted_idx = np.argsort(importance)[::-1]
    top_n = min(20, len(feat_names))
    idx = sorted_idx[:top_n]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(range(top_n), importance[idx][::-1], color="steelblue")
    ax.set_yticks(range(top_n))
    ax.set_yticklabels([feat_names[i] for i in idx][::-1])
    ax.set_xlabel("VSN Weight")
    ax.set_title(f"TFT Variable Importance – M4 – {target}")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


# ── Main ──────────────────────────────────────────────────────────

def run_tft():
    layers = ["M1", "M2", "M3", "M4", "M5"]
    targets = ["direction", "volatility", "price"]
    all_rows = []

    for layer in layers:
        for target in targets:
            print(f"\n{'='*60}")
            print(f"TFT | Layer={layer} | Target={target}")
            print(f"{'='*60}")

            X_train, X_val, X_test, y_train, y_val, y_test, scaler, feat_names, test_idx = \
                prepare_sequences(layer, target, LOOKBACK)

            n_features = X_train.shape[2]

            if target == "direction":
                y_train_t = torch.tensor(y_train + 1, dtype=torch.long)  # {-1,0,1}→{0,1,2}
                y_val_t = torch.tensor(y_val + 1, dtype=torch.long)
                output_size = 3
                criterion = nn.CrossEntropyLoss()
            else:
                y_train_t = torch.tensor(y_train, dtype=torch.float32)
                y_val_t = torch.tensor(y_val, dtype=torch.float32)
                output_size = 1
                criterion = nn.MSELoss()

            X_train_t = torch.tensor(X_train, dtype=torch.float32)
            X_val_t = torch.tensor(X_val, dtype=torch.float32)

            train_ds = TensorDataset(X_train_t, y_train_t)
            val_ds = TensorDataset(X_val_t, y_val_t)
            train_loader = DataLoader(train_ds, batch_size=TFT_BATCH, shuffle=True)
            val_loader = DataLoader(val_ds, batch_size=TFT_BATCH, shuffle=False)

            model = SimplifiedTFT(
                n_features=n_features,
                hidden_size=TFT_HIDDEN,
                n_heads=TFT_ATTENTION_HEADS,
                dropout=TFT_DROPOUT,
                output_size=output_size,
            ).to(device)

            model = train_model(model, train_loader, val_loader, criterion, target)
            preds, attn_weights = predict(model, X_test, target)

            if target == "direction":
                metrics = direction_metrics(y_test, preds)
            else:
                metrics = regression_metrics(y_test, preds)

            row = results_row("TFT", layer, target, metrics)
            all_rows.append(row)
            print(f"  → {row}")

            if layer in ("M4", "M5"):
                attn_path = OUT_DIR / f"tft_attention_{layer}_{target}.png"
                plot_attention_heatmap(attn_weights, target, attn_path)
                print(f"  → Saved attention map: {attn_path}")

                vi_path = OUT_DIR / f"tft_variable_importance_{layer}_{target}.png"
                plot_variable_importance(model, X_test, feat_names, target, vi_path)
                print(f"  → Saved variable importance: {vi_path}")

    save_results(all_rows, OUT_DIR / "tft_results.csv")
    print("\n[done] TFT experiments complete.")


if __name__ == "__main__":
    run_tft()
