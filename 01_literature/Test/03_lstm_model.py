"""
LSTM model for oil price prediction — ablation framework (M1–M4 × 3 targets).
"""
from __future__ import annotations

import copy

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib
matplotlib.use("Agg")

from config import (
    SEED, OUT_DIR, LOOKBACK,
    LSTM_HIDDEN, LSTM_LAYERS, LSTM_DROPOUT,
    LSTM_EPOCHS, LSTM_PATIENCE, LSTM_LR, LSTM_BATCH,
)
from data_loader import load_weekly, prepare_sequences
from evaluation import direction_metrics, regression_metrics, results_row, save_results


class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout, output_size):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        _, (h_n, _) = self.lstm(x)
        return self.fc(h_n[-1])


def _make_loader(X: np.ndarray, y: np.ndarray, batch_size: int,
                 shuffle: bool, is_cls: bool) -> DataLoader:
    X_t = torch.tensor(X, dtype=torch.float32)
    if is_cls:
        y_t = torch.tensor(y.astype(int) + 1, dtype=torch.long)  # {-1,0,1} → {0,1,2}
    else:
        y_t = torch.tensor(y, dtype=torch.float32)
    return DataLoader(TensorDataset(X_t, y_t), batch_size=batch_size, shuffle=shuffle)


def train_lstm(model, train_loader, val_loader, criterion, optimizer,
               epochs, patience, device, is_classification=False):
    best_val_loss = float("inf")
    best_state = None
    wait = 0

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            out = model(xb)
            if not is_classification:
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
                out = model(xb)
                if not is_classification:
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


def predict(model, X: np.ndarray, device, is_cls: bool) -> np.ndarray:
    model.eval()
    X_t = torch.tensor(X, dtype=torch.float32).to(device)
    with torch.no_grad():
        out = model(X_t)
    if is_cls:
        preds = out.argmax(dim=1).cpu().numpy() - 1  # {0,1,2} → {-1,0,1}
    else:
        preds = out.squeeze(-1).cpu().numpy()
    return preds


def run_lstm():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    df = load_weekly()
    rows = []

    for layer in ["M1", "M2", "M3", "M4", "M5"]:
        for target_key in ["direction", "volatility", "price"]:
            print(f"  LSTM | {layer} | {target_key}")
            X_tr, X_v, X_te, y_tr, y_v, y_te, scaler, feat_names, test_idx = \
                prepare_sequences(layer, target_key, LOOKBACK, df=df)

            if len(X_tr) < 30:
                print(f"    [skip] insufficient data ({len(X_tr)} samples)")
                continue

            is_cls = (target_key == "direction")
            output_size = 3 if is_cls else 1
            input_size = X_tr.shape[2]

            train_loader = _make_loader(X_tr, y_tr, LSTM_BATCH, shuffle=True, is_cls=is_cls)
            val_loader = _make_loader(X_v, y_v, LSTM_BATCH, shuffle=False, is_cls=is_cls)

            model = LSTMModel(input_size, LSTM_HIDDEN, LSTM_LAYERS,
                              LSTM_DROPOUT, output_size).to(device)
            criterion = nn.CrossEntropyLoss() if is_cls else nn.MSELoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=LSTM_LR)

            model = train_lstm(model, train_loader, val_loader, criterion, optimizer,
                               LSTM_EPOCHS, LSTM_PATIENCE, device, is_classification=is_cls)

            y_pred = predict(model, X_te, device, is_cls)
            metrics = direction_metrics(y_te, y_pred) if is_cls else regression_metrics(y_te, y_pred)
            print(f"    {metrics}")

            rows.append(results_row("LSTM", layer, target_key, metrics))

    save_results(rows, OUT_DIR / "lstm_results.csv")


if __name__ == "__main__":
    run_lstm()
