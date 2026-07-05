"""
Rolling-origin (walk-forward) trainer for the deep representation-level models,
mirroring backtest.rolling exactly (expanding window, min_train warm-up, refit
every `retrain_every` weeks, strictly no look-ahead) so the output res DataFrame
is drop-in compatible with backtest.metrics.evaluate / incremental_tests.

Per fold: past-only feature scalers + per-fold target standardisation are fit on
the training slice only; a small deep model (deep_dataset windows -> r_hat) is
trained with Adam + inner-validation early stopping; the single test week is then
predicted and the price reconstructed P_hat = P_t * exp(r_hat).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from .deep_dataset import apply_scalers, fit_scalers
from .fusion import DeepForecastModel

# config name -> (ordered modalities, short model name used in result columns)
CONFIGS = {
    "ship": (["ship"], "GNN"),
    "fin": (["fin"], "TCN"),
    "rs": (["rs"], "RS"),
    "fusion": (["fin", "ship"], "Fusion"),
    "m4rep": (["fin", "rs", "ship"], "M4rep"),
}


def _to_tensors(scaled: dict, device: str) -> dict:
    return {k: torch.from_numpy(v).to(device) for k, v in scaled.items()}


def _make_model(ds: dict, modalities: list[str]) -> DeepForecastModel:
    return DeepForecastModel(
        modalities,
        f_aoi=ds["aoi"].shape[-1], f_choke=ds["choke"].shape[-1],
        f_fin=ds["fin"].shape[-1],
        rs_emb_dim=(ds["rs"].shape[-1] if ds.get("with_rs") else 1024),
        n_sites=ds["aoi"].shape[2])


def _forward(model, X: dict, idx_arr):
    kw = dict(aoi=X["aoi"][idx_arr], choke=X["choke"][idx_arr],
              adj=X["adj"][idx_arr], fin=X["fin"][idx_arr])
    if "rs" in X:
        kw["rs"] = X["rs"][idx_arr]
        kw["rs_mask"] = X["rs_mask"][idx_arr]
    return model(**kw)[0]


def _train_fold(ds: dict, sc: dict, i: int, modalities: list[str], seed: int,
                epochs: int, lr: float, weight_decay: float, batch: int,
                val_weeks: int, device: str) -> tuple[nn.Module, float, float]:
    """Train on samples [0, i); early-stop on the last `val_weeks` of the fold."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    X = _to_tensors(apply_scalers(ds, sc, slice(0, i)), device)
    r = ds["r_next"][:i].astype(np.float32)
    r_mean, r_std = float(np.mean(r)), float(np.std(r) + 1e-8)
    y = torch.from_numpy(((r - r_mean) / r_std).astype(np.float32)).to(device)

    n = i
    n_val = min(val_weeks, max(10, n // 5))
    tr_idx = np.arange(0, n - n_val)
    va_idx = np.arange(n - n_val, n)

    model = _make_model(ds, modalities).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    lossf = nn.MSELoss()

    best_state, best_val, patience, bad = None, np.inf, 12, 0
    for _ in range(epochs):
        model.train()
        perm = np.random.permutation(tr_idx)
        for b in range(0, len(perm), batch):
            bi = perm[b:b + batch]
            opt.zero_grad()
            loss = lossf(_forward(model, X, bi), y[bi])
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
        model.eval()
        with torch.no_grad():
            vloss = float(lossf(_forward(model, X, va_idx), y[va_idx]))
        if vloss < best_val - 1e-5:
            best_val, bad = vloss, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, r_mean, r_std


def rolling_origin_deep(ds: dict, label: str, config: str, min_train: int = 104,
                        retrain_every: int = 13, seed: int = 42, epochs: int = 80,
                        lr: float = 1e-3, weight_decay: float = 1e-4,
                        batch: int = 32, val_weeks: int = 52,
                        device: str = "cpu", verbose: bool = True) -> pd.DataFrame:
    """Walk-forward deep backtest for one config -> res DataFrame."""
    modalities, mname = CONFIGS[config]
    idx = ds["idx"]
    Pt, Pn = ds["P_t"], ds["P_next"]
    rn, rnow = ds["r_next"], ds["r_now"]
    n = len(idx)
    col = f"{label}_{mname}"

    model = None
    r_mean = r_std = 0.0
    rows, n_fits = [], 0
    for i in range(n):
        if i < min_train:
            continue
        if model is None or ((i - min_train) % retrain_every == 0):
            sc = fit_scalers(ds, train_n=i)
            model, r_mean, r_std = _train_fold(
                ds, sc, i, modalities, seed, epochs, lr, weight_decay, batch,
                val_weeks, device)
            n_fits += 1
            if verbose:
                print(f"    fit #{n_fits} @ week {idx[i].date()} (train={i})", flush=True)

        Xte = _to_tensors(apply_scalers(ds, sc, slice(i, i + 1)), device)
        model.eval()
        with torch.no_grad():
            out = _forward(model, Xte, slice(0, 1))
        rhat = float(out.item()) * r_std + r_mean
        rows.append({"date": idx[i], "P_t": Pt[i], "P_next_actual": Pn[i],
                     "r_actual": rn[i], "r_now": rnow[i],
                     "r_hat_M0": 0.0, "P_hat_M0": Pt[i],
                     f"r_hat_{col}": rhat, f"P_hat_{col}": Pt[i] * np.exp(rhat)})

    res = pd.DataFrame(rows).set_index("date")
    res.attrs.update(label=label, config=config, model=mname, n_fits=n_fits,
                     n_test=len(res), lookback=ds["lookback"])
    return res
