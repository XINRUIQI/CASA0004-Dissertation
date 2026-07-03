"""
Deep early-fusion time-series baseline (LSTM/GRU) for M0..M4.

Companion to run_baseline.py (the Ridge/XGB flat tabular baseline). It reuses the
SAME data layer (backtest.data) and the SAME evaluation (backtest.metrics), so
its numbers are directly comparable to the tabular baselines on identical test
weeks, target and protocol.

Why this exists (RQ2): run_baseline gives the flat FEATURE-level fusion via
tree/linear models. This script adds the missing deep EARLY-FUSION reference:
every selected numeric column (M1..M4) is fed as a [lookback, features] sequence
into ONE shared recurrent net -- no modality-specific encoders. It is the
"stack-all-columns-into-one-time-series-model" baseline that the later
modality-aware representation fusion (contribution layer) must beat. Together
with M0/Ridge/XGB it completes the "flat vs modality-aware" comparison.

Protocol (locked, identical to run_baseline):
  window 2019-2026, lookback 4, expanding rolling-origin (min_train=104,
  retrain_every=13), single-task regression of r_{t+1} then reconstruct
  P_hat = P_t * exp(r_hat); metrics on the reconstructed price.

Regularisation (small weekly sample ~360, M0 is very strong): small hidden
size, dropout, weight decay, target standardisation, and early stopping on a
time-ordered inner-validation tail -- to avoid the negative-R2 blow-ups seen
with earlier unregularised deep runs.

Outputs (-> 05_outputs/baselines/<modality>/):
  baseline_deep_metrics[_<suffix>].csv       M0 + (M1) + {modality} LSTM (+DM/CW vs M1)
  baseline_deep_predictions[_<suffix>].csv   per test week P_hat / r_hat
  backtest_deep[_<suffix>].png               price track + RMSE + skill vs M0

Run:
  python3 04_code/scripts/run_deep_baseline.py --modality M1
  python3 04_code/scripts/run_deep_baseline.py --modality M4
  python3 04_code/scripts/run_deep_baseline.py --modality M4 --m2-features anom
  # fast smoke test:
  python3 04_code/scripts/run_deep_baseline.py --modality M1 --epochs 40 --retrain-every 26

Requires: torch, pandas, numpy, scikit-learn (via backtest.data), scipy, matplotlib.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

SCRIPTS_DIR = Path(__file__).resolve().parent
SRC_DIR = SCRIPTS_DIR.parent / "src"
sys.path.insert(0, str(SRC_DIR))

from backtest import data, metrics                       # noqa: E402

BASE_OUT = data.ROOT / "05_outputs/baselines"
NESTED_BASE = "M1"
MODEL_TAG = "LSTM"                                        # column/name suffix


# ----------------------------------------------------------------------------
# Model
# ----------------------------------------------------------------------------
class RecurrentRegressor(nn.Module):
    """Shared early-fusion recurrent net: [B, L, F] -> scalar r_hat.

    All selected columns are one flat feature vector per week; there is no
    per-modality encoder -- that is the point of the early-fusion baseline.
    """

    def __init__(self, n_feat: int, hidden: int = 48, layers: int = 1,
                 dropout: float = 0.2, arch: str = "lstm"):
        super().__init__()
        rnn_cls = nn.LSTM if arch == "lstm" else nn.GRU
        self.rnn = rnn_cls(n_feat, hidden, num_layers=layers, batch_first=True,
                           dropout=dropout if layers > 1 else 0.0)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.rnn(x)                             # [B, L, H]
        return self.head(out[:, -1, :]).squeeze(-1)      # [B] (last step)


# ----------------------------------------------------------------------------
# Reshape flat lagged matrix -> sequences
# ----------------------------------------------------------------------------
def to_sequences(ds: dict, lookback: int) -> tuple[np.ndarray, int]:
    """[n, n_raw*L] (lag-major from make_lagged) -> [n, L, n_raw], oldest->newest.

    make_lagged lays columns out as [all feats @lag0, all feats @lag1, ...] with
    lag0 = t (newest). reshape(n, L, n_raw) recovers [sample, lag, feature]; we
    then flip the lag axis so the RNN reads oldest -> newest (t-L+1 .. t).
    """
    n_raw = ds["n_raw"]
    X = ds["X"]
    n = X.shape[0]
    seq = X.reshape(n, lookback, n_raw)                  # [n, lag, feat]; lag0=t
    seq = seq[:, ::-1, :].copy()                         # oldest -> newest
    return seq.astype(np.float32), n_raw


# ----------------------------------------------------------------------------
# Train one model on a training fold (past-only) with inner-val early stopping
# ----------------------------------------------------------------------------
def train_one(Xtr_seq: np.ndarray, rtr: np.ndarray, val_weeks: int, seed: int,
              hidden: int, layers: int, dropout: float, arch: str,
              max_epochs: int, patience: int, lr: float, wd: float,
              device: str):
    torch.manual_seed(seed)
    np.random.seed(seed)

    F = Xtr_seq.shape[2]
    # Feature standardisation, fit on the TRAIN FOLD only (over [n*L, F]).
    flat = Xtr_seq.reshape(-1, F)
    mu = flat.mean(axis=0)
    sd = flat.std(axis=0)
    sd[sd < 1e-8] = 1.0
    # Target standardisation (r ~ 1e-2 scale) -> stabler optimisation.
    rmu = float(rtr.mean())
    rsd = float(rtr.std())
    rsd = rsd if rsd > 1e-8 else 1.0

    def scale(a: np.ndarray) -> np.ndarray:
        return (a - mu) / sd

    # Time-ordered inner validation tail for early stopping.
    if len(Xtr_seq) >= val_weeks + 30:
        tr_sl = slice(0, len(Xtr_seq) - val_weeks)
        va_sl = slice(len(Xtr_seq) - val_weeks, None)
    else:
        tr_sl, va_sl = slice(0, len(Xtr_seq)), None

    Xtr_t = torch.tensor(scale(Xtr_seq[tr_sl]), device=device)
    ytr_t = torch.tensor(((rtr[tr_sl] - rmu) / rsd).astype(np.float32), device=device)
    if va_sl is not None:
        Xva_t = torch.tensor(scale(Xtr_seq[va_sl]), device=device)
        yva = rtr[va_sl]                                 # raw r space for selection

    model = RecurrentRegressor(F, hidden, layers, dropout, arch).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    lossf = nn.MSELoss()

    best_state, best_val, bad = None, np.inf, 0
    for _ in range(max_epochs):
        model.train()
        opt.zero_grad()
        loss = lossf(model(Xtr_t), ytr_t)
        loss.backward()
        opt.step()

        if va_sl is None:
            continue
        model.eval()
        with torch.no_grad():
            pv = model(Xva_t).cpu().numpy() * rsd + rmu
        ve = float(np.sqrt(np.mean((pv - yva) ** 2)))
        if ve < best_val - 1e-7:
            best_val, bad = ve, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return model, (mu, sd, rmu, rsd)


def predict_one(model, scaler, Xte_seq: np.ndarray, device: str) -> np.ndarray:
    mu, sd, rmu, rsd = scaler
    Xt = torch.tensor(((Xte_seq - mu) / sd).astype(np.float32), device=device)
    with torch.no_grad():
        return model(Xt).cpu().numpy() * rsd + rmu


# ----------------------------------------------------------------------------
# Rolling-origin (walk-forward), same protocol as backtest.rolling
# ----------------------------------------------------------------------------
def rolling_deep(ds: dict, label: str, min_train: int, retrain_every: int,
                 seed: int, val_weeks: int, hp: dict, device: str) -> pd.DataFrame:
    idx = ds["idx"]
    Pt, Pn, rn, rnow = ds["P_t"], ds["P_next"], ds["r_next"], ds["r_now"]
    seq, n_raw = to_sequences(ds, hp["lookback"])
    n = len(idx)
    name = f"{label}_{MODEL_TAG}"

    model, scaler, rows, n_fits = None, None, [], 0
    for i in range(n):
        if i < min_train:                                # expanding warm-up
            continue
        if (model is None) or ((i - min_train) % retrain_every == 0):
            model, scaler = train_one(
                seq[:i], rn[:i], val_weeks, seed, device=device,
                hidden=hp["hidden"], layers=hp["layers"], dropout=hp["dropout"],
                arch=hp["arch"], max_epochs=hp["max_epochs"],
                patience=hp["patience"], lr=hp["lr"], wd=hp["wd"])
            n_fits += 1
        rhat = float(predict_one(model, scaler, seq[i:i + 1], device)[0])
        rows.append({"date": idx[i], "P_t": Pt[i], "P_next_actual": Pn[i],
                     "r_actual": rn[i], "r_now": rnow[i],
                     "r_hat_M0": 0.0, "P_hat_M0": Pt[i],
                     f"r_hat_{name}": rhat, f"P_hat_{name}": Pt[i] * np.exp(rhat)})

    res = pd.DataFrame(rows).set_index("date")
    res.attrs.update(label=label, n_features=len(ds["feat_names"]),
                     n_raw=n_raw, n_fits=n_fits)
    return res


def run_config(df, dico, modality, m2_features, lookback, min_train,
               retrain_every, seed, val_weeks, hp, device) -> pd.DataFrame:
    cols = data.select_features(dico, modality, m2_features)
    ds = data.build_dataset(df, cols, lookback, feature_mode="all")
    hp = {**hp, "lookback": lookback}
    return rolling_deep(ds, modality, min_train, retrain_every, seed,
                        val_weeks, hp, device)


# ----------------------------------------------------------------------------
# Summary (reuses metrics.evaluate + incremental_tests -> comparable to tabular)
# ----------------------------------------------------------------------------
def build_summary(res_main, modality, res_base):
    main_model = f"{modality}_{MODEL_TAG}"
    m_main = metrics.evaluate(res_main, [main_model])
    if res_base is None:
        summ = m_main
        summ["DM_p_vs_M1"] = np.nan
        summ["CW_stat_vs_M1"] = np.nan
        summ["CW_p_vs_M1"] = np.nan
        return summ

    base_model = f"{NESTED_BASE}_{MODEL_TAG}"
    m_base = metrics.evaluate(res_base, [base_model]).drop(
        index=["Naive_DirPersist"], errors="ignore")
    m_main_only = m_main.drop(index=["M0_RW"], errors="ignore")
    summ = pd.concat([m_base, m_main_only])
    order = ["M0_RW", base_model, main_model, "Naive_DirPersist"]
    summ = summ.reindex([o for o in order if o in summ.index])

    summ["DM_p_vs_M1"] = np.nan
    summ["CW_stat_vs_M1"] = np.nan
    summ["CW_p_vs_M1"] = np.nan
    inc = metrics.incremental_tests(res_main, res_base, modality, NESTED_BASE, MODEL_TAG)
    summ.loc[main_model, "DM_p_vs_M1"] = inc["DM_p_vs_base"]
    summ.loc[main_model, "CW_stat_vs_M1"] = inc["CW_stat_vs_base"]
    summ.loc[main_model, "CW_p_vs_M1"] = inc["CW_p_vs_base"]
    return summ


def merge_predictions(res_main, modality, res_base):
    common = ["P_t", "P_next_actual", "r_actual", "r_now", "r_hat_M0", "P_hat_M0"]
    pred = res_main[common].copy()
    if res_base is not None:
        extra = [c for c in res_base.columns if c not in common]
        pred = pred.join(res_base[extra])
    extra = [c for c in res_main.columns if c not in common and c not in pred.columns]
    return pred.join(res_main[extra])


# ----------------------------------------------------------------------------
# Plot
# ----------------------------------------------------------------------------
def make_plot(pred, summ, modality, path, suffix):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5),
                                   gridspec_kw={"width_ratios": [2.4, 1]})
    ax1.plot(pred.index, pred["P_next_actual"], color="black", lw=1.6, label="Actual P(t+1)")
    ax1.plot(pred.index, pred["P_hat_M0"], color="grey", lw=1.0, ls="--", label="M0 RW")
    for name, c in ((f"{NESTED_BASE}_{MODEL_TAG}", "tab:green"),
                    (f"{modality}_{MODEL_TAG}", "tab:purple")):
        col = f"P_hat_{name}"
        if col in pred:
            ax1.plot(pred.index, pred[col], lw=1.0, alpha=0.85, color=c, label=name)
    ax1.set_title(f"Deep early-fusion ({MODEL_TAG}) — next-week Brent{suffix}")
    ax1.set_ylabel("USD / barrel")
    ax1.legend(fontsize=8)
    ax1.grid(alpha=0.3)

    bars = summ["RMSE"].dropna()
    bcolors = ["grey" if i == "M0_RW" else "tab:purple" for i in bars.index]
    ax2.bar(range(len(bars)), bars.values, color=bcolors)
    ax2.axhline(bars.get("M0_RW", np.nan), color="grey", ls="--", lw=0.8)
    ax2.set_xticks(range(len(bars)))
    ax2.set_xticklabels(bars.index, rotation=35, ha="right", fontsize=7)
    ax2.set_title("RMSE on price (lower=better)")
    ax2.grid(alpha=0.3, axis="y")

    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description="Deep early-fusion baseline M0..M4 (LSTM/GRU).")
    ap.add_argument("--modality", default="M1", choices=list(data.MODALITY_SETS))
    ap.add_argument("--m2-features", default="anom", choices=list(data.M2_FEATURE_MODES))
    ap.add_argument("--lookback", type=int, default=4)
    ap.add_argument("--min-train", type=int, default=104)
    ap.add_argument("--retrain-every", type=int, default=13)
    ap.add_argument("--val-weeks", type=int, default=52)
    ap.add_argument("--arch", choices=["lstm", "gru"], default="lstm")
    ap.add_argument("--hidden", type=int, default=48)
    ap.add_argument("--layers", type=int, default=1)
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--patience", type=int, default=20)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--no-plot", action="store_true")
    ap.add_argument("--matrix", default=None, help="override merged matrix path")
    ap.add_argument("--dict", default=None, help="override feature dictionary path")
    args = ap.parse_args()

    has_m2 = "M2" in data.MODALITY_SETS[args.modality]
    tag = args.tag if args.tag else (args.m2_features if has_m2 else "")

    dict_path = args.dict
    if args.matrix and dict_path is None:
        mp = Path(args.matrix)
        dict_stem = mp.stem.replace("weekly_feature_matrix", "weekly_feature_dictionary")
        dict_path = (mp.parent / (dict_stem + mp.suffix)) if mp.is_absolute() else dict_stem + mp.suffix

    OUT_DIR = BASE_OUT / args.modality.lower()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = data.load_matrix(path=args.matrix)
    dico = data.load_dict(path=dict_path)

    hp = dict(hidden=args.hidden, layers=args.layers, dropout=args.dropout,
              arch=args.arch, max_epochs=args.epochs, patience=args.patience,
              lr=args.lr, wd=args.wd)

    print(f"Merged matrix: {df.shape}  {df.index.min().date()} ~ {df.index.max().date()}")
    print(f"Deep config: modality={args.modality} | arch={args.arch} "
          f"hidden={args.hidden} layers={args.layers} dropout={args.dropout} "
          f"| lookback={args.lookback} | retrain_every={args.retrain_every} "
          f"| m2={args.m2_features if has_m2 else '-'} | device={args.device}\n")

    t0 = time.time()
    res_main = run_config(df, dico, args.modality, args.m2_features, args.lookback,
                          args.min_train, args.retrain_every, args.seed,
                          args.val_weeks, hp, args.device)
    print(f"  {args.modality:3s}: {res_main.attrs['n_raw']:3d} raw x{args.lookback} "
          f"| fits={res_main.attrs['n_fits']} | test={len(res_main)} "
          f"({res_main.index.min().date()}~{res_main.index.max().date()})")

    res_base = None
    if args.modality != NESTED_BASE:
        res_base = run_config(df, dico, NESTED_BASE, args.m2_features, args.lookback,
                              args.min_train, args.retrain_every, args.seed,
                              args.val_weeks, hp, args.device)
        common = res_main.index.intersection(res_base.index)
        res_main, res_base = res_main.loc[common], res_base.loc[common]
        print(f"  {NESTED_BASE:3s}: base for CW/DM | common test weeks={len(common)}")

    summ = build_summary(res_main, args.modality, res_base)
    pred = merge_predictions(res_main, args.modality, res_base)

    suffix = f"_{tag}" if tag else ""
    met_path = OUT_DIR / f"baseline_deep_metrics{suffix}.csv"
    pred_path = OUT_DIR / f"baseline_deep_predictions{suffix}.csv"
    summ.to_csv(met_path)
    pred.to_csv(pred_path)

    print("\n" + "=" * 100)
    print(summ.to_string(float_format=lambda x: f"{x:8.4f}"))
    print("=" * 100)
    print("skill>0 beats M0.  DM_p_better_than_M0<0.05: sig. better than M0.")
    if res_base is not None:
        print("CW_p_vs_M1<0.05: added modality gives SIGNIFICANT nested increment over M1 (deep).")

    plot_path = None
    if not args.no_plot:
        plot_path = OUT_DIR / f"backtest_deep{suffix}.png"
        make_plot(pred, summ, args.modality, plot_path,
                  f"  [{args.modality}{suffix}, {args.arch}, L={args.lookback}]")

    print(f"\nElapsed {time.time()-t0:.0f}s")
    print(f"Saved: {met_path}\n       {pred_path}"
          + (f"\n       {plot_path}" if plot_path else ""))


if __name__ == "__main__":
    main()
