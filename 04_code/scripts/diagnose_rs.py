"""
Remote-sensing (Prithvi Channel-A) branch diagnostics — WHY is the RS modality
weak/harmful in the deep fusion? Answers, mostly WITHOUT retraining:

  Q4  coverage / cloud / temporal frequency per AOI, and whether the frozen
      embedding actually MOVES over time (dynamic signal) or just encodes a
      static "which place" signature (between-site vs within-site variance).
  Q2  meanpool vs cls embedding: agreement + which carries more temporal signal.
  Q3  is the embedding related to oil price / shipping / inventory / port /
      the hand-crafted Channel-B indices? (PCA of the monthly embedding vs each
      block; |corr|>2/sqrt(n) ~ 5% noise floor; where does the chain break?)
  Q5  (optional, --gate) train one gated m4rep model and read the fusion gate
      alpha: did the model learn to DOWN-WEIGHT the RS modality?

Run:
  python3 04_code/scripts/diagnose_rs.py            # Q4/Q2/Q3 (no training)
  python3 04_code/scripts/diagnose_rs.py --gate     # + Q5 (trains one model)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPTS_DIR = Path(__file__).resolve().parent
SRC_DIR = SCRIPTS_DIR.parent / "src"
sys.path.insert(0, str(SRC_DIR))

from backtest import data                                  # noqa: E402
from models.deep_dataset import (EMB_CLS, EMB_INDEX,       # noqa: E402
                                 EMB_MEANPOOL)


def _load_emb():
    mp = np.load(EMB_MEANPOOL).astype(np.float64)
    cls = np.load(EMB_CLS).astype(np.float64)
    idx = pd.read_csv(EMB_INDEX)
    idx["month"] = pd.to_datetime(idx["obs_month_start"])
    return mp, cls, idx


# --------------------------------------------------------------------------- #
# Q4a  coverage / cloud / frequency
# --------------------------------------------------------------------------- #
def coverage_report(idx: pd.DataFrame) -> None:
    print("=" * 96)
    print("Q4a. PER-AOI COVERAGE / CLOUD / FREQUENCY  (Prithvi monthly embeddings)")
    print("-" * 96)
    print(f"{'site':16s} {'type':9s} {'n_obs':>5s} {'span':>17s} "
          f"{'medGap_mo':>9s} {'maxGap_mo':>9s} {'cloud%':>7s} {'>40%cld':>7s} "
          f"{'nScenes':>7s}")
    for sid, g in idx.sort_values("month").groupby("site_id"):
        g = g.sort_values("month")
        months = g["month"]
        gaps = months.diff().dropna().dt.days / 30.44
        name = g["site_name"].iloc[0][:15]
        st = str(g["site_type"].iloc[0])[:8]
        cl = g["mean_cloud"].astype(float)
        print(f"{name:16s} {st:9s} {len(g):5d} "
              f"{months.min().date().strftime('%y-%m')}~{months.max().date().strftime('%y-%m')} "
              f"{(gaps.median() if len(gaps) else 0):9.1f} "
              f"{(gaps.max() if len(gaps) else 0):9.1f} "
              f"{cl.mean():7.1f} {(cl > 40).mean()*100:7.0f} "
              f"{g['n_scenes'].astype(float).mean():7.1f}")
    print("-" * 96)
    print("Embeddings are MONTHLY at best; the forecast target is WEEKLY -> a "
          "monthly signal is as-of held for ~4 weeks (frequency mismatch).")
    print("=" * 96 + "\n")


# --------------------------------------------------------------------------- #
# Q4b  does the embedding move over time? (dynamic vs static)
# --------------------------------------------------------------------------- #
def variance_report(mp: np.ndarray, idx: pd.DataFrame) -> None:
    print("=" * 96)
    print("Q4b. DYNAMIC vs STATIC  (between-AOI vs within-AOI-over-time variance)")
    print("-" * 96)
    site_ids = idx["site_id"].to_numpy()
    site_means, within = [], []
    print(f"{'site':16s} {'temporalStd':>11s}  (mean over 1024 dims of std-over-time)")
    for sid, g in idx.groupby("site_id"):
        rows = g["emb_row"].to_numpy()
        E = mp[rows]                                   # (n_months, 1024)
        site_means.append(E.mean(0))
        tstd = float(E.std(0).mean())
        within.append(E.var(0).mean())
        print(f"{g['site_name'].iloc[0][:15]:16s} {tstd:11.3f}")
    site_means = np.stack(site_means)                  # (n_sites, 1024)
    between_var = float(site_means.var(0).mean())
    within_var = float(np.mean(within))
    frac_dyn = within_var / (within_var + between_var)
    global_std = mp.std(0)
    near_dead = int((global_std < 0.05 * global_std.mean()).sum())
    print("-" * 96)
    print(f"between-AOI variance (which place) : {between_var:.3f}")
    print(f"within-AOI temporal variance (time): {within_var:.3f}")
    print(f"=> temporal fraction = within/(within+between) = {frac_dyn*100:.1f}%  "
          f"(low => embedding mostly encodes location, not change)")
    print(f"near-constant dims (std < 5% of mean std): {near_dead}/1024")
    print("=" * 96 + "\n")


# --------------------------------------------------------------------------- #
# Q2  meanpool vs cls
# --------------------------------------------------------------------------- #
def meanpool_vs_cls(mp: np.ndarray, cls: np.ndarray, idx: pd.DataFrame) -> None:
    print("=" * 96)
    print("Q2. MEANPOOL vs CLS EMBEDDING")
    print("-" * 96)
    a = mp / (np.linalg.norm(mp, axis=1, keepdims=True) + 1e-9)
    b = cls / (np.linalg.norm(cls, axis=1, keepdims=True) + 1e-9)
    cos = float(np.mean(np.sum(a * b, axis=1)))
    # temporal signal per kind (within-site temporal std, mean over sites)
    def _tstd(E):
        v = []
        for _, g in idx.groupby("site_id"):
            v.append(E[g["emb_row"].to_numpy()].std(0).mean())
        return float(np.mean(v))
    print(f"mean row-wise cosine(meanpool, cls) : {cos:.3f}  "
          f"(1=identical direction)")
    print(f"within-AOI temporal std  meanpool={_tstd(mp):.3f}  cls={_tstd(cls):.3f}  "
          f"(higher => more temporal signal)")
    print("Prior sweep: meanpool skill -0.05% vs cls -0.55% (both ~ random walk).")
    print("=" * 96 + "\n")


# --------------------------------------------------------------------------- #
# Q3  is the embedding related to oil / shipping / inventory / port / Chan-B?
# --------------------------------------------------------------------------- #
def _pca(X: np.ndarray, k: int) -> np.ndarray:
    Xc = (X - X.mean(0)) / (X.std(0) + 1e-9)
    U, S, _ = np.linalg.svd(Xc, full_matrices=False)
    return U[:, :k] * S[:k]


def relation_report(mp: np.ndarray, idx: pd.DataFrame, k: int = 5) -> None:
    from scipy import stats
    print("=" * 96)
    print(f"Q3. EMBEDDING vs OIL / SHIPPING / INVENTORY / PORT / CHANNEL-B "
          f"(top-{k} PCs of the monthly cross-AOI-mean embedding)")
    print("-" * 96)
    # monthly cross-AOI mean embedding
    mrows = {m: [] for m in idx["month"].unique()}
    for _, r in idx.iterrows():
        mrows[r["month"]].append(int(r["emb_row"]))
    months = sorted(mrows)
    Em = np.stack([mp[mrows[m]].mean(0) for m in months])   # (n_months, 1024)
    pcs = _pca(Em, k)
    emb_month = pd.DataFrame(pcs, index=pd.DatetimeIndex(months),
                             columns=[f"PC{i+1}" for i in range(k)])

    df = data.load_matrix()
    monthly = df.resample("MS").mean(numeric_only=True)
    monthly["brent_ret"] = np.log(monthly["brent_price"]).diff()

    def _block(patterns):
        cols = []
        for p in patterns:
            cols += [c for c in monthly.columns if p(c)]
        return list(dict.fromkeys(cols))

    blocks = {
        "oil(price/ret)": ["brent_price", "brent_ret"],
        "inventory": [c for c in ["crude_stocks_excl_spr", "cushing_stocks"]
                      if c in monthly.columns],
        "shipping": _block([lambda c: c.endswith("_total_hours"),
                            lambda c: c.endswith("_n_tanker")])[:6],
        "port": [c for c in ["pw_exp_hubs_export_vol", "pw_imp_hubs_import_vol"]
                 if c in monthly.columns],
        "channelB(NTL/NDBI anom)": _block(
            [lambda c: c.startswith("NTL_anom_"),
             lambda c: c.startswith("NDBI_anom_")]),
    }
    print(f"{'block':26s} {'n_cols':>6s} {'n_obs':>5s} {'noise floor':>11s} "
          f"{'max|corr|':>9s} {'#sig':>5s} {'CCA r1':>7s} {'CCA perm-p':>10s}")
    for name, cols in blocks.items():
        cols = [c for c in cols if c in monthly.columns]
        if not cols:
            print(f"{name:26s} {0:6d}  (no columns found)")
            continue
        M = emb_month.join(monthly[cols], how="inner").dropna()
        if len(M) < 20:
            print(f"{name:26s} {len(cols):6d} {len(M):5d}  (too few obs)")
            continue
        P = M[emb_month.columns].to_numpy()
        Y = M[cols].to_numpy()
        n = len(M)
        floor = 2 / np.sqrt(n)
        C = np.array([[abs(np.corrcoef(P[:, i], Y[:, j])[0, 1])
                       for j in range(Y.shape[1])] for i in range(P.shape[1])])
        maxc = float(np.nanmax(C))
        nsig = int(np.nansum(C > floor))
        r1, permp = _cca_perm(P, Y)
        print(f"{name:26s} {len(cols):6d} {n:5d} {floor:11.3f} "
              f"{maxc:9.3f} {nsig:5d} {r1:7.3f} {permp:10.3f}")
    print("-" * 96)
    print("max|corr|<noise floor & CCA perm-p>0.05 => no detectable linear "
          "relation (embedding ~ unrelated to that block).")
    print("Where the chain breaks: if not even related to Channel-B / shipping / "
          "inventory (real activity), the frozen embedding isn't capturing "
          "oil-relevant activity at all.")
    print("=" * 96 + "\n")


def _cca_perm(P: np.ndarray, Y: np.ndarray, B: int = 1000, seed: int = 0):
    """First canonical correlation + permutation p-value (row-shuffle Y)."""
    def _cc(A, Bm):
        A = (A - A.mean(0)) / (A.std(0) + 1e-9)
        Bm = (Bm - Bm.mean(0)) / (Bm.std(0) + 1e-9)
        qa, _ = np.linalg.qr(A)
        qb, _ = np.linalg.qr(Bm)
        s = np.linalg.svd(qa.T @ qb, compute_uv=False)
        return float(np.clip(s[0], 0, 1))
    r1 = _cc(P, Y)
    rng = np.random.default_rng(seed)
    null = np.array([_cc(P, Y[rng.permutation(len(Y))]) for _ in range(B)])
    return r1, float((null >= r1).mean())


# --------------------------------------------------------------------------- #
# Q5  gate alpha (optional; trains one model)
# --------------------------------------------------------------------------- #
def gate_report(epochs: int, lookback: int, seed: int) -> None:
    import torch
    from models.deep_dataset import (apply_scalers, build_deep_dataset,
                                     fit_scalers)
    from models.deep_rolling import CONFIGS, _to_tensors, _train_fold

    df = data.load_matrix(); dico = data.load_dict()
    ds = build_deep_dataset(df, dico, lookback=lookback)
    n = len(ds["idx"])
    mods, mname, ftype = CONFIGS["m4rep"]              # fin+rs+ship, gated
    sc = fit_scalers(ds, train_n=n)
    model, _, _ = _train_fold(ds, sc, n, mods, seed, epochs, 1e-3, 1e-4, 32, 52,
                              "cpu", {}, ftype)
    X = _to_tensors(apply_scalers(ds, sc, slice(0, n)), "cpu")
    model.eval()
    with torch.no_grad():
        kw = dict(aoi=X["aoi"], choke=X["choke"], adj=X["adj"], fin=X["fin"],
                  rs=X["rs"], rs_mask=X["rs_mask"])
        _, info = model(**kw)
    gate = info["gate"].cpu().numpy()                  # (N, 3)
    order = info["gate_order"]                         # ["fin","rs","ship"]
    print("=" * 96)
    print(f"Q5. GATED-FUSION ALPHA (m4rep, trained on all {n} samples, "
          f"epochs={epochs})")
    print("-" * 96)
    print(f"{'modality':10s} {'mean_alpha':>10s} {'std':>7s} {'min':>7s} {'max':>7s}")
    for i, m in enumerate(order):
        a = gate[:, i]
        print(f"{m:10s} {a.mean():10.3f} {a.std():7.3f} {a.min():7.3f} {a.max():7.3f}")
    ri = order.index("rs")
    print("-" * 96)
    print(f"RS mean gate weight = {gate[:, ri].mean():.3f} of 1.0 "
          f"(equal share = {1/len(order):.3f}). "
          f"{'DOWN-weighted' if gate[:, ri].mean() < 1/len(order) else 'NOT down-weighted'}"
          " by the learned gate.")
    print("=" * 96 + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gate", action="store_true", help="also train + read gate")
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--lookback", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    mp, cls, idx = _load_emb()
    print(f"\nPrithvi embeddings: meanpool{mp.shape} cls{cls.shape}  "
          f"index rows={len(idx)}  sites={idx['site_id'].nunique()}\n")
    coverage_report(idx)
    variance_report(mp, idx)
    meanpool_vs_cls(mp, cls, idx)
    relation_report(mp, idx)
    if args.gate:
        gate_report(args.epochs, args.lookback, args.seed)


if __name__ == "__main__":
    main()
