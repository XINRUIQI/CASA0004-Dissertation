"""
Deep dataset alignment for the representation-level fusion baseline.

Aligns the 17-node shipping graph tensor (m3_graph17_tensors.npz) and the M1
finance series onto the SAME weekly target/index that the flat baseline uses
(backtest.data.build_dataset), so the deep model is trained and tested on
identical weeks and the identical target r_{t+1}=log(P_{t+1}/P_t). This is what
makes the deep-vs-flat DM/Clark-West comparison fair.

For each usable test date d we build a lookback window of the past `lookback`
weeks:
  aoi   (lookback, 11, F_aoi)     AOI node features
  choke (lookback,  6, F_choke)   chokepoint node features
  adj   (lookback, 17, 17)        per-week adjacency (O-D + static)
  fin   (lookback, F_fin)         M1 finance features (levels; past-only filled)

Standardisation is NOT applied here — it must be fit per rolling-origin fold on
the training slice only (see deep_rolling.fit_scalers), to stay leak-free.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

SRC_DIR = Path(__file__).resolve().parents[1]      # 04_code/src
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from backtest import data                            # noqa: E402

GRAPH17_NPZ = (data.ROOT / "03_data/processed/M3/outputs/m3_graph17_tensors.npz")
EMB_DIR = data.ROOT / "03_data/processed/M2/outputs"
EMB_MEANPOOL = EMB_DIR / "s2_prithvi_emb_meanpool.npy"
EMB_CLS = EMB_DIR / "s2_prithvi_emb_cls.npy"
EMB_INDEX = EMB_DIR / "s2_prithvi_emb_index.csv"
RS_PUB_LAG_DAYS = 15   # conservative availability = month-end + 15 days


def _site_expanding_demean(emb: np.ndarray, idx: pd.DataFrame) -> np.ndarray:
    """Within-site, past-only (expanding, inclusive) demeaning of the monthly
    embeddings: a_j = e_j - mean(e_1..e_j) per site in chronological order.

    Removes the static 'which-site' scene signature (~80% of the frozen-embedding
    variance; see diagnose_rs.py) so the RS branch sees the temporal anomaly only.
    Leak-free: month j's anomaly uses only months with availability <= avail_j,
    all already published by the week that first uses month j. This mirrors the
    within-site standardized anomaly used for the hand-crafted Channel-B indices
    and the [P032] rationale (NTL captures cross-sectional, not within-site
    temporal, variation -> use within-site anomalies).
    """
    out = emb.copy()
    for _, g in idx.groupby("site_id"):
        rows = g.sort_values("obs_month_start")["emb_row"].to_numpy().astype(int)
        E = emb[rows]
        expmean = np.cumsum(E, axis=0) / np.arange(1, len(E) + 1)[:, None]
        out[rows] = (E - expmean).astype(np.float32)
    return out


def _build_rs_full(weeks: pd.DatetimeIndex, sites: list[str],
                   rs_kind: str = "meanpool"):
    """As-of align frozen Prithvi embeddings (month -> W-FRI, no look-ahead).

    Returns rs_full (T, n_sites, 1024) with NaN where no past embedding is
    available yet, and rs_valid (T, n_sites) in {0,1}. Availability of a month's
    embedding = month-end + RS_PUB_LAG_DAYS; each week takes the most recent
    already-available monthly embedding (merge_asof backward).

    rs_kind: "meanpool"/"cls" use the raw embedding; the "*_anom" variants
    ("meanpool_anom"/"cls_anom") apply within-site past-only expanding demeaning
    first (see _site_expanding_demean) to strip the static site signature.
    """
    base_kind = "cls" if rs_kind.startswith("cls") else "meanpool"
    demean = rs_kind.endswith("_anom")
    emb = np.load(EMB_MEANPOOL if base_kind == "meanpool" else EMB_CLS).astype(np.float32)
    idx = pd.read_csv(EMB_INDEX)
    idx["avail"] = (pd.to_datetime(idx["obs_month_start"])
                    + pd.offsets.MonthEnd(0) + pd.Timedelta(days=RS_PUB_LAG_DAYS))
    if demean:
        emb = _site_expanding_demean(emb, idx)
    D = emb.shape[1]
    T = len(weeks)
    rs_full = np.full((T, len(sites), D), np.nan, np.float32)
    rs_valid = np.zeros((T, len(sites)), np.float32)
    wk_df = pd.DataFrame({"week": pd.DatetimeIndex(weeks)}).sort_values("week")
    for si, s in enumerate(sites):
        sub = (idx[idx["site_id"] == s].sort_values("avail")[["avail", "emb_row"]]
               .reset_index(drop=True))
        if sub.empty:
            continue
        m = pd.merge_asof(wk_df, sub, left_on="week", right_on="avail",
                          direction="backward")
        rows = m["emb_row"].to_numpy()
        for ti in range(T):
            er = rows[ti]
            if not np.isnan(er):
                rs_full[ti, si, :] = emb[int(er)]
                rs_valid[ti, si] = 1.0
    return rs_full, rs_valid


def build_deep_dataset(df: pd.DataFrame, dico: pd.DataFrame,
                       npz_path: "Path | str" = GRAPH17_NPZ,
                       lookback: int = 4, with_rs: bool = True,
                       rs_kind: str = "meanpool",  # or cls / meanpool_anom / cls_anom
                       window_start: str = data.WINDOW_START,
                       window_end: str = data.WINDOW_END) -> dict:
    """Assemble aligned deep-model arrays over the flat baseline's test weeks.

    Returns a dict with idx, r_next (target), P_t, P_next, r_now, and the raw
    (unscaled) window tensors aoi/choke/adj/fin (+ rs/rs_mask if with_rs) plus
    feature-name lists.
    """
    # 1) Target / price / index from the flat builder (lookback=1 keeps the
    #    earliest usable weeks; the deep lookback is applied separately below).
    m1_cols = data.select_features(dico, "M1")
    base = data.build_dataset(df, m1_cols, lookback=1,
                              window_start=window_start, window_end=window_end)
    idx = base["idx"]
    tgt = {"r_next": base["r_next"], "P_t": base["P_t"],
           "P_next": base["P_next"], "r_now": base["r_now"]}

    # 2) Graph tensor bundle.
    g = np.load(npz_path, allow_pickle=True)
    gweeks = pd.to_datetime([str(w) for w in g["weeks"]])
    gpos = {w: i for i, w in enumerate(gweeks)}
    aoi_all = g["aoi_features"].astype(np.float32)     # (Tg,11,Fa)
    choke_all = g["choke_features"].astype(np.float32)  # (Tg,6,Fc)
    adj_all = g["adjacency"].astype(np.float32)         # (Tg,17,17)
    n_aoi = aoi_all.shape[1]
    sites = [str(x) for x in g["node_ids"][:n_aoi]]

    # 3) Finance series (M1 levels), past-only filled (ffill + leading 0).
    fin_df = data.fill_features(df[m1_cols])            # index = all W-FRI
    fin_pos = {d: i for i, d in enumerate(fin_df.index)}
    fin_vals = fin_df.to_numpy(dtype=np.float32)        # (Tf, F_fin)

    # 3b) Frozen Prithvi RS embeddings, as-of aligned to fin_df.index (no leak).
    rs_full = rs_valid = None
    if with_rs:
        rs_full, rs_valid = _build_rs_full(fin_df.index, sites, rs_kind)

    # 4) Build aligned lookback windows for every usable date that has a full
    #    graph window and a full finance window.
    keep, AW, CW, AdW, FW, RW, RMW = [], [], [], [], [], [], []
    for t, d in enumerate(idx):
        gi = gpos.get(d)
        fi = fin_pos.get(d)
        if gi is None or fi is None:
            continue
        if gi - lookback + 1 < 0 or fi - lookback + 1 < 0:
            continue
        gsl = slice(gi - lookback + 1, gi + 1)
        fsl = slice(fi - lookback + 1, fi + 1)
        AW.append(aoi_all[gsl]); CW.append(choke_all[gsl]); AdW.append(adj_all[gsl])
        FW.append(fin_vals[fsl])
        if with_rs:
            RW.append(rs_full[fsl]); RMW.append(rs_valid[fsl])
        keep.append(t)

    keep = np.array(keep)
    out = {
        "idx": idx[keep],
        "r_next": tgt["r_next"][keep],
        "P_t": tgt["P_t"][keep],
        "P_next": tgt["P_next"][keep],
        "r_now": tgt["r_now"][keep],
        "aoi": np.stack(AW),          # (N,L,11,Fa)
        "choke": np.stack(CW),        # (N,L,6,Fc)
        "adj": np.stack(AdW),         # (N,L,17,17)
        "fin": np.stack(FW),          # (N,L,F_fin)
        "aoi_feature_names": [str(x) for x in g["aoi_feature_names"]],
        "choke_feature_names": [str(x) for x in g["choke_feature_names"]],
        "fin_feature_names": list(m1_cols),
        "node_ids": [str(x) for x in g["node_ids"]],
        "sites": sites,
        "lookback": lookback,
        "with_rs": with_rs,
    }
    if with_rs:
        out["rs"] = np.stack(RW)      # (N,L,11,1024)
        out["rs_mask"] = np.stack(RMW)  # (N,L,11)
    return out


def _fit_scaler(x: np.ndarray, feat_axis: int) -> tuple[np.ndarray, np.ndarray]:
    """Mean/std over all axes except `feat_axis` (nan-aware); std floored."""
    axes = tuple(a for a in range(x.ndim) if a != feat_axis)
    m = np.nanmean(x, axis=axes, keepdims=True)
    s = np.nanstd(x, axis=axes, keepdims=True)
    s = np.where(s < 1e-6, 1.0, s)
    return m.astype(np.float32), s.astype(np.float32)


def fit_scalers(ds: dict, train_n: int) -> dict:
    """Fit per-feature scalers on the FIRST `train_n` samples only (past-only).

    aoi/choke/fin are z-scored on their last (feature) axis; adj is left as-is
    (only its >0 pattern is used by the encoder as an attention mask).
    """
    sc = {}
    sc["aoi"] = _fit_scaler(ds["aoi"][:train_n], feat_axis=3)
    sc["choke"] = _fit_scaler(ds["choke"][:train_n], feat_axis=3)
    sc["fin"] = _fit_scaler(ds["fin"][:train_n], feat_axis=2)
    if ds.get("with_rs"):
        sc["rs"] = _fit_scaler(ds["rs"][:train_n], feat_axis=3)
    return sc


def apply_scalers(ds: dict, sc: dict, sl: slice) -> dict:
    """Return scaled + nan-filled tensors for the sample slice `sl`."""
    def _ap(key):
        m, s = sc[key]
        return np.nan_to_num((ds[key][sl] - m) / s).astype(np.float32)
    result = {
        "aoi": _ap("aoi"), "choke": _ap("choke"), "fin": _ap("fin"),
        "adj": np.nan_to_num(ds["adj"][sl]).astype(np.float32),
    }
    if "rs" in sc:
        result["rs"] = _ap("rs")
        result["rs_mask"] = np.nan_to_num(ds["rs_mask"][sl]).astype(np.float32)
    return result


if __name__ == "__main__":
    import backtest.data as _d
    df = _d.load_matrix()
    dico = _d.load_dict()
    ds = build_deep_dataset(df, dico, lookback=4)
    print("aligned deep dataset:")
    for k in ["aoi", "choke", "adj", "fin"]:
        print(f"  {k:6s}: {ds[k].shape}")
    print(f"  N samples: {len(ds['idx'])}  "
          f"weeks {ds['idx'].min().date()} ~ {ds['idx'].max().date()}")
    print(f"  fin feats ({len(ds['fin_feature_names'])}): "
          f"{ds['fin_feature_names'][:5]}...")
    sc = fit_scalers(ds, train_n=104)
    scaled = apply_scalers(ds, sc, slice(0, 5))
    print(f"  scaled aoi[:5] finite={np.isfinite(scaled['aoi']).all()}  "
          f"mean~{scaled['aoi'].mean():.3f}")
