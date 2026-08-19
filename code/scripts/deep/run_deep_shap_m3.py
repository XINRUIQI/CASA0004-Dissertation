"""
Seed-42 input-level GradientSHAP + weekly node attention for M3_Deep_gated.

Walk-forward matches run_deep_interpret_m3.py / deep_rolling._train_fold:
  lookback=4, min_train=104, retrain_every=13, epochs=80, seed=42, CPU.

Explains destandardised r̂_{t+1} (not the network's z-scored output, not price).
SHAP features are the flattened raw inputs (fin + AOI + chokepoint), 1088 dims.
Adjacency is context only: for week t, background and test both use adj_t.

Background samples come from that checkpoint's training window [0, i_fit).
No future OOS week is used as background.

Outputs (-> results/baselines/Deep/M3_Deep/):
  deep_m3_shap_lock_periods.csv
  deep_m3_shap_lock_groups.csv
  deep_m3_shap_feature_map.csv
  deep_m3_shap_period_membership.csv
  deep_m3_shap_values.npy              (n_weeks, 1088) signed phi
  deep_m3_shap_weekly_long.csv.gz      date, lag, feature, node, group, shap
  deep_m3_shap_additivity.csv
  deep_m3_shap_qc_mean_abs.csv         full-sample mean_t I_{t,f}
  deep_m3_node_attention_weekly.csv    date, node_id, node_attention

Run:
  python3 code/scripts/deep/run_deep_shap_m3.py
  python3 code/scripts/deep/run_deep_shap_m3.py --max-weeks 2 --epochs 5
  python3 code/scripts/deep/run_deep_shap_m3.py --lock-only
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

SCRIPTS_DIR = Path(__file__).resolve().parent
SRC_DIR = SCRIPTS_DIR.parent.parent / "src"
sys.path.insert(0, str(SRC_DIR))

from backtest import data                                # noqa: E402
from model_naming import deep_out_dir                    # noqa: E402
from models.deep_dataset import (apply_scalers, build_deep_dataset,  # noqa: E402
                                 fit_scalers)
from models.deep_rolling import CONFIGS, _to_tensors, _train_fold  # noqa: E402

OUT_DIR = deep_out_dir(data.ROOT, "M3")
CONFIG_KEY = "m3_deep_gated"

# ---------------------------------------------------------------------------
# Locked periods (OOS prediction-origin dates)
# Full OOS, calendar years 2021–2025, four event windows. No early/late split.
# ---------------------------------------------------------------------------
EVENT_WINDOW_WEEKS = 8
EVENTS = [
    ("event_russia_ukraine", "2022-02-24", "Russia–Ukraine"),
    ("event_eu_ru_oil_ban", "2022-06-01", "EU RU oil ban announced"),
    ("event_opec_plus", "2023-04-02", "OPEC+ surprise cut"),
    ("event_red_sea", "2023-11-19", "Houthi Red Sea attacks"),
]
CALENDAR_YEARS = list(range(2021, 2026))

# ---------------------------------------------------------------------------
# Locked feature groups
# ---------------------------------------------------------------------------
OIL_PRICE_BASIS = {
    "brent_price", "wti_price", "brent_log_return", "wti_log_return",
    "brent_wti_spread", "brent_f1_spot_log_basis", "brent_roll_week",
}
EIA_FUNDAMENTALS = {
    "crude_stocks_excl_spr", "cushing_stocks", "crude_production",
    "crude_imports", "crude_exports", "refinery_crude_input",
    "refinery_utilisation", "gasoline_supplied", "distillate_supplied",
    "jet_fuel_supplied", "crude_stocks_change", "cushing_stocks_change",
}
GPR = {"gpr"}
FIN_MACRO = {
    "vix", "dollar_index", "treasury_10y", "fed_funds_rate", "sp500_log_return",
    "ovx", "gold_return", "global_econ_activity", "nonoil_industrial_commodity",
    "cadusd_log_return", "dgs10_change",
}

GROUP_META = [
    ("oil_price_basis", "finance",
     "Oil prices, returns, Brent–WTI spread, futures basis and roll"),
    ("financial_macro", "finance",
     "VIX/OVX, dollar, rates, equities, gold, activity, CAD, DGS10 change"),
    ("eia_fundamentals", "finance",
     "EIA stocks, production, trade, refining and product supplied"),
    ("gpr", "finance", "Geopolitical risk index"),
    ("GFW", "shipping", "All gfw_* features on AOI and chokepoint nodes"),
    ("PortWatch_chokepoint", "shipping",
     "pw_* features on the 6 chokepoint nodes"),
    ("PortWatch_directional", "shipping",
     "pw_* features on the 11 AOI nodes (port calls and directional flows)"),
    ("SAR", "shipping", "sar_* detections on AOI and chokepoint nodes"),
    ("shipping_other", "shipping", "Any remaining shipping feature"),
]


def finance_group(feat: str) -> str:
    if feat in OIL_PRICE_BASIS:
        return "oil_price_basis"
    if feat in EIA_FUNDAMENTALS:
        return "eia_fundamentals"
    if feat in GPR:
        return "gpr"
    if feat in FIN_MACRO:
        return "financial_macro"
    raise KeyError(f"unassigned finance feature: {feat}")


def shipping_group(feat: str, node_type: str) -> str:
    if feat.startswith("gfw_"):
        return "GFW"
    if feat.startswith("sar_"):
        return "SAR"
    if feat.startswith("pw_"):
        if node_type == "choke":
            return "PortWatch_chokepoint"
        if node_type == "aoi":
            return "PortWatch_directional"
    return "shipping_other"


def build_feature_map(ds: dict) -> pd.DataFrame:
    """C-order flatten of (fin, aoi, choke) matching pack_inputs()."""
    L = int(ds["lookback"])
    fin_names = list(ds["fin_feature_names"])
    aoi_names = list(ds["aoi_feature_names"])
    choke_names = list(ds["choke_feature_names"])
    node_ids = list(ds["node_ids"])
    n_aoi = ds["aoi"].shape[2]
    aoi_nodes = node_ids[:n_aoi]
    choke_nodes = node_ids[n_aoi:]

    missing = (set(fin_names)
               - OIL_PRICE_BASIS - EIA_FUNDAMENTALS - GPR - FIN_MACRO)
    extra = (OIL_PRICE_BASIS | EIA_FUNDAMENTALS | GPR | FIN_MACRO) - set(fin_names)
    if missing or extra:
        raise ValueError(f"finance group lock mismatch missing={missing} extra={extra}")

    rows = []
    idx = 0
    for slot in range(L):
        lag = L - slot  # slot 0 = oldest -> lag=L; slot L-1 = current -> lag=1
        for feat in fin_names:
            rows.append({
                "feature_index": idx, "channel": "fin", "slot": slot, "lag": lag,
                "node_id": "", "node_type": "", "feature": feat,
                "group": finance_group(feat),
            })
            idx += 1
    for slot in range(L):
        lag = L - slot
        for node in aoi_nodes:
            for feat in aoi_names:
                rows.append({
                    "feature_index": idx, "channel": "aoi", "slot": slot,
                    "lag": lag, "node_id": node, "node_type": "aoi",
                    "feature": feat, "group": shipping_group(feat, "aoi"),
                })
                idx += 1
    for slot in range(L):
        lag = L - slot
        for node in choke_nodes:
            for feat in choke_names:
                rows.append({
                    "feature_index": idx, "channel": "choke", "slot": slot,
                    "lag": lag, "node_id": node, "node_type": "choke",
                    "feature": feat, "group": shipping_group(feat, "choke"),
                })
                idx += 1
    fmap = pd.DataFrame(rows)
    expected = L * (len(fin_names) + n_aoi * len(aoi_names)
                    + len(choke_nodes) * len(choke_names))
    if len(fmap) != expected:
        raise ValueError(f"feature map size {len(fmap)} != expected {expected}")
    return fmap


def pack_inputs(fin: torch.Tensor, aoi: torch.Tensor,
                choke: torch.Tensor) -> torch.Tensor:
    b = fin.shape[0]
    return torch.cat([fin.reshape(b, -1), aoi.reshape(b, -1),
                      choke.reshape(b, -1)], dim=1)


def unpack_inputs(x: torch.Tensor, L: int, n_fin: int, n_aoi: int,
                  n_fa: int, n_choke: int, n_fc: int
                  ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    b = x.shape[0]
    i0 = 0
    n1 = L * n_fin
    fin = x[:, i0:i0 + n1].reshape(b, L, n_fin)
    i0 += n1
    n2 = L * n_aoi * n_fa
    aoi = x[:, i0:i0 + n2].reshape(b, L, n_aoi, n_fa)
    i0 += n2
    n3 = L * n_choke * n_fc
    choke = x[:, i0:i0 + n3].reshape(b, L, n_choke, n_fc)
    return fin, aoi, choke


class InputShapWrapper(nn.Module):
    """Flattened (fin, aoi, choke) -> destandardised r_hat. adj is a buffer."""

    def __init__(self, model: nn.Module, L: int, n_fin: int, n_aoi: int,
                 n_fa: int, n_choke: int, n_fc: int,
                 r_mean: float, r_std: float):
        super().__init__()
        self.model = model
        self.L = L
        self.n_fin = n_fin
        self.n_aoi = n_aoi
        self.n_fa = n_fa
        self.n_choke = n_choke
        self.n_fc = n_fc
        self.r_mean = float(r_mean)
        self.r_std = float(r_std)
        self.register_buffer("adj", torch.zeros(1, L, n_aoi + n_choke,
                                                n_aoi + n_choke))

    def set_adj(self, adj: torch.Tensor) -> None:
        if adj.dim() == 3:
            adj = adj.unsqueeze(0)
        self.adj = adj.detach()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        fin, aoi, choke = unpack_inputs(
            x, self.L, self.n_fin, self.n_aoi, self.n_fa,
            self.n_choke, self.n_fc)
        adj = self.adj.expand(x.shape[0], -1, -1, -1)
        r_hat, _ = self.model(aoi=aoi, choke=choke, adj=adj, fin=fin)
        return r_hat * self.r_std + self.r_mean


def expected_gradients(wrapper: InputShapWrapper, x: torch.Tensor,
                       bg: torch.Tensor, n_steps: int
                       ) -> tuple[np.ndarray, float, float]:
    """Expected Gradients / GradientSHAP with a midpoint Riemann sum.

    Averages Integrated Gradients from every background sample. Completeness
    is sum(phi) ≈ f(x) - mean_b f(x_b). x is (1, D), bg is (M, D).
    """
    wrapper.eval()
    m, d = bg.shape
    alphas = ((torch.arange(n_steps, device=x.device, dtype=x.dtype) + 0.5)
              / n_steps)
    delta = x - bg
    x_interp = (bg.unsqueeze(1)
                + alphas.view(1, n_steps, 1) * delta.unsqueeze(1))
    x_interp = x_interp.reshape(m * n_steps, d).detach().requires_grad_(True)
    y = wrapper(x_interp)
    grad = torch.autograd.grad(y.sum(), x_interp)[0]
    grad = grad.view(m, n_steps, d).mean(dim=1)
    phi = (grad * delta).mean(dim=0)
    with torch.no_grad():
        f_x = float(wrapper(x).item())
        f_bg = float(wrapper(bg).mean().item())
    return phi.detach().cpu().numpy().astype(np.float32), f_x, f_bg


def lock_period_table() -> pd.DataFrame:
    rows = [
        {"period_id": "full", "label": "Full OOS", "kind": "all",
         "start": "", "end": "", "event_date": "",
         "window_weeks": "", "rule": "all scored OOS weeks"},
    ]
    for y in CALENDAR_YEARS:
        rows.append({
            "period_id": f"year_{y}", "label": str(y), "kind": "calendar",
            "start": f"{y}-01-01", "end": f"{y + 1}-01-01",
            "event_date": "", "window_weeks": "",
            "rule": f"{y}-01-01 <= date < {y + 1}-01-01",
        })
    for pid, d, lab in EVENTS:
        rows.append({
            "period_id": pid, "label": lab, "kind": "event",
            "start": "", "end": "", "event_date": d,
            "window_weeks": EVENT_WINDOW_WEEKS,
            "rule": (f"event_date ± {EVENT_WINDOW_WEEKS} weeks, "
                     "inclusive both ends"),
        })
    return pd.DataFrame(rows)


def period_membership(dates: pd.DatetimeIndex) -> pd.DataFrame:
    rows = []
    for d in dates:
        flags = {
            "full": True,
        }
        for y in CALENDAR_YEARS:
            flags[f"year_{y}"] = (pd.Timestamp(f"{y}-01-01")
                                  <= d < pd.Timestamp(f"{y + 1}-01-01"))
        for pid, ev, _ in EVENTS:
            centre = pd.Timestamp(ev)
            lo = centre - pd.Timedelta(weeks=EVENT_WINDOW_WEEKS)
            hi = centre + pd.Timedelta(weeks=EVENT_WINDOW_WEEKS)
            flags[pid] = lo <= d <= hi
        for pid, inn in flags.items():
            if inn:
                rows.append({"date": d, "period_id": pid})
    return pd.DataFrame(rows)


def lock_group_table() -> pd.DataFrame:
    rows = []
    for gid, mod, desc in GROUP_META:
        rows.append({"group_id": gid, "modality": mod, "description": desc})
    return pd.DataFrame(rows)


def write_long_shap(path: Path, dates: pd.DatetimeIndex,
                    phi: np.ndarray, fmap: pd.DataFrame) -> None:
    """Gzipped long table: one row per (week, flattened feature)."""
    n_weeks, n_feat = phi.shape
    keep = ["lag", "channel", "node_id", "feature", "group"]
    meta = fmap[keep]
    date_rep = np.repeat(dates.to_numpy(), n_feat)
    tiled = pd.concat([meta] * n_weeks, ignore_index=True)
    tiled.insert(0, "date", date_rep)
    tiled["shap_signed"] = phi.reshape(-1)
    tiled["shap_abs"] = np.abs(phi).reshape(-1)
    tiled.to_csv(path, index=False, compression="gzip")


def qc_mean_abs(phi: np.ndarray, fmap: pd.DataFrame) -> pd.DataFrame:
    """Full-sample mean_t I_{t,f} = mean_t sum_lag |phi_{t,lag,f,node}|."""
    abs_mean = np.abs(phi).mean(axis=0)
    tmp = fmap.copy()
    tmp["mean_abs_shap"] = abs_mean
    key = ["channel", "node_id", "feature", "group"]
    out = (tmp.groupby(key, dropna=False)["mean_abs_shap"].sum()
           .reset_index()
           .sort_values("mean_abs_shap", ascending=False))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(
        description="M3_Deep_gated seed-42 weekly input SHAP + node attention")
    ap.add_argument("--lookback", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--min-train", type=int, default=104)
    ap.add_argument("--retrain-every", type=int, default=13)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-background", type=int, default=64)
    ap.add_argument("--n-steps", type=int, default=32,
                    help="IG interpolation steps per background sample")
    ap.add_argument("--max-weeks", type=int, default=0,
                    help="0 = all OOS weeks; >0 for a smoke test")
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--lock-only", action="store_true",
                    help="rewrite period lock + membership from existing "
                         "deep_m3_shap_dates.csv; do not recompute SHAP")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    periods = lock_period_table()
    groups = lock_group_table()
    periods.to_csv(OUT_DIR / "deep_m3_shap_lock_periods.csv", index=False)
    groups.to_csv(OUT_DIR / "deep_m3_shap_lock_groups.csv", index=False)
    if args.lock_only:
        dates_path = OUT_DIR / "deep_m3_shap_dates.csv"
        if not dates_path.exists():
            raise FileNotFoundError(
                f"{dates_path} missing; run without --lock-only first")
        dates_ix = pd.DatetimeIndex(pd.read_csv(dates_path)["date"])
        memb = period_membership(dates_ix)
        memb.to_csv(OUT_DIR / "deep_m3_shap_period_membership.csv",
                    index=False)
        print(f"Rewrote lock periods ({len(periods)} rows) and membership "
              f"({len(memb)} rows) from {len(dates_ix)} dates.", flush=True)
        print(memb.groupby("period_id").size().to_string(), flush=True)
        return

    df = data.load_matrix()
    dico = data.load_dict()
    ds = build_deep_dataset(df, dico, lookback=args.lookback, with_rs=False)
    fmap = build_feature_map(ds)
    fmap.to_csv(OUT_DIR / "deep_m3_shap_feature_map.csv", index=False)

    modalities, _, fusion_type = CONFIGS[CONFIG_KEY]
    idx = ds["idx"]
    n = len(idx)
    L = int(ds["lookback"])
    n_fin = ds["fin"].shape[-1]
    n_aoi = ds["aoi"].shape[2]
    n_fa = ds["aoi"].shape[-1]
    n_choke = ds["choke"].shape[2]
    n_fc = ds["choke"].shape[-1]
    node_ids = list(ds["node_ids"])
    n_feat = len(fmap)
    device = args.device

    oos_pos = [i for i in range(n) if i >= args.min_train]
    if args.max_weeks and args.max_weeks > 0:
        oos_pos = oos_pos[:args.max_weeks]
    n_oos = len(oos_pos)
    print(f"=== M3_Deep_gated SHAP seed={args.seed} lb={L} "
          f"epochs={args.epochs} n_oos={n_oos} D={n_feat} "
          f"bg={args.n_background} steps={args.n_steps} ===", flush=True)
    print(f"  feature map: fin={L}x{n_fin} aoi={L}x{n_aoi}x{n_fa} "
          f"choke={L}x{n_choke}x{n_fc}", flush=True)

    phi_all = np.zeros((n_oos, n_feat), dtype=np.float32)
    dates = []
    add_rows = []
    att_rows = []
    model = None
    sc = None
    wrapper = None
    bg_flat = None
    r_mean = r_std = 0.0
    rng = None
    fit_week = None

    long_path = OUT_DIR / "deep_m3_shap_weekly_long.csv.gz"
    if long_path.exists():
        long_path.unlink()

    for k, i in enumerate(oos_pos):
        if model is None or ((i - args.min_train) % args.retrain_every == 0):
            sc = fit_scalers(ds, train_n=i)
            model, r_mean, r_std = _train_fold(
                ds, sc, i, modalities, args.seed, args.epochs, 1e-3, 1e-4, 32,
                52, device, {"d": 32, "gat_layers": 2, "tcn_layers": 2},
                fusion_type=fusion_type)
            model.eval()
            wrapper = InputShapWrapper(
                model, L, n_fin, n_aoi, n_fa, n_choke, n_fc, r_mean, r_std
            ).to(device)
            Xtr = _to_tensors(apply_scalers(ds, sc, slice(0, i)), device)
            packed = pack_inputs(Xtr["fin"], Xtr["aoi"], Xtr["choke"])
            n_tr = packed.shape[0]
            n_bg = min(args.n_background, n_tr)
            rng = np.random.default_rng(args.seed + i)
            bg_idx = rng.choice(n_tr, size=n_bg, replace=False)
            bg_flat = packed[bg_idx].detach()
            fit_week = idx[i]
            print(f"  fit @ {idx[i].date()} train={i} bg={n_bg}", flush=True)

        Xte = _to_tensors(apply_scalers(ds, sc, slice(i, i + 1)), device)
        wrapper.set_adj(Xte["adj"])
        x_flat = pack_inputs(Xte["fin"], Xte["aoi"], Xte["choke"])

        model.eval()
        with torch.no_grad():
            _, info = model(aoi=Xte["aoi"], choke=Xte["choke"],
                            adj=Xte["adj"], fin=Xte["fin"])
        att = info["ship_site_att"][0].detach().cpu().numpy()
        for j, name in enumerate(node_ids):
            att_rows.append({
                "date": idx[i], "node_id": name,
                "node_attention": float(att[j]),
            })

        phi, f_x, f_bg = expected_gradients(
            wrapper, x_flat, bg_flat, args.n_steps)
        phi_all[k] = phi
        dates.append(idx[i])
        sum_phi = float(phi.sum())
        resid = f_x - f_bg - sum_phi
        add_rows.append({
            "date": idx[i], "f_x": f_x, "e_f_bg": f_bg,
            "sum_phi": sum_phi, "residual": resid,
            "abs_residual": abs(resid),
            "rel_residual": abs(resid) / (abs(f_x - f_bg) + 1e-8),
            "checkpoint_week": str(fit_week.date()),
        })
        if (k + 1) % 13 == 0 or k + 1 == n_oos:
            med = np.median([r["abs_residual"] for r in add_rows])
            print(f"    shap {k + 1}/{n_oos}  {idx[i].date()}  "
                  f"|resid|median={med:.4g}", flush=True)

    dates_ix = pd.DatetimeIndex(dates)
    np.save(OUT_DIR / "deep_m3_shap_values.npy", phi_all)
    pd.Series(dates_ix, name="date").to_csv(
        OUT_DIR / "deep_m3_shap_dates.csv", index=False)

    print("  writing long SHAP table …", flush=True)
    write_long_shap(long_path, dates_ix, phi_all, fmap)

    add_df = pd.DataFrame(add_rows)
    add_df.to_csv(OUT_DIR / "deep_m3_shap_additivity.csv", index=False)
    att_df = pd.DataFrame(att_rows)
    att_df.to_csv(OUT_DIR / "deep_m3_node_attention_weekly.csv", index=False)
    memb = period_membership(dates_ix)
    memb.to_csv(OUT_DIR / "deep_m3_shap_period_membership.csv", index=False)
    qc = qc_mean_abs(phi_all, fmap)
    qc.to_csv(OUT_DIR / "deep_m3_shap_qc_mean_abs.csv", index=False)

    n_bad = int((add_df["rel_residual"] > 0.10).sum())
    print("\nAdditivity:",
          f"median |resid|={add_df['abs_residual'].median():.4g}",
          f"p90={add_df['abs_residual'].quantile(0.9):.4g}",
          f"weeks with rel_resid>0.10: {n_bad}/{len(add_df)}",
          flush=True)
    print("Top I_f (full-sample mean |SHAP| summed over lags):", flush=True)
    print(qc.head(12).to_string(index=False), flush=True)
    print(f"\nSaved under {OUT_DIR}", flush=True)
    for name in [
        "deep_m3_shap_lock_periods.csv",
        "deep_m3_shap_lock_groups.csv",
        "deep_m3_shap_feature_map.csv",
        "deep_m3_shap_period_membership.csv",
        "deep_m3_shap_values.npy",
        "deep_m3_shap_weekly_long.csv.gz",
        "deep_m3_shap_additivity.csv",
        "deep_m3_node_attention_weekly.csv",
        "deep_m3_shap_qc_mean_abs.csv",
    ]:
        print(f"  {name}", flush=True)


if __name__ == "__main__":
    main()
