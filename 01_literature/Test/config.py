"""
Project-wide configuration: paths, variable subsets, and hyperparameters.
Follows project_plan_20260601.md Section 5–6 & beatrice_task_literature_matrix.md.
"""
from __future__ import annotations
from pathlib import Path

# ── Paths ────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_CSV = PROJECT_ROOT / "03_data" / "processed" / "weekly_features.csv"
OUT_DIR = Path(__file__).resolve().parent / "results"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Targets ──────────────────────────────────────────────────────
TARGETS = {
    "direction": "target_brent_direction_next_1w",
    "volatility": "target_brent_vol_next_1w",
    "price": "target_brent_price_next_1w",
}

# ── Beatrice-recommended variable subsets (literature-backed) ────
# M1 = mechanism-based 10-variable set (post-close-reading, see
# beatrice_task_literature_matrix.md §①). OVX preferred over VIX (P052);
# ΔDGS10 (change) instead of level (P076). Price lags/returns are added via
# LAG_MA_SPECS on brent_price below.
M1_VARS = [
    "brent_price",                  # oil-price dynamics (lags/ma/mom via LAG_MA)
    "crude_stocks_change",          # supply / market balance (P053 strong)
    "global_econ_activity",         # global demand — Kilian REA (P053 strong)
    "nonoil_industrial_commodity",  # global demand — industrial materials (P053)
    "futures_spread",               # market tightness / expectations (P053/P054)
    "ovx",                          # oil-specific uncertainty (P052: OVX>VIX)
    "gpr",                          # precautionary demand — geopolitical risk (P076)
    "dgs10_change",                 # rates / carry cost — ΔDGS10 (P076)
    "gold_return",                  # commodity comovement / safe haven (P004)
    "commodity_fx",                 # FX channel — CAD/AUD (P053>broad USD)
]

M2_RS_ADD = [
    "ntl_ntl_avg_rad_mean_P001",  # Rotterdam
    "ntl_ntl_avg_rad_mean_P002",  # Fujairah
    "ntl_ntl_avg_rad_mean_P003",  # Ras Tanura
    "ntl_ntl_avg_rad_mean_P005",  # Houston
    "ntl_ntl_avg_rad_mean_P006",  # Ningbo
    "frt_fill_level_fujairah",    # FRT fill level (CV-derived)
]

M3_SHIP_PW = [
    "pw_hormuz_n_tanker", "pw_hormuz_capacity_tanker",
    "pw_suez_n_tanker", "pw_suez_capacity_tanker",
    "pw_malacca_n_tanker", "pw_malacca_capacity_tanker",
    "pw_all_n_tanker_sum",
]

M3_SHIP_GFW = [
    "gfw_hormuz_total_hours", "gfw_hormuz_total_vessels",
    "gfw_suez_total_hours", "gfw_suez_total_vessels",
    "gfw_malacca_total_hours", "gfw_malacca_total_vessels",
    "gfw_all_total_hours_sum",
]

M3_SHIP_EMODNET = [
    "emodnet_mean_rotterdam", "emodnet_max_rotterdam",
    "emodnet_mean_suez", "emodnet_max_suez",
]

M3_SHIP_ADD = M3_SHIP_GFW + M3_SHIP_EMODNET  # GFW (2012+) + EMODnet density (2017+)

# ── M5 supplementary: GDELT NLP/event features (Discussion/Appendix) ─
M5_GDELT_ADD = [
    "gdelt_oil_geo_event_count",
    "gdelt_oil_geo_avg_tone",
    "gdelt_oil_geo_avg_goldstein",
    "gdelt_oil_geo_negative_share",
    "gdelt_oil_geo_conflict_share",
    "gdelt_transport_disruption_event_count",
    "gdelt_transport_disruption_avg_tone",
    "gdelt_transport_disruption_avg_goldstein",
    "gdelt_transport_negative_share",
    "gdelt_chokepoint_event_count",
    "gdelt_combined_event_count",
    "gdelt_oil_geo_event_count_4w_ma",
    "gdelt_transport_event_count_4w_ma",
]

# ── Lag / MA / Momentum features (Strategy A1) ───────────────────
# Bases restricted to the new M1 set (downgraded vars removed: vix/dollar_index/
# crude_production/sp500_return_pct/treasury_10y). brent_price lags/ma/mom carry
# the "price lags + returns" information from the literature (P053/P001/P072).
LAG_MA_SPECS: dict[str, list[dict]] = {
    "brent_price": [
        {"type": "lag",  "periods": 1},
        {"type": "lag",  "periods": 4},
        {"type": "ma",   "window": 4},
        {"type": "ma",   "window": 12},
        {"type": "mom",  "window": 4},
        {"type": "mom",  "window": 12},
    ],
    "ovx": [
        {"type": "lag",  "periods": 1},
        {"type": "ma",   "window": 4},
    ],
    "crude_stocks_change": [
        {"type": "ma",   "window": 4},
    ],
    "futures_spread": [
        {"type": "lag",  "periods": 1},
        {"type": "ma",   "window": 4},
    ],
    "gpr": [
        {"type": "lag",  "periods": 1},
        {"type": "ma",   "window": 4},
    ],
    "dgs10_change": [
        {"type": "lag",  "periods": 1},
    ],
    "commodity_fx": [
        {"type": "ma",   "window": 4},
    ],
    "gold_return": [
        {"type": "ma",   "window": 4},
    ],
}

LAYER_FEATURES: dict[str, list[str]] = {
    "M1": M1_VARS,
    "M2": M1_VARS + M2_RS_ADD,
    "M3": M1_VARS + M3_SHIP_ADD,
    "M4": M1_VARS + M2_RS_ADD + M3_SHIP_ADD,
    "M5": M1_VARS + M2_RS_ADD + M3_SHIP_ADD + M5_GDELT_ADD,
}

# ── Train / Val / Test split (temporal, no shuffle) ──────────────
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# ── LSTM / TFT hyperparameters ───────────────────────────────────
LOOKBACK = 12          # weeks of history fed to sequence models
LSTM_HIDDEN = 64
LSTM_LAYERS = 2
LSTM_DROPOUT = 0.2
LSTM_EPOCHS = 200
LSTM_PATIENCE = 20
LSTM_LR = 1e-3
LSTM_BATCH = 32

TFT_HIDDEN = 32
TFT_ATTENTION_HEADS = 4
TFT_DROPOUT = 0.1
TFT_EPOCHS = 150
TFT_PATIENCE = 15
TFT_LR = 1e-3
TFT_BATCH = 32

# ── ST-GNN hyperparameters ───────────────────────────────────────
GNN_HIDDEN = 32
GNN_LAYERS = 2
GNN_DROPOUT = 0.2
GNN_EPOCHS = 200
GNN_PATIENCE = 20
GNN_LR = 1e-3
GNN_BATCH = 32

# ── AOI & chokepoint topology for ST-GNN ─────────────────────────
AOI_NODES = [
    "P001", "P002", "P003", "P004", "P005",
    "P006", "P007", "P008", "P009", "P010", "P011",
]
AOI_NAMES = {
    "P001": "Rotterdam", "P002": "Fujairah", "P003": "Ras Tanura",
    "P004": "Jurong Island", "P005": "Houston", "P006": "Ningbo",
    "P007": "Jamnagar", "P008": "Basra", "P009": "Ulsan",
    "P010": "Kharg Island", "P011": "Yanbu",
}
CHOKEPOINT_EDGES = [
    ("P003", "P002", "hormuz"), ("P008", "P002", "hormuz"),
    ("P010", "P002", "hormuz"),
    ("P004", "P006", "malacca"), ("P004", "P009", "malacca"),
    ("P011", "P001", "suez"),
    ("P011", "P003", "mandeb"),
    ("P005", "P004", "panama"),
    ("P001", "P002", "cape"),
]

SEED = 42
