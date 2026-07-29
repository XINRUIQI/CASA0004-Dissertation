# 04_code — Runbook (reproduction)

Run all commands from the **repository root**. By default, assume `03_data/processed/` products already exist — this is the fastest path to reproduce the paper’s main tables.

Full method / result narrative:

- Flat: `00_admin/最新待整理/flat_baseline_full_walkthrough_EN.md`
- Deep: `00_admin/最新待整理/deep_model_full_walkthrough_EN.md`
- Locked hyperparameters: `06_writing/Appendix/appendix_C_config.md`

---

## 1. Environment

```bash
cd "/path/to/casa0004 Dissertation"
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
python3 -m pip install -U pip
python3 -m pip install -r 04_code/requirements.txt
```

| Item | Note |
| --- | --- |
| Python | **3.9.x** (tested 3.9.6) |
| Device | Deep main analysis defaults to **CPU** |
| Not required | `transformers`, online Prithvi download (embeddings are precomputed) |
| macOS | Deep scripts **read** existing Flat M1 predictions to avoid loading torch+xgboost in one process (OpenMP conflict) |

Core packages: see `requirements.txt` (numpy / pandas / scipy / scikit-learn / xgboost / torch / matplotlib / shap).

---

## 2. Preflight checks

```bash
# Required for Flat
test -f 03_data/processed/merge/outputs/weekly_feature_matrix.csv
test -f 03_data/processed/merge/outputs/weekly_feature_dictionary.csv

# Extra for Deep
test -f 03_data/processed/M3/outputs/m3_graph17_tensors.npz
test -f 03_data/processed/M2/outputs/s2_prithvi_emb_meanpool.npy
test -f 03_data/processed/M2/outputs/s2_prithvi_emb_index.csv
```

Layout:

```text
04_code/
├── requirements.txt
├── scripts/
│   ├── flat/
│   │   ├── run_baseline.py          # ★ Flat single entry (M0–M4)
│   │   └── M{1..4}_Flat/            # sweep / robustness / shap
│   ├── deep/
│   │   ├── run_deep_baseline.py     # ★ Deep main entry
│   │   ├── run_deep_{sweep,fusion_matrix,advanced}.py
│   │   ├── run_deep_interpret*.py / run_deep_xattn_viz.py
│   │   └── diagnose_*.py …
│   └── tools/
│       ├── subperiod_eval.py
│       ├── migrate_model_names.py
│       └── relocate_deep_outputs.py
└── src/
    ├── backtest/                    # Flat rolling backtest
    ├── models/                      # encoders + fusion + deep_*
    └── model_naming.py
```

---

## 3. Flat baselines (main analysis)

Shared engine: `04_code/src/backtest/`. Each modality runs Ridge + XGB; nested layers (M2/3/4) re-run M1 on the same weeks and report DM vs M0 and Clark–West vs M1.

```bash
# Main defaults: lookback=4, min_train=104, retrain_every=13, val_weeks=52, seed=42
python3 04_code/scripts/flat/run_baseline.py --modality M1
python3 04_code/scripts/flat/run_baseline.py --modality M2 --m2-features anom
python3 04_code/scripts/flat/run_baseline.py --modality M3              # m3-tier=full (default)
python3 04_code/scripts/flat/run_baseline.py --modality M4
```

Common flags:

| Flag | Default | Meaning |
| --- | --- | --- |
| `--modality` | M1 | M1 / M2 / M3 / M4 |
| `--m2-features` | anom | anom (main) / level / literature / aoi4 / … |
| `--m3-tier` | full | full (main, 113 cols) / core (robustness) |
| `--lookback` | 4 | lookback weeks |
| `--seed` | 42 | random seed |
| `--leave-one-aoi-out` | off | M2-bearing: drop each AOI in turn |
| `--matrix` / `--dict` | merge defaults | e.g. watermask matrix variants |
| `--replot-only` | off | rebuild plots from existing CSVs only |

**Outputs** → `05_outputs/baselines/Flat/M*_Flat/`

| File | Content |
| --- | --- |
| `baseline_metrics.csv` | RMSE / MAE / skill / DM / CW, etc. |
| `baseline_predictions.csv` | weekly predictions (Deep reads the M1 file) |
| `backtest.png` | backtest figure |

For M2 with `anom`, filenames may be suffixed (`baseline_metrics_anom.csv`); see the script docstring.

### 3.1 Flat robustness / SHAP / sweep

```bash
python3 04_code/scripts/flat/M1_Flat/sweep_m1.py
python3 04_code/scripts/flat/M2_Flat/shap_m2.py
python3 04_code/scripts/flat/M2_Flat/robustness_m2.py
python3 04_code/scripts/flat/M2_Flat/sweep_m2.py
python3 04_code/scripts/flat/M3_Flat/shap_m3.py
python3 04_code/scripts/flat/M3_Flat/robustness_m3.py
python3 04_code/scripts/flat/M3_Flat/sweep_m3.py
python3 04_code/scripts/flat/M4_Flat/shap_m4.py
python3 04_code/scripts/flat/M4_Flat/robustness_m4.py
python3 04_code/scripts/flat/M4_Flat/sweep_m4.py
```

---

## 4. Deep baselines (main analysis)

**Prerequisite**: `05_outputs/baselines/Flat/M1_Flat/baseline_predictions.csv` exists (run Flat M1 in §3 first).

```bash
python3 04_code/scripts/deep/run_deep_baseline.py
# Subset example:
python3 04_code/scripts/deep/run_deep_baseline.py --modes m3_deep_gated,m4_deep_gated
```

Default modes: `m1_deep, m_ship_gnn, m_rs_deep, m3_deep_gated, m2_deep_gated, m4_deep_gated, m4_deep_concat`.

| Flag | Default | Meaning |
| --- | --- | --- |
| `--lookback` | 4 | aligned with Flat |
| `--min-train` / `--retrain-every` | 104 / 13 | same as Flat |
| `--epochs` | 80 | early stopping patience=12 (see `deep_rolling`) |
| `--seed` | 42 | main seed |
| `--replot-only` | off | redraw `deep_backtest.png` from existing CSVs |

**Outputs** → `05_outputs/baselines/Deep/` (on macOS this may be the same path as `deep/`)

| Path | Content |
| --- | --- |
| `_cross/deep_metrics.csv` | main metrics table |
| `_cross/deep_cw.csv` | CW / DM comparisons |
| `_cross/deep_predictions.csv` | predictions |
| `_cross/deep_backtest.png` | figure |
| `M*_Deep/baseline_*.csv` | slim per-tier exports |

### 4.1 Deep extensions

```bash
python3 04_code/scripts/deep/run_deep_fusion_matrix.py          # 3×3 fusion matrix
python3 04_code/scripts/deep/run_deep_sweep.py                  # seed / lookback / dim, etc.
python3 04_code/scripts/deep/run_deep_advanced.py               # advanced ablations
python3 04_code/scripts/deep/run_deep_interpret.py --seeds 42,1,2 --lookback 4
python3 04_code/scripts/deep/run_deep_interpret_m3.py
python3 04_code/scripts/deep/run_deep_xattn_viz.py
```

Diagnostics (optional): `diagnose_deep.py`, `diagnose_rs.py`, `compare_rs_anom.py`, `multiseed_rs_anom.py`.

---

## 5. Sub-period evaluation (Flat + Deep, offline)

Needs existing Flat / Deep prediction CSVs:

```bash
python3 04_code/scripts/tools/subperiod_eval.py
# → 05_outputs/baselines/subperiod/subperiod_summary.csv
```

---

## 6. Suggested order (shortest closed loop)

```text
Install deps
  → Flat M1
  → Flat M2 / M3 / M4 (can run in parallel terminals)
  → Deep baseline
  → (optional) fusion_matrix / interpret / subperiod / Flat SHAP
```

To cross-check reported numbers: open `05_outputs/baselines/Flat/*/baseline_metrics*.csv` and `Deep/_cross/deep_metrics.csv`, and compare with `00_admin/最新待整理/项目逻辑与结果总览_CN.md` (or the EN walkthroughs).

---

## 7. Data rebuild (optional; needs local raw)

`03_data/raw/` is gitignored. Only if rebuilding features:

```bash
# M1 / M2 / M3 → merge
python3 03_data/processed/M1/py/build_m1_weekly.py
python3 03_data/processed/M2/py/build_m2_weekly.py
# python3 03_data/processed/M2/py/build_m2_weekly.py --watermask   # robustness B4
python3 03_data/processed/M3/py/aggregate_shipping_to_weekly.py
python3 03_data/processed/merge/py/build_feature_matrix.py

# Deep extras
python3 03_data/processed/M3/py/build_m3_graph17.py
# Prithvi embedding export (separate env with foundation-model stack; not this requirements)
# python3 03_data/processed/M2/py/precompute_s2_embeddings.py
```

Sources: `03_data/Dataset/external_sources.md`.  
Variables and lags: `06_writing/Appendix/appendix_A_data.md`.

**Do not mix lag conventions**: Flat GFW presence **+4w**, PortWatch **+1w**; Deep graph GFW event/O-D **+2w**, SAR **+4w**.

---

## 8. Utility scripts

| Script | Role |
| --- | --- |
| `tools/subperiod_eval.py` | early / late sub-period table |
| `tools/migrate_model_names.py` | historical name migration (rarely needed) |
| `tools/relocate_deep_outputs.py` | Deep output layout tidy (rarely needed) |

---

## 9. Troubleshooting

| Symptom | Fix |
| --- | --- |
| Deep: Missing M1 predictions | Run `run_baseline.py --modality M1` first |
| Missing merge CSV / graph npz / emb npy | Check `03_data/processed/`; or rebuild via §7 |
| macOS torch + xgboost conflict | Run Flat and Deep in separate processes; Deep already reads Flat CSVs |
| Numbers differ slightly from thesis | Confirm seed=42, lookback=4, main modality settings; for tiny CPU nondeterminism, prefer the CSVs already stored in the repo |
| `Deep/` vs `deep/` | On case-insensitive APFS they are the same path |
