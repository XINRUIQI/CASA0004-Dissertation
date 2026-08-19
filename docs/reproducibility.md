# Reproducibility

Authoritative run instructions:

1. [`../Readme.md`](../Readme.md) — overview, software environment, protocol
2. [`../code/README.md`](../code/README.md) — **full runbook**
3. [`appendix/appendix_C_config_EN.md`](appendix/appendix_C_config_EN.md) — locked hyperparameter grids
4. [`appendix/appendix_A_data_EN.md`](appendix/appendix_A_data_EN.md) — variables / AOIs / lags / graph edges

---

## Minimal checklist

### Software

```bash
python3 -m pip install -r code/requirements.txt
```

Python **3.9.6** (CPython, macOS); pinned packages in `code/requirements.txt`. Training and evaluation do **not** need `transformers`.

### Inputs (already in the repository)

| Path | Role |
| --- | --- |
| `data/processed/merge/outputs/weekly_feature_matrix.csv` | weekly feature matrix |
| `data/processed/merge/outputs/weekly_feature_dictionary.csv` | feature dictionary |
| `data/processed/M3/outputs/m3_graph17_tensors.npz` | deep shipping graph |
| `data/processed/M2/outputs/s2_prithvi_emb_meanpool.npy` | RS embeddings |
| `data/processed/M2/outputs/s2_prithvi_emb_index.csv` | embedding index |

### Commands (main results)

```bash
# From repository root
python3 code/scripts/flat/run_baseline.py --modality M1
python3 code/scripts/flat/run_baseline.py --modality M2 --m2-features anom
python3 code/scripts/flat/run_baseline.py --modality M3
python3 code/scripts/flat/run_baseline.py --modality M4
python3 code/scripts/deep/run_deep_baseline.py
```

### Output locations

| Result | Path |
| --- | --- |
| Flat metrics / preds | `results/baselines/Flat/M*_Flat/` |
| Deep metrics / CW / preds | `results/baselines/Deep/_cross/` |
| Sub-period (optional) | `results/baselines/subperiod/` |

Compare the CSVs already stored under `results/` with a fresh run if you need to check numerical agreement.

---

## Protocol (main specification)

| Item | Locked value |
| --- | --- |
| lookback | 4 |
| min_train | 104 |
| retrain_every | 13 |
| val_weeks | 52 |
| seed (main) | 42 (robustness: 1, 2) |
| Flat main settings | M2=`anom`; M3=`full` |
| Deep main fusion | gated; representation dim d=32 |

Rebuild-from-raw and licensing: [`../data/sources.md`](../data/sources.md). Raw downloads themselves are not in git (`data/raw/`).
