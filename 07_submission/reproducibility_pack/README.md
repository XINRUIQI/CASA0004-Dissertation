# Reproducibility Pack — Index

This folder is the submission-facing entry for reproduction. Code and data remain in their repository paths; this file lists the **minimal reproduce path** and documentation pointers.

Authoritative run instructions:

1. [`../../Readme.md`](../../Readme.md) — project overview, software environment, protocol  
2. [`../../04_code/README.md`](../../04_code/README.md) — **full runbook**  
3. [`../../06_writing/Appendix/appendix_C_config.md`](../../06_writing/Appendix/appendix_C_config.md) — locked hyperparameter grids  
4. [`../../06_writing/Appendix/appendix_A_data.md`](../../06_writing/Appendix/appendix_A_data.md) — variables / AOIs / lags / graph edges  

---

## Minimal reproduction checklist

### Software

```bash
python3 -m pip install -r 04_code/requirements.txt
```

Python **3.9.6** (CPython, macOS); pinned packages in `04_code/requirements.txt` (numpy 2.0.2, pandas 2.3.3, scipy 1.13.1, scikit-learn 1.6.1, xgboost 2.1.4, torch 2.8.0, matplotlib 3.9.4, shap 0.49.1). Training and evaluation do **not** need `transformers`. Full table: [`../../Readme.md`](../../Readme.md) § Software environment.

### Inputs (processed products already in repo)

| Path | Role |
| --- | --- |
| `03_data/processed/merge/outputs/weekly_feature_matrix.csv` | weekly feature matrix |
| `03_data/processed/merge/outputs/weekly_feature_dictionary.csv` | feature dictionary |
| `03_data/processed/M3/outputs/m3_graph17_tensors.npz` | deep shipping graph |
| `03_data/processed/M2/outputs/s2_prithvi_emb_meanpool.npy` | RS embeddings |
| `03_data/processed/M2/outputs/s2_prithvi_emb_index.csv` | embedding index |

### Commands (main results)

```bash
# From repository root
python3 04_code/scripts/flat/run_baseline.py --modality M1
python3 04_code/scripts/flat/run_baseline.py --modality M2 --m2-features anom
python3 04_code/scripts/flat/run_baseline.py --modality M3
python3 04_code/scripts/flat/run_baseline.py --modality M4
python3 04_code/scripts/deep/run_deep_baseline.py
```

### Output locations

| Result | Path |
| --- | --- |
| Flat metrics / preds | `05_outputs/baselines/Flat/M*_Flat/` |
| Deep metrics / CW / preds | `05_outputs/baselines/Deep/_cross/` |
| Sub-period (optional) | `05_outputs/baselines/subperiod/` |

Main-number narrative: `00_admin/最新待整理/项目逻辑与结果总览_CN.md` (see also EN walkthroughs).

---

## Protocol summary (changing these is not the main specification)

| Item | Locked value |
| --- | --- |
| lookback | 4 |
| min_train | 104 |
| retrain_every | 13 |
| val_weeks | 52 |
| seed (main) | 42 (robustness: 1, 2) |
| Flat main settings | M2=`anom`; M3=`full` |
| Deep main fusion | gated; representation dim d=32 |

---

## End-to-end narrative docs

| Arm | Walkthrough |
| --- | --- |
| Flat | `00_admin/最新待整理/flat_baseline_full_walkthrough_{CN,EN}.md` |
| Deep | `00_admin/最新待整理/deep_model_full_walkthrough_{CN,EN}.md` |

Rebuild-from-raw and licensing: `03_data/Dataset/external_sources.md` (raw itself is not in git).

---

## Packaging suggestion (before submission)

If you need a standalone zip, include at least:

```text
reproducibility_pack/
├── README.md                 # this file
├── requirements.txt          # copy from 04_code/
├── code/                     # 04_code/scripts + 04_code/src (or point to the repo)
├── data_processed/           # merge CSV + dictionary + graph17 npz + prithvi npy/index
└── expected_outputs/         # optional: stored baseline_metrics / deep_metrics for comparison
```

The repo currently uses path references; copy physically into this folder when preparing the final submission package.
