# Appendix C — Hyperparameter grids & locked settings

This appendix records the software versions and locked settings used for the
reported results. Installation commands and scripts are in the GitHub repository.

---

## C.1 Software environment

Python 3.9.6 (CPython, macOS). Core packages:

| Package | Version | Role |
| --- | --- | --- |
| numpy | 2.0.2 | arrays |
| pandas | 2.3.3 | weekly matrix |
| scipy | 1.13.1 | *p* values |
| scikit-learn | 1.6.1 | Ridge, scaling |
| xgboost | 2.1.4 | Flat XGBoost |
| torch | 2.8.0 | Deep encoders and fusion (CPU) |
| matplotlib | 3.9.4 | figures |
| shap | 0.49.1 | attribution |

The Deep remote-sensing branch uses pre-computed frozen Prithvi-EO-2.0 embeddings.
Training and evaluation do not load the foundation model.

---

## C.2 Shared protocol and seeds

Flat and Deep use the same rolling-origin schedule.

| Item | Value |
| --- | --- |
| Sample | 2019–2025 (365 weeks) |
| Lookback | **4 weeks** |
| Initial training (not scored) | **104 weeks** |
| Re-estimation interval | **13 weeks** |
| Inner validation | **52 weeks** (tail of each training fold) |
| Evaluation sample | **257 weeks** |
| Main seed | **42** |
| Robustness seeds | **1, 2** |

---

## C.3 Flat search grids

Hyperparameters are chosen inside each training fold on the inner-validation
segment only.

| Learner | Grid |
| --- | --- |
| Ridge (α) | {0.1, 1.0, 10.0, 100.0, 1000.0} |
| XGBoost `max_depth` | {2, 3} |
| XGBoost `learning_rate` | {0.03, 0.05} |
| XGBoost `n_estimators` | {200, 400} |
| XGBoost fixed | `subsample=0.8`, `colsample_bytree=0.8`, `reg_lambda=1.0` |

---

## C.4 Locked Deep architecture and training

The main specification is locked to lookback 4 and latent size 32, matching the
Flat lookback. Sensitivity is reported in Appendix B.

### C.4.1 Encoders

| Encoder | Settings | Output |
| --- | --- | --- |
| Finance TCN | 2 layers, kernel 3, causal, dropout 0.1 | 32-d |
| Remote sensing | frozen Prithvi embeddings (1024-d), temporal then site attention | 32-d |
| Shipping GAT | 17 nodes, 2 GAT layers, 4 heads, then 2-layer TCN | 32-d |

### C.4.2 Fusion

| Option | Role |
| --- | --- |
| Concatenation | alternative |
| **Gated fusion** | **main specification** (modality weights for RQ3) |
| Cross-attention (finance as query, 4 heads) | secondary comparison |

### C.4.3 Training

| Item | Value |
| --- | --- |
| Optimiser | Adam |
| Learning rate | 1e-3 |
| Weight decay | 1e-4 |
| Dropout | 0.1 |
| Batch size | 32 |
| Maximum epochs | 80 |
| Early stopping | inner validation, patience 12 |
| Device | CPU |
| Seed | 42 (robustness: 1, 2) |

After early stopping, the checkpoint with the lowest inner-validation loss is
kept for the subsequent forecast block. The model is not refit on the combined
training and validation sample.
