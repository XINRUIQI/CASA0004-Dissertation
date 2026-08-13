# Appendix C — Hyperparameter grids & locked settings

> This appendix pins every value needed to reproduce the reported results.
> §C.1 lists the software environment; §C.2 the shared leakage-safe protocol;
> §C.3 the Flat search grids; §C.4 the locked Deep architecture and training;
> §C.5 the entry-point scripts and output paths.

---

## C.1 Software environment

Reproduced with **Python 3.9.6** (CPython, macOS). Full pinned list in
`04_code/requirements.txt`; core packages:

| Package | Version | Role |
| --- | --- | --- |
| numpy | 2.0.2 | arrays |
| pandas | 2.3.3 | weekly matrix, joins |
| scipy | 1.13.1 | DM / Clark-West test statistics |
| scikit-learn | 1.6.1 | Ridge, pipelines, VarianceThreshold, StandardScaler |
| xgboost | 2.1.4 | Flat XGBoost learner |
| torch | 2.8.0 | Deep encoders + fusion (CPU) |
| matplotlib | 3.9.4 | figures |
| shap | 0.49.1 | Flat feature attribution |

> **On foundation models**: the Deep RS branch consumes
> **pre-computed frozen Prithvi-EO-2.0 embeddings** stored on disk; training and
> evaluation do **not** import `transformers` or fetch any model online (the test
> machine has no `transformers` installed). The one-off embedding-export step is
> separate and not part of this environment.

Install:

```bash
python3 -m pip install -r 04_code/requirements.txt
```

---

## C.2 Shared leakage-safe protocol (Flat = Deep)

Both families use the identical rolling-origin schedule so architecture
differences are not confounded with protocol differences.

| Item | Value |
| --- | --- |
| Comparison window | 2019–2025 (365 weeks in the merged matrix) |
| Lookback | **4 weeks** |
| Warm-up `min_train` | **104 weeks** (not scored) |
| Refit cadence `retrain_every` | **13 weeks** |
| Inner-validation `val_weeks` | **52 weeks** (tail of each training fold) |
| Common scored test span | **257 weeks** (2021-01 → 2025-12) |
| Target | one-week log return \(r_{t+1}\), reconstructed to price |
| Primary metric | RMSE + MAE on reconstructed price; skill vs M0 |
| Primary test | Diebold–Mariano with HLN small-sample correction, on reconstructed-price squared error, for **every** formal comparison (vs M0, vs S1, and Flat vs Deep) |
| Test direction | one-sided where the research question is directional (vs M0, vs S1, Deep vs Flat); two-sided for Deep fusion-mechanism comparisons |
| Multiplicity | Holm within three frozen families: benchmark (18), RQ1 (15), RQ2 (14); raw and adjusted p both reported, formal claims on adjusted |
| Supplementary test | Clark–West, **Ridge only** (5 comparisons), never for XGBoost or Deep |
| Unified test table | `05_outputs/tests/test_table_main.csv` via `04_code/scripts/tools/build_test_tables.py` |
| Seed | **42** (main); 1, 2 for robustness |

**Variance estimation in the DM statistic.** The loss differential is
\(d_t = L_{\text{reference},t} - L_{\text{candidate},t}\), so a positive statistic
means the candidate is the more accurate forecast. Its long-run variance is
estimated with the usual truncation at \(h-1\) autocovariances, where \(h\) is the
forecast horizon. Every comparison in this study is a one-week-ahead forecast, so
\(h = 1\) and no autocovariance term enters: the variance reduces to
\(\hat{\gamma}_0 / T\), where \(\hat{\gamma}_0 = T^{-1}\sum_t (d_t - \bar{d})^2\).
The Harvey–Leybourne–Newbold finite-sample factor
\(\sqrt{[T + 1 - 2h + h(h-1)/T]\,/\,T}\) is then applied, which at \(h = 1\)
equals \(\sqrt{(T-1)/T}\), and the statistic is referred to a \(t\) distribution
with \(T-1\) degrees of freedom. Implementation:
`04_code/src/backtest/metrics.py::dm_test`.

---

## C.3 Flat search grids

Chosen inside each training fold on the past `val_weeks` slice only
(`backtest/rolling.py::tune_hyperparams`); no test-set peeking. Every pipeline
begins with `VarianceThreshold(0.0)` (drops in-fold constant columns), and the
linear model adds `StandardScaler` (fit on the training fold only). Source:
`04_code/src/backtest/models.py`.

| Learner | Grid |
| --- | --- |
| Ridge (α) | {0.1, 1.0, 10.0, 100.0, 1000.0} (default 10.0) |
| XGBoost `max_depth` | {2, 3} |
| XGBoost `learning_rate` | {0.03, 0.05} |
| XGBoost `n_estimators` | {200, 400} |
| XGBoost fixed | `subsample=0.8`, `colsample_bytree=0.8`, `reg_lambda=1.0` |

---

## C.4 Locked Deep architecture & training

Sweeps explored lookback ∈ {4, 8, 12}, representation dim ∈ {32, 64}, fusion ∈
{concat, gated, cross-attention}, dropout / weight-decay, and seeds; the main
specification is **locked to lookback = 4 and d = 32** for protocol parity with
the flat baselines. Sensitivity is reported in Appendix B. Sources:
`deep_rolling.py`, `finance_encoder.py`, `rs_encoder.py`, `shipping_encoder.py`,
`fusion.py`.

### C.4.1 Encoders

| Encoder | Key settings | Output |
| --- | --- | --- |
| Finance TCN | `d_model=32`, `tcn_layers=2`, `kernel=3`, causal, `dropout=0.1` | z_fin, **32-d** |
| RS (frozen Prithvi) | `emb_dim=1024`, `n_sites=11`, `d_model=64`; temporal-attention + AOI-site-attention pooling | z_rs, **32-d** |
| Shipping graph | 17 nodes (11 AOI + 6 chokepoints); type-specific projection + node-type embedding; **GAT layers = 2, heads = 4**; `log1p(O-D flow)` as attention prior (learned `edge_scale`); adjacency symmetrised + self-looped; then **TCN layers = 2**; node-attention pooling; `d_model=64` | z_ship, **32-d**; ≈ 42k params |

### C.4.2 Fusion

| Option | Role |
| --- | --- |
| encoder-concat | fusion-ladder floor |
| **gated** (softmax gate over modality embeddings) | **main reported model** (gate weights also feed RQ3) |
| cross-attention (finance as query, `n_heads=4`) | comparative (single-seed best but higher variance) |

### C.4.3 Training

| Item | Value |
| --- | --- |
| Optimiser | Adam |
| Learning rate | `1e-3` |
| Weight decay | `1e-4` |
| Dropout | `0.1` |
| Batch size | `32` |
| Epochs (max) | `80` |
| Early stopping | on last `val_weeks` of fold, `patience=12` |
| Modality dropout | `0.0` main (`0.3` robustness arm) |
| Device | CPU |
| Seed | `42` (robustness: 1, 2) |

After early stopping, the weights from the epoch with the lowest inner-validation
loss are restored and used for the subsequent forecast block. The model is not
refit on the inner-validation weeks.

---

## C.5 Entry points & outputs

| Purpose | Script | Output dir |
| --- | --- | --- |
| Flat M0–M4 baselines | `04_code/scripts/flat/run_baseline.py` (+ `flat/M{1..4}_Flat/*.py`) | `05_outputs/baselines/Flat/M*_Flat/` |
| Deep baselines & fusion | `04_code/scripts/deep/run_deep_baseline.py` | `05_outputs/baselines/Deep/{M*_Deep,_cross}/` |
| Deep sweeps (seed/lookback/dim/reg) | `04_code/scripts/deep/run_deep_sweep.py` | `05_outputs/baselines/Deep/_cross/deep_sweep_summary.csv` |
| Fusion matrix (3×3) | `run_deep_fusion_matrix.py` | `deep_fusion_matrix.{csv,png}` |
| Advanced ablations (fusion/dropout/sub-period) | `run_deep_advanced.py` | `deep_advanced_summary.csv` |
| Sub-period early/late (Flat + Deep, offline) | `subperiod_eval.py` | `05_outputs/baselines/subperiod/subperiod_summary.csv` |
| **Frozen comparison families + Holm** | `04_code/scripts/tools/build_test_tables.py` | `05_outputs/tests/test_table_{main,cw_supplementary,robustness}.csv` |
| Interpretability (gates, attention) | `run_deep_interpret.py`, `run_deep_interpret_m3.py`, `run_deep_xattn_viz.py` | `deep_interpret*.png`, `deep_*gate*.csv` |
| Feature matrix build | `03_data/processed/**/build_*.py`, `merge/py/build_feature_matrix.py` | `03_data/processed/merge/outputs/` |

Reproduce end-to-end:

```bash
python3 -m pip install -r 04_code/requirements.txt
python3 04_code/scripts/flat/run_baseline.py --modality M3      # flat example
python3 04_code/scripts/deep/run_deep_baseline.py               # deep main
python3 04_code/scripts/tools/subperiod_eval.py                  # early/late table
python3 04_code/scripts/tools/build_test_tables.py               # all reported p-values
```
