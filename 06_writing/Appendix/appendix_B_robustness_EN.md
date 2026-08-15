# Appendix B — Extra results & robustness

> All checks share the single leakage-safe protocol (2019–2025, lookback 4,
> expanding rolling-origin, 257 common scored weeks). Every p-value below is a
> one-sided DM-HLN test on reconstructed-price squared error, as defined in
> Section 3.7.2; Clark–West appears for Ridge only. Ablation arms in B.3 are
> exploratory and sit outside the three frozen comparison families, so they carry
> raw p-values only. They **qualify** the main findings; they do not replace the
> main-analysis specs (§ Chapter 4). Sources are given per table so every number
> is traceable.

---

## B.1 Sub-period: early (≤2022) vs late (≥2023)

Split at 2023-01-01 (matching `run_deep_advanced.py`). Skill vs M0 in %.
Early/late scored offline with `subperiod_eval.py` →
`05_outputs/baselines/subperiod/subperiod_summary.csv`. Full-sample Deep fusion
skills in this table follow B.2 (fusion matrix).

| Model | full | early (≤2022) | late (≥2023) |
| --- | ---: | ---: | ---: |
| M0 (random walk) | 0 | 0 | 0 |
| M1_Flat Ridge / XGB | −2.52 / −5.22 | −2.85 / −4.51 | −1.98 / −6.37 |
| M2_Flat Ridge / XGB | −6.31 / −6.95 | −6.53 / −7.11 | −5.94 / −6.67 |
| M3_Flat Ridge / XGB | −7.11 / −6.17 | −7.91 / −6.94 | −5.80 / −4.91 |
| M4_Flat Ridge / XGB | −9.26 / −8.53 | −9.83 / −10.52 | −8.32 / −5.21 |
| M1_Deep | −2.36 | −1.33 | −4.02 |
| M_ship_GNN (shipping only) | −0.22 | −0.24 | −0.17 |
| M_rs_deep (RS only) | −2.30 | −3.07 | −1.04 |
| **M3_Deep_gated (main)** | **+0.15** | **+0.33** | **−0.13** |
| M2_Deep_gated | −2.43 | −3.06 | −1.41 |
| M4_Deep_gated | −0.67 | −1.36 | +0.47 |
| M4_Deep_Concat | −8.30 | −12.77 | −0.58 |

**Reading**: no flat model beats M0 in either sub-period. Among deep models,
M3_Deep_gated has both the largest full-sample skill (+0.15) and the strongest
early-period skill (+0.33), but is marginally negative late (−0.13);
M4_Deep_gated shows the opposite profile (−1.36 early, +0.47 late), and
**No deep configuration in this table
is positive in both sub-periods**, so the small full-sample gain of the main model
is not evenly distributed over time. The sub-period decomposition is run only on the
gated and concat specifications, so cross-attention is outside the scope of this
table; its instability is documented across seeds in B.4 instead. Gated finance+shipping is retained as the
main model on full-sample skill and on the nested shipping increment (Chapter 4);
this split is reported as a limitation on how stable that gain is, not as
supporting evidence.

---

## B.2 Deep fusion matrix (3 modality combos × 3 fusions)

seed 42, lookback 4, 257 weeks. Source: `run_deep_fusion_matrix.py` →
`05_outputs/baselines/Deep/_cross/deep_fusion_matrix.csv`. Skill vs M0 in %.

| Combo | Concat | Gated | Cross-Attn | DM p vs M0, best variant (raw / Holm) |
| --- | ---: | ---: | ---: | --- |
| **M3_Deep** (fin+ship) | −0.22 | **+0.15** | **+1.00** | xattn 0.257 / 1.000 |
| M2_Deep (fin+rs) | −2.01 | −2.43 | −5.87 | — |
| M4_Deep (fin+rs+ship) | −8.30 | −0.68 | +0.19 | xattn 0.427 / 1.000 |

**Reading**: on this seed, M0 is cleared only where shipping is present, and only
under adaptive fusion: M3 clears it under gated and cross-attention, M4 only under
cross-attention, and M2 (fin+rs) never. Plain concatenation clears M0 in no combo.
That pattern would invite the conclusion that the gain comes from weighting the
modalities rather than stacking them — but the conclusion does not survive
reseeding: averaged over seeds 42, 1 and 2, the M3 ordering inverts to concat
(−0.27%) > gated (−0.51%) > cross-attention (−3.01%), so the apparent premium on
adaptive weighting is a seed-42 artefact (B.4). None of the three positive skills
is statistically distinguishable from M0 either: the largest, cross-attention at
M3, has a raw one-sided DM p of 0.257, and every member of the 18-test benchmark
family has a Holm-adjusted p of 1.000. The positive entries in this table are
therefore descriptive orderings on one seed, not evidence of superiority over the
no-change forecast, and the fusion column should not be read as a ranking of
fusion mechanisms; B.4 gives the seed-averaged comparison.

---

## B.3 Flat robustness

### B.3.1 M2 leave-one-AOI-out (LOAO)

Source: `05_outputs/baselines/Flat/M2_Flat/baseline_loao_anom.csv`. Main M2
(anom) base RMSE: Ridge 4.4136, XGBoost 4.4402. ΔRMSE = dropped − full.

| Dropped AOI | Ridge RMSE | Δ | XGB RMSE | Δ |
| --- | ---: | ---: | ---: | ---: |
| (none / full) | 4.4136 | 0 | 4.4402 | 0 |
| Basra | 4.4237 | +0.010 | 4.4617 | +0.021 |
| Fujairah | 4.3748 | −0.039 | 4.4219 | −0.018 |
| Houston | 4.4079 | −0.006 | 4.3492 | −0.091 |
| Jamnagar | 4.4095 | −0.004 | 4.4023 | −0.038 |
| Jurong | 4.4487 | +0.035 | 4.4499 | +0.010 |
| Kharg | 4.3427 | −0.071 | 4.4269 | −0.013 |
| NingboZhoushan | 4.3973 | −0.016 | 4.3861 | −0.054 |
| RasTanura | 4.4059 | −0.008 | 4.4011 | −0.039 |
| Rotterdam | 4.4236 | +0.010 | 4.4458 | +0.006 |
| Ulsan | 4.4229 | +0.009 | 4.4104 | −0.030 |
| Yanbu | 4.3737 | −0.040 | 4.3926 | −0.048 |

Removing any single AOI leaves the result essentially unchanged: the largest
shift is 0.091 for XGBoost (Houston) and 0.071 for Ridge (Kharg), each about 2%
of RMSE. Dropping a site more often helps than hurts—seven of eleven sites for
Ridge and eight of eleven for XGBoost give a lower RMSE—so no individual
location carries the remote-sensing signal.

### B.3.2 M3 leave-one-channel-out (LOCHO)

seed 42, 257 weeks. Source:
`05_outputs/baselines/Flat/M3_Flat/robustness_m3_summary.csv`.
Clark–West is not reported for XGBoost (§3.7.2). M1 baselines: Ridge 4.2563,
XGB 4.3684.

| Arm | Ridge RMSE | skill % | CW p vs M1 | XGB RMSE | skill % | DM p vs M1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| full (113 cols, main) | 4.4471 | −7.11 | 0.297 | 4.4080 | −6.17 | 0.633 |
| core (38) | 4.3554 | −4.90 | 0.309 | 4.4207 | −6.48 | 0.727 |
| portwatch-only | 4.3471 | −4.70 | 0.127 | 4.3387 | −4.50 | 0.391 |
| gfw-only | 4.3831 | −5.57 | 0.354 | 4.3884 | −5.70 | 0.622 |
| gfw-presence | 4.3219 | −4.10 | 0.313 | 4.3590 | −4.99 | 0.450 |
| gfw-aggregate | **4.2438** | −2.22 | **0.032** | 4.3605 | −5.03 | 0.426 |
| tanker-only | 4.2857 | −3.23 | 0.111 | 4.3317 | −4.33 | 0.384 |

**Reading**: four arms—tanker-only, PortWatch-only, GFW-presence and
GFW-aggregate—do sit slightly below the finance-only M1 XGBoost baseline of
4.3684 in RMSE, so the descriptive ordering is not uniformly against shipping.
The evidence is nevertheless weak: the smallest one-sided DM p across the seven
arms is 0.384, no arm beats M0, and the main 113-column specification is among
the least favourable (p = 0.633). For Ridge, where Clark–West is admissible, only
the GFW-aggregate arm falls below 5% (CW p = 0.032), and that is also the only
Ridge arm whose RMSE (4.2438) sits below M1 Ridge (4.2563), so the supplementary
test and the descriptive ordering agree there. The M3 shipping signal is
therefore not established by a channel-level ablation.

### B.3.3 M2 water-masked RS variant

Source: `baseline_metrics_anom_watermask.csv`. Masking water pixels lowers XGB
RMSE slightly, from 4.4402 to 4.4136, and the one-sided DM p against M1 moves
from 0.822 to 0.687; M2 still does not beat M0 (skill −6.3%). De-noising the
remote-sensing inputs therefore changes neither the ranking nor the inference: RS
value is limited, which motivates modality-aware fusion (RQ2).

### B.3.4 M4 leave-one-modality-out (LOMO)

Source: `Flat/M4_Flat/robustness_m4_summary.csv`. Dropping RS from M4 (i.e. M1+M3)
lowers XGB RMSE from 4.5061 to 4.4080, and dropping shipping instead gives 4.4402;
the finance-only arm is still the most accurate at 4.3684. The one-sided DM p
against M1 is 0.847 for full M4, 0.633 without RS and 0.822 without shipping, so
no arm is distinguishable from the finance-only baseline. Each added modality
raises RMSE in flat concatenation, and no valid nested increment is detectable.

### B.3.5 Lookback sweep

M1 source: `Flat/M1_Flat/sweep_summary.csv`. L1/L8/L12 are the untuned
robustness grid (`retrain_every=1`); **L4_tuned** is the locked protocol.
M2–M4 sources: `sweep_m2_summary.csv`, `sweep_m3_summary.csv`,
`sweep_m4_summary.csv`, all on the locked L4_tuned protocol. Skill vs M0 in %.
n = 260 / 257 / 253 / 249 at L1 / L4 / L8 / L12; each row is scored against
its own M0.

**M1 lookback / tuning grid**

| Config | Ridge RMSE | skill % | XGB RMSE | skill % | n |
| --- | ---: | ---: | ---: | ---: | ---: |
| L1_all (untuned) | 4.818 | −16.5 | 4.686 | −13.3 | 260 |
| L4_all (untuned) | 5.909 | −42.3 | 4.694 | −13.1 | 257 |
| L8_all (untuned) | 6.561 | −57.3 | 4.747 | −13.8 | 253 |
| L12_all (untuned) | 6.901 | −65.1 | 4.840 | −15.8 | 249 |
| **L4_tuned (main)** | **4.256** | **−2.52** | **4.368** | **−5.22** | 257 |

**M2 / M3 / M4 lookback (tuned)**

| Set | L | Ridge RMSE | skill % | XGB RMSE | skill % | n |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| M2 anom | 1 | 4.230 | −2.25 | 4.510 | −9.03 | 260 |
| M2 anom | **4** | **4.414** | **−6.31** | **4.440** | **−6.95** | 257 |
| M2 anom | 8 | 4.526 | −8.49 | 4.668 | −11.90 | 253 |
| M3 | 1 | 4.306 | −4.09 | 4.546 | −9.90 | 260 |
| M3 | **4** | **4.447** | **−7.11** | **4.408** | **−6.17** | 257 |
| M3 | 8 | 4.549 | −9.04 | 4.541 | −8.86 | 253 |
| M4 anom | 1 | 4.369 | −5.62 | 4.581 | −10.75 | 260 |
| M4 anom | **4** | **4.536** | **−9.26** | **4.506** | **−8.53** | 257 |
| M4 anom | 8 | 4.577 | −9.70 | 4.551 | −9.10 | 253 |

No Flat lookback beats M0. Untuned M1 Ridge deteriorates sharply as the window
lengthens. Under the locked protocol, L4 remains the least-bad Flat cell at
every information set.

### B.3.6 Stationarised M1 (`feature-mode = returns`)

Source: `sweep_summary.csv`, lookback 4. The locked main row is `L4_tuned`.

| Config | Ridge RMSE | skill % | XGB RMSE | skill % |
| --- | ---: | ---: | ---: | ---: |
| L4_all (untuned) | 5.909 | −42.3 | 4.694 | −13.1 |
| L4_returns (untuned) | 6.225 | −49.9 | 4.522 | −8.91 |
| **L4_tuned (main)** | **4.256** | **−2.52** | **4.368** | **−5.22** |
| L4_returns_tuned | 4.297 | −3.50 | 4.430 | −6.69 |

Stationarising M1 does not beat M0 and does not improve on the locked `all`
tuned specification.

### B.3.7 C2 dimensionality reduction (M2)

Source: `Flat/M2_Flat/c2_summary.csv`. Same L4 window, but hyperparameters are
**fixed** (Ridge α=1000, XGB depth=2 / 200 trees) so the comparison isolates
the reducer. The M1 RMSE in this table therefore differs slightly from Table 4.1.

| Arm | Ridge M2 RMSE | skill % | XGB M2 RMSE | skill % |
| --- | ---: | ---: | ---: | ---: |
| all-55 | 4.4136 | −6.31 | 4.4471 | −7.11 |
| pca-90 | 4.4164 | −6.37 | 4.4409 | −6.96 |
| elastic | 4.4136 | −6.31 | 4.4471 | −7.11 |
| shap-top20 | 4.3655 | −5.15 | 4.4750 | −7.78 |

PCA and ElasticNet leave Ridge unchanged to reporting precision; SHAP-top20
lowers Ridge RMSE slightly (4.414 → 4.366) but it remains above M1. No arm beats
M0. The weak M2 increment is therefore not an artefact of using all 55 anomaly
columns.

### B.3.8 Publication-lag sweep

A.3.4. GFW monthly presence ∈ {1, 4, 8} weeks on Flat M3; `MONTHLY_LAG_WEEKS`
∈ {3, 5, 7} on Flat M1. Locked values are GFW **+4** and monthly **+5**. Extra
lags are a further calendar shift on the already-lagged merged columns.
Source: `05_outputs/_experiments/lag_robustness/lag_robustness_summary.csv`.
n = 257 throughout.

**GFW monthly presence (Flat M3)**

| Lag (w) | Ridge RMSE | skill % | XGB RMSE | skill % | XGB DM p vs M1 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 4.407 | −6.15 | 4.801 | −15.63 | 0.972 |
| **4 (locked)** | **4.447** | **−7.11** | **4.408** | **−6.17** | **0.633** |
| 8 | 4.334 | −4.38 | 4.396 | −5.88 | 0.603 |

**MONTHLY_LAG_WEEKS (Flat M1: REA + non-oil commodity)**

| Lag (w) | Ridge RMSE | skill % | XGB RMSE | skill % |
| ---: | ---: | ---: | ---: | ---: |
| 3 | 4.255 | −2.49 | 4.399 | −5.95 |
| **5 (locked)** | **4.256** | **−2.52** | **4.368** | **−5.22** |
| 7 | 4.245 | −2.23 | 4.388 | −5.68 |

No lag beats M0. Shortening GFW to +1w makes XGBoost substantially worse
(4.801). Lengthening GFW to +8w trims Ridge RMSE slightly (4.334) but skill
stays negative. Monthly-macro lag moves M1 by at most 0.03 RMSE. The locked
buffers are therefore not the reason Flat alternative data fail to clear M0.

---

## B.4 Deep multi-seed & sweeps

Source: `05_outputs/baselines/Deep/_cross/deep_seed_summary.csv` (and
`deep_seed_pooled.csv`), pooled by `pool_deep_seeds.py` on the fusion-matrix
protocol: seeds 42, 1, 2; lookback 4; d 32; epochs 80. The seed-42 column is the
same run as B.2.

| Config | seed 42 | seed 1 | seed 2 | mean ± sd | positive |
| --- | ---: | ---: | ---: | --- | ---: |
| S3 concat | −0.22 | −0.64 | +0.07 | **−0.27% ± 0.35** | 1/3 |
| **S3 gated** (main) | +0.15 | −1.39 | −0.29 | −0.51% ± 0.80 | 1/3 |
| S4 gated | −0.68 | −0.86 | −1.19 | −0.91% ± 0.26 | 0/3 |
| S4 xattn | +0.19 | −0.87 | −5.01 | −1.90% ± 2.75 | 1/3 |
| S3 xattn | **+1.00** | −2.87 | −7.14 | −3.01% ± **4.07** | 1/3 |

The M2 combos and S4 concat are reseeded nowhere and are excluded from this
table: a single run is not a multi-seed mean.

**Every mean is below M0.** The positive headline figures in Chapter 4 and B.2
(S3 gated +0.15%, S3 cross-attention +1.00%) are seed-42 outcomes, not expected skill:
averaged over seeds, no Deep configuration beats the no-change benchmark, and each
S3 fusion is positive in exactly one of three seeds. This is the sharpest single
limitation on the Deep results and is carried into Chapter 5.

Within the S3 block the seed-42 ranking is **exactly reversed** by reseeding.
On seed 42 the order is xattn (+1.00) > gated (+0.15) > concat (−0.22); across
seeds it is concat (−0.27) > gated (−0.51) > xattn (−3.01), and dispersion widens
in the same order (0.35 < 0.80 < 4.07). The more adaptive the fusion, the higher
its single-seed ceiling and the worse its seed-averaged skill — which is what
makes single-seed selection actively misleading here, since it systematically
picks the highest-variance operator. Cross-attention is excluded on that basis: its
mean is an order of magnitude worse and its spread an order of magnitude wider than
either alternative. Concat and gated, by contrast, are **not** separable — the
0.24-point gap between their means is smaller than the cross-seed sd of either, so
concat's nominally better mean is not evidence that it forecasts better. Gated is
retained as the main specification because it is the only one of the two that
exposes modality gates, which are the object of the RQ3 analysis (Section 4.6);
that is a design requirement, not a claim that it is the more accurate operator.

### B.4.1 Hyper-parameter grid

Single seed=42, M3 gated, epochs 80. Source: `deep_sweep_summary.csv`,
`group=hyper`. Locked cell: lookback 4, d 32, 2+2 layers.

| lookback | d | GAT+TCN layers | RMSE | skill % | n |
| ---: | ---: | --- | ---: | ---: | ---: |
| **4** | **32** | **2+2 (main)** | **4.1456** | **+0.15** | 257 |
| 4 | 64 | 2+2 | 4.1831 | −0.75 | 257 |
| 8 | 32 | 2+2 | 4.1612 | +0.25 | 253 |
| 8 | 64 | 2+2 | 4.2161 | −1.06 | 253 |
| 12 | 32 | 2+2 | 4.1833 | −0.06 | 249 |
| 12 | 64 | 2+2 | 4.2259 | −1.08 | 249 |
| 4 | 32 | 1+1 | 4.2042 | −1.26 | 257 |

lb 8 / d 32 is nominally best (+0.25%). **d 64 is always worse.** Halving depth
also degrades skill. The main model stays locked at lookback 4, d 32 for Flat
protocol parity.

### B.4.2 Remote-sensing branch

`m_rs_deep` only, seed 42. Source: `group=rs`. Default: lr=1e-3, wd=1e-4,
dropout=0.1, meanpool.

| Variant | lr | wd | dropout | RMSE | skill % |
| --- | ---: | ---: | ---: | ---: | ---: |
| **meanpool (default)** | 1e-3 | 1e-4 | 0.1 | 4.247 | −2.30 |
| cls token | 1e-3 | 1e-4 | 0.1 | 4.610 | −11.03 |
| meanpool | 1e-3 | 1e-4 | 0.3 | 4.191 | −0.94 |
| meanpool | 1e-3 | 1e-3 | 0.1 | 4.249 | −2.33 |
| meanpool | 1e-3 | 1e-3 | 0.3 | 4.189 | −0.90 |
| meanpool | 3e-4 | 1e-4 | 0.1 | 4.228 | −1.85 |
| meanpool | 3e-4 | 1e-4 | 0.3 | 4.190 | −0.93 |
| meanpool | 3e-4 | 1e-3 | 0.1 | 4.220 | −1.64 |
| meanpool | 3e-4 | 1e-3 | 0.3 | 4.192 | −0.97 |

cls is markedly worse. Tuning at most lifts meanpool from −2.30% to about
−0.90%; **every cell remains negative.**

### B.4.3 Regularisation grid (M3 gated)

Default cell repeated from B.4.1. Source: `group=reg`.

| lr | wd | dropout | RMSE | skill % |
| ---: | ---: | ---: | ---: | ---: |
| **1e-3** | **1e-4** | **0.1 (main)** | **4.1456** | **+0.15** |
| 1e-3 | 1e-4 | 0.3 | 4.1458 | +0.15 |
| 1e-3 | 1e-3 | 0.1 | 4.1453 | +0.16 |
| 1e-3 | 1e-3 | 0.3 | 4.1431 | +0.21 |
| 3e-4 | 1e-4 | 0.1 | 4.1537 | −0.05 |
| 3e-4 | 1e-4 | 0.3 | 4.1441 | +0.18 |
| 3e-4 | 1e-3 | 0.1 | 4.1569 | −0.12 |
| 3e-4 | 1e-3 | 0.3 | 4.1434 | +0.20 |

Skill stays near zero (about −0.12% to +0.21%). Regularisation does not move the
headline result off the M0 line.

### B.4.4 Modality dropout and longer window

Source: `Deep/M4_Deep/deep_advanced_summary.csv`, seed 42. The mt78 row is scored
on 283 weeks; compare skill, not RMSE, with the 257-week tables.

| Arm | n | RMSE | skill vs M0 |
| --- | ---: | ---: | ---: |
| M4 gated (main window) | 257 | 4.180 | −0.67% |
| M4 gated, dropout 0.3 | 257 | 4.159 | −0.17% |
| M4 xattn, dropout 0.3 | 257 | 4.126 | +0.63% |
| M4 gated, min_train=78 | 283 | 4.050 | −0.76% |

ModDrop slightly improves gated M4 but does not clear M0. The longer warm-up
window remains negative versus M0. Main protocol keeps `modality_dropout=0` and
`min_train=104`.

---

## B.5 Other robustness (not tabulated)

These checks were run; they do not change the Chapter 4 ranking and are not
given their own tables.

- **M2 literature / aoi4 / ntlall.** Sparse RS contracts. At L4, XGB RMSE is
  4.322 / 4.340 / 4.345 against full anom 4.440. Cutting columns lowers RMSE
  slightly; **none beats M0.**
- **M2 level.** Raw index levels collapse to the M1 RMSE exactly (Ridge 4.256,
  XGB 4.368).
- **Cross-attention sub-periods.** B.1 is gated/concat only; xattn instability
  is the multi-seed record in B.4.
- **M2 three fusions and S4 concat have `n_seeds = 1`.** Excluded from the B.4
  mean table; a single run is not a multi-seed mean.
- **SHAP** (M2/M3/M4) is interpretability, not a robustness test.

---

## B.6 Frozen comparison families

The three families defined in Section 3.7.2 are listed here in full. Membership is
fixed by the research questions before any p-value is inspected, and Holm's
adjustment is applied within each family separately. Every row is a DM-HLN test on
squared errors of the reconstructed price over the same 257 forecast origins.
Source: `05_outputs/tests/test_table_main.csv`, generated by
`04_code/scripts/tools/build_test_tables.py`. Skill is the percentage RMSE
reduction of the candidate relative to the reference, so a positive value means
the candidate is the more accurate forecast.

### B.6.1 Benchmark family (18 comparisons)

| # | Reference | Candidate | Direction | RMSE ref → cand | Skill % | Raw p | Holm p |
| ---: | --- | --- | --- | --- | ---: | ---: | ---: |
| 1 | M0 | Ridge S1 | one-sided | 4.1518 → 4.2563 | −2.52 | 0.9429 | 1.0000 |
| 2 | M0 | Ridge S2 | one-sided | 4.1518 → 4.4136 | −6.31 | 0.9801 | 1.0000 |
| 3 | M0 | Ridge S3 | one-sided | 4.1518 → 4.4471 | −7.11 | 0.9471 | 1.0000 |
| 4 | M0 | Ridge S4 | one-sided | 4.1518 → 4.5361 | −9.26 | 0.9787 | 1.0000 |
| 5 | M0 | XGB S1 | one-sided | 4.1518 → 4.3684 | −5.22 | 0.9825 | 1.0000 |
| 6 | M0 | XGB S2 | one-sided | 4.1518 → 4.4402 | −6.95 | 0.9965 | 1.0000 |
| 7 | M0 | XGB S3 | one-sided | 4.1518 → 4.4080 | −6.17 | 0.9904 | 1.0000 |
| 8 | M0 | XGB S4 | one-sided | 4.1518 → 4.5061 | −8.53 | 0.9942 | 1.0000 |
| 9 | M0 | Deep S1 | one-sided | 4.1518 → 4.2499 | −2.36 | 0.9773 | 1.0000 |
| 10 | M0 | Deep gated S2 | one-sided | 4.1518 → 4.2528 | −2.43 | 0.9483 | 1.0000 |
| 11 | M0 | Deep gated S3 | one-sided | 4.1518 → 4.1456 | **+0.15** | 0.4266 | 1.0000 |
| 12 | M0 | Deep gated S4 | one-sided | 4.1518 → 4.1801 | −0.68 | 0.9492 | 1.0000 |
| 13 | M0 | Deep concat S2 | one-sided | 4.1518 → 4.2352 | −2.01 | 0.9983 | 1.0000 |
| 14 | M0 | Deep concat S3 | one-sided | 4.1518 → 4.1611 | −0.22 | 0.6461 | 1.0000 |
| 15 | M0 | Deep concat S4 | one-sided | 4.1518 → 4.4964 | −8.30 | 0.9228 | 1.0000 |
| 16 | M0 | Deep xattn S2 | one-sided | 4.1518 → 4.3956 | −5.87 | 0.9990 | 1.0000 |
| 17 | M0 | Deep xattn S3 | one-sided | 4.1518 → 4.1102 | **+1.00** | 0.2572 | 1.0000 |
| 18 | M0 | Deep xattn S4 | one-sided | 4.1518 → 4.1437 | **+0.19** | 0.4273 | 1.0000 |

Only three of the eighteen specifications reduce RMSE relative to the no-change
forecast, all of them Deep and all of them containing shipping. The smallest raw
p in the family is 0.257, so none approaches significance even before adjustment,
and every Holm-adjusted p equals 1.000.

### B.6.2 RQ1 family (15 comparisons)

| # | Reference | Candidate | Direction | RMSE ref → cand | Skill % | Raw p | Holm p |
| ---: | --- | --- | --- | --- | ---: | ---: | ---: |
| 1 | Ridge S1 | Ridge S2 | one-sided | 4.2563 → 4.4136 | −3.69 | 0.9561 | 1.0000 |
| 2 | Ridge S1 | Ridge S3 | one-sided | 4.2563 → 4.4471 | −4.48 | 0.8474 | 1.0000 |
| 3 | Ridge S1 | Ridge S4 | one-sided | 4.2563 → 4.5361 | −6.57 | 0.9383 | 1.0000 |
| 4 | Ridge S2 | Ridge S4 | one-sided | 4.4136 → 4.5361 | −2.77 | 0.8302 | 1.0000 |
| 5 | Ridge S3 | Ridge S4 | one-sided | 4.4471 → 4.5361 | −2.00 | 0.9212 | 1.0000 |
| 6 | XGB S1 | XGB S2 | one-sided | 4.3684 → 4.4402 | −1.64 | 0.8215 | 1.0000 |
| 7 | XGB S1 | XGB S3 | one-sided | 4.3684 → 4.4080 | −0.91 | 0.6326 | 1.0000 |
| 8 | XGB S1 | XGB S4 | one-sided | 4.3684 → 4.5061 | −3.15 | 0.8465 | 1.0000 |
| 9 | XGB S2 | XGB S4 | one-sided | 4.4402 → 4.5061 | −1.48 | 0.7360 | 1.0000 |
| 10 | XGB S3 | XGB S4 | one-sided | 4.4080 → 4.5061 | −2.22 | 0.7910 | 1.0000 |
| 11 | Deep S1 | Deep gated S2 | one-sided | 4.2499 → 4.2528 | −0.07 | 0.5213 | 1.0000 |
| 12 | Deep S1 | Deep gated S3 | one-sided | 4.2499 → 4.1456 | **+2.45** | **0.0410** | 0.6150 |
| 13 | Deep S1 | Deep gated S4 | one-sided | 4.2499 → 4.1801 | **+1.64** | 0.0682 | 0.9553 |
| 14 | Deep gated S2 | Deep gated S4 | one-sided | 4.2528 → 4.1801 | **+1.71** | 0.1094 | 1.0000 |
| 15 | Deep gated S3 | Deep gated S4 | one-sided | 4.1456 → 4.1801 | −0.83 | 0.8368 | 1.0000 |

Adding a modality never improves a Flat learner: all ten Ridge and XGBoost rows
have negative skill. The three positive rows are all Deep, and the shipping
increment from Deep S1 to gated S3 is the only one with a raw p below 5%
(0.0410). It does not survive adjustment within the fifteen-test family
(Holm p = 0.615), so it is reported as nominal evidence.

### B.6.3 RQ2 family (14 comparisons)

| # | Reference | Candidate | Direction | RMSE ref → cand | Skill % | Raw p | Holm p |
| ---: | --- | --- | --- | --- | ---: | ---: | ---: |
| 1 | Ridge S1 | Deep S1 | one-sided | 4.2563 → 4.2499 | +0.15 | 0.4663 | 1.0000 |
| 2 | XGB S1 | Deep S1 | one-sided | 4.3684 → 4.2499 | +2.71 | 0.0965 | 0.7648 |
| 3 | Ridge S2 | Deep gated S2 | one-sided | 4.4136 → 4.2528 | +3.64 | 0.0956 | 0.7648 |
| 4 | XGB S2 | Deep gated S2 | one-sided | 4.4402 → 4.2528 | +4.22 | **0.0423** | 0.4233 |
| 5 | Ridge S3 | Deep gated S3 | one-sided | 4.4471 → 4.1456 | +6.78 | 0.0640 | 0.5757 |
| 6 | XGB S3 | Deep gated S3 | one-sided | 4.4080 → 4.1456 | +5.95 | **0.0099** | 0.1322 |
| 7 | Ridge S4 | Deep gated S4 | one-sided | 4.5361 → 4.1801 | +7.85 | **0.0288** | 0.3173 |
| 8 | XGB S4 | Deep gated S4 | one-sided | 4.5061 → 4.1801 | +7.23 | **0.0094** | 0.1322 |
| 9 | Deep gated S2 | Deep concat S2 | two-sided | 4.2528 → 4.2352 | +0.41 | 0.7742 | 1.0000 |
| 10 | Deep gated S2 | Deep xattn S2 | two-sided | 4.2528 → 4.3956 | −3.36 | **0.0259** | 0.3113 |
| 11 | Deep gated S3 | Deep concat S3 | two-sided | 4.1456 → 4.1611 | −0.37 | 0.5884 | 1.0000 |
| 12 | Deep gated S3 | Deep xattn S3 | two-sided | 4.1456 → 4.1102 | +0.85 | 0.6231 | 1.0000 |
| 13 | Deep gated S4 | Deep concat S4 | two-sided | 4.1801 → 4.4964 | −7.57 | 0.1874 | 1.0000 |
| 14 | Deep gated S4 | Deep xattn S4 | two-sided | 4.1801 → 4.1437 | +0.87 | 0.4353 | 1.0000 |

All eight matched Deep–Flat rows favour Deep, and the margin widens as the
information set grows, from +0.15% at S1 against Ridge to +7.85% at S4. Four have
raw p below 5%, but the smallest Holm-adjusted p in the family is 0.132, so the
primary test supports no formal claim that the Deep pathway outperforms the Flat
pathway. Among the fusion mechanisms, no gated–concat or gated–cross-attention
contrast is distinguishable after adjustment; the strongest nominal contrast is
the gated-over-cross-attention margin at S2 (−3.36%, raw p = 0.026, Holm p =
0.311), while the largest raw gap, gated over concat at S4 (−7.57%), is far from
significant at raw p = 0.187.

### B.6.4 Supplementary Clark–West tests (Ridge only)

Clark–West presumes that the smaller specification is a parameter restriction of
the larger one. Only the Ridge information-set extensions come close to that
structure, and even there the penalty is re-tuned at every re-estimation, so these
five rows are supplementary rather than primary. They are reported for
completeness and are not used for any formal claim. Source:
`05_outputs/tests/test_table_cw_supplementary.csv`.

| # | Reference | Candidate | RMSE ref → cand | CW statistic | Raw p | Holm p |
| ---: | --- | --- | --- | ---: | ---: | ---: |
| 1 | Ridge S1 | Ridge S2 | 4.2563 → 4.4136 | 0.066 | 0.4738 | 1.0000 |
| 2 | Ridge S1 | Ridge S3 | 4.2563 → 4.4471 | 0.532 | 0.2975 | 1.0000 |
| 3 | Ridge S1 | Ridge S4 | 4.2563 → 4.5361 | 0.386 | 0.3499 | 1.0000 |
| 4 | Ridge S2 | Ridge S4 | 4.4136 → 4.5361 | 0.594 | 0.2763 | 1.0000 |
| 5 | Ridge S3 | Ridge S4 | 4.4471 → 4.5361 | 0.350 | 0.3630 | 1.0000 |

No Ridge extension is significant even before adjustment, and every RMSE moves in
the wrong direction, so the supplementary test agrees with the primary one here.
This is the contrast that motivates restricting Clark–West to Ridge: applied to
XGBoost, the same adjustment term produced apparent significance for arms whose
RMSE was worse than the baseline (B.3.2).

---

## B.7 Sensitivity analyses

Each check re-runs the same three frozen families and applies Holm within
(check × family), so the adjustment mirrors the main table and is never pooled
across checks. These analyses are exploratory: they qualify the primary result
but do not replace it. Source: `05_outputs/tests/test_table_robustness.csv`.

| Check | Comparisons | Holm p < 0.05 |
| --- | ---: | ---: |
| Absolute-error loss | 47 | 3 |
| Early sub-period ≤2022 | 47 | 0 |
| Late sub-period ≥2023 | 47 | 1 |
| Two-sided sensitivity | 41 | 1 |
| One-sided sensitivity | 6 | 0 |

### B.7.1 Absolute-error loss

Replacing squared error with absolute error changes the RQ2 conclusion. Three
matched Deep–Flat comparisons survive Holm within the fourteen-test family:

| Comparison | RMSE change | MAE change | Raw p | Holm p |
| --- | ---: | ---: | ---: | ---: |
| XGB S3 → Deep gated S3 | +5.95% | +8.26% | 0.0014 | **0.0171** |
| Ridge S4 → Deep gated S4 | +7.85% | +8.80% | 0.0011 | **0.0140** |
| XGB S4 → Deep gated S4 | +7.23% | +9.62% | 0.0010 | **0.0139** |

The primary test is defined on squared error because that is the loss the headline
RMSE metric reports, so the formal RQ2 conclusion remains the null result of
B.6.3. The divergence is informative rather than contradictory: squared error is
dominated by a small number of large-move weeks, whereas absolute error weights
every week equally. The Deep advantage is therefore concentrated in typical weeks
and does not extend to the weeks that drive RMSE. No benchmark or RQ1 comparison
survives under this loss.

### B.7.2 Early and late sub-periods

Splitting at the end of 2022 gives 102 early and 155 late origins. No comparison
survives Holm in the early sub-period, where the RQ2 minimum is 0.387. One
survives in the late sub-period: Ridge S4 → Deep gated S4, +8.09% RMSE and +9.86%
MAE, raw p = 0.0027 and Holm p = 0.038. A single surviving comparison in one half
of the sample, with the same contrast absent in the other half, is weak evidence;
it is consistent with the sub-period instability already documented in B.1 rather
than with a stable Deep advantage.

### B.7.3 Two-sided sensitivity

Section 3.7.2 uses one-sided tests where the research question is directional. To
show how much the conclusions depend on that choice, every one-sided comparison is
re-run two-sided, and the six two-sided fusion comparisons are re-run one-sided.
The RQ2 conclusion is unchanged: its smallest two-sided Holm p is 0.151, against
0.132 under the primary one-sided test, so neither version supports a formal
claim. One benchmark comparison becomes significant, and its direction matters:
M0 versus Deep xattn S2 gives Holm p = 0.038 with RMSE −5.87% and MAE −7.11%,
that is, cross-attention on finance plus remote sensing is detectably **worse**
than the no-change forecast. The two-sided variant can reject in either
direction, and here it rejects against the model.
