# Appendix B — Extra results & robustness

> All checks share the single leakage-safe protocol (2019–2025, lookback 4,
> expanding rolling-origin, 257 common scored weeks, CW vs M1 / DM vs M0). They
> **reinforce** the main findings; they do not replace the main-analysis specs
> (§ Chapter 4). Sources are given per table so every number is traceable.

---

## B.1 Sub-period: early (≤2022) vs late (≥2023)

Split at 2023-01-01 (matching `run_deep_advanced.py`). Skill vs M0 in %.
Scored offline from saved predictions with `subperiod_eval.py` →
`05_outputs/baselines/subperiod/subperiod_summary.csv`; numbers reproduce the
main pipeline exactly.

| Model | full | early (≤2022) | late (≥2023) |
| --- | ---: | ---: | ---: |
| M0 (random walk) | 0 | 0 | 0 |
| M1_Flat Ridge / XGB | −2.52 / −5.22 | −2.85 / −4.51 | −1.98 / −6.37 |
| M2_Flat Ridge / XGB | −6.31 / −6.95 | −6.53 / −7.11 | −5.94 / −6.67 |
| M3_Flat Ridge / XGB | −6.71 / −6.68 | −7.22 / −6.68 | −5.87 / −6.69 |
| M4_Flat Ridge / XGB | −8.99 / −8.57 | −9.33 / −9.07 | −8.45 / −7.74 |
| M1_Deep | −2.36 | −1.33 | −4.02 |
| M_ship_GNN (shipping only) | −0.22 | −0.24 | −0.17 |
| M_rs_deep (RS only) | −2.30 | −3.07 | −1.04 |
| **M3_Deep_gated (main)** | **+0.16** | **+0.33** | **−0.13** |
| M2_Deep_gated | −2.43 | −3.06 | −1.41 |
| M4_Deep_gated | −0.67 | −1.36 | +0.47 |
| M4_Deep_Concat | −8.30 | −12.77 | −0.58 |

**Reading**: no flat model beats M0 in either sub-period. Among deep models,
M3_Deep_gated has both the largest full-sample skill (+0.16) and the strongest
early-period skill (+0.33), but is marginally negative late (−0.13);
M4_Deep_gated shows the opposite profile (−1.36 early, +0.47 late), and
cross-attention–driven M4 gains remain concentrated late. **No deep configuration
is positive in both sub-periods**, so the small full-sample gain of the main model
is not evenly distributed over time. Gated finance+shipping is retained as the
main model on full-sample skill and on the nested shipping increment (Chapter 4);
this split is reported as a limitation on how stable that gain is, not as
supporting evidence.

---

## B.2 Deep fusion matrix (3 modality combos × 3 fusions)

seed 42, lookback 4, 257 weeks. Source: `run_deep_fusion_matrix.py` →
`05_outputs/baselines/Deep/_cross/deep_fusion_matrix.csv`. Skill vs M0 in %.

| Combo | Concat | Gated | Cross-Attn | CW p vs M0 (best) |
| --- | ---: | ---: | ---: | --- |
| **M3_Deep** (fin+ship) | −0.22 | **+0.16** | **+1.01** | xattn 0.029 ✅ |
| M2_Deep (fin+rs) | −1.93 | −2.43 | −5.89 | — |
| M4_Deep (fin+rs+ship) | −8.30 | −0.67 | +0.33 | xattn 0.026 ✅ |

**Reading**: M0 is cleared only where shipping is present, and only under
adaptive fusion: M3 clears it under gated and cross-attention, M4 only under
cross-attention, and M2 (fin+rs) never. Plain concatenation clears M0 in no combo,
so the gain depends on weighting the modalities rather than on stacking them.
Cross-attention gives the single-seed peak but is less stable across seeds (see
B.4), so gated is the main reported fusion.

---

## B.3 Flat robustness

### B.3.1 M2 leave-one-AOI-out (LOAO)

Source: `05_outputs/baselines/Flat/M2_Flat/baseline_metrics_anom_loao.csv` (+ full
per-AOI dRMSE in `baseline_loao_anom.csv`). Removing any single AOI leaves the M2
result essentially unchanged (|dRMSE| small, no single site drives a positive
contribution), i.e. the weak RS signal is diffuse rather than one-site-driven.

### B.3.2 M3 leave-one-channel-out (LOCHO)

seed 42, 257 weeks. Source:
`05_outputs/baselines/Flat/M3_Flat/robustness_m3_summary.csv`. Skill vs M0 (%)
and CW p vs M1 (nested increment) for XGB.

| Arm | XGB skill vs M0 | XGB CW p vs M1 |
| --- | ---: | --- |
| full (113 cols, main) | −6.68 | **0.0002** ✅ |
| core (38) | −7.81 | 0.096 |
| portwatch-only | −4.91 | **0.0003** ✅ |
| gfw-only | −5.70 | 0.047 ✅ |
| gfw-presence | −4.99 | 0.039 ✅ |
| gfw-aggregate | −4.62 | 0.094 |
| tanker-only | −4.60 | **0.0018** ✅ |

**Reading**: the nested shipping increment over M1 (CW p) is significant
across several channel subsets — strongest for tanker/PortWatch flows — so the
M3 signal is not an artefact of one data source, even though no flat arm beats M0
in absolute RMSE.

### B.3.3 M2 water-masked RS variant

Source: `baseline_metrics_anom_watermask.csv`. Masking water pixels lifts the M2
nested increment (XGB CW p vs M1 = 0.028 ✅ vs 0.085 un-masked) but M2 still does
not beat M0 (skill −6.3%). De-noising only makes RS marginally significant → RS
value is limited → motivates modality-aware fusion (RQ2).

### B.3.4 M4 leave-one-modality-out (LOMO)

Source: `Flat/M4_Flat/robustness_m4_summary.csv`. Dropping RS from M4 (i.e. M1+M3)
keeps the significant nested increment (XGB CW p vs M1 = 0.0002 ✅), whereas the
full M4 adds RS without accuracy gain — flat multi-modal concatenation cannot
improve accuracy and significance together.

---

## B.4 Deep multi-seed & sweeps

Source: `04_code/scripts/deep/run_deep_sweep.py` → `deep_sweep_summary.csv` (seeds 42, 1, 2;
lookback 4; d 32).

| Config | skill vs M0 (3 seeds) | Note |
| --- | --- | --- |
| **finship gated** | −0.50% ± 0.80 | **best 3-seed mean**; spread far tighter than cross-attention → main model |
| m4rep gated | −0.93% ± 0.29 | tightest spread, but centred well below M0; adding RS gives no gain |
| m4 xattn | −1.85% ± **2.80** | seed 42 best (+0.33%) but seed 2 collapses to −5.01% → not main |

All three means are below M0, so the positive headline figures reported in
Chapter 4 (finship gated +0.16%, cross-attention +1.01%) are seed-42 outcomes
rather than expected skill: averaged over seeds, no Deep configuration beats the
no-change benchmark. Gated finance+shipping is selected on the best seed-averaged
mean and on its low dispersion relative to cross-attention, not on a claim of
positive expected skill. This is the sharpest single limitation on the Deep
results and is carried into Chapter 5.

Single-seed (42) hyper-sweep, finship gated: lookback 4/8/12 × d 32/64 →
lb 8 d 32 best (+0.25%) > lb 4 d 32 (+0.16%) > lb 12 d 32 (−0.08%); **d 64 always
worse** (−0.76 / −1.11 / −1.11; short weekly sample). Halving encoder depth
(1 GAT + 1 TCN layer instead of 2 + 2) also degrades skill to −1.26%, so the main
setting is not simply over-parameterised. Main model stays locked at lookback 4,
d 32 for flat protocol parity.

---

## B.5 Other robustness (documented, not tabulated here)

- **Lookback sweep** L1/8/12 per layer — `sweep_m*` + `deep_sweep`.
- **C2 dimensionality reduction** (PCA / ElasticNet) for M2 —
  `c2_summary.csv`; increment not pure over-fitting.
- **feature-mode = returns** stationarised variant — numerical robustness.
- **min_train = 78 longer window** — Appendix-level; main protocol
  keeps min_train = 104.
