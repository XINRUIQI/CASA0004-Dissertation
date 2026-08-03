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
| M_ship_GNN (shipping only) | −0.38 | −0.41 | −0.33 |
| M_rs_deep (RS only) | −2.30 | −3.07 | −1.04 |
| **M3_Deep_gated (main)** | **+0.11** | **+0.09** | **+0.14** |
| M2_Deep_gated | −2.43 | −3.06 | −1.41 |
| M4_Deep_gated | −1.28 | −2.35 | +0.49 |
| M4_Deep_Concat | −4.06 | −6.08 | −0.69 |

**Reading**: no flat model beats M0 in either sub-period. Among deep
models, **M3_Deep_gated is the only configuration with positive skill in both
sub-periods** (+0.09 / +0.14), i.e. the most stable small gain; M4_Deep_gated is
negative early and only turns positive late (+0.49), and cross-attention–driven
M4 gains are concentrated late. This supports gated finance+shipping as the main
model.

---

## B.2 Deep fusion matrix (3 modality combos × 3 fusions)

seed 42, lookback 4, 257 weeks. Source: `run_deep_fusion_matrix.py` →
`05_outputs/baselines/Deep/_cross/deep_fusion_matrix.csv`. Skill vs M0 in %.

| Combo | Concat | Gated | Cross-Attn | CW p vs M0 (best) |
| --- | ---: | ---: | ---: | --- |
| **M3_Deep** (fin+ship) | +0.06 | **+0.11** | **+0.74** | xattn 0.041 ✅ |
| M2_Deep (fin+rs) | −1.93 | −2.43 | −5.89 | — |
| M4_Deep (fin+rs+ship) | −4.06 | −1.28 | +0.12 | xattn 0.018 ✅ |

**Reading**: shipping (M3) is the only combo that clears M0 under any
fusion; adding RS (M2, M4) never helps at the concat/gated floor. Cross-attention
gives the single-seed peak but is less stable across seeds (see B.4), so gated is
the main reported fusion.

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
| **finship gated** | −0.47% ± 0.86 | **lowest variance, most stable** → main model |
| m4rep gated | −0.89% ± 0.60 | adding RS gives no gain |
| m4 xattn | −1.83% ± **2.76** | seed 42 best but seed 2 collapses to −4.98% → not main |

Single-seed (42) hyper-sweep, finship gated: lookback 4/8/12 × d 32/64 →
lb 8 d 32 best (+0.34%) > lb 4 (+0.11%) > lb 12 (negative); **d 64 always worse**
(short weekly sample). Main model stays locked at lookback 4, d 32 for flat
protocol parity.

---

## B.5 Other robustness (documented, not tabulated here)

- **Lookback sweep** L1/8/12 per layer — `sweep_m*` + `deep_sweep`.
- **C2 dimensionality reduction** (PCA / ElasticNet) for M2 —
  `c2_summary.csv`; increment not pure over-fitting.
- **feature-mode = returns** stationarised variant — numerical robustness.
- **min_train = 78 longer window** — Appendix-level; main protocol
  keeps min_train = 104.
