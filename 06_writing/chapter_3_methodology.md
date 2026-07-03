# Chapter 3 — Methodology

## 3.1 Research design

This dissertation adopts a **two-layer design** under one shared, leakage-safe evaluation protocol.

**Core empirical layer (baselines / ablation).** Three model families predict the next-week Brent price under identical conditions: M0 (no-change random walk), a regularised linear model (Ridge), and a gradient-boosted tree (XGBoost). Across these, the data modality is varied in a nested ablation — **M1** (financial / macro), **M2** (M1 + remote sensing), **M3** (M1 + shipping), **M4** (all) — so the incremental value of each alternative-data modality can be isolated (RQ1).

**Integration and empirical-testing layer (contribution).** The same modalities are then fused at the *representation* level using modality-specific encoders and gated / cross-attention fusion, and compared head-to-head with the flat feature / early fusion of the core layer (RQ2). This layer **integrates existing methods rather than proposing a new algorithm**.

The sole prediction target is the next-week Brent price \(P_{t+1}\); the model is trained on the one-week log return \(r_{t+1}=\log(P_{t+1}/P_t)\) and the price is reconstructed as \(\hat P_{t+1}=P_t\,e^{\hat r_{t+1}}\). Direction and returns are derived from the predicted price for evaluation only (single-task regression). All configurations share a common window (2019–2026), weekly Friday-aligned sampling, a 4-week lookback, expanding rolling-origin backtesting, and the same chronological train/validation/test ordering, so performance differences reflect **data and architecture, not protocol**.

## 3.2 Data sources
<!-- Dataset inventory table (link to 03_data/external_sources.md) -->
<!-- Dataset feasibility assessment -->

### 3.2.X Remote sensing site selection: oil infrastructure AOIs

This study extracts monthly spectral indices (NDVI, NDWI, NDBI, BSI) from Sentinel-2 and Landsat, and nighttime radiance from VIIRS, at a set of oil-infrastructure areas of interest (AOIs). Each AOI is defined as a 5 km circular buffer around the facility centroid. The selection of 11 globally distributed sites follows a three-criterion framework: **throughput/capacity rank**, **geographic and supply-chain diversity**, and **remote sensing observability**.

#### Criterion 1 — Throughput and capacity rank

Each selected site ranks among the highest-capacity facilities in its functional category, as documented by industry-standard sources. The quantitative justification is summarised in Table 3.X.

**Table 3.X — Selected AOI sites and capacity rankings**

| ID | Site | Type | Country | Capacity / Throughput | Rank | Source |
|----|------|------|---------|-----------------------|------|--------|
| P001 | Port of Rotterdam | Port | Netherlands | 397M tonnes/yr (2024) | Europe #1 port | Eurostat (2025) |
| P002 | Fujairah Oil Terminal | Terminal | UAE | 1.5M bpd ADCOP bypass + global #2 bunkering hub | Strategic bypass | ADNOC; Marine Insight |
| P003 | Ras Tanura Terminal | Terminal | Saudi Arabia | 9M bpd design capacity; 90% of Saudi exports | World #1 export terminal | IMF PortWatch; Saudi Aramco |
| P004 | Jurong Island | Refinery | Singapore | 605,000 bpd | World #12 refinery | Oil & Gas Journal via Wikipedia |
| P005 | Houston Ship Channel | Port | United States | >3M bpd combined refining cluster | US #1 waterborne tonnage port | US DoT; OGIM |
| P006 | Ningbo-Zhoushan Port | Port | China | 1.37B tonnes/yr (2024); 185M tonnes crude | World #1 cargo port; China #1 crude import | Xinhua; Global Economic Indicator |
| P007 | Jamnagar Refinery | Refinery | India | 1,240,000 bpd | World #1 refinery | Oil & Gas Journal via Wikipedia |
| P008 | Basra Oil Terminal | Terminal | Iraq | >3.3M bpd; 95%+ of Iraq exports | Middle East #3 export terminal | Marine Insight |
| P009 | Ulsan Refinery | Refinery | South Korea | 840,000 bpd | World #3 refinery | SK Energy; Oil & Gas Journal |
| P010 | Kharg Island Terminal | Terminal | Iran | 1.5–2.5M bpd; 90–96% of Iran exports | Iran sole major export hub | Kpler; Iran International (2026) |
| P011 | Yanbu Export Terminal | Terminal | Saudi Arabia | 4.5M bpd nominal loading capacity | Red Sea #1 crude terminal | Argus Media; Aramco |

Note: The OGIM database (Global Oil Infrastructure Mapping Project, May 2024) confirms the existence and operational status of all 11 facilities but does not include throughput or capacity fields. Capacity figures are therefore sourced from the Oil & Gas Journal world refinery survey, IMF PortWatch, Eurostat maritime transport statistics, and operator reports as cited above.

#### Criterion 2 — Geographic and supply-chain diversity

The 11 sites are distributed across three functional stages of the global oil supply chain and five geographic regions, ensuring that no single region or supply-chain segment dominates the remote sensing signal:

- **Supply / export** (4 sites): Ras Tanura, Basra, Kharg Island, and Yanbu collectively represent the three largest OPEC producers (Saudi Arabia, Iraq, Iran) and cover both Persian Gulf and Red Sea export routes.
- **Transit / storage** (2 sites): Fujairah (Hormuz bypass and bunkering) and Jurong Island (Malacca Strait refining node) capture mid-chain logistics activity.
- **Demand / import-refining** (5 sites): Rotterdam (European Brent hub), Houston (US Gulf Coast), Ningbo-Zhoushan (China demand proxy), Jamnagar (India demand proxy), and Ulsan (East Asian refining centre) represent the major consumption regions.

This design mirrors the geographic structure of the six chokepoint-level shipping features (Hormuz, Suez, Malacca, Bab el-Mandeb, Panama, Cape of Good Hope), allowing cross-modal spatial alignment between remote sensing and maritime indicators.

#### Criterion 3 — Remote sensing observability

All selected sites feature large-footprint ground infrastructure (tank farms, refinery complexes, port berths) that produce measurable spectral and radiance signatures at the spatial resolutions of Sentinel-2 (10–20 m), Landsat (30 m), and VIIRS (∼750 m):

- **Optical indices**: NDBI (Normalised Difference Built-up Index) captures changes in built-up area and impervious surface density; BSI (Bare Soil Index) responds to construction activity and cleared land; NDVI and NDWI provide complementary signals of vegetation displacement and water-body changes associated with terminal expansion or land reclamation.
- **Nighttime radiance**: VIIRS DNB monthly composites capture continuous nighttime operational intensity. Halpern et al. (2022) demonstrate a strong correlation (Spearman Rs = 0.84, p < 0.01) between VIIRS nighttime light intensity and container port throughput across 601 global anchorage areas, validating NTL as a proxy for port-level economic activity. The AOI buffer radius of 5 km follows the design used by Guetta-Jeanrenaud et al. (2025), who define port AOIs as square buffers with a 3 km radius centred on World Port Index coordinates; we adopt a slightly larger circular buffer to accommodate the spatial extent of major refinery and terminal complexes.

#### Precedents in the literature

The selection of a focused set of globally representative sites is consistent with recent satellite-based oil market studies:

- **Wang et al. (2023)** select 8 US oil storage areas (Cushing OK, Houston TX, Texas City TX, Beaumont TX, Corpus Christi TX, Lake Charles LA, Baton Rouge LA, Mississippi River LA) based on inventory concentration: PADD 2 (Cushing) and PADD 3 (Gulf Coast) account for over 70% of total US crude oil stocks. Their per-site FRT cloudiness measures predict WTI weekly returns with statistical significance for 7 of 8 areas.
- **Guetta-Jeanrenaud et al. (2025)** train port-level trade prediction models on 64 US ports classified as Small or larger by the NGA World Port Index, combining SAR (Sentinel-1), nighttime lights (VIIRS), and WPI attributes. Oil terminal depth emerges as the single most important predictor, while satellite variables capture temporal dynamics that static port attributes cannot.
- **Elvidge et al. (2023)** use VIIRS Nightfire to automatically identify ∼20,000 persistent infrared emitters globally, including refineries and gas flares, demonstrating that satellite-detected industrial heat signatures serve as reliable monthly production proxies.
- **ECB Working Paper 3198 (2025)** uses Sentinel-5P TROPOMI tropospheric NO₂ as a real-time proxy for oil demand across advanced and emerging economies, achieving significant improvements in nowcasting accuracy — an approach that complements the facility-level optical and NTL indicators used in this study.

#### Robustness: leave-one-AOI-out sensitivity analysis

To verify that the model results are not driven by the inclusion or exclusion of any single site, a leave-one-AOI-out sensitivity analysis is conducted alongside the modality-level ablation study (M1 → M4). For each of the 11 AOIs, the remote sensing features derived from that site are excluded, and the M3 (market + text + RS) and M4 (full multimodal) models are re-estimated. The resulting distribution of performance changes (ΔRMSE, ΔR²) across all 11 leave-out runs provides evidence of whether the remote sensing contribution is robust to site selection or unduly influenced by individual AOIs. Results are reported in Section 4.X.

#### References

- Elvidge, C.D. et al. (2023). Global satellite monitoring of exothermic industrial activity via infrared emissions. *Remote Sensing*, 15(19), 4760.
- Guetta-Jeanrenaud, L. et al. (2025). Watching trade from space: Nowcasting and spatial extrapolation of port-level maritime trade using satellite imagery. *arXiv:2604.15444*.
- Halpern, B.S. et al. (2022). A global assessment of night lights as an indicator for shipping activity in anchorage areas. *Remote Sensing*, 14(5), 1079.
- Wang, Y. et al. (2023). Cloud cover and expected oil returns. *Humanities and Social Sciences Communications*, 10, 60.
- ECB (2025). Can satellites predict oil demand? *ECB Working Paper Series*, No. 3198.

## 3.3 Data preprocessing

All variables are aligned to their real **publication timestamp**, not the statistical reference date, to prevent look-ahead. EIA Weekly Petroleum Status Report series (reference "week ending Friday", released the following Wednesday) are lagged one week at source; monthly geopolitical / activity series carry conservative 1–5 week release lags; remote-sensing observations use an as-of join at month-end + 15 days; shipping is lagged GFW +4 weeks and PortWatch +1 week. Self-checks confirm every alignment is a lag, never a forward shift.

**Publication-lag treatment of monthly predictors.** Three M1 predictors are natively monthly — geopolitical risk (`gpr`; Caldara & Iacoviello, 2022), the Kilian index of global real economic activity (`global_econ_activity`), and the IMF global industrial-materials price index (`nonoil_industrial_commodity`; FRED `PINDUINDEXM`). Because the raw files record only the statistical reference month and not the row-level release date, these series are mapped to the weekly grid by month-end forward-fill followed by a **fixed, conservative publication lag** rather than a full real-time vintage reconstruction. This respects the real-time alignment principle stressed by Alquist et al. (2013) [P053], Kilian (2009) [P052] and Costa et al. (2021) [P072] — a predictor must enter the model by its release date, never by its reference period — while acknowledging that genuine data-vintage merges (Baumeister & Kilian, 2015 [P054]) are not reconstructed here. The lags are tiered by how quickly each source publishes (Table 3.Y).

**Table 3.Y — Conservative publication lags for the three monthly M1 predictors**

| Predictor | Source | Lag | Rationale | First valid week |
|-----------|--------|-----|-----------|------------------|
| `gpr` | Caldara–Iacoviello news-based index | +1 week | Scholar-maintained index, typically updated in the first days of the following month; short release chain, so only a minimal buffer is needed | 2006-01-13 |
| `global_econ_activity` | Dallas Fed Kilian REA | +5 weeks | Monthly research index compiled from dry-cargo freight rates; available well after month-end | 2006-02-10 |
| `nonoil_industrial_commodity` | IMF via FRED `PINDUINDEXM` | +5 weeks | Official commodity index typically released mid-to-late in the following month | 2006-03-10 |

The five-week value is a **conservative heuristic, not a literature constant**: no oil-forecasting study prescribes a single monthly-macro lag, and the appropriate figure depends on the series and on whether vintage data are used. Absent row-level release dates, +5 weeks leaves roughly one month of slack after the reference period, so a Friday-origin forecast never consumes a month's final macro value before it was public ("late rather than early"). The first-valid-week anchors in Table 3.Y double as leakage self-checks (`gpr` +1 week, the two slower series +5 weeks from their respective data starts); robustness to this assumption can be assessed by re-running with the monthly lag set to 3, 5 and 7 weeks.

Monthly values are **not** forward-filled into repeated weekly values. Each remote-sensing observation instead carries `days_since_obs` (age), `valid_mask` and `modality_mask`, so staleness and missingness stay explicit. Remote-sensing signals are expressed as **within-site standardised anomalies** on an expanding, past-only window (min 12 months): each index is de-seasonalised and z-scored against that site's own history, removing cross-site scale and seasonality (following the night-lights standardisation literature) without using future information. A water-masked variant (MNDWI, land-only pixels; McFeeters 1996, Xu 2006) is produced for robustness at water-dominated terminals.

All scalers, variance filters and any feature selection are fit **inside the training fold only**. The three modalities are merged on a common Friday index into a single leakage-safe matrix (**365 weeks × 221 columns** for 2019–2026), each column documented with its modality, group, release lag and coverage. The default M2 feature contract is the **55 within-site anomalies**; level, age, and near-constant availability columns are excluded from the main analysis by design.

## 3.4 Analytical framework

**Core empirical layer.** M0 is the random walk (\(\hat r=0\)). The tabular baselines flatten the past \(L\) weeks of every selected column into one feature vector and fit Ridge (with StandardScaler) and XGBoost, each with time-aware inner-validation tuning. These flat tabular models — all columns concatenated with no modality-specific structure — are the feature / early-fusion reference that the contribution layer must beat.

**Integration layer (contribution).** Each modality is encoded separately before fusion: a small TCN / GRU for the financial sequence (cf. Foroutan & Lahmiri, 2024 [P001]); an Earth-observation branch that feeds Sentinel-2 patches through a **frozen** foundation model (Prithvi-EO-2.0 [P094] / SatMAE [P095]) to obtain image embeddings, then temporal- and site-attention pooling (cf. mTAN [P099]; Gohari et al., 2024 [P039]); and a shipping branch pairing a graph encoder (GAT / Graph WaveNet [P091]; crude-oil maritime GNNs [P062/P063]) with a temporal model. The three embeddings are fused with increasing sophistication — encoder-concatenation, **Gated Fusion** (Gated Multimodal Units [P096], whose per-sample gates also serve RQ3 interpretability), and cross-modal attention [P039]. Because monthly imagery and lagged shipping are frequently missing, training uses **modality masking / modality dropout** (Ma et al., 2022 [P097]; ModDrop [P100]) and time-gap handling (GRU-D [P098]; mTAN [P099]). This organisation follows the multimodal taxonomy of Baltrušaitis et al. (2019) [P101]: the baselines are early / feature-level fusion, whereas the contribution layer learns a **joint representation with model-based fusion**.

**Model-selection justification.** The random walk is a mandatory benchmark (Alquist et al., 2013 [P053]); Ridge and XGBoost are strong tabular baselines rather than assumed winners (Costa et al., 2021 [P072]; Yılmaz & Zehir, 2026 [P076]) and supply the flat feature / early-fusion reference for RQ2; TFT / GNN architectures ([P089/P091/P062/P063]) are candidate encoders for the contribution layer. All models are kept deliberately small and strongly regularised given the ~360-week sample.

## 3.5 Evaluation metrics

All metrics are computed on the **reconstructed price** \(\hat P_{t+1}\): RMSE, MAE, directional accuracy, and skill relative to M0 (\(1-\text{RMSE}/\text{RMSE}_{M0}\)). Two forecast-accuracy tests are used (following Diebold & Mariano, 1995 [P058]): the **Diebold–Mariano** test (HLN small-sample correction) for the non-nested comparison against M0, and the **Clark–West** MSPE-adjusted test for the *nested* increments (M2/M3/M4 over M1), since the standard DM statistic is biased for nested models. Every comparison uses the same target, horizon, out-of-sample dates and rolling origin. Interpretability uses **SHAP** (Lundberg & Lee, 2017 [P059]) aggregated to the Market / Remote-sensing / Shipping modality groups, with feature selection performed inside each training window to avoid leakage; SHAP importance is read as "what the model used", not as causal evidence or a substitute for the ablation / DM tests. Robustness checks include leave-one-AOI-out, the water-masked remote-sensing variant, and lookback sensitivity. Direction labels, where reported, use a flat threshold at the 33rd percentile of \(|r|\) on the training fold.

## 3.6 Ethical considerations

All data are public or appropriately licensed: EIA and FRED (financial / macro), Copernicus Sentinel-2 and VIIRS via Google Earth Engine (remote sensing), IMF PortWatch and Global Fishing Watch (shipping). No personal data are used; AIS is aggregated to chokepoint / port level and never used to track individual operators. Dark-vessel and sanctions-related activity, where discussed, is treated only in aggregate as a data-coverage caveat, not to identify specific vessels. The pipeline is fully scripted and leakage-audited for reproducibility, and the compute footprint is kept low by pre-computing frozen foundation-model embeddings once rather than training large image models. Forecasts are a research artefact and **not investment advice**.
