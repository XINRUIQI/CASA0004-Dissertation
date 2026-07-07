# Chapter 1 — Introduction

## 1.1 Background and context

Crude oil price forecasting is a central challenge in energy economics, financial risk management, and supply-chain disruption analysis. Brent crude, as the dominant global pricing benchmark, is influenced by a complex interplay of market fundamentals (inventories, production, refinery activity), macroeconomic conditions (interest rates, exchange rates, equity markets), exogenous disruption events (supply shocks, sanctions, OPEC decisions), and physical supply-chain dynamics (shipping flows, infrastructure utilisation). Traditional forecasting models rely primarily on structured market data, but the increasing availability of alternative data sources — including satellite remote sensing of oil infrastructure and AIS-derived maritime shipping analytics — offers an opportunity to construct richer, multimodal representations of the oil market.

This dissertation investigates whether integrating heterogeneous, spatially grounded data modalities — remote-sensing indicators of oil-infrastructure activity and maritime shipping flows through global chokepoints — can improve short-term Brent crude oil price forecasting beyond what conventional market-fundamental models achieve, and, in particular, whether *how* these modalities are fused (preserving each modality's structure versus flattening them into a single table) changes that answer.

## 1.2 Research questions

**RQ1 (incremental value): Do remote-sensing and shipping modalities add incremental out-of-sample value for weekly Brent forecasting over a financial/macroeconomic baseline, and does any configuration beat the no-change random walk?**

This question establishes baseline performance using structured market data (EIA weekly petroleum reports, Brent/WTI dynamics, FRED macroeconomic and risk indicators) and then, through a nested ablation (M1 → M2 → M3 → M4), tests whether adding remote-sensing indicators (Sentinel-2 NDVI/NDWI/NDBI/BSI and VIIRS night-light anomalies at 11 oil-infrastructure AOIs spanning export terminals, transit hubs and demand-side refineries/ports) and maritime shipping features (IMF PortWatch chokepoint transits, AIS/GFW tanker activity) yields statistically significant, correctly signed error reductions — assessed with Diebold–Mariano and Clark–West tests against both the random walk (M0) and the financial baseline (M1).

**RQ2 (fusion architecture): On identical data, does modality-aware representation-level fusion outperform flat feature / early fusion?**

This is the core comparison. Rather than proposing a new algorithm, the study integrates existing methods — frozen Earth-observation embeddings, modality-specific encoders, and gated / cross-attention fusion — and compares this representation-level design head-to-head with the flat feature fusion of the baseline layer under one leakage-safe protocol.

**RQ3 (interpretability): Do the fusion gating / attention weights reveal which modality the model relies on across different market regimes, and do these align with known supply shocks and geopolitical events?**

Using the learned gating weights together with SHAP-based attribution, and a regime-conditional reading (e.g. 2020 COVID, 2022 Russia–Ukraine, 2023–24 Red Sea crisis), this question examines whether alternative-data signals matter more during supply disruptions and exogenous shocks than in stable market conditions.

## 1.3 Research aims and objectives

**Aim:** To develop and evaluate a modality-aware multimodal machine-learning framework for short-term Brent crude oil price forecasting that integrates market fundamentals, satellite remote sensing, and maritime shipping data, and to test whether representation-level fusion improves on flat feature fusion.

**Objectives:**

1. Construct a unified weekly feature matrix over the 2019–2025 comparison window, aligning heterogeneous data sources across temporal frequencies (daily, weekly, monthly) and spatial scales (site-level AOIs, chokepoint-level, national/global) with strict, leakage-safe release-time alignment.
2. Design and implement a nested ablation (M1–M4) under a shared rolling-origin protocol to isolate the marginal predictive contribution of each data modality, tested with Diebold–Mariano and Clark–West statistics.
3. Integrate modality-specific encoders and gated / cross-attention fusion, and compare this representation-level design against flat feature fusion on identical data.
4. Apply interpretable methods (fusion gating weights and SHAP) to identify the most informative modalities and features across market regimes.

## 1.4 Contribution

This research makes three contributions:

1. **Methodological:** Rather than proposing a new fusion operator, it integrates existing methods (frozen EO embeddings, modality-specific encoders, gated / cross-attention fusion, and missing-modality handling) and, for the first time in weekly Brent forecasting, systematically compares representation-level modality-aware fusion against flat feature fusion under one leakage-safe protocol with Diebold–Mariano and Clark–West testing.
2. **Empirical:** It provides evidence on the conditions under which alternative spatial data sources are most informative — specifically, whether remote sensing and shipping signals offer greater predictive power during supply disruptions and market crises compared to stable market periods.
3. **Practical:** The framework demonstrates how openly available geospatial data (Sentinel-2, VIIRS, IMF PortWatch, GFW) can complement traditional market intelligence for energy market participants, policymakers, and risk managers.

## 1.5 Dissertation structure

- **Chapter 2 — Literature Review:** Reviews crude oil price forecasting methods, shipping and remote sensing as economic proxies, multimodal forecasting and fusion architectures, and forecast-evaluation methodology, identifying the research gap.
- **Chapter 3 — Methodology:** Describes the research design, dataset inventory, preprocessing and leakage-safe alignment, the baseline models (random walk, Ridge, XGBoost, deep early-fusion) and the modality-aware fusion architecture (modality-specific encoders with gated / cross-attention fusion), the ablation framework, and the evaluation strategy.
- **Chapter 4 — Results:** Presents EDA findings, M1–M4 ablation results, and SHAP-based feature importance analysis across market regimes.
- **Chapter 5 — Discussion:** Interprets results, discusses implications, limitations, and future work.
- **Chapter 6 — Conclusion:** Summarises findings and contributions.
