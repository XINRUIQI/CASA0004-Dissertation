# Chapter 1 — Introduction

## 1.1 Background and context

Crude oil price forecasting is a central challenge in energy economics, financial risk management, and supply-chain disruption analysis. Brent crude, as the dominant global pricing benchmark, is influenced by a complex interplay of market fundamentals (inventories, production, refinery activity), macroeconomic conditions (interest rates, exchange rates, equity markets), exogenous disruption events (supply shocks, sanctions, OPEC decisions), and physical supply-chain dynamics (shipping flows, infrastructure utilisation). Traditional forecasting models rely primarily on structured market data, but the increasing availability of alternative data sources — including satellite remote sensing, AIS-derived shipping analytics, and NLP-processed market reports — offers an opportunity to construct richer, multimodal representations of the oil market.

This dissertation investigates whether integrating heterogeneous, spatially grounded data modalities — remote sensing indicators of oil-infrastructure activity, maritime shipping flows through global chokepoints, and NLP-extracted signals from official market reports — can improve short-term Brent crude oil price forecasting beyond what conventional market-fundamental models achieve.

## 1.2 Research questions

**RQ1: To what extent can conventional market fundamentals and macroeconomic indicators predict short-term Brent crude oil price movements over the 2006–2025 period?**

This question establishes the baseline forecasting performance using structured market data (EIA weekly petroleum reports, Brent/WTI price dynamics, FRED macroeconomic indicators), against which multimodal enhancements are benchmarked.

**RQ2: Does the progressive integration of text-derived (NLP), remote-sensing, and shipping-activity features yield statistically significant improvements in forecast accuracy, and which modalities contribute the most?**

This question is addressed through a systematic ablation study (M1 → M2 → M3 → M4), progressively adding text signals (OPEC MOMR sentiment, GDELT disruption-event intensity), remote-sensing indicators (Landsat/Sentinel-2 NDVI, NDWI, NDBI, BSI; VIIRS night-light radiance at 11 oil-infrastructure AOIs spanning export terminals, transit hubs, and demand-side refineries/ports), and maritime shipping features (IMF PortWatch chokepoint transits, AIS-derived tanker activity, EMODnet vessel density) to the market-fundamental baseline.

**RQ3: Which spatial and temporal features from remote sensing and shipping data are most informative for oil price prediction, and how do their contributions vary across different market regimes?**

Using SHAP-based feature importance and regime-conditional analysis (e.g., crisis vs. stable periods such as 2008 GFC, 2014 oil crash, 2020 COVID, 2022 Russia–Ukraine, 2023–24 Red Sea crisis), this question examines whether alternative data sources provide stronger signals during supply disruptions and exogenous shocks than during stable market conditions.

## 1.3 Research aims and objectives

**Aim:** To develop and evaluate a multimodal machine learning framework for short-term Brent crude oil price forecasting that integrates market fundamentals, NLP-extracted text signals, satellite remote sensing, and maritime shipping data.

**Objectives:**

1. Construct a unified weekly feature matrix spanning 2006–2025, aligning heterogeneous data sources across temporal frequencies (daily, weekly, monthly) and spatial scales (site-level AOIs, chokepoint-level, national/global).
2. Design and implement a progressive ablation experiment (M1–M4) to isolate the marginal predictive contribution of each data modality.
3. Apply interpretable machine learning methods (SHAP) to identify the most informative spatial and temporal features across market regimes.
4. Evaluate forecast performance using both classification (direction prediction) and regression (return magnitude) metrics under a strict temporal train/validation/test split.

## 1.4 Contribution

This research makes three contributions:

1. **Methodological:** It is among the first to systematically integrate satellite-derived oil-infrastructure indicators and AIS-based maritime shipping features into a unified oil price forecasting framework, with a rigorous ablation design that quantifies each modality's marginal value.
2. **Empirical:** It provides evidence on the conditions under which alternative spatial data sources are most informative — specifically, whether remote sensing and shipping signals offer greater predictive power during supply disruptions and market crises compared to stable market periods.
3. **Practical:** The framework demonstrates how openly available geospatial data (Landsat, Sentinel-2, VIIRS, IMF PortWatch, GFW) can complement traditional market intelligence for energy market participants, policymakers, and risk managers.

## 1.5 Dissertation structure

- **Chapter 2 — Literature Review:** Reviews crude oil price forecasting methods, the use of shipping and remote sensing data as economic proxies, and multimodal forecasting approaches, identifying the research gap.
- **Chapter 3 — Methodology:** Describes the research design, dataset inventory, preprocessing pipeline, feature engineering, model architecture (XGBoost/LightGBM, Temporal Fusion Transformer), ablation framework, and evaluation strategy.
- **Chapter 4 — Results:** Presents EDA findings, M1–M4 ablation results, and SHAP-based feature importance analysis across market regimes.
- **Chapter 5 — Discussion:** Interprets results, discusses implications, limitations, and future work.
- **Chapter 6 — Conclusion:** Summarises findings and contributions.
