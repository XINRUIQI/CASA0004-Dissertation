# Reading Note — Paper 01: Oil Price Forecasting with XGBoost (ML)

## Citation

Tissaoui, K., Zaghdoudi, T., Hakimi, A., & Zaghdoudi, K. (2023). Do Gas Price and Uncertainty Indices Forecast Crude Oil Prices? Fresh Evidence Through XGBoost Modeling. *Computational Economics*, 62, 663–687.

- **DOI**: [10.1007/s10614-022-10305-y](https://doi.org/10.1007/s10614-022-10305-y)
- **Published**: September 2022 (online) / 2023 (volume)

---

## Core Method

- **Model**: XGBoost (eXtreme Gradient Boosting), compared against SVM and ARIMAX(p,d,q).
- **Target**: WTI crude oil price forecasting.
- **Features**: Gas price, oil implied volatility (OVX), economic policy uncertainty (EPU), exogenous risk index (GPR), and other uncertainty indices.
- **Interpretability**: Shapley Additive Explanations (SHAP) used to analyse feature importance and decompose individual prediction contributions.
- **Evaluation**: Out-of-sample RMSE, MAE, MAPE, and convergence speed.

---

## Key Findings

1. Machine learning models (XGBoost, SVM) significantly outperform traditional econometric models (ARIMAX) in forecasting WTI crude oil prices, owing to their ability to capture nonlinear, non-stationary relationships.
2. XGBoost achieves the best overall performance among all models, with the lowest error metrics and fastest convergence, attributed to its regularisation mechanisms and ensemble boosting structure.
3. SHAP analysis reveals that oil implied volatility (OVX) and gas price are the most informative predictors, capturing valuable information about uncertainty transmission to crude oil markets.
4. Uncertainty indices (EPU, GPR) contribute meaningful but secondary forecasting power, validating the hypothesis that macroeconomic and exogenous uncertainty channels affect oil prices.

---

## Relevance to This Dissertation

| Aspect | Connection |
|--------|------------|
| **Model choice** | XGBoost is one of the candidate ML models in this dissertation's forecasting framework; this paper provides a strong benchmark and hyperparameter tuning reference. |
| **Feature engineering** | The paper's use of macro-financial uncertainty indices (VIX/OVX, EPU) as predictors directly aligns with the "Market Fundamentals" pillar of this dissertation. |
| **Interpretability** | SHAP-based feature importance analysis provides a methodological template for explaining which data modalities (market, NLP, remote sensing, shipping) drive the dissertation's multimodal predictions. |
| **Baseline comparison** | The paper's XGBoost vs. ARIMAX comparison framework can be replicated in this dissertation to demonstrate the added value of ML over traditional econometrics. |

---

## Limitations

1. **Univariate target only**: Focuses exclusively on WTI prices; generalisability to Brent crude (the dissertation's target) is not tested.
2. **No deep learning comparison**: Does not benchmark against LSTM, Transformer, or other DL architectures, leaving open questions about potential performance ceilings.
3. **Single data modality**: Relies solely on structured numerical time-series features; no text (NLP), image (remote sensing), or spatial (AIS) data sources are incorporated.
4. **Static feature set**: Does not explore time-varying feature relevance or regime-switching dynamics that may be critical during supply shocks or market crises.
5. **Limited temporal granularity**: Does not explore intraday or weekly forecasting horizons.

---

## Notes for Dissertation Integration

- Use as a methodological reference for the XGBoost component of the ensemble/stacking framework.
- Adopt the SHAP interpretation pipeline for cross-modal feature importance analysis.
- Brent-specific replication with expanded feature set can serve as a natural extension cited in the literature review.
