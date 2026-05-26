# Reading Note — Paper 02: Oil Price Forecasting with Deep Learning

## Citation

Foroutan, P., & Lahmiri, S. (2024). Deep learning systems for forecasting the prices of crude oil and precious metals. *Financial Innovation*, 10, Article 111, 1–40.

- **DOI**: [10.1186/s40854-024-00637-z](https://doi.org/10.1186/s40854-024-00637-z)
- **Published**: 16 July 2024 (Open Access)

---

## Core Method

- **Models**: Systematic comparison of 16 deep-learning and machine-learning architectures, including:
  - Recurrent: LSTM, BiLSTM, GRU, BiGRU
  - Time-embedding variants: T2V-BiLSTM, T2V-BiGRU
  - Convolutional: CNN-LSTM, TCN (Temporal Convolutional Network)
  - Transformer-based architectures
  - Machine learning baselines: Random Forest, LightGBM, SVR, XGBoost
- **Targets**: Daily prices of WTI crude oil, Brent crude oil, gold, and silver.
- **Sliding window**: Multiple input sequence lengths tested (e.g., 5-day, 10-day, 30-day windows) to evaluate the effect of lookback period on prediction accuracy.
- **Evaluation**: MAE, MAPE, RMSE across all commodity markets.

---

## Key Findings

1. **TCN achieves the best overall performance** for WTI, Brent, and silver forecasting, with the lowest MAE values (WTI: 1.444, Brent: 1.295, Silver: 0.346), demonstrating that dilated causal convolutions effectively capture long-range temporal dependencies in oil price series.
2. **CNN-LSTM is best for gold**, achieving MAE of 15.188 with a 30-day input window, suggesting that the hybrid architecture combining spatial feature extraction (CNN) with sequential modelling (LSTM) is well-suited for precious metals.
3. **LightGBM is the top machine-learning model**, delivering performance comparable to TCN in several settings, challenging the assumption that deep learning always dominates gradient boosting on tabular time-series tasks.
4. **Sliding window length matters**: Longer lookback windows (30-day) generally improve Brent and gold predictions, while shorter windows (5–10 day) sometimes suffice for WTI, reflecting differences in mean-reversion characteristics.
5. **Brent-specific results** are explicitly reported, making this paper directly applicable to the dissertation's target commodity.

---

## Relevance to This Dissertation

| Aspect | Connection |
|--------|------------|
| **Brent coverage** | One of the few large-scale DL benchmarks that explicitly includes Brent crude oil, directly supporting the dissertation's price target. |
| **Architecture comparison** | Provides a comprehensive baseline ranking (TCN > CNN-LSTM > BiGRU > LSTM > LightGBM) that informs the dissertation's model selection. |
| **TCN as candidate** | TCN's strong performance motivates its inclusion as a temporal encoder in the dissertation's multimodal fusion architecture. |
| **Lookback window analysis** | The sliding window experiments inform hyperparameter decisions for the dissertation's time-series input design. |
| **DL vs. ML trade-off** | LightGBM's competitive performance supports a hybrid ensemble strategy combining DL temporal encoders with gradient boosting for tabular features. |

---

## Limitations

1. **Univariate price-only input**: All models use only historical price as input; no exogenous features (macro indicators, sentiment, supply data) are incorporated, limiting the practical forecasting ceiling.
2. **No multimodal fusion**: Text, image, or spatial data are not considered, leaving the incremental value of alternative data sources unexplored.
3. **Point forecasts only**: The study produces deterministic predictions without uncertainty quantification (no prediction intervals or probabilistic outputs).
4. **No structural break analysis**: The models are not evaluated for robustness across distinct market regimes (e.g., COVID-19 crash, Ukraine conflict), which would test generalisation under distributional shift.
5. **Single-step forecasting**: Multi-horizon or multi-step-ahead forecasting is not explored, which is often more practically relevant for trading and risk management.

---

## Notes for Dissertation Integration

- Cite as the primary DL benchmark reference for Brent crude oil price forecasting.
- Use TCN and CNN-LSTM results as motivation for choosing temporal encoder architectures.
- Highlight the "univariate limitation" as a gap this dissertation addresses through multimodal data fusion.
- The LightGBM finding supports including a gradient boosting branch alongside deep learning components.
