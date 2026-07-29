# Dissertation Draft Structure — Brief

  
A Modality-Aware Spatio-Temporal Fusion Framework for Brent Crude Oil Forecasting Using Financial Time Series, Satellite Imagery and Maritime Networks

---

## Chapter 1 — Introduction *(~900–1,200 words)*



### 1.1 Background and forecasting challenge

- Brent as global benchmark; weekly horizon hard to beat; M0 / no-change is the bar
- Alternative data: AIS / PortWatch shipping; RS (NTL, site optical indices / EO embeddings) as physical-market proxies
- Signals noisy, asynchronous, possibly price-*responding*; dominant practice = flat concat → loses temporal, site and network structure



### 1.2 Research gap

1. Incremental value unclear: few studies report **both** nested gain vs M1 **and** absolute skill vs M0 under leakage-safe evaluation
2. Fusion architectures lack fair comparison: flat vs representation-level under one protocol
3. Attribution often detached from predictive value (explain models that fail M0)



### 1.3 Aim and research questions

- **Aim:** Integrate frozen EO embeddings, modality encoders, fusion modules and rolling-origin protocol into one reproducible comparison framework — integration + fair comparison, not a new operator
- State RQ1–RQ3



### 1.4 Contributions

1. **Primary:** First systematic nested M1→M2/M3/M4 ablation for finance / RS / shipping on weekly Brent (RQ1)
2. **Method-integration:** Paired Flat (Ridge / XGBoost) vs Deep (encoders + gated / cross-attention) at matched sets (RQ2)
3. **Interpretability (supporting):** Gates and site–node attention on specs that beat M0 — Deep M3, Deep M4 where relevant (RQ3)



### 1.5 Dissertation structure

- Brief roadmap Chapters 2–6

---



## Chapter 2 — Literature Review 

### 2.1 Crude-oil forecasting and benchmark difficulty

### 2.2 Financial and macroeconomic predictors

### 2.3 Shipping activity as alternative data

### 2.4 Remote sensing for energy / economic monitoring

### 2.5 Multimodal and temporal fusion methods

### 2.6 Forecast evaluation and interpretability

### 2.7 Research gap and conceptual framework

---



## Chapter 3 — Data and Methods *(~2,500–3,000 words)*



### 3.1 Research design

- Flat: tabular concat → Ridge / XGBoost (early fusion)
- Deep: modality encoders → gated fusion (main) / cross-attention (alt) / concat (control)
- M0–M4 shared across families; pair at matched sets to isolate representation



### 3.2 Prediction target and timeline

- Target: next-week Brent P_{t+1}; train on r_{t+1}=\log(P_{t+1}/P_t); evaluate reconstructed price
- M0 = \hat P_{t+1}=P_t ≡ zero return
- Friday weeks; 2019–2025; 257-week scored span



### 3.3 Data sources

- **Finance:** EIA, FRED, Yahoo, etc.
- **RS:** 11 AOIs. Flat = monthly S2 indices + VIIRS NTL anomalies; Deep = frozen Prithvi-EO-2.0 embeddings (S2 only, no VIIRS)
- **Shipping:** PortWatch + AIS. Flat = tabular features; Deep = 17-node heterogeneous graph (11 AOIs + 6 chokepoints)



### 3.4 Temporal alignment, lags, missingness

- Release-timestamp alignment; modality-specific lags (e.g. EIA / PortWatch ~1 week)
- Flat: past-only fill; Deep: explicit missing masks



### 3.5 M0 benchmark and M1–M4 information sets

- M0: no-change (not trained)
- M1 finance → M2 +RS / M3 +shipping (parallel) → M4 all



### 3.6 Flat models

- 4-week lookback flattened; Ridge + XGBoost; tune per fold → Appendix C



### 3.7 Deep models

- Finance: causal TCN; RS: per-site temporal + site attention on frozen embeddings; Shipping: GAT on heterogeneous graph
- Fusion: gated (main text), cross-attention (comparative), concat (control)



### 3.8–3.9 Training and leakage-free protocol

- Expanding rolling-origin; first 104 weeks init + validation; refit every 13 weeks; scaling fit in-fold only



### 3.10 Evaluation, tests, interpretability

- Metrics: RMSE, MAE on price; DirAcc auxiliary; skill vs M0 = 100(1-\mathrm{RMSE}*\mathrm{model}/\mathrm{RMSE}*\mathrm{M0})
- **CW** for nested increments (e.g. Ridge M1 vs M2/M3/M4); **DM** for non-nested pairs (Flat vs Deep)
- Interpretability on M0-beating specs: modality gates + site/node attention

---



## Chapter 4 — Results *(~2,500–3,200 words)*

*Organised by RQ logic — not “Flat block then Deep block” only.*

### 4.1 Descriptive overview

- 257-week scored span; one-week-ahead Brent; M0 anchor; RMSE / MAE / skill / CW / DM



### 4.2 Flat results *(RQ1)*

- **No Flat model beats M0** (all negative skill)
- M3 > M1 (shipping helps nested); M2 ≈ no gain over M1 (RS weak)
- Key numbers (XGB RMSE / skill vs M0): M0 4.152; M1 −5.2%; M2 −6.9%; M3 −6.7%; M4 −8.6%
- CW significance ≠ beating M0



### 4.3 Deep results *(RQ1, deep arm)*

- M2 fails M0; **M3 beats M0** (gated +0.11%, xattn +0.74%); M4 does not clearly dominate M3
- Key numbers (gated): M1 −2.4%; M2 −2.4%; M3 **+0.11%**; M4 −1.3%



### 4.4 Flat vs Deep *(core RQ2)*

- Paired at matched sets (XGB Flat vs gated Deep):
  - M1: −5.2% vs −2.4% | M2: −6.9% vs −2.4% | M3: −6.7% vs **+0.11%** | M4: −8.6% vs −1.3%
- Deep advantage concentrated in shipping-inclusive multimodal settings



### 4.5 Robustness

- Flat: lookback, RS variants, shipping tiers, LOAO — pattern holds
- Deep: seeds, lookback, repr size, fusion type, early/late windows — gated M3 more stable than xattn across seeds



### 4.6 Interpretability *(RQ3)*

- Mean gates ~0.56 finance / 0.44 shipping; time-varying around RU war, EU oil ban, OPEC+ cut, Houthi attacks
- Node attention: Hormuz, Suez, Bab el-Mandeb, Panama, Cape of Good Hope
- Association ≠ causation

---



## Chapter 5 — Discussion *(~1,600–1,900 words)*



### 5.1 RQ1 — Do alternative data help?

- Flat: M0 best; shipping helps vs M1; RS does not
- Deep: small M3/M4 gain vs M0; RS secondary to shipping; absolute gains limited



### 5.2 RQ2 — Representation-level fusion vs flat?

- Paired evidence: Deep wins when shipping enters; finance-only Deep gain limited
- Advantage from preserving temporal / site / network structure + fusion — not different data



### 5.3 RQ3 — What does the model rely on?

- Deep M3 (and M4) gates + spatial attention; shipping weight rises in stress windows
- Model dependence, not causal identification



### 5.4 Implications

- Fair protocol matters as much as fusion design; nested M1 + absolute M0 contrasts jointly informative
- Null Flat results informative: early fusion ≠ automatic value



### 5.5 Limitations

- Weekly horizon; modest scored sample; proxy noise / reverse causality; frozen EO; graph choices; seed sensitivity (xattn)



### 5.6 Future research

- Longer history; richer AIS graph; missing-modality stress; other horizons / commodities

---



## Chapter 6 — Conclusion *(~400–700 words)*



### 6.1 Summary

- Flat: M0 wins; shipping helps vs M1 only
- Deep: small M3/M4 vs M0; RS weak
- Paired: Deep > Flat in shipping-inclusive multimodal settings
- Interpretability aligned with M0-beating specs



### 6.2 Contributions

- Nested multimodal comparison under one leakage-safe protocol
- Paired Flat–Deep at matched information sets
- Gating + spatial attribution where forecast evidence supports it



### 6.3 Final conclusion

- Shipping adds incremental value over finance; modality-aware Deep can yield small further gain vs M0; RS unstable
- Gains modest — useful but not breakthrough forecasting

---



## References



## Appendices

- **A.** Variable dictionaries; AOI / chokepoint lists; lag table; shipping graph edge definition
- **B.** Extra robustness tables & figures (lookback, LOAO, LOCHO, LOMO, water-mask RS, fusion matrix, seeds, early/late)
- **C.** Hyperparameter grids and locked Deep settings

