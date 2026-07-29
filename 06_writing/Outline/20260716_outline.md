# A Modality-Aware Spatio-Temporal Fusion Framework for Brent Crude Oil Forecasting Using Financial Time Series, Satellite Imagery and Maritime Networks

## Chapter 1 — Introduction *(~900–1,200)*

*background → gap → aim/RQ → contributions → structure*

### 1.1 Background and forecasting challenge

- **Brent’s importance:** Oil-price forecasting in energy economics, risk management and supply chains; Brent as the global pricing benchmark. Sample window: Friday-ending weekly Brent spot over **2019–2025** (COVID, 2022 energy shock, normalisation).
- **Strong naïve benchmark:** Weekly Brent is hard to beat out of sample; the no-change / random-walk forecast is a strong bar that any new data or model claim must clear.
- **Alternative-data potential:** AIS / PortWatch shipping and satellite RS (NTL, site optical indices / image embeddings) as physical-market proxies for trade, chokepoint congestion and infrastructure activity.
- **Practical challenges:** Signals are noisy, asynchronous, and possibly price-*responding* rather than price-*leading*; small weekly sample. Dominant practice still flattens sources into a wide table and early-fuses with finance (flat concat), discarding finance temporal structure, RS site structure and shipping network structure.



### 1.2 Research gap

1. **Incremental value of alternative data is unclear.** Few studies jointly report **nested increments** over a financial baseline *and* **absolute skill** against the random walk under leakage-safe evaluation — nested-only overstates alternative data; random-walk-only can hide economically meaningful but weak signals.
2. **Fusion architectures lack fair comparison.** Multi-source oil studies rarely compare flat concat vs representation-level modality-aware fusion under **one** shared protocol (common pattern: best-vs-best across families, or only one fusion style).
3. **Attribution lacks benchmark conditioning.** Interpretability is often detached from predictive value: heavy attribution for models that fail the relevant benchmark cannot support a “signals are useful” narrative.



### 1.3 Aim and research questions

*do the data help → does fusion architecture matter on the same data → what does the model rely on when value exists.*

**Aim:** Integrate frozen EO embeddings, modality-specific encoders, fusion modules and a leakage-safe rolling-origin protocol with formal forecast-comparison tests into one reproducible comparison framework — not a new neural operator, but integration + fair comparison.

- **RQ1:** Do remote-sensing and shipping indicators add incremental out-of-sample value over a financial baseline and the random-walk benchmark?
- **RQ2:** Does modality-aware representation-level fusion outperform flat feature fusion when both use the same underlying data and the same evaluation protocol?
- **RQ3:** Can modality-level interpretability reveal which signals the model relies on across different market conditions?



### 1.4 Contributions ？？？

1. **Primary (empirical / applied) — nested multimodal comparison.** The first systematic nested comparison of finance, satellite remote sensing and shipping within one weekly Brent design. A branching modality ablation isolates the out-of-sample increment from adding remote sensing (M2), adding shipping (M3), and combining both (M4), relative to the financial baseline and the random walk, identifying which sources help under which benchmark (RQ1).
2. **Method-integration — paired Flat vs Deep.** Under the same data and the same protocol, compares flat early fusion (Ridge / XGBoost) with modality-aware representation-level fusion (encoders + gated / cross-attention) at matched information sets (RQ2).
3. **Interpretability (supporting).** For models that improve on M0, uses gating and site–node attention to show which modalities and spatial nodes the model relies on under different market conditions (RQ3).



### 1.5 Dissertation structure

- Chapter **2** Literature review
- Chapter **3** Methodology
- Chapter **4** Results
- Chapter **5** Discussion
- Chapter **6** Conclusion

---



## Chapter 2 — Literature Review *(~2,500–3,500)*

---



## Chapter 3 — Methodology *(~2,500–3,000)*



### 3.1 Research design

- This study empirically compares Flat and Deep under one fair protocol, evaluating out-of-sample forecast performance across information sets and representation choices.
- Two architecture families share the same information sets, timeline, and metrics.
  - **Flat:** modality-derived features are combined in a common tabular representation and estimated using Ridge and XGBoost, providing classical early-fusion baselines.
  - **Deep:** each modality is processed by a modality-specific encoder before matched-dimensional latent representations are combined through representation-level fusion.
- Comparisons use the **M0** benchmark and the **M1–M4** information sets defined in §3.5. Within each architecture family, moving across M1–M4 isolates information increments; pairing Flat and Deep at matched information sets isolates representation / fusion. Together these contrasts separate *which* information is available from *how* it is represented. M0 anchors absolute skill.
- How each RQ is tested:
  1. **RQ1:** within each modelling family, compare M2, M3 and M4 with the finance-only M1 baseline, and compare all model forecasts with M0. Statistical tests are selected according to whether the competing forecast specifications are formally nested.
  2. **RQ2:** compare Flat and Deep pairwise at matched information sets, for example M3_Flat versus M3_Deep.
  3. **RQ3:** interpretability is applied to specifications that improve on M0 (Deep M3, and Deep M4 where relevant), using modality gates and site/node attention.



### 3.2 Prediction target and timeline

- **Forecasting objective:** predict the next-week Friday Brent spot price P_{t+1} using information available at forecast origin t.
- **Modelling target:** one-week log return r_{t+1}=\log(P_{t+1}/P_t).
- **Price reconstruction:** \hat P_{t+1\mid t}=P_t\exp(\hat r_{t+1\mid t}).
- **M0 equivalence:** the no-change price forecast \hat P_{t+1\mid t}=P_t corresponds to the zero-return forecast \hat r_{t+1\mid t}=0.
- Reported metrics and economic interpretation use reconstructed prices, not the internal return target.
- Calendar: Friday-ending weeks.
- Sample window: 2019–2025. Merged matrix ≈ 365 weeks. Common scored test span: 257 weeks (2021-01 to 2025-12).
- Forecast horizon: one week ahead. Directional accuracy is an auxiliary metric only and is not part of the training loss.



### 3.3 Data sources

- Three modalities: finance, remote sensing, and shipping. Flat and Deep use the same underlying sources; they differ in how each source is represented.
- **Finance:** weekly oil-market and macro series from EIA, FRED, Yahoo and related indicators. This is the baseline before alternative data are added.
- **Remote sensing:** same 11 AOIs for Flat and Deep, but different products from a shared Sentinel-2 optical source family. **Flat:** monthly Sentinel-2 optical indices (NDVI/NDWI/NDBI/BSI) plus VIIRS night-time lights, as site-level anomalies. **Deep:** frozen Prithvi-EO-2.0 embeddings from monthly Sentinel-2 image patches only — no VIIRS. (Details in Appendix A.)
- **Shipping:** PortWatch chokepoint and port tanker flows, plus AIS-based vessel activity. Flat uses tabular shipping features; Deep uses a weekly 17-node heterogeneous graph (11 AOIs + 6 chokepoints). (Details in §3.7 and Appendix A.)



### 3.4 Temporal alignment, lags, missingness

- All series are aligned to the Friday-ending weekly calendar.
- Predictors enter only after their real publication time, so the model never uses future information.
- Different sources have different release lags; examples include EIA and PortWatch about one week, and slower monthly series with longer buffers.
- Flat models fill missing values using only past observations. Deep models keep explicit masks for missing modalities or sites instead of silently filling them away.



### 3.5 Benchmark M0 and matched information sets M1–M4

- **M0** is the no-change / random-walk benchmark: next week’s price equals this week’s price. It is not trained and anchors absolute skill. It is not one of the modality sets.
- **M1–M4** are the modality (information) sets:
  - M1: finance only
  - M2: finance + remote sensing
  - M3: finance + shipping
  - M4: finance + remote sensing + shipping
- M2 and M3 are parallel branches from M1; M4 combines both.



### 3.6 Flat models

- Flat models concatenate all available numeric features for a given modality set into one weekly table, then flatten the last four weeks into one row.
- Two learners are used:
  - **Ridge:** linear model with L2 regularisation; a transparent linear early-fusion baseline.
  - **XGBoost:** non-linear tree ensemble. Captures interactions that Ridge misses, but still does not preserve modality structure.
- Both predict the log return and reconstruct price.
- Hyperparameters are chosen inside each training fold on a past validation slice only. Exact grids in Appendix C.



### 3.7 Deep models

- Same modality sets, calendar, and validation protocol as Flat. The difference is representation and fusion, not the forecast target.
- Each available modality is encoded into a matched-dimensional representation, then fused.
- **Finance:** A causal TCN models temporal dependencies in the weekly financial sequence.
- **Remote sensing:** Deep RS uses frozen Prithvi-EO embeddings from monthly Sentinel-2 patches (no VIIRS). Embeddings are kept per site and aggregated by temporal and site attention, so the site dimension is not collapsed before encoding.
- **Shipping:** Shipping is encoded as a weekly heterogeneous graph with 17 nodes (11 AOIs and 6 chokepoints). Edges combine time-varying voyage flows between AOIs with fixed AOI–chokepoint links; a GAT with temporal encoding aggregates this network into a modality representation. Exact edge construction is reported in Appendix A; GAT depth, heads and related layer settings are in Appendix C.
- Each encoder is specified by its inputs, network structure, outputs, and why that architecture fits the modality.
- Fusion options for RQ2: concat as a simple control; gated fusion as the main reported design; cross-attention as an advanced alternative.
- The fused representation maps to the same return/price target as Flat. Training details are in Appendix C.



### 3.8 Hyperparameter selection

- Flat: tune inside each training fold on past validation weeks only.
- Deep: lock a main configuration after sweeps; report sensitivity to seed, lookback, fusion type, representation size and regularisation in Results / Appendix C.
- Exact values such as representation size, GAT depth/heads and grids are in Appendix C. Flat and Deep select and lock hyperparameters under the same protocol so that the comparison remains fair.



### 3.9 Leakage-free validation protocol

- Expanding-window rolling-origin backtest: train only on past weeks, then forecast one week ahead.
- The first 104 weeks form the initial estimation and validation period and are not scored. The four-week lookback creates a separate sequence warm-up requirement. Thereafter models are refit every 13 weeks. Common scored test span: 257 weeks (2021-01 to 2025-12).
- Any scaling or filtering is fit inside the training fold only.
- Flat and Deep share the same fold calendar so architecture comparisons are fair.



### 3.10 Evaluation, tests, interpretability

- **Metrics.** Primary metrics on reconstructed price: RMSE and MAE for every comparison. Directional accuracy is auxiliary.
- **RMSE skill vs M0** (reported as a percentage in tables):

\mathrm{Skill}=100\times\left(1-\frac{\mathrm{RMSE}*{\mathrm{model}}}{\mathrm{RMSE}*{\mathrm{M0}}}\right).

Skill > 0 beats M0 on RMSE; = 0 matches M0; < 0 is worse than M0.

- **Comparison logic.** The study evaluates both incremental value versus M1 and absolute skill versus M0, and distinguishes information-set nesting from formal model nesting: a larger modality set does not automatically make two forecasts nested for testing.
- **Test choice.** Tests follow the forecast-specification relationship:
  - **Clark–West (2007):** an MSPE-adjusted test for whether a larger model improves on a smaller one when the smaller forecast specification is nested in the larger; used for nested increments (e.g. Ridge M1 versus Ridge M2/M3/M4 where nesting is justified).
  - **Diebold–Mariano (1995):** a test of equal predictive accuracy based on the mean loss differential between two forecasts; used for non-nested paired comparisons (e.g. Flat versus Deep, XGBoost or Deep settings that change hyperparameters or architecture). A small-sample adjustment is noted where relevant.
  - Every comparison also reports RMSE and MAE effect sizes versus M0 and, where relevant, versus M1.
- **Interpretability.** Applied to specifications that improve on M0 (Deep M3, and Deep M4 where relevant); diagnostics are modality gate weights and site/node attention.
- **Reproducibility.** Data are public or licensed; the pipeline is scripted.

---



## Chapter 4 — Results *(~2,500–3,200)*



### 4.1 Descriptive overview

This chapter reports out-of-sample one-week-ahead Brent price forecasts on the common scored evaluation span defined in Section 3.9 (257 weeks, January 2021–December 2025). 

### 4.2 Flat-model results

Table 4.1 summarises out-of-sample Flat performance on the common scored span. 

Every flat learned model has negative skill versus M0: under flat early fusion, the no-change benchmark retains the best absolute error. Contrasts within M1–M4 are nonetheless informative. 

Relative to finance-only M1, adding shipping in M3 improves performance, suggesting that shipping still carries some incremental information in the flat setting. 

By contrast, remote sensing under the main flat specification (M2) does not show a clear gain over M1. 

**Table 4.1 — Flat out-of-sample performance** *(n = 257)*


| Set | Content                  | Ridge RMSE | Ridge skill vs M0 | XGB RMSE | XGB skill vs M0 |
| --- | ------------------------ | ---------- | ----------------- | -------- | --------------- |
| M0  | no-change benchmark      | 4.152      | —                 | 4.152    | —               |
| M1  | finance only             | 4.256      | −2.5%             | 4.368    | −5.2%           |
| M2  | finance + remote sensing | 4.414      | −6.3%             | 4.440    | −6.9%           |
| M3  | finance + shipping       | 4.430      | −6.7%             | 4.429    | −6.7%           |
| M4  | finance + RS + shipping  | 4.525      | −9.0%             | 4.507    | −8.6%           |




### 4.3 Deep-model results

Table 4.2 summarises out-of-sample Deep performance by information set. For single-modality M1 only the finance encoder applies; its result is reported in the gated column for comparability with later multimodal rows. 

Finance plus remote sensing (M2) still fails to beat M0, so remote sensing remains weak in the deep setting as well. 

By contrast, finance plus shipping (M3_Deep) improves on M0 under both gated and cross-attention fusion. Clearing the no-change benchmark is substantively meaningful for weekly Brent, even though skill versus M0 remains modest in magnitude (about +0.11% under gated fusion and +0.74% under cross-attention). 

M4 does not clearly dominate M3: adding remote sensing on top of finance and shipping often fails to reduce absolute error further. 

The main text reports gated fusion; cross-attention is included as a comparative architecture. Encoder-concatenation and the full fusion matrix are given in the appendix.

**Table 4.2 — Deep out-of-sample performance** *(gated = main reported fusion)*


| Set | Content                  | Gated RMSE | Gated skill vs M0 | Xattn RMSE | Xattn skill vs M0 |
| --- | ------------------------ | ---------- | ----------------- | ---------- | ----------------- |
| M0  | no-change benchmark      | 4.152      | —                 | 4.152      | —                 |
| M1  | finance only             | 4.250      | −2.4%             | —          | —                 |
| M2  | finance + remote sensing | 4.253      | −2.4%             | —          | —                 |
| M3  | finance + shipping       | 4.147      | **+0.11%**        | 4.121      | **+0.74%**        |
| M4  | finance + RS + shipping  | 4.205      | −1.3%             | 4.147      | +0.12%            |




### 4.4 Flat versus Deep

Table 4.3 compares Flat and Deep at matched information sets, holding data content fixed in order to isolate representation and fusion. 

Deep gains are clearest in multimodal settings, especially once shipping enters; replacing only the finance branch with a deep encoder yields a weaker improvement. 

Representation-level, modality-aware fusion therefore outperforms flat counterparts in selected multimodal settings, particularly with shipping.

**Table 4.3 — Paired Flat versus Deep** 


| Pair | Flat RMSE | Deep RMSE | skill flat | skill deep |
| ---- | --------- | --------- | ---------- | ---------- |
| M1   | 4.368     | 4.250     | −5.2%      | −2.4%      |
| M2   | 4.440     | 4.253     | −6.9%      | −2.4%      |
| M3   | 4.429     | 4.147     | −6.7%      | +0.11%     |
| M4   | 4.507     | 4.205     | −8.6%      | −1.3%      |




### 4.5 Robustness and sensitivity

For Flat models, the checks cover lookback length, remote-sensing feature variants, shipping feature tiers, and leave-one-modality-out analysis for M4. The results align with the main text: no flat model beats M0; shipping still helps relative to finance-only M1; adding remote sensing can raise absolute RMSE. 

For Deep models, the checks cover random seeds, lookback length, representation size, fusion type, and early versus late test windows. The gated finance-plus-shipping configuration provides the more stable small positive skill; cross-attention can look stronger on a single seed but varies more across seeds. 

At matched multimodal information sets, the Deep advantage over Flat remains under these checks, especially in shipping-inclusive settings.

### 4.6 Interpretability

Over the test span, mean gate weights are about 0.56 for finance and 0.44 for shipping; the allocation evolves over time and adjusts around events such as the Russia–Ukraine war, the EU Russia oil-ban announcement, the OPEC+ surprise cut and Houthi Red Sea attacks. 

Shipping node attention concentrates on major chokepoints — the highest mean weights are Hormuz, Suez, Bab el-Mandeb, Panama and the Cape of Good Hope. 

Modality gates capture modality-level dependence and node attention captures spatial dependence; together they describe model dependence rather than causal effects on oil prices.

## Chapter 5 — Discussion *(~1,600–1,900)*



### 5.1 RQ1 — Do alternative data help?

- Under Flat, no model beats M0: weekly Brent remains hard to forecast beyond the no-change benchmark.
- Within the Flat family, shipping still adds value: M3 improves on finance-only M1, while remote sensing shows no clear gain over M1; full flat RS features add little, and sparse or cleaned variants do not overturn this main result.
- Under Deep, finance + shipping shows only a **small** improvement over M0; adding remote sensing on top often brings no further gain.
- Overall, shipping has value relative to finance, but absolute gains remain limited and do not support strong economic claims under the weekly design.



### 5.2 RQ2 — Does representation-level fusion beat flat fusion?

- At matched information sets, Deep outperforms Flat most clearly once shipping enters.
- Replacing Flat with Deep on finance alone yields limited gains; the architecture gap opens mainly in multimodal settings.
- That advantage comes from preserving temporal, site and network structure, and from how modalities are fused — not from giving Deep a different data mix.
- The Deep advantage is concentrated in selected shipping-inclusive settings and does not extend across all specifications.



### 5.3 RQ3 — What does the model rely on when value exists?

- Interpretability evidence comes mainly from Deep M3 (and Deep M4).
- Modality gates correspond to modality-level dependence; node and site attention correspond to spatial dependence.
- Shipping gate weight tends to rise in disruption or chokepoint-stress windows; spatial attention often concentrates on major routes and export nodes.
- These patterns describe model dependence rather than causal effects on prices; a high shipping gate alone does not identify which port or chokepoint is attended.



### 5.4 Implications

- In alternative-data fusion research, a fair shared evaluation protocol matters as much as proposing a new fusion module for how credible the conclusions are.
- In this study, the nested M1 contrasts and the absolute M0 contrasts jointly show that shipping can help relative to finance while absolute gains remain small.
- Gains over M0 are too small to support a forecasting-breakthrough conclusion.
- Null or near-null Flat results are themselves informative: early fusion of alternative data is not automatically useful.



### 5.5 Limitations

- The study uses a weekly horizon and a modest scored sample after warm-up.
- Alternative-data proxies are noisy and may respond to prices as well as lead them.
- Frozen EO embeddings, shipping-graph construction, and missingness rules all affect Deep results.
- Some Deep configurations, especially cross-attention, are sensitive to random seeds.



### 5.6 Future research

- Extend the best Deep specifications to longer history and more seeds.
- Enrich the shipping graph and strengthen stress tests under missing modalities.
- Apply the same Flat–Deep protocol to other forecast horizons or related energy commodities.

---



## Chapter 6 — Conclusion *(~400–700)*



### 6.1 Summary of findings

- Under Flat, M0 is best and M1–M4 do not beat the no-change benchmark; within the Flat family, shipping still improves on finance only.
- Under Deep, M0 remains strong; Deep M3 / M4 show only a small improvement over M0, and remote sensing is secondary to shipping.
- At matched multimodal information sets, Deep outperforms Flat, most clearly in shipping-inclusive settings.
- For models that improve on M0, modality gates and spatial attention show when and where shipping is relied upon.



### 6.2 Contributions

- A systematic nested comparison of finance, remote sensing and shipping for weekly Brent under one leakage-safe protocol, separating incremental value versus the finance baseline from absolute performance versus the no-change benchmark.
- A paired Flat versus Deep comparison at matched information sets, separating differences in data content from differences in representation and fusion design.
- Gating and spatial attribution on models that improve on M0, so explanation stays aligned with forecast evidence.



### 6.3 Final conclusion

- In this weekly Brent design, shipping adds incremental value over finance, and modality-aware Deep fusion can yield a small further improvement over the no-change benchmark; remote sensing does not show stable help.
- These gains remain modest: alternative data and representation-level fusion can be useful, but not enough to support strong forecasting or strong economic claims.

---



## References



## Appendices

- **A.** Full variable dictionaries (M1–M4); AOI / chokepoint node lists; lag table; shipping graph edge definition (voyage flows, AOI–chokepoint links, adjacency handling, edge-weight transform)
- **B.** Extra result / robustness tables & figures (lookback, LOAO, LOCHO, LOMO, water-mask RS variant, fusion matrix, seeds, early/late)
- **C.** Hyperparameter grids and locked Deep settings (representation size, GAT layers/heads, seeds, software / config paths)

