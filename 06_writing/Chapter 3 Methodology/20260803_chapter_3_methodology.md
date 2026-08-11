# Chapter 3 — Methodology *(~3,200)*

## 3.1 Research design

This chapter sets out how the study answers the research questions in Section 1.2. In brief, every learned forecast is judged against a simple no-change benchmark in which next week’s Brent price equals this week’s price. The study then asks whether remote sensing and shipping add useful information beyond financial time series, and whether modelling those inputs as one weekly table differs from encoding each data type separately before combining them. All comparisons use the same weekly forecast dates, sample window and evaluation rules, so that changes in the data can be separated from changes in how the data are modelled.

The no-change benchmark is denoted M0. At each forecast origin \(t\), M0 sets the one-week-ahead Brent price forecast equal to the current weekly price

\[
\hat{P}_{t+1|t}=P_t.
\]

M0 needs no parameter estimation and contains no predictors. It is a reference forecast, not one of the information sets below. Every learned model is compared with M0 on the same evaluation sample. A model improves on M0 when its out-of-sample RMSE is lower. Once models predict log returns and then reconstruct prices, M0 is the same as forecasting a zero return.

The predictors are organised into four information sets. M1 uses financial time series only (financial, macroeconomic and oil-market series). M2 adds remote sensing to M1; M3 adds shipping to M1; and M4 adds both. M2 and M3 are parallel additions to M1, not successive steps on one ladder; M4 combines both additions.

**Table 3.1 — Information sets**

| Set | Content                                                                     |
| --- | --------------------------------------------------------------------------- |
| M1  | Financial time series only (financial, macroeconomic and oil-market series) |
| M2  | M1 + remote sensing                                                         |
| M3  | M1 + shipping                                                               |
| M4  | M1 + remote sensing + shipping                                              |

Comparing M2 with M1 measures the contribution of remote sensing when added alone. Comparing M3 with M1 measures the contribution of shipping. Comparing M4 with M1 evaluates their joint contribution. Two further comparisons ask whether each source still helps once the other is already included. M4 versus M3 tests remote sensing given shipping, and M4 versus M2 tests shipping given remote sensing. All comparisons keep the same one-week horizon and Friday weekly calendar.

Two model families are applied to these information sets. The Flat family puts all selected predictors into one weekly table—stacking recent weeks into a single row—and fits Ridge and XGBoost. This early joining of features is called flat feature fusion. The Deep family keeps each data type separate at first. Financial series, remote-sensing imagery and shipping-network inputs each pass through their own encoder, and the outputs are then combined. The main Deep design learns how much weight to give each data type (gated fusion). Simple joining of the encoder outputs, and an attention-based alternative, are kept as comparisons. Flat and Deep share sites and forecast dates, but the remote-sensing products they use are not identical. That difference is treated as part of the Flat–Deep contrast.

Paired Flat–Deep comparisons measure the overall difference between two modelling strategies. One fits a single weekly table directly. The other encodes each data type separately and then combines the results. The information set, forecast dates and evaluation sample are held constant. The two families also differ in model class and capacity, so these comparisons are not read as isolating the fusion method alone. To assess fusion itself, the Deep family compares simple concatenation, gated fusion and cross-attention while keeping the encoders and input data fixed. These two layers answer RQ2. The first contrasts Flat versus Deep on the same information set (for example M3_Flat versus M3_Deep). The second contrasts fusion variants within Deep.

The research questions map onto the design as follows. RQ1 uses the M1–M4 comparisons within each model family, together with the comparison of every learned forecast against M0. The M4–M3 and M4–M2 contrasts ask whether either added source still helps once the other is present.

RQ3 is restricted to Deep models that improve on M0 by the pre-defined criterion. Those models are identified from the results; they are not chosen in advance by name. For them, the study reports how much weight the model places on finance, remote sensing and shipping, and which sites or network nodes receive more attention under different market conditions. These quantities describe what the model relies on. They are not treated as proof of causal importance.

Figure 3.1 summarises the design. M0 is the no-change reference. M1 branches into M2 (plus remote sensing), M3 (plus shipping) and M4 (both). Flat and Deep are estimated and evaluated on the same information sets under a shared expanding-window procedure.

*[Figure 3.1 — Research design flowchart. M0 benchmark; M1→M2/M3/M4 branching; paired Flat vs Deep; link to expanding-window evaluation.]*

## 3.2 Prediction target and timeline

Let \(P_t\) denote the last available daily Brent spot-price observation in week \(t\), where each week ends on Friday, measured in US dollars per barrel. The quantity reported in the results is the one-week-ahead price \(P_{t+1}\). Models are not trained directly on the price level. They predict the one-week logarithmic return

\[
r_{t+1}=\log\!\left(\frac{P_{t+1}}{P_t}\right)
\]

and reconstruct the price forecast as

\[
\hat{P}_{t+1|t}=P_t\exp\!\left(\hat{r}_{t+1|t}\right).
\]

Log returns are used to reduce the strong persistence in the price level and to express the forecasting task in terms of proportional weekly changes. RMSE, MAE and skill versus M0 are computed from the reconstructed price forecasts. Directional accuracy is reported separately as an auxiliary statistic based on the sign of the predicted and observed returns. Under this mapping, the no-change benchmark \(\hat{P}_{t+1|t}=P_t\) is exactly the same as forecasting a zero return \(\hat{r}_{t+1|t}=0\).

All series are organised on a Friday-ending weekly calendar. The modelling window covers 2019–2025 and provides a common weekly index of 365 observations (4 January 2019 to 26 December 2025). Flat models use a merged weekly feature table on this index. Deep models use the same dates, but keep financial, remote-sensing and shipping inputs in their own sequence or graph form rather than one shared table. The first 104 weeks are reserved for initial estimation. Three further weeks are needed to form the first four-week input sequence, and the final week is excluded because \(P_{t+1}\) is unavailable. This leaves 257 forecast origins for evaluation, from 22 January 2021 to 19 December 2025. At each origin \(t\), forecasts may use only information that was actually available at that forecast date.

## 3.3 Geographic scope and monitoring sites

Because the prediction target is the global Brent benchmark rather than a local physical cargo price at a single terminal, the study does not use one contiguous study region. Spatial information instead comes from eleven oil-infrastructure monitoring sites and six maritime chokepoints. Together they cover major supply, transit and demand locations in the international oil system. Figure 3.2 places these sites and chokepoints on a world map. Full site names, coordinates, patch sizes and graph edge definitions are in Appendix A.

The eleven sites are ports, refineries and export terminals chosen for infrastructure capacity, geographic and supply-chain coverage, and observability in the available satellite products. In the Flat pathway, remote-sensing features are summarised inside a 5-km circular buffer around each site. In the Deep pathway, image patches are cut around each site. Patch size follows facility type and local spatial constraints. Ports use larger patches, refineries intermediate ones, and terminals smaller ones.

The shipping network adds six chokepoints to the eleven sites. They are the Strait of Hormuz, the Suez Canal, the Strait of Malacca, Bab el-Mandeb, the Panama Canal and the Cape of Good Hope. This gives a weekly network with seventeen nodes. Two kinds of link are used. First, directed site-to-site links record observed voyages between the eleven AOIs from Global Fishing Watch port-visit sequences. The weekly link weight is the voyage count for each origin–destination pair, so these links change from week to week. Second, fixed site–chokepoint links connect each AOI to the chokepoint(s) on its main oil-trade corridor. These links are set in advance. They are not inferred from weekly vessel tracks or nearest-neighbour distance. PortWatch and Automatic Identification System (AIS) measures enter mainly as node attributes rather than as pairwise links.

Flat and Deep use the same underlying port and chokepoint observations and the same weekly forecast dates. Flat turns them into tabular predictors. Deep keeps the node structure and the links between nodes.

*[Figure 3.2 — World map of 11 oil-infrastructure AOIs and 6 maritime chokepoints used for remote-sensing and shipping inputs.]*

## 3.4 Data sources

Three data blocks enter the design. They are financial time series (financial, macroeconomic and oil-market series), satellite remote sensing, and maritime shipping. Flat and Deep share the same Friday calendar and the same study sites and shipping-network scope. The product used for each block before it enters a model may differ (Table 3.2). For Flat models, the predictors are those in the merged weekly feature table. Deep models use the same weekly dates, but keep each block in its own input form.

**Table 3.2 — Datasets, variables and sources**

| Modality                   | Dataset / product                                      | Key variables                                                               | Source                                       |
| -------------------------- | ------------------------------------------------------ | --------------------------------------------------------------------------- | -------------------------------------------- |
| Financial time series (M1) | Oil-market and macro weekly series                     | Prices, inventories, production, interest rates, GPR and related indicators | EIA, FRED, Yahoo Finance and related sources |
| Remote sensing (Flat)      | Sentinel-2 optical indices and VIIRS night-time lights | Site-level anomalies at 11 AOIs (NDVI, NDWI, NDBI, BSI; NTL)                | Sentinel-2; VIIRS                            |
| Remote sensing (Deep)      | Frozen Prithvi-EO-2.0 embeddings                       | Monthly Sentinel-2 image-patch embeddings at the same 11 AOIs (no VIIRS)    | Prithvi-EO-2.0 / Sentinel-2                  |
| Shipping (Flat)            | PortWatch and AIS tabular features                     | Port and chokepoint tanker flows; vessel-activity features                  | IMF PortWatch; AIS                           |
| Shipping (Deep)            | Same sources, represented as a graph                   | Weekly heterogeneous graph with 17 nodes (11 AOIs and 6 chokepoints)        | PortWatch; AIS                               |

The financial block is assembled from weekly oil-market and macro-financial series from the US Energy Information Administration (EIA), Federal Reserve Economic Data (FRED), Yahoo Finance and related scholarly indicators. The series include crude prices and spreads, inventories, production and refinery activity, volatility and risk measures, interest rates, exchange rates, futures-based oil indicators and geopolitical risk. This block is M1 before remote sensing or shipping is added. In the Deep pathway it is treated as one input stream for the finance encoder, even though the predictors extend beyond prices alone.

Remote-sensing inputs are observed over the eleven AOIs. Flat and Deep share these sites but use different products from a common Sentinel-2 optical source family. Flat remote sensing uses monthly Sentinel-2 optical indices (NDVI, NDWI, NDBI and BSI) together with VIIRS night-time lights, converted to site-level anomalies. Deep remote sensing uses frozen Prithvi-EO-2.0 embeddings extracted from monthly Sentinel-2 image patches at the same AOIs and excludes VIIRS. Early Deep trials that included VIIRS night-time lights were noisy and added little useful signal; keeping them worsened performance, so VIIRS was dropped from the Deep pathway. Systematic numerical ablations from those early trials were not retained. This choice is also consistent with evidence that night-time lights capture cross-sectional brightness differences better than within-site temporal variation (Small, 2021). The reported Deep pathway therefore uses Sentinel-2 image embeddings only. Shared AOIs keep spatial coverage matched across pathways. Differences in product and representation form part of the Flat–Deep contrast and limit a pure architecture comparison on identical remote-sensing features.

Shipping inputs combine IMF PortWatch measures of chokepoint and port tanker flows with AIS-derived vessel-activity indicators for the network in Section 3.3. In the Flat pathway these signals enter as weekly table features. In the Deep pathway they enter as the seventeen-node network already described. In both pathways, shipping is treated as a proxy for physical trade and congestion rather than as a direct measure of next week’s price.

**Ethical considerations.** The study uses secondary aggregate data only and does not involve human participants. It was approved under the UCL low-risk ethics process. All datasets are used under their published research terms of use.

## 3.5 Temporal alignment, lags, missingness

All series are aligned to the Friday-ending weekly calendar. Predictors enter only after their real publication time, so the model never uses future information at any forecast origin. Release lags differ across sources. EIA and PortWatch series typically become available with a lag of about one week, while slower monthly series require longer buffers before they are allowed to enter. Missingness is handled differently in the two pathways. Flat models fill missing values using only past observations within the available history. Deep models keep an explicit missing marker for absent modalities or sites instead of filling them in silently, so the model can see what was unavailable at that forecast date.

## 3.6 Flat models

Flat models implement flat feature fusion. For a given information set, all available numeric features are concatenated into one weekly table, and the most recent four weeks are flattened into a single row for each forecast origin. Two learners are estimated on this table. Ridge is a linear model with L2 regularisation (Hoerl and Kennard, 1970) and serves as a transparent linear baseline that combines features at the outset. XGBoost is a non-linear gradient-boosted tree ensemble (Chen and Guestrin, 2016) that can capture interactions missed by Ridge, but still does not preserve modality-specific structure. Regularised linear and tree-based learners are both common in short-horizon oil-price forecasting with large predictor sets (Costa et al., 2021; Yılmaz and Zehir, 2026); they are used here as Flat baselines rather than as a claim that either algorithm is universally optimal. Both models predict the one-week-ahead log return and then reconstruct price. Hyperparameters are chosen inside each training fold on past validation weeks only. Exact search grids are in Appendix C.

## 3.7 Deep models

Deep models use the same information sets, Friday calendar and validation protocol as Flat. The difference is how inputs are represented and combined, not the forecast target. Each available modality is first turned into a fixed-size representation; those representations are then combined into one forecast. The three encoders are described below by input, purpose and output.

**Finance encoder.** The input is the weekly financial time series block (M1), including prices, inventories, macro and oil-market indicators. These series are dense temporal sequences, so the encoder must learn short-run dependence without using future weeks. The output is one finance representation for the forecast origin. The architecture is a causal temporal convolutional network (TCN; Bai, Kolter and Koltun, 2018). Causal convolutions prevent look-ahead within the sequence, and TCNs have been competitive for short-horizon crude-price forecasting relative to several deep and tree baselines (Foroutan and Lahmiri, 2024).

**Remote-sensing encoder.** The input is monthly Sentinel-2 image-patch embeddings at the eleven AOIs, extracted with a frozen Prithvi-EO-2.0 model (VIIRS night-time lights are excluded). Sites are kept distinct until after encoding, so spatial location is not collapsed into a single early average. The output is one remote-sensing representation for the forecast origin, formed by weighting across time and sites. The architecture uses frozen embeddings plus temporal and site attention.

**Shipping encoder.** The input is the weekly seventeen-node shipping network from Section 3.3. Shipping information is relational because ports and corridors are linked, so a graph model fits better than a flat row of counts. The output is one shipping representation for the forecast origin. The architecture is a graph attention network (GAT; Veličković et al., 2018) with temporal encoding. Graph neural networks have been used to model crude-oil and vessel-traffic networks as relational, time-varying processes (Ouyang et al., 2022; Liang et al., 2022). GAT is used here because neighbour weights fit a sparse port–chokepoint network and later support site-level interpretation (RQ3). Layer settings are in Appendix C.

**Fusion (RQ2).** Once each available modality has a representation, three ways of combining them are compared. Simple concatenation joins the representations without adaptive weighting and serves as a control. Gated fusion is the main reported design. It learns how much weight to give each modality. Cross-attention is retained as an advanced alternative that lets modalities attend to one another. The fused representation is mapped to the same return and price target as Flat. Training details are in Appendix C.

## 3.8 Hyperparameter selection

Hyperparameters are selected under a shared protocol so that Flat–Deep comparisons remain fair. For Flat models, tuning uses only past validation weeks inside each training fold. For Deep models, searching the full architecture at every fold is too costly. A limited search is run first, then one main configuration is fixed for the primary results. Sensitivity checks follow. Exact grids and layer settings are in Appendix C.

## 3.9 Validation protocol

Evaluation uses an expanding window. At each forecast origin the model is trained only on past weeks and then produces a one-week-ahead forecast. This design prevents the use of future information in training or preprocessing. The first 104 weeks form the initial estimation and validation period and are not included in the evaluation metrics. Forming the first four-week input sequence requires three additional weeks before the first evaluated origin. Thereafter models are refit every 13 weeks. The common evaluation span covers 257 weeks from 22 January 2021 to 19 December 2025. Any scaling or filtering is fit on the training period only. Flat and Deep share the same evaluation calendar, so architecture comparisons hold the evaluation design fixed.

Figure 3.3 shows this expanding-window design.

*[Figure 3.3 — Expanding-window evaluation flowchart.]*

## 3.10 Evaluation, tests, interpretability

Primary metrics are computed on reconstructed prices. Every comparison reports RMSE and MAE. Directional accuracy is retained only as an auxiliary measure. Relative performance versus M0 is summarised by RMSE skill—the percentage improvement in RMSE relative to M0—reported as a percentage in the result tables.

\[
\mathrm{Skill}=100\times\left(1-\frac{\mathrm{RMSE}_{\mathrm{model}}}{\mathrm{RMSE}_{\mathrm{M0}}}\right).
\]

Skill greater than zero means the model beats M0 on RMSE. Skill equal to zero matches M0. Skill less than zero is worse than M0.

The study reports both absolute skill versus M0 and incremental value versus M1. Statistical tests are chosen by the type of comparison, not by the size of the modality set alone. Adding remote sensing or shipping enlarges the information set, but that does not by itself make two forecasts nested for testing. When one forecast specification is nested in another—for example Ridge M1 versus Ridge M2, M3 or M4 under the same learner—Clark–West (2007) is used to test whether the larger model improves mean squared prediction error. When the comparison is not nested—for example Flat versus Deep, or XGBoost versus a Deep setting that changes hyperparameters or architecture—Diebold–Mariano (1995) is used to test equal predictive accuracy. A small-sample adjustment is noted where relevant. Every comparison also reports RMSE and MAE differences versus M0 and, where relevant, versus M1.

Interpretability diagnostics are applied only to specifications that improve on M0. The main cases are Deep M3 and, where relevant, Deep M4. The diagnostics report modality gate weights together with site or node attention.

---
