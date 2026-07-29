# Chapter 3 — Methodology *(~2,500–3,000)*

## 3.1 Research design

This chapter specifies the empirical design used to answer the research questions in Section 1.2. The design has three components that remain fixed across comparisons: a no-change benchmark, four information sets organised around a common financial time series block, and two model families evaluated using the same weekly forecast dates, sample window and evaluation rules. This structure separates changes in the information supplied to a model from changes in how that information is represented and combined.

The no-change benchmark, denoted M0, is defined first. At each forecast origin \(t\), M0 sets the one-week-ahead Brent price forecast equal to the current weekly price:

\[
\hat{P}_{t+1|t}=P_t.
\]

M0 is a driftless random-walk forecast and requires no parameter estimation. It is a reference forecast rather than an information set: it contains no predictors. Every learned model is compared with M0 on the same evaluation sample. A specification is described as improving on M0 when it achieves a lower out-of-sample RMSE on that sample; formal forecast-comparison tests are reported separately. The weekly price \(P_t\), the log-return training target and the mapping between return-space forecasts and reconstructed prices are defined in Section 3.2; under that mapping, M0 coincides with a zero-return forecast.

The predictors are organised into four information sets around a common financial time series block. M1 contains financial time series only (financial, macroeconomic and oil-market series). M2 adds remote-sensing data to M1; M3 adds shipping data to M1; and M4 adds both remote sensing and shipping. M2 and M3 are parallel extensions of M1 rather than a single nested sequence, while M4 contains both branches.

**Table 3.1 — Information sets**

| Set | Content |
| --- | --- |
| M1 | Financial time series only (financial, macroeconomic and oil-market series) |
| M2 | M1 + remote sensing |
| M3 | M1 + shipping |
| M4 | M1 + remote sensing + shipping |

Comparing M2 with M1 measures the contribution of remote sensing when added alone, while M3 versus M1 measures the contribution of shipping. M4 versus M1 evaluates their joint contribution. Two additional comparisons are used to assess conditional contributions: M4 versus M3 tests whether remote sensing adds value once shipping is present, and M4 versus M2 tests whether shipping adds value once remote sensing is present. All comparisons retain the same forecast horizon and weekly calendar.

Two model families are applied to these information sets. In the Flat family, the selected inputs are converted into weekly features, placed in a single predictor table and used to fit Ridge and XGBoost models. This is referred to as flat feature fusion. In the Deep family, financial, remote-sensing and shipping inputs are processed by modality-specific encoders before their representations are combined. Gated representation-level fusion is the primary specification. Representation concatenation and cross-attention are included as comparison variants. Representation concatenation occurs after the modality-specific encoders and is therefore distinct from the feature-level concatenation used in the Flat family. Encoder and fusion details are provided in Section 3.7.

Matched Flat–Deep comparisons evaluate the overall difference between flat tabular modelling and modality-specific representation learning when the information set, forecast dates and evaluation sample are held constant. Because the two families also differ in model class and capacity, these comparisons are not interpreted as isolating the fusion mechanism alone. Within the Deep family, comparisons among representation concatenation, gated fusion and cross-attention assess the fusion mechanism while keeping the modality-specific encoders and input data fixed.

The research questions map onto these comparisons. RQ1 is assessed through comparisons of M2, M3 and M4 with M1 within each model family, together with comparisons of every learned forecast against M0. The M4–M3 and M4–M2 contrasts provide additional evidence on whether either alternative modality adds value in the presence of the other. The evaluation section specifies the test used for each forecast pair, distinguishing nested from non-nested forecast-model comparisons rather than relying on predictor-set nesting alone.

RQ2 is assessed at two levels. Matched Flat–Deep pairs, such as M3_Flat and M3_Deep, compare the two overall modelling strategies on the same information set. Comparisons among the Deep fusion variants then examine whether gated or cross-attention fusion improves on representation concatenation when the encoders are held fixed.

RQ3 is restricted to Deep specifications that satisfy the pre-defined M0 improvement criterion. The qualifying specifications are identified in Chapter 4 rather than selected in advance by model name. Modality-gate weights are used to examine the model’s relative reliance on the three modalities, while site or node attention identifies which spatial locations receive greater weight under different market conditions. These quantities are interpreted descriptively as evidence about model reliance, not as evidence of causal importance.

Figure 3.1 summarises the research design as a flowchart: the no-change benchmark M0 anchors absolute forecast skill; financial time series M1 branches into M2 (plus remote sensing), M3 (plus shipping) and M4 (both); and Flat and Deep families are estimated at matched information sets under a shared rolling-origin protocol (Section 3.9).

*[Figure 3.1 — Research design flowchart: M0 benchmark; M1→M2/M3/M4 branching; paired Flat vs Deep; link to rolling-origin evaluation.]*

## 3.2 Prediction target and timeline

Let \(P_t\) denote the last available daily Brent spot-price observation in week \(t\), where each week ends on Friday, measured in US dollars per barrel. The quantity reported in the results is the one-week-ahead price \(P_{t+1}\). Models are not trained directly on the price level. They predict the one-week logarithmic return

\[
r_{t+1}=\log\!\left(\frac{P_{t+1}}{P_t}\right)
\]

and reconstruct the price forecast as

\[
\hat{P}_{t+1|t}=P_t\exp\!\left(\hat{r}_{t+1|t}\right).
\]

Log returns are used to reduce the strong persistence in the price level and to express the forecasting task in terms of proportional weekly changes. RMSE, MAE and benchmark skill scores are computed from the reconstructed price forecasts. Directional accuracy is reported separately as an auxiliary statistic based on the sign of the predicted and observed returns. Under this mapping, the no-change benchmark \(\hat{P}_{t+1|t}=P_t\) coincides exactly with the zero-return forecast \(\hat{r}_{t+1|t}=0\).

All series are organised on a Friday-ending weekly calendar. The modelling window covers 2019–2025 and provides a common weekly index of 365 observations (4 January 2019 to 26 December 2025). Flat models use a merged weekly feature table on this index; Deep models use the same dates but modality-specific sequence and graph tensors rather than a single flat matrix. After reserving the first 104 weeks for initial estimation, allowing three additional weeks to form the first four-week input sequence, and excluding the final week because \(P_{t+1}\) is unavailable, 257 scored forecast origins remain, from 22 January 2021 to 19 December 2025. At each origin \(t\), forecasts may use only information that was actually available at that forecast date; publication lags and alignment rules are set out in Section 3.5.

## 3.3 Geographic scope and monitoring sites

Because the prediction target is the global Brent benchmark rather than a local physical cargo price at a single terminal, the study has a distributed rather than contiguous geographic scope. Spatial information enters the models through two components: eleven remote-sensing areas of interest (AOIs) and a seventeen-node shipping graph.

The eleven monitoring locations are Houston Ship Channel (USA); Port of Rotterdam (Netherlands); Ningbo–Zhoushan Port (China); Jamnagar Refinery (India); Jurong Island (Singapore); Ulsan Refinery (South Korea); Basra Oil Terminal (Iraq); Fujairah Oil Terminal (United Arab Emirates); Kharg Island Terminal (Iran); Ras Tanura Terminal (Saudi Arabia); and Yanbu Export Terminal (Saudi Arabia). The sites were selected to balance infrastructure capacity, geographic and supply-chain coverage, and observability in the available satellite products. Together they cover major supply, transit and demand locations in the international oil system. These locations define the geographic AOIs. Flat remote-sensing features are summarised inside a circular buffer with a 5-km radius around each site. Deep remote-sensing images are extracted using site-specific patches whose sizes reflect the facility type and local spatial constraints (larger for ports, intermediate for refineries and smaller for terminals); the exact coordinates and patch radii are reported in Appendix A.

The shipping component adds six maritime chokepoints—Strait of Hormuz, Suez Canal, Strait of Malacca, Bab el-Mandeb, Panama Canal and the Cape of Good Hope—to the eleven AOIs, yielding a heterogeneous graph with seventeen nodes. AOI and chokepoint nodes have different feature spaces. Two edge types are used. Directed AOI-to-AOI edges are built from Global Fishing Watch port-visit sequences that identify vessel movements between the eleven AOIs; the weekly edge weight is the observed voyage count \(n_{\mathrm{voyages}}\) for each origin–destination pair, so these edges change from week to week. These O–D edges are distinct from PortWatch and AIS presence or transit indicators, which enter mainly as node features rather than as pairwise links.

AOI–chokepoint edges are separate. They are fixed, undirected and binary, and they remain present every week. They are not inferred from weekly vessel tracks, nearest-neighbour distance or co-occurrence statistics. Instead, each AOI is linked in advance to the chokepoint(s) that define its primary oil-trade corridor role in the international supply chain: Persian Gulf export and transit sites (Fujairah, Ras Tanura, Basra, Kharg) to Hormuz; East and South-East Asian import-route sites (Jurong, Ningbo–Zhoushan, Ulsan) to Malacca; the Red Sea export terminal at Yanbu to Suez and Bab el-Mandeb; Rotterdam to Suez and the Cape of Good Hope as the Europe-bound and Cape alternative corridors; and Houston to Panama. Jamnagar is retained as a demand-side AOI without a dedicated chokepoint link. The full AOI–chokepoint edge list and any edge-weight transforms are reported in Appendix A. Node features also vary by week. Encoder details are given in Section 3.7. The Flat and Deep pathways use the same underlying port- and chokepoint observations and the same weekly forecast dates. The Flat pathway aggregates them into tabular predictors, whereas the Deep pathway retains their node-level organisation and graph relationships. Figure 3.2 maps the AOIs and chokepoints.

*[Figure 3.2 — Map of 11 oil-infrastructure AOIs and 6 maritime chokepoints used for remote-sensing and shipping inputs.]*

## 3.4 Data sources

Three modality blocks enter the design: financial time series (financial, macroeconomic and oil-market series); satellite remote sensing; and maritime shipping. Flat and Deep models use the same underlying sources and the same Friday calendar; they differ in the representation applied to each source before prediction. Table 3.2 summarises the datasets. Variable definitions, site lists and graph edges are documented in Appendix A. Feature counts used in estimation for Flat models are those of the merged weekly feature table; Deep models use the same weekly dates with modality-specific tensors.

**Table 3.2 — Datasets, variables and sources**

| Modality | Dataset / product | Key variables | Source |
| --- | --- | --- | --- |
| Financial time series (M1) | Oil-market and macro weekly series | Prices, inventories, production, interest rates, GPR and related indicators | EIA, FRED, Yahoo Finance and related sources |
| Remote sensing (Flat) | Sentinel-2 optical indices and VIIRS night-time lights | Site-level anomalies at 11 AOIs (NDVI, NDWI, NDBI, BSI; NTL) | Sentinel-2; VIIRS |
| Remote sensing (Deep) | Frozen Prithvi-EO-2.0 embeddings | Monthly Sentinel-2 image-patch embeddings at the same 11 AOIs (no VIIRS) | Prithvi-EO-2.0 / Sentinel-2 |
| Shipping (Flat) | PortWatch and AIS tabular features | Port and chokepoint tanker flows; vessel-activity features | IMF PortWatch; AIS |
| Shipping (Deep) | Same sources, represented as a graph | Weekly heterogeneous graph with 17 nodes (11 AOIs and 6 chokepoints) | PortWatch; AIS |

The financial time series block is assembled from weekly oil-market and macro-financial series drawn from the US Energy Information Administration (EIA), Federal Reserve Economic Data (FRED), Yahoo Finance and related scholarly indicators. The series include crude prices and spreads, inventories, production and refinery activity, volatility and risk measures, interest rates, exchange rates, futures-based oil indicators and geopolitical risk. This block is M1—financial time series only—before remote sensing or shipping is added. In the Deep pathway it is encoded as a single modality stream—referred to below as the finance encoder—even though the predictors extend beyond prices alone.

Remote-sensing inputs are observed over the eleven AOIs listed in Section 3.3. Flat and Deep models share these AOIs but use different products from a common Sentinel-2 optical source family. Flat remote sensing uses monthly Sentinel-2 optical indices (NDVI, NDWI, NDBI and BSI) together with VIIRS night-time lights, converted to site-level anomalies. Deep remote sensing uses frozen Prithvi-EO-2.0 embeddings extracted from monthly Sentinel-2 image patches at the same AOIs and excludes VIIRS. Shared AOIs keep spatial coverage matched across pathways; differences in product and representation form part of the Flat–Deep contrast and are detailed in Appendix A.

Shipping inputs combine IMF PortWatch measures of chokepoint and port tanker flows with AIS-derived vessel-activity indicators for the network described in Section 3.3. In the Flat pathway these signals enter as tabular weekly features. In the Deep pathway they are represented as the seventeen-node heterogeneous graph: time-varying AOI-to-AOI edges come from GFW voyage O–D counts, AOI–chokepoint edges are fixed corridor links assigned by each site’s primary oil-trade route role (Section 3.3), and PortWatch/AIS measures enter mainly as node features. Graph construction is specified in Section 3.7 and Appendix A. In both pathways, shipping is treated as a proxy for physical trade and congestion rather than as a direct measure of next week’s price.

**Ethical considerations.** The dissertation uses secondary, aggregate data products and does not involve human participants, interviews or surveys. Market and macro series (EIA, FRED, Yahoo Finance and related indicators), Earth-observation products (Sentinel-2, VIIRS and frozen Prithvi-EO embeddings) and maritime indicators (IMF PortWatch and AIS-derived vessel-activity measures) are accessed under their published terms of use for research. The modelling features are market aggregates, site-level environmental summaries and shipping-activity indicators; they are not used to identify private individuals. AIS-based inputs enter as processed activity measures for commercial and energy-related traffic at ports and chokepoints, consistent with the providers’ intended analytical use, and are not redistributed beyond the licensed research pipeline where redistribution is restricted. Overall risk is minimal: the ethical issue is whether data are used as expected under licence and research purpose, which this design observes by restricting use to forecasting evaluation, documenting sources, and keeping the processing pipeline reproducible.

## 3.5 Temporal alignment, lags, missingness

All series are aligned to the Friday-ending weekly calendar defined in Section 3.2. Predictors enter only after their real publication time, so the model never uses future information at any forecast origin. Release lags differ across sources: EIA and PortWatch series typically become available with a lag of about one week, while slower monthly series require longer buffers before they are allowed to enter. Missingness is handled differently in the two pathways. Flat models fill missing values using only past observations within the available history. Deep models retain explicit masks for missing modalities or sites rather than silently filling them away, so that absence remains visible to the encoders and fusion layers.

## 3.6 Flat models

Flat models implement flat feature fusion as defined in Section 1.2. For a given information set, all available numeric features are concatenated into one weekly table, and the most recent four weeks are flattened into a single row for each forecast origin. Two learners are estimated on this representation. Ridge is a linear model with L2 regularisation and serves as a transparent linear early-fusion baseline. XGBoost is a non-linear tree ensemble that can capture interactions missed by Ridge, but still does not preserve modality-specific structure. Both models predict the one-week-ahead log return and reconstruct price as in Section 3.2. Hyperparameters are chosen inside each training fold on a past validation slice only; the exact grids are reported in Appendix C.

## 3.7 Deep models

Deep models use the same information sets, Friday calendar and validation protocol as Flat. The difference lies in representation and fusion, not in the forecast target. Each available modality is encoded into a matched-dimensional representation and the representations are then combined. The finance encoder uses a causal temporal convolutional network (TCN) to model temporal dependencies in the weekly financial sequence. Deep remote sensing uses frozen Prithvi-EO embeddings from monthly Sentinel-2 patches and excludes VIIRS; embeddings are kept per site and aggregated by temporal and site attention, so the site dimension is not collapsed before encoding. Shipping is encoded as a weekly heterogeneous graph with seventeen nodes—the eleven AOIs and six chokepoints from Section 3.3. Directed AOI-to-AOI edges use weekly GFW voyage O–D counts; AOI–chokepoint edges are fixed, undirected corridor links assigned by each site’s primary oil-trade route role (Section 3.3), not by weekly tracks or nearest-neighbour distance; PortWatch and AIS indicators enter mainly as node features. A graph attention network (GAT) with temporal encoding aggregates this network into a modality representation. Exact edge construction is reported in Appendix A; GAT depth, number of heads and related layer settings are given in Appendix C. Each encoder is therefore specified by its inputs, network structure, outputs and the reason that architecture fits the modality.

For RQ2, three fusion options are considered. Simple concatenation provides a control that combines modality representations without adaptive weighting. Gated fusion is the main reported design. Cross-attention is retained as an advanced alternative. The fused representation is mapped to the same return and price target as Flat; training details are in Appendix C.

## 3.8 Hyperparameter selection

Hyperparameters are selected under a shared protocol so that Flat–Deep comparisons remain fair. For Flat models, tuning is performed inside each training fold on past validation weeks only. For Deep models, full per-fold neural architecture search is costly. Limited sweeps are therefore run first, after which one main configuration—architecture depths, representation size, main fusion choice and related settings—is fixed for the primary reported results. Sensitivity to random seed, lookback length, fusion type, representation size and regularisation is then reported in Chapter 4 and Appendix C. The sequence is search, lock the main setting, then sensitivity analysis; the main Deep configuration is not chosen arbitrarily. Exact values such as representation size, GAT depth and number of heads, and the Flat search grids, are given in Appendix C.

## 3.9 Leakage-free validation protocol

Evaluation follows an expanding-window rolling-origin backtest: at each origin the model is trained only on past weeks and then produces a one-week-ahead forecast. The first 104 weeks form the initial estimation and validation period and are not scored. The four-week lookback creates a separate sequence warm-up requirement. Thereafter models are refit every 13 weeks. The common scored test span covers 257 weeks from 22 January 2021 to 19 December 2025. Any scaling or filtering is fit inside the training fold only. Flat and Deep share the same fold calendar, so architecture comparisons hold the evaluation design fixed.

Figure 3.3 summarises the expanding-window rolling-origin protocol: an initial unscored estimation window, one-week-ahead forecasts, and periodic refitting on past information only.

*[Figure 3.3 — Expanding-window rolling-origin evaluation flowchart.]*

## 3.10 Evaluation, tests, interpretability

Primary metrics are computed on reconstructed prices. Every comparison reports RMSE and MAE; directional accuracy is retained only as an auxiliary measure. Relative performance versus M0 is summarised by RMSE skill, reported as a percentage in the result tables:

\[
\mathrm{Skill}=100\times\left(1-\frac{\mathrm{RMSE}_{\mathrm{model}}}{\mathrm{RMSE}_{\mathrm{M0}}}\right).
\]

Skill greater than zero means the model beats M0 on RMSE; skill equal to zero matches M0; skill less than zero is worse than M0.

The study evaluates both incremental value versus M1 and absolute skill versus M0. It also distinguishes information-set nesting from formal model nesting: a larger modality set does not automatically make two forecasts nested for statistical testing. Test choice follows the forecast-specification relationship. Clark–West (2007) provides an MSPE-adjusted test of whether a larger model improves on a smaller one when the smaller forecast specification is nested in the larger; it is used for nested increments, for example Ridge M1 versus Ridge M2, M3 or M4 where nesting is justified. Diebold–Mariano (1995) tests equal predictive accuracy from the mean loss differential between two forecasts and is used for non-nested paired comparisons, such as Flat versus Deep or XGBoost and Deep settings that change hyperparameters or architecture; a small-sample adjustment is noted where relevant. Every comparison also reports RMSE and MAE effect sizes versus M0 and, where relevant, versus M1.

Interpretability diagnostics are applied only to specifications that improve on M0—primarily Deep M3, and Deep M4 where relevant—and consist of modality gate weights together with site or node attention. Data used in the study are public or licensed, and the processing and estimation pipeline is scripted to support reproducibility.

---
