# A Modality-Aware Spatio-Temporal Fusion Framework for Brent Crude Oil Forecasting Using Financial Time Series, Satellite Imagery and Maritime Networks


## Abstract *(~200 words)*


Brent crude is the main benchmark for internationally traded oil, and its short-term movements affect hedging, budgeting and market-risk decisions. The 2019–2025 period spans the COVID-19 pandemic, the 2022 energy crisis and subsequent market adjustment. This dissertation asks whether satellite remote sensing and maritime shipping data add predictive information beyond financial time series for one-week-ahead Brent price forecasts. It also compares flat feature-level fusion with modality-specific encoding before fusion. Ridge and XGBoost represent the flat approach; deep models encode the three data sources separately and then fuse them. A financial-time-series-only specification is compared with alternatives adding remote sensing, shipping, or both. All models follow the same rolling-origin out-of-sample protocol, use only information available at each forecast date, and are evaluated against a no-change benchmark that sets next week’s price equal to this week’s. On a common evaluation sample, no flat model outperforms this benchmark. Shipping still improves accuracy relative to the financial-time-series-only specification, whereas remote sensing does not. Deep models combining financial time series and shipping data achieve a small gain over the benchmark, but adding remote sensing brings no clear further improvement. The advantage of deep models over flat models is clearest when shipping data are included. Where deep models improve on the benchmark, modality gates and spatial attention are used to show which sources the forecasts rely on. Overall, predictive value depends on the information source and fusion design, not simply on adding more data.


## Chapter 1 — Introduction *(~600 words)*


### 1.1 Importance and background


Crude oil occupies a central place in the world economy. Movements in oil prices affect inflation, trade balances, fiscal revenues in producer countries and the operating costs of energy-intensive industries. These effects transmit quickly through financial markets, real activity and supply chains. Oil-price forecasting therefore matters in energy economics, and for governments, firms and investors concerned with risk management, hedging and planning.


Oil remains a core commodity in the global energy system. Recent years have underlined how costly price surprises can be. The COVID-19 period brought an abrupt demand collapse and an uneven recovery; the 2022 energy crisis then produced a sharp supply and price shock; the years that followed saw partial normalisation under continued geopolitical and macroeconomic uncertainty. In such an environment, claims that new data or more elaborate models improve forecasts need to be tested carefully against strong and transparent benchmarks.



Among crude-oil benchmarks, Brent serves as the global pricing benchmark for a large share of internationally traded oil. This dissertation focuses on Friday-ending weekly Brent spot prices over 2019–2025 and on one-week-ahead out-of-sample forecasts. At the weekly horizon, the no-change forecast is difficult to outperform. That simple rule—predicting that next week’s price equals this week’s price—is a demanding reference point. Any claim that alternative data or a new fusion method helps must clear this bar, not only improve on a weaker or differently specified competitor.


These data provide complementary views of the oil system. Financial, macroeconomic and oil-market variables describe market conditions over time. Remote sensing represents spatial activity at specific oil-related sites through spectral indicators, night-time lights and image embeddings. AIS and PortWatch data describe time-varying vessel activity across ports and major chokepoints, including the network relationships between them, and serve as proxies for seaborne trade flows and congestion. In this dissertation, multimodal forecasting therefore refers to combining temporal market data, spatial Earth-observation data and spatiotemporal shipping-network data within the same forecasting task.


Two practical difficulties follow. First, the signals are noisy and arrive on different schedules; they may also respond to prices rather than lead them, and the weekly sample is relatively small. Second, a common approach is to organise heterogeneous inputs in a single feature table and combine them before modelling. That flat early-fusion approach is convenient for classical models such as Ridge regression or gradient-boosted trees, but it does not explicitly model modality-specific structure: temporal dynamics in financial time series, site structure in remote sensing, and network structure in shipping.


This raises the empirical question addressed in this dissertation. Can remote sensing and shipping improve one-week-ahead Brent price forecasts beyond financial time series and the no-change benchmark? And when the underlying data are held fixed, does keeping each modality’s structure before fusion outperform flat feature fusion? The detailed research gap is developed after the literature review in Chapter 2. The next section states the aim and research questions.


### 1.2 Aim and research questions

The aim of this dissertation is not to propose a new neural-network building block. It is to build one reproducible comparison framework in which the same weekly Brent price forecasting task is evaluated under a shared protocol. That framework brings together a financial time series block; satellite and shipping inputs (including frozen Earth-observation image embeddings where used); and a rolling forecast design that never uses future information, together with formal statistical tests of whether one forecast improves on another. Here, flat feature fusion means placing all inputs in one feature table before modelling, whereas representation-level fusion means encoding each modality separately before combining the resulting representations. In short, the thesis contribution is integration and fair comparison, not a new model operator.


Three research questions organise the study.

**RQ1.** Do remote-sensing and shipping indicators add incremental out-of-sample value over financial time series and the no-change benchmark?

**RQ2.** Does modality-aware representation-level fusion outperform flat feature fusion when both use the same underlying data and the same evaluation protocol?

**RQ3.** Can modality-level interpretability reveal which signals the model relies on across different market conditions?

The logic is sequential: first ask whether the data help; then ask whether fusion architecture matters when the data are held fixed; then, only where predictive value exists, ask what the model relies on.