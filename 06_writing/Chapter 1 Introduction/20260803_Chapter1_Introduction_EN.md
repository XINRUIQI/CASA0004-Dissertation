# A Modality-Aware Spatio-Temporal Fusion Framework for Brent Crude Oil Forecasting Using Financial Time Series, Satellite Imagery and Maritime Networks

---

## Abstract *(~250 words)*

Brent crude is one of the principal benchmarks for internationally traded oil and a key reference price in the global energy market. Its short-term movements affect energy costs, inflation, trade balances and fiscal revenues, and therefore influence decisions made by firms and governments. Using weekly data from 2019 to 2025, this study examines whether satellite remote-sensing and shipping data provide incremental value beyond financial time series for one-week-ahead Brent price forecasting. It also compares flat feature fusion with modality-specific encoding followed by fusion. Flat models combine all selected inputs in a single feature table, whereas deep models encode each data source separately before fusing the resulting representations. For both model families, the study compares a financial-time-series-only specification with alternatives that add remote sensing, shipping or both.

The models are evaluated against a no-change benchmark that sets next week’s price equal to this week’s price. The results show that no flat model outperforms this benchmark, although shipping data provide limited evidence of incremental predictive information. Deep models combining financial time series and shipping data achieve a small improvement over the benchmark. Remote-sensing data provide no clear additional benefit. The advantage of deep models over flat models is most evident when shipping data are included. This study further uses modality gates to show which data sources the best-performing deep model relies on most. Overall, predictive value depends largely on fusion design rather than simply on adding more data.

---



## Chapter 1 — Introduction *(~700 words)*



### 1.1 Importance and background

Crude oil occupies a central place in the global economy and energy system. Oil-price movements affect inflation, trade balances, fiscal revenues in producer countries and the operating costs of energy-intensive industries. These effects spread through financial markets, economic activity and supply chains. They therefore shape the risk management, hedging, budgeting and planning decisions of governments, firms and investors. Recent years have shown the costs of unexpected oil-price movements. The COVID-19 period brought an abrupt collapse in demand and an uneven recovery. The 2022 energy crisis then produced major supply and price shocks, followed by only partial normalisation amid continued geopolitical and macroeconomic uncertainty. This volatility increases the importance of reliable oil-price forecasts. It also means that claims about new data or more complex models must be tested against strong and transparent benchmarks.

That uncertainty is not only historical. Recent conflict has again shown how quickly oil prices and seaborne trade can move when key maritime choke points are disrupted or avoided. Governments watch such shocks for inflation control, fiscal planning, energy security and trade policy. A better short-term oil-price model would help them gauge risk and timing. It would not replace market judgment, but it could support planning when physical flows and prices shift together. In such an environment, claims that new data or more elaborate models improve forecasts need to be tested carefully against strong and transparent benchmarks.

At the weekly horizon, the no-change forecast is difficult to outperform. This simple method predicts that next week’s price will be equal to this week’s price. It therefore provides a demanding benchmark for evaluating alternative data and forecasting methods. A model should not be considered useful merely because it outperforms a weaker or differently specified model. It must also be compared directly with the no-change benchmark.

The three data sources considered in this dissertation provide complementary views of the oil system. Financial, macroeconomic and oil-market variables describe changes in market conditions over time. Remote sensing captures spatial activity at selected oil-related sites through spectral indicators, night-time lights and image representations. AIS and PortWatch data describe changes in vessel activity across ports and major chokepoints. They also capture network relationships between locations and provide proxies for seaborne trade flows and congestion. In this dissertation, multimodal forecasting refers to combining temporal market data, spatial Earth-observation data and spatiotemporal shipping-network data in the same forecasting task.

These data create two practical challenges. First, the signals are noisy and arrive on different schedules. They may also respond to oil prices rather than predict them, while the available weekly sample is relatively small. Second, a common approach places all heterogeneous inputs in a single feature table before modelling. This flat feature-fusion approach is convenient for conventional models such as Ridge regression and gradient-boosted trees. However, it does not explicitly preserve the temporal structure of financial time series, the site structure of remote sensing or the network structure of shipping data.

This dissertation therefore addresses two empirical questions. First, do remote-sensing and shipping data improve one-week-ahead Brent price forecasts when they are added to financial time series? Can the resulting models outperform the no-change benchmark? Second, when the underlying data remain the same, does encoding each modality separately before fusion perform better than combining all inputs in a single feature table? The next section presents the study aim and formal research questions.

### 1.2 Aim and research questions

The main aim of this dissertation is to develop a reproducible comparison framework for evaluating how different data sources and model designs perform in one-week-ahead Brent price forecasting. The framework combines financial time-series data, satellite remote-sensing data and shipping data. It uses a rolling-origin forecasting design that prevents the use of future information and applies formal statistical tests to compare predictive performance. Flat feature fusion places all inputs in a single feature table before modelling. Representation-level fusion encodes each modality separately and then combines the resulting representations. This framework enables consistent comparisons of the incremental value of different data sources and the effects of different fusion designs.

The study is organised around three research questions.

**RQ1.** Compared with models using only financial time-series data, do remote-sensing and shipping data improve one-week-ahead Brent price forecasts?

**RQ2.** When using the same underlying data, does modality-aware representation-level fusion outperform flat feature fusion?

**RQ3.** Which data sources do the models rely on under different market conditions?