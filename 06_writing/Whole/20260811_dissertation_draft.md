# A Modality-Aware Spatio-Temporal Fusion Framework for Brent Crude Oil Forecasting Using Financial Time Series, Satellite Imagery and Maritime Networks

---

## Abstract *(~200 words)*

Brent crude is one of the principal benchmarks for internationally traded oil and a key reference price in the global energy market. Its short-term movements affect energy costs, inflation, trade balances and fiscal revenues, and therefore influence decisions made by firms and governments. Using weekly data from 2019 to 2025, this study examines whether satellite remote-sensing and shipping data provide incremental value beyond financial time series for one-week-ahead Brent price forecasting. It also compares flat feature fusion with modality-specific encoding followed by fusion. Flat models combine all selected inputs in a single feature table, whereas deep models encode each data source separately before fusing the resulting representations. For both model families, the study compares a financial-time-series-only specification with alternatives that add remote sensing, shipping or both.

The models are evaluated against a no-change benchmark that sets next week’s price equal to this week’s price. The results show that no flat model outperforms this benchmark, although shipping data provide limited evidence of incremental predictive information. Deep models combining financial time series and shipping data achieve a small improvement over the benchmark. Remote-sensing data provide no clear additional benefit. The advantage of deep models over flat models is most evident when shipping data are included. This study further uses modality gates to show which data sources the best-performing deep model relies on most. Overall, predictive value depends more on how multimodal data are used—especially how modalities are encoded and fused—than on simply adding more data.

---

## Chapter 1 — Introduction *(~600 words)*

### 1.1 Importance and background

Crude oil occupies a central place in the global economy and energy system. Oil-price movements affect inflation, trade balances, fiscal revenues in producer countries and the operating costs of energy-intensive industries. These effects spread through financial markets, economic activity and supply chains. They therefore shape the risk management, hedging, budgeting and planning decisions of governments, firms and investors.

Crude oil is not a homogeneous commodity: individual grades differ in density, sulphur content, production location and transport accessibility, and their prices are commonly expressed relative to a small number of benchmarks. Among the most widely used benchmarks are Brent, West Texas Intermediate (WTI) and Dubai/Oman (U.S. Energy Information Administration, 2014). Brent is a benchmark complex rooted in light, low-sulphur, waterborne crude oils from the North Sea. It is widely used as a reference for internationally traded crude. WTI is a US crude benchmark, with pricing centred on Cushing, Oklahoma, while Dubai/Oman is commonly used to price Middle Eastern crude exported to Asian markets (Wittner, 2020). Brent and WTI respond to many of the same global market conditions, but their spread can change with differences in regional supply, inventories and transport constraints. This dissertation forecasts Brent because of its international and waterborne orientation, which aligns more closely with the ports, shipping routes and maritime chokepoints in the alternative data. WTI is nevertheless retained as a financial predictor and as a component of the Brent–WTI spread. No fixed volatility ranking between Brent and WTI is assumed. Whether shipping activity contains incremental predictive information for Brent is tested empirically in this dissertation.

Recent years have shown the costs of unexpected oil-price movements. The COVID-19 period brought an abrupt collapse in demand and an uneven recovery. The 2022 energy crisis then produced major supply and price shocks, followed by only partial normalisation amid continued geopolitical and macroeconomic uncertainty. More recent geopolitical disruptions have further shown how quickly oil prices and seaborne trade can respond when key maritime chokepoints are disrupted or bypassed. Governments monitor such shocks for inflation control, fiscal planning, energy security and trade policy. A better short-term oil-price model would help them gauge risk and timing. Although they cannot replace market judgement, such forecasts could support decision-making when physical flows and prices move together.

At the weekly horizon, the no-change forecast is difficult to outperform. This simple method predicts that next week’s price will be equal to this week’s price. It therefore provides a demanding benchmark for alternative data and methods. A model should not be considered useful merely because it outperforms a weaker or differently specified competitor. It must also be evaluated directly against the no-change benchmark.

The three data sources considered in this dissertation provide complementary views of the oil system. Financial, macroeconomic and oil-market variables describe changes in market conditions over time. Remote sensing captures spatial activity at selected oil-related sites through spectral indicators, night-time lights and image representations. AIS and PortWatch data describe changes in vessel activity across ports and major chokepoints. They also capture network relationships between locations and provide proxies for seaborne trade flows and congestion. In this dissertation, multimodal forecasting refers to combining temporal market data, spatial Earth-observation data and spatiotemporal shipping-network data in the same forecasting task.

These data create two practical challenges. First, the signals are noisy and arrive on different schedules. They may also respond to oil prices rather than predict them, while the available weekly sample is relatively small. Second, a common approach places all heterogeneous inputs in a single feature table before modelling. This flat feature-fusion approach is convenient for conventional models such as Ridge regression and gradient-boosted trees. However, it does not explicitly preserve the temporal structure of financial time series, the site structure of remote sensing or the network structure of shipping data.

This dissertation therefore addresses two empirical questions. First, do remote-sensing and shipping data improve one-week-ahead Brent price forecasts when they are added to financial time series? Can the resulting models outperform the no-change benchmark? Second, when the underlying data remain the same, does encoding each modality separately before fusion perform better than combining all inputs in a single feature table? The next section presents the study aim and formal research questions.

### 1.2 Aim and research questions

The main aim of this dissertation is to develop a reproducible comparison framework for evaluating how different data sources and model designs perform in one-week-ahead Brent price forecasting. The framework combines financial time-series data, satellite remote-sensing data and shipping data. It uses a rolling-origin forecasting design that prevents the use of future information and applies formal statistical tests to compare predictive performance. Flat feature fusion places all inputs in a single feature table before modelling. Representation-level fusion encodes each modality separately and then combines the resulting representations. This framework enables consistent comparisons of the incremental value of different data sources and the effects of different fusion designs.

The study is organised around three research questions.

**RQ1.** Compared with models using only financial time-series data, do remote-sensing and shipping data improve one-week-ahead Brent price forecasts?

**RQ2.** When using the same underlying data, does modality-aware representation-level fusion outperform flat feature fusion?

**RQ3.** Which data sources do the models rely on under different market conditions?

---



# Chapter 2 — Literature Review

This chapter reviews the main bodies of literature that support the study. It first examines the economic drivers and empirical benchmarks of crude-oil price forecasting, followed by the application of machine-learning methods. It then reviews how shipping activity reflects oil-market conditions and how satellite remote sensing provides oil-market information. Next, it introduces multimodal learning and fusion methods. Finally, it identifies the research gaps and positions this dissertation within the existing literature.

## 2.1 Crude-oil price drivers and forecasting benchmarks

Research on oil-price movements shows that similar price changes can arise from economically different sources. Kilian (2009) distinguishes among shocks to global crude-oil production, shocks to aggregate demand for industrial commodities and demand shocks specific to the oil market. Oil prices respond differently to these shocks, and each shock has a different relationship with global economic activity and oil production. This distinction provides an economic basis for using variables related to supply, demand, market expectations and macroeconomic conditions in oil-price forecasting.

A separate literature examines whether these economic relationships produce accurate forecasts. Alquist, Kilian and Vigfusson (2013) compare a wide range of forecasting models with the no-change forecast, which sets the future spot price equal to the current price. Many alternative models fail to outperform this simple benchmark consistently, particularly in real-time and out-of-sample evaluations. They also distinguish economic predictability from practical forecastability. A variable may have an economic relationship with future oil prices without reducing forecast errors in a finite out-of-sample period. Baumeister and Kilian (2015) find that combining forecasts from several econometric models can produce more stable results across periods and forecast horizons than relying on a single model. These studies establish the no-change forecast as an important benchmark and show that model performance can vary substantially across evaluation periods.

## 2.2 Machine learning in crude-oil price forecasting

Machine learning has expanded the range of methods used in oil-price forecasting. These methods can process large predictor sets and capture nonlinear relationships and interactions. Costa et al. (2021) evaluate 23 methods using 315 macroeconomic and financial variables. They find that no single method performs best at every forecast horizon. Machine-learning methods are competitive at short horizons, but econometric, market-based and combined forecasts also perform strongly in some settings. Yılmaz and Zehir (2026) compare econometric and tree-based models for Brent returns using macro-financial and geopolitical variables. Light Gradient Boosting Machine (LightGBM) produces the most consistent results across their reported settings, but the broader comparison again shows that performance depends on the forecast horizon, predictor set and evaluation design.

Deep-learning studies focus more directly on learning temporal representations. Foroutan and Lahmiri (2024) compare a range of methods for next-day WTI and Brent spot-price forecasting and report strong performance from temporal convolutional networks and LightGBM. Simsek et al. (2024) combine Long Short-Term Memory (LSTM) feature extraction with XGBoost for WTI price prediction. Graph-based methods have also been introduced. Zhao, Xue and Cheng (2023), for example, model time-varying relationships among economic and financial variables and use a spatial–temporal graph neural network to forecast WTI futures. Their graph represents statistical relationships among predictors rather than a physical transportation network.

Overall, machine learning provides greater flexibility for modelling high-dimensional data, nonlinearities and temporal interactions. However, the literature does not establish that one class of models consistently dominates conventional econometric or market-based approaches. Reported performance depends strongly on the target, horizon, sample, information set and benchmark. These findings support the use of a common rolling out-of-sample design and a no-change benchmark when comparing different data sources and model architectures.

## 2.3 Shipping activity as an oil-market signal

A large share of international crude-oil trade is transported by sea. Shipping activity can therefore provide information about physical oil flows, regional supply conditions, congestion and disruptions at ports or major chokepoints. Automatic Identification System (AIS) data record vessel identities, positions and movements. Although they do not directly record cargo quantities, processed AIS observations can be used to estimate tanker movements and maritime trade.

Adland, Jia and Strandenes (2017) compare AIS-based estimates of seaborne crude-oil exports with customs statistics and find that aggregate estimates are broadly consistent with official data. Yan et al. (2020) combine tanker trajectories with vessel characteristics and draught information to estimate voyage-level oil flows. Their estimates for major oil-importing and oil-exporting countries are strongly correlated with Joint Organisations Data Initiative statistics. These studies provide evidence that vessel movements can serve as proxies for the physical transportation of crude oil.

AIS-based indicators can also improve the timeliness of trade measurement. Arslanalp, Marini and Tumbarello (2019) construct high-frequency trade indicators from vessel movements and port calls. IMF PortWatch extends this approach by combining information on vessel activity, ports, chokepoints, ship characteristics and estimated carrying capacity to produce daily indicators of maritime trade (Arslanalp et al., 2026). Compared with conventional trade statistics, these data can describe changes in maritime activity with shorter reporting delays.

However, the relationship between shipping activity and future oil prices is not necessarily one-directional. Mi et al. (2022) identify relationships between oil-price changes and tanker port-call frequency, docking time, gross tonnage and the number of tankers at ports in major crude-exporting countries. Mi et al. (2023) also find nonlinear and regionally heterogeneous relationships between oil prices and tanker port calls. In both studies, shipping activity responds to oil prices. Their results therefore show that contemporaneous associations do not establish that vessel activity leads future price movements.

AIS-based measures are also indirect and incomplete. Cargo type and quantity must often be inferred from vessel characteristics, routes or draught, and some vessel activity is not observed in public tracking systems. Paolo et al. (2024) show that a meaningful share of transport- and energy-vessel activity is absent from public vessel-position data. These limitations do not make AIS data unusable, but they mean that shipping variables should be treated as noisy proxies for physical trade rather than direct measurements of oil supply.

The existing literature establishes that shipping data can measure changes in maritime trade and crude-oil transportation. However, direct evidence that these data improve short-term Brent price forecasts remains limited.

## 2.4 Maritime networks and graph-based modelling

Maritime-network studies provide a way to preserve relationships among locations. Aggregate indicators summarise port calls, vessel counts or chokepoint traffic, whereas network models represent ports or regions as nodes and vessel movements as links. This representation retains connections between locations and allows activity at one part of the network to be modelled in relation to activity elsewhere.

Ouyang et al. (2022) construct a crude-oil transportation network and use an LSTM–GCN model to forecast weekly traffic at network nodes. Liang et al. (2022) use a spatiotemporal multigraph convolutional network for vessel-traffic forecasting, while Zhao et al. (2022) use a dynamic graph neural network to predict regional vessel inflows, outflows and traffic volumes. These studies show that maritime activity is relational and changes over time. Preserving this structure may therefore provide information that is lost when shipping activity is reduced to a small set of aggregate indicators.

The literature shows that network representations can capture relationships between ports and routes. However, these methods have mainly been used to forecast shipping activity itself. There is limited direct evidence on whether graph-based representations of maritime networks improve oil-price forecasts.

## 2.5 Remote sensing as an oil-market signal

Remote sensing provides repeated observations of oil-related infrastructure, industrial activity and maritime locations. Satellite observations may therefore contain information about oil demand, storage, port activity and trade. However, their economic meaning depends on the physical signal being measured and the mechanism connecting that signal to the oil market.

Several studies connect satellite observations with oil-market conditions. Hao and Wang (2023) use cloud-cover observations above floating-roof oil tanks in major US storage areas. Because the roofs rise and fall with the volume of oil stored, their shadows in clear-sky satellite imagery can be used to estimate inventories before official EIA releases. Cloud cover obscures the tanks and therefore reduces the satellite-based inventory information available to market participants. The authors argue that the resulting information uncertainty may encourage firms to hold larger precautionary inventories. Consistent with this proposed channel, they find that greater cloud cover is followed by higher inventories and lower WTI returns in the following week. They interpret this as an information effect rather than a direct weather effect on oil supply or demand. Bricongne et al. (2026) use satellite observations of tropospheric NO₂, a short-lived pollutant emitted primarily by fossil-fuel combustion, to nowcast national oil demand. They find that daily NO₂ data improve nowcasting accuracy relative to models using conventional predictors, showing that satellite observations can provide timely information about changes in oil consumption.

Other studies use satellite imagery to measure oil-related infrastructure and trade. Wang et al. (2019) estimate the dimensions and structural capacity of oil tanks from high-resolution imagery. Jung (2026) combines radar observations, night-time lights and port characteristics to nowcast port-level maritime trade. Polinov, Bookman and Levin (2022) also identify a relationship between night-time lights and shipping activity in anchorage areas. Together, these studies show that remote sensing can capture physical and economic activity connected to oil storage and maritime trade.

However, remote-sensing variables remain indirect measures of oil-market conditions. A signal that measures storage capacity does not necessarily reflect current inventories, while port activity does not directly measure future oil prices. Short-term changes may also reflect cloud cover, observation conditions or irregular data availability. Evidence obtained from one sensor or target therefore cannot automatically be applied to another.

Existing studies demonstrate that satellite observations contain information related to oil demand, storage and maritime trade. Nevertheless, there is limited direct evidence on whether remote-sensing indicators or satellite-image representations improve short-term Brent price forecasts after financial and oil-market variables are already included.

## 2.6 Multimodal learning and data fusion

Multimodal learning refers to methods that process and combine information from two or more types of data. Each type of data is treated as a modality, and the purpose of multimodal learning is to use their complementary information while accounting for differences in structure, scale and availability.

Baltrušaitis, Ahuja and Morency (2019) identify representation, translation, alignment, fusion and co-learning as five core challenges in multimodal machine learning. This dissertation focuses on fusion, specifically the distinction between feature-level and representation-level fusion. Feature-level fusion places observed or engineered variables from all modalities in a common feature table before modelling. Representation-level fusion processes each modality separately before combining the resulting representations.

The two approaches retain different amounts of modality-specific structure. Feature-level fusion is compatible with conventional regression and tree-based models, but it may reduce temporal, spatial and network data to a common tabular format. Representation-level fusion can maintain separate processing streams for different data sources. Arevalo et al. (2017) propose a gated multimodal unit that combines modality-specific representations through input-dependent gates. The contribution of each modality can therefore change across observations. Gohari et al. (2024) apply a related modality-aware approach to financial forecasting by combining textual reports and numerical economic series. Their results show that separate representations and cross-modal interactions can improve performance in a financial time-series setting.

Representation learning is also important for satellite imagery. For example, SatMAE (Cong et al., 2022) learns representations from temporal and multispectral satellite observations. Prithvi-EO-2.0 (Szwarcman et al., 2026) is pretrained on global multitemporal Earth-observation imagery and incorporates temporal and location information. CROMA (Fuller, Millard and Green, 2023) separately processes optical and radar observations before producing a joint representation. These models demonstrate that pretrained encoders can preserve spatial, spectral and temporal information that may not be captured by manually engineered satellite indicators. However, their evaluations mainly concern remote-sensing tasks such as classification, segmentation and disaster mapping rather than commodity-price forecasting.

Multimodal data also create alignment and missing-data problems. Financial data, shipping observations and satellite imagery may be recorded at different frequencies and become available at different times. An entire modality may be missing for some observations, or individual observations within a modality may be irregular or delayed. Ma et al. (2022) show that multimodal models can be sensitive to missing modalities, while Neverova et al. (2016) propose ModDrop, which randomly drops one or more entire modalities during training to improve robustness when modalities are missing at test time. Time-series methods such as GRU-D (Che et al., 2018) and Multi-Time Attention Networks (Shukla and Marlin, 2021) explicitly represent missingness and irregular observation times. These studies show that adding more modalities does not automatically resolve differences in data availability and timing.

Some multimodal architectures provide internal quantities that can be inspected. Modality gates can indicate how strongly the fitted model weights different data sources, while attention mechanisms can show how weights are distributed across inputs or representations. These quantities can help describe model behaviour, but they should not be treated as direct causal explanations. Jain and Wallace (2019) show that different attention patterns can sometimes produce similar predictions. Gates and attention weights therefore indicate how a model processes information, not how the underlying economic system is causally determined.

Existing multimodal research provides methods for preserving temporal, spatial and network structures before fusion. However, there is limited direct evidence on whether such methods outperform flat feature fusion in oil-price forecasting when both approaches use the same underlying information.

## 2.7 Research gaps and positioning of this dissertation

The literature produces four main conclusions. First, short-term oil-price forecasting is difficult, and simple no-change forecasts remain strong benchmarks. Second, shipping data provide timely but indirect measures of physical trade, oil transportation and congestion. Third, remote sensing can measure oil-related demand, infrastructure, port activity and information availability, but the meaning of each signal depends on its physical and economic mechanism. Fourth, multimodal methods can preserve the distinct structures of financial time series, satellite imagery and shipping networks before combining them.

Three research gaps follow from these findings.

First, the predictive value of shipping and remote-sensing data for Brent prices remains unclear. Oil-price forecasting studies mainly use historical prices, macroeconomic variables, financial indicators and oil-market data. Shipping research more often predicts trade or vessel activity, while remote-sensing research generally measures demand, infrastructure or port activity. These studies show that shipping and satellite observations contain economically relevant information, but they provide limited direct evidence on whether these sources improve one-week-ahead Brent price forecasts beyond financial time-series data.

Second, existing studies process shipping and remote-sensing data in different ways. Economic applications usually convert these data into numeric indicators and place them in a common feature table. Maritime-network and Earth-observation studies instead use graph models or pretrained encoders to preserve network, spatial or temporal structure. These approaches are usually examined in separate applications rather than compared in the same oil-price forecasting task. The literature therefore does not show whether modality-specific encoding performs better than flat feature fusion when both methods use the same underlying data.

Third, forecasting studies often use different evaluation settings, and analyses of model reliance are rarely connected directly to predictive improvements. Published studies differ in their forecast targets, horizons, samples, information sets and benchmarks. Their reported results are therefore not always directly comparable. In addition, studies that examine feature importance, modality gates or attention weights often report these results separately from out-of-sample forecasting performance. As a result, the literature provides limited evidence on whether a model’s reliance on a particular data source is associated with an actual improvement over a common benchmark.

This dissertation addresses these gaps through a shared rolling-origin out-of-sample framework for one-week-ahead Brent price forecasting. It first compares financial time-series data with information sets that add shipping data, remote-sensing data or both. It then compares flat feature fusion with modality-aware representation-level fusion using matched underlying data. All models are evaluated against the same no-change benchmark, and formal statistical tests are used to assess differences in predictive performance. Modality-level interpretation is applied to describe which data sources the better-performing models rely on under different market conditions. These results are interpreted as descriptions of model behaviour rather than evidence of causal relationships.

---



# Chapter 3 — Methodology *(~3,200)*



## 3.1 Research design

This chapter sets out how the study answers the research questions in Section 1.2. In brief, every learned forecast is judged against a simple no-change benchmark in which next week’s Brent price equals this week’s price. The study then asks whether remote sensing and shipping add useful information beyond financial time series, and whether modelling those inputs as one weekly table differs from encoding each data type separately before combining them. All comparisons use the same weekly forecast dates, sample window and evaluation rules, so that changes in the data can be separated from changes in how the data are modelled.

The no-change benchmark is denoted M0. At each forecast origin t, M0 sets the one-week-ahead Brent price forecast equal to the current weekly price


\hat{P}_{t+1|t}=P_t.


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

Let P_t denote the last available daily Brent spot-price observation in week t, where each week ends on Friday, measured in US dollars per barrel. The quantity reported in the results is the one-week-ahead price P_{t+1}. Models are not trained directly on the price level. They predict the one-week logarithmic return


r_{t+1}=\log\left(\frac{P_{t+1}}{P_t}\right)


and reconstruct the price forecast as


\hat{P}*{t+1|t}=P_t\exp\left(\hat{r}*{t+1|t}\right).


Log returns are used to reduce the strong persistence in the price level and to express the forecasting task in terms of proportional weekly changes. RMSE, MAE and skill versus M0 are computed from the reconstructed price forecasts. Directional accuracy is reported separately as an auxiliary statistic based on the sign of the predicted and observed returns. Under this mapping, the no-change benchmark \hat{P}*{t+1|t}=P_t is exactly the same as forecasting a zero return \hat{r}*{t+1|t}=0.

All series are organised on a Friday-ending weekly calendar. The modelling window covers 2019–2025 and provides a common weekly index of 365 observations (4 January 2019 to 26 December 2025). Flat models use a merged weekly feature table on this index. Deep models use the same dates, but keep financial, remote-sensing and shipping inputs in their own sequence or graph form rather than one shared table. The first 104 weeks are reserved for initial estimation. Three further weeks are needed to form the first four-week input sequence, and the final week is excluded because P_{t+1} is unavailable. This leaves 257 forecast origins for evaluation, from 22 January 2021 to 19 December 2025. At each origin t, forecasts may use only information that was actually available at that forecast date.

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


\mathrm{Skill}=100\times\left(1-\frac{\mathrm{RMSE}*{\mathrm{model}}}{\mathrm{RMSE}*{\mathrm{M0}}}\right).


Skill greater than zero means the model beats M0 on RMSE. Skill equal to zero matches M0. Skill less than zero is worse than M0.

The study reports both absolute skill versus M0 and incremental value versus M1. Statistical tests are chosen by the type of comparison, not by the size of the modality set alone. Adding remote sensing or shipping enlarges the information set, but that does not by itself make two forecasts nested for testing. When one forecast specification is nested in another—for example Ridge M1 versus Ridge M2, M3 or M4 under the same learner—Clark–West (2007) is used to test whether the larger model improves mean squared prediction error. When the comparison is not nested—for example Flat versus Deep, or XGBoost versus a Deep setting that changes hyperparameters or architecture—Diebold–Mariano (1995) is used to test equal predictive accuracy. A small-sample adjustment is noted where relevant. Every comparison also reports RMSE and MAE differences versus M0 and, where relevant, versus M1.

Interpretability diagnostics are applied only to specifications that improve on M0. The main cases are Deep M3 and, where relevant, Deep M4. The diagnostics report modality gate weights together with site or node attention.

---



# Chapter 4 — Results（～**1,200** ）



## 4.1 Descriptive overview

This chapter reports out-of-sample one-week-ahead Brent forecasts on the common evaluation sample of 257 weeks (22 January 2021–19 December 2025). Performance is summarised by RMSE on reconstructed prices and by RMSE skill versus the no-change benchmark M0 (Murphy, 1988). Skill is positive when RMSE is lower than M0 and negative when it is higher. On this sample the M0 RMSE is 4.152 USD per barrel.

Weekly Brent log returns have near-zero mean and clear volatility clustering. Exploratory checks show only weak contemporaneous association between remote-sensing anomalies and returns. Shipping enters as a noisy proxy for trade and congestion, not as a direct measure of next week’s price.

## 4.2 Flat-model results

Table 4.1 reports Flat out-of-sample performance for Ridge and XGBoost across M0–M4. Every learned Flat specification has negative skill versus M0, so the no-change forecast remains the best absolute-error benchmark in the Flat family.

**Table 4.1 — Flat out-of-sample performance** *(n = 257)*


| Set | Content                           | Ridge RMSE | Ridge skill vs M0 | XGB RMSE | XGB skill vs M0 |
| --- | --------------------------------- | ---------- | ----------------- | -------- | --------------- |
| M0  | no-change benchmark               | 4.152      | —                 | 4.152    | —               |
| M1  | financial time series only        | 4.256      | −2.5%             | 4.368    | −5.2%           |
| M2  | financial time series + RS        | 4.414      | −6.3%             | 4.440    | −6.9%           |
| M3  | financial time series + shipping  | 4.430      | −6.7%             | 4.429    | −6.7%           |
| M4  | financial time series + RS + ship | 4.525      | −9.0%             | 4.507    | −8.6%           |


Finance-only M1 records the lowest Flat RMSE among learned sets (Ridge 4.256, −2.5%; XGBoost 4.368, −5.2%). Adding remote sensing (M2) or shipping (M3) raises RMSE relative to M1 under both learners. The full Flat set M4 is weakest (Ridge 4.525, −9.0%; XGBoost 4.507, −8.6%). Ridge and XGBoost agree: M1 is best among Flat learners, M4 is worst, and neither remote sensing nor shipping reduces absolute RMSE below the finance-only Flat baseline.

Under early feature fusion, noisy alternative-data proxies do not improve one-week-ahead Brent RMSE relative to M0 or to finance alone. For RQ1, Flat results therefore show no absolute out-of-sample gain from remote sensing or shipping.

## 4.3 Deep-model results

Table 4.2 reports Deep performance by information set. Gated fusion is the main Deep specification; cross-attention is a comparison where multimodal fusion applies. For M1 only the finance encoder is active. M1 and M2 both fail to beat M0 (gated RMSE 4.250 and 4.253; both −2.4% skill). Absolute error barely moves when remote sensing enters.

**Table 4.2 — Deep out-of-sample performance** *(gated = main specification)*


| Set | Content                           | Gated RMSE | Gated skill vs M0 | Cross-attn RMSE | Cross-attn skill vs M0 |
| --- | --------------------------------- | ---------- | ----------------- | --------------- | ---------------------- |
| M0  | no-change benchmark               | 4.152      | —                 | 4.152           | —                      |
| M1  | financial time series only        | 4.250      | −2.4%             | —               | —                      |
| M2  | financial time series + RS        | 4.253      | −2.4%             | —               | —                      |
| M3  | financial time series + shipping  | 4.147      | +0.11%            | 4.121           | +0.74%                 |
| M4  | financial time series + RS + ship | 4.205      | −1.3%             | 4.147           | +0.12%                 |


Once shipping is included, gated M3 reduces RMSE to 4.147 (+0.11% skill). Cross-attention on the same set reaches 4.121 (+0.74%) on this reported seed. Shipping is the modality that moves Deep forecasts across the M0 line relative to Deep M1. Gated M4 rises again to 4.205 (−1.3%); cross-attention M4 is near M0 at +0.12% but does not displace gated M3 as the main finding. The gated margin is small and should not be over-read on a short weekly sample; Section 4.5 returns to seed sensitivity.

For RQ1 under Deep, shipping-inclusive forecasts clear M0 by a modest margin, while remote sensing does not add a comparable absolute-error gain.

## 4.4 Flat versus Deep

**Table 4.3 — Paired Flat versus Deep**
*(Flat = Table 4.1 XGBoost; Deep = Table 4.2 gated; percentages are skill versus M0)*


| Pair | Flat RMSE | Deep RMSE | Flat skill vs M0 | Deep skill vs M0 |
| ---- | --------- | --------- | ---------------- | ---------------- |
| M1   | 4.368     | 4.250     | −5.2%            | −2.4%            |
| M2   | 4.440     | 4.253     | −6.9%            | −2.4%            |
| M3   | 4.429     | 4.147     | −6.7%            | +0.11%           |
| M4   | 4.507     | 4.205     | −8.6%            | −1.3%            |


Deep has lower RMSE than Flat in every matched pair. Finance-only and finance-plus-RS pairs improve on Flat but remain negative versus M0. The decisive pair is M3: Flat skill −6.7% versus gated Deep +0.11%—the only matched pair in which Deep also beats M0. Deep M4 improves on Flat M4 but stays negative versus M0 and does not improve on Deep M3.

For RQ2, representation-level Deep modelling reduces RMSE relative to Flat at every matched set, but an M0-beating paired outcome appears only when shipping is included.

## 4.5 Robustness and sensitivity

Appendix B collects the detailed robustness tables. Flat checks that vary lookback and feature settings produce no Flat specification that beats M0. Finance-only M1 remains the strongest Flat absolute-error baseline; remote sensing stays weak and is not driven by a single site. Nested Clark–West tests versus M1 in Appendix B detect incremental information over the financial baseline for some XGBoost shipping specifications, even when absolute RMSE remains higher than M1 and skill versus M0 remains negative. Shipping can therefore show a nested Flat signal without overturning Table 4.1’s absolute-error ranking.

Deep checks that vary random seeds and fusion choices leave gated finance-plus-shipping as the more stable small positive-skill configuration. Cross-attention can exceed gated fusion on one seed, as in Table 4.2 for M3, but varies more across seeds. Larger encoder width than the main setting tends to worsen performance on the short weekly sample. Sub-period splits leave gated M3 positive in both early and late windows. The matched Deep advantage over Flat, especially with shipping, survives these checks.

These checks leave the RQ1–RQ2 rankings unchanged: Flat absolute gains remain absent; Deep’s small shipping-centred M0 clearance is the more stable positive case.

## 4.6 Interpretability

Interpretability is restricted to Deep specifications that improve on M0, principally Deep M3, using seeds 42, 1 and 2. Reported patterns are those that agree across seeds. Modality gates give each modality’s fusion-weight share; shipping node attention identifies which graph locations receive weight. A high shipping gate does not by itself mean the model focuses on a particular chokepoint; spatial detail is read from node attention.

For Deep M3, mean gates are about 0.56 (financial time series) and 0.44 (shipping). Week-level shipping-gate paths are unstable across seeds, so single-seed event stories are not warranted. Among pre-specified event windows (±8 weeks), only the Russia–Ukraine announcement window (February 2022) shows a shipping-gate rise across all three seeds. The Red Sea window (November 2023) rises in two seeds and falls in one, and is not retained. Spatially, the Strait of Hormuz is the only chokepoint in the top attention set for all three seeds. Figure 4.1 summarises the main Deep M3 gate and attention diagnostics; further panels are in Appendix B.

Figure 4.1 — Deep M3 modality gates and shipping-node attention (multi-seed summary).

*[Figure 4.1 — Deep M3 interpretability: modality gates and shipping-node attention.]*

For RQ3, when Deep shipping-inclusive forecasts clear M0, the stable main-text reliance pattern is shared weight on finance and shipping, with Hormuz as the only cross-seed spatial focus. These diagnostics describe model dependence after a stability filter; they do not identify causal drivers of Brent prices.

---



# Chapter 5 — Discussion *(~1,600)*

Chapter 4 reported out-of-sample one-week-ahead Brent forecasts under a shared evaluation protocol. This chapter interprets those findings against the oil-forecasting, alternative-data and multimodal-learning literatures reviewed in Chapters 1–2, and against the broader energy-security and market-monitoring agendas introduced there. The discussion begins narrowly with the three research questions, then widens to implications, transferability, limitations and future work. Critical reflection here means asking how the results relate to prior studies and policy-relevant debates—not merely listing project constraints.

## 5.1 RQ1 — Do alternative data help?

RQ1 asked whether remote sensing and shipping add out-of-sample value beyond financial time series and the no-change benchmark (M0), which sets next week’s price equal to this week’s. The answer depends on the contrast used, and that dependence is itself part of the finding. No Flat model outperforms M0. This accords with the short-horizon oil-forecasting literature that treats the no-change forecast as a demanding reference (Alquist, Kilian and Vigfusson, 2013). Within the Flat family, Table 4.1 shows that absolute RMSE rises when remote sensing or shipping is added to the finance-only set (M1). Nested Clark–West tests nevertheless detect incremental information for some XGBoost shipping specifications relative to M1, even though skill versus M0 remains negative. Absolute-error rankings and nested increments should therefore be read together; shipping can show a nested Flat signal without overturning the absolute ranking in which M1 remains best among Flat learners.

Under the Deep pathway, the finance-plus-shipping specification (M3) records a small positive skill versus M0. Adding remote sensing on top of that combination often brings no further reduction in error. Shipping is therefore the more informative alternative modality in this weekly Brent design, while remote sensing contributes little to one-week-ahead forecast skill. The positive skill against M0 is a substantive result in a setting where many learned models fail that benchmark. At the same time, the margin is modest, and this study does not evaluate trading costs, hedging profit and loss, or other economic criteria. The claim therefore remains one of statistical and benchmark value, not of ready operational use.

This differs from much of the AIS and satellite work in Chapter 2. Those studies often show that ships or satellites carry trade or activity information. They less often test whether the same data improve one-week-ahead Brent forecasts against both a financial baseline and M0. The present results do not deny that physical information. They show a simpler point: measuring trade is not the same as beating a hard weekly price benchmark. Signals may also be noisy or move with prices rather than ahead of them. Nested gains over financial time series and skill versus M0 should therefore be reported together.

## 5.2 RQ2 — Does representation-level fusion beat flat fusion?

RQ2 asked whether modality-aware representation-level fusion outperforms flat feature fusion when the underlying data and evaluation protocol are held fixed. On matched information sets, the Deep pathway records lower out-of-sample RMSE than the Flat pathway for every pair: finance only, finance plus remote sensing, finance plus shipping, and the full set. In that paired sense, representation-level fusion outperforms flat fusion throughout. The size of the gap, and whether Deep also outperforms M0, still depends on which modalities are included. Gains remain limited for finance-only and finance-plus-remote-sensing pairs. The only matched pair that also beats M0 is finance plus shipping.

The finding sits between two literatures. Flat early fusion remains a convenient default for classical high-dimensional oil-price learners, but it does not retain network structure. Gated and modality-aware models show that separate streams can matter (Arevalo et al., 2017; Gohari et al., 2024), yet those studies are not weekly Brent designs that combine AIS–PortWatch graphs with a no-change price benchmark. The paired results therefore complement both lines of work. The RMSE advantage of Deep over Flat is uniform under matched sets. Preserving shipping-network structure is what turns that advantage into skill versus M0. The same Deep machinery does not make remote sensing decisive for weekly Brent. Cross-attention can raise performance under a single random seed, but it is less stable across seeds than gated fusion. Preference among Deep fusion rules is therefore conditional, even though the Flat-versus-Deep RMSE ranking is not.

## 5.3 RQ3 — What does the model rely on when value exists?

RQ3 asked whether modality-level interpretability can show which signals the model relies on when forecasts already have predictive value. Analysis is therefore limited to Deep specifications that improve on M0, principally Deep M3, and follows a multi-seed rule: only patterns that agree across seeds 42, 1 and 2 are treated as main-text findings. Mean modality gates place substantial weight on both financial time series and shipping (about 0.56 and 0.44). Week-by-week shipping-gate paths are unstable across seeds, so fine-grained event stories based on one seed are not warranted. Among the pre-specified disruption windows, only the Russia–Ukraine announcement window of February 2022 shows a shipping-gate rise that co-moves across all three seeds. The Red Sea disruption window centred on November 2023 does not: the shipping gate rises in two seeds and falls in one, and is therefore not reported as a robust main-text result. Spatially, the Strait of Hormuz is the only maritime chokepoint that appears in the top-ranked attention set for all three seeds.

That reading matches a cautious view of attention and gates. Such weights describe operations inside a fitted model and need not identify causal features (Jain and Wallace, 2019). It also differs from monitoring narratives—common in energy-security and trade commentary after the 2022 crisis—that treat a single disruption window, or one seed’s chokepoint map, as actionable evidence. The diagnostics support a narrower claim. When Deep shipping-inclusive forecasts outperform M0, the Strait of Hormuz is the only spatial focus stable enough to emphasise in the main text. Event-window gate changes are discussed only where they survive the multi-seed filter. These quantities remain model-dependence diagnostics rather than causal explanations of Brent prices, and they should not be read as stand-alone policy alerts.

## 5.4 Implications

The immediate implication is methodological. For this weekly Brent task, a shared evaluation protocol matters as much as a new fusion module. Nested contrasts against financial time series and absolute contrasts against M0 jointly show that shipping can help, and that under Deep it can outperform the no-change benchmark, while remote sensing adds little. Flat early fusion of alternative data is not automatically useful. Teams that are offered multimodal “signals” can therefore require the same double test used here before treating those signals as decision-relevant at a weekly horizon: nested gain over financial time series, and skill versus a no-change rule.

That recommendation is deliberately procedural rather than a claim that Deep M3 should replace existing hedges or enter live trading. Chapter 1 framed oil-price surprises as relevant to risk management, budgeting and planning for governments, firms and investors. The results speak to that agenda, but they do not “support policy” in a vague sense. They suggest a concrete evaluative standard. Physical-flow monitors based on AIS, PortWatch or satellite products can remain useful for describing disruption. They should not be equated with proven one-week-ahead Brent forecast value unless they clear both baselines. After the 2022 energy crisis and later maritime disruptions, that distinction matters for energy-security monitoring, trade planning and inflation-sensitive fiscal management: better description of physical stress is not the same as a better short-horizon price forecast.

More broadly, alternative-data and Earth-observation providers can report nested gains and M0 skill together, so that nested-only improvements are not oversold. Methods researchers can reuse the matched Flat–Deep design for other commodities, horizons or multimodal economic series where one modality has relational structure worth preserving. The transferable object is the comparison protocol—information ladders, paired architectures and double baselines—not a single trained weekly Brent model.

## 5.5 Limitations

Several constraints bound how far the claims can travel. The forecast horizon is weekly, and the scored sample after warm-up is modest, so small skill differences should not be over-interpreted. Alternative-data proxies are noisy and may respond to prices as well as lead them. Frozen Earth-observation embeddings, shipping-graph construction and missingness rules affect Deep results; cross-attention is especially sensitive to the random seed. Matched Flat–Deep comparisons also differ in model class and capacity, so they isolate the overall modelling pathway more cleanly than a single fusion operator. In addition, the Flat and Deep remote-sensing inputs are not identical: Flat uses spectral indices and VIIRS night-light anomalies, whereas Deep uses frozen Sentinel-2 image embeddings and excludes VIIRS. The paired architecture contrast therefore reflects differences in the full modelling pathway, not a pure operator contrast on the same remote-sensing features. Finally, the study does not conduct an economic evaluation of trading or hedging performance, so practical value for desks or ministries remains untested.

## 5.6 Future research and closing statement

Future work can extend the strongest Deep specifications to longer histories and more seeds; enrich the shipping graph and missing-modality stress tests; and apply the same Flat–Deep protocol to other horizons or related energy commodities. Where data allow, a stricter like-for-like remote-sensing comparison between Flat and Deep would isolate architecture more cleanly. Economic evaluation—transaction costs, simple hedging rules or stress scenarios around major disruptions—would test whether the small statistical edge against M0 survives criteria that matter to users. Those extensions would show whether the shipping-centred pattern of modest positive skill generalises beyond this weekly Brent window.

Taken together, the study’s point is evaluative as much as predictive. Alternative data and representation-level fusion can help one-week-ahead Brent forecasting, but only conditionally, and only when judged against strong baselines already central to the oil-forecasting literature and to the risk-management agendas that motivate short-horizon price work. Keeping that standard explicit is the main recommendation this dissertation offers to researchers, data providers and users who must decide what counts as evidence.

---



# Chapter 6 — Conclusion *(~300)*

Short-horizon oil-price surprises matter for hedging, budgeting and market-risk decisions. The 2019–2025 window spans the COVID-19 shock, the 2022 energy crisis and later market adjustment, when physical disruption and prices often moved together—yet weekly Brent remains hard to improve upon once a no-change benchmark is imposed. This dissertation therefore asked whether satellite remote sensing and maritime shipping add predictive information beyond financial time series for one-week-ahead Brent prices, and whether modality-aware representation-level fusion outperforms flat feature fusion. The design used a shared leakage-safe rolling-origin protocol, an M0–M4 information ladder, paired Flat (Ridge/XGBoost) and Deep (modality encoders plus gated or cross-attention fusion) models, and formal nested and non-nested forecast tests.

No Flat model beats M0, though shipping still helps relative to financial time series while remote sensing does not. Under Deep, finance plus shipping (M3) shows only a small positive skill versus M0, and adding remote sensing (M4) does not clearly dominate M3. At matched sets, Deep outperforms Flat most clearly once shipping enters. Where forecasts improve on M0, multi-seed-stable diagnostics show substantial average shipping-gate weight and the Strait of Hormuz as the only cross-seed-stable chokepoint focus; Red Sea event-window gate moves do not survive the multi-seed consistency filter and are therefore not treated as robust main-text findings. These readings describe model dependence, not causal price drivers.

## The contribution is integrative: a leakage-safe nested multimodal comparison of financial time series, remote sensing and shipping; paired Flat–Deep contrasts at matched information sets; joint reporting of nested increments and absolute skill versus M0; and interpretability kept behind a predictive-value and multi-seed filter. For risk-management and energy-crisis monitoring agendas already raised in the introduction, the practical change is evaluative rather than operational—require double tests against financial time series and M0, and treat only cross-seed-stable diagnostics as discussable model reliance. Alternative data and representation-level fusion can help, but strong baselines come first and absolute gains remain modest.



# Appendix A — Data: variable dictionary, AOI/chokepoint lists, lags, graph edges

> Merged weekly matrix `weekly_feature_matrix.csv` = 365 weeks × 212 columns
> (2019-01-04 → 2025-12-26): 31 M1 + 55 M2 + 113 M3 + 11 masks + 2 targets.
> Per-variable literature/industry sourcing is in the modality data dictionaries
> (`03_data/processed/M{1,2,3}/*_data_dictionary.md`); this appendix consolidates
> the model-facing dictionary, the site lists, the publication lags and the
> shipping-graph edge construction.

Information sets: **M0** random walk (`brent_price` only) · **M1**
finance (31) · **M2** M1 + remote sensing (55) · **M3** M1 + shipping full (113)
· **M4** M1+M2+M3 (199). Flat flattens each column over a 4-week lookback
(lag 0–3); Deep keeps modality structure inside encoders.

---



## A.1 Variable dictionary (M1–M4)



### A.1.1 M1 — Finance / macro (31)

**Prices & derived (5)**: `brent_price`, `wti_price`, `brent_log_return`,
`wti_log_return`, `brent_wti_spread` (EIA spot; log returns = ln(Pₜ/Pₜ₋₁)).

**EIA WPSR fundamentals (12, +1 w)**:
`crude_stocks_excl_spr`, `cushing_stocks`, `crude_production`, `crude_imports`,
`crude_exports`, `refinery_crude_input`, `refinery_utilisation`,
`gasoline_supplied`, `distillate_supplied`, `jet_fuel_supplied`,
`crude_stocks_change`, `cushing_stocks_change`.

**Macro-financial (5)**: `vix` (FRED VIXCLS), `dollar_index`
(DTWEXBGS), `treasury_10y` (DGS10), `fed_funds_rate` (DFF),
`sp500_log_return` (^GSPC).

**Derived market/macro (9)**: `ovx`, `gpr`, `gold_return`,
`global_econ_activity` (Kilian REA), `nonoil_industrial_commodity`
(IMF PINDUINDEXM), `brent_f1_spot_log_basis`, `brent_roll_week` (dummy),
`cadusd_log_return`, `dgs10_change`.

### A.1.2 M2 — Remote sensing (55 = 5 indices × 11 AOI)

Naming `{index}_anom_{AOI}`; `anom` = within-site deseasonalised z-score
(expanding, past-only). Raw `level` and staleness/mask columns are not modelled.


| Index  | Meaning                   | Formula                               | Source        |
| ------ | ------------------------- | ------------------------------------- | ------------- |
| `NDVI` | vegetation greenness      | (B8−B4)/(B8+B4)                       | Sentinel-2 SR |
| `NDWI` | surface water/moisture    | (B3−B8)/(B3+B8)                       | Sentinel-2 SR |
| `NDBI` | built-up                  | (B11−B8)/(B11+B8)                     | Sentinel-2 SR |
| `BSI`  | bare soil / storage yards | ((B11+B4)−(B8+B2))/((B11+B4)+(B8+B2)) | Sentinel-2 SR |
| `NTL`  | night-time light activity | VIIRS DNB `avg_rad`                   | VIIRS DNB     |


> Literature arm (C1) = `NTL_anom` of Fujairah / RasTanura / Rotterdam / Houston
> (4 cols).



### A.1.3 M3 — Shipping, flat full tier (113 = GFW 49 + PortWatch 64)

Naming `gfw_{cp}_{stat}` and `pw_{cp}_{stat}` over 6 chokepoints, plus
cross-chokepoint aggregates and PortWatch port export/import volumes. Main model
= full 113 (the hand-picked 38-col *core* tier is a robustness arm; full is XGB-
optimal, see Appendix B / `m3_data_dictionary.md` §11).


| Family                                                             | Meaning                                                    |
| ------------------------------------------------------------------ | ---------------------------------------------------------- |
| `gfw_{cp}_total_hours` / `total_vessels` / `cargo_hours`           | GFW vessel-presence hours / distinct vessels / cargo hours |
| `gfw_{cp}_bunker_hours` / `other_hours` / `other_share`            | bunker / other-vessel presence & share                     |
| `gfw_{cp}_total_hours_mom_pct` / `mean_presence_hours_per_vessel`  | month-over-month %; per-vessel congestion proxy            |
| `gfw_all_total_hours_sum` / `gfw_all_activity_zmean`               | cross-chokepoint aggregate (sum / leak-free z-mean)        |
| `pw_{cp}_n_tanker` / `n_total` / `capacity_tanker` / `capacity`    | PortWatch tanker / all-vessel transit count & capacity     |
| `pw_{cp}_tanker_share` / `tanker_cap_share` / `avg_tanker_size`    | tanker shares; average tanker DWT                          |
| `pw_{cp}_n_tanker_wow_pct` / `capacity_tanker_4w_ma`               | week-over-week %; 4-week MA                                |
| `pw_all_*` (n_tanker_sum, n_total_sum, tanker_share)               | cross-chokepoint tanker aggregates                         |
| `pw_tanker_exp_imp_net` / `_asym` / `_log_ratio` / `_4w_ma`        | export−import net / asymmetry / log-ratio                  |
| `pw_exp_hubs_export_vol` / `pw_imp_hubs_import_vol` (+ `_wow_pct`) | export/import hub tanker tonnage                           |



### A.1.4 Deep shipping graph node features (not in flat matrix)

The Deep arm does **not** use the flat 113 columns; it builds a 17-node graph
(`m3_graph17_tensors.npz`). Node feature spaces differ by type (heterogeneous).

**AOI node features (11 per AOI node)**:
`pw_portcalls_tanker`, `pw_portcalls_cargo`, `pw_import_tanker`,
`pw_export_tanker`, `gfw_n_visits`, `gfw_dwell_hrs_mean`, `gfw_dwell_hrs_median`,
`gfw_self_loops`, `sar_detections_total`, `sar_detections_dark`, `sar_dark_share`.

**Chokepoint node features (20 per node = GFW 8 + PortWatch 9 + SAR 3)**:
GFW `{total_hours, total_vessels, cargo_hours, bunker_hours, other_hours, other_share, total_hours_mom_pct, mean_presence_hours_per_vessel}`;
PortWatch `{n_tanker, n_total, capacity_tanker, capacity, tanker_share, tanker_cap_share, avg_tanker_size, n_tanker_wow_pct, capacity_tanker_4w_ma}`;
SAR `{detections_total, detections_dark, dark_share}`.

The Deep RS branch uses frozen **Prithvi-EO-2.0 embeddings** (1024-d per
AOI-month) rather than the M2 indices; VIIRS is Flat-only.

---



## A.2 AOI and chokepoint node lists



### A.2.1 11 oil-infrastructure AOIs

Fixed node order P001–P010 (graph AOI index 0–10). 5 km analysis buffer;
AOI-differentiated Sentinel-2 patch sizes. Source:
`aoi_oil_infrastructure_sites.md`.


| ID   | Site            | Country      | Type     | Role               | Chokepoint    | (lon, lat)      |
| ---- | --------------- | ------------ | -------- | ------------------ | ------------- | --------------- |
| P001 | Rotterdam       | Netherlands  | port     | pricing / import   | Suez · Cape   | 4.145, 51.950   |
| P002 | Fujairah        | UAE          | terminal | transit / storage  | Hormuz        | 56.356, 25.199  |
| P003 | Ras Tanura      | Saudi Arabia | terminal | export             | Hormuz        | 50.157, 26.643  |
| P004 | Jurong Island   | Singapore    | refinery | transit / refining | Malacca       | 103.708, 1.274  |
| P005 | Houston         | USA          | port     | import / refining  | Panama        | −95.100, 29.736 |
| P006 | Ningbo-Zhoushan | China        | port     | import             | Malacca       | 121.982, 29.935 |
| P007 | Jamnagar        | India        | refinery | refining           | —             | 69.860, 22.345  |
| P008 | Basra           | Iraq         | terminal | export             | Hormuz        | 48.810, 29.681  |
| P009 | Ulsan           | South Korea  | refinery | refining           | Malacca       | 129.343, 35.433 |
| P010 | Kharg Island    | Iran         | terminal | export             | Hormuz        | 50.324, 29.231  |
| P011 | Yanbu           | Saudi Arabia | terminal | export             | Suez · Mandeb | 38.229, 23.961  |



### A.2.2 6 maritime chokepoints

Fixed node order (graph index 11–16), from EIA World Oil Transit Chokepoints.


| Short code | Chokepoint        |
| ---------- | ----------------- |
| `hormuz`   | Strait of Hormuz  |
| `suez`     | Suez Canal        |
| `malacca`  | Strait of Malacca |
| `mandeb`   | Bab el-Mandeb     |
| `panama`   | Panama Canal      |
| `cape`     | Cape of Good Hope |


---



## A.3 Publication-lag table

Every predictor enters the weekly (Friday-ending) matrix only after its
conservative real-time availability. Lags are fixed as constants at the top of
each builder. Flat and Deep share the same sources but differ for shipping.

### A.3.1 Flat arm


| Source                                                       | freq → weekly       | Lag                  | Constant · script                                     |
| ------------------------------------------------------------ | ------------------- | -------------------- | ----------------------------------------------------- |
| Daily finance (Brent/WTI/VIX/DXY/DGS10/DFF/S&P/gold/OVX/CAD) | daily → Fri last    | **0**                | `daily_to_weekly_last` · `build_m1_weekly.py`         |
| EIA WPSR fundamentals                                        | weekly → Fri        | **+1 w**             | `EIA_LAG_WEEKS=1` · `build_m1_weekly.py`              |
| GPR                                                          | daily → weekly mean | **+1 w**             | `GPR_LAG_WEEKS=1` · `build_m1_weekly.py`              |
| Monthly macro (REA, non-oil commodity)                       | month-end ffill     | **+5 w**             | `MONTHLY_LAG_WEEKS=5` · `build_m1_weekly.py`          |
| Sentinel-2 indices + VIIRS (M2)                              | monthly as-of       | **month-end + 15 d** | `PUB_LAG_DAYS=15` · `build_m2_weekly.py`              |
| PortWatch chokepoint/port flows                              | daily → Fri sum     | **+1 w**             | `PW_LAG_WEEKS=1` · `aggregate_shipping_to_weekly.py`  |
| GFW monthly presence (flat M3, 113 cols)                     | month-end ffill     | **+4 w**             | `GFW_LAG_WEEKS=4` · `aggregate_shipping_to_weekly.py` |


> Merge check: EIA already lagged at source, merge re-shift = 0
> (`EIA_WPSR_LAG_WEEKS=0`, `build_feature_matrix.py`).



### A.3.2 Deep arm (17-node graph)

Finance and RS identical to A.3.1 (Deep RS = Channel-A Prithvi embeddings, also
month-end + 15 d). Only the shipping graph differs.


| Graph stream                                         | Role                  | Lag      | Constant · script                                     |
| ---------------------------------------------------- | --------------------- | -------- | ----------------------------------------------------- |
| PortWatch node counts                                | node features         | **+1 w** | `PW_LAG_WEEKS=1` · `build_m3_graph_weekly.py`         |
| GFW events / voyages (O-D)                           | edges + node features | **+2 w** | `GFW_EVENT_LAG_WEEKS=2` · `build_m3_graph_weekly.py`  |
| GFW SAR dark-vessel                                  | node features         | **+4 w** | `SAR_LAG_WEEKS=4` · `build_m3_graph_weekly.py`        |
| GFW monthly presence (chokepoint node features)      | node features         | **+4 w** | `GFW_LAG_WEEKS=4` (inherited) · `build_m3_graph17.py` |
| EMODnet density (optional cross-check, not in model) | —                     | **+8 w** | `EMODNET_LAG_WEEKS=8` · `build_emodnet_weekly.py`     |



### A.3.3 Why GFW is +4 w (Flat) but +2 w (Deep)

Different GFW products, not the same stream lagged differently. **Flat +4 w** =
monthly vessel-presence columns (`gfw_{cp}_*`, 49 of the 113): a calendar month
is only complete at month end + a conservative ~~1-month availability buffer
(project-level conservatism, **not** an official 4-week release rule). **Deep
+2 w** = near-real-time AIS event/voyage O-D stream (~~96 h) with a conservative
two-week buffer. The two are not interchangeable.

### A.3.4 Lag robustness

GFW monthly presence testable at lag ∈ {1, 4, 8} w; `MONTHLY_LAG_WEEKS` at
{3, 5, 7} w; all exposed as CLI flags (`--gfw-lag`, `--eia-lag`, …) so the whole
matrix can be rebuilt without code edits. Results in Appendix B.

---



## A.4 Shipping graph edge definition

The Deep shipping branch encodes a **weekly 17-node heterogeneous graph**
(11 AOIs + 6 chokepoints, fixed order). Combined adjacency is (T, 17, 17),
averaging ~63.8 edges/week. Sources: `build_m3_graph17.py`,
`m3_data_dictionary.md` §12, `shipping_encoder.py`.

### A.4.1 Dynamic O-D voyage edges (AOI→AOI)

Directed AOI→AOI edges from GFW voyage counts; edge weight = `n_voyages` for
that week's directed lane (`from ≠ to`; self-loops removed to a node feature).
Different every week; 96 lanes, 106 992 voyages total (top lanes e.g.
Ningbo↔Singapore, Fujairah↔Singapore, Singapore↔Rotterdam). Directionality
verified (`P006→P004 ≠ P004→P006`). Lag +2 w.

### A.4.2 Static AOI↔chokepoint edges

Fixed undirected links by geographic association (12 undirected edges), present
every week (`aoi_oil_infrastructure_sites.md` §4):


| Chokepoint | Linked AOIs            |
| ---------- | ---------------------- |
| `hormuz`   | P002, P003, P008, P010 |
| `suez`     | P001, P011             |
| `malacca`  | P004, P006, P009       |
| `mandeb`   | P011                   |
| `cape`     | P001                   |
| `panama`   | P005                   |



### A.4.3 Adjacency handling & edge-weight transform

- **Combine**: dynamic O-D block (11×11) placed in the AOI sub-block; static
AOI↔chokepoint edges broadcast over all weeks → combined (T, 17, 17).
- **Symmetrise + self-loop**: for message passing the adjacency is symmetrised
and self-looped (dense 17×17 boolean mask; dense is simpler than sparse for
this tiny dynamic graph).
- **Edge-weight transform (attention prior)**: the O-D flow enters the GAT as
`log1p(flow)` scaled by a **learned gain** `edge_scale`, i.e. busy lanes get a
higher attention prior instead of the flow being discarded by the boolean
adjacency; the model can down-weight the prior if unhelpful.
- **Encoder**: type-specific projection (`F_aoi=11`, `F_choke=20` → `d_model=64`)
  - node-type embedding → 2-layer dense multi-head GAT (heads = 4) → causal TCN
  (lookback L) → node-attention pooling → 32-d `z_ship` (~42k params). Node-
  attention weights feed RQ3 (which port/chokepoint the branch weights).
  Encoder details in Appendix C.

---



# Appendix B — Extra results & robustness

> All checks share the single leakage-safe protocol (2019–2025, lookback 4,
> expanding rolling-origin, 257 common scored weeks, CW vs M1 / DM vs M0). They
> **reinforce** the main findings; they do not replace the main-analysis specs
> (§ Chapter 4). Sources are given per table so every number is traceable.

---



## B.1 Sub-period: early (≤2022) vs late (≥2023)

Split at 2023-01-01 (matching `run_deep_advanced.py`). Skill vs M0 in %.
Scored offline from saved predictions with `subperiod_eval.py` →
`05_outputs/baselines/subperiod/subperiod_summary.csv`; numbers reproduce the
main pipeline exactly.


| Model                      | full          | early (≤2022) | late (≥2023)  |
| -------------------------- | ------------- | ------------- | ------------- |
| M0 (random walk)           | 0             | 0             | 0             |
| M1_Flat Ridge / XGB        | −2.52 / −5.22 | −2.85 / −4.51 | −1.98 / −6.37 |
| M2_Flat Ridge / XGB        | −6.31 / −6.95 | −6.53 / −7.11 | −5.94 / −6.67 |
| M3_Flat Ridge / XGB        | −6.71 / −6.68 | −7.22 / −6.68 | −5.87 / −6.69 |
| M4_Flat Ridge / XGB        | −8.99 / −8.57 | −9.33 / −9.07 | −8.45 / −7.74 |
| M1_Deep                    | −2.36         | −1.33         | −4.02         |
| M_ship_GNN (shipping only) | −0.38         | −0.41         | −0.33         |
| M_rs_deep (RS only)        | −2.30         | −3.07         | −1.04         |
| **M3_Deep_gated (main)**   | **+0.11**     | **+0.09**     | **+0.14**     |
| M2_Deep_gated              | −2.43         | −3.06         | −1.41         |
| M4_Deep_gated              | −1.28         | −2.35         | +0.49         |
| M4_Deep_Concat             | −4.06         | −6.08         | −0.69         |


**Reading**: no flat model beats M0 in either sub-period. Among deep
models, **M3_Deep_gated is the only configuration with positive skill in both
sub-periods** (+0.09 / +0.14), i.e. the most stable small gain; M4_Deep_gated is
negative early and only turns positive late (+0.49), and cross-attention–driven
M4 gains are concentrated late. This supports gated finance+shipping as the main
model.

---



## B.2 Deep fusion matrix (3 modality combos × 3 fusions)

seed 42, lookback 4, 257 weeks. Source: `run_deep_fusion_matrix.py` →
`05_outputs/baselines/Deep/_cross/deep_fusion_matrix.csv`. Skill vs M0 in %.


| Combo                  | Concat | Gated     | Cross-Attn | CW p vs M0 (best) |
| ---------------------- | ------ | --------- | ---------- | ----------------- |
| **M3_Deep** (fin+ship) | +0.06  | **+0.11** | **+0.74**  | xattn 0.041       |
| M2_Deep (fin+rs)       | −1.93  | −2.43     | −5.89      | —                 |
| M4_Deep (fin+rs+ship)  | −4.06  | −1.28     | +0.12      | xattn 0.018       |


**Reading**: shipping (M3) is the only combo that clears M0 under any
fusion; adding RS (M2, M4) never helps at the concat/gated floor. Cross-attention
gives the single-seed peak but is less stable across seeds (see B.4), so gated is
the main reported fusion.

---



## B.3 Flat robustness



### B.3.1 M2 leave-one-AOI-out (LOAO)

Source: `05_outputs/baselines/Flat/M2_Flat/baseline_metrics_anom_loao.csv` (+ full
per-AOI dRMSE in `baseline_loao_anom.csv`). Removing any single AOI leaves the M2
result essentially unchanged (|dRMSE| small, no single site drives a positive
contribution), i.e. the weak RS signal is diffuse rather than one-site-driven.

### B.3.2 M3 leave-one-channel-out (LOCHO)

seed 42, 257 weeks. Source:
`05_outputs/baselines/Flat/M3_Flat/robustness_m3_summary.csv`. Skill vs M0 (%)
and CW p vs M1 (nested increment) for XGB.


| Arm                   | XGB skill vs M0 | XGB CW p vs M1 |
| --------------------- | --------------- | -------------- |
| full (113 cols, main) | −6.68           | **0.0002**     |
| core (38)             | −7.81           | 0.096          |
| portwatch-only        | −4.91           | **0.0003**     |
| gfw-only              | −5.70           | 0.047          |
| gfw-presence          | −4.99           | 0.039          |
| gfw-aggregate         | −4.62           | 0.094          |
| tanker-only           | −4.60           | **0.0018**     |


**Reading**: the nested shipping increment over M1 (CW p) is significant
across several channel subsets — strongest for tanker/PortWatch flows — so the
M3 signal is not an artefact of one data source, even though no flat arm beats M0
in absolute RMSE.

### B.3.3 M2 water-masked RS variant

Source: `baseline_metrics_anom_watermask.csv`. Masking water pixels lifts the M2 nested increment (XGB CW p vs M1 = 0.028  vs 0.085 un-masked) but M2 still does not beat M0 (skill −6.3%). De-noising only makes RS marginally significant → RS value is limited → motivates modality-aware fusion (RQ2).

### B.3.4 M4 leave-one-modality-out (LOMO)

Source: `Flat/M4_Flat/robustness_m4_summary.csv`. Dropping RS from M4 (i.e. M1+M3) keeps the significant nested increment (XGB CW p vs M1 = 0.0002 ), whereas the full M4 adds RS without accuracy gain — flat multi-modal concatenation cannot improve accuracy and significance together.

---



## B.4 Deep multi-seed & sweeps

Source: `04_code/scripts/deep/run_deep_sweep.py` → `deep_sweep_summary.csv` (seeds 42, 1, 2;
lookback 4; d 32).


| Config            | skill vs M0 (3 seeds) | Note                                                   |
| ----------------- | --------------------- | ------------------------------------------------------ |
| **finship gated** | −0.47% ± 0.86         | **lowest variance, most stable** → main model          |
| m4rep gated       | −0.89% ± 0.60         | adding RS gives no gain                                |
| m4 xattn          | −1.83% ± **2.76**     | seed 42 best but seed 2 collapses to −4.98% → not main |


Single-seed (42) hyper-sweep, finship gated: lookback 4/8/12 × d 32/64 →
lb 8 d 32 best (+0.34%) > lb 4 (+0.11%) > lb 12 (negative); **d 64 always worse**
(short weekly sample). Main model stays locked at lookback 4, d 32 for flat
protocol parity.

---



## B.5 Other robustness (documented, not tabulated here)

- **Lookback sweep** L1/8/12 per layer — `sweep_m`* + `deep_sweep`.
- **C2 dimensionality reduction** (PCA / ElasticNet) for M2 —
`c2_summary.csv`; increment not pure over-fitting.
- **feature-mode = returns** stationarised variant — numerical robustness.
- **min_train = 78 longer window** — Appendix-level; main protocol
keeps min_train = 104.

---



# Appendix C — Hyperparameter grids & locked settings

> This appendix pins every value needed to reproduce the reported results.
> §C.1 lists the software environment; §C.2 the shared leakage-safe protocol;
> §C.3 the Flat search grids; §C.4 the locked Deep architecture and training;
> §C.5 the entry-point scripts and output paths.

---



## C.1 Software environment

Reproduced with **Python 3.9.6** (CPython, macOS). Full pinned list in
`04_code/requirements.txt`; core packages:


| Package      | Version | Role                                                |
| ------------ | ------- | --------------------------------------------------- |
| numpy        | 2.0.2   | arrays                                              |
| pandas       | 2.3.3   | weekly matrix, joins                                |
| scipy        | 1.13.1  | DM / Clark-West test statistics                     |
| scikit-learn | 1.6.1   | Ridge, pipelines, VarianceThreshold, StandardScaler |
| xgboost      | 2.1.4   | Flat XGBoost learner                                |
| torch        | 2.8.0   | Deep encoders + fusion (CPU)                        |
| matplotlib   | 3.9.4   | figures                                             |
| shap         | 0.49.1  | Flat feature attribution                            |


> **On foundation models**: the Deep RS branch consumes
> **pre-computed frozen Prithvi-EO-2.0 embeddings** stored on disk; training and
> evaluation do **not** import `transformers` or fetch any model online (the test
> machine has no `transformers` installed). The one-off embedding-export step is
> separate and not part of this environment.

Install:

```bash
python3 -m pip install -r 04_code/requirements.txt
```

---



## C.2 Shared leakage-safe protocol (Flat = Deep)

Both families use the identical rolling-origin schedule so architecture
differences are not confounded with protocol differences.


| Item                          | Value                                                 |
| ----------------------------- | ----------------------------------------------------- |
| Comparison window             | 2019–2025 (365 weeks in the merged matrix)            |
| Lookback                      | **4 weeks**                                           |
| Warm-up `min_train`           | **104 weeks** (not scored)                            |
| Refit cadence `retrain_every` | **13 weeks**                                          |
| Inner-validation `val_weeks`  | **52 weeks** (tail of each training fold)             |
| Common scored test span       | **257 weeks** (2021-01 → 2025-12)                     |
| Target                        | one-week log return r_{t+1}, reconstructed to price   |
| Primary metric                | RMSE + MAE on reconstructed price; skill vs M0        |
| Nested test                   | Clark–West (vs M1, and vs M0 for "beats random walk") |
| Non-nested test               | Diebold–Mariano, HLN small-sample corrected           |
| Seed                          | **42** (main); 1, 2 for robustness                    |


---



## C.3 Flat search grids

Chosen inside each training fold on the past `val_weeks` slice only
(`backtest/rolling.py::tune_hyperparams`); no test-set peeking. Every pipeline
begins with `VarianceThreshold(0.0)` (drops in-fold constant columns), and the
linear model adds `StandardScaler` (fit on the training fold only). Source:
`04_code/src/backtest/models.py`.


| Learner                 | Grid                                                      |
| ----------------------- | --------------------------------------------------------- |
| Ridge (α)               | {0.1, 1.0, 10.0, 100.0, 1000.0} (default 10.0)            |
| XGBoost `max_depth`     | {2, 3}                                                    |
| XGBoost `learning_rate` | {0.03, 0.05}                                              |
| XGBoost `n_estimators`  | {200, 400}                                                |
| XGBoost fixed           | `subsample=0.8`, `colsample_bytree=0.8`, `reg_lambda=1.0` |


---



## C.4 Locked Deep architecture & training

Sweeps explored lookback ∈ {4, 8, 12}, representation dim ∈ {32, 64}, fusion ∈
{concat, gated, cross-attention}, dropout / weight-decay, and seeds; the main
specification is **locked to lookback = 4 and d = 32** for protocol parity with
the flat baselines. Sensitivity is reported in Appendix B. Sources:
`deep_rolling.py`, `finance_encoder.py`, `rs_encoder.py`, `shipping_encoder.py`,
`fusion.py`.

### C.4.1 Encoders


| Encoder             | Key settings                                                                                                                                                                                                                                                                      | Output                         |
| ------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------ |
| Finance TCN         | `d_model=32`, `tcn_layers=2`, `kernel=3`, causal, `dropout=0.1`                                                                                                                                                                                                                   | z_fin, **32-d**                |
| RS (frozen Prithvi) | `emb_dim=1024`, `n_sites=11`, `d_model=64`; temporal-attention + AOI-site-attention pooling                                                                                                                                                                                       | z_rs, **32-d**                 |
| Shipping graph      | 17 nodes (11 AOI + 6 chokepoints); type-specific projection + node-type embedding; **GAT layers = 2, heads = 4**; `log1p(O-D flow)` as attention prior (learned `edge_scale`); adjacency symmetrised + self-looped; then **TCN layers = 2**; node-attention pooling; `d_model=64` | z_ship, **32-d**; ≈ 42k params |



### C.4.2 Fusion


| Option                                            | Role                                                 |
| ------------------------------------------------- | ---------------------------------------------------- |
| encoder-concat                                    | fusion-ladder floor                                  |
| **gated** (softmax gate over modality embeddings) | **main reported model** (gate weights also feed RQ3) |
| cross-attention (finance as query, `n_heads=4`)   | comparative (single-seed best but higher variance)   |



### C.4.3 Training


| Item             | Value                                      |
| ---------------- | ------------------------------------------ |
| Optimiser        | Adam                                       |
| Learning rate    | `1e-3`                                     |
| Weight decay     | `1e-4`                                     |
| Dropout          | `0.1`                                      |
| Batch size       | `32`                                       |
| Epochs (max)     | `80`                                       |
| Early stopping   | on last `val_weeks` of fold, `patience=12` |
| Modality dropout | `0.0` main (`0.3` robustness arm)          |
| Device           | CPU                                        |
| Seed             | `42` (robustness: 1, 2)                    |


---



## C.5 Entry points & outputs


| Purpose                                        | Script                                                                       | Output dir                                                |
| ---------------------------------------------- | ---------------------------------------------------------------------------- | --------------------------------------------------------- |
| Flat M0–M4 baselines                           | `04_code/scripts/flat/run_baseline.py` (+ `flat/M{1..4}_Flat/*.py`)          | `05_outputs/baselines/Flat/M*_Flat/`                      |
| Deep baselines & fusion                        | `04_code/scripts/deep/run_deep_baseline.py`                                  | `05_outputs/baselines/Deep/{M*_Deep,_cross}/`             |
| Deep sweeps (seed/lookback/dim/reg)            | `04_code/scripts/deep/run_deep_sweep.py`                                     | `05_outputs/baselines/Deep/_cross/deep_sweep_summary.csv` |
| Fusion matrix (3×3)                            | `run_deep_fusion_matrix.py`                                                  | `deep_fusion_matrix.{csv,png}`                            |
| Advanced ablations (fusion/dropout/sub-period) | `run_deep_advanced.py`                                                       | `deep_advanced_summary.csv`                               |
| Sub-period early/late (Flat + Deep, offline)   | `subperiod_eval.py`                                                          | `05_outputs/baselines/subperiod/subperiod_summary.csv`    |
| Interpretability (gates, attention)            | `run_deep_interpret.py`, `run_deep_interpret_m3.py`, `run_deep_xattn_viz.py` | `deep_interpret*.png`, `deep_*gate*.csv`                  |
| Feature matrix build                           | `03_data/processed/**/build_*.py`, `merge/py/build_feature_matrix.py`        | `03_data/processed/merge/outputs/`                        |


Reproduce end-to-end:

```bash
python3 -m pip install -r 04_code/requirements.txt
python3 04_code/scripts/flat/run_baseline.py --modality M3      # flat example
python3 04_code/scripts/deep/run_deep_baseline.py               # deep main
python3 04_code/scripts/tools/subperiod_eval.py                  # early/late table
```

---



## References

Aas, K., Jullum, M. and Løland, A. (2021). ‘Explaining individual predictions when features are dependent: more accurate approximations to Shapley values’, *Artificial Intelligence*, 298, 103502. doi: 10.1016/j.artint.2021.103502.

Adland, R., Jia, H. and Strandenes, S.P. (2017). ‘Are AIS-based trade volume estimates reliable? The case of crude oil exports’, *Maritime Policy & Management*, 44(5), pp. 657–665. doi: 10.1080/03088839.2017.1309470.

Alquist, R., Kilian, L. and Vigfusson, R.J. (2013). ‘Forecasting the price of oil’, in Elliott, G. and Timmermann, A. (eds.) *Handbook of Economic Forecasting*. Vol. 2A. Amsterdam: Elsevier, pp. 427–507. doi: 10.1016/B978-0-444-53683-9.00008-6.

Arevalo, J., Solorio, T., Montes-y-Gómez, M. and González, F.A. (2017). ‘Gated multimodal units for information fusion’, *ICLR 2017 Workshop Track*. Toulon, France, 24–26 April. Available at: [https://openreview.net/forum?id=S12_nquOe](https://openreview.net/forum?id=S12_nquOe) (Accessed: 1 July 2026).

Arslanalp, S., Marini, M. and Tumbarello, P. (2019). *Big data on vessel traffic: nowcasting trade flows in real time*. IMF Working Paper WP/19/275. Washington, DC: International Monetary Fund. Available at: [https://www.imf.org/en/publications/wp/issues/2019/12/13/big-data-on-vessel-traffic-nowcasting-trade-flows-in-real-time-48837](https://www.imf.org/en/publications/wp/issues/2019/12/13/big-data-on-vessel-traffic-nowcasting-trade-flows-in-real-time-48837) (Accessed: 1 July 2026).

Arslanalp, S., Exton, O., Gao, C., Kamali, P., Saraiva, M., Sozzi, A. and Verschuur, J. (2026). *Nowcasting country-level trade estimates using IMF PortWatch*. IMF Working Paper WP/26/99. Washington, DC: International Monetary Fund. doi: 10.5089/9798229046893.001.

Bai, S., Kolter, J.Z. and Koltun, V. (2018). ‘An empirical evaluation of generic convolutional and recurrent networks for sequence modeling’, arXiv:1803.01271 [Preprint]. Available at: [https://arxiv.org/abs/1803.01271](https://arxiv.org/abs/1803.01271) (Accessed: 3 August 2026).

Baltrušaitis, T., Ahuja, C. and Morency, L.-P. (2019). ‘Multimodal machine learning: a survey and taxonomy’, *IEEE Transactions on Pattern Analysis and Machine Intelligence*, 41(2), pp. 423–443. doi: 10.1109/TPAMI.2018.2798607.

Baumeister, C. and Kilian, L. (2015). ‘Forecasting the real price of oil in a changing world: a forecast combination approach’, *Journal of Business & Economic Statistics*, 33(3), pp. 338–351. doi: 10.1080/07350015.2014.949342.

Bricongne, J.-C., Macalos, J., Meunier, B., Milis, J. and Pical, T. (2026). *Can satellites predict oil demand?* ECB Working Paper Series No. 3198. Frankfurt am Main: European Central Bank. Available at: [https://www.ecb.europa.eu/pub/pdf/scpwps/ecb.wp3198~e3858c52a3.en.pdf](https://www.ecb.europa.eu/pub/pdf/scpwps/ecb.wp3198~e3858c52a3.en.pdf) (Accessed: 1 July 2026).

Che, Z., Purushotham, S., Cho, K., Sontag, D. and Liu, Y. (2018). ‘Recurrent neural networks for multivariate time series with missing values’, *Scientific Reports*, 8, 6085. doi: 10.1038/s41598-018-24271-9.

Chen, T. and Guestrin, C. (2016). ‘XGBoost: a scalable tree boosting system’, in *Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining*. San Francisco, CA: Association for Computing Machinery, pp. 785–794. doi: 10.1145/2939672.2939785.

Clark, T.E. and West, K.D. (2007). ‘Approximately normal tests for equal predictive accuracy in nested models’, *Journal of Econometrics*, 138(1), pp. 291–311. doi: 10.1016/j.jeconom.2006.05.023.

Cong, Y., Khanna, S., Meng, C., Liu, P., Rozi, E., He, Y., Burke, M., Lobell, D.B., et al. (2022). ‘SatMAE: pre-training transformers for temporal and multi-spectral satellite imagery’, *Advances in Neural Information Processing Systems*, 35, pp. 197–211.

Costa, A.B.R., Ferreira, P.C.G., Gaglianone, W.P., Guillén, O.T.C., Issler, J.V. and Lin, Y. (2021). ‘Machine learning and oil price point and density forecasting’, *Energy Economics*, 102, 105494. doi: 10.1016/j.eneco.2021.105494.

Diebold, F.X. and Mariano, R.S. (1995). ‘Comparing predictive accuracy’, *Journal of Business & Economic Statistics*, 13(3), pp. 253–263. doi: 10.1080/07350015.1995.10524599.

Foroutan, P. and Lahmiri, S. (2024). ‘Deep learning systems for forecasting the prices of crude oil and precious metals’, *Financial Innovation*, 10, 111. doi: 10.1186/s40854-024-00637-z.

Fuller, A., Millard, K. and Green, J.R. (2023). ‘CROMA: remote sensing representations with contrastive radar-optical masked autoencoders’, *Advances in Neural Information Processing Systems*, 36, pp. 5506–5538.

Gneiting, T. and Raftery, A.E. (2007). ‘Strictly proper scoring rules, prediction, and estimation’, *Journal of the American Statistical Association*, 102(477), pp. 359–378. doi: 10.1198/016214506000001437.

Gibson, J., Olivia, S., Boe-Gibson, G. and Li, C. (2021). ‘Which night lights data should we use in economics, and where?’, *Journal of Development Economics*, 149, 102602. doi: 10.1016/j.jdeveco.2020.102602.

Gohari, H.E., Dang, X.-H., Shah, S.Y. and Zerfos, P. (2024). ‘Modality-aware transformer for financial time series forecasting’, in *Proceedings of the 5th ACM International Conference on AI in Finance (ICAIF ’24)*. New York: Association for Computing Machinery, pp. 677–685. doi: 10.1145/3677052.3698654.

Hao, X. and Wang, Y. (2023). ‘Cloud cover and expected oil returns’, *Humanities and Social Sciences Communications*, 10, 605. doi: 10.1057/s41599-023-02128-5.

Hoerl, A.E. and Kennard, R.W. (1970). ‘Ridge regression: biased estimation for nonorthogonal problems’, *Technometrics*, 12(1), pp. 55–67. doi: 10.1080/00401706.1970.10488634.

Jain, S. and Wallace, B.C. (2019). ‘Attention is not explanation’, *Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies*. Minneapolis, MN: Association for Computational Linguistics, pp. 3543–3556. doi: 10.18653/v1/N19-1357.

Jung, Y. (2026). ‘Watching trade from space: nowcasting and spatial extrapolation of port-level maritime trade using satellite imagery’, arXiv:2604.15444 [Preprint]. Available at: [https://arxiv.org/abs/2604.15444](https://arxiv.org/abs/2604.15444) (Accessed: 1 July 2026).

Kilian, L. (2009). ‘Not all oil price shocks are alike: disentangling demand and supply shocks in the crude oil market’, *American Economic Review*, 99(3), pp. 1053–1069. doi: 10.1257/aer.99.3.1053.

Liang, M., Liu, R.W., Zhan, Y., Li, H., Zhu, F. and Wang, F.-Y. (2022). ‘Fine-grained vessel traffic flow prediction with a spatio-temporal multigraph convolutional network’, *IEEE Transactions on Intelligent Transportation Systems*, 23(12), pp. 23694–23707. doi: 10.1109/TITS.2022.3199160.

Lundberg, S.M. and Lee, S.-I. (2017). ‘A unified approach to interpreting model predictions’, *Advances in Neural Information Processing Systems*, 30, pp. 4765–4774. Available at: [https://papers.nips.cc/paper/7062-a-unified-approach-to-interpreting-model-predictions](https://papers.nips.cc/paper/7062-a-unified-approach-to-interpreting-model-predictions) (Accessed: 1 July 2026).

Ma, M., Ren, J., Zhao, L., Testuggine, D. and Peng, X. (2022). ‘Are multimodal transformers robust to missing modality?’, in *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*. New Orleans, LA: IEEE, pp. 18177–18186. doi: 10.1109/CVPR52688.2022.01764.

Mi, J.J., Meng, X., Chen, Y. and Wang, Y. (2022). ‘The impact of the crude oil price on tankers’ port-call features: mining the information in automatic identification system’, *Journal of Marine Science and Engineering*, 10(10), 1559. doi: 10.3390/jmse10101559.

Mi, J.J., Zang, X., Lo, K.L. and Chen, Y. (2023). ‘The nonlinear relationship between oil prices and the number of tankers’ port calls: evidence from AIS data’, *Procedia Computer Science*, 221, pp. 870–877. doi: 10.1016/j.procs.2023.08.063.

Neverova, N., Wolf, C., Taylor, G.W. and Nebout, F. (2016). ‘ModDrop: adaptive multi-modal gesture recognition’, *IEEE Transactions on Pattern Analysis and Machine Intelligence*, 38(8), pp. 1692–1706. doi: 10.1109/TPAMI.2015.2461544.

Ouyang, Q., Sun, T., Xue, Y. and Liu, Z. (2022). ‘Long short-term memory and graph convolution network for forecasting the crude oil traffic flow’, *IEEE Access*, 10, pp. 18922–18932. doi: 10.1109/ACCESS.2022.3150852.

Paolo, F.S., Kroodsma, D., Raynor, J., Hochberg, T., Davis, P., Cleary, J., Marsaglia, L., Orofino, S., et al. (2024). ‘Satellite mapping reveals extensive industrial activity at sea’, *Nature*, 625, pp. 85–91. doi: 10.1038/s41586-023-06825-8.

Patton, A.J. (2011). ‘Volatility forecast comparison using imperfect volatility proxies’, *Journal of Econometrics*, 160(1), pp. 246–256. doi: 10.1016/j.jeconom.2010.03.034.

Pesaran, M.H. and Timmermann, A. (1992). ‘A simple nonparametric test of predictive performance’, *Journal of Business & Economic Statistics*, 10(4), pp. 461–465. doi: 10.1080/07350015.1992.10509922.

Polinov, S., Bookman, R. and Levin, N. (2022). ‘A global assessment of night lights as an indicator for shipping activity in anchorage areas’, *Remote Sensing*, 14(5), 1079. doi: 10.3390/rs14051079.

Shukla, S.N. and Marlin, B.M. (2021). ‘Multi-time attention networks for irregularly sampled time series’, *International Conference on Learning Representations (ICLR 2021)*. Online, 3–7 May. Available at: [https://openreview.net/forum?id=4c0J6lwQ4](https://openreview.net/forum?id=4c0J6lwQ4) (Accessed: 1 July 2026).

Simsek, A.I., Bulut, E., Gur, Y.E. and Gültekin Tarla, E. (2024). ‘A novel approach to predict WTI crude spot oil price: LSTM-based feature extraction with Xgboost regressor’, *Energy*, 309, 133102. doi: 10.1016/j.energy.2024.133102.

Small, C. (2021). ‘Spatiotemporal characterization of VIIRS night light’, *Frontiers in Remote Sensing*, 2, 775399. doi: 10.3389/frsen.2021.775399.

Szwarcman, D., Roy, S., Fraccaro, P., Gíslason, Þ.E., Blumenstiel, B., Ghosal, R., de Oliveira, P.H., de Sousa Almeida, J.L., et al. (2026). ‘Prithvi-EO-2.0: a versatile multitemporal foundation model for earth observation applications’, *IEEE Transactions on Geoscience and Remote Sensing*, 64, 4400120. doi: 10.1109/TGRS.2025.3642610.

U.S. Energy Information Administration (2014). ‘Benchmarks play an important role in pricing crude oil’, *Today in Energy*, 28 October. Available at: [https://www.eia.gov/todayinenergy/detail.php?id=18571](https://www.eia.gov/todayinenergy/detail.php?id=18571) (Accessed: 11 August 2026).

Veličković, P., Cucurull, G., Casanova, A., Romero, A., Liò, P. and Bengio, Y. (2018). ‘Graph attention networks’, *International Conference on Learning Representations (ICLR 2018)*. Vancouver, Canada, 30 April–3 May. Available at: [https://openreview.net/forum?id=rJXMpikCZ](https://openreview.net/forum?id=rJXMpikCZ) (Accessed: 3 August 2026).

Wang, T., Li, Y., Yu, S. and Liu, Y. (2019). ‘Estimating the volume of oil tanks based on high-resolution remote sensing images’, *Remote Sensing*, 11(7), 793. doi: 10.3390/rs11070793.

Wittner, M. (2020). *Global crude benchmarks: Brent sets the standard*. Intercontinental Exchange. Available at: [https://www.ice.com/global-crude-benchmarks-brent-sets-the-standard](https://www.ice.com/global-crude-benchmarks-brent-sets-the-standard) (Accessed: 11 August 2026).

Wu, Z., Pan, S., Long, G., Jiang, J. and Zhang, C. (2019). ‘Graph WaveNet for deep spatial-temporal graph modeling’, in *Proceedings of the Twenty-Eighth International Joint Conference on Artificial Intelligence (IJCAI-19)*. Macao, China, 10–16 August. International Joint Conferences on Artificial Intelligence Organization, pp. 1907–1913. doi: 10.24963/ijcai.2019/264.

Yan, Z., Xiao, Y., Cheng, L., Chen, S., Zhou, X., Ruan, X., Li, M., He, R., et al. (2020). ‘Analysis of global marine oil trade based on automatic identification system (AIS) data’, *Journal of Transport Geography*, 83, 102637. doi: 10.1016/j.jtrangeo.2020.102637.

Yılmaz, T.E. and Zehir, C. (2026). ‘Strategic risk based forecasting of Brent crude oil prices: a comparative analysis of econometric and machine learning models’, *Entropy*, 28(5), 539. doi: 10.3390/e28050539.

Zhao, C., Li, X., Zuo, M., Mo, L. and Yang, C. (2022). ‘Spatiotemporal dynamic network for regional maritime vessel flow prediction amid COVID-19’, *Transport Policy*, 129, pp. 78–89. doi: 10.1016/j.tranpol.2022.09.029.

Zhao, G., Xue, M. and Cheng, L. (2023). ‘A new hybrid model for multi-step WTI futures price forecasting based on self-attention mechanism and spatial–temporal graph neural network’, *Resources Policy*, 85, 103956. doi: 10.1016/j.resourpol.2023.103956.
