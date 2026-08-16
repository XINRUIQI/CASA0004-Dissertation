# A Modality-Aware Spatio-Temporal Fusion Framework for Brent Crude Oil Forecasting Using Financial Time Series, Satellite Imagery and Maritime Networks

---

## Abstract *(~200 words)*

Brent crude is one of the principal benchmarks for internationally traded oil and a key reference price in the global energy market. Its short-term movements affect energy costs, inflation, trade balances and fiscal revenues, and therefore influence decisions made by firms and governments. Using weekly data from 2019 to 2025, this study examines whether satellite remote-sensing and shipping data provide incremental value beyond financial time series for one-week-ahead Brent price forecasting. It also compares flat feature fusion with modality-specific encoding followed by fusion. Flat models combine all selected inputs in a single feature table, whereas deep models encode each data source separately before fusing the resulting representations. For both model families, the study compares a financial-time-series-only specification with alternatives that add remote sensing, shipping or both.

The models are evaluated against a no-change benchmark that sets next week’s price equal to this week’s price. The results show that no flat model outperforms this benchmark, although shipping data provide limited evidence of incremental predictive information. Deep models combining financial time series and shipping data achieve a small improvement over the benchmark. Remote-sensing data provide no clear additional benefit. The advantage of deep models over flat models is most evident when shipping data are included. This study further uses modality gates to show which data sources the best-performing deep model relies on most. Overall, predictive value depends more on how multimodal data are used—especially how modalities are encoded and fused—than on simply adding more data.

---


## Chapter 1 — Introduction *(~600 words)*

### 1.1 Importance and background

Crude oil occupies a central place in the global economy and energy system. Oil-price movements affect inflation, trade balances, fiscal revenues in producer countries and the operating costs of energy-intensive industries. These effects spread through financial markets, economic activity and supply chains. They therefore shape the risk management, hedging, budgeting and planning decisions of governments, firms and investors.

Crude oil is not a homogeneous commodity: individual grades differ in density, sulphur content, production location and transport accessibility, and their prices are commonly expressed relative to a small number of benchmarks. Among the most widely used benchmarks are Brent, West Texas Intermediate (WTI) and Dubai/Oman (U.S. Energy Information Administration, 2014). Brent is a benchmark complex rooted in light, low-sulphur, waterborne crude oils from the North Sea. It is widely used as a reference for internationally traded crude. WTI is a US crude benchmark, with pricing centred on Cushing, Oklahoma, while Dubai/Oman is commonly used to price Middle Eastern crude exported to Asian markets (Wittner, 2020). Although Brent and WTI respond to many of the same global market conditions, differences in regional supply, inventory levels and transport constraints can cause their prices to diverge. This dissertation forecasts Brent because its role as an international waterborne benchmark aligns more closely with the ports, shipping routes and maritime chokepoints represented in the alternative data. WTI is nevertheless retained as a financial predictor and as a component of the Brent–WTI spread. No fixed volatility ranking between Brent and WTI is assumed. Whether shipping activity contains incremental predictive information for Brent is tested empirically in this dissertation.

Recent years have shown the costs of unexpected oil-price movements. The COVID-19 period brought an abrupt collapse in demand and an uneven recovery. The 2022 energy crisis then produced major supply and price shocks, followed by only partial normalisation amid continued geopolitical and macroeconomic uncertainty. More recent geopolitical disruptions have further shown how quickly oil prices and seaborne trade can respond when key maritime chokepoints are disrupted or bypassed. Governments monitor such shocks for inflation control, fiscal planning, energy security and trade policy. A better short-term oil-price model would help them gauge risk and timing. Although they cannot replace market judgement, such forecasts could support decision-making when physical flows and prices move together.

At the weekly horizon, the no-change forecast is difficult to outperform. This simple method predicts that next week’s price will be equal to this week’s price. It therefore provides a demanding benchmark for alternative data and methods. A model should not be considered useful merely because it outperforms a weaker or differently specified competitor. It must also be evaluated directly against the no-change benchmark.

The three data sources considered in this dissertation provide complementary views of the oil system. Financial, macroeconomic and oil-market variables describe changes in market conditions over time. Remote sensing captures spatial activity at selected oil-related sites through spectral indicators, night-time lights and image representations. AIS and PortWatch data describe changes in vessel activity across ports and major chokepoints. They also capture network relationships between locations and provide proxies for seaborne trade flows and congestion. In this dissertation, multimodal forecasting refers to combining temporal market data, spatial Earth-observation data and spatiotemporal shipping-network data in the same forecasting task.

These data create two practical challenges. First, the signals are noisy and arrive on different schedules. They may also respond to oil prices rather than predict them, while the available weekly sample is relatively small. Second, a common approach places all heterogeneous inputs in a single feature table before modelling. This flat feature-fusion approach is convenient for conventional models such as Ridge regression and gradient-boosted trees. However, it does not explicitly preserve the temporal structure of financial time series, the site structure of remote sensing or the network structure of shipping data.

This dissertation therefore addresses two empirical questions. First, do remote-sensing and shipping data improve one-week-ahead Brent price forecasts when they are added to financial time series? Can the resulting models outperform the no-change benchmark? Second, when the underlying data remain the same, does encoding each modality separately before fusion perform better than combining all inputs in a single feature table? The next section presents the study aim and formal research questions.

### 1.2 Aim and research questions

The main aim of this dissertation is to develop a reproducible comparison framework for evaluating how different data sources and model designs perform in one-week-ahead Brent price forecasting. The framework combines financial time-series data, satellite remote-sensing data and shipping data. It uses a rolling-origin forecasting design that prevents the use of future information and applies formal statistical tests to compare predictive performance. Flat feature fusion places all inputs in a single feature table before modelling. Representation-level fusion encodes each modality separately and then combines the resulting representations. This framework enables consistent comparisons of the incremental value of different data sources and the effects of different fusion designs. The empirical setting uses weekly Friday-ending Brent spot prices from 2019 to 2025, spanning periods of demand, supply and maritime disruption, with remote-sensing and shipping inputs covering eleven oil-related monitoring sites and six major maritime chokepoints.

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


# Chapter 3 — Methodology

## 3.1 Research design

The baseline for this study is a simple no-change benchmark, which predicts that next week’s Brent price will equal this week’s price. All learned models are evaluated against this benchmark. The available information is divided into different information sets, and models are trained and evaluated separately on these sets to examine whether different types of data provide useful predictive information and whether different ways of combining information affect forecasting performance. All specifications share the same weekly forecast dates, sample window and evaluation rules.

The no-change benchmark is denoted M0. At each forecast origin t, where P_t is the Brent price in week t, M0 sets the one-week-ahead price forecast equal to the current weekly price:

\hat{P}_{t+1|t}=P_t.

The predictors are organised into four information sets. S1 contains financial time series only, including financial, macroeconomic and oil-market variables. S2 adds remote sensing to S1, S3 adds shipping to S1, and S4 adds both modalities. S2 and S3 are parallel extensions of S1 rather than successive stages, while S4 combines the two. Table 3.1 lists the four sets together with the M0 benchmark.

**Table 3.1 — Information sets**

| Set            | Variables                                                                   |
| -------------- | --------------------------------------------------------------------------- |
| Benchmark (M0) | Last week's price                                                           |
| S1             | Financial time series only (financial, macroeconomic and oil-market series) |
| S2             | S1 + remote sensing                                                         |
| S3             | S1 + shipping                                                               |
| S4             | S1 + remote sensing + shipping                                              |

| ------ | ----------------- |

Two model families are applied to these information sets. TThe Flat family first combines all selected predictors into a weekly feature table. The most recent weeks are then flattened into a single row and used to fit Ridge and XGBoost. This early joining of features is called flat feature fusion. The Deep family initially keeps each data type separate. Financial, remote-sensing and shipping data are encoded independently. The resulting representations are then combined, with gated fusion learning how much weight to assign to each data type. Flat and Deep are compared using the same information set and evaluation sample. Because the two families also differ in model class, capacity and some modality-specific representations, these comparisons are interpreted as comparisons between overall modelling strategies rather than as tests of fusion alone. Fusion is assessed more directly within the Deep family by comparing simple concatenation, gated fusion and cross-attention while holding the encoders and inputs fixed. Together, these comparisons address RQ2.

Within each model family, comparisons across S1–S4 use the same forecasting method and evaluation sample; only the information included in the model changes. S2 against S1 measures the contribution of adding remote sensing alone, while S3 against S1 measures the contribution of shipping. S4 against S1 measures their joint contribution. S4 against S3 and S4 against S2 examine whether each source still adds useful information once the other is already included. These comparisons, together with each model’s comparison against M0, address RQ1.

RQ3 is restricted to Deep specifications that outperform M0 under the predefined criterion. For these specifications, the study reports the weights assigned to finance, remote sensing and shipping data, and examines how model attention patterns vary across different market conditions. These quantities are used to describe which information the models rely on.

Figure 3.1 summarises the research design.

Figure 3.1

**Figure 3.1 — Research design: data blocks, the M0 benchmark and information sets S1–S4, the Flat and Deep families, and the shared expanding-window evaluation.**

## 3.2 Prediction target and sample period

Let P_t denote the last available daily Brent spot-price observation in week t, where each week ends on Friday, measured in US dollars per barrel. The forecast target is next week’s price P_{t+1}. Models are not trained directly on the price level. They predict the one-week logarithmic return

r_{t+1}=\log\left(\frac{P_{t+1}}{P_t}\right)

and reconstruct the price forecast as

\hat{P}*{t+1|t}=P_t\exp\left(\hat{r}*{t+1|t}\right).

Log returns are used to reduce the strong persistence in the price level and to express the forecasting task in terms of proportional weekly changes. RMSE and the percentage improvement in RMSE over M0 are computed from the reconstructed price forecasts. Under this mapping, the no-change benchmark \hat{P}*{t+1|t}=P_t is exactly the same as forecasting a zero return \hat{r}*{t+1|t}=0.

The modelling window covers 2019–2025 and provides a common weekly index of 365 observations (4 January 2019 to 26 December 2025). The training and evaluation samples are separated in time on an expanding window, rather than by random assignment, to prevent future information from leaking into model fitting. The full validation protocol is in Section 3.6.

## 3.3 Geographic scope and monitoring sites

Because the prediction target is the global Brent benchmark rather than a local physical cargo price at a single terminal, the study does not use one specific study region. Spatial information instead comes from eleven oil-infrastructure monitoring sites and six maritime chokepoints. Together they cover major supply, transit, refining and demand locations in the international oil system. Figure 3.3 places these sites and chokepoints on a world map. Full site names, coordinates, patch sizes and graph edge definitions are in Appendix A.

The eleven sites comprise ports, refineries and export terminals selected purposively for their strategic roles and observability in the available satellite products.Flat remote-sensing features are summarised within a circular buffer of 5 km radius around each site. Deep image patches are centred on the same sites but vary in size by facility type and local spatial constraints. The sizes are generally larger for ports, intermediate for refineries and smaller for terminals.

The shipping graph augments the eleven sites with six maritime chokepoints: the Strait of Hormuz, the Suez Canal, the Strait of Malacca, Bab el-Mandeb, the Panama Canal and the Cape of Good Hope. The resulting weekly graph contains seventeen nodes and two forms of connection. Dynamic links between the eleven AOIs are directed origin–destination pairs, weighted by the number of voyages counted in each week from Global Fishing Watch (GFW) port-visit sequences. Fixed links are undirected. Each site is connected to the chokepoint or chokepoints on its main documented oil-trade corridor. These links are defined in advance rather than inferred from weekly vessel movements or geographic proximity. Complete edge definitions are reported in Appendix A.4, and graph encoding is described in Section 3.5.2.

Figure 3.3

**Figure 3.3 — Spatial coverage of the 11 oil-infrastructure AOIs, six maritime chokepoints and fixed AOI–chokepoint corridor links used in the shipping graph.**

## 3.4 Data sources and preparation

### 3.4.1 Data sources

The study uses three broad types of data: financial time series, satellite remote sensing and maritime shipping data. These three data types are referred to as the financial, remote-sensing and shipping data blocks. Flat and Deep models cover the same monitoring locations, but use different products and representations for the remote-sensing and shipping blocks (Table 3.2). Flat models organise the data in a merged weekly feature table, whereas Deep models retain modality-specific sequences, image embeddings and graph inputs.

**Table 3.2 — Datasets, variables and sources**

| Modality              | Dataset / product                                                                     | Key variables                                                                                                                  | Source                                                                                                                                                                                                                                                |
| --------------------- | ------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Financial time series | Oil-market and macro-financial series at daily, weekly and monthly native frequencies | Prices, inventories, production, interest rates, GPR and related indicators                                                    | [EIA](https://www.eia.gov/petroleum/supply/weekly/); [FRED](https://fred.stlouisfed.org/); [Yahoo Finance](https://finance.yahoo.com/); [Dallas Fed IGREA](https://www.dallasfed.org/research/igrea); [GPR](https://www.matteoiacoviello.com/gpr.htm) |
| Remote sensing (Flat) | Sentinel-2 optical indices and VIIRS night-time lights                                | Site-level anomalies at 11 AOIs (NDVI, NDWI, NDBI, BSI; NTL)                                                                   | [Sentinel-2 via GEE](https://developers.google.com/earth-engine/datasets/catalog/COPERNICUS_S2_SR_HARMONIZED); [VIIRS via GEE](https://developers.google.com/earth-engine/datasets/catalog/NOAA_VIIRS_DNB_MONTHLY_V1_VCMSLCFG)                        |
| Remote sensing (Deep) | Monthly Sentinel-2 image patches                                                      | Frozen Prithvi-EO-2.0 embeddings at the same 11 AOIs                                                                           | [Prithvi-EO-2.0](https://huggingface.co/ibm-nasa-geospatial); [Sentinel-2 via GEE](https://developers.google.com/earth-engine/datasets/catalog/COPERNICUS_S2_SR_HARMONIZED)                                                                           |
| Shipping (Flat)       | PortWatch and Global Fishing Watch AIS and SAR data                                   | Port and chokepoint tanker flows; vessel-activity features                                                                     | [IMF PortWatch](https://portwatch.imf.org/) (AIS-derived); [Global Fishing Watch](https://globalfishingwatch.org/our-apis/) (AIS- and SAR-derived)                                                                                                    |
| Shipping (Deep)       | PortWatch and Global Fishing Watch AIS and SAR data                                   | Weekly node attributes, dynamic voyage links and fixed corridor links for a 17-node graph comprising 11 AOIs and 6 chokepoints | [IMF PortWatch](https://portwatch.imf.org/) (AIS-derived); [Global Fishing Watch](https://globalfishingwatch.org/our-apis/) (AIS- and SAR-derived)                                                                                                    |

| -------- | ----------------------------------------------- | ------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |

The financial block combines oil-market and macro-financial series from the US Energy Information Administration (EIA), Federal Reserve Economic Data (FRED), Yahoo Finance, the Dallas Fed Index of Global Real Economic Activity, and the Caldara–Iacoviello geopolitical risk (GPR) index. These series are observed at different native frequencies and include crude prices and spreads, inventories, production and refinery activity, volatility and risk measures, interest rates, exchange rates, and futures-based oil indicators.

For the remote-sensing block, the Flat and Deep pathways extract inputs from the same eleven AOIs, but differ in their spatial coverage, products and representations. The Flat pathway uses Sentinel-2 optical indices and VIIRS night-time lights, whereas the Deep pathway uses frozen Prithvi-EO-2.0 embeddings derived from Sentinel-2 image patches and has no separate VIIRS input stream.

The shipping block covers activity at the eleven AOIs and six chokepoints. IMF PortWatch supplies tanker-flow measures at ports and chokepoints. Global Fishing Watch supplies measures of vessel presence and activity duration. These variables serve as proxies for physical shipping activity, tanker movements and congestion. In the Flat pathway, PortWatch and GFW enter as weekly tabular features. In the Deep pathway, the 17-node graph uses PortWatch flows, GFW AIS vessel-presence and activity measures, and GFW SAR-derived dark-vessel detections as node attributes, while dynamic AOI–AOI voyage links are constructed from GFW port-visit sequences. Full variable definitions are reported in Appendix A.1.

### 3.4.2 Temporal alignment

All series are aligned to a common Friday-ending weekly calendar. Daily observations are converted using end-of-week values, weekly means or weekly sums as appropriate, while monthly series are carried forward only after their assumed availability dates. Monthly remote-sensing products are aligned to their conservative availability dates, with the most recent eligible composite carried forward to avoid look-ahead bias.

Publication timing is approximated using source- and product-specific fixed lag buffers rather than observation-level release timestamps. Exact aggregation rules, lag constants and implementation scripts are reported in Appendix A.3. One-week buffers are applied to EIA fundamentals and PortWatch flows, while monthly macroeconomic series, remote-sensing products and individual GFW AIS and SAR products receive longer buffers.

### 3.4.3 Data quality and missing values

Monthly optical composites are cloud-filtered before the indices are constructed, and cloud-quality indicators are not used as predictors. On the weekly calendar, mean coverage across the four optical indices is approximately 97 per cent, while VIIRS night-time-light anomalies are fully observed. Site-level coverage and counts of independent monthly composites are reported in Appendix A.5.

After temporal alignment, the remaining gaps occur almost entirely before each series’ first valid observation. These leading gaps are set to zero for remote-sensing variables and filled with the training-fold median for each shipping-count variable.Deep finance inputs contain no missing values after merging. Missing remote-sensing embeddings and shipping-graph values are set to zero after scaling. All imputation and scaling parameters are estimated separately within each training window.

## 3.5 Forecasting models

### 3.5.1 Flat models

Flat models implement early feature-level fusion. For each information set, all available numeric features are concatenated into a weekly feature table, and the most recent four weeks are flattened into a single row for each forecast origin. Ridge applies L2 regularisation (Hoerl and Kennard, 1970) and serves as a transparent linear comparator. XGBoost is a non-linear gradient-boosted tree ensemble (Chen and Guestrin, 2016) that captures nonlinearities and interactions not represented by Ridge. Because both learners operate on the same flattened table, neither preserves modality-specific structure. Both predict the one-week-ahead log return and then reconstruct the corresponding price forecast. Hyperparameter selection follows the time-ordered procedure described in Section 3.6, with exact search grids reported in Appendix C.

### 3.5.2 Deep models

Deep models encode each modality separately and fuse the resulting representations for S2–S4.

The finance encoder applies a causal temporal convolutional network (TCN; Bai, Kolter and Koltun, 2018) to the four-week financial sequence retained across S1–S4. It produces one finance representation per forecast origin using only current and earlier positions at each convolutional layer.

The remote-sensing encoder receives monthly embeddings for the 11 AOIs, extracted from Sentinel-2 Surface Reflectance Harmonized patches using a frozen Prithvi-EO-2.0-300M encoder. The patches are adapted to the six-band convention of the HLS-pretrained encoder, with band mapping, standardisation and resampling documented in Appendix A. Temporal attention combines the four-week embeddings for each AOI. Site attention then combines the 11 AOIs into one remote-sensing representation.

The shipping encoder applies a graph attention network with temporal encoding (GAT; Veličković et al., 2018) to the weekly 17-node graph over the four-week lookback, producing one shipping representation per forecast origin. The graph is constructed as described in Section 3.3. For message passing, the directed voyage links and undirected corridor links are combined in a symmetrised adjacency matrix, so the encoder does not retain edge direction or type. Symmetrised voyage counts are used as a prior in the attention calculation. Adjacency and edge-weighting details are reported in Appendix A.4.3.

Fusion is applied only to Deep models for S2–S4, while S1 passes its finance representation directly to the regression head. Three mechanisms are compared. Gated fusion is designated as the main design because it provides forecast-origin-specific modality weights for the subsequent interpretability analysis. These weights are non-negative and sum to one at each forecast origin. Encoder concatenation and cross-attention are used only as alternatives. The resulting representation is trained by mean squared error to predict the one-week-ahead log return. Fixed fusion settings are reported in Appendix C.4.2–C.4.3.

Figure 3.5 summarises the three encoders, the fusion stage used for S2–S4, and the regression head.

Figure 3.5

**Figure 3.5 — Deep model architecture: modality-specific encoders, fusion (gated fusion as the main specification) and the regression head that predicts the one-week-ahead log return.**

## 3.6 Estimation and validation

### 3.6.1 Expanding-window estimation and re-estimation

With a four-week input window and a one-week forecast horizon, the 365 weekly observations yield 361 eligible input–target sequences. The first three observations cannot yet form a complete four-week input sequence, and the final week serves only as a target rather than a forecast origin. The first 104 eligible sequences form the initial training period and are not included in the evaluation metrics. The evaluation span covers 257 weeks from 22 January 2021 to 19 December 2025, with corresponding target dates from 29 January 2021 to 26 December 2025. Flat and Deep share this evaluation calendar.

A one-week-ahead forecast is produced at every forecast origin t, using only information available by that date. Each model is first estimated at the start of the evaluation period and re-estimated every 13 forecast origins as the training window expands. Between re-estimations, the fitted model and preprocessing parameters remain fixed, while the input window is updated at every forecast origin. This produces 20 estimation blocks: the first 19 contain 13 forecast origins each, and the final block contains 10. At each re-estimation, the training sample includes only observations whose following week’s target price is already known. Target prices that become available later are added at the next scheduled re-estimation. All preprocessing parameters calculated from the data are estimated using only the corresponding training sample.

Figure 3.2 presents the full schedule, while Figure 3.4 illustrates a single re-estimation origin.

Figure 3.4

**Figure 3.4 — A re-estimation origin showing the training fold, inner-validation weeks, four-week input window and one-week-ahead target. Between re-estimations, only the as-of input window advances.**

### 3.6.2 Model and Hyperparameter selection

Flat models re-select Ridge and XGBoost hyperparameters at each scheduled re-estimation, then refit on the full estimation sample available at that date. Deep models instead use a configuration fixed before evaluation, including the latent size. They share the four-week lookback used by the Flat models, which covers approximately one update cycle of the monthly remote-sensing and macroeconomic inputs. At each Deep re-estimation, inner validation is used for early stopping, and the checkpoint with the lowest validation loss is retained. The selected Deep checkpoint is used for the following forecasting. The model is not retrained using the combined training and validation data. Sensitivity analyses using the evaluation sample are reported separately in Appendix B and are not used to select or revise the main specification. Search grids, fixed configurations and early-stopping settings are reported in Appendix C.

## 3.7 Model Evaluation and Interpretation

### 3.7.1 Forecast Evaluation

The primary evaluation metrics are calculated from reconstructed price forecasts over the common sample of T=257 forecast origins. For model m,

\mathrm{RMSE}_m

\sqrt{
\frac{1}{T}
\sum_{t=1}^{T}
\left(P_{t+1}-\hat{P}_{m,t+1\mid t}\right)^2
}

Performance relative to the no-change benchmark M0 is summarised by the percentage improvement in RMSE

RMSE improvement vs M0m​(%)=100×(1−RMSEM0​RMSEm​​).

Here, P_{t+1} is the observed price and \hat{P}_{m,t+1\mid t} is the price forecast produced by model m at forecast origin t. A positive value indicates a lower RMSE than M0, zero indicates equal RMSE, and a negative value indicates worse performance.

### 3.7.2 Model Interpretation and Robustness Checks

Model interpretability concerns identifying where a model’s predictive ability comes from, including which data sources and features contribute more or less to its predictions. For Deep alternative-data models with positive RMSE improvement relative to M0, SHapley Additive exPlanations (SHAP) values are calculated to quantify each input’s contribution (Lundberg and Lee, 2017).
Absolute SHAP values are aggregated by data source and, where applicable, spatial site or node. Results are reported for the full sample, by year, and within predefined ±8-week event windows. For gated models, weekly modality weights are also reported to describe how the model allocates weight across financial, remote-sensing and shipping representations. SHAP and gate weights describe model attribution and internal allocation rather than causal importance.Robustness is assessed by rerunning the prespecified Deep models with random seeds and comparing their RMSE.

】

## 3.8 Ethical considerations

The study uses only secondary, aggregate data and does not involve human participants. It received approval through UCL’s low-risk ethics process. All datasets were used in accordance with their published licences and terms of use, including the Copernicus open licence for Sentinel-2, the open distribution terms for VIIRS night-time lights, and the research-use terms of IMF PortWatch and Global Fishing Watch. Remote-sensing and vessel-activity variables are analysed only at the aggregate site or chokepoint level; no attempt is made to identify individual vessels, operators or persons.

Analysis was conducted in Python, and the code required to reproduce the analysis is available on GitHub: [repository link]. Package versions, configuration settings and random-seed specifications are provided in Appendix C.

---


# Chapter 4 — Results

## 4.2 Flat-model results

Table 4.1 reports the out-of-sample performance of the Flat Ridge and XGBoost models across feature sets S1–S4, with M0 shown for comparison. All eight Flat models have higher RMSE than M0 and therefore record negative RMSE improvement.

**Table 4.1 — Flat out-of-sample performance** *(n = 257)*

| Set       | Variables                            | Model         | RMSE  | Improvement vs M0 (%) |
| --------- | ------------------------------------ | ------------- | ----- | --------------------- |
| Benchmark |                                      | M0            | 4.152 |                       |
| S1        | financial time series                | M1-Flat-Ridge | 4.256 | −2.5%                 |
|           |                                      | M1-Flat-XGB   | 4.368 | −5.2%                 |
| S2        | financial time series + RS           | M2-Flat-Ridge | 4.414 | −6.3%                 |
|           |                                      | M2-Flat-XGB   | 4.440 | −6.9%                 |
| S3        | financial time series + shipping     | M3-Flat-Ridge | 4.553 | −9.7%                 |
|           |                                      | M3-Flat-XGB   | 4.357 | −4.9%                 |
| S4        | financial time series + RS + shiping | M4-Flat-Ridge | 4.539 | −9.3%                 |
|           |                                      | M4-Flat-XGB   | 4.412 | −6.3%                 |

*Note:* Positive values indicate lower RMSE than M0.

For Ridge, S1 has the lowest RMSE and S3 the highest; adding remote sensing, shipping, or both raises RMSE relative to S1. For XGBoost, S3 records a slightly lower RMSE than S1 (4.357 versus 4.368), while S2 and S4 remain higher than S1. No Flat model records a positive RMSE improvement relative to M0. Overall, the Flat family performs worse than the no-change benchmark across all information sets.

The Flat results therefore provide no evidence of improvement relative to the no-change benchmark. Remote sensing consistently reduces performance for both Ridge and XGBoost. Shipping slightly improves XGBoost relative to S1 but worsens Ridge performance, and in neither case is the improvement sufficient to outperform M0.

## 4.3 Deep-model results

Table 4.2 reports Deep-model performance across S1–S4. Gated fusion is the prespecified main specification, while cross-attention is reported as a secondary comparison. No cross-attention result is reported for S1 because only the finance encoder is active.

**Table 4.2 — Deep out-of-sample performance** *(gated = main specification)*：XAttn

| Set       | Variables                            | Model         | RMSE  | Improvement vs M0 (%) |
| --------- | ------------------------------------ | ------------- | ----- | --------------------- |
| Benchmark |                                      | M0            | 4.152 |                       |
| S1        | financial time series                | M1-Deep       | 4.250 | −2.4%                 |
| S2        | financial time series + RS           | M2-Deep-Gated | 4.253 | −2.4%                 |
|           |                                      | M2-Deep-XAttn | 4.396 | −5.9%                 |
| S3        | financial time series + shipping     | M3-Deep-Gated | 4.146 | +0.15%                |
|           |                                      | M3-Deep-XAttn | 4.110 | +1.00%                |
| S4        | financial time series + RS + shiping | M4-Deep-Gated | 4.180 | −0.67%                |
|           |                                      | M4-Deep-XAttn | 4.144 | +0.19%                |

The gated S1 and S2 models record similar RMSE of 4.250 and 4.253, both higher than that of M0. Adding remote sensing therefore provides no descriptive improvement. Neither reported fusion approach reduces RMSE relative to the finance-only Deep model at S2. With shipping included, gated S3 records the lowest RMSE among the gated models at 4.146, improving on M0 by 0.15%. Gated S4 rises to 4.180, 0.67% worse than M0, indicating that adding remote sensing to S3 does not provide a further improvement.

On the reported seed-42 run, cross-attention has a higher RMSE than gated fusion at S2, at 4.396 compared with 4.253, but lower RMSEs at S3 and S4. Cross-attention records RMSEs of 4.110 and 4.144 at S3 and S4, corresponding to RMSE improvements of 1.00% and 0.19%. However, none of the comparisons between gated fusion and cross-attention passes the Holm correction. The positive improvements of cross-attention over M0 at S3 and S4 are also not statistically significant. These results are therefore reported only as descriptive secondary comparisons and do not provide evidence that cross-attention is superior.

Overall, the Deep family performs better than the Flat family, although most Deep specifications still do not outperform M0. For RQ1, shipping provides the clearest improvement. S3 achieves the best performance under both gated fusion and cross-attention, and both models outperform M0 in the reported run. Remote sensing provides little additional value. It does not improve the finance-only model at S2 and weakens the gated model when added to shipping at S4.

## 4.4 Flat versus Deep

Table 4.3 compares the main Deep model with both Flat models within each feature set. The feature-set category, forecast dates and evaluation sample are held constant. The main Deep pathway uses the finance-only Deep model at S1 and gated fusion at S2–S4.

**Table 4.3 — Matched Flat–Deep comparisons by feature set** *(n = 257)*

| Feature set | Flat model    | Flat RMSE | Deep model    | Deep RMSE | **Deep vs Flat (%)** |
| ----------- | ------------- | --------- | ------------- | --------- | -------------------- |
| S1          | Ridge         | 4.256     | M1–Deep       | 4.250     | +0.15%               |
| S1          | M1–Flat–XGB   | 4.368     | M1–Deep       | 4.250     | +2.71%               |
| S2          | M2–Flat–Ridge | 4.414     | M2–Deep–Gated | 4.253     | +3.64%               |
| S2          | M2–Flat–XGB   | 4.440     | M2–Deep–Gated | 4.253     | +4.22%               |
| S3          | M3–Flat–Ridge | 4.553     | M3–Deep–Gated | 4.146     | +8.95%               |
| S3          | M3–Flat–XGB   | 4.357     | M3–Deep–Gated | 4.146     | +4.85%               |
| S4          | M4–Flat–Ridge | 4.539     | M4–Deep–Gated | 4.180     | +7.90%               |
| S4          | M4–Flat–XGB   | 4.412     | M4–Deep–Gated | 4.180     | +5.26%               |

*Note. Positive values indicate a lower Deep RMSE than the matched Flat model.*

Figure 4.2

**Figure 4.2 — Paired slopes from Flat XGBoost to Deep gated fusion at each information set, with S3 highlighted.**

Across all four feature sets, the main Deep model records lower RMSE than both Ridge and XGBoost. The reduction ranges from 0.15% against Ridge at S1 to 8.95% against Ridge at S3. At S1 and S2, the main Deep models improve on both Flat learners but remain worse than M0. S3 is the only feature set which has lower RMSE than M0. Although the main Deep S4 model improves substantially over both Flat models, it remains worse than M0 and does not improve on the main Deep S3 model.

Overall, this comparisons show that the main Deep pathway records lower RMSE than both Flat learners across all four feature sets. For RQ2, this provides consistent evidence that modality-aware representation-level modelling performs better than flat feature fusion when using matched information sets. However, because the Deep and Flat pathways also differ in model architecture and data representation, the improvement cannot be attributed to representation-level fusion alone. Nevertheless, the comparison suggests that how heterogeneous data are organised and represented may affect the extent to which different information is retained and used by the model, and that preserving the distinct structure and characteristics of different data types may be valuable for future research.

## 4.5 Interpretability

Following the eligibility rule in Section 3.7.2, interpretation is reported for gated Deep model on S3. Table 4.4 combines period-specific forecast performance, modality-gate weights and absolute SHAP attribution for the 257 forecast origins.

**Table 4.4 — Period-specific performance and attribution for gated Deep S3**

| Period         | n   | RMSE  | Improvement vs M0 (%) | Gate finance | Gate shipping | SHAP finance | SHAP shipping |
| -------------- | --- | ----- | --------------------- | ------------ | ------------- | ------------ | ------------- |
| Full sample    | 257 | 4.146 | +0.15%                | 0.558        | 0.442         | 96.8%        | 3.2%          |
| 2021           | 50  | 3.014 | −1.98%                | 0.521        | 0.479         | 95.8%        | 4.2%          |
| 2022           | 52  | 6.613 | +0.77%                | 0.519        | 0.481         | 96.4%        | 3.6%          |
| 2023           | 52  | 3.790 | −0.37%                | 0.481        | 0.519         | 94.8%        | 5.2%          |
| 2024           | 52  | 3.083 | −0.79%                | 0.520        | 0.480         | 95.0%        | 5.0%          |
| 2025           | 51  | 2.960 | +0.99%                | 0.749        | 0.251         | 99.2%        | 0.8%          |
| Russia–Ukraine | 16  | 8.822 | +1.15%                | 0.536        | 0.464         | 96.5%        | 3.5%          |
| EU oil ban     | 16  | 5.059 | −0.21%                | 0.524        | 0.476         | 96.0%        | 4.0%          |
| OPEC+          | 16  | 4.358 | +0.76%                | 0.485        | 0.515         | 95.7%        | 4.3%          |
| Red Sea        | 16  | 3.265 | +2.55%                | 0.489        | 0.511         | 94.1%        | 5.9%          |

*Note. RMSE is evaluated in price levels, while SHAP attributes predicted log returns. Event windows are ±8 weeks.*

Financial inputs dominate absolute SHAP throughout the sample, accounting for 96.8% of full-sample attribution compared with 3.2% for shipping. Shipping attribution is relatively higher in 2023–2024 and during the Red Sea window, at between 5.0% and 5.9%, but falls to 0.8% in 2025. Its contribution is therefore small and episodic rather than persistently elevated.

The main-run gate allocates average weights of 55.8% to finance and 44.2% to shipping, a substantially more balanced division than the SHAP attribution. This contrast reflects the difference between internal representation weighting and output attribution; SHAP is therefore used as the primary basis for interpreting RQ3.

At the input-group level, EIA variables provide the largest full-sample contribution at 43.6%, followed by financial and macroeconomic variables at 31.4%. All twenty highest-ranked individual features are financial inputs, led by crude production, Cushing stocks and the federal funds rate. No shipping subgroup contributes more than 2.0% in any reported period, although PortWatch and SAR become modestly more prominent during the Red Sea window.

Within the shipping representation, the highest full-sample node shares belong to Jurong, Hormuz, Suez, the Cape route and Bab el-Mandeb. Jurong and Hormuz lead the rankings from 2021 to 2023, while Suez, Bab el-Mandeb and the Cape route occupy the first three positions in 2024. During the Red Sea window, attribution is distributed across several locations, with no individual node accounting for more than 12% of shipping attribution.

For RQ3, the model relies predominantly on financial information across all market conditions. Shipping provides a much smaller and more episodic contribution, becoming relatively more important in some periods of market and trade disruption, particularly in 2023–2024 and during the Red Sea event window. Its geographic focus also shifts over time across major ports and chokepoints rather than remaining concentrated in one location. Overall, the results suggest that financial data provide the model’s core predictive information, while shipping data act as a supplementary source whose importance increases under particular market conditions.

## 4.6 Robustness

**Table 4.5 — Random-seed robustness of all Deep specifications** *(improvement vs M0, %)*

| Set | Model          | Main-run improvement | Across-run mean ± SD | Positive runs |
| --- | -------------- | -------------------- | -------------------- | ------------- |
| S1  | M1-Deep        | −2.36%               | −1.00% ± 1.33        | 1/3           |
| S2  | M2-Deep-Gated  | −2.43%               | −3.15% ± 1.67        | 0/3           |
|     | M2-Deep-Concat | −2.01%               | −1.79% ± 0.77        | 0/3           |
|     | M2-Deep-XAttn  | −5.87%               | −3.57% ± 2.77        | 0/3           |
| S3  | M3-Deep-Gated  | +0.15%               | −0.51% ± 0.80        | 1/3           |
|     | M3-Deep-Concat | −0.22%               | −0.27% ± 0.35        | 1/3           |
|     | M3-Deep-XAttn  | +1.00%               | −3.01% ± 4.07        | 1/3           |
| S4  | M4-Deep-Gated  | −0.68%               | −0.91% ± 0.26        | 0/3           |
|     | M4-Deep-Concat | −8.30%               | −3.79% ± 3.95        | 0/3           |
|     | M4-Deep-XAttn  | +0.19%               | −1.90% ± 2.75        | 1/3           |

*Note. The seed-42 column is the main reported run in Tables 4.2–4.4. S1 has no fusion operator.*

Table 4.5 reports the results of rerunning all Deep specifications with multiple random seeds. None of the models achieves a positive mean RMSE improvement relative to M0 across runs, and only five of the thirty individual runs are positive. S3 concatenation has the best mean result, but it is still negative at −0.27%, while the main gated S3 model records −0.51%. Gated S3 also outperforms S1 in only one of the three matched runs, and all S2 runs remain worse than M0. The positive improvements observed in the main run are therefore sensitive to random initialisation. Overall, although the Deep models do not consistently outperform M0, some specifications, particularly S3, show predictive potential and merit further investigation. The better-performing Deep specifications remain broadly close to M0 rather than demonstrating a consistent forecasting advantage.

---


# Chapter 5 — Discussion

## 5.1 RQ1 — Do alternative data help?

RQ1 asked whether remote sensing and shipping add out-of-sample value beyond financial time series and the no-change benchmark. The answer depends on the modelling pathway. Within the Flat family, no model outperforms no-change benchmark, consistent with the short-horizon oil-forecasting literature that treats the no-change forecast as a demanding reference (Alquist, Kilian and Vigfusson, 2013) Relative to the finance-only S1 specification, remote sensing increases forecast error for both Ridge and XGBoost, while shipping increases error for Ridge but slightly reduces it for XGBoost. The latter result shows that shipping is not uniformly detrimental within the Flat pathway, but the improvement remains insufficient to outperform M0. Overall, simply adding alternative-data features to Flat models does not produce additional predictive value against the no-change benchmark.

Under the Deep pathway, adding remote sensing to finance in S2 does not improve forecast accuracy over either S1 or M0. By contrast, adding shipping to finance in S3 produces a small positive RMSE improvement relative to M0, making S3 the best-performing specification in the main gated pathway. Extending S3 with remote sensing in S4 does not further reduce forecast error. The secondary cross-attention results show the same ordering across the multimodal specifications, with S3 performing best, followed by S4 and S2. Shipping is therefore the more informative alternative modality in this weekly Brent design, while remote sensing contributes little to one-week-ahead predictive accuracy. The improvement is notable because the other main modelling specifications fail to beat M0, although its magnitude remains modest. It should be interpreted as evidence of predictive rather than operational value.

Distributed observations of port and chokepoint activity appear more useful for short-horizon forecasting than the selected site-level remote-sensing proxies. The lack of remote-sensing gains may reflect a temporal and spatial mismatch between monthly, localised AOI signals and a weekly global benchmark price.

These findings refine the AIS and satellite literature reviewed in Chapter 2. Existing studies often demonstrate that ships and satellites contain information about trade or physical activity (Adland, Jia and Strandenes, 2017; Yan et al., 2020; Hao and Wang, 2023), but less often ask whether those signals improve one-week-ahead Brent forecasts relative to both a financial baseline and M0. The present results distinguish informational content from predictive value. The ability to measure trade or industrial activity does not necessarily produce a forecast improvement against a demanding weekly benchmark.

## 5.2 RQ2 — Does representation-level fusion beat flat fusion?

RQ2 asked whether representation-level Deep modelling outperforms flat feature fusion when the information sets and evaluation protocol are held fixed. Across the matched multimodal sets S2–S4, the main Deep pathway records lower RMSE than both Flat learners. The differences are largest in the shipping-inclusive S3 and S4 sets, although only S3 in the main Deep pathway also outperforms M0. The results therefore favour the Deep pathway over early feature concatenation, while showing that an advantage over Flat models does not necessarily translate into an improvement over the no-change benchmark.

Early-fusion approaches combine heterogeneous predictors within a single feature space, whereas multimodal models retain separate representations before fusion (Arevalo et al., 2017; Gohari et al., 2024). The larger Deep advantage in the shipping-inclusive S3 and S4 sets extends this comparison to weekly Brent forecasting. From a spatial perspective, the result suggests that representation-level modelling may be particularly useful for observations distributed across networks of ports and chokepoints. Treating the same observations as independent tabular features does not explicitly represent these spatial relationships. This pattern is consistent with spatial relationships among shipping nodes providing useful structure.

However, the matched comparisons evaluate complete modelling pathways, including their encoders and fusion strategies. They therefore support the overall Deep approach but do not identify the preservation of network structure or any individual fusion operator as the cause of its lower RMSE.

## 5.3 RQ3 — What does the model rely on when value exists?

RQ3 asked how the model uses information when an alternative-data specification improves on the benchmark. The gate and SHAP results show that internal representation weights do not translate directly into contributions to model output. The forecast continues to rely primarily on financial and EIA information, with shipping providing a complementary signal. The coexistence of a relatively small shipping attribution and an improvement over M0 indicates that a modality can add predictive value without dominating the forecast.

This complementary role also varies with the type of disruption. Shipping becomes relatively more prominent during the Red Sea period, whereas financial inputs remain more prominent during the Russia–Ukraine window. This contrast is consistent with transport-specific disruptions increasing the relevance of maritime activity, while broader geopolitical shocks may affect Brent through a wider combination of supply expectations, inventories and financial-market channels. The spatial pattern supports the same interpretation. The model’s focus shifts from Jurong and Hormuz in earlier years towards Suez, Bab el-Mandeb and the Cape route in 2024. This pattern broadly coincides with disruption and rerouting around the Red Sea. Yet no single location dominates within that window. The model therefore appears to respond to changes in the spatial configuration of shipping activity rather than relying persistently on one chokepoint.

Attention and gate weights describe operations within a fitted model rather than causal relationships (Jain and Wallace, 2019), while SHAP attributes predicted log returns rather than explaining the causes of price movements. The temporal and spatial diagnostics can therefore identify periods and transport corridors for further investigation, but should not be treated as stand-alone policy alerts.

## 5.4 Implications

Model choice should be aligned with the structure of the input data. For relational data such as shipping networks, preserving within-modality structure before fusion may be more appropriate than direct concatenation. Whatever framework is used, each new modality should be evaluated under the same rolling out-of-sample protocol against finance-only S1 and no-change M0. The former shows whether the modality adds information beyond conventional predictors, while the latter shows whether the complete system improves on a simple forecasting rule.

These models offer diagnostic rather than causal insight into oil prices. They show which information is useful for next-week prediction and can direct attention to periods and parts of the supply network that warrant further investigation, but they do not identify the mechanisms generating price movements. Their role is therefore to organise predictive evidence for market analysis rather than to substitute for structural or causal explanations of oil-price formation.

Spatial data should be assessed according to how well their scale, frequency and structure match the forecasting target, rather than by geographic specificity alone. Remote sensing describes conditions at selected facilities, so its monthly signals may be more suitable for monitoring facility activity or regional production. Shipping data capture flows across connected ports, chokepoints and corridors and therefore better reflect disruption and adjustment across the global oil supply network. They can complement weekly Brent monitoring, although their contribution remains too limited to replace financial and EIA information. For energy-security monitoring, trade planning and inflation-sensitive fiscal management, better observation of physical stress does not necessarily improve one-week-ahead Brent forecasts.

## 5.5 Limitations

With only 257 rolling out-of-sample forecasts, the small improvement of Deep S3 over M0, concentrated in some subperiods and event windows, provides evidence of limited and conditional rather than stable predictive value. The findings are specific to one-week-ahead Brent forecasting and should not be generalised to other horizons, regional oil prices or targets such as volatility.

Despite publication lags and as-of alignment, the study uses revised historical series rather than real-time vintages and therefore does not fully reproduce the information available at each forecast origin. Monthly GFW and remote-sensing inputs are repeated between releases, limiting their ability to capture short-lived changes. Because alternative-data variables are indirect proxies and preprocessing choices such as missing-value treatment and frozen Earth-observation embeddings introduce additional assumptions, model performance cannot be attributed solely to the underlying signals.

The spatial findings are conditional on a purposively selected network of eleven sites and six chokepoints, concentrated on Gulf export and Asian import and refining hubs while excluding Russian, West African and Latin American loading regions. Selecting partly for satellite observability introduced further geographic bias, while graph symmetrisation removed the direction and type of connections. Node attributions therefore describe dependence within this constructed network rather than a comprehensive ranking of global oil infrastructure.

The Flat–Deep comparisons evaluate complete modelling pathways rather than individual components because the models differ in class, capacity, encoders and fusion procedures. Their remote-sensing inputs are not fully identical, and the Deep shipping pathway also introduces an explicit graph structure. Performance differences therefore cannot be attributed specifically to the shipping GAT or gated fusion.The gated model was selected as the main Deep specification because it provides the modality weights required for RQ3, not because gated fusion was shown to outperform other fusion mechanisms. Finally, RMSE alone does not establish operational value because trading costs and returns, hedging outcomes and policy interventions were not evaluated.

## 5.6 Future research and closing statement

Future research should test whether the shipping contribution persists over longer periods and across a broader oil-transport network. A longer evaluation using archived releases and expanded coverage of Russian Baltic and Black Sea ports, West African loading regions and Latin American exporters would extend the evidence beyond the present sample and selected corridors. Where AIS coverage is incomplete, SAR-based vessel detection could extend observation to tracking gaps and dark-fleet activity.

Transferability should be tested across forecasting targets and markets. The matched Flat–Deep design could be applied to longer horizons, regional oil prices and related energy commodities, including WTI, LNG and natural gas. Applying the same design to grain or metals would test whether network representation remains useful beyond energy markets. Remote sensing could also be evaluated against facility activity, regional production or longer-horizon prices that better match its spatial and temporal scale. Where data allow, a stricter like-for-like remote-sensing comparison between Flat and Deep would isolate architecture more cleanly.

Future research should test practical value beyond RMSE using transaction-cost-adjusted trading, simple hedging rules and forecast performance during major disruptions. For public-sector use, probabilistic forecasts and scenario ranges could be compared with existing financial and EIA monitoring to assess whether shipping information improves the timing or prioritisation of investigation in energy-security monitoring, trade planning and inflation-sensitive budgeting. Model attribution should remain a diagnostic aid rather than an automatic policy alert.

Future research should test practical value beyond RMSE using transaction-cost-adjusted trading, simple hedging rules and forecast performance during major disruptions. For public-sector use, probabilistic forecasts and scenario ranges could be compared with existing financial and EIA monitoring to assess whether shipping information improves the timing or prioritisation of investigation in energy-security monitoring, trade planning and inflation-sensitive budgeting. Evaluating specific interventions, such as strategic reserve releases, sanctions or fiscal responses, would require combining forecasting with causal identification or decision-analysis frameworks to estimate counterfactual effects on prices and supply together with policy costs. Model attribution should remain a diagnostic aid rather than an automatic policy alert.

---


# Chapter 6 — Conclusion

Short-horizon oil-price surprises affect hedging, budgeting and market-risk decisions, yet weekly Brent forecasts remain difficult to improve on a simple no-change rule. Using weekly data from 2019 to 2025, this dissertation examined whether satellite remote sensing and maritime shipping add predictive information beyond conventional financial and physical-market indicators, and whether preserving the structure of each data type improves on flat feature fusion. Under a common rolling out-of-sample design, each data addition was compared with both a finance-only model and the no-change benchmark, separating incremental contribution from overall forecast performance.

The results show that additional data and model complexity do not automatically produce better forecasts. Remote sensing added no forecast value, whether introduced alone or on top of shipping. Shipping was more useful, but no Flat model outperformed the no-change benchmark. Among the main specifications, only the modality-aware finance-plus-shipping model achieved a small improvement. Across matched information sets, modality-aware models generally produced lower errors than flat fusion. This shows that preserving data structure can improve relative performance without guaranteeing forecasts that outperform the benchmark. Model diagnostics indicated that predictions remained led by financial and EIA information, while shipping made a smaller contribution that varied across disruption periods and locations in the transport network. These patterns show how the model used the available information but do not identify the causes of oil-price movements.

The main contribution is a clearer standard for deciding when alternative spatial data constitute forecast evidence. Their value depends on whether their scale, frequency and structure match the forecasting target, and whether they improve on both conventional information and a simple forecasting rule. Remote sensing of local facility conditions may be better suited to facility or regional monitoring, while shipping observations can complement monitoring of network-wide disruption and rerouting. For energy-security monitoring, trade planning and inflation-sensitive budgeting, financial and EIA indicators should therefore remain the core, with spatial data providing additional context rather than stand-alone policy signals. Better observation of physical stress is not the same as better short-horizon price prediction.

Future research should test the transferability of these findings using longer real-time histories, broader transport networks and stricter like-for-like model comparisons, alongside economic and policy evaluation. Rather than identifying a universally superior model or data source, this study shows that the predictive value of alternative data depends on alignment with the target, how the data are represented and whether improvements hold against credible benchmarks.

---


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
| P007 | Jamnagar        | India        | refinery | refining           | Hormuz        | 69.860, 22.345  |
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
averaging ~65.8 edges/week. Sources: `build_m3_graph17.py`,
`m3_data_dictionary.md` §12, `shipping_encoder.py`.

### A.4.1 Dynamic O-D voyage edges (AOI→AOI)

Directed AOI→AOI edges from GFW voyage counts; edge weight = `n_voyages` for
that week's directed lane (`from ≠ to`; self-loops removed to a node feature).
Different every week; 96 lanes, 106 992 voyages total (top lanes e.g.
Ningbo↔Singapore, Fujairah↔Singapore, Singapore↔Rotterdam). Directionality
verified (`P006→P004 ≠ P004→P006`). Lag +2 w.

### A.4.2 Static AOI↔chokepoint edges

Fixed undirected links by geographic association (13 undirected edges), present
every week (`aoi_oil_infrastructure_sites.md` §4). Every AOI carries at least one
corridor link: P007 (Jamnagar) is a demand-side refinery rather than a Gulf
export terminal, but its crude slate is dominated by Persian Gulf loadings, so it
is attached to Hormuz on the import side.


| Chokepoint | Linked AOIs                  |
| ---------- | ---------------------------- |
| `hormuz`   | P002, P003, P007, P008, P010 |
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
> expanding rolling-origin, 257 common scored weeks). Every p-value below is a
> one-sided DM-HLN test on reconstructed-price squared error, as defined in
> Section 3.7.2; Clark–West appears for Ridge only. Ablation arms in B.3 are
> exploratory and sit outside the three frozen comparison families, so they carry
> raw p-values only. They **qualify** the main findings; they do not replace the
> main-analysis specs (§ Chapter 4). Sources are given per table so every number
> is traceable.

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
| M3_Flat Ridge / XGB        | −7.11 / −6.17 | −7.91 / −6.94 | −5.80 / −4.91 |
| M4_Flat Ridge / XGB        | −9.26 / −8.53 | −9.83 / −10.52 | −8.32 / −5.21 |
| M1_Deep                    | −2.36         | −1.33         | −4.02         |
| M_ship_GNN (shipping only) | −0.22         | −0.24         | −0.17         |
| M_rs_deep (RS only)        | −2.30         | −3.07         | −1.04         |
| **M3_Deep_gated (main)**   | **+0.16**     | **+0.33**     | **−0.13**     |
| M2_Deep_gated              | −2.43         | −3.06         | −1.41         |
| M4_Deep_gated              | −0.67         | −1.36         | +0.47         |
| M4_Deep_Concat             | −8.30         | −12.77        | −0.58         |


**Reading**: no flat model beats M0 in either sub-period. Among deep models,
M3_Deep_gated has both the largest full-sample skill (+0.16) and the strongest
early-period skill (+0.33), but is marginally negative late (−0.13);
M4_Deep_gated shows the opposite profile (−1.36 early, +0.47 late), and
**No deep configuration in this table
is positive in both sub-periods**, so the small full-sample gain of the main model
is not evenly distributed over time. The sub-period decomposition is run only on the
gated and concat specifications, so cross-attention is outside the scope of this
table; its instability is documented across seeds in B.4 instead. Gated finance+shipping is retained as the
main model on full-sample skill and on the nested shipping increment (Chapter 4);
this split is reported as a limitation on how stable that gain is, not as
supporting evidence.

---



## B.2 Deep fusion matrix (3 modality combos × 3 fusions)

seed 42, lookback 4, 257 weeks. Source: `run_deep_fusion_matrix.py` →
`05_outputs/baselines/Deep/_cross/deep_fusion_matrix.csv`. Skill vs M0 in %.


| Combo                  | Concat | Gated     | Cross-Attn | DM p vs M0, best variant (raw / Holm) |
| ---------------------- | ------ | --------- | ---------- | ------------------------------------- |
| **M3_Deep** (fin+ship) | −0.22  | **+0.15** | **+1.00**  | xattn 0.257 / 1.000                   |
| M2_Deep (fin+rs)       | −2.01  | −2.43     | −5.87      | —                                     |
| M4_Deep (fin+rs+ship)  | −8.30  | −0.68     | +0.19      | xattn 0.427 / 1.000                   |


**Reading**: on this seed, M0 is cleared only where shipping is present, and only
under adaptive fusion: M3 clears it under gated and cross-attention, M4 only under
cross-attention, and M2 (fin+rs) never. Plain concatenation clears M0 in no combo.
That pattern would invite the conclusion that the gain comes from weighting the
modalities rather than stacking them — but the conclusion does not survive
reseeding: averaged over seeds 42, 1 and 2, the M3 ordering inverts to concat
(−0.27%) > gated (−0.51%) > cross-attention (−3.01%), so the apparent premium on
adaptive weighting is a seed-42 artefact (B.4).
None of the three positive skills is statistically distinguishable from M0 either:
the largest, cross-attention at M3, has a raw one-sided DM p of 0.257, and every
member of the 18-test benchmark family has a Holm-adjusted p of 1.000. The positive
entries in this table are therefore descriptive orderings on one seed, not evidence
of superiority over the no-change forecast, and the fusion column should not be read
as a ranking of fusion mechanisms; B.4 gives the seed-averaged comparison.

---



## B.3 Flat robustness



### B.3.1 M2 leave-one-AOI-out (LOAO)

Source: `05_outputs/baselines/Flat/M2_Flat/baseline_loao_anom.csv`, which runs the
eleven leave-one-AOI-out arms under the main M2 configuration (base RMSE 4.4136
Ridge, 4.4402 XGBoost). Removing any single AOI leaves the result essentially
unchanged: the largest shift is 0.091 for XGBoost (Houston) and 0.071 for Ridge
(Kharg), each about 2% of RMSE. Dropping a site more often helps than hurts—seven
of eleven sites for Ridge and eight of eleven for XGBoost give a lower RMSE—so no
individual location carries the remote-sensing signal, and the weak M2 result
reflects diffuse noise rather than one dominant site.

### B.3.2 M3 leave-one-channel-out (LOCHO)

seed 42, 257 weeks. Source:
`05_outputs/baselines/Flat/M3_Flat/robustness_m3_summary.csv`. XGB RMSE, skill vs
M0 (%), and the one-sided DM-HLN p against M1 XGB (RMSE 4.3684). Clark–West is
not reported for XGBoost (§3.7.2).


| Arm                   | XGB RMSE | XGB skill vs M0 | XGB DM p vs M1 |
| --------------------- | -------- | --------------- | -------------- |
| full (113 cols, main) | 4.4080   | −6.17           | 0.633          |
| core (38)             | 4.4207   | −6.48           | 0.727          |
| portwatch-only        | 4.3387   | −4.50           | 0.391          |
| gfw-only              | 4.3884   | −5.70           | 0.622          |
| gfw-presence          | 4.3590   | −4.99           | 0.450          |
| gfw-aggregate         | 4.3605   | −5.03           | 0.426          |
| tanker-only           | 4.3317   | −4.33           | 0.384          |


**Reading**: four arms—tanker-only, PortWatch-only, GFW-presence and
GFW-aggregate—do sit slightly below the finance-only M1 XGBoost baseline of
4.3684 in RMSE, so the descriptive ordering is not uniformly against shipping.
The evidence is nevertheless weak: the smallest one-sided DM p across the seven
arms is 0.384, no arm beats M0, and the main 113-column specification is among
the least favourable (p = 0.633). For Ridge, where Clark–West is admissible, only
the GFW-aggregate arm falls below 5% (CW p = 0.032), and that is also the only
Ridge arm whose RMSE (4.2438) sits below M1 Ridge (4.2563), so the supplementary
test and the descriptive ordering agree there. The M3 shipping signal is
therefore not established by a channel-level ablation.

### B.3.3 M2 water-masked RS variant

Source: `baseline_metrics_anom_watermask.csv`. Masking water pixels lowers XGB RMSE slightly, from 4.4402 to 4.4136, and the one-sided DM p against M1 moves from 0.822 to 0.687; M2 still does not beat M0 (skill −6.3%). De-noising the remote-sensing inputs therefore changes neither the ranking nor the inference: RS value is limited, which motivates modality-aware fusion (RQ2).

### B.3.4 M4 leave-one-modality-out (LOMO)

Source: `Flat/M4_Flat/robustness_m4_summary.csv`. Dropping RS from M4 (i.e. M1+M3) lowers XGB RMSE from 4.5061 to 4.4080, and dropping shipping instead gives 4.4402; the finance-only arm is still the most accurate at 4.3684. The one-sided DM p against M1 is 0.847 for full M4, 0.633 without RS and 0.822 without shipping, so no arm is distinguishable from the finance-only baseline. Each added modality raises RMSE in flat concatenation, and no valid nested increment is detectable.

---



## B.4 Deep multi-seed & sweeps

Individual runs come from `run_deep_multiseed.py` (each invocation writes its own
`deep_seed_*.csv`) and from `run_deep_fusion_matrix.py` / `run_deep_sweep.py`.
`scripts/tools/pool_deep_seeds.py` then pools those files, de-duplicates on
(config, seed) keeping the higher epoch budget, and re-aggregates the mean and SD
below → `deep_seed_pooled.csv` (one row per config × seed) and
`deep_seed_summary.csv` (the table below). The figures are therefore recomputed
from the pooled runs rather than transcribed, and adding seeds later means adding a
file, not editing a table. All rows are now on the matrix protocol: seeds 42, 1, 2;
lookback 4; d 32; epochs 80. The 11 sweep rows at epochs 60 are superseded and drop
out during de-duplication.


| Config              | seed 42   | seed 1 | seed 2 | mean ± sd            | positive |
| ------------------- | --------: | -----: | -----: | -------------------- | -------: |
| M3 concat           | −0.22     | −0.64  | +0.07  | **−0.27% ± 0.35**    | 1/3      |
| **M3 gated** (main) | +0.15     | −1.39  | −0.29  | −0.51% ± 0.80        | 1/3      |
| M4 gated            | −0.68     | −0.86  | −1.19  | −0.91% ± 0.26        | 0/3      |
| M4 xattn            | +0.19     | −0.87  | −5.01  | −1.90% ± 2.75        | 1/3      |
| M3 xattn            | **+1.00** | −2.87  | −7.14  | −3.01% ± **4.07**    | 1/3      |


The seed-42 column is the matrix-protocol re-run, so M3 gated reads +0.15% here
against the +0.16% quoted from the baseline run in Table 4.2; the 0.01-point gap is
run-to-run reduction-order noise on the same protocol, not a different setting.
The M2 combos and M4 concat are reseeded nowhere, so they appear in
`deep_seed_summary.csv` with `n_seeds = 1` and are excluded from this table: a
single run is not a multi-seed mean.

**Every mean is below M0.** The positive headline figures in Chapter 4 (M3 gated
+0.16%, M3 cross-attention +1.00%) are seed-42 outcomes, not expected skill:
averaged over seeds, no Deep configuration beats the no-change benchmark, and each
M3 fusion is positive in exactly one of three seeds. This is the sharpest single
limitation on the Deep results and is carried into Chapter 5.

Within the M3 block the seed-42 ranking is **exactly reversed** by reseeding.
On seed 42 the order is xattn (+1.00) > gated (+0.15) > concat (−0.22); across
seeds it is concat (−0.27) > gated (−0.51) > xattn (−3.01), and dispersion widens
in the same order (0.35 < 0.80 < 4.07). The more adaptive the fusion, the higher
its single-seed ceiling and the worse its seed-averaged skill — which is what
makes single-seed selection actively misleading here, since it systematically
picks the highest-variance operator. Cross-attention is excluded on that basis: its
mean is an order of magnitude worse and its spread an order of magnitude wider than
either alternative. Concat and gated, by contrast, are **not** separable — the
0.24-point gap between their means is smaller than the cross-seed sd of either, so
concat's nominally better mean is not evidence that it forecasts better. Gated is
retained as the main specification because it is the only one of the two that
exposes modality gates, which are the object of the RQ3 analysis (Section 4.6);
that is a design requirement, not a claim that it is the more accurate operator.

Single-seed (42) hyper-sweep, finship gated, epochs 80 (same protocol as the
matrix): lookback 4/8/12 × d 32/64 → lb 8 d 32 best (+0.25%) > lb 4 d 32 (+0.15%)
> lb 12 d 32 (−0.06%); **d 64 always worse** (−0.75 / −1.06 / −1.08; short weekly
sample). Halving encoder depth (1 GAT + 1 TCN layer instead of 2 + 2) also
degrades skill to −1.26%, so the main setting is not simply over-parameterised.
The default cell (lb 4 d 32) now matches the matrix seed-42 figure to the last
reported digit. Main model stays locked at lookback 4, d 32 for flat protocol
parity.

---



## B.5 Other robustness (documented, not tabulated here)

- **Lookback sweep** L1/8/12 per layer — `sweep_m`* + `deep_sweep`.
- **C2 dimensionality reduction** (PCA / ElasticNet) for M2 —
`c2_summary.csv`; increment not pure over-fitting.
- **feature-mode = returns** stationarised variant — numerical robustness.
- **min_train = 78 longer window** — Appendix-level; main protocol
keeps min_train = 104.
- **Missing-data treatment** is locked in Section 3.4.3 (RS anomalies to zero;
shipping counts to the training-fold median) and is **not** a robustness axis.

---

## B.6 Frozen comparison families

The three families defined in Section 3.7.2 are listed here in full. Membership is
fixed by the research questions before any p-value is inspected, and Holm's
adjustment is applied within each family separately. Every row is a DM-HLN test on
squared errors of the reconstructed price over the same 257 forecast origins.
Source: `05_outputs/tests/test_table_main.csv`, generated by
`04_code/scripts/tools/build_test_tables.py`. Skill is the percentage RMSE
reduction of the candidate relative to the reference, so a positive value means
the candidate is the more accurate forecast.

### B.6.1 Benchmark family (18 comparisons)

| # | Reference | Candidate | Direction | RMSE ref → cand | Skill % | Raw p | Holm p |
| ---: | --- | --- | --- | --- | ---: | ---: | ---: |
| 1 | M0 | Ridge S1 | one-sided | 4.1518 → 4.2563 | −2.52 | 0.9429 | 1.0000 |
| 2 | M0 | Ridge S2 | one-sided | 4.1518 → 4.4136 | −6.31 | 0.9801 | 1.0000 |
| 3 | M0 | Ridge S3 | one-sided | 4.1518 → 4.4471 | −7.11 | 0.9471 | 1.0000 |
| 4 | M0 | Ridge S4 | one-sided | 4.1518 → 4.5361 | −9.26 | 0.9787 | 1.0000 |
| 5 | M0 | XGB S1 | one-sided | 4.1518 → 4.3684 | −5.22 | 0.9825 | 1.0000 |
| 6 | M0 | XGB S2 | one-sided | 4.1518 → 4.4402 | −6.95 | 0.9965 | 1.0000 |
| 7 | M0 | XGB S3 | one-sided | 4.1518 → 4.4080 | −6.17 | 0.9904 | 1.0000 |
| 8 | M0 | XGB S4 | one-sided | 4.1518 → 4.5061 | −8.53 | 0.9942 | 1.0000 |
| 9 | M0 | Deep S1 | one-sided | 4.1518 → 4.2499 | −2.36 | 0.9773 | 1.0000 |
| 10 | M0 | Deep gated S2 | one-sided | 4.1518 → 4.2528 | −2.43 | 0.9483 | 1.0000 |
| 11 | M0 | Deep gated S3 | one-sided | 4.1518 → 4.1456 | **+0.15** | 0.4266 | 1.0000 |
| 12 | M0 | Deep gated S4 | one-sided | 4.1518 → 4.1801 | −0.68 | 0.9492 | 1.0000 |
| 13 | M0 | Deep concat S2 | one-sided | 4.1518 → 4.2352 | −2.01 | 0.9983 | 1.0000 |
| 14 | M0 | Deep concat S3 | one-sided | 4.1518 → 4.1611 | −0.22 | 0.6461 | 1.0000 |
| 15 | M0 | Deep concat S4 | one-sided | 4.1518 → 4.4964 | −8.30 | 0.9228 | 1.0000 |
| 16 | M0 | Deep xattn S2 | one-sided | 4.1518 → 4.3956 | −5.87 | 0.9990 | 1.0000 |
| 17 | M0 | Deep xattn S3 | one-sided | 4.1518 → 4.1102 | **+1.00** | 0.2572 | 1.0000 |
| 18 | M0 | Deep xattn S4 | one-sided | 4.1518 → 4.1437 | **+0.19** | 0.4273 | 1.0000 |

Only three of the eighteen specifications reduce RMSE relative to the no-change
forecast, all of them Deep and all of them containing shipping. The smallest raw
p in the family is 0.257, so none approaches significance even before adjustment,
and every Holm-adjusted p equals 1.000.

### B.6.2 RQ1 family (15 comparisons)

| # | Reference | Candidate | Direction | RMSE ref → cand | Skill % | Raw p | Holm p |
| ---: | --- | --- | --- | --- | ---: | ---: | ---: |
| 1 | Ridge S1 | Ridge S2 | one-sided | 4.2563 → 4.4136 | −3.69 | 0.9561 | 1.0000 |
| 2 | Ridge S1 | Ridge S3 | one-sided | 4.2563 → 4.4471 | −4.48 | 0.8474 | 1.0000 |
| 3 | Ridge S1 | Ridge S4 | one-sided | 4.2563 → 4.5361 | −6.57 | 0.9383 | 1.0000 |
| 4 | Ridge S2 | Ridge S4 | one-sided | 4.4136 → 4.5361 | −2.77 | 0.8302 | 1.0000 |
| 5 | Ridge S3 | Ridge S4 | one-sided | 4.4471 → 4.5361 | −2.00 | 0.9212 | 1.0000 |
| 6 | XGB S1 | XGB S2 | one-sided | 4.3684 → 4.4402 | −1.64 | 0.8215 | 1.0000 |
| 7 | XGB S1 | XGB S3 | one-sided | 4.3684 → 4.4080 | −0.91 | 0.6326 | 1.0000 |
| 8 | XGB S1 | XGB S4 | one-sided | 4.3684 → 4.5061 | −3.15 | 0.8465 | 1.0000 |
| 9 | XGB S2 | XGB S4 | one-sided | 4.4402 → 4.5061 | −1.48 | 0.7360 | 1.0000 |
| 10 | XGB S3 | XGB S4 | one-sided | 4.4080 → 4.5061 | −2.22 | 0.7910 | 1.0000 |
| 11 | Deep S1 | Deep gated S2 | one-sided | 4.2499 → 4.2528 | −0.07 | 0.5213 | 1.0000 |
| 12 | Deep S1 | Deep gated S3 | one-sided | 4.2499 → 4.1456 | **+2.45** | **0.0410** | 0.6150 |
| 13 | Deep S1 | Deep gated S4 | one-sided | 4.2499 → 4.1801 | **+1.64** | 0.0682 | 0.9553 |
| 14 | Deep gated S2 | Deep gated S4 | one-sided | 4.2528 → 4.1801 | **+1.71** | 0.1094 | 1.0000 |
| 15 | Deep gated S3 | Deep gated S4 | one-sided | 4.1456 → 4.1801 | −0.83 | 0.8368 | 1.0000 |

Adding a modality never improves a Flat learner: all ten Ridge and XGBoost rows
have negative skill. The three positive rows are all Deep, and the shipping
increment from Deep S1 to gated S3 is the only one with a raw p below 5%
(0.0410). It does not survive adjustment within the fifteen-test family
(Holm p = 0.615), so it is reported as nominal evidence.

### B.6.3 RQ2 family (14 comparisons)

| # | Reference | Candidate | Direction | RMSE ref → cand | Skill % | Raw p | Holm p |
| ---: | --- | --- | --- | --- | ---: | ---: | ---: |
| 1 | Ridge S1 | Deep S1 | one-sided | 4.2563 → 4.2499 | +0.15 | 0.4663 | 1.0000 |
| 2 | XGB S1 | Deep S1 | one-sided | 4.3684 → 4.2499 | +2.71 | 0.0965 | 0.7648 |
| 3 | Ridge S2 | Deep gated S2 | one-sided | 4.4136 → 4.2528 | +3.64 | 0.0956 | 0.7648 |
| 4 | XGB S2 | Deep gated S2 | one-sided | 4.4402 → 4.2528 | +4.22 | **0.0423** | 0.4233 |
| 5 | Ridge S3 | Deep gated S3 | one-sided | 4.4471 → 4.1456 | +6.78 | 0.0640 | 0.5757 |
| 6 | XGB S3 | Deep gated S3 | one-sided | 4.4080 → 4.1456 | +5.95 | **0.0099** | 0.1322 |
| 7 | Ridge S4 | Deep gated S4 | one-sided | 4.5361 → 4.1801 | +7.85 | **0.0288** | 0.3173 |
| 8 | XGB S4 | Deep gated S4 | one-sided | 4.5061 → 4.1801 | +7.23 | **0.0094** | 0.1322 |
| 9 | Deep gated S2 | Deep concat S2 | two-sided | 4.2528 → 4.2352 | +0.41 | 0.7742 | 1.0000 |
| 10 | Deep gated S2 | Deep xattn S2 | two-sided | 4.2528 → 4.3956 | −3.36 | **0.0259** | 0.3113 |
| 11 | Deep gated S3 | Deep concat S3 | two-sided | 4.1456 → 4.1611 | −0.37 | 0.5884 | 1.0000 |
| 12 | Deep gated S3 | Deep xattn S3 | two-sided | 4.1456 → 4.1102 | +0.85 | 0.6231 | 1.0000 |
| 13 | Deep gated S4 | Deep concat S4 | two-sided | 4.1801 → 4.4964 | −7.57 | 0.1874 | 1.0000 |
| 14 | Deep gated S4 | Deep xattn S4 | two-sided | 4.1801 → 4.1437 | +0.87 | 0.4353 | 1.0000 |

All eight matched Deep–Flat rows favour Deep, and the margin widens as the
information set grows, from +0.15% at S1 against Ridge to +7.85% at S4. Four have
raw p below 5%, but the smallest Holm-adjusted p in the family is 0.132, so the
primary test supports no formal claim that the Deep pathway outperforms the Flat
pathway. Among the fusion mechanisms, no gated–concat or gated–cross-attention
contrast is distinguishable after adjustment; the strongest nominal contrast is
the gated-over-cross-attention margin at S2 (−3.36%, raw p = 0.026, Holm p =
0.311), while the largest raw gap, gated over concat at S4 (−7.57%), is far from
significant at raw p = 0.187.

### B.6.4 Supplementary Clark–West tests (Ridge only)

Clark–West presumes that the smaller specification is a parameter restriction of
the larger one. Only the Ridge information-set extensions come close to that
structure, and even there the penalty is re-tuned at every re-estimation, so these
five rows are supplementary rather than primary. They are reported for
completeness and are not used for any formal claim. Source:
`05_outputs/tests/test_table_cw_supplementary.csv`.

| # | Reference | Candidate | RMSE ref → cand | CW statistic | Raw p | Holm p |
| ---: | --- | --- | --- | ---: | ---: | ---: |
| 1 | Ridge S1 | Ridge S2 | 4.2563 → 4.4136 | 0.066 | 0.4738 | 1.0000 |
| 2 | Ridge S1 | Ridge S3 | 4.2563 → 4.4471 | 0.532 | 0.2975 | 1.0000 |
| 3 | Ridge S1 | Ridge S4 | 4.2563 → 4.5361 | 0.386 | 0.3499 | 1.0000 |
| 4 | Ridge S2 | Ridge S4 | 4.4136 → 4.5361 | 0.594 | 0.2763 | 1.0000 |
| 5 | Ridge S3 | Ridge S4 | 4.4471 → 4.5361 | 0.350 | 0.3630 | 1.0000 |

No Ridge extension is significant even before adjustment, and every RMSE moves in
the wrong direction, so the supplementary test agrees with the primary one here.
This is the contrast that motivates restricting Clark–West to Ridge: applied to
XGBoost, the same adjustment term produced apparent significance for arms whose
RMSE was worse than the baseline (B.3.2).

---

## B.7 Sensitivity analyses

Each check re-runs the same three frozen families and applies Holm within
(check × family), so the adjustment mirrors the main table and is never pooled
across checks. These analyses are exploratory: they qualify the primary result
but do not replace it. Source: `05_outputs/tests/test_table_robustness.csv`.

| Check | Comparisons | Holm p < 0.05 |
| --- | ---: | ---: |
| Absolute-error loss | 47 | 3 |
| Early sub-period ≤2022 | 47 | 0 |
| Late sub-period ≥2023 | 47 | 1 |
| Two-sided sensitivity | 41 | 1 |
| One-sided sensitivity | 6 | 0 |

### B.7.1 Absolute-error loss

Replacing squared error with absolute error changes the RQ2 conclusion. Three
matched Deep–Flat comparisons survive Holm within the fourteen-test family:

| Comparison | RMSE change | MAE change | Raw p | Holm p |
| --- | ---: | ---: | ---: | ---: |
| XGB S3 → Deep gated S3 | +5.95% | +8.26% | 0.0014 | **0.0171** |
| Ridge S4 → Deep gated S4 | +7.85% | +8.80% | 0.0011 | **0.0140** |
| XGB S4 → Deep gated S4 | +7.23% | +9.62% | 0.0010 | **0.0139** |

The primary test is defined on squared error because that is the loss the headline
RMSE metric reports, so the formal RQ2 conclusion remains the null result of
B.6.3. The divergence is informative rather than contradictory: squared error is
dominated by a small number of large-move weeks, whereas absolute error weights
every week equally. The Deep advantage is therefore concentrated in typical weeks
and does not extend to the weeks that drive RMSE. No benchmark or RQ1 comparison
survives under this loss.

### B.7.2 Early and late sub-periods

Splitting at the end of 2022 gives 102 early and 155 late origins. No comparison
survives Holm in the early sub-period, where the RQ2 minimum is 0.387. One
survives in the late sub-period: Ridge S4 → Deep gated S4, +8.09% RMSE and +9.86%
MAE, raw p = 0.0027 and Holm p = 0.038. A single surviving comparison in one half
of the sample, with the same contrast absent in the other half, is weak evidence;
it is consistent with the sub-period instability already documented in B.1 rather
than with a stable Deep advantage.

### B.7.3 Two-sided sensitivity

Section 3.7.2 uses one-sided tests where the research question is directional. To
show how much the conclusions depend on that choice, every one-sided comparison is
re-run two-sided, and the six two-sided fusion comparisons are re-run one-sided.
The RQ2 conclusion is unchanged: its smallest two-sided Holm p is 0.151, against
0.132 under the primary one-sided test, so neither version supports a formal
claim. One benchmark comparison becomes significant, and its direction matters:
M0 versus Deep xattn S2 gives Holm p = 0.038 with RMSE −5.87% and MAE −7.11%,
that is, cross-attention on finance plus remote sensing is detectably **worse**
than the no-change forecast. The two-sided variant can reject in either
direction, and here it rejects against the model.

### B.7.4 Leave-out influence diagnostic

The diagnostic drops the eight weeks with the largest absolute loss differential
and re-runs the comparison, to check whether a surviving result is carried by a
few observations. Because the dropped weeks are selected from the observed loss
differential, it is data-dependent and is an influence diagnostic, not a test; no
familywise adjustment is applied to it. It is triggered only for comparisons that
are Holm-significant in the main table, and since none is, the diagnostic produced
no rows in this study.

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
| Primary test                  | Diebold–Mariano with HLN small-sample correction, on reconstructed-price squared error, for **every** formal comparison (vs M0, vs S1, and Flat vs Deep) |
| Test direction                | one-sided where the research question is directional (vs M0, vs S1, Deep vs Flat); two-sided for Deep fusion-mechanism comparisons |
| Multiplicity                  | Holm within three frozen families: benchmark (18), RQ1 (15), RQ2 (14); raw and adjusted p both reported, formal claims on adjusted |
| Supplementary test            | Clark–West, **Ridge only** (5 comparisons), never for XGBoost or Deep |
| Unified test table            | `05_outputs/tests/test_table_main.csv` via `04_code/scripts/tools/build_test_tables.py` |
| Seed                          | **42** (main); 1, 2 for robustness                    |

**Variance estimation in the DM statistic.** The loss differential is
\(d_t = L_{\text{reference},t} - L_{\text{candidate},t}\), so a positive statistic
means the candidate is the more accurate forecast. Its long-run variance is
estimated with the usual truncation at \(h-1\) autocovariances, where \(h\) is the
forecast horizon. Every comparison in this study is a one-week-ahead forecast, so
\(h = 1\) and no autocovariance term enters: the variance reduces to
\(\hat{\gamma}_0 / T\), where \(\hat{\gamma}_0 = T^{-1}\sum_t (d_t - \bar{d})^2\).
The Harvey–Leybourne–Newbold finite-sample factor
\(\sqrt{[T + 1 - 2h + h(h-1)/T]\,/\,T}\) is then applied, which at \(h = 1\)
equals \(\sqrt{(T-1)/T}\), and the statistic is referred to a \(t\) distribution
with \(T-1\) degrees of freedom. Implementation:
`04_code/src/backtest/metrics.py::dm_test`.


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
| **Frozen comparison families + Holm**          | `04_code/scripts/tools/build_test_tables.py`                                 | `05_outputs/tests/test_table_{main,cw_supplementary,robustness}.csv` |
| Interpretability (gates, attention)            | `run_deep_interpret.py`, `run_deep_interpret_m3.py`, `run_deep_xattn_viz.py` | `deep_interpret*.png`, `deep_*gate*.csv`                  |
| Feature matrix build                           | `03_data/processed/**/build_*.py`, `merge/py/build_feature_matrix.py`        | `03_data/processed/merge/outputs/`                        |


Reproduce end-to-end:

```bash
python3 -m pip install -r 04_code/requirements.txt
python3 04_code/scripts/flat/run_baseline.py --modality M3      # flat example
python3 04_code/scripts/deep/run_deep_baseline.py               # deep main
python3 04_code/scripts/tools/subperiod_eval.py                  # early/late table
python3 04_code/scripts/tools/build_test_tables.py               # all reported p-values
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
