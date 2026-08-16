# Chapter 3 — Methodology ( 3,599 words)

## 3.1 Research design

## 3.1 研究设计

This chapter sets out how the study answers the research questions in Section 1.2. Every learned forecast is judged against a simple no-change benchmark in which next week’s Brent price equals this week’s price. The study then asks whether remote sensing and shipping add useful information beyond financial time series, and whether modelling those inputs as one weekly table differs from encoding each data type separately before combining them. For models that improve on the benchmark, it also asks what they rely on. All specifications share the same weekly forecast dates, sample window and evaluation rules, so that differences in the information set can be separated from differences in modelling strategy.

本章说明研究如何回答第 1.2 节的研究问题。每个经学习得到的预测都对照一个简单的不变预测基准，即下周 Brent 价格等于本周价格。研究再问遥感与航运是否在金融时序之外仍提供有用信息，以及把这些输入作为一张周表来建模，是否不同于先按数据类型分别编码再组合。对优于基准的模型，研究还问它依赖哪些信息。所有设定共用同一周度预测日、样本窗口与评估规则，从而把信息集差异与建模策略差异分开。

The no-change benchmark is denoted M0. At each forecast origin t, where P_t is the Brent price in week t, M0 sets the one-week-ahead price forecast equal to the current weekly price:

不变预测基准记为 M0。设 P_t 为第 t 周的 Brent 价格，则在每个预测起点 t，M0 将提前一周的价格预测设为当前周价格：

\hat{P}_{t+1|t}=P_t.

M0 needs no parameter estimation and contains no predictors. It is a reference forecast rather than one of the learned specifications below, and every learned model is compared with it over the same out-of-sample evaluation period.

M0 无需参数估计，也不含预测变量。它是参照预测，不属于下文经过学习的设定；每个学习模型都在同一样本外评价期上与 M0 比较。

The predictors are organised into four information sets. S1 contains financial time series only, comprising financial, macroeconomic and oil-market variables. S2 adds remote sensing to S1, S3 adds shipping to S1, and S4 adds both modalities. S2 and S3 are parallel extensions of S1 rather than successive stages, while S4 combines the two. Table 3.1 lists the four sets together with the M0 benchmark.

预测变量组织为四个信息集。S1 仅含金融时序，包括金融、宏观与油市变量。S2 在 S1 上加入遥感，S3 在 S1 上加入航运，S4 两者都加。S2 与 S3 是对 S1 的平行扩展，而非先后阶段；S4 合并二者。表 3.1 将四个集合与 M0 基准一并列出。

**Table 3.1 — Information sets**

**表 3.1 — 信息集**

| Set            | Variables                                                                   |
| -------------- | --------------------------------------------------------------------------- |
| Benchmark (M0) | Last week's price                                                           |
| S1             | Financial time series only (financial, macroeconomic and oil-market series) |
| S2             | S1 + remote sensing                                                         |
| S3             | S1 + shipping                                                               |
| S4             | S1 + remote sensing + shipping                                              |

| 集合           | 变量                                                     |
| ------------ | ------------------------------------------------------ |
| 基准（M0）       | 上一周价格                                                  |
| S1           | 仅金融时序（金融、宏观与油市序列）                                      |
| S2           | S1 + 遥感                                                |
| S3           | S1 + 航运                                                |
| S4           | S1 + 遥感 + 航运                                           |

Within each model family, the information-set contrasts hold the forecasting method and evaluation sample constant and change only the information set. S2 against S1 measures the contribution of remote sensing added on its own, and S3 against S1 that of shipping. S4 against S1 measures their joint contribution. S4 against S3 and S4 against S2 ask whether each source still helps once the other is already included. These contrasts, together with each model’s comparison against M0, address RQ1.

在每一模型族内部，信息集对比保持预测方法与评价样本不变，只改变信息集。S2 对 S1 度量单独加入遥感的贡献，S3 对 S1 度量航运的贡献。S4 对 S1 度量二者的联合贡献。S4 对 S3 与 S4 对 S2 则问：在另一数据源已经纳入后，这一源是否仍有帮助。这些对比连同每个模型与 M0 的比较，共同回答 RQ1。

Two model families are applied to these information sets. The Flat family puts all selected predictors into one weekly table—stacking recent weeks into a single row—and fits Ridge and XGBoost. This early joining of features is called flat feature fusion. The Deep family keeps each data type separate at first. Financial series, remote-sensing imagery and shipping-network inputs each pass through their own encoder, and the outputs are then combined. The main Deep design, gated fusion, learns how much weight to give each data type. Flat and Deep are then compared on the same information set and the same evaluation sample. These comparisons measure the overall difference between the two modelling strategies. They do not isolate the effect of fusion alone, because the two families also differ in model class, capacity and some modality-specific input products and representations (Table 3.2). Fusion is assessed more directly within the Deep family. Simple concatenation, gated fusion and cross-attention are compared with the encoders and inputs held fixed. Together these two comparisons address RQ2.

两套模型族应用于这些信息集。Flat 族把所选预测变量放入一张周度表——把最近几周叠成一行——再拟合 Ridge 与 XGBoost。这种一开始就合并特征的做法称为扁平特征融合。Deep 族则先按数据类型分开处理。金融序列、遥感影像与航运网络输入各自经过自己的编码器，再把输出组合起来。Deep 的主设计即门控融合，学习给各类数据多少权重。随后在相同信息集与相同评价样本上比较 Flat 与 Deep。这类比较衡量的是两种整体建模策略的差异。它不能单独分离出融合方式的作用，因为两族在模型类型、容量以及部分模态专属输入产品与表征上同样存在差异（见表 3.2）。融合方式本身在 Deep 族内部得到更直接的评估。在编码器与输入保持不变的前提下，比较简单拼接、门控融合与交叉注意力。这两组比较共同回答 RQ2。

RQ3 is restricted to Deep models that improve on M0 according to the predefined criterion **defined in Section 3.7？？要写3.7吗**. For these specifications the study reports the weights assigned to finance, remote sensing and shipping, together with the sites or network nodes receiving greater attention under different market conditions. These quantities indicate what a model relies on. They are not interpreted as evidence of causal importance.

RQ3 仅限于按预先设定的准则优于 M0 的 Deep 设定（**判定准则见第 3.7 节？？要写 3.7 吗**）。对这些设定，本研究报告其赋予金融、遥感与航运的权重，以及不同市场条件下受到更多关注的站点或网络节点。这些量说明模型依赖哪些信息，但不被解释为因果重要性的证据。

Figure 3.1 summarises the research design.

图 3.1 概括研究设计。

Figure 3.1

**Figure 3.1 — Research design: data blocks, the M0 benchmark and information sets S1–S4, the Flat and Deep families, and the shared expanding-window evaluation.**

**图 3.1 — 研究设计：数据块、M0 基准与信息集 S1–S4、Flat 与 Deep 两族，以及共用的扩展窗评估。**

## 3.2 Prediction target and timeline

## 3.2 预测目标与时间轴

Let P_t denote the last available daily Brent spot-price observation in week t, where each week ends on Friday, measured in US dollars per barrel. The forecast target is next week’s price P_{t+1}. Models are not trained directly on the price level. They predict the one-week logarithmic return

令 P_t 表示第 t 周内最后一个可获得的 Brent 现货价格日度观测，各周于周五结束，单位为美元/桶。预测目标是下一周的价格 P_{t+1}。模型不直接在价格水平上训练，而是预测一周对数收益

r_{t+1}=\log\left(\frac{P_{t+1}}{P_t}\right)

and reconstruct the price forecast as

并按下式重构价格预测：

\hat{P}_{t+1|t}=P_t\exp\left(\hat{r}_{t+1|t}\right).

Log returns are used to reduce the strong persistence in the price level and to express the forecasting task in terms of proportional weekly changes. RMSE, MAE and skill versus M0 are computed from the reconstructed price forecasts. Under this mapping, the no-change benchmark \hat{P}_{t+1|t}=P_t is exactly the same as forecasting a zero return \hat{r}_{t+1|t}=0.

使用对数收益是为了减弱价格水平的强持续性，并将预测任务表示为周度比例变化。RMSE、MAE 以及相对 M0 的 skill 均根据重构后的价格预测计算。在此对应关系下，不变预测基准 \hat{P}_{t+1|t}=P_t 与预测收益为零 \hat{r}_{t+1|t}=0 完全一致。

The modelling window covers 2019–2025 and provides a common weekly index of 365 observations (4 January 2019 to 26 December 2025). The training and evaluation samples are separated in time on an expanding window, rather than by random assignment, to prevent future information from leaking into model fitting. The full validation protocol is in Section 3.6.

建模窗口覆盖 2019–2025 年，提供含 365 个观测的共同周索引（2019 年 1 月 4 日至 2025 年 12 月 26 日）。训练样本与评价样本在扩展窗口下按时间分开，而非随机划分，以避免未来信息泄漏到模型拟合中。完整验证协议见第 3.6 节。

## 3.3 Geographic scope and monitoring sites

## 3.3 地理范围与监测站点

Because the prediction target is the global Brent benchmark rather than a local physical cargo price at a single terminal, the study does not use one contiguous study region. Spatial information instead comes from eleven oil-infrastructure monitoring sites and six maritime chokepoints. Together they cover major supply, transit, refining and demand locations in the international oil system. Figure 3.3 places these sites and chokepoints on a world map. Full site names, coordinates, patch sizes and graph edge definitions are in Appendix A.

由于预测对象是全球 Brent 基准，而非单一码头的本地实物货价，本研究不采用一块连续的地理研究区。空间信息来自十一个石油基础设施监测站点与六个航运咽喉。它们共同覆盖国际石油体系中的主要供给、中转、炼化与需求区位。图 3.3 在世界地图上标出这些站点与咽喉。完整站名、坐标、裁剪范围与图边定义见附录 A。

**The eleven sites are ports, refineries and export terminals, purposively selected for strategic coverage of supply, transit, refining and demand locations and for observability in the available satellite products.In the Flat pathway, remote-sensing features are summarised inside a circular buffer with a radius of 5 km around each site. In the Deep pathway, image patches are cut around each site, and patch size follows facility type and local spatial constraints: generally larger for ports, intermediate for refineries and smaller for terminals. The two pathways therefore observe the same locations over different spatial extents.**

**The eleven sites comprise ports, refineries and export terminals selected purposively for their strategic roles and observability in the available satellite products.Flat remote-sensing features are summarised within a circular buffer of 5 km radius around each site. Deep image patches are centred on the same sites but vary in size by facility type and local spatial constraints, so the two pathways cover different spatial extents.**简洁版替换？


十一个站点为港口、炼厂与出口码头，按目的性原则选取，兼顾供给、中转、炼化与需求区位的战略覆盖，以及在可用卫星产品中的可观测性。
十一个站点包括港口、炼厂和出口码头，并依据其战略作用以及在可用卫星产品中的可观测性进行目的性选取。
Flat 路径在以各站点为中心、半径 5 km 的圆形缓冲区内汇总遥感特征。Deep 路径按站点裁剪影像块，裁剪大小随设施类型与当地空间约束而定：一般港口较大、炼厂居中、码头较小。因此两条路径观测同一批地点，但空间范围不同。

The shipping graph augments the eleven sites with six maritime chokepoints: the Strait of Hormuz, the Suez Canal, the Strait of Malacca, Bab el-Mandeb, the Panama Canal and the Cape of Good Hope.[^cape] 
**The resulting weekly graph contains seventeen nodes and two forms of connection. Dynamic links between the eleven AOIs are directed origin–destination pairs, weighted by the number of voyages counted in each week from Global Fishing Watch port-visit sequences, and therefore change from week to week. Fixed links are undirected. They connect each site to the chokepoint or chokepoints on its documented principal oil-trade corridor and are specified ex ante rather than inferred from weekly vessel movements or geographic proximity. Complete edge definitions are reported in Appendix A.4, and graph encoding is described in Section 3.5.2.**
**The resulting weekly graph contains 17 nodes and two edge classes. Directed AOI–AOI links are weighted by weekly voyage counts derived from GFW port-visit sequences, while fixed undirected links connect sites to chokepoints on predefined oil-trade corridors. Complete edge definitions are reported in Appendix A.4, and graph encoding is described in Section 3.5.2.**简洁版替换？

航运图在十一个站点之外加入六个航运咽喉：霍尔木兹海峡、苏伊士运河、马六甲海峡、曼德海峡、巴拿马运河与好望角。[^cape] 由此形成包含十七个节点、两类连接的周度图。十一个 AOI 之间的动态边为有向的起点–终点对，权重取各周内由 Global Fishing Watch 港口访问序列统计到的航次数，因而随周变化。固定边为无向。它们将每个站点与其已记录的主要石油贸易走廊上的一个或多个咽喉相连，并事先指定，而非由周度船舶移动或地理邻近推断。完整边定义见附录 A.4，图编码方法见第 3.5.2 节。

Figure 3.3

**Figure 3.3 — Spatial coverage of the 11 oil-infrastructure AOIs and six maritime chokepoints.**

**图 3.3 — 研究的空间覆盖：11 个石油基础设施 AOI 与 6 个航运咽喉。**

[^cape]: The Cape of Good Hope is included as a major oil-trade route rather than a narrow chokepoint in the sense of the EIA World Oil Transit Chokepoints report.

好望角按主要石油贸易航路纳入，而非 EIA《世界石油运输咽喉》口径下的狭义咽喉。

## 3.4 Data sources and preparation

## 3.4 数据来源与准备

### 3.4.1 Data sources

### 3.4.1 数据来源

The study uses three data blocks comprising financial time series, satellite remote sensing and maritime shipping data. Flat and Deep use the same monitoring geography but differ in the products and representations used for the remote-sensing and shipping blocks (Table 3.2). Flat models use predictors assembled in a merged weekly feature table, whereas Deep models retain modality-specific sequences, image embeddings and graph inputs.

本研究使用三个数据块，包括金融时序、卫星遥感与航运数据。Flat 与 Deep 使用相同的监测地理，但遥感块与航运块所用的产品与表征不同（表 3.2）。Flat 模型使用合并周度特征表中的预测变量，Deep 模型则保留模态专属的序列、影像嵌入与图输入。

**The study organises predictors into financial, remote-sensing and shipping blocks (Table 3.2). Flat models assemble them in a merged weekly feature table, whereas Deep models retain modality-specific sequences, image embeddings and graph inputs.**简洁版替换？

本研究将预测变量划分为金融、遥感与航运三个数据块（表 3.2）。Flat 模型将其汇入一张周度特征表，Deep 模型则保留模态专属的序列、影像嵌入与图输入。

**Table 3.2 — Datasets, variables and sources**

**表 3.2 — 数据集、变量与来源**

| Modality              | Dataset / product                                                                     | Key variables                                                                                                                  | Source                                                                                                                                                                                                                                                |
| --------------------- | ------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Financial time series | Oil-market and macro-financial series at daily, weekly and monthly native frequencies | Prices, inventories, production, interest rates, GPR and related indicators                                                    | [EIA](https://www.eia.gov/petroleum/supply/weekly/); [FRED](https://fred.stlouisfed.org/); [Yahoo Finance](https://finance.yahoo.com/); [Dallas Fed IGREA](https://www.dallasfed.org/research/igrea); [GPR](https://www.matteoiacoviello.com/gpr.htm) |
| Remote sensing (Flat) | Sentinel-2 optical indices and VIIRS night-time lights                                | Site-level anomalies at 11 AOIs (NDVI, NDWI, NDBI, BSI; NTL)                                                                   | [Sentinel-2 via GEE](https://developers.google.com/earth-engine/datasets/catalog/COPERNICUS_S2_SR_HARMONIZED); [VIIRS via GEE](https://developers.google.com/earth-engine/datasets/catalog/NOAA_VIIRS_DNB_MONTHLY_V1_VCMSLCFG)                        |
| Remote sensing (Deep) | Monthly Sentinel-2 image patches                                                      | Frozen Prithvi-EO-2.0 embeddings at the same 11 AOIs                                                                           | [Prithvi-EO-2.0](https://huggingface.co/ibm-nasa-geospatial); [Sentinel-2 via GEE](https://developers.google.com/earth-engine/datasets/catalog/COPERNICUS_S2_SR_HARMONIZED)                                                                           |
| Shipping (Flat)       | PortWatch and Global Fishing Watch AIS data                                           | Port and chokepoint tanker flows; vessel-activity features                                                                     | [IMF PortWatch](https://portwatch.imf.org/) (AIS-derived); [Global Fishing Watch](https://globalfishingwatch.org/our-apis/) (AIS-derived)                                                                                                             |
| Shipping (Deep)       | PortWatch and Global Fishing Watch AIS and SAR data                                   | Weekly node attributes, dynamic voyage links and fixed corridor links for a 17-node graph comprising 11 AOIs and 6 chokepoints | [IMF PortWatch](https://portwatch.imf.org/) (AIS-derived); [Global Fishing Watch](https://globalfishingwatch.org/our-apis/) (AIS- and SAR-derived)                                                                                                    |

| 模态       | 数据集 / 产品                                        | 关键变量                                        | 来源                                                                                                                                                                                                                                                |
| -------- | ----------------------------------------------- | ------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 金融时序     | 日、周、月原生频率的油市与宏观金融序列                             | 价格、库存、产量、利率、GPR 及相关指标                       | [EIA](https://www.eia.gov/petroleum/supply/weekly/)；[FRED](https://fred.stlouisfed.org/)；[Yahoo Finance](https://finance.yahoo.com/)；[Dallas Fed IGREA](https://www.dallasfed.org/research/igrea)；[GPR](https://www.matteoiacoviello.com/gpr.htm) |
| 遥感（Flat） | Sentinel-2 光学指数与 VIIRS 夜光                       | 11 个 AOI 的站点级距平（NDVI、NDWI、NDBI、BSI 与 NTL）   | [Sentinel-2 via GEE](https://developers.google.com/earth-engine/datasets/catalog/COPERNICUS_S2_SR_HARMONIZED)；[VIIRS via GEE](https://developers.google.com/earth-engine/datasets/catalog/NOAA_VIIRS_DNB_MONTHLY_V1_VCMSLCFG)                     |
| 遥感（Deep） | 月度 Sentinel-2 影像块                               | 同一 11 个 AOI 的冻结 Prithvi-EO-2.0 嵌入           | [Prithvi-EO-2.0](https://huggingface.co/ibm-nasa-geospatial)；[Sentinel-2 via GEE](https://developers.google.com/earth-engine/datasets/catalog/COPERNICUS_S2_SR_HARMONIZED)                                                                        |
| 航运（Flat） | PortWatch 与 Global Fishing Watch 的 AIS 数据       | 港口与咽喉油轮流量；船舶活动特征                            | [IMF PortWatch](https://portwatch.imf.org/)（AIS 衍生）；[Global Fishing Watch](https://globalfishingwatch.org/our-apis/)（AIS 衍生）                                                                                                                      |
| 航运（Deep） | PortWatch 与 Global Fishing Watch 的 AIS 与 SAR 数据 | 17 节点周度图的节点属性、动态航次边与固定走廊边（11 个 AOI 与 6 个咽喉） | [IMF PortWatch](https://portwatch.imf.org/)（AIS 衍生）；[Global Fishing Watch](https://globalfishingwatch.org/our-apis/)（AIS 与 SAR 衍生）                                                                                                                |

The financial block combines oil-market and macro-financial series from the US Energy Information Administration (EIA), Federal Reserve Economic Data (FRED), Yahoo Finance, the Dallas Fed Index of Global Real Economic Activity, and the Caldara–Iacoviello geopolitical risk (GPR) index. 
**These series are observed at different native frequencies and include crude prices and spreads, inventories, production and refinery activity, volatility and risk measures, interest rates, exchange rates, futures-based oil indicators and geopolitical risk.**
**Observed at daily, weekly and monthly native frequencies, these variables enter the Deep finance encoder as a single multivariate sequence.**简洁版替换？

 In the Deep pathway, these variables form a single multivariate input sequence for the finance encoder, spanning oil-market, macro-financial and geopolitical indicators.

金融块汇总来自美国能源信息署（EIA）、联邦储备经济数据（FRED）、Yahoo Finance、达拉斯联储全球实际经济活动指数和 Caldara–Iacoviello 地缘政治风险（GPR）指数的油市与宏观金融序列。这些序列具有不同的原生频率，包括原油价格与价差、库存、产量与炼厂活动、波动与风险度量、利率、汇率、基于期货的油市指标以及地缘政治风险。在 Deep 路径中，这些变量构成金融编码器的单一多元输入序列，涵盖油市、宏观金融与地缘政治指标。

The Flat pathway uses Sentinel-2 optical indices and VIIRS night-time lights, whereas the Deep pathway uses frozen Prithvi-EO-2.0 embeddings derived from Sentinel-2 image patches and has no separate VIIRS input stream.

两条路径均从同一十一个 AOI 提取遥感输入，但空间范围、产品与表征不同。Flat 路径使用 Sentinel-2 光学指数与 VIIRS 夜光，Deep 路径使用由 Sentinel-2 影像块得到的冻结 Prithvi-EO-2.0 嵌入，且无单独的 VIIRS 输入流。

**The shipping block covers activity at the eleven AOIs and six chokepoints. IMF PortWatch supplies AIS-derived tanker-flow measures at ports and chokepoints. Global Fishing Watch supplies AIS-derived measures of vessel presence and activity duration, together with port-visit records. In the Flat pathway, PortWatch and GFW AIS-derived series enter as weekly tabular features, while SAR-derived dark-vessel variables are not included in the main feature set. In the Deep pathway, GFW SAR-derived dark-vessel detections enter as node attributes in the 17-node graph, while port-visit sequences are used to construct dynamic AOI–AOI voyage links. In both pathways, these variables serve as proxies for physical shipping activity, tanker movements and congestion.**

航运块覆盖十一个 AOI 与六个咽喉的活动。IMF PortWatch 提供港口与咽喉的 AIS 衍生油轮流量。Global Fishing Watch 提供 AIS 衍生的船舶存在量与活动时长，以及港口访问记录。在 Flat 路径中，PortWatch 与 GFW 的 AIS 衍生序列以周度表格特征进入，主特征集不含 SAR 暗船变量。在 Deep 路径中，GFW 的 SAR 暗船检测作为十七节点图的节点属性进入，港口访问序列则用于构造 AOI 之间的动态航次边。两条路径都将这些变量视为实物航运活动、油轮移动与拥堵的代理。

**The shipping block combines PortWatch tanker-flow measures with GFW vessel-activity and port-visit data. Flat models use the AIS-derived series as weekly tabular features and exclude SAR dark-vessel variables, whereas Deep models add SAR detections as node attributes and use port visits to construct dynamic voyage links. These inputs serve as proxies for physical shipping activity, tanker movements and congestion. Full variable definitions are reported in Appendix A.1.**简洁版替换？

航运块结合 PortWatch 油轮流量指标与 GFW 船舶活动和港口访问数据。Flat 模型将 AIS 衍生序列作为周度表格特征，并排除 SAR 暗船变量；Deep 模型则将 SAR 棬测加入节点属性，并使用港口访问记录构造动态航次边。这些输入作为实物航运活动、油轮移动与拥堵的代理。完整变量定义见附录 A.1。

### 3.4.2 Temporal alignment and publication lags

### 3.4.2 时间对齐与发布滞后

**All series are aligned to a common Friday-ending weekly calendar before modelling. Daily market series use the last available observation within each week, daily GPR is averaged over the week, and daily PortWatch tanker counts and capacities are summed weekly. Monthly macroeconomic and GFW series are carried forward from month-end only after their assumed availability dates.**
**All series are aligned to a common Friday-ending weekly calendar. Daily observations are converted using end-of-week values, weekly means or weekly sums as appropriate, while monthly series are carried forward only after their assumed availability dates.所有序列均对齐至共同的周五截止周历。日度观测根据变量性质采用周末值、周均值或周总和，月度序列则只有在假定可用日期到达后才向后延续。**
全部序列在建模前对齐到共同的周五截止周历。日度市场序列取各周内最后一个可得观测，日度 GPR 在周内取均值，日度 PortWatch 油轮计数与运力按周求和。月度宏观与 GFW 序列仅在假定可得日之后，才自月末向前填入。

Monthly remote-sensing products, including Flat anomalies and Deep Prithvi embeddings, are attached by as-of alignment and become eligible only after their conservative availability dates. The most recent eligible composite is then carried forward across subsequent Fridays, so the four-week input window may contain repeated monthly vectors.
**Monthly remote-sensing products are aligned as of their conservative availability dates and the most recent eligible composite is carried forward, so a four-week input window may contain repeated monthly vectors.月度遥感产品按其保守可用日期进行 as-of 对齐，最近一期已可用的合成结果随后向后延续，因此四周输入窗口可能包含重复的月度向量。**
月度遥感产品（包括 Flat 距平与 Deep Prithvi 嵌入）采用 as-of 对齐，仅在保守可得日已过之后才进入。最近一期合格合成再向后续周五沿用，因此四周输入窗中可能出现重复的月度向量。

Publication timing is represented by source- and product-specific fixed lag buffers implemented in the data builders rather than by observation-level release timestamps. 
**One-week buffers are applied to EIA fundamentals and PortWatch flows, while monthly macroeconomic series, remote-sensing products and individual GFW AIS and SAR products receive longer, product-specific buffers. Exact lag constants and implementation scripts are reported in Appendix A.3. The downloaded series are currently available revised histories.**
**Publication timing is approximated using source- and product-specific fixed lag buffers rather than observation-level release timestamps. Exact aggregation rules, lag constants and implementation scripts are reported in Appendix A.3. The analysis uses currently available revised histories rather than real-time vintages.发布时间通过按来源和产品设定的固定滞后缓冲近似，而不是逐条采用观测值的实际发布时间戳。具体聚合规则、滞后常数与实现脚本见附录 A.3。本研究使用当前可得的修订后历史序列，而非实时 vintage 数据。**

发布时间以数据构建脚本中按来源与产品设定的固定滞后缓冲表示，而不是按每条观测的官方发布时间戳。EIA 基本面与 PortWatch 流量施加一周缓冲，月度宏观序列、遥感产品以及 GFW 的各 AIS 与 SAR 产品则使用更长的产品专属缓冲。精确滞后常数与实现脚本见附录 A.3。所用下载序列为当前可得的修订后历史。

### 3.4.3 Missing data and quality control

### 3.4.3 缺失数据与质量控制

**Optical-image availability varies across sites and months. During monthly compositing in Google Earth Engine, cloud probability and valid-observation counts are used to filter scenes and pixels before monthly medians are calculated. These quality indicators are not included as predictors.

光学影像在不同站点与月份的可得性并不相同。在 Google Earth Engine 月度合成中，云概率与有效观测计数用于在计算月度中位数之前筛选场景与像元。这些质量指标不作为预测变量进入模型。

On the resulting weekly calendar, mean coverage across the four optical indices is approximately 97 per cent, with eight of the eleven sites fully observed, while VIIRS night-time-light anomalies are fully observed. Site-level coverage rates and counts of independent monthly composites are reported in Appendix A.5.

在由此得到的周历上，四个光学指数的平均覆盖率约为 97%，其中十一个站点中有八个为满观测，VIIRS 夜光距平为满观测。站点级覆盖率与独立月度合成计数见附录 A.5。**

**Monthly optical composites are cloud-filtered before the indices are constructed, and cloud-quality indicators are not used as predictors. On the weekly calendar, mean coverage across the four optical indices is approximately 97 per cent, while VIIRS night-time-light anomalies are fully observed. Site-level coverage and counts of independent monthly composites are reported in Appendix A.5.月度光学合成影像在构建指数前进行云筛选，云质量指标不作为预测变量。在周度日历上，四项光学指数的平均覆盖率约为 97%，VIIRS 夜间灯光距平则完全可用。站点级覆盖率与独立月度合成数量见附录 A.5。**

**After temporal alignment, remaining gaps in Flat predictors are forward-filled using past observations only. Residual leading gaps are set to zero for remote-sensing anomalies and imputed using training-fold medians for PortWatch and other shipping-count variables. Deep finance inputs are complete after merging. After training-window scaling, missing Deep remote-sensing embeddings and shipping-graph values are set to zero. Remote-sensing embeddings additionally retain binary availability masks that exclude missing positions from attention, whereas shipping graphs do not use missingness masks. All imputation and scaling parameters are estimated within each training window.**

时间对齐后，Flat 预测变量中的剩余缺口仅使用过去观测做前向填充。残余前导缺口中，遥感距平设为零，PortWatch 与其他航运计数变量则使用训练折中位数填补。Deep 金融输入在合并后是完整的。经过训练窗口缩放后，缺失的 Deep 遥感嵌入与航运图数值均设为零。遥感嵌入还保留二元可用性掩码，使注意力排除缺失位置，而航运图不使用缺失掩码。所有填补与缩放参数均在各训练窗口内估计。
**After temporal alignment, gaps in Flat predictors are forward-filled using past observations only. Residual leading gaps are set to zero for remote-sensing anomalies and imputed with training-fold medians for shipping-count variables. Deep finance inputs are complete after merging. Missing Deep remote-sensing embeddings and shipping-graph values are zero-filled after training-window scaling, but only remote-sensing embeddings retain availability masks. All imputation and scaling parameters are estimated within the corresponding training window.时间对齐后，Flat 预测变量的缺口仅使用过去观测进行前向填充。剩余前导缺口中，遥感距平设为零，航运计数变量使用训练折中位数填补。Deep 金融输入在合并后不存在缺失。Deep 遥感嵌入和航运图数值在训练窗口缩放后以零填补，但只有遥感嵌入保留可用性掩码。所有填补与缩放参数均在相应训练窗口内估计。**

## 3.5 Forecasting models

## 3.5 预测模型

### 3.5.1 Flat models

### 3.5.1 Flat 模型

Flat models implement early feature-level fusion. For each information set, all available numeric features are concatenated into a weekly feature table, and the most recent four weeks are flattened into a single row for each forecast origin. Ridge applies L2 regularisation (Hoerl and Kennard, 1970) and serves as a transparent linear comparator. XGBoost is a non-linear gradient-boosted tree ensemble (Chen and Guestrin, 2016) that captures nonlinearities and interactions not represented by Ridge. Because both learners operate on the same flattened table, neither preserves modality-specific structure. Both predict the one-week-ahead log return and then reconstruct the corresponding price forecast. Hyperparameter selection follows the time-ordered procedure described in Section 3.6, with exact search grids reported in Appendix C.

Flat 模型采用早期特征级融合。对每个信息集，所有可用数值特征拼接为一张周度特征表，并在每个预测起点将最近四周展平为一行。该表上估计两种学习器。Ridge 采用 L2 正则化（Hoerl and Kennard, 1970），作为透明的线性对照。XGBoost 是非线性梯度提升树集成（Chen and Guestrin, 2016），用于捕捉 Ridge 无法表示的非线性与交互。由于两种学习器都作用于同一张展平表，因此都不保留模态专属结构。二者均预测提前一周的对数收益，再还原相应的价格预测。超参数选择遵循第 3.6 节所述的时序程序，具体搜索网格见附录 C。

### 3.5.2 Deep models

### 3.5.2 Deep 模型

Deep models encode each modality separately and fuse the resulting representations for S2–S4.

Deep 模型对每个模态分别编码，并在 S2–S4 上融合所得表征。

**Finance encoder.** The finance encoder receives the weekly financial block that constitutes S1 and is retained in S2–S4, including prices, inventories and macro-financial, oil-market and geopolitical indicators. A causal temporal convolutional network (TCN; Bai, Kolter and Koltun, 2018) maps the four-week input sequence to one finance representation for each forecast origin, using only the current and earlier positions at each convolutional layer. 
**金融编码器。** 金融编码器接收构成 S1 并保留在 S2–S4 中的周度金融块，包括价格、库存以及宏观金融、油市与地缘政治指标。因果时间卷积网络（TCN；Bai, Kolter and Koltun, 2018）将四周输入序列映射为每个预测起点的一个金融表征，各卷积层只使用当前位置及其之前的位置。TCN 此前已用于短期原油价格预测（Foroutan and Lahmiri, 2024）。

**The finance encoder applies a causal temporal convolutional network (TCN; Bai, Kolter and Koltun, 2018) to the four-week financial sequence retained across S1–S4. It produces one finance representation per forecast origin using only current and earlier positions at each convolutional layer.金融编码器将因果时间卷积网络（TCN；Bai, Kolter and Koltun, 2018）应用于 S1–S4 均保留的四周金融序列。各卷积层仅使用当前位置及其之前的位置，并为每个预测起点生成一个金融表征。**

**Remote-sensing encoder.** The remote-sensing encoder receives monthly embeddings for the 11 AOIs, extracted from Sentinel-2 image patches using a frozen Prithvi-EO-2.0-300M model. Although Prithvi-EO-2.0 was pretrained on six-band NASA HLS imagery at 30 m, the Deep pathway uses Sentinel-2 Surface Reflectance Harmonized patches rather than HLS directly. The patches are adapted to the model’s input convention using bands B2, B3, B4, B8A, B11 and B12, standardised with the published per-band statistics, and bilinearly resampled to 224 × 224 before embedding extraction. Temporal and site attention operate over the four-week lookback, keeping site-specific embeddings distinct until they are pooled into one remote-sensing representation for each forecast origin.

**遥感编码器。** 遥感编码器接收 11 个 AOI 的月度嵌入，由冻结的 Prithvi-EO-2.0-300M 模型从 Sentinel-2 影像块中提取。虽然 Prithvi-EO-2.0 以 30 m 六波段 NASA HLS 影像预训练，Deep 路径使用的是 Sentinel-2 地表反射率和谐化影像块，而非直接使用 HLS。影像块按模型输入约定选取 B2、B3、B4、B8A、B11 与 B12，用公布的逐波段统计量标准化，再双线性重采样至 224 × 224 后提取嵌入。时间与站点注意力作用于四周回看窗，使站点专属嵌入在池化为每个预测起点的单一遥感表征之前保持区分。

**The remote-sensing encoder receives monthly embeddings for the 11 AOIs, extracted from Sentinel-2 Surface Reflectance Harmonized patches using a frozen Prithvi-EO-2.0-300M encoder. The patches are adapted to the six-band convention of the HLS-pretrained encoder, with band mapping, standardisation and resampling documented in Appendix A. Temporal and site attention pool the site-specific embeddings over the four-week lookback into one representation per forecast origin.遥感编码器接收 11 个 AOI 的月度嵌入，这些嵌入由冻结的 Prithvi-EO-2.0-300M 编码器从 Sentinel-2 地表反射率和谐化影像块中提取。影像块按照该 HLS 预训练编码器的六波段约定进行适配，波段映射、标准化与重采样方法见附录 A。时间与站点注意力在四周回看窗口上汇总站点专属嵌入，为每个预测起点生成一个表征。**

**Shipping encoder.** The shipping encoder receives the weekly 17-node graph described in Section 3.3. A graph attention network with temporal encoding (GAT; Veličković et al., 2018) exchanges information across connected nodes over the four-week lookback and produces one shipping representation for each forecast origin. In the implementation, the two link classes described in the implementation, fixed structural links and dynamic voyage links are combined in a symmetrised weekly adjacency matrix with self-loops, so edge direction and type are not retained during message passing. Log-transformed voyage counts, scaled by a learned coefficient, are added to the attention logits so that busier lanes can receive greater weight.

**航运编码器。** 航运编码器接收第 3.3 节所述的周度 17 节点图。带时间编码的图注意力网络（GAT；Veličković et al., 2018）使相连节点在四周回看窗上交换信息，并为每个预测起点生成一个航运表征。实现上，实现中所述的两类边，即固定结构边与动态航次边，被合并到一个经对称化并加入自环的周度邻接矩阵中，因此消息传递不保留边的方向与类型。对数变换后的航次数经可学习系数缩放后加入注意力 logits，使较繁忙的航道可以获得更大权重。节点级注意力仅作 RQ3 的描述性检查，不解释为因果重要性。
**The shipping encoder applies a graph attention network with temporal encoding (GAT; Veličković et al., 2018) to the weekly 17-node graph over the four-week lookback, producing one shipping representation per forecast origin. Fixed structural and dynamic voyage links are combined without retaining edge direction or type during message passing, while voyage intensity informs attention weights. Adjacency and edge-weighting details are reported in Appendix A.4.3.航运编码器将带时间编码的图注意力网络（GAT；Veličković et al., 2018）应用于四周回看窗口内的周度 17 节点图，为每个预测起点生成一个航运表征。固定结构边与动态航次边在消息传递中合并，不保留边的方向与类型，但航次强度会影响注意力权重。邻接关系与边权实现细节见附录 A.4.3。**
### 3.5.3 Fusion mechanisms

### 3.5.3 融合机制
**Fusion is applied only to the multimodal information sets S2–S4, while S1 passes its single finance representation directly to the regression head. Three mechanisms are compared, with gated fusion as the main design and encoder concatenation and cross-attention as alternatives.

融合仅用于多模态信息集 S2–S4，S1 则将其单一金融表征直接传入回归头。比较三种机制，以门控融合为主要设计，编码器拼接与交叉注意力为备选。

Gated fusion and encoder concatenation operate on pooled 32-dimensional modality representations. Under gated fusion, a small multilayer perceptron maps the concatenated representations to one logit per available modality at each forecast origin. A softmax converts these logits into non-negative weights that sum to one, and the fused representation is the weighted sum of the modality representations. Encoder concatenation instead projects the concatenated representations back to 32 dimensions without assigning explicit modality weights. Cross-attention accesses pre-pooling remote-sensing site tokens, shipping node tokens or both, depending on the information set, with the finance representation as the query. It therefore does not produce modality gate weights. The fused representation is passed to a regression head trained using mean squared error on the one-week-ahead log return, after which the price forecast is reconstructed as for the Flat models. Fixed model settings, optimisation, early stopping and sensitivity configurations are reported in Appendix C.**

门控融合与编码器拼接作用于池化后的 32 维模态表征。门控融合中，一个小型多层感知机在每个预测起点将拼接后的表征映射为每个可用模态一个 logit。softmax 将这些 logit 转为非负且和为 1 的权重，融合表征为各模态表征的加权和。编码器拼接则将拼接后的表征映回 32 维，但不赋予显式模态权重。交叉注意力按信息集访问池化前的遥感站点 token、航运节点 token 或二者，并以金融表征作为 query，因此不产生模态门控权重。融合表征输入回归头，以提前一周对数收益的均方误差训练，再按与 Flat 模型相同的方式还原价格预测。固定模型设定、优化、早停与敏感性配置见附录 C。


**Fusion is applied only to S2–S4, while S1 passes its finance representation directly to the regression head. Three mechanisms are compared. Gated fusion, the main design, assigns non-negative modality weights that sum to one at each forecast origin. Encoder concatenation combines pooled modality representations without explicit weights, while cross-attention uses the finance representation to query pre-pooling remote-sensing site tokens, shipping node tokens or both and does not produce gate weights. The resulting representation is trained by mean squared error to predict the one-week-ahead log return, from which the price forecast is reconstructed. Fixed fusion settings are reported in Appendix C.4.2–C.4.3.融合仅用于 S2–S4，S1 的金融表征则直接进入回归头。研究比较三种机制。作为主要设计的门控融合在每个预测起点分配总和为 1 的非负模态权重。编码器拼接在不设置显式权重的情况下组合池化后的模态表征，交叉注意力则以金融表征查询池化前的遥感站点 token、航运节点 token 或两者，并且不产生门控权重。所得表征以均方误差训练，用于预测提前一周的对数收益，并据此还原价格预测。固定融合设置见附录 C.4.2–C.4.3。**
## 3.6 Estimation and validation

## 3.6 估计与验证

### 3.6.1 Expanding-window estimation and re-estimation

### 3.6.1 扩展窗估计与重估

With a four-week input window and a one-week forecast horizon, the 365 weekly observations yield 361 eligible input–target sequences. The first three observations cannot yet form a complete four-week input sequence, and the final week serves only as a target rather than a forecast origin. The first 104 eligible sequences form the initial training period and are not included in the evaluation metrics. The evaluation span covers 257 weeks from 22 January 2021 to 19 December 2025, with corresponding target dates from 29 January 2021 to 26 December 2025. Flat and Deep share this evaluation calendar.

在四周输入窗口与提前一周预测期下，365 个周度观测形成 361 个合格的输入–目标序列。最前面三个观测尚不足以构成完整的四周输入序列，最后一周只作为目标而不能作为预测起点。前 104 个合格序列构成初始训练期，不纳入评价指标。评价跨度为 2021 年 1 月 22 日至 2025 年 12 月 19 日的 257 周，相应目标日期为 2021 年 1 月 29 日至 2025 年 12 月 26 日。Flat 与 Deep 共用这一评价日历。

**With a four-week input window and a one-week forecast horizon, the 365 weekly observations yield 361 eligible sequences. The first 104 form the initial training period, while the remaining 257 constitute the common evaluation sample, with forecast origins from 22 January 2021 to 19 December 2025 and target dates from 29 January 2021 to 26 December 2025.在四周输入窗口与提前一周预测期下，365 个周度观测形成 361 个可用序列。前 104 个构成初始训练期，其余 257 个构成共同评价样本，预测起点为 2021 年 1 月 22 日至 2025 年 12 月 19 日，目标日期为 2021 年 1 月 29 日至 2025 年 12 月 26 日。**

A one-week-ahead forecast is produced at every origin t, using only information observable by that date. Each model specification is first estimated at the initial evaluation origin and re-estimated every 13 forecast origins as the training window expands. Between scheduled re-estimations the fitted model and preprocessing parameters are held fixed; each origin still receives a new as-of input window. The final evaluation block contains 10 origins rather than 13. For each model specification, this schedule produces 20 estimation blocks, with the first 19 covering 13 forecast origins each and the final block covering 10. Each estimation uses only input–target pairs whose targets are observable by that date. Newly realised targets enter the training sample only at subsequent re-estimations, and all data-dependent preprocessing is estimated from the corresponding training sample.

每个预测起点 t 都产生一次提前一周预测，且仅使用截至该日可观测的信息。每个模型设定先在最初评价起点估计，并随训练窗扩展每隔 13 个预测起点重估一次。两次预定重估之间，拟合模型与预处理参数保持固定；各起点仍使用新的 as-of 输入窗。最后一个评价块含 10 个起点，而非 13 个。对每个模型设定，该安排产生 20 个估计块：前 19 块各覆盖 13 个预测起点，最后一块覆盖 10 个。每次估计仅使用目标在该日已可观测的输入–目标对。新实现的目标只在之后的重估中进入训练样本，所有数据依赖型预处理均由相应训练样本估计。

**Each model specification is estimated at the first evaluation origin and re-estimated every 13 origins as the training window expands. Between fits, the model and preprocessing parameters remain fixed while the as-of input window advances. This yields 20 estimation blocks, with 19 covering 13 origins and the final block covering 10. Each fit uses only targets observable by that date, and all preprocessing is estimated from the same training sample.每个模型设定在首个评价起点进行估计，随后随训练窗口扩展每隔 13 个起点重估一次。两次拟合之间，模型与预处理参数保持固定，as-of 输入窗口继续向前推进。由此形成 20 个估计块，其中 19 个覆盖 13 个起点，最后一个覆盖 10 个。每次拟合仅使用当时已经可观测的目标，所有预处理也仅根据同一训练样本估计。**

Figure 3.2 presents the full schedule, while Figure 3.4 illustrates a single re-estimation origin.

图 3.2 给出完整安排，图 3.4 展示单个重估起点。


Figure 3.4

**Figure 3.4 — A re-estimation origin showing the training fold, inner-validation weeks, four-week input window and one-week-ahead target. Between re-estimations, only the as-of input window advances.**

**图 3.4 — 一次重估起点，展示训练折、内部验证周、四周输入窗与提前一周目标。两次重估之间，仅 as-of 输入窗口向前推进。**

### 3.6.2 Hyperparameter selection and fixed model settings

### 3.6.2 超参数选择与固定模型设定

Flat models re-select Ridge and XGBoost hyperparameters at each scheduled re-estimation on an inner time-ordered validation segment, then refit on the full estimation sample available at that date. Deep models instead use a configuration fixed before evaluation on design grounds, including the four-week lookback shared with Flat models and a common 32-dimensional latent size across encoders. At each Deep re-estimation, inner validation is used for early stopping, and the checkpoint with the lowest validation loss is retained for the subsequent forecast block. After checkpoint selection, Deep models are not refitted on the combined training and validation sample at that re-estimation. Sensitivity analyses using the evaluation sample are reported separately in Appendix B and are not used to select or revise the main specification. Search grids, fixed configurations and early-stopping settings are reported in Appendix C.

Flat 与 Deep 使用不同的超参数选择程序。Flat 模型在每次预定重估时，于内部时序验证段上重新选择 Ridge 与 XGBoost 超参数，再在该日可得的全部估计样本上重新拟合。Deep 模型则在评价前按设计原则固定配置，包括与 Flat 共用的四周回看窗，以及各编码器共用的 32 维潜在维度。每次 Deep 重估用内部验证做早停，并保留验证损失最低的 checkpoint 用于随后的预测块。选定 checkpoint 之后，Deep 模型不再将该次重估的训练与验证样本合并后重新拟合。利用评价样本所做的敏感性分析另行报告于附录 B，不用于选择或修改主设定。搜索网格、固定配置与早停设定见附录 C。

**At each scheduled re-estimation, Flat models select Ridge and XGBoost hyperparameters using an inner time-ordered validation segment and then refit on the full available estimation sample. Deep model settings are fixed before evaluation. Inner validation is used only for early stopping, and the selected checkpoint is retained for the subsequent forecast block without refitting on the combined training and validation sample. Evaluation-sample sensitivity analyses are reported in Appendix B, while search grids, fixed settings and early-stopping details are reported in Appendix C.在每次预定重估时，Flat 模型使用内部时序验证段选择 Ridge 与 XGBoost 超参数，再利用全部可得估计样本重新拟合。Deep 模型设定在评价前固定。内部验证仅用于早停，选定的 checkpoint 用于后续预测块，不再使用合并后的训练与验证样本重新拟合。基于评价样本的敏感性分析见附录 B，搜索网格、固定设定与早停细节见附录 C。**

## 3.7 Forecast evaluation and model interpretation

## 3.7 预测评估与模型解释

### 3.7.1 Error metrics and skill scores

### 3.7.1 误差指标与 skill 得分

The primary evaluation metrics are calculated from reconstructed price forecasts over the common sample of \(T=257\) forecast origins. For model \(m\),

主要评价指标根据共同样本中 \(T=257\) 个预测起点的重构价格预测计算。对于模型 \(m\)，

\[
\mathrm{RMSE}_m
=
\sqrt{
\frac{1}{T}
\sum_{t=1}^{T}
\left(P_{t+1}-\hat{P}_{m,t+1\mid t}\right)^2
}
\]

and

以及

\[
\mathrm{MAE}_m
=
\frac{1}{T}
\sum_{t=1}^{T}
\left|P_{t+1}-\hat{P}_{m,t+1\mid t}\right|.
\]

Performance relative to the no-change benchmark M0 is summarised by RMSE skill, expressed as a percentage.

相对不变预测基准 M0 的表现以 RMSE skill 汇总，并表示为百分比。

\[
\mathrm{Skill}_m
=
100\times
\left(
1-
\frac{\mathrm{RMSE}_m}
{\mathrm{RMSE}_{\mathrm{M0}}}
\right)
\]

Here, \(P_{t+1}\) is the observed price and \(\hat{P}_{m,t+1\mid t}\) is the price forecast produced by model \(m\) at origin \(t\). Positive skill indicates lower RMSE than M0, zero indicates equal RMSE, and negative skill indicates worse performance.

其中，\(P_{t+1}\) 为观测价格，\(\hat{P}_{m,t+1\mid t}\) 为模型 \(m\) 在预测起点 \(t\) 作出的价格预测。Skill 为正表示 RMSE 低于 M0，为零表示与 M0 相同，为负则表示表现更差。

### 3.7.2 Forecast-comparison tests

### 3.7.2 预测比较检验

Formal forecast comparisons use the Diebold–Mariano test (Diebold and Mariano, 1995) with the Harvey, Leybourne and Newbold (1997) finite-sample correction, applied to squared errors of reconstructed price forecasts. Test direction is determined before observing the results. Comparisons against M0, information-set extensions within the same learner and matched Deep–Flat comparisons are one-sided, while comparisons among Deep fusion mechanisms are two-sided.

正式预测比较采用带 Harvey、Leybourne 与 Newbold（1997）有限样本修正的 Diebold–Mariano 检验（Diebold and Mariano, 1995），并作用于重构价格预测的平方误差。检验方向在观察结果之前确定。各模型与 M0 的比较、同一学习器内的信息集扩展比较以及匹配的 Deep–Flat 比较采用单侧检验，Deep 融合机制之间的比较则采用双侧检验。

Three comparison families are defined before testing, comprising 18 benchmark comparisons, 15 RQ1 comparisons and 14 RQ2 comparisons. Holm’s (1979) adjustment is applied separately within each family. Formal inference uses Holm-adjusted p-values, while raw p-values are reported as nominal evidence. Exact family membership, supplementary Clark–West tests, variance estimation and exploratory sensitivity analyses are reported in Appendix B, with implementation details in Appendix C.

检验前预先定义三个比较族，分别包含 18 项基准比较、15 项 RQ1 比较和 14 项 RQ2 比较。Holm（1979）调整分别在各族内实施。正式推断依据 Holm 调整后的 p 值，原始 p 值仅作为名义证据报告。各比较族的具体构成、补充 Clark–West 检验、方差估计与探索性敏感性分析见附录 B，实现细节见附录 C。

### 3.7.3 Model interpretation

### 3.7.3 模型解释

RQ3 diagnostics are restricted to Deep specifications with positive mean out-of-sample RMSE skill across three random seeds. This is a descriptive eligibility criterion and does not imply statistically significant superiority to M0. Gated models provide modality weights and encoder-level site or node attention, cross-attention models provide token-level attention, and concatenation models retain only the applicable encoder-level diagnostics. Patterns are interpreted only when they remain stable across seeds, while unstable or approximately uniform attention is not given substantive interpretation. These diagnostics describe internal model weighting rather than causal importance.

RQ3 诊断仅用于三个随机种子上平均样本外 RMSE skill 为正的 Deep 设定。这是描述性准入标准，并不表示在统计上显著优于 M0。门控模型提供模态权重以及编码器层面的站点或节点注意力，交叉注意力模型提供 token 层面的注意力，拼接模型则仅保留适用的编码器层面诊断。只有跨种子保持稳定的模式才加以解释，不稳定或近似均匀的注意力不作实质性解读。这些诊断描述模型内部加权，而非因果重要性。

## 3.8 Ethical considerations and reproducibility

## 3.8 伦理考量与可复现性

The study uses only secondary, aggregate data and does not involve human participants. It received approval through UCL’s low-risk ethics process. All datasets were used in accordance with their published licences and terms of use, including the Copernicus open licence for Sentinel-2, the open distribution terms for VIIRS night-time lights, and the research-use terms of IMF PortWatch and Global Fishing Watch. Remote-sensing and vessel-activity variables are analysed only at the aggregate site or chokepoint level; no attempt is made to identify individual vessels, operators or persons.

本研究仅使用二手、汇总型数据，不涉及人类参与者。研究已通过 UCL 低风险伦理审批。所有数据集均按其公布的许可与使用条款使用，包括 Sentinel-2 的 Copernicus 开放许可、VIIRS 夜光的开放分发条款，以及 IMF PortWatch 与 Global Fishing Watch 的研究使用条款。遥感与船舶活动变量仅在站点或咽喉的汇总层面分析；不试图识别单船、运营商或个人。

Analysis was conducted in Python, and the code required to reproduce the analysis is available on GitHub: [repository link]. Package versions, configuration settings and random-seed specifications are provided in Appendix C.

分析在 Python 中完成，复现所需代码见 GitHub：[repository link]。软件包版本、配置设定与随机种子说明见附录 C。

---
