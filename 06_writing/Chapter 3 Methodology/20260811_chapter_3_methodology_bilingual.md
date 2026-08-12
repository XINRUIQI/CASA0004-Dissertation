# Chapter 3 — Methodology *(~3,200)*

## 3.1 Research design

## 3.1 研究设计

This chapter sets out how the study answers the research questions in Section 1.2. Every learned forecast is judged against a simple no-change benchmark in which next week’s Brent price equals this week’s price. The study then asks whether remote sensing and shipping add useful information beyond financial time series, and whether modelling those inputs as one weekly table differs from encoding each data type separately before combining them. For models that improve on the benchmark, it also asks what they rely on. All specifications share the same weekly forecast dates, sample window and evaluation rules, so that differences in the information set can be separated from differences in modelling strategy.

本章说明研究如何回答第 1.2 节的研究问题。每个学习到的预测都对照一个简单的不变预测基准，即下周 Brent 价格等于本周价格。研究再问遥感与航运是否在金融时序之外仍提供有用信息，以及把这些输入压成一张周表建模是否不同于先按数据类型分别编码再组合。对优于基准的模型，研究还问它依赖哪些信息。所有设定共用同一周度预测日、样本窗口与评估规则，从而把“用了什么数据”的差异，与“如何建模”的差异分开。

The no-change benchmark is denoted M0. At each forecast origin t, where P_t is the Brent price in week t, M0 sets the one-week-ahead price forecast equal to the current weekly price:

\hat{P}_{t+1|t}=P_t.

M0 needs no parameter estimation and contains no predictors. It is a reference forecast rather than one of the learned specifications below, and every learned model is compared with it over the same out-of-sample evaluation period.

不变预测基准记为 M0。设 P_t 为第 t 周的 Brent 价格，则在每个预测起点 t，M0 将提前一周的价格预测设为当前周价格：

\hat{P}_{t+1|t}=P_t.

M0 无需参数估计，也不含预测变量。它是参照预测，不属于下文经过学习的设定；每个学习模型都在同一样本外评价期上与 M0 比较。

The predictors are organised into four information sets. S1 contains financial time series only, comprising financial, macroeconomic and oil-market variables. S2 adds remote sensing to S1, S3 adds shipping to S1, and S4 adds both modalities. S2 and S3 are parallel extensions of S1 rather than successive stages, while S4 combines the two. Table 3.1 lists the four sets together with the M0 benchmark.

预测变量组织为四个信息集。S1 仅用金融时序（金融、宏观与油市序列）。S2 在 S1 上加遥感；S3 在 S1 上加航运；S4 两者都加。S2 与 S3 是对 S1 的平行扩展，不是一条梯子上的先后步骤；S4 合并两支扩展。表 3.1 将四个集合与 M0 基准一并列出。

**Table 3.1 — Information sets**

**表 3.1 — 信息集**


| Set            | Variables                                                                   |
| -------------- | --------------------------------------------------------------------------- |
| Benchmark (M0) | Last week's price                                                           |
| S1             | Financial time series only (financial, macroeconomic and oil-market series) |
| S2             | S1 + remote sensing                                                         |
| S3             | S1 + shipping                                                               |
| S4             | S1 + remote sensing + shipping                                              |


Within each model family, the information-set contrasts hold the forecasting method and evaluation sample constant and change only the information set. S2 against S1 measures the contribution of remote sensing added on its own, and S3 against S1 that of shipping. S4 against S1 measures their joint contribution. S4 against S3 and S4 against S2 ask whether each source still helps once the other is already included. These contrasts, together with each model’s comparison against M0, address RQ1.

所有对比都保持模型不变，只改变信息集。S2 对 S1 度量单独加入遥感的贡献，S3 对 S1 度量航运的贡献。S4 对 S1 度量二者的联合贡献。S4 对 S3 与 S4 对 S2 则问：在另一数据源已经纳入后，这一源是否仍有帮助。这些对比连同每个模型与 M0 的比较，共同回答 RQ1。

Two model families are applied to these information sets. The Flat family puts all selected predictors into one weekly table—stacking recent weeks into a single row—and fits Ridge and XGBoost. This early joining of features is called flat feature fusion. The Deep family keeps each data type separate at first. Financial series, remote-sensing imagery and shipping-network inputs each pass through their own encoder, and the outputs are then combined. The main Deep design, gated fusion, learns how much weight to give each data type. Flat and Deep are then compared on the same information set and the same evaluation sample. These comparisons measure the overall difference between the two modelling strategies. They do not isolate the effect of fusion alone, because the two families also differ in model class, capacity and some modality-specific input products and representations (Table 3.2). Fusion is assessed more directly within the Deep family. Simple concatenation, gated fusion and cross-attention are compared with the encoders and inputs held fixed. Together these two comparisons address RQ2.

两套模型族应用于这些信息集。Flat 族把所选预测变量压成一张周度表——把最近几周叠成一行——再拟合 Ridge 与 XGBoost。这种一开始就合并特征的做法，称为扁平特征融合。Deep 族则先按数据类型分开处理。金融序列、遥感影像与航运网络输入各自经过自己的编码器，再把输出组合起来。Deep 的主设计学习给各类数据多少权重（门控融合）。随后在相同信息集与相同评价样本上比较 Flat 与 Deep。这类比较衡量的是两种整体建模策略的差异。它不能单独分离出融合方式的作用，因为两族在模型类型、容量以及部分模态专属输入产品与表征上同样存在差异（见表 3.2）。融合方式本身在 Deep 族内部得到更直接的评估。在编码器与输入保持不变的前提下，比较简单拼接、门控融合与交叉注意力。这两组比较共同回答 RQ2。

RQ3 is restricted to Deep models that improve on M0 according to the 
predefined criterion**defined in Section 3.7**. For these specifications the study reports the weights assigned to finance, remote sensing and shipping, together with the sites or network nodes receiving greater attention under different market conditions. These quantities indicate what a model relies on. They are not interpreted as evidence of causal importance.

RQ3 仅限于在样本外评价期上优于 M0 的 Deep 设定，判定准则见第 3.7 节。对这些设定，本研究报告其赋予金融、遥感与航运输入的权重，以及不同市场条件下受到更多关注的站点或网络节点。这些量说明模型依赖哪些信息，但不被解释为因果重要性的证据。

Figure 3.1 summarises the research design.

图 3.1 概括研究设计。

Figure 3.1

**Figure 3.1 — Research design: data blocks, the M0 benchmark and information sets S1–S4, the Flat and Deep families, and the shared expanding-window evaluation.**

**图 3.1 — 研究设计：数据块、M0 基准与信息集 S1–S4、Flat 与 Deep 两族，以及共用的扩展窗评估。**

## 3.2 Prediction target and timeline

## 3.2 预测目标与时间轴

Let P_t denote the last available daily Brent spot-price observation in week t, where each week ends on Friday, measured in US dollars per barrel. The forecast target is next week’s price P_{t+1}. Models are not trained directly on the price level. They predict the one-week logarithmic return

r_{t+1}=\log\left(\frac{P_{t+1}}{P_t}\right)

and reconstruct the price forecast as

\hat{P}*{t+1|t}=P_t\exp\left(\hat{r}*{t+1|t}\right).

Log returns are used to reduce the strong persistence in the price level and to express the forecasting task in terms of proportional weekly changes. RMSE, MAE and skill versus M0 are computed from the reconstructed price forecasts. Under this mapping, the no-change benchmark \hat{P}*{t+1|t}=P_t is exactly the same as forecasting a zero return \hat{r}*{t+1|t}=0.

令 P_t 表示第 t 个周五截止周内最后一个可获得的 Brent 现货价格日度观测值，单位为美元/桶。预测目标是下一周的价格 P_{t+1}。模型不直接在价格水平上训练，而是预测一周对数收益

r_{t+1}=\log\left(\frac{P_{t+1}}{P_t}\right),

并按

\hat{P}*{t+1|t}=P_t\exp\left(\hat{r}*{t+1|t}\right)

重构价格预测。使用对数收益是为了减弱价格水平序列的强持续性，并将预测任务表示为周度比例变化。RMSE、MAE 以及相对 M0 的 skill 均根据重构后的价格预测计算。在此对应关系下，不变预测基准 \hat{P}*{t+1|t}=P_t 与预测收益为零 \hat{r}*{t+1|t}=0 完全一致。

 The modelling window covers 2019–2025 and provides a common weekly index of 365 observations (4 January 2019 to 26 December 2025). The training and evaluation samples are separated in time on an expanding window, rather than by random assignment, to prevent future information from leaking into model fitting. The full validation protocol is in Section 3.6.

建模窗口覆盖 2019–2025 年，提供含 365 个观测的共同周索引（2019 年 1 月 4 日至 2025 年 12 月 26 日）。训练样本与评价样本在扩展窗口下按时间分开，而非随机划分，以避免未来信息泄漏到模型拟合中。完整验证协议见第 3.6 节。

## 3.3 Geographic scope and monitoring sites

## 3.3 地理范围与监测站点

Because the prediction target is the global Brent benchmark rather than a local physical cargo price at a single terminal, the study does not use one contiguous study region. Spatial information instead comes from eleven oil-infrastructure monitoring sites and six maritime chokepoints. Together they cover major supply, transit, refining and demand locations in the international oil system. Figure 3.3 places these sites and chokepoints on a world map. Full site names, coordinates, patch sizes and graph edge definitions are in Appendix A.

由于预测对象是全球 Brent 基准价格，而非单一码头的现货成交价格，本研究不采用一块连续的地理研究区。空间信息来自十一个石油基础设施监测站点与六个航运咽喉，共同覆盖国际石油体系中的主要供给、中转、炼化与需求区位。图 3.3 在世界地图上标出这些站点与咽喉。完整站名、坐标、裁剪范围与图边定义见附录 A。

The eleven sites are ports, refineries and export terminals, purposively selected for strategic coverage of supply, transit, refining and demand locations and for observability in the available satellite products. In the Flat pathway, remote-sensing features are summarised inside a circular buffer with a radius of 5 km around each site. In the Deep pathway, image patches are cut around each site, and patch size follows facility type and local spatial constraints: generally larger for ports, intermediate for refineries and smaller for terminals. The two pathways therefore observe the same locations over different spatial extents.

十一个站点为港口、炼厂与出口码头，按目的性原则选取，兼顾供给、中转、炼化与需求区位的战略覆盖，以及在可用卫星产品中的可观测性。Flat 路径在以站点为中心、半径 5 km 的圆形缓冲区内汇总遥感特征；Deep 路径按站点裁剪影像块，裁剪大小随设施类型与当地空间条件而定：一般港口较大、炼厂居中、码头较小。因此两条路径观测同一批地点，但空间范围不同。

The shipping graph augments the eleven sites with six maritime chokepoints: the Strait of Hormuz, the Suez Canal, the Strait of Malacca, Bab el-Mandeb, the Panama Canal and the Cape of Good Hope.[^cape] The resulting weekly graph contains seventeen nodes and two forms of connection. Dynamic links between the eleven AOIs are directed origin–destination pairs, weighted by the number of voyages counted in each week from Global Fishing Watch port-visit sequences, and therefore change from week to week. Fixed links are undirected. They connect each site to the chokepoint or chokepoints on its documented principal oil-trade corridor and are specified ex ante rather than inferred from weekly vessel movements or geographic proximity. Complete edge definitions are reported in Appendix A.4, and graph encoding is described in Section 3.5.2.

航运图在十一个站点之外加入六个航运咽喉：霍尔木兹海峡、苏伊士运河、马六甲海峡、曼德海峡、巴拿马运河与好望角。[^cape] 由此形成包含十七个节点、两类连接的周度图。十一个 AOI 之间的动态边为有向的起点–终点对，权重取各周内由 Global Fishing Watch 港口访问序列统计到的航次数，因而随周变化。固定边为无向，将每个站点与其主要石油贸易航路上的一个或多个咽喉相连，依据既有航运路线事先指定，而非由周度船舶活动或地理邻近推断。完整边定义见附录 A.4，图编码方法见第 3.5.2 节。

Figure 3.3

**Figure 3.3 — Spatial coverage of the study: 11 oil-infrastructure AOIs, 6 maritime chokepoints and the fixed AOI–chokepoint corridor links used in the shipping graph.**

**图 3.3 — 研究的空间覆盖：11 个石油基础设施 AOI、6 个航运咽喉，以及航运图中使用的固定 AOI–咽喉走廊连接。**

[^cape]: The Cape of Good Hope is included as a major oil-trade route rather than a narrow chokepoint in the sense of the EIA World Oil Transit Chokepoints report. / 好望角按主要石油贸易航路纳入；在 EIA《世界石油运输咽喉》口径下，它并非狭义的咽喉。

## 3.4 Data sources and preparation

## 3.4 数据来源与准备

### 3.4.1 Data sources

### 3.4.1 数据来源

Three data blocks enter the design: financial time series, satellite remote sensing and maritime shipping. Flat and Deep share the same Friday-ending calendar and monitoring geography, although the specific products and representations used in each pathway may differ after cleaning and alignment (Table 3.2). Flat models use predictors assembled in a merged weekly feature table, whereas Deep models retain modality-specific sequence, image-embedding and graph inputs.

三类数据块进入研究设计，分别为金融时序、卫星遥感与航运。Flat 与 Deep 使用相同的周五截止日历和监测地理范围，但各块经清洗与对齐后，两条路径使用的具体产品与表征可能不同（见表 3.2）。Flat 模型使用合并周度特征表中的预测变量，Deep 模型则保留模态专属的序列、影像嵌入与图输入。

**Table 3.2 — Datasets, variables and sources**

**表 3.2 — 数据集、变量与来源**


| Modality              | Dataset / product                                                            | Key variables                                                               | Source                                                                                                                                                                                                                                                |
| --------------------- | ---------------------------------------------------------------------------- | --------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Financial time series | Oil-market and macro-financial series at daily, weekly and monthly native frequencies | Prices, inventories, production, interest rates, GPR and related indicators | [EIA](https://www.eia.gov/petroleum/supply/weekly/); [FRED](https://fred.stlouisfed.org/); [Yahoo Finance](https://finance.yahoo.com/); [Dallas Fed IGREA](https://www.dallasfed.org/research/igrea); [GPR](https://www.matteoiacoviello.com/gpr.htm) |
| Remote sensing (Flat) | Sentinel-2 optical indices and VIIRS night-time lights                       | Site-level anomalies at 11 AOIs (NDVI, NDWI, NDBI, BSI; NTL)                | [Sentinel-2 via GEE](https://developers.google.com/earth-engine/datasets/catalog/COPERNICUS_S2_SR_HARMONIZED); [VIIRS via GEE](https://developers.google.com/earth-engine/datasets/catalog/NOAA_VIIRS_DNB_MONTHLY_V1_VCMSLCFG)                        |
| Remote sensing (Deep) | Frozen Prithvi-EO-2.0 embeddings                                             | Monthly Sentinel-2 image-patch embeddings at the same 11 AOIs               | [Prithvi-EO-2.0](https://huggingface.co/ibm-nasa-geospatial); [Sentinel-2 via GEE](https://developers.google.com/earth-engine/datasets/catalog/COPERNICUS_S2_SR_HARMONIZED)                                                                           |
| Shipping (Flat)       | PortWatch and Global Fishing Watch tabular features                          | Port and chokepoint tanker flows; vessel-activity features                  | [IMF PortWatch](https://portwatch.imf.org/) (AIS-derived); [Global Fishing Watch](https://globalfishingwatch.org/our-apis/) (AIS-derived)                                                                                                            |
| Shipping (Deep)       | PortWatch and Global Fishing Watch network inputs                            | Weekly 17-node graph comprising 11 AOIs and 6 chokepoints                   | [IMF PortWatch](https://portwatch.imf.org/) (AIS-derived); [Global Fishing Watch](https://globalfishingwatch.org/our-apis/) (AIS- and SAR-derived)                                                                                                    |



| 模态       | 数据集 / 产品                                              | 关键变量                                       | 来源                                                                                                                                                                                                                                                |
| -------- | ----------------------------------------------------- | ------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 金融时序     | 日、周、月原生频率的油市与宏观金融序列                                   | 价格、库存、产量、利率、GPR 及相关指标                      | [EIA](https://www.eia.gov/petroleum/supply/weekly/)；[FRED](https://fred.stlouisfed.org/)；[Yahoo Finance](https://finance.yahoo.com/)；[Dallas Fed IGREA](https://www.dallasfed.org/research/igrea)；[GPR](https://www.matteoiacoviello.com/gpr.htm) |
| 遥感（Flat） | Sentinel-2 光学指数与 VIIRS 夜光                             | 11 个 AOI 的站点级异常（NDVI、NDWI、NDBI、BSI；NTL）    | [Sentinel-2 via GEE](https://developers.google.com/earth-engine/datasets/catalog/COPERNICUS_S2_SR_HARMONIZED)；[VIIRS via GEE](https://developers.google.com/earth-engine/datasets/catalog/NOAA_VIIRS_DNB_MONTHLY_V1_VCMSLCFG)                     |
| 遥感（Deep） | 冻结 Prithvi-EO-2.0 嵌入                                  | 同一 11 个 AOI 的月度 Sentinel-2 影像块嵌入           | [Prithvi-EO-2.0](https://huggingface.co/ibm-nasa-geospatial) / [Sentinel-2 via GEE](https://developers.google.com/earth-engine/datasets/catalog/COPERNICUS_S2_SR_HARMONIZED)                                                                      |
| 航运（Flat） | PortWatch 与 Global Fishing Watch 表格特征                 | 港口与咽喉油轮流量；船舶活动特征                           | [IMF PortWatch](https://portwatch.imf.org/)（AIS 衍生）；[Global Fishing Watch](https://globalfishingwatch.org/our-apis/)（AIS 衍生）                                                                                                                      |
| 航运（Deep） | PortWatch 与 Global Fishing Watch 网络输入                 | 包含 11 个 AOI 与 6 个咽喉的 17 节点周度图               | [IMF PortWatch](https://portwatch.imf.org/)（AIS 衍生）；[Global Fishing Watch](https://globalfishingwatch.org/our-apis/)（AIS 与 SAR 衍生）                                                                                                                |


The financial block combines oil-market and macro-financial series from the US Energy Information Administration (EIA), Federal Reserve Economic Data (FRED), Yahoo Finance, the Dallas Fed Index of Global Real Economic Activity, and the Caldara–Iacoviello geopolitical risk (GPR) index. These series are observed at different native frequencies and include crude prices and spreads, inventories, production and refinery activity, volatility and risk measures, interest rates, exchange rates, futures-based oil indicators and geopolitical risk. This block is S1 before remote sensing or shipping is added. In the Deep pathway it is treated as one input stream for the finance encoder, even though the predictors extend beyond prices alone.

金融块汇总来自美国能源信息署（EIA）、联邦储备经济数据（FRED）、Yahoo Finance、达拉斯联储全球实际经济活动指数和 Caldara–Iacoviello 地缘政治风险（GPR）指数的油市与宏观金融序列。这些序列具有不同的原生频率，包括原油价格与价差、库存、产量与炼厂活动、波动与风险度量、利率、汇率、基于期货的油市指标以及地缘政治风险。该块即加入遥感或航运前的 S1。在 Deep 路径中，它作为金融编码器的一路输入，尽管预测变量远不止价格本身。

Remote-sensing inputs are observed over the eleven AOIs. Flat and Deep share these sites but use different products from a common Sentinel-2 optical source family. Flat remote sensing uses monthly Sentinel-2 optical indices (NDVI, NDWI, NDBI and BSI) together with VIIRS night-time lights, converted to site-level anomalies, while Deep remote sensing uses frozen Prithvi-EO-2.0 embeddings extracted from monthly Sentinel-2 image patches at the same AOIs and does not include a separate VIIRS stream. The AOI locations are shared across pathways, while the geographic area included in the model, remote-sensing products and representations differ.

遥感输入观测于十一个 AOI。Flat 与 Deep 共享这些站点，但使用来自共同 Sentinel-2 光学源族的不同产品。Flat 遥感采用月度 Sentinel-2 光学指数（NDVI、NDWI、NDBI 与 BSI）及 VIIRS 夜光，并转换为站点级异常；Deep 遥感采用同一站点上月度 Sentinel-2 影像块提取的冻结 Prithvi-EO-2.0 嵌入，且不含单独的 VIIRS 夜光输入流。两条路径共享相同的 AOI 位置，但纳入模型的地理面积、遥感产品与表征不同。

Shipping inputs describe the eleven-site and six-chokepoint network. IMF PortWatch supplies AIS-derived chokepoint and port tanker flows. Global Fishing Watch supplies AIS-derived vessel-presence, activity-duration and port-visit indicators. For the Deep pathway only, GFW SAR dark-vessel detections also enter as node attributes on the seventeen-node graph. GFW port-visit sequences are additionally used to construct the dynamic AOI–AOI voyage links. In the Flat pathway, PortWatch and GFW AIS-derived series enter as weekly table features; SAR dark-vessel variables are not included in the Flat main feature set. In both pathways, these variables are treated as proxies for physical shipping activity, tanker movements and congestion rather than as a direct measure of next week’s price.

航运输入描述由十一个站点与六个咽喉组成的网络。IMF PortWatch 提供基于 AIS 的咽喉与港口油轮流量。Global Fishing Watch 提供基于 AIS 的船舶存在、活动时长与港口访问指标。仅在 Deep 路径中，GFW 的 SAR 暗船检测还作为十七节点图的节点属性进入。GFW 港口访问序列还用于构造 AOI 之间的动态航次边。Flat 路径中，PortWatch 与 GFW 的 AIS 衍生序列以周度表格特征进入；Flat 主特征集不含 SAR 暗船变量。两条路径都将这些变量视为实物航运活动、油轮移动与拥堵的代理，而非下周价格的直接量测。

### 3.4.2 Temporal alignment and publication lags

### 3.4.2 时间对齐与发布滞后

All series are mapped onto the common Friday-ending weekly calendar. Native frequencies are converted before modelling. Daily market series such as Brent, equity and volatility indices enter as the Friday last observation; daily GPR is averaged within each Friday-ending week. Daily PortWatch tanker counts and capacities are summed over the week. Monthly series—including macro indicators and GFW vessel-presence—are carried forward from month-end to the weekly dates that follow, subject to the lag buffers below. Monthly remote-sensing products (Flat anomalies and Deep Prithvi embeddings) are attached by as-of alignment: each Friday uses the most recent monthly composite whose conservative availability date has already passed, rather than a constant within-month fill that conceals staleness.

Predictors enter according to source-specific availability rules and conservative publication buffers implemented as fixed lags in the builders, not by tracking each series’ official release timestamp. The design therefore blocks look-ahead from future weeks. It does not reconstruct vintage data releases: downloaded series are the currently available revised histories rather than ALFRED-style real-time vintages, so revision effects are not removed. Illustrative buffers are one week for EIA fundamentals and PortWatch flows; two weeks for GFW port-visit and voyage streams used in the Deep graph; four weeks for GFW monthly vessel-presence and for SAR dark-vessel features; about five weeks for monthly macro series; and month-end plus fifteen days for remote-sensing composites. These GFW products are lagged separately and must not be treated as one common GFW delay. The full lag table, with constants and scripts, is in Appendix A.3.

全部序列映射到共同的周五截止周历。原生频率在建模前先转换：Brent、股市与波动率等日度市场序列取周五最后一个观测；日度 GPR 在周五截止周内取均值；PortWatch 日度油轮计数与运力在周内求和。月度序列——包括宏观指标与 GFW 船舶存在量——自月末向前填到随后的周日期，并施加下文滞后缓冲。月度遥感产品（Flat 距平与 Deep Prithvi 嵌入）采用 as-of 对齐：每个周五使用保守可得日已过的最近一个月度合成，而不是在月内做掩盖陈旧程度的常数值填充。

预测变量按来源专属的可得规则与保守发布缓冲进入；缓冲在构建脚本中以固定滞后实现，而非逐条追踪官方发布时间戳。该设计阻断来自未来周的前瞻。它并不重建实时历史版本：所用下载为当前可得的修订后历史，而非 ALFRED 式实时 vintage，因此未消除修订效应。示意缓冲为：EIA 基本面与 PortWatch 流量约一周；Deep 图所用 GFW 港口访问与航次流约两周；GFW 月度船舶存在量与 SAR 暗船特征约四周；月度宏观约五周；遥感合成取月末加十五天。上述 GFW 产品分别设滞后，不得当作同一 GFW 延迟。完整滞后表（含常数与脚本）见附录 A.3。

### 3.4.3 Missing data and quality control

### 3.4.3 缺失数据与质量控制

Optical availability varies across sites and months. In Google Earth Engine compositing, cloud probability and valid-observation counts are used to filter scenes and pixels when forming monthly medians; they are not entered as predictors. This keeps cloud-quality metadata out of the feature set, but it does not imply that models cannot learn from missingness patterns where those patterns remain visible through masks or filled gaps.

Coverage is reported for the Flat Sentinel-2 anomaly series after monthly compositing and anomaly construction. Appendix A gives site-level non-missing rates on the weekly calendar after as-of alignment. Because one eligible monthly anomaly can remain the most recent available value across several Fridays, that rate is a weekly-calendar availability measure rather than a count of independent site–month composites. On that measure, eight of the eleven sites are fully observed for the four optical indices, the cross-site mean is about 97 per cent, and VIIRS night-time-light anomalies are fully observed.

Missingness is then handled differently in the two modelling pathways. In Flat models, numeric predictors are forward-filled using only past values on the weekly calendar; any residual leading gaps after that fill are set to zero so that every specification shares the same evaluation weeks. In Deep models, finance inputs receive the same past-only forward fill and leading-zero treatment. Remote-sensing embeddings that are unavailable at a site–week remain missing and are paired with a binary availability mask; temporal and site attention are computed only over unmasked positions, while scaled missing embeddings are set to zero for numerical stability. Shipping-graph tensors with residual gaps are scaled on the training window and filled with zeros; unlike remote sensing, they do not carry a separate node-level missing mask into the encoder.

光学数据在不同站点与月份的可得性并不相同。在 Google Earth Engine 合成中，云概率与有效观测计数用于筛选场景与像元以形成月度中位数合成，不作为预测变量进入模型。这样把云质量元数据排除在特征集之外，但并不意味着模型无法从掩码或填补后仍可见的缺失模式中学习。

覆盖率针对月度合成与距平构建之后的 Flat Sentinel-2 距平序列报告。附录 A 给出 as-of 对齐到周历后的站点级非缺失率。由于一个已可得的月度距平可在多个周五上仍是最近可用值，该比率是周历可得性度量，而非独立的站点–月份合成计数。按该度量，四个光学指数在十一个站点中有八个满观测，跨站均值约 97%，VIIRS 夜光距平为满观测。

两条建模路径随后以不同方式处理缺失。Flat 模型对数值预测变量仅用周历上的过去值做向前填充；填充后仍残留的前导缺口置零，以使各设定共享相同评价周。Deep 模型中，金融输入采用相同的过去向向前填充与前导置零。某站点–周不可得的遥感嵌入保持缺失，并配以二元可用掩码；时间与站点注意力仅在未掩码位置上计算，缩放后的缺失嵌入为数值稳定置零。航运图张量中的残余缺口在训练窗上缩放后以零填充；与遥感不同，它们不向编码器传入单独的节点级缺失掩码。

## 3.5 Forecasting models

## 3.5 预测模型

### 3.5.1 Flat models

### 3.5.1 Flat 模型

Flat models implement flat feature fusion. For a given information set, all available numeric features are concatenated into one weekly table, and the most recent four weeks are flattened into a single row for each forecast origin. Two learners are estimated on this table. Ridge is a linear model with L2 regularisation (Hoerl and Kennard, 1970) and serves as a transparent linear baseline that combines features at the outset. XGBoost is a non-linear gradient-boosted tree ensemble (Chen and Guestrin, 2016) that can capture interactions missed by Ridge, but still does not preserve modality-specific structure. Regularised linear and tree-based learners are both common in short-horizon oil-price forecasting with large predictor sets (Costa et al., 2021; Yılmaz and Zehir, 2026); they are used here as Flat baselines rather than as a claim that either algorithm is universally optimal. Both models predict the one-week-ahead log return and then reconstruct price. Hyperparameters are chosen inside each training fold on past validation weeks only. Exact search grids are in Appendix C.

Flat 模型实现扁平特征融合。对给定信息集，将全部可用数值特征拼成一张周表，并在每个预测起点将最近四周压成一行。该表上估计两种学习器。Ridge 是带 L2 正则的线性模型（Hoerl and Kennard, 1970），作为一开始就合并特征的透明线性基线；XGBoost 是非线性梯度提升树集成（Chen and Guestrin, 2016），可捕捉 Ridge 错过的交互，但仍不保留各模态特有结构。正则化线性与树模型在大预测变量集的短期限油价预测中均常见（Costa et al., 2021; Yılmaz and Zehir, 2026）；此处用作 Flat 基线，而非声称任一算法普遍最优。二者均预测提前一周的对数收益，再还原价格。超参数仅在各训练折内、用过去验证周选择。精确搜索网格见附录 C。

### 3.5.2 Deep models

### 3.5.2 Deep 模型

Deep models use the same information sets, Friday calendar and validation protocol as Flat. The difference is how inputs are represented and combined, not the forecast target. Each available modality is first turned into a fixed-size representation; those representations are then combined into one forecast. The three encoders are described below by input, purpose and output.

Deep 模型与 Flat 使用相同的信息集、周五日历与验证协议。差异在输入如何被表征与组合，而非预测目标。每个可用模态先转为固定维度的表征，再把这些表征组合成一次预测。下文三个编码器按输入、用途与输出说明。

**Finance encoder.** The input is the weekly financial time series block (S1), including prices, inventories, macro and oil-market indicators. These series are dense temporal sequences, so the encoder must learn short-run dependence without using future weeks. The output is one finance representation for the forecast origin. The architecture is a causal temporal convolutional network (TCN; Bai, Kolter and Koltun, 2018). Causal convolutions prevent look-ahead within the sequence, and TCNs have been competitive for short-horizon crude-price forecasting relative to several deep and tree baselines (Foroutan and Lahmiri, 2024).

**金融编码器。** 输入为周度金融时序块（S1），包括价格、库存、宏观与油市指标。这些序列是密集时间序列，编码器须在不使用未来周的前提下学习短期依赖。输出为该预测时点的一个金融表征。架构为因果时间卷积网络（TCN；Bai, Kolter and Koltun, 2018）。因果卷积避免序列内前瞻，且相对多种深度与树基线，TCN 在短期限原油价格预测中具有竞争力（Foroutan and Lahmiri, 2024）。

**Remote-sensing encoder.** The input is monthly Sentinel-2 image-patch embeddings at the eleven AOIs, extracted with a frozen Prithvi-EO-2.0 model. VIIRS night-time lights are excluded because the frozen encoder is applied to Sentinel-2 optical patches rather than to a separate night-lights stream. Sites are kept distinct until after encoding, so spatial location is not collapsed into a single early average. The output is one remote-sensing representation for the forecast origin, formed by weighting across time and sites. The architecture uses frozen embeddings plus temporal and site attention.

**遥感编码器。** 输入为十一个 AOI 上的月度 Sentinel-2 影像块嵌入，由冻结的 Prithvi-EO-2.0 模型提取。不含 VIIRS 夜光，是因为冻结编码器作用于 Sentinel-2 光学影像块，而非单独的夜光输入流。编码完成前保持站点可区分，避免过早把空间位置压成单一均值。输出为该预测时点的一个遥感表征，由时间与站点加权得到。架构为冻结嵌入，外加时间与站点注意力。

**Shipping encoder.** The input is the weekly seventeen-node shipping network from Section 3.3. Shipping information is relational because ports and corridors are linked, so a graph model fits better than a flat row of counts. The output is one shipping representation for the forecast origin. The architecture is a graph attention network (GAT; Veličković et al., 2018) with temporal encoding. Graph neural networks have been used to model crude-oil and vessel-traffic networks as relational, time-varying processes (Ouyang et al., 2022; Liang et al., 2022). GAT is used here because neighbour weights fit a sparse port–chokepoint network and later support site-level interpretation (RQ3). Both kinds of link from Section 3.3 enter one weekly adjacency matrix, which is symmetrised and given self-loops before attention; log-transformed voyage counts then enter as a continuous attention prior. Edge direction and edge type are therefore not used in message passing. Layer settings are in Appendix C.

**航运编码器。** 输入为第 3.3 节的周度十七节点航运网络。航运信息具有关系结构，因为港口与走廊相互连接，因此图模型比一行扁平计数更合适。输出为该预测时点的一个航运表征。架构为带时间编码的图注意力网络（GAT；Veličković et al., 2018）。图神经网络已用于将原油与船舶交通网络建模为关系性、时变过程（Ouyang et al., 2022; Liang et al., 2022）。此处采用 GAT，是因为邻居权重适合稀疏港口–咽喉网络，并便于后续站点级解释（RQ3）。第 3.3 节的两类边进入同一个周度邻接矩阵；该矩阵在注意力计算前被对称化并加入自环，对数变换后的航次数作为连续注意力先验。因此消息传递既不使用边的方向，也不区分边的类型。层设置见附录 C。

### 3.5.3 Fusion mechanisms

### 3.5.3 融合机制

**Fusion (RQ2).** Once each available modality has a representation, three ways of combining them are compared. Simple concatenation joins the representations without adaptive weighting and serves as a control. Gated fusion is the main reported design. It learns how much weight to give each modality. Cross-attention is retained as an advanced alternative that lets modalities attend to one another. The fused representation is mapped to the same return and price target as Flat. Training details are in Appendix C.

**融合（RQ2）。** 各可用模态得到表征后，比较三种组合方式。简单拼接在无自适应加权下连接表征，作为对照。门控融合是主要报告设计，它学习给各模态多少权重。交叉注意力保留为进阶备选，允许模态相互关注。融合表征映射到与 Flat 相同的收益与价格目标。训练细节见附录 C。

## 3.6 Estimation and validation

## 3.6 估计与验证

### 3.6.1 Expanding-window validation design

### 3.6.1 扩展窗口验证设计

With a four-week input window and a one-week forecast horizon, the 365 weekly observations yield 361 eligible input–target sequences. The first three observations cannot yet form a complete four-week input sequence, and the final week serves only as a target rather than a forecast origin. Evaluation uses an expanding window. At each origin t, the model forecasts P_{t+1} using only information observable by that date. Each fit uses only input–target pairs whose targets were already observable at the estimation date, and the fitted model then produces one-week-ahead forecasts. This design prevents the use of future information in training or preprocessing. A realised target may enter a later re-estimation sample once it has become observable, but it never enters the sample used to generate its own forecast. The first 104 eligible sequences form the initial estimation period and are not included in the evaluation metrics. The validation weeks used for tuning are taken from inside each training fold rather than from a separate held-out block. The common evaluation span covers 257 weeks from 22 January 2021 to 19 December 2025, with corresponding target dates from 29 January 2021 to 26 December 2025. Any scaling or filtering is fit on the training period only. Flat and Deep share the same evaluation calendar, so architecture comparisons hold the evaluation design fixed.

在四周输入窗口和提前一周预测期下，365 个周度观测可形成 361 个合法的“输入–目标”样本。最前面 3 个观测尚不足以构成完整的四周输入序列，而最后一周只作为目标、不能再作为预测起点。评估采用扩展窗。在每个预测起点 t，模型仅使用截至该日期可观测的信息预测 P_{t+1}。每次拟合仅使用目标在该估计日期之前已可观测的“输入–目标”样本，随后由拟合好的模型给出提前一周预测。该设计避免在训练或预处理中使用未来信息。已实现的目标在可观测之后可以进入之后的重估样本，但不会进入生成其自身预测的估计样本。前 104 个合法样本为初始估计期，不纳入评估指标。调参所用的验证周取自各训练折内部，而非另行划出的留出区块。共同评估跨度为 2021 年 1 月 22 日至 2025 年 12 月 19 日的 257 周，相应目标日期为 2021 年 1 月 29 日至 2025 年 12 月 26 日。任何缩放或过滤仅在训练期内拟合。Flat 与 Deep 共享同一评估日历，从而使架构比较在固定评估设计下进行。

Figure 3.2 gives the calendar view of this design. Figure 3.4 shows one forecast origin in detail: the training fold, the inner validation weeks, the four-week input window and the one-week-ahead target.

图 3.2 给出该设计在日历上的整体视图。图 3.4 进一步展示单个预测起点的内部结构：训练折、内部验证周、四周输入窗与提前一周的目标。

Figure 3.4

**Figure 3.4 — Anatomy of one forecast origin: the training fold with its inner validation weeks, the four-week input window and the one-week-ahead target. The same structure is repeated at each of the 13 origins in a test block.**

**图 3.4 — 单个预测起点的结构：包含内部验证周的训练折、四周输入窗与提前一周的预测目标。测试块内的 13 个起点均重复这一结构。**

### 3.6.2 Hyperparameter selection

### 3.6.2 超参数选择

Hyperparameters are selected under a shared protocol so that Flat–Deep comparisons remain fair. For Flat models, tuning uses only past validation weeks inside each training fold. For Deep models, searching the full architecture at every fold is too costly. A limited search is run first, then one main configuration is fixed for the primary results. Sensitivity checks follow. Exact grids and layer settings are in Appendix C.

超参数在共享协议下选择，以使 Flat–Deep 比较保持公平。Flat 模型仅在各训练折内、用过去验证周调参。Deep 模型若在每一折都完整搜索架构成本过高，故先做有限搜索，再固定主配置作主要结果；随后做敏感性检查。细节见附录 C。

### 3.6.3 Model fitting and re-estimation schedule

### 3.6.3 模型拟合与重估安排

The model is fitted at the first out-of-sample origin. Thereafter the estimation window expands, and the model is re-estimated every 13 forecast origins. Fitted parameters are retained between scheduled re-estimations.

模型在第一个样本外预测起点拟合。此后估计窗口逐步扩展，并每隔 13 个预测起点重新估计一次。两次预定重估之间沿用已有参数。

## 3.7 Forecast evaluation and model interpretation

## 3.7 预测评估与模型解释

### 3.7.1 Error metrics and skill scores

### 3.7.1 误差指标与 skill

Primary metrics are computed on reconstructed prices. Every comparison reports RMSE and MAE. Relative performance versus M0 is summarised by RMSE skill—the percentage improvement in RMSE relative to M0—reported as a percentage in the result tables.

\mathrm{Skill}=100\times\left(1-\frac{\mathrm{RMSE}*{\mathrm{model}}}{\mathrm{RMSE}*{\mathrm{M0}}}\right).

Skill greater than zero means the model beats M0 on RMSE. Skill equal to zero matches M0. Skill less than zero is worse than M0.

主指标在重构价格上计算。每次比较均报告 RMSE 与 MAE。相对 M0 的表现以 RMSE skill（相对 M0 的 RMSE 百分比改善）汇总，并在结果表中以百分比报告。

\mathrm{Skill}=100\times\left(1-\frac{\mathrm{RMSE}*{\mathrm{model}}}{\mathrm{RMSE}*{\mathrm{M0}}}\right).

Skill 大于零表示模型在 RMSE 上优于 M0。等于零与 M0 持平。小于零则差于 M0。

### 3.7.2 Forecast-comparison tests

### 3.7.2 预测比较检验

The study reports both absolute skill versus M0 and incremental value versus S1. Statistical tests are chosen by the type of comparison, not by the size of the modality set alone. Adding remote sensing or shipping enlarges the information set, but that does not by itself make two forecasts nested for testing. When one forecast specification is nested in another—for example Ridge S1 versus Ridge S2, S3 or S4 under the same learner—Clark–West (2007) is used to test whether the larger model improves mean squared prediction error. When the comparison is not nested—for example Flat versus Deep, or XGBoost versus a Deep setting that changes hyperparameters or architecture—Diebold–Mariano (1995) is used to test equal predictive accuracy. A small-sample adjustment is noted where relevant. Every comparison also reports RMSE and MAE differences versus M0 and, where relevant, versus S1.

研究同时报告相对 M0 的绝对 skill 与相对 S1 的增量价值。统计检验按比较类型选择，而非仅按模态集大小。加入遥感或航运会扩大信息集，但这本身并不使两个预测在检验上嵌套。当一个预测设定嵌套于另一个时——例如在同一学习器下 Ridge S1 对 Ridge S2、S3 或 S4——使用 Clark–West（2007）检验较大模型是否改善均方预测误差。当比较不嵌套时——例如 Flat 对 Deep，或改变超参或架构的 XGBoost 与 Deep 设定——使用 Diebold–Mariano（1995）检验等预测精度。相关时注明小样本调整。每次比较亦报告相对 M0、以及相关时相对 S1 的 RMSE 与 MAE 差异。

### 3.7.3 Model interpretation

### 3.7.3 模型解释

Interpretability diagnostics are applied only to specifications that improve on M0. The main cases are Deep S3 and, where relevant, Deep S4. The diagnostics report modality gate weights together with site or node attention.

可解释性诊断仅用于相对 M0 有改善的设定。主要为 Deep S3，以及相关时的 Deep S4。诊断报告模态门控权重与站点或节点注意力。

## 3.8 Ethical considerations and reproducibility

## 3.8 伦理考量与可复现性

The study uses secondary aggregate data only and does not involve human participants. It was approved under the UCL low-risk ethics process. Two features of the data still require reflection. First, the AIS and Global Fishing Watch records identify individual vessels rather than individuals, and they are used here only in aggregate form, as weekly node attributes and origin–destination voyage counts for the seventeen-node network. No attempt is made to identify vessel owners, operators or crew, and no vessel-level results are reported. Second, satellite observation of oil ports, refineries and export terminals raises dual-use considerations, because the same imagery that supports market analysis also describes critical infrastructure. The study therefore uses only openly licensed moderate-resolution products—Sentinel-2 at 10 m and VIIRS night-time lights—at which individuals and operational detail are not discernible, and it observes publicly documented facility locations. Results are reported as site-level aggregates. The interpretability outputs in Section 3.7 describe where a model places weight; they are not assessments of any individual facility or operator. All datasets are used under their published terms for research use, including the Copernicus open licence for Sentinel-2, open distribution terms for VIIRS night-time lights, and the research terms of IMF PortWatch and Global Fishing Watch.

本研究仅使用二手汇总数据，不涉及人类参与者，并已按 UCL 低风险伦理流程获批。数据本身仍有两点需要说明。其一，AIS 与 Global Fishing Watch 记录识别的是船舶而非个人，且本研究仅以汇总形式使用，即十七节点网络的周度节点属性与起讫航次计数；不尝试识别船东、运营方或船员，也不报告单船层面的结果。其二，对油港、炼厂与出口码头的卫星观测涉及双重用途问题，因为支持市场分析的影像同样描述了关键基础设施。为此本研究仅使用开放许可的中等分辨率产品——10 m 的 Sentinel-2 与 VIIRS 夜光——在该分辨率下无法辨识个人与设施运行细节，且所观测的设施位置均为公开信息。结果以站点级汇总形式报告。第 3.7 节的可解释性输出说明模型把权重放在哪里，不构成对任何具体设施或运营方的评估。所有数据集均按其公开的研究使用条款使用，包括 Sentinel-2 的 Copernicus 开放许可、VIIRS 夜光的开放分发条款，以及 IMF PortWatch 与 Global Fishing Watch 的研究条款。

Analysis was carried out in Python. Data preparation, model estimation and evaluation are organised as scripted pipelines rather than manual steps, so that the weekly calendar, the expanding-window splits and the evaluation metrics are produced by the same code for every specification. Random seeds are fixed for the Deep models, and sensitivity to the seed is reported alongside the main results rather than left implicit. Package versions and exact configuration settings are listed in Appendix C. The code repository is private during writing and will be made available with the submitted dissertation.

分析使用 Python 完成。数据准备、模型估计与评估均以脚本化流程组织，而非手工步骤，从而使周历、扩展窗划分与评估指标在所有设定下都由同一套代码产生。Deep 模型固定随机种子，并将种子敏感性与主结果一并报告，而不是留作隐含假设。软件包版本与精确配置见附录 C。代码仓库在写作期间为私有，将随论文提交一并提供。

---

