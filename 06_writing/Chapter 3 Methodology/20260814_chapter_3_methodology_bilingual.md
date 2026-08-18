# Chapter 3 — Methodology ( 3,599 words)

## 3.1 Research design

## 3.1 研究设计

The baseline for this study is a simple no-change benchmark, which predicts that next week’s Brent price will equal this week’s price. All learned models are evaluated against this benchmark. The available information is divided into different information sets, and models are trained and evaluated separately on these sets to examine whether different types of data provide useful predictive information and whether different ways of combining information affect forecasting performance. All specifications share the same weekly forecast dates, sample window and evaluation rules.

本研究的基线是一个简单的不变预测基准，即预测下周 Brent 价格等于本周价格。所有学习得到的模型均对照该基准进行评价。可用信息被划分为不同的信息集，并分别在各信息集上训练和评价模型，以考察不同类型的数据是否提供有用的预测信息，以及不同的信息组合方式是否影响预测表现。所有设定共用同一周度预测日、样本窗口与评价规则。

The no-change benchmark is denoted M0. At each forecast origin t, where P_t is the Brent price in week t, M0 sets the one-week-ahead price forecast equal to the current weekly price:

不变预测基准记为 M0。设 P_t 为第 t 周的 Brent 价格，则在每个预测起点 t，M0 将提前一周的价格预测设为当前周价格：

\hat{P}_{t+1\mid t}=P_t.

The predictors are organised into four information sets. S1 contains financial time series only, including financial, macroeconomic and oil-market variables. S2 adds remote sensing to S1, S3 adds shipping to S1, and S4 adds both modalities. S2 and S3 are parallel extensions of S1 rather than successive stages, while S4 combines the two. Table 3.1 lists the four sets together with the M0 benchmark.

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



| 集合     | 变量                |
| ------ | ----------------- |
| 基准（M0） | 上一周价格             |
| S1     | 仅金融时序（金融、宏观与油市序列） |
| S2     | S1 + 遥感           |
| S3     | S1 + 航运           |
| S4     | S1 + 遥感 + 航运      |


Two model families are applied to these information sets. The Flat family first combines all selected predictors into a weekly feature table. The most recent weeks are then flattened into a single row and used to fit Ridge and XGBoost. This early joining of features is called flat feature fusion. The Deep family initially keeps each data type separate. Financial, remote-sensing and shipping data are encoded independently. The resulting representations are then combined, with gated fusion learning how much weight to assign to each data type. Flat and Deep are compared using the same information set and evaluation sample. Because the two families also differ in model class, capacity and some modality-specific representations, these comparisons are interpreted as comparisons between overall modelling strategies rather than as tests of fusion alone. Fusion is assessed more directly within the Deep family by comparing simple concatenation, gated fusion and cross-attention while holding the encoders and inputs fixed. Together, these comparisons address RQ2.

两套模型族应用于这些信息集。Flat 族首先将所有选定的预测变量合并为一张周度特征表。随后将最近几周的数据展平为一行，用于拟合 Ridge 和 XGBoost。这种在输入阶段直接合并特征的方式称为扁平特征融合。Deep 族则先将不同类型的数据分开处理。金融、遥感和航运数据分别进行编码。随后将得到的表征组合起来，并通过门控融合学习为不同数据类型分配权重。Flat 与 Deep 在相同的信息集和评价样本上进行比较。由于两类模型在模型类型、容量以及部分模态的表征方式上也存在差异，因此这类比较反映的是两种整体建模策略的差异，而不是单独检验融合方式本身。融合方式在 Deep 族内部得到更直接的比较：在编码器和输入保持不变的条件下，比较简单拼接、门控融合与交叉注意力。这些比较共同回答 RQ2。

Within each model family, comparisons across S1–S4 use the same forecasting method and evaluation sample; only the information included in the model changes. S2 against S1 measures the contribution of adding remote sensing alone, while S3 against S1 measures the contribution of shipping. S4 against S1 measures their joint contribution. S4 against S3 and S4 against S2 examine whether each source still adds useful information once the other is already included. These comparisons, together with each model’s comparison against M0, address RQ1.

在每个模型族中，S1–S4 的比较使用相同的预测方法和评价样本，仅改变模型所使用的信息。S2 对 S1 衡量单独加入遥感后带来的贡献，S3 对 S1 衡量加入航运后的贡献。S4 对 S1 衡量两类信息共同加入后的贡献。S4 对 S3 和 S4 对 S2 则分别考察：当另一类数据已经纳入模型后，新增的数据源是否仍能提供有用信息。这些比较连同各模型与 M0 的比较，共同回答 RQ1。

RQ3 is restricted to Deep specifications that outperform M0 under the predefined criterion. For these specifications, the study reports the weights assigned to finance, remote sensing and shipping data, and examines how model attention patterns vary across different market conditions. These quantities are used to describe which information the models rely on.

RQ3 仅针对按照预先设定的准则优于 M0 的 Deep 设定。对于这些设定，本研究报告其赋予金融、遥感与航运数据的权重，并考察不同市场条件下模型注意力模式的变化。这些结果用于描述模型依赖哪些信息。

The overall research design is summarised in Figure 3.1.

总体研究设计见图 3.1。

**Figure 3.1. Research design and forecasting workflow.**

**图 3.1. 研究设计与预测流程。**

## 3.2 Prediction target and sample period

## 3.2 预测目标与样本期

Let P_t denote the last available daily Brent spot-price observation in week t, where each week ends on Friday, measured in US dollars per barrel. The forecast target is next week’s price P_{t+1}. Models are not trained directly on the price level. They predict the one-week logarithmic return

令 P_t 表示第 t 周内最后一个可获得的 Brent 现货价格日度观测，各周于周五结束，单位为美元/桶。预测目标是下一周的价格 P_{t+1}。模型不直接在价格水平上训练，而是预测一周对数收益

r_{t+1}=\log\left(\frac{P_{t+1}}{P_t}\right)

and reconstruct the price forecast as

并按下式重构价格预测：

\hat{P}_{t+1\mid t}=P_t\exp\left(\hat{r}_{t+1\mid t}\right).

Log returns are used to reduce the strong persistence in the price level and to express the forecasting task in terms of proportional weekly changes. Root Mean Squared Error (RMSE) and $\Delta\mathrm{RMSE}$, the percentage improvement in RMSE over M0, are computed from the reconstructed price forecasts. Under this mapping, the no-change benchmark \hat{P}_{t+1\mid t}=P_t is exactly the same as forecasting a zero return \hat{r}_{t+1\mid t}=0.

使用对数收益是为了减弱价格水平的强持续性，并将预测任务表示为周度比例变化。均方根误差（RMSE）以及 $\Delta\mathrm{RMSE}$（相对 M0 的 RMSE 百分比改善）均根据重构后的价格预测计算。在此对应关系下，不变预测基准 \hat{P}_{t+1\mid t}=P_t 与预测收益为零 \hat{r}_{t+1\mid t}=0 完全一致。

The modelling window covers 2019–2025 and provides a common weekly index of 365 observations (4 January 2019 to 26 December 2025). The training and evaluation samples are separated in time on an expanding window, rather than by random assignment, to prevent future information from leaking into model fitting. Figure 3.2 shows the Brent price and the weekly log returns over this window. The full validation protocol is in Section 3.6.

建模窗口覆盖 2019–2025 年，提供含 365 个观测的共同周索引（2019 年 1 月 4 日至 2025 年 12 月 26 日）。训练样本与评价样本在扩展窗口下按时间分开，而非随机划分，以避免未来信息泄漏到模型拟合中。图 3.2 展示该窗口内的 Brent 价格与周对数收益。完整验证协议见第 3.6 节。

**Figure 3.2. Brent spot price and weekly log returns.**

**图 3.2. Brent 现货价格与周对数收益。**

## 3.3 Geographic scope

## 3.3 地理范围

Because the prediction target is the global Brent benchmark rather than a local physical cargo price at a single terminal, the study does not use one specific study region. Spatial information instead comes from 11 oil-infrastructure monitoring sites and 6 maritime chokepoints. Rather than constituting a spatially exhaustive or geographically balanced sample, these locations were purposively selected to span different functional positions in the international oil system, including supply, transit, refining, demand and market access. Figure 3.3 places these sites and chokepoints on a world map. Full site names, coordinates, functional roles, patch sizes and graph edge definitions are in Appendix A.

由于预测对象是全球 Brent 基准，而非单一码头的本地实物货价，本研究不采用一块特定的地理研究区。空间信息来自 11 个石油基础设施监测站点与 6 个航运咽喉。这些位置并不构成对全球石油体系的完整或地理均衡抽样，而是通过目的性选取，覆盖供给、中转、炼化、需求与市场接入等不同功能环节。图 3.3 在世界地图上标出这些站点与咽喉。完整站名、坐标、功能角色、影像块尺寸与图边定义见附录 A。

The 11 sites comprise ports, refineries and export terminals selected purposively for their strategic roles and observability in the available satellite products. Flat remote-sensing features are summarised within a circular buffer of 5 km radius around each site. Deep image patches are centred on the same sites but vary in size by facility type and local spatial constraints. The sizes are generally larger for ports, intermediate for refineries and smaller for terminals.

这 11 个站点包括港口、炼油厂和出口码头，并根据其战略作用以及在现有卫星产品中的可观测性进行目的性选取。Flat 路径的遥感特征在以各站点为中心、半径 5 km 的圆形缓冲区内进行汇总。Deep 路径的影像块同样以这些站点为中心，但其大小会根据设施类型和当地空间条件进行调整。总体而言，港口使用较大的影像块，炼油厂居中，码头较小。

The shipping graph augments the 11 sites with 6 maritime chokepoints: the Strait of Hormuz, the Suez Canal, the Strait of Malacca, Bab el-Mandeb, the Panama Canal and the Cape of Good Hope. The resulting weekly graph contains 17 nodes and two edge classes. Dynamic voyage edges between the 11 AOIs are directed origin–destination pairs, weighted by the number of voyages counted in each week from Global Fishing Watch (GFW) port-visit sequences, and therefore change from week to week. Fixed corridor edges are undirected. Each site is connected to the chokepoint or chokepoints on its main documented oil-trade corridor. These edges are defined in advance rather than inferred from weekly vessel movements or geographic proximity. Complete edge definitions are reported in Appendix A, and graph encoding is described in Section 3.5.2.

航运图在 11 个站点之外加入 6 个航运咽喉：霍尔木兹海峡、苏伊士运河、马六甲海峡、曼德海峡、巴拿马运河与好望角。由此形成包含 17 个节点、两类边的周度图。11 个 AOI 之间的动态航次边为有向的起点–终点对，权重取各周内由 Global Fishing Watch 港口访问序列统计到的航次数，因而随周变化。固定走廊边为无向。每个站点与其主要石油贸易走廊上的一个或多个咽喉相连。这些边预先设定，而不是根据周度船舶移动或地理邻近关系推断。完整边定义见附录 A，图编码方法见第 3.5.2 节。

**Figure 3.3. Spatial distribution of the 11 oil-infrastructure AOIs and 6 maritime chokepoints.**

**图 3.3. 11 个石油基础设施 AOI 与 6 个航运咽喉的空间分布。**

## 3.4 Data sources and preparation

## 3.4 数据来源与准备

### 3.4.1 Data sources

### 3.4.1 数据来源

The study uses three broad types of data: financial time series, satellite remote sensing and maritime shipping data. These three data types are referred to as the financial, remote-sensing and shipping data blocks. Flat and Deep models cover the same monitoring locations, but use different products and representations for the remote-sensing and shipping blocks (Table 3.2). Flat models organise the data in a merged weekly feature table, whereas Deep models retain modality-specific sequences, image embeddings and graph inputs.

本研究使用三大类数据：金融时序、卫星遥感和航运数据。本文将这三类数据分别称为金融、遥感和航运数据块。Flat 与 Deep 模型覆盖相同的监测地点，但遥感和航运数据块使用的产品与表征方式不同（表 3.2）。Flat 模型将数据整理为一张合并的周度特征表，而 Deep 模型则保留模态专属的序列、影像嵌入和图输入。

**Table 3.2 — Datasets, variables and sources**

**表 3.2 — 数据集、变量与来源**


| Modality              | Dataset / product                                                                     | Key variables                                                                                                                  | Source                                                                                                                                                                                                                                                |
| --------------------- | ------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Financial time series | Oil-market and macro-financial series at daily, weekly and monthly native frequencies | Prices, inventories, production, interest rates, GPR and related indicators                                                    | [EIA](https://www.eia.gov/petroleum/supply/weekly/); [FRED](https://fred.stlouisfed.org/); [Yahoo Finance](https://finance.yahoo.com/); [Dallas Fed IGREA](https://www.dallasfed.org/research/igrea); [GPR](https://www.matteoiacoviello.com/gpr.htm) |
| Remote sensing (Flat) | Sentinel-2 optical indices and VIIRS night-time lights                                | Site-level anomalies at 11 AOIs (NDVI, NDWI, NDBI, BSI; NTL)                                                                   | [Sentinel-2 via GEE](https://developers.google.com/earth-engine/datasets/catalog/COPERNICUS_S2_SR_HARMONIZED); [VIIRS via GEE](https://developers.google.com/earth-engine/datasets/catalog/NOAA_VIIRS_DNB_MONTHLY_V1_VCMSLCFG)        |
| Remote sensing (Deep) | Monthly Sentinel-2 image patches                                                      | Frozen Prithvi-EO-2.0 embeddings at the same 11 AOIs                                                                           | [Prithvi-EO-2.0](https://huggingface.co/ibm-nasa-geospatial); [Sentinel-2 via GEE](https://developers.google.com/earth-engine/datasets/catalog/COPERNICUS_S2_SR_HARMONIZED)                                                                           |
| Shipping (Flat)       | PortWatch and Global Fishing Watch AIS and SAR data                                   | Port and chokepoint tanker flows; vessel-activity features                                                                     | [IMF PortWatch](https://portwatch.imf.org/) (AIS-derived); [Global Fishing Watch](https://globalfishingwatch.org/our-apis/) (AIS- and SAR-derived)                                                                                                    |
| Shipping (Deep)       | PortWatch and Global Fishing Watch AIS and SAR data                                   | Weekly node attributes, dynamic voyage edges and fixed corridor edges for a 17-node graph comprising 11 AOIs and 6 chokepoints | [IMF PortWatch](https://portwatch.imf.org/) (AIS-derived); [Global Fishing Watch](https://globalfishingwatch.org/our-apis/) (AIS- and SAR-derived)                                                                                                    |



| 模态       | 数据集 / 产品                                        | 关键变量                                        | 来源                                                                                                                                                                                                                                                |
| -------- | ----------------------------------------------- | ------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 金融时序     | 日、周、月原生频率的油市与宏观金融序列                             | 价格、库存、产量、利率、GPR 及相关指标                       | [EIA](https://www.eia.gov/petroleum/supply/weekly/)；[FRED](https://fred.stlouisfed.org/)；[Yahoo Finance](https://finance.yahoo.com/)；[Dallas Fed IGREA](https://www.dallasfed.org/research/igrea)；[GPR](https://www.matteoiacoviello.com/gpr.htm) |
| 遥感（Flat） | Sentinel-2 光学指数与 VIIRS 夜光                       | 11 个 AOI 的站点级距平（NDVI、NDWI、NDBI、BSI 与 NTL）   | [Sentinel-2 via GEE](https://developers.google.com/earth-engine/datasets/catalog/COPERNICUS_S2_SR_HARMONIZED)；[VIIRS via GEE](https://developers.google.com/earth-engine/datasets/catalog/NOAA_VIIRS_DNB_MONTHLY_V1_VCMSLCFG)                     |
| 遥感（Deep） | 月度 Sentinel-2 影像块                               | 同一 11 个 AOI 的冻结 Prithvi-EO-2.0 嵌入           | [Prithvi-EO-2.0](https://huggingface.co/ibm-nasa-geospatial)；[Sentinel-2 via GEE](https://developers.google.com/earth-engine/datasets/catalog/COPERNICUS_S2_SR_HARMONIZED)                                                                        |
| 航运（Flat） | PortWatch 与 Global Fishing Watch 的 AIS 与 SAR 数据 | 港口与咽喉油轮流量；船舶活动特征                            | [IMF PortWatch](https://portwatch.imf.org/)（AIS 衍生）；[Global Fishing Watch](https://globalfishingwatch.org/our-apis/)（AIS 与 SAR 衍生）                                                                                                                |
| 航运（Deep） | PortWatch 与 Global Fishing Watch 的 AIS 与 SAR 数据 | 17 节点周度图的节点属性、动态航次边与固定走廊边（11 个 AOI 与 6 个咽喉） | [IMF PortWatch](https://portwatch.imf.org/)（AIS 衍生）；[Global Fishing Watch](https://globalfishingwatch.org/our-apis/)（AIS 与 SAR 衍生）                                                                                                                |


The financial block combines oil-market and macro-financial series from the US Energy Information Administration (EIA), Federal Reserve Economic Data (FRED), Yahoo Finance, the Dallas Fed Index of Global Real Economic Activity, and the Caldara–Iacoviello geopolitical risk (GPR) index. These series are observed at different native frequencies and include crude prices and spreads, inventories, production and refinery activity, volatility and risk measures, interest rates, exchange rates, and futures-based oil indicators.

金融数据块汇总了来自美国能源信息署（EIA）、联邦储备经济数据（FRED）、Yahoo Finance、达拉斯联储全球实际经济活动指数，以及 Caldara–Iacoviello 地缘政治风险（GPR）指数的油市和宏观金融序列。这些序列具有不同的原生时间频率，涵盖原油价格及价差、库存、产量与炼厂活动、波动与风险指标、利率、汇率，以及基于期货的油市指标。

For the remote-sensing block, the Flat and Deep pathways extract inputs from the same 11 AOIs, but differ in their spatial coverage, products and representations. The Flat pathway uses Sentinel-2 optical indices and VIIRS night-time lights, whereas the Deep pathway uses frozen Prithvi-EO-2.0 embeddings derived from Sentinel-2 image patches and has no separate VIIRS input stream. Preliminary experiments indicated that a separate VIIRS stream added little useful signal and reduced predictive performance, so it was excluded from the final Deep specification.

对于遥感数据块，Flat 和 Deep 两条路径从相同的 11 个 AOI 提取输入，但空间范围、产品与表征不同。Flat 路径使用 Sentinel-2 光学指数与 VIIRS 夜光，Deep 路径使用由 Sentinel-2 影像块得到的冻结 Prithvi-EO-2.0 嵌入，且无单独的 VIIRS 输入流。预实验表明，单独的 VIIRS 流几乎未增加有用信号，并降低了预测表现，因此未纳入最终 Deep 设定。

The shipping block covers activity at the 11 AOIs and 6 chokepoints. IMF PortWatch supplies tanker-flow measures at ports and chokepoints. Global Fishing Watch supplies measures of vessel presence and activity duration. These variables serve as proxies for physical shipping activity, tanker movements and congestion. In the Flat pathway, PortWatch and GFW enter as weekly tabular features. In the Deep pathway, the 17-node graph uses PortWatch flows, GFW AIS vessel-presence and activity measures, and GFW Synthetic Aperture Radar (SAR)-derived dark-vessel detections as node attributes. Dynamic voyage edges are constructed from GFW port-visit sequences. Full variable definitions are reported in Appendix A.

航运数据块覆盖 11 个 AOI 和 6 个航运咽喉的活动。IMF PortWatch 提供港口和咽喉的油轮流量指标。Global Fishing Watch 提供船舶存在情况和活动时长指标。这些变量用于表征实物航运活动、油轮移动和拥堵情况。在 Flat 路径中，PortWatch 和 GFW 数据以周度表格特征的形式输入模型。在 Deep 路径中，17 节点图将 PortWatch 流量、GFW 的 AIS 船舶存在与活动指标，以及 GFW 基于合成孔径雷达（SAR）的暗船检测作为节点属性。动态航次边由 GFW 港口访问序列构建。完整变量定义见附录 A。

Figure 3.4 works through one Friday origin at Jurong Island. It shows the two remote-sensing spatial supports, the values available at that date, and the shipping neighbourhood after product-specific lags.

图 3.4 以裕廊岛上的一个周五预测起点为例，展示两种遥感空间范围、该日可用的取值，以及按产品分别滞后之后的航运邻域。

**Figure 3.4. Worked example of spatial inputs and representations for Jurong Island at the 14 March 2025 forecast origin.**

**图 3.4. 裕廊岛空间输入与表征工作示例，预测起点为 2025 年 3 月 14 日。**

### 3.4.2 Spatial aggregation and temporal alignment

### 3.4.2 空间聚合与时间对齐

Spatially referenced data are first aggregated to the monitoring locations used in the models. Remote-sensing observations are aggregated from pixels to AOI–month products. PortWatch and GFW series are aggregated from vessel events or traffic counts to AOIs and chokepoints. The identity of each site or node is retained before the series enter the models.

带有空间参照的数据首先聚合到模型所使用的监测位置。遥感观测由像元聚合为 AOI–月度产品。PortWatch 与 GFW 序列由船舶事件或交通计数聚合到 AOI 与咽喉。各站点或节点的身份在进入模型前予以保留。

All series are aligned to a common Friday-ending weekly calendar. Daily observations are converted using end-of-week values, weekly means or weekly sums as appropriate, while monthly series are carried forward only after their assumed availability dates. Monthly remote-sensing products are aligned to their conservative availability dates, with the most recent eligible composite carried forward to avoid look-ahead bias.

所有时间序列均对齐至统一的、以周五为周末的周度日历。对于日度数据，根据变量性质分别采用周末值、周均值或周总和进行周度聚合；对于月度数据，则仅在其假定的可获得日期之后向前填充。月度遥感产品按照较为保守的可获得日期进行对齐，并仅向前延用最近一期已符合可用条件的合成数据，以避免前视偏差。

Publication timing is approximated using source- and product-specific fixed lag buffers rather than observation-level release timestamps. Exact aggregation rules, lag constants and implementation scripts are reported in Appendix A. One-week buffers are applied to EIA fundamentals and PortWatch flows, while monthly macroeconomic series, remote-sensing products and individual GFW AIS and SAR products receive longer buffers.

发布时间通过按来源和产品设定的固定滞后缓冲近似，而不是逐条采用观测值的实际发布时间戳。具体聚合规则、滞后常数与实现脚本见附录 A。EIA 基本面与 PortWatch 流量施加一周缓冲，月度宏观序列、遥感产品以及 GFW 的各 AIS 与 SAR 产品则使用更长的缓冲。

### 3.4.3 Data quality and missing values

### 3.4.3 数据质量与缺失值

Monthly optical composites are cloud-filtered before the indices are constructed, and cloud-quality indicators are not used as predictors. On the weekly calendar, mean coverage across the four optical indices is approximately 97 per cent, while VIIRS night-time-light anomalies are fully observed. Site-level coverage and counts of independent monthly composites are reported in Appendix A.

月度光学合成影像在构建指数前进行云筛选，云质量指标不作为预测变量。在周度日历上，四项光学指数的平均覆盖率约为 97%，VIIRS 夜间灯光距平则完全可用。站点级覆盖率与独立月度合成数量见附录 A。

After temporal alignment, the remaining gaps occur almost entirely before each series’ first valid observation. These leading gaps are set to zero for remote-sensing variables and filled with the training-fold median for each shipping-count variable. Deep finance inputs contain no missing values after merging. Missing remote-sensing embeddings and shipping-graph values are set to zero after scaling. All imputation and scaling parameters are estimated separately within each training window.

时间对齐后，剩余的缺失值几乎全部出现在各时间序列首次有效观测之前。对于这些前置缺失值，遥感变量统一填充为 0，而各航运计数变量则使用对应训练折的中位数进行填补。Deep 金融输入在合并后不存在缺失值。缺失的遥感嵌入和航运图数值在缩放后设为 0。所有填补与缩放参数均在各训练窗口内分别估计。

## 3.5 Forecasting models

## 3.5 预测模型

### 3.5.1 Flat models

### 3.5.1 Flat 模型

Flat models implement early feature-level fusion. For each information set, all available numeric features are concatenated into a weekly feature table, and the most recent four weeks are flattened into a single row for each forecast origin. Ridge applies L2 regularisation (Hoerl and Kennard, 1970) and serves as a transparent linear comparator. XGBoost is a nonlinear gradient-boosted tree ensemble (Chen and Guestrin, 2016) that captures nonlinearities and interactions not represented by Ridge. Because both learners operate on the same flattened table, neither preserves modality-specific structure. Both predict the one-week-ahead log return and then reconstruct the corresponding price forecast. Hyperparameter selection follows the time-ordered procedure described in Section 3.6, with exact search grids reported in Appendix C.

Flat 模型采用早期特征级融合。对每个信息集，所有可用数值特征拼接为一张周度特征表，并在每个预测起点将最近四周展平为一行。Ridge 采用 L2 正则化（Hoerl and Kennard, 1970），作为透明的线性对照。XGBoost 是非线性梯度提升树集成（Chen and Guestrin, 2016），用于捕捉 Ridge 无法表示的非线性与交互。由于两种学习器都作用于同一张展平表，因此都不保留模态专属结构。二者均预测提前一周的对数收益，再还原相应的价格预测。超参数选择遵循第 3.6 节所述的时序程序，具体搜索网格见附录 C。

### 3.5.2 Deep models

### 3.5.2 Deep 模型

Deep models encode each modality separately, retaining its temporal and, where applicable, site or network organisation before fusing the resulting representations for S2–S4.

Deep 模型分别编码各模态，在融合 S2–S4 的模态表征之前，保留各模态的时间结构，并在适用时保留其站点或网络组织。

The finance encoder applies a temporal convolutional network (TCN; Bai, Kolter and Koltun, 2018) to the four-week financial sequence retained across S1–S4. It produces one finance representation per forecast origin using only current and earlier positions at each convolutional layer.

金融编码器将时间卷积网络（TCN；Bai, Kolter and Koltun, 2018）应用于 S1–S4 均保留的四周金融序列。各卷积层仅使用当前位置及其之前的位置，并为每个预测起点生成一个金融表征。

The remote-sensing encoder receives monthly embeddings for the 11 AOIs, extracted from Sentinel-2 Surface Reflectance Harmonized patches using a frozen Prithvi-EO-2.0-300M encoder (Szwarcman et al., 2026). Prithvi-EO-2.0 is a publicly released Earth-observation foundation model. This study uses the published weights as a fixed encoder and does not retrain or fine-tune them. The patches are adapted to the six-band convention of the HLS-pretrained encoder, with band mapping, standardisation and resampling documented in Appendix A. Temporal attention combines the four-week embeddings for each AOI. Site attention then combines the 11 AOIs into one remote-sensing representation.

遥感编码器接收 11 个 AOI 的月度嵌入，这些嵌入由冻结的 Prithvi-EO-2.0-300M 编码器从 Sentinel-2 地表反射率和谐化影像块中提取（Szwarcman 等，2026）。Prithvi-EO-2.0 是公开发布的对地观测基础模型。本研究使用其已发布权重作为固定编码器，不对其进行再训练或微调。影像块按照该 HLS 预训练编码器的六波段约定进行适配，波段映射、标准化与重采样方法见附录 A。时间注意力先汇总每个 AOI 的四周嵌入，随后站点注意力将 11 个 AOI 汇总为一个遥感表征。

The shipping encoder applies a graph attention network (GAT) with temporal encoding (Veličković et al., 2018) to the weekly 17-node graph over the four-week lookback, producing one shipping representation per forecast origin. The graph’s nodes and two edge classes are as described in Section 3.3, with dynamic voyage edges directed and fixed corridor edges undirected. For message passing, the directed voyage edges and undirected corridor edges are combined in a symmetrised adjacency matrix, so the encoder does not retain edge direction or type.

航运编码器将带时间编码的图注意力网络（GAT；Veličković 等，2018）应用于四周回看窗口内的周度 17 节点图，为每个预测起点生成一个航运表征。图的节点与两类边如第 3.3 节所述，动态航次边为有向，固定走廊边为无向。进入消息传递时，有向航次边与无向走廊边合并为一张对称化邻接矩阵，因此编码器不再保留边的方向与类型。

The two edge classes before combination and symmetrisation are shown in Figure 3.5.

合并与对称化之前的两类边见图 3.5。

**Figure 3.5. Dynamic voyage and fixed corridor edges in the 17-node shipping graph.**

**图 3.5. 17 节点航运图中的动态航次边与固定走廊边。**

Fusion is applied only to Deep models for S2–S4, while S1 passes its finance representation directly to the regression head. Three mechanisms are compared. Gated fusion is designated as the main design because it provides forecast-origin-specific modality weights for the subsequent interpretability analysis. These weights are non-negative and sum to one at each forecast origin. Encoder concatenation and cross-attention are used only as alternatives. The resulting representation is trained by mean squared error to predict the one-week-ahead log return. Encoder and training settings are reported in Appendix C.

融合仅用于 S2–S4 的 Deep 模型，S1 的金融表征则直接进入回归头。研究比较了三种机制。门控融合被指定为主要设计，因为它能为后续可解释性分析提供随预测起点变化的模态权重。这些权重非负，且在每个预测起点上总和为 1。编码器拼接与交叉注意力仅作为备选。所得表征以均方误差训练，用于预测提前一周的对数收益。编码器与训练设定见附录 C。

Figure 3.6 summarises the Deep model architecture and fusion mechanisms.

图 3.6 概括 Deep 模型架构与融合机制。

**Figure 3.6. Deep model architecture and multimodal fusion mechanisms.**

**图 3.6. Deep 模型架构与多模态融合机制。**

## 3.6 Training and validation

## 3.6 训练与验证

### 3.6.1 Expanding-window estimation and re-estimation

### 3.6.1 扩展窗估计与重估

With a four-week input window and a one-week forecast horizon, the 365 weekly observations yield 361 eligible input–target sequences. The first three observations cannot yet form a complete four-week input sequence, and the final week serves only as a target rather than a forecast origin. The first 104 eligible sequences form the initial training period and are not included in the evaluation metrics. The evaluation span covers 257 weeks from 22 January 2021 to 19 December 2025, with corresponding target dates from 29 January 2021 to 26 December 2025. Flat and Deep share this evaluation calendar.

在四周输入窗口与提前一周预测期下，365 个周度观测形成 361 个合格的输入–目标序列。最前面三个观测尚不足以构成完整的四周输入序列，最后一周只作为目标而不能作为预测起点。前 104 个合格序列构成初始训练期，不纳入评价指标。评价跨度为 2021 年 1 月 22 日至 2025 年 12 月 19 日的 257 周，相应目标日期为 2021 年 1 月 29 日至 2025 年 12 月 26 日。Flat 与 Deep 共用这一评价日历。

A one-week-ahead forecast is produced at every forecast origin t, using only information available by that date. Each model is first estimated at the start of the evaluation period and re-estimated every 13 forecast origins as the training window expands. Between re-estimations, the fitted model and preprocessing parameters remain fixed, while the input window is updated at every forecast origin. This produces 20 estimation blocks: the first 19 contain 13 forecast origins each, and the final block contains 10. At each re-estimation, the training sample includes only observations whose following week’s target price is already known. Target prices that become available later are added at the next scheduled re-estimation. All preprocessing parameters calculated from the data are estimated using only the corresponding training sample.

每个预测起点 t 都进行一次提前一周预测，并且只使用截至该日已经可获得的信息。每个模型首先在评价期开始时进行估计，随后随着训练窗口扩展，每隔 13 个预测起点重新估计一次。在两次重新估计之间，模型和预处理参数保持不变，但输入窗口会在每个预测起点更新。由此形成 20 个估计块：前 19 个各包含 13 个预测起点，最后一个包含 10 个。每次重新估计时，训练样本只包含下一周目标价格已经确定的观测。之后才获得的目标价格会在下一次预定的重新估计时加入训练样本。所有根据数据计算的预处理参数也只使用对应的训练样本进行估计。

Figure 3.7 summarises the expanding-window schedule, while Figure 3.8 shows how the four-week input window and one-week-ahead target advance within the first estimation block.

图 3.7 概括扩展窗安排，图 3.8 展示四周输入窗口与提前一周目标如何在第一个估计块内向前推进。

**Figure 3.7. Expanding-window estimation and evaluation schedule.**

**图 3.7. 扩展窗估计与评价安排。**

**Figure 3.8. Testing sample within one testing block (the first block shown). The fitted model is held fixed; t is the last of the four input weeks, and the model predicts week t+1. Each testing block contains 13 predictions, except the last.**

**图 3.8. 一个测试块内的测试样本（以第一个测试块为例）。拟合模型保持固定；t 为四周输入的最后一周，模型预测第 t+1 周。除最后一个测试块外，每个测试块包含 13 次预测。**

### 3.6.2 Model and Hyperparameter selection

### 3.6.2 模型与超参数选择

Flat models re-select Ridge and XGBoost hyperparameters on an inner validation set at each scheduled re-estimation, then refit on the full estimation sample available at that date. Deep models instead use a configuration fixed before evaluation, including the latent size. They share the four-week lookback used by the Flat models, which covers approximately one update cycle of the monthly remote-sensing and macroeconomic inputs. At each Deep re-estimation, inner validation is used for early stopping, and the checkpoint with the lowest validation loss is retained. The selected Deep checkpoint is used for the subsequent forecasting. The model is not retrained using the combined training and validation data. Sensitivity analyses using the evaluation sample are reported separately in Appendix B and are not used to select or revise the main specification. Search grids, fixed configurations and early-stopping settings are reported in Appendix C.

Flat 模型在每次预定的重新估计时，先在内部验证集上重新选择 Ridge 和 XGBoost 的超参数，再使用当时可获得的完整估计样本重新拟合。Deep 模型则使用在评价开始前固定的配置，包括潜在维度。它们与 Flat 模型共用四周回看窗口，该窗口大约覆盖月度遥感与宏观输入的一个更新周期。每次重新估计 Deep 模型时，使用内部验证进行早停，并保留验证损失最低的 checkpoint。选定的 Deep checkpoint 直接用于后续预测，模型不会再使用合并后的训练集和验证集重新训练。基于评价样本的敏感性分析单独报告在附录 B 中，不用于选择或修改主要模型设定。搜索网格、固定配置和早停设置见附录 C。

## 3.7 Model Evaluation and Interpretation

## 3.7 模型评价与解释

### 3.7.1 Forecast Evaluation

### 3.7.1 预测评价

The primary evaluation metrics are calculated from reconstructed price forecasts over the common sample of n=257 forecast origins.

主要评价指标根据重构后的价格预测，在 n=257 个共同预测起点上计算。

\mathrm{RMSE}=
\sqrt{
\frac{1}{n}
\sum_{t=1}^{n}
\left(P_{t+1}-\hat{P}_{t+1\mid t}\right)^2
}.

Performance relative to the no-change benchmark M0 is summarised by $\Delta\mathrm{RMSE}$:

相对不变预测基准 M0 的表现，以 $\Delta\mathrm{RMSE}$ 来概括：

\Delta\mathrm{RMSE}
=
100\times\left(1-\frac{\mathrm{RMSE}}{\mathrm{RMSE}_{\mathrm{M0}}}\right).

Here, P_{t+1} is the observed price and \hat{P}_{t+1\mid t} is the price forecast produced by the model at forecast origin t. The index t=1,…,n orders the 257 evaluation origins in time; it does not index the 365-week calendar. A positive $\Delta\mathrm{RMSE}$ indicates a lower RMSE than M0, zero indicates equal RMSE, and a negative value indicates worse performance.

其中 P_{t+1} 为观测价格，\hat{P}_{t+1\mid t} 为该模型在预测起点 t 给出的价格预测。下标 t=1,…,n 按时间排列这 257 个评价起点，而不是全部 365 个日历周。正的 $\Delta\mathrm{RMSE}$ 表示 RMSE 低于 M0，零表示 RMSE 相同，负值表示表现更差。

### 3.7.2 Model Interpretation and Robustness Checks

### 3.7.2 模型解释与稳健性检验

Model interpretability concerns identifying where a model’s predictive ability comes from, including which data sources and features contribute more or less to its predictions. For the main Deep specification with positive $\Delta\mathrm{RMSE}$, SHAP values are calculated to quantify each input’s contribution (Lundberg and Lee, 2017). Absolute SHAP values are aggregated by data source and, where applicable, spatial site or node. Results are reported for the full sample, by year, and within predefined ±8-week event windows. For gated models, weekly modality weights are also reported to describe how the model allocates weight across financial, remote-sensing and shipping representations. SHAP and gate weights describe model attribution and internal allocation rather than causal importance. Robustness is assessed by rerunning the prespecified Deep models with random seeds and comparing their RMSE.

模型可解释性关注识别模型预测能力的来源，包括哪些数据源和特征对预测贡献更大或更小。对于具有正 $\Delta\mathrm{RMSE}$ 的主要 Deep 设定，计算 SHapley Additive exPlanations（SHAP）值以量化各输入的贡献（Lundberg and Lee, 2017）。绝对 SHAP 值按数据源汇总，并在适用时按空间站点或节点汇总。结果报告全样本、分年以及预先设定的 ±8 周事件窗口。对于门控模型，还报告周度模态权重，以描述模型如何在金融、遥感与航运表征之间分配权重。SHAP 与门控权重描述的是模型归因与内部分配，而非因果重要性。稳健性通过以随机种子重跑预先指定的 Deep 模型并比较其 RMSE 来评估。

## 3.8 Ethical considerations

## 3.8 伦理考量

The study uses only secondary, aggregate data and does not involve human participants. It received approval through UCL’s low-risk ethics process. All datasets were used in accordance with their published licences and terms of use, including the Copernicus open licence for Sentinel-2, the open distribution terms for VIIRS night-time lights, and the research-use terms of IMF PortWatch and Global Fishing Watch. Remote-sensing and vessel-activity variables are analysed only at the aggregate site or chokepoint level; no attempt is made to identify individual vessels, operators or persons.

本研究仅使用二手、汇总型数据，不涉及人类参与者。研究已通过 UCL 低风险伦理审批。所有数据集均按其公布的许可与使用条款使用，包括 Sentinel-2 的 Copernicus 开放许可、VIIRS 夜光的开放分发条款，以及 IMF PortWatch 与 Global Fishing Watch 的研究使用条款。遥感与船舶活动变量仅在站点或咽喉的汇总层面分析；不试图识别单船、运营商或个人。

Analysis was conducted in Python, and the code required to reproduce the analysis is available on GitHub ([https://github.com/XINRUIQI/CASA0004-Dissertation](https://github.com/XINRUIQI/CASA0004-Dissertation)). Package versions, random seeds and installation commands are documented in the repository README. Search grids and locked training settings are reported in Appendix C.

分析在 Python 中完成，复现所需代码见 GitHub（[https://github.com/XINRUIQI/CASA0004-Dissertation](https://github.com/XINRUIQI/CASA0004-Dissertation)）。软件包版本、随机种子与安装命令见仓库 README。搜索网格与锁定训练设定见附录 C。

---
