# Chapter 3 — Methodology 

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

RQ3 is restricted to Deep models that improve on M0 according to the  predefined criterion **defined in Section 3.7？？要写3.7吗**. For these specifications the study reports the weights assigned to finance, remote sensing and shipping, together with the sites or network nodes receiving greater attention under different market conditions. These quantities indicate what a model relies on. They are not interpreted as evidence of causal importance.

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

The study uses three data blocks comprising financial time series, satellite remote sensing and maritime shipping data. Flat and Deep use the same monitoring geography but differ in the products and representations used for the remote-sensing and shipping blocks (Table 3.2). Flat models use predictors assembled in a merged weekly feature table, whereas Deep models retain modality-specific sequences, image embeddings and graph inputs.

本研究使用三类数据块，涵盖金融时序、卫星遥感与航运数据。Flat 与 Deep 使用相同的周五截止日历和监测地理范围，但各块经清洗与对齐后，两条路径使用的具体产品与表征可能不同（见表 3.2）。Flat 模型使用合并周度特征表中的预测变量，Deep 模型则保留模态专属的序列、影像嵌入与图输入。

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
| 航运（Flat） | PortWatch 与 Global Fishing Watch 的 AIS 数据       | 港口与咽喉油轮流量、船舶活动特征                            | [IMF PortWatch](https://portwatch.imf.org/)（AIS 衍生）；[Global Fishing Watch](https://globalfishingwatch.org/our-apis/)（AIS 衍生）                                                                                                                      |
| 航运（Deep） | PortWatch 与 Global Fishing Watch 的 AIS 与 SAR 数据 | 17 节点周度图的节点属性、动态航次边与固定走廊边（11 个 AOI 与 6 个咽喉） | [IMF PortWatch](https://portwatch.imf.org/)（AIS 衍生）；[Global Fishing Watch](https://globalfishingwatch.org/our-apis/)（AIS 与 SAR 衍生）                                                                                                                |


The financial block combines oil-market and macro-financial series from the US Energy Information Administration (EIA), Federal Reserve Economic Data (FRED), Yahoo Finance, the Dallas Fed Index of Global Real Economic Activity, and the Caldara–Iacoviello geopolitical risk (GPR) index. These series are observed at different native frequencies and include crude prices and spreads, inventories, production and refinery activity, volatility and risk measures, interest rates, exchange rates, futures-based oil indicators and geopolitical risk. S1 uses the financial block alone. In the Deep pathway, these variables form a single multivariate input sequence for the finance encoder, spanning oil-market, macro-financial and geopolitical indicators.

金融块汇总来自美国能源信息署（EIA）、联邦储备经济数据（FRED）、Yahoo Finance、达拉斯联储全球实际经济活动指数和 Caldara–Iacoviello 地缘政治风险（GPR）指数的油市与宏观金融序列。这些序列具有不同的原生频率，包括原油价格与价差、库存、产量与炼厂活动、波动与风险度量、利率、汇率、基于期货的油市指标以及地缘政治风险。S1 仅使用该金融块。在 Deep 路径中，这些变量构成金融编码器的单一多元输入序列，涵盖油市、宏观金融与地缘政治指标。

Both pathways derive remote-sensing inputs for the same eleven AOIs but use different spatial footprints, products and representations. The Flat pathway uses Sentinel-2 optical indices and VIIRS night-time lights, whereas the Deep pathway uses frozen Prithvi-EO-2.0 embeddings derived from Sentinel-2 image patches and has no separate VIIRS input stream.

两条路径均从同一十一个 AOI 提取遥感输入，但空间范围、产品与表征不同。Flat 路径使用 Sentinel-2 光学指数与 VIIRS 夜光，Deep 路径使用由 Sentinel-2 影像块得到的冻结 Prithvi-EO-2.0 嵌入，且无单独的 VIIRS 输入流。

The shipping block covers activity at the eleven AOIs and six chokepoints. IMF PortWatch supplies AIS-derived tanker-flow measures at ports and chokepoints. Global Fishing Watch supplies AIS-derived measures of vessel presence and activity duration, together with port-visit records. In the Flat pathway, PortWatch and GFW AIS-derived series enter as weekly tabular features, while SAR-derived dark-vessel variables are not included in the main feature set. In the Deep pathway, GFW SAR-derived dark-vessel detections enter as node attributes in the 17-node graph, while port-visit sequences are used to construct dynamic AOI–AOI voyage links. In both pathways, these variables serve as proxies for physical shipping activity, tanker movements and congestion.

航运块覆盖十一个 AOI 与六个咽喉的活动。IMF PortWatch 提供港口与咽喉的 AIS 衍生油轮流量。Global Fishing Watch 提供 AIS 衍生的船舶存在量与活动时长，以及港口访问记录。在 Flat 路径中，PortWatch 与 GFW 的 AIS 衍生序列以周度表格特征进入，主特征集不含 SAR 暗船变量。在 Deep 路径中，GFW 的 SAR 暗船检测作为十七节点图的节点属性进入，港口访问序列则用于构造 AOI 之间的动态航次边。两条路径都将这些变量视为实物航运活动、油轮移动与拥堵的代理。

### 3.4.2 Temporal alignment and publication lags

### 3.4.2 时间对齐与发布滞后

All series are aligned to a common Friday-ending weekly calendar before modelling. Daily market series use the last available observation within each week, daily GPR is averaged over the week, and daily PortWatch tanker counts and capacities are summed weekly. Monthly macroeconomic and GFW series are carried forward from month-end only after their assumed availability dates.

全部序列在建模前对齐到共同的周五截止周历。日度市场序列取各周内最后一个可得观测，日度 GPR 在周内取均值，日度 PortWatch 油轮计数与运力按周求和。月度宏观与 GFW 序列仅在假定可得日之后，才自月末向前填入。

Monthly remote-sensing products, including Flat anomalies and Deep Prithvi embeddings, are attached by as-of alignment and become eligible only after their conservative availability dates. The most recent eligible composite is then carried forward across subsequent Fridays, so the four-week input window may contain repeated monthly vectors.

月度遥感产品（包括 Flat 距平与 Deep Prithvi 嵌入）采用 as-of 对齐，仅在保守可得日已过之后才进入。最近一期合格合成再向后续周五沿用，因此四周输入窗中可能出现重复的月度向量。

Publication timing is represented by source- and product-specific fixed lag buffers implemented in the data builders rather than by observation-level release timestamps. One-week buffers are applied to EIA fundamentals and PortWatch flows, while monthly macroeconomic series, remote-sensing products and individual GFW AIS and SAR products receive longer, product-specific buffers. Exact lag constants and implementation scripts are reported in Appendix A.3. The downloaded series are currently available revised histories.

发布时间以数据构建脚本中按来源与产品设定的固定滞后缓冲表示，而不是按每条观测的官方发布时间戳。EIA 基本面与 PortWatch 流量施加一周缓冲，月度宏观序列、遥感产品以及 GFW 的各 AIS 与 SAR 产品则使用更长的产品专属缓冲。精确滞后常数与实现脚本见附录 A.3。所用下载为当前可得的修订后历史。

### 3.4.3 Missing data and quality control

### 3.4.3 缺失数据与质量控制

Optical-image availability varies across sites and months. During monthly compositing in Google Earth Engine, cloud probability and valid-observation counts are used to filter scenes and pixels before monthly medians are calculated. These quality indicators are not included as predictors.

光学影像在不同站点与月份的可得性并不相同。在 Google Earth Engine 月度合成中，云概率与有效观测计数用于在计算月度中位数之前筛选场景与像元。这些质量指标不作为预测变量进入模型。

On the resulting weekly calendar, mean coverage across the four optical indices is approximately 97 per cent, with eight of the eleven sites fully observed, while VIIRS night-time-light anomalies are fully observed. Site-level coverage rates and counts of independent monthly composites are reported in Appendix A.5.

在由此得到的周历上，四个光学指数的平均覆盖率约为 97%，其中十一个站点中有八个为满观测，VIIRS 夜光距平为满观测。站点级覆盖率与独立月度合成计数见附录 A.5。

After temporal alignment, remaining gaps in Flat predictors are forward-filled using past observations only. Residual leading gaps are then handled by feature family. Remote-sensing anomaly features are set to zero, denoting no anomaly, whereas PortWatch and other shipping-count variables are imputed using medians estimated within each training fold. Deep finance inputs receive the same past-only forward fill, but no leading gaps remain in the merged finance matrix, so the zero fallback is not invoked. After training-window scaling, unavailable Deep remote-sensing embeddings and residual gaps in shipping-graph tensors are set to zero. Remote-sensing embeddings additionally carry binary availability masks that exclude missing positions from encoder attention, whereas shipping-graph inputs have no corresponding node-level mask.

时间对齐之后，Flat 预测变量中的剩余缺口仅用过去观测向前填充。残留的前导缺口再按变量族分别处理。遥感距平特征置零，表示无异常。PortWatch 及其他航运计数字段则用各训练折内估计的中位数填补。Deep 金融输入采用相同的过去向向前填充，但合并后的金融矩阵中已无前导缺口，因此不会触发置零。训练窗缩放之后，不可得的 Deep 遥感嵌入与航运图张量中的残余缺口均置零。遥感嵌入另配二元可用掩码，使编码器注意力排除缺失位置，航运图输入则没有对应的节点级掩码。

## 3.5 Forecasting models



## 3.5 预测模型



### 3.5.1 Flat models



### 3.5.1 Flat 模型

Flat models implement early feature-level fusion. For a given information set, all available numeric features are concatenated into one weekly table, and the most recent four weeks are flattened into a single row for each forecast origin. Two learners are estimated on this table. Ridge is a linear model with L2 regularisation (Hoerl and Kennard, 1970) and serves as a transparent linear comparator.XGBoost is a non-linear gradient-boosted tree ensemble (Chen and Guestrin, 2016) that can capture nonlinearities and interactions not represented by Ridge, but likewise does not preserve modality-specific structure. Both models predict the one-week-ahead log return and then reconstruct price. Hyperparameter selection follows the time-ordered procedure described in Section 3.6, with exact search grids reported in Appendix C.

Flat 模型实现扁平特征融合。对给定信息集，将全部可用数值特征拼成一张周表，并在每个预测起点将最近四周压成一行。该表上估计两种学习器。Ridge 是带 L2 正则的线性模型（Hoerl and Kennard, 1970），作为一开始就合并特征的透明线性基线；XGBoost 是非线性梯度提升树集成（Chen and Guestrin, 2016），可捕捉 Ridge 错过的交互，但仍不保留各模态特有结构。二者均预测提前一周的对数收益，再还原价格。超参数选择遵循第 3.6 节的时序程序，精确搜索网格见附录 C。

### 3.5.2 Deep models



### 3.5.2 Deep 模型

Deep models encode each available modality separately and then combine the resulting representations. 
Deep 模型先对每个可用模态分别编码，再组合所得表征。下文三个编码器按输入、用途与输出说明。

**Finance encoder.** 

The input is the weekly financial block that constitutes S1 and is retained in S2–S4, including prices, inventories and macro-financial, oil-market and geopolitical indicators.
A causal temporal convolutional network (TCN; Bai, Kolter and Koltun, 2018) maps the four-week input sequence to one finance representation for each forecast origin, using only the current and earlier positions at each convolutional layer. TCNs have been used for short-horizon crude-price forecasting (Foroutan and Lahmiri, 2024).

**金融编码器。** 输入为周度金融时序块（S1），包括价格、库存、宏观经济与油市指标。这些序列是密集时间序列，编码器须在不使用未来周的前提下学习短期依赖。输出为该预测时点的一个金融表征。架构为因果时间卷积网络（TCN；Bai, Kolter and Koltun, 2018）。因果卷积避免序列内前瞻。TCN 已用于短期限原油价格预测（Foroutan and Lahmiri, 2024）。

**Remote-sensing encoder.** The input is monthly Sentinel-2 image-patch embeddings at the eleven AOIs, extracted with a frozen Prithvi-EO-2.0 model. Although Prithvi-EO-2.0 was pretrained on six-band NASA HLS imagery at 30 m, the Deep pathway uses Sentinel-2 Surface Reflectance Harmonized patches rather than HLS directly. The patches were adapted to the model’s input convention using bands B2, B3, B4, B8A, B11 and B12, standardised with the published per-band statistics, and bilinearly resampled to 224 × 224 before encoding. Site representations remain distinct until attention pooling rather than being collapsed into an early spatial average.Temporal and site attention operate over the four-week lookback and pool the site-specific embeddings into one remote-sensing representation for each forecast origin.


**遥感编码器。** 输入为十一个 AOI 上的月度 Sentinel-2 影像块嵌入，由冻结的 Prithvi-EO-2.0 模型提取。虽然 Prithvi-EO-2.0 以六波段 NASA HLS 30 m 影像预训练，Deep 路径使用的是 Sentinel-2 地表反射率和谐化影像块，而非直接使用 HLS。影像块按模型输入约定选取 B2、B3、B4、B8A、B11 与 B12，用公布的波段统计量标准化，再双线性重采样至 224 × 224 后编码。编码完成前保持站点可区分，避免过早把空间位置压成单一均值。输出为该预测时点的一个遥感表征，由时间与站点加权得到。架构为冻结嵌入，外加在四周回看窗上的时间与站点注意力。层设置见附录 C。

**Shipping encoder.** The input is the weekly seventeen-node shipping graph described in Section 3.3. A graph attention network with temporal encoding (GAT; Veličković et al., 2018) exchanges information across connected nodes over the four-week lookback and produces one shipping representation for each forecast origin. In the implementation, the two link classes are combined in a symmetrised weekly adjacency matrix with self-loops, so edge direction and type are not retained during message passing. Edge direction and type are therefore not retained during message passing. Node-level attention is inspected descriptively for RQ3 rather than interpreted as causal importance. Layer settings are reported in Appendix C.

**航运编码器。** 输入为第 3.3 节所述的周度十七节点航运图。港口、码头与咽喉相互连接，因此编码器保留这些关系，而不是把每个节点当成互不相关的一行计数。输出为该预测起点的一个航运表征。带时间编码的图注意力网络（GAT；Veličković et al., 2018）让相连节点在四周回看窗上交换信息。实现上，两类边进入同一个周度邻接矩阵，该矩阵被对称化并加入自环。对数变换后的航次数乘以可学习系数后加到注意力 logits 上，使更繁忙的航道可以获得更大权重。因此消息传递不保留边的方向与类型。节点级注意力供 RQ3 作描述性检查，不解释为因果重要性。层设置见附录 C。

### 3.5.3 Fusion mechanisms



### 3.5.3 融合机制

Fusion is applied only to the multimodal information sets S2–S4; S1 contains a single finance representation and does not need a fusion step. Three mechanisms are compared, with gated fusion as the main design and encoder concatenation and cross-attention as alternatives. 

Gated fusion and encoder concatenation operate on the pooled 32-dimensional modality representations. Under gated fusion, a small MLP maps the concatenated representations to one logit per available modality; softmax converts these logits into non-negative scalar weights that sum to one, and the fused representation is their weighted sum.Encoder concatenation joins the same vectors and uses an MLP to project the concatenated representation back to 32 dimensions without assigning explicit modality weights.Cross-attention instead accesses the pre-pooling remote-sensing site tokens and shipping node tokens, using the finance representation as the query. The fused representation is passed to a regression head trained by mean squared error on the one-week-ahead log return; price is then reconstructed as in the Flat models. Layer settings, the optimiser and search grids are in Appendix C.

融合仅用于多模态信息集 S2–S4；S1 只有一个金融表征，不需要融合。比较三种组合规则。编码器拼接用固定映射连接各模态摘要，不赋予随周变化的权重。门控融合为主要报告设计，它学习在每个预测起点给各可用模态多少权重。交叉注意力则让金融表征去关注遥感与航运的站点或节点细节。

门控融合与编码器拼接作用于池化后的 32 维模态表征。门控融合中，一个小型 MLP 将拼接后的表征映射为每个可用模态一个 logit，再经 softmax 转为非负且和为 1 的标量权重，融合表征为各模态表征的加权和。编码器拼接将同样的向量连接后，用固定 MLP 映回 32 维。交叉注意力则使用池化前的遥感站点 token 与航运节点 token，并以金融表征作为 query。融合表征输入回归头，以均方误差训练、预测提前一周的对数收益；价格还原方式与 Flat 相同。层设置、优化器与搜索网格见附录 C。

## 3.6 Estimation and validation



## 3.6 估计与验证



### 3.6.1 Expanding-window estimation and re-estimation



### 3.6.1 扩展窗估计与重估

With a four-week input window and a one-week forecast horizon, the 365 weekly observations yield 361 eligible input–target sequences. The first three observations cannot yet form a complete four-week input sequence, and the final week serves only as a target rather than a forecast origin. The first 104 eligible sequences form the initial estimation period and are not included in the evaluation metrics. The evaluation span covers 257 weeks from 22 January 2021 to 19 December 2025, with corresponding target dates from 29 January 2021 to 26 December 2025. Flat and Deep share this evaluation calendar.

A one-week-ahead forecast is produced at every origin t, using only information observable by that date. Models are estimated separately for each information set and learner at the first evaluation origin, and are re-estimated every 13 origins as the training window expands. Between scheduled re-estimations the fitted model and preprocessing parameters are held fixed; each origin still receives a new as-of input window. The final evaluation block contains 10 origins rather than 13. This produces 20 scheduled fits: 19 covering 13 origins each and a final fit covering 10 origins. Each estimation uses only input–target pairs whose targets were already observable at the estimation date. A realised target may enter a later re-estimation sample once it has become observable, but it never enters the sample used to generate its own forecast. All data-dependent preprocessing, including imputation and scaling, is estimated using the training sample available at the corresponding re-estimation date.

Figure 3.2 shows this schedule on the calendar. Figure 3.4 shows one re-estimation origin: the training fold, the inner validation weeks used at that fit, the four-week input window and the one-week-ahead target.

在四周输入窗口和提前一周预测期下，365 个周度观测可形成 361 个合法的“输入–目标”样本。最前面 3 个观测尚不足以构成完整的四周输入序列，而最后一周只作为目标、不能再作为预测起点。前 104 个合法样本为初始估计期，不纳入评估指标。评估跨度为 2021 年 1 月 22 日至 2025 年 12 月 19 日的 257 周，相应目标日期为 2021 年 1 月 29 日至 2025 年 12 月 26 日。Flat 与 Deep 共享这一评估日历。

每个预测起点 t 都产生一次提前一周预测，且仅使用截至该日可观测的信息。各信息集与各学习器在第一个评价起点分别估计，并随训练窗扩展每隔 13 个起点重估一次。两次预定重估之间，拟合模型与预处理参数保持固定；各起点仍使用新的 as-of 输入窗。最后一个评价块含 10 个起点，而非 13 个。由此共 20 次预定拟合：19 次各覆盖 13 个起点，最后一次覆盖 10 个起点。每次估计仅使用目标在该估计日之前已可观测的“输入–目标”样本。已实现的目标在可观测之后可以进入之后的重估样本，但不会进入生成其自身预测的估计样本。包括填补与缩放在内的所有数据依赖型预处理，均仅使用相应重估日期可得的训练样本估计。

图 3.2 给出该安排在日历上的视图。图 3.4 展示一次重估起点：训练折、该次拟合所用的内部验证周、四周输入窗与提前一周的目标。

Figure 3.4

**Figure 3.4 — One re-estimation origin: the training fold with nested inner validation weeks, the four-week input window and the one-week-ahead target. Between re-estimations, the fitted model and preprocessing parameters remain fixed while the as-of input window advances.**

**图 3.4 — 一次重估起点：含嵌套内部验证周的训练折、四周输入窗与提前一周目标。两次重估之间，拟合模型与预处理参数保持固定，as-of 输入窗口则随预测起点向前推进。**

### 3.6.2 Hyperparameter selection and fixed model settings



### 3.6.2 超参数选择与固定模型设定

Both families respect the same temporal separation between estimation and evaluation, but they are tuned differently. Flat models re-select Ridge and XGBoost hyperparameters at each scheduled re-estimation on an inner time-ordered validation segment, then refit on the full estimation sample available at that date. Deep models use a configuration fixed in advance on design grounds: a four-week lookback to match Flat, and a common 32-dimensional latent size across encoders. Later sensitivity checks on the evaluation sample are descriptive and do not change this specification (Appendix B). At each Deep re-estimation, inner validation is used for early stopping, and the checkpoint with the lowest validation loss is retained for the subsequent forecast block. Unlike Flat models, Deep models are not subsequently refitted on the inner-validation weeks. Grids, locked settings and early-stopping details are in Appendix C.

两族遵守相同的估计–评价时间分隔，但调参方式不同。Flat 模型在每次预定重估时，于内部时序验证段上重新选择 Ridge 与 XGBoost 超参数，再在该日可得的全部估计样本上重新拟合。Deep 模型的主配置按设计原则事先固定：四周回看窗与 Flat 对齐，各编码器共用 32 维潜在表征。随后在评价样本上的敏感性检查只作描述，不改变这一配置（附录 B）。每次 Deep 重估用内部验证做早停，并保留验证损失最低的 checkpoint 用于随后的预测块。与 Flat 模型不同，Deep 模型此后不再将内部验证周纳入重新拟合。网格、锁定设定与早停细节见附录 C。

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

