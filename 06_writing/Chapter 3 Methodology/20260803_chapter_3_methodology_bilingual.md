# Chapter 3 — Methodology *(~3,200)*

# 第 3 章 — 方法 *(约 3,200 词)*

## 3.1 Research design

## 3.1 研究设计

This chapter sets out how the study answers the research questions in Section 1.2. In brief, every learned forecast is judged against a simple no-change benchmark in which next week’s Brent price equals this week’s price. The study then asks whether remote sensing and shipping add useful information beyond financial time series, and whether modelling those inputs as one weekly table differs from encoding each data type separately before combining them. All comparisons use the same weekly forecast dates, sample window and evaluation rules, so that changes in the data can be separated from changes in how the data are modelled.

本章说明研究如何回答第 1.2 节的研究问题。简言之，每个学习到的预测都对照一个简单的不变预测基准，即下周 Brent 价格等于本周价格。研究再问遥感与航运是否在金融时序之外仍提供有用信息，以及把这些输入压成一张周表建模是否不同于先按数据类型分别编码再组合。所有比较共用同一周度预测日、样本窗口与评估规则，从而把“用了什么数据”的变化，与“如何建模”的变化分开。

The no-change benchmark is denoted M0. At each forecast origin t, M0 sets the one-week-ahead Brent price forecast equal to the current weekly price


\hat{P}_{t+1|t}=P_t.


M0 needs no parameter estimation and contains no predictors. It is a reference forecast, not one of the information sets below. Every learned model is compared with M0 on the same evaluation sample. A model improves on M0 when its out-of-sample RMSE is lower. Once models predict log returns and then reconstruct prices, M0 is the same as forecasting a zero return.

不变预测基准记为 M0。在每个预测起点 t，M0 将提前一周的 Brent 价格预测设为当前周价格


\hat{P}_{t+1|t}=P_t.


M0 无需参数估计，也不含预测变量。它是参照预测，不属于下文的信息集。每个学习模型都在同一评价样本上与 M0 比较。样本外 RMSE 更低，即相对 M0 有改善。在先预测对数收益再还原价格时，M0 与预测收益为零是同一回事。

The predictors are organised into four information sets. M1 uses financial time series only (financial, macroeconomic and oil-market series). M2 adds remote sensing to M1; M3 adds shipping to M1; and M4 adds both. M2 and M3 are parallel additions to M1, not successive steps on one ladder; M4 combines both additions.

预测变量组织为四个信息集。M1 仅用金融时序（金融、宏观与油市序列）。M2 在 M1 上加遥感；M3 在 M1 上加航运；M4 两者都加。M2 与 M3 是对 M1 的平行扩展，不是一条梯子上的先后步骤；M4 合并两支扩展。

**Table 3.1 — Information sets**

**表 3.1 — 信息集**


| Set | Content                                                                     |
| --- | --------------------------------------------------------------------------- |
| M1  | Financial time series only (financial, macroeconomic and oil-market series) |
| M2  | M1 + remote sensing                                                         |
| M3  | M1 + shipping                                                               |
| M4  | M1 + remote sensing + shipping                                              |



| 集合  | 内容                |
| --- | ----------------- |
| M1  | 仅金融时序（金融、宏观与油市序列） |
| M2  | M1 + 遥感           |
| M3  | M1 + 航运           |
| M4  | M1 + 遥感 + 航运      |


Comparing M2 with M1 measures the contribution of remote sensing when added alone. Comparing M3 with M1 measures the contribution of shipping. Comparing M4 with M1 evaluates their joint contribution. Two further comparisons ask whether each source still helps once the other is already included. M4 versus M3 tests remote sensing given shipping, and M4 versus M2 tests shipping given remote sensing. All comparisons keep the same one-week horizon and Friday weekly calendar.

将 M2 与 M1 比较，度量单独加入遥感的贡献；M3 对 M1 度量航运的贡献；M4 对 M1 评估二者合用的贡献。另两组比较问在已有另一数据源时这一源是否仍有帮助。M4 对 M3 看已有航运时的遥感，M4 对 M2 看已有遥感时的航运。所有比较保持同一提前一周预测期与周五周历。

Two model families are applied to these information sets. The Flat family puts all selected predictors into one weekly table—stacking recent weeks into a single row—and fits Ridge and XGBoost. This early joining of features is called flat feature fusion. The Deep family keeps each data type separate at first. Financial series, remote-sensing imagery and shipping-network inputs each pass through their own encoder, and the outputs are then combined. The main Deep design learns how much weight to give each data type (gated fusion). Simple joining of the encoder outputs, and an attention-based alternative, are kept as comparisons. Flat and Deep share sites and forecast dates, but the remote-sensing products they use are not identical. That difference is treated as part of the Flat–Deep contrast.

两套模型族应用于这些信息集。Flat 族把所选预测变量压成一张周度表——把最近几周叠成一行——再拟合 Ridge 与 XGBoost。这种一开始就合并特征的做法，称为扁平特征融合。Deep 族则先按数据类型分开处理。金融序列、遥感影像与航运网络输入各自经过自己的编码器，再把输出组合起来。Deep 的主设计学习给各类数据多少权重（门控融合）。编码器输出的简单拼接，以及一种基于注意力的备选，作为对照。Flat 与 Deep 共享站点与预测日，但所用遥感产品并不完全相同。该差异视为 Flat–Deep 对照的一部分。

Paired Flat–Deep comparisons measure the overall difference between two modelling strategies. One fits a single weekly table directly. The other encodes each data type separately and then combines the results. The information set, forecast dates and evaluation sample are held constant. The two families also differ in model class and capacity, so these comparisons are not read as isolating the fusion method alone. To assess fusion itself, the Deep family compares simple concatenation, gated fusion and cross-attention while keeping the encoders and input data fixed. These two layers answer RQ2. The first contrasts Flat versus Deep on the same information set (for example M3_Flat versus M3_Deep). The second contrasts fusion variants within Deep.

Flat 与 Deep 的配对比较，衡量的是两种整体建模方式的差异。一种是一张表直接建模，另一种是先分类型编码再组合。比较时信息集、预测日与评价样本保持不变。但两族所用模型类型与容量也不同，因此这一比较不能单独说明“融合方式”本身谁更好。要评估融合方式，应在 Deep 族内部比较简单拼接、门控融合与交叉注意力，此时编码器与输入数据保持不变。这两层对应 RQ2。第一层在同一信息集上比较 Flat 与 Deep（例如 M3_Flat 对 M3_Deep）。第二层在 Deep 内比较融合方式。

The research questions map onto the design as follows. RQ1 uses the M1–M4 comparisons within each model family, together with the comparison of every learned forecast against M0. The M4–M3 and M4–M2 contrasts ask whether either added source still helps once the other is present.

研究问题与设计的对应如下。RQ1 依靠各族内的 M1–M4 比较，以及每个学习预测相对 M0 的比较。M4–M3 与 M4–M2 再问在已有另一数据源时新加的源是否仍有帮助。

RQ3 is restricted to Deep models that improve on M0 by the pre-defined criterion. Those models are identified from the results; they are not chosen in advance by name. For them, the study reports how much weight the model places on finance, remote sensing and shipping, and which sites or network nodes receive more attention under different market conditions. These quantities describe what the model relies on. They are not treated as proof of causal importance.

RQ3 仅限于按预先设定准则相对 M0 有改善的 Deep 模型。入选模型按结果判定，不事先按名称选定。对这些模型，报告它给金融、遥感与航运各多少权重，以及不同市场条件下哪些站点或网络节点更受关注。这些量说明模型依赖什么，不作因果重要性的证明。

Figure 3.1 summarises the design. M0 is the no-change reference. M1 branches into M2 (plus remote sensing), M3 (plus shipping) and M4 (both). Flat and Deep are estimated and evaluated on the same information sets under a shared expanding-window procedure.

图 3.1 概括整体设计。M0 为不变预测参照。M1 分支为 M2（加遥感）、M3（加航运）与 M4（二者）。Flat 与 Deep 在相同信息集上估计，并共用同一扩展窗评估程序。

*[Figure 3.1 — Research design flowchart. M0 benchmark; M1→M2/M3/M4 branching; paired Flat vs Deep; link to expanding-window evaluation.]*

*[图 3.1 — 研究设计流程图。M0 基准；M1→M2/M3/M4 分支；配对 Flat 与 Deep；衔接扩展窗评估。]*

## 3.2 Prediction target and timeline



## 3.2 预测目标与时间轴

Let P_t denote the last available daily Brent spot-price observation in week t, where each week ends on Friday, measured in US dollars per barrel. The quantity reported in the results is the one-week-ahead price P_{t+1}. Models are not trained directly on the price level. They predict the one-week logarithmic return


r_{t+1}=\log\left(\frac{P_{t+1}}{P_t}\right)


and reconstruct the price forecast as


\hat{P}*{t+1|t}=P_t\exp\left(\hat{r}*{t+1|t}\right).


Log returns are used to reduce the strong persistence in the price level and to express the forecasting task in terms of proportional weekly changes. RMSE, MAE and skill versus M0 are computed from the reconstructed price forecasts. Directional accuracy is reported separately as an auxiliary statistic based on the sign of the predicted and observed returns. Under this mapping, the no-change benchmark \hat{P}*{t+1|t}=P_t is exactly the same as forecasting a zero return \hat{r}*{t+1|t}=0.

令 P_t 表示第 t 个周五截止周内最后一个可获得的 Brent 现货价格日度观测值，单位为美元/桶。结果中报告的量是提前一周的价格 P_{t+1}。模型不直接在价格水平上训练，而是预测一周对数收益


r_{t+1}=\log\left(\frac{P_{t+1}}{P_t}\right),


并按


\hat{P}*{t+1|t}=P_t\exp\left(\hat{r}*{t+1|t}\right)


重构价格预测。使用对数收益是为了减弱价格水平序列的强持续性，并将预测任务表示为周度比例变化。RMSE、MAE 以及相对 M0 的 skill 均根据重构后的价格预测计算。方向准确率则根据预测收益与实际收益的符号另行计算，并仅作为辅助统计量报告。在此对应关系下，不变预测基准 \hat{P}*{t+1|t}=P_t 与预测收益为零 \hat{r}*{t+1|t}=0 完全一致。

All series are organised on a Friday-ending weekly calendar. The modelling window covers 2019–2025 and provides a common weekly index of 365 observations (4 January 2019 to 26 December 2025). Flat models use a merged weekly feature table on this index. Deep models use the same dates, but keep financial, remote-sensing and shipping inputs in their own sequence or graph form rather than one shared table. The first 104 weeks are reserved for initial estimation. Three further weeks are needed to form the first four-week input sequence, and the final week is excluded because P_{t+1} is unavailable. This leaves 257 forecast origins for evaluation, from 22 January 2021 to 19 December 2025. At each origin t, forecasts may use only information that was actually available at that forecast date.

全部序列按周五截止的周历组织。建模窗口覆盖 2019–2025 年，提供含 365 个观测的共同周索引（2019 年 1 月 4 日至 2025 年 12 月 26 日）。Flat 模型在该索引上使用合并后的周度特征表。Deep 模型使用相同日期，但把金融、遥感与航运输入各自保留为序列或图形式，而不是并成一张共享表。前 104 周预留作初始估计。再留出 3 周以形成第一个四周输入序列，并因无法获得 P_{t+1} 而排除最后一周。由此剩余 257 个纳入评估的预测时点，起止为 2021 年 1 月 22 日至 2025 年 12 月 19 日。在每个起点 t，预测只能使用该预测时点实际可获得的信息。

## 3.3 Geographic scope and monitoring sites



## 3.3 地理范围与监测站点

Because the prediction target is the global Brent benchmark rather than a local physical cargo price at a single terminal, the study does not use one contiguous study region. Spatial information instead comes from eleven oil-infrastructure monitoring sites and six maritime chokepoints. Together they cover major supply, transit and demand locations in the international oil system. Figure 3.2 places these sites and chokepoints on a world map. Full site names, coordinates, patch sizes and graph edge definitions are in Appendix A.

由于预测对象是全球 Brent 基准价格，而非单一码头的现货成交价格，本研究不采用一块连续的地理研究区。空间信息来自十一个油气基础设施监测站点与六个航运咽喉，共同覆盖国际石油体系中的主要供给、中转与需求区位。图 3.2 在世界地图上标出这些站点与咽喉。完整站名、坐标、裁剪范围与图边定义见附录 A。

The eleven sites are ports, refineries and export terminals chosen for infrastructure capacity, geographic and supply-chain coverage, and observability in the available satellite products. In the Flat pathway, remote-sensing features are summarised inside a 5-km circular buffer around each site. In the Deep pathway, image patches are cut around each site. Patch size follows facility type and local spatial constraints. Ports use larger patches, refineries intermediate ones, and terminals smaller ones.

十一个站点为港口、炼厂与出口码头，选择时兼顾基础设施产能、地理与供应链覆盖，以及在可用卫星产品中的可观测性。Flat 路径在每个站点周围 5 km 圆形缓冲区内汇总遥感特征。Deep 路径则按站点裁剪影像块。裁剪大小随设施类型与当地空间条件变化。港口较大，炼厂居中，码头较小。

The shipping network adds six chokepoints to the eleven sites. They are the Strait of Hormuz, the Suez Canal, the Strait of Malacca, Bab el-Mandeb, the Panama Canal and the Cape of Good Hope. This gives a weekly network with seventeen nodes. Two kinds of link are used. First, directed site-to-site links record observed voyages between the eleven AOIs from Global Fishing Watch port-visit sequences. The weekly link weight is the voyage count for each origin–destination pair, so these links change from week to week. Second, fixed site–chokepoint links connect each AOI to the chokepoint(s) on its main oil-trade corridor. These links are set in advance. They are not inferred from weekly vessel tracks or nearest-neighbour distance. PortWatch and Automatic Identification System (AIS) measures enter mainly as node attributes rather than as pairwise links.

航运网络在十一个站点之外加入六个咽喉，分别为霍尔木兹海峡、苏伊士运河、马六甲海峡、曼德海峡、巴拿马运河与好望角。由此形成含十七个节点的周度网络。边分为两类。第一类是站点之间的有向连接，来自 Global Fishing Watch 港口访问序列所记录的航次。每周边权为各起点–终点对的航次数，因此这些边随周变化。第二类是站点与咽喉之间的固定连接，按各站主贸易走廊事先指定，而非由周度船舶轨迹或最近邻距离推断。PortWatch 与船舶自动识别系统（AIS）测度主要作为节点属性进入，而不是成对连接。

Flat and Deep use the same underlying port and chokepoint observations and the same weekly forecast dates. Flat turns them into tabular predictors. Deep keeps the node structure and the links between nodes.

Flat 与 Deep 使用相同的港口与咽喉底层观测，并共享同一周度预测日。Flat 将其整理为表格预测变量。Deep 则保留节点结构及节点之间的连接。

*[Figure 3.2 — World map of 11 oil-infrastructure AOIs and 6 maritime chokepoints used for remote-sensing and shipping inputs.]*

*[图 3.2 — 遥感与航运输入所用的 11 个油气基础设施 AOI 与 6 个航运咽喉世界分布图。]*

## 3.4 Data sources



## 3.4 数据来源

Three data blocks enter the design. They are financial time series (financial, macroeconomic and oil-market series), satellite remote sensing, and maritime shipping. Flat and Deep share the same Friday calendar and the same study sites and shipping-network scope. The product used for each block before it enters a model may differ (Table 3.2). For Flat models, the predictors are those in the merged weekly feature table. Deep models use the same weekly dates, but keep each block in its own input form.

三类数据块进入设计，分别为金融时序（金融、宏观与油市序列）、卫星遥感与航运。Flat 与 Deep 使用同一周五日历与相同的研究站点/航运网络范围。各块在进入模型前所用产品可以不同（见表 3.2）。Flat 所用预测变量以合并周度特征表为准。Deep 使用相同周日期，但把各块保留为各自的输入形式。

**Table 3.2 — Datasets, variables and sources**

**表 3.2 — 数据集、变量与来源**


| Modality                   | Dataset / product                                      | Key variables                                                               | Source                                       |
| -------------------------- | ------------------------------------------------------ | --------------------------------------------------------------------------- | -------------------------------------------- |
| Financial time series (M1) | Oil-market and macro weekly series                     | Prices, inventories, production, interest rates, GPR and related indicators | EIA, FRED, Yahoo Finance and related sources |
| Remote sensing (Flat)      | Sentinel-2 optical indices and VIIRS night-time lights | Site-level anomalies at 11 AOIs (NDVI, NDWI, NDBI, BSI; NTL)                | Sentinel-2; VIIRS                            |
| Remote sensing (Deep)      | Frozen Prithvi-EO-2.0 embeddings                       | Monthly Sentinel-2 image-patch embeddings at the same 11 AOIs (no VIIRS)    | Prithvi-EO-2.0 / Sentinel-2                  |
| Shipping (Flat)            | PortWatch and AIS tabular features                     | Port and chokepoint tanker flows; vessel-activity features                  | IMF PortWatch; AIS                           |
| Shipping (Deep)            | Same sources, represented as a graph                   | Weekly heterogeneous graph with 17 nodes (11 AOIs and 6 chokepoints)        | PortWatch; AIS                               |



| 模态       | 数据集 / 产品                  | 关键变量                                       | 来源                          |
| -------- | ------------------------- | ------------------------------------------ | --------------------------- |
| 金融时序（M1） | 油市与宏观周序列                  | 价格、库存、产量、利率、GPR 及相关指标                      | EIA、FRED、Yahoo Finance 等    |
| 遥感（Flat） | Sentinel-2 光学指数与 VIIRS 夜光 | 11 个 AOI 的站点级异常（NDVI、NDWI、NDBI、BSI；NTL）    | Sentinel-2；VIIRS            |
| 遥感（Deep） | 冻结 Prithvi-EO-2.0 嵌入      | 同一 11 个 AOI 的月度 Sentinel-2 影像块嵌入（不含 VIIRS） | Prithvi-EO-2.0 / Sentinel-2 |
| 航运（Flat） | PortWatch 与 AIS 表格特征      | 港口与咽喉油轮流量；船舶活动特征                           | IMF PortWatch；AIS           |
| 航运（Deep） | 同源数据，图表示                  | 含 17 个节点的周度异质图（11 个 AOI 与 6 个咽喉）           | PortWatch；AIS               |


The financial block is assembled from weekly oil-market and macro-financial series from the US Energy Information Administration (EIA), Federal Reserve Economic Data (FRED), Yahoo Finance and related scholarly indicators. The series include crude prices and spreads, inventories, production and refinery activity, volatility and risk measures, interest rates, exchange rates, futures-based oil indicators and geopolitical risk. This block is M1 before remote sensing or shipping is added. In the Deep pathway it is treated as one input stream for the finance encoder, even though the predictors extend beyond prices alone.

金融块由美国能源信息署（EIA）、联邦储备经济数据（FRED）、Yahoo Finance 及相关学术指标的周度油市与宏观金融序列汇总而成，包括原油价格与价差、库存、产量与炼厂活动、波动与风险度量、利率、汇率、基于期货的油市指标以及地缘政治风险。该块即加入遥感或航运前的 M1。在 Deep 路径中，它作为金融编码器的一路输入，尽管预测变量远不止价格本身。

Remote-sensing inputs are observed over the eleven AOIs. Flat and Deep share these sites but use different products from a common Sentinel-2 optical source family. Flat remote sensing uses monthly Sentinel-2 optical indices (NDVI, NDWI, NDBI and BSI) together with VIIRS night-time lights, converted to site-level anomalies. Deep remote sensing uses frozen Prithvi-EO-2.0 embeddings extracted from monthly Sentinel-2 image patches at the same AOIs and excludes VIIRS. Early Deep trials that included VIIRS night-time lights were noisy and added little useful signal; keeping them worsened performance, so VIIRS was dropped from the Deep pathway. Systematic numerical ablations from those early trials were not retained. This choice is also consistent with evidence that night-time lights capture cross-sectional brightness differences better than within-site temporal variation (Small, 2021). The reported Deep pathway therefore uses Sentinel-2 image embeddings only. Shared AOIs keep spatial coverage matched across pathways. Differences in product and representation form part of the Flat–Deep contrast and limit a pure architecture comparison on identical remote-sensing features.

遥感输入观测于十一个 AOI。Flat 与 Deep 共享这些站点，但使用来自共同 Sentinel-2 光学源族的不同产品。Flat 遥感采用月度 Sentinel-2 光学指数（NDVI、NDWI、NDBI 与 BSI）及 VIIRS 夜光，并转换为站点级异常。Deep 遥感采用同一站点上月度 Sentinel-2 影像块提取的冻结 Prithvi-EO-2.0 嵌入，且不含 VIIRS。早期 Deep 试验曾纳入 VIIRS 夜光，但噪声大、有效信息少，保留后表现变差，因此从 Deep 路径剔除。这些早期试验的系统数值消融结果未保留。这一选择也与文献一致：夜光更擅长跨地点亮度差异，站点内时间变异较难解释（Small, 2021）。故正文报告的 Deep 路径仅使用 Sentinel-2 影像嵌入。共享 AOI 使两条路径的空间覆盖保持匹配。产品与表征差异构成 Flat–Deep 对照的一部分，并限制在完全相同遥感特征上的纯架构比较。

Shipping inputs combine IMF PortWatch measures of chokepoint and port tanker flows with AIS-derived vessel-activity indicators for the network in Section 3.3. In the Flat pathway these signals enter as weekly table features. In the Deep pathway they enter as the seventeen-node network already described. In both pathways, shipping is treated as a proxy for physical trade and congestion rather than as a direct measure of next week’s price.

航运输入结合 IMF PortWatch 的咽喉与港口油轮流量测度，以及第 3.3 节网络的 AIS 衍生船舶活动指标。Flat 路径中，这些信号以周度表格特征进入。Deep 路径中，它们进入前文所述的十七节点网络。两条路径都将航运视为实物贸易与拥堵的代理，而非下周价格的直接量测。

**Ethical considerations.** The study uses secondary aggregate data only and does not involve human participants. It was approved under the UCL low-risk ethics process. All datasets are used under their published research terms of use.

**伦理考量。** 本研究仅使用二手汇总数据，不涉及人类参与者，并已按 UCL 低风险伦理流程获批。所有数据集均按其公开研究使用条款使用。

## 3.5 Temporal alignment, lags, missingness



## 3.5 时间对齐、滞后期与缺失

All series are aligned to the Friday-ending weekly calendar. Predictors enter only after their real publication time, so the model never uses future information at any forecast origin. Release lags differ across sources. EIA and PortWatch series typically become available with a lag of about one week, while slower monthly series require longer buffers before they are allowed to enter. Missingness is handled differently in the two pathways. Flat models fill missing values using only past observations within the available history. Deep models keep an explicit missing marker for absent modalities or sites instead of filling them in silently, so the model can see what was unavailable at that forecast date.

全部序列对齐到周五截止周历。预测变量仅在真实发布时刻之后进入，因此任一预测起点都不会使用未来信息。不同来源的发布滞后不同。EIA 与 PortWatch 序列通常约一周后可用，更慢的月度序列则需要更长缓冲才允许进入。缺失处理在两条路径中不同。Flat 模型仅用可得历史中的过去观测填补缺失。Deep 模型对当时缺失的模态或站点保留显式缺失标记，而不是悄悄填掉，从而使模型知道该预测日缺少什么。

## 3.6 Flat models



## 3.6 Flat 模型

Flat models implement flat feature fusion. For a given information set, all available numeric features are concatenated into one weekly table, and the most recent four weeks are flattened into a single row for each forecast origin. Two learners are estimated on this table. Ridge is a linear model with L2 regularisation (Hoerl and Kennard, 1970) and serves as a transparent linear baseline that combines features at the outset. XGBoost is a non-linear gradient-boosted tree ensemble (Chen and Guestrin, 2016) that can capture interactions missed by Ridge, but still does not preserve modality-specific structure. Regularised linear and tree-based learners are both common in short-horizon oil-price forecasting with large predictor sets (Costa et al., 2021; Yılmaz and Zehir, 2026); they are used here as Flat baselines rather than as a claim that either algorithm is universally optimal. Both models predict the one-week-ahead log return and then reconstruct price. Hyperparameters are chosen inside each training fold on past validation weeks only. Exact search grids are in Appendix C.

Flat 模型实现扁平特征融合。对给定信息集，将全部可用数值特征拼成一张周表，并在每个预测起点将最近四周压成一行。该表上估计两种学习器。Ridge 是带 L2 正则的线性模型（Hoerl and Kennard, 1970），作为一开始就合并特征的透明线性基线；XGBoost 是非线性梯度提升树集成（Chen and Guestrin, 2016），可捕捉 Ridge 错过的交互，但仍不保留各模态特有结构。正则化线性与树模型在大预测变量集的短期限油价预测中均常见（Costa et al., 2021; Yılmaz and Zehir, 2026）；此处用作 Flat 基线，而非声称任一算法普遍最优。二者均预测提前一周的对数收益，再还原价格。超参数仅在各训练折内、用过去验证周选择。精确搜索网格见附录 C。

## 3.7 Deep models



## 3.7 Deep 模型

Deep models use the same information sets, Friday calendar and validation protocol as Flat. The difference is how inputs are represented and combined, not the forecast target. Each available modality is first turned into a fixed-size representation; those representations are then combined into one forecast. The three encoders are described below by input, purpose and output.

Deep 模型与 Flat 使用相同的信息集、周五日历与验证协议。差异在输入如何被表征与组合，而非预测目标。每个可用模态先转为固定维度的表征，再把这些表征组合成一次预测。下文三个编码器按输入、用途与输出说明。

**Finance encoder.** The input is the weekly financial time series block (M1), including prices, inventories, macro and oil-market indicators. These series are dense temporal sequences, so the encoder must learn short-run dependence without using future weeks. The output is one finance representation for the forecast origin. The architecture is a causal temporal convolutional network (TCN; Bai, Kolter and Koltun, 2018). Causal convolutions prevent look-ahead within the sequence, and TCNs have been competitive for short-horizon crude-price forecasting relative to several deep and tree baselines (Foroutan and Lahmiri, 2024).

**金融编码器。** 输入为周度金融时序块（M1），包括价格、库存、宏观与油市指标。这些序列是密集时间序列，编码器须在不使用未来周的前提下学习短期依赖。输出为该预测时点的一个金融表征。架构为因果时间卷积网络（TCN；Bai, Kolter and Koltun, 2018）。因果卷积避免序列内前瞻，且相对多种深度与树基线，TCN 在短期限原油价格预测中具有竞争力（Foroutan and Lahmiri, 2024）。

**Remote-sensing encoder.** The input is monthly Sentinel-2 image-patch embeddings at the eleven AOIs, extracted with a frozen Prithvi-EO-2.0 model (VIIRS night-time lights are excluded). Sites are kept distinct until after encoding, so spatial location is not collapsed into a single early average. The output is one remote-sensing representation for the forecast origin, formed by weighting across time and sites. The architecture uses frozen embeddings plus temporal and site attention.

**遥感编码器。** 输入为十一个 AOI 上的月度 Sentinel-2 影像块嵌入，由冻结的 Prithvi-EO-2.0 模型提取（不含 VIIRS 夜光）。编码完成前保持站点可区分，避免过早把空间位置压成单一均值。输出为该预测时点的一个遥感表征，由时间与站点加权得到。架构为冻结嵌入，外加时间与站点注意力。

**Shipping encoder.** The input is the weekly seventeen-node shipping network from Section 3.3. Shipping information is relational because ports and corridors are linked, so a graph model fits better than a flat row of counts. The output is one shipping representation for the forecast origin. The architecture is a graph attention network (GAT; Veličković et al., 2018) with temporal encoding. Graph neural networks have been used to model crude-oil and vessel-traffic networks as relational, time-varying processes (Ouyang et al., 2022; Liang et al., 2022). GAT is used here because neighbour weights fit a sparse port–chokepoint network and later support site-level interpretation (RQ3). Layer settings are in Appendix C.

**航运编码器。** 输入为第 3.3 节的周度十七节点航运网络。航运信息具有关系结构，因为港口与走廊相互连接，因此图模型比一行扁平计数更合适。输出为该预测时点的一个航运表征。架构为带时间编码的图注意力网络（GAT；Veličković et al., 2018）。图神经网络已用于将原油与船舶交通网络建模为关系性、时变过程（Ouyang et al., 2022; Liang et al., 2022）。此处采用 GAT，是因为邻居权重适合稀疏港口–咽喉网络，并便于后续站点级解释（RQ3）。层设置见附录 C。

**Fusion (RQ2).** Once each available modality has a representation, three ways of combining them are compared. Simple concatenation joins the representations without adaptive weighting and serves as a control. Gated fusion is the main reported design. It learns how much weight to give each modality. Cross-attention is retained as an advanced alternative that lets modalities attend to one another. The fused representation is mapped to the same return and price target as Flat. Training details are in Appendix C.

**融合（RQ2）。** 各可用模态得到表征后，比较三种组合方式。简单拼接在无自适应加权下连接表征，作为对照。门控融合是主要报告设计，它学习给各模态多少权重。交叉注意力保留为进阶备选，允许模态相互关注。融合表征映射到与 Flat 相同的收益与价格目标。训练细节见附录 C。

## 3.8 Hyperparameter selection



## 3.8 超参数选择

Hyperparameters are selected under a shared protocol so that Flat–Deep comparisons remain fair. For Flat models, tuning uses only past validation weeks inside each training fold. For Deep models, searching the full architecture at every fold is too costly. A limited search is run first, then one main configuration is fixed for the primary results. Sensitivity checks follow. Exact grids and layer settings are in Appendix C.

超参数在共享协议下选择，以使 Flat–Deep 比较保持公平。Flat 模型仅在各训练折内、用过去验证周调参。Deep 模型若在每一折都完整搜索架构成本过高，故先做有限搜索，再固定主配置作主要结果；随后做敏感性检查。细节见附录 C。

## 3.9 Validation protocol



## 3.9 验证协议

Evaluation uses an expanding window. At each forecast origin the model is trained only on past weeks and then produces a one-week-ahead forecast. This design prevents the use of future information in training or preprocessing. The first 104 weeks form the initial estimation and validation period and are not included in the evaluation metrics. Forming the first four-week input sequence requires three additional weeks before the first evaluated origin. Thereafter models are refit every 13 weeks. The common evaluation span covers 257 weeks from 22 January 2021 to 19 December 2025. Any scaling or filtering is fit on the training period only. Flat and Deep share the same evaluation calendar, so architecture comparisons hold the evaluation design fixed.

评估采用扩展窗。在每个预测起点，模型仅用过去周训练，再给出提前一周预测。该设计避免在训练或预处理中使用未来信息。前 104 周为初始估计与验证期，不纳入评估指标。形成第一个四周输入序列还需额外 3 周，之后才进入第一个纳入评估的预测时点。此后每 13 周重拟合。共同评估跨度为 2021 年 1 月 22 日至 2025 年 12 月 19 日的 257 周。任何缩放或过滤仅在训练期内拟合。Flat 与 Deep 共享同一评估日历，从而使架构比较在固定评估设计下进行。

Figure 3.3 shows this expanding-window design.

图 3.3 展示该扩展窗设计。

*[Figure 3.3 — Expanding-window evaluation flowchart.]*

*[图 3.3 — 扩展窗评估流程图。]*

## 3.10 Evaluation, tests, interpretability



## 3.10 评估、检验与可解释性

Primary metrics are computed on reconstructed prices. Every comparison reports RMSE and MAE. Directional accuracy is retained only as an auxiliary measure. Relative performance versus M0 is summarised by RMSE skill—the percentage improvement in RMSE relative to M0—reported as a percentage in the result tables.

\[
\mathrm{Skill}=100\times\left(1-\frac{\mathrm{RMSE}_{\mathrm{model}}}{\mathrm{RMSE}_{\mathrm{M0}}}\right).
\]

Skill greater than zero means the model beats M0 on RMSE. Skill equal to zero matches M0. Skill less than zero is worse than M0.

The study reports both absolute skill versus M0 and incremental value versus M1. Statistical tests are chosen by the type of comparison, not by the size of the modality set alone. Adding remote sensing or shipping enlarges the information set, but that does not by itself make two forecasts nested for testing. When one forecast specification is nested in another—for example Ridge M1 versus Ridge M2, M3 or M4 under the same learner—Clark–West (2007) is used to test whether the larger model improves mean squared prediction error. When the comparison is not nested—for example Flat versus Deep, or XGBoost versus a Deep setting that changes hyperparameters or architecture—Diebold–Mariano (1995) is used to test equal predictive accuracy. A small-sample adjustment is noted where relevant. Every comparison also reports RMSE and MAE differences versus M0 and, where relevant, versus M1.

Interpretability diagnostics are applied only to specifications that improve on M0. The main cases are Deep M3 and, where relevant, Deep M4. The diagnostics report modality gate weights together with site or node attention.

主指标在重构价格上计算。每次比较均报告 RMSE 与 MAE。方向准确率仅作辅助度量。相对 M0 的表现以 RMSE skill（相对 M0 的 RMSE 百分比改善）汇总，并在结果表中以百分比报告。

\[
\mathrm{Skill}=100\times\left(1-\frac{\mathrm{RMSE}_{\mathrm{model}}}{\mathrm{RMSE}_{\mathrm{M0}}}\right).
\]

Skill 大于零表示模型在 RMSE 上优于 M0。等于零与 M0 持平。小于零则差于 M0。

研究同时报告相对 M0 的绝对 skill 与相对 M1 的增量价值。统计检验按比较类型选择，而非仅按模态集大小。加入遥感或航运会扩大信息集，但这本身并不使两个预测在检验上嵌套。当一个预测设定嵌套于另一个时——例如在同一学习器下 Ridge M1 对 Ridge M2、M3 或 M4——使用 Clark–West（2007）检验较大模型是否改善均方预测误差。当比较不嵌套时——例如 Flat 对 Deep，或改变超参或架构的 XGBoost 与 Deep 设定——使用 Diebold–Mariano（1995）检验等预测精度。相关时注明小样本调整。每次比较亦报告相对 M0、以及相关时相对 M1 的 RMSE 与 MAE 差异。

可解释性诊断仅用于相对 M0 有改善的设定。主要为 Deep M3，以及相关时的 Deep M4。诊断报告模态门控权重与站点或节点注意力。

---

