# Chapter 3 — Methodology *(~2,500–3,000)*

# 第 3 章 — 方法 *(约 2,500–3,000 词)*

## 3.1 Research design

## 3.1 研究设计

This chapter specifies the empirical design used to answer the research questions in Section 1.2. The design has three components that remain fixed across comparisons: a no-change benchmark, four information sets organised around a common financial time series block, and two model families evaluated using the same weekly forecast dates, sample window and evaluation rules. This structure separates changes in the information supplied to a model from changes in how that information is represented and combined.

本章说明用以回答第 1.2 节研究问题的实证设计。设计包含在各对照中保持不变的三个要素：不变预测基准、围绕同一金融时序块组织的四个信息集，以及使用同一周度预测日、样本窗口与评估规则的两套模型族。该结构把供给模型的信息变化，与信息如何被表征与组合的变化区分开来。

The no-change benchmark, denoted M0, is defined first. At each forecast origin \(t\), M0 sets the one-week-ahead Brent price forecast equal to the current weekly price:

\[
\hat{P}_{t+1|t}=P_t.
\]

M0 is a driftless random-walk forecast and requires no parameter estimation. It is a reference forecast rather than an information set: it contains no predictors. Every learned model is compared with M0 on the same evaluation sample. A specification is described as improving on M0 when it achieves a lower out-of-sample RMSE on that sample; formal forecast-comparison tests are reported separately. The weekly price \(P_t\), the log-return training target and the mapping between return-space forecasts and reconstructed prices are defined in Section 3.2; under that mapping, M0 coincides with a zero-return forecast.

首先定义不变预测基准 M0。在每个预测起点 \(t\)，M0 将提前一周的 Brent 价格预测设为当前周价格：

\[
\hat{P}_{t+1|t}=P_t.
\]

M0 是无漂移随机游走预测，无需参数估计。它是参照预测而非信息集：不含预测变量。每个学习模型都在同一评价样本上与 M0 比较。当某设定在该样本上取得更低的样本外 RMSE 时，称其相对 M0 有改善；正式预测比较检验另行报告。周价格 \(P_t\)、对数收益训练目标，以及收益空间预测与重构价格之间的映射，见第 3.2 节；在该映射下，M0 与零收益预测一致。

The predictors are organised into four information sets around a common financial time series block. M1 contains financial time series only (financial, macroeconomic and oil-market series). M2 adds remote-sensing data to M1; M3 adds shipping data to M1; and M4 adds both remote sensing and shipping. M2 and M3 are parallel extensions of M1 rather than a single nested sequence, while M4 contains both branches.

预测变量围绕同一金融时序块组织为四个信息集。M1 仅含金融时序（金融、宏观与油市序列）；M2 在 M1 上加入遥感；M3 在 M1 上加入航运；M4 同时加入遥感与航运。M2 与 M3 是对 M1 的平行扩展，而非单一嵌套序列；M4 包含两支。

**Table 3.1 — Information sets**

**表 3.1 — 信息集**

| Set | Content |
| --- | --- |
| M1 | Financial time series only (financial, macroeconomic and oil-market series) |
| M2 | M1 + remote sensing |
| M3 | M1 + shipping |
| M4 | M1 + remote sensing + shipping |

| 集合 | 内容 |
| --- | --- |
| M1 | 仅金融时序（金融、宏观与油市序列） |
| M2 | M1 + 遥感 |
| M3 | M1 + 航运 |
| M4 | M1 + 遥感 + 航运 |

Comparing M2 with M1 measures the contribution of remote sensing when added alone, while M3 versus M1 measures the contribution of shipping. M4 versus M1 evaluates their joint contribution. Two additional comparisons are used to assess conditional contributions: M4 versus M3 tests whether remote sensing adds value once shipping is present, and M4 versus M2 tests whether shipping adds value once remote sensing is present. All comparisons retain the same forecast horizon and weekly calendar.

将 M2 与 M1 比较度量单独加入遥感的贡献；M3 对 M1 度量航运的贡献；M4 对 M1 评估二者联合贡献。另用两组条件对照：M4 对 M3 检验在已有航运时遥感是否仍有价值；M4 对 M2 检验在已有遥感时航运是否仍有价值。所有比较保持同一预测期与周历。

Two model families are applied to these information sets. In the Flat family, the selected inputs are converted into weekly features, placed in a single predictor table and used to fit Ridge and XGBoost models. This is referred to as flat feature fusion. In the Deep family, financial, remote-sensing and shipping inputs are processed by modality-specific encoders before their representations are combined. Gated representation-level fusion is the primary specification. Representation concatenation and cross-attention are included as comparison variants. Representation concatenation occurs after the modality-specific encoders and is therefore distinct from the feature-level concatenation used in the Flat family. Encoder and fusion details are provided in Section 3.7.

两套模型族应用于这些信息集。Flat 族将所选输入转为周度特征，放入单一预测表，并拟合 Ridge 与 XGBoost；称为扁平特征融合。Deep 族则对金融、遥感与航运输入先经模态专属编码器处理，再组合其表征。门控表示级融合为主设定；表示拼接与交叉注意力为对照变体。表示拼接发生在模态编码器之后，因而不同于 Flat 族的特征级拼接。编码器与融合细节见第 3.7 节。

Matched Flat–Deep comparisons evaluate the overall difference between flat tabular modelling and modality-specific representation learning when the information set, forecast dates and evaluation sample are held constant. Because the two families also differ in model class and capacity, these comparisons are not interpreted as isolating the fusion mechanism alone. Within the Deep family, comparisons among representation concatenation, gated fusion and cross-attention assess the fusion mechanism while keeping the modality-specific encoders and input data fixed.

匹配的 Flat–Deep 比较在信息集、预测日与评价样本固定时，评估扁平表格建模与模态专属表示学习的整体差异。因两族在模型类别与容量上亦不同，这些比较不解释为单独隔离融合机制。在 Deep 族内，表示拼接、门控融合与交叉注意力之间的比较，在编码器与输入数据固定下评估融合机制。

The research questions map onto these comparisons. RQ1 is assessed through comparisons of M2, M3 and M4 with M1 within each model family, together with comparisons of every learned forecast against M0. The M4–M3 and M4–M2 contrasts provide additional evidence on whether either alternative modality adds value in the presence of the other. The evaluation section specifies the test used for each forecast pair, distinguishing nested from non-nested forecast-model comparisons rather than relying on predictor-set nesting alone.

研究问题对应上述对照。RQ1 通过各族内将 M2、M3、M4 与 M1 比较，以及将全部学习预测与 M0 比较来评估；M4–M3 与 M4–M2 对照进一步说明在已有另一另类模态时，任一模态是否仍有价值。评估节为每对预测指定所用检验，区分嵌套与非嵌套的预测模型比较，而非仅依赖预测变量集嵌套。

RQ2 is assessed at two levels. Matched Flat–Deep pairs, such as M3_Flat and M3_Deep, compare the two overall modelling strategies on the same information set. Comparisons among the Deep fusion variants then examine whether gated or cross-attention fusion improves on representation concatenation when the encoders are held fixed.

RQ2 分两层评估。匹配的 Flat–Deep 对（如 M3_Flat 与 M3_Deep）在同一信息集上比较两种整体建模策略；Deep 融合变体之间的比较则在编码器固定时，考察门控或交叉注意力是否优于表示拼接。

RQ3 is restricted to Deep specifications that satisfy the pre-defined M0 improvement criterion. The qualifying specifications are identified in Chapter 4 rather than selected in advance by model name. Modality-gate weights are used to examine the model’s relative reliance on the three modalities, while site or node attention identifies which spatial locations receive greater weight under different market conditions. These quantities are interpreted descriptively as evidence about model reliance, not as evidence of causal importance.

RQ3 仅限于满足预先设定的相对 M0 改善准则的 Deep 设定。入选设定在第 4 章识别，而非事先按模型名称选定。模态门控权重用于考察模型对三模态的相对依赖；站点或节点注意力则识别不同市场条件下哪些空间位置获得更大权重。这些量作描述性证据，说明模型依赖，不作因果重要性证据。

Figure 3.1 summarises the research design as a flowchart: the no-change benchmark M0 anchors absolute forecast skill; financial time series M1 branches into M2 (plus remote sensing), M3 (plus shipping) and M4 (both); and Flat and Deep families are estimated at matched information sets under a shared rolling-origin protocol (Section 3.9).

图 3.1 以流程图概括研究设计：不变预测基准 M0 锚定绝对预测技能得分；金融时序 M1 分支为 M2（加遥感）、M3（加航运）与 M4（二者）；Flat 与 Deep 族在匹配信息集上、于共享滚动起点协议下估计（第 3.9 节）。

*[Figure 3.1 — Research design flowchart: M0 benchmark; M1→M2/M3/M4 branching; paired Flat vs Deep; link to rolling-origin evaluation.]*

*[图 3.1 — 研究设计流程图：M0 基准；M1→M2/M3/M4 分支；配对 Flat 与 Deep；衔接滚动起点评估。]*

## 3.2 Prediction target and timeline

## 3.2 预测目标与时间轴

Let \(P_t\) denote the last available daily Brent spot-price observation in week \(t\), where each week ends on Friday, measured in US dollars per barrel. The quantity reported in the results is the one-week-ahead price \(P_{t+1}\). Models are not trained directly on the price level. They predict the one-week logarithmic return

\[
r_{t+1}=\log\!\left(\frac{P_{t+1}}{P_t}\right)
\]

and reconstruct the price forecast as

\[
\hat{P}_{t+1|t}=P_t\exp\!\left(\hat{r}_{t+1|t}\right).
\]

Log returns are used to reduce the strong persistence in the price level and to express the forecasting task in terms of proportional weekly changes. RMSE, MAE and benchmark skill scores are computed from the reconstructed price forecasts. Directional accuracy is reported separately as an auxiliary statistic based on the sign of the predicted and observed returns. Under this mapping, the no-change benchmark \(\hat{P}_{t+1|t}=P_t\) coincides exactly with the zero-return forecast \(\hat{r}_{t+1|t}=0\).

令 \(P_t\) 表示第 \(t\) 个周五截止周内最后一个可获得的 Brent 现货价格日度观测值，单位为美元/桶。结果中报告的量是提前一周的价格 \(P_{t+1}\)。模型不直接在价格水平上训练，而是预测一周对数收益

\[
r_{t+1}=\log\!\left(\frac{P_{t+1}}{P_t}\right),
\]

并按

\[
\hat{P}_{t+1|t}=P_t\exp\!\left(\hat{r}_{t+1|t}\right)
\]

重构价格预测。使用对数收益是为了减弱价格水平序列的强持续性，并将预测任务表示为周度比例变化。RMSE、MAE 和相对基准的预测技能得分均根据重构后的价格预测计算。方向准确率则根据预测收益与实际收益的符号另行计算，并仅作为辅助统计量报告。在此映射下，不变预测基准 \(\hat{P}_{t+1|t}=P_t\) 与零收益预测 \(\hat{r}_{t+1|t}=0\) 完全一致。

All series are organised on a Friday-ending weekly calendar. The modelling window covers 2019–2025 and provides a common weekly index of 365 observations (4 January 2019 to 26 December 2025). Flat models use a merged weekly feature table on this index; Deep models use the same dates but modality-specific sequence and graph tensors rather than a single flat matrix. After reserving the first 104 weeks for initial estimation, allowing three additional weeks to form the first four-week input sequence, and excluding the final week because \(P_{t+1}\) is unavailable, 257 scored forecast origins remain, from 22 January 2021 to 19 December 2025. At each origin \(t\), forecasts may use only information that was actually available at that forecast date; publication lags and alignment rules are set out in Section 3.5.

全部序列按周五截止的周历组织。建模窗口覆盖 2019–2025 年，提供含 365 个观测的共同周索引（2019 年 1 月 4 日至 2025 年 12 月 26 日）。Flat 模型在该索引上使用合并周度特征表；Deep 模型使用相同日期，但采用模态专属的序列与图张量，而非单一扁平矩阵。预留前 104 周作初始估计，再留出额外 3 周以形成第一个四周输入序列，并因无法获得 \(P_{t+1}\) 而排除最后一周后，剩余 257 个可计分预测起点，起止为 2021 年 1 月 22 日至 2025 年 12 月 19 日。在每个起点 \(t\)，预测只能使用该预测时点实际可获得的信息；发布滞后与对齐规则见第 3.5 节。

## 3.3 Geographic scope and monitoring sites

## 3.3 地理范围与监测站点

Because the prediction target is the global Brent benchmark rather than a local physical cargo price at a single terminal, the study has a distributed rather than contiguous geographic scope. Spatial information enters the models through two components: eleven remote-sensing areas of interest (AOIs) and a seventeen-node shipping graph.

由于预测对象是全球 Brent 基准价格，而非单一码头的现货成交价格，本研究的空间范围是分布式的，而不是一个连续的地理研究区。空间信息通过两部分进入模型：11 个遥感兴趣区（AOI）与 17 节点航运图。

The eleven monitoring locations are Houston Ship Channel (USA); Port of Rotterdam (Netherlands); Ningbo–Zhoushan Port (China); Jamnagar Refinery (India); Jurong Island (Singapore); Ulsan Refinery (South Korea); Basra Oil Terminal (Iraq); Fujairah Oil Terminal (United Arab Emirates); Kharg Island Terminal (Iran); Ras Tanura Terminal (Saudi Arabia); and Yanbu Export Terminal (Saudi Arabia). The sites were selected to balance infrastructure capacity, geographic and supply-chain coverage, and observability in the available satellite products. Together they cover major supply, transit and demand locations in the international oil system. These locations define the geographic AOIs. Flat remote-sensing features are summarised inside a circular buffer with a 5-km radius around each site. Deep remote-sensing images are extracted using site-specific patches whose sizes reflect the facility type and local spatial constraints (larger for ports, intermediate for refineries and smaller for terminals); the exact coordinates and patch radii are reported in Appendix A.

十一个监测地点为：休斯敦航道（美国）、鹿特丹港（荷兰）、宁波–舟山港（中国）、贾姆纳格尔炼厂（印度）、裕廊岛（新加坡）、蔚山炼厂（韩国）、巴士拉原油码头（伊拉克）、富查伊拉原油码头（阿联酋）、哈尔克岛码头（伊朗）、Ras Tanura 码头（沙特）与延布出口码头（沙特）。站点选择兼顾基础设施产能、地理与供应链覆盖，以及在可用卫星产品中的可观测性，共同覆盖国际石油体系中的主要供给、中转与需求区位。这些地点构成地理 AOI。Flat 遥感特征在每个站点周围半径 5 km 的圆形缓冲区内汇总；Deep 遥感影像采用站点特定的裁剪范围，其大小根据设施类型和当地空间条件确定（港口较大、炼厂居中、码头较小）；各站点坐标与具体裁剪半径见附录 A。

The shipping component adds six maritime chokepoints—Strait of Hormuz, Suez Canal, Strait of Malacca, Bab el-Mandeb, Panama Canal and the Cape of Good Hope—to the eleven AOIs, yielding a heterogeneous graph with seventeen nodes. AOI and chokepoint nodes have different feature spaces. Two edge types are used. Directed AOI-to-AOI edges are built from Global Fishing Watch port-visit sequences that identify vessel movements between the eleven AOIs; the weekly edge weight is the observed voyage count \(n_{\mathrm{voyages}}\) for each origin–destination pair, so these edges change from week to week. These O–D edges are distinct from PortWatch and AIS presence or transit indicators, which enter mainly as node features rather than as pairwise links.

AOI–chokepoint edges are separate. They are fixed, undirected and binary, and they remain present every week. They are not inferred from weekly vessel tracks, nearest-neighbour distance or co-occurrence statistics. Instead, each AOI is linked in advance to the chokepoint(s) that define its primary oil-trade corridor role in the international supply chain: Persian Gulf export and transit sites (Fujairah, Ras Tanura, Basra, Kharg) to Hormuz; East and South-East Asian import-route sites (Jurong, Ningbo–Zhoushan, Ulsan) to Malacca; the Red Sea export terminal at Yanbu to Suez and Bab el-Mandeb; Rotterdam to Suez and the Cape of Good Hope as the Europe-bound and Cape alternative corridors; and Houston to Panama. Jamnagar is retained as a demand-side AOI without a dedicated chokepoint link. The full AOI–chokepoint edge list and any edge-weight transforms are reported in Appendix A. Node features also vary by week. Encoder details are given in Section 3.7. The Flat and Deep pathways use the same underlying port- and chokepoint observations and the same weekly forecast dates. The Flat pathway aggregates them into tabular predictors, whereas the Deep pathway retains their node-level organisation and graph relationships. Figure 3.2 maps the AOIs and chokepoints.

航运部分在十一个 AOI 之外加入六个航运咽喉——霍尔木兹海峡、苏伊士运河、马六甲海峡、曼德海峡、巴拿马运河与好望角——形成含十七个节点的异质图。AOI 与咽喉节点具有不同的特征空间。边分为两类：有向的 AOI→AOI 边来自 Global Fishing Watch 港口访问序列所识别的十一站间船舶移动；每周边权为各起点–终点对的观测航次数 \(n_{\mathrm{voyages}}\)，因此这些边随周变化。此类 O–D 边不同于 PortWatch 与 AIS 的在场或过境指标——后者主要作为节点特征进入，而非成对连接。

AOI–咽喉边另行设定。它们是固定、无向、二值边，每周都存在；并非由周度船舶轨迹、最近邻距离或共现统计推断得到。而是事先按各 AOI 在国际石油供应链中的**主贸易走廊角色**连接对应咽喉：波斯湾出口与中转站（富查伊拉、Ras Tanura、巴士拉、哈尔克）连霍尔木兹；东亚与东南亚进口路径站（裕廊、宁波–舟山、蔚山）连马六甲；红海出口终端延布连苏伊士与曼德海峡；鹿特丹连苏伊士及作为替代线的好望角；休斯敦连巴拿马。贾姆纳格尔作为需求侧 AOI，不单列咽喉连接。完整 AOI–咽喉边表及边权变换见附录 A。节点特征亦按周变化。编码器细节见第 3.7 节。Flat 与 Deep 建模路径使用相同的港口与航运咽喉底层观测以及相同的周度预测日：Flat 路径将其汇总为表格预测变量，Deep 路径则保留节点级组织与图关系。图 3.2 标示 AOI 与咽喉位置。

*[Figure 3.2 — Map of 11 oil-infrastructure AOIs and 6 maritime chokepoints used for remote-sensing and shipping inputs.]*

*[图 3.2 — 遥感与航运输入所用的 11 个油气基础设施 AOI 与 6 个航运咽喉分布图。]*

## 3.4 Data sources

## 3.4 数据来源

Three modality blocks enter the design: financial time series (financial, macroeconomic and oil-market series); satellite remote sensing; and maritime shipping. Flat and Deep models use the same underlying sources and the same Friday calendar; they differ in the representation applied to each source before prediction. Table 3.2 summarises the datasets. Variable definitions, site lists and graph edges are documented in Appendix A. Feature counts used in estimation for Flat models are those of the merged weekly feature table; Deep models use the same weekly dates with modality-specific tensors.

三类模态块进入设计：金融时序（金融、宏观与油市序列）；卫星遥感；以及航运。Flat 与 Deep 使用相同底层来源与同一周五日历；差异在于预测前对各来源采用的表征。表 3.2 汇总数据集。变量定义、站点列表与图边见附录 A。估计所用特征数：Flat 以合并周度特征表为准；Deep 使用相同周日期与模态专属张量。

**Table 3.2 — Datasets, variables and sources**

**表 3.2 — 数据集、变量与来源**

| Modality | Dataset / product | Key variables | Source |
| --- | --- | --- | --- |
| Financial time series (M1) | Oil-market and macro weekly series | Prices, inventories, production, interest rates, GPR and related indicators | EIA, FRED, Yahoo Finance and related sources |
| Remote sensing (Flat) | Sentinel-2 optical indices and VIIRS night-time lights | Site-level anomalies at 11 AOIs (NDVI, NDWI, NDBI, BSI; NTL) | Sentinel-2; VIIRS |
| Remote sensing (Deep) | Frozen Prithvi-EO-2.0 embeddings | Monthly Sentinel-2 image-patch embeddings at the same 11 AOIs (no VIIRS) | Prithvi-EO-2.0 / Sentinel-2 |
| Shipping (Flat) | PortWatch and AIS tabular features | Port and chokepoint tanker flows; vessel-activity features | IMF PortWatch; AIS |
| Shipping (Deep) | Same sources, represented as a graph | Weekly heterogeneous graph with 17 nodes (11 AOIs and 6 chokepoints) | PortWatch; AIS |

| 模态 | 数据集 / 产品 | 关键变量 | 来源 |
| --- | --- | --- | --- |
| 金融时序（M1） | 油市与宏观周序列 | 价格、库存、产量、利率、GPR 及相关指标 | EIA、FRED、Yahoo Finance 等 |
| 遥感（Flat） | Sentinel-2 光学指数与 VIIRS 夜光 | 11 个 AOI 的站点级异常（NDVI、NDWI、NDBI、BSI；NTL） | Sentinel-2；VIIRS |
| 遥感（Deep） | 冻结 Prithvi-EO-2.0 嵌入 | 同一 11 个 AOI 的月度 Sentinel-2 影像块嵌入（不含 VIIRS） | Prithvi-EO-2.0 / Sentinel-2 |
| 航运（Flat） | PortWatch 与 AIS 表格特征 | 港口与咽喉油轮流量；船舶活动特征 | IMF PortWatch；AIS |
| 航运（Deep） | 同源数据，图表示 | 含 17 个节点的周度异质图（11 个 AOI 与 6 个咽喉） | PortWatch；AIS |

The financial time series block is assembled from weekly oil-market and macro-financial series drawn from the US Energy Information Administration (EIA), Federal Reserve Economic Data (FRED), Yahoo Finance and related scholarly indicators. The series include crude prices and spreads, inventories, production and refinery activity, volatility and risk measures, interest rates, exchange rates, futures-based oil indicators and geopolitical risk. This block is M1—financial time series only—before remote sensing or shipping is added. In the Deep pathway it is encoded as a single modality stream—referred to below as the finance encoder—even though the predictors extend beyond prices alone.

金融时序块由美国能源信息署（EIA）、联邦储备经济数据（FRED）、Yahoo Finance 及相关学术指标的周度油市与宏观金融序列汇总而成，包括原油价格与价差、库存、产量与炼厂活动、波动与风险度量、利率、汇率、基于期货的油市指标以及地缘政治风险。该块即加入遥感或航运前的 M1（仅金融时序）。在 Deep 路径中，它作为单一模态流编码——下文称为金融编码器——尽管预测变量远不止价格本身。

Remote-sensing inputs are observed over the eleven AOIs listed in Section 3.3. Flat and Deep models share these AOIs but use different products from a common Sentinel-2 optical source family. Flat remote sensing uses monthly Sentinel-2 optical indices (NDVI, NDWI, NDBI and BSI) together with VIIRS night-time lights, converted to site-level anomalies. Deep remote sensing uses frozen Prithvi-EO-2.0 embeddings extracted from monthly Sentinel-2 image patches at the same AOIs and excludes VIIRS. Shared AOIs keep spatial coverage matched across pathways; differences in product and representation form part of the Flat–Deep contrast and are detailed in Appendix A.

遥感输入观测于第 3.3 节所列十一个 AOI。Flat 与 Deep 共享这些 AOI，但使用来自共同 Sentinel-2 光学源族的不同产品。Flat 遥感采用月度 Sentinel-2 光学指数（NDVI、NDWI、NDBI 与 BSI）及 VIIRS 夜光，并转换为站点级异常。Deep 遥感采用同一 AOI 上月度 Sentinel-2 影像块提取的冻结 Prithvi-EO-2.0 嵌入，且不含 VIIRS。共享 AOI 使两条路径的空间覆盖保持匹配；产品与表征差异构成 Flat–Deep 对照的一部分，细节见附录 A。

Shipping inputs combine IMF PortWatch measures of chokepoint and port tanker flows with AIS-derived vessel-activity indicators for the network described in Section 3.3. In the Flat pathway these signals enter as tabular weekly features. In the Deep pathway they are represented as the seventeen-node heterogeneous graph: time-varying AOI-to-AOI edges come from GFW voyage O–D counts, AOI–chokepoint edges are fixed corridor links assigned by each site’s primary oil-trade route role (Section 3.3), and PortWatch/AIS measures enter mainly as node features. Graph construction is specified in Section 3.7 and Appendix A. In both pathways, shipping is treated as a proxy for physical trade and congestion rather than as a direct measure of next week’s price.

航运输入结合 IMF PortWatch 的咽喉与港口油轮流量测度，以及第 3.3 节所述网络的 AIS 衍生船舶活动指标。Flat 路径中，这些信号以周度表格特征进入；Deep 路径中，它们表示为十七节点异质图：时变 AOI→AOI 边来自 GFW 航次 O–D 计数，AOI–咽喉边为按各站主贸易走廊角色预先指定的固定连接（见第 3.3 节），PortWatch/AIS 测度主要作为节点特征进入。构图见第 3.7 节与附录 A。两条路径都将航运视为实物贸易与拥堵的代理，而非下周价格的直接量测。

**Ethical considerations.** The dissertation uses secondary, aggregate data products and does not involve human participants, interviews or surveys. Market and macro series (EIA, FRED, Yahoo Finance and related indicators), Earth-observation products (Sentinel-2, VIIRS and frozen Prithvi-EO embeddings) and maritime indicators (IMF PortWatch and AIS-derived vessel-activity measures) are accessed under their published terms of use for research. The modelling features are market aggregates, site-level environmental summaries and shipping-activity indicators; they are not used to identify private individuals. AIS-based inputs enter as processed activity measures for commercial and energy-related traffic at ports and chokepoints, consistent with the providers’ intended analytical use, and are not redistributed beyond the licensed research pipeline where redistribution is restricted. Overall risk is minimal: the ethical issue is whether data are used as expected under licence and research purpose, which this design observes by restricting use to forecasting evaluation, documenting sources, and keeping the processing pipeline reproducible.

**伦理考量。** 本论使用二手、汇总型数据产品，不涉及人类参与者、访谈或问卷。市场与宏观序列（EIA、FRED、Yahoo Finance 及相关指标）、对地观测产品（Sentinel-2、VIIRS 与冻结 Prithvi-EO 嵌入）以及海事指标（IMF PortWatch 与 AIS 衍生船舶活动测度）均按公开研究使用条款获取。建模特征为市场汇总、站点级环境摘要与航运活动指标，不用于识别私人个体。基于 AIS 的输入以港口与咽喉处商业及能源相关交通的加工活动测度进入，符合提供方预期的分析用途；在限制再分发的情形下，不超出许可研究流水线另行传播。总体风险很低：伦理问题在于数据是否按许可与研究目的被预期使用；本设计通过将用途限于预测评估、记录来源并保持处理流水线可复现来遵守这一点。

## 3.5 Temporal alignment, lags, missingness

## 3.5 时间对齐、滞后期与缺失

All series are aligned to the Friday-ending weekly calendar defined in Section 3.2. Predictors enter only after their real publication time, so the model never uses future information at any forecast origin. Release lags differ across sources: EIA and PortWatch series typically become available with a lag of about one week, while slower monthly series require longer buffers before they are allowed to enter. Missingness is handled differently in the two pathways. Flat models fill missing values using only past observations within the available history. Deep models retain explicit masks for missing modalities or sites rather than silently filling them away, so that absence remains visible to the encoders and fusion layers.

全部序列对齐到第 3.2 节定义的周五截止周历。预测变量仅在真实发布时刻之后进入，因此任一预测起点都不会使用未来信息。不同来源的发布滞后不同：EIA 与 PortWatch 序列通常约一周后可用，更慢的月度序列则需要更长缓冲才允许进入。缺失处理在两条路径中不同。Flat 模型仅用可得历史中的过去观测填补缺失；Deep 模型对缺失模态或站点保留显式掩码，而非静默填补，从而使缺失对编码器与融合层仍可见。

## 3.6 Flat models

## 3.6 Flat 模型

Flat models implement flat feature fusion as defined in Section 1.2. For a given information set, all available numeric features are concatenated into one weekly table, and the most recent four weeks are flattened into a single row for each forecast origin. Two learners are estimated on this representation. Ridge is a linear model with L2 regularisation and serves as a transparent linear early-fusion baseline. XGBoost is a non-linear tree ensemble that can capture interactions missed by Ridge, but still does not preserve modality-specific structure. Both models predict the one-week-ahead log return and reconstruct price as in Section 3.2. Hyperparameters are chosen inside each training fold on a past validation slice only; the exact grids are reported in Appendix C.

Flat 模型实现第 1.2 节所定义的扁平特征融合。对给定信息集，将全部可用数值特征拼成一张周表，并在每个预测起点将最近四周压成一行。该表征上估计两种学习器。Ridge 是带 L2 正则的线性模型，作为透明的线性早融合基线；XGBoost 是非线性树集成，可捕捉 Ridge 错过的交互，但仍不保留模态特有结构。二者均预测提前一周的对数收益，并按第 3.2 节重构价格。超参数仅在各训练折内、用过去验证切片选择；精确网格见附录 C。

## 3.7 Deep models

## 3.7 Deep 模型

Deep models use the same information sets, Friday calendar and validation protocol as Flat. The difference lies in representation and fusion, not in the forecast target. Each available modality is encoded into a matched-dimensional representation and the representations are then combined. The finance encoder uses a causal temporal convolutional network (TCN) to model temporal dependencies in the weekly financial sequence. Deep remote sensing uses frozen Prithvi-EO embeddings from monthly Sentinel-2 patches and excludes VIIRS; embeddings are kept per site and aggregated by temporal and site attention, so the site dimension is not collapsed before encoding. Shipping is encoded as a weekly heterogeneous graph with seventeen nodes—the eleven AOIs and six chokepoints from Section 3.3. Directed AOI-to-AOI edges use weekly GFW voyage O–D counts; AOI–chokepoint edges are fixed, undirected corridor links assigned by each site’s primary oil-trade route role (Section 3.3), not by weekly tracks or nearest-neighbour distance; PortWatch and AIS indicators enter mainly as node features. A graph attention network (GAT) with temporal encoding aggregates this network into a modality representation. Exact edge construction is reported in Appendix A; GAT depth, number of heads and related layer settings are given in Appendix C. Each encoder is therefore specified by its inputs, network structure, outputs and the reason that architecture fits the modality.

For RQ2, three fusion options are considered. Simple concatenation provides a control that combines modality representations without adaptive weighting. Gated fusion is the main reported design. Cross-attention is retained as an advanced alternative. The fused representation is mapped to the same return and price target as Flat; training details are in Appendix C.

Deep 模型与 Flat 使用相同的信息集、周五日历与验证协议；差异在表征与融合，而非预测目标。每个可用模态被编码为匹配维度的表征，再加以组合。金融编码器使用因果时间卷积网络（TCN）建模周度金融序列的时间依赖。Deep 遥感使用月度 Sentinel-2 影像块的冻结 Prithvi-EO 嵌入且不含 VIIRS；嵌入按站点保留，经时间与站点注意力聚合，编码前不压掉站点维。航运编码为含十七个节点的周度异质图——即第 3.3 节的十一个 AOI 与六个咽喉。有向 AOI→AOI 边使用周度 GFW 航次 O–D 计数；AOI–咽喉边为按各站主贸易走廊角色预先指定的固定无向连接（见第 3.3 节），而非由周度轨迹或最近邻距离推断；PortWatch 与 AIS 指标主要作为节点特征进入。带时间编码的图注意力网络（GAT）将该网络聚合成模态表征。边构造详见附录 A；GAT 深度、头数及相关层设置见附录 C。因此，每个编码器均按输入、网络结构、输出及为何适合该模态加以说明。

就 RQ2 而言，考虑三种融合选项。简单拼接作为对照，在无自适应加权下组合模态表征；门控融合为主报告设计；交叉注意力保留为进阶备选。融合表征映射到与 Flat 相同的收益与价格目标；训练细节见附录 C。

## 3.8 Hyperparameter selection

## 3.8 超参数选择

Hyperparameters are selected under a shared protocol so that Flat–Deep comparisons remain fair. For Flat models, tuning is performed inside each training fold on past validation weeks only. For Deep models, full per-fold neural architecture search is costly. Limited sweeps are therefore run first, after which one main configuration—architecture depths, representation size, main fusion choice and related settings—is fixed for the primary reported results. Sensitivity to random seed, lookback length, fusion type, representation size and regularisation is then reported in Chapter 4 and Appendix C. The sequence is search, lock the main setting, then sensitivity analysis; the main Deep configuration is not chosen arbitrarily. Exact values such as representation size, GAT depth and number of heads, and the Flat search grids, are given in Appendix C.

超参数在共享协议下选择，以使 Flat–Deep 比较保持公平。Flat 模型仅在各训练折内、用过去验证周调参。Deep 模型若逐折做全量神经架构搜索成本过高，故先进行有限 sweep，再为主要报告结果固定一套主配置——架构深度、表示维度、主融合选择及相关设定；随后在第 4 章与附录 C 报告对随机种子、回看长度、融合类型、表示维度与正则的敏感性。顺序为搜索、锁定主设定、再做敏感性分析；主 Deep 配置并非任意选定。表示维度、GAT 深度与头数，以及 Flat 搜索网格等精确值见附录 C。

## 3.9 Leakage-free validation protocol

## 3.9 无泄漏验证协议

Evaluation follows an expanding-window rolling-origin backtest: at each origin the model is trained only on past weeks and then produces a one-week-ahead forecast. The first 104 weeks form the initial estimation and validation period and are not scored. The four-week lookback creates a separate sequence warm-up requirement. Thereafter models are refit every 13 weeks. The common scored test span covers 257 weeks from 22 January 2021 to 19 December 2025. Any scaling or filtering is fit inside the training fold only. Flat and Deep share the same fold calendar, so architecture comparisons hold the evaluation design fixed.

评估采用扩展窗滚动起点回测：在每个起点，模型仅用过去周训练，再给出提前一周预测。前 104 周为初始估计与验证期，不计分；四周回看另构成序列预热要求。之后每 13 周重拟合。共同计分测试跨度为 2021 年 1 月 22 日至 2025 年 12 月 19 日的 257 周。任何缩放或过滤仅在训练折内拟合。Flat 与 Deep 共享同一折日历，从而使架构比较在固定评估设计下进行。

Figure 3.3 summarises the expanding-window rolling-origin protocol: an initial unscored estimation window, one-week-ahead forecasts, and periodic refitting on past information only.

图 3.3 概括扩展窗滚动起点协议：初始不计分估计窗、提前一周预测，以及仅基于过去信息的周期性重拟合。

*[Figure 3.3 — Expanding-window rolling-origin evaluation flowchart.]*

*[图 3.3 — 扩展窗滚动起点评估流程图。]*

## 3.10 Evaluation, tests, interpretability

## 3.10 评估、检验与可解释性

Primary metrics are computed on reconstructed prices. Every comparison reports RMSE and MAE; directional accuracy is retained only as an auxiliary measure. Relative performance versus M0 is summarised by RMSE skill, reported as a percentage in the result tables:

\[
\mathrm{Skill}=100\times\left(1-\frac{\mathrm{RMSE}_{\mathrm{model}}}{\mathrm{RMSE}_{\mathrm{M0}}}\right).
\]

Skill greater than zero means the model beats M0 on RMSE; skill equal to zero matches M0; skill less than zero is worse than M0.

The study evaluates both incremental value versus M1 and absolute skill versus M0. It also distinguishes information-set nesting from formal model nesting: a larger modality set does not automatically make two forecasts nested for statistical testing. Test choice follows the forecast-specification relationship. Clark–West (2007) provides an MSPE-adjusted test of whether a larger model improves on a smaller one when the smaller forecast specification is nested in the larger; it is used for nested increments, for example Ridge M1 versus Ridge M2, M3 or M4 where nesting is justified. Diebold–Mariano (1995) tests equal predictive accuracy from the mean loss differential between two forecasts and is used for non-nested paired comparisons, such as Flat versus Deep or XGBoost and Deep settings that change hyperparameters or architecture; a small-sample adjustment is noted where relevant. Every comparison also reports RMSE and MAE effect sizes versus M0 and, where relevant, versus M1.

Interpretability diagnostics are applied only to specifications that improve on M0—primarily Deep M3, and Deep M4 where relevant—and consist of modality gate weights together with site or node attention. Data used in the study are public or licensed, and the processing and estimation pipeline is scripted to support reproducibility.

主指标在重构价格上计算。每次比较均报告 RMSE 与 MAE；方向准确率仅作辅助度量。相对 M0 的表现以 RMSE skill 汇总，并在结果表中以百分比报告：

\[
\mathrm{Skill}=100\times\left(1-\frac{\mathrm{RMSE}_{\mathrm{model}}}{\mathrm{RMSE}_{\mathrm{M0}}}\right).
\]

Skill 大于零表示模型在 RMSE 上优于 M0；等于零与 M0 持平；小于零则差于 M0。

研究同时评估相对 M1 的增量价值与相对 M0 的绝对 skill，并区分信息集嵌套与形式模型嵌套：模态集变大并不自动使两个预测在统计检验上嵌套。检验选择跟随预测设定关系。Clark–West（2007）提供 MSPE 调整检验，用于在较小预测设定嵌套于较大设定时判断较大模型是否改善较小模型；用于嵌套增量，例如在嵌套成立时 Ridge M1 对 Ridge M2、M3 或 M4。Diebold–Mariano（1995）基于两预测平均损失差检验等预测精度，用于非嵌套配对，例如 Flat 对 Deep，或改变超参或架构的 XGBoost 与 Deep 设定；相关时注明小样本调整。每次比较亦报告相对 M0、以及相关时相对 M1 的 RMSE 与 MAE 效应量。

可解释性诊断仅用于相对 M0 有改善的设定——主要为 Deep M3，以及相关时的 Deep M4——包括模态门控权重与站点或节点注意力。研究所用数据公开或有许可，处理与估计流水线脚本化，以支持可复现。

---
