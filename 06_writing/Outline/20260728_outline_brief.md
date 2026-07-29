# Dissertation Draft Structure — Brief

# 论文草稿结构 — 简版

*Revised 2026-07-28: Ch.1 aligned with Andy (Abstract; importance/background → aims/RQ; detailed gap stays in Ch.2 §2.6); earlier Taylor notes retained for Ch.3–6.*

*2026-07-28 修订：第1章按 Andy 对齐（摘要；重要性/背景 → 目标/RQ；详细空白留在第2章 §2.6）；第3–6章仍保留此前 Taylor 调整说明。*

A Modality-Aware Spatio-Temporal Fusion Framework for Brent Crude Oil Forecasting Using Financial Time Series, Satellite Imagery and Maritime Networks

面向金融时序、卫星遥感与航运网络的模态感知时空融合框架：Brent 原油价格预测

---

## Abstract

## 摘要

*Andy: Who cares → What did you do → How → Results → Why useful*

*Andy：谁关心 / 为何重要 → 做了什么 → 怎么做 → 结果 → 为何有用*

- **Who cares / why important:** Oil prices shape inflation, trade balances, producer revenues and energy-intensive costs. Brent is the global pricing benchmark; reliable short-horizon forecasts matter for risk management and supply-chain planning, especially through recent shocks (COVID, 2022 energy crisis, subsequent normalisation).
- **What:** This dissertation builds a reproducible comparison framework for one-week-ahead weekly Brent forecasting that combines three multimodal physical-market proxies — finance time series, satellite remote sensing, and maritime / shipping networks — and compares flat early fusion with modality-aware representation-level fusion under one protocol.
- **How:** Nested information sets (M1 finance-only → M2 +RS / M3 +shipping / M4 both) are evaluated against a no-change benchmark (M0) with rolling-origin, leakage-safe out-of-sample scoring and formal forecast-comparison tests; Flat (Ridge / XGBoost) is paired with Deep (modality encoders + gated / cross-attention fusion) at matched sets; interpretability (gates, site–node attention) is applied where forecasts improve on M0.
- **Results:** Under Flat, no learned model beats M0, though shipping still helps relative to finance-only M1 while remote sensing does not. Under Deep, finance + shipping (M3) shows small positive skill versus M0; adding RS (M4) does not clearly dominate M3. At matched multimodal sets, Deep outperforms Flat most clearly once shipping enters.
- **Why useful:** The protocol helps analysts and alternative-data providers judge multimodal claims against both a finance baseline and the no-change benchmark, and can be reused beyond weekly Brent.

- **谁关心 / 为何重要：** 油价影响通胀、贸易差额、产油国财政与高耗能成本。Brent 是全球定价基准；可靠的短期限预测对风险管理与供应链规划重要，尤其在近年冲击（新冠、2022 能源危机、后续常态化）中。
- **做了什么：** 本论构建可复现的对照框架，用于提前一周的周度 Brent 预测，整合三类多模态实物市场代理——金融时序、卫星遥感与航运 / 海事网络——并在同一协议下比较扁平早融合与模态感知表示级融合。
- **怎么做：** 嵌套信息集（M1 仅金融 → M2 +遥感 / M3 +航运 / M4 二者）相对不变预测基准（M0），用滚动起点、无泄漏样本外计分与正式预测比较检验；Flat（Ridge / XGBoost）与 Deep（模态编码器 + 门控 / 交叉注意力）在匹配信息集上配对；可解释性（门控、站点–节点注意力）仅用于相对 M0 有改善的设定。
- **结果：** Flat 下无学习模型打过 M0，但航运相对仅金融 M1 仍有帮助而遥感没有。Deep 下金融 + 航运（M3）相对 M0 有小幅正 skill；再加遥感（M4）并不明显主导 M3。匹配多模态信息集上，Deep 在航运进入后相对 Flat 优势最清晰。
- **为何有用：** 该协议帮助分析师与另类数据提供方同时相对金融基线与不变预测基准判断多模态主张，并可复用于周度 Brent 之外的场景。

---

## Chapter 1 — Introduction *(~900–1,200 words)*

## 第 1 章 — 绪论 *(约 900–1,200 词)*

*Andy: importance / background → (global energy, recent shocks) → focus on study topic → aims / RQs. Detailed research gap stays in Chapter 2 (§2.6).*

*Andy：重要性 / 背景 →（全球能源、近年冲击）→ 收束到本研究主题 → 目标 / RQ。详细研究空白留在第 2 章（§2.6）。*



### 1.1 Importance and background

### 1.1 重要性与背景

- **Why oil prices matter:** Crude oil sits at the centre of the world economy. Price movements affect inflation, trade balances, fiscal revenues in producer countries and operating costs in energy-intensive industries; they also transmit quickly through financial markets and real activity. Governments, firms and investors therefore have a lasting interest in short-horizon oil-price forecasts for risk management and planning.
- **Global energy context and recent developments:** Oil remains a core energy commodity in the global energy system. The sample window spans major recent shocks — COVID-era demand collapse and recovery, the **2022 energy crisis**, and subsequent normalisation — which make persistence, volatility and benchmark difficulty especially salient for any forecasting claim.
- **Brent as pricing benchmark:** Among crude markers, **Brent** is the global pricing benchmark for a large share of internationally traded oil. This dissertation focuses on **Friday-ending weekly Brent spot** over **2019–2025**, and on one-week-ahead out-of-sample forecasts. Weekly Brent is hard to beat; a simple **no-change** forecast (next week’s price equals this week’s) is a strong bar that any new data or model claim must clear.
- **Study focus — three multimodal proxies:** The thesis examines whether three physical-market information sources improve those forecasts when used together under one design: (i) **finance** time series; (ii) **satellite remote sensing** (night-time lights, site optical indices / EO image embeddings); (iii) **maritime / shipping networks** (AIS, PortWatch) as proxies for trade, chokepoint congestion and infrastructure activity. Signals are noisy, asynchronous, and may *respond to* prices rather than *lead* them; the weekly sample is small. Dominant practice still flattens all sources into one wide table and early-fuses with finance, discarding finance temporal structure, RS site structure and shipping network structure.
- **Bridge to aims:** Whether these multimodal signals improve weekly Brent forecasts beyond a finance baseline and the no-change benchmark — and whether modality-aware fusion helps relative to flat fusion on the same data — is the empirical question this dissertation addresses. The detailed research gap is developed after the literature review (Chapter 2, §2.6).

- **为何油价重要：** 原油处于世界经济中心。价格变动影响通胀、贸易差额、产油国财政与高耗能行业成本，并经金融市场与实体活动快速传导。政府、企业与投资者因此长期关注短期限油价预测，用于风险管理与规划。
- **全球能源语境与近年发展：** 石油仍是全球能源体系中的核心商品。样本窗口覆盖近年重大冲击——新冠时期需求塌陷与复苏、**2022 能源危机**及后续常态化——使持续性、波动与基准难度对任何预测主张都尤为突出。
- **Brent 作为定价基准：** 在原油标记中，**Brent** 是大部分国际贸易原油的全球定价基准。本论聚焦 **2019–2025** 的**周五截止周度 Brent 现货**，以及提前一周的样本外预测。周度 Brent 难打；简单的**不变预测**（下周价格等于本周）是任何新数据或新模型主张都必须越过的高门槛。
- **研究焦点 — 三类多模态代理：** 本论检验三类实物市场信息源在同一设计下联合使用时能否改善上述预测：（i）**金融**时序；（ii）**卫星遥感**（夜光、站点光学指数 / EO 影像嵌入）；（iii）**航运 / 海事网络**（AIS、PortWatch），作为贸易、咽喉拥堵与基础设施活动的代理。信号嘈杂、异步，且可能是价格的*响应*而非*领先*；周度样本量小。主流做法仍将多源压成宽表并与金融早融合，从而丢掉金融时间结构、遥感站点结构与航运网络结构。
- **过渡到目标：** 这些多模态信号能否在金融基线与不变预测基准之上改善周度 Brent 预测——以及在相同数据上模态感知融合是否优于扁平融合——是本论要回答的实证问题。详细研究空白在文献综述之后展开（第 2 章 §2.6）。



### 1.2 Aim and research questions

### 1.2 研究目标与研究问题

*Logic: do the data help → does fusion architecture matter on the same data → what does the model rely on when value exists.*

*逻辑：数据是否有用 → 同一数据下融合架构是否重要 → 有价值时模型依赖什么。*

- **Aim:** Build one reproducible comparison framework that puts frozen EO embeddings, modality-specific encoders, fusion modules, and a rolling-origin forecast protocol (no look-ahead) in the same design — **integration and fair comparison, not a new neural operator**.

- **目标：** 构建一套可复现的对照框架，将冻结 EO 嵌入、模态专属编码器、融合模块与滚动起点预测协议（无前视）放在同一设计中——**集成与公平比较，而非新的神经算子**。

- **RQ1:** Do remote-sensing and shipping indicators add incremental out-of-sample value over a financial baseline and the random-walk benchmark?
- **RQ2:** Does modality-aware (representation-level) fusion outperform flat feature fusion when both use the same underlying data and the same evaluation protocol?
- **RQ3:** Can modality-level interpretability reveal which signals the model relies on across different market conditions?

- **RQ1：** 遥感与航运指标是否在金融基线与随机游走基准之上带来样本外增量价值？
- **RQ2：** 在相同底层数据与相同评估协议下，模态感知（表示级）融合是否优于扁平特征融合？
- **RQ3：** 模态级可解释性能否揭示模型在不同市场条件下依赖哪些信号？

---



## Chapter 2 — Literature Review

## 第 2 章 — 文献综述

Crude oil occupies a central place in the world economy. Price movements affect inflation, trade balances, fiscal revenues in producer countries and the operating costs of energy-intensive industries. Because these effects propagate quickly through financial markets and real activity, governments, firms and investors have long sought forecasts of oil prices. At the same time, oil prices are volatile, shaped by both physical supply–demand conditions and expectations, which makes reliable prediction difficult. This chapter reviews the literatures that define that forecasting problem and the data and methods used to address it.

原油在世界经济中占据核心位置。价格变动会影响通胀、贸易差额、产油国财政收入，以及能源密集型产业的运营成本。这些影响会迅速传导至金融市场与实体活动，因此政府、企业与投资者长期寻求油价预测。与此同时，油价波动剧烈，同时受实物供需与预期共同塑造，可靠预测并不容易。本章综述界定这一预测问题的相关文献，以及用以应对该问题的数据与方法。

## 2.1 Crude-oil price forecasting



## 2.1 原油价格预测



### 2.1.1 Econometric foundations and the benchmark problem



### 2.1.1 计量经济学基础与基准问题

Oil prices are difficult to predict out of sample. Kilian (2009) shows that price movements should be understood through different structural channels, including crude-oil supply shocks, aggregate-demand shocks and oil-specific precautionary-demand shocks. That structural decomposition is not a forecasting model. It does, however, offer a useful principle for thinking about predictors: variables are more persuasive when they have a plausible economic link to supply, demand or uncertainty, rather than being included only because they are available.

原油价格在样本外很难被准确预测。Kilian (2009) 指出，油价变动应从不同结构性渠道理解，包括原油供给冲击、总需求冲击和石油特定的预防性需求冲击。该结构分解本身并不是预测模型，但它为思考预测变量提供了有用原则：当变量与供给、需求或不确定性具有合理经济联系时更有说服力，而不是仅仅因为数据可得就被纳入。

A separate strand of work focuses on forecasting performance itself. Alquist, Kilian and Vigfusson (2013) show that a simple no-change forecast—equivalently, a random walk without drift, which predicts that the future price equals the current price—is extremely difficult to beat, especially out of sample. Their review also emphasises that in-sample fit does not imply forecasting skill, and that claims of predictability are typically examined with real-time data alignment, recursive or rolling evaluation and formal forecast-comparison tests. Baumeister and Kilian (2015) further show that forecast combinations across different economic mechanisms can be more robust than reliance on a single predictor set. Taken together, this literature treats a strong no-change (random-walk) benchmark, and economically motivated financial or oil-market predictors, as central reference points against which more elaborate models are judged.

另一支文献直接关注预测表现。Alquist, Kilian and Vigfusson (2013) 表明，简单的无变化预测——等价于无漂移随机游走，即预测未来价格等于当前价格——极难被超越，尤其是在样本外。他们的综述还强调，样本内拟合并不等同于预测能力；关于可预测性的主张通常需要结合实时数据对齐、递归或滚动评估，以及正式的预测比较检验来考察。Baumeister and Kilian (2015) 进一步指出，跨不同经济机制的预测组合往往比依赖单一预测变量集更加稳健。综合来看，该文献将强无变化（随机游走）基准，以及具有经济动机的金融或原油市场预测变量，视为评判更复杂模型的核心参照。

### 2.1.2 Machine-learning approaches to oil-price forecasting



### 2.1.2 原油价格预测中的机器学习方法

Machine-learning studies have introduced tree ensembles, regularised linear models, deep learning and hybrid models into oil-price forecasting. Costa et al. (2021) compare a broad set of methods over a large macro-financial predictor set and find that useful predictors vary across horizons and over time. XGBoost can be a strong competitor, but it does not dominate uniformly. Yılmaz and Zehir (2026) show that geopolitical risk, market volatility and interest-rate variables can add value for Brent-return forecasting, with LightGBM outperforming XGBoost in their setting. Foroutan and Lahmiri (2024) report strong performance from temporal convolutional networks and gradient-boosting models, but their focus on price-level prediction also illustrates a common problem: because oil prices are highly persistent, low one-step price-level errors can partly reflect the fact that P_{t+1} is usually close to P_t.

机器学习研究已经将树模型、正则化线性模型、深度学习和混合模型引入原油价格预测。Costa et al. (2021) 在大规模宏观金融预测变量集上比较了多种方法，发现有用预测变量会随预测期限和时间变化。XGBoost 可以是强有力的竞争者，但并不在所有情境下都占优。Yılmaz and Zehir (2026) 表明，地缘政治风险、市场波动率和利率变量能够为 Brent 收益率预测提供增量信息，其中 LightGBM 在他们的设定中优于 XGBoost。Foroutan and Lahmiri (2024) 报告称，时间卷积网络和梯度提升模型表现较强，但其对价格水平预测的关注也反映出一个常见问题：由于原油价格高度持久，较低的一步价格水平误差可能部分来自 P_{t+1} 通常接近 P_t 这一事实。

Several recent studies use more complex architectures, but their results should be read cautiously. Simsek et al. (2024) report that hybrid designs combining LSTM feature extraction with XGBoost can achieve very high R^2 values, yet such results may be sensitive to preprocessing choices such as pre-split normalisation. Graph-based oil-price models have also appeared. Zhao, Xue and Cheng (2023), for example, combine a self-attention-learned dynamic graph with Graph WaveNet (Wu et al., 2019) for multi-step WTI futures forecasting. Their graph represents non-Euclidean relations among predictors rather than a physical shipping or geographic network, and the absence of a no-change benchmark makes it difficult to assess whether the model improves over a random walk. Overall, this strand of work shows that flexible machine-learning methods can exploit large predictor sets, but that complexity alone does not guarantee genuine forecasting gains over simple benchmarks.

一些近期研究采用了更复杂的模型结构，但其结果需要谨慎解读。例如，Simsek et al. (2024) 报告称，将 LSTM 特征提取与 XGBoost 相结合的混合设计可取得极高的 R^2，但这类结果可能对预处理选择非常敏感，例如是否存在数据划分前的标准化泄漏。图模型也开始出现在原油价格预测中。Zhao, Xue and Cheng (2023) 将自注意力学习到的动态图与 Graph WaveNet (Wu et al., 2019) 结合，用于多步 WTI 期货预测。其图结构表示的是预测变量之间的非欧几里得关系，而不是物理意义上的航运网络或地理网络；同时，由于缺少无变化基准，模型是否真正优于随机游走仍难以判断。总体而言，这一支文献表明，灵活的机器学习方法可以利用大型预测变量集，但复杂度本身并不能保证相对简单基准的真实预测增益。

Across both econometric and machine-learning work, three themes recur. Lagged prices and the no-change forecast remain serious competitors. Financial and oil-market variables—volatility, geopolitical risk, interest rates, exchange rates, futures or market-based oil indicators, as well as inventories, production and refinery activity—are widely used as economically motivated predictors linked to supply, demand and uncertainty. Claims that additional data sources improve forecasts are therefore usually judged by whether they add information beyond these established references, not only by whether a more elaborate model fits better in sample.

在计量与机器学习两条线上，有三个主题反复出现。滞后价格与无变化预测始终是严肃的竞争者。金融与原油市场变量——波动率、地缘政治风险、利率、汇率、期货或市场型原油指标，以及库存、产量与炼厂活动——被广泛用作与供给、需求与不确定性相联系的、具有经济动机的预测变量。因此，关于额外数据源能否改善预测的主张，通常要看它们是否在这些既有参照之外提供了新信息，而不能仅看更复杂的模型是否在样本内拟合更好。

## 2.2 Shipping activity and oil markets



## 2.2 航运活动与原油市场



### 2.2.1 AIS and maritime activity as trade-flow proxies



### 2.2.1 AIS 与海运活动作为贸易流代理变量

Automatic Identification System (AIS) vessel tracking has become an important high-frequency proxy for physical trade. Adland et al. (2017) validate AIS-derived crude-export estimates against official statistics, while Yan et al. (2020) show that global marine oil trade is concentrated around major chokepoints such as Hormuz, Malacca and Suez. Arslanalp, Marini and Tumbarello (2019) and IMF PortWatch (Arslanalp et al., 2026) further demonstrate how vessel movements can be used to nowcast trade activity. This literature supports treating shipping data as a plausible source of physical-market information, without assuming that AIS counts directly measure crude flows or predict prices.

自动识别系统（Automatic Identification System, AIS）船舶轨迹数据已经成为衡量实物贸易活动的重要高频代理变量。Adland et al. (2017) 将基于 AIS 估算的原油出口量与官方统计数据进行验证；Yan et al. (2020) 则表明，全球海上原油贸易高度集中于霍尔木兹海峡、马六甲海峡和苏伊士运河等关键咽喉。Arslanalp, Marini and Tumbarello (2019) 以及 IMF PortWatch 方法（Arslanalp et al., 2026）进一步展示了如何利用船舶移动数据对贸易活动进行即时预测。这些文献支持将航运数据视为实物市场信息的合理来源，但并不意味着 AIS 计数能够直接衡量原油流量或预测价格。

Capacity-weighted indicators and draught-change measures are generally more informative than simple vessel counts, because they better approximate cargo movement. AIS data also require careful filtering to exclude non-trade activity, and moving averages must be constructed without using future observations. In forecasting settings, a shipping indicator is informative only if it is both economically meaningful and available at the forecast origin. In practice, chokepoint transit volumes and tanker-presence measures are therefore usually read as activity proxies at key maritime nodes rather than as direct measures of global supply.

与简单船舶数量相比，按运力加权的指标和吃水变化指标通常更具信息含量，因为它们更接近货物流动本身。AIS 数据也需要仔细过滤，以排除非贸易活动；移动平均等处理也必须避免使用未来观测值。在预测情境中，航运指标只有在既具有经济含义、又在预测时点真实可得时，才可能提供信息。实践中，咽喉通行量和油轮存在度指标通常被理解为关键海运节点的活动代理，而不是全球供给的直接测量。

### 2.2.2 Reverse causality and proxy limitations



### 2.2.2 反向因果与代理变量限制

The relationship between shipping activity and oil prices is not one-directional. Mi et al. (2022) and Mi, Zang, Lo and Chen (2023) study how crude-oil prices affect tanker port-call activity, rather than how tanker activity predicts prices. Their findings indicate that the relationship is non-linear and regionally heterogeneous, and that statistically significant relationships may explain only a small share of variation. Shipping data may contain useful information about physical trade and congestion, but they may also respond to oil prices rather than lead them.

航运活动与原油价格之间的关系并不是单向的。Mi et al. (2022) 以及 Mi, Zang, Lo and Chen (2023) 研究的是原油价格如何影响油轮港口停靠活动，而不是油轮活动如何预测价格。他们的发现表明，这一关系具有非线性和区域异质性，统计显著的关系可能只能解释较小比例的变异。航运数据可能包含关于实物贸易和拥堵的有用信息，但它也可能是对原油价格变化的反应，而非领先信号。

Chokepoint and port indicators are also imperfect proxies for crude-oil flows. A vessel's previous port is not always the cargo origin; crude oil may be blended or re-sold; and ship-to-ship transfers can obscure the true trade route. Paolo et al. (2024) further show that a substantial amount of industrial activity at sea is absent from AIS. Shipping indicators are therefore best interpreted as noisy proxies for physical-market conditions, not as direct measurements of global oil supply.

咽喉和港口指标也只是原油流动的不完美代理。船舶的上一港口并不总是货物原产地，原油可能被混合、转售，船对船转运也可能模糊真实贸易路径。Paolo et al. (2024) 进一步表明，大量海上工业活动并未被 AIS 覆盖。因此，航运指标最好被理解为对实物市场状况的噪声代理变量，而不是全球原油供给的直接测量。

### 2.2.3 From flat shipping indicators to maritime structure



### 2.2.3 从扁平航运指标到海运结构

Most oil-related uses of AIS and PortWatch-style data convert shipping activity into tabular features such as port calls, vessel counts or chokepoint transit volumes. This is useful, but it discards the network structure of maritime activity. Maritime trade is inherently spatial and relational: ports, terminals, chokepoints and routes form a connected transport system. Studies such as Ouyang et al. (2022) and Liang et al. (2022) show that graph-based models can learn spatial-temporal structure in vessel-flow prediction. Those studies forecast traffic flows rather than oil prices. They support the idea that maritime structure is learnable, while leaving open whether such representations help in price forecasting.

多数与原油相关的 AIS 和 PortWatch 数据应用，会将航运活动转化为港口停靠数、船舶数量或咽喉通行量等表格型特征。这种做法有实际价值，但会丢失海运活动本身的网络结构。海上贸易本质上具有空间性和关系性：港口、码头、咽喉和航线共同构成一个相互连接的运输系统。Ouyang et al. (2022) 和 Liang et al. (2022) 等研究表明，图模型可以学习船舶流量预测中的时空结构。这些研究预测的是交通流，而不是原油价格。它们支持“海运结构可以被学习”这一判断，但海运表示是否有助于价格预测仍是开放问题。

## 2.3 Satellite imagery and remote sensing



## 2.3 卫星影像与遥感



### 2.3.1 Remote sensing as an economic proxy



### 2.3.1 遥感作为经济代理变量

Remote sensing provides a physical view of economic activity, infrastructure and environmental conditions. In oil-related applications, night-time lights, NO₂, cloud cover and high-resolution imagery have all been used as indirect indicators of economic activity, trade, demand or inventory information. The literature is cautious about what these signals can and cannot measure.

遥感数据提供了观察经济活动、基础设施和环境条件的物理视角。在原油相关应用中，夜间灯光、NO₂、云量和高分辨率影像都曾被用作经济活动、贸易、需求或库存信息的间接指标。相关文献对这些信号能够衡量什么、不能衡量什么持较为谨慎的态度。

Night-time lights are one of the most widely used remote-sensing proxies, but their usefulness depends on scale. Polinov, Bookman and Levin (2022) find that night-time lights correlate with anchorage activity at a broad cross-sectional scale, yet they do not reliably track tanker activity at a single port. Gibson et al. (2021) similarly show that VIIRS night-time lights are more suitable than DMSP for facility-scale work, but that night lights capture cross-sectional differences more reliably than within-unit temporal variation. Raw radiance is therefore a weak candidate as a direct time-series measure of oil activity; within-site anomalies are usually more defensible than raw levels when the objective is forecasting over time.

夜间灯光是最常用的遥感代理变量之一，但其有效性依赖于空间尺度。Polinov, Bookman and Levin (2022) 发现，夜间灯光在较大的横截面尺度上与锚地活动相关，但并不能可靠追踪单一港口的油轮活动。Gibson et al. (2021) 同样表明，VIIRS 夜间灯光比 DMSP 更适合设施尺度研究，但夜间灯光更擅长捕捉横截面差异，而非单位内部的时间变化。因此，原始辐亮度作为原油活动的直接时间序列指标较弱；当目标是时间序列预测时，站点内部异常值通常比原始水平值更站得住脚。

Other remote-sensing indicators provide different mechanisms. Hao and Wang (2023) link cloud cover over US storage regions to next-week WTI returns through an information-availability channel: when clouds obstruct optical observation of storage tanks, market uncertainty about inventories may increase. Bricongne et al. (2026) use tropospheric NO₂ to nowcast national oil demand, but their results also show that the incremental value of remote-sensing variables can weaken inside non-linear models. Wang et al. (2019) estimate oil-tank structural capacity from high-resolution images, which supports infrastructure measurement but not high-frequency inventory estimation at Sentinel-2 resolution.

其他遥感指标对应不同机制。Hao and Wang (2023) 将美国储油区上空云量与下一周 WTI 收益率联系起来，其机制是信息可得性：当云层遮挡储油罐的光学观测时，市场关于库存的不确定性可能上升。Bricongne et al. (2026) 使用对流层 NO₂ 对国家层面的石油需求进行即时预测，但他们的结果也显示，当进入非线性模型后，遥感变量的增量价值可能减弱。Wang et al. (2019) 利用高分辨率影像估计储油罐结构容量，这支持基础设施测量，但并不等同于在 Sentinel-2 分辨率下进行高频库存估计。

### 2.3.2 Limits of direct RS-to-price claims



### 2.3.2 直接 RS-to-price 主张的限制

Satellite data may contain oil-relevant information, but the literature does not support a simple claim that satellite imagery directly predicts oil prices. Most remote-sensing indicators are upstream proxies: they may reflect industrial activity, port activity, storage observability or demand conditions, which may then influence prices through supply-demand expectations. The mechanism is indirect and may vary across locations, sensors and time horizons.

卫星数据可能包含与原油市场相关的信息，但文献并不支持“卫星影像可以直接预测原油价格”的简单主张。多数遥感指标是上游代理变量：它们可能反映工业活动、港口活动、库存可观测性或需求条件，而这些因素又可能通过供需预期影响价格。该机制是间接的，并且可能随地点、传感器和预测期限而变化。

Jung (2026) provides a useful example of this limitation. The study combines satellite-derived indicators with port attributes to nowcast port-level trade, but the target is trade rather than price, and the model relies on engineered tabular features rather than learned image representations. This pattern is common: remote sensing is often converted into flat numeric columns before entering an economic model. Such features may be informative, yet it remains an open question whether preserving image or site-level representations adds value in oil-price forecasting.

Jung (2026) 提供了一个有用例子。该研究将卫星衍生指标与港口属性结合，用于即时预测港口层面的贸易，但其预测目标是贸易而非价格，并且模型依赖工程化表格特征，而不是学习到的图像表示。这一模式很常见：遥感数据通常先被压缩为扁平数值列，再进入经济模型。这些特征可能具有信息量，但保留图像或站点层面的表示是否能在油价预测中增加价值，仍是开放问题。

## 2.4 Multimodal forecasting and fusion



## 2.4 多模态预测与融合



### 2.4.1 From multi-source data to multimodal learning



### 2.4.1 从多源数据到多模态学习

A useful distinction in the wider literature is between multi-source feature fusion and multimodal representation learning. Baltrušaitis, Ahuja and Morency (2019) define multimodal learning as involving representation, translation, alignment, fusion and co-learning. Within this taxonomy, fusion can occur at the feature level, decision level or representation level. In much of the oil-price forecasting literature, heterogeneous data are combined through early feature-level fusion: financial indicators, shipping counts and satellite-derived indices are concatenated into a single table and passed to a conventional model. This approach is practical, but it treats all inputs as ordinary numeric predictors and may discard modality-specific structure.

更广泛文献中的一个有用区分是：多源特征融合并不等同于多模态表示学习。Baltrušaitis, Ahuja and Morency (2019) 将多模态学习定义为涵盖表示、转换、对齐、融合和协同学习等问题。在该分类框架下，融合可以发生在特征层、决策层或表示层。在多数原油价格预测研究中，异质数据通常通过早期特征层融合被合并：金融指标、航运计数和卫星衍生指数被拼接成一个表格，然后输入常规模型。这种方法具有实践便利性，但它会将所有输入都视作普通数值预测变量，并可能丢失模态特有结构。

A modality-aware alternative is to encode different data types separately before fusing their representations. At the fusion step itself, learned gating offers a way to weight modalities dynamically rather than treating them equally: Arevalo et al. (2017) propose gated multimodal units in which the model learns, for each input, how much each modality contributes to the fused representation. Gohari et al. (2024) provide a relevant precedent in financial time-series forecasting, showing that modality-aware transformers can outperform naïve concatenation when combining different financial information sources. Their application involves text and numeric data rather than satellite imagery, maritime networks and oil-market variables. The study illustrates the potential of modality-aware forecasting, but it does not establish whether such fusion helps in crude-oil price prediction.

模态感知的替代方案是先分别编码不同数据类型，再融合它们的表示。在融合环节本身，可学习的门控提供了一种动态加权各模态、而非同等对待的机制：Arevalo et al. (2017) 提出门控多模态单元（gated multimodal units），让模型针对每个输入学习各模态对融合表示的贡献权重。Gohari et al. (2024) 在金融时间序列预测中提供了一个相关先例，表明在结合不同金融信息来源时，模态感知 Transformer 可以优于简单拼接。但该研究处理的是文本和数值数据，而不是卫星影像、海运网络和原油市场变量。它说明了模态感知预测的潜力，但并不能确立这种融合是否有助于原油价格预测。

### 2.4.2 Representation learning for Earth-observation data



### 2.4.2 地球观测数据的表示学习

Recent Earth-observation foundation models provide one route to transforming satellite imagery into representations. Models such as SatMAE (Cong et al., 2022) and Prithvi-EO-2.0 (Szwarcman et al., 2026) use self-supervised pretraining to produce transferable image embeddings. Multisensor models such as CROMA (Fuller et al., 2023) further show that optical and radar data may benefit from modality-specific encoders before fusion, because the sensors differ in channel structure, noise and physical meaning.

近期地球观测基础模型为将卫星影像转化为表示提供了一条路径。SatMAE (Cong et al., 2022) 和 Prithvi-EO-2.0 (Szwarcman et al., 2026) 等模型使用自监督预训练来生成可迁移的图像嵌入。CROMA (Fuller et al., 2023) 等多传感器模型进一步表明，光学和雷达数据在融合前可能需要模态特定编码器，因为不同传感器在通道结构、噪声和物理含义上存在差异。

The EO foundation-model literature is mainly methodological. Most evaluations target land-cover classification, segmentation or related remote-sensing tasks, not economic forecasting or oil prices. These models show that satellite imagery can be represented by pretrained encoders; they do not yet provide direct evidence that such representations improve commodity-price forecasts.

地球观测基础模型文献主要是方法论层面的。多数评估针对土地覆盖分类、语义分割或相关遥感任务，而不是经济预测或原油价格。这些模型说明卫星影像可以被预训练编码器表示，但尚未提供此类表示能够改善大宗商品价格预测的直接证据。

### 2.4.3 Missing and asynchronous modalities



### 2.4.3 缺失模态与异步模态

A further challenge is that alternative data sources are often incomplete or asynchronous. Optical satellite imagery is affected by cloud cover, radar and optical sensors have different revisit cycles, and shipping or macro-financial data may be released at different frequencies. General multimodal-learning studies show that models can degrade when modalities are missing unless missing-modality training or modality dropout is used (Ma et al., 2022; Neverova et al., 2016), and related work on irregularly sampled time series further motivates the use of masks and time-since-observation signals (Che et al., 2018; Shukla and Marlin, 2021).

另一个挑战是，替代数据来源通常存在缺失和时间不同步问题。光学卫星影像受云量影响，雷达和光学传感器具有不同重访周期，航运数据和宏观金融数据也可能以不同频率发布。一般多模态学习研究表明，如果不进行缺失模态训练或模态 dropout，模型在模态缺失时可能显著退化 (Ma et al., 2022; Neverova et al., 2016)；针对不规则采样时间序列的相关研究也进一步说明，掩码和距上次观测时间等信号具有价值 (Che et al., 2018; Shukla and Marlin, 2021)。

Missingness and temporal misalignment are therefore part of the multimodal forecasting problem itself, not only a data-cleaning detail. Comparisons between flat feature fusion and representation-level fusion in applied settings typically need to take the availability, timing and reliability of each modality into account.

因此，缺失性和时间错位本身就是多模态预测问题的一部分，而不仅仅是数据清洗细节。在应用情境中比较扁平特征融合与表示层融合时，通常需要考虑每个模态的可得性、时间对齐和可靠性。

## 2.5 Forecast evaluation and interpretability



## 2.5 预测评估与可解释性



### 2.5.1 Forecast comparison



### 2.5.1 预测比较

Because the random walk is difficult to beat, lower RMSE or MAE alone is not usually treated as sufficient evidence of improved forecasting skill. Diebold and Mariano (1995) provide the standard framework for testing equal predictive accuracy across competing forecasts. For nested models, where one model extends another by adding predictors, Clark and West (2007) provide a more appropriate test under squared-error loss. In studies that add new predictors or modalities, these tests are commonly used to ask whether an apparent error reduction is statistically meaningful relative to a simpler baseline, rather than whether a larger model can reduce error in one sample.

由于随机游走很难被超越，仅仅报告较低的 RMSE 或 MAE 通常不足以被视为更强预测能力的充分证据。Diebold and Mariano (1995) 提供了比较竞争预测之间预测精度是否相等的标准框架。对于嵌套模型，即一个模型通过增加预测变量扩展另一个模型的情形，Clark and West (2007) 在平方误差损失下提供了更合适的检验方法。在加入新预测变量或新模态的研究中，这些检验常被用来判断表面的误差下降相对更简单基准是否具有统计意义，而不是一个更大的模型是否能在某个样本中降低误差。

In practice, out-of-sample comparison, horizon-specific performance and formal tests of predictive accuracy are the tools most often used when researchers evaluate whether shipping, remote-sensing or other multimodal inputs improve on financial or no-change references.

实践中，当研究者评估航运、遥感或其他多模态输入是否相对金融或无变化参照有所改进时，最常用的工具是样本外比较、分预测期限的表现，以及正式的预测精度检验。

### 2.5.2 Interpretability and modality-level explanation



### 2.5.2 可解释性与模态层面解释

Forecast accuracy tests can show whether a model improves, but they do not explain which signals drive the improvement. Two complementary attribution routes appear in the applied literature. For flat tabular models, SHAP (Lundberg and Lee, 2017) can attribute predictions to features and be aggregated to groups of variables or modalities. Learned modality gates and site- or node-level attention provide native diagnostics of model dependence (Arevalo et al., 2017; as commonly used in graph-attention settings): they describe which modalities and spatial locations the model relies on over time, but they do not constitute causal explanations. Attribution therefore describes model behaviour; it does not establish causal effects on oil prices.

预测精度检验可以说明模型是否改进，但不能解释哪些信号推动了改进。应用文献中常见两条互补的归因路径。对扁平表格模型，SHAP (Lundberg and Lee, 2017) 可将预测归因到特征，并进一步聚合到变量组或模态层面。可学习的模态门控与站点/节点注意力为模型依赖提供原生诊断量（Arevalo et al., 2017；图注意力文献中的常规用法），用于描述随时间倚重哪些模态与空间位置；它们不构成因果解释。因此，归因描述的是模型行为，并不确立对油价的因果效应。

## 2.6 Synthesis, research gap and positioning



## 2.6 综合、研究空白与本文定位



### 2.6.1 Synthesis of the literature



### 2.6.1 文献综合

Four conclusions emerge from the review. Oil-price forecasting is difficult because oil prices are highly persistent and the random-walk (no-change) benchmark is strong. Financial variables provide an essential economically informed baseline because they capture persistence, uncertainty, monetary conditions, exchange-rate channels and market expectations. Shipping and remote-sensing data are plausible alternative-data sources, but they are noisy and indirect proxies rather than direct measurements of future prices. Multimodal learning offers tools for preserving modality-specific structure, but these tools have not been systematically tested in the specific setting of weekly Brent forecasting.

综合上述文献，可以得出四点结论。第一，原油价格预测很困难，因为原油价格高度持久，随机游走（无变化）基准很强。第二，金融变量构成必要的、有经济含义的基准，因为它们能够捕捉价格持续性、不确定性、货币条件、汇率渠道和市场预期。第三，航运和遥感数据是有潜力的替代数据来源，但它们是噪声较强、机制间接的代理变量，而不是未来价格的直接测量。第四，多模态学习为保留模态特定结构提供了工具，但这些工具尚未在周度 Brent 价格预测这一具体情境中得到系统检验。

The following table summarises the observable signal, economic channel and main limitation of each of the four literatures, with key citations for each strand.

下表总结了四类文献各自的可观测信号、经济渠道与主要限制，并为每条文献线索列出关键引用。


| Data source / literature           | Observable signal                                                                                                          | Economic channel                                                                        | Main limitation                                                             | Key references                                                                                                                                                 |
| ---------------------------------- | -------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------- | --------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Financial and oil-market variables | Lagged price, inventories, production/refinery activity, volatility, GPR, rates, exchange rates, futures/market indicators | Persistence, uncertainty, macro-financial conditions, market expectations               | Strong benchmark; difficult to improve upon                                 | Kilian (2009); Alquist et al. (2013); Baumeister and Kilian (2015); Costa et al. (2021); Yılmaz and Zehir (2026)                                               |
| Shipping / AIS / PortWatch         | Tanker flows, port calls, chokepoint transits, capacity-weighted activity                                                  | Physical trade, supply disruption, congestion, regional flow changes                    | Reverse causality, noisy cargo inference, missing AIS activity              | Adland et al. (2017); Yan et al. (2020); Arslanalp et al. (2019, 2026); Mi et al. (2022, 2023); Paolo et al. (2024); Ouyang et al. (2022); Liang et al. (2022) |
| Remote sensing                     | Night-time lights, NO₂, cloud cover, site-level imagery or embeddings                                                      | Industrial activity, demand conditions, inventory observability, infrastructure signals | Indirect mechanism, weak within-site temporal variation, cloud/missing data | Gibson et al. (2021); Polinov et al. (2022); Hao and Wang (2023); Wang et al. (2019); Bricongne et al. (2026); Jung (2026)                                     |
| Multimodal learning                | Modality-specific representations and fusion                                                                               | Preservation of heterogeneous structure before prediction                               | Limited direct evidence in oil-price forecasting                            | Baltrušaitis et al. (2019); Arevalo et al. (2017); Gohari et al. (2024); Cong et al. (2022); Fuller et al. (2023); Szwarcman et al. (2026); Ma et al. (2022)   |



| 数据来源 / 文献            | 可观测信号                                 | 经济渠道                    | 主要限制                      | 主要文献                                                                                                                                                           |
| -------------------- | ------------------------------------- | ----------------------- | ------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 金融与原油市场变量            | 滞后价格、库存、产量/炼厂活动、波动率、GPR、利率、汇率、期货/市场指标 | 价格持续性、不确定性、宏观金融条件、市场预期  | 基准很强，难以进一步改进              | Kilian (2009); Alquist et al. (2013); Baumeister and Kilian (2015); Costa et al. (2021); Yılmaz and Zehir (2026)                                               |
| 航运 / AIS / PortWatch | 油轮流量、港口停靠、咽喉通行、运力加权活动                 | 实物贸易、供给扰动、拥堵、区域流动变化     | 反向因果、货物推断有噪声、AIS 活动缺失     | Adland et al. (2017); Yan et al. (2020); Arslanalp et al. (2019, 2026); Mi et al. (2022, 2023); Paolo et al. (2024); Ouyang et al. (2022); Liang et al. (2022) |
| 遥感                   | 夜间灯光、NO₂、云量、站点影像或嵌入                   | 工业活动、需求条件、库存可观测性、基础设施信号 | 机制间接、站点内部时间变化较弱、云层/缺失数据问题 | Gibson et al. (2021); Polinov et al. (2022); Hao and Wang (2023); Wang et al. (2019); Bricongne et al. (2026); Jung (2026)                                     |
| 多模态学习                | 模态特定表示与融合                             | 在预测前保留异质数据结构            | 在原油价格预测中缺少直接证据            | Baltrušaitis et al. (2019); Arevalo et al. (2017); Gohari et al. (2024); Cong et al. (2022); Fuller et al. (2023); Szwarcman et al. (2026); Ma et al. (2022)   |




### 2.6.2 Research gap



### 2.6.2 研究空白

Existing oil-price forecasting studies have made progress in both financial modelling and machine-learning methods, but three linked gaps remain for weekly Brent forecasting with finance, shipping and remote sensing.

现有原油价格预测研究已经在金融建模和机器学习方法上取得进展，但在金融、航运与遥感共同进入周度 Brent 预测时，仍存在三个相互关联的空白。

First, the incremental value of alternative data is unclear. Few studies jointly report nested increments over a financial baseline and absolute skill against the random-walk benchmark under leakage-safe evaluation. Nested-only comparisons can overstate alternative data; random-walk-only comparisons can hide economically meaningful but weak signals.

第一，另类数据的增量价值不清楚。很少有研究在无泄漏评估下同时报告相对金融基线的嵌套增量与相对随机游走的绝对 skill。只做嵌套比较会夸大另类数据；只做随机游走比较可能掩盖经济上有意义但偏弱的信号。

Second, fusion architectures lack fair comparison. Most alternative-data applications still reduce shipping and satellite signals to engineered tabular features and concatenate them with financial predictors. Multi-source oil studies rarely compare flat feature fusion against representation-level modality-aware fusion under one shared protocol; common patterns are best-versus-best comparisons across model families, or only one fusion style. What remains unclear is whether any gain comes from the alternative data themselves, or from preserving modality-specific structure before prediction.

第二，融合架构缺少公平对照。多数替代数据应用仍将航运与卫星信号压缩为工程化表格特征，再与金融预测变量拼接。多源油价研究很少在同一共享协议下比较扁平特征融合与表示级模态感知融合；常见做法是跨模型族的冠军对决，或只做一种融合。因此尚不清楚收益究竟来自另类数据本身，还是来自在预测前保留模态特有结构。

Third, attribution is often detached from predictive value. Interpretability analyses are frequently presented for models without clear evidence that those models improve on the relevant forecast benchmarks, which weakens any claim that the highlighted signals are useful.

第三，归因常与预测价值脱节。可解释性分析经常针对尚未清楚证明优于相关预测基准的模型展开，这会削弱“所突出信号有用”的主张。

Together, these gaps point to a need for weekly Brent work that reports both nested and absolute comparisons, pairs flat and modality-aware fusion under one leakage-safe protocol, and links modality-level interpretation to models that already show predictive value against the no-change benchmark.

合起来，这些空白指向一项需求：周度 Brent 研究应同时报告嵌套与绝对比较，在同一无泄漏协议下配对扁平融合与模态感知融合，并将模态级解释与已相对无变化基准显示预测价值的模型联系起来。

### 2.6.3 Positioning of this dissertation



### 2.6.3 本文定位

This dissertation is positioned as an empirical integration and comparison study rather than a proposal of a new neural architecture. It brings together three strands of literature: the oil-forecasting literature's emphasis on strong baselines and rigorous out-of-sample testing; the alternative-data literature's use of shipping and satellite proxies for economic activity; and the multimodal-learning literature's distinction between flat feature fusion and representation-level modality-aware fusion.

本文定位为一项经验性整合与比较研究，而不是提出一种新的神经网络结构。它结合了三类文献：原油预测文献对强基准和严格样本外检验的强调；替代数据文献将航运和卫星信号作为经济活动代理变量的做法；以及多模态学习文献对扁平特征融合与表示层模态感知融合的区分。

The dissertation therefore asks three linked questions:

因此，本文提出三个相互关联的研究问题：

- **RQ1:** Do remote-sensing and shipping indicators add incremental out-of-sample value over a financial baseline and the random-walk benchmark?
- **RQ1：** 遥感和航运指标是否能在金融基准模型和随机游走基准之上提供样本外增量预测价值？
- **RQ2:** Does modality-aware representation-level fusion outperform flat feature fusion when both use the same underlying data?
- **RQ2：** 在使用相同底层数据的情况下，模态感知的表示层融合是否优于扁平特征融合？
- **RQ3:** Can modality-level interpretability reveal which signals the model relies on across different market conditions?
- **RQ3：** 模态层面的可解释性分析是否能够揭示模型在不同市场条件下依赖哪些信号？

Both claims — that alternative data are useful and that representation-level fusion is superior — are treated as empirical questions to be tested under a consistent evaluation framework, not as assumptions.

替代数据是否有用、表示层融合是否更优——这两个主张都被视为需要在统一评估框架下检验的经验问题，而不是本文的预设前提。

---



## Chapter 3 — Data and Methods *(~2,500–3,000 words)*

## 第 3 章 — 数据与方法 *(约 2,500–3,000 词)*

*Structure after Taylor: M0 defined in 3.1; former §3.5 merged here; data table in 3.3; hyperparameter “lock” explained; eval logic kept clear.*

*据 Taylor 调整结构：M0 在 3.1 定义；原 §3.5 并入此处；数据表放 3.3；解释超参“锁定”；评估逻辑保持清晰。*



### 3.1 Research design *(incl. M0 and M1–M4)*

### 3.1 研究设计 *(含 M0 与 M1–M4)*

**First — what M0 is**

**首先 — M0 是什么**

- **M0** = no-change / random-walk benchmark: next week’s price equals this week’s price, \(\hat{P}_{t+1}=P_t\) (≡ zero-return forecast). **Not trained.** Anchors **absolute skill**. Not one of the modality sets.

- **M0** = 无变化 / 随机游走基准：下周价格等于本周价格，\(\hat{P}_{t+1}=P_t\)（≡ 零收益预测）。**不训练。** 锚定**绝对 skill**。不属于任一模态信息集。

**Information sets M1–M4** *(former §3.5, now here)*

**信息集 M1–M4** *(原 §3.5，现并入此处)*

- M1: finance only
- M2: finance + remote sensing
- M3: finance + shipping
- M4: finance + remote sensing + shipping
- M2 and M3 are **parallel branches** from M1; M4 combines both.

- M1：仅金融
- M2：金融 + 遥感
- M3：金融 + 航运
- M4：金融 + 遥感 + 航运
- M2 与 M3 是自 M1 出发的**平行分支**；M4 二者并用。

**Two architecture families** (same M0–M4, timeline, metrics)

**两类架构族**（同 M0–M4、时间线、指标）

- **Flat:** put modality features in one table → Ridge / XGBoost (early fusion)
- **Deep:** modality-specific encoders → gated fusion (main) / cross-attention (alt) / concat (control)
- Within family, M1→M2/M3/M4 isolates *which* information; paired Flat vs Deep at matched sets isolates *how* it is represented.

- **Flat：** 模态特征入一张表 → Ridge / XGBoost（早融合）
- **Deep：** 模态专属编码器 → 门控融合（主）/ 交叉注意力（对照）/ concat（控制）
- 族内 M1→M2/M3/M4 分离*用了哪些*信息；匹配集上配对 Flat vs Deep 分离*如何*表示。

**How each RQ is tested**

**各 RQ 如何检验**

1. **RQ1:** within each family, compare M2/M3/M4 vs finance-only M1, and all models vs M0
2. **RQ2:** Flat vs Deep pairwise at matched sets (e.g. M3_Flat vs M3_Deep)
3. **RQ3:** interpretability only on specs that beat M0 (Deep M3, and Deep M4 where relevant)

1. **RQ1：** 各族内比较 M2/M3/M4 相对仅金融 M1，以及全部模型相对 M0
2. **RQ2：** 匹配集上两两比较 Flat vs Deep（如 M3_Flat vs M3_Deep）
3. **RQ3：** 可解释性仅用于打过 M0 的设定（Deep M3，以及相关时的 Deep M4）



### 3.2 Prediction target and timeline

### 3.2 预测目标与时间线

- Target: next-week Brent \(P_{t+1}\); train on \(r_{t+1}=\log(P_{t+1}/P_t)\); evaluate reconstructed price
- M0 equivalence: \(\hat{P}_{t+1}=P_t\) ≡ zero return (same definition as §3.1)
- Friday weeks; 2019–2025; 257-week scored span

- 目标：下周 Brent \(P_{t+1}\)；用 \(r_{t+1}=\log(P_{t+1}/P_t)\) 训练；评估重建价格
- M0 等价：\(\hat{P}_{t+1}=P_t\) ≡ 零收益（与 §3.1 定义一致）
- 周五周；2019–2025；257 周计分区间



### 3.3 Data sources

### 3.3 数据来源

- Three modalities; Flat and Deep share underlying sources, differ in representation.
- **Table 3.1 — Datasets, variables and sources** *(include in chapter body)*

- 三种模态；Flat 与 Deep 共享底层来源，表示方式不同。
- **表 3.1 — 数据集、变量与来源** *(写入正文)*

| Modality / 模态 | Dataset / product / 数据集 | Key variables (outline-level) / 关键变量（大纲级） | Source / 来源 |
| --- | --- | --- | --- |
| Finance / 金融 | Oil-market & macro weekly series / 油市与宏观周度序列 | Prices, inventory, production, rates, related indicators / 价格、库存、产量、利率及相关指标 | EIA, FRED, Yahoo, etc. |
| Remote sensing (Flat) / 遥感（Flat） | Sentinel-2 optical indices + VIIRS NTL / Sentinel-2 光学指数 + VIIRS 夜光 | Site anomalies at 11 AOIs (NDVI/NDWI/NDBI/BSI; NTL) / 11 个 AOI 站点异常 | Sentinel-2; VIIRS |
| Remote sensing (Deep) / 遥感（Deep） | Frozen Prithvi-EO-2.0 embeddings / 冻结 Prithvi-EO-2.0 嵌入 | Monthly S2 image-patch embeddings at same 11 AOIs (no VIIRS) / 同 11 个 AOI 的月度 S2 影像块嵌入（无 VIIRS） | Prithvi-EO-2.0 / Sentinel-2 |
| Shipping (Flat) / 航运（Flat） | PortWatch + AIS tabular features / PortWatch + AIS 表格特征 | Port / chokepoint tanker flows; vessel activity features / 港口 / 咽喉油轮流量；船舶活动特征 | IMF PortWatch; AIS |
| Shipping (Deep) / 航运（Deep） | Same sources as graph / 同来源构图 | 17-node heterogeneous graph (11 AOIs + 6 chokepoints) / 17 节点异质图（11 AOI + 6 咽喉） | PortWatch; AIS |

- Full variable dictionaries, AOI / chokepoint lists → Appendix A; Deep shipping graph edges → §3.6 / Appendix A.

- 完整变量词典、AOI / 咽喉列表 → 附录 A；Deep 航运图边 → §3.6 / 附录 A。



### 3.4 Temporal alignment, lags, missingness

### 3.4 时间对齐、滞后与缺失

- Release-timestamp alignment; modality-specific lags (e.g. EIA / PortWatch ~1 week)
- Flat: past-only fill; Deep: explicit missing masks

- 按发布时间戳对齐；模态专属滞后（如 EIA / PortWatch ~1 周）
- Flat：仅用过去信息填补；Deep：显式缺失掩码



### 3.5 Flat models

### 3.5 Flat 模型

- 4-week lookback flattened into one row; Ridge + XGBoost; tune per fold on past validation only → Appendix C

- 4 周回看压成一行；Ridge + XGBoost；每折仅在过去验证集上调参 → 附录 C



### 3.6 Deep models

### 3.6 Deep 模型

- Finance: causal TCN; RS: per-site temporal + site attention on frozen embeddings; Shipping: GAT on heterogeneous graph
- Fusion: gated (main text), cross-attention (comparative), concat (control)

- 金融：因果 TCN；遥感：冻结嵌入上的站点时序 + 站点注意力；航运：异质图上的 GAT
- 融合：门控（正文主结果）、交叉注意力（对照）、concat（控制）



### 3.7 Hyperparameter selection

### 3.7 超参数选择

- **Flat:** tune inside each training fold on past validation weeks only; grids in Appendix C.
- **Deep — what “lock a main configuration” means:** because full per-fold neural search is costly, run limited sweeps first, then **fix one main setting** (architecture depths / representation size / main fusion choice, etc.) for the primary reported results; afterwards report **sensitivity** to seed, lookback, fusion type, representation size and regularisation in Results / Appendix C. Not arbitrary: search → lock main → sensitivity.
- Flat and Deep follow the same protocol so the paired comparison stays fair.

- **Flat：** 每折训练内仅在过去验证周上调参；网格见附录 C。
- **Deep — “锁定主配置”指什么：** 因逐折全量神经搜索成本高，先做有限扫描，再**固定一套主设定**（架构深度 / 表示维度 / 主融合选择等）用于主要报告结果；之后在结果 / 附录 C 报告对种子、回看、融合类型、表示维度与正则的**敏感性**。非随意：搜索 → 锁定主配置 → 敏感性。
- Flat 与 Deep 遵循同一协议，以保持配对比较公平。



### 3.8 Leakage-free validation protocol

### 3.8 无泄漏验证协议

- Expanding-window rolling-origin; first 104 weeks init + validation (not scored); refit every 13 weeks; scaling fit in-fold only
- Common scored span: 257 weeks; Flat and Deep share the same fold calendar

- 扩展窗滚动起点；前 104 周作初始化 + 验证（不计分）；每 13 周重拟合；缩放仅在折内拟合
- 共用计分区间：257 周；Flat 与 Deep 共享同一折日历



### 3.9 Evaluation, tests, interpretability

### 3.9 评估、检验与可解释性

- **Metrics:** RMSE, MAE on reconstructed price; DirAcc auxiliary.
- **Skill vs M0:** \(\mathrm{Skill}=100\times(1-\mathrm{RMSE}_{\mathrm{model}}/\mathrm{RMSE}_{\mathrm{M0}})\); >0 beats M0; =0 matches; <0 worse.
- **Comparison logic:** report both incremental value **versus M1** and absolute skill **versus M0**. Information-set nesting (adding a modality) does **not** automatically imply nested forecast testing.
- **Tests:** **CW** for nested increments (e.g. Ridge M1 vs M2/M3/M4); **DM** for non-nested pairs (Flat vs Deep); always report RMSE/MAE effect sizes.
- **Interpretability:** only on specs that improve on M0 (Deep M3, and Deep M4 where relevant) — modality gates + site/node attention.
- **Reproducibility:** public/licensed data; scripted pipeline.

- **指标：** 重建价格上的 RMSE、MAE；DirAcc 为辅助。
- **相对 M0 的 skill：** \(\mathrm{Skill}=100\times(1-\mathrm{RMSE}_{\mathrm{model}}/\mathrm{RMSE}_{\mathrm{M0}})\)；>0 打过 M0；=0 持平；<0 更差。
- **比较逻辑：** 同时报告相对 **M1** 的增量与相对 **M0** 的绝对 skill。信息集嵌套（加模态）**并不**自动意味着嵌套预测检验。
- **检验：** **CW** 用于嵌套增量（如 Ridge M1 vs M2/M3/M4）；**DM** 用于非嵌套配对（Flat vs Deep）；始终报告 RMSE/MAE 效应量。
- **可解释性：** 仅用于相对 M0 有改善的设定（Deep M3，以及相关时的 Deep M4）——模态门控 + 站点/节点注意力。
- **可复现性：** 公开/授权数据；脚本化流水线。

---



## Chapter 4 — Results *(~2,500–3,200 words)*

## 第 4 章 — 结果 *(约 2,500–3,200 词)*

*Organised by RQ logic — not “Flat block then Deep block” only.*

*按 RQ 逻辑组织——而非仅“先 Flat 块再 Deep 块”。*

### 4.1 Descriptive overview

### 4.1 描述性概览

- 257-week scored span; one-week-ahead Brent; M0 anchor; RMSE / MAE / skill / CW / DM

- 257 周计分区间；一周前瞻 Brent；M0 锚定；RMSE / MAE / skill / CW / DM



### 4.2 Flat results *(RQ1)*

### 4.2 Flat 结果 *(RQ1)*

*Keep this narrative order (Taylor: clearly presented).*

*保持该叙述顺序（Taylor：呈现清晰）。*

- Absolute: **no Flat model beats M0** (all negative skill vs no-change)
- Nested within Flat: M3 > M1 (shipping helps vs finance-only); M2 ≈ no gain over M1 (RS weak)
- Key numbers (XGB RMSE / **skill vs M0**): M0 4.152; M1 −5.2%; M2 −6.9%; M3 −6.7%; M4 −8.6%
- CW significance ≠ beating M0
- Table 4.1 — Flat out-of-sample performance

- 绝对：**无一 Flat 模型打过 M0**（相对无变化 skill 均为负）
- Flat 内嵌套：M3 > M1（航运相对仅金融有增益）；M2 ≈ 相对 M1 无增益（遥感弱）
- 关键数字（XGB RMSE / **相对 M0 skill**）：M0 4.152；M1 −5.2%；M2 −6.9%；M3 −6.7%；M4 −8.6%
- CW 显著 ≠ 打过 M0
- 表 4.1 — Flat 样本外表现



### 4.3 Deep results *(RQ1, deep arm)*

### 4.3 Deep 结果 *(RQ1，Deep 臂)*

- M2 fails M0; **M3 beats M0** (gated +0.11%, xattn +0.74%); M4 does not clearly dominate M3
- Key numbers (gated **skill vs M0**): M1 −2.4%; M2 −2.4%; M3 **+0.11%**; M4 −1.3%
- Table 4.2 — Deep out-of-sample performance

- M2 未过 M0；**M3 打过 M0**（门控 +0.11%，交叉注意力 +0.74%）；M4 未明显主导 M3
- 关键数字（门控 **相对 M0 skill**）：M1 −2.4%；M2 −2.4%；M3 **+0.11%**；M4 −1.3%
- 表 4.2 — Deep 样本外表现



### 4.4 Flat vs Deep *(core RQ2)*

### 4.4 Flat vs Deep *(核心 RQ2)*

- Paired at matched sets (XGB Flat vs gated Deep). **Table 4.3** columns:
  - Flat RMSE | Deep RMSE | **Flat skill vs M0 (%)** | **Deep skill vs M0 (%)**
- *Clarification for readers:* the two % columns are each model’s RMSE skill **versus the no-change benchmark M0** — **not** the % change in RMSE between Flat and Deep.
- Numbers (skill vs M0): M1 −5.2% vs −2.4% | M2 −6.9% vs −2.4% | M3 −6.7% vs **+0.11%** | M4 −8.6% vs −1.3%
- Deep advantage concentrated in shipping-inclusive multimodal settings

- 匹配集配对（XGB Flat vs 门控 Deep）。**表 4.3** 列：
  - Flat RMSE | Deep RMSE | **Flat 相对 M0 skill (%)** | **Deep 相对 M0 skill (%)**
- *给读者的澄清：* 两列 % 分别是各模型相对**无变化基准 M0** 的 RMSE skill——**不是** Flat 与 Deep 之间 RMSE 的百分比变化。
- 数字（相对 M0 skill）：M1 −5.2% vs −2.4% | M2 −6.9% vs −2.4% | M3 −6.7% vs **+0.11%** | M4 −8.6% vs −1.3%
- Deep 优势集中在含航运的多模态设定



### 4.5 Robustness

### 4.5 稳健性

- Flat: lookback, RS variants, shipping tiers, LOAO — pattern holds
- Deep: seeds, lookback, repr size, fusion type, early/late windows — gated M3 more stable than xattn across seeds

- Flat：回看、遥感变体、航运分层、LOAO——模式成立
- Deep：种子、回看、表示维度、融合类型、早/晚期窗——门控 M3 跨种子比交叉注意力更稳



### 4.6 Interpretability *(RQ3)*

### 4.6 可解释性 *(RQ3)*

- **Modality gate weights** (not vague “allocation”): mean ≈ **0.56 finance / 0.44 shipping** over the test span — the share of fusion weight the model places on each modality.
- **Time variation:** these gate weights shift around stress events (Russia–Ukraine war announcement, EU Russia oil-ban, OPEC+ surprise cut, Houthi Red Sea attacks); shipping weight tends to rise or become more volatile in those windows — interesting but descriptive only.
- **Shipping node attention:** concentrates on major chokepoints — Hormuz, Suez, Bab el-Mandeb, Panama, Cape of Good Hope.
- Applied only on M0-beating specs (Deep M3, Deep M4 where relevant). **Model dependence ≠ causal effects** on oil prices.

- **模态门控权重**（非笼统“分配”）：测试区间均值约 **0.56 金融 / 0.44 航运**——模型对各模态融合权重的份额。
- **时变：** 门控权重在压力事件附近变动（俄乌开战公告、欧盟对俄石油禁令、OPEC+ 意外减产、胡塞红海袭击）；航运权重在这些窗口往往上升或更波动——有趣但仅描述性。
- **航运节点注意力：** 集中于主要咽喉——霍尔木兹、苏伊士、曼德海峡、巴拿马、好望角。
- 仅用于打过 M0 的设定（Deep M3，相关时 Deep M4）。**模型依赖 ≠ 对油价的因果效应**。

---



## Chapter 5 — Discussion *(~1,600–1,900 words)*

## 第 5 章 — 讨论 *(约 1,600–1,900 词)*



### 5.1 RQ1 — Do alternative data help?

### 5.1 RQ1 — 另类数据是否有用？

- Flat: M0 best; shipping helps vs M1; RS does not
- Deep: small M3/M4 gain vs M0; RS secondary to shipping; absolute gains limited

- Flat：M0 最优；航运相对 M1 有增益；遥感无
- Deep：M3/M4 相对 M0 有小增益；遥感次于航运；绝对增益有限



### 5.2 RQ2 — Representation-level fusion vs flat?

### 5.2 RQ2 — 表示级融合 vs 扁平？

- Paired evidence: Deep wins when shipping enters; finance-only Deep gain limited
- Advantage from preserving temporal / site / network structure + fusion — not different data

- 配对证据：航运进入后 Deep 胜出；仅金融时 Deep 增益有限
- 优势来自保留时间 / 站点 / 网络结构 + 融合——而非不同数据



### 5.3 RQ3 — What does the model rely on?

### 5.3 RQ3 — 模型依赖什么？

- Deep M3 (and M4) gates + spatial attention; shipping weight rises in stress windows
- Model dependence, not causal identification

- Deep M3（及 M4）门控 + 空间注意力；航运权重在压力窗口上升
- 模型依赖，非因果识别



### 5.4 Implications

### 5.4 启示

**Method / evidence design**

**方法 / 证据设计**

- Fair protocol matters as much as fusion design; nested M1 + absolute M0 contrasts jointly informative
- Null Flat results informative: early fusion ≠ automatic value

- 公平协议与融合设计同等重要；嵌套 M1 + 绝对 M0 对照共同有信息量
- Flat 的空结果亦有信息：早融合 ≠ 自动有价值

**Broader use — who benefits / how the framework can be used**

**更广用途 — 谁受益 / 框架如何用**

- **Energy / commodity analysts and risk teams:** a template to test whether multimodal alternative data beat a hard weekly no-change bar before acting on “signals”
- **Alternative-data / EO product providers:** nest gains over a finance baseline *and* report skill vs M0, to avoid overselling nested-only improvements
- **Methods researchers:** reusable paired Flat vs modality-aware Deep design at matched information sets (transferable to other commodities / horizons)
- **Monitoring (cautious):** gate and chokepoint attention as exploratory stress diagnostics — not trading rules or causal claims

- **能源 / 大宗分析师与风险团队：** 在依据“信号”行动前，检验多模态另类数据是否越过周度无变化硬门槛的模板
- **另类数据 / EO 产品方：** 相对金融基线报嵌套增益，*并*报告相对 M0 的 skill，避免夸大仅嵌套改善
- **方法研究者：** 匹配信息集上可复用的 Flat vs 模态感知 Deep 配对设计（可迁移到其他商品 / 视界）
- **监测（谨慎）：** 门控与咽喉注意力作为探索性压力诊断——非交易规则或因果主张



### 5.5 Limitations

### 5.5 局限

- Weekly horizon; modest scored sample; proxy noise / reverse causality; frozen EO; graph choices; seed sensitivity (xattn)

- 周度视界；计分样本有限；代理噪声 / 反向因果；冻结 EO；图结构选择；种子敏感（交叉注意力）



### 5.6 Future research

### 5.6 未来研究

- Longer history; richer AIS graph; missing-modality stress; other horizons / commodities

- 更长历史；更丰富 AIS 图；缺模态压力；其他视界 / 商品

---



## Chapter 6 — Conclusion *(~400–700 words)*

## 第 6 章 — 结论 *(约 400–700 词)*

*No subsections (Taylor): a few continuous paragraphs summarising findings and implications.*

*无分节（Taylor）：用若干连续段落总结发现与启示。*

**Paragraph 1 — key findings.** Under Flat early fusion, no learned model beats the no-change benchmark M0, though shipping still improves on finance-only M1 while remote sensing does not. Under Deep, M0 remains strong; finance + shipping (M3) shows only a small positive skill versus M0, and adding RS (M4) does not clearly dominate M3. At matched multimodal sets, Deep outperforms Flat most clearly once shipping enters. Where forecasts beat M0, modality gates and chokepoint attention describe when and where the model relies on shipping.

**段落 1 — 关键发现。** Flat 早融合下，无一学习模型打过无变化基准 M0，尽管航运仍相对仅金融 M1 有改善而遥感无。Deep 下 M0 仍强；金融 + 航运（M3）相对 M0 仅有小幅正 skill，加入遥感（M4）未明显主导 M3。在匹配的多模态集上，航运进入后 Deep 对 Flat 优势最清晰。在预测打过 M0 处，模态门控与咽喉注意力描述模型何时、何处依赖航运。

**Paragraph 2 — method and broader implications.** The thesis contribution is a leakage-safe nested multimodal comparison plus a paired Flat–Deep contrast at matched information sets, with interpretability kept aligned with predictive evidence. The same protocol can help analysts and data providers judge alternative-data claims against both a finance baseline and the random walk, and can be reused as a comparison template beyond weekly Brent.

**段落 2 — 方法与更广启示。** 本论贡献是无泄漏嵌套多模态对照，加上匹配信息集上的配对 Flat–Deep 对比，并使可解释性与预测证据对齐。同一协议可帮助分析师与数据方同时相对金融基线与随机游走判断另类数据主张，并可复用为周度 Brent 之外的对照模板。

**Paragraph 3 — closing.** In this weekly design, shipping adds incremental value over finance and modality-aware Deep fusion can yield a small further gain versus M0; remote sensing is unstable. Gains are modest — useful evidence, not breakthrough forecasting.

**段落 3 — 收束。** 在本周度设计中，航运相对金融有增量价值，模态感知 Deep 融合相对 M0 可再带来小幅增益；遥感不稳定。增益有限——有用证据，而非突破性预测。

---



## References

## 参考文献

Adland, R., Jia, H. and Strandenes, S.P. (2017). ‘Are AIS-based trade volume estimates reliable? The case of crude oil exports’, *Maritime Policy & Management*, 44(5), pp. 657–665. doi: 10.1080/03088839.2017.1309470.

Alquist, R., Kilian, L. and Vigfusson, R.J. (2013). ‘Forecasting the price of oil’, in Elliott, G. and Timmermann, A. (eds.) *Handbook of Economic Forecasting*. Vol. 2A. Amsterdam: Elsevier, pp. 427–507. doi: 10.1016/B978-0-444-53683-9.00008-6.

Arevalo, J., Solorio, T., Montes-y-Gómez, M. and González, F.A. (2017). ‘Gated multimodal units for information fusion’, *ICLR 2017 Workshop Track*. Toulon, France, 24–26 April. Available at: [https://openreview.net/forum?id=S12_nquOe](https://openreview.net/forum?id=S12_nquOe) (Accessed: 1 July 2026).

Arslanalp, S., Marini, M. and Tumbarello, P. (2019). *Big data on vessel traffic: nowcasting trade flows in real time*. IMF Working Paper WP/19/275. Washington, DC: International Monetary Fund. Available at: [https://www.imf.org/en/publications/wp/issues/2019/12/13/big-data-on-vessel-traffic-nowcasting-trade-flows-in-real-time-48837](https://www.imf.org/en/publications/wp/issues/2019/12/13/big-data-on-vessel-traffic-nowcasting-trade-flows-in-real-time-48837) (Accessed: 1 July 2026).

Arslanalp, S., Exton, O., Gao, C., Kamali, P., Saraiva, M., Sozzi, A. and Verschuur, J. (2026). *Nowcasting country-level trade estimates using IMF PortWatch*. IMF Working Paper WP/26/99. Washington, DC: International Monetary Fund. doi: 10.5089/9798229046893.001.

Baltrušaitis, T., Ahuja, C. and Morency, L.-P. (2019). ‘Multimodal machine learning: a survey and taxonomy’, *IEEE Transactions on Pattern Analysis and Machine Intelligence*, 41(2), pp. 423–443. doi: 10.1109/TPAMI.2018.2798607.

Baumeister, C. and Kilian, L. (2015). ‘Forecasting the real price of oil in a changing world: a forecast combination approach’, *Journal of Business & Economic Statistics*, 33(3), pp. 338–351. doi: 10.1080/07350015.2014.949342.

Bricongne, J.-C., Macalos, J., Meunier, B., Milis, J. and Pical, T. (2026). *Can satellites predict oil demand?* ECB Working Paper Series No. 3198. Frankfurt am Main: European Central Bank. Available at: [https://www.ecb.europa.eu/pub/pdf/scpwps/ecb.wp3198~e3858c52a3.en.pdf](https://www.ecb.europa.eu/pub/pdf/scpwps/ecb.wp3198~e3858c52a3.en.pdf) (Accessed: 1 July 2026).

Che, Z., Purushotham, S., Cho, K., Sontag, D. and Liu, Y. (2018). ‘Recurrent neural networks for multivariate time series with missing values’, *Scientific Reports*, 8, 6085. doi: 10.1038/s41598-018-24271-9.

Clark, T.E. and West, K.D. (2007). ‘Approximately normal tests for equal predictive accuracy in nested models’, *Journal of Econometrics*, 138(1), pp. 291–311. doi: 10.1016/j.jeconom.2006.05.023.

Cong, Y., Khanna, S., Meng, C., Liu, P., Rozi, E., He, Y., Burke, M., Lobell, D.B., et al. (2022). ‘SatMAE: pre-training transformers for temporal and multi-spectral satellite imagery’, *Advances in Neural Information Processing Systems*, 35, pp. 197–211.

Costa, A.B.R., Ferreira, P.C.G., Gaglianone, W.P., Guillén, O.T.C., Issler, J.V. and Lin, Y. (2021). ‘Machine learning and oil price point and density forecasting’, *Energy Economics*, 102, 105494. doi: 10.1016/j.eneco.2021.105494.

Diebold, F.X. and Mariano, R.S. (1995). ‘Comparing predictive accuracy’, *Journal of Business & Economic Statistics*, 13(3), pp. 253–263. doi: 10.1080/07350015.1995.10524599.

Foroutan, P. and Lahmiri, S. (2024). ‘Deep learning systems for forecasting the prices of crude oil and precious metals’, *Financial Innovation*, 10, 111. doi: 10.1186/s40854-024-00637-z.

Fuller, A., Millard, K. and Green, J.R. (2023). ‘CROMA: remote sensing representations with contrastive radar-optical masked autoencoders’, *Advances in Neural Information Processing Systems*, 36, pp. 5506–5538.

Gibson, J., Olivia, S., Boe-Gibson, G. and Li, C. (2021). ‘Which night lights data should we use in economics, and where?’, *Journal of Development Economics*, 149, 102602. doi: 10.1016/j.jdeveco.2020.102602.

Gohari, H.E., Dang, X.-H., Shah, S.Y. and Zerfos, P. (2024). ‘Modality-aware transformer for financial time series forecasting’, in *Proceedings of the 5th ACM International Conference on AI in Finance (ICAIF ’24)*. New York: Association for Computing Machinery, pp. 677–685. doi: 10.1145/3677052.3698654.

Hao, X. and Wang, Y. (2023). ‘Cloud cover and expected oil returns’, *Humanities and Social Sciences Communications*, 10, 605. doi: 10.1057/s41599-023-02128-5.

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

Polinov, S., Bookman, R. and Levin, N. (2022). ‘A global assessment of night lights as an indicator for shipping activity in anchorage areas’, *Remote Sensing*, 14(5), 1079. doi: 10.3390/rs14051079.

Shukla, S.N. and Marlin, B.M. (2021). ‘Multi-time attention networks for irregularly sampled time series’, *International Conference on Learning Representations (ICLR 2021)*. Online, 3–7 May. Available at: [https://openreview.net/forum?id=4c0J6lwQ4](https://openreview.net/forum?id=4c0J6lwQ4)_ (Accessed: 1 July 2026).

Simsek, A.I., Bulut, E., Gur, Y.E. and Gültekin Tarla, E. (2024). ‘A novel approach to predict WTI crude spot oil price: LSTM-based feature extraction with Xgboost regressor’, *Energy*, 309, 133102. doi: 10.1016/j.energy.2024.133102.

Szwarcman, D., Roy, S., Fraccaro, P., Gíslason, Þ.E., Blumenstiel, B., Ghosal, R., de Oliveira, P.H., de Sousa Almeida, J.L., et al. (2026). ‘Prithvi-EO-2.0: a versatile multitemporal foundation model for earth observation applications’, *IEEE Transactions on Geoscience and Remote Sensing*, 64, 4400120. doi: 10.1109/TGRS.2025.3642610.

Wang, T., Li, Y., Yu, S. and Liu, Y. (2019). ‘Estimating the volume of oil tanks based on high-resolution remote sensing images’, *Remote Sensing*, 11(7), 793. doi: 10.3390/rs11070793.

Wu, Z., Pan, S., Long, G., Jiang, J. and Zhang, C. (2019). ‘Graph WaveNet for deep spatial-temporal graph modeling’, in *Proceedings of the Twenty-Eighth International Joint Conference on Artificial Intelligence (IJCAI-19)*. Macao, China, 10–16 August. International Joint Conferences on Artificial Intelligence Organization, pp. 1907–1913. doi: 10.24963/ijcai.2019/264.

Yan, Z., Xiao, Y., Cheng, L., Chen, S., Zhou, X., Ruan, X., Li, M., He, R., et al. (2020). ‘Analysis of global marine oil trade based on automatic identification system (AIS) data’, *Journal of Transport Geography*, 83, 102637. doi: 10.1016/j.jtrangeo.2020.102637.

Yılmaz, T.E. and Zehir, C. (2026). ‘Strategic risk based forecasting of Brent crude oil prices: a comparative analysis of econometric and machine learning models’, *Entropy*, 28(5), 539. doi: 10.3390/e28050539.

Zhao, G., Xue, M. and Cheng, L. (2023). ‘A new hybrid model for multi-step WTI futures price forecasting based on self-attention mechanism and spatial–temporal graph neural network’, *Resources Policy*, 85, 103956. doi: 10.1016/j.resourpol.2023.103956.



## Appendices

## 附录

- **A.** Variable dictionaries; AOI / chokepoint lists; lag table; shipping graph edge definition
- **B.** Extra robustness tables & figures (lookback, LOAO, LOCHO, LOMO, water-mask RS, fusion matrix, seeds, early/late)
- **C.** Hyperparameter grids and locked Deep settings

- **A.** 变量词典；AOI / 咽喉列表；滞后期表；航运图边定义
- **B.** 额外稳健性表与图（回看、LOAO、LOCHO、LOMO、水体掩膜遥感、融合矩阵、种子、早/晚期）
- **C.** 超参网格与锁定的 Deep 设定
