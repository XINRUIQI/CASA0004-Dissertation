# A Modality-Aware Spatio-Temporal Fusion Framework for Brent Crude Oil Forecasting Using Financial Time Series, Satellite Imagery and Maritime Networks

# 面向金融时序、卫星遥感与航运网络的模态感知时空融合框架：Brent 原油价格预测

---

## Abstract *(~200 words)*

## 摘要 *（约 200 词）*

Brent crude is the main benchmark for internationally traded oil, and its short-term movements affect hedging, budgeting and market-risk decisions. The 2019–2025 period spans the COVID-19 pandemic, the 2022 energy crisis and subsequent market adjustment. This dissertation asks whether satellite remote sensing and maritime shipping data add predictive information beyond financial time series for one-week-ahead Brent price forecasts. It also compares flat feature-level fusion with modality-specific encoding before fusion. Ridge and XGBoost represent the flat approach; deep models encode the three data sources separately and then fuse them. A financial-time-series-only specification is compared with alternatives adding remote sensing, shipping, or both. All models follow the same rolling-origin out-of-sample protocol, use only information available at each forecast date, and are evaluated against a no-change benchmark that sets next week’s price equal to this week’s. On a common evaluation sample, no flat model outperforms this benchmark. Shipping still improves accuracy relative to the financial-time-series-only specification, whereas remote sensing does not. Deep models combining financial time series and shipping data achieve a small gain over the benchmark, but adding remote sensing brings no clear further improvement. The advantage of deep models over flat models is clearest when shipping data are included. Where deep models improve on the benchmark, modality gates and spatial attention are used to show which sources the forecasts rely on. Overall, predictive value depends on the information source and fusion design, not simply on adding more data.

Brent 原油是国际贸易原油的主要定价基准，其短期波动影响对冲、预算与市场风险决策。2019–2025 年覆盖新冠疫情、2022 年能源危机及随后的市场调整。本论检验卫星遥感与航运数据，能否在金融时序之外，为提前一周的 Brent 价格预测提供增量信息；并比较扁平特征级融合与先分模态编码再融合。Ridge 与 XGBoost 代表扁平路径；深度模型对三类数据源分别编码后再融合。仅用金融时序的设定与加入遥感、航运或二者的备选设定相对照。所有模型共用同一滚动起点样本外协议，仅使用各预测日当时可获信息，并以“下周价格等于本周价格”的不变预测为评价基准。在共同评价样本上，无扁平模型优于该基准；航运相对仅金融时序设定仍改善精度，遥感则没有。深度模型在金融时序加航运时相对基准有小幅增益，再加遥感并无明显进一步改善。深度相对扁平的优势在纳入航运时最清晰。在深度模型打过基准之处，用模态门控与空间注意力展示预测依赖哪些数据源。总体而言，预测价值取决于信息源与融合设计，而非单纯堆加更多数据。

---

## Chapter 1 — Introduction *(~600 words)*

## 第 1 章 — 绪论 *(约 600 词)*

### 1.1 Importance and background

### 1.1 重要性与背景

Crude oil occupies a central place in the world economy. Movements in oil prices affect inflation, trade balances, fiscal revenues in producer countries and the operating costs of energy-intensive industries. These effects transmit quickly through financial markets, real activity and supply chains. Oil-price forecasting therefore matters in energy economics, and for governments, firms and investors concerned with risk management, hedging and planning.

原油在世界经济中占据核心位置。油价变动影响通胀、贸易差额、产油国财政收入以及高耗能行业的运营成本。这些影响会迅速传导至金融市场、实体活动与供应链。因此油价预测在能源经济学中具有重要性，也关系到政府、企业与投资者的风险管理、对冲与规划。

Oil remains a core commodity in the global energy system. Recent years have underlined how costly price surprises can be. The COVID-19 period brought an abrupt demand collapse and an uneven recovery. The 2022 energy crisis then produced a sharp supply and price shock. The years that followed saw only partial normalisation under continued geopolitical and macroeconomic uncertainty.

石油仍是全球能源体系中的核心商品。近年经历再次表明价格意外的代价有多大。新冠时期出现需求骤降与不均衡复苏。2022 年能源危机带来剧烈的供给与价格冲击。随后几年在持续的地缘与宏观不确定性下只是部分回归常态。

That uncertainty is not only historical. Recent conflict has again shown how quickly oil prices and seaborne trade can move when key maritime choke points are disrupted or avoided. Governments watch such shocks for inflation control, fiscal planning, energy security and trade policy. A better short-term oil-price model would help them gauge risk and timing. It would not replace market judgment, but it could support planning when physical flows and prices shift together. In such an environment, claims that new data or more elaborate models improve forecasts need to be tested carefully against strong and transparent benchmarks.

这种不确定性并非只属于历史。近期冲突再次表明，当关键航运咽喉受阻或被绕行时，油价与海运贸易会迅速变动。政府出于通胀管理、财政规划、能源安全与贸易政策需要监测此类冲击。更可靠的短期油价模型有助于它们评估风险与时机。它不能替代市场判断，但可在实物流动与价格同步变动时支持规划。在这种环境下，任何“新数据”或“更复杂模型”改善预测的主张，都必须对照强且透明的基准加以检验。

Among crude-oil benchmarks, Brent serves as the global pricing benchmark for a large share of internationally traded oil. This dissertation focuses on Friday-ending weekly Brent spot prices over 2019–2025 and on one-week-ahead out-of-sample forecasts. At the weekly horizon, the no-change forecast is difficult to outperform. That simple rule—predicting that next week’s price equals this week’s price—is a demanding reference point. Any claim that alternative data or a new fusion method helps must clear this bar, not only improve on a weaker or differently specified competitor.

在原油定价基准中，Brent 是大部分国际贸易原油的全球定价基准。本论聚焦 2019–2025 年周五截止的周度 Brent 现货，以及提前一周的样本外预测。在周度尺度上，不变预测很难被超越。这一简单规则——即预测下周价格等于本周价格——是一道很高的门槛。任何关于另类数据或新融合方法“有用”的主张，都必须越过这道门槛，而不能只相对更弱或设定不同的对手取得改善。

These data provide complementary views of the oil system. Financial, macroeconomic and oil-market variables describe market conditions over time. Remote sensing represents spatial activity at specific oil-related sites through spectral indicators, night-time lights and image embeddings. AIS and PortWatch data describe time-varying vessel activity across ports and major chokepoints, including the network relationships between them, and serve as proxies for seaborne trade flows and congestion. Chapter 3 maps the eleven monitoring sites and six maritime chokepoints used in these spatial inputs. In this dissertation, multimodal forecasting therefore refers to combining temporal market data, spatial Earth-observation data and spatiotemporal shipping-network data within the same forecasting task.

这些数据从不同角度描述石油体系。金融、宏观经济和石油市场变量反映随时间变化的市场状况；遥感数据通过光谱指标、夜光与影像嵌入表征特定石油相关地点的空间活动；AIS 与 PortWatch 数据则描述港口和主要航运咽喉上的时变船舶活动及其网络关系，并作为海运贸易流与拥堵的代理。第 3 章给出这些空间输入所用的十一个监测站点与六个航运咽喉的分布图。因此，本文所称的多模态预测，是指在同一预测任务中融合时间性的市场数据、空间性的对地观测数据，以及具有时空网络结构的航运数据。

Two practical difficulties follow. First, the signals are noisy and arrive on different schedules. They may also respond to prices rather than lead them, and the weekly sample is relatively small. Second, a common approach is to organise heterogeneous inputs in a single feature table and combine them before modelling. That flat early-fusion approach is convenient for classical models such as Ridge regression or gradient-boosted trees. It does not explicitly model modality-specific structure. This includes temporal dynamics in financial time series, site structure in remote sensing, and network structure in shipping.

由此带来两点实践困难。第一，信号嘈杂且到达节奏不一。它们也可能是对价格的响应而非领先，同时周度样本量相对较小。第二，一种常见做法是把异质输入整理成一张特征表，并在建模前加以组合。这种扁平早融合便于 Ridge 或梯度提升树等经典模型。但并未显式建模各模态特有结构。这包括金融时序的时间动态、遥感的站点结构，以及航运的网络结构。

This raises the empirical question addressed in this dissertation. Can remote sensing and shipping improve one-week-ahead Brent price forecasts beyond **financial time series** and the no-change benchmark? And when the underlying data are held fixed, does keeping each modality’s structure before fusion outperform flat feature fusion? The detailed research gap is developed after the literature review in Chapter 2. The next section states the aim and research questions.

这正是本论要回答的实证问题：遥感与航运能否在**金融时序**与不变预测基准之上，改善提前一周的 Brent **价格**预测？在底层数据保持不变时，先保留各模态结构再融合，是否优于扁平特征融合？详细研究空白在第 2 章文献综述之后展开。下一节给出研究目标与研究问题。

### 1.2 Aim and research questions

### 1.2 研究目标与研究问题

The aim of this dissertation is not to propose a new neural-network building block. It is to build one reproducible comparison framework for the same weekly Brent **price** forecasting task under a shared protocol. The framework combines financial time series with satellite and shipping inputs. It uses a rolling design that never uses future information, and it tests whether one forecast improves on another. Flat feature fusion places all inputs in one table before modelling. Representation-level fusion encodes each modality first, then combines the representations. The contribution is integration and fair comparison, not a new model operator. Implementation details follow in Chapter 3.

本论的目标不是提出一种新的神经网络构件，而是在共享协议下，为同一周度 Brent **价格**预测任务构建一套可复现的对照框架。该框架将金融时序与卫星、航运输入结合；采用不使用未来信息的滚动设计，并检验一个预测是否优于另一个。扁平特征融合在建模前将全部输入放入同一特征表。表示级融合先对各模态分别编码，再融合所得表征。贡献是集成与公平比较，而非新的模型算子。实施细节见第 3 章。

Three research questions organise the study.

三项研究问题组织全文。

**RQ1.** Do remote-sensing and shipping indicators add incremental out-of-sample value over financial time series and the no-change benchmark?

**RQ1.** 遥感与航运指标是否在金融时序与不变预测基准之上带来样本外增量价值？

**RQ2.** Does modality-aware representation-level fusion outperform flat feature fusion when both use the same underlying data and the same evaluation protocol?

**RQ2.** 在相同底层数据与相同评估协议下，模态感知的表示级融合是否优于扁平特征融合？

**RQ3.** Can modality-level interpretability reveal which signals the model relies on across different market conditions?

**RQ3.** 模态级可解释性能否揭示模型在不同市场条件下依赖哪些信号？

The logic is sequential. First ask whether the data help. Then ask whether fusion architecture matters. Then, only where predictive value exists, ask what the model relies on.

逻辑是递进的。先问数据是否有用。再问融合架构是否重要。最后仅在已有预测价值之处，问模型依赖什么。
