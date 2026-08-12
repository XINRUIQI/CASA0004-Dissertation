# A Modality-Aware Spatio-Temporal Fusion Framework for Brent Crude Oil Forecasting Using Financial Time Series, Satellite Imagery and Maritime Networks

# 面向金融时序、卫星遥感与航运网络的模态感知时空融合框架：Brent 原油价格预测

---

## Abstract *(~200 words)*

## 摘要 *（约 200 词）*

Brent crude is one of the principal benchmarks for internationally traded oil and a key reference price in the global energy market. Its short-term movements affect energy costs, inflation, trade balances and fiscal revenues, and therefore influence decisions made by firms and governments. Using weekly data from 2019 to 2025, this study examines whether satellite remote-sensing and shipping data provide incremental value beyond financial time series for one-week-ahead Brent price forecasting. It also compares flat feature fusion with modality-specific encoding followed by fusion. Flat models combine all selected inputs in a single feature table, whereas deep models encode each data source separately before fusing the resulting representations. For both model families, the study compares a financial-time-series-only specification with alternatives that add remote sensing, shipping or both.

Brent 原油是国际贸易原油的主要定价基准之一，也是全球能源市场的核心参考价格。其短期波动会影响能源成本、通胀、贸易收支和财政收入，并影响企业与政府的相关决策。本研究使用 2019—2025 年的周度数据，检验卫星遥感与航运数据能否在金融时序之外，为提前一周的 Brent 价格预测提供增量价值。本研究还比较扁平特征融合与先分模态编码再融合。扁平模型将所有选定输入合并到同一张特征表中，深度模型则分别编码各类数据源，再将其融合。对于两类模型，本研究均比较仅使用金融时序的设定，以及分别加入遥感、航运或二者的设定。

The models are evaluated against a no-change benchmark that sets next week’s price equal to this week’s price. The results show that no flat model outperforms this benchmark, although shipping data provide limited evidence of incremental predictive information. Deep models combining financial time series and shipping data achieve a small improvement over the benchmark. Remote-sensing data provide no clear additional benefit. The advantage of deep models over flat models is most evident when shipping data are included. This study further uses modality gates to show which data sources the best-performing deep model relies on most. Overall, predictive value depends more on how multimodal data are used—especially how modalities are encoded and fused—than on simply adding more data.

模型以“下周价格等于本周价格”的不变预测作为基准。研究结果表明，没有扁平模型优于该基准，但航运数据表现出有限的增量预测信息。结合金融时序与航运数据的深度模型相对基准取得了小幅改善。遥感数据没有带来明确的额外收益。在加入航运数据时，深度模型相对扁平模型的优势最明显。本研究进一步使用了模态门控展示表现最佳的深度模型主要依赖哪些数据源。总体而言，预测价值更多取决于多模态数据如何被使用——尤其是各模态如何被编码与融合——而不是单纯增加更多数据。

---



## Chapter 1 — Introduction *(~600 words)*



## 第 1 章 — 绪论 *(约 600 词)*



### 1.1 Importance and background



### 1.1 重要性与背景

Crude oil occupies a central place in the global economy and energy system. Oil-price movements affect inflation, trade balances, fiscal revenues in producer countries and the operating costs of energy-intensive industries. These effects spread through financial markets, economic activity and supply chains. They therefore shape the risk management, hedging, budgeting and planning decisions of governments, firms and investors. 

原油在全球经济和能源体系中占据核心位置。油价变动会影响通胀、贸易差额、产油国财政收入以及高耗能行业的运营成本。这些影响会通过金融市场、经济活动和供应链进一步传导。因此，油价变动会影响政府、企业和投资者的风险管理、对冲、预算与规划决策。

Crude oil is not a homogeneous commodity: individual grades differ in density, sulphur content, production location and transport accessibility, and their prices are commonly expressed relative to a small number of benchmarks. Among the most widely used benchmarks are Brent, West Texas Intermediate (WTI) and Dubai/Oman (U.S. Energy Information Administration, 2014). Brent is a benchmark complex rooted in light, low-sulphur, waterborne crude oils from the North Sea. It is widely used as a reference for internationally traded crude. WTI is a US crude benchmark, with pricing centred on Cushing, Oklahoma, while Dubai/Oman is commonly used to price Middle Eastern crude exported to Asian markets (Wittner, 2020). Although Brent and WTI respond to many of the same global market conditions, differences in regional supply, inventory levels and transport constraints can cause their prices to diverge. This dissertation forecasts Brent because its role as an international waterborne benchmark aligns more closely with the ports, shipping routes and maritime chokepoints represented in the alternative data. WTI is nevertheless retained as a financial predictor and as a component of the Brent–WTI spread. No fixed volatility ranking between Brent and WTI is assumed. Whether shipping activity contains incremental predictive information for Brent is tested empirically in this dissertation.

原油并非同质商品：不同油品在密度、硫含量、产地与运输可达性上存在差异，其价格通常相对少数基准油报价。使用最广的基准包括 Brent、西得克萨斯中质原油（West Texas Intermediate, WTI）以及迪拜/阿曼（Dubai/Oman）（U.S. Energy Information Administration, 2014）。Brent 是以北海轻质低硫、可海运原油为基础的基准体系，广泛用作国际贸易原油的参考价格。WTI 是美国轻质低硫原油的定价基准，定价中心在俄克拉荷马州库欣；迪拜/阿曼则常用于为输往亚洲市场的中东原油定价（Wittner, 2020）。Brent 与 WTI 会对许多相同的全球市场条件作出反应，但在区域供给、库存与运输约束出现差异时，两者价差可能变化。本研究预测 Brent，因为其国际性与海运导向，与另类数据所代表的全球港口、航运路线和航运咽喉更为契合。尽管如此，WTI 仍作为金融预测变量保留，并构成 Brent–WTI 价差的一部分。本文不预设 Brent 与 WTI 之间固定的波动高低排序。航运活动是否对 Brent 具有增量预测信息，将由本研究进行实证检验。

Recent years have shown the costs of unexpected oil-price movements. The COVID-19 period brought an abrupt collapse in demand and an uneven recovery. The 2022 energy crisis then produced major supply and price shocks, followed by only partial normalisation amid continued geopolitical and macroeconomic uncertainty. More recent geopolitical disruptions have further shown how quickly oil prices and seaborne trade can respond when key maritime chokepoints are disrupted or bypassed. Governments monitor such shocks for inflation control, fiscal planning, energy security and trade policy. A better short-term oil-price model would help them gauge risk and timing. Although they cannot replace market judgement, such forecasts could support decision-making when physical flows and prices move together. 

近年来的市场变化进一步表明，油价意外波动可能造成较高代价。新冠疫情期间，原油需求骤降，随后出现不均衡复苏。2022 年能源危机又带来显著的供给与价格冲击。此后，市场在持续的地缘政治和宏观经济不确定性下仅实现部分正常化。近期冲突再次表明，当关键航运咽喉受阻或被绕开时，油价与海运贸易会迅速作出反应。政府出于通胀管理、财政规划、能源安全与贸易政策需要监测此类冲击。更可靠的短期油价模型有助于它们评估风险与时机。尽管它们无法替代市场判断，但当实物流动与价格同步变动时，此类预测仍可支持决策。

At the weekly horizon, the no-change forecast is difficult to outperform. This simple method predicts that next week’s price will be equal to this week’s price. It therefore provides a demanding benchmark for alternative data and methods. A model should not be considered useful merely because it outperforms a weaker or differently specified competitor. It must also be evaluated directly against the no-change benchmark.

在周度预测中，不变预测通常很难被超越。这种简单方法假设下周价格等于本周价格，因此为评价另类数据和预测方法提供了一个较高的基准。一个模型不能仅因优于较弱或设定不同的模型而被认为具有预测价值。它还必须与不变预测基准进行直接比较。

The three data sources considered in this dissertation provide complementary views of the oil system. Financial, macroeconomic and oil-market variables describe changes in market conditions over time. Remote sensing captures spatial activity at selected oil-related sites through spectral indicators, night-time lights and image representations. AIS and PortWatch data describe changes in vessel activity across ports and major chokepoints. They also capture network relationships between locations and provide proxies for seaborne trade flows and congestion. In this dissertation, multimodal forecasting refers to combining temporal market data, spatial Earth-observation data and spatiotemporal shipping-network data in the same forecasting task.

本研究考察的三类数据从不同角度描述石油市场。金融、宏观经济和石油市场变量反映市场状况随时间的变化。遥感数据通过光谱指标、夜间灯光和影像表征，描述特定石油相关地点的空间活动。AIS 与 PortWatch 数据描述港口和主要航运咽喉的船舶活动变化。它们还保留不同地点之间的网络关系，并可作为海运贸易流量和拥堵情况的代理变量。本文所称的多模态预测，是指在同一预测任务中结合时间性的市场数据、空间性的对地观测数据，以及具有时空网络结构的航运数据。

These data create two practical challenges. First, the signals are noisy and arrive on different schedules. They may also respond to oil prices rather than predict them, while the available weekly sample is relatively small. Second, a common approach places all heterogeneous inputs in a single feature table before modelling. This flat feature-fusion approach is convenient for conventional models such as Ridge regression and gradient-boosted trees. However, it does not explicitly preserve the temporal structure of financial time series, the site structure of remote sensing or the network structure of shipping data.

这些数据带来两项实际挑战。第一，各类信号含有噪声，观测频率和发布时间也不相同。它们还可能是对油价变化的反应，而不是油价的领先信号。同时，可用的周度样本相对较小。第二，一种常见方法是在建模前将所有异质输入放入同一张特征表中。这种扁平特征融合方法便于应用 Ridge 回归和梯度提升树等传统模型，但无法明确保留金融时序的时间结构、遥感数据的站点结构以及航运数据的网络结构。

This dissertation therefore addresses two empirical questions. First, do remote-sensing and shipping data improve one-week-ahead Brent price forecasts when they are added to financial time series? Can the resulting models outperform the no-change benchmark? Second, when the underlying data remain the same, does encoding each modality separately before fusion perform better than combining all inputs in a single feature table? The next section presents the study aim and formal research questions.

因此，本研究主要回答两个实证问题。第一，在金融时序中加入遥感与航运数据，能否改善提前一周的 Brent 价格预测？由此得到的模型能否优于不变预测基准？第二，在使用相同底层数据的情况下，先分别编码各类数据再进行融合，是否比将所有输入合并到同一张特征表中表现更好？下一节将介绍本研究的目标和正式研究问题。

### 1.2 Aim and research questions



### 1.2 研究目标与研究问题

The main aim of this dissertation is to develop a reproducible comparison framework for evaluating how different data sources and model designs perform in one-week-ahead Brent price forecasting. The framework combines financial time-series data, satellite remote-sensing data and shipping data. It uses a rolling-origin forecasting design that prevents the use of future information and applies formal statistical tests to compare predictive performance. Flat feature fusion places all inputs in a single feature table before modelling. Representation-level fusion encodes each modality separately and then combines the resulting representations. This framework enables consistent comparisons of the incremental value of different data sources and the effects of different fusion designs. The empirical setting is weekly, Friday-ending Brent spot prices from 2019 to 2025, with remote-sensing and shipping inputs covering eleven oil-related monitoring sites and six major maritime chokepoints.

本研究的主要目标是构建一套可复现的对照框架，用于评估不同数据来源和模型设计在提前一周的 Brent 价格预测中的表现。该框架结合金融时序数据、卫星遥感数据和航运数据。框架采用滚动起点预测设计，以避免使用未来信息，并通过正式统计检验比较不同模型的预测表现。扁平特征融合在建模前将所有输入放入同一张特征表中。表示级融合则分别编码各类数据，再将所得表征进行融合。该框架能够一致地比较不同数据来源的增量价值，以及不同融合设计对预测表现的影响。本研究的实证设定为 2019—2025 年周五截止的周度 Brent 现货价格，遥感与航运输入覆盖十一个石油相关监测站点与六个主要航运咽喉。

The study is organised around three research questions.

本研究围绕以下三个研究问题展开。

**RQ1.** Compared with models using only financial time-series data, do remote-sensing and shipping data improve one-week-ahead Brent price forecasts?

**RQ1.** 与仅使用金融时序数据的模型相比，加入遥感与航运数据能否改善提前一周的 Brent 价格预测？

**RQ2.** When using the same underlying data, does modality-aware representation-level fusion outperform flat feature fusion?

**RQ2.** 在使用相同底层数据的情况下，模态感知的表示级融合是否优于扁平特征融合？

**RQ3.** Which data sources do the models rely on under different market conditions?

**RQ3.** 模型在不同市场条件下依赖哪些数据来源？