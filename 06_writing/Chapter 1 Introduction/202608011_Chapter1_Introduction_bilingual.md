# A Modality-Aware Spatio-Temporal Fusion Framework for Brent Crude Oil Forecasting Using Financial Time Series, Satellite Imagery and Maritime Networks

# 面向金融时序、卫星遥感与航运网络的模态感知时空融合框架：Brent 原油价格预测

---

## Abstract *(~200 words)*

## 摘要 *（约 200 词）*

Brent crude is one of the principal benchmarks for internationally traded oil and a key reference price in the global energy market. Its short-term movements affect energy costs, inflation, trade balances and fiscal revenues, and therefore influence decisions made by firms and governments. Using weekly data from 2019 to 2025, this study examines whether satellite remote-sensing and shipping data provide incremental value beyond financial time series for one-week-ahead Brent price forecasting. It also compares flat feature fusion with modality-specific encoding followed by fusion. Flat models combine all selected inputs in a single feature table, whereas deep models encode each data source separately before fusing the resulting representations. For both model families, the study tests whether remote sensing and shipping, separately or together, add predictive value beyond conventional financial information.

Brent 原油是国际贸易原油的主要定价基准之一，也是全球能源市场的核心参考价格。其短期波动会影响能源成本、通胀、贸易收支和财政收入，并影响企业与政府的相关决策。本研究使用 2019—2025 年的周度数据，检验卫星遥感与航运数据能否在金融时序之外，为提前一周的 Brent 价格预测提供增量价值。本研究还比较扁平特征融合与先分模态编码再融合。扁平模型将所有选定输入合并到同一张特征表中，深度模型则分别编码各类数据源，再将其融合。对于两类模型，本研究均检验遥感与航运单独或一并加入时，能否在传统金融信息之外提供预测价值。

The models are evaluated against a no-change benchmark that sets next week’s price equal to this week’s price. The results show that no flat model outperforms this benchmark, although shipping data provide limited evidence of incremental predictive information. Deep models combining financial time series and shipping data achieve a small improvement over the benchmark. Remote-sensing data provide no clear additional benefit. Deep models consistently outperform Flat models, with the largest advantage observed when shipping data are included. This study further uses gated fusion and SHapley Additive exPlanations (SHAP) to show which data sources the deep models rely on most. Overall, predictive value depends more on how multimodal data are used, particularly how modalities are encoded and fused, than on simply adding more data. By highlighting the data sources and geographical locations that provide potential signals of supply-chain change, the framework could help governments monitor how geopolitical tensions and conflicts may affect oil prices and international trade, thereby informing energy-security and trade policy.

模型以“下周价格等于本周价格”的不变预测作为基准。研究结果表明，没有扁平模型优于该基准，但航运数据提供了有限的增量预测信息证据。结合金融时序与航运数据的深度模型相对基准取得了小幅改善。遥感数据没有带来明确的额外收益。Deep 模型始终优于 Flat 模型，在纳入航运数据时优势最大。本研究进一步使用门控融合与 SHapley Additive exPlanations（SHAP），展示深度模型主要依赖哪些数据源。总体而言，预测价值更多取决于多模态数据如何被使用——尤其是各模态如何被编码与融合——而不是单纯增加更多数据。通过突出可能提供供应链变化潜在信号的数据来源与地理位置，该框架可帮助政府监测地缘政治紧张与冲突如何影响油价与国际贸易，从而为能源安全与贸易政策提供参考。

---



## Chapter 1 — Introduction *(~600 words)*



## 第 1 章 — 绪论 *(约 600 词)*



### 1.1 Background



### 1.1 背景

Crude oil occupies a central place in the global economy and energy system. Oil-price movements affect inflation, trade balances, fiscal revenues in producer countries and the operating costs of energy-intensive industries. These effects spread through financial markets, economic activity and supply chains. They therefore shape the risk management, hedging, budgeting and planning decisions of governments, firms and investors. 

原油在全球经济和能源体系中占据核心位置。油价变动会影响通胀、贸易差额、产油国财政收入以及高耗能行业的运营成本。这些影响会通过金融市场、经济活动和供应链进一步传导。因此，油价变动会影响政府、企业和投资者的风险管理、对冲、预算与规划决策。

Crude grades differ in density, sulphur content, production location and transport access, so their prices are typically quoted against benchmarks. The main benchmarks are Brent, West Texas Intermediate (WTI) and Dubai/Oman (U.S. Energy Information Administration, 2014). Brent is a benchmark complex rooted in light, low-sulphur, waterborne crude oils from the North Sea. It is widely used as a reference for internationally traded crude. WTI is a US crude benchmark, with pricing centred on Cushing, Oklahoma, while Dubai/Oman is commonly used to price Middle Eastern crude exported to Asian markets (U.S. Energy Information Administration, 2014). Although Brent and WTI share global drivers, regional supply, inventories and transport constraints can cause their prices to diverge. This dissertation forecasts Brent because its international waterborne role aligns with the ports, shipping routes and maritime chokepoints represented in the spatial data.

不同原油在密度、硫含量、产地与运输可达性上存在差异，其价格通常相对基准报价。主要基准为 Brent、西得克萨斯中质原油（West Texas Intermediate, WTI）以及迪拜/阿曼（Dubai/Oman）（U.S. Energy Information Administration, 2014）。Brent 是以北海轻质低硫、可海运原油为基础的基准体系，广泛用作国际贸易原油的参考价格。WTI 是美国原油基准，定价中心在俄克拉荷马州库欣；迪拜/阿曼则常用于为输往亚洲市场的中东原油定价（U.S. Energy Information Administration, 2014）。尽管 Brent 与 WTI 共享全球驱动因素，区域供给、库存与运输约束仍可能导致两者价格分化。本研究预测 Brent，因为其国际海运角色与空间数据所代表的港口、航运路线和航运咽喉相一致。

Recent years have shown the costs of unexpected oil-price movements. The COVID-19 period brought an abrupt collapse in demand and an uneven recovery. The 2022 energy crisis then produced major supply and price shocks, followed by only partial normalisation amid continued geopolitical and macroeconomic uncertainty. More recent geopolitical disruptions have further shown how quickly oil prices and seaborne trade can respond when key maritime chokepoints are disrupted or bypassed. Governments monitor such shocks for inflation control, fiscal planning, energy security and trade policy. 
At the global level, initiatives such as the Joint Organisations Data Initiative (JODI) seek to improve the transparency of oil-market data used for energy-security monitoring. This study therefore asks whether publicly observed shipping and satellite signals, when added to financial time series, improve short-term Brent forecasts that could help governments and firms assess risk.

近年来的情况表明，油价意外波动可能造成较高代价。新冠疫情期间，原油需求骤降，随后出现不均衡复苏。2022 年能源危机随后带来显著的供给与价格冲击，此后市场在持续的地缘政治和宏观经济不确定性下仅实现部分正常化。近期的地缘政治扰动进一步表明，当关键航运咽喉受阻或被绕开时，油价与海运贸易会迅速作出反应。政府出于通胀管理、财政规划、能源安全与贸易政策需要监测此类冲击。在全球层面，诸如联合石油数据倡议（Joint Organisations Data Initiative, JODI）等机制旨在提高用于能源安全监测的石油市场数据透明度。因此，本研究考察：在金融时间序列的基础上加入公开可观测的航运和卫星信号，能否提升短期 Brent 价格预测，从而帮助政府和企业评估风险。

At the weekly horizon, the no-change forecast is difficult to outperform. This simple method predicts that next week’s price will be equal to this week’s price. It therefore provides a demanding benchmark for spatial data and forecasting methods. A model should not be considered useful merely because it outperforms a weaker or differently specified competitor. It must also be evaluated directly against the no-change benchmark.

在周度预测中，不变预测通常很难被超越。这种简单方法假设下周价格等于本周价格，因此为空间数据与预测方法提供了一个较高的基准。一个模型不能仅因优于较弱或设定不同的模型而被认为具有预测价值。它还必须与不变预测基准进行直接比较。

The three data sources considered in this dissertation provide complementary views of the oil system. Financial, macroeconomic and oil-market variables describe changes in market conditions over time. Remote sensing provides geographically explicit indicators of industrial activity at 11 oil-related sites through spectral indices, night-time lights and image representations. Automatic Identification System and PortWatch data provide geographically explicit observations of vessel activity across ports and 6 major maritime chokepoints. The shipping data also represent network relationships between locations and provide proxies for seaborne trade flows and congestion. Although Brent is observed as a single global benchmark, the underlying processes of oil production, refining and maritime transport are geographically distributed. The spatial-data inputs are therefore organised around spatially distributed monitoring sites and connected transport nodes, linking site- and network-level observations to a common forecasting target. In this dissertation, multimodal forecasting refers to combining temporal market data, spatial Earth-observation data and spatiotemporal shipping-network data within the same forecasting task.

本研究考察的三类数据为石油体系提供互补视角。金融、宏观经济和石油市场变量反映市场状况随时间的变化。遥感数据通过光谱指数、夜间灯光和影像表征，为 11 个与石油相关的地点提供具有明确地理空间属性的工业活动指标。船舶自动识别系统（Automatic Identification System）和 PortWatch 数据则提供港口及 6 个主要海上咽喉的船舶活动观测。这些航运数据还刻画不同地点之间的网络关系，并为海运贸易流量和拥堵程度提供代理指标。尽管 Brent 是以单一全球基准的形式被观测，但石油生产、炼化和海上运输等底层过程在地理上是分布式的。因此，空间数据输入围绕空间上分散的监测地点及相互连接的运输节点进行组织，从而将站点层面和网络层面的观测与一个共同的预测目标联系起来。在本论文中，多模态预测是指在同一预测任务中，将时间维度的市场数据、空间维度的地球观测数据，以及具有时空属性的航运网络数据进行整合。

These data create two practical challenges. First, the signals are noisy and arrive on different schedules. They may also respond to oil prices rather than predict them, while the available weekly sample is relatively small. Second, a common approach places all heterogeneous inputs in a single feature table before modelling. This flat feature fusion approach is convenient for conventional models such as Ridge regression and gradient-boosted trees. However, it does not explicitly preserve the temporal structure of financial time series, the site structure of remote sensing or the network structure of shipping data.

这些数据带来两项实际挑战。第一，各类信号含有噪声，观测频率和发布时间也不相同。它们还可能是对油价变化的反应，而不是油价的领先信号。同时，可用的周度样本相对较小。第二，一种常见方法是在建模前将所有异质输入放入同一张特征表中。这种扁平特征融合方法便于应用 Ridge 回归和梯度提升树等传统模型，但无法明确保留金融时序的时间结构、遥感数据的站点结构以及航运数据的网络结构。

This dissertation therefore addresses two empirical questions. First, do remote-sensing and shipping data improve one-week-ahead Brent price forecasts when they are added to financial time series? Can the resulting models outperform the no-change benchmark? Second, when the underlying data remain the same, does encoding each modality separately before fusion perform better than combining all inputs in a single feature table? The next section presents the study aim and formal research questions.

因此，本研究主要回答两个实证问题。第一，在金融时序中加入遥感与航运数据，能否改善提前一周的 Brent 价格预测？由此得到的模型能否优于不变预测基准？第二，在使用相同底层数据的情况下，先分别编码各类数据再进行融合，是否比将所有输入合并到同一张特征表中表现更好？下一节将介绍本研究的目标和正式研究问题。

### 1.2 Research aim


### 1.2 研究目标

The main aim of this dissertation is to develop a reproducible comparison framework for evaluating how different data sources and model designs perform in one-week-ahead Brent price forecasting. The framework combines financial time-series data, satellite remote-sensing data and shipping data. It uses a rolling-origin design that restricts each forecast to information available at the forecast origin and compares models by forecast accuracy. Flat feature fusion places all inputs in a single feature table before modelling. Representation-level fusion encodes each modality separately and then combines the resulting representations. This framework enables consistent comparisons of the incremental value of different data sources and the effects of different fusion designs. The empirical setting uses weekly Friday-ending Brent spot prices from 2019 to 2025, spanning periods of demand, supply and maritime disruption, with remote-sensing and shipping inputs covering 11 oil-related monitoring sites and 6 major maritime chokepoints.

本研究的主要目标是构建一套可复现的对照框架，用于评估不同数据来源和模型设计在提前一周的 Brent 价格预测中的表现。该框架结合金融时序数据、卫星遥感数据和航运数据。框架采用滚动起点预测设计，使每次预测仅使用该预测起点已可获得的信息，并通过预测精度比较不同模型。扁平特征融合在建模前将所有输入放入同一张特征表中。表示级融合则分别编码各类数据，再将所得表征进行融合。该框架能够一致地比较不同数据来源的增量价值，以及不同融合设计对预测表现的影响。本研究以 2019—2025 年周五截止的周度 Brent 现货价格为实证对象，样本涵盖需求、供给与海上运输扰动时期，遥感与航运输入覆盖 11 个石油相关监测站点和 6 个主要航运咽喉。

The study is organised around three research questions.

本研究围绕以下三个研究问题展开。

**RQ1.** Compared with models using only financial time-series data, do remote-sensing and shipping data improve one-week-ahead Brent price forecasts?

**RQ1.** 与仅使用金融时序数据的模型相比，加入遥感与航运数据能否改善提前一周的 Brent 价格预测？

**RQ2.** When using the same information set, does modality-aware representation-level fusion outperform flat feature fusion?

**RQ2.** 在使用相同信息集的情况下，模态感知的表示级融合是否优于扁平特征融合？

**RQ3.** Which data sources and spatial nodes do the models rely on, and how does this reliance vary across forecast periods?

**RQ3.** 模型依赖哪些数据来源和空间节点，这种依赖如何随预测时期而变化？

### 1.3 Dissertation structure

### 1.3 论文结构

The remainder of this dissertation is organised as follows. Chapter 2 reviews four related strands of literature covering oil-price forecasting benchmarks, machine-learning methods, shipping and satellite signals, and multimodal fusion. It then identifies the research gap addressed by the research questions. Chapter 3 presents the comparison framework, including the information sets, the Flat and Deep pathways, the rolling-origin evaluation against the no-change benchmark, and the ethical considerations relating to the spatial data. Chapter 4 reports the out-of-sample results for RQ1, RQ2 and RQ3. Chapter 5 interprets the findings in relation to the literature and discusses the implications and limitations. Chapter 6 concludes the dissertation.

本文其余部分安排如下。第 2 章综述四支相关文献，涵盖油价预测基准、机器学习方法、航运与卫星信号以及多模态融合，并指出本研究问题所针对的研究空白。第 3 章介绍对照框架，包括信息集、Flat 与 Deep 两条路径、相对不变预测基准的滚动起点评价，以及与空间数据有关的伦理考量。第 4 章报告 RQ1、RQ2 与 RQ3 的样本外结果。第 5 章结合文献解释研究发现，并讨论其含义与局限。第 6 章为全文结论。