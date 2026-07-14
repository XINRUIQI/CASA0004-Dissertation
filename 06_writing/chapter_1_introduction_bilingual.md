# Chapter 1 — Introduction
# 第 1 章 — 绪论

## 1.1 Background and motivation
## 1.1 研究背景与动机

Crude oil price forecasting remains a central problem in energy economics, financial risk management and supply-chain analysis. Brent crude is the dominant global pricing benchmark, yet weekly movements are difficult to predict out of sample. A large forecasting literature shows that the no-change, or random-walk, forecast is extremely hard to beat, especially at short horizons (Alquist, Kilian and Vigfusson, 2013). Any claim that new data or more complex models improve prediction must therefore clear a high bar: improvement over an economically informed financial baseline, and ideally over the random walk itself.

原油价格预测仍然是能源经济学、金融风险管理和供应链分析中的核心问题。Brent 原油是全球主要定价基准，但周度价格变动在样本外很难准确预测。大量预测文献表明，无变化预测（随机游走）极难被超越，尤其是在短预测期限上（Alquist, Kilian and Vigfusson, 2013）。因此，任何关于新数据或更复杂模型改善预测的主张，都必须跨越很高的门槛：相对于具有经济含义的金融基线有所提升，并最好相对于随机游走本身也有提升。

In parallel, alternative data have become more accessible. Automatic Identification System (AIS) and PortWatch-style shipping indicators provide high-frequency proxies for physical trade and chokepoint congestion. Satellite remote sensing — including night-time lights and site-level optical indices or image embeddings — offers a physical view of infrastructure activity. These sources are plausible proxies for supply, demand and disruption, but they are noisy, often asynchronous, and may respond to prices as well as lead them.

与此同时，替代数据变得更加可得。自动识别系统（AIS）和 PortWatch 类航运指标为实物贸易与咽喉拥堵提供了高频代理。卫星遥感——包括夜间灯光，以及站点级光学指数或影像嵌入——则提供了基础设施活动的物理视角。这些来源是供给、需求与冲击的合理代理，但噪声大、常常异步，并且可能对价格作出反应，而不一定领先价格。

Most oil-related applications still convert these heterogeneous signals into engineered numeric columns and concatenate them with financial predictors. That flat, early-fusion approach is practical, but it discards modality-specific structure: temporal dynamics in finance, site structure in remote sensing, and network structure in shipping. Whether the *way* modalities are represented and fused matters — beyond simply adding more columns — is an open empirical question for weekly Brent forecasting.

多数与原油相关的应用仍将这些异构信号转化为人工数值特征，并与金融预测变量拼接。这种扁平、早融合做法便于实现，但会丢失模态特有结构：金融的时序动态、遥感的站点结构，以及航运的网络结构。对周度 Brent 预测而言，除“加入更多列”之外，模态被如何表示与融合是否真正重要，仍是一个开放的实证问题。

## 1.2 Research problem
## 1.2 研究问题

This dissertation addresses three linked problems. First, it is unclear whether remote-sensing and shipping indicators add incremental out-of-sample value over a financial baseline once evaluation is leakage-safe and statistically rigorous. Second, it is unclear whether modality-aware representation-level fusion outperforms flat feature fusion when both use the same underlying information. Third, even when a model improves, it is often unclear which modality the model relies on, and whether those attributions are economically interpretable.

本文关注三个相互关联的问题。第一，在无泄漏且统计严谨的评估下，遥感与航运指标是否相对金融基线带来样本外增量价值，尚不清楚。第二，在使用相同底层信息时，模态感知的表示级融合是否优于扁平特征融合，尚不清楚。第三，即便模型有所改进，模型依赖哪些模态、这些归因是否具有经济可解释性，往往也不清楚。

## 1.3 Research gap
## 1.3 研究空白

Existing studies have made progress on financial oil-price forecasting and on alternative-data proxies, but two gaps remain. Multi-source oil studies rarely compare flat concatenation with representation-level modality-aware fusion under one shared protocol. Few studies jointly report both nested incremental value over a financial baseline and absolute skill against the random walk. Without that joint standard, it is easy to overstate the usefulness of alternative data or of deeper architectures.

既有研究在金融油价预测与替代数据代理方面已有进展，但仍存在两处空白。多源油价研究很少在统一协议下比较扁平拼接与表示级模态感知融合。很少有研究同时报告相对金融基线的嵌套增量价值，以及相对随机游走的绝对预测能力。缺少这一双重标准时，很容易夸大替代数据或更深架构的作用。

## 1.4 Research questions
## 1.4 研究问题（RQ）

This dissertation asks:

本文提出以下研究问题：

- **RQ1:** Do remote-sensing and shipping indicators add incremental out-of-sample value over a financial baseline and the random-walk benchmark?
- **RQ1：** 遥感与航运指标是否在金融基线和随机游走基准之上带来样本外增量价值？

- **RQ2:** Does modality-aware representation-level fusion outperform flat feature fusion when both use the same underlying data?
- **RQ2：** 在使用相同底层数据时，模态感知表示级融合是否优于扁平特征融合？

- **RQ3:** Can modality-level interpretability reveal which signals the model relies on across different market conditions?
- **RQ3：** 模态级可解释性分析能否揭示模型在不同市场条件下依赖哪些信号？

The aim is not to propose a new neural operator. It is to integrate existing methods into a coherent comparison design and to test these questions under a leakage-safe rolling-origin protocol with formal forecast-comparison tests.

本文的目标不是提出新的神经算子，而是将既有方法集成为一个连贯的比较设计，并在无泄漏滚动起点协议与正式预测比较检验下回答上述问题。

## 1.5 Contributions
## 1.5 研究贡献

The dissertation makes four contributions.

本文有四点贡献。

1. **Unified information ladder.** It constructs a common M0–M4 design — random walk, finance, finance+remote sensing, finance+shipping, and all modalities — applied consistently to both flat and deep architectures.
2. **统一信息集阶梯。** 构建共同的 M0–M4 设计——随机游走、金融、金融+遥感、金融+航运、全模态——并一致应用于扁平与深度两类架构。

3. **Fair evaluation.** It evaluates all models under one rolling-origin expanding-window protocol, with metrics on reconstructed price and Diebold–Mariano / Clark–West tests that distinguish absolute skill from nested incremental information.
4. **公平评估。** 在统一的滚动起点扩展窗口协议下评估所有模型，以还原价格上的指标，并用 Diebold–Mariano / Clark–West 检验区分绝对预测能力与嵌套增量信息。

5. **Paired Flat–Deep comparison.** It compares deep and flat models by matched information set (M1–M4), rather than only comparing the single best model in each family.
6. **配对的 Flat–Deep 比较。** 按匹配的信息集（M1–M4）比较深度与扁平模型，而不是只比较各类中的单一最优模型。

7. **Disciplined interpretability.** Interpretability analysis focuses on models with predictive value relative to the relevant benchmark, with supplementary SHAP for models that show significant nested gains over M1 even when they do not beat M0.
8. **有纪律的可解释性。** 可解释性分析聚焦相对相关基准具有预测价值的模型；对虽未超过 M0、但相对 M1 有显著嵌套增量的模型，提供补充性 SHAP 分析。

## 1.6 Dissertation structure
## 1.6 论文结构

Chapter 2 reviews the literatures on oil-price forecasting, shipping and remote-sensing proxies, multimodal fusion, and forecast evaluation. Chapter 3 presents the data, information sets, flat and deep architectures, validation protocol and evaluation design. Chapter 4 reports results organised around experimental overview, flat models, deep models, paired Flat–Deep comparison, robustness and interpretability. Chapter 5 answers RQ1–RQ3 and discusses implications, limitations and future work. Chapter 6 concludes.

第 2 章综述油价预测、航运与遥感代理、多模态融合及预测评估文献。第 3 章介绍数据、信息集、扁平与深度架构、验证协议与评估设计。第 4 章按实验概览、扁平模型、深度模型、配对 Flat–Deep 比较、稳健性与可解释性组织结果。第 5 章回答 RQ1–RQ3，并讨论含义、局限与未来工作。第 6 章总结全文。

Throughout, nested incremental value and absolute skill are kept conceptually separate. That separation is essential for an honest reading of alternative-data and deep-fusion results in a market where the random walk remains difficult to beat.

全文始终将嵌套增量价值与绝对 skill 在概念上分开。在随机游走仍然难被超越的市场中，这一区分对于诚实解读替代数据与深度融合结果至关重要。
