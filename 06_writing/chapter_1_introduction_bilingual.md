# Chapter 1 — Introduction
# 第 1 章 — 绪论

## 1.1 Background and motivation
## 1.1 研究背景与动机

Crude oil price forecasting remains a central problem in energy economics, financial risk management and supply-chain analysis. Brent crude is the dominant global pricing benchmark, yet weekly movements are difficult to predict out of sample. A large forecasting literature shows that the no-change, or random-walk, forecast is extremely hard to beat, especially at short horizons (Alquist, Kilian and Vigfusson, 2013). Any claim that new data or more complex models improve prediction must therefore clear a high bar: improvement over an economically informed financial baseline, and ideally over the random walk itself.

原油价格预测仍然是能源经济学、金融风险管理和供应链分析中的核心问题。Brent 原油是全球主要定价基准，但周度价格变动在样本外很难准确预测。大量预测文献表明，无变化预测（随机游走）极难被超越，尤其是在短预测期限上（Alquist, Kilian and Vigfusson, 2013）。因此，任何关于新数据或更复杂模型改善预测的主张，都必须跨越很高的门槛：相对于具有经济含义的金融基线有所提升，并最好相对于随机游走本身也有提升。

Empirically, this study focuses on weekly Friday-ending Brent spot prices over **2019–2025** (approximately 365 merged weeks; a common out-of-sample test span of about 257 weeks after warm-up). The window covers COVID-era disruption, the 2022 energy shock and the subsequent normalisation phase, and therefore provides a stringent setting in which to ask whether alternative physical-market signals add forecast value when the random walk remains difficult to beat.

实证上，本文聚焦 **2019–2025** 年周五截止周的 Brent 现货价（合并约 365 周；warm-up 后共同样本外测试约 257 周）。该区间覆盖新冠冲击、2022 年能源冲击及随后的常态化阶段，因而构成一个严格场景：在随机游走仍难被超越的市场中，检验另类实物市场信号能否带来预测价值。

In parallel, alternative data have become more accessible. Automatic Identification System (AIS) and PortWatch-style shipping indicators provide high-frequency proxies for physical trade and chokepoint congestion. Satellite remote sensing — including night-time lights and site-level optical indices or image embeddings — offers a physical view of infrastructure activity. These sources are plausible proxies for supply, demand and disruption, but they are noisy, often asynchronous, and may respond to prices as well as lead them.

与此同时，替代数据变得更加可得。自动识别系统（AIS）和 PortWatch 类航运指标为实物贸易与咽喉拥堵提供了高频代理。卫星遥感——包括夜间灯光，以及站点级光学指数或影像嵌入——则提供了基础设施活动的物理视角。这些来源是供给、需求与冲击的合理代理，但噪声大、常常异步，并且可能对价格作出反应，而不一定领先价格。

Most oil-related applications still convert these heterogeneous signals into engineered numeric columns and concatenate them with financial predictors. That flat, early-fusion approach is practical, but it discards modality-specific structure: temporal dynamics in finance, site structure in remote sensing, and network structure in shipping. Whether the *way* modalities are represented and fused matters — beyond simply adding more columns — is therefore an open empirical question for weekly Brent forecasting, and it motivates the research problems set out below.

多数与原油相关的应用仍将这些异构信号转化为人工数值特征，并与金融预测变量拼接。这种扁平、早融合做法便于实现，但会丢失模态特有结构：金融的时序动态、遥感的站点结构，以及航运的网络结构。对周度 Brent 预测而言，除“加入更多列”之外，模态被如何表示与融合是否真正重要，仍是一个开放的实证问题，并由此引出下文的研究问题。

## 1.2 Research problem
## 1.2 研究问题

Against this background — a strong random-walk benchmark, noisy alternative data, and a prevailing flat early-fusion practice — this dissertation addresses three linked empirical problems. Formal research questions are stated in Section 1.4; here the focus is on the nature of the uncertainty.

在这一背景下——强随机游走基准、噪声较大的另类数据，以及占主导的扁平早融合惯例——本文同时面对三个相互关联的实证问题。正式研究问题见第 1.4 节；此处先说明不确定性的性质。

First, it remains unclear whether remote-sensing and shipping indicators add incremental out-of-sample value over a financial baseline once evaluation is leakage-safe and statistically rigorous, and whether any nested gains are large enough to beat the random walk in absolute terms. Second, even when the same underlying information is used, it is unclear whether modality-aware representation-level fusion — modality-specific encoders with gated or cross-attention fusion — outperforms flat feature concatenation; that is, whether architecture differences matter beyond “adding more columns”. Third, even when a model improves, it is often unclear which modality or spatial node the model relies on, and whether those attributions shift across market conditions in an economically interpretable way.

第一，在无泄漏且统计严谨的评估下，遥感与航运指标是否相对金融基线带来样本外增量价值，以及即便存在嵌套增量，是否足以在绝对意义上击败随机游走，仍不清楚。第二，即便使用相同底层信息，模态感知的表示级融合——模态专属编码器配合门控或交叉注意力融合——是否优于扁平特征拼接，即架构差异是否独立于“多加几列”，仍不清楚。第三，即便模型有所改进，模型依赖哪些模态或空间节点、这些归因是否随市场条件变化且具有经济可解释性，往往也不清楚。

Answering these questions requires a shared protocol — common information sets, timeline and metrics — together with formal forecast-comparison tests that separate nested incremental value over finance (Clark–West versus M1) from absolute skill against the random walk (Diebold–Mariano versus M0). Without that discipline, it is easy to overstate the usefulness of alternative data or of deeper architectures. The next section summarises the corresponding gaps in the existing literature.

回答上述问题需要统一协议——同一信息集、时间线与指标——以及正式的预测比较检验，以区分相对金融基线的嵌套增量价值（Clark–West 相对 M1）与相对随机游走的绝对预测能力（Diebold–Mariano 相对 M0）。缺少这一纪律时，很容易夸大另类数据或更深架构的作用。下一节将上述不确定性收束为文献中的研究空白。

## 1.3 Research gap
## 1.3 研究空白

Existing studies have made progress on financial oil-price forecasting and on alternative-data proxies, but three related gaps remain. Multi-source oil studies rarely compare flat concatenation with representation-level modality-aware fusion under one shared, leakage-safe protocol; comparisons often pit each family’s best model against the other, or report only one fusion style. Few studies jointly report both nested incremental value over a financial baseline and absolute skill against the random walk — a dual standard that is essential, because reporting only nested gains can overstate alternative data, while reporting only random-walk comparisons can conceal economically meaningful but weak signals. Finally, interpretability is often detached from predictive value: extensive attribution for models that fail the relevant benchmark does not support a claim that the signals are useful.

既有研究在金融油价预测与替代数据代理方面已有进展，但仍存在三处相互关联的空白。多源油价研究很少在统一、无泄漏协议下比较扁平拼接与表示级模态感知融合；常见做法是各族最优模型互比，或只报告一种融合方式。很少有研究同时报告相对金融基线的嵌套增量价值与相对随机游走的绝对预测能力——这一双重标准不可或缺：只报嵌套增量会夸大另类数据，只报相对随机游走又可能掩盖有经济含义但较弱的信号。最后，可解释性常与预测价值脱节：对未通过相关基准检验的模型做大量归因，难以支撑“信号真的有用”的叙事。

This dissertation does not aim to invent a new neural operator. Instead, it fills the gaps above through integration and fair comparison: a common information ladder, a locked evaluation protocol, paired Flat–Deep contrasts, and disciplined interpretability. These design choices map directly onto the research questions below.

本文不追求发明新的神经算子，而以集成与公平比较填补上述空白：统一信息集阶梯、锁定评估协议、配对的 Flat–Deep 对照，以及有纪律的可解释性。这些设计选择直接对应下文的研究问题。

## 1.4 Research questions
## 1.4 研究问题（RQ）

The three questions proceed from whether alternative data help, to whether fusion architecture matters on the same data, to what the model relies on when predictive value exists.

三个问题按递进展开：先问另类数据是否有用，再问相同数据下融合架构是否重要，最后问在存在预测价值时模型依赖什么。

- **RQ1:** Do remote-sensing and shipping indicators add incremental out-of-sample value over a financial baseline (M1) and the random-walk benchmark (M0)?
- **RQ1：** 遥感与航运指标是否在金融基线（M1）和随机游走基准（M0）之上带来样本外增量价值？

RQ1 is answered through an M0–M4 information-set ablation ladder within both the flat and deep architectures, using Clark–West tests for nested increments over M1 and Diebold–Mariano tests for absolute skill against M0.

RQ1 通过扁平与深度两类架构内的 M0–M4 信息集消融阶梯回答，以 Clark–West 检验相对 M1 的嵌套增量，以 Diebold–Mariano 检验相对 M0 的绝对 skill。

- **RQ2:** Does modality-aware representation-level fusion outperform flat feature fusion when both use the same underlying data and the same evaluation protocol?
- **RQ2：** 在使用相同底层数据与相同评估协议时，模态感知表示级融合是否优于扁平特征融合？

RQ2 is answered by paired comparisons of Flat models (Ridge / XGBoost) and Deep models (modality encoders with gated, concat or cross-attention fusion) at matched information sets, rather than by comparing only the champion model in each family.

RQ2 通过在匹配信息集上配对比较扁平模型（Ridge / XGBoost）与深度模型（模态编码器 + 门控 / 拼接 / 交叉注意力融合）来回答，而不是只比较各族的冠军模型。

- **RQ3:** Can modality-level interpretability reveal which signals the model relies on across different market conditions?
- **RQ3：** 模态级可解释性分析能否揭示模型在不同市场条件下依赖哪些信号？

RQ3 draws on Flat-side SHAP and Deep-side gating weights and site/node attention, with interpretability prioritised for models that already show predictive value relative to the relevant benchmark (Section 3.10).

RQ3 依托扁平侧 SHAP 与深度侧门控权重及站点/节点注意力；解释优先落在相对相关基准已显示预测价值的模型上（第 3.10 节）。

The aim is not to propose a new neural operator. It is to integrate frozen Earth-observation foundation-model embeddings, modality-specific encoders, fusion modules and a leakage-safe rolling-origin protocol with formal forecast-comparison tests into one reproducible comparison framework.

本文的目标不是提出新的神经算子，而是将冻结的地球观测基础模型嵌入、模态专属编码器、融合模块，以及无泄漏滚动起点协议与正式预测比较检验，集成到一套可复现的比较框架中。

## 1.5 Contributions
## 1.5 研究贡献

The dissertation makes three contributions that map onto RQ1–RQ3. Contribution type = **application + integration + fair comparison**; no new fusion operator is proposed. The evaluation protocol is not listed separately.

本文有三点贡献，对应 RQ1–RQ3。贡献类型 = **application + integration + 公平比较**，不提出新算子。评估协议不单独成条。

1. **Primary contribution (empirical / applied) — nested multimodal comparison.** First systematic nested comparison of finance, satellite remote sensing and shipping within one weekly Brent design. The M0→M1→M2→M3→M4 ladder isolates the out-of-sample increment from adding remote sensing, adding shipping, and using all modalities—relative to the financial baseline (M1) and the random walk (M0). The contribution is not “using three data sources” as such, but showing **which sources help, which do not, and relative to which benchmark** (RQ1).
2. **主贡献（实证 / 应用）— 多模态嵌套对照。** 第一次把金融 + 卫星遥感 + 航运放进同一套周频 Brent 设计做嵌套对照；用 M0→M1→M2→M3→M4 分离「加遥感 / 加航运 / 全模态」相对金融基线（M1）与随机游走（M0）的样本外增量。贡献不在于「用了三个数据源」，而在于**系统比清楚谁有用、谁没用、相对什么基准有用**（RQ1）。

3. **Method-integration contribution — paired Flat vs Deep.** Under the same data and the same protocol, compares flat early fusion (Ridge / XGBoost) with modality-aware representation-level fusion (encoders + gated / cross-attention), paired by information set (M1–M4)—not champion-versus-champion (RQ2).
4. **方法集成贡献 — 配对 Flat vs Deep。** 同一数据、同一协议下，对比扁平早融合（Ridge / XGBoost）与模态感知表示级融合（编码器 + 门控 / 交叉注意力）；按信息集配对，不是冠军互打（RQ2）。

5. **Interpretability contribution (supporting).** On models that already show predictive value, uses SHAP, gating weights and site/node attention to show which modalities and spatial nodes the model relies on under different market conditions (RQ3).
6. **解释性贡献（辅助）。** 在已有预测价值的模型上，用 SHAP / 门控 / 站点–节点注意力说明不同市场条件下依赖哪些模态和空间节点（RQ3）。

## 1.6 Dissertation structure
## 1.6 论文结构

Chapter 2 reviews the literatures on oil-price forecasting and the random-walk benchmark, shipping and remote-sensing proxies, multimodal fusion, and forecast evaluation, and closes on the gap and research questions. Chapter 3 presents the prediction target and timeline, the three modalities, the M0–M4 information sets, flat and deep architectures, the leakage-safe validation protocol, metrics and tests, and the interpretability rule. Chapter 4 reports results in an order aligned with the research questions: experimental overview; flat-model evidence for RQ1; deep-model evidence within RQ1; paired Flat–Deep comparison for RQ2; robustness; and interpretability for RQ3. Chapter 5 answers RQ1–RQ3 and discusses implications, limitations and future work, keeping nested incremental value and absolute skill separate throughout. Chapter 6 concludes in a short summary without introducing new numerical dumps.

第 2 章综述油价预测与随机游走基准、航运与遥感代理、多模态融合及预测评估文献，并收束到研究空白与研究问题。第 3 章介绍预测目标与时间线、三模态数据、M0–M4 信息集、扁平与深度架构、无泄漏验证协议、指标与检验，以及可解释性规则。第 4 章按研究问题组织结果：实验概览；面向 RQ1 的扁平证据；深度架构内的 RQ1 证据；面向 RQ2 的配对 Flat–Deep 比较；稳健性；以及面向 RQ3 的可解释性。第 5 章逐条回答 RQ1–RQ3，讨论含义、局限与未来工作，并始终将嵌套增量价值与绝对 skill 分开。第 6 章以简短总结收束，不堆砌新的数字结果。

Throughout, nested incremental value and absolute skill are kept conceptually separate. That separation is essential for an honest reading of alternative-data and deep-fusion results in a market where the random walk remains difficult to beat.

全文始终将嵌套增量价值与绝对 skill 在概念上分开。在随机游走仍然难被超越的市场中，这一区分对于诚实解读替代数据与深度融合结果至关重要。
