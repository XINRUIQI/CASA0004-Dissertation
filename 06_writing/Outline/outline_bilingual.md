# A Modality-Aware Spatio-Temporal Fusion Framework for Brent Crude Oil Forecasting Using Financial Time Series, Satellite Imagery and Maritime Networks

# 面向金融时序、卫星遥感与航运网络的模态感知时空融合框架：Brent 原油价格预测

## Chapter 1 — Introduction *(~900–1,200)*

## 第 1 章 — 绪论 *(约 900–1,200 词)*

*background → gap → aim/RQ → contributions → structure*

*背景 → 空白 → 目标/研究问题 → 贡献 → 结构*

### 1.1 Background and forecasting challenge

### 1.1 背景与预测挑战

- **Brent's importance:** Oil-price forecasting in energy economics, risk management and supply chains; Brent as the global pricing benchmark. Sample window: Friday-ending weekly Brent spot over **2019–2025** (COVID, 2022 energy shock, normalisation).
- **Strong naïve benchmark:** Weekly Brent is hard to beat out of sample; the no-change / random-walk forecast is a strong bar that any new data or model claim must clear.
- **Alternative-data potential:** AIS / PortWatch shipping and satellite RS (NTL, site optical indices / image embeddings) as physical-market proxies for trade, chokepoint congestion and infrastructure activity.
- **Practical challenges:** Signals are noisy, asynchronous, and possibly price-*responding* rather than price-*leading*; small weekly sample. Dominant practice still flattens sources into a wide table and early-fuses with finance (flat concat), discarding finance temporal structure, RS site structure and shipping network structure.

- **Brent 的重要性：** 油价预测在能源经济学、风险管理与供应链中的地位；Brent 作为全球定价基准。样本窗口：周五截止的周度 Brent 现货，**2019–2025**（新冠、2022 能源冲击、后续常态化）。
- **强朴素基准：** 周度 Brent 样本外难打；无变化 / 随机游走预测是任何新数据或新模型主张都必须越过的高门槛。
- **另类数据潜力：** AIS / PortWatch 航运与卫星遥感（夜光、站点光学指数 / 影像嵌入）作为实物市场代理，刻画贸易、咽喉拥堵与基础设施活动。
- **实践难点：** 信号嘈杂、异步，且可能是价格的*响应*而非*领先*；周度样本量小。主流做法仍是将多源压成宽表并与金融早融合（flat concat），从而丢掉金融时间结构、遥感站点结构与航运网络结构。

### 1.2 Research gap

### 1.2 研究空白

The literature contains three related gaps:

文献中存在三个相互关联的空白：

1. **Incremental value of alternative data is unclear.** Few studies jointly report **nested increments** over a financial baseline *and* **absolute skill** against the random walk under leakage-safe evaluation — nested-only overstates alternative data; random-walk-only can hide economically meaningful but weak signals.
2. **Fusion architectures lack fair comparison.** Multi-source oil studies rarely compare flat concat vs representation-level modality-aware fusion under **one** shared protocol (common pattern: best-vs-best across families, or only one fusion style).
3. **Attribution lacks benchmark conditioning.** Interpretability is often detached from predictive value: heavy attribution for models that fail the relevant benchmark cannot support a "signals are useful" narrative.

1. **另类数据的增量价值不清楚。** 很少有研究在无泄漏评估下同时报告相对金融基线的**嵌套增量**与相对随机游走的**绝对 skill**——只报嵌套会夸大另类数据；只报随机游走可能掩盖经济上有意义但偏弱的信号。
2. **融合架构缺少公平对照。** 多源油价研究很少在**同一**共享协议下比较 flat concat 与表示级模态感知融合（常见模式是跨族冠军对决，或只做一种融合）。
3. **归因缺少基准条件。** 可解释性常与预测价值脱节：对未通过相关基准的模型做大量归因，不足以支撑"信号有用"的叙事。

### 1.3 Aim and research questions

### 1.3 研究目标与研究问题

*do the data help → does fusion architecture matter on the same data → what does the model rely on when value exists.*

*数据是否有用 → 同一数据下融合架构是否重要 → 有价值时模型依赖什么。*

**Aim:** Integrate frozen EO embeddings, modality-specific encoders, fusion modules and a leakage-safe rolling-origin protocol with formal forecast-comparison tests into one reproducible comparison framework — not a new neural operator, but integration + fair comparison.

**目标：** 将冻结 EO 嵌入、模态专属编码器、融合模块与无泄漏滚动起点协议及正式预测比较检验，整合进一套可复现的对照框架——不是提出新的神经算子，而是集成 + 公平比较。

- **RQ1:** Do remote-sensing and shipping indicators add incremental out-of-sample value over a financial baseline and the random-walk benchmark?
- **RQ2:** Does modality-aware representation-level fusion outperform flat feature fusion when both use the same underlying data and the same evaluation protocol?
- **RQ3:** Can modality-level interpretability reveal which signals the model relies on across different market conditions?

- **RQ1：** 遥感与航运指标是否在金融基线与随机游走基准之上带来样本外增量价值？
- **RQ2：** 在相同底层数据与相同评估协议下，模态感知的表示级融合是否优于扁平特征融合？
- **RQ3：** 模态级可解释性能否揭示模型在不同市场条件下依赖哪些信号？

### 1.4 Contributions ？？？

### 1.4 贡献

1. **Primary (empirical / applied) — nested multimodal comparison.** The first systematic nested comparison of finance, satellite remote sensing and shipping within one weekly Brent design. A branching modality ablation isolates the out-of-sample increment from adding remote sensing (M2), adding shipping (M3), and combining both (M4), relative to the financial baseline and the random walk, identifying which sources help under which benchmark (RQ1).
2. **Method-integration — paired Flat vs Deep.** Under the same data and the same protocol, compares flat early fusion (Ridge / XGBoost) with modality-aware representation-level fusion (encoders + gated / cross-attention) at matched information sets (RQ2).
3. **Interpretability (supporting).** For models that improve on M0, uses gating and site–node attention to show which modalities and spatial nodes the model relies on under different market conditions (RQ3).

1. **主贡献（实证 / 应用）— 嵌套多模态对照。** 第一次在同一周度 Brent 设计内系统嵌套比较金融、卫星遥感与航运。分支式模态消融分离加遥感（M2）、加航运（M3）与二者并用（M4）相对金融基线与随机游走的样本外增量，从而识别哪些来源在何种基准下有用（RQ1）。
2. **方法集成 — 配对 Flat vs Deep。** 在相同数据与相同协议下，按匹配信息集比较扁平早融合（Ridge / XGBoost）与模态感知表示级融合（编码器 + 门控 / 交叉注意力）（RQ2）。
3. **可解释性（支撑）。** 对相对 M0 有改善的模型，用门控与站点–节点注意力展示不同市场条件下模型依赖哪些模态与空间节点（RQ3）。

### 1.5 Dissertation structure

### 1.5 论文结构

- Chapter **2** Literature review
- Chapter **3** Methodology
- Chapter **4** Results
- Chapter **5** Discussion
- Chapter **6** Conclusion

- 第 **2** 章 文献综述
- 第 **3** 章 研究方法
- 第 **4** 章 结果
- 第 **5** 章 讨论
- 第 **6** 章 结论

---

## Chapter 2 — Literature Review *(~2,500–3,500)*

## 第 2 章 — 文献综述

---

## Chapter 3 — Methodology *(~2,500–3,000)*

## 第 3 章 — 研究方法 *(约 2,500–3,000 词)*

### 3.1 Research design

### 3.1 研究设计

- This study empirically compares Flat and Deep under one fair protocol, evaluating out-of-sample forecast performance across information sets and representation choices.
- Two architecture families share the same information sets, timeline, and metrics.
  - **Flat:** modality-derived features are combined in a common tabular representation and estimated using Ridge and XGBoost, providing classical early-fusion baselines.
  - **Deep:** each modality is processed by a modality-specific encoder before matched-dimensional latent representations are combined through representation-level fusion.
- Comparisons use the **M0** benchmark and the **M1–M4** information sets defined in §3.5. Within each architecture family, moving across M1–M4 isolates information increments; pairing Flat and Deep at matched information sets isolates representation / fusion. Together these contrasts separate *which* information is available from *how* it is represented. M0 anchors absolute skill.
- How each RQ is tested:
  1. **RQ1:** within each modelling family, compare M2, M3 and M4 with the finance-only M1 baseline, and compare all model forecasts with M0. Statistical tests are selected according to whether the competing forecast specifications are formally nested.
  2. **RQ2:** compare Flat and Deep pairwise at matched information sets, for example M3_Flat versus M3_Deep.
  3. **RQ3:** interpretability is applied to specifications that improve on M0 (Deep M3, and Deep M4 where relevant), using modality gates and site/node attention.

- 本研究在统一公平协议下对 Flat 与 Deep 做实证对照，比较不同信息集与表示方式的样本外预测表现。
- 两类架构共享相同信息集、时间线与指标。
  - **Flat：** 将各模态衍生特征并入统一表格表示，用 Ridge 与 XGBoost 估计，作为经典早融合基线。
  - **Deep：** 各模态先经模态专属编码器处理，再在匹配维度的潜表示上做表示级融合。
- 对照使用 §3.5 定义的 **M0** 基准与 **M1–M4** 信息集。各族架构内沿 M1–M4 移动以分离信息增量；在匹配信息集上配对 Flat 与 Deep，以分离表示 / 融合。二者合起来区分*有哪些信息*与*信息如何被表示*。M0 锚定绝对 skill。
- 各 RQ 如何检验：
  1. **RQ1：** 在各族建模框架内，将 M2、M3、M4 与仅金融的 M1 比较，并将全部模型预测与 M0 比较。统计检验按竞争预测设定是否形式上嵌套来选择。
  2. **RQ2：** 在匹配信息集上逐对比较 Flat 与 Deep，例如 M3_Flat 对 M3_Deep。
  3. **RQ3：** 可解释性应用于相对 M0 有改善的设定（Deep M3，必要时 Deep M4），工具为模态门控与站点/节点注意力。

### 3.2 Prediction target and timeline

### 3.2 预测目标与时间线

- **Forecasting objective:** predict the next-week Friday Brent spot price P_{t+1} using information available at forecast origin t.
- **Modelling target:** one-week log return r_{t+1}=\log(P_{t+1}/P_t).
- **Price reconstruction:** \hat P_{t+1\mid t}=P_t\exp(\hat r_{t+1\mid t}).
- **M0 equivalence:** the no-change price forecast \hat P_{t+1\mid t}=P_t corresponds to the zero-return forecast \hat r_{t+1\mid t}=0.
- Reported metrics and economic interpretation use reconstructed prices, not the internal return target.
- Calendar: Friday-ending weeks.
- Sample window: 2019–2025. Merged matrix ≈ 365 weeks. Common scored test span: 257 weeks (2021-01 to 2025-12).
- Forecast horizon: one week ahead. Directional accuracy is an auxiliary metric only and is not part of the training loss.

- **预测目标（forecasting objective）：** 利用预测起点 t 时可得信息，预测下一周周五 Brent 现货价 P_{t+1}。
- **建模目标（modelling target）：** 一周对数收益率 r_{t+1}=\log(P_{t+1}/P_t)。
- **价格还原：** \hat P_{t+1\mid t}=P_t\exp(\hat r_{t+1\mid t})。
- **M0 等价：** 无变化价格预测 \hat P_{t+1\mid t}=P_t 对应零收益预测 \hat r_{t+1\mid t}=0。
- 报告指标与经济解释使用还原价格，而非内部收益目标。
- 日历：周五截止周。
- 样本窗口：2019–2025。合并矩阵约 365 周。共同计分测试区间：257 周（2021-01 至 2025-12）。
- 预测步长：一周向前。方向准确率仅为辅助指标，不进入训练损失。

### 3.3 Data sources

### 3.3 数据来源

- Three modalities: finance, remote sensing, and shipping. Flat and Deep use the same underlying sources; they differ in how each source is represented.
- **Finance:** weekly oil-market and macro series from EIA, FRED, Yahoo and related indicators. This is the baseline before alternative data are added.
- **Remote sensing:** same 11 AOIs for Flat and Deep, but different products from a shared Sentinel-2 optical source family. **Flat:** monthly Sentinel-2 optical indices (NDVI/NDWI/NDBI/BSI) plus VIIRS night-time lights, as site-level anomalies. **Deep:** frozen Prithvi-EO-2.0 embeddings from monthly Sentinel-2 image patches only — no VIIRS. (Details in Appendix A.)
- **Shipping:** PortWatch chokepoint and port tanker flows, plus AIS-based vessel activity. Flat uses tabular shipping features; Deep uses a weekly 17-node heterogeneous graph (11 AOIs + 6 chokepoints). (Details in §3.7 and Appendix A.)

- 三模态：金融、遥感、航运。Flat 与 Deep 使用相同底层来源族，差异在于如何表示。
- **金融：** 来自 EIA、FRED、Yahoo 及相关指标的周度油市与宏观序列。这是加入另类数据前的基线。
- **遥感：** Flat 与 Deep 共用同一批 11 个 AOI，但来自共享 Sentinel-2 光学源族的不同产品。**Flat：** 月度 Sentinel-2 光学指数（NDVI/NDWI/NDBI/BSI）加 VIIRS 夜间灯光，以站点级异常表示。**Deep：** 仅用月度 Sentinel-2 影像块上的冻结 Prithvi-EO-2.0 嵌入——不含 VIIRS。（细节见附录 A。）
- **航运：** PortWatch 咽喉与港口油轮流量，以及基于 AIS 的船舶活动。Flat 用表格航运特征；Deep 用周度 17 节点异质图（11 个 AOI + 6 个咽喉）。（细节见 §3.7 与附录 A。）

### 3.4 Temporal alignment, lags, missingness

### 3.4 时间对齐、滞后期与缺失

- All series are aligned to the Friday-ending weekly calendar.
- Predictors enter only after their real publication time, so the model never uses future information.
- Different sources have different release lags; examples include EIA and PortWatch about one week, and slower monthly series with longer buffers.
- Flat models fill missing values using only past observations. Deep models keep explicit masks for missing modalities or sites instead of silently filling them away.

- 全部序列对齐到周五截止的周历。
- 预测变量仅在真实发布时间之后进入，模型从不使用未来信息。
- 不同来源有不同发布滞后；例如 EIA 与 PortWatch 约一周，较慢的月度序列有更长缓冲。
- Flat 模型仅用过去观测填补缺失。Deep 模型对缺失模态或站点保留显式掩码，而非静默填平。

### 3.5 Benchmark M0 and matched information sets M1–M4

### 3.5 基准 M0 与匹配信息集 M1–M4

- **M0** is the no-change / random-walk benchmark: next week's price equals this week's price. It is not trained and anchors absolute skill. It is not one of the modality sets.
- **M1–M4** are the modality (information) sets:
  - M1: finance only
  - M2: finance + remote sensing
  - M3: finance + shipping
  - M4: finance + remote sensing + shipping
- M2 and M3 are parallel branches from M1; M4 combines both.

- **M0** 为无变化 / 随机游走基准：下周价格等于本周价格。不训练，锚定绝对 skill。不属于模态集合。
- **M1–M4** 为模态（信息）集合：
  - M1：仅金融
  - M2：金融 + 遥感
  - M3：金融 + 航运
  - M4：金融 + 遥感 + 航运
- M2 与 M3 是从 M1 分出的平行分支；M4 合并二者。

### 3.6 Flat models

### 3.6 扁平模型

- Flat models concatenate all available numeric features for a given modality set into one weekly table, then flatten the last four weeks into one row.
- Two learners are used:
  - **Ridge:** linear model with L2 regularisation; a transparent linear early-fusion baseline.
  - **XGBoost:** non-linear tree ensemble. Captures interactions that Ridge misses, but still does not preserve modality structure.
- Both predict the log return and reconstruct price.
- Hyperparameters are chosen inside each training fold on a past validation slice only. Exact grids in Appendix C.

- Flat 模型将给定模态集下全部可得数值特征拼成一张周表，再把最近四周压成一行。
- 使用两种学习器：
  - **Ridge：** L2 正则线性模型；透明的线性早融合基线。
  - **XGBoost：** 非线性树集成。可捕捉 Ridge 错过的交互，但仍不保留模态结构。
- 二者均预测对数收益并还原价格。
- 超参数仅在各训练折内、用过去验证切片选取。精确网格见附录 C。

### 3.7 Deep models

### 3.7 深度模型

- Same modality sets, calendar, and validation protocol as Flat. The difference is representation and fusion, not the forecast target.
- Each available modality is encoded into a matched-dimensional representation, then fused.
- **Finance:** A causal TCN models temporal dependencies in the weekly financial sequence.
- **Remote sensing:** Deep RS uses frozen Prithvi-EO embeddings from monthly Sentinel-2 patches (no VIIRS). Embeddings are kept per site and aggregated by temporal and site attention, so the site dimension is not collapsed before encoding.
- **Shipping:** Shipping is encoded as a weekly heterogeneous graph with 17 nodes (11 AOIs and 6 chokepoints). Edges combine time-varying voyage flows between AOIs with fixed AOI–chokepoint links; a GAT with temporal encoding aggregates this network into a modality representation. Exact edge construction is reported in Appendix A; GAT depth, heads and related layer settings are in Appendix C.
- Each encoder is specified by its inputs, network structure, outputs, and why that architecture fits the modality.
- Fusion options for RQ2: concat as a simple control; gated fusion as the main reported design; cross-attention as an advanced alternative.
- The fused representation maps to the same return/price target as Flat. Training details are in Appendix C.

- 与 Flat 相同的模态集、日历与验证协议。差异在表示与融合，不在预测目标。
- 各可得模态编码为匹配维度的表示，再融合。
- **金融：** 因果 TCN 建模周度金融序列上的时间依赖。
- **遥感：** Deep RS 使用月度 Sentinel-2 影像块上的冻结 Prithvi-EO 嵌入（不含 VIIRS）。嵌入按站点保留，再经时间注意力与站点注意力聚合，因此站点维在编码前不被压扁。
- **航运：** 编码为周度异质图，17 个节点（11 个 AOI + 6 个咽喉）。边结合 AOI 之间时变航次流与固定的 AOI–咽喉连接；GAT 加时间编码将该网络聚合为模态表示。边构造见附录 A；GAT 层数、头数及相关层设置见附录 C。
- 各编码器分别给出输入、网络结构、输出，以及该架构与对应模态的匹配理由。
- RQ2 的融合选项：concat 作为简单对照；门控融合作为主报告设计；交叉注意力作为进阶备选。
- 融合表示映射到与 Flat 相同的收益/价格目标。训练细节见附录 C。

### 3.8 Hyperparameter selection

### 3.8 超参数选择

- Flat: tune inside each training fold on past validation weeks only.
- Deep: lock a main configuration after sweeps; report sensitivity to seed, lookback, fusion type, representation size and regularisation in Results / Appendix C.
- Exact values such as representation size, GAT depth/heads and grids are in Appendix C. Flat and Deep select and lock hyperparameters under the same protocol so that the comparison remains fair.

- Flat：仅在各训练折内、用过去验证周调参。
- Deep：sweep 后锁定主配置；在结果章 / 附录 C 报告对种子、回看长度、融合类型、表示维度与正则的敏感性。
- 表示维度、GAT 层数/头数与网格等精确值见附录 C。Flat 与 Deep 在同一协议下选取与锁定超参数，以保持比较公平。

### 3.9 Leakage-free validation protocol

### 3.9 无泄漏验证协议

- Expanding-window rolling-origin backtest: train only on past weeks, then forecast one week ahead.
- The first 104 weeks form the initial estimation and validation period and are not scored. The four-week lookback creates a separate sequence warm-up requirement. Thereafter models are refit every 13 weeks. Common scored test span: 257 weeks (2021-01 to 2025-12).
- Any scaling or filtering is fit inside the training fold only.
- Flat and Deep share the same fold calendar so architecture comparisons are fair.

- 扩展窗口滚动起点回测：仅用过去周训练，再向前预测一周。
- 前 104 周构成初始估计与验证期，不计分。四周回看另构成序列热身要求。此后每 13 周重训一次。共同计分测试区间：257 周（2021-01 至 2025-12）。
- 任何标准化或滤波仅在各折训练切片内拟合。
- Flat 与 Deep 共享同一折日历，以保证架构比较公平。

### 3.10 Evaluation, tests, interpretability

### 3.10 评估、检验与可解释性

- **Metrics.** Primary metrics on reconstructed price: RMSE and MAE for every comparison. Directional accuracy is auxiliary.
- **RMSE skill vs M0** (reported as a percentage in tables):

- **指标。** 还原价格上的主指标：每次比较均报告 RMSE 与 MAE。方向准确率为辅助指标。
- **相对 M0 的 RMSE skill**（表中以百分比报告）：

\mathrm{Skill}=100\times\left(1-\frac{\mathrm{RMSE}*{\mathrm{model}}}{\mathrm{RMSE}*{\mathrm{M0}}}\right).

Skill > 0 beats M0 on RMSE; = 0 matches M0; < 0 is worse than M0.

Skill > 0 表示 RMSE 优于 M0；= 0 与 M0 相同；< 0 弱于 M0。

- **Comparison logic.** The study evaluates both incremental value versus M1 and absolute skill versus M0, and distinguishes information-set nesting from formal model nesting: a larger modality set does not automatically make two forecasts nested for testing.
- **Test choice.** Tests follow the forecast-specification relationship:
  - **Clark–West (2007):** an MSPE-adjusted test for whether a larger model improves on a smaller one when the smaller forecast specification is nested in the larger; used for nested increments (e.g. Ridge M1 versus Ridge M2/M3/M4 where nesting is justified).
  - **Diebold–Mariano (1995):** a test of equal predictive accuracy based on the mean loss differential between two forecasts; used for non-nested paired comparisons (e.g. Flat versus Deep, XGBoost or Deep settings that change hyperparameters or architecture). A small-sample adjustment is noted where relevant.
  - Every comparison also reports RMSE and MAE effect sizes versus M0 and, where relevant, versus M1.
- **Interpretability.** Applied to specifications that improve on M0 (Deep M3, and Deep M4 where relevant); diagnostics are modality gate weights and site/node attention.
- **Reproducibility.** Data are public or licensed; the pipeline is scripted.

- **比较逻辑。** 本研究同时评估相对 M1 的增量价值与相对 M0 的绝对 skill，并区分信息集嵌套与形式模型嵌套：更大的模态集并不自动使两个预测在检验意义上嵌套。
- **检验选择。** 按预测设定关系选择检验：
  - **Clark–West (2007)：** 在较小预测设定嵌套于较大设定时，对 MSPE 差分做调整，检验较大模型是否改进较小模型；用于嵌套增量（例如在嵌套成立时，Ridge M1 对 Ridge M2/M3/M4）。
  - **Diebold–Mariano (1995)：** 基于两个预测损失差分的均值，检验等预测精度；用于非嵌套配对比较（例如 Flat 对 Deep，或改变超参/架构的 XGBoost、Deep 设定）。必要时注明小样本调整。
  - 每次比较另报告相对 M0、以及相关情形下相对 M1 的 RMSE 与 MAE 效应量。
- **可解释性。** 应用于相对 M0 有改善的设定（Deep M3，必要时 Deep M4）；诊断量为模态门控权重与站点/节点注意力。
- **可复现。** 数据公开或有许可；流水线脚本化。

---

## Chapter 4 — Results *(~2,500–3,200)*

## 第 4 章 — 结果 *(约 2,500–3,200 词)*

### 4.1 Descriptive overview

### 4.1 描述性概览

This chapter reports out-of-sample one-week-ahead Brent price forecasts on the common scored evaluation span defined in Section 3.9 (257 weeks, January 2021–December 2025).

本章报告第 3.9 节定义的共同计分评估区间（2021 年 1 月至 2025 年 12 月，共 257 周）上、一周向前的 Brent 价格样本外预测结果。

### 4.2 Flat-model results

### 4.2 扁平模型结果

Table 4.1 summarises out-of-sample Flat performance on the common scored span.

Every flat learned model has negative skill versus M0: under flat early fusion, the no-change benchmark retains the best absolute error. Contrasts within M1–M4 are nonetheless informative.

Relative to finance-only M1, adding shipping in M3 improves performance, suggesting that shipping still carries some incremental information in the flat setting.

By contrast, remote sensing under the main flat specification (M2) does not show a clear gain over M1.

表 4.1 汇总扁平模型在共同计分区间上的样本外表现。

全部扁平学习模型相对 M0 的 skill 均为负：在扁平早融合架构下，无变化基准仍保持最优绝对误差。尽管如此，M1–M4 内部对照仍有信息量。

相对仅金融的 M1，加入航运的 M3 有所改善，表明航运在扁平设定下仍能提供一定的增量信息。

相比之下，主设定下的遥感分支（M2）相对 M1 并未展现清晰的扁平收益。

**Table 4.1 — Flat out-of-sample performance** *(n = 257)*

**表 4.1 — Flat 样本外表现** *（n = 257）*


| Set | Content                  | Ridge RMSE | Ridge skill vs M0 | XGB RMSE | XGB skill vs M0 |
| --- | ------------------------ | ---------- | ----------------- | -------- | --------------- |
| M0  | no-change benchmark      | 4.152      | —                 | 4.152    | —               |
| M1  | finance only             | 4.256      | −2.5%             | 4.368    | −5.2%           |
| M2  | finance + remote sensing | 4.414      | −6.3%             | 4.440    | −6.9%           |
| M3  | finance + shipping       | 4.430      | −6.7%             | 4.429    | −6.7%           |
| M4  | finance + RS + shipping  | 4.525      | −9.0%             | 4.507    | −8.6%           |




### 4.3 Deep-model results

### 4.3 深度模型结果

Table 4.2 summarises out-of-sample Deep performance by information set. For single-modality M1 only the finance encoder applies; its result is reported in the gated column for comparability with later multimodal rows.

Finance plus remote sensing (M2) still fails to beat M0, so remote sensing remains weak in the deep setting as well.

By contrast, finance plus shipping (M3_Deep) improves on M0 under both gated and cross-attention fusion. Clearing the no-change benchmark is substantively meaningful for weekly Brent, even though skill versus M0 remains modest in magnitude (about +0.11% under gated fusion and +0.74% under cross-attention).

M4 does not clearly dominate M3: adding remote sensing on top of finance and shipping often fails to reduce absolute error further.

The main text reports gated fusion; cross-attention is included as a comparative architecture. Encoder-concatenation and the full fusion matrix are given in the appendix.

表 4.2 汇总深度模型在各信息集上的样本外表现。对单模态的 M1，仅金融编码器适用，其结果列于门控栏以便与后续多模态结果对照。

金融加遥感的 M2 仍未优于 M0，说明在深度设定下遥感同样偏弱。

相比之下，金融加航运的 M3_Deep 在门控与交叉注意力下均优于 M0。对周度 Brent 而言，越过无变化基准本身具有实质意义；同时，相对 M0 的 skill 幅度仍然有限（门控约 +0.11%，交叉注意力约 +0.74%）。

M4 并未明显优于 M3：在金融与航运之上再加入遥感，往往不能进一步降低绝对误差。

主文报告门控融合；交叉注意力作为对照架构。encoder-concat 与完整融合矩阵见附录。

**Table 4.2 — Deep out-of-sample performance** *(gated = main reported fusion)*

**表 4.2 — Deep 样本外表现** *（门控为主要报告融合）*


| Set | Content                  | Gated RMSE | Gated skill vs M0 | Xattn RMSE | Xattn skill vs M0 |
| --- | ------------------------ | ---------- | ----------------- | ---------- | ----------------- |
| M0  | no-change benchmark      | 4.152      | —                 | 4.152      | —                 |
| M1  | finance only             | 4.250      | −2.4%             | —          | —                 |
| M2  | finance + remote sensing | 4.253      | −2.4%             | —          | —                 |
| M3  | finance + shipping       | 4.147      | **+0.11%**        | 4.121      | **+0.74%**        |
| M4  | finance + RS + shipping  | 4.205      | −1.3%             | 4.147      | +0.12%            |




### 4.4 Flat versus Deep

### 4.4 Flat 对 Deep

Table 4.3 compares Flat and Deep at matched information sets, holding data content fixed in order to isolate representation and fusion.

Deep gains are clearest in multimodal settings, especially once shipping enters; replacing only the finance branch with a deep encoder yields a weaker improvement.

Representation-level, modality-aware fusion therefore outperforms flat counterparts in selected multimodal settings, particularly with shipping.

表 4.3 在匹配信息集上配对比较 Flat 与 Deep，从而在固定数据内容的条件下隔离表示与融合方式的差异。

Deep 的优势在多模态、尤其是含航运的设定中最为清晰；仅将金融分支从扁平替换为深度编码器，收益相对有限。

因此，表示级、模态感知融合在选定的多模态设定（尤其含航运）上优于对应的扁平模型。

**Table 4.3 — Paired Flat versus Deep**

**表 4.3 — 配对 Flat 与 Deep**


| Pair | Flat RMSE | Deep RMSE | skill flat | skill deep |
| ---- | --------- | --------- | ---------- | ---------- |
| M1   | 4.368     | 4.250     | −5.2%      | −2.4%      |
| M2   | 4.440     | 4.253     | −6.9%      | −2.4%      |
| M3   | 4.429     | 4.147     | −6.7%      | +0.11%     |
| M4   | 4.507     | 4.205     | −8.6%      | −1.3%      |




### 4.5 Robustness and sensitivity

### 4.5 稳健性与敏感性

For Flat models, the checks cover lookback length, remote-sensing feature variants, shipping feature tiers, and leave-one-modality-out analysis for M4. The results align with the main text: no flat model beats M0; shipping still helps relative to finance-only M1; adding remote sensing can raise absolute RMSE.

For Deep models, the checks cover random seeds, lookback length, representation size, fusion type, and early versus late test windows. The gated finance-plus-shipping configuration provides the more stable small positive skill; cross-attention can look stronger on a single seed but varies more across seeds.

At matched multimodal information sets, the Deep advantage over Flat remains under these checks, especially in shipping-inclusive settings.

对扁平模型，检查包括回看长度、遥感特征变体、航运特征层级，以及 M4 的留一模态分析。结果与主文一致：无一扁平模型击败 M0；航运相对仅金融的 M1 仍有帮助；加入遥感有时会抬高绝对 RMSE。

对深度模型，检查包括随机种子、回看长度、表示维度、融合类型，以及测试窗的早/晚期划分。门控融合下的金融加航运配置提供更稳定的小幅正 skill；交叉注意力可在单一种子上表现更强，但跨种子波动更大。

在匹配的多模态信息集上，Deep 相对 Flat 的优势在稳健性检查中仍然成立，尤其是在含航运的设定中。

### 4.6 Interpretability

### 4.6 可解释性

Over the test span, mean gate weights are about 0.56 for finance and 0.44 for shipping; the allocation evolves over time and adjusts around events such as the Russia–Ukraine war, the EU Russia oil-ban announcement, the OPEC+ surprise cut and Houthi Red Sea attacks.

Shipping node attention concentrates on major chokepoints — the highest mean weights are Hormuz, Suez, Bab el-Mandeb, Panama and the Cape of Good Hope.

Modality gates capture modality-level dependence and node attention captures spatial dependence; together they describe model dependence rather than causal effects on oil prices.

测试期内平均门控权重约为金融 0.56、航运 0.44；该分配随时间演变，并在俄乌冲突、欧盟对俄石油限制公告、OPEC+ 意外减产及胡塞红海袭击等事件附近出现调整。

航运节点注意力集中于主要咽喉——均值最高的节点依次为霍尔木兹、苏伊士、曼德海峡、巴拿马与好望角。

模态门控刻画模态级依赖，节点注意力刻画空间级依赖；二者共同描述模型依赖结构，而非对油价的因果效应。

## Chapter 5 — Discussion *(~1,600–1,900)*

## 第 5 章 — 讨论 *(约 1,600–1,900 词)*

### 5.1 RQ1 — Do alternative data help?

### 5.1 RQ1 — 另类数据是否有用？

- Under Flat, no model beats M0: weekly Brent remains hard to forecast beyond the no-change benchmark.
- Within the Flat family, shipping still adds value: M3 improves on finance-only M1, while remote sensing shows no clear gain over M1; full flat RS features add little, and sparse or cleaned variants do not overturn this main result.
- Under Deep, finance + shipping shows only a **small** improvement over M0; adding remote sensing on top often brings no further gain.
- Overall, shipping has value relative to finance, but absolute gains remain limited and do not support strong economic claims under the weekly design.

- Flat 下无一模型击败 M0：周度 Brent 仍难超越无变化基准。
- Flat 族内，航运仍有增量：M3 优于仅金融的 M1；遥感相对 M1 无清晰收益，完整扁平 RS 特征增量有限，稀疏或清洗变体亦不推翻这一主结果。
- Deep 下，金融 + 航运相对 M0 仅有**小幅**改善；在此之上再加遥感往往无进一步收益。
- 综合而言，航运相对金融有价值，但绝对收益仍然有限，不足以在周度设计下支撑强经济主张。

### 5.2 RQ2 — Does representation-level fusion beat flat fusion?

### 5.2 RQ2 — 表示级融合是否优于扁平融合？

- At matched information sets, Deep outperforms Flat most clearly once shipping enters.
- Replacing Flat with Deep on finance alone yields limited gains; the architecture gap opens mainly in multimodal settings.
- That advantage comes from preserving temporal, site and network structure, and from how modalities are fused — not from giving Deep a different data mix.
- The Deep advantage is concentrated in selected shipping-inclusive settings and does not extend across all specifications.

- 在匹配信息集上，Deep 在航运进入后最明显优于 Flat。
- 仅金融设定下把 Flat 换成 Deep 收益有限；架构差距主要在多模态设定中打开。
- 该优势来自保留时间、站点与网络结构，以及模态如何融合，而非 Deep 使用了不同数据配比。
- Deep 优势集中在选定含航运设定，未扩展到全部设定。

### 5.3 RQ3 — What does the model rely on when value exists?

### 5.3 RQ3 — 有价值时模型依赖什么？

- Interpretability evidence comes mainly from Deep M3 (and Deep M4).
- Modality gates correspond to modality-level dependence; node and site attention correspond to spatial dependence.
- Shipping gate weight tends to rise in disruption or chokepoint-stress windows; spatial attention often concentrates on major routes and export nodes.
- These patterns describe model dependence rather than causal effects on prices; a high shipping gate alone does not identify which port or chokepoint is attended.

- 可解释性证据主要来自 Deep M3（及 Deep M4）。
- 模态门控对应模态级依赖；节点与站点注意力对应空间级依赖。
- 航运门控权重倾向在中断或咽喉压力窗口上升；空间注意力常集中于主要航线与出口节点。
- 这些模式描述模型依赖结构，而非对价格的因果效应；高航运门控本身也不等于识别具体港口或咽喉被关注。

### 5.4 Implications

### 5.4 含义

- In alternative-data fusion research, a fair shared evaluation protocol matters as much as proposing a new fusion module for how credible the conclusions are.
- In this study, the nested M1 contrasts and the absolute M0 contrasts jointly show that shipping can help relative to finance while absolute gains remain small.
- Gains over M0 are too small to support a forecasting-breakthrough conclusion.
- Null or near-null Flat results are themselves informative: early fusion of alternative data is not automatically useful.

- 在另类数据融合研究中，公平共享的评估协议与提出新融合模块同样决定结论的可信度。
- 在本研究中，相对 M1 的嵌套对照与相对 M0 的绝对对照合起来表明：航运相对金融可以有帮助，但绝对收益仍然很小。
- 相对 M0 的收益很小，不足以支撑预测突破式结论。
- Flat 的零结果或近零结果本身有信息量：另类数据的早融合并不自动有用。

### 5.5 Limitations

### 5.5 局限

- The study uses a weekly horizon and a modest scored sample after warm-up.
- Alternative-data proxies are noisy and may respond to prices as well as lead them.
- Frozen EO embeddings, shipping-graph construction, and missingness rules all affect Deep results.
- Some Deep configurations, especially cross-attention, are sensitive to random seeds.

- 研究采用周度视界，热身后计分样本量有限。
- 另类数据代理嘈杂，可能既响应价格也可能领先价格。
- 冻结 EO 嵌入、航运图构造与缺失规则都会影响 Deep 结果。
- 部分 Deep 配置（尤其交叉注意力）对随机种子敏感。

### 5.6 Future research

### 5.6 未来研究

- Extend the best Deep specifications to longer history and more seeds.
- Enrich the shipping graph and strengthen stress tests under missing modalities.
- Apply the same Flat–Deep protocol to other forecast horizons or related energy commodities.

- 将最佳 Deep 设定扩展到更长历史与更多种子。
- 丰富航运图，并加强对缺失模态的压力测试。
- 在同一 Flat–Deep 协议下扩展到其他预测视界或相关能源商品。

---

## Chapter 6 — Conclusion *(~400–700)*

## 第 6 章 — 结论 *(约 400–700 词)*

### 6.1 Summary of findings

### 6.1 发现摘要

- Under Flat, M0 is best and M1–M4 do not beat the no-change benchmark; within the Flat family, shipping still improves on finance only.
- Under Deep, M0 remains strong; Deep M3 / M4 show only a small improvement over M0, and remote sensing is secondary to shipping.
- At matched multimodal information sets, Deep outperforms Flat, most clearly in shipping-inclusive settings.
- For models that improve on M0, modality gates and spatial attention show when and where shipping is relied upon.

- Flat 下 M0 最优，M1–M4 均未击败无变化基准；但 Flat 族内航运相对仅金融仍有改善。
- Deep 下 M0 仍然很强；Deep M3 / M4 相对 M0 仅有小幅改善，且遥感次于航运。
- 在匹配多模态信息集上，Deep 优于 Flat，这一差距在含航运设定中最明显。
- 对相对 M0 有改善的模型，模态门控与空间注意力显示航运在何时、何处被模型依赖。

### 6.2 Contributions

### 6.2 贡献

- A systematic nested comparison of finance, remote sensing and shipping for weekly Brent under one leakage-safe protocol, separating incremental value versus the finance baseline from absolute performance versus the no-change benchmark.
- A paired Flat versus Deep comparison at matched information sets, separating differences in data content from differences in representation and fusion design.
- Gating and spatial attribution on models that improve on M0, so explanation stays aligned with forecast evidence.

- 在同一无泄漏协议下，对周度 Brent 完成金融、遥感与航运的系统嵌套比较，区分相对金融基线的增量与相对无变化基准的绝对表现。
- 在匹配信息集上配对 Flat 与 Deep，从而分离数据内容差异与表示 / 融合设计差异。
- 对相对 M0 有改善的模型做门控与空间归因，使解释与预测证据对齐。

### 6.3 Final conclusion

### 6.3 最终结论

- In this weekly Brent design, shipping adds incremental value over finance, and modality-aware Deep fusion can yield a small further improvement over the no-change benchmark; remote sensing does not show stable help.
- These gains remain modest: alternative data and representation-level fusion can be useful, but not enough to support strong forecasting or strong economic claims.

- 在本周度 Brent 设计中，航运相对金融有增量，模态感知 Deep 融合可进一步带来相对无变化基准的小幅改善；遥感则未显示稳定帮助。
- 这些收益幅度有限：另类数据与表示级融合可以有用，但不足以支撑强预测或强经济主张。

---

## References

## 参考文献

## Appendices

## 附录

- **A.** Full variable dictionaries (M1–M4); AOI / chokepoint node lists; lag table; shipping graph edge definition (voyage flows, AOI–chokepoint links, adjacency handling, edge-weight transform)
- **B.** Extra result / robustness tables & figures (lookback, LOAO, LOCHO, LOMO, water-mask RS variant, fusion matrix, seeds, early/late)
- **C.** Hyperparameter grids and locked Deep settings (representation size, GAT layers/heads, seeds, software / config paths)

- **A.** 完整变量词典（M1–M4）；AOI / 咽喉节点列表；滞后期表；航运图边定义（航次流、AOI–咽喉连接、邻接处理、边权变换）
- **B.** 额外结果 / 稳健性表与图（回看、LOAO、LOCHO、LOMO、水体掩膜遥感变体、融合矩阵、种子、早/晚期）
- **C.** 超参数网格与锁定的 Deep 设定（表示维度、GAT 层数/头数、种子、软件 / 配置路径）
