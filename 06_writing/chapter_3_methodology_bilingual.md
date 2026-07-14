# Chapter 3 — Data and Methods
# 第 3 章 — 数据与方法

This chapter describes the empirical design used to answer RQ1–RQ3. The guiding principle is fairness: flat and deep models are compared on the same information sets, the same forecast timeline and the same leakage-safe validation protocol. Detailed variable dictionaries, hyperparameter grids and long feature tables are placed in the Appendix so that the chapter remains focused on design choices that affect inference.

本章说明用于回答 RQ1–RQ3 的实证设计。指导原则是公平性：扁平与深度模型在相同信息集、相同预测时间线与相同无泄漏验证协议下比较。详细变量词典、超参数网格和长特征表放入附录，使本章聚焦影响推断的设计选择。

## 3.1 Research design
## 3.1 研究设计

The study is an empirical comparison rather than a proposal of a new fusion operator. It integrates existing methods — regularised linear and tree models for flat fusion; modality-specific encoders with gated or cross-attention fusion for representation-level learning — and asks whether alternative data and fusion design improve weekly Brent forecasts under a shared protocol.

本研究是一项实证比较，而非提出新的融合算子。它集成既有方法——扁平融合用正则化线性与树模型；表示级学习用模态专属编码器加门控或交叉注意力融合——并在统一协议下检验替代数据与融合设计是否改善周度 Brent 预测。

Two architecture families are evaluated on the same M0–M4 information ladder. The **flat** family concatenates available numeric features into one table. The **deep** family encodes finance, remote sensing and shipping separately, then fuses the resulting representations. This design separates the effect of *which* information is available from the effect of *how* that information is represented.

两类架构在同一 M0–M4 信息阶梯上评估。**扁平**族将可得数值特征拼成一张表。**深度**族分别编码金融、遥感与航运，再融合所得表示。该设计把“有哪些信息”与“信息如何被表示”的效应分开。

The comparison is organised so that each research question maps onto a concrete contrast. RQ1 is answered by moving up the M0–M4 ladder within each architecture and by testing nested increments against M1 as well as absolute skill against M0. RQ2 is answered by pairing flat and deep models that share the same information set. RQ3 is answered by SHAP, gating and attention diagnostics under the interpretability rule in Section 3.10. This mapping keeps Methods, Results and Discussion aligned.

比较的组织方式使每个研究问题对应到具体对照。RQ1 通过在各族架构内沿 M0–M4 阶梯上行，并同时检验相对 M1 的嵌套增量与相对 M0 的绝对 skill 来回答。RQ2 通过共享同一信息集的扁平与深度模型配对来回答。RQ3 通过第 3.10 节可解释性规则下的 SHAP、门控与注意力诊断来回答。该映射使方法、结果与讨论保持对齐。

## 3.2 Prediction targets and forecasting timeline
## 3.2 预测目标与预测时间线

The sole research target is next-week Brent spot price \(P_{t+1}\) (USD/barrel), aligned to Friday close or the last trading day of the week. Models are trained on the one-week log return
\[
r_{t+1}=\log(P_{t+1}/P_t),
\]
and the price forecast is reconstructed as \(\hat P_{t+1}=P_t\,e^{\hat r_{t+1}}\). Training on returns is an internal learning choice; evaluation and economic interpretation are reported on reconstructed prices. Directional accuracy is an auxiliary metric and is not part of the training loss.

唯一研究目标是下一周 Brent 现货价格 \(P_{t+1}\)（美元/桶），对齐周五收盘或该周最后一个交易日。模型在一周对数收益率
\[
r_{t+1}=\log(P_{t+1}/P_t)
\]
上训练，价格预测还原为 \(\hat P_{t+1}=P_t\,e^{\hat r_{t+1}}\)。在收益率上训练是内部学习选择；评估与经济解释均报告在还原价格上。方向准确率是辅助指标，不进入训练损失。

The common comparison window is 2019–2025. The main lookback is four weeks, matching the locked flat-protocol setting used for fair Flat–Deep comparison. Forecast origins use only information available at that origin.

共同比较窗口为 2019–2025。主模型回看长度为四周，与锁定的扁平协议一致，以保证 Flat–Deep 公平比较。预测起点仅使用该起点时真实可得的信息。

## 3.3 Data sources
## 3.3 数据来源

Three modalities enter the design.

设计纳入三种模态。

**Finance (M1).** Weekly market and oil-fundamental series include Brent and WTI prices, EIA inventories, production, imports/exports and refinery utilisation, together with volatility, exchange-rate, interest-rate, equity and geopolitical-risk indicators. These variables form the economically informed baseline before alternative data are added.

**金融（M1）。** 周度市场与原油基本面序列包括 Brent 与 WTI 价格、EIA 库存、产量、进出口与炼厂利用率，以及波动率、汇率、利率、股指与地缘政治风险指标。这些变量在加入替代数据前构成具有经济含义的基线。

**Remote sensing (M2).** Site-level signals are constructed for 11 oil-infrastructure areas of interest (AOIs) spanning export terminals, transit/storage nodes and demand-side ports/refineries. The flat arm uses engineered monthly optical indices (e.g. NDVI/NDWI/NDBI/BSI) and VIIRS night-time lights, expressed as within-site anomalies where appropriate. The deep arm additionally uses frozen Earth-observation embeddings from monthly Sentinel-2 patches (Prithvi-EO style representations), with site and temporal attention learned on top of the frozen backbone.

**遥感（M2）。** 为 11 个石油基础设施兴趣区（AOI）构建站点级信号，覆盖出口码头、中转/仓储节点与需求侧港口/炼厂。扁平分支使用人工月度光学指数（如 NDVI/NDWI/NDBI/BSI）和 VIIRS 夜间灯光，并在适当时表示为站内异常。深度分支额外使用月度 Sentinel-2 影像块上的冻结地球观测嵌入（Prithvi-EO 风格表示），并在冻结骨干之上学习站点与时间注意力。

**Shipping (M3).** Shipping inputs combine IMF PortWatch chokepoint and port tanker activity with Global Fishing Watch vessel-presence, port-visit and voyage information. In the flat arm these become high-dimensional tabular features. In the deep arm they are organised as a weekly 17-node heterogeneous graph (11 AOIs + 6 chokepoints) with dynamic voyage edges and static AOI–chokepoint links.

**航运（M3）。** 航运输入结合 IMF PortWatch 咽喉与港口油轮活动，以及 Global Fishing Watch 船舶存在度、港口停靠与航次信息。扁平分支将其转为高维表格特征；深度分支将其组织为周度 17 节点异质图（11 个 AOI + 6 个咽喉），含动态航次边与静态 AOI–咽喉连接。

Full source lists, capacity justifications for AOI selection and complete feature dictionaries are reported in Appendix A.

完整来源清单、AOI 选择的运力依据与完整特征词典见附录 A。

The eleven AOIs were chosen for throughput rank, geographic and supply-chain diversity, and remote-sensing observability. They cover major Persian Gulf and Red Sea export terminals, transit/storage nodes near key routes, and demand-side ports or refineries in Europe, North America and Asia. The six chokepoints used in the shipping modality — including Hormuz, Suez, Malacca, Bab el-Mandeb, Panama and the Cape of Good Hope — provide a complementary route-level view of physical trade disruption risk.

十一个 AOI 按运力位次、地理与供应链多样性，以及遥感可观测性选取。它们覆盖波斯湾与红海主要出口码头、关键航线附近的中转/仓储节点，以及欧洲、北美与亚洲的需求侧港口或炼厂。航运模态中的六个咽喉——包括霍尔木兹、苏伊士、马六甲、曼德海峡、巴拿马与好望角——为实物贸易中断风险提供互补的航线级视角。

## 3.4 Temporal alignment, lagging and missingness
## 3.4 时间对齐、滞后期与缺失

All predictors are aligned by **publication timestamp**, not by statistical reference date. Series that are released after their reference period — for example EIA weekly petroleum reports typically published the following Wednesday — enter the model only after they would have been available at the forecast origin. Monthly remote-sensing observations use a conservative as-of rule (month-end plus a release buffer). Shipping blocks apply source-specific publication lags before any derived moving averages are used in forecasting.

所有预测变量按**发布时间戳**对齐，而非按统计参考日期对齐。在参考期之后才发布的序列——例如通常于次周三发布的 EIA 周报——仅在预测起点真实可得之后进入模型。月度遥感观测采用保守的 as-of 规则（月末加发布缓冲）。航运数据块在用于预测前施加来源特定的发布滞后期，衍生移动平均亦遵守该约束。

Missingness is treated as part of the forecasting problem. Flat models use past-only filling rules that never borrow future values. Deep models retain modality masks and, where trained, modality dropout, so that irregular satellite revisits and incomplete shipping coverage are visible to the network rather than silently imputed away.

缺失被视为预测问题本身的一部分。扁平模型使用仅向过去填充、绝不借用未来值的规则。深度模型保留模态掩码，并在训练时使用模态 dropout，使不规则卫星重访与不完整航运覆盖对网络可见，而不是被静默填平。

## 3.5 M0–M4 information sets
## 3.5 M0–M4 信息集

The information ladder is identical for flat and deep arms:

扁平与深度分支使用相同的信息阶梯：

| Set | Contents |
|-----|----------|
| **M0** | No-change / random-walk benchmark: \(\hat P_{t+1}=P_t\) |
| **M1** | Finance / macro-oil fundamentals only |
| **M2** | M1 + remote sensing |
| **M3** | M1 + shipping |
| **M4** | M1 + remote sensing + shipping |

| 集合 | 内容 |
|------|------|
| **M0** | 无变化 / 随机游走基准：\(\hat P_{t+1}=P_t\) |
| **M1** | 仅金融 / 宏观—原油基本面 |
| **M2** | M1 + 遥感 |
| **M3** | M1 + 航运 |
| **M4** | M1 + 遥感 + 航运 |

This ladder supports nested tests of incremental value (RQ1) and paired architecture comparisons at each information set (RQ2).

该阶梯支持增量价值的嵌套检验（RQ1），以及各信息集上的配对架构比较（RQ2）。

## 3.6 Flat models
## 3.6 扁平模型

Flat models concatenate all columns available for a given information set into one weekly feature matrix. Two learners are used as complementary flat baselines: **Ridge** regression, representing a linear regularised model, and **XGBoost**, representing a non-linear tree ensemble. Early concatenation is deliberate: it is the dominant practice in multi-source oil forecasting and therefore the correct foil for representation-level fusion.

扁平模型将给定信息集下的全部列拼入一张周度特征矩阵。使用两类互补的扁平基线学习器：**Ridge** 回归代表线性正则化模型，**XGBoost** 代表非线性树集成。早拼接是有意为之：它是多源油价预测中的主流做法，因而也是表示级融合的正确对照。

Hyperparameters for Ridge and XGBoost are selected within each training fold using past data only. The important design point is not the exact grid, but that both learners see the same lagged feature matrix and the same rolling-origin schedule as the deep arm. This prevents a spurious architecture comparison in which one side enjoys a more generous protocol.

Ridge 与 XGBoost 的超参数在各训练折内仅用历史数据选择。关键设计点不在于具体网格，而在于两类学习器与深度分支看到相同的滞后特征矩阵与相同的滚动起点日程。这避免了其中一方享受更宽松协议的虚假架构比较。

## 3.7 Deep and representation-level models
## 3.7 深度与表示级模型

The deep architecture learns a 32-dimensional representation for each available modality and then fuses those representations.

深度架构为每个可得模态学习 32 维表示，再融合这些表示。

- **Finance encoder:** a causal temporal convolutional network over the lagged finance window.
- **金融编码器：** 在滞后金融窗口上的因果时间卷积网络。
- **Remote-sensing encoder:** frozen EO embeddings per AOI, followed by temporal and site attention with availability masks.
- **遥感编码器：** 各 AOI 的冻结 EO 嵌入，随后是带可得性掩码的时间与站点注意力。
- **Shipping encoder:** graph attention over the 17-node heterogeneous weekly graph, followed by a temporal convolution and node pooling.
- **航运编码器：** 在 17 节点异质周度图上的图注意力，随后时间卷积与节点池化。

Fusion mechanisms include encoder concatenation, **gated** fusion and **cross-attention** fusion. The main reported deep specification uses gated fusion for stability; cross-attention is retained as an advanced, higher-variance alternative. The prediction head maps the fused representation to \(\hat r_{t+1}\) and reconstructs price exactly as in the flat arm.

融合机制包括编码器拼接、**门控**融合与**交叉注意力**融合。主报告的深度设定采用门控融合以保持稳定；交叉注意力保留为方差更高的进阶方案。预测头将融合表示映射为 \(\hat r_{t+1}\)，并以与扁平分支完全相同的方式还原价格。

## 3.8 Training and hyperparameter selection
## 3.8 训练与超参数选择

Deep models are trained with Adam, early stopping on an inner validation slice within each training fold, and moderate regularisation. Representation dimension, lookback and regularisation strength were explored in sweeps; the main specification locks lookback = 4 and dimension = 32 to remain protocol-aligned with the flat baselines. Full search grids and seed lists are reported in Appendix C.

深度模型使用 Adam、各训练折内验证切片上的早停，以及适度正则化。表示维度、回看长度与正则强度经扫描探索；主设定锁定 lookback = 4、维度 = 32，以与扁平基线协议对齐。完整搜索网格与种子列表见附录 C。

## 3.9 Leakage-free validation protocol
## 3.9 无泄漏验证协议

All models use an **expanding-window rolling-origin** backtest. The first 104 weeks are used only for warm-start training and are not scored. Thereafter the model is refit every 13 weeks on all past observations and produces one-step forecasts for subsequent weeks. The common test span covers 257 weeks from 2021-01 to 2025-12. Standardisation and any anomaly transforms are fit on the training slice of each fold only.

所有模型使用**扩展窗口滚动起点**回测。前 104 周仅用于热身训练，不计分。此后每隔 13 周在全部历史观测上重训，并对后续周次做一步预测。共同测试区间覆盖 2021-01 至 2025-12 共 257 周。标准化与任何异常变换仅在各折的训练切片上拟合。

## 3.10 Evaluation, DM/CW tests and interpretability
## 3.10 评估、DM/CW 检验与可解释性

Primary metrics are RMSE and MAE on reconstructed price, skill relative to M0,
\[
\mathrm{skill}=1-\frac{\mathrm{RMSE}_{\mathrm{model}}}{\mathrm{RMSE}_{\mathrm{M0}}},
\]
and directional accuracy as an auxiliary measure. Formal tests follow the nesting structure of the claim being made:

主要指标为还原价格上的 RMSE 与 MAE、相对 M0 的 skill
\[
\mathrm{skill}=1-\frac{\mathrm{RMSE}_{\mathrm{model}}}{\mathrm{RMSE}_{\mathrm{M0}}},
\]
以及作为辅助度量的方向准确率。正式检验遵循所主张内容的嵌套结构：

- **Clark–West** for nested increments (e.g. M2/M3/M4 versus M1; learned models versus nested no-change).
- **Clark–West** 用于嵌套增量（如 M2/M3/M4 相对 M1；学习模型相对嵌套的无变化基准）。
- **Diebold–Mariano** (HLN small-sample corrected) for non-nested comparisons (e.g. deep versus flat models of different families).
- **Diebold–Mariano**（HLN 小样本修正）用于非嵌套比较（如不同族的深度与扁平模型）。

A significant Clark–West increment over M1 is **not** treated as evidence of beating M0. Absolute skill and nested information are reported separately throughout.

相对 M1 显著的 Clark–West 增量**不**被视为击败 M0 的证据。绝对 skill 与嵌套信息在全文中分开报告。

**Interpretability rule.** Interpretability analysis is primarily conducted for the best-performing models that outperform the relevant benchmark. Supplementary SHAP analysis is also reported for models showing statistically significant incremental gains over M1, even where they do not surpass M0. For deep models, gate weights and site/chokepoint attention provide modality-level diagnostics. Attribution describes model dependence; it does not establish causal effects.

**可解释性规则。** 可解释性分析主要针对相对相关基准表现最佳的模型。对虽未超过 M0、但相对 M1 显示统计显著增量收益的模型，亦报告补充性 SHAP 分析。对深度模型，门控权重与站点/咽喉注意力提供模态级诊断。归因描述的是模型依赖，并不确立因果效应。

Data are public or licensed research sources and contain no personal identifiers. Code, configuration files and output paths are retained so that the main tables can be regenerated.

数据为公开或获许可的研究来源，不含个人身份信息。保留代码、配置文件与输出路径，以便复现主要表格。
