# Chapter 4 — Results
# 第 4 章 — 结果

This chapter reports out-of-sample results under the locked protocol of Chapter 3. The presentation follows the research questions: experimental overview; flat-model evidence for RQ1; deep-model evidence within the representation-level arm; paired Flat–Deep comparisons for RQ2; robustness; and interpretability for RQ3. Unless noted, metrics are computed on reconstructed price over the common 257-week test span (2021-01 to 2025-12).

本章报告第 3 章锁定协议下的样本外结果。叙述按研究问题展开：实验概览；面向 RQ1 的扁平模型证据；表示级分支内的深度模型证据；面向 RQ2 的配对 Flat–Deep 比较；稳健性；以及面向 RQ3 的可解释性。除非另有说明，指标均在共同 257 周测试区间（2021-01 至 2025-12）的还原价格上计算。

## 4.1 Descriptive analysis and experimental overview
## 4.1 描述性分析与实验概览

Weekly Brent log returns have near-zero mean and clear volatility clustering. Persistence makes the no-change forecast a strong competitor: on the test set, M0 achieves an RMSE of **4.152 USD/barrel**. All learned models are judged against this number through skill, DM/CW tests and absolute error.

周度 Brent 对数收益率均值接近零，并呈现明显的波动率聚集。价格持久性使无变化预测成为强竞争者：在测试集上，M0 的 RMSE 为 **4.152 美元/桶**。所有学习模型均通过 skill、DM/CW 检验与绝对误差对照该数字。

Exploratory checks on remote-sensing anomalies show only weak contemporaneous association with Brent returns, and several stronger associations occur at non-positive leads. This tempers expectations for M2 and motivates formal testing rather than visual claims of predictability. Shipping activity is likewise treated as a noisy physical proxy, not as a direct measure of future price.

对遥感异常的探索性检查显示，其与 Brent 收益率的同期关联较弱，若干更强关联出现在非正领先期。这降低了对 M2 的预期，并促使采用正式检验而非凭可视化宣称可预测性。航运活动同样被视为有噪声的实物代理，而非未来价格的直接度量。

## 4.2 Flat-model results
## 4.2 扁平模型结果

Table 4.1 summarises flat M0–M4 performance.

表 4.1 汇总扁平 M0–M4 表现。

**Table 4.1 — Flat M0–M4 out-of-sample performance (price RMSE, 257 weeks)**
**表 4.1 — 扁平 M0–M4 样本外表现（价格 RMSE，257 周）**

| Modality | Model | RMSE | MAE | DirAcc | Skill vs M0 | DM_p (vs M0) | CW_p (vs M1) |
|----------|-------|-----:|----:|-------:|------------:|-------------:|-------------:|
| **M0** | Random walk | **4.152** | 3.011 | – | 0.0% | – | – |
| M1 Finance | Ridge | 4.332 | 3.081 | 0.490 | −4.3% | 0.91 | – |
| | XGB | 4.771 | 3.406 | 0.525 | −14.9% | 0.98 | – |
| M2 +RS | Ridge | 4.411 | 3.208 | 0.518 | −6.3% | 0.96 | 0.212 |
| | XGB | 4.643 | 3.300 | 0.506 | −11.8% | 0.98 | **0.006** |
| M3 +Shipping | Ridge | 4.592 | 3.278 | 0.502 | −10.6% | 0.96 | 0.481 |
| | XGB | 4.456 | 3.227 | 0.498 | −7.3% | 0.96 | **2.5e-5** |
| M4 All | Ridge | 4.560 | 3.313 | 0.482 | −9.8% | 0.96 | 0.228 |
| | XGB | 4.470 | 3.284 | 0.502 | −7.7% | 0.99 | **1.7e-4** |

Three patterns stand out.

有三个突出模式。

**M0 versus M1.** Neither Ridge nor XGBoost on finance alone beats M0. The best flat finance model is Ridge (skill −4.3%), still worse than the random walk on RMSE. Absolute weekly Brent forecasting remains difficult.

**M0 与 M1。** 仅金融信息上的 Ridge 与 XGBoost 均未击败 M0。最佳扁平金融模型是 Ridge（skill −4.3%），RMSE 仍差于随机游走。周度 Brent 的绝对预测仍然困难。

**Incremental value of M2/M3/M4 over M1.** Under XGBoost, Clark–West tests show significant nested increments for M2 (p = 0.006), M3 (p = 2.5×10⁻⁵) and M4 (p = 1.7×10⁻⁴). Shipping provides the strongest nested signal among flat alternatives. Under Ridge, none of these nested increments is significant. High-dimensional shipping in particular appears model-sensitive: XGBoost can use it, while linear Ridge cannot.

**M2/M3/M4 相对 M1 的增量。** 在 XGBoost 下，Clark–West 显示 M2（p = 0.006）、M3（p = 2.5×10⁻⁵）与 M4（p = 1.7×10⁻⁴）具有显著嵌套增量。航运是扁平替代数据中最强的嵌套信号。在 Ridge 下，这些嵌套增量均不显著。高维航运尤其对模型敏感：XGBoost 可以利用它，线性 Ridge 则不能。

**Absolute skill versus M0.** Despite nested gains over M1, every flat learned model has negative skill and non-significant DM tests against M0. A significant Clark–West increment over finance is therefore not evidence of beating the random walk.

**相对 M0 的绝对 skill。** 尽管相对 M1 有嵌套收益，所有扁平学习模型的 skill 均为负，且相对 M0 的 DM 检验不显著。因此，相对金融基线显著的 Clark–West 增量，并不等于击败随机游走。

## 4.3 Deep-model results
## 4.3 深度模型结果

Table 4.2 reports selected deep configurations on the same 257-week span (seed 42, lookback 4, gated fusion unless noted).

表 4.2 报告相同 257 周区间上的选定深度配置（种子 42，回看 4，除非注明否则为门控融合）。

**Table 4.2 — Selected deep out-of-sample results**
**表 4.2 — 选定深度模型样本外结果**

| Model | RMSE | Skill vs M0 | DirAcc | CW vs M0 | Notes |
|-------|-----:|------------:|-------:|---------:|-------|
| M0 random walk | 4.152 | 0.0% | – | – | Benchmark |
| Deep finance (Mfin) | 4.250 | −2.4% | 0.494 | 0.315 | Finance encoder only |
| Deep RS (Mrs) | 4.247 | −2.3% | 0.459 | 0.928 | RS encoder only |
| Deep shipping (Mship) | 4.168 | −0.4% | 0.506 | 0.496 | Shipping graph only |
| **Deep M3 (Mfinship, gated)** | **4.147** | **+0.11%** | 0.529 | 0.166 | Finance + shipping |
| Deep M2 (Mfinrs, gated) | 4.253 | −2.4% | 0.475 | 0.769 | Finance + RS |
| Deep M4 (Mfull, gated) | 4.205 | −1.3% | 0.502 | 0.894 | All three modalities |
| Deep M3 (xattn) | 4.121 | +0.74% | 0.549 | **0.041** | Higher ceiling, less stable |
| Deep M4 (xattn) | 4.147 | +0.12% | 0.564 | **0.018** | Significant CW vs M0; seed-sensitive |

Within the deep arm, adding shipping to finance is the clearest modality gain: nested CW for Mfinship versus Mfin is highly significant (p ≈ 0.0006). Adding remote sensing alone does not produce a competitive deep model, and adding RS on top of finance+shipping does not improve gated M4 relative to gated M3. Cross-attention can push M3/M4 skill above zero with significant CW versus M0 in the main seed, but multi-seed checks show larger variance than gated fusion (Section 4.5). The honest headline for the stable gated specification is therefore modest: deep M3 achieves a small positive skill, while M0 remains a very strong competitor and gated M4 does not dominate M3.

在深度分支内，向金融加入航运是最清晰的模态收益：Mfinship 相对 Mfin 的嵌套 CW 高度显著（p ≈ 0.0006）。单独加入遥感无法形成有竞争力的深度模型；在金融+航运之上再加遥感，门控 M4 也不优于门控 M3。交叉注意力可在主种子下把 M3/M4 的 skill 推到零以上并对 M0 的 CW 显著，但多种子检查显示其方差大于门控融合（第 4.5 节）。对稳定的门控设定，诚实表述应是：深度 M3 取得小幅正 skill，而 M0 仍是很强的竞争者，门控 M4 并不主导 M3。

## 4.4 Flat versus deep comparison
## 4.4 扁平与深度比较

RQ2 requires **paired** comparisons by information set, not only a comparison of each family’s best model.

RQ2 要求按信息集进行**配对**比较，而不是只比较各族的最优模型。

**Table 4.3 — Paired Flat vs Deep comparisons (illustrative main specs)**
**表 4.3 — 配对 Flat vs Deep 比较（主设定示意）**

| Information set | Flat reference | Deep reference | Directional reading |
|-----------------|----------------|----------------|---------------------|
| M1 Finance | Ridge 4.332 / XGB 4.771 | Mfin 4.250 | Deep finance improves on flat finance RMSE, but still negative skill vs M0 |
| M2 +RS | Ridge/XGB worse than M0 | Mfinrs 4.253 | No evidence that deep RS fusion recovers a strong signal |
| M3 +Shipping | XGB 4.456 (CW vs M1 strong; skill < 0) | Mfinship 4.147 (skill +0.11%) | Clearest paired gain for representation-level shipping |
| M4 All | XGB 4.470 (skill < 0) | Mfull gated 4.205 / xattn 4.147 | Deep M4 can approach or slightly exceed M0 in selected specs; not uniformly dominant |

| 信息集 | 扁平对照 | 深度对照 | 方向性解读 |
|--------|----------|----------|------------|
| M1 金融 | Ridge 4.332 / XGB 4.771 | Mfin 4.250 | 深度金融改善扁平金融 RMSE，但相对 M0 仍为负 skill |
| M2 +遥感 | Ridge/XGB 差于 M0 | Mfinrs 4.253 | 无证据表明深度遥感融合恢复了强信号 |
| M3 +航运 | XGB 4.456（对 M1 的 CW 强；skill < 0） | Mfinship 4.147（skill +0.11%） | 表示级航运的配对收益最清晰 |
| M4 全模态 | XGB 4.470（skill < 0） | Mfull 门控 4.205 / 交叉注意力 4.147 | 选定设定下深度 M4 可接近或略超 M0；并非一致占优 |

Under strict non-nested DM tests, deep models do not uniformly and significantly dominate their flat counterparts across all information sets. The strongest and most coherent paired evidence appears in **shipping-inclusive** settings, where representation-level encoding of maritime structure helps more than flat concatenation of high-dimensional shipping columns. Finance+RS pairs remain weak in both architectures. Accordingly, the results support a cautious claim: deep models outperformed their flat counterparts in selected multimodal settings, particularly when shipping information was included — not a blanket statement that deep models consistently beat flat models.

在严格的非嵌套 DM 检验下，深度模型并未在所有信息集上一致且显著地主导对应扁平模型。最强且最连贯的配对证据出现在**包含航运**的设定中：对海运结构的表示级编码，比高维航运列的扁平拼接更有帮助。金融+遥感配对在两类架构中都偏弱。因此，结果支持审慎表述：深度模型在选定的多模态设定中优于对应扁平模型，尤其在纳入航运信息时——而不是笼统声称深度模型 consistently 优于扁平模型。

Two further readings follow from the paired table. First, architecture and information set interact: the same deep machinery that helps shipping does not rescue remote sensing. Second, comparing only the best deep model with the best flat model would overstate generality, because the deep advantage is concentrated in M3-type settings. Chapter 5 returns to whether this concentration is economically plausible.

由配对表还可得到两点进一步解读。第一，架构与信息集存在交互：有助于航运的同一套深度机制并不能挽救遥感。第二，若只比较最佳深度模型与最佳扁平模型，会夸大普遍性，因为深度优势集中在 M3 类设定。第 5 章将回到这种集中是否具有经济合理性。

## 4.5 Robustness and sensitivity analysis
## 4.5 稳健性与敏感性分析

Several checks probe whether the main patterns are artefacts of a single design choice.

若干检查用于检验主要模式是否由单一设计选择造成。

**Remote-sensing processing.** An MNDWI water mask at water-dominated export terminals strengthens the flat M2 XGBoost nested increment (CW p from 0.006 to 8.5×10⁻⁵), consistent with surface-water noise diluting optical indices. Leave-one-AOI-out runs indicate a diffuse rather than single-site RS contribution.

**遥感处理。** 在水面主导的出口码头施加 MNDWI 水体掩膜后，扁平 M2 XGBoost 的嵌套增量增强（CW p 由 0.006 至 8.5×10⁻⁵），与水面噪声稀释光学指数的判断一致。Leave-one-AOI-out 表明遥感贡献是弥散的，而非单站驱动。

**Lookback and capacity.** For deep finance+shipping, lookback 8 can improve point skill in a single-seed sweep, but lookback 4 remains the locked main specification for protocol alignment with flat models. Encoder width d = 64 is consistently worse than d = 32, warning against over-parameterisation on a short weekly sample.

**回看与容量。** 对深度金融+航运，单一种子扫描中 lookback 8 可改善点估计 skill，但主设定仍锁定 lookback 4，以与扁平模型协议对齐。编码器宽度 d = 64 一致差于 d = 32，警示在短周度样本上不宜过度参数化。

**Seeds and fusion mechanism.** Gated fusion is more stable across seeds than cross-attention. Cross-attention can achieve the best single-seed skill for shipping-inclusive models, but also exhibits large adverse seeds. Main conclusions therefore emphasise gated results, with cross-attention reported as a high-ceiling, higher-variance alternative.

**种子与融合机制。** 门控融合跨种子比交叉注意力更稳定。交叉注意力可在含航运模型上取得最佳单一种子 skill，但也出现大幅不利种子。主结论因此强调门控结果，交叉注意力作为“上限更高、方差更大”的方案报告。

**Missing-modality regularisation.** Moderate modality dropout can slightly help selected deep configurations, but does not overturn the ranking that shipping-inclusive models dominate finance+RS models.

**缺失模态正则。** 适度模态 dropout 可轻微帮助部分深度配置，但不会颠覆“含航运模型优于金融+遥感模型”的排序。

## 4.6 Interpretability results
## 4.6 可解释性结果

Primary interpretability focuses on models with the clearest predictive value. For deep models, this means shipping-inclusive specifications with non-negative or near-zero skill and economically coherent modality structure. Gate and attention diagnostics indicate elevated reliance on shipping representations relative to remote sensing, consistent with the performance tables.

主要可解释性分析聚焦预测价值最清晰的模型。对深度模型而言，即 skill 非负或接近零、且模态结构具有经济连贯性的含航运设定。门控与注意力诊断显示，相对遥感，模型更依赖航运表示，这与绩效表一致。

Supplementary SHAP is reported for flat M4/M3 XGBoost even though these models do not beat M0, because they show significant nested gains over M1. Global mean-|SHAP| on flat M4 attributes about **56%** of contribution to shipping, **31%** to finance and **13%** to remote sensing. This is interpreted narrowly: shipping helps explain why flat XGBoost improves on finance-only M1; it is not a claim of absolute forecast superiority over the random walk.

对扁平 M4/M3 XGBoost 报告补充性 SHAP，尽管它们未击败 M0，因为它们相对 M1 显示显著嵌套收益。扁平 M4 的全局 mean-|SHAP| 大约将 **56%** 贡献归于航运、**31%** 归于金融、**13%** 归于遥感。解释应狭义理解：航运有助于说明为何扁平 XGBoost 优于仅金融的 M1；这并非宣称其绝对预测优于随机游走。

Site and chokepoint attentions occasionally concentrate on major export and transit nodes (e.g. Hormuz/Suez-linked activity), but these patterns are treated as associations with model dependence, not as causal identification of price determinants.

站点与咽喉注意力偶尔集中于主要出口与中转节点（如与霍尔木兹/苏伊士相关的活动），但这些模式被视为与模型依赖相关的关联，而非对价格决定因素的因果识别。
