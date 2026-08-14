# Chapter 4 — Results（**1,200 words** ）

# 第 4 章 — 结果

## 4.1 Descriptive overview

## 4.1 描述性概览

This chapter reports one-week-ahead Brent forecasts for 257 rolling forecast origins from 22 January 2021 to 19 December 2025. The corresponding target dates extend from 29 January 2021 to 26 December 2025. Models predict next-week log returns, which are converted to prices for evaluation. Performance is measured using  RMSE improvement vs M0, whose RMSE is 4.152 USD per barrel. Positive improvement indicates a lower RMSE than M0.

本章报告 257 个滚动预测起点上的提前一周 Brent 预测。预测起点为 2021 年 1 月 22 日至 2025 年 12 月 19 日，对应的目标日期为 2021 年 1 月 29 日至 2025 年 12 月 26 日。模型预测下一周的对数收益率，并将其转换为价格进行评价。模型表现采用 RMSE 及相对于 M0 的 衡量，M0 的 RMSE 为每桶 4.152 美元。正  表示 RMSE 低于 M0。 

Weekly returns are centred near zero, with large movements clustered in several periods. Remote-sensing anomalies show weak contemporaneous associations with returns. Their incremental predictive value is therefore assessed through the rolling out-of-sample comparisons below.

周度收益率接近零均值，较大幅度的变动集中在若干时期。遥感异常与收益率之间的同期关联较弱。因此，其增量预测价值将通过下文的滚动样本外比较进行评价。

## 4.2 Flat-model results



## 4.2 Flat 模型结果

Table 4.1 reports the out-of-sample performance of the Flat Ridge and XGBoost models across feature sets S1–S4, with M0 shown for comparison. All eight Flat models have higher RMSE than M0 and therefore record negative RMSE improvement.

表 4.1 报告 Flat Ridge 与 XGBoost 模型在特征集 S1–S4 上的样本外表现，并列出 M0 作为比较基准。八个已训练模型的 RMSE 均高于 M0，因此其 均为负值。

**Table 4.1 — Flat out-of-sample performance** *(n = 257)*

**表 4.1 — Flat 模型样本外表现** *（n = 257）*


| Set       | Variables                         | Model         | RMSE  | Improvement vs M0 (%) |
| --------- | --------------------------------- | ------------- | ----- | --------------------- |
| Benchmark |                                   | M0-Ridge      | 4.152 |                       |
|           |                                   | M0-XGB        | 4.152 |                       |
| S1        | financial time series             | M1-Flat-Ridge | 4.256 | −2.5%                 |
|           |                                   | M1-Flat-XGB   | 4.368 | −5.2%                 |
| S2        | financial time series + RS        | M2-Flat-Ridge | 4.414 | −6.3%                 |
|           |                                   | M2-Flat-XGB   | 4.440 | −6.9%                 |
| S3        | financial time series + shipping  | M3-Flat-Ridge | 4.553 | −9.7%                 |
|           |                                   | M3-Flat-XGB   | 4.357 | −4.9%                 |
| S4        | financial time series + RS + ship | M4-Flat-Ridge | 4.539 | −9.3%                 |
|           |                                   | M4-Flat-XGB   | 4.412 | −6.3%                 |


*Note:* Positive values indicate lower RMSE than M0.

For Ridge, S1 has the lowest RMSE and S3 the highest; adding remote sensing, shipping, or both raises RMSE relative to S1. For XGBoost, S3 records a slightly lower RMSE than S1 (4.357 versus 4.368), while S2 and S4 remain higher than S1. No Flat model records a positive RMSE improvement relative to M0. The Flat results therefore provide no descriptive evidence that alternative data improve forecasts relative to the no-change benchmark.

对 Ridge 而言，S1 的 RMSE 最低、S3 最高；加入遥感、航运或两者均抬高相对于 S1 的 RMSE。对 XGBoost 而言，S3 的 RMSE 略低于 S1（4.357 对 4.368），而 S2 与 S4 仍高于 S1。没有任何 Flat 模型取得相对于 M0 的正 RMSE improvement。因此，Flat 结果未提供另类数据相对不变基准改善预测的描述性证据。

## 4.3 Deep-model results



## 4.3 Deep 模型结果

Table 4.2 reports Deep-model performance across S1–S4. Gated fusion is the prespecified main specification, while cross-attention is reported as a secondary comparison where multimodal fusion applies. No cross-attention result is reported for S1 because only the finance encoder is active.

表 4.2 报告 Deep 模型在 S1–S4 上的表现。门控融合是预先设定的主要模型，交叉注意力则作为适用于多模态特征集的次要比较。S1 仅启用金融编码器，因此不报告交叉注意力结果。

**Table 4.2 — Deep out-of-sample performance** *(gated = main specification)*


| Set       | Variables                         | Model         | RMSE  | Improvement vs M0 (%) |
| --------- | --------------------------------- | ------------- | ----- | --------------------- |
| Benchmark |                                   | M0            | 4.152 |                       |
| S1        | financial time series             | M1-Deep       | 4.250 | −2.4%                 |
| S2        | financial time series + RS        | M2-Deep-Gated | 4.253 | −2.4%                 |
|           |                                   | M2-Deep-XAttn | 4.396 | −5.9%                 |
| S3        | financial time series + shipping  | M3-Deep-Gated | 4.146 | +0.15%                |
|           |                                   | M3-Deep-XAttn | 4.110 | +1.00%                |
| S4        | financial time series + RS + ship | M4-Deep-Gated | 4.180 | −0.67%                |
|           |                                   | M4-Deep-XAttn | 4.144 | +0.19%                |


The gated S1 and S2 models record similar RMSE of 4.250 and 4.253, both higher than that of M0. Adding remote sensing therefore provides no descriptive improvement. Neither reported fusion approach reduces RMSE relative to the finance-only Deep model at S2.

门控 S1 与 S2 模型的 RMSE 均高于 M0，分别为 4.250 和 4.253。两种融合方法均未能在 S2 上相对于仅金融的 Deep 模型降低 RMSE。  

With shipping included, gated S3 records the lowest RMSE among the gated models at 4.146, improving on M0 by 0.15%. Gated S4 rises to 4.180, 0.67% worse than M0, indicating that adding remote sensing to S3 does not provide a further improvement.

On the reported seed-42 run, cross-attention has a higher RMSE than gated fusion at S2, at 4.396 compared with 4.253, but lower RMSEs at S3 and S4. Cross-attention records RMSEs of 4.110 and 4.144 at S3 and S4, corresponding to RMSE improvements of 1.00% and 0.19%. These results are therefore reported as descriptive secondary comparisons rather than evidence that cross-attention is superior.  

在报告的随机种子 42 结果中，交叉注意力在 S2 上的 RMSE 高于门控融合，分别为 4.396 和 4.253，但在 S3 和 S4 上取得了更低的 RMSE。交叉注意力在 S3 和 S4 上的 RMSE 分别为 4.110 和 4.144，对应的正 分别为 1.00% 和 0.19%。然而，门控融合与交叉注意力之间的比较均未通过 Holm 校正。交叉注意力在 S3 和 S4 上相对于 M0 的正 也均不显著。因此，这些结果仅作为描述性的次要比较报告，而不构成交叉注意力更优的证据。

For RQ1, the within-learner comparisons show that shipping reduces RMSE relative to M0 only in the main Deep pathway. In the Flat family, XGBoost S3 is slightly below S1 but still worse than M0, and Ridge S3 is worse than S1. Remote sensing provides no improvement either alone or when added to shipping.

对于 RQ1，同一学习器内部的比较表明，航运数据仅在主要 Deep 路径中相对于 M0 降低了 RMSE。在 Flat 族中，XGBoost 的 S3 略低于 S1，但仍弱于 M0；Ridge 的 S3 弱于 S1。遥感无论单独加入还是在航运基础上加入均未带来改善。

## 4.4 Flat versus Deep



## 4.4 Flat 与 Deep 的配对比较

Table 4.3 compares the main Deep pathway with both Flat learners within each feature set. The feature-set category, forecast dates and evaluation sample are held constant. The main Deep pathway uses the finance-only Deep model at S1 and gated fusion at S2–S4. These comparisons evaluate the overall Flat and Deep modelling pathways rather than isolating the fusion operator alone.

表 4.3 在每个特征集内，将主要 Deep 路径分别与两种 Flat 学习器进行比较，并保持特征集类别、预测日期和评价样本一致。主要 Deep 路径在 S1 使用仅金融的 Deep 模型，在 S2–S4 使用门控融合。这些比较评价的是完整的 Flat 与 Deep 建模路径，而不是单独识别融合算子的作用。

**Table 4.3 — Matched Flat–Deep comparisons by feature set** *(n = 257)*


| Feature set | Flat model    | Flat RMSE | Main Deep model | Deep RMSE | **Deep vs Flat (%)** |
| ----------- | ------------- | --------- | --------------- | --------- | -------------------- |
| S1          | Ridge         | 4.256     | M1–Deep         | 4.250     | +0.15%               |
| S1          | M1–Flat–XGB   | 4.368     | M1–Deep         | 4.250     | +2.71%               |
| S2          | M2–Flat–Ridge | 4.414     | M2–Deep–Gated   | 4.253     | +3.64%               |
| S2          | M2–Flat–XGB   | 4.440     | M2–Deep–Gated   | 4.253     | +4.22%               |
| S3          | M3–Flat–Ridge | 4.553     | M3–Deep–Gated   | 4.146     | +8.95%               |
| S3          | M3–Flat–XGB   | 4.357     | M3–Deep–Gated   | 4.146     | +4.85%               |
| S4          | M4–Flat–Ridge | 4.539     | M4–Deep–Gated   | 4.180     | +7.90%               |
| S4          | M4–Flat–XGB   | 4.412     | M4–Deep–Gated   | 4.180     | +5.26%               |


*Note. Positive values indicate a lower Deep RMSE than the matched Flat model. Percentages use unrounded RMSEs.*

*注。Deep RMSE reduction 为 Deep 相对对应 Flat 模型的 RMSE 降幅，正值表示 Deep 更低。百分比由未四舍五入的 RMSE 计算。*

Figure 4.2

**Figure 4.2 — Paired slopes from Flat XGBoost to Deep gated fusion at each information set, with S3 highlighted.**

**图 4.2 — 各信息集上由 Flat XGBoost 到 Deep 门控融合的配对斜率，S3 高亮。**

Across all four feature sets, the main Deep model records lower RMSE than both Ridge and XGBoost. The reduction ranges from 0.15% against Ridge at S1 to 8.95% against Ridge at S3. The difference at S1 is therefore negligible, while larger reductions appear once alternative data are included.

在每个特征集上，主要 Deep 模型的 RMSE 均低于两种 Flat 学习器，因此所有匹配比较在描述层面均有利于 Deep 路径。降幅从 S1 相对于 Ridge 的 0.15% 到 S3 相对于 Ridge 的 8.95% 不等。因此，S1 上的差异可以忽略，而加入另类数据后，Deep 相对于 Flat 的 RMSE 降幅更大。

At S1 and S2, the main Deep models improve on both Flat learners but remain worse than M0. S3 is the only feature set which has lower RMSE than M0. Although the main Deep S4 model improves substantially over both Flat models, it remains worse than M0 and does not improve on the main Deep S3 model.

在 S1 和 S2 上，主要 Deep 模型均优于两种 Flat 学习器，但仍弱于 M0。S3 是唯一一个主要 Deep 模型的 RMSE 同时低于 M0 的特征集。尽管主要 Deep S4 模型相对于两种 Flat 模型均有明显改善，但其表现仍弱于 M0，也未优于主要 Deep S3 模型。

For RQ2, the matched comparisons show that the main Deep pathway records lower RMSE than both Flat learners at every feature set.

对于 RQ2，配对比较表明，主要 Deep 路径在每个特征集上的 RMSE 均低于两种 Flat 学习器。

## 4.5 Robustness and sensitivity



## 4.5 稳健性与敏感性

Appendix B collects the detailed robustness tables. Flat checks that vary lookback and feature settings produce no Flat specification that beats M0. Ridge’s finance-only S1 remains its strongest absolute-error specification; XGBoost’s lowest RMSE is S3, still above M0. Remote sensing stays weak and is not driven by a single site. Under the primary one-sided DM-HLN test, the seven shipping channel arms in Appendix B give p values against S1 between 0.384 and 0.727, with the AIS-only 113-column arm at 0.633, so no XGBoost shipping specification in that appendix is distinguishable from the financial baseline. Four arms—tanker-only, PortWatch-only, GFW-presence and GFW-aggregate—do reduce RMSE slightly relative to S1, which keeps the descriptive ordering from being uniformly against shipping, but no valid test supports a Flat nested shipping increment. Clark–West is not reported for XGBoost under the frozen test plan (Section 3.7.2); among the Ridge arms, where it is admissible, only GFW-aggregate falls below 5% (p = 0.032), and that is also the only Ridge arm whose RMSE sits below S1. Figure 4.3 shows the raw and Holm-adjusted p-values for the RQ1 and RQ2 families side by side.

附录 B 汇集详细稳健性表。改变回看与特征设定的 Flat 检查均未产生优于 M0 的 Flat 设定。Ridge 仍以仅金融 S1 为最强绝对误差设定；XGBoost 的最低 RMSE 为 S3，但仍高于 M0。遥感仍弱且非单站驱动。在主检验（单侧 DM–HLN）下，附录 B 七个航运通道臂相对 S1 的 p 值介于 0.384 与 0.727 之间，其中仅 AIS 的 113 列臂为 0.633，因此该附录中没有任何 XGBoost 航运设定能与金融基线区分开。有四个臂——仅油轮、仅 PortWatch、GFW 存在度与 GFW 聚合——的 RMSE 略低于 S1，使描述性排序不至于一致不利于航运，但没有任何有效检验支持 Flat 下的嵌套航运增量。按冻结检验方案（3.7.2 节），XGBoost 不报告 Clark–West；在可用该检验的 Ridge 各臂中，仅 GFW 聚合臂低于 5%（p = 0.032），而该臂也是唯一 RMSE 低于 S1 的 Ridge 臂。图 4.3 并列展示 RQ1 与 RQ2 两族的原始 p 值与 Holm 调整后 p 值。

Figure 4.3

**Figure 4.3 — Raw and Holm-adjusted DM-HLN p-values for the 15 RQ1 and 14 RQ2 comparisons, paired within each family.**

**图 4.3 — RQ1 族 15 项与 RQ2 族 14 项比较的 DM–HLN 原始 p 值与 Holm 调整后 p 值，族内配对显示。**

Deep checks that vary random seeds and fusion choices leave no configuration reliably positive. For gated finance-plus-shipping the mean skill across seeds 42, 1 and 2 is −0.51% (± 0.80), so the +0.16% in Table 4.2 is a seed-42 outcome rather than expected skill, and averaged over seeds no Deep configuration beats M0. Reseeding also reverses the fusion ranking outright. On seed 42 the S3 order is cross-attention (+1.00%) > gated (+0.15%) > concat (−0.22%); across the three seeds it becomes concat (−0.27% ± 0.35) > gated (−0.51% ± 0.80) > cross-attention (−3.01% ± 4.07), dispersion widening in the same order. The single-seed peak therefore belongs to the least stable operator: cross-attention posts the best figure in Table 4.2 and the worst seed-averaged skill of the three, collapsing to −7.14% on one seed. Gated is retained as the main specification because it is the fusion that exposes modality gates for the Section 4.6 analysis, not because it is more accurate than concat — the gap between their seed-averaged means is smaller than either configuration's own cross-seed spread, so the two are not separable on accuracy. Larger encoder width than the main setting tends to worsen performance on the short weekly sample, as does halving encoder depth. The sub-period split is also less favourable: gated S3 is positive in the early window (+0.33%) but marginally negative in the late window (−0.13%), and no Deep configuration is positive in both. The small full-sample gain is therefore neither evenly distributed over time nor robust to reseeding, and both facts are reported as limitations rather than as further support. The matched Deep advantage over Flat, especially with shipping, survives these checks.

改变随机种子与融合方式的 Deep 检查中，没有任何配置稳定为正。门控金融加航运在种子 42、1、2 上的平均 skill 为 −0.51%（± 0.80），故表 4.2 中的 +0.16% 是 seed=42 的结果而非期望 skill；跨种子平均后无任何 Deep 配置击败 M0。重新设定种子还彻底反转了融合排序：在 seed=42 上，S3 的排序为交叉注意力（+1.00%）> 门控（+0.15%）> 拼接（−0.22%）；跨三个种子则变为拼接（−0.27% ± 0.35）> 门控（−0.51% ± 0.80）> 交叉注意力（−3.01% ± 4.07），且离散度按同一顺序扩大。因此单种子峰值恰属最不稳定的算子：交叉注意力在表 4.2 中数值最优，跨种子平均 skill 却在三者中最差，并在其中一个种子上跌至 −7.14%。保留门控为主要设定，是因为它是能为 4.6 节分析产出模态门控的融合方式，而非因为它比拼接更准——两者跨种子均值之差小于任一配置自身的跨种子离散度，故在准确度上无法区分。大于主设定的编码器宽度在短周度样本上往往恶化表现，将编码器层数减半亦然。子期划分同样不利：门控 S3 早窗为正（+0.33%），晚窗略为负（−0.13%），且无任何 Deep 配置在两窗均为正。因此这一小幅全样本增益既未在时间上均匀分布，也不稳健于重新设定种子；两点均作为局限报告，而非进一步的支持证据。匹配集上 Deep 相对 Flat 的优势——尤其含航运时——在这些检查下仍然成立。

These checks leave the RQ1–RQ2 rankings unchanged: Flat absolute gains remain absent; Deep’s small shipping-centred M0 clearance is the more stable positive case. Figure 4.4 compares full-sample, early and late skill across both pathways.

这些检查不改变 RQ1–RQ2 的排序：Flat 绝对增益仍缺位；Deep 以航运为中心的小幅越过 M0，仍是更稳定的正面情形。图 4.4 比较两条路径在全样本、早期与晚期三个区间上的 skill。

Figure 4.4

**Figure 4.4 — RMSE skill versus M0 for the full sample, the early window (before 2023) and the late window (from 2023), shown separately for the Flat and Deep pathways.**

**图 4.4 — 全样本、早期窗（2023 年前）与晚期窗（2023 年起）相对 M0 的 RMSE skill，Flat 与 Deep 两条路径分面显示。**

## 4.6 Interpretability



## 4.6 可解释性

Interpretability is restricted to Deep specifications that improve on M0, principally Deep S3, using seeds 42, 1 and 2. Reported patterns are those that agree across seeds. Modality gates give each modality’s fusion-weight share; shipping node attention identifies which graph locations receive weight. A high shipping gate does not by itself mean the model focuses on a particular chokepoint; spatial detail is read from node attention.

可解释性仅限于相对 M0 有改善的 Deep 设定，主要为 Deep S3，使用种子 42、1 与 2。所报告的模式为跨种子一致者。模态门控给出各模态融合权重份额；航运节点注意力识别图中哪些位置获得较高权重。高航运门控本身不等于模型关注某一咽喉；空间细节由节点注意力读取。

For Deep S3, mean gates are about 0.61 (financial time series) and 0.39 (shipping). Week-level shipping-gate paths are unstable across seeds—pairwise correlations between the weekly paths range from −0.05 to 0.50—so single-seed event stories are not warranted. Among pre-specified event windows (±8 weeks), the Russia–Ukraine announcement window (February 2022) is the only one in which all three seeds move the shipping gate in the same direction, and there the gate falls rather than rises. The Red Sea window (November 2023) is mixed across seeds and is not retained. Spatially, the Strait of Hormuz carries the highest mean shipping-node attention and the best mean rank, but it enters the top-five set in only two of the three seeds, and no chokepoint is top-five in all three. Figure 4.5 reports the event-window gate shifts and Figure 4.6 the node-attention stability; the weekly gate paths behind them are in Appendix Figure B.1, and the remote-sensing site attention of gated S4, which does not meet the RQ3 admission rule, is in Appendix Figure B.2.

就 Deep S3 而言，门控均值约为金融时序 0.61、航运 0.39。航运门控周度路径跨种子不稳——各种子周度路径两两相关介于 −0.05 与 0.50 之间——故不宜讲单种子事件。预先设定事件窗（±8 周）中，仅 2022 年 2 月俄乌公告窗在三种子上同向移动，且方向为下降而非上升。2023 年 11 月红海窗跨种子方向不一，不保留。空间上，霍尔木兹的航运节点平均注意力与平均排名均居首，但仅在三个种子中的两个进入前五，且无任何咽喉在三种子中均进入前五。图 4.5 报告事件窗上的门控变化，图 4.6 报告节点注意力的稳定性；其背后的周度门控轨迹见附录图 B.1，未满足 RQ3 准入规则的门控 S4 的遥感站点注意力见附录图 B.2。

Figure 4.5

**Figure 4.5 — Change in the Deep S3 shipping gate around four pre-specified event windows, one point per seed.**

**图 4.5 — 四个预先设定事件窗附近 Deep S3 航运门控的变化，每个种子一个点。**

Figure 4.6

**Figure 4.6 — Left: mean shipping-node attention ±1 SD across seeds for gated S3, annotated with how many seeds place each node in the top five. Right: attention share divided by the uniform share for the three fusion mechanisms, showing that gated fusion is selective while cross-attention is close to uniform.**

**图 4.6 — 左：门控 S3 各航运节点跨种子的平均注意力 ±1 标准差，并标注该节点在几个种子中进入前五。右：三种融合机制的注意力份额除以均匀份额，显示门控融合具有选择性而交叉注意力接近均匀。**

For RQ3, when Deep shipping-inclusive forecasts clear M0, the stable main-text reliance pattern is shared weight on finance and shipping, with Hormuz the highest-weighted network location but not a focus on which all seeds agree. These diagnostics describe model dependence after a stability filter; they do not identify causal drivers of Brent prices.

对 RQ3 而言，当含航运的 Deep 预测越过 M0 时，正文稳定的依赖模式是金融与航运共享权重；霍尔木兹是权重最高的网络位置，但并非所有种子一致认同的焦点。这些诊断描述稳定性过滤后的模型依赖，不识别 Brent 价格的因果驱动。