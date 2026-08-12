# Chapter 4 — Results（**1,200 words** ）

# 第 4 章 — 结果

## 4.1 Descriptive overview

## 4.1 描述性概览

This chapter reports out-of-sample one-week-ahead Brent forecasts on the common evaluation sample of 257 weeks (22 January 2021–19 December 2025). Performance is summarised by RMSE on reconstructed prices and by RMSE skill versus the no-change benchmark M0 (Murphy, 1988). Skill is positive when RMSE is lower than M0 and negative when it is higher. On this sample the M0 RMSE is 4.152 USD per barrel.

本章报告共同评价样本上提前一周的 Brent 样本外预测；样本为 257 周（2021 年 1 月 22 日至 2025 年 12 月 19 日）。表现以重构价格上的 RMSE，以及相对不变预测基准 M0 的 RMSE skill 概括（Murphy, 1988）。Skill 为正表示 RMSE 低于 M0，为负则表示更高。该样本上 M0 的 RMSE 为每桶 4.152 美元。

Weekly Brent log returns have near-zero mean and clear volatility clustering. Exploratory checks show only weak contemporaneous association between remote-sensing anomalies and returns. Shipping enters as a noisy proxy for trade and congestion, not as a direct measure of next week’s price.

周度 Brent 对数收益均值接近零并有明显波动聚集。探索性检查显示遥感异常与收益的同期关联偏弱。航运作为贸易与拥堵的有噪声代理进入，而非下周价格的直接度量。

## 4.2 Flat-model results



## 4.2 Flat 模型结果

Table 4.1 reports Flat out-of-sample performance for Ridge and XGBoost across the M0 benchmark and information sets S1–S4. Every learned Flat specification has negative skill versus M0, so the no-change forecast remains the best absolute-error benchmark in the Flat family.

表 4.1 报告 Ridge 与 XGBoost 在 M0 基准与信息集 S1–S4 上的 Flat 样本外表现。所有经学习的 Flat 设定相对 M0 均为负 skill，故在 Flat 族中不变预测仍是绝对误差意义上的最佳基准。

**Table 4.1 — Flat out-of-sample performance** *(n = 257)*

**表 4.1 — Flat 模型样本外表现** *（n = 257）*


| Set   | Variables                       | Ridge RMSE | Ridge skill vs M0 | XGB RMSE | XGB skill vs M0 |
| ----- | ------------------------------- | ---------- | ----------------- | -------- | --------------- |
| M0  | no-change benchmark               | 4.152      | —                 | 4.152    | —               |
| S1  | financial time series only        | 4.256      | −2.5%             | 4.368    | −5.2%           |
| S2  | financial time series + RS        | 4.414      | −6.3%             | 4.440    | −6.9%           |
| S3  | financial time series + shipping  | 4.430      | −6.7%             | 4.429    | −6.7%           |
| S4  | financial time series + RS + ship | 4.525      | −9.0%             | 4.507    | −8.6%           |


Finance-only S1 records the lowest Flat RMSE among learned sets (Ridge 4.256, −2.5%; XGBoost 4.368, −5.2%). Adding remote sensing (S2) or shipping (S3) raises RMSE relative to S1 under both learners. The full Flat set S4 is weakest (Ridge 4.525, −9.0%; XGBoost 4.507, −8.6%). Ridge and XGBoost agree: S1 is best among Flat learners, S4 is worst, and neither remote sensing nor shipping reduces absolute RMSE below the finance-only Flat baseline.

在学习到的 Flat 设定中，仅金融的 S1 的 RMSE 最低（Ridge 4.256，−2.5%；XGBoost 4.368，−5.2%）。加入遥感（S2）或航运（S3）后，两种学习器相对 S1 的 RMSE 均上升。全模态 Flat 设定 S4 最弱（Ridge 4.525，−9.0%；XGBoost 4.507，−8.6%）。Ridge 与 XGBoost 一致：Flat 中 S1 最好、S4 最差；遥感与航运均未能把绝对 RMSE 降到低于仅金融基线。

Under early feature fusion, noisy alternative-data proxies do not improve one-week-ahead Brent RMSE relative to M0 or to finance alone. For RQ1, Flat results therefore show no absolute out-of-sample gain from remote sensing or shipping.

在扁平早融合下，有噪声的另类数据代理未能相对 M0 或仅金融改善提前一周 Brent 的 RMSE。对 RQ1 而言，Flat 结果因此未显示遥感或航运的绝对样本外增益。

## 4.3 Deep-model results



## 4.3 Deep 模型结果

Table 4.2 reports Deep performance by information set. Gated fusion is the main Deep specification; cross-attention is a comparison where multimodal fusion applies. For S1 only the finance encoder is active. S1 and S2 both fail to beat M0 (gated RMSE 4.250 and 4.253; both −2.4% skill). Absolute error barely moves when remote sensing enters.

表 4.2 按信息集报告 Deep 表现。门控融合为主要 Deep 设定；交叉注意力为多模态融合处的对照。S1 仅金融编码器参与。S1 与 S2 均未优于 M0（门控 RMSE 4.250 与 4.253；skill 均为 −2.4%）。遥感进入后绝对误差几乎不动。

**Table 4.2 — Deep out-of-sample performance** *(gated = main specification)*

**表 4.2 — Deep 模型样本外表现** *（门控融合为主要设定）*


| Set   | Variables                       | Gated RMSE | Gated skill vs M0 | Cross-attn RMSE | Cross-attn skill vs M0 |
| ----- | ------------------------------- | ---------- | ----------------- | --------------- | ---------------------- |
| M0  | no-change benchmark               | 4.152      | —                 | 4.152           | —                      |
| S1  | financial time series only        | 4.250      | −2.4%             | —               | —                      |
| S2  | financial time series + RS        | 4.253      | −2.4%             | —               | —                      |
| S3  | financial time series + shipping  | 4.145      | +0.16%            | 4.110           | +1.01%                 |
| S4  | financial time series + RS + ship | 4.180      | −0.67%            | 4.138           | +0.33%                 |


Once shipping is included, gated S3 reduces RMSE to 4.145 (+0.16% skill). Cross-attention on the same set reaches 4.110 (+1.01%) on this reported seed. Shipping is the modality that moves Deep forecasts across the M0 line relative to Deep S1. Gated S4 rises again to 4.180 (−0.67%); cross-attention S4 is above M0 at +0.33% but does not displace gated S3 as the main finding. The gated margin is small and should not be over-read on a short weekly sample; Section 4.5 returns to seed sensitivity.

一旦纳入航运，门控 S3 将 RMSE 降至 4.145（+0.16% skill）。同一信息集上交叉注意力在此报告种子上达到 4.110（+1.01%）。相对 Deep S1，航运是使 Deep 预测越过 M0 的模态。门控 S4 回升至 4.180（−0.67%）；交叉注意力 S4 以 +0.33% 高于 M0，但不取代门控 S3 作为主发现。门控增益幅度很小，在较短周度样本上不宜过度解读；第 4.5 节回到种子敏感性。

For RQ1 under Deep, shipping-inclusive forecasts clear M0 by a modest margin, while remote sensing does not add a comparable absolute-error gain.

对 Deep 路径下的 RQ1 而言，含航运预测以小幅优势越过 M0，而遥感未提供可相比的绝对误差增益。

## 4.4 Flat versus Deep



## 4.4 Flat 与 Deep 的配对比较

**Table 4.3 — Paired Flat versus Deep**  
*(Flat = Table 4.1 XGBoost; Deep = Table 4.2 gated; percentages are skill versus M0)*

**表 4.3 — Flat 与 Deep 的配对比较**  
*（Flat = 表 4.1 XGBoost；Deep = 表 4.2 门控；百分比为相对 M0 的 skill）*


| Pair | Flat RMSE | Deep RMSE | Flat skill vs M0 | Deep skill vs M0 |
| ---- | --------- | --------- | ---------------- | ---------------- |
| S1   | 4.368     | 4.250     | −5.2%            | −2.4%            |
| S2   | 4.440     | 4.253     | −6.9%            | −2.4%            |
| S3   | 4.429     | 4.145     | −6.7%            | +0.16%           |
| S4   | 4.507     | 4.180     | −8.6%            | −0.67%           |


Deep has lower RMSE than Flat in every matched pair. Finance-only and finance-plus-RS pairs improve on Flat but remain negative versus M0. The decisive pair is S3: Flat skill −6.7% versus gated Deep +0.16%—the only matched pair in which Deep also beats M0. Deep S4 improves on Flat S4 but stays negative versus M0 and does not improve on Deep S3.

每一匹配配对中 Deep 的 RMSE 均低于 Flat。仅金融与金融加遥感相对 Flat 有改善，但相对 M0 仍为负。决定性配对是 S3：Flat skill −6.7%，门控 Deep +0.16%——唯一 Deep 同时优于 M0 的匹配对。Deep S4 优于 Flat S4，但相对 M0 仍为负，且未优于 Deep S3。

For RQ2, representation-level Deep modelling reduces RMSE relative to Flat at every matched set, but an M0-beating paired outcome appears only when shipping is included.

对 RQ2 而言，表示级 Deep 在每一匹配集上相对 Flat 降低 RMSE，但配对结果越过 M0 仅出现在含航运时。

## 4.5 Robustness and sensitivity



## 4.5 稳健性与敏感性

Appendix B collects the detailed robustness tables. Flat checks that vary lookback and feature settings produce no Flat specification that beats M0. Finance-only S1 remains the strongest Flat absolute-error baseline; remote sensing stays weak and is not driven by a single site. Nested Clark–West tests versus S1 in Appendix B detect incremental information over the financial baseline for some XGBoost shipping specifications, even when absolute RMSE remains higher than S1 and skill versus M0 remains negative. Shipping can therefore show a nested Flat signal without overturning Table 4.1’s absolute-error ranking.

附录 B 汇集详细稳健性表。改变回看与特征设定的 Flat 检查均未产生优于 M0 的 Flat 设定。仅金融 S1 仍是最强 Flat 绝对误差基线；遥感仍弱且非单站驱动。附录 B 中相对 S1 的嵌套 Clark–West 检验，在部分 XGBoost 航运设定上检出相对金融基线的增量信息，即便绝对 RMSE 仍高于 S1、相对 M0 的 skill 仍为负。因此航运可在 Flat 下显示嵌套信号，却不推翻表 4.1 的绝对误差排序。

Deep checks that vary random seeds and fusion choices leave gated finance-plus-shipping as the best configuration on average, but not a reliably positive one. Across seeds 42, 1 and 2 its mean skill is −0.50% (± 0.80), so the +0.16% in Table 4.2 is a seed-42 outcome rather than expected skill, and averaged over seeds no Deep configuration beats M0. Cross-attention can exceed gated fusion on one seed, as in Table 4.2 for S3, but is far more dispersed (−1.85% ± 2.80, with one seed at −5.01%). Larger encoder width than the main setting tends to worsen performance on the short weekly sample, as does halving encoder depth. The sub-period split is also less favourable: gated S3 is positive in the early window (+0.33%) but marginally negative in the late window (−0.13%), and no Deep configuration is positive in both. The small full-sample gain is therefore neither evenly distributed over time nor robust to reseeding, and both facts are reported as limitations rather than as further support. The matched Deep advantage over Flat, especially with shipping, survives these checks.

改变随机种子与融合方式的 Deep 检查中，门控金融加航运平均而言仍是最优配置，但并非稳定为正。在种子 42、1、2 上其平均 skill 为 −0.50%（± 0.80），故表 4.2 中的 +0.16% 是 seed=42 的结果而非期望 skill；跨种子平均后无任何 Deep 配置击败 M0。交叉注意力可在单一种子上超过门控（如表 4.2 的 S3），但离散度大得多（−1.85% ± 2.80，其中一个种子低至 −5.01%）。大于主设定的编码器宽度在短周度样本上往往恶化表现，将编码器层数减半亦然。子期划分同样不利：门控 S3 早窗为正（+0.33%），晚窗略为负（−0.13%），且无任何 Deep 配置在两窗均为正。因此这一小幅全样本增益既未在时间上均匀分布，也不稳健于重新设定种子；两点均作为局限报告，而非进一步的支持证据。匹配集上 Deep 相对 Flat 的优势——尤其含航运时——在这些检查下仍然成立。

These checks leave the RQ1–RQ2 rankings unchanged: Flat absolute gains remain absent; Deep’s small shipping-centred M0 clearance is the more stable positive case.

这些检查不改变 RQ1–RQ2 的排序：Flat 绝对增益仍缺位；Deep 以航运为中心的小幅越过 M0，仍是更稳定的正面情形。

## 4.6 Interpretability



## 4.6 可解释性

Interpretability is restricted to Deep specifications that improve on M0, principally Deep S3, using seeds 42, 1 and 2. Reported patterns are those that agree across seeds. Modality gates give each modality’s fusion-weight share; shipping node attention identifies which graph locations receive weight. A high shipping gate does not by itself mean the model focuses on a particular chokepoint; spatial detail is read from node attention.

可解释性仅限于相对 M0 有改善的 Deep 设定，主要为 Deep S3，使用种子 42、1 与 2。所报告的模式为跨种子一致者。模态门控给出各模态融合权重份额；航运节点注意力识别图中哪些位置获得较高权重。高航运门控本身不等于模型关注某一咽喉；空间细节由节点注意力读取。

For Deep S3, mean gates are about 0.61 (financial time series) and 0.39 (shipping). Week-level shipping-gate paths are unstable across seeds—pairwise correlations between the weekly paths range from −0.05 to 0.50—so single-seed event stories are not warranted. Among pre-specified event windows (±8 weeks), the Russia–Ukraine announcement window (February 2022) is the only one in which all three seeds move the shipping gate in the same direction, and there the gate falls rather than rises. The Red Sea window (November 2023) is mixed across seeds and is not retained. Spatially, the Strait of Hormuz carries the highest mean shipping-node attention and the best mean rank, but it enters the top-five set in only two of the three seeds, and no chokepoint is top-five in all three. Figure 4.1 summarises the Deep S3 gate and attention diagnostics; further panels are in Appendix B.

就 Deep S3 而言，门控均值约为金融时序 0.61、航运 0.39。航运门控周度路径跨种子不稳——各种子周度路径两两相关介于 −0.05 与 0.50 之间——故不宜讲单种子事件。预先设定事件窗（±8 周）中，仅 2022 年 2 月俄乌公告窗在三种子上同向移动，且方向为下降而非上升。2023 年 11 月红海窗跨种子方向不一，不保留。空间上，霍尔木兹的航运节点平均注意力与平均排名均居首，但仅在三个种子中的两个进入前五，且无任何咽喉在三种子中均进入前五。图 4.1 汇总 Deep S3 门控与注意力诊断；其余面板见附录 B。

Figure 4.1 — Deep S3 modality gates and shipping-node attention (multi-seed summary).

*[Figure 4.1 — Deep S3 interpretability: modality gates and shipping-node attention.]*

*[图 4.1 — Deep S3 可解释性：模态门控与航运节点注意力。]*

For RQ3, when Deep shipping-inclusive forecasts clear M0, the stable main-text reliance pattern is shared weight on finance and shipping, with Hormuz the highest-weighted network location but not a focus on which all seeds agree. These diagnostics describe model dependence after a stability filter; they do not identify causal drivers of Brent prices.

对 RQ3 而言，当含航运的 Deep 预测越过 M0 时，正文稳定的依赖模式是金融与航运共享权重；霍尔木兹是权重最高的网络位置，但并非所有种子一致认同的焦点。这些诊断描述稳定性过滤后的模型依赖，不识别 Brent 价格的因果驱动。