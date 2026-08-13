# Chapter 4 — Results（**1,200 words** ）

# 第 4 章 — 结果

## 4.1 Descriptive overview

## 4.1 描述性概览

This chapter reports one-week-ahead Brent forecasts for 257 rolling origins from 22 January 2021 to 19 December 2025; the target dates extend from 29 January 2021 to 26 December 2025. Models predict log returns, which are converted to prices for evaluation. Performance is measured by RMSE and skill relative to M0, whose RMSE is 4.152 USD per barrel.

Weekly returns are centred near zero, with large movements clustered in several periods. Remote-sensing anomalies show weak contemporaneous associations with returns. Their incremental value is assessed through the rolling out-of-sample comparisons below.

## 4.2 Flat-model results

## 4.2 Flat 模型结果

Table 4.1 reports the out-of-sample performance of Ridge and XGBoost under M1–M4, with M0 shown for comparison. All learned specifications have higher RMSE than M0 and therefore negative skill.

**Table 4.1 — Flat out-of-sample performance** *(n = 257)*

**表 4.1 — Flat 模型样本外表现** *（n = 257）*


| Set | Variables                         | Ridge RMSE | Ridge skill vs M0 | XGB RMSE | XGB skill vs M0 |
| --- | --------------------------------- | ---------- | ----------------- | -------- | --------------- |
| M0  | no-change benchmark               | 4.152      | —                 | 4.152    | —               |
| S1  | financial time series only        | 4.256      | −2.5%             | 4.368    | −5.2%           |
| S2  | financial time series + RS        | 4.414      | −6.3%             | 4.440    | −6.9%           |
| S3  | financial time series + shipping  | 4.447      | −7.1%             | 4.408    | −6.2%           |
| S4  | financial time series + RS + ship | 4.536      | −9.3%             | 4.506    | −8.5%           |


*Note: Skill is measured relative to M0; positive values indicate lower RMSE.*

Among the learned specifications, M1 performs best for both Ridge (4.256) and XGBoost (4.368), whereas M4 performs worst. Adding remote sensing, shipping, or both does not reduce RMSE relative to finance alone. Thus, within the Flat early-fusion framework, alternative data provide neither an incremental RMSE reduction over M1 nor a lower RMSE than M0. Formal RQ1 inference is assessed using the prespecified DM–HLN tests with Holm adjustment.

## 4.3 Deep-model results

## 4.3 Deep 模型结果

Table 4.2 reports Deep performance by information set. Gated fusion is the main Deep specification; cross-attention is a comparison where multimodal fusion applies, so it has no entry at S1, where only the finance encoder is active. S1 and S2 both fail to beat M0 (gated RMSE 4.250 and 4.253; both −2.4% skill). Gated absolute error barely moves when remote sensing enters, and cross-attention at S2 is markedly worse (4.396, −5.9%), so no fusion mechanism recovers value from remote sensing alone.

表 4.2 按信息集报告 Deep 表现。门控融合为主要 Deep 设定；交叉注意力为多模态融合处的对照，故在仅金融编码器参与的 S1 上无对应数值。S1 与 S2 均未优于 M0（门控 RMSE 4.250 与 4.253；skill 均为 −2.4%）。遥感进入后门控的绝对误差几乎不动，而 S2 上的交叉注意力明显更差（4.396，−5.9%），故没有任何融合机制能单靠遥感取得价值。

**Table 4.2 — Deep out-of-sample performance** *(gated = main specification)*

**表 4.2 — Deep 模型样本外表现** *（门控融合为主要设定）*


| Set | Variables                         | Gated RMSE | Gated skill vs M0 | Cross-attn RMSE | Cross-attn skill vs M0 |
| --- | --------------------------------- | ---------- | ----------------- | --------------- | ---------------------- |
| M0  | no-change benchmark               | 4.152      | —                 | 4.152           | —                      |
| S1  | financial time series only        | 4.250      | −2.4%             | —               | —                      |
| S2  | financial time series + RS        | 4.253      | −2.4%             | 4.396           | −5.9%                  |
| S3  | financial time series + shipping  | 4.145      | +0.16%            | 4.110           | +1.00%                 |
| S4  | financial time series + RS + ship | 4.180      | −0.67%            | 4.144           | +0.19%                 |


Once shipping is included, gated S3 reduces RMSE to 4.145 (+0.16% skill). Cross-attention on the same set reaches 4.110 (+1.00%) on this reported seed. Shipping is the modality that moves Deep forecasts across the M0 line relative to Deep S1. Gated S4 rises again to 4.180 (−0.67%); cross-attention S4 is above M0 at +0.19% but does not displace gated S3 as the main finding. Both cross-attention entries are single-seed figures from the fusion matrix and do not survive reseeding — cross-attention has the worst seed-averaged skill of the three S3 fusions (Appendix B.4) — so they are reported as descriptive comparisons rather than as the better specification. The gated margin is likewise small and should not be over-read on a short weekly sample; Section 4.5 returns to seed sensitivity.

一旦纳入航运，门控 S3 将 RMSE 降至 4.145（+0.16% skill）。同一信息集上交叉注意力在此报告种子上达到 4.110（+1.00%）。相对 Deep S1，航运是使 Deep 预测越过 M0 的模态。门控 S4 回升至 4.180（−0.67%）；交叉注意力 S4 以 +0.19% 高于 M0，但不取代门控 S3 作为主发现。两个交叉注意力数值均来自融合矩阵的单一种子，且不耐重新设定种子——在三种 S3 融合中，交叉注意力的跨种子平均 skill 最差（见附录 B.4）——故仅作描述性对照报告，而非「更优设定」。门控的增益幅度同样很小，在较短周度样本上不宜过度解读；第 4.5 节回到种子敏感性。

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
| S3   | 4.408     | 4.145     | −6.2%            | +0.16%           |
| S4   | 4.506     | 4.180     | −8.5%            | −0.67%           |


Figure 4.2

**Figure 4.2 — Paired slopes from Flat XGBoost to Deep gated fusion at each information set, with S3 highlighted.**

**图 4.2 — 各信息集上由 Flat XGBoost 到 Deep 门控融合的配对斜率，S3 高亮。**

Deep has lower RMSE than Flat in every matched pair. Finance-only and finance-plus-RS pairs improve on Flat but remain negative versus M0. S3 is the pivotal pair descriptively: Flat skill −6.2% versus gated Deep +0.16%—the only matched pair in which Deep also beats M0. Deep S4 improves on Flat S4 but stays negative versus M0 and does not improve on Deep S3. The paired margin widens as the information set grows, from +2.71% at S1 against XGBoost to +7.23% at S4, and four of the eight matched rows have raw p below 5%, including S3 at 0.010 and S4 at 0.009. The smallest Holm-adjusted p in the fourteen-comparison RQ2 family is nevertheless 0.132, so the primary test supports no formal claim that the Deep pathway outperforms the Flat pathway. The ordering is consistent and repeated across all four sets, but it remains nominal evidence.

每一匹配配对中 Deep 的 RMSE 均低于 Flat。仅金融与金融加遥感相对 Flat 有改善，但相对 M0 仍为负。描述层面上的关键配对是 S3：Flat skill −6.2%，门控 Deep +0.16%——唯一 Deep 同时优于 M0 的匹配对。Deep S4 优于 Flat S4，但相对 M0 仍为负，且未优于 Deep S3。配对增益随信息集扩大而变大，从 S1 相对 XGBoost 的 +2.71% 增至 S4 的 +7.23%，八个匹配对中有四个原始 p 值低于 5%，其中 S3 为 0.010、S4 为 0.009。但 14 项 RQ2 族中最小的 Holm 调整后 p 值为 0.132，故主检验不支持「Deep 路径优于 Flat 路径」的正式论断。该排序在四个信息集上一致且重复出现，但仍属名义证据。

For RQ2, representation-level Deep modelling reduces RMSE relative to Flat at every matched set, but an M0-beating paired outcome appears only when shipping is included, and no paired comparison survives Holm adjustment.

对 RQ2 而言，表示级 Deep 在每一匹配集上相对 Flat 降低 RMSE，但配对结果越过 M0 仅出现在含航运时，且没有任何配对比较能通过 Holm 调整。

## 4.5 Robustness and sensitivity

## 4.5 稳健性与敏感性

Appendix B collects the detailed robustness tables. Flat checks that vary lookback and feature settings produce no Flat specification that beats M0. Finance-only S1 remains the strongest Flat absolute-error baseline; remote sensing stays weak and is not driven by a single site. Under the primary one-sided DM-HLN test, the seven shipping channel arms in Appendix B give p values against S1 between 0.384 and 0.727, with the main 113-column arm at 0.633, so no XGBoost shipping specification is distinguishable from the financial baseline. Four arms—tanker-only, PortWatch-only, GFW-presence and GFW-aggregate—do reduce RMSE slightly relative to S1, which keeps the descriptive ordering from being uniformly against shipping, but no valid test supports a Flat nested shipping increment. Clark–West is not reported for XGBoost under the frozen test plan (Section 3.7.2); among the Ridge arms, where it is admissible, only GFW-aggregate falls below 5% (p = 0.032), and that is also the only Ridge arm whose RMSE sits below S1. Figure 4.3 shows the raw and Holm-adjusted p-values for the RQ1 and RQ2 families side by side.

附录 B 汇集详细稳健性表。改变回看与特征设定的 Flat 检查均未产生优于 M0 的 Flat 设定。仅金融 S1 仍是最强 Flat 绝对误差基线；遥感仍弱且非单站驱动。在主检验（单侧 DM–HLN）下，附录 B 七个航运通道臂相对 S1 的 p 值介于 0.384 与 0.727 之间，作为主设定的 113 列臂为 0.633，因此没有任何 XGBoost 航运设定能与金融基线区分开。有四个臂——仅油轮、仅 PortWatch、GFW 存在度与 GFW 聚合——的 RMSE 略低于 S1，使描述性排序不至于一致不利于航运，但没有任何有效检验支持 Flat 下的嵌套航运增量。按冻结检验方案（3.7.2 节），XGBoost 不报告 Clark–West；在可用该检验的 Ridge 各臂中，仅 GFW 聚合臂低于 5%（p = 0.032），而该臂也是唯一 RMSE 低于 S1 的 Ridge 臂。图 4.3 并列展示 RQ1 与 RQ2 两族的原始 p 值与 Holm 调整后 p 值。

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