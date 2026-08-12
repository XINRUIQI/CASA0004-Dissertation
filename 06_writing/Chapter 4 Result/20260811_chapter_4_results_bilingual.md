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

Table 4.1 reports Flat out-of-sample performance for Ridge and XGBoost across M0–M4. Every learned Flat specification has negative skill versus M0, so the no-change forecast remains the best absolute-error benchmark in the Flat family.

表 4.1 报告 Ridge 与 XGBoost 在 M0–M4 上的 Flat 样本外表现。所有经学习的 Flat 设定相对 M0 均为负 skill，故在 Flat 族中不变预测仍是绝对误差意义上的最佳基准。

**Table 4.1 — Flat out-of-sample performance** *(n = 257)*

**表 4.1 — Flat 模型样本外表现** *（n = 257）*


| Model | Variables (feature set)         | Ridge RMSE | Ridge skill vs M0 | XGB RMSE | XGB skill vs M0 |
| ----- | ------------------------------- | ---------- | ----------------- | -------- | --------------- |
| M0  | no-change benchmark               | 4.152      | —                 | 4.152    | —               |
| M1  | financial time series only        | 4.256      | −2.5%             | 4.368    | −5.2%           |
| M2  | financial time series + RS        | 4.414      | −6.3%             | 4.440    | −6.9%           |
| M3  | financial time series + shipping  | 4.430      | −6.7%             | 4.429    | −6.7%           |
| M4  | financial time series + RS + ship | 4.525      | −9.0%             | 4.507    | −8.6%           |


Finance-only M1 records the lowest Flat RMSE among learned sets (Ridge 4.256, −2.5%; XGBoost 4.368, −5.2%). Adding remote sensing (M2) or shipping (M3) raises RMSE relative to M1 under both learners. The full Flat set M4 is weakest (Ridge 4.525, −9.0%; XGBoost 4.507, −8.6%). Ridge and XGBoost agree: M1 is best among Flat learners, M4 is worst, and neither remote sensing nor shipping reduces absolute RMSE below the finance-only Flat baseline.

在学习到的 Flat 设定中，仅金融的 M1 的 RMSE 最低（Ridge 4.256，−2.5%；XGBoost 4.368，−5.2%）。加入遥感（M2）或航运（M3）后，两种学习器相对 M1 的 RMSE 均上升。全模态 Flat 设定 M4 最弱（Ridge 4.525，−9.0%；XGBoost 4.507，−8.6%）。Ridge 与 XGBoost 一致：Flat 中 M1 最好、M4 最差；遥感与航运均未能把绝对 RMSE 降到低于仅金融基线。

Under early feature fusion, noisy alternative-data proxies do not improve one-week-ahead Brent RMSE relative to M0 or to finance alone. For RQ1, Flat results therefore show no absolute out-of-sample gain from remote sensing or shipping.

在扁平早融合下，有噪声的另类数据代理未能相对 M0 或仅金融改善提前一周 Brent 的 RMSE。对 RQ1 而言，Flat 结果因此未显示遥感或航运的绝对样本外增益。

## 4.3 Deep-model results



## 4.3 Deep 模型结果

Table 4.2 reports Deep performance by information set. Gated fusion is the main Deep specification; cross-attention is a comparison where multimodal fusion applies. For M1 only the finance encoder is active. M1 and M2 both fail to beat M0 (gated RMSE 4.250 and 4.253; both −2.4% skill). Absolute error barely moves when remote sensing enters.

表 4.2 按信息集报告 Deep 表现。门控融合为主要 Deep 设定；交叉注意力为多模态融合处的对照。M1 仅金融编码器参与。M1 与 M2 均未优于 M0（门控 RMSE 4.250 与 4.253；skill 均为 −2.4%）。遥感进入后绝对误差几乎不动。

**Table 4.2 — Deep out-of-sample performance** *(gated = main specification)*

**表 4.2 — Deep 模型样本外表现** *（门控融合为主要设定）*


| Model | Variables (feature set)         | Gated RMSE | Gated skill vs M0 | Cross-attn RMSE | Cross-attn skill vs M0 |
| ----- | ------------------------------- | ---------- | ----------------- | --------------- | ---------------------- |
| M0  | no-change benchmark               | 4.152      | —                 | 4.152           | —                      |
| M1  | financial time series only        | 4.250      | −2.4%             | —               | —                      |
| M2  | financial time series + RS        | 4.253      | −2.4%             | —               | —                      |
| M3  | financial time series + shipping  | 4.147      | +0.11%            | 4.121           | +0.74%                 |
| M4  | financial time series + RS + ship | 4.205      | −1.3%             | 4.147           | +0.12%                 |


Once shipping is included, gated M3 reduces RMSE to 4.147 (+0.11% skill). Cross-attention on the same set reaches 4.121 (+0.74%) on this reported seed. Shipping is the modality that moves Deep forecasts across the M0 line relative to Deep M1. Gated M4 rises again to 4.205 (−1.3%); cross-attention M4 is near M0 at +0.12% but does not displace gated M3 as the main finding. The gated margin is small and should not be over-read on a short weekly sample; Section 4.5 returns to seed sensitivity.

一旦纳入航运，门控 M3 将 RMSE 降至 4.147（+0.11% skill）。同一信息集上交叉注意力在此报告种子上达到 4.121（+0.74%）。相对 Deep M1，航运是使 Deep 预测越过 M0 的模态。门控 M4 回升至 4.205（−1.3%）；交叉注意力 M4 以 +0.12% 接近 M0，但不取代门控 M3 作为主发现。门控增益幅度很小，在较短周度样本上不宜过度解读；第 4.5 节回到种子敏感性。

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
| M1   | 4.368     | 4.250     | −5.2%            | −2.4%            |
| M2   | 4.440     | 4.253     | −6.9%            | −2.4%            |
| M3   | 4.429     | 4.147     | −6.7%            | +0.11%           |
| M4   | 4.507     | 4.205     | −8.6%            | −1.3%            |


Deep has lower RMSE than Flat in every matched pair. Finance-only and finance-plus-RS pairs improve on Flat but remain negative versus M0. The decisive pair is M3: Flat skill −6.7% versus gated Deep +0.11%—the only matched pair in which Deep also beats M0. Deep M4 improves on Flat M4 but stays negative versus M0 and does not improve on Deep M3.

每一匹配配对中 Deep 的 RMSE 均低于 Flat。仅金融与金融加遥感相对 Flat 有改善，但相对 M0 仍为负。决定性配对是 M3：Flat skill −6.7%，门控 Deep +0.11%——唯一 Deep 同时优于 M0 的匹配对。Deep M4 优于 Flat M4，但相对 M0 仍为负，且未优于 Deep M3。

For RQ2, representation-level Deep modelling reduces RMSE relative to Flat at every matched set, but an M0-beating paired outcome appears only when shipping is included.

对 RQ2 而言，表示级 Deep 在每一匹配集上相对 Flat 降低 RMSE，但配对结果越过 M0 仅出现在含航运时。

## 4.5 Robustness and sensitivity



## 4.5 稳健性与敏感性

Appendix B collects the detailed robustness tables. Flat checks that vary lookback and feature settings produce no Flat specification that beats M0. Finance-only M1 remains the strongest Flat absolute-error baseline; remote sensing stays weak and is not driven by a single site. Nested Clark–West tests versus M1 in Appendix B detect incremental information over the financial baseline for some XGBoost shipping specifications, even when absolute RMSE remains higher than M1 and skill versus M0 remains negative. Shipping can therefore show a nested Flat signal without overturning Table 4.1’s absolute-error ranking.

附录 B 汇集详细稳健性表。改变回看与特征设定的 Flat 检查均未产生优于 M0 的 Flat 设定。仅金融 M1 仍是最强 Flat 绝对误差基线；遥感仍弱且非单站驱动。附录 B 中相对 M1 的嵌套 Clark–West 检验，在部分 XGBoost 航运设定上检出相对金融基线的增量信息，即便绝对 RMSE 仍高于 M1、相对 M0 的 skill 仍为负。因此航运可在 Flat 下显示嵌套信号，却不推翻表 4.1 的绝对误差排序。

Deep checks that vary random seeds and fusion choices leave gated finance-plus-shipping as the more stable small positive-skill configuration. Cross-attention can exceed gated fusion on one seed, as in Table 4.2 for M3, but varies more across seeds. Larger encoder width than the main setting tends to worsen performance on the short weekly sample. Sub-period splits leave gated M3 positive in both early and late windows. The matched Deep advantage over Flat, especially with shipping, survives these checks.

改变随机种子与融合方式的 Deep 检查中，门控金融加航运仍是更稳定的小幅正 skill 设定。交叉注意力可在单一种子上超过门控（如表 4.2 的 M3），但跨种子波动更大。大于主设定的编码器宽度在短周度样本上往往恶化表现。子期划分下门控 M3 在早、晚窗均为正。匹配集上 Deep 相对 Flat 的优势——尤其含航运时——在这些检查下仍然成立。

These checks leave the RQ1–RQ2 rankings unchanged: Flat absolute gains remain absent; Deep’s small shipping-centred M0 clearance is the more stable positive case.

这些检查不改变 RQ1–RQ2 的排序：Flat 绝对增益仍缺位；Deep 以航运为中心的小幅越过 M0，仍是更稳定的正面情形。

## 4.6 Interpretability



## 4.6 可解释性

Interpretability is restricted to Deep specifications that improve on M0, principally Deep M3, using seeds 42, 1 and 2. Reported patterns are those that agree across seeds. Modality gates give each modality’s fusion-weight share; shipping node attention identifies which graph locations receive weight. A high shipping gate does not by itself mean the model focuses on a particular chokepoint; spatial detail is read from node attention.

可解释性仅限于相对 M0 有改善的 Deep 设定，主要为 Deep M3，使用种子 42、1 与 2。所报告的模式为跨种子一致者。模态门控给出各模态融合权重份额；航运节点注意力识别图中哪些位置获得较高权重。高航运门控本身不等于模型关注某一咽喉；空间细节由节点注意力读取。

For Deep M3, mean gates are about 0.56 (financial time series) and 0.44 (shipping). Week-level shipping-gate paths are unstable across seeds, so single-seed event stories are not warranted. Among pre-specified event windows (±8 weeks), only the Russia–Ukraine announcement window (February 2022) shows a shipping-gate rise across all three seeds. The Red Sea window (November 2023) rises in two seeds and falls in one, and is not retained. Spatially, the Strait of Hormuz is the only chokepoint in the top attention set for all three seeds. Figure 4.1 summarises the main Deep M3 gate and attention diagnostics; further panels are in Appendix B.

就 Deep M3 而言，门控均值约为金融时序 0.56、航运 0.44。航运门控周度路径跨种子不稳，故不宜讲单种子事件。预先设定事件窗（±8 周）中，仅 2022 年 2 月俄乌公告窗在三种子上同向上升。2023 年 11 月红海窗两升一降，不保留。空间上霍尔木兹是唯一三种子均进入注意力前列的咽喉。图 4.1 汇总 Deep M3 主要门控与注意力诊断；其余面板见附录 B。

Figure 4.1 — Deep M3 modality gates and shipping-node attention (multi-seed summary).

*[Figure 4.1 — Deep M3 interpretability: modality gates and shipping-node attention.]*

*[图 4.1 — Deep M3 可解释性：模态门控与航运节点注意力。]*

For RQ3, when Deep shipping-inclusive forecasts clear M0, the stable main-text reliance pattern is shared weight on finance and shipping, with Hormuz as the only cross-seed spatial focus. These diagnostics describe model dependence after a stability filter; they do not identify causal drivers of Brent prices.

对 RQ3 而言，当含航运的 Deep 预测越过 M0 时，正文稳定的依赖模式是金融与航运共享权重，且霍尔木兹为唯一跨种子空间焦点。这些诊断描述稳定性过滤后的模型依赖，不识别 Brent 价格的因果驱动。