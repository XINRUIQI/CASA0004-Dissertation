# Chapter 4 — Results *(~2,500–3,200)*

# 第 4 章 — 结果 *(约 2,500–3,200 词)*

## 4.1 Descriptive overview

## 4.1 描述性概览

This chapter reports out-of-sample one-week-ahead Brent price forecasts on the common scored span in Section 3.9 (257 weeks, 22 January 2021–19 December 2025). Relative performance versus the no-change benchmark M0 is summarised by RMSE skill (Murphy, 1988), defined in Section 3.10 as

\[
\mathrm{Skill}=100\times\left(1-\frac{\mathrm{RMSE}_{\mathrm{model}}}{\mathrm{RMSE}_{\mathrm{M0}}}\right).
\]

Positive skill means lower RMSE than M0; negative skill means worse. On this span M0 RMSE is 4.152 USD/barrel. Weekly Brent log returns have near-zero mean and clear volatility clustering. Exploratory remote-sensing anomalies show only weak contemporaneous association with returns, and several stronger associations occur at non-positive leads. Shipping is treated as a noisy trade-and-congestion proxy, not as a direct measure of next week’s price. The chapter follows Flat results (RQ1), Deep results, paired Flat–Deep comparisons (RQ2), robustness, and interpretability where predictive value exists (RQ3).

本章在第 3.9 节共同计分跨度上报告样本外提前一周 Brent 价格预测（257 周，2021 年 1 月 22 日至 2025 年 12 月 19 日）。相对不变预测基准 M0 的表现以 RMSE skill 汇总（Murphy, 1988），定义见第 3.10 节：

\[
\mathrm{Skill}=100\times\left(1-\frac{\mathrm{RMSE}_{\mathrm{model}}}{\mathrm{RMSE}_{\mathrm{M0}}}\right).
\]

正 skill 表示 RMSE 低于 M0；负 skill 表示更差。该跨度上 M0 的 RMSE 为每桶 4.152 美元。周度 Brent 对数收益均值接近零并有明显波动聚集。探索性遥感异常与收益的同期关联偏弱，若干更强关联出现在非正领先期。航运视为有噪声的贸易与拥堵代理，而非下周价格的直接量测。本章依次报告 Flat 结果（RQ1）、Deep 结果、配对 Flat–Deep（RQ2）、稳健性，以及在已有预测价值处的可解释性（RQ3）。

## 4.2 Flat-model results

## 4.2 Flat 模型结果

Table 4.1 summarises Flat performance. Every learned Flat model has negative skill versus M0. Relative to financial time series M1, adding shipping in M3 still improves performance, while remote sensing in M2 does not show a clear gain over M1; the full Flat set M4 raises absolute RMSE further.

表 4.1 汇总 Flat 表现。每个学习到的 Flat 模型相对 M0 均为负 skill。相对金融时序 M1，M3 加入航运仍可改善表现，而 M2 的遥感相对 M1 无明显增益；全模态 Flat 集 M4 进一步提高绝对 RMSE。

**Table 4.1 — Flat out-of-sample performance** *(n = 257)*

**表 4.1 — Flat 样本外表现** *（n = 257）*

| Set | Content                           | Ridge RMSE | Ridge skill vs M0 | XGB RMSE | XGB skill vs M0 |
| --- | --------------------------------- | ---------- | ----------------- | -------- | --------------- |
| M0  | no-change benchmark               | 4.152      | —                 | 4.152    | —               |
| M1  | financial time series only        | 4.256      | −2.5%             | 4.368    | −5.2%           |
| M2  | financial time series + RS        | 4.414      | −6.3%             | 4.440    | −6.9%           |
| M3  | financial time series + shipping  | 4.430      | −6.7%             | 4.429    | −6.7%           |
| M4  | financial time series + RS + ship | 4.525      | −9.0%             | 4.507    | −8.6%           |

Nested Clark–West tests versus M1 can detect incremental information from added modalities under selected learners, especially when shipping enters. Diebold–Mariano tests against M0 remain consistent with negative skill: a nested gain over financial time series is not evidence of beating M0. Directional accuracy is auxiliary and does not reverse the RMSE ranking.

相对 M1 的嵌套 Clark–West 检验可在选定学习器下检出新增模态的增量信息，尤其在航运进入时。相对 M0 的 Diebold–Mariano 检验与负 skill 一致：相对金融时序的嵌套增益不等于打过 M0。方向准确率仅为辅助，不能推翻 RMSE 排序。

## 4.3 Deep-model results

## 4.3 Deep 模型结果

Table 4.2 summarises Deep performance by information set. For M1 only the finance encoder applies; its result is placed in the gated column for comparability. M1 and M2 both fail to beat M0. Financial time series plus shipping (M3) improves on M0 under gated and cross-attention fusion, with modest skill of about +0.11% (gated) and +0.74% (cross-attention). Nested Deep contrasts likewise identify shipping as the clearest modality gain over Deep M1. Gated M4 does not clearly dominate M3: adding remote sensing on top often fails to cut absolute error further.

表 4.2 按信息集汇总 Deep 表现。M1 仅启用金融编码器，结果放在门控列以便可比。M1 与 M2 均未打过 M0。金融时序加航运（M3）在门控与交叉注意力下均相对 M0 有改善，skill 约 +0.11%（门控）与 +0.74%（交叉注意力）。Deep 内部嵌套对照同样显示航运相对 Deep M1 的模态增益最清晰。门控 M4 并不明显主导 M3：再加遥感往往不能进一步降低绝对误差。

Gated fusion is the main reported Deep design; cross-attention is a comparative architecture with a higher single-seed ceiling but greater sensitivity (Section 4.5). Encoder-concatenation and the full fusion matrix are in the appendix.

门控融合为主报告 Deep 设计；交叉注意力为对照架构，单一种子上限更高但更敏感（第 4.5 节）。编码器拼接与完整融合矩阵见附录。

**Table 4.2 — Deep out-of-sample performance** *(gated = main reported fusion)*

**表 4.2 — Deep 样本外表现** *（gated = 主报告融合）*

| Set | Content                           | Gated RMSE | Gated skill vs M0 | Xattn RMSE | Xattn skill vs M0 |
| --- | --------------------------------- | ---------- | ----------------- | ---------- | ----------------- |
| M0  | no-change benchmark               | 4.152      | —                 | 4.152      | —                 |
| M1  | financial time series only        | 4.250      | −2.4%             | —          | —                 |
| M2  | financial time series + RS        | 4.253      | −2.4%             | —          | —                 |
| M3  | financial time series + shipping  | 4.147      | **+0.11%**        | 4.121      | **+0.74%**        |
| M4  | financial time series + RS + ship | 4.205      | −1.3%             | 4.147      | +0.12%            |

## 4.4 Flat versus Deep

## 4.4 Flat 对比 Deep

Table 4.3 compares Flat and Deep at matched information sets for RQ2. The percentage columns are each model’s skill versus M0, not the Flat-to-Deep RMSE change.

表 4.3 在匹配信息集上比较 Flat 与 Deep，服务 RQ2。百分比列为各模型相对 M0 的 skill，不是 Flat 到 Deep 的 RMSE 变化率。

Deep gains are clearest once shipping enters. Deep M1 improves on Flat M1 in RMSE but remains negative versus M0. Finance-plus-RS pairs stay weak in both families. The clearest paired gain is M3: Flat skill −6.7% versus Deep gated +0.11%. Deep M4 has lower RMSE than Flat M4 but neither gated Deep M4 nor the Flat counterpart beats M0, and gated Deep M4 does not dominate Deep M3. The Deep advantage is therefore conditional on shipping-inclusive settings rather than uniform across all pairs.

Deep 增益在航运进入后最清晰。Deep M1 在 RMSE 上优于 Flat M1，但相对 M0 仍为负。金融加遥感配对在两族都偏弱。最清晰配对增益是 M3：Flat skill −6.7%，Deep 门控 +0.11%。Deep M4 的 RMSE 低于 Flat M4，但门控 Deep M4 与 Flat 对照均未打过 M0，且门控 Deep M4 不主导 Deep M3。因此 Deep 优势取决于含航运设定，而非所有配对一律成立。

**Table 4.3 — Paired Flat versus Deep**

**表 4.3 — 配对 Flat 对比 Deep**

| Pair | Flat RMSE | Deep RMSE | Flat skill vs M0 | Deep skill vs M0 |
| ---- | --------- | --------- | ---------------- | ---------------- |
| M1   | 4.368     | 4.250     | −5.2%            | −2.4%            |
| M2   | 4.440     | 4.253     | −6.9%            | −2.4%            |
| M3   | 4.429     | 4.147     | −6.7%            | +0.11%           |
| M4   | 4.507     | 4.205     | −8.6%            | −1.3%            |

## 4.5 Robustness and sensitivity

## 4.5 稳健性与敏感性

Appendix B collects the full robustness tables. Flat checks (lookback, remote-sensing variants, shipping tiers, leave-one-modality-out for M4) leave the main ranking unchanged: no Flat model beats M0; shipping still helps relative to M1; remote sensing remains weak and any nested Flat RS signal is diffuse rather than single-site driven.

附录 B 汇总完整稳健性表。Flat 检查（回看、遥感变体、航运层级、M4 的 leave-one-modality-out）不改变主排序：无 Flat 模型打过 M0；航运相对 M1 仍有帮助；遥感仍弱，且任何嵌套 Flat 遥感信号偏弥散而非单站驱动。

Deep checks cover seeds, lookback, representation size, fusion type, and early versus late windows. Gated finance-plus-shipping remains the more stable small positive-skill configuration; cross-attention can look stronger on one seed but varies more across seeds. Larger encoder width than the locked setting tends to worsen performance on the short weekly sample. The matched-set Deep advantage over Flat, especially with shipping, survives these checks.

Deep 检查覆盖种子、回看、表示维度、融合类型与早/晚窗。门控金融加航运仍是更稳定的小幅正 skill 配置；交叉注意力可在单一种子更强，但跨种子波动更大。大于锁定设定的编码器宽度在短周度样本上往往恶化表现。匹配集上 Deep 相对 Flat 的优势——尤其含航运时——在这些检查下仍然成立。

## 4.6 Interpretability

## 4.6 可解释性

Interpretability is restricted to Deep specifications that improve on M0—primarily Deep M3. Claims follow a multi-seed rule (seeds 42, 1 and 2): only cross-seed-stable foci are locked in the main text. Modality gates give each modality’s fusion-weight share; shipping node attention identifies which graph locations receive weight. A high shipping gate does not mean the model is “looking at Hormuz”; spatial detail lives in node attention.

可解释性仅限于相对 M0 有改善的 Deep 设定——主要为 Deep M3。宣称遵循多种子规则（种子 42、1、2）：主文只锁定跨 seed 稳定焦点。模态门控给出各模态融合权重份额；航运节点注意力识别图中哪些位置获权。高航运门控不等于模型“在看霍尔木兹”；空间细节在节点注意力层。

For Deep M3, mean gates are about 0.56 (financial time series) and 0.44 (shipping). Week-level shipping-gate paths are unstable across seeds, so fine-grained single-seed event stories are not warranted. Event-window checks (±8 weeks) retain only the Russia–Ukraine announcement window (February 2022) as a cross-seed co-rising case. EU oil-ban and OPEC+ cut windows co-move but shipping weight falls; the Houthi Red Sea window (November 2023) is unstable (2↑1↓) and is not locked. Spatially, Hormuz is the only chokepoint in the top set for all three seeds (3/3). Supporting figures are in Appendix B.

对 Deep M3，门控均值约为金融时序 0.56、航运 0.44。航运门控周度路径跨 seed 不稳，故不宜按单种子讲细事件。事件窗（±8 周）仅保留 2022 年 2 月俄乌公告窗为跨 seed 同向上升；欧盟禁运与 OPEC+ 减产窗同向但航运权重下降；2023 年 11 月红海窗不稳（两升一降）不写死。空间上霍尔木兹是唯一 3/3 进入前列的咽喉。支撑图见附录 B。

These diagnostics describe model dependence under a stability filter. They do not identify causal drivers of Brent prices.

这些诊断是在稳定性过滤下的模型依赖描述，不识别 Brent 价格的因果驱动。
