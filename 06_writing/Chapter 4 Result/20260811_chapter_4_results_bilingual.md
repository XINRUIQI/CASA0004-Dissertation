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

在报告的随机种子 42 结果中，交叉注意力在 S2 上的 RMSE 高于门控融合，分别为 4.396 和 4.253，但在 S3 和 S4 上取得了更低的 RMSE。交叉注意力在 S3 和 S4 上的 RMSE 分别为 4.110 和 4.144，对应的正改善分别为 1.00% 和 0.19%。因此，这些结果仅作为描述性的次要比较报告，而不构成交叉注意力更优的证据。

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

## 4.5 Interpretability

Following the eligibility rule in Section 3.7.2, interpretation is reported for gated Deep S3, the only model with positive RMSE improvement relative to M0. Table 4.4 combines period-specific forecast performance, modality-gate weights and absolute SHAP attribution for the 257 forecast origins.

根据第 3.7.2 节的准入规则，可解释性分析针对门控 Deep S3 展开。该模型是主要 Deep 路径中唯一相对 M0 取得正 RMSE improvement 的另类数据模型。表 4.4 汇总 257 个预测起点上的分时期预测表现、模态门控权重和绝对 SHAP 归因。  

**Table 4.4 — Period-specific performance and attribution for gated Deep S3**

**表 4.4 — 门控 Deep S3 的分时期表现与归因**


| Period         | n   | RMSE  | Improvement vs M0 (%) | Gate finance | Gate shipping | SHAP finance | SHAP shipping |
| -------------- | --- | ----- | --------------------- | ------------ | ------------- | ------------ | ------------- |
| Full sample    | 257 | 4.146 | +0.15%                | 0.558        | 0.442         | 96.8%        | 3.2%          |
| 2021           | 50  | 3.014 | −1.98%                | 0.521        | 0.479         | 95.8%        | 4.2%          |
| 2022           | 52  | 6.613 | +0.77%                | 0.519        | 0.481         | 96.4%        | 3.6%          |
| 2023           | 52  | 3.790 | −0.37%                | 0.481        | 0.519         | 94.8%        | 5.2%          |
| 2024           | 52  | 3.083 | −0.79%                | 0.520        | 0.480         | 95.0%        | 5.0%          |
| 2025           | 51  | 2.960 | +0.99%                | 0.749        | 0.251         | 99.2%        | 0.8%          |
| Russia–Ukraine | 16  | 8.822 | +1.15%                | 0.536        | 0.464         | 96.5%        | 3.5%          |
| EU oil ban     | 16  | 5.059 | −0.21%                | 0.524        | 0.476         | 96.0%        | 4.0%          |
| OPEC+          | 16  | 4.358 | +0.76%                | 0.485        | 0.515         | 95.7%        | 4.3%          |
| Red Sea        | 16  | 3.265 | +2.55%                | 0.489        | 0.511         | 94.1%        | 5.9%          |


*Note. RMSE is evaluated in price levels, while SHAP attributes predicted log returns; gate weights and SHAP shares are not directly comparable. Event windows are ±8 weeks.*

*注。RMSE 按价格水平评价，SHAP 则归因于预测的对数收益率；门控权重与 SHAP 份额不可直接比较。事件窗口为事件日前后各 8 周。*

Financial inputs dominate absolute SHAP throughout the sample, accounting for 96.8% of full-sample attribution compared with 3.2% for shipping. Shipping attribution is relatively higher in 2023–2024 and during the Red Sea window, at between 5.0% and 5.9%, but falls to 0.8% in 2025. Its contribution is therefore small and episodic rather than persistently elevated.

金融输入在整个样本内始终主导绝对 SHAP，在完整样本中占 96.8%，航运仅占 3.2%。航运归因在 2023 至 2024 年及红海窗口内相对较高，介于 5.0%和 5.9%之间，但在 2025 年降至 0.8%。因此，航运贡献整体较小，只在部分时期暂时上升。

The main-run gate allocates average weights of 55.8% to finance and 44.2% to shipping, a substantially more balanced division than the SHAP attribution. This contrast reflects the difference between internal representation weighting and output attribution; SHAP is therefore used as the primary basis for interpreting RQ3.

主要运行中的门控权重平均向金融和航运表示分配 55.8%和 44.2%，明显比 SHAP 归因更加均衡。这种差异反映了内部表示加权与模型输出归因衡量的是不同内容。因此，RQ3 的主要解释依据是 SHAP。

At the input-group level, EIA variables provide the largest full-sample contribution at 43.6%, followed by financial and macroeconomic variables at 31.4%. All twenty highest-ranked individual features are financial inputs, led by crude production, Cushing stocks and the federal funds rate. No shipping subgroup contributes more than 2.0% in any reported period, although PortWatch and SAR become modestly more prominent during the Red Sea window.

在输入组层面，EIA 变量的完整样本贡献最大，为 43.6%，其次是金融与宏观变量的 31.4%。排名前二十的单项特征均为金融输入，其中原油产量、Cushing 库存和联邦基金利率排名最高。所有报告时期内，单个航运子组的贡献均未超过 2.0%，但 PortWatch 和 SAR 在红海窗口中相对更加突出。

Within the shipping representation, the highest full-sample node shares belong to Jurong, Hormuz, Suez, the Cape route and Bab el-Mandeb. Jurong and Hormuz lead the rankings from 2021 to 2023, while Suez, Bab el-Mandeb and the Cape route occupy the first three positions in 2024. During the Red Sea window, attribution is distributed across several locations, with no individual node accounting for more than 12% of shipping attribution. These conditional node shares do not represent their contribution to total model attribution.

在航运表示内部，完整样本节点份额最高的是裕廊、霍尔木兹、苏伊士、好望角航线和曼德海峡。2021 至 2023 年主要由裕廊和霍尔木兹领先，而 2024 年排名前三的节点转为苏伊士、曼德海峡和好望角航线。在红海窗口内，归因分布于多个地点，没有任何单一节点占航运归因的 12%以上。这些条件节点份额不代表其占模型总归因的比例。

For RQ3, the gated Deep S3 model relies predominantly on financial information across all reported periods. Shipping attribution remains small but varies over time, while its internal geographic focus changes across years rather than remaining concentrated on a single chokepoint.

对于 RQ3，门控 Deep S3 模型在所有报告时期均主要依赖金融信息。航运归因整体较小但随时间变化，其内部地域重点也在不同年份之间发生变化，而没有持续集中于某一个咽喉节点。

## 4.6 Robustness

## 4.6 稳健性

**Table 4.5 — Random-seed robustness of all Deep specifications** *(improvement vs M0, %)*

**表 4.5 — 全部 Deep 设定的随机种子稳健性** *（相对 M0 的 RMSE improvement，%）*


| Set | Model          | Main-run improvement | Across-run mean ± SD | Positive runs |
| --- | -------------- | -------------------- | -------------------- | ------------- |
| S1  | M1-Deep        | −2.36%               | −1.00% ± 1.33        | 1/3           |
| S2  | M2-Deep-Gated  | −2.43%               | −3.15% ± 1.67        | 0/3           |
|     | M2-Deep-Concat | −2.01%               | −1.79% ± 0.77        | 0/3           |
|     | M2-Deep-XAttn  | −5.87%               | −3.57% ± 2.77        | 0/3           |
| S3  | M3-Deep-Gated  | +0.15%               | −0.51% ± 0.80        | 1/3           |
|     | M3-Deep-Concat | −0.22%               | −0.27% ± 0.35        | 1/3           |
|     | M3-Deep-XAttn  | +1.00%               | −3.01% ± 4.07        | 1/3           |
| S4  | M4-Deep-Gated  | −0.68%               | −0.91% ± 0.26        | 0/3           |
|     | M4-Deep-Concat | −8.30%               | −3.79% ± 3.95        | 0/3           |
|     | M4-Deep-XAttn  | +0.19%               | −1.90% ± 2.75        | 1/3           |


*Note. The seed-42 column is the main reported run in Tables 4.2–4.4. S1 has no fusion operator.*

*注。seed 42 列为表 4.2–4.4 的主报告运行。S1 无融合算子。*

Table 4.5 reports the results of rerunning all Deep specifications with several random seeds. None records a positive mean RMSE improvement relative to M0, and only five of the thirty individual runs are positive. The best mean result is −0.27% for S3 concatenation, while the main gated S3 model records −0.51%. Gated S3 also outperforms S1 in only one of the three matched runs, and all S2 runs remain worse than M0. The positive results in the main run are therefore sensitive to random initialisation.

表 4.5 报告全部 Deep 模型在若干随机种子下的重复运行结果。所有模型在跨运行平均后均未取得相对于 M0 的正 RMSE improvement，三十次运行中也只有五次为正。S3 拼接模型的平均结果最好，但仍为 −0.27%；主要门控 S3 模型的平均结果为 −0.51%。门控 S3 也只在三次匹配运行中的一次优于 S1，且全部 S2 运行均弱于 M0。因此，主要运行中的正改善对随机初始化较为敏感。