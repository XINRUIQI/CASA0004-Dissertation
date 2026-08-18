# Chapter 4 — Results（**1,200 words** ）

# 第 4 章 — 结果

## 4.2 Flat-model results

## 4.2 Flat 模型结果

Table 4.1 reports the out-of-sample performance of the Flat Ridge and XGBoost models across feature sets S1–S4, with M0 shown for comparison. All eight Flat models have higher RMSE than M0 and therefore record negative RMSE improvement.

表 4.1 报告 Flat Ridge 与 XGBoost 模型在特征集 S1–S4 上的样本外表现，并列出 M0 作为比较基准。八个已训练模型的 RMSE 均高于 M0，因此其改善均为负值。

**Table 4.1 — Flat out-of-sample performance** *(n = 257)*

**表 4.1 — Flat 模型样本外表现** *（n = 257）*


| Set       | Variables                         | Model         | RMSE  | Improvement vs M0 (%) |
| --------- | --------------------------------- | ------------- | ----- | --------------------- |
| Benchmark |                                   | M0            | 4.152 |                       |
| S1        | financial time series             | M1-Flat-Ridge | 4.256 | −2.5%                 |
|           |                                   | M1-Flat-XGB   | 4.368 | −5.2%                 |
| S2        | financial time series + RS        | M2-Flat-Ridge | 4.414 | −6.3%                 |
|           |                                   | M2-Flat-XGB   | 4.440 | −6.9%                 |
| S3        | financial time series + shipping  | M3-Flat-Ridge | 4.553 | −9.7%                 |
|           |                                   | M3-Flat-XGB   | 4.357 | −4.9%                 |
| S4        | financial time series + RS + shiping | M4-Flat-Ridge | 4.539 | −9.3%                 |
|           |                                   | M4-Flat-XGB   | 4.412 | −6.3%                 |


*Note:* Positive values indicate lower RMSE than M0.

For Ridge, S1 has the lowest RMSE and S3 the highest; adding remote sensing, shipping, or both raises RMSE relative to S1. For XGBoost, S3 records a slightly lower RMSE than S1 (4.357 versus 4.368), while S2 and S4 remain higher than S1. No Flat model records a positive RMSE improvement relative to M0. Overall, the Flat family performs worse than the no-change benchmark across all information sets. 

对 Ridge 而言，S1 的 RMSE 最低、S3 最高；加入遥感、航运或两者均抬高相对于 S1 的 RMSE。对 XGBoost 而言，S3 的 RMSE 略低于 S1（4.357 对 4.368），而 S2 与 S4 仍高于 S1。没有任何 Flat 模型取得相对于 M0 的正 RMSE improvement。总体来看，Flat 模型族在所有信息集上的表现均差于不变预测基准。

The Flat results therefore provide no evidence of improvement relative to the no-change benchmark. Remote sensing consistently reduces performance for both Ridge and XGBoost. Shipping slightly improves XGBoost relative to S1 but worsens Ridge performance, and in neither case is the improvement sufficient to outperform M0.

因此，Flat 结果未提供相对不变基准有所改善的证据。遥感数据在 Ridge 和 XGBoost 中均使预测表现下降。航运数据相对于 S1 略微改善了 XGBoost，但明显降低了 Ridge 的表现，而且两种情况下都不足以优于 M0。

## 4.3 Deep-model results

## 4.3 Deep 模型结果

Table 4.2 reports Deep-model performance across S1–S4. Gated fusion is the prespecified main specification, while cross-attention is reported as a secondary comparison. No cross-attention result is reported for S1 because only the finance encoder is active.

表 4.2 报告 Deep 模型在 S1–S4 上的表现。门控融合是预先设定的主要模型，交叉注意力则作为次要比较。S1 仅启用金融编码器，因此不报告交叉注意力结果。

**Table 4.2 — Deep out-of-sample performance** *(gated = main specification)*【备注：说一下表格里的XAttn 的全称】


| Set       | Variables                         | Model         | RMSE  | Improvement vs M0 (%) |
| --------- | --------------------------------- | ------------- | ----- | --------------------- |
| Benchmark |                                   | M0            | 4.152 |                       |
| S1        | financial time series             | M1-Deep       | 4.250 | −2.4%                 |
| S2        | financial time series + RS        | M2-Deep-Gated | 4.253 | −2.4%                 |
|           |                                   | M2-Deep-XAttn | 4.396 | −5.9%                 |
| S3        | financial time series + shipping  | M3-Deep-Gated | 4.146 | +0.15%                |
|           |                                   | M3-Deep-XAttn | 4.110 | +1.00%                |
| S4        | financial time series + RS + shiping | M4-Deep-Gated | 4.180 | −0.67%                |
|           |                                   | M4-Deep-XAttn | 4.144 | +0.19%                |


The gated S1 and S2 models record similar RMSE of 4.250 and 4.253, both higher than that of M0. Adding remote sensing therefore provides no descriptive improvement. Neither reported fusion approach reduces RMSE relative to the finance-only Deep model at S2. With shipping included, gated S3 records the lowest RMSE among the gated models at 4.146, improving on M0 by 0.15%. Gated S4 rises to 4.180, 0.67% worse than M0, indicating that adding remote sensing to S3 does not provide a further improvement.

门控 S1 与 S2 模型的 RMSE 均高于 M0，分别为 4.250 和 4.253。两种融合方法均未能在 S2 上相对于仅金融的 Deep 模型降低 RMSE。加入航运数据后，门控融合的 S3 在各门控模型中取得最低 RMSE，为 4.146，相比 M0 改善了 0.15%。门控融合的 S4 的 RMSE 上升至 4.180，比 M0 高 0.67%，表明在 S3 的基础上进一步加入遥感数据并未带来额外改善。

On the reported seed-42 run, cross-attention has a higher RMSE than gated fusion at S2, at 4.396 compared with 4.253, but lower RMSEs at S3 and S4. Cross-attention records RMSEs of 4.110 and 4.144 at S3 and S4, corresponding to RMSE improvements of 1.00% and 0.19%. These results are therefore reported as descriptive secondary comparisons rather than evidence that cross-attention is superior.  

On the reported seed-42 run, cross-attention has a higher RMSE than gated fusion at S2, at 4.396 compared with 4.253, but lower RMSEs at S3 and S4. Cross-attention records RMSEs of 4.110 and 4.144 at S3 and S4, corresponding to RMSE improvements of 1.00% and 0.19%. However, none of the comparisons between gated fusion and cross-attention passes the Holm correction. The positive improvements of cross-attention over M0 at S3 and S4 are also not statistically significant. These results are therefore reported only as descriptive secondary comparisons and do not provide evidence that cross-attention is superior.

在报告的随机种子 42 结果中，交叉注意力在 S2 上的 RMSE 高于门控融合，分别为 4.396 和 4.253，但在 S3 和 S4 上取得了更低的 RMSE。交叉注意力在 S3 和 S4 上的 RMSE 分别为 4.110 和 4.144，对应的正改善分别为 1.00% 和 0.19%。然而，门控融合与交叉注意力之间的比较均未通过 Holm 校正。交叉注意力在 S3 和 S4 上相对于 M0 的正 也均不显著。因此，这些结果仅作为描述性的次要比较报告，而不构成交叉注意力更优的证据。

Overall, the Deep family performs better than the Flat family, although most Deep specifications still do not outperform M0. For RQ1, shipping provides the clearest improvement. S3 achieves the best performance under both gated fusion and cross-attention, and both models outperform M0 in the reported run. Remote sensing provides little additional value. It does not improve the finance-only model at S2 and weakens the gated model when added to shipping at S4.

总体来看，Deep 模型族的表现优于 Flat 模型族，但大多数 Deep 设定仍未超过 M0。对于 RQ1，航运数据带来的改善最为明显。S3 在门控融合和交叉注意力下均取得各自最好的表现，并且在报告的运行结果中都优于 M0。相比之下，遥感数据带来的额外价值较小。在 S2 中，遥感数据未能改善仅使用金融数据的模型；在 S4 中，将遥感数据加入航运数据后，门控模型的表现反而下降。

## 4.4 Flat versus Deep

## 4.4 Flat 与 Deep 的配对比较

Table 4.3 compares the main Deep model with both Flat models within each feature set. The feature-set category, forecast dates and evaluation sample are held constant. The main Deep pathway uses the finance-only Deep model at S1 and gated fusion at S2–S4. 

表 4.3 在每个特征集内，将主要 Deep 模型分别与两种 Flat 模型进行比较，并保持特征集类别、预测日期和评价样本一致。主要 Deep 路径在 S1 使用仅金融的 Deep 模型，在 S2–S4 使用门控融合。

**Table 4.3 — Matched Flat–Deep comparisons by feature set** *(n = 257)*

| Feature set | Flat model    | Flat RMSE | Deep model | Deep RMSE | **Deep vs Flat (%)** |
| ----------- | ------------- | --------- | --------------- | --------- | -------------------- |
| S1          | Ridge         | 4.256     | M1–Deep         | 4.250     | +0.15%               |
| S1          | M1–Flat–XGB   | 4.368     | M1–Deep         | 4.250     | +2.71%               |
| S2          | M2–Flat–Ridge | 4.414     | M2–Deep–Gated   | 4.253     | +3.64%               |
| S2          | M2–Flat–XGB   | 4.440     | M2–Deep–Gated   | 4.253     | +4.22%               |
| S3          | M3–Flat–Ridge | 4.553     | M3–Deep–Gated   | 4.146     | +8.95%               |
| S3          | M3–Flat–XGB   | 4.357     | M3–Deep–Gated   | 4.146     | +4.85%               |
| S4          | M4–Flat–Ridge | 4.539     | M4–Deep–Gated   | 4.180     | +7.90%               |
| S4          | M4–Flat–XGB   | 4.412     | M4–Deep–Gated   | 4.180     | +5.26%               |


*Note. Positive values indicate a lower Deep RMSE than the matched Flat model.*

*注。Deep RMSE reduction 为 Deep 相对对应 Flat 模型的 RMSE 降幅，正值表示 Deep 更低。*

Figure 4.2

**Figure 4.2 — Paired slopes from Flat XGBoost to Deep gated fusion at each information set, with S3 highlighted.**

**图 4.2 — 各信息集上由 Flat XGBoost 到 Deep 门控融合的配对斜率，S3 高亮。**

Across all four feature sets, the main Deep model records lower RMSE than both Ridge and XGBoost. The reduction ranges from 0.15% against Ridge at S1 to 8.95% against Ridge at S3. At S1 and S2, the main Deep models improve on both Flat learners but remain worse than M0. S3 is the only feature set which has lower RMSE than M0. Although the main Deep S4 model improves substantially over both Flat models, it remains worse than M0 and does not improve on the main Deep S3 model.

在全部四个特征集上，主要 Deep 模型的 RMSE 都低于 Ridge 和 XGBoost。相较于 Ridge，其 RMSE 降幅从 S1 的 0.15% 到 S3 的 8.95% 不等。在 S1 和 S2 上，主要 Deep 模型虽然优于两个 Flat 模型，但表现仍弱于 M0。S3 是唯一一个 RMSE 低于 M0 的特征集。尽管主要 Deep S4 模型相比两个 Flat 模型都有明显改善，但其表现仍弱于 M0，也没有优于主要 Deep S3 模型。

Overall, this comparisons show that the main Deep pathway records lower RMSE than both Flat learners across all four feature sets. For RQ2, this provides consistent evidence that modality-aware representation-level modelling performs better than flat feature fusion when using matched information sets. However, because the Deep and Flat pathways also differ in model architecture and data representation, the improvement cannot be attributed to representation-level fusion alone. Nevertheless, the comparison suggests that how heterogeneous data are organised and represented may affect the extent to which different information is retained and used by the model, and that preserving the distinct structure and characteristics of different data types may be valuable for future research.

总体来看，比较结果表明，主要 Deep 路径在全部四个特征集上的 RMSE 均低于两种 Flat 学习器。对于 RQ2，这一结果一致表明，在使用匹配信息集时，模态感知的表征级建模优于扁平特征融合。不过，由于 Deep 与 Flat 路径在模型架构和数据表征上也存在差异，因此这一改善不能完全归因于表征级融合本身。尽管如此，这一比较表明，异质数据的组织和表征方式可能会影响不同信息在模型中的保留与利用程度，并说明在处理不同类型的数据时，保留其各自的数据结构和特征可能具有研究价值。

## 4.5 Interpretability

Following the eligibility rule in Section 3.7.2, interpretation is reported for gated Deep model on S3. Table 4.4 combines period-specific forecast performance, modality-gate weights and absolute SHAP attribution for the 257 forecast origins.

根据第 3.7.2 节的准入规则，可解释性分析针对门控 Deep S3 模型展开。表 4.4 汇总 257 个预测起点上的分时期预测表现、模态门控权重和绝对 SHAP 归因。  

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


*Note. RMSE is evaluated in price levels, while SHAP attributes predicted log returns. Event windows are ±8 weeks.*

*注。RMSE 按价格水平评价，SHAP 则归因于预测的对数收益率。事件窗口为事件日前后各 8 周。*

Financial inputs dominate absolute SHAP throughout the sample, accounting for 96.8% of full-sample attribution compared with 3.2% for shipping. Shipping attribution is relatively higher in 2023–2024 and during the Red Sea window, at between 5.0% and 5.9%, but falls to 0.8% in 2025. Its contribution is therefore small and episodic rather than persistently elevated.

金融输入在整个样本内始终主导绝对 SHAP，在完整样本中占 96.8%，航运仅占 3.2%。航运归因在 2023 至 2024 年及红海窗口内相对较高，介于 5.0%和 5.9%之间，但在 2025 年降至 0.8%。因此，航运贡献整体较小，只在部分时期暂时上升。

The main-run gate allocates average weights of 55.8% to finance and 44.2% to shipping, a substantially more balanced division than the SHAP attribution. This contrast reflects the difference between internal representation weighting and output attribution; SHAP is therefore used as the primary basis for interpreting RQ3.

主要运行中的门控权重平均向金融和航运表示分配 55.8%和 44.2%，明显比 SHAP 归因更加均衡。这种差异反映了内部表示加权与模型输出归因衡量的是不同内容。因此，RQ3 的主要解释依据是 SHAP。

At the input-group level, EIA variables provide the largest full-sample contribution at 43.6%, followed by financial and macroeconomic variables at 31.4%. All twenty highest-ranked individual features are financial inputs, led by crude production, Cushing stocks and the federal funds rate. No shipping subgroup contributes more than 2.0% in any reported period, although PortWatch and SAR become modestly more prominent during the Red Sea window.

在输入组层面，EIA 变量的完整样本贡献最大，为 43.6%，其次是金融与宏观变量的 31.4%。排名前二十的单项特征均为金融输入，其中原油产量、Cushing 库存和联邦基金利率排名最高。所有报告时期内，单个航运子组的贡献均未超过 2.0%，但 PortWatch 和 SAR 在红海窗口中相对更加突出。

Within the shipping representation, the highest full-sample node shares belong to Jurong, Hormuz, Suez, the Cape route and Bab el-Mandeb. Jurong and Hormuz lead the rankings from 2021 to 2023, while Suez, Bab el-Mandeb and the Cape route occupy the first three positions in 2024. During the Red Sea window, attribution is distributed across several locations, with no individual node accounting for more than 12% of shipping attribution.

在航运表示内部，完整样本节点份额最高的是裕廊、霍尔木兹、苏伊士、好望角航线和曼德海峡。2021 至 2023 年主要由裕廊和霍尔木兹领先，而 2024 年排名前三的节点转为苏伊士、曼德海峡和好望角航线。在红海窗口内，归因分布于多个地点，没有任何单一节点占航运归因的 12%以上。

For RQ3, the model relies predominantly on financial information across all market conditions. Shipping provides a much smaller and more episodic contribution, becoming relatively more important in some periods of market and trade disruption, particularly in 2023–2024 and during the Red Sea event window. Its geographic focus also shifts over time across major ports and chokepoints rather than remaining concentrated in one location. Overall, the results suggest that financial data provide the model’s core predictive information, while shipping data act as a supplementary source whose importance increases under particular market conditions.

对于 RQ3，模型在不同市场条件下均主要依赖金融信息。航运数据的整体贡献明显较小，并呈现出阶段性特征，在部分市场和贸易扰动时期相对更加重要，尤其是在 2023–2024 年以及红海事件窗口内。航运信息的地域重点也会随时间在不同主要港口和咽喉之间变化，而不会长期集中于单一地点。总体而言，金融数据构成模型预测的核心信息来源，而航运数据则作为补充信息源，其重要性会在特定市场条件下上升。

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

Table 4.5 reports the results of rerunning all Deep specifications with multiple random seeds. None of the models achieves a positive mean RMSE improvement relative to M0 across runs, and only five of the thirty individual runs are positive. S3 concatenation has the best mean result, but it is still negative at −0.27%, while the main gated S3 model records −0.51%. Gated S3 also outperforms S1 in only one of the three matched runs, and all S2 runs remain worse than M0. The positive improvements observed in the main run are therefore sensitive to random initialisation. Overall, although the Deep models do not consistently outperform M0, some specifications, particularly S3, show predictive potential and merit further investigation. The better-performing Deep specifications remain broadly close to M0 rather than demonstrating a consistent forecasting advantage.

表 4.5 报告了全部 Deep 模型在多个随机种子下的重复运行结果。所有模型在跨运行平均后均未取得相对于 M0 的正 RMSE improvement，三十次运行中也只有五次为正。S3 拼接模型的平均结果最好，但仍为 −0.27%；主要门控 S3 模型的平均结果为 −0.51%。门控 S3 也只在三次匹配运行中的一次优于 S1，且全部 S2 运行均弱于 M0。因此，主运行中的正改善对随机初始化较为敏感。总体而言，Deep 模型虽然尚未表现出稳定优于 M0 的预测能力，但部分设定，尤其是 S3，显示出一定的预测潜力和进一步研究价值。表现较好的 Deep 设定仍主要接近 M0，而未形成稳定的预测优势。
