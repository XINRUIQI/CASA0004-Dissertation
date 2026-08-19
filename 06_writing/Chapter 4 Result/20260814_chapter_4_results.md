# Chapter 4 — Results（**1,200 words** ）

# 第 4 章 — 结果

## 4.1 Flat-model results

## 4.1 Flat 模型结果

Table 4.1 reports the out-of-sample performance of the Flat Ridge and XGBoost models across feature sets S1–S4, with M0 shown for comparison. Figure 4.1 visualises the $\Delta\mathrm{RMSE}$ values reported in Table 4.1. All eight Flat models have higher RMSE than M0 and therefore record negative $\Delta\mathrm{RMSE}$.

表 4.1 报告 Flat Ridge 与 XGBoost 模型在特征集 S1–S4 上的样本外表现，并列出 M0 作为比较。图 4.1 展示表 4.1 中报告的 $\Delta\mathrm{RMSE}$。八个 Flat 模型的 RMSE 均高于 M0，因此 $\Delta\mathrm{RMSE}$ 均为负值。

**Table 4.1 — Flat out-of-sample performance** *(n = 257)*

**表 4.1 — Flat 模型样本外表现** *（n = 257）*


| Set       | Variables                             | Model         | RMSE  | $\Delta\mathrm{RMSE}$ (%) |
| --------- | ------------------------------------- | ------------- | ----- | --------------------- |
| Benchmark |                                       | M0            | 4.152 |                       |
| S1        | financial time series                 | M1-Flat-Ridge | 4.256 | −2.52%                |
|           |                                       | M1-Flat-XGB   | 4.368 | −5.22%                |
| S2        | financial time series + RS            | M2-Flat-Ridge | 4.414 | −6.31%                |
|           |                                       | M2-Flat-XGB   | 4.440 | −6.95%                |
| S3        | financial time series + shipping      | M3-Flat-Ridge | 4.553 | −9.66%                |
|           |                                       | M3-Flat-XGB   | 4.357 | −4.94%                |
| S4        | financial time series + RS + shipping | M4-Flat-Ridge | 4.539 | −9.32%                |
|           |                                       | M4-Flat-XGB   | 4.412 | −6.27%                |


*Note:* Lower RMSE values indicate better forecasting performance. Positive $\Delta\mathrm{RMSE}$ values indicate a lower RMSE than M0, with larger values indicating greater improvement.

*注：* RMSE 越低表示预测表现越好。正的 $\Delta\mathrm{RMSE}$ 表示 RMSE 低于 M0，数值越大表示改善越大。

**Figure 4.1. $\Delta\mathrm{RMSE}$ for the Flat models.**

**图 4.1. Flat 模型的 $\Delta\mathrm{RMSE}$。**


For Ridge, S1 has the lowest RMSE and S3 the highest. Adding remote sensing, shipping or both raises RMSE relative to S1. For XGBoost, S3 records a slightly lower RMSE than S1 (4.357 versus 4.368), while S2 and S4 remain higher than S1. No Flat model records a positive $\Delta\mathrm{RMSE}$. Overall, the Flat family performs worse than the no-change benchmark across all information sets. 

对 Ridge 而言，S1 的 RMSE 最低、S3 最高。加入遥感、航运或两者均抬高相对于 S1 的 RMSE。对 XGBoost 而言，S3 的 RMSE 略低于 S1（4.357 对 4.368），而 S2 与 S4 仍高于 S1。没有任何 Flat 模型取得正 $\Delta\mathrm{RMSE}$。总体来看，Flat 模型族在所有信息集上的表现均差于不变预测基准。

The Flat results therefore provide no evidence of improvement relative to the no-change benchmark. Remote sensing consistently reduces performance for both Ridge and XGBoost. Shipping slightly improves XGBoost relative to S1 but worsens Ridge performance, and in neither case is the improvement sufficient to outperform M0.

因此，Flat 结果未提供相对不变基准有所改善的证据。遥感数据在 Ridge 和 XGBoost 中均使预测表现下降。航运数据相对于 S1 略微改善了 XGBoost，但降低了 Ridge 的表现，而且两种情况下都不足以优于 M0。

## 4.2 Deep-model results



## 4.2 Deep 模型结果

Table 4.2 reports Deep-model performance across S1–S4. Gated fusion is the prespecified main specification, while cross-attention is reported as a secondary comparison. No gated or cross-attention result is reported for S1 because only the finance encoder is active. Figure 4.2 visualises the $\Delta\mathrm{RMSE}$ values reported in Table 4.2.

表 4.2 报告 Deep 模型在 S1–S4 上的表现。门控融合是预先设定的主要设定，交叉注意力则作为次要比较。S1 仅启用金融编码器，因此不报告门控或交叉注意力结果。图 4.2 展示表 4.2 中报告的 $\Delta\mathrm{RMSE}$。

**Table 4.2 — Deep out-of-sample performance** *(gated = main specification; n = 257)*

**表 4.2 — Deep 模型样本外表现** *（gated = 主要设定；n = 257）*


| Set       | Variables                             | Model                   | RMSE  | $\Delta\mathrm{RMSE}$ (%) |
| --------- | ------------------------------------- | ----------------------- | ----- | --------------------- |
| Benchmark |                                       | M0                      | 4.152 |                       |
| S1        | financial time series                 | M1-Deep                 | 4.250 | −2.36%                |
| S2        | financial time series + RS            | M2-Deep-Gated           | 4.253 | −2.43%                |
|           |                                       | M2-Deep-Cross-attention | 4.396 | −5.87%                |
| S3        | financial time series + shipping      | M3-Deep-Gated           | 4.146 | +0.15%                |
|           |                                       | M3-Deep-Cross-attention | 4.110 | +1.00%                |
| S4        | financial time series + RS + shipping | M4-Deep-Gated           | 4.180 | −0.67%                |
|           |                                       | M4-Deep-Cross-attention | 4.144 | +0.19%                |


*Note:* Lower RMSE values indicate better forecasting performance. Positive $\Delta\mathrm{RMSE}$ values indicate a lower RMSE than M0, with larger values indicating greater improvement.

*注：* RMSE 越低表示预测表现越好。正的 $\Delta\mathrm{RMSE}$ 表示 RMSE 低于 M0，数值越大表示改善越大。


**Figure 4.2. $\Delta\mathrm{RMSE}$ for the Deep models.**

**图 4.2. Deep 模型的 $\Delta\mathrm{RMSE}$。**



The S1 and the gated S2 models record similar RMSEs of 4.250 and 4.253, both higher than that of M0. Neither reported fusion approach reduces RMSE relative to the finance-only Deep model at S2. Adding remote sensing therefore provides no descriptive improvement. With shipping included, gated S3 records the lowest RMSE among the gated models, at 4.146, representing a 0.15% improvement over M0. The RMSE of gated S4 rises to 4.180, which is 0.67% worse than M0. This indicates that adding remote sensing to S3 does not provide a further improvement.

S1 与门控 S2 模型的 RMSE 相近，分别为 4.250 和 4.253，均高于 M0。两种已报告的融合方法均未能在 S2 上相对于仅金融的 Deep 模型降低 RMSE。因此，加入遥感并未带来描述性改善。加入航运后，门控 S3 在各门控模型中 RMSE 最低，为 4.146，相对 M0 改善 0.15%。门控 S4 的 RMSE 升至 4.180，比 M0 差 0.67%。这表明在 S3 基础上加入遥感并未带来进一步改善。

On the reported run, cross-attention has a higher RMSE than gated fusion at S2, at 4.396 compared with 4.253, but lower RMSEs at S3 and S4. Cross-attention records RMSEs of 4.110 and 4.144 at S3 and S4, corresponding to $\Delta\mathrm{RMSE}$ of 1.00% and 0.19%. These results are therefore reported only as descriptive secondary comparisons, rather than evidence that cross-attention is superior.

在报告的运行结果中，交叉注意力在 S2 上的 RMSE 高于门控融合，分别为 4.396 和 4.253，但在 S3 和 S4 上取得了更低的 RMSE。交叉注意力在 S3 和 S4 上的 RMSE 分别为 4.110 和 4.144，对应的 $\Delta\mathrm{RMSE}$ 分别为 1.00% 和 0.19%。因此，这些结果仅作为描述性的次要比较报告，而不构成交叉注意力更优的证据。

Overall, the Deep family performs better than the Flat family, although most Deep specifications still do not outperform M0. For RQ1, shipping provides the clearest improvement. S3 achieves the best performance under both gated fusion and cross-attention, and both models outperform M0 in the reported run. Remote sensing provides little additional value. It does not improve the finance-only model at S2 and weakens the gated model when added to shipping at S4.

总体来看，Deep 模型族的表现优于 Flat 模型族，但大多数 Deep 设定仍未超过 M0。对于 RQ1，航运数据带来的改善最为明显。S3 在门控融合和交叉注意力下均取得各自最好的表现，并且在报告的运行结果中都优于 M0。相比之下，遥感数据带来的额外价值较小。在 S2 中，遥感数据未能改善仅使用金融数据的模型；在 S4 中，将遥感数据加入航运数据后，门控模型的表现反而下降。

## 4.3 Flat versus Deep



## 4.3 Flat 与 Deep

Table 4.3 compares the main Deep model with both Flat models within each feature set. The feature-set category, forecast dates and evaluation sample are held constant. The main Deep pathway uses the finance-only Deep model at S1 and gated fusion at S2–S4. 

表 4.3 在每个特征集内，将主要 Deep 模型分别与两种 Flat 模型进行比较，并保持特征集类别、预测日期和评价样本一致。主要 Deep 路径在 S1 使用仅金融的 Deep 模型，在 S2–S4 使用门控融合。

**Table 4.3 — Matched Flat–Deep comparisons by feature set** *(n = 257)*

**表 4.3 — 按特征集配对的 Flat–Deep 比较** *（n = 257）*


| Feature set | Flat model    | Flat RMSE | Deep model    | Deep RMSE | **Deep vs Flat (%)** |
| ----------- | ------------- | --------- | ------------- | --------- | -------------------- |
| S1          | M1–Flat–Ridge | 4.256     | M1–Deep       | 4.250     | +0.15%               |
| S1          | M1–Flat–XGB   | 4.368     | M1–Deep       | 4.250     | +2.71%               |
| S2          | M2–Flat–Ridge | 4.414     | M2–Deep–Gated | 4.253     | +3.64%               |
| S2          | M2–Flat–XGB   | 4.440     | M2–Deep–Gated | 4.253     | +4.22%               |
| S3          | M3–Flat–Ridge | 4.553     | M3–Deep–Gated | 4.146     | +8.95%               |
| S3          | M3–Flat–XGB   | 4.357     | M3–Deep–Gated | 4.146     | +4.85%               |
| S4          | M4–Flat–Ridge | 4.539     | M4–Deep–Gated | 4.180     | +7.90%               |
| S4          | M4–Flat–XGB   | 4.412     | M4–Deep–Gated | 4.180     | +5.26%               |


*Note:* Positive values in the Deep vs Flat (%) column indicate that the Deep model has a lower RMSE than the matched Flat model. Larger values indicate a greater relative improvement by the Deep model.

*注：* Deep vs Flat (%) 列中的正值表示 Deep 模型的 RMSE 低于配对的 Flat 模型。数值越大表示 Deep 模型的相对改善越大。

Figure 4.3 shows the change in RMSE from Flat XGBoost to the matched gated Deep model within each information set. Across all four feature sets, the main Deep model records lower RMSE than both Ridge and XGBoost. The reduction ranges from 0.15% against Ridge at S1 to 8.95% against Ridge at S3. At S1 and S2, the main Deep models improve on both Flat learners but remain worse than M0. S3 is the only information set for which both Deep specifications perform better than M0. Although the main Deep S4 model improves substantially over both Flat models, it remains worse than M0 and does not improve on the main Deep S3 model.

图 4.3 展示各信息集内从 Flat XGBoost 到配对门控 Deep 模型的 RMSE 变化。在全部四个特征集上，主要 Deep 模型的 RMSE 都低于 Ridge 和 XGBoost。相较于 Ridge，降幅从 S1 的 0.15% 到 S3 的 8.95% 不等。在 S1 和 S2 上，主要 Deep 模型虽优于两个 Flat 学习器，但仍弱于 M0。S3 是唯一一个两种 Deep 设定都优于 M0 的信息集。尽管主要 Deep S4 模型相对两个 Flat 模型都有明显改善，但仍弱于 M0，也未优于主要 Deep S3 模型。

**Figure 4.3. RMSE change from Flat XGBoost to gated Deep models.**

**图 4.3. 从 Flat XGBoost 到门控 Deep 模型的 RMSE 变化。**

Overall, these comparisons show that the main Deep pathway records lower RMSE than both Flat learners across all four feature sets. For RQ2, this provides consistent evidence that modality-aware representation-level modelling performs better than flat feature fusion when using matched information sets. Since the Deep and Flat pathways also differ in other respects, the improvement cannot be attributed to representation-level fusion alone. The comparison nevertheless suggests that the organisation and representation of heterogeneous data affect what information the model retains and uses. Preserving the distinct structure of each data type therefore warrants further investigation.

总体来看，这些比较表明，主要 Deep 路径在全部四个特征集上的 RMSE 均低于两种 Flat 学习器。对于 RQ2，这一结果一致表明，在使用匹配信息集时，模态感知的表征级建模优于扁平特征融合。由于 Deep 与 Flat 路径在其他方面也存在差异，因此这一改善不能仅归因于表征级融合。尽管如此，这一比较表明，异质数据的组织与表征方式会影响模型保留和使用哪些信息。因此，保留各数据类型的特有结构值得进一步研究。

## 4.4 Interpretability

## 4.4 可解释性

Following the eligibility rule in Section 3.7.2, interpretation is reported for the gated Deep model on S3. Table 4.4 combines period-specific forecast performance, modality-gate weights and absolute SHAP attribution for the 257 forecast origins.

根据第 3.7.2 节的准入规则，可解释性分析针对门控 Deep S3 模型展开。表 4.4 汇总 257 个预测起点上的分时期预测表现、模态门控权重和绝对 SHAP 归因。  

**Table 4.4 — Period-specific performance and attribution for gated Deep S3**

**表 4.4 — 门控 Deep S3 的分时期表现与归因**


| Period         | n   | RMSE  | $\Delta\mathrm{RMSE}$ (%) | Gate finance | Gate shipping | SHAP finance | SHAP shipping |
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

金融输入在整个样本内始终主导绝对 SHAP，在完整样本中占 96.8%，航运仅占 3.2%。航运归因在 2023–2024 年及红海窗口内相对较高，介于 5.0% 和 5.9% 之间，但在 2025 年降至 0.8%。因此，其贡献较小且呈阶段性，而非持续偏高。

The main-run gate allocates average weights of 55.8% to finance and 44.2% to shipping, a substantially more balanced division than the SHAP attribution. This contrast reflects the difference between internal representation weighting and output attribution. SHAP is therefore used as the primary basis for interpreting RQ3.

主要运行中的门控平均向金融和航运分配 55.8% 和 44.2% 的权重，明显比 SHAP 归因更加均衡。这一对比反映了内部表征加权与输出归因的差异。因此以 SHAP 作为解释 RQ3 的主要依据。

At the input-group level, EIA variables provide the largest full-sample contribution at 43.6%, followed by financial and macroeconomic variables at 31.4%. All twenty highest-ranked individual features are financial inputs, led by crude production, Cushing stocks and the federal funds rate. No shipping subgroup contributes more than 2.0% in any reported period, although PortWatch and SAR become modestly more prominent during the Red Sea window.

在输入组层面，EIA 变量的完整样本贡献最大，为 43.6%，其次是金融与宏观变量的 31.4%。排名前二十的单项特征均为金融输入，其中原油产量、Cushing 库存和联邦基金利率排名最高。所有报告时期内，单个航运子组的贡献均未超过 2.0%，但 PortWatch 和 SAR 在红海窗口中相对更加突出。

Within the shipping representation, the highest full-sample node shares belong to Jurong, Hormuz, Suez, the Cape route and Bab el-Mandeb. Jurong and Hormuz lead the rankings from 2021 to 2023, while Suez, Bab el-Mandeb and the Cape route occupy the first three positions in 2024. Figure 4.4 maps this redistribution between 2022 and 2024.

在航运表征内部，全样本节点份额最高的是裕廊、霍尔木兹、苏伊士、好望角航线与曼德海峡。2021 至 2023 年由裕廊与霍尔木兹领跑，而 2024 年苏伊士、曼德海峡与好望角航线占据前三位。图 4.4 绘制了 2022 年与 2024 年之间的这一再分配。

Figure 4.5 shows how the within-shipping node shares vary across forecast origins. During the Red Sea window, shipping attribution is distributed across several nodes. Across the full window, no single node accounts for more than 12% of the total.

图 4.5 展示航运内部节点份额如何随预测起点变化。在红海窗口内，航运归因分布在多个节点。就整个窗口而言，没有任何单一节点占总归因的 12% 以上。

For RQ3, the model relies predominantly on financial information across the reported periods. Shipping provides a much smaller and more episodic contribution. It becomes relatively more important during some periods of market and trade disruption, particularly in 2023–2024 and during the Red Sea event window. Its geographic focus also shifts over time across major ports and chokepoints rather than remaining concentrated in one location. Overall, the results suggest that financial data provide the model’s core predictive information, while shipping data act as a supplementary source whose importance increases under particular market conditions.

对于 RQ3，模型在各报告时期均主要依赖金融信息。航运数据的整体贡献明显较小，并呈现出阶段性特征。它在部分市场和贸易扰动时期相对更加重要，尤其是在 2023–2024 年以及红海事件窗口内。其地理重点也会随时间在主要港口和咽喉之间变化，而不会长期集中于单一地点。总体而言，结果表明金融数据构成模型的核心预测信息，而航运数据则作为补充来源，其重要性会在特定市场条件下上升。

**Figure 4.4. Annual mean node-level shares of absolute SHAP within the shipping modality, 2022 and 2024.**

**图 4.4. 航运模态内绝对 SHAP 的年度平均节点份额，2022 年与 2024 年。**

**Figure 4.5. Temporal variation in node-level shipping attribution for gated Deep S3.** Values are six-week trailing averages of each node’s share of absolute SHAP within the shipping modality.

**图 4.5. 门控 Deep S3 航运节点归因的时间变化。** 数值为各节点占航运模态内绝对 SHAP 份额的六周滚动均值。

## 4.5 Robustness



## 4.5 稳健性

**Table 4.5 — Random-seed robustness of all Deep specifications** *($\Delta\mathrm{RMSE}$, %)*

**表 4.5 — 全部 Deep 设定的随机种子稳健性** *（$\Delta\mathrm{RMSE}$，%）*


| Set | Model                   | Main-run $\Delta\mathrm{RMSE}$ | Across-run mean ± SD | Positive runs |
| --- | ----------------------- | ------------------------------ | -------------------- | ------------- |
| S1  | M1-Deep                 | −2.36%               | −1.00% ± 1.33        | 1/3           |
| S2  | M2-Deep-Gated           | −2.43%               | −3.15% ± 1.67        | 0/3           |
|     | M2-Deep-Concat          | −2.01%               | −1.79% ± 0.77        | 0/3           |
|     | M2-Deep-Cross-attention | −5.87%               | −3.57% ± 2.77        | 0/3           |
| S3  | M3-Deep-Gated           | +0.15%               | −0.51% ± 0.80        | 1/3           |
|     | M3-Deep-Concat          | −0.22%               | −0.27% ± 0.35        | 1/3           |
|     | M3-Deep-Cross-attention | +1.00%               | −3.01% ± 4.07        | 1/3           |
| S4  | M4-Deep-Gated           | −0.67%               | −0.91% ± 0.26        | 0/3           |
|     | M4-Deep-Concat          | −8.30%               | −3.79% ± 3.95        | 0/3           |
|     | M4-Deep-Cross-attention | +0.19%               | −1.90% ± 2.75        | 1/3           |


*Note. The seed-42 column is the main reported run in Tables 4.2–4.4. S1 has no fusion operator. Individual seeds are shown in Appendix B.2.*

*注。seed 42 列为表 4.2–4.4 的主报告运行。S1 无融合算子。各次种子见附录 B.2。*

Table 4.5 reports the results of rerunning all Deep specifications with multiple random seeds. None of the models achieves a positive mean $\Delta\mathrm{RMSE}$ across runs, and only five of the thirty individual runs are positive. No S2 specification records a positive $\Delta\mathrm{RMSE}$ in any seed. S3 concatenation has the best mean result, but it is still negative at −0.27%, while the main gated S3 model records −0.51%. Across seeds, the S3 order reverses, with concatenation (−0.27%) remaining closest to M0, followed by gated (−0.51%) and cross-attention (−3.01%). Gated S3 also outperforms S1 in only one of the three matched runs. The positive $\Delta\mathrm{RMSE}$ values observed in the main run are therefore sensitive to random initialisation. Gated fusion remains the main specification because it supplies modality weights for RQ3, not because it is the more accurate operator. Overall, although the Deep models do not consistently outperform M0, some specifications such as S3 show predictive potential and merit further investigation. The better-performing Deep specifications remain broadly close to M0 rather than demonstrating a consistent forecasting advantage.

表 4.5 报告了全部 Deep 设定在多个随机种子下的重复运行结果。所有模型在跨运行平均后均未取得正的 $\Delta\mathrm{RMSE}$，三十次运行中也只有五次为正。S2 的任何设定在任一种子上均为负。S3 拼接模型的平均结果最好，但仍为 −0.27%；主要门控 S3 模型的平均结果为 −0.51%。跨种子后，S3 排序反转，拼接（−0.27%）仍最接近 M0，其次为门控（−0.51%）和交叉注意力（−3.01%）。门控 S3 也只在三次匹配运行中的一次优于 S1。因此，主运行中观察到的正 $\Delta\mathrm{RMSE}$ 对随机初始化较为敏感。仍以门控为主设定，是因为它为 RQ3 提供模态权重，而不是因为它更准确。总体而言，Deep 模型虽然尚未稳定优于 M0，但部分设定如 S3 显示出一定的预测潜力和进一步研究价值。表现较好的 Deep 设定仍大体接近 M0，而未形成稳定的预测优势。