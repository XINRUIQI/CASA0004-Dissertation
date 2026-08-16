# Chapter 5 — Discussion

## 5.1 RQ1 — Do alternative data help?

RQ1 asked whether remote sensing and shipping add out-of-sample value beyond financial time series and the no-change benchmark. The answer depends on the modelling pathway. Within the Flat family, no model outperforms no-change benchmark, consistent with the short-horizon oil-forecasting literature that treats the no-change forecast as a demanding reference (Alquist, Kilian and Vigfusson, 2013) Rela.tive to the finance-only S1 specification, remote sensing increases forecast error for both Ridge and XGBoost, while shipping increases error for Ridge but slightly reduces it for XGBoost. The latter result shows that shipping is not uniformly detrimental within the Flat pathway, but the improvement remains insufficient to outperform M0. Overall, simply adding alternative-data features to Flat models does not produce additional predictive value against the no-change benchmark.

研究问题一询问：遥感与航运能否在金融时序和不变预测基准之外带来样本外价值。答案取决于建模路径。在 Flat 模型族中，没有任何模型优于 M0，这与短期限油价预测文献将不变预测视为严格参照的判断一致（Alquist, Kilian and Vigfusson, 2013）。相对于仅金融的 S1 设定，加入遥感会同时提高 Ridge 与 XGBoost 的预测误差；加入航运则提高 Ridge 的误差，但略微降低 XGBoost 的误差。后一结果表明，航运在 Flat 路径中并非一律有害，但这一改善仍不足以优于 M0。总体而言，仅仅向 Flat 模型加入另类数据特征，并不能相对不变预测基准产生额外的预测价值。

Under the Deep pathway, adding remote sensing to finance in S2 does not improve forecast accuracy over either S1 or M0. By contrast, adding shipping to finance in S3 produces a small positive RMSE improvement relative to M0, making S3 the best-performing specification in the main gated pathway. Extending S3 with remote sensing in S4 does not further reduce forecast error. The secondary cross-attention results show the same ordering across the multimodal specifications, with S3 performing best, followed by S4 and S2. Shipping is therefore the more informative alternative modality in this weekly Brent design, while remote sensing contributes little to one-week-ahead predictive accuracy. The improvement is notable because the other main modelling specifications fail to beat M0, although its magnitude remains modest.**it should be interpreted as evidence of predictive rather than operational value.**是否有必要加？

在 Deep 路径下，S2 在金融数据的基础上加入遥感数据后，相较于 S1 或 M0，并未提升预测准确性。相比之下，S3 在金融数据中加入航运数据后，相对于 M0 的 RMSE 出现了小幅改善，因此 S3 成为主要门控路径中表现最佳的模型设定。进一步在 S3 的基础上加入遥感数据形成 S4，并未继续降低预测误差。次要的交叉注意力结果也显示出相同的多模态模型排序：S3 表现最佳，其次是 S4，最后是 S2。因此，在这一以周度布伦特原油为对象的预测设计中，航运数据是更具信息价值的替代模态，而遥感数据对于提前一周的预测准确性贡献很小。 这一改善值得关注，因为其他主要模型设定均未能优于 M0。

Distributed observations of port and chokepoint activity appear more useful for short-horizon forecasting than the selected site-level remote-sensing proxies. The lack of remote-sensing gains may reflect a temporal and spatial mismatch between monthly, localised AOI signals and a weekly global benchmark price.

港口与咽喉的分布式观测，对短期限预测似乎比所选的站点级遥感代理更有用。遥感未能带来增益，可能反映月度、局部 AOI 信号与周度全球基准价格之间在时间和空间上的错配。

These findings refine the AIS and satellite literature reviewed in Chapter 2. Existing studies often demonstrate that ships and satellites contain information about trade or physical activity (Adland, Jia and Strandenes, 2017; Yan et al., 2020; Hao and Wang, 2023), but less often ask whether those signals improve one-week-ahead Brent forecasts relative to both a financial baseline and M0. The present results distinguish informational content from predictive value. The ability to measure trade or industrial activity does not necessarily produce a forecast improvement against a demanding weekly benchmark.

这些发现细化了第 2 章所回顾的 AIS 与卫星文献。既有研究通常表明船舶和卫星包含贸易或实物活动信息（Adland, Jia and Strandenes, 2017; Yan et al., 2020; Hao and Wang, 2023），却较少追问这些信号能否同时相对于金融基线和 M0 改善提前一周的 Brent 预测。本文结果将信息含量与预测价值区分开来。能够测量贸易或工业活动，并不必然意味着能够在严格的周度基准下改善预测。

## 5.2 RQ2 — Does representation-level fusion beat flat fusion?

RQ2 asked whether representation-level Deep modelling outperforms flat feature fusion when the information sets and evaluation protocol are held fixed. Across the matched multimodal sets S2–S4, the main Deep pathway records lower RMSE than both Flat learners. The differences are largest in the shipping-inclusive S3 and S4 sets, although only S3 in the main Deep pathway also outperforms M0. The results therefore favour the Deep pathway over early feature concatenation, while showing that an advantage over Flat models does not necessarily translate into an improvement over the no-change benchmark.

研究问题二询问：在信息集与评估协议保持固定时，表示级 Deep 建模是否优于扁平特征融合。在匹配的多模态集合 S2–S4 上，主 Deep 路径的 RMSE 均低于两种 Flat 学习器。差异在含航运的 S3 与 S4 上最大，但主 Deep 路径中仅 S3 同时优于 M0。因此，结果更支持 Deep 路径而非早期特征拼接，同时也表明：相对 Flat 模型的优势，并不必然转化为相对不变预测基准的改善。

Early-fusion approaches combine heterogeneous predictors within a single feature space, whereas multimodal models retain separate representations before fusion (Arevalo et al., 2017; Gohari et al., 2024). The larger Deep advantage in the shipping-inclusive S3 and S4 sets extends this comparison to weekly Brent forecasting. From a spatial perspective, the result suggests that representation-level modelling may be particularly useful for observations distributed across networks of ports and chokepoints. Treating the same observations as independent tabular features does not explicitly represent these spatial relationships. This pattern is consistent with spatial relationships among shipping nodes providing useful structure.

早期融合方法将异质性预测变量整合到同一个特征空间中，而多模态模型则在融合之前保留各模态相对独立的表示。在包含航运数据的 S3 和 S4 设定中，Deep 模型表现出更明显的优势，这将上述比较进一步拓展到了周度布伦特原油价格预测情境。从空间维度来看，这一结果表明，对于分布在港口和关键航运咽喉节点网络中的观测数据，基于表示层的建模方式可能尤其有用。若将这些观测简单视为彼此独立的表格型特征，则无法显式刻画其空间关系。这一结果与如下解释一致：航运节点之间的空间关系可能提供了有助于预测的结构性信息。

However, the matched comparisons evaluate complete modelling pathways, including their encoders and fusion strategies. They therefore support the overall Deep approach but do not identify the preservation of network structure or any individual fusion operator as the cause of its lower RMSE.

然而，匹配比较评估的是完整建模路径，包括其编码器与融合策略。因此，它们支持整体 Deep 路径，但并不能将更低的 RMSE 归因于网络结构的保留，或归因于任何一个单独的融合算子。

## 5.3 RQ3 — What does the model rely on when value exists?

RQ3 asked how the model uses information when an alternative-data specification improves on the benchmark. The gate and SHAP results show that internal representation weights do not translate directly into contributions to model output. The forecast continues to rely primarily on financial and EIA information, with shipping providing a complementary signal. The coexistence of a relatively small shipping attribution and an improvement over M0 indicates that a modality can add predictive value without dominating the forecast.

研究问题三询问：当另类数据设定相对基准具有价值时，模型如何使用信息。门控与 SHAP 结果之间的分歧表明，内部表征加权并不等同于对最终预测的贡献。预测仍然锚定于金融与实物市场信息，而航运作为补充信号对这一核心加以细化，而非取而代之。因此，航运归因相对较小、却仍能相对 M0 取得改善，二者并存，说明增量价值可以来自有限但有针对性的贡献。

This complementary role also varies with the type of disruption. Shipping becomes relatively more prominent during the Red Sea period, whereas financial inputs remain more prominent during the Russia–Ukraine window. This contrast is consistent with transport-specific disruptions increasing the relevance of maritime activity, while broader geopolitical shocks may affect Brent through a wider combination of supply expectations, inventories and financial-market channels. The spatial pattern supports the same interpretation. The model’s focus shifts from Jurong and Hormuz in earlier years towards Suez, Bab el-Mandeb and the Cape route in 2024. This pattern broadly coincides with disruption and rerouting around the Red Sea. Yet no single location dominates within that window. The model therefore appears to respond to changes in the spatial configuration of shipping activity rather than relying persistently on one chokepoint.

这一补充角色也随扰动类型而变化。航运在红海时期相对更为突出，而俄乌窗口则更明显由金融主导。这一对照与如下解释一致：针对运输环节的扰动会提高海上活动的相关性，而更广的地缘政治冲击可能通过供给预期、库存与金融市场等更广泛渠道影响 Brent。空间格局支持同一解读。模型的关注从较早年份的裕廊岛与霍尔木兹，转向 2024 年的苏伊士、曼德海峡与好望角航线，在地理上与红海周边的扰动与改道相一致。然而，在该窗口内没有任何单一地点占据主导。因此，模型似乎是在响应航运网络配置的变化，而不是持续依赖某一个咽喉。

Attention and gate weights describe operations within a fitted model rather than causal relationships (Jain and Wallace, 2019), while SHAP attributes predicted log returns rather than explaining the causes of price movements. The temporal and spatial diagnostics can therefore identify periods and transport corridors for further investigation, but should not be treated as stand-alone policy alerts.

注意力与门控权重描述拟合模型内部的运算，而非因果关系（Jain and Wallace, 2019）；SHAP 归因于预测的对数收益率，也不能解释价格变化的成因。因此，时间与空间诊断可以帮助识别值得进一步调查的时期和运输通道，但不应被视为独立的政策警报。

## 5.4 Implications

## 5.4 启示

Model choice should be aligned with the structure of the input data. For relational data such as shipping networks, preserving within-modality structure before fusion may be more appropriate than direct concatenation. Whatever framework is used, each new modality should be evaluated under the same rolling out-of-sample protocol against finance-only S1 and no-change M0. The former shows whether the modality adds information beyond conventional predictors, while the latter shows whether the complete system improves on a simple forecasting rule.

模型选择应与输入数据的结构相匹配。对于航运网络等具有明确关系结构的数据，在融合前保留模态内部结构可能比直接拼接更为合适。无论采用何种框架，新模态都应在相同的滚动样本外协议下分别与仅金融 S1 和不变预测 M0 比较。前者用于判断新模态是否在传统预测变量之外增加信息，后者用于判断完整系统能否优于简单预测规则。

These models offer diagnostic rather than causal insight into oil prices. They show which information is useful for next-week prediction and can direct attention to periods and parts of the supply network that warrant further investigation, but they do not identify the mechanisms generating price movements. Their role is therefore to organise predictive evidence for market analysis rather than to substitute for structural or causal explanations of oil-price formation.

这些模型为理解油价提供的是诊断性线索，而不是因果解释。它们能够显示哪些信息有助于预测下一周价格，并将调查重点指向值得关注的时期和供应网络环节，但不能识别价格变化的形成机制。因此，其价值在于为市场分析筛选值得进一步考察的预测关系，而不是替代对油价形成过程的结构性或因果解释。

Spatial data should be assessed according to how well their scale, frequency and structure match the forecasting target, rather than by geographic specificity alone. Remote sensing describes conditions at selected facilities, so its monthly signals may be more suitable for monitoring facility activity or regional production. Shipping data capture flows across connected ports, chokepoints and corridors and therefore better reflect disruption and adjustment across the global oil supply network. They can complement weekly Brent monitoring, although their contribution remains too limited to replace financial and EIA information. For energy-security monitoring, trade planning and inflation-sensitive fiscal management, better observation of physical stress does not necessarily improve one-week-ahead Brent forecasts.

空间数据应根据其尺度、频率和结构与预测目标的匹配程度进行评价，而不能仅凭其具有明确的地理定位便判断其预测价值。遥感描述选定设施的局部状态，其月度信号可能更适合监测设施活动或地区产出。航运数据描述相互连接的港口、咽喉和运输通道之间的流动，因此更能反映全球石油供应网络的扰动与调整。航运数据可以补充周度 Brent 监测，但其贡献仍不足以取代金融与 EIA 信息。对于能源安全监测、贸易规划和关注通胀的财政管理，更清楚地观测实物压力并不一定能够改善提前一周的 Brent 预测。

## 5.5 Limitations

## 5.5 局限

With only 257 rolling out-of-sample forecasts, the small improvement of Deep S3 over M0, concentrated in some subperiods and event windows, provides evidence of limited and conditional rather than stable predictive value. The findings are specific to one-week-ahead Brent forecasting and should not be generalised to other horizons, regional oil prices or targets such as volatility.

本文仅包含 257 个滚动样本外预测，Deep S3 相对于 M0 的小幅改善又主要集中于部分子时期和事件窗口，因此只能说明航运数据具有有限且有条件的预测价值，而不能视为稳定优势。相关结论仅适用于提前一周的 Brent 预测，不能直接推广到其他预测期、地区油价或波动率等目标。

Despite publication lags and as-of alignment, the study uses revised historical series rather than real-time vintages and therefore does not fully reproduce the information available at each forecast origin. Monthly GFW and remote-sensing inputs are repeated between releases, limiting their ability to capture short-lived changes. Because alternative-data variables are indirect proxies and preprocessing choices such as missing-value treatment and frozen Earth-observation embeddings introduce additional assumptions, model performance cannot be attributed solely to the underlying signals.

尽管考虑了发布时滞并按当时可得信息进行对齐，本文使用的仍是修订后的历史序列而非实时数据版本，因此无法完全复现每个预测起点的信息集。月度 GFW 与遥感输入在两次发布之间跨周重复，限制了其捕捉短期变化的能力。另类数据属于间接代理，缺失值处理和冻结的对地观测嵌入又引入了额外假设，因此模型表现不能完全归因于底层数据的预测信号。

The spatial findings are conditional on a purposively selected network of eleven sites and six chokepoints, concentrated on Gulf export and Asian import and refining hubs while excluding Russian, West African and Latin American loading regions. Selecting partly for satellite observability introduced further geographic bias, while graph symmetrisation removed the direction and type of connections. Node attributions therefore describe dependence within this constructed network rather than a comprehensive ranking of global oil infrastructure.

空间结果取决于有目的构建的十一个站点与六个咽喉网络。该网络主要集中于海湾出口以及亚洲进口和炼化枢纽，未覆盖俄罗斯、西非和拉丁美洲的装货地区。以卫星可观测性作为选站条件进一步引入了地理偏差，航运图的对称化处理也去除了连接的方向与类型。因此，节点归因仅描述模型在这一特定网络内部的依赖，而不是对全球石油基础设施的完整重要性排序。

The Flat–Deep comparisons evaluate complete modelling pathways rather than individual components because the models differ in class, capacity, encoders and fusion procedures. Their remote-sensing inputs are not fully identical, and the Deep shipping pathway also introduces an explicit graph structure. Performance differences therefore cannot be attributed specifically to the shipping GAT or gated fusion.The gated model was selected as the main Deep specification because it provides the modality weights required for RQ3, not because gated fusion was shown to outperform other fusion mechanisms. Finally, RMSE alone does not establish operational value because trading costs and returns, hedging outcomes and policy interventions were not evaluated.

Flat–Deep 的比较评估的是完整的建模路径，而不是单个模型组件，因为这些模型在模型类别、模型容量、编码器以及融合方式等方面均存在差异。此外，它们使用的遥感输入并不完全相同，而 Deep 航运路径还额外引入了显式的图结构。因此，模型性能上的差异不能被明确归因于航运 GAT 或门控融合机制本身。之所以选择门控模型作为主要的 Deep 模型设定，是因为该模型能够提供回答 RQ3 所需的模态权重，而并非因为已有证据表明门控融合优于其他融合机制。最后，仅凭 RMSE 并不足以证明模型具有实际运营价值，因为本研究并未评估交易成本与收益、套期保值效果以及政策干预等方面。

## 5.6 Future research

## 5.6 未来研究与收束

Future research should test whether the shipping contribution persists over longer periods and across a broader oil-transport network. A longer evaluation using archived releases and expanded coverage of Russian Baltic and Black Sea ports, West African loading regions and Latin American exporters would extend the evidence beyond the present sample and selected corridors. Where AIS coverage is incomplete, SAR-based vessel detection could extend observation to tracking gaps and dark-fleet activity.

未来研究应检验航运数据的贡献能否在更长时期和更广泛的石油运输网络中持续存在。使用存档发布数据延长评价时期，并将监测范围扩展至俄罗斯波罗的海与黑海港口、西非装货地区和拉丁美洲出口地，可以将证据扩展到当前样本与选定通道之外。对于 AIS 覆盖不足的新纳入地区，可将本文已经使用的 SAR 暗船检测指标同步扩展，用以补充 AIS 未能观测到的船舶活动。

Transferability should be tested across forecasting targets and markets. The matched Flat–Deep design could be applied to longer horizons, regional oil prices and related energy commodities, including WTI, LNG and natural gas. Applying the same design to grain or metals would test whether network representation remains useful beyond energy markets. Remote sensing could also be evaluated against facility activity, regional production or longer-horizon prices that better match its spatial and temporal scale. Where data allow, a stricter like-for-like remote-sensing comparison between Flat and Deep would isolate architecture more cleanly.
研究设计的可迁移性需要在不同预测目标和市场中加以检验。匹配的 Flat–Deep 设计可应用于更长预测期、地区油价以及 WTI、LNG 和天然气等相关能源商品。扩展至粮食或金属还可检验网络表示在能源市场之外是否仍然有效。遥感也可用于预测与其时空尺度更匹配的设施活动、地区产出或长期价格。在数据允许时，对 Flat 与 Deep 开展更严格的同遥感输入比较，可以更清晰地分离架构效应。

Future research should test practical value beyond RMSE using transaction-cost-adjusted trading, simple hedging rules and forecast performance during major disruptions. For public-sector use, probabilistic forecasts and scenario ranges could be compared with existing financial and EIA monitoring to assess whether shipping information improves the timing or prioritisation of investigation in energy-security monitoring, trade planning and inflation-sensitive budgeting. Model attribution should remain a diagnostic aid rather than an automatic policy alert.

Future research should test practical value beyond RMSE using transaction-cost-adjusted trading, simple hedging rules and forecast performance during major disruptions. For public-sector use, probabilistic forecasts and scenario ranges could be compared with existing financial and EIA monitoring to assess whether shipping information improves the timing or prioritisation of investigation in energy-security monitoring, trade planning and inflation-sensitive budgeting. **Evaluating specific interventions, such as strategic reserve releases, sanctions or fiscal responses, would require combining forecasting with causal identification or decision-analysis frameworks to estimate counterfactual effects on prices and supply together with policy costs.有必要吗？**Model attribution should remain a diagnostic aid rather than an automatic policy alert.

未来研究应通过计入交易成本的交易表现、简单对冲规则和重大扰动期间的预测表现，检验模型在 RMSE 之外是否具有实际价值。对于公共部门，可将概率预测和情景区间与现有金融和 EIA 监测进行比较，检验航运信息能否改善能源安全监测、贸易规划和关注通胀的预算流程中调查工作的时机与优先顺序。**若要评价战略储备释放、制裁或财政应对等具体干预，未来研究还需将预测模型与因果识别或决策分析框架结合，估计政策对价格和供应的反事实影响及其成本。**模型归因仍应作为辅助诊断，而不能直接转化为政策警报。