# Chapter 5 — Discussion


## 5.1 RQ1 — Do spatial data help?

RQ1 asked whether remote sensing and shipping add out-of-sample value beyond financial time series. The answer depends on the modelling pathway. 

研究问题一询问：遥感与航运能否在金融时序之外带来样本外价值。答案取决于建模路径。

Within the Flat family, no model outperforms no-change benchmark, consistent with the short-horizon oil-forecasting literature that treats the no-change forecast as a demanding reference (Alquist, Kilian and Vigfusson, 2013). Relative to the finance-only S1 specification, remote sensing increases forecast error for both Ridge and XGBoost, while shipping increases error for Ridge but slightly reduces it for XGBoost. The latter result shows that shipping is not uniformly detrimental within the Flat pathway, but the improvement remains insufficient to outperform M0. Overall, simply adding spatial features to Flat models does not produce additional predictive value .

在 Flat 模型族中，没有任何模型优于 M0，这与短期限油价预测文献将不变预测视为严格参照的判断一致（Alquist, Kilian and Vigfusson, 2013）。相对于仅金融的 S1 设定，加入遥感会同时提高 Ridge 与 XGBoost 的预测误差；加入航运则提高 Ridge 的误差，但略微降低 XGBoost 的误差。后一结果表明，航运在 Flat 路径中并非一律有害，但这一改善仍不足以优于 M0。总体而言，仅仅向 Flat 模型加入另类数据特征，并不能产生额外的预测价值。

Under the Deep pathway, adding remote sensing to finance in S2 does not improve forecast accuracy over either S1 or M0. By contrast, adding shipping to finance in S3 produces a small positive RMSE improvement relative to M0, making S3 the best-performing specification in the main gated pathway. Extending S3 with remote sensing in S4 does not further reduce forecast error. The secondary cross-attention results show the same ordering across the multimodal specifications, with S3 performing best, followed by S4 and S2. Shipping is therefore the more informative spatial source in this design, while remote sensing contributes little to predictive accuracy. The improvement is notable because the other main modelling specifications fail to beat M0, although the gain remains limited.

在 Deep 路径下，S2 在金融数据的基础上加入遥感数据后，相较于 S1 或 M0，并未提升预测准确性。相比之下，S3 在金融数据中加入航运数据后，相对于 M0 的 RMSE 出现了小幅改善，因此 S3 成为主要门控路径中表现最佳的模型设定。进一步在 S3 的基础上加入遥感数据形成 S4，并未继续降低预测误差。次要的交叉注意力结果也显示出相同的多模态模型排序：S3 表现最佳，其次是 S4，最后是 S2。因此，在这一设计中，航运数据是更具信息价值的替代模态，而遥感数据对于预测准确性贡献很小。这一改善值得注意，因为其他主要模型设定均未能优于 M0，不过其改善幅度仍然较小。

Observations of activity across multiple ports and chokepoints appear to be more useful for short-term forecasting than the selected remote-sensing measures from individual sites. The limited value of remote sensing may be due to a mismatch in scale: the satellite data are monthly and local, while Brent is constructed at a weekly frequency.

多个港口和航运咽喉的活动数据，似乎比单个站点的遥感指标更适合用于短期预测。遥感数据作用有限，可能是因为两类数据在尺度上并不匹配：卫星数据是月度且局部的，而 Brent 价格按周度构建。

These findings refine the AIS and satellite literature reviewed in Chapter 2. Existing studies often demonstrate that ships and satellites contain information about trade or physical activity (Adland, Jia and Strandenes, 2017; Yan et al., 2020; Hao and Wang, 2023), but less often ask whether those signals improve one-week-ahead Brent forecasts relative to both a financial baseline and M0. The present results distinguish informational content from predictive value. The ability to measure trade or industrial activity does not necessarily produce a forecast improvement against a demanding weekly benchmark.

这些发现细化了第 2 章所回顾的 AIS 与卫星文献。既有研究通常表明船舶和卫星包含贸易或实物活动信息（Adland, Jia and Strandenes, 2017; Yan et al., 2020; Hao and Wang, 2023），却较少追问这些信号能否同时相对于金融基线和 M0 改善提前一周的 Brent 预测。本文结果将信息含量与预测价值区分开来。能够测量贸易或工业活动，并不必然意味着能够在严格的周度基准下改善预测。

These findings add to the AIS and satellite studies reviewed in Chapter 2. Previous studies often demonstrate that vessel and satellite data can capture trade or physical activity (Adland, Jia and Strandenes, 2017; Yan et al., 2020; Hao and Wang, 2023), but fewer studies test whether this information improves Brent forecasts relative to both financial models and the no-change benchmark. The results here show that useful information does not always lead to predictive power. Data that capture trade or industrial activity may still fail to improve forecasting performance against a strong benchmark.

这些发现进一步补充了第 2 章回顾的 AIS 与卫星研究。既有研究通常表明，船舶和卫星数据能够反映贸易和实物活动（Adland, Jia and Strandenes, 2017; Yan et al., 2020; Hao and Wang, 2023），但较少有研究检验这些信息能否同时相对于金融模型和 M0 改善提前一周的 Brent 预测。本文结果表明，有用的信息并不一定会带来更好的预测。能够反映贸易或工业活动的数据，仍可能无法在较强的基准下改善预测表现。

## 5.2 RQ2 — Does representation-level fusion beat flat fusion?

RQ2 asked whether representation-level Deep modelling outperforms flat feature fusion under the same information sets and evaluation protocol. Across the matched multimodal sets S2–S4, the main Deep pathway records lower RMSE than the Flat learners. The largest differences appear in S3 and S4, both of which include shipping data. The results therefore show that the Deep pathway outperforms early feature concatenation under the current framework. However, outperforming the Flat models does not necessarily mean outperforming the no-change benchmark.

RQ2 关注的是：在使用相同信息集和评价方案的情况下，表征级 Deep 建模是否优于扁平特征融合。在匹配的多模态信息集 S2–S4 中，主要 Deep 路径的 RMSE 均低于 Flat 学习器。其中，差异在包含航运数据的 S3 和 S4 中最为明显。因此，在当前研究框架下，Deep 路径优于早期特征拼接。不过，优于 Flat 模型并不意味着一定能够优于不变预测基准。

Early-fusion approaches combine heterogeneous predictors within a single feature space, whereas multimodal models retain separate representations before fusion (Arevalo et al., 2017; Gohari et al., 2024). This study extends the comparison to weekly Brent forecasting. The larger Deep advantage in S3 and S4 provides further support for the value of representation-level modelling. From a spatial perspective, representation-level modelling may be particularly useful for observations distributed across networks of ports and chokepoints. Spatial relationships among shipping nodes can provide additional structural information. By contrast, treating these observations as independent tabular features does not explicitly preserve these relationships.

早期融合方法将异质预测变量合并到同一个特征空间中，而多模态模型则在融合之前保留各类数据各自的表征（Arevalo et al., 2017; Gohari et al., 2024）。本研究将这一比较拓展到周度 Brent 预测。S3 和 S4 中更明显的 Deep 优势进一步支持了表征级建模的价值。从空间角度来看，对于分布在港口和航运咽喉网络中的观测，表征级建模可能尤其有用。航运节点之间的空间关系可以提供额外的结构信息。相比之下，将这些观测作为彼此独立的表格特征处理，无法显式保留这些空间关系。

However, the comparisons in this study evaluate complete modelling pathways, including their encoders and fusion strategies. They therefore support the overall Deep approach but do not identify the preservation of network structure or any individual fusion operator as the cause of its better performance.

不过，本研究比较的是完整的建模路径，其中包括各自的编码器和融合策略。因此，这些结果支持 Deep 整体方法的优势，但不能说明其较好表现是由网络结构的保留，或某一种具体的融合方式所导致的。

## 5.3 RQ3 — What does the model rely on?

RQ3 asked how the model uses information when a specification improves on the benchmark. The gate and SHAP results show that the internally learned weights assigned to each data source do not directly correspond to their contributions to the model output. The forecast still relies mainly on financial and EIA information, while shipping provides incremental predictive information beyond the financial inputs. Although shipping has a relatively small SHAP contribution, gated Deep S3 still records a small RMSE improvement relative to M0. This suggests that a modality can add predictive value even if it accounts for only a small share of the model’s output.

RQ3 关注的是，当某一模型设定优于基准时，模型如何利用不同来源的信息。门控权重和 SHAP 结果表明，模型内部为各数据源学习得到的权重，并不直接对应这些数据源对模型输出的贡献。模型的预测仍主要依赖金融和 EIA 信息，而航运数据则在金融输入之外提供了增量预测信息。虽然航运数据的 SHAP 贡献相对较小，但 S3 模型仍然优于 M0。这表明，即使某一模态对模型输出的贡献占比较小，它仍可能带来预测价值。

This incremental role also varies with the type of disruption. Shipping becomes relatively more important during the Red Sea period, while financial inputs remain more prominent during the Russia–Ukraine window. This difference is consistent with the nature of the two disruptions. Transport-specific disruptions may increase the relevance of maritime activity, whereas broader geopolitical shocks may affect Brent through several channels, including supply expectations, inventories and financial markets. The spatial pattern support this interpretation. The model’s focus shifts from Jurong and Hormuz in earlier years to Suez, Bab el-Mandeb and the Cape route in 2024. This broadly matches the disruption and rerouting around the Red Sea. However, no single location dominates during this period. The model therefore appears to respond to changes in the geographic pattern of shipping activity rather than consistently relying on the same chokepoints.

这种增量作用也会随扰动类型而变化。在红海事件期间，航运信息的相对重要性更高，而在俄乌冲突窗口内，金融输入仍然更为突出。这种差异与两类扰动本身的性质相符。与运输直接相关的扰动可能会提高航运活动的重要性，而更广泛的地缘政治冲击则可能通过多个渠道影响 Brent，包括供应预期、库存和金融市场。空间层面的结果也支持这一解释。模型的关注重点从较早年份的裕廊和霍尔木兹，转向 2024 年的苏伊士、曼德海峡和好望角航线。这一变化与红海地区的扰动和航线改道大体一致。不过，在这一时期并没有任何单一地点占据主导。模型似乎更多是在响应航运活动地域分布的变化，而不是持续依赖同一组航运咽喉。

Attention and gate weights describe operations within fitted models rather than causal relationships. SHAP attributes the model’s Brent predictions rather than explaining the causes of price movements. Therefore, the temporal and spatial analysis can highlight periods and transport corridors that deserve further study, but it should not be interpreted as a direct signal for policy action.

注意力和门控权重反映的是已拟合模型内部的运作方式，而不是因果关系。SHAP 用于解释模型对 Brent 的预测结果，而不是解释价格变动的原因。因此，时间和空间分析可以帮助识别值得进一步研究的时期和运输走廊，但不应将其直接解读为政策行动的信号。

## 5.4 Implications

## 5.4 启示

Model choice should be aligned with the structure of the input data. Relational data, such as shipping networks, capture relationships among observations that may carry useful information. Preserving this structure before fusion may therefore be more appropriate than direct feature concatenation. More generally, new data sources should be evaluated under a common out-of-sample framework. Their value should be compared both against a model using only established predictors and against a simple benchmark forecast. The first comparison tests whether the new data add information beyond conventional inputs, while the second tests whether the full modelling system improves on a basic forecasting rule.

模型选择应与输入数据的结构相匹配。对于航运网络这类关系型数据，在融合之前保留模态内部的结构，可能比直接进行特征拼接更合适。无论采用何种框架，每一种新增模态都应在相同的滚动样本外评价方案下，分别与仅使用金融数据的 S1 和不变预测基准 M0 进行比较。前者用于判断该模态是否在传统预测变量之外提供了额外信息，后者则用于判断完整模型是否优于一个简单的预测规则。

These models provide diagnostic rather than causal insight into oil prices. They show which information is useful for one-week-ahead prediction and help identify periods or parts of the supply network that deserve further analysis. However, they do not explain the mechanisms that cause oil prices to move. Their main value is therefore to integrate and assess predictive information from different sources, thereby supporting further analysis of oil-market conditions.

这些模型提供的是对油价的诊断性信息，而不是因果解释。它们可以显示哪些信息有助于提前一周的预测，并帮助识别值得进一步分析的时期或供应网络中的部分环节。然而，它们并不能解释导致油价变动的具体机制。因此，这类模型的主要价值在于整合并评估来自不同数据源的预测信息，从而支持对油市状况的进一步分析。

Spatial data should be assessed according to how well their scale, frequency and structure match the analytical task. Remote sensing describes conditions at selected facilities, so it may be more suitable for monitoring facility activity or regional production. Shipping data capture flows across connected ports, chokepoints and routes, and therefore better reflect disruptions and adjustments across the global oil supply network. Better observation of such physical activity may be useful for applications such as energy-security monitoring, trade planning and inflation-sensitive fiscal management, but it does not necessarily improve one-week-ahead Brent forecasts. More broadly, this study suggests that the value of a data source depends on the task for which it is used. A data source may be useful for forecasting, monitoring, explanation or risk detection, without being equally useful for all of them. In particular, monitoring value and predictive value should be assessed separately, since data that effectively capture physical activity or market stress may not necessarily improve forecasts.

空间数据应根据其尺度、频率和结构与具体分析任务的匹配程度进行评价。遥感数据反映特定设施的状况，因此可能更适合监测设施活动或区域生产。航运数据则反映相互连接的港口、航运咽喉和航线之间的流动，因此更能体现全球石油供应网络中的中断和调整。对这类实物活动进行更好的观测，可能有助于能源安全监测、贸易规划和对通胀敏感的财政管理等应用，但并不一定能够改善提前一周的 Brent 预测。更广泛地说，本研究表明，一种数据源的价值取决于其所服务的具体任务。某类数据可能适用于预测、监测、解释或风险识别，但并不一定在所有任务中都同样有效。尤其需要将监测价值与预测价值分别评价，因为能够有效反映实物活动或市场压力的数据，并不一定能够改善预测。


## 5.5 Limitations

## 5.5 局限

The sample size in this study is limited because of the availability of the different data sources. The evaluation contains only 257 rolling out-of-sample forecasts. Deep S3 improves only slightly over M0, and the improvement is concentrated in certain periods and event windows. This suggests that its predictive value is limited and conditional rather than consistently stable. The findings are specific to one-week-ahead Brent forecasting and should not be generalised to other forecast horizons.

本研究受不同类型数据可得性的限制，样本数量相对有限。评价样本仅包含 257 个滚动样本外预测。Deep S3 相对于 M0 仅有小幅改善，而且这种改善主要集中在部分时期和事件窗口。这表明其预测价值较为有限，并具有一定条件性，而不是持续稳定地存在。本研究的结果仅适用于提前一周的 Brent 预测，不应直接推广到其他预测期限。

The study applies a range of methods to reduce information leakage, including publication lags and as-of alignment, but limitations in data availability and preprocessing remain. The study uses revised historical series rather than real-time vintages and therefore does not fully reproduce the information available at each forecast origin. Monthly Global Fishing Watch data and remote-sensing inputs are carried forward weekly between releases, limiting their ability to capture short-lived changes. Some variables are also indirect proxies, while preprocessing and representation choices such as missing-value treatment and the use of fixed Prithvi-EO-2.0 embeddings introduce additional assumptions. The observed model performance therefore cannot be attributed to the source data alone.

本研究采用了多种方法减少信息泄漏，包括发布时间滞后和 as-of 对齐，但数据可得性和预处理方面仍存在一定限制。本研究使用的是修订后的历史序列，而非实时 vintage 数据，因此无法完全复现各预测起点当时实际可获得的信息。月度 GFW 和遥感输入在两次发布之间按周向后沿用，因此捕捉短期变化的能力有限。此外，部分变量属于间接代理指标，而缺失值处理以及使用固定的 Prithvi-EO-2.0 嵌入等预处理和表征方式也引入了额外假设。因此，观察到的模型表现不能仅归因于原始数据本身。

The spatial findings are conditional on the coverage and construction of a purposively selected network of eleven sites and six chokepoints. The site sample is weighted toward Gulf export and transshipment nodes and Asian import and refining hubs, although it also includes Rotterdam and Houston. No site-level AOIs are located in Russia, West Africa or Latin America, although the Panama Canal is included as a chokepoint. These omissions partly reflect constraints on consistent cross-source data availability and satellite observability over the full study period, although using observability as a site-selection criterion may itself introduce geographic bias. The importance assigned to individual nodes by the model therefore reflects dependence within this constructed network rather than a comprehensive ranking of global oil infrastructure.

本研究的空间结果受到所选航运网络覆盖范围和构建方式的限制。空间分析基于有目的地选取的 11 个站点和 6 个航运咽喉，主要集中在海湾地区的出口节点以及亚洲的进口和炼化中心，同时未覆盖俄罗斯、西非和拉丁美洲地区。将卫星可观测性作为站点选择标准之一，也可能带来一定的地理偏差。因此，模型赋予各个节点的重要性只反映其在这一特定构建网络中的相对作用，而不能视为对全球石油基础设施的全面排名。

The comparison between the Flat and Deep families evaluates differences between complete modelling pathways rather than differences in any single model component. The two pathways differ in model class, data representation and fusion approach. For example, the Deep shipping pathway additionally introduces an explicit graph structure. Therefore, the observed performance differences cannot be clearly attributed to any single model architecture or to the gated-fusion mechanism itself.

Flat 模型族与 Deep 模型族的比较评估的是完整建模路径之间的差异，而不是单一模型组件的差异。两条路径在模型类别、数据表征和融合方式等方面均存在差异。例如，Deep 航运路径额外引入了显式的图结构。因此，观察到的性能差异不能明确归因于某一种模型架构或门控融合机制本身。

## 5.6 Future research

## 5.6 未来研究

Future research should test whether the value of shipping data remains stable over longer periods and across a wider oil-transport network. A longer-term evaluation could use archived data releases and include additional regions such as Russian Baltic and Black Sea ports, West African loading areas and Latin American exporters. This would provide evidence from a longer time period and broader geographic coverage. Where AIS data are incomplete, SAR-based vessel detection could help capture vessel activity not visible in AIS data, including dark-fleet activity.

未来研究应进一步检验航运数据的价值是否能在更长时间范围和更广泛的石油运输网络中保持稳定。更长期的评价可以使用历史存档版本的数据，并纳入更多地区，例如俄罗斯波罗的海和黑海港口、西非装载区以及拉丁美洲出口地区。这样可以在更长的时间范围和更广的地理覆盖下进一步检验现有结果。在 AIS 数据不完整的地区，可以使用基于 SAR 的船舶检测来补充观测，以捕捉 AIS 数据中无法看到的船舶活动，包括暗船活动。

Future research should test the transferability of this framework by applying it to other forecasting tasks and markets. The same Flat–Deep comparison could be applied to other energy commodities and markets, such as WTI crude oil, LNG and natural gas. It could also be extended to grain and metal markets to test whether modelling network structure remains useful outside the energy sector. Remote sensing should be evaluated for targets that better match its spatial and temporal scale, such as facility activity, regional production or longer-term price movements. Where possible, future studies should also compare Flat and Deep remote-sensing models using the same underlying inputs. This would make it easier to separate the effect of model architecture from differences in data representation.

未来研究应通过将这一框架应用于其他预测任务和市场，检验其可迁移性。相同的 Flat–Deep 比较方法可以应用于其他能源品种和市场，例如 WTI 原油、LNG 和天然气。该框架还可以扩展到粮食和金属市场，以检验网络结构建模在能源领域之外是否仍然具有价值。遥感数据则应针对与其空间和时间尺度更匹配的目标进行评价，例如设施活动、区域产量或更长期的价格变化。在条件允许的情况下，未来研究还应使用相同的底层输入比较 Flat 和 Deep 遥感模型。这样可以更清楚地区分模型架构与数据表征差异各自带来的影响。

Future research should evaluate the practical value of the forecasts. For public-sector applications, probabilistic forecasts and scenario ranges could be compared with existing financial and EIA monitoring. This would help assess whether shipping information improves the timing and focus of energy-security monitoring, trade planning or inflation-sensitive budgeting. Evaluating policy actions such as strategic reserve releases, sanctions or fiscal responses would require more than forecasting alone. Forecasting models would need to be combined with causal or decision-analysis methods to estimate how these actions affect prices, supply and policy costs. Model attribution should therefore be used as a diagnostic tool rather than as an automatic policy signal.

未来研究应进一步评估预测结果的实际应用价值。对于公共部门应用，可以将概率预测和情景范围与现有的金融和 EIA 监测体系进行比较，从而判断航运信息是否能够改善能源安全监测、贸易规划或对通胀敏感的预算管理中的判断时机和关注重点。评价战略储备释放、制裁或财政应对等政策行动，仅依靠预测模型是不够的。还需要将预测模型与因果分析或决策分析方法结合，以评估这些政策对价格、供应和政策成本的影响。因此，模型归因结果应作为辅助诊断工具，而不应直接作为自动化的政策信号。
