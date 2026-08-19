# Chapter 5 — Discussion

# 第 5 章 — 讨论

## 5.1 RQ1 — Do spatial data help?

## 5.1 RQ1 — 空间数据是否有帮助？

RQ1 asked whether remote sensing and shipping add out-of-sample value beyond financial time series. The answer depends on the modelling pathway. 

RQ1 询问遥感与航运能否在金融时序之外带来样本外价值。答案取决于建模路径。

Within the Flat family, no model outperforms the no-change benchmark, consistent with the short-horizon oil-forecasting literature that treats the no-change forecast as a demanding reference (Alquist, Kilian and Vigfusson, 2013). Relative to the finance-only S1 specification, remote sensing increases forecast error for both Ridge and XGBoost, while shipping increases error for Ridge but slightly reduces it for XGBoost. The latter result shows that shipping is not uniformly beneficial within the Flat pathway, since the improvement remains insufficient to outperform M0. Overall, simply adding spatial features to Flat models does not produce additional predictive value .

在 Flat 模型族中，没有任何模型优于不变预测基准，这与短期限油价预测文献将不变预测视为严格参照的判断一致（Alquist, Kilian and Vigfusson, 2013）。相对于仅金融的 S1 设定，加入遥感会同时提高 Ridge 与 XGBoost 的预测误差；加入航运则提高 Ridge 的误差，但略微降低 XGBoost 的误差。后一结果表明，航运在 Flat 路径中并非一律有益，因为这一改善仍不足以优于 M0。总体而言，仅仅向 Flat 模型加入空间特征，并不能产生额外的预测价值。

Under the Deep pathway, adding remote sensing to finance in S2 does not improve forecast accuracy over either S1 or M0. By contrast, adding shipping to finance in S3 produces a small positive $\Delta\mathrm{RMSE}$, making S3 the best-performing specification in the main gated pathway. Extending S3 with remote sensing in S4 does not further reduce forecast error. The secondary cross-attention results show the same ordering across the multimodal specifications, with S3 performing best, followed by S4 and S2. Shipping is therefore the more informative spatial source in this design, while remote sensing contributes little to predictive accuracy. The improvement is notable because the other main modelling specifications fail to beat M0, although the gain remains limited.

在 Deep 路径下，S2 在金融基础上加入遥感后，相较于 S1 或 M0，并未提升预测准确性。相比之下，S3 在金融中加入航运后录得小幅正 $\Delta\mathrm{RMSE}$，因此 S3 成为主要门控路径中表现最佳的设定。进一步在 S3 基础上加入遥感形成 S4，并未继续降低预测误差。次要的交叉注意力结果也显示出相同的多模态设定排序：S3 表现最佳，其次是 S4，最后是 S2。因此，在这一设计中，航运是更具信息量的空间数据源，而遥感对预测精度贡献很小。这一改善值得注意，因为其他主要建模设定均未能优于 M0，不过其改善幅度仍然有限。

Observations of activity across multiple ports and chokepoints appear to be more useful for short-term forecasting than the selected remote-sensing measures from individual sites. The limited value of remote sensing may be due to a mismatch in scale. The satellite data are monthly and local, while Brent is constructed at a weekly frequency.

多个港口和咽喉的活动观测，似乎比来自单个站点的所选遥感指标更适合用于短期预测。遥感作用有限，可能是因为尺度不匹配。卫星数据是月度且局部的，而 Brent 按周度构建。

These findings refine the AIS and satellite literature reviewed in Chapter 2. Existing studies often demonstrate that ships and satellites contain information about trade or physical activity (Adland, Jia and Strandenes, 2017; Yan et al., 2020; Hao and Wang, 2023), but less often ask whether those signals improve one-week-ahead Brent forecasts relative to both a financial baseline and M0. The present results distinguish informational content from predictive value. The ability to measure trade or industrial activity does not necessarily produce a forecast improvement against a demanding weekly benchmark.

这些发现细化了第 2 章所回顾的 AIS 与卫星文献。既有研究通常表明船舶和卫星包含贸易或实物活动信息（Adland, Jia and Strandenes, 2017; Yan et al., 2020; Hao and Wang, 2023），却较少追问这些信号能否同时相对于金融基线和 M0 改善提前一周的 Brent 预测。本文结果将信息含量与预测价值区分开来。能够测量贸易或工业活动，并不必然意味着能够在严格的周度基准下改善预测。

## 5.2 RQ2 — Does representation-level fusion beat flat fusion?

## 5.2 RQ2 — 表征级融合是否优于扁平融合？

RQ2 asked whether representation-level Deep modelling outperforms flat feature fusion under the same information sets and evaluation protocol. Across the matched multimodal sets S2–S4, the main Deep pathway records lower RMSE than the Flat learners. The largest differences appear in S3 and S4, both of which include shipping data. The results therefore show that the Deep pathway outperforms early feature concatenation under the current framework. However, outperforming the Flat models does not necessarily mean outperforming the no-change benchmark.

RQ2 询问：在相同信息集和评价方案下，表征级 Deep 建模是否优于扁平特征融合。在匹配的多模态集合 S2–S4 中，主要 Deep 路径的 RMSE 均低于 Flat 学习器。最大差异出现在均包含航运数据的 S3 和 S4。因此，在当前框架下，Deep 路径优于早期特征拼接。不过，优于 Flat 模型并不必然意味着优于不变预测基准。

This study extends the comparison between early-fusion approaches and multimodal models (Arevalo et al., 2017; Emami-Gohari et al., 2024) to weekly Brent forecasting. The larger Deep advantage in S3 and S4 provides further support for the value of representation-level modelling. From a spatial perspective, representation-level modelling may be particularly useful for observations distributed across networks of ports and chokepoints. Spatial relationships among shipping nodes can provide additional structural information. By contrast, treating these observations as independent tabular features does not explicitly preserve these relationships.

早期融合方法将异质预测变量合并到同一个特征空间中，而多模态模型则在融合之前保留各自表征（Arevalo et al., 2017; Emami-Gohari et al., 2024）。本研究将这一比较拓展到周度 Brent 预测。S3 和 S4 中更大的 Deep 优势进一步支持了表征级建模的价值。从空间角度来看，对于分布在港口和咽喉网络中的观测，表征级建模可能尤其有用。航运节点之间的空间关系可以提供额外的结构信息。相比之下，将这些观测作为彼此独立的表格特征处理，无法显式保留这些关系。

However, the comparisons in this study evaluate complete modelling pathways, including their encoders and fusion strategies. They therefore support the overall Deep approach but do not identify the preservation of network structure or any individual fusion operator as the cause of its better performance.

不过，本研究比较的是完整建模路径，包括其编码器与融合策略。因此，这些结果支持 Deep 整体方法，但不能将较好表现归因于网络结构的保留或某一种具体融合算子。

## 5.3 RQ3 — What does the model rely on?

## 5.3 RQ3 — 模型依赖什么？

RQ3 asked how the model uses information when a specification improves on the benchmark. The gate and SHAP results show that the internally learned weights assigned to each data source do not directly correspond to their contributions to the model output. The forecast still relies mainly on financial and EIA information, while shipping provides incremental predictive information beyond the financial inputs. Although shipping has a relatively small SHAP contribution, gated Deep S3 still records a small positive $\Delta\mathrm{RMSE}$. This suggests that a modality can add predictive value even if it accounts for only a small share of the model’s output.

RQ3 询问：当某一设定优于基准时，模型如何使用信息。门控与 SHAP 结果表明，模型内部为各数据源学习得到的权重，并不直接对应这些数据源对模型输出的贡献。预测仍主要依赖金融和 EIA 信息，而航运则在金融输入之外提供增量预测信息。虽然航运的 SHAP 贡献相对较小，但门控 Deep S3 仍录得小幅正 $\Delta\mathrm{RMSE}$。这表明，即使某一模态仅占模型输出的较小份额，仍可能带来预测价值。

This incremental role also varies with the type of disruption. Shipping becomes relatively more important during the Red Sea period, while financial inputs remain more prominent during the Russia–Ukraine window. This difference is consistent with the nature of the two disruptions. Transport-specific disruptions may increase the relevance of maritime activity, whereas broader geopolitical shocks may affect Brent through several channels, including supply expectations, inventories and financial markets. The spatial pattern supports this interpretation. The model’s focus shifts from Jurong and Hormuz in earlier years to Suez, Bab el-Mandeb and the Cape route in 2024. This broadly matches the disruption and rerouting around the Red Sea. However, no single location dominates during this period. The model therefore appears to respond to changes in the geographic pattern of shipping activity rather than consistently relying on the same chokepoints.

这种增量作用也会随扰动类型而变化。在红海时期，航运相对更重要；而在俄乌窗口内，金融输入仍然更为突出。这种差异与两类扰动的性质相符。与运输直接相关的扰动可能会提高航运活动的相关性，而更广泛的地缘政治冲击则可能通过多个渠道影响 Brent，包括供应预期、库存和金融市场。空间格局也支持这一解释。模型的关注从较早年份的裕廊和霍尔木兹，转向 2024 年的苏伊士、曼德海峡和好望角航线。这一变化与红海周边的扰动和改道大体一致。不过，在这一时期并没有任何单一地点占据主导。因此，模型似乎是在响应航运活动地理格局的变化，而不是持续依赖同一组咽喉。

Attention and gate weights describe operations within fitted models rather than causal relationships. SHAP attributes the model’s Brent predictions rather than explaining the causes of price movements. Therefore, the temporal and spatial analysis can highlight periods and transport corridors that deserve further study, but it should not be interpreted as a direct signal for policy action.

注意力和门控权重描述的是已拟合模型内部的运算，而不是因果关系。SHAP 归因于模型对 Brent 的预测，而不是解释价格变动的原因。因此，时间和空间分析可以突出值得进一步研究的时期和运输走廊，但不应将其解读为政策行动的直接信号。

## 5.4 Implications

## 5.4 启示

Model choice should be aligned with the structure of the input data. Relational data, such as shipping networks, capture relationships among observations that may carry useful information. Preserving this structure before fusion may therefore be more appropriate than direct feature concatenation. More generally, new data sources should be evaluated under a common out-of-sample framework. Their value should be compared both against a model using only established predictors and against a simple benchmark forecast. The first comparison tests whether the new data add information beyond conventional inputs, while the second tests whether the full modelling system improves on a basic forecasting rule.

模型选择应与输入数据的结构相匹配。关系型数据（如航运网络）刻画观测之间可能携带有用信息的关系。因此，在融合之前保留这一结构，可能比直接进行特征拼接更合适。更一般地，新数据源应在共同的样本外框架下评价。其价值应同时对照仅使用既有预测变量的模型，以及简单的基准预测。前者检验新数据是否在传统输入之外提供信息，后者检验完整建模系统是否优于一条基本预测规则。

These comparisons establish predictive value rather than causal explanation. The models show which information is useful for one-week-ahead prediction and help identify periods or parts of the supply network that deserve further analysis. However, they do not explain the mechanisms that cause oil prices to move. Their main value is therefore to integrate and assess predictive information from different sources, thereby supporting further analysis of oil-market conditions.

这些比较确立的是预测价值，而不是因果解释。模型显示哪些信息有助于提前一周预测，并帮助识别值得进一步分析的时期或供应网络环节。然而，它们并不解释导致油价变动的机制。因此，其主要价值在于整合并评估来自不同来源的预测信息，从而支持对油市状况的进一步分析。

This distinction also means that a data source should not be judged by forecasting performance alone. Spatial data should be assessed according to how well their scale, frequency and structure match the analytical task. Remote sensing describes conditions at selected facilities, so it may be more suitable for monitoring facility activity or regional production. Shipping data capture flows across connected ports, chokepoints and routes, and therefore better reflect disruptions and adjustments across the global oil supply network. Better observation of such physical activity may be useful for applications such as energy-security monitoring, trade planning and inflation-sensitive fiscal management, but it does not necessarily improve one-week-ahead Brent forecasts. More broadly, this study suggests that the value of a data source depends on the task for which it is used. A data source may be valuable for one task, such as forecasting, monitoring, diagnostic analysis or risk detection, without being equally valuable for others. In particular, its monitoring and predictive value should be assessed separately.

这一区分也意味着，不应仅凭预测表现来判断某一数据源。空间数据应根据其尺度、频率和结构与分析任务的匹配程度进行评价。遥感描述所选设施的状况，因此可能更适合监测设施活动或区域生产。航运数据刻画相互连接的港口、咽喉和航线之间的流动，因此更能反映全球石油供应网络中的中断和调整。对这类实物活动进行更好的观测，可能有助于能源安全监测、贸易规划和对通胀敏感的财政管理等应用，但并不一定能够改善提前一周的 Brent 预测。更广泛地说，本研究表明，一种数据源的价值取决于其所服务的任务。某类数据可能对某一任务有价值，例如预测、监测、诊断分析或风险识别，但并不一定对其他任务同样有价值。尤其需要分别评价其监测价值与预测价值。

## 5.5 Limitations

## 5.5 局限

The sample size in this study is limited by the availability of the aligned data sources. The evaluation contains only 257 rolling out-of-sample forecasts. Deep S3 improves only slightly over M0, and the improvement is concentrated in certain periods and event windows. This suggests that its predictive value is limited and conditional rather than consistently stable. The findings are specific to one-week-ahead Brent forecasting and should not be generalised to other forecast horizons.

本研究的样本量受对齐后数据源可得性的限制。评价仅包含 257 个滚动样本外预测。Deep S3 相对 M0 仅有小幅改善，且改善集中在部分时期和事件窗口。这表明其预测价值有限且有条件，而不是持续稳定。研究发现仅适用于提前一周的 Brent 预测，不应推广到其他预测期限。

The study applies a range of methods to reduce information leakage, including publication lags and as-of alignment, but limitations in data availability and preprocessing remain. The study uses revised historical series rather than real-time vintages and therefore does not fully reproduce the information available at each forecast origin. Monthly Global Fishing Watch data and remote-sensing inputs are carried forward weekly between releases, limiting their ability to capture short-lived changes. Some variables are also indirect proxies, while preprocessing and representation choices such as missing-value treatment and the use of fixed Prithvi-EO-2.0 embeddings introduce additional assumptions. The observed model performance therefore cannot be attributed to the source data alone.

本研究采用了多种方法减少信息泄漏，包括发布时间滞后和 as-of 对齐，但数据可得性与预处理方面仍存在局限。本研究使用修订后的历史序列，而非实时 vintage，因此无法完全复现各预测起点当时可获得的信息。月度 Global Fishing Watch 数据与遥感输入在两次发布之间按周向前沿用，因而捕捉短暂变化的能力有限。部分变量也属于间接代理，而缺失值处理以及使用固定 Prithvi-EO-2.0 嵌入等预处理与表征选择也引入了额外假设。因此，观察到的模型表现不能仅归因于源数据本身。

The spatial findings are conditional on the coverage and construction of a purposively selected network of 11 sites and 6 chokepoints. The site sample is weighted towards Gulf export and transshipment nodes and Asian import and refining hubs, although it also includes Rotterdam and Houston. No site-level AOIs are located in Russia or West Africa. These omissions partly reflect the difficulty of obtaining consistent cross-source data and usable satellite imagery throughout the study period. At the same time, using image availability as a site-selection criterion may introduce geographic bias. The importance assigned to individual nodes by the model therefore reflects dependence within this constructed network rather than a comprehensive ranking of global oil infrastructure.

空间发现取决于有目的选取的 11 个站点与 6 个咽喉所构成网络的覆盖范围与构建方式。站点样本偏向海湾出口与中转节点以及亚洲进口与炼化枢纽，不过也包括鹿特丹和休斯顿。俄罗斯或西非没有站点级 AOI。这些遗漏部分反映了在整个研究期内获取一致的跨源数据与可用卫星影像的困难。同时，将影像可得性作为站点选择标准，也可能引入地理偏差。因此，模型赋予各节点的重要性反映的是在这一构建网络内部的依赖，而不是对全球石油基础设施的全面排名。

The comparison between the Flat and Deep families evaluates differences between complete modelling pathways rather than differences in any single model component. The two pathways differ in model class, data representation and fusion approach. For example, the Deep shipping pathway additionally introduces an explicit graph structure. The remote-sensing inputs also differ, as VIIRS night-time-light data are included in the Flat pathway but not in the Deep pathway. Therefore, the observed performance differences cannot be clearly attributed to any single model architecture or to the gated-fusion mechanism itself.

Flat 与 Deep 模型族的比较评估的是完整建模路径之间的差异，而不是单一模型组件的差异。两条路径在模型类别、数据表征和融合方式上均存在差异。例如，Deep 航运路径额外引入了显式的图结构。遥感输入也不同：VIIRS 夜间灯光纳入 Flat 路径，但未纳入 Deep 路径。因此，观察到的表现差异不能明确归因于某一种模型架构或门控融合机制本身。

## 5.6 Future research

## 5.6 未来研究

A key next step is to determine whether the value of shipping data remains stable over longer periods and across a wider oil-transport network. A longer-term evaluation could use archived data releases and include additional regions such as Russian Baltic and Black Sea ports, West African loading areas and Latin American exporters. This would provide evidence from a longer time period and broader geographic coverage. The SAR-based vessel-detection layer used in this study could also be extended to regions where AIS coverage is less complete, particularly those with substantial dark-fleet activity.

关键的下一步是确定航运数据的价值能否在更长时间范围和更广泛的石油运输网络中保持稳定。更长期的评价可以使用存档数据发布，并纳入更多地区，例如俄罗斯波罗的海和黑海港口、西非装载区以及拉丁美洲出口地区。这样可以获得更长时间范围和更广地理覆盖下的证据。本研究所用的基于 SAR 的船舶检测层，也可扩展到 AIS 覆盖较不完整的地区，尤其是暗船活动较多的区域。

The transferability of this framework could be assessed by applying it to other forecasting tasks and markets. The same Flat–Deep comparison could be applied to other energy commodities and markets, such as WTI crude oil, liquefied natural gas and natural gas. It could also be extended to grain and metal markets to test whether modelling network structure remains useful outside the energy sector. Remote sensing should be evaluated for targets that better match its spatial and temporal scale, such as facility activity, regional production or longer-term price movements. Where possible, future studies should also compare Flat and Deep remote-sensing models using the same underlying inputs. This would make it easier to separate the effect of model architecture from differences in data representation.

可通过将这一框架应用于其他预测任务和市场，检验其可迁移性。相同的 Flat–Deep 比较可以应用于其他能源品种和市场，例如 WTI 原油、液化天然气和天然气。也可以扩展到粮食和金属市场，以检验网络结构建模在能源领域之外是否仍然有用。遥感则应针对与其空间和时间尺度更匹配的目标进行评价，例如设施活动、区域产量或更长期的价格变动。在条件允许时，未来研究还应使用相同底层输入比较 Flat 和 Deep 遥感模型。这样更容易将模型架构的影响与数据表征差异分开。

Beyond forecast accuracy, future work should examine the practical value of these forecasts. For public-sector applications, probabilistic forecasts and scenario ranges could be compared with existing financial and EIA monitoring. This would help assess whether shipping information improves the timing and focus of energy-security monitoring, trade planning or inflation-sensitive budgeting. Evaluating policy actions such as strategic reserve releases, sanctions or fiscal responses would require more than forecasting alone. Forecasting models would need to be combined with causal or decision-analysis methods to estimate how these actions affect prices, supply and policy costs. Model-attribution results should therefore be treated as diagnostic evidence that can inform further policy analysis, rather than as a direct basis for policy action.

除预测精度外，未来工作还应考察这些预测的实际价值。对于公共部门应用，可以将概率预测和情景范围与现有金融和 EIA 监测进行比较。这有助于判断航运信息能否改善能源安全监测、贸易规划或对通胀敏感的预算管理中的时机与关注重点。评价战略储备释放、制裁或财政应对等政策行动，仅靠预测是不够的。还需要将预测模型与因果或决策分析方法结合，以估计这些行动如何影响价格、供应和政策成本。因此，模型归因结果应作为可支持进一步政策分析的诊断性证据，而不是政策行动的直接依据。