# Chapter 5 — Discussion *(~1,600–1,900)*

# 第 5 章 — 讨论 *(约 1,600–1,900 词)*

## 5.1 RQ1 — Do alternative data help?

## 5.1 RQ1 — 另类数据是否有用？

RQ1 asked whether remote sensing and shipping add out-of-sample value over financial time series and M0. The answer depends on the contrast, and that dependence is itself the finding. No Flat model beats M0. This aligns with the oil-forecasting literature that treats the no-change forecast as a hard short-horizon bar (Alquist, Kilian and Vigfusson, 2013). Within Flat, shipping still helps relative to M1 while remote sensing does not; sparse or cleaned RS variants do not overturn that ranking.

RQ1 问遥感与航运是否在金融时序与 M0 之上带来样本外价值。答案取决于对照，而这种依赖本身就是发现。无 Flat 模型打过 M0。这与把不变预测视为短期限高门槛的油价文献一致（Alquist, Kilian and Vigfusson, 2013）。Flat 族内航运相对 M1 仍有帮助而遥感没有；稀疏或清洗遥感变体不推翻该排序。

Under Deep, finance plus shipping yields only a small positive skill versus M0, and adding remote sensing often brings no further gain. Shipping is the more informative alternative modality here, but the absolute margin is narrow. That differs from much of the AIS and satellite oil literature in Chapter 2, which more often shows that vessel or Earth-observation proxies carry information about trade, demand or infrastructure without testing one-week-ahead Brent skill against both a financial baseline and M0. The results do not deny physical-market information in those proxies; they show that, at the weekly Brent horizon under leakage-safe evaluation, such information does not automatically become large absolute forecast gains. Nested increments and absolute skill must be reported together.

在 Deep 下，金融加航运相对 M0 仅有小幅正 skill，再加遥感往往无进一步增益。此处航运是更有信息量的另类模态，但绝对幅度很窄。这不同于第 2 章许多 AIS 与卫星石油研究：它们更多证明船舶或对地观测代理含贸易、需求或基础设施信息，却较少同时相对金融基线与 M0 检验提前一周 Brent skill。本结果不否认这些代理的实物市场信息；它表明在周度 Brent 与无泄漏评估下，此类信息不会自动变成大幅绝对预测增益。嵌套增量与绝对 skill 必须一并报告。

## 5.2 RQ2 — Does representation-level fusion beat flat fusion?

## 5.2 RQ2 — 表示级融合是否优于扁平融合？

RQ2 asked whether modality-aware representation-level fusion outperforms flat feature fusion when data and protocol are fixed. Deep outperforms Flat most clearly once shipping enters; Deep on financial time series alone yields limited gains. The architecture gap opens mainly where relational structure can be preserved—here a shipping network.

RQ2 问在数据与协议固定时表示级融合是否优于扁平特征融合。Deep 在航运进入后相对 Flat 优势最清晰；仅金融时序上 Deep 增益有限。架构差距主要在有关系结构可保留处打开——此处是航运网络。

This sits between two literatures. Flat early fusion remains the convenient default for classical high-dimensional oil-price learners, but does not keep network structure. Gated and modality-aware models (Arevalo et al., 2017; Gohari et al., 2024) show that separate streams can matter, yet those studies are not weekly Brent designs with AIS–PortWatch graphs and a no-change price benchmark. The paired results therefore complement both: preserving shipping structure can help under matched sets, while the same Deep machinery does not rescue remote sensing. Cross-attention can raise a single-seed ceiling but is less stable across seeds than gated fusion, so the architectural claim remains conditional.

这介于两支文献之间。扁平早融合仍是高维油价学习器的便捷默认，但不保留网络结构。门控与模态感知模型（Arevalo et al., 2017；Gohari et al., 2024）表明分流通路可以有用，但并非带 AIS–PortWatch 图与不变价格基准的周度 Brent 设计。配对结果因此互补二者：匹配集下保留航运结构可有帮助，但同一 Deep 机制不能挽救遥感。交叉注意力可提高单种子上限，跨 seed 稳定性不如门控，故架构主张仍是有条件的。

## 5.3 RQ3 — What does the model rely on when value exists?

## 5.3 RQ3 — 有价值时模型依赖什么？

RQ3 asked whether modality-level interpretability reveals which signals the model relies on. Evidence is restricted to specifications with predictive value, mainly Deep M3. Mean gates put substantial weight on financial time series and shipping, but week-level shipping-gate paths are unstable across seeds. Only the Russia–Ukraine announcement window shows a cross-seed co-rising shipping gate; the Red Sea window is not locked. Hormuz is the only chokepoint focus in the top set for all three seeds.

RQ3 问模态级可解释性能否揭示模型依赖哪些信号。证据限于已有预测价值的设定，主要为 Deep M3。均值门控对金融时序与航运权重可观，但航运门控周度路径跨 seed 不稳。仅俄乌公告窗显示跨 seed 同向上升；红海窗不写死。霍尔木兹是唯一 3/3 进入前列的咽喉焦点。

That reading matches a cautious view of attention and gates: such weights describe operations inside a fitted model and need not identify causal features (Jain and Wallace, 2019). It also differs from monitoring narratives that would treat a single disruption window—or one seed’s chokepoint map—as actionable evidence. The useful claim is narrower: when Deep shipping-inclusive forecasts clear M0, Hormuz is the only spatial focus stable enough for the main text, and event-window gate moves must pass a multi-seed filter.

这与对注意力与门控的谨慎立场一致：此类权重描述拟合模型内部运算，未必识别因果特征（Jain and Wallace, 2019）。也不同于把单一扰动窗或单种子咽喉图当作可行动证据的监测叙事。有用主张更窄：当含航运 Deep 预测越过 M0 时，霍尔木兹是正文唯一够稳的空间焦点；事件窗门控变动须通过多种子过滤。

## 5.4 Implications

## 5.4 启示

For evidence design, a shared evaluation protocol matters as much as a new fusion module. Nested M1 contrasts and absolute M0 contrasts jointly show that shipping can help relative to financial time series while absolute gains remain small. Gains over M0 are too small for a forecasting-breakthrough claim. Near-null Flat results are informative: early fusion of alternative data is not automatically useful.

就证据设计而言，共享评估协议与提出新融合模块同样重要。嵌套 M1 与绝对 M0 对照共同表明：航运相对金融时序可有帮助，但绝对增益仍小。相对 M0 的增益不足以支撑预测突破主张。近零的 Flat 结果本身有信息：另类数据早融合并非自动有用。

Two agendas already raised in Chapters 1–2 sharpen what the framework can and cannot do. First, oil-price surprises matter for risk management, hedging and budgeting (Chapter 1). The practical implication is not that Deep M3 should replace existing hedges, but that teams who are offered multimodal “signals” can require the same double test used here—nested gain over financial time series *and* skill versus a no-change bar—before treating those signals as decision-relevant at a weekly horizon. Second, the 2019–2025 window includes the 2022 energy crisis and later market adjustment (Chapter 1). In that setting, AIS–PortWatch and satellite products are often promoted as near-real-time monitors of physical disruption. The results caution against converting a single chokepoint map or a single disruption-window gate spike into an operational alert: only multi-seed-stable foci (here Hormuz) and filters that survive across seeds warrant discussion, and even then as model-dependence diagnostics rather than policy instruments.

第 1–2 章已提出的两项议程进一步明确该框架能做什么、不能做什么。第一，油价意外关系到风险管理、对冲与预算（第 1 章）。实践含义不是用 Deep M3 取代现有对冲，而是：当团队面对多模态“信号”时，可要求采用本文同一双重检验——相对金融时序的嵌套增益*与*相对不变预测的 skill——再把这些信号视为周度决策相关。第二，2019–2025 窗口包含 2022 年能源危机及随后调整（第 1 章）。在此背景下，AIS–PortWatch 与卫星产品常被宣传为实物扰动的近实时监测。结果提醒：不宜把单一咽喉图或单一扰动窗门控尖峰直接转成操作警报；只有跨 seed 稳定焦点（此处为霍尔木兹）及经多种子过滤的变动才值得讨论，且即便如此也是模型依赖诊断，而非政策工具。

More broadly, alternative-data and Earth-observation providers can report nested gains and M0 skill together to avoid overselling nested-only improvements. Methods researchers can reuse the matched Flat–Deep design for other commodities or horizons.

更广而言，另类数据与对地观测提供方应同时报告嵌套增益与相对 M0 的 skill，避免只推销嵌套改善。方法研究者可在其他商品或预测期复用匹配 Flat–Deep 设计。

## 5.5 Limitations

## 5.5 局限

The study uses a weekly horizon and a modest scored sample after warm-up, so small skill differences should not be over-read. Alternative-data proxies are noisy and may respond to prices as well as lead them. Frozen Earth-observation embeddings, shipping-graph construction and missingness rules affect Deep results; cross-attention is especially seed-sensitive. Matched Flat–Deep comparisons also differ in model class and capacity, so they isolate the overall modelling strategy more cleanly than a single fusion operator.

研究使用周度预测期，预热后计分样本有限，故不宜过度解读小幅 skill 差异。另类数据代理嘈杂，可能既响应也领先价格。冻结对地观测嵌入、航运图构建与缺失规则影响 Deep 结果；交叉注意力对种子尤其敏感。匹配 Flat–Deep 比较在模型类别与容量上亦不同，故更干净分离的是整体建模策略，而非单一融合算子。

## 5.6 Future research

## 5.6 未来研究

Future work can extend the best Deep specifications to longer history and more seeds; enrich the shipping graph and missing-modality stress tests; and apply the same Flat–Deep protocol to other horizons or related energy commodities. That would test whether the shipping-centred, modest-positive-skill pattern survives outside this weekly Brent window.

未来研究可将最佳 Deep 设定扩展到更长历史与更多种子；丰富航运图与缺失模态压力测试；并将同一 Flat–Deep 协议用于其他预测期或相关能源商品，以检验以航运为中心、小幅正 skill 的模式能否在本周度 Brent 窗口之外成立。
