# Chapter 5 — Discussion
# 第 5 章 — 讨论

This chapter interprets the results against RQ1–RQ3. It does not repeat the tables of Chapter 4. The emphasis is on why patterns arise, what they mean for oil-price forecasting and multimodal learning, and where the evidence remains weak.

本章对照 RQ1–RQ3 解释结果，不重复第 4 章表格。重点在于模式为何出现、对油价预测与多模态学习意味着什么，以及证据在何处仍然薄弱。

## 5.1 Answer to RQ1 — Do alternative data improve forecasting?
## 5.1 回答 RQ1 — 替代数据是否改善预测？

**Remote sensing (M2) is limited.** Across flat and deep arms, finance+RS configurations remain far from beating M0 and are often no better than finance alone in practical RMSE terms. Several mechanisms help explain this. Site-level optical and night-light proxies are indirect; they may track activity or observability rather than next-week price. Cloud cover, monthly revisit and within-site inertia weaken the weekly signal. Exploratory correlations are weak and sometimes coincident rather than leading. Water-masking can strengthen nested increments, which confirms noise in the optical channel, but strengthening a nested test is not the same as producing absolute skill.

**遥感（M2）提升有限。** 在扁平与深度分支中，金融+遥感配置都远未击败 M0，实用 RMSE 上也常常不优于仅金融。若干机制有助于解释这一点。站点级光学与夜间灯光代理是间接的，可能追踪活动或可观测性，而非下一周价格。云量、月度重访与站内惯性削弱周度信号。探索性相关较弱，有时是同期而非领先。水体掩膜可增强嵌套增量，这确认了光学通道中的噪声，但增强嵌套检验并不等于产生绝对 skill。

**Shipping (M3) is more informative.** Shipping provides the most consistent alternative-data signal. In the flat arm, XGBoost finds a highly significant nested increment of M3 over M1; in the deep arm, adding a shipping representation to finance is the cleanest nested gain and can produce small positive skill. Economically, chokepoint tanker flows and port activity are closer to physical trade and disruption than monthly site indices. Empirically, however, shipping remains a noisy proxy and may partly respond to prices, so reverse causality cannot be ruled out.

**航运（M3）更有信息量。** 航运提供最一致的替代数据信号。扁平分支中，XGBoost 发现 M3 相对 M1 高度显著的嵌套增量；深度分支中，向金融加入航运表示是最干净的嵌套收益，并可产生小幅正 skill。从经济上看，咽喉油轮流与港口活动比月度站点指数更接近实物贸易与冲击。但经验上航运仍是有噪声的代理，并可能部分对价格作出反应，因此不能排除反向因果。

**Does M4 add complementary information?** Not clearly. Full-modality models do not dominate finance+shipping. In several deep specifications, adding RS to M3 leaves performance unchanged or worse, suggesting redundancy or noise absorption rather than complementary signal. Multimodal completeness is therefore not automatically desirable.

**M4 是否带来互补信息？** 并不清楚。全模态模型并未主导金融+航运。在若干深度设定中，向 M3 加入遥感使表现不变或变差，暗示冗余或噪声吸收，而非互补信号。因此，多模态“齐全”并不自动可取。

**Why beat M1 yet lose to M0?** Nested significance and absolute skill answer different questions. Alternative data can reduce error relative to a misspecified or incomplete finance-only learner without overturning the persistence embodied in the no-change forecast. At a weekly horizon, much of next week’s price is already in this week’s price. Statistical significance of an increment should not be confused with economic or operational superiority over a simple benchmark.

**为何能胜过 M1 却输给 M0？** 嵌套显著性与绝对 skill 回答不同问题。替代数据可以相对设定不当或不完整的仅金融学习器降低误差，却不足以推翻无变化预测所体现的持久性。在周度期限上，下一周价格的很大一部分已包含在本周价格中。增量的统计显著，不应被等同于相对简单基准的经济或操作优势。

## 5.2 Answer to RQ2 — Is representation-level fusion better than flat concatenation?
## 5.2 回答 RQ2 — 表示级融合是否优于扁平拼接？

Paired comparisons show a **conditional**, not universal, advantage for deep models. The clearest gains appear when shipping is included: representing maritime activity as a graph/temporal encoder helps more than flattening dozens of collinear shipping columns into a table. Finance-only deep models modestly improve on weak flat finance learners, but still fail to beat M0. Finance+RS pairs remain weak in both architectures, so representation learning does not magically create signal where the underlying proxy is weak.

配对比较显示，深度模型的优势是**有条件的**，而非普遍的。最清晰的收益出现在纳入航运时：将海运活动表示为图/时序编码器，比把数十个共线航运列压成一张表更有帮助。仅金融的深度模型相对较弱的扁平金融学习器有适度改善，但仍未击败 M0。金融+遥感配对在两类架构中都弱，因此表示学习无法在底层代理本身薄弱处凭空创造信号。

What, then, drives deep gains where they exist? Likely a combination of non-linear temporal modelling and modality-specific structure preservation — especially for shipping — rather than fusion sophistication alone. Cross-attention can raise the ceiling in shipping-inclusive settings, but gated fusion is more stable across seeds. Encoder concatenation of three modalities is clearly worse than gated fusion, reinforcing that naive joining of representations can recreate the pathology of flat early fusion.

那么，存在深度收益时由什么驱动？更可能是非线性时序建模与模态专属结构保留——尤其对航运——的组合，而非融合机制本身的精巧。交叉注意力可在含航运设定中抬高上限，但门控融合跨种子更稳定。三模态的编码器拼接明显差于门控融合，说明对表示的朴素拼接可能重现扁平早融合的病理。

Is the added complexity worth it? Only selectively. For dissertation and research purposes, the deep arm is justified because it answers RQ2 under a fair protocol. For practical weekly forecasting, the gains over M0 remain small, and a random walk is still hard to displace. Complexity is warranted as an empirical test of fusion design, not as an automatic operational upgrade.

增加的复杂度是否值得？只在选择性意义上值得。就学位论文与研究目的而言，深度分支是合理的，因为它在公平协议下回答 RQ2。就实用周度预测而言，相对 M0 的收益仍然很小，随机游走仍难被取代。复杂度作为融合设计的实证检验是正当的，但不能自动视为运营升级。

## 5.3 Answer to RQ3 — Interpretable modality and spatial dependence?
## 5.3 回答 RQ3 — 是否揭示可解释的模态与空间依赖？

Interpretability results are coherent with the performance ranking. Shipping receives the largest share of flat SHAP mass in M4, and deep gates/attentions likewise emphasise shipping over remote sensing. This supports the reading that maritime physical-flow proxies are the main alternative-data channel in this weekly setting.

可解释性结果与绩效排序一致。扁平 M4 中航运占据最大 SHAP 份额，深度门控/注意力同样更强调航运而非遥感。这支持如下解读：在本周度设定中，海运实物流代理是主要的替代数据通道。

There is suggestive concentration on major chokepoints and export-linked nodes during stressed periods, but the evidence should not be over-interpreted. Gate weights and SHAP values describe what the fitted model uses. They do not identify causal supply shocks, nor do they prove that a high-weight feature “moved” the oil price. The dissertation therefore treats interpretability as complementary diagnosis: useful for checking economic plausibility and for explaining incremental gains over M1, not for causal storytelling.

在压力时期，注意力对主要咽喉与出口相关节点存在提示性集中，但证据不宜过度解读。门控权重与 SHAP 值描述的是拟合模型使用了什么，并不能识别因果供给冲击，也不能证明高权重特征“推动”了油价。因此本文将可解释性视为互补诊断：有助于检查经济合理性，并解释相对 M1 的增量收益，而不是用于因果叙事。

## 5.4 Implications
## 5.4 研究含义

**Theoretical and empirical.** The results underline two evaluation standards that multimodal commodity studies often blur: nested incremental value and absolute skill against a strong no-change benchmark. They also show that fusion design matters most when a modality has structure that flat tables discard. Where the proxy is weak, neither flat nor deep fusion rescues it.

**理论与实证。** 结果强调多模态大宗商品研究常混淆的两套评估标准：嵌套增量价值，以及相对强无变化基准的绝对 skill。结果还表明，当某模态具有扁平表会丢弃的结构时，融合设计最重要。若代理本身薄弱，扁平或深度融合都救不了它。

**Practical.** For energy-market monitoring, shipping indicators are the more promising open alternative-data source at a weekly horizon. Remote-sensing site indices remain useful for mechanism exploration and robustness checks, but should not be assumed to deliver forecast gains. Any operational use should retain the random walk as a live competitor and treat small positive skill with caution.

**实践。** 对能源市场监测而言，在周度期限上航运指标是更有希望的开放替代数据源。遥感站点指数对机制探索与稳健性检查仍有用，但不应假定其带来预测收益。任何运营使用都应保留随机游走作为实时竞争者，并对小幅正 skill 保持谨慎。

A further implication concerns how multimodal results should be written up. It is tempting to summarise the project as “deep beats flat” or “alternative data work”. The evidence supports a narrower sentence: representation-level fusion helps mainly when the modality carries structured physical information that tables discard, and even then absolute gains over the random walk are small. Writing the claim at that resolution is part of the contribution.

另一含义涉及多模态结果应如何书写。很容易把项目概括为“深度优于扁平”或“替代数据有效”。证据支持更窄的句子：表示级融合主要在模态携带表格会丢弃的结构化物理信息时有帮助，即便如此，相对随机游走的绝对收益仍然很小。把主张写到这一分辨率，本身也是贡献的一部分。

## 5.5 Limitations
## 5.5 局限

The sample is weekly and relatively short once a common multimodal window is imposed. Shipping and RS proxies are noisy and may be subject to reverse causality. Frozen EO embeddings and the 17-node graph encode strong design choices; alternative graphs, sensors or backbones might change magnitudes. Some advanced deep results are seed-sensitive. Publication-lag assumptions for monthly series are conservative heuristics rather than full real-time vintage reconstructions.

样本为周度，且在施加共同多模态窗口后相对较短。航运与遥感代理有噪声，并可能存在反向因果。冻结 EO 嵌入与 17 节点图编码了较强的设计选择；替代图结构、传感器或骨干可能改变幅度。部分进阶深度结果对种子敏感。月度序列的发布滞后假设是保守启发式，而非完整实时 vintage 重建。

## 5.6 Future research
## 5.6 未来研究

Future work could extend the best shipping-inclusive models to longer histories, enrich AIS voyage graphs, stress-test missing-modality regimes more systematically, and examine other horizons or benchmarks (e.g. WTI). Regime-conditional interpretability could be formalised with pre-registered event windows. Finally, forecast combinations of the random walk with small-skill deep models may be more useful operationally than replacing the benchmark outright.

未来工作可将最佳含航运模型扩展到更长历史，丰富 AIS 航次图，更系统地压力测试缺失模态情形，并考察其他期限或基准（如 WTI）。体制条件可解释性可用预先登记的事件窗口加以形式化。最后，将随机游走与小 skill 深度模型做预测组合，在运营上可能比直接替换基准更有用。
