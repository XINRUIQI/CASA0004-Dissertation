# Chapter 6 — Conclusion
# 第 6 章 — 结论

## 6.1 Summary of findings
## 6.1 主要发现总结

This dissertation compared flat feature fusion and modality-aware representation-level fusion for weekly Brent forecasting under one leakage-safe protocol.

本文在统一无泄漏协议下，比较了周度 Brent 预测中的扁平特征融合与模态感知表示级融合。

On **RQ1**, remote sensing adds at most limited nested information and does not produce absolute skill over the random walk. Shipping is the more informative alternative modality: it improves on finance-only models in nested tests and, in selected deep specifications, can yield small positive skill. Full multimodal models do not clearly dominate finance+shipping. Across both architectures, beating M0 remains difficult.

就 **RQ1** 而言，遥感至多带来有限嵌套信息，且无法相对随机游走产生绝对 skill。航运是更有信息量的替代模态：在嵌套检验中改善仅金融模型，并在选定深度设定中可产生小幅正 skill。全模态模型并未清晰主导金融+航运。两类架构下，击败 M0 仍然困难。

On **RQ2**, deep models outperform their flat counterparts in selected multimodal settings, particularly when shipping is included and represented with modality-specific structure. The advantage is conditional, not uniform across all information sets and seeds.

就 **RQ2** 而言，深度模型在选定多模态设定中优于对应扁平模型，尤其在纳入航运并以模态专属结构表示时。该优势是有条件的，并非在所有信息集与种子上一致。

On **RQ3**, interpretability diagnostics consistently emphasise shipping over remote sensing. These attributions help explain incremental gains over M1, but they identify model dependence rather than causal price drivers.

就 **RQ3** 而言，可解释性诊断一致强调航运而非遥感。这些归因有助于解释相对 M1 的增量收益，但识别的是模型依赖，而非因果价格驱动因素。

## 6.2 Contributions
## 6.2 贡献

The contribution is empirical and integrative: a unified M0–M4 ladder; paired Flat–Deep comparisons; joint reporting of nested increments and absolute skill; and a disciplined interpretability rule that privileges models with predictive value while still allowing limited explanation of significant gains over M1.

贡献是实证与集成性的：统一的 M0–M4 阶梯；配对的 Flat–Deep 比较；同时报告嵌套增量与绝对 skill；以及有纪律的可解释性规则——优先具有预测价值的模型，同时仍允许对相对 M1 的显著收益作有限解释。

## 6.3 Final conclusion
## 6.3 最终结论

Strong baselines come first. Alternative data and deeper fusion can matter, but mainly through structured shipping information and careful evaluation design. Small gains over the random walk should be stated cautiously; significant improvement over a financial baseline is not by itself a licence to claim superior absolute forecasting skill.

强基线优先。替代数据与更深融合可以起作用，但主要通过对结构化航运信息的利用与谨慎的评估设计。相对随机游走的小幅收益应谨慎表述；相对金融基线的显著改进本身，不足以宣称具有更优的绝对预测能力。
