# Appendix B — Extra results & robustness / 附录 B — 额外结果与稳健性

> All checks share the single leakage-safe protocol (2019–2025, lookback 4,
> expanding rolling-origin, 257 common scored weeks, CW vs M1 / DM vs M0). They
> **reinforce** the main findings; they do not replace the main-analysis specs
> (§ Chapter 4). Sources are given per table so every number is traceable.
>
> 所有检查共用同一无泄漏协议（2019–2025、回看 4、扩展滚动起点、257 计分周、
> CW vs M1 / DM vs M0）。它们**加固**主结论，不替代主分析口径（见第 4 章）。
> 每表标注来源，数字可追溯。

---

## B.1 Sub-period: early (≤2022) vs late (≥2023) / 子期：早期 vs 晚期

Split at 2023-01-01 (matching `run_deep_advanced.py`). Skill vs M0 in %.
Scored offline from saved predictions with `subperiod_eval.py` →
`05_outputs/baselines/subperiod/subperiod_summary.csv`; numbers reproduce the
main pipeline exactly.

以 2023-01-01 切分（与 `run_deep_advanced.py` 一致）。skill vs M0（%）。由
`subperiod_eval.py` 离线读取预测产出，与主流程数字一致。

| Model / 模型 | full | early (≤2022) | late (≥2023) |
| --- | ---: | ---: | ---: |
| M0 (random walk) | 0 | 0 | 0 |
| M1_Flat Ridge / XGB | −2.52 / −5.22 | −2.85 / −4.51 | −1.98 / −6.37 |
| M2_Flat Ridge / XGB | −6.31 / −6.95 | −6.53 / −7.11 | −5.94 / −6.67 |
| M3_Flat Ridge / XGB | −6.71 / −6.68 | −7.22 / −6.68 | −5.87 / −6.69 |
| M4_Flat Ridge / XGB | −8.99 / −8.57 | −9.33 / −9.07 | −8.45 / −7.74 |
| M1_Deep | −2.36 | −1.33 | −4.02 |
| M_ship_GNN (shipping only) | −0.22 | −0.24 | −0.17 |
| M_rs_deep (RS only) | −2.30 | −3.07 | −1.04 |
| **M3_Deep_gated (main)** | **+0.16** | **+0.33** | **−0.13** |
| M2_Deep_gated | −2.43 | −3.06 | −1.41 |
| M4_Deep_gated | −0.67 | −1.36 | +0.47 |
| M4_Deep_Concat | −8.30 | −12.77 | −0.58 |

**Reading / 解读**：no flat model beats M0 in either sub-period. Among deep
models, M3_Deep_gated has both the largest full-sample skill (+0.16) and the
strongest early-period skill (+0.33), but is marginally negative late (−0.13);
M4_Deep_gated shows the opposite profile (−1.36 early, +0.47 late), and
cross-attention–driven M4 gains remain concentrated late. **No deep configuration
is positive in both sub-periods**, so the small full-sample gain of the main model
is not evenly distributed over time. Gated finance+shipping is retained as the
main model on full-sample skill and on the nested shipping increment (Chapter 4);
this split is reported as a limitation on how stable that gain is, not as
supporting evidence. / 两个子期内无扁平模型击败 M0。深度模型中 M3_Deep_gated 的全
样本 skill 最高（+0.16）、早期最强（+0.33），但晚期略为负（−0.13）；M4_Deep_gated
恰好相反（早期 −1.36、晚期 +0.47）。**没有任何深度配置在两个子期都为正**，故主模型
的小幅全样本增益在时间上分布不均。仍以 gated 金融+航运为主模型，依据是全样本 skill
与航运嵌套增量（第 4 章）；此处子期划分作为该增益稳定性的局限说明，而非支持性证据。

---

## B.2 Deep fusion matrix (3 modality combos × 3 fusions) / 深度融合矩阵

seed 42, lookback 4, 257 weeks. Source: `run_deep_fusion_matrix.py` →
`05_outputs/baselines/Deep/_cross/deep_fusion_matrix.csv`. Skill vs M0 in %.

| Combo / 组合 | Concat | Gated | Cross-Attn | CW p vs M0 (best) |
| --- | ---: | ---: | ---: | --- |
| **M3_Deep** (fin+ship) | −0.22 | **+0.16** | **+1.01** | xattn 0.029 ✅ |
| M2_Deep (fin+rs) | −1.93 | −2.43 | −5.89 | — |
| M4_Deep (fin+rs+ship) | −8.30 | −0.67 | +0.33 | xattn 0.026 ✅ |

**Reading / 解读**：M0 is cleared only where shipping is present, and only under
adaptive fusion: M3 clears it under gated and cross-attention, M4 only under
cross-attention, and M2 (fin+rs) never. Plain concatenation clears M0 in no combo,
so the gain depends on weighting the modalities rather than on stacking them.
Cross-attention gives the single-seed peak but is less stable across seeds (see
B.4), so gated is the main reported fusion. / 仅在含航运且使用自适应融合时越过 M0：
M3 在门控与交叉注意力下越过，M4 仅在交叉注意力下越过，M2（金融+遥感）从不越过。简单
拼接在任一组合下都未越过 M0，说明增益来自对模态加权而非堆叠。交叉注意力单 seed 峰值
最高但跨 seed 不稳（见 B.4），故主报告用 gated。

---

## B.3 Flat robustness / 扁平稳健性

### B.3.1 M2 leave-one-AOI-out (LOAO) / 遥感留一站点

Source: `05_outputs/baselines/Flat/M2_Flat/baseline_metrics_anom_loao.csv` (+ full
per-AOI dRMSE in `baseline_loao_anom.csv`). Removing any single AOI leaves the M2
result essentially unchanged (|dRMSE| small, no single site drives a positive
contribution), i.e. the weak RS signal is diffuse rather than one-site-driven.

移除任一 AOI 后 M2 结果基本不变（|dRMSE| 小，无单站带来正贡献）——遥感弱信号弥散、
非单站驱动。

### B.3.2 M3 leave-one-channel-out (LOCHO) / 航运留一数据源

seed 42, 257 weeks. Source:
`05_outputs/baselines/Flat/M3_Flat/robustness_m3_summary.csv`. Skill vs M0 (%)
and CW p vs M1 (nested increment) for XGB.

| Arm / 臂 | XGB skill vs M0 | XGB CW p vs M1 |
| --- | ---: | --- |
| full (113 cols, main) | −6.68 | **0.0002** ✅ |
| core (38) | −7.81 | 0.096 |
| portwatch-only | −4.91 | **0.0003** ✅ |
| gfw-only | −5.70 | 0.047 ✅ |
| gfw-presence | −4.99 | 0.039 ✅ |
| gfw-aggregate | −4.62 | 0.094 |
| tanker-only | −4.60 | **0.0018** ✅ |

**Reading / 解读**：the nested shipping increment over M1 (CW p) is significant
across several channel subsets — strongest for tanker/PortWatch flows — so the
M3 signal is not an artefact of one data source, even though no flat arm beats M0
in absolute RMSE. / 航运相对 M1 的嵌套增量在多个子源上显著（tanker/PortWatch 最
强），说明 M3 信号非单一来源假象；但绝对 RMSE 上无扁平臂击败 M0。

### B.3.3 M2 water-masked RS variant / 遥感水体掩膜变体

Source: `baseline_metrics_anom_watermask.csv`. Masking water pixels lifts the M2
nested increment (XGB CW p vs M1 = 0.028 ✅ vs 0.085 un-masked) but M2 still does
not beat M0 (skill −6.3%). De-noising only makes RS marginally significant → RS
value is limited → motivates modality-aware fusion (RQ2).

去水面像素使 M2 嵌套增量略增（XGB CW p vs M1 = 0.028，未掩膜 0.085），但 M2 仍不
击败 M0（skill −6.3%）。去噪也才勉强显著 → 遥感价值有限 → 支撑「需模态感知融合」。

### B.3.4 M4 leave-one-modality-out (LOMO) / 全模态留一模态

Source: `Flat/M4_Flat/robustness_m4_summary.csv`. Dropping RS from M4 (i.e. M1+M3)
keeps the significant nested increment (XGB CW p vs M1 = 0.0002 ✅), whereas the
full M4 adds RS without accuracy gain — flat multi-modal concatenation cannot
improve accuracy and significance together.

从 M4 去掉遥感（即 M1+M3）保留显著嵌套增量（XGB CW p vs M1 = 0.0002），而完整 M4
加入遥感并无精度增益——扁平多模态拼接无法同时改善精度与显著性。

---

## B.4 Deep multi-seed & sweeps / 深度多 seed 与扫描

Source: `04_code/scripts/deep/run_deep_sweep.py` → `deep_sweep_summary.csv` (seeds 42, 1, 2;
lookback 4; d 32).

| Config / 配置 | skill vs M0 (3 seeds) | Note / 解读 |
| --- | --- | --- |
| **finship gated** | −0.50% ± 0.80 | **best 3-seed mean**; spread far tighter than cross-attention → main model |
| m4rep gated | −0.93% ± 0.29 | tightest spread, but centred well below M0; adding RS gives no gain |
| m4 xattn | −1.85% ± **2.80** | seed 42 best (+0.33%) but seed 2 collapses to −5.01% → not main |

All three means are below M0, so the positive headline figures reported in
Chapter 4 (finship gated +0.16%, cross-attention +1.01%) are seed-42 outcomes
rather than expected skill: averaged over seeds, no Deep configuration beats the
no-change benchmark. Gated finance+shipping is selected on the best seed-averaged
mean and on its low dispersion relative to cross-attention, not on a claim of
positive expected skill. This is the sharpest single limitation on the Deep
results and is carried into Chapter 5.

三个配置的均值均低于 M0，故第 4 章报告的正向数字（finship gated +0.16%、交叉注意力
+1.01%）是 seed=42 的结果而非期望 skill：跨 seed 平均后，没有任何 Deep 配置击败无变化
基准。选定门控金融+航运的依据是其跨 seed 均值最高、且离散度远小于交叉注意力，而非
「期望 skill 为正」。这是 Deep 结果最主要的局限，并在第 5 章延续说明。

Single-seed (42) hyper-sweep, finship gated: lookback 4/8/12 × d 32/64 →
lb 8 d 32 best (+0.25%) > lb 4 d 32 (+0.16%) > lb 12 d 32 (−0.08%); **d 64 always
worse** (−0.76 / −1.11 / −1.11; short weekly sample). Halving encoder depth
(1 GAT + 1 TCN layer instead of 2 + 2) also degrades skill to −1.26%, so the main
setting is not simply over-parameterised. Main model stays locked at lookback 4,
d 32 for flat protocol parity.

单 seed=42 超参扫描（finship gated）：lb 8/d 32 最优（+0.25%）> lb 4/d 32（+0.16%）>
lb 12/d 32（−0.08%）；**d 64 一律更差**（−0.76 / −1.11 / −1.11；短周度样本）。将编码器
层数减半（GAT 与 TCN 各 1 层，而非各 2 层）同样恶化至 −1.26%，说明主设定并非单纯过参数化。
主模型仍锁 lb 4、d 32 以对齐扁平协议。

---

## B.5 Other robustness (documented, not tabulated here) / 其他稳健性

- **Lookback sweep** L1/8/12 per layer / 各层回看扫描 — `sweep_m*` + `deep_sweep`.
- **C2 dimensionality reduction** (PCA / ElasticNet) for M2 / M2 降维对照 —
  `c2_summary.csv`; increment not pure over-fitting.
- **feature-mode = returns** stationarised variant / 平稳化变体 — numerical robustness.
- **min_train = 78 longer window** / 更长测试窗 — Appendix-level; main protocol
  keeps min_train = 104.
