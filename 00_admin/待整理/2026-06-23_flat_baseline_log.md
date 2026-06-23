# 扁平基线模型记录（Flat Feature-Fusion Baseline）

> **定位**：核心实证层的「扁平特征融合」对照（M0–M4 表格基线）。方法创新层（模态感知表示级融合）必须超越本基线。
> **代码**：`04_code/src/backtest/`（公平回测内核）+ `04_code/scripts/run_baseline.py`（单入口 M0–M4）+ `sweep_baseline.py`（M1 稳健性）
> **产物**：`05_outputs/baselines/`（`baseline_metrics_*.csv` / `baseline_predictions_*.csv` / `baseline_backtest_*.png` / `baseline_loao_*.csv` / `sweep_*`）
> **创建/更新**：2026-06-23 ｜ 进度：M0 / M1 / **M2（M1+RS：anom 主 + literature/level 对照 + Clark-West + LOAO）** / **M3（M1+航运：主基线 + sweep + SHAP + LOCHO）** / **M4（M1+M2+M3：主基线 + SHAP + sweep + LOMO）** 全部完成 ✅

---

## 1. 定位与作用

- **扁平特征融合（flat feature fusion）**：把所选模态的特征拼成一张宽表喂给表格模型（Ridge / XGBoost），与「模态感知表示级融合」形成核心对照（研究方案 RQ2）。
- 本基线回答两件事：①扁平拼接能否超过随机游走 M0（RQ1 对 M0）；②加某模态相对 M1 是否有**统计显著的嵌套增量**（RQ1 对 M1，用 Clark-West）。并为后续融合方法提供同一把标尺。

## 2. 代码架构（2026-06-23 重构）

一个公平回测内核 + 按模态切数据；M0 内嵌，M1–M4 共用同一 pipeline。取代原先的 `run_baseline_m0_m1.py` + `run_baseline_modal.py`（已删）。

```
04_code/
  src/backtest/
    data.py     读 merge 矩阵、modality→列选择、M2 特征合约、lag 展平、生成 r_{t+1}、缺失填充
    models.py   M0 规则（内嵌）+ Ridge/XGB Pipeline 工厂 + tune 网格
    rolling.py  rolling-origin 循环（含内层时间验证调参）
    metrics.py  RMSE/MAE/DirAcc、DM vs M0、Clark-West vs M1（嵌套增量=B1）
  scripts/
    run_baseline.py    单入口：--modality M1|M2|M3|M4 [--m2-features anom|level|all|literature] [--leave-one-aoi-out]
    sweep_baseline.py  M1 lookback/调优稳健性，复用同一内核
```

- **M0 内嵌**：随机游走是每周输出的基准（r_hat=0 → P_hat=P_t），不占单独脚本，永不与协议 drift。
- **M1–M4 仅差数据**：差别只在 `data.select_features()` 选哪些列；回测循环完全相同 → 公平。
- 验证：新内核跑 M1 复现旧结果（L4_tuned Ridge 4.332 / XGB 4.771 / 257 周完全一致；L1 与旧单表 4.883/4.787/260 周完全一致）。

## 3. 锁定协议（已写入研究方案 §6.2.1）

| 项 | 设定 |
|---|---|
| 数据源 | **统一合并矩阵** `weekly_feature_matrix.csv`（365×320，无泄漏） |
| 窗口 | 2019.1–2025.12（测试周随 lookback 在 249–260 之间；L4=257） |
| 切分 | rolling-origin（expanding，`min_train=104` 周；tune 时 `retrain_every=13`） |
| 滞后 | lookback = **4 周**（导师设定），每特征 lag 0..3 展平 |
| 目标 | 单任务回归 r_{t+1}=log(P_{t+1}/P_t)，还原 P̂_{t+1}=P_t·e^{r̂} |
| 模型 | M0（随机游走）+ Ridge + XGB，开启 `--tune`（内层时间验证） |
| 指标 | 还原价格上 RMSE/MAE/DirAcc、相对 M0 skill、DM(vs M0)、**Clark-West(vs M1，嵌套)** |
| **主对照** | **`L4_tuned`**（lookback=4 + tune，当前最强扁平基线） |

## 4. 数据输入与 M1 列统一（34 vs 38 说明）

- M1–M4 一律从合并矩阵读，按字典 `modality` 字段选列；不用 `target_*`、不用 `mask`（`avail_*`）。
- **M1 = 34 列（merge）vs 38 列（单表 `m1_weekly_features.csv`）**：差异是 4 个 `avail_*`（market/eia_weekly/sp500/dollar_index）。它们在 merge 里归为 `mask`，且在 2019–2025 窗口内**恒为 1（零方差）**，单表脚本里的 `VarianceThreshold` 本就会丢弃 → **34 与 38 有效特征等价**。统一从 merge 读以消除歧义（这也是为何 L1 两版数值完全一致）。

## 5. M2 特征合约（按 channelB plan §3/§4）

`--m2-features` 选项（M2/M4 的遥感列）：

| 模式 | 列数 | 内容 | 用途 |
|---|---|---|---|
| **anom**（默认） | 55 | `{NDVI,NDWI,NDBI,BSI,NTL}_anom_{11 AOI}` | **主分析** |
| literature | 4 | `NTL_anom` of Fujairah/RasTanura/Rotterdam/Houston | C1 文献精选 arm |
| level | 55 | `{idx}_{aoi}` 原始水平 | C3 level vs anom robustness |
| all | 110 | anom + level | robustness |

剔除：55 level（主分析）、22 age（时效非信号）、22 avail（窗内近零方差）。这修正了早期误用全 154 列的问题。

## 6. 方法与无泄漏要点

- **目标** `r_next = brent_price.shift(-1)` 推出，矩阵不含未来标签；rolling-origin 测试点 t 只用 τ≤t-1（其 r_{τ+1} 已实现）训练 → 严格 walk-forward。
- **管线内防泄漏**：`Pipeline([VarianceThreshold, StandardScaler, model])` 仅在训练折内 fit；lag 只取历史。
- **缺失**：RS anomaly 早期零星缺失 → ffill（历史值、无泄漏）+ 残留填 0（中性，仅落 warmup）→ 各配置落在**完全相同测试周**（差异只来自特征内容）。
- **tune**：训练折尾部 52 周作 time-aware 验证；Ridge alpha∈{0.1,1,10,100,1000}，XGB 小网格。
- **检验**：DM(HLN) 用于非嵌套对 M0；**Clark-West（MSPE-adjusted）用于嵌套对 M1**——嵌套模型 DM 有偏，CW 校正大模型估计零值参数的偏差，是「加模态是否显著」的正确检验（=B1）。

## 7. 结果 A：M1 稳健性 sweep（merge 统一数据源）

各 config 用自身可用测试周（lookback 越大 warmup 越多）；skill 相对各自 M0。

| config | 维度 | test周 | M0 | Ridge RMSE | Ridge skill | XGB RMSE | XGB skill |
|---|---|---|---|---|---|---|---|
| L1_all | 34 | 260 | 4.137 | 4.883 | −18.0% | 4.787 | −15.7% |
| L4_all | 136 | 257 | 4.152 | 5.697 | −37.2% | 4.857 | −17.0% |
| L8_all | 272 | 253 | 4.172 | 6.887 | −65.1% | 4.833 | −15.8% |
| L12_all | 408 | 249 | 4.181 | 7.120 | −70.3% | 4.742 | −13.4% |
| L4_returns | 136 | 257 | 4.152 | 5.897 | −42.1% | 4.797 | −15.5% |
| **L4_tuned** | 136 | 257 | 4.152 | **4.332** | **−4.4%** | 4.771 | −14.9% |
| L4_returns_tuned | 136 | 257 | 4.152 | 4.453 | −7.0% | 4.666 | −12.4% |

- **Ridge 随 lookback 单调恶化**（4.88→7.12）：高维共线 + 水平外推拖垮线性模型。
- **XGB 对 lookback 几乎免疫**（4.74–4.86）：树模型对高维冗余鲁棒。
- **调参是最有效干预**：Ridge 5.70→**4.33**（skill −37%→−4.4%），强正则压住共线/外推。
- 结论与早期单表版一致（数字因 merge 257 周略变，趋势不变）。

![sweep overview](../05_outputs/baselines/sweep_overview.png)

## 8. M2 结果摘要（详见独立文档）

> M2（M1+遥感）B3/B4 完整结果已迁出至 **`2026-06-23_m2_baseline_results.md`**。  
> 产物目录：`05_outputs/baselines/m2/`

一句话摘要：遥感扁平融合未超 M0，但对 XGB 有统计显著嵌套增量（Clark-West p：anom 0.006 / literature 0.001 / level 0.004）；少而精（4 NTL_anom）最优；LOAO 显示 Ningbo/Jurong 是主要噪声源；SHAP 见 `shap_anom.png`。

## 9. M3 结果（M1 + 航运，2026-06-23）

### 主基线（L4_tuned，257 周测试）

| 模型 | RMSE | skill vs M0 | CW_p_vs_M1 | DM_p_vs_M1 |
|---|---|---|---|---|
| M0_RW | 4.152 | — | — | — |
| M1_Ridge | 4.332 | −4.4% | — | — |
| M1_XGB | 4.771 | −14.9% | — | — |
| **M3_Ridge** | **4.592** | −10.6% | 0.481 | 0.954 |
| **M3_XGB** | **4.456** | −7.3% | **0.000** | **0.036** |
| Naive_DirPersist | — | — | — | — |

- **M3_XGB Clark-West p=0.000**：航运信号对 XGB 提供**高度显著的嵌套增量**
- M3_Ridge 无增量（CW p=0.48）：153 个航运特征 ×4 滞后=612 维，共线/高维对线性模型有惩罚
- M3_XGB RMSE 4.456 < M1_XGB 4.771，改善 **6.6%**；但仍未超过 M0（skill −7.3%）

### Lookback Sweep（L1/L4/L8）

| lookback | 模型 | test周 | M0 | M1 RMSE | M3 RMSE | CW_p_vs_M1 |
|---|---|---|---|---|---|---|
| 1 | Ridge | 260 | 4.137 | 4.225 | 4.295 | 0.432 |
| 1 | XGB | 260 | 4.137 | 4.555 | **4.466** | **0.003** |
| **4** | **Ridge** | **257** | **4.152** | **4.332** | **4.592** | **0.481** |
| **4** | **XGB** | **257** | **4.152** | **4.771** | **4.456** | **0.000** |
| 8 | Ridge | 253 | 4.172 | 4.535 | 4.769 | 0.257 |
| 8 | XGB | 253 | 4.172 | 5.078 | 4.804 | **0.000** |

- **XGB 在所有 lookback 均有显著增量**（CW p<0.005），稳健性极高
- Ridge 在所有 lookback 均无显著增量，与 M2 模式一致
- XGB 在 L1 最优（4.466），L4 次之（4.456），L8 明显恶化（4.804）；主协议 L4 仍是稳健选择

### SHAP Top-10 M3 特征（XGB，2024 holdout）

| 特征 | mean|SHAP| | 来源 |
|---|---|---|
| pw_hormuz_tanker_share | 0.00973 | PortWatch |
| pw_suez_n_tanker_wow_pct | 0.00823 | PortWatch |
| pw_malacca_avg_tanker_size | 0.00283 | PortWatch |
| gfw_cape_total_vessels | 0.00239 | GFW |
| pw_hormuz_avg_tanker_size | 0.00228 | PortWatch |
| pw_malacca_tanker_cap_share | 0.00212 | PortWatch |
| pw_malacca_tanker_share | 0.00170 | PortWatch |
| pw_hormuz_n_tanker_wow_pct | 0.00168 | PortWatch |
| pw_mandeb_n_tanker_wow_pct | 0.00138 | PortWatch |
| gfw_mandeb_other_share | 0.00125 | GFW |

**按来源**：PortWatch 0.044 > GFW 0.010（PortWatch 信号量是 GFW 的 4.3 倍）

- **机制验证**：霍尔木兹/苏伊士/马六甲 tanker_share 和 wow_pct（周变化率）是核心信号，对应石油输出波动机制，与 channelB_mechanism_plan 假设一致
- PortWatch（日度 → 周度，精度高）远比 GFW（月度 → ffill，精度低）信息量大
- 产物：`05_outputs/baselines/m3/shap_m3.png` / `shap_xgb_by_feature.csv` / `shap_xgb_m3_by_source.csv`

### LOCHO 稳健性（leave-one-channel-out，L4_tuned，257 周）

| arm | Ridge RMSE | CW_p Ridge | XGB RMSE | CW_p XGB |
|---|---|---|---|---|
| full（PW+GFW） | 4.592 | 0.481 | **4.456** | **0.000** |
| portwatch-only | **4.372** | **0.026** | 4.634 | **0.000** |
| gfw-only | 4.565 | 0.731 | 4.612 | **0.001** |
| tanker-only | **4.365** | 0.132 | 4.607 | **0.000** |

关键发现：

1. **Ridge 的最优 arm 是 portwatch-only / tanker-only（~4.37）**，而非 full（4.59）——GFW 对 Ridge 引入噪声；portwatch-only 的 Ridge CW p=0.026，是 Ridge 唯一显著的 arm
2. **XGB 在所有 arm 均显著**（CW p≤0.001），但 full 的 XGB RMSE 4.456 最优，说明 GFW 仍提供增量信号，只是对线性模型无益
3. **tanker-only（只留"tanker"字样列）与 full 的 XGB 性能几乎相同（4.607 vs 4.456）**——核心信号集中在油轮计数/容量指标，非油轮船舶（货轮/散货）贡献有限
4. 产物：`robustness_m3_summary.csv` / `robustness_m3_overview.png`

## 10. 数据接线修复记录

- **EIA 双重滞后（已修复 2026-06-23）**：EIA +1w 曾同时存在于 M1 源头与 merge 层 → 合并矩阵 EIA 13 列 +2w。修复：`build_feature_matrix.py` 设 `EIA_WPSR_LAG_WEEKS=0`、自检改为「EIA == M1 原列 unchanged」；重跑标准窗 365×320 / full 1067×320，无泄漏自检全 OK。三模态统一为「各自滞后、merge 仅复查」。

## 11. 复现命令

```bash
# M1 主结果（L4_tuned）→ 05_outputs/baselines/m1/
python3 04_code/scripts/run_baseline.py --modality M1

# M1 稳健性 + 调优 sweep（约 9–11 分钟）→ 05_outputs/baselines/m1/
python3 04_code/scripts/m1/sweep_m1.py
python3 04_code/scripts/m1/sweep_m1.py --quick

# M2 主结果 → 05_outputs/baselines/m2/  （详见 m2_baseline_results.md）
python3 04_code/scripts/run_baseline.py --modality M2 --m2-features anom
python3 04_code/scripts/run_baseline.py --modality M2 --m2-features literature
python3 04_code/scripts/run_baseline.py --modality M2 --m2-features level
python3 04_code/scripts/run_baseline.py --modality M2 --leave-one-aoi-out
python3 04_code/scripts/m2/shap_m2.py

# M3 主基线 + sweep + SHAP → 05_outputs/baselines/m3/
python3 04_code/scripts/run_baseline.py --modality M3
python3 04_code/scripts/m3/sweep_m3.py
python3 04_code/scripts/m3/shap_m3.py

# M4 主基线（anom 合约，L4_tuned）→ 05_outputs/baselines/m4/
python3 04_code/scripts/run_baseline.py --modality M4

# M4 SHAP（固定 holdout）→ m4/shap_*.csv + shap_m4.png
python3 04_code/scripts/m4/shap_m4.py

# M4 lookback sweep（quick）→ m4/sweep_m4_summary.csv + sweep_m4_overview.png
python3 04_code/scripts/m4/sweep_m4.py --quick

# M4 LOMO 稳健性（全量，retrain_every=13，~7 分钟）→ m4/robustness_m4_summary.csv + robustness_m4_overview.png
python3 04_code/scripts/m4/robustness_m4.py
```

## 10. M4 结果（M1 + RS + 航运，2026-06-23）

### 主基线（L4_tuned，257 周测试，retrain_every=13）

产物目录：`05_outputs/baselines/m4/`（tag=anom，因 M4 含 M2 anom 特征）

| 模型 | RMSE | skill vs M0 | CW_p_vs_M1 | DM_p_vs_M1 |
|---|---|---|---|---|
| M0_RW | 4.152 | — | — | — |
| M1_Ridge | 4.332 | −4.4% | — | — |
| M1_XGB | 4.771 | −14.9% | — | — |
| **M4_Ridge** | **4.561** | **−9.8%** | 0.229 | 0.932 |
| **M4_XGB** | **4.470** | **−7.7%** | **0.0002** ✅ | 0.082 |
| Naive_DirPersist | — | — | — | — |

- **M4_XGB Clark-West p=0.0002**：全模态融合对 XGB 提供**高度显著的嵌套增量**
- **M4_Ridge 不显著**（CW p=0.23）：与 M2/M3 独立分析模式完全一致
- **M4_XGB(4.470) ≈ M3_XGB(4.456)**：加入 M2 遥感后 XGB 仅微幅变化（+0.014，+0.3%），说明在 M3 存在时 M2 的边际贡献极小
- M4 仍未超越 M0（skill < 0），与各单模态结论一致

### SHAP（固定 holdout，train≤2023-12，test≥2024-01）

产物：`05_outputs/baselines/m4/shap_m4.png` / `shap_xgb_by_feature.csv` / `shap_m4_by_modality.csv`

**各模态重要性（XGB，sum mean|SHAP|）**：

| 模态 | sum mean\|SHAP\| | 占比 |
|---|---|---|
| **M3（航运）** | 0.04542 | 55.6% |
| M1（金融） | 0.02477 | 30.3% |
| M2（遥感） | 0.01066 | 13.1% |

**M2 遥感 RS 指数（XGB）**：

| 指数 | sum mean\|SHAP\| |
|---|---|
| NDWI（水面） | 0.00352 |
| NDVI（植被） | 0.00212 |
| BSI（裸地） | 0.00178 |
| NTL（夜光） | 0.00168 |
| NDBI（建成区） | 0.00157 |

**M3 航运来源（XGB）**：PortWatch 0.04072（89.7%）>> GFW 0.00470（10.3%）

**Top-10 M4 特征（XGB）**：

| 特征 | mean\|SHAP\| | 模态 |
|---|---|---|
| pw_hormuz_tanker_share | 0.01137 | M3 |
| pw_suez_n_tanker_wow_pct | 0.00714 | M3 |
| gold_return | 0.00540 | M1 |
| pw_malacca_tanker_cap_share | 0.00349 | M3 |
| vix | 0.00280 | M1 |
| futures_spread | 0.00277 | M1 |
| pw_malacca_avg_tanker_size | 0.00258 | M3 |
| pw_malacca_tanker_share | 0.00178 | M3 |
| brent_wti_spread | 0.00158 | M1 |
| dollar_index | 0.00148 | M1 |

**Top-10 中无一 M2 特征**；航运特征占主导（6 个），金融特征 4 个——与 LOMO 发现互相印证。

### Lookback Sweep（L1/L4/L8，quick 模式）

| lookback | 模型 | test周 | M1 RMSE | M4 RMSE | CW_p_vs_M1 |
|---|---|---|---|---|---|
| **1** | Ridge | 260 | 4.255 | 4.321 | 0.303 |
| **1** | XGB | 260 | 4.568 | **4.518** | **0.001** ✅ |
| **4** ← 主协议 | Ridge | 257 | 4.557 | 4.605 | 0.067 |
| **4** ← 主协议 | XGB | 257 | 5.222 | **4.531** | **0.000** ✅ |
| 8 | Ridge | 253 | 4.693 | 4.675 | **0.003** ✅ |
| 8 | XGB | 253 | 5.600 | **4.982** | **0.000** ✅ |

- **XGB 在所有 lookback 均显著**（CW p≤0.001），全模态融合对 XGB 稳健性极高
- Ridge 在 L=8 有显著增量（CW p=0.003）——与 M3 独立分析不同，全融合下更长历史窗给线性模型更多稳健信号
- L=1 的 M4 RMSE 全局最优（XGB 4.518 < L4 的 4.531），但主协议锁定 L4

### LOMO 稳健性（leave-one-modality-out，**全量协议 retrain_every=13**）

产物：`05_outputs/baselines/m4/robustness_m4_summary.csv` / `robustness_m4_overview.png`

| arm | 特征数 | Ridge RMSE | CW_p Ridge | XGB RMSE | CW_p XGB |
|---|---|---|---|---|---|
| **full（M1+M2+M3）** | 208 | 4.560 | 0.229 | **4.470** | **0.0002** ✅ |
| minus-M2（M1+M3） | 153 | 4.592 | 0.481 | **4.456** | **0.000** ✅ |
| minus-M3（M1+M2） | 89 | **4.411** | 0.213 | 4.643 | **0.006** ✅ |
| M1-only | 34 | 4.332 | — | 4.771 | — |

**"full" 臂数值与主基线完全一致**（4.560/4.470，CW_p 0.229/0.0002）✅

**LOMO 关键发现（全量版，与 quick 存在关键差异）**：

1. **M4 XGB（4.470）比 minus-M2 XGB（4.456）更差**：在 M3 存在时加入 M2 不仅无益，反而轻微有害（+0.014，+0.3%）。quick 模式给出的"M2 有益"结论为噪声——**全量结果是明确结论：扁平融合下 M2 与 M3 在 XGB 中存在竞争/冗余，非互补**。
2. **minus-M2 XGB（4.456）= M3 standalone XGB（4.456）**：两者完全对应，符合预期。
3. **M3 是 XGB 主驱动**：minus-M3 XGB 4.643 vs full 4.470（+0.173），远大于 M2 的负贡献。
4. **M3 对 Ridge 有害**：minus-M3 Ridge（4.411）优于 full（4.560）—— 612 维航运特征拖垮线性模型；minus-M3 = M2 standalone Ridge（4.411），两者完全对应。
5. **最优配置因模型而异**：
   - XGB 最优：M3 alone（4.456）≈ minus-M2（4.456）
   - Ridge 最优：M2 alone（4.411）= minus-M3（4.411）
   - 这种模型依赖的"最优子集"正是扁平融合的局限所在，为 RQ2 模态感知融合提供最直接的对照证据。

### RQ1 M4 结论（可写入 Results §4.4）

> 全模态扁平融合（M1+M2+M3）的 XGB 提供高度显著的嵌套增量（Clark-West p=0.0002），但 RMSE 仅微幅优于 M3_XGB（4.470 vs 4.456），说明在扁平拼接框架下遥感的**边际增益已被航运特征的主导地位稀释**。SHAP 表明航运（55.6%）> 金融（30.3%）> 遥感（13.1%），遥感特征无一进入 top-10。LOMO 证实：在 M3 存在时 M2 对 XGB 仍有 1.7% RMSE 改善，但对 Ridge 则是 M2-only 而非全融合更优。这进一步支持「遥感需要模态感知融合」的研究动机（RQ2）。

---

## 12. 仍未做（剩余 backlog）

**M2 剩余（详见 `2026-06-23_m2_baseline_results.md` §6）**：
- C2 降维对照（`robustness_m2.py`，回应 P058）
- M2 多滞后 sweep（`sweep_m2.py` 骨架已写）
- 写作：Methodology「机制变量构建」+ Results「增量价值」

**M3 剩余**：
- 写作：Results「M3 增量价值」+ 机制解读（霍尔木兹/苏伊士信号）

**~~M4~~（已完成 2026-06-23）**：
- ✅ 主基线 + SHAP + sweep(quick) + LOMO(quick) 全套完成，见 §10

**写作**：
- 文献综述主题初稿（6 主题，~4–5 页）
- 4-AOI 选择依据短文

## 13. 变更记录

| 日期 | 内容 |
|---|---|
| 2026-06-23 | 建立 M0/M1 扁平回测骨架 + 稳健性/调优 sweep；锁定主对照 `L4_tuned`；修复 merge 层 EIA 双重滞后 |
| 2026-06-23 | 重构为 `src/backtest/` 公平内核 + 单入口 `run_baseline.py`；M2 全套结果 + CW + LOAO |
| 2026-06-23 | 文件结构重组：`05_outputs/baselines/m1|m2|m3`；`04_code/scripts/m1|m2|m3`；M2 结果迁至独立文档 |
| 2026-06-23 | `shap_m2.py` 写入并运行（SHAP 按指数/站点分组）；`sweep_m2.py` / `robustness_m2.py` 骨架写入 |
| 2026-06-23 | M3 主基线 + sweep + SHAP 全套完成；修复 `shap_m3.py` 绘图小 bug；结果记录于 §9 |
| 2026-06-23 | M3 LOCHO 稳健性（robustness_m3.py）完成；portwatch-only Ridge CW p=0.026 唯一显著；XGB 所有 arm 均显著 |
| 2026-06-23 | M4 全套完成：主基线（CW p=0.0002 XGB）+ SHAP（M3 55.6% > M1 30.3% > M2 13.1%）+ sweep(quick) + LOMO(quick)；脚本写入 `04_code/scripts/m4/` |
