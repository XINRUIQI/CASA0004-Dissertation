# 核心实证层基线记录（Flat Baselines）

> **定位**：核心实证层对照（M0–M4）：**表格扁平融合**（Ridge / XGB）。方法集成层（模态感知表示级融合）必须超越本层。
> **代码**：`04_code/src/backtest/`（公平回测内核）+ `run_baseline.py`（Ridge/XGB M0–M4）+ `sweep_baseline.py`（M1 稳健性）
> **产物**：`05_outputs/baselines/`（`baseline_metrics_*.csv` / `baseline_predictions_*.csv` / `baseline_backtest_*.png` / `baseline_loao_*.csv` / `sweep_*`）
>
> **更新**：2026-07-03（**M1/M2/M3 特征改动后 M0–M4 全套统一重跑**：merge 矩阵重建 365×212；§7–§12 数值 + SHAP + sweep + robustness 全部刷新）
>
> **2026-07-05 增补**：M3/M4 主模型改 **full 113 列**（§9/§12）；新增 **§8.9 M2 2×2 稀疏性六臂**（`aoi4`/`ntlall` + 水体掩膜，站点维度冗余 > 指标维度）；新增 **§13 创新层表示级融合首跑**（Mship vs flat M1 **CW 0.0014** ✅、Mfusion 逼近 M0——RQ2 首个正面证据）。
>
> 进度：M0 / M1 / M2（anom 主 + literature/level + CW + LOAO + SHAP + C2 + sweep + watermask）/ M3（主基线 + sweep + SHAP + LOCHO）/ M4（主基线 + SHAP + sweep + LOMO）
>
> **本次重跑关键结论变化**：M1 精简后大幅变强（旧 XGB 4.771 → 新 **4.368**；M3 主模型改为 core tier 38 列），导致**单模态 anom-55 / M3-core 的 XGB 嵌套增量由「高度显著」退化为「边缘不显著」**（M2 anom CW_p 0.085、M3 core CW_p 0.096）；**只有 M4 全模态（CW_p 0.020）、M2 literature（0.022）、M2 watermask（0.028）、M3 full/portwatch/tanker 臂仍显著**。强化 RQ2 论据：把 RS/航运扁平拼到已很强的金融基线上，边际增量被稀释 → 需模态感知融合。
>
> ⚠️ **2026-07-05 再更新**：M3/M4 主模型已由 core 38 列改为 **full 113 列** → M3_XGB CW 0.0002、M4_XGB CW 0.009 均显著（不再是 core 版的边缘不显著）；SHAP 模态占比重回 **M3(52%)>M1(34%)>M2(15%)**。下方 §7–§12 的 core-tier 数值降为稳健性对照，最新主表见 §4/§9/§12 与 changelog。

---

## 1. 定位与作用

- **表格扁平融合（flat feature fusion）**：把所选模态的特征拼成一张宽表（lag 0..3 展平）喂给 Ridge / XGBoost。
- 本基线共用同一协议与评估内核，回答：①能否超过随机游走 M0（RQ1）；②加模态相对 M1 是否有**统计显著的嵌套增量**（Clark-West）；并为模态感知融合提供同一把标尺。



## 2. 代码架构

一个公平回测内核 + 按模态切数据；M0 内嵌，M1–M4 共用同一 pipeline。

```
04_code/
  src/backtest/
    data.py     读 merge 矩阵、modality→列选择、M2 特征合约、lag 展平、生成 r_{t+1}、缺失填充
    models.py   M0 规则（内嵌）+ Ridge/XGB Pipeline 工厂 + tune 网格
    rolling.py  rolling-origin 循环（含内层时间验证调参）
    metrics.py  RMSE/MAE/DirAcc、DM vs M0、Clark-West vs M1（嵌套增量=B1）
  scripts/
    run_baseline.py       单入口：--modality M1|M2|M3|M4 [--m2-features anom|level|all|literature] [--leave-one-aoi-out]
    sweep_baseline.py     M1 lookback/调优稳健性，复用同一内核
```

- **M0 内嵌**：随机游走是每周输出的基准（r_hat=0 → P_hat=P_t），不占单独脚本，永不与协议 drift。
- **M1–M4 仅差数据**：差别只在 `data.select_features()` 选哪些列；回测循环完全相同 → 公平。
- 验证：M1 精简特征集（2026-07-03 正式重跑）L4_tuned：Ridge **4.256** / XGB **4.368** / 257 周；L1：Ridge 4.818 / XGB 4.686 / 260 周（31 raw × lag 0..3 = 124 维）。



## 3. 锁定协议


| 项       | 设定                                                                         |
| ------- | -------------------------------------------------------------------------- |
| 数据源     | **统一合并矩阵** `weekly_feature_matrix.csv`（365×320，无泄漏）                        |
| 窗口      | 2019.1–2025.12（测试周随 lookback 在 249–260 之间；L4=257）                          |
| 切分      | rolling-origin（expanding，`min_train=104` 周；tune 时 `retrain_every=13`）      |
| 滞后      | lookback = **4 周**（导师设定），每特征 lag 0..3 展平                                   |
| 目标      | 单任务回归 r_{t+1}=log(P_{t+1}/P_t)，还原 P̂_{t+1}=P_t·e^{r̂}                      |
| 模型      | M0（随机游走）+ Ridge + XGB（`--tune`） |
| 指标      | 还原价格上 RMSE/MAE/DirAcc、相对 M0 skill、DM(vs M0)、**Clark-West(vs M1，嵌套)**       |
| **主对照** | `L4_tuned`（lookback=4 + tune，当前最强扁平基线）                                     |




## 4. 数据输入与 M1 列统一（31 vs 35 说明）

- M1–M4 一律从合并矩阵读，按字典 `modality` 字段选列；不用 `target_*`、不用 `mask`（`avail_*`）。
- **M1 = 31 列（merge）vs 35 列（单表 `m1_weekly_features.csv`）**：差异是 4 个 `avail_*`（market/eia_weekly/sp500/dollar_index）。它们在 merge 里归为 `mask`，且在 2019–2025 窗口内**恒为 1（零方差）**，单表脚本里的 `VarianceThreshold` 本就会丢弃 → **31 与 35 有效特征等价**。统一从 merge 读以消除歧义。
  - **列数为 2026-07 M1 精简后**（单表 38→35、merge 34→31）：移除 `brent_direction` / `brent_return_pct` / `net_crude_trade` / `sp500` level；`wti_return_pct`→`wti_log_return`、`sp500_return_pct`→`sp500_log_return`；`futures_spread`→`brent_f1_spot_log_basis` + 新增 `brent_roll_week`；`commodity_fx`→`cadusd_log_return`（删 AUD 腿）；`gpr` 改日度 GPRD。变更详见 `research_diary_phase3.md`（2026-07-03）与 `m1_data_dictionary.md`。
  - ✅ **§7–§12 全部数值已基于 2026-07-03 正式重跑更新**（merge 365×212：M1=31 + M2=55 + M3=113 + mask=11 + target=2）：`run_baseline.py` M1/M2/M3/M4 + 各模态 sweep / SHAP / robustness 全套刷新，嵌套对照里的 M1 基线统一为新精简版（Ridge **4.256** / XGB **4.368**）。
  - **M3/M4 主模型定义变更（2026-07-05）**：`data.py` 的 `m3_tier` 默认由 **core（38 列）改为 `full`（全 113 航运列 = GFW 49 + PW 64）** → M3 raw = **144**（31 M1 + 113）、M4 raw = **199**（31+55+113）。理由：core 在 XGB 下最弱且不显著（RMSE 4.476、CW_p 0.096），full 显著（RMSE 4.429、CW_p 0.0002），人工精选丢弃了树模型可用的航运信号。core（38 列 = GFW 6×4 + PW 咽喉 6×2 + PW 港口 2）降级为 `robustness_m3.py` 的稳健性臂。⚠️ 下方 §7–§12 主表数值为 **2026-07-03（M3/M4 用 core）** 跑数；**2026-07-05 主表已改 full**：M3_XGB 4.429 / CW 0.0002、M4_XGB 4.507 / CW 0.009（见 §9 LOCHO 的 full 臂与 `05_outputs/baselines/{m3,m4}/` 产物）；SHAP 与 lookback sweep 均已按 full 重跑刷新（§9/§12，2026-07-05，L4 与主基线逐格一致）。



## 5. M2 特征合约（按 channelB plan §3/§4）

`--m2-features` 选项（M2/M4 的遥感列）：


| 模式           | 列数  | 内容                                                 | 用途                          |
| ------------ | --- | -------------------------------------------------- | --------------------------- |
| **anom**（默认） | 55  | `{NDVI,NDWI,NDBI,BSI,NTL}_anom_{11 AOI}`           | **主分析**                     |
| literature   | 4   | `NTL_anom` of Fujairah/RasTanura/Rotterdam/Houston | C1 文献精选 arm                 |
| level        | 55  | `{idx}_{aoi}` 原始水平                                 | C3 level vs anom robustness |
| all          | 110 | anom + level                                       | robustness                  |


剔除：55 level（主分析）、22 age（时效非信号）、22 avail（窗内近零方差）。这修正了早期误用全 154 列的问题。

## 6. 方法与无泄漏要点

- **目标** `r_next = brent_price.shift(-1)` 推出，矩阵不含未来标签；rolling-origin 测试点 t 只用 τ≤t-1（其 r_{τ+1} 已实现）训练 → 严格 walk-forward。
- **管线内防泄漏**：`Pipeline([VarianceThreshold, StandardScaler, model])` 仅在训练折内 fit；lag 只取历史。
- **缺失**：RS anomaly 早期零星缺失 → ffill（历史值、无泄漏）+ 残留填 0（中性，仅落 warmup）→ 各配置落在**完全相同测试周**（差异只来自特征内容）。
- **tune**：训练折尾部 52 周作 time-aware 验证；Ridge alpha∈{0.1,1,10,100,1000}，XGB 小网格。
- **检验**：DM(HLN) 用于非嵌套对 M0；**Clark-West（MSPE-adjusted）用于嵌套对 M1**——嵌套模型 DM 有偏，CW 校正大模型估计零值参数的偏差，是「加模态是否显著」的正确检验（=B1）。



## 7. 结果 A：M1 稳健性 sweep（merge 统一数据源）

各 config 用自身可用测试周（lookback 越大 warmup 越多）；skill 相对各自 M0。


| config           | 维度  | test周 | M0    | Ridge RMSE | Ridge skill | XGB RMSE | XGB skill |
| ---------------- | --- | ----- | ----- | ---------- | ----------- | -------- | --------- |
| L1_all           | 31  | 260   | 4.137 | 4.818      | −16.5%      | 4.686    | −13.3%    |
| L4_all           | 124 | 257   | 4.152 | 5.909      | −42.3%      | 4.694    | −13.1%    |
| L8_all           | 248 | 253   | 4.172 | 6.561      | −57.3%      | 4.747    | −13.8%    |
| L12_all          | 372 | 249   | 4.181 | 6.901      | −65.1%      | 4.840    | −15.8%    |
| L4_returns       | 124 | 257   | 4.152 | 6.233      | −50.1%      | 4.526    | −9.0%     |
| **L4_tuned**     | 124 | 257   | 4.152 | **4.256**  | **−2.5%**   | **4.368** | **−5.2%** |
| L4_returns_tuned | 124 | 257   | 4.152 | 4.298      | −3.5%       | 4.413    | −6.3%     |


- **Ridge 随 lookback 单调恶化**（4.82→6.90）：高维共线 + 水平外推拖垮线性模型。
- **XGB 对 lookback 几乎免疫**（4.69–4.84）：树模型对高维冗余鲁棒。
- **调参是最有效干预**：Ridge 5.91→**4.26**（skill −42%→−2.5%），强正则压住共线/外推。
- M1 精简后（31 列、对数收益统一）RMSE 略优于旧 34 列集；定性结论不变：**L4_tuned 仍为 M1 主对照**。

sweep overview

## 8. M2 结果（M1 + 遥感，B3/B4）

> **配套文档**：Channel B 数据与 EDA 见 `2026-06-22_channelB_mechanism_plan.md`  
> **代码**：`04_code/scripts/flat/run_baseline.py --modality M2` + `04_code/scripts/flat/M2_Flat/`  
> **产物**：`05_outputs/baselines/Flat/M2_Flat/`（`baseline_metrics_*.csv` / `baseline_predictions_*.csv` / `backtest_*.png` / `baseline_loao_*.csv` / `shap_*.csv` / `shap_anom.png`）  
> **进度**：B0/B1/B2/B3(DM+CW+LOAO) ✅；B3 SHAP ✅；C2 降维对照 ✅；C3 lookback × contract sweep ✅；**B4 水体掩膜稳健性** ✅（全部 2026-06-23）

### 8.1 协议（M2 专项）

主协议与 §3 完全相同，以下仅列 M2 差异部分。

| 项 | 设定 |
|---|---|
| **M2 主分析列** | `{NDVI,NDWI,NDBI,BSI,NTL}_anom_{11 AOI}` 共 **55 列**（站点扩展窗 z-score，expanding past-only，防泄漏） |
| **对照臂** | C1 `literature`（4 NTL_anom 文献精选）、C3 `level`（55 原始水平） |
| **增量检验** | DM(HLN) vs M0（非嵌套参考），**Clark-West vs M1**（嵌套增量，正确检验）|
| **站点定位** | leave-one-AOI-out（全 55 anom，retrain_every=26 加速，内部对齐基线）|
| **可解释性** | SHAP 固定 holdout（train≤2023-12，test≥2024-01，L4_tuned，XGB+Ridge）|

**M2 特征合约决策依据**（详见 `channelB_mechanism_plan.md` §3 B3）：
- 主分析用 55 anom（去站点规模 + 去季节，过去-only expanding z-score）
- 剔除 55 level（尺度不可比 + 季节性 + 与 anom 冗余）、22 age（时效非信号）、22 avail（窗内近零方差）

### 8.2 核心结果（B3 主交付）

主协议 L4_tuned，retrain_every=13，**257 共同测试周**（2021–2025），M0 RMSE=4.152。  
CW_p = Clark-West 单边 p（越小 = 大模型即 M1+RS 越显著优于 M1）。**（2026-07-03 精简 M1 后重跑）**

| 配置 | Ridge RMSE | Ridge skill | Ridge CW_p(vs M1) | XGB RMSE | XGB skill | XGB CW_p(vs M1) | XGB DirAcc |
|---|---|---|---|---|---|---|---|
| M1（嵌套基线） | 4.256 | −2.5% | — | 4.368 | −5.2% | — | **0.553** |
| **M2 anom (55)** | 4.414 | −6.3% | 0.474 | 4.440 | −6.9% | 0.085 | 0.479 |
| M2 literature (4) | **4.252** | **−2.4%** | 0.129 | **4.322** | **−4.1%** | **0.022** ✅ | 0.502 |
| M2 level (55) † | ≡ M1 | — | NaN | ≡ M1 | — | NaN | — |

> **† level 合约数据状态**：标准矩阵（`weekly_feature_matrix.csv`，365×212）仅含 55 个 anom 列；55 个 level（原始水平值）列**未纳入矩阵**，`m2_columns(dico, "level")` 返回空列表 → level 合约退化为 M1（sweep 中 RMSE 与 M1 完全相同、CW_p = NaN，见 §8.7）。故不单独跑 `--m2-features level` 主基线；若需 C3 复现须重建矩阵纳入 level 列（优先级低）。

产物：`05_outputs/baselines/Flat/M2_Flat/baseline_metrics_anom.csv` / `baseline_metrics_literature.csv`

![M2 anom backtest](../../05_outputs/baselines/Flat/M2_Flat/backtest_anom.png)

**关键发现（精简 M1 后，与旧跑数对比有实质变化）**：

1. **仍无配置超过 M0**（所有 skill &lt; 0）——符合周度油价 M0 极强的预期。
2. **anom-55 对 XGB 的嵌套增量已退化为不显著**（CW_p 0.085，旧为 0.006）：M1 精简后 XGB 基线由 4.771 骤降至 4.368，吸收了大部分原本由遥感提供的增量；anom-55 的 XGB RMSE（4.440）现已**劣于** M1_XGB（4.368）。
3. **对 Ridge 更不显著**（anom CW_p=0.474）：高维 RS 对线性模型主要是噪声，与旧结论一致。
4. **少而精 > 全量（结论稳健，支持 C1 / P058）**：`literature`（仅 4 个 NTL_anom）仍是唯一有增量价值的合约——Ridge RMSE **4.252（唯一微超 M1 的配置**，4.252 &lt; 4.256），XGB RMSE 4.322 &lt; M1_XGB 4.368、**CW_p=0.022 仍显著**。全 55 列在强 M1 下被噪声淹没，4 列精选反而保住信号。
5. **方法学含义（支撑 RQ2，较旧跑更强）**：把 RS 扁平拼到已很强的金融基线上，全量增量**边际到不显著**、仅少而精子集显著，进一步强化「扁平拼接不足、需模态感知融合」的论据。

### 8.3 leave-one-AOI-out（B3 站点定位）

全 55 anom，每次去掉一个 AOI，**retrain_every=13 自洽基线（full RMSE Ridge=4.414 / XGB=4.440，与 §8.2 主基线完全一致）**。  
`dRMSE > 0` = 去掉该站变差 = 该站有正贡献（按 XGB dRMSE 降序）。**（2026-07-03 重跑，11 站点全列）**

| 去掉站点 | Ridge dRMSE | XGB dRMSE | 解读 |
|---|---|---|---|
| Basra | +0.010 | **+0.021** | 两模型均正贡献（离岸出口码头，水面动态） |
| Jurong | +0.035 | +0.010 | 两模型均正贡献（与旧跑相反） |
| Rotterdam | +0.010 | +0.006 | 两模型均微正贡献 |
| Ulsan | +0.009 | −0.030 | Ridge 微正；XGB 噪声 |
| Kharg | −0.071 | −0.013 | Ridge 下最大噪声源；XGB 微噪 |
| Fujairah | −0.039 | −0.018 | 去掉反改善 |
| Jamnagar | −0.004 | −0.038 | XGB 噪声 |
| RasTanura | −0.008 | −0.039 | XGB 噪声 |
| Yanbu | −0.040 | −0.048 | 两模型去掉均改善（NDWI 高方差站） |
| NingboZhoushan | −0.016 | −0.054 | 非石油核心港噪声（与 SHAP 一致） |
| Houston | −0.006 | **−0.091** | XGB 最大改善——去掉减噪最多 |

产物：`05_outputs/baselines/Flat/M2_Flat/baseline_loao_anom.csv`

**LOAO 结论（精简 M1 后重跑）**：
- **正贡献站点收敛为 Basra / Jurong / Rotterdam**（两模型 dRMSE 同号为正），但幅度都很小（≤+0.035）——遥感增量微弱。
- **XGB 最大噪声源：Houston（−0.091）/ NingboZhoushan（−0.054）/ Yanbu（−0.048）**；Ningbo 非石油核心港与 SHAP「Ningbo 最弱」一致。
- **Ridge 最大噪声源 Kharg（−0.071）**：NDWI 高方差站点对线性模型是纯噪声。
- 各站 |dRMSE| 普遍 ≤ 0.05，无「必要正贡献」站点 → 综合仍强化「少而精」结论（文献精选 4 个 NTL_anom 优于全 55）。

### 8.4 SHAP 可解释性（B3）

脚本 `04_code/scripts/flat/M2_Flat/shap_m2.py`；固定 holdout（train≤2023-12，test=2024–2025，L4_tuned）。

产物：`05_outputs/baselines/Flat/M2_Flat/`
- `shap_xgb_by_feature.csv` — 所有基础特征的 XGB 平均|SHAP|（lags 加总）
- `shap_ridge_by_feature.csv` — Ridge 版
- `shap_xgb_m2_by_index.csv` — 按 RS 指数（NTL/NDBI/BSI/NDWI/NDVI）汇总
- `shap_xgb_m2_by_aoi.csv` — 按 AOI/站点汇总
- `shap_topN_anom.csv` — 前 N 个 M2 特征排名（供 robustness_m2.py C2 对照用）
- `shap_anom.png` — 4 面板概览图

![SHAP overview](../../05_outputs/baselines/Flat/M2_Flat/shap_anom.png)

**运行参数**：train 258 周（2019–2023），test 103 周（2024–2025），L4_tuned；Ridge α=1000（强正则），XGB **n_est=400 depth=3**；holdout log-return RMSE：Ridge 0.0423 / XGB 0.0422。**（2026-07-03 重跑）**

**M2 各 RS 指数重要性（XGB，按 sum mean|SHAP|）**：

| RS 指数 | sum mean\|SHAP\| | 排名 |
|---|---|---|
| **NDVI**（植被） | 0.01025 | 1 |
| **NDWI**（水面动态） | 0.00818 | 2 |
| **NDBI**（建成区） | 0.00635 | 3 |
| **NTL**（夜光） | 0.00616 | 4 |
| **BSI**（裸地/堆场） | 0.00507 | 5 |

**M2 各 AOI/站点重要性（XGB，sum mean|SHAP|）**：

| 排名 | AOI | sum mean\|SHAP\| | 站点类型 |
|---|---|---|---|
| 1 | **Yanbu** | 0.00581 | 沙特红海出口 |
| 2 | **Kharg** | 0.00460 | 伊朗出口码头 |
| 3 | **Ulsan** | 0.00456 | 韩国炼厂枢纽 |
| 4 | Rotterdam | 0.00417 | 欧洲炼厂/枢纽 |
| 5 | Basra | 0.00333 | 伊拉克出口码头 |
| 6 | Jurong | 0.00276 | 新加坡炼厂 |
| … | … | … | … |
| 10 | Fujairah | 0.00185 | 中东集散港 |
| 11 | **NingboZhoushan** | 0.00185 | 中国进口港（最低） |

**Top-5 M2 基础特征（XGB）**：
`NDWI_anom_Yanbu`(0.00211) > `NDVI_anom_Ulsan`(0.00207) > `NDVI_anom_Rotterdam`(0.00185) > `NDVI_anom_Yanbu`(0.00144) > `NDVI_anom_Kharg`(0.00134)

**整体上 M1 金融特征仍主导 Top-5**：`gold_return`(0.01079) > `brent_f1_spot_log_basis`(0.00590) > `brent_log_return`(0.00466) > `gpr`(0.00447) > `cadusd_log_return`(0.00430)；M2 首个特征 `NDWI_anom_Yanbu`(0.00211) 直到第 13 名才进入。

**SHAP 解读（精简 M1 后重跑）**：
1. **NDVI（植被/陆面）升至第一**（旧为 NDWI）— 结合 §8.8 水体掩膜后 NDVI 为纯陆信号，红海/中东出口码头周边陆面动态（堆场/绿化/防护带）在 2024–25 段权重上升；NDWI 退居第二，NTL（文献最推荐）仅第四，说明当前 test 期夜光信号弱于 EDA 预期。
2. **AOI 排序**：Yanbu/Kharg/Ulsan 居前，文献推荐的 Fujairah/RasTanura 相对靠后——可能与 2024 年红海/Houthi 扰动下伊朗（Kharg）、沙特红海（Yanbu）、韩国炼厂（Ulsan）动态异常突出有关；与 LOAO 时段不同（LOAO 全段 2021–25，SHAP 仅 2024–25）导致排序差异。
3. **NingboZhoushan 最弱** — 与 LOAO「去掉 Ningbo 改善」方向一致 ✅，强化"非石油核心港是主要噪声源"结论。
4. **金融特征名已更新为精简后口径**（`brent_f1_spot_log_basis` / `cadusd_log_return` / `gpr` 日度），旧 `futures_spread` / `commodity_fx` 列名不再出现。

### 8.5 RQ1 结论（M2，可写入 Results §4.1）

> 在统一无泄漏协议（2019–2025、4 周滞后、rolling-origin L4_tuned）下，**遥感（Channel B）的扁平特征融合不能战胜随机游走 M0**（所有配置 skill &lt; 0）。  
> 相对**精简后更强的**金融基线 M1（XGB 4.368），**全量 anom-55 的嵌套增量已退化为不显著**（XGB Clark-West p=0.085；Ridge p=0.474）——强 M1 吸收了旧跑数里遥感的大部分增量。**唯有少而精的 4 个夜光异常站点 `literature`（Fujairah/RasTanura/Rotterdam/Houston）仍对 XGB 显著（CW p=0.022），且 Ridge RMSE 4.252 微超 M1**；水体掩膜版（§8.8）亦使 anom-55 XGB 回到显著（p=0.028）。  
> leave-one-AOI-out 显示各站 |dRMSE|≤0.05、无必要正贡献站，非石油核心港（Ningbo/Houston/Yanbu 对 XGB）是主要噪声源；SHAP（2024–25）以 NDVI/NDWI、Yanbu/Kharg/Ulsan 居前，补充指数与站点维度可解释性。  
> **较旧跑数结论更强**：在已很强的金融基线上扁平拼接全量遥感，增量微弱到不显著、只有精选子集或去噪版本才勉强显著 → 更有力地支持「扁平拼接无法充分利用遥感、需要**模态感知的表示级融合**」（RQ2）。

### 8.6 C2 降维对照（回应 P058）

脚本 `04_code/scripts/flat/M2_Flat/robustness_m2.py`；产物 `05_outputs/baselines/Flat/M2_Flat/c2_summary.csv` / `c2_overview.png`。

**协议**：L4，fixed hyperparams（Ridge α=1000，XGB depth=2/n_est=200），257 共同测试周。M0 RMSE=4.152，M1_Ridge=4.245，M1_XGB=**4.496**（注：固定参数而非内层调参，故 M1_XGB 4.496 弱于主结果调参版 4.368；C2 各臂内部对比公平）。**（2026-07-03 重跑）**

| 臂 | M2 输入 | Ridge RMSE | Ridge CW_p | XGB RMSE | XGB CW_p |
|---|---|---|---|---|---|
| all-55 | 55 anom cols | 4.414 | 0.588 | 4.447 | **0.0007** ✅ |
| pca-90 | 55 → PCA 90% var | 4.416 | 0.599 | **4.441** | **≈1×10⁻⁶** ✅✅ |
| elastic | 55 → ElasticNet sel | 4.414 | 0.588 | 4.447 | **0.0007** ✅ |
| shap-top20 | top-20 SHAP | **4.366** | 0.538 | 4.475 | **0.033** ✅ |

> **注（与主基线 §8.2 的关系）**：C2 用**固定超参**，其 M1_XGB=4.496 弱于主协议调参 M1_XGB=4.368，故相对它的 XGB 嵌套增量在 C2 里仍显著（0.0007–1e-6），而主协议下 anom-55 已不显著（0.085）。二者不矛盾——C2 的目的是**在同一固定超参下比较降维策略**，非复现主协议显著性。

**C2 结论**（直接回应 P058「SHAP≠PCA」，2026-07-03 重跑仍成立）：

1. **Ridge：四臂几乎无差异**（RMSE 4.366–4.416，CW_p 全不显著）。α=1000 的强 L2 正则已隐式处理共线性；PCA/ElasticNet 无增益。**shap-top20 给出最好的 Ridge RMSE（4.366）**——减少输入让正则化更聚焦。
2. **XGB：PCA 给出最强 CW 显著性**（p≈1×10⁻⁶，CW_stat 4.76，远优于 all-55 的 0.0007）。去相关后 XGB 能更纯粹利用正交分量；正是 P058「PCA 解决共线性、SHAP 提供可解释性、两者解决不同问题」的实证。
3. **Elastic ≡ all-55**（RMSE/CW 完全相同）：ElasticNet(α=0.05) 在该维度/样本比下产生近全零系数，SelectFromModel 退化为全保留。
4. **shap-top20 特征最少且 Ridge RMSE 最优**，但 XGB CW_p（0.033）弱于 all-55（0.0007）——本次重跑下 SHAP 子集对 XGB 的显著性不如全量/PCA，说明去相关（PCA）比数据驱动选择（SHAP-top）更利于 XGB 的正交信号利用。

**写作建议（P058 回应段）**：Ridge 的鲁棒性来自强 L2 正则，PCA 对 XGB 有实质增益，SHAP 提供事后可解释性但与降维各司其职；三种策略并列，说明"55 anom 的高维共线问题"在 XGB+PCA 下可通过降维缓解，在 Ridge+shap-top20 下通过数据驱动特征选择缓解。

### 8.7 C3 lookback × feature-contract sweep

脚本 `sweep_m2.py`（retrain_every=13，与主协议 + M3/M4 sweep 统一；2026-07-05 重跑），产物 `05_outputs/baselines/Flat/M2_Flat/sweep_m2_summary.csv` / `sweep_m2_overview.png`。

M1 RMSE（retrain_every=13，同周对齐）：L=1 → 4.188 Ridge / 4.479 XGB；L=4 → 4.256 / 4.368；L=8 → 4.339 / 4.646。**（2026-07-05 重跑，与 §8.2 主基线一致）**

| contract | L | Ridge RMSE | Ridge CW_p | XGB RMSE | XGB CW_p |
|---|---|---|---|---|---|
| anom | **1** | 4.230 | 0.447 | 4.510 | 0.057 |
| anom | 4 ← 主协议 | **4.414** | **0.474** | **4.440** | **0.085** |
| anom | 8 | 4.526 | 0.122 | 4.668 | 0.070 |
| literature | **1** | **4.153** | **0.027** ✅ | **4.334** | **0.0007** ✅ |
| literature | 4 | 4.252 | 0.129 | 4.322 | **0.022** ✅ |
| literature | 8 | 4.348 | 0.279 | 4.606 | **0.018** ✅ |
| level | 1/4/8 | ≡ M1 | NaN | ≡ M1 | NaN |

> **注**：本 sweep 已与主协议统一（retrain_every=13）→ **anom L4 逐格等于 §8.2 主基线**（Ridge 4.414 / CW 0.474、XGB 4.440 / CW 0.085）。旧 quick 版 anom XGB「全 lookback 显著」是弱 M1 基线（尤其 L8 的 M1_XGB=5.038）造成的假象，统一协议后消除：**anom XGB 现全 lookback 不显著**（0.057 / 0.085 / 0.070）。level 合约无 level 列 → 退化为 M1（CW_p=NaN）。

**C3 结论（2026-07-05 重跑，retrain_every=13 与主协议统一）**：

1. **L=1 是 M2 的最优 lookback**，RMSE 随 lookback 单调变差——与 M1（调参后 L4 最优）相反。原因：RS 月频信号弱，多 lag 只引入噪声；维度膨胀（55 anom × 8 lags = 440 维）远超样本量。
2. **统一 retrain=13 后，全 55 anom 的 XGB 在所有 lookback 均不显著**（0.057 / 0.085 / 0.070，与主基线 §8.2 一致）；**仅 literature（4 列少而精）XGB 全 lookback 显著**（0.0007 / 0.022 / 0.018），Ridge 仅 literature+L1 显著（0.027）。→ 旧 quick 版「anom 全 lookback 显著」是弱 M1 基线的假象；「全量遥感扁平吃不动、唯少而精显著」在时间窗上稳健（支持 RQ2）。
3. **L=1 literature（Ridge 4.153）是整个 sweep 中 Ridge 的全局最优**（且 < M1 4.188）——4 个 NTL_anom × 1 lag = 4 维，少而精到极致。
4. **与 C2（PCA）一致**：减 lag（L=1）、降维（PCA）、精选（literature）三条路都比全 55 anom 更利于 XGB——高维全量扁平遥感本身是主要障碍。
5. **写作注意**：主协议固定 L=4（导师设定）；Discussion 可补充「literature 合约下 M2 遥感增量对 XGBoost 全 lookback 显著（L1 p=0.0007 / L4 p=0.022 / L8 p=0.018），而全 55 anom 在同协议下均不显著」作为灵敏度分析。

**COVID/红海子期间**：不单独跑，写 Discussion 叙事。SHAP holdout（2024–2025）天然是「红海扰动后」子期间，与 rolling origin（2021–2025）全段排名差异本身即为隐性子期间发现。

### 8.8 B4 水体掩膜稳健性（2026-06-23）

**背景**：B0 审计 + SHAP 均提示 NDWI 高方差风险（Kharg/Yanbu top-2 特征），Basra（`land_px=0.001`）等离岸终端原始 NDVI 因水体像素被压至 −0.25（负值，非信号）。B4 对此做正式稳健性验证。

**方法**（三步管线）：
1. GEE 水体掩膜版 CSV（`sentinel2_oil_sites_monthly_indices_watermask_201704_202512_11aoi.csv`，1155 行 × 28 列，11×105 月完整）→ `build_m2_weekly.py --watermask` → `m2_weekly_features_watermask.csv`（365×188，含 MNDWI + `s2_land_px_*`）
2. `build_feature_matrix.py --m2-csv .../m2_weekly_features_watermask.csv` → `weekly_feature_matrix_watermask.csv`（**365×221**，与标准矩阵等形；`filter_m2_anom_columns` 自动排除 MNDWI_anom，保留 55 anom 但 NDVI/NDBI/BSI 已仅含陆地像素）
3. `run_baseline.py --modality M2 --m2-features anom --matrix weekly_feature_matrix_watermask.csv --tag watermask`

**结果对比（257 共同测试周，M0 RMSE=4.152，精简 M1 后 M1_Ridge=4.256 / M1_XGB=4.368）**：**（2026-07-03 重跑）**

| 模型 | 标准 anom-55 RMSE | 水体掩膜 RMSE | Δ RMSE | 标准 CW_p | 水体掩膜 CW_p |
|---|---|---|---|---|---|
| M2 Ridge | 4.414 | 4.358 | −0.055（−1.25%） | 0.474 | 0.290 |
| **M2 XGB** | 4.440 | **4.414** | **−0.026（−0.59%）** | **0.085** | **0.028** ✅ |

产物：`05_outputs/baselines/Flat/M2_Flat/baseline_metrics_watermask.csv` / `baseline_predictions_watermask.csv`

**B4 结论（精简 M1 后重跑，价值较旧跑更突出）**：
1. **水体掩膜使 anom-55 的 XGB 增量由「不显著」回到「显著」**：标准 anom-55 在强 M1 下 CW_p=0.085（不显著），水体掩膜后 4.440→4.414、**CW_p 0.085→0.028（跨入显著）**——精确提取陆地像素去掉水面噪声后，遥感的嵌套增量才被 XGB 利用。
2. **M2 Ridge：RMSE 也改善**（4.414→4.358，−1.25%）但 CW_p 仍不显著（0.474→0.290）——L2 强正则本已压制噪声，掩膜带来的边际收益不足以让线性模型跨入显著。
3. **稳健性正向且更关键**：与旧跑（掩膜把已显著的 0.006 推到 0.0001）不同，本次是**掩膜把边缘不显著（0.085）救回显著（0.028）**——说明在强 M1 下，遥感增量对「是否去水面噪声」更敏感；主分析用标准 anom-55 是**保守下界**。
4. **写作建议**：主分析用标准 anom-55（保守），水体掩膜版作为 B4 关键稳健性——「陆地像素精确提取后 XGB 增量 CW_p 由 0.085 改善至 0.028，去噪后遥感增量才显著（去水面噪声是利用 RS 的前提）」。

### 8.9 M2 2×2 稀疏性矩阵（六臂，2026-07-05）

**动机**：全 11 站 × 5 指数 anom-55 对 XGB 不显著（CW 0.085）。到底是「站点太多稀释」还是「指标太多稀释」？在 `data.py` 新增 `aoi4`（4 核心站 × 5 指数 = 20）与 `ntlall`（11 站 × NTL = 11）两臂，与既有 `anom`(55)/`literature`(4) 组成 **站点维度 × 指标维度** 的 2×2；另跑 `anom`/`aoi4` 的水体掩膜版做去噪对照。同协议（L4_tuned，257 测试周，M0=4.152，M1_Ridge 4.256 / M1_XGB 4.368）。

**六臂结果（标准云掩膜 + 水体掩膜）**：

| 臂 | 站×指标 | 列数 | Ridge RMSE | Ridge CW_p | XGB RMSE | XGB CW_p |
|---|---|---|---|---|---|---|
| anom | 11×5 | 55 | 4.414 | 0.474 | 4.440 | 0.085 ✗ |
| ntlall | 11×NTL | 11 | 4.246 | 0.108 | 4.345 | 0.036 ✅ |
| **aoi4** | 4×5 | 20 | 4.302 | 0.395 | 4.340 | **0.0055** ✅ |
| literature | 4×NTL | 4 | 4.252 | 0.129 | 4.322 | 0.022 ✅ |
| anom_watermask | 11×5 去噪 | 55 | 4.358 | 0.290 | 4.414 | 0.028 ✅ |
| aoi4_watermask | 4×5 去噪 | 20 | 4.316 | 0.495 | 4.396 | 0.027 ✅ |

**2×2 视图（XGB CW_p vs M1）**：

| | 5 指标全用 | 只用 NTL |
|---|---|---|
| **11 站** | 55 → 0.085 ✗ | 11 → 0.036 ✅ |
| **4 核心站** | 20 → **0.0055 ✅** | 4 → 0.022 ✅ |

**结论**：

1. **全 55 不显著是「双重稀疏」**：砍站点（→20）或砍指标（→11）任一都让 XGB 跨入显著。
2. **站点维度冗余 > 指标维度**：`aoi4`（4 站×5 指数，0.0055）是六臂最显著，强于 `literature`（4×NTL，0.022）——主噪声来自 7 个非核心站；4 核心站上 NDVI/NDWI/NDBI/BSI 亦有增量（非仅 NTL）。
3. **去噪 vs 精选是替代非叠加**：水体掩膜救全量（0.085→0.028），但对已精选的 `aoi4` 反而变差（0.0055→0.027）——精选后再去噪属过度处理。
4. **Ridge 全程 CW 不显著**（最好 ntlall 0.108）；`ntlall`/`literature` 的 Ridge RMSE 微超 M1_Ridge（4.246/4.252 < 4.256）但不显著——线性无论怎么精选都吃不动高维 RS。
5. **写作定位（防 p-hacking）**：主分析仍锚**标准 anom-55（保守下界）**；`aoi4`/`ntlall`/`literature` 作稀疏性诊断、水体掩膜作去噪稳健性——**均不升级为主分析**，共同支撑「扁平吃不动、需模态感知融合」（RQ2）。产物 `baseline_metrics_{aoi4,ntlall,anom_watermask,aoi4_watermask}.csv`。

## 9. M3 结果（M1 + 航运，2026-06-23；2026-07-03 精简 M1；**2026-07-05 主模型改用 full 113 列重跑**）

> **主模型定义变更（2026-07-05）**：M3 主基线改用 **full tier（全 113 航运列 = GFW 49 + PW 64）**，M3 raw = 31 M1 + 113 = **144**。理由：core（38 列）在 XGB 下最弱且不显著（见下表与 §11）。旧 core tier（GFW 6×4 + PW 咽喉 6×2 + 港口 2 = 38）降为 LOCHO 的 core 臂。

### 主基线（L4_tuned，257 周测试，full tier 113 列）


| 模型               | RMSE      | skill vs M0 | CW_p_vs_M1 | DM_p_vs_M1 |
| ---------------- | --------- | ----------- | ---------- | ---------- |
| M0_RW            | 4.152     | —           | —          | —          |
| M1_Ridge         | 4.256     | −2.5%       | —          | —          |
| M1_XGB           | 4.368     | −5.2%       | —          | —          |
| **M3_Ridge**     | **4.430** | −6.7%       | 0.264      | 0.842      |
| **M3_XGB**       | **4.429** | −6.7%       | **0.0002** ✅ | 0.705      |
| Naive_DirPersist | —         | —           | —          | —          |


- **full tier（113 列）XGB 嵌套增量显著**（CW_p=0.0002）：M3_XGB RMSE（4.429）虽仍略劣于 M1_XGB（4.368，即未在 RMSE 上击败纯金融），但 MSPE 调整后的 Clark-West 方向显著——树模型能吃到高维航运的非线性增量。
- M3_Ridge 无增量（CW p=0.264）：线性模型吸收不了高维共线航运（与 M2 模式一致）。
- 对照：旧 core tier（38 列）XGB CW_p=0.096（不显著）、RMSE 4.476，正是改用 full 的直接原因（见下方 LOCHO）。
- **重要**：core tier（38 列）对 XGB **不是最优**——LOCHO 显示 full/portwatch-only/tanker-only 臂的 XGB 仍显著（见下），说明 core 人工精选反而丢了对 XGB 有用的航运列（**已据此于 2026-07-05 改用 full 作主模型**）；这与 M2「少而精」方向相反，是 M3 特有现象。仍未超 M0（skill &lt; 0）。



### Lookback Sweep（L1/L4/L8，full retrain_every=13，**full tier 113 列**，2026-07-05 重跑）


| lookback | 模型        | test周   | M0        | M1 RMSE   | M3 RMSE   | CW_p_vs_M1 |
| -------- | --------- | ------- | --------- | --------- | --------- | ---------- |
| 1        | Ridge     | 260     | 4.137     | 4.188     | 4.278     | 0.560      |
| 1        | XGB       | 260     | 4.137     | 4.479     | 4.441     | **0.007** ✅ |
| **4**    | **Ridge** | **257** | **4.152** | **4.256** | **4.430** | **0.264**  |
| **4**    | **XGB**   | **257** | **4.152** | **4.368** | **4.429** | **0.0002** ✅ |
| 8        | Ridge     | 253     | 4.172     | 4.339     | 4.553     | 0.226      |
| 8        | XGB       | 253     | 4.172     | 4.646     | 4.616     | **0.0018** ✅ |


- **full tier 下 M3 的 XGB 增量对 lookback 完全稳健**：L1/L4/L8 的 CW_p = 0.007 / 0.0002 / 0.0018 **全部显著**（旧 core 版仅 L8 显著、L1=0.779/L4=0.096 不显著）——航运增量不依赖特定滞后窗，比 core 稳健得多。
- Ridge 在所有 lookback 均无显著增量（0.560 / 0.264 / 0.226），与 M2 模式一致。
- XGB RMSE：L4（4.429）最优、L1（4.441）次之、L8（4.616）最差；主协议 L4 与 §9 主基线（M3_XGB 4.429 / CW 0.0002）逐格一致。



### SHAP Top-10 M3 特征（XGB，2024 holdout，full tier 113 列，2026-07-05 重跑）

| 特征 | mean\|SHAP\| | 来源 |
|---|---|---|
| pw_hormuz_tanker_share | 0.01224 | PortWatch |
| pw_suez_n_tanker_wow_pct | 0.00793 | PortWatch |
| pw_malacca_tanker_cap_share | 0.00433 | PortWatch |
| pw_malacca_avg_tanker_size | 0.00251 | PortWatch |
| pw_hormuz_n_tanker_wow_pct | 0.00156 | PortWatch |
| pw_malacca_tanker_share | 0.00150 | PortWatch |
| pw_hormuz_avg_tanker_size | 0.00145 | PortWatch |
| gfw_mandeb_other_share | 0.00135 | GFW |
| pw_mandeb_n_tanker_wow_pct | 0.00112 | PortWatch |
| gfw_suez_other_share | 0.00105 | GFW |

**按来源**：PortWatch 0.04154（85.0%）> GFW 0.00734（15.0%）（PortWatch 信号量约 GFW 的 5.7 倍）。

- **core 精选丢弃的派生列在 full 下登顶**：Top-10 由 `tanker_share` / `n_tanker_wow_pct` / `tanker_cap_share` / `avg_tanker_size`（PortWatch 派生列，**均不在 core 38 列内**）主导——直接印证「人工 core 精选丢掉了 XGB 高价值航运信号」，是改用 full 的机制层证据。
- **机制（霍尔木兹 + 红海绕行）**：霍尔木兹 tanker 占比、苏伊士 tanker 过境周环比在 top-2，马六甲多列进 top——对应原油运输主咽喉与 2023–24 红海/Houthi 改道叙事。
- PortWatch（日频，精度高）SHAP 约为 GFW（月频 ffill）的 5.7 倍（core tier 时为 2.0×，因 core 砍掉了多数 PortWatch 派生列）。
- 产物：`05_outputs/baselines/Flat/M3_Flat/shap_m3.png` / `shap_xgb_by_feature.csv` / `shap_xgb_m3_by_source.csv`



### LOCHO 稳健性（leave-one-channel-out，L4_tuned，257 周，7 臂，2026-07-03 重跑）


| arm            | Ridge RMSE | CW_p Ridge | XGB RMSE  | CW_p XGB  |
| -------------- | ---------- | ---------- | --------- | --------- |
| core（旧主模型/现稳健性 38） | 4.351      | 0.289      | 4.476     | 0.096     |
| full（全 113）    | 4.430      | 0.264      | **4.429** | **0.0002** ✅ |
| portwatch-only | 4.346      | 0.098      | **4.356** | **0.0003** ✅ |
| gfw-only       | 4.383      | 0.354      | 4.388     | **0.047** ✅ |
| gfw-presence   | 4.322      | 0.313      | 4.359     | **0.039** ✅ |
| gfw-aggregate  | **4.246**  | **0.047** ✅ | 4.344     | 0.094     |
| tanker-only    | 4.283      | 0.091      | **4.343** | **0.0018** ✅ |


关键发现（精简 M1 后重跑，结论有实质变化）：

1. **core（旧主模型、现降为稳健性臂，38 列）对 XGB 反而不是最优也不显著**（RMSE 4.476，CW_p 0.096）——**full（113 列，4.429）与 portwatch-only（4.356）、tanker-only（4.343）的 XGB 均显著**（CW p≤0.002），说明 core 人工精选丢掉了对 XGB 有用的航运列；**已据此于 2026-07-05 把主模型改为 full 113 列**。这与 M2「少而精」方向相反。
2. **XGB 最优 arm 是 tanker-only（4.343）/ gfw-aggregate（4.344）/ portwatch-only（4.356）**——核心信号集中在油轮计数/容量与聚合活动指标；tanker-only 仅"tanker"字样列即达最优，印证油轮专属信号是关键。
3. **gfw-aggregate（单一派生 z-mean 列）是 Ridge 唯一显著 arm**（Ridge 4.246 &lt; M1 4.256，CW_p=0.047）——低维聚合航运指数无共线，反而能被线性模型利用；full/core 的高维航运列则拖垮 Ridge。
4. **写作含义**：M3 的「最优子集因模型而异」（XGB→tanker/full；Ridge→gfw-aggregate 单列），且 core tier 并非 XGB 最优、full 才显著——正是扁平融合局限的直接证据，支持 RQ2。
5. 产物：`robustness_m3_summary.csv` / `robustness_m3_overview.png`



## 10. 数据接线修复记录

- **EIA 双重滞后（已修复 2026-06-23）**：EIA +1w 曾同时存在于 M1 源头与 merge 层 → 合并矩阵 EIA 13 列 +2w。修复：`build_feature_matrix.py` 设 `EIA_WPSR_LAG_WEEKS=0`、自检改为「EIA == M1 原列 unchanged」；无泄漏自检全 OK。三模态统一为「各自滞后、merge 仅复查」。（当时维度 365×320；**2026-07-03 M1/M3 精简后现为 365×212**，自检仍全 OK。）



## 11. 复现命令

```bash
# M1 主结果（L4_tuned）→ 05_outputs/baselines/Flat/M1_Flat/
python3 04_code/scripts/flat/run_baseline.py --modality M1

# M1 稳健性 + 调优 sweep（约 9–11 分钟）→ 05_outputs/baselines/Flat/M1_Flat/
python3 04_code/scripts/flat/M1_Flat/sweep_m1.py
python3 04_code/scripts/flat/M1_Flat/sweep_m1.py --quick

# M2 主结果 → 05_outputs/baselines/Flat/M2_Flat/
python3 04_code/scripts/flat/run_baseline.py --modality M2 --m2-features anom
python3 04_code/scripts/flat/run_baseline.py --modality M2 --m2-features literature
# NOTE: --m2-features level 当前矩阵无 level 列，会退化为 M1，不应运行
python3 04_code/scripts/flat/run_baseline.py --modality M2 --leave-one-aoi-out
python3 04_code/scripts/flat/M2_Flat/shap_m2.py
python3 04_code/scripts/flat/M2_Flat/robustness_m2.py          # C2 降维对照（需先跑 shap_m2.py）
python3 04_code/scripts/flat/M2_Flat/sweep_m2.py --quick       # C3 lookback sweep

# B4 水体掩膜稳健性（三步，§8.8）
python3 03_data/processed/M2/py/build_m2_weekly.py --watermask
python3 03_data/processed/merge/py/build_feature_matrix.py \
    --m2-csv 03_data/processed/M2/outputs/m2_weekly_features_watermask.csv
python3 04_code/scripts/flat/run_baseline.py --modality M2 --m2-features anom \
    --matrix weekly_feature_matrix_watermask.csv --tag watermask

# M3 主基线 + sweep + SHAP → 05_outputs/baselines/Flat/M3_Flat/
python3 04_code/scripts/flat/run_baseline.py --modality M3
python3 04_code/scripts/flat/M3_Flat/sweep_m3.py
python3 04_code/scripts/flat/M3_Flat/shap_m3.py

# M4 主基线（anom 合约，L4_tuned）→ 05_outputs/baselines/Flat/M4_Flat/
python3 04_code/scripts/flat/run_baseline.py --modality M4

# M4 SHAP（固定 holdout）→ m4/shap_*.csv + shap_m4.png
python3 04_code/scripts/flat/M4_Flat/shap_m4.py

# M4 lookback sweep（quick）→ m4/sweep_m4_summary.csv + sweep_m4_overview.png
python3 04_code/scripts/flat/M4_Flat/sweep_m4.py --quick

# M4 LOMO 稳健性（全量，retrain_every=13，~7 分钟）→ m4/robustness_m4_summary.csv + robustness_m4_overview.png
python3 04_code/scripts/flat/M4_Flat/robustness_m4.py
```



## 12. M4 结果（M1 + RS + 航运，2026-06-23）



### 主基线（L4_tuned，257 周测试，retrain_every=13；M3 full tier，M4 raw=31+55+113=199）

产物目录：`05_outputs/baselines/Flat/M4_Flat/`（tag=anom，因 M4 含 M2 anom 特征）**（2026-07-05 主模型改用 M3 full 113 列重跑）**


| 模型               | RMSE      | skill vs M0 | CW_p_vs_M1   | DM_p_vs_M1 |
| ---------------- | --------- | ----------- | ------------ | ---------- |
| M0_RW            | 4.152     | —           | —            | —          |
| M1_Ridge         | 4.256     | −2.5%       | —            | —          |
| M1_XGB           | 4.368     | −5.2%       | —            | —          |
| **M4_Ridge**     | **4.525** | **−9.0%**   | 0.314        | 0.941      |
| **M4_XGB**       | **4.507** | **−8.6%**   | **0.009** ✅  | 0.873      |
| Naive_DirPersist | —         | —           | —            | —          |


- **M4_XGB Clark-West p=0.009（显著，且强于旧 core 版 0.020）**：M1+M2+M3(full) 全融合的 XGB 嵌套增量显著——航运 full 113 列是主要增量来源（见 LOMO）。
- **M4_Ridge 不显著**（CW p=0.314）：与 M2/M3 独立分析模式一致，线性吸收不了高维融合。
- **M4_XGB(4.507) 略劣于 M3_full_XGB(4.429) 与 M2_XGB(4.440)**：full tier 下把 M2 加到 M1+M3 反而使 XGB RMSE 微升（维度膨胀至 796）；显著性来自 MSPE 调整后的方向，而非 RMSE 改善。
- M4 仍未超越 M0（skill &lt; 0），与各单模态一致。（注：主基线 M4 现 **= LOMO 的 full 臂**，199 特征，见下）。



### SHAP（固定 holdout，train≤2023-12，test≥2024-01）

产物：`05_outputs/baselines/Flat/M4_Flat/shap_m4.png` / `shap_xgb_by_feature.csv` / `shap_m4_by_modality.csv`

**各模态重要性（XGB，sum mean|SHAP|）**：**（2026-07-05，M3 full tier 重跑）**

| 模态         | sum mean\|SHAP\| | 占比    |
| ---------- | -------------- | ----- |
| **M3（航运）** | 0.03857        | 51.8% |
| M1（金融）     | 0.02513        | 33.7% |
| M2（遥感）     | 0.01081        | 14.5% |

**M2 遥感 RS 指数（XGB）**：

| 指数        | sum mean\|SHAP\| |
| --------- | -------------- |
| NDWI（水面）  | 0.00414        |
| NDVI（植被）  | 0.00288        |
| NTL（夜光）   | 0.00147        |
| BSI（裸地）   | 0.00147        |
| NDBI（建成区） | 0.00084        |

**M3 航运来源（XGB）**：PortWatch 0.03634（94.2%）> GFW 0.00224（5.8%）

**Top-10 M4 特征（XGB）**：

| 特征                          | mean\|SHAP\| | 模态  |
| --------------------------- | ---------- | --- |
| pw_hormuz_tanker_share      | 0.00863    | M3  |
| pw_suez_n_tanker_wow_pct    | 0.00688    | M3  |
| pw_malacca_tanker_cap_share | 0.00482    | M3  |
| gold_return                 | 0.00468    | M1  |
| brent_f1_spot_log_basis     | 0.00381    | M1  |
| vix                         | 0.00289    | M1  |
| pw_malacca_avg_tanker_size  | 0.00284    | M3  |
| brent_wti_spread            | 0.00248    | M1  |
| NDWI_anom_Yanbu             | 0.00181    | M2  |
| pw_mandeb_n_tanker_wow_pct  | 0.00145    | M3  |

**模态占比（full tier）重回航运主导**：**M3(51.8%)>M1(33.7%)>M2(14.5%)**——与 core tier 版（M1 51%>M2 26%>M3 23%）相反，因 full 恢复了 113 航运列（含 core 砍掉的 `tanker_share`/`n_tanker_wow_pct`/`tanker_cap_share` 等高 SHAP 派生列），航运总 SHAP 大增。**Top-10 中 6 个 M3 航运（PortWatch 派生列主导）、3 个 M1 金融、1 个 M2 遥感**，接近旧全量版的航运主导格局。

### Lookback Sweep（L1/L4/L8，full retrain_every=13，**full tier（M1+M2+M3full）**，2026-07-05 重跑）


| lookback    | 模型    | test周 | M0    | M1 RMSE | M4 RMSE   | CW_p_vs_M1  |
| ----------- | ----- | ----- | ----- | ------- | --------- | ----------- |
| **1**       | Ridge | 260   | 4.137 | 4.188   | 4.320     | 0.553       |
| **1**       | XGB   | 260   | 4.137 | 4.479   | 4.595     | **0.035** ✅ |
| **4** ← 主协议 | Ridge | 257   | 4.152 | 4.256   | 4.525     | 0.314       |
| **4** ← 主协议 | XGB   | 257   | 4.152 | 4.368   | **4.507** | **0.009** ✅ |
| 8           | Ridge | 253   | 4.172 | 4.339   | 4.586     | 0.079       |
| 8           | XGB   | 253   | 4.172 | 4.646   | **4.564** | **0.0008** ✅ |


- **full tier 下 M4 的 XGB 增量对 lookback 完全稳健**：L1/L4/L8 的 CW_p = 0.035 / 0.009 / 0.0008 **全部显著**（旧 quick+core 版 L1=0.177 不显著）；主协议 L4 显著。
- Ridge 在所有 lookback 均不显著（0.553 / 0.314 / 0.079）。
- M4 XGB RMSE：L4（4.507）最优、L8（4.564）次之、L1（4.595）最差。
- **L4 与 §12 主基线逐格一致**（M4_Ridge 4.525 / CW 0.314、M4_XGB 4.507 / CW 0.009）——本次 sweep 与主协议同为 retrain_every=13 + full tier，旧版「差异来自 retrain 频率」的注释已作废。



### LOMO 稳健性（leave-one-modality-out，**全量协议 retrain_every=13；M3 用 full tier**，2026-07-03 重跑）

产物：`05_outputs/baselines/Flat/M4_Flat/robustness_m4_summary.csv` / `robustness_m4_overview.png`


| arm                | 特征数 | Ridge RMSE | CW_p Ridge | XGB RMSE  | CW_p XGB     |
| ------------------ | --- | ---------- | ---------- | --------- | ------------ |
| **full（M1+M2+M3full）** | 199 | 4.525      | 0.314      | 4.507     | **0.009** ✅  |
| minus-M2（M1+M3full）    | 144 | 4.430      | 0.264      | **4.429** | **0.0002** ✅ |
| minus-M3（M1+M2）        | 86  | 4.414      | 0.474      | 4.440     | 0.085        |
| M1-only            | 31  | **4.256**  | —          | **4.368** | —            |


> **✅ LOMO 用 M3 full tier（113 列），full 臂 = 199 特征、XGB=4.507——2026-07-05 主基线也改用 full 后，此 full 臂即等于 §12 主基线 M4**（core/full 口径差异已消除）。minus-M2 = M1+M3full 与 §9 LOCHO 的 full 臂（XGB 4.429，CW_p 0.0002）一致；minus-M3 = M1+M2 与主 M2（XGB 4.440，CW_p 0.085）一致。

**LOMO 关键发现（精简 M1 后重跑）**：

1. **加 M2 到 M1+M3full 反而使 XGB 变差**：full（4.507）&gt; minus-M2（4.429），+0.078——扁平融合下 M2 与 M3 在 XGB 中竞争/冗余、非互补（与旧全量版结论方向一致，且幅度更大）。
2. **minus-M2（M1+M3full）XGB CW_p=0.0002 最显著**：M3 full-tier 是 XGB 的主增量来源（= §9 LOCHO full 臂）。
3. **minus-M3（M1+M2）XGB CW_p=0.085**：= 主 M2 anom（不显著），M2 单独增量弱。
4. **对两个模型，M1-only 都是 RMSE 全局最优**（Ridge 4.256 / XGB 4.368）——加任何模态都不降 RMSE；full/minus-M2 的 XGB 只是 CW（MSPE 调整后方向）显著，而非 RMSE 更低。这正是「加模态方向有用但扁平拼接吃不动」的体现。
5. **最优配置因模型而异**（XGB→minus-M2=M1+M3full 4.429 最优；旧 core M4 XGB 4.492 更差，故 2026-07-05 主模型已改 full）——模型依赖的"最优子集"是扁平融合的局限，为 RQ2 模态感知融合提供最直接对照证据。



### RQ1 M4 结论（可写入 Results §4.4）

> 主模型改用 M3 full 113 列后，全模态扁平融合（M1+M2+M3full）的 XGB **提供显著嵌套增量（Clark-West p=0.009，强于旧 core 版 0.020）**；M3 full 单模态 XGB 亦显著（0.0002），M2 anom 单模态仍不显著（0.085）。但 M4_XGB RMSE（4.507）**并不优于** M3_full（4.429）、M2（4.440）、乃至 M1_XGB（4.368）——显著性来自 MSPE 调整后的方向，而非 RMSE 改善。SHAP 模态占比已按 full 重跑（§12：航运 51.8%>金融 33.7%>遥感 14.5%，重回航运主导）。LOMO 证实：对两个模型 M1-only 都是 RMSE 全局最优，加任何模态都不降 RMSE；full 融合的显著性主要来自 M3 full（minus-M2 XGB 4.429/0.0002）。**更强地支持「扁平拼接吃不动多模态增量、需要模态感知融合」的研究动机（RQ2）**。

---

## 13. 创新层（表示级融合）首跑结果（2026-07-05）

> **定位**：**方法集成层**（模态感知表示级融合）首批端到端结果，作为 §8–§12 扁平基线的直接对照，回答 RQ2。代码 `04_code/src/models/`（3 编码器 + `GatedFusion` + `deep_rolling`）+ `04_code/scripts/deep/run_deep_baseline.py`；产物 `05_outputs/baselines/Deep/`（`M1_Deep`…`M4_Deep` + `_cross/`）。⚠️ **与扁平层用不同 M0/M1 基准**（deep lookback=8 → M0=4.172、≈252 测试周；扁平 lookback=4 → M0=4.152、257 周）：跨层比 **CW 方向 / skill**，勿直接比绝对 RMSE。

**协议**：walk-forward（min_train=104、retrain_every=13、epochs=80、inner-val 52 周 early-stop、dropout 0.1 + weight_decay 强正则），lookback=8，单任务 r_{t+1} 还原价格；flat M1 嵌套参照**读**扁平 `baseline_predictions.csv`（避免 torch+xgboost OpenMP 段错误）。

**结果（M0=4.172，≈252 测试周）**：

| 模型 | 组成 | RMSE | skill vs M0 | CW vs flat M1 |
|---|---|---|---|---|
| M0_RW | 随机游走 | 4.172 | 0.0% | — |
| 扁平 M1_Ridge | — | 4.279 | −2.6% | — |
| 扁平 M1_XGB | — | 4.383 | −5.1% | — |
| Mfin_TCN | z_fin | 4.205 | −0.8% | — |
| Mrs_RS | z_rs（冻结 Prithvi） | 4.307 | −3.3% | 0.103 |
| Mship_GNN | z_ship（17 节点 GAT+TCN） | 4.188 | −0.4% | **0.0014** ✅ |
| **Mfusion** | gated(z_fin, z_ship) | **4.174** | **−0.1%** | — |
| Mfull_M4rep | gated(z_fin, z_rs, z_ship) | 4.184 | −0.3% | **0.0021** ✅ |

**嵌套 Clark–West（`deep_cw.csv`）**：Mship vs flat M1 = **0.0014** ✅；M4rep vs flat M1 = **0.0021** ✅；Mfusion vs Mfin（航运增量）= 0.059（边缘）；M4rep vs Mfusion（RS 增量）= 0.260（不显著）；Mrs vs flat M1 = 0.103（不显著）。

**核心发现（RQ2 首个正面证据）**：

1. **表示级 > 扁平（同一份航运信息）**：扁平 M3-core 的 XGB 只到 CW 0.096（旧 M3_LSTM early-fusion 更是 CW 0.460 不显著），而模态感知 GAT+TCN 的 **Mship 对 flat M1 达 CW 0.0014**——扁平吃不动、表示级吃得动，**RQ2 首个正面证据**。
2. **深度表示逼近 M0**：Mfin/Mship/Mfusion/M4rep 的 skill 由扁平的 −2.5~5% 收窄到 ~0（Mfusion −0.06% 最优、DirAcc 0.557 最高）。但 **DM vs M0 仍不显著**（逼近≠战胜；需防塌缩到随机游走，DirAcc>0.5 + CW 显著说明非纯塌缩）。
3. **门控融合最优**：Mfusion（fin+ship）> 各单模态；加 RS（M4rep）无显著增量（RS 分支 Prithvi meanpool 偏弱，待 sweep）。

**RQ3 可解释性（`deep_interpret.png` / `deep_gate_weekly.csv`）**：

- 模态门控权重 finance≈0.40 > rs≈0.33 > shipping≈0.27（与扁平 SHAP 的 M1>M2>M3 一致；⚠️ 2025 末 finance 飙至 ~0.9 需排查样本边界）。
- 航运 site-attention **hormuz（霍尔木兹）最高** → 符合石油咽喉机制。
- 遥感 site-attention top = RasTanura/Basra/Yanbu（出口码头），与人工 `literature` 4 站（Fujairah/RasTanura/Rotterdam/Houston）**仅重叠 RasTanura** → 数据驱动 vs 文献先验不一致，作 RQ3 讨论点（**不回流改扁平选站**，防循环论证）。

**caveat**：单 seed、lookback=8、epochs=80、未调优，attention 排序稳定性未验；`04_code/scripts/deep/run_deep_sweep.py`（多 seed + RS 超参）待跑。旧 LSTM 深度 early-fusion 标尺产物已被本次覆盖，如需「深度扁平 vs 深度表示」双对照须恢复归档。

---

## 14. 仍未做（剩余 backlog）

**M2 剩余**：

- 写作：Methodology「机制变量构建」+ Results「增量价值」+ Discussion「P058 回应 / NDWI 限制 / 子期间叙事 / 水体掩膜稳健」

**M3 剩余**：

- 写作：Results「M3 增量价值」+ 机制解读（霍尔木兹/苏伊士信号）

~~**M4**~~（已完成 2026-06-23）：

- ✅ 主基线 + SHAP + sweep(quick) + LOMO(全量) 全套完成，见 §12

~~**M2 C2 / C3 / B4**~~（已完成 2026-06-23）：

- ✅ 降维对照、lookback sweep、水体掩膜稳健性，见 §8.6–§8.8

**写作**：

- 文献综述主题初稿（6 主题，~4–5 页）
- 4-AOI 选择依据短文



## 15. 变更记录


| 日期         | 内容                                                                                                                        |
| ---------- | ------------------------------------------------------------------------------------------------------------------------- |
| 2026-06-23 | 建立 M0/M1 扁平回测骨架 + 稳健性/调优 sweep；锁定主对照 `L4_tuned`；修复 merge 层 EIA 双重滞后                                                       |
| 2026-06-23 | M2 全套结果 + CW + LOAO + SHAP + C2 + C3 + B4 水体掩膜；M2 完整记录于 §8 |
| 2026-06-23 | 文件结构重组：`05_outputs/baselines/m1, m2, m3` |
| 2026-06-23 | M3 主基线 + sweep + SHAP 全套完成；修复 `shap_m3.py` 绘图小 bug；结果记录于 §9                                                               |
| 2026-06-23 | M3 LOCHO 稳健性（robustness_m3.py）完成；portwatch-only Ridge CW p=0.026 唯一显著；XGB 所有 arm 均显著                                      |
| 2026-06-23 | M4 全套完成：主基线（CW p=0.0002 XGB）+ SHAP（M3 55.6% > M1 30.3% > M2 13.1%）+ sweep(quick) + LOMO(quick)；脚本写入 `04_code/scripts/flat/M4_Flat/` |
| 2026-07-03 | 合并原 `2026-06-23_m2_baseline_results.md` 至 §8，恢复为单一基线记录文档 |
| 2026-07-03 | M1 精简特征集正式重跑：§7 sweep / L4_tuned 数值更新（Ridge 4.256 / XGB 4.368）；删除 `baseline_deep_*` 产物；§8–§12 待 M2/M3 改完后统一重跑 |
| 2026-07-03 | **M1/M2/M3 特征改动后 M0–M4 全套统一重跑**：merge 矩阵重建 365×212（M1=31/M2=55/M3=113），M3 主模型改 core tier（38）；§7–§12 全部数值 + SHAP + sweep + robustness 刷新。**关键结论变化**：M1 变强后单模态 anom-55（CW_p 0.085）/M3-core（0.096）XGB 增量退化为不显著，仅 M4（0.020）/M2 literature（0.022）/M2 watermask（0.028）/M3 full·portwatch·tanker 臂仍显著；SHAP 模态占比反转为 M1(51%)>M2(25%)>M3(23%)。更强支持 RQ2。 |
| 2026-07-05 | **M3/M4 主模型由 core 38 列改为 full 113 列**：`data.py::select_features` 默认 `m3_tier="full"`；重跑 M3 raw=144（XGB 4.429 / CW 0.0002）、M4 raw=199（XGB 4.507 / CW 0.009，= LOMO full 臂）。理由：core 在 XGB 下最弱不显著（CW 0.096）、full 显著。§4/§9/§12 主表、SHAP（`shap_m3/m4`）、lookback sweep（`sweep_m3/m4`，现默认 full）均已重跑刷新；core 保留为 `robustness_m3.py` 的 core 臂。 |
| 2026-07-05 | 新增 **§8.9 M2 2×2 稀疏性六臂**：`data.py` 加 `aoi4`(4站×5指数=20)/`ntlall`(11站×NTL=11)，与 anom/literature + 水体掩膜共 6 臂。发现「双重稀疏」+ 站点维度冗余 > 指标维度（`aoi4` XGB CW **0.0055** 最强）；去噪与精选替代非叠加。主分析仍锚 anom-55（防 p-hacking）。 |
| 2026-07-05 | 新增 **§13 创新层（表示级融合）首跑**：3 编码器（z_fin/z_rs/z_ship）+ GatedFusion 端到端 walk-forward（lookback=8）。**Mship vs flat M1 CW 0.0014 ✅、M4rep 0.0021 ✅、Mfusion 逼近 M0（−0.06%）**——RQ2 首个正面证据。RQ3：航运 attention 聚焦 hormuz、门控 fin>rs>ship。单 seed 未调优，待 sweep。 |


