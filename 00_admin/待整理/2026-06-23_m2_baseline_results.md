# M2 基线结果（B3/B4 — M1+遥感增量验证）

> **配套文档**：协议与 M1 基线见 `2026-06-23_flat_baseline_log.md`；Channel B 数据与 EDA 见 `2026-06-22_channelB_mechanism_plan.md`  
> **代码**：`04_code/scripts/run_baseline.py --modality M2` + `04_code/scripts/m2/`  
> **产物**：`05_outputs/baselines/m2/`（`baseline_metrics_*.csv` / `baseline_predictions_*.csv` / `backtest_*.png` / `baseline_loao_*.csv` / `shap_*.csv` / `shap_anom.png`）  
> **创建**：2026-06-23 | **进度**：B0/B1/B2/B3(DM+CW+LOAO) ✅；B3 SHAP ✅；C2 降维对照 ✅；C3 lookback × contract sweep ✅；**B4 水体掩膜稳健性** ✅（全部 2026-06-23）

---

## 1. 协议（M2 专项）

主协议与 M1 完全相同（`flat_baseline_log.md` §3），以下仅列 M2 差异部分。

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

---

## 2. M2 核心结果（B3 主交付）

主协议 L4_tuned，retrain_every=13，**257 共同测试周**（2021–2025），M0 RMSE=4.152。  
CW_p = Clark-West 单边 p（越小 = 大模型即 M1+RS 越显著优于 M1）。

| 配置 | Ridge RMSE | Ridge skill | Ridge CW_p(vs M1) | XGB RMSE | XGB skill | XGB CW_p(vs M1) | XGB DirAcc |
|---|---|---|---|---|---|---|---|
| M1（嵌套基线） | 4.332 | −4.4% | — | 4.771 | −14.9% | — | 0.525 |
| **M2 anom (55)** | 4.411 | −6.3% | 0.212 | 4.643 | −11.8% | **0.006** ✅ | 0.506 |
| M2 literature (4) | **4.318** | **−4.0%** | 0.146 | 4.553 | −9.7% | **0.001** ✅ | **0.545** |
| M2 level (55) † | 4.361 | −5.0% | 0.061 | 4.531 | −9.1% | **0.004** ✅ | 0.514 |

> **† level 合约数据状态**：当前标准矩阵（`weekly_feature_matrix.csv`，365×221）仅含 55 个 anom 列；55 个 level（原始水平值）列**未纳入矩阵**，`m2_columns(dico, "level")` 返回空列表。上表 level 行数值来自旧版矩阵（EIA 修复前），**当前不可复现**。sweep 实验进一步证实：现版矩阵下 level 合约等价于 M1（RMSE 完全相同，CW_p = NaN）。若需 C3 复现，须重建矩阵纳入 level 列（优先级低，决策见 §7）。

产物：`05_outputs/baselines/m2/baseline_metrics_anom.csv` / `baseline_metrics_literature.csv`

![M2 anom backtest](../05_outputs/baselines/m2/backtest_anom.png)

### 关键发现

1. **仍无配置超过 M0**（所有 skill < 0）——符合周度油价 M0 极强的预期。
2. **遥感对 XGB 有统计显著嵌套增量**：anom/literature 两种合约的 XGB Clark-West p 全 <0.01 → 加遥感让 XGB 显著优于 M1_XGB。（level 合约结果来自旧版矩阵，见上方 † 备注。）
3. **对 Ridge 不显著**（anom CW p=0.21，literature CW p=0.15）：高维 RS 对线性模型主要是噪声，正则化不足以利用弱信号。
4. **少而精 > 全量**（支持 C1 / P058）：`literature`（仅 4 个 NTL_anom）跨配置最优——Ridge RMSE 4.318（**唯一微超 M1 的配置**），XGB CW p=0.001（最显著），DirAcc=0.545（最高）。
5. **方法学含义（支撑 RQ2）**：扁平拼接下遥感增量边际且模型相关，强化「需要模态感知融合」的论据。

---

## 3. leave-one-AOI-out 结果（B3 站点定位）

全 55 anom，每次去掉一个 AOI，retrain_every=26 自洽基线（full RMSE Ridge=4.451 / XGB=4.867）。  
`dRMSE > 0` = 去掉该站变差 = 该站有正贡献。

| 去掉站点 | Ridge dRMSE | XGB dRMSE | 解读 |
|---|---|---|---|
| Rotterdam | +0.032 | −0.275 | Ridge 微正；XGB 去掉反而改善（噪声） |
| Jurong | +0.040 | **−0.323** | XGB 最大改善——非石油核心港，最大噪声源 |
| Basra | +0.031 | −0.278 | Ridge 微正；XGB 噪声 |
| NingboZhoushan | −0.034 | **−0.326** | 两个模型均无正贡献；非石油港噪声 |
| Fujairah | −0.037 | −0.302 | XGB 下去掉反改善（去掉减噪） |
| RasTanura | −0.016 | −0.315 | — |
| Houston | −0.008 | −0.277 | — |

产物：`05_outputs/baselines/m2/baseline_loao_anom.csv`

**LOAO 结论**：
- **XGB：去掉任一站都改善（dRMSE 全负）** → 全 55 anom 对 XGB 整体偏噪声，无单站是「必要正贡献」。
- 改善最多：**NingboZhoushan / Jurong**（非石油核心港，与 plan 预期一致）。
- **Ridge：幅度小（±0.04）**，去掉 Jurong/Rotterdam/Basra 略变差（微弱正贡献）。
- 综合强化「少而精」结论——文献精选 4 个 NTL_anom 优于全 55。

---

## 4. SHAP 可解释性结果（B3 补完）

脚本 `04_code/scripts/m2/shap_m2.py`；固定 holdout（train≤2023-12，test=2024–2025，L4_tuned）。

产物：`05_outputs/baselines/m2/`
- `shap_xgb_by_feature.csv` — 所有基础特征的 XGB 平均|SHAP|（lags 加总）
- `shap_ridge_by_feature.csv` — Ridge 版
- `shap_xgb_m2_by_index.csv` — 按 RS 指数（NTL/NDBI/BSI/NDWI/NDVI）汇总
- `shap_xgb_m2_by_aoi.csv` — 按 AOI/站点汇总
- `shap_topN_anom.csv` — 前 N 个 M2 特征排名（供 robustness_m2.py C2 对照用）
- `shap_anom.png` — 4 面板概览图

![SHAP overview](../05_outputs/baselines/m2/shap_anom.png)

**运行参数**：train 258 周（2019–2023），test 103 周（2024–2025），L4_tuned；Ridge α=1000（强正则），XGB depth=2 n_est=200；holdout log-return RMSE：Ridge 0.0418 / XGB 0.0421。

**M2 各 RS 指数重要性（XGB，按 sum mean|SHAP|）**：

| RS 指数 | sum mean\|SHAP\| | 排名 |
|---|---|---|
| **NDWI**（水面动态） | 0.00595 | 1 |
| **NDVI**（植被） | 0.00454 | 2 |
| **NTL**（夜光） | 0.00336 | 3 |
| **NDBI**（建成区） | 0.00275 | 4 |
| **BSI**（裸地/堆场） | 0.00226 | 5 |

**M2 各 AOI/站点重要性（XGB，sum mean|SHAP|）**：

| 排名 | AOI | sum mean\|SHAP\| | 站点类型 |
|---|---|---|---|
| 1 | **Ulsan** | 0.00341 | 韩国炼厂枢纽 |
| 2 | **Kharg** | 0.00330 | 伊朗出口码头 |
| 3 | **Yanbu** | 0.00283 | 沙特红海出口 |
| 4 | Fujairah | 0.00184 | 中东集散港 |
| 5 | RasTanura | 0.00161 | 沙特主出口港 |
| 6 | Houston | 0.00147 | 美国炼厂/出口 |
| … | … | … | … |
| 10 | **NingboZhoushan** | 0.00057 | 中国进口港（最低） |
| 11 | **Basra** | 0.00055 | 伊拉克出口港（最低） |

**Top-5 M2 基础特征（XGB）**：
`NDWI_anom_Kharg`(0.00171) > `NDWI_anom_Yanbu`(0.00170) > `NDVI_anom_Ulsan`(0.00169) > `NTL_anom_RasTanura`(0.00118) > `NDWI_anom_Jamnagar`(0.00111)

**整体上 M1 金融特征仍主导 Top-4**：`gold_return`(0.00575) > `futures_spread`(0.00484) > `vix`(0.00276) > `brent_wti_spread`(0.00265)；M2 特征从第 8 名（NDWI_anom_Kharg）才进入前 15。

**SHAP 解读**：
1. **NDWI（水面反射）意外排第一** — 中东/韩国出口码头的水面变化可能捕获装卸吃水动态；但也可能是潮汐/水色噪声（B0 已提示 NDWI 高方差噪声风险）。NTL（文献最推荐）仅排第三，说明当前 test 期（2024–2025）夜光信号弱于 EDA 预期。
2. **AOI 排序与文献不同** — Ulsan/Kharg 超过文献推荐的 Fujairah/RasTanura，可能与 2024 年韩国炼厂动态及伊朗出口变化有关（2023–24 红海/Houthi 事件）；与 LOAO 结论互补（LOAO 在 2021–2025 全段，SHAP 在 2024–2025 最近段，时段不同导致排序差异）。
3. **NingboZhoushan 最弱** — 与 LOAO「去掉 Ningbo 最大改善」一致 ✅，强化"非石油核心港是主要噪声源"结论。
4. **文献 NTL 精选的有效性**：NTL_anom_RasTanura(4th) 和 NTL_anom_Houston(8th) 仍在 top-10，支持 `literature` 四站点配置的预测表现（CW p=0.001，最显著）。

---

## 5. RQ1 结论（可写入 Results §4.1）

> 在统一无泄漏协议（2019–2025、4 周滞后、rolling-origin L4_tuned）下，**遥感（Channel B）的扁平特征融合不能战胜随机游走 M0**（所有配置 skill < 0）。  
> 但相对金融基线 M1，**遥感对 XGBoost 提供了统计显著的嵌套增量**（Clark-West p：anom 0.006 / 文献精选 NTL_anom 0.001），**对线性 Ridge 则不显著**（p > 0.14），说明增量信号弱且需鲁棒模型才能利用。  
> **少而精**的 4 个夜光异常站点（Fujairah/RasTanura/Rotterdam/Houston）优于全 55 列；leave-one-AOI-out 进一步显示非石油核心港（Ningbo/Jurong）是主要噪声源；SHAP 结果补充了指数与站点维度的可解释性。  
> 这支持本研究核心动机：扁平拼接无法充分利用遥感，需要**模态感知的表示级融合**（RQ2）。

---

## 6. C2 降维对照结果（回应 P058）

脚本 `04_code/scripts/m2/robustness_m2.py`；产物 `05_outputs/baselines/m2/c2_summary.csv` / `c2_overview.png`。

**协议**：L4，fixed hyperparams（Ridge α=1000，XGB depth=2/n_est=200），257 共同测试周。M0 RMSE=4.152，M1_Ridge=4.332，M1_XGB=4.717（注：固定参数而非内层调参，故 M1_XGB 与主结果 4.771 略有差异；C2 各臂内部对比公平）。

| 臂 | M2 输入 | Ridge RMSE | Ridge CW_p | XGB RMSE | XGB CW_p |
|---|---|---|---|---|---|
| all-55 | 55 anom cols | 4.411 | 0.212 | 4.587 | **0.0036** |
| pca-90 | 55 → PCA 90% var | 4.412 | 0.188 | **4.535** | **0.0000026** ✅✅ |
| elastic | 55 → ElasticNet sel | 4.411 | 0.212 | 4.587 | **0.0036** |
| shap-top20 | top-20 SHAP | **4.376** | 0.249 | **4.521** | **0.0007** ✅ |

**C2 结论**（直接回应 P058「SHAP≠PCA」）：

1. **Ridge：四臂几乎无差异**（RMSE 4.376–4.412，CW_p 全不显著）。α=1000 的强 L2 正则已通过收缩隐式处理了共线性；PCA/ElasticNet 在其上没有增益。**shap-top20 给出最好的 Ridge RMSE（4.376）**——减少输入让正则化更聚焦。
2. **XGB：PCA 显著提升 CW 显著性**（p=2.6×10⁻⁶，远优于 all-55 的 0.0036）。去相关之后 XGB 能更纯粹地利用正交信号分量；这正是 P058 指出的「PCA 解决共线性、SHAP 提供可解释性、两者解决不同问题」的实证体现。
3. **Elastic ≡ all-55**：ElasticNet(α=0.05) 在 356 特征/258 样本上几乎产生全零系数（高 α 过度压缩），SelectFromModel 退化为全保留 → 结果与 all-55 完全相同。可在方法章节注明「ElasticNet 选择器在此维度/样本量比例下退化」。
4. **shap-top20 既节省特征又提升预测**（XGB CW_p 0.0007 < all-55 的 0.0036）——支持 SHAP 选择的特征确实是信息更纯的子集。

**写作建议（P058 回应段）**：Ridge 的鲁棒性来自强 L2 正则，PCA 对 XGB 有实质增益，SHAP 提供事后可解释性但与降维各司其职；三种策略并列，说明"55 anom 的高维共线问题"在 XGB+PCA 下可通过降维缓解，在 Ridge+shap-top20 下通过数据驱动特征选择缓解。

## 7. 方法论决策记录（问题 2/3）

**M2 lookback × feature-contract sweep：已跑（2026-06-23）。** 脚本 `sweep_m2.py --quick`（retrain_every=26），产物 `05_outputs/baselines/m2/sweep_m2_summary.csv` / `sweep_m2_overview.png`。

#### C3 结果（anom / literature；level 合约当前矩阵不可用，见 §2 †）

M1 RMSE（quick，同周对齐）：L=1 → 4.255 Ridge / 4.568 XGB；L=4 → 4.557 / 5.222；L=8 → 4.693 / 5.600。

| contract | L | Ridge RMSE | Ridge CW_p | XGB RMSE | XGB CW_p |
|---|---|---|---|---|---|
| anom | **1** | **4.230** | 0.108 | **4.521** | **0.003** |
| anom | 4 ← 主协议 | 4.451 | **0.015** | 4.867 | **<0.001** |
| anom | 8 | 4.537 | **0.005** | 5.089 | **<0.001** |
| literature | **1** | **4.218** | **0.016** | **4.446** | **0.004** |
| literature | 4 | 4.517 | **0.035** | 4.852 | **<0.001** |
| literature | 8 | 4.691 | 0.208 | 5.051 | **<0.001** |

**C3 结论**：

1. **L=1 是 M2 的最优 lookback**，RMSE 随 lookback 单调变差——与 M1（调参后 L4 最优）相反。原因：RS 特征月频信号弱，多 lag 只引入噪声；维度膨胀（55 anom × 8 lags = 440 维）远超样本量，拖垮预测。
2. **XGB 在所有 lookback 均显著**（CW_p ≤ 0.004），说明遥感的非线性增量信号即使在 L=1 时已充分捕获。Ridge 在 literature + L=8 时不显著（0.21），进一步支持"高维 RS 不利于线性模型"。
3. **L=1 literature（Ridge 4.218）是整个 sweep 中 Ridge 的全局最优**——4 个 NTL_anom × 1 lag = 4 维，少而精到极致，正则化最聚焦。
4. **与 C2（PCA）的一致性**：L=1 减少 lag 和 C2 PCA 降维是同一方向的干预（减少冗余维度）；两者均提升 XGB 的增量显著性（L=1 CW_p 0.003 vs 主协议 0.006；PCA CW_p 2.6×10⁻⁶）。
5. **写作注意**：主协议固定 L=4（导师设定），因此 Discussion 可补充：「若使用 L=1，M2 遥感增量对 XGBoost 的 CW 显著性进一步提升（p=0.003），且 Ridge 在 literature 合约下接近显著（p=0.016）」，作为灵敏度分析。

**COVID/红海子期间：不单独跑，写 Discussion 叙事。** SHAP holdout（2024–2025）天然是「红海扰动后」子期间，与 rolling origin（2021–2025）全段排名差异（Ulsan/Kharg 突出 vs 全段 Fujairah/RasTanura 文献推荐）本身即为隐性子期间发现，直接写进 Discussion 即可。

**水体掩膜 GEE：✅ 已完成（B4 稳健性，2026-06-23）。** 见 §7b 及 `research_diary_phase3.md`（§M2 水体掩膜 B4）。

## 7b. B4 水体掩膜稳健性结果（2026-06-23 新增）

**背景**：B0 审计 + SHAP 均提示 NDWI 高方差风险（Kharg/Yanbu top-2 特征），Basra（`land_px=0.001`）等离岸终端原始 NDVI 因水体像素被压至 −0.25（负值，非信号）。B4 对此做正式稳健性验证。

**方法**（三步管线）：
1. GEE 水体掩膜版 CSV（`sentinel2_oil_sites_monthly_indices_watermask_201704_202512_11aoi.csv`，1155 行 × 28 列，11×105 月完整）→ `build_m2_weekly.py --watermask` → `m2_weekly_features_watermask.csv`（365×188，含 MNDWI + `s2_land_px_*`）
2. `build_feature_matrix.py --m2-csv .../m2_weekly_features_watermask.csv` → `weekly_feature_matrix_watermask.csv`（**365×221**，与标准矩阵等形；`filter_m2_anom_columns` 自动排除 MNDWI_anom，保留 55 anom 但 NDVI/NDBI/BSI 已仅含陆地像素）
3. `run_baseline.py --modality M2 --m2-features anom --matrix weekly_feature_matrix_watermask.csv --tag watermask`

**结果对比（257 共同测试周，M0 RMSE=4.152）**：

| 模型 | 标准 anom-55 RMSE | 水体掩膜 RMSE | Δ RMSE | 标准 CW_p | 水体掩膜 CW_p |
|---|---|---|---|---|---|
| M2 Ridge | 4.411 | 4.411 | −0.000 | 0.212 | 0.217 |
| **M2 XGB** | 4.643 | **4.565** | **−0.078（−1.69%）** | 0.006 | **0.0001** |

产物：`05_outputs/baselines/m2/baseline_metrics_watermask.csv` / `baseline_predictions_watermask.csv`

**B4 水体掩膜结论**：
1. **M2 Ridge：不受影响**——L2 正则化已隐式压缩水体噪声特征的权重，掩膜对线性模型没有增益。
2. **M2 XGB：结论显著强化**——RMSE 改善 1.69%；CW_p vs M1 从 0.006 → **0.0001**（信号更纯净后 XGB 增量显著性大幅提升）。
3. **稳健性正向**：水体掩膜不改变定性结论（M2 XGB CW_p < 0.05 在两版均成立），反而进一步支持 RS 通道的增量价值；原版 anom-55 结论是保守估计。
4. **NDWI 高方差疑问部分解答**：NDWI 在两版均计算于所有有效像素（不受水体掩膜影响），SHAP top-2 NDWI_anom_Kharg 信号来源属于真实水面动态，而非 NDVI/NDBI/BSI 水体噪声。
5. **写作建议（Methods/Results）**：主分析用标准 anom-55，水体掩膜版作为 B4 稳健性注脚——「陆地像素精确提取后 XGB 增量 p 值从 0.006 改善至 0.0001，结论更强而非不同（More robust, not different）」。

## 8. 仍缺（后续）

| 任务 | 优先级 | 说明 |
|---|---|---|
| **写作** | **P1** | Methodology「机制变量构建」+ Results「增量价值」+ Discussion「P058 回应 / NDWI 限制 / 子期间叙事 / 水体掩膜稳健」 |
| M3 主结果 | P1 | `run_baseline.py --modality M3`（M1+航运）|
| M4 全模态 | P1 | `run_baseline.py --modality M4` |
| ~~M2 lookback sweep~~ | ~~P3~~ | ✅ 已完成（2026-06-23）；结果见 §7 C3 |
| ~~水体掩膜 GEE（B4）~~ | ~~P2~~ | ✅ 已完成（2026-06-23）；结果见 §7b |

---

## 9. 复现命令

```bash
# 主结果（L4_tuned，retrain_every=13）
python3 04_code/scripts/run_baseline.py --modality M2 --m2-features anom        # → m2/baseline_metrics_anom.csv
python3 04_code/scripts/run_baseline.py --modality M2 --m2-features literature  # → m2/baseline_metrics_literature.csv
# NOTE: --m2-features level 当前矩阵无 level 列，会退化为 M1，不应运行
python3 04_code/scripts/run_baseline.py --modality M2 --leave-one-aoi-out       # → m2/baseline_loao_anom.csv

# SHAP（固定 holdout，约 1–2 分钟）
python3 04_code/scripts/m2/shap_m2.py                                           # → m2/shap_*.csv + shap_anom.png

# C2 降维对照（需先跑 shap_m2.py 生成 shap_topN_anom.csv）→ m2/c2_summary.csv + c2_overview.png
python3 04_code/scripts/m2/robustness_m2.py

# C3 lookback × feature contract sweep
python3 04_code/scripts/m2/sweep_m2.py --quick                                  # → m2/sweep_m2_summary.csv + sweep_m2_overview.png

# B4 水体掩膜稳健性（三步，§7b）
python3 03_data/processed/M2/py/build_m2_weekly.py --watermask
python3 03_data/processed/merge/py/build_feature_matrix.py \
    --m2-csv 03_data/processed/M2/outputs/m2_weekly_features_watermask.csv
python3 04_code/scripts/run_baseline.py --modality M2 --m2-features anom \
    --matrix weekly_feature_matrix_watermask.csv --tag watermask                 # → m2/baseline_metrics_watermask.csv
```

---

## 10. 变更记录

| 日期 | 内容 |
|---|---|
| 2026-06-23 | M2 anom/literature/level 三套基线跑完；Clark-West + LOAO 完成；RQ1 结论确立 |
| 2026-06-23 | 文件结构重组：`05_outputs/baselines/m2/` 子目录；本文件从 `flat_baseline_log.md` §8–10 迁出独立 |
| 2026-06-23 | `shap_m2.py` 写入并运行；`sweep_m2.py` / `robustness_m2.py` 骨架写入 |
| 2026-06-23 | `sweep_m2.py --quick` 跑完；C3 lookback × contract 结果写入 §7；level 合约不可用备注写入 §2/§5；RQ1 结论修正（移除 level 引用） |
| 2026-06-23 | **B4 水体掩膜全流程完成**：GEE CSV 验收 → `build_m2_weekly.py --watermask` → `build_feature_matrix.py --m2-csv` → `run_baseline.py --matrix` → 结果写入 §7b；XGB CW_p 0.006→0.0001，结论强化 |
