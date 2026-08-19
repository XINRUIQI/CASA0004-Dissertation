# Channel B（机制变量通道）工作计划

> **配套主方案**：`00_admin/2026-06-22_research_plan_e2e_multimodal.md` §4.2 双通道设计
> **定位**：把遥感影像转成有经济/物理含义的人工指标（NDVI/NDWI/NDBI/BSI + NTL，共 5 个；FRT 已去掉，见 §6），形成「日期×站点×指标」时间序列，服务于**机制解释、统计验证、可解释性**。
> **最后更新**：2026-07-03

---

## 0. Channel B 在整个框架中的三重角色

1. **机制解释**：用可解释指标回答"卫星看到的活动变化是什么"（植被、水面、建成区、裸地/堆场、夜光活动）。
2. **统计验证（RQ1）**：在 M0/M1 之上，作为 M2 遥感模态做"增量价值"消融——M1 vs M1+ChannelB，DM/Clark–West 检验。
3. **方法对照基线（RQ2）**：Channel B = **扁平指标特征融合**；它正是 Channel A（冻结 EO 大模型 image embedding）要超越的对象。"手工指标 vs 表示学习"这一核心对照，**没有 Channel B 就不成立**。

> 即使消融结果是"遥感无显著增量"（旧 EDA 已有此迹象），只要用 DM/Clark–West 严谨地建立这个结论，本身就是合格贡献。Channel B 的价值不依赖"一定要赢 M1"。

---



## 1. 现有资产盘点


| 资产             | 文件（`data/raw/02_sentinel2/Channel B/`）                                             | 字段                                                                         | 窗口/频率                      |
| -------------- | ------------------------------------------------------------------------------------- | -------------------------------------------------------------------------- | -------------------------- |
| S2 光学指标        | `sentinel2_oil_sites_monthly_indices_201704_202512_11aoi.csv`                         | NDVI/NDWI/NDBI/BSI（各 + `_std`）、`cloud_probability`、`valid_obs_count`、站点元数据 | 2017-04 ~ 2025-12，月，11 AOI |
| VIIRS 夜光       | `viirs_oil_sites_monthly_nightlights_201401_202512_11aoi.csv`                         | `ntl_avg_rad_mean/max/stddev`、`ntl_cf_cvg_mean`                            | 2014-01 ~ 2025-12，月，11 AOI |
| GEE 导出脚本       | `extract_sentinel2_monthly_indices_gee.js`、`extract_viirs_monthly_nightlights_gee.js` | 可复跑/扩展                                                                     | —                          |
| S2 水体掩膜版（B4）   | `sentinel2_oil_sites_monthly_indices_watermask_201704_202512_11aoi.csv`               | 同上 + MNDWI + `land_px`；NDVI/NDBI/BSI 仅陆地像素                                 | 2017-04 ~ 2025-12，月，11 AOI |
| GEE 水体掩膜脚本（B4） | `extract_sentinel2_monthly_indices_watermask_gee.js`                                  | B4 稳健性对照                                                                   | —                          |


**已知数据问题（B0 已量化，B4 水体掩膜已部分缓解）：**

- terminal 类站点（Fujairah/Singapore/Ningbo 等）大量月份 NDVI/NDWI 行为空——云掩膜 + 水面占主导，光学指标在纯水/码头面意义弱。
- S2 指标 2017-04 起、VIIRS 2014 起，与标准比较窗 **2019–2026** 对齐时早期周缺失属正常。

---



## 2. 关键缺口（开工前必须确认）

> 以下为 **2026-06-22 开工前** 的记录，保留作历史；当前均已在 B0–B1 解决。

1. `data/processed/M2/` ~~为空~~ → 已有周频特征表、聚合脚本，并通过 `build_feature_matrix.py` + `code/scripts/flat/run_baseline.py` 接入 M0–M4 消融（原 `01_literature/Test/` 已废弃）。
2. ~~月度原始水平值、无 z-score、无异步对齐~~ → B1 已实现站点 expanding z-score（anom）+ as-of 周对齐 + `days_since_obs`。

---



## 3. 接下来的步骤（B0 → B4）



### B0 — 数据审计与质量报告 ✅ 已完成（2026-06-23）

- 脚本 `data/processed/M2/py/audit_m2_coverage.py`；产物 `outputs/m2_coverage_report.csv` + 4 张热图（缺失 / 覆盖 / 方差 / 云量）。
- **B0 结论（2019–2026，84 个月）：**
  - **覆盖率非问题**：11 站 × 5 指标覆盖率 0.93–1.00；terminal 反而 0.99–1.00（缺失集中在窗外的 2017–2018，双星组网前）。
  - **云量**：terminal（中东干旱）云量最低（Basra 4.0 / Ras Tanura 4.9 / Yanbu 5.7），光学质量好；最多云为 Jamnagar 15.8 / Rotterdam 11.4。
  - **信息量按站点分化，不能按类型一刀切**：高方差如 Basra / Kharg / Ras Tanura 的 NDWI（水面动态）；近常数（低信号）如 Fujairah / Yanbu 的 NDVI、Basra / Ras Tanura 的 NTL。
  - **高方差 ≠ 有用信号**：水面 terminal 的 NDWI 高波动可能是潮汐 / 泥沙 / 水色噪声（当前 GEE 只做云掩膜、未做水体掩膜）→ 去留交 B3 数据驱动（SHAP / leave-one-AOI-out），B0 不硬删。



### B1 — 机制特征构建 ✅ 已完成（2026-06-23）

- 新建 `data/processed/M2/py/build_m2_weekly.py`。
- **5 个指标**：NDVI / NDWI / NDBI / BSI（S2 白天光学指标）+ NTL（VIIRS 夜光）。
- 对每个指标 × 每个 AOI 生成三种形态：
  - `level`：原始月度值（保留，供解释）；
  - `anom`：**站点扩展窗 z-score**（过去-only，min 12 个月）——**5 个指标全部都做**（含 4 个光学指标，理由见下「为什么做站点 z-score」）；
  - `mom`：月环比变化（可选）。

> **为什么对光学指标也做站点 z-score（去站点规模 + 去季节）**
>
> - 「光学指标」= 来自 Sentinel-2 **白天反射影像**的 NDVI/NDWI/NDBI/BSI（NTL 是 VIIRS 夜光，单独类别）。
> - **去站点规模**：站点间绝对值不可比（如 Kharg 的夜光/裸地基底天然就比 Rotterdam 高）；直接喂原始值，模型学到的是「站点之间的固定差异」，而非「某站点相对自己平时更活跃了」。站点 z-score = `(x − 该站点历史均值) / 该站点历史标准差`，把每个站点拉到同一可比尺度（≈ 异常 anomaly）。
> - **去季节**：NDVI（植被）、BSI（裸地）、NDWI（水面）都有强日历季节周期；不处理，模型看到的是「地球的四季」而非「石油活动」。各站点气候带不同，必须**按站点**去季节。
> - **防泄漏**：均值/标准差/季节气候态一律用 **expanding window（只用过去，min 12 月）**，绝不用全期（全期均值含未来 = 前视泄漏）。月频 + 短窗下若按 12 个日历月分组去季节样本太少，折中用「过去 12 个月滚动均值」去趋势，或把 month-of-year 作为特征交给模型。

> **方法论与文献依据（站点级标准化距平 / within-site standardized anomaly）**——三个标准组件叠加，写作时可直接引用：
>
> 1. **z-score 标准化**：通用 ML/统计预处理（standardization）。
> 2. **站点级去均值（去规模）** = 面板数据 within transformation / 固定效应思想（Wooldridge 面板计量教材）；夜光经济学普遍要求站点内标准化 / 相对自身基线，而非比绝对值（Henderson, Storeygard & Weil 2012 *AER*；Donaldson & Storeygard 2016 *JEP*；Gibson et al. 2021，已见 `external_sources.md`）。
> 3. **去季节（标准化距平 standardized anomaly）** = 植被/气候遥感标准做法，同源指数 VCI（Kogan 1995）、SPI（McKee et al. 1993）；另一主流去季节法 STL 分解（Cleveland et al. 1990）。
>
> - **本项目增强点**：以上统计量一律用 **expanding window（仅用过去）** 而非文献常用的全样本，满足样本外预测无泄漏要求，可作为方法严谨性写入 Methodology。
> - ⚠️ 引用前在 Google Scholar 复核年份 / 期刊 / 页码。

- **元数据列（随观测携带）**：`observation_date`、`days_since_obs (age)`、`valid_obs_count`（观测质量 / 可信度，保留）、`sensor`、`valid_mask`（该周是否有有效观测）、`modality_mask`（M2 模态是否可用）。
- `**cloud_probability` 不作为入模特征**：它是数据质量指标而非活动信号，仅用于 B0 审计与无效观测过滤（避免把"云多 / 云少"误学成价格信号）；以 `valid_obs_count` 作为保留的可信度标记。
- **异步对齐到 W-FRI**（方案 §4.4，防泄漏关键）：
  - 按真实可得日 + 保守发布滞后对齐；
  - 每周记录 `值 + days_since_obs + valid_mask + modality_mask`；**不把月值 ffill 成多个相同周值**，让缺失 / 陈旧显式可见；
  - 任何滚动统计只用历史，scaler 仅在训练集 fit。
- 输出：
  - `data/processed/M2/outputs/m2_weekly_features.csv`（宽表：`{index}_{aoi}` + `{index}_anom_{aoi}` + `avail_`*，对齐 2019–2026）；
  - `m2_eda_weekly.csv`（tidy 长表：date, site_id, site_type, index, level, anom, mom, age, cloud…，供 EDA/面板分析）。
  - B4 水体掩膜版：`m2_weekly_features_watermask.csv`（365×188）+ `m2_eda_weekly_watermask.csv`（`--watermask`）。
- 同步在 `data/Dataset/Dataset_Overview4.ipynb` 的 M2 词典里补 Channel B 逐变量说明；完整列字典见 `data/processed/M2/m2_ChannelB_data_dictionary.md`。
- **验收**：`m2_weekly_features.csv` 经 `data/processed/merge/py/build_feature_matrix.py` 并入 `weekly_feature_matrix.csv`，由 `code/scripts/flat/run_baseline.py --modality M2` 读入。
- **产出（2026-06-23）**：脚本 `build_m2_weekly.py`；`outputs/m2_weekly_features.csv`（365 周 × 154 特征：5 指标 ×11 站 ×(level+anom) + 2 模态 ×11 站 ×(age+avail)）+ `m2_eda_weekly.csv`（20075 行 tidy）。S2 可用率 0.998 / NTL 1.000，anom 缺失 3%。防泄漏 sanity 通过（level 月内重复、`days_since_obs` 每周 +7、发布滞后下新观测才跳变）。



### B2 — 机制验证 EDA ✅ 已完成（2026-06-23）

- 新建 `data/processed/M2/py/eda_m2_mechanism.py`。
- 内容：
  - 每指标各站点时序图，叠加已知事件（2020 COVID、2022 俄乌、2023–24 红海/Houthi、Hormuz 紧张、炼厂检修）；
  - 与 Brent 周对数收益、与 **M1 基本面**的**滞后相关 / 交叉相关 / Granger**（严格滞后，无前视）——例如：refinery 站点的 NDBI/NTL 是否领先 `refinery_utilisation`；export terminal 的 NTL 是否与 `crude_exports` 同步。
  - 按**站点类型**（port/refinery/terminal）聚合成可解释复合活动指数。
- 输出：时序/相关/lead-lag 图 → `data/processed/M2/outputs/m2_eda_*.png`
- **验收**：能给出 2–3 条"机制故事"（如"X 站点夜光异常领先出口 N 周"），写进 Methodology/Results。
- **产出（2026-06-23）**：脚本 `eda_m2_mechanism.py`；`outputs/m2_eda_*.png`（站点类型复合 anom 时序 / lead-lag 热图 / 与 M1 同期相关 / Houston 专项）+ `m2_eda_leadlag_corr.csv`。
- **B2 结论**：① 整体相关弱（|corr| < 0.15，符合周油价难预测 + NTL↔油价弱代理）；② **多数较强相关落在负 lag**（Channel B 滞后/同期反应油价，而非领先）→ 对 RQ1 增量预测是**审慎信号**；③ 亮点 **NDWI_terminal**（中东出口码头水面动态）lag+1 Granger p=0.029、lag+3 领先相关 ≈0.14（多重比较下需谨慎）；④ 含义：手工指标领先信号有限，**反向支撑"需要 Channel A 表示学习 + 门控融合"** 的论证；**B3 已确认 XGB 嵌套增量显著（CW_p=0.006），但 vs M0 仍无 skill**。



### B3 — 增量价值 + 可解释性消融 ✅ 已完成（2026-06-23；M1 基线 2026-07-03 重跑）

- 建模层：`code/src/backtest/` + `code/scripts/flat/run_baseline.py`（替代旧 `01_literature/Test/`）。
- 跑 **M0 / M1 / M1+M2** 表格模型（Ridge + XGB；Channel B 为 tabular，深度模型留给主框架）。
- 检验：**Diebold–Mariano / Clark–West**（嵌套模型 vs M1 用 Clark–West）。
- 解释：**SHAP**（`code/scripts/flat/M2_Flat/shap_m2.py`）——按 RS 指数 / AOI 聚合。
- **leave-one-AOI-out**（`run_baseline.py --leave-one-aoi-out`）；~~按站点类型~~ 分组消融 **未单独实现**（B2 仅有类型复合 EDA）。
- **降维对照 C2**（`code/scripts/flat/M2_Flat/robustness_m2.py`）：All-55 / PCA-90% / ElasticNet / SHAP-top20。
- **验收**：✅ "M1 vs M1+ChannelB" 对比表 + SHAP 图 + LOAO；已回答 RQ1。



#### B3 产出与结论（2026-06-23 / 07-03）


| 项                         | 脚本 / 产物                                                                                                  |
| ------------------------- | -------------------------------------------------------------------------------------------------------- |
| 主基线 M0/M1/M2              | `run_baseline.py --modality M2 --m2-features anom` → `results/baselines/Flat/M2_Flat/baseline_metrics_anom.csv` |
| literature 子集（4 NTL_anom） | `--m2-features literature` → `baseline_metrics_literature.csv`                                           |
| level 对照（B4）              | `--m2-features level` → `baseline_metrics_level.csv`                                                     |
| LOAO                      | `--leave-one-aoi-out` → `baseline_loao_anom.csv`                                                         |
| SHAP                      | `shap_m2.py` → `shap_anom.png` + `shap_*_by_{feature,index,aoi}.csv`                                     |
| C2 降维                     | `robustness_m2.py` → `c2_summary.csv` / `c2_overview.png`                                                |
| 完整记录                      | `00_admin/待整理/flat_baseline_log.md` §8                                                                   |


**B3 结论（主协议 L4，lookback=4，2019–2025）：**

- **vs M0**：无模型 beat 随机游走（M2 Ridge skill −6.3%，DM 不显著）——周频 Brent 难预测，与 B2 一致。
- **vs M1（RQ1 嵌套增量）**：M2 **XGB** Clark–West **p = 0.006**（显著）；M2 Ridge CW_p = 0.21（不显著）→ 增量**模型依赖**（非线性树能挖 RS 信号，线性不能）。
- **LOAO**：逐站剔除多数略降 RMSE → 信号**分散**于多站，非单站驱动。
- **C2**：PCA-90 / SHAP-top20 下 XGB CW 仍显著（p ≈ 2.6e-6 / 7.1e-4）→ 增量非纯过拟合 55 维。
- **literature 4 NTL**：XGB CW 亦显著（L4 p ≈ 2.0e-5）→ 文献精选子集与全量 55 anom 结论一致。



#### M2 入模列决策（154 列 → 用哪些）

> 实测窗口 **2019-01 ~ 2025-12（365 周，M2 整段落入，无需裁剪）**。154 列 = `level` 55 + `anom` 55 + `age` 22 + `avail` 22。


| 类别                      | 列数  | 2019–2025 实测                 | 决策                                             |
| ----------------------- | --- | ---------------------------- | ---------------------------------------------- |
| `{idx}_anom_{aoi}`      | 55  | std≈1.20，无常数列                | ✅ **入模主力**                                     |
| `{idx}_{aoi}` level     | 55  | std≈4.21，尺度不可比+含季节+与 anom 冗余 | ❌ 主分析不用；仅 B4「level vs anom」robustness          |
| `s2/ntl_age_days_{aoi}` | 22  | 28–271 天，有变化但属时效非信号          | 🔶 主分析不用；可选聚合版作 robustness                     |
| `s2/ntl_avail_{aoi}`    | 22  | 8024 值中仅 8 个=0，**近零方差**      | ❌ 剔除（窗内恒≈1；其价值在端到端 missing-modality，非 tabular） |


**三层用法（兼顾 RQ1 + 可解释性 + P058 降维对照）：**

- **主分析（答 RQ1 遥感整体增量）**：全部 **55 anom**，但**必须配降维/正则**并列对照 `All-55 / +PCA / +ElasticNet / +SHAP-selected`（P058：SHAP 不解决共线性，PCA 与 SHAP 解决不同问题）。55 anom 对 365 周 ≈1:6.6 偏高维 + 站点/指标间强共线，不可直接全丢。
- **可解释层**：突出文献核心 **NTL_anom**（Fujairah / Ras Tanura / Rotterdam / Houston，必要时加 Yanbu/Kharg 出口枢纽）+ SHAP；leave-one-AOI-out 用全 55 anom（需站点级保留）。
- **指标文献先验**（给 SHAP/筛选排序）：①NTL_anom（P024/P032 最支持）→ ②NDBI_anom/BSI_anom（建成区/裸地堆场，工业相关）→ ③NDWI_anom（水面，B0 提示噪声）、NDVI_anom（植被，P069 明确降级）。

**文献对照（矩阵 §②，P024/P025/P032/P069/P055）：**

- 「用站点内异常 z-score、不用原始水平」「必须 VIIRS」——与 `anom` 一致 ✅。
- 文献只推 **4 个 NTL 站点 + 1 质量变量 ≈ 5–7 列**；NDVI 降级、删非石油 Ningbo。B1 全量 55 anom = 候选池，B3 据此走「文献精选 vs 全量+降维」**两条都报**：全量答 RQ1，精选+SHAP+leave-one-AOI-out 答可解释性。
- 「保留 `valid_obs_count`」是数据质量/审计用途，**非预测特征**——与「质量/时效列不入模」一致。



### B4 — 稳健性 + 写作 + 为 RQ2 铺路 🔶 大部分完成（2026-06-23）

- 稳健性：
  - ✅ lookback 1/4/8 周 × anom/literature/level → `code/scripts/flat/M2_Flat/sweep_m2.py` → `sweep_m2_summary.csv`
  - ✅ level vs anom（见 sweep `level` contract + 主基线 `--m2-features level`）
  - ✅ leave-one-AOI-out（B3 已做）
  - ✅ **水体掩膜版** GEE → `build_m2_weekly.py --watermask` → `build_feature_matrix.py --m2-csv ...watermask.csv` → `run_baseline.py --matrix weekly_feature_matrix_watermask.csv --tag watermask`；M2 XGB CW_p **0.006 → 8.5e-5**（增量更强，结论保守）
  - ❌ **common-sample vs max-sample** 显式对照未做
- 文档：
  - ✅ `m2_ChannelB_data_dictionary.md`（含 §7 水体掩膜）
  - ✅ `06_writing/chapter_3_methodology.md`（机制变量、z-score、as-of、水体掩膜）
  - ✅ `06_writing/chapter_4_results.md`（M2 增量表 + 稳健性段落）
- **为 RQ2 铺路**：
  - ✅ Channel B 已固定为 M2 扁平指标基线（全套 baseline 可复现）
  - 🔶 Channel A（Prithvi embedding）数据工程进行中：`precompute_s2_embeddings.py`、`s2_patch_index.csv` 等（2026-07-03）
  - ❌ **「手工指标 vs 表示学习」正式对照实验** 待 Channel A 入模后启动

---



## 3.5 已实现脚本说明（B0–B4）

> 数据工程脚本路径 `data/processed/M2/py/`，数据产物 `data/processed/M2/outputs/`；建模/消融脚本 `code/scripts/`，结果 `results/baselines/Flat/M2_Flat/`。



### `audit_m2_coverage.py`（B0 — 数据审计）

**干什么：** 在正式建模前，对 Channel B 两张**原始月度 CSV**做质量体检，回答「2019–2026 比较窗内，11 站 × 5 指标能不能用、缺多少、有没有信号」。


| 项      | 内容                                                                                                                                                                                                            |
| ------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **输入** | `raw/02_sentinel2/Channel B/sentinel2_oil_sites_monthly_indices_*.csv`（S2：NDVI/NDWI/NDBI/BSI + 云量/有效观测数） `viirs_oil_sites_monthly_nightlights_*.csv`（VIIRS：NTL）                                               |
| **处理** | 1. 按比较窗（默认 2019–2026）裁剪 2. 对每个 (站点, 指标) 算 **coverage** = 非空月数 / 总月数 3. 算窗口内 **temporal std** = 时间方差，作为「信息量」代理（近常数列即使不缺失也几乎无信号） 4. 汇总 S2 的 `cloud_probability`、`valid_obs_count` 作数据质量上下文 5. 画 4 张热图 / 柱状图     |
| **输出** | `m2_coverage_report.csv`（逐站点×指标 tidy 报告） `m2_s2_missing_heatmap.png`（站点×月 S2 有效/缺失） `m2_coverage_heatmap.png`（站点×指标覆盖率） `m2_variability_heatmap.png`（站点×指标 std，按指标归一） `m2_cloud_validobs.png`（各站平均云量 & 有效观测数） |
| **运行** | `python3 data/processed/M2/py/audit_m2_coverage.py` 可选 `--start 2017-04 --end 2025-12` 改窗口                                                                                                                 |


**关键结论（已写入 §3 B0）：** 2019–2026 内覆盖率 0.93–1.00，terminal 并不缺数据；问题在「站点间/指标间信息量分化」，不能按站点类型一刀切剔除。

---



### `build_m2_weekly.py`（B1 — 周频特征构建）

**干什么：** 把两张**原始月度表**转成**无泄漏、可入模**的周频（W-FRI）特征表——Channel B 进入 M2 消融前的核心数据工程步骤。


| 项      | 内容                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             |
| ------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **输入** | 同上两张 Channel B 月度 CSV（用全历史算 anomaly，再对齐到比较窗）                                                                                                                                                                                                                                                                                                                                                                                                                                                   |
| **处理** | 1. **统一长表**：11 站 × 5 指标（NDVI/NDWI/NDBI/BSI/NTL） 2. **三形态**（每站×每指标）： · `level` — 原始月度值（解释用） · `anom` — 站点扩展窗标准化距平：先去季节（expanding 月气候态）再 expanding z-score（min 12 月、**仅用过去**） · `mom` — 月环比一阶差分 3. **月→周异步对齐**：as-of join，可得日 = 月末 + 15 天发布滞后；**不是哑 ffill**——值可在月内重复，但每周携带递增的 `days_since_obs`，新观测发布时才跳变 4. **元数据 mask**：`valid_mask`（有无有效观测）、`modality_mask`（age ≤ 100 天视为模态可用）；`cloud_probability` **不入模**，保留 `valid_obs_count` 作可信度 5. **宽表 pivot**：level + anom 列 + S2/VIIRS 各自的 age/avail 列 |
| **输出** | `m2_weekly_features.csv` — 宽表，365 周 × 154 特征（5×11×(level+anom) + 2×11×(age+avail)） `m2_eda_weekly.csv` — 长表，20075 行（365×11×5），含 level/anom/mom + 全部元数据，供 EDA                                                                                                                                                                                                                                                                                                                              |
| **运行** | `python3 data/processed/M2/py/build_m2_weekly.py` 可选 `--no-deseasonalize`（只做 z-score、不去季节） 可选 `--watermask`（B4 水体掩膜版 → `*_watermask.csv`） 可选 `--start / --end` 改窗口                                                                                                                                                                                                                                                                                                                          |


**设计要点：** `anom` 是入模主力；防泄漏 sanity 已验证（2019-01 初周看到的是 2018-11 观测，而非尚未发布的 12 月值）。

---



### `eda_m2_mechanism.py`（B2 — 机制验证 EDA）

**干什么：** 用 B1 长表 + M1 周频表，做**可解释性**探索——Channel B 指标有没有经济/物理含义、与油价和基本面的时序关系是领先还是滞后（**不替代 B3 的样本外预测检验**）。


| 项      | 内容                                                                                                                                                                                                                                                                                                                                                                                                                |
| ------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **输入** | `outputs/m2_eda_weekly.csv`（B1 长表） `processed/M1/outputs/m1_weekly_features.csv`（Brent 收益、美国基本面等）                                                                                                                                                                                                                                                                                                            |
| **处理** | 1. **站点类型复合指数**：按 port / refinery / terminal 对每指标的 `anom` 取均值，画 2019–2026 时序并叠加事件线（COVID、俄乌、红海等） 2. **Lead-lag 交叉相关**：15 个复合特征 vs `brent_log_return`，lag −8…+8 周（lag>0 = Channel B **领先**市场） 3. **同期相关热图**：各站 NTL/NDBI anomaly vs M1 关键变量（Brent、炼厂利用率、进出口等） 4. **Houston 专项**：唯一美国 AOI，对比 NTL/NDBI anomaly 与美国 `refinery_utilisation` / `crude_exports` 5. **Granger 因果**（可选）：对 lead-lag 最强的 4 个复合特征做 maxlag=4 检验 |
| **输出** | `m2_eda_anom_by_type.png` — 五指标×三站点类型复合 anomaly 时序 `m2_eda_leadlag_brent.png` — lead-lag 相关热图 `m2_eda_corr_m1.png` — 各站 NTL/NDBI vs M1 同期相关 `m2_eda_houston_us.png` — Houston vs 美国基本面 `m2_eda_leadlag_corr.csv` — lead-lag 数值表                                                                                                                                                                                   |
| **运行** | `python3 data/processed/M2/py/eda_m2_mechanism.py`                                                                                                                                                                                                                                                                                                                                                             |


**解读注意：** M1 基本面是**美国**（EIA），Channel B 站点是**全球**；干净机制链是「全球站点活动 → Brent 价格」，只有 Houston 与美国基本面有地理对应。B2 结论（弱相关、多数负 lag）已写入 §3 B2 与 `chapter_4_results.md`；**B3 样本外增量**见 §3 B3（XGB CW_p=0.006）。

---



### `run_baseline.py` + `code/src/backtest/`（B3 — M0/M1/M2 消融）

**干什么：** 统一 rolling-origin 引擎；M2 = M1 + Channel B 55 anom（或 literature/level 合约）。


| 项      | 内容                                                                                                                            |
| ------ | ----------------------------------------------------------------------------------------------------------------------------- |
| **输入** | `processed/merge/outputs/weekly_feature_matrix.csv`（55 anom 已 merge）                                                          |
| **运行** | `python3 code/scripts/flat/run_baseline.py --modality M2 --m2-features anom`                                                    |
|        | `python3 code/scripts/flat/run_baseline.py --modality M2 --leave-one-aoi-out`                                                   |
|        | `python3 code/scripts/flat/run_baseline.py --modality M2 --matrix weekly_feature_matrix_watermask.csv --tag watermask`          |
| **输出** | `results/baselines/Flat/M2_Flat/baseline_metrics_*.csv` / `baseline_predictions_*.csv` / `backtest_*.png` / `baseline_loao_anom.csv` |


---



### `shap_m2.py`（B3 — 可解释性）


| 项      | 内容                                                                          |
| ------ | --------------------------------------------------------------------------- |
| **运行** | `python3 code/scripts/flat/M2_Flat/shap_m2.py`                                     |
| **输出** | `shap_anom.png`；`shap_xgb_m2_by_{index,aoi}.csv`；`shap_topN_anom.csv`（供 C2） |


---



### `robustness_m2.py`（B3/B4 — C2 降维对照）


| 项      | 内容                                                                              |
| ------ | ------------------------------------------------------------------------------- |
| **运行** | `python3 code/scripts/flat/M2_Flat/robustness_m2.py`                                   |
| **输出** | `c2_summary.csv` / `c2_overview.png`（All-55 / PCA-90 / ElasticNet / SHAP-top20） |


---



### `sweep_m2.py`（B4 — lookback × feature contract）


| 项      | 内容                                                                                     |
| ------ | -------------------------------------------------------------------------------------- |
| **运行** | `python3 code/scripts/flat/M2_Flat/sweep_m2.py`                                               |
| **输出** | `sweep_m2_summary.csv` / `sweep_m2_overview.png`（anom / literature / level × L1/L4/L8） |


---



## 4. 目标产物（time-series 表格示例）

宽表 `m2_weekly_features.csv`（节选）：


| week_fri   | NTL_anom_Houston | NDBI_anom_Houston | BSI_anom_RasTanura | …   | avail_m2_Houston |
| ---------- | ---------------- | ----------------- | ------------------ | --- | ---------------- |
| 2024-01-05 | 0.42             | 0.11              | −0.30              | …   | 1                |


长表 `m2_eda_weekly.csv`（节选，供面板/EDA）：


| week_fri   | site_id | site_type | index | level | anom | age_days | cloud_frac |
| ---------- | ------- | --------- | ----- | ----- | ---- | -------- | ---------- |
| 2024-01-05 | P005    | port      | NTL   | 67.2  | 0.42 | 9        | 0.08       |


---



## 5. 防泄漏自检（每步必查，呼应方案 §6.4）

- [x] 月度指标按**真实可得日 + 保守滞后**对齐，绝不用当周尚未发布的值。（B1 as-of + 15d 滞后；`build_m2_weekly.py` sanity 通过）
- [x] 站点 z-score / 滚动统计**仅用过去**（expanding，min 12 月）。（B1 `anom` 列）
- [x] **不**把月值 ffill 成多个相同周值；改为 `值 + age + mask`。（B1 设计 + merge 文档）
- [x] 标准化/特征选择只在训练集内 fit。（`code/src/backtest/rolling.py` 每 fold 内 fit）
- [x] 不用中心化移动平均（避免 P018 式前视）。

---



## 6. 决策记录

**已决策（2026-06-23）：**

1. **去掉 FRT**：Channel B 固定为 5 个指标 NDVI/NDWI/NDBI/BSI + NTL。理由：旧「浮顶充填率」S2 10m 无法复现（P055，已否决）；新「火点/热异常」需另采 VIIRS Nightfire/S3 SLSTR，本阶段不扩范围。如日后想补，作为 Future Work 单列。
2. **光学指标也做站点 z-score**：4 个 S2 指标与 NTL 一律做站点扩展窗 z-score（去规模 + 去季节，过去-only），见 §3 B1 说明。
3. **比较窗锁定 2019–2026**：与主方案 M0–M4 标准化窗口一致。
4. `cloud_probability` **不入模**：仅作 B0 审计 / 无效观测过滤；保留 `valid_obs_count` 作可信度标记。
5. **不按站点类型先验剔除**（基于 B0）：terminal 的 NDVI/NDWI 在 2019–2026 覆盖 0.99–1.00、并不缺失，故全部纳入 B1（5 指标 × 11 站）；站点 z-score 后近常数列（如 Fujairah/Yanbu NDVI、Basra/Ras Tanura NTL）自然弱化，最终去留由 B3 的 SHAP / leave-one-AOI-out 数据驱动决定。
6. **M2 入模列 = 55 个** `anom`（详见 §3 B3「M2 入模列决策」）：剔除 55 level（仅 robustness）+ 22 avail（窗内近零方差）；age 主分析不用。主分析全 55 anom + PCA/正则/SHAP 降维对照（答 RQ1），可解释层用文献核心 NTL_anom 子集 + SHAP + leave-one-AOI-out。

**待确认 / 后续（2026-07-03）：**

1. **RQ2 正式对照**：Channel A embedding 入模后，跑「Channel B（M2 anom-55）vs Channel A（image embedding）」同协议消融。
2. **common-sample vs max-sample** 显式稳健性（B4 遗留）。
3. **按站点类型**（port/refinery/terminal）分组消融（B3 计划项；当前仅有 B2 类型复合 EDA + 逐 AOI LOAO）。
4. 更新 `Dataset_Overview4.ipynb` 中长表文件名为 `m2_eda_weekly.csv`（若仍引用旧名）。

---



## 7. 一周排期建议（执行记录）


| 天     | 任务                                       | 状态                               |
| ----- | ---------------------------------------- | -------------------------------- |
| D1    | B0 审计（FRT 已决：去掉）                         | ✅ 2026-06-23                     |
| D2–D3 | B1 `build_m2_weekly.py` + 数据字典           | ✅ 2026-06-23（字典 07-03 扩 §7 水体掩膜） |
| D4    | B2 机制 EDA                                | ✅ 2026-06-23                     |
| D5–D6 | B3 消融 + DM/CW + SHAP + leave-one-AOI-out | ✅ 2026-06-23（M1 基线 07-03 重跑）     |
| D7    | B4 稳健性 + 写作 + RQ2 对照接口                   | 🔶 稳健性+写作 ✅；RQ2 对照待 Channel A    |


