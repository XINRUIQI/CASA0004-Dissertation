# Research Diary

---

## 2026-05-06

### What I did

- First meeting with supervisor Beatrice Taylor.
与导师 Beatrice Taylor 进行了第一次会议。
- Discussed data access challenges.
讨论了数据获取方面的困难。
- Beatrice suggested looking into ShipRSImageNet.
Beatrice 建议查看 ShipRSImageNet 数据集。

### Next tasks

- Check ShipRSImageNet categories and timestamps.
检查 ShipRSImageNet 的类别和时间戳。
- **Build dataset inventory table.**  
建立数据集清单表。
- Start literature review on oil price forecasting and remote sensing indicators.
开始阅读原油价格预测和遥感指标相关文献。

---

## 2026-05-22

### What I did

- Reorganised dissertation project folder structure.  
重新整理了论文项目文件夹结构。
- Assigned priority levels to the references.
给参考文献标好了等级。**
- **Completed the first two categories of the dataset table: 1) Market & Financial Data and 2) Official Reports & Structured Market Text.
整理完 dataset table 的前两类：1) Market & Financial Data 和 2) Official Reports & Structured Market Text。

### Decisions made

- **In 1) Market & Financial Data, EIA Short-Term Energy Outlook (STEO) is monthly, so it will not be used for now.**  
在 1) Market & Financial Data 里，EIA Short-Term Energy Outlook (STEO) 的时间尺度为月度，因此暂时不用。
- In 1) Market & Financial Data, ICE and CME require paid subscriptions, so they were removed.**
在 1) Market & Financial Data 里，ICE 和 CME 需要付费订阅，因此删掉。
- **For 1D. Macro-financial control variables, it is still uncertain whether USD-related variables are needed.**  
对于 1D. Macro-financial control variables，目前不确定是否需要美元相关变量，待定。

---

## 2026-05-25

### What I did

**Data acquisition (bulk download day)  数据采集（集中下载日）**

- Downloaded 241 EIA STEO monthly PDF reports (2006–2025) via `download_steo_archive.py`.  
通过脚本批量下载了 241 份 EIA STEO 月度 PDF 报告（2006–2025）。
- Queried GDELT via BigQuery: obtained oil-disruption daily features and transport-disruption event records (2019-01 – 2026-05, 83 CSV shards).
通过 BigQuery 查询 GDELT：获取了石油扰动事件每日特征和运输中断事件记录（2019-01 至 2026-05，83 个 CSV 分片）。
- Scraped Aramco press release links and full-text content using Playwright.  
使用 Playwright 爬取了 Aramco 新闻发布链接及正文内容。——没爬到几个
- Downloaded OGIM (Global Oil & Gas Infrastructure Mapper) core + extended node data and Global Oil & Gas Extraction Tracker.  
下载了 OGIM 核心/扩展基础设施节点数据及 Global Oil & Gas Extraction Tracker。
- Downloaded NGA World Port Index (2026) and EMODnet vessel density monthly data (2017–2025).  
下载了 NGA 世界港口索引（2026）及 EMODnet 月度船舶密度数据（2017–2025）。
- Downloaded S&P 500 daily data from Yahoo Finance as macro-financial control variable.  
从 Yahoo Finance 下载了 S&P 500 日频数据作为宏观金融控制变量。

**Remote sensing pipeline  遥感数据流水线**

- Wrote 3 Google Earth Engine scripts: Sentinel-2 monthly indices, Landsat backfill monthly indices, and VIIRS monthly nightlights — all for 8 oil-infrastructure AOIs.  
编写了 3 个 GEE 脚本：Sentinel-2 月度指数、Landsat 回填月度指数、VIIRS 月度夜间灯光——覆盖 8 个石油基础设施 AOI。
- Exported and downloaded CSV outputs; built `aggregate_remote_sensing_to_weekly.py` to resample to weekly frequency.  
导出并下载了 CSV；编写 `aggregate_remote_sensing_to_weekly.py` 重采样至周频。

**Shipping data pipeline  航运数据流水线**

- Built `download_portwatch_chokepoints.py`: fetches IMF PortWatch daily transit data for 6 oil-critical chokepoints (Hormuz, Suez, Malacca, Panama, Bab el-Mandeb, Cape of Good Hope) from 2019 onward.  
编写 `download_portwatch_chokepoints.py`：从 IMF PortWatch 获取 6 条关键石油咽喉航道日频过境数据（2019 年起）。
- Built `download_gfw_vessel_presence.py`: fetches GFW 4Wings API monthly vessel presence hours for the same 6 chokepoints (2012–2018), filling the pre-PortWatch gap.  
编写 `download_gfw_vessel_presence.py`：通过 GFW 4Wings API 获取 2012–2018 年同 6 条航道的月度船舶在场时数，填补 PortWatch 之前的空白。
- Built `aggregate_shipping_to_weekly.py`: merges PortWatch daily + GFW monthly into a single wide-format weekly shipping feature table.  
编写 `aggregate_shipping_to_weekly.py`：将 PortWatch 日频 + GFW 月频合并为统一的宽格式周频航运特征表。

**Feature matrix & analysis  特征矩阵与分析**

- Built `build_feature_matrix.py` to combine all processed weekly features (market, macro, text/GDELT, remote sensing, shipping) into a unified feature matrix.  
编写 `build_feature_matrix.py`，将所有已处理的周频特征（市场、宏观、文本/GDELT、遥感、航运）合并为统一特征矩阵。

### Decisions made

- **Use GFW vessel presence (2012–2018) + PortWatch (2019–2026) as a combined shipping proxy to cover the full study period.**  
用 GFW 船舶在场数据（2012–2018）+ PortWatch（2019–2026）组合覆盖整个研究期间的航运代理变量。
- **Defined 8 oil-infrastructure AOIs for remote sensing extraction** (to be documented in `aoi_oil_infrastructure.csv`).  
为遥感提取定义了 8 个石油基础设施 AOI（记录在 `aoi_oil_infrastructure.csv` 中）。
- **Weekly (Friday-ending) as the unified temporal resolution** for all feature modalities.  
以每周五截止的周频作为所有特征模态的统一时间分辨率。

### Next tasks

- Prepare progress update and questions for Meeting 02 (27 May).  
为第二次导师会议（5 月 27 日）准备进度更新和问题。
- Review quality of GDELT features — check whether disruption event counts are noisy.
审查 GDELT 特征质量——检查扰动事件计数是否噪声过大。
- Verify remote sensing CSV outputs (check for missing months / AOIs).  
验证遥感 CSV 输出（检查是否有缺失月份/AOI）。

---

## 2026-06-05

### What I did

- Second meeting with supervisor Beatrice Taylor.
与导师 Beatrice Taylor 进行了第二次会议。
- Presented dataset inventory table and variable overview.
展示了数据集清单表和变量概览。

### Supervisor feedback

- Variable table and dataset table are clear and detailed — good progress.
变量表和数据集表清晰详细——进度良好。
- Need to **narrow scope**: too many variables risk collinearity and overfitting; data volume too large for timeline.
需要**缩小范围**：变量太多有共线性和过拟合风险；数据量对于时间线来说太大。
- Proposed **progressive 4-model framework**:
提出**渐进式4模型框架**：
  - Model 1: Financial data only (5–10 variables) 仅金融数据
  - Model 2: Model 1 + remote sensing 加遥感数据
  - Model 3: Model 1 + shipping data 加航运数据
  - Model 4: All modalities combined 所有模态合并
- Use **literature** to justify variable and model selection (e.g. XGBoost).
用**文献**来论证变量和模型选择。
- Consider **PCA** for dimensionality reduction.
考虑用**PCA**降维。

### Decisions made

- Adopt 4-model progressive ablation framework.
采用4模型渐进式消融框架。
- Phase 02 officially starts: literature review + variable/model selection.
Phase 02 正式开始：文献综述 + 变量/模型选择。

### Next tasks

- Structured literature review: for each paper, extract datasets, variables, model, results.
结构化文献综述：对每篇论文，提取数据集、变量、模型、结果。
- Select 5–10 financial variables for Model 1 baseline.
为 Model 1 基线选择 5–10 个金融变量。
- Define remote sensing and shipping variable subsets for Models 2 and 3.
为 Model 2 和 3 确定遥感和航运变量子集。
- Decide modelling approach based on literature.
根据文献决定建模方法。
- Explore PCA for dimensionality reduction.
探索 PCA 降维。

---

## 2026-06-16

### What I did

**Phase 02 文献精读（M1 金融变量组，7/7 篇完成）**

- 深度精读 P052 (Kilian 2009)、P053 (Alquist & Kilian 2013)、P054 (Baumeister & Kilian 2015)、P004 (LSTM+XGBoost 2024)、P001 (Foroutan & Lahmiri 2024)、P072 (Costa et al. 2021)、P076 (Yılmaz & Zehir 2026)，每篇产出独立 reading notes（`01_literature/reading_notes/01/P0xx_notes.md`）。
- 更新 `beatrice_task_literature_matrix.md`：对 P054、P004、P001 做了**重大修正**——P054 组合的是 6 个模型而非 6 个变量；P004 只用 4 个宏观变量且 R²=0.999 疑似数据泄漏；P001 是纯价格 M0 基线而非 M1 依据。
- 基于文献确定 M1 变量子集（9 个）：`brent_price`, `brent_log_return`, `crude_stocks_change`, `crude_production`, `refinery_utilisation`, `vix`, `dollar_index`, `sp500_return_pct`, `treasury_10y`。
- 确定 M2 遥感子集（5 个 NTL + 1 个 FRT）、M3 航运子集（GFW 7 + PortWatch 7 + EMODnet 4），并新增 M5 补充层（GDELT 13 个事件特征，供 Discussion/Appendix 使用）。

**EDA 探索性分析**

- 编写 `01_literature/EDA/eda_beatrice_variables.py`，对 M1–M4 变量子集生成 10 张图 + 2 张 CSV：
  - 摘要统计、缺失热图、M1/M2/M3 时序图、各层相关性矩阵、与目标变量相关性、分布图、滚动相关、Top 特征散点图。
- 主要发现：RS/航运变量缺失集中在 2012 年前（GFW）和 2017 年前（EMODnet）；`flat` 类样本极少（约 5%），三分类极度不平衡。

**CV 增强特征工程（Phase 0.5 部分完成）**

| 脚本 | 路径 | 功能 | 输出 |
|---|---|---|---|
| `extract_emodnet_density.py` | `04_code/scripts/` | 从 EMODnet 月度 GeoTIFF 提取 Rotterdam + Suez 船舶密度统计 | `emodnet_density_monthly.csv`（4 列） |
| `frt_fill_level_estimation.py` | `04_code/scripts/` | Sentinel-2 影像 → Hough 圆检测 → LAB 阴影比 → 液位估算 | `frt_fill_level_monthly.csv` + 24 张检测可视化 PNG |
| `merge_cv_features.py` | `04_code/scripts/` | 月频 ffill → W-FRI 对齐，合并进 `weekly_features.csv` | 更新后的统一特征矩阵 |

- FRT 结果：Fujairah 2023–2024 共 24 个月，约一半月份检测到 1–3 个罐（`n_tanks`），液位 0.03–0.84；检测不稳定，部分月份 `n_tanks=0`。
- EMODnet 覆盖 2017–2025，与 GFW (2012+) 和 PortWatch (2019+) 形成互补时间窗口。

**完整建模 pipeline 搭建与首次全量运行**

在 `01_literature/Test/` 下搭建并运行了 project_plan 中的 Tier 1–3 全部模型：

| 模块 | 文件 | 说明 |
|---|---|---|
| 配置 | `config.py` | M1–M5 变量定义、lag/MA 规格、超参数、AOI 图拓扑 |
| 数据 | `data_loader.py` | 表格/序列加载、70/15/15 时序划分、lag/MA 特征工程 |
| 评估 | `evaluation.py` | Accuracy/F1/DA、RMSE/MAE/R²、Diebold–Mariano test |
| Tier 1 | `01_baselines.py`, `02_xgboost_model.py` | LR/Ridge/RF/SVM + XGBoost × M1–M5 × 3 targets |
| Tier 2 | `03_lstm_model.py`, `04_tft_model.py` | 2-layer LSTM + 简化版 TFT（纯 PyTorch） |
| Tier 3 | `05_stgnn_model.py` | GCN + GRU 时空图网络，航运加权邻接矩阵 |
| 整合 | `run_all.py` | 顺序运行全部模型 → 105 行结果 + 12 张对比热力图/柱状图 |
| 补充实验 | `test_lag_ma_improvement.py`, `test_gfw_vs_pw.py`, `test_cv_enhancement.py` | lag/MA 效果、GFW vs PortWatch、CV 特征 A/B 测试 |
| 可视化 | `extra_visualizations.py` | 6 张额外 dashboard 图 |

- 终端成功运行 `run_all.py`（exit code 0），结果保存至 `01_literature/Test/results/`。

### Results 主要实验结果

**方向预测（3-class，test ≈ 155 weeks for M1）**

| 最佳模型 | Layer | Accuracy | Macro F1 | 备注 |
|---|---|---:|---:|---|
| RandomForest | M3 | 54.5% | 0.37 | 全模型最高 |
| SVM | M2 | 48.4% | 0.33 | |
| TFT | M2 | 49.5% | 0.28 | 深度模型中最好 |
| XGBoost | M1 | 38.1% | 0.26 | **低于** LR/RF/SVM |
| ST-GNN | M1 | 46.5% | 0.23 | 几乎只预测 down 类 |

- 所有模型对 `flat` 类 recall = 0（样本太少，模型全部忽略中间类）。
- M2/M3/M4 相对 M1 的方向预测**没有一致改善**；M3 RandomForest 是唯一明显优于 M1 的组合。

**价格回归（test RMSE, $/bbl）**

| 最佳模型 | Layer | RMSE | R² | 备注 |
|---|---|---:|---:|---|
| Ridge | M1 | 3.36 | 0.835 | **全实验最佳** |
| Ridge | M2 | 3.11 | 0.845 | RS 略改善 |
| RandomForest | M3 | 3.45 | 0.794 | |
| XGBoost | M4 | 4.18 | 0.714 | |
| LSTM | M2 | 74.4 | −89.3 | 严重过拟合 |
| ST-GNN | M5 | 25.1 | −9.3 | 严重过拟合 |

- 线性模型（Ridge）全面优于树模型和深度模型；深度模型 R² 大量为负。
- lag/MA 特征对 M1 price 帮助显著：XGBoost RMSE 4.95 → 3.92，R² 0.64 → 0.78。

**波动率回归**

- 最佳 RMSE ≈ 0.009（XGBoost M3），但所有模型 R² ≤ 0（波动率本身难预测）。
- SVM 波动率 RMSE 高达 0.09–0.10，应排除。

**补充 A/B 实验**

| 实验 | 关键发现 |
|---|---|
| lag/MA | M1 direction accuracy 48.4%→45.8%（略降），但 M1 price R² 0.64→0.78（显著提升）→ **保留 lag/MA** |
| GFW vs PortWatch | GFW-only (727 wks) 全面优于 PW-only (364 wks)；**不应简单拼接 GFW+PW**（重叠期仅 362 wks，性能反而下降） |
| CV 增强 (FRT/EMODnet) | FRT 方向 accuracy 75%（仅 12 test weeks，样本过小不可信）；EMODnet 方向 60.3% vs 无 EMODnet 54.5%（414 vs 727 weeks）→ 有潜力但需更长样本验证 |

**SHAP / Attention 输出**

- XGBoost M4/M5 的 SHAP summary plot（direction/price/volatility × 6 张）已生成。
- TFT M4/M5 的 attention heatmap + variable importance 已生成。

### Decisions made

- M1 变量从文献机制出发固定为 9 个（不再追求"5–10 个"的简化说法，Costa et al. 证明有效变量数随时间变化）。
- 必须加入**随机游走/不变预测基准**（P053 要求，当前 pipeline 缺失）。
- 深度模型预测目标应从 price level 改为 **log return 或 price change**（当前 price level 数值过大导致 LSTM/TFT/ST-GNN 梯度爆炸）。
- GFW 和 PortWatch **分开评价**，不做简单列拼接；主分析用 GFW-only（更长样本）。
- FRT 保留为概念验证特征，正式 M2 先用 NTL-only 子集，FRT 有数据月份才启用。
- M5 (GDELT) 定位调整为 Discussion/Appendix 补充实验，不纳入主消融框架。

### What needs fixing 该怎样修改

1. **加入随机游走基准**：direction = 预测与本周同方向；price = 预测本周价格；volatility = 预测本周波动率。P053 强调这是必须击败的强基准。
2. **深度模型目标变量**：price task 改用 `brent_log_return` 或 `target_brent_return_next_1w`，并对输入做 StandardScaler（序列模型目前未标准化 price target）。
3. **三分类不平衡**：考虑改为**二分类**（up/down，去掉 flat）或使用 `class_weight='balanced'` / SMOTE；当前 flat recall = 0 使 macro F1 无意义。
4. **XGBoost 调参**：当前默认参数表现差于 Ridge/RF，需 grid search 或 optuna；early stopping 的 validation set 可能过小。
5. **GFW/PW 时间对齐策略**：M3 应设计为"GFW 列 + PW 列各自 ffill，取 union 时间索引"，而非要求两源同时非缺失（否则样本从 727 骤降至 362）。
6. **FRT 检测改进**：当前 Hough 参数对 10m S2 分辨率过敏感，`n_tanks=0` 月份过多；需下载真实 GEE 导出 patches 或改用 YOLOv8 检测（project_plan Phase 3.5）。
7. **PCA 降维**：M2 (137 feat) / M4 (254 feat) 尚未做 PCA，Meeting 02 建议的降维方案未实施。
8. **正式 Diebold–Mariano 检验**：`evaluation.py` 已实现但未在 `run_all.py` 中批量输出跨层对比的 DM p-value 表。
9. **M1 变量微调**：文献建议补充 OVX（石油波动率指数）、Brent-WTI spread、gold price（P004）；S&P 500 作为全球需求代理证据较弱（P053）。
10. **ST-GNN 图结构**：当前用简化邻接矩阵，project_plan 5.4 的正式供应链图（OGIM/GOGET/NGA WPI）尚未构建。

### Cursor `.plan.md` 追踪（`~/.cursor/plans/`）

> Plan 文件不在 repo 内，由 Cursor Agent 生成；以下对照 plan 内 todo 状态与**实际代码/数据**是否完成。

#### 1. `frt_+_emodnet_implementation_69e97589.plan.md` — FRT + EMODnet CV 增强

| Plan todo | Plan 标记 | 实际状态 |
|---|---|---|
| EMODnet 栅格提取 (`extract_emodnet_density.py`) | ✅ completed | **真正完成** |
| GEE S2 patch 导出 (`export_s2_patches_frt_gee.js`) | ✅ completed | **未完成/未入库** — 脚本不在 repo |
| FRT 检测 + 阴影分析 (`frt_fill_level_estimation.py`) | ✅ completed | **仅 demo 版** — 用 `create_demo_patches()` 合成影像，非真实 S2 |
| 合并特征 + 重跑模型 (`merge_cv_features.py`) | ✅ completed | **已完成** + `test_cv_enhancement.py` |

Plan 原文仍未做：GEE 导出 Fujairah 真实 S2（~100 张）并下载；可选 YOLOv8（实际用 Hough Circle）；Rotterdam FRT；用真实影像重跑 M2/M4。

#### 2. `result_improvement_strategy_25a262ef.plan.md` — 结果改善策略

| Plan todo | Plan 标记 | 实际状态 |
|---|---|---|
| A1: lag/MA 特征 | ❌ pending | **已实现**（`LAG_MA_SPECS` + `data_loader.py`），plan 未更新 |
| A2: M3 改用 GFW 扩大样本 | ❌ pending | **部分完成**（`M3_SHIP_GFW` + `test_gfw_vs_pw.py`），union 对齐策略未修 |
| A3: direction 改二分类 | ❌ pending | **未做**（flat recall 仍为 0） |
| B3: GDELT 作为 M5 补充实验 | ❌ pending | **已实现**（`M5_GDELT_ADD` + `run_all.py`），plan 未更新 |
| 重跑并对比改善幅度 | ❌ pending | **部分完成**（`test_lag_ma_improvement.py` 等），系统性 rerun 未闭环 |

Plan 原文仍未做：A3 二分类或放宽 flat 阈值；A4 波动率 log 变换/离散化；按 Phase 1→2→3 顺序系统性重跑。C1/C3（扩年份、改日频）plan 标注不推荐，可忽略。

#### 3. `gfw_sar_船舶检测集成_75cbbdc4.plan.md` — GFW SAR 暗船（方案 B）

**5 个 todo 全部 ❌ pending，零进展：**

- [ ] `download_gfw_sar_detections.py`（GFW SAR API，`public-global-sar-presence`）
- [ ] SAR 月度 → 周频，合并进 `weekly_shipping_features.csv`
- [ ] 重建 `weekly_features.csv` + 更新 `feature_groups.json`
- [ ] SAR 时序 + 暗船比例 EDA 图
- [ ] 重跑 ablation 验证 SAR 增量价值

预计工作量 ~1 天；对应 `船舶检测可行性分析` plan 的推荐方案 B。

#### 4. `expand_aoi_sites_96edc498.plan.md` — AOI 8 → 11 扩站

| Plan todo | Plan 标记 | 实际状态 |
|---|---|---|
| 更新 `aoi_oil_infrastructure.csv` + 3 个 GEE 脚本 | ✅ completed | **已完成** |
| GEE 全量重跑 11 站 CSV 导出 | ❌ pending | **需手动在 GEE 操作** |
| 重跑 `aggregate_remote_sensing_to_weekly.py` 及下游 | ❌ pending | **未做** — P009/P010/P011 尚无遥感数据 |
| leave-one-AOI-out 敏感性测试 | ❌ pending | **未做** |

#### 5. `dataset_之后的研究路线_cbf180fc.plan.md` — 早期路线图

全部 todo 标 ✅ completed，但基于旧框架（M2=Text/NLP 等）。**以 `project_plan_20260601.md` 为准，不再跟进。**

#### 6. `船舶检测可行性分析_6b6811be.plan.md` — 决策分析（无 todo）

结论：推荐方案 B（GFW SAR 暗船）→ 即 plan #3，尚未实施；方案 A（自跑 S2 检测）不推荐；方案 C（不做，写 Future Work）论文中未定稿。

#### 7. `fix_pipeline_data_chain_e6f331d0.plan.md`

全部 ✅ completed，属于**另一个项目**（AI 城市 adoption notebook），与 dissertation 无关。

#### Plan 汇总（按优先级）

```
✅ 实质完成
  └─ frt+emodnet：EMODnet 全流程
  └─ expand AOI：脚本/CSV 更新
  └─ result_improvement：A1 lag/MA、B3 M5（plan 未更新）

⚠️ Plan 标 completed 但实质未完成
  └─ frt+emodnet：真实 S2 GEE 导出 + 真实 FRT（目前只有 demo）

❌ Plan 仍 pending、代码未做
  └─ result_improvement：A3 二分类、A4 波动率变换、系统性 rerun
  └─ gfw_sar：全部 5 步
  └─ expand_aoi：GEE 重导出、重聚合、leave-one-out
```

---

### Plan items NOT done 尚未完成的 plan 项

> **合并来源（只记录在 diary，不修改原文件）：**
> - `00_admin/project_plan_20260601.md` — §7 Methodology Pipeline、§12 Phase 0–4、§5.4 供应链图、§11 评估框架
> - Cursor plans（`~/.cursor/plans/`）— 见上文「Cursor `.plan.md` 追踪」
>
> **状态图例：** ✅ 完成　🔶 部分完成　⬜ 未做　⏸ 已搁置/低优先级

---

#### A. Methodology Pipeline（project_plan §7）

| Step | 内容 | 状态 | 缺口 / 对应 Cursor plan |
|---|---|---|---|
| 1 | 数据采集、清洗、时间对齐 | ✅ | GFW/PW **union 对齐**未修（`result_improvement` A2） |
| 2 | 供应链节点识别与图构建 | ⬜ | 静态数据已下载；**正式图未建**（§5.4；ST-GNN 仍用简化邻接矩阵） |
| 3 | 代理变量与特征工程 | 🔶 | 金融/RS/航运脚本已有；SAR 暗船、港口拥堵、11 站 RS 重导出未完成 |
| 4 | 文献综述与变量筛选 | 🔶 | M1 7/7 精读完成；M2/M3/GNN 组、PCA、`feature_groups.json` 未收尾 |
| 5 | Baseline ML（XGBoost × M1–M4） | 🔶 | 首次全量运行完成；**缺随机游走基准、调参、DM 检验、checkpoint 未过** |
| 6 | 扩展建模（LSTM/TFT/ST-GNN） | 🔶 | 代码+首次运行完成；**深度模型过拟合未修复**；TFT 未用 `pytorch-forecasting` |
| 7 | 评估、解释与稳健性 | ⬜ | SHAP/attention 图已有；**跨层解读、子期间分析、DM p-value 表**未做 |

---

#### B. Phase 0 — 文献综述与变量筛选（project_plan §12 Phase 0）

| # | project_plan 任务 | 状态 | 备注 |
|---|---|---|---|
| 1 | 结构化文献综述（Paper/Task/Model/Variables/Evaluation/Findings） | 🔶 | M1 组 7 篇 ✅ + 独立 notes；②③④ 组及速读 8 篇 ⬜ |
| 2 | 基于文献 refine M1 金融变量 | 🔶 | `config.py` 定 9 个；文献建议 OVX/gold/Brent-WTI spread 等待建（`feature_groups M1.json` 有 `M1_to_build` 清单） |
| 3 | 文献论证 M2 遥感变量 | 🔶 | matrix ② 组有推荐 5 NTL + 可选 FRT；**无独立 reading notes** |
| 4 | 文献论证 M3 航运变量 | 🔶 | matrix ③ 组有推荐 7 PortWatch + 可选 GFW/congestion；**无独立 reading notes** |
| 5 | 更新 `feature_groups.json` → M1–M4 结构 | ⬜ | 旧版 ~27 变量 M1 仍在用；草稿 `feature_groups M1.json` 未合并入主文件 |
| 6 | 从变量表移除 text/GDELT 主框架 | ✅ | Meeting 02 决策；M5 GDELT 仅作 Appendix 补充（Cursor `result_improvement` B3 已实现） |

**Phase 0 未完成清单：**
- [ ] ② 遥感精读：P069/P024/P025/P032/P055 → 独立 reading notes
- [ ] ③ 航运精读：P016/P017/P070/P018/P084 → 独立 reading notes
- [ ] ④ GNN 精读：G1–G6 → matrix 建组 + notes（project_plan §13.2）
- [ ] 速读 8 篇：P014/P015/P056/P057/P061/P075/P077/P073
- [ ] PCA 降维方案确定（Meeting 02 建议；§6 注；或 SHAP 筛选替代，P059）
- [ ] 合并/更新 `feature_groups.json` 与 `01_literature/Test/config.py` 变量定义一致

---

#### C. Phase 0.5 — 卫星衍生特征（project_plan §12 + §5.3）

| # | project_plan 任务 | 状态 | Cursor plan 交叉引用 |
|---|---|---|---|
| 1 | GFW SAR 暗船计数（6 航道） | ⬜ | `gfw_sar_船舶检测集成` — **5 步全 pending** |
| 2 | 港口锚泊拥堵（11 AOI） | ⬜ | 同上 plan 可选扩展；project_plan §5.3B |
| 3 | 合并 shipping 矩阵 + 更新 feature_groups | ⬜ | 依赖 #1/#2；EMODnet 已单独合并（`frt+emodnet`） |
| — | EMODnet 密度（Rotterdam + Suez） | ✅ | `frt+emodnet` — 非 project_plan 原文，已额外完成 |
| — | FRT 液位估算 | 🔶 | `frt+emodnet` — **demo only**；真实 S2 见 Phase 3.5 |

**Phase 0.5 未完成清单：**
- [ ] `download_gfw_sar_detections.py`（dataset: `public-global-sar-presence`）
- [ ] SAR 聚合至周频 → `weekly_shipping_features.csv`
- [ ] 重建 `weekly_features.csv` + 更新 `feature_groups.json`
- [ ] SAR 时序 + 暗船比例 EDA
- [ ] 重跑 ablation 验证 SAR 增量
- [ ] 港口拥堵特征（`port_congestion_{aoi}`，P084 依据）

---

#### D. Phase 1 — Tier 1：Baselines + XGBoost × M1–M4（project_plan §12 + §9 Tier 1）

| # | project_plan 任务 | 状态 | 备注 |
|---|---|---|---|
| 1 | M1 上训练 LR/Ridge/RF/SVM + XGBoost | ✅ | `01_baselines.py` + `02_xgboost_model.py` |
| 2 | 同套模型跑 M2/M3/M4 | ✅ | 含 M5 补充层 |
| 3 | M1–M4 横向对比表 | 🔶 | `all_results_combined.csv` 已有；**正式解读未写** |
| 4 | XGBoost SHAP → 模态/特征贡献 | 🔶 | 图已生成；**跨层文字分析未写** |
| 5 | **Checkpoint：结果是否支撑 dissertation core** | ⬜ | **当前否** — 多模态未一致改善；XGBoost 弱于 Ridge/RF |

**Phase 1 未完成清单（含 Cursor `result_improvement`）：**
- [ ] 随机游走 / 不变预测基准（P053/P054 必备；project_plan §11 隐含）
- [ ] XGBoost 调参（grid search / optuna）
- [ ] M2/M3/M4 上应用 PCA（project_plan §10.2）
- [ ] 跨层 DM 检验批量输出（project_plan §11.3；P058）
- [ ] A3 方向改二分类或放宽 flat 阈值（Cursor plan）
- [ ] A4 波动率 log 变换 / 高-low 二分类（Cursor plan）
- [ ] GFW/PW union 时间索引 + 各自 ffill（Cursor plan A2）
- [ ] lag/MA 已落地 ✅；**系统性 rerun 闭环** ⬜
- [ ] 决定 dissertation core 是否 = Tier 1 only

---

#### E. Phase 2 — Tier 2：LSTM + TFT × M1–M4（project_plan §12 + §9 Tier 2）

| # | project_plan 任务 | 状态 | 备注 |
|---|---|---|---|
| 1 | LSTM + sliding window | ✅ | `03_lstm_model.py`，lookback=12 |
| 2 | LSTM 跑 M1–M4 | ✅ | 严重过拟合（price R² 大量为负） |
| 3 | TFT via `pytorch-forecasting` | 🔶 | 用**纯 PyTorch 简化版**替代（`04_tft_model.py`） |
| 4 | TFT static/known/unknown 输入设计 | ⬜ | project_plan §10.4 设计未完全实现 |
| 5 | LSTM/TFT vs XGBoost 跨层对比 | 🔶 | 热力图已有；**结论未写** |
| 6 | TFT attention → 时间步/变量解读 | 🔶 | 图已生成；**跨层分析未写** |

**Phase 2 未完成清单：**
- [ ] 深度模型 target 改 log return + StandardScaler（见「What needs fixing #2」）
- [ ] 修复 LSTM/TFT 过拟合后重跑
- [ ] （可选）迁移至 `pytorch-forecasting` 官方 TFT
- [ ] TFT attention 跨层解读写入 Results/Discussion

---

#### F. Phase 3 — Tier 3：ST-GNN × M1–M4（project_plan §12 + §9 Tier 3 + §5.4）

| # | project_plan 任务 | 状态 | 备注 |
|---|---|---|---|
| 1 | 构建供应链图（11 nodes, 6+ edges） | ⬜ | OGIM/GOGET/NGA WPI 静态数据已有 |
| 2 | 实现 ST-GNN（参考 G2 GWNet-Attn） | 🔶 | `05_stgnn_model.py` 简化 GCN+GRU 版 |
| 3 | M1–M4 训练 | ✅ | 效果差 |
| 4 | vs XGBoost/LSTM 对比 | 🔶 | 结果在 `stgnn_results.csv`；未解读 |
| 5 | 图结构是否带来增量 | ⬜ | 当前无法回答（图未正式构建 + 模型未收敛） |

**Phase 3 未完成清单：**
- [ ] 正式供应链图构建与文献论证（G1/G5）
- [ ] ST-GNN 训练修复（target/标准化/图结构）
- [ ] leave-one-AOI-out 敏感性测试（Cursor `expand_aoi`）

---

#### G. Phase 3.5 — FRT CV 增强（project_plan §12，可选）

| # | project_plan 任务 | 状态 | Cursor plan 交叉引用 |
|---|---|---|---|
| 1 | 选定 2–3 站点（Fujairah/Rotterdam/Cushing） | 🔶 | 仅 Fujairah demo |
| 2 | Sentinel-2 影像时间序列 | ⬜ | `export_s2_patches_frt_gee.js` **不在 repo** |
| 3 | YOLOv8 检测 → 阴影 → 液位 | 🔶 | 实际用 Hough Circle；YOLOv8 未装 |
| 4 | 月频 → 周频 → M2/M4 | ✅ | demo 数据已合并 |
| 5 | 重跑 XGBoost M2/M4 对比 | 🔶 | `test_cv_enhancement.py` 已跑；**样本仅 12 test weeks** |
| 6 | 时间不足则作 proof-of-concept | 🔶 | 当前定位：概念验证 + Future Work |

**Phase 3.5 未完成清单：**
- [ ] GEE 导出 Fujairah 真实 S2 patch（~100 张）并下载
- [ ] Rotterdam FRT（project_plan 原文 2–3 站点）
- [ ] 真实影像下重跑 M2/M4；或论文中明确标注 demo 限制

---

#### H. Phase 4 — 分析与写作（project_plan §12）

| # | project_plan 任务 | 状态 |
|---|---|---|
| 1 | 完整 2D 评估表（models × layers × metrics） | 🔶 数据有，解读 ⬜ |
| 2 | 跨层/跨模型 SHAP 分析 | 🔶 图有，文字 ⬜ |
| 3 | 子期间稳健性（COVID、红海、Hormuz 等） | ⬜ project_plan §11.5 |
| 4 | Results / Discussion / Conclusion | ⬜ |
| 5 | Future Work 段落（graph/text/GNN/FRT 扩展） | ⬜ |
| — | Introduction / Literature Review / Methodology 初稿 | ⬜ |

---

#### I. 数据与基础设施（project_plan §5 + Cursor `expand_aoi`）

| 任务 | 状态 | 来源 |
|---|---|---|
| AOI 8→11：CSV + GEE 脚本更新 | ✅ | `expand_aoi` |
| GEE 全量重跑 11 站 CSV 导出 | ⬜ | `expand_aoi` — P009/P010/P011 **无遥感数据** |
| 重跑 `aggregate_remote_sensing_to_weekly.py` 及下游 | ⬜ | `expand_aoi` |
| Test pipeline 迁移至 `04_code/` | ⬜ | diary backlog |
| Meeting 03 进度汇报材料 | ⬜ | — |

---

#### J. 评估框架缺口（project_plan §11，跨 Phase）

- [ ] 随机游走基准（§11 对照必备，P053）
- [ ] M2 vs M1、M3 vs M1、M4 vs M1 等横向对比的 **DM test p-value 表**（§11.3，P058）
- [ ] 纵向对比：LSTM vs XGBoost、TFT vs LSTM、ST-GNN vs XGBoost（§11.4）— 数据部分有，检验 ⬜
- [ ] PCA loading 分析 或 SHAP-selected vs PCA-reduced 对照（§11.5，P059/Jung P068）
- [ ] 危机子期间 vs 平静期对比（§11.5）

---

#### K. 优先级排序（合并两来源）

| 优先级 | 任务 | 来源 |
|---|---|---|
| **P0** | 随机游走基准 + 二分类 + GFW/PW union 对齐 + 系统性 rerun | `result_improvement` + §11 |
| **P0** | DM 检验输出 + checkpoint 判断 dissertation core | §12 Phase 1 |
| **P1** | GEE 重导出 11 站 + 重聚合 pipeline | `expand_aoi` + §7 Step 3 |
| **P1** | 更新 `feature_groups.json` + 遥感/航运文献各 ≥2 篇精读 | §12 Phase 0 |
| **P1** | Meeting 03 汇报材料 | diary |
| **P2** | GFW SAR 暗船全套（~1 天） | `gfw_sar` + §12 Phase 0.5 |
| **P2** | XGBoost 调参 + PCA（M2/M4） | §10.2 + Phase 1 |
| **P2** | 深度模型修复 + ST-GNN 正式图 | §12 Phase 2–3 |
| **P3** | 真实 S2 FRT 链路（5–7 天） | `frt+emodnet` + §12 Phase 3.5 |
| **P3** | 子期间分析 + 全部写作初稿 | §12 Phase 4 |
| **⏸** | 日频重构 / 向前扩年至 1990（对多模态无帮助） | `result_improvement` C1/C3 |
| **⏸** | 自跑 S2 船舶检测（方案 A） | `船舶检测可行性分析` — 不推荐 |

---

#### L. 已搁置 / 不再跟进

- `dataset_之后的研究路线_cbf180fc.plan.md` — 旧框架（M2=Text），以 project_plan 为准
- `fix_pipeline_data_chain_e6f331d0.plan.md` — 其他项目，与 dissertation 无关
- project_plan 主框架内 GDELT/Text — 已移除；M5 仅 Appendix

### Next tasks（按 §K 优先级）

1. P0：`result_improvement` — A3 二分类 + 随机游走基准 + GFW/PW union + 系统性 rerun + DM 表
2. P1：`expand_aoi` — GEE 11 站重导出 + 重跑聚合；更新 `feature_groups.json`
3. P1：遥感/航运文献精读；Meeting 03 汇报
4. P2：`gfw_sar` 全套；XGBoost 调参 + PCA
5. P3：真实 FRT S2 链路或明确写入 Future Work；Phase 4 写作

---

# Task List

> **Current phase:** Phase 02 — Literature review & variable/model selection（建模 pipeline 已提前搭建，进入 Phase 03 预备）
> **Details:** `00_admin/project_plan_20260601.md` + Cursor plans（`~/.cursor/plans/`）
> **Next meeting:** Meeting 03 — Jun 17

## Current Sprint 当前冲刺 (Meeting 03 — Jun 17)

**文献**
- [x] M1 金融变量文献精读 7/7 篇（含 P076）
- [ ] 遥感/航运文献组精读 + 独立 reading notes
- [ ] 更新 `feature_groups.json` 至 M1–M4 结构

**数据 / 特征（Cursor plans）**
- [x] EMODnet 密度提取（plan: frt+emodnet）
- [x] FRT demo pipeline（plan: frt+emodnet — 真实 S2 未做）
- [ ] GEE 重导出 11 站遥感（plan: expand_aoi）
- [ ] GFW SAR 暗船全套（plan: gfw_sar — 5 步全 pending）
- [ ] 港口锚泊拥堵特征

**建模 / 修复（Cursor plan: result_improvement）**
- [x] lag/MA 特征工程（plan 仍标 pending，代码已实现）
- [x] M5 GDELT 补充实验（plan 仍标 pending，代码已实现）
- [x] 完整 pipeline 首次全量运行（Tier 1–3）
- [ ] A3 方向改二分类
- [ ] 随机游走基准 + 深度模型修复 + GFW/PW union 对齐
- [ ] 系统性 rerun + DM 检验输出
- [ ] PCA 降维（M2/M4）
- [ ] Meeting 03 汇报材料

## Cursor Plan 待办速查

| Plan 文件 | 未完成项 |
|---|---|
| `result_improvement_strategy_25a262ef` | A3 二分类、A4 波动率变换、系统性 rerun |
| `gfw_sar_船舶检测集成_75cbbdc4` | 全部 5 步（下载/聚合/重建/EDA/ablation） |
| `expand_aoi_sites_96edc498` | GEE 重导出、重聚合、leave-one-out |
| `frt_+_emodnet_implementation_69e97589` | 真实 S2 GEE 导出、真实 FRT、Rotterdam |

## Backlog 待办

- 将 Test pipeline 迁移至 `04_code/`
- 撰写 Introduction / Literature Review / Methodology 初稿
- Results / Discussion / Conclusion 初稿
- 子期间稳健性分析（COVID、红海危机等）
- ST-GNN 正式供应链图 + SHAP/TFT 文字解读

## Completed 已完成

- Set up project folder structure 搭建项目文件夹结构
- Create research diary, prompt log, meeting note templates 创建研究日记、提示词日志、会议笔记模板
- Reorganise meeting 01 notes into structured format 将第一次会议笔记整理为结构化格式
- Check ShipRSImageNet categories and timestamps 检查 ShipRSImageNet 类别和时间戳
- Search for open AIS / vessel tracking / shipping activity datasets 搜索开放的 AIS/船舶跟踪/航运活动数据集
- Build initial dataset inventory table in `03_data/external_sources.md` 在 `03_data/external_sources.md` 中建立初始数据集清单表
- Build detailed variable overview in `03_data/Dataset_Overview.ipynb` 在 `03_data/Dataset_Overview.ipynb` 中建立详细变量概览
- Read initial oil forecasting and remote sensing papers 阅读初步的油价预测和遥感论文
- Data acquisition: EIA, GDELT, Aramco, OGIM, GOGET, WPI, EMODnet, S&P 500 数据采集
- Remote sensing pipeline: Sentinel-2, Landsat, VIIRS for 8→11 AOIs（脚本已更新，GEE 重导出 pending）
- Shipping pipeline: PortWatch + GFW combined weekly features 航运数据流水线
- Build unified feature matrix (`build_feature_matrix.py`) 构建统一特征矩阵
- Present dataset inventory at Meeting 02 (Jun 05) 在第二次会议展示数据集清单
- M1 文献精读 7 篇 + beatrice_task_literature_matrix 更新
- EDA for M1–M4 variable subsets (`01_literature/EDA/`)
- CV features: EMODnet density + FRT demo pipeline (`merge_cv_features.py`)
- Full modelling pipeline (Tier 1–3) first run (`01_literature/Test/`)
- lag/MA 特征 + M5 GDELT 层 + 3 组 A/B 实验（result_improvement plan 部分落地）

