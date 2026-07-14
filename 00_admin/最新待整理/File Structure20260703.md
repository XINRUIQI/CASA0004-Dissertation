# CASA0004 Dissertation 项目结构总览

**项目根目录：** `/Users/shirly/CASA/casa0004 Dissertation/`  
**组织逻辑：** 按论文工作流编号 `00–07`，从行政管理 → 文献 → 数据 → 代码 → 输出 → 写作 → 提交。  
**模态逻辑：** 数据与特征按 **M1（金融）/ M2（遥感）/ M3（航运）** 分层；消融实验 M0–M4 见 `04_code/` 与 `05_outputs/baselines/`。  
**最后更新：** 2026-07-07

---

## 顶层结构

```text
casa0004 Dissertation/
├── .gitignore
├── Readme.md                          # 项目说明、目录规范、工作流
│
├── 00_admin/                          # 行政管理
├── 01_literature/                     # 文献矩阵 + 阅读笔记 + PDF（本地）
├── 02_ai_conversations/               # AI 对话记录（Phase 1 / Phase 2）
├── 03_data/                           # 数据（Dataset 文档 + processed 按模态 + raw）
├── 04_code/                           # 可复用建模代码（backtest 框架 + M1–M4 脚本）
├── 05_outputs/                        # 基准实验输出（baselines/m1–m4）
├── 06_writing/                        # 论文章节（Markdown）
└── 07_submission/                     # 最终提交（占位）
```

---

## 完整目录树（按文件夹展开）

### 根目录


| 文件/目录        | 说明                                                    |
| ------------ | ----------------------------------------------------- |
| `Readme.md`  | 论文题目占位、RQ、项目结构说明、命名规则、每日工作流                           |
| `.gitignore` | 忽略 `03_data/raw/`、`papers_pdf/`、`*.parquet`、`*.zip` 等 |


---

## `00_admin/` — 行政管理

```text
00_admin/
├── File Structure20260703.md          # 本文件（项目结构总览）
├── research_diary_phase1.md           # 研究日志 Phase 1
├── research_diary_phase2.md           # 研究日志 Phase 2
├── research_diary_phase3.md           # 研究日志 Phase 3（当前活跃）
├── timeline.md
├── meeting_notes/
│   ├── Meeting01 20260506.txt
│   ├── Meeting01_Mail 20260506.txt
│   ├── meeting01 20260506.md
│   ├── Meeting02 20260527.md
│   ├── Meeting02 20260527.txt
│   ├── Meeting02_Mail 20260527 copy.txt
│   ├── Meeting03 2020260617.md
│   ├── Meeting03 20260617.txt
│   ├── Meeting03_Mail 20260617.txt
│   └── Meeting04_prep_20260703.md     # Meeting 04 准备材料
├── 待整理/                              # 进行中方案与实验日志（待归档）
│   ├── 2026-06-22_research_plan_e2e_multimodal.md
│   ├── 2026-06-22_channelB_mechanism_plan.md
│   └── flat_baseline_log.md           # M0–M4 扁平基线实验唯一主记录（M2 §8）
└── Inactive/                          # 归档旧版文档
    ├── File Structure20260506.md
    ├── Meeting_KLP_20260609/
    ├── Proposal of Self-proposed Topic20260410.docx
    ├── project_plan_20260527.md
    ├── project_plan_20260601.md
    ├── stathord.jpg
    ├── 视觉.md
    └── 论文dataset20260411.docx
```

**说明：** 研究方案与基准实验日志暂存于 `待整理/`；活跃管理文件保留在根级；旧版文档在 `Inactive/`。

**文档变更（2026-07-03）：** 原 `2026-06-23_m2_baseline_results.md` 已合并入 `flat_baseline_log.md` §8 并删除；基线协议、M1–M4 结果、M2 完整分析现均指向单一文件。

---

## `01_literature/` — 文献

```text
01_literature/
├── literatue.md                       # 文献综述笔记
├── literature_matrix.md               # 文献矩阵（Markdown 版）
├── literature_matrix_P001_P101.xlsx   # 文献矩阵总表（101 篇）
├── papers_pdf/                        # PDF 本地存放（gitignore，34 篇）
│   ├── 01/                            # Finance: P001, P004, P052–P054, P072, P076
│   ├── 02/                            # RS: P024, P025, P032, P055, P069
│   ├── 03/                            # Shipping: P016–P018, P062, P066, P070
│   ├── 04/                            # 评估/可解释性: P058, P059
│   ├── 05/                            # M4 融合: P068
│   ├── 06/                            # 创新-金融+图编码器: P039, P063, P091
│   ├── 07/                            # 创新-EO基础模型: P094–P095, P103–P104, P106, P108, P110, P113
│   ├── 08/                            # 创新-EO多模态融合: P105, P107, P109, P111, P112, P114
│   ├── 09/                            # 创新-融合机制与缺失异步: P096–P101, P115
│   └── 10/                            # 创新-时序架构参考: P088, P089
└── reading_notes/
    ├── 01 Finance/                    # P001, P004, P052–P054, P072, P076
    ├── 02 RS/                         # P024, P025, P032, P055, P069
    ├── 03 Shipping/                   # P016–P018, P062, P066, P070
    ├── 04 评估方法 + 可解释性：M2/M3/M4 怎么比、怎么解释/   # P058, P059
    ├── 05 M4 多模态融合 + 降维逻辑/     # P068
    ├── 06 创新-金融+图编码器/           # P039, P063, P091（3 篇）
    ├── 07 创新-EO基础模型/              # P094–P095, P103–P104, P106, P108, P110, P113（8 篇）
    ├── 08 创新-EO多模态融合/            # P105, P107, P109, P111, P112, P114（6 篇）
    ├── 09 创新-融合机制与缺失异步/      # P096–P101, P115（7 篇）
    ├── 10 创新-时序架构参考/          # P088, P089（2 篇）
    └── Template/                      # _template.md + paper_01–05 示例
```

**Readme 中规划但尚未出现：** `references.bib`

**说明（2026-07）：** 早期原型实验目录 `Test/` 与 `EDA/` 已移除；建模代码已迁移至 `04_code/`，输出至 `05_outputs/`。

---

## `02_ai_conversations/` — AI 记录

```text
02_ai_conversations/
├── prompt_log.md
├── Phase 1/
│   ├── AI 2026-05-22.md
│   ├── AI 2026-05-25.md
│   ├── AI 2026-05-26 11&6.md
│   ├── AI 2026-05-26 dataset整理.md
│   └── AI 2026-05-26 shiprsimagenet_verification.md
└── Phase 2/
    ├── 1.md … 6.md
    ├── AI 2026-06-05.md
    ├── AI 2026-06-16.md
    ├── AI 2026-06-16 Featuers.md
    ├── variables_and_M1-M4_results.md
    └── variables_and_M1-M4_results_EN.md
```

---

## `03_data/` — 数据

```text
03_data/
├── Dataset/                           # 数据文档与探索 notebook
│   ├── Dataset_Original.0.ipynb
│   ├── Dataset_Original.1.ipynb
│   ├── Dataset_Overview.ipynb
│   ├── Dataset_Overview3.ipynb
│   ├── Dataset_Overview4.ipynb        # 当前主 notebook
│   └── external_sources.md            # M1/M2/M3 外部数据源清单
│
├── processed/                         # 按模态分层的周频特征管线（已产出）
│   ├── M1/
│   │   ├── m1_data_dictionary.md
│   │   ├── py/build_m1_weekly.py
│   │   └── outputs/m1_weekly_features.csv
│   ├── M2/
│   │   ├── m2_data_dictionary.md
│   │   ├── py/
│   │   │   ├── build_m2_weekly.py
│   │   │   ├── audit_m2_coverage.py
│   │   │   ├── audit_s2_patches.py
│   │   │   ├── eda_m2_mechanism.py
│   │   │   └── s2_patch_utils.py
│   │   └── outputs/
│   │       ├── m2_weekly_features.csv
│   │       ├── m2_weekly_features_watermask.csv
│   │       ├── s2_patch_index.csv
│   │       ├── s2_patch_coverage_report.csv
│   │       ├── m2_coverage_report.csv
│   │       ├── m2_eda_*.csv / *.png   # EDA 与覆盖审计图
│   │       └── s2_patch_validity_heatmap.png
│   ├── M3/
│   │   ├── m3_data_dictionary.md
│   │   ├── py/aggregate_shipping_to_weekly.py
│   │   └── outputs/m3_weekly_features.csv
│   └── merge/
│       ├── py/build_feature_matrix.py
│       └── outputs/
│           ├── weekly_feature_matrix.csv
│           ├── weekly_feature_matrix_full.csv
│           ├── weekly_feature_matrix_watermask.csv
│           ├── weekly_feature_dictionary.csv
│           ├── weekly_feature_dictionary_full.csv
│           └── weekly_feature_dictionary_watermask.csv
│
└── raw/                               # 原始数据（整目录 gitignore，~1123 文件）
    ├── 00_spatial_anchors/            # 10 文件 — GIS 空间锚点
    ├── 01_market_financial/           # 32 文件 — M1 金融/宏观
    ├── 02_sentinel2/                  # ~980 文件 — M2 遥感（Channel A/B + patches）
    └── 03_shipping/                   # 106 文件 — M3 航运/AIS
```

**组织变更（2026-06 → 2026-07）：**

- 原 `03_data/` 根级 notebook 与 `external_sources.md` 迁入 `Dataset/`。
- 原扁平 `processed/` 重构为 **M1/M2/M3/merge** 四层，每层 `py/` + `outputs/`；**M1–M3 与 merge 均已产出**。
- `aoi_oil_infrastructure_sites.md` 位于 `raw/02_sentinel2/`（与 GEE 脚本同目录）。
- Channel A **S2 patch 影像**（967 个 `.tif`）已下载至 `raw/02_sentinel2/Channel A/s2_patches/`。
- Channel B 新增 **watermask** 变体（`extract_sentinel2_monthly_indices_watermask_gee.js` + 对应 CSV）。
- 原 `01_literature/Test/` 与 `EDA/` 中的 M2 EDA 图表已迁移至 `processed/M2/outputs/`。
- 新增 `processed/M2/m2_data_dictionary.md`（Channel B 数据字典，与 M1/M3 对齐；含 5 指标精确波段公式、as-of 对齐与建模合约）。

---

### `raw/00_spatial_anchors/`（10 文件）— 油气基础设施 GIS

```text
00_spatial_anchors/
├── GOGET/Global-Oil-and-Gas-Extraction-Tracker-March-2026.xlsx
├── GOIT/GEM-GOIT-Oil-NGL-Pipelines-2025-03.{xlsx,geojson,gpkg}
├── OGIM/ogim_{core,extended}_infrastructure_nodes_global_20240515.csv
└── world_port_index/NGA_WPI_ports_static_2026.csv + 字段说明 PDF
```

---

### `raw/01_market_financial/`（32 文件）— M1 金融/宏观

```text
01_market_financial/
├── download_m1_raw.py                 # 统一下载入口（EIA 手动 / FRED·Yahoo·Other 自动）
├── manifest.csv                       # 来源审计（SHA-256、覆盖区间、series ID）
├── EIA/
│   ├── Brent/EIA_brent_spot_price_daily*.xls
│   ├── WTI/EIA_WTI_cushing_crude_price_daily*.xls
│   └── Weekly Petroleum Status Report/EIA_*_weekly*.xls
├── FRED/                              # VIX、DXY、利率、IMF 工业原料等
├── Yahoo/                             # S&P 500、OVX、BZF 期货、CAD/AUD、GCF 黄金
└── Other/                             # DallasFed igrea、GPR (data_gpr_export.dta)
```

**构建：** `processed/M1/py/build_m1_weekly.py` → `outputs/m1_weekly_features.csv`（38 列，2006-01 ~ 2025-12）

---

### `raw/02_sentinel2/`（~980 文件）— M2 遥感/夜光

```text
02_sentinel2/
├── aoi_oil_infrastructure.csv         # 11 AOI 坐标与类型
├── aoi_oil_infrastructure_sites.md    # 11 AOI 站点说明
├── load_aoi_config_gee.js             # GEE 共享 AOI 配置
├── Channel A/                         # 影像 patch 导出（端到端多模态）
│   ├── export_s2_patches_multimodal_gee.js
│   ├── export_s2_patches_multimodal_gee_bundled.js
│   ├── s2_patch_exclusions.csv        # 排除/无效 patch 清单
│   └── s2_patches/                    # 967 个 GeoTIFF（11 AOI × 2019–2026，gitignore）
│       └── S2_P{001–011}_{Site}_{YYYY}_{MM}.tif
└── Channel B/                         # 机制变量（tabular 指标）
    ├── extract_sentinel2_monthly_indices_gee{,_bundled,_watermask}.js
    ├── extract_viirs_monthly_nightlights_gee{,_bundled}.js
    ├── sentinel2_oil_sites_monthly_indices_201704_202512_11aoi.csv
    ├── sentinel2_oil_sites_monthly_indices_watermask_201704_202512_11aoi.csv
    └── viirs_oil_sites_monthly_nightlights_201401_202512_11aoi.csv
```

**双通道设计（见 `00_admin/待整理/2026-06-22_research_plan_e2e_multimodal.md`）：**

- **Channel A**：11 AOI × 月度 Sentinel-2 patch（6 波段 + SCL），供预训练 EO 模型提取 embedding。
- **Channel B**：NDVI/NDWI/NDBI/BSI/NTL/FRT 等 tabular 机制变量；含标准版与 watermask 版。

---

### `raw/03_shipping/`（106 文件）— M3 航运/AIS

```text
03_shipping/
├── GFW/
│   ├── download_gfw_vessel_presence.py
│   └── gfw_chokepoint_vessel_presence_monthly.csv
├── IMF Portwatch/
│   ├── download_portwatch_chokepoints.py
│   ├── download_portwatch_ports.py
│   ├── portwatch_chokepoints_daily.csv
│   └── portwatch_ports_daily.csv
└── emodnet_vessel_density_monthly_2017-2025/
    ├── EMODnet_HA_Vessel_Density_10.zip
    ├── EMODnet_HA_Vessel_DensityMap_20251206_MD.xml
    └── vesseldensity_10_YYYYMMDD.tif   # 96 个月度 GeoTIFF（2017–2024）
```

**构建：** `processed/M3/py/aggregate_shipping_to_weekly.py` → `outputs/m3_weekly_features.csv`

---

## `04_code/` — 建模代码

```text
04_code/
├── scripts/
│   ├── run_baseline.py                # 扁平特征基准（Ridge / XGB / ARIMA / Naive）
│   ├── m1/sweep_m1.py
│   ├── m2/
│   │   ├── sweep_m2.py
│   │   ├── robustness_m2.py
│   │   └── shap_m2.py
│   ├── m3/
│   │   ├── sweep_m3.py
│   │   ├── robustness_m3.py
│   │   └── shap_m3.py
│   └── m4/
│       ├── sweep_m4.py
│       ├── robustness_m4.py
│       └── shap_m4.py
├── src/
│   └── backtest/                      # 滚动回测框架
│       ├── __init__.py
│       ├── data.py
│       ├── metrics.py
│       ├── models.py
│       └── rolling.py
└── notebooks/                         # 空（占位）
```

**说明：** 由原 `01_literature/Test/` 迁移而来；入口脚本 `run_baseline.py`，各模态 sweep / robustness / SHAP 分目录存放。

---

## `05_outputs/` — 分析输出

```text
05_outputs/
└── baselines/
    ├── m1/                            # M1 金融基准
    │   ├── baseline_{metrics,predictions}.csv
    │   └── sweep_{overview.png,summary.csv}
    ├── m2/                            # M2 遥感基准（anom / level / watermask / literature / LOAO）
    │   ├── baseline_*_{metrics,predictions}.csv   # 多设定变体
    │   ├── backtest_{anom,watermask}.png
    │   ├── shap_*.{png,csv}           # Ridge / XGB + 按 AOI / index 分解
    │   └── sweep_m2_{overview.png,summary.csv}
    ├── m3/                            # M3 航运基准
    │   ├── baseline_{metrics,predictions}.csv
    │   ├── backtest.png
    │   ├── shap_m3*.{png,csv}
    │   └── sweep_m3_{overview.png,summary.csv}
    └── m4/                            # M4 多模态融合基准
        ├── baseline_anom_*.{csv}
        ├── backtest_anom.png
        ├── shap_m4*.{png,csv}         # 含按 modality / M2 index / M3 source 分解
        └── sweep_m4_{overview.png,summary.csv}
```

**说明：** 完整实验记录见 `00_admin/待整理/flat_baseline_log.md`（§7 M1 sweep · §8 M2 · §9 M3 · §12 M4 · §11 复现命令）。

---

## `06_writing/` — 论文写作（7 章文件）

```text
06_writing/
├── outline.md
├── chapter_1_introduction.md
├── chapter_2_literature_review.md
├── chapter_3_methodology.md
├── chapter_4_results.md
├── chapter_5_discussion.md
└── references_notes.md
```

---

## `07_submission/` — 提交占位（3 空目录）

```text
07_submission/
├── appendix/.gitkeep
├── final_pdf/.gitkeep
└── reproducibility_pack/.gitkeep
```

---

## 数据流与组织要点

```text
03_data/raw/{M1,M2,M3}
        ↓
03_data/processed/{M1,M2,M3,merge}
        ↓
04_code/scripts/  ──→  05_outputs/baselines/{m1,m2,m3,m4}
        ↓
06_writing
        ↑
00_admin / 01_literature
```


| 主题     | 内容                                                                                                                                            |
| ------ | --------------------------------------------------------------------------------------------------------------------------------------------- |
| 研究焦点   | 布伦特油价预测；三模态：**M1 金融/宏观**、**M2 遥感（S2 patch + VIIRS 夜光）**、**M3 航运（PortWatch/GFW/EMODnet）**；端到端表示级融合 vs 扁平特征融合                                   |
| 消融层级   | M0 基准 → M1 金融 → M1+M2 → M1+M3 → M1+M2+M3（M4）；代码见 `04_code/`，结果见 `05_outputs/baselines/`                                                        |
| 比较窗口   | 标准化 **2019–2026**（与导师确认）；M1 历史可延至 2006 做稳健性                                                                                                   |
| Git 策略 | 原始数据、PDF 文献、大文件（parquet/zip/tif）不入库；代码、processed CSV、写作 Markdown 可跟踪                                                                          |
| 关键文档   | `00_admin/待整理/2026-06-22_research_plan_e2e_multimodal.md`、`03_data/Dataset/external_sources.md`、`03_data/raw/02_sentinel2/aoi_oil_infrastructure_sites.md`、`00_admin/待整理/flat_baseline_log.md` |


---

## 文件数量汇总


| 目录                             | 约略文件数 | 状态                              |
| ------------------------------ | ----- | ------------------------------- |
| `03_data/raw/`                 | ~1123 | 本地存在，gitignore（含 967 S2 patch） |
| `03_data/processed/`           | ~35   | tracked（M1–M3 + merge 全产出）      |
| `03_data/Dataset/`             | ~6    | tracked                         |
| `01_literature/reading_notes/` | ~40   | tracked（101 篇矩阵，40 篇笔记）         |
| `01_literature/papers_pdf/`    | ~34   | 本地 PDF，gitignore                |
| `04_code/`                     | ~21   | tracked（backtest 框架 + 脚本）      |
| `05_outputs/baselines/`        | ~65   | tracked（M1–M4 基准结果）             |
| `06_writing/`                  | 7     | tracked                         |
| `00_admin/`                    | ~26   | tracked（含 `File Structure20260703.md`） |
| `02_ai_conversations/`         | ~17   | tracked                         |
| `07_submission/`               | 0     | 占位                              |

