# CASA0004 Dissertation 项目结构总览

**项目根目录：** `/Users/shirly/CASA/casa0004 Dissertation/`  
**组织逻辑：** 按论文工作流编号 `00–07`，从行政管理 → 文献 → 数据 → 代码 → 输出 → 写作 → 提交。  
**模态逻辑：** 数据与特征按 **M1（金融）/ M2（遥感）/ M3（航运）** 分层；消融实验 M0–M4 见 `01_literature/Test/`。  
**最后更新：** 2026-06-22

---

## 顶层结构

```text
casa0004 Dissertation/
├── .gitignore
├── Readme.md                          # 项目说明、目录规范、工作流
│
├── 00_admin/                          # 行政管理
├── 01_literature/                     # 文献 + 原型建模（Test/）+ EDA
├── 02_ai_conversations/               # AI 对话记录（Phase 1 / Phase 2）
├── 03_data/                           # 数据（Dataset 文档 + processed 按模态 + raw）
├── 04_code/                           # 可复用代码占位（当前为空）
├── 05_outputs/                        # 图表/模型结果占位（当前为空，输出在 Test/results/）
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
├── File Structure20260527.md          # 本文件（项目结构总览）
├── 2026-06-22_research_plan_e2e_multimodal.md   # 端到端多模态研究方案
├── research_diary.md                  # 研究日志
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
│   └── Meeting03_Mail 20260617.txt
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

**说明：** 提案、旧版项目计划、早期结构文档已移入 `Inactive/`；活跃管理文件保留在根级。

---

## `01_literature/` — 文献与原型实验

```text
01_literature/
├── literatue.md                       # 文献综述笔记
├── literature_matrix.md               # 文献矩阵（Markdown 版）
├── literature_matrix_P001_P093.xlsx   # 文献矩阵总表（93 篇）
├── papers_pdf/
│   └── .gitkeep                       # PDF 不上传 Git
├── reading_notes/
│   ├── 01 Finance/                    # P001, P004, P052–P054, P072, P076 …
│   ├── 02 RS/                         # P024, P025, P032, P055, P069 …
│   ├── 03 Shipping/                   # P016–P018, P070 …
│   ├── 04 /                           # P058, P059 …
│   ├── 05/                            # P068 …
│   └── Template/                      # _template.md + paper_01–05 示例
├── EDA/                               # 探索性数据分析（Beatrice 变量）
│   ├── eda_beatrice_variables.py
│   └── 01–10_*.png                    # 时序、相关、缺失、分布图
└── Test/                              # M0–M5 消融原型管线（~131 文件）
    ├── config.py
    ├── data_loader.py
    ├── evaluation.py
    ├── 01_baselines.py                # M0 基准
    ├── 02_xgboost_model.py
    ├── 03_lstm_model.py
    ├── 04_tft_model.py
    ├── 05_stgnn_model.py
    ├── run_all.py
    ├── extra_visualizations.py
    ├── test_*.py                      # CV、GFW vs PW、lag/MA 等诊断
    ├── results/                       # 当前版消融图表
    ├── results_before_m2/             # M2 更新前快照
    └── results_before_m3update/       # M3 更新前快照
```

**Readme 中规划但尚未出现：** `references.bib`

**说明：** 建模代码与消融输出暂存于 `Test/`，待稳定后迁移至 `04_code/` 与 `05_outputs/`。

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
│   ├── Dataset_Overview3.ipynb        # 当前主 notebook
│   ├── Dataset_Overview3 copy.ipynb
│   ├── aoi_oil_infrastructure_sites.md   # 11 AOI 站点说明
│   └── external_sources.md            # M1/M2/M3 外部数据源清单
│
├── processed/                         # 按模态分层的周频特征管线
│   ├── M1/
│   │   ├── py/build_m1_weekly.py
│   │   └── outputs/m1_weekly_features.csv   # 38 列周频金融特征
│   ├── M2/
│   │   ├── py/                        # 遥感聚合脚本（待建）
│   │   └── outputs/
│   ├── M3/
│   │   ├── py/aggregate_shipping_to_weekly.py
│   │   └── outputs/
│   └── merge/
│       ├── py/                        # 多模态合并脚本（待建）
│       └── outputs/
│
└── raw/                               # 原始数据（整目录 gitignore）
    ├── 00_spatial_anchors/            # 10 文件 — GIS 空间锚点
    ├── 01_market_financial/           # 32 文件 — M1 金融/宏观
    ├── 02_sentinel2/                  # 11 文件 — M2 遥感（Channel A/B）
    └── 03_shipping/                   # 106 文件 — M3 航运/AIS
```

**组织变更（2026-06）：**

- 原 `03_data/` 根级 notebook 与 `external_sources.md` 迁入 `Dataset/`。
- 原扁平 `processed/`（`weekly_features.csv` 等）重构为 **M1/M2/M3/merge** 四层，每层 `py/` + `outputs/`。
- 原 `raw/02_reports/`、`raw/03_news/` 已退出活跃数据目录（文本模态移除）。
- 原 `04_sentinel2/` → `02_sentinel2/`；原 `05_shipping/` → `03_shipping/`；原 `06_spatial_nodes/` → `00_spatial_anchors/`。

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

### `raw/02_sentinel2/`（11 文件）— M2 遥感/夜光

```text
02_sentinel2/
├── aoi_oil_infrastructure.csv         # 11 AOI 坐标与类型
├── load_aoi_config_gee.js             # GEE 共享 AOI 配置
├── Channel A/                         # 影像 patch 导出（端到端多模态）
│   ├── export_s2_patches_multimodal_gee.js
│   └── export_s2_patches_multimodal_gee_bundled.js
└── Channel B/                         # 机制变量（tabular 指标）
    ├── extract_sentinel2_monthly_indices_gee{,_bundled}.js
    ├── extract_viirs_monthly_nightlights_gee{,_bundled}.js
    ├── sentinel2_oil_sites_monthly_indices_201704_202512_11aoi.csv
    └── viirs_oil_sites_monthly_nightlights_201401_202512_11aoi.csv
```

**双通道设计（见 `2026-06-22_research_plan_e2e_multimodal.md`）：**

- **Channel A**：11 AOI × 月度 Sentinel-2 patch（6 波段 + SCL），供预训练 EO 模型提取 embedding。
- **Channel B**：NDVI/NDWI/NDBI/BSI/NTL/FRT 等 tabular 机制变量，保留经济可解释性。

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

**构建：** `processed/M3/py/aggregate_shipping_to_weekly.py` → `outputs/`（待产出）

---

## `04_code/` — 代码（占位）

```text
04_code/
├── notebooks/                         # 空
├── scripts/.gitkeep                   # 原 build_feature_matrix.py 等已移除
└── src/.gitkeep                       # 可复用模块（尚未实现）
```

**说明：** 当前活跃建模代码在 `01_literature/Test/`；稳定后迁移至此目录。

---

## `05_outputs/` — 分析输出（占位）

```text
05_outputs/                            # 当前为空
```

**说明：** 早期 EDA 图表（`01_brent_price_timeline.png` 等）与消融输出已迁移至 `01_literature/EDA/` 与 `01_literature/Test/results/`；正式论文图表待管线稳定后回写此目录。

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
01_literature/Test/  ──→  01_literature/Test/results/
        ↓                        ↓
04_code/ (待迁移)          05_outputs/ (待回写)
        ↓
06_writing
        ↑
00_admin / 01_literature
```


| 主题     | 内容                                                                                                                                            |
| ------ | --------------------------------------------------------------------------------------------------------------------------------------------- |
| 研究焦点   | 布伦特油价预测；三模态：**M1 金融/宏观**、**M2 遥感（S2 patch + VIIRS 夜光）**、**M3 航运（PortWatch/GFW/EMODnet）**；端到端表示级融合 vs 扁平特征融合                                   |
| 消融层级   | M0 基准 → M1 金融 → M1+M2 → M1+M3 → M1+M2+M3（M4）；原型见 `01_literature/Test/`                                                                        |
| 比较窗口   | 标准化 **2019–2026**（与导师确认）；M1 历史可延至 2006 做稳健性                                                                                                   |
| Git 策略 | 原始数据、PDF 文献、大文件（parquet/zip/tif）不入库；代码、processed CSV、写作 Markdown 可跟踪                                                                          |
| 关键文档   | `00_admin/2026-06-22_research_plan_e2e_multimodal.md`、`03_data/Dataset/external_sources.md`、`03_data/Dataset/aoi_oil_infrastructure_sites.md` |


---

## 文件数量汇总


| 目录                             | 约略文件数 | 状态                     |
| ------------------------------ | ----- | ---------------------- |
| `03_data/raw/`                 | ~160  | 本地存在，gitignore         |
| `03_data/processed/`           | ~5    | 部分 tracked（M1 产出 + 脚本） |
| `03_data/Dataset/`             | ~8    | tracked                |
| `01_literature/Test/`          | ~131  | tracked                |
| `01_literature/EDA/`           | ~13   | tracked                |
| `01_literature/reading_notes/` | ~25   | tracked                |
| `05_outputs/`                  | 0     | 占位                     |
| `06_writing/`                  | 7     | tracked                |
| 其余 admin / AI / code           | ~40   | tracked                |


