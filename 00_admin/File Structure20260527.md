# CASA0004 Dissertation 项目结构总览

**项目根目录：** `/Users/shirly/CASA/casa0004 Dissertation/`  
**组织逻辑：** 按论文工作流编号 `00–07`，从行政管理 → 文献 → 数据 → 代码 → 输出 → 写作 → 提交。  
**最后更新：** 2026-06-09

---

## 顶层结构

```text
casa0004 Dissertation/
├── .gitignore
├── Readme.md                          # 项目说明、目录规范、工作流
├── bp_press_release_links.csv         # 根目录散落的数据链接文件
│
├── 00_admin/                          # 行政管理
├── 01_literature/                     # 文献管理
├── 02_ai_conversations/               # AI 对话记录
├── 03_data/                           # 数据（原始 + 处理后 + notebooks）
├── 04_code/                           # 可复用代码与脚本
├── 05_outputs/                        # 图表、表格、模型结果
├── 06_writing/                        # 论文章节（Markdown）
└── 07_submission/                     # 最终提交（占位）
```

---

## 完整目录树（按文件夹展开）

### 根目录


| 文件/目录                        | 说明                                                    |
| ---------------------------- | ----------------------------------------------------- |
| `Readme.md`                  | 论文题目占位、RQ、项目结构说明、命名规则、每日工作流                           |
| `.gitignore`                 | 忽略 `03_data/raw/`、`papers_pdf/`、`*.parquet`、`*.zip` 等 |
| `bp_press_release_links.csv` | BP 新闻稿链接（未归入 `03_data/`）                              |


---

## `00_admin/` — 行政管理

```text
00_admin/
├── File Structure.md                  # 本文件（项目结构总览）
├── Proposal of Self-proposed Topic20260410.docx
├── 论文dataset20260411.docx
├── research_diary_20260527.md         # 研究日志
├── timeline.md
└── meeting_notes/
    ├── Meeting01 20260506.txt
    ├── Meeting01_Mail 20260506.txt
    ├── meeting01 20260506.md
    ├── Meeting02 20260527.md
    ├── Meeting02 20260527.txt
    └── Meeting02_Mail 20260527.txt
```

**说明：** 原有的 `phases/`、`supervisor_feedback/` 子目录及 `2026-05-06_meeting_01_beatrice.md` 已移除。

---

## `01_literature/` — 文献

```text
01_literature/
├── literatue.md                       # 文献综述笔记
├── papers_pdf/
│   └── .gitkeep                       # PDF 不上传 Git
└── reading_notes/
    ├── .gitkeep
    ├── _template.md
    ├── paper_01_oil_price_ml.md
    ├── paper_02_oil_price_dl.md
    ├── paper_03_remote_sensing_economic_proxy.md
    ├── paper_04_ais_shipping_oil.md
    └── paper_05_multimodal_forecasting.md
```

**Readme 中规划但尚未出现：** `literature_matrix.xlsx`、`references.bib`

---

## `02_ai_conversations/` — AI 记录

```text
02_ai_conversations/
├── prompt_log.md
├── AI 2026-05-22.md
├── AI 2026-05-25.md
├── AI 2026-05-26 11&6.md
├── AI 2026-05-26 dataset整理.md
└── AI 2026-05-26 shiprsimagenet_verification.md
```

---

## `03_data/` — 数据

```text
03_data/
├── external_sources.md              # 外部数据源清单
├── Dataset_Original.0.ipynb
├── Dataset_Original.1.ipynb
├── Dataset_Overview.ipynb
│
├── processed/                         # 清洗/聚合后的周频特征
│   ├── .gitkeep
│   ├── aggregate_gdelt_to_weekly.py
│   ├── build_weekly_time_index.py
│   ├── feature_groups.json
│   ├── weekly_time_index.csv
│   ├── weekly_features.csv
│   ├── weekly_features.parquet        # gitignore
│   ├── weekly_remote_sensing_features.csv
│   ├── weekly_shipping_features.csv
│   └── weekly_text_features.csv
│
└── raw/                               # 原始数据（整目录 gitignore）
    ├── 01_market_financial/           # 21 文件
    ├── 02_reports/                    # ~321 文件
    ├── 03_news/                       # ~207 文件
    ├── 04_sentinel2/                  # 11 文件
    ├── 05_shipping/                   # ~104 文件
    └── 06_spatial_nodes/              # 8 文件
```

---

### `raw/01_market_financial/`（21 文件）

```text
01_market_financial/
├── EIA_steo_monthly_2022_2027.xlsx
├── 1A Oil Price/
│   ├── EIA_brent_spot_price_daily*.xls
│   └── EIA_brent_spot_price_weekly*.xls
├── 1B Benchmark comparison/
│   ├── EIA_WTI_cushing_crude_price_daily*.xls
│   └── EIA_WTI_cushing_crude_price_weekly*.xls
├── 1C EIA Weekly Petroleum Status Report/   # 10 个 EIA 周度基本面 xls
└── 1D Macro-financial control variables/
    ├── FRED_*.csv（VIX、利率、美元指数等）
    ├── Yahoo_sp500_daily_*.csv
    └── download_sp500_yahoo.py
```

---

### `raw/02_reports/`（~321 文件）

```text
02_reports/
├── eia_steo/                          # 235 个 steo_YYYY_MM.pdf + download_steo_archive.py
├── opec_momr/                         # 65 个月度 OPEC MOMR PDF（含 2026 年 1–4 月）
└── opec_asb/                          # 20 个年度 OPEC ASB PDF（asb-2006.pdf … asb-2025.pdf）
```

---

### `raw/03_news/`（~207 文件）

```text
03_news/
├── .gitkeep
├── GDET/                              # GDELT 扰动事件/运输中断特征
│   ├── gdelt_oil_disruption_daily_*.csv（多版本/校准版）
│   ├── gdelt_transport_disruption_daily_*.csv
│   ├── gdelt1.0_* / gdelt2.0_*        # 10 个文本占位文件
│   └── gdelt_oil_disruption_daily_features_details_01011920-20052026/
│       └── exports_events-*.csv       # 83 个明细导出文件
├── aramco/
│   ├── news/                          # 爬虫脚本 + 新闻链接/正文 CSV（4 文件）
│   └── reports/                       # ~98 份 Saudi Aramco 财报/演示 PDF + 2 xlsx
└── shell_annual_reports/
    ├── scrape_shell_playwright.py
    ├── shell_annual_report_2022.pdf
    └── shell_annual_report_2025.pdf
```

---

### `raw/04_sentinel2/`（11 文件）— 遥感/夜光

```text
04_sentinel2/
├── aoi_oil_infrastructure.csv
├── aggregate_remote_sensing_to_weekly.py
├── extract_*_gee.js                   # GEE 提取脚本（Sentinel-2 / Landsat / VIIRS）
├── landsat_oil_sites_monthly_indices_2006_2017_{8,11}aoi.csv
├── sentinel2_oil_sites_monthly_indices_2017_2025_{8,11}aoi.csv
└── viirs_oil_sites_monthly_nightlights_2012_2025_{8,11}aoi.csv
```

---

### `raw/05_shipping/`（~104 文件）— 航运/AIS

```text
05_shipping/
├── aggregate_shipping_to_weekly.py
├── GFW/
│   ├── .gfw_token
│   ├── download_gfw_vessel_presence.py
│   └── gfw_chokepoint_vessel_presence_monthly.csv
├── IMF Portwatch/
│   ├── download_portwatch_chokepoints.py
│   └── portwatch_chokepoints_daily.csv
└── emodnet_vessel_density_monthly_2017-2025/
    ├── EMODnet_HA_Vessel_Density_10.zip
    ├── EMODnet_HA_Vessel_DensityMap_20251206_MD.xml
    └── vesseldensity_10_YYYYMMDD.tif   # 96 个月度 GeoTIFF（2017–2024）
```

---

### `raw/06_spatial_nodes/`（8 文件）— 油气基础设施 GIS

```text
06_spatial_nodes/
├── GOGET/Global-Oil-and-Gas-Extraction-Tracker-March-2026.xlsx
├── GOIT/GEM-GOIT-Oil-NGL-Pipelines-2025-03.{xlsx,geojson,gpkg}
├── OGIM/ogim_{core,extended}_infrastructure_nodes_global_20240515.csv
└── world_port_index/NGA_WPI_ports_static_2026.csv + 字段说明 PDF
```

---

## `04_code/` — 代码

```text
04_code/
├── README.md
├── notebooks/                         # 空
├── scripts/
│   ├── .gitkeep
│   ├── build_feature_matrix.py        # 构建特征矩阵
│   ├── run_eda.py                     # 探索性数据分析
│   └── run_ablation_experiments.py    # 消融实验 + SHAP
└── src/
    └── .gitkeep                       # 可复用模块（尚未实现）
```

**说明：** 数据探索 notebook 目前放在 `03_data/`，而非 `04_code/notebooks/`。

---

## `05_outputs/` — 分析输出

```text
05_outputs/
├── figures/
│   ├── .gitkeep
│   ├── 01_brent_price_timeline.png
│   ├── 02_brent_return_distribution.png
│   ├── 03_eia_fundamentals.png
│   ├── 04_macro_indicators.png
│   ├── 05_correlation_heatmap.png
│   ├── 06_remote_sensing_timeseries.png
│   ├── 07_nightlight_timeseries.png
│   ├── 08_modality_coverage.png
│   ├── 09_ablation_results.png
│   ├── 10_shap_feature_importance.png
│   ├── 11_gdelt_disruption_timeseries.png
│   ├── 12_gdelt_transport_timeseries.png
│   ├── 13_shipping_gfw_vessel_hours.png
│   └── 14_shipping_portwatch_transits.png
├── model_results/
│   ├── .gitkeep
│   ├── ablation_results.csv
│   ├── ablation_summary.csv
│   └── shap_feature_importance.csv
├── maps/.gitkeep                      # 空
└── tables/
    ├── .gitkeep
    ├── feature_table_en.md
    └── feature_table_zh.md
```

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
03_data/raw → 03_data/processed → 04_code/scripts → 05_outputs → 06_writing
                                                        ↑
                                              01_literature  00_admin
```


| 主题          | 内容                                                                                                       |
| ----------- | -------------------------------------------------------------------------------------------------------- |
| 研究焦点        | 布伦特油价预测，多模态特征：市场基本面、宏观、官方报告、GDELT 新闻、遥感（Sentinel-2/Landsat/VIIRS）、航运密度（EMODnet/GFW/Portwatch）、油气基础设施 GIS |
| Git 策略      | 原始数据、PDF 文献、大文件（parquet/zip）不入库；代码、processed CSV、图表、写作 Markdown 可跟踪                                      |
| 与 Readme 差异 | 实际 raw 子目录比 Readme 模板更细（01–06 编号）；notebook 在 `03_data/`；原 `phases/`、`supervisor_feedback/` 已移除            |
| 待清理         | `bp_press_release_links.csv` 可移入 `03_data/raw/03_news/`                                                  |


---

## 文件数量汇总


| 目录                               | 约略文件数 | 状态             |
| -------------------------------- | ----- | -------------- |
| `03_data/raw/`                   | ~672  | 本地存在，gitignore |
| `03_data/processed/` + notebooks | ~13   | 部分 tracked     |
| `raw/02_reports/`                | ~321  | raw 内          |
| `raw/03_news/`                   | ~207  | raw 内          |
| `raw/05_shipping/`               | ~104  | raw 内          |
| `05_outputs/`                    | 23    | tracked        |
| `06_writing/`                    | 7     | tracked        |
| 其余 admin/literature/code/AI      | ~30   | tracked        |
