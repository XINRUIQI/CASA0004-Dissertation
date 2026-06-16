# Project Plan — Pre-Meeting 02 Snapshot

# 项目方案 — Meeting 02（2026-06-05）之前的最新版本

> **Snapshot date 快照日期:** 2026-06-04 (the day before Meeting 02)
>
> **Source files 来源文件:** Proposal (2026-04-10), research diary, AI conversations (05-22 ~ 05-26), Dataset_Overview.ipynb, processed data scripts, feature_groups.json

---

## Table of Contents 目录

- Part A — Project Overview 项目概览
  1. [Research Title 研究题目](#1-research-title-研究题目)
  2. [Core Idea 核心思路](#2-core-idea-核心思路)
  3. [Prediction Targets 预测目标](#3-prediction-targets-预测目标)
  4. [Unified Temporal Resolution 统一时间分辨率](#4-unified-temporal-resolution-统一时间分辨率wfri)
  5. [Data Architecture 数据体系](#5-data-architecture-数据体系六大类)
  6. [Feature Matrix & Ablation Framework 特征矩阵与消融框架](#6-feature-matrix--ablation-framework-特征矩阵与消融框架)
  7. [Methodology Pipeline 方法论流程](#7-methodology-pipeline-方法论流程)
  8. [Completed Work Summary 已完成工作汇总](#8-completed-work-summary-已完成工作汇总)
- Part B — Modelling Plan 建模方案
  9. [Model Candidates per Layer 各层候选模型](#9-model-candidates-per-layer-各层候选模型)
  10. [Model Comparison Summary 模型对比总结](#10-model-comparison-summary-模型对比总结)
  11. [Evaluation Framework 评估框架](#11-evaluation-framework-评估框架)
  12. [Practical Implementation Plan 实施计划](#12-practical-implementation-plan-实施计划)
  13. [Literature References for Model Selection 模型选择文献依据](#13-literature-references-for-model-selection-模型选择文献依据)

---

# Part A — Project Overview 项目概览

---

## 1. Research Title 研究题目

**Multimodal Forecasting of Brent Crude Oil: Daily Inputs for Next-Week Return Direction and Realised Volatility**

多模态布伦特原油预测：日频输入预测下周收益率方向与已实现波动率

---

## 2. Core Idea 核心思路

Not treating Brent crude oil as a univariate financial time series, but viewing it as the outcome of a global oil supply-chain system — from production regions, export terminals, shipping routes, storage facilities, refining centres to end-use markets.

不把 Brent 原油价格单纯当作一维金融时间序列来预测，而是把它视为一个全球石油供应链系统的产出——从产油区、出口终端、航运通道、储运设施、炼厂到终端市场。

The study combines four modalities:
研究结合四类模态：

1. **Financial / macro indicators** 金融/宏观指标
2. **Textual event signals** (GDELT, OPEC/EIA reports, company announcements) 文本事件信号
3. **Remote sensing** (optical indices + night-time lights) 遥感（光学指数 + 夜间灯光）
4. **Shipping activity** (chokepoint transits + vessel presence) 航运活动

---

## 3. Prediction Targets 预测目标

| Target 目标变量 | Task 任务类型 | Definition 定义 |
|---|---|---|
| `target_brent_direction_next_1w` | 3-class classification 三分类 | 1 = up (>+0.5%), 0 = flat (±0.5%), −1 = down (<−0.5%) |
| `target_brent_vol_next_1w` | Regression 回归 | 下周 5 个交易日对数收益率标准差 |
| `target_brent_price_next_1w` | Regression 回归 | 下一周五收盘价 |

---

## 4. Unified Temporal Resolution 统一时间分辨率：W-FRI

### 4.1 Decision 决定

All feature modalities are aligned to a **Friday-ending weekly frequency** (`W-FRI` in pandas).

所有特征模态统一对齐到**周五截止的周频**。

- **Study period 研究期间:** 2006-01-06 ~ 2025-12-26
- **Sample size 样本量:** ~1,043 weeks 周

### 4.2 Data-Driven Justification 数据依据

#### Primary reason 主要原因

**EIA Weekly Petroleum Status Report (WPSR) statistical cycle**
**EIA 周度石油状况报告的统计周期**

EIA WPSR — the source of 10 core US petroleum fundamental variables — uses a **Friday-to-Friday statistical week**. Although the report is published on Wednesday, the data collection period ends on Friday. This makes Friday the natural anchor point for weekly aggregation.

EIA WPSR — 10 个核心美国石油基本面变量的数据源 — 采用**周五至周五**的统计周期。虽然报告在周三发布，但数据采集截止日为周五，因此周五是周频聚合的自然锚点。

Implementation in `build_weekly_time_index.py`:

```python
def align_weekly_to_friday(df: pd.DataFrame) -> pd.DataFrame:
    """Align weekly EIA data (various weekdays) to nearest Friday."""
    df = df.copy()
    df.index = df.index + pd.to_timedelta((4 - df.index.dayofweek) % 7, unit="D")
    return df[~df.index.duplicated(keep="last")]
```

#### Secondary reasons 次要原因

| Data source 数据源 | Original freq. 原始频率 | Alignment method 对齐方法 | Rationale 原因 |
|---|---|---|---|
| Brent / WTI price | Daily (business days) | `resample("W-FRI").last()` | 周五收盘价是一周最后一个交易价格 |
| FRED macro (VIX, DXY, DGS10, DFF) | Daily (business days) | `resample("W-FRI").last()` | 与价格一致；捕捉周末前状态 |
| S&P 500 | Daily (business days) | `resample("W-FRI").last()` | 同上 |
| EIA WPSR (10 series) | Weekly (various weekdays) | `align_weekly_to_friday()` | **EIA 统计周以周五结束** |
| GDELT events | Daily / 15-min (v2.0) | `resample("W-FRI").sum()` / `.mean()` | 按周五截止周聚合事件 |
| Remote sensing (S2 / Landsat / VIIRS) | Monthly | `ffill()` → `resample("W-FRI").last()` | 月频前填充后取周五快照 |
| PortWatch chokepoints | Daily | `resample("W-FRI").sum()` | 按周五截止周累加过境量 |
| GFW vessel presence | Monthly | `ffill()` → `resample("W-FRI").last()` | 同遥感 |

#### Financial market convention 金融市场惯例

Financial markets close on Friday and are shut over the weekend. Friday close represents the final consensus price of the trading week.

金融市场在周五收盘后进入周末休市。周五收盘价代表一周的最终共识价格。

- `target_brent_direction_next_1w` compares **next Friday close vs this Friday close**
- `target_brent_vol_next_1w` computes std from the **following Mon–Fri** (5 trading days)
- This ensures clean, non-overlapping weekly prediction windows

---

## 5. Data Architecture 数据体系：六大类

### 5.1 Market & Financial Data 市场与金融数据

**5.1A Target oil price 目标油价**

- EIA Europe Brent Spot Price FOB (daily → weekly last)

**5.1B Reference oil price 参考油价**

- EIA Cushing, OK WTI Spot Price FOB (daily → weekly last)

**5.1C US petroleum fundamentals 美国石油基本面**

10 series from EIA WPSR:

| # | Variable 变量 | Unit 单位 |
|---|---|---|
| 1 | Commercial crude stocks (excl. SPR) 商业原油库存 | Thousand Barrels |
| 2 | Cushing crude stocks 库欣库存 | Thousand Barrels |
| 3 | Crude production 产量 | Thousand Barrels/Day |
| 4 | Crude imports 进口 | Thousand Barrels/Day |
| 5 | Crude exports 出口 | Thousand Barrels/Day |
| 6 | Refinery crude input 炼厂投入 | Thousand Barrels/Day |
| 7 | Refinery utilisation 开工率 | Percent |
| 8 | Gasoline product supplied 汽油需求 | Thousand Barrels/Day |
| 9 | Distillate fuel oil supplied 馏分油需求 | Thousand Barrels/Day |
| 10 | Jet fuel supplied 航煤需求 | Thousand Barrels/Day |

**5.1D Macro-financial control variables 宏观金融控制变量**

| # | Variable 变量 | Source 来源 |
|---|---|---|
| 1 | Nominal Broad U.S. Dollar Index 美元指数 | FRED (DTWEXBGS) |
| 2 | Effective Federal Funds Rate 联邦基金利率 | FRED (DFF) |
| 3 | 10-Year Treasury Yield 10 年期国债收益率 | FRED (DGS10) |
| 4 | CBOE VIX 波动率指数 | FRED (VIXCLS) |
| 5 | S&P 500 标普 500 | Yahoo Finance (^GSPC) |

**Derived features 衍生特征**

- `brent_wti_spread`, `brent_return_pct`, `wti_return_pct`, `brent_log_return`
- `brent_direction` (3-class: 1=up / 0=flat / −1=down)
- `brent_vol_4w`, `brent_vol_12w`
- `crude_stocks_change`, `cushing_stocks_change`, `net_crude_trade`
- `sp500_return_pct`

> **Total M1 features M1 特征总数: 27**

---

### 5.2 Official Reports & Structured Market Text 官方报告与结构化文本

| Dataset 数据集 | Frequency 频率 | Status 状态 |
|---|---|---|
| OPEC Monthly Oil Market Report (MOMR) | Monthly | Source confirmed 来源已确认 |
| EIA STEO Report Text | Monthly | 241 PDFs downloaded 已下载 (2006–2025) |
| OPEC Annual Statistical Bulletin (ASB) | Annual | Source confirmed 来源已确认 |

**Planned use 规划用途:** LLM-based extraction of monthly market sentiment variables (e.g. `momr_supply_sentiment`, `steo_brent_outlook_sentiment`), forward-filled to weekly.

用 LLM 从月度报告中提取市场情绪变量，前向填充至周频。

---

### 5.3 News, Company Announcements & Event Data 新闻与事件数据

| Dataset 数据集 | Status 状态 | Coverage 覆盖 |
|---|---|---|
| GDELT Event Database | Downloaded & calibrated 已下载并校准 | v1.0 (2006 ~ 2015-02) + v2.0 (2015-02 ~ 2025-12) |
| Aramco News & Media | 20 links scraped | Partial 部分 |
| BP Press Releases | Confirmed 已确认 | 2011–2026, ~1,058 articles |
| Shell News & Media | 60 articles scraped | 2025–2026 |
| TotalEnergies | Blocked at page 2 | Partial 部分 |

**GDELT processing 处理流程:**

BigQuery → oil-geopolitical daily + transport-disruption daily → v1.0/v2.0 overlap calibration → `aggregate_gdelt_to_weekly.py` → weekly features.

> **Total M2 features M2 特征总数: 26** (GDELT-derived weekly text/event variables)

---

### 5.4 Remote Sensing (Optical / Night Lights) 遥感数据

**Three data sources 三个数据源:**

| Source 数据源 | Time range 时间 | Extracted indices 提取指标 |
|---|---|---|
| Sentinel-2 | 2017-04 ~ 2025-12 | NDVI, NDWI, NDBI, BSI (mean + std) + valid obs count |
| Landsat (L5/L7/L8) | 2006-01 ~ 2017-03 | NDVI, NDWI, NDBI, BSI (mean only) + valid obs count |
| VIIRS DNB Monthly | 2014-01 ~ 2025-12 | NTL avg_rad mean/max/stddev + cloud-free obs count |

**11 AOI sites 11 个石油基础设施 AOI:**

| ID | Site 站点 | Type 类型 | Country 国家 | Supply-chain role 供应链角色 |
|---|---|---|---|---|
| P001 | Rotterdam 鹿特丹 | Port 港口 | Netherlands | Brent 定价枢纽 / 欧洲需求中心 |
| P002 | Fujairah 富查伊拉 | Terminal 码头 | UAE | 霍尔木兹出口侧 / ADCOP bypass |
| P003 | Ras Tanura 拉斯塔努拉 | Terminal 码头 | Saudi Arabia | 全球最大原油出口终端 |
| P004 | Jurong Island 裕廊岛 | Refinery 炼厂 | Singapore | 马六甲节点 / 东南亚最大炼厂 |
| P005 | Houston 休斯顿 | Port 港口 | USA | 美国 #1 水运港 |
| P006 | Ningbo-Zhoushan 宁波舟山 | Port 港口 | China | 全球 #1 货物吞吐量港口 |
| P007 | Jamnagar 贾姆纳格尔 | Refinery 炼厂 | India | 全球 #1 单体炼厂 (1.24M bpd) |
| P008 | Basra 巴士拉 | Terminal 码头 | Iraq | 伊拉克唯一原油出口通道 |
| P009 | Ulsan 蔚山 | Refinery 炼厂 | South Korea | 全球 #3 炼厂 (840k bpd) |
| P010 | Kharg Island 哈格岛 | Terminal 码头 | Iran | 伊朗 90–96% 原油出口通道 |
| P011 | Yanbu 延布 | Terminal 码头 | Saudi Arabia | Petroline 终点 / 霍尔木兹 bypass |

**Site selection logic 选站逻辑:**

OPEC top-3 export terminals + 2 chokepoint transit hubs + 5 largest demand-side ports/refineries.

**Processing pipeline 处理流程:**

GEE scripts → monthly CSV → `aggregate_remote_sensing_to_weekly.py` → Landsat+S2 concat → optical+VIIRS merge → long→wide → monthly→weekly (ffill + W-FRI last)

> **Total M3 features M3 特征总数: ~110** (10 feature types × 11 AOIs)

---

### 5.5 Shipping & Port Activity 航运与港口活动

**6 oil-critical chokepoints 6 条石油关键咽喉航道:**

| Chokepoint 航道 | 2024 oil flow 流量 | % global 占比 | Linked AOIs 关联站点 |
|---|---|---|---|
| Strait of Hormuz 霍尔木兹 | 20.7M bpd | ~26% | P002, P003, P008, P010 |
| Strait of Malacca 马六甲 | 22.5M bpd | ~28% | P004, P006, P009 |
| Suez Canal 苏伊士 | 4.8M bpd | ~6% | P011, P001 |
| Bab el-Mandeb 曼德海峡 | 4.1M bpd | ~5% | P011 |
| Panama Canal 巴拿马 | 2.0M bpd | ~2.5% | P005 |
| Cape of Good Hope 好望角 | 9.3M bpd | ~12% | P001 |

**Combined coverage 合计覆盖:** ~80% of global seaborne oil trade (EIA 2025)

**Data sources 数据来源:**

| Source 来源 | Period 时段 | Frequency 频率 | Script 脚本 |
|---|---|---|---|
| IMF PortWatch | 2019–2026 | Daily → weekly sum | `download_portwatch_chokepoints.py` |
| GFW 4Wings API | 2012–2018 | Monthly → ffill → weekly | `download_gfw_vessel_presence.py` |

Merged via `aggregate_shipping_to_weekly.py`.

> **Total M4 features M4 特征总数: ~100** (GFW 49 + PortWatch 51)

---

### 5.6 Supply-Chain Spatial Nodes 供应链空间节点

| Dataset 数据集 | Type 类型 | Planned use 规划用途 |
|---|---|---|
| NGA World Port Index (2026) | Static 静态 | Port / terminal node definitions |
| OGIM (Global Oil Infrastructure Mapping) | Static 静态 | 147,445 facility records |
| GOGET (Oil & Gas Extraction Tracker) | Static 静态 | Upstream field / project nodes |
| GOIT (Oil Infrastructure Tracker) | Static 静态 | Pipeline edge data |

**Planned use 规划用途:** Graph structure for potential ST-GNN modelling (not yet implemented).

为 ST-GNN 建模提供图结构（尚未实施）。

---

## 6. Feature Matrix & Ablation Framework 特征矩阵与消融框架

All processed weekly features merged via `build_feature_matrix.py` → `weekly_features.csv`.

Variable groups defined in `feature_groups.json`:

| Layer 层 | Modalities 模态 | New features 新增 | Cumulative 累计 |
|---|---|---|---|
| **M1** | Market + Macro 市场+宏观 | 27 | 27 |
| **M2** | + Text / GDELT 文本/事件 | +26 | 53 |
| **M3** | + Remote Sensing 遥感 | +110 | 163 |
| **M4** | + Shipping 航运 | +100 | 263 |

---

## 7. Methodology Pipeline 方法论流程

From Proposal (2026-04-10), 7-step design:

| Step 步骤 | Description 描述 | Status 状态 |
|---|---|---|
| 1 | Data collection, cleaning, temporal alignment 数据采集、清洗、时间对齐 | ✅ Mostly complete 基本完成 |
| 2 | Supply-chain node identification & graph construction 供应链节点识别与图构建 | ⬜ Static data downloaded; graph not built 静态数据已下载，图未构建 |
| 3 | Proxy construction & feature engineering 代理变量构造与特征工程 | ✅ Scripts complete for all 4 modalities 4 个模态脚本均已完成 |
| 4 | Exploratory analysis & lead-lag testing 探索性分析与领先-滞后检验 | 🔶 Partial (initial XGBoost + EDA) 部分（初步 XGBoost + EDA） |
| 5 | Baseline ML modelling 基线 ML 建模 | 🔶 Initial XGBoost (~42% accuracy) 初步 XGBoost |
| 6 | Multimodal framework (Vision encoder, LLM, ST-GNN) 多模态框架 | ⬜ Not yet implemented 尚未实施 |
| 7 | Evaluation, interpretation & robustness (SHAP, ablation) 评估与稳健性分析 | 🔶 SHAP attempted on initial model 初步 SHAP |

---

## 8. Completed Work Summary 已完成工作汇总

**Infrastructure 基础设施:**

- [x] Project folder structure 项目文件夹结构
- [x] Research diary, prompt log, meeting note templates 研究日记、提示词日志、会议笔记模板
- [x] Complete dataset inventory table (`Dataset_Overview.ipynb`) 完整数据集清单表

**Data acquisition 数据采集:**

- [x] Market & financial: EIA Brent/WTI, EIA WPSR (10 series), FRED (5 series), Yahoo S&P 500
- [x] Text / events: GDELT via BigQuery (83 CSV shards), EIA STEO PDFs (241), partial company news
- [x] Remote sensing: 3 GEE scripts → Sentinel-2 / Landsat / VIIRS for 11 AOIs
- [x] Shipping: PortWatch (6 chokepoints, 2019+), GFW (6 chokepoints, 2012–2018)
- [x] Spatial nodes: OGIM, GOGET, GOIT, NGA WPI

**Data processing 数据处理:**

- [x] `build_weekly_time_index.py` — market + macro → weekly
- [x] `aggregate_gdelt_to_weekly.py` — GDELT events → weekly (with v1.0/v2.0 calibration)
- [x] `aggregate_remote_sensing_to_weekly.py` — RS monthly → weekly
- [x] `aggregate_shipping_to_weekly.py` — shipping daily/monthly → weekly
- [x] `build_feature_matrix.py` — merge all → unified matrix (1,043 weeks × 263 features)
- [x] `feature_groups.json` — 4-layer ablation variable groups

**Modelling (initial) 建模（初步）:**

- [x] XGBoost on partial features (~42% accuracy)
- [x] SHAP feature importance analysis
- [x] EDA visualisations

**Literature 文献:**

- [x] Initial collection: 51 papers (36 relevant, 3 borderline, 12 excluded)

---

# Part B — Modelling Plan 建模方案

> This is the modelling plan as envisioned before Meeting 02. It includes all 4 modalities (M1–M4), including the text/GDELT modality that was later removed.
>
> 此为 Meeting 02 之前的建模方案，包含全部 4 个模态（M1–M4），包括后来被移除的文本/GDELT 模态。

---

## 9. Model Candidates per Layer 各层候选模型

### 9.1 M1 — Market + Macro Baseline 金融基线

**Input 输入:** 27 features (Brent/WTI prices, returns, volatility, EIA fundamentals, FRED macro-financial)

| Model 模型 | Task 任务 | Role 角色 | Rationale 选择理由 |
|---|---|---|---|
| **Logistic Regression** | Direction (3-class) | Simplest baseline 最简基线 | Linear benchmark; if non-linear models do not beat this, the non-linearity is unjustified. 线性基准；如果非线性模型无法超越，则非线性不成立 |
| **Ridge Regression** | Volatility / Price | Simplest baseline 最简基线 | Regularised linear regression for continuous targets 正则化线性回归 |
| **XGBoost Classifier** | Direction (3-class) | Primary tree model 主树模型 | Most widely used in oil price ML literature (P001, P003, P004); handles tabular data well; built-in feature importance; already attempted in initial modelling (~42% accuracy). 油价预测 ML 文献中最常用 |
| **XGBoost Regressor** | Volatility / Price | Primary tree model | Same as above |
| **Random Forest** | Both | Secondary tree model 辅助树模型 | Ensemble benchmark from P003 (multi-model comparison) 集成基准 |
| **SVR / SVM** | Both | Classical ML baseline 经典 ML 基线 | Used in P003 for oil price direction 在 P003 中用于油价方向预测 |

**Recommended pipeline M1 推荐管线:**

```
Direction task 方向任务:
  Logistic Regression (baseline) → XGBoost (main) → Random Forest (secondary)

Volatility task 波动率任务:
  Ridge Regression (baseline) → XGBoost (main)
```

**Why not deep learning for M1 为什么 M1 不用深度学习:**

M1 has only 27 features and ~1,043 weekly samples. Deep learning models (LSTM, Transformer) tend to overfit with this sample size and low-dimensional input.

M1 仅有 27 个特征和约 1,043 个周样本。深度学习模型在此样本量和低维输入下容易过拟合。

---

### 9.2 M2 — M1 + Text/GDELT Events 加文本/事件信号

**Input 输入:** 53 features (M1 27 + GDELT 26)

| Model 模型 | Task | Role | Rationale |
|---|---|---|---|
| **XGBoost** | Both | Main model (same as M1) 主模型 | Must keep the same model to isolate the effect of adding text features. 必须使用相同模型以隔离文本特征的效果 |
| **Logistic / Ridge** | Both | Linear baseline | Same architecture as M1 for fair comparison 与 M1 相同架构 |
| **Random Forest** | Both | Secondary | Same as M1 |
| **LSTM** | Direction | Optional sequence model 可选序列模型 | GDELT features have temporal dynamics (event spikes, tone trends); LSTM can capture sequential dependencies. P004 reference. GDELT 特征有时间动态；LSTM 可捕捉周序列依赖 |

**M2-specific considerations M2 特殊考虑:**

- GDELT event counts are **noisy** — may need smoothing (4-week MA already included) GDELT 事件计数噪声较大
- GDELT v1.0 → v2.0 calibration introduces a structural break risk at 2015-02 校准引入结构性断裂风险
- If GDELT features add no predictive power, this supports dropping text data 如果无预测力则支持移除文本数据

**Planned LLM pipeline (not yet implemented) 规划中的 LLM 管线（尚未实施）:**

```
OPEC MOMR (monthly PDF)  ──┐
                            ├→ LLM extraction → monthly sentiment variables
EIA STEO (monthly PDF)   ──┘    (e.g. momr_supply_sentiment,
                                 steo_brent_outlook_sentiment)
                                      ↓
                              forward-fill to weekly → merge into M2

Candidate LLMs 候选 LLM:
  - FinBERT (domain-specific, efficient)
  - GPT-4 / Claude (zero-shot for complex reports)
  - Fine-tuned BERT on energy-market text

Variables to extract 待提取变量:
  - momr_supply_sentiment, momr_demand_revision_signal
  - momr_refinery_margin_signal, momr_tanker_disruption_signal
  - momr_inventory_tightness, momr_geopolitical_risk_flag
  - steo_brent_outlook_sentiment, steo_inventory_build_signal
  - steo_opec_policy_support_flag, steo_supply_growth_signal
```

---

### 9.3 M3 — M2 + Remote Sensing 加遥感数据

**Input 输入:** 163 features (M2 53 + RS 110)

| Model | Task | Role | Rationale |
|---|---|---|---|
| **XGBoost** | Both | Main model (same) | Consistency for fair ablation comparison 一致性以保证公平消融比较 |
| **Logistic / Ridge** | Both | Linear baseline | Same |
| **Random Forest** | Both | Secondary | Same |
| **XGBoost + PCA** | Both | Variant 变体 | RS features (11 AOIs × multiple indices) are highly correlated; PCA → 2–5 PCs. 遥感特征高度相关；PCA 压缩 |
| **LSTM / GRU** | Direction | Optional | RS features are slow-moving (monthly ffill); sequence models may capture gradual trends. 遥感慢变化；序列模型可捕捉渐进趋势 |

**M3-specific considerations M3 特殊考虑:**

- 110 RS features on ~1,043 samples → high risk of **curse of dimensionality**; PCA strongly recommended 维度灾难风险高，强烈建议 PCA
- RS features are monthly forward-filled → many consecutive weeks have identical values 月频前填充导致连续周值相同
- Cloud-cover missing values (P004 Singapore only 64.2% coverage) need handling 云覆盖缺失值需处理

**PCA pipeline PCA 管线:**

```
110 RS features
  ↓
PCA within subgroups:
  - Optical indices: {NDVI, NDWI, NDBI, BSI} × 11 AOIs → 44 features → 3–5 PCs
  - NTL features: {mean, max, stddev, cf_cvg} × 11 AOIs → 44 features → 2–3 PCs
  - Metadata: {valid_obs_count, sensor_flag} × 11 → keep or drop
  ↓
~5–8 RS principal components + M2 features → XGBoost
```

---

### 9.4 M4 — Full Multimodal (M3 + Shipping) 全模态

**Input 输入:** 263 features (M3 163 + Shipping 100)

| Model | Task | Role | Rationale |
|---|---|---|---|
| **XGBoost** | Both | Main model | Full multimodal; core ablation model 全模态；核心消融对比模型 |
| **Logistic / Ridge** | Both | Linear baseline | Same |
| **XGBoost + PCA** | Both | Recommended variant 推荐变体 | 263 features / 1,043 samples requires dimensionality reduction 需要降维 |
| **LSTM** | Direction | Optional deep model | PCA-compressed features make LSTM input manageable (~35–40 dims) PCA 压缩后 LSTM 输入可控 |
| **TFT** | Both | Advanced multimodal 进阶多模态 | TFT natively supports static + known + unknown time-varying inputs; outputs interpretable attention weights. P039, P041, P042 references. 原生多类型输入；可解释注意力 |
| **ST-GNN** | Both | Experimental 实验性 | 11 AOI nodes + 6 chokepoint edges; node features = RS; edge features = shipping. Proposal Step 6. 供应链图建模 |

**M4-specific considerations M4 特殊考虑:**

- 263 features / 1,043 samples ≈ 1:4 ratio → severe overfitting risk without PCA 过拟合风险严重
- Shipping features have **split coverage**: GFW (2012–2018) + PortWatch (2019–2025) 航运特征分段覆盖
- M4 ablation answers the core question: does the full multimodal set outperform subsets? M4 消融回答核心问题

**PCA pipeline for M4 M4 的 PCA 管线:**

```
263 features
  ↓
Keep raw: M1 market/macro (27 features) — already low-dimensional
  ↓
PCA: GDELT text/event (26 features) → 3–5 PCs
PCA: RS optical (44 features) → 3–5 PCs
PCA: RS NTL (44 features) → 2–3 PCs
PCA: GFW shipping (49 features) → 2–3 PCs
PCA: PortWatch shipping (51 features) → 2–3 PCs
  ↓
~27 raw + ~13–19 PCs ≈ 40–46 features → XGBoost / LSTM / TFT
```

---

## 10. Model Comparison Summary 模型对比总结

### 10.1 Main pipeline (all layers) 主管线（所有层通用）

| Model | Direction task | Volatility task | Role |
|---|---|---|---|
| Logistic Regression | ✅ | — | Linear classification baseline |
| Ridge Regression | — | ✅ | Linear regression baseline |
| **XGBoost Classifier** | ✅ | — | **Primary model** |
| **XGBoost Regressor** | — | ✅ | **Primary model** |
| Random Forest | ✅ | ✅ | Secondary ensemble benchmark |
| SVM / SVR | ✅ | ✅ | Classical ML reference |

### 10.2 Optional / advanced models 可选/进阶模型

| Model | Layers | Task | Rationale |
|---|---|---|---|
| **LSTM** | M2, M3, M4 | Direction | Captures temporal dependencies 捕捉时间依赖 |
| **GRU** | M2, M3, M4 | Direction | Lighter LSTM alternative LSTM 的轻量替代 |
| **TFT** | M4 | Both | Best for multimodal; interpretable attention 最适合多模态 |
| **CNN-1D** | M3, M4 | Both | Local temporal pattern extraction 局部时间模式提取 |
| **MLP** | All | Both | Simple neural network baseline 简单神经网络基线 |
| **ST-GNN** | M4 | Both | Experimental: supply-chain graph 实验性：供应链图 |

### 10.3 LLM-based components (text modality) LLM 组件（文本模态）

| Component | Layer | Input | Output |
|---|---|---|---|
| **FinBERT** / fine-tuned BERT | M2 preprocessing | OPEC MOMR / EIA STEO text | Monthly sentiment scores → weekly |
| **GPT-4 / Claude** (zero-shot) | M2 preprocessing | Same | Structured variable extraction |

These are **preprocessing** components, not prediction models. Their outputs become features in M2.

这些是**预处理**组件，不是预测模型。输出成为 M2 的特征。

---

## 11. Evaluation Framework 评估框架

### 11.1 Direction prediction (3-class) 方向预测

| Metric | Description |
|---|---|
| Overall accuracy | Correct predictions / total |
| Macro F1-score | Average F1 across 3 classes (handles class imbalance) |
| Per-class precision / recall | Especially for "up" and "down" classes |
| Confusion matrix | 3×3 matrix showing misclassification patterns |
| Directional accuracy (DA) | Correct direction predictions (excluding "flat") |

### 11.2 Volatility / price regression 波动率/价格回归

| Metric | Description |
|---|---|
| RMSE | Root mean squared error |
| MAE | Mean absolute error |
| R² | Coefficient of determination |
| QLIKE | Quasi-likelihood loss (volatility-specific) |

### 11.3 Cross-layer comparison 跨层比较

| Comparison | What it tests 检验内容 |
|---|---|
| M2 vs M1 | 加入 GDELT 文本/事件是否改善预测？ |
| M3 vs M1 | 加入遥感是否改善？ |
| M3 vs M2 | 遥感在文本之上是否有额外价值？ |
| M4 vs M3 | 加入航运是否进一步改善？ |
| M4 vs M1 | 全模态 vs 仅金融 |

### 11.4 Interpretation tools 解释工具

| Tool | Use |
|---|---|
| **SHAP** | Feature importance; per-prediction explanations; modality contribution |
| **Ablation analysis** | M1 → M4 progressive comparison |
| **PCA loading analysis** | Which raw variables drive each principal component |
| **TFT attention weights** | Which time steps and features the model attends to (M4 only) |
| **Sub-period analysis** | Crisis periods (COVID, Red Sea, Hormuz) vs calm periods |

---

## 12. Practical Implementation Plan 实施计划

### Phase 1: Baseline (M1)
1. Train Logistic Regression, Ridge, XGBoost, Random Forest, SVM on M1 (27 vars)
2. Evaluate on held-out test set (e.g. last 2 years: 2024–2025)
3. SHAP on XGBoost → identify top financial features
4. Establish baseline metrics

### Phase 2: Text layer (M2)
1. Merge GDELT features into M1
2. (Planned) LLM extraction on OPEC MOMR / EIA STEO → monthly sentiment features
3. Train same model suite on M2 (53 vars)
4. Compare M2 vs M1
5. If GDELT adds noise → consider dropping or smoothing

### Phase 3: Remote sensing layer (M3)
1. Merge RS features; apply PCA (110 → 5–8 PCs)
2. Train same model suite + optional LSTM
3. Compare M3 vs M2 vs M1
4. SHAP: which RS features / AOIs contribute most?

### Phase 4: Shipping layer (M4)
1. Merge shipping features; apply PCA (100 → 5–8 PCs)
2. Train same model suite + optional TFT
3. Full M1–M4 comparison table
4. (Experimental) ST-GNN with supply-chain graph

### Phase 5: Analysis & writing
1. Comprehensive evaluation table (all models × all layers × all metrics)
2. SHAP analysis across layers
3. Sub-period robustness analysis
4. Write Results, Discussion, Conclusion chapters

---

## 13. Literature References for Model Selection 模型选择文献依据

| Model | Paper ID | Reference | Relevance |
|---|---|---|---|
| XGBoost | P001, P003, P004 | Foroutan & Lahmiri (2024); multi-model comparison (2025); LSTM+XGBoost (2024) | Most cited ML model for oil price forecasting |
| LSTM | P004, P006 | LSTM+XGBoost hybrid (2024); MLP/CNN/Transformer comparison (2024) | Sequence modelling for temporal dynamics |
| Random Forest | P003 | Multi-model comparison (2025) | Ensemble baseline |
| SVM / SVR | P003 | Multi-model comparison (2025) | Classical ML reference |
| MLP / CNN | P006 | MLP/CNN/Transformer comparison (2024) | Neural network baselines |
| TFT / Transformer | P039, P041, P042 | Multimodal Transformer (2024); multi-faceted attention (2025); Transformer survey (2024) | Multimodal fusion architecture |
| FinBERT / LLM | — | General NLP literature | Text feature extraction (planned, not implemented) |
| ST-GNN | — | Proposal Step 6 | Experimental supply-chain graph modelling |
