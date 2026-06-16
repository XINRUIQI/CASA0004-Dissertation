# Project Plan — Post-Meeting 02

# 项目方案 — Meeting 02（2026-06-05）之后的最新版本

> **Last updated 最后更新:** 2026-06-09
>
> **Changes from pre-Meeting 02 与 Meeting 02 前的主要变动:**
> 1. Research question refocused: "does multimodal data improve oil price prediction?" 研究问题重新聚焦
> 2. Text/GDELT/LLM modality removed 文本/GDELT/LLM 模态移除
> 3. Ablation framework restructured: M1(Fin) → M2(Fin+RS) → M3(Fin+Ship) → M4(All) 消融框架重构
> 4. All models must run M1–M4 for fair comparison 所有模型必须跑完 M1–M4
> 5. GNN papers (G1–G6) added to literature 新增 GNN 文献

---

## Table of Contents 目录

- Part A — Project Overview 项目概览
  1. [Research Title 研究题目](#1-research-title-研究题目)
  2. [Core Idea 核心思路](#2-core-idea-核心思路)
  3. [Prediction Targets 预测目标](#3-prediction-targets-预测目标)
  4. [Unified Temporal Resolution 统一时间分辨率](#4-unified-temporal-resolution-统一时间分辨率wfri)
  5. [Data Architecture 数据体系](#5-data-architecture-数据体系三大类)
  6. [Feature Matrix & Ablation Framework 特征矩阵与消融框架](#6-feature-matrix--ablation-framework-特征矩阵与消融框架)
  7. [Methodology Pipeline 方法论流程](#7-methodology-pipeline-方法论流程)
  8. [Completed Work Summary 已完成工作汇总](#8-completed-work-summary-已完成工作汇总)
- Part B — Modelling Plan 建模方案
  9. [Two-Dimensional Comparison Design 二维对比设计](#9-two-dimensional-comparison-design-二维对比设计)
  10. [Model Architectures 模型架构](#10-model-architectures-模型架构)
  11. [Evaluation Framework 评估框架](#11-evaluation-framework-评估框架)
  12. [Practical Implementation Plan 实施计划](#12-practical-implementation-plan-实施计划)
  13. [Literature References 文献依据](#13-literature-references-文献依据)

---

# Part A — Project Overview 项目概览

---

## 1. Research Title 研究题目

**Can Multimodal Remote Sensing and Shipping Data Improve Brent Crude Oil Price Prediction?**

多模态遥感与航运数据能否改善布伦特原油价格预测？

---

## 2. Core Idea 核心思路

> **Old framing 旧定位:** "How to accurately predict oil prices?" — build the most accurate model
>
> **New framing 新定位:** "Does incorporating remote sensing and shipping data improve oil price prediction compared to financial data alone?" — systematic comparison of data modalities

The study does **not** aim to build a trading model or achieve the highest possible accuracy. Instead, it investigates the **marginal contribution** of different data modalities to prediction performance via a progressive ablation framework.

本研究**不**以构建交易模型或追求最高准确率为目标，而是通过渐进式消融框架，研究不同数据模态对预测性能的**边际贡献**。

Three data modalities are retained:
保留三类数据模态：

1. **Financial / market data** 金融/市场数据
2. **Remote sensing data** 遥感数据（NTL、光学指数）
3. **Shipping activity data** 航运活动数据（AIS、咽喉航道）

~~Text / NLP / LLM~~ — removed per Meeting 02 to reduce complexity.
~~文本 / NLP / LLM~~ — 根据 Meeting 02 决策移除以降低复杂度。

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

EIA WPSR — the source of core US petroleum fundamental variables — uses a **Friday-to-Friday statistical week**. Friday is the natural anchor point for weekly aggregation.

#### Alignment methods 各数据源对齐方法

| Data source 数据源 | Original freq. 原始频率 | Alignment method 对齐方法 |
|---|---|---|
| Brent / WTI price | Daily (business days) | `resample("W-FRI").last()` |
| FRED macro (VIX, DXY, DGS10, DFF) | Daily (business days) | `resample("W-FRI").last()` |
| S&P 500 | Daily (business days) | `resample("W-FRI").last()` |
| EIA WPSR (10 series) | Weekly (various weekdays) | `align_weekly_to_friday()` |
| Remote sensing (S2 / Landsat / VIIRS) | Monthly | `ffill()` → `resample("W-FRI").last()` |
| PortWatch chokepoints | Daily | `resample("W-FRI").sum()` |
| GFW vessel presence | Monthly | `ffill()` → `resample("W-FRI").last()` |

---

## 5. Data Architecture 数据体系：三大类

> **Change 变动:** Sections 5.2 (Official Reports) and 5.3 (News/GDELT) from pre-Meeting 02 plan have been **removed**. Text/event data is no longer part of the project scope.

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

> **Note 注意:** Variable selection should be further refined based on literature review (Meeting 02 action point). ~200 features is too many; aim for a curated subset justified by literature.
> 变量选择需通过文献综述进一步精简（Meeting 02 行动项）。约 200 个特征过多，应基于文献精选子集。

---

### 5.2 Remote Sensing (Optical / Night Lights) 遥感数据

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

**Processing pipeline 处理流程:**

GEE scripts → monthly CSV → Landsat+S2 concat → optical+VIIRS merge → long→wide → monthly→weekly (ffill + W-FRI last)

**PCA consideration PCA 考虑:** 110 RS features on ~1,043 samples → dimensionality may be high. Meeting 02 建议可考虑在各子组内使用 PCA 降维（如 optical indices 一组、NTL 一组），具体 PC 数量待建模阶段确定。

> **Total RS features (GEE-derived) 遥感特征总数（GEE 衍生）: ~110**

#### CV-enhanced RS feature (optional) CV 增强遥感特征（可选亮点）

**Floating Roof Tank (FRT) fill-level estimation 浮顶储油罐液位估算**

| Item | Detail |
|---|---|
| Principle 原理 | 浮顶储油罐的顶盖浮在油面上，油越少顶盖越低，罐壁内侧月牙形阴影越大。通过卫星影像测量阴影面积反推储油量。 |
| Pipeline 技术路线 | YOLOv8 检测储油罐 → HSV/LAB 色彩空间阴影提取 → 阴影面积比 → 液位估算 |
| Target sites 目标站点 | 仅 2–3 个大型站点（Fujairah P002、Cushing、Rotterdam P001），浮顶罐直径 >60m |
| Data 数据 | Sentinel-2 (10m)，大型罐勉强可用；概念验证级别 |
| Output 输出 | `frt_fill_level_{site}` — 月频 → ffill → weekly，作为 RS 特征加入 M2/M4 |
| Effort 工作量 | 5–7 天 |
| Priority 优先级 | **可选**（Tier 1 建模完成后再做；若时间不足放 Future Work） |
| Value 价值 | 极高 — 这是对冲基金（Orbital Insight、Kayrros）的核心业务。EIA 只公布美国库存，Fujairah/Rotterdam 等站点无公开库存数据，卫星估算可提供市场上不存在的独家信息。 |

---

### 5.3 Shipping & Port Activity 航运与港口活动

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

| Source 来源 | Period 时段 | Frequency 频率 |
|---|---|---|
| IMF PortWatch | 2019–2026 | Daily → weekly sum |
| GFW 4Wings API | 2012–2018 | Monthly → ffill → weekly |

**PCA consideration PCA 考虑:** 100 shipping features 可能存在高度相关性。可考虑对 GFW (49) 和 PortWatch (51) 分别做 PCA，具体待建模阶段确定。

> **Total Shipping features (existing) 航运特征总数（现有）: ~100**

#### Satellite-derived shipping features (recommended) 卫星衍生航运特征（推荐）

**A. GFW SAR dark vessel count 暗船计数**

| Item | Detail |
|---|---|
| Principle 原理 | 部分船舶关闭 AIS 以规避制裁或监管（"暗船"），在 AIS 数据中不可见，但在 SAR 卫星影像中可检测。GFW 已提供 SAR 检测数据。 |
| Pipeline 技术路线 | GFW SAR detection data → 按 6 条咽喉航道空间过滤 → 统计无 AIS 匹配的 SAR 检测目标数 |
| Output 输出 | `dark_vessel_count_{chokepoint}` — 6 条航道 × 1 个变量 = 6 个新特征 |
| Effort 工作量 | ~1 天 |
| Priority 优先级 | **推荐**（数据处理任务，非 CV 模型训练） |
| Value 价值 | 中高 — 暗船是 AIS 的盲区，提供了常规 shipping 数据无法覆盖的信号（制裁规避、非法运输） |

**B. Port anchorage congestion 港口锚泊拥堵**

| Item | Detail |
|---|---|
| Principle 原理 | 在 AOI 港口区域统计锚泊/停泊船舶数量，作为港口拥堵程度的代理指标。拥堵高 → 供应链瓶颈 → 可能影响油价。 |
| Pipeline 技术路线 | GFW SAR detection data → 按 11 个 AOI 港口空间过滤 → 统计月度锚泊船舶数 |
| Output 输出 | `port_congestion_{aoi}` — 11 个 AOI × 1 个变量 = 11 个新特征 |
| Effort 工作量 | 2–3 天（可与方案 A 同时做，同源数据） |
| Priority 优先级 | **推荐** |
| Value 价值 | 中 — 港口拥堵是供应链瓶颈的直接代理指标 |

> **Total Shipping features (with satellite-derived) 航运特征总数（含卫星衍生）: ~100 + 6 + 11 = ~117**

---

### 5.4 Supply-Chain Graph Structure 供应链图结构（用于 ST-GNN）

| Component 组件 | Source 来源 | Content 内容 |
|---|---|---|
| **Nodes 节点** (11) | 5.2 AOI sites | 11 个石油基础设施站点 |
| **Edges 边** (6+) | 5.3 chokepoints + known trade routes | 6 条咽喉航道 + 主要贸易航线 |
| **Node features** | RS data (NTL, optical indices) per AOI | 每个节点的遥感时序特征 |
| **Edge features** | Shipping data (transit volume, vessel count) | 每条边的航运时序特征 |
| **Global features** | Financial/market data (M1 variables) | 全局金融变量 |

**Static data sources for graph construction 图构建静态数据:**

| Dataset 数据集 | Type 类型 | Use 用途 |
|---|---|---|
| NGA World Port Index (2026) | Static 静态 | Port / terminal node definitions |
| OGIM (Global Oil Infrastructure Mapping) | Static 静态 | 147,445 facility records |
| GOGET (Oil & Gas Extraction Tracker) | Static 静态 | Upstream field / project nodes |
| GOIT (Oil Infrastructure Tracker) | Static 静态 | Pipeline edge data |

> **Status 状态:** Static data downloaded; graph not yet implemented. Graph construction approach to be justified by literature (G1 LGCOTFF, G5 STMGCN).
> 静态数据已下载；图尚未实现。图构建方法需通过文献论证。

---

## 6. Feature Matrix & Ablation Framework 特征矩阵与消融框架

> **Change 变动:** Text/GDELT modality removed. M2 is now RS, M3 is now Shipping.

| Layer 层 | Data 数据 | Features 特征数 |
|---|---|---|
| **M1** | Financial / Market 金融/市场 | 27 |
| **M2** | M1 + Remote Sensing 遥感 | 27 + ~110 = ~137 (+ FRT fill-level if available) |
| **M3** | M1 + Shipping 航运 | 27 + ~117 = ~144 (含 SAR 暗船 + 港口拥堵) |
| **M4** | M1 + RS + Shipping 全部 | 27 + ~110 + ~117 = ~254 (+ FRT if available) |

> **Note 注意:** PCA 可作为降维选项（Meeting 02 建议），但尚未实施。具体 PC 数量需在实际建模阶段根据解释方差比确定。

---

## 7. Methodology Pipeline 方法论流程

| Step 步骤 | Description 描述 | Status 状态 |
|---|---|---|
| 1 | Data collection, cleaning, temporal alignment 数据采集、清洗、时间对齐 | ✅ Mostly complete |
| 2 | Supply-chain node identification & graph construction 供应链节点识别与图构建 | ⬜ Static data downloaded; graph not built |
| 3 | Proxy construction & feature engineering 代理变量构造与特征工程 | ✅ Scripts complete for Financial, RS, Shipping |
| 4 | **Literature review & variable selection** 文献综述与变量筛选 | 🔶 In progress (3-week sprint) |
| 5 | Baseline ML modelling (XGBoost × M1–M4) 基线 ML 建模 | 🔶 Initial XGBoost (~42% accuracy on old M1) |
| 6 | Extended modelling (LSTM × M1–M4, ST-GNN × M1–M4) 扩展建模 | ⬜ Not yet implemented |
| 7 | Evaluation, interpretation & robustness (SHAP, ablation) 评估与稳健性分析 | ⬜ Not yet started |

> **Change 变动:** Step 4 added (literature-driven variable selection, per Meeting 02). Step 6 simplified: Vision Encoder and LLM removed; ST-GNN retained.

---

## 8. Completed Work Summary 已完成工作汇总

**Infrastructure 基础设施:**

- [x] Project folder structure
- [x] Research diary, meeting note templates
- [x] Complete dataset inventory table (`Dataset_Overview.ipynb`)

**Data acquisition 数据采集:**

- [x] Market & financial: EIA Brent/WTI, EIA WPSR (10 series), FRED (5 series), Yahoo S&P 500
- [x] ~~Text / events: GDELT, EIA STEO PDFs~~ — **removed from scope**
- [x] Remote sensing: 3 GEE scripts → Sentinel-2 / Landsat / VIIRS for 11 AOIs
- [x] Shipping: PortWatch (6 chokepoints, 2019+), GFW (6 chokepoints, 2012–2018)
- [x] Spatial nodes: OGIM, GOGET, GOIT, NGA WPI

**Data processing 数据处理:**

- [x] `build_weekly_time_index.py` — market + macro → weekly
- [x] ~~`aggregate_gdelt_to_weekly.py`~~ — **removed from scope**
- [x] `aggregate_remote_sensing_to_weekly.py` — RS monthly → weekly
- [x] `aggregate_shipping_to_weekly.py` — shipping daily/monthly → weekly
- [x] `build_feature_matrix.py` — merge all → unified matrix
- [x] `feature_groups.json` — needs updating to new M1–M4 structure

**Modelling (initial) 建模（初步）:**

- [x] XGBoost on partial features (~42% accuracy)
- [x] SHAP feature importance analysis
- [x] EDA visualisations

**Literature 文献:**

- [x] Initial collection: 57 papers (47 relevant, 4 borderline, 8 excluded)
- [ ] Structured literature review (3-week sprint — in progress)

---

# Part B — Modelling Plan 建模方案

---

## 9. Two-Dimensional Comparison Design 二维对比设计

The core experiment uses a **2D comparison matrix**: varying both data modality (columns) and model architecture (rows). Each model must run across **all four data configurations** M1–M4 to ensure fair comparison.

核心实验采用**二维对比矩阵**：同时变化数据模态（列）和模型架构（行）。每个模型必须在 **M1–M4 全部四种数据配置**上运行，以确保公平对比。

### 9.1 Comparison matrix 对比矩阵

```
                   M1(Fin)    M2(Fin+RS)    M3(Fin+Ship)    M4(All)
Baselines            ✓            ✓              ✓             ✓
XGBoost (必做)        ✓            ✓              ✓             ✓
LSTM    (推荐)        ✓            ✓              ✓             ✓
TFT     (推荐)        ✓            ✓              ✓             ✓
ST-GNN  (进阶)        ✓            ✓              ✓             ✓
```

### 9.2 What each comparison direction answers 每个对比方向回答什么

| Direction 方向 | How 怎么看 | Research question 研究问题 |
|---|---|---|
| **Horizontal 横向** (same row) | Fix model, vary data | 多模态数据是否改善预测？ |
| **Vertical 纵向** (same column) | Fix data, vary model | 哪种模型最适合这类数据？ |
| **Global best 全局最优** | Best cell in entire matrix | 最佳的数据+模型组合是什么？ |

### 9.3 Priority tiers 优先级分层

| Tier | Models × Layers | Est. effort 预计工作量 | Deliverable 交付物 |
|---|---|---|---|
| **Tier 1 必做** | XGBoost × M1–M4 + Baselines × M1–M4 | 2–3 weeks | 回答"多模态是否改善预测" |
| **Tier 2 推荐** | LSTM + TFT × M1–M4 | +2–3 weeks | 回答"时序模型是否更好地利用 RS/Shipping"；TFT 额外提供可解释注意力 |
| **Tier 3 进阶** | ST-GNN × M1–M4 | +2–3 weeks | 回答"图建模是否进一步提升" |

> Tier 1 alone is sufficient for a complete dissertation. Tiers 2–3 add depth and novelty.
> 仅 Tier 1 就足以支撑一篇完整的论文。Tier 2–3 增加深度和创新性。

---

## 10. Model Architectures 模型架构

### 10.1 Baselines 基线模型（对照组）

| Model 模型 | Direction task | Volatility task | Role 角色 |
|---|---|---|---|
| Logistic Regression | ✅ | — | Linear classification baseline |
| Ridge Regression | — | ✅ | Linear regression baseline |
| Random Forest | ✅ | ✅ | Ensemble benchmark |
| SVM / SVR | ✅ | ✅ | Classical ML reference |

### 10.2 XGBoost — Primary model 主模型（必做）

- **Layers 层:** M1–M4 (all)
- **Tasks 任务:** Direction (3-class) + Volatility (regression) + Price (regression)
- **Rationale:** Most widely used in oil price ML literature (P001, P003, P004); handles tabular data well; built-in feature importance
- **PCA:** Apply to M2/M3/M4 to control dimensionality

### 10.3 LSTM — Sequence model 序列模型（推荐）

- **Layers 层:** M1–M4 (all)
- **Tasks 任务:** Direction (3-class) + Volatility + Price
- **Rationale:** Captures temporal dependencies in RS and shipping features that XGBoost may miss
- **Input:** Feature vectors → sliding window (e.g. 12-week lookback)
- **References:** P004 (LSTM+XGBoost for WTI), P006 (MLP/CNN/Transformer comparison)

### 10.4 TFT — Temporal Fusion Transformer（推荐）

- **Layers 层:** M1–M4 (all)
- **Tasks 任务:** Direction (3-class) + Volatility + Price
- **Rationale:** 原生支持多类型输入（静态 / 已知未来 / 未知变量分别编码）；内置 Variable Selection Network 自动筛选有用特征；输出可解释注意力权重，天然支持模态贡献分析
- **Input design 输入设计:**
  - Static covariates: AOI type, chokepoint ID（用于 M2/M3/M4 中区分不同站点/航道）
  - Known future: calendar features (week-of-year, month)
  - Unknown observed: financial, RS, shipping features (depending on layer)
- **Implementation 实现:** `pytorch-forecasting` library (built-in TFT class)
- **Risk 风险:** ~1,043 周样本对 Transformer 偏少，需注意过拟合（dropout, early stopping）
- **References:** P039 (Modality-aware Transformer), P041 (EMAT), P042 (Transformer survey)

### 10.5 ST-GNN — Graph model 图模型（进阶）

- **Layers 层:** M1–M4 (all)
- **Tasks 任务:** Direction (3-class) + Volatility + Price
- **Graph structure 图结构:** 11 AOI nodes + 6 chokepoint edges (from Section 5.4)
- **Node features:** RS data per AOI (M2), or RS+financial (M1+M2)
- **Edge features:** Shipping data per chokepoint (M3)
- **Global features:** Financial/market variables (M1)
- **Rationale:** Oil price is influenced by spatial supply-chain dynamics; graph structure captures information propagation across connected infrastructure nodes
- **References:** G1 (LGCOTFF — crude oil maritime network + LSTM-GCN), G2 (GWNet-Attn — ST-GNN for WTI futures), G5 (STMGCN — AIS maritime traffic graph)

**How ST-GNN handles M1–M4 ST-GNN 如何适配 M1–M4:**

| Layer | Node features | Edge features | Global features |
|---|---|---|---|
| M1 | Financial vars (same for all nodes) | None (unweighted edges) | Financial vars |
| M2 | RS per AOI + Financial | None | Financial vars |
| M3 | Financial vars | Shipping per chokepoint | Financial vars |
| M4 | RS per AOI + Financial | Shipping per chokepoint | Financial vars |

### 10.6 Removed models 已移除的模型

| Model | Reason for removal 移除原因 |
|---|---|
| ~~LLM / FinBERT~~ | Text modality removed |
| ~~Vision Encoder (end-to-end)~~ | 不做端到端图像输入；CV 仅用于特征提取（FRT 液位估算、SAR 暗船计数），输出为数值特征 |

---

## 11. Evaluation Framework 评估框架

### 11.1 Direction prediction (3-class) 方向预测

| Metric | Description |
|---|---|
| Overall accuracy | Correct predictions / total |
| Macro F1-score | Average F1 across 3 classes (handles class imbalance) |
| Per-class precision / recall | Especially for "up" and "down" classes |
| Directional accuracy (DA) | Correct direction predictions (excluding "flat") |
| Confusion matrix | 3×3 matrix showing misclassification patterns |

### 11.2 Volatility regression 波动率回归

| Metric | Description |
|---|---|
| RMSE | Root mean squared error |
| MAE | Mean absolute error |
| R² | Coefficient of determination |

### 11.3 Cross-layer comparison (horizontal) 跨层对比（横向）

| Comparison | What it tests 检验内容 |
|---|---|
| M2 vs M1 | 加入遥感数据是否改善预测？ |
| M3 vs M1 | 加入航运数据是否改善预测？ |
| M4 vs M1 | 全模态 vs 仅金融 |
| M4 vs M2 | 在遥感基础上加航运是否有额外价值？ |
| M4 vs M3 | 在航运基础上加遥感是否有额外价值？ |

### 11.4 Cross-model comparison (vertical) 跨模型对比（纵向）

| Comparison | What it tests |
|---|---|
| LSTM vs XGBoost (same layer) | 时序模型是否更好地利用 RS/Shipping？ |
| TFT vs LSTM (same layer) | Transformer 注意力机制是否优于 LSTM？ |
| TFT vs XGBoost (same layer) | 深度时序模型 vs 树模型，哪个更有效？ |
| ST-GNN vs XGBoost (same layer) | 图结构建模是否额外有效？ |
| ST-GNN vs TFT (same layer) | 空间图结构 vs 时间注意力，哪个更重要？ |

### 11.5 Interpretation tools 解释工具

| Tool | Use |
|---|---|
| **SHAP** | Feature importance; per-prediction explanations; modality contribution |
| **Ablation analysis** | M1 → M4 progressive comparison |
| **PCA loading analysis** | Which raw variables drive each principal component |
| **Sub-period analysis** | Crisis periods (COVID, Red Sea, Hormuz) vs calm periods |

---

## 12. Practical Implementation Plan 实施计划

### Phase 0: Literature review & variable selection (3 weeks) 文献综述与变量筛选

> **Current phase 当前阶段** (Week of 2026-06-09)

1. Structured literature review using Meeting 02 template (Paper, Task, Model, Variables, Frequency, Evaluation, Findings)
2. Based on literature, refine financial variables for M1 (aim for curated subset)
3. Identify and justify RS features for M2 from literature
4. Identify and justify shipping variables for M3 from literature
5. Update `feature_groups.json` to new M1–M4 structure
6. Remove text/GDELT from variable table and project plan

### Phase 0.5: Satellite-derived feature engineering (+1–3 days) 卫星衍生特征工程

> 在文献综述期间或之后，穿插完成

1. GFW SAR dark vessel count: 按 6 条咽喉航道空间过滤 → 6 个新 shipping 特征
2. Port anchorage congestion: 按 11 个 AOI 空间过滤 → 11 个新 shipping 特征
3. Merge into shipping feature matrix → update `feature_groups.json`

### Phase 1: Tier 1 — XGBoost + Baselines × M1–M4 (2–3 weeks)

1. Train Logistic/Ridge, XGBoost, RF, SVM on M1 (financial baseline)
2. Train same suite on M2, M3, M4
3. Full M1–M4 horizontal comparison table
4. SHAP analysis on XGBoost → which modalities/features contribute most
5. **Checkpoint: if results are solid here, the dissertation has a complete core.**

### Phase 2: Tier 2 — LSTM + TFT × M1–M4 (2–3 weeks)

1. Implement LSTM with sliding window
2. Train LSTM on M1–M4
3. Implement TFT via `pytorch-forecasting` (configure static / known / unknown inputs)
4. Train TFT on M1–M4
5. Compare LSTM, TFT, XGBoost across all layers
6. TFT attention weights → which time steps and variables matter most per layer

### Phase 3: Tier 3 — ST-GNN × M1–M4 (2–3 weeks, if time permits)

1. Construct supply-chain graph (11 nodes, 6+ edges) — reference G1 LGCOTFF
2. Implement ST-GNN (reference G2 GWNet-Attn architecture)
3. Train on M1–M4
4. Compare with XGBoost and LSTM
5. Does graph structure add value beyond tabular features?

### Phase 3.5: CV enhancement — FRT fill-level estimation (+5–7 days, optional)

> **前提：Tier 1 已完成，有余力时再做**

1. 选定 2–3 个大型站点（Fujairah P002、Rotterdam P001、或 Cushing）
2. 获取 Sentinel-2 影像时间序列
3. YOLOv8 检测浮顶储油罐 → 阴影提取 → 液位估算
4. 输出月频 fill-level → ffill → weekly → 加入 M2/M4 RS 特征
5. 重跑 XGBoost M2/M4，对比加入 FRT 前后的性能变化
6. 若时间不足，在论文中作为 proof-of-concept 概念验证呈现

### Phase 4: Analysis & Writing (3–4 weeks)

1. Comprehensive 2D evaluation table (all models × all layers × all metrics)
2. SHAP analysis across layers and models
3. Sub-period robustness analysis
4. Write Results, Discussion, Conclusion chapters
5. Future Work: discuss potential for full supply-chain graph, text/LLM modality, GNN extensions, FRT scaling to more sites

---

## 13. Literature References 文献依据

### 13.1 Model selection 模型选择

| Model | Paper ID | Reference | Relevance |
|---|---|---|---|
| XGBoost | P001, P003, P004 | Foroutan & Lahmiri (2024); multi-model comparison (2025); LSTM+XGBoost (2024) | Most cited ML model for oil price forecasting |
| LSTM | P004, P006 | LSTM+XGBoost hybrid (2024); MLP/CNN/Transformer comparison (2024) | Sequence modelling for temporal dynamics |
| Random Forest | P003 | Multi-model comparison (2025) | Ensemble baseline |
| SVM / SVR | P003 | Multi-model comparison (2025) | Classical ML reference |
| TFT | P039, P041, P042 | Modality-aware Transformer (2024); EMAT (2025); Transformer survey (2026) | Multimodal fusion with interpretable attention |
| ST-GNN | **G1**, **G2**, G3 | LGCOTFF (2022); GWNet-Attn (2023); BiLSTM-GCN (2023) | GNN for crude oil maritime network and price prediction |

### 13.2 GNN-specific references GNN 专用文献

| ID | Reference | Relevance |
|---|---|---|
| **G1** | LGCOTFF (IEEE Access, 2022) | LSTM+GCN for crude oil maritime traffic flow; graph construction method |
| **G2** | GWNet-Attn (Resources Policy, 2023) | Self-Attention + Graph WaveNet for WTI futures price |
| **G3** | BiLSTM-GCN (Mathematics, 2023) | BiLSTM+GCN for crude oil price forecasting |
| **G4** | Russian Oil GNN (2025) | GNN for oil trade flow forecasting under sanctions |
| **G5** | STMGCN (Southampton) | AIS-based maritime traffic graph + spatio-temporal multigraph CNN |
| **G6** | ITSG-LSTM (Eng. Applications of AI, 2025) | Inter-country trade similarity graph for port throughput |
| P043–P045 | Supply chain GNN papers | Graph construction methodology reference |
| P047–P048 | STGAT / ST-GNN for price prediction | Architecture reference (stock/electricity → transferable to oil) |

### 13.3 Variable selection 变量选择

| Modality | Key papers | What to extract |
|---|---|---|
| Financial | P001, P004 | Input variable lists for oil price ML |
| Shipping | P014, P016, P017 | AIS-derived variables (port-call, dwell time, trade volume) |
| Remote Sensing | P024, P025, P032 | NTL proxy, cloud cover proxy, data source selection |
