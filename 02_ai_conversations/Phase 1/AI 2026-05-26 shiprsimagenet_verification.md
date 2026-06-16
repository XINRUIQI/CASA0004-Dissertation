# ShipRSImageNet Dataset Verification

> **本文件包含：**
> - ShipRSImageNet 数据集验证结果（类别层级、Oil Tanker 类别存在性、时间戳缺失确认）
> - 该数据集对本论文的适用性评估（不适合时间序列、适合模型训练和方法论展示）
> - 全项目"geopolitical"相关词汇替换映射表（替换为 disruption/exogenous 等中性表述）
> - 涉及修改的 16 个文件清单及未修改内容说明

> Supervisor requested verification of ShipRSImageNet categories and timestamp availability.

## Basic Info

- **Source**: https://github.com/zzndream/ShipRSImageNet
- **Paper**: Z. Zhang et al., "ShipRSImageNet: A Large-Scale Fine-Grained Dataset for Ship Detection in High-Resolution Optical Remote Sensing Images", IEEE JSTARS, 2021. DOI: 10.1109/JSTARS.2021.3104230
- **Images**: 3,435 (each ~930×930 px)
- **Annotated instances**: 17,573 ships
- **Annotation types**: Horizontal Bounding Box (HBB), Oriented Bounding Box (OBB), Polygon masks

## Category Hierarchy (4 levels)

- **Level 0**: Ship / Not Ship
- **Level 1**: Category (e.g., Warship, Merchant, etc.)
- **Level 2**: Subcategory
- **Level 3**: 50 specific ship types

### Oil-relevant categories at Level 3

- **Oil Tanker** (directly relevant)
- Container Ship, Cargo, RoRo, Barge (indirectly relevant for trade activity)

### Full 50 Level-3 classes

Other Ship, Other Warship, Submarine, Other Aircraft Carrier, Enterprise, Nimitz, Midway, Ticonderoga, Other Destroyer, Atago DD, Arleigh Burke DD, Hatsuyuki DD, Hyuga DD, Asagiri DD, Other Frigate, Perry FF, Patrol, Other Landing, YuTing LL, YuDeng LL, YuDao LL, YuZhao LL, Austin LL, Osumi LL, Wasp LL, LSD 41 LL, LHA LL, Commander, Other Auxiliary Ship, Medical Ship, Test Ship, Training Ship, AOE, Masyuu AS, Sanantonio AS, EPF, Other Merchant, Container Ship, RoRo, Cargo, Barge, Tugboat, Ferry, Yacht, Sailboat, Fishing Vessel, **Oil Tanker**, Hovercraft, Motorboat, Dock

## Timestamp Verification

**Result: NO timestamps available.**

- The dataset is a **static image classification dataset** — no per-image capture date, timestamp, or temporal metadata is provided.
- No geographic coordinates are attached to individual images.
- Images come from "various sensors, satellite platforms, locations, and seasons" but this metadata is not exposed per instance.

## Implications for This Dissertation

- ShipRSImageNet **cannot** be used for time-series analysis of tanker activity.
- It **can** be used for:
  - Pre-training / fine-tuning an oil tanker detection model (transfer learning)
  - Demonstrating ship classification capability in the methodology chapter
  - Benchmarking classification accuracy on the "Oil Tanker" class
- For actual time-series tanker activity features, this project relies on **AIS-derived data** (IMF PortWatch, GFW, EMODnet, NOAA AIS) and **GEE-based remote sensing indices** instead.

1. ShipRSImageNet 里面有没有 Oil Tanker 这种油轮类别？
2. 这个数据集有没有时间戳，能不能用于油价预测里的时间序列航运特征？
有 Oil Tanker 类别，但没有时间戳，也没有每张图的地理坐标。


| 用途 | 是否适合 | 原因 |
|---|---|---|
| 周度油价预测特征 | 不适合 | 没有时间戳，不能构造 2005–2025 的时间序列 |
| 航运活动监测 | 不适合 | 没有连续时间、没有固定 AOI |
| 油轮识别模型训练 | 适合 | 有 Oil Tanker 类别和标注框 |
| 方法论展示 / Appendix | 适合 | 可以证明你核查过遥感船舶识别数据，但不纳入主模型 |






## 修改范围

排除 `02_ai_conversations/`（历史对话记录，保留原样）后，全项目零残留。

## 词汇替换映射

| 原词 | 替换为 |
|---|---|
| `geopolitical analysis` | `supply-chain disruption analysis` |
| `geopolitical events` | `exogenous disruption events` |
| `geopolitical intensity` | `disruption-event intensity` |
| `geopolitical shocks` | `exogenous shocks` |
| `geopolitical crises` | `market crises` |
| `geopolitical risk` | `disruption risk` / `exogenous risk` / `exogenous disruption risk` |
| `Geopolitical`（标题） | `Event Signals` / `Disruption` |
| `oil-geopolitical` | `oil-disruption` |
| `Oil-Geopolitical` | `Oil-Disruption` |
| `geopolitical sensitivity` | `data sensitivity` |
| `political, conflict, sanction` | `conflict, sanction, disruption` |
| `地缘政治` | `外部扰动` / `扰动事件` |
| `石油地缘政治` | `石油扰动事件` |
| `地缘政治风险` | `外部扰动风险` |

## 涉及的文件（共 16 个）

| 类别 | 文件 |
|---|---|
| 论文写作 | `chapter_1_introduction.md` |
| 代码 | `aggregate_gdelt_to_weekly.py`、`run_eda.py` |
| 特征表格 | `feature_table_en.md`、`feature_table_zh.md` |
| 管理文档 | `meeting_02_prep_20260527.md`、`File Structure.md`、`research_diary_20260527.md` |
| 文献笔记 | `paper_01_oil_price_ml.md`、`paper_05_multimodal_forecasting.md` |
| 数据集 Notebook | `Dataset_Overview.ipynb`、`Dataset_Original.0.ipynb`、`Dataset_Original.1.ipynb` |
| 原始数据文件 | 8 个 GDELT 文件或目录从 `*geopolitical*` 重命名为 `*disruption*` |
| 输出图表 | `11_gdelt_geopolitical_timeseries.png` → `11_gdelt_disruption_timeseries.png` |

## 未修改的内容

- `02_ai_conversations/`：历史 AI 对话记录，保留原样。
- CSV 内部列名 `gdelt_oil_geo_*`：`geo` 是模糊缩写，也可以理解为 `geographic`。修改这些列名需要重新生成全部 processed CSV，因此保留不变。