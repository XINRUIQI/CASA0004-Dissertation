# ShipRSImageNet Dataset Verification

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

