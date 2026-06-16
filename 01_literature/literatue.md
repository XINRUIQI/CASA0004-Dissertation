# 文献分类总表（Post-Meeting 02 + GNN + TFT + CV）

> **Last updated:** 2026-06-16（编号统一为 P001–P093）
>
> 对应最新项目方案：M1(Fin) → M2(Fin+RS) → M3(Fin+Ship) → M4(All)
> 模型：XGBoost (必做) / LSTM + TFT (推荐) / ST-GNN (进阶) / Baselines (对照)

---

## 一、分类总表


| ID                                               | 优先级        | 归类            | 用途说明                                                      |
| ------------------------------------------------ | ---------- | ------------- | --------------------------------------------------------- |
| **— 油价理论基础 / 预测基准（M1 变量合理性 + Methodology）—** |            |               |                                                           |
| **P052**                                         | ✅ **A 核心** | **核心-油价理论**   | Kilian 2009 — 油价冲击分解（供给/总需求/石油特定需求），M1 变量选择理论基础          |
| **P053**                                         | ✅ **A 核心** | **核心-油价预测**   | Alquist, Kilian & Vigfusson 2013 — 油价预测综述，样本期/变量/基准模型方法论依据   |
| **P054**                                         | ✅ **A 核心** | **核心-油价预测**   | Baumeister & Kilian 2015 — 基本面+实时数据预测，补强 EIA/FRED/Brent 变量逻辑  |
| **P058**                                         | ✅ **A 核心** | **核心-评估方法**   | Diebold & Mariano 1995 — 预测精度比较检验，M2/M3/M4 vs M1 显著性检验依据     |
| **P059**                                         | ✅ **A 核心** | **核心-可解释性**   | Lundberg & Lee 2017 — SHAP 原始论文，XGBoost+SHAP 解释方法必引            |
| **— 油价预测 ML 方法（支持 Tier 1 XGBoost + Baselines）—** |            |               |                                                           |
| P001                                             | ✅ A 核心     | 方法-油价预测 ML    | 深度学习预测油价，DL baseline 参考                                   |
| P002                                             | ✅ C 方法     | 方法-油价预测 ML    | ML 预测原油波动率，评估指标参考                                         |
| P003                                             | ✅ D 备用     | 方法-油价预测 ML    | 多种 ML 模型对比油价方向预测                                          |
| P004                                             | ✅ A 核心     | 方法-油价预测 ML    | LSTM+XGBoost 预测 WTI，核心模型参考                                |
| P005                                             | ✅ C 方法     | 方法-油价预测 ML    | 混合 ML 模型预测油价                                              |
| P006                                             | ✅ C 方法     | 方法-油价预测 ML    | MLP/CNN/Transformer 对比预测波动率                               |
| **— Shipping / AIS / 航运（M3 变量选择）—**              |            |               |                                                           |
| P014                                             | ✅ A 核心     | 核心-AIS/航运     | AIS 贸易量估算原油出口，定义 shipping 变量提取方法                          |
| P015                                             | ✅ A 核心     | 核心-AIS/航运     | AIS 数据分析全球海上石油贸易                                          |
| P016                                             | ✅ A 核心     | 核心-AIS/航运     | 油价对油轮港口停靠特征的影响，port-call frequency / dwell time           |
| P017                                             | ✅ A 核心     | 核心-AIS/航运     | 油价与油轮港口停靠次数的非线性关系                                         |
| P018                                  | ✅ A 核心     | 核心-AIS/航运     | Marini/IMF 船舶大数据实时预测贸易流量                            |
| **P068**                                         | ✅ **A 核心** | **核心-RS/航运**  | Jung 2026 — SAR+NTL+港口特征 nowcasting 港口贸易，RS↔Shipping 融合首选    |
| **P069**                                         | ✅ **A 核心** | **核心-RS/油价**  | Meunier 2026 — 卫星数据能否预测石油需求，M2 遥感变量进入油价预测的理论依据           |
| **P070**                                         | ✅ **A 核心** | **核心-AIS/航运** | IMF 2026 — PortWatch 国家贸易 nowcasting，PortWatch 方法论文            |
| **P061**                              | ✅ **A 核心** | **核心-RS/SAR**  | GFW Sentinel-1 SAR 船舶检测（GFW SAR 方法 + 数据文档），dark vessel 特征依据 |
| **P071**                                         | ✅ A 核心     | 方法-遥感/SAR     | Paolo 2022 — xView3-SAR 暗船检测，AIS 不可见船舶识别                     |
| **P057**                                         | ✅ **A 核心** | **核心-RS/SAR**  | Paolo et al. 2024 (Nature) — SAR 揭示 AIS 遗漏海上工业活动，dark vessel 强支撑 |
| **P060**                                         | ✅ B 重要     | 数据-AIS/航运     | OECD 2024 — 船舶轨迹数据用于贸易/港口/供应链监测，Data Section 引用              |
| **P072**                                         | ✅ A/B 重要   | 方法-油价预测 ML    | Costa 2021 — ML 油价点预测与密度预测，XGBoost/RF/SVR baseline 补充        |
| **P073**                                         | ✅ A/B 重要   | 方法-油价预测 ML    | Daneshvar 2022 — LSTM/Bi-LSTM 预测 Brent                          |
| **P074**                                         | ✅ B 重要     | 背景-油价预测       | Qin 2023 — ML+Google Search 预测油价，外部高频信息对照背景                   |
| **P075**                                         | ✅ B 重要     | 方法-油价预测 ML    | Cohen 2025 — 短期油价预测综合实验，baseline 与评价指标参考                    |
| **P076**                                         | ✅ B 重要     | 重要-金融/油价      | Yılmaz 2026 — GPR/VIX/美元等战略风险变量预测 Brent returns              |
| **P077**                                         | ✅ B 重要     | 方法-波动率预测      | Chung 2024 — GARCH+ML 能源市场波动率，支撑 target_brent_vol_next_1w   |
| **P078**                                         | ✅ A/B 重要   | 核心-夜间灯光       | Levin — NTL 综述与展望，VIIRS/Black Marble 理论背景                      |
| **P079**                                         | ✅ B 重要     | 方法-夜间灯光       | Doll 2008 — NTL 专题指南，航运灯光与经济活动                                |
| **P080**                                         | ✅ B 重要     | 重要-夜间灯光/供需    | Dasgupta 2022 — NTL 与短期供需冲击 proxy                               |
| **P081**                                         | ✅ B 重要     | 方法-遥感/SAR     | Li 2025 — 光学+SAR 协同暗船检测                                       |
| **P082**                                         | ⚠️ B/C 备用  | 数据-暗船         | GFW Dark Vessels Project，AIS 盲区背景                              |
| **P083**                                         | ⚠️ B/C 备用  | 数据-暗船/NTL     | GFW VIIRS 识别暗船，Future Work 参考                                  |
| **P084**                                         | ✅ A/B 重要   | 核心-航运/拥堵      | AIS+XGBoost+SHAP 预测港口拥堵，port_congestion_{aoi} 直接依据          |
| **P085**                                         | ✅ B 重要     | 重要-航运/港口网络    | Carlini 2025 — AIS+WPI 全球港口网络 ML 分析                           |
| **P086**                                         | ✅ B 重要     | 方法-航运/ST-GNN  | AIS-TGNN 2026 — TGAT 港口拥堵预测，ST-GNN+AIS 思路                     |
| **P087**                                         | ⚠️ B/C 备用  | 方法-航运/图ML     | Gouareb 2022 — 图 ML 船舶目的地预测，AIS 图结构背景                         |
| **P088**                                         | ⚠️ B/C 备用  | 方法-航运/Transformer | TransES-ETA 2025 — Transformer 港口 ETA，航运时序背景                |
| **P089**                                         | ✅ **A 核心** | **核心-TFT**    | Lim 2021 — Temporal Fusion Transformer 原始论文，TFT 架构必引          |
| **P090**                                         | ⚠️ B 备用    | 方法-Transformer | Informer 2021 — 长序列时序 Transformer，TFT 替代参考                    |
| **P091**                                         | ✅ **A 核心** | **核心-GNN/时空** | Graph WaveNet 2019 — ST-GNN 基础架构，GWNet-Attn (P063) 必引           |
| **P092**                                         | ✅ B 重要     | 方法-GNN/时空     | ST-GRAT 2019 — 时空图注意力，动态空间依赖                                |
| **P093**                                         | ✅ B 重要     | 方法-GNN/时空     | STGAT 2020 — 时空图注意力交通流预测，架构可迁移                              |
| P019                                             | ✅ B 重要     | 重要-航运         | AIS 与运费关系                                                 |
| P049                                             | ✅ B 重要     | 重要-遥感/石油基础设施  | 遥感识别全球石油资产，refinery/storage proxy 构建                      |
| P050                                             | ✅ B 重要     | 重要-航运/贸易网络    | 全球原油贸易网络时空动态，支持 shipping 变量和图结构                           |
| P051                                             | ⚠️ D 备用    | 备用-供应链        | 下游石油供应链建模，图拓扑设计参考                                         |
| **— Remote Sensing / 遥感（M2 变量选择）—**              |            |               |                                                           |
| P020                                             | ✅ C 方法     | 方法-遥感/船舶检测    | ShipRSImageNet 数据集论文                                      |
| P021                                             | ✅ A 核心     | 核心-遥感/数据集     | ShipRSImageNet 数据集本身                                      |
| P022                                             | ✅ C 方法     | 方法-遥感/船舶检测    | 卫星图像船舶识别                                                  |
| P023                                             | ✅ C 方法     | 方法-遥感/船舶检测    | Airbus 卫星船舶检测                                             |
| P024                                             | ✅ A 核心     | 核心-遥感/夜间灯光    | 夜间灯光作为锚地航运活动指标，NTL→shipping 变量逻辑                          |
| P025                                             | ✅ B 重要     | 重要-遥感/油价      | 云层覆盖与预期石油回报，RS→oil price 直接关联                             |
| **P055**                                         | ✅ **A 核心** | **核心-遥感/FRT** | Wang 2019 — 高分辨率遥感+阴影估算油罐体积，FRT fill-level 光学方法首选           |
| **P056**                                         | ✅ **A 核心** | **核心-遥感/FRT** | Villamil Lopez 2021 — SAR 监测浮顶油罐充填，FRT 多源遥感（SAR）支撑          |
| P026                                             | ✅ C 方法     | 方法-遥感/储油检测    | YOLOX 检测油罐，FRT 液位估算技术参考                                   |
| P027                                             | ✅ C 方法     | 方法-遥感/储油检测    | YOLOv7 检测储油罐                                              |
| P028                                             | ✅ B 重要     | 重要-遥感/数据集     | 地上储油罐遥感数据集                                                |
| P029                                             | ✅ B 重要     | 重要-遥感/数据集     | 油罐检测数据集                                                   |
| P030                                             | ✅ C 方法     | 方法-夜间灯光       | 夜间灯光衡量经济增长，NTL proxy 理论基础                                 |
| P031                                             | ✅ C 方法     | 方法-夜间灯光       | 夜间灯光作为经济代理的精度评估                                           |
| P032                                             | ✅ A 核心     | 核心-夜间灯光       | 经济学中应选哪种 NTL 数据，数据源选择指南                                   |
| P033                                             | ✅ B 重要     | 重要-遥感/经济      | IMF 卫星数据经济监测                                              |
| P034                                             | ⚠️ D 备用    | 备用-遥感         | 卫星检测城市市场，与油价关联弱                                           |
| P035                                             | ⚠️ D 备用    | 备用-夜间灯光       | 夜间灯光与出口增长，关联间接                                            |
| P036                                             | ✅ B 重要     | 重要-夜间灯光       | 世界银行 NTL 作为经济代理的方法论                                       |
| P037                                             | ✅ C 方法     | 方法-遥感/数据      | NASA Black Marble 产品论文，数据源参考                              |
| P038                                             | ⚠️ D 备用    | 备用-遥感         | 50 年夜间灯光观测综述，背景知识                                         |
| **— Transformer / TFT / 多模态融合（支持 Tier 2 TFT）—**  |            |               |                                                           |
| P039                                             | ✅ B 重要     | 方法-多模态/TFT    | 多模态 Transformer 用于金融时间序列，TFT 架构参考                         |
| P041                                             | ✅ B 重要     | 方法-多模态/TFT    | 多方面注意力 Transformer 用于金融预测                                 |
| P042                                             | ✅ B 重要     | 重要-综述/TFT     | Transformer 时间序列预测综述，模型选型参考                               |
| **— GNN / 图神经网络（支持 Tier 3 ST-GNN）—**             |            |               |                                                           |
| P043                                             | ✅ C 方法     | 方法-GNN/供应链    | GNN 建模供应链，图构建方法参考                                         |
| P044                                             | ✅ C 方法     | 方法-GNN/供应链    | 图神经网络供应链预测，GNN 架构参考                                       |
| P045                                             | ✅ C 方法     | 方法-GNN/供应链    | 混合 GAT-LSTM 供应链预测，时空融合方法                                  |
| P046                                             | ⚠️ D 备用    | 方法-GNN/基准     | SupplyGraph 基准数据集，图构建参考                                   |
| P047                                             | ✅ C 方法     | 方法-GNN/时空     | 时空图注意力网络 STGAT，架构可迁移                                      |
| P048                                             | ✅ C 方法     | 方法-GNN/能源价格   | 时空 GNN 预测电价，能源价格 GNN 方法参考                                 |
| **P062**                                           | ✅ **A 核心** | **核心-GNN/原油** | LGCOTFF — LSTM+GCN 原油海运网络（港口=节点，航线=边），图构建首选参考             |
| **P063**                                           | ✅ **A 核心** | **核心-GNN/油价** | GWNet-Attn — Self-Attention + Graph WaveNet 直接预测 WTI 期货价格 |
| **P064**                                           | ✅ B 重要     | 重要-GNN/油价     | BiLSTM-GCN — 组合模型预测原油价格                                   |
| **P065**                                           | ✅ B 重要     | 重要-GNN/贸易     | Russian Oil GNN — GNN 预测石油贸易流量（含制裁场景）                     |
| **P066**                                           | ✅ B 重要     | 重要-GNN/航运     | STMGCN — AIS 构建海上交通图 + 时空多图卷积网络                           |
| **P067**                                           | ✅ B 重要     | 重要-GNN/港口     | ITSG-LSTM — 国家间贸易相似性图 + GCN-LSTM 预测港口吞吐量                  |
| **— Text / NLP / 新闻情感（已移出核心范围，保留备查）—**           |            |               |                                                           |
| P007                                             | 📋 D 备查    | 备查-Text/NLP   | LLM 情感信号预测 WTI — 文本模态已移除，Lit Review 中可提及                  |
| P008                                             | 📋 D 备查    | 备查-Text/NLP   | 新闻情感预测油价 — 备查                                             |
| P009                                             | 📋 D 备查    | 备查-Text/NLP   | OPEC+ 新闻预测油价 — 备查                                         |
| P010                                             | 📋 D 备查    | 备查-Text/NLP   | 新闻事件+LLM 时间序列预测 — 备查                                      |
| P011                                             | 📋 D 备查    | 备查-Text/NLP   | 全球新闻预测短期油价 — 备查                                           |
| P012                                             | 📋 D 备查    | 备查-Text/NLP   | 新闻预测油价波动方向 — 备查                                           |
| P013                                             | 📋 D 备查    | 备查-Text/NLP   | 时间+语义融合预测大宗商品 — 备查                                        |
| P040                                             | 📋 D 备查    | 备查-Text/多模态   | MM-iTransformer 融合文本数据 — 备查                               |


---

## 二、汇总统计


| 状态                     | 数量    | 涉及论文                                                                          |
| ---------------------- | ----- | ----------------------------------------------------------------------------- |
| ✅ 核心/重要/方法             | 81 篇  | P001–P006, P014–P033, P036–P037, P039, P041–P050, P062–P067, P068–P093, P052–P060（去重） |
| ⚠️ 边缘/备用               | 9 篇   | P034, P035, P038, P046, P051, P082, P083, P087, P088, P090                    |
| 📋 备查（Text/NLP，不在核心范围） | 8 篇   | P007–P013, P040                                                               |
| 📦 纯数据集                 | 4 项   | P021, P028, P029, P046                              |
| **合计**                    | **93** | P001–P093 连续编号；按 Tier 1–7 重要性排序                              |


---

## 三、备查论文的处置说明

P007–P013 和 P040 共 8 篇 Text/NLP 论文，根据 Meeting 02 决策（移除文本模态）不再作为核心文献，但**不删除**：

- **Literature Review 中**：简要提及"已有研究探索了新闻情感/LLM 信号对油价预测的作用"，引用 P007–P009 作为代表
- **Discussion 中**：如果模型结果不佳，可提及"未纳入文本信号可能是局限之一"
- **Future Work 中**：建议未来加入文本模态（引用 P007–P013）

---

## 四、当前保留的五个文献方向


| 方向                          | 服务于              | 关键论文                          |
| --------------------------- | ---------------- | ----------------------------- |
| **Financial / Market data** | M1 baseline 变量选择 | P052, P053, P054, P004, P059  |
| **Remote Sensing (RS)**     | M2 遥感代理变量        | P068, P069, P055, P056, P057  |
| **Shipping / AIS**          | M3 航运特征          | P070, P016, P017, P061, P060  |
| **Evaluation**              | M1–M4 模型比较       | P058, P084, P059              |
| **Transformer / TFT**       | Tier 2 模型架构      | P089, P039, P042              |
| **GNN / Graph**             | Tier 3 ST-GNN 架构 | P091, P063, P062, P066              |


---

## 五、文献阅读优先级


| 模态/方向             | 优先精读                              | 理由                                              |
| ----------------- | --------------------------------- | ----------------------------------------------- |
| Financial         | **P052, P053, P054, P004**        | Kilian 油价理论 + 预测基准 + LSTM+XGBoost 核心            |
| Evaluation        | **P058, P059**                    | Diebold-Mariano 模型比较 + SHAP 可解释性                 |
| Shipping          | **P016, P017, P070, P057**        | oil price ↔ shipping + PortWatch + Nature SAR 暗船   |
| Remote Sensing    | **P068, P055, P056, P061**        | RS↔Shipping + FRT 液位（光学/SAR）+ GFW SAR 数据          |
| TFT / Transformer | **P089, P039**                    | TFT 原始论文 + 多模态金融 Transformer                    |
| GNN / 图建模         | **P091, P063, P062**                  | Graph WaveNet 基础 + GWNet-Attn + 海运网络图构建         |


---

## 六、优先精读清单（Tier 1–2，共 30 篇）

精读完 **序 1–30** 后，可确定：M1 变量合理性、M1–M4 比较方法（DM 检验）、三模态变量定义、以及 XGBoost/LSTM/TFT/ST-GNN 架构选择。

---

# 论文阅读优先级清单（按重要性排序，共 93 篇）

> **排序逻辑：** 油价理论基础 → 核心模型与评估方法 → 三模态变量锚点 → 模型架构原始论文 → 模态补充 → 方法参考 → 背景/数据集/备查
>
> **编号说明：** P001–P093 连续编号（P018 = Marini/IMF vessel traffic nowcasting；P061 = GFW SAR 方法 + 数据文档）

## Tier 1 — 最优先精读（直接定义研究框架，序 1–15）


| 序   | ID   | 论文                                              | 理由                                      |
| --- | ---- | ----------------------------------------------- | --------------------------------------- |
| 1   | P052 | Kilian (2009) — Not All Oil Price Shocks Are Alike | M1 变量理论基础：供给/总需求/石油特定需求冲击分解              |
| 2   | P053 | Alquist, Kilian & Vigfusson (2013) — Forecasting Oil | Methodology 预测基准：样本期、变量选择、模型设定            |
| 3   | P054 | Baumeister & Kilian (2015) — Real Price of Oil  | EIA/FRED/Brent 基本面与实时数据预测逻辑               |
| 4   | P004 | Simsek / Li — LSTM+XGBoost WTI                  | 核心模型组合 + SHAP，直接可复制                      |
| 5   | P059 | Lundberg & Lee (2017) — SHAP                    | XGBoost 可解释性原始论文，与 P004 配套                |
| 6   | P016 | Mi et al. (2022) — Port-Call Features           | 唯一直接研究 oil price → shipping 变量             |
| 7   | P017 | Mi & Zang (2023) — Port-Call Count              | oil price ↔ port-call 非线性，M3 核心依据        |
| 8   | P024 | Polinov et al. (2022) — NTL Shipping            | 唯一 NTL 量化航运活动，RS ↔ Shipping 桥梁            |
| 9   | P068 | Jung (2026) — Watching Trade from Space         | SAR+NTL+港口 nowcasting，新 plan 融合首选          |
| 10  | P058 | Diebold & Mariano (1995) — DM Test              | M2/M3/M4 vs M1 预测精度显著性检验                   |
| 11  | P070 | IMF (2026) — PortWatch Country-Level Trade      | 正在使用的 PortWatch 最新方法论文                    |
| 12  | P069 | Meunier (2026) — Can Satellites Predict Oil Demand | 卫星→石油需求，M2 遥感进入油价预测的理论依据                 |
| 13  | P063   | GWNet-Attn (2023) — ST-GNN WTI                  | 唯一直接 GNN 预测油价；配合 P091 理解架构                |
| 14  | P089 | Lim et al. (2021) — Temporal Fusion Transformer | TFT 原始核心文献                              |
| 15  | P091 | Wu et al. (2019) — Graph WaveNet               | ST-GNN 基础架构，P063 必引前置                       |


## Tier 2 — 核心支撑（数据模态与架构直接依据，序 16–30）


| 序   | ID          | 论文                                      | 理由                                   |
| --- | ----------- | --------------------------------------- | ------------------------------------ |
| 16  | P062          | LGCOTFF (2022) — LSTM+GCN Maritime      | 原油海运网络图构建首选参考                        |
| 17  | P014        | Adland et al. (2017) — AIS Trade Volume | 从 AIS 提取 shipping 变量的方法               |
| 18  | P061   | GFW Sentinel-1 SAR Vessel Detections    | SAR dark vessel 数据源（方法 + 数据文档）       |
| 19  | P057        | Paolo et al. (2024) — Nature SAR at Sea | AIS 遗漏海上活动，SAR 补足盲区                   |
| 20  | P025        | Hao & Wang (2023) — Cloud Cover         | 唯一直接 RS → oil returns                 |
| 21  | P032        | Gibson et al. (2021) — Which NTL        | NTL 数据源选择（DMSP/VIIRS/Black Marble）    |
| 22  | P001        | Foroutan & Lahmiri (2024) — DL Oil      | Financial baseline 变量列表              |
| 23  | P055        | Wang et al. (2019) — FRT Volume from RS | 光学遥感 FRT 液位估算首选                       |
| 24  | P056        | Villamil Lopez (2021) — SAR Oil Tank Filling | SAR 监测浮顶油罐充填                        |
| 25  | P084        | Port Congestion — AIS + XGBoost + SHAP  | port_congestion_{aoi} 直接依据           |
| 26  | P071        | Paolo et al. (2022) — xView3-SAR        | SAR 暗船检测方法，与 P057 配套                  |
| 27  | P039        | Modality-aware Transformer (2024)       | 多模态 Transformer 金融时序，TFT 设计参考        |
| 28  | P042        | Transformer Survey (2026)               | 时序 Transformer 综述与模型选型               |
| 29  | P066          | STMGCN (Southampton)                    | AIS → 海上交通图 + 时空多图卷积                 |
| 30  | P018   | Marini et al. (2019) — Vessel Traffic Big Data | IMF 早期 AIS nowcasting 经典文献           |


## Tier 3 — 重要补充（航运/遥感理论/GNN 应用，序 31–50）


| 序   | ID   | 论文                                              | 理由                       |
| --- | ---- | ----------------------------------------------- | ------------------------ |
| 31  | P015 | Yan et al. (2020) — Global Marine Oil           | AIS 全球石油贸易分析             |
| 32  | P060 | OECD (2024) — An Ocean of Data                  | AIS 数据用于贸易/港口，Data Section |
| 33  | P050 | Niu et al. (2023) — Trade Network               | 原油贸易网络时空动态               |
| 34  | P078 | Levin et al. — NTL Review                       | NTL 综述与展望                |
| 35  | P033 | IMF (2024) — Satellite Monitoring               | 卫星衍生经济指标概览               |
| 36  | P080 | Dasgupta (2022) — NTL Supply–Demand Shocks      | NTL 短期供需冲击 proxy         |
| 37  | P049 | Kruitwagen et al. (2021) — Global Oil Assets    | 遥感识别全球石油基础设施             |
| 38  | P030 | Henderson et al. (2012) — NTL Economic Growth   | NTL 经济增长理论基础             |
| 39  | P072 | Costa (2021) — ML Oil Price Point & Density     | XGBoost/RF/SVR baseline  |
| 40  | P073 | Daneshvar (2022) — LSTM/Bi-LSTM Brent           | 直接预测 Brent               |
| 41  | P076 | Yılmaz (2026) — Strategic Risk Brent Returns  | GPR/VIX/美元等 M1 变量        |
| 42  | P077 | Chung (2024) — GARCH + ML Energy Volatility     | target_brent_vol 波动率支撑   |
| 43  | P002 | Luo et al. (2024) — ML Crude Volatility         | ML 波动率评估指标参考             |
| 44  | P064   | BiLSTM-GCN (2023)                               | GCN+RNN 预测原油价格           |
| 45  | P065   | Russian Oil GNN (2025)                          | GNN 预测石油贸易（制裁场景）         |
| 46  | P067   | ITSG-LSTM (2025)                                | 贸易相似性图 + 港口吞吐量预测         |
| 47  | P085 | Carlini (2025) — Port Networks ML               | AIS+WPI 全球港口网络           |
| 48  | P086 | AIS-TGNN (2026) — Port Congestion TGAT          | ST-GNN+AIS 港口拥堵          |
| 49  | P041 | EMAT (2025)                                     | 多方面注意力 Transformer 金融预测  |
| 50  | P092 | ST-GRAT (2019)                                  | 时空图注意力，动态空间依赖            |


## Tier 4 — 方法参考（ST-GNN/ML/CV 补充，序 51–65）


| 序   | ID   | 论文                             | 理由                       |
| --- | ---- | ------------------------------ | ------------------------ |
| 51  | P047 | STGAT — 时空图注意力                 | STGAT 架构可迁移              |
| 52  | P093 | STGAT (2020) — Traffic Flow    | 时空图注意力交通流参考              |
| 53  | P043 | GNN Supply Chain               | 供应链图构建参考                 |
| 54  | P048 | ST-GNN Electricity Price       | 能源价格 ST-GNN 可迁移          |
| 55  | P003 | Directional Forecasting (2025) | 多模型方向预测对比（SVM/RF 等）      |
| 56  | P005 | Hybrid ML (2025)               | 混合 ML 方法参考               |
| 57  | P006 | MLP/CNN/Transformer (2024)   | 深度学习对比实验设计               |
| 58  | P075 | Cohen (2025) — Short-Term Oil  | 短期油价综合实验与评价指标            |
| 59  | P074 | Qin (2023) — ML + Google Search | 外部高频信息对照背景               |
| 60  | P020 | ShipRSImageNet Paper           | 船舶检测数据集论文                |
| 61  | P022 | Self-Supervised Ship ID (2026) | 卫星船舶识别                   |
| 62  | P081 | Li (2025) — Optical+SAR Dark Ship | 光学+SAR 协同暗船检测            |
| 63  | P026 | YOLOX Oil Tank (2022)          | SAR 油罐检测，FRT 补充          |
| 64  | P023 | Airbus Ship Detection          | 卫星船舶检测                   |
| 65  | P031 | Nordhaus & Chen (2015) — NTL   | NTL 精度评估                 |


## Tier 5 — 背景与边缘（序 66–80）


| 序   | ID   | 论文                          | 理由                       |
| --- | ---- | --------------------------- | ------------------------ |
| 66  | P079 | Doll (2008) — NTL Guide     | 早期 NTL 专题指南              |
| 67  | P036 | World Bank — NTL Proxy      | NTL 方法论总结                |
| 68  | P019 | Bayes — AIS & Freight Rates | AIS 与运费关系                |
| 69  | P090 | Informer (2021)             | 长序列 Transformer 备选（TFT 过重时） |
| 70  | P087 | Gouareb (2022) — Vessel Destination Graph ML | AIS 图结构背景               |
| 71  | P088 | TransES-ETA (2025)          | 航运 Transformer ETA 背景    |
| 72  | P082 | GFW — Dark Vessels Project  | AIS 盲区背景                 |
| 73  | P083 | GFW — VIIRS Dark Vessels    | VIIRS 暗船，Future Work     |
| 74  | P027 | YOLOv7 Storage Tank (2024)  | 大尺度遥感储罐检测                |
| 75  | P044 | Graph Neural Poisson        | GNN 供应链预测架构              |
| 76  | P045 | GAT-LSTM Supply Chain       | GAT+LSTM 时空融合            |
| 77  | P034 | Urban Markets Satellite     | 背景-卫星经济分析                |
| 78  | P035 | NTL & Export Growth         | 背景-NTL 与出口               |
| 79  | P038 | 50 Years NTL (2022)         | 背景-NTL 历史综述              |
| 80  | P051 | Downstream Oil Supply Chain | 图拓扑参考，优先级低               |


## Tier 6 — 纯数据集 📦（序 81–85）


| 序   | ID   | 数据集                         | 用途           |
| --- | ---- | --------------------------- | ------------ |
| 81  | P037 | NASA Black Marble           | 数据源产品文档      |
| 82  | P021 | ShipRSImageNet GitHub       | 船舶检测图像数据集    |
| 83  | P028 | Robinson/Meng — AST Dataset | 地上储油罐遥感数据集   |
| 84  | P029 | Rizk & Chehade — Oil Tank   | 油罐检测数据集      |
| 85  | P046 | SupplyGraph Benchmark       | 供应链 GNN 基准数据集 |


## Tier 7 — 备查 Text/NLP（已移出核心范围）📋（序 86–93）


| 序   | ID   | 论文                               | 在论文中的使用方式            |
| --- | ---- | -------------------------------- | --------------------- |
| 86  | P007 | LLM Sentiment WTI (2026)         | Lit Review 提及 LLM 方向  |
| 87  | P009 | OPEC+ News Predictor (2024)      | Lit Review 提及新闻信号     |
| 88  | P008 | News Sentiment Oil (2025)        | Discussion 未纳入文本可能是局限 |
| 89  | P011 | Global News Oil (2025)           | Future Work 文本模态      |
| 90  | P012 | News Volatility Direction (2025) | Discussion / Future Work |
| 91  | P013 | Temporal Semantic Fusion (2025)  | 同上                    |
| 92  | P010 | News + LLM Time Series (2024)    | 方法参考（未来加入 LLM）        |
| 93  | P040 | MM-iTransformer Text (2025)      | 多模态+文本融合备查            |


