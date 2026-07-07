# 文献分类总表（Post-Meeting 02 + GNN + TFT + CV）

> **Last updated:** 2026-07-07（编号 P001–P135 + R001–R021；R = M1 数据字典 §4.6 业界/机构 Report 层）
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
| **P102**                                         | ✅ B 重要     | **重要-油价/WTI-Brent** | Büyükşahin et al. 2013 — 物理/纸面市场与 WTI–Brent 价差机制，支撑 `brent_wti_spread` 经济含义 |
| **P103**                                         | ✅ B 重要     | **重要-油价/出口禁令**   | Agerton & Upton 2019 — 分解原油价差（国内运输约束 vs 出口禁令），支撑 `crude_exports` 制度断点说明 |
| **P104**                                         | ✅ B 重要     | **重要-油价预测**   | Yu, Wang & Lai 2008 — EMD+NN 集成预测 WTI/Brent 现货价，支撑 `brent_price`/`wti_price` 作预测目标 |
| **P105**                                         | ✅ B 重要     | **重要-油价预测**   | Abdollahi & Ebrahimi 2020 — 混合模型预测 Brent 价格水平 |
| **P106**                                         | ✅ B 重要     | **重要-油价预测**   | Ye, Zyren & Shore 2005 — WTI 现货月频预测（相对库存），EIA 经典库存—价格方法 |
| **P107**                                         | ⚠️ C 方法     | **方法-油价/收益**   | Chen, Chiu & Hsiao 2021 — Brent 投资辅助指数（收益/波动风险度量） |
| **P108**                                         | ⚠️ B/C 备用  | **重要-油价/事件**   | Ma, Xiong & Bao 2021 — 俄沙油价战事件研究（WTI/Brent/Oman 现货与期货收益） |
| **P109**                                         | ✅ B 重要     | **方法-波动率**    | Chen, Zerilli & Baum 2019 — WTI/Brent **现货**对数收益随机波动率与 VaR/CVaR |
| **P110**                                         | ⚠️ C 方法     | **方法-波动率**    | Zhang, Yao, He & Ripple 2019 — 原油市场波动率 regime-switching GARCH（日/周 WTI & Brent） |
| **P111**                                         | ⚠️ C 方法     | **方法-油价预测 ML** | Wang & Chen 2025 — XGBoost 等 ML 预测 WTI 收益 |
| **P112**                                         | ✅ B 重要     | **重要-油价/WTI-Brent** | Scheitrum, Carter & Revoredo-Giha 2018 — WTI–Brent 价差 2011 结构断点 |
| **P113**                                         | ✅ B 重要     | **重要-油价/WTI-Brent** | Bravo Caro et al. 2020 — WTI–Brent 价差 FCVAR/长记忆与全球化测度 |
| **P114**                                         | ⚠️ C 方法     | **方法-波动率**    | Charles & Darné 2017 — 原油市场波动率预测（GARCH/GAS/MSM + 跳跃） |
| **P115**                                         | ✅ B 重要     | **重要-EIA/库存**  | Bu 2014 — 库存公告与油价波动（库存信息冲击） |
| **P116**                                         | ✅ B 重要     | **重要-EIA/库存**  | Armstrong et al. 2021 — 库存意外、分歧与价格漂移 |
| **P117**                                         | ✅ B 重要     | **重要-EIA/库存**  | Kim, Baek & Heo 2020 — Cushing/Global/US 库存 SVAR |
| **P118**                                         | ✅ B 重要     | **重要-EIA/炼厂**  | Kaufmann et al. 2008 — 炼厂开工率与实际油价（月频） |
| **P119**                                         | ✅ B 重要     | **重要-EIA/基本面** | Zagaglia 2010 — FAVAR 能源数量信息集（进出口/炼厂/成品油） |
| **P120**                                         | ⚠️ C 方法     | **重要-EIA/基本面** | Malliaris & Malliaris 2021 — 微观基本面驱动油价（1986–2020） |
| **P121**                                         | ⚠️ C 方法     | **方法-油价预测 ML** | Wei 2026 — LASSO/ML 预测原油期货（含产量/进口增长） |
| **P122**                                         | ✅ B 重要     | **方法-油价预测 ML** | Tissaoui et al. 2023 — XGBoost+SHAP；VIX/OVX 预测 WTI |
| **P123**                                         | ✅ B 重要     | **重要-宏观/油价**  | He, Wang & Lai 2010 — 美元、全球经济活动与油价协整 |
| **P124**                                         | ⚠️ C 方法     | **方法-宏观/油价**  | Qadan & Cohen 2024 — 利率不确定性与油价收益/波动 |
| **P125**                                         | ⚠️ C 方法     | **重要-宏观/油价**  | Basistha & Kurov 2015 — 货币政策意外与能源价格 |
| **P126**                                         | ✅ B 重要     | **重要-宏观/油股**  | Kilian & Park 2009 — 油价冲击对美国股市的影响 |
| **P127**                                         | ✅ B 重要     | **重要-宏观/油股**  | Sadorsky 1999 — 油价冲击与股市；S&P 500 连续复合收益 |
| **P128**                                         | ⚠️ C 方法     | **重要-宏观/油股**  | Lu et al. 2021 — WTI 与 S&P 500 股指期货 log-returns |
| **P129**                                         | ⚠️ C 方法     | **重要-宏观/油股**  | Hussain et al. 2022 — 油价与欧洲工业股指（含 S&P 500/Brent 收益） |
| **P130**                                         | ⚠️ C 方法     | **重要-宏观/油股**  | Roy et al. 2023 — 油市互联与统一对数收益口径 |
| **P131**                                         | ✅ B 重要     | **重要-宏观/GPR**  | Caldara & Iacoviello 2022 — GPR 指数原始论文，支撑 `gpr` |
| **P132**                                         | ⚠️ C 方法     | **重要-跨资产**    | Kang, McIver & Yoon 2017 — 金/油/农产品期货溢出 |
| **P133**                                         | ✅ B 重要     | **重要-宏观/商品**  | Kilian & Zhou 2018 — 全球商品需求波动；工业原料价格 |
| **P134**                                         | ✅ B 重要     | **重要-期限结构**   | Valenti 2022 — Brent 期货–现货价差进入 SVAR |
| **P135**                                         | ✅ B 重要     | **重要-宏观/汇率**  | Chen, Rogoff & Rossi 2010 — 商品货币预测商品价格 |
| **— 业界/机构 Report 层（M1 数据字典 §4.6 · 非学术论文）—** |            |               |                                                           |
| **R001**                                         | 📄 Report  | **EIA/价格**    | EIA 原油现货价表（Brent–Europe / WTI–Cushing） |
| **R002**                                         | 📄 Report  | **EIA/价差**    | EIA Today in Energy — Brent–WTI 价差扩大 |
| **R003**                                         | 📄 Report  | **EIA/价格**    | EIA Today in Energy — Daily Prices |
| **R004**                                         | 📄 Report  | **OPEC/月报**   | OPEC Monthly Oil Market Report |
| **R005**                                         | 📄 Report  | **CME/WTI**   | CME WTI Insights 周报 |
| **R006**                                         | 📄 Report  | **ICE/风险**    | ICE Risk Model 2.0 Methodology |
| **R007**                                         | 📄 Report  | **EIA/WPSR**  | EIA Weekly Petroleum Status Report |
| **R008**                                         | 📄 Report  | **EIA/WPSR**  | EIA Today in Energy — WPSR 概述 |
| **R009**                                         | 📄 Report  | **EIA/需求**    | EIA FAQ — product supplied |
| **R010**                                         | 📄 Report  | **EIA/需求**    | EIA 馏分油周度 product supplied 序列 |
| **R011**                                         | 📄 Report  | **CME/交割**    | CME WTI 期货 Cushing 交割说明 |
| **R012**                                         | 📄 Report  | **CME/EIA**   | CME Econoday — EIA 周报日历 |
| **R013**                                         | 📄 Report  | **Reuters/EIA** | Reuters EIA 周度市场报道（例证） |
| **R014**                                         | 📄 Report  | **Cboe/VIX**  | Cboe VIX 指数说明 |
| **R015**                                         | 📄 Report  | **EIA/宏观**    | EIA Markets & Finance（美元计价） |
| **R016**                                         | 📄 Report  | **CME/宏观**    | CME Economic Data and Crude Oil |
| **R017**                                         | 📄 Report  | **MSCI/风险**   | RiskMetrics Technical Document（log returns） |
| **R018**                                         | 📄 Report  | **Cboe/OVX**  | Cboe OVX 原油波动率指数 |
| **R019**                                         | 📄 Report  | **Dallas Fed** | Kilian REA 指数（igrea）数据页 |
| **R020**                                         | 📄 Report  | **IMF/商品**    | IMF Primary Commodity Prices |
| **R021**                                         | 📄 Report  | **GPR/数据**    | Caldara–Iacoviello GPR 数据页（≠ P131 论文） |
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

### ⑥ 方法集成与实证检验层（创新层，P094–P101 + P103–P115 · 2026-06-30 精读补入）

> 支撑研究方案 §4.2 通道 A / §4.4 缺失与异步 / §5 模态编码器与门控融合；与上方各模态表分开列出。P094–P101 均已写 reading note（`reading_notes/06`–`10 创新-*` 子目录，见下表）+ `literature_matrix.md` §⑥ 完整条目；**P103–P115（含 P113 RobSense、P115 MM-RSFM 综述）2026-07-07 新增**。⚠️ 均非油价实证，为方法骨架，须经本项目消融验证。
>
> **§⑥ 创新层子目录（2026-07-07 细分）：** `06 创新-金融+图编码器`（P039/P063/P091）· `07 创新-EO基础模型`（P094/P095/P103/P104/P106/P108/P110/P113）· `08 创新-EO多模态融合`（P105/P107/P109/P111/P112/P114）· `09 创新-融合机制与缺失异步`（P096–P101/P115）· `10 创新-时序架构参考`（P088/P089）。PDF 对应 `papers_pdf/06`–`10/`。
>
> ⚠️ **编号消歧：** **§⑥ P103** = Reed et al. (2023) Scale-MAE；**§① P103** = Agerton & Upton (2019) 出口禁令/价差（M1 `crude_exports` 支撑）——二者无关。**§⑥ P104** = Tseng et al. (2024) Presto；**§① P104** = Yu et al. (2008) EMD+NN（M1 价格变量支撑）——二者无关。**§⑥ P105** = Fuller et al. (2023) CROMA；**§① P105** = Abdollahi & Ebrahimi (2020) Brent 价格混合预测（M1 `brent_price` 支撑）——二者无关。**§⑥ P106** = Xiong et al. (2025) DOFA；**§① P106** = Ye et al. (2005) WTI 现货预测（M1 `wti_price` 支撑）——二者无关；reading note 见 `07 创新-EO基础模型/P106.md`。**§⑥ P107** = Astruc et al. (2024) OmniSat；**§① P107** = Chen et al. (2021) Brent 投资辅助指数（M1 `brent_log_return` 支撑）——二者无关。**§⑥ P108** = Danish et al. (2025) TerraFM；**§① P108** = Ma et al. (2021) 俄沙油价战事件研究（M1 `wti_log_return` 支撑）——二者无关。**§⑥ P109** = Hong et al. (2021) S2FL；**§① P109** = Chen et al. (2019) 现货收益 SV/VaR（M1 `brent_log_return`/`wti_log_return` 支撑）——二者无关。**§⑥ P110** = Weber & Beneke (2025) PyViT-FUSE；**§① P110** = Zhang et al. (2019) 原油市场波动率 GARCH（M1 `brent_log_return` 支撑）——二者无关。**§⑥ P111** = Guo et al. (2025) CCFormer；**§① P111** = Wang & Chen (2025) WTI 收益 ML 预测（M1 `wti_log_return` 支撑）——二者无关；reading note 见 `08 创新-EO多模态融合/P111.md`。**§⑥ P112** = Zhao et al. (2025) CFFormer；**§① P112** = Scheitrum et al. (2018) WTI–Brent 价差 2011 结构断点（M1 `brent_wti_spread` 支撑）——二者无关。**§⑥ P113** = Do et al. (2025) RobSense；**§① P113** = Bravo Caro et al. (2020) WTI–Brent 价差 FCVAR（M1 `brent_wti_spread` 支撑）——二者无关。**§⑥ P114** = Wang, Chen, Ma et al. (2024) ShaSpec；**§① P114** = Charles & Darné (2017) 原油波动率 GARCH/GAS/MSM（M1 `brent_log_return`/`wti_log_return` 支撑）——二者无关。**§⑥ P115** = Zhou, Qian & Gamba (2025) MM-RSFM 综述；**§①-c P115** = Bu (2014) EIA 库存公告信息冲击（M1 `crude_stocks_excl_spr`/`crude_stocks_change` 支撑）——二者无关。

| ID | 优先级 | 归类 | 用途说明 |
| --- | --- | --- | --- |
| **P094** | ✅ A 核心 | 核心-EO 基础模型 | Prithvi-EO-2.0（Szwarcman 2024, arXiv:2412.02732）— 冻结多时相 EO 基础模型提 image embedding，M2 通道 A 编码器 |
| **P095** | ✅ A 核心 | 核心-EO 基础模型 | SatMAE（Cong 2022, NeurIPS）— 多光谱 / 时序卫星 MAE 预训练，遥感表示学习 |
| **P103** | ✅ A 核心 | 核心-EO 基础模型 | Scale-MAE（Reed 2023, ICCV, arXiv:2212.14532）— GSD-aware MAE + Laplacian decoder；多尺度 RGB 表征；frozen kNN/linear probe |
| **P104** | ✅ A 核心 | 核心-EO 基础模型 | Presto（Tseng 2024, arXiv:2304.14065v4）— 轻量级 pixel-timeseries Transformer；多源 S1/S2/ERA5/NDVI + structured masking；冻结 EO encoder + 轻量预测头 |
| **P105** | ✅ A 核心 | 核心-EO 多模态融合 | CROMA（Fuller 2023, NeurIPS, arXiv:2311.00566）— S1+S2 contrastive MAE；radar/optical/joint 三编码器；optional unimodal；2D-ALiBi 可变 AOI |
| **P106** | ✅ A 核心 | 核心-EO 基础模型 | DOFA（Xiong 2025, arXiv:2403.15356v3）— wavelength-conditioned dynamic hypernetwork + shared Transformer；5 类异构 EO 模态；MIM + distillation；DOFA+ 自 DINOv2；frozen backbone + lightweight head |
| **P107** | ✅ A 核心 | 核心-EO 多模态融合 | OmniSat（Astruc 2024, ECCV, arXiv:2404.08351）— VHR + S2/S1 时序自监督融合；modality-specific encoder + cross-attention combiner；支撑 M4 EO 子网络架构 |
| **P108** | ✅ A 核心 | 核心-EO 基础模型 | TerraFM（Danish 2025, arXiv:2506.06281）— S1+S2 统一 foundation model；modality-as-augmentation + cross-attention；frozen backbone + linear/kNN probing；GEO-Bench / Copernicus-Bench |
| **P110** | ✅ A 核心 | 核心-EO 基础模型 | PyViT-FUSE（Weber & Beneke 2025, ICLR ML4RS）— SPOT+S1+S2+L8 混合分辨率 24 bands；SwAV + cross-attention fusion；band drop 缺失鲁棒；frozen encoder + 轻量 decoder |
| **P113** | ✅ A 核心 | 核心-EO 基础模型 | RobSense（Do et al. 2025, CVPR）— SatlasPretrain MS+SAR T=8；MAE + TDA (KL) + latent reconstructors；static/temporal/incomplete 适配；frozen encoder 路线 |
| **P109** | ✅ A 核心 | 核心-shared+specific 融合 | S2FL（Hong 2021, *ISPRS JPRS*, arXiv:2105.10196）— 多模态 RS 分解为 shared/specific 子空间；manifold alignment；支撑 RQ2 消融 ladder |
| **P111** | ✅ A 核心 | 核心-cross-attention 融合 | CCFormer（Guo 2025, *Sensors*, DOI: 10.3390/s25185698）— HSI+LiDAR 双分支 + 双向 cross-attention；支撑 RQ2 early fusion vs modality-aware fusion 对照 |
| **P112** | ✅ A 核心 | 核心-cross-fusion 分割 | CFFormer（Zhao 2025, *IEEE TGRS*）— optical+SAR/DSM dual-stream BiFormer；FCM 模态校正 + FFM cross-attention；支撑 M4 EO 融合结构与校正层设计 |
| **P114** | ✅ A 核心 | 核心-shared+specific + 缺失模态 | ShaSpec（Wang et al. 2024, arXiv:2307.14126v2）— shared + specific encoder + residual fusion；DAO/DCO；训练/测试缺失模态；shared 平均补足（非重建） |
| **P096** | ✅ A 核心 | 核心-多模态融合 | GMU（Arevalo 2017, ICLR-W）— 门控多模态单元，Gated Fusion 主模型依据 |
| **P097** | ✅ A 核心 | 核心-缺失模态 | Ma 2022（CVPR）— 多模态 Transformer 缺失模态鲁棒性，modality masking |
| **P098** | ✅ A 核心 | 核心-缺失 / 异步 | GRU-D（Che 2018, Sci Rep）— mask + time-interval 建模缺失，对应 age / valid_mask |
| **P099** | ✅ A 核心 | 核心-不规则时序 | mTAN（Shukla & Marlin 2021, ICLR）— 连续时间嵌入 + 时间注意力，异步对齐 |
| **P100** | ✅ A 核心 | 核心-缺失模态 | ModDrop（Neverova 2016, TPAMI）— modality dropout + 分模态预训练 |
| **P101** | ✅ A 核心 | 核心-多模态综述 | Baltrušaitis 2019（TPAMI）— 多模态 ML 分类框架，early vs representation fusion 术语锚点 |
| **P115** | ✅ A 核心 | 核心-MM-RSFM 综述 | Zhou, Qian & Gamba 2025（*Remote Sensing*）— MM-RSFM 分类与数据集地图；frozen/fine-tuned RS encoder 范式；M4 EO 模态 taxonomy 与局限 |

---

## 二、汇总统计


| 状态                     | 数量    | 涉及论文                                                                          |
| ---------------------- | ----- | ----------------------------------------------------------------------------- |
| ✅ 核心/重要/方法             | 112 篇 | P001–P006, P014–P033, P036–P037, P039, P041–P050, P062–P067, P068–P093, P094–P101, P102–P135, P052–P060（去重） |
| ⚠️ 边缘/备用               | 9 篇   | P034, P035, P038, P046, P051, P082, P083, P087, P088, P090                    |
| 📋 备查（Text/NLP，不在核心范围） | 8 篇   | P007–P013, P040                                                               |
| 📦 纯数据集                 | 4 项   | P021, P028, P029, P046                              |
| 📄 业界/机构 Report         | 21 项  | R001–R021（M1 数据字典 §4.6；按 URL 去重） |
| **合计**                    | **156** | P001–P135（135 篇论文）+ R001–R021（21 项 Report）         |


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

# 论文阅读优先级清单（按重要性排序，共 135 篇）

> **排序逻辑：** 油价理论基础 → 核心模型与评估方法 → 三模态变量锚点 → 模型架构原始论文 → 模态补充 → 方法参考 → 背景/数据集/备查
>
> **编号说明：** P001–P135 连续编号（P102–P135 = M1 数据字典支撑文献；P094–P101 + **§⑥ P103–P115** = ⑥ 创新/集成层；⚠️ §⑥ 同号条目 ≠ §① 同号条目）

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


## Tier 3 — 重要补充（航运/遥感理论/GNN 应用，序 31–52）


| 序   | ID   | 论文                                              | 理由                       |
| --- | ---- | ----------------------------------------------- | ------------------------ |
| 31  | P015 | Yan et al. (2020) — Global Marine Oil           | AIS 全球石油贸易分析             |
| 32  | P060 | OECD (2024) — An Ocean of Data                  | AIS 数据用于贸易/港口，Data Section |
| 33  | P050 | Niu et al. (2023) — Trade Network               | 原油贸易网络时空动态               |
| 34  | P078 | Levin et al. — NTL Review                       | NTL 综述与展望                |
| 35  | P033 | IMF (2024) — Satellite Monitoring               | 卫星衍生经济指标概览               |
| 36  | P080 | Dasgupta (2022) — NTL Supply–Demand Shocks      | NTL 短期供需冲击 proxy         |
| 37  | P049 | Kruitwagen et al. (2021) — Global Oil Assets    | 遥感识别全球石油基础设施             |
| 38  | P112 | Scheitrum et al. (2018) — WTI–Brent Spread      | 价差 2011 结构断点，M1 `brent_wti_spread` |
| 39  | P113 | Bravo Caro et al. (2020) — WTI–Brent FCVAR      | 价差长记忆/全球化测度，与 P112/P102 联读 |
| 40  | P030 | Henderson et al. (2012) — NTL Economic Growth   | NTL 经济增长理论基础             |
| 41  | P072 | Costa (2021) — ML Oil Price Point & Density     | XGBoost/RF/SVR baseline  |
| 42  | P073 | Daneshvar (2022) — LSTM/Bi-LSTM Brent           | 直接预测 Brent               |
| 43  | P076 | Yılmaz (2026) — Strategic Risk Brent Returns  | GPR/VIX/美元等 M1 变量        |
| 44  | P077 | Chung (2024) — GARCH + ML Energy Volatility     | target_brent_vol 波动率支撑   |
| 45  | P002 | Luo et al. (2024) — ML Crude Volatility         | ML 波动率评估指标参考             |
| 46  | P064   | BiLSTM-GCN (2023)                               | GCN+RNN 预测原油价格           |
| 47  | P065   | Russian Oil GNN (2025)                          | GNN 预测石油贸易（制裁场景）         |
| 48  | P067   | ITSG-LSTM (2025)                                | 贸易相似性图 + 港口吞吐量预测         |
| 49  | P085 | Carlini (2025) — Port Networks ML               | AIS+WPI 全球港口网络           |
| 50  | P086 | AIS-TGNN (2026) — Port Congestion TGAT          | ST-GNN+AIS 港口拥堵          |
| 51  | P041 | EMAT (2025)                                     | 多方面注意力 Transformer 金融预测  |
| 52  | P092 | ST-GRAT (2019)                                  | 时空图注意力，动态空间依赖            |


## Tier 4 — 方法参考（ST-GNN/ML/CV 补充，序 53–67）


| 序   | ID   | 论文                             | 理由                       |
| --- | ---- | ------------------------------ | ------------------------ |
| 53  | P047 | STGAT — 时空图注意力                 | STGAT 架构可迁移              |
| 54  | P093 | STGAT (2020) — Traffic Flow    | 时空图注意力交通流参考              |
| 55  | P043 | GNN Supply Chain               | 供应链图构建参考                 |
| 56  | P048 | ST-GNN Electricity Price       | 能源价格 ST-GNN 可迁移          |
| 57  | P003 | Directional Forecasting (2025) | 多模型方向预测对比（SVM/RF 等）      |
| 58  | P005 | Hybrid ML (2025)               | 混合 ML 方法参考               |
| 59  | P006 | MLP/CNN/Transformer (2024)   | 深度学习对比实验设计               |
| 60  | P075 | Cohen (2025) — Short-Term Oil  | 短期油价综合实验与评价指标            |
| 61  | P074 | Qin (2023) — ML + Google Search | 外部高频信息对照背景               |
| 62  | P020 | ShipRSImageNet Paper           | 船舶检测数据集论文                |
| 63  | P022 | Self-Supervised Ship ID (2026) | 卫星船舶识别                   |
| 64  | P081 | Li (2025) — Optical+SAR Dark Ship | 光学+SAR 协同暗船检测            |
| 65  | P026 | YOLOX Oil Tank (2022)          | SAR 油罐检测，FRT 补充          |
| 66  | P023 | Airbus Ship Detection          | 卫星船舶检测                   |
| 67  | P031 | Nordhaus & Chen (2015) — NTL   | NTL 精度评估                 |


## Tier 5 — 背景与边缘（序 68–82）


| 序   | ID   | 论文                          | 理由                       |
| --- | ---- | --------------------------- | ------------------------ |
| 68  | P079 | Doll (2008) — NTL Guide     | 早期 NTL 专题指南              |
| 69  | P036 | World Bank — NTL Proxy      | NTL 方法论总结                |
| 70  | P019 | Bayes — AIS & Freight Rates | AIS 与运费关系                |
| 71  | P090 | Informer (2021)             | 长序列 Transformer 备选（TFT 过重时） |
| 72  | P087 | Gouareb (2022) — Vessel Destination Graph ML | AIS 图结构背景               |
| 73  | P088 | TransES-ETA (2025)          | 航运 Transformer ETA 背景    |
| 74  | P082 | GFW — Dark Vessels Project  | AIS 盲区背景                 |
| 75  | P083 | GFW — VIIRS Dark Vessels    | VIIRS 暗船，Future Work     |
| 76  | P027 | YOLOv7 Storage Tank (2024)  | 大尺度遥感储罐检测                |
| 77  | P044 | Graph Neural Poisson        | GNN 供应链预测架构              |
| 78  | P045 | GAT-LSTM Supply Chain       | GAT+LSTM 时空融合            |
| 79  | P034 | Urban Markets Satellite     | 背景-卫星经济分析                |
| 80  | P035 | NTL & Export Growth         | 背景-NTL 与出口               |
| 81  | P038 | 50 Years NTL (2022)         | 背景-NTL 历史综述              |
| 82  | P051 | Downstream Oil Supply Chain | 图拓扑参考，优先级低               |


## Tier 6 — 纯数据集 📦（序 83–87）


| 序   | ID   | 数据集                         | 用途           |
| --- | ---- | --------------------------- | ------------ |
| 83  | P037 | NASA Black Marble           | 数据源产品文档      |
| 84  | P021 | ShipRSImageNet GitHub       | 船舶检测图像数据集    |
| 85  | P028 | Robinson/Meng — AST Dataset | 地上储油罐遥感数据集   |
| 86  | P029 | Rizk & Chehade — Oil Tank   | 油罐检测数据集      |
| 87  | P046 | SupplyGraph Benchmark       | 供应链 GNN 基准数据集 |


## Tier 7 — 备查 Text/NLP（已移出核心范围）📋（序 88–95）


| 序   | ID   | 论文                               | 在论文中的使用方式            |
| --- | ---- | -------------------------------- | --------------------- |
| 88  | P007 | LLM Sentiment WTI (2026)         | Lit Review 提及 LLM 方向  |
| 89  | P009 | OPEC+ News Predictor (2024)      | Lit Review 提及新闻信号     |
| 90  | P008 | News Sentiment Oil (2025)        | Discussion 未纳入文本可能是局限 |
| 91  | P011 | Global News Oil (2025)           | Future Work 文本模态      |
| 92  | P012 | News Volatility Direction (2025) | Discussion / Future Work |
| 93  | P013 | Temporal Semantic Fusion (2025)  | 同上                    |
| 94  | P010 | News + LLM Time Series (2024)    | 方法参考（未来加入 LLM）        |
| 95  | P040 | MM-iTransformer Text (2025)      | 多模态+文本融合备查            |


