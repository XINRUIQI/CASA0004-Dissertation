# 特征清单 — 多模态周频特征矩阵

**数据集：** 1,043 周（2006-01-06 ~ 2025-12-26）| **263 个特征**，4 个模态 | **统一频率：** 周五截止的周频

---

## M1 — 市场 + 宏观基本面（27 个特征）

基准模态：布伦特/WTI 价格、EIA 周度石油状况报告、FRED 宏观金融指标。

| # | 特征名 | 描述 | 数据源 | 原始频率 | 覆盖率 |
|---|--------|------|--------|----------|--------|
| 1 | `brent_price` | 布伦特现货价格（美元/桶） | EIA | 日频→周频（取末值） | 100% |
| 2 | `wti_price` | WTI 库欣现货价格（美元/桶） | EIA | 日频→周频（取末值） | 100% |
| 3 | `brent_wti_spread` | 布伦特-WTI 价差 | 衍生 | 周频 | 100% |
| 4 | `brent_return_pct` | 布伦特周环比收益率（%） | 衍生 | 周频 | 100% |
| 5 | `wti_return_pct` | WTI 周环比收益率（%） | 衍生 | 周频 | 100% |
| 6 | `brent_log_return` | 布伦特对数收益率 | 衍生 | 周频 | 100% |
| 7 | `brent_vol_4w` | 4 周滚动波动率（对数收益率标准差） | 衍生 | 周频 | 100% |
| 8 | `brent_vol_12w` | 12 周滚动波动率 | 衍生 | 周频 | 100% |
| 9 | `crude_stocks_excl_spr` | 美国商业原油库存（不含 SPR，千桶） | EIA WPSR | 周频 | 100% |
| 10 | `cushing_stocks` | 库欣原油库存（千桶） | EIA WPSR | 周频 | 100% |
| 11 | `crude_production` | 美国原油产量（千桶/日） | EIA WPSR | 周频 | 100% |
| 12 | `crude_imports` | 美国原油进口（千桶/日） | EIA WPSR | 周频 | 100% |
| 13 | `crude_exports` | 美国原油出口（千桶/日） | EIA WPSR | 周频 | 100% |
| 14 | `refinery_crude_input` | 炼厂原油投入量（千桶/日） | EIA WPSR | 周频 | 100% |
| 15 | `refinery_utilisation` | 炼厂开工率（%） | EIA WPSR | 周频 | 100% |
| 16 | `gasoline_supplied` | 汽油供应量（千桶/日） | EIA WPSR | 周频 | 100% |
| 17 | `distillate_supplied` | 馏分油供应量（千桶/日） | EIA WPSR | 周频 | 100% |
| 18 | `jet_fuel_supplied` | 航空煤油供应量（千桶/日） | EIA WPSR | 周频 | 100% |
| 19 | `crude_stocks_change` | 原油库存周变动 | 衍生 | 周频 | 100% |
| 20 | `cushing_stocks_change` | 库欣库存周变动 | 衍生 | 周频 | 100% |
| 21 | `net_crude_trade` | 净原油贸易（进口减出口） | 衍生 | 周频 | 100% |
| 22 | `sp500` | 标普 500 指数收盘价 | Yahoo Finance | 日频→周频（取末值） | 100% |
| 23 | `vix` | CBOE 波动率指数（VIX）收盘价 | FRED | 日频→周频（取末值） | 100% |
| 24 | `dollar_index` | 贸易加权美元指数（DTWEXBGS） | FRED | 日频→周频（取末值） | 100% |
| 25 | `treasury_10y` | 10 年期美国国债收益率（%） | FRED | 日频→周频（取末值） | 100% |
| 26 | `fed_funds_rate` | 有效联邦基金利率（%） | FRED | 日频→周频（取末值） | 100% |
| 27 | `sp500_return_pct` | 标普 500 周收益率（%） | 衍生 | 周频 | 99.9% |

---

## M2 — 文本 / 事件信号（+26 个特征 → 累计 53）

GDELT 1.0/2.0 校准日频事件聚合至周频。两个事件域：石油扰动事件强度、运输中断信号。

| # | 特征名 | 描述 | 数据源 | 聚合方式 | 覆盖率 |
|---|--------|------|--------|----------|--------|
| 1 | `gdelt_oil_geo_event_count` | 石油扰动事件周计数 | GDELT | 日频求和 | 100% |
| 2 | `gdelt_oil_geo_total_mentions` | 石油扰动事件媒体提及总数 | GDELT | 日频求和 | 100% |
| 3 | `gdelt_oil_geo_avg_tone` | 石油扰动事件平均媒体语调 | GDELT | 日频均值 | 100% |
| 4 | `gdelt_oil_geo_avg_goldstein` | 石油扰动事件平均 Goldstein 分数 | GDELT | 日频均值 | 100% |
| 5 | `gdelt_negative_event_count` | 负面石油扰动事件计数 | GDELT | 日频求和 | 100% |
| 6 | `gdelt_conflict_event_count` | 冲突类石油扰动事件计数 | GDELT | 日频求和 | 100% |
| 7 | `gdelt_sanction_event_count` | 制裁类石油扰动事件计数 | GDELT | 日频求和 | 100% |
| 8 | `gdelt_key_oil_region_event_count` | 关键产油区事件计数 | GDELT | 日频求和 | 100% |
| 9 | `gdelt_transport_disruption_event_count` | 运输中断事件周计数 | GDELT | 日频求和 | 100% |
| 10 | `gdelt_transport_disruption_total_mentions` | 运输中断事件媒体提及数 | GDELT | 日频求和 | 100% |
| 11 | `gdelt_transport_disruption_avg_tone` | 运输中断事件平均语调 | GDELT | 日频均值 | 100% |
| 12 | `gdelt_transport_disruption_avg_goldstein` | 运输中断事件平均 Goldstein 分数 | GDELT | 日频均值 | 100% |
| 13 | `gdelt_transport_negative_event_count` | 负面运输事件计数 | GDELT | 日频求和 | 100% |
| 14 | `gdelt_transport_unrest_conflict_event_count` | 动乱/冲突运输事件计数 | GDELT | 日频求和 | 100% |
| 15 | `gdelt_transport_sanction_event_count` | 制裁类运输事件计数 | GDELT | 日频求和 | 100% |
| 16 | `gdelt_chokepoint_event_count` | 咽喉航道相关事件计数 | GDELT | 日频求和 | 100% |
| 17 | `gdelt_oil_geo_event_count_4w_ma` | 石油扰动事件 4 周滚动均值 | 衍生 | 滚动 | 100% |
| 18 | `gdelt_oil_geo_event_count_wow_pct` | 石油扰动事件环比变化率（%） | 衍生 | — | 99.9% |
| 19 | `gdelt_oil_geo_negative_share` | 负面事件占比（石油扰动） | 衍生 | — | 100% |
| 20 | `gdelt_oil_geo_conflict_share` | 冲突事件占比（石油扰动） | 衍生 | — | 100% |
| 21 | `gdelt_oil_geo_avg_tone_4w_ma` | 石油扰动语调 4 周滚动均值 | 衍生 | 滚动 | 100% |
| 22 | `gdelt_transport_event_count_4w_ma` | 运输事件 4 周滚动均值 | 衍生 | 滚动 | 100% |
| 23 | `gdelt_transport_event_count_wow_pct` | 运输事件环比变化率（%） | 衍生 | — | 99.9% |
| 24 | `gdelt_transport_negative_share` | 负面事件占比（运输） | 衍生 | — | 100% |
| 25 | `gdelt_transport_avg_tone_4w_ma` | 运输语调 4 周滚动均值 | 衍生 | 滚动 | 100% |
| 26 | `gdelt_combined_event_count` | 石油扰动 + 运输事件合计 | 衍生 | — | 100% |

---

## M3 — 遥感（+110 个特征 → 累计 163）

10 种特征类型 × 11 个 AOI 站点 = 110 个特征。光学指数来自 Sentinel-2（2017–2025）和 Landsat（2006–2017）回填；夜间灯光来自 VIIRS DNB（2012–2025）。

**11 个 AOI 站点：**

| ID | 站点 | 类型 | 国家 |
|----|------|------|------|
| P001 | 鹿特丹港 | 港口 | 荷兰 |
| P002 | 富查伊拉石油码头 | 码头 | 阿联酋 |
| P003 | 拉斯塔努拉码头 | 码头 | 沙特阿拉伯 |
| P004 | 裕廊岛 | 炼厂 | 新加坡 |
| P005 | 休斯顿航道 | 港口 | 美国 |
| P006 | 宁波-舟山港 | 港口 | 中国 |
| P007 | 贾姆纳格尔炼厂 | 炼厂 | 印度 |
| P008 | 巴士拉石油码头 | 码头 | 伊拉克 |
| P009 | 蔚山炼厂 | 炼厂 | 韩国 |
| P010 | 哈格岛码头 | 码头 | 伊朗 |
| P011 | 延布出口码头 | 码头 | 沙特阿拉伯 |

**逐 AOI 特征类型（每类 ×11 站点）：**

| # | 特征模式 | 描述 | 数据源 | 频率 | 覆盖率 |
|---|---------|------|--------|------|--------|
| 1 | `opt_NDVI_{site}` | 归一化植被指数 | Sentinel-2 / Landsat | 月频→周频（前填充） | 99–100% |
| 2 | `opt_NDWI_{site}` | 归一化水体指数 | Sentinel-2 / Landsat | 月频→周频（前填充） | 99–100% |
| 3 | `opt_NDBI_{site}` | 归一化建成区指数 | Sentinel-2 / Landsat | 月频→周频（前填充） | 99–100% |
| 4 | `opt_BSI_{site}` | 裸土指数 | Sentinel-2 / Landsat | 月频→周频（前填充） | 99–100% |
| 5 | `opt_valid_obs_count_{site}` | 每月有效观测计数 | Sentinel-2 / Landsat | 月频→周频 | 100% |
| 6 | `opt_sensor_flag_{site}` | 传感器来源（0=Landsat, 1=S2） | 衍生 | 月频→周频 | 100% |
| 7 | `ntl_ntl_avg_rad_mean_{site}` | VIIRS 夜间灯光平均辐射亮度 | VIIRS DNB | 月频→周频（前填充） | 60% |
| 8 | `ntl_ntl_avg_rad_max_{site}` | VIIRS 夜间灯光最大辐射亮度 | VIIRS DNB | 月频→周频（前填充） | 60% |
| 9 | `ntl_ntl_avg_rad_stddev_{site}` | VIIRS 夜间灯光辐射标准差 | VIIRS DNB | 月频→周频（前填充） | 60% |
| 10 | `ntl_ntl_cf_cvg_mean_{site}` | VIIRS 无云覆盖比例 | VIIRS DNB | 月频→周频（前填充） | 60% |

> 注：NTL（VIIRS）覆盖率为 60%，因为 VIIRS 数据从 2012-04 开始，覆盖 1,043 周中的 ~626 周。

---

## M4 — 航运（+100 个特征 → 累计 263）

6 条石油关键咽喉航道的海运活动。GFW 4Wings 船舶在场数据（月频，2012–2018）+ IMF PortWatch 日频过境数据（2019–2025）组合。

**6 条咽喉航道：** 霍尔木兹海峡、苏伊士运河、马六甲海峡、曼德海峡、巴拿马运河、好望角。

**GFW 特征（8 类 × 6 航道 + 1 汇总 = 49）：**

| # | 特征模式 | 描述 | 数据源 | 频率 | 覆盖率 |
|---|---------|------|--------|------|--------|
| 1 | `gfw_{choke}_total_hours` | 船舶在场总时数 | GFW 4Wings | 月频→周频（前填充） | 69.7% |
| 2 | `gfw_{choke}_total_vessels` | 不同船舶总数 | GFW 4Wings | 月频→周频 | 69.7% |
| 3 | `gfw_{choke}_cargo_hours` | 货船在场时数 | GFW 4Wings | 月频→周频 | 69.7% |
| 4 | `gfw_{choke}_bunker_hours` | 加油船在场时数 | GFW 4Wings | 月频→周频 | 69.7% |
| 5 | `gfw_{choke}_other_hours` | 其他船舶在场时数 | GFW 4Wings | 月频→周频 | 69.7% |
| 6 | `gfw_{choke}_nontanker_hours` | 非油轮船舶在场时数 | GFW 4Wings | 月频→周频 | 69.7% |
| 7 | `gfw_{choke}_other_share` | 其他船舶占比 | 衍生 | 月频→周频 | 69.7% |
| 8 | `gfw_{choke}_total_hours_mom_pct` | 船舶时数月环比变化率（%） | 衍生 | 月频→周频 | 69.3% |
| 9 | `gfw_all_total_hours_sum` | 全航道船舶总时数 | 衍生 | 周频 | 69.7% |

**PortWatch 特征（8 类 × 6 航道 + 3 汇总 = 51）：**

| # | 特征模式 | 描述 | 数据源 | 频率 | 覆盖率 |
|---|---------|------|--------|------|--------|
| 1 | `pw_{choke}_n_tanker` | 油轮过境周计数 | IMF PortWatch | 日频求和→周频 | 35.0% |
| 2 | `pw_{choke}_n_total` | 所有船舶过境周计数 | IMF PortWatch | 日频求和→周频 | 35.0% |
| 3 | `pw_{choke}_capacity_tanker` | 油轮周运力（DWT） | IMF PortWatch | 日频求和→周频 | 35.0% |
| 4 | `pw_{choke}_capacity` | 所有船舶周运力（DWT） | IMF PortWatch | 日频求和→周频 | 35.0% |
| 5 | `pw_{choke}_tanker_share` | 油轮过境占比 | 衍生 | 周频 | 35.0% |
| 6 | `pw_{choke}_tanker_cap_share` | 油轮运力占比 | 衍生 | 周频 | 35.0% |
| 7 | `pw_{choke}_n_tanker_wow_pct` | 油轮过境周环比变化率（%） | 衍生 | 周频 | 34.9% |
| 8 | `pw_{choke}_capacity_tanker_4w_ma` | 油轮运力 4 周滚动均值 | 衍生 | 周频 | 35.0% |
| 9 | `pw_all_n_tanker_sum` | 全航道油轮过境总数 | 衍生 | 周频 | 35.0% |
| 10 | `pw_all_n_total_sum` | 全航道船舶过境总数 | 衍生 | 周频 | 35.0% |
| 11 | `pw_all_tanker_share` | 全航道油轮占比 | 衍生 | 周频 | 35.0% |

> 注：GFW 覆盖 2012–2018（69.7%）；PortWatch 覆盖 2019–2025（35.0%）。两者组合覆盖整个研究期间的航运活动。

---

## 目标变量

| 特征名 | 描述 | 任务类型 | 定义 |
|--------|------|----------|------|
| `target_brent_price_next_1w` | 下周布伦特价格（美元/桶） | 回归 | 下一周五的收盘价 |
| `target_brent_vol_next_1w` | 下周已实现波动率 | 回归 | 下一周 5 个交易日对数收益率的标准差 |
| `target_brent_direction_next_1w` | 下周价格方向（三分类） | 分类 | 1=涨（>+0.5%）, 0=持平（±0.5%）, −1=跌（<−0.5%） |

---

## 消融实验设计

| 实验 | 包含模态 | 累计特征数 |
|------|---------|-----------|
| **M1** | 市场 + 宏观 | 27 |
| **M2** | M1 + 文本/GDELT | 53 |
| **M3** | M2 + 遥感 | 163 |
| **M4** | M3 + 航运 | 263 |
