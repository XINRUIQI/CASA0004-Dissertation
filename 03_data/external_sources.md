# External Data Sources

> 记录所有外部数据来源、获取方式和使用许可。
## M1 Core Variables — Dataset Sources

> 与 `01_literature/beatrice_task_literature_matrix.md` §①「M1 推荐变量」**序号一致**，共 10 项；按 Kilian 三类机制（供给 / 全球需求 / 预防性需求 + 市场金融条件）组织。
> 全部对齐周五截止周频（W-FRI），研究期 2006-01 ~ 2025-12。所有来源均公开免费、无需 API key。
>
> **构建分工：**
> - **已有**（#1–2, #6 中 `vix`）：`03_data/processed/build_weekly_time_index.py` → `weekly_time_index.csv`
> - **待构建**（#3–5, #6 中 `ovx`, #7–10）：`03_data/processed/build_m1_to_build.py` → `m1_to_build_weekly.csv`，再经 `merge_m1_to_build.py` 合并入主表

| 序 | 变量 | 机制 | 数据集 / 指标 | 提供方 | 标识符 (series/ticker) | URL | 原始频率 | → 周频处理 | 构建 | 覆盖起点 |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | `brent_price` lags + `brent_log_return` | 油价自身动态 | Europe Brent Spot FOB | EIA | 本地 `EIA_brent_spot_price_daily*.xls` | https://www.eia.gov/petroleum/gasdiesel/ | 日 | 周最后值；`brent_log_return = log(P_t / P_{t-1})`；滞后项建模阶段扩展 | 已有 | 2006-01 |
| 2 | `crude_stocks_change` 库存变化 | 供给 / 市场平衡 | Commercial Crude Stocks (excl. SPR) | EIA WPSR | 本地 `EIA_commercial_crude_stocks_weekly*.xls` | https://www.eia.gov/petroleum/supply/weekly/ | 周 | 对齐 W-FRI 后一阶差分 | 已有 | 2006-01 |
| 3 | `global_econ_activity`（Kilian 指数 / 全球 IP / PMI）| 全球需求 | Index of Global Real Economic Activity (Kilian) | Dallas Fed（备用 OECD CLI）| igrea (xlsx/csv)；备用 FRED `OECDLOLITOAASTSAM` | https://www.dallasfed.org/research/igrea | 月 | 月末 ffill + **5 周**发布滞后 | 待构建 | 2006-02 |
| 4 | `nonoil_industrial_commodity`（CRB 工业 / 金属）| 全球需求 | Global Price Index of Industrial Materials | IMF (via FRED) | FRED `PINDUINDEXM` | https://fred.stlouisfed.org/series/PINDUINDEXM | 月 | 月末 ffill + **5 周**发布滞后 | 待构建 | 2006-02 |
| 5 | `futures_spread` 期货–现货价差 | 市场紧张 / 预期 | Brent 近月期货 − Brent 现货（log 价差）| ICE (期货, Yahoo) + EIA (现货) | `BZ=F` − 本地 `brent_price`（备用 FRED `DCOILBRENTEU`）| https://finance.yahoo.com/quote/BZ%3DF | 日 | 周最后值，`log(fut) − log(spot)` | 待构建 | 2007-08 |
| 6 | `ovx`（优先）/ `vix` | 石油特定不确定性 | CBOE Crude Oil Volatility Index / CBOE VIX | CBOE (via Yahoo / FRED) | `^OVX`（备用 FRED `OVXCLS`）；FRED `VIXCLS` | https://finance.yahoo.com/quote/%5EOVX；https://fred.stlouisfed.org/series/VIXCLS | 日 | 周最后值 | `ovx` 待构建；`vix` 已有 | OVX 2007-05；VIX 2006-01 |
| 7 | `gpr` 地缘政治风险 | 预防性需求 | Geopolitical Risk Index | Caldara & Iacoviello (2022) | 文件 `data_gpr_export.xls`，列 `GPR` | https://www.matteoiacoviello.com/gpr.htm | 月 | 月末 ffill + **1 周**发布滞后 | 待构建 | 2006-01 |
| 8 | `dgs10_change`（ΔDGS10，10Y 收益率变化 / 一阶差分）| 利率 / 持有成本 | 美国 10 年期国债收益率变化 | FRED (Board of Governors) | 由本地 `treasury_10y`（FRED `DGS10`）派生 | https://fred.stlouisfed.org/series/DGS10 | 日 | 周最后值 → 一阶差分（**不用水平值**）| 待构建（源列已有）| 2006-01 |
| 9 | `gold_return`（`gold_price` 衍生）| 商品联动 / 避险 | LBMA Gold Price PM (USD) | ICE/LBMA (via FRED) | FRED `GOLDPMGBD228NLBM`（备用 yfinance `GC=F`）| https://fred.stlouisfed.org/series/GOLDPMGBD228NLBM | 日 | 周最后值 → 对数收益率 | 待构建 | 2006-01 |
| 10 | `commodity_fx`（CAD/AUD，优先于宽美元）| 汇率渠道 | 商品货币强度（CAD/USD、AUD/USD 均值）| Yahoo Finance（备用 FRED）| `CADUSD=X`, `AUDUSD=X`（备用 FRED `DEXCAUS`, `DEXUSAL`）| https://finance.yahoo.com/quote/CADUSD=X | 日 | 周最后值 → 两者周 % 变化均值 | 待构建 | 2006-01 |

**说明：**
- 序号与文献矩阵 §① 一一对应；#1 中 `brent_price` 滞后项在建模阶段扩展，不单列变量名（P076）。
- #6 文献建议 **OVX 优先于 VIX**（P052）；M1 同时保留两者，建模时可做共线性检查。
- #8 水平值 `treasury_10y`（DGS10）未通过单位根检验（P076），M1 使用一阶差分 `dgs10_change`。
- #10 宽口径美元指数（DXY）证据有限（P053/P004），改用商品出口国汇率 CAD/AUD。
- `ovx`、`futures_spread` 覆盖起点较晚（OVX 2007-05；Brent 期货 2007-08），早期周缺失属正常。
- `gpr` 为新闻文本聚合指数；文本模态已移除（Meeting 02），归入 M1 作低频地缘政治风险代理。
- `futures_spread` 为近月连续合约与现货之差的**近似**期限价差。
- 月频变量（#3, #4, #7）发布滞后为保守估计，避免前视偏差；进入模型前仍须统一滞后处理。
- 引用：GPR — Caldara, D. & Iacoviello, M. (2022) *Measuring Geopolitical Risk*, AER；Kilian REA — Kilian (2009)，Dallas Fed 维护更新版。

---

## M2 Remote Sensing Variables — Dataset Sources

> 与 `01_literature/beatrice_task_literature_matrix.md` §②「M2 推荐变量」对应。精读修正后采用**精简的三层设计**：① **动态 NTL 活动信号**（站点内部 z-score 异常，**非原始辐射水平**，因 P024/P032 指出 NTL 横截面强、时间弱且不识别油轮，Santos NTL↔油轮 Rs=−0.07）；② **遥感观测质量**（VIIRS 无云覆盖 / S2 无云观测，P025/P032：先作数据质量而非市场信息不确定性代理）；③ **S2 白天光学信息可得性**（云比例，P025 信息缺口机制）+ **静态容量权重**（P055 无法支撑充填率 → 改用 P024 横截面规模代理）。
> 覆盖 11 个油基础设施 AOI（中心点 + 5 km buffer，见 `aoi_oil_infrastructure.csv`），核心石油 AOI 为 Rotterdam / Fujairah / Ras Tanura / US Gulf(Houston)。对齐 W-FRI；VIIRS 自 2014、S2 自 2017。所有来源经 Google Earth Engine 公开免费获取（Code Editor 导出，无需 Python key）。
>
> **构建链路：**
> 1. GEE 导出（`03_data/raw/04_sentinel2/extract_*_gee.js`）→ 月频原始 CSV（VIIRS / Sentinel-2 / Landsat）。
> 2. `03_data/raw/04_sentinel2/build_m2_clean_features.py` → `weekly_m2_clean_features.csv`（44 列 = 4 类 × 11 AOI，月→周 ffill）+ `aoi_capacity_weights.csv`（静态）。
> 3. `04_code/scripts/build_feature_matrix.py` 合并入 `weekly_features.csv`，登记于 `feature_groups.json["M2_rs_clean"]`（44 个特征）。

| 数据集 / 指标 | 提供方 | 接口 / 标识 | 原始频率 | → 周频处理 | 派生特征（写入主表） | 覆盖起点 |
|---|---|---|---|---|---|---|
| **夜间灯光辐亮度**（`avg_rad` 均值 + `cf_cvg` 无云覆盖） | NASA/NOAA VIIRS DNB（via GEE） | `NOAA/VIIRS/DNB/MONTHLY_V1/VCMSLCFG`；11 AOI × 5 km | 月 | 月→日 ffill→W-FRI last | `ntl_anomaly_{aoi}`（站点扩展窗 z-score，过去-only，min 12 月）、`ntl_valid_obs_count_{aoi}`（=`cf_cvg`，数据质量） | 2014-01 |
| **Sentinel-2 地表光学 + 云概率**（`valid_obs_count`, `cloud_probability`） | Copernicus Sentinel-2（via GEE） | `COPERNICUS/S2_SR_HARMONIZED` + S2 cloud prob；11 AOI × 5 km | 月 | 月→日 ffill→W-FRI last | `s2_clear_obs_count_{aoi}`（白天光学观测可用性）、`s2_cloud_fraction_{aoi}`（=`cloud_probability`/100，信息缺口代理） | 2017-04 |
| **AOI 容量权重**（静态，规模代理） | 由 VIIRS 长期平均辐亮度派生 | `aoi_capacity_weights.csv`（11 AOI 归一化权重） | 静态 | — | `aoi_capacity_weight`（用于跨 AOI 容量加权聚合，**非高频特征**） | — |

**说明 / 注意（呼应文献矩阵 §② 精读修正）：**

- ⚠️ **用动态异常而非原始水平**：变量名由旧版 `ntl_ntl_avg_rad_mean_{P00x}` 改为 `ntl_anomaly_{aoi}`（站点 z-score），因 NTL 时间维度弱、原始水平受 AOI 规模与城市灯光污染影响（P024/P032）。
- ⚠️ **NTL 不是油轮代理**：Santos 实测 NTL↔油轮 Rs=−0.07 → 仅作综合锚泊/港口活动信号，建模时须结合油轮占比 / AIS（P024）。
- ⚠️ **VIIRS（非 DMSP）+ post-2012 子样本**（VCMSLCFG 2014 起）；月度合成已经 GEE 侧 mask；缺失仅用历史插补，避免 P024 双向插值的前视泄漏。
- ⚠️ **`valid_obs` 定位**：VIIRS `cf_cvg` 先作**数据质量**变量（夜间云量安慰剂检验不显著，不享有 P025 浮顶库存机制）；P025 的"信息可得性→收益率"机制对应 **S2 白天无云观测**（`s2_cloud_fraction`）。
- ⚠️ **不做浮顶充填率**：P055 估的是油罐**结构容量（V=πr²h）**、需亚米级影像（S2 10 m 不可复现），**不能支撑 `frt_fill_level`**；改用各 AOI 长期平均辐亮度作 **`aoi_capacity_weight`**（P024 横截面 NTL↔港口规模 Rs=0.69–0.84）。
- ⚠️ **样本期对齐**：含 `s2_*` 时 M2 周频样本自 2017 起；遥感为需求侧/上游信息代理（卫星活动→石油需求→供需预期→油价，P069），增量价值须由 "M1 vs M1+M2" 消融 + Clark–West/DM 检验证明,而非直接预测油价。
- 引用：NTL 航运代理 — Polinov, Bookman & Levin (2022)；NTL 数据选择 — Gibson, Olivia, Boe-Gibson & Li (2021)；云覆盖与油价信息 — Hao & Wang (2023)；卫星预测石油需求 — Bricongne et al. (2026, ECB WP 3198)；油罐容量遥感 — Wang, Li, Yu & Liu (2019)。

---

## M3 Shipping Variables — Dataset Sources

> 与 `beatrice_task_literature_matrix.md` §③「M3 推荐变量」对应。核心:tanker-specific（油轮专属）的**流量强度 + 运力(DWT)加权 + 平均船型 + 区域/咽喉拆分 + 出口-进口方向性不对称 + 拥堵代理**。全部对齐周五截止周频（W-FRI），覆盖 6 个油相关咽喉（Hormuz/Suez/Malacca/Bab el-Mandeb/Panama/Cape）。所有 PortWatch 来源公开免费、无需 key；GFW 需免费 API token。
>
> **构建链路：**
> 1. 原始下载（`03_data/raw/05_shipping/`）：PortWatch 咽喉/港口 + GFW presence。
> 2. `aggregate_shipping_to_weekly.py` → `03_data/processed/weekly_shipping_features.csv`（PortWatch 日→周 sum；GFW 月→周 ffill）。
> 3. 并入主表 `weekly_features.csv` / `weekly_features.parquet`，登记于 `feature_groups.json["M3_add_shipping"]`（共 119 个候选特征）。派生/同步脚本：`04_code/scripts/add_avg_tanker_size.py`、`04_code/scripts/sync_shipping_features.py`。

| 数据集 / 指标 | 提供方 | 接口 / 标识 | 原始频率 | → 周频处理 | 派生特征（写入主表） | 覆盖起点 |
|---|---|---|---|---|---|---|
| **咽喉过境**（按船型 `n_*`/`capacity_*`） | IMF PortWatch `Daily_Chokepoints_Data` | ArcGIS FeatureServer（org `weJ1QsnbMYJlCHdG`）；6 咽喉 portid | 日 | W-FRI 求和 | `pw_{choke}_n_tanker`、`_n_total`、`_capacity_tanker`、`_capacity`、`_tanker_share`、`_tanker_cap_share`、`_avg_tanker_size`(=cap/n)、`_n_tanker_wow_pct`、`_capacity_tanker_4w_ma`；汇总 `pw_all_n_tanker_sum`/`_n_total_sum`/`_tanker_share` | 2019-01 |
| **港口级进出口**（`import_tanker`/`export_tanker` 吨位） | IMF PortWatch `Daily_Ports_Data` | ArcGIS FeatureServer；14 油轮枢纽 portid | 日 | W-FRI 求和（按出口/进口篮子） | `pw_exp_hubs_export_vol`、`pw_imp_hubs_import_vol`、`pw_tanker_exp_imp_net`、`pw_tanker_exp_imp_asym`、`pw_tanker_exp_imp_log_ratio`、`pw_tanker_exp_imp_asym_4w_ma`、`pw_exp_hubs_export_vol_wow_pct` | 2019-01 |
| **AIS 船舶 presence**（按船型 hours/vessels） | Global Fishing Watch 4Wings `public-global-presence` | Report API（POST，Bearer token，6 咽喉多边形） | 月 | 月→日 ffill→W-FRI last | `gfw_{choke}_total_hours`、`_total_vessels`、`_cargo_hours`、`_bunker_hours`、`_other_hours`、`_nontanker_hours`、`_other_share`、`_total_hours_mom_pct`、**`_dwell_hours_per_vessel`(=hours/vessels 拥堵/停留代理)**；汇总 `gfw_all_total_hours_sum` | 2012-01 |

**出口/进口枢纽篮子（港口级方向性，`download_portwatch_ports.py`）：**

- **出口枢纽**：Ras Tanura、Juaymah、Yanbu（沙特）、Ras Laffan（卡塔尔）、Primorsk、Novorossiysk（俄）、Corpus Christi（美）、Sidi Kerir（埃及 SUMED）、Bonny（尼日利亚）。
- **进口/炼化枢纽**：Rotterdam（荷）、Singapore、Ningbo（中）、Chiba（日）、Ulsan（韩）。
- 角色经数据验证（出口枢纽 `export_tanker ≫ import_tanker`，进口枢纽相反）。

**说明 / 注意（呼应文献矩阵 §③ 精读修正）：**

- ⚠️ `pw_{choke}_n_tanker` 是**油轮交通强度 / 海运流量粗代理**，非 P016 的 port-call frequency、非精确石油贸易量（P016/P018）。
- ⚠️ **方向性**用 PortWatch **港口级** `import_tanker`/`export_tanker`（吨位估计，源自 AIS 吃水，不内嵌油价 → 防泄漏）；PortWatch **咽喉数据无方向字段**，故方向性不可由咽喉数据构造（P018）。
- ⚠️ **拥堵**无专用锚地等待数据，用 GFW presence 的 `total_hours/total_vessels`（每船停留时长）作 dwell 代理（P016 dwell-time 概念）。
- ⚠️ **样本期对齐**：PortWatch 系列 2019+，GFW 系列 2012+；油价方向为 price→shipping（P016/P017），建模须**严格滞后、按发布时点对齐、后向滚动**（避免 P018 中心 MA 前视泄漏）。
- 引用：PortWatch — Arslanalp, Exton, Gao, Kamali, Saraiva, Sozzi & Verschuur (2026, IMF WP/26/99)；Arslanalp, Marini & Tumbarello (2019, IMF WP/19/275)。GFW — Global Fishing Watch 4Wings API。油价—航运关系 — Mi et al. (2022, 2023)。
