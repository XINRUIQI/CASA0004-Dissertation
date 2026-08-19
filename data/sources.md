# External Data Sources

> 记录所有外部数据来源、获取方式和使用许可。

## M1 Core Variables — Dataset Sources

> 与 `01_literature/beatrice_task_literature_matrix.md` §①「M1 推荐变量」**序号一致**，共 10 项；按 Kilian 三类机制（供给 / 全球需求 / 预防性需求 + 市场金融条件）组织。
> 全部对齐周五截止周频（W-FRI），研究期 2006-01 ~ 2025-12。所有来源均公开免费、无需 API key。
>
> **构建分工（两层管线：RAW 按数据提供方分目录 → PROCESSED 单表）：**
>
> - **① RAW 原始层**（`data/raw/01_market_financial/`，按提供方分 `EIA/`、`FRED/`、`Yahoo/`、`Other/`）：统一入口 `download_m1_raw.py` —— `EIA/` 为手动网站导出 `.xls`（只登记不下载），`FRED/`、`Yahoo/`、`Other/` 自动下载并缓存；落盘后写根目录 `manifest.csv`（字段：`variable, raw_file, kind, category, source, series_id_or_ticker, source_url, frequency, native_unit, download_utc, n_rows, coverage_start, coverage_end, sha256, notes`；其中 `category` 即提供方 `eia/fred/yahoo/other/local`）以便离线复现与来源审计。
> - **② PROCESSED 构建层**（`data/processed/M1/`）：`py/build_m1_weekly.py`（**默认离线，只读本地 raw**）→ `outputs/m1_weekly_features.csv`（**单表 35 列周频**：基础锚点 + 9 个派生变量 + EIA 成品油供应等扩展列，所有列平级，无中间表、无单独 merge 步骤）。`--online` 仅在某原始文件缺失时临时联网；`--refresh-raw` 先调用 `download_m1_raw.py` 再构建；`--base-only` 只建基础锚点、跳过 8 个派生列。
> - **本地派生（不另存 raw，manifest 记 `kind=local`）**：`#8 dgs10_change` 由本地 `treasury_10y`（`FRED/` 的 `DGS10`）一阶差分；`#5 brent_f1_spot_log_basis` 现货端取本地 `brent_price`（`EIA/Brent` 日度），仅期货端 `BZ=F` 落盘 `Yahoo/`。


| 序   | 变量                                              | 机制        | 数据集 / 指标                                        | 提供方                         | 标识符 (series/ticker)                                                             | URL                                                                                                                                                                  | 原始频率 | → 周频处理                                                 | 构建                                                                   | 覆盖起点                    |
| --- | ----------------------------------------------- | --------- | ----------------------------------------------- | --------------------------- | ------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---- | ------------------------------------------------------ | -------------------------------------------------------------------- | ----------------------- |
| 1   | `brent_price` lags + `brent_log_return`         | 油价自身动态    | Europe Brent Spot FOB                           | EIA                         | 本地 `EIA/Brent/EIA_brent_spot_price_daily*.xls`                                  | [https://www.eia.gov/petroleum/gasdiesel/](https://www.eia.gov/petroleum/gasdiesel/)                                                                                 | 日    | 周最后值；`brent_log_return = log(P_t / P_{t-1})`；滞后项建模阶段扩展 | 已有                                                                   | 2006-01                 |
| 2   | `crude_stocks_change` 库存变化                      | 供给 / 市场平衡 | Commercial Crude Stocks (excl. SPR)             | EIA WPSR                    | 本地 `EIA/Weekly Petroleum Status Report/EIA_commercial_crude_stocks_weekly*.xls` | [https://www.eia.gov/petroleum/supply/weekly/](https://www.eia.gov/petroleum/supply/weekly/)                                                                         | 周    | 对齐 W-FRI 后一阶差分                                         | 已有                                                                   | 2006-01                 |
| 3   | `global_econ_activity`（Kilian 指数 / 全球 IP / PMI） | 全球需求      | Index of Global Real Economic Activity (Kilian) | Dallas Fed（备用 OECD CLI）     | igrea (xlsx/csv)；备用 FRED `OECDLOLITOAASTSAM`                                    | [https://www.dallasfed.org/research/igrea](https://www.dallasfed.org/research/igrea)                                                                                 | 月    | 月末 ffill + **5 周**发布滞后                                 | 已构建（`Other/DallasFed_igrea_monthly.xlsx`）                            | 2006-02                 |
| 4   | `nonoil_industrial_commodity`（CRB 工业 / 金属）      | 全球需求      | Global Price Index of Industrial Materials      | IMF (via FRED)              | FRED `PINDUINDEXM`                                                              | [https://fred.stlouisfed.org/series/PINDUINDEXM](https://fred.stlouisfed.org/series/PINDUINDEXM)                                                                     | 月    | 月末 ffill + **5 周**发布滞后                                 | 已构建（`FRED/FRED_PINDUINDEXM_monthly.csv`）                             | 2006-02                 |
| 5   | `brent_f1_spot_log_basis` 前月期货–现货对数基差（+ `brent_roll_week` 换月哑变量） | 市场紧张 / 预期 | Brent 前月期货 − Brent 现货（对数 basis，非纯期限结构）           | ICE (期货, Yahoo) + EIA (现货)  | `BZ=F` − 本地 `brent_price`（未 back-adjust；备用 FRED `DCOILBRENTEU`）                | [https://finance.yahoo.com/quote/BZ%3DF](https://finance.yahoo.com/quote/BZ%3DF)                                                                                     | 日    | 各取周五最后值，`log(fut) − log(spot)`                         | 已构建（`Yahoo/Yahoo_BZF_daily.csv` + 本地 `brent_price`）                  | 2007-08                 |
| 6   | `ovx`（优先）/ `vix`                                | 石油特定不确定性  | CBOE Crude Oil Volatility Index / CBOE VIX      | CBOE (via Yahoo / FRED)     | `^OVX`（备用 FRED `OVXCLS`）；FRED `VIXCLS`                                          | [https://finance.yahoo.com/quote/%5EOVX；https://fred.stlouisfed.org/series/VIXCLS](https://finance.yahoo.com/quote/%5EOVX；https://fred.stlouisfed.org/series/VIXCLS) | 日    | 周最后值                                                   | `ovx` 已构建（`Yahoo/Yahoo_OVX_daily.csv`）；`vix` 来自 `FRED/VIXCLS`        | OVX 2007-05；VIX 2006-01 |
| 7   | `gpr` 地缘政治风险                                    | 预防性需求     | Geopolitical Risk Index（日度 GPRD）              | Caldara & Iacoviello (2022) | 文件 `data_gpr_daily_recent.xls`，列 `GPRD`                                         | [https://www.matteoiacoviello.com/gpr.htm](https://www.matteoiacoviello.com/gpr.htm)                                                                                 | 日    | 周均值 + **1 周**发布滞后                                     | 已构建（`Other/data_gpr_daily_recent.xls`；旧月度 `.dta` 作 fallback）        | 2006-01                 |
| 8   | `dgs10_change`（ΔDGS10，10Y 收益率变化 / 一阶差分）         | 利率 / 持有成本 | 美国 10 年期国债收益率变化                                 | FRED (Board of Governors)   | 由本地 `treasury_10y`（FRED `DGS10`）派生                                              | [https://fred.stlouisfed.org/series/DGS10](https://fred.stlouisfed.org/series/DGS10)                                                                                 | 日    | 周最后值 → 一阶差分（**不用水平值**）                                 | 已构建（本地派生，`kind=local`）                                               | 2006-01                 |
| 9   | `gold_return`（`gold_price` 衍生）                  | 商品联动 / 避险 | LBMA Gold Price PM (USD)                        | ICE/LBMA (via FRED)         | FRED `GOLDPMGBD228NLBM`（备用 yfinance `GC=F`）                                     | [https://fred.stlouisfed.org/series/GOLDPMGBD228NLBM](https://fred.stlouisfed.org/series/GOLDPMGBD228NLBM)                                                           | 日    | 周最后值 → 对数收益率                                           | 已构建（`Yahoo/Yahoo_GCF_daily.csv`，当前快照，见下注）                            | 2006-01                 |
| 10  | `cadusd_log_return`（CAD 单腿，优先于宽美元）              | 汇率渠道      | 商品货币强度（CAD/USD 对数收益）                          | Yahoo Finance（备用 FRED）      | `CADUSD=X`（备用 FRED `DEXCAUS` 取倒数）                                              | [https://finance.yahoo.com/quote/CADUSD=X](https://finance.yahoo.com/quote/CADUSD=X)                                                                                 | 日    | 周最后值 → 对数收益（AUD 腿删，留稳健性）                             | 已构建（`Yahoo/Yahoo_CADUSD_daily.csv`）                                | 2006-01                 |


**说明：**

- 序号与文献矩阵 §① 一一对应；#1 中 `brent_price` 滞后项在建模阶段扩展，不单列变量名（P076）。
- #6 文献建议 **OVX 优先于 VIX**（P052）；M1 同时保留两者，建模时可做共线性检查。
- #8 水平值 `treasury_10y`（DGS10）未通过单位根检验（P076），M1 使用一阶差分 `dgs10_change`。
- #10 宽口径美元指数（DXY）证据有限（P053/P004），改用商品出口国汇率 CAD/AUD。
- `ovx`、`brent_f1_spot_log_basis` 覆盖起点较晚（OVX 2007-05；Brent 期货 2007-08），早期周缺失属正常。
- M1 覆盖窗口：2006-01-06 ~ 2025-12-26
- `gpr` 为新闻文本聚合指数；文本模态已移除（Meeting 02），归入 M1 作低频地缘政治风险代理。
- `brent_f1_spot_log_basis`（原 `futures_spread`）为前月连续合约与现货之差的**对数基差 proxy**（非纯期限结构；未 back-adjust，配 `brent_roll_week` 控制换月）。
- 月频变量（#3, #4, #7）发布滞后为保守估计，避免前视偏差；进入模型前仍须统一滞后处理。
- **扩展列（不属 Core-10，M1 需求侧扩展特征）**：EIA WPSR 成品油供应 `gasoline_supplied`（汽油）/ `distillate_supplied`（馏分油/柴油）/ `jet_fuel_supplied`（喷气燃料），均为美国周频需求代理（千桶/日），落盘 `EIA/Weekly Petroleum Status Report/`，已并入产出表；连同基础锚点与 9 个派生列，`m1_weekly_features.csv` 共 **35 列**（相对早期版本已移除已实现波动率 `brent_vol_4w/12w`、方向 `brent_direction`、简单收益率 `brent_return_pct`、净贸易 `net_crude_trade`、`sp500` 指数水平；`wti_return_pct`/`sp500_return_pct`→对数收益，`commodity_fx`→`cadusd_log_return`（删 AUD 腿），`futures_spread`→`brent_f1_spot_log_basis` + 新增 `brent_roll_week`，`gpr` 改日度 GPRD 周均值；理由见研究日志 2026-06-23 / 2026-07-03）。
- **原始层（按提供方分目录，离线复现 + 来源审计）**：
  - `EIA/`（手动 `.xls`，只登记不下载）：`EIA/Brent/EIA_brent_spot_price_daily*.xls`、`EIA/WTI/EIA_WTI_cushing_crude_price_daily*.xls`、`EIA/Weekly Petroleum Status Report/EIA_*_weekly*.xls`（库存 / 产量 / 进出口 / 炼厂利用率 + `gasoline_supplied` / `distillate_supplied` / `jet_fuel_supplied`）。
  - `FRED/`（自动下载）：`FRED_*_DGS10_*.csv`（10Y 收益率）、`FRED_*_VIXCLS_*.csv`（VIX）、`FRED_*_DTWEXBGS_*.csv`（美元指数）、`FRED_*_DFF_*.csv`（联邦基金利率）、`FRED_PINDUINDEXM_monthly.csv`（IMF 工业原料）。
  - `Yahoo/`（自动下载）：`Yahoo_sp500_daily*.csv`（S&P 500）、`Yahoo_OVX_daily.csv`（OVX）、`Yahoo_BZF_daily.csv`（Brent 近月期货）、`Yahoo_CADUSD_daily.csv`（商品货币 CAD；`Yahoo_AUDUSD_daily.csv` 仍在但 AUD 腿已不入表）、`Yahoo_GCF_daily.csv`（黄金，见下注）。
  - `Other/`（自动下载）：`DallasFed_igrea_monthly.xlsx`（Kilian REA）、`data_gpr_daily_recent.xls`（日度 GPRD，列 `GPRD`）、`data_gpr_export.dta`（旧月度 GPR，fallback）。
  - 根目录 `manifest.csv` 记录每个文件的 `category`（提供方）、来源 URL、标识、下载时间、行数、覆盖区间与 SHA-256。
- ⚠️ `**gold_return` 实际快照源**：FRED 序列 `GOLDPMGBD228NLBM`（LBMA Gold PM）近期在 FRED 端不稳定（404/超时），当前快照由备用源 **Yahoo `GC=F`（COMEX 黄金期货）** 落盘于 `Yahoo/Yahoo_GCF_daily.csv`；两者的周度对数收益率经核对一致（max|Δ|≈3e-16）。canonical 主源仍记为 FRED，`manifest.csv` 如实记录实际下载源。
- 引用：GPR — Caldara, D. & Iacoviello, M. (2022) *Measuring Geopolitical Risk*, AER；Kilian REA — Kilian (2009)，Dallas Fed 维护更新版。

---

## M2 Remote Sensing Variables — Dataset Sources

> 与端到端多模态方案 `00_admin/2026-06-22_research_plan_e2e_multimodal.md` §4.2 对齐。遥感不再压成扁平指标列，而采用**双通道设计**（表示学习 + 经济解释并存）：
>
> - **通道 A — 影像表示（核心创新）**：导出 11 AOI × 月度 **Sentinel-2 patch**（6 波段 B2/B3/B4/B8A/B11/B12 + SCL 云掩膜），喂**冻结预训练 EO 大模型**（Prithvi-EO-2.0 / SatMAE，只提特征不微调）得到每张 patch 的 image embedding。
> - **通道 B — 机制变量（经济解释）**：保留 VIIRS NTL + S2 派生的 NDVI/NDWI/NDBI/BSI 等人工指标与云比例，作站点活动 / 信息可得性代理（沿用旧版三层设计的机制解释，但降为辅助通道）。
> - 覆盖 11 个油基础设施 AOI（见 `aoi_oil_infrastructure.csv`，逐站档案见 `aoi_oil_infrastructure_sites.md`；核心石油 AOI 为 Rotterdam / Fujairah / Ras Tanura / Houston）。**按站点类型差异化 patch**（默认 port 6.4 km / refinery 5.12 km / terminal 2.56 km，并按目视校核做单站微调：Fujairah / Kharg / Yanbu 放大到 3.2 km，Basra 缩小到 1.6 km；实际 half-size 取 CSV 的 `patch_half_m`），统一窗口 **2019-01 ~ 2026-06**（与 M0–M4 标准化比较窗对齐）。所有来源经 Google Earth Engine 公开免费获取（Code Editor 导出，无需 Python key）。
>
> **构建链路（AOI 配置：`aoi_oil_infrastructure.csv` → `sync_aoi_csv_to_gee.py` 生成共享的 `load_aoi_config_gee.js`；各脚本另有内联 AOI 的 `*_bundled.js` 独立版，可单文件粘贴进 GEE 运行）：**
>
> 1. **通道 A**（`Channel A/`）：GEE 导出 `data/raw/02_sentinel2/Channel A/export_s2_patches_multimodal_gee.js` → 每个 (site, month) 一张 6 波段（B2/B3/B4/B8A/B11/B12，SCL 云掩膜后月度中值）GeoTIFF + `S2_patches_manifest_ALL.csv`（每 site×月 `n_scenes` 有效影像数、`min_cloud`/`mean_cloud` 云量、`patch_px` 像素数、`crs`）。冻结 EO 大模型对每张 patch 预计算 image embedding（一次性，CPU / Colab GPU）。
> 2. **通道 B**（`Channel B/`）：`extract_sentinel2_monthly_indices_gee.js` → `sentinel2_oil_sites_monthly_indices_201704_202512_11aoi.csv`（NDVI/NDWI/NDBI/BSI 均值+标准差、`cloud_probability`、`valid_obs_count`）；`extract_viirs_monthly_nightlights_gee.js` → `viirs_oil_sites_monthly_nightlights_201401_202512_11aoi.csv`（`avg_rad` 均值/最大/标准差、`cf_cvg` 无云覆盖）→ 站点扩展窗 z-score 等机制变量。
> 3. **异步对齐**：不再把月度值 ffill 成多个相同周值；每条遥感观测记录 `image_embedding, observation_date, days_since_observation(age), cloud_fraction, sensor_type, valid_mask`，配合 modality mask + time-gap embedding 显式建模缺失（方案 §4.4）。


| 通道         | 数据集 / 指标                                                             | 提供方                            | 接口 / 标识                                                                                                          | 原始频率      | 输出                                                                                                 | 覆盖起点    |
| ---------- | -------------------------------------------------------------------- | ------------------------------ | ---------------------------------------------------------------------------------------------------------------- | --------- | -------------------------------------------------------------------------------------------------- | ------- |
| **A 影像表示** | **Sentinel-2 多光谱 patch**（6 波段 B2/B3/B4/B8A/B11/B12 + SCL 云掩膜，月度中值合成） | Copernicus Sentinel-2（via GEE） | `COPERNICUS/S2_SR_HARMONIZED`；11 AOI × 差异化 patch（port 6.4 / refinery 5.12 / terminal 2.56 km，单站微调 3.2 / 1.6 km）@ 10 m，per-site UTM | 月（受云、不规则；空月跳过） | 每 (site, month) GeoTIFF + `S2_patches_manifest_ALL.csv` → 冻结 EO 大模型 image embedding；`days_since_obs / cloud_fraction / valid_mask` | 2019-01 |
| **B 机制变量** | **Sentinel-2 地表光学指标 + 云概率**（NDVI/NDWI/NDBI/BSI 均值+标准差、`cloud_probability`、`valid_obs_count`） | Copernicus Sentinel-2（via GEE） | `COPERNICUS/S2_SR_HARMONIZED` + `COPERNICUS/S2_CLOUD_PROBABILITY`（s2cloudless + SCL 掩膜）；11 AOI，5 km 圆缓冲 | 月 | `s2_ndvi/ndwi/ndbi/bsi_{aoi}`（各含 `_std`）、`s2_cloud_probability_{aoi}`（信息缺口代理）、`s2_valid_obs_count_{aoi}`；产物 `sentinel2_oil_sites_monthly_indices_201704_202512_11aoi.csv` | 2017-04 |
| **B 机制变量** | **夜间灯光辐亮度**（`avg_rad` 均值/最大/标准差 + `cf_cvg` 无云覆盖） | NASA/NOAA VIIRS DNB（via GEE） | `NOAA/VIIRS/DNB/MONTHLY_V1/VCMSLCFG`；11 AOI，5 km 圆缓冲 | 月 | 原始 `ntl_avg_rad_mean/max/stddev_{aoi}`、`ntl_cf_cvg_mean_{aoi}`；派生 `ntl_anomaly_{aoi}`（站点扩展窗 z-score，过去-only，min 12 月）、`ntl_valid_obs_count_{aoi}`（数据质量）；产物 `viirs_oil_sites_monthly_nightlights_201401_202512_11aoi.csv` | 2014-01 |


**说明 / 注意（双通道设计 + 文献矩阵 §② 精读修正）：**

- ⚠️ **通道 A 是方法集成与实证检验核心**：6 波段 patch 对齐 Prithvi-EO 基础模型所用的 HLS 6-band 集；EO 大模型**只做特征提取不微调**，仅训练轻量 temporal attention（同 AOI 多月）+ site attention（11 AOI 加权池化）。
- ⚠️ **差异化 patch**：脚本 `PATCH_HALF_BY_TYPE` 默认 half-size = port 3200 m / refinery 2560 m / terminal 1280 m（即 full = 6.4 / 5.12 / 2.56 km），实际导出以 `aoi_oil_infrastructure.csv` 的 `patch_half_m` 为准并含单站覆盖微调（Fujairah/Kharg/Yanbu = 1600 m → 3.2 km；Basra = 800 m → 1.6 km）；EO 编码器后续 resize 到固定输入，故各类型像素尺寸不同无碍。
- ⚠️ **通道 B 用动态异常而非原始水平**：`ntl_anomaly_{aoi}`（站点 z-score），因 NTL 时间维度弱、原始水平受 AOI 规模与城市灯光污染影响（P024/P032）。
- ⚠️ **NTL 不是油轮代理**：Santos 实测 NTL↔油轮 Rs=−0.07 → 仅作综合锚泊/港口活动信号（P024）。
- ⚠️ **不做浮顶充填率**：P055 估的是油罐结构容量（V=πr²h）、需亚米级影像（S2 10 m 不可复现），不构造 `frt_fill_level`。
- ⚠️ **样本期对齐**：统一比较窗 2019–2026（与 patch 导出窗一致）；通道 B 的 VIIRS 自 2014、S2 指标自 2017，仅作辅助且按发布/可得日对齐。遥感为需求侧/上游信息代理（卫星活动→石油需求→供需预期→油价，P069），增量价值须由 "M1 vs M1+M2" 消融 + Clark–West/DM 检验证明，而非直接预测油价。
- 引用：EO 基础模型 — Prithvi-EO-2.0 (NASA-IMPACT/IBM)、SatMAE；NTL 航运代理 — Polinov, Bookman & Levin (2022)；NTL 数据选择 — Gibson, Olivia, Boe-Gibson & Li (2021)；云覆盖与油价信息 — Hao & Wang (2023)；卫星预测石油需求 — Bricongne et al. (2026, ECB WP 3198)。

---

## M3 Shipping Variables — Dataset Sources

> 与 `beatrice_task_literature_matrix.md` §③「M3 推荐变量」对应。核心:tanker-specific（油轮专属）的**流量强度 + 运力(DWT)加权 + 平均船型 + 区域/咽喉拆分 + 出口-进口方向性不对称 + 拥堵代理**。全部对齐周五截止周频（W-FRI），覆盖 6 个油相关咽喉（Hormuz/Suez/Malacca/Bab el-Mandeb/Panama/Cape）。所有 PortWatch 来源公开免费、无需 key；GFW 需免费 API token。
>
> **构建链路：**
>
> 1. 原始下载（`data/raw/03_shipping/`）：PortWatch 咽喉/港口 + GFW presence。
> 2. `aggregate_shipping_to_weekly.py` → `data/processed/weekly_shipping_features.csv`（PortWatch 日→周 sum；GFW 月→周 ffill）。
> 3. 并入主表 `weekly_features.csv` / `weekly_features.parquet`，登记于 `feature_groups.json["M3_add_shipping"]`（共 119 个候选特征）。派生/同步脚本：`code/scripts/add_avg_tanker_size.py`、`code/scripts/sync_shipping_features.py`。


| 数据集 / 指标                                       | 提供方                                                  | 接口 / 标识                                                                                                                                               | 原始频率 | → 周频处理                               | 派生特征（写入主表）                                                                                                                                                                                                                           | 覆盖起点    |
| ---------------------------------------------- | ---------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------- | ---- | ------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ------- |
| **咽喉过境**（按船型 `n_`*/`capacity_*`）               | IMF PortWatch `Daily_Chokepoints_Data`               | ArcGIS FeatureServer（org `weJ1QsnbMYJlCHdG`）；6 咽喉 portid                                                                                              | 日    | W-FRI 求和                             | `pw_{choke}_n_tanker`、`_n_total`、`_capacity_tanker`、`_capacity`、`_tanker_share`、`_tanker_cap_share`、`_avg_tanker_size`(=cap/n)、`_n_tanker_wow_pct`、`_capacity_tanker_4w_ma`；汇总 `pw_all_n_tanker_sum`/`_n_total_sum`/`_tanker_share`  | 2019-01 |
| **港口级进出口**（`import_tanker`/`export_tanker` 吨位） | IMF PortWatch `Daily_Ports_Data`                     | ArcGIS FeatureServer；14 油轮枢纽 portid                                                                                                                   | 日    | W-FRI 求和（按出口/进口篮子）                   | `pw_exp_hubs_export_vol`、`pw_imp_hubs_import_vol`、`pw_tanker_exp_imp_net`、`pw_tanker_exp_imp_asym`、`pw_tanker_exp_imp_log_ratio`、`pw_tanker_exp_imp_asym_4w_ma`、`pw_exp_hubs_export_vol_wow_pct`                                     | 2019-01 |
| **AIS 船舶 presence**（按船型 hours/vessels）         | Global Fishing Watch 4Wings `public-global-presence` | Report API（POST，Bearer token，6 咽喉多边形）                                                                                                                 | 月    | 月→日 ffill→W-FRI last                 | `gfw_{choke}_total_hours`、`_total_vessels`、`_cargo_hours`、`_bunker_hours`、`_other_hours`、`_other_share`、`_total_hours_mom_pct`、`**_mean_presence_hours_per_vessel`(=hours/vessels 存在强度/拥堵粗代理，原名 `_dwell_hours_per_vessel`，2026-07-03 改名)**；汇总 `gfw_all_total_hours_sum`（`_nontanker_hours` 已于 2026-07-03 删除） | 2012-01 |
| **AIS 船舶密度栅格**（月度 vessel density）              | EMODnet Human Activities                             | `EMODnet_HA_Vessel_Density`（约 1 km 栅格 GeoTIFF，`vesseldensity_10_YYYYMMDD.tif`）；本地 `data/raw/03_shipping/emodnet_vessel_density_monthly_2017-2025/` | 月    | AOI/咽喉多边形区域统计 → 月→日 ffill→W-FRI last | `emodnet_{node}_vessel_density`（节点/咽喉空间密度，区域贸易流强度代理）                                                                                                                                                                                 | 2017-01 |


**出口/进口枢纽篮子（港口级方向性，`download_portwatch_ports.py`）：**

- **出口枢纽**：Ras Tanura、Juaymah、Yanbu（沙特）、Ras Laffan（卡塔尔）、Primorsk、Novorossiysk（俄）、Corpus Christi（美）、Sidi Kerir（埃及 SUMED）、Bonny（尼日利亚）。
- **进口/炼化枢纽**：Rotterdam（荷）、Singapore、Ningbo（中）、Chiba（日）、Ulsan（韩）。
- 角色经数据验证（出口枢纽 `export_tanker ≫ import_tanker`，进口枢纽相反）。

**说明 / 注意（呼应文献矩阵 §③ 精读修正）：**

- ⚠️ `pw_{choke}_n_tanker` 是**油轮交通强度 / 海运流量粗代理**，非 P016 的 port-call frequency、非精确石油贸易量（P016/P018）。
- ⚠️ **方向性**用 PortWatch **港口级** `import_tanker`/`export_tanker`（吨位估计，源自 AIS 吃水，不内嵌油价 → 防泄漏）；PortWatch **咽喉数据无方向字段**，故方向性不可由咽喉数据构造（P018）。
- ⚠️ **拥堵**无专用锚地等待数据，用 GFW presence 的 `total_hours/total_vessels`（每船停留时长）作 dwell 代理（P016 dwell-time 概念）。
- ⚠️ **EMODnet vessel density** 为月度栅格（约 1 km），覆盖范围以欧洲/全球为主，作节点级 AIS 空间密度补充（与 GFW presence 互为交叉验证）；按 AOI/咽喉多边形做区域统计后并入，覆盖 2017+。
- ⚠️ **样本期对齐**：PortWatch 系列 2019+，GFW 系列 2012+，EMODnet 2017+；统一比较窗 2019–2026。油价方向为 price→shipping（P016/P017），建模须**严格滞后、按发布时点对齐、后向滚动**（避免 P018 中心 MA 前视泄漏）。
- 引用：PortWatch — Arslanalp, Exton, Gao, Kamali, Saraiva, Sozzi & Verschuur (2026, IMF WP/26/99)；Arslanalp, Marini & Tumbarello (2019, IMF WP/19/275)。GFW — Global Fishing Watch 4Wings API。EMODnet — EMODnet Human Activities Vessel Density Map。油价—航运关系 — Mi et al. (2022, 2023)。

---

## M3 Stage-2 — 动态异质图补采数据（节点特征 / O-D 有向边 / dark vessel）

> **定位**：阶段 1（PortWatch 咽喉/港口 + GFW 咽喉 presence 的扁平聚合）已闭环并跑完 M3 基线。本节登记**阶段 2「17 节点动态异质图 G(t)」的新采集 raw 数据**——把航运模态从「6 咽喉聚合特征」升级为「11 AOI 节点 + 6 咽喉 + 节点间有向边」的图结构。三个新下载脚本均对齐 M2 遥感所用的**同一批 11 AOI**（`data/raw/02_sentinel2/aoi_oil_infrastructure.csv`），统一窗口 2019–2025（对齐 M0–M4 比较窗）。所有源公开免费；GFW 复用阶段 1 的免费 API token。
>
> 产物目录：`data/raw/03_shipping/{IMF Portwatch, GFW}/`（`raw/` 整目录 gitignore，脚本入库）。

### 1. 11 AOI 图节点 ↔ PortWatch `portid` 映射（探测于 2026-07-03）

| site_id | short | 类型 | PortWatch portid | PortWatch 名称 | 匹配质量 |
| ------- | ----- | ---- | ---------------- | -------------- | -------- |
| P001 | Rotterdam | port | `port1114` | Rotterdam | 精确 |
| P002 | Fujairah | terminal | `port362` | Fujairah | 精确 |
| P003 | RasTanura | terminal | `port1091` | Ras Tanura | 精确 |
| P004 | Jurong | refinery | `port1201` | Singapore | **代理**（裕廊岛炼厂并入 Singapore；PortWatch 无独立裕廊油港） |
| P005 | Houston | port | `port481` | Houston | 精确 |
| P006 | Ningbo | port | `port824` | Ningbo | 精确（Zhoushan 为独立 `port1429`，未并） |
| P007 | Jamnagar | refinery | `port1199` | Sikka | **代理**（Reliance Jamnagar 专用外港 Sikka；PortWatch 无 Jamnagar） |
| P008 | Basra | terminal | `port2479` | Basrah Oil Terminal | 精确（油码头，非城市港 `port20`） |
| P009 | Ulsan | refinery | `port1338` | Ulsan | 精确 |
| P010 | Kharg | terminal | `port2164` | Kharg Island | 精确 |
| P011 | Yanbu | terminal | `port570` | Yanbu (King Fahd Port) | 精确 |

### 2. 三个数据源 / 脚本 / 产物

| 通道 | 数据集 / 指标 | 提供方 | 接口 / 标识 | 原始频率 | 脚本 → 产物 | 覆盖 |
| ---- | ------------ | ------ | ----------- | -------- | ----------- | ---- |
| **节点特征** | 11 AOI 单港 tanker/cargo 停靠数 + 进出口吨位 | IMF PortWatch `Daily_Ports_Data` | ArcGIS FeatureServer（org `weJ1QsnbMYJlCHdG`），11 portid | 日 | `IMF Portwatch/download_portwatch_aoi_nodes.py` → `portwatch_aoi_nodes_daily.csv`（**30074 行 = 11 节点 × 2734 天**，字段 `portcalls_tanker/cargo/portcalls`、`import/export_tanker`、`import/export`） | 2019-01 ~ 2026-06 |
| **O-D 有向边 + 节点 dwell** | AOI port-visit 事件（含 `vessel.id`、`durationHrs`、锚地） | GFW Events API v3 | `POST /v3/events`，`public-global-port-visits-events:v3.0`，`types=[PORT_VISIT]`，`vesselTypes=[BUNKER,CARGO]`，`confidences=[3,4]`，`geometry`=AOI 方形多边形 | 事件级 | `GFW/download_gfw_port_visits.py` → `gfw_aoi_port_visits.csv`（原始事件，节点级，含 dwell）+ `gfw_aoi_od_voyages.csv`（同船相邻 AOI 停靠 → 有向边 `from_site→to_site`、`transit_days`） | 2019 ~ 2025 |
| **dark vessel（暗船）** | SAR 检测数，AIS 匹配 / 未匹配拆分 | GFW 4Wings API | `POST /v3/4wings/report`，`public-global-sar-presence:latest`，`temporal-resolution=MONTHLY`，`filters[0]=matched='false'`(dark)/`'true'`(matched)，6 咽喉 + 11 AOI 多边形 | 月 | `GFW/download_gfw_sar_detections.py` → `gfw_sar_detections_monthly.csv`（每 region×月 `detections_total/dark/matched` + `dark_share`） | 2019 ~ 2025 |

### 3. API 与关键限制（写作/建模必读 caveat）

- ⚠️ **油轮无法在 events 层精确隔离**：GFW Events `vesselTypes` 枚举为 `[BUNKER, CARGO, CARRIER, FISHING, GEAR, OTHER, PASSENGER, SEISMIC_VESSEL, SUPPORT, NON_FISHING, RESEARCH]`，**无 TANKER**。实测纯原油港油轮归 **CARGO**（Ras Tanura 2024 `CARGO=1812` vs `BUNKER=3`；Kharg `CARGO=89` vs `BUNKER=0`）——`BUNKER` 仅加油船。故本项目取 `CARGO+BUNKER`：**油相关 AOI（Ras Tanura/Kharg/Basra/Yanbu/Fujairah）的 CARGO 近似即油轮**，综合港（Rotterdam/Singapore/Houston/Ningbo/Ulsan）为混合货运——O-D 边应理解为「油港主导的货运连通」而非纯油轮流。
- ⚠️ **O-D = 11 节点诱导子图**：边 `AOI_i→AOI_j` 表示「同一船在 AOI_i 之后**下一次被观测到的 AOI 停靠**是 AOI_j」，中途非 AOI 港口停靠不在观测内。这正是图模块所需的 AOI 子图 O-D。
- ✅ **节点级 dwell 升级**：`port_visit.durationHrs` 是每次停靠的真实停留时长，**比阶段 1 用 GFW presence `total_hours/total_vessels` 的粗代理精确**。⚠️ 但存在跨年**超长异常值**（如某船 `durationHrs≈114181`≈13 年，实为 AIS 长期在港/拼接），processed 阶段须按上限（如 ≤ 30 天）裁剪。
- ⚠️ **dark vessel = SAR 未匹配 AIS 检测**（P057 Paolo 2024 / P061 GFW SAR）：`detections_total = dark + matched`（实测 Hormuz 2024Q1 `4799 = 2388 + 2411`，暗船占比≈50%，与霍尔木兹伊朗制裁油轮关闭 AIS 一致）。SAR 覆盖 2017+、单请求 ≤ **366 天**（故按年分段），单位为**检测数**非小时。
- ⚠️ **AOI 多边形**：脚本默认方形 half-size = **0.25°（~25–28 km）**，覆盖泊位 + 近岸锚地；波斯湾密集区（Ras Tanura/Kharg/Basra/Fujairah）虽多边形不重叠，仍可用事件 `anchorage_id` 细分。可用 `--buffer-deg` 调整。
- ⚠️ **样本期与防泄漏**：本节 raw 统一 2019–2025；周频化 + 发布滞后（同阶段 1 的 GFW +4 周 / PortWatch +1 周思路）在 processed 建图阶段施加，raw 不含滞后。

### 4. 复现命令

```bash
cd data/raw/03_shipping/IMF\ Portwatch && python3 download_portwatch_aoi_nodes.py
cd data/raw/03_shipping/GFW && python3 download_gfw_port_visits.py --start 2019 --end 2025   # --resume 断点续传
cd data/raw/03_shipping/GFW && python3 download_gfw_sar_detections.py --start 2019 --end 2025
```

### 5. 实测产出概览（2026-07-03 全量下载）

| 产物 | 规模 | 关键统计 |
| ---- | ---- | -------- |
| `portwatch_aoi_nodes_daily.csv` | 30074 行（11 节点 × 2734 天，2019-01~2026-06） | Kharg 纯出口（`import_tanker`=0）、Basra 出口主导、Jurong/Singapore 转运量最大 |
| `gfw_aoi_port_visits.csv` | 570521 visits / 38300 unique vessels | 节点平均 dwell（小时）：Kharg 234、Jamnagar 167、Ras Tanura 126 最长；综合港 Rotterdam 61、Ulsan 57 |
| `gfw_aoi_od_voyages.csv` | 532221 边（cross-node 106992，self-loop 425229） | Top 航线：Ningbo↔Singapore(≈1.5万)、Fujairah↔Singapore、Singapore↔Rotterdam、Singapore→Ras Tanura（返程装油） |
| `gfw_sar_detections_monthly.csv` | 1419 行（17 region × 84 月，2019-01~2025-12） | **暗船占比：Kharg 79%、Basra 56%、Hormuz 51%、Suez 47%、Malacca 38%**（制裁"影子船队"信号）；合规港 Fujairah 5%、Rotterdam 11%、Ulsan 10% |

> ⚠️ **dark vessel 亮点即研究信号**：Kharg（伊朗唯一主要出口枢纽）暗船 79% + Basra（伊拉克）56% + Hormuz 51% 直接对应 OPEC/受制裁国油轮关闭 AIS 的"影子船队"，对供给风险 → 油价机制有强解释潜力（P057/P061）。

> **阶段 2 processed（已完成 2026-07-03）**：
> ① **周频图张量** → `data/processed/M3/py/build_m3_graph_weekly.py` 产出 `m3_graph_nodes_weekly.csv`（391 周 × 11 站 × 11 特征）+ `m3_graph_edges_weekly.csv`（O-D 有向边）+ `m3_graph_darkvessel_weekly.csv`（17 区域）+ `m3_graph_tensors.npz`（node `(391,11,11)` + adjacency `(391,11,11)`）；含 dwell ≤ 720h / transit ≤ 90d 裁剪、发布滞后 PW +1w / GFW-event +2w / SAR +4w、有向性与防泄漏自检通过。
> ② **EMODnet zonal stats** → `build_emodnet_weekly.py` 产出 `m3_emodnet_density_weekly.csv`。⚠️ EMODnet 为 **EPSG:3035 欧洲栅格**，17 区域中**仅 Rotterdam + Suez 有有效像素**，其余在覆盖外（NaN）；栅格覆盖 2017-2024（无 2025），故仅作 **Rotterdam 交叉验证**补充，非全球图特征。
> ③ **完整 17 节点异质图** → `build_m3_graph17.py` 产出 `m3_graph17_tensors.npz`（11 AOI + 6 咽喉节点 + AOI↔咽喉静态边 + 动态 O-D，`adjacency (391,17,17)`）；**z_ship 编码器**（类型专属投影 + 异质 GAT + 时序 TCN → 32 维，~4.2 万参数）见 `code/src/models/shipping_encoder.py`。
> 字段与形状详见 `data/processed/M3/m3_data_dictionary.md` Stage-2 节（§12.9 / §12.10）。

