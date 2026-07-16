# M2 遥感（Channel B）机制特征矩阵 — 数据字典

> 对应数据文件：`03_data/processed/M2/outputs/m2_weekly_features.csv`（一行一周，154 特征，入模用）
> 配套 EDA 表：`03_data/processed/M2/outputs/m2_eda_weekly.csv`（一行 = 一周×一站×一指标，供面板分析）
> 构建脚本：`03_data/processed/M2/py/build_m2_weekly.py`
> 上游 EDA / 机制文档：`03_data/processed/M2/m2_ChannelB_mechanism_plan_2026-06-22.md`；建模结果：`00_admin/待整理/flat_baseline_log.md` §8  
> 最后更新：2026-07-03

---

## 0. 定位：M2 = 双通道遥感中的 Channel B（机制变量）

M2 遥感模态采用**双通道设计**：

- **Channel A（影像表示，方法学核心）**：11 AOI × 月度 Sentinel-2 6 波段 patch → 冻结 EO 大模型（Prithvi-EO-2.0 / SatMAE）image embedding。**见** `03_data/processed/M2/m2_ChannelA_data_dictionary.md`（patch 尺寸、审计、Prithvi 嵌入产物 `s2_prithvi_emb_*.npy`）。
- **Channel B（机制变量，经济解释，辅助）**：VIIRS 夜光 + Sentinel-2 光学指数的站点活动 / 信息可得性代理。**本字典描述的即 Channel B 的周频建模表**。

Channel B 是「手工指标」扁平基线，用于回答 RQ1（遥感是否对 M1 有增量），并作为 RQ2「手工指标 vs 表示学习」的对照臂。

---



## 1. 基本信息


| 项      | 值                                                               |
| ------ | --------------------------------------------------------------- |
| 时间索引   | `week_fri`，每周五截止（W-FRI）                                         |
| 时间范围   | 2019-01-04 ~ 2025-12-26（365 周）                                  |
| 形状     | 365 行 × 155 列（`week_fri` + **154 特征**）                          |
| 统一比较窗口 | 与表范围一致（`m2_weekly_features.csv` 构建时即裁剪到 2019–2025 标准比较窗）          |
| 缺失约定   | 空单元格 = 该周该变量尚不可得（NaN）：或无 as-of 有效观测、或距平历史不足 12 月；不做隐式填补         |
| 频率     | 周频（月度遥感按 as-of 对齐到周五，**不做常数值 ffill**，改以 `age + mask` 显式表达陈旧/缺失） |
| 模态     | M2 = 遥感 Channel B（5 指标 × 11 油基础设施 AOI）                          |


**154 特征的构成：**


| 组                | 列数  | 命名模板                                     | 说明                         |
| ---------------- | --- | ---------------------------------------- | -------------------------- |
| 指标水平值 level      | 55  | `{idx}_{aoi}`                            | 5 指标 × 11 站；原始月度值（仅解释/稳健性） |
| 站点内标准化距平 anom    | 55  | `{idx}_anom_{aoi}`                       | 5 指标 × 11 站；**入模主力**       |
| S2 观测时效 / 可用性    | 22  | `s2_age_days_{aoi}` / `s2_avail_{aoi}`   | 11 站 ×（age + avail）        |
| VIIRS 观测时效 / 可用性 | 22  | `ntl_age_days_{aoi}` / `ntl_avail_{aoi}` | 11 站 ×（age + avail）        |


> 5 指标 = `NDVI` / `NDWI` / `NDBI` / `BSI`（Sentinel-2 光学）+ `NTL`（VIIRS 夜光）。

#### 三种形态：`level` / `anom` / `mom`


| 形态 | 含义 | 所在表 |
| --- | --- | --- |
| `level` | 原始值（如 NDVI=0.45、NTL=12.3），含季节、站间不可比 | `m2_weekly_features.csv` + `m2_eda_weekly.csv` |
| `anom` | 相对该站该指标**自己的历史**，去季节 + 标准化；**主分析用这个** | 同上 |
| `mom` | 这个月比上个月涨/跌多少（简单差分）；EDA 备选 | **仅** `m2_eda_weekly.csv` |


**`anom` 怎么理解（两步，均在月度上算完，再对齐到周频，见 §3）**

1. **去季节**：对**同一个站、同一个指标**，看历史上所有**同月份**（所有 1 月、所有 2 月……）的平均值，当作「这个月份的正常水平」。  
   残差 `resid = 实际值 − 该月正常值`：正 → 比这个站「往年这个月份」更绿/更亮；负 → 比往年同月更差。
2. **站点内 z-score**：再对该站的 `resid` 做标准化（至少需 12 个月历史），得到「偏了几个标准差」。  
   前 12 个月 `anom` 为 NaN（2019 窗内部分 S2 站点首个有效周晚于 2019-01-04，见 §4.3）。  
   统计量一律 **expanding、past-only**（含当月，不用未来），避免回测泄漏。

**`mom`**

公式：`mom_t = level_t − level_{t−1}`（月频一阶差分，无去季节、无标准化）。`m2_weekly_features.csv` **不含** `mom`；需月环比请读 `m2_eda_weekly.csv`（水体掩膜版：`m2_eda_weekly_watermask.csv`）。

#### S2 / VIIRS 观测时效与可用性怎么算

S2 四指数与 VIIRS **共用同一套对齐逻辑**（`align_weekly()`），按**模态**分别写入 `m2_weekly_features.csv`（S2 四指数共享一套 age/avail，NTL 单独一套）：


| 步骤 | 规则 |
| --- | --- |
| 代表观测日 | `observation_date = 月初 + 14 天`（月中第 15 日，`OBS_DAY=15`） |
| 保守可得日 | `availability_date = 月末 + 15 天`（`PUB_LAG_DAYS=15`，模拟发布延迟） |
| 周频匹配 | 对每个 `(site, index, week_fri)` 做 `merge_asof(direction="backward")`：取 `availability_date ≤ week_fri` 的**最近一条** `level` 非空的月度观测 |
| 观测时效 age | `days_since_obs = week_fri − observation_date`（天）；月内同一观测重复多周，age 每周 **+7** |
| 有效掩膜 | `valid_mask = 1` 若匹配到有效观测，否则 `0` |
| 模态可用 | `modality_mask = valid_mask 且 days_since_obs ≤ 100`（`MAX_AGE_DAYS=100`） |


**透视写入 `m2_weekly_features.csv`**（`to_wide()`，中间结果在 `m2_eda_weekly.csv` 的 `days_since_obs` / `modality_mask` 列）：


| `m2_weekly_features.csv` 列 | 来源（`m2_eda_weekly.csv` 字段） | 聚合 |
| --- | --- | --- |
| `s2_age_days_{aoi}` | S2 模态行的 `days_since_obs` | 按 `(week_fri, short_name)` 取 first（四指数同站同周 age 相同） |
| `s2_avail_{aoi}` | S2 模态行的 `modality_mask` | 按站取 **max**（任一 S2 指数可用则记 1） |
| `ntl_age_days_{aoi}` | VIIRS / NTL 行的 `days_since_obs` | 按站取 first |
| `ntl_avail_{aoi}` | VIIRS 行的 `modality_mask` | 按站取 max |


S2 与 NTL 的差异来自**原始月度表与云缺**，不是公式不同：S2 受云影响可能出现长月缺口（age 上限可达 ~271 天），VIIRS 连续性好（age 上限 ~61 天）。**主分析不入模** age/avail（§6）；合并矩阵 `weekly_feature_matrix.csv` 仅派生 `avail_m2`（任一站 RS 足够新则为 1）。

---



## 2. 构建脚本做了什么（`build_m2_weekly.py`）

**定位**：Channel B 的 Layer 2（processed）构建器。把两张**原始月度表**（Sentinel-2 光学指数 + VIIRS 夜光）转成**无泄漏、可入模**的周频表 `m2_weekly_features.csv`。核心是：① 每个 (站点, 指标) 派生 `level / anom / mom` 三种形态；② 月→周用**保守可得日的 as-of join**（非 ffill），每周同时携带 `age + mask`，让陈旧/缺失显式可见。

### 2.1 输入 → 输出


| 项            | 内容                                                                                                                                       |
| ------------ | ---------------------------------------------------------------------------------------------------------------------------------------- |
| 输入根目录        | `03_data/raw/02_sentinel2/Channel B/`                                                                                                    |
| S2 光学指数      | `sentinel2_oil_sites_monthly_indices_201704_202512_11aoi.csv`（NDVI/NDWI/NDBI/BSI 均值+标准差、`cloud_probability`、`valid_obs_count`），2017-04 起 |
| VIIRS 夜光     | `viirs_oil_sites_monthly_nightlights_201401_202512_11aoi.csv`（`ntl_avg_rad_mean/max/stddev`、`ntl_cf_cvg_mean`），2014-01 起                 |
| S2 水体掩膜版（B4） | `sentinel2_oil_sites_monthly_indices_watermask_201704_202512_11aoi.csv`（另含 MNDWI + `land_px`），见 §7                                       |
| 输出             | `03_data/processed/M2/outputs/m2_weekly_features.csv`（唯一入模产物，365×155）                                                                 |
| 输出（EDA）        | `03_data/processed/M2/outputs/m2_eda_weekly.csv`（20075 行，含 `mom` + 全部元数据）                                                        |
| 周频骨架         | `pd.date_range("2019-01-01", "2025-12-31", freq="W-FRI")`（默认窗口 `--start/--end` 可改）                                                       |


> ⚠️ 两张原始月度 CSV 经 Google Earth Engine 导出，**不入 git**（`raw/` 整目录 gitignore），本地存于 `03_data/raw/02_sentinel2/Channel B/`；缺失时脚本无法运行，需先从 GEE 重新导出（导出脚本 `extract_sentinel2_monthly_indices_gee{,_bundled,_watermask}.js` / `extract_viirs_monthly_nightlights_gee{,_bundled}.js` 已入库，另见 `external_sources.md` M2 节）。



### 2.2 处理流程（`main()` 调用顺序）

1. **载入 → 统一月度中间表** `load_monthly_long()`：S2 四指数 `melt` 成 tidy 行（`level` = 各指数月均值，`qual` = `valid_obs_count`）；VIIRS 取 `ntl_avg_rad_mean` 为 `level`、`ntl_cf_cvg_mean` 为 `qual`；两者按 `[site_id, index, date]` 拼接。
2. **派生三形态** `add_anomaly()`：对每个 (站点, 指标) 生成 `level / anom / mom`（详见 §1、§2.3）。
3. **月 → 周 as-of 对齐** `align_weekly()`：以「可得日 = 月末 + 15 天」做 backward `merge_asof`，每个周五映射到**最近一条已发布的有效月度观测**；同时计算 `days_since_obs / valid_mask / modality_mask`（详见 §1、§3）。
4. **透视列** `to_wide()`：由对齐后的 tidy 行生成 `m2_weekly_features.csv` 的 `{idx}_{aoi}`、`{idx}_anom_{aoi}` 及 `s2_/ntl_` 的 `age_days` / `avail`（S2 四指数共享一套观测时效，NTL 单独一套）；同步写出 `m2_eda_weekly.csv`。
5. **站点显示顺序**：按 `site_type`（port → refinery → terminal）再按短名字母序排列（见 §4.1）。
6. **写盘 + 自检**：写出 `m2_weekly_features.csv` / `m2_eda_weekly.csv`，打印形状、各模态平均可用率、各指标 anom 缺失比、以及 Houston NDVI 前 10 周的防泄漏 sanity。



### 2.3 三种形态：`level` / `anom` / `mom`（均 past-only，无泄漏）


| 形态      | 含义                                    | 去季节      | 标准化        | 用途                 | 所在表 |
| ------- | ------------------------------------- | -------- | ---------- | ------------------ | --- |
| `level` | 原始月度值（S2 指数月均 / NTL 平均辐亮度）            | 否        | 否          | 解释 / C3 稳健性（尺度不可比） | `m2_weekly_features.csv` + `m2_eda_weekly.csv` |
| `anom`  | **站点内标准化距平**：去季节残差再 expanding z-score | 是（月历气候态） | 是（z-score） | **入模主力**（跨站可比、去季节） | 同上 |
| `mom`   | 月环比一阶差分 `level.diff()`                | —        | —          | EDA 备选（对符号变化稳健）    | **仅** `m2_eda_weekly.csv` |


`anom` **计算（**`add_anomaly`**，全部 expanding、仅用历史）：**

1. **去季节**：按 (站点, 指标, 月份 month-of-year) 取 `expanding(min_periods=1).mean()` 作月历气候态 `clim`（含当月、无未来），残差 `resid = level − clim`。
2. **标准化**：按 (站点, 指标) 对 `resid` 取 `expanding(min_periods=12)` 的均值 `mu` 与标准差 `sd`，`anom = (resid − mu) / sd`；`±inf → NaN`。
3. 因 `min_periods=12`，每个 (站点, 指标) 序列**前 12 个月无 anom**（NaN）——这是部分站点距平首个有效周晚于 2019-01-04 的直接原因（见 §4.3）。

> 方法学依据（写作可引用，详见 channelB plan §B1）：z-score 标准化（通用预处理）＋ 站点内去均值（面板 within / 固定效应，Wooldridge；夜光经济学要求站点内相对基线，Henderson–Storeygard–Weil 2012、Donaldson–Storeygard 2016、Gibson et al. 2021）＋ 去季节标准化距平（同源 VCI：Kogan 1995；SPI：McKee et al. 1993）。**本项目增强**：统计量一律 expanding（仅用过去），满足样本外无泄漏。



### 2.4 运行模式


| 参数                   | 作用                                                                                      |
| -------------------- | --------------------------------------------------------------------------------------- |
| （默认）                 | 标准模式：4 个 S2 指数 + NTL，输出 `m2_weekly_features.csv`（365×155）                               |
| `--watermask`        | B4：改用 MNDWI 水体掩膜版 S2 CSV，加 MNDWI 指数 + `s2_land_px_*`，输出 `*_watermask.csv`（365×188，见 §7） |
| `--no-deseasonalize` | 只做 z-score、跳过月历去季节（稳健性对照）                                                               |
| `--start / --end`    | 改周频窗口（默认 `2019-01 .. 2025-12`）                                                          |


---



## 3. 防泄漏与发布时间戳对齐（关键）

所有变量按「真实可得时间」对齐：在周五 T 预测下一周时，只使用截至 T 已发布的信息（no look-ahead）。与 M1/M3 的「月末 ffill + 固定 shift」不同，M2 采用**逐观测 as-of join + 显式 age/mask**，是本项目最细的对齐方式。


| 环节       | 规则                                                         | 常量                 |
| -------- | ---------------------------------------------------------- | ------------------ |
| 代表观测日    | `observation_date = 月初 + 14 天`（≈ 月中第 15 天）                 | `OBS_DAY = 15`     |
| 保守可得日    | `availability_date = 月末 + 15 天`（保守发布延迟）                    | `PUB_LAG_DAYS=15`  |
| 月 → 周对齐  | `merge_asof(direction="backward")`：每个周五取「可得日 ≤ 该周五」的最近有效观测 | —                  |
| 观测时效 age | `days_since_obs = week_fri − observation_date`（天）          | —                  |
| 有效掩膜     | `valid_mask = 是否匹配到任一 as-of 有效观测（1/0）`                     | —                  |
| 模态掩膜     | `modality_mask = valid_mask 且 age ≤ 100 天`（观测足够新才算模态可用）    | `MAX_AGE_DAYS=100` |
| 距平最短历史   | 距平 z-score 至少需 12 个月历史                                     | `MIN_HIST=12`      |


要点：

- **不做常数值 ffill**：月度遥感值在一个月内会在多个周五重复，但每周 `days_since_obs` 随之 +7 递增（陈旧显式可见），新观测发布后才跳变——比「把月值填成多个相同周值」更诚实，也为端到端 missing-modality 建模（Channel A）保留 age/mask 接口。
- **as-of 而非按月份标签回填**：以「月末 + 15 天」为可得日，确保当月遥感值只在其保守发布后的第一个周五进入矩阵，杜绝「用当周尚未发布的值」。
- **距平仅用历史**：去季节气候态与 z-score 统计量均 expanding（含当月、无未来）；scaler/统计量绝不使用测试期信息。
- **防泄漏 sanity（脚本内置）**：`level` 月内重复、`days_since_obs` 每周 +7、发布滞后下新观测才跳变——三项自检通过（见 channelB plan §B1「产出」）。

---



## 4. 字段说明

> 覆盖率为全表（365 周）非空占比；首个有效周为该列第一个非 NaN 的周五。



### 4.1 站点清单（11 个油基础设施 AOI）

> 5 km 圆缓冲；`m2_weekly_features.csv` 列顺序 = 按类型 port → refinery → terminal，再按短名字母序。核心石油 AOI（文献支撑最强）：Rotterdam / Fujairah / RasTanura / Houston。


| 列顺序 | 短名 `{aoi}`       | `site_id` | 站点全名                    | 类型       | 地区  |
| --- | ---------------- | --------- | ----------------------- | -------- | --- |
| 1   | `Houston`        | P005      | Houston Ship Channel    | port     | 美国  |
| 2   | `NingboZhoushan` | P006      | Ningbo-Zhoushan Port    | port     | 中国  |
| 3   | `Rotterdam`      | P001      | Port of Rotterdam       | port     | 荷兰  |
| 4   | `Jamnagar`       | P007      | Jamnagar Refinery       | refinery | 印度  |
| 5   | `Jurong`         | P004      | Singapore Jurong Island | refinery | 新加坡 |
| 6   | `Ulsan`          | P009      | Ulsan Refinery          | refinery | 韩国  |
| 7   | `Basra`          | P008      | Basra Oil Terminal      | terminal | 伊拉克 |
| 8   | `Fujairah`       | P002      | Fujairah Oil Terminal   | terminal | 阿联酋 |
| 9   | `Kharg`          | P010      | Kharg Island Terminal   | terminal | 伊朗  |
| 10  | `RasTanura`      | P003      | Ras Tanura Terminal     | terminal | 沙特  |
| 11  | `Yanbu`          | P011      | Yanbu Export Terminal   | terminal | 沙特  |




### 4.2 遥感指标水平值 `level`（55 列：`{idx}_{aoi}`）

> 每个指标对 11 站各一列，命名 `{idx}_{aoi}`（如 `NDVI_Houston`、`NTL_RasTanura`）。下表波段公式**取自 GEE 提取脚本**（`extract_sentinel2_monthly_indices_gee.js`、`extract_viirs_monthly_nightlights_gee.js`），非泛化概念式。`level` 仅用于解释与 C3「level vs anom」稳健性，**主分析不用**（尺度不可比 + 含季节 + 与 anom 冗余）。

Sentinel-2 波段：`B2` = 蓝、`B3` = 绿、`B4` = 红、`B8` = 近红外（NIR，10 m）、`B11` = 短波红外（SWIR1）。GEE 用 `normalizedDifference([a, b]) = (a − b)/(a + b)`。


| 指标 `{idx}` | 含义                            | 精确公式（GEE 波段）                                        | 来源                  | 单位           | 覆盖率    |
| ---------- | ----------------------------- | --------------------------------------------------- | ------------------- | ------------ | ------ |
| `NDVI`     | 归一化植被指数（植被绿度）                 | `(B8 − B4) / (B8 + B4)`                             | S2 SR（月中值合成，AOI 均值） | 无量纲 [−1,1]   | 99.9%  |
| `NDWI`     | 归一化水体指数（水面/湿度，McFeeters 1996） | `(B3 − B8) / (B3 + B8)`                             | S2 SR（月中值合成，AOI 均值） | 无量纲 [−1,1]   | 99.9%  |
| `NDBI`     | 归一化建成区指数（不透水面/建筑，Zha 2003）    | `(B11 − B8) / (B11 + B8)`                           | S2 SR（月中值合成，AOI 均值） | 无量纲 [−1,1]   | 99.9%  |
| `BSI`      | 裸土指数（裸地/堆场）                   | `((B11+B4) − (B8+B2)) / ((B11+B4) + (B8+B2))`       | S2 SR（月中值合成，AOI 均值） | 无量纲 [−1,1]   | 99.9%  |
| `NTL`      | 夜间灯光辐亮度（夜间活动强度）               | VIIRS DNB 月度 `avg_rad` 的 AOI 均值（`ntl_avg_rad_mean`） | VIIRS DNB（月度）       | nW·cm⁻²·sr⁻¹ | 100.0% |
| `MNDWI` †  | 改进归一化水体指数（Xu 2006）            | `(B3 − B11) / (B3 + B11)`                           | S2 SR（**仅水体掩膜版**）   | 无量纲 [−1,1]   | —      |


> † `MNDWI` 仅出现在 B4 水体掩膜版（§7），标准版无此列。
> 覆盖率为 11 站均值：S2 四指数 level 唯一未满的是 `Jurong`（98.4%，早期云缺）；`NTL` level 全 11 站 100%。



#### 4.2.1 GEE 提取处理链（`level` 的原始口径）

**Sentinel-2 光学指数**（`COPERNICUS/S2_SR_HARMONIZED` + `COPERNICUS/S2_CLOUD_PROBABILITY`，5 km 圆缓冲）：

1. **场景预筛**：`CLOUDY_PIXEL_PERCENTAGE ≤ 60`（`CLOUD_FILTER`）。
2. **双重云掩膜**：s2cloudless 概率 `< 40`（`CLD_PRB_THRESH`）视为晴空；再叠加 SCL 掩膜，剔除 SCL ∈ {3 云影, 8 中概率云, 9 高概率云, 10 卷云, 11 雪/冰}。
3. **月度合成**：晴空像元取 **median** 合成 → AOI 内 `reduceRegions(mean)`，`scale = 20 m`；`_std`（标准差）同步导出（`m2_weekly_features.csv` 未用）。
4. `valid_obs_count` = 当月进入合成的有效场景数（数据质量指标，非入模；见 §4.5）。

**VIIRS 夜光**（`NOAA/VIIRS/DNB/MONTHLY_V1/VCMSLCFG`，5 km 圆缓冲）：

1. **质量掩膜**：剔除 `avg_rad < 0`（杂散光 / 数据伪影）。
2. **区域统计**：AOI 内 `reduceRegions(mean/max/stdDev)`，`scale = 500 m`；`level` 取 `avg_rad` 均值（`ntl_avg_rad_mean`）。
3. `ntl_cf_cvg_mean` = 无云观测计数（数据质量指标，非入模）。



### 4.3 站点内标准化距平 `anom`（55 列：`{idx}_anom_{aoi}`）—— 入模主力

> 每个指标对 11 站各一列，命名 `{idx}_anom_{aoi}`（如 `NTL_anom_Fujairah`）。这是**唯一并入统一特征矩阵**的 M2 列集（见 §6）。计算见 §1、§2.3；单位为无量纲（站点内 z-score，标准差单位）。

因距平需 ≥12 个月历史（`MIN_HIST=12`）＋ 早期云缺，**S2 距平**各站首个有效周不同：


| 站点                                           | S2 距平覆盖率 | S2 距平首个有效周 | NTL 距平覆盖率 | 备注             |
| -------------------------------------------- | -------- | ---------- | --------- | -------------- |
| Houston                                      | 100.0%   | 2019-01-04 | 100.0%    | 满覆盖            |
| Rotterdam                                    | 100.0%   | 2019-01-04 | 100.0%    | 满覆盖            |
| Jamnagar                                     | 100.0%   | 2019-01-04 | 100.0%    | 满覆盖            |
| Basra / Fujairah / Kharg / RasTanura / Yanbu | 100.0%   | 2019-01-04 | 100.0%    | 满覆盖            |
| Ulsan                                        | 93.4%    | 2019-06-21 | 100.0%    | 早期 S2 观测不足     |
| NingboZhoushan                               | 89.9%    | 2019-09-20 | 100.0%    | 早期 S2 观测不足     |
| Jurong                                       | 83.8%    | 2020-02-21 | 100.0%    | S2 云缺最严重（覆盖最低） |


> 全体均值：S2 四指数 anom ≈ 97.0%，NTL anom = 100.0%。**所有 NTL 距平满覆盖**（VIIRS 自 2014 起、云影响小，到 2019 窗内历史充足）。



### 4.4 观测时效与模态可用性（44 列）

> S2 四指数共享**一套** S2 观测时效（同一 patch 同时算四指数），VIIRS 单独一套；故按模态各 11 站生成 age + avail。


| 字段模板                 | 含义                                   | 单位    | 覆盖率    | 取值范围（窗内）        |
| -------------------- | ------------------------------------ | ----- | ------ | --------------- |
| `s2_age_days_{aoi}`  | 该周距最近一次 **S2** 有效观测的天数（陈旧度）          | 天     | 99.9%  | 28 ~ 271（中位 45） |
| `s2_avail_{aoi}`     | **S2** 模态掩膜：最近有效观测是否足够新（age ≤ 100 天） | {0,1} | 99.9%  | 多数 = 1          |
| `ntl_age_days_{aoi}` | 该周距最近一次 **VIIRS** 有效观测的天数            | 天     | 100.0% | 28 ~ 61（中位 45）  |
| `ntl_avail_{aoi}`    | **VIIRS** 模态掩膜：age ≤ 100 天           | {0,1} | 100.0% | 多数 = 1          |


> `age` 越大表示该周使用的遥感值越陈旧（月内会 +7 递增）；S2 的 age 上限（271 天）大于 NTL（61 天），反映 S2 受云影响、部分站点存在长月缺口。**这两族属于观测时效/可信度元数据，主分析不入模**（avail 窗内近零方差；age 为时效非活动信号），价值在端到端 missing-modality 建模。



### 4.5 变量选择依据（文献先验，写作可引用）

> 详见 `01_literature/literature_matrix.md` §② 与 channelB plan §B3。诚实边界：这些文献用作**变量/形态选择依据**，遥感增量的预测价值由本研究 SHAP + 消融 + Clark–West/DM 检验判定（结果见 `flat_baseline_log.md` §8）。


| 主张                                                                     | 依据                                                                     |
| ---------------------------------------------------------------------- | ---------------------------------------------------------------------- |
| 用**站点内异常 z-score**、不用原始水平                                              | 夜光经济学（Henderson–Storeygard–Weil 2012；Gibson et al. 2021）；与 `anom` 口径一致 |
| **必须 VIIRS**（而非 DMSP）夜光                                                | P024 / P032                                                            |
| 指标先验排序：`NTL` > `NDBI`/`BSI`（建成/裸地，工业相关）> `NDWI`（水面，噪声）、`NDVI`（植被，明确降级） | P024/P032（NTL）；P069（NDVI 降级）；B0 审计（NDWI 噪声）                            |
| 文献核心 NTL 站点：Fujairah / RasTanura / Rotterdam / Houston                 | 构成 C1「文献精选」臂（§6）                                                       |
| `valid_obs_count` / `cloud_probability` 为数据质量指标，**非预测特征**              | 避免把「云多/云少」误学成价格信号（仅用于 B0 审计与过滤）                                        |
| **NTL 不是油轮代理**（仅综合港口/锚泊活动信号）                                           | Santos 实测 NTL↔油轮 Rs ≈ −0.07（P024）                                      |
| 不构造浮顶油罐充填率                                                             | P055 需亚米级影像，S2 10 m 不可复现                                               |


> **机制 EDA（B2）小结**：整体相关弱（|corr| < 0.15）、多数较强相关落在**负 lag**（Channel B 多为滞后/同期反应而非领先油价）；亮点 `NDWI` terminal lag+1 Granger p≈0.029。含义：手工指标领先信号有限，**反向支撑「需 Channel A 表示学习 + 门控融合」**（RQ2 动机）。

---



## 5. 派生变量公式


| 字段                       | 公式                                                                    |
| ------------------------ | --------------------------------------------------------------------- |
| `{idx}_{aoi}`（level）     | S2：该 AOI 该指数的月度均值；NTL：`ntl_avg_rad_mean`                              |
| 去季节残差 `resid`            | `level − clim`，`clim = expanding_mean(level by 月份, min 1)`（past-only） |
| `{idx}_anom_{aoi}`       | `(resid − μ) / σ`，`μ,σ = expanding(resid, min 12)` 的均值/标准差；±inf→NaN   |
| `mom`（仅 `m2_eda_weekly.csv`） | `level.diff()`（按站点×指标的月环比一阶差分） |
| `s2_/ntl_age_days_{aoi}` | `week_fri − observation_date`（天）                                      |
| `s2_/ntl_avail_{aoi}`    | `1{ 匹配到有效观测 且 age ≤ 100 天 }`                                          |


---



## 6. 建模合约：154 列 → 用哪些（`--m2-features`）

> 决策依据见 channelB plan §3 B3；建模入口 `04_code/scripts/flat/run_baseline.py --modality M2 --m2-features {...}` 与 `04_code/src/backtest/data.py`。**只有 55 个 anom 列被并入统一特征矩阵** `03_data/processed/merge/outputs/weekly_feature_matrix.csv`（另派生 `avail_m2`）；level/age/avail 留在 `m2_weekly_features.csv` 供稳健性/EDA，不并入。


| `--m2-features` | 列数  | 内容                                                       | 用途                   |
| --------------- | --- | -------------------------------------------------------- | -------------------- |
| `anom`（默认）      | 55  | `{NDVI,NDWI,NDBI,BSI,NTL}_anom_{11 AOI}`                 | **主分析**（答 RQ1）       |
| `literature`    | 4   | `NTL_anom` of Fujairah / RasTanura / Rotterdam / Houston | C1 文献精选臂（可解释）        |
| `level`         | 55  | `{idx}_{aoi}` 原始水平                                       | C3 level vs anom 稳健性 |
| `all`           | 110 | anom + level                                             | 稳健性                  |


**入模列决策（实测窗 2019–2025）：**


| 类别                  | 列数  | 决策                                             |
| ------------------- | --- | ---------------------------------------------- |
| `{idx}_anom_{aoi}`  | 55  | ✅ **入模主力**（跨站可比、去季节、std≈1.2 无常数列）              |
| `{idx}_{aoi}` level | 55  | ❌ 主分析不用；仅 C3 稳健性（尺度不可比 + 含季节 + 与 anom 冗余）      |
| `s2_/ntl_age_days`  | 22  | 🔶 主分析不用（时效非信号）；可选聚合版作稳健性                      |
| `s2_/ntl_avail`     | 22  | ❌ 剔除（窗内近零方差；价值在端到端 missing-modality，非 tabular） |


> ⚠️ **level 合约当前状态**：标准矩阵 `weekly_feature_matrix.csv` 只含 55 个 anom 列，未纳入 55 个 level；因此 `--m2-features level/all` 在现版矩阵下会退化为等价 M1（详见 `flat_baseline_log.md` §8.2 †）。若需 C3 复现须重建矩阵纳入 level（优先级低）。
> **三层用法**：主分析全 55 anom + PCA/ElasticNet/SHAP 降维对照（答 RQ1，P058：SHAP 不解决共线性）；可解释层突出 `literature` 4 列 NTL + SHAP；`leave-one-AOI-out` 用全 55 anom。

---



## 7. B4 水体掩膜稳健性版（`--watermask`）

针对 Basra/Kharg 等 NDWI 受水面主导的问题，重跑 GEE 加水体掩膜、仅陆地像素算光学指数（呼应 McFeeters 1996 / Xu 2006）。


| 项                  | 值                                                                                                                       |
| ------------------ | ----------------------------------------------------------------------------------------------------------------------- |
| 输入                 | `sentinel2_oil_sites_monthly_indices_watermask_201704_202512_11aoi.csv`                                                 |
| 输出                 | `m2_weekly_features_watermask.csv`（365 行 × 188 列）；`m2_eda_weekly_watermask.csv`                                 |
| 相对标准版新增            | `MNDWI` 指数（`MNDWI_{aoi}` + `MNDWI_anom_{aoi}`，共 22 列）+ `s2_land_px_{aoi}`（陆地像素占比，11 列）                                  |
| MNDWI              | 改进归一化水体指数 `(B3 − B11)/(B3 + B11)`（Green − SWIR1，Xu 2006）                                                                |
| 水体掩膜逻辑             | 逐影像先算 MNDWI；`MNDWI > 0`（`WATER_THRESH = 0`）判为水体像元，**从 NDVI/NDBI/BSI 中掩除**（仅保留陆地像元）；`NDWI` 本身是水体指数**不掩膜**、`MNDWI` 亦全像元保留 |
| `s2_land_px_{aoi}` | 该站陆地像素占比 `= 1 − mean(is_water)` [0,1]；`m2_eda_weekly_watermask.csv` 另有 `low_land_coverage` 标志（`land_px < 0.05`，`LOW_LAND_THRESH`）                   |
| 入模影响               | 合并时 `filter_m2_anom_columns` 仍只取 55 个 `{NDVI,NDWI,NDBI,BSI,NTL}_anom`，**自动排除** `MNDWI_anom`；但 NDVI/NDBI/BSI 已仅含陆地像元     |


> 结论（`flat_baseline_log.md` §8.4）：M2 Ridge 不受影响；**M2 XGB 结论强化**（RMSE 改善约 1.7%，CW_p vs M1 从 0.006 → 0.0001），进一步支持 RS 通道增量价值；原版 anom-55 为保守估计。

---



## 8. `m2_eda_weekly.csv`（供 EDA / 面板）

每行 = 一个 (周, 站点, 指标)；20075 行 = 365 周 × 11 站 × 5 指标。


| 列                                                    | 含义                               |
| ---------------------------------------------------- | -------------------------------- |
| `week_fri`                                           | 周五（W-FRI）                        |
| `site_id` / `short_name` / `site_name` / `site_type` | 站点标识与类型                          |
| `modality` / `sensor`                                | `S2`（Sentinel-2）或 `VIIRS`        |
| `index`                                              | `NDVI/NDWI/NDBI/BSI/NTL`         |
| `level` / `anom` / `mom`                             | 三种形态（见 §1、§2.3）                  |
| `observation_date` / `days_since_obs`                | 代表观测日与陈旧天数                       |
| `valid_mask` / `modality_mask`                       | 有效掩膜 / 模态掩膜                      |
| `valid_obs_count`                                    | 该月有效观测数（S2）/ 无云覆盖（NTL）——数据质量，非入模 |
| `land_px` / `low_land_coverage`                      | **仅** `m2_eda_weekly_watermask.csv`：陆地像素占比 / 低陆地覆盖标志     |


---



## 9. 备注与已知缺口

- **早期距平缺失（均属数据质量，不影响 NTL）**：`Jurong`（83.8%，首周 2020-02-21）、`NingboZhoushan`（89.9%，2019-09-20）、`Ulsan`（93.4%，2019-06-21）的 S2 四指数距平因「≥12 月历史 + 早期云缺」而前段为 NaN；三站 NTL 距平均 100%。建模管线对残缺按历史 ffill（无泄漏）+ 残留填 0（中性）处理，各配置落在相同测试周。
- `level` **与** `anom` **的关系**：`anom` 已去季节 + 站点内标准化，是主分析口径；`level` 保留仅供解释与 C3。二者高度相关但 `level` 尺度不可比。
- `mom` **仅在** `m2_eda_weekly.csv`：`m2_weekly_features.csv` 不含月环比；如需请读 EDA 表或在建模层派生。
- **原始月度 CSV 不入 git（按项目 Git 策略，**`raw/` **整目录 gitignore）**：由 GEE 导出，本地存于 `03_data/raw/02_sentinel2/Channel B/`（S2 指数自 2017-04、VIIRS 自 2014-01；水体掩膜版 S2 另有一份）。提取脚本 `extract_sentinel2_monthly_indices_gee{,_watermask}.js`、`extract_viirs_monthly_nightlights_gee.js` 已入库。`m2_weekly_features.csv` 已裁到 2019–2025 标准窗；重建须先备齐原始 CSV。
- **Channel A 不在 `m2_weekly_features.csv`**：影像 patch / embedding 见 `m2_ChannelA_data_dictionary.md`；与 Channel B 手工指标表分离，在贡献层做「手工指标 vs 表示学习」对照（RQ2）。
- **预测目标不在本矩阵内**：M2 只含遥感特征；唯一核心目标（下一周 Brent 价格）由 M1 的 `brent_price` 在合并/建模阶段前瞻一周生成。M2 与 M1/M3 的合并见 `03_data/processed/merge/py/build_feature_matrix.py`。

---



## 10. 复现命令

```bash
cd "03_data/processed/M2/py"
python3 build_m2_weekly.py                     # 标准版 → m2_weekly_features.csv (365×155)
python3 build_m2_weekly.py --watermask         # B4 水体掩膜版 → *_watermask.csv (365×188)
python3 build_m2_weekly.py --no-deseasonalize  # 只 z-score、不去季节（稳健性）
python3 build_m2_weekly.py --start 2019-01 --end 2025-12   # 改窗口

# 并入统一特征矩阵（仅取 55 anom；派生 avail_m2）
cd "../../merge/py"
python3 build_feature_matrix.py                                   # 标准矩阵
python3 build_feature_matrix.py --m2-csv ../M2/outputs/m2_weekly_features_watermask.csv  # B4
```

