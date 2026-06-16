# AI 2026-05-26

---

> **本文件包含：**
> - 金融控制变量详细说明（美元指数、联邦基金利率、VIX、S&P 500、CPI 的数据内容、模型角色和可生成变量）
> - 官方报告类数据整理（OPEC MOMR、EIA STEO、OPEC ASB 的 LLM 提取方案和可生成变量示例）
> - 新闻事件数据整理（GDELT BigQuery SQL 链接、v1/v2 校准方法、BP/Shell/TotalEnergies 下载说明）
> - 遥感数据质量检查（Landsat/Sentinel-2/VIIRS 结构完整性验证、11 个 AOI 的光学空值覆盖率统计）
> - 航运数据优先级评估（PortWatch 和 GFW 为核心，NOAA AIS 和 EMODnet 不建议下载的原因）
> - 供应链图节点筛选（OGIM 13 个基础设施类别的保留/丢弃决策）
> - 天气数据不建议下载的理由（已被现有数据间接覆盖、投入产出比低）

## 一、金融

### 1D

这些变量应该进入主预测模型，例如 XGBoost、LSTM、Temporal Fusion Transformer 或 multimodal fusion model，作为外部控制变量。


| Dataset                                         | 里面实际记录的信息                                                                       | 怎么用在项目里                                                            | 可以生成的变量示例                                                                                                    | 在模型里的角色                                                                |
| ----------------------------------------------- | ------------------------------------------------------------------------------- | ------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------ | ---------------------------------------------------------------------- |
| FRED Nominal Broad U.S. Dollar Index            | 记录美元相对于一篮子主要贸易伙伴货币的名义汇率强弱。因为 Brent / WTI 等国际油价通常以美元计价，美元走强通常会影响非美元国家的购买成本和油价压力。 | 将日度或工作日数据聚合到周度，用来控制美元汇率变化对油价的影响。尤其适合解释油价变化中不是由供需本身造成、而是由美元强弱带来的部分。 | `usd_index_weekly_mean`; `usd_index_weekly_change`; `usd_appreciation_signal`; `usd_volatility_4w`           | 核心宏观金融控制变量。用于控制美元计价效应，避免模型把汇率冲击误判为石油供需冲击。                              |
| FRED Effective Federal Funds Rate               | 记录美国有效联邦基金利率，反映美国短期货币政策和流动性环境。利率上升通常意味着融资成本上升、美元走强、经济需求可能降温。                    | 聚合到周度或月度后加入模型，用来控制美国货币政策变化对油价、需求预期和金融市场风险偏好的影响。                    | `fed_funds_rate`; `fed_rate_change`; `monetary_tightening_signal`; `interest_rate_level`                     | 宏观政策控制变量。用于捕捉利率环境和流动性变化对油价的间接影响。                                       |
| CBOE Volatility Index (VIX)                     | 记录美股市场隐含波动率，常被视为全球金融市场风险情绪或避险情绪指标。VIX 上升通常表示市场不确定性增强。                           | 将日度 VIX 聚合到周度，用来控制金融市场恐慌、风险偏好下降、宏观不确定性上升对油价的影响。                    | `vix_weekly_mean`; `vix_weekly_max`; `vix_change`; `risk_aversion_signal`; `financial_stress_flag`           | 金融风险情绪控制变量。帮助区分“金融市场风险冲击”和“真实石油供需冲击”。                                  |
| S&P 500 Index                                   | 记录美国股票市场整体表现，可作为全球风险资产表现和经济预期的 proxy。股市上涨通常代表风险偏好较强，股市下跌可能代表经济担忧或避险情绪增强。        | 聚合到周度，提取收益率和波动率，用来控制宏观增长预期、金融市场风险偏好和资产价格联动对油价的影响。                  | `sp500_weekly_return`; `sp500_volatility_4w`; `equity_market_sentiment`; `risk_on_signal`; `risk_off_signal` | 金融市场控制变量。用于捕捉油价与风险资产之间的共同波动。                                           |
| U.S. Consumer Price Index / Inflation Indicator | 记录美国消费价格水平或通胀变化，反映宏观通胀压力。能源价格本身会影响通胀，而通胀也会影响货币政策和市场预期。                          | 通常按月度使用，并前向填充到周度模型。用于控制通胀环境，帮助模型理解油价处于高通胀还是低通胀背景下。                 | `cpi_yoy`; `cpi_mom`; `inflation_pressure`; `real_oil_price_proxy`; `macro_price_pressure`                   | 宏观背景控制变量。用于控制通胀环境，并可辅助构造 real oil price 或 inflation-adjusted features。 |


## 二、官方报告-官方机构怎么看整个市场

全球市场层面的供需判断
市场整体情感、供需修正方向


| Dataset                                | 里面实际记录的信息                                                                                      | 哪些内容适合给 LLM 提取                                                                        | 在模型里的角色                                                                        |
| -------------------------------------- | ---------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------ |
| OPEC Monthly Oil Market Report (MOMR)  | OPEC 每月发布的油市报告，记录原油价格、全球经济、世界石油需求、世界石油供给、炼厂运行、油轮市场、原油与成品油贸易、商业库存、供需平衡等内容。                      | 适合提取月度市场叙事和事件解释，例如油价变化原因、供需变化、库存变化、炼厂约束、油轮市场压力、贸易流变化和地缘政治风险。                          | 核心月度文本数据。可用 LLM 提取 OPEC 视角下的市场解释变量，并前向填充到周度模型中。                                |
| EIA STEO Report Text                   | EIA 每月短期能源展望文本，记录 Brent 价格展望、全球油价、全球液体燃料消费和生产、库存变化、OPEC+ 政策假设、中国库存、美国原油产量、炼油利润、天然气、天气和宏观经济假设等。 | 适合提取预测型叙事和官方预期变化，例如 EIA 是否认为 Brent 将上涨或下跌、库存是否累积、供给是否增长、需求是否放缓、OPEC+ 是否支撑价格、炼厂风险是否上升。 | 核心月度预测叙事数据。可与 MOMR 对照使用：MOMR 代表 OPEC 市场解释视角，STEO 代表 EIA / 美国官方预测视角。            |
| OPEC Annual Statistical Bulletin (ASB) | OPEC 年度统计公报，记录储量、产量、钻机、油井、炼厂能力、炼厂吞吐、油品需求、原油和成品油贸易、油轮运费、油价、天然气储量和进出口等长期结构性数据。                   | 不适合提取高频新闻情绪，更适合提取长期结构变化，例如产能扩张、炼厂能力变化、出口方向变化、油轮运费水平、长期供给能力和贸易结构。                      | 年度结构性背景数据。适合作为长期控制变量、supply-chain graph node weights 和年度结构特征，不适合直接解释日度或周度短期波动。 |


### 可以生成的 LLM 变量示例

#### OPEC Monthly Oil Market Report (MOMR)

```text
momr_supply_sentiment
momr_demand_revision_signal
momr_refinery_margin_signal
momr_tanker_disruption_signal
momr_inventory_tightness
momr_geopolitical_risk_flag
momr_market_balance_sentiment
```

#### EIA STEO Report Text

```text
steo_brent_outlook_sentiment
steo_inventory_build_signal
steo_opec_policy_support_flag
steo_china_stockpiling_signal
steo_supply_growth_signal
steo_refinery_margin_risk
steo_demand_outlook_revision
```

## 三、新闻-市场上发生了什么具体事件

供给冲击、地缘风险、企业决策
公司层面的行动 + 事件层面的冲击

### GDELT

GDELT 的BIG Query SQL链接
[https://console.cloud.google.com/bigquery?project=casa0004&ws=!1m22!1m7!12m5!1m3!1scasa0004!2sus-central1!3s421d4ad5-0c99-4e4b-8675-dbf82501c6b0!2e1!23sTREE_NODE_SELECTION!1m7!12m5!1m3!1scasa0004!2sus-central1!3sf5f4bd9a-2636-4f96-a050-bec4d110caaa!2e1!23sTREE_NODE_SELECTION!1m5!16m3!1m1!1scasa0004!3e12!23sTREE_NODE_SELECTION](https://console.cloud.google.com/bigquery?project=casa0004&ws=!1m22!1m7!12m5!1m3!1scasa0004!2sus-central1!3s421d4ad5-0c99-4e4b-8675-dbf82501c6b0!2e1!23sTREE_NODE_SELECTION!1m7!12m5!1m3!1scasa0004!2sus-central1!3sf5f4bd9a-2636-4f96-a050-bec4d110caaa!2e1!23sTREE_NODE_SELECTION!1m5!16m3!1m1!1scasa0004!3e12!23sTREE_NODE_SELECTION)

GDELT1.0 2006.1.1-2015.12.31
GDELT2.0 2015.2.21-2025.12.31
用重叠期（2015-02 ~ 2015-12）做校准，对 v1 进行缩放对齐到 v2 的量级

1.oil_geopolitical_daily ：20060101-20150220 采用gdelt1.0；20150221-20251231采用gdelt2.0
2.transport_disruption：20060101-20150220 采用gdelt1.0；20150221-20251231采用gdelt2.0

Oil Geopolitical（校准幅度大）


| 列类型                                        | 校准方式          | 系数          |
| ------------------------------------------ | ------------- | ----------- |
| count 类（event_count, conflict, sanction 等） | 乘法 v1 × ratio | 2.08–2.35x  |
| total_mentions                             | 乘法            | 1.03x（几乎不变） |
| avg_tone                                   | 加法 v1 + diff  | +0.23       |
| avg_goldstein                              | 加法            | -0.13       |


接缝检查：校准后 2015-02-20 → 2015-02-21 = 37,150 → 29,234（0.79x），属于正常日间波动，没有人工跳变。

Transport Disruption（几乎不需要校准）

count 类比率都在 1.00–1.11x，基本原样通过。total_mentions 做了 0.39x 缩放（和 Oil Geo 一样，GDELT 1.0 的 mentions 计算方式不同）。

每个文件都包含 `gdelt_version` 列

- `0` = GDELT 1.0 校准后（2006-01-01 ~ 2015-02-20，3,338/3,348 天）
- `1` = GDELT 2.0 原始（2015-02-21 ~ 2025-12-31，3,967/3,957 天）

这两个文件可以直接用于后续的周度聚合 → 入 M2 feature matrix。

### BP

[https://www.bp.com/en/global/corporate/investors/results-reporting-and-presentations/archive.html](https://www.bp.com/en/global/corporate/investors/results-reporting-and-presentations/archive.html)

**时间覆盖：** 2005–2026，按年份 + 季度整齐排列，最长的历史

**每季度提供的下载：**

- Stock exchange announcement (= 季度 press release，PDF) — 下这个
- Group databook (XLSX) — 结构化数据，可选
- Presentation slides + script (PDF) — 可选
- Q&A transcript (PDF) — 可选

**手动工作量估算：** 2005–2026 = ~84 个季度。如果只下 press release，就是 84 个 PDF，每个页面点一次，大约 30–40 分钟

**另外：** 还有每年一份 Annual Report（~21 份）

## Shell — 中等难度

[https://www.shell.com/investors/results-and-reporting/quarterly-results.html](https://www.shell.com/investors/results-and-reporting/quarterly-results.html)

**Historical Quarterly Results 部分：** 有历史季度，但页面上目前可见的需要滚动确认

**每季度提供：**

- Quarterly press release (PDF) — 下这个
- Quarterly unaudited results (PDF) — 可选
- Quarterly databook (XLSX) — 可选
- Quarterly slides (PDF) — 可选

**问题：** Shell 的 PDF URL 是 JCR 动态路径（极长），不容易猜测，手动下载必须从页面逐个点击

**手动工作量估算：** 取决于历史数据回溯多远。如果 2015–2026 约 44 个季度 press releases

## TotalEnergies — 页面最友好

[https://totalenergies.com/investors/investors-presentations](https://totalenergies.com/investors/investors-presentations)

**页面结构：** 按年份分组的表格，每行有 Press release | Presentation | Replay 三列，直接点链接下载

**覆盖年份可见：** 2019/2020–2026

**手动工作量估算：** 约 24–28 个季度 press releases，页面结构清晰，10–15 分钟搞定

**另外：** Reports 页面有 URD (Universal Registration Document) = 年报，2019–2025 约 7 份

## 四、遥感数据

### 结构完整性：全部通过


| 数据源        | 月份范围                     | AOI | 行数   | 缺行  |
| ---------- | ------------------------ | --- | ---- | --- |
| Landsat    | 2006-01 ~ 2017-03 (135月) | 11  | 1485 | 0   |
| Sentinel-2 | 2017-04 ~ 2025-12 (105月) | 11  | 1155 | 0   |
| VIIRS 夜光   | 2014-01 ~ 2025-12 (144月) | 11  | 1584 | 0   |


三个 CSV 的 `(site_id × month)` 网格 100% 完整，没有缺失月份或缺失 AOI。

### 数值空值情况（行存在但指标为 NaN）

VIIRS 夜光：零空值，144 个月 × 11 站点全部有数据。

光学指标（云/无影像导致的 NaN）：


| 站点                | Landsat 空月 / 135 | S2 空月 / 105 | 合计有效 / 240 | 覆盖率   |
| ----------------- | ---------------- | ----------- | ---------- | ----- |
| P001 Rotterdam    | 41               | 2           | 197        | 82.1% |
| P002 Fujairah     | 19               | 10          | 211        | 87.9% |
| P003 Ras Tanura   | 14               | 1           | 225        | 93.8% |
| P004 Singapore    | 59               | 27          | 154        | 64.2% |
| P005 Houston      | 11               | 0           | 229        | 95.4% |
| P006 Ningbo       | 20               | 20          | 200        | 83.3% |
| P007 Jamnagar     | 20               | 12          | 208        | 86.7% |
| P008 Basra        | 17               | 4           | 219        | 91.2% |
| P009 Ulsan        | 17               | 18          | 205        | 85.4% |
| P010 Kharg Island | 34               | 9           | 197        | 82.1% |
| P011 Yanbu        | 21               | 2           | 217        | 90.4% |


### 关键发现：

P004 Singapore 覆盖率最低（64.2%），热带地区云覆盖严重，Landsat 期缺 44%、S2 期缺 26%。

P001 Rotterdam、P010 Kharg Island 也偏低（~82%），前者受北欧多云影响，后者受中东沙尘影响。

P005 Houston 最好（95.4%），S2 期间零缺失。

这些 NaN 是正常的云/无数据缺失，前向填充到周频时会被上一个有效月的值覆盖，不影响流水线运行。

## 五、Shipping

#### 第一层：必须保留，进入主模型


| 数据集                     | 你的判断                      | 我的评估        | 建议调整               |
| ----------------------- | ------------------------- | ----------- | ------------------ |
| IMF PortWatch           | 核心，2019–2025              | 完全同意        | 最高优先级，下载成本最低、信号最直接 |
| GFW AIS Vessel Presence | 核心，                       | -           | -                  |
| NOAA AIS                | 核心，下载 Gulf Coast          | 同意方向，但需控制范围 | 工作量容易低估，见下方详述      |
| EMODnet Vessel Density  | 是，偏 monthly/supplementary | 同意          | 作为欧洲航线补充，不是核心      |


 两个关键调整建议

1. NOAA AIS — 工作量风险提醒；建议：不下载

NOAA AIS 原始数据极其庞大（单月 US 沿海数据可达数十 GB），需要：

- 按 vessel type 过滤油轮（MMSI → vessel type 查表）
- 空间裁剪到 Gulf Coast / Houston 区域
- 聚合为港口级别的周频指标（tanker count、waiting time proxy 等）

对硕士论文而言，这是性价比最低的数据源。建议：
如果时间充裕 → 保留，但限定 Houston Ship Channel 一个港口 + 2015–2025 十年
如果时间紧张 → 降级为可选，用 GFW AIS 替代覆盖 2012–2018 的缺口


| 因素      | 分析                                                 |
| ------- | -------------------------------------------------- |
| 覆盖      | 仅美国沿海水域                                            |
| 数据量     | 极其庞大（单月数十 GB 原始 AIS 广播点）                           |
| 处理难度    | 需要 vessel type 过滤、空间裁剪、MMSI 去重、聚合 — 工程量接近一个独立项目    |
| 与现有数据重叠 | Houston 的活动已被 GFW（2012–2018）和 PortWatch（2019+）间接覆盖 |
| 适合场景    | 博士论文 / 专项 AIS 研究，不适合硕士论文的一个模态                      |


结论：性价比最低的数据源。硕士论文完全不需要。

1. EMODnet Vessel Density 建议：暂不下载


| 因素      | 分析                                        |
| ------- | ----------------------------------------- |
| 覆盖      | 仅欧洲水域（北海、地中海）                             |
| 时间      | 2017–2024，月频                              |
| 与现有数据重叠 | GFW 已覆盖 2012–2018，PortWatch 已覆盖 2019–2025 |
| 增量价值    | 仅对欧洲航线（Suez-Rotterdam 方向）提供更高空间分辨率的油轮密度   |
| 工作量     | 需下载 GeoTIFF 栅格文件 → 空间裁剪 → 提取区域均值          |


结论：你的 6 个咽喉中只有 Suez 处于 EMODnet 覆盖范围，而 PortWatch + GFW 已经覆盖了该区域。边际贡献极小，投入产出比低。如果论文写作时需要欧洲航线的额外论据，再回来下也不迟。

#### 第二层：保留但不进主模型 — 完全同意


| 数据集                    | 你的定位                    | 评估                             |
| ---------------------- | ----------------------- | ------------------------------ |
| GFW SAR Detections     | Case study / validation | 正确。可以用来验证"暗船"对 AIS 覆盖缺口的影响     |
| Piraeus AIS            | Method validation       | 正确。展示 AIS → 港口活动指标的清洗管线        |
| VENμS Vessel Detection | Proof-of-concept        | 正确。有时间戳+AIS 标注，适合展示卫星→航运代理的可行性 |


这三个放在论文 Method / Discussion 章节中讨论即可，不需要进入特征矩阵。

#### 第三层：放 Appendix 或删除 — 完全同意


| 数据集            | 你的判断                         | 评估                     |
| -------------- | ---------------------------- | ---------------------- |
| ShipRSImageNet | CV extension                 | 正确。静态图像无时间戳，与周度油价预测无关  |
| OpenSARShip    | SAR classification extension | 正确。同上                  |
| FGSCR-42       | 可删或 appendix                 | 正确。与油价预测关系最弱，可以直接从主表删除 |


## 六、图


| CATEGORY                        | 是否保留      | 原因                                           |
| ------------------------------- | --------- | -------------------------------------------- |
| CRUDE OIL REFINERIES            | 保留，核心     | 炼厂节点，和原油需求、加工能力、供应链中断有关                      |
| PETROLEUM TERMINALS             | 保留，核心     | 油品/原油终端，适合连接港口、航运和储运活动                       |
| LNG FACILITIES                  | 保留，可选核心   | 虽然偏天然气，但能源基础设施和航运/终端活动相关                     |
| OFFSHORE PLATFORMS              | 保留，核心     | 海上生产节点，适合做供应端/遥感/AIS附近活动                     |
| GATHERING AND PROCESSING        | 保留，核心     | 上游处理设施，适合作为油气生产链节点                           |
| TANK BATTERIES                  | 保留，可选     | 储油/集输节点，但数量可能较多，可先保留再筛                       |
| NATURAL GAS COMPRESSOR STATIONS | 可选        | 偏天然气；如果做 oil + gas infrastructure graph 可以保留 |
| OIL AND NATURAL GAS PIPELINES   | 暂时不建议全保留  | 数量可能很大，且线性网络点位处理复杂                           |
| OIL AND NATURAL GAS WELLS       | 第一轮不保留    | 数量巨大，很多是 dry/abandoned，对油价预测噪音大              |
| NATURAL GAS FLARING DETECTIONS  | 可选，不建议第一轮 | 更适合 methane/emissions 项目，不是油价主变量             |
| INJECTION AND DISPOSAL          | 不保留       | 注入/废弃处置类，和油价关系较弱                             |
| EQUIPMENT AND COMPONENTS        | 不保留       | 太零散，空间节点意义弱                                  |
| STATIONS - OTHER                | 不保留       | 类别不清晰，噪音大                                    |


## 七、天气
不建议下载的原因
### 1. 天气对油价的影响已被现有数据间接覆盖

| 天气效应 | 已被哪个数据源捕获 |
|---|---|
| 飓风导致炼厂停产 | EIA 周报中 refinery_utilisation 会直接下降 |
| 极端天气事件冲击 | GDELT oil_geopolitical_daily 会捕获相关新闻 |
| 炼厂活动异常 | 遥感（VIIRS夜光 + 光学指标）会反映 |
| 航运中断 | PortWatch 咽喉点数据会体现 |

### 2. 投入产出比低

ERA5：全球 0.25° 栅格数据量巨大，需要 CDS 注册 + 空间裁剪到你的 11 个 AOI + 时间聚合到周度。工程量大，但产出可能只是几个 binary flag（极端天气 = 1/0）。

IBTrACS：事件极其稀疏，20 年里真正影响石油基础设施的飓风可能就十几次。对周度模型来说样本量太小，几乎不可能被学到。

### 3. 你的模型已经足够复杂

你当前已有 4 个模态（Market/Macro + Text + Remote Sensing + Shipping），再加天气会增加：

- 额外的预处理管线
- 更多的特征维度
- 对硕士论文来说 scope 太大

## 建议

如果你觉得论文 Discussion 中需要讨论天气影响，可以：

- 用 EIA refinery_utilisation 的异常下降点作为 proxy（不用下新数据）
- 在 limitations 中提到"未纳入显式天气变量"作为未来工作方向即可

结论：不下载，省下时间专注于已有四个模态的质量和模型融合。