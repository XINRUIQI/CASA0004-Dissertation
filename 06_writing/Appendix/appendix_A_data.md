# Appendix A — Data: variable dictionary, AOI/chokepoint lists, lags, graph edges

# 附录 A — 数据：变量词典、AOI/咽喉列表、滞后期表、航运图边定义

---

## A.1 Variable dictionary / 变量词典

### A.1.1 Finance / macro (31) / 金融·宏观

Daily series enter as the Friday last value. Log returns are ln(Pₜ/Pₜ₋₁)
and are not multiplied by 100. 

日频序列取周五最后值。对数收益为 ln(Pₜ/Pₜ₋₁)，不乘 100。

**Prices & derived (5)** / 价格与派生


| Variable           | Meaning / 含义                        | Source  |
| ------------------ | ----------------------------------- | ------- |
| `brent_price`      | Brent spot (USD/bbl) / Brent 现货     | EIA     |
| `wti_price`        | WTI Cushing spot (USD/bbl) / WTI 现货 | EIA     |
| `brent_log_return` | Brent weekly log return / 周对数收益     | derived |
| `wti_log_return`   | WTI weekly log return / 周对数收益       | derived |
| `brent_wti_spread` | Brent − WTI (USD/bbl) / 价差          | derived |


**EIA WPSR fundamentals (12)** / EIA 周报基本面


| Variable                | Meaning / 含义                                       | Unit             |
| ----------------------- | -------------------------------------------------- | ---------------- |
| `crude_stocks_excl_spr` | commercial crude stocks excl. SPR / 商业原油库存（不含 SPR） | thousand barrels |
| `cushing_stocks`        | Cushing crude stocks / Cushing 原油库存                | thousand barrels |
| `crude_production`      | U.S. crude production / 美国原油产量                     | thousand bbl/d   |
| `crude_imports`         | crude imports / 原油进口                               | thousand bbl/d   |
| `crude_exports`         | crude exports / 原油出口                               | thousand bbl/d   |
| `refinery_crude_input`  | refinery crude input / 炼厂原油加工量                     | thousand bbl/d   |
| `refinery_utilisation`  | refinery utilisation / 炼厂开工率                       | %                |
| `gasoline_supplied`     | gasoline product supplied / 汽油表观需求                 | thousand bbl/d   |
| `distillate_supplied`   | distillate product supplied / 馏分油表观需求              | thousand bbl/d   |
| `jet_fuel_supplied`     | jet-fuel product supplied / 航空煤油表观需求               | thousand bbl/d   |
| `crude_stocks_change`   | weekly change in commercial stocks / 商业库存周变化       | thousand barrels |
| `cushing_stocks_change` | weekly change in Cushing stocks / Cushing 库存周变化    | thousand barrels |


**Macro-financial (5)** / 宏观金融


| Variable           | Meaning / 含义                             | Source        |
| ------------------ | ---------------------------------------- | ------------- |
| `vix`              | CBOE equity-volatility index / 股票隐含波动率   | FRED VIXCLS   |
| `dollar_index`     | broad nominal USD index / 美元名义广义指数       | FRED DTWEXBGS |
| `treasury_10y`     | 10-year Treasury yield / 10 年期美债收益率      | FRED DGS10    |
| `fed_funds_rate`   | effective federal funds rate / 联邦基金有效利率  | FRED DFF      |
| `sp500_log_return` | S&P 500 weekly log return / 标普 500 周对数收益 | Yahoo ^GSPC   |


**Derived market/macro (9)** / 衍生市场·宏观


| Variable                      | Meaning / 含义                                                 | Source                 |
| ----------------------------- | ------------------------------------------------------------ | ---------------------- |
| `ovx`                         | CBOE crude-oil volatility index / 原油隐含波动率                    | Yahoo ^OVX             |
| `gpr`                         | geopolitical-risk index (weekly mean of daily GPRD) / 地缘政治风险 | Caldara–Iacoviello     |
| `gold_return`                 | gold weekly log return / 黄金周对数收益                             | FRED / Yahoo GC=F      |
| `global_econ_activity`        | Kilian real economic activity (REA) / 全球实体经济活动               | Dallas Fed IGREA       |
| `nonoil_industrial_commodity` | non-fuel industrial materials price index / 非燃料工业原料价格指数      | IMF PINDUINDEXM (FRED) |
| `brent_f1_spot_log_basis`     | front-month futures − spot log basis / 前月期货—现货对数基差           | Yahoo BZ=F vs EIA spot |
| `brent_roll_week`             | front-month roll-week dummy {0,1} / 换月周哑变量                   | calendar-derived       |
| `cadusd_log_return`           | CAD/USD weekly log return / 加元周对数收益                          | Yahoo CADUSD=X         |
| `dgs10_change`                | weekly change in `treasury_10y` / 10 年美债收益率周变化               | derived                |


`brent_f1_spot_log_basis` = ln(BZ=F) − ln(`brent_price`); the futures leg is not back-adjusted, so `brent_roll_week` flags the calendar week of the roll. / 期货腿未 back-adjust，故以换月周哑变量控制合约切换。

### A.1.2 Remote sensing / 遥感

**Flat layout (55 = 5 indices × 11 AOI)** / 扁平布局

Naming `{index}_anom_{AOI}`; `anom` = within-site deseasonalised z-score
(expanding, past-only). Raw `level` and staleness/mask columns are not modelled.
命名 `{指数}_anom_{AOI}`；主分析用站内去季节距平。


| Index / 指数 | Meaning / 含义                      | Formula / 波段                          | Source        |
| ---------- | --------------------------------- | ------------------------------------- | ------------- |
| `NDVI`     | vegetation greenness / 植被绿度       | (B8−B4)/(B8+B4)                       | Sentinel-2 SR |
| `NDWI`     | surface water/moisture / 水面·湿度    | (B3−B8)/(B3+B8)                       | Sentinel-2 SR |
| `NDBI`     | built-up / 建成区                    | (B11−B8)/(B11+B8)                     | Sentinel-2 SR |
| `BSI`      | bare soil / storage yards / 裸土·堆场 | ((B11+B4)−(B8+B2))/((B11+B4)+(B8+B2)) | Sentinel-2 SR |
| `NTL`      | night-time light activity / 夜光活动  | VIIRS DNB `avg_rad`                   | VIIRS DNB     |


**Deep layout (Prithvi embeddings)** / 深度布局

Frozen **Prithvi-EO-2.0-300M** embeddings (1024-d per AOI-month). Prithvi was
pretrained on six-band NASA HLS. This study uses Sentinel-2 Surface Reflectance
Harmonized patches, not HLS. VIIRS is Flat-only.

冻结 **Prithvi-EO-2.0-300M** 嵌入（每 AOI-月 1024 维）。Prithvi 以六波段 NASA HLS
预训练；本研究用 Sentinel-2 地表反射率和谐化影像块，而非 HLS。VIIRS 仅用于 Flat。


| Step / 步骤         | Detail / 细节                                                                                      |
| ----------------- | ------------------------------------------------------------------------------------------------ |
| Bands / 波段        | Sentinel-2 `B2, B3, B4, B8A, B11, B12` in HLS order (blue, green, red, narrow NIR, SWIR1, SWIR2) |
| Standardise / 标准化 | published Prithvi `config.json` per-band mean and standard deviation; nodata (0) set to 1e-4     |
| Resample / 重采样    | bilinear resize to 224 × 224                                                                     |
| Embedding / 嵌入    | frozen encoder; mean-pool of patch tokens → 1024-d; weights never updated                        |


### A.1.3 Shipping / 航运

**Flat layout (164 columns)** / 扁平布局

Naming `gfw_{cp}_{stat}` and `pw_{cp}_{stat}` over 6 chokepoints, plus
cross-chokepoint aggregates and PortWatch port export/import volumes, plus
`sar_{region}_{total,dark,share}`. The main Flat shipping specification uses
all 164 columns.


| Family / 变量族                                                       | Meaning / 含义                                                                                    |
| ------------------------------------------------------------------ | ----------------------------------------------------------------------------------------------- |
| `gfw_{cp}_total_hours` / `total_vessels` / `cargo_hours`           | GFW vessel-presence hours / distinct vessels / cargo hours / 存在时长·船数·货船时长                       |
| `gfw_{cp}_bunker_hours` / `other_hours` / `other_share`            | bunker / other-vessel presence & share / 加油船·其他船                                                |
| `gfw_{cp}_total_hours_mom_pct` / `mean_presence_hours_per_vessel`  | month-over-month %; per-vessel congestion proxy / 月环比·每船拥堵代理                                    |
| `gfw_all_total_hours_sum` / `gfw_all_activity_zmean`               | cross-chokepoint aggregate (sum / leak-free z-mean) / 跨咽喉聚合                                     |
| `pw_{cp}_n_tanker` / `n_total` / `capacity_tanker` / `capacity`    | PortWatch tanker / all-vessel transit count & capacity / 过境艘次·运力                                |
| `pw_{cp}_tanker_share` / `tanker_cap_share` / `avg_tanker_size`    | tanker shares; average tanker DWT / 占比·平均吨位                                                     |
| `pw_{cp}_n_tanker_wow_pct` / `capacity_tanker_4w_ma`               | week-over-week %; 4-week MA / 周环比·4 周均值                                                         |
| `pw_all_*` (n_tanker_sum, n_total_sum, tanker_share)               | cross-chokepoint tanker aggregates / 跨咽喉汇总                                                      |
| `pw_tanker_exp_imp_net` / `_asym` / `_log_ratio` / `_4w_ma`        | export−import net / asymmetry / log-ratio / 出口−进口净额·不对称                                         |
| `pw_exp_hubs_export_vol` / `pw_imp_hubs_import_vol` (+ `_wow_pct`) | export/import hub tanker tonnage / 出口·进口枢纽吨位                                                    |
| `sar_{region}_{total,dark,share}`                                  | GFW SAR detections: total / unmatched (dark) / dark share; 17 regions × 3 / SAR 检测总数·未匹配（暗船）·占比 |


**Deep layout — AOI node features (11 per node)** / 深度布局：AOI 节点特征

Node feature spaces differ by type. / 节点特征按类型异质。


| Variable               | Meaning / 含义                           | Source    |
| ---------------------- | -------------------------------------- | --------- |
| `pw_portcalls_tanker`  | tanker port calls (weekly sum) / 油轮停靠数 | PortWatch |
| `pw_portcalls_cargo`   | cargo port calls / 货船停靠数               | PortWatch |
| `pw_import_tanker`     | tanker import tonnage / 油轮进口吨位         | PortWatch |
| `pw_export_tanker`     | tanker export tonnage / 油轮出口吨位         | PortWatch |
| `gfw_n_visits`         | port-visit count / 到港次数                | GFW AIS   |
| `gfw_dwell_hrs_mean`   | mean dwell hours / 平均停留时长              | GFW AIS   |
| `gfw_dwell_hrs_median` | median dwell hours / 中位停留时长            | GFW AIS   |
| `gfw_self_loops`       | same-AOI repeat calls / 同站重复停靠         | GFW AIS   |
| `sar_detections_total` | SAR detections / SAR 检测总数              | GFW SAR   |
| `sar_detections_dark`  | unmatched (dark) detections / 未匹配暗船    | GFW SAR   |
| `sar_dark_share`       | dark / total / 暗船占比                    | GFW SAR   |


Dwell hours keep only `durationHrs` ≤ 720 h (30 days); longer stays are set to missing. / 停留时长仅保留 ≤ 720 小时的停靠，超长置缺失。

**Deep layout — chokepoint node features (20 per node)** / 深度布局：咽喉节点特征

Same families as the Flat layout, attached to the six chokepoint nodes rather
than flattened. / 与扁平布局同族，挂在六个咽喉节点上而非扁平化。


| Block         | Features                                                                                                                                                 |
| ------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------- |
| GFW (8)       | `total_hours`, `total_vessels`, `cargo_hours`, `bunker_hours`, `other_hours`, `other_share`, `total_hours_mom_pct`, `mean_presence_hours_per_vessel`     |
| PortWatch (9) | `n_tanker`, `n_total`, `capacity_tanker`, `capacity`, `tanker_share`, `tanker_cap_share`, `avg_tanker_size`, `n_tanker_wow_pct`, `capacity_tanker_4w_ma` |
| SAR (3)       | `detections_total`, `detections_dark`, `dark_share`                                                                                                      |


---



## A.2 AOI and chokepoint node lists / AOI 与咽喉节点列表



### A.2.1 11 oil-infrastructure AOIs / 11 个石油基础设施 AOI

Fixed node order P001–P011 (graph AOI index 0–10). Flat remote-sensing
features use a circular buffer of 5 km radius at every site. Deep Sentinel-2
patches are square and site-specific: 6.4 km for ports, 5.12 km for
refineries, and 1.6–3.2 km for terminals after visual coverage checks.

固定节点顺序为 P001–P011（图中 AOI 索引 0–10）。Flat 遥感特征在各站统一使用
半径 5 km 的圆形缓冲区。Deep 的 Sentinel-2 影像块为正方形，按站点类型与目视
覆盖校核设定边长：港口 6.4 km，炼厂 5.12 km，码头 1.6–3.2 km。


| Site ID | Site name / 站点         | Country / region           | Facility type / 设施类型 | Functional role / 功能角色 | Latitude | Longitude | Flat buffer | Deep patch size | Chokepoint / 关联咽喉 |
| ------- | ---------------------- | -------------------------- | -------------------- | ---------------------- | -------- | --------- | ----------- | --------------- | ----------------- |
| P001    | Rotterdam / 鹿特丹        | Netherlands / Europe       | port                 | pricing / import       | 51.950   | 4.145     | 5 km        | 6.4 km          | Suez · Cape       |
| P002    | Fujairah / 富查伊拉        | UAE / Middle East          | terminal             | transit / storage      | 25.199   | 56.356    | 5 km        | 3.2 km          | Hormuz            |
| P003    | Ras Tanura / 拉斯塔努拉     | Saudi Arabia / Middle East | terminal             | export                 | 26.643   | 50.157    | 5 km        | 2.56 km         | Hormuz            |
| P004    | Jurong Island / 裕廊岛    | Singapore / Asia           | refinery             | transit / refining     | 1.274    | 103.708   | 5 km        | 5.12 km         | Malacca           |
| P005    | Houston / 休斯顿          | USA / North America        | port                 | import / refining      | 29.736   | −95.100   | 5 km        | 6.4 km          | Panama            |
| P006    | Ningbo-Zhoushan / 宁波舟山 | China / East Asia          | port                 | import                 | 29.935   | 121.982   | 5 km        | 6.4 km          | Malacca           |
| P007    | Jamnagar / 贾姆纳格尔       | India / South Asia         | refinery             | refining               | 22.345   | 69.860    | 5 km        | 5.12 km         | Hormuz            |
| P008    | Al Basrah Terminal / 巴士拉码头 | Iraq / Middle East         | terminal             | export                 | 29.681   | 48.810    | 5 km        | 1.6 km          | Hormuz            |
| P009    | Ulsan / 蔚山             | South Korea / East Asia    | refinery             | refining               | 35.433   | 129.343   | 5 km        | 5.12 km         | Malacca           |
| P010    | Kharg Island / 哈格岛     | Iran / Middle East         | terminal             | export                 | 29.231   | 50.324    | 5 km        | 3.2 km          | Hormuz            |
| P011    | Yanbu / 延布             | Saudi Arabia / Middle East | terminal             | export                 | 23.961   | 38.229    | 5 km        | 3.2 km          | Suez · Mandeb     |


Flat buffer is a circular radius. Deep patch size is the side length of a square image chip centred on the same coordinate. The Chokepoint column lists each site's assigned oil-trade corridor(s).

Flat 缓冲为圆形半径；Deep 裁剪边长为正方形影像块边长，中心与站点坐标相同。Chokepoint 列为各站对应的石油贸易走廊。

### A.2.2 6 maritime chokepoints / 6 个海运咽喉

Fixed node order (graph index 11–16), from EIA World Oil Transit Chokepoints.


| Short code | Chokepoint / 咽喉           |
| ---------- | ------------------------- |
| `hormuz`   | Strait of Hormuz / 霍尔木兹海峡 |
| `suez`     | Suez Canal / 苏伊士运河        |
| `malacca`  | Strait of Malacca / 马六甲海峡 |
| `mandeb`   | Bab el-Mandeb / 曼德海峡      |
| `panama`   | Panama Canal / 巴拿马运河      |
| `cape`     | Cape of Good Hope / 好望角   |


---



## A.3 Publication-lag table / 发布滞后总表

Every predictor enters the weekly (Friday-ending) matrix only after its
conservative as-of availability. Flat and Deep share the same sources but differ
for shipping. This table is the as-of rule used in the main analysis.
每个变量只在其保守 as-of 可得日后进入周频矩阵。Flat 与 Deep 共享来源，但航运滞后不同。本表即主分析使用的 as-of 规则。

### A.3.1 Flat models / Flat 模型


| Source / 来源                                                  | freq → weekly                   | Lag / 滞后             |
| ------------------------------------------------------------ | ------------------------------- | -------------------- |
| Daily finance (Brent/WTI/VIX/DXY/DGS10/DFF/S&P/gold/OVX/CAD) | daily → Friday last             | **0**                |
| EIA WPSR fundamentals                                        | weekly → Friday                 | **+1 w**             |
| GPR                                                          | daily → weekly mean             | **+1 w**             |
| Monthly macro (REA, non-oil commodity)                       | month-end, then carried forward | **+5 w**             |
| Sentinel-2 indices + VIIRS (M2)                              | monthly as-of                   | **month-end + 15 d** |
| PortWatch chokepoint/port flows                              | daily → Friday sum              | **+1 w**             |
| GFW monthly presence (49 of 164 Flat shipping columns)       | month-end, then carried forward | **+4 w**             |
| GFW SAR dark-vessel (51 of 164 Flat shipping columns)        | month-end, then carried forward | **+4 w**             |


EIA series are lagged once at construction and are not lagged again at merge.

### A.3.2 Deep models / Deep 模型

Finance and remote sensing use the same as-of dates as A.3.1 (Deep remote
sensing = monthly Prithvi embeddings, also month-end + 15 days). Only the
17-node shipping graph differs.


| Graph stream / 图数据流                             | Role / 用途             | Lag / 滞后 |
| ----------------------------------------------- | --------------------- | -------- |
| PortWatch node counts                           | node features         | **+1 w** |
| GFW events / voyages (O-D)                      | edges + node features | **+2 w** |
| GFW SAR dark-vessel                             | node features         | **+4 w** |
| GFW monthly presence (chokepoint node features) | node features         | **+4 w** |




### A.3.3 Why GFW is +4 w (Flat) but +2 w (Deep) / 为何 GFW 扁平 +4、深度 +2

These are different GFW products, not one stream lagged two ways. Flat +4 w
is monthly vessel presence, with a conservative availability buffer rather
than an official four-week release. Deep +2 w is near-real-time AIS voyage
O–D. The two are not interchangeable.

两个数字对应不同 GFW 产品，不是同一数据流改滞后。Flat +4 周为月频船舶存在
（保守可得性缓冲，非官方发布规则）；Deep +2 周为近实时 AIS 航次 O–D。不可互换。

---



## A.4 Shipping graph edge definition / 航运图边定义

Deep models encode shipping as a **weekly 17-node heterogeneous graph**
(11 AOIs + 6 chokepoints, fixed order). Combined adjacency averages about 66
edges per week.

### A.4.1 Dynamic O-D voyage edges (AOI→AOI) / 动态航次 O-D 边

Directed AOI→AOI edges from GFW voyage counts; edge weight = `n_voyages` for
that week's directed lane (`from ≠ to`; self-loops removed to a node feature).
Different every week; 96 lanes, 106 992 voyages total (top lanes e.g.
Ningbo↔Singapore, Fujairah↔Singapore, Singapore↔Rotterdam). Directionality
verified (`P006→P004 ≠ P004→P006`). Lag +2 w.

有向 AOI→AOI 边，边权 = 当周该有向 lane 的 `n_voyages`；每周动态；有向性经自检。

### A.4.2 Static AOI↔chokepoint edges / 静态 AOI↔咽喉边

Fixed undirected corridor links (13 undirected edges), specified in advance from
each site's main documented oil-trade corridor rather than inferred from weekly
vessel movements or geographic proximity. Present every week.
Every AOI carries at least one corridor link: P007 (Jamnagar) is a demand-side
refinery rather than a Gulf export terminal, but its crude slate is dominated by
Persian Gulf loadings, so it is attached to Hormuz on the import side.
固定无向走廊边（13 条），按各站点主要石油贸易走廊预先设定，而非由周度船舶移动
或地理邻近关系推断。每周均在。每个 AOI 至少有一条走廊边；P007 为需求侧炼厂，
原油进料以波斯湾装载为主，故在进口侧连接霍尔木兹。


| Chokepoint | Linked AOIs                  |
| ---------- | ---------------------------- |
| `hormuz`   | P002, P003, P007, P008, P010 |
| `suez`     | P001, P011                   |
| `malacca`  | P004, P006, P009             |
| `mandeb`   | P011                         |
| `cape`     | P001                         |
| `panama`   | P005                         |




### A.4.3 Adjacency handling & edge-weight transform / 邻接处理与边权变换

- **Combine**: dynamic O-D block (11×11) placed in the AOI sub-block; static
AOI↔chokepoint edges broadcast over all weeks → combined (T, 17, 17). /
动态 O-D 块 + 静态边广播 → 组合邻接。
- **Symmetrise + self-loop**: for message passing the adjacency is symmetrised
and self-looped (dense 17×17 boolean mask; dense is simpler than sparse for
this tiny dynamic graph). / 消息传递前对称化 + 自环。
- **Edge-weight transform (attention prior)**: `log1p` of the symmetrised O-D  
flow is **added to the GAT attention logits**, scaled by a **learned gain  
`edge_scale**`, then softmax; it is not a multiplier on the attention weights  
and is not used as an edge feature in message passing. Busy lanes therefore  
receive a higher prior, and the model can down-weight it if unhelpful. /  
边权变换：对称化后的 O-D 流量取 `log1p`，乘以可学习增益后  
**加到 GAT 注意力 logits 上**再 softmax；不是乘在注意力权重上，也不作为  
边特征进入消息传递。

---



## A.5 Flat remote-sensing coverage / Flat 遥感覆盖

Site-level rates for Section 3.4.3. Weekly-calendar coverage counts a month
for every later Friday that still uses it, so it is not a count of independent
site–month composites.

对应第 3.4.3 节的站点覆盖。周历覆盖会把同一月合成计到后续仍在使用它的每个周五，
故不是独立站点–月合成数。


| Site                                      | Weekly-calendar S2 anomaly coverage | First Friday with S2 anomaly | Monthly S2 composite completeness | Weekly NTL anomaly |
| ----------------------------------------- | ----------------------------------- | ---------------------------- | --------------------------------- | ------------------ |
| Houston                                   | 100.0%                              | 2019-01-04                   | 100%                              | 100%               |
| Rotterdam                                 | 100.0%                              | 2019-01-04                   | 100%                              | 100%               |
| Jamnagar                                  | 100.0%                              | 2019-01-04                   | 100%                              | 100%               |
| Al Basrah Terminal, Fujairah, Kharg, Ras Tanura, Yanbu | 100.0%                              | 2019-01-04                   | 100%                              | 100%               |
| Ulsan                                     | 93.4%                               | 2019-06-21                   | 100%                              | 100%               |
| Ningbo-Zhoushan                           | 89.9%                               | 2019-09-20                   | 100%                              | 100%               |
| Jurong Island                             | 83.8%                               | 2020-02-21                   | 98.4%                             | 100%               |


The shortfall at Ulsan and Ningbo-Zhoushan is the expanding 12-month history
required to define an anomaly, not missing monthly composites. Jurong Island
combines that warm-up with residual cloud gaps.

蔚山与宁波舟山的缺口来自构造距平所需的 12 个月历史，而不是月度合成缺失。裕廊岛
同时包含这段预热与残余云缺口。