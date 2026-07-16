# Appendix A — Data: variable dictionary, AOI/chokepoint lists, lags, graph edges

# 附录 A — 数据：变量词典、AOI/咽喉列表、滞后期表、航运图边定义

> Merged weekly matrix `weekly_feature_matrix.csv` = 365 weeks × 212 columns
> (2019-01-04 → 2025-12-26): 31 M1 + 55 M2 + 113 M3 + 11 masks + 2 targets.
> Per-variable literature/industry sourcing is in the modality data dictionaries
> (`03_data/processed/M{1,2,3}/*_data_dictionary.md`); this appendix consolidates
> the model-facing dictionary, the site lists, the publication lags and the
> shipping-graph edge construction.
>
> 合并周频矩阵 = 365 周 × 212 列（31 M1 + 55 M2 + 113 M3 + 11 掩膜 + 2 目标）。
> 逐变量文献/业界来源见各模态数据字典；本附录汇编入模词典、站点列表、发布滞后与
> 航运图边构造。

Information sets / 信息集：**M0** random walk (`brent_price` only) · **M1**
finance (31) · **M2** M1 + remote sensing (55) · **M3** M1 + shipping full (113)
· **M4** M1+M2+M3 (199). Flat flattens each column over a 4-week lookback
(lag 0–3); Deep keeps modality structure inside encoders.

---

## A.1 Variable dictionary (M1–M4) / 变量词典

### A.1.1 M1 — Finance / macro (31) / 金融·宏观

**Prices & derived (5)**: `brent_price`, `wti_price`, `brent_log_return`,
`wti_log_return`, `brent_wti_spread` (EIA spot; log returns = ln(Pₜ/Pₜ₋₁)).

**EIA WPSR fundamentals (12, +1 w)** / EIA 周报基本面：
`crude_stocks_excl_spr`, `cushing_stocks`, `crude_production`, `crude_imports`,
`crude_exports`, `refinery_crude_input`, `refinery_utilisation`,
`gasoline_supplied`, `distillate_supplied`, `jet_fuel_supplied`,
`crude_stocks_change`, `cushing_stocks_change`.

**Macro-financial (5)** / 宏观金融：`vix` (FRED VIXCLS), `dollar_index`
(DTWEXBGS), `treasury_10y` (DGS10), `fed_funds_rate` (DFF),
`sp500_log_return` (^GSPC).

**Derived market/macro (9)** / 衍生市场·宏观：`ovx`, `gpr`, `gold_return`,
`global_econ_activity` (Kilian REA), `nonoil_industrial_commodity`
(IMF PINDUINDEXM), `brent_f1_spot_log_basis`, `brent_roll_week` (dummy),
`cadusd_log_return`, `dgs10_change`.

### A.1.2 M2 — Remote sensing (55 = 5 indices × 11 AOI) / 遥感

Naming `{index}_anom_{AOI}`; `anom` = within-site deseasonalised z-score
(expanding, past-only). Raw `level` and staleness/mask columns are not modelled.
命名 `{指数}_anom_{AOI}`；主分析用站内去季节距平。

| Index / 指数 | Meaning / 含义 | Formula / 波段 | Source |
| --- | --- | --- | --- |
| `NDVI` | vegetation greenness / 植被绿度 | (B8−B4)/(B8+B4) | Sentinel-2 SR |
| `NDWI` | surface water/moisture / 水面·湿度 | (B3−B8)/(B3+B8) | Sentinel-2 SR |
| `NDBI` | built-up / 建成区 | (B11−B8)/(B11+B8) | Sentinel-2 SR |
| `BSI` | bare soil / storage yards / 裸土·堆场 | ((B11+B4)−(B8+B2))/((B11+B4)+(B8+B2)) | Sentinel-2 SR |
| `NTL` | night-time light activity / 夜光活动 | VIIRS DNB `avg_rad` | VIIRS DNB |

> Literature arm (C1) = `NTL_anom` of Fujairah / RasTanura / Rotterdam / Houston
> (4 cols). / 文献精选臂 = 4 站 NTL_anom。

### A.1.3 M3 — Shipping, flat full tier (113 = GFW 49 + PortWatch 64) / 航运扁平主层

Naming `gfw_{cp}_{stat}` and `pw_{cp}_{stat}` over 6 chokepoints, plus
cross-chokepoint aggregates and PortWatch port export/import volumes. Main model
= full 113 (the hand-picked 38-col *core* tier is a robustness arm; full is XGB-
optimal, see Appendix B / `m3_data_dictionary.md` §11).

| Family / 变量族 | Meaning / 含义 |
| --- | --- |
| `gfw_{cp}_total_hours` / `total_vessels` / `cargo_hours` | GFW vessel-presence hours / distinct vessels / cargo hours / 存在时长·船数·货船时长 |
| `gfw_{cp}_bunker_hours` / `other_hours` / `other_share` | bunker / other-vessel presence & share / 加油船·其他船 |
| `gfw_{cp}_total_hours_mom_pct` / `mean_presence_hours_per_vessel` | month-over-month %; per-vessel congestion proxy / 月环比·每船拥堵代理 |
| `gfw_all_total_hours_sum` / `gfw_all_activity_zmean` | cross-chokepoint aggregate (sum / leak-free z-mean) / 跨咽喉聚合 |
| `pw_{cp}_n_tanker` / `n_total` / `capacity_tanker` / `capacity` | PortWatch tanker / all-vessel transit count & capacity / 过境艘次·运力 |
| `pw_{cp}_tanker_share` / `tanker_cap_share` / `avg_tanker_size` | tanker shares; average tanker DWT / 占比·平均吨位 |
| `pw_{cp}_n_tanker_wow_pct` / `capacity_tanker_4w_ma` | week-over-week %; 4-week MA / 周环比·4 周均值 |
| `pw_all_*` (n_tanker_sum, n_total_sum, tanker_share) | cross-chokepoint tanker aggregates / 跨咽喉汇总 |
| `pw_tanker_exp_imp_net` / `_asym` / `_log_ratio` / `_4w_ma` | export−import net / asymmetry / log-ratio / 出口−进口净额·不对称 |
| `pw_exp_hubs_export_vol` / `pw_imp_hubs_import_vol` (+ `_wow_pct`) | export/import hub tanker tonnage / 出口·进口枢纽吨位 |

### A.1.4 Deep shipping graph node features (not in flat matrix) / 深度图节点特征

The Deep arm does **not** use the flat 113 columns; it builds a 17-node graph
(`m3_graph17_tensors.npz`). Node feature spaces differ by type (heterogeneous).
深度臂用 17 节点图，节点特征按类型异质。

**AOI node features (11 per AOI node)** / AOI 节点特征：
`pw_portcalls_tanker`, `pw_portcalls_cargo`, `pw_import_tanker`,
`pw_export_tanker`, `gfw_n_visits`, `gfw_dwell_hrs_mean`, `gfw_dwell_hrs_median`,
`gfw_self_loops`, `sar_detections_total`, `sar_detections_dark`, `sar_dark_share`.

**Chokepoint node features (20 per node = GFW 8 + PortWatch 9 + SAR 3)** /
咽喉节点特征：GFW `{total_hours, total_vessels, cargo_hours, bunker_hours,
other_hours, other_share, total_hours_mom_pct, mean_presence_hours_per_vessel}`;
PortWatch `{n_tanker, n_total, capacity_tanker, capacity, tanker_share,
tanker_cap_share, avg_tanker_size, n_tanker_wow_pct, capacity_tanker_4w_ma}`;
SAR `{detections_total, detections_dark, dark_share}`.

The Deep RS branch uses frozen **Prithvi-EO-2.0 embeddings** (1024-d per
AOI-month) rather than the M2 indices; VIIRS is Flat-only. / 深度 RS 分支用冻结
Prithvi-EO-2.0 嵌入（每 AOI-月 1024 维），不含 VIIRS。

---

## A.2 AOI and chokepoint node lists / AOI 与咽喉节点列表

### A.2.1 11 oil-infrastructure AOIs / 11 个石油基础设施 AOI

Fixed node order P001–P010 (graph AOI index 0–10). 5 km analysis buffer;
AOI-differentiated Sentinel-2 patch sizes. Source:
`aoi_oil_infrastructure_sites.md`.

| ID | Site / 站点 | Country / 国家 | Type / 类型 | Role / 角色 | Chokepoint / 关联咽喉 | (lon, lat) |
| --- | --- | --- | --- | --- | --- | --- |
| P001 | Rotterdam / 鹿特丹 | Netherlands | port | pricing / import | Suez · Cape | 4.145, 51.950 |
| P002 | Fujairah / 富查伊拉 | UAE | terminal | transit / storage | Hormuz | 56.356, 25.199 |
| P003 | Ras Tanura / 拉斯塔努拉 | Saudi Arabia | terminal | export | Hormuz | 50.157, 26.643 |
| P004 | Jurong Island / 裕廊岛 | Singapore | refinery | transit / refining | Malacca | 103.708, 1.274 |
| P005 | Houston / 休斯顿 | USA | port | import / refining | Panama | −95.100, 29.736 |
| P006 | Ningbo-Zhoushan / 宁波舟山 | China | port | import | Malacca | 121.982, 29.935 |
| P007 | Jamnagar / 贾姆纳格尔 | India | refinery | refining | — | 69.860, 22.345 |
| P008 | Basra / 巴士拉 | Iraq | terminal | export | Hormuz | 48.810, 29.681 |
| P009 | Ulsan / 蔚山 | South Korea | refinery | refining | Malacca | 129.343, 35.433 |
| P010 | Kharg Island / 哈格岛 | Iran | terminal | export | Hormuz | 50.324, 29.231 |
| P011 | Yanbu / 延布 | Saudi Arabia | terminal | export | Suez · Mandeb | 38.229, 23.961 |

### A.2.2 6 maritime chokepoints / 6 个海运咽喉

Fixed node order (graph index 11–16), from EIA World Oil Transit Chokepoints.

| Short code | Chokepoint / 咽喉 |
| --- | --- |
| `hormuz` | Strait of Hormuz / 霍尔木兹海峡 |
| `suez` | Suez Canal / 苏伊士运河 |
| `malacca` | Strait of Malacca / 马六甲海峡 |
| `mandeb` | Bab el-Mandeb / 曼德海峡 |
| `panama` | Panama Canal / 巴拿马运河 |
| `cape` | Cape of Good Hope / 好望角 |

---

## A.3 Publication-lag table / 发布滞后总表

Every predictor enters the weekly (Friday-ending) matrix only after its
conservative real-time availability. Lags are fixed as constants at the top of
each builder. Flat and Deep share the same sources but differ for shipping.
每个变量只在其保守实时可得后进入周频矩阵；扁平与深度共享来源但航运不同。

### A.3.1 Flat arm / 扁平臂

| Source / 来源 | freq → weekly | Lag | Constant · script |
| --- | --- | --- | --- |
| Daily finance (Brent/WTI/VIX/DXY/DGS10/DFF/S&P/gold/OVX/CAD) | daily → Fri last | **0** | `daily_to_weekly_last` · `build_m1_weekly.py` |
| EIA WPSR fundamentals | weekly → Fri | **+1 w** | `EIA_LAG_WEEKS=1` · `build_m1_weekly.py` |
| GPR | daily → weekly mean | **+1 w** | `GPR_LAG_WEEKS=1` · `build_m1_weekly.py` |
| Monthly macro (REA, non-oil commodity) | month-end ffill | **+5 w** | `MONTHLY_LAG_WEEKS=5` · `build_m1_weekly.py` |
| Sentinel-2 indices + VIIRS (M2) | monthly as-of | **month-end + 15 d** | `PUB_LAG_DAYS=15` · `build_m2_weekly.py` |
| PortWatch chokepoint/port flows | daily → Fri sum | **+1 w** | `PW_LAG_WEEKS=1` · `aggregate_shipping_to_weekly.py` |
| GFW monthly presence (flat M3, 113 cols) | month-end ffill | **+4 w** | `GFW_LAG_WEEKS=4` · `aggregate_shipping_to_weekly.py` |

> Merge check: EIA already lagged at source, merge re-shift = 0
> (`EIA_WPSR_LAG_WEEKS=0`, `build_feature_matrix.py`). / 合并仅复查，不再二次移位。

### A.3.2 Deep arm (17-node graph) / 深度臂

Finance and RS identical to A.3.1 (Deep RS = Channel-A Prithvi embeddings, also
month-end + 15 d). Only the shipping graph differs.

| Graph stream / 图数据流 | Role | Lag | Constant · script |
| --- | --- | --- | --- |
| PortWatch node counts | node features | **+1 w** | `PW_LAG_WEEKS=1` · `build_m3_graph_weekly.py` |
| GFW events / voyages (O-D) | edges + node features | **+2 w** | `GFW_EVENT_LAG_WEEKS=2` · `build_m3_graph_weekly.py` |
| GFW SAR dark-vessel | node features | **+4 w** | `SAR_LAG_WEEKS=4` · `build_m3_graph_weekly.py` |
| GFW monthly presence (chokepoint node features) | node features | **+4 w** | `GFW_LAG_WEEKS=4` (inherited) · `build_m3_graph17.py` |
| EMODnet density (optional cross-check, not in model) | — | **+8 w** | `EMODNET_LAG_WEEKS=8` · `build_emodnet_weekly.py` |

### A.3.3 Why GFW is +4 w (Flat) but +2 w (Deep) / 为何 GFW 扁平 +4、深度 +2

Different GFW products, not the same stream lagged differently. **Flat +4 w** =
monthly vessel-presence columns (`gfw_{cp}_*`, 49 of the 113): a calendar month
is only complete at month end + a conservative ~1-month availability buffer
(project-level conservatism, **not** an official 4-week release rule). **Deep
+2 w** = near-real-time AIS event/voyage O-D stream (~96 h) with a conservative
two-week buffer. The two are not interchangeable.

两个数字对应不同 GFW 产品：扁平 +4 周用于月频存在列（保守可得性缓冲，非官方发布
规则）；深度 +2 周用于近实时 AIS 事件/航次 O-D。不可互换。

### A.3.4 Lag robustness / 滞后稳健性

GFW monthly presence testable at lag ∈ {1, 4, 8} w; `MONTHLY_LAG_WEEKS` at
{3, 5, 7} w; all exposed as CLI flags (`--gfw-lag`, `--eia-lag`, …) so the whole
matrix can be rebuilt without code edits. Results in Appendix B.

---

## A.4 Shipping graph edge definition / 航运图边定义

The Deep shipping branch encodes a **weekly 17-node heterogeneous graph**
(11 AOIs + 6 chokepoints, fixed order). Combined adjacency is (T, 17, 17),
averaging ~63.8 edges/week. Sources: `build_m3_graph17.py`,
`m3_data_dictionary.md` §12, `shipping_encoder.py`.

### A.4.1 Dynamic O-D voyage edges (AOI→AOI) / 动态航次 O-D 边

Directed AOI→AOI edges from GFW voyage counts; edge weight = `n_voyages` for
that week's directed lane (`from ≠ to`; self-loops removed to a node feature).
Different every week; 96 lanes, 106 992 voyages total (top lanes e.g.
Ningbo↔Singapore, Fujairah↔Singapore, Singapore↔Rotterdam). Directionality
verified (`P006→P004 ≠ P004→P006`). Lag +2 w.

有向 AOI→AOI 边，边权 = 当周该有向 lane 的 `n_voyages`；每周动态；有向性经自检。

### A.4.2 Static AOI↔chokepoint edges / 静态 AOI↔咽喉边

Fixed undirected links by geographic association (12 undirected edges), present
every week (`aoi_oil_infrastructure_sites.md` §4):

| Chokepoint | Linked AOIs |
| --- | --- |
| `hormuz` | P002, P003, P008, P010 |
| `suez` | P001, P011 |
| `malacca` | P004, P006, P009 |
| `mandeb` | P011 |
| `cape` | P001 |
| `panama` | P005 |

### A.4.3 Adjacency handling & edge-weight transform / 邻接处理与边权变换

- **Combine**: dynamic O-D block (11×11) placed in the AOI sub-block; static
  AOI↔chokepoint edges broadcast over all weeks → combined (T, 17, 17). /
  动态 O-D 块 + 静态边广播 → 组合邻接。
- **Symmetrise + self-loop**: for message passing the adjacency is symmetrised
  and self-looped (dense 17×17 boolean mask; dense is simpler than sparse for
  this tiny dynamic graph). / 消息传递前对称化 + 自环。
- **Edge-weight transform (attention prior)**: the O-D flow enters the GAT as
  `log1p(flow)` scaled by a **learned gain `edge_scale`**, i.e. busy lanes get a
  higher attention prior instead of the flow being discarded by the boolean
  adjacency; the model can down-weight the prior if unhelpful. / 边权变换：O-D
  流量以 `log1p(flow)` × 可学习增益 `edge_scale` 作为注意力先验，繁忙航道获更高
  先验，模型可自适应弱化。
- **Encoder**: type-specific projection (`F_aoi=11`, `F_choke=20` → `d_model=64`)
  + node-type embedding → 2-layer dense multi-head GAT (heads = 4) → causal TCN
  (lookback L) → node-attention pooling → 32-d `z_ship` (~42k params). Node-
  attention weights feed RQ3 (which port/chokepoint the branch weights). /
  编码器见附录 C。
