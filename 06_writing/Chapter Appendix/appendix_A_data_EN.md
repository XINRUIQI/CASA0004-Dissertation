# Appendix A — Dataset Details

---

## A.1 Variable dictionary

All variables were selected from the literature reviewed in Chapter 2.

### A.1.1 Finance / macro (31)

Daily series enter as the Friday last value. Log returns are ln(Pₜ/Pₜ₋₁)
and are not multiplied by 100.

The Literature column cites Chapter 2 sources for inclusion. It does not
claim that the series improves one-week-ahead Brent forecasts.

**Prices & derived (5)**

| Variable | Meaning | Source | Literature |
| --- | --- | --- | --- |
| `brent_price` | Brent spot (USD/bbl) | EIA | EIA (2014); Wittner (2020) |
| `wti_price` | WTI Cushing spot (USD/bbl) | EIA | Hao and Wang (2023) |
| `brent_log_return` | Brent weekly log return | derived | Hao and Wang (2023) |
| `wti_log_return` | WTI weekly log return | derived | Hao and Wang (2023) |
| `brent_wti_spread` | Brent − WTI (USD/bbl) | derived | Alquist, Kilian and Vigfusson (2013) |

**EIA WPSR fundamentals (12)**

Source: EIA *Weekly Petroleum Status Report* for all 12 series.

| Variable | Meaning | Unit | Literature |
| --- | --- | --- | --- |
| `crude_stocks_excl_spr` | commercial crude stocks excl. SPR | thousand barrels | Alquist, Kilian and Vigfusson (2013); Hao and Wang (2023) |
| `cushing_stocks` | Cushing crude stocks | thousand barrels | Alquist, Kilian and Vigfusson (2013); Hao and Wang (2023) |
| `crude_production` | U.S. crude production | thousand bbl/d | Kilian (2009) |
| `crude_imports` | crude imports | thousand bbl/d | Kilian (2009) |
| `crude_exports` | crude exports | thousand bbl/d | Kilian (2009); Alquist, Kilian and Vigfusson (2013) |
| `refinery_crude_input` | refinery crude input | thousand bbl/d | Kilian (2009) |
| `refinery_utilisation` | refinery utilisation | % | Kilian (2009) |
| `gasoline_supplied` | gasoline product supplied | thousand bbl/d | Kilian (2009); Costa et al. (2021) |
| `distillate_supplied` | distillate product supplied | thousand bbl/d | Kilian (2009); Costa et al. (2021) |
| `jet_fuel_supplied` | jet-fuel product supplied | thousand bbl/d | Kilian (2009); Costa et al. (2021) |
| `crude_stocks_change` | weekly change in commercial stocks | thousand barrels | Alquist, Kilian and Vigfusson (2013); Hao and Wang (2023) |
| `cushing_stocks_change` | weekly change in Cushing stocks | thousand barrels | Alquist, Kilian and Vigfusson (2013); Hao and Wang (2023) |

**Macro-financial (5)**

| Variable | Meaning | Source | Literature |
| --- | --- | --- | --- |
| `vix` | CBOE equity-volatility index | FRED VIXCLS | Costa et al. (2021); Yılmaz and Zehir (2026) |
| `dollar_index` | broad nominal USD index | FRED DTWEXBGS | Costa et al. (2021); Yılmaz and Zehir (2026) |
| `treasury_10y` | 10-year Treasury yield | FRED DGS10 | Costa et al. (2021); Yılmaz and Zehir (2026) |
| `fed_funds_rate` | effective federal funds rate | FRED DFF | Costa et al. (2021); Yılmaz and Zehir (2026) |
| `sp500_log_return` | S&P 500 weekly log return | Yahoo ^GSPC | Costa et al. (2021); Yılmaz and Zehir (2026) |

**Derived market/macro (9)**

| Variable | Meaning | Source | Literature |
| --- | --- | --- | --- |
| `ovx` | CBOE crude-oil volatility index | Yahoo ^OVX | Kilian (2009); Yılmaz and Zehir (2026) |
| `gpr` | geopolitical-risk index (weekly mean of daily GPRD) | Caldara–Iacoviello | Kilian (2009); Yılmaz and Zehir (2026) |
| `gold_return` | gold weekly log return | FRED / Yahoo GC=F | Costa et al. (2021) |
| `global_econ_activity` | Kilian real economic activity (REA) | Dallas Fed IGREA | Kilian (2009) |
| `nonoil_industrial_commodity` | non-fuel industrial materials price index | IMF PINDUINDEXM (FRED) | Kilian (2009); Baumeister and Kilian (2015) |
| `brent_f1_spot_log_basis` | front-month futures − spot log basis | Yahoo BZ=F vs EIA spot | Alquist, Kilian and Vigfusson (2013) |
| `brent_roll_week` | front-month roll-week dummy {0,1} | calendar-derived | Alquist, Kilian and Vigfusson (2013) |
| `cadusd_log_return` | CAD/USD weekly log return | Yahoo CADUSD=X | Costa et al. (2021) |
| `dgs10_change` | weekly change in `treasury_10y` | derived | Costa et al. (2021); Yılmaz and Zehir (2026) |

`brent_f1_spot_log_basis` = ln(BZ=F) − ln(`brent_price`); the futures leg is not back-adjusted, so `brent_roll_week` flags the calendar week of the roll. The basis is a public-data proxy for near-term tightness, not a pure calendar spread.

### A.1.2 Remote sensing

**Flat layout (55 = 5 indices × 11 AOI)**

Naming `{index}_anom_{AOI}`; `anom` = within-site deseasonalised z-score
(expanding, past-only). Raw `level` and staleness/mask columns are not modelled.

| Index | Meaning | Formula | Source | Literature |
| --- | --- | --- | --- | --- |
| `NDVI` | vegetation greenness | (B8−B4)/(B8+B4) | Sentinel-2 SR | land-cover control, not oil demand; contrast Bricongne et al. (2026) NO₂ |
| `NDWI` | surface water/moisture | (B3−B8)/(B3+B8) | Sentinel-2 SR | exploratory (water-adjacent terminals) |
| `NDBI` | built-up | (B11−B8)/(B11+B8) | Sentinel-2 SR | Wang et al. (2019); Jung (2026) |
| `BSI` | bare soil / storage yards | ((B11+B4)−(B8+B2))/((B11+B4)+(B8+B2)) | Sentinel-2 SR | Wang et al. (2019); Jung (2026) |
| `NTL` | night-time light activity | VIIRS DNB `avg_rad` | VIIRS DNB | Polinov, Bookman and Levin (2022); Jung (2026) |

The Flat pathway uses within-site anomalies, not raw levels: night-time
lights mainly capture cross-sectional scale, and raw radiance is a poor
tanker proxy (Polinov, Bookman and Levin, 2022). NDBI and BSI mark yards
and tank farms, not tank-fill; filling rates as in Wang et al. (2019)
need sub-metre imagery. Cloud-quality fields filter composites only;
Hao and Wang’s (2023) cloud-cover mechanism is not replicated.

**Deep layout (Prithvi embeddings)**

Frozen **Prithvi-EO-2.0-300M** embeddings (1024-d per AOI-month; Szwarcman
et al., 2026; cf. SatMAE, Cong et al., 2022). Prithvi was pretrained on
six-band NASA HLS. This study uses Sentinel-2 Surface Reflectance
Harmonized patches, not HLS. Weights are frozen because the weekly Brent
sample is too small to fine-tune a 300-million-parameter encoder. VIIRS
is Flat-only.

| Step | Detail |
| --- | --- |
| Bands | Sentinel-2 `B2, B3, B4, B8A, B11, B12` in HLS order (blue, green, red, narrow NIR, SWIR1, SWIR2) |
| Standardise | published Prithvi `config.json` per-band mean and standard deviation; nodata (0) set to 1e-4 |
| Resample | bilinear resize to 224 × 224 |
| Embedding | frozen encoder; mean-pool of patch tokens → 1024-d; weights never updated |

### A.1.3 Shipping

**Flat layout (164 columns)**

Naming `gfw_{cp}_{stat}` and `pw_{cp}_{stat}` over 6 chokepoints, plus
cross-chokepoint aggregates and PortWatch port export/import volumes, plus
`sar_{region}_{total,dark,share}`. The main Flat shipping specification uses
all 164 columns.

| Family | Meaning | Literature |
| --- | --- | --- |
| `gfw_{cp}_total_hours` / `total_vessels` / `cargo_hours` | GFW vessel-presence hours / distinct vessels / cargo hours | Arslanalp, Marini and Tumbarello (2019) |
| `gfw_{cp}_bunker_hours` / `other_hours` / `other_share` | bunker / other-vessel presence & share | Arslanalp, Marini and Tumbarello (2019) |
| `gfw_{cp}_total_hours_mom_pct` / `mean_presence_hours_per_vessel` | month-over-month %; per-vessel congestion proxy | author-derived; congestion concept in Mi et al. (2022) |
| `gfw_all_total_hours_sum` / `gfw_all_activity_zmean` | cross-chokepoint aggregate (sum / leak-free z-mean) | author-derived from GFW |
| `pw_{cp}_n_tanker` / `n_total` / `capacity_tanker` / `capacity` | PortWatch tanker / all-vessel transit count & capacity | Arslanalp, Marini and Tumbarello (2019); Arslanalp et al. (2026); Adland, Jia and Strandenes (2017); Yan et al. (2020) |
| `pw_{cp}_tanker_share` / `tanker_cap_share` / `avg_tanker_size` | tanker shares; average tanker DWT | author-derived from PortWatch |
| `pw_{cp}_n_tanker_wow_pct` / `capacity_tanker_4w_ma` | week-over-week %; 4-week MA | author-derived from PortWatch |
| `pw_all_*` (n_tanker_sum, n_total_sum, tanker_share) | cross-chokepoint tanker aggregates | author-derived from PortWatch |
| `pw_tanker_exp_imp_net` / `_asym` / `_log_ratio` / `_4w_ma` | export−import net / asymmetry / log-ratio | author-derived; directional flows in Yan et al. (2020) |
| `pw_exp_hubs_export_vol` / `pw_imp_hubs_import_vol` (+ `_wow_pct`) | export/import hub tanker tonnage | Arslanalp et al. (2026); Yan et al. (2020) |
| `sar_{region}_{total,dark,share}` | GFW SAR detections: total / unmatched (dark) / dark share; 17 regions × 3 | Paolo et al. (2024) |

PortWatch `n_tanker` is a liquid-bulk transit count, not unique ships and
not crude-only; `capacity_tanker` is a DWT transit-capacity proxy, not
loaded barrels. GFW presence is general maritime activity, not oil
volume. Derived shares, week-over-week changes and cross-chokepoint sums
are author-constructed from native PortWatch and GFW fields.

**Deep layout — AOI node features (11 per node)**

Node feature spaces differ by type. Graph edges follow maritime-network
studies (Ouyang et al., 2022; Liang et al., 2022; Zhao et al., 2022).

| Variable | Meaning | Source | Literature |
| --- | --- | --- | --- |
| `pw_portcalls_tanker` | tanker port calls (weekly sum) | PortWatch | Mi et al. (2022, 2023); Arslanalp et al. (2026) |
| `pw_portcalls_cargo` | cargo port calls | PortWatch | Arslanalp et al. (2026) |
| `pw_import_tanker` | tanker import tonnage | PortWatch | Arslanalp et al. (2026); Yan et al. (2020) |
| `pw_export_tanker` | tanker export tonnage | PortWatch | Arslanalp et al. (2026); Yan et al. (2020) |
| `gfw_n_visits` | port-visit count | GFW AIS | Mi et al. (2022, 2023) |
| `gfw_dwell_hrs_mean` | mean dwell hours | GFW AIS | Mi et al. (2022, 2023) |
| `gfw_dwell_hrs_median` | median dwell hours | GFW AIS | Mi et al. (2022, 2023) |
| `gfw_self_loops` | same-AOI repeat calls | GFW AIS | author-derived from GFW visits |
| `sar_detections_total` | SAR detections | GFW SAR | Paolo et al. (2024) |
| `sar_detections_dark` | unmatched (dark) detections | GFW SAR | Paolo et al. (2024) |
| `sar_dark_share` | dark / total | GFW SAR | Paolo et al. (2024) |

Dwell hours keep only `durationHrs` ≤ 720 h (30 days); longer stays are set to missing.

**Deep layout — chokepoint node features (20 per node)**

Same families as the Flat layout, attached to the six chokepoint nodes rather
than flattened.

| Block | Features | Literature |
| --- | --- | --- |
| GFW (8) | `total_hours`, `total_vessels`, `cargo_hours`, `bunker_hours`, `other_hours`, `other_share`, `total_hours_mom_pct`, `mean_presence_hours_per_vessel` | same as Flat GFW |
| PortWatch (9) | `n_tanker`, `n_total`, `capacity_tanker`, `capacity`, `tanker_share`, `tanker_cap_share`, `avg_tanker_size`, `n_tanker_wow_pct`, `capacity_tanker_4w_ma` | same as Flat PortWatch |
| SAR (3) | `detections_total`, `detections_dark`, `dark_share` | Paolo et al. (2024) |

Mi et al. (2022, 2023) mainly document oil prices affecting tanker port
calls, so shipping is treated as a potentially bidirectional signal, not
a proven leading indicator. SAR dark-vessel detections correct incomplete
public AIS (Paolo et al., 2024); they are not a sanctions variable.

---

## A.2 AOI and chokepoint node lists

### A.2.1 11 oil-infrastructure AOIs

Fixed node order P001–P011 (graph AOI index 0–10). Flat remote-sensing
features use a circular buffer of 5 km radius at every site. Deep Sentinel-2
patches are square and site-specific: 6.4 km for ports, 5.12 km for
refineries, and 1.6–3.2 km for terminals after visual coverage checks.

| Site ID | Site name | Country / region | Facility type | Functional role | Latitude | Longitude | Flat buffer | Deep patch size | Chokepoint |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| P001 | Rotterdam | Netherlands / Europe | port | pricing / import | 51.950 | 4.145 | 5 km | 6.4 km | Suez · Cape |
| P002 | Fujairah | UAE / Middle East | terminal | transit / storage | 25.199 | 56.356 | 5 km | 3.2 km | Hormuz |
| P003 | Ras Tanura | Saudi Arabia / Middle East | terminal | export | 26.643 | 50.157 | 5 km | 2.56 km | Hormuz |
| P004 | Jurong Island | Singapore / Asia | refinery | transit / refining | 1.274 | 103.708 | 5 km | 5.12 km | Malacca |
| P005 | Houston | USA / North America | port | import / refining | 29.736 | −95.100 | 5 km | 6.4 km | Panama |
| P006 | Ningbo-Zhoushan | China / East Asia | port | import | 29.935 | 121.982 | 5 km | 6.4 km | Malacca |
| P007 | Jamnagar | India / South Asia | refinery | refining | 22.345 | 69.860 | 5 km | 5.12 km | Hormuz |
| P008 | Al Basrah Terminal | Iraq / Middle East | terminal | export | 29.681 | 48.810 | 5 km | 1.6 km | Hormuz |
| P009 | Ulsan | South Korea / East Asia | refinery | refining | 35.433 | 129.343 | 5 km | 5.12 km | Malacca |
| P010 | Kharg Island | Iran / Middle East | terminal | export | 29.231 | 50.324 | 5 km | 3.2 km | Hormuz |
| P011 | Yanbu | Saudi Arabia / Middle East | terminal | export | 23.961 | 38.229 | 5 km | 3.2 km | Suez · Mandeb |

Flat buffer is a circular radius. Deep patch size is the side length of a square image chip centred on the same coordinate. The Chokepoint column lists each site's assigned oil-trade corridor(s).

### A.2.2 6 maritime chokepoints

Fixed node order (graph index 11–16), from EIA World Oil Transit Chokepoints.

| Short code | Chokepoint |
| --- | --- |
| `hormuz` | Strait of Hormuz |
| `suez` | Suez Canal |
| `malacca` | Strait of Malacca |
| `mandeb` | Bab el-Mandeb |
| `panama` | Panama Canal |
| `cape` | Cape of Good Hope |

---

## A.3 Publication-lag table

Every predictor enters the weekly (Friday-ending) matrix only after its
conservative as-of availability. Flat and Deep share the same sources but differ
for shipping. This table is the as-of rule used in the main analysis.

### A.3.1 Flat models

| Source | freq → weekly | Lag |
| --- | --- | --- |
| Daily finance (Brent/WTI/VIX/DXY/DGS10/DFF/S&P/gold/OVX/CAD) | daily → Friday last | **0** |
| EIA WPSR fundamentals | weekly → Friday | **+1 w** |
| GPR | daily → weekly mean | **+1 w** |
| Monthly macro (REA, non-oil commodity) | month-end, then carried forward | **+5 w** |
| Sentinel-2 indices + VIIRS (M2) | monthly as-of | **month-end + 15 d** |
| PortWatch chokepoint/port flows | daily → Friday sum | **+1 w** |
| GFW monthly presence (49 of 164 Flat shipping columns) | month-end, then carried forward | **+4 w** |
| GFW SAR dark-vessel (51 of 164 Flat shipping columns) | month-end, then carried forward | **+4 w** |

EIA series are lagged once at construction and are not lagged again at merge.

### A.3.2 Deep models

Finance and remote sensing use the same as-of dates as A.3.1 (Deep remote
sensing = monthly Prithvi embeddings, also month-end + 15 days). Only the
17-node shipping graph differs.

| Graph stream | Role | Lag |
| --- | --- | --- |
| PortWatch node counts | node features | **+1 w** |
| GFW events / voyages (O-D) | edges + node features | **+2 w** |
| GFW SAR dark-vessel | node features | **+4 w** |
| GFW monthly presence (chokepoint node features) | node features | **+4 w** |

---

## A.4 Shipping graph edge definition

Deep models encode shipping as a **weekly 17-node heterogeneous graph**
(11 AOIs + 6 chokepoints, fixed order). Combined adjacency averages about 66
edges per week.

### A.4.1 Dynamic voyage edges (AOI→AOI)

Directed AOI→AOI edges from GFW voyage counts; edge weight = `n_voyages` for
that week's directed lane (`from ≠ to`; self-loops removed to a node feature).
Different every week; 96 lanes, 106 992 voyages total (top lanes e.g.
Ningbo↔Singapore, Fujairah↔Singapore, Singapore↔Rotterdam). Directionality
verified (`P006→P004 ≠ P004→P006`). Lag +2 w.

### A.4.2 Fixed corridor edges (AOI↔chokepoint)

Fixed corridor edges are undirected (13 edges), specified in advance from
each site's main documented oil-trade corridor rather than inferred from weekly
vessel movements or geographic proximity. Present every week.
Every AOI carries at least one fixed corridor edge: P007 (Jamnagar) is a demand-side
refinery rather than a Gulf export terminal, but its crude slate is dominated by
Persian Gulf loadings, so it is attached to Hormuz on the import side.

| Chokepoint | Linked AOIs |
| --- | --- |
| `hormuz` | P002, P003, P007, P008, P010 |
| `suez` | P001, P011 |
| `malacca` | P004, P006, P009 |
| `mandeb` | P011 |
| `cape` | P001 |
| `panama` | P005 |

### A.4.3 Adjacency handling & edge-weight transform

- **Combine**: dynamic voyage-edge block (11×11) placed in the AOI sub-block;
  fixed corridor edges broadcast over all weeks → combined (T, 17, 17). In the
  stored adjacency the O-D block remains directed.
- **Symmetrise + self-loop**: symmetrisation is applied only in the encoder.
  For message passing the adjacency is then symmetrised and self-looped
  (dense 17×17 boolean mask; dense is simpler than sparse for this tiny
  dynamic graph).
- **Edge-weight transform (attention prior)**: `log1p` of the symmetrised O-D
  flow is **added to the GAT attention logits**, scaled by a **learned gain
  `edge_scale`**, then softmax; it is not a multiplier on the attention weights
  and is not used as an edge feature in message passing. Busy lanes therefore
  receive a higher prior, and the model can down-weight it if unhelpful.
- **Encoder**: type-specific projection (`F_aoi=11`, `F_choke=20` → `d_model=64`)
  + node-type embedding → 2-layer dense multi-head GAT (heads = 4, LeakyReLU
  slope 0.2) → causal TCN (kernel 3; lookback L) → node-attention pooling →
  32-d `z_ship` (~42k params). Node-attention weights feed RQ3 (which
  port/chokepoint the branch weights). Encoder details in Appendix C.

---

## A.5 Flat remote-sensing coverage

Site-level rates for Section 3.4.3. Weekly-calendar coverage counts a month
for every later Friday that still uses it, so it is not a count of independent
site–month composites.

| Site | Weekly-calendar S2 anomaly coverage | First Friday with S2 anomaly | Monthly S2 composite completeness | Weekly NTL anomaly |
| --- | --- | --- | --- | --- |
| Houston | 100.0% | 2019-01-04 | 100% | 100% |
| Rotterdam | 100.0% | 2019-01-04 | 100% | 100% |
| Jamnagar | 100.0% | 2019-01-04 | 100% | 100% |
| Al Basrah Terminal, Fujairah, Kharg, Ras Tanura, Yanbu | 100.0% | 2019-01-04 | 100% | 100% |
| Ulsan | 93.4% | 2019-06-21 | 100% | 100% |
| Ningbo-Zhoushan | 89.9% | 2019-09-20 | 100% | 100% |
| Jurong Island | 83.8% | 2020-02-21 | 98.4% | 100% |

The shortfall at Ulsan and Ningbo-Zhoushan is the expanding 12-month history
required to define an anomaly, not missing monthly composites. Jurong Island
combines that warm-up with residual cloud gaps.
