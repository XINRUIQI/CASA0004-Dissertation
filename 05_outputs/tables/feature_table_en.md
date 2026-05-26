# Feature Inventory — Multimodal Weekly Feature Matrix

**Dataset:** 1,043 weeks (2006-01-06 ~ 2025-12-26) | **263 features** across 4 modalities | **Unified frequency:** Friday-ending weekly

---

## M1 — Market + Macro Fundamentals (27 features)

Baseline modality: Brent/WTI prices, EIA Weekly Petroleum Status Report, FRED macro-financial indicators.

| # | Feature | Description | Source | Original Freq. | Coverage |
|---|---------|-------------|--------|----------------|----------|
| 1 | `brent_price` | Brent spot price (USD/bbl) | EIA | Daily → Wk (last) | 100% |
| 2 | `wti_price` | WTI Cushing spot price (USD/bbl) | EIA | Daily → Wk (last) | 100% |
| 3 | `brent_wti_spread` | Brent minus WTI spread | Derived | Weekly | 100% |
| 4 | `brent_return_pct` | Brent week-over-week return (%) | Derived | Weekly | 100% |
| 5 | `wti_return_pct` | WTI week-over-week return (%) | Derived | Weekly | 100% |
| 6 | `brent_log_return` | Brent log return | Derived | Weekly | 100% |
| 7 | `brent_vol_4w` | 4-week rolling volatility (log return std) | Derived | Weekly | 100% |
| 8 | `brent_vol_12w` | 12-week rolling volatility | Derived | Weekly | 100% |
| 9 | `crude_stocks_excl_spr` | US commercial crude stocks excl. SPR (k bbl) | EIA WPSR | Weekly | 100% |
| 10 | `cushing_stocks` | Cushing OK crude stocks (k bbl) | EIA WPSR | Weekly | 100% |
| 11 | `crude_production` | US crude production (k bpd) | EIA WPSR | Weekly | 100% |
| 12 | `crude_imports` | US crude imports (k bpd) | EIA WPSR | Weekly | 100% |
| 13 | `crude_exports` | US crude exports (k bpd) | EIA WPSR | Weekly | 100% |
| 14 | `refinery_crude_input` | Refinery crude oil input (k bpd) | EIA WPSR | Weekly | 100% |
| 15 | `refinery_utilisation` | Refinery utilisation rate (%) | EIA WPSR | Weekly | 100% |
| 16 | `gasoline_supplied` | Motor gasoline supplied (k bpd) | EIA WPSR | Weekly | 100% |
| 17 | `distillate_supplied` | Distillate fuel oil supplied (k bpd) | EIA WPSR | Weekly | 100% |
| 18 | `jet_fuel_supplied` | Kerosene-type jet fuel supplied (k bpd) | EIA WPSR | Weekly | 100% |
| 19 | `crude_stocks_change` | Weekly change in crude stocks | Derived | Weekly | 100% |
| 20 | `cushing_stocks_change` | Weekly change in Cushing stocks | Derived | Weekly | 100% |
| 21 | `net_crude_trade` | Net crude trade (imports − exports) | Derived | Weekly | 100% |
| 22 | `sp500` | S&P 500 index close | Yahoo Finance | Daily → Wk (last) | 100% |
| 23 | `vix` | CBOE VIX close | FRED | Daily → Wk (last) | 100% |
| 24 | `dollar_index` | Trade-weighted USD index (DTWEXBGS) | FRED | Daily → Wk (last) | 100% |
| 25 | `treasury_10y` | 10-year Treasury yield (%) | FRED | Daily → Wk (last) | 100% |
| 26 | `fed_funds_rate` | Effective federal funds rate (%) | FRED | Daily → Wk (last) | 100% |
| 27 | `sp500_return_pct` | S&P 500 weekly return (%) | Derived | Weekly | 99.9% |

---

## M2 — Text / Event Signals (+26 features → cumulative 53)

GDELT 1.0/2.0 calibrated daily events aggregated to weekly frequency. Two event domains: oil-disruption intensity and transport-disruption signals.

| # | Feature | Description | Source | Aggregation | Coverage |
|---|---------|-------------|--------|-------------|----------|
| 1 | `gdelt_oil_geo_event_count` | Weekly oil-disruption event count | GDELT | Daily sum | 100% |
| 2 | `gdelt_oil_geo_total_mentions` | Total media mentions of oil-disruption events | GDELT | Daily sum | 100% |
| 3 | `gdelt_oil_geo_avg_tone` | Average media tone of oil-disruption events | GDELT | Daily mean | 100% |
| 4 | `gdelt_oil_geo_avg_goldstein` | Average Goldstein scale of oil-disruption events | GDELT | Daily mean | 100% |
| 5 | `gdelt_negative_event_count` | Negative oil-disruption event count | GDELT | Daily sum | 100% |
| 6 | `gdelt_conflict_event_count` | Conflict-related oil-disruption event count | GDELT | Daily sum | 100% |
| 7 | `gdelt_sanction_event_count` | Sanction-related oil-disruption event count | GDELT | Daily sum | 100% |
| 8 | `gdelt_key_oil_region_event_count` | Events in key oil-producing regions | GDELT | Daily sum | 100% |
| 9 | `gdelt_transport_disruption_event_count` | Transport disruption event count | GDELT | Daily sum | 100% |
| 10 | `gdelt_transport_disruption_total_mentions` | Media mentions of transport disruption | GDELT | Daily sum | 100% |
| 11 | `gdelt_transport_disruption_avg_tone` | Average tone of transport disruption events | GDELT | Daily mean | 100% |
| 12 | `gdelt_transport_disruption_avg_goldstein` | Average Goldstein scale (transport) | GDELT | Daily mean | 100% |
| 13 | `gdelt_transport_negative_event_count` | Negative transport events | GDELT | Daily sum | 100% |
| 14 | `gdelt_transport_unrest_conflict_event_count` | Unrest / conflict transport events | GDELT | Daily sum | 100% |
| 15 | `gdelt_transport_sanction_event_count` | Sanction-related transport events | GDELT | Daily sum | 100% |
| 16 | `gdelt_chokepoint_event_count` | Chokepoint-related events | GDELT | Daily sum | 100% |
| 17 | `gdelt_oil_geo_event_count_4w_ma` | 4-week MA of oil-disruption event count | Derived | Rolling | 100% |
| 18 | `gdelt_oil_geo_event_count_wow_pct` | WoW % change in oil-disruption event count | Derived | — | 99.9% |
| 19 | `gdelt_oil_geo_negative_share` | Share of negative events (oil-disruption) | Derived | — | 100% |
| 20 | `gdelt_oil_geo_conflict_share` | Share of conflict events (oil-disruption) | Derived | — | 100% |
| 21 | `gdelt_oil_geo_avg_tone_4w_ma` | 4-week MA of oil-disruption avg tone | Derived | Rolling | 100% |
| 22 | `gdelt_transport_event_count_4w_ma` | 4-week MA of transport event count | Derived | Rolling | 100% |
| 23 | `gdelt_transport_event_count_wow_pct` | WoW % change in transport event count | Derived | — | 99.9% |
| 24 | `gdelt_transport_negative_share` | Share of negative events (transport) | Derived | — | 100% |
| 25 | `gdelt_transport_avg_tone_4w_ma` | 4-week MA of transport avg tone | Derived | Rolling | 100% |
| 26 | `gdelt_combined_event_count` | Combined oil-disruption + transport event count | Derived | — | 100% |

---

## M3 — Remote Sensing (+110 features → cumulative 163)

10 feature types × 11 AOI sites = 110 features. Optical indices from Sentinel-2 (2017–2025) and Landsat (2006–2017) backfill; nighttime lights from VIIRS DNB (2012–2025).

**11 AOI sites:**

| ID | Site | Type | Country |
|----|------|------|---------|
| P001 | Port of Rotterdam | Port | Netherlands |
| P002 | Fujairah Oil Terminal | Terminal | UAE |
| P003 | Ras Tanura Terminal | Terminal | Saudi Arabia |
| P004 | Jurong Island | Refinery | Singapore |
| P005 | Houston Ship Channel | Port | United States |
| P006 | Ningbo-Zhoushan Port | Port | China |
| P007 | Jamnagar Refinery | Refinery | India |
| P008 | Basra Oil Terminal | Terminal | Iraq |
| P009 | Ulsan Refinery | Refinery | South Korea |
| P010 | Kharg Island Terminal | Terminal | Iran |
| P011 | Yanbu Export Terminal | Terminal | Saudi Arabia |

**Per-AOI feature types (×11 sites each):**

| # | Feature Pattern | Description | Source | Freq. | Coverage |
|---|----------------|-------------|--------|-------|----------|
| 1 | `opt_NDVI_{site}` | Normalised Difference Vegetation Index | Sentinel-2 / Landsat | Monthly → Wk (ffill) | 99–100% |
| 2 | `opt_NDWI_{site}` | Normalised Difference Water Index | Sentinel-2 / Landsat | Monthly → Wk (ffill) | 99–100% |
| 3 | `opt_NDBI_{site}` | Normalised Difference Built-up Index | Sentinel-2 / Landsat | Monthly → Wk (ffill) | 99–100% |
| 4 | `opt_BSI_{site}` | Bare Soil Index | Sentinel-2 / Landsat | Monthly → Wk (ffill) | 99–100% |
| 5 | `opt_valid_obs_count_{site}` | Valid observation count per month | Sentinel-2 / Landsat | Monthly → Wk | 100% |
| 6 | `opt_sensor_flag_{site}` | Sensor source (0 = Landsat, 1 = S2) | Derived | Monthly → Wk | 100% |
| 7 | `ntl_ntl_avg_rad_mean_{site}` | VIIRS nightlight mean radiance | VIIRS DNB | Monthly → Wk (ffill) | 60% |
| 8 | `ntl_ntl_avg_rad_max_{site}` | VIIRS nightlight max radiance | VIIRS DNB | Monthly → Wk (ffill) | 60% |
| 9 | `ntl_ntl_avg_rad_stddev_{site}` | VIIRS nightlight radiance std dev | VIIRS DNB | Monthly → Wk (ffill) | 60% |
| 10 | `ntl_ntl_cf_cvg_mean_{site}` | VIIRS cloud-free coverage fraction | VIIRS DNB | Monthly → Wk (ffill) | 60% |

> Note: NTL (VIIRS) coverage is 60% because VIIRS data starts from 2012-04, covering ~626 of 1,043 weeks.

---

## M4 — Shipping (+100 features → cumulative 263)

Maritime shipping activity at 6 oil-critical chokepoints. GFW 4Wings vessel presence (monthly, 2012–2018) + IMF PortWatch daily transits (2019–2025) combined.

**6 chokepoints:** Strait of Hormuz, Suez Canal, Strait of Malacca, Bab el-Mandeb, Panama Canal, Cape of Good Hope.

**GFW features (8 types × 6 chokepoints + 1 aggregate = 49):**

| # | Feature Pattern | Description | Source | Freq. | Coverage |
|---|----------------|-------------|--------|-------|----------|
| 1 | `gfw_{choke}_total_hours` | Total vessel presence hours | GFW 4Wings | Monthly → Wk (ffill) | 69.7% |
| 2 | `gfw_{choke}_total_vessels` | Total distinct vessels | GFW 4Wings | Monthly → Wk | 69.7% |
| 3 | `gfw_{choke}_cargo_hours` | Cargo vessel hours | GFW 4Wings | Monthly → Wk | 69.7% |
| 4 | `gfw_{choke}_bunker_hours` | Bunker vessel hours | GFW 4Wings | Monthly → Wk | 69.7% |
| 5 | `gfw_{choke}_other_hours` | Other vessel hours | GFW 4Wings | Monthly → Wk | 69.7% |
| 6 | `gfw_{choke}_nontanker_hours` | Non-tanker vessel hours | GFW 4Wings | Monthly → Wk | 69.7% |
| 7 | `gfw_{choke}_other_share` | Other vessel share of total | Derived | Monthly → Wk | 69.7% |
| 8 | `gfw_{choke}_total_hours_mom_pct` | MoM % change in vessel hours | Derived | Monthly → Wk | 69.3% |
| 9 | `gfw_all_total_hours_sum` | All-chokepoint total hours | Derived | Weekly | 69.7% |

**PortWatch features (8 types × 6 chokepoints + 3 aggregates = 51):**

| # | Feature Pattern | Description | Source | Freq. | Coverage |
|---|----------------|-------------|--------|-------|----------|
| 1 | `pw_{choke}_n_tanker` | Weekly tanker transit count | IMF PortWatch | Daily sum → Wk | 35.0% |
| 2 | `pw_{choke}_n_total` | Weekly total vessel transit count | IMF PortWatch | Daily sum → Wk | 35.0% |
| 3 | `pw_{choke}_capacity_tanker` | Weekly tanker capacity (DWT) | IMF PortWatch | Daily sum → Wk | 35.0% |
| 4 | `pw_{choke}_capacity` | Weekly total vessel capacity (DWT) | IMF PortWatch | Daily sum → Wk | 35.0% |
| 5 | `pw_{choke}_tanker_share` | Tanker share of total transits | Derived | Weekly | 35.0% |
| 6 | `pw_{choke}_tanker_cap_share` | Tanker capacity share | Derived | Weekly | 35.0% |
| 7 | `pw_{choke}_n_tanker_wow_pct` | WoW % change in tanker transits | Derived | Weekly | 34.9% |
| 8 | `pw_{choke}_capacity_tanker_4w_ma` | 4-week MA of tanker capacity | Derived | Weekly | 35.0% |
| 9 | `pw_all_n_tanker_sum` | All-chokepoint tanker transit sum | Derived | Weekly | 35.0% |
| 10 | `pw_all_n_total_sum` | All-chokepoint total transit sum | Derived | Weekly | 35.0% |
| 11 | `pw_all_tanker_share` | All-chokepoint tanker share | Derived | Weekly | 35.0% |

> Note: GFW covers 2012–2018 (69.7%); PortWatch covers 2019–2025 (35.0%). Together they span the full study period for shipping activity.

---

## Target Variables

| Feature | Description | Type | Definition |
|---------|-------------|------|------------|
| `target_brent_price_next_1w` | Next-week Brent price level (USD/bbl) | Regression | Friday closing price of the following week |
| `target_brent_vol_next_1w` | Next-week realised volatility | Regression | Std dev of daily log returns over next 5 trading days |
| `target_brent_direction_next_1w` | Next-week price direction (3-class) | Classification | 1 = up (>+0.5%), 0 = flat (±0.5%), −1 = down (<−0.5%) |

---

## Ablation Experiment Design

| Experiment | Modalities | Cumulative Features |
|------------|-----------|-------------------|
| **M1** | Market + Macro | 27 |
| **M2** | M1 + Text / GDELT | 53 |
| **M3** | M2 + Remote Sensing | 163 |
| **M4** | M3 + Shipping | 263 |
