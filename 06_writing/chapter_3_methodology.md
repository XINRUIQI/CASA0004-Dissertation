# Chapter 3 — Methodology

## 3.1 Research design

## 3.2 Data sources
<!-- Dataset inventory table (link to 03_data/external_sources.md) -->
<!-- Dataset feasibility assessment -->

### 3.2.X Remote sensing site selection: oil infrastructure AOIs

This study extracts monthly spectral indices (NDVI, NDWI, NDBI, BSI) from Sentinel-2 and Landsat, and nighttime radiance from VIIRS, at a set of oil-infrastructure areas of interest (AOIs). Each AOI is defined as a 5 km circular buffer around the facility centroid. The selection of 11 globally distributed sites follows a three-criterion framework: **throughput/capacity rank**, **geographic and supply-chain diversity**, and **remote sensing observability**.

#### Criterion 1 — Throughput and capacity rank

Each selected site ranks among the highest-capacity facilities in its functional category, as documented by industry-standard sources. The quantitative justification is summarised in Table 3.X.

**Table 3.X — Selected AOI sites and capacity rankings**

| ID | Site | Type | Country | Capacity / Throughput | Rank | Source |
|----|------|------|---------|-----------------------|------|--------|
| P001 | Port of Rotterdam | Port | Netherlands | 397M tonnes/yr (2024) | Europe #1 port | Eurostat (2025) |
| P002 | Fujairah Oil Terminal | Terminal | UAE | 1.5M bpd ADCOP bypass + global #2 bunkering hub | Strategic bypass | ADNOC; Marine Insight |
| P003 | Ras Tanura Terminal | Terminal | Saudi Arabia | 9M bpd design capacity; 90% of Saudi exports | World #1 export terminal | IMF PortWatch; Saudi Aramco |
| P004 | Jurong Island | Refinery | Singapore | 605,000 bpd | World #12 refinery | Oil & Gas Journal via Wikipedia |
| P005 | Houston Ship Channel | Port | United States | >3M bpd combined refining cluster | US #1 waterborne tonnage port | US DoT; OGIM |
| P006 | Ningbo-Zhoushan Port | Port | China | 1.37B tonnes/yr (2024); 185M tonnes crude | World #1 cargo port; China #1 crude import | Xinhua; Global Economic Indicator |
| P007 | Jamnagar Refinery | Refinery | India | 1,240,000 bpd | World #1 refinery | Oil & Gas Journal via Wikipedia |
| P008 | Basra Oil Terminal | Terminal | Iraq | >3.3M bpd; 95%+ of Iraq exports | Middle East #3 export terminal | Marine Insight |
| P009 | Ulsan Refinery | Refinery | South Korea | 840,000 bpd | World #3 refinery | SK Energy; Oil & Gas Journal |
| P010 | Kharg Island Terminal | Terminal | Iran | 1.5–2.5M bpd; 90–96% of Iran exports | Iran sole major export hub | Kpler; Iran International (2026) |
| P011 | Yanbu Export Terminal | Terminal | Saudi Arabia | 4.5M bpd nominal loading capacity | Red Sea #1 crude terminal | Argus Media; Aramco |

Note: The OGIM database (Global Oil Infrastructure Mapping Project, May 2024) confirms the existence and operational status of all 11 facilities but does not include throughput or capacity fields. Capacity figures are therefore sourced from the Oil & Gas Journal world refinery survey, IMF PortWatch, Eurostat maritime transport statistics, and operator reports as cited above.

#### Criterion 2 — Geographic and supply-chain diversity

The 11 sites are distributed across three functional stages of the global oil supply chain and five geographic regions, ensuring that no single region or supply-chain segment dominates the remote sensing signal:

- **Supply / export** (4 sites): Ras Tanura, Basra, Kharg Island, and Yanbu collectively represent the three largest OPEC producers (Saudi Arabia, Iraq, Iran) and cover both Persian Gulf and Red Sea export routes.
- **Transit / storage** (2 sites): Fujairah (Hormuz bypass and bunkering) and Jurong Island (Malacca Strait refining node) capture mid-chain logistics activity.
- **Demand / import-refining** (5 sites): Rotterdam (European Brent hub), Houston (US Gulf Coast), Ningbo-Zhoushan (China demand proxy), Jamnagar (India demand proxy), and Ulsan (East Asian refining centre) represent the major consumption regions.

This design mirrors the geographic structure of the six chokepoint-level shipping features (Hormuz, Suez, Malacca, Bab el-Mandeb, Panama, Cape of Good Hope), allowing cross-modal spatial alignment between remote sensing and maritime indicators.

#### Criterion 3 — Remote sensing observability

All selected sites feature large-footprint ground infrastructure (tank farms, refinery complexes, port berths) that produce measurable spectral and radiance signatures at the spatial resolutions of Sentinel-2 (10–20 m), Landsat (30 m), and VIIRS (∼750 m):

- **Optical indices**: NDBI (Normalised Difference Built-up Index) captures changes in built-up area and impervious surface density; BSI (Bare Soil Index) responds to construction activity and cleared land; NDVI and NDWI provide complementary signals of vegetation displacement and water-body changes associated with terminal expansion or land reclamation.
- **Nighttime radiance**: VIIRS DNB monthly composites capture continuous nighttime operational intensity. Halpern et al. (2022) demonstrate a strong correlation (Spearman Rs = 0.84, p < 0.01) between VIIRS nighttime light intensity and container port throughput across 601 global anchorage areas, validating NTL as a proxy for port-level economic activity. The AOI buffer radius of 5 km follows the design used by Guetta-Jeanrenaud et al. (2025), who define port AOIs as square buffers with a 3 km radius centred on World Port Index coordinates; we adopt a slightly larger circular buffer to accommodate the spatial extent of major refinery and terminal complexes.

#### Precedents in the literature

The selection of a focused set of globally representative sites is consistent with recent satellite-based oil market studies:

- **Wang et al. (2023)** select 8 US oil storage areas (Cushing OK, Houston TX, Texas City TX, Beaumont TX, Corpus Christi TX, Lake Charles LA, Baton Rouge LA, Mississippi River LA) based on inventory concentration: PADD 2 (Cushing) and PADD 3 (Gulf Coast) account for over 70% of total US crude oil stocks. Their per-site FRT cloudiness measures predict WTI weekly returns with statistical significance for 7 of 8 areas.
- **Guetta-Jeanrenaud et al. (2025)** train port-level trade prediction models on 64 US ports classified as Small or larger by the NGA World Port Index, combining SAR (Sentinel-1), nighttime lights (VIIRS), and WPI attributes. Oil terminal depth emerges as the single most important predictor, while satellite variables capture temporal dynamics that static port attributes cannot.
- **Elvidge et al. (2023)** use VIIRS Nightfire to automatically identify ∼20,000 persistent infrared emitters globally, including refineries and gas flares, demonstrating that satellite-detected industrial heat signatures serve as reliable monthly production proxies.
- **ECB Working Paper 3198 (2025)** uses Sentinel-5P TROPOMI tropospheric NO₂ as a real-time proxy for oil demand across advanced and emerging economies, achieving significant improvements in nowcasting accuracy — an approach that complements the facility-level optical and NTL indicators used in this study.

#### Robustness: leave-one-AOI-out sensitivity analysis

To verify that the model results are not driven by the inclusion or exclusion of any single site, a leave-one-AOI-out sensitivity analysis is conducted alongside the modality-level ablation study (M1 → M4). For each of the 11 AOIs, the remote sensing features derived from that site are excluded, and the M3 (market + text + RS) and M4 (full multimodal) models are re-estimated. The resulting distribution of performance changes (ΔRMSE, ΔR²) across all 11 leave-out runs provides evidence of whether the remote sensing contribution is robust to site selection or unduly influenced by individual AOIs. Results are reported in Section 4.X.

#### References

- Elvidge, C.D. et al. (2023). Global satellite monitoring of exothermic industrial activity via infrared emissions. *Remote Sensing*, 15(19), 4760.
- Guetta-Jeanrenaud, L. et al. (2025). Watching trade from space: Nowcasting and spatial extrapolation of port-level maritime trade using satellite imagery. *arXiv:2604.15444*.
- Halpern, B.S. et al. (2022). A global assessment of night lights as an indicator for shipping activity in anchorage areas. *Remote Sensing*, 14(5), 1079.
- Wang, Y. et al. (2023). Cloud cover and expected oil returns. *Humanities and Social Sciences Communications*, 10, 60.
- ECB (2025). Can satellites predict oil demand? *ECB Working Paper Series*, No. 3198.

## 3.3 Data preprocessing
<!-- Cleaning, alignment, feature engineering -->

## 3.4 Analytical framework
<!-- Model architecture -->
<!-- Model selection justification -->

## 3.5 Evaluation metrics

## 3.6 Ethical considerations
