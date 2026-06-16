# Reading Note — Paper 03: Remote Sensing as Economic Activity Proxy

## Citation

Lehnert, P., Niederberger, M., Backes-Gellner, U., & Bettinger, E. (2023). Proxying Economic Activity with Daytime Satellite Imagery: Filling Data Gaps across Time and Space. *PNAS Nexus*, 2(4), pgad099.

- **DOI**: [10.1093/pnasnexus/pgad099](https://doi.org/10.1093/pnasnexus/pgad099)
- **Published**: April 2023 (Open Access)
- **IZA Discussion Paper**: No. 15555 (September 2022)

---

## Core Method

- **Data source**: Landsat daytime satellite imagery (30-m resolution), spanning a historical time series from 1984 to present.
- **Approach**: Machine-learning models applied to land surface classification features extracted from multispectral satellite imagery to construct a continuous economic activity proxy.
- **Surface groups**: Satellite pixels classified into land cover categories (built-up, vegetation, water, bare soil, etc.), with the proportional composition of surface types serving as the predictive feature set.
- **Proxy construction**: OLS and machine-learning regression models map surface composition to known economic indicators (GDP, employment, tax revenue) at various regional granularities.
- **Validation**: Cross-validated against official economic statistics in Germany, including both former East and West German regions; outperforms nighttime light (NTL) intensity as an economic proxy.

---

## Key Findings

1. **Daytime imagery outperforms nighttime lights**: The Landsat-based proxy predicts economic activity more precisely at smaller regional levels and over longer time horizons than DMSP-OLS or VIIRS nighttime light intensity data.
2. **30-m spatial resolution enables fine-grained analysis**: Unlike NTL data (which saturates in urban cores and has ~750m resolution), Landsat imagery captures within-city variation in economic development.
3. **Historical coverage back to 1984**: The Landsat archive enables construction of economic proxies for periods and regions where no official statistics exist (e.g., East Germany before reunification), enabling longitudinal economic analysis.
4. **Global generalisability**: The procedure is designed to be transferable to any region worldwide, particularly valuable for developing countries with poor statistical infrastructure.
5. **Land cover composition is informative**: The share of built-up area, impervious surfaces, and vegetation patterns are highly predictive of local economic output, validating the "physical footprint" hypothesis of economic activity.

---

## Relevance to This Dissertation

| Aspect | Connection |
|--------|------------|
| **Methodological foundation** | Establishes the scientific basis for using satellite imagery as an economic/industrial activity proxy — a core assumption in this dissertation's remote sensing pipeline. |
| **Landsat & Sentinel-2 link** | The paper uses Landsat imagery; Sentinel-2 (used in this dissertation) offers comparable or superior spectral bands at 10-m resolution, supporting the extension of this methodology. |
| **Feature engineering** | The surface composition approach (built-up %, vegetation indices, impervious surface ratio) directly informs the spectral feature engineering strategy for oil infrastructure monitoring in this dissertation. |
| **Oil infrastructure monitoring** | If satellite-derived surface features proxy economic activity, they can similarly proxy oil-sector activity: refinery utilisation, storage tank fill levels, pipeline construction, and port infrastructure changes. |
| **Complementarity with NTL** | The paper's demonstration that daytime imagery outperforms NTL motivates the dissertation's dual use of Sentinel-2 (daytime) and VIIRS nighttime lights for supply-side proxy construction. |

---

## Limitations

1. **Not directly applied to commodity markets**: The paper proxies general economic activity (GDP), not commodity-specific supply/demand indicators; the translation to oil market signals requires additional domain assumptions.
2. **Static land cover classification**: The proxy relies on land use/land cover (LULC) composition, which changes slowly and may not capture short-term fluctuations in industrial activity relevant to monthly/weekly oil price forecasting.
3. **Cloud cover and data gaps**: Landsat imagery is affected by cloud contamination, limiting temporal frequency in cloudy regions (e.g., tropical oil-producing countries); the paper does not fully address compositing strategies.
4. **Linear proxy assumption**: The OLS-based mapping from surface features to economic output assumes a roughly linear relationship, which may not hold for heavy-industry or extractive-economy regions.
5. **No causal mechanism**: The correlation between surface features and economic activity is demonstrated empirically but the paper does not provide a structural model explaining which physical processes drive the relationship.

---

## Notes for Dissertation Integration

- Cite as foundational evidence that satellite imagery can serve as a reliable proxy for economic activity, justifying the "Remote Sensing" pillar.
- Use the daytime-vs-NTL comparison to motivate combining Sentinel-2 vegetation indices with VIIRS nighttime lights in the dissertation's feature set.
- Highlight the temporal granularity limitation as a gap: this dissertation constructs monthly composites from Sentinel-2 to achieve higher frequency than the annual snapshots used in Lehnert et al.
- The "30-m resolution" argument directly supports using Sentinel-2's 10-m multispectral bands for finer-grained oil infrastructure monitoring.
