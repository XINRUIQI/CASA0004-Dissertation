# Reading Note — Paper 04: AIS/Shipping Data and Oil Trade Estimation

## Citation

Li, Y., Bai, X., Wang, Q., & Ma, Z. (2022). A big data approach to cargo type prediction and its implications for oil trade estimation. *Transportation Research Part E: Logistics and Transportation Review*, 165, 102831.

- **DOI**: [10.1016/j.tre.2022.102831](https://doi.org/10.1016/j.tre.2022.102831)
- **Published**: September 2022

---

## Core Method

- **Data source**: Automatic Identification System (AIS) data covering 752 coated product tankers over 2017–2020, providing real-time vessel position, speed, draft, and voyage information.
- **Task**: Two-stage cargo type classification:
  1. **Stage 1**: Distinguish Clean Petroleum Products (CPP) trips from Dirty Petroleum Products (DPP) trips.
  2. **Stage 2**: For DPP trips, further classify between crude oil and dirty refined product oil cargo.
- **Model**: Random Forest (RF) ensemble learning, compared against Logistic Regression, SVM, k-NN, Decision Tree, and Neural Networks.
- **Features**: Domain-knowledge-driven feature engineering based on consultation with industry experts, including voyage characteristics (distance, duration, draft change), port-level features (loading/discharge terminal type), and vessel specifications.
- **Application**: The classified cargo types are aggregated to estimate global crude oil trade volumes carried by coated product tankers, revealing that approximately 8% of crude trade occurs via these "hidden" tanker segments.

---

## Key Findings

1. **Random Forest achieves best classification performance** across multiple evaluation metrics (accuracy, precision, recall, F1-score), outperforming all alternative classifiers for both stages of cargo type prediction.
2. **~8% of crude oil trade is carried by coated product tankers**, a segment invisible to traditional trade statistics and conventional tanker tracking that only monitors VLCCs and Suezmax vessels.
3. **Draft change is a critical feature**: The difference between loaded and ballast draft serves as a strong signal for distinguishing cargo types, as crude oil is denser than clean petroleum products.
4. **Temporal patterns are consistent with market events**: Estimated crude oil trade volumes from 2017–2020 align with known supply disruptions (US sanctions on Iran/Venezuela, OPEC+ cuts, COVID-19 demand shock), validating the AIS-derived estimates.
5. **High-frequency trade monitoring**: AIS data enables near-real-time estimation of oil trade flows at a granularity far exceeding official statistics (which are published with months of delay).

---

## Relevance to This Dissertation

| Aspect | Connection |
|--------|------------|
| **AIS as oil market data** | Directly demonstrates that AIS vessel tracking data can be transformed into meaningful oil supply/demand signals — the core premise of the dissertation's "Maritime Shipping" data pillar. |
| **Feature engineering template** | The domain-knowledge-driven feature construction (draft change, voyage duration, port type) provides a blueprint for engineering AIS-based features for the dissertation's tanker activity indicators. |
| **Supply-side proxy** | Estimated crude oil trade volumes serve as a near-real-time supply-side proxy, complementing the dissertation's other supply indicators (OPEC production, inventories, rig counts). |
| **Random Forest benchmark** | The paper's RF classification pipeline can be adapted as a preprocessing step in the dissertation to classify tanker cargoes before aggregating shipping features for the price forecasting model. |
| **Hidden trade detection** | The finding that 8% of crude trade is "invisible" to conventional statistics motivates incorporating AIS data to capture market signals that traditional fundamental data miss. |

---

## Limitations

1. **Classification, not forecasting**: The paper classifies cargo types ex-post but does not use the classified data for price forecasting; the causal link from AIS-derived trade volumes to price movements is not established.
2. **Limited to coated product tankers**: Only one tanker segment is studied; the methodology needs extension to VLCCs, Suezmaxes, and Aframaxes, which carry the bulk of crude oil.
3. **Expert-dependent feature engineering**: Feature construction relies heavily on industry domain expertise, which may not be fully transferable or reproducible without similar expert consultation.
4. **No spatial analysis**: The paper does not exploit the geographic/spatial dimension of AIS data (e.g., chokepoint analysis at Strait of Hormuz, Suez Canal) that could provide additional predictive signals.
5. **Short time span (2017–2020)**: The 4-year window limits the ability to assess model robustness across different oil market cycles; longer historical AIS data would strengthen validation.

---

## Notes for Dissertation Integration

- Cite as the primary reference for AIS-based oil trade estimation, justifying the inclusion of maritime shipping data in the multimodal framework.
- Adapt the two-stage classification approach as a data preprocessing pipeline: first classify tanker cargoes, then aggregate voyage-level features into monthly shipping activity indicators.
- Use the "hidden 8% trade" finding as motivation for why traditional oil market fundamentals data are incomplete and AIS-derived features add informational value.
- Extend the methodology by adding spatial features (chokepoint transit counts, port-level congestion indices) not explored in this paper.
