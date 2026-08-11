# Chapter 4 — Results（～**1,200** ）

## 4.1 Descriptive overview

This chapter reports out-of-sample one-week-ahead Brent forecasts on the common evaluation sample of 257 weeks (22 January 2021–19 December 2025). Performance is summarised by RMSE on reconstructed prices and by RMSE skill versus the no-change benchmark M0 (Murphy, 1988). Skill is positive when RMSE is lower than M0 and negative when it is higher. On this sample the M0 RMSE is 4.152 USD per barrel.

Weekly Brent log returns have near-zero mean and clear volatility clustering. Exploratory checks show only weak contemporaneous association between remote-sensing anomalies and returns. Shipping enters as a noisy proxy for trade and congestion, not as a direct measure of next week’s price.

## 4.2 Flat-model results

Table 4.1 reports Flat out-of-sample performance for Ridge and XGBoost across M0–M4. Every learned Flat specification has negative skill versus M0, so the no-change forecast remains the best absolute-error benchmark in the Flat family.

**Table 4.1 — Flat out-of-sample performance** *(n = 257)*


| Set | Content                           | Ridge RMSE | Ridge skill vs M0 | XGB RMSE | XGB skill vs M0 |
| --- | --------------------------------- | ---------- | ----------------- | -------- | --------------- |
| M0  | no-change benchmark               | 4.152      | —                 | 4.152    | —               |
| M1  | financial time series only        | 4.256      | −2.5%             | 4.368    | −5.2%           |
| M2  | financial time series + RS        | 4.414      | −6.3%             | 4.440    | −6.9%           |
| M3  | financial time series + shipping  | 4.430      | −6.7%             | 4.429    | −6.7%           |
| M4  | financial time series + RS + ship | 4.525      | −9.0%             | 4.507    | −8.6%           |


Finance-only M1 records the lowest Flat RMSE among learned sets (Ridge 4.256, −2.5%; XGBoost 4.368, −5.2%). Adding remote sensing (M2) or shipping (M3) raises RMSE relative to M1 under both learners. The full Flat set M4 is weakest (Ridge 4.525, −9.0%; XGBoost 4.507, −8.6%). Ridge and XGBoost agree: M1 is best among Flat learners, M4 is worst, and neither remote sensing nor shipping reduces absolute RMSE below the finance-only Flat baseline.

Under early feature fusion, noisy alternative-data proxies do not improve one-week-ahead Brent RMSE relative to M0 or to finance alone. For RQ1, Flat results therefore show no absolute out-of-sample gain from remote sensing or shipping.

## 4.3 Deep-model results

Table 4.2 reports Deep performance by information set. Gated fusion is the main Deep specification; cross-attention is a comparison where multimodal fusion applies. For M1 only the finance encoder is active. M1 and M2 both fail to beat M0 (gated RMSE 4.250 and 4.253; both −2.4% skill). Absolute error barely moves when remote sensing enters.

**Table 4.2 — Deep out-of-sample performance** *(gated = main specification)*


| Set | Content                           | Gated RMSE | Gated skill vs M0 | Cross-attn RMSE | Cross-attn skill vs M0 |
| --- | --------------------------------- | ---------- | ----------------- | --------------- | ---------------------- |
| M0  | no-change benchmark               | 4.152      | —                 | 4.152           | —                      |
| M1  | financial time series only        | 4.250      | −2.4%             | —               | —                      |
| M2  | financial time series + RS        | 4.253      | −2.4%             | —               | —                      |
| M3  | financial time series + shipping  | 4.147      | +0.11%            | 4.121           | +0.74%                 |
| M4  | financial time series + RS + ship | 4.205      | −1.3%             | 4.147           | +0.12%                 |


Once shipping is included, gated M3 reduces RMSE to 4.147 (+0.11% skill). Cross-attention on the same set reaches 4.121 (+0.74%) on this reported seed. Shipping is the modality that moves Deep forecasts across the M0 line relative to Deep M1. Gated M4 rises again to 4.205 (−1.3%); cross-attention M4 is near M0 at +0.12% but does not displace gated M3 as the main finding. The gated margin is small and should not be over-read on a short weekly sample; Section 4.5 returns to seed sensitivity.

For RQ1 under Deep, shipping-inclusive forecasts clear M0 by a modest margin, while remote sensing does not add a comparable absolute-error gain.

## 4.4 Flat versus Deep

**Table 4.3 — Paired Flat versus Deep**  
*(Flat = Table 4.1 XGBoost; Deep = Table 4.2 gated; percentages are skill versus M0)*


| Pair | Flat RMSE | Deep RMSE | Flat skill vs M0 | Deep skill vs M0 |
| ---- | --------- | --------- | ---------------- | ---------------- |
| M1   | 4.368     | 4.250     | −5.2%            | −2.4%            |
| M2   | 4.440     | 4.253     | −6.9%            | −2.4%            |
| M3   | 4.429     | 4.147     | −6.7%            | +0.11%           |
| M4   | 4.507     | 4.205     | −8.6%            | −1.3%            |


Deep has lower RMSE than Flat in every matched pair. Finance-only and finance-plus-RS pairs improve on Flat but remain negative versus M0. The decisive pair is M3: Flat skill −6.7% versus gated Deep +0.11%—the only matched pair in which Deep also beats M0. Deep M4 improves on Flat M4 but stays negative versus M0 and does not improve on Deep M3.

For RQ2, representation-level Deep modelling reduces RMSE relative to Flat at every matched set, but an M0-beating paired outcome appears only when shipping is included.

## 4.5 Robustness and sensitivity

Appendix B collects the detailed robustness tables. Flat checks that vary lookback and feature settings produce no Flat specification that beats M0. Finance-only M1 remains the strongest Flat absolute-error baseline; remote sensing stays weak and is not driven by a single site. Nested Clark–West tests versus M1 in Appendix B detect incremental information over the financial baseline for some XGBoost shipping specifications, even when absolute RMSE remains higher than M1 and skill versus M0 remains negative. Shipping can therefore show a nested Flat signal without overturning Table 4.1’s absolute-error ranking.

Deep checks that vary random seeds and fusion choices leave gated finance-plus-shipping as the more stable small positive-skill configuration. Cross-attention can exceed gated fusion on one seed, as in Table 4.2 for M3, but varies more across seeds. Larger encoder width than the main setting tends to worsen performance on the short weekly sample. Sub-period splits leave gated M3 positive in both early and late windows. The matched Deep advantage over Flat, especially with shipping, survives these checks.

These checks leave the RQ1–RQ2 rankings unchanged: Flat absolute gains remain absent; Deep’s small shipping-centred M0 clearance is the more stable positive case.

## 4.6 Interpretability

Interpretability is restricted to Deep specifications that improve on M0, principally Deep M3, using seeds 42, 1 and 2. Reported patterns are those that agree across seeds. Modality gates give each modality’s fusion-weight share; shipping node attention identifies which graph locations receive weight. A high shipping gate does not by itself mean the model focuses on a particular chokepoint; spatial detail is read from node attention.

For Deep M3, mean gates are about 0.56 (financial time series) and 0.44 (shipping). Week-level shipping-gate paths are unstable across seeds, so single-seed event stories are not warranted. Among pre-specified event windows (±8 weeks), only the Russia–Ukraine announcement window (February 2022) shows a shipping-gate rise across all three seeds. The Red Sea window (November 2023) rises in two seeds and falls in one, and is not retained. Spatially, the Strait of Hormuz is the only chokepoint in the top attention set for all three seeds. Figure 4.1 summarises the main Deep M3 gate and attention diagnostics; further panels are in Appendix B.

Figure 4.1 — Deep M3 modality gates and shipping-node attention (multi-seed summary).

*[Figure 4.1 — Deep M3 interpretability: modality gates and shipping-node attention.]*

For RQ3, when Deep shipping-inclusive forecasts clear M0, the stable main-text reliance pattern is shared weight on finance and shipping, with Hormuz as the only cross-seed spatial focus. These diagnostics describe model dependence after a stability filter; they do not identify causal drivers of Brent prices.