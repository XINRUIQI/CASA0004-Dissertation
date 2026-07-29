# Chapter 4 — Results *(~2,500–3,200)*

## 4.1 Descriptive overview

This chapter reports out-of-sample one-week-ahead Brent price forecasts on the common scored span in Section 3.9 (257 weeks, 22 January 2021–19 December 2025). Relative performance versus the no-change benchmark M0 is summarised by RMSE skill (Murphy, 1988), defined in Section 3.10 as

\[
\mathrm{Skill}=100\times\left(1-\frac{\mathrm{RMSE}_{\mathrm{model}}}{\mathrm{RMSE}_{\mathrm{M0}}}\right).
\]

Positive skill means lower RMSE than M0; negative skill means worse. On this span M0 RMSE is 4.152 USD/barrel. Weekly Brent log returns have near-zero mean and clear volatility clustering. Exploratory remote-sensing anomalies show only weak contemporaneous association with returns, and several stronger associations occur at non-positive leads. Shipping is treated as a noisy trade-and-congestion proxy, not as a direct measure of next week’s price. The chapter follows Flat results (RQ1), Deep results, paired Flat–Deep comparisons (RQ2), robustness, and interpretability where predictive value exists (RQ3).

## 4.2 Flat-model results

Table 4.1 summarises Flat performance. Every learned Flat model has negative skill versus M0. Relative to financial time series M1, adding shipping in M3 still improves performance, while remote sensing in M2 does not show a clear gain over M1; the full Flat set M4 raises absolute RMSE further.

**Table 4.1 — Flat out-of-sample performance** *(n = 257)*

| Set | Content                           | Ridge RMSE | Ridge skill vs M0 | XGB RMSE | XGB skill vs M0 |
| --- | --------------------------------- | ---------- | ----------------- | -------- | --------------- |
| M0  | no-change benchmark               | 4.152      | —                 | 4.152    | —               |
| M1  | financial time series only        | 4.256      | −2.5%             | 4.368    | −5.2%           |
| M2  | financial time series + RS        | 4.414      | −6.3%             | 4.440    | −6.9%           |
| M3  | financial time series + shipping  | 4.430      | −6.7%             | 4.429    | −6.7%           |
| M4  | financial time series + RS + ship | 4.525      | −9.0%             | 4.507    | −8.6%           |

Nested Clark–West tests versus M1 can detect incremental information from added modalities under selected learners, especially when shipping enters. Diebold–Mariano tests against M0 remain consistent with negative skill: a nested gain over financial time series is not evidence of beating M0. Directional accuracy is auxiliary and does not reverse the RMSE ranking.

## 4.3 Deep-model results

Table 4.2 summarises Deep performance by information set. For M1 only the finance encoder applies; its result is placed in the gated column for comparability. M1 and M2 both fail to beat M0. Financial time series plus shipping (M3) improves on M0 under gated and cross-attention fusion, with modest skill of about +0.11% (gated) and +0.74% (cross-attention). Nested Deep contrasts likewise identify shipping as the clearest modality gain over Deep M1. Gated M4 does not clearly dominate M3: adding remote sensing on top often fails to cut absolute error further.

Gated fusion is the main reported Deep design; cross-attention is a comparative architecture with a higher single-seed ceiling but greater sensitivity (Section 4.5). Encoder-concatenation and the full fusion matrix are in the appendix.

**Table 4.2 — Deep out-of-sample performance** *(gated = main reported fusion)*

| Set | Content                           | Gated RMSE | Gated skill vs M0 | Xattn RMSE | Xattn skill vs M0 |
| --- | --------------------------------- | ---------- | ----------------- | ---------- | ----------------- |
| M0  | no-change benchmark               | 4.152      | —                 | 4.152      | —                 |
| M1  | financial time series only        | 4.250      | −2.4%             | —          | —                 |
| M2  | financial time series + RS        | 4.253      | −2.4%             | —          | —                 |
| M3  | financial time series + shipping  | 4.147      | **+0.11%**        | 4.121      | **+0.74%**        |
| M4  | financial time series + RS + ship | 4.205      | −1.3%             | 4.147      | +0.12%            |

## 4.4 Flat versus Deep

Table 4.3 compares Flat and Deep at matched information sets for RQ2. The percentage columns are each model’s skill versus M0, not the Flat-to-Deep RMSE change.

Deep gains are clearest once shipping enters. Deep M1 improves on Flat M1 in RMSE but remains negative versus M0. Finance-plus-RS pairs stay weak in both families. The clearest paired gain is M3: Flat skill −6.7% versus Deep gated +0.11%. Deep M4 has lower RMSE than Flat M4 but neither gated Deep M4 nor the Flat counterpart beats M0, and gated Deep M4 does not dominate Deep M3. The Deep advantage is therefore conditional on shipping-inclusive settings rather than uniform across all pairs.

**Table 4.3 — Paired Flat versus Deep**

| Pair | Flat RMSE | Deep RMSE | Flat skill vs M0 | Deep skill vs M0 |
| ---- | --------- | --------- | ---------------- | ---------------- |
| M1   | 4.368     | 4.250     | −5.2%            | −2.4%            |
| M2   | 4.440     | 4.253     | −6.9%            | −2.4%            |
| M3   | 4.429     | 4.147     | −6.7%            | +0.11%           |
| M4   | 4.507     | 4.205     | −8.6%            | −1.3%            |

## 4.5 Robustness and sensitivity

Appendix B collects the full robustness tables. Flat checks (lookback, remote-sensing variants, shipping tiers, leave-one-modality-out for M4) leave the main ranking unchanged: no Flat model beats M0; shipping still helps relative to M1; remote sensing remains weak and any nested Flat RS signal is diffuse rather than single-site driven.

Deep checks cover seeds, lookback, representation size, fusion type, and early versus late windows. Gated finance-plus-shipping remains the more stable small positive-skill configuration; cross-attention can look stronger on one seed but varies more across seeds. Larger encoder width than the locked setting tends to worsen performance on the short weekly sample. The matched-set Deep advantage over Flat, especially with shipping, survives these checks.

## 4.6 Interpretability

Interpretability is restricted to Deep specifications that improve on M0—primarily Deep M3. Claims follow a multi-seed rule (seeds 42, 1 and 2): only cross-seed-stable foci are locked in the main text. Modality gates give each modality’s fusion-weight share; shipping node attention identifies which graph locations receive weight. A high shipping gate does not mean the model is “looking at Hormuz”; spatial detail lives in node attention.

For Deep M3, mean gates are about 0.56 (financial time series) and 0.44 (shipping). Week-level shipping-gate paths are unstable across seeds, so fine-grained single-seed event stories are not warranted. Event-window checks (±8 weeks) retain only the Russia–Ukraine announcement window (February 2022) as a cross-seed co-rising case. EU oil-ban and OPEC+ cut windows co-move but shipping weight falls; the Houthi Red Sea window (November 2023) is unstable (2↑1↓) and is not locked. Spatially, Hormuz is the only chokepoint in the top set for all three seeds (3/3). Supporting figures are in Appendix B.

These diagnostics describe model dependence under a stability filter. They do not identify causal drivers of Brent prices.
