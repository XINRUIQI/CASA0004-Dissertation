# Chapter 2 — Literature Review

*(Draft v1, 2026-07-03. Thematic synthesis, ~5 pages. Citations carry the internal
paper ID [Pxxx] in the reference list for traceability; verify final bibliographic
details before submission.)*

This chapter reviews five bodies of work that together define the gap this
dissertation addresses. Section 2.1 establishes what is already known about
crude-oil price forecasting and, crucially, how hard the random-walk benchmark
is to beat. Sections 2.2 and 2.3 review the two "alternative-data" modalities —
maritime/AIS shipping and satellite remote sensing — as economic proxies, and
show that most existing work reduces them to hand-crafted numeric columns.
Section 2.4 turns to multimodal machine learning and the fusion architectures
that make a representation-level alternative possible. Section 2.5 states the
resulting gap and positions the contribution.

## 2.1 Crude oil price forecasting

**Econometric foundations.** The modern understanding of oil prices rests on
distinguishing the *sources* of price movements. Kilian (2009) [P052] decomposes
oil-price shocks into crude-supply, aggregate-demand, and oil-specific
(precautionary) demand components, using a global activity index constructed from
dry-bulk shipping freight rates. This decomposition is not a forecasting model,
but it provides the mechanism-based logic for variable selection adopted here:
predictors should span supply, global demand, and precautionary/uncertainty
channels rather than being chosen to "reach ten variables". The forecasting
handbook of Alquist, Kilian & Vigfusson (2013) [P053] is the methodological
anchor: it establishes that the **no-change (random-walk) forecast is a very
strong benchmark** that complex models routinely fail to beat out of sample,
that in-sample predictability does not imply out-of-sample skill, and that
evaluation must use real-time data alignment, recursive/rolling windows, and
formal accuracy tests (Diebold–Mariano). Baumeister & Kilian (2015) [P054] add
that forecast *combinations* across genuinely different economic mechanisms tend
to outperform any single model, and that equal weighting is hard to beat when
data are revised — directly motivating a modality-by-modality ablation design.

**Machine-learning approaches.** A large ML literature applies tree ensembles,
regularised linear models, and deep networks to oil prices. Costa et al. (2021)
[P072] evaluate 22 methods over 315 macro-financial series and find that
short-horizon accuracy is often best from Adaptive LASSO / Elastic Net, with
XGBoost a *strong non-linear benchmark* rather than a universal winner, and —
importantly — that the set of useful predictors changes with horizon and over
time. Yılmaz & Zehir (2026) [P076] show geopolitical-risk (GPR), VIX and interest
-rate changes carry incremental value for Brent returns, with LightGBM
significantly beating XGBoost on a Diebold–Mariano test. Foroutan & Lahmiri
(2024) [P001] compare sixteen models on price levels and find TCN and gradient
-boosting strongest, while cautioning that single-step price-level errors are
optimistic because \(P_{t+1}\approx P_t\). Hybrid designs such as LSTM-feature-
extraction + XGBoost (Simsek et al., 2024 [P004]) report near-perfect \(R^2\)
that most likely reflects pre-split normalisation leakage, underscoring why
leakage-safe protocols and random-walk baselines are non-negotiable. The
collective lesson is that oil-price forecasting is dominated by a strong
benchmark, requires mechanism-based features and rigorous out-of-sample
evaluation, and treats the price's own lag as one of the strongest predictors.

## 2.2 Shipping activity and oil markets

**AIS as a trade-flow proxy.** Automatic Identification System (AIS) vessel
tracking has become a standard high-frequency proxy for physical trade. Adland
et al. (2017) [P014] validate AIS-derived crude-export volumes against customs
statistics; Yan et al. (2020) [P015] map global marine oil trade and confirm its
concentration through a few chokepoints (Hormuz, Malacca, Suez). Arslanalp,
Marini & Tumbarello (2019) [P018] and the IMF PortWatch methodology (2026) [P070]
formalise nowcasting of trade from vessel movements, with two lessons carried
into this work: (i) **capacity-weighted (DWT) and draught-change indicators
outperform simple vessel counts**, and (ii) filtering non-trade activity and
avoiding centred moving averages (which leak the future) are prerequisites for a
valid signal.

**Oil price and shipping are causally entangled.** A key subtlety, easy to
overlook, is the *direction* of the relationship. Mi et al. (2022) [P016] and Mi,
Zang, Lo & Chen (2023) [P017] both study *oil price → tanker port-call activity*,
not the reverse, and find the relationship non-linear, regionally heterogeneous,
and — while statistically significant — carried by regressions with very low
\(R^2\) (statistical significance is not predictive value). This has two
implications for a study that instead uses shipping to predict price: shipping
features must be strictly lagged and release-time aligned, and the potential
for reverse causality must be discussed explicitly. Moreover, chokepoint transit
is only a *coarse* proxy for oil trade — the previous port is not the cargo
origin, and crude is routinely re-sold, blended or transferred at sea — so
shipping indicators should be tanker-specific, capacity-weighted, split by
chokepoint/region, and validated against official oil-flow statistics before
being trusted as predictors.

## 2.3 Satellite imagery and remote sensing

**Night-time lights and activity proxies.** Remote sensing offers a physically
grounded, high-frequency view of oil infrastructure. Night-time lights (NTL) are
the most-used proxy, but the literature is more cautious than headline claims
suggest. Polinov, Bookman & Levin (2022) [P024] show NTL correlates with
anchorage shipping activity at the *cross-sectional* scale, yet the NTL–tanker
correlation at a single port is essentially zero (Rs ≈ −0.07) — NTL is not a
tanker counter. Gibson et al. (2021) [P032] demonstrate that VIIRS is far
superior to DMSP for facility-scale work and, critically, that NTL captures
cross-sectional differences well but *within-unit temporal variation* poorly.
Both results push the design toward **within-site standardised anomalies rather
than raw radiance**, the approach adopted in this study's mechanism channel.

**Remote sensing as a direct price/demand signal.** Direct RS→oil evidence is
thin but instructive. Hao & Wang (2023) [P025] link cloud cover over US storage
regions to next-week WTI returns through an *information-availability* channel
(clouds obscure daytime optical observation of floating-roof tanks, raising
inventory uncertainty), a small but significant effect that maps onto Sentinel-2
clear-observation counts rather than night-time radiance. The ECB study by
Bricongne et al. (2026) [P069] uses tropospheric NO₂ to nowcast national oil
*demand*, but its incremental value collapses inside non-linear models and it
predicts demand, not price — so any RS→price claim requires an explicit
"activity → demand/supply → price" transmission argument and its own ablation.
Wang et al. (2019) [P055] estimate oil-tank *structural capacity* (not fill
level) from sub-metre imagery, which supports static capacity weighting but not
high-frequency inventory features at Sentinel-2 resolution. Finally, Jung (2026)
[P068] fuses SAR, NTL and port attributes to nowcast port-level trade with
XGBoost, showing remote sensing can be turned into tabular features without
training an image CNN — but also that no single fusion configuration is
universally best. Across this work, remote sensing is best framed as an
upstream/demand-side proxy whose incremental value must be established by
ablation and mechanism checks, not assumed.

## 2.4 Multimodal forecasting

**A taxonomy for fusion.** Baltrušaitis, Ahuja & Morency (2019) [P101] provide
the organising framework: multimodal learning spans representation, translation,
alignment, fusion and co-learning, where *fusion* is early (feature-level),
late (decision-level) or hybrid, and *representation* is either joint (a shared
latent space) or coordinated. In these terms, the near-universal practice in
oil-price work — concatenating NDVI, vessel counts and macro variables into one
table — is **early, feature-level fusion**; a modality-aware alternative learns a
joint representation via modality-specific encoders and model-based fusion. The
most relevant time-series precedent is the Modality-aware Transformer of Gohari
et al. (2024) [P039], which shows on financial series that a structured
modality-aware design outperforms naïve concatenation and yields interpretable
cross-modal attention — but it uses only two modalities (text + numeric), targets
equities not commodities, includes no image/spatial modality, and does not handle
missing modalities.

**Building blocks for a representation-level model.** Several components make a
modality-aware oil-price model feasible. Earth-observation foundation models —
Prithvi-EO-2.0 (Szwarcman et al., 2024) [P094] and SatMAE (Cong et al., 2022)
[P095] — are self-supervised transformers pre-trained on multispectral/temporal
satellite imagery whose frozen encoders yield transferable image embeddings,
avoiding training an image model on a few hundred weekly samples; both, however,
are validated only on land-cover/segmentation tasks, never on price. Gated
Multimodal Units (Arevalo et al., 2017) [P096] learn input-dependent gates that
weight each modality's contribution, providing both a fusion mechanism and a
built-in interpretability handle. Robustness to missing modalities — unavoidable
with cloud-limited monthly satellite data and publication-lagged shipping — is
addressed by Ma et al. (2022) [P097], who show multimodal transformers degrade
sharply under missing inputs unless missing-modality training is used, and by the
modality-dropout idea of ModDrop (Neverova et al., 2016) [P100]. Irregular,
asynchronous observation is handled by GRU-D (Che et al., 2018) [P098] via masks
and time-since-last-observation, and by mTAN (Shukla & Marlin, 2021) [P099] via
learned continuous-time embeddings — both directly relevant to aligning monthly
imagery with weekly prices. For the temporal/graph backbones, the Temporal Fusion
Transformer (Lim et al., 2021) [P089], Graph WaveNet (Wu et al., 2019) [P091] and
crude-oil maritime GNNs (LGCOTFF [P062]; GWNet-Attn [P063]) supply candidate
encoders. Crucially, **none of these has been validated on crude-oil price
forecasting**: they are methodological scaffolding whose value in this setting
must be demonstrated empirically, not asserted.

## 2.5 Research gap and positioning

**The gap.** Oil-price forecasting has a very strong random-walk benchmark
(§2.1); shipping and remote sensing are credible but noisy, causally entangled,
coarse proxies whose incremental value is unproven (§2.2–2.3); and multimodal
machine learning offers representation-level fusion tools that have not been
brought to this problem (§2.4). The decisive observation is that virtually all
existing oil-price studies using alternative data stop at **multi-source
heterogeneous feature fusion**: imagery is compressed to NDVI/NTL, shipping to
vessel counts, and everything enters one flat table fed to a numeric model, which
therefore only ever "sees" an ordinary multivariate series. Whether *preserving
each modality's structure and fusing in representation space* adds value over
flat feature fusion — in crude-oil forecasting specifically — has not been
tested under a fair, leakage-safe protocol.

**Positioning of this dissertation.** This work does **not** propose a new fusion
operator, network layer or loss. Its contribution is one of *application,
integration and systematic empirical comparison*: it integrates existing methods
(frozen EO foundation-model embeddings, modality-specific encoders, gated/cross
-attention fusion, and missing-modality/asynchronous-time handling) and, for the
first time in weekly Brent forecasting, systematically compares representation-
level modality-aware fusion against flat feature/early fusion under one
leakage-safe protocol with Diebold–Mariano and Clark–West testing. Three
questions follow: **RQ1** — do remote sensing and shipping add incremental
out-of-sample value over a financial baseline (and over the random walk)?
**RQ2** — does modality-aware representation-level fusion outperform flat feature
fusion on identical data? **RQ3** — do gating/attention weights reveal which
modality the model relies on in different market regimes? Preliminary baseline
evidence already motivates RQ2: high-dimensional shipping features yield a
significant nested increment under a tree model but not under a flat linear
model, suggesting that *how* heterogeneous modalities are combined,
not merely whether they are included, matters.

---

## References (keyed to internal IDs; verify final details before submission)

- Adland, R. et al. (2017). *AIS-based crude-oil export volume estimation.* [P014]
- Alquist, R., Kilian, L., & Vigfusson, R. J. (2013). *Forecasting the Price of Oil.* Handbook of Economic Forecasting, 2A. [P053]
- Arevalo, J., Solorio, T., Montes-y-Gómez, M., & González, F. A. (2017). *Gated Multimodal Units for Information Fusion.* ICLR Workshop. [P096]
- Arslanalp, S., Marini, M., & Tumbarello, P. (2019). *Big Data on Vessel Traffic: Nowcasting Trade Flows in Real Time.* IMF WP/19/275. [P018]
- Baltrušaitis, T., Ahuja, C., & Morency, L.-P. (2019). *Multimodal Machine Learning: A Survey and Taxonomy.* IEEE TPAMI 41(2). [P101]
- Baumeister, C., & Kilian, L. (2015). *Forecasting the Real Price of Oil in a Changing World: A Forecast Combination Approach.* [P054]
- Bricongne, J.-C., Macalos, J.-P., Meunier, B., et al. (2026). *Can Satellites Predict Oil Demand?* ECB WP 3198. [P069]
- Che, Z., Purushotham, S., Cho, K., et al. (2018). *Recurrent Neural Networks for Multivariate Time Series with Missing Values (GRU-D).* Scientific Reports 8:6085. [P098]
- Cong, Y., Khanna, S., Meng, C., et al. (2022). *SatMAE: Pre-training Transformers for Temporal and Multi-Spectral Satellite Imagery.* NeurIPS. [P095]
- Costa, A. B. et al. (2021). *Machine Learning and Oil Price Point and Density Forecasting.* [P072]
- Foroutan, P., & Lahmiri, S. (2024). *Deep learning systems for forecasting the prices of crude oil and precious metals.* Financial Innovation 10:111. [P001]
- Gibson, J., Olivia, S., Boe-Gibson, G., & Li, C. (2021). *Which Night Lights Data Should We Use in Economics, and Where?* J. Development Economics 149. [P032]
- Gohari, H. E., Dang, X.-H., Shah, S. Y., & Zerfos, P. (2024). *Modality-aware Transformer for Financial Time Series Forecasting.* ICAIF '24. [P039]
- Hao, J., & Wang, Y. (2023). *Cloud Cover and Expected Oil Returns.* Humanities and Social Sciences Communications 10:605. [P025]
- Jung (2026). *Watching Trade from Space: Nowcasting Port-Level Maritime Trade.* [P068]
- Kilian, L. (2009). *Not All Oil Price Shocks Are Alike.* American Economic Review. [P052]
- Lim, B. et al. (2021). *Temporal Fusion Transformers for Interpretable Multi-horizon Time Series Forecasting.* [P089]
- Ma, M., Ren, J., Zhao, L., Testuggine, D., & Peng, X. (2022). *Are Multimodal Transformers Robust to Missing Modality?* CVPR. [P097]
- Mi, J. et al. (2022). *The Impact of the Crude Oil Price on Tankers' Port-Call Features.* JMSE 10(10). [P016]
- Mi, J., Zang, ..., Lo, ..., & Chen, ... (2023). *The Nonlinear Relationship between Oil Prices and Tankers' Port Calls.* Procedia CS 221. [P017]
- Neverova, N., Wolf, C., Taylor, G. W., & Nebout, F. (2016). *ModDrop: Adaptive Multi-Modal Gesture Recognition.* IEEE TPAMI. [P100]
- Polinov, S., Bookman, R., & Levin, N. (2022). *A Global Assessment of Night Lights as an Indicator for Shipping Activity in Anchorage Areas.* Remote Sensing 14(5). [P024]
- IMF (2026). *Nowcasting Country-Level Trade Using IMF PortWatch.* [P070]
- Shukla, S. N., & Marlin, B. M. (2021). *Multi-Time Attention Networks for Irregularly Sampled Time Series (mTAN).* ICLR. [P099]
- Simsek, ..., Bulut, ..., Gur, ..., & Gültekin Tarla, ... (2024). *LSTM-based feature extraction with XGBoost Regressor for WTI.* Energy 309. [P004]
- Szwarcman, D., Roy, S., Fraccaro, P., et al. (2024). *Prithvi-EO-2.0: A Versatile Multi-Temporal Foundation Model for Earth Observation.* arXiv:2412.02732. [P094]
- Wang, T., Li, ..., Yu, ..., & Liu, ... (2019). *Estimating the Volume of Oil Tanks Based on High-Resolution Remote Sensing Images.* Remote Sensing 11(7). [P055]
- Wu, Z., Pan, S., Long, G., Jiang, J., & Zhang, C. (2019). *Graph WaveNet for Deep Spatial-Temporal Graph Modeling.* IJCAI. [P091]
- Yan, Z. et al. (2020). *Analysis of global marine oil trade from AIS.* [P015]
- Yılmaz, ..., & Zehir, ... (2026). *Strategic-Risk-Based Forecasting of Brent Crude Oil Prices.* Entropy 28(5). [P076]
