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
leakage-safe protocols and random-walk baselines are non-negotiable. Graph-based
deep learning has also reached the problem: Zhao, Xue & Cheng (2023) [P063] couple
a self-attention-learned dynamic graph with Graph WaveNet to forecast multi-step
WTI futures and report gains over other neural models — but their "spatial" graph
encodes *non-Euclidean relations among predictors*, not a geographic or shipping
network, and they omit the no-change benchmark for a highly persistent price
level, so superiority over a random walk is not established. A caveat cutting
across this literature is that most evidence concerns WTI at daily or monthly
frequency, and often on price levels; weekly Brent — the target here — yields
fewer observations per year and an even more dominant random walk, so published
gains do not transfer automatically. The collective lesson is that oil-price
forecasting is dominated by a strong benchmark, requires mechanism-based features
and rigorous out-of-sample evaluation, and treats the price's own lag as one of
the strongest predictors.

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

**From flat indicators to shipping networks.** Beyond scalar counts, maritime
activity has an intrinsic *network* structure that a flat table discards. The IMF
PortWatch methodology (2026) [P070] standardises chokepoint- and port-level
transit series and is the operational source adopted here, while Paolo et al.
(2024) [P057] show from global SAR that a substantial share of at-sea industrial
activity is absent from AIS altogether — a reminder that shipping proxies are
themselves incomplete. To exploit structure rather than discard it, crude-oil
work increasingly represents ports, terminals and chokepoints as graph nodes:
Ouyang et al. (2022) [P062] couple supply-chain graph convolution with an LSTM to
forecast crude-tanker traffic flow, and Liang et al. (2022) [P066] use a
spatio-temporal multi-graph network (distance, interaction and correlation graphs)
for fine-grained vessel-flow prediction. Both predict *traffic*, not price, so
they establish only that a maritime graph is learnable — motivating, but not
proving, a shipping encoder for Brent.

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
[P095] — are self-supervised transformers whose frozen encoders yield transferable
image embeddings, avoiding training an image model on a few hundred weekly samples.
Because optical and radar sensors differ in channel structure and noise, multimodal
EO models encode each modality with a dedicated *modality-specific encoder* before
fusing them: CROMA (Fuller et al., 2023) [P105] is representative, learning separate
radar and optical encoders coupled by contrastive and masked-autoencoding
objectives, and a growing family (DOFA [P106], OmniSat [P107], TerraFM [P108])
extends the idea to more sensors, while cross-attention fusion [P111; P112] and
shared/modality-specific decompositions [P109] consistently outperform naïve
concatenation on segmentation benchmarks. A recent survey of these multimodal EO
foundation models [P115] confirms the trend but also flags their open problems —
limited cross-modal transfer and no standard evaluation — and, like every model
above, they are validated on land-cover tasks, never on price. Gated Multimodal
Units (Arevalo et al., 2017) [P096] learn input-dependent gates that weight each
modality's contribution, providing both a fusion mechanism and a built-in
interpretability handle. Robustness to missing modalities — unavoidable
with cloud-limited monthly satellite data and publication-lagged shipping — is
among the EO open problems noted above and one of the most consequential here. In
general multimodal learning it is addressed by Ma et al. (2022) [P097],
who show multimodal transformers degrade sharply under missing inputs unless
missing-modality training is used, by the modality-dropout idea of ModDrop
(Neverova et al., 2016) [P100], and by the shared-specific feature modelling of
ShaSpec (Wang et al., 2024) [P114], which regenerates a missing modality's
embedding from the available ones rather than its raw signal. Directly in the EO
setting, RobSense (Do et al., 2025) [P113] adds uni-modal latent reconstructors
that recover multispectral/SAR representations from incomplete inputs and is
evaluated explicitly across increasing missing rates, and PyViT-FUSE (Weber &
Beneke, 2025) [P110] uses band-drop training with learnable empty tokens to
remain robust when whole sensors or bands are absent. Irregular,
asynchronous observation is handled by GRU-D (Che et al., 2018) [P098] via masks
and time-since-last-observation, and by mTAN (Shukla & Marlin, 2021) [P099] via
learned continuous-time embeddings — both directly relevant to aligning monthly
imagery with weekly prices. For the temporal/graph backbones, the Temporal Fusion
Transformer (Lim et al., 2021) [P089] and Graph WaveNet (Wu et al., 2019) [P091]
supply candidate temporal and spatio-temporal encoders, the latter also underpinning
the crude-oil maritime graphs discussed in §2.2 [P062; P066]. Crucially, **none of
these has been validated on crude-oil price
forecasting**: they are methodological scaffolding whose value in this setting
must be demonstrated empirically, not asserted.

## 2.5 Forecast evaluation and interpretability

**Comparing forecasts, not just error numbers.** Because the random walk is so
strong (§2.1), a lower RMSE or MAE is not by itself evidence that a model is
genuinely better — the gap may be sampling noise over one out-of-sample path.
Diebold & Mariano (1995) [P058] provide the standard test of equal expected
predictive accuracy under general loss, and the forecasting handbook [P053]
operationalises the surrounding protocol: real-time data alignment,
recursive/rolling evaluation, horizon-by-horizon comparison, and correction for
the serial correlation induced by overlapping multi-step forecasts. A subtlety
specific to this study is that M2–M4 are *nested* extensions of the financial
baseline M1, and under quadratic loss the standard DM asymptotics can be distorted
for nested models; Clark & West (2007) therefore supply the appropriate test of
whether an added modality delivers a statistically significant, correctly signed
reduction in out-of-sample loss. Critically, a significant DM/CW result requires
*both* a favourable sign of the mean loss differential *and* a small p-value:
significance alone does not establish superiority.

**Explaining which modality matters.** Establishing that a forecast improves does
not reveal *what* drives the improvement. SHAP (Lundberg & Lee, 2017) [P059] offers
a model-agnostic, additive attribution that can be aggregated to the modality level
(financial / remote-sensing / shipping), turning an otherwise opaque tree or
network into a mechanism-level account of which signals it relies on and when. The
two instruments are complementary and non-substitutable — DM/CW say *whether*
alternative data help, SHAP says *through which* features — and both must be read
against leakage-safe, in-window feature selection rather than computed post hoc on
the full sample. SHAP is, moreover, an attribution of model behaviour, not a causal
claim. This two-layer evidence chain (significance test + attribution) is adopted
throughout the dissertation to keep every modality claim falsifiable.

## 2.6 Research gap and positioning

**Existing alternative-data oil forecasting.** A handful of studies do connect
alternative data to oil, but each stops short of the design tested here. On the
remote-sensing side, Hao & Wang (2023) [P025] link cloud-cover observability to
weekly WTI returns, and Bricongne et al. (2026) [P069] nowcast national oil
*demand* from tropospheric NO₂ — both feed a single engineered signal into an
otherwise standard model, and the latter predicts demand rather than price. On the
shipping side, PortWatch-style pipelines [P070] and Jung (2026) [P068] convert
vessel movements and imagery into tabular indicators that nowcast *trade*, not
price. The methodologically richest precedents remain unimodal or bimodal:
GWNet-Attn [P063] adds a learned variable graph but only for WTI futures, with no
image or shipping modality, and the Modality-aware Transformer [P039] fuses text
and numeric series for interest rates, not commodity prices. Across all of them,
alternative data are either compressed into flat columns or used to nowcast an
intermediate quantity; none preserves three heterogeneous modalities and fuses
them at the representation level for weekly Brent.

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
modality the model relies on in different market regimes? These questions are
motivated, not pre-empted, by the baseline layer: its detailed M0–M4 results
(Chapter 4) already indicate that *how* modalities are combined — not merely
whether they are included — is what matters for RQ2.

---

## References (keyed to internal IDs; verify final details before submission)

- Adland, R. et al. (2017). *AIS-based crude-oil export volume estimation.* [P014]
- Alquist, R., Kilian, L., & Vigfusson, R. J. (2013). *Forecasting the Price of Oil.* Handbook of Economic Forecasting, 2A. [P053]
- Arevalo, J., Solorio, T., Montes-y-Gómez, M., & González, F. A. (2017). *Gated Multimodal Units for Information Fusion.* ICLR Workshop. [P096]
- Arslanalp, S., Marini, M., & Tumbarello, P. (2019). *Big Data on Vessel Traffic: Nowcasting Trade Flows in Real Time.* IMF WP/19/275. [P018]
- Astruc, G., Gonthier, N., Mallet, C., & Landrieu, L. (2024). *OmniSat: Self-Supervised Modality Fusion for Earth Observation.* arXiv:2404.08351. [P107]
- Baltrušaitis, T., Ahuja, C., & Morency, L.-P. (2019). *Multimodal Machine Learning: A Survey and Taxonomy.* IEEE TPAMI 41(2). [P101]
- Baumeister, C., & Kilian, L. (2015). *Forecasting the Real Price of Oil in a Changing World: A Forecast Combination Approach.* [P054]
- Bricongne, J.-C., Macalos, J.-P., Meunier, B., et al. (2026). *Can Satellites Predict Oil Demand?* ECB WP 3198. [P069]
- Che, Z., Purushotham, S., Cho, K., et al. (2018). *Recurrent Neural Networks for Multivariate Time Series with Missing Values (GRU-D).* Scientific Reports 8:6085. [P098]
- Clark, T. E., & West, K. D. (2007). *Approximately Normal Tests for Equal Predictive Accuracy in Nested Models.* Journal of Econometrics 138(1), 291–311.
- Cong, Y., Khanna, S., Meng, C., et al. (2022). *SatMAE: Pre-training Transformers for Temporal and Multi-Spectral Satellite Imagery.* NeurIPS. [P095]
- Costa, A. B. et al. (2021). *Machine Learning and Oil Price Point and Density Forecasting.* [P072]
- Danish, M. S., Munir, M. A., Shah, S. R. A., Khan, M. H., Anwer, R. M., Laaksonen, J., Khan, F. S., & Khan, S. (2025). *TerraFM: A Scalable Foundation Model for Unified Multisensor Earth Observation.* arXiv:2506.06281. [P108]
- Diebold, F. X., & Mariano, R. S. (1995). *Comparing Predictive Accuracy.* Journal of Business & Economic Statistics 13(3), 253–263. [P058]
- Do, M. K., Han, K., Lai, P., Phan, K. T., & Xiang, W. (2025). *RobSense: A Robust Multi-modal Foundation Model for Remote Sensing with Static, Temporal, and Incomplete Data Adaptability.* CVPR. [P113]
- Foroutan, P., & Lahmiri, S. (2024). *Deep learning systems for forecasting the prices of crude oil and precious metals.* Financial Innovation 10:111. [P001]
- Fuller, A., Millard, K., & Green, J. (2023). *CROMA: Remote Sensing Representations with Contrastive Radar-Optical Masked Autoencoders.* NeurIPS. [P105]
- Gibson, J., Olivia, S., Boe-Gibson, G., & Li, C. (2021). *Which Night Lights Data Should We Use in Economics, and Where?* J. Development Economics 149. [P032]
- Gohari, H. E., Dang, X.-H., Shah, S. Y., & Zerfos, P. (2024). *Modality-aware Transformer for Financial Time Series Forecasting.* ICAIF '24. [P039]
- Guo, H., Tian, B., & Liu, W. (2025). *CCFormer: Cross-Modal Cross-Attention Transformer for Classification of Hyperspectral and LiDAR Data.* Sensors 25(18):5698. [P111]
- Hao, J., & Wang, Y. (2023). *Cloud Cover and Expected Oil Returns.* Humanities and Social Sciences Communications 10:605. [P025]
- Hong, D., Hu, J., Yao, J., Chanussot, J., & Zhu, X. X. (2021). *Multimodal Remote Sensing Benchmark Datasets for Land Cover Classification with a Shared and Specific Feature Learning Model (S2FL).* ISPRS Journal of Photogrammetry and Remote Sensing. [P109]
- Jung, Y. (2026). *Watching Trade from Space: Nowcasting and Spatial Extrapolation of Port-Level Maritime Trade Using Satellite Imagery.* arXiv:2604.15444. [P068]
- Kilian, L. (2009). *Not All Oil Price Shocks Are Alike.* American Economic Review. [P052]
- Liang et al. (2022). *Fine-Grained Vessel Traffic Flow Prediction with a Spatio-Temporal Multi-Graph Convolutional Network (STMGCN).* [P066]
- Lim, B. et al. (2021). *Temporal Fusion Transformers for Interpretable Multi-horizon Time Series Forecasting.* [P089]
- Lundberg, S. M., & Lee, S.-I. (2017). *A Unified Approach to Interpreting Model Predictions (SHAP).* NeurIPS. [P059]
- Ma, M., Ren, J., Zhao, L., Testuggine, D., & Peng, X. (2022). *Are Multimodal Transformers Robust to Missing Modality?* CVPR. [P097]
- Mi, J. et al. (2022). *The Impact of the Crude Oil Price on Tankers' Port-Call Features.* JMSE 10(10). [P016]
- Mi, J. et al. (2023). *The Nonlinear Relationship between Oil Prices and Tankers' Port Calls.* Procedia CS 221. [P017]
- Neverova, N., Wolf, C., Taylor, G. W., & Nebout, F. (2016). *ModDrop: Adaptive Multi-Modal Gesture Recognition.* IEEE TPAMI. [P100]
- Ouyang et al. (2022). *Long Short-Term Memory and Graph Convolution Network for Forecasting the Crude Oil Traffic Flow (LGCOTFF).* IEEE Access. [P062]
- Paolo, F. S. et al. (2024). *Satellite Mapping Reveals Extensive Industrial Activity at Sea.* Nature 625. [P057]
- Polinov, S., Bookman, R., & Levin, N. (2022). *A Global Assessment of Night Lights as an Indicator for Shipping Activity in Anchorage Areas.* Remote Sensing 14(5). [P024]
- IMF (2026). *Nowcasting Country-Level Trade Using IMF PortWatch.* [P070]
- Shukla, S. N., & Marlin, B. M. (2021). *Multi-Time Attention Networks for Irregularly Sampled Time Series (mTAN).* ICLR. [P099]
- Simsek et al. (2024). *LSTM-based feature extraction with XGBoost Regressor for WTI.* Energy 309. [P004]
- Szwarcman, D., Roy, S., Fraccaro, P., et al. (2024). *Prithvi-EO-2.0: A Versatile Multi-Temporal Foundation Model for Earth Observation.* arXiv:2412.02732. [P094]
- Wang, H., Chen, Y., Ma, C., Avery, J., Hull, L., & Carneiro, G. (2024). *Multi-modal Learning with Missing Modality via Shared-Specific Feature Modelling (ShaSpec).* arXiv:2307.14126. [P114]
- Wang, T. et al. (2019). *Estimating the Volume of Oil Tanks Based on High-Resolution Remote Sensing Images.* Remote Sensing 11(7). [P055]
- Weber, M., & Beneke, C. (2025). *PyViT-FUSE: A Foundation Model for Multisensor Earth Observation Data.* ICLR ML4RS Workshop. [P110]
- Wu, Z., Pan, S., Long, G., Jiang, J., & Zhang, C. (2019). *Graph WaveNet for Deep Spatial-Temporal Graph Modeling.* IJCAI. [P091]
- Xiong, Z., Wang, Y., Zhang, F., Stewart, A. J., Hanna, J., Borth, D., Papoutsis, I., Le Saux, B., Camps-Valls, G., & Zhu, X. X. (2025). *Neural Plasticity-Inspired Multimodal Foundation Model for Earth Observation (DOFA).* arXiv:2403.15356. [P106]
- Yan, Z. et al. (2020). *Analysis of global marine oil trade from AIS.* [P015]
- Yılmaz and Zehir (2026). *Strategic-Risk-Based Forecasting of Brent Crude Oil Prices.* Entropy 28(5). [P076]
- Zhao, G., Xue, M., & Cheng, L. (2023). *A New Hybrid Model for Multi-Step WTI Futures Price Forecasting Based on Self-Attention Mechanism and Spatial–Temporal Graph Neural Network (GWNet-Attn).* Resources Policy 85:103956. [P063]
- Zhao, J., Zhang, M., Zhou, Z., Wang, Z., Lang, F., Shi, H., & Zheng, N. (2025). *CFFormer: A Cross-Fusion Transformer Framework for the Semantic Segmentation of Multisource Remote Sensing Images.* IEEE TGRS 63:4401117. [P112]
- Zhou, G., Qian, L., & Gamba, P. (2025). *Advances on Multimodal Remote Sensing Foundation Models for Earth Observation Downstream Tasks: A Survey.* Remote Sensing 17(21):3532. [P115]
