# Chapter 2 — Literature Review (~4000)

Oil-price forecasts matter because price movements feed into inflation, trade balances, producer revenues and energy costs, and therefore into hedging, budgeting and market-risk decisions. This chapter reviews the literatures that define how such forecasts are built and judged—econometric and machine-learning oil-price models, shipping and remote-sensing proxies, multimodal fusion, and evaluation practice—and then states the research gap this dissertation addresses.

## 2.1 Crude-oil price forecasting

### 2.1.1 Structural accounts and empirical benchmarks

Research on oil-price movements emphasises that similar price changes may arise from economically different sources. Kilian (2009) distinguishes among shocks to global crude-oil production, shocks to aggregate demand for industrial commodities and demand shocks specific to the oil market. The responses of the real price of oil differ across these shocks, as do their relationships with global economic activity and oil production. This structural account has been influential because it separates supply disturbances from changes in general economic activity and oil-market-specific demand, rather than treating every oil-price movement as the outcome of a single process.

A separate literature examines whether these and other economic relationships generate useful forecasts. Alquist, Kilian and Vigfusson (2013) compare a wide range of models with the no-change forecast, which sets the future spot price equal to the current price. Many alternatives fail to improve consistently on this benchmark, particularly in real time and out of sample. They also distinguish population predictability from forecastability: an economic variable may be related to future oil prices without producing lower forecast errors in a finite evaluation sample. Baumeister and Kilian (2015) study forecast combinations across six econometric specifications and find that combinations can produce more stable performance across horizons and periods than individual models. This literature characterises oil-price forecasting as a setting in which simple benchmarks remain competitive and model rankings are often unstable.

### 2.1.2 Evidence from machine-learning forecasts

Machine learning has widened the set of methods used in oil-price forecasting, particularly in studies with large predictor sets. Costa et al. (2021) evaluate 23 methods using 315 macroeconomic and financial variables in a pseudo-out-of-sample forecasting exercise. They find that no single method dominates across all forecast horizons. Machine-learning methods are competitive at short horizons. At horizons of up to six months, the strongest forecasts include LASSO-based models, oil-futures-based forecasts, a vector error-correction model and the Schwartz–Smith model, while forecast combinations become more relevant at longer horizons. These horizon-dependent results caution against treating XGBoost—or any other algorithm—as a default choice without horizon-specific evaluation. Using monthly Brent returns together with the geopolitical risk index, the VIX and the US ten-year Treasury yield, Yılmaz and Zehir (2026) compare econometric and tree-based models under a rolling-origin design. LightGBM records the most consistent performance across their reported horizons and train–test configurations. Both studies show that rankings vary with the horizon, predictor set and evaluation design.

Deep-learning and hybrid studies focus more directly on learned temporal representations. Foroutan and Lahmiri (2024) compare 16 models for next-day WTI (West Texas Intermediate, the main US crude benchmark) and Brent spot-price forecasting. TCN and LightGBM are among the strongest methods in their experiments, with TCN producing the lowest Brent errors across the input lengths considered. Simsek et al. (2024) combine LSTM feature extraction with XGBoost regression and report very high in-sample explanatory power. The two studies differ in their targets, evaluation samples, preprocessing and reported performance measures. More generally, comparisons based on highly persistent price levels are difficult to interpret because small one-step errors may partly reflect the proximity of consecutive prices rather than large gains in predictive content.

Graph-based methods represent another development in this literature. Zhao, Xue and Cheng (2023) use self-attention to estimate time-varying interactions among economic and financial variables and apply Graph WaveNet (Wu et al., 2019) to multi-step WTI futures forecasting. The graph in their model represents statistical relationships among predictors rather than a geographic or transport network. The proposed model outperforms the fitted baselines reported in the study, although the evaluation does not include a no-change price forecast.

Across these studies, machine learning is associated with greater flexibility in modelling high-dimensional predictors, nonlinearities and temporal interactions. The empirical results do not, however, establish a stable ranking in which a single class of algorithms consistently dominates econometric models, market-based forecasts or forecast combinations. Differences in targets, horizons, samples and benchmarks remain central to the variation in reported results.

## 2.2 Shipping activity and oil markets



### 2.2.1 AIS-based measurement of seaborne trade

Automatic Identification System (AIS) data record vessel identities, positions and movements rather than the quantities or types of cargo carried. A growing literature has nevertheless developed methods for converting these records into estimates of maritime trade. Adland, Jia and Strandenes (2017) compare AIS-derived estimates of seaborne crude-oil exports with customs statistics. Their aggregate estimates align reasonably well with official data, although discrepancies vary across countries and periods because pipelines and transshipment are not fully observed. Yan et al. (2020) combine tanker trajectories with vessel shape, size and draught to estimate voyage-level oil flows. Their estimates for major importers and exporters are strongly correlated with Joint Organisations Data Initiative (JODI) statistics, and their 2017 results identify the Middle East–Malacca–East Asia corridor as the largest route in the global marine oil network.

AIS data have also been used for high-frequency estimates of broader trade activity. Arslanalp, Marini and Tumbarello (2019) construct indicators from filtered port calls and show that they can improve the timeliness of trade monitoring. IMF PortWatch extends this approach by combining vessel movements, port and chokepoint information, ship characteristics and estimated cargo capacity to produce daily indicators of maritime trade (Arslanalp et al., 2026). These studies differ in spatial coverage and commodity detail, but they share a reliance on processed vessel activity as an indirect measure of trade.

### 2.2.2 Measurement limitations and price–shipping relationships

The accuracy of AIS-derived indicators depends on how vessel observations are translated into estimates of activity. Simple vessel counts assign the same weight to ships of different sizes and do not distinguish laden voyages from ballast movements. Vessel capacity and changes in draught provide additional information about likely cargo movement, but draught fields, port calls and vessel classifications may be incomplete or inconsistent (Yan et al., 2020; Arslanalp, Marini and Tumbarello, 2019). Transshipment introduces a further difficulty because a vessel’s observed itinerary may not coincide with the economic origin or final destination of its cargo (Adland, Jia and Strandenes, 2017).

Coverage is also incomplete. Paolo et al. (2024) combine satellite imagery with vessel-position data and estimate that 21–30% of transport- and energy-vessel activity is absent from public tracking systems. Part of this gap may reflect weak satellite reception rather than deliberate non-broadcasting by vessels. AIS-derived measures consequently observe only part of maritime activity and contain errors arising from reception, classification and cargo attribution.

Studies of oil prices and tanker activity add a separate interpretive issue. Mi et al. (2022) examine associations between oil-price changes and tanker port-call frequency, average docking time, total gross tonnage and the number of distinct tankers at ports in major crude-exporting countries. Mi et al. (2023) also model tanker port calls as a response to oil prices and report nonlinear and regionally heterogeneous relationships. Their dependent variable is shipping activity rather than oil prices. The results therefore document price-to-shipping responses and show why contemporaneous correlations between the two series do not establish a single direction of influence.

### 2.2.3 Network models of maritime flows

Maritime activity is represented in different ways across the vessel-flow literature. Aggregate studies use port calls, vessel counts, capacity-weighted transits or chokepoint volumes, whereas network studies represent ports or regions as nodes and vessel movements as links. The latter representation retains origin–destination relationships and allows traffic at one location to be modelled in relation to activity elsewhere in the network.

Ouyang et al. (2022) construct a crude-oil maritime transportation network from vessel trajectories, route information, crude-oil berths and supply–demand links. Their LSTM–GCN forecasts weekly traffic flows at network nodes. Liang et al. (2022) use a spatiotemporal multigraph convolutional network for fine-grained vessel-traffic forecasting, while Zhao et al. (2022) employ a dynamic graph neural network to predict regional vessel inflows, outflows and traffic volumes. These studies address different spatial scales and definitions of traffic, but all treat maritime movement as a relational and time-varying process.

## 2.3 Satellite imagery and remote sensing



### 2.3.1 Remote sensing as economic measurement

Remote sensing provides repeated observations of infrastructure, emissions and activity patterns, but different sensors measure different physical phenomena. Night-time lights record emitted radiance; atmospheric observations can capture pollutants such as tropospheric NO₂; cloud products describe observation conditions; and optical or synthetic-aperture radar imagery records surface structure. Economic interpretations are therefore usually tied to a specific mechanism linking the observed signal to an activity of interest.

Night-time-light studies illustrate how measurement properties vary with spatial and temporal scale. Polinov, Bookman and Levin (2022) find strong cross-sectional associations between VIIRS night-time lights and country-level shipping indicators across hundreds of anchorage areas. They also report that activity is difficult to estimate where only a small number of vessels generate limited light. Gibson et al. (2021) find that VIIRS is more informative than DMSP as a spatial proxy for subnational GDP, particularly at finer spatial levels and in less densely populated areas. Both applications are primarily concerned with differences across places.

Temporal variation is more difficult to interpret. Small (2021) shows that spatial differences account for most of the observed variation in VIIRS night-time lights and that some month-to-month variation is associated with viewing geometry, atmospheric conditions, background luminance and other features of the imaging process. The literature consequently distinguishes persistent differences in brightness across locations from changes within the same location over time.

### 2.3.2 Applications to oil, trade and infrastructure

Remote-sensing variables have been linked to oil markets through several distinct channels. Hao and Wang (2023) use MODIS (Moderate Resolution Imaging Spectroradiometer) cloud-cover observations over floating-roof tanks in eight major US storage areas. They find that greater cloudiness in one week predicts lower WTI returns in the following week. Their explanation is based on information availability: cloud cover obstructs optical observation of storage tanks and may reduce the inventory information available to market participants. Bricongne et al. (2026) study a different mechanism by using satellite observations of tropospheric NO₂ to nowcast national oil demand. Across advanced and emerging economies, NO₂ improves accuracy relative to autoregressive models and models using conventional predictors, with the largest gains reported for nonlinear models, particularly neural networks.

Other studies use imagery to measure infrastructure or trade rather than prices. Wang et al. (2019) estimate the height, radius and structural volume of oil tanks from high-resolution Gaofen-2 optical imagery. Their method measures storage capacity but not the quantity of oil held in a tank at a particular time. Jung (2026) combines Sentinel-1 synthetic-aperture radar measures, VIIRS night-time lights and port attributes in an XGBoost model to nowcast monthly port-level trade. Satellite variables help to track changes within ports over time, while static port characteristics account for much of the cross-sectional variation. In this application, remote-sensing observations enter the model as engineered numeric features rather than learned image representations.

These applications show that the economic content of remote sensing is specific to the observed signal and outcome. Cloud cover has been studied as a constraint on information, NO₂ as an indicator of combustion and demand, high-resolution imagery as a measure of infrastructure, and combined satellite features as indicators of port trade. Evidence obtained for one of these mechanisms does not automatically extend to other sensors, spatial scales or economic outcomes.

## 2.4 Multimodal learning and heterogeneous data



### 2.4.1 Multimodal learning and fusion strategies

Baltrušaitis, Ahuja and Morency (2019) define multimodal machine learning as the processing and relating of information from multiple modalities. Their taxonomy organises the field around five challenges: representation, translation, alignment, fusion and co-learning. Within the fusion literature, studies are also commonly distinguished by the stage at which information is combined. Input- or feature-level fusion combines observed or engineered features before modelling; representation-level fusion combines outputs from modality-specific encoders; and decision-level fusion combines model predictions.

These strategies make different assumptions about the structure retained from each data source. Feature-level fusion represents heterogeneous inputs in a common predictor space and is compatible with many conventional statistical and machine-learning models. Representation-level approaches preserve separate processing streams for at least part of the model. Arevalo et al. (2017) propose the Gated Multimodal Unit, which uses multiplicative gates to combine modality-specific representations in an input-dependent manner. The model was introduced for multimodal classification tasks involving text and images rather than for time-series forecasting.

Gohari et al. (2024) apply modality-aware modelling in a financial time-series setting. Their model uses separate streams together with intra-modal and inter-modal attention to combine Federal Reserve reports and numerical economic series when forecasting US interest rates. It outperforms several Transformer and time-series baselines across most of the reported settings. The application differs from the original gated-unit study in both data and architecture, but both treat the contribution of a modality as something that can vary across observations rather than as a fixed relationship.

### 2.4.2 Representation learning in Earth observation

Self-supervised learning has expanded the ways in which satellite imagery can be represented. SatMAE (Cong et al., 2022) adapts masked-autoencoder pretraining to temporal and multispectral satellite imagery and incorporates temporal and spectral information into the learning process. Prithvi-EO-2.0 (Szwarcman et al., 2026) is pretrained on global multitemporal samples from the Harmonized Landsat and Sentinel-2 archive and incorporates temporal and location embeddings. Both models use large collections of unlabelled imagery to learn representations that can be transferred to downstream Earth-observation tasks.

CROMA (Fuller, Millard and Green, 2023) focuses on relationships between Earth-observation sensors. It separately encodes spatially and temporally aligned optical and radar observations, applies cross-modal contrastive learning and then produces a joint representation through a fusion encoder. The separate processing streams reflect differences between optical and radar data in channel structure, noise and physical interpretation.

Evaluations of SatMAE, Prithvi-EO-2.0 and CROMA primarily cover land-cover classification, semantic segmentation, disaster mapping and related remote-sensing tasks. Across these applications, pretrained encoders provide transferable representations for a range of downstream Earth-observation tasks.

### 2.4.3 Missing and irregular observations

Multimodal data may be incomplete in more than one sense. In some cases, an entire modality is absent; in others, the modality exists but its observations are irregular, delayed or recorded at a different frequency. The first problem has been studied in the missing-modality literature. Ma et al. (2022) find that multimodal Transformers can be sensitive to the absence of one or more modalities and that robustness varies across fusion strategies and datasets. Neverova et al. (2016) introduce ModDrop, which randomly removes modality channels during training, and report improved robustness in gesture-recognition tasks when inputs are unavailable.

A related time-series literature examines irregular observation times. GRU-D (Che et al., 2018) incorporates observation masks and the time elapsed since the previous observation, allowing both missingness and observation age to affect the hidden state. Multi-Time Attention Networks (Shukla and Marlin, 2021) use continuous-time embeddings and attention to represent a variable number of irregularly timed observations. These approaches differ from missing-modality methods because they focus on the timing and availability of observations within a data stream rather than the absence of an entire stream.

Together, these studies distinguish between two sources of incompleteness that are sometimes conflated in multimodal applications: the absence of a whole modality and irregular sampling within an available modality. They also show that alignment and missingness are modelling problems in their own right, rather than properties resolved automatically by combining additional data sources.

## 2.5 Forecast evaluation and model interpretation



### 2.5.1 Measuring and comparing predictive accuracy

Forecast evaluation involves both the choice of an evaluation criterion and the assessment of uncertainty around observed performance differences. Point forecasts are commonly summarised using loss measures such as mean absolute error and root mean squared error. Probabilistic forecasts can instead be evaluated using proper scoring rules, including the Brier score for binary outcomes and log loss (Gneiting and Raftery, 2007). Directional forecasts have also motivated specialised procedures. Pesaran and Timmermann (1992), for example, develop a test of whether predicted and realised directions are independent.

Volatility forecasting raises an additional issue because realised volatility or variance is itself estimated from observed returns. Patton (2011) analyses forecast comparison when the volatility proxy is imperfect and identifies conditions under which particular loss functions preserve the ranking of competing forecasts. QLIKE has consequently become common in variance-forecast evaluation alongside squared- and absolute-error measures.

Formal comparison tests examine whether observed loss differences are distinguishable from sampling variation. Diebold and Mariano (1995) develop a general test of equal expected predictive loss that permits non-quadratic loss functions and serially correlated loss differentials. Clark and West (2007) consider the more specific case of explicitly nested models under squared-error loss. Their adjustment addresses the tendency of parameter estimation in the larger model to increase its out-of-sample error under the null. The two procedures therefore address related but different forecast-comparison settings.

### 2.5.2 Model interpretation and its limits

The increasing use of machine learning in forecasting has been accompanied by greater interest in post-hoc interpretation. SHAP provides an additive decomposition of an individual prediction into feature attributions (Lundberg and Lee, 2017). Aggregating these attributions across observations can produce global summaries of model behaviour, while grouping features can provide higher-level summaries. The resulting values depend, fhowever, on how the absence of a feature is represented and on assumptions about relationships among predictors.

Feature dependence is particularly important in economic data. Aas, Jullum and Løland (2021) show that independence-based SHAP procedures may evaluate unrealistic combinations of correlated predictors and develop approximations that account for dependence. Their results demonstrate that feature attribution is not invariant to the distributional assumptions used to construct the comparison.

Some model architectures also expose internal weights that can be inspected. The Gated Multimodal Unit of Arevalo et al. (2017), for example, produces input-dependent gate values, while attention-based models assign weights across elements of an input or representation. These quantities describe operations inside the fitted model, but their status as explanations is contested. Jain and Wallace (2019) show that substantially different attention patterns can sometimes produce similar predictions and that attention weights need not align with other measures of feature importance. More generally, feature attributions, gates and attention weights describe relationships within a predictive model; they do not by themselves identify causal effects in the process being forecast.

## 2.6 Synthesis, research gap and positioning



### 2.6.1 Synthesis of the literature

Four conclusions emerge from the review. Oil-price forecasting is difficult because oil prices are highly persistent and the no-change benchmark is strong. Economic and financial predictors are widely used in this literature because they capture persistence, uncertainty, monetary conditions, exchange-rate channels and market expectations. Shipping and remote-sensing data are plausible alternative-data sources, but they are noisy and indirect proxies rather than direct measurements of future prices. Multimodal learning offers tools for preserving modality-specific structure, but these tools have not been systematically tested for commodity-price forecasting with heterogeneous shipping and satellite inputs.

The following table summarises the observable signal, economic channel and main limitation of each of the four literatures, with key citations for each strand.


| Data source / literature           | Observable signal                                                                                                          | Economic channel                                                                        | Main limitation                                                             | Key references                                                                                                                                                                                                                     |
| ---------------------------------- | -------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------- | --------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Financial and oil-market variables | Lagged price, inventories, production/refinery activity, volatility, GPR, rates, exchange rates, futures/market indicators | Persistence, uncertainty, macro-financial conditions, market expectations               | Strong benchmark; difficult to improve upon                                 | Kilian (2009); Alquist, Kilian and Vigfusson (2013); Baumeister and Kilian (2015); Costa et al. (2021); Yılmaz and Zehir (2026)                                                                                                    |
| Shipping / AIS / PortWatch         | Tanker flows, port calls, chokepoint transits, capacity-weighted activity                                                  | Physical trade, supply disruption, congestion, regional flow changes                    | Directionality, noisy cargo inference, missing AIS activity                 | Adland, Jia and Strandenes (2017); Yan et al. (2020); Arslanalp, Marini and Tumbarello (2019); Arslanalp et al. (2026); Mi et al. (2022, 2023); Paolo et al. (2024); Ouyang et al. (2022); Liang et al. (2022); Zhao et al. (2022) |
| Remote sensing                     | Night-time lights, NO₂, cloud cover, site-level imagery or embeddings                                                      | Industrial activity, demand conditions, inventory observability, infrastructure signals | Indirect mechanism, weak within-site temporal variation, cloud/missing data | Gibson et al. (2021); Polinov, Bookman and Levin (2022); Small (2021); Hao and Wang (2023); Wang et al. (2019); Bricongne et al. (2026); Jung (2026)                                                                               |
| Multimodal learning                | Modality-specific representations and fusion                                                                               | Preservation of heterogeneous structure before prediction                               | Limited direct evidence in oil-price forecasting                            | Baltrušaitis, Ahuja and Morency (2019); Arevalo et al. (2017); Gohari et al. (2024); Cong et al. (2022); Fuller, Millard and Green (2023); Szwarcman et al. (2026); Ma et al. (2022)                                               |




### 2.6.2 Research gap

Taken together, the literatures reviewed above reveal three unresolved issues at the intersection of oil-price forecasting, alternative data and multimodal learning.

First, evidence on the predictive value of shipping and remote-sensing data remains fragmented. Oil-price forecasting studies have concentrated mainly on historical prices and macro-financial or oil-market variables. By contrast, AIS and remote-sensing studies have more often examined maritime traffic, trade, oil demand, infrastructure or information availability. These studies show that shipping and satellite observations contain economically relevant information, but there is limited direct evidence on whether they improve one-week-ahead Brent price forecasts beyond established predictors.

Second, the literatures represent alternative data in different ways. Many economic applications reduce shipping and satellite observations to engineered numeric indicators. Research on maritime networks and Earth-observation foundation models instead preserves relational or spatial structure through graph-based and pretrained representations. These neighbouring literatures demonstrate that such structures can be modelled, but they do not provide a controlled comparison between engineered features and learned representations in oil-market forecasting. It therefore remains unclear whether retaining modality-specific structure provides predictive information beyond that contained in the underlying data.

Third, predictive performance and model interpretation are not examined within a consistent comparative setting across studies. Published results differ in their forecast targets, horizons, samples, information sets, benchmarks and evaluation procedures. Some comparisons also change the input data and model architecture simultaneously, making it difficult to determine whether an observed improvement comes from the additional information, its representation or the forecasting algorithm. Interpretability analyses similarly use different feature-attribution methods or model-internal weights and are often reported separately from out-of-sample performance. This limits comparison of how models use different information sources across targets and market conditions. Examining predictive performance and model reliance within the same evaluation setting would provide a clearer account of both forecast differences and model behaviour, without treating attribution as evidence of causality.

These gaps concern three related issues: whether alternative data contain incremental predictive information, whether their representation affects forecasting performance, and whether predictive performance and model reliance can be compared within a common evaluation design.

### 2.6.3 Positioning of this dissertation

These gaps motivate the research questions stated in Section 1.2. This dissertation addresses them through an empirical comparison of one-week-ahead Brent price forecasts under a shared out-of-sample design: financial time series are expanded with shipping and/or remote sensing, and flat feature fusion is paired with modality-aware fusion on matched information sets. Interpretability is used only to describe modality reliance where forecasts already improve on the no-change benchmark, not as causal evidence. Implementation details follow in Chapter 3.

---



## References

Aas, K., Jullum, M. and Løland, A. (2021). ‘Explaining individual predictions when features are dependent: more accurate approximations to Shapley values’, *Artificial Intelligence*, 298, 103502. doi: 10.1016/j.artint.2021.103502.

Adland, R., Jia, H. and Strandenes, S.P. (2017). ‘Are AIS-based trade volume estimates reliable? The case of crude oil exports’, *Maritime Policy & Management*, 44(5), pp. 657–665. doi: 10.1080/03088839.2017.1309470.

Alquist, R., Kilian, L. and Vigfusson, R.J. (2013). ‘Forecasting the price of oil’, in Elliott, G. and Timmermann, A. (eds.) *Handbook of Economic Forecasting*. Vol. 2A. Amsterdam: Elsevier, pp. 427–507. doi: 10.1016/B978-0-444-53683-9.00008-6.

Arevalo, J., Solorio, T., Montes-y-Gómez, M. and González, F.A. (2017). ‘Gated multimodal units for information fusion’, *ICLR 2017 Workshop Track*. Toulon, France, 24–26 April. Available at: [https://openreview.net/forum?id=S12_nquOe](https://openreview.net/forum?id=S12_nquOe) (Accessed: 1 July 2026).

Arslanalp, S., Marini, M. and Tumbarello, P. (2019). *Big data on vessel traffic: nowcasting trade flows in real time*. IMF Working Paper WP/19/275. Washington, DC: International Monetary Fund. Available at: [https://www.imf.org/en/publications/wp/issues/2019/12/13/big-data-on-vessel-traffic-nowcasting-trade-flows-in-real-time-48837](https://www.imf.org/en/publications/wp/issues/2019/12/13/big-data-on-vessel-traffic-nowcasting-trade-flows-in-real-time-48837) (Accessed: 1 July 2026).

Arslanalp, S., Exton, O., Gao, C., Kamali, P., Saraiva, M., Sozzi, A. and Verschuur, J. (2026). *Nowcasting country-level trade estimates using IMF PortWatch*. IMF Working Paper WP/26/99. Washington, DC: International Monetary Fund. doi: 10.5089/9798229046893.001.

Baltrušaitis, T., Ahuja, C. and Morency, L.-P. (2019). ‘Multimodal machine learning: a survey and taxonomy’, *IEEE Transactions on Pattern Analysis and Machine Intelligence*, 41(2), pp. 423–443. doi: 10.1109/TPAMI.2018.2798607.

Baumeister, C. and Kilian, L. (2015). ‘Forecasting the real price of oil in a changing world: a forecast combination approach’, *Journal of Business & Economic Statistics*, 33(3), pp. 338–351. doi: 10.1080/07350015.2014.949342.

Bricongne, J.-C., Macalos, J., Meunier, B., Milis, J. and Pical, T. (2026). *Can satellites predict oil demand?* ECB Working Paper Series No. 3198. Frankfurt am Main: European Central Bank. Available at: [https://www.ecb.europa.eu/pub/pdf/scpwps/ecb.wp3198~e3858c52a3.en.pdf](https://www.ecb.europa.eu/pub/pdf/scpwps/ecb.wp3198~e3858c52a3.en.pdf) (Accessed: 1 July 2026).

Che, Z., Purushotham, S., Cho, K., Sontag, D. and Liu, Y. (2018). ‘Recurrent neural networks for multivariate time series with missing values’, *Scientific Reports*, 8, 6085. doi: 10.1038/s41598-018-24271-9.

Clark, T.E. and West, K.D. (2007). ‘Approximately normal tests for equal predictive accuracy in nested models’, *Journal of Econometrics*, 138(1), pp. 291–311. doi: 10.1016/j.jeconom.2006.05.023.

Cong, Y., Khanna, S., Meng, C., Liu, P., Rozi, E., He, Y., Burke, M., Lobell, D.B., et al. (2022). ‘SatMAE: pre-training transformers for temporal and multi-spectral satellite imagery’, *Advances in Neural Information Processing Systems*, 35, pp. 197–211.

Costa, A.B.R., Ferreira, P.C.G., Gaglianone, W.P., Guillén, O.T.C., Issler, J.V. and Lin, Y. (2021). ‘Machine learning and oil price point and density forecasting’, *Energy Economics*, 102, 105494. doi: 10.1016/j.eneco.2021.105494.

Diebold, F.X. and Mariano, R.S. (1995). ‘Comparing predictive accuracy’, *Journal of Business & Economic Statistics*, 13(3), pp. 253–263. doi: 10.1080/07350015.1995.10524599.

Foroutan, P. and Lahmiri, S. (2024). ‘Deep learning systems for forecasting the prices of crude oil and precious metals’, *Financial Innovation*, 10, 111. doi: 10.1186/s40854-024-00637-z.

Fuller, A., Millard, K. and Green, J.R. (2023). ‘CROMA: remote sensing representations with contrastive radar-optical masked autoencoders’, *Advances in Neural Information Processing Systems*, 36, pp. 5506–5538.

Gneiting, T. and Raftery, A.E. (2007). ‘Strictly proper scoring rules, prediction, and estimation’, *Journal of the American Statistical Association*, 102(477), pp. 359–378. doi: 10.1198/016214506000001437.

Gibson, J., Olivia, S., Boe-Gibson, G. and Li, C. (2021). ‘Which night lights data should we use in economics, and where?’, *Journal of Development Economics*, 149, 102602. doi: 10.1016/j.jdeveco.2020.102602.

Gohari, H.E., Dang, X.-H., Shah, S.Y. and Zerfos, P. (2024). ‘Modality-aware transformer for financial time series forecasting’, in *Proceedings of the 5th ACM International Conference on AI in Finance (ICAIF ’24)*. New York: Association for Computing Machinery, pp. 677–685. doi: 10.1145/3677052.3698654.

Hao, X. and Wang, Y. (2023). ‘Cloud cover and expected oil returns’, *Humanities and Social Sciences Communications*, 10, 605. doi: 10.1057/s41599-023-02128-5.

Jain, S. and Wallace, B.C. (2019). ‘Attention is not explanation’, *Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies*. Minneapolis, MN: Association for Computational Linguistics, pp. 3543–3556. doi: 10.18653/v1/N19-1357.

Jung, Y. (2026). ‘Watching trade from space: nowcasting and spatial extrapolation of port-level maritime trade using satellite imagery’, arXiv:2604.15444 [Preprint]. Available at: [https://arxiv.org/abs/2604.15444](https://arxiv.org/abs/2604.15444) (Accessed: 1 July 2026).

Kilian, L. (2009). ‘Not all oil price shocks are alike: disentangling demand and supply shocks in the crude oil market’, *American Economic Review*, 99(3), pp. 1053–1069. doi: 10.1257/aer.99.3.1053.

Liang, M., Liu, R.W., Zhan, Y., Li, H., Zhu, F. and Wang, F.-Y. (2022). ‘Fine-grained vessel traffic flow prediction with a spatio-temporal multigraph convolutional network’, *IEEE Transactions on Intelligent Transportation Systems*, 23(12), pp. 23694–23707. doi: 10.1109/TITS.2022.3199160.

Lundberg, S.M. and Lee, S.-I. (2017). ‘A unified approach to interpreting model predictions’, *Advances in Neural Information Processing Systems*, 30, pp. 4765–4774. Available at: [https://papers.nips.cc/paper/7062-a-unified-approach-to-interpreting-model-predictions](https://papers.nips.cc/paper/7062-a-unified-approach-to-interpreting-model-predictions) (Accessed: 1 July 2026).

Ma, M., Ren, J., Zhao, L., Testuggine, D. and Peng, X. (2022). ‘Are multimodal transformers robust to missing modality?’, in *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*. New Orleans, LA: IEEE, pp. 18177–18186. doi: 10.1109/CVPR52688.2022.01764.

Mi, J.J., Meng, X., Chen, Y. and Wang, Y. (2022). ‘The impact of the crude oil price on tankers’ port-call features: mining the information in automatic identification system’, *Journal of Marine Science and Engineering*, 10(10), 1559. doi: 10.3390/jmse10101559.

Mi, J.J., Zang, X., Lo, K.L. and Chen, Y. (2023). ‘The nonlinear relationship between oil prices and the number of tankers’ port calls: evidence from AIS data’, *Procedia Computer Science*, 221, pp. 870–877. doi: 10.1016/j.procs.2023.08.063.

Neverova, N., Wolf, C., Taylor, G.W. and Nebout, F. (2016). ‘ModDrop: adaptive multi-modal gesture recognition’, *IEEE Transactions on Pattern Analysis and Machine Intelligence*, 38(8), pp. 1692–1706. doi: 10.1109/TPAMI.2015.2461544.

Ouyang, Q., Sun, T., Xue, Y. and Liu, Z. (2022). ‘Long short-term memory and graph convolution network for forecasting the crude oil traffic flow’, *IEEE Access*, 10, pp. 18922–18932. doi: 10.1109/ACCESS.2022.3150852.

Paolo, F.S., Kroodsma, D., Raynor, J., Hochberg, T., Davis, P., Cleary, J., Marsaglia, L., Orofino, S., et al. (2024). ‘Satellite mapping reveals extensive industrial activity at sea’, *Nature*, 625, pp. 85–91. doi: 10.1038/s41586-023-06825-8.

Patton, A.J. (2011). ‘Volatility forecast comparison using imperfect volatility proxies’, *Journal of Econometrics*, 160(1), pp. 246–256. doi: 10.1016/j.jeconom.2010.03.034.

Pesaran, M.H. and Timmermann, A. (1992). ‘A simple nonparametric test of predictive performance’, *Journal of Business & Economic Statistics*, 10(4), pp. 461–465. doi: 10.1080/07350015.1992.10509922.

Polinov, S., Bookman, R. and Levin, N. (2022). ‘A global assessment of night lights as an indicator for shipping activity in anchorage areas’, *Remote Sensing*, 14(5), 1079. doi: 10.3390/rs14051079.

Shukla, S.N. and Marlin, B.M. (2021). ‘Multi-time attention networks for irregularly sampled time series’, *International Conference on Learning Representations (ICLR 2021)*. Online, 3–7 May. Available at: [https://openreview.net/forum?id=4c0J6lwQ4](https://openreview.net/forum?id=4c0J6lwQ4) (Accessed: 1 July 2026).

Simsek, A.I., Bulut, E., Gur, Y.E. and Gültekin Tarla, E. (2024). ‘A novel approach to predict WTI crude spot oil price: LSTM-based feature extraction with Xgboost regressor’, *Energy*, 309, 133102. doi: 10.1016/j.energy.2024.133102.

Small, C. (2021). ‘Spatiotemporal characterization of VIIRS night light’, *Frontiers in Remote Sensing*, 2, 775399. doi: 10.3389/frsen.2021.775399.

Szwarcman, D., Roy, S., Fraccaro, P., Gíslason, Þ.E., Blumenstiel, B., Ghosal, R., de Oliveira, P.H., de Sousa Almeida, J.L., et al. (2026). ‘Prithvi-EO-2.0: a versatile multitemporal foundation model for earth observation applications’, *IEEE Transactions on Geoscience and Remote Sensing*, 64, 4400120. doi: 10.1109/TGRS.2025.3642610.

Wang, T., Li, Y., Yu, S. and Liu, Y. (2019). ‘Estimating the volume of oil tanks based on high-resolution remote sensing images’, *Remote Sensing*, 11(7), 793. doi: 10.3390/rs11070793.

Wu, Z., Pan, S., Long, G., Jiang, J. and Zhang, C. (2019). ‘Graph WaveNet for deep spatial-temporal graph modeling’, in *Proceedings of the Twenty-Eighth International Joint Conference on Artificial Intelligence (IJCAI-19)*. Macao, China, 10–16 August. International Joint Conferences on Artificial Intelligence Organization, pp. 1907–1913. doi: 10.24963/ijcai.2019/264.

Yan, Z., Xiao, Y., Cheng, L., Chen, S., Zhou, X., Ruan, X., Li, M., He, R., et al. (2020). ‘Analysis of global marine oil trade based on automatic identification system (AIS) data’, *Journal of Transport Geography*, 83, 102637. doi: 10.1016/j.jtrangeo.2020.102637.

Yılmaz, T.E. and Zehir, C. (2026). ‘Strategic risk based forecasting of Brent crude oil prices: a comparative analysis of econometric and machine learning models’, *Entropy*, 28(5), 539. doi: 10.3390/e28050539.

Zhao, C., Li, X., Zuo, M., Mo, L. and Yang, C. (2022). ‘Spatiotemporal dynamic network for regional maritime vessel flow prediction amid COVID-19’, *Transport Policy*, 129, pp. 78–89. doi: 10.1016/j.tranpol.2022.09.029.

Zhao, G., Xue, M. and Cheng, L. (2023). ‘A new hybrid model for multi-step WTI futures price forecasting based on self-attention mechanism and spatial–temporal graph neural network’, *Resources Policy*, 85, 103956. doi: 10.1016/j.resourpol.2023.103956.