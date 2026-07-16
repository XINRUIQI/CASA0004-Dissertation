# Chapter 2 — Literature Review (~3200words)

This chapter reviews the literature needed to position this dissertation. Section 2.1 reviews crude-oil price forecasting and explains why the random-walk benchmark and financial baselines are difficult to beat. Sections 2.2 and 2.3 examine two alternative-data sources — maritime/AIS shipping activity and satellite remote sensing — as economic proxies for oil-market conditions. Section 2.4 reviews multimodal forecasting and the distinction between flat feature fusion and representation-level modality-aware fusion. Section 2.5 reviews forecast evaluation and interpretability standards. Section 2.6 synthesises these strands into the research gap addressed by this dissertation.

## 2.1 Crude-oil price forecasting



### 2.1.1 Econometric foundations and the benchmark problem

Oil prices are difficult to predict out of sample. Kilian (2009) shows that price movements should be understood through different structural channels, including crude-oil supply shocks, aggregate-demand shocks and oil-specific precautionary-demand shocks. The decomposition is not itself a forecasting model, but it does offer a useful principle for predictor selection: useful variables should have a plausible economic connection to supply, demand or uncertainty, rather than being included only because they are available.

Alquist, Kilian and Vigfusson (2013) provide the key forecasting benchmark for this dissertation. They show that the no-change, or random-walk, forecast is extremely difficult to beat in oil-price forecasting, especially out of sample. Their review also emphasises that in-sample fit does not imply forecasting skill, and that claims of predictability should be tested using real-time data alignment, recursive or rolling evaluation and formal forecast-comparison tests. Baumeister and Kilian (2015) further show that forecast combinations across different economic mechanisms can be more robust than reliance on a single predictor set. Taken together, these studies set a high bar: any alternative-data model must be evaluated against a strong no-change benchmark and an economically informed financial baseline, not only against other machine-learning models.

### 2.1.2 Machine-learning approaches to oil-price forecasting

Machine-learning studies have introduced tree ensembles, regularised linear models, deep learning and hybrid models into oil-price forecasting. Costa et al. (2021) compare a broad set of methods over a large macro-financial predictor set and find that useful predictors vary across horizons and over time. XGBoost can be a strong benchmark, but it does not dominate uniformly. Yılmaz and Zehir (2026) show that geopolitical risk, market volatility and interest-rate variables can add value for Brent-return forecasting, with LightGBM outperforming XGBoost in their setting. Foroutan and Lahmiri (2024) report strong performance from temporal convolutional networks and gradient-boosting models, but their focus on price-level prediction also illustrates a common problem: because oil prices are highly persistent, low one-step price-level errors can partly reflect the fact that P_{t+1} is usually close to P_t.

Several recent studies use more complex architectures, but their results should be read cautiously. Simsek et al. (2024) report that hybrid designs combining LSTM feature extraction with XGBoost can achieve very high R^2 values, yet such results may be sensitive to preprocessing choices such as pre-split normalisation. Graph-based oil-price models have also appeared. Zhao, Xue and Cheng (2023), for example, combine a self-attention-learned dynamic graph with Graph WaveNet (Wu et al., 2019) for multi-step WTI futures forecasting. Their graph represents non-Euclidean relations among predictors rather than a physical shipping or geographic network, and the absence of a no-change benchmark makes it difficult to assess whether the model improves over a random walk. I treat these papers as evidence that model complexity alone is not enough; they are background rather than direct templates for the weekly Brent forecasting problem addressed here.

For this dissertation, three points from this literature matter. The price's own lag and the random-walk forecast must be treated as serious competitors, not weak baselines. Financial and oil-market fundamentals provide an economically meaningful baseline before alternative data are added: alongside volatility, geopolitical risk, interest rates, exchange rates and market-based oil indicators, inventories, production and refinery activity are natural proxies for Kilian's supply and demand channels. Any claim that shipping or remote sensing improves forecasting must rest on incremental out-of-sample value over that baseline, not on raw error reduction alone.

## 2.2 Shipping activity and oil markets



### 2.2.1 AIS and maritime activity as trade-flow proxies

Automatic Identification System (AIS) vessel tracking has become an important high-frequency proxy for physical trade. Adland et al. (2017) validate AIS-derived crude-export estimates against official statistics, while Yan et al. (2020) show that global marine oil trade is concentrated around major chokepoints such as Hormuz, Malacca and Suez. Arslanalp, Marini and Tumbarello (2019) and IMF PortWatch (Arslanalp et al., 2026) further demonstrate how vessel movements can be used to nowcast trade activity. This literature supports treating shipping data as a plausible source of physical-market information, without assuming that AIS counts directly measure crude flows or predict prices.

Capacity-weighted indicators and draught-change measures are generally more informative than simple vessel counts, because they better approximate cargo movement. AIS data also require careful filtering to exclude non-trade activity, and moving averages must be constructed without using future observations. A shipping indicator is useful for forecasting only if it is both economically meaningful and available at the forecast origin. In practice, that means chokepoint transit volumes and tanker-presence measures are better read as activity proxies at key maritime nodes than as direct measures of global supply.

### 2.2.2 Reverse causality and proxy limitations

The relationship between shipping activity and oil prices is not one-directional. Mi et al. (2022) and Mi, Zang, Lo and Chen (2023) study how crude-oil prices affect tanker port-call activity, rather than how tanker activity predicts prices. Their findings indicate that the relationship is non-linear and regionally heterogeneous, and that statistically significant relationships may explain only a small share of variation. Shipping data may contain useful information about physical trade and congestion, but they may also respond to oil prices rather than lead them.

Chokepoint and port indicators are also imperfect proxies for crude-oil flows. A vessel's previous port is not always the cargo origin; crude oil may be blended or re-sold; and ship-to-ship transfers can obscure the true trade route. Paolo et al. (2024) further show that a substantial amount of industrial activity at sea is absent from AIS. Shipping indicators should therefore be interpreted as noisy proxies for physical-market conditions, not direct measurements of global oil supply.

### 2.2.3 From flat shipping indicators to maritime structure

Most oil-related uses of AIS and PortWatch-style data convert shipping activity into tabular features such as port calls, vessel counts or chokepoint transit volumes. This is useful, but it discards the network structure of maritime activity. Maritime trade is inherently spatial and relational: ports, terminals, chokepoints and routes form a connected transport system. Studies such as Ouyang et al. (2022) and Liang et al. (2022) show that graph-based models can learn spatial-temporal structure in vessel-flow prediction. Those studies forecast traffic flows rather than oil prices. They justify the idea that maritime structure is learnable, but they do not prove that a maritime representation improves Brent-price forecasting.

## 2.3 Satellite imagery and remote sensing



### 2.3.1 Remote sensing as an economic proxy

Remote sensing provides a physical view of economic activity, infrastructure and environmental conditions. In oil-related applications, night-time lights, NO₂, cloud cover and high-resolution imagery have all been used as indirect indicators of economic activity, trade, demand or inventory information. The literature is cautious about what these signals can and cannot measure.

Night-time lights are one of the most widely used remote-sensing proxies, but their usefulness depends on scale. Polinov, Bookman and Levin (2022) find that night-time lights correlate with anchorage activity at a broad cross-sectional scale, yet they do not reliably track tanker activity at a single port. Gibson et al. (2021) similarly show that VIIRS night-time lights are more suitable than DMSP for facility-scale work, but that night lights capture cross-sectional differences more reliably than within-unit temporal variation. Raw radiance should not be treated as a direct time-series measure of oil activity. Within-site anomalies are more defensible than raw levels when the research objective is forecasting over time.

Other remote-sensing indicators provide different mechanisms. Hao and Wang (2023) link cloud cover over US storage regions to next-week WTI returns through an information-availability channel: when clouds obstruct optical observation of storage tanks, market uncertainty about inventories may increase. Bricongne et al. (2026) use tropospheric NO₂ to nowcast national oil demand, but their results also show that the incremental value of remote-sensing variables can weaken inside non-linear models. Wang et al. (2019) estimate oil-tank structural capacity from high-resolution images, which supports infrastructure measurement but not high-frequency inventory estimation at Sentinel-2 resolution.

### 2.3.2 Limits of direct RS-to-price claims

Satellite data may contain oil-relevant information, but the literature does not support a simple claim that satellite imagery directly predicts oil prices. Most remote-sensing indicators are upstream proxies: they may reflect industrial activity, port activity, storage observability or demand conditions, which may then influence prices through supply-demand expectations. The mechanism is indirect and may vary across locations, sensors and time horizons.

Jung (2026) provides a useful example of this limitation. The study combines satellite-derived indicators with port attributes to nowcast port-level trade, but the target is trade rather than price, and the model relies on engineered tabular features rather than learned image representations. This is representative of a broader pattern: remote sensing is often converted into flat numeric columns before entering an economic model. Such features may be valuable, but what remains unclear is whether preserving image or site-level representations adds forecasting value for weekly Brent prices.

## 2.4 Multimodal forecasting and fusion



### 2.4.1 From multi-source data to multimodal learning

A key distinction in this dissertation is between multi-source feature fusion and multimodal representation learning. Baltrušaitis, Ahuja and Morency (2019) define multimodal learning as involving representation, translation, alignment, fusion and co-learning. Within this taxonomy, fusion can occur at the feature level, decision level or representation level. In much of the oil-price forecasting literature, heterogeneous data are combined through early feature-level fusion: financial indicators, shipping counts and satellite-derived indices are concatenated into a single table and passed to a conventional model. This approach is practical, but it treats all inputs as ordinary numeric predictors and may discard modality-specific structure.

A modality-aware alternative is to encode different data types separately before fusing their representations. At the fusion step itself, learned gating offers a way to weight modalities dynamically rather than treating them equally: Arevalo et al. (2017) propose gated multimodal units in which the model learns, for each input, how much each modality contributes to the fused representation. Gohari et al. (2024) provide a relevant precedent in financial time-series forecasting, showing that modality-aware transformers can outperform naïve concatenation when combining different financial information sources. Their application involves text and numeric data rather than satellite imagery, maritime networks and oil-market variables. The study motivates modality-aware forecasting, but it does not answer whether such fusion is useful for crude-oil price prediction.

### 2.4.2 Representation learning for Earth-observation data

Recent Earth-observation foundation models provide one route to transforming satellite imagery into representations. Models such as SatMAE (Cong et al., 2022) and Prithvi-EO-2.0 (Szwarcman et al., 2026) use self-supervised pretraining to produce transferable image embeddings. Multisensor models such as CROMA (Fuller et al., 2023) further show that optical and radar data may benefit from modality-specific encoders before fusion, because the sensors differ in channel structure, noise and physical meaning. These models are relevant because weekly oil-price datasets are usually too small to train an image encoder from scratch.

The EO foundation-model literature is mainly methodological rather than evidential for commodity-price forecasting. Most evaluations target land-cover classification, segmentation or related remote-sensing tasks, not economic forecasting or oil prices. The models show that satellite imagery can be represented by pretrained encoders; they do not show that such representations improve Brent forecasting.

### 2.4.3 Missing and asynchronous modalities

A further challenge is that alternative data sources are often incomplete or asynchronous. Optical satellite imagery is affected by cloud cover, radar and optical sensors have different revisit cycles, and shipping or macro-financial data may be released at different frequencies. General multimodal-learning studies show that models can degrade when modalities are missing unless missing-modality training or modality dropout is used (Ma et al., 2022; Neverova et al., 2016), and related work on irregularly sampled time series further motivates the use of masks and time-since-observation signals (Che et al., 2018; Shukla and Marlin, 2021).

Missingness and temporal misalignment are part of the multimodal forecasting problem itself, not a secondary data-cleaning issue. Any fair comparison between flat feature fusion and representation-level fusion must account for the availability, timing and reliability of each modality.

## 2.5 Forecast evaluation and interpretability



### 2.5.1 Forecast comparison

Because the random walk is difficult to beat, lower RMSE or MAE alone is not sufficient evidence of improved forecasting skill. Diebold and Mariano (1995) provide the standard framework for testing equal predictive accuracy across competing forecasts. For nested models, where one model extends another by adding predictors, Clark and West (2007) provide a more appropriate test under squared-error loss. These tests are especially relevant when evaluating alternative data, because the key question is not whether a large model can reduce error in one sample, but whether an added modality produces a statistically meaningful improvement over a simpler baseline.

Alternative-data claims should therefore be evaluated through out-of-sample comparison, horizon-specific performance and formal tests of predictive accuracy. That standard applies whenever a financial baseline is compared with models that add shipping, remote sensing or multimodal representations.

### 2.5.2 Interpretability and modality-level explanation

Forecast accuracy tests can show whether a model improves, but they do not explain which signals drive the improvement. Two complementary attribution routes are relevant here. For flat tabular models, SHAP (Lundberg and Lee, 2017) can attribute predictions to features and be aggregated to the modality level. Learned modality gates and site- or node-level attention provide native diagnostics of model dependence (Arevalo et al., 2017; as commonly used in graph-attention settings): they describe which modalities and spatial locations the model relies on over time, but they do not constitute causal explanations. Modality-level explanation matters because the dissertation is concerned not only with whether forecasts improve, but also with whether finance, shipping and remote-sensing signals enter the model in economically interpretable ways.

Attribution should also be benchmark-conditioned. Heavy interpretation of models that fail the relevant forecast benchmark cannot support a claim that the underlying signals are useful. In this dissertation, modality-level diagnostics are therefore emphasised for specifications that improve on the no-change benchmark. Attribution describes model behaviour; it does not establish causal effects on oil prices.

## 2.6 Synthesis, research gap and positioning



### 2.6.1 Synthesis of the literature

Four conclusions emerge from the review. Oil-price forecasting is difficult because oil prices are highly persistent and the random-walk benchmark is strong. Financial variables provide an essential economically informed baseline because they capture persistence, uncertainty, monetary conditions, exchange-rate channels and market expectations. Shipping and remote-sensing data are plausible alternative-data sources, but they are noisy and indirect proxies rather than direct measurements of future prices. Multimodal learning offers tools for preserving modality-specific structure, but these tools have not been systematically tested in the specific setting of weekly Brent forecasting.

The following table summarises the observable signal, economic channel and main limitation of each of the four literatures, with key citations for each strand.


| Data source / literature           | Observable signal                                                                                                          | Economic channel                                                                        | Main limitation                                                             | Key references                                                                                                                                                 |
| ---------------------------------- | -------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------- | --------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Financial and oil-market variables | Lagged price, inventories, production/refinery activity, volatility, GPR, rates, exchange rates, futures/market indicators | Persistence, uncertainty, macro-financial conditions, market expectations               | Strong benchmark; difficult to improve upon                                 | Kilian (2009); Alquist et al. (2013); Baumeister and Kilian (2015); Costa et al. (2021); Yılmaz and Zehir (2026)                                               |
| Shipping / AIS / PortWatch         | Tanker flows, port calls, chokepoint transits, capacity-weighted activity                                                  | Physical trade, supply disruption, congestion, regional flow changes                    | Reverse causality, noisy cargo inference, missing AIS activity              | Adland et al. (2017); Yan et al. (2020); Arslanalp et al. (2019, 2026); Mi et al. (2022, 2023); Paolo et al. (2024); Ouyang et al. (2022); Liang et al. (2022) |
| Remote sensing                     | Night-time lights, NO₂, cloud cover, site-level imagery or embeddings                                                      | Industrial activity, demand conditions, inventory observability, infrastructure signals | Indirect mechanism, weak within-site temporal variation, cloud/missing data | Gibson et al. (2021); Polinov et al. (2022); Hao and Wang (2023); Wang et al. (2019); Bricongne et al. (2026); Jung (2026)                                     |
| Multimodal learning                | Modality-specific representations and fusion                                                                               | Preservation of heterogeneous structure before prediction                               | Limited direct evidence in oil-price forecasting                            | Baltrušaitis et al. (2019); Arevalo et al. (2017); Gohari et al. (2024); Cong et al. (2022); Fuller et al. (2023); Szwarcman et al. (2026); Ma et al. (2022)   |




### 2.6.2 Research gap

Existing oil-price forecasting studies have made progress in both financial modelling and machine-learning methods, but three linked gaps remain for weekly Brent forecasting with finance, shipping and remote sensing.

First, the incremental value of alternative data is unclear. Few studies jointly report nested increments over a financial baseline and absolute skill against the random-walk benchmark under leakage-safe evaluation. Nested-only comparisons can overstate alternative data; random-walk-only comparisons can hide economically meaningful but weak signals.

Second, fusion architectures lack fair comparison. Most alternative-data applications still reduce shipping and satellite signals to engineered tabular features and concatenate them with financial predictors. Multi-source oil studies rarely compare flat feature fusion against representation-level modality-aware fusion under one shared protocol; common patterns are best-versus-best comparisons across model families, or only one fusion style. What remains unclear is whether any gain comes from the alternative data themselves, or from preserving modality-specific structure before prediction.

Third, attribution often lacks benchmark conditioning. Interpretability is frequently detached from predictive value: heavy attribution for models that fail the relevant benchmark cannot support a narrative that the signals are useful.

Together, these gaps motivate a weekly Brent design that reports both nested and absolute comparisons, pairs flat and modality-aware fusion under one leakage-safe protocol, and concentrates modality-level interpretation on models that already show predictive value against the no-change benchmark.

### 2.6.3 Positioning of this dissertation

This dissertation is positioned as an empirical integration and comparison study rather than a proposal of a new neural architecture. It brings together three strands of literature: the oil-forecasting literature's emphasis on strong baselines and rigorous out-of-sample testing; the alternative-data literature's use of shipping and satellite proxies for economic activity; and the multimodal-learning literature's distinction between flat feature fusion and representation-level modality-aware fusion.

The dissertation therefore asks three linked questions:

- **RQ1:** Do remote-sensing and shipping indicators add incremental out-of-sample value over a financial baseline and the random-walk benchmark?
- **RQ2:** Does modality-aware representation-level fusion outperform flat feature fusion when both use the same underlying data?
- **RQ3:** Can modality-level interpretability reveal which signals the model relies on across different market conditions?

Both claims — that alternative data are useful and that representation-level fusion is superior — are treated as empirical questions to be tested under a consistent evaluation framework, not as assumptions.

---

## References

Adland, R., Jia, H. and Strandenes, S.P. (2017). ‘Are AIS-based trade volume estimates reliable? The case of crude oil exports’, *Maritime Policy & Management*, 44(5), pp. 657–665. doi: 10.1080/03088839.2017.1309470.

Alquist, R., Kilian, L. and Vigfusson, R.J. (2013). ‘Forecasting the price of oil’, in Elliott, G. and Timmermann, A. (eds.) *Handbook of Economic Forecasting*. Vol. 2A. Amsterdam: Elsevier, pp. 427–507. doi: 10.1016/B978-0-444-53683-9.00008-6.

Arevalo, J., Solorio, T., Montes-y-Gómez, M. and González, F.A. (2017). ‘Gated multimodal units for information fusion’, *ICLR 2017 Workshop Track*. Toulon, France, 24–26 April. Available at: https://openreview.net/forum?id=S12_nquOe (Accessed: 1 July 2026).

Arslanalp, S., Marini, M. and Tumbarello, P. (2019). *Big data on vessel traffic: nowcasting trade flows in real time*. IMF Working Paper WP/19/275. Washington, DC: International Monetary Fund. Available at: https://www.imf.org/en/publications/wp/issues/2019/12/13/big-data-on-vessel-traffic-nowcasting-trade-flows-in-real-time-48837 (Accessed: 1 July 2026).

Arslanalp, S., Exton, O., Gao, C., Kamali, P., Saraiva, M., Sozzi, A. and Verschuur, J. (2026). *Nowcasting country-level trade estimates using IMF PortWatch*. IMF Working Paper WP/26/99. Washington, DC: International Monetary Fund. doi: 10.5089/9798229046893.001.

Baltrušaitis, T., Ahuja, C. and Morency, L.-P. (2019). ‘Multimodal machine learning: a survey and taxonomy’, *IEEE Transactions on Pattern Analysis and Machine Intelligence*, 41(2), pp. 423–443. doi: 10.1109/TPAMI.2018.2798607.

Baumeister, C. and Kilian, L. (2015). ‘Forecasting the real price of oil in a changing world: a forecast combination approach’, *Journal of Business & Economic Statistics*, 33(3), pp. 338–351. doi: 10.1080/07350015.2014.949342.

Bricongne, J.-C., Macalos, J., Meunier, B., Milis, J. and Pical, T. (2026). *Can satellites predict oil demand?* ECB Working Paper Series No. 3198. Frankfurt am Main: European Central Bank. Available at: https://www.ecb.europa.eu/pub/pdf/scpwps/ecb.wp3198~e3858c52a3.en.pdf (Accessed: 1 July 2026).

Che, Z., Purushotham, S., Cho, K., Sontag, D. and Liu, Y. (2018). ‘Recurrent neural networks for multivariate time series with missing values’, *Scientific Reports*, 8, 6085. doi: 10.1038/s41598-018-24271-9.

Clark, T.E. and West, K.D. (2007). ‘Approximately normal tests for equal predictive accuracy in nested models’, *Journal of Econometrics*, 138(1), pp. 291–311. doi: 10.1016/j.jeconom.2006.05.023.

Cong, Y., Khanna, S., Meng, C., Liu, P., Rozi, E., He, Y., Burke, M., Lobell, D.B., et al. (2022). ‘SatMAE: pre-training transformers for temporal and multi-spectral satellite imagery’, *Advances in Neural Information Processing Systems*, 35, pp. 197–211.

Costa, A.B.R., Ferreira, P.C.G., Gaglianone, W.P., Guillén, O.T.C., Issler, J.V. and Lin, Y. (2021). ‘Machine learning and oil price point and density forecasting’, *Energy Economics*, 102, 105494. doi: 10.1016/j.eneco.2021.105494.

Diebold, F.X. and Mariano, R.S. (1995). ‘Comparing predictive accuracy’, *Journal of Business & Economic Statistics*, 13(3), pp. 253–263. doi: 10.1080/07350015.1995.10524599.

Foroutan, P. and Lahmiri, S. (2024). ‘Deep learning systems for forecasting the prices of crude oil and precious metals’, *Financial Innovation*, 10, 111. doi: 10.1186/s40854-024-00637-z.

Fuller, A., Millard, K. and Green, J.R. (2023). ‘CROMA: remote sensing representations with contrastive radar-optical masked autoencoders’, *Advances in Neural Information Processing Systems*, 36, pp. 5506–5538.

Gibson, J., Olivia, S., Boe-Gibson, G. and Li, C. (2021). ‘Which night lights data should we use in economics, and where?’, *Journal of Development Economics*, 149, 102602. doi: 10.1016/j.jdeveco.2020.102602.

Gohari, H.E., Dang, X.-H., Shah, S.Y. and Zerfos, P. (2024). ‘Modality-aware transformer for financial time series forecasting’, in *Proceedings of the 5th ACM International Conference on AI in Finance (ICAIF ’24)*. New York: Association for Computing Machinery, pp. 677–685. doi: 10.1145/3677052.3698654.

Hao, X. and Wang, Y. (2023). ‘Cloud cover and expected oil returns’, *Humanities and Social Sciences Communications*, 10, 605. doi: 10.1057/s41599-023-02128-5.

Jung, Y. (2026). ‘Watching trade from space: nowcasting and spatial extrapolation of port-level maritime trade using satellite imagery’, arXiv:2604.15444 [Preprint]. Available at: https://arxiv.org/abs/2604.15444 (Accessed: 1 July 2026).

Kilian, L. (2009). ‘Not all oil price shocks are alike: disentangling demand and supply shocks in the crude oil market’, *American Economic Review*, 99(3), pp. 1053–1069. doi: 10.1257/aer.99.3.1053.

Liang, M., Liu, R.W., Zhan, Y., Li, H., Zhu, F. and Wang, F.-Y. (2022). ‘Fine-grained vessel traffic flow prediction with a spatio-temporal multigraph convolutional network’, *IEEE Transactions on Intelligent Transportation Systems*, 23(12), pp. 23694–23707. doi: 10.1109/TITS.2022.3199160.

Lundberg, S.M. and Lee, S.-I. (2017). ‘A unified approach to interpreting model predictions’, *Advances in Neural Information Processing Systems*, 30, pp. 4765–4774. Available at: https://papers.nips.cc/paper/7062-a-unified-approach-to-interpreting-model-predictions (Accessed: 1 July 2026).

Ma, M., Ren, J., Zhao, L., Testuggine, D. and Peng, X. (2022). ‘Are multimodal transformers robust to missing modality?’, in *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*. New Orleans, LA: IEEE, pp. 18177–18186. doi: 10.1109/CVPR52688.2022.01764.

Mi, J.J., Meng, X., Chen, Y. and Wang, Y. (2022). ‘The impact of the crude oil price on tankers’ port-call features: mining the information in automatic identification system’, *Journal of Marine Science and Engineering*, 10(10), 1559. doi: 10.3390/jmse10101559.

Mi, J.J., Zang, X., Lo, K.L. and Chen, Y. (2023). ‘The nonlinear relationship between oil prices and the number of tankers’ port calls: evidence from AIS data’, *Procedia Computer Science*, 221, pp. 870–877. doi: 10.1016/j.procs.2023.08.063.

Neverova, N., Wolf, C., Taylor, G.W. and Nebout, F. (2016). ‘ModDrop: adaptive multi-modal gesture recognition’, *IEEE Transactions on Pattern Analysis and Machine Intelligence*, 38(8), pp. 1692–1706. doi: 10.1109/TPAMI.2015.2461544.

Ouyang, Q., Sun, T., Xue, Y. and Liu, Z. (2022). ‘Long short-term memory and graph convolution network for forecasting the crude oil traffic flow’, *IEEE Access*, 10, pp. 18922–18932. doi: 10.1109/ACCESS.2022.3150852.

Paolo, F.S., Kroodsma, D., Raynor, J., Hochberg, T., Davis, P., Cleary, J., Marsaglia, L., Orofino, S., et al. (2024). ‘Satellite mapping reveals extensive industrial activity at sea’, *Nature*, 625, pp. 85–91. doi: 10.1038/s41586-023-06825-8.

Polinov, S., Bookman, R. and Levin, N. (2022). ‘A global assessment of night lights as an indicator for shipping activity in anchorage areas’, *Remote Sensing*, 14(5), 1079. doi: 10.3390/rs14051079.

Shukla, S.N. and Marlin, B.M. (2021). ‘Multi-time attention networks for irregularly sampled time series’, *International Conference on Learning Representations (ICLR 2021)*. Online, 3–7 May. Available at: https://openreview.net/forum?id=4c0J6lwQ4_ (Accessed: 1 July 2026).

Simsek, A.I., Bulut, E., Gur, Y.E. and Gültekin Tarla, E. (2024). ‘A novel approach to predict WTI crude spot oil price: LSTM-based feature extraction with Xgboost regressor’, *Energy*, 309, 133102. doi: 10.1016/j.energy.2024.133102.

Szwarcman, D., Roy, S., Fraccaro, P., Gíslason, Þ.E., Blumenstiel, B., Ghosal, R., de Oliveira, P.H., de Sousa Almeida, J.L., et al. (2026). ‘Prithvi-EO-2.0: a versatile multitemporal foundation model for earth observation applications’, *IEEE Transactions on Geoscience and Remote Sensing*, 64, 4400120. doi: 10.1109/TGRS.2025.3642610.

Wang, T., Li, Y., Yu, S. and Liu, Y. (2019). ‘Estimating the volume of oil tanks based on high-resolution remote sensing images’, *Remote Sensing*, 11(7), 793. doi: 10.3390/rs11070793.

Wu, Z., Pan, S., Long, G., Jiang, J. and Zhang, C. (2019). ‘Graph WaveNet for deep spatial-temporal graph modeling’, in *Proceedings of the Twenty-Eighth International Joint Conference on Artificial Intelligence (IJCAI-19)*. Macao, China, 10–16 August. International Joint Conferences on Artificial Intelligence Organization, pp. 1907–1913. doi: 10.24963/ijcai.2019/264.

Yan, Z., Xiao, Y., Cheng, L., Chen, S., Zhou, X., Ruan, X., Li, M., He, R., et al. (2020). ‘Analysis of global marine oil trade based on automatic identification system (AIS) data’, *Journal of Transport Geography*, 83, 102637. doi: 10.1016/j.jtrangeo.2020.102637.

Yılmaz, T.E. and Zehir, C. (2026). ‘Strategic risk based forecasting of Brent crude oil prices: a comparative analysis of econometric and machine learning models’, *Entropy*, 28(5), 539. doi: 10.3390/e28050539.

Zhao, G., Xue, M. and Cheng, L. (2023). ‘A new hybrid model for multi-step WTI futures price forecasting based on self-attention mechanism and spatial–temporal graph neural network’, *Resources Policy*, 85, 103956. doi: 10.1016/j.resourpol.2023.103956.
