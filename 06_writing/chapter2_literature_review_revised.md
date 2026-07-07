# Chapter 2 — Literature Review

*(Revised draft. The review has been narrowed to focus on the literature needed to justify the research gap. Internal paper IDs [Pxxx] are retained for traceability and should be replaced with full bibliographic details before final submission.)*

This chapter reviews the literature needed to position this dissertation. Section 2.1 reviews crude-oil price forecasting and explains why the random-walk benchmark and financial baselines are difficult to beat. Sections 2.2 and 2.3 examine two alternative-data sources — maritime/AIS shipping activity and satellite remote sensing — as economic proxies for oil-market conditions. Section 2.4 reviews multimodal forecasting and the distinction between flat feature fusion and representation-level modality-aware fusion. Section 2.5 reviews forecast evaluation and interpretability standards. Section 2.6 synthesises these strands into the research gap addressed by this dissertation.

## 2.1 Crude-oil price forecasting

### 2.1.1 Econometric foundations and the benchmark problem

A central lesson from the oil-price forecasting literature is that oil prices are difficult to predict out of sample. Kilian (2009) [P052] shows that oil-price movements should be understood through different structural channels, including crude-oil supply shocks, aggregate-demand shocks and oil-specific precautionary-demand shocks. Although this decomposition is not itself a forecasting model, it provides an important principle for predictor selection: useful variables should have a plausible economic connection to supply, demand or uncertainty, rather than being included only because they are available.

Alquist, Kilian and Vigfusson (2013) [P053] provide the key forecasting benchmark for this dissertation. They show that the no-change, or random-walk, forecast is extremely difficult to beat in oil-price forecasting, especially out of sample. Their review also emphasises that in-sample fit does not imply forecasting skill, and that claims of predictability should be tested using real-time data alignment, recursive or rolling evaluation and formal forecast-comparison tests. Baumeister and Kilian (2015) [P054] further show that forecast combinations across different economic mechanisms can be more robust than reliance on a single predictor set. Together, these studies imply that any alternative-data model must be evaluated not only against other machine-learning models, but also against a strong no-change benchmark and an economically informed financial baseline.

### 2.1.2 Machine-learning approaches to oil-price forecasting

Machine-learning studies have introduced tree ensembles, regularised linear models, deep learning and hybrid models into oil-price forecasting. Costa et al. (2021) [P072] compare a broad set of methods over a large macro-financial predictor set and find that useful predictors vary across horizons and over time. Their results suggest that non-linear models such as XGBoost can be strong benchmarks, but they do not dominate uniformly. Yılmaz and Zehir (2026) [P076] show that geopolitical risk, market volatility and interest-rate variables can add value for Brent-return forecasting, with LightGBM outperforming XGBoost in their setting. Foroutan and Lahmiri (2024) [P001] report strong performance from temporal convolutional networks and gradient-boosting models, but their focus on price-level prediction also illustrates a common problem: because oil prices are highly persistent, low one-step price-level errors can partly reflect the fact that \(P_{t+1}\) is usually close to \(P_t\).

Several recent studies use more complex architectures, but their results should be interpreted cautiously. Hybrid designs such as LSTM feature extraction combined with XGBoost have reported very high \(R^2\) values [P004], yet such results may be sensitive to preprocessing choices such as pre-split normalisation. Graph-based oil-price models have also appeared. Zhao, Xue and Cheng (2023) [P063], for example, combine a self-attention-learned dynamic graph with Graph WaveNet (Wu et al., 2019) [P091] for multi-step WTI futures forecasting. However, their graph represents non-Euclidean relations among predictors rather than a physical shipping or geographic network, and the absence of a no-change benchmark makes it difficult to assess whether the model improves over a random walk.

The implication for this dissertation is threefold. First, the price's own lag and the random-walk benchmark must be treated as serious competitors, not weak baselines. Second, financial and oil-market fundamentals provide an economically meaningful baseline before alternative data are added: alongside volatility, geopolitical risk, interest rates, exchange rates and market-based oil indicators, physical supply–demand variables such as inventories, production and refinery activity are natural proxies for Kilian's supply and demand channels. Third, any claim that shipping or remote sensing improves forecasting must be based on incremental out-of-sample value over this baseline, rather than on raw error reduction alone.

## 2.2 Shipping activity and oil markets

### 2.2.1 AIS and maritime activity as trade-flow proxies

Automatic Identification System (AIS) vessel tracking has become an important high-frequency proxy for physical trade. Adland et al. (2017) [P014] validate AIS-derived crude-export estimates against official statistics, while Yan et al. (2020) [P015] show that global marine oil trade is concentrated around major chokepoints such as Hormuz, Malacca and Suez. Arslanalp, Marini and Tumbarello (2019) [P018] and the IMF PortWatch methodology [P070] further demonstrate how vessel movements can be used to nowcast trade activity.

This literature supports the use of shipping data in oil-market analysis, but it also highlights important limitations. Capacity-weighted indicators and draught-change measures are generally more informative than simple vessel counts, because they better approximate cargo movement. At the same time, AIS data require careful filtering to exclude non-trade activity, and moving averages must be constructed without using future observations. These points matter for forecasting because a shipping indicator can only be useful if it is both economically meaningful and available at the forecast origin.

### 2.2.2 Reverse causality and proxy limitations

The relationship between shipping activity and oil prices is not one-directional. Mi et al. (2022) [P016] and Mi, Zang, Lo and Chen (2023) [P017] study how crude-oil prices affect tanker port-call activity, rather than how tanker activity predicts prices. Their findings indicate that the relationship is non-linear and regionally heterogeneous, but also that statistically significant relationships may explain only a small share of variation. This creates an important caution for oil-price forecasting: shipping data may contain useful information about physical trade and congestion, but they may also respond to oil prices rather than lead them.

Chokepoint and port indicators are also imperfect proxies for crude-oil flows. A vessel's previous port is not always the cargo origin, crude oil may be blended or re-sold, and ship-to-ship transfers can obscure the true trade route. Paolo et al. (2024) [P057] further show that a substantial amount of industrial activity at sea is absent from AIS. For these reasons, shipping indicators should be interpreted as noisy proxies for physical-market conditions, not direct measurements of global oil supply.

### 2.2.3 From flat shipping indicators to maritime structure

Most oil-related uses of AIS and PortWatch-style data convert shipping activity into tabular features such as port calls, vessel counts or chokepoint transit volumes. This is useful, but it discards the network structure of maritime activity. Maritime trade is inherently spatial and relational: ports, terminals, chokepoints and routes form a connected transport system. Studies such as Ouyang et al. (2022) [P062] and Liang et al. (2022) [P066] show that graph-based models can learn spatial-temporal structure in vessel-flow prediction. However, these studies forecast traffic flows rather than oil prices. They therefore justify the idea that maritime structure is learnable, but they do not prove that a maritime representation improves Brent-price forecasting.

## 2.3 Satellite imagery and remote sensing

### 2.3.1 Remote sensing as an economic proxy

Remote sensing provides a physical view of economic activity, infrastructure and environmental conditions. In oil-related applications, night-time lights, NO₂, cloud cover and high-resolution imagery have all been used as indirect indicators of economic activity, trade, demand or inventory information. However, the literature is cautious about what these signals can and cannot measure.

Night-time lights are one of the most widely used remote-sensing proxies, but their usefulness depends on scale. Polinov, Bookman and Levin (2022) [P024] find that night-time lights correlate with anchorage activity at a broad cross-sectional scale, yet they do not reliably track tanker activity at a single port. Gibson et al. (2021) [P032] similarly show that VIIRS night-time lights are more suitable than DMSP for facility-scale work, but that night-time lights capture cross-sectional differences more reliably than within-unit temporal variation. This suggests that raw radiance should not be treated as a direct time-series measure of oil activity. Within-site anomalies are more defensible than raw levels when the research objective is forecasting over time.

Other remote-sensing indicators provide different mechanisms. Hao and Wang (2023) [P025] link cloud cover over US storage regions to next-week WTI returns through an information-availability channel: when clouds obstruct optical observation of storage tanks, market uncertainty about inventories may increase. Bricongne et al. (2026) [P069] use tropospheric NO₂ to nowcast national oil demand, but their results also show that the incremental value of remote-sensing variables can weaken inside non-linear models. Wang et al. (2019) [P055] estimate oil-tank structural capacity from high-resolution images, which supports infrastructure measurement but not high-frequency inventory estimation at Sentinel-2 resolution.

### 2.3.2 Limits of direct RS-to-price claims

The remote-sensing literature therefore supports the idea that satellite data may contain oil-relevant information, but it does not support a simple claim that satellite imagery directly predicts oil prices. Most remote-sensing indicators are upstream proxies: they may reflect industrial activity, port activity, storage observability or demand conditions, which may then influence prices through supply-demand expectations. The mechanism is indirect and may vary across locations, sensors and time horizons.

Jung (2026) [P068] provides a useful example of this limitation. The study combines satellite-derived indicators with port attributes to nowcast port-level trade, but the target is trade rather than price, and the model relies on engineered tabular features rather than learned image representations. This is representative of a broader pattern: remote sensing is often converted into flat numeric columns before entering an economic model. Such features may be valuable, but the literature has not yet established whether preserving image or site-level representations adds forecasting value for weekly Brent prices.

## 2.4 Multimodal forecasting and fusion

### 2.4.1 From multi-source data to multimodal learning

A key distinction in this dissertation is between multi-source feature fusion and multimodal representation learning. Baltrušaitis, Ahuja and Morency (2019) [P101] define multimodal learning as involving representation, translation, alignment, fusion and co-learning. Within this taxonomy, fusion can occur at the feature level, decision level or representation level. In much of the oil-price forecasting literature, heterogeneous data are combined through early feature-level fusion: financial indicators, shipping counts and satellite-derived indices are concatenated into a single table and passed to a conventional model. This approach is practical, but it treats all inputs as ordinary numeric predictors and may discard modality-specific structure.

A modality-aware alternative is to encode different data types separately before fusing their representations. At the fusion step itself, learned gating offers a way to weight modalities dynamically rather than treating them equally: Arevalo et al. (2017) [P096] propose gated multimodal units in which the model learns, for each input, how much each modality contributes to the fused representation. Gohari et al. (2024) [P039] provide a relevant precedent in financial time-series forecasting, showing that modality-aware transformers can outperform naïve concatenation when combining different financial information sources. However, their application involves text and numeric data rather than satellite imagery, maritime networks and oil-market variables. The study therefore motivates modality-aware forecasting but does not directly answer whether such fusion is useful for crude-oil price prediction.

### 2.4.2 Representation learning for Earth-observation data

Recent Earth-observation foundation models provide one route to transforming satellite imagery into representations. Models such as SatMAE [P095] and Prithvi-EO-2.0 [P094] use self-supervised pretraining to produce transferable image embeddings. Multisensor models such as CROMA [P105] further show that optical and radar data may benefit from modality-specific encoders before fusion, because the sensors differ in channel structure, noise and physical meaning. These models are relevant because weekly oil-price datasets are usually too small to train an image encoder from scratch.

However, this literature also has clear limits for the present dissertation. Most Earth-observation foundation models are evaluated on land-cover classification, segmentation or related remote-sensing tasks, not on economic forecasting or commodity prices. Their relevance is therefore methodological rather than evidential: they show that satellite imagery can be represented by pretrained encoders, but they do not show that such representations improve Brent forecasting.

### 2.4.3 Missing and asynchronous modalities

A further challenge is that alternative data sources are often incomplete or asynchronous. Optical satellite imagery is affected by cloud cover, radar and optical sensors have different revisit cycles, and shipping or macro-financial data may be released at different frequencies. General multimodal-learning studies show that models can degrade when modalities are missing unless missing-modality training or modality dropout is used [P097; P100], and related work on irregularly sampled time series further motivates the use of masks and time-since-observation signals [P098; P099].

For this dissertation, the importance of this literature is not that any single architecture must be adopted. Rather, it shows that missingness and temporal misalignment are part of the multimodal forecasting problem itself. A fair comparison between flat feature fusion and representation-level fusion must therefore account for the availability, timing and reliability of each modality.

## 2.5 Forecast evaluation and interpretability

### 2.5.1 Forecast comparison

Because the random walk is difficult to beat, lower RMSE or MAE alone is not sufficient evidence of improved forecasting skill. Diebold and Mariano (1995) [P058] provide the standard framework for testing equal predictive accuracy across competing forecasts. For nested models, where one model extends another by adding predictors, Clark and West (2007) provide a more appropriate test under squared-error loss. These tests are especially relevant when evaluating alternative data, because the key question is not whether a large model can reduce error in one sample, but whether an added modality produces a statistically meaningful improvement over a simpler baseline.

The literature therefore implies that alternative-data claims should be evaluated through out-of-sample comparison, horizon-specific performance and formal tests of predictive accuracy. This is particularly important when comparing a financial baseline with models that add shipping, remote sensing or multimodal representations.

### 2.5.2 Interpretability and modality-level explanation

Forecast accuracy tests can show whether a model improves, but they do not explain which signals drive the improvement. SHAP (Lundberg and Lee, 2017) [P059] provides one way to attribute model predictions to features and can be aggregated to the modality level. In this setting, modality-level explanation is important because the dissertation is not only interested in whether the model forecasts better, but also whether financial, shipping and remote-sensing signals contribute in economically interpretable ways.

However, interpretability methods should be used cautiously. SHAP explains model behaviour; it does not establish causal effects. A high attribution for a shipping or satellite feature does not prove that the underlying physical activity caused the oil-price movement. Interpretability is therefore best treated as complementary to forecast-comparison tests: accuracy tests assess whether a modality helps prediction, while attribution helps describe how the trained model uses that modality.

## 2.6 Synthesis, research gap and positioning

### 2.6.1 Synthesis of the literature

The reviewed literature leads to four conclusions. First, oil-price forecasting is difficult because oil prices are highly persistent and the random-walk benchmark is strong. Second, financial variables provide an essential economically informed baseline because they capture persistence, uncertainty, monetary conditions, exchange-rate channels and market expectations. Third, shipping and remote-sensing data are plausible alternative-data sources, but they are noisy and indirect proxies rather than direct measurements of future prices. Fourth, multimodal learning offers tools for preserving modality-specific structure, but these tools have not been systematically tested in the specific setting of weekly Brent forecasting.

The following table summarises the observable signal, economic channel and main limitation of each of the four literatures.

| Data source / literature | Observable signal | Economic channel | Main limitation |
|---|---|---|---|
| Financial and oil-market variables | Lagged price, inventories, production/refinery activity, volatility, GPR, rates, exchange rates, futures/market indicators | Persistence, uncertainty, macro-financial conditions, market expectations | Strong benchmark; difficult to improve upon |
| Shipping / AIS / PortWatch | Tanker flows, port calls, chokepoint transits, capacity-weighted activity | Physical trade, supply disruption, congestion, regional flow changes | Reverse causality, noisy cargo inference, missing AIS activity |
| Remote sensing | Night-time lights, NO₂, cloud cover, site-level imagery or embeddings | Industrial activity, demand conditions, inventory observability, infrastructure signals | Indirect mechanism, weak within-site temporal variation, cloud/missing data |
| Multimodal learning | Modality-specific representations and fusion | Preservation of heterogeneous structure before prediction | Limited direct evidence in oil-price forecasting |

### 2.6.2 Research gap

Existing oil-price forecasting studies have made progress in both financial modelling and machine-learning methods, but most alternative-data applications still reduce heterogeneous data to engineered tabular features. Shipping activity is usually represented as counts, flows or chokepoint indicators; satellite imagery is usually represented through indices such as night-time lights, NO₂ or cloud measures; and these variables are then concatenated with financial predictors in a flat feature table.

This creates a specific gap. To the best of my knowledge, existing studies have not yet systematically tested whether preserving modality-specific representations of financial, shipping and remote-sensing data improves weekly Brent forecasting relative to flat feature fusion under the same leakage-safe out-of-sample evaluation protocol. In other words, the open question is not simply whether more data improve prediction, but whether the way heterogeneous data are represented and fused affects forecasting performance.

### 2.6.3 Positioning of this dissertation

This dissertation is positioned as an empirical integration and comparison study rather than a proposal of a new neural architecture. It brings together three strands of literature: the oil-forecasting literature's emphasis on strong baselines and rigorous out-of-sample testing; the alternative-data literature's use of shipping and satellite proxies for economic activity; and the multimodal-learning literature's distinction between flat feature fusion and representation-level modality-aware fusion.

The dissertation therefore asks three linked questions:

- **RQ1:** Do remote-sensing and shipping indicators add incremental out-of-sample value over a financial baseline and the random-walk benchmark?
- **RQ2:** Does modality-aware representation-level fusion outperform flat feature fusion when both use the same underlying data?
- **RQ3:** Can modality-level interpretability reveal which signals the model relies on across different market conditions?

By framing the contribution in this way, the dissertation avoids claiming that alternative data are automatically useful or that representation-level fusion is necessarily superior. Instead, it treats both claims as empirical questions to be tested under a consistent evaluation framework.

---

## References

*(Internal IDs are retained for traceability. Before submission, replace this list with a complete Harvard/APA-style reference list including author names, year, title, venue, volume, pages and DOI/arXiv/working-paper information where applicable.)*

- Adland, R. et al. (2017). *AIS-based crude-oil export volume estimation.* [P014]
- Alquist, R., Kilian, L., & Vigfusson, R. J. (2013). *Forecasting the Price of Oil.* Handbook of Economic Forecasting, 2A. [P053]
- Arevalo, J., Solorio, T., Montes-y-Gómez, M., & González, F. A. (2017). *Gated Multimodal Units for Information Fusion.* ICLR Workshop. [P096]
- Arslanalp, S., Marini, M., & Tumbarello, P. (2019). *Big Data on Vessel Traffic: Nowcasting Trade Flows in Real Time.* IMF WP/19/275. [P018]
- Baltrušaitis, T., Ahuja, C., & Morency, L.-P. (2019). *Multimodal Machine Learning: A Survey and Taxonomy.* IEEE TPAMI 41(2). [P101]
- Baumeister, C., & Kilian, L. (2015). *Forecasting the Real Price of Oil in a Changing World: A Forecast Combination Approach.* [P054]
- Bricongne, J.-C., Macalos, J.-P., Meunier, B., et al. (2026). *Can Satellites Predict Oil Demand?* ECB WP 3198. [P069]
- Che, Z., Purushotham, S., Cho, K., et al. (2018). *Recurrent Neural Networks for Multivariate Time Series with Missing Values (GRU-D).* Scientific Reports 8:6085. [P098]
- Clark, T. E., & West, K. D. (2007). *Approximately Normal Tests for Equal Predictive Accuracy in Nested Models.* Journal of Econometrics 138(1), 291–311.
- Cong, Y., Khanna, S., Meng, C., et al. (2022). *SatMAE: Pre-training Transformers for Temporal and Multi-Spectral Satellite Imagery.* NeurIPS. [P095]
- Costa, A. B. et al. (2021). *Machine Learning and Oil Price Point and Density Forecasting.* [P072]
- Diebold, F. X., & Mariano, R. S. (1995). *Comparing Predictive Accuracy.* Journal of Business & Economic Statistics 13(3), 253–263. [P058]
- Foroutan, P., & Lahmiri, S. (2024). *Deep learning systems for forecasting the prices of crude oil and precious metals.* Financial Innovation 10:111. [P001]
- Fuller, A., Millard, K., & Green, J. (2023). *CROMA: Remote Sensing Representations with Contrastive Radar-Optical Masked Autoencoders.* NeurIPS. [P105]
- Gibson, J., Olivia, S., Boe-Gibson, G., & Li, C. (2021). *Which Night Lights Data Should We Use in Economics, and Where?* Journal of Development Economics 149. [P032]
- Gohari, H. E., Dang, X.-H., Shah, S. Y., & Zerfos, P. (2024). *Modality-aware Transformer for Financial Time Series Forecasting.* ICAIF '24. [P039]
- Hao, J., & Wang, Y. (2023). *Cloud Cover and Expected Oil Returns.* Humanities and Social Sciences Communications 10:605. [P025]
- Jung, Y. (2026). *Watching Trade from Space: Nowcasting and Spatial Extrapolation of Port-Level Maritime Trade Using Satellite Imagery.* arXiv:2604.15444. [P068]
- Kilian, L. (2009). *Not All Oil Price Shocks Are Alike.* American Economic Review. [P052]
- Liang et al. (2022). *Fine-Grained Vessel Traffic Flow Prediction with a Spatio-Temporal Multi-Graph Convolutional Network (STMGCN).* [P066]
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
- Wang, T. et al. (2019). *Estimating the Volume of Oil Tanks Based on High-Resolution Remote Sensing Images.* Remote Sensing 11(7). [P055]
- Wu, Z., Pan, S., Long, G., Jiang, J., & Zhang, C. (2019). *Graph WaveNet for Deep Spatial-Temporal Graph Modeling.* IJCAI. [P091]
- Yan, Z. et al. (2020). *Analysis of global marine oil trade from AIS.* [P015]
- Yılmaz and Zehir (2026). *Strategic-Risk-Based Forecasting of Brent Crude Oil Prices.* Entropy 28(5). [P076]
- Zhao, G., Xue, M., & Cheng, L. (2023). *A New Hybrid Model for Multi-Step WTI Futures Price Forecasting Based on Self-Attention Mechanism and Spatial–Temporal Graph Neural Network (GWNet-Attn).* Resources Policy 85:103956. [P063]
