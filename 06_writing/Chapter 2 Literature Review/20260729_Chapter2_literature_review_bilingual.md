# Chapter 2 — Literature Review (~4000)

# 第 2 章 — 文献综述 *（约 4000 词）*

Oil-price forecasts matter because price movements feed into inflation, trade balances, producer revenues and energy costs, and therefore into hedging, budgeting and market-risk decisions. This chapter reviews the literatures that define how such forecasts are built and judged—econometric and machine-learning oil-price models, shipping and remote-sensing proxies, multimodal fusion, and evaluation practice—and then states the research gap this dissertation addresses.

油价预测之所以重要，是因为价格变动会传导至通胀、贸易差额、产油国收入与能源成本，并进而影响对冲、预算与市场风险决策。本章综述界定此类预测如何构建与评判的文献——计量与机器学习油价模型、航运与遥感代理、多模态融合以及评估实践——并据此提出本论所针对的研究空白。

## 2.1 Crude-oil price forecasting

## 2.1 原油价格预测

### 2.1.1 Structural accounts and empirical benchmarks

### 2.1.1 结构解释与经验基准

Research on oil-price movements emphasises that similar price changes may arise from economically different sources. Kilian (2009) distinguishes among shocks to global crude-oil production, shocks to aggregate demand for industrial commodities and demand shocks specific to the oil market. The responses of the real price of oil differ across these shocks, as do their relationships with global economic activity and oil production. This structural account has been influential because it separates supply disturbances from changes in general economic activity and oil-market-specific demand, rather than treating every oil-price movement as the outcome of a single process.

油价变动研究强调：相似的价格变化可能来自经济含义不同的冲击来源。Kilian（2009）区分全球原油产量冲击、工业品总需求冲击以及石油市场特有的需求冲击。真实油价对这些冲击的响应不同，它们与全球经济活动和石油产量的关系也不同。这一结构解释之所以有影响，是因为它把供给扰动与一般经济活动变化、以及油市特有需求区分开来，而不是把每一次油价波动都当作同一过程的结果。

A separate literature examines whether these and other economic relationships generate useful forecasts. Alquist, Kilian and Vigfusson (2013) compare a wide range of models with the no-change forecast, which sets the future spot price equal to the current price. Many alternatives fail to improve consistently on this benchmark, particularly in real time and out of sample. They also distinguish population predictability from forecastability: an economic variable may be related to future oil prices without producing lower forecast errors in a finite evaluation sample. Baumeister and Kilian (2015) study forecast combinations across six econometric specifications and find that combinations can produce more stable performance across horizons and periods than individual models. This literature characterises oil-price forecasting as a setting in which simple benchmarks remain competitive and model rankings are often unstable.

另有文献考察这些及其他经济关系能否产生有用的预测。Alquist、Kilian 与 Vigfusson（2013）将大量模型与不变预测比较——不变预测将未来现货价设为当前价格。许多备选模型并不能持续优于该基准，尤其是在实时与样本外情形下。他们还区分总体可预测性与可预报性：某一经济变量可能与未来油价相关，却未必在有限评价样本中带来更低的预测误差。Baumeister 与 Kilian（2015）研究六种计量设定的预测组合，发现组合往往比单一模型在跨期限与跨时期上更稳定。该文献把油价预测刻画为：简单基准仍具竞争力，而模型排序常不稳定。

### 2.1.2 Evidence from machine-learning forecasts

### 2.1.2 机器学习预测证据

Machine learning has widened the set of methods used in oil-price forecasting, particularly in studies with large predictor sets. Costa et al. (2021) evaluate 23 methods using 315 macroeconomic and financial variables in a pseudo-out-of-sample forecasting exercise. They find that no single method dominates across all forecast horizons. Machine-learning methods are competitive at short horizons. At horizons of up to six months, the strongest forecasts include LASSO-based models, oil-futures-based forecasts, a vector error-correction model and the Schwartz–Smith model, while forecast combinations become more relevant at longer horizons. These horizon-dependent results caution against treating XGBoost—or any other algorithm—as a default choice without horizon-specific evaluation. Using monthly Brent returns together with the geopolitical risk index, the VIX and the US ten-year Treasury yield, Yılmaz and Zehir (2026) compare econometric and tree-based models under a rolling-origin design. LightGBM records the most consistent performance across their reported horizons and train–test configurations. Both studies show that rankings vary with the horizon, predictor set and evaluation design.

机器学习扩大了油价预测所用方法集，尤其是在大预测变量集研究中。Costa 等（2021）在伪样本外预测中用 315 个宏观与金融变量评估 23 种方法，发现没有单一方法在所有预测期上都占优。机器学习方法在短预测期上具有竞争力；在至多六个月的预测期上，最强预测包括基于 LASSO 的模型、基于原油期货的预测、向量误差修正模型与 Schwartz–Smith 模型，而更长预测期上预测组合更相关。这些依赖预测期的结果提醒：不宜在缺少期限特异评估时，把 XGBoost——或任何其他算法——当作默认选择。Yılmaz 与 Zehir（2026）以月度 Brent 收益结合地缘政治风险指数、VIX 与美国十年期国债收益率，在滚动起点设计下比较计量与树模型；在其报告的预测期与训练–测试配置上，LightGBM 表现最稳定。两项研究都表明，排序会随预测期、预测变量集与评估设计而变化。

Deep-learning and hybrid studies focus more directly on learned temporal representations. Foroutan and Lahmiri (2024) compare 16 models for next-day WTI (West Texas Intermediate, the main US crude benchmark) and Brent spot-price forecasting. TCN and LightGBM are among the strongest methods in their experiments, with TCN producing the lowest Brent errors across the input lengths considered. Simsek et al. (2024) combine LSTM feature extraction with XGBoost regression and report very high in-sample explanatory power. The two studies differ in their targets, evaluation samples, preprocessing and reported performance measures. More generally, comparisons based on highly persistent price levels are difficult to interpret because small one-step errors may partly reflect the proximity of consecutive prices rather than large gains in predictive content.

深度学习与混合研究更直接关注学得的时间表征。Foroutan 与 Lahmiri（2024）比较 16 种模型用于次日 WTI（西得克萨斯中质原油，美国主要原油基准）与 Brent 现货价格预测。实验中 TCN 与 LightGBM 属于最强方法，且在所考虑的输入长度上 TCN 的 Brent 误差最低。Simsek 等（2024）将 LSTM 特征提取与 XGBoost 回归结合，并报告极高的样本内解释力。两项研究在目标、评价样本、预处理与报告的表现度量上不同。更一般地，基于高度持续的价格水平做比较难以解释，因为较小的一步误差可能部分反映相邻价格接近，而非预测内容的大幅提升。

Graph-based methods represent another development in this literature. Zhao, Xue and Cheng (2023) use self-attention to estimate time-varying interactions among economic and financial variables and apply Graph WaveNet (Wu et al., 2019) to multi-step WTI futures forecasting. The graph in their model represents statistical relationships among predictors rather than a geographic or transport network. The proposed model outperforms the fitted baselines reported in the study, although the evaluation does not include a no-change price forecast.

图方法是该文献的另一发展。Zhao、Xue 与 Cheng（2023）用自注意力估计经济与金融变量间的时变交互，并将 Graph WaveNet（Wu 等，2019）用于多步 WTI 期货预测。其图表示的是预测变量之间的统计关系，而非地理或运输网络。所提模型优于文中报告的拟合基线，但评价未包含不变价格预测。

Across these studies, machine learning is associated with greater flexibility in modelling high-dimensional predictors, nonlinearities and temporal interactions. The empirical results do not, however, establish a stable ranking in which a single class of algorithms consistently dominates econometric models, market-based forecasts or forecast combinations. Differences in targets, horizons, samples and benchmarks remain central to the variation in reported results.

综合这些研究，机器学习与在高维预测变量、非线性与时间交互建模上的更大灵活性相关。然而经验结果并未确立稳定排序，使某一类算法持续主导计量模型、市场型预测或预测组合。目标、预测期、样本与基准的差异，仍是报告结果变异的核心原因。

## 2.2 Shipping activity and oil markets

## 2.2 航运活动与石油市场

### 2.2.1 AIS-based measurement of seaborne trade

### 2.2.1 基于 AIS 的海运贸易测度

Automatic Identification System (AIS) data record vessel identities, positions and movements rather than the quantities or types of cargo carried. A growing literature has nevertheless developed methods for converting these records into estimates of maritime trade. Adland, Jia and Strandenes (2017) compare AIS-derived estimates of seaborne crude-oil exports with customs statistics. Their aggregate estimates align reasonably well with official data, although discrepancies vary across countries and periods because pipelines and transshipment are not fully observed. Yan et al. (2020) combine tanker trajectories with vessel shape, size and draught to estimate voyage-level oil flows. Their estimates for major importers and exporters are strongly correlated with Joint Organisations Data Initiative (JODI) statistics, and their 2017 results identify the Middle East–Malacca–East Asia corridor as the largest route in the global marine oil network.

自动识别系统（AIS）数据记录船舶身份、位置与移动，而非所载货物的数量或类型。尽管如此，不断增长的文献已发展出将这些记录转化为海运贸易估计的方法。Adland、Jia 与 Strandenes（2017）将 AIS 衍生的海运原油出口估计与海关统计比较。其总量估计与官方数据大体一致，但因管线与转运未被充分观测，国别与时期差异仍存在。Yan 等（2020）结合油轮轨迹与船形、尺寸与吃水，估计航次级油流。其对主要进出口国的估计与联合组织数据倡议（JODI）统计高度相关，且 2017 年结果将中东–马六甲–东亚走廊识别为全球海运石油网络中的最大航线。

AIS data have also been used for high-frequency estimates of broader trade activity. Arslanalp, Marini and Tumbarello (2019) construct indicators from filtered port calls and show that they can improve the timeliness of trade monitoring. IMF PortWatch extends this approach by combining vessel movements, port and chokepoint information, ship characteristics and estimated cargo capacity to produce daily indicators of maritime trade (Arslanalp et al., 2026). These studies differ in spatial coverage and commodity detail, but they share a reliance on processed vessel activity as an indirect measure of trade.

AIS 数据也被用于更广义贸易活动的高频估计。Arslanalp、Marini 与 Tumbarello（2019）由过滤后的港口停靠构建指标，并表明可改善贸易监测的及时性。IMF PortWatch 扩展该方法，结合船舶移动、港口与咽喉信息、船舶特征与估计运力，生成日度海运贸易指标（Arslanalp 等，2026）。这些研究在空间覆盖与商品细节上不同，但都依赖加工后的船舶活动作为贸易的间接测度。

### 2.2.2 Measurement limitations and price–shipping relationships

### 2.2.2 测度局限与价格–航运关系

The accuracy of AIS-derived indicators depends on how vessel observations are translated into estimates of activity. Simple vessel counts assign the same weight to ships of different sizes and do not distinguish laden voyages from ballast movements. Vessel capacity and changes in draught provide additional information about likely cargo movement, but draught fields, port calls and vessel classifications may be incomplete or inconsistent (Yan et al., 2020; Arslanalp, Marini and Tumbarello, 2019). Transshipment introduces a further difficulty because a vessel’s observed itinerary may not coincide with the economic origin or final destination of its cargo (Adland, Jia and Strandenes, 2017).

AIS 衍生指标的准确性，取决于如何将船舶观测转化为活动估计。简单船舶计数对不同尺寸船舶赋予相同权重，且不区分满载航次与压载航行。船舶运力与吃水变化可提供货物移动的额外信息，但吃水字段、港口停靠与船舶分类可能不完整或不一致（Yan 等，2020；Arslanalp、Marini 与 Tumbarello，2019）。转运带来进一步困难：船舶观测航程未必与货物的经济起点或最终目的地一致（Adland、Jia 与 Strandenes，2017）。

Coverage is also incomplete. Paolo et al. (2024) combine satellite imagery with vessel-position data and estimate that 21–30% of transport- and energy-vessel activity is absent from public tracking systems. Part of this gap may reflect weak satellite reception rather than deliberate non-broadcasting by vessels. AIS-derived measures consequently observe only part of maritime activity and contain errors arising from reception, classification and cargo attribution.

覆盖也不完整。Paolo 等（2024）结合卫星影像与船舶位置数据，估计 21–30% 的运输与能源船舶活动未出现在公共追踪系统中。部分缺口可能反映卫星接收弱，而非船舶故意不广播。因此，AIS 衍生测度只观测到海上活动的一部分，并含有来自接收、分类与货物归属的误差。

Studies of oil prices and tanker activity add a separate interpretive issue. Mi et al. (2022) examine associations between oil-price changes and tanker port-call frequency, average docking time, total gross tonnage and the number of distinct tankers at ports in major crude-exporting countries. Mi et al. (2023) also model tanker port calls as a response to oil prices and report nonlinear and regionally heterogeneous relationships. Their dependent variable is shipping activity rather than oil prices. The results therefore document price-to-shipping responses and show why contemporaneous correlations between the two series do not establish a single direction of influence.

油价与油轮活动研究另增解释问题。Mi 等（2022）考察油价变化与主要原油出口国港口油轮停靠频率、平均靠泊时间、总吨位及不同油轮数量的关联。Mi 等（2023）也将油轮港口停靠建模为对油价的响应，并报告非线性与区域异质关系。其因变量是航运活动而非油价。因此结果记录的是价格到航运的响应，并说明两序列同期相关为何不能确立单一影响方向。

### 2.2.3 Network models of maritime flows

### 2.2.3 海运流网络模型

Maritime activity is represented in different ways across the vessel-flow literature. Aggregate studies use port calls, vessel counts, capacity-weighted transits or chokepoint volumes, whereas network studies represent ports or regions as nodes and vessel movements as links. The latter representation retains origin–destination relationships and allows traffic at one location to be modelled in relation to activity elsewhere in the network.

船舶流文献以不同方式表示海上活动。总量研究使用港口停靠、船舶计数、运力加权通行或咽喉流量；网络研究则将港口或区域表示为节点、船舶移动表示为边。后者保留起点–终点关系，并使一处交通可相对网络其他位置的活动加以建模。

Ouyang et al. (2022) construct a crude-oil maritime transportation network from vessel trajectories, route information, crude-oil berths and supply–demand links. Their LSTM–GCN forecasts weekly traffic flows at network nodes. Liang et al. (2022) use a spatiotemporal multigraph convolutional network for fine-grained vessel-traffic forecasting, while Zhao et al. (2022) employ a dynamic graph neural network to predict regional vessel inflows, outflows and traffic volumes. These studies address different spatial scales and definitions of traffic, but all treat maritime movement as a relational and time-varying process.

Ouyang 等（2022）由船舶轨迹、航线信息、原油泊位与供需连接构建原油海运网络；其 LSTM–GCN 预测网络节点的周度交通流。Liang 等（2022）用时空多图卷积网络做细粒度船舶交通预测；Zhao 等（2022）用动态图神经网络预测区域船舶流入、流出与交通量。这些研究针对不同空间尺度与交通定义，但都将海上移动视为关系性且时变的过程。

## 2.3 Satellite imagery and remote sensing

## 2.3 卫星影像与遥感

### 2.3.1 Remote sensing as economic measurement

### 2.3.1 作为经济测度的遥感

Remote sensing provides repeated observations of infrastructure, emissions and activity patterns, but different sensors measure different physical phenomena. Night-time lights record emitted radiance; atmospheric observations can capture pollutants such as tropospheric NO₂; cloud products describe observation conditions; and optical or synthetic-aperture radar imagery records surface structure. Economic interpretations are therefore usually tied to a specific mechanism linking the observed signal to an activity of interest.

遥感提供基础设施、排放与活动模式的重复观测，但不同传感器测量不同物理现象。夜光记录发射辐射；大气观测可捕捉对流层 NO₂ 等污染物；云产品描述观测条件；光学或合成孔径雷达影像记录地表结构。因此经济解释通常绑定到连接观测信号与目标活动的具体机制。

Night-time-light studies illustrate how measurement properties vary with spatial and temporal scale. Polinov, Bookman and Levin (2022) find strong cross-sectional associations between VIIRS night-time lights and country-level shipping indicators across hundreds of anchorage areas. They also report that activity is difficult to estimate where only a small number of vessels generate limited light. Gibson et al. (2021) find that VIIRS is more informative than DMSP as a spatial proxy for subnational GDP, particularly at finer spatial levels and in less densely populated areas. Both applications are primarily concerned with differences across places.

夜光研究说明测度性质如何随空间与时间尺度变化。Polinov、Bookman 与 Levin（2022）在数百个锚地发现 VIIRS 夜光与国家级航运指标之间存在强截面关联；并报告在仅少数船舶产生有限灯光处，活动难以估计。Gibson 等（2021）发现，作为次国家 GDP 的空间代理，VIIRS 比 DMSP 更有信息量，尤其在更细空间尺度与人口密度较低地区。两项应用主要关心跨地点差异。

Temporal variation is more difficult to interpret. Small (2021) shows that spatial differences account for most of the observed variation in VIIRS night-time lights and that some month-to-month variation is associated with viewing geometry, atmospheric conditions, background luminance and other features of the imaging process. The literature consequently distinguishes persistent differences in brightness across locations from changes within the same location over time.

时间变异更难解释。Small（2021）表明，空间差异解释了 VIIRS 夜光观测变异的大部分，且部分月际变异与观测几何、大气条件、背景亮度及其他成像过程特征相关。因此文献区分跨地点亮度的持续差异，与同一地点随时间的变化。

### 2.3.2 Applications to oil, trade and infrastructure

### 2.3.2 对石油、贸易与基础设施的应用

Remote-sensing variables have been linked to oil markets through several distinct channels. Hao and Wang (2023) use MODIS (Moderate Resolution Imaging Spectroradiometer) cloud-cover observations over floating-roof tanks in eight major US storage areas. They find that greater cloudiness in one week predicts lower WTI returns in the following week. Their explanation is based on information availability: cloud cover obstructs optical observation of storage tanks and may reduce the inventory information available to market participants. Bricongne et al. (2026) study a different mechanism by using satellite observations of tropospheric NO₂ to nowcast national oil demand. Across advanced and emerging economies, NO₂ improves accuracy relative to autoregressive models and models using conventional predictors, with the largest gains reported for nonlinear models, particularly neural networks.

遥感变量通过若干不同渠道与油市相连。Hao 与 Wang（2023）使用美国八个主要储存区浮顶油罐上空的 MODIS（中分辨率成像光谱仪）云量观测，发现一周云量更高可预测随后一周更低的 WTI 收益。其解释基于信息可得性：云层阻碍对油罐的光学观测，可能减少市场参与者可得的库存信息。Bricongne 等（2026）研究另一机制：用对流层 NO₂ 卫星观测对国家石油需求做现时预测。在发达与新兴经济体中，相对自回归模型与使用常规预测变量的模型，NO₂ 改善精度，非线性模型——尤其神经网络——报告的增益最大。

Other studies use imagery to measure infrastructure or trade rather than prices. Wang et al. (2019) estimate the height, radius and structural volume of oil tanks from high-resolution Gaofen-2 optical imagery. Their method measures storage capacity but not the quantity of oil held in a tank at a particular time. Jung (2026) combines Sentinel-1 synthetic-aperture radar measures, VIIRS night-time lights and port attributes in an XGBoost model to nowcast monthly port-level trade. Satellite variables help to track changes within ports over time, while static port characteristics account for much of the cross-sectional variation. In this application, remote-sensing observations enter the model as engineered numeric features rather than learned image representations.

其他研究用影像测度基础设施或贸易而非价格。Wang 等（2019）由高分辨率高分二号光学影像估计油罐高度、半径与结构体积；其方法测度储存容量，而非某一时刻罐内油量。Jung（2026）在 XGBoost 模型中结合 Sentinel-1 合成孔径雷达测度、VIIRS 夜光与港口属性，对月度港口级贸易做现时预测。卫星变量有助于追踪港口内随时间的变化，而静态港口特征解释截面变异的大部分。该应用中，遥感观测以工程化数值特征进入模型，而非学得的影像表征。

These applications show that the economic content of remote sensing is specific to the observed signal and outcome. Cloud cover has been studied as a constraint on information, NO₂ as an indicator of combustion and demand, high-resolution imagery as a measure of infrastructure, and combined satellite features as indicators of port trade. Evidence obtained for one of these mechanisms does not automatically extend to other sensors, spatial scales or economic outcomes.

这些应用表明，遥感的经济内容特异于观测信号与结果。云量被研究为信息约束，NO₂ 为燃烧与需求指标，高分辨率影像为基础设施测度，组合卫星特征为港口贸易指标。针对某一机制获得的证据，并不自动延伸到其他传感器、空间尺度或经济结果。

## 2.4 Multimodal learning and heterogeneous data

## 2.4 多模态学习与异质数据

### 2.4.1 Multimodal learning and fusion strategies

### 2.4.1 多模态学习与融合策略

Baltrušaitis, Ahuja and Morency (2019) define multimodal machine learning as the processing and relating of information from multiple modalities. Their taxonomy organises the field around five challenges: representation, translation, alignment, fusion and co-learning. Within the fusion literature, studies are also commonly distinguished by the stage at which information is combined. Input- or feature-level fusion combines observed or engineered features before modelling; representation-level fusion combines outputs from modality-specific encoders; and decision-level fusion combines model predictions.

Baltrušaitis、Ahuja 与 Morency（2019）将多模态机器学习定义为处理并关联来自多种模态的信息。其分类围绕五类挑战组织该领域：表示、转换、对齐、融合与协同学习。在融合文献中，研究也常按信息组合发生的阶段区分。输入级或特征级融合在建模前组合观测或工程化特征；表示级融合组合模态专属编码器的输出；决策级融合组合模型预测。

These strategies make different assumptions about the structure retained from each data source. Feature-level fusion represents heterogeneous inputs in a common predictor space and is compatible with many conventional statistical and machine-learning models. Representation-level approaches preserve separate processing streams for at least part of the model. Arevalo et al. (2017) propose the Gated Multimodal Unit, which uses multiplicative gates to combine modality-specific representations in an input-dependent manner. The model was introduced for multimodal classification tasks involving text and images rather than for time-series forecasting.

这些策略对各数据源所保留结构作不同假设。特征级融合在共同预测空间中表示异质输入，并兼容许多传统统计与机器学习模型。表示级方法至少在模型的一部分保留分离的处理流。Arevalo 等（2017）提出门控多模态单元，用乘性门控按输入依赖方式组合模态专属表征。该模型最初用于涉及文本与图像的多模态分类，而非时间序列预测。

Gohari et al. (2024) apply modality-aware modelling in a financial time-series setting. Their model uses separate streams together with intra-modal and inter-modal attention to combine Federal Reserve reports and numerical economic series when forecasting US interest rates. It outperforms several Transformer and time-series baselines across most of the reported settings. The application differs from the original gated-unit study in both data and architecture, but both treat the contribution of a modality as something that can vary across observations rather than as a fixed relationship.

Gohari 等（2024）在金融时序设定中应用模态感知建模。其模型用分离流以及模态内与模态间注意力，在预测美国利率时组合美联储报告与数值经济序列；在多数报告设定上优于若干 Transformer 与时间序列基线。该应用在数据与架构上均不同于原始门控单元研究，但二者都将模态贡献视为可随观测变化，而非固定关系。

### 2.4.2 Representation learning in Earth observation

### 2.4.2 对地观测中的表示学习

Self-supervised learning has expanded the ways in which satellite imagery can be represented. SatMAE (Cong et al., 2022) adapts masked-autoencoder pretraining to temporal and multispectral satellite imagery and incorporates temporal and spectral information into the learning process. Prithvi-EO-2.0 (Szwarcman et al., 2026) is pretrained on global multitemporal samples from the Harmonized Landsat and Sentinel-2 archive and incorporates temporal and location embeddings. Both models use large collections of unlabelled imagery to learn representations that can be transferred to downstream Earth-observation tasks.

自监督学习拓展了卫星影像的表示方式。SatMAE（Cong 等，2022）将掩码自编码器预训练适配到时序与多光谱卫星影像，并把时间与光谱信息纳入学习。Prithvi-EO-2.0（Szwarcman 等，2026）在 Harmonized Landsat and Sentinel-2 档案的全球多时相样本上预训练，并纳入时间与位置嵌入。二者都用大量无标注影像学习可迁移到下游对地观测任务的表征。

CROMA (Fuller, Millard and Green, 2023) focuses on relationships between Earth-observation sensors. It separately encodes spatially and temporally aligned optical and radar observations, applies cross-modal contrastive learning and then produces a joint representation through a fusion encoder. The separate processing streams reflect differences between optical and radar data in channel structure, noise and physical interpretation.

CROMA（Fuller、Millard 与 Green，2023）关注对地观测传感器之间的关系。它分别编码空间与时间对齐的光学与雷达观测，施加跨模态对比学习，再经融合编码器生成联合表征。分离处理流反映光学与雷达数据在通道结构、噪声与物理解释上的差异。

Evaluations of SatMAE, Prithvi-EO-2.0 and CROMA primarily cover land-cover classification, semantic segmentation, disaster mapping and related remote-sensing tasks. Across these applications, pretrained encoders provide transferable representations for a range of downstream Earth-observation tasks.

对 SatMAE、Prithvi-EO-2.0 与 CROMA 的评估主要覆盖土地覆被分类、语义分割、灾害制图及相关遥感任务。在这些应用中，预训练编码器为一系列下游对地观测任务提供可迁移表征。

### 2.4.3 Missing and irregular observations

### 2.4.3 缺失与不规则观测

Multimodal data may be incomplete in more than one sense. In some cases, an entire modality is absent; in others, the modality exists but its observations are irregular, delayed or recorded at a different frequency. The first problem has been studied in the missing-modality literature. Ma et al. (2022) find that multimodal Transformers can be sensitive to the absence of one or more modalities and that robustness varies across fusion strategies and datasets. Neverova et al. (2016) introduce ModDrop, which randomly removes modality channels during training, and report improved robustness in gesture-recognition tasks when inputs are unavailable.

多模态数据可能在不止一种意义上不完整。有时整模态缺失；有时模态存在但其观测不规则、延迟或以不同频率记录。前一问题在缺失模态文献中已有研究。Ma 等（2022）发现多模态 Transformer 可能对一个或多个模态缺失敏感，且稳健性随融合策略与数据集而变。Neverova 等（2016）提出 ModDrop，在训练中随机移除模态通道，并报告在输入不可用时手势识别任务稳健性改善。

A related time-series literature examines irregular observation times. GRU-D (Che et al., 2018) incorporates observation masks and the time elapsed since the previous observation, allowing both missingness and observation age to affect the hidden state. Multi-Time Attention Networks (Shukla and Marlin, 2021) use continuous-time embeddings and attention to represent a variable number of irregularly timed observations. These approaches differ from missing-modality methods because they focus on the timing and availability of observations within a data stream rather than the absence of an entire stream.

相关时间序列文献考察不规则观测时间。GRU-D（Che 等，2018）纳入观测掩码与距上次观测的时间，使缺失与观测年龄均可影响隐状态。Multi-Time Attention Networks（Shukla 与 Marlin，2021）用连续时间嵌入与注意力表示可变数量的不规则定时观测。这些方法不同于缺失模态方法，因其关注数据流内观测的时间与可得性，而非整条流的缺失。

Together, these studies distinguish between two sources of incompleteness that are sometimes conflated in multimodal applications: the absence of a whole modality and irregular sampling within an available modality. They also show that alignment and missingness are modelling problems in their own right, rather than properties resolved automatically by combining additional data sources.

合在一起，这些研究区分多模态应用中有时被混同的两类不完整来源：整模态缺失，与可用模态内的不规则采样。它们也表明，对齐与缺失本身就是建模问题，而非通过堆加更多数据源即可自动解决的属性。

## 2.5 Forecast evaluation and model interpretation

## 2.5 预测评估与模型解释

### 2.5.1 Measuring and comparing predictive accuracy

### 2.5.1 预测精度的度量与比较

Forecast evaluation involves both the choice of an evaluation criterion and the assessment of uncertainty around observed performance differences. Point forecasts are commonly summarised using loss measures such as mean absolute error and root mean squared error. Probabilistic forecasts can instead be evaluated using proper scoring rules, including the Brier score for binary outcomes and log loss (Gneiting and Raftery, 2007). Directional forecasts have also motivated specialised procedures. Pesaran and Timmermann (1992), for example, develop a test of whether predicted and realised directions are independent.

预测评估既涉及评价准则的选择，也涉及对观测表现差异不确定性的评估。点预测常用平均绝对误差与均方根误差等损失度量汇总。概率预测则可用恰当评分规则评价，包括二值结果的 Brier 分数与对数损失（Gneiting 与 Raftery，2007）。方向预测也催生了专门程序；例如 Pesaran 与 Timmermann（1992）提出检验预测方向与实现方向是否独立的检验。

Volatility forecasting raises an additional issue because realised volatility or variance is itself estimated from observed returns. Patton (2011) analyses forecast comparison when the volatility proxy is imperfect and identifies conditions under which particular loss functions preserve the ranking of competing forecasts. QLIKE has consequently become common in variance-forecast evaluation alongside squared- and absolute-error measures.

波动率预测另增问题，因为已实现波动或方差本身由观测收益估计。Patton（2011）分析波动代理不完美时的预测比较，并识别特定损失函数在何种条件下保持竞争预测的排序。因此，QLIKE 与平方误差、绝对误差度量一道，在方差预测评估中变得常见。

Formal comparison tests examine whether observed loss differences are distinguishable from sampling variation. Diebold and Mariano (1995) develop a general test of equal expected predictive loss that permits non-quadratic loss functions and serially correlated loss differentials. Clark and West (2007) consider the more specific case of explicitly nested models under squared-error loss. Their adjustment addresses the tendency of parameter estimation in the larger model to increase its out-of-sample error under the null. The two procedures therefore address related but different forecast-comparison settings.

正式比较检验考察观测损失差异能否与抽样变异区分。Diebold 与 Mariano（1995）提出等期望预测损失的一般检验，允许非二次损失函数与序列相关的损失差。Clark 与 West（2007）考虑平方误差损失下显式嵌套模型的更特殊情形；其调整针对零假设下较大模型参数估计抬高样本外误差的倾向。因此两程序针对相关但不同的预测比较设定。

### 2.5.2 Model interpretation and its limits

### 2.5.2 模型解释及其限度

The increasing use of machine learning in forecasting has been accompanied by greater interest in post-hoc interpretation. SHAP provides an additive decomposition of an individual prediction into feature attributions (Lundberg and Lee, 2017). Aggregating these attributions across observations can produce global summaries of model behaviour, while grouping features can provide higher-level summaries. The resulting values depend, however, on how the absence of a feature is represented and on assumptions about relationships among predictors.

机器学习在预测中的更多使用，伴随对事后解释兴趣的上升。SHAP 将单次预测加性分解为特征归因（Lundberg 与 Lee，2017）。跨观测汇总这些归因可得到模型行为的全局摘要，而特征分组可提供更高层摘要。然而所得值取决于如何表示特征缺失，以及对预测变量间关系的假设。

Feature dependence is particularly important in economic data. Aas, Jullum and Løland (2021) show that independence-based SHAP procedures may evaluate unrealistic combinations of correlated predictors and develop approximations that account for dependence. Their results demonstrate that feature attribution is not invariant to the distributional assumptions used to construct the comparison.

特征依赖在经济数据中尤为重要。Aas、Jullum 与 Løland（2021）表明，基于独立性的 SHAP 程序可能评估相关预测变量的不现实组合，并发展考虑依赖的近似。其结果说明，特征归因并不对用于构建比较的分布假设不变。

Some model architectures also expose internal weights that can be inspected. The Gated Multimodal Unit of Arevalo et al. (2017), for example, produces input-dependent gate values, while attention-based models assign weights across elements of an input or representation. These quantities describe operations inside the fitted model, but their status as explanations is contested. Jain and Wallace (2019) show that substantially different attention patterns can sometimes produce similar predictions and that attention weights need not align with other measures of feature importance. More generally, feature attributions, gates and attention weights describe relationships within a predictive model; they do not by themselves identify causal effects in the process being forecast.

一些模型架构也暴露可检查的内部权重。例如 Arevalo 等（2017）的门控多模态单元产生输入依赖的门控值，而基于注意力的模型对输入或表征的元素赋权。这些量描述拟合模型内部的运算，但其作为解释的地位存在争议。Jain 与 Wallace（2019）表明，显著不同的注意力模式有时可产生相似预测，且注意力权重未必与其他特征重要性度量对齐。更一般地，特征归因、门控与注意力权重描述的是预测模型内的关系；它们本身并不能识别被预测过程中的因果效应。

## 2.6 Synthesis, research gap and positioning

## 2.6 综合、研究空白与定位

### 2.6.1 Synthesis of the literature

### 2.6.1 文献综合

Four conclusions emerge from the review. Oil-price forecasting is difficult because oil prices are highly persistent and the no-change benchmark is strong. Economic and financial predictors are widely used in this literature because they capture persistence, uncertainty, monetary conditions, exchange-rate channels and market expectations. Shipping and remote-sensing data are plausible alternative-data sources, but they are noisy and indirect proxies rather than direct measurements of future prices. Multimodal learning offers tools for preserving modality-specific structure, but these tools have not been systematically tested for commodity-price forecasting with heterogeneous shipping and satellite inputs.

综述得出四点结论。油价预测困难，因为油价高度持续且不变预测基准很强。经济与金融预测变量在该文献中被广泛使用，因其捕捉持续性、不确定性、货币条件、汇率渠道与市场预期。航运与遥感数据是合理的另类数据来源，但它们是嘈杂且间接的代理，而非未来价格的直接量测。多模态学习提供保留模态特有结构的工具，但这些工具尚未在异质航运与卫星输入的大宗商品价格预测中得到系统检验。

The following table summarises the observable signal, economic channel and main limitation of each of the four literatures, with key citations for each strand.

下表汇总四支文献各自的可观测信号、经济渠道与主要局限，并给出各脉络的关键引用。

| Data source / literature | Observable signal | Economic channel | Main limitation | Key references |
| --- | --- | --- | --- | --- |
| Financial and oil-market variables | Lagged price, inventories, production/refinery activity, volatility, GPR, rates, exchange rates, futures/market indicators | Persistence, uncertainty, macro-financial conditions, market expectations | Strong benchmark; difficult to improve upon | Kilian (2009); Alquist, Kilian and Vigfusson (2013); Baumeister and Kilian (2015); Costa et al. (2021); Yılmaz and Zehir (2026) |
| Shipping / AIS / PortWatch | Tanker flows, port calls, chokepoint transits, capacity-weighted activity | Physical trade, supply disruption, congestion, regional flow changes | Directionality, noisy cargo inference, missing AIS activity | Adland, Jia and Strandenes (2017); Yan et al. (2020); Arslanalp, Marini and Tumbarello (2019); Arslanalp et al. (2026); Mi et al. (2022, 2023); Paolo et al. (2024); Ouyang et al. (2022); Liang et al. (2022); Zhao et al. (2022) |
| Remote sensing | Night-time lights, NO₂, cloud cover, site-level imagery or embeddings | Industrial activity, demand conditions, inventory observability, infrastructure signals | Indirect mechanism, weak within-site temporal variation, cloud/missing data | Gibson et al. (2021); Polinov, Bookman and Levin (2022); Small (2021); Hao and Wang (2023); Wang et al. (2019); Bricongne et al. (2026); Jung (2026) |
| Multimodal learning | Modality-specific representations and fusion | Preservation of heterogeneous structure before prediction | Limited direct evidence in oil-price forecasting | Baltrušaitis, Ahuja and Morency (2019); Arevalo et al. (2017); Gohari et al. (2024); Cong et al. (2022); Fuller, Millard and Green (2023); Szwarcman et al. (2026); Ma et al. (2022) |

| 数据来源 / 文献 | 可观测信号 | 经济渠道 | 主要局限 | 关键文献 |
| --- | --- | --- | --- | --- |
| 金融与油市变量 | 滞后价格、库存、产量/炼厂活动、波动率、GPR、利率、汇率、期货/市场指标 | 持续性、不确定性、宏观金融条件、市场预期 | 基准很强，难以进一步改进 | Kilian (2009); Alquist, Kilian and Vigfusson (2013); Baumeister and Kilian (2015); Costa et al. (2021); Yılmaz and Zehir (2026) |
| 航运 / AIS / PortWatch | 油轮流量、港口停靠、咽喉通行、运力加权活动 | 实物贸易、供给扰动、拥堵、区域流量变化 | 方向性、嘈杂的货物推断、AIS 活动缺失 | Adland, Jia and Strandenes (2017); Yan et al. (2020); Arslanalp, Marini and Tumbarello (2019); Arslanalp et al. (2026); Mi et al. (2022, 2023); Paolo et al. (2024); Ouyang et al. (2022); Liang et al. (2022); Zhao et al. (2022) |
| 遥感 | 夜光、NO₂、云量、站点级影像或嵌入 | 工业活动、需求条件、库存可观测性、基础设施信号 | 机制间接、站点内时间变异弱、云/缺失数据 | Gibson et al. (2021); Polinov, Bookman and Levin (2022); Small (2021); Hao and Wang (2023); Wang et al. (2019); Bricongne et al. (2026); Jung (2026) |
| 多模态学习 | 模态专属表征与融合 | 预测前保留异质结构 | 油价预测中直接证据有限 | Baltrušaitis, Ahuja and Morency (2019); Arevalo et al. (2017); Gohari et al. (2024); Cong et al. (2022); Fuller, Millard and Green (2023); Szwarcman et al. (2026); Ma et al. (2022) |

### 2.6.2 Research gap

### 2.6.2 研究空白

Taken together, the literatures reviewed above reveal three unresolved issues at the intersection of oil-price forecasting, alternative data and multimodal learning.

综合以上文献，在油价预测、另类数据与多模态学习交汇处，揭示出三个未决问题。

First, evidence on the predictive value of shipping and remote-sensing data remains fragmented. Oil-price forecasting studies have concentrated mainly on historical prices and macro-financial or oil-market variables. By contrast, AIS and remote-sensing studies have more often examined maritime traffic, trade, oil demand, infrastructure or information availability. These studies show that shipping and satellite observations contain economically relevant information, but there is limited direct evidence on whether they improve one-week-ahead Brent price forecasts beyond established predictors.

第一，航运与遥感数据预测价值的证据仍然碎片化。油价预测研究主要集中在历史价格与宏观金融或油市变量；相比之下，AIS 与遥感研究更常考察海上交通、贸易、石油需求、基础设施或信息可得性。这些研究表明航运与卫星观测含有经济相关信息，但关于它们能否在既有预测变量之上改善提前一周 Brent **价格**预测的直接证据有限。

Second, the literatures represent alternative data in different ways. Many economic applications reduce shipping and satellite observations to engineered numeric indicators. Research on maritime networks and Earth-observation foundation models instead preserves relational or spatial structure through graph-based and pretrained representations. These neighbouring literatures demonstrate that such structures can be modelled, but they do not provide a controlled comparison between engineered features and learned representations in oil-market forecasting. It therefore remains unclear whether retaining modality-specific structure provides predictive information beyond that contained in the underlying data.

第二，文献以不同方式表示另类数据。许多经济应用将航运与卫星观测压缩为工程化数值指标；海运网络与对地观测基础模型研究则通过图与预训练表征保留关系或空间结构。相邻文献表明此类结构可被建模，但未在油市预测中提供工程化特征与学得表征之间的受控比较。因此，保留模态特有结构是否在底层数据所含信息之外提供预测信息，仍不清楚。

Third, predictive performance and model interpretation are not examined within a consistent comparative setting across studies. Published results differ in their forecast targets, horizons, samples, information sets, benchmarks and evaluation procedures. Some comparisons also change the input data and model architecture simultaneously, making it difficult to determine whether an observed improvement comes from the additional information, its representation or the forecasting algorithm. Interpretability analyses similarly use different feature-attribution methods or model-internal weights and are often reported separately from out-of-sample performance. This limits comparison of how models use different information sources across targets and market conditions. Examining predictive performance and model reliance within the same evaluation setting would provide a clearer account of both forecast differences and model behaviour, without treating attribution as evidence of causality.

第三，预测表现与模型解释未在各研究间一致的对照设定中加以考察。已发表结果在预测目标、预测期、样本、信息集、基准与评估程序上不同。一些比较还同时改变输入数据与模型架构，难以判断观测改善来自额外信息、其表征还是预测算法。可解释性分析同样使用不同的特征归因方法或模型内部权重，且常与样本外表现分开报告。这限制了跨目标与市场条件比较模型如何使用不同信息源。在同一评估设定中同时考察预测表现与模型依赖，可更清晰说明预测差异与模型行为，而不把归因当作因果证据。

These gaps concern three related issues: whether alternative data contain incremental predictive information, whether their representation affects forecasting performance, and whether predictive performance and model reliance can be compared within a common evaluation design.

这些空白涉及三个相关问题：另类数据是否含有增量预测信息；其表征是否影响预测表现；以及预测表现与模型依赖能否在共同评估设计下加以比较。

### 2.6.3 Positioning of this dissertation

### 2.6.3 本论定位

These gaps motivate the research questions stated in Section 1.2. This dissertation addresses them through an empirical comparison of one-week-ahead Brent price forecasts under a shared out-of-sample design: financial time series are expanded with shipping and/or remote sensing, and flat feature fusion is paired with modality-aware fusion on matched information sets. Interpretability is used only to describe modality reliance where forecasts already improve on the no-change benchmark, not as causal evidence. Implementation details follow in Chapter 3.

这些空白引出第 1.2 节所述研究问题。本论通过共享样本外设计下提前一周 Brent **价格**预测的实证比较加以回应：在金融时序上扩展航运和/或遥感，并在匹配信息集上配对扁平特征融合与模态感知融合。可解释性仅用于在预测已相对不变基准有改善之处描述模态依赖，不作因果证据。实施细节见第 3 章。

---

## References

## 参考文献

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
