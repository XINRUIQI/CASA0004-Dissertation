# Chapter 2 — Literature Review

# 第 2 章 — 文献综述

Movements in crude oil prices affect inflation, trade balances, public revenues in oil-exporting economies and production costs in energy-intensive sectors. Policymakers, firms and investors therefore have a practical interest in forecasting these movements. The task remains difficult, however, because oil prices respond both to observed changes in production, inventories and transportation and to expectations about future supply and demand. The relative importance of these influences also changes over time, making forecasting relationships unstable. Forecasting oil prices is therefore both of practical interest and difficult.

原油价格变动会影响通胀、贸易收支、石油出口经济体的财政收入，以及能源密集型产业的生产成本。因此，政策制定者、企业和投资者都对预测油价变化具有现实需求。然而，这项任务仍然困难，因为油价既会对产量、库存和运输等实际变化作出反应，也会受到市场对未来供需状况预期的影响。这些因素的相对重要性还会随时间变化，使预测关系缺乏稳定性。因此，预测油价既有现实需求，也很困难。

## 2.1 Crude-oil price forecasting

## 2.1 原油价格预测

### 2.1.1 Structural accounts and empirical benchmarks

### 2.1.1 结构性解释与经验基准

Research on oil-price movements emphasises that similar price changes may arise from economically different sources. Kilian (2009) distinguishes among shocks to global crude-oil production, shocks to aggregate demand for industrial commodities and demand shocks specific to the oil market. The responses of the real price of oil differ across these shocks, as do their relationships with global economic activity and oil production. This structural account has been influential because it separates supply disturbances from changes in general economic activity and oil-market-specific demand, rather than treating every oil-price movement as the outcome of a single process.

油价研究强调，表面上相似的价格变化可能源自经济性质不同的因素。Kilian (2009) 区分了全球原油产量冲击、全球工业品总需求冲击以及石油市场特定需求冲击。实际油价对这些冲击的反应并不相同，它们与全球经济活动及原油产量的关系也存在差异。该结构性解释的影响在于，它将供给扰动、总体经济活动变化和石油市场特定需求区分开来，而不是把所有油价变化视为同一过程的结果。

A separate literature examines whether these and other economic relationships generate useful forecasts. Alquist, Kilian and Vigfusson (2013) compare a wide range of models with the no-change forecast, which sets the future spot price equal to the current price. Many alternatives fail to improve consistently on this benchmark, particularly in real time and out of sample. They also distinguish population predictability from forecastability: an economic variable may be related to future oil prices without producing lower forecast errors in a finite evaluation sample. Baumeister and Kilian (2015) study forecast combinations across six econometric specifications and find that combinations can produce more stable performance across horizons and periods than individual models. This literature characterises oil-price forecasting as a setting in which simple benchmarks remain competitive and model rankings are often unstable.

另一支文献考察这些以及其他经济关系能否产生有效预测。Alquist, Kilian and Vigfusson (2013) 将多种模型与无变化预测进行比较；无变化预测以当前现货价格作为未来价格的预测值。许多替代方法都无法持续优于这一基准，尤其是在实时和样本外评估中。作者还区分了总体可预测性与实际预测能力：某个经济变量可能与未来油价相关，但未必能在有限的评估样本中降低预测误差。Baumeister and Kilian (2015) 比较了六种计量模型的预测组合，发现组合预测在不同期限和时期内可能比单一模型表现得更加稳定。这些研究共同表明，在油价预测中，简单基准长期具有竞争力，而不同模型的排名往往并不稳定。

### 2.1.2 Evidence from machine-learning forecasts

### 2.1.2 机器学习预测的经验证据

Machine learning has widened the set of methods used in oil-price forecasting, particularly in studies with large predictor sets. Costa et al. (2021) evaluate 23 methods using 315 macroeconomic and financial variables in a pseudo-out-of-sample forecasting exercise. They find that no single method dominates across all forecast horizons. Machine-learning methods are competitive at short horizons. At horizons of up to six months, the strongest forecasts include LASSO-based models, oil-futures-based forecasts, a vector error-correction model and the Schwartz–Smith model, while forecast combinations become more relevant at longer horizons. These horizon-dependent results caution against treating XGBoost—or any other algorithm—as a default choice without horizon-specific evaluation. Using monthly Brent returns together with the geopolitical risk index, the VIX and the US ten-year Treasury yield, Yılmaz and Zehir (2026) compare econometric and tree-based models under a rolling-origin design. LightGBM records the most consistent performance across their reported horizons and train–test configurations. Both studies show that rankings vary with the horizon, predictor set and evaluation design.

机器学习拓展了油价预测所使用的方法范围，尤其是在包含大量候选预测变量的研究中。Costa et al. (2021) 使用 315 个宏观经济和金融变量，在伪样本外预测实验中比较了 23 种方法。研究没有发现任何方法能够在所有预测期限上占据优势。机器学习方法在短期内具有竞争力；在不超过六个月的期限上，表现较好的方法还包括基于 LASSO 的模型、基于原油期货的预测、向量误差修正模型和 Schwartz–Smith 模型，而预测组合在更长期限上更加重要。这种期限差异意味着，不应将 XGBoost 或其他任何算法预先视为默认的最优选择。Yılmaz and Zehir (2026) 使用月度 Brent 收益率、地缘政治风险指数、VIX 和美国10年期国债收益率，在滚动起点设计下比较计量模型与树模型。LightGBM 在其报告的不同期限和训练—测试设定中表现最为稳定。两项研究都表明，模型排名会随预测期限、预测变量集合和评估设计而变化。

Deep-learning and hybrid studies focus more directly on learned temporal representations. Foroutan and Lahmiri (2024) compare 16 models for next-day WTI (West Texas Intermediate, the main US crude benchmark) and Brent spot-price forecasting. TCN and LightGBM are among the strongest methods in their experiments, with TCN producing the lowest Brent errors across the input lengths considered. Simsek et al. (2024) combine LSTM feature extraction with XGBoost regression and report very high in-sample explanatory power. The two studies differ in their targets, evaluation samples, preprocessing and reported performance measures. More generally, comparisons based on highly persistent price levels are difficult to interpret because small one-step errors may partly reflect the proximity of consecutive prices rather than large gains in predictive content.

深度学习与混合模型研究更加关注由模型学习的时间序列表示。Foroutan and Lahmiri (2024) 比较了 16 种用于预测下一日 WTI（西得克萨斯中质原油，美国主要原油基准）和 Brent 现货价格的模型。在其实验中，TCN 与 LightGBM 均属于表现较好的方法，其中 TCN 在所考察的不同输入长度下取得了最低的 Brent 预测误差。Simsek et al. (2024) 将 LSTM 特征提取与 XGBoost 回归结合，并报告了很高的样本内解释度。这两项研究在预测目标、评估样本、预处理流程和报告指标方面存在差异。更一般地说，基于高度持久的价格水平进行模型比较并不容易，因为较小的一步预测误差可能部分来自相邻价格本身较为接近，而不完全代表预测信息的大幅增加。

Graph-based methods represent another development in this literature. Zhao, Xue and Cheng (2023) use self-attention to estimate time-varying interactions among economic and financial variables and apply Graph WaveNet (Wu et al., 2019) to multi-step WTI futures forecasting. The graph in their model represents statistical relationships among predictors rather than a geographic or transport network. The proposed model outperforms the fitted baselines reported in the study, although the evaluation does not include a no-change price forecast.

图模型是这一文献中的另一项发展。Zhao, Xue and Cheng (2023) 使用自注意力估计经济与金融变量之间随时间变化的关系，并将 Graph WaveNet（Wu et al., 2019）用于多步 WTI 期货价格预测。该模型中的图表示预测变量之间的统计关系，而不是地理网络或运输网络。该模型优于研究中报告的拟合基准，但其评估没有包含无变化价格预测。

Across these studies, machine learning is associated with greater flexibility in modelling high-dimensional predictors, nonlinearities and temporal interactions. The empirical results do not, however, establish a stable ranking in which a single class of algorithms consistently dominates econometric models, market-based forecasts or forecast combinations. Differences in targets, horizons, samples and benchmarks remain central to the variation in reported results.

综合这些研究，机器学习的主要特点是能够更灵活地处理高维预测变量、非线性关系和时间交互。但现有结果并没有形成一种稳定排名，即某一类算法能够持续优于计量模型、市场型预测或预测组合。不同研究在预测目标、期限、样本和基准方面的差异，仍然是其结果不一致的重要来源。

## 2.2 Shipping activity and oil markets

## 2.2 航运活动与原油市场

### 2.2.1 AIS-based measurement of seaborne trade

### 2.2.1 基于 AIS 的海运贸易测量

Automatic Identification System (AIS) data record vessel identities, positions and movements rather than the quantities or types of cargo carried. A growing literature has nevertheless developed methods for converting these records into estimates of maritime trade. Adland, Jia and Strandenes (2017) compare AIS-derived estimates of seaborne crude-oil exports with customs statistics. Their aggregate estimates align reasonably well with official data, although discrepancies vary across countries and periods because pipelines and transshipment are not fully observed. Yan et al. (2020) combine tanker trajectories with vessel shape, size and draught to estimate voyage-level oil flows. Their estimates for major importers and exporters are strongly correlated with Joint Organisations Data Initiative (JODI) statistics, and their 2017 results identify the Middle East–Malacca–East Asia corridor as the largest route in the global marine oil network.

船舶自动识别系统（AIS）记录的是船舶身份、位置和航行状态，而不是船舶所载货物的数量或类型。尽管如此，越来越多的研究开始将这些记录转化为海运贸易估计。Adland, Jia and Strandenes (2017) 将基于 AIS 推算的海运原油出口量与海关统计进行比较。其总体估计与官方数据较为一致，但由于管道运输和转运无法被完整观测，不同国家和时期的差异程度并不相同。Yan et al. (2020) 结合油轮轨迹、船体形状、尺寸和吃水，估算航程层面的原油流量。其对主要进出口国的估计与 Joint Organisations Data Initiative（JODI，联合组织数据倡议）统计高度相关，2017 年的结果还将“中东—马六甲海峡—东亚”识别为全球海上原油网络中规模最大的航线。

AIS data have also been used for high-frequency estimates of broader trade activity. Arslanalp, Marini and Tumbarello (2019) construct indicators from filtered port calls and show that they can improve the timeliness of trade monitoring. IMF PortWatch extends this approach by combining vessel movements, port and chokepoint information, ship characteristics and estimated cargo capacity to produce daily indicators of maritime trade (Arslanalp et al., 2026). These studies differ in spatial coverage and commodity detail, but they share a reliance on processed vessel activity as an indirect measure of trade.

AIS 数据还被用于构建更广泛贸易活动的高频估计。Arslanalp, Marini and Tumbarello (2019) 根据经过筛选的靠港记录构建指标，并表明这些指标可以提高贸易监测的及时性。IMF PortWatch 在此基础上结合船舶活动、港口与咽喉信息、船舶属性和估计货运能力，生成日度海运贸易指标（Arslanalp et al., 2026）。这些研究在空间覆盖和商品细分程度上有所不同，但都依赖经过处理的船舶活动间接衡量贸易。

### 2.2.2 Measurement limitations and price–shipping relationships

### 2.2.2 测量局限与价格—航运关系

The accuracy of AIS-derived indicators depends on how vessel observations are translated into estimates of activity. Simple vessel counts assign the same weight to ships of different sizes and do not distinguish laden voyages from ballast movements. Vessel capacity and changes in draught provide additional information about likely cargo movement, but draught fields, port calls and vessel classifications may be incomplete or inconsistent (Yan et al., 2020; Arslanalp, Marini and Tumbarello, 2019). Transshipment introduces a further difficulty because a vessel’s observed itinerary may not coincide with the economic origin or final destination of its cargo (Adland, Jia and Strandenes, 2017).

基于 AIS 的指标是否准确，取决于研究如何将船舶观测转化为活动估计。简单的船舶计数会赋予不同规模的船舶相同权重，也不能区分载货航行与压载航行。船舶运力和吃水变化能够提供有关潜在货物流动的额外信息，但吃水字段、靠港记录和船舶分类可能并不完整或一致（Yan et al., 2020；Arslanalp, Marini and Tumbarello, 2019）。转运则带来另一项困难，因为船舶的观测航线未必与货物的经济来源地或最终目的地一致（Adland, Jia and Strandenes, 2017）。

Coverage is also incomplete. Paolo et al. (2024) combine satellite imagery with vessel-position data and estimate that 21–30% of transport- and energy-vessel activity is absent from public tracking systems. Part of this gap may reflect weak satellite reception rather than deliberate non-broadcasting by vessels. AIS-derived measures consequently observe only part of maritime activity and contain errors arising from reception, classification and cargo attribution.

AIS 的覆盖范围也并不完整。Paolo et al. (2024) 将卫星影像与船舶位置数据结合，估计有 21%–30% 的运输与能源船舶活动没有出现在公共追踪系统中。部分缺口可能来自卫星接收能力不足，而不一定是船舶主动停止发送信号。因此，基于 AIS 的指标只能观测到部分海运活动，并会受到信号接收、船舶分类和货物归属误差的影响。

Studies of oil prices and tanker activity add a separate interpretive issue. Mi et al. (2022) examine associations between oil-price changes and tanker port-call frequency, average docking time, total gross tonnage and the number of distinct tankers at ports in major crude-exporting countries. Mi et al. (2023) also model tanker port calls as a response to oil prices and report nonlinear and regionally heterogeneous relationships. Their dependent variable is shipping activity rather than oil prices. The results therefore document price-to-shipping responses and show why contemporaneous correlations between the two series do not establish a single direction of influence.

关于油价与油轮活动的研究还提出了关系方向方面的问题。Mi et al. (2022) 考察了油价变化与主要原油出口国港口的油轮靠港频次、平均靠泊时间、总吨位及不同油轮数量之间的关联。Mi et al. (2023) 同样将油轮靠港活动建模为对油价的响应，并发现这种关系具有非线性和区域异质性。这些研究的因变量是航运活动而不是油价，因此其结果记录的是价格对航运的影响，也说明两类序列之间的同期相关并不能确定单一的影响方向。

### 2.2.3 Network models of maritime flows

### 2.2.3 海运流量的网络模型

Maritime activity is represented in different ways across the vessel-flow literature. Aggregate studies use port calls, vessel counts, capacity-weighted transits or chokepoint volumes, whereas network studies represent ports or regions as nodes and vessel movements as links. The latter representation retains origin–destination relationships and allows traffic at one location to be modelled in relation to activity elsewhere in the network.

船舶流量文献采用了不同方式表示海运活动。总体层面的研究通常使用靠港次数、船舶数量、运力加权通行量或咽喉通行量；网络研究则将港口或区域表示为节点，将船舶移动表示为连接。后一种表示保留了起点—终点关系，并允许研究者结合网络中其他位置的活动来建模某一地点的交通流。

Ouyang et al. (2022) construct a crude-oil maritime transportation network from vessel trajectories, route information, crude-oil berths and supply–demand links. Their LSTM–GCN forecasts weekly traffic flows at network nodes. Liang et al. (2022) use a spatiotemporal multigraph convolutional network for fine-grained vessel-traffic forecasting, while Zhao et al. (2022) employ a dynamic graph neural network to predict regional vessel inflows, outflows and traffic volumes. These studies address different spatial scales and definitions of traffic, but all treat maritime movement as a relational and time-varying process.

Ouyang et al. (2022) 根据船舶轨迹、航线信息、原油泊位和供需联系构建海运原油运输网络，并使用 LSTM–GCN 预测网络节点的每周交通流。Liang et al. (2022) 使用时空多图卷积网络预测细粒度船舶交通，Zhao et al. (2022) 则采用动态图神经网络预测区域船舶流入量、流出量和交通规模。这些研究涉及不同空间尺度和交通流定义，但都将海运活动视为具有关系性且随时间变化的过程。

## 2.3 Satellite imagery and remote sensing

## 2.3 卫星影像与遥感

### 2.3.1 Remote sensing as economic measurement

### 2.3.1 遥感作为经济测量工具

Remote sensing provides repeated observations of infrastructure, emissions and activity patterns, but different sensors measure different physical phenomena. Night-time lights record emitted radiance; atmospheric observations can capture pollutants such as tropospheric NO₂; cloud products describe observation conditions; and optical or synthetic-aperture radar imagery records surface structure. Economic interpretations are therefore usually tied to a specific mechanism linking the observed signal to an activity of interest.

遥感能够重复观测基础设施、排放和活动模式，但不同传感器测量的是不同物理现象。夜间灯光记录地表发出的辐亮度；大气观测可以捕捉对流层 NO₂ 等污染物；云产品描述观测条件；光学或合成孔径雷达影像则记录地表结构。因此，遥感信号的经济解释通常依赖于一种具体机制，将观测到的物理信号与所研究的经济活动联系起来。

Night-time-light studies illustrate how measurement properties vary with spatial and temporal scale. Polinov, Bookman and Levin (2022) find strong cross-sectional associations between VIIRS night-time lights and country-level shipping indicators across hundreds of anchorage areas. They also report that activity is difficult to estimate where only a small number of vessels generate limited light. Gibson et al. (2021) find that VIIRS is more informative than DMSP as a spatial proxy for subnational GDP, particularly at finer spatial levels and in less densely populated areas. Both applications are primarily concerned with differences across places.

夜间灯光研究说明，测量特性会随空间和时间尺度变化。Polinov, Bookman and Levin (2022) 发现，在数百个锚地之间，VIIRS 夜间灯光与国家层面的航运指标存在较强的横截面关联；但当船舶数量较少、产生的灯光有限时，活动规模难以被准确估计。Gibson et al. (2021) 发现，VIIRS 作为次国家尺度 GDP 的空间代理变量比 DMSP 更具信息量，尤其是在更细空间层级和人口密度较低的地区。两项研究主要关注不同地点之间的差异。

Temporal variation is more difficult to interpret. Small (2021) shows that spatial differences account for most of the observed variation in VIIRS night-time lights and that some month-to-month variation is associated with viewing geometry, atmospheric conditions, background luminance and other features of the imaging process. The literature consequently distinguishes persistent differences in brightness across locations from changes within the same location over time.

夜间灯光的时间变化更难解释。Small (2021) 表明，VIIRS 夜间灯光的大部分观测变异来自空间差异，而部分月度变化与观测角度、大气条件、背景亮度及其他成像过程有关。因此，相关文献通常区分不同地点之间长期存在的亮度差异，以及同一地点内部随时间发生的变化。

### 2.3.2 Applications to oil, trade and infrastructure

### 2.3.2 在原油、贸易与基础设施研究中的应用

Remote-sensing variables have been linked to oil markets through several distinct channels. Hao and Wang (2023) use MODIS (Moderate Resolution Imaging Spectroradiometer) cloud-cover observations over floating-roof tanks in eight major US storage areas. They find that greater cloudiness in one week predicts lower WTI returns in the following week. Their explanation is based on information availability: cloud cover obstructs optical observation of storage tanks and may reduce the inventory information available to market participants. Bricongne et al. (2026) study a different mechanism by using satellite observations of tropospheric NO₂ to nowcast national oil demand. Across advanced and emerging economies, NO₂ improves accuracy relative to autoregressive models and models using conventional predictors, with the largest gains reported for nonlinear models, particularly neural networks.

遥感变量通过多种不同渠道与原油市场发生联系。Hao and Wang (2023) 使用美国八个主要储油区浮顶储油罐上空的 MODIS（中分辨率成像光谱仪）云量观测，发现本周云量增加能够预测下一周更低的 WTI 收益率。其解释基于信息可得性：云层会阻碍对储油罐的光学观测，并可能减少市场参与者能够获得的库存信息。Bricongne et al. (2026) 研究了另一种机制，利用卫星观测的对流层 NO₂ 对国家层面的石油需求进行即时预测。在发达经济体和新兴经济体中，NO₂ 相对于自回归模型和使用传统预测变量的模型提高了准确率，其中非线性模型、尤其是神经网络取得的改善最大。

Other studies use imagery to measure infrastructure or trade rather than prices. Wang et al. (2019) estimate the height, radius and structural volume of oil tanks from high-resolution Gaofen-2 optical imagery. Their method measures storage capacity but not the quantity of oil held in a tank at a particular time. Jung (2026) combines Sentinel-1 synthetic-aperture radar measures, VIIRS night-time lights and port attributes in an XGBoost model to nowcast monthly port-level trade. Satellite variables help to track changes within ports over time, while static port characteristics account for much of the cross-sectional variation. In this application, remote-sensing observations enter the model as engineered numeric features rather than learned image representations.

其他研究利用影像测量基础设施或贸易，而不是价格。Wang et al. (2019) 根据高分二号高分辨率光学影像估计储油罐的高度、半径和结构容积。该方法测量的是储存能力，而不是特定时点储油罐内的实际原油数量。Jung (2026) 将 Sentinel-1 合成孔径雷达指标、VIIRS 夜间灯光和港口属性输入 XGBoost，对港口月度贸易进行即时预测。卫星变量有助于追踪港口内部随时间发生的变化，而静态港口特征解释了相当一部分横截面差异。在这一应用中，遥感观测以工程化数值特征的形式进入模型，而不是以学习得到的图像表示进入模型。

These applications show that the economic content of remote sensing is specific to the observed signal and outcome. Cloud cover has been studied as a constraint on information, NO₂ as an indicator of combustion and demand, high-resolution imagery as a measure of infrastructure, and combined satellite features as indicators of port trade. Evidence obtained for one of these mechanisms does not automatically extend to other sensors, spatial scales or economic outcomes.

这些应用表明，遥感数据的经济含义取决于具体观测信号和研究对象。云量被用于研究信息获取限制，NO₂ 被用于反映燃烧与需求，高分辨率影像被用于测量基础设施，多种卫星特征的组合则被用于表征港口贸易。针对其中一种机制获得的证据，不能自动推广至其他传感器、空间尺度或经济结果。

## 2.4 Multimodal learning and heterogeneous data

## 2.4 多模态学习与异质数据

### 2.4.1 Multimodal learning and fusion strategies

### 2.4.1 多模态学习与融合策略

Baltrušaitis, Ahuja and Morency (2019) define multimodal machine learning as the processing and relating of information from multiple modalities. Their taxonomy organises the field around five challenges: representation, translation, alignment, fusion and co-learning. Within the fusion literature, studies are also commonly distinguished by the stage at which information is combined. Input- or feature-level fusion combines observed or engineered features before modelling; representation-level fusion combines outputs from modality-specific encoders; and decision-level fusion combines model predictions.

Baltrušaitis, Ahuja and Morency (2019) 将多模态机器学习定义为对多种模态的信息进行处理并建立联系。他们将该领域概括为五类问题：表示、转换、对齐、融合和协同学习。在融合文献中，研究还常按照信息被组合的阶段加以区分。输入层或特征层融合在建模之前合并观测特征或工程化特征；表示层融合合并模态特定编码器的输出；决策层融合则组合不同模型的预测结果。

These strategies make different assumptions about the structure retained from each data source. Feature-level fusion represents heterogeneous inputs in a common predictor space and is compatible with many conventional statistical and machine-learning models. Representation-level approaches preserve separate processing streams for at least part of the model. Arevalo et al. (2017) propose the Gated Multimodal Unit, which uses multiplicative gates to combine modality-specific representations in an input-dependent manner. The model was introduced for multimodal classification tasks involving text and images rather than for time-series forecasting.

这些策略对于保留各类数据结构的方式具有不同假设。特征层融合将异质输入放入共同的预测变量空间，因此可以与许多常规统计模型和机器学习模型结合。表示层方法则至少在模型的部分阶段保留独立的数据处理路径。Arevalo et al. (2017) 提出门控多模态单元，利用乘法门控以随输入变化的方式组合模态特定表示。该模型最初用于包含文本和图像的多模态分类任务，而不是时间序列预测。

Gohari et al. (2024) apply modality-aware modelling in a financial time-series setting. Their model uses separate streams together with intra-modal and inter-modal attention to combine Federal Reserve reports and numerical economic series when forecasting US interest rates. It outperforms several Transformer and time-series baselines across most of the reported settings. The application differs from the original gated-unit study in both data and architecture, but both treat the contribution of a modality as something that can vary across observations rather than as a fixed relationship.

Gohari et al. (2024) 将模态感知建模应用于金融时间序列。其模型通过独立数据流、模态内注意力和模态间注意力，将美联储报告与数值经济序列结合，用于预测美国利率。在大多数报告设定中，该模型优于若干 Transformer 和时间序列基准。该研究在数据和架构上均不同于最初的门控单元应用，但两者都允许不同模态的贡献随观测变化，而不是将其设定为固定关系。

### 2.4.2 Representation learning in Earth observation

### 2.4.2 地球观测中的表示学习

Self-supervised learning has expanded the ways in which satellite imagery can be represented. SatMAE (Cong et al., 2022) adapts masked-autoencoder pretraining to temporal and multispectral satellite imagery and incorporates temporal and spectral information into the learning process. Prithvi-EO-2.0 (Szwarcman et al., 2026) is pretrained on global multitemporal samples from the Harmonized Landsat and Sentinel-2 archive and incorporates temporal and location embeddings. Both models use large collections of unlabelled imagery to learn representations that can be transferred to downstream Earth-observation tasks.

自监督学习拓展了卫星影像的表示方式。SatMAE (Cong et al., 2022) 将掩码自编码器预训练应用于多时相和多光谱卫星影像，并在学习过程中纳入时间和光谱信息。Prithvi-EO-2.0 (Szwarcman et al., 2026) 使用 Harmonized Landsat and Sentinel-2 数据库中的全球多时相样本进行预训练，并加入时间和位置嵌入。两种模型都利用大规模无标签影像学习可以迁移到下游地球观测任务的表示。

CROMA (Fuller, Millard and Green, 2023) focuses on relationships between Earth-observation sensors. It separately encodes spatially and temporally aligned optical and radar observations, applies cross-modal contrastive learning and then produces a joint representation through a fusion encoder. The separate processing streams reflect differences between optical and radar data in channel structure, noise and physical interpretation.

CROMA (Fuller, Millard and Green, 2023) 关注不同地球观测传感器之间的关系。该模型分别编码在空间和时间上对齐的光学与雷达观测，进行跨模态对比学习，再通过融合编码器生成联合表示。独立的处理路径反映了光学与雷达数据在通道结构、噪声特征和物理含义上的差异。

Evaluations of SatMAE, Prithvi-EO-2.0 and CROMA primarily cover land-cover classification, semantic segmentation, disaster mapping and related remote-sensing tasks. Across these applications, pretrained encoders provide transferable representations for a range of downstream Earth-observation tasks.

SatMAE、Prithvi-EO-2.0 和 CROMA 的评估主要涉及土地覆盖分类、语义分割、灾害制图及相关遥感任务。在这些应用中，预训练编码器为多种下游地球观测任务提供了可迁移的表示。

### 2.4.3 Missing and irregular observations

### 2.4.3 缺失模态与不规则观测

Multimodal data may be incomplete in more than one sense. In some cases, an entire modality is absent; in others, the modality exists but its observations are irregular, delayed or recorded at a different frequency. The first problem has been studied in the missing-modality literature. Ma et al. (2022) find that multimodal Transformers can be sensitive to the absence of one or more modalities and that robustness varies across fusion strategies and datasets. Neverova et al. (2016) introduce ModDrop, which randomly removes modality channels during training, and report improved robustness in gesture-recognition tasks when inputs are unavailable.

多模态数据可能以不同方式呈现不完整性。在某些情况下，整个模态完全缺失；在另一些情况下，模态仍然存在，但观测不规则、存在延迟或采用不同频率记录。第一类问题主要由缺失模态文献研究。Ma et al. (2022) 发现，多模态 Transformer 可能对一个或多个模态的缺失较为敏感，而且不同融合策略和数据集的稳健性并不相同。Neverova et al. (2016) 提出 ModDrop，在训练过程中随机移除模态通道，并在手势识别任务中发现，当输入不可用时，该方法可以提高模型稳健性。

A related time-series literature examines irregular observation times. GRU-D (Che et al., 2018) incorporates observation masks and the time elapsed since the previous observation, allowing both missingness and observation age to affect the hidden state. Multi-Time Attention Networks (Shukla and Marlin, 2021) use continuous-time embeddings and attention to represent a variable number of irregularly timed observations. These approaches differ from missing-modality methods because they focus on the timing and availability of observations within a data stream rather than the absence of an entire stream.

相关的时间序列文献考察了不规则观测时间。GRU-D（Che et al., 2018）纳入观测掩码和距上次观测的时间，使缺失状态与观测时效共同影响隐藏状态。Multi-Time Attention Networks（Shukla and Marlin, 2021）使用连续时间嵌入和注意力机制，表示数量不定且时间不规则的观测。这些方法与缺失模态方法有所不同，因为它们关注的是单个数据流内部观测的时间与可得性，而不是整个数据流的缺失。

Together, these studies distinguish between two sources of incompleteness that are sometimes conflated in multimodal applications: the absence of a whole modality and irregular sampling within an available modality. They also show that alignment and missingness are modelling problems in their own right, rather than properties resolved automatically by combining additional data sources.

综合来看，这些研究区分了多模态应用中有时会被混为一谈的两类不完整性：整个模态的缺失，以及可用模态内部的不规则采样。它们还表明，时间对齐与缺失性本身就是需要处理的建模问题，并不会因为组合更多数据来源而自动消失。

## 2.5 Forecast evaluation and model interpretation

## 2.5 预测评估与模型解释

### 2.5.1 Measuring and comparing predictive accuracy

### 2.5.1 预测精度的测量与比较

Forecast evaluation involves both the choice of an evaluation criterion and the assessment of uncertainty around observed performance differences. Point forecasts are commonly summarised using loss measures such as mean absolute error and root mean squared error. Probabilistic forecasts can instead be evaluated using proper scoring rules, including the Brier score for binary outcomes and log loss (Gneiting and Raftery, 2007). Directional forecasts have also motivated specialised procedures. Pesaran and Timmermann (1992), for example, develop a test of whether predicted and realised directions are independent.

预测评估既涉及评估指标的选择，也涉及对观测表现差异的不确定性进行判断。点预测通常使用平均绝对误差和均方根误差等损失指标进行概括。概率预测则可以采用适当评分规则进行评估，其中包括用于二元结果的 Brier 分数和对数损失（Gneiting and Raftery, 2007）。方向预测还产生了专门的评估方法。例如，Pesaran and Timmermann (1992) 提出了检验预测方向与实际方向是否独立的方法。

Volatility forecasting raises an additional issue because realised volatility or variance is itself estimated from observed returns. Patton (2011) analyses forecast comparison when the volatility proxy is imperfect and identifies conditions under which particular loss functions preserve the ranking of competing forecasts. QLIKE has consequently become common in variance-forecast evaluation alongside squared- and absolute-error measures.

波动率预测还面临一个额外问题，即已实现波动率或方差本身也是根据观测收益估计得到的。Patton (2011) 分析了波动率代理变量不完美时的预测比较，并给出了特定损失函数能够保持竞争预测排序的条件。因此，在方差预测评估中，QLIKE 经常与平方误差和绝对误差指标一同使用。

Formal comparison tests examine whether observed loss differences are distinguishable from sampling variation. Diebold and Mariano (1995) develop a general test of equal expected predictive loss that permits non-quadratic loss functions and serially correlated loss differentials. Clark and West (2007) consider the more specific case of explicitly nested models under squared-error loss. Their adjustment addresses the tendency of parameter estimation in the larger model to increase its out-of-sample error under the null. The two procedures therefore address related but different forecast-comparison settings.

正式的预测比较检验用于判断观测到的损失差异能否与抽样波动区分开来。Diebold and Mariano (1995) 提出了一般性的期望预测损失相等检验，允许使用非平方损失函数，也允许损失差序列存在序列相关。Clark and West (2007) 研究了平方误差损失下明确嵌套模型这一更具体的情形。其调整处理了在原假设成立时，较大模型的参数估计可能增加样本外误差的问题。因此，两种方法针对的是相互关联但并不相同的预测比较情境。

### 2.5.2 Model interpretation and its limits

### 2.5.2 模型解释及其局限

The increasing use of machine learning in forecasting has been accompanied by greater interest in post-hoc interpretation. SHAP provides an additive decomposition of an individual prediction into feature attributions (Lundberg and Lee, 2017). Aggregating these attributions across observations can produce global summaries of model behaviour, while grouping features can provide higher-level summaries. The resulting values depend, however, on how the absence of a feature is represented and on assumptions about relationships among predictors.

随着机器学习在预测中的应用增加，研究者也更加关注事后解释。SHAP 将单个预测分解为可加的特征归因值（Lundberg and Lee, 2017）。在多个观测之间汇总这些归因值，可以形成有关模型行为的总体概括；对特征进行分组，则可以得到更高层级的总结。不过，最终结果取决于如何表示某个特征的“缺失”，以及如何假设预测变量之间的关系。

Feature dependence is particularly important in economic data. Aas, Jullum and Løland (2021) show that independence-based SHAP procedures may evaluate unrealistic combinations of correlated predictors and develop approximations that account for dependence. Their results demonstrate that feature attribution is not invariant to the distributional assumptions used to construct the comparison.

特征依赖关系在经济数据中尤其重要。Aas, Jullum and Løland (2021) 表明，基于独立性假设的 SHAP 方法可能会评估相关预测变量之间不现实的组合，并提出了能够处理变量依赖关系的近似方法。其结果说明，特征归因并不会独立于构建比较时所采用的分布假设。

Some model architectures also expose internal weights that can be inspected. The Gated Multimodal Unit of Arevalo et al. (2017), for example, produces input-dependent gate values, while attention-based models assign weights across elements of an input or representation. These quantities describe operations inside the fitted model, but their status as explanations is contested. Jain and Wallace (2019) show that substantially different attention patterns can sometimes produce similar predictions and that attention weights need not align with other measures of feature importance. More generally, feature attributions, gates and attention weights describe relationships within a predictive model; they do not by themselves identify causal effects in the process being forecast.

部分模型架构还会产生可以被观察的内部权重。例如，Arevalo et al. (2017) 的门控多模态单元会生成随输入变化的门控值，而注意力模型则会在输入或表示的不同元素之间分配权重。这些量描述了拟合模型内部的运算，但其是否能够构成解释仍存在争议。Jain and Wallace (2019) 表明，明显不同的注意力模式有时可以产生相似预测，而且注意力权重未必与其他特征重要性指标一致。更一般地说，特征归因、门控值和注意力权重描述的是预测模型内部的关系，它们本身并不能识别被预测过程中的因果效应。

## 2.6 Synthesis, research gap and positioning

## 2.6 综合、研究空白与本文定位

### 2.6.1 Synthesis of the literature

### 2.6.1 文献综合

Four conclusions emerge from the review. Oil-price forecasting is difficult because oil prices are highly persistent and the no-change benchmark is strong. Economic and financial predictors are widely used in this literature because they capture persistence, uncertainty, monetary conditions, exchange-rate channels and market expectations. Shipping and remote-sensing data are plausible alternative-data sources, but they are noisy and indirect proxies rather than direct measurements of future prices. Multimodal learning offers tools for preserving modality-specific structure, but these tools have not been systematically tested for commodity-price forecasting with heterogeneous shipping and satellite inputs.

综合上述文献，可以得出四点结论。第一，原油价格预测很困难，因为原油价格高度持久，无变化基准很强。第二，经济与金融预测变量在该文献中被广泛使用，因为它们能够捕捉价格持续性、不确定性、货币条件、汇率渠道和市场预期。第三，航运和遥感数据是有潜力的替代数据来源，但它们是噪声较强、机制间接的代理变量，而不是未来价格的直接测量。第四，多模态学习为保留模态特定结构提供了工具，但这些工具尚未在包含异质航运与卫星输入的大宗商品价格预测中得到系统检验。

The following table summarises the observable signal, economic channel and main limitation of each of the four literatures, with key citations for each strand.

下表总结了四类文献各自的可观测信号、经济渠道与主要限制，并为每条文献线索列出关键引用。


| Data source / literature           | Observable signal                                                                                                          | Economic channel                                                                        | Main limitation                                                             | Key references                                                                                                                                                                     |
| ---------------------------------- | -------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------- | --------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Financial and oil-market variables | Lagged price, inventories, production/refinery activity, volatility, GPR, rates, exchange rates, futures/market indicators | Persistence, uncertainty, macro-financial conditions, market expectations               | Strong benchmark; difficult to improve upon                                 | Kilian (2009); Alquist, Kilian and Vigfusson (2013); Baumeister and Kilian (2015); Costa et al. (2021); Yılmaz and Zehir (2026)                                                                   |
| Shipping / AIS / PortWatch         | Tanker flows, port calls, chokepoint transits, capacity-weighted activity                                                  | Physical trade, supply disruption, congestion, regional flow changes                    | Directionality, noisy cargo inference, missing AIS activity                 | Adland, Jia and Strandenes (2017); Yan et al. (2020); Arslanalp, Marini and Tumbarello (2019); Arslanalp et al. (2026); Mi et al. (2022, 2023); Paolo et al. (2024); Ouyang et al. (2022); Liang et al. (2022); Zhao et al. (2022) |
| Remote sensing                     | Night-time lights, NO₂, cloud cover, site-level imagery or embeddings                                                      | Industrial activity, demand conditions, inventory observability, infrastructure signals | Indirect mechanism, weak within-site temporal variation, cloud/missing data | Gibson et al. (2021); Polinov, Bookman and Levin (2022); Small (2021); Hao and Wang (2023); Wang et al. (2019); Bricongne et al. (2026); Jung (2026)                                           |
| Multimodal learning                | Modality-specific representations and fusion                                                                               | Preservation of heterogeneous structure before prediction                               | Limited direct evidence in oil-price forecasting                            | Baltrušaitis, Ahuja and Morency (2019); Arevalo et al. (2017); Gohari et al. (2024); Cong et al. (2022); Fuller, Millard and Green (2023); Szwarcman et al. (2026); Ma et al. (2022)                       |



| 数据来源 / 文献            | 可观测信号                                 | 经济渠道                    | 主要限制                      | 主要文献                                                                                                                                                                               |
| -------------------- | ------------------------------------- | ----------------------- | ------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 金融与原油市场变量            | 滞后价格、库存、产量/炼厂活动、波动率、GPR、利率、汇率、期货/市场指标 | 价格持续性、不确定性、宏观金融条件、市场预期  | 基准很强，难以进一步改进              | Kilian (2009); Alquist, Kilian and Vigfusson (2013); Baumeister and Kilian (2015); Costa et al. (2021); Yılmaz and Zehir (2026)                                                                   |
| 航运 / AIS / PortWatch | 油轮流量、港口停靠、咽喉通行、运力加权活动                 | 实物贸易、供给扰动、拥堵、区域流动变化     | 关系方向、货物推断有噪声、AIS 活动缺失     | Adland, Jia and Strandenes (2017); Yan et al. (2020); Arslanalp, Marini and Tumbarello (2019); Arslanalp et al. (2026); Mi et al. (2022, 2023); Paolo et al. (2024); Ouyang et al. (2022); Liang et al. (2022); Zhao et al. (2022) |
| 遥感                   | 夜间灯光、NO₂、云量、站点影像或嵌入                   | 工业活动、需求条件、库存可观测性、基础设施信号 | 机制间接、站点内部时间变化较弱、云层/缺失数据问题 | Gibson et al. (2021); Polinov, Bookman and Levin (2022); Small (2021); Hao and Wang (2023); Wang et al. (2019); Bricongne et al. (2026); Jung (2026)                                           |
| 多模态学习                | 模态特定表示与融合                             | 在预测前保留异质数据结构            | 在原油价格预测中缺少直接证据            | Baltrušaitis, Ahuja and Morency (2019); Arevalo et al. (2017); Gohari et al. (2024); Cong et al. (2022); Fuller, Millard and Green (2023); Szwarcman et al. (2026); Ma et al. (2022)                       |


### 2.6.2 Research gap

### 2.6.2 研究空白

Taken together, the literatures reviewed above reveal three unresolved issues at the intersection of oil-price forecasting, alternative data and multimodal learning.

综合以上文献，原油价格预测、替代数据与多模态学习的交叉领域仍存在三个尚未解决的问题。

First, evidence on the predictive value of shipping and remote-sensing data remains fragmented. Oil-price forecasting studies have concentrated mainly on historical prices and macro-financial or oil-market variables. By contrast, AIS and remote-sensing studies have more often examined maritime traffic, trade, oil demand, infrastructure or information availability. These studies show that shipping and satellite observations contain economically relevant information, but there is limited direct evidence on whether they improve one-week-ahead Brent return forecasts beyond established predictors.

第一，关于航运和遥感数据预测价值的证据仍较为分散。油价预测研究主要集中于历史价格、宏观金融变量和原油市场变量；相比之下，AIS 与遥感研究更多考察海运流量、贸易、石油需求、基础设施或信息可得性。这些研究表明，航运和卫星观测包含具有经济含义的信息，但它们能否在既有预测变量之外改善提前一周的 Brent 收益率预测，目前仍缺少直接证据。

Second, the literatures represent alternative data in different ways. Many economic applications reduce shipping and satellite observations to engineered numeric indicators. Research on maritime networks and Earth-observation foundation models instead preserves relational or spatial structure through graph-based and pretrained representations. These neighbouring literatures demonstrate that such structures can be modelled, but they do not provide a controlled comparison between engineered features and learned representations in oil-market forecasting. It therefore remains unclear whether retaining modality-specific structure provides predictive information beyond that contained in the underlying data.

第二，不同文献对替代数据采用了不同的表示方式。许多经济应用将航运和卫星观测压缩为工程化数值指标；海运网络和地球观测基础模型研究则通过图表示或预训练表示保留关系结构与空间结构。这些相邻领域的研究表明，此类结构可以被建模，但尚未在油价预测中对工程化特征与学习表示进行受控比较。因此，在底层数据相同的情况下，保留模态特有结构能否提供额外预测信息仍不清楚。

Third, predictive performance and model interpretation are not examined within a consistent comparative setting across studies. Published results differ in their forecast targets, horizons, samples, information sets, benchmarks and evaluation procedures. Some comparisons also change the input data and model architecture simultaneously, making it difficult to determine whether an observed improvement comes from the additional information, its representation or the forecasting algorithm. Interpretability analyses similarly use different feature-attribution methods or model-internal weights and are often reported separately from out-of-sample performance. This limits comparison of how models use different information sources across targets and market conditions. Examining predictive performance and model reliance within the same evaluation setting would provide a clearer account of both forecast differences and model behaviour, without treating attribution as evidence of causality.

第三，现有研究尚未在统一的比较环境下同时考察预测表现与模型解释。不同研究在预测目标、预测期限、样本、信息集、基准模型和评估程序等方面存在差异。有些比较还会同时改变输入数据和模型架构，因此很难判断观测到的改善究竟来自新增信息、信息表示方式，还是预测算法本身。可解释性分析同样采用不同的特征归因方法或模型内部权重，而且经常与样本外预测表现分开报告。这使得研究者难以比较模型在不同预测目标和市场状况下如何使用不同信息来源。在相同评估环境中考察预测表现与模型依赖，可以更清楚地说明预测差异和模型行为，同时避免将归因结果解释为因果证据。

These gaps concern three related issues: whether alternative data contain incremental predictive information, whether their representation affects forecasting performance, and whether predictive performance and model reliance can be compared within a common evaluation design.

这些空白涉及三个相互关联的问题：替代数据是否包含增量预测信息，数据表示方式是否影响预测表现，以及能否在统一评估设计下同时比较预测表现与模型对不同信息来源的依赖。

### 2.6.3 Positioning of this dissertation

### 2.6.3 本文定位

This dissertation addresses these issues through an empirical integration and comparison study rather than by proposing a new neural architecture. It examines one-week-ahead Brent returns within a common out-of-sample design that brings together financial and oil-market variables, shipping activity and remote-sensing observations. The analysis separates changes in the information set from changes in data representation by comparing the incremental contribution of the alternative-data sources and the relative performance of flat and modality-aware fusion. Model-based attributions are then used to examine how reliance on the three information sources varies across market conditions. These attributions are interpreted alongside predictive performance as descriptions of fitted-model behaviour, rather than as evidence of causal effects. Implementation details are presented in Chapter 3.

本文通过一项经验性整合与比较研究回应上述问题，而不是提出新的神经网络架构。研究在统一的样本外设计下考察提前一周的 Brent 收益率，并结合金融与原油市场变量、航运活动和遥感观测。通过比较替代数据来源的增量贡献，以及扁平融合与模态感知融合的相对表现，分析将信息集变化与数据表示方式变化区分开来。随后，模型归因用于考察模型对三类信息来源的依赖如何随市场状况变化。这些归因与预测表现结合解读，用于描述拟合模型的行为，而不是作为因果效应的证据。具体实现将在第 3 章中说明。

The dissertation therefore asks three linked questions:

因此，本文提出三个相互关联的研究问题：

- **RQ1:** Do remote-sensing and shipping indicators add incremental out-of-sample value over a financial baseline and the no-change benchmark?
- **RQ1：** 遥感和航运指标是否能在金融基准模型和无变化基准之上提供样本外增量预测价值？
- **RQ2:** Does modality-aware representation-level fusion outperform flat feature fusion when both use the same underlying data?
- **RQ2：** 在使用相同底层数据的情况下，模态感知的表示层融合是否优于扁平特征融合？
- **RQ3:** Can modality-level interpretability reveal which signals the model relies on across different market conditions?
- **RQ3：** 模态层面的可解释性分析是否能够揭示模型在不同市场条件下依赖哪些信号？

Both claims — that alternative data are useful and that representation-level fusion is superior — are treated as empirical questions to be tested under a consistent evaluation framework, not as assumptions.

替代数据是否有用、表示层融合是否更优——这两个主张都被视为需要在统一评估框架下检验的经验问题，而不是本文的预设前提。

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