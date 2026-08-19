# Chapter 2 — Literature Review

# 第 2 章 — 文献综述

This chapter reviews the main bodies of literature related to the study. It first examines the economic drivers and empirical benchmarks of crude-oil price forecasting, followed by the application of machine-learning methods. It then reviews how shipping activity reflects oil-market conditions and how satellite remote sensing provides oil-market information. Next, it introduces multimodal learning and fusion methods. Finally, it synthesises the four literatures, identifies the research gaps and positions this dissertation within the existing literature.

本章综述与本研究相关的主要文献。首先介绍原油价格预测的经济驱动因素和经验预测基准，随后讨论机器学习方法在油价预测中的应用。接下来介绍航运活动如何反映油市状况，以及卫星遥感如何提供与油市相关的信息。随后介绍多模态学习及其融合方法。最后，本章综合四支文献，提出研究空白，并说明本文在现有文献中的定位。

## 2.1 Crude-oil price drivers and forecasting benchmarks

## 2.1 原油价格驱动因素与预测基准

Research on oil-price movements shows that similar price changes can arise from economically different sources. Kilian (2009) distinguishes among shocks to global crude-oil production, shocks to aggregate demand for industrial commodities and demand shocks specific to the oil market. Oil prices respond differently to these shocks, and each shock has a different relationship with global economic activity and oil production. This distinction provides an economic basis for using variables related to supply, demand, market expectations and macroeconomic conditions in oil-price forecasting.

油价变动研究表明，相似的价格变化可能来自经济含义不同的冲击。Kilian（2009）区分全球原油产量冲击、工业品总需求冲击，以及石油市场特有的需求冲击。油价对这些冲击的反应不同，各类冲击与全球经济活动和原油产量的关系也不相同。这一区分为在油价预测中使用供给、需求、市场预期和宏观经济条件等变量提供了经济理论基础。

A separate literature examines whether these economic relationships produce accurate forecasts. Alquist, Kilian and Vigfusson (2013) compare a wide range of forecasting models with the no-change forecast, which sets the future spot price equal to the current price. Many alternative models fail to outperform this simple benchmark consistently, particularly in real-time and out-of-sample evaluations. They also distinguish economic predictability from practical forecastability. A variable may have an economic relationship with future oil prices without reducing forecast errors in a finite out-of-sample period. Baumeister and Kilian (2015) find that combining forecasts from several econometric models can produce more stable results across periods and forecast horizons than relying on a single model. These studies highlight the importance of the no-change forecast as a benchmark and show that model performance can vary substantially across evaluation periods and forecast horizons.

另一类文献考察这些经济关系能否产生准确的预测。Alquist、Kilian 与 Vigfusson（2013）将多种预测模型与不变预测进行比较。不变预测将未来现货价格设为当前价格。许多备选模型无法持续优于这一简单基准，尤其是在实时和样本外评价中。他们还区分经济上的可预测性与实际的可预报性。某个变量可能与未来油价存在经济关系，但未必能在有限的样本外时期降低预测误差。Baumeister 与 Kilian（2015）发现，组合多个计量模型的预测，可能比依赖单一模型在不同阶段和预测期上取得更稳定的结果。这些研究强调不变预测作为评价基准的重要性，并表明模型表现可能随评价时期和预测期发生较大变化。

## 2.2 Machine learning in crude-oil price forecasting

## 2.2 机器学习在原油价格预测中的应用

Machine learning has expanded the range of methods used in oil-price forecasting. These methods can process large predictor sets and capture nonlinear relationships and interactions. Costa et al. (2021) evaluate 23 methods using 315 macroeconomic and financial variables. They find that no single method performs best at every forecast horizon. Machine-learning methods are competitive at short horizons, but econometric, market-based and combined forecasts also perform strongly in some settings. Yılmaz and Zehir (2026) compare econometric and tree-based models for Brent returns using macro-financial and geopolitical variables. Light Gradient Boosting Machine (LightGBM) produces the most consistent results across their reported settings, but the broader comparison again shows that performance depends on the forecast horizon, predictor set and evaluation design.

机器学习扩大了原油价格预测使用的方法范围。这些方法能够处理较大的预测变量集合，并刻画非线性关系与变量交互。Costa 等（2021）使用 315 个宏观经济和金融变量，比较 23 种预测方法。他们发现，没有一种方法在所有预测期上都表现最佳。机器学习方法在短预测期上具有竞争力，但计量模型、市场型预测和预测组合在部分设定下同样表现较强。Yılmaz 与 Zehir（2026）使用宏观金融和地缘政治变量，比较用于 Brent 收益预测的计量模型和树模型。轻量梯度提升机（Light Gradient Boosting Machine，LightGBM）在其报告的设定中表现最稳定，但整体比较再次表明，模型表现取决于预测期、预测变量集合和评价设计。

Deep-learning studies focus more directly on learning temporal representations. Foroutan and Lahmiri (2024) compare a range of methods for next-day WTI and Brent spot-price forecasting and report strong performance from temporal convolutional networks and LightGBM. Simsek et al. (2024) combine Long Short-Term Memory (LSTM) feature extraction with Extreme Gradient Boosting (XGBoost) for WTI price prediction. Graph-based methods have also been introduced. Zhao, Xue and Cheng (2023), for example, model time-varying relationships among economic and financial variables and use a spatial–temporal graph neural network to forecast WTI futures. Their graph represents statistical relationships among predictors rather than a physical transportation network.

深度学习研究更直接地关注时间表征的学习。Foroutan 与 Lahmiri（2024）比较多种用于次日 WTI 和 Brent 现货价格预测的方法，并报告时序卷积网络和 LightGBM 具有较强表现。Simsek 等（2024）将基于长短期记忆网络（Long Short-Term Memory，LSTM）的特征提取与极端梯度提升（Extreme Gradient Boosting，XGBoost）结合，用于预测 WTI 价格。图模型也开始用于油价预测。例如，Zhao、Xue 与 Cheng（2023）对经济和金融变量之间的时变关系进行建模，并使用时空图神经网络预测 WTI 期货。其图结构表示预测变量之间的统计关系，而不是实际的运输网络。

Overall, machine learning provides greater flexibility for modelling high-dimensional data, nonlinearities and temporal interactions. However, the literature does not establish that this flexibility consistently translates into superior forecasts. Reported performance varies with the target, forecast horizon, sample, information set, benchmark and evaluation design. Taken together, the evidence for a general out-of-sample forecasting advantage of machine learning remains mixed.

总体而言，机器学习在建模高维数据、非线性关系和时间交互作用方面具有更大的灵活性。然而，现有文献尚未证明这种灵活性能够持续转化为更优的预测表现。已有研究所报告的预测性能会因预测目标、预测期限、样本、信息集、基准模型以及评估设计的不同而有所差异。综合来看，目前关于机器学习是否具有普遍的样本外预测优势，相关证据仍然不一致。

## 2.3 Shipping activity as an oil-market signal

## 2.3 作为油市信号的航运活动

A large share of international crude-oil trade is transported by sea. Shipping activity can therefore provide information about physical oil flows, regional supply conditions, congestion and disruptions at ports or major chokepoints. Automatic Identification System (AIS) data record vessel identities, positions and movements. Although they do not directly record cargo quantities, processed AIS observations can be used to estimate tanker movements and maritime trade.

国际原油贸易中有很大一部分通过海运完成。因此，航运活动可以提供有关实物原油流动、区域供给状况、港口拥堵以及主要航运咽喉中断的信息。船舶自动识别系统（AIS）数据记录船舶身份、位置和移动。尽管 AIS 不直接记录货物数量，但经过处理的 AIS 观测可以用于估计油轮活动和海运贸易。

Adland, Jia and Strandenes (2017) compare AIS-based estimates of seaborne crude-oil exports with customs statistics and find that aggregate estimates are broadly consistent with official data. Yan et al. (2020) combine tanker trajectories with vessel characteristics and draught information to estimate voyage-level oil flows. Their estimates for major oil-importing and oil-exporting countries are strongly correlated with Joint Organisations Data Initiative (JODI) statistics. These studies provide evidence that vessel movements can serve as proxies for the physical transportation of crude oil. 

Adland、Jia 与 Strandenes（2017）将基于 AIS 的海运原油出口估计与海关统计进行比较，发现总量估计与官方数据大体一致。Yan 等（2020）结合油轮轨迹、船舶特征和吃水信息，估计航次层面的石油流量。他们对主要石油进口国和出口国的估计与联合组织数据倡议（Joint Organisations Data Initiative, JODI）统计高度相关。这些研究表明，船舶移动可以作为原油实物运输的代理变量。

Beyond their consistency with official statistics, AIS-based indicators can also improve the timeliness of trade measurement. Arslanalp, Marini and Tumbarello (2019) construct high-frequency trade indicators from vessel movements and port calls. Arslanalp et al. (2026) describe how the International Monetary Fund (IMF) PortWatch platform extends this approach by combining information on vessel activity, ports, chokepoints, ship characteristics and estimated carrying capacity to produce daily indicators of maritime trade. Compared with conventional trade statistics, these indicators can capture changes in maritime activity with shorter reporting delays.

除与官方统计大体一致外，基于 AIS 的指标还可以提高贸易测量的及时性。Arslanalp、Marini 与 Tumbarello（2019）利用船舶移动和港口停靠构建高频贸易指标。Arslanalp 等（2026）说明国际货币基金组织（International Monetary Fund, IMF）的 PortWatch 平台如何在此方法上扩展，结合船舶活动、港口、航运咽喉、船舶特征和估计运力，生成日度海运贸易指标。与传统贸易统计相比，这些指标能够以更短的报告延迟捕捉海运活动变化。

However, more timely measurement of shipping activity does not necessarily imply predictive value for oil prices, because the relationship between shipping activity and future oil prices may not be one-directional. Mi et al. (2022) identify relationships between oil-price changes and tanker port-call frequency, docking time, gross tonnage and the number of tankers at ports in major crude-exporting countries. Mi et al. (2023) also find nonlinear and regionally heterogeneous relationships between oil prices and tanker port calls. In both studies, shipping activity responds to oil prices. Their results therefore show that contemporaneous associations do not establish that vessel activity leads future price movements.

然而，更及时地测量航运活动并不必然意味着其对油价具有预测价值，因为航运活动与未来油价之间的关系可能并非单向。Mi 等（2022）发现，油价变化与主要原油出口国港口的油轮停靠频率、靠泊时间、总吨位和油轮数量存在关联。Mi 等（2023）也发现，油价与油轮港口停靠数量之间存在非线性和区域异质关系。在这两项研究中，航运活动是对油价变化的反应。因此，这些结果表明，同期相关关系不能证明船舶活动能够领先未来油价。

AIS-based measures are also indirect and incomplete. Cargo type and quantity must often be inferred from vessel characteristics, routes or draught, and some vessel activity is not observed in public tracking systems. Paolo et al. (2024) show that a meaningful share of transport- and energy-vessel activity is absent from public vessel-position data. These limitations do not make AIS data unusable, but they mean that shipping variables should be treated as noisy proxies for physical trade rather than direct measurements of oil supply.

AIS 测度还具有间接性和不完整性。货物类型和数量通常需要根据船舶特征、航线或吃水进行推断，部分船舶活动也不会出现在公共追踪系统中。Paolo 等（2024）表明，公共船舶位置数据遗漏了一定比例的运输和能源船舶活动。这些局限并不意味着 AIS 数据无法使用，但意味着航运变量应被视为实物贸易的含噪代理，而不是原油供给的直接测量。

The existing literature establishes that shipping data can measure changes in maritime trade and crude-oil transportation. Direct evidence that these data improve short-term Brent price forecasts remains limited.

现有文献证明，航运数据可以用于测量海运贸易和原油运输的变化。这些数据能否改善短期 Brent 价格预测，目前仍缺乏充分的直接证据。

## 2.4 Maritime networks and graph-based modelling

## 2.4 海运网络与图模型

Maritime-network studies provide a way to preserve relationships among locations. Aggregate indicators summarise port calls, vessel counts or chokepoint traffic, whereas network models represent ports or regions as nodes and vessel movements as links. This representation retains connections between locations and allows activity at one part of the network to be modelled in relation to activity elsewhere.

海运网络研究提供了保留不同地点之间关系的方法。总量指标通常汇总港口停靠、船舶数量或航运咽喉流量，而网络模型将港口或地区表示为节点，将船舶移动表示为连接。这种表示方式能够保留不同地点之间的联系，并使网络中某一部分的活动可以结合其他地点的活动进行建模。

Ouyang et al. (2022) construct a crude-oil transportation network and use an LSTM–GCN model to forecast weekly traffic at network nodes. Liang et al. (2022) use a spatiotemporal multigraph convolutional network for vessel-traffic forecasting, while Zhao et al. (2022) use a dynamic graph neural network to predict regional vessel inflows, outflows and traffic volumes. These studies show that maritime activity is relational and changes over time.

Ouyang 等（2022）构建原油海运网络，并使用 LSTM–GCN 预测网络节点的周度交通流。Liang 等（2022）使用时空多图卷积网络预测船舶交通，Zhao 等（2022）则使用动态图神经网络预测区域船舶流入、流出和交通量。这些研究表明，海上活动具有关系结构并随时间变化。

The literature shows that network representations can capture relationships between ports and routes. However, these methods have mainly been used to forecast shipping activity itself. There is limited direct evidence on whether graph-based representations of maritime networks improve oil-price forecasts.

现有文献表明，网络表示能够刻画港口和航线之间的关系。然而，这些方法主要用于预测航运活动本身。关于海运网络的图表示能否改善油价预测，目前仍缺乏直接证据。

## 2.5 Remote sensing as an oil-market signal

## 2.5 作为油市信号的遥感数据

Remote sensing provides repeated observations of oil-related infrastructure, industrial activity and maritime locations. Satellite observations may therefore contain information about oil demand, storage, port activity and trade. However, their economic meaning depends on the physical signal being measured and the mechanism connecting that signal to the oil market.

遥感能够重复观测石油相关基础设施、工业活动和海上地点。因此，卫星观测可能包含有关石油需求、储存、港口活动和贸易的信息。不过，遥感数据的经济含义取决于其测量的物理信号，以及该信号与石油市场之间的具体联系。

Several studies use satellite observations to measure oil-related infrastructure and examine changing oil-market conditions. Wang et al. (2019) estimate the dimensions and structural capacity of oil tanks from high-resolution imagery, showing that physical storage infrastructure can be quantified remotely. Hao and Wang (2023) examine cloud cover above floating-roof oil tanks in major US storage areas. Because roof height varies with stored volume, tank shadows in clear-sky satellite imagery can reveal inventories before official EIA releases, whereas cloud cover obscures this signal and increases information uncertainty. The authors argue that this uncertainty may encourage precautionary inventory holdings. Consistent with this mechanism, they find that greater cloud cover is followed by higher inventories and lower WTI returns in the following week. They interpret this as an information effect rather than a direct weather effect on oil supply or demand. Bricongne et al. (2026) use satellite observations of tropospheric NO₂, a short-lived pollutant emitted primarily by fossil-fuel combustion, to nowcast national oil demand. They find that daily NO₂ data improve nowcasting accuracy relative to models using conventional predictors, showing that satellite observations can provide timely information about changes in oil consumption.

部分研究使用卫星观测测量石油相关基础设施，并考察油市状况的变化。Wang 等（2019）根据高分辨率影像估计油罐的尺寸和结构容量，表明实体储存基础设施可以通过遥感加以量化。Hao 与 Wang（2023）考察美国主要储存区浮顶油罐上空的云量。由于浮顶高度随储存量变化，晴空卫星影像中的油罐阴影可在官方 EIA 数据公布前揭示库存，而云层会遮蔽这一信号并增加信息不确定性。作者认为，这种不确定性可能促使企业持有预防性库存。与这一机制一致，他们发现云量较高之后库存更高、下一周 WTI 收益更低。他们将这一关系解释为信息效应，而非天气对石油供给或需求的直接影响。Bricongne 等（2026）使用卫星观测到的对流层 NO₂——一种主要由化石燃料燃烧排放的短寿命污染物——对国家石油需求进行现时预测。他们发现，日度 NO₂ 数据相对于使用传统预测变量的模型提高了现时预测精度，说明卫星观测能够及时反映石油消费变化。

Other studies use satellite observations to measure oil-related infrastructure and trade. Jung (2026) combines radar observations, night-time lights and port characteristics to nowcast port-level maritime trade. Polinov, Bookman and Levin (2022) also identify a relationship between night-time lights and shipping activity in anchorage areas. Together, these studies show that remote sensing can capture physical and economic activity associated with maritime trade.

其他研究使用卫星观测测量石油相关基础设施和贸易。Jung（2026）结合雷达观测、夜间灯光和港口特征，对港口级海运贸易进行现时预测。Polinov、Bookman 与 Levin（2022）也发现夜间灯光与锚地航运活动之间存在关系。这些研究共同表明，遥感可以捕捉与海运贸易相关的实体与经济活动。

Nevertheless, remote-sensing variables remain indirect measures of oil-market conditions. A signal that measures storage capacity does not necessarily reflect current inventories, while port activity does not directly measure future oil prices. Short-term changes may also reflect cloud cover, observation conditions or irregular data availability. Evidence obtained from one sensor or target therefore cannot automatically be applied to another.

尽管如此，遥感变量仍是油市状况的间接测量。反映储存容量的信号不一定代表当前库存，港口活动也不直接测量未来油价。短期变化还可能受到云层、观测条件或数据不规则性的影响。因此，针对某种传感器或预测对象得到的结论，不能自动推广到其他数据和任务。

Existing studies demonstrate that satellite observations provide information about oil demand, inventories and maritime trade. Yet direct evidence of their incremental value for short-term Brent price forecasting beyond conventional financial and oil-market information remains limited.

现有研究表明，卫星观测能够提供有关石油需求、库存和海运贸易的信息。但在常规金融与油市信息之外，它们对短期 Brent 价格预测是否具有增量价值，目前仍缺乏直接证据。

## 2.6 Multimodal learning

## 2.6 多模态学习

Multimodal learning refers to methods that process and combine information from two or more types of data. Each type of data is treated as a modality, and the purpose of multimodal learning is to use their complementary information while accounting for differences in structure, scale and availability.

多模态学习是指处理并结合两种或两种以上数据类型的方法。每一种数据类型都可以被视为一种模态。多模态学习的目标是在考虑不同数据结构、尺度和可得性差异的同时，利用各类数据之间的互补信息。

Baltrušaitis, Ahuja and Morency (2019) identify representation, translation, alignment, fusion and co-learning as five core challenges in multimodal machine learning. Among these challenges, fusion concerns how information from different modalities is combined. A useful distinction is between feature-level and representation-level fusion. Feature-level fusion places observed or engineered variables from all modalities in a common feature table before modelling. Representation-level fusion processes each modality separately before combining the resulting representations.

Baltrušaitis、Ahuja 与 Morency（2019）将表示、翻译、对齐、融合和协同学习视为多模态机器学习的五项核心挑战。其中，融合关注如何结合来自不同模态的信息。一个有用的区分是特征级融合与表示级融合。特征级融合在建模前将各模态的观测变量或工程化变量放入共同的特征表。表示级融合则先分别处理各类数据，再融合所得表征。

The two approaches retain different amounts of modality-specific structure. Feature-level fusion is compatible with conventional regression and tree-based models, but it may reduce temporal, spatial and network data to a common tabular format. Representation-level fusion can maintain separate processing streams for different data sources. Arevalo et al. (2017) propose a gated multimodal unit that combines modality-specific representations through input-dependent gates. The contribution of each modality can therefore change across observations. Emami-Gohari et al. (2024) apply a related modality-aware approach to financial forecasting by combining textual reports and numerical economic series. Their results show that separate representations and cross-modal interactions can improve performance in a financial time-series setting.

两种方法对模态特有结构的保留程度不同。特征级融合可以直接用于传统回归和树模型，但可能会将时间、空间和网络数据压缩为统一的表格格式。表示级融合则可以为不同数据来源保留独立的处理路径。Arevalo 等（2017）提出门控多模态单元，通过依赖输入的门控机制融合各模态表征。因此，不同模态的贡献可以随观测而变化。Emami-Gohari 等（2024）将相关的模态感知方法应用于金融预测，结合文本报告和数值经济序列。其结果表明，在金融时间序列场景中，分别建立表征并刻画跨模态交互可能改善预测表现。

Representation learning is also important for satellite imagery. For example, SatMAE (Cong et al., 2022) learns representations from temporal and multispectral satellite observations. Prithvi-EO-2.0 (Szwarcman et al., 2026) is pretrained on global multitemporal Earth-observation imagery and incorporates temporal and location information. CROMA (Fuller, Millard and Green, 2023) separately processes optical and radar observations before producing a joint representation. These models demonstrate that pretrained encoders can preserve spatial, spectral and temporal information that may not be captured by manually engineered satellite indicators. However, these models have mainly been evaluated on remote-sensing tasks such as classification, segmentation and disaster mapping, while their predictive value for commodity-price forecasting remains largely unexamined.

表示学习对于卫星影像同样重要。SatMAE（Cong 等，2022）从时序和多光谱卫星观测中学习表征。Prithvi-EO-2.0（Szwarcman 等，2026）在全球多时相对地观测影像上进行预训练，并纳入时间和位置信息。CROMA（Fuller、Millard 与 Green，2023）分别处理光学和雷达观测，再生成联合表征。这些模型表明，预训练编码器能够保留人工遥感指标可能无法充分刻画的空间、光谱和时间信息。不过，这些模型主要在分类、分割和灾害制图等遥感任务上接受评价，而其对于大宗商品价格预测的预测价值在很大程度上仍未得到检验。

Multimodal data also create alignment and missing-data problems. Data from different modalities may be observed at different frequencies and become available at different times. An entire modality may be missing for some observations, or individual observations within a modality may be irregular or delayed. Ma et al. (2022) show that multimodal models can be sensitive to missing modalities, while Neverova et al. (2016) propose ModDrop, which randomly drops one or more entire modalities during training to improve robustness when modalities are missing at test time. Time-series methods such as GRU-D (Che et al., 2018) and Multi-Time Attention Networks (Shukla and Marlin, 2021) explicitly represent missingness and irregular observation times. These studies show that adding more modalities does not automatically resolve differences in data availability and timing.

多模态数据还会带来对齐与缺失问题。不同模态的数据可能以不同频率被观测，也可能在不同时间变得可用。部分样本可能缺失整个模态，某一模态内部的观测也可能不规则或延迟。Ma 等（2022）表明，多模态模型可能对模态缺失较为敏感。Neverova 等（2016）则提出 ModDrop，即在训练期间随机丢弃一个或多个完整模态，以提高测试时模态缺失情况下的稳健性。GRU-D（Che 等，2018）和 Multi-Time Attention Networks（Shukla 与 Marlin，2021）等时间序列方法对缺失状态和不规则观测时间进行显式表示。这些研究表明，增加更多模态并不会自动解决数据可得时间和观测频率之间的差异。

Some multimodal architectures provide internal quantities that can be inspected. Modality gates can indicate how strongly the fitted model weights different data sources, while attention mechanisms can show how weights are distributed across inputs or representations. These quantities can help describe model behaviour, but they should not be treated as direct causal explanations. Jain and Wallace (2019) show that different attention patterns can sometimes produce similar predictions. Gates and attention weights therefore indicate how a model processes information, not how the underlying economic system is causally determined.

部分多模态架构会产生可供检查的内部量。模态门控可以显示拟合模型为不同数据来源分配的相对权重，注意力机制则可以展示权重如何分布于输入或表征之间。这些量有助于描述模型行为，但不应被直接视为因果解释。Jain 与 Wallace（2019）表明，不同的注意力分布有时可以产生相似的预测。因此，门控和注意力权重反映模型如何处理信息，而不是经济系统如何被因果决定。

Overall, existing multimodal research presents several approaches to representing and combining heterogeneous data, while also highlighting challenges related to alignment, missingness and interpretation.

总体而言，现有多模态研究提出了若干表示和结合异质数据的方法，同时也指出了对齐、缺失和解释方面的挑战。

## 2.7 Synthesis of the literature

## 2.7 文献综合

The review yields four conclusions. Oil-price forecasting remains challenging because price persistence makes the no-change benchmark difficult to outperform. Economic and financial variables are widely used because they capture macro-financial conditions and market expectations. Shipping and remote-sensing data offer plausible but noisy and indirect alternative signals. Multimodal learning can preserve modality-specific structure, but has yet to be systematically tested in commodity-price forecasting using heterogeneous shipping and satellite inputs.

综述得出四点结论。油价预测仍然具有挑战性，因为价格持续性使不变预测基准难以被超越。经济与金融变量被广泛使用，因其捕捉宏观金融条件和市场预期。航运与遥感数据提供合理但嘈杂且间接的另类信号。多模态学习能够保留模态特有结构，但尚未在使用异质航运与卫星输入的大宗商品价格预测中得到系统检验。

The following table summarises the observable signal, economic channel and main limitation of each of the four literatures, with key citations for each strand.

下表汇总四支文献各自的可观测信号、经济渠道与主要局限，并给出各脉络的关键引用。

**Table 2.1 — Literature synthesis**

**表 2.1 — 四支文献的可观测信号、经济渠道与主要局限**


| Data source / literature           | Observable signal                                                                                                          | Economic channel                                                                        | Main limitation                                             | Key references                                                                                                                                                                                                                     |
| ---------------------------------- | -------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------- | ----------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Financial and oil-market variables | Lagged price, inventories, production/refinery activity, volatility, GPR, rates, exchange rates, futures/market indicators | Persistence, uncertainty, macro-financial conditions, market expectations               | Strong benchmark; difficult to improve upon                 | Kilian (2009); Alquist, Kilian and Vigfusson (2013); Baumeister and Kilian (2015); Costa et al. (2021); Yılmaz and Zehir (2026)                                                                                                    |
| Shipping / AIS / PortWatch         | Tanker flows, port calls, chokepoint transits, capacity-weighted activity                                                  | Physical trade, supply disruption, congestion, regional flow changes                    | Directionality, noisy cargo inference, missing AIS activity | Adland, Jia and Strandenes (2017); Yan et al. (2020); Arslanalp, Marini and Tumbarello (2019); Arslanalp et al. (2026); Mi et al. (2022, 2023); Paolo et al. (2024); Ouyang et al. (2022); Liang et al. (2022); Zhao et al. (2022) |
| Remote sensing                     | Night-time lights, NO₂, cloud cover, site-level imagery or embeddings                                                      | Industrial activity, demand conditions, inventory observability, infrastructure signals | Indirect mechanism, cloud/missing data                      | Polinov, Bookman and Levin (2022); Hao and Wang (2023); Wang et al. (2019); Bricongne et al. (2026); Jung (2026)                                                                                                                   |
| Multimodal learning                | Modality-specific representations and fusion                                                                               | Preservation of heterogeneous structure before prediction                               | Limited direct evidence in oil-price forecasting            | Baltrušaitis, Ahuja and Morency (2019); Arevalo et al. (2017); Emami-Gohari et al. (2024); Cong et al. (2022); Fuller, Millard and Green (2023); Szwarcman et al. (2026); Ma et al. (2022)                                         |



| 数据来源 / 文献            | 可观测信号                                 | 经济渠道                    | 主要局限                 | 关键文献                                                                                                                                                                                                                               |
| -------------------- | ------------------------------------- | ----------------------- | -------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 金融与油市变量              | 滞后价格、库存、产量/炼厂活动、波动率、GPR、利率、汇率、期货/市场指标 | 持续性、不确定性、宏观金融条件、市场预期    | 基准很强，难以进一步改进         | Kilian (2009); Alquist, Kilian and Vigfusson (2013); Baumeister and Kilian (2015); Costa et al. (2021); Yılmaz and Zehir (2026)                                                                                                    |
| 航运 / AIS / PortWatch | 油轮流量、港口停靠、咽喉通行、运力加权活动                 | 实物贸易、供给扰动、拥堵、区域流量变化     | 方向性、嘈杂的货物推断、AIS 活动缺失 | Adland, Jia and Strandenes (2017); Yan et al. (2020); Arslanalp, Marini and Tumbarello (2019); Arslanalp et al. (2026); Mi et al. (2022, 2023); Paolo et al. (2024); Ouyang et al. (2022); Liang et al. (2022); Zhao et al. (2022) |
| 遥感                   | 夜光、NO₂、云量、站点级影像或嵌入                    | 工业活动、需求条件、库存可观测性、基础设施信号 | 机制间接、云/缺失数据          | Polinov, Bookman and Levin (2022); Hao and Wang (2023); Wang et al. (2019); Bricongne et al. (2026); Jung (2026)                                                                                                                   |
| 多模态学习                | 模态专属表征与融合                             | 预测前保留异质结构               | 油价预测中直接证据有限          | Baltrušaitis, Ahuja and Morency (2019); Arevalo et al. (2017); Emami-Gohari et al. (2024); Cong et al. (2022); Fuller, Millard and Green (2023); Szwarcman et al. (2026); Ma et al. (2022)                                         |


## 2.8 Research gaps

## 2.8 研究空白

Three research gaps follow from these findings.

这些结论引出三项研究空白。

First, the predictive value of shipping and remote-sensing data for Brent prices remains unclear. Oil-price forecasting studies mainly use historical prices, macroeconomic variables, financial indicators and oil-market data. Shipping research more often predicts trade or vessel activity, while remote-sensing research generally measures demand, infrastructure or port activity. These studies show that shipping and satellite observations contain economically relevant information, but they provide limited direct evidence on whether these sources improve one-week-ahead Brent price forecasts beyond financial time-series data.

第一，航运与遥感数据对 Brent 价格的预测价值仍不明确。油价预测研究主要使用历史价格、宏观经济变量、金融指标和油市数据。航运研究更多预测贸易或船舶活动，遥感研究通常测量需求、基础设施或港口活动。这些研究表明，航运和卫星观测包含具有经济意义的信息，但关于这些数据能否在金融时序数据之外改善提前一周的 Brent 价格预测，直接证据仍然有限。

Second, existing studies process shipping and remote-sensing data in different ways. Economic applications usually convert these data into numeric indicators and place them in a common feature table. Maritime-network and Earth-observation studies instead use graph models or pretrained encoders to preserve network, spatial or temporal structure. These approaches are usually examined in separate applications rather than compared in the same oil-price forecasting task. The literature therefore does not show whether modality-specific encoding performs better than flat feature fusion when both methods use the same underlying data.

第二，现有研究采用不同方式处理航运和遥感数据。经济学应用通常将这些数据转换为数值指标，再与其他变量共同放入一张特征表。海运网络和对地观测研究则使用图模型或预训练编码器，以保留数据的网络、空间或时间结构。这两类方法通常应用于不同的研究任务，尚未在同一个油价预测任务中进行直接比较。因此，在使用相同底层数据时，模态专属编码是否优于扁平特征融合，目前仍不明确。

Third, forecasting studies often use different evaluation settings, and analyses of model reliance are rarely connected directly to predictive improvements. Published studies differ in their forecast targets, horizons, samples, information sets and benchmarks. Their reported results are therefore not always directly comparable. In addition, studies that examine feature importance, modality gates or attention weights often report these results separately from out-of-sample forecasting performance. As a result, the literature provides limited evidence on whether a model’s reliance on a particular data source is associated with an actual improvement over a common benchmark.

第三，现有预测研究通常采用不同的评价设定，而且模型依赖分析很少与预测改善直接结合。已发表研究使用不同的预测目标、预测期、样本、信息集和评价基准，因此其报告结果并不总能直接比较。此外，分析特征重要性、模态门控或注意力权重的研究，通常将这些结果与样本外预测表现分开报告。因此，现有文献仍无法清楚说明，模型对某一数据来源的依赖是否对应其相对于共同基准的实际预测改善。

This dissertation addresses these gaps through a shared rolling-origin out-of-sample framework for one-week-ahead Brent price forecasting. It first compares financial time-series data with information sets that add remote-sensing data, shipping data or both. It then compares flat feature fusion with modality-aware representation-level fusion using matched underlying data. All models are evaluated against the same no-change benchmark in terms of forecast accuracy. Model interpretations are used to describe model reliance across prediction dates and geographic locations.

本文通过一套共享的滚动起点样本外框架回应这些研究空白，用于提前一周的 Brent 价格预测。研究首先比较金融时序数据，以及分别加入遥感数据、航运数据或二者的信息集。随后，在使用匹配底层数据的情况下，比较扁平特征融合与模态感知的表示级融合。所有模型均使用相同的不变预测基准，以预测精度进行评价。模型解释用于描述模型在不同预测日期和地理位置上的依赖。