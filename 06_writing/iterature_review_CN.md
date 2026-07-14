# 第 2 章 — 文献综述

本章综述与本论文研究定位直接相关的文献。第 2.1 节回顾原油价格预测研究，并解释为什么随机游走基准和金融基准模型很难被超越。第 2.2 节和第 2.3 节分别讨论两类替代数据来源——海运/AIS 航运活动和卫星遥感——如何作为原油市场状况的经济代理变量。第 2.4 节回顾多模态预测研究，并区分扁平特征融合与表示层面的模态感知融合。第 2.5 节讨论预测评估与可解释性标准。第 2.6 节综合上述文献，提出本文所关注的研究空白。

## 2.1 原油价格预测

### 2.1.1 计量经济学基础与基准问题

原油价格在样本外很难被准确预测。Kilian (2009) [P052] 指出，原油价格变动应从不同结构性渠道理解，包括原油供给冲击、总需求冲击和石油特定的预防性需求冲击。该分解本身并不是一个预测模型，但它为预测变量选择提供了一个有用原则：有用的变量应当与供给、需求或不确定性具有合理的经济联系，而不是仅仅因为数据可得就被纳入模型。

Alquist, Kilian and Vigfusson (2013) [P053] 为本文提供了关键的预测基准。他们表明，无变化预测，即随机游走预测，在原油价格预测中极难被超越，尤其是在样本外预测中。他们的综述还强调，样本内拟合并不等同于预测能力；任何关于可预测性的主张都应基于实时数据对齐、递归或滚动窗口评估，以及正式的预测比较检验。Baumeister and Kilian (2015) [P054] 进一步指出，跨不同经济机制的预测组合往往比依赖单一预测变量集更加稳健。综合来看，这些研究设定了很高的检验门槛：任何替代数据模型不仅需要与其他机器学习模型比较，还必须与强随机游走基准和具有经济含义的金融基准模型比较。

### 2.1.2 原油价格预测中的机器学习方法

机器学习研究已经将树模型、正则化线性模型、深度学习和混合模型引入原油价格预测。Costa et al. (2021) [P072] 在大规模宏观金融预测变量集上比较了多种方法，发现有用预测变量会随预测期限和时间变化而变化。XGBoost 可以构成强有力的基准，但并不在所有情境下都占优。Yılmaz and Zehir (2026) [P076] 表明，地缘政治风险、市场波动率和利率变量能够为 Brent 收益率预测提供增量信息，其中 LightGBM 在他们的设定中优于 XGBoost。Foroutan and Lahmiri (2024) [P001] 报告称，时间卷积网络和梯度提升模型表现较强，但其对价格水平预测的关注也反映出一个常见问题：由于原油价格高度持久，较低的一步价格水平误差可能部分来自 P_{t+1} 通常接近 P_t 这一事实。

一些近期研究采用了更复杂的模型结构，但其结果需要谨慎解读。例如，将 LSTM 特征提取与 XGBoost 相结合的混合设计报告了极高的 R^2 [P004]，但这类结果可能对预处理选择非常敏感，例如是否存在数据划分前的标准化泄漏。图模型也开始出现在原油价格预测中。Zhao, Xue and Cheng (2023) [P063] 将自注意力学习到的动态图与 Graph WaveNet (Wu et al., 2019) [P091] 结合，用于多步 WTI 期货预测。其图结构表示的是预测变量之间的非欧几里得关系，而不是物理意义上的航运网络或地理网络；同时，由于缺少无变化基准，模型是否真正优于随机游走仍难以判断。我将这些研究视为说明“模型复杂度本身并不足够”的证据；它们是背景文献，而不是本文周度 Brent 预测问题的直接设计模板。

对本文而言，该文献中有三点尤其重要。第一，价格自身滞后项和随机游走基准必须被视为强竞争基准，而不是弱基准。第二，金融与原油市场基本面应在加入替代数据前构成具有经济意义的金融基准：除波动率、地缘政治风险、利率、汇率和市场型原油指标外，库存、产量、炼厂活动等实物供需变量也是 Kilian 供给与需求渠道的天然代理。第三，任何关于航运或遥感数据提升预测能力的主张，都应建立在相对于该金融基准的样本外增量价值之上，而不能仅依赖原始误差下降。

## 2.2 航运活动与原油市场



### 2.2.1 AIS 与海运活动作为贸易流代理变量

自动识别系统（Automatic Identification System, AIS）船舶轨迹数据已经成为衡量实物贸易活动的重要高频代理变量。Adland et al. (2017) [P014] 将基于 AIS 估算的原油出口量与官方统计数据进行验证；Yan et al. (2020) [P015] 则表明，全球海上原油贸易高度集中于霍尔木兹海峡、马六甲海峡和苏伊士运河等关键咽喉。Arslanalp, Marini and Tumbarello (2019) [P018] 以及 IMF PortWatch 方法 [P070] 进一步展示了如何利用船舶移动数据对贸易活动进行即时预测。这些文献支持将航运数据视为实物市场信息的合理来源，但并不意味着 AIS 计数能够直接衡量原油流量或预测价格。

与简单船舶数量相比，按运力加权的指标和吃水变化指标通常更具信息含量，因为它们更接近货物流动本身。AIS 数据也需要仔细过滤，以排除非贸易活动；移动平均等处理也必须避免使用未来观测值。一个航运指标只有在其既具有经济含义、又在预测时点真实可得时，才可能对预测有用。在实践中，这意味着咽喉通行量和油轮存在度指标更适合被理解为关键海运节点的活动代理，而不是全球供给的直接测量。

### 2.2.2 反向因果与代理变量限制

航运活动与原油价格之间的关系并不是单向的。Mi et al. (2022) [P016] 以及 Mi, Zang, Lo and Chen (2023) [P017] 研究的是原油价格如何影响油轮港口停靠活动，而不是油轮活动如何预测价格。他们的发现表明，这一关系具有非线性和区域异质性，统计显著的关系可能只能解释较小比例的变异。航运数据可能包含关于实物贸易和拥堵的有用信息，但它也可能是对原油价格变化的反应，而非领先信号。

咽喉和港口指标也只是原油流动的不完美代理。船舶的上一港口并不总是货物原产地，原油可能被混合、转售，船对船转运也可能模糊真实贸易路径。Paolo et al. (2024) [P057] 进一步表明，大量海上工业活动并未被 AIS 覆盖。因此，航运指标应被理解为对实物市场状况的噪声代理变量，而不是全球原油供给的直接测量。

### 2.2.3 从扁平航运指标到海运结构

多数与原油相关的 AIS 和 PortWatch 数据应用，会将航运活动转化为港口停靠数、船舶数量或咽喉通行量等表格型特征。这种做法有实际价值，但会丢失海运活动本身的网络结构。海上贸易本质上具有空间性和关系性：港口、码头、咽喉和航线共同构成一个相互连接的运输系统。Ouyang et al. (2022) [P062] 和 Liang et al. (2022) [P066] 等研究表明，图模型可以学习船舶流量预测中的时空结构。这些研究预测的是交通流，而不是原油价格。它们支持“海运结构可以被学习”这一判断，但并没有证明海运表示一定能提升 Brent 价格预测。

## 2.3 卫星影像与遥感



### 2.3.1 遥感作为经济代理变量

遥感数据提供了观察经济活动、基础设施和环境条件的物理视角。在原油相关应用中，夜间灯光、NO₂、云量和高分辨率影像都曾被用作经济活动、贸易、需求或库存信息的间接指标。相关文献对这些信号能够衡量什么、不能衡量什么持较为谨慎的态度。

夜间灯光是最常用的遥感代理变量之一，但其有效性依赖于空间尺度。Polinov, Bookman and Levin (2022) [P024] 发现，夜间灯光在较大的横截面尺度上与锚地活动相关，但并不能可靠追踪单一港口的油轮活动。Gibson et al. (2021) [P032] 同样表明，VIIRS 夜间灯光比 DMSP 更适合设施尺度研究，但夜间灯光更擅长捕捉横截面差异，而非单位内部的时间变化。原始辐亮度不应被简单视为原油活动的直接时间序列指标。当研究目标是时间序列预测时，站点内部异常值比原始水平值更具合理性。

其他遥感指标对应不同机制。Hao and Wang (2023) [P025] 将美国储油区上空云量与下一周 WTI 收益率联系起来，其机制是信息可得性：当云层遮挡储油罐的光学观测时，市场关于库存的不确定性可能上升。Bricongne et al. (2026) [P069] 使用对流层 NO₂ 对国家层面的石油需求进行即时预测，但他们的结果也显示，当进入非线性模型后，遥感变量的增量价值可能减弱。Wang et al. (2019) [P055] 利用高分辨率影像估计储油罐结构容量，这支持基础设施测量，但并不等同于在 Sentinel-2 分辨率下进行高频库存估计。

### 2.3.2 直接 RS-to-price 主张的限制

卫星数据可能包含与原油市场相关的信息，但文献并不支持“卫星影像可以直接预测原油价格”的简单主张。多数遥感指标是上游代理变量：它们可能反映工业活动、港口活动、库存可观测性或需求条件，而这些因素又可能通过供需预期影响价格。该机制是间接的，并且可能随地点、传感器和预测期限而变化。

Jung (2026) [P068] 提供了一个有用例子。该研究将卫星衍生指标与港口属性结合，用于即时预测港口层面的贸易，但其预测目标是贸易而非价格，并且模型依赖工程化表格特征，而不是学习到的图像表示。这代表了更广泛的模式：遥感数据通常先被压缩为扁平数值列，再进入经济模型。这些特征可能具有价值，但尚不清楚的是，保留图像或站点层面的表示是否能够提升周度 Brent 价格预测。

## 2.4 多模态预测与融合



### 2.4.1 从多源数据到多模态学习

本文的一个关键区分是：多源特征融合并不等同于多模态表示学习。Baltrušaitis, Ahuja and Morency (2019) [P101] 将多模态学习定义为涵盖表示、转换、对齐、融合和协同学习等问题。在该分类框架下，融合可以发生在特征层、决策层或表示层。在多数原油价格预测研究中，异质数据通常通过早期特征层融合被合并：金融指标、航运计数和卫星衍生指数被拼接成一个表格，然后输入常规模型。这种方法具有实践便利性，但它会将所有输入都视作普通数值预测变量，并可能丢失模态特有结构。

模态感知的替代方案是先分别编码不同数据类型，再融合它们的表示。在融合环节本身，可学习的门控提供了一种动态加权各模态、而非同等对待的机制：Arevalo et al. (2017) [P096] 提出门控多模态单元（gated multimodal units），让模型针对每个输入学习各模态对融合表示的贡献权重。Gohari et al. (2024) [P039] 在金融时间序列预测中提供了一个相关先例，表明在结合不同金融信息来源时，模态感知 Transformer 可以优于简单拼接。但该研究处理的是文本和数值数据，而不是卫星影像、海运网络和原油市场变量。它能够说明模态感知预测具有潜力，但并不能直接回答这种融合是否适用于原油价格预测。

### 2.4.2 地球观测数据的表示学习

近期地球观测基础模型为将卫星影像转化为表示提供了一条路径。SatMAE [P095] 和 Prithvi-EO-2.0 [P094] 等模型使用自监督预训练来生成可迁移的图像嵌入。CROMA [P105] 等多传感器模型进一步表明，光学和雷达数据在融合前可能需要模态特定编码器，因为不同传感器在通道结构、噪声和物理含义上存在差异。这些模型与本文相关，是因为周度原油价格样本通常太小，不足以从零开始训练图像编码器。

地球观测基础模型文献对大宗商品价格预测而言，主要是方法论上的参考，而不是经验证据上的支撑。多数评估针对土地覆盖分类、语义分割或相关遥感任务，而不是经济预测或原油价格。这些模型说明卫星影像可以被预训练编码器表示，但并未证明这些表示能够改善 Brent 价格预测。

### 2.4.3 缺失模态与异步模态

另一个挑战是，替代数据来源通常存在缺失和时间不同步问题。光学卫星影像受云量影响，雷达和光学传感器具有不同重访周期，航运数据和宏观金融数据也可能以不同频率发布。一般多模态学习研究表明，如果不进行缺失模态训练或模态 dropout，模型在模态缺失时可能显著退化 [P097; P100]；针对不规则采样时间序列的相关研究也进一步说明，掩码和距上次观测时间等信号具有价值 [P098; P099]。

缺失性和时间错位本身就是多模态预测问题的一部分，而不是次要的数据清洗问题。扁平特征融合与表示层融合之间的公平比较，必须考虑每个模态的可得性、时间对齐和可靠性。

## 2.5 预测评估与可解释性



### 2.5.1 预测比较

由于随机游走很难被超越，仅仅报告较低的 RMSE 或 MAE 并不足以证明模型具有更强预测能力。Diebold and Mariano (1995) [P058] 提供了比较竞争预测之间预测精度是否相等的标准框架。对于嵌套模型，即一个模型通过增加预测变量扩展另一个模型的情形，Clark and West (2007) 在平方误差损失下提供了更合适的检验方法。这些检验在评估替代数据时尤其重要，因为关键问题不是一个大模型是否在某个样本中降低了误差，而是新增模态是否相对于更简单基准产生了统计上有意义的改进。

替代数据的预测价值应通过样本外比较、分预测期限的表现和正式预测精度检验来评估。这一标准适用于将金融基准模型与加入航运、遥感或多模态表示的模型进行比较的任何情形。

### 2.5.2 可解释性与模态层面解释

预测精度检验可以说明模型是否改进，但不能解释哪些信号推动了改进。SHAP (Lundberg and Lee, 2017) [P059] 提供了一种将模型预测归因于特征的方法，并且可以进一步聚合到模态层面。在本文情境中，模态层面解释很重要，因为论文不仅关心模型是否预测得更好，也关心金融、航运和遥感信号是否以经济上可解释的方式参与预测。

SHAP 解释的是模型行为，而不是因果效应。某个航运或卫星特征具有较高 attribution，并不意味着其背后的物理活动导致了原油价格变化。可解释性最好被视为预测比较检验的补充：精度检验评估某个模态是否有助于预测，而归因分析帮助描述训练后的模型如何使用该模态。

## 2.6 综合、研究空白与本文定位



### 2.6.1 文献综合

综合上述文献，可以得出四点结论。第一，原油价格预测很困难，因为原油价格高度持久，随机游走基准很强。第二，金融变量构成必要的、有经济含义的基准，因为它们能够捕捉价格持续性、不确定性、货币条件、汇率渠道和市场预期。第三，航运和遥感数据是有潜力的替代数据来源，但它们是噪声较强、机制间接的代理变量，而不是未来价格的直接测量。第四，多模态学习为保留模态特定结构提供了工具，但这些工具尚未在周度 Brent 价格预测这一具体情境中得到系统检验。

下表总结了四类文献各自的可观测信号、经济渠道与主要限制。


| 数据来源 / 文献            | 可观测信号                                 | 经济渠道                    | 主要限制                      |
| -------------------- | ------------------------------------- | ----------------------- | ------------------------- |
| 金融与原油市场变量            | 滞后价格、库存、产量/炼厂活动、波动率、GPR、利率、汇率、期货/市场指标 | 价格持续性、不确定性、宏观金融条件、市场预期  | 基准很强，难以进一步改进              |
| 航运 / AIS / PortWatch | 油轮流量、港口停靠、咽喉通行、运力加权活动                 | 实物贸易、供给扰动、拥堵、区域流动变化     | 反向因果、货物推断有噪声、AIS 活动缺失     |
| 遥感                   | 夜间灯光、NO₂、云量、站点影像或嵌入                   | 工业活动、需求条件、库存可观测性、基础设施信号 | 机制间接、站点内部时间变化较弱、云层/缺失数据问题 |
| 多模态学习                | 模态特定表示与融合                             | 在预测前保留异质数据结构            | 在原油价格预测中缺少直接证据            |




### 2.6.2 研究空白

现有原油价格预测研究已经在金融建模和机器学习方法上取得进展，但多数替代数据应用仍然会将异质数据压缩为工程化表格特征。航运活动通常被表示为计数、流量或咽喉指标；卫星影像通常被表示为夜间灯光、NO₂ 或云量等指数；这些变量随后与金融预测变量拼接为一个扁平特征表。

尚不清楚的是，增量价值究竟来自替代数据本身，还是来自在预测之前保留其模态特有结构。现有研究尚未在同一个防止数据泄漏的样本外评估框架下，系统比较金融、航运和遥感特征的扁平拼接与表示层面的模态感知融合对周度 Brent 价格预测的影响。待回答的问题并不只是“更多数据是否改善预测”，而是“异质数据如何被表示和融合，是否会影响预测表现”。

### 2.6.3 本文定位

本文定位为一项经验性整合与比较研究，而不是提出一种新的神经网络结构。它结合了三类文献：原油预测文献对强基准和严格样本外检验的强调；替代数据文献将航运和卫星信号作为经济活动代理变量的做法；以及多模态学习文献对扁平特征融合与表示层模态感知融合的区分。

因此，本文提出三个相互关联的研究问题：

- **RQ1：** 遥感和航运指标是否能在金融基准模型和随机游走基准之上提供样本外增量预测价值？
- **RQ2：** 在使用相同底层数据的情况下，模态感知的表示层融合是否优于扁平特征融合？
- **RQ3：** 模态层面的可解释性分析是否能够揭示模型在不同市场条件下依赖哪些信号？

替代数据是否有用、表示层融合是否更优——这两个主张都被视为需要在统一评估框架下检验的经验问题，而不是本文的预设前提。

---



## 参考文献

*（内部编号暂时保留，方便追踪。最终提交前，请将本列表替换为完整 Harvard / APA 格式参考文献，包括作者、年份、标题、出版物、卷号、页码以及 DOI、arXiv 或 working paper 信息。）*

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

