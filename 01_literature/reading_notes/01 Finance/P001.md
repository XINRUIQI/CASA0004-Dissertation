
# Reading Note — Paper P001: Price-Only Deep Learning for Oil and Precious Metal Forecasting  
# 阅读笔记 — P001：基于纯价格序列的原油与贵金属深度学习预测

---

## Citation  
## 文献信息

Foroutan, P., & Lahmiri, S. (2024). Deep learning systems for forecasting the prices of crude oil and precious metals. *Financial Innovation*, 10, Article 111.

- **DOI**: [10.1186/s40854-024-00637-z](https://doi.org/10.1186/s40854-024-00637-z)
- **Published / 发表时间**: 2024
- **Journal / 期刊**: *Financial Innovation*
- **Research type / 研究类型**: Comparative machine-learning and deep-learning forecasting study  
  机器学习与深度学习预测模型比较研究

---

## Research Objective  
## 研究目标

The paper investigates which machine-learning and deep-learning architectures are most effective for forecasting the next-day prices of crude oil and precious metals using only their historical price series.

The study focuses on two main experimental dimensions:

1. The choice of forecasting model.
2. The length of the historical input window.

The paper does not attempt to explain the economic causes of oil-price changes. Instead, it focuses on whether advanced nonlinear models can extract useful predictive patterns from historical prices.

本文研究的核心问题是：在只使用商品自身历史价格序列的条件下，哪一种机器学习或深度学习模型最适合预测原油和贵金属的下一交易日价格。

论文主要比较两个实验维度：

1. 预测模型的选择；
2. 历史输入窗口的长度。

该论文并不重点解释油价变化的经济原因，而是研究复杂非线性模型能否从历史价格中提取有效的预测模式。

---

## Research Questions  
## 研究问题

The paper addresses the following questions:

1. Which deep-learning model provides the most accurate forecasts for crude oil, gold, and silver prices?
2. Can one model consistently outperform other models across different commodity markets?
3. Which historical input-window length is most informative?
4. Do hybrid neural-network models outperform standalone models?
5. How do recurrent, convolutional, temporal-convolutional, and machine-learning models differ in forecasting performance?

论文主要回答以下问题：

1. 哪一种深度学习模型能够最准确地预测原油、黄金和白银价格？
2. 是否存在一种模型可以在不同商品市场中持续优于其他模型？
3. 多长的历史输入窗口最具有预测价值？
4. 混合神经网络是否优于单一模型？
5. 循环神经网络、卷积网络、时序卷积网络和传统机器学习模型的预测表现有何差异？

---

## Dataset  
## 数据集

The study uses daily closing spot prices for four commodity markets:

| Market | Data source |
|---|---|
| WTI crude oil | U.S. Energy Information Administration |
| Brent crude oil | U.S. Energy Information Administration |
| Gold | KITCO |
| Silver | KITCO |

- **Time period**: 4 January 2000 to 25 March 2022
- **Frequency**: Daily
- **Sample size**: 5,426 observations for each market
- **Data alignment**: Only common trading dates across all four markets are retained

研究使用四个商品市场的日度现货收盘价：

| 市场 | 数据来源 |
|---|---|
| WTI 原油 | 美国能源信息署 EIA |
| Brent 原油 | 美国能源信息署 EIA |
| 黄金 | KITCO |
| 白银 | KITCO |

- **时间范围**：2000年1月4日至2022年3月25日
- **数据频率**：日度
- **样本数量**：每个市场5,426条观测
- **日期处理**：仅保留四个市场共同存在交易数据的日期

The paper does not include platinum.

论文没有使用铂金数据。

---

## Input Variables  
## 输入变量

The study adopts a univariate, price-only forecasting framework.

For each commodity, the input consists only of its own historical price levels:

\[
X_t = [P_{t-s+1}, P_{t-s+2}, \ldots, P_t]
\]

where \(s\) is the sliding-window length.

The model predicts:

\[
\hat{P}_{t+1}
\]

The following input-window lengths are tested:

- 5 days
- 30 days
- 60 days
- 90 days

No external explanatory variables are included.

The study does not use:

- Macroeconomic variables
- Oil production
- Oil consumption
- Inventories
- Futures spreads
- Exchange rates
- Interest rates
- Technical indicators
- News sentiment
- Shipping activity
- Remote-sensing data
- Historical log returns as model inputs

本文采用单变量、纯价格序列预测框架。

对于每一种商品，输入仅包括该商品自身过去若干日的价格水平：

\[
X_t = [P_{t-s+1}, P_{t-s+2}, \ldots, P_t]
\]

其中 \(s\) 为滑动窗口长度。

模型预测：

\[
\hat{P}_{t+1}
\]

论文测试了以下四种窗口：

- 5日
- 30日
- 60日
- 90日

论文没有加入外部解释变量，包括：

- 宏观经济变量
- 原油产量
- 原油消费量
- 库存
- 期货价差
- 汇率
- 利率
- 技术指标
- 新闻情绪
- 航运活动
- 遥感数据
- 历史对数收益率

The Time2Vector models generate learnable temporal embeddings, but these embeddings are transformations of time information rather than external economic variables.

Time2Vector 模型会生成可学习的时间嵌入，但这些嵌入属于时间信息的转换，并不是外部经济变量。

---

## Forecasting Target  
## 预测目标

The forecasting target is the next-day spot-price level for each commodity.

\[
y_t = P_{t+1}
\]

The task is therefore:

- One-step-ahead forecasting
- Price-level forecasting
- Continuous regression

The paper does not forecast:

- Log returns
- Price direction
- Volatility
- Prediction intervals
- Long-horizon or multi-step prices

预测目标是每种商品下一交易日的现货价格水平：

\[
y_t = P_{t+1}
\]

因此，该任务属于：

- 单步预测；
- 价格水平预测；
- 连续变量回归。

论文没有预测：

- 对数收益率；
- 油价上涨或下跌方向；
- 波动率；
- 预测区间；
- 长期或多步价格。

---

## Data Split  
## 数据划分

The observations are divided chronologically rather than randomly.

| Dataset | Share | Time period |
|---|---:|---|
| Training set | 65% | 2000-01-04 to 2014-06-15 |
| Validation set | 25% | 2014-06-16 to 2020-01-02 |
| Test set | 10% | 2020-01-03 to 2022-03-25 |

This chronological split reduces the risk of training the models using future observations.

The test period includes:

- The COVID-19 crisis
- The April 2020 oil-price collapse
- The beginning of the Russia–Ukraine conflict in 2022

论文按照时间顺序划分数据，而不是随机划分。

| 数据集 | 比例 | 时间范围 |
|---|---:|---|
| 训练集 | 65% | 2000-01-04 至 2014-06-15 |
| 验证集 | 25% | 2014-06-16 至 2020-01-02 |
| 测试集 | 10% | 2020-01-03 至 2022-03-25 |

这种时间顺序划分降低了模型利用未来数据训练的风险。

测试时期包含：

- COVID-19危机；
- 2020年4月油价暴跌；
- 2022年俄乌冲突初期。

---

## Data Preprocessing  
## 数据预处理

The price series are normalised to the interval \([0,1]\) using Min-Max scaling:

\[
x_t' =
\frac{x_t-\min(x)}
{\max(x)-\min(x)}
\]

The paper states that normalisation is used to:

- Reduce scale differences
- Improve neural-network training
- Accelerate parameter optimisation
- Reduce the influence of noise

价格数据通过 Min-Max 方法归一化到 \([0,1]\)：

\[
x_t' =
\frac{x_t-\min(x)}
{\max(x)-\min(x)}
\]

论文认为归一化可以：

- 缩小数据尺度差异；
- 改善神经网络训练；
- 加快参数更新；
- 降低噪声的影响。

For replication, the scaling parameters should be estimated using the training data only and then applied to the validation and test data.

在复现过程中，归一化参数应当只使用训练集计算，然后应用到验证集和测试集，避免产生数据泄漏。

---

## Core Method  
## 核心方法

The study compares 16 forecasting models:

- 12 deep-learning models
- 4 machine-learning baseline models

论文共比较16种预测模型：

- 12种深度学习模型；
- 4种机器学习基线模型。

### Deep-Learning Models  
### 深度学习模型

1. Long Short-Term Memory — LSTM
2. Bidirectional LSTM — BiLSTM
3. Gated Recurrent Unit — GRU
4. Bidirectional GRU — BiGRU
5. Time2Vector-BiLSTM — T2V-BiLSTM
6. Time2Vector-BiGRU — T2V-BiGRU
7. Convolutional Neural Network — CNN
8. CNN-BiLSTM
9. CNN-BiGRU
10. Temporal Convolutional Network — TCN
11. TCN-BiLSTM
12. TCN-BiGRU

### Machine-Learning Baselines  
### 机器学习基线模型

1. Random Forest
2. LightGBM
3. Support Vector Regression — SVR
4. K-Nearest Neighbours — KNN

---

## Model Architecture Interpretation  
## 模型结构解释

### LSTM and GRU

LSTM and GRU are recurrent neural networks designed to capture temporal dependencies.

LSTM uses input, forget, and output gates, whereas GRU uses a simpler structure with fewer parameters.

The paper finds that GRU-related models frequently outperform corresponding LSTM models.

LSTM和GRU属于用于捕捉时间依赖关系的循环神经网络。

LSTM包含输入门、遗忘门和输出门，而GRU结构更简单、参数更少。

论文发现，GRU系列模型经常优于对应的LSTM模型。

### Bidirectional Networks

BiLSTM and BiGRU process each historical input window in both forward and reverse order.

The reverse processing occurs only within the historical window and does not necessarily imply the use of observations after the prediction date.

BiLSTM和BiGRU从正向和反向处理同一个历史输入窗口。

反向处理只发生在历史窗口内部，并不必然意味着模型使用了预测日期之后的数据。

### CNN

The CNN extracts local temporal patterns from historical prices, such as short-term trends and local fluctuations.

However, a conventional CNN may have limited ability to capture long-term temporal dependence.

CNN主要从历史价格中提取局部时间模式，例如短期趋势和局部波动。

但普通CNN捕捉长期时间依赖关系的能力相对有限。

### TCN

The Temporal Convolutional Network uses:

- Causal convolutions
- Dilated convolutions
- Residual connections
- Dropout regularisation
- Fully connected output layers

Causal convolutions ensure that predictions are based only on current and historical observations.

Dilated convolutions allow the model to cover a long historical receptive field without requiring a very deep network.

时序卷积网络TCN使用：

- 因果卷积；
- 扩张卷积；
- 残差连接；
- Dropout正则化；
- 全连接输出层。

因果卷积确保模型只使用当前和过去的信息。

扩张卷积使模型不需要建立非常深的网络，就能够覆盖较长的历史范围。

### Time2Vector

Time2Vector transforms time into learnable periodic and non-periodic representations.

It is intended to help the model identify recurring temporal patterns without manually constructing seasonal variables.

Time2Vector将时间转换为可学习的周期性和非周期性表示。

其目的是帮助模型自动识别重复出现的时间规律，而不需要人工构造季节性特征。

### Hybrid Models

The hybrid CNN-RNN and TCN-RNN architectures first extract local or temporal features and then pass them into BiLSTM or BiGRU layers.

However, the empirical results show that hybrid architectures do not consistently outperform standalone models.

CNN-RNN和TCN-RNN混合架构首先提取局部或时间特征，然后将这些特征输入BiLSTM或BiGRU。

但论文结果表明，混合模型并没有持续优于单一模型。

---

## Training Configuration  
## 模型训练设置

The main training settings include:

| Hyperparameter | Value |
|---|---:|
| Maximum epochs | 50 |
| Batch size | 32 |
| Dropout rate | 0.2 |
| Initial learning rate | 0.001 |
| Optimiser | Adam |
| Training loss | Mean Squared Error |
| Early stopping patience | 10 epochs |
| Number of repeated runs | 10 |

Hyperparameters are selected using grid search on the validation set.

The reported prediction errors are averages across ten independent runs to reduce the effect of random initialisation.

主要训练设置包括：

| 超参数 | 数值 |
|---|---:|
| 最大训练轮数 | 50 |
| Batch size | 32 |
| Dropout比例 | 0.2 |
| 初始学习率 | 0.001 |
| 优化器 | Adam |
| 训练损失函数 | 均方误差 MSE |
| Early stopping patience | 10轮 |
| 重复运行次数 | 10次 |

作者通过验证集上的网格搜索选择超参数。

最终报告的预测误差是10次独立运行的平均值，以降低随机初始化的影响。

---

## Evaluation Metrics  
## 评价指标

The paper evaluates forecasts using:

1. Mean Absolute Error — MAE
2. Mean Absolute Percentage Error — MAPE
3. Root Mean Squared Error — RMSE

\[
MAE =
\frac{1}{n}
\sum_{i=1}^{n}
|\hat{y}_i-y_i|
\]

\[
MAPE =
\frac{100}{n}
\sum_{i=1}^{n}
\left|
\frac{\hat{y}_i-y_i}{y_i}
\right|
\]

\[
RMSE =
\sqrt{
\frac{1}{n}
\sum_{i=1}^{n}
(\hat{y}_i-y_i)^2
}
\]

论文使用以下指标评价预测表现：

1. 平均绝对误差 MAE；
2. 平均绝对百分比误差 MAPE；
3. 均方根误差 RMSE。

所有指标都是数值越低，代表预测误差越小。

---

## Key Findings  
## 核心发现

### 1. TCN provides the strongest overall deep-learning performance  
### 1. TCN总体上是表现最强的深度学习模型

TCN achieves the best MAE results for:

- WTI
- Brent
- Silver

Its best reported MAE values are approximately:

| Market | Best model | Best MAE |
|---|---|---:|
| WTI | TCN | 1.444 |
| Brent | TCN | 1.295 |
| Silver | TCN | 0.346 |

TCN在以下市场获得最低MAE：

- WTI；
- Brent；
- Silver。

其主要最优结果约为：

| 市场 | 最佳模型 | 最低MAE |
|---|---|---:|
| WTI | TCN | 1.444 |
| Brent | TCN | 1.295 |
| 白银 | TCN | 0.346 |

### 2. BiGRU performs best for gold  
### 2. BiGRU在黄金市场表现最好

For gold, the best-performing model is BiGRU with a 30-day historical input window.

Its reported MAE is approximately:

\[
MAE = 15.188
\]

This indicates that there is no single model that dominates all commodity markets.

对于黄金，表现最好的模型是使用30日输入窗口的BiGRU。

其MAE约为：

\[
MAE = 15.188
\]

这说明不存在一种模型能够在所有商品市场中都取得绝对最优结果。

### 3. LightGBM is the strongest machine-learning model  
### 3. LightGBM是表现最强的传统机器学习模型

LightGBM provides performance comparable to TCN in several oil-price forecasting experiments.

It outperforms many recurrent and hybrid deep-learning models despite having a simpler tabular structure.

LightGBM在部分原油预测实验中的表现接近TCN。

尽管其结构比深度学习模型简单，但它优于许多循环神经网络和混合模型。

### 4. More complex models do not necessarily perform better  
### 4. 更复杂的模型不一定表现更好

Hybrid architectures such as:

- CNN-BiLSTM
- CNN-BiGRU
- TCN-BiLSTM
- TCN-BiGRU

do not consistently outperform standalone TCN, GRU, or BiGRU models.

以下混合架构并未持续优于单独的TCN、GRU或BiGRU：

- CNN-BiLSTM；
- CNN-BiGRU；
- TCN-BiLSTM；
- TCN-BiGRU。

### 5. Longer historical windows are not always better  
### 5. 更长的历史窗口不一定更好

The optimal window length differs across markets and models.

A 90-day input window does not consistently outperform 30-day or 60-day windows.

Long windows may provide more historical information, but they may also introduce irrelevant or outdated patterns.

最优窗口长度会随着市场和模型而变化。

90日窗口并没有稳定优于30日或60日窗口。

较长窗口虽然提供更多历史信息，但也可能引入无关或已经失效的模式。

### 6. GRU variants generally outperform corresponding LSTM variants  
### 6. GRU系列通常优于对应的LSTM系列

The empirical results frequently show:

- GRU outperforming LSTM
- BiGRU outperforming BiLSTM
- T2V-BiGRU outperforming T2V-BiLSTM

This may reflect the simpler structure and lower parameter count of GRU models.

论文结果经常表现为：

- GRU优于LSTM；
- BiGRU优于BiLSTM；
- T2V-BiGRU优于T2V-BiLSTM。

这可能与GRU结构更简单、参数数量更少有关。

---

## Main Contribution  
## 主要贡献

The paper provides a broad comparison of recurrent, convolutional, temporal-convolutional, hybrid, and tree-based models under a common experimental framework.

Its main contribution is methodological rather than economic.

It demonstrates that:

1. TCN is a competitive architecture for commodity-price forecasting.
2. LightGBM remains a strong benchmark against deep-learning models.
3. Input-window selection materially affects forecasting results.
4. Hybrid models are not automatically superior.
5. Price-only models can achieve relatively low forecasting errors during short-horizon price-level prediction.

该论文在统一实验框架下比较了循环网络、卷积网络、时序卷积网络、混合模型和树模型。

其主要贡献属于方法层面，而不是经济机制层面。

论文说明：

1. TCN是具有竞争力的商品价格预测模型；
2. LightGBM是深度学习模型不可忽视的强基线；
3. 输入窗口长度会显著影响预测结果；
4. 混合模型不一定自动优于简单模型；
5. 在短期价格水平预测中，纯价格模型也可以获得相对较低的误差。

---

## Relevance to This Dissertation  
## 对本论文项目的借鉴意义

| Aspect | Connection to the dissertation |
|---|---|
| Price-only baseline | The study provides a direct methodological reference for constructing an M0 model based only on historical Brent prices and price-derived features. |
| Model choice | TCN should be considered alongside LSTM, TFT, and ST-GNN as a representative sequence model. |
| Tree-model benchmark | LightGBM or XGBoost should be retained as a strong tabular baseline because tree models may match or outperform complex neural networks. |
| Window sensitivity | The dissertation should compare multiple historical look-back windows rather than selecting one window arbitrarily. |
| Crisis evaluation | Performance should be evaluated separately during normal periods and major oil-market disruptions. |
| Multimodal research gap | The absence of macroeconomic, text, shipping, and remote-sensing information provides a clear motivation for the dissertation's multimodal extension. |
| Ablation design | The paper supports a controlled comparison between price-only and expanded information sets. |

| 方面 | 与本论文项目的关系 |
|---|---|
| 纯价格基线 | 该论文可以直接用于构建只包含Brent历史价格和价格衍生特征的M0模型。 |
| 模型选择 | 除LSTM、TFT和ST-GNN外，项目应考虑加入TCN作为代表性序列模型。 |
| 树模型基线 | 应保留LightGBM或XGBoost作为强表格数据基线，因为树模型可能达到或超过复杂神经网络。 |
| 窗口敏感性 | 项目应比较多个历史回看窗口，而不是任意选择一个固定窗口。 |
| 危机期评价 | 应分别评价正常时期和重大石油市场冲击时期的预测表现。 |
| 多模态研究缺口 | 论文没有使用宏观、文本、航运和遥感数据，这为本项目的多模态扩展提供明确研究动机。 |
| 消融实验设计 | 论文支持在纯价格信息集与扩展信息集之间进行受控比较。 |

---

## Recommended Role in the Dissertation Framework  
## 在论文框架中的建议定位

This paper should be classified as a methodological benchmark for the M0 price-only forecasting stage.

A suitable framework is:

\[
M0 =
\text{Historical oil-price information}
\]

\[
M1 =
M0 + \text{Financial, macroeconomic and fundamental variables}
\]

\[
M2 =
M1 + \text{Text and sentiment features}
\]

\[
M3 =
M2 + \text{Shipping and port-activity features}
\]

\[
M4 =
M3 + \text{Remote-sensing and spatial-network features}
\]

The paper should not be used as the main justification for selecting the variables in M1–M4 because it does not empirically test those external variables.

It is more appropriate to use the paper to justify:

- The construction of a price-only benchmark
- The inclusion of TCN
- The inclusion of LightGBM
- Sliding-window sensitivity analysis
- Model-complexity controls

该论文应被定位为M0纯价格预测阶段的方法基准文献。

建议的模型框架为：

\[
M0 =
\text{历史油价信息}
\]

\[
M1 =
M0 + \text{金融、宏观和基本面变量}
\]

\[
M2 =
M1 + \text{文本和情绪特征}
\]

\[
M3 =
M2 + \text{航运和港口活动特征}
\]

\[
M4 =
M3 + \text{遥感和空间网络特征}
\]

由于该论文没有实证检验外部变量，因此不适合作为M1–M4具体变量选择的核心依据。

更适合使用该论文支持：

- 建立纯价格基线；
- 加入TCN模型；
- 加入LightGBM模型；
- 开展滑动窗口敏感性分析；
- 控制模型复杂度。

---

## Specific Lessons for Model Design  
## 对模型设计的具体启示

### 1. Introduce a separate M0 price-only baseline  
### 1. 单独设置M0纯价格基线

The dissertation should separate price-history information from wider financial and fundamental information.

A possible M0 specification is:

\[
M0 =
\{
P_{t-1:t-k},
r_{t-1:t-k},
\text{rolling volatility},
\text{price momentum}
\}
\]

This allows the marginal forecasting contribution of external variables to be measured more clearly.

项目应将历史价格信息与更广泛的金融和基本面信息分开。

一个可能的M0为：

\[
M0 =
\{
P_{t-1:t-k},
r_{t-1:t-k},
\text{滚动波动率},
\text{价格动量}
\}
\]

这样可以更清楚地衡量外部变量的边际预测贡献。

### 2. Include both TCN and tree-based models  
### 2. 同时使用TCN和树模型

The paper suggests that a suitable core comparison would include:

- Random walk
- Autoregression or ARIMA
- XGBoost or LightGBM
- LSTM
- TCN
- TFT
- ST-GNN

It is not necessary to reproduce all 16 models because the dissertation's main research question concerns information modalities rather than identifying the best neural-network architecture.

论文表明，一个合理的核心模型组合可以包括：

- Random walk；
- 自回归或ARIMA；
- XGBoost或LightGBM；
- LSTM；
- TCN；
- TFT；
- ST-GNN。

没有必要复制全部16种模型，因为本论文的核心问题是比较不同信息模态，而不是寻找最优神经网络结构。

### 3. Test multiple weekly look-back windows  
### 3. 测试多个周度回看窗口

Because the dissertation uses weekly data, possible windows include:

- 4 weeks
- 8 weeks
- 13 weeks
- 26 weeks
- 52 weeks

The optimal window may differ by:

- Forecast horizon
- Model
- Data modality
- Market regime

由于本项目使用周度数据，可以测试：

- 4周；
- 8周；
- 13周；
- 26周；
- 52周。

最优窗口可能随着以下因素变化：

- 预测期限；
- 模型；
- 数据模态；
- 市场状态。

### 4. Hold the experimental conditions constant across M0–M4  
### 4. 在M0–M4之间保持实验条件一致

To attribute an improvement to a newly added modality, the following should remain identical:

- Target variable
- Forecast horizon
- Sample period
- Train-validation-test folds
- Historical window
- Evaluation metrics
- Hyperparameter-search budget

为了将性能改善归因于新加入的数据模态，以下设置应保持一致：

- 目标变量；
- 预测期限；
- 样本时期；
- 训练、验证和测试划分；
- 历史窗口；
- 评价指标；
- 超参数搜索预算。

### 5. Evaluate crisis and normal periods separately  
### 5. 分别评价危机期和正常时期

The dissertation can examine whether external modalities are particularly useful during:

- The 2008 global financial crisis
- The 2014–2016 oil-price collapse
- The 2020 COVID-19 shock
- The 2022 Russia–Ukraine shock
- Major shipping and geopolitical disruptions

A likely research hypothesis is that price-only models may perform adequately during normal periods, while text, shipping, and remote-sensing variables provide greater value during supply-chain disruptions.

项目可以检验外部模态在以下时期是否更加有用：

- 2008年全球金融危机；
- 2014—2016年油价暴跌；
- 2020年COVID-19冲击；
- 2022年俄乌冲突；
- 重大航运和地缘政治中断。

一个合理的研究假设是：纯价格模型可能在正常时期表现较好，而文本、航运和遥感变量在供应链中断时期具有更高的增量价值。

---

## Limitations  
## 局限性

### 1. Price-only information set  
### 1. 仅使用价格信息

The study excludes the main economic and physical drivers of oil prices.

It does not include:

- Supply
- Demand
- Inventories
- Production
- Futures-market structure
- Exchange rates
- Interest rates
- Economic activity
- Geopolitical shocks
- Shipping flows
- News sentiment

Therefore, the models may capture price persistence without explaining the mechanisms behind oil-price changes.

论文没有使用油价的主要经济和物理驱动因素，包括：

- 供给；
- 需求；
- 库存；
- 产量；
- 期货市场结构；
- 汇率；
- 利率；
- 经济活动；
- 地缘政治冲击；
- 航运流量；
- 新闻情绪。

因此，模型可能主要捕捉价格持续性，而无法解释油价变化背后的机制。

### 2. No random-walk or no-change benchmark  
### 2. 缺少随机游走或价格不变基线

The paper does not clearly compare the complex models with the important benchmark:

\[
\hat{P}_{t+1}=P_t
\]

Because daily oil-price levels are highly persistent, a sophisticated model must demonstrate that it improves upon this simple forecast.

论文没有明确将复杂模型与以下重要基线进行比较：

\[
\hat{P}_{t+1}=P_t
\]

由于日度油价水平具有高度持续性，复杂模型必须证明其表现优于这一简单预测。

### 3. Price-level forecasting may produce optimistic error measures  
### 3. 价格水平预测可能产生过于乐观的误差结果

For one-day-ahead prediction:

\[
P_{t+1} \approx P_t
\]

A model can obtain a low MAE by closely reproducing the latest observed price.

This does not necessarily mean that it can predict price changes or market turning points.

在下一日预测中：

\[
P_{t+1} \approx P_t
\]

模型只要接近复制最近一期价格，就可能得到较低MAE。

这并不必然意味着模型能够预测价格变化或市场转折点。

### 4. No return, direction, or volatility forecasts  
### 4. 没有预测收益率、方向或波动率

The paper only predicts price levels.

A more comprehensive design should also consider:

\[
r_{t+h}
=
\log(P_{t+h})-\log(P_t)
\]

and:

\[
D_{t+h}
=
\mathbb{1}(r_{t+h}>0)
\]

论文只预测价格水平。

更完整的实验设计还应考虑：

\[
r_{t+h}
=
\log(P_{t+h})-\log(P_t)
\]

以及：

\[
D_{t+h}
=
\mathbb{1}(r_{t+h}>0)
\]

### 5. MAPE is problematic for negative or near-zero WTI prices  
### 5. MAPE不适合负值或接近零的WTI价格

The WTI sample includes negative prices during April 2020.

MAPE becomes difficult to interpret when the true price is:

- Negative
- Zero
- Close to zero

The dissertation should therefore prioritise:

- MAE
- RMSE
- MASE
- sMAPE
- Out-of-sample \(R^2\)
- Directional accuracy

WTI样本包含2020年4月的负油价。

当真实价格为以下情况时，MAPE很难解释：

- 负数；
- 零；
- 接近零。

因此，本项目应优先考虑：

- MAE；
- RMSE；
- MASE；
- sMAPE；
- 样本外 \(R^2\)；
- 方向准确率。

### 6. Single fixed train-test split  
### 6. 只使用一次固定的数据划分

The study uses one chronological training-validation-test split.

Its findings may therefore be sensitive to the selected test period.

A stronger design would use:

- Rolling-window evaluation
- Expanding-window evaluation
- Multiple forecast origins

论文只使用一次按时间划分的训练、验证和测试集。

结果可能受到特定测试时期的影响。

更稳健的设计应使用：

- 滚动窗口评价；
- 扩展窗口评价；
- 多个预测起点。

### 7. Limited formal statistical testing  
### 7. 缺少正式的预测显著性检验

The paper mainly compares numerical error metrics.

It does not fully establish whether small differences in forecasting errors are statistically significant.

The dissertation can add:

- Diebold–Mariano tests
- Bootstrap confidence intervals
- Model Confidence Set tests

论文主要比较预测误差数值。

但并未充分检验不同模型之间较小的误差差异是否具有统计显著性。

本项目可以加入：

- Diebold–Mariano检验；
- Bootstrap置信区间；
- Model Confidence Set检验。

### 8. Limited interpretability  
### 8. 可解释性有限

Because the study mainly uses historical prices, it cannot explain which economic or physical variables drive the forecasts.

The dissertation can improve interpretability through:

- SHAP
- Permutation importance
- Feature-group ablation
- Modality-level contribution analysis
- Temporal attention analysis

由于论文主要使用历史价格，因此无法解释哪些经济或物理变量驱动预测。

本项目可以通过以下方法提高可解释性：

- SHAP；
- Permutation importance；
- 特征组消融实验；
- 模态贡献分析；
- 时间注意力分析。

---

## Research Gap Created by the Paper  
## 该论文留下的研究缺口

The paper demonstrates that historical prices contain useful short-term forecasting information.

However, it does not establish whether physical and informational signals provide incremental forecasting value beyond price history.

This creates a direct research opportunity:

> Can macro-financial, textual, shipping, remote-sensing, and spatial-network information improve Brent crude-oil forecasts beyond strong price-only TCN and LightGBM baselines?

论文证明历史价格包含一定的短期预测信息。

但它没有检验物理供应链信号和信息信号能否在历史价格之外提供额外预测价值。

因此形成了一个直接研究机会：

> 宏观金融、文本、航运、遥感和空间网络信息，能否在强纯价格TCN和LightGBM基线之上进一步改善Brent原油价格预测？

---

## Notes for Dissertation Integration  
## 在毕业论文中的使用方式

### Literature Review

Use the paper in a subsection such as:

**Price-Only Machine-Learning and Deep-Learning Approaches**

The paper can support the following statements:

1. Deep-learning models can capture nonlinear structures in commodity-price time series.
2. TCN is a competitive model for short-term oil-price forecasting.
3. Tree-based boosting models remain strong alternatives to neural networks.
4. Window length is an important modelling decision.
5. Complex hybrid architectures do not necessarily outperform simpler models.
6. Existing price-only studies do not incorporate physical supply-chain information.

在文献综述中，可以将该论文放入以下小节：

**基于纯价格序列的机器学习与深度学习方法**

该论文可以支持以下论点：

1. 深度学习可以捕捉商品价格序列中的非线性结构；
2. TCN是短期油价预测中的有竞争力模型；
3. 基于Boosting的树模型仍然是神经网络的重要替代方案；
4. 历史窗口长度是重要的建模决策；
5. 复杂混合模型不一定优于简单模型；
6. 现有纯价格研究没有纳入物理供应链信息。

### Methodology

The dissertation can adopt:

- Chronological data splitting
- Sliding-window construction
- TCN as a sequence benchmark
- LightGBM as a tabular benchmark
- Repeated neural-network training
- Early stopping
- MAE and RMSE

The dissertation should improve upon the paper by adding:

- Random-walk benchmark
- Rolling or expanding-window validation
- Train-only scaling
- Multi-horizon forecasting
- MASE and out-of-sample \(R^2\)
- Diebold–Mariano tests
- SHAP and modality ablation
- Release-lag-aware feature alignment

本项目可以借鉴：

- 按时间顺序划分数据；
- 构造滑动窗口；
- 将TCN作为序列模型基线；
- 将LightGBM作为表格数据基线；
- 重复训练神经网络；
- Early stopping；
- MAE和RMSE。

同时应在以下方面改进：

- 加入随机游走基线；
- 使用滚动或扩展窗口验证；
- 仅使用训练集计算归一化参数；
- 开展多期限预测；
- 使用MASE和样本外 \(R^2\)；
- 进行Diebold–Mariano检验；
- 使用SHAP和模态消融；
- 根据数据真实发布时间处理特征对齐。

### Results

A suitable results table is:

| Model | M0 Price | M1 Financial | M2 Text | M3 Shipping | M4 RS/Graph |
|---|---:|---:|---:|---:|---:|
| Random walk | Error | — | — | — | — |
| XGBoost / LightGBM | Error | Error | Error | Error | Error |
| TCN | Error | Error | Error | Error | Error |
| TFT | Error | Error | Error | Error | Error |
| ST-GNN | — | — | — | — | Error |

The incremental improvement can be calculated as:

\[
\text{Improvement}_{M_k}
=
\frac{
Error(M0)-Error(M_k)
}{
Error(M0)
}
\times 100
\]

建议的结果表为：

| 模型 | M0价格 | M1金融 | M2文本 | M3航运 | M4遥感/图 |
|---|---:|---:|---:|---:|---:|
| Random walk | 误差 | — | — | — | — |
| XGBoost / LightGBM | 误差 | 误差 | 误差 | 误差 | 误差 |
| TCN | 误差 | 误差 | 误差 | 误差 | 误差 |
| TFT | 误差 | 误差 | 误差 | 误差 | 误差 |
| ST-GNN | — | — | — | — | 误差 |

边际改善可以计算为：

\[
\text{Improvement}_{M_k}
=
\frac{
Error(M0)-Error(M_k)
}{
Error(M0)
}
\times 100
\]

---

## Suggested Literature Matrix Entry  
## 建议的文献矩阵条目

| Field | Entry |
|---|---|
| ID | P001 |
| Authors | Foroutan & Lahmiri |
| Year | 2024 |
| Full title | Deep learning systems for forecasting the prices of crude oil and precious metals |
| Dataset | Daily WTI, Brent, Gold and Silver spot closing prices, 2000-01-04 to 2022-03-25 |
| Variables | Historical price levels only; sliding windows of 5, 30, 60 and 90 days; no external predictors |
| Models | 12 DL models and 4 ML models: LSTM, BiLSTM, GRU, BiGRU, T2V models, CNN models, TCN models, Random Forest, LightGBM, SVR and KNN |
| Target | Next-day commodity spot-price level |
| Best results | TCN best for WTI, Brent and Silver; BiGRU best for Gold; LightGBM strongest ML model |
| Main contribution | Broad comparison of price-only sequence and machine-learning models |
| Main limitation | No macroeconomic, fundamental, textual, shipping or remote-sensing variables |
| Dissertation relevance | M0 price-only benchmark; supports TCN, LightGBM and window-sensitivity analysis |
| Priority | High for methodology; medium for variable selection |
| Recommended section | Literature Review — Price-only ML/DL forecasting; Methodology — model benchmarks |

---

## One-Sentence Summary  
## 一句话总结

This paper shows that TCN and LightGBM are strong models for next-day commodity price-level forecasting using historical prices alone, but its price-only design leaves open the central question of whether macroeconomic, textual, shipping, and remote-sensing signals provide incremental predictive value.

该论文表明，TCN和LightGBM在仅使用历史价格预测下一日商品价格水平时具有较强表现，但其纯价格设计没有回答宏观、文本、航运和遥感信号是否能够提供额外预测价值这一核心问题。

---

## Overall Assessment  
## 总体评价

- **Relevance to oil-price forecasting**: High  
  **与油价预测的相关性**：高

- **Relevance to model selection**: High  
  **与模型选择的相关性**：高

- **Relevance to M1 financial-variable selection**: Low to medium  
  **与M1金融变量选择的相关性**：低至中等

- **Relevance to multimodal forecasting**: Indirect but important  
  **与多模态预测的相关性**：间接但重要

- **Recommended role**: Strong M0 price-only methodological benchmark  
  **建议定位**：M0纯价格模型的强方法基准

- **Main models worth adopting**: TCN and LightGBM  
  **最值得借鉴的模型**：TCN和LightGBM

- **Main design worth adopting**: Multiple look-back windows and chronological validation  
  **最值得借鉴的实验设计**：多历史窗口比较和按时间验证

- **Main weakness to address**: Lack of external physical, financial and informational predictors  
  **项目需要弥补的主要不足**：缺少外部物理、金融和信息类预测变量