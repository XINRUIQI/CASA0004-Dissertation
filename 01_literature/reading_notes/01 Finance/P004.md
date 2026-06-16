# Reading Note — P004: WTI Crude Spot Price Forecasting with LSTM–XGBoost  
# 阅读笔记 — P004：基于 LSTM–XGBoost 的 WTI 原油现货价格预测

## Citation / 文献信息

Simsek, A. I., Bulut, E., Gur, Y. E., & Gültekin Tarla, E. (2024). A novel approach to Predict WTI crude spot oil price: LSTM-based feature extraction with Xgboost Regressor. *Energy*, 309, 133102.

- **DOI**: [10.1016/j.energy.2024.133102](https://doi.org/10.1016/j.energy.2024.133102)
- **Received / 收稿日期**: 19 January 2024
- **Revised / 修订日期**: 27 August 2024
- **Accepted / 接收日期**: 4 September 2024
- **Published online / 在线发表日期**: 10 September 2024
- **Journal / 期刊**: *Energy*
- **Target commodity / 预测对象**: WTI crude oil spot price / WTI 原油现货价格

---

## Research Objective / 研究目标

### English

The paper develops a hybrid forecasting model that combines Long Short-Term Memory networks with XGBoost to predict the WTI crude oil spot price.

The central idea is to use LSTM to extract temporal and nonlinear representations from macro-financial time-series variables and then use XGBoost as the final regression model.

The authors examine whether this hybrid architecture can outperform standalone machine-learning, deep-learning, and reinforcement-learning models.

### 中文

本文构建了一个结合长短期记忆网络与 XGBoost 的混合预测模型，用于预测 WTI 原油现货价格。

其核心思路是先利用 LSTM 从宏观金融时间序列中提取时间依赖关系和非线性隐含特征，再使用 XGBoost 完成最终的价格回归预测。

作者重点检验该混合架构是否能够优于单独的机器学习、深度学习和强化学习模型。

---

## Dataset / 数据集

### English

- **Frequency**: Monthly
- **Period**: January 1986 to May 2023
- **Number of observations**: 449
- **Training-test split**: 80% training and 20% testing
- **Training observations**: 359
- **Testing observations**: 90
- **Missing values**: The authors report no missing observations.
- **Scaling**: Min-Max normalisation to the range from 0 to 1

### 中文

- **数据频率**：月频
- **研究时期**：1986 年 1 月至 2023 年 5 月
- **观测数量**：449
- **训练测试划分**：80% 训练集，20% 测试集
- **训练集数量**：359
- **测试集数量**：90
- **缺失值**：作者称数据不存在缺失值
- **标准化方法**：Min-Max 归一化，将变量转换到 0 至 1 区间

---

## Variables / 变量

### Target Variable / 目标变量

- **WTI crude oil spot price**
- **WTI 原油现货价格**

Source / 来源:

- U.S. Energy Information Administration, EIA
- 美国能源信息署

### Predictor Variables / 预测变量

1. **US Dollar Index**
   - Source: Investing.com
   - 美元指数
   - 用于控制美元价值变化对以美元计价的原油价格所产生的影响。

2. **Gold Price**
   - Source: Investing.com
   - 黄金价格
   - 用于反映大宗商品市场联动、避险需求和通胀预期。

3. **Index of Global Real Economic Activity**
   - Source: Federal Reserve Bank of Dallas
   - 全球实际经济活动指数
   - 用于反映全球商品需求和实体经济活动变化。

4. **US 10-Year Bond Yield**
   - Source: Federal Reserve Bank of St. Louis
   - 美国 10 年期国债收益率
   - 用于反映长期利率、货币金融环境和宏观预期。

### Important Clarification / 重要说明

The paper does **not** use technical indicators such as:

- Moving Average, MA
- Exponential Moving Average, EMA
- Relative Strength Index, RSI
- Moving Average Convergence Divergence, MACD
- Bollinger Bands

The paper also does not clearly report the use of explicit WTI price-lag variables.

本文**没有**使用以下技术指标：

- 移动平均线 MA
- 指数移动平均线 EMA
- 相对强弱指数 RSI
- MACD
- 布林带

论文也没有清楚说明是否将 WTI 历史价格滞后项作为单独的输入特征。

---

## Core Method / 核心方法

### Proposed Architecture / 提出的模型架构

```text
Raw monthly macro-financial data
            ↓
Min-Max normalisation
            ↓
Training-test split
            ↓
RandomSearch hyperparameter optimisation
            ↓
LSTM temporal feature extraction
            ↓
Extracted latent time-series features
            ↓
XGBoost regression
            ↓
WTI spot-price prediction
            ↓
SHAP interpretation
```

```text
原始月度宏观金融数据
            ↓
Min-Max 归一化
            ↓
训练集与测试集划分
            ↓
RandomSearch 超参数优化
            ↓
LSTM 时间序列特征提取
            ↓
隐含时间序列特征
            ↓
XGBoost 回归
            ↓
WTI 现货价格预测
            ↓
SHAP 模型解释
```

### Role of LSTM / LSTM 的作用

#### English

LSTM is used primarily as a feature extractor rather than as the final forecasting layer.

It is intended to learn:

- long-term temporal dependencies;
- nonlinear patterns;
- interactions across macro-financial variables;
- latent representations of the time series.

#### 中文

本文中的 LSTM 主要作为特征提取器，而不是最终的预测层。

它被用于学习：

- 长期时间依赖关系；
- 非线性模式；
- 不同宏观金融变量之间的交互关系；
- 时间序列中的隐含特征表达。

### Role of XGBoost / XGBoost 的作用

#### English

The features extracted by the LSTM are passed to XGBoost, which performs the final regression.

The authors argue that XGBoost provides:

- strong nonlinear modelling capacity;
- regularisation against overfitting;
- effective learning from complex feature interactions;
- high predictive accuracy.

#### 中文

LSTM 提取出的特征被输入 XGBoost，由 XGBoost 完成最终的回归预测。

作者认为 XGBoost 具有以下优势：

- 较强的非线性建模能力；
- 通过正则化降低过拟合风险；
- 能够学习复杂的变量交互关系；
- 具有较高的预测精度。

### Hyperparameter Optimisation / 超参数优化

- **Method**: RandomSearch
- **Cross-validation**: 15-fold cross-validation
- **方法**：随机搜索
- **交叉验证**：15 折交叉验证

The authors prefer RandomSearch because it explores a wide hyperparameter space with lower computational cost than exhaustive GridSearch.

作者认为，RandomSearch 可以随机探索较广的超参数空间，相比穷举式 GridSearch 具有更低的计算成本。

---

## Benchmark Models / 对比模型

The proposed LSTM–XGBoost model is compared with five standalone models:

本文将 LSTM–XGBoost 混合模型与以下五种单独模型进行比较：

1. **DDPG**
   - Deep Deterministic Policy Gradient
   - 深度确定性策略梯度强化学习模型

2. **LSTM**
   - Long Short-Term Memory
   - 长短期记忆网络

3. **SVR**
   - Support Vector Regression
   - 支持向量回归

4. **CNN**
   - Convolutional Neural Network
   - 卷积神经网络

5. **XGBoost**
   - Standalone XGBoost regressor
   - 单独的 XGBoost 回归模型

---

## Evaluation Metrics / 评估指标

The models are evaluated using:

模型使用以下指标进行评估：

- **MAE**: Mean Absolute Error / 平均绝对误差
- **MSE**: Mean Squared Error / 均方误差
- **RMSE**: Root Mean Squared Error / 均方根误差
- **MAPE**: Mean Absolute Percentage Error / 平均绝对百分比误差
- **R²**: Coefficient of Determination / 决定系数

Lower MAE, MSE, RMSE and MAPE values indicate better performance, whereas a higher R² indicates stronger explanatory or predictive fit.

MAE、MSE、RMSE 和 MAPE 越低，表示预测误差越小；R² 越高，表示模型拟合或预测表现越好。

---

## Main Results / 主要结果

### Test-Set Performance / 测试集表现

| Model | R² | MAE | MAPE | MSE | RMSE |
|---|---:|---:|---:|---:|---:|
| DDPG | 0.8660 | 0.0545 | 0.1468 | 0.0071 | 0.0840 |
| LSTM | 0.8546 | 0.0652 | 0.1983 | 0.0077 | 0.0875 |
| SVR | 0.8999 | 0.0559 | 0.2038 | 0.0053 | 0.0726 |
| CNN | 0.8498 | 0.0666 | 0.2091 | 0.0079 | 0.0889 |
| XGBoost | 0.8790 | 0.0540 | 0.1567 | 0.0064 | 0.0798 |
| **LSTM–XGBoost** | **0.9991** | **0.0037** | **0.0099** | **0.00004** | **0.0070** |

### Finding 1: Hybrid Model Dominates the Benchmarks  
### 发现一：混合模型显著优于对比模型

#### English

The proposed LSTM–XGBoost model reports the best performance across all evaluation metrics.

Its reported test-set R² is 0.9991, compared with:

- 0.8999 for SVR;
- 0.8790 for standalone XGBoost;
- 0.8660 for DDPG;
- 0.8546 for LSTM;
- 0.8498 for CNN.

The results suggest that combining temporal representation learning with tree-based nonlinear regression substantially improves model fit within the authors' experimental design.

#### 中文

提出的 LSTM–XGBoost 模型在所有评估指标中均取得最佳结果。

其测试集 R² 达到 0.9991，而其他模型分别为：

- SVR：0.8999；
- 单独 XGBoost：0.8790；
- DDPG：0.8660；
- LSTM：0.8546；
- CNN：0.8498。

在作者设定的实验框架下，结果表明，将 LSTM 的时间特征学习能力与 XGBoost 的非线性回归能力结合，可以明显提高模型拟合表现。

---

### Finding 2: LSTM Feature Extraction Improves XGBoost  
### 发现二：LSTM 特征提取提高了 XGBoost 表现

#### English

Standalone XGBoost performs reasonably well, but its reported test R² of 0.8790 is substantially below the 0.9991 reported for the hybrid model.

The authors interpret this difference as evidence that the latent temporal features extracted by the LSTM provide useful information that is not fully represented in the original variables.

#### 中文

单独 XGBoost 已表现出一定的预测能力，但其测试集 R² 为 0.8790，明显低于混合模型报告的 0.9991。

作者据此认为，LSTM 提取的隐含时间特征包含了原始变量中未被 XGBoost 直接充分利用的信息。

---

### Finding 3: SHAP Identifies Important Macro-Financial Predictors  
### 发现三：SHAP 揭示重要宏观金融变量

#### English

The SHAP analysis indicates that the following variables have substantial effects on the model output:

- US Dollar Index;
- Gold Price;
- US 10-Year Bond Yield.

The Global Real Economic Activity Index generally has SHAP values closer to zero and therefore appears to have a smaller contribution in the fitted model.

The paper reports that high values of the US Dollar Index, Gold Price and US 10-Year Bond Yield are generally associated with positive SHAP contributions, while lower values tend to produce negative contributions.

#### 中文

SHAP 分析显示，以下变量对模型输出具有较明显的影响：

- 美元指数；
- 黄金价格；
- 美国 10 年期国债收益率。

全球实际经济活动指数的 SHAP 值通常更接近零，因此在该模型中的贡献相对较弱。

论文报告称，较高的美元指数、黄金价格和美国 10 年期国债收益率通常对应正向 SHAP 贡献，而较低取值通常对应负向贡献。

---

### Finding 4: The Model Appears to Fit High-Volatility Periods  
### 发现四：模型能够拟合高波动时期

#### English

The authors argue that the model successfully represents oil-price movements during major episodes, including:

- the late-1980s oil-market disturbances;
- the 2008 global financial crisis;
- the mid-2010s oil-price collapse;
- the 2020 COVID-19 pandemic.

They interpret this as evidence that the model can capture nonlinear relationships during periods of elevated volatility.

However, the paper does not conduct a separate crisis-period holdout test or formal regime-specific evaluation.

#### 中文

作者认为，该模型能够较好地表示以下重大事件期间的油价变化：

- 20 世纪 80 年代末的石油市场冲击；
- 2008 年全球金融危机；
- 2010 年代中期的油价崩盘；
- 2020 年新冠疫情。

作者据此认为，模型能够捕捉高波动时期的复杂非线性关系。

但是，论文没有单独设置危机时期留出样本，也没有进行正式的分阶段或机制转换检验。

---

## SHAP Interpretation / SHAP 解释

### English

SHAP is used to estimate the contribution of each predictor to individual model predictions.

This allows the authors to examine:

- which variables exert the strongest influence;
- whether high and low feature values have different directions of influence;
- how predictor contributions vary across observations;
- whether the model captures complex macro-financial relationships.

### 中文

SHAP 用于估计每一个预测变量对单次模型预测结果的边际贡献。

作者借此分析：

- 哪些变量对模型影响最大；
- 变量高值和低值是否具有不同的影响方向；
- 特征贡献是否会随不同观测值发生变化；
- 模型是否能够学习复杂的宏观金融联系。

### Important Interpretation Warning / 重要解释注意事项

SHAP values describe the behaviour of the fitted model. They do not establish that a predictor causally changes oil prices.

SHAP 值反映的是训练后模型的决策行为，并不能证明某个变量会因果性地改变油价。

---

## Relevance to This Dissertation / 对本论文项目的借鉴意义

| Aspect / 方面 | Connection to the Dissertation / 与本项目的联系 |
|---|---|
| **Model architecture / 模型架构** | The LSTM-feature-extraction plus XGBoost-regression structure provides an advanced hybrid-model option after the standalone XGBoost baselines have been completed. / LSTM 特征提取加 XGBoost 回归可作为完成单独 XGBoost 基准模型后的高级混合模型。 |
| **M1 variable selection / M1 变量选择** | The paper directly supports the inclusion of the US Dollar Index, Gold Price, US 10-Year Treasury Yield and a global economic activity indicator in the financial and macroeconomic feature group. / 论文可直接支持在金融与宏观变量组中加入美元指数、黄金价格、美国 10 年期国债收益率和全球经济活动指标。 |
| **Standalone XGBoost / 单独 XGBoost** | Because the paper evaluates standalone XGBoost separately, it supports using XGBoost as a common model across M1, M1+M2, M1+M2+M3 and M1+M2+M3+M4 experiments. / 由于论文单独评估了 XGBoost，因此可支持使用统一的 XGBoost 模型比较 M1 至 M4 不同模态组合。 |
| **Temporal feature extraction / 时间特征提取** | LSTM can be used to summarise rolling sequences of weekly market, text, shipping and remote-sensing features before prediction. / LSTM 可以用于提取周度金融、文本、航运和遥感变量序列中的时间特征。 |
| **Interpretability / 可解释性** | SHAP provides a template for analysing whether financial, NLP, shipping or remote-sensing features contribute most strongly to the final forecast. / SHAP 可用于分析金融、NLP、航运和遥感特征中，哪一种模态对最终预测贡献最大。 |
| **Nonlinear relationships / 非线性关系** | The paper supports the argument that oil-price relationships may be nonlinear and difficult to capture using conventional linear models alone. / 该论文支持油价与宏观金融变量之间可能存在复杂非线性关系的观点。 |
| **Crisis analysis / 危机分析** | The dissertation can extend the paper by examining whether modality importance changes during the 2008 crisis, the 2014–2016 oil-price collapse, COVID-19 and geopolitical disruptions. / 本项目可以进一步检验金融危机、油价暴跌、疫情和地缘政治冲突时期，不同模态的重要性是否发生变化。 |

---

## Suggested M1 Mapping / 对 M1 的变量映射

| Paper Variable / 论文变量 | Possible Dissertation Variable / 项目对应变量 | Role / 作用 |
|---|---|---|
| US Dollar Index | FRED Broad US Dollar Index or DXY | Exchange-rate and commodity-pricing control / 汇率与美元计价效应 |
| Gold Price | Gold spot price or gold return | Commodity and safe-haven indicator / 大宗商品与避险信号 |
| US 10-Year Bond Yield | FRED DGS10 | Interest-rate and macro-expectation control / 长期利率和宏观预期 |
| Global Real Economic Activity | Kilian Index or alternative global activity index | Global oil-demand proxy / 全球石油需求代理变量 |
| WTI Spot Price | Brent spot price or Brent weekly return | Replace the original prediction target / 替换原论文预测目标 |

---

## Limitations / 局限性

### 1. Small Sample Size  
### 1. 样本量较小

#### English

The model uses only 449 monthly observations. This is a limited sample for estimating and validating a complex LSTM–XGBoost architecture.

The extremely high reported R² may partly reflect the small dataset, the persistence of oil-price levels, or the validation design.

#### 中文

本文仅使用 449 个月度观测值。对于结构较复杂的 LSTM–XGBoost 模型而言，该样本量相对有限。

论文报告的极高 R² 可能部分受到小样本、油价水平序列高度持续性以及验证方式的影响。

---

### 2. Forecast Horizon Is Not Clearly Defined  
### 2. 预测步长定义不清晰

#### English

The paper does not clearly specify whether the model predicts:

- the same-period WTI price;
- the next month's price;
- a multi-step-ahead price;
- or a target generated using a particular lag structure.

This makes it difficult to determine whether the task is genuine ex-ante forecasting or contemporaneous prediction.

#### 中文

论文没有清楚说明模型预测的是：

- 同期 WTI 价格；
- 下一个月的油价；
- 多步未来油价；
- 还是通过某种滞后结构构建的目标。

因此，难以判断该任务是真正的事前预测，还是使用同期变量进行价格拟合。

---

### 3. Potential Data Leakage  
### 3. 潜在的数据泄漏风险

#### English

The methodology describes normalisation before discussing the training-test split.

If the Min-Max scaler was fitted using the complete dataset, information from the test period would have entered the preprocessing stage.

The dissertation should fit all scalers only on the training sample and then apply the fitted transformation to validation and test data.

#### 中文

论文的方法部分先描述数据归一化，随后才描述训练集与测试集划分。

如果 Min-Max scaler 使用了完整数据集进行拟合，那么测试时期的信息可能已经进入预处理过程。

本项目应只在训练集上拟合 scaler，再将训练阶段得到的转换参数应用到验证集和测试集。

---

### 4. Ordinary K-Fold Validation May Be Unsuitable for Time Series  
### 4. 普通 K 折验证可能不适用于时间序列

#### English

The authors use 15-fold cross-validation but do not clearly state that the folds preserve temporal order.

Random K-fold validation can mix earlier and later observations, allowing future information to influence model selection.

A forecasting dissertation should instead use:

- chronological train-validation-test splits;
- expanding-window validation;
- rolling-window validation;
- or `TimeSeriesSplit`.

#### 中文

作者使用了 15 折交叉验证，但没有清楚说明交叉验证是否保持了时间顺序。

随机 K 折可能会混合早期和后期观测，使未来信息影响模型选择。

本项目应改用：

- 按时间顺序划分训练集、验证集和测试集；
- 扩展窗口验证；
- 滚动窗口验证；
- 或 `TimeSeriesSplit`。

---

### 5. Inconsistent Metric Scales  
### 5. 评估指标尺度不一致

#### English

The training and test errors in Table 3 appear to be reported on the normalised scale, while several cross-validation MAE, MSE and RMSE values appear to be reported on the original price scale.

This makes direct comparison across training, testing and cross-validation results difficult.

#### 中文

表 3 中训练集和测试集的误差似乎是在归一化尺度上报告，而部分交叉验证的 MAE、MSE 和 RMSE 则似乎使用了原始价格尺度。

因此，训练、测试和交叉验证结果之间难以进行直接比较。

---

### 6. Extremely High Reported Accuracy Requires Caution  
### 6. 极高预测精度需要谨慎解释

#### English

A test R² of 0.9991 and an RMSE of 0.007 are unusually strong for crude-oil forecasting.

The paper does not provide sufficient robustness tests to rule out:

- data leakage;
- contemporaneous predictor use;
- inappropriate cross-validation;
- target misalignment;
- or overfitting.

The reported performance should therefore be treated as evidence within the paper's experimental setting rather than as a guaranteed level of real-world forecasting accuracy.

#### 中文

对于原油价格预测而言，测试集 R² 为 0.9991、RMSE 为 0.007 是非常高的结果。

论文没有提供足够的稳健性检验来排除以下可能性：

- 数据泄漏；
- 使用同期预测变量；
- 不合适的交叉验证；
- 目标变量与特征时间错位；
- 过拟合。

因此，这些结果更适合被理解为论文特定实验设定下的表现，而不能直接视为真实应用中的可实现预测精度。

---

### 7. Weak Traditional Baseline Design  
### 7. 传统基准模型不足

#### English

The comparison includes several machine-learning and deep-learning models, but does not include important oil-price forecasting benchmarks such as:

- random walk;
- no-change forecast;
- historical mean;
- ARIMA or ARIMAX;
- VAR;
- futures-based forecasts.

This weakens the claim that the hybrid model is superior to conventional forecasting methods.

#### 中文

论文虽然比较了多种机器学习和深度学习模型，但没有加入以下常见油价预测基准：

- 随机游走；
- 无变化预测；
- 历史均值；
- ARIMA 或 ARIMAX；
- VAR；
- 基于期货价格的预测。

因此，论文关于混合模型优于传统预测方法的结论缺少充分的基准支持。

---

### 8. Limited Predictor Set  
### 8. 输入变量较少

#### English

Only four macro-financial predictors are included.

The model excludes several potentially important factors, such as:

- oil production;
- inventories;
- consumption and demand;
- futures spreads;
- OPEC decisions;
- geopolitical conflict;
- sanctions;
- shipping activity;
- refinery disruptions;
- weather shocks;
- textual sentiment.

#### 中文

模型只使用了四个宏观金融预测变量。

以下可能影响油价的重要因素没有进入模型：

- 石油产量；
- 库存；
- 消费与需求；
- 期货价差；
- OPEC 决策；
- 地缘政治冲突；
- 制裁；
- 航运活动；
- 炼厂中断；
- 极端天气；
- 文本情绪。

---

### 9. Only WTI Is Examined  
### 9. 仅研究 WTI

#### English

The model is developed only for WTI spot prices.

Its transferability to Brent is not established because Brent and WTI differ in:

- regional market structure;
- physical delivery locations;
- transportation constraints;
- benchmark composition;
- exposure to global seaborne trade.

#### 中文

本文仅预测 WTI 现货价格。

由于 Brent 与 WTI 在以下方面存在差异，因此不能默认模型能够直接迁移：

- 区域市场结构；
- 实物交割地点；
- 运输约束；
- 基准价格构成；
- 对全球海运贸易的暴露程度。

---

### 10. Monthly Frequency Only  
### 10. 仅使用月频数据

#### English

The study does not test weekly, daily or intraday horizons.

The dissertation's weekly Brent forecasting problem may involve shorter-lived signals from news, shipping and remote-sensing variables that monthly data cannot capture.

#### 中文

本文没有检验周频、日频或日内预测。

本项目的 Brent 周度预测可能包含新闻、航运和遥感数据中的短期信号，而这些信号在月度聚合后可能被削弱。

---

### 11. SHAP Does Not Demonstrate Causality  
### 11. SHAP 不代表因果关系

#### English

The SHAP analysis explains how the trained model uses the variables, but it does not establish causal relationships between the predictors and oil prices.

In addition, the paper does not fully explain how the LSTM's latent features are mapped back to the original variables in the SHAP analysis.

#### 中文

SHAP 分析只能解释训练后的模型如何使用变量，不能证明预测变量与油价之间存在因果关系。

此外，论文没有充分说明，在 LSTM 已经提取隐含特征之后，XGBoost 的 SHAP 值如何准确映射回四个原始变量。

---

### 12. No Statistical Forecast-Comparison Test  
### 12. 缺少预测差异显著性检验

#### English

The paper compares error metrics but does not report formal tests such as the Diebold-Mariano test.

It is therefore unclear whether the differences in forecast errors are statistically significant.

#### 中文

论文比较了不同模型的误差指标，但没有报告 Diebold-Mariano 等正式预测比较检验。

因此，无法确定不同模型预测误差之间的差异是否具有统计显著性。

---

## Relevance to M1–M4 Framework / 对 M1–M4 框架的意义

### M1 — Financial and Market Variables  
### M1 — 金融与市场变量

This paper is most directly relevant to M1.

It supports including:

- US Dollar Index;
- Gold Price;
- US 10-Year Treasury Yield;
- Global Economic Activity;
- lagged Brent prices or returns, added by this dissertation.

该论文与 M1 的联系最直接。

它支持纳入：

- 美元指数；
- 黄金价格；
- 美国 10 年期国债收益率；
- 全球经济活动；
- 由本项目另外加入的 Brent 历史价格或收益率滞后项。

---

### M2 — Text and Event Variables  
### M2 — 文本与事件变量

The paper does not contain NLP or event-text features.

The dissertation can extend it using:

- GDELT conflict and sanction indicators;
- OPEC report sentiment;
- EIA outlook signals;
- company-announcement sentiment;
- geopolitical-event embeddings.

本文不包含 NLP 或文本事件变量。

本项目可以扩展加入：

- GDELT 冲突与制裁指标；
- OPEC 报告情绪；
- EIA 市场展望信号；
- 公司公告情绪；
- 地缘政治事件嵌入特征。

---

### M3 — Shipping and Port Activity  
### M3 — 航运与港口活动

The paper does not use shipping variables.

The dissertation can test whether port calls, tanker presence and chokepoint activity provide incremental forecasting value beyond the four macro-financial variables.

本文没有使用航运变量。

本项目可以检验港口挂靠、油轮活动和咽喉航道流量，是否能够在四个宏观金融变量之外提供额外预测能力。

---

### M4 — Remote Sensing  
### M4 — 遥感变量

The paper does not use satellite or spatial observations.

The dissertation can add storage, infrastructure activity or night-time-light indicators and test whether they provide information not captured by conventional market data.

本文不包含卫星影像或空间观测数据。

本项目可以加入石油储存、基础设施活动或夜间灯光等遥感指标，并检验这些变量是否提供传统市场数据未能覆盖的信息。

---

## Recommended Dissertation Use / 论文中建议使用方式

### Literature Review / 文献综述

Use the paper to support the following argument:

> Recent crude-oil forecasting studies increasingly combine deep temporal feature extraction with tree-based ensemble regression. Simsek et al. (2024), for example, use LSTM to extract latent time-series features from four macro-financial variables and subsequently apply XGBoost to predict monthly WTI spot prices. Their results indicate that the hybrid model outperforms standalone LSTM, CNN, SVR, DDPG and XGBoost models. However, the study is limited by a small monthly dataset, an unclear forecast horizon and insufficiently documented time-series validation.

可在文献综述中使用以下论述：

> 近期原油价格预测研究开始将深度时间特征提取与树模型集成回归相结合。例如，Simsek 等（2024）使用 LSTM 从四个宏观金融变量中提取隐含时间序列特征，并进一步使用 XGBoost 预测月度 WTI 现货价格。研究结果显示，混合模型优于单独的 LSTM、CNN、SVR、DDPG 和 XGBoost。然而，该研究存在月度样本量较小、预测步长定义不清以及时间序列验证过程说明不足等问题。

---

### Methodology / 方法部分

Use the paper as a reference for an optional advanced model:

```text
Weekly M1–M4 feature sequences
            ↓
LSTM or Temporal Encoder
            ↓
Latent temporal representation
            ↓
XGBoost regression
            ↓
One-week-ahead Brent forecast
```

可以将其作为高级混合模型的架构参考：

```text
M1–M4 周度特征序列
            ↓
LSTM 或其他时间编码器
            ↓
隐含时间特征表达
            ↓
XGBoost 回归
            ↓
提前一周的 Brent 预测
```

However, standalone XGBoost should be completed first so that the marginal contribution of each data modality remains interpretable.

但是，应先完成单独 XGBoost 的 M1–M4 消融实验，从而确保每一种数据模态的边际贡献仍然可以清楚解释。

---

### Explainability / 可解释性部分

The dissertation can use SHAP to compare:

- global feature importance;
- modality-level importance;
- feature importance by forecast horizon;
- feature importance during crisis and non-crisis periods;
- SHAP interaction effects;
- stability of feature rankings across rolling windows.

本项目可以使用 SHAP 比较：

- 全局特征重要性；
- 模态层面的重要性；
- 不同预测步长下的变量重要性；
- 危机期和非危机期的变量重要性；
- SHAP 交互效应；
- 滚动窗口中特征排名的稳定性。

---

## Proposed Improvements for Replication / 在本项目中应进行的改进

1. Replace WTI with Brent.
   - 将 WTI 替换为 Brent。

2. Use weekly rather than monthly data.
   - 使用周频而不是月频数据。

3. Define a clear forecasting target, such as:
   - 明确定义预测目标，例如：

```text
Target at week t = Brent return from week t to week t+1
```

4. Lag all predictors to ensure they are available at the forecast origin.
   - 对所有预测变量进行滞后处理，保证变量在预测时点已经可获得。

5. Fit scalers only on the training data.
   - 只在训练集上拟合标准化参数。

6. Use expanding-window or rolling-window validation.
   - 使用扩展窗口或滚动窗口验证。

7. Add conventional benchmarks:
   - 加入传统基准模型：

```text
Naive / no-change
Random walk
Historical mean
ARIMA or ARIMAX
Linear regression
Standalone XGBoost
```

8. Conduct ablation tests:
   - 进行模态消融实验：

```text
M1
M1 + M2
M1 + M2 + M3
M1 + M2 + M3 + M4
```

9. Report Diebold-Mariano tests.
   - 报告 Diebold-Mariano 检验。

10. Evaluate performance separately during major market regimes.
    - 分别评估不同市场机制和重大冲击时期的表现。

11. Compare the hybrid model against LSTM, TFT and other temporal models using the same information set.
    - 在相同输入变量条件下，将混合模型与 LSTM、TFT 等时间模型进行比较。

12. Report both price-level and return-based forecasting results.
    - 同时报告价格水平预测和收益率预测结果。

---

## Notes for Dissertation Integration / 论文写作整合备注

- Use this paper as a **hybrid-model architecture reference**.
- 将该论文作为**混合模型架构参考**。

- Use it as evidence supporting the inclusion of Dollar Index, Gold, US 10-Year Yield and Global Economic Activity in M1.
- 使用该论文支持在 M1 中加入美元指数、黄金、美国 10 年期国债收益率和全球经济活动指标。

- Use standalone XGBoost as the main M1–M4 comparison model before introducing the hybrid architecture.
- 在引入混合架构之前，先使用单独 XGBoost 完成 M1–M4 比较实验。

- Adopt SHAP for model explanation, but do not automatically retain only the top-ranked features.
- 可以采用 SHAP 解释模型，但不应机械地只保留排名最高的变量。

- Do not cite the paper as evidence that SHAP provides causal interpretation.
- 不应将该论文用于证明 SHAP 具有因果解释能力。

- Do not directly reproduce its random K-fold design.
- 不应直接复制其普通随机 K 折验证设计。

- Treat the reported R² of 0.9991 cautiously because the forecast horizon and temporal validation procedure are insufficiently documented.
- 由于预测步长和时间验证过程说明不足，应谨慎解释其报告的 0.9991 测试集 R²。

- The dissertation's main contribution is not simply replacing WTI with Brent. The stronger extension is to add NLP, shipping and remote-sensing modalities under a leakage-free rolling forecasting framework.
- 本项目的主要创新不应只是将 WTI 替换为 Brent，而应是在无数据泄漏的滚动预测框架下加入 NLP、航运和遥感等多模态数据。

---

## One-Sentence Summary / 一句话总结

### English

Simsek et al. (2024) propose an LSTM–XGBoost hybrid model for forecasting monthly WTI spot prices using four macro-financial predictors and report exceptionally high predictive accuracy, but unclear target alignment, possible preprocessing or validation leakage and inconsistent metric scales mean that its architecture is more transferable than its reported performance.

### 中文

Simsek 等（2024）提出了一个使用四个宏观金融变量预测月度 WTI 现货价格的 LSTM–XGBoost 混合模型，并报告了极高的预测精度；但由于预测目标时间对应关系不清、预处理或验证可能存在数据泄漏风险，以及评估指标尺度不一致，其模型架构的借鉴价值高于其所报告的具体预测表现。

