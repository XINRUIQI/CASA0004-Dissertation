# Reading Note — P076: Strategic Risk-Based Brent Crude Oil Forecasting  
# 阅读笔记 — P076：基于战略风险的 Brent 原油预测

## Citation / 文献信息

Yılmaz, T. E., & Zehir, C. (2026). Strategic Risk Based Forecasting of Brent Crude Oil Prices: A Comparative Analysis of Econometric and Machine Learning Models. *Entropy*, 28(5), 539.

- **DOI**: [10.3390/e28050539](https://doi.org/10.3390/e28050539)
- **Received / 收稿日期**: 10 April 2026
- **Accepted / 接收日期**: 8 May 2026
- **Published / 发表日期**: 9 May 2026
- **Journal / 期刊**: *Entropy*
- **Target market / 研究市场**: Brent crude oil
- **Forecasting target / 预测目标**: Monthly Brent logarithmic returns rather than the Brent price level  
  实际预测目标是 Brent 月度对数收益率，而不是美元/桶形式的价格水平。

---

## Research Objective / 研究目标

### English

This paper examines whether a small set of strategic risk indicators can improve the forecasting of Brent crude oil returns. It compares traditional econometric models with tree-based machine learning models and investigates whether nonlinear algorithms can capture relationships among oil returns, geopolitical risk, financial-market uncertainty, and long-term interest rates more effectively than linear models.

The study defines **strategic risk** as a multidimensional concept consisting of:

1. geopolitical risk;
2. financial-market uncertainty;
3. macro-financial and monetary conditions.

### 中文

本文研究少量具有明确经济含义的“战略风险指标”能否提高 Brent 原油收益率的预测精度，并比较传统计量经济模型与树模型机器学习方法的表现。

作者将 **strategic risk（战略风险）** 定义为三个相互关联但含义不同的风险渠道：

1. 地缘政治风险；
2. 金融市场不确定性；
3. 宏观金融和货币条件。

文章的核心问题是：Brent 收益率与这些风险指标之间是否存在传统线性模型难以捕捉的非线性关系、阈值效应和变量交互。

---

## Research Questions / 研究问题

1. **Do GPR, VIX, and U.S. long-term interest rates provide predictive information beyond historical Brent returns?**  
   在控制 Brent 历史收益率后，GPR、VIX 和美国长期利率是否仍能提供额外预测信息？

2. **Do machine learning models outperform traditional econometric models?**  
   机器学习模型是否优于 ARIMAX 和 ARIMAX-GARCH 等传统计量模型？

3. **Are the results stable across different forecasting horizons and train-test splits?**  
   模型排名在不同预测步长和训练集—测试集划分下是否稳定？

4. **Which internal and external variables contribute most to Brent return forecasting?**  
   Brent 自身滞后项和外部风险变量中，哪些变量对预测贡献最大？

---

## Dataset / 数据集

- **Frequency / 频率**: Monthly / 月度
- **Period / 时间范围**: January 2001 – December 2025  
  2001年1月—2025年12月
- **Number of observations / 观测数量**: 300 monthly observations / 约300个月度观测
- **Primary target source / 目标变量来源**: FRED Brent Europe spot price series, DCOILBRENTEU
- **Other FRED series / 其他 FRED 数据**:
  - VIXCLS: CBOE Volatility Index
  - DGS10: 10-Year Treasury Constant Maturity Rate
- **GPR source / GPR 来源**: Caldara–Iacoviello Geopolitical Risk dataset
- **Missing values / 缺失值处理**:
  - Missing daily observations were excluded when monthly averages were calculated.
  - No interpolation or statistical imputation was applied.
  - 月度聚合时忽略缺失的日度观测，没有采用插值或统计填补。

### Frequency alignment / 频率统一

Daily Brent prices, VIX, and DGS10 were converted into monthly averages. GPR was already available at monthly frequency.

Brent、VIX 和 DGS10 原本是日度数据，作者先将其转换为月平均值；GPR 原本就是月度指标。

### Important implication / 重要含义

Monthly aggregation ensures frequency consistency and reduces high-frequency noise, but it may smooth short-lived market shocks, abrupt geopolitical developments, and rapid financial-market reactions.

月度聚合有利于统一数据频率和降低噪声，但也可能削弱短期战争冲击、制裁宣布、金融恐慌和原油市场急剧调整等信息。

---

## Target Variable / 目标变量

The dependent variable is the monthly logarithmic return of Brent crude oil:

\[
r_t = 100 \times [\ln(P_t)-\ln(P_{t-1})]
\]

where:

- \(r_t\) is the monthly Brent return in percentage terms;
- \(P_t\) is the average Brent price in month \(t\);
- \(P_{t-1}\) is the average Brent price in the previous month.

中文解释：

- \(r_t\)：第 \(t\) 月的 Brent 对数收益率；
- \(P_t\)：第 \(t\) 月 Brent 日度价格的月平均值；
- \(P_{t-1}\)：前一个月的 Brent 月平均价格。

### Important clarification / 重要澄清

Although the title refers to “Brent crude oil prices”, the empirical model forecasts **Brent returns**, not the price level itself.

虽然论文标题写的是 Brent crude oil prices，但实证部分实际预测的是 **Brent 月度对数收益率**，并不是直接预测下一期 Brent 为多少美元一桶。

---

## Explanatory Variables / 解释变量

### 1. Brent return lags / Brent 收益率滞后项

The machine learning models use eight lags of Brent returns:

\[
r_{t-1}, r_{t-2}, \ldots, r_{t-8}
\]

These variables capture:

- short-term autocorrelation;
- momentum and reversal effects;
- persistence after market shocks;
- nonlinear temporal interactions;
- medium-term memory in oil returns.

机器学习模型使用 Brent 收益率的1至8期滞后项，用于捕捉短期记忆、动量、反转、冲击持续性和非线性时间依赖。

The ACF and PACF mainly indicate a short-memory process, with the strongest signal occurring at lag 1. Nevertheless, the authors retain eight lags to allow machine learning models to identify medium-term nonlinear relationships.

ACF 和 PACF 显示最明显的相关性主要出现在第一期，但作者仍保留8期滞后值，以便树模型学习更复杂的中期依赖关系。

---

### 2. GPR: Geopolitical Risk Index / 地缘政治风险指数

- **Source / 来源**: Caldara and Iacoviello
- **Frequency / 频率**: Monthly
- **Model input / 模型输入**: \(GPR_{t-1}\)
- **Transformation / 处理**: Used in its original index form after stationarity testing

GPR measures the intensity of newspaper discussions related to geopolitical tensions, wars, terrorism, military threats, and international conflict.

GPR 衡量新闻报道中与战争、恐怖主义、军事威胁、国际冲突和地缘政治紧张局势有关的内容强度。

The one-period lag is used to ensure that only information available before the forecasting month is included.

论文使用滞后一期的 GPR，目的是降低同期信息泄漏，使模型更接近真实预测，而不是事后解释或 nowcasting。

---

### 3. VIX: Financial-Market Uncertainty / 金融市场不确定性

- **Source / 来源**: FRED VIXCLS
- **Original frequency / 原始频率**: Daily
- **Aggregation / 聚合方法**: Monthly average
- **Model input / 模型输入**: \(VIX_{t-1}\)

VIX is used as a proxy for:

- investor risk aversion;
- expected equity-market volatility;
- financial stress;
- market fear;
- global uncertainty transmission.

VIX 用于反映投资者风险厌恶、金融市场恐慌、股票市场预期波动和全球金融压力。

The descriptive analysis shows that major increases in VIX correspond to periods such as the 2008 global financial crisis and the COVID-19 shock, when Brent returns also experienced large negative movements.

描述性结果显示，2008年金融危机和2020年疫情期间 VIX 急剧上升，同时 Brent 收益率出现明显的负向冲击。

---

### 4. DGS10: U.S. 10-Year Treasury Yield / 美国10年期国债收益率

- **Source / 来源**: FRED DGS10
- **Original frequency / 原始频率**: Daily
- **Aggregation / 聚合方法**: Monthly average
- **Economic meaning / 经济含义**:
  - long-term interest rates;
  - monetary conditions;
  - financing costs;
  - inflation expectations;
  - global liquidity conditions.

DGS10 水平反映美国长期利率、融资成本、货币政策预期、通胀预期和全球流动性条件。

### Stationarity treatment / 平稳性处理

The level of DGS10 did not pass the ADF and PP unit-root tests. The study therefore uses the first difference:

\[
\Delta DGS10_t = DGS10_t-DGS10_{t-1}
\]

DGS10 水平值没有通过 ADF 和 PP 单位根检验，因此实证模型实际使用的是利率的一阶差分 \(\Delta DGS10\)。

This distinction is important: the model captures recent changes in interest rates rather than relying only on the absolute level of the 10-year yield.

这意味着模型主要考察长期利率最近是上升还是下降，而不只是利率处于多高的绝对水平。

---

### 5. Rolling Shannon Entropy / 滚动 Shannon 熵

The study also constructs a rolling Shannon entropy measure from Brent returns.

It is intended to represent:

- informational complexity;
- disorder in the return-generating process;
- market unpredictability;
- uncertainty not fully captured by conventional volatility measures.

作者还根据 Brent 收益率构建滚动 Shannon entropy，用于衡量收益率生成过程的信息复杂度、无序程度和不可预测性。

The variable is included as:

\[
Entropy_{t-1}
\]

The correlation between entropy and the absolute value of Brent returns is reported as approximately \(-0.1505\), suggesting that entropy does not simply duplicate conventional volatility.

Entropy 与 Brent 绝对收益率的相关系数约为 \(-0.1505\)，说明它试图捕捉的信息复杂度并不完全等同于普通价格波动。

---

## Complete Machine Learning Feature Set / 完整机器学习特征集

The general machine learning specification is:

\[
r_t =
f(
r_{t-1},r_{t-2},\ldots,r_{t-8},
GPR_{t-1},
VIX_{t-1},
\Delta DGS10_{t-1},
Entropy_{t-1}
)
\]

The complete machine learning model therefore contains approximately 12 predictors:

- 8 Brent return lags;
- 3 strategic risk indicators;
- 1 entropy variable.

因此，该论文并不是“仅用3个变量预测油价”。完整的机器学习模型实际上包括约12个特征，其中价格自身滞后项占多数。

---

## Models / 模型

### 1. ARIMAX

The benchmark conditional-mean model is ARIMAX:

- the ARIMA component models the historical dependence of Brent returns;
- exogenous variables represent strategic risk factors.

Based on `auto.arima`, ACF, and PACF diagnostics, the selected specification is:

\[
ARIMAX(0,0,1)
\]

ARIMAX 是文章中的线性基准模型，用来检验 Brent 历史动态和外生风险变量之间的线性关系。

---

### 2. ARIMAX-gjrGARCH

Residual diagnostics show that the ARIMAX residuals contain significant ARCH effects and volatility clustering.

The authors compare:

- sGARCH;
- eGARCH;
- gjrGARCH.

According to the reported AIC, BIC, and log-likelihood values, gjrGARCH with a skewed Student's \(t\) distribution is selected.

ARIMAX 残差存在显著的条件异方差和波动聚集，因此作者在均值方程之外建立 GARCH 类条件方差模型。

gjrGARCH 可以捕捉：

- time-varying volatility;
- volatility persistence;
- heavy-tailed returns;
- asymmetric responses to positive and negative shocks.

需要注意，GARCH 的主要任务是预测条件方差，而不是直接提高收益率均值的点预测精度。

---

### 3. Random Forest

Random Forest represents the bagging approach.

Key characteristics:

- bootstrap resampling;
- random feature selection;
- multiple decorrelated decision trees;
- averaging across trees;
- variance reduction.

Random Forest 通过 Bootstrap 抽样和随机选择特征建立多棵相对独立的树，再对结果进行平均，主要作用是降低预测方差。

---

### 4. XGBoost

XGBoost is a gradient boosting model in which trees are built sequentially to correct previous prediction errors.

Main advantages:

- nonlinear relationship modelling;
- interaction detection;
- regularisation;
- flexible loss optimisation;
- relatively strong control of overfitting.

XGBoost 通过连续建立决策树修正前一轮误差，适合处理非线性、阈值效应和变量交互。

---

### 5. LightGBM

LightGBM is also a gradient boosting model but generally uses leaf-wise tree growth.

Potential advantages:

- efficient optimisation;
- flexible nonlinear fitting;
- interaction modelling;
- strong performance on structured tabular data.

LightGBM 通常优先分裂能带来最大损失下降的叶节点，在结构化表格数据中往往具有较强的预测能力。

Because the sample contains only about 300 observations, careful regularisation and hyperparameter restrictions are particularly important.

由于本文只有约300个月度观测，LightGBM 也存在明显的过拟合风险，需要严格控制叶节点数量、树深、学习率和最小叶节点样本量。

---

## Hyperparameter Tuning / 超参数调优

The machine learning models are tuned using predefined grid-search procedures.

Parameters considered include:

### XGBoost

- maximum tree depth;
- learning rate;
- subsampling ratio;
- column sampling ratio;
- minimum child weight;
- gamma.

### Random Forest

- number of candidate predictors at each split;
- minimum node size;
- number of trees.

### LightGBM

- number of leaves;
- learning rate;
- feature fraction;
- bagging fraction;
- regularisation parameters;
- minimum leaf size.

作者通过网格搜索比较低、中、高复杂度的候选参数组合，以寻找较低的样本外预测误差。

---

## Training and Validation / 训练与验证

### Chronological splits / 时间顺序划分

The data are divided chronologically rather than randomly.

Three train-test splits are examined:

- 80% training / 20% testing;
- 75% training / 25% testing;
- 70% training / 30% testing.

数据按照时间顺序划分，避免未来数据进入训练集。作者还通过三种训练—测试比例检查结果是否依赖某一次特定划分。

---

### Rolling-origin forecasting / 滚动预测

The models are evaluated using a rolling-origin or recursively expanding forecasting framework.

General process:

1. estimate the model using currently available historical data;
2. produce an out-of-sample forecast;
3. move the forecast origin forward;
4. re-estimate or update the model;
5. repeat until the test period is completed.

滚动预测比随机交叉验证更符合真实油价预测，因为每个预测时点只能使用当时已经公开的信息。

---

### Forecast horizons / 预测步长

The study evaluates:

- 1-step ahead;
- 3-step ahead;
- 6-step ahead;
- 9-step ahead.

在月度数据下，这大致对应未来1个月、3个月、6个月和9个月的预测。

---

## Evaluation Metrics / 评价指标

### Primary metric / 主要指标

The main forecast error metric is symmetric mean absolute percentage error:

\[
sMAPE =
\frac{1}{n}
\sum_{t=1}^{n}
\frac{2|y_t-\hat{y}_t|}
{|y_t|+|\hat{y}_t|}
\]

The study also uses the Diebold–Mariano test to assess whether differences in forecast errors are statistically significant.

文章主要使用 sMAPE 比较模型，并使用 Diebold–Mariano 检验判断模型之间的预测误差差异是否具有统计显著性。

### Caution / 注意

Because the target is a return series that frequently takes values close to zero, sMAPE may become unstable when both actual and predicted values are small.

由于收益率经常接近零，sMAPE 的分母可能过小，因此该指标不应成为收益率预测的唯一评价标准。

---

## Key Findings / 核心发现

### 1. LightGBM delivers the strongest overall forecasting performance  
### 1. LightGBM 整体预测表现最好

Across the majority of forecasting horizons and all three train-test configurations, LightGBM reports the lowest or nearly the lowest sMAPE.

在80/20、75/25和70/30三种时间划分下，LightGBM 在绝大多数预测步长中取得最低或接近最低的 sMAPE。

The result suggests that Brent returns are influenced by nonlinear relationships and interactions that are more effectively captured by gradient boosting than by linear econometric models.

该结果说明 Brent 收益率与战略风险变量之间可能存在非线性、阈值和交互作用，boosting 模型比纯线性模型更容易捕捉这些结构。

---

### 2. LightGBM significantly outperforms the competing models  
### 2. LightGBM 相对其他模型的优势具有统计显著性

For the 80/20 split and one-step-ahead forecast, the Diebold–Mariano results are:

| Comparison / 模型比较 | DM statistic | p-value |
|---|---:|---:|
| LightGBM vs. XGBoost | -2.6154 | 0.005739 |
| LightGBM vs. ARIMAX | -2.3963 | 0.009994 |
| LightGBM vs. Random Forest | -1.9290 | 0.029447 |
| LightGBM vs. ARIMAX-gjrGARCH | -3.5324 | 0.000421 |
| ARIMAX vs. ARIMAX-gjrGARCH | -1.4476 | 0.076698 |

Negative DM statistics indicate lower losses for the first model.

结果显示，LightGBM 在该设定下显著优于 XGBoost、ARIMAX、Random Forest 和 ARIMAX-gjrGARCH。

---

### 3. GARCH improves volatility representation but not point-return forecasts  
### 3. GARCH 改善波动建模，但没有明显改善收益率点预测

ARIMAX-gjrGARCH successfully captures:

- volatility clustering;
- time-varying variance;
- asymmetric shock effects;
- heavy-tailed behaviour.

However, its sMAPE is generally worse than ordinary ARIMAX and LightGBM.

这并不意味着 gjrGARCH 是无效模型，而是说明：

- GARCH 主要解释条件方差；
- 收益率点预测主要关注条件均值；
- 更准确的波动率模型不一定带来更准确的收益率均值预测。

---

### 4. Historical Brent returns remain highly important  
### 4. Brent 自身历史收益率仍然非常重要

The first lag of Brent returns is the most important individual feature in the LightGBM model.

这说明虽然文章强调外部战略风险，但模型的主要预测基础之一仍然是 Brent 自身的历史价格动态。

---

### 5. GPR is the most important external strategic-risk predictor  
### 5. GPR 是最重要的外部战略风险变量

Reported LightGBM feature importance:

| Feature / 特征 | Gain | Cover | Frequency |
|---|---:|---:|---:|
| Lag1 | 0.1160 | 0.0820 | 0.0827 |
| GPR | 0.1085 | 0.1036 | 0.1014 |
| Lag4 | 0.1007 | 0.1072 | 0.1053 |
| Lag6 | 0.0922 | 0.1057 | 0.1023 |
| Lag7 | 0.0915 | 0.0999 | 0.1013 |
| Lag8 | 0.0864 | 0.0883 | 0.0866 |
| Lag3 | 0.0827 | 0.0862 | 0.0906 |
| VIX | 0.0789 | 0.0809 | 0.0778 |
| Lag5 | 0.0734 | 0.0724 | 0.0699 |
| Lag2 | 0.0721 | 0.0655 | 0.0679 |
| ΔDGS10 | 0.0643 | 0.0708 | 0.0739 |
| Entropy | 0.0333 | 0.0376 | 0.0403 |

GPR is second only to Brent Lag1 and is more important than VIX, interest-rate changes, and entropy.

GPR 的 Gain 仅次于 Brent 第一阶滞后项，是所有外部变量中贡献最高的指标。

---

### 6. VIX and ΔDGS10 provide moderate predictive information  
### 6. VIX 和 ΔDGS10 具有中等程度的预测贡献

VIX and changes in the U.S. 10-year Treasury yield contribute to the model, but their importance is lower than that of GPR and several Brent lag variables.

VIX 和利率变化并不是无用变量，但在本文模型中，它们的重要性低于 GPR 和多个历史收益率滞后项。

---

### 7. Entropy provides only a small incremental improvement  
### 7. Entropy 只带来较小的增量改善

Entropy has the lowest reported gain among all predictors.

It contributes complementary information, but it is not a primary forecasting driver.

Entropy 对预测具有一定补充价值，但并不是本文模型中的主要预测变量。

---

## Ablation Analysis / 消融实验

The study compares three LightGBM feature sets:

| Feature set / 特征组合 | sMAPE |
|---|---:|
| Only Brent return lags / 仅收益率滞后项 | 1.4902 |
| Lags + strategic risk variables / 滞后项 + 战略风险变量 | 1.4789 |
| Lags + strategic risk variables + entropy / 再加入 entropy | 1.4718 |

### Interpretation / 解释

1. Historical Brent returns already contain substantial predictive information.  
   Brent 历史收益率本身已经包含较多预测信息。

2. GPR, VIX, and ΔDGS10 improve performance beyond historical lags.  
   加入 GPR、VIX 和 ΔDGS10 后，预测误差进一步下降。

3. Entropy adds a smaller additional gain.  
   Entropy 带来的进一步改善相对有限。

4. The improvement from external variables is positive but not extremely large.  
   外部风险变量具有增量价值，但改善幅度并不是非常巨大。

---

## Main Contribution / 主要贡献

### English

The paper contributes by integrating geopolitical risk, financial uncertainty, macro-financial conditions, historical oil returns, and an entropy-based complexity measure within a unified comparison of econometric and machine learning models.

Its main methodological contribution is the combination of:

- economically interpretable predictors;
- lagged information to reduce leakage;
- rolling-origin out-of-sample evaluation;
- multiple train-test splits;
- multiple forecasting horizons;
- model-comparison tests;
- feature importance;
- ablation analysis.

### 中文

本文的主要贡献是将地缘政治风险、金融不确定性、利率条件、历史油价动态和信息熵统一纳入计量经济与机器学习比较框架。

其较有价值的方法设计包括：

- 使用具有经济意义的变量；
- 所有预测变量采用滞后形式；
- 使用滚动样本外预测；
- 比较多种训练—测试划分；
- 比较多个预测步长；
- 使用 DM 检验；
- 报告特征重要性；
- 进行特征组消融实验。

---

## Relevance to This Dissertation / 对本项目的借鉴意义

| Aspect / 方面 | Connection to the dissertation / 与本项目的联系 |
|---|---|
| **M1 market and macro-financial variables / M1 市场与宏观金融变量** | The paper directly supports including VIX, geopolitical risk, and interest-rate variables in the structured market module. It also suggests using changes in DGS10 rather than relying only on its level. 该论文直接支撑在 M1 中加入 VIX、GPR 和利率指标，并提示 DGS10 应优先测试一阶差分或周度变化。 |
| **GPR variable / GPR 变量** | GPR is the most important external predictor in the LightGBM model. The dissertation should therefore consider adding the Caldara–Iacoviello GPR index as a benchmark geopolitical-risk feature. GPR 是文中最重要的外部变量，因此可作为本项目的基础地缘政治风险指标。 |
| **Model choice / 模型选择** | LightGBM should be evaluated alongside XGBoost rather than treated only as an optional secondary model. 本项目应将 LightGBM 与 XGBoost 并列纳入核心候选模型。 |
| **Econometric baseline / 计量基准** | ARIMAX provides an interpretable linear benchmark, while GARCH-type models are more appropriate as volatility benchmarks than as guaranteed improvements to return point forecasts. ARIMAX 可作为线性基准，GARCH 更适合作为波动率任务的基准。 |
| **Rolling validation / 滚动验证** | The rolling-origin design provides a direct template for the dissertation's weekly expanding-window or rolling-window evaluation. 滚动预测框架可直接用于本项目的周度样本外验证。 |
| **Multiple horizons / 多预测期** | The study supports comparing short-, medium-, and longer-horizon forecasts. 本项目可比较未来1周、4周和12周等预测期。 |
| **Feature engineering / 特征工程** | Price lags should remain a strong baseline before adding text, remote sensing, and shipping signals. 在加入文本、遥感和航运信息前，应先建立以 Brent 滞后项为核心的强基准。 |
| **Ablation study / 消融实验** | The paper's feature-group ablation offers a template for testing the incremental value of each modality. 可建立 M1、M1+M2、M1+M3、M1+M4 和 Full Model，比较各模态的增量贡献。 |
| **Interpretability / 可解释性** | Feature importance can be extended to SHAP, permutation importance, rolling SHAP, and regime-specific explanations. 本项目应在普通 Gain importance 基础上进一步使用 SHAP 和滚动解释。 |
| **M2 NLP module / M2 文本模块** | GPR is itself a news-based aggregate index, but it does not replace customised NLP features extracted from OPEC, EIA, GDELT, or company reports. GPR 可作为文本风险基准，但不能替代本项目自建的高频 NLP 特征。 |
| **M3 remote sensing / M3 遥感模块** | The paper provides no direct evidence for remote-sensing variables. Remote sensing must be justified through separate literature and dissertation ablation tests. 本文不能直接证明遥感数据有效，只能作为加入遥感前的结构化数据基准。 |
| **M4 shipping and port activity / M4 航运与港口活动** | The paper does not include AIS, PortWatch, tanker traffic, or chokepoint variables. These modalities should be evaluated as additional information beyond the strategic-risk baseline. 本文不涉及 AIS、港口活动或油轮流量，本项目可检验这些变量是否提供额外预测信息。 |

---

## Recommended Dissertation Adaptation / 建议在本项目中的具体应用

### Baseline hierarchy / 基准模型层级

1. **Naive benchmark**
   - zero-return forecast;
   - historical mean;
   - last-value forecast.

2. **Autoregressive baseline**
   - Brent return lags only.

3. **M1 structured model**
   - Brent lags;
   - VIX or OVX;
   - GPR;
   - U.S. dollar index;
   - ΔDGS10;
   - inventory and supply-demand variables.

4. **Multimodal extensions**
   - M1 + NLP;
   - M1 + remote sensing;
   - M1 + shipping;
   - M1 + supply-chain network features;
   - full multimodal model.

---

### Suggested model comparison / 建议模型比较

- Historical mean or zero-return benchmark;
- ARIMA or ARIMAX;
- Random Forest;
- XGBoost;
- LightGBM;
- optional LSTM, TCN, or Transformer if sample size and computational resources permit.

不能因为该论文中 LightGBM 最好，就预设它在本项目中也一定最好。所有模型应在统一的数据、时间窗口和评价指标下比较。

---

### Suggested forecasting horizons / 建议预测步长

For weekly data:

- 1-week ahead;
- 4-week ahead;
- 12-week ahead.

对于周度项目，可分别代表短期、月度和季度尺度的预测能力。

---

### Suggested evaluation metrics / 建议评价指标

For return regression:

- MAE;
- RMSE;
- out-of-sample \(R^2\);
- directional accuracy;
- Diebold–Mariano test.

For direction classification:

- accuracy;
- balanced accuracy;
- precision;
- recall;
- F1;
- MCC;
- AUC.

For volatility forecasting:

- QLIKE;
- MAE;
- RMSE.

本项目不应只使用 sMAPE，尤其是在目标变量为接近零的收益率时。

---

### Suggested ablation design / 建议消融实验

| Model | Inputs |
|---|---|
| B0 | Naive forecast |
| B1 | Brent historical lags |
| M1 | B1 + structured market and macro variables |
| M1 + M2 | Add NLP and event features |
| M1 + M3 | Add remote-sensing features |
| M1 + M4 | Add shipping and port features |
| Full | All modalities |
| Full − M2 | Remove NLP |
| Full − M3 | Remove remote sensing |
| Full − M4 | Remove shipping |

This design can determine whether each new modality contains incremental information beyond conventional price and macro-financial variables.

该设计可以判断文本、遥感和航运变量究竟是真的增加预测能力，还是仅仅重复已有的市场信息。

---

## Limitations / 局限性

### 1. Small sample size / 样本量较小

The study contains only about 300 monthly observations, while the machine learning models use approximately 12 predictors and multiple hyperparameters.

只有约300个月度观测，却同时使用多个特征和复杂树模型，存在较明显的过拟合风险。

---

### 2. Title-target mismatch / 标题与目标变量不完全一致

The title refers to crude oil price forecasting, but the empirical target is monthly Brent returns.

论文结果不能被直接解释为“LightGBM 最适合预测 Brent 美元价格水平”。

---

### 3. Monthly aggregation / 月度频率可能损失短期信息

Monthly averages may obscure:

- immediate war shocks;
- sanctions;
- OPEC announcements;
- sudden tanker rerouting;
- weekly inventory surprises;
- short-lived financial stress.

这使得论文对高频或周度预测的适用性受到限制。

---

### 4. sMAPE for returns / sMAPE 对收益率可能不稳定

Returns frequently approach zero, which may create unstable percentage-based errors.

本项目应增加 MAE、RMSE、方向准确率和样本外 \(R^2\)。

---

### 5. Limited baseline set / 简单基准不足

The paper does not clearly emphasise comparisons with very simple forecasts such as:

- zero return;
- historical mean;
- random walk;
- last observed return.

对于金融收益率预测，复杂模型必须显著优于简单零收益或历史均值基准，才能体现真实预测价值。

---

### 6. Feature importance is not causal / 特征重要性不代表因果关系

LightGBM gain importance indicates predictive usefulness, not causal influence.

GPR 排名较高并不意味着 GPR 上升必然导致 Brent 上涨或下跌。

Gain importance may also be affected by:

- correlated predictors;
- variable scale;
- number of possible split points;
- model structure.

---

### 7. Limited interpretability / 可解释性有限

The paper reports gain, cover, and frequency importance but does not provide a detailed SHAP analysis of:

- direction of effects;
- nonlinear thresholds;
- variable interactions;
- individual forecasts;
- time-varying relevance.

本项目可通过 SHAP dependence、SHAP interaction 和 rolling SHAP 进一步改进。

---

### 8. No regime-specific forecasting analysis / 缺少分状态预测

The study discusses crises such as 2008 and 2020, but it does not systematically compare performance in:

- normal periods;
- high-VIX periods;
- high-GPR periods;
- supply-shock periods;
- geopolitical-crisis periods.

本项目可分别评估正常期和危机期，从而判断文本、航运和遥感信号是否主要在供应中断时期发挥作用。

---

### 9. Real-time data availability is not fully addressed / 实时可用性问题讨论不足

Although the explanatory variables are lagged, the paper does not fully analyse:

- publication delays;
- data revisions;
- real-time data vintages;
- when monthly GPR values become observable.

本项目在把月度指标转换为周度指标时，必须按照实际发布日期进行前向填充，不能将月底最终值回填到月初。

---

### 10. Entropy construction is insufficiently documented / Entropy 构造细节不足

The paper does not clearly report all implementation details of the rolling Shannon entropy variable, such as the exact moving-window length and discretisation choices.

这会降低 entropy 变量的可复制性。

---

### 11. Internal inconsistency in the GARCH discussion / GARCH 表述存在内部不一致

One part of the empirical discussion refers to a symmetric GARCH specification as the best fit, while the formal model-selection table reports gjrGARCH as having the best information criteria and uses gjrGARCH in subsequent forecasting comparisons.

论文正文中关于最佳 GARCH 类型的文字描述与表4存在轻微不一致，应以表4和后续 ARIMAX-gjrGARCH 分析为准。

---

## Notes for Dissertation Integration / 论文写作整合笔记

### Literature review use / 文献综述用途

This paper can be placed under:

- machine learning for crude oil forecasting;
- strategic-risk predictors;
- geopolitical and financial uncertainty;
- econometric versus tree-based models;
- Brent return forecasting.

可归入以下文献类别：

- 机器学习油价预测；
- 地缘政治和金融不确定性；
- Brent 收益率预测；
- ARIMAX/GARCH 与 boosting 模型比较；
- 战略风险变量的增量预测价值。

---

### Claims that this paper can support / 本文可以支撑的论点

1. GPR, VIX, and long-term interest-rate changes contain incremental information for Brent return forecasting.  
   GPR、VIX 和长期利率变化对 Brent 收益率具有额外预测信息。

2. Nonlinear tree-based boosting models may outperform linear econometric models.  
   非线性 boosting 模型可能优于线性计量模型。

3. LightGBM is a relevant candidate model for structured oil-market data.  
   LightGBM 值得纳入油价结构化数据预测的候选模型。

4. Rolling-origin evaluation is appropriate for oil-price forecasting.  
   滚动样本外验证适合油价预测。

5. External variables should be evaluated through ablation analysis rather than assumed to be useful.  
   外部变量的价值应通过消融实验验证。

6. Volatility modelling and return point forecasting are distinct tasks.  
   波动率建模与收益率点预测属于不同任务。

---

### Claims that this paper cannot directly support / 本文不能直接支撑的论点

1. Remote-sensing variables improve oil-price forecasts.  
   不能直接证明遥感变量有效。

2. AIS and port activity improve Brent forecasting.  
   不能直接证明 AIS 和港口活动有效。

3. LLM or NLP features improve prediction.  
   不能直接证明 LLM 或自建 NLP 特征有效。

4. Multimodal models necessarily outperform structured-data models.  
   不能证明多模态模型一定更好。

5. Five to ten variables are always sufficient or optimal.  
   不能证明5—10个变量是普遍最优数量。

6. LightGBM will necessarily outperform XGBoost in weekly multimodal data.  
   不能证明 LightGBM 在本项目周度多模态数据中一定优于 XGBoost。

---

## Potential Extension of the Paper / 本项目相对该论文的扩展

The dissertation can extend the paper from:

\[
\text{Historical Brent returns}
+
\text{Strategic risk indicators}
\]

to:

\[
\text{Market and macro-financial variables}
+
\text{NLP and event signals}
+
\text{Remote-sensing indicators}
+
\text{Shipping and port activity}
+
\text{Supply-chain network structure}
\]

The central extension is not merely adding more variables. It is testing whether each modality contributes genuinely new information beyond conventional price and strategic-risk variables.

本项目的创新不应只是“加入更多数据”，而应检验：

- 文本是否在 GPR 之外提供更高频的事件信息；
- 航运活动是否在库存和价格变量之外提供供应链信息；
- 遥感是否提供官方统计尚未反映的实物库存或设施活动信号；
- 多模态模型是否在正常期和危机期均稳定优于 M1 基准。

---

## One-Sentence Takeaway / 一句话总结

### English

This paper shows that lagged geopolitical risk, financial uncertainty, and interest-rate changes provide modest but meaningful incremental information for forecasting monthly Brent returns, while LightGBM captures their nonlinear interactions more effectively than ARIMAX, Random Forest, XGBoost, and ARIMAX-gjrGARCH in the reported experiments.

### 中文

本文表明，滞后的地缘政治风险、金融不确定性和长期利率变化能够在 Brent 历史收益率之外提供有限但有意义的增量预测信息，而在作者的月度样本和滚动预测设定中，LightGBM 对这些非线性关系和交互作用的捕捉优于 ARIMAX、Random Forest、XGBoost 和 ARIMAX-gjrGARCH。

---

## Suggested Dissertation Positioning Sentence / 建议用于论文中的定位句

### English

Building on Yılmaz and Zehir's strategic-risk forecasting framework, this dissertation treats lagged market and macro-financial indicators as a structured baseline and investigates whether textual, remote-sensing, shipping, and supply-chain information provides statistically significant incremental predictive value for weekly Brent crude oil returns.

### 中文

在 Yılmaz 和 Zehir 战略风险预测框架的基础上，本研究将滞后的市场和宏观金融指标作为结构化基准，并进一步检验文本、遥感、航运及供应链信息能否为 Brent 周度收益率提供具有统计显著性的增量预测价值。
