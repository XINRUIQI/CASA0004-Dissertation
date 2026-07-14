# Meeting 02 with Beatrice ｜ 第二次导师会议

**Date 日期:** 2026-06-05
**Supervisor 导师:** Beatrice Taylor
**Topic 主题:** Refocusing research objective, simplifying modelling framework, and literature-driven variable selection
　重新聚焦研究目标、简化建模框架与文献驱动的变量选择

## 1. Main discussion points 主要讨论内容

### 1.1 Research objective needs refocusing 项目目标需重新聚焦

- Beatrice advised **against** framing the goal as "building the most accurate oil price prediction model" — oil price prediction is inherently difficult, and many banks/institutions already do it; achieving high accuracy in a dissertation is unrealistic.
  Beatrice 建议**不要**把目标定位为"做出最准确的油价预测模型"——油价预测本身极其困难，很多银行和机构都在做，在 dissertation 中实现高准确率并不现实。
- The research question should shift from:
  研究问题应从：
  > ~~"How to accurately predict oil prices?"~~
  > ~~"如何准确预测油价？"~~

  to:
  调整为：
  > **"Can incorporating remote sensing and shipping data improve oil price prediction compared to financial data alone?"**
  > **"相比仅使用金融数据，加入遥感和航运数据是否能提升油价预测效果？"**

- This framing is more achievable within the dissertation scope and easier to articulate.
  这一框架在论文范围内更具可完成性，也更容易解释。

### 1.2 Model design: progressive comparison framework 模型设计：分阶段对比框架

- Beatrice proposed a clear progressive multi-model comparison structure:
  Beatrice 提出了一个清晰的渐进式多模型对比结构：

| Model 模型 | Data used 使用数据 | Purpose 目的 |
|---|---|---|
| Model 1 | Financial / market data | Baseline 基线 |
| Model 2 | Financial + remote sensing | 看遥感数据是否改善预测 |
| Model 3 | Financial + shipping | 看航运数据是否改善预测 |
| Model 4 | Financial + remote sensing + shipping | 看多模态组合是否进一步改善 |

- The focus is **not** on how high the absolute accuracy is, but on **comparing performance changes across models** — e.g. whether RMSE decreases or directional accuracy improves.
  重点**不是**绝对准确率有多高，而是**对比不同模型之间的表现变化**——例如 RMSE 是否下降、方向预测是否改善。

### 1.3 Drop text data to reduce complexity 删除文本数据以降低复杂度

- Beatrice explicitly advised **removing text data** from the current scope.
  Beatrice 明确建议当前阶段**去掉文本数据**。
- Reason: text data introduces a new modality with significant additional workload (text collection, cleaning, NLP feature extraction, sentiment analysis), which would dilute focus given the current timeline.
  原因：文本数据引入了一个全新的模态，涉及大量额外工作（文本收集、清洗、NLP 特征提取、情感分析等），在当前进度下会分散精力。
- The project should retain only three data types:
  项目应仅保留三类数据：
  1. **Financial / market data** 金融/市场数据
  2. **Remote sensing data** 遥感数据
  3. **Shipping data** 航运数据

### 1.4 Stop adding variables — select from literature 不要继续堆变量——从文献中筛选

- Beatrice noted the current problem is **too many variables, not too few** (~200 features already).
  Beatrice 指出当前的问题是**变量太多而非太少**（已有约 200 个特征）。
- Continuing to add variables risks making the model harder to interpret and devolving into "throw everything in and see what happens".
  继续添加变量会使模型更难解释，容易变成"把所有东西丢进模型看结果"的模式。
- Next step should be a **structured literature review** to identify which variables others have used. For each paper, record:
  下一步应进行**结构化文献综述**，梳理别人使用了哪些变量。每篇文献需记录：

| Field 字段 | What to record 需要记录的内容 |
|---|---|
| Paper | Author, year, title 作者、年份、题目 |
| Task | Price, returns, volatility, or direction prediction 预测油价/收益率/波动率/方向 |
| Model | XGBoost, Random Forest, LSTM, SVR, etc. |
| Variables | Input features used 使用了哪些输入变量 |
| Data frequency | Daily / weekly / monthly |
| Evaluation | RMSE, MAE, accuracy, directional accuracy, etc. |
| Main finding | Which variables or models were effective 哪些变量或模型有效 |

- Then select a **small, well-justified set of variables** based on the literature.
  然后根据文献选择一组**有依据的精简变量**。

### 1.5 Low accuracy is not a fatal problem 准确率低不是致命问题

- Many oil price prediction papers report accuracy around ~50%; this is normal because oil price prediction is an inherently hard problem.
  许多油价预测论文的准确率仅约 50%；这是正常的，因为油价预测本身就是一个非常困难的问题。
- Much of the existing literature targets **trading applications**, which demands high accuracy — but this dissertation does not need to serve that purpose.
  现有文献大多面向**交易应用**，对准确率要求很高——但本论文不必服务于该目的。
- The dissertation can be framed as:
  本论文可以定位为：
  > **Studying the marginal contribution of different data types to prediction performance, rather than building a trading model.**
  > **研究不同数据类型对预测性能的边际贡献，而非构建交易模型。**
- As long as models are systematically compared and results are well-explained (why certain data helped or didn't), the research is valid.
  只要系统地比较模型并合理解释结果（为什么某些数据有帮助或没有帮助），研究就是有效的。

### 1.6 Remote sensing: feasible but needs clearer features 遥感可行但需明确特征

- If using raw satellite images directly, a computer vision model would be needed — which adds complexity.
  如果直接使用原始卫星图像，需要引入计算机视觉模型——这会增加复杂度。
- The current remote sensing features are not yet well-defined; they need to be grounded in literature.
  目前遥感特征尚未明确定义，需要通过文献进一步确认。
- A safer approach is to use **derived indices and proxy variables** rather than building a complex CV pipeline. Examples include:
  更稳妥的做法是使用**衍生指数和代理变量**，而非构建复杂的 CV 流程。例如：
  - NDVI / NDBI / NDWI / BSI
  - Night-time lights 夜间灯光
  - Storage / refinery activity proxy 储油/炼油活动代理指标
  - Port / industrial activity proxy 港口/工业活动代理指标
- The key is to first establish **why** these variables are theoretically linked to oil price prediction.
  关键是首先建立这些变量与油价预测之间的**理论联系**。

### 1.7 Consider PCA for dimensionality reduction 可考虑 PCA 降维

- Asked about how many features are appropriate given ~20 years × 52 weeks of data. Beatrice did not give a fixed number, but stressed that the variable count should remain **reasonable and interpretable**.
  问到 20 年 × 52 周的数据量下多少特征合适。Beatrice 没有给出固定数字，但强调变量数量需**合理且可解释**。
- She suggested **PCA (Principal Component Analysis)** as a dimensionality reduction tool:
  她建议使用 **PCA（主成分分析）** 作为降维工具：
  - If shipping or remote sensing variables are highly correlated, PCA can compress them into a few components.
    如果航运或遥感变量高度相关，PCA 可以将它们压缩为少数几个主成分。
  - E.g. multiple shipping indicators → 1–2 shipping components; multiple remote sensing indices → a few RS components.
    例如：多个航运指标 → 1–2 个航运主成分；多个遥感指标 → 几个遥感主成分。
- This reduces model complexity and improves interpretability.
  这样可以减少模型复杂度，也能提高解释性。

### 1.8 Literature search strategy 文献搜索方法

- Use **Google Scholar** with targeted keyword combinations, e.g.:
  使用 **Google Scholar**，结合关键词组合搜索，例如：
  - `machine learning oil price prediction remote sensing`
- Restrict the time range — ML-related papers should prioritise **recent years**.
  限制时间范围——机器学习相关论文最好看**近年的**。
- Prioritise **highly cited** papers.
  优先看**引用次数较高**的论文。
- Verify relevance carefully: some papers that appear relevant actually predict *petrol pollution*, not oil price.
  仔细验证相关性：有些看似相关的论文实际上预测的是*石油污染*，而不是油价。
- AI tools can help find literature, but you **must manually verify** on Scholar that the paper actually exists, is relevant, and is cited.
  AI 工具可以帮助找文献，但**必须自己去 Scholar 验证**论文是否真实存在、是否相关、是否被引用。

### 1.9 Next steps: 3-week literature review sprint 下一步任务：3 周文献综述冲刺

- Core task for the next 3 weeks: **focus on literature review**.
  接下来 3 周的核心任务：**集中做文献综述**。
- Compile a literature notes file (Markdown / Word / Excel), with the following structure per paper:
  整理一份文献笔记文件（Markdown / Word / Excel），每篇论文按以下结构记录：

```
## Paper N: Author (Year)

### Research task
预测 Brent / WTI price / return / volatility / direction

### Data used
Financial variables, macro variables, remote sensing, shipping, news, etc.

### Variables
列出具体变量

### Model
XGBoost / RF / LSTM / SVR / Transformer, etc.

### Evaluation metrics
RMSE / MAE / accuracy / directional accuracy

### Main findings
总结哪些变量有效、模型表现如何

### Usefulness for my dissertation
这篇文献如何帮助我选择变量或模型
```

## Core takeaway 本次会议核心结论

> **不要继续扩大数据和变量，而应收缩范围，通过文献确定关键变量，把 dissertation 重点放在"不同数据模态是否改善油价预测"上。**
>
> 下一步路线：**读文献 → 选变量 → 简化模型 → 对比不同数据组合的预测提升。**

## 2. Supervisor feedback 导师反馈

- **Positive 正面:** Dataset inventory and variable tables are clear and detailed; good progress overall.
  数据集清单和变量表清晰详细；总体进度良好。
- **Refocus 重新聚焦:** Shift the research question from "accurate prediction" to "does multimodal data improve prediction".
  将研究问题从"准确预测"转向"多模态数据是否改善预测"。
- **Simplify 简化:** Drop text data; retain only financial, remote sensing, and shipping data.
  去掉文本数据；仅保留金融、遥感和航运数据。
- **Reduce variables 减少变量:** ~200 features is too many; use literature to justify a smaller, curated set.
  约 200 个特征太多；用文献来论证一组更精简的变量。
- **Progressive modelling 渐进式建模:** Frame the research around how each data modality contributes to prediction, via the 4-model comparison framework.
  通过 4 模型对比框架，围绕各数据模态对预测的贡献展开研究。
- **Accuracy expectations 准确率预期:** ~50% is normal for oil price prediction; the project's value lies in systematic comparison, not absolute performance.
  油价预测中 ~50% 的准确率是正常的；项目价值在于系统性比较，而非绝对表现。
- **PCA:** Consider PCA to compress correlated variables within each modality group.
  考虑使用 PCA 压缩各模态组内高度相关的变量。
- **Literature search 文献搜索:** Use Google Scholar with targeted keywords; verify AI-suggested papers manually; prioritise recent, highly-cited work.
  使用 Google Scholar 配合关键词搜索；手动验证 AI 推荐的论文；优先看近期高引用论文。
- **Next 3 weeks 接下来 3 周:** Focus entirely on structured literature review.
  全力集中于结构化文献综述。

## 3. Decisions made 确定的决策

- Reframe the research question to focus on **whether multimodal data improves oil price prediction**.
  将研究问题重新聚焦于**多模态数据是否改善油价预测**。
- Adopt the 4-model progressive comparison framework (financial → +remote sensing → +shipping → combined).
  采用 4 模型渐进式对比框架（金融 → +遥感 → +航运 → 合并）。
- **Remove text data** from the project scope.
  从项目范围中**移除文本数据**。
- Stop adding new variables; use literature review to **curate a smaller, justified variable set**.
  停止添加新变量；通过文献综述**精选一组有依据的变量**。
- Remote sensing features should use **derived indices/proxies** rather than raw images + CV models (unless literature strongly supports it).
  遥感特征应使用**衍生指数/代理变量**，而非原始图像 + CV 模型（除非文献有力支持）。
- The dissertation does not aim to build a trading model; it studies the **marginal contribution** of different data modalities.
  论文目标不是构建交易模型，而是研究不同数据模态的**边际贡献**。
- **PCA** is on the table as a dimensionality reduction tool for correlated features within each modality.
  **PCA** 作为各模态内相关特征的可选降维工具。

## 4. Action points before next meeting 下次会议前的行动项

**Priority: 3-week literature review sprint 优先：3 周文献综述冲刺**

- [ ] Conduct a structured literature review using the template above (Paper, Task, Model, Variables, Data frequency, Evaluation, Main finding).
  按上述模板进行结构化文献综述（论文、任务、模型、变量、数据频率、评估指标、主要发现）。
- [ ] Based on literature, select a curated subset of financial variables for Model 1 (baseline).
  基于文献，为模型 1（基线）精选金融变量子集。
- [ ] Identify and justify remote sensing features (indices/proxies) for Model 2 from literature.
  从文献中确定并论证模型 2 使用的遥感特征（指数/代理变量）。
- [ ] Identify and justify shipping variables for Model 3 from literature.
  从文献中确定并论证模型 3 使用的航运变量。
- [ ] Decide on modelling approach (e.g. XGBoost, LSTM, etc.) based on literature findings.
  根据文献发现决定建模方法（如 XGBoost、LSTM 等）。
- [ ] Refine the research question to reflect the new framing (multimodal contribution comparison).
  调整研究问题以反映新定位（多模态贡献比较）。
- [ ] Remove text data from the variable table and project plan.
  从变量表和项目计划中移除文本数据。
- [ ] Explore PCA for dimensionality reduction within each modality group (shipping, remote sensing).
  探索在各模态组（航运、遥感）内使用 PCA 降维。
- [ ] Set up a literature notes file (Markdown / Word / Excel) using the template from Section 1.9.
  按第 1.9 节模板建立文献笔记文件（Markdown / Word / Excel）。

---

> **Source 来源:** Meeting recording transcript + student notes.
> 会议录音转录 + 学生笔记。
