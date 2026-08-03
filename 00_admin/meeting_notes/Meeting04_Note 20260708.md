# Meeting 04 with Beatrice ｜ 第四次导师会议

**Date 日期:** 2026-07-08  
**Supervisor 导师:** Beatrice Taylor  
**Topic 主题:** Reviewing flat vs deep multimodal results against Model 0, clarifying how to present findings, and shifting to dissertation structure and writing  
　对照模型 0 审视扁平与深度多模态结果、明确结果呈现方式，并转向 dissertation 结构与写作

## 1. Main discussion points 主要讨论内容

### 1.1 Progress since the previous meeting 上次会议后的进展

- Following Meeting 03, the student standardised the comparison and added **Model 0** (no-change / persistence baseline: next-week price equals current price).
  按第三次会议要求，已完成标准化比较，并加入 **模型 0**（不变预测基准：下一周油价等于当前油价）。
- Under the original **flat** architecture (tabular / CSV features into ML models such as XGBoost), **none of M1–M4 beat Model 0**.
  在原有的**扁平**架构下（表格 / CSV 特征输入 XGBoost 等机器学习模型），**M1–M4 均未能超过模型 0**。
- Flat remote-sensing features were CSV summaries generated via **Google Earth Engine (GEE)** from imagery. Literature suggested that compressing images / graphs into flat tables may lose spatial, temporal, or relational information.
  扁平框架下的遥感特征是经 **Google Earth Engine (GEE)** 从影像生成的 CSV 汇总。文献提示：将影像 / 图结构压成扁平表，可能损失空间、时间或关系信息。
- The student therefore changed the **input pathway**, not merely “a few features”:
  因此学生改变的是**输入路径**，而非只改几个特征：
  1. First tried modality-specific encoders to produce embeddings (e.g. about **32-dimensional** features) and then feed them into ML / tree models.  
     先尝试用各模态编码器得到 embedding（例如约 **32 维**特征），再输入机器学习 / 树模型。  
  2. Tree models did **not** perform well on those embeddings.  
     树模型在这些 embedding 上**表现不好**。  
  3. The prediction framework was therefore shifted to **deep learning**, keeping modality-specific encoders.  
     因此将预测框架改为**深度学习**，并保留各模态专用编码器。
- Beatrice initially summarised this as “encoding then into XGBoost”; the student corrected that the final framework is deep learning, not tree models on embeddings.
  Beatrice 起初概括为“先编码再进 XGBoost”；学生澄清最终框架是深度学习，而非 embedding + 树模型。
- Beatrice confirmed that results can now be discussed, but the priority is no longer further model redesign; it is **writing up** and presenting findings clearly.
  Beatrice 确认可以开始讨论结果，但下一阶段重点不再是继续改模型，而是**写作**并清晰呈现发现。

### 1.2 Why the architecture was changed 为什么改变架构

- The new design uses modality-specific encoders before prediction:
  新设计在预测前使用各模态专用编码器：

| Modality 模态 | Encoder / model 编码器 / 模型 | Rationale 理由 |
|---|---|---|
| Financial / tabular 金融表格 | **TCN** (Temporal Convolutional Network) | Capture temporal structure; let the network discover useful signals rather than only hand-selected flat features / 捕捉时间结构；让网络自行发现有用信号，而非仅靠手工扁平特征 |
| Remote sensing 遥感 | **Presto** (remote-sensing / computer-vision-style encoder) | Use satellite images directly rather than GEE CSV summaries that compress information / 直接使用卫星影像，而非压缩信息的 GEE CSV 汇总 |
| Shipping 航运 | **GNN** (Graph Neural Network) | Shipping data are graph-structured; GNN preserves relational structure / 航运数据呈图结构；GNN 保留关系结构 |

- Beatrice asked whether these methods come from the literature. The student confirmed that each method is literature-based, but combining them in one oil-price forecasting project is the student's own contribution.
  Beatrice 询问这些方法是否来自文献。学生确认各方法均有文献依据，但将它们组合进同一油价预测项目是本人的贡献。
- Beatrice pressed on whether the student can explain each deep network: how many layers / what the structure is, what the inputs and outputs are, what the network is doing, and why that architecture is necessary for that modality.
  Beatrice 追问学生能否解释每个深度网络：有多少层 / 结构是什么、输入输出是什么、网络在做什么、为何该模态需要该架构。
- The student was **not fully clear** on some structural details in the meeting. Beatrice stressed that the dissertation must make these explanations clear; otherwise the modelling section will not be convincing.
  会上学生对部分结构细节**说不清**。Beatrice 强调论文必须写清楚，否则建模部分难以令人信服。

### 1.3 Flat-model results: Model 0 remains best 扁平模型结果：模型 0 仍然最优

- Looking at RMSE on the printouts: **every complex flat model is worse than Model 0**.
  对照打印结果中的 RMSE：**所有复杂扁平模型都差于模型 0**。
- Important presentation point:
  重要呈现要点：

  > **Null or near-null results are acceptable.** Model 0 remains essential because it shows that most complex models do not beat a simple persistence forecast.  
  > **无效或近似无效的结果是可以接受的。** 模型 0 必须保留，因为它表明多数复杂模型无法超过简单的持续性预测。

- Beatrice asked whether Model 0 should even be included; answer: **yes, very important**, because it shows the results are largely null / near-null.
  Beatrice 问是否还要把模型 0 写进项目；答案：**必须写，非常重要**，因为它表明结果大体为无效 / 近无效。
- Internal comparison among flat models is still useful:
  扁平模型之间的内部比较仍然有价值：
  - Relative to **M1 (financial only)**, **M3 (financial + shipping)** improves. The email notes emphasise this M3-vs-M1 finding.
    相对 **M1（仅金融）**，**M3（金融 + 航运）** 有所改善。邮件笔记强调这一 M3-vs-M1 发现。
  - In the meeting discussion of the table, Beatrice also noted that **M3 XGBoost and M4 XGBoost both look better than M1** internally.
    当面看表时，Beatrice 也指出内部比较中 **M3 XGBoost 与 M4 XGBoost 都看起来优于 M1**。
  - Cautious claim: shipping adds some information beyond financial data.
    谨慎结论：航运在金融数据之外提供了一定信息。
  - However, this internal improvement still does **not** overturn Model 0 as the best overall flat baseline.
    但这一内部改善仍**不足以**推翻模型 0 作为扁平框架下的整体最优基准。

### 1.4 Deep-model results: small gains for M3 and M4 深度模型结果：M3 与 M4 有小幅提升

- Intermediate comparison discussed in the meeting:
  会上还讨论了中间层比较：
  - Deep **financial-only** already looks better than flat financial-only.
    深度**仅金融**已看起来优于扁平仅金融。
  - But that deep financial-only model is **still worse than Model 0**.
    但该深度仅金融模型**仍差于模型 0**。
- After the full deep / encoded architecture comparison:
  在完整深度 / 编码架构比较后：
  - **Model 0 still performs very strongly**.
    **模型 0 仍然表现很强。**
  - **M3_Deep_gated** (financial + shipping) and **M4_Deep_gated** (financial + shipping + remote sensing) show a **small improvement** over Model 0.
    **深度 M3**（金融 + 航运）和 **深度 M4**（金融 + 航运 + 遥感）相对模型 0 有**小幅改善**。
  - Beatrice asked whether this is a big improvement; she judged it as **minor**, not a large breakthrough.
    Beatrice 追问这是否算大幅改善；她判断为**轻微**，并非大幅突破。
- M4_Deep_gated does not clearly outperform M3_Deep_gated. Therefore shipping appears more useful than remote sensing in the current setup; adding remote sensing does not further improve the best shipping-inclusive model.
  深度 M4 并未明显优于深度 M3。因此，在当前设定下，航运似乎比遥感更有用；加入遥感并未进一步提升包含航运的最佳模型。
- Student interpretation discussed: shipping is the most useful added modality; using graph structure helps the shipping signal come through.
  学生解释：航运是最有用的新增模态；使用图结构有助于航运信号发挥作用。
- Across modalities / model pairs, the **deep architecture generally outperforms the flat architecture**.
  跨模态 / 模型对比较时，**深度架构整体优于扁平架构**。

### 1.5 How to present the three key modelling results 如何呈现三项核心建模结果

- Student asked whether including Model 0–M4 flat plus Model 0–M4 deep is too much.
  学生问：扁平 M0–M4 加上深度 M0–M4 是否太多。
- Beatrice: since the work has already been done, **include all of it**, provided the tables are clear.
  Beatrice：既然已经做了，**都写进去**，前提是表格足够清楚。
- Presentation narrative Beatrice recommended:
  Beatrice 建议的呈现叙事：
  - You tried many model and data combinations; nothing really improves on Model 0 in the flat setting.
    尝试了很多模型与数据组合；扁平设定下几乎没有真正超过模型 0 的。
  - In the deep setting, one / two models show only a **minor** improvement.
    深度设定下，一两个模型仅有**轻微**改善。
  - Explain all models, but emphasise that only these deep models show any improvement over baseline.
    解释所有模型，但强调只有这些深度模型相对基准有改善。
- Organise the results section around **three clear findings**:
  结果部分围绕**三项清晰发现**组织：

| Finding 发现 | Content 内容 |
|---|---|
| 1. Flat models 扁平模型 | Model 0 is best. M1–M4 all fail to beat Model 0. Internally, shipping-inclusive models improve on financial-only (email focus: M3 vs M1; meeting table talk also noted M3 and M4 vs M1). / 模型 0 最优；M1–M4 均未超过模型 0；内部比较中含航运模型优于仅金融（邮件侧重 M3 vs M1；当面看表也提到 M3 与 M4 vs M1）。 |
| 2. Deep models 深度模型 | Model 0 remains strong. M3_Deep_gated and M4_Deep_gated show only a small improvement over Model 0. / 模型 0 仍然很强；深度 M3 与深度 M4 仅相对模型 0 有小幅改善。 |
| 3. Flat vs deep 扁平 vs 深度 | Deep architecture is better across the model set. / 深度架构在整组模型中整体更优。 |

- All tables must make column meanings explicit (RMSE, CW / other metrics, model names, flat vs deep).
  所有表格必须清楚标明各列含义（RMSE、CW / 其他指标、模型名称、扁平 vs 深度）。
- Naming should be consistent, e.g. M1_Flat / M3_Flat / M3_Deep_gated / M4_Deep_gated.
  命名应保持一致，例如 M1_Flat / M3_Flat / M3_Deep_gated / M4_Deep_gated。
- Comparisons should be reported **both** against Model 0 and among M1–M4.
  比较应**同时**报告相对模型 0 的结果，以及 M1–M4 之间的内部比较。

### 1.6 Explain evaluation metrics clearly, especially CW 清晰解释评估指标，尤其是 CW

- Beatrice repeatedly asked what **CW** measures after looking at the printouts.
  Beatrice 看打印结果时多次询问 **CW** 衡量的是什么。
- Concrete contradiction discussed: e.g. **M3 XGBoost** can look **worse on RMSE** than another model, but **better on CW**. Without a clear definition, this is confusing.
  当面讨论的具体矛盾：例如 **M3 XGBoost** 可能在 **RMSE 上更差**，却在 **CW 上更好**。没有清晰定义就会令人困惑。
- Beatrice also questioned why emphasise CW comparisons when models are already significantly worse than Model 0 on RMSE.
  Beatrice 也质疑：既然 RMSE 上已明显差于模型 0，为何还要强调 CW 比较。
- Student said CW is used in oil-price forecasting literature; smaller may be better; one attempted explanation was that it relates to stability / trustworthiness. The explanation in the meeting was still unclear.
  学生称 CW 用于油价预测文献；可能越小越好；曾尝试解释为与稳定性 / 可信度有关。会上解释仍不清楚。
- Before using CW to claim improvement, the dissertation must define:
  在用 CW 声称改善之前，论文必须定义：
  - The exact formula  
    准确公式
  - Whether smaller or larger values are better  
    数值越小越好还是越大越好
  - Why the metric is used in oil-price forecasting literature  
    为何油价预测文献使用该指标
  - How it relates to RMSE / MAE / directional accuracy  
    它与 RMSE / MAE / 方向准确率的关系
- Without this explanation, CW looks like an opaque or ad-hoc metric rather than a justified evaluation choice.
  若缺少这一解释，CW 会显得不透明或像临时构造的指标，而非有依据的评估选择。

### 1.7 Do not keep redesigning models; start writing 不要继续改模型，开始写作

- The student wondered whether further feature-structure / input changes were still needed.
  学生犹豫是否仍需继续改特征结构 / 输入方式。
- Beatrice’s view was clear:
  Beatrice 的观点很明确：

  > **At this stage, stop major model redesign and start writing the dissertation.**  
  > **现阶段应停止大规模模型改造，开始撰写 dissertation。**

- Enough modelling work has already been done to support a results narrative. Further tinkering risks delaying writing.
  已有足够建模工作支撑结果叙述；继续微调会延误写作。
- The student acknowledged spending **too much time on models / methods** and needing to shift to how the dissertation will be presented.
  学生承认在**模型 / 方法上花了太多时间**，需要转向论文如何呈现。

### 1.8 Literature review feedback 文献综述反馈

- The student has drafted a literature review using previous dissertations as structural references.
  学生已参考往届 dissertation 结构撰写文献综述初稿。
- AI assisted mainly on **structure**; the student said they read the papers themselves and selected the literature.
  AI 主要协助**结构**；学生表示文献是自己阅读并筛选的。
- Approximate length discussed: around **2,500–4,000 words** for the literature review (student also mentioned ~3,000); the full dissertation upper limit is about **10,000 words**. Excess wording can be cut later.
  讨论的大致篇幅：文献综述约 **2,500–4,000 词**（学生也提过约 3,000）；整篇 dissertation 上限约 **10,000 词**。多余文字之后可以删减。
- Beatrice liked the idea of summary tables for approaches and limitations, but noted that the current table is **missing citations**.
  Beatrice 认可用表格概括方法与局限的思路，但指出当前表格**缺少文献引用**。
- Action: send the literature-review draft to Beatrice for formal written feedback.
  行动：将文献综述初稿发给 Beatrice，以便获得正式书面反馈。

### 1.9 Variable importance / SHAP only for models that beat Model 0 仅对优于模型 0 的模型做变量重要性 / SHAP

- The student proposed analysing which features or modalities matter in different periods (e.g. shipping more important in one year, remote sensing in another), using methods such as **SHAP**.
  学生提议用 **SHAP** 等方法分析不同时期哪些特征或模态更重要（例如某年航运更重要、某年遥感更重要）。
- Beatrice agreed this is useful, but only for models that actually improve on Model 0:
  Beatrice 同意这有价值，但仅适用于确实优于模型 0 的模型：
  - Prioritise **M3_Deep_gated** (preferred, because it improves without relying on the weaker remote-sensing add-on)
    优先分析 **深度 M3**（更优先，因其改善不依赖较弱的遥感增量）
  - Optionally also analyse **M4_Deep_gated**
    也可一并分析 **深度 M4**
  - Do **not** run variable-importance analysis on models worse than the baseline
    **不要**对差于基准的模型做变量重要性分析
- Focus on the deep models rather than the flat models for this interpretability step.
  这一可解释性步骤应聚焦深度模型，而非扁平模型。

### 1.10 Dissertation structure is the next deliverable 下一步交付物：dissertation 结构草稿

- Beatrice asked for a **draft dissertation structure within one week**, including:
  Beatrice 要求在**一周内**提交 **dissertation 结构草稿**，内容包括：
  - Section and subsection headings  
    章与节标题
  - A few bullet points under each part describing the main content  
    每一部分用若干要点说明主要内容
- Suggested high-level sections: Introduction, Literature Review, Methods, Results, Discussion.
  建议的高层结构：Introduction、Literature Review、Methods、Results、Discussion。
- Methods will need substantial space for **data explanation**; Results should explicitly cover flat models, deep models, and flat-vs-deep comparison.
  Methods 需要较多篇幅解释**数据**；Results 应明确覆盖扁平模型、深度模型，以及扁平与深度对比。
- Formal submission discussed as around **20 August**; next meeting: **Wednesday 29 July 2026**, expected to be the **last in-person meeting before submission**. Beatrice would send a **Teams invite**.
  正式提交讨论为约 **8 月 20 日**；下次会议：**2026 年 7 月 29 日（周三）**，预计为**提交前最后一次当面会议**。Beatrice 会发 **Teams 邀请**。

## Core takeaway 本次会议核心结论

> **现阶段最重要的不是继续改模型，而是把已有结果写成清晰叙事，并规划 dissertation 结构。**
>
> **核心结果可概括为三点：（1）扁平模型中模型 0 最优，但航运相对纯金融有内部改善；（2）深度模型中深度 M3 / M4 相对模型 0 仅有小幅改善；（3）深度架构整体优于扁平架构。中间层还有：深度仅金融优于扁平仅金融，但仍差于 M0。SHAP 只做优于模型 0 的深度模型。CW 必须定义清楚。一周内提交论文结构草稿，并发送文献综述初稿。**

## 2. Supervisor feedback 导师反馈

- **Model 0 is essential 模型 0 必不可少:** It shows that most complex models fail to beat a simple no-change forecast; null results should still be presented.
  它表明多数复杂模型无法超过简单不变预测；无效结果仍应呈现。
- **Stop major redesign 停止大规模改造:** Enough modelling has been done; shift to writing and presentation. The student has already spent too much time on models.
  建模已足够；应转向写作与结果呈现。学生在模型上已花过多时间。
- **Explain every deep model 解释每一个深度模型:** Inputs, outputs, architecture / layers, and why that encoder fits the modality must be clear — the meeting showed this still needs work.
  必须说清输入、输出、架构 / 层数，以及为何该编码器适合该模态——会上已显示这一点仍需加强。
- **Clarify CW and all table columns 澄清 CW 与所有表格列:** Especially where CW and RMSE disagree; otherwise CW looks ad-hoc.
  尤其在 CW 与 RMSE 不一致时；否则 CW 会像临时指标。
- **Present three result blocks 呈现三块结果:** Flat vs M0; Deep vs M0; Flat vs Deep. Include all completed models if tables are clear.
  扁平相对 M0；深度相对 M0；扁平相对深度。若表格清楚，已完成的模型都可纳入。
- **Compare both ways 双向比较:** Against Model 0, and internally among M1–M4.
  既相对模型 0，也在 M1–M4 内部比较。
- **Shipping is the more useful added modality 航运是更有用的新增模态:** It improves on financial-only models; remote sensing is weaker in current results.
  它改善了仅金融模型；遥感在当前结果中较弱。
- **SHAP only for winning models 仅对获胜模型做 SHAP:** Prefer M3_Deep_gated; optionally also M4_Deep_gated; not flat / losing models.
  优先深度 M3；可选深度 M4；不做扁平 / 落败模型。
- **Cite literature in summary tables 汇总表中加入文献引用:** Approach / limitation tables need references.
  方法 / 局限表需要引用。
- **Send structure + literature review soon 尽快发送结构与文献综述:** Structure draft within one week; literature review for formal feedback.
  一周内交结构草稿；文献综述供正式反馈。

## 3. Decisions made 确定的决策

- Keep **Model 0** as the central baseline in the dissertation narrative.
  在论文叙事中继续以 **模型 0** 作为核心基准。
- Include both **flat** and **deep** architectures in the report, since both have already been implemented; do not drop completed work just because there are many tables.
  报告中同时纳入**扁平**与**深度**两套架构，因为两者均已实现；不要因为表格多就丢掉已完成工作。
- Present results as three blocks: flat models, deep models, and flat-vs-deep comparison.
  结果按三块呈现：扁平模型、深度模型、扁平与深度对比。
- Claim shipping value cautiously: useful relative to financial-only models, but only a minor improvement over Model 0 in the deep setting.
  谨慎表述航运价值：相对仅金融模型有用，但在深度设定下相对模型 0 仅有小幅改善。
- Treat remote sensing as less helpful than shipping in the current evidence.
  在当前证据下，将遥感视为不如航运有帮助。
- Add clear definitions for **CW** and all evaluation metrics, especially where metrics conflict.
  为 **CW** 及所有评估指标补充清晰定义，尤其在指标冲突时。
- Run **SHAP / variable-importance** analysis preferentially for **M3_Deep_gated**, optionally also **M4_Deep_gated**.
  **SHAP / 变量重要性**优先做 **深度 M3**，可选同时做 **深度 M4**。
- Do not continue major model redesign; prioritise dissertation structure and writing.
  不再继续大规模模型改造；优先 dissertation 结构与写作。
- Send Beatrice a draft dissertation outline within one week, plus the literature-review draft.
  一周内向 Beatrice 发送 dissertation 结构草稿，并附文献综述初稿。
- Next meeting fixed for **29 July 2026** (last in-person before submission around **20 August**); Teams invite to follow.
  下次会议定为 **2026 年 7 月 29 日**（约 **8 月 20 日**提交前最后一次当面会）；随后发 Teams 邀请。

## 4. Action points before next meeting 下次会议前的行动项

**Priority: dissertation structure, literature-review revision, and clear results write-up  
优先：dissertation 结构、文献综述修改，以及清晰的结果撰写**

- [ ] Draft the full dissertation structure (sections + subsections + bullet points for each part) and send it to Beatrice within one week.
  起草完整 dissertation 结构（章、节及各部分要点），并在一周内发给 Beatrice。
- [ ] Send the current literature-review draft for formal feedback.
  发送当前文献综述初稿以获取正式反馈。
- [ ] Add citations to literature-review summary tables (methods, limitations, gaps).
  在文献综述汇总表中补充引用（方法、局限、研究缺口）。
- [ ] Write clear definitions for RMSE, CW, and any other evaluation metrics used in result tables; resolve the RMSE-vs-CW contradiction in wording.
  为结果表中的 RMSE、CW 及其他评估指标撰写清晰定义；在文字中化解 RMSE 与 CW 的矛盾。
- [ ] Standardise model naming across flat and deep results (e.g. M1_Flat–M4, M1_Deep–M4).
  统一扁平与深度结果中的模型命名（如 M1_Flat–M4、M1_Deep–M4）。
- [ ] Redraft result tables so every column is self-explanatory to a non-specialist reader.
  重做结果表，使每一列对非专业读者也自解释。
- [ ] Write the results narrative around the three agreed findings: flat vs M0; deep vs M0; flat vs deep.
  围绕三项约定发现撰写结果叙述：扁平相对 M0；深度相对 M0；扁平相对深度。
- [ ] Also report the intermediate finding if useful: deep financial-only beats flat financial-only, but still loses to Model 0.
  如有用，也可报告中间发现：深度仅金融优于扁平仅金融，但仍输给模型 0。
- [ ] Explain TCN, Presto, and GNN in methods: inputs, outputs, architecture / layers, and literature justification; note the pathway from GEE CSV → embedding attempts → deep models.
  在方法部分解释 TCN、Presto 与 GNN：输入、输出、架构 / 层数及文献依据；并说明从 GEE CSV → embedding 尝试 → 深度模型的路径。
- [ ] Run SHAP / variable-importance analysis for M3_Deep_gated, and optionally M4_Deep_gated.
  对深度 M3 运行 SHAP / 变量重要性分析，可选同时分析深度 M4。
- [ ] Do not add SHAP for models that fail to beat Model 0.
  不对未能超过模型 0 的模型增加 SHAP。
- [ ] Avoid further major architecture changes unless required to correct an error.
  除非为了修正错误，否则避免进一步大规模架构改动。
- [ ] Shift time from further modelling toward dissertation presentation / structure.
  把时间从继续建模转向论文呈现 / 结构。
- [ ] Prepare materials for the final pre-submission meeting on **29 July 2026** (submission around **20 August**).
  为 **2026 年 7 月 29 日**提交前最后一次会议准备材料（正式提交约 **8 月 20 日**）。

---

> **Source 来源:** Meeting recording transcript (`Meeting04_20260708.txt`) + supervisor email notes (`Meeting04_Mail 20260708.txt`, Beatrice Taylor, 2026-07-08/09).  
> 会议录音转录（`Meeting04_20260708.txt`）+ 导师会后邮件笔记（`Meeting04_Mail 20260708.txt`，Beatrice Taylor，2026-07-08/09）。
>
> **Update 更新:** Patched against the recording checklist to add previously weak/missing points (GEE CSV origin; embedding→ML then deep pathway; Beatrice’s structure probing; deep financial-only intermediate result; CW vs RMSE contradiction; M3/M4 internal comparison nuance; include-all-completed-models; student spent too long on models; due ~20 Aug; Teams invite).  
> 已按录音核对清单补齐原先偏弱/遗漏要点（GEE CSV 来源；embedding→ML 再改 deep 的路径；Beatrice 追问结构；深度仅金融中间结果；CW 与 RMSE 矛盾；M3/M4 内部比较细节；已做模型都可写入；学生建模耗时过多；交稿约 8/20；Teams 邀请）。
