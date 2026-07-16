# Prompt for Claude — Mini-Conference PPT + Script

# Claude 用 Prompt — 学位论文 Mini-Conference 幻灯片 + 讲稿

> Copy everything below the line into Claude. Attach the files listed in `MiniConference_files_to_send.md`.  
> 把分隔线以下全部粘贴进 Claude，并按 `MiniConference_files_to_send.md` 上传附件。

---

You are helping me prepare for a **dissertation mini-conference** presentation.  
你在帮我准备**学位论文 mini-conference**演讲。

## Event constraints / 活动硬约束（strict）


| EN                                                           | 中文               |
| ------------------------------------------------------------ | ---------------- |
|                                                              |                  |
| 7 min present + 5 min feedback                               | 7 分钟讲 + 5 分钟反馈   |
| **3–4 slides max**, strict on time                           | **最多 3–4 页**，卡时间 |
| Slides in English                                            | 幻灯片英文            |
| Spoken script: English primary + optional short Chinese cues | 讲稿以英文为主，可附简短中文提示 |
| Audience does **not** know my project                        | 听众**不了解**我的项目    |




## Goal / 目标

Tell a clear story for newcomers:  
给完全不了解的人讲清一条故事线：

1. What problem & why（问题与动机）
2. What I ask + how I test it（问什么 + 怎么验）
3. What each data type does in the simple table approach（简单拼表时，各类数据有什么用）
4. Same data, smarter use — does it help?（同样数据换更聪明的用法，有没有更好）

**Not** a progress checklist. **Not** a random-walk-centred talk.  
**不是**进度清单。**不要**以「打不赢随机游走」当主叙事。

## Produce / 请产出

1. 4-slide outline (title + bullets + speaker notes); `.pptx` if possible
2. Timed ~7 min script (6:30–7:00), slide-by-slide
3. 1-page Q&A cheat sheet
4. Simple academic design; large fonts; one idea per slide



## One-sentence pitch（plain / 通俗一句）

I predict **next-week oil price (Brent)** using three kinds of data: **markets/finance**, **satellites**, and **ships**. I ask: which data helps; is dumping everything into one big table enough; and does learning each data type separately then combining them work better?  

我用三类数据预测**下周布伦特油价**：金融市场、卫星、航运。我依次问：哪些数据有用；全部塞进一张大表够不够；每种数据分开学再合起来会不会更好。

## Title / 标题

*A Modality-Aware Spatio-Temporal Fusion Framework for Brent Crude Oil Forecasting Using Financial Time Series, Satellite Imagery and Maritime Networks*

## Plain-language dictionary（MUST use on slides / 幻灯片必须用通俗说法）


| Avoid on slides 幻灯片避免          | Prefer 改用                                                                        |
| ------------------------------ | -------------------------------------------------------------------------------- |
| flat / wide table jargon alone | **one big table of numbers** / 一张数字大表                                            |
| representation-level fusion    | **learn each data type separately, then combine** / 每种数据分开学，再合起来                 |
| gated fusion                   | **smart weighted combination**（可括号 gated）/ 智能加权组合                                |
| modality                       | **data type** / 数据类型                                                             |
| encoder                        | **specialised learner for that data** / 该类数据的专用学习器                               |
| nested increment / CW / DM     | say “statistically meaningful signal” if needed; **no p-value tables on slides** |


Slide 1 must **not** say “flat” or “fusion” without a plain gloss.  
Slide 1 不得只甩 flat/fusion 而不解释。

## Research questions — on Slide 2 only / RQ 只放 Slide 2

Use plain wording on slides:

- **RQ1:** Beyond finance, do satellite / shipping help predict next-week oil price?  
在金融之外，卫星 / 航运对预测下周油价有没有帮助？
- **RQ2:** Same data — is “one big table” enough, or is “learn-each-then-combine” better?  
同样数据下，「一张大表」够不够，还是「分开学再合」更好？
- **RQ3:** Can the model show which data type it relies on over time?  
模型能否说明不同时期更依赖哪类数据？



## Models you may name（do not invent others）/ 可点名的模型（勿编造）



### Flat / 大表层（Slide 3）

- Always run **both**:
  - **Ridge** = regularised linear benchmark（稳、怕高维噪声）
  - **XGBoost (XGB)** = tree / boosting benchmark（能抓非线性）
- **Why these two (Q&A / speaker note):**  
  **Chosen from the literature’s usual oil-forecasting benchmark ladder** — not after a horse-race of every model.  
  Literature hierarchy: random walk / no-change → **regularised linear** (Ridge/LASSO/Elastic Net family; e.g. Costa [P072]) → **tree ensembles** (XGBoost as the common strong nonlinear benchmark; e.g. P072, P004, Jung multi-source tabular).  
  中文口径：「扁平层按文献常见基准固定两类：正则化线性用 Ridge，树模型用 XGBoost。」  
  If pressed: strong benchmarks, not proven globally best; LightGBM is a parallel candidate in some papers (P076) — I use XGBoost as the tree-family representative.
- **Why report both results:** shipping is **model-sensitive** — XGB finds a nested signal; Ridge often cannot. That contrast is part of the flat-layer story.
- **Tuning (not model choice):** fixed walk-forward protocol (`L4_tuned`); hyperparameters on the **last 52 weeks of each training window** (never on the test set).

消融：

- M0 = naive “price stays the same”
- M1 = finance only
- M2 = finance + satellite
- M3 = finance + shipping *(no satellite)*
- M4 = all three in one big table



### Deep / 分开学再合层（Slide 4）— main names only


| Name                                        | Plain meaning                                                          |
| ------------------------------------------- | ---------------------------------------------------------------------- |
| **Finance-only learner** (Mfin)             | specialised learner on finance series                                  |
| **Satellite-only** (Mrs)                    | specialised learner on satellite                                       |
| **Shipping-only** (Mship)                   | graph learner on shipping network                                      |
| **Finance + shipping** (Mfinship / Mfusion) | combine finance + shipping with smart weights — **main positive case** |
| **Finance + satellite** (Mfinrs)            | combine finance + satellite                                            |
| **All three** (Mfull / M4rep)               | all three with smart weights                                           |
| **Simple concat control** (Mconcat)         | glue the three learned vectors with no smart weighting                 |


Do **not** dump Cross-Attention numbers on the main slides (optional backup only).

## Locked 4-slide arc / 锁定四页结构



### Slide 1 — Problem & motivation（~1.5 min）问题与动机

Three bullets only:

1. **Predict:** next-week **weekly Brent** oil price（hard problem）
2. **Why try:** besides markets, **satellites** (oil-port activity) and **ships** (chokepoints / flows) may show supply–demand earlier
3. **My angle (plain, no jargon):**
  First: which of these data help?  
   Next: is putting them all into **one big table** enough?  
   Then: does **learning each data type separately and combining them** work better?

Hook: “Not only ‘add satellite/shipping or not’ — but which help, whether a big table is enough, and whether a smarter combination helps.”

### Slide 2 — RQs first, then how I test（~1.5–2 min）先 RQ，再设计

**Order on slide (mandatory):**

1. **Three RQs** (plain wording above)
2. **Three data types:** finance series / satellite images & indices / shipping network
3. **Fair rules:** no peeking at the future; rolling / walk-forward tests; same score for everyone
4. **Two ways of using the data:**
  - **Big-table ladder M0–M4** (branching from M1 — **NOT** a chain):  
   M0 same-price; M1 finance; **M2 = M1+satellite**; **M3 = M1+shipping**; **M4 = all three**  
  - **Learn-each-then-combine:** specialised learner per data type + smart weighted combination

Transition at **end** of slide:  
“I’ll start with the big-table results — they motivate the smarter combination.”

**Critical:** M2 and M3 are **parallel** add-ons on M1. Never say “finance → add satellite → then add shipping.”

### Slide 3 — Big table: what each data type does（~2 min）大表：各类数据作用

**Centre the story on modalities — NOT on random walk.**  
Mention the strong naive benchmark at most **one short clause**.

Must cover:

1. **Which data helps (big table):**
  - Shipping adds a **statistically meaningful** nested signal on top of finance (XGB); Ridge often cannot use it → signal exists, simple linear use fails  
  - Satellite add-on is **weak / not clearly helpful** in the main full-column setup
2. **How much the model leans on each type (SHAP on M4, XGB):**
  shipping **~52%** > finance **~34%** > satellite **~15%**  
   → shipping matters far more than satellite in the big-table model (~3× the SHAP share; ~37 percentage points higher)
3. Close: “In the big table, shipping looks useful; satellite looks weak — but even useful shipping is hard to turn into clean gains when everything is flattened together.”

Optional one clause: naive same-price benchmark remains hard to beat on raw error — then move on.

### Slide 4 — Same data, smarter use（~1.5 min）同样数据，换用法

**Again: modality contrast + big-table vs specialised-learner contrast. Soft-pedal random walk.**

Must cover (use these magnitudes; do not invent new %):

1. **Shipping vs satellite (smarter combination):**
  - Finance+shipping is the **strongest** combo (only main gated setup with skill slightly above the naive benchmark: about **+0.11%**)  
  - Finance+satellite stays clearly worse (about **−2.4%** skill)  
   → shipping helps; satellite still weak under the same smarter pipeline
2. **Same combo: big table vs specialised learner** (skill vs naive benchmark; XGB for big table):

  | Combo               | Big table (XGB) | Specialised learner | Takeaway                           |
  | ------------------- | --------------- | ------------------- | ---------------------------------- |
  | Finance + shipping  | about **−6.7%** | about **+0.11%**    | smarter use recovers shipping      |
  | Finance + satellite | about **−7.0%** | about **−2.4%**     | improves a bit, still weak         |
  | All three           | about **−8.6%** | about **−1.3%**     | better than big table, still mixed |

3. **Smart weighting beats naive glue-together** of the three learned vectors
4. Soft ask: how to frame the claim (shipping value vs overall difficulty); keep weak satellite?



## Results policy / 结果口径

- Slide 3–4: modality roles + big-table vs specialised-learner contrasts  
- Allowed numbers: SHAP 52/34/15; skill ≈ −6.7%→+0.11% (shipping); −7.0%→−2.4% (satellite); −8.6%→−1.3% (all three)  
- **No** p-value tables; **no** fusion RMSE grids on slides  
- Random walk / naive benchmark: at most a light clause, not the headline



## DO / DO NOT

**DO**

- Plain language for classmates  
- Correct M0–M4 branching  
- RQ on Slide 2 **first**  
- Modality-centred Slide 3–4

**DO NOT**

- Progress-update tone  
- “Clearly beats random walk / SOTA”  
- Wrong sequential M0–M4 ladder  
- Lead with “nobody beats random walk” as the main message of Slide 3–4  
- Invent percentages



## Inputs attached / 附件

1. `MiniConference_briefing.md` — trust for constraints + messaging
2. `Meeting04_prep_20260707.md` — fuller narrative / architecture
3. `Meeting04_spoken_script_20260707.md` — background only; **rewrite** to this arc
4. Optional: flat SHAP figure for Slide 3



## Output format / 输出格式



### A. Slide deck

Each slide: Title / on-slide bullets (≤5, plain English) / visual / speaker notes  

### B. Timed script

~7 min with `[0:00–1:30]` stamps; follow Slide 1→4 exactly  

### C. Q&A cheat sheet

Include: novelty; **why Ridge+XGB → “from literature benchmark ladder, not horse-race”**; why not only XGB on all columns; why keep satellite; big-table vs specialised learner; did you beat the naive benchmark (honest, brief); why M2∥M3  

### D. Morning checklist

9am slide send; arrive early; one rehearsal  

Start now.