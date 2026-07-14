# Mini-Conference Briefing（14 July 2026）
# Mini-Conference 简报（给 Claude / 自己核对）

Use this as the source of truth. Meeting04 files are background only.  
以本文件为准；Meeting04 仅作背景。

## Event / 活动

| Item | Detail |
|------|--------|
| When | 14 July 2026 |
| Where | LG11, Bentham House, London WC1H 0EG |
| Talk | **7 min** + **5 min** feedback |
| Slides | **3–4 max**; send by **9:00am** |
| Audience | 不了解项目的同学与老师 → **通俗语言** |

## Goal / 目标

故事线：动机 → RQ+设计 → 大表里各类数据作用 → 同样数据换用法有没有更好。  

**不是**进度汇报；**不要**以随机游走为主叙事。

## Pitch（plain / 通俗）

EN: Predict next-week Brent with finance + satellite + shipping. Ask: which data help; is one big table enough; does learn-each-then-combine work better?  
中文：用金融、卫星、航运预测下周油价。先问哪些有用；一张大表够不够；分开学再合会不会更好。

## Title

*A Modality-Aware Spatio-Temporal Fusion Framework for Brent Crude Oil Forecasting Using Financial Time Series, Satellite Imagery and Maritime Networks*

## Plain language / 通俗替换

| 避免 | 改用 |
|------|------|
| flat | one big table / 一张数字大表 |
| representation-level fusion | learn each data type separately, then combine / 分开学再合 |
| gated fusion | smart weighted combination / 智能加权组合 |
| modality | data type / 数据类型 |

## Slide 1 vs RQ

| Slide 1 | RQ（Slide 2 最前） |
|---------|-------------------|
| 动机：为什么做 | 可检验问题：实验要回答什么 |

## RQs（plain）

1. Beyond finance, do satellite / shipping help?  
2. Same data: one big table vs learn-each-then-combine — which is better?  
3. Can the model show which data type it relies on over time?

## Models / 模型清单（汇报时用）

### Big-table layer（Slide 3）大表层

| Model | Role |
|-------|------|
| **Ridge** | 正则化线性基准；稳，但高维噪声下常吃不到航运信号 |
| **XGBoost** | 树/提升基准；能抓非线性；航运嵌套增量主要在它上面显著 |

**为何选这两个（文献，非赛马）：**  
按油价预测文献常见基准阶梯预先固定两类——随机游走 → **正则化线性**（Ridge 代表；P072 等）→ **树模型**（XGBoost 代表；P072 / P004 / 多源表格如 Jung）。  
口头一句：「扁平层按文献常见基准固定两类：正则化线性用 Ridge，树模型用 XGBoost。」  
追问再补：是强基准，不是已被证明全局最优。

**为何两个都报：** 同一份航运数据，XGB 有信号、Ridge 常没有 → 「信号在、简单线性用不好」。  
**怎么选参（≠怎么选模型）：** 固定滚动协议 `L4_tuned`；每个训练窗**尾部 52 周**验证选超参，不看测试集。

| Spec | Content |
|------|---------|
| M0 | 下周价 = 本周价 |
| M1 | 仅金融 |
| M2 | M1 + 卫星 |
| M3 | M1 + 航运（无卫星） |
| M4 | 三者一张大表 |

### Learn-each-then-combine（Slide 4）分开学再合

| Name | Plain |
|------|-------|
| Mfin | 仅金融专用学习器 |
| Mrs | 仅卫星 |
| Mship | 仅航运（图） |
| **Mfinship / Mfusion** | 金融+航运智能加权 — **主正向** |
| Mfinrs | 金融+卫星 |
| Mfull / M4rep | 三数据智能加权 |
| Mconcat | 三向量直接粘贴（对照） |

Cross-Attention 仅 backup，不进主 4 页。

## Design map / 设计地图

**Fair rules:** 无未来信息；滚动回测；同一套评分。  

**M0–M4:** 从 M1 **分叉**（M2∥M3），不是「先卫星再航运」一条链。

## Locked 4 slides / 锁定四页

1. **Problem & motivation** — 预测什么；为何卫星/航运；三步通俗切入（有没有用 → 大表够不够 → 分开学再合）  
2. **RQs first → data → fair rules → two ways** — 过渡句放页末  
3. **Big table: modality roles** — 航运有用、卫星弱；SHAP 52% / 34% / 15%；少提随机游走  
4. **Same data, smarter use** — 航运 vs 卫星；大表 vs 专用学习器（下表）；智能加权 > 直接粘贴  

### Numbers allowed on slides / 可上屏数字

**SHAP (M4, XGB):** shipping ~52% > finance ~34% > satellite ~15%  
（航运 SHAP 占比约为卫星的约 3 倍；高出约 37 个百分点。勿写不准确的「好了 40%」 unless you explicitly mean this share gap and say so.)

**Skill vs naive same-price benchmark（越大越好；负=更差）**

| Combo | Big table (XGB) | Specialised learner | 怎么说 |
|-------|-----------------|---------------------|--------|
| Finance + shipping | ≈ **−6.7%** | ≈ **+0.11%** | 大表浪费航运；分开学再合把航运捡回来 |
| Finance + satellite | ≈ **−7.0%** | ≈ **−2.4%** | 有改善，但仍弱 |
| All three | ≈ **−8.6%** | ≈ **−1.3%** | 好过大表，整体仍杂 |

## Results policy

- 主叙事 = **各类数据作用** + **大表 vs 分开学**  
- 随机游走最多一句带过  
- 不上 p 值表、不上 fusion RMSE 网格  

## Closing ask（optional）

How to frame the main claim (shipping value vs overall difficulty)? Keep weak satellite?

## Do not invent

No invented citations, metrics, or architectures beyond Meeting04 / this brief.
