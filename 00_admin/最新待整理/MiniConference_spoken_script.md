# Mini-Conference Spoken Script（~7 min）
# 14 July 2026 — 讲稿（英文为主；略偏技术，但仍可给外行人听懂）

> 口径：锁定四页弧；数字只用 briefing 允许的；随机游走最多一句带过。  
> 「多一点点技术」= 点名 Ridge / XGBoost / TCN / GAT / Prithvi / gated fusion / walk-forward / SHAP，但每处先给一句白话。

---

## Slide 1 — Problem & motivation　`[0:00–1:30]`

**EN:**

Hi everyone. My project is about forecasting **next-week Brent crude oil prices**.

Oil markets already give us a lot of financial information — futures, spreads, macro series. But physical supply and demand often show up first in other places: **satellite imagery** around oil ports, and **shipping networks** through chokepoints like Hormuz or the Red Sea.

So my question is not only “should I add satellite and shipping?” It’s three steps:

1. Which of these data types actually help?
2. Is dumping them into **one big numerical table** enough?
3. Or is it better to **learn each data type with its own specialised model, then combine** them — what the literature calls **representation-level fusion**, as opposed to early feature-level concatenation?

That’s the story.

**中文提示：** 预测下周 Brent → 金融之外还有卫星/航运 → 三步：谁有用；一张大表够不够；分开学再合（表示级融合）会不会更好。

---

## Slide 2 — RQs first, then design　`[1:30–3:15]`

**EN:**

Three research questions.

**RQ1:** Beyond finance, do satellite or shipping help predict next-week oil price?  
**RQ2:** Same data — is one big table enough, or does learn-each-then-combine work better?  
**RQ3:** Can the model show which data type it relies on over time?

Data: three modalities with different structure — weekly **finance time series**; **satellite images** (I use a frozen EO foundation model, **Prithvi**, plus site indices); and a **shipping graph** of ports and chokepoints.

Fair rules: no future leakage; **walk-forward / rolling** evaluation; same scoring for everyone.

Two ways of using the data:

- **Big-table ladder, M0–M4**, branching from finance-only M1:  
  M0 = price stays the same; M1 = finance; **M2 = finance + satellite**; **M3 = finance + shipping**; **M4 = all three** in one table.  
  On this layer I run the literature’s usual benchmarks: **Ridge** for regularised linear, **XGBoost** for nonlinear trees — not a horse-race of every model.

- **Learn-each-then-combine:** a **TCN** for finance, a **GAT + TCN** for the shipping graph, Prithvi embeddings for satellite, then **gated fusion** — the model learns, week by week, how much to trust each modality.

I’ll start with the big-table results — they motivate the smarter combination.

**中文提示：** 先念三个 RQ → 三模态结构不同 → walk-forward 公平规则 → M0–M4 从 M1 分叉（不是链式）→ Ridge+XGB；深度侧 TCN / GAT+TCN / Prithvi + 门控融合。

---

## Slide 3 — Big table: modality roles　`[3:15–5:15]`

**EN:**

So what happens in the big table?

**Shipping helps — but only if the model can use it.** On top of finance, the nested shipping increment is statistically meaningful under **XGBoost**; under **Ridge**, that same high-dimensional shipping block often disappears into noise. So the signal is there; a simple linear model can’t reliably pick it up.

**Satellite is weak** in the main full-column setup — adding it doesn’t clearly help.

**SHAP** on the full multimodal table (M4, XGBoost) tells the same story about relative importance:  
shipping about **52%**, finance about **34%**, satellite about **15%**. Shipping’s share is roughly three times satellite’s.

One short caveat: the naive same-price benchmark remains hard to beat on raw error — that’s well known in oil forecasting. But the more interesting point for this talk is modality roles: **in the big table, shipping looks useful; satellite looks weak** — and even useful shipping is hard to turn into clean gains when everything is flattened together.

**中文提示：** 航运有用但模型敏感（XGB 有、Ridge 常无）→ 卫星弱 → SHAP 52/34/15 → 一句带过随机游走 → 收束到「扁平用不好」。

---

## Slide 4 — Same data, smarter use　`[5:15–7:00]`

**EN:**

Now the same data, but each modality gets its own encoder, then **gated fusion**.

First, **shipping versus satellite** under the smarter pipeline:  
finance + shipping is the strongest combo — skill versus the naive benchmark about **+0.11%**, the only main gated setup that edges slightly positive.  
Finance + satellite stays clearly worse, about **−2.4%**. So even with specialised learners, shipping helps and satellite stays weak.

Second, **same combination, big table versus specialised learner** — skill versus naive, XGBoost for the table side:

| Combo | Big table | Specialised + gated |
|-------|-----------|---------------------|
| Finance + shipping | ≈ **−6.7%** | ≈ **+0.11%** |
| Finance + satellite | ≈ **−7.0%** | ≈ **−2.4%** |
| All three | ≈ **−8.6%** | ≈ **−1.3%** |

So for shipping, the big table largely wastes the signal; the graph encoder plus gated fusion recovers it. Smart weighting also beats simply concatenating the three learned vectors with no gate.

Closing ask: how should I frame the main claim — shipping value versus overall forecast difficulty — and should I keep the weak satellite branch for completeness?

Thank you — happy to take feedback.

**中文提示：** 门控下航运 vs 卫星 → 三组 skill 对照表念清楚 → 智能加权 > 直接粘贴 → 收尾提问。

---

## Timing check / 自检

| Slide | Target | Content lock |
|-------|--------|--------------|
| 1 | ~1.5 min | 动机 + 三步通俗切入；可出现 representation-level / feature-level 对照一句 |
| 2 | ~1.5–2 min | RQ 最前；M2∥M3；点名 Ridge/XGB + TCN/GAT/Prithvi/gated |
| 3 | ~2 min | 模态作用 + SHAP 52/34/15；少提随机游走 |
| 4 | ~1.5 min | 三组 skill 数字；不问 p 值表 |

## Technical depth note / 「多一点点技术」放在哪

讲稿里**口头点名**即可，不必展开公式：

- 大表：Ridge / XGBoost / walk-forward / SHAP  
- 深度：Prithvi（冻结）/ TCN / GAT / gated fusion / 32-d representation（若被问再补）  
- **不要**上屏或口述：p 值表、fusion RMSE 网格、Cross-Attention 主数字
