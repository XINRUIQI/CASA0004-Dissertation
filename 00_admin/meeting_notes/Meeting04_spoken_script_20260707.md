# Meeting 04 — Spoken Script (Bilingual) / 口播讲稿（双语）

> **Say in English.** Chinese below is for your own reference only.
> **用英语讲。** 中文仅供你自己对照，不必念出来。
> No detailed numbers · simple words · ~10–15 min + discussion
> 详细版 / Full version: `Meeting04_prep_20260707.md`

---

## Part 1 — Why fuse, not flat-concatenate / 为什么要融合（3–4 min）

### The big picture / 大背景

**EN:**

First, a bit of context from the literature. Oil price forecasting has a very strong simple benchmark: just assume next week's price is about the same as this week's. A lot of complex models still struggle to beat that.

So if we add remote sensing or shipping data, it's not enough to just "add more columns." We need to actually **use** that information well. And that's where flat concatenation falls short — the data goes in, but the model doesn't use it efficiently.

**中文（参考）：**

文献里反复说：油价预测最简单的基准很强——假设下周价格和这周差不多，很多复杂模型打不过。加遥感、航运不能只是多加几列，要真的用起来；扁平拼接的问题就是信息进去了但用不好。

---

### Three modalities, three different structures / 三个模态结构不同

**EN:**

The three modalities in my project are really not the same kind of thing.

Finance is a regular time series — one row per week.

Satellite data is **imagery**. The signal is in the pixels. A few NDVI columns don't fully capture that.

Shipping is more like a **network** — ports, chokepoints, who connects to whom. Something like a Red Sea rerouting event is really a network story, not just "how many ships."

If you squash all three into columns and paste them into one table, you lose the graph structure and you lose the image structure. The multimodal literature calls this **early, feature-level fusion** — everything gets concatenated up front. The alternative is to let each modality learn its own representation first, then fuse.

I'm not the first person to think this way. People already use graph neural networks for tanker traffic. Remote sensing has foundation models like Prithvi and SatMAE to extract features from images. And in financial time series, modality-aware designs have been shown to work better than naive concatenation.

**中文（参考）：**

三个模态不是同一种东西：金融是时间序列；卫星是影像，信号在像素里；航运是网络（港口、咽喉、谁连谁）。压成列再拼表，图结构和影像结构都没了。文献把这叫早期特征级融合；更好的做法是每个模态先学自己的表示再融合。航运有人用图网络、遥感有 Prithvi/SatMAE、金融时序也有人做过模态感知结构。

---

### Flat models can't re-weight by regime / 不能按时期动态调权

**EN:**

Another problem with flat models: different market periods rely on different information. When there's a Red Sea disruption, shipping probably matters more.

But a flat model gives each feature a more or less **fixed** weight. It can't say, "this week, trust shipping more."

That's why I use **gated fusion** — the model learns, at each time point, how much to trust each modality. That helps prediction, and it also feeds into interpretability: which modality matters when.

**中文（参考）：**

不同时期依赖的信息不一样（比如红海问题时航运更重要），但扁平模型给的是固定权重，说不出"这周更信航运"。所以我用门控融合，让模型自己学每个时点该重视哪个模态，也回答可解释性问题。

---

### Missing data, lags, different frequencies / 缺失、滞后、频率不同

**EN:**

There's also a practical issue. Remote sensing is monthly and often cloudy. Shipping data comes with publication lags. Finance is weekly. The three streams are **out of sync and incomplete**, and a flat table doesn't handle that well.

The literature has tools for this — missing-modality training, irregular time series models, and so on. Flat concatenation doesn't really use them.

**中文（参考）：**

遥感月度且有云、航运有发布滞后、金融周度——三种数据不同步、不完整，扁平表很难处理。文献里有缺失模态、不规则时序等方法，扁平拼接基本用不上。

---

### The gap in the literature / 文献空白

**EN:**

When I looked at existing work that connects alternative data to oil prices, **most of it still stops at flat fusion** — turn everything into numeric columns and feed them in.

Some papers predict trade or demand, not price. Some only use one or two modalities. Some have rich methods but no imagery and no shipping network.

As far as I can tell, **no one** has kept all three heterogeneous modalities — finance, remote sensing and shipping — and fused them at the representation level for **weekly Brent**, under a fair comparison with flat fusion. That's the gap I'm trying to fill.

**中文（参考）：**

现有工作绝大多数还是做成几列数字拼进模型；有的预测贸易/需求不是价格；有的只有一两个模态。据我掌握，还没有人在周频 Brent 上同时保留三模态做表示级融合，并与扁平方法公平对照——这是我要填的空白。

---

### What my own flat baselines showed / 我的扁平实验说明了什么

**EN:**

This is also why I decided to build the fusion layer.

I ran the flat baselines properly first. A few things came out.

First — and I'll be honest about this — **no flat model consistently beats the random walk**. That's a real finding, and it matches the oil forecasting literature.

But second, and more important: **the same shipping data works very differently in different models**. Tree models can pick up some signal. Linear models basically can't. And a hand-picked "core" feature set is not even the best for tree models.

Third, when you bolt remote sensing and shipping onto an already strong financial baseline in a flat way, the **increment gets diluted**.

So my reading is: the problem may not be "is there any signal at all?" but **"how do we fuse it?"** Flat concatenation wastes the signal.

**中文（参考）：**

这也是我决定做融合层的原因。扁平基线跑完后：第一，没有扁平模型稳定打败随机游走（诚实结论）；第二，同一份航运数据换模型效果差很多；第三，拼到强金融基线上增量被稀释。问题可能不在数据有没有价值，而在怎么融合——扁平拼接把信号浪费了。

---

### One-line summary / 小结

**EN:**

I'm not trying to invent a brand-new fusion operator or loss function. The building blocks already exist in the literature — frozen remote-sensing foundation models, graph neural networks, gated fusion, missing-modality handling.

**My contribution is more about integration**: putting these pieces together, and running the first fair comparison between representation-level fusion and flat fusion on weekly Brent.

**中文（参考）：**

我不是要发明新算子或新损失。文献里零件都有。我的贡献更像是集成——把这些方法拼起来，第一次在周频 Brent 上公平比较表示级融合和扁平融合。

---

## Part 2 — What the framework looks like / 框架长什么样（2–3 min）

**EN:**

The overall idea is quite simple — think of three pipelines that meet in the middle.

**Pipeline one: finance.** The financial series goes through a small temporal encoder and comes out as a compact vector.

**Pipeline two: remote sensing.** Satellite images go through a pretrained model called Prithvi to get embeddings. I **don't fine-tune** it — the weekly sample is too small and I'd worry about overfitting. Then I add temporal and site attention to pull together information across oil sites and months.

**Pipeline three: shipping.** I don't treat this as a normal table. I treat it as a **graph**: eleven oil sites plus six chokepoints — seventeen nodes in total. A graph attention layer learns spatial relationships; a temporal model learns how things change over time; then everything is pooled into one vector.

Those three vectors go into **gated fusion**. The model decides how much to trust finance, remote sensing or shipping at each point in time. After fusion, it predicts next week's Brent price.

I've also set up a few comparison arms:

- finance only, shipping only, remote sensing only — to test single-modality value;
- finance plus shipping with gating — this is the most promising result so far;
- all three modalities;
- simple concatenation and cross-attention — to compare fusion methods.

And this maps cleanly onto my three research questions:

- **RQ1**: do remote sensing and shipping add value on top of finance?
- **RQ2**: does representation-level fusion beat flat fusion? — that's the core one.
- **RQ3**: can gating weights tell us which modality the model relies on, and when?

**中文（参考）：**

整体三条流水线汇到中间：金融→时序编码器；遥感→Prithvi 提 embedding（不微调）+ 时间/站点注意力；航运→十七节点图（GAT 学空间 + 时序学时间）→ 门控融合 → 预测下周油价。对照臂：单模态、金融+航运门控、全模态、拼接与交叉注意力。对应 RQ1 增量价值、RQ2 融合方式（核心）、RQ3 可解释性。

---

## Part 3 — What I did this phase / 这段时间做了什么（2–3 min）

**EN:**

I'll break this into four blocks.

**Block one: data.** I cleaned up the pipeline for all three modalities, with a strong focus on **no information leakage** — for example, EIA data that only comes out on Wednesday shouldn't be usable before that; remote sensing and shipping are aligned to real publication times too. I ended up with one unified weekly feature matrix. Per your advice last meeting, I **dropped the cloud-fraction column** for remote sensing and kept only the valid-observation count. I also built a dynamic graph for shipping, for the graph neural network later.

**Block two: flat baselines — the comparison group.** Under one shared protocol, I finished M0 through M4: random walk, Ridge, XGBoost, and a deep early-fusion version. Each modality has statistical tests, SHAP explanations, and robustness checks — things like water masking for remote sensing and leave-one-channel-out for shipping. This layer answers RQ1 and gives the fusion layer a **fair yardstick**.

**Block three: the fusion layer — the main new work.** I actually **implemented and ran** the framework I just described. Three encoders, gated fusion, prediction head — all wired up. Training was stable; I didn't get the kind of collapse we worried about before, with lots of negative R² values. That addresses your overfitting concern from last meeting.

**Block four: literature and writing.** I've closed the literature gap for the fusion layer, drafted chapter 2, and started chapter 3. The items you asked for last meeting — M0 baseline, unified protocol, clear prediction target, variable descriptions, dropping cloud fraction, starting the lit review — are all done.

**中文（参考）：**

四块：① 数据管线理顺、防泄漏、删云量留有效观测、建航运动态图；② 扁平 M0–M4 跑完（随机游走/Ridge/XGB/深度早融合）+ 检验/SHAP/稳健性，作对照标尺；③ 融合层写进代码并跑通，训练稳定、无 R² 崩溃；④ 文献补齐、Ch2 草稿、Ch3 动笔，上次会议要求都做了。

---

## Part 4 — What the results look like / 结果怎么样（2–3 min）

**EN:**

I'll stay away from specific numbers and just talk about direction.

**Flat layer — honest, a bit frustrating.**

Overall, the random walk is still very strong. Most complex models don't beat it — and I'll report that honestly; it's consistent with the literature.

But that doesn't mean remote sensing and shipping are useless. In some setups, statistical tests show they **do add information** on top of a pure financial baseline — it's just that absolute accuracy still doesn't beat the random walk. Those two things are not contradictory, and I need to explain that clearly in the thesis.

Shipping is the clearest example: **change the model, and the result changes a lot**. So the signal is there — flat fusion just doesn't use it well.

SHAP also shows that in the full multimodal model, shipping features tend to matter most, finance second, remote sensing third but still non-zero.

So my takeaway from the flat layer is: **the data is not dead — the fusion method is the problem.**

**Fusion layer — the promising direction.**

The fusion layer is still early, but there's already one important signal:

**The same shipping data that flat early-fusion couldn't use becomes useful once you encode it with a graph neural network** — the nested increment over the financial baseline becomes statistically meaningful.

Going one step further, finance plus shipping with gated fusion is the **first setup where the error edges below the random walk**. It's not rock-solid statistically yet, but the direction is right.

The full three-modality gated model also shows a meaningful increment over the flat financial baseline.

The remote-sensing branch is still weak. I think the Prithvi embeddings and hyperparameters need more tuning — that's on the to-do list.

**The one sentence I'd like you to remember:**

It's not "I built a model that crushes everything" — not yet.

It's: **how you fuse really matters.**

Same data, same evaluation protocol — flat concatenation wastes the signal; representation-level fusion starts to recover it. That's exactly what RQ2 is about, and I now have early evidence for it.

**中文（参考）：**

不讲具体数字，只说方向。扁平层：随机游走依然很强，我会如实写；但遥感/航运相对金融基线有时确有增量，只是绝对精度仍打不过随机游走——两者不矛盾。航运换模型效果差很多，说明信号在、扁平用不好；SHAP 显示航运贡献最大。结论：数据不是没信号，是融合方式有问题。融合层：同一份航运数据，扁平早融合用不了，图神经网络编码后增量变显著；金融+航运门控融合是第一个误差略好于随机游走的配置（统计还不算特别稳）；全模态门控也有显著增量；遥感分支还偏弱待调。最想记住的一句：不是碾压一切的模型，而是融合方式真的 matters——同样数据同样协议，扁平浪费信号，表示级融合能把信号捡回来，RQ2 已有初步证据。

---

## If asked: next steps / 如果对方问下一步（~30 sec）

**EN:**

Short term, three things: finish the fusion layer — cross-attention, missing-modality handling, remote-sensing tuning, gating visualisations; put flat and fusion results in one fair comparison table to answer RQ2 properly; and push on chapters 3 and 4.

**中文（参考）：**

短期三件事：融合层收尾（交叉注意力、缺失模态、遥感调参、门控可视化）；扁平和融合结果放同一张公平对照表正式答 RQ2；推进 Ch3/Ch4 写作。

---

