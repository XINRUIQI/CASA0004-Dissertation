## Part 1 — Why fuse, not flat-concatenate / 为什么要融合（3–4 min）

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



## 1. 为什么要融合多模态，而不是扁平拼接（文献支撑）/ Why fuse, not flat-concatenate (with literature)

**中文：** 核心论点：油价预测的随机游走极强，任何另类数据都必须被**高效**利用；而把三个结构迥异的模态压成一张宽表，会在信息进入模型前就把结构和信号浪费掉。**本节的目的，是让每一个设计决策都能指到具体文献。**

**EN:**  Core argument: the random-walk benchmark in oil forecasting is very strong, so any alternative data must be used *efficiently*; compressing three structurally different modalities into one wide table wastes their structure and signal before the model ever sees it.

### 1.1 前提：油价预测的三条铁律 / The three hard facts of oil forecasting

**中文：**

- **随机游走极强**：Alquist–Kilian–Vigfusson 预测手册 [P053] 确立"不变（随机游走）基准"是复杂模型难以战胜的强基线；Foroutan & Lahmiri [P001] 提醒单步价格误差因 P_{t+1}\approx P_t 而"看起来很好"。→ 任何加进来的信号都必须**证明**自己，且要被高效利用（扁平拼接的低效正是问题）。
- **要用机制特征**：Kilian [P052] 把油价冲击分解为供给 / 总需求 / 预防性需求；变量应覆盖这些机制通道，而非"凑够十个"。
- **组合不同机制优于单一模型**：Baumeister & Kilian [P054] 表明跨不同经济机制的预测**组合**通常优于任一单模型——这正是"多模态 + 按模态消融"的经济学依据（RQ1）。

**EN:** The random walk is a benchmark complex models routinely fail to beat [P053], and single-step price-level errors look deceptively good because P_{t+1}\approx P_t [P001]; predictors should span supply/demand/precautionary mechanisms [P052]; and combinations across *different* mechanisms tend to beat any single model [P054] — the rationale for a multimodal, modality-by-modality design (RQ1).

### 1.2 每个模态都有结构，扁平拼接把结构丢了 / Each modality has structure that flat concatenation discards

**中文：**

- **定义层面**：Baltrušaitis 等多模态综述 [P101] 把"把 NDVI、船舶计数、宏观变量拼成一张表"明确归类为**早期特征级融合**；其对立面是用**模态专属编码器**学联合表示（representation-level）。本研究做的正是后者。
- **航运本质是图**：现有原油航运工作已把港口/咽喉建成图节点——Ouyang 等 [P062]（供应链图卷积 + LSTM 预测油轮流量）、Liang 等 [P066]（时空多图网络）、Graph WaveNet [P091]；IMF PortWatch [P070] 提供咽喉/港口过境序列。扁平的"每节点计数列"把这张网络拓扑丢了。
- **卫星是高维影像**：EO 基础模型 Prithvi-EO-2.0 [P094] / SatMAE [P095] 的**冻结编码器**能产出可迁移影像表示；且因传感器结构/噪声不同，多模态 EO 模型用**模态专属编码器**（如 CROMA [P105]）先分别编码再融合——几列 NDVI 做不到。
- **原始辐射/NDVI 是错误的表示**：Polinov 等 [P024] 指出单港 NTL–油轮相关几乎为零（NTL 不是数船器）；Gibson 等 [P032] 指出 NTL 擅长横截面差异、拙于**站内时间变化** → 所以用**站内标准化异常**而非原始值（Hao & Wang [P025] 的可观测性通道同理）。
- **先例**：Gohari 等 Modality-aware Transformer [P039] 在金融时序上证明"模态感知结构 > 朴素拼接"，且给出可解释的跨模态注意力。

**EN:** The multimodal survey [P101] classifies "concatenating NDVI, vessel counts and macro variables into one table" as *early feature-level fusion*; the alternative is a joint representation via modality-specific encoders. Shipping is intrinsically a **graph** — crude-oil work already models ports/chokepoints as graph nodes [P062][P066][P091][P070], structure a flat count-per-node table discards. Satellite data is high-dimensional imagery best handled by **frozen EO foundation models** [P094][P095] with modality-specific encoders [P105]. Raw radiance/NDVI is the wrong representation (NTL is not a tanker counter [P024]; NTL captures cross-sectional but not within-site temporal variation [P032]), hence within-site anomalies [P025]. And a modality-aware design already beats naïve concatenation on financial series [P039].

### 1.3 扁平模型无法按时期动态调权 / Flat models cannot weight modalities per regime

**中文：**

- 有用的预测变量**随时间与预测期变化**（Costa 等 [P072]）；扁平模型给的是**静态**特征权重，说不出"这一周多看航运"。
- **门控多模态单元** [P096] 学习输入相关的门控，动态给每个模态加权，并**自带可解释性**——直接对应本研究的门控融合与 **RQ3**。
- 交叉注意力 [P111][P112]、共享/专属特征分解 [P109] 在 EO 基准上**一致优于朴素拼接**——支持把 Cross-Attention 作为对照臂。

**EN:** Useful predictors change over time and horizon [P072], yet a flat model gives *static* feature weights. Gated Multimodal Units [P096] learn input-dependent gates that weight each modality and provide built-in interpretability (→ RQ3); cross-attention [P111][P112] and shared/specific decompositions [P109] consistently beat naïve concatenation on EO benchmarks — motivating the Cross-Attention comparison arm.

### 1.4 高维、异构、缺失、异步：扁平模型处理不好 / High-dim, heterogeneous, missing, asynchronous — flat models handle these poorly

**中文：**

- **异构**：光学与雷达的通道结构/噪声不同 → 需模态专属编码器（CROMA [P105]，及 DOFA [P106] / OmniSat [P107] / TerraFM [P108] 一系）。
- **缺失模态**（云遮挡月度影像 + 发布滞后航运不可避免）：多模态 Transformer 在缺失输入下急剧退化，除非专门做缺失训练（Ma 等 [P097]）；ModDrop [P100]、ShaSpec [P114]、RobSense [P113]、PyViT-FUSE 的 band-drop [P110] 都是应对方案。
- **不规则/异步观测**（月度影像对齐周度价格）：GRU-D [P098]、mTAN [P099] 用掩码 + 距上次观测时间 / 连续时间嵌入处理。
- 这也解释了我扁平基线里"线性 Ridge 吃不下 113 维航运"（见 §1.6 / §4.1）。

**EN:** Sensors differ in channel structure and noise, so each needs a modality-specific encoder [P105][P106][P107][P108]. Missing modalities are unavoidable (cloud-limited monthly imagery + publication-lagged shipping): multimodal transformers degrade sharply unless trained for it [P097], addressed by ModDrop [P100], ShaSpec [P114], RobSense [P113] and band-drop [P110]. Irregular/asynchronous observation (monthly imagery vs weekly price) is handled by GRU-D [P098] and mTAN [P099]. This also explains why linear Ridge cannot use 113-d shipping in my flat baselines (§1.6/§4.1).

### 1.5 空白：现有油价 + 另类数据研究都停在扁平拼接 / The gap: existing work stops at flat fusion

**中文：** 少数把另类数据接到油价的研究，都止步于"多源异构特征融合（扁平表）"：Hao & Wang [P025]、Bricongne 等 [P069] 各喂**单一**工程信号进标准模型（后者预测需求而非价格）；PortWatch [P070]、Jung [P068] 把船舶/影像转成表格指标 nowcast **贸易**而非价格（且指出无单一融合配置全局最优）；方法最丰富的 GWNet-Attn [P063] 只有变量图、仅 WTI 期货、无影像/航运；Modality-aware Transformer [P039] 是文本+数值两模态、利率而非商品。**没有一个**在**周频 Brent** 上保留三个异构模态并做表示级融合——这就是本研究填的空白。

**EN:** The few alternative-data-to-oil studies all stop at multi-source heterogeneous *flat* feature fusion: [P025][P069] feed a single engineered signal into a standard model (the latter predicts demand, not price); [P070][P068] convert vessels/imagery into tabular indicators that nowcast *trade*, not price (and note no single fusion config is universally best); the richest precedents remain unimodal/bimodal — GWNet-Attn [P063] (variable graph, WTI futures only, no image/shipping) and the Modality-aware Transformer [P039] (text+numeric, interest rates). **None** preserves three heterogeneous modalities and fuses them at the representation level for weekly Brent.

### 1.6 我的扁平基线已实证暴露这个局限 / My own flat baselines already expose it empirically

**中文：**（数字见 §4.1）扁平拼接把遥感/航运接到强金融基线上 → 增量被稀释；**航运"模型敏感"**（XGB CW 0.0002 能用、线性 Ridge 0.264 用不了）；**最优子集因模型而异**（人工精选 core 对 XGB 反而不是最优）。这与 Costa [P072]"有用变量随时间/模型而变"一致，是扁平融合局限的直接症状。→ **假设**：保留结构 + 模态专属编码 + 门控融合，应能用上扁平**浪费掉**的信号。

**EN:** (Numbers in §4.1.) Flat fusion onto a strong financial baseline dilutes the increment; shipping is model-sensitive (XGB CW 0.0002 usable, linear Ridge 0.264 not); the optimal subset is model-dependent (hand-picked core not best for XGB) — consistent with [P072] and a direct symptom of the flat-fusion limitation. **Hypothesis:** preserving structure + modality-specific encoding + gated fusion should recover the signal flat fusion wastes.

## 2. 我的框架 / My framework

**中文：** 模态感知的时空融合框架——每个模态先由**专属编码器**学一个 32 维表示，再由**门控加权**动态组合，端到端预测下一周 Brent 价格。

```text
【三模态编码】
金融序列  ── Finance Encoder (TCN) ──────────────────────────────► z_fin  (32维)

卫星影像  ──► 冻结 Prithvi-EO-2.0 ──► embedding (1024维)
                                      │
                                      ▼
                            时间注意力 + AOI-site 注意力
                                      │
                                      ▼
                                   z_rs  (32维)

航运动态图 ── GAT(空间) → TCN(时间) → 节点注意力池化 ─────────────► z_ship (32维)

【融合与预测】
金融数据 ──► z_fin  ──┐
卫星数据 ──► z_rs   ──┼──► 门控加权 ──► z_fused ──► 回归头 ──► r̂（涨跌幅）
航运数据 ──► z_ship ──┘                              │
                                                    ▼
                              P̂_{t+1} = P_t × e^(r̂)  （预测下周油价，美元/桶）
```

**EN:** A modality-aware spatio-temporal fusion framework: each modality is first encoded into a 32-d representation, then combined by **gated weighting** into an end-to-end next-week Brent price predictor.

```text
[Modality encoders]
Financial series  ── Finance Encoder (TCN) ───────────────────────► z_fin  (32-d)

Satellite imagery ──► frozen Prithvi-EO-2.0 ──► embedding (1024-d)
                                              │
                                              ▼
                            temporal attention + AOI-site attention
                                              │
                                              ▼
                                           z_rs  (32-d)

Shipping graph    ── GAT(spatial) → TCN(temporal) → node-attn pool ─► z_ship (32-d)

[Fusion & prediction]
Finance  ──► z_fin  ──┐
RS       ──► z_rs   ──┼──► gated weighting ──► z_fused ──► regression head ──► r̂ (log return)
Shipping ──► z_ship ──┘                                              │
                                                                     ▼
                                           P̂_{t+1} = P_t × exp(r̂)  (next-week Brent, USD/bbl)
```





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

