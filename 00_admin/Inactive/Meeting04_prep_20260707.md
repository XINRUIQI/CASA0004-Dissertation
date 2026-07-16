# Phase 03 Progress Report

---

## 0. 汇报逻辑 / How I'll walk through this

**中文：** 我们**上次见面时只有扁平基线层**（把所有列拼成一张宽表喂给模型）。这次的主线是**多模态融合框架**。我按四步讲：

1. **为什么**要融合多模态，而不是简单把所有列扁平拼接；
2. 我的**框架**长什么样；
3. 这个阶段具体**做了什么**；
4. **结果**如何。

**EN:** When we last met, only the **flat baseline layer** existed (all columns concatenated into one wide table). Today's through-line is the **multimodal fusion framework**, in four steps: (1) **why** fuse rather than flat-concatenate everything; (2) what my **framework** looks like; (3) what I **did** this phase; (4) the **results**.

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

### 1.7 一表看清：proposal 每一步 ↔ 文献支撑 / One table: every design step ↔ its literature


| 设计决策 / Design step                   | 文献支撑 / Literature                                                                                               |
| ------------------------------------ | --------------------------------------------------------------------------------------------------------------- |
| 始终对照随机游走 + 机制特征 + DM/Clark–West 严格评估 | Alquist–Kilian–Vigfusson [P053]、Kilian [P052]、Baumeister–Kilian [P054]、Diebold–Mariano [P058]、Clark–West (2007) |
| 多模态 + 按模态消融（RQ1）                     | 预测组合 [P054]；多模态分类学 [P101]                                                                                       |
| **表示级融合 vs 扁平拼接**（RQ2 核心）            | 分类学 [P101]；金融时序先例 [P039]；模态专属编码 [P105]                                                                          |
| 航运当**图**（GAT+TCN），而非计数列              | [P062][P066][P091][P070]                                                                                        |
| 遥感用**冻结 EO 基础模型** embedding          | Prithvi [P094]、SatMAE [P095]、CROMA [P105]                                                                       |
| 用**站内标准化异常**而非原始辐射/NDVI              | [P024][P032][P025]                                                                                              |
| **门控**动态加权 + 可解释（RQ3）                | Gated MU [P096]；预测变量随时变 [P072]                                                                                  |
| **交叉注意力**融合对照臂                       | [P039][P111][P112][P109]                                                                                        |
| **缺失模态**建模（modality-dropout / 重建）    | [P097][P100][P114][P113][P110]                                                                                  |
| **不规则/异步**时间对齐                       | GRU-D [P098]、mTAN [P099]                                                                                        |
| 金融/时序编码器（TCN/Transformer）            | TCN 最强 [P001]；TFT [P089]                                                                                        |
| **SHAP** 模态级归因（解释而非因果）               | Lundberg–Lee [P059]                                                                                             |


**中文一句话：** 从"对照随机游走"到"门控融合"再到"缺失模态处理"，proposal 的每一步都有文献先例；**本研究的新意不在发明新算子，而在把它们集成、并首次在周频 Brent 上做公平的表示级 vs 扁平对照。**

**EN bottom line:** from benchmarking against the random walk to gated fusion to missing-modality handling, every step has a precedent; the novelty is not a new operator but **integrating them and running the first fair representation-vs-flat comparison on weekly Brent.**

---



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


| 编码器 / Encoder      | 结构 / Architecture                                           | 为什么这样设计（含文献）/ Rationale (with lit.)                                                                      |
| ------------------ | ----------------------------------------------------------- | -------------------------------------------------------------------------------------------------------- |
| **z_fin** 金融       | Linear + LayerNorm + 2 层因果 TCN                              | 规则时序 → 卷积捕捉短期动量/均值回复；TCN 在油价上被证为最强之一 [P001]，TFT [P089] 备选                                                |
| **z_ship** 航运      | 类型专属投影 + node-type embedding + 2 层多头 GAT + 因果 TCN + 节点注意力池化 | 把航运当**图**：GAT 学空间（港口↔咽喉）、TCN 学时间；17 节点（11 AOI + 6 咽喉）动态异质图。图建模先例 [P062][P066][P091]，序列源 PortWatch [P070] |
| **z_rs** 遥感        | **冻结 Prithvi-EO-2.0** → 1024 维 embedding → 时间注意力 + 站点注意力 → 32 维 | 影像先经 EO 基础模型编码（backbone 不微调 [P094][P095]），再经轻量注意力聚合时空信号；模态专属编码 [P105]                              |
| **GatedFusion** 融合 | softmax 门控凸组合 z_fused = Σαᵢzᵢ                               | 按时点动态给模态权重（→ RQ3）[P096]；对照臂 Encoder-Concat（= 早融合稻草人 [P101]）与 Cross-Attention [P039][P111][P112]          |
| **回归头** / head     | MLP(32→32→1) → r̂；还原 P̂ = P_t·e^(r̂)                        | 预测对数收益而非价格水平（更平稳）；与扁平基线同一目标定义，保证公平对比                                                              |

**研究问题映射 / RQ mapping：**

- **RQ1 增量价值**：加遥感/航运是否在金融基线之上提升样本外预测？
- **RQ2 融合方式**：模态感知**表示级**融合是否优于**扁平**特征融合？（← 本框架的核心对照）
- **RQ3 可解释性**：门控权重能否揭示不同时期依赖哪种模态、哪个港口/航道？

---



## 3. 这个阶段做了什么 / What I did this phase



### 3.1 数据地基：无泄漏特征矩阵 / Leakage-safe feature matrix

**中文：** 三模态各自在自己的脚本内完成发布滞后，merge 只复查。合并矩阵 = **365 周 × 212 列**（31 金融 + 55 遥感 + 113 航运 + 11 mask + 2 target），无泄漏自检全通过。遥感按你 Meeting 03 的要求**删除云量、仅留有效观测数**；另建 17 节点动态异质图张量作为航运 GNN 的输入。

**EN:** Each modality applies its own publication lag; the merge layer only re-checks. Merged matrix = **365 weeks × 212 columns**, all leakage checks pass. Per your Meeting 03 note, RS **drops cloud fraction, keeps valid-observation count**; I also built a 17-node dynamic heterogeneous graph tensor as the shipping GNN input.

### 3.2 扁平 M0–M4：作为公平对照的标尺 / Flat M0–M4 as the fair comparison scale

**中文：** 四类基线、同一协议：**M0 随机游走 · Ridge · XGBoost · 深度 LSTM 早融合**。这一层既是 RQ1 的答案，也是 RQ2 里"扁平"这一边的对照，还提供了 §1.6 的动机证据。配套 DM / Clark–West / SHAP / 多种 leave-one-out 稳健性（含 M2 水体掩膜、M3 LOCHO 七臂）。

**EN:** Four baseline families under one protocol (**random-walk M0, Ridge, XGBoost, deep LSTM early-fusion**) — the RQ1 answer, the "flat" side of the RQ2 contrast, and the motivating evidence for §1.6, with full DM / Clark–West / SHAP / leave-one-out robustness.

### 3.3 方法集成层：三编码器 + 门控融合端到端跑通（本阶段核心新增）/ Integration layer end-to-end (the main new work)

**中文：** 把 §2 的框架**实现并跑通**：三个模态专属编码器 + 门控融合 + 回归头，walk-forward 协议与扁平一致（lookback=8，min_train=104，retrain_every=13，强正则 + inner-val early-stopping）。已实现的臂：单模态 `fin`/`ship`/`rs`、`fusion`（fin+ship）、`m4rep`（fin+rs+ship 门控）、`concat`（Encoder-Concat 对照）。深度模型全程数值稳定、无负 R² 崩溃（回应 Meeting 03 的过拟合担忧）。

**EN:** I **implemented and ran** the §2 framework end-to-end: three modality-specific encoders + gated fusion + regression head, same walk-forward protocol as the flat layer, with strong regularisation and inner-validation early-stopping. Arms implemented: single-modality `fin`/`ship`/`rs`, `fusion` (fin+ship), `m4rep` (gated fin+rs+ship), and `concat` (Encoder-Concat comparison). Deep models are numerically stable throughout — no negative-R² collapse (addressing your Meeting 03 overfitting caution).

### 3.4 文献 / 写作 / 上次要求 / Literature, writing, Meeting 03 asks

**中文：**

- **文献闭环**：补齐融合层文献 P094–P115（冻结 EO 基础模型、门控多模态单元、GRU-D/mTAN、ModDrop 缺失模态、多模态综述）——是"方法骨架"，无一在油价上验证过，有效性须靠本项目消融自证。
- **写作**：Ch2 文献综述草稿 v1（5 主题，~5 页）；Ch3 方法部分成型（研究设计 + AOI 选择 + RS 方法）。
- **Meeting 03 八项要求**：M0 ✅ / 统一协议 ✅ / 明确价格目标 ✅ / 变量描述列 ✅ / 删云量 ✅ / 机制 EDA ✅ / 深度模型防过拟合 ✅ / 按主题写文献 ✅。

**EN:** Closed the fusion-layer literature (P094–P115, architectural scaffolding, none validated on oil prices); Ch2 lit-review draft v1 (5 themes) and a partial Ch3 methodology; and all eight Meeting 03 asks addressed (M0, unified protocol, price target, description columns, cloud-fraction removal, mechanism EDA, overfitting guards, thematic lit review).

---



## 4. 结果如何 / Results



### 4.1 扁平层（标尺 + §1.6 的动机证据）/ Flat layer (the scale + the §1.6 motivation)

257 测试周，M0 RMSE=4.152：


| 模态 / Modality     | 模型 / Model  | RMSE          | skill vs M0   | CW_p vs M1（嵌套增量）     |
| ----------------- | ----------- | ------------- | ------------- | -------------------- |
| **M0** 随机游走 / RW  | —           | **4.152**     | 0.0%          | —                    |
| M1 金融 / finance   | Ridge / XGB | 4.256 / 4.368 | −2.5% / −5.2% | —                    |
| M2 +遥感 (anom-55)  | Ridge / XGB | 4.414 / 4.440 | −6.3% / −6.9% | 0.474 / 0.085        |
| M3 +航运 (full-113) | Ridge / XGB | 4.430 / 4.429 | −6.7% / −6.7% | 0.264 / **0.0002** ✅ |
| M4 全模态 (199)      | Ridge / XGB | 4.525 / 4.507 | −9.0% / −8.6% | 0.314 / **0.009** ✅  |


> `skill vs M0` > 0 才算超过随机游走；`CW_p` < 0.05 = 相对 M1 有显著嵌套增量（**不等于**击败 M0）。
> 读法：**扁平层无一模型超过 M0**，且航运的增量高度依赖模型（XGB 能用、Ridge 用不了）——正是 §1 说的扁平局限。



### 4.2 表示级融合（回报）/ Representation-level fusion (the payoff)

253 共同测试周，M0 RMSE=4.172：


| 模型 / Model                    | RMSE      | skill vs M0  | CW_p vs 扁平 M1 / flat M1    |
| ----------------------------- | --------- | ------------ | -------------------------- |
| M0 随机游走 / RW                  | 4.172     | 0.0%         | —                          |
| M1 金融（扁平 Ridge）/ flat         | 4.279     | −2.6%        | —                          |
| Mfin (TCN)                    | 4.206     | −0.8%        | —                          |
| Mrs (frozen Prithvi)          | 4.307     | −3.3%        | 0.103                      |
| **Mship (GAT+TCN)**           | 4.189     | −0.4%        | **0.0017** ✅               |
| **Mfusion (fin+ship, gated)** | **4.158** | **+0.34%** ✅ | shipping incr. **0.043** ✅ |
| Mconcat (encoder-concat)      | 4.210     | −0.9%        | **0.017** ✅                |
| Mfull (fin+rs+ship, gated)    | 4.218     | −1.1%        | **0.0046** ✅               |




### 4.3 RQ2 关键对照（最值得讲的一页）/ The RQ2 headline

**中文：** **同一份航运数据，处理方式决定了它有没有用：**

- 扁平早融合（113 列航运堆进一个 RNN）：`M3_LSTM` CW p=0.46，**利用不了**。
- 模态感知编码器（GAT 学空间 + TCN 学时间）：`Mship` CW p=0.0017，**显著**；与金融门控融合后 `Mfusion` 是**所有模型里第一个把 RMSE 压到 M0 以下**（skill +0.34%，DM 尚不显著 p=0.35，方向属实）。

这正是 RQ2「表示级融合 vs 扁平融合」要的对照——**同一份数据、同一套协议，差别只在融合方式**。

**EN:** **The same shipping data is useful or not depending on how it is processed.** Flat early-fusion (`M3_LSTM`) is CW-insignificant (p=0.46); the modality-aware encoder (`Mship`) is significant (CW p=0.0017), and gated fusion with finance (`Mfusion`) is the **first model to edge below M0** (skill +0.34%; DM p=0.35, not yet significant). Same data, same protocol — only the fusion differs.

### 4.4 SHAP + 一句话结论 / SHAP + bottom line


| 模态 / Modality          | SHAP 占比 / share (M4 XGB) |
| ---------------------- | ------------------------ |
| M3 航运 / shipping       | **51.8%**                |
| M1 金融 / finance        | 33.7%                    |
| M2 遥感 / remote sensing | 14.5%                    |


**中文一句话：** 扁平层诚实地打不过随机游走，且浪费掉航运信号；一旦保留模态结构做表示级融合，**同一份航运信号变得可用、并首次逼近 M0**——这就是"为什么要融合"的实证答案。

**EN bottom line:** the flat layer honestly cannot beat the random walk and wastes the shipping signal; once modality structure is preserved via representation-level fusion, the *same* shipping signal becomes usable and first approaches M0 — the empirical answer to "why fuse."

RQ2 对照：M4 SHAP 模态贡献 / M4 SHAP by modality

---



## 5. 待与导师确认 / Questions for supervisor

**中文：**

1. **贡献定位**：确认「不提新算子/层/损失，而是**集成**既有方法 + **首次**在原油周频价格预测中系统检验表示级 vs 扁平融合」是否站得住？
2. **诚实核心结论怎么写**：无一模型显著击败 M0，但表示级融合有显著嵌套增量且首次逼近 M0——这个故事作为 Distinction 论文是否足够，还是需要更强的绝对精度目标？
3. **创新层收尾范围**：Cross-Attention + modality-dropout（缺失模态）+ 多 seed 是否都进正文，还是主推门控、其余入 Appendix？
4. **写作顺序**：建议 Ch3 方法 → Ch4 结果 → 回填 Ch2/Ch1，是否认可？

**EN:**

1. **Contribution framing** — is "integrate existing methods + first systematic representation-vs-flat fusion test in weekly crude-oil price forecasting" defensible?
2. **Presenting the honest core result** — no model significantly beats M0, yet representation-level fusion gives a significant nested increment and first approaches M0. Strong enough for a Distinction, or do I need a stronger absolute-accuracy target?
3. **Scope to finish the integration layer** — Cross-Attention + modality-dropout + multi-seed all in the main text, or lead with gating and appendix the rest?
4. **Writing order** — Ch3 → Ch4 → backfill Ch2/Ch1. Agree?

---



## 6. 下一步 / Next steps

**中文：**

- 创新层收尾：Cross-Attention 臂 + modality-dropout（缺失模态建模）+ RS 分支超参 sweep（当前 `Mrs` 偏弱）；门控/站点注意力可视化（RQ3）。
- flat vs modality-aware 正式对照：深度 253 周 / 扁平 257 周取交集并列成表（正式回答 RQ2）。
- 写作：Ch3 方法（机制变量 + 无泄漏对齐 + 编码器/融合）、Ch4 结果（M0–M4 + DM/CW + SHAP + 稳健性）。

**EN:** Finish the integration layer (Cross-Attention, modality-dropout, RS-branch sweep, gating/site-attention visualisations for RQ3); align deep (253-week) and flat (257-week) windows into one formal RQ2 table; write Ch3 and Ch4.

---



## 附录：可复现产物 / Appendix: reproducibility


| 类别 / Item                     | 路径 / Path                                                                          |
| ----------------------------- | ---------------------------------------------------------------------------------- |
| 合并特征矩阵 / merged matrix        | `03_data/processed/merge/outputs/weekly_feature_matrix.csv`（365×212）               |
| 扁平基线入口 / flat entry           | `04_code/scripts/flat/run_baseline.py`                                                  |
| 深度融合入口 / deep entry           | `04_code/scripts/deep/run_deep_baseline.py` + `04_code/scripts/deep/run_deep_sweep.py`                       |
| 三编码器 + 融合 / encoders + fusion | `04_code/src/models/{finance_encoder,shipping_encoder,rs_encoder,fusion}.py`       |
| 17 节点图张量 / graph tensors      | `03_data/processed/M3/outputs/m3_graph17_tensors.npz`                              |
| 扁平结果记录 / flat results log     | `00_admin/待整理/flat_baseline_log.md`（§4/§8/§9/§12/§13）                              |
| 深度结果 / deep results           | `05_outputs/baselines/Deep/_cross/deep_metrics.csv` + `deep_cw.csv` + `deep_backtest.png` |


深度融合回测 / deep fusion backtest