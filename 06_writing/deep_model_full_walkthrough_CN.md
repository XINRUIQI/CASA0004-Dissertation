# 深度模型（创新层 / 方法集成层）完整流程详解

> **本文目的**：面向**完全不了解本项目**的读者，把「深度模型（创新层）」从下载数据、遥感去云、洗数据、卫星表示学习、航运图构建、三编码器融合、参数选择、回测训练，到结果、统计检验、可解释性、稳健性，**每一步全部写清楚**。
>
> **一句话定位**：深度模型是本论文的**方法集成与实证检验层（contribution 层）**。它**不**把遥感/航运加工成更多数值列拼成一张大宽表（那是「扁平模型」做的事，见 `flat_baseline_full_walkthrough_CN.md`），而是用**三个模态专属编码器**分别学习「金融状态 / 遥感活动 / 航运网络」的**表示（embedding）**，再通过**门控/交叉注意力**动态融合，端到端预测下一周 Brent 油价。
>
> **它要回答的核心问题（RQ2）**：相同数据下，**保留模态结构的表示级融合** 是否优于 **把所有列压成一张表的扁平融合**？
>
> 对应代码：`04_code/src/models/`（编码器 + 融合 + 回测内核）、`04_code/scripts/run_deep_*.py`（5 个入口脚本）；结果：`05_outputs/baselines/deep/`；总览：`00_admin/2026-07-05_研究方案与进度总览.md` §2.4。

---

## 目录

1. [先搞清楚：深度模型 vs 扁平模型](#1)
2. [第 0 步：原始数据下载（三模态）](#2)
3. [第 1 步：遥感去云与月度合成](#3)
4. [第 2 步：Prithvi 卫星表示预计算（冻结基础模型）](#4)
5. [第 3 步：航运 17 节点动态异质图构建](#5)
6. [第 4 步：金融序列准备](#6)
7. [第 5 步：深度数据集对齐（无泄漏、样本量）](#7)
8. [第 6 步：三个模态编码器（架构 + 参数）](#8)
9. [第 7 步：融合模块（门控 / 拼接 / 交叉注意力）](#9)
10. [第 8 步：回测协议（怎么训练的）](#10)
11. [第 9 步：主结果](#11)
12. [第 10 步：怎么分析结果（统计检验）](#12)
13. [第 11 步：稳健性检验](#13)
14. [第 12 步：可解释性（RQ3）](#14)
15. [第 13 步：进阶消融](#15)
16. [一句话总结 + 复现命令](#16)
17. [附录：完整变量与参数清单](#17)

---

<a name="1"></a>
## 1. 先搞清楚：深度模型 vs 扁平模型

**预测目标**（和扁平模型完全一致，才能公平对比）：下一周 Brent 现货价 \(P_{t+1}\)（美元/桶，周五截止）。模型不直接预测价格，而是预测**对数收益** \(r_{t+1}=\ln(P_{t+1}/P_t)\)，再还原成价格 \(\hat P_{t+1}=P_t\cdot e^{\hat r}\)。用对数收益是因为价格非平稳（有趋势），收益更平稳、更适合建模。

**两种融合范式的区别（这是全论文的卖点）**：

| | 扁平融合（baseline） | 表示级融合（本文深度模型） |
|---|---|---|
| 处理方式 | 三模态所有列**拍平**拼成一张宽表（212 列） | 三模态各自进**专属编码器**学 32 维表示 |
| 模型 | Ridge / XGBoost / LSTM early-fusion | 三编码器 + 门控/交叉注意力融合 |
| 模态结构 | 丢失（列之间无结构） | 保留（时序、图、AOI 站点结构都在编码器里） |
| 缺失模态 | 靠填充值硬塞 | 编码器 + 掩码 + modality dropout 显式建模 |

**贡献类型**：不发明新的融合算子/网络层/损失函数，而是把既有方法（冻结 EO 基础模型 + 模态专属编码器 + 门控/交叉注意力 + 缺失模态建模）**集成**为一个连贯系统，并**首次**在原油周频预测中、于统一无泄漏协议 + DM/Clark–West 检验下，系统对比「表示级融合 vs 扁平融合」。

**三个模态编码器一览**：

```text
金融序列 ──► Finance Encoder（因果 TCN）───────────────────► z_fin  (32维)
卫星影像 ──► 冻结 Prithvi-EO-2.0 → embedding
              └► 时间注意力 + AOI 站点注意力 ─────────────► z_rs   (32维)
航运动态图 ──► GAT(空间) → 因果 TCN(时间) → 节点注意力池化 ─► z_ship (32维)
      z_fin, z_rs, z_ship
        └► 门控交叉模态融合 ──► z_fused ──► 回归头 ──► r̂_{t+1} ──► 还原价格 P̂_{t+1}
```

---

<a name="2"></a>
## 2. 第 0 步：原始数据下载（三模态）

三类模态各有独立来源，全部存 `03_data/raw/`（原始层，整个 `raw/` 不进 git）。**深度模型与扁平模型共用完全相同的原始数据**，差别只在后续「怎么用」。

### 2.1 M1 金融/宏观（`raw/01_market_financial/`）

| 来源 | 内容 | 频率 |
|---|---|---|
| EIA 美国能源信息署 | Brent/WTI 现货价、周报 WPSR（库存/产量/进出口/炼厂投料） | 日频/周频 |
| FRED 美联储 | VIX、美元指数、10 年美债、联邦基金利率、工业原料指数 | 日频/月频 |
| Yahoo Finance | 标普500、原油波动率 OVX、Brent 期货、加元汇率、黄金 | 日频 |
| 学者维护 | Kilian 全球活动指数、Caldara–Iacoviello 地缘政治风险 GPR | 月频/日频 |

### 2.2 M2 遥感（`raw/02_sentinel2/`）

**两条通道**（深度模型主要用 Channel A，扁平模型用 Channel B）：

- **Channel A**（`Channel A/s2_patches/`）：11 个油站的**月度 Sentinel-2 6 波段影像切片**（GeoTIFF），从 Google Earth Engine 导出。波段 B2/B3/B4/B8A/B11/B12，正好对应 Prithvi 期望的 6 个 HLS 波段。**这是深度模型 z_rs 的输入源**。
- **Channel B**（`Channel B/*.csv`）：从影像预先算好的**月度光学指数 CSV**（NDVI/NDWI/NDBI/BSI 自 2017-04；VIIRS 夜光 NTL 自 2014-01）。扁平模型用这条。

**11 个 AOI 油站**（P001–P011）：Rotterdam（鹿特丹）、Fujairah（富查伊拉）、RasTanura（拉斯坦努拉）、Jurong（裕廊）、Houston（休斯顿）、NingboZhoushan（宁波舟山）、Jamnagar（贾姆纳格尔）、Basra（巴士拉）、Ulsan（蔚山）、Kharg（哈尔克岛）、Yanbu（延布）。类型分港口/炼厂/出口终端。

### 2.3 M3 航运（`raw/03_shipping/`）

| 来源 | 内容 | 频率 | 起始 |
|---|---|---|---|
| IMF PortWatch | 6 咽喉道过境油轮/货轮数量、运力；港口进出口油轮量 | 日频 | 2019+ |
| GFW（Global Fishing Watch） | 咽喉道 AIS 船舶存在小时数；AOI 港口访问事件+停泊时长；AOI↔AOI 航次 O-D；SAR 暗船检测 | 月频/事件 | 2012+ |

**6 个咽喉道**：hormuz（霍尔木兹）、suez（苏伊士）、malacca（马六甲）、mandeb（曼德海峡）、panama（巴拿马）、cape（好望角）。

---

<a name="3"></a>
## 3. 第 1 步：遥感去云与月度合成

卫星影像最大的敌人是**云**。这一步在 Google Earth Engine（GEE）里完成，得到「每个油站每月一张干净影像」。

### 3.1 去云 + 月度中值合成（GEE 端）

- **数据源**：`COPERNICUS/S2_SR_HARMONIZED`（Sentinel-2 地表反射率，已跨传感器谐一）。
- **去云逻辑**：对每个月内所有过境场景，用云概率 / SCL 场景分类波段剔除云、云影、雪等像素，再对该月剩余的干净像素做**中值合成（median composite）**，生成该月一张代表影像。中值合成能进一步压掉零星残云和异常值。
- **导出**：按 AOI 差异化切片大小导出（港口 6.4km / 炼厂 5.12km / 出口终端 2.56km），存为 6 波段 GeoTIFF（Channel A）；同时算好 NDVI/NDWI/NDBI/BSI 月度指数导出为 CSV（Channel B）。

### 3.2 切片质量校验（`s2_patch_utils.py`）

导出后不是所有切片都能用，用两道闸门筛：

1. **人工排除表** `s2_patch_exclusions.csv`：明显有问题的（残云重、错切）手动列入。
2. **有效像素比例** `has_valid_pixels()`：切片必须是 6 波段、且**非零像素 ≥ 0.5%**（`MIN_NONZERO_FRAC=0.005`），否则视为空导出（GEE 有时导出全零）。

只有 `valid_mask==1`（文件存在 + 未排除 + 像素有效）的切片才进入下一步表示计算。**全量约 967 个可用切片**（跨 11 站、2017-04 起）。

---

<a name="4"></a>
## 4. 第 2 步：Prithvi 卫星表示预计算（冻结基础模型）

这是深度模型区别于扁平模型的**第一个关键创新**：不再人工从影像算 5 个指数，而是用一个**遥感基础大模型**把整张影像编码成一个 1024 维向量（embedding），让模型自己去「看图」。

脚本：`03_data/processed/M2/py/precompute_s2_embeddings.py`

### 4.1 用哪个模型 + 为什么冻结

- **模型**：`ibm-nasa-geospatial/Prithvi-EO-2.0-300M`（IBM–NASA 联合的遥感基础模型，约 **3 亿参数**，ViT 架构，在海量 HLS 卫星影像上预训练）。从 HuggingFace 下载权重 `Prithvi_EO_V2_300M.pt`。
- **为什么正好能用**：导出的 6 波段（B2/B3/B4/B8A/B11/B12）**恰好等于** Prithvi 期望的 6 个 HLS 波段（Blue/Green/Red/NIR-narrow/SWIR1/SWIR2），同物理波段、同顺序，无需波段重映射，直接复用模型自带的 mean/std 归一化。
- **冻结（frozen）**：`model.eval()` + `requires_grad_(False)` + `torch.no_grad()`，**骨干网络一个参数都不训练**。这是小样本下的关键防过拟合措施——365 周的样本量根本喂不动 3 亿参数。下游只训一个轻量注意力头（第 6 步）。

### 4.2 每张切片 → 1024 维向量的流水线

对每个（站点，月份）切片（单帧 T=1）：

1. 读 6 波段 GeoTIFF → `float32`；
2. 归一化 `(x-mean)/std`（用 Prithvi config 自带的 mean/std）；把 0 值（nodata/切片边缘）映射为 Prithvi 约定常数 `1e-4`；
3. 双线性 resize 到 **224×224**（在 CPU 上做，避开 MPS 抗锯齿缺陷）；差异化切片大小 resize 后各站点恰好落在 Prithvi ~30m 训练分辨率附近；
4. 过 Prithvi `forward_features`（**不做 MAE 掩码**）→ 取最后一个 block（post-norm）输出 `[B, 197, 1024]`；
5. **两种池化**：对 196 个 patch token 做**均值池化 → meanpool 向量 [1024]**（主用）；同时保留 cls token（备选）。

### 4.3 产物

- `s2_prithvi_emb_meanpool.npy` `[N, 1024]`：主表示（深度模型用这个）。
- `s2_prithvi_emb_cls.npy` `[N, 1024]`：cls token 备选（稳健性对比，实测更差）。
- `s2_prithvi_emb_index.csv`：N 行对齐 `.npy` 行序 + 元数据（site_id、month、obs_month_start、cloud、n_scenes 等）。missing 的（站点,月份）**显式缺失**，不 ffill，供后续 as-of 对齐。

> 注意：总览提到 `[963, 1024]`——即约 963 张成功嵌入的切片。

---

<a name="5"></a>
## 5. 第 3 步：航运 17 节点动态异质图构建

这是**第二个关键创新**：不再把航运拍成一堆数值列，而是显式建成一张**图**——节点是港口和咽喉道，边是船的实际航次（O-D 流量），并且**每周动态变化**。

### 5.1 清洗与滞后（`aggregate_shipping_to_weekly.py` + `build_m3_graph_weekly.py`）

- **统一到 W-FRI（周五截止周）**：PortWatch 日频求和到周；GFW 月频 ffill 到周；用 **union 索引**（不是取交集），确保 GFW 早期（2012+）和 PortWatch 晚期样本都不丢（修复了旧版 727→362 掉样本 bug）。
- **发布滞后（无泄漏关键）**：值只能在其真实发布后使用——PortWatch **+1 周**，GFW 港访/航次 **+2 周**，GFW SAR 暗船月频 **+4 周**。wow_pct/4w_ma 等派生量在滞后**之前**先算好，再整体前移。
- **数据质量裁剪**：港口停泊时长上限 720h（30 天，超出=AIS 长停/拼接伪影 → NaN）；航次过境上限 90 天（超出=中途未观测停靠 → 均值置 NaN，但边仍计数）。

### 5.2 17 节点异质图的组装（`build_m3_graph17.py`）

**节点（17 个）** = 11 AOI 油站（P001–P011）+ 6 咽喉道，节点顺序固定（AOI 0–10，咽喉 11–16）。

**为什么叫"异质图"**：AOI 节点和咽喉节点的**特征空间不同**（F_aoi ≠ F_choke），所以编码器要用**节点类型专属投影**再共享消息传递。

- **AOI 节点特征（11 维/节点）**：`pw_portcalls_tanker/cargo`、`pw_import_tanker`、`pw_export_tanker`、`gfw_n_visits`、`gfw_dwell_hrs_mean/median`、`gfw_self_loops`、`sar_detections_total/dark`、`sar_dark_share`。
- **咽喉节点特征（20 维/节点）**：GFW 8 项（total_hours、total_vessels、cargo/bunker/other_hours、other_share、total_hours_mom_pct、mean_presence）+ PortWatch 9 项（n_tanker、n_total、capacity、tanker_share 等）+ SAR 3 项。

**边（adjacency，每周一张 17×17）**：
- **动态 O-D 边**（AOI→AOI）：来自 GFW 航次数 `n_voyages`，**每周不同**，反映真实船流。
- **静态 AOI↔咽喉边**：按地理归属固定连接（如 hormuz 连 P002/P003/P008/P010；malacca 连 P004/P006/P009），每周都在。

**产物**：`m3_graph17_tensors.npz`，含 `aoi_features (T,11,11)`、`choke_features (T,6,20)`、`adjacency (T,17,17)` 等，T 为周数。

---

<a name="6"></a>
## 6. 第 4 步：金融序列准备

金融模态**不做图、不做影像**，直接用 M1 的 31 列周频序列（Brent/WTI、EIA 库存/产量/进出口/炼厂、VIX、DXY、利率、S&P500、期货基差、GPR 等）。

- 从合并矩阵 `weekly_feature_matrix.csv` 取 M1 列（`data.select_features(dico, "M1")`）。
- **过去向填充**（`fill_features`）：ffill（只用过去，无未来）+ 开头残余 NaN 填 0（中性）。得到覆盖所有 W-FRI 的 `fin_df`（levels 水平值，不做差分——因为编码器内部有 LayerNorm + 过去向标准化）。

---

<a name="7"></a>
## 7. 第 5 步：深度数据集对齐（无泄漏、样本量）

脚本：`04_code/src/models/deep_dataset.py` → `build_deep_dataset()`。这一步把三模态**对齐到同一套周度目标/索引**（与扁平基线 `build_dataset` 完全相同的周），这是深度 vs 扁平公平对比的前提。

### 7.1 目标与索引

- 用扁平构建器（lookback=1）先取出 `idx`（可用周）、目标 `r_next`（=r_{t+1}）、`P_t`、`P_next`、`r_now`。窗口 **2019-01-01 ~ 2025-12-31**。

### 7.2 滑窗（lookback）

对每个可用日 d，构造过去 `lookback` 周的窗口。**主模型 lookback=4 周**（导师设定 + 对齐扁平协议）。每个样本包含：
- `aoi (L, 11, 11)`、`choke (L, 6, 20)`、`adj (L, 17, 17)`：航运图窗口；
- `fin (L, 31)`：金融序列窗口；
- `rs (L, 11, 1024)`、`rs_mask (L, 11)`：遥感 embedding 窗口 + 可用掩码。

只有**图窗口和金融窗口都完整**的日才保留 → 得到 N 个对齐样本。

### 7.3 遥感的无泄漏 as-of 对齐 + 站内去均值

- **as-of 对齐**：月度 Prithvi embedding 的**可用日 = 月末 + 15 天**（`RS_PUB_LAG_DAYS=15`，保守发布延迟）。每周用 `merge_asof(backward)` 取**最近一个已发布**的月度 embedding，绝不用未来。缺失处 NaN + mask=0。
- **站内过去向去均值**（`meanpool_anom` 可选）：对每个站点，按时间顺序做 expanding（含当月、无未来）去均值 `a_j = e_j - mean(e_1..e_j)`。目的是剥掉「这是哪个站」的静态场景签名（占冻结 embedding 方差约 80%），让 RS 分支只看**时间异常**。主结果用原始 meanpool。

### 7.4 过去向标准化（每 fold 重新拟合）

**标准化不在这里做**，必须在回测每个 fold 上**只用训练片**拟合（`fit_scalers`），杜绝泄漏：
- aoi/choke/fin/rs 在特征轴上 z-score（nan-aware，std 下限保护）；
- adj 不标准化（只用其 >0 的连接模式当注意力掩码）；
- 目标 `r_next` 也做 per-fold 标准化（`(r-mean)/std`），预测后还原。

### 7.5 样本量

- 合并矩阵：**365 周 × 212 列**（31 M1 + 55 M2 + 113 M3 + 11 mask + 2 target）。
- 深度对齐后的可用样本 N 视 lookback 而定；回测 `min_train=104`（前 104 周暖机不测），**共同测试期 257 周（2021–2025）**。

---

<a name="8"></a>
## 8. 第 6 步：三个模态编码器（架构 + 参数）

代码：`04_code/src/models/{finance,rs,shipping}_encoder.py`。三者都输出 **32 维（d=32）** 表示。为什么统一 32 维小维度？小样本下高维必过拟合（超参 sweep 证实 d=64 一律变差）。

### 8.1 z_fin — 金融 TCN 编码器（`finance_encoder.py`）

输入 `(B, L, 31)` → 输出 `z_fin (B, 32)`。
- 线性投影 31→32 + LayerNorm；
- **因果 TCN**（`TemporalTCN`）：2 层 1D 卷积，kernel=3，残差连接，dropout；
- **自适应膨胀**：lookback ≤ 5（如主模型 4）用**连续卷积（dilation=1）**；lookback ≥ 6 用**指数膨胀（1,2,4…）**扩大感受野。这是 sweep 证据驱动的设计——短窗密采样更好，长窗需要膨胀。
- 取最后时间步 → 线性头 + ReLU → 32 维。

### 8.2 z_rs — 遥感 Prithvi 注意力编码器（`rs_encoder.py`）

输入 `rs (B, L, 11, 1024)` + `mask (B, L, 11)` → 输出 `z_rs (B, 32)` + 站点注意力 `(B, 11)`。**Prithvi 骨干已冻结**，这里只学轻量注意力：
- 线性投影 1024→64 + LayerNorm + dropout；
- **时间注意力**（每站在 lookback 上）：用可学习 query 打分，被 mask（可用性）遮蔽 → softmax 加权 → 每站一个向量；
- **站点注意力**（在 11 个 AOI 上）：可学习 query 打分（被「窗口内是否有有效观测」遮蔽）→ softmax → 加权求和 → 32 维。站点注意力权重供 RQ3 解读「更看重哪个油站」。

### 8.3 z_ship — 航运图 GAT + TCN 编码器（`shipping_encoder.py`）

输入 `aoi (B,L,11,11)`、`choke (B,L,6,20)`、`adj (B,L,17,17)` → `z_ship (B,32)` + 节点注意力 `(B,17)`。约 **4.2 万参数**。
- **异质输入**：AOI/咽喉各自线性投影到 d_model=64 + 节点类型 embedding（0=AOI,1=咽喉）+ LayerNorm；
- **空间 GAT**（`DenseGATLayer`）：2 层多头（heads=4）稠密图注意力（17×17 用布尔掩码，因图小、动态，稠密比稀疏 torch_geometric 更简洁）。邻接对称化 + 自环。**创新点 P1-4**：把 **O-D 流量对数 `log1p(flow)` 作为注意力先验**（可学习增益 `edge_scale`）——繁忙航道天然获得更高注意力，而不是被布尔邻接丢掉边权。航运增量因此走强。
- **时间 TCN**：每节点在 lookback 上过因果 TCN（同样自适应膨胀）；
- **节点注意力池化**：加性注意力在 17 节点上打分 → softmax → 图表示 32 维。节点权重供 RQ3 解读「更关注哪个港口/咽喉」。

---

<a name="9"></a>
## 9. 第 7 步：融合模块（门控 / 拼接 / 交叉注意力）

代码：`04_code/src/models/fusion.py` → `DeepForecastModel`。把若干模态的 32 维表示融成一个 32 维 `z_fused`，再过回归头 → r̂。三种融合是 RQ2 的「融合阶梯」：

### 9.1 GatedFusion（门控，主模型 ✅）
`alpha = softmax(MLP([z_1..z_m]))`；`z = Σ alpha_i · z_i`。一个小 MLP 根据当前样本算出各模态权重（凸组合），**每个样本、每周都可不同**。`alpha` 就是 RQ3 的「模态门控权重」（这周更信金融还是航运）。

### 9.2 ConcatFusion（编码器拼接，阶梯地板）
`z = ReLU(Linear([z_1..z_m]))`。模态仍各自编码，但用**固定 MLP 混合**，无逐样本门控、无交叉注意力。它是对照：门控/交叉注意力比它好的部分，才能归因于「融合机制」本身，而非「分模态编码」。

### 9.3 CrossModalAttentionFusion（交叉注意力，进阶）
金融 `z_fin` 作 **Query**，去 attend 遥感/航运的**节点 token**（RS 11 站 + 航运 17 节点 = 28 个 token）：`z = LN(z_fin + gamma·CrossAttn(z_fin, tokens))`，4 头。交叉注意力权重供 RQ3 解读「金融状态在关注哪个节点/航道」。

### 9.4 回归头 + modality dropout
- **回归头**：`Linear(32,32)→ReLU→Dropout→Linear(32,1)` → 标量 r̂。
- **modality dropout**（可选，ModDrop 风格）：训练时按概率随机丢整个模态（至少留 1 个），提升缺失模态鲁棒性。

**配置表**（`deep_rolling.CONFIGS`）：`fin`(仅金融)、`ship`(仅航运)、`rs`(仅遥感)、`fusion`(金融+航运,门控)、`finrs`(金融+遥感,门控)、`m4rep`(三模态,门控)、`m4xattn`(三模态,交叉注意力)、`m4concat`(三模态,拼接)。

---

<a name="10"></a>
## 10. 第 8 步：回测协议（怎么训练的）

代码：`04_code/src/models/deep_rolling.py` → `rolling_origin_deep()`。**与扁平基线逐字对齐**，保证公平。

### 10.1 Rolling-origin（滚动起点 / walk-forward）
- **扩张窗口**：`min_train=104`（前 104 周只训不测），此后逐周向前测；
- **每 13 周重训一次**（`retrain_every=13`）：每个 fold 只用**该测试周之前**的样本训练（`slice(0, i)`），严格无未来；
- 每个测试周用其所属 fold 的模型预测这一周，还原价格 `P̂ = P_t·exp(r̂)`。

### 10.2 单 fold 训练细节（`_train_fold`）
- 训练片内再切出**最后 52 周做 inner-validation** 早停（patience=12）；
- 优化器 **Adam**，lr=1e-3，weight_decay=1e-4，batch=32，epochs=80（上限，早停会提前停）；
- 损失 **MSE**（在标准化后的 r 上）；梯度裁剪 max_norm=5.0；
- 目标 per-fold 标准化，预测后 `r̂·r_std + r_mean` 还原。

### 10.3 关键超参（主模型）
lookback=**4**、d=**32**、gat_layers=**2**、tcn_layers=**2**、dropout=0.1、seed=42、融合=**gated**。

> ⚠️ 工程坑：macOS 上 xgboost 与 torch 同进程会因重复 OpenMP 段错误，故深度脚本**读取**扁平 M1 预测（`baseline_predictions.csv`），不在同进程重跑 xgb。

---

<a name="11"></a>
## 11. 第 9 步：主结果

来源：`run_deep_baseline.py` → `05_outputs/baselines/deep/deep_metrics.csv`（257 共同测试周 2021–2025，seed=42，lookback=4）。

| 模型 | RMSE | skill vs M0 | 方向准确率 | CW vs M0 | DM vs M1 |
|---|--:|--:|--:|--:|--:|
| M0 随机游走 | 4.152 | 0.0% | – | – | – |
| M1_Ridge / XGB（扁平参照） | 4.256 / 4.368 | −2.5% / −5.2% | 0.498 / 0.553 | 0.530 / 0.104 | — |
| Mfin（z_fin TCN） | 4.250 | −2.4% | 0.494 | 0.315 | — |
| Mrs（z_rs Prithvi） | 4.247 | −2.3% | 0.459 | 0.928 | 0.457 |
| Mship（z_ship 图） | 4.168 | −0.4% | 0.506 | 0.496 | 0.106 |
| **Mfinship（金融+航运，门控）** | **4.147** | **+0.11%** | 0.529 | 0.166 | 0.061 |
| Mfinrs（金融+遥感） | 4.253 | −2.4% | 0.475 | 0.769 | 0.485 |
| Mfull / M4rep（三模态门控） | 4.205 | −1.3% | 0.502 | 0.894 | 0.239 |
| Mconcat（三模态拼接） | 4.320 | −4.1% | 0.494 | 0.637 | 0.650 |

**RMSE = 价格上的均方根误差（越低越好）；skill = 相对随机游走的改进（>0 才算赢过 RW）。**

### 11.1 融合矩阵（RQ2 融合阶梯：3 模态组合 × 3 融合方式）

来源：`run_deep_fusion_matrix.py` → `deep_fusion_matrix.csv` / `deep_fusion_matrix.png`（257 周，seed=42，lookback=4，epochs=80，同一协议一次跑齐 9 格）。这是把「每个融合模型 × 三种拼接方式」补全后的完整矩阵（项目实现的融合机制就是这三种：Encoder-Concat / Gated / Cross-Attention）。

**skill vs M0（%），>0 才赢过随机游走**：

| 模态组合 | Encoder-Concat | Gated 门控 | Cross-Attention |
|---|--:|--:|--:|
| **Mfinship**（金融+航运） | +0.06 | +0.11 | **+0.74** |
| **Mfinrs**（金融+遥感） | −1.93 | −2.43 | −5.89 |
| **Mfull**（三模态） | −4.06 | −1.28 | +0.12 |

**每格完整指标**：

| 组合 | 融合方式 | RMSE | skill vs M0 | 方向准确率 | CW vs M0 | DM vs M1 |
|---|---|--:|--:|--:|--:|--:|
| Mfinship | Concat | 4.149 | +0.06% | 0.525 | 0.195 | **0.043** ✅ |
| Mfinship | Gated | 4.147 | +0.11% | 0.529 | 0.166 | 0.061 |
| **Mfinship** | **Cross-Attn** | **4.121** | **+0.74%** | 0.549 | **0.041** ✅ | 0.055 |
| Mfinrs | Concat | 4.232 | −1.93% | 0.533 | 0.971 | 0.373 |
| Mfinrs | Gated | 4.253 | −2.4% | 0.475 | 0.769 | 0.485 |
| Mfinrs | Cross-Attn | 4.396 | −5.89% | 0.455 | 0.898 | 0.913 |
| Mfull | Concat | 4.320 | −4.1% | 0.494 | 0.637 | 0.650 |
| Mfull | Gated | 4.205 | −1.3% | 0.502 | 0.894 | 0.239 |
| **Mfull** | **Cross-Attn** | **4.147** | **+0.12%** | 0.564 | **0.018** ✅ | 0.088 |

**矩阵读出的新发现**：

1. **金融+航运（Mfinship）三种融合全部 skill>0**，是最强的模态组合；其 **Cross-Attention 为全场最佳**（+0.74%、RMSE 4.121、CWvsM0 **0.041** 显著击败随机游走）——甚至优于三模态 Mfull，说明遥感加进来是净拖累。
2. **金融+遥感（Mfinrs）三种融合全部 skill<0**，且 Cross-Attention 最差（−5.89%）——遥感噪声被交叉注意力放大，再次印证遥感模态内在弱。
3. **Cross-Attention 只对「含航运」的组合有效**（Mfinship +0.74、Mfull +0.12，CWvsM0 均显著），对「金融+遥感」是灾难——融合机制的价值取决于模态本身是否有信号，而非机制本身。
4. **门控 vs 拼接**：三模态上门控明显优于拼接（−1.28% vs −4.06%）；双模态上两者接近（都被航运信号带动）。
5. ⚠️ 以上为**单 seed（42）**结论，尤其 Cross-Attention 需配合多 seed 稳健性看——§13 已指出 xattn 多 seed 方差大（±2.76、seed2 崩盘）；故主模型仍锁 **Gated**，Cross-Attention 列为「上限高但不稳」的进阶结果。

---

<a name="12"></a>
## 12. 第 10 步：怎么分析结果（统计检验）

光看 RMSE 差一点点没意义，必须做**统计显著性检验**（`04_code/src/backtest/metrics.py`）。本项目最讲究的是**用对检验**：

### 12.1 三种指标
- **RMSE skill vs M0**：`1 - RMSE_model/RMSE_M0`，>0 表示比随机游走准。
- **方向准确率（DirAcc）**：预测涨跌方向对的比例（辅助指标，不进损失）。
- **RMSE、MAE**：价格误差绝对量。

### 12.2 两种检验，别用错（本项目的严谨点）
- **Clark–West（CW，嵌套用）**：当小模型**嵌套**在大模型里（如「金融」嵌套在「金融+航运」；随机游走 r̂=0 嵌套在任何模型里）。CW 修正了嵌套模型下 DM 的偏差。
  - **CW vs M0**：诚实回答「能否击败随机游走？」——M0 嵌套在任何模型里，合法。
  - **深度内部模态增量**（fusion vs fin、m4rep vs fusion、finrs vs fin）也是嵌套 → 用 CW。
- **Diebold–Mariano（DM，非嵌套用，HLN 小样本修正）**：当两模型**不嵌套**（模型类 + 特征集都不同，如深度 vs 扁平 M1、门控 vs 拼接）。此时 CW 会被夸大，DM 才对。

> **2026-07-07 口径修订（重要）**：此前把「深度 vs 扁平 M1」当嵌套用 CW（偏乐观），改用严格 **DM(non-nested)** 后，RQ2「表示级显著优于扁平」的结论从强结论**下调**为「方向一致、未达显著」。这是诚实科研的体现。

### 12.3 结果怎么读（`deep_cw.csv`）
- **航运增量（Mfinship vs Mfin）嵌套 CW = 0.00057** ✅——在金融表示上加航运表示**显著**降 MSE，这是 RQ2「表示级融合有用」**最干净的正向证据**。
- **遥感增量（Mfinrs vs Mfin）CW = 0.019** ✅，但**加到「金融+航运」之上无增量**（M4rep vs Mfinship CW=0.78 ✗，被航运挤掉）。
- **深度 vs 扁平 M1（DM）均 >0.05**（Mship 0.106 最接近）：改严格检验后不再显著。
- **门控 vs 拼接（DM=0.22）**：门控更好但未达显著；拼接 Mconcat 全场最差（−4.1%）。

### 12.4 六点核心发现
1. 仅 **Mfinship（金融+航运）skill 转正（+0.11%）**，唯一 skill>0 的深度模型；但 CWvsM0=0.166 仍不显著——**无一显著击败 M0**（全项目诚实主结论）。
2. **航运表示是最硬的正向证据**（嵌套 CW 0.00057），GAT 引入 O-D 流量先验后进一步走强。
3. vs 扁平 M1 改用严格 DM 后**不再显著**（较此前偏乐观口径明确下调）。
4. **门控 > 朴素拼接**（架构消融方向一致）。
5. **遥感表示内在弱**：Mrs −2.3%、DirAcc 0.459（<0.5），cls 更差（−11%），加到 fin+ship 无增量——冻结 Prithvi 单模态对**周频**油价帮助有限（与扁平 M2 结论一致）。
6. 以上强化 RQ2：弱增量需要模态感知融合才能方向一致叠加。

---

<a name="13"></a>
## 13. 第 11 步：稳健性检验

脚本：`run_deep_sweep.py` → `deep_sweep_summary.csv`（lookback=4）。目的：证明主结论不是单 seed / 单超参碰运气。

- **多 seed（各 3 seed：42/1/2）**：fusion −0.47% ± **0.86（方差最小=最稳）**；m4rep −0.89% ± 0.60；m4xattn −1.83% ± **2.76（最不稳，seed2 崩到 −4.98%）**。CWvsM0 基本 0/3，DMvsM1 fusion 1/3——**稳的是门控，不稳的是交叉注意力**。
- **超参 sweep（fusion,seed42）**：lb=8 d=32 最优（+0.34%，DMvsM1 0.041）> lb=4（+0.11%）> lb=12（负）；**d=64 一律变差**（佐证「编码器维度要小」）。**主模型仍锁 lb=4** 以对齐扁平做公平对比。
- **RS 正则网格（P1-5）**：meanpool 全负（最佳 −0.90%）、cls −11%——**RS 弱是内在的，调参救不了**。
- **主 fusion 正则网格（P1-6）**：整片 skill≈0，dp=0.3 略好（+0.29%）——**主模型对正则稳健**。

---

<a name="14"></a>
## 14. 第 12 步：可解释性（RQ3）

脚本：`run_deep_interpret.py`（门控 + 节点注意力）+ `run_deep_xattn_viz.py`（交叉注意力）。walk-forward 时额外记录每周的融合 info dict，绝不看未来。

- **模态门控均值（门控主模型）**：金融 **0.44** > 遥感 **0.348** > 航运 **0.212**。把门控权重画成时间堆叠图，叠上已知供给/地缘事件线（俄乌 2022-02、EU 禁俄油 2022-06、OPEC+ 意外减产 2023-04、胡塞红海袭击 2023-11）看时间吻合度。
- **航运节点注意力**：最关注 **霍尔木兹咽喉** + P003/P009/P001/P005（对应红海绕行/出口终端叙事）。
- **遥感站点注意力**：最关注 P004/P008/P009/P001/P006（出口终端 AOI）。
- **交叉注意力（金融 Query 对 28 token）**：→ 航运 **0.575** / → 遥感 **0.425**；Top token 是遥感站但权重高度均匀（~0.04），注意力「很平」→ 呼应遥感信息弱/冗余。
- **一个讨论点**：两种融合机制对「遥感 vs 航运」倚重**相反**（门控 RS>航运；xattn 航运>RS）。

产物：`deep_gate_weekly.csv`、`deep_interpret.png`、`deep_xattn_weekly.csv`、`deep_xattn_viz.png`。

---

<a name="15"></a>
## 15. 第 13 步：进阶消融

脚本：`run_deep_advanced.py` → `deep_advanced_summary.csv`（lookback=4, seed=42）。

| arm | skill vs M0 | DirAcc | CWvsM0 | DMvsM1 |
|---|--:|--:|--:|--:|
| M4concat（拼接） | −4.06% | 0.494 | 0.637 | 0.650 |
| M4rep 门控 | −1.28% | 0.502 | 0.894 | 0.239 |
| **M4 交叉注意力（金融 Query）** | **+0.12%** | 0.564 | **0.018** | 0.088 |
| M4rep + modality dropout 0.3 | −0.19% | 0.498 | 0.316 | 0.096 |
| **M4-xattn + dropout 0.3** | **+0.62%** | 0.560 | **0.008** | **0.050** |

- **交叉注意力单 seed 可达最佳**：seed42+lb4 下 skill 转正、且是**唯一 CWvsM0 显著击败随机游走**的配置（0.018，加 dropout 后 0.008 且 DMvsM1 0.050 双显著）；**但多 seed 极不稳（±2.76）→ 列为「上限存在但不稳」的进阶结果，主模型仍是门控**。
- **子期**：xattn 早/晚期均稳定 +0.12%；gated 早期 −2.35% / 晚期（2023–25）转正 +0.49%。

---

<a name="16"></a>
## 16. 一句话总结 + 复现命令

**一句话**：RQ1 增量弱（遥感尤甚，全模型 CWvsM0 均不显著、仍难破 M0）；**RQ2 部分正向**——最硬证据是「金融表示上加航运表示」嵌套 CW **0.00057** 显著、且门控>拼接，但用严格 DM 对照扁平 M1 **未达显著**（方向一致）；**RQ3 门控与注意力可解释**（金融主导、聚焦霍尔木兹与出口终端，两融合机制倚重相反）。交叉注意力单 seed 可最佳但多 seed 不稳，列进阶。

**复现命令**：
```bash
# 0) 前置：先跑扁平 M1（深度脚本要读其预测）
python3 04_code/scripts/run_baseline.py --modality M1

# 1) 上游数据（如需重建）
python3 03_data/processed/M2/py/build_m2_weekly.py                 # 遥感月→周
python3 03_data/processed/M2/py/precompute_s2_embeddings.py        # Prithvi embedding
python3 03_data/processed/M3/py/aggregate_shipping_to_weekly.py    # 航运→周
python3 03_data/processed/M3/py/build_m3_graph_weekly.py           # AOI 图
python3 03_data/processed/M3/py/build_m3_graph17.py                # 17 节点图张量

# 2) 深度主结果 + 检验
python3 04_code/scripts/run_deep_baseline.py                      # 主结果表
python3 04_code/scripts/run_deep_sweep.py                         # 稳健性 sweep
python3 04_code/scripts/run_deep_interpret.py                     # RQ3 门控/节点注意力
python3 04_code/scripts/run_deep_xattn_viz.py                     # RQ3 交叉注意力
python3 04_code/scripts/run_deep_advanced.py                     # 进阶消融
```

---

<a name="17"></a>
## 17. 附录：完整变量与参数清单

### A. 数据规格
| 项 | 值 |
|---|---|
| 统一窗口 | 2019-01-01 ~ 2025-12-31 |
| 合并矩阵 | 365 周 × 212 列（31 M1 + 55 M2 + 113 M3 + 11 mask + 2 target） |
| 共同测试期 | 257 周（2021–2025） |
| 预测目标 | r_{t+1}=ln(P_{t+1}/P_t)，还原价格 P̂=P_t·e^r̂ |
| lookback（主） | 4 周 |
| RS 发布滞后 | 月末 + 15 天（as-of backward） |
| 航运滞后 | PortWatch +1w、GFW 港访/航次 +2w、SAR +4w |

### B. 模型/训练超参
| 项 | 值 |
|---|---|
| 表示维度 d | 32（各编码器输出） |
| GAT | 2 层，4 头，d_model=64，O-D 流量对数先验 |
| TCN | 2 层，kernel=3，自适应膨胀（lb≤5 用 dilation=1） |
| RS 编码器 | 冻结 Prithvi-EO-2.0-300M（3 亿参数）+ 时间/站点注意力，投影 1024→64→32 |
| 融合 | 门控（主）/ 拼接 / 交叉注意力（4 头） |
| 优化器 | Adam，lr=1e-3，weight_decay=1e-4 |
| batch / epochs | 32 / 80（inner-val 早停 patience=12） |
| 梯度裁剪 | max_norm=5.0 |
| 回测 | min_train=104，retrain_every=13，seed=42 |

### C. 图节点与咽喉
- **11 AOI**：P001 Rotterdam、P002 Fujairah、P003 RasTanura、P004 Jurong、P005 Houston、P006 NingboZhoushan、P007 Jamnagar、P008 Basra、P009 Ulsan、P010 Kharg、P011 Yanbu。
- **6 咽喉**：hormuz、suez、malacca、mandeb、panama、cape。

### D. 关键文件
| 类别 | 路径 |
|---|---|
| 编码器 | `04_code/src/models/{finance,rs,shipping}_encoder.py` |
| 融合 | `04_code/src/models/fusion.py` |
| 数据对齐 | `04_code/src/models/deep_dataset.py` |
| 回测内核 | `04_code/src/models/deep_rolling.py` |
| 入口脚本 | `04_code/scripts/run_deep_{baseline,sweep,interpret,advanced,xattn_viz}.py` |
| 图张量 | `03_data/processed/M3/outputs/m3_graph17_tensors.npz` |
| Prithvi embedding | `03_data/processed/M2/outputs/s2_prithvi_emb_meanpool.npy` |
| 结果 | `05_outputs/baselines/deep/` |
