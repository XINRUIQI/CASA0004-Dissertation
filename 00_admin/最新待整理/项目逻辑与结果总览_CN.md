# 项目逻辑与结果总览

> **最后更新**：2026-07-28  
> **用途**：一页讲清「逻辑 + 主结果 + RQ 回答」；细节数字与稳健性表见 `2026-07-15_研究方案与进度总览.md`。  
> **配套**：扁平变量 → `2026-07-28_扁平模型变量清单.md`；进度 → `2026-07-15_研究方案与进度总览.md`；结构 → `2026-07-07_File Structure.md`。

---

## 0. 一句话定位

**拟定标题**：A Modality-Aware Spatio-Temporal Fusion Framework for Brent Crude Oil Forecasting Using Financial Time Series, Satellite Imagery and Maritime Networks  
（融合金融时序、卫星影像与海运网络的**模态感知时空**布伦特原油价格预测框架）

**核心问题**：周频 Brent 油价极难预测（随机游走 M0 是强基准）。本项目检验：① 遥感 / 航运等另类数据能否在样本外带来增量；② 把多模态「拍平拼表」是否足够，还是必须保留模态结构的**表示级融合**；③ 不同市场时期各模态、各港口/站点谁更重要。

**贡献类型**：不是发明新网络层，而是把冻结 EO 基础模型 + 模态专属编码器 + 门控/交叉注意力 + 无泄漏回测 + DM/Clark–West 检验**集成**为一套系统，并在**同一协议**下首次系统对比「扁平融合 vs 表示级融合」。

---

## 1. 研究问题（RQ）


| RQ      | 问题                                       | 对应方法                                   | 结果章节     |
| ------- | ---------------------------------------- | -------------------------------------- | -------- |
| **RQ1** | 遥感 / 航运是否在 M1 金融基线之上、相对 M0 随机游走，带来样本外增量？ | M0–M4 消融阶梯；CW vs M1；DM vs M0           | §4.2–4.3 |
| **RQ2** | 相同数据下，表示级融合是否优于扁平特征拼接？                   | Flat Ridge/XGB vs Deep 编码器+融合；按信息集配对比较 | §4.4     |
| **RQ3** | 能否解释不同市场条件下各模态、各空间节点的相对重要性？              | Flat SHAP；Deep 门控 α、节点/站点注意力           | §4.6     |


---



## 2. 项目逻辑：两层架构 × 消融阶梯



### 2.1 预测目标（全项目统一）

- **对外目标**：下一周 Brent 现货价 P_{t+1}（美元/桶，周五截止，W-FRI）
- **训练目标**：对数收益 r_{t+1}=\ln(P_{t+1}/P_t)，还原 \hat P_{t+1}=P_t\cdot e^{\hat r}
- **样本窗**：2019–2025，合并矩阵 **365 周 × 213 列**（含 `week_ending_friday`；数据列 212 = 31+55+113+11 avail+2 target）；共同测试期 **257 周（~2021–2025）**
- **M0 基准 RMSE**：≈ **4.152** $/bbl



### 2.2 消融阶梯 M0 → M4（逐层加模态）


| 信息集    | 内容                            | 扁平特征宽度（L=4 展平）  |
| ------ | ----------------------------- | --------------- |
| **M0** | 随机游走：\hat r=0，\hat P=P_t      | —               |
| **M1** | 金融/宏观（31 列）                   | 31×4 = **124**  |
| **M2** | M1 + 遥感季节异常 anom（55 列，11 AOI） | 86×4 = **344**  |
| **M3** | M1 + 航运 full tier（113 列）      | 144×4 = **576** |
| **M4** | M1 + M2 + M3 全模态              | 199×4 = **796** |




### 2.3 两条建模臂（同一数据、同一协议、公平对比）

```text
                    ┌─────────────────────────────────────────┐
                    │         统一无泄漏回测协议               │
                    │  lookback=4 · min_train=104 ·           │
                    │  retrain_every=13 · inner-val=52w       │
                    └─────────────────────────────────────────┘
                                        │
              ┌─────────────────────────┴─────────────────────────┐
              ▼                                                   ▼
    【扁平臂 Flat Baseline】                          【深度臂 Deep / 表示级融合】
    三模态列拍平 → 宽表                              三模态各自进专属编码器 → 32维表示
    Ridge / XGBoost                                  门控 / 拼接 / 交叉注意力融合
    核心实证对照层                                    方法集成与 RQ2 对照层
```

**设计原则**：

- 主分析用**最朴素的全量口径**（M2 全 55 列 anom、M3 full 113 列），不挑最优子集，避免 p-hacking
- 稀疏化 / 水体掩膜 / LOAO / LOMO 等作为**稳健性臂**，加固「扁平吃不动 → 需要融合」的主线
- 同时报告 **嵌套增量（CW vs M1）** 与 **绝对技能（skill / DM vs M0）**——二者含义不同



### 2.4 公平比较协议（锁定参数）


| 参数            | 值                                                                |
| ------------- | ---------------------------------------------------------------- |
| 日历            | W-FRI，2019–2025                                                  |
| lookback      | 4 周（主）                                                           |
| min_train     | 104 周（≈2 年 warm-up）                                              |
| retrain_every | 13 周（约季度重训）                                                      |
| 内层验证          | 训练折最后 52 周                                                       |
| n_test        | 257 周                                                            |
| 指标            | RMSE、MAE、DirAcc；skill vs M0 = 1-\mathrm{RMSE}/\mathrm{RMSE}_{M0} |
| 检验            | DM(HLN) vs M0；Clark–West vs M1（嵌套 M2/3/4）；深度 vs 扁平配对 DM          |


---



## 3. 三个模态与数据处理



### 3.1 M1 金融/宏观（31 列）

- **来源**：EIA（Brent/WTI、WPSR 库存/产量/进出口/炼厂）、FRED（VIX、DXY、利率等）、Yahoo（S&P500、OVX、期货、黄金、汇率）、Kilian REA、GPR
- **关键滞后**：EIA WPSR +1 周；月频宏观 +1~5 周
- **脚本**：`03_data/processed/M1/py/build_m1_weekly.py`



### 3.2 M2 遥感（55 列 anom / 深度用 Prithvi embedding）

- **11 个 AOI**（5 km 缓冲）：Houston、Rotterdam★、Ningbo-Zhoushan、Jamnagar、Jurong、Ulsan、Basra、Fujairah★、Kharg、Ras Tanura★、Yanbu★（★= 文献核心 NTL 站点）
- **扁平臂**：Sentinel-2 月度 NDVI/NDWI/NDBI/BSI + VIIRS 夜光 → 站点内季节异常 anom（55 列）
- **深度臂**：冻结 **Prithvi-EO-2.0** patch embedding（1024-d meanpool）→ 时间注意力 + 站点注意力 → 32-d
- **对齐**：月频 as-of + PUB_LAG 15 天；云筛选 CLOUDY≤60
- **脚本**：`03_data/processed/M2/py/build_m2_weekly.py`



### 3.3 M3 航运（113 列 full / 深度用 17 节点图）

- **6 咽喉**：Hormuz、Suez、Malacca、Mandeb、Panama、Cape
- **来源**：IMF PortWatch（油轮过境/港口流量）、GFW AIS（船舶存在/航次）
- **扁平臂**：113 列 full tier（core 38 列仅作稳健性——对 XGB 反而最弱）
- **深度臂**：**17 节点动态异质图**（11 AOI + 6 咽喉）→ GAT + TCN + 节点注意力 → 32-d
- **关键滞后（扁平 vs 深度勿混写）**：
  - 扁平：GFW **月频 presence +4 周**；PortWatch **+1 周**
  - 深度图：GFW **event/航次/O-D +2 周**；SAR 暗船 **+4 周**（未进扁平 113 列）
- **脚本**：`03_data/processed/M3/py/aggregate_shipping_to_weekly.py`、`build_m3_graph17.py`



### 3.4 合并矩阵

- `03_data/processed/merge/outputs/weekly_feature_matrix.csv`：**365 周 × 213 列**（去掉日期索引后常称 212 数据列）
- 长历史：`weekly_feature_matrix_full.csv`（~1067 周，主分析未用）
- 无泄漏：as-of 合并、发布滞后、训练折内 fit scaler/筛选器
- 变量清单：`00_admin/最新待整理/2026-07-28_扁平模型变量清单.md`

---



## 4. 模型设计



### 4.1 扁平模型（M0–M4）


| 模型                             | 说明                                                                     |
| ------------------------------ | ---------------------------------------------------------------------- |
| **M0**                         | 规则基准，不训练                                                               |
| **Ridge**                      | VarianceThreshold → StandardScaler → Ridge(α)；网格 α∈{0.1,1,10,100,1000} |
| **XGBoost**                    | VarianceThreshold → XGBRegressor；浅树 + 子采样 + L2（小 n 高 p 设计）             |
| **LSTM early-fusion**（历史，§5.5） | 宽表列 → 单一共享 RNN；2026-07-03 跑过 M3 CW≈0.46 不显著；产物已被表示级入口覆盖                |


- 入口：`04_code/scripts/flat/run_baseline.py`（Ridge/XGB 主路径）
- 每层配套：回测、DM/CW、SHAP、稳健性 sweep



### 4.2 深度模型（表示级融合）

**三个编码器（各 → 32 维）**：


| 编码器        | 输入                | 架构                                          |
| ---------- | ----------------- | ------------------------------------------- |
| **z_fin**  | (L, 31) 金融序列      | 因果 TCN（2 层，自适应膨胀）→ 投影                       |
| **z_rs**   | 冻结 Prithvi 1024-d | proj → 时间注意力（每站）→ 站点注意力（11 AOI）             |
| **z_ship** | 17 节点异质图          | 类型投影 + 2 层 GAT（O-D 流量作注意力先验）+ TCN + 节点注意力池化 |


**融合方式（RQ2 对照）**：


| 融合             | 机制                                                              | 角色               |
| -------------- | --------------------------------------------------------------- | ---------------- |
| **Concat**     | 编码器输出直接拼接 → MLP                                                 | 地板 / 对照          |
| **Gated（主模型）** | \alpha=\mathrm{softmax}(\mathrm{MLP}([z_i]))，z=\sum\alpha_i z_i | 主分析              |
| **Cross-Attn** | 金融作 Query，对 28 个 RS/航运 token                                    | 进阶（单 seed 最佳但不稳） |


**训练**：Adam lr=1e-3，batch=32，epochs≤80，早停 patience=12；可选 modality dropout 0.3

**深度配置命名**：`fin`、`rs`、`ship`、`m3_deep_gated`（fusion）、`m2_deep_gated`、`m4_deep_gated`（gated 全模态）、`m4_deep_xattn`、`m4_deep_concat`

- 代码：`04_code/src/models/`（`finance_encoder.py`、`rs` 相关、`shipping_encoder.py`、`fusion.py`）
- 入口：`04_code/scripts/deep/run_deep_{baseline,sweep,interpret,advanced,xattn_viz}.py`

---



## 5. 已完成的工作



### 5.1 数据管线 ✅

- [x] M1 单脚本无泄漏周频构建（35→31 列精简版）
- [x] M2 月→周 as-of + 11 AOI Sentinel-2 Channel B + expanding anomaly
- [x] M3 PortWatch/GFW 三源统一到 W-FRI；17 节点图张量 `m3_graph17_tensors.npz`
- [x] 合并矩阵 365×212 + 全变量数据字典
- [x] Prithvi-EO embedding 预计算（深度 RS 通道）



### 5.2 扁平基线全套 ✅

- [x] M0–M4 × Ridge/XGB 主回测（L4_tuned）
- [x] DM vs M0、Clark–West vs M1
- [x] 各层 SHAP 可解释性
- [x] 稳健性：lookback sweep、M2 LOAO/水体掩膜/literature 臂/C2 降维、M3 LOCHO、M4 LOMO
- [x] 产物：`05_outputs/baselines/Flat/M*_Flat/`



### 5.3 深度表示级融合 ✅

- [x] 三模态编码器 + 门控/拼接/交叉注意力融合
- [x] 与扁平同协议 rolling-origin 回测（主协议 lookback=**4**）
- [x] 多 seed / lookback / dropout / RS pool 稳健性 sweep
- [x] RQ3：门控权重、节点/站点注意力、交叉注意力可视化
- [x] **RQ3 多 seed 稳定性（2026-07-16）**：seeds {42,1,2}；Hormuz Top-5 = 3/3；α_shipping 周度相关弱 → 正文只写跨 seed 稳定焦点
- [x] 产物：`05_outputs/baselines/Deep/`（`M*_Deep/` + `_cross/`；含 `deep_gate_stability.csv`）



### 5.4 写作与文档（2026-07-28 刷新）

- [x] Meeting 04（2026-07-08）：停止大改模型 → Phase 04 写作为主（`research_diary_phase4.md`）
- [x] 论文 outline 最新版：`06_writing/Outline/20260728_outline_brief.md`（按 Taylor 反馈修订）
- [x] Ch2 文献综述最新双语稿：`Chapter 2  Literature Review/20260728_literature_review_双语.md`
- [x] 章节双语草稿：Ch1 / Ch3 / Ch4 / Ch5 / Ch6；Appendix A–C
- [x] 扁平/深度完整流程 walkthrough（中英）；进度总览 / 项目逻辑 / 变量清单已对齐
- [ ] **当前主线**：按 07-28 outline **合稿润色**（Ch1 空白+回应；Ch3 先 M0；Ch4 按 RQ + skill vs M0；Ch5 Implications；Ch6 连续段落）
- [ ] Meeting 05（2026-07-29）读稿 → 之后进 `07_submission/`



### 5.5 探索与决策记录（试过 / 讨论过，但未入主分析）

> **原则**：主分析锚最朴素口径；下列项要么作**对照/地板**、要么作**稳健性臂**、要么**仅文献/Backlog**——不等于「没做」。


| 类别                  | 内容                                             | 做了什么                                                               | 为何未入主分析                                                                                             |
| ------------------- | ---------------------------------------------- | ------------------------------------------------------------------ | --------------------------------------------------------------------------------------------------- |
| **深度 early-fusion** | LSTM/GRU 把所有列 reshape 成 `[L, F]` 喂**单一共享 RNN** | 2026-07-03 跑过 M1–M4；例：`M3_LSTM` RMSE 4.370、CW vs M1 **0.460（不显著）** | 与表示级管线共用 `04_code/scripts/deep/run_deep_baseline.py`，07-05 被模态感知入口**覆盖**；产物需 git 回溯。结论：扁平深度早融合同样吃不动航运 → 支撑 RQ2           |
| **融合机制**            | Encoder-Concat                                 | 已跑（M4concat skill −4.1%）                                           | 作 RQ2 **地板对照**；门控/xattn 均优于 concat                                                                  |
| **融合机制**            | Cross-Attention                                | 已跑 + 多 seed + 子期                                                   | 单 seed 最佳（finship xattn skill +0.74%，CW vs M0≈0.018）；**多 seed ±2.76 极不稳** → 列**进阶**，主模型仍为 **Gated** |
| **M3 特征 tier**      | core 38 列                                      | LOCHO 7 臂完整跑过                                                      | XGB 最弱且不显著（CW 0.096）；full 113 显著 → core **降为稳健性臂**                                                  |
| **M2 稀疏/去噪臂**       | literature(4 NTL)、aoi4(4站×5指数)、ntlall、水体掩膜 B4  | 六臂 2×2 稀疏 + 掩膜全套                                                   | 部分 CW 显著（aoi4 最强 0.0055），但升主分析 = **p-hacking + 自拆 RQ2 主线**                                          |
| **RS embedding**    | Prithvi **cls** vs **meanpool**                | sweep 已比                                                           | cls 更差（约 −11% skill）；主用 meanpool                                                                    |
| **RS 预处理**          | meanpool_anom（站点 expanding 去均值）                | 代码/文档有选项                                                           | 主结果用 raw meanpool；anom 版未锁为主                                                                        |
| **EO 骨干**           | SatMAE                                         | 文献入库 P095                                                          | 实现选 **Prithvi-EO-2.0**；SatMAE 未接入                                                                   |
| **金融编码器**           | TFT [P089]                                     | 文献讨论                                                               | 小样本 + 无法保留 RS/航运结构 → 选 **因果 TCN**；TFT 不作独立基线                                                        |
| **金融编码器**           | GRU                                            | `finance_encoder.py` 文档提及 TCN/GRU                                  | 仅实现 TCN；GRU 在 backlog                                                                               |
| **模态 M5**           | GDELT 地缘文本                                     | Meeting 03 决策                                                      | **仅 Appendix**，不纳入 M0–M4 主消融                                                                        |
| **M2 特征合约**         | level / all(110) / 早期误用 154 列                  | level 合约曾定义                                                        | 矩阵无 level 列；主分析锁 **anom-55**                                                                        |
| **M1 变量**           | 38 列旧版、AUD 商品货币、独立波动率目标                        | 已精简重跑                                                              | 31 列 + 唯一目标=价格；波动率预测已取消                                                                             |
| **M2 QC**           | S2 cloud fraction 入模                           | Meeting 03 删除                                                      | 保留 valid_obs_count；云量仅 QC                                                                           |
| **M3 扩展**           | GFW SAR 暗船、EMODnet 栅格 zonal                    | 部分 processed 产物                                                    | 主模型未用；列 **Backlog**                                                                                 |
| **超参（深度）**          | lookback=8、d=64                                | sweep 已跑                                                           | lb=8 单 seed skill 更好，但 **lb=4 锁死**以对齐扁平；d=64 一律更差                                                   |


---



## 6. 核心结果



### 6.1 扁平模型（RQ1）

**主结果（L4_tuned，n=257，M0=4.152）**：


| 模型             | RMSE          | skill vs M0   | CW_p vs M1           |
| -------------- | ------------- | ------------- | -------------------- |
| M0 随机游走        | 4.152         | —             | —                    |
| M1 Ridge / XGB | 4.256 / 4.368 | −2.5% / −5.2% | —                    |
| M2 Ridge / XGB | 4.414 / 4.440 | −6.3% / −6.9% | 0.474 / 0.085        |
| M3 Ridge / XGB | 4.430 / 4.429 | −6.7% / −6.7% | 0.264 / **0.0002** ✓ |
| M4 Ridge / XGB | 4.525 / 4.507 | −9.0% / −8.6% | 0.314 / **0.009** ✓  |


**要点**：

1. **无一扁平模型击败 M0**（skill 全 < 0）——周频 Brent 随机游走极强
2. **遥感 M2 全量 anom**：嵌套增量不显著（CW 0.085~0.474）；稀疏 literature 臂 / 水体掩膜可勉强显著
3. **航运 M3 XGB**：CW vs M1 **高度显著（0.0002）**，但 RMSE 仍差于 M1——嵌套显著 ≠ 绝对优于 M0
4. **M4 XGB**：CW 显著、RMSE 最差——信号存在，扁平宽表无法干净 harvest
5. SHAP：M3/M4 航运模态占比最高（~52%），霍尔木兹/苏伊士 tanker 信号突出



### 6.2 深度模型（RQ1 深度臂 + 融合消融）

**主结果（seed=42，L=4，n=257）**：


| 配置                      | RMSE      | skill vs M0 | 备注                      |
| ----------------------- | --------- | ----------- | ----------------------- |
| M1_Deep（金融 TCN）         | 4.250     | −2.4%       | ≈ 扁平 M1 Ridge           |
| M_rs_deep（遥感）           | 4.247     | −2.3%       | 弱                       |
| M_ship_GNN（航运图）         | 4.168     | −0.4%       | 最接近 M0                  |
| **M3_Deep_gated gated** | **4.147** | **+0.11%**  | 首个 skill>0              |
| M2_Deep_gated gated     | 4.253     | −2.4%       | RS 拖累                   |
| M4rep gated             | 4.205     | −1.3%       | 全模态门控                   |
| M4concat                | 4.320     | −4.1%       | 最差融合                    |
| **M3_Deep_gated xattn** | **4.121** | **+0.74%**  | 全场最佳 RMSE               |
| M4 xattn                | 4.147     | +0.12%      | CW vs M0 ≈ 0.018 ✓      |
| M4 xattn + drop0.3      | 4.126     | +0.62%      | CW≈0.008；DM vs M1≈0.050 |


**要点**：

1. **金融+航运**（finship）：三种融合均 skill>0；xattn 最佳但多 seed 不稳定（±2.76）
2. **金融+遥感**（finrs）：三种融合均 skill<0——RS 在周频油价上内在弱
3. **全模态 M4**：concat 最差；加 RS 到 fin+ship 常**损害**绝对 RMSE
4. 门控 > 朴素拼接（架构消融方向一致）
5. 航运嵌套增量（fusion vs fin）CW **0.00057** 显著——深度内部最硬证据



### 6.3 扁平 vs 深度配对（RQ2 核心）

**按信息集固定模态内容、只换架构**：


| 配对  | Flat RMSE | Deep RMSE | skill flat | skill deep | DM vs XGB   | DM vs Ridge |
| --- | --------- | --------- | ---------- | ---------- | ----------- | ----------- |
| M1  | 4.368     | 4.250     | −5.2%      | −2.4%      | 0.097       | 0.466       |
| M2  | 4.440     | 4.253     | −7.0%      | −2.4%      | **0.042** ✓ | 0.096       |
| M3  | 4.429     | 4.147     | −6.7%      | +0.11%     | **0.010** ✓ | 0.062       |
| M4  | 4.507     | 4.205     | −8.6%      | −1.3%      | **0.005** ✓ | **0.036** ✓ |


**要点**：

- 深度在 **M2/M3/M4（尤其含航运）** 上显著优于扁平 XGB；M1  alone 未达 5% 显著
- 不能宣称「深度永远更好」——增益主要在**多模态 + 航运**场景
- 结论表述：*表示级融合在选定多模态设定下优于扁平对照，尤其当航运进入信息集时*



### 6.4 可解释性（RQ3）


| 层级 | 发现 |
| --- | --- |
| **Flat SHAP** | M3 霍尔木兹 tanker share、苏伊士 wow；M4 模态占比 shipping≈52% > finance≈34% > RS≈15% |
| **Deep 门控 α** | 均值 finance≈0.44 > RS≈0.35 > shipping≈0.21；跨 seed 排序稳定 |
| **多 seed 稳定性（07-16）** | Hormuz 进 Top-5 = **3/3**；RS 稳定站：Ningbo / Rotterdam；α_shipping 周度相关弱；事件窗仅俄乌 3/3 同向上升可写，红海窗不写死 |
| **Deep 节点注意力** | 航运图聚焦霍尔木兹 + 出口终端；写作只宣称跨 seed 稳定焦点 |
| **Cross-Attn** | 金融 Query 对航运 token≈0.575 > RS≈0.425；与门控倚重方向相反；xattn 解释性放附录 |
| **caveat** | 高 α_shipping ≠ 模型「在看 Hormuz」——空间细节在节点/站点注意力层；关联≠因果 |




### 6.5 稳健性检验完整清单（已全部跑过 ✅）

**设计逻辑**：主分析各层只有**一个**正式口径（§2.3）；下列实验回答「换滞后/换子集/去掉一块后结论是否仍成立」，**加固**主结论，**不替代**主表。

#### 扁平臂


| 实验                       | 层           | 改了什么                                      | 主要发现                                                         | 脚本/产物                                    |
| ------------------------ | ----------- | ----------------------------------------- | ------------------------------------------------------------ | ---------------------------------------- |
| **Lookback sweep**       | M1/M2/M3/M4 | L∈{1,4,8,12}；L4_returns                   | Ridge 随 L 恶化；XGB 对 L 较稳；**L4_tuned 锁主**                      | `sweep_m1.py` 等 → `sweep_*_overview.png` |
| **feature-mode=returns** | 各层          | 趋势列改收益口径                                  | 数值稳健性；定性结论不变                                                 | `sweep_m1.py` L4_returns                 |
| **LOAO**                 | M2          | 每次去掉 1 个 AOI（11 站）                        | 增量弥散、非单站驱动                                                   | `run_baseline.py --leave-one-aoi-out`    |
| **2×2 稀疏 + 水体掩膜**        | M2          | aoi4 / ntlall / literature / watermask 六臂 | 全 55 不显著=**双重稀疏**；aoi4 CW≈0.0055 最强但仍不升主分析                   | `robustness_m2.py`、B4 掩膜管线               |
| **C2 降维**                | M2          | PCA / ElasticNet 对照                       | PCA 对 XGB 去共线增益最大；回应 P058                                    | `robustness_m2.py`（需先 SHAP）              |
| **contract sweep**       | M2          | anom vs level vs all                      | level 当前矩阵不可用；主锁 anom                                        | `sweep_m2.py`                            |
| **LOCHO**                | M3          | 7 臂：core/full/PW-only/GFW-only/…          | **full/tanker/PW-only XGB 均显著**；core 最弱 → 主改 full            | `robustness_m3.py`                       |
| **M3 lookback**          | M3          | L1/L4/L8                                  | XGB 增量**全 L 显著**（0.007/0.0002/0.0018）                        | `sweep_m3.py`                            |
| **LOMO**                 | M4          | 去掉 M2 或 M3 或仅 M1                          | minus-M2(M1+M3) XGB 最优 4.429；**加 M2 反而变差**；M1-only RMSE 全局最优 | `robustness_m4.py`                       |
| **M4 lookback**          | M4          | L1/L4/L8                                  | XGB CW 全 L 显著（0.035/0.009/0.0008）                            | `sweep_m4.py`                            |




#### 深度臂


| 实验                   | 改了什么                                          | 主要发现                                      | 脚本/产物                                  |
| -------------------- | --------------------------------------------- | ----------------------------------------- | -------------------------------------- |
| **Multi-seed**       | seeds {42,1,2} × {finship, m4rep, m4xattn}    | 见 §6.6；**gated 最稳，xattn 最不稳**             | `04_code/scripts/deep/run_deep_sweep.py` → `deep_sweep.png` |
| **Hyperparam**       | lb∈{4,8,12} × d∈{32,64} × gat layers          | lb=8 单 seed 略优但 **lb=4 锁主**；**d=64 一律更差** | `04_code/scripts/deep/run_deep_sweep.py` group=hyper        |
| **RS 分支 grid**       | meanpool/cls × lr/wd/dropout                  | RS 弱是**内在的**；调参救不回来                       | group=rs                               |
| **Fusion 正则 grid**   | finship lr/wd/dropout                         | 整片 skill≈0；**dropout=0.3 略好**             | group=reg                              |
| **Modality dropout** | 0 vs 0.3                                      | xattn+drop0.3：skill +0.62%，CW≈0.008       | `run_deep_advanced.py`                 |
| **融合矩阵**             | fin+ship / fin+rs / M4 × {gated,concat,xattn} | fin+ship 三融合均 skill>0；fin+rs 三融合均<0       | `run_deep_fusion_matrix.py`            |
| **子期 split**         | 早期 ≤2022 vs 晚期 ≥2023                          | gated 晚期转正；xattn 两期均 +0.12%               | `run_deep_advanced.py`                 |
| **min_train=78**     | 更长测试窗（可选）                                     | 附录级；主协议 min_train=104                     | 文档/outline 提及                          |




### 6.6 多 seed 结果（深度，`04_code/scripts/deep/run_deep_sweep.py`，lookback=4，d=32）


| 配置                | skill vs M0（3 seeds） | CW vs M0 显著 seeds | DM vs M1 显著 seeds | 解读                                      |
| ----------------- | -------------------- | ----------------- | ----------------- | --------------------------------------- |
| **finship gated** | **−0.47% ± 0.86**    | 0/3               | **1/3**           | **方差最小、最稳**；主模型依据                       |
| **m4rep gated**   | −0.89% ± 0.60        | 0/3               | 0/3               | 加 RS 无增益，略逊于 finship                    |
| **m4xattn**       | −1.83% ± **2.76**    | 1/3               | 0/3               | seed42 可最佳；seed2 可崩至 −4.98% → **不作主模型** |


**单 seed=42 补充（hyper sweep，finship）**：


| lookback | d   | skill vs M0 | 备注                 |
| -------- | --- | ----------- | ------------------ |
| 4        | 32  | +0.11%      | **主协议锁定**（对齐扁平 L4） |
| 8        | 32  | +0.34%      | 单 seed 更优，但不改主对照   |
| 12       | 32  | 负           | —                  |
| 4/8      | 64  | 更差          | 小样本勿用大维            |


---



## 7. 三个 RQ 的一句话回答


| RQ      | 结论                                                                                |
| ------- | --------------------------------------------------------------------------------- |
| **RQ1** | 扁平层无一击败 M0；航运有显著嵌套信号（CW vs M1），RS 弱除非稀疏化/去噪；深度 finship/xattn 可 skill>0 但仍难显著击败 M0 |
| **RQ2** | 配对比较：深度在 M3/M4（含航运）显著优于扁平；门控 > concat；xattn 单 seed 最佳但不稳                          |
| **RQ3** | 门控、SHAP、节点/站点注意力互补；跨 seed 稳定焦点为 Hormuz（及 RS Ningbo/Rotterdam）；α_shipping 周度不稳故事件叙事须克制（关联≠因果） |


---



## 8. 项目结构速查

```text
casa0004 Dissertation/
├── 00_admin/最新待整理/   # 进度总览、变量清单、本文件、walkthrough
├── 01_literature/         # 矩阵 + 笔记 + 往年样例
├── 03_data/               # raw + processed（M1/M2/M3/merge）
├── 04_code/               # flat / deep 脚本 + encoders
├── 05_outputs/baselines/  # Flat/ + Deep/ + subperiod/
├── 06_writing/            # Outline/ · Chapter 2/ · 各章双语 · Appendix/
└── 07_submission/         # 交稿占位
```


| 复现入口 | 命令/路径 |
| --- | --- |
| 扁平 M0–M4 | `04_code/scripts/flat/run_baseline.py` |
| 深度融合 | `04_code/scripts/deep/run_deep_baseline.py` |
| 深度稳健性 | `04_code/scripts/deep/run_deep_sweep.py` |
| 深度可解释性 / 多 seed | `04_code/scripts/deep/run_deep_interpret.py --seeds 42,1,2` |
| 特征矩阵 | `03_data/processed/merge/outputs/weekly_feature_matrix.csv` |
| 17 节点图 | `03_data/processed/M3/outputs/m3_graph17_tensors.npz` |
| 当前 outline | `06_writing/Outline/20260728_outline_brief.md` |
| Ch2 最新稿 | `06_writing/Chapter 2  Literature Review/20260728_literature_review_双语.md` |


---



## 9. 诚实的总括

1. **M0 随机游走是强基准**——绝对 RMSE 增益很小，经济意义有限；但嵌套检验与架构对照仍有方法学价值。
2. **航运 > 遥感**——扁平与深度一致；航运信号与咽喉流量/地缘扰动机制吻合。
3. **扁平融合的局限**正是 RQ2 的动机——高维 anom / 异构航运列拼进宽表，增量被稀释或噪声放大。
4. **表示级融合**在含航运的多模态设定下更有效地 harvest 信号，但复杂度与 seed 敏感性（尤其 xattn）是真实代价。
5. **下一步（07-28）**：按 `20260728_outline_brief.md` 合稿润色；Meeting 05（07-29）前可读稿；RQ3 正文只写跨 seed 稳定焦点；深度 LOAO/leave-one-node 为可选、不阻塞交稿。

---

