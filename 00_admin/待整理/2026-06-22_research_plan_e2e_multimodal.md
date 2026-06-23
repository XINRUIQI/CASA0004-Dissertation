# 研究方案：端到端多模态原油价格预测框架

> **拟定标题 / Working title**
> A Modality-Aware Spatio-Temporal Fusion Framework for Brent Crude Oil Forecasting Using Financial Time Series, Satellite Imagery and Maritime Networks
> **融合金融时序、卫星影像与海运网络的模态感知时空布伦特原油价格预测框架**

---

## 0. 一句话定位

> 不是把遥感和航运加工成更多数值列再拼成一张表，而是用**模态专属编码器**分别学习「金融状态表示、遥感活动表示、航运网络表示」，再通过**门控交叉注意力**学习三者在不同市场时期的动态贡献，端到端预测**下一周 Brent 价格**（训练时以对数价格变化为内部目标，再还原为价格）。

本方案把项目分成两层：

- **核心实证层（baseline / ablation）**：现有工程特征 + XGBoost / 线性 / TCN，做稳健、可复现的 M0–M4 消融。
- **方法创新层（contribution）**：真正的模态感知端到端多模态网络，回答「保留模态结构的表示级融合是否优于把所有数据压成一张表」。

---

## 1. 研究背景与缺口

油价预测文献长期以金融与基本面数值变量为主，近年开始引入卫星遥感（作为石油活动代理）和航运/AIS 数据（作为供给与贸易流代理）。但绝大多数工作仍停留在**多源异构特征级融合（multi-source heterogeneous feature fusion）**：影像被压成 NDVI/NDWI/NTL，航运被压成船舶数/通过量，最终所有变量进入同一张特征表交给一个数值时序模型。模型实际看到的仍是一个普通的多变量数值序列。

**研究缺口**：在原油市场预测中，是否将不同模态保留各自结构、由专属编码器在表示空间融合，能在金融基线之上带来增量预测能力？模态级表示融合是否优于扁平特征融合？

---

## 2. 研究问题（Research Questions）

- **RQ1（增量价值）**：在金融/宏观基线（M1）之上，加入遥感模态、航运模态是否提升样本外预测？是否优于不变基准 M0？
- **RQ2（方法创新）**：在相同数据下，**模态感知表示级融合**（模态专属编码器 + 门控交叉注意力）是否优于**扁平特征融合**（所有数值列拼一张表喂 XGBoost / 早融合 LSTM）？
- **RQ3（可解释性）**：融合门控权重能否揭示「不同市场时期模型更依赖哪种模态、关注哪个港口/航道」，并与已知供给冲击、地缘事件、库存变化在时间上吻合？

---

## 3. 预测目标定义（唯一核心目标：价格）

**唯一核心预测目标**：预测下一周周五（或该周最后一个可交易日）的 Brent 现货价格  P_{t+1} ，单位美元/桶。

研究目标始终只有「价格」一个。训练时不直接让网络输出完整价格水平，而是预测**一周对数价格变化**——这只是模型内部更适合学习的数学表达，并非第二个预测目标。

- **模型训练目标（内部表达）**： r_{t+1} = \log(P_{t+1}/P_t) 。
- **模型最终输出（还原价格）**： \hat P_{t+1} = P_t\, e^{\hat r_{t+1}} 。
- **辅助评估指标（由预测价格派生，非独立预测任务）**：方向  \mathrm{sign}(\hat P_{t+1}-P_t) 、收益率，仅用于评估与可解释性。
- **不做**：独立波动率预测（不设波动率预测头、不设波动率损失项）。


| 层面     | 最终定义                                              |
| ------ | ------------------------------------------------- |
| 研究目标   | 准确预测下一周 Brent 价格  P_{t+1}                          |
| 模型训练目标 | 下一周对数价格变化  r_{t+1}=\log(P_{t+1}/P_t)              |
| 模型最终输出 | 还原后的下一周 Brent 价格  \hat P_{t+1}=P_t e^{\hat r_{t+1}} |
| 辅助指标   | 由预测价格计算的方向、收益率（仅评估）                              |
| 不做     | 独立波动率预测                                           |


> 这不是同时设置两个预测目标。研究目标仍然只有价格；对数变化只是模型内部更适合学习的数学表达。因此训练为**单任务回归**（仅  r_{t+1} ），方向/收益率在还原出价格后派生，用于诊断与解释，不参与训练，避免「价格预测涨、方向预测跌」式自相矛盾。

---

## 4. 数据与模态

预测起点统一为每周五收盘，预测目标为下一周。统一比较窗 **2019.1–2025.12**（与导师确认的标准化窗口；M1/M2/M3 数据均覆盖至 2025-12，窗内每周模态齐备且 target 完整）；最佳模型可在更长历史（全量 union 2006–2026，见 merge 脚本 `--full`）做第二阶段稳健性测试。

### 4.1 金融模态（Financial Temporal Modality）

- 内容：Brent、WTI、EIA Weekly 库存/产量/进出口/炼厂利用率、VIX、DXY、利率、S&P 500、期货价差、地缘政治风险指数等（约 20–27 变量）。
- 输入形态：过去 L 周窗口序列，shape ≈ `[batch, L=4, ~20 vars]`（4 周对齐导师初始设定，后做 1/8/12 周稳健性）。

### 4.2 遥感模态（Earth Observation Modality）— 正在重新导出

- **新数据**：11 AOI × 月度 Sentinel-2 patch（6 波段 B2/B3/B4/B8A/B11/B12 + SCL 云掩膜），按站点类型差异化 patch（port 6.4km / refinery 5.12km / terminal 2.56km），2019–2026。导出脚本：`03_data/raw/02_sentinel2/export_s2_patches_multimodal_gee.js`，附 `manifest.csv`（每 site×月有效影像数、云量）。
- **双通道设计**（表示学习 + 经济解释并存）：
  - 通道 A — 影像表示：冻结预训练 EO 大模型（Prithvi-EO-2.0 / SatMAE）提取每张 patch 的 image embedding。
  - 通道 B — 机制变量：保留现有 NDVI/NDWI/NDBI/BSI/NTL/FRT 等人工指标。
- 时间频率：月频、受云影响、不规则；VIIRS 月频。

### 4.3 航运模态（Maritime Network Modality）

- 当前已有：6 咽喉航道 PortWatch 日频 + GFW 月度 vessel presence + EMODnet 月度密度。
- 完整版需补采（阶段二）：11 港口节点的 port-level 活动、油轮船型/运力/dwell time/锚泊数/dark vessel、节点间 O-D 流（GFW voyages/port-visits，必要时商业 AIS）。
- 图结构：17 节点（11 港口/码头/炼厂 + 6 chokepoint）的异质动态图 G(t)。
- **重要约束**：金融变量**不复制到图节点**当节点特征；金融单独走全局分支，最后融合。

### 4.4 异步对齐与缺失（关键，决定可信度）

- **按发布时间戳对齐，而非统计周末日**：EIA WPSR 统计周「截至周五」，但通常**次周三上午发布**——周五预测时不可使用刚结束这一周的库存值，应按 release timestamp 对齐。卫星按影像真实可得日期；PortWatch/GFW 按数据发布延迟。
- **不再把月度值 ffill 成多个相同周值**。每条遥感观测记录：`image_embedding, observation_date, days_since_observation(age), cloud_fraction, sensor_type, valid_mask`。
- 缺失模态显式建模：modality mask + time-gap embedding + 训练期 modality dropout + missing-modality 训练，而非用普通填补抹平。

---

## 5. 模型架构：Modality-Aware Spatio-Temporal Fusion Network

```text
金融序列 [Brent,WTI,inventory,VIX,DXY...]
        └─► Finance Temporal Encoder (TCN/GRU) ─────────► z_fin (32维)

卫星影像序列 [11 AOI × 月度多光谱 patch]
        └─► 冻结 EO Encoder (Prithvi/SatMAE) → image embeddings
              └─► 时间注意力(同AOI多月) + AOI-site 注意力 ──► z_rs (32维)
                  (+ 双通道并入 NDVI/NTL/FRT 机制变量)

航运动态图 G(t) [17节点 + 边特征]
        └─► GAT/GCN(空间) → TCN/GRU(L周时间) ───────────► z_ship (32维)

      z_fin, z_rs, z_ship
        └─► Gated Cross-Modal Fusion ──► z_fused (64维)
              └── log-return head (回归, 唯一) ─► r̂_{t+1}
                    └─► 还原价格 P̂_{t+1}=P_t·e^(r̂)   ← 模型最终输出
                          └─(派生) 方向 / 收益率：仅作辅助评估指标，不参与训练
```

### 5.1 三个模态编码器

- **Finance**：2–3 层小型 TCN（或 GRU），输出 32 维。最稳妥。
- **EO**：冻结 Prithvi-EO-2.0 / SatMAE（数亿参数，**只做特征提取不微调**）→ 预计算 image embeddings → 仅训练轻量 temporal attention（同 AOI 多月聚合）+ site attention（11 AOI 加权池化，学习哪个站点更重要），输出 32 维。
- **Shipping**：1–2 层 GAT + 小型 TCN，输出 32 维。

### 5.2 融合（三选一，递进对照）

- **方案一 Encoder-Concat**：`z = concat(z_fin, z_rs, z_ship)` → MLP。基础多模态对照。
- **方案二 Gated Fusion（推荐主模型）**：
 \alpha_t = \mathrm{softmax}(\mathrm{MLP}[z_{fin}, z_{rs}, z_{ship}, mask_t]) ，
 z_{fused} = \alpha_{fin} z_{fin} + \alpha_{rs} z_{rs} + \alpha_{ship} z_{ship} 。
门控权重作为 RQ3 的模态级解释。
- **方案三 Cross-Attention（进阶，金融做 Query）**：
 c_{rs} = \mathrm{CrossAttn}(z_{fin}, H_{rs}, H_{rs}) ， c_{ship} = \mathrm{CrossAttn}(z_{fin}, H_{ship}, H_{ship}) ，
 z_{fused} = z_{fin} + \gamma_{rs} c_{rs} + \gamma_{ship} c_{ship} 。
保留每个 AOI/影像、每个节点/航道的 token，可分析「当前金融状态关注了哪个港口/航道」。

### 5.3 损失函数（单任务回归）

唯一训练损失为对数价格变化的回归损失（Huber / MSE）： L = L_{return}(\hat r_{t+1}, r_{t+1}) 。
不设方向头、不设波动率头；方向与收益率在还原出  \hat P_{t+1}  后派生，仅用于评估与可解释性，不进入损失。

---

## 6. 实验设计

### 6.1 两个维度交叉

**模态维度（数据）× 架构维度（融合方法）**，回答 RQ1 与 RQ2。


| 模态配置 | 使用编码器                        |
| ---- | ---------------------------- |
| M0   | 不变基准 \hat p_{t+1}=p_t （随机游走） |
| M1   | Finance                      |
| M2   | Finance + RS                 |
| M3   | Finance + Shipping           |
| M4   | Finance + RS + Shipping      |



| 架构               | 融合方法                | 定位                    |
| ---------------- | ------------------- | --------------------- |
| XGBoost          | 原始工程特征拼接            | 多源表格基线                |
| LSTM/TFT-Early   | 所有数值变量进同一时序模型       | Early-fusion baseline |
| Encoder-Concat   | 三编码器 + embedding 拼接 | 基础多模态                 |
| **Gated Fusion** | 三编码器 + 动态门控         | **推荐主模型**             |
| Cross-Attention  | 模态 token 交叉注意力      | 进阶主模型                 |


核心对照：**Flat feature fusion vs Modality-aware representation fusion**。

### 6.2 公平比较协议（导师 Priority 1–3）

- 统一窗口 **2019.1–2025.12**、统一 4 周滞后、相同时间顺序 train/val/test 划分、相同目标定义（价格，经对数变化训练）与评估指标。
- 必含 **M0（随机游走  \hat P_{t+1}=P_t ）** 作为主基准；派生方向指标另比 directional persistence 等朴素基准。
- **rolling-origin backtesting**；DM 检验比较预测精度差异；bootstrap 置信区间。
- 评估指标：在**还原价格  \hat P_{t+1} ** 上算回归 RMSE/MAE/R²、相对随机游走的提升；由价格派生方向 accuracy/F1（辅助）；不单列波动率指标；并报告是否显著优于 M0。

#### 6.2.1 扁平对照基线（Flat baseline，已锁定 2026-06-23）

作为 M0–M4 的统一**扁平特征融合**对照（后续加模态 M2/M3/M4 与方法创新层都以此为标尺）：

- **特征装配**：各模态在自身聚合脚本中完成发布滞后后的周频表（M1 含 EIA +1w、月频 1–5w；M2 +15d as-of；M3 GFW +4w / PW +1w）合并，每个特征 lag 0..3 展平为一行；窗口 2019.1–2025.12。
- **协议**：rolling-origin（expanding，min_train=104 周，tune 时 retrain_every=13），lookback=4 周（导师设定），目标单任务回归 r_{t+1}，还原 \hat P_{t+1}=P_t e^{\hat r_{t+1}}。
- **模型**：M0（随机游走）+ M1_Ridge + M1_XGB，开启**内层时间验证调参**（`--tune`：Ridge alpha + XGB 小网格，验证集 = 训练折尾部 52 周）；feature_mode=all 为主，returns（趋势列平稳化）作稳健性附加。
- **主对照锁定 = `L4_tuned`**（lookback=4 + tune，与 4 周设定一致且为当前最强扁平基线）；另保留 **`L1`**（lookback=1，38 维、~30s）作轻量 sanity。
- **当前 M0/M1 结果**（260 测试周 2021–2025，M0 RMSE=4.14）：L4_tuned Ridge=4.38（skill −5.9%）、L12 XGB=4.70（−13.6%）；扁平 M1 仍未超 M0，但调参后差距由强显著（DM≈5.0）收敛到边缘（DM≈1.7）。代码 `04_code/scripts/run_baseline_m0_m1.py` + `sweep_baseline.py`；详见 `00_admin/2026-06-23_flat_baseline_log.md`。

> ✅ 数据接线（已修复 2026-06-23）：EIA +1w 滞后下沉到 **M1 源头**（`build_m1_weekly.py`）；merge 层 `build_feature_matrix.py` 已设 `EIA_WPSR_LAG_WEEKS=0`「仅复查、不再 shift」，自检改为「EIA == M1 原列 unchanged」。重跑标准窗 365×320 / full 1067×320，无泄漏自检全 OK。三模态统一为「各自滞后、merge 仅复查」。

### 6.3 稳健性

不同滞后（1/8/12 周）、不同时间切分、OVX↔VIX 替换、共同样本 vs 最大样本、不同缺失处理、leave-one-AOI-out、特征选择稳定性。

### 6.4 防泄漏自检清单

- 每个变量按真实发布时间戳可见；月频变量加保守发布滞后；杜绝月末值回填月初；周五截止统一；target 严格在下一周；滞后/移动平均只用历史；特征选择/标准化仅在训练集内 fit。

---

## 7. 实现路线（分阶段，先做导师要的，再做创新层）

### 阶段 0 — 无泄漏特征矩阵 + 公平 M0–M4（最高优先，进下次会议）

1. 按发布时间戳重对齐所有变量（EIA 周三、遥感真实可得日、航运发布延迟）。
2. 遥感加 `days_since_obs / valid_mask / modality_mask`，停止相同周值 ffill。
3. 统一目标为价格（训练用对数价格变化，输出还原为价格）；方向/收益率派生为辅助指标；不做波动率预测。
4. 2019–2026、4 周滞后、相同切分下跑 M0–M4，产出对比表。

### 阶段 1 — 模态感知融合（方法创新核心，成本可控）

1. 预计算遥感 image embeddings（冻结 Prithvi/SatMAE，用阶段已导出的 patch）。
2. 实现三个轻量编码器 + Encoder-Concat + Gated Fusion，对比 flat vs modality-aware。
3. rolling-origin CV + modality dropout + 强正则；始终对比 M0。

### 阶段 2 — 完整端到端（可选/进阶，时间允许再做）

1. Cross-Attention 融合；保留 AOI/节点 token 做注意力可视化。
2. 航运升级为船舶级动态图（补采 PortWatch port-level + GFW voyages）。
3. 在更长历史窗口做第二阶段测试。

---

## 8. 可行性与风险


| 风险           | 说明                                | 缓解                                                               |
| ------------ | --------------------------------- | ---------------------------------------------------------------- |
| **样本量小**     | 2019–2026 仅约 360 周，滑窗高度自相关，独立信息更少 | 冻结大模型只提特征不训练；编码器维度小（32）；强正则 + dropout；rolling-origin CV；始终与 M0 比 |
| **难以战胜随机游走** | 周度油价 M0 极强                        | 以「相对 M0 的提升 + DM 检验显著性」为主指标，而非绝对精度                               |
| **遥感月频 + 云** | 真实观测稀疏                            | age/valid_mask 显式建模；缺失模态训练；双通道保留机制变量                             |
| **航运图数据缺口**  | 当前无 O-D/船型/dwell                  | 阶段 1 先用现有聚合特征做节点/咽喉编码；船舶级图列为阶段 2                                 |
| **算力/工程量**   | EO patch + FM 推理 + 图              | patch 已差异化导出（~2GB）；FM 用 CPU/Colab GPU 预计算一次性 embedding           |
| **偏离导师优先级**  | 导师要求先做公平比较 + 写文献综述                | 阶段 0 先交付；阶段 1 架构升级先用一页设计与 Beatrice 确认再投入                         |


---

## 9. 交付物

- 无泄漏统一特征矩阵 + 数据字典（含发布时间戳）。
- M0–M4 × 架构维度的公平比较表（含 DM 检验、置信区间）。
- 模态感知融合模型（Gated / Cross-Attention）及门控权重可视化。
- 稳健性结果（滞后、切分、leave-one-AOI-out 等）。
- 可复现代码包（`04_code/src/` 模块化）。

---

## 10. 待与导师确认的决策点

1. 实验窗已锁定 **2019.1–2025.12**（数据止于 2025-12，窗内 target 完整）；最佳模型在全量 union 2006–2026（merge `--full`）做更长历史第二阶段稳健性。

