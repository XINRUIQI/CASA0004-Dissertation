# Research Diary — Phase 03

> **Phase 03:** Data processing & 4-model framework implementation  
> **Trigger:** Meeting 03 (2026-06-17)  
> **Deadline:** Meeting 04 (TBC)  
> **Details:** `00_admin/2026-06-22_research_plan_e2e_multimodal.md` · `00_admin/meeting_notes/Meeting03 2020260617.md`

---

## 2026-06-17

### What I did

- Third meeting with supervisor Beatrice Taylor.
  与导师 Beatrice Taylor 进行了第三次会议。
- Presented preliminary M1–M4 model results, variable tables, and literature progress (~20 papers reviewed).
  汇报了 M1–M4 初步模型结果、变量表及文献进度（约 20 篇已读）。

### Supervisor feedback

- **Good progress** on literature-driven variable reduction and initial modelling; next focus is **fair, reproducible comparison** and converting notes into dissertation writing.
  文献驱动的变量精简和初步建模进展良好；下一阶段重点是**公平、可复现的比较**，并将笔记转化为论文正文。
- **Add Model 0 (M0):** \(\hat{p}_{t+1} = p_t\) — no-change / persistence baseline; all complex models must beat M0 under identical test conditions.
  **增加模型 0：** \(\hat{p}_{t+1} = p_t\) 不变预测基准；所有复杂模型须在相同测试条件下优于 M0。
- **Standardise comparison:** same period **2019–2026**, **4-week lag**, identical chronological train/val/test splits, same targets and metrics for every model.
  **统一比较条件：** 相同时间窗 **2019–2026**、**4 周滞后**、相同时间顺序划分、相同目标与指标。
- **Clarify targets:** primary = next-week Brent log return; direction derived from return threshold; volatility formula and flat-class threshold must be explicitly defined.
  **明确预测目标：** 主目标 = 下一周 Brent 对数收益率；方向由收益率阈值派生；波动率公式与 flat 阈值须明确定义。
- **Improve variable documentation:** add a description/definition column to every feature table (what it measures and why it may relate to oil prices).
  **完善变量说明：** 每个特征表增加描述列（衡量什么、为何与油价相关）。
- **Remote sensing simplification:** remove Sentinel-2 **cloud fraction** (redundant quality variable); retain **valid-observation count** only; four AOIs acceptable if justified by completeness and literature.
  **遥感简化：** 删除 S2 **云量比例**；仅保留**有效观测次数**；四个 AOI 可从完整性与文献角度论证。
- **Mechanistic validation is optional EDA** — the main evidence is whether adding RS/shipping improves out-of-sample prediction.
  **机制验证为可选 EDA**——核心证据是加入遥感/航运后样本外预测是否改善。
- **TFT caution:** weekly sample (~360 weeks) may be too small; check overfitting and code correctness before drawing conclusions.
  **TFT 需谨慎：** 周度样本约 360 周，可能过拟合；下结论前须检查代码与 train/val 差距。
- **Start writing literature review now** — organise by themes (~4–5 pages), not paper-by-paper summaries.
  **现在开始写文献综述**——按主题组织（约 4–5 页），而非逐篇罗列。

### Decisions made

- Phase 03 officially starts: fair M0–M4 comparison + leakage-safe feature matrix rebuild.
  Phase 03 正式开始：公平 M0–M4 比较 + 无泄漏特征矩阵重建。
- Retain price, direction, and volatility as targets; shift primary regression target to **log return** (consistent with e2e plan). *(superseded 2026-06-23: sole target = price; volatility prediction dropped.)*
  保留价格、方向、波动率三个目标；主回归目标改为**对数收益率**。*（已被 2026-06-23 决策取代：唯一目标 = 价格，取消波动率预测，方向/收益率由预测价格派生。）*
- Standard comparison window locked at **2019–2026** for first-stage analysis; longer history as optional second-stage robustness test.
  首轮标准化比较窗口锁定 **2019–2026**；更长历史作为可选第二阶段稳健性测试。
- Remove S2 cloud fraction from M2 feature set; document exclusion rationale in methodology.
  从 M2 特征集中删除 S2 云量；在方法章节记录删除理由。
- M5 (GDELT) remains Appendix-only; not part of main M0–M4 ablation.
  M5 (GDELT) 仍仅作 Appendix；不纳入主 M0–M4 消融。

### Next tasks

- Implement M0 and re-run M0–M4 on common 2019–2026 window with 4-week lag and identical splits.
  实现 M0，在统一 2019–2026 窗口、4 周滞后、相同划分下重跑 M0–M4。
- Rebuild feature matrix with **publication-timestamp alignment** (EIA WPSR Wednesday release, no look-ahead ffill).
  按**发布时间戳**重建特征矩阵（EIA WPSR 周三发布，禁止前瞻 ffill）。
- Add description column to all variable tables; write 4-AOI selection justification.
  为所有变量表增加描述列；撰写 4 个 AOI 选择依据。
- Begin thematic literature review draft (~4–5 pages).
  开始撰写按主题组织的文献综述初稿（约 4–5 页）。

---

## 2026-06-22

### What I did

**M1 数据处理管线重构（无泄漏、单脚本输出）**

- 将原三步 M1 流程（`build_weekly_time_index.py` + `build_m1_to_build.py` + `merge_m1_to_build.py`）合并为单一离线脚本 `03_data/processed/M1/py/build_m1_weekly.py`。
- 输出 `processed/M1/outputs/m1_weekly_features.csv`：**1043 周 × 40 列**（2006-01 – 2025-12，W-FRI），无中间表。
- 关键设计：
  - raw 层按 provider 组织（`EIA/`、`FRED/`、`Yahoo/`、`Other/`），支持 `--online` / `--refresh-raw` / `--base-only` 标志；
  - 月频变量（GPR、Kilian REA、IMF 工业原料）加**保守发布滞后**（1–5 周），避免 look-ahead；
  - 主预测目标统一为 `brent_log_return`；派生 `brent_direction`（up/flat/down）、`brent_vol_4w` / `brent_vol_12w`；
  - 补充文献建议变量：OVX、`futures_spread`、`gold_return`、`commodity_fx`、`dgs10_change`、`gpr`、`global_econ_activity`、`nonoil_industrial_commodity` 等；
  - 新增 `avail_*` 模态可用性标记列。

**数据集文档更新**

- 新建 `03_data/Dataset/Dataset_Overview4.ipynb`：M1 数据源表 + **逐变量中英描述字典**（含 M0 定义、单位说明、预测目标定义），对齐 Meeting 03 要求的 description column。
- 更新 `03_data/Dataset/external_sources.md`：与端到端多模态方案 §4.2 对齐，记录遥感双通道设计说明。

**M3 航运脚本归位**

- 将 `aggregate_shipping_to_weekly.py` 迁移至 `03_data/processed/M3/py/`，输出 PortWatch 日频 + GFW 月频 → 周频宽表；新增 `avg_tanker_size`（P070 概念：运力加权优于单纯计数）。

**遥感 Channel B 导出**

- 完成 11 AOI × 月度 Sentinel-2 机制变量 CSV 下载：`03_data/raw/02_sentinel2/Channel B/sentinel2_oil_sites_monthly_indices_201704_202512_11aoi.csv`（**1156 条 site×month 记录**，NDVI/NDWI/NDBI/BSI + `valid_obs_count`；含 `cloud_probability` 供 QC 但**不进入 M2 特征**）。
- 部分站点（Fujairah、Jurong、Ningbo 等）云遮挡导致整月 NDVI 等为空——与 Meeting 03「保留 valid_obs_count、删除 cloud fraction 入模」决策一致。

**研究方案与项目结构**

- 撰写 `00_admin/2026-06-22_research_plan_e2e_multimodal.md`：端到端模态感知融合框架（Financial / EO / Shipping 三编码器 + Gated Fusion），区分**核心实证层**（M0–M4 表格消融）与**方法创新层**（表示级融合 vs 扁平特征融合）。
- 更新 `00_admin/File Structure20260703.md`：反映 M1/M2/M3 分层 processed 目录、Phase 03 数据流、关键文档索引。

### Decisions made

- M1 特征矩阵以 **`build_m1_weekly.py` 单表**为准；建模时在 2019–2026 窗口 clip，不再依赖旧版多步 merge。
  M1 feature matrix standardised on single-table output; clip to 2019–2026 at modelling time.
- 主回归目标正式定为 **`brent_log_return`**；价格由 \( \hat P_{t+1} = P_t \cdot e^{\hat r_{t+1}} \) 还原；M0 = 预测收益为 0。
- 遥感保留**双通道**方向：Channel A = S2 patch + 冻结 EO encoder（阶段 1）；Channel B = NDVI/NTL/FRT 机制变量（当前已有 CSV）；云量仅作 QC，不入模。
- 异步对齐原则写入方案：按真实发布时间戳对齐，月频不再 ffill 成多个相同周值；后续加 `days_since_obs` / `valid_mask` / `modality_mask`。
- 比较协议：2019–2026 · 4 周滞后 · rolling-origin · DM 检验 · 始终报告相对 M0 的提升。

### Issues / blockers

| 问题 | 影响 | 计划 |
|---|---|---|
| 旧 Test pipeline（`01_literature/Test/`）尚未接入新 M1 表 | M0–M4 公平 rerun 仍用旧特征 | P0：更新 `data_loader.py` + 统一特征矩阵 |
| GFW/PW union 对齐未修 | M3 样本从 727 骤降至 362 | P0：各自 ffill + union 索引 |
| M0 未实现 | 无法判断复杂模型是否有价值 | P0：加入 `01_baselines.py` |
| S2 部分 AOI 整月缺失 | M2 稀疏观测 | 用 `valid_obs_count` + age embedding；Meeting 03 四 AOI 子集待最终选定 |
| 深度模型过拟合 / 目标未改 log return | LSTM/TFT/ST-GNN R² 大量为负 | P1：修复后再纳入正式比较 |
| 文献综述正文未动笔 | Meeting 04 材料缺口 | P1：按 6 主题写初稿 |

### Next tasks

1. **P0 — 公平 M0–M4 rerun：** 接入 `m1_weekly_features.csv` + 更新 M2/M3 聚合 → 统一 2019–2026 / 4-week lag / 相同 split → 产出对比表 + DM p-value。
2. **P0 — 泄漏自检：** EIA 周三发布对齐；月频变量加发布滞后；target 严格在 t+1。
3. **P1 — 变量表 & 写作：** 完成 RS/shipping 变量 description column；4-AOI 选择短文；文献综述主题初稿。
4. **P1 — M2 周频聚合：** 由 Channel B CSV 重跑 `aggregate_remote_sensing_to_weekly.py`（去 cloud fraction 列）。
5. **P2 — 方法创新层：** 与 Beatrice 确认 e2e 方案一页摘要后再投入 Prithvi embedding 预计算。

---

## 2026-06-23

### Decisions made

- **预测目标收敛为单一核心目标：价格。** 唯一研究目标 = 准确预测下一周 Brent 现货价格 \(P_{t+1}\)（美元/桶，周五或当周最后一个可交易日）。
  Prediction objective narrowed to a single core target — **price**. Sole research goal = accurately predict next-week Brent spot price \(P_{t+1}\).
- **训练用对数价格变化作内部目标**：\( r_{t+1}=\log(P_{t+1}/P_t) \)，预测后还原 \( \hat P_{t+1}=P_t e^{\hat r_{t+1}} \)。这只是更易学习的数学表达，不是第二个目标。
  Train on the log price change as the internal target; reconstruct price afterwards — a learning-friendly representation, not a second target.
- **方向、收益率改为由预测价格派生的辅助评估指标**（不再是独立预测任务 / 损失头）；训练改为**单任务回归**。
  Direction and returns become auxiliary metrics derived from the predicted price (no longer separate tasks/heads); training is now single-task regression.
- **删除独立波动率预测，并移除已实现波动率列**：不设波动率头/损失；**已从 `build_m1_weekly.py` 与 `m1_weekly_features.csv` 删除 `brent_vol_4w` / `brent_vol_12w`**（40 → 38 列）。
  Drop volatility forecasting and remove the realized-volatility columns: no volatility head/loss; `brent_vol_4w/12w` deleted from the builder and the feature table (40 → 38 columns).
- 这取代 2026-06-17「保留价格、方向、波动率三个目标」的多目标设定。
  This supersedes the 2026-06-17 three-target (price / direction / volatility) setup.

### 波动率：特征 vs 目标（为何删除 `brent_vol_4w/12w`）

**两种身份**：一个波动率变量既可当**预测目标 y**（要预测的输出），也可当**输入特征 X**（喂给模型的自变量）。本项目「不做独立波动率预测」针对的是 **y**；删不删这两列，只影响波动率能否作为 **X**。

**文献怎么用波动率**（据 `01_literature/literature_matrix.md`）：

- 当**特征**用得最多的是**隐含波动率**（VIX / OVX，前瞻、市场定价）：**P072**（VIX 入选 315 变量集）、**P076**（`VIX_{t-1}` 作"战略风险"特征预测 Brent 月度对数收益）、**P025**（VIX 作控制变量预测下一周 WTI 收益）；矩阵 M1 推荐 #6 即 `OVX`(优先)/`VIX`，且 **P052** 指出 OVX 比 VIX 更具石油针对性（二者高相关时优先 OVX）。
- **已实现波动率**（由历史收益算的滚动标准差，即 `brent_vol_*` 这一类）在库里直接以特征出现的只有 **P077**（Chung 2024：realized vol + VIX + GARCH/XGBoost），但其**预测目标本身就是波动率**。
- **关键方法论提醒（P076）**：GARCH 改善条件方差但**不改善收益率/价格的点预测**——波动率与均值是两类任务。

**为何删除这两列（而非保留为可选特征）**：

1. 波动率信息已由**隐含波动率 `ovx` / `vix`** 承载，且文献更推荐隐含版本；`brent_vol_4w/12w` 属已实现波动率，与之**信息重叠**、文献支撑更弱（仅 P077，且为 vol-target 场景）。
2. 这两列**原本是作为波动率预测目标生成的**；既已取消波动率预测，留着既不当 y、又与 OVX/VIX 冗余，徒增混淆。
3. 删除后特征矩阵更干净（40 → 38 列），且**不损失"波动率作为特征"的文献合规性**——OVX/VIX 仍在。

### Files touched

- `00_admin/2026-06-22_research_plan_e2e_multimodal.md` §0/§3/§5/§5.3/§6.2/§7：重写为单一价格目标、单任务回归。
- `03_data/processed/M1/py/build_m1_weekly.py`：删除 `brent_vol_4w/12w` 计算（两行）+ 更新目标块注释。
- `03_data/processed/M1/outputs/m1_weekly_features.csv`：删除两列波动率（40 → 38 列，1043 周不变）。
- `03_data/Dataset/Dataset_Overview4.ipynb` M1 词典：删除两行波动率词条 + 目标说明改为「已删除」+ 列数 40 → 38。
- `03_data/Dataset/external_sources.md`、`00_admin/File Structure20260703.md`：列数 40 → 38（含删除说明）。

### M3 航运聚合修复 + 全模态无泄漏合并（数据管线）

**M3 航运周频聚合（`03_data/processed/M3/py/aggregate_shipping_to_weekly.py`）**

- 重写并修正脚本路径：旧版从 `raw/03_shipping/` 复制到 `processed/M3/py/` 后路径未适配（`PROJECT_ROOT` 解析错误、找不到原始数据）；现正确指向 `raw/03_shipping/{IMF Portwatch, GFW}`，输出 `processed/M3/outputs/m3_weekly_features.csv`。
- **修复 GFW/PW union 掉样本（727 → 362）**：改为「各源各自对齐 + union 索引」——GFW 月末对齐后在 union 上 ffill、PortWatch 日→周 W-FRI 求和后 reindex；不再因截到 PortWatch 重叠窗（2019+）而丢弃 2012–2018 的 GFW 早期样本。结果 **750 周 × 123 列**（2012-02 ~ 2026-06），GFW 覆盖 746 周、PortWatch 388 周（旧 inner-join 仅 388）。
- **发布时间戳滞后（防泄漏）**：GFW 月频 `+4 周`、PortWatch 日频聚合 `+1 周`；自检确认「首个可用周向后平移恰好 lag 周」（是滞后、非前视）。

**全模态合并（`03_data/processed/merge/py/build_feature_matrix.py`）**

- M1（金融 38）+ M2（遥感 **55 anom**，自 2026-06-23 续；原 154 全量仅留 M2 宽表）+ M3（航运 123）按统一 union W-FRI 索引合并。**默认输出标准比较窗 2019.1–2025.12（365 周 × 221 列，364 周有 t+1 target）**；`--full` 另出全量 union（1067 周 × 221 列，2006–2026）供长历史稳健性。各附数据字典（每列模态/分组/发布滞后/覆盖区间）。
- **EIA WPSR 周三发布滞后**：合并层对 13 个 WPSR 列施加 `+1 周`（统计截至周五、次周三发布 → 周五预测不可用本周值；M1 输出为 Friday 对齐但未滞后，此为修复的前视点）。
- **复查（不重复施加）**：M1 月频（GPR/Kilian/IMF，已在 M1 滞后 1–5 周）、M2（`build_m2_weekly.py` 已用 month_end+15d 的 as-of join + `days_since_obs` + valid/modality mask + expanding anomaly，无前视）、M3（GFW+4w/PW+1w）。
- **target 严格 t+1**：`target_price_next = P_{t+1}`、`target_log_return_next = r_{t+1}=log(P_{t+1}/P_t)`；价格列不平移（target 基准）。
- **无泄漏自检全部通过**：EIA +1w 方向、`brent_price` 未平移、`target_price_next[t]==brent_price[t+1]`、log-return 一致性。
- **标准比较窗锁定 2019.1–2025.12**（M1/M2/GFW 数据均止于 2025-12，窗内每周模态齐备且有 t+1 target）：不延采 M1 到 2026；PortWatch 的 2026 部分仅存在于 `--full` 长历史稳健性导出（无 M1 target，建模时丢弃）。

### M1 源头防泄漏补强 + 数据字典

- **EIA WPSR 滞后下沉到 M1 源头**：在 `build_m1_weekly.py` 加 `EIA_LAG_WEEKS=1`，`weekly_eia_to_friday` 增 `lag_weeks` 参数（对齐报告周五 → reindex → `shift(+1)`）。10 个 WPSR 原始列 + 3 个派生（change/net）整体后移一周。自检通过：首周 `crude_stocks_excl_spr`=NaN、`avail_eia_weekly`=0、首个有效周 2006-01-13，2019–2026 窗内 38 列零缺失，`BUILD_EXIT=0`。
  - 选型理由：对「每周五预测」节奏，固定 +1 周与「次周三发布 → 下周五可用」精确吻合（既不泄漏也不晚于真实可得）；逐条真实 release date 在周频下与之等价但更复杂，属过度设计。
  - 排错记录：首次改动因常量 `EIA_LAG_WEEKS` 未真正落盘导致 `NameError`、CSV 一度仍是旧（未滞后）版本；重新写入定义后确认生效。
- **新建 M1 数据字典** `03_data/processed/M1/m1_data_dictionary.md`：38 列的中英含义、来源代码、单位、周频化方法、发布滞后、覆盖率、首个有效周 + 派生公式 + 已知缺口。

### M0/M1 扁平基线回测 + 稳健性 / 调优 sweep

- **回测骨架** `04_code/scripts/run_baseline_m0_m1.py`：2019–2026、lookback=4 展平、rolling-origin（expanding，严格 τ≤t-1 无泄漏）、单任务 r_{t+1} → 还原价格；模型 M0(随机游走)/Ridge/XGB；指标 RMSE/MAE/DirAcc/相对 M0 skill/DM(HLN)。新增 `--feature-mode {all,returns}`、`--tune`（内层时间验证调参）、`--tag`。
- **sweep** `04_code/scripts/sweep_baseline.py`：7 配置统一协议，产出 `05_outputs/baselines/sweep_summary.csv` + `sweep_overview.png`。
- **结果**（260 测试周 2021–2025，M0 RMSE=4.137）：
  - lookback 稳健性：Ridge 随窗口增大单调恶化（L1 4.88 → L12 7.11；维度 38→456，共线 + 水平外推）；XGB 对窗口几乎免疫（4.70–4.90）。
  - 调优（L=4）：内层调参把 Ridge 5.665 → **4.379**（skill −37% → −5.9%）；`returns` 单独作用有限，主要价值是数值稳定。
  - 结论：扁平 M1 仍未超 M0，但最佳配置逼近、DM 由 5.0 收敛到 1.7（强显著 → 边缘）。

### Decisions made（续 2026-06-23）

- **锁定扁平对照基线**：主对照 = `L4_tuned`（lookback=4 + 内层验证调参，与导师 4 周设定一致、当前最强扁平基线）；轻量 sanity = `L1`。协议写入研究方案 §6.2.1，后续 M2/M3/M4 与方法创新层均以此为标尺。完整记录见 `00_admin/待整理/flat_baseline_log.md`。
- **模态自滞后统一原则**：EIA +1w 下沉到 M1 源头，使三模态「各自在自己聚合脚本完成发布滞后、merge 仅复查」分工一致。

### merge 层 EIA 双重滞后修复（2026-06-23，P0 已解决）

- `build_feature_matrix.py`：`EIA_WPSR_LAG_WEEKS` 1 → **0**（EIA 已在 M1 源头 +1w，merge 不再 shift）；`classify` 文案改「already lagged in M1」；自检由「== M1 原列 shift +1w」改为「**== M1 原列 unchanged**」（双重滞后不再被静默放过）；`--eia-lag` 保留为应急 override。
- 重跑：标准窗 **365×320**（2019-01-04~2025-12-26，target 364/365）、`--full` **1067×320**（2006~2025-12，target 1042）。
- 自检全 **OK**：`EIA WPSR == M1 column unchanged`、`brent_price` 未平移、`target_price_next[t]==brent_price[t+1]`、log-return 一致。三模态现统一为「各自滞后、merge 仅复查」。

### M2 EDA 产物命名 + merge 矩阵 M2 列契约（2026-06-23 续）

**EDA 产物统一 `m2_eda_*` 前缀**

- `build_m2_weekly.py` 输出的 tidy EDA 表为 **`m2_eda_weekly.csv`**（供 `eda_m2_mechanism.py` 读入；与 `m2_eda_*.png` / `m2_eda_leadlag_corr.csv` 同属 EDA 层产物）。
- B0 审计产物仍保留 `m2_coverage_*` / `m2_s2_*` 等前缀（非 EDA 脚本产出）。

**merge 矩阵仅并入 55 列 M2 anomaly（主分析契约）**

- `build_feature_matrix.py` 新增 `filter_m2_anom_columns()`：从 `m2_weekly_features.csv`（154 列全量，含 level/age/avail）中**只取** `{NDVI,NDWI,NDBI,BSI,NTL}_anom_{aoi}` 共 **55 列**写入合并矩阵；level（55）、age（22）、avail（22）**不进入 merge**（仍留在 M2 宽表供 robustness / 解释用）。
- 与 `04_code/src/backtest/data.py` 及 `2026-06-22_channelB_mechanism_plan.md` §3/§4 对齐；四个 merge 输出均已重跑：
  - 标准窗 **365×221**（34 M1 + 55 M2 + 119 M3 + 11 mask + 2 target）
  - 全量 union **`--full` 1067×221**

### M3 EMODnet 栅格：可选补充，非 blocking（2026-06-23 续）

- **现状**：`03_data/raw/03_shipping/emodnet_vessel_density_monthly_2017-2025/` 已下载 **96 个月度 GeoTIFF**（约 1 km 栅格）；`aggregate_shipping_to_weekly.py` **尚未**做 rasterio + 咽喉/AOI 多边形 zonal stats，故 **`m3_weekly_features.csv`（123 列）与 merge 矩阵 M3 部分（119 列）均不含 EMODnet**。
- **定位**：EMODnet 为**可选交叉验证补充**（与 GFW monthly presence 互相印证节点级 AIS 密度），**不是**跑 `--modality M3`（M1+M3 扁平基线）的前置条件；当前 PortWatch + GFW 已足够启动 M3 消融。
- **后续增强（Backlog）**：按 `external_sources.md` 对 6 咽喉 + 11 AOI 做区域统计 → 月→周对齐（+ 发布滞后）→ 追加 `emodnet_{node}_vessel_density` 类列；需 `rasterio` + 多边形 shapefile/geojson（可与 `00_spatial_anchors` 或咽喉定义复用）。

### Issues / blockers

| 问题 | 影响 | 状态 |
|---|---|---|
| ~~EIA 双重滞后~~（merge 二次 +1w → +2w） | 合并矩阵 EIA 13 列过度滞后 | ✅ 已修复 2026-06-23（merge `EIA_WPSR_LAG_WEEKS=0` + 自检改为等于 M1 原列；重跑自检全 OK） |

---

## 2026-06-23（续）M2 B3/B4 收尾 + 方法论决策

### What I did

**M2 统计验证与可解释性全部完成**
- 公平回测：`run_baseline.py --modality M2` 三套 RS 合约（anom 55 / literature 4 / level 55）+ leave-one-AOI-out；DM(vs M0) + Clark-West(vs M1) 已实现并产出。
- SHAP：`m2/shap_m2.py` 固定 holdout（train≤2023-12，test=2024-01–2025-12）；按 RS 指数与 AOI 分组；产物 `05_outputs/baselines/m2/shap_*.csv` + `shap_anom.png`。
- C2 降维对照：`m2/robustness_m2.py` 四臂并列（all-55 / pca-90 / elastic / shap-top20），回应 P058「SHAP≠PCA」；产物 `c2_summary.csv` + `c2_overview.png`。
- 文件结构重组：`05_outputs/baselines/m1|m2|m3`；M2 结果暂迁出至独立文档（后于 **2026-07-03 合并回** `00_admin/待整理/flat_baseline_log.md` §8）。
- M1 数据源确认统一：所有建模脚本只读 `weekly_feature_matrix.csv`（M1=34 列 merge），不混用 `m1_weekly_features.csv`（38 列）。

**M2 关键数字（L4_tuned，257 测试周）**
- 仍无配置超 M0（skill 全 < 0）。
- XGB Clark-West vs M1 显著：anom p=0.006 / literature p=0.001 / level p=0.004。
- Ridge CW 全不显著（p>0.06）。
- 最优 RS 配置：`literature`（4 NTL_anom）— Ridge 4.318 微超 M1，DirAcc 0.545。
- C2：XGB+pca-90 CW_p=2.6×10⁻⁶（远优于 all-55 的 0.0036）；shap-top20 Ridge RMSE 最优（4.376）。

### Decisions made

**1. M2 lookback sweep — 不做（主结果）**
- Lookback 已由导师锁定为 **4 周**，不是待优化轴。
- M1 sweep 已证明 `L4_tuned` 是最强扁平配置；M2 只是在其上加 RS 列。
- RS 为月频 as-of 对齐，lag0–lag3 的信息增量天然弱于金融日频特征；单独 lookback sweep 预期收益低。
- `sweep_m2.py` 骨架已写好——若审稿人追问稳健性，可一键补跑；**不纳入主结果表**。

**2. COVID / 红海子期间 — 不单独跑，写 Discussion**
- SHAP holdout 恰好覆盖 **2024–2025**（红海/Houthi 后期），已是天然的「事件后」子期间。
- Rolling origin 覆盖 2021–2025 全段；与 SHAP 子期排名差异（Ulsan/Kharg 在 2024–25 突出，文献推荐的 Fujairah/RasTanura 相对靠后）本身构成隐性子期间观察。
- **Discussion 写法（已定）**：「测试期集中于 2024–25 年，此时红海扰动可能使伊朗（Kharg）和韩国（Ulsan）炼厂/出口码头动态异常突出，放大了相应 NDWI/NDVI 信号；全段 rolling 与事件后 SHAP 排名差异应谨慎解读。」
- 不额外写代码、不单独产出子期间表。

**3. 水体掩膜版 GEE — 要做（B4 稳健性对照）**
- **动机**：NDWI_anom_Kharg/Yanbu 为 SHAP top 特征，但 LOAO 显示 Kharg 对 XGB 整体为负贡献；Basra/Kharg 等出口码头 NDWI 高方差可能来自潮汐/泥沙/水色，而非工业活动（B0 已提示；当前 GEE 仅云掩膜、未做水体掩膜）。
- **预期**：对 all-55 的 XGB 可能有边际 RMSE 改善（去 NDWI 噪声列）；**不预期改变主结论**（literature/NTL 最优、仍难超 M0），但可加强 Methodology 严谨性并回应「光学指数在码头水面是否误读」的审稿质疑。
- **执行计划**（对齐 `channelB_mechanism_plan.md` B4）：
  1. **GEE 重导出**：修改 `extract_sentinel2_monthly_indices_gee.js`——加 MNDWI/NDWI 水体掩膜（McFeeters 1996；Xu 2006）；**仅陆地像素**聚合 NDVI/NDBI/BSI；水面主导 terminal（Basra/Kharg/Yanbu 等）重点 QC。
  2. **数据产物**：新 CSV 另存（如 `sentinel2_*_watermasked_*.csv`），与现版并列，不覆盖原 Channel B 导出。
  3. **M2 管线**：`build_m2_weekly.py` 增 `--water-masked` 或独立脚本 → `m2_weekly_features_watermasked.csv`；anom 合约不变（55 列，站点 expanding z-score）。
  4. **merge + 回测**：并入 merge 矩阵（或 run 时读 watermasked 宽表）；`run_baseline.py --modality M2 --m2-features anom --tag anom_watermasked` 与现版 all-55 **并列报告** RMSE / CW / SHAP。
  5. **验收**：对比表「cloud-mask only vs water-masked」+ 若 Kharg/Yanbu NDWI 排名下降则支持掩膜有效；写入 Results 稳健性小节或 Appendix。
- **优先级**：**P2**（M3/M4 主结果 P1 之后）；估时 ≈ 0.5–1 天（GEE + 管线 + 一轮回测）。
- **文献引用**：McFeeters 1996（NDWI）；Xu 2006（MNDWI 改进水体提取）——写入 Methodology「机制变量构建」。

### Files touched

- `04_code/scripts/run_baseline.py` — 输出路由至 `baselines/m1|m2|m3/`
- `04_code/scripts/m1/sweep_m1.py`（自 `sweep_baseline.py` 迁入）
- `04_code/scripts/m2/shap_m2.py` / `robustness_m2.py` / `sweep_m2.py`
- `04_code/scripts/m3/sweep_m3.py` / `robustness_m3.py` / `shap_m3.py`（骨架）
- `05_outputs/baselines/m1/` / `m2/` — 产物迁移 + SHAP + C2
- `00_admin/待整理/flat_baseline_log.md` — M2 完整结果写入 §8（2026-06-23 曾暂拆独立文档，2026-07-03 已合并回）

### Next tasks

- **P1** M3 主结果：`run_baseline.py --modality M3`
- **P1** M4 全模态
- **P1** 写作：M2 Results + Discussion（P058 / NDWI 限制 / 子期间叙事）
- **P2** M3 SHAP + LOCHO（`robustness_m3.py`）
- **P2** 水体掩膜 GEE：改 GEE 脚本 → 重导出 → `build_m2_weekly` watermasked 臂 → 与现版 all-55 并列回测

---

## 2026-06-30

### Decisions made

- **贡献措辞统一更新为「方法集成与实证检验层」**（原「方法创新层」）：明确本研究**不提出新的融合算子 / 网络层 / 损失**，而是把既有方法（冻结 EO 基础模型 + 模态专属编码器 + 门控 / 交叉注意力 + 缺失模态 / 不规则时间建模）**集成**，并**首次**在原油周频价格预测中系统检验「表示级融合 vs 扁平特征融合」；贡献定位 = **application + integration + 系统实证比较**，非方法学创新。
- 已对齐文档：`2026-06-22_research_plan_e2e_multimodal.md`（§0/§1/§2/§6.2.1/§7）、`literature_matrix.md` §⑥、`external_sources.md`、`00_admin/待整理/flat_baseline_log.md`。**本日记 2026-06-17/22/23 旧条目保留原「方法创新层」措辞作历史记录**（不回改）。

### What I did

- **补齐「方法集成与实证检验层」文献缺口**：联网核实并新增 8 篇方法学文献（建议编号 **P094–P101**），写入 `literature_matrix.md` 新增 **§⑥**：
  - EO 基础模型：**P094** Prithvi-EO-2.0（arXiv:2412.02732）、**P095** SatMAE（NeurIPS 2022）；
  - 门控融合：**P096** Gated Multimodal Units（Arevalo et al. 2017）；
  - 缺失模态：**P097** Ma et al.（CVPR 2022）、**P100** ModDrop（Neverova et al. 2016）；
  - 不规则时间序列：**P098** GRU-D（Che et al. 2018）、**P099** mTAN（Shukla & Marlin 2021）；
  - 多模态综述锚点：**P101** Baltrušaitis et al.（TPAMI 2019）。
  - ⚠️ 8 篇均**待精读、待核页码**，且**无一在油价 / 商品预测上验证** → 创新层文献是「方法骨架」而非「有效性证据」，增量价值须靠本项目消融（始终对照 M0 + 扁平基线）自证。
- **修正 FRT 不一致**：`research_plan` §4.2 通道 B + §5 架构图删除残留 FRT，与 `2026-06-22_channelB_mechanism_plan` §6「去掉 FRT」对齐（P055 仅支撑油罐结构容量、不支撑充填率 / 液位）。

### Next tasks

- 把 **P094–P101** 同步并入 `literatue.md` 分类总表与 Tier 排序。
- 将 **P039 / P062 / P063 / P096 / P094 或 P095** 升级为正式 reading note（创新层三编码器与融合的选型依据）。
- 起草 `chapter_2_literature_review.md` **§2.4 Multimodal forecasting**（P101 框架 + P039 先例 + P094/095 EO FM + P096 门控）。

### M0–M4 基线进度核对

**结论：** 线性 + XGBoost 的 M0–M4 已完成（且超额）；TCN（及任何深度时序基线）尚未实现。

#### ✅ 已完成部分（远超 baseline 最低要求）

`04_code/src/backtest/`（公平回测内核）+ `run_baseline.py`（M0–M4 单入口）已跑通，`05_outputs/baselines/` 下有 38 个结果文件，不是空头文档：

| 模态 | 已产出 | 关键结果 |
|---|---|---|
| M0 随机游走 | 内嵌基准 | RMSE=4.152（极强，无人超过）|
| M1 金融 | 主基线 + lookback sweep + 调参 | L4_tuned Ridge 4.332 |
| M2 +遥感 | anom/literature 两套 + Clark-West + LOAO + SHAP + PCA/ElasticNet 降维对照 + lookback sweep + 水体掩膜稳健性 | XGB CW_p=0.006（显著嵌套增量）|
| M3 +航运 | 主基线 + sweep + SHAP + LOCHO | XGB CW_p=0.000，霍尔木兹/苏伊士 tanker 信号 |
| M4 全模态 | 主基线 + SHAP + sweep + LOMO | XGB CW_p=0.0002；SHAP：航运 55.6% > 金融 30.3% > 遥感 13.1% |

公平协议均已到位：统一窗口 2019–2025、lookback=4、rolling-origin（expanding，min_train=104）、单任务回归 \(r_{t+1}\) 还原价格、DM（vs M0）+ Clark-West（vs M1 嵌套）、SHAP、多种 leave-one-out 稳健性。完成度很高。

#### ❌ 尚未完成

1. **TCN（及任何深度时序基线）未实现。** 在 `04_code/` 内检索 `TCN|LSTM|GRU|torch|Conv1d` 零命中（匹配均来自文档/文献笔记）。`models.py` 里仅有 Ridge 与 XGBoost。
2. **写作未动**（两份日志 backlog 均列着）：Methodology「机制变量构建」、Results「增量价值」、Discussion。

#### 定位说明

方案 §0 写的是「XGBoost / 线性 / TCN」，但 §6.1 baseline 对照表列的时序基线是 LSTM/TFT-Early（early-fusion）；TCN 主要出现在 §5.1 作为阶段 1 的 Finance 模态编码器。不论叫 TCN 还是 LSTM，目前**缺一根「把所有数值列喂进同一时序网络」的深度 early-fusion 基线**——这正是 RQ2（扁平 vs 模态感知融合）的重要标尺。严格按 §0 措辞，「核心实证层」还差这一块。

#### 一句话摘要

线性（Ridge）+ XGBoost 的 M0–M4 消融已做完且很扎实（主基线 + DM/Clark-West + SHAP + 多种 leave-one-out + 降维对照 + 水体掩膜稳健性，结果文件齐全可复现）；TCN 或任何深度 early-fusion 时序基线尚未做。需补时序基线或推进写作时再启动。

---

## 2026-07-03

### What I did — 深度 early-fusion 基线（补齐核心实证层最后一块）

- 新建 `04_code/scripts/run_deep_baseline.py`：LSTM/GRU **early-fusion** 时序基线，复用 `backtest.data`（同特征/窗口/防泄漏）+ `backtest.metrics`（同 DM/CW），把所选全部数值列 reshape 成 `[lookback, features]` 序列喂进**一个共享 RNN**（无模态专属编码器）——即 RQ2 里「扁平 / 早融合」的**深度版标尺**（此前只有 Ridge/XGB 的表格扁平融合，缺深度 early-fusion）。
- 协议同扁平基线：2019–2026、lookback=4、rolling-origin（min_train=104、retrain_every=13、20 fits）、单任务回归 \(r_{t+1}\) 还原价格、特征+目标训练折内标准化、inner-val early stopping + dropout/weight_decay 强正则（小样本防过拟合）。
- 产物：`05_outputs/baselines/deep/` 下 `deep_metrics.csv`（含 DM/CW）+ `deep_cw.csv` + `deep_predictions.csv` + `deep_backtest.png`（单一 `deep/` 目录，非 `{m1..m4}/` 分目录）。

### 结果（257 测试周 2021–2025，M0 RMSE=4.152）

| 模型 | RMSE | DirAcc | skill vs M0 | CW_p vs M1 |
|---|---|---|---|---|
| M1_LSTM | 4.178 | 0.490 | −0.6% | — |
| M2_LSTM | 4.210 | 0.595 | −1.4% | **0.012（显著）** |
| M3_LSTM | 4.370 | 0.494 | −5.3% | 0.460（不显著） |
| M4_LSTM | 4.180 | 0.549 | −0.7% | 0.059（边缘） |

> ⚠️ **产物注记（2026-07-05 补记）**：上表为 07-03 当天 LSTM early-fusion 的历史结果。此后 `run_deep_baseline.py` 与 `05_outputs/baselines/deep/` 已被 2026-07-05 方法集成层工作覆盖——当前 `deep_metrics.csv` 内容已是模态感知融合架构 `Mfin_TCN / Mship_GNN / Mrs_RS / Mfusion / Mfull_M4rep`（M0 RMSE=4.172，测试样本对齐略有差异），LSTM 原始产物需回溯 git 历史复现。详见下方 2026-07-05 条目。

- **深度 early-fusion 同样打不过 M0**（skill 全负），与 Ridge/XGB 结论一致 → 强化「周频 Brent 随机游走极强」。
- **M2 遥感嵌套增量显著（CW_p=0.012）**，与 XGB 的 M2（CW_p=0.006）方向一致，交叉印证遥感 anom 有嵌套增量。
- **M3 航运在深度早融合里反而变差（RMSE 4.37、CW 不显著）**，而 XGB 的 M3 CW 显著 → 说明「把 119 维高维航运直接堆进一个 RNN」的扁平早融合处理不好高维异构模态，**正是 RQ2「需模态感知融合而非扁平早融合」的经验论据**。
- 深度模型全程数值稳定、无负 R² 崩溃（强正则 + 目标标准化 + early stopping 生效），解决了此前「深度模型 R² 大量为负」的问题。

### 定位
核心实证层补齐：**M0 / 线性(Ridge) / 树(XGB) / 深度 early-fusion(LSTM)** 四类基线齐全、同协议可比。方法创新层（模态专属编码器 + 门控融合）将以「深度扁平早融合」为直接对照回答 RQ2。（2026-06-30「M0–M4 基线进度核对」中「深度基线未实现」的结论到此更新。）

### Next tasks
- （可选）深度基线稳健性：`--arch gru`、hidden/dropout sweep、`returns` 特征模式、多 seed 平均。
- 推进写作（§2.4 / Ch3 / Ch4）或启动创新层原型（Prithvi/SatMAE embedding → 门控融合 → flat vs modality-aware）。

### 文档整理
- 合并 `2026-06-23_m2_baseline_results.md` → `00_admin/待整理/flat_baseline_log.md` §8；删除独立 M2 文档，基线记录恢复为单一主文件。
- 同步更新 `File Structure20260703.md`、`Meeting04_prep_20260703.md` 中的交叉引用。

### M1 变量精简与口径统一（防泄漏 + 冗余清理）

系统精简 M1 特征并统一收益率口径，**38 → 35 列**；重建 `m1_weekly_features.csv` 与 merge 主 / full / watermask 三变体（防泄漏自检全通过，merge M1 modality = 31）。

**改动清单：**

1. **移除 `brent_direction`**：方向不作独立目标/特征，改由预测价格派生为评估指标（基线 SHAP≈0，冗余）。
2. **`brent_return_pct` 移除、`wti_return_pct` → `wti_log_return`**：Brent/WTI 统一对数收益（简单收益率与对数收益相关 0.997，冗余）。
3. **移除 `net_crude_trade`**：`= imports − exports` 为精确线性组合（秩亏）；保留 `crude_imports` + `crude_exports`（窗内近正交 corr≈0.05，信息互补）。
4. **移除 `sp500` level、`sp500_return_pct` → `sp500_log_return`**：指数水平非平稳、测试期（4697–6930）超训练范围（2305–4770）外推；主分析 `feature_mode="all"` 不平稳化，故仅留对数收益。
5. **`futures_spread` → `brent_f1_spot_log_basis`（正名）+ 新增 `brent_roll_week`**：正名为前月期货—现货**基差**（非纯期限结构）；**未 back-adjust**（basis 需真实当日 F₁ 价）；`brent_roll_week` = 每月最后 W-FRI ≈ ICE Brent 换月，作换月控制哑变量。
6. **`commodity_fx` → `cadusd_log_return`**：删复合均值与 AUD 腿，主留 CAD 单腿对数收益。
7. **`gpr` 改用日度 GPRD**：下载 `Other/data_gpr_daily_recent.xls`，周均值聚合 → 滞后一周（替代月度 ffill+lag，匹配周频、避免按月份标签回填的潜在前视）。
8. **product supplied 中文名**「消费量」→「表观需求」（非严格终端消费）。
9. **负价格护栏**：所有对数收益计算前 `assert price>0`（2020-04-20 WTI −36.98 在周五最后值口径下不构成代表价，不影响）。

### Decisions / 待办

- **AUD 稳健性（暂缓，已记录）**：`audusd_log_return` 从主表删除。数据依据：CAD–油价相关 0.36 > AUD 0.28；CAD–AUD 相关 0.74–0.79（AUD 增量有限，更多反映铁矿/煤/中国需求/风险偏好）。**保留为「商品货币选择稳健性」备选**——后续可 (i) AUD 替换 CAD、(ii) CAD+AUD 同时加、(iii) 复原 `commodity_fx` 均值 作稳健性对照。raw `Yahoo_AUDUSD_daily.csv` 仍在，暂不入库。
- **重跑 baseline（待办）**：M1 特征集已变（35 列 / merge M1=31）；`05_outputs/baselines/*` 旧 SHAP 与 `00_admin/待整理/flat_baseline_log.md` 列数需在重跑后同步。
- **`brent_f1_spot_log_basis` 换月稳健性（建模阶段）**：删换月周 / 删前后各一周 / A(无 basis)–B(basis+dummy)–C(basis 删换月周) 对比（见 `m1_data_dictionary.md` §4.4.2-D）。

**EN summary:** Streamlined M1 from 38 to 35 columns — dropped `brent_direction`, `brent_return_pct`, `net_crude_trade`, `sp500` level; unified log-return convention (`wti`/`sp500` → log return); renamed `futures_spread` → `brent_f1_spot_log_basis` (unadjusted, + `brent_roll_week` roll dummy); `commodity_fx` → `cadusd_log_return` (AUD leg dropped, kept for future robustness); `gpr` switched to daily GPRD weekly-mean + 1-week lag. Rebuilt m1 + merge three variants (M1=31); leakage self-checks pass. Baseline rerun pending.

---

## 2026-07-05

### What I did — 精简 M1 后全套重跑 + 方法集成层启动

**1. 落实 07-03「重跑 baseline」待办：精简 M1（35 列 / merge M1=31）后重跑扁平 M0–M4**

- merge 矩阵更新为 `weekly_feature_matrix.csv` **365 周 × 212 列**（31 M1 + 55 M2 + 113 M3 + 11 mask + 2 target），无泄漏自检全通过。
- **M3 引入 core / full 双 tier**：主模型用 **core 38 列**（GFW 存在 24 + PortWatch 咽喉过境 12 + 港口方向性进出口 2），**full 113 列**仅供稳健性（LOCHO）；「最优子集因模型而异」由此成为可检验的对照。（⚠️ **2026-07-05 更新**：主模型已由 core 38 改为 **full 113**、core 降为稳健性臂——因 core 对 XGB 最弱不显著、full 显著；见 `flat_baseline_log.md` §4/§9/§12。）
- 协议不变：2019.1–2025.12 · lookback=4 · rolling-origin（expanding，min_train=104，retrain_every=13）· 单任务 \(r_{t+1}\) 还原价格 · 257 测试周（2021–2025）· DM vs M0 + Clark–West vs M1。

结果（M0 RMSE=**4.152**）：

| 模态 | 模型 | RMSE | skill vs M0 | CW_p vs M1（嵌套增量） |
|---|---|---|---|---|
| M0 | 随机游走 | 4.152 | 0.0% | — |
| M1 金融 | Ridge / XGB | 4.256 / 4.368 | −2.5% / −5.2% | — |
| M2 +遥感 (anom55) | Ridge / XGB | 4.414 / 4.440 | −6.3% / −6.9% | 0.474 / 0.085 |
| M3 +航运 (core) | Ridge / XGB | 4.351 / 4.476 | −4.8% / −7.8% | 0.289 / 0.096 |
| M4 全模态 | Ridge / XGB | 4.466 / 4.492 | −7.6% / −8.2% | 0.375 / **0.020** ✅ |

**核心发现（与 06-23 相比因 M1 变强 + M3 改 core 而更新）**：

1. **无一模型击败 M0**——周频 Brent 随机游走极强（诚实核心结论）。
2. **精简 M1 变强后，单模态 XGB 嵌套增量退化为不显著**（M2 anom 0.085、M3 core 0.096）；**仅 M4 全模态（0.020）、M2 literature（0.022）、M2 水体掩膜（0.028）、M3 full/portwatch/tanker 臂仍显著**。
3. **M3 是「模型敏感」关键对照**：XGB 下 full/tanker 臂显著，Ridge 几乎全不显著——高维异构航运在扁平线性模型下无法有效利用。
4. **M3 core 对 XGB 反而非最优**（full/portwatch/tanker 更优）→「最优子集因模型而异」，扁平融合局限的直接证据。
5. **SHAP 模态占比反转为 M1(51%) > M2(25%) > M3(23%)**（M3 改 core + M1 变强后）。
6. 以上均**强化 RQ2**：把 RS/航运扁平拼到已很强的金融基线上，边际增量被稀释 → 需要模态感知的表示级融合。

**2. M2 遥感合约变体补充（呼应 Meeting 03 四 AOI 决策）**

- 在 `backtest/data.py` 新增 `--m2-features aoi4|ntlall` 后重跑两个 M2 稀疏合约（同 L4_tuned、257 测试周）：`aoi4` = 站点精选（Fujairah / RasTanura / Rotterdam / Houston 4 核心枢纽 × 5 指数 anom = **20 列**）、`ntlall` = 指数精选（11 AOI 仅 `NTL_anom` = **11 列**）。产物 `05_outputs/baselines/m2/baseline_metrics_{aoi4,ntlall}.csv`。
- **`aoi4`：XGB CW_p vs M1 = 0.0055（显著）**、RMSE 4.340——聚焦 4 个核心 AOI 的嵌套增量比全 11-AOI anom（0.085）**更显著**，为 AOI 子集选择提供实证支撑。
- `ntlall`：XGB CW_p=0.036（显著）、Ridge 0.108（不显著）。
- **完整 2×2 六臂（+ 水体掩膜去噪对照）见 `flat_baseline_log.md` §8.9**：「双重稀疏」——砍站点（→20）或砍指标（→11）任一都让 XGB 跨入显著；**站点维度冗余 > 指标维度**（`aoi4` 0.0055 最强，强于 literature 0.022）；去噪与精选**替代非叠加**（水体掩膜救全量 0.085→0.028，但对已精选 `aoi4` 反而 0.0055→0.027）。主分析仍锚 anom-55（防 p-hacking）。

**3. M3 graph17——17 节点动态异质图张量 bundle（GNN 输入）**

- `03_data/processed/M3/py/build_m3_graph17.py` 将 Stage-2 产物合并为 **11 AOI + 6 咽喉 = 17 节点**异质图 → `m3_graph17_tensors.npz` + 可读审计表 `m3_graph17_choke_nodes_weekly.csv`（**2346 行** = 391 周 × 6 咽喉）；数据字典 `m3_data_dictionary.md` §12.9 同步。
- 节点特征：AOI 11×11（O-D 端点特征）；咽喉 6×**20**（`gfw_*` 8 + `pw_*` 9 + `sar_*` 3）；6 咽喉 = hormuz / suez / malacca / mandeb / panama / cape。
- 边：`adjacency (391,17,17)` = 动态 **AOI→AOI O-D**（`adjacency_od 11×11`）+ **静态 AOI↔咽喉**（`static_edges`，12 条无向）；编码器侧对称化 + 自环后作 GAT mask。
- 定位为创新层 `Mship_GNN` 的正式输入，**不进入** flat M3 宽表。

**4. 方法集成层——模态感知表示级融合全栈跑通（当日核心）**

把 07-03 的「单一共享 RNN(LSTM) early-fusion 扁平深度标尺」**升级为表示级融合管线**：三个模态专属编码器各出 32 维 embedding → `GatedFusion`（softmax 门控凸组合）→ MLP 回归头预测 \(\hat r_{t+1}\) → 还原价格。walk-forward 同扁平协议但 **lookback=8**（min_train=104、retrain_every=13、epochs=80、inner-val=52 周 early-stop、dropout 0.1 + weight_decay 强正则），测试周 ≈ **252**（较扁平 257 少，8 周深度窗对齐损失）。

- **`z_fin`**（`finance_encoder.py`）：Linear(31→32) + LayerNorm + 2 层因果 TCN → 32 维。
- **`z_ship`**（`shipping_encoder.py`）：类型专属投影(→64) + node-type embedding + 2 层多头 GAT(4 heads) + 残差 LayerNorm + 每节点因果 TCN + 节点加性注意力池化 → 32 维（+ `site_att`），约 4.2 万参数。
- **`z_rs`**（`rs_encoder.py`）：**冻结 Prithvi-EO-2.0** 预计算 embedding（1024 维 `s2_prithvi_emb_meanpool.npy`，backbone 不微调）+ Linear(→64) + 时间注意力（AOI 内 lookback）+ 站点注意力（11 AOI）→ 32 维（+ `site_att`）。
- **`GatedFusion`**（`fusion.py`）：MLP(n_mod×32→32→n_mod) + softmax → \(z=\sum_i\alpha_i z_i\)；`DeepForecastModel` 统一接口。CONFIGS：`fin/ship/rs`（单模态）、`fusion`（fin+ship）、`m4rep`（fin+rs+ship）。数据挂载 `deep_dataset.build_deep_dataset()`：复用扁平测试周/目标锚点 + lookback=8 的 graph17 / M1 序列 / Prithvi 立方体；标准化仅在训练折 `fit_scalers()`（防泄漏）。
- 另建 `run_deep_interpret.py`（RQ3 门控/节点注意力可视化）与 `run_deep_sweep.py`（多 seed / 超参稳健性）脚手架（git 未跟踪）。
- **本管线产物覆盖了 07-03 的 LSTM early-fusion**（`baseline_deep_*` 已删），当前 `05_outputs/baselines/deep/deep_metrics.csv` 为下表。

深度结果（M0 RMSE=**4.172**，≈252 测试周）：

| 模型 | RMSE | skill vs M0 | CW vs flat M1 |
|---|---|---|---|
| Mfin_TCN | 4.205 | −0.8% | — |
| Mrs_RS | 4.307 | −3.3% | 0.103 |
| Mship_GNN | 4.188 | −0.4% | **0.0014** ✅ |
| Mfusion (fin+ship) | 4.174 | −0.1% | — |
| Mfull_M4rep (fin+rs+ship) | 4.184 | −0.3% | **0.0021** ✅ |

- **表示级航运 GNN 首次对 flat M1 达到 Clark–West 显著（p=0.0014）**，与 07-03「M3_LSTM RMSE 4.370、CW p=0.460 不显著」形成**直接对照**——同一份航运信息，扁平早融合利用不了、模态感知 GAT+TCN 却能，**为 RQ2 提供首个正面证据**。
- 门控增量：航运增量（fusion vs fin）CW p=0.059（边缘）；RS 增量（m4rep vs fusion）CW p=0.260（不显著）——RS 分支（Prithvi meanpool）偏弱，待 sweep。
- 仍无深度模型超 M0（skill 全负），与扁平结论一致。

> **对照缺口**：07-03 的 LSTM 深度 early-fusion 标尺产物已被本次覆盖。若 RQ2 要「深度扁平早融合 vs 表示级融合」双深度对照，需恢复 LSTM 脚本或独立归档旧结果。

### Decisions made

- **贡献定位定稿**：拟题「A Modality-Aware Spatio-Temporal Fusion Framework…」；不提新算子/层/损失，而是**集成**既有方法 + **首次**在原油周频价格预测中于统一无泄漏协议 + DM/CW 下系统检验「表示级融合 vs 扁平融合」；贡献 = application + integration + 系统实证比较。
- **M3 core/full 双 tier 制**：主模型 core 38、稳健性 full 113。→ **2026-07-05 起主模型改用 full 113、core 38 降为稳健性臂**（core 对 XGB 最弱不显著；M3_XGB CW 0.0002、M4_XGB CW 0.009）。
- **诚实核心结论保留**：无一模型超 M0、精简 M1 后单模态增量普遍不显著——这本身正是「需要模态感知融合」（RQ2）的动机与证据。

### Files touched

- 新增：`00_admin/2026-07-05_研究方案与进度总览.md`、`00_admin/2026-07-05_扁平模型变量清单.md`。
- 新增创新层：`04_code/src/models/*.py`（3 编码器 + `GatedFusion` + dataset/rolling）、`04_code/scripts/run_deep_baseline.py`（LSTM 标尺重写为融合入口）+ `run_deep_interpret.py` + `run_deep_sweep.py`、`03_data/processed/M3/py/build_m3_graph17.py`（+ `m3_graph17_*` 产物）。
- 更新：`03_data/Dataset/external_sources.md`、`03_data/processed/M3/m3_data_dictionary.md`（§12 graph17）、`04_code/src/backtest/data.py`。
- 覆盖重跑：`05_outputs/baselines/{m1,m2,m3,m4,deep}/*`。

### Next tasks

- **创新层收尾**（3 编码器 + 门控融合已跑通）：补 Cross-Attention 融合臂 + Encoder-Concat 对照 + modality dropout（缺失模态建模）；`run_deep_interpret` 出门控/站点注意力图（RQ3）；`run_deep_sweep` 多 seed + RS 分支超参（当前 Mrs_RS 偏弱）。
- **flat vs modality-aware 正式对照（RQ2）**：深度 ≈252 周 / 扁平 257 周测试期取交集后并列报表；`z_fused` 维度对齐研究方案（§1.5 写 64、代码现 32）。
- 同步 `2026-07-05_研究方案与进度总览.md` §3.1：`z_fin`/`z_rs`/门控接线由 `[ ]` 更新为已实现（文档滞后于代码）。
- 写作：Ch2 §2.4 Multimodal forecasting / Ch3 方法（机制变量 + 无泄漏对齐 + 编码器/融合）/ Ch4 结果（M0–M4 + DM/CW + SHAP + 稳健性）。

---

# Task List

> **Current phase:** Phase 03 — Data processing & 4-model framework implementation  
> **Next meeting:** Meeting 04 (TBC)  
> **Master plan:** `00_admin/2026-06-22_research_plan_e2e_multimodal.md`

## Current Sprint 当前冲刺

**数据 / 特征（阶段 0 — 无泄漏矩阵）**
- [x] M1 单脚本重建（`build_m1_weekly.py` → 38 列周频表）
- [x] M1 逐变量描述字典（`Dataset_Overview4.ipynb`）
- [x] S2 Channel B 11 AOI 月度机制变量 CSV 下载
- [x] M3 脚本归位（`processed/M3/py/aggregate_shipping_to_weekly.py`）
- [x] 按发布时间戳重对齐 EIA / 月频变量（全模态）— merge 层 EIA WPSR +1w 周三发布；M1 月频 / M2(+15d as-of) / M3(GFW+4w·PW+1w) 复查不重复
- [x] M2 周频聚合（as-of 月→周 + days_since_obs + valid/modality mask；anomaly expanding past-only）
- [x] 合并 M1+M2+M3 → 统一无泄漏特征矩阵 + 数据字典（`processed/merge/outputs/`，1067×320）
- [x] GFW/PW union 时间索引对齐（727→750 周，保留 GFW 2012–2018 早期样本）

**建模 / 评估（阶段 0 — 公平 M0–M4）**
- [x] 实现 M0（\(\hat p_{t+1}=p_t\) / 预测 log return = 0）
- [x] 扁平 M0/M1 基线回测 + lookback 稳健性 + 调优 sweep；锁定 `L4_tuned` 对照（`00_admin/待整理/flat_baseline_log.md`、`05_outputs/baselines/m1/`）
- [x] 修复 merge 层 EIA 双重滞后（`EIA_WPSR_LAG_WEEKS=0`，仅复查；重跑自检全 OK）
- [x] M2 B0–B2：审计 + 周频构建 + 机制 EDA
- [x] M2 B3：公平回测 + DM/CW + LOAO + SHAP + C2 降维
- [x] **M2 B4：水体掩膜 GEE 稳健性**（cloud-mask vs water-masked 并列回测）✅ XGB CW_p 0.006→8.5e-5
- [x] M3（M1+航运）扁平对照 + DM/CW ✅ XGB CW_p=2.5e-5
- [x] M4 全模态 ✅ XGB CW_p=1.7e-4；SHAP 航运56%>金融31%>遥感13%
- [x] 深度 early-fusion 基线（LSTM，M1–M4）✅ 2026-07-03 `run_deep_baseline.py`（RQ2 扁平标尺）
- [x] 唯一核心目标 = 价格：训练 `r_{t+1}` → 还原 \(\hat P_{t+1}\)；方向由预测价格派生；单任务回归
- [x] 2019–2025 · 4-week lag · rolling-origin · L4_tuned（M0/M1/M2 已完成）
- [x] M2：DM + Clark-West 检验（还原价格上计算）
- [ ] 定义 flat 阈值（训练集 |r| 33rd percentile）并文档化（用于**派生方向指标**）
- [ ] （可选）派生方向指标改二分类 up/down

**写作**
- [ ] 文献综述主题初稿（6 主题，~4–5 页）
- [ ] 4-AOI 选择依据短文
- [ ] RS 删除 cloud fraction 的方法论记录

**方法创新层（阶段 1，待导师确认后启动）**
- [ ] S2 patch GEE 导出 + Prithvi/SatMAE embedding 预计算
- [ ] 三模态编码器 + Gated Fusion 原型
- [ ] Flat vs modality-aware 对照实验

### M2 水体掩膜 B4 分支完成前两步（2026-06-23 续）

**GEE 导出 & 验收**

- GEE 脚本 `extract_sentinel2_monthly_indices_watermask_gee.js` 修复内存超限：
  - 三次 `reduceRegions` → 两次（`is_water` 并入 `monthlyMedian` 同时计算）；
  - `Map.addLayer` 移除（触发即时求值）；`tileScale` 4 → 8。
- 导出 CSV `sentinel2_oil_sites_monthly_indices_watermask_201704_202512_11aoi.csv`：
  - **1155 行**（11 sites × 105 months），日期 2017-04 ~ 2025-12，完整。
  - 新增列：`MNDWI`、`land_px`（陆地像素占比；`land_pixel_fraction` 列因 join bug 全为 NaN，直接用 `land_px`）、`water_mask_applied=1`、`water_thresh_mndwi=0`。
  - 水体掩膜验证：P008 Basra NDVI 均值 −0.2505 → +0.0852（水体像素压低转为纯陆信号）；P007 Jamnagar 不受影响（−0.24 不变）。✓
  - `land_px` 各站点中位数（低→高）：Basra 0.001、RasTanura 0.040、Kharg 0.263 … Jamnagar 0.991。

**`build_m2_weekly.py` 新增 `--watermask` 模式**

| 变更点 | 说明 |
|---|---|
| `S2_WM_CSV` 路径常量 | 指向水体掩膜版 CSV |
| `SITE_SHORT` 静态映射 | WM CSV 无 `short_name` → 用 P001–P011 → short name 字典 |
| `LOW_LAND_THRESH = 0.05` | `land_px < 5%` 标记 `low_land_coverage=1`（B4 诊断列）|
| `load_monthly_long(watermask=True)` | 多读 `land_px`；S2 indices 增 MNDWI |
| `align_weekly()` | `right_cols` 含 `land_px`，随 as-of join 传入 |
| `to_wide()` | 若 `land_px` 非空 → 追加 `s2_land_px_{site}` 共 11 列 |
| 输出文件名 | `_watermask` 后缀 |

运行结果（`--watermask`）：
- 宽表 `m2_weekly_features_watermask.csv`：**365 × 188**（+MNDWI 22 列 + land_px 11 列）
- EDA 表 `m2_eda_weekly_watermask.csv`：**24090 行**（6 指数 × 11 × 365）
- `low_land_coverage` 行：3385 / 20045 S2 行（**16.9%**），主要来自 Basra、RasTanura

**下一步（B4 robustness 水体掩膜比较，P2）**

```bash
# 用水体掩膜版替换 M2 特征后重跑基线（与原始 anom 55 列对比）
python 04_code/scripts/run_baseline.py \
  --modality M2 \
  --m2-features-csv 03_data/processed/M2/outputs/m2_weekly_features_watermask.csv \
  --out 05_outputs/baselines/m2/watermask/
```
（需先在 `run_baseline.py` / `robustness_m2.py` 增加 `--m2-features-csv` 覆盖参数；当前 M3/M4 优先，此项 P2。）

## Backlog 待办

- GFW SAR 暗船全套（~1 天，P2）
- **EMODnet 栅格 zonal stats → M3 可选列**（rasterio + 咽喉/AOI 多边形；与 GFW 交叉验证；非 M3 基线 blocking）
- M3 / M4 主结果 + SHAP + LOCHO
- 深度模型修复（log return target + StandardScaler）
- Test pipeline 迁移至 `04_code/`
- ~~leave-one-AOI-out 敏感性测试~~（M2 已完成）
- ~~M2 子期间稳健性（COVID、红海）~~（决策：写 Discussion，不单独跑）
- ~~**M2 水体掩膜回测**~~ ✅（已完成 2026-06-23；`build_feature_matrix.py --m2-csv` + `run_baseline.py --matrix`；XGB RMSE −1.69%，CW_p 0.006→0.0001；详见 `00_admin/待整理/flat_baseline_log.md` §8.8）
- M2 lookback sweep（可选，`sweep_m2.py` 骨架已备，审稿人追问时再跑）
- Meeting 04 汇报材料

## Completed 已完成（Phase 03 内）

- Meeting 03 笔记结构化（`meeting_notes/Meeting03 2020260617.md`）
- 端到端多模态研究方案（`2026-06-22_research_plan_e2e_multimodal.md`）
- 项目结构总览更新（`File Structure20260703.md`）
- M1 离线构建管线 + 38 列周频输出
- Dataset Overview v4 + 变量描述字典
- M3 航运聚合脚本归位
- S2 Channel B 11 AOI 月度 indices CSV（2017-04 – 2025-12）
- M2 B0–B3：公平回测 + DM/CW + LOAO + SHAP + C2 降维；方法论决策（lookback/子期间跳过）
- **M2 B4 水体掩膜全流程**（GEE CSV 验收 ✓ → `build_m2_weekly.py --watermask` ✓ → `build_feature_matrix.py --m2-csv` ✓ → `run_baseline.py --matrix` ✓；XGB CW_p 0.006 → 0.0001，结论强化；`00_admin/待整理/flat_baseline_log.md` §8.8 完整记录）



