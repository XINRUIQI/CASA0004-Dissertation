# Research Diary — Phase 04

> **Phase 04:** Dissertation writing — structure, results narrative, appendices & submission prep  
> **Trigger:** Meeting 04 (2026-07-08)  
> **Deadline:** Meeting 05 / final pre-submission meeting (2026-07-29); dissertation submission thereafter  
> **Details:** `00_admin/meeting_notes/Meeting04_Note 20260708.md` · `Meeting04_Mail 20260708.txt` · `06_writing/outline.md`

> **Format note:** Phase 04 diaries start **from Meeting 04 onward**. Pre-Meeting-04 modelling & data work remains in `research_diary_phase3.md`.

---

## 2026-07-08

### What I did

- Fourth meeting with supervisor Beatrice Taylor.
  与导师 Beatrice Taylor 进行了第四次会议。
- Reviewed flat vs deep multimodal results against Model 0; clarified how to present findings; shifted priority from model redesign to dissertation structure and writing.
  对照模型 0 审视扁平与深度多模态结果；明确结果呈现方式；重心从改模型转向 dissertation 结构与写作。

### Supervisor feedback

- **Stop major redesign.** Enough modelling done; write and present clearly.
  **停止大规模改造。** 建模已足够；应清晰写作与呈现。
- **Model 0 is essential.** Null / near-null flat results are acceptable and informative.
  **模型 0 必不可少。** 扁平侧无效或近无效结果可接受、有信息量。
- **Three result blocks:** flat vs M0; deep vs M0; flat vs deep.
  **三块结果：** 扁平相对 M0；深度相对 M0；扁平相对深度。
- **Shipping > remote sensing** in current evidence; deep M3/M4 only a **small** gain over M0.
  当前证据下航运强于遥感；深度 M3/M4 相对 M0 仅有**小幅**改善。
- **Explain every deep encoder** (inputs, outputs, architecture, why that modality).
  **解释每个深度编码器**（输入、输出、架构、为何适合该模态）。
- **Clarify CW / all table columns** so a non-specialist can read them.
  **澄清 CW 与所有表列**，使非专业读者可读。
- **Interpretability only for models that beat M0** — prioritise M3_Deep_gated; optionally M4_Deep_gated.
  **仅对击败 M0 的模型做可解释性**——优先深度 M3；可选深度 M4。
- **Deliverables within ~1 week:** dissertation structure (sections + bullets) + literature-review draft for formal feedback.
  **约一周内交付：** dissertation 结构草稿 + 文献综述初稿供正式反馈。
- Next meeting: **Wed 29 July 2026** (last before submission).
  下次会议：**2026-07-29（提交前最后一次）**。

### Decisions made

- Phase 04 officially starts: writing-first, no further major architecture changes unless error-fix.
  Phase 04 正式开始：写作优先；除非纠错，不再大规模改架构。
- Keep M0 central; present flat + deep; claim shipping value cautiously.
  以 M0 为叙事中心；同时报告扁平与深度；谨慎表述航运价值。
- SHAP / gates / attention only for deep models that improve on M0.
  仅对相对 M0 有改善的深度模型做 SHAP / 门控 / 注意力。
- Outline + lit-review to Beatrice before Meeting 05.
  Meeting 05 前向 Beatrice 发送大纲与文献综述。

### Next tasks

- [ ] Draft full dissertation outline (EN/CN) and send to Beatrice.
  起草完整大纲（中英）并发送给 Beatrice。
- [ ] Revise / send literature-review draft (add citations to summary tables).
  修订并发送文献综述初稿（汇总表补引用）。
- [ ] Standardise Flat/Deep naming; define RMSE / skill / CW / DM in Methods.
  统一 Flat/Deep 命名；在方法章定义 RMSE / skill / CW / DM。
- [ ] Write Results around the three agreed blocks; Methods explain TCN / EO / GAT.
  按三块写结果；方法章解释 TCN / EO / GAT。
- [ ] Interpretability for M3_Deep_gated (gates + node attention); optional M4.
  对深度 M3 做可解释性（门控 + 节点注意力）；可选 M4。
- [ ] Prepare Appendices A–C (dictionaries, robustness, locked configs).
  准备附录 A–C（词典、稳健性、锁定配置）。

---

## 2026-07-15（前后）写作与结果整理

### What I did

- Consolidated research plan / progress overview (`2026-07-15_研究方案与进度总览.md`) and project logic overview for writing.
  整理研究方案与进度总览、项目逻辑总览，供写作引用。
- Advanced `06_writing/outline.md` / `outline_CN.md`：Ch1–6 bullets aligned to RQ1–RQ3；Results tables for Flat / Deep / Flat-vs-Deep；Appendix A/B/C placeholders.
  推进大纲：第 1–6 章要点对齐 RQ；结果主表；附录条目。
- Literature-review drafts refreshed under `06_writing/Chapter 2 Literature Review/` (`20260716_literature_review_*`).
  更新文献综述草稿（供发给 Beatrice）。
- Deep interpretability for main M3 gated arm (`run_deep_interpret_m3.py` → `deep_m3_interpret.png` / gate CSVs).
  主设定深度 M3 门控可解释性脚本与图。

### Decisions made

- Appendices stay as **supporting material** for the final dissertation; **do not** send A/B/C as a standalone pack to Beatrice now — she asked for **outline + lit review**.
  附录保留为最终论文支撑材料；**现在不必**单独把 A/B/C 发给 Beatrice——她要的是**大纲 + 文献综述**。
- Outline Appendices lines are enough for her structure check.
  大纲里的附录条目足够她检查结构。

### Next tasks

- Send outline + literature review to Beatrice (priority).
  优先发送大纲与文献综述。
- Flesh Appendices A–C from scattered dictionaries / outputs into citable drafts.
  把散落词典/产物汇编成可引用附录草稿。

---

## 2026-07-16

### What I did

**附录缺口核查与补齐（对照 outline Appendices A–B–C）**

对照大纲附录三项，核查「实验侧是否齐 / 写作侧是否成稿」：

| 附录 | 原先状态 | 本日动作 |
| --- | --- | --- |
| **A** 变量词典 · AOI/咽喉 · 滞后 · 图边 | 材料散落在各 `*_data_dictionary.md` / 脚本；**无一页总表**；GFW Flat +4w vs Deep event +2w 易混写 | 汇编成稿 |
| **B** 稳健性（回看、LOAO、LOMO、种子、早/晚…） | 产物多在 `05_outputs/`；**早/晚仅 Deep M4**；大纲未列 LOCHO / 水体掩膜 / 融合矩阵 | 子期通用脚本 + 成稿 + 大纲增列 |
| **C** 超参网格 · 锁定 Deep · 软件路径 | 设定在代码默认值；**无 `requirements.txt`** | 环境文件 + 锁定表成稿 |

**附录 A — `06_writing/appendix_A_data.md`**

- 完整变量词典（M1 31 / M2 55 / M3 flat 113 / Deep 图节点特征）。
- 11 AOI（P001–P011）+ 6 咽喉列表（坐标、类型、关联咽喉）。
- **发布滞后总表**：Flat / Deep **分栏**；脚注写清 **GFW 扁平 +4w（月频 presence）≠ 深度 +2w（事件/航次 O-D）**。
- 航运图边定义：动态 O-D、静态 AOI↔咽喉、邻接对称化+自环、`log1p(flow)×edge_scale`。
- （旧单独文件 `appendix_A_lags.md` 已并入后删除，避免重复。）

**附录 B — `06_writing/appendix_B_robustness.md` + 子期产物**

- 新建 **`04_code/scripts/tools/subperiod_eval.py`**（**不覆盖** `run_deep_advanced.py`）：复用相同 `SPLIT=2023-01-01` 与 CW/DM 口径，离线对 Flat M1–M4 + Deep 预测 CSV 切 early/late。
- 产出 `05_outputs/baselines/subperiod/subperiod_summary.csv`；数字与 `deep_advanced_summary` 中 M4 子期一致（如 M4_Deep_gated 早 −2.35% / 晚 +0.49%）。
- **关键发现写入 B.1：** **M3_Deep_gated 是唯一早、晚两期 skill 皆为正的配置**（+0.09 / +0.14）→ 支撑「gated finship 最稳主模型」。
- B 另汇编：融合矩阵 3×3、LOAO、**LOCHO**、水体掩膜、LOMO、多 seed / lookback·d 扫描。
- `outline.md` / `outline_CN.md` 附录 B 条目增列：LOCHO、水体掩膜、融合矩阵。

**附录 C — `06_writing/appendix_C_config.md` + `04_code/requirements.txt`**

- `requirements.txt`：核心锁定依赖（Python 3.9.6；numpy / pandas / scipy / sklearn / xgboost / torch / matplotlib / shap）；注明 **Prithvi 嵌入离线预计算，训练不依赖 transformers**。
- 锁定设定表：共享协议（lb=4, min_train=104, retrain=13, val=52, seed=42）；Flat Ridge/XGB 网格；Deep d=32、GAT 2 层 / heads=4、TCN 2、gated 主报、Adam lr/wd/epochs/patience；入口脚本与输出路径。

### Decisions made

- 附录 A/B/C **写作侧已成稿**，可被正文 §3.4 / §4.5 / §3.8 引用；发给 Taylor 时仍以 **outline + 文献综述** 为主，附录不必单独邮件。
  Appendices A–C drafted for citation; Taylor email still = outline + lit review only.
- 早/晚子期用离线切片即可，**不必重训**。
  Early/late via offline slice; no retrain.
- 主叙事继续锁定：Flat 无模型击败 M0；航运相对 M1 有增量；Deep M3_gated 小幅正 skill 且跨子期最稳。
  Main narrative unchanged; M3_Deep_gated remains the stable positive-skill specification.

### Issues / blockers

| 问题 | 影响 | 计划 |
|---|---|---|
| outline / 文献综述尚未发出 | Meeting 04 行动项未完成 | 本周内发给 Beatrice |
| 附录 A 逐变量文献来源仍指向各模态词典 | 附录 A 保持精简；全文 citation 在数据字典 | 正文引用附录即可，不重复粘贴 |
| 完整 dissertation 初稿未齐 | Meeting 05 前需推进 Methods/Results | 按 outline 扩写第 3–4 章 |

### Next tasks

- [ ] 发送 `outline.md`（或精简版）+ 文献综述草稿给 Beatrice。
- [ ] 按大纲扩写 Methods（编码器说明、CW/DM 定义）与 Results 三块叙事。
- [ ] 确认 RQ3 图（`deep_m3_interpret.png`）进正文 §4.6。
- [ ] Meeting 05（2026-07-29）材料：结构反馈落实 + 正文草稿进度。

---

## Phase 04 status snapshot

> **Current phase:** Phase 04 — dissertation writing & submission prep  
> **Next meeting:** Meeting 05 — Wed **2026-07-29** (last before submission)  
> **Priority deliverables to Beatrice now:** outline + literature review（**not** the appendix pack alone）

### Appendices pack（2026-07-16）

| File | Role |
| --- | --- |
| `06_writing/appendix_A_data.md` | 变量词典 · AOI/咽喉 · 滞后总表 · 图边 |
| `06_writing/appendix_B_robustness.md` | 稳健性汇编（含早/晚、LOCHO、水掩膜、融合矩阵） |
| `05_outputs/baselines/subperiod/subperiod_summary.csv` | Flat+Deep 子期 skill 表 |
| `04_code/scripts/tools/subperiod_eval.py` | 通用早/晚评分脚本 |
| `06_writing/appendix_C_config.md` | 超参网格 · 锁定 Deep · 路径 |
| `04_code/requirements.txt` | 核心软件环境 |

### Completed in Phase 04 so far

- [x] Meeting 04 笔记结构化（`Meeting04_Note 20260708.md`）
- [x] Outline EN/CN 主骨架 + 附录条目更新
- [x] 附录 A/B/C 成稿 + requirements + 子期脚本/表
- [ ] 大纲 + 文献综述发出给 Beatrice
- [ ] Methods / Results 正文扩写
- [ ] Meeting 05 材料
