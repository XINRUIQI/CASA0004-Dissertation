# CASA0004 Dissertation 项目结构总览

**项目根目录：** `/Users/shirly/CASA/casa0004 Dissertation/`  
**组织逻辑：** 按论文工作流编号 `00–07`，从行政管理 → 文献 → 数据 → 代码 → 输出 → 写作 → 提交。  
**模态逻辑：** 数据与特征按 **M1（金融）/ M2（遥感）/ M3（航运）** 分层；消融实验 M0–M4 见 `04_code/` 与 `05_outputs/baselines/`。  
**最后更新：** 2026-07-28（承接 07-07 版：路径改为 `最新待整理/`；写作目录、Meeting 04、RQ3 产物、Phase 04 日记已对齐现状）

---

## 顶层结构

```text
casa0004 Dissertation/
├── .gitignore
├── Readme.md                          # 项目说明、目录规范、工作流
│
├── 00_admin/                          # 行政管理（日记、会议、进度文档）
├── 01_literature/                     # 文献矩阵 + 阅读笔记 + PDF（本地）
├── 02_ai_conversations/               # AI 对话记录（Phase 1–3 + Mini_Conference）
├── 03_data/                           # 数据（Dataset + processed 按模态 + raw）
├── 04_code/                           # 回测框架 + Flat/Deep 脚本 + 编码器
├── 05_outputs/                        # 基准实验输出（baselines/Flat + Deep + subperiod）
├── 06_writing/                        # 论文章节、Outline、Appendix、Thesis structure
└── 07_submission/                     # 最终提交（占位）
```

---

## `00_admin/` — 行政管理

```text
00_admin/
├── research_diary_phase1.md … phase3.md   # 选题→建模日志
├── research_diary_phase4.md               # Meeting 04 起：写作主线日志
├── timeline.md
├── meeting_notes/
│   ├── Meeting01–03_*                     # 早期会议笔记 / 邮件 / txt
│   ├── Meeting04_Note 20260708.md         # Meeting 04 正式笔记
│   ├── Meeting04_Mail / Meeting04_20260708.txt
│   ├── 20260716_*_BT_feedback.pdf         # Taylor 结构 / 文献反馈
│   └── …
├── 最新待整理/                              # ★ 当前活跃方案与结果文档（原「待整理」已迁此）
│   ├── 2026-06-22_research_plan_e2e_multimodal.md   # 方法设计蓝图
│   ├── 2026-07-15_研究方案与进度总览.md              # ★ 进度主索引（写作判断 07-28 已刷新）
│   ├── 2026-07-07_研究方案与进度总览.md              # 历史快照（指向 07-15）
│   ├── 2026-07-07_研究方案与进度总览 copy.md         # 过时副本，可归档
│   ├── 2026-07-28_扁平模型变量清单.md                # ★ 扁平变量最新版
│   ├── 2026-07-07_扁平模型变量清单.md                # 历史快照
│   ├── 2026-07-07_File Structure.md                 # 本文件
│   ├── 项目逻辑与结果总览_CN.md                      # 结果 synthesis
│   ├── flat_baseline_log.md                         # 扁平基线实验主记录
│   ├── flat_baseline_full_walkthrough_{CN,EN}.md
│   ├── deep_model_full_walkthrough_{CN,EN}.md
│   └── flat_model_{CN,EN}.md / *.html / copy*
└── Inactive/                          # 归档：旧 File Structure、Meeting04 prep、旧进度总览等
```

**说明：** 活跃文档在 `最新待整理/`；Meeting 04 prep 等已迁入 `Inactive/`。进度请以 **`2026-07-15_研究方案与进度总览.md`** 为准。

---

## `01_literature/` — 文献

```text
01_literature/
├── literatue.md / literature_matrix.md
├── literature_matrix_P001_P101.xlsx
├── literature_matrix_P001_P115.xlsx      # 扩展矩阵
├── papers_pdf/                          # PDF 本地（gitignore），按主题 01–10 分夹
├── reading_notes/                       # 01 Finance … 10 时序架构 + Template
└── 往年Distinction论文/                 # 样例论文 PDF（含 MRes And Proposal/）
```

**Readme 中规划但尚未出现：** `references.bib`（可交稿前从矩阵导出）。

---

## `02_ai_conversations/` — AI 记录

```text
02_ai_conversations/
├── prompt_log.md
├── Phase 1/ … Phase 3/
└── Mini_Conference/
```

---

## `03_data/` — 数据

```text
03_data/
├── Dataset/                             # notebooks + external_sources.md
├── processed/
│   ├── M1/  py/build_m1_weekly.py → outputs/m1_weekly_features.csv（~36 列含日期与 avail）
│   ├── M2/  build_m2_weekly + audit/eda + Channel B / watermask 产物
│   ├── M3/  aggregate_shipping_to_weekly.py
│   │        build_m3_graph17.py → m3_graph17_tensors.npz（深度航运图）
│   │        + EMODnet / darkvessel / edges / nodes 等
│   └── merge/ build_feature_matrix.py
│              → weekly_feature_matrix.csv          # ★ 365×213（标准窗）
│              → weekly_feature_matrix_full.csv     # 长历史 ~1067 周
│              → *_watermask.csv / *_dictionary.csv
└── raw/                                 # gitignore：00_spatial / 01_market / 02_sentinel2 / 03_shipping
```

**合并矩阵口径：** 365 周 × 213 列 = `week_ending_friday` + 212 数据列（31+55+113+11 avail+2 target）。详见 `2026-07-28_扁平模型变量清单.md`。

**M3 滞后（勿混写）：** 扁平 GFW presence **+4w**、PortWatch **+1w**；深度图 GFW event/O-D **+2w**、SAR **+4w**。

---

## `04_code/` — 建模代码

```text
04_code/
├── scripts/
│   ├── flat/
│   │   ├── run_baseline.py
│   │   └── M{1..4}_Flat/{sweep,robustness,shap}_m*.py
│   ├── deep/
│   │   ├── run_deep_baseline.py          # 主入口（表示级融合）
│   │   ├── run_deep_{sweep,interpret,advanced,fusion_matrix,xattn_viz}.py
│   │   ├── run_deep_interpret_m3.py
│   │   └── diagnose_* / compare_rs_anom / multiseed_rs_anom.py
│   └── tools/  subperiod_eval / migrate_model_names / relocate_deep_outputs
├── src/
│   ├── backtest/                         # Flat 滚动回测
│   ├── models/                           # finance/rs/shipping encoders + fusion + deep_*
│   └── model_naming.py
└── notebooks/                            # 占位
```

---

## `05_outputs/` — 分析输出

```text
05_outputs/baselines/
├── Flat/M{1..4}_Flat/                    # metrics / predictions / SHAP / 稳健性图
├── Deep/
│   ├── M{1..4}_Deep/                     # 分模态 baseline + interpret
│   │   └── M4_Deep/ 含 deep_gate_stability.csv 等 RQ3 多 seed 产物（2026-07-16）
│   └── _cross/                           # deep_metrics / cw / sweep / fusion_matrix …
└── subperiod/                            # 子期评估
```

> macOS 上 `Deep/` 与 `deep/` 可能指向同一目录（大小写不敏感）。

---

## `06_writing/` — 论文写作（2026-07-28）

```text
06_writing/
├── Outline/
│   ├── 20260728_outline_brief.md         # ★ 当前结构纲要（Taylor 反馈修订版）
│   ├── outline.md / outline_brief.md / outline_bilingual.md
│   └── outline copy.md                   # 旧稿
├── Chapter 2  Literature Review/
│   ├── 20260728_literature_review_双语.md # ★ Ch2 最新双语稿
│   ├── 20260716_literature_review_EN.md
│   ├── 20260716_literature_review_bilingual无P序号.md
│   ├── literature_review_bilingual.md / .html / .pdf
│   └── 20260703_literature_review.md
├── chapter_1_introduction.md + _bilingual.md
├── chapter_3_methodology.md + _bilingual.md
├── chapter_4_results.md + _bilingual.md
├── chapter_5_discussion.md + _bilingual.md
├── chapter_6_conclusion_bilingual.md
├── Appendix/
│   ├── appendix_A_data.md
│   ├── appendix_B_robustness.md
│   └── appendix_C_config.md
├── Thesis structure.docx
└── references_notes.md
```

**写作状态（摘要）：** Meeting 04（07-08）后进入 Phase 04；07-16 送审 Ch2/结构；07-28 按反馈修订 outline + Ch2。当前主线 = 按 `20260728_outline_brief.md` **合稿润色**（非从零起稿）。详见进度总览 §3.2。

---

## `07_submission/` — 提交占位

```text
07_submission/
├── appendix/.gitkeep
├── final_pdf/.gitkeep
└── reproducibility_pack/.gitkeep
```

---

## 数据流与组织要点

```text
03_data/raw/{01_market,02_sentinel2,03_shipping}
        ↓
03_data/processed/{M1,M2,M3,merge}
        ↓
04_code/scripts/{flat,deep}  ──→  05_outputs/baselines/{Flat,Deep,subperiod}
        ↓
06_writing（Outline → 各章双语稿 → 07_submission）
        ↑
00_admin/最新待整理 + 01_literature
```

| 主题 | 内容 |
| --- | --- |
| 研究焦点 | 周频 Brent；三模态；扁平融合 vs 模态感知表示级融合 |
| 消融 | M0→M4；Flat Ridge/XGB + Deep 编码器/门控/交叉注意力 |
| 比较窗 | 2019–2025 标准窗（365 周）；lookback=4；测试约 257 周 |
| 文档入口 | 进度：`2026-07-15_研究方案与进度总览.md`；变量：`2026-07-28_扁平模型变量清单.md`；逻辑：`项目逻辑与结果总览_CN.md` |
| Git | raw / papers_pdf / 大 tif·zip·parquet 不入库 |

---

## 文件数量汇总（约，2026-07-28）

| 目录 | 约略 | 状态 |
| --- | --- | --- |
| `03_data/raw/` | ~1100+ | 本地，gitignore（含 S2 patches） |
| `03_data/processed/` | ~70 | tracked（含 graph17 等） |
| `04_code/` | ~38 `.py` | tracked |
| `05_outputs/baselines/` | ~110+ | tracked（Flat+Deep+RQ3） |
| `06_writing/` | 章节 + Outline + Appendix | 写作主线活跃 |
| `00_admin/最新待整理/` | 方案/结果文档集 | 活跃索引 |
| `07_submission/` | 占位 | 待 Meeting 05 后填充 |
