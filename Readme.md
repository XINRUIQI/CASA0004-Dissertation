# Thesis Project — CASA0004 Dissertation

## 论文题目

暂定：

## 研究问题

RQ1:
RQ2:
RQ3:

## 核心论点

本文认为：

## 项目结构

```
├── 00_admin/              — 行政管理
│   ├── meeting_notes/     — 导师会议记录（每次一个 .md）
│   ├── supervisor_feedback/ — 导师反馈汇总 + 待办清单
│   ├── research_diary.md  — 每日研究日志
│   ├── task_list.md       — 任务清单
│   └── timeline.md        — 项目时间线
│
├── 01_literature/         — 文献管理
│   ├── papers_pdf/        — PDF 原文（不上传 Git）
│   ├── reading_notes/     — 每篇文献一个 .md 阅读笔记
│   ├── literature_matrix.xlsx — 文献矩阵总表
│   └── references.bib     — Zotero 导出的引用库
│
├── 02_ai_conversations/   — AI 对话记录
│   ├── chatgpt_exports/   — ChatGPT 等工具导出
│   ├── useful_outputs/    — 有价值的 AI 输出
│   └── prompt_log.md      — AI 使用日志（最重要）
│
├── 03_data/               — 数据管理
│   ├── raw/               — 原始数据（不修改）
│   │   ├── oil_prices/
│   │   ├── ais/
│   │   ├── satellite/
│   │   └── official_reports/
│   ├── processed/         — 清洗后的数据
│   ├── external_sources.md — 外部数据源清单
│   └── data_dictionary.xlsx — 数据字典
│
├── 04_code/               — 代码
│   ├── notebooks/         — Jupyter notebooks
│   ├── scripts/           — Python 脚本
│   ├── src/               — 可复用模块
│   └── README.md
│
├── 05_outputs/            — 输出成果
│   ├── figures/
│   ├── tables/
│   ├── maps/
│   └── model_results/
│
├── 06_writing/            — 论文写作（Markdown 先写，后转 Word）
│   ├── outline.md
│   ├── chapter_1_introduction.md
│   ├── chapter_2_literature_review.md
│   ├── chapter_3_methodology.md
│   ├── chapter_4_results.md
│   ├── chapter_5_discussion.md
│   └── references_notes.md
│
├── 07_submission/         — 最终提交
│   ├── final_pdf/
│   ├── appendix/
│   └── reproducibility_pack/
│
├── .gitignore
└── Readme.md
```

## 文件命名规则

```
YYYY-MM-DD_topic_version.ext
```

示例：
- `2026-05-22_literature_search_v01.md`
- `2026-05-23_dataset_inventory_v02.xlsx`
- `2026-06-01_supervisor_feedback_meeting2.md`

## 文献管理：三层结构

1. **PDF 原文** → `01_literature/papers_pdf/AuthorYear_shorttitle.pdf`
2. **阅读笔记** → `01_literature/reading_notes/Author_Year.md`
3. **文献矩阵** → `01_literature/literature_matrix.xlsx`

## 数据管理原则

- `raw/` = 原始数据，**不修改**
- `processed/` = 清洗后的数据
- 所有数据来源记录在 `external_sources.md`

## 每日工作流

1. 打开 Cursor workspace
2. 看 `task_list.md`
3. 更新当天 `research_diary.md`
4. 阅读文献，写 `reading_notes`
5. 把文献加入 `literature_matrix`
6. 有新数据源就写进 `external_sources.md`
7. 用 AI 后记录进 `prompt_log.md`
8. 把成熟内容转移到 `chapter_x.md`
9. 每天 Git commit 一次

## 章节结构

1. Introduction
2. Literature Review
3. Methodology
4. Results
5. Discussion
6. Conclusion

## 写作规则

- 不直接引用 AI 生成内容
- 所有事实必须有正式来源
- 使用英式英语 / 中文学术风格
- 每段必须有明确论点
- 文献引用使用 Zotero citekey
