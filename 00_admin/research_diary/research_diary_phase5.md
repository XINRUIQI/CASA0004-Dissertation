# Research Diary — Phase 05

> **Phase 05:** Writing & revision  
> **Trigger:** Meeting 04 (2026-07-08) / modelling complete  
> **Deadline:** dissertation submission (August 2026)  
> **CASA appendix:** `06_writing/CASA-MSc-thesis-main/11-appendix-D.Rmd` · `tables/research_log.xlsx`  
> Handbook §3.7: the research log **must** appear as an appendix, with supervisory meeting dates and at least one sentence on what was discussed.

Modelling notes remain in `research_diary_phase3.md` and `research_diary_phase4.md`. This file covers write-up, supervisor feedback, and the submission research log.

---

## 2026-07-08

### What I did

- Fourth meeting with Beatrice Taylor (see phase 4 diary for the full note).
  第四次导师会议（详细记录见 phase 4）。
- Priority shifted from model redesign to dissertation structure and writing.
  重心从改模型转为结构与写作。

### Next tasks

- Outline + literature-review draft for Beatrice.
- Results in three blocks: Flat vs M0; Deep vs M0; Flat vs Deep.

---

## 2026-07-16

### What I did

- Sent / prepared outline and literature-review draft.
  准备大纲与文献综述草稿。
- Compiled Appendices A–C (data dictionary, robustness, locked settings).
  汇编附录 A–C。
- Received written feedback from Beatrice on structure and the literature review
  (`20260716_draft_structure_BT_feedback.pdf`, `20260716_literature_review_BT_feedback.pdf`).
  收到结构与文献综述的书面反馈。

---

## 2026-07-28

### What I did

- Revised Chapter 1 to the CASA / Andy opening sequence: why it matters → what was done → how → results.
  按 CASA / Andy 开篇顺序改第 1 章。

---

## 2026-07-29

### What I did

- Fifth and final supervision meeting with Beatrice Taylor.
  第五次、也是提交前最后一次正式会议。
- Discussed draft deadline (**5 August**), sentence-level clarity, policy as broader geopolitical context, a choke-point / study-site map, aiming nearer 10,000 than 12,000 words, story versus methods, ethics statement, and the Flat/Deep remote-sensing mismatch (tabular indices vs frozen Prithvi), which should be acknowledged rather than rebuilt.
  讨论草稿截止、句式、政策语境、咽喉地图、字数、叙事与方法、伦理声明，以及扁平/深度遥感输入不一致——应写明而非重做。

---

## 2026-08-03

### What I did

- Submitted a full draft.
  提交全文草稿。
- Received written comments (`20260803_dissertation_draft_BTfeedback.pdf`).
  收到全文书面反馈。

---

## 2026-08-11 – 2026-08-14

### What I did

- Revised Chapters 3–4 (methodology and results).
  修订第 3–4 章。
- Rebuilt the figure set: study-site map, Jurong worked example, Flat/Deep RMSE comparisons, node SHAP map and heatmap, Appendix B seed-robustness figure.
  重做插图系统。
- Aligned Bookdown numbering, Harvard CSL, and `references.bib`.
  对齐 Bookdown 图号、哈佛体例与参考文献库。

---

## 2026-08-17

### What I did

- Compiled the **CASA research log** as **Appendix D**, following `CASA Thesis.Rmd` (`Date` / `Task` table plus a dedicated meetings table for handbook §3.7).
  按 CASA 模板写成 **附录 D 研究日志**；督导会面单独成表，满足手册“必须附录 + 会面日期 + 至少一句讨论内容”。
- Files:
  - `06_writing/CASA-MSc-thesis-main/11-appendix-D.Rmd` (English submission copy)
  - `06_writing/Chapter Appendix/appendix_D_research_log.md` (bilingual)
  - `06_writing/CASA-MSc-thesis-main/tables/research_log.xlsx`
  - `_bookdown.yml` updated to include Appendix D
  - Meeting 2 date corrected to **27 May 2026**; Chapter 3 main text left unchanged.

### Decisions made

- Keep existing Appendices A–C (data / robustness / config). The handbook research log is **Appendix D**, not a replacement for Appendix A.
  数据词典仍为附录 A；研究日志作为附录 D，不挤掉现有附录。
- Five supervisory meetings only in D.1; 16 July and 3 August written comments are noted in prose, not listed as extra meetings.
  D.1 只列五次正式会面；两次书面反馈写在段里，不充会面。

---

### Inventory: 六个章节正文中实际引用的文献（共 41 条）

#### 第 1 章 Introduction（2 条）

1. U.S. Energy Information Administration (2014)
2. Wittner (2020)

#### 第 2 章 Literature Review（34 条）

**2.1 油价驱动与预测基准**

1. Kilian (2009)
2. Alquist, Kilian and Vigfusson (2013)
3. Baumeister and Kilian (2015)

**2.2 机器学习与油价预测**

1. Costa et al. (2021)
2. Yılmaz and Zehir (2026)
3. Foroutan and Lahmiri (2024)
4. Simsek et al. (2024)
5. Zhao, Xue and Cheng (2023)

**2.3 航运作为油市信号**

1. Adland, Jia and Strandenes (2017)
2. Yan et al. (2020)
3. Arslanalp, Marini and Tumbarello (2019)
4. Arslanalp et al. (2026)
5. Mi et al. (2022)
6. Mi et al. (2023)
7. Paolo et al. (2024)

**2.4 海运网络与图模型**

1. Ouyang et al. (2022)
2. Liang et al. (2022)
3. Zhao et al. (2022)

**2.5 遥感作为油市信号**

1. Hao and Wang (2023)
2. Bricongne et al. (2026)
3. Wang et al. (2019)
4. Jung (2026)
5. Polinov, Bookman and Levin (2022)

**2.6 多模态学习与融合**

1. Baltrušaitis, Ahuja and Morency (2019)
2. Arevalo et al. (2017)
3. Gohari et al. (2024)
4. Cong et al. (2022) — SatMAE
5. Szwarcman et al. (2026) — Prithvi-EO-2.0
6. Fuller, Millard and Green (2023) — CROMA
7. Ma et al. (2022)
8. Neverova et al. (2016)
9. Che et al. (2018)
10. Shukla and Marlin (2021)
11. Jain and Wallace (2019)



#### 第 3 章 Methodology（5 条，第 2 章正文未出现）

1. Hoerl and Kennard (1970) — Ridge
2. Chen and Guestrin (2016) — XGBoost
3. Bai, Kolter and Koltun (2018) — TCN
4. Veličković et al. (2018) — GAT
5. Lundberg and Lee (2017) — SHAP



#### 第 4 章 Results

无新增正式引用（SHAP 沿用 Lundberg and Lee, 2017）。

#### 第 5 章 Discussion（6 条，均已在第 2 章出现）

- Alquist, Kilian and Vigfusson (2013)
- Adland, Jia and Strandenes (2017)
- Yan et al. (2020)
- Hao and Wang (2023)
- Arevalo et al. (2017)
- Gohari et al. (2024)



#### 第 6 章 Conclusion

无引用。

### Decisions made

- Full-draft References currently has **50** entries: all **41** in-text citations are present; **9** unused entries should be removed if Harvard “cited-only” is applied.
全文稿 References 现有 50 条：41 条正文引用均已收录；若按哈佛“仅收录正文引用”，应删除 9 条未引用文献。



### Unused in full-draft References（正文从未引用，9 条）

1. Aas, Jullum and Løland (2021)
2. Clark and West (2007)
3. Diebold and Mariano (1995)
4. Gneiting and Raftery (2007)
5. Gibson et al. (2021)
6. Patton (2011)
7. Pesaran and Timmermann (1992)
8. Small (2021)
9. Wu et al. (2019)

