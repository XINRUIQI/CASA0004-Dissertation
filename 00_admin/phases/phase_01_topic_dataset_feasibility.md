# Phase 01 — Topic Framing & Dataset Feasibility

**Triggered by:** Meeting 01 with Beatrice (2026-05-06)
**Target deadline:** Meeting 02 (2026-05-27)
**Status:** In progress

## Core question

我的题目能不能做？有没有开放数据？哪些数据可以支持 oil price / shipping / satellite / AIS 方向？

## Tasks

- [ ] 明确研究方向（oil price forecasting + spatial/shipping indicators）
- [ ] 检查 ShipRSImageNet 是否有 oil tanker 类别
- [ ] 检查 ShipRSImageNet 是否有 timestamp
- [ ] 搜索 open-access AIS / vessel tracking / shipping activity 数据集
- [ ] 搜索 port activity / satellite imagery 数据集
- [ ] 搜索 oil price 数据集（Brent, WTI）
- [ ] 建立 dataset inventory table → `03_data/external_sources.md`
- [ ] 开始读核心文献（至少 5 篇 oil forecasting）
- [ ] 准备 Meeting 02 的汇报材料和问题

## Expected outputs

| Output | Location | Status |
|--------|----------|--------|
| Dataset inventory v01 | `03_data/external_sources.md` | To do |
| Literature matrix v01 | `01_literature/literature_matrix.xlsx` | To do |
| ShipRSImageNet feasibility note | `03_data/raw/satellite/` | To do |
| Initial research questions | `06_writing/chapter_1_introduction.md` | To do |
| Meeting 01 notes | `00_admin/meeting_notes/2026-05-06_meeting_01_beatrice.md` | Done |

## Supervisor feedback driving this phase

> "I think might be a good goal: have an idea of exactly what's available in each piece of data — how many years of data, how regular is the timestamp, is it global or specific regions?"
>
> "Have a table: dataset name, URL, who maintains it, timestamp, whatever information you think is helpful."
>
> — Beatrice, Meeting 01

## Decisions made during this phase

- The project will begin by exploring open-access datasets.
- Dataset feasibility first, modelling design second.
- Literature review should cover both oil forecasting and spatial/remote-sensing indicators.

## Reflection (fill in at end of phase)

<!-- What worked? What didn't? What changed? -->
