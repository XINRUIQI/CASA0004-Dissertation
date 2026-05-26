# Research Diary

---

## 2026-05-06

### What I did
- First meeting with supervisor Beatrice Taylor.
  与导师 Beatrice Taylor 进行了第一次会议。
- Discussed data access challenges.
  讨论了数据获取方面的困难。
- Beatrice suggested looking into ShipRSImageNet.
  Beatrice 建议查看 ShipRSImageNet 数据集。

### Next tasks
- Check ShipRSImageNet categories and timestamps.
  检查 ShipRSImageNet 的类别和时间戳。
- **Build dataset inventory table.**  
  建立数据集清单表。
- Start literature review on oil price forecasting and remote sensing indicators.
  开始阅读原油价格预测和遥感指标相关文献。

---

## 2026-05-22

### What I did
- Reorganised dissertation project folder structure.  
  重新整理了论文项目文件夹结构。
- Assigned priority levels to the references.
  给参考文献标好了等级。**
- **Completed the first two categories of the dataset table: 1) Market & Financial Data and 2) Official Reports & Structured Market Text.
  整理完 dataset table 的前两类：1) Market & Financial Data 和 2) Official Reports & Structured Market Text。

### Decisions made
- **In 1) Market & Financial Data, EIA Short-Term Energy Outlook (STEO) is monthly, so it will not be used for now.**  
  在 1) Market & Financial Data 里，EIA Short-Term Energy Outlook (STEO) 的时间尺度为月度，因此暂时不用。
- In 1) Market & Financial Data, ICE and CME require paid subscriptions, so they were removed.**
  在 1) Market & Financial Data 里，ICE 和 CME 需要付费订阅，因此删掉。
- **For 1D. Macro-financial control variables, it is still uncertain whether USD-related variables are needed.**  
  对于 1D. Macro-financial control variables，目前不确定是否需要美元相关变量，待定。

---

## 2026-05-25

### What I did

**Data acquisition (bulk download day)  数据采集（集中下载日）**

- Downloaded 241 EIA STEO monthly PDF reports (2006–2025) via `download_steo_archive.py`.  
  通过脚本批量下载了 241 份 EIA STEO 月度 PDF 报告（2006–2025）。
- Queried GDELT via BigQuery: obtained oil-disruption daily features and transport-disruption event records (2019-01 – 2026-05, 83 CSV shards).
  通过 BigQuery 查询 GDELT：获取了石油扰动事件每日特征和运输中断事件记录（2019-01 至 2026-05，83 个 CSV 分片）。
- Scraped Aramco press release links and full-text content using Playwright.  
  使用 Playwright 爬取了 Aramco 新闻发布链接及正文内容。——没爬到几个
- Downloaded OGIM (Global Oil & Gas Infrastructure Mapper) core + extended node data and Global Oil & Gas Extraction Tracker.  
  下载了 OGIM 核心/扩展基础设施节点数据及 Global Oil & Gas Extraction Tracker。
- Downloaded NGA World Port Index (2026) and EMODnet vessel density monthly data (2017–2025).  
  下载了 NGA 世界港口索引（2026）及 EMODnet 月度船舶密度数据（2017–2025）。
- Downloaded S&P 500 daily data from Yahoo Finance as macro-financial control variable.  
  从 Yahoo Finance 下载了 S&P 500 日频数据作为宏观金融控制变量。

**Remote sensing pipeline  遥感数据流水线**

- Wrote 3 Google Earth Engine scripts: Sentinel-2 monthly indices, Landsat backfill monthly indices, and VIIRS monthly nightlights — all for 8 oil-infrastructure AOIs.  
  编写了 3 个 GEE 脚本：Sentinel-2 月度指数、Landsat 回填月度指数、VIIRS 月度夜间灯光——覆盖 8 个石油基础设施 AOI。
- Exported and downloaded CSV outputs; built `aggregate_remote_sensing_to_weekly.py` to resample to weekly frequency.  
  导出并下载了 CSV；编写 `aggregate_remote_sensing_to_weekly.py` 重采样至周频。

**Shipping data pipeline  航运数据流水线**

- Built `download_portwatch_chokepoints.py`: fetches IMF PortWatch daily transit data for 6 oil-critical chokepoints (Hormuz, Suez, Malacca, Panama, Bab el-Mandeb, Cape of Good Hope) from 2019 onward.  
  编写 `download_portwatch_chokepoints.py`：从 IMF PortWatch 获取 6 条关键石油咽喉航道日频过境数据（2019 年起）。
- Built `download_gfw_vessel_presence.py`: fetches GFW 4Wings API monthly vessel presence hours for the same 6 chokepoints (2012–2018), filling the pre-PortWatch gap.  
  编写 `download_gfw_vessel_presence.py`：通过 GFW 4Wings API 获取 2012–2018 年同 6 条航道的月度船舶在场时数，填补 PortWatch 之前的空白。
- Built `aggregate_shipping_to_weekly.py`: merges PortWatch daily + GFW monthly into a single wide-format weekly shipping feature table.  
  编写 `aggregate_shipping_to_weekly.py`：将 PortWatch 日频 + GFW 月频合并为统一的宽格式周频航运特征表。

**Feature matrix & analysis  特征矩阵与分析**

- Built `build_feature_matrix.py` to combine all processed weekly features (market, macro, text/GDELT, remote sensing, shipping) into a unified feature matrix.  
  编写 `build_feature_matrix.py`，将所有已处理的周频特征（市场、宏观、文本/GDELT、遥感、航运）合并为统一特征矩阵。


### Decisions made

- **Use GFW vessel presence (2012–2018) + PortWatch (2019–2026) as a combined shipping proxy to cover the full study period.**  
  用 GFW 船舶在场数据（2012–2018）+ PortWatch（2019–2026）组合覆盖整个研究期间的航运代理变量。
- **Defined 8 oil-infrastructure AOIs for remote sensing extraction** (to be documented in `aoi_oil_infrastructure.csv`).  
  为遥感提取定义了 8 个石油基础设施 AOI（记录在 `aoi_oil_infrastructure.csv` 中）。
- **Weekly (Friday-ending) as the unified temporal resolution** for all feature modalities.  
  以每周五截止的周频作为所有特征模态的统一时间分辨率。

### Next tasks
- Prepare progress update and questions for Meeting 02 (27 May).  
  为第二次导师会议（5 月 27 日）准备进度更新和问题。
- Review quality of GDELT features — check whether disruption event counts are noisy.
  审查 GDELT 特征质量——检查扰动事件计数是否噪声过大。
- Verify remote sensing CSV outputs (check for missing months / AOIs).  
  验证遥感 CSV 输出（检查是否有缺失月份/AOI）。






  # Task List

> **Current phase:** Phase 01 — Topic framing & dataset feasibility
> **Details:** `00_admin/phases/phase_01_topic_dataset_feasibility.md`

## Current Sprint (before Meeting 02 — 27 May)

- [ ] Check ShipRSImageNet categories (e.g. whether it includes oil tankers)
- [ ] Check whether ShipRSImageNet includes timestamps
- [ ] Search for open AIS / vessel tracking / shipping activity datasets
- [ ] Build initial dataset inventory table in `03_data/external_sources.md`
- [ ] Read 5 core oil forecasting papers
- [ ] Read literature on satellite-based ship detection
- [ ] Prepare progress update and questions for next meeting

## Backlog

- [ ] Populate `01_literature/literature_matrix.xlsx`
- [ ] Write reading notes for each core paper
- [ ] 搭建分析 pipeline
- [ ] 撰写 Introduction 初稿
- [ ] 撰写 Literature Review 初稿
- [ ] 撰写 Methodology 初稿

## Completed

- [x] Set up project folder structure
- [x] Create research diary, prompt log, meeting note templates
- [x] Reorganise meeting 01 notes into structured format
