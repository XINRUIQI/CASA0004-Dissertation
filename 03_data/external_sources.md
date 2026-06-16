# External Data Sources

> 记录所有外部数据来源、获取方式和使用许可。

## M1 Core Variables — Dataset Sources

> 与 `01_literature/beatrice_task_literature_matrix.md` §①「M1 推荐变量」**序号一致**，共 10 项；按 Kilian 三类机制（供给 / 全球需求 / 预防性需求 + 市场金融条件）组织。
> 全部对齐周五截止周频（W-FRI），研究期 2006-01 ~ 2025-12。所有来源均公开免费、无需 API key。
>
> **构建分工：**
>
> - **已有**（#1–2, #6 中 `vix`）：`03_data/processed/build_weekly_time_index.py` → `weekly_time_index.csv`
> - **待构建**（#3–5, #6 中 `ovx`, #7–10）：`03_data/processed/build_m1_to_build.py` → `m1_to_build_weekly.csv`，再经 `merge_m1_to_build.py` 合并入主表


| 序   | 变量                                              | 机制        | 数据集 / 指标                                        | 提供方                         | 标识符 (series/ticker)                                  | URL                                                                                                                                                                  | 原始频率 | → 周频处理                                                 | 构建                 | 覆盖起点                    |
| --- | ----------------------------------------------- | --------- | ----------------------------------------------- | --------------------------- | ---------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---- | ------------------------------------------------------ | ------------------ | ----------------------- |
| 1   | `brent_price` lags + `brent_log_return`         | 油价自身动态    | Europe Brent Spot FOB                           | EIA                         | 本地 `EIA_brent_spot_price_daily*.xls`                 | [https://www.eia.gov/petroleum/gasdiesel/](https://www.eia.gov/petroleum/gasdiesel/)                                                                                 | 日    | 周最后值；`brent_log_return = log(P_t / P_{t-1})`；滞后项建模阶段扩展 | 已有                 | 2006-01                 |
| 2   | `crude_stocks_change` 库存变化                      | 供给 / 市场平衡 | Commercial Crude Stocks (excl. SPR)             | EIA WPSR                    | 本地 `EIA_commercial_crude_stocks_weekly*.xls`         | [https://www.eia.gov/petroleum/supply/weekly/](https://www.eia.gov/petroleum/supply/weekly/)                                                                         | 周    | 对齐 W-FRI 后一阶差分                                         | 已有                 | 2006-01                 |
| 3   | `global_econ_activity`（Kilian 指数 / 全球 IP / PMI） | 全球需求      | Index of Global Real Economic Activity (Kilian) | Dallas Fed（备用 OECD CLI）     | igrea (xlsx/csv)；备用 FRED `OECDLOLITOAASTSAM`         | [https://www.dallasfed.org/research/igrea](https://www.dallasfed.org/research/igrea)                                                                                 | 月    | 月末 ffill + **5 周**发布滞后                                 | 待构建                | 2006-02                 |
| 4   | `nonoil_industrial_commodity`（CRB 工业 / 金属）      | 全球需求      | Global Price Index of Industrial Materials      | IMF (via FRED)              | FRED `PINDUINDEXM`                                   | [https://fred.stlouisfed.org/series/PINDUINDEXM](https://fred.stlouisfed.org/series/PINDUINDEXM)                                                                     | 月    | 月末 ffill + **5 周**发布滞后                                 | 待构建                | 2006-02                 |
| 5   | `futures_spread` 期货–现货价差                        | 市场紧张 / 预期 | Brent 近月期货 − Brent 现货（log 价差）                   | ICE (期货, Yahoo) + EIA (现货)  | `BZ=F` − 本地 `brent_price`（备用 FRED `DCOILBRENTEU`）    | [https://finance.yahoo.com/quote/BZ%3DF](https://finance.yahoo.com/quote/BZ%3DF)                                                                                     | 日    | 周最后值，`log(fut) − log(spot)`                            | 待构建                | 2007-08                 |
| 6   | `ovx`（优先）/ `vix`                                | 石油特定不确定性  | CBOE Crude Oil Volatility Index / CBOE VIX      | CBOE (via Yahoo / FRED)     | `^OVX`（备用 FRED `OVXCLS`）；FRED `VIXCLS`               | [https://finance.yahoo.com/quote/%5EOVX；https://fred.stlouisfed.org/series/VIXCLS](https://finance.yahoo.com/quote/%5EOVX；https://fred.stlouisfed.org/series/VIXCLS) | 日    | 周最后值                                                   | `ovx` 待构建；`vix` 已有 | OVX 2007-05；VIX 2006-01 |
| 7   | `gpr` 地缘政治风险                                    | 预防性需求     | Geopolitical Risk Index                         | Caldara & Iacoviello (2022) | 文件 `data_gpr_export.xls`，列 `GPR`                     | [https://www.matteoiacoviello.com/gpr.htm](https://www.matteoiacoviello.com/gpr.htm)                                                                                 | 月    | 月末 ffill + **1 周**发布滞后                                 | 待构建                | 2006-01                 |
| 8   | `dgs10_change`（ΔDGS10，10Y 收益率变化 / 一阶差分）         | 利率 / 持有成本 | 美国 10 年期国债收益率变化                                 | FRED (Board of Governors)   | 由本地 `treasury_10y`（FRED `DGS10`）派生                   | [https://fred.stlouisfed.org/series/DGS10](https://fred.stlouisfed.org/series/DGS10)                                                                                 | 日    | 周最后值 → 一阶差分（**不用水平值**）                                 | 待构建（源列已有）          | 2006-01                 |
| 9   | `gold_return`（`gold_price` 衍生）                  | 商品联动 / 避险 | LBMA Gold Price PM (USD)                        | ICE/LBMA (via FRED)         | FRED `GOLDPMGBD228NLBM`（备用 yfinance `GC=F`）          | [https://fred.stlouisfed.org/series/GOLDPMGBD228NLBM](https://fred.stlouisfed.org/series/GOLDPMGBD228NLBM)                                                           | 日    | 周最后值 → 对数收益率                                           | 待构建                | 2006-01                 |
| 10  | `commodity_fx`（CAD/AUD，优先于宽美元）                  | 汇率渠道      | 商品货币强度（CAD/USD、AUD/USD 均值）                      | Yahoo Finance（备用 FRED）      | `CADUSD=X`, `AUDUSD=X`（备用 FRED `DEXCAUS`, `DEXUSAL`） | [https://finance.yahoo.com/quote/CADUSD=X](https://finance.yahoo.com/quote/CADUSD=X)                                                                                 | 日    | 周最后值 → 两者周 % 变化均值                                      | 待构建                | 2006-01                 |


**说明：**

- 序号与文献矩阵 §① 一一对应；#1 中 `brent_price` 滞后项在建模阶段扩展，不单列变量名（P076）。
- #6 文献建议 **OVX 优先于 VIX**（P052）；M1 同时保留两者，建模时可做共线性检查。
- #8 水平值 `treasury_10y`（DGS10）未通过单位根检验（P076），M1 使用一阶差分 `dgs10_change`。
- #10 宽口径美元指数（DXY）证据有限（P053/P004），改用商品出口国汇率 CAD/AUD。
- `ovx`、`futures_spread` 覆盖起点较晚（OVX 2007-05；Brent 期货 2007-08），早期周缺失属正常。
- `gpr` 为新闻文本聚合指数；文本模态已移除（Meeting 02），归入 M1 作低频地缘政治风险代理。
- `futures_spread` 为近月连续合约与现货之差的**近似**期限价差。
- 月频变量（#3, #4, #7）发布滞后为保守估计，避免前视偏差；进入模型前仍须统一滞后处理。
- 引用：GPR — Caldara, D. & Iacoviello, M. (2022) *Measuring Geopolitical Risk*, AER；Kilian REA — Kilian (2009)，Dallas Fed 维护更新版。

