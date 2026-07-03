# M1 金融/宏观特征矩阵 — 数据字典

> 对应数据文件：`03_data/processed/M1/outputs/m1_weekly_features.csv`
> 构建脚本：`03_data/processed/M1/py/build_m1_weekly.py`
> 最后更新：2026-07-03

---

## 1. 基本信息


| 项      | 值                                          |
| ------ | ------------------------------------------ |
| 时间索引   | `week_ending_friday`，每周五截止（W-FRI）          |
| 时间范围   | 2006-01-06 ~ 2025-12-26                    |
| 形状     | 1043 行 × 35 列                              |
| 统一比较窗口 | 2019-01-04 ~ 2025-12-26（365 周），窗口内 35 列零缺失 |
| 缺失约定   | 空单元格 = 该周该变量尚不可得（NaN），不做隐式填补               |
| 频率     | 周频（所有模态统一对齐到周五）                            |


---



## 2. 构建脚本做了什么（`build_m1_weekly.py`）

**定位**：M1 金融/宏观特征的 Layer 2（processed）构建器，采用「单脚本、单表、离线优先」设计。它将原先「建周频索引 → 构建衍生列 → merge 合并」三步流水线收拢为一次运行：从 `raw/01_market_financial/` 按供应来源（EIA / FRED / Yahoo / Other）读取原始文件，在内存中完成重采样、发布滞后与派生计算，直接写出唯一产物 `m1_weekly_features.csv`——不产生中间 CSV，也不需要单独的 merge 步骤。默认完全依赖本地 raw；个别源缺失时跳过该列并告警，整体仍可复现。

### 2.1 输入 → 输出


| 项        | 内容                                                          |
| -------- | ----------------------------------------------------------- |
| 输入根目录    | `03_data/raw/01_market_financial/`                          |
| `EIA/`   | Brent / WTI 日频现货 (.xls)、周报 WPSR（手工下载 .xls）                  |
| `FRED/`  | DGS10、VIXCLS、DTWEXBGS、DFF、PINDUINDEXM（+ gold / ovx 备选）      |
| `Yahoo/` | ^GSPC、^OVX、BZ=F、CADUSD=X、AUDUSD=X、GC=F                      |
| `Other/` | Dallas Fed Kilian REA (igrea)、Caldara–Iacoviello GPR        |
| 输出       | `03_data/processed/M1/outputs/m1_weekly_features.csv`（唯一产物） |
| 时间索引     | `pd.date_range("2006-01-01", "2025-12-31", freq="W-FRI")`   |




### 2.2 处理流程（`main()` 调用顺序）

1. **建周频骨架**：生成 2006–2025 的每周五（W-FRI）日期轴，作为所有列统一对齐的基准。
2. **基础块** `build_base()`：
  - 价格：EIA Brent/WTI 日频 → 每周五最后值（先聚合、后算收益率，计算前 `assert price>0`）；派生 `brent_log_return / wti_log_return / brent_wti_spread`（Brent、WTI 统一对数收益口径）。
  - EIA 周报：按 `WPSR_MAP` 逐个读入 10 个 WPSR 字段 → 对齐到报告周五 → +1 周发布滞后；派生 `crude_stocks_change / cushing_stocks_change`。
  - FRED 宏观日频（VIX / 美元指数 / 10Y / 联邦基金）→ 每周五最后值。
  - Yahoo 标普 500 → 每周五最后值；派生 `sp500_log_return`（与 Brent/WTI 同口径：`ln(Pₜ/Pₜ₋₁)`，不乘 100）。
3. **衍生块** `TO_BUILD`（除非 `--base-only`）：逐个构建 9 个变量 `ovx / gpr / gold_return / global_econ_activity / nonoil_industrial_commodity / brent_f1_spot_log_basis / brent_roll_week / cadusd_log_return / dgs10_change`（`gpr` 用日度 GPRD 周均值）；单个失败仅跳过并记录警告，不中断整体。
4. **模态可用性标志**：按对应列是否非空生成 `avail_market / avail_eia_weekly / avail_sp500 / avail_dollar_index`。
5. **写盘 + 汇总**：裁剪到研究区间、写出 CSV，并打印形状、各列非空覆盖率与警告清单。



### 2.3 三类重采样规则（核心函数）


| 函数                     | 适用来源                 | 规则                                 |
| ---------------------- | -------------------- | ---------------------------------- |
| `daily_to_weekly_last` | 日频价格 / 宏观            | `resample("W-FRI").last()` 取每周五最后值 |
| `weekly_eia_to_friday` | EIA 周报               | 报告日先对齐到所在周五，再整体 `shift(+1 周)` 发布滞后 |
| `monthly_to_weekly`    | 月频（GPR / REA / 工业原料） | 月末对齐 → 前向填充到周频 → `shift(lag)` 保守滞后 |


> 滞后常量集中在脚本顶部：`EIA_LAG_WEEKS=1`、`GPR_LAG_WEEKS=1`、`MONTHLY_LAG_WEEKS=5`（防泄漏理由见第 3 节）。



### 2.4 运行模式与容错

- **默认离线**：仅读本地 raw；缺文件则打印 `[warn]` 并跳过该列，不崩溃（保证可复现）。
- `**--online`**：本地缺失时在线补抓（FRED `fredgraph.csv` / yfinance），带超时重试与指数退避。
- `**--refresh-raw**`：先运行 `download_m1_raw.py` 再构建。
- `**--base-only**`：只建基础块，跳过 9 个衍生变量。
- **分来源解析器**：`read_eia_xls`（EIA .xls 的 "Data 1" 表）、`parse_two_col`（FRED/yfinance 两列）、`parse_gpr`、`parse_kilian`。

---



## 3. 防泄漏与发布时间戳对齐（关键）

所有变量按「真实可得时间」对齐：在周五 T 预测下一周时，只使用截至 T 已发布的信息（no look-ahead）。


| 来源类别                 | 周频化方法            | 发布滞后      | 常量                    |
| -------------------- | ---------------- | --------- | --------------------- |
| 价格 / 宏观金融（日频）        | 取每周五收盘最后值        | 0（市场当日可得） | —                     |
| EIA 周报（WPSR）         | 对齐到报告周五后整体后移 1 周 | **+1 周**  | `EIA_LAG_WEEKS=1`     |
| GPR 地缘风险（月频）         | 月末对齐 + 前向填充到周频   | +1 周      | `GPR_LAG_WEEKS=1`     |
| 其他月频宏观（REA / 工业原料价格） | 月末对齐 + 前向填充到周频   | +5 周      | `MONTHLY_LAG_WEEKS=5` |


要点：

- **EIA WPSR**：统计周「截至周五 T」，实际「次周三（T+5 天）」才发布。因此每条报告整体后移一周，使其只在真实发布后的第一个周五进入矩阵。这正是 10 个 EIA 原始列首个有效周从 `2006-01-13` 开始（首周 `2006-01-06` 置 NaN）的原因。对「每周五预测」节奏而言，+1 周滞后既不泄漏、也不晚于真实可得（信息最优）。
- **月频宏观**目前仍采用「月末对齐 + 前向填充 + 保守滞后」。这是 M1 金融层的处理；遥感/航运模态在阶段 0 将改为显式 `age / valid_mask`（不再 ffill）。



### 3.1 月频发布滞后 rationale（GPR +1 周 vs REA/工业原料 +5 周）

**设计原则（有文献支撑）**：预测时点 T 只能使用截至 T 已公开的信息；变量应按**真实发布时间戳**而非统计参考期进入模型，不能把某月最终值回填到该月所有周（Alquist, Kilian & Vigfusson, 2013 [P053]；Kilian, 2009 [P052]；Costa et al., 2021 [P072]）。官方宏观与商品统计普遍存在发布延迟与后续修订（P053 §11；Bricongne et al., 2026 [P069] 指出官方石油消费统计约滞后 2–3 个月），因此月频变量在周频预测中必须额外处理。

**实现方式（本项目简化）**：raw 层通常只有统计月份、无逐条发布日，故 M1 采用固定规则 `monthly_to_weekly`：月末对齐 → 前向填充到 W-FRI → `shift(lag_weeks)`。这比理想的 **real-time vintage / as-of join**（P054；P053 推荐）粗糙，但在无 ALFRED 式历史版本库时，用固定 shift 做**保守上界**是常见工程折中。

**为何分档（1 周 vs 5 周）——按数据源发布链快慢分 tier，非单一文献常数**：


| 变量                            | 滞后   | 分档理由                                                                                                |
| ----------------------------- | ---- | --------------------------------------------------------------------------------------------------- |
| `gpr`                         | +1 周 | Caldara–Iacoviello GPR 为学者维护的新闻聚合指数，月频但通常**次月上旬**即更新上月值；发布链短，取最小保守缓冲（ffill 后再推 1 周 ≈ 跨月后第一个完整周才可见）。 |
| `global_econ_activity`        | +5 周 | Dallas Fed Kilian REA：月频研究指数，底层海运运价汇总，实际可得晚于统计月末。                                                   |
| `nonoil_industrial_commodity` | +5 周 | IMF 工业原料价格指数（FRED `PINDUINDEXM`）：官方商品指数，典型发布晚于参考月（常在中下旬）。                                           |


**+5 周是否为「常见做法」？** 文献**不**规定「月频宏观一律 shift 5 周」。学界共识是原则（发布日对齐、避免 revised-data 伪样本外），具体 weeks 取决于序列与 vintage 数据。固定 +5 周是本项目的**保守启发式**（heuristic）：在缺少逐月发布日的情况下，约等于「参考月结束后留 ~1 个月余量」，确保周五预测时不提前用到当月宏观终值；**宁晚勿早**。更严格的做法是用 FRED/ALFRED real-time vintages 或逐序列核对发布日历（P054 的 real-time 设计）；本论文 M2 已用 `month_end + 15 d` as-of join 做更细对齐，M1 月频仍保留此简化 tier。

**自检锚点**：`gpr` 首个有效周 `2006-01-13`（+1 周）；`global_econ_activity` `2006-02-10`（+5 周）；`nonoil_industrial_commodity` `2006-03-10`（数据起点 2006-02 + 5 周）。若需稳健性，可对 `MONTHLY_LAG_WEEKS` 做 3/5/7 周敏感性（backlog）。

---



## 4. 字段说明

> 覆盖率为全样本（1043 周）非空占比；首个有效周为该列第一个非 NaN 的周五。



### 4.1 价格与派生（5 列）


| 字段                 | 含义                              | 来源                      | 单位   | 周频化     | 滞后  | 覆盖率    | 首个有效周      |
| ------------------ | ------------------------------- | ----------------------- | ---- | ------- | --- | ------ | ---------- |
| `brent_price`      | Brent 现货价（**预测目标基于此**）          | EIA 日频现货 (DCOILBRENTEU) | 美元/桶 | 日→周五最后值 | 0   | 100.0% | 2006-01-06 |
| `wti_price`        | WTI(Cushing) 现货价                | EIA 日频现货 (DCOILWTICO)   | 美元/桶 | 日→周五最后值 | 0   | 100.0% | 2006-01-06 |
| `brent_log_return` | Brent 周对数收益（其下一周值 r₍ₜ₊₁₎ 为训练目标） | 派生                      | 无量纲  | —       | 0   | 99.9%  | 2006-01-13 |
| `wti_log_return`   | WTI 周对数收益（与 Brent 同口径）          | 派生                      | 无量纲  | —       | 0   | 99.9%  | 2006-01-13 |
| `brent_wti_spread` | Brent − WTI 价差                  | 派生                      | 美元/桶 | —       | 0   | 100.0% | 2006-01-06 |




#### 4.1.1 变量选择依据：文献与业界支撑

> §4.1 的 5 个价格/派生特征均为原油价格研究中的**标准变量**，同时具备学术文献与业界/机构材料（EIA、OPEC、CME、ICE）的常用性支撑。下表给出各变量的代表性出处与用途；完整学术引用见「参考来源」（`[P025]`/`[P104]`–`[P114]` 与 `[R001]`–`[R021]` 均已并入主文献表（`literature_matrix.md` §①-b；Report 见 §4.6 / `literatue.md` §一-b））。


| 变量                 | 代表性学术文献                                                                                                                                                                                                                                                                 | 文献中的用法                                                                                                                                             | 代表性业界/机构材料                                                                   | 结论                                                                                         |
| ------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------ |
| `brent_price`      | Yu, Wang & Lai (2008, *Energy Economics*) [P104]；Abdollahi & Ebrahimi (2020, *Energy*) [P105]                                                                                                                                                                           | 作预测目标（Brent 价格 Pₜ），研究价格水平趋势、非线性与预测误差                                                                                                               | EIA 日/周现货表长期发布 Brent–Europe；ICE Brent Index 用于期货现金结算 [R001]                  | 全球原油预测**核心价格变量**，极常用                                                                       |
| `wti_price`        | Ye, Zyren & Shore (2005, *Int. J. Forecasting*) [P106]；Yu, Wang & Lai (2008) [P104]                                                                                                                                                                                     | 作预测目标，或作 Brent 预测的跨市场解释变量                                                                                                                          | EIA 发布 WTI–Cushing 现货价；CME 视 WTI 为全球主要基准 [R001][R005]                        | 美国原油市场**核心基准变量**，极常用                                                                       |
| `brent_log_return` | Chen, Zerilli & Baum (2019, *Energy Economics*) [P109]；Charles & Darné (2017, *Energy Economics*) [P114]；Zhang, Yao, He & Ripple (2019, *Int. Rev. Econ. & Finance*) [P110]；Chen, Chiu & Hsiao (2021, *Sustainability*) [P107]                                          | rₜ = ln(Pₜ/Pₜ₋₁)，用于 GARCH / 随机波动率 / VaR / 波动预测与投资风险度量                                                                                              | EIA / OPEC MOMR 报收益变化；ICE Risk Model 用 return risk factor 做 VaR [R003][R006] | 学术计量**标准变量**；本项目训练目标口径                                                                     |
| `wti_log_return`   | Hao & Wang (2023, *Humanit. Soc. Sci. Commun.*) [P025]；Chen, Zerilli & Baum (2019, *Energy Economics*) [P109]；Charles & Darné (2017, *Energy Economics*) [P114]；Wang & Chen (2025, *J. Risk Financial Manag.*) [P111]；Ma, Xiong & Bao (2021, *Energy Economics*) [P108] | 均以 **EIA 现货价**对数一阶差分 ln(Pₜ/Pₜ₋₁) 为收益：[P025] 以「本周二 − 上周二」对数价构造**周度** WTI 收益并预测下一周（与本项目周度目标最贴合），[P109] 用于随机波动率/VaR/CVaR，[P114] 用于 GARCH/GAS/MSM 波动预测 | CME 市场分析常报「WTI 本周涨跌」周表现 [R005]                                               | **EIA 现货 → 对数收益为标准做法**（[P025]/[P109] 均用 EIA WTI 现货，支撑本项目「现货而非期货」口径）；[P025] 周频 WTI 收益率目标最贴合 |
| `brent_wti_spread` | Scheitrum, Carter & Revoredo-Giha (2018, *Energy Economics*) [P112]；Bravo Caro, Golpe, Iglesias & Vides (2020, *Energy Economics*) [P113]                                                                                                                               | 衡量海运原油市场与美国内陆原油市场的相对价格状态                                                                                                                           | EIA 长期发布 Brent–WTI 价差分析；OPEC MOMR 月报近月期货价差 [R002][R004]                      | 文献与业界**都常用，且有明确经济含义**                                                                      |


**参考来源（§4.1 学术文献 → 主文献表 P025/P104–P114；业界材料 → §4.6 R001–R021）：**

*学术文献（详见* `01_literature/literature_matrix.md` *§①-b）*

- [P104] Yu, L., Wang, S. & Lai, K.K. (2008). Forecasting crude oil price with an EMD-based neural network ensemble learning paradigm. *Energy Economics*, 30(5), 2623–2635. [https://doi.org/10.1016/j.eneco.2008.05.003](https://doi.org/10.1016/j.eneco.2008.05.003)
- [P105] Abdollahi, H. & Ebrahimi, S.B. (2020). A new hybrid model for forecasting Brent crude oil price. *Energy*, 200, 117520. [https://doi.org/10.1016/j.energy.2020.117520](https://doi.org/10.1016/j.energy.2020.117520)
- [P106] Ye, M., Zyren, J. & Shore, J. (2005). A monthly crude oil spot price forecasting model using relative inventories. *International Journal of Forecasting*, 21(3), 491–501. [https://doi.org/10.1016/j.ijforecast.2005.01.001](https://doi.org/10.1016/j.ijforecast.2005.01.001)
- [P107] Chen, Y.-W., Chiu, C.-Y. & Hsiao, M.-C. (2021). An auxiliary index for reducing Brent crude investment risk. *Sustainability*, 13(9), 5050. [https://doi.org/10.3390/su13095050](https://doi.org/10.3390/su13095050)
- [P108] Ma, R.R., Xiong, T. & Bao, Y. (2021). The Russia–Saudi Arabia oil price war during the COVID-19 pandemic. *Energy Economics*, 102, 105517. [https://doi.org/10.1016/j.eneco.2021.105517](https://doi.org/10.1016/j.eneco.2021.105517)
- [P109] Chen, L., Zerilli, P. & Baum, C.F. (2019). Leverage effects and stochastic volatility in spot oil returns. *Energy Economics*, 79(C), 111–129. [https://doi.org/10.1016/j.eneco.2018.03.032](https://doi.org/10.1016/j.eneco.2018.03.032)
- [P110] Zhang, Y.-J., Yao, T., He, L.-Y. & Ripple, R.D. (2019). Volatility forecasting of crude oil market: Can the regime switching GARCH model beat the single-regime GARCH models? *International Review of Economics & Finance*, 59, 302–317. [https://doi.org/10.1016/j.iref.2018.09.006](https://doi.org/10.1016/j.iref.2018.09.006)
- [P111] Wang, L. & Chen, X. (2025). Sustainable factor augmented machine learning models for crude oil return forecasting. *Journal of Risk and Financial Management*, 18(7), 351. [https://doi.org/10.3390/jrfm18070351](https://doi.org/10.3390/jrfm18070351)
- [P112] Scheitrum, D.P., Carter, C.A. & Revoredo-Giha, C. (2018). WTI and Brent futures pricing structure. *Energy Economics*, 72(C), 462–469. [https://doi.org/10.1016/j.eneco.2018.04.039](https://doi.org/10.1016/j.eneco.2018.04.039)
- [P113] Bravo Caro, J.M., Golpe, A.A., Iglesias, J. & Vides, J.C. (2020). A new way of measuring the WTI–Brent spread: Globalization, shock persistence and common trends. *Energy Economics*, 85, 104546. [https://doi.org/10.1016/j.eneco.2019.104546](https://doi.org/10.1016/j.eneco.2019.104546)
- [P114] Charles, A. & Darné, O. (2017). Forecasting crude-oil market volatility: Further evidence with jumps. *Energy Economics*, 67(C), 508–519. [https://doi.org/10.1016/j.eneco.2017.09.002](https://doi.org/10.1016/j.eneco.2017.09.002)
- [P025] Hao, X. & Wang, Y. (2023). Cloud cover and expected oil returns. *Humanities and Social Sciences Communications*, 10, 605.（已收录于主文献表，详见 `01_literature/literature_matrix.md` §②）[https://doi.org/10.1057/s41599-023-02128-5](https://doi.org/10.1057/s41599-023-02128-5)

> 业界/机构材料见 §4.6 `[R001]`–`[R021]`。



### 4.2 EIA 周报基本面（12 列，均 +1 周发布滞后）

> 注：`crude_exports` 在 **2015-12 美国原油出口禁令取消**前后存在制度性结构断裂——2015 年前长期低位、边际预测信息有限，2016 年后信息量更强；完整解读与文献见 §6。


| 字段                      | 含义                         | 来源       | 单位   | 周频化    | 滞后   | 覆盖率   | 首个有效周      |
| ----------------------- | -------------------------- | -------- | ---- | ------ | ---- | ----- | ---------- |
| `crude_stocks_excl_spr` | 商业原油库存（不含 SPR）             | EIA WPSR | 千桶   | 周→周五对齐 | +1 周 | 99.9% | 2006-01-13 |
| `cushing_stocks`        | Cushing 原油库存               | EIA WPSR | 千桶   | 周→周五对齐 | +1 周 | 99.9% | 2006-01-13 |
| `crude_production`      | 美国国内原油产量                   | EIA WPSR | 千桶/日 | 周→周五对齐 | +1 周 | 99.9% | 2006-01-13 |
| `crude_imports`         | 原油进口                       | EIA WPSR | 千桶/日 | 周→周五对齐 | +1 周 | 99.9% | 2006-01-13 |
| `crude_exports`         | 原油出口                       | EIA WPSR | 千桶/日 | 周→周五对齐 | +1 周 | 99.9% | 2006-01-13 |
| `refinery_crude_input`  | 炼厂原油加工量                    | EIA WPSR | 千桶/日 | 周→周五对齐 | +1 周 | 99.9% | 2006-01-13 |
| `refinery_utilisation`  | 炼厂开工率                      | EIA WPSR | %    | 周→周五对齐 | +1 周 | 99.9% | 2006-01-13 |
| `gasoline_supplied`     | 汽油表观需求（product supplied）   | EIA WPSR | 千桶/日 | 周→周五对齐 | +1 周 | 99.9% | 2006-01-13 |
| `distillate_supplied`   | 馏分油表观需求（product supplied）  | EIA WPSR | 千桶/日 | 周→周五对齐 | +1 周 | 99.9% | 2006-01-13 |
| `jet_fuel_supplied`     | 航空煤油表观需求（product supplied） | EIA WPSR | 千桶/日 | 周→周五对齐 | +1 周 | 99.9% | 2006-01-13 |
| `crude_stocks_change`   | 商业原油库存周变化                  | 派生 diff  | 千桶   | —      | +1 周 | 99.8% | 2006-01-20 |
| `cushing_stocks_change` | Cushing 库存周变化              | 派生 diff  | 千桶   | —      | +1 周 | 99.8% | 2006-01-20 |




#### 4.2.1 变量选择依据：文献与业界支撑

> §4.2 的 12 列 EIA 周报基本面变量，均可在同行评审文献与官方/业界材料（EIA、CME、Reuters）中找到明确使用；下表给出代表性出处与用途。须守住两条诚实边界：(1) **「变量被文献使用过」不等于「该变量已被证明能提升周频 Brent 价格预测」**——库存与库存意外的市场影响/预测证据最强，产量、炼厂、成品油消费更多出现在结构模型、价格发现或月频因子模型中；(2) 多数经典文献为**月频或结构/因子模型**（SVAR、FAVAR、regime NN），而本项目为**周频、扁平特征 + ML**，故这些文献用作**变量选择依据**，其增量价值仍由本研究的 SHAP + 消融 + DM/CW 检验判定。完整学术引用见「参考来源」（`[P115]`–`[P121]` 已并入主文献表 `01_literature/literature_matrix.md` §①-c；`[R001]`–`[R021]` 已并入主文献表 `01_literature/literatue.md` §一-b（Report 层））。


| 变量                      | 代表性学术文献                                                                                                                        | 文献中的用法（含口径提示）                                                            | 代表性业界/机构材料                                                        | 支撑程度         |
| ----------------------- | ------------------------------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------ | ----------------------------------------------------------------- | ------------ |
| `crude_stocks_excl_spr` | Bu (2014) [P115]；Armstrong, Cardella & Sabah (2021) [P116]；Malliaris & Malliaris (2021) [P120]                                 | 商业原油库存（不含 SPR）作为市场平衡/库存信息冲击的核心量。注：高频研究多用**库存变化/意外**而非绝对水平                | EIA WPSR 将 U.S. commercial crude stocks excl. SPR 列为重点关注指标 [R007] | 非常强          |
| `cushing_stocks`        | Kim, Baek & Heo (2020) [P117]                                                                                                  | 以 Global / US / **Cushing** 库存建 SVAR，识别库存的缓冲与投机响应                        | EIA/CME 视 Cushing 为 WTI 实物交割与定价中心 [R007][R011]                    | 非常强（尤宜 WTI）  |
| `crude_production`      | Armstrong et al. (2021) [P116]（库存意外的构成项，稳健性）；Wei (2026) [P121]（LASSO 候选，含美国产量增长）；Malliaris & Malliaris (2021) [P120]           | 供给侧基本面变量。注：P053 指出产量对油价的**增量预测力偏弱**                                      | EIA WPSR 美国石油平衡表逐周发布原油产量 [R007]                                   | 强（供给侧标准变量）   |
| `crude_imports`         | Zagaglia (2010) [P119]（能源数量信息集含 total crude imports）；Wei (2026) [P121]（含美国/全球进口增长）                                             | 进口作为供给来源，用于因子提取或特征筛选                                                     | EIA WPSR 单独发布原油进出口表 [R007]                                        | 强            |
| `crude_exports`         | 以周度出口作**单独价格预测特征**的经典文献较少；相关研究多考察出口限制解除对 WTI–Brent 价差的影响（Scheitrum et al. 2018，见 §4.1 [P112]）                                  | 出口能力/限制→基准价差机制，而非单一价格预测因子                                                | 交易层面常用周度出口解释库存下降与 WTI 对国际市场的联动 [R013]                             | 行业强；学术上偏价差机制 |
| `refinery_crude_input`  | Zagaglia (2010) [P119]（crude oil refinery net input）；Armstrong et al. (2021) [P116]（refiner input 为库存意外构成项）                    | 炼厂对原油的直接需求（refinery runs / crude runs）                                   | EIA WPSR 发布炼厂投入与产出数据 [R007]                                       | 强            |
| `refinery_utilisation`  | Kaufmann, Dées, Gasteuil & Mann (2008) [P118]                                                                                  | 炼厂开工率进入**实际油价方程**并做一步样本外预测。注：关系为**负**（利用率↑→实际油价↓），且用于解释 2004–06 涨价（月频背景） | EIA 将炼厂可运营产能利用率列为重点指标；Reuters/CME 常与加工量并报 [R007][R013]            | 强（注意负向、月频）   |
| `gasoline_supplied`     | Zagaglia (2010) [P119]（motor gasoline product supplied，月频 FAVAR）；Malliaris & Malliaris (2021) [P120]（petroleum supplied 作消费代理） | 下游燃料需求的高频代理。注：周频 product supplied **噪声较大**                               | EIA product supplied 用作汽油消费高频代理（驾驶季/节假日分析）[R009]                  | 强（周数据噪声化）    |
| `distillate_supplied`   | Zagaglia (2010) [P119]（distillate fuel oil product supplied）                                                                   | 柴油/取暖油需求，反映货运、工业与冬季取暖                                                    | EIA 单独发布馏分油周度 product supplied [R010]                             | 强            |
| `jet_fuel_supplied`     | Zagaglia (2010) [P119]（jet fuel product supplied，纳入广义能源信息集）                                                                    | 航空燃料需求进入影响油价的能源因子面板                                                      | EIA/CME 按周跟踪 jet fuel demand（常用四周均值同比）[R009][R011]                | 有依据；行业针对性较强  |
| `crude_stocks_change`   | Bu (2014) [P115]；Armstrong et al. (2021) [P116]                                                                                | 实际库存变化与**库存意外**——学术证据最强的变量之一（意外在数秒内进入 WTI 期货价）                           | CME/Reuters 的 headline 即本周 build/draw 及其与分析师预期之差 [R012][R013]     | 最强之一         |
| `cushing_stocks_change` | Kim, Baek & Heo (2020) [P117]                                                                                                  | Cushing 库存对供需冲击的动态响应；学术常用水平/结构冲击，未必命名为一阶差分                               | 市场极重视 Cushing 周度增减（可限制 WTI 上行）[R011]                              | 行业非常强        |


**参考来源（§4.2 学术文献 → 主文献表 P115–P121；业界材料 → §4.6 R001–R021）：**

*学术文献（详见* `01_literature/literature_matrix.md` *§①-c）*

- [P115] Bu, H. (2014). Effect of inventory announcements on crude oil price volatility. *Energy Economics*, 46, 485–494. [https://doi.org/10.1016/j.eneco.2014.05.015](https://doi.org/10.1016/j.eneco.2014.05.015)（核心结论：起作用的是「库存信息冲击」而非实际库存变化本身）
- [P116] Armstrong, W.J., Cardella, L. & Sabah, N. (2021). Information shocks, disagreement, and drift. *Journal of Financial Economics*, 140(3), 916–940. [https://doi.org/10.1016/j.jfineco.2021.02.002](https://doi.org/10.1016/j.jfineco.2021.02.002)
- [P117] Kim, S., Baek, J. & Heo, E. (2020). Crude oil inventories: The two faces of Janus? *Empirical Economics*, 59(2), 1003–1018. [https://doi.org/10.1007/s00181-019-01660-1](https://doi.org/10.1007/s00181-019-01660-1)
- [P118] Kaufmann, R.K., Dées, S., Gasteuil, A. & Mann, M. (2008). Oil prices: The role of refinery utilization, futures markets and non-linearities. *Energy Economics*, 30(5), 2609–2622. [https://doi.org/10.1016/j.eneco.2008.04.010](https://doi.org/10.1016/j.eneco.2008.04.010)
- [P119] Zagaglia, P. (2010). Macroeconomic factors and oil futures prices: A data-rich model. *Energy Economics*, 32(2), 409–417. [https://doi.org/10.1016/j.eneco.2009.11.003](https://doi.org/10.1016/j.eneco.2009.11.003)
- [P120] Malliaris, A.G. & Malliaris, M. (2021). What microeconomic fundamentals drove global oil prices during 1986–2020? *Journal of Risk and Financial Management*, 14(8), 391. [https://doi.org/10.3390/jrfm14080391](https://doi.org/10.3390/jrfm14080391)
- [P121] Wei, X. (2026). Forecasting crude oil futures price with energy uncertainty: Evidence from machine learning methods. *PLoS One*, 21(2), e0341496. [https://doi.org/10.1371/journal.pone.0341496](https://doi.org/10.1371/journal.pone.0341496)

> 业界/机构材料见 §4.6 `[R001]`–`[R021]`。



### 4.3 宏观金融（5 列）


| 字段                 | 含义           | 来源            | 单位  | 周频化     | 滞后  | 覆盖率    | 首个有效周      |
| ------------------ | ------------ | ------------- | --- | ------- | --- | ------ | ---------- |
| `vix`              | CBOE 波动率指数   | FRED VIXCLS   | 指数  | 日→周五最后值 | 0   | 100.0% | 2006-01-06 |
| `dollar_index`     | 美元名义广义指数     | FRED DTWEXBGS | 指数  | 日→周五最后值 | 0   | 100.0% | 2006-01-06 |
| `treasury_10y`     | 10 年期美债收益率   | FRED DGS10    | %   | 日→周五最后值 | 0   | 100.0% | 2006-01-06 |
| `fed_funds_rate`   | 联邦基金有效利率     | FRED DFF      | %   | 日→周五最后值 | 0   | 100.0% | 2006-01-06 |
| `sp500_log_return` | 标普 500 周对数收益 | 派生            | 无量纲 | —       | 0   | 99.9%  | 2006-01-13 |




#### 4.3.1 变量选择依据：文献与业界支撑

> §4.3 的 6 个宏观金融变量是原油价格研究的标准跨市场（控制）变量，均有同行评审文献与机构材料支撑。沿用前节的诚实边界：这些文献多为**月频/结构模型或事件研究**，且部分变量（股市、联邦基金）与油价的关系方向随冲击来源或频率而变，故此处用作**变量选择依据**，其在周频 Brent 预测中的增量价值由本研究的 SHAP + 消融 + DM/CW 检验判定。完整学术引用见「参考来源」（`[P122]`–`[P130]` 已并入主文献表 `01_literature/literature_matrix.md` §①-d；`[R001]`–`[R021]` 已并入主文献表 `01_literature/literatue.md` §一-b（Report 层））。


| 变量                 | 代表性学术文献                                                                                                                               | 文献中的用法（含口径提示）                                                                                                                                                                                                                                       | 代表性业界/机构材料                                           | 支撑程度                                  |
| ------------------ | ------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------- | ------------------------------------- |
| `vix`              | Tissaoui et al. (2023) [P122]                                                                                                         | VIX 作为金融市场不确定性指标进入 XGBoost + SHAP，用于预测 WTI；跨市场风险传导                                                                                                                                                                                                  | Cboe 定义 VIX 为 S&P 500 期权隐含的预期波动率基准 [R014]            | 很强（油市专属性弱于 OVX）                       |
| `dollar_index`     | He, Wang & Lai (2010) [P123]                                                                                                          | 贸易加权美元指数与实际油价、Kilian 经济活动指数协整                                                                                                                                                                                                                       | EIA 指出原油以美元计价，美元变动改变非美经济体购买成本 [R015]                 | 很强（下一周预测宜优先用周变化而非水平）                  |
| `treasury_10y`     | Qadan & Cohen (2024) [P124]                                                                                                           | 10Y 收益率的前瞻波动率（Bond-VIX 式）预测油价收益与波动。注：直接支撑的是"利率不确定性"，水平值主要表达利率制度状态                                                                                                                                                                                   | CME 将利率与美元、股市、油价联合分析 [R016]                          | 中等偏强                                  |
| `fed_funds_rate`   | Basistha & Kurov (2015) [P125]                                                                                                        | 货币政策意外→能源价格：日内窗口显著负向，但日度以上累积响应与月频 SVAR 均不显著                                                                                                                                                                                                         | CME 将联邦基金利率列为宏观关键指标 [R016]                           | 中等（周频多数周不变，短期信息有限）                    |
| `sp500_log_return` | Sadorsky (1999) [P127]；Kilian & Park (2009) [P126]；Lu et al. (2021) [P128]；Hussain et al. (2022) [P129]；Roy, Soni & Deb (2023) [P130] | Sadorsky 明确用 S&P 500 **continuously compounded return**；Lu et al. 用 WTI 与 S&P 500 期货 **log-returns**；Hussain et al. 对 S&P 500 与 Brent 计算日度连续复合收益；Roy et al. 对 S&P 500、Brent、WTI 等统一用 `ln(Pₜ/Pₜ₋₁)`。注：非平稳 `sp500` **指数水平已移除**（测试期超训练范围、外推风险），仅保留对数收益 | RiskMetrics / MSCI 以**对数收益**为 VaR、波动率与相关性估计基础 [R017] | 很强（已取代 `sp500` 水平；与 Brent/WTI 对数口径一致） |


**参考来源（§4.3 学术文献 → 主文献表 P122–P130；业界材料 → §4.6 R001–R021）：**

*学术文献（详见* `01_literature/literature_matrix.md` *§①-d）*

- [P122] Tissaoui, K., Zaghdoudi, T., Hakimi, A. & Nsaibi, M. (2023). Do gas price and uncertainty indices forecast crude oil prices? Fresh evidence through XGBoost modeling. *Computational Economics*, 62(2), 663–687. [https://doi.org/10.1007/s10614-022-10305-y](https://doi.org/10.1007/s10614-022-10305-y)
- [P123] He, Y., Wang, S. & Lai, K.K. (2010). Global economic activity and crude oil prices: A cointegration analysis. *Energy Economics*, 32(4), 868–876. [https://doi.org/10.1016/j.eneco.2009.12.005](https://doi.org/10.1016/j.eneco.2009.12.005)
- [P124] Qadan, M. & Cohen, G. (2024). Uncertainty about interest rates and crude oil prices. *Financial Innovation*, 10, 9. [https://doi.org/10.1186/s40854-023-00551-w](https://doi.org/10.1186/s40854-023-00551-w)
- [P125] Basistha, A. & Kurov, A. (2015). The impact of monetary policy surprises on energy prices. *Journal of Futures Markets*, 35(1), 87–103. [https://doi.org/10.1002/fut.21639](https://doi.org/10.1002/fut.21639)
- [P126] Kilian, L. & Park, C. (2009). The impact of oil price shocks on the U.S. stock market. *International Economic Review*, 50(4), 1267–1287. [https://doi.org/10.1111/j.1468-2354.2009.00568.x](https://doi.org/10.1111/j.1468-2354.2009.00568.x)
- [P127] Sadorsky, P. (1999). Oil price shocks and stock market activity. *Energy Economics*, 21(5), 449–469. [https://doi.org/10.1016/S0140-9883(99)00020-1](https://doi.org/10.1016/S0140-9883(99)00020-1)（实际股票收益 = S&P 500 连续复合收益 − 通胀率）
- [P128] Lu, X., Liu, K., Lai, K.K. & Cui, H. (2021). The relationship between crude oil futures market and Chinese/US stock index futures market based on breakpoint test. *Entropy*, 23(9), 1172. [https://doi.org/10.3390/e23091172](https://doi.org/10.3390/e23091172)（WTI 与 S&P 500 股指期货 log-returns）
- [P129] Hussain, S.M., Naveed, A., Ahmed, S. et al. (2022). Disaggregating the impact of oil prices on European industrial equity indices: A spatial econometric analysis. *Empirical Economics*, 62(6), 2673–2692. [https://doi.org/10.1007/s00181-021-02116-1](https://doi.org/10.1007/s00181-021-02116-1)（S&P 500 与 Brent 日度连续复合收益）
- [P130] Roy, A., Soni, A. & Deb, S. (2023). A wavelet-based methodology to compare the impact of pandemic versus Russia–Ukraine conflict on crude oil sector and its interconnectedness with other energy and non-energy markets. *Energy Economics*, 124, 106830. [https://doi.org/10.1016/j.eneco.2023.106830](https://doi.org/10.1016/j.eneco.2023.106830)（S&P 500、Brent、WTI 等统一对数收益）

> 业界/机构材料见 §4.6 `[R001]`–`[R021]`。



### 4.4 衍生市场/宏观变量（9 列，脚本中的 to-build 块）


| 字段                            | 含义                             | 来源                                           | 单位       | 周频化       | 滞后   | 覆盖率    | 首个有效周      |
| ----------------------------- | ------------------------------ | -------------------------------------------- | -------- | --------- | ---- | ------ | ---------- |
| `ovx`                         | 原油波动率指数                        | Yahoo ^OVX / FRED OVXCLS                     | 指数       | 日→周五最后值   | 0    | 93.3%  | 2007-05-11 |
| `gpr`                         | 地缘政治风险指数（日度 GPRD）              | Caldara & Iacoviello (data_gpr_daily_recent) | 指数       | 日→周均值     | +1 周 | 99.9%  | 2006-01-13 |
| `gold_return`                 | 黄金周对数收益                        | FRED GOLDPMGBD228NLBM / Yahoo GC=F           | 无量纲      | 日→周→对数收益  | 0    | 99.9%  | 2006-01-13 |
| `global_econ_activity`        | 全球实体经济活动指数（Kilian REA）         | Dallas Fed (igrea)                           | 指数（去趋势%） | 月→周 ffill | +5 周 | 99.5%  | 2006-02-10 |
| `nonoil_industrial_commodity` | 非燃料工业原料价格指数                    | FRED PINDUINDEXM                             | 指数       | 月→周 ffill | +5 周 | 99.1%  | 2006-03-10 |
| `brent_f1_spot_log_basis`     | Brent 前月期货—现货对数基差（basis proxy） | Yahoo BZ=F vs `brent_price`（EIA）             | 无量纲      | 日→周最后值→差分 | 0    | 92.0%  | 2007-08-03 |
| `brent_roll_week`             | Brent 前月连续合约换月周哑变量             | 派生（日历月末最后 W-FRI）                             | {0,1}    | —         | 0    | 100.0% | 2006-01-06 |
| `cadusd_log_return`           | CAD/USD 周对数收益（油价敏感商品货币）        | Yahoo CADUSD=X                               | 无量纲      | 日→周→对数收益  | 0    | 99.9%  | 2006-01-13 |
| `dgs10_change`                | 10 年美债收益率周变化                   | 派生（`treasury_10y`）                           | 百分点      | —         | 0    | 99.9%  | 2006-01-13 |




#### 4.4.1 变量选择依据：文献与业界支撑

> §4.4 的 9 个衍生市场/宏观变量对应 Kilian 的供给—总需求—预防性需求机制及跨资产渠道。沿用诚实边界：多数为**结构模型（SVAR）、溢出模型或月频**证据，用作变量选择与构造依据，增量价值由 SHAP + 消融 + DM/CW 判定。`cadusd_log_return`（由原 `commodity_fx` 精简为 CAD 单腿）支撑文献为 Chen, Rogoff & Rossi (2010, *QJE*) [P135]。`brent_f1_spot_log_basis` / `brent_roll_week` 的口径分层见 §4.4.2。完整学术引用见「参考来源」（`[P052]`/`[P122]`/`[P124]`/`[P131]`–`[P135]` 已并入主文献表 `01_literature/literature_matrix.md` §①-e；`[R001]`–`[R021]` 已并入主文献表 `01_literature/literatue.md` §一-b（Report 层））。


| 变量                            | 代表性学术文献                                              | 文献中的用法（含口径提示）                                                                                                                          | 代表性业界/机构材料                                                                                                                  | 支撑程度                                                                                                                                 |
| ----------------------------- | ---------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------ |
| `ovx`                         | Tissaoui et al. (2023) [P122]                        | OVX 作为油市专属不确定性指标，是 XGBoost + SHAP 中的重要预测信息                                                                                             | Cboe OVX：以原油 ETF 期权、VIX 式方法计算的原油隐含波动率 [R018]                                                                                | 非常强（油市专属性 > VIX）                                                                                                                     |
| `gpr`                         | Caldara & Iacoviello (2022) [P131]                   | 基于新闻文本的地缘政治风险指数，广泛用于预测油价/波动                                                                                                            | GPR 官方数据页（月度/日度序列）[R021]                                                                                                    | 很强（须按真实发布时间滞后，勿按月份标签回填）                                                                                                              |
| `gold_return`                 | Kang, McIver & Yoon (2017) [P132]                    | 黄金—原油收益/波动溢出（DECO-GARCH + 溢出指数）                                                                                                        | 跨资产交易台共同观察金/油（避险、通胀、美元代理）[R016]                                                                                             | 中等（跨资产扩展变量，优先级低于 OVX/美元/期限结构）                                                                                                        |
| `global_econ_activity`        | Kilian (2009) [P052]；Kilian & Zhou (2018) [P133]     | Kilian 全球实体经济活动指数（干散货运价）衡量全球工业商品需求，且被论证优于 GDP/IP 代理                                                                                    | Dallas Fed 维护修正版 REA 指数（igrea）[R019]                                                                                        | 非常强（月度慢变的全球需求状态，而非短期冲击）                                                                                                              |
| `nonoil_industrial_commodity` | Kilian & Zhou (2018) [P133]                          | 工业商品价格/需求可作全球实体活动代理，用于建模工业商品市场                                                                                                         | IMF Global price of Industrial Materials（FRED `PINDUINDEXM`）[R020]                                                          | 中等偏强（可能与 REA、美元、铜价高相关）                                                                                                               |
| `brent_f1_spot_log_basis`     | Valenti (2022) [P134]；Scheitrum et al. (2018) [P112] | Valenti：Brent **期货—现货价差**进入 SVAR，替代库存代理、捕捉 price discovery 与信息摩擦（原文为 **3 个月** 合约，非前月）。Scheitrum：严格期限结构为同市场不同到期月份价差（如 F₄−F₁），**非**现货减前月 | EIA 区分 Dated Brent 现货与 Brent 前月期货并持续分析二者 spread [R001]；行业亦用现货—近月关系讨论即时松紧，但 contango/backwardation 更常指 F₁−F₂ 等同市场日历价差 [R016] | **概念强**；本字段为 **public-data implementation of a Brent front-month futures–spot basis proxy**（免费数据工程实现，非 ICE 标准 basis；构造细节见 §4.4.2、§6） |
| `brent_roll_week`             | —（方法学推论）                                             | 连续期货换月产生机械断点（rollover discontinuity）；显式识别换月周是常见控制策略，但**单一哑变量通常只能吸收换月周平均水平差异**，未必消除全部断点或方差变化                                            | ICE Brent 前月合约按交割日历在月末附近切换；CME 教育材料亦强调期货合约到期/换月机制 [R011]                                                                    | **方法学支撑充分**；属**控制变量**而非完整解决方案（稳健性见 §4.4.2、§6）                                                                                        |
| `cadusd_log_return`           | Chen, Rogoff & Rossi (2010) [P135]                   | 商品货币汇率对全球商品价格有稳健的样本内外预测力，因汇率具前瞻性；CAD 最直接暴露于原油（加拿大产油国）                                                                                  | CAD 被视为油价敏感货币；CME 讨论油价与商品货币的联系 [R016]                                                                                       | 经济逻辑成立；本项目采 CAD 单腿对数收益，AUD（与 CAD 0.79 相关、油价关联更弱）删除主表、留稳健性（research diary 2026-07-03）                                                 |
| `dgs10_change`                | Qadan & Cohen (2024) [P124]                          | 利率变化/利率不确定性与油价收益、波动相关                                                                                                                  | 市场关注收益率"变化多少个基点"而非绝对水平 [R016]                                                                                               | 较强（是否与 `treasury_10y` 水平同时保留取决于模型）                                                                                                   |


**参考来源（§4.4 学术文献 → 主文献表 P052/P122/P124/P131–P135；业界材料 → §4.6 R001–R021）：**

*学术文献（详见* `01_literature/literature_matrix.md` *§①-e）*

- [P122] Tissaoui, K., Zaghdoudi, T., Hakimi, A. & Nsaibi, M. (2023). Do gas price and uncertainty indices forecast crude oil prices? Fresh evidence through XGBoost modeling. *Computational Economics*, 62(2), 663–687. [https://doi.org/10.1007/s10614-022-10305-y](https://doi.org/10.1007/s10614-022-10305-y)
- [P131] Caldara, D. & Iacoviello, M. (2022). Measuring geopolitical risk. *American Economic Review*, 112(4), 1194–1225. [https://doi.org/10.1257/aer.20191823](https://doi.org/10.1257/aer.20191823)
- [P132] Kang, S.H., McIver, R. & Yoon, S.-M. (2017). Dynamic spillover effects among crude oil, precious metal, and agricultural commodity futures markets. *Energy Economics*, 62, 19–32. [https://doi.org/10.1016/j.eneco.2016.12.011](https://doi.org/10.1016/j.eneco.2016.12.011)
- [P052] Kilian, L. (2009). Not all oil price shocks are alike: Disentangling demand and supply shocks in the crude oil market. *American Economic Review*, 99(3), 1053–1069. [https://doi.org/10.1257/aer.99.3.1053](https://doi.org/10.1257/aer.99.3.1053)
- [P133] Kilian, L. & Zhou, X. (2018). Modeling fluctuations in the global demand for commodities. *Journal of International Money and Finance*, 88, 54–78. [https://doi.org/10.1016/j.jimonfin.2018.05.006](https://doi.org/10.1016/j.jimonfin.2018.05.006)
- [P134] Valenti, D. (2022). Modelling the global price of oil: Is there any role for the oil futures-spot spread? *The Energy Journal*, 43(2), 41–66. [https://doi.org/10.5547/01956574.43.2.dval](https://doi.org/10.5547/01956574.43.2.dval)
- [P135] Chen, Y.-C., Rogoff, K.S. & Rossi, B. (2010). Can exchange rates forecast commodity prices? *Quarterly Journal of Economics*, 125(3), 1145–1194（NBER WP 13901, 2008）. [https://doi.org/10.1162/qjec.2010.125.3.1145](https://doi.org/10.1162/qjec.2010.125.3.1145)
- [P124] Qadan, M. & Cohen, G. (2024). Uncertainty about interest rates and crude oil prices. *Financial Innovation*, 10, 9. [https://doi.org/10.1186/s40854-023-00551-w](https://doi.org/10.1186/s40854-023-00551-w)

> 业界/机构材料见 §4.6 `[R001]`–`[R021]`。



#### 4.4.2 `brent_f1_spot_log_basis` 与 `brent_roll_week`：定义分层与免费数据工程处理

> **总判断**：由旧名 `futures_spread` 正名为 `brent_f1_spot_log_basis`，并配套 `brent_roll_week`，整体有文献与行业基础；但须区分「**直接受文献支持的定义**」与「**为免费公开数据作出的工程处理**」。论文中宜称为 *public-data implementation of a Brent front-month futures–spot basis proxy*，而非声称其为标准、无误差的 ICE/EIA 官方 Brent basis。

**A. 正名为 basis（直接支撑）**

$$
\text{brentf1spotlogbasis}*t = \ln(F*{1,t}^{\text{Brent}}) - \ln(S_t^{\text{Brent}})
$$

- 这是典型的 **front-month futures–spot basis**（前月期货减现货）。EIA 明确区分 Brent 前月期货与 Dated Brent 现货市场，并持续分析二者之间的 spread [R001]；前月合约在到期后切换至下一交割月份亦为行业常规定义 [R011]。
- **符号约定**：本字段采用 **F₁ − Spot**（非部分文献的 S − F）。**正值** = 前月期货高于现货；**负值** = 前月期货低于现货。
- 周频构造：期货腿（Yahoo `BZ=F`）与现货腿（`brent_price`，EIA）均先取该周 **W-FRI 最后值**，再取对数差。

**B. basis ≠ 纯期限结构（直接支撑）**

- 严格 **期货期限结构（term structure）** 描述同一期货市场中不同到期月份价格之间的关系（F₁, F₂, F₃, …），例如 Scheitrum et al. (2018) [P112] 将某种原油的 inter-temporal spread 定义为 **F₄ − F₁**，而非现货减前月。
- 因此应表述为：`brent_f1_spot_log_basis` captures the relative pricing of the Brent front-month futures contract and the Brent spot benchmark — a **futures–spot basis proxy**, **not** a pure calendar-spread measure of the Brent futures term structure。
- 行业评论有时亦用现货—前月关系讨论即时供应松紧；但更严格的 contango/backwardation 与 curve structure 通常使用 F₁−F₂、F₁−F₆ 等同市场合约价差 [R016]。故写作「**不是纯期限结构指标**」，而非「与期限结构完全无关」。

**C. 使用未 back-adjust 期货价格（方法论推论，有依据）**

- 期货合约到期即终止；将不同合约拼成连续前月序列会在换月点产生价格不连续。连续期货的拼接与 back-adjustment 会**人为改变历史价格水平**，进而改变所计算的现金—期货关系（方法论上见连续合约构造文献的一般讨论；由此推论：若研究目标是**当时实际可观测**的 F₁−S，则 basis 应使用**未 back-adjust** 的原始可观测价格）。
- 若使用 back-adjusted 价格 F₁,ₜ^adj = F₁,ₜ + Aₜ，则 F₁,ₜ^adj − Sₜ 含调整项 Aₜ，**不再是**当时市场真实价差。本实现 deliberately 不对 `BZ=F` 做 back-adjustment。

**D.** `brent_roll_week`**（方法学支撑，但为控制变量）**

- 定义：该周 Brent 前月连续合约标的发生切换时取 1，否则 0。实现上近似为**每个自然月的最后一个 W-FRI**（对 ICE 精确到期日历的 documented approximation）。
- 连续期货换月可能产生机械跳跃，显式识别 rollover week 有充分方法学依据；但**不应声称单一 dummy 完全解决换月问题**——不同合约间价差可随时间变化，哑变量通常仅控制换月周的平均水平差异。
- **建议稳健性检验**（建模阶段，非本字典范围）：
  1. 删除所有 `brent_roll_week = 1` 的观测；
  2. 删除换月周前后各一周；
  3. 比较模型 A（无 basis）、B（basis + roll dummy）、C（basis，但删除换月周）：若 B 与 C 均优于 A 且结论接近，可更有把握认为 basis 捕捉的是经济信号而非拼接伪影。

**E. 免费数据实现的已知局限（诚实边界）**

- 期货腿为 Yahoo `BZ=F`（ICE Brent 前月连续合约近似），现货腿为 EIA Dated Brent（`brent_price`）；**数据来源、定价时点与收盘时间并不完全一致**。
- Valenti [P134] 使用的是 **3 个月** Brent 期货—现货价差；本字段为**前月** proxy，经济含义相近但非同一合约。
- 旧 CSV 若仍显示列名 `futures_spread`，系重建前遗留；重建后列名与本文档一致。



### 4.6 业界/机构材料（Report · R001–R021）

> 自 §4.1–§4.4 局部编号 `[b*]` 去重并入主文献表；与学术论文 `[Pxxx]` 分层管理。完整登记表亦见 `01_literature/literature_matrix.md` §①-f。


| ID       | 机构/来源                | 材料说明                                                                         | URL                                                                                                                                                                                                                                                                                                                                                                                                                                                         |
| -------- | -------------------- | ---------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **R001** | U.S. EIA             | Spot Prices for Crude Oil and Petroleum Products（Brent–Europe / WTI–Cushing） | [https://www.eia.gov/dnav/pet/PET_PRI_SPT_S1_D.htm](https://www.eia.gov/dnav/pet/PET_PRI_SPT_S1_D.htm)                                                                                                                                                                                                                                                                                                                                                      |
| **R002** | U.S. EIA             | Today in Energy — *Widening Brent–WTI price spread*                          | [https://www.eia.gov/todayinenergy/detail.php?id=33572](https://www.eia.gov/todayinenergy/detail.php?id=33572)                                                                                                                                                                                                                                                                                                                                              |
| **R003** | U.S. EIA             | Today in Energy — Daily Prices                                               | [https://www.eia.gov/todayinenergy/prices.php](https://www.eia.gov/todayinenergy/prices.php)                                                                                                                                                                                                                                                                                                                                                                |
| **R004** | OPEC                 | Monthly Oil Market Report（ICE Brent 月环比等）                                    | [https://www.opec.org/opec_web/en/publications/338.htm](https://www.opec.org/opec_web/en/publications/338.htm)                                                                                                                                                                                                                                                                                                                                              |
| **R005** | CME Group            | WTI Insights newsletter（全球基准 / 周表现）                                          | [https://www.cmegroup.com/newsletters/wti-insights-by-pvm/wti-insights-by-pvm-2024-05-14.html](https://www.cmegroup.com/newsletters/wti-insights-by-pvm/wti-insights-by-pvm-2024-05-14.html)                                                                                                                                                                                                                                                                |
| **R006** | ICE                  | ICE Risk Model 2.0 Methodology（return risk factors / VaR）                    | [https://www.ice.com/clear-us/ice-risk-model/methodology](https://www.ice.com/clear-us/ice-risk-model/methodology)                                                                                                                                                                                                                                                                                                                                          |
| **R007** | U.S. EIA             | Weekly Petroleum Status Report（WPSR）                                         | [https://www.eia.gov/petroleum/supply/weekly/](https://www.eia.gov/petroleum/supply/weekly/)                                                                                                                                                                                                                                                                                                                                                                |
| **R008** | U.S. EIA             | Today in Energy — WPSR provides comprehensive balances                       | [https://www.eia.gov/todayinenergy/detail.php?id=3371](https://www.eia.gov/todayinenergy/detail.php?id=3371)                                                                                                                                                                                                                                                                                                                                                |
| **R009** | U.S. EIA             | FAQ — How do we calculate product supplied?                                  | [https://www.eia.gov/tools/faqs/faq.php?id=1394](https://www.eia.gov/tools/faqs/faq.php?id=1394)                                                                                                                                                                                                                                                                                                                                                            |
| **R010** | U.S. EIA             | Weekly U.S. Product Supplied of Distillate Fuel Oil                          | [https://www.eia.gov/dnav/pet/hist/LeafHandler.ashx?f=w&n=pet&s=wdiupus2](https://www.eia.gov/dnav/pet/hist/LeafHandler.ashx?f=w&n=pet&s=wdiupus2)                                                                                                                                                                                                                                                                                                          |
| **R011** | CME Group            | Delivery of WTI futures（Cushing 交割）                                          | [https://www.cmegroup.com/education/courses/introduction-to-crude-oil/crude-oil-fundamentals/delivery-of-wti-futures.html](https://www.cmegroup.com/education/courses/introduction-to-crude-oil/crude-oil-fundamentals/delivery-of-wti-futures.html)                                                                                                                                                                                                        |
| **R012** | CME Group            | Econoday — EIA Petroleum Status Report 日历/解读                                 | [https://www.cmegroup.com/education/events/econoday/672390](https://www.cmegroup.com/education/events/econoday/672390)                                                                                                                                                                                                                                                                                                                                      |
| **R013** | Reuters              | EIA 周度库存/进出口/炼厂/成品油报道（例证 2025-06-25、2025-07-23）                              | [https://www.reuters.com/business/energy/us-crude-fuel-inventories-fall-higher-demand-eia-says-2025-06-25/](https://www.reuters.com/business/energy/us-crude-fuel-inventories-fall-higher-demand-eia-says-2025-06-25/)；[https://www.reuters.com/business/energy/us-crude-gasoline-stockpiles-fall-distillates-build-eia-says-2025-07-23/](https://www.reuters.com/business/energy/us-crude-gasoline-stockpiles-fall-distillates-build-eia-says-2025-07-23/) |
| **R014** | Cboe                 | VIX Index（S&P 500 期权隐含波动率）                                                   | [https://www.cboe.com/tradable-products/vix/](https://www.cboe.com/tradable-products/vix/)                                                                                                                                                                                                                                                                                                                                                                  |
| **R015** | U.S. EIA             | Markets & Finance（原油美元计价 / 汇率影响）                                             | [https://www.eia.gov/finance/](https://www.eia.gov/finance/)                                                                                                                                                                                                                                                                                                                                                                                                |
| **R016** | CME Group            | Economic Data and Crude Oil（利率/美元/股市/跨资产）                                    | [https://www.cmegroup.com/education/courses/introduction-to-crude-oil/crude-oil-fundamentals/economic-data-and-crude-oil.html](https://www.cmegroup.com/education/courses/introduction-to-crude-oil/crude-oil-fundamentals/economic-data-and-crude-oil.html)                                                                                                                                                                                                |
| **R017** | MSCI / RiskMetrics   | RiskMetrics Technical Document（1996）— log returns / VaR 基础                   | [https://www.msci.com/documents/10199/dbb975aa-5dc2-4441-aa2d-ae34ab5f0945](https://www.msci.com/documents/10199/dbb975aa-5dc2-4441-aa2d-ae34ab5f0945)                                                                                                                                                                                                                                                                                                      |
| **R018** | Cboe                 | OVX — Crude Oil Volatility Index                                             | [https://www.cboe.com/us/indices/dashboard/OVX/](https://www.cboe.com/us/indices/dashboard/OVX/)                                                                                                                                                                                                                                                                                                                                                            |
| **R019** | Dallas Fed           | Index of Global Real Economic Activity（igrea）                                | [https://www.dallasfed.org/research/igrea](https://www.dallasfed.org/research/igrea)                                                                                                                                                                                                                                                                                                                                                                        |
| **R020** | IMF                  | Primary Commodity Prices（Industrial Materials → FRED `PINDUINDEXM`）          | [https://www.imf.org/en/Research/commodity-prices](https://www.imf.org/en/Research/commodity-prices)                                                                                                                                                                                                                                                                                                                                                        |
| **R021** | Caldara & Iacoviello | GPR Index **数据页**（≠ 学术论文 P131）                                               | [https://www.matteoiacoviello.com/gpr.htm](https://www.matteoiacoviello.com/gpr.htm)                                                                                                                                                                                                                                                                                                                                                                        |


> **查重说明（2026-07-03）：** 21 条 Report 按**唯一 URL/出版物**去重（原 §4.3/§4.4 的 CME Economic Data 合并为 **R016**）。学术论文 **P001–P135** 按 DOI 查重：**无重复编号**；跨节复用 **P052 / P122 / P124** 为有意交叉引用。`reading_notes/Template/` 模板与正式 P 编号 DOI 相同，未另占编号。**R021** 为 GPR 数据页，与学术论文 **P131** 分层不重复。



### 4.5 模态可用性标志（4 列）


| 字段                   | 含义                                   | 来源  | 单位    | 覆盖率    | 首个有效周      |
| -------------------- | ------------------------------------ | --- | ----- | ------ | ---------- |
| `avail_market`       | 价格模态可用（`brent_price` 非空）             | 派生  | {0,1} | 100.0% | 2006-01-06 |
| `avail_eia_weekly`   | EIA 周报可用（`crude_stocks_excl_spr` 非空） | 派生  | {0,1} | 100.0% | 2006-01-06 |
| `avail_sp500`        | 标普 500 可用                            | 派生  | {0,1} | 100.0% | 2006-01-06 |
| `avail_dollar_index` | 美元指数可用                               | 派生  | {0,1} | 100.0% | 2006-01-06 |


> 注：`avail_eia_weekly` 在 2006-01-06 取值为 0（该周 EIA 报告尚未发布），其余周为 1。

---



## 5. 派生变量公式


| 字段                        | 公式                                                                  |
| ------------------------- | ------------------------------------------------------------------- |
| `brent_log_return`        | `ln(Pₜ / Pₜ₋₁)`；训练目标为下一周值 `rₜ₊₁ = ln(Pₜ₊₁/Pₜ)`                      |
| `wti_log_return`          | `ln(wtiₜ / wtiₜ₋₁)`（与 Brent 同口径，不乘 100）                             |
| `brent_wti_spread`        | `brent_price − wti_price`                                           |
| `crude_stocks_change`     | `diff(crude_stocks_excl_spr)`                                       |
| `cushing_stocks_change`   | `diff(cushing_stocks)`                                              |
| `sp500_log_return`        | `ln(sp500ₜ / sp500ₜ₋₁)`（与 Brent/WTI 同口径，不乘 100）                     |
| `gold_return`             | `ln(goldₜ / goldₜ₋₁)`                                               |
| `brent_f1_spot_log_basis` | `ln(BZ=F front-month) − ln(brent_price)`；**未 back-adjust**；见 §4.4.2 |
| `brent_roll_week`         | 每月最后一个 W-FRI 取 1，否则 0（换月控制哑变量）                                      |
| `cadusd_log_return`       | `ln(CADUSDₜ / CADUSDₜ₋₁)`（CAD/USD 对数收益；AUD 腿已删，见 research diary）    |
| `dgs10_change`            | `diff(treasury_10y)`                                                |


---



## 6. 备注与已知缺口

- **早期缺失（均在 2019–2026 比较窗口之外）**：
  - `ovx` 自 2007-05 起（OVX 指数 2007 年推出）。
  - `brent_f1_spot_log_basis` 自 2007-08 起（Brent 前月期货序列 `BZ=F` 起步较晚）；`brent_roll_week` 自 2006-01-06 起全覆盖（日历派生，无外部数据缺口）。
  - `global_econ_activity` / `nonoil_industrial_commodity` 因 +5 周保守滞后，前几周为 NaN。
- 派生 `*_change` / `*_log_return` / `*_return` 列首周为 NaN（差分/对数收益需要前一周值）。
- **对数收益口径统一**：`brent_log_return` / `wti_log_return` / `sp500_log_return` / `gold_return` / `cadusd_log_return` 均为 `ln(Pₜ/Pₜ₋₁)`，不乘 100；与 Sadorsky (1999) [P127]、Roy et al. (2023) [P130] 等文献及 RiskMetrics 惯例一致。
- **GPR 改用日度 GPRD**：`gpr` 由 Caldara–Iacoviello **日度 Recent GPR（GPRD）** 周均值聚合到 W-FRI 后滞后一周（原为月度值 ffill + 滞后）。日度→周频匹配周度预测节奏，避免月度值「按月份标签回填」的潜在前视；滞后一周确保数据在预测周五已发布（日度 GPRD 有 1–2 天发布延迟）。原始文件 `Other/data_gpr_daily_recent.xls`（官网每日更新）。
- `commodity_fx` **精简为** `cadusd_log_return`：原 `mean(ΔCAD, ΔAUD)` 复合指标改为 **CAD/USD 单腿对数收益**。CAD 与油价相关（0.36 全样本）强于 AUD（0.28），且 CAD/AUD 相关 0.74–0.79（AUD 增量有限、更多反映铁矿/中国/风险偏好）；AUD 腿从主表删除，作为「商品货币选择稳健性」备选（暂不入库，见 research diary 2026-07-03）。
- **负价格护栏（对数收益）**：`brent_log_return` / `wti_log_return` 按「日频→每周五最后值→再取对数收益」顺序计算，计算前 `assert price>0`。2020-04-20 WTI 日内负结算价（−36.98 美元/桶）落在周一，不构成任何周五代表价（周五 WTI 序列最低仅 15.48，全为正），故不影响周频对数收益；断言可防未来数据更新/换源引入负周五价导致 `ln` 无效。`sp500_log_return` 由周频指数点位直接取对数差分，S&P 500 恒为正，无此风险。
- `crude_exports` **的制度性结构断裂（2015 出口禁令取消，需谨慎解读）**：该列覆盖率达 99.9%（自 2006-01-13 起，见 §4.2），属**制度性结构变化（regime break）**而非数据缺失：
  - **制度背景**：美国自 1975 年《能源政策与保护法》(EPCA) 起长期禁止大部分原油出口，直至 **2015-12-18** 国会取消禁令。故 2015 年前 `crude_exports` 长期处于极低水平（基本仅对加拿大等豁免出口），2016 年起随出口自由化快速抬升，序列在 2015/2016 前后发生量级跃迁；`crude_imports` 与 `crude_exports` 分列保留，不再派生净贸易项。
  - **对建模的含义**：禁令期低位取值主要反映**政策约束**而非市场供需/套利信号，边际预测信息有限；2016 年后 `crude_exports` 才更充分承载美国出口能力与跨市场套利信息。本项目统一比较窗口 2019–2025 完全落在取消后的「新制度」内，训练/评估不受早期断点直接干扰；但若纳入 2016 年前样本或做全样本解释（如 SHAP 归因），应显式处理该断点（2016 断点子样本 / 结构断点稳健性 / regime 哑变量）。
  - **文献联系（出口限制 ↔ WTI–Brent 价格重新整合）**：出口禁令使美国原油无法经跨大西洋空间套利外流，2010 年底/2011 年初起 WTI 相对 Brent 出现结构性折价与协整断裂（叠加页岩油增产与 Cushing 库存/管输瓶颈）；禁令取消后空间套利恢复、`brent_wti_spread` 收窄、两序列重新整合（Scheitrum, Carter & Revoredo-Giha, 2018 = §4.1 [P112]；Bravo Caro et al., 2020 = §4.1 [P113]）。价差物理/纸面市场机制见 **P102** [Büyükşahin et al., 2013]；价差成因分解（国内运输约束 vs 出口禁令）见 **P103** [Agerton & Upton, 2019]（均已并入主文献表 `01_literature/literatue.md`）。**Caveat**：主流证据支持取消后价差收窄与重新整合，但受取消后可用样本较短限制，正式协整是否稳定恢复的证据部分仍属 mixed。故 `crude_exports` 与 `brent_wti_spread` 在解释上高度关联，跨变量归因需注意共线性。
- `brent_f1_spot_log_basis` **/** `brent_roll_week`**（Brent 前月期货—现货 basis proxy + 换月控制）**：
  - **命名**：由原 `futures_spread` 正名，以准确反映 front-month futures–spot basis 含义；完整定义分层见 §4.4.2。
  - **文献—工程边界**：学术上 Valenti (2022) [P134] 支持「Brent 期货—现货价差」作为 SVAR 信息变量；EIA [R001] 支持现货与前月期货 spread 的行业分析口径；Scheitrum et al. (2018) [P112] 支持「期限结构 ≠ 现货−前月」的区分。本实现为免费公开数据（Yahoo `BZ=F` + EIA 现货）下的 **basis proxy**，非 ICE 官方 basis。
  - **未 back-adjust**：为保留当时可观测的 F₁−S，期货腿不做 back-adjustment（方法论推论，见 §4.4.2-C）。
  - **换月**：`brent_roll_week` 吸收连续合约换月的机械断点，但单一哑变量不充分；建模时应配合删换月周 / 删前后各一周 / A–B–C 模型对比等稳健性（§4.4.2-D）。
  - **数据局限**：Yahoo 前月连续合约与 EIA 现货在时点与定价基准上不完全一致；论文表述应避免过度声称「标准 Brent basis」。
- **预测目标不在本矩阵内**：唯一核心目标为下一周 Brent 价格 `Pₜ₊₁`；训练用 `rₜ₊₁=ln(Pₜ₊₁/Pₜ)`，输出还原 `P̂ₜ₊₁=Pₜ·e^(r̂)`。目标/标签需在建模脚本中由 `brent_price` **前瞻一周**生成，本矩阵只含特征、不含未来标签（避免泄漏）。

---



## 7. 复现命令

```bash
cd 03_data/processed/M1/py
python build_m1_weekly.py            # 离线，从本地 raw 构建（默认）
python build_m1_weekly.py --online   # 缺失源时在线补抓（FRED/Yahoo）
python build_m1_weekly.py --base-only # 仅基础块，跳过 9 个衍生变量
```

