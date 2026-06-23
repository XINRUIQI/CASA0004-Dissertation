# M1 金融/宏观特征矩阵 — 数据字典

> 对应数据文件：`03_data/processed/M1/outputs/m1_weekly_features.csv`
> 构建脚本：`03_data/processed/M1/py/build_m1_weekly.py`
> 版本：防泄漏修订版（EIA 周报已加 +1 周发布滞后）
> 最后更新：2026-06-23

---

## 1. 基本信息


| 项      | 值                                              |
| ------ | ---------------------------------------------- |
| 时间索引   | `week_ending_friday`，每周五截止（W-FRI）              |
| 时间范围   | 2006-01-06 ~ 2025-12-26                        |
| 形状     | 1043 行 × 38 列                                  |
| 统一比较窗口 | 2019-01-04 ~ 2025-12-26（365 周），**窗口内 38 列零缺失** |
| 缺失约定   | 空单元格 = 该周该变量尚不可得（NaN），不做隐式填补                   |
| 频率     | 周频（所有模态统一对齐到周五）                                |


---

## 2. 构建脚本做了什么（`build_m1_weekly.py`）

**定位**：离线优先的「单脚本单表」构建器——把原先三步流水线（`build_weekly_time_index.py` + `build_m1_to_build.py` + `merge_m1_to_build.py`）合并为一个脚本，直接从按供应商组织的 raw 层读取，输出唯一一张周频特征表（无中间 CSV、无单独 merge 步骤）。

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
2. **基础块 `build_base()`**：
  - 价格：EIA Brent/WTI 日频 → 每周五最后值；派生 `brent_return_pct / brent_log_return / brent_direction / wti_return_pct / brent_wti_spread`。
  - EIA 周报：按 `WPSR_MAP` 逐个读入 10 个 WPSR 字段 → 对齐到报告周五 → +1 周发布滞后；派生 `crude_stocks_change / cushing_stocks_change / net_crude_trade`。
  - FRED 宏观日频（VIX / 美元指数 / 10Y / 联邦基金）→ 每周五最后值。
  - Yahoo 标普 500 → 每周五最后值；派生 `sp500_return_pct`。
3. **衍生块 `TO_BUILD`**（除非 `--base-only`）：逐个构建 8 个变量 `ovx / gpr / gold_return / global_econ_activity / nonoil_industrial_commodity / futures_spread / commodity_fx / dgs10_change`；单个失败仅跳过并记录警告，不中断整体。
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
- `**--base-only**`：只建基础块，跳过 8 个衍生变量。
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

---

## 4. 字段说明

> 覆盖率为全样本（1043 周）非空占比；首个有效周为该列第一个非 NaN 的周五。

### 4.1 价格与派生（7 列）


| 字段                 | 含义                              | 来源                      | 单位       | 周频化     | 滞后  | 覆盖率    | 首个有效周      |
| ------------------ | ------------------------------- | ----------------------- | -------- | ------- | --- | ------ | ---------- |
| `brent_price`      | Brent 现货价（**预测目标基于此**）          | EIA 日频现货 (DCOILBRENTEU) | 美元/桶     | 日→周五最后值 | 0   | 100.0% | 2006-01-06 |
| `wti_price`        | WTI(Cushing) 现货价                | EIA 日频现货 (DCOILWTICO)   | 美元/桶     | 日→周五最后值 | 0   | 100.0% | 2006-01-06 |
| `brent_return_pct` | Brent 周环比收益率                    | 派生                      | %        | —       | 0   | 99.9%  | 2006-01-13 |
| `brent_log_return` | Brent 周对数收益（其下一周值 r₍ₜ₊₁₎ 为训练目标） | 派生                      | 无量纲      | —       | 0   | 99.9%  | 2006-01-13 |
| `brent_direction`  | Brent 周方向（涨/平/跌，阈值 ±0.5%）       | 派生                      | {-1,0,1} | —       | 0   | 100.0% | 2006-01-06 |
| `wti_return_pct`   | WTI 周环比收益率                      | 派生                      | %        | —       | 0   | 99.9%  | 2006-01-13 |
| `brent_wti_spread` | Brent − WTI 价差                  | 派生                      | 美元/桶     | —       | 0   | 100.0% | 2006-01-06 |


### 4.2 EIA 周报基本面（13 列，均 +1 周发布滞后）


| 字段                      | 含义                      | 来源       | 单位   | 周频化    | 滞后   | 覆盖率   | 首个有效周      |
| ----------------------- | ----------------------- | -------- | ---- | ------ | ---- | ----- | ---------- |
| `crude_stocks_excl_spr` | 商业原油库存（不含 SPR）          | EIA WPSR | 千桶   | 周→周五对齐 | +1 周 | 99.9% | 2006-01-13 |
| `cushing_stocks`        | Cushing 原油库存            | EIA WPSR | 千桶   | 周→周五对齐 | +1 周 | 99.9% | 2006-01-13 |
| `crude_production`      | 美国国内原油产量                | EIA WPSR | 千桶/日 | 周→周五对齐 | +1 周 | 99.9% | 2006-01-13 |
| `crude_imports`         | 原油进口                    | EIA WPSR | 千桶/日 | 周→周五对齐 | +1 周 | 99.9% | 2006-01-13 |
| `crude_exports`         | 原油出口                    | EIA WPSR | 千桶/日 | 周→周五对齐 | +1 周 | 99.9% | 2006-01-13 |
| `refinery_crude_input`  | 炼厂原油加工量                 | EIA WPSR | 千桶/日 | 周→周五对齐 | +1 周 | 99.9% | 2006-01-13 |
| `refinery_utilisation`  | 炼厂开工率                   | EIA WPSR | %    | 周→周五对齐 | +1 周 | 99.9% | 2006-01-13 |
| `gasoline_supplied`     | 汽油消费量（product supplied） | EIA WPSR | 千桶/日 | 周→周五对齐 | +1 周 | 99.9% | 2006-01-13 |
| `distillate_supplied`   | 馏分油消费量                  | EIA WPSR | 千桶/日 | 周→周五对齐 | +1 周 | 99.9% | 2006-01-13 |
| `jet_fuel_supplied`     | 航空煤油消费量                 | EIA WPSR | 千桶/日 | 周→周五对齐 | +1 周 | 99.9% | 2006-01-13 |
| `crude_stocks_change`   | 商业原油库存周变化               | 派生 diff  | 千桶   | —      | +1 周 | 99.8% | 2006-01-20 |
| `cushing_stocks_change` | Cushing 库存周变化           | 派生 diff  | 千桶   | —      | +1 周 | 99.8% | 2006-01-20 |
| `net_crude_trade`       | 净原油贸易（进口 − 出口）          | 派生       | 千桶/日 | —      | +1 周 | 99.9% | 2006-01-13 |


### 4.3 宏观金融（6 列）


| 字段                 | 含义          | 来源            | 单位  | 周频化     | 滞后  | 覆盖率    | 首个有效周      |
| ------------------ | ----------- | ------------- | --- | ------- | --- | ------ | ---------- |
| `vix`              | CBOE 波动率指数  | FRED VIXCLS   | 指数  | 日→周五最后值 | 0   | 100.0% | 2006-01-06 |
| `dollar_index`     | 美元名义广义指数    | FRED DTWEXBGS | 指数  | 日→周五最后值 | 0   | 100.0% | 2006-01-06 |
| `treasury_10y`     | 10 年期美债收益率  | FRED DGS10    | %   | 日→周五最后值 | 0   | 100.0% | 2006-01-06 |
| `fed_funds_rate`   | 联邦基金有效利率    | FRED DFF      | %   | 日→周五最后值 | 0   | 100.0% | 2006-01-06 |
| `sp500`            | 标普 500 指数   | Yahoo ^GSPC   | 指数点 | 日→周五最后值 | 0   | 100.0% | 2006-01-06 |
| `sp500_return_pct` | 标普 500 周收益率 | 派生            | %   | —       | 0   | 99.9%  | 2006-01-13 |


### 4.4 衍生市场/宏观变量（8 列，脚本中的 to-build 块）


| 字段                            | 含义                        | 来源                                     | 单位       | 周频化       | 滞后   | 覆盖率   | 首个有效周      |
| ----------------------------- | ------------------------- | -------------------------------------- | -------- | --------- | ---- | ----- | ---------- |
| `ovx`                         | 原油波动率指数                   | Yahoo ^OVX / FRED OVXCLS               | 指数       | 日→周五最后值   | 0    | 93.3% | 2007-05-11 |
| `gpr`                         | 地缘政治风险指数                  | Caldara & Iacoviello (data_gpr_export) | 指数       | 月→周 ffill | +1 周 | 99.9% | 2006-01-13 |
| `gold_return`                 | 黄金周对数收益                   | FRED GOLDPMGBD228NLBM / Yahoo GC=F     | 无量纲      | 日→周→对数收益  | 0    | 99.9% | 2006-01-13 |
| `global_econ_activity`        | 全球实体经济活动指数（Kilian REA）    | Dallas Fed (igrea)                     | 指数（去趋势%） | 月→周 ffill | +5 周 | 99.5% | 2006-02-10 |
| `nonoil_industrial_commodity` | 非燃料工业原料价格指数               | FRED PINDUINDEXM                       | 指数       | 月→周 ffill | +5 周 | 99.1% | 2006-03-10 |
| `futures_spread`              | Brent 期货−现货对数价差（期限结构）     | Yahoo BZ=F vs `brent_price`            | 无量纲      | 日→周       | 0    | 92.0% | 2007-08-03 |
| `commodity_fx`                | 商品货币周变化率均值（CAD、AUD 对 USD） | Yahoo CADUSD=X, AUDUSD=X               | 无量纲      | 日→周→变化率   | 0    | 99.9% | 2006-01-13 |
| `dgs10_change`                | 10 年美债收益率周变化              | 派生（`treasury_10y`）                     | 百分点      | —         | 0    | 99.9% | 2006-01-13 |


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


| 字段                      | 公式                                               |
| ----------------------- | ------------------------------------------------ |
| `brent_return_pct`      | `pct_change(brent_price) × 100`                  |
| `brent_log_return`      | `ln(Pₜ / Pₜ₋₁)`；训练目标为下一周值 `rₜ₊₁ = ln(Pₜ₊₁/Pₜ)`   |
| `brent_direction`       | `+1 if ret>0.5%`，`−1 if ret<−0.5%`，否则 `0`        |
| `brent_wti_spread`      | `brent_price − wti_price`                        |
| `crude_stocks_change`   | `diff(crude_stocks_excl_spr)`                    |
| `cushing_stocks_change` | `diff(cushing_stocks)`                           |
| `net_crude_trade`       | `crude_imports − crude_exports`                  |
| `sp500_return_pct`      | `pct_change(sp500) × 100`                        |
| `gold_return`           | `ln(goldₜ / goldₜ₋₁)`                            |
| `futures_spread`        | `ln(BZ=F) − ln(brent_price)`                     |
| `commodity_fx`          | `mean( pct_change(CADUSD), pct_change(AUDUSD) )` |
| `dgs10_change`          | `diff(treasury_10y)`                             |


---

## 6. 备注与已知缺口

- **早期缺失（均在 2019–2026 比较窗口之外）**：
  - `ovx` 自 2007-05 起（OVX 指数 2007 年推出）。
  - `futures_spread` 自 2007-08 起（所用 Brent 期货序列起步较晚）。
  - `global_econ_activity` / `nonoil_industrial_commodity` 因 +5 周保守滞后，前几周为 NaN。
- 派生 `*_change` / `*_return` 列首周为 NaN（差分/收益率需要前一周值）。
- `brent_direction` 首周无前值，按定义取 0（非缺失）。
- **预测目标不在本矩阵内**：唯一核心目标为下一周 Brent 价格 `Pₜ₊₁`；训练用 `rₜ₊₁=ln(Pₜ₊₁/Pₜ)`，输出还原 `P̂ₜ₊₁=Pₜ·e^(r̂)`。目标/标签需在建模脚本中由 `brent_price` **前瞻一周**生成，本矩阵只含特征、不含未来标签（避免泄漏）。

---

## 7. 复现命令

```bash
cd 03_data/processed/M1/py
python build_m1_weekly.py            # 离线，从本地 raw 构建（默认）
python build_m1_weekly.py --online   # 缺失源时在线补抓（FRED/Yahoo）
python build_m1_weekly.py --base-only # 仅基础块，跳过 8 个衍生变量
```

