# M3 航运/海运网络特征矩阵 — 数据字典

> 对应数据文件：`03_data/processed/M3/outputs/m3_weekly_features.csv`
> 构建脚本：`03_data/processed/M3/py/aggregate_shipping_to_weekly.py`
> 版本：union 对齐修订版（GFW/PW 各自对齐 + union 索引，修复 727→362 掉样本；发布滞后已加）
> 最后更新：2026-06-23

---

## 1. 基本信息


| 项      | 值                                                       |
| ------ | ------------------------------------------------------- |
| 时间索引   | `week_ending_friday`，每周五截止（W-FRI）                       |
| 时间范围   | 2012-02-03 ~ 2026-06-12（中间产物保留全历史，建模时裁剪）               |
| 形状     | 750 行 × 123 列                                           |
| 统一比较窗口 | 2019-01-04 ~ 2025-12-26（365 周）；窗内 GFW 100%、PortWatch 约 99.7% |
| 缺失约定   | 空单元格 = 该周该变量尚不可得（NaN），不做隐式填补                            |
| 频率     | 周频（GFW 月频上采样、PortWatch 日频下采样，统一对齐到周五）                   |
| 模态     | M3 = 海运/AIS 网络（6 个油相关咽喉 + 出口/进口枢纽方向性）                    |


---

## 2. 构建脚本做了什么（`aggregate_shipping_to_weekly.py`）

**定位**：把三个原始航运源（PortWatch 日频咽喉过境、PortWatch 日频港口进出口、GFW 月频船舶存在）各自归约到原生频率后，**统一 reindex 到同一条 union 周五索引**再列拼接，输出唯一一张周频宽表。核心是修复旧版「先 join 后截窗」导致的 GFW 早期样本丢失，并对每个源加无前视的发布滞后。

### 2.1 输入 → 输出


| 项                | 内容                                                              |
| ---------------- | --------------------------------------------------------------- |
| 输入根目录            | `03_data/raw/03_shipping/`                                      |
| `IMF Portwatch/` | `portwatch_chokepoints_daily.csv`（6 咽喉，2019+）、`portwatch_ports_daily.csv`（油轮枢纽进出口，2019+） |
| `GFW/`           | `gfw_chokepoint_vessel_presence_monthly.csv`（6 咽喉 AIS 存在，2012+） |
| 输出               | `03_data/processed/M3/outputs/m3_weekly_features.csv`（唯一产物）     |
| union 索引         | `pd.date_range(min_start, max_end, freq="W-FRI")`（跨所有源取并集）      |


> EMODnet 月度船舶密度栅格（约 1 km GeoTIFF）**暂未并入**：需 rasterio + 咽喉/AOI 多边形区域统计，列为后续交叉验证补充（见 `external_sources.md` M3 节）。

### 2.2 处理流程（`main()` 调用顺序）

1. **各源原生归约（pre-lag、pre-union）**：
  - PortWatch 咽喉：按咽喉分组，日频 `resample("W-FRI").sum(min_count=1)`；派生 `tanker_share / tanker_cap_share / avg_tanker_size / n_tanker_wow_pct / capacity_tanker_4w_ma`；跨咽喉汇总 `pw_all_*`。
  - PortWatch 港口：按 CSV 的 `role`（出口/进口）篮子聚合 `export_tanker` / `import_tanker` 吨位，构造方向性不对称族。
  - GFW：按咽喉分组的月度水平值，月频派生 `nontanker_hours / other_share / total_hours_mom_pct / dwell_hours_per_vessel`，再 `+ MonthEnd(0)` 对齐月末。
2. **建 union 索引**：取三源时间跨度并集的每周五（W-FRI）轴。
3. **各自对齐到 union + 发布滞后**：
  - 月频 GFW：`reindex(union, method="ffill")`（月末 → 其后各周五）→ `shift(+4 周)`。
  - 周频 PortWatch：`reindex(union)`（无填充）→ `shift(+1 周)`。
4. **列拼接**：三块在同一 union 索引上 `concat(axis=1)`，断言列名唯一。
5. **模态可用性标志**：`avail_gfw / avail_pw_chokepoints / avail_pw_ports / avail_shipping`（滞后后计算，反映真实可见性）。
6. **写盘 + 自检**：裁剪到研究区间、写出 CSV，打印 union 行数、各源覆盖、每咽喉覆盖，以及「首个可用周向后平移恰好 lag 周」的无前视方向自检。

### 2.3 三类重采样规则（核心函数）


| 函数               | 适用来源              | 规则                                            |
| ---------------- | ----------------- | --------------------------------------------- |
| `align_weekly`   | PortWatch 周频（日→周） | `reindex(union)` 不填充 → `shift(+1 周)` 发布滞后     |
| `align_monthly`  | GFW 月频            | 月末对齐 → `reindex(union, ffill)` → `shift(+4 周)` |
| `build_union_index` | 全部源             | 取所有源 `min(start)..max(end)` 生成统一 W-FRI 轴      |


> 「各自 ffill + union 索引」是修复 727→362 掉样本的关键：每个源先在各自原生频率对齐，再统一 reindex 到 union，**不**用会丢行的 inner-join。

### 2.4 运行模式与容错

- **默认**：从本地 raw 构建；缺文件则打印 `[skip]` 并跳过该源，不崩溃。
- `**--gfw-lag N**` / `**--pw-lag N**`：覆盖默认发布滞后周数（便于与导师确认保守滞后后复跑）。
- `**--no-lag**`：关闭所有滞后（**仅诊断用，有泄漏**）。
- 滞后常量集中在脚本顶部：`GFW_LAG_WEEKS=4`、`PW_LAG_WEEKS=1`。

---

## 3. 防泄漏、发布时间戳对齐与 union 修复（关键）

所有变量按「真实可得时间」对齐：在周五 T 预测下一周时，只使用截至 T 已发布的信息（no look-ahead）。


| 来源类别            | 周频化方法                | 发布滞后     | 常量              |
| --------------- | -------------------- | -------- | --------------- |
| PortWatch（日频聚合） | 日 → 周五求和             | **+1 周** | `PW_LAG_WEEKS=1`  |
| GFW（月频存在）       | 月末对齐 + 前向填充到周频       | **+4 周** | `GFW_LAG_WEEKS=4` |


要点：

- **union 修复（727→362）**：GFW 月频覆盖 2012–2025（上采到约 727 周），PortWatch 日频 2019–2026（约 362 周）。旧版「先 join 后把窗口截到 PortWatch 重叠区」会**丢掉 2012–2018 的 GFW 早期样本**，样本骤降至 362。本版改为各源各自对齐后 reindex 到 **union 索引**，结果 **750 周**（GFW 覆盖 746 周、PortWatch 388 周），不再丢早期样本。
- **GFW +4 周**：月度存在是整月聚合，月末才完整，且 4Wings 聚合发布有延迟；取月末 + 约 1 个月保守滞后。首个可用周从 `2012-03-02` 开始（union 首周 `2012-02-03` 经 +4 周后生效）。
- **PortWatch +1 周**：截至周五 T 的周聚合在该周五收盘时尚不可下载，整体后移 1 周。首个可用周从 `2019-01-11` 开始。
- **方向自检**：脚本验证每个源「首个可用周相对未滞后版本恰好后移 lag 周」（是滞后、非前视），三源均通过。
- 月频 GFW 仍采用「月末对齐 + 前向填充 + 保守滞后」（同月各周值重复）；如需显式 `age / valid_mask`（如遥感 M2 的处理），可在后续升级。

---

## 4. 字段说明

> 覆盖率为全样本（750 周）非空占比；首个有效周为该列第一个非 NaN 的周五。
> 6 个咽喉短码：`hormuz`（霍尔木兹）、`suez`（苏伊士运河）、`malacca`（马六甲）、`mandeb`（曼德海峡 Bab el-Mandeb）、`panama`（巴拿马运河）、`cape`（好望角）。

### 4.1 GFW 咽喉船舶存在（6 咽喉 × 9 指标 + 1 汇总 = 55 列）

下表每个 `gfw_{cp}_*` 指标对 6 个咽喉各有一列（`{cp}` 取上述短码）；来源 GFW 4Wings `public-global-presence`，月频 → 月末 ffill → **+4 周**滞后；首个有效周 `2012-03-02`（`*_mom_pct` 为 `2012-03-30`）。


| 字段模板                          | 含义                                    | 单位     | 周频化       | 滞后   | 覆盖率   |
| ------------------------------- | ------------------------------------- | ------ | --------- | ---- | ----- |
| `gfw_{cp}_total_hours`          | 咽喉多边形内船舶存在总时长                         | 船·小时   | 月→周 ffill | +4 周 | 99.5% |
| `gfw_{cp}_total_vessels`        | 唯一船舶数                                 | 艘      | 月→周 ffill | +4 周 | 99.5% |
| `gfw_{cp}_cargo_hours`          | 货船存在时长                                | 船·小时   | 月→周 ffill | +4 周 | 99.5% |
| `gfw_{cp}_bunker_hours`         | 加油/补给船存在时长                            | 船·小时   | 月→周 ffill | +4 周 | 99.5% |
| `gfw_{cp}_other_hours`          | 其他船型存在时长                              | 船·小时   | 月→周 ffill | +4 周 | 99.5% |
| `gfw_{cp}_nontanker_hours`      | 非油轮时长（cargo + bunker，派生）             | 船·小时   | 月频派生      | +4 周 | 99.5% |
| `gfw_{cp}_other_share`          | 其他船型占比（other/total，派生）               | 比例     | 月频派生      | +4 周 | 99.5% |
| `gfw_{cp}_total_hours_mom_pct`  | 总存在时长月环比（派生）                          | %      | 月频派生      | +4 周 | 98.9% |
| `gfw_{cp}_dwell_hours_per_vessel` | 每船平均停留时长（拥堵/dwell 代理，P016；派生）       | 小时/艘   | 月频派生      | +4 周 | 99.5% |
| `gfw_all_total_hours_sum`       | 6 咽喉总存在时长之和（汇总）                       | 船·小时   | 月频派生      | +4 周 | 99.5% |


### 4.2 PortWatch 咽喉过境（6 咽喉 × 9 指标 + 3 汇总 = 57 列）

下表每个 `pw_{cp}_*` 指标对 6 个咽喉各有一列；来源 IMF PortWatch `Daily_Chokepoints_Data`，日频 → 周五求和 → **+1 周**滞后；首个有效周 `2019-01-11`（`*_wow_pct` 为 `2019-01-18`）。


| 字段模板                            | 含义                                | 单位        | 周频化     | 滞后   | 覆盖率   |
| --------------------------------- | --------------------------------- | --------- | ------- | ---- | ----- |
| `pw_{cp}_n_tanker`                | 油轮过境数（周求和）                        | 艘        | 日→周 sum | +1 周 | 51.5% |
| `pw_{cp}_n_total`                 | 全部船舶过境数（周求和）                      | 艘        | 日→周 sum | +1 周 | 51.5% |
| `pw_{cp}_capacity_tanker`         | 油轮运力（周求和）                         | 载重吨(DWT 估) | 日→周 sum | +1 周 | 51.5% |
| `pw_{cp}_capacity`                | 全部船舶运力（周求和）                       | 载重吨(DWT 估) | 日→周 sum | +1 周 | 51.5% |
| `pw_{cp}_tanker_share`            | 油轮数占比（n_tanker/n_total，派生）        | 比例       | 周频派生    | +1 周 | 51.5% |
| `pw_{cp}_tanker_cap_share`        | 油轮运力占比（cap_tanker/cap，派生）         | 比例       | 周频派生    | +1 周 | 51.5% |
| `pw_{cp}_avg_tanker_size`         | 平均油轮船型（cap_tanker/n_tanker，P070；派生） | 载重吨/艘    | 周频派生    | +1 周 | 51.5% |
| `pw_{cp}_n_tanker_wow_pct`        | 油轮数周环比（派生）                        | %        | 周频派生    | +1 周 | 51.3% |
| `pw_{cp}_capacity_tanker_4w_ma`   | 油轮运力 4 周移动平均（派生）                  | 载重吨      | 周频派生    | +1 周 | 51.5% |
| `pw_all_n_tanker_sum`             | 6 咽喉油轮过境总数（汇总）                    | 艘        | 周频派生    | +1 周 | 51.5% |
| `pw_all_n_total_sum`              | 6 咽喉总过境数（汇总）                      | 艘        | 周频派生    | +1 周 | 51.5% |
| `pw_all_tanker_share`            | 6 咽喉合计油轮数占比（汇总）                   | 比例       | 周频派生    | +1 周 | 51.5% |


### 4.3 PortWatch 港口方向性进出口（7 列）

来源 IMF PortWatch `Daily_Ports_Data`，按出口/进口枢纽篮子聚合油轮吨位，日频 → 周五求和 → **+1 周**滞后；首个有效周 `2019-01-11`（`*_wow_pct` 为 `2019-01-18`）。出口枢纽：Ras Tanura/Juaymah/Yanbu/Ras Laffan/Primorsk/Novorossiysk/Corpus Christi/Sidi Kerir/Bonny；进口枢纽：Rotterdam/Singapore/Ningbo/Chiba/Ulsan。


| 字段                              | 含义                                  | 单位     | 周频化     | 滞后   | 覆盖率   |
| ------------------------------- | ----------------------------------- | ------ | ------- | ---- | ----- |
| `pw_exp_hubs_export_vol`        | 出口枢纽油轮出口吨位（周求和）                     | 公吨     | 日→周 sum | +1 周 | 51.7% |
| `pw_imp_hubs_import_vol`        | 进口枢纽油轮进口吨位（周求和）                     | 公吨     | 日→周 sum | +1 周 | 51.7% |
| `pw_tanker_exp_imp_net`         | 出口−进口净吨位（派生）                        | 公吨     | 周频派生    | +1 周 | 51.7% |
| `pw_tanker_exp_imp_asym`        | 出口−进口不对称度 net/(exp+imp)（派生）         | 无量纲 [-1,1] | 周频派生 | +1 周 | 51.7% |
| `pw_tanker_exp_imp_log_ratio`   | 出口/进口对数比 ln((exp+1)/(imp+1))（派生）    | 无量纲    | 周频派生    | +1 周 | 51.7% |
| `pw_tanker_exp_imp_asym_4w_ma`  | 不对称度 4 周移动平均（派生）                    | 无量纲    | 周频派生    | +1 周 | 51.7% |
| `pw_exp_hubs_export_vol_wow_pct`| 出口枢纽出口吨位周环比（派生）                     | %      | 周频派生    | +1 周 | 51.6% |


### 4.4 模态可用性标志（4 列）


| 字段                     | 含义                              | 来源 | 单位    | 覆盖率   | 首个有效周      |
| ---------------------- | ------------------------------- | -- | ----- | ----- | ---------- |
| `avail_gfw`            | GFW 数据可用（任一 GFW 列非空）            | 派生 | {0,1} | 100.0% | 2012-02-03 |
| `avail_pw_chokepoints` | PortWatch 咽喉数据可用                | 派生 | {0,1} | 100.0% | 2012-02-03 |
| `avail_pw_ports`       | PortWatch 港口方向数据可用              | 派生 | {0,1} | 100.0% | 2012-02-03 |
| `avail_shipping`       | 任一航运数据可用（GFW 或 PortWatch）       | 派生 | {0,1} | 100.0% | 2012-02-03 |


> 注：avail 列本身每周都有 0/1 取值（覆盖率 100%）；其值在早期/滞后边界周为 0（如 2012-02-03 经 GFW +4 周滞后后仍无值，标 0）。

---

## 5. 派生变量公式


| 字段                              | 公式                                                |
| ------------------------------- | ------------------------------------------------- |
| `gfw_{cp}_nontanker_hours`      | `cargo_hours + bunker_hours`                      |
| `gfw_{cp}_other_share`          | `other_hours / total_hours`                       |
| `gfw_{cp}_total_hours_mom_pct`  | `pct_change(total_hours) × 100`（月频上）             |
| `gfw_{cp}_dwell_hours_per_vessel` | `total_hours / total_vessels`                   |
| `pw_{cp}_tanker_share`          | `n_tanker / n_total`                              |
| `pw_{cp}_tanker_cap_share`      | `capacity_tanker / capacity`                      |
| `pw_{cp}_avg_tanker_size`       | `capacity_tanker / n_tanker`                      |
| `pw_{cp}_n_tanker_wow_pct`      | `pct_change(n_tanker) × 100`（周频上）               |
| `pw_{cp}_capacity_tanker_4w_ma` | `rolling(4, min_periods=1).mean(capacity_tanker)` |
| `pw_tanker_exp_imp_net`         | `exp_hubs_export_vol − imp_hubs_import_vol`        |
| `pw_tanker_exp_imp_asym`        | `net / (exp_vol + imp_vol)`                       |
| `pw_tanker_exp_imp_log_ratio`   | `ln((exp_vol + 1) / (imp_vol + 1))`               |
| `pw_tanker_exp_imp_asym_4w_ma`  | `rolling(4, min_periods=1).mean(asym)`            |


---

## 6. 备注与已知缺口

- **样本期不一致（按设计保留）**：GFW 自 2012、PortWatch 自 2019；中间产物保留 2012–2026 全历史以支持长历史稳健性。**标准比较窗 2019.1–2025.12 内 GFW 100%、PortWatch 约 99.7%**（仅 PortWatch 因 +1 周滞后使首周 2019-01-04 为 NaN）。
- **PortWatch 全样本覆盖率约 51%** 属正常：其数据 2019 才开始，2012–2018 的早期周仅 GFW 有值（这正是 union 修复保留的样本）。
- 派生 `*_mom_pct` / `*_wow_pct` 列因需前一期值，首个有效周比对应水平列晚一期（GFW 晚一月、PortWatch 晚一周）。
- **方向性仅来自港口数据**：PortWatch 咽喉数据无进出口方向字段，方向性不对称族由港口级 `import_tanker`/`export_tanker` 吨位估计构造（源自 AIS 吃水，不内嵌油价 → 防泄漏，P018）。
- **拥堵代理**：无专用锚地等待数据，用 GFW `total_hours/total_vessels`（每船停留时长）作 dwell 代理（P016）。
- **EMODnet 未并入**：月度密度栅格需 GIS 区域统计，列为后续交叉验证补充。
- **预测目标不在本矩阵内**：M3 仅含航运特征；唯一核心目标（下一周 Brent 价格）由 M1 的 `brent_price` 在合并/建模阶段前瞻一周生成。M3 与 M1/M2 的合并见 `03_data/processed/merge/py/build_feature_matrix.py`。

---

## 7. 复现命令

```bash
cd 03_data/processed/M3/py
python aggregate_shipping_to_weekly.py                 # 默认：GFW +4 周、PortWatch +1 周
python aggregate_shipping_to_weekly.py --gfw-lag 4 --pw-lag 1   # 显式指定滞后
python aggregate_shipping_to_weekly.py --no-lag        # 关闭滞后（仅诊断，有泄漏）
```
