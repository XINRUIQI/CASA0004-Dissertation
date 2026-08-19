# M3 航运/海运网络特征矩阵 — 数据字典

> 对应数据文件：`data/processed/M3/outputs/m3_weekly_features.csv`
> 构建脚本：`data/processed/M3/py/aggregate_shipping_to_weekly.py`
> 版本：union 对齐修订版（GFW/PW 各自对齐 + union 索引，修复 727→362 掉样本；发布滞后已加）
> 最后更新：2026-07-05（M3/M4 主模型由 core 38 列改为 full 113 列，见 §11）

---

## 1. 基本信息


| 项      | 值                                                            |
| ------ | ------------------------------------------------------------ |
| 时间索引   | `week_ending_friday`，每周五截止（W-FRI）                            |
| 时间范围   | 2012-02-03 ~ 2026-06-12（中间产物保留全历史，建模时裁剪）                     |
| 形状     | 750 行 × 123 列                                                |
| 统一比较窗口 | 2019-01-04 ~ 2025-12-26（365 周）；窗内 GFW 100%、PortWatch 约 99.7% |
| 缺失约定   | 空单元格 = 该周该变量尚不可得（NaN），不做隐式填补                                 |
| 频率     | 周频（GFW 月频上采样、PortWatch 日频下采样，统一对齐到周五）                        |
| 模态     | M3 = 海运/AIS 网络（6 个油相关咽喉 + 出口/进口枢纽方向性）                        |


---

## 2. 构建脚本做了什么（`aggregate_shipping_to_weekly.py`）

**定位**：把三个原始航运源（PortWatch 日频咽喉过境、PortWatch 日频港口进出口、GFW 月频船舶存在）各自归约到原生频率后，**统一 reindex 到同一条 union 周五索引**再列拼接，输出唯一一张周频宽表。核心是修复旧版「先 join 后截窗」导致的 GFW 早期样本丢失，并对每个源加无前视的发布滞后。

### 2.1 输入 → 输出


| 项                | 内容                                                                                       |
| ---------------- | ---------------------------------------------------------------------------------------- |
| 输入根目录            | `data/raw/03_shipping/`                                                               |
| `IMF Portwatch/` | `portwatch_chokepoints_daily.csv`（6 咽喉，2019+）、`portwatch_ports_daily.csv`（油轮枢纽进出口，2019+） |
| `GFW/`           | `gfw_chokepoint_vessel_presence_monthly.csv`（6 咽喉 AIS 存在，2012+）                          |
| 输出               | `data/processed/M3/outputs/m3_weekly_features.csv`（唯一产物）                              |
| union 索引         | `pd.date_range(min_start, max_end, freq="W-FRI")`（跨所有源取并集）                               |


> EMODnet 月度船舶密度栅格（约 1 km GeoTIFF）**暂未并入**：需 rasterio + 咽喉/AOI 多边形区域统计，列为后续交叉验证补充（见 `external_sources.md` M3 节）。

### 2.2 处理流程（`main()` 调用顺序）

1. **各源原生归约（pre-lag、pre-union）**：
  - PortWatch 咽喉：按咽喉分组，日频 `resample("W-FRI").sum(min_count=1)`；派生 `tanker_share / tanker_cap_share / avg_tanker_size / n_tanker_wow_pct / capacity_tanker_4w_ma`；跨咽喉汇总 `pw_all_*`。
  - PortWatch 港口：按 CSV 的 `role`（出口/进口）篮子聚合 `export_tanker` / `import_tanker` 吨位，构造方向性不对称族。
  - GFW：按咽喉分组的月度水平值，月频派生 `other_share / total_hours_mom_pct / mean_presence_hours_per_vessel`，再 `+ MonthEnd(0)` 对齐月末。
2. **建 union 索引**：取三源时间跨度并集的每周五（W-FRI）轴。
3. **各自对齐到 union + 发布滞后**：
  - 月频 GFW：`reindex(union, method="ffill")`（月末 → 其后各周五）→ `shift(+4 周)`。
  - 周频 PortWatch：`reindex(union)`（无填充）→ `shift(+1 周)`。
4. **列拼接**：三块在同一 union 索引上 `concat(axis=1)`，断言列名唯一。
5. **模态可用性标志**：`avail_gfw / avail_pw_chokepoints / avail_pw_ports / avail_shipping`（滞后后计算，反映真实可见性）。
6. **写盘 + 自检**：裁剪到研究区间、写出 CSV，打印 union 行数、各源覆盖、每咽喉覆盖，以及「首个可用周向后平移恰好 lag 周」的无前视方向自检。

### 2.3 三类重采样规则（核心函数）


| 函数                  | 适用来源              | 规则                                             |
| ------------------- | ----------------- | ---------------------------------------------- |
| `align_weekly`      | PortWatch 周频（日→周） | `reindex(union)` 不填充 → `shift(+1 周)` 发布滞后      |
| `align_monthly`     | GFW 月频            | 月末对齐 → `reindex(union, ffill)` → `shift(+4 周)` |
| `build_union_index` | 全部源               | 取所有源 `min(start)..max(end)` 生成统一 W-FRI 轴       |


> 「各自 ffill + union 索引」是修复 727→362 掉样本的关键：每个源先在各自原生频率对齐，再统一 reindex 到 union，**不**用会丢行的 inner-join。

### 2.4 运行模式与容错

- **默认**：从本地 raw 构建；缺文件则打印 `[skip]` 并跳过该源，不崩溃。
- `**--gfw-lag N**` / `**--pw-lag N**`：覆盖默认发布滞后周数（便于与导师确认保守滞后后复跑）。
- `**--no-lag**`：关闭所有滞后（**仅诊断用，有泄漏**）。
- 滞后常量集中在脚本顶部：`GFW_LAG_WEEKS=4`、`PW_LAG_WEEKS=1`。

---

## 3. 防泄漏、发布时间戳对齐与 union 修复（关键）

所有变量按「真实可得时间」对齐：在周五 T 预测下一周时，只使用截至 T 已发布的信息（no look-ahead）。


| 来源类别            | 周频化方法          | 发布滞后     | 常量                |
| --------------- | -------------- | -------- | ----------------- |
| PortWatch（日频聚合） | 日 → 周五求和       | **+1 周** | `PW_LAG_WEEKS=1`  |
| GFW（月频存在）       | 月末对齐 + 前向填充到周频 | **+4 周** | `GFW_LAG_WEEKS=4` |


要点：

- **union 修复（727→362）**：GFW 月频覆盖 2012–2025（上采到约 727 周），PortWatch 日频 2019–2026（约 362 周）。旧版「先 join 后把窗口截到 PortWatch 重叠区」会**丢掉 2012–2018 的 GFW 早期样本**，样本骤降至 362。本版改为各源各自对齐后 reindex 到 **union 索引**，结果 **750 周**（GFW 覆盖 746 周、PortWatch 388 周），不再丢早期样本。
- **GFW +4 周**：月度存在是整月聚合，月末才完整；另加 **保守可得性滞后**（`GFW_LAG_WEEKS=4`），用于防前视并反映可能的处理/发布延迟——**并非** GFW 官方统一 4 周发布规则（GFW `public-global-presence` 通常更新至约 96 小时前）。首个可用周从 `2012-03-02` 开始。建议稳健性检验 lag ∈ {1, 4, 8} 周（见 §9.3）。
- **PortWatch +1 周**：截至周五 T 的周聚合在该周五收盘时尚不可下载，整体后移 1 周。首个可用周从 `2019-01-11` 开始。
- **方向自检**：脚本验证每个源「首个可用周相对未滞后版本恰好后移 lag 周」（是滞后、非前视），三源均通过。
- 月频 GFW 仍采用「月末对齐 + 前向填充 + 保守滞后」（同月各周值重复）；如需显式 `age / valid_mask`（如遥感 M2 的处理），可在后续升级。

---

## 4. 字段说明

> 覆盖率为全样本（750 周）非空占比；首个有效周为该列第一个非 NaN 的周五。
> 6 个咽喉短码：`hormuz`（霍尔木兹）、`suez`（苏伊士运河）、`malacca`（马六甲）、`mandeb`（曼德海峡 Bab el-Mandeb）、`panama`（巴拿马运河）、`cape`（好望角）。

### 4.1 GFW 咽喉船舶存在（6 咽喉 × 8 指标 + 1 汇总 = 49 列）

下表每个 `gfw_{cp}_*` 指标对 6 个咽喉各有一列（`{cp}` 取上述短码）；来源 GFW 4Wings `public-global-presence`，月频 → 月末 ffill → **+4 周**滞后；首个有效周 `2012-03-02`（`*_mom_pct` 为 `2012-03-30`）。

> **变更（2026-07-03，已重建 CSV/矩阵）**：`nontanker_hours`（原 cargo+bunker）已**删除**；`dwell_hours_per_vessel` 已改名为 **`mean_presence_hours_per_vessel`**（含义 = total_hours/total_vessels，为**存在强度/拥堵粗代理**，非 P016 严格 dwell）。**文献支撑、经济含义边界与 core/extended 分级见 §9**；进主模型清单见 §11。
>
> 标 ★ = 核心（进主模型，6×4=24 列）；标 ▽ = 扩展（消融/附录）；标 ◆ = **不进主模型、单独实验**（见 §11.1）。


| 字段模板                              | 含义                            | 单位   | 周频化       | 滞后   | 覆盖率   | 层级 |
| --------------------------------- | ----------------------------- | ---- | --------- | ---- | ----- | -- |
| `gfw_{cp}_total_hours`            | 咽喉多边形内船舶存在总时长                 | 船·小时 | 月→周 ffill | +4 周 | 99.5% | ★ |
| `gfw_{cp}_total_vessels`          | 月度唯一船舶数（全船型）                  | 艘    | 月→周 ffill | +4 周 | 99.5% | ★ |
| `gfw_{cp}_cargo_hours`            | 货船存在时长                        | 船·小时 | 月→周 ffill | +4 周 | 99.5% | ★ |
| `gfw_{cp}_total_hours_mom_pct`    | 总存在时长月环比（派生）                  | %    | 月频派生      | +4 周 | 98.9% | ★ |
| `gfw_{cp}_mean_presence_hours_per_vessel` | 每船平均存在时长（total_hours/total_vessels，存在强度/拥堵粗代理；非严格 dwell，见 §9.1；派生） | 小时/艘 | 月频派生      | +4 周 | 99.5% | ◆ GFW-Presence 实验 |
| `gfw_all_activity_zmean`          | 6 咽喉 total_hours 过去-only 扩展窗 z-score 均值（**建模阶段派生**，leak-free；替代下方 raw sum） | 无量纲 | 建模阶段派生 | +4 周 | — | ◆ GFW-Aggregate 基准 |
| `gfw_{cp}_bunker_hours`           | 加油/补给船存在时长                    | 船·小时 | 月→周 ffill | +4 周 | 99.5% | ▽ |
| `gfw_{cp}_other_hours`            | 其他船型存在时长                      | 船·小时 | 月→周 ffill | +4 周 | 99.5% | ▽ |
| `gfw_{cp}_other_share`            | 其他船型占比（other/total，派生）        | 比例   | 月频派生      | +4 周 | 99.5% | ▽ |
| `gfw_all_total_hours_sum`         | 6 咽喉总存在时长之和（汇总；**已弃用**，改用 `gfw_all_activity_zmean`，见 §9.2/§11） | 船·小时 | 月频派生      | +4 周 | 99.5% | — |


### 4.2 PortWatch 咽喉过境（6 咽喉 × 9 指标 + 3 汇总 = 57 列）

下表每个 `pw_{cp}_*` 指标对 6 个咽喉各有一列；来源 IMF PortWatch `Daily_Chokepoints_Data`，日频 → 周五求和 → **+1 周**滞后；首个有效周 `2019-01-11`（`*_wow_pct` 为 `2019-01-18`）。

> **PortWatch 原生字段解释边界（P070 / WP/26/99）**
>
> - `**n_tanker**`：PortWatch 字段 `n_tanker` 的日/周求和，表示 **tanker（液货船）过境艘次**（vessel transits / transit events），**不是**去重后的独立船舶数，**也不是**严格意义上的「原油油轮数」。IMF 的 tanker 分类为 **liquid bulk vessel grouping**，可含原油船、成品油船、化学品船、LNG/LPG 等液货运输船。
> - `**capacity_tanker**`：PortWatch 字段 `capacity_tanker` 的日/周求和，表示 **tanker 过境运力（DWT-equivalent transit capacity proxy）**——通过该航道的液货船载重吨能力估计，**不是**实际装载原油吨数、出口量或贸易量。应视为 maritime-trade / shipping-activity **粗代理**（P070），非官方石油贸易统计的替代品。


| 字段模板                            | 含义                                                         | 单位      | 周频化     | 滞后   | 覆盖率   |
| ------------------------------- | ---------------------------------------------------------- | ------- | ------- | ---- | ----- |
| `pw_{cp}_n_tanker`              | **tanker（液货船）过境艘次**（周求和；PortWatch 原生 `n_tanker`）           | 艘次      | 日→周 sum | +1 周 | 51.5% |
| `pw_{cp}_n_total`               | 全部商业船舶过境艘次（周求和；PortWatch 原生 `n_total`）                     | 艘次      | 日→周 sum | +1 周 | 51.5% |
| `pw_{cp}_capacity_tanker`       | **tanker 过境运力**（DWT-equivalent transit capacity proxy；周求和） | DWT（估计） | 日→周 sum | +1 周 | 51.5% |
| `pw_{cp}_capacity`              | 全部商业船舶过境运力（周求和；PortWatch 原生 `capacity`）                    | DWT（估计） | 日→周 sum | +1 周 | 51.5% |
| `pw_{cp}_tanker_share`          | tanker 过境艘次占比（n_tanker/n_total，派生）                         | 比例      | 周频派生    | +1 周 | 51.5% |
| `pw_{cp}_tanker_cap_share`      | tanker 过境运力占比（cap_tanker/cap，派生）                           | 比例      | 周频派生    | +1 周 | 51.5% |
| `pw_{cp}_avg_tanker_size`       | 单次 tanker 过境平均 DWT（cap_tanker/n_tanker，派生）                 | DWT/艘次  | 周频派生    | +1 周 | 51.5% |
| `pw_{cp}_n_tanker_wow_pct`      | tanker 过境艘次周环比（派生）                                         | %       | 周频派生    | +1 周 | 51.3% |
| `pw_{cp}_capacity_tanker_4w_ma` | tanker 过境运力 4 周 trailing 均值（派生）                            | DWT     | 周频派生    | +1 周 | 51.5% |
| `pw_all_n_tanker_sum`           | 6 航道 tanker 过境艘次之和（汇总；跨航道可重复计数）                            | 艘次      | 周频派生    | +1 周 | 51.5% |
| `pw_all_n_total_sum`            | 6 航道全部商业船舶过境艘次之和（汇总）                                       | 艘次      | 周频派生    | +1 周 | 51.5% |
| `pw_all_tanker_share`           | 6 航道合计 tanker 艘次占比（ratio of sums，汇总）                       | 比例      | 周频派生    | +1 周 | 51.5% |


### 4.3 PortWatch 港口方向性进出口（7 列）

来源 IMF PortWatch `Daily_Ports_Data`，按出口/进口枢纽篮子聚合 tanker（液货）吨位估计，日频 → 周五求和 → **+1 周**滞后；首个有效周 `2019-01-11`（`*_wow_pct` 为 `2019-01-18`）。出口枢纽：Ras Tanura/Juaymah/Yanbu/Ras Laffan/Primorsk/Novorossiysk/Corpus Christi/Sidi Kerir/Bonny；进口枢纽：Rotterdam/Singapore/Ningbo/Chiba/Ulsan。

> **解释边界与分级见 §11**。`export_vol`/`import_vol` 为**选定 tanker 出口/进口枢纽篮子**的 AIS 吨位**估计**（liquid bulk，非海关原油量、非纯原油港）；`net` 非全球净出口；`asym` 与 `log_ratio` 高度冗余（本地 corr≈0.9999）。标 ★ = 核心；标 ▽ = 扩展/稳健性。


| 字段                              | 含义                                  | 单位     | 周频化     | 滞后   | 覆盖率   | 层级 |
| ------------------------------- | ----------------------------------- | ------ | ------- | ---- | ----- | -- |
| `pw_exp_hubs_export_vol`        | 选定出口枢纽 tanker 出口吨位估计（周求和）           | 公吨（估计） | 日→周 sum | +1 周 | 51.7% | ★ |
| `pw_imp_hubs_import_vol`        | 选定进口枢纽 tanker 进口吨位估计（周求和）           | 公吨（估计） | 日→周 sum | +1 周 | 51.7% | ★ |
| `pw_tanker_exp_imp_log_ratio`   | 出口/进口对数比 ln((exp+1)/(imp+1))（派生；扩展首选，训练窗中心化见 §11） | 无量纲    | 周频派生    | +1 周 | 51.7% | ▽ |
| `pw_tanker_exp_imp_net`         | 出口−进口净吨位（派生；非全球净出口，两侧港口数不等）         | 公吨     | 周频派生    | +1 周 | 51.7% | ▽ |
| `pw_tanker_exp_imp_asym`        | 出口−进口不对称度 net/(exp+imp)（派生；与 log_ratio 冗余，二选一） | 无量纲 [-1,1] | 周频派生 | +1 周 | 51.7% | ▽ |
| `pw_tanker_exp_imp_asym_4w_ma`  | 不对称度 4 周 trailing 均值（派生；与 lookback=4 冗余） | 无量纲    | 周频派生    | +1 周 | 51.7% | ▽ |
| `pw_exp_hubs_export_vol_wow_pct`| 出口枢纽出口吨位周环比（派生；建议改 log change，见 §11） | %      | 周频派生    | +1 周 | 51.6% | ▽ |


### 4.4 模态可用性标志（4 列）

`avail_*` 在合并矩阵中标为 `modality='mask'`，**不作普通预测特征**进入 Ridge/XGB（`code/src/backtest/data.py`：targets 与 mask 从不入模型）；用于缺失模态门控 / 长样本诊断。


| 字段                     | 含义                              | 来源 | 单位    | 指示变量非空率 | availability=1 比例（2019–2025） | 首次 =1     |
| ---------------------- | ------------------------------- | -- | ----- | ------- | --------------------------- | ---------- |
| `avail_gfw`            | GFW 数据可用（任一 GFW 列非空）            | 派生 | {0,1} | 100%    | 100%                        | 2012-03-02 |
| `avail_pw_chokepoints` | PortWatch 咽喉数据可用                | 派生 | {0,1} | 100%    | 99.7%（1 周为 0）               | 2019-01-11 |
| `avail_pw_ports`       | PortWatch 港口方向数据可用（标准窗内与 chokepoints 相同） | 派生 | {0,1} | 100%    | 99.7%（1 周为 0）               | 2019-01-11 |
| `avail_shipping`       | 任一航运数据可用（GFW ∨ PW；确定性 OR，冗余，见 §11 建议删） | 派生 | {0,1} | 100%    | 100%                        | 2012-03-02 |

> ⚠️ 「指示变量非空率 100%」≠「模态 100% 可用」：前者指该列每周都有 0/1 取值；后者为 `=1` 的比例（见上表）。「首次 =1」才是模态真正首个可用周（非指示变量起始周 2012-02-03）。长样本中 `avail_pw_*` ≈ post-2019 dummy，SHAP 不可解释为因果（见 §11）。

---

## 5. 派生变量公式


| 字段                                | 公式                                                    |
| --------------------------------- | ----------------------------------------------------- |
| `gfw_{cp}_other_share`            | `other_hours / total_hours`                           |
| `gfw_{cp}_total_hours_mom_pct`    | `pct_change(total_hours) × 100`（月频上）                  |
| `gfw_{cp}_mean_presence_hours_per_vessel` | `total_hours / total_vessels`                 |
| `pw_{cp}_tanker_share`            | `n_tanker / n_total`                                  |
| `pw_{cp}_tanker_cap_share`        | `capacity_tanker / capacity`                          |
| `pw_{cp}_avg_tanker_size`         | `capacity_tanker / n_tanker`（DWT/艘次；n_tanker=0 → NaN） |
| `pw_{cp}_n_tanker_wow_pct`        | `pct_change(n_tanker) × 100`（周频上）                     |
| `pw_{cp}_capacity_tanker_4w_ma`   | `rolling(4, min_periods=1).mean(capacity_tanker)`     |
| `pw_tanker_exp_imp_net`           | `exp_hubs_export_vol − imp_hubs_import_vol`           |
| `pw_tanker_exp_imp_asym`          | `net / (exp_vol + imp_vol)`                           |
| `pw_tanker_exp_imp_log_ratio`     | `ln((exp_vol + 1) / (imp_vol + 1))`                   |
| `pw_tanker_exp_imp_asym_4w_ma`    | `rolling(4, min_periods=1).mean(asym)`                |


---

## 6. 备注与已知缺口

- **样本期不一致（按设计保留）**：GFW 自 2012、PortWatch 自 2019；中间产物保留 2012–2026 全历史以支持长历史稳健性。**标准比较窗 2019.1–2025.12 内 GFW 100%、PortWatch 约 99.7%**（仅 PortWatch 因 +1 周滞后使首周 2019-01-04 为 NaN）。
- **PortWatch 全样本覆盖率约 51%** 属正常：其数据 2019 才开始，2012–2018 的早期周仅 GFW 有值（这正是 union 修复保留的样本）。
- 派生 `*_mom_pct` / `*_wow_pct` 列因需前一期值，首个有效周比对应水平列晚一期（GFW 晚一月、PortWatch 晚一周）。
- **方向性仅来自港口数据**：PortWatch 咽喉数据无进出口方向字段，方向性不对称族由港口级 `import_tanker`/`export_tanker` 吨位估计构造（源自 AIS 吃水，不内嵌油价 → 防泄漏，P018）。
- **拥堵/存在强度代理**：无专用锚地等待数据；用 GFW `total_hours/total_vessels`（CSV 列名 `mean_presence_hours_per_vessel`）作存在强度/拥堵**粗代理**，**非** P016 意义下的靠泊 dwell time（见 §9）。
- **EMODnet 未并入**：月度密度栅格需 GIS 区域统计，列为后续交叉验证补充。
- **预测目标不在本矩阵内**：M3 仅含航运特征；唯一核心目标（下一周 Brent 价格）由 M1 的 `brent_price` 在合并/建模阶段前瞻一周生成。M3 与 M1/M2 的合并见 `data/processed/merge/py/build_feature_matrix.py`。

---

## 7. 复现命令

```bash
cd data/processed/M3/py
python aggregate_shipping_to_weekly.py                 # 默认：GFW +4 周、PortWatch +1 周
python aggregate_shipping_to_weekly.py --gfw-lag 4 --pw-lag 1   # 显式指定滞后
python aggregate_shipping_to_weekly.py --no-lag        # 关闭滞后（仅诊断，有泄漏）
```

---

## 8. 解释

本节说明 M3 原始 CSV 的内容、空间单元含义（咽喉 vs 枢纽），以及三源如何汇入周频矩阵。原始数据均由 Python 脚本自动下载（非手动导出），落盘于 `data/raw/03_shipping/`。

### 8.1 概念：咽喉 vs 枢纽


| 概念                 | 定义                                     | 在本项目中的度量                                         |
| ------------------ | -------------------------------------- | ------------------------------------------------ |
| **咽喉（Chokepoint）** | 油轮航线上**必须经过**的地理狭窄通道/瓶颈（非海关等行政关口）      | **过路流量（transit）**：有多少船、多少运力**经过**该通道；**无进出口方向**  |
| **枢纽（Hub）**        | 石油供应链上的**装卸节点**（特指油轮装/卸货的港口，非泛指任意物流节点） | **方向性流量**：每个港口油轮**进了多少吨、出了多少吨**（import / export） |


**咽喉示例**（6 个）：霍尔木兹、苏伊士、马六甲、曼德海峡、巴拿马运河、好望角。大量海运被「挤」过这些窄道，可反映全球航运是否通畅、是否拥堵。

**枢纽示例**（14 个）：


| 类型                      | 港口                                                                                                | 供应链角色               |
| ----------------------- | ------------------------------------------------------------------------------------------------- | ------------------- |
| **出口枢纽**（export hub）    | Ras Tanura, Juaymah, Yanbu, Ras Laffan, Primorsk, Novorossiysk, Corpus Christi, Sidi Kerir, Bonny | 原油**装船出口**（供给端）     |
| **进口/炼化枢纽**（import hub） | Rotterdam, Singapore, Ningbo, Chiba, Ulsan                                                        | 原油**卸货进口**或进炼厂（需求端） |


> **术语**：本项目 PortWatch 指标中 **tanker** 指 IMF 的 **liquid bulk（液货船）** 分组，非邮轮（cruise ship），也**非**严格意义上的「原油油轮」。`n_tanker` 等为**过境艘次**（transit events），非去重独立船数；`capacity_tanker` 为 **DWT-equivalent transit capacity proxy**，非实际装载原油吨数。均为海运活动/贸易**粗代理**（P070/P018），非精确石油贸易量、非靠港频率（P016）。

**一句话区别**：咽喉 = 航线上的「必经之路」（过路、无方向）；枢纽 = 供应链上的「装卸节点」（有 import / export 方向）。

### 8.2 原始 CSV 一览


| 文件                                               | 下载脚本                                | 数据源                                                                              | 频率  | 覆盖    |
| ------------------------------------------------ | ----------------------------------- | -------------------------------------------------------------------------------- | --- | ----- |
| `IMF Portwatch/portwatch_chokepoints_daily.csv`  | `download_portwatch_chokepoints.py` | IMF PortWatch ArcGIS `Daily_Chokepoints_Data`                                    | 日频  | 2019+ |
| `IMF Portwatch/portwatch_ports_daily.csv`        | `download_portwatch_ports.py`       | IMF PortWatch ArcGIS `Daily_Ports_Data`                                          | 日频  | 2019+ |
| `GFW/gfw_chokepoint_vessel_presence_monthly.csv` | `download_gfw_vessel_presence.py`   | GFW 4Wings Report API（`public-global-presence`；需 `GFW_API_TOKEN` 或 `.gfw_token`） | 月频  | 2012+ |


三源经 `aggregate_shipping_to_weekly.py` 聚合成 `m3_weekly_features.csv`（750 周 × 123 列）：PortWatch 日 → 周五求和 + **+1 周**滞后；GFW 月 → 周 ffill + **+4 周**滞后。

### 8.3 `portwatch_chokepoints_daily.csv` — 咽喉过境流量

**规模**：16,164 行 × 21 列 | **频率**：日频 | **覆盖**：2019-01-01 ~ 2026-05-17

**空间单元**：6 个油相关咽喉，**每日每咽喉一行**。


| portid      | 名称                         |
| ----------- | -------------------------- |
| chokepoint6 | Strait of Hormuz（霍尔木兹）     |
| chokepoint1 | Suez Canal（苏伊士）            |
| chokepoint5 | Malacca Strait（马六甲）        |
| chokepoint2 | Panama Canal（巴拿马）          |
| chokepoint4 | Bab el-Mandeb Strait（曼德海峡） |
| chokepoint7 | Cape of Good Hope（好望角）     |


**核心信息**（AIS 推断的过境统计，**无进出口方向**）：


| 列组    | 字段                                                                                         | 含义                                                                             |
| ----- | ------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------ |
| 时间/标识 | `date`, `year`, `month`, `day`, `portid`, `portname`                                       | 日期与咽喉标识                                                                        |
| 过境艘次  | `n_tanker`, `n_container`, `n_dry_bulk`, `n_general_cargo`, `n_roro`, `n_cargo`, `n_total` | 各船型当日**过境艘次**（transit events，非去重独立船数）                                          |
| 过境运力  | `capacity_tanker`, `capacity_container`, …, `capacity_cargo`, `capacity`                   | 各船型 **DWT-equivalent transit capacity**（估计）及合计；`capacity_tanker` **非**实际装载原油吨数 |


建模时主要用 `n_tanker` / `capacity_tanker`（液货船过境强度 + 过境运力代理），再派生 `tanker_share`、`avg_tanker_size` 等（见 §4.2、§5）。

### 8.4 `portwatch_ports_daily.csv` — 港口级油轮进出口

**规模**：38,080 行 × 9 列 | **频率**：日频 | **覆盖**：2019-01-01 ~ 2026-06-12

**空间单元**：14 个油轮枢纽（9 出口 + 5 进口/炼化），**每日每港一行**（角色见 §8.1 枢纽表）。


| 字段                                      | 含义                                           |
| --------------------------------------- | -------------------------------------------- |
| `date`, `portid`, `portname`, `country` | 日期、港口标识、国别                                   |
| `portcalls_tanker`                      | 当日油轮靠港/活动次数                                  |
| `import_tanker`                         | 油轮**进口吨位**（公吨，AIS 吃水估计）                      |
| `export_tanker`                         | 油轮**出口吨位**（公吨）                               |
| `short`, `role`                         | 脚本加的短码（如 `primorsk`）与角色（`export` / `import`） |


与咽喉表的区别：本表有**方向性**（import vs export），用于构造 `pw_tanker_exp_imp_asym` 等（见 §4.3）；咽喉表只有双向过境总量。

### 8.5 `gfw_chokepoint_vessel_presence_monthly.csv` — GFW AIS 船舶存在

**规模**：1,008 行 × 12 列 | **频率**：月频 | **覆盖**：2012-01 ~ 2025-12（6 咽喉 × 168 月）

**空间单元**：与 PortWatch 相同的 6 个咽喉，但使用 GFW 脚本定义的**矩形 bbox 多边形**（非 PortWatch 官方几何）。


| 字段                                                                                   | 含义                |
| ------------------------------------------------------------------------------------ | ----------------- |
| `date`                                                                               | 月份（如 `2012-01`）   |
| `chokepoint`                                                                         | 咽喉名称              |
| `total_hours`                                                                        | 多边形内船舶存在总时长（船·小时） |
| `total_vessels`                                                                      | 唯一船舶数             |
| `cargo_hours` / `bunker_hours` / `other_hours` / `fishing_hours` / `passenger_hours` | 按 GFW 船型拆分的存在时长   |
| `cargo_vessels` / `bunker_vessels` / `other_vessels`                                 | 各船型船舶数            |


GFW 以渔业为主，油轮多归入 `OTHER` 或 `BUNKER`，故用 `total_hours` / `total_vessels` 作海运活动代理，再派生存在强度指标 `mean_presence_hours_per_vessel`（见 §9.1–§9.2）。历史比 PortWatch 长（2012+），但粒度为**月**而非日。

### 8.6 三源关系（汇总）

```
PortWatch 咽喉  →  日频过境流量/运力（6 咽喉，2019+，无方向）
PortWatch 港口  →  日频油轮进出口吨位（14 枢纽，2019+，有方向）
GFW            →  月频 AIS 存在时长/船数（6 咽喉，2012+，补充早期 + 拥堵代理）
```

更新原始数据：在 `data/raw/03_shipping/` 对应子目录重跑上述三个 `download_*.py`，再执行 §7 复现命令重建周频矩阵。

---

## 9. GFW 特征：文献支撑、经济含义与分级

本节记录 GFW 派生特征的**文献/官方依据**、**支撑强度**、**论文表述边界**，以及建议的 **core / extended** 分层。评估日期：2026-07-03。

### 9.1 文献支撑评估表


| 特征                                                        | 支撑强度              | 判断                                                                                                                  | 建议                                                                                                                             |
| --------------------------------------------------------- | ----------------- | ------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------ |
| `gfw_{cp}_total_hours`                                    | **B+**            | GFW 官方原生指标，表示咽喉多边形内 AIS **船舶存在强度**（presence hours）。文献多用船数、靠港、运力、航次；较少直接把 presence hours 当作石油流量。                     | **保留（core）**。解释为「船舶存在强度 / 海运活动代理」，**不得**写成石油贸易量。                                                                               |
| `gfw_{cp}_total_vessels`                                  | **A−**            | 与文献中独立船舶数、交通量高度接近。P016 用不同油轮数量；ONS 发布 unique ships；IMF PortWatch 用每周到港/过境船数（见 [REF-GFW-1]、[REF-GFW-2]、[REF-GFW-4]）。 | **强烈保留（core）**。全船型计数，定位为**一般海运活动**，非油轮专属流量。                                                                                    |
| `gfw_{cp}_cargo_hours`                                    | **B+**            | 船型拆分有行业依据。ONS 将 cargo ships 与 tankers 从其他船型分离，以增强货物流动指标解释力（[REF-GFW-2]）。                                            | **保留（core）**。定位为一般货运活动或全球贸易需求代理，**非**原油运输量。                                                                                    |
| `gfw_{cp}_bunker_hours`                                   | **B−**            | 加油/补给船活动是合理海运服务指标；AIS 可用于识别 bunkering，但通常需锚泊识别、空间邻近与船对船匹配，不能只靠 polygon 内出现（[REF-GFW-3]）。                            | **可选（extended）**。必须称「加油/补给船存在时长」，**不得**称燃油补给量。                                                                                 |
| `gfw_{cp}_other_hours`                                    | **C**             | `OTHER` 为高度异质剩余类（可能含客船、服务船、分类不确定船；**GFW 下油轮多落在此类**）。无作为油价/原油物流核心指标的直接依据。                                            | **extended / 质量控制**。背景交通或分类质量变量，不作核心油运变量。                                                                                      |
| ~~`gfw_{cp}_nontanker_hours`~~（原 cargo+bunker） | **C / 已删除**     | `cargo + bunker` ≠ 全部「非油轮」；且 bunker/other 与 tanker 在 GFW 分类下**不互斥**。 | **已删除**（2026-07-03；build 脚本 + CSV + 三矩阵已重建）。如需货+补给合计，可由 `cargo_hours`+`bunker_hours` 现算。 |
| `gfw_{cp}_other_share` (= other/total)                    | **C+**            | 船型构成比例思路合理，但 `other/total` 无直接文献公式，解释力取决于 `other` 可靠性。                                                              | **extended**。交通构成或数据质量特征，非核心油运变量。                                                                                              |
| `gfw_{cp}_total_hours_mom_pct`                            | **B+**            | ONS 展示船舶交通量/访问次数的**月环比**，并与进出口增长率对照（[REF-GFW-2]）。                                                                   | **保留（core）**。稳健性可测 `diff(log1p(hours))` 以减少极端百分比。                                                                              |
| `gfw_{cp}_mean_presence_hours_per_vessel` (= total_hours/vessels；原名 `dwell_hours_per_vessel`) | **概念 B+；当前公式 C+** | P016 等用平均靠泊/turnaround/dwell 衡量拥堵，但通常需进出事件、低速/锚泊/轨迹分段。当前公式仅为**平均存在强度**（[REF-GFW-1]）。 | **已改名（core，2026-07-03）**为 `mean_presence_hours_per_vessel`；论文写 *coarse proxy for transit duration or congestion, not observed dwell time*。 |
| `gfw_all_total_hours_sum`                                 | **B（用法有问题）**      | ONS 有 all-ports 汇总先例（[REF-GFW-2]）；六通道覆盖重要石油航线。但 raw hours 求和受多边形面积、通过时间、AIS 覆盖、船型构成影响，且同一船可经多节点 → **重复计数**。         | **不建议 raw sum 作 core**。改为滚动训练窗内 z-score 均值 `gfw_all_activity_zmean`（建模阶段计算），或弃用汇总。                                             |


### 9.2 建议变量分级（建模口径）

**核心 GFW（6 咽喉 × 5 = 30 列）**——主 M3 消融与论文主表：


| 列模板                                       | 经济定位                                                       |
| ----------------------------------------- | ---------------------------------------------------------- |
| `gfw_{cp}_total_hours`                    | 船舶存在强度                                                     |
| `gfw_{cp}_total_vessels`                  | 月度唯一船舶数（全船型）                                               |
| `gfw_{cp}_cargo_hours`                    | 货运活动 / 贸易需求代理                                              |
| `gfw_{cp}_total_hours_mom_pct`            | 存在强度月环比变化                                                  |
| `gfw_{cp}_mean_presence_hours_per_vessel` | 每船平均存在强度（拥堵粗代理；已从 `dwell_hours_per_vessel` 改名，2026-07-03） |


**汇总（1 列，建模阶段派生，非 build 脚本静态列）**：

- `gfw_all_activity_zmean`：六咽喉 `total_hours` 在**各滚动训练窗**内 z-score 后取均值（防泄漏；替代 `gfw_all_total_hours_sum`）。

**扩展 / 稳健性（extended，6×3 = 18 列）**——扩展模型或消融，不进核心 M3：

- `gfw_{cp}_bunker_hours`
- `gfw_{cp}_other_hours`
- `gfw_{cp}_other_share`

**建议删除或仅留 extended 且改名**：

- ~~`gfw_{cp}_nontanker_hours`~~ → **已删除**（2026-07-03；build 脚本 + CSV + 三个合并矩阵已重建）。与 `cargo_hours`+`bunker_hours` 冗余，如需可现算。

> 当前 `m3_weekly_features.csv` 仍输出全部 55 列 GFW；core/extended 分层在 feature dictionary 或建模脚本中筛选即可，不必立即删列。

### 9.3 方法学边界（写作与稳健性）

**月频 → 周频 ffill**

- 合理混频对齐，但**不创造新周度信息**；同月约 4 周重复 → 「以周频表承载的月度状态变量」。
- GFW 4Wings API 支持 `DAILY` 分辨率；若重采，**日频聚合到 W-FRI 优于月 ffill**（当前下载脚本为 `MONTHLY`）。

**+4 周滞后**

- 有 ONS 约 4 周 reference-to-release 先例（[REF-GFW-2]），但**非 GFW 统一标准**。
- 论文表述：*A conservative four-week availability lag was imposed to prevent look-ahead bias and to reflect potential processing and publication delays.*
- **不要**写：*GFW data are officially released with a four-week lag.*
- 时间戳：`2012-01` 先对齐 `MonthEnd` 再 ffill、再 +4 周 → 1 月数据约 3 月初可用，可能偏保守；建议 lag ∈ {1, 4, 8} 周稳健性。

**GFW 船型与油轮**

- 下载脚本注明：GFW 以渔业为主，油轮多归入 `OTHER` 或 `BUNKER`；不同 API/接口船型枚举不完全一致（如无独立 `TANKER`、或有 `BUNKER_OR_TANKER`）→ 不能假设 cargo/bunker/other 与 tanker 互斥。

**文献因果方向（P016 / P017）**

- P016、P017 主要支持 **oil price → shipping activity**（靠港频率、独立船数、停留时间等）。
- **并不直接证明** shipping activity → **future** oil price；样本外预测有效性仍须 **M1 vs M1+M3 消融 + DM 检验**。
- AIS 推算原油贸易与官方数据总体较吻合，但国别/时间上有管道、转运等差异 → 变量定位为**粗代理**。

**空间单元用语**

- 六处宜称 **six oil-critical maritime transit corridors**；好望角为重要石油贸易/绕行路线，EIA 口径下** technically 非狭义 chokepoint**（与 PortWatch 命名对齐时可脚注说明）。

### 9.4 参考文献与官方来源


| 编号          | 类型       | 文献 / 来源                                                                                                                                                                                                        | URL                                                                                                                                                                                                                                                      | 本项目用途                                                                                                 |
| ----------- | -------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------- |
| [REF-GFW-1] | 期刊       | **P016** — The Impact of the Crude Oil Price on Tankers' Port-Call Features: Mining the Information in Automatic Identification System. *Journal of Marine Science and Engineering*, 10(10), 1559. MDPI, 2022. | [https://www.mdpi.com/2077-1312/10/10/1559](https://www.mdpi.com/2077-1312/10/10/1559)                                                                                                                                                                   | 独立油轮数、靠港频率、**port dwell time** 构念；支持 `total_vessels` 与存在强度类变量，**非**当前 `hours/vessels` 公式的严格 dwell 定义。 |
| [REF-GFW-2] | 官方报告     | **ONS Data Science Campus** — *Faster indicators of UK economic activity: more timely and relevant shipping indicators* (AIS shipping indicators project).                                                     | [https://datasciencecampus.ons.gov.uk/projects/faster-indicators-of-uk-economic-activity-improving-the-shipping-indicators/](https://datasciencecampus.ons.gov.uk/projects/faster-indicators-of-uk-economic-activity-improving-the-shipping-indicators/) | unique ships、cargo/tanker 分离、**MoM 变化率**、多港口汇总指标、约 **4 周发布滞后**先例。                                     |
| [REF-GFW-3] | 期刊       | Extraction of Bunkering Services from Automatic Identification System Data and Their International Comparisons. *Sustainability*, 15(24), 16711. MDPI, 2023.                                                   | [https://www.mdpi.com/2071-1050/15/24/16711](https://www.mdpi.com/2071-1050/15/24/16711)                                                                                                                                                                 | `bunker_hours` 仅作存在代理；真正 bunkering 需锚泊/船对船识别。                                                         |
| [REF-GFW-4] | 官方 / 方法论 | **IMF PortWatch** — Arslanalp et al. (2026, WP/26/99); Arslanalp, Marini & Tumbarello (2019, WP/19/275). 日频过境/港口 AIS 指标。                                                                                       | [https://portwatch.imf.org/](https://portwatch.imf.org/)                                                                                                                                                                                                 | `total_vessels` 的对照：IMF Weekly/daily transit & port-call 口径；M3 中 PortWatch 为主、GFW 为 2012+ 补充。         |
| [REF-GFW-5] | API 文档   | **Global Fishing Watch** — 4Wings Report API, dataset `public-global-presence:latest`.                                                                                                                         | [https://globalfishingwatch.org/our-apis/](https://globalfishingwatch.org/our-apis/)                                                                                                                                                                     | presence hours/vessels 原生定义；支持 MONTHLY/DAILY；数据通常更新至约 **96 小时前**（与 +4 周保守滞后区分）。                       |


**项目内交叉引用**：P016 → `01_literature/reading_notes/03 Shipping/P016.md`；PortWatch 方法论 → `external_sources.md` §M3；GFW 下载 → `data/raw/03_shipping/GFW/download_gfw_vessel_presence.py`。

---

## 10. PortWatch 特征：文献支撑、分级与稳健性模型设计

本节记录 PortWatch 特征族的**文献/官方依据**、**原生 vs 派生性质**、建议的 **core / extended / control / aggregate / directional 分层**，以及一组 **稳健性/消融模型设计**（Core / B / C / D / E）。评估日期：2026-07-03。

> ⚠️ **本节为设计文档，代码未改**：分层与消融臂尚未写入 `code/scripts/flat/M3_Flat/robustness_m3.py` 的 `select_m3_arm()`，也未重跑任何模型；字段定义与单位见 §4.2–§4.3、§5，此处不重复。

### 10.1 文献支撑评估表（原生 vs 派生）

PortWatch 是 IMF 为监测海运贸易与关键航道中断构建的 AIS 指标体系，原生提供分船型过境次数与运力估计，文献/官方支撑**强于 GFW 组**。但 57 列咽喉特征中**仅 4 类为 PortWatch 原生**，其余为作者派生/汇总。

| 特征 | 支撑强度 | 性质 | 判断 |
| ---- | -------- | ---- | ---- |
| `pw_{cp}_n_tanker` | **A** | PortWatch 原生 | 强支撑，M3 核心。**tanker 过境艘次**（liquid bulk，非严格原油油轮、非独立船数），见 §4.2 |
| `pw_{cp}_n_total` | **A** | PortWatch 原生 | 强支撑，但油价针对性低于 tanker；宜作控制变量 |
| `pw_{cp}_capacity_tanker` | **A** | PortWatch 原生 | 强支撑，M3 核心。**DWT-equivalent transit capacity proxy**，非实际装载原油吨数 |
| `pw_{cp}_capacity` | **A** | PortWatch 原生 | 强支撑，但更接近总体海运贸易；宜作控制变量 |
| `pw_{cp}_tanker_share` | **B+** | 作者派生 | 船型构成比例，经济含义清晰（相对 `n_total` 归一） |
| `pw_{cp}_tanker_cap_share` | **B+** | 作者派生 | 运力构成；与 `tanker_share` 相关（hormuz corr≈0.72），不与其同入核心 |
| `pw_{cp}_avg_tanker_size` | **B+** | 作者派生 | 平均 DWT/艘次；与 P070 船型拆分逻辑一致，**非** P070 油价实证变量。⚠️ `≡ capacity_tanker / n_tanker`（精确恒等） |
| `pw_{cp}_n_tanker_wow_pct` | **B** | 作者派生 | 周环比常用，但低基数/零值敏感；稳健性建议 `Δlog1p` |
| `pw_{cp}_capacity_tanker_4w_ma` | **B+** | 作者派生 | trailing 平滑，合理（须 trailing，非 centered；本项目已 trailing + 滞后） |
| `pw_all_n_tanker_sum` | **B** | 作者汇总 | 六航道 tanker 过境艘次之和（**跨航道可重复计数**） |
| `pw_all_n_total_sum` | **B−** | 作者汇总 | 一般贸易意义，对原油机制较弱；可降级 |
| `pw_all_tanker_share` | **B+** | 作者汇总 | ratio-of-sums，网络层 tanker intensity，口径正确 |

> **写作口径**：不宜写「57 列均为文献既有变量」；应写 *The PortWatch feature family combines native IMF AIS-based transit-count and vessel-capacity indicators with author-derived composition, growth, smoothing, and cross-corridor aggregation features.*

### 10.2 建议变量分级（建模口径）

| 层 | 列 | 数量 | 定位 |
| -- | -- | ---- | ---- |
| **核心 core** | `pw_{cp}_n_tanker`、`pw_{cp}_capacity_tanker` | 6×2=12（或 Hormuz/Suez/Malacca 3×2=6） | 原生、最强支撑、最贴近原油运输机制（数量 + 运力） |
| **扩展 extended** | `tanker_share`、`tanker_cap_share`、`avg_tanker_size`、`n_tanker_wow_pct`、`capacity_tanker_4w_ma` | 6×5=30 | 有依据的派生；消融/robustness |
| **控制 control** | `n_total`、`capacity` | 6×2=12 | 全球贸易/总体海运背景，非油运核心 |
| **汇总 aggregate** | `pw_all_n_tanker_sum`、`pw_all_tanker_share`（保留）；`pw_all_n_total_sum`（降级） | 3 | 网络层总体活动 |
| **方向性 directional** | 港口级 7 列（`pw_exp_hubs_export_vol`、`pw_imp_hubs_import_vol`、`pw_tanker_exp_imp_net`/`_asym`/`_log_ratio`/`_asym_4w_ma`、`pw_exp_hubs_export_vol_wow_pct`） | 7 | 出口-进口方向不对称，独有信息，建议保留 |

合计 12+30+12+3+7 = **64 列 PortWatch**（与 CSV 一致）。

> 当前建模管线 `code/src/backtest/data.py::select_features()` **按 modality 标签选列**，故现状是 **64 列全部进 M3/M4**，靠 Ridge L2 / XGB + 调参隐式降权；如要「仅核心进模型」，需加 `tier` 标签或新增消融臂（见 §10.3）。

### 10.3 稳健性 / 消融模型设计（Core / B / C / D / E）

> 约定：**每个臂 = M1 + 下列 PortWatch 子集**，与现有 `robustness_m3.py` 各臂一致，对 M1 做增量检验（Clark–West / DM）。下列仅列 PortWatch 部分。

| 臂 | 组成 | 列数 | 回答的问题 | 状态 |
| -- | ---- | ---- | ---------- | ---- |
| **PortWatch_Core** | 6×`n_tanker` + 6×`capacity_tanker` | 12 | 原生「数量 + 运力」是否足够 | 设计 |
| **B. Composition** | Core + 6×`tanker_share` | 18 | 相对构成（占总交通比例）是否有增量 | 设计 |
| **C. Smoothed** | 6×`n_tanker` + 6×`capacity_tanker_4w_ma` | 12 | 平滑 capacity（降噪）是否改变结果（固定 `n_tanker` 的受控 A/B） | 设计 |
| **D. Size（修正版）** | 6×`n_tanker` + 6×`avg_tanker_size` | 12 | 船型规模维度 vs 总运力维度（**替换** capacity，非叠加） | 设计 |
| **E. Aggregate benchmark** | `pw_all_n_tanker_sum` + `pw_all_tanker_share`（**独立，不与六航道局部同入**） | 2 | 仅全球总体航道活动是否已足以预测 | 设计 |

**逐臂说明与判断：**

- **B ✅**：`tanker_share = n_tanker/n_total` 引入 `n_total` 归一 → 真·新增维度；**不加** `tanker_cap_share`（与 `tanker_share` corr≈0.72，共线）。
- **C ✅**：固定 `n_tanker`、仅 `capacity → capacity_4w_ma`，是隔离「平滑效应」的干净受控对比；`capacity_tanker_4w_ma` 为 trailing + 已滞后，无前视。
- **D ⚠️→已修正**：`avg_tanker_size ≡ capacity_tanker / n_tanker`（**精确恒等**，误差 ~3.7e-9；log 空间与 `n_tanker`、`capacity_tanker` 完全共线）。故**不采用**「Core + avg_size（18 列）」的叠加设计（会引入恒等冗余），改为**替换** capacity_tanker → avg_tanker_size（12 列），把「(count, 总运力)」重参数化为「(count, 平均船型)」，干净隔离 size 维度。
- **E ✅（带解释 caveat）**：`pw_all_n_tanker_sum` 为跨 6 航道**原始求和**（同一船经多咽喉会**重复计数**、各航道尺度不同）→ 是「航道网络活动指数」，非全球独立船数；`pw_all_tanker_share` 为 ratio-of-sums，口径正确。作独立精简基准，不与局部变量同入主模型。

**命名待办**：`avg_tanker_size` 若要用名 `mean_tanker_dwt_per_transit`，需先在 `aggregate_shipping_to_weekly.py` 改名并重建（同 `nontanker→cargo_bunker` 流程）；否则消融臂直接引用现名 `pw_{cp}_avg_tanker_size`。

**多重比较**：Core/B/C/D/E + 现有 full/pw-only/gfw-only/tanker-only 臂数较多；须**预先登记为 robustness、全部汇报**（不挑最优 arm），写作时说明多重比较，避免 p-hacking 质疑。

### 10.4 方法学边界（写作与稳健性）

- **`capacity = capacity_cargo + capacity_tanker` 非精确恒等**：本地数据仅约 50% 行满足（ArcGIS 导出含取整/定义差异）；勿在论文写严格等式。`n_total = n_cargo + n_tanker` 则本地 100% 成立。
- **`+1 周` 为作者保守信息滞后**，非 PortWatch 官方发布滞后（PortWatch 近实时）。表述：*author-imposed one-week information lag to avoid using the incomplete/revisable current week*；稳健性可测 lag 0 vs 1。
- **51.5% 覆盖率是 2019 前结构性不可用**（PortWatch 自 2019-01），非 2019+ 期内缺失：标准窗 2019.1–2025.12 内约 99.7%。写作应拆分「全样本 51.5%」与「2019+ 子样本≈完整」，并说明含 PortWatch 的层须与长样本 M0–M2 分开公平比较。
- **高共线性**是本组最大统计问题（非缺文献）：`capacity_tanker ≡ n_tanker×avg_size`、`tanker_share`/`tanker_cap_share` 相关、cap 的 4w_ma 等 → 对 Ridge 有多重共线，对 XGB 会稀释 SHAP、增小样本过拟合风险。365 周样本 × 数十列 → **核心收缩 + 分层消融**比全列同入更稳。
- **ArcGIS 日期防御**：World Bank 教程报告过 ArcGIS 带时区时间戳把 1/1 归到上年 12/31 的情况；本地 `portwatch_chokepoints_daily.csv` 实测 `date` 与 `year/month/day` **0 处不一致**（当前无此问题）。稳健做法仍可在 build 用 `pd.to_datetime(dict(year=,month=,day=))` 重建日期再聚合。
- **因果方向**：P016/P017/P070 支持 price→shipping 的构念与关联，**未**证明 shipping→future price 的样本外预测力；增量价值须 M1 vs M1+M3 消融 + DM/CW。P070 另提醒 **tanker trade value 由燃料价格指数构造 → 会内嵌油价**，本项目故优先用**物理量**（艘次/运力）而非 trade value，防泄漏。

### 10.5 参考文献与官方来源

| 编号 | 类型 | 文献 / 来源 | URL | 本项目用途 |
| ---- | ---- | ----------- | --- | ---------- |
| [REF-PW-1] | 官方 / 方法论 | **P070** — Arslanalp et al. (2026). *Nowcasting Country-level Trade Estimates using IMF PortWatch*. IMF WP/26/99. | https://www.imf.org/-/media/files/publications/wp/2026/english/wpiea2026099-source-pdf.pdf | 船型拆分（tanker/liquid bulk、dry bulk、container）、吃水实物量、tanker trade value 内嵌油价的泄漏警示 |
| [REF-PW-2] | 官方 / 方法论 | Arslanalp et al. (2025). *Nowcasting Global Trade from Space*. IMF WP/25/93. | https://www.imf.org/-/media/Files/Publications/WP/2025/English/wpiea2025093-print-pdf.ashx | transit calls 作海运活动基础指标；关键航道 2019+；24+ critical maritime passages |
| [REF-PW-3] | 期刊 / 方法论 | Arslanalp, Marini & Tumbarello (2019). *Big Data on Vessel Traffic: Nowcasting Trade Flows in Real Time*. IMF WP/19/275. | https://www.imf.org/en/publications/wp/issues/2019/12/13/big-data-on-vessel-traffic-nowcasting-trade-flows-in-real-time-48837 | PortWatch 前身方法；运力(DWT) 优于单纯船数 |
| [REF-PW-4] | 官方教程 | **World Bank** — Chokepoints Monitor（Alternative Data for Crisis）方法论 | https://worldbank.github.io/alternative-data-for-crisis/notebooks/disruptions-business-trade/chokepoints-monitor.html | `n_tanker`/`n_cargo`/`n_total`/`capacity` 用法、7 日滚动均值、ArcGIS 时区日期 caveat、危机期对比 |
| [REF-PW-5] | 官方 | **EIA** — Oil tanker sizes（AFRA：Handysize/Aframax/Suezmax/VLCC，按 DWT 分级） | https://www.eia.gov/todayinenergy/detail.php?id=17991 | DWT 是 tanker size 标准尺度；支撑 `avg_tanker_size` 的行业含义 |
| [REF-PW-6] | 官方报告 | **UNCTAD** — Review of Maritime Transport（船队按 数量 + 总 DWT + average vessel size 报告） | https://unctad.org/en/PublicationChapters/rmt2017ch2_en.pdf | `avg_tanker_size` 的行业先例（count / total DWT / avg size 并报） |
| [REF-PW-7] | 数据接口 | **IMF PortWatch ArcGIS FeatureServer** — `Daily_Chokepoints_Data` / `Daily_Ports_Data`（org `weJ1QsnbMYJlCHdG`） | https://services9.arcgis.com/weJ1QsnbMYJlCHdG/ArcGIS/rest/services/ | 原生字段 `n_*`/`capacity_*`/`import_tanker`/`export_tanker` 出处 |

**项目内交叉引用**：P070 → `01_literature/reading_notes/03 Shipping/P070.md`；PortWatch 下载 → `data/raw/03_shipping/IMF Portwatch/download_portwatch_{chokepoints,ports}.py`；建模选列 → `code/src/backtest/data.py`；消融臂 → `code/scripts/flat/M3_Flat/robustness_m3.py`（待新增 Core/B/C/D/E）。

---

## 11. M3 主模型（full）与 core tier 稳健性臂（4.1–4.4 汇总）

> **主模型（2026-08-14 起）= full tier：164 航运列（GFW 49 + PortWatch 64 + SAR 暗船 51）**。113 列 AIS 宽表仍为 core 之前的 full 航运块；17 区 GFW SAR（`sar_*_{total,dark,share}`）并入合并矩阵，与 Deep 图节点同一数据源、同一 +4 周滞后。§11.1 的 38 列 core 仍为稳健性臂。
>
> **代码现状（2026-08-14）**：`m3_weekly_features.csv` 仍为 GFW 49 + PortWatch 64；SAR 51 列在 `build_feature_matrix.py` 从 `m3_graph_darkvessel_weekly.csv` 并入合并矩阵。`select_features(..., m3_tier="full")` 默认选全部 164 航运列。`gfw_all_activity_zmean` 仍为建模阶段派生、仅 `gfw-aggregate` 实验注入。`--m3-tier core` 仍为 38 列稳健性臂。

### 11.1 core tier 清单（38 列，稳健性臂；每咽喉短码：hormuz/suez/malacca/mandeb/panama/cape）

**4.1 GFW 核心（6×4 = 24）**

| # | 列（× 6 咽喉） | 经济定位 |
| - | ------------ | -------- |
| 1 | `gfw_{cp}_total_hours` | 船舶存在强度 |
| 2 | `gfw_{cp}_total_vessels` | 月度唯一船舶数（全船型） |
| 3 | `gfw_{cp}_cargo_hours` | 货运活动 / 贸易需求代理 |
| 4 | `gfw_{cp}_total_hours_mom_pct` | 存在强度月环比 |

> **不进同一主模型**（各自独立实验，见 §11.4）：
> - `gfw_{cp}_mean_presence_hours_per_vessel`（6 列）→ **GFW-Presence 替换实验**
> - `gfw_all_activity_zmean`（1 列，建模阶段 leak-free 派生）→ **GFW-Aggregate 汇总基准**

**4.2 PortWatch 咽喉（6×2 = 12；可选精简 3×2 = 6：hormuz/suez/malacca）**

| # | 列（× 6 咽喉） | 经济定位 |
| - | ------------ | -------- |
| 1 | `pw_{cp}_n_tanker` | tanker（液货）过境艘次 |
| 2 | `pw_{cp}_capacity_tanker` | tanker 过境运力（DWT proxy） |

**4.3 PortWatch 港口方向性（2）**

| # | 列 | 经济定位 |
| - | -- | -------- |
| 1 | `pw_exp_hubs_export_vol` | 选定出口枢纽 tanker 出口吨位估计 |
| 2 | `pw_imp_hubs_import_vol` | 选定进口枢纽 tanker 进口吨位估计 |

**4.4 availability masks（主模型 = 0 列）**

- 2019–2025 common-window 主实验：`avail_*` 近乎常数 → **不进主模型**（现管线已标 `modality='mask'`，本就不入模型）。
- 2012–2025 长样本 / 缺失模态模型：保留 `avail_gfw`、`avail_pw_chokepoints`、`avail_pw_ports` 作**门控**（非经济预测变量）；`avail_shipping` 冗余（确定性 OR）建议删。

**core tier 合计**：GFW 核心 24 + PortWatch 咽喉 12 + 港口方向 2 = **38 列**（3×2 精简版为 32）。此为**稳健性臂之一（非主模型）**；主模型用 full 113 列。增量价值靠 M1 vs M1+M3 消融 + CW/DM。

### 11.2 扩展 / 稳健性 / 附录（不进主模型，保留在 CSV）

| 组 | 列 |
| -- | -- |
| GFW 扩展 | `gfw_{cp}_bunker_hours`、`gfw_{cp}_other_hours`、`gfw_{cp}_other_share`（6×3=18） |
| PortWatch 咽喉扩展 | `tanker_share`、`tanker_cap_share`、`avg_tanker_size`、`n_tanker_wow_pct`（→ 稳健性改 `wow_logchg`）、`capacity_tanker_4w_ma`（6×5=30） |
| PortWatch 控制 | `pw_{cp}_n_total`、`pw_{cp}_capacity`（6×2=12） |
| PortWatch 汇总 | `pw_all_n_tanker_sum`、`pw_all_tanker_share`（保留）；`pw_all_n_total_sum`（降级）；`gfw_all_total_hours_sum`（弃用） |
| 方向性扩展 | `pw_tanker_exp_imp_log_ratio`（→ 训练窗中心化 `_dev`，扩展首选）、`net`、`asym`、`asym_4w_ma`、`export_vol_wow_pct`（→ `wow_logchg`） |
| 消融臂 | PortWatch_Core / B_Composition / C_Smoothed / D_Size / E_Aggregate（见 §10.3） |

### 11.3 GFW 分离实验（不进主模型）

`mean_presence` 与 `gfw_all_activity_zmean` 与主模型核心高度相关或为其确定性函数（`mean_presence ≡ total_hours/total_vessels`；zmean 为 6 咽喉 `total_hours` 的汇总），故不与核心同入一个主模型，改各自单独检验：

| 实验 | arm（`robustness_m3.py`） | 组成（+ M1） | 回答的问题 |
| ---- | ------------------------ | ------------ | ---------- |
| **GFW-Presence 替换实验** | `gfw-presence` | GFW 核心 24 + 6×`mean_presence_hours_per_vessel` = 30 | 每船平均存在强度（dwell/拥堵粗代理）相对 GFW 核心是否有增量 |
| **GFW-Aggregate 汇总基准** | `gfw-aggregate` | `gfw_all_activity_zmean`（1，建模阶段派生） | 仅一个全网络活动汇总指数是否已足以预测 |

对照臂：`core`（稳健性 38，曾为旧主模型）、`gfw-only`（全 GFW）、`full`（现主模型，全航运）。

### 11.4 实现状态

**已实现（2026-07-03）**

1. ✅ `select_features(dico, modality, m3_tier="full")`：**M3/M4 主模型默认用全部 113 航运列**（2026-07-05 起）；`--m3-tier core` 切回 38 列 core 稳健性臂（`data.py`：`GFW_CORE_SUFFIXES`(4) / `gfw_core_columns` / `PW_CHOKE_CORE_SUFFIXES` / `PW_PORTS_CORE` / `m3_core_columns()` 仍保留供 core 臂与 GFW 子实验）。
2. ✅ GFW 分离实验选择器：`data.gfw_core_columns()`(24) / `data.gfw_presence_columns()`(6) / `data.GFW_ZMEAN_COL`；`robustness_m3.py` 新增 `core / gfw-presence / gfw-aggregate` 臂（默认 arms 已含）。
3. ✅ `gfw_all_activity_zmean`：`add_gfw_activity_zmean()` 在 `build_dataset` 内派生（过去-only 扩展窗 z-score 均值，leak-free），仅在 `gfw-aggregate` 实验请求时注入；CSV/矩阵无需改。
4. ✅ 冒烟自检：`run_baseline --modality M3` → 2026-07-05 起 `144 raw = 31 M1 + 113 M3 full`（旧 core 为 69）；end-to-end 通过。

**待实现（可选）**

5. 重跑正式 baseline/sweep/shap（M3、M4）与 `robustness_m3.py`（含新臂）以刷新 `results/`（本次仅冒烟测试，未覆盖正式产物）。
6. 可选改名：`avg_tanker_size → mean_tanker_dwt_per_transit`、`*_wow_pct → *_wow_logchg`、方向 `net → selected_hubs_export_import_spread_mt`。
7. 长样本缺失模态模型删 `avail_shipping`、按需合并 `avail_pw_*`。

---

## 12. Stage-2 动态异质图 processed 产物（节点 / O-D 边 / 暗船 / EMODnet）

> **定位**：§1–§11 描述的是**扁平 M3**（6 咽喉聚合宽表 `m3_weekly_features.csv`，进 M0–M4 基线）。本节描述**阶段 2「动态异质图 G(t)」**的 processed 产物，把航运升级为 **11 AOI 节点 + 有向 O-D 边 + 暗船**的图结构，供创新层 GNN/时空模型使用。
> 构建脚本：`data/processed/M3/py/build_m3_graph_weekly.py`（图张量）、`build_emodnet_weekly.py`（EMODnet）。原始来源与采集见 `external_sources.md`「M3 Stage-2」节。最后更新：2026-07-03。

### 12.1 产物一览

| 产物 | 形态 | 形状 | 说明 |
| ---- | ---- | ---- | ---- |
| `outputs/m3_graph_nodes_weekly.csv` | long（week × site） | 4301 行（391 周 × 11 站） | 11 AOI 节点周频特征（11 列 + `avail_node`） |
| `outputs/m3_graph_edges_weekly.csv` | long（week × from × to） | 30272 行 | AOI→AOI 有向 O-D 边（仅跨节点） |
| `outputs/m3_graph_darkvessel_weekly.csv` | long（week × region） | 6647 行 | 17 区域（11 AOI + 6 咽喉）SAR 暗船 |
| `outputs/m3_graph_tensors.npz` | 张量 | node `(391,11,11)` + adj `(391,11,11)` | 直接喂 GNN 的节点特征张量 + 邻接张量 |
| `outputs/m3_emodnet_density_weekly.csv` | long（week × region） | 5491 行 | EMODnet 船舶密度 zonal（⚠️ 仅 Rotterdam+Suez 有效） |
| `outputs/m3_graph17_tensors.npz` | 张量 bundle | aoi(391,11,11)+choke(391,6,20)+adj(391,17,17) | **完整 17 节点异质图**（11 AOI + 6 咽喉），见 §12.9 |
| `outputs/m3_graph17_choke_nodes_weekly.csv` | long（week × chokepoint） | 2346 行 | 6 咽喉节点特征（可读审计） |

时间索引统一 `week_ending_friday`（W-FRI）；比较窗 2019-01-04 起（图张量 391 周至 2026-06-26，含全量，建模裁 2019–2025）。

### 12.2 节点表 `m3_graph_nodes_weekly.csv`（11 特征）

| 字段 | 含义 | 来源 | 频率化 | 滞后 |
| ---- | ---- | ---- | ------ | ---- |
| `pw_portcalls_tanker` | 油轮港口停靠数（周和） | PortWatch AOI 日频 | 日→W-FRI sum | +1 周 |
| `pw_portcalls_cargo` | 货船停靠数 | PortWatch | 日→周 sum | +1 周 |
| `pw_import_tanker` / `pw_export_tanker` | 油轮进/出口吨位（周和） | PortWatch | 日→周 sum | +1 周 |
| `gfw_n_visits` | 该周到达该 AOI 的 port-visit 数 | GFW port visits | 事件→周 count | +2 周 |
| `gfw_dwell_hrs_mean` / `_median` | 停靠停留时长（**节点级 dwell**，均值/中位） | GFW `port_visit.durationHrs` | 事件→周聚合 | +2 周 |
| `gfw_self_loops` | 同 AOI 连续重复停靠数（周转代理） | GFW voyages（self-loop） | 事件→周 count | +2 周 |
| `sar_detections_total` / `sar_detections_dark` | SAR 总/暗船检测数 | GFW SAR（AOI） | 月末 ffill→周 | +4 周 |
| `sar_dark_share` | 暗船占比 = dark/total | GFW SAR | 月频派生 | +4 周 |
| `avail_node` | 该 (周,站) 任一特征非空 = 1 | 派生 | — | — |

> **裁剪**：`gfw_dwell_hrs_*` 仅计 `durationHrs ≤ 720h`（30 天）的停靠，超长（AIS 长期在港/拼接异常，raw 有 13 年个例）置 NaN。

### 12.3 O-D 边表 `m3_graph_edges_weekly.csv`

| 字段 | 含义 |
| ---- | ---- |
| `week_ending_friday` | 到达周（按到达 to_site 的 `arrive_time` 归入 W-FRI） |
| `from_site` / `to_site` | 有向边起点/终点 AOI（`from ≠ to`，self-loop 已剔除入节点特征） |
| `n_voyages` | 该周该有向 lane 的航次数（边权重） |
| `mean_transit_days` | 平均航段天数（`transit_days ∈ (0, 90]` 内均值；超 90 天=中途有未观测非 AOI 停靠，仅排除出均值，边仍计数） |

滞后 +2 周（同 GFW-event）。96 条 lane、总 106992 航次；Top：Ningbo→Singapore、Singapore→Ningbo、Fujairah↔Singapore、Singapore↔Rotterdam。

### 12.4 暗船表 `m3_graph_darkvessel_weekly.csv`（17 区域）

`region_type ∈ {aoi, chokepoint}`；`region_id` = P001–P011 或咽喉全名；`region_short` = 咽喉短码。字段 `detections_total / detections_dark / detections_matched / dark_share`，SAR 月末 ffill→周 + **4 周**滞后。空单元格 = 滞后边界尚不可得（NaN）。

> **研究亮点（周均暗船占比）**：Kharg 79% · Basra 56% · Hormuz 52% · Suez 45% · Malacca 37%（制裁"影子船队"信号）；合规港 Fujairah 7% · Jurong 7% · Ningbo 9% · Rotterdam 12%。

### 12.5 图张量 `m3_graph_tensors.npz`

| 键 | 形状 | 含义 |
| -- | ---- | ---- |
| `weeks` | (391,) | W-FRI 字符串轴（2019-01-04 ~ 2026-06-26） |
| `site_ids` | (11,) | 节点顺序 P001…P011 |
| `node_feature_names` | (11,) | §12.2 的 11 个特征名（顺序即张量最后一维） |
| `node_features` | (391, 11, 11) | (T, N, F) 节点特征；缺失 = NaN |
| `adjacency_n_voyages` | (391, 11, 11) | (T, N, N) 有向邻接（边权 = `n_voyages`，无边 = 0，对角=0） |

**自检通过**：有向性 `P006→P004`(15261) ≠ `P004→P006`(14233)；三源滞后方向均为「首个可用周后移 = 发布滞后、非前视」。

### 12.6 EMODnet 表 `m3_emodnet_density_weekly.csv`

| 字段 | 含义 |
| ---- | ---- |
| `emodnet_density_mean` / `_max` / `_sum` | 区域多边形内船舶密度统计（排除 nodata −9999 与负值） |
| `emodnet_n_valid_px` | 有效像素数（覆盖诊断） |

滞后 +8 周（EMODnet 发布延迟大）。⚠️ **覆盖限制（关键）**：EMODnet HA 为 **EPSG:3035 欧洲栅格**，17 区域中**仅 Rotterdam（`mean≈13`）与 Suez（`mean≈1`）落在覆盖内**，其余 15 区域全程 NaN；栅格仅 2017-2024（周表止于 2025-03，之后不 ffill 陈旧值）。故 EMODnet **仅作 Rotterdam 的 AIS 密度交叉验证**，不进图张量。

### 12.7 复现命令

```bash
cd data/processed/M3/py
python build_m3_graph_weekly.py                       # 节点+边+暗船+张量（默认滞后）
python build_m3_graph_weekly.py --no-lag              # 诊断（有泄漏）
python build_emodnet_weekly.py                        # EMODnet zonal（欧洲覆盖）
```

### 12.8 已知缺口 / 后续

- **11 AOI 核心张量 vs 完整 17 节点图**：`m3_graph_tensors.npz`（§12.5）是 O-D 边端点的 11 AOI 核心；**完整 17 节点异质图已组装**为 `m3_graph17_tensors.npz`（§12.9，含 6 咽喉节点 + AOI↔咽喉静态边）。
- **O-D 为 AOI 诱导子图**：边表示同船相邻 AOI 停靠，中途非 AOI 港口不观测。
- **油轮≈CARGO 近似**：port visits 无法在 events 层隔离油轮，油港节点 CARGO 近似油轮、综合港为混合货运（详见 `external_sources.md` caveat）。
- **未并入扁平主矩阵**：图张量/17 节点 bundle 是创新层专用结构，不进 `build_feature_matrix.py` 的扁平 M4 表；其消费者是 §12.10 的 z_ship 编码器。

### 12.9 完整 17 节点异质图 `m3_graph17_tensors.npz`

构建脚本 `build_m3_graph17.py`：把 §12.5 的 11 AOI 张量 + **咽喉节点特征**（扁平 `m3_weekly_features.csv` 的 `gfw_{cp}_*`/`pw_{cp}_*` + 暗船表咽喉行）+ **AOI↔咽喉静态边** 组装为 17 节点异质图。所有输入均已在上游滞后（AOI +1/+2/+4w、咽喉 gfw +4w/pw +1w、SAR +4w），**不再额外 shift**。

**节点顺序（固定）**：AOI 0–10 = `P001…P011`；chokepoint 11–16 = `hormuz, suez, malacca, mandeb, panama, cape`。**异质**：AOI 与咽喉特征空间不同（`F_aoi=11 ≠ F_choke=20`），由编码器用类型专属输入投影统一。

| npz 键 | 形状 | 含义 |
| ------ | ---- | ---- |
| `weeks` | (391,) | W-FRI 轴 2019-01-04 ~ 2026-06-26 |
| `node_ids` / `node_types` | (17,) | 节点 id 与类型（aoi/chokepoint） |
| `aoi_feature_names` / `choke_feature_names` | (11,)/(20,) | 特征名（张量最后一维顺序） |
| `aoi_features` | (391, 11, 11) | AOI 节点特征（同 §12.2） |
| `choke_features` | (391, 6, 20) | 咽喉节点特征 = gfw(8) + pw(9) + sar(3) |
| `adjacency_od` | (391, 11, 11) | 动态有向 O-D（AOI 块） |
| `static_edges` | (17, 17) | AOI↔咽喉静态边（13 条无向） |
| `adjacency` | (391, 17, 17) | 组合邻接（O-D 动态 + 静态广播，平均 65.8 边/周） |

**静态边**（`aoi_oil_infrastructure_sites.md` §4）：hormuz–{P002,P003,P007,P008,P010}、suez–{P001,P011}、malacca–{P004,P006,P009}、mandeb–P011、cape–P001、panama–P005。其中 P007（Jamnagar）为需求侧炼厂，原油进料以波斯湾装载为主，故在进口侧连接 hormuz，使每个 AOI 至少有一条走廊边。配套可读长表 `m3_graph17_choke_nodes_weekly.csv`（6 咽喉 × 391 周）。复现：`python build_m3_graph17.py`。

### 12.10 z_ship 编码器 `code/src/models/shipping_encoder.py`

创新层**航运模态分支**（研究计划 §5.1「1–2 层 GAT + 小 TCN → 32 维」）。消费 §12.9 的 17 节点时序图，输出 `z_ship (B,32)` + `site_att (B,17)`（节点重要度，供 RQ3 模态内可解释）。

**架构**：类型专属投影（`F_aoi/F_choke → d_model`）+ node-type embedding → **2 层 dense 多头 GAT**（邻接对称化 + 自环掩码，融合动态 O-D 与静态 AOI↔咽喉边）→ **因果 TCN**（lookback L 周时序）→ 节点注意力池化 → MLP head → 32 维。约 **4.2 万参数**（小样本友好，呼应 §8「编码器维度小 + 强正则」）。

| 接口 | 说明 |
| ---- | ---- |
| `ShippingGraphEncoder(f_aoi=11, f_choke=20, d_model=64, d_out=32, gat_layers=2, heads=4, tcn_layers=2)` | 编码器构造 |
| `forward(aoi_feat (B,L,11,11), choke_feat (B,L,6,20), adj (B,L,17,17))` | → `(z_ship (B,32), site_att (B,17))` |
| `load_graph17_windows(npz, lookback=8)` | 切滚动窗口（含标准化；⚠️ smoke 用全局 z-score，**训练须改 past-only expanding**） |

**冒烟自检**（`python shipping_encoder.py`）：384 个 lookback=8 窗口 → `z_ship (32,32)` 全 finite、`site_att` 行和=1.0、反向 `grad_norm≈36`（可训练）。

> ✅ **已接入端到端回测**（见 §12.11）：z_ship 与金融 z_fin 的门控融合 + rolling-origin 单任务回归训练已实现并跑出结果。遥感 z_rs（冻结 Prithvi EO）分支尚未接入。

### 12.11 端到端表示级融合基线（deep：z_ship / z_fin / gated fusion）

创新层的 flat-vs-representation 对照（研究计划 §5.2 / RQ2）。在**与扁平基线完全相同的协议**（`min_train=104`、`retrain_every=13`、目标 `r_{t+1}=log(P_{t+1}/P_t)`、指标在还原价格上算）下训练深度模态编码器并回测，与 flat M1 在相同测试周做 Clark-West 对照。

**模块**（`code/src/models/`）：

| 文件 | 作用 |
| ---- | ---- |
| `deep_dataset.py` | 把 17 节点图张量 + M1 金融序列对齐到扁平 `build_dataset` 的同一 target/idx，切 lookback=8 窗口；per-fold past-only 标准化接口 |
| `finance_encoder.py` | `z_fin`：M1 金融序列 → 小型因果 TCN → 32 维 |
| `fusion.py` | `GatedFusion`（softmax 门控）+ `DeepForecastModel`（mode = ship / fin / fusion）→ 回归头 → r_hat |
| `deep_rolling.py` | rolling-origin 深度训练循环（Adam + inner-val 早停 + 每 fold 目标标准化），输出与 `backtest.metrics` 兼容的 res 表 |
| `code/scripts/deep/run_deep_baseline.py` | 入口：跑 ship/fin/fusion + 读入 flat M1 预测 + evaluate + Clark-West |

> ⚠️ **工程注意**：`code/scripts/deep/run_deep_baseline.py` **不 import xgboost**（不重跑 flat M1，改读 `results/baselines/Flat/M1_Flat/baseline_predictions.csv`）——xgboost 与 torch 同进程在 macOS 因重复 OpenMP runtime **段错误**。需先 `run_baseline.py --modality M1` 生成该预测。

**结果**（253 共同测试周 2021-02 ~ 2025-12，`epochs=80` 早停，`seed=42`）：

| 模型 | RMSE | skill vs M0 | DirAcc |
| ---- | ---- | ----------- | ------ |
| M0（随机游走） | 4.1717 | 0.0% | – |
| M1_Ridge（扁平） | 4.2791 | −2.6% | 0.490 |
| M1_XGB（扁平） | 4.3833 | −5.1% | 0.553 |
| Mship（z_ship 图） | 4.1878 | −0.4% | 0.522 |
| Mfin（z_fin TCN） | 4.2045 | −0.8% | 0.506 |
| **Mfusion（门控融合）** | **4.1711** | **+0.02%** | **0.569** |

**嵌套 Clark-West（单边 p）**：

| 对照 | CW_stat | CW_p |
| ---- | ------- | ---- |
| 表示级融合 vs 扁平 M1（Mfusion vs M1_Ridge） | 3.36 | **0.0004** |
| 航运表示 vs 扁平 M1（Mship vs M1_Ridge） | 3.00 | **0.0014** |
| 航运增量 vs 纯金融深度（Mfusion vs Mfin） | 1.58 | 0.057 |

**解读（RQ2 核心证据）**：
- **表示级融合显著优于扁平特征融合**：Mfusion / Mship 相对 flat M1_Ridge 的 CW p = 0.0004 / 0.0014（极显著）；且深度模型不像扁平模型被高维稀释——flat M1 skill 为 −2.6%/−5.1%，而三个深度模型都贴近 M0，**Mfusion 是唯一 skill ≥ 0（+0.02%）且 DirAcc 最高（0.569）** 的模型。
- **航运表示的边际增量**：Mfusion vs Mfin 的 CW p = 0.057（边际显著），门控融合较纯金融深度略有航运增量。
- **仍未显著击败 M0**：所有模型 DM_p(vs M0) 均不显著（与本项目一贯的诚实结论一致：周频 Brent 随机游走极强）。
- ⚠️ **caveat**：Mfusion/Mship 相对 flat M1 的 CW 为**近似嵌套**（深度模型含相同金融输入 + 表示学习，非严格线性嵌套）；小样本（253 周）、未做深度超参 sweep、单 seed；z_rs 未接入。

**产物**：``results/baselines/Deep/_cross/`（`deep_metrics.csv` / `deep_cw.csv` / `deep_predictions.csv` / `deep_backtest.png`）+ 各 `M*_Deep/baseline_*.csv`。复现：`python3 code/scripts/deep/run_deep_baseline.py --modes ship,fin,fusion --lookback 8 --epochs 80`。