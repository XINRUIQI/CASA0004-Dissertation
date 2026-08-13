# 论文插图清单 / Figure index

所有图均为 PNG（300 dpi，供预览与 Word）+ PDF（矢量，供 LaTeX / 排版）。
图号按正文首次提及顺序编排。正文尚未插入，此处仅记录建议位置。

## 第 3 章 — 方法

| 图号 | 文件 | 内容 | 建议位置 |
| --- | --- | --- | --- |
| 3.1 | `fig_3_1_research_design` | 研究设计流程图：M0 基准、S1→S2/S3/S4、Flat 与 Deep 配对、共享评估、RQ1–RQ3 | 3.1 节 |
| 3.2 | `fig_3_2_expanding_window` | 周历上的估计/评估划分：104 周初始估计、20 次重估、257 个评估起点 | 3.6.1 节 |
| 3.3 | `fig_3_3_study_sites_map` | 11 个 AOI + 6 个咽喉世界分布图，含 13 条静态 AOI–咽喉边与波斯湾放大图 | 3.3 节 |
| 3.4 | `fig_3_4_forecast_origin` | 一次重估起点：训练折 + 嵌套内部验证周、四周输入窗、提前一周目标；区块内沿用参数 | 3.6.1 节 |

行号对应 `06_writing/Chapter 3 Methodology/20260811_chapter_3_methodology_bilingual.md`。
图 3.2 与图 3.4 原为旧图 3.3 的上下两个面板，现拆为两图：日历视图放 3.6.1 节，
单次重估结构同节。整合稿 `06_writing/Whole/20260811_dissertation_draft.md` 尚未同步。

图 3.1 另有 Mermaid 版本 `fig_3_1_research_design.mmd`（可在 Typora / Obsidian / GitHub 直接渲染，
便于改字），`fig_3_1_research_design_mermaid_preview.png` 为其渲染预览。两版内容一致。

## 第 4 章 — 结果

| 图号 | 文件 | 内容 | 建议位置 |
| --- | --- | --- | --- |
| 4.1 | `fig_4_1_skill_bars` | M1–M4 × Ridge / XGBoost / gated / cross-attn 的 RMSE skill 分组条形，0 线即 M0 | 4.1 节末（第 15 行后），作全章总览 |
| 4.2 | `fig_4_2_flat_vs_deep_slope` | Flat XGBoost → Deep gated 配对斜率，M3 高亮 | 4.4 节表 4.3 之后（第 100 行） |
| 4.3 | `fig_4_3_incremental_tests` | RQ1（15 项）与 RQ2（14 项）两族的 DM–HLN 原始 p 与 Holm 调整后 p 配对哑铃图 | 4.5 节第 118 行后 |
| 4.4 | `fig_4_4_subperiod_skill` | full / early(<2023) / late(≥2023) 三期 skill，Flat 与 Deep 双面板 | 4.5 节末（第 126 行后） |
| 4.5 | `fig_4_5_event_gate_shifts` | 4 个事件窗 × 3 种子的航运门控变化 dot plot | 4.6 节，替换第 142–146 行占位符 |
| 4.6 | `fig_4_6_node_attention_stability` | 左：gated S3 航运节点注意力均值 ±1 SD，标注 top-5 跨种子命中次数；右：三个机制的注意力份额除以均匀份额，显示 gated 有选择性而交叉注意力接近均匀 | 4.6 节，紧随图 4.5 |

行号对应 `06_writing/Chapter 4 Result/20260811_chapter_4_results_bilingual.md`。

## 附录 B

| 图号 | 文件 | 内容 |
| --- | --- | --- |
| B.1 | `fig_B_1_gate_paths_seeds` | 三模态门控周度轨迹与跨种子带，说明周度路径不稳 |
| B.2 | `fig_B_2_rs_site_attention` | gated S4 的 11 个遥感站点注意力均值 ±1 SD。S4 未跑过 M0，按 RQ3 准入规则不入正文，仅作稳定性诊断 |

## RQ3 准入规则对图的影响

RQ3 只解释相对 M0 有正 RMSE skill 的 Deep 单元，符合条件的是三个：
xattn S3（+1.002%）、xattn S4（+0.194%）、gated S3（+0.149%）。
gated S4 为 −0.681%，因此其模态权重与遥感站点注意力只能进附录（图 B.2），
不能作为 RQ3 正文结论。图 4.5、4.6 与 B.1 的门控内容均取自 S3。

## 动图（演示用，不入论文）

`anim_expanding_window.gif` 是图 3.2 的动画版：21 帧、约 18 秒、循环播放，
背景为 Brent 周度价格曲线，训练窗随重拟合逐次扩展，13 周测试块前移，
已计分周数从 0/257 累加至 257/257。适合答辩或 mini-conference 幻灯片。

`anim_expanding_window.html` 为同一动画的网页播放器（带播放/暂停/逐帧控制），
可直接用浏览器打开。`anim_frame_warmup/refit01/final.png` 是三个关键帧静图，
若需在纸质材料上表现动态过程，可并排放这三张。

生成：`python3 04_code/scripts/figures/make_window_animation.py [--fps 1.4]`

## 数据来源

图 3.3 的 AOI 坐标取自附录 A.2.1；咽喉坐标为 EIA World Oil Transit Chokepoints 的
代表坐标，硬编码于绘图脚本顶部（附录 A 目前未列咽喉坐标表，如后续补表需与脚本对齐）。

图 3.2 与图 3.4 的切分参数取自 `run_baseline.py` 与 `run_deep_baseline.py` 的 argparse 默认值
（lookback=4、min_train=104、retrain_every=13、val_weeks=52），测试周边界直接读自
`baseline_predictions.csv`。

第 4 章各图的数值来源：Flat 各设定取 `subperiod_summary.csv` 的 `period=full` 行，
gated / cross-attn 取 `deep_fusion_matrix.csv`，M0 取 `deep_metrics.csv`。已逐个核对，
与表 4.1–4.3 一致。

图 4.3 的全部 p 值改由统一检验表 `05_outputs/tests/test_table_main.csv` 提供
（生成脚本 `04_code/scripts/tools/build_test_tables.py`），不再从各 CSV 分别取值，
以免正文、表、图三处标准不一致。图中每项均为重构价格平方误差上的 DM–HLN 检验，
Holm 在族内调整（RQ1 为 15 项，RQ2 为 14 项）。
图 4.6 右panel 与图 B.2 的注意力份额取自 `deep_m3_gate_stability.csv`、
`deep_gate_stability.csv`（3 个种子均值）与 `deep_m3_xattn_weekly.csv`、
`deep_xattn_weekly.csv`（种子 42 单次运行的 257 周均值）。

## 两处待办

**一、图 4.1 比表 4.2 多一个数。** 表 4.2 中 M2 的 cross-attention 记为“—”，但
`deep_fusion_matrix.csv` 有该结果（RMSE 4.396，skill −5.89%），图中已按数据画出。
需在“补进表 4.2”与“从图中移除”之间取舍。

**二、图 4.5、4.6、B.1 的 Deep M3 / M4 口径已对齐（2026-08-12 完成）。** 已补跑

```bash
python3 04_code/scripts/deep/run_deep_interpret_m3.py --seeds 42,1,2
```

在 `M3_Deep/` 下生成 `deep_m3_gate_events.csv`、`deep_m3_gate_stability.csv`、
`deep_m3_gate_band_weekly.csv` 与三个 seed 的 `deep_m3_gate_weekly_seed*.csv`。
`make_result_figures.py` 已相应改为：图 4.5、4.6 与 B.1 的门控内容全部读 M3。
遥感站点注意力原为图 4.6 右panel，2026-08-13 按 RQ3 准入规则移入附录图 B.2，
腾出的右panel 改画三个机制的注意力选择性对比。
图 B.1 按 band 文件中实际存在的模态自动决定面板数（S3 为 2 个）。
图 4.3、4.5、4.6 的标题不再硬编码结论，改为由数据推导，避免结果变化后标题与数字不符。

## 重新生成

```bash
python3 04_code/scripts/figures/make_design_figure.py    # 图 3.1
python3 04_code/scripts/figures/make_method_figures.py   # 图 3.2、3.3、3.4
python3 04_code/scripts/figures/make_result_figures.py   # 图 4.1–4.6、B.1、B.2
```

`make_result_figures.py` 与 `make_method_figures.py` 支持 `--only` 单独重画，
参数即图号，例如 `--only 4.1 4.3`。图 3.3 需要 `geopandas` 与
`03_data/raw/00_spatial_anchors/naturalearth/` 下的 Natural Earth 110m 矢量。
