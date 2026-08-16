# 论文插图清单 / Figure index

所有图均为 PNG（300 dpi）+ PDF。图号按正文首次提及顺序编排。

第 3 章图号与生成命令见 `Old/figure_index.md`。下文只记当前第 4 章图。

## 第 4 章 — 结果

| 图号 | 文件 | 内容 | 建议位置 |
| --- | --- | --- | --- |
| 4.1 | `fig_4_1_price_returns` | 上：Brent 价格；下：周对数收益。灰底为评价期（2021-01 起），四色带为表 4.4 的 ±8 周事件窗 | 4.1 节 |
| 4.2 | `fig_4_2_flat_vs_deep_slope` | Flat XGBoost → Deep gated 配对斜率，S3 高亮，M0 水平线 | 4.4 节，表 4.3 之后 |
| 4.3 | `fig_4_3_shap_modality` | 全样本、2021–2025、四事件窗的绝对 SHAP 份额（金融 vs 航运）；浅色点为门控权重 | 4.5 节，表 4.4 之后 |
| 4.4 | `fig_4_4_node_shap_map` | 同裁切、同面积比例的两面板（全样本 \| 2024）。面积 ∝ 航运内 SHAP 份额，锁定 18% | 4.5 节，节点段 |
| 4.5 | `fig_4_5_node_shap_heatmap` | 17 节点 × 257 周，颜色 = 6 周滚动航运内 \|SHAP\| 份额；事件窗为竖线 | 4.5 节，紧随图 4.4 |
| 4.6 | `fig_4_6_seed_robustness` | 各 Deep 设定 3 个种子点 + 均值，0 线即相对 M0；seed 42 为菱形 | 4.6 节，表 4.5 之后 |

不画 S1–S4 skill 柱状图、门控变动点图、注意力条形。旧文件仍在盘上，但不入正文。

## 口径

- 图 4.2 的 Flat XGB RMSE 读各 `baseline_metrics*.csv`，与表 4.1 一致（S3 = 4.357），不用过期的 `subperiod_summary.csv`。
- 图 4.3–4.5 的量是节点/模态绝对 SHAP 份额，不是门控、不是 GAT 注意力。门控只在图 4.3 用浅色点标出，图注写明二者不可直接比。
- 图 4.4 / 4.5 同色标（0–18%）、同 17 节点。地图只标注裕廊、霍尔木兹、苏伊士、好望角、曼德；裁切 20°W–150°E、40°S–55°N，休斯敦与巴拿马在框外。
- 热图纵轴咽喉在上、AOI 在下，含裕廊。红海窗内无节点超过航运归因的 12%。

## 重新生成

```bash
python3 04_code/scripts/figures/make_chapter4_figures.py
python3 04_code/scripts/figures/make_chapter4_figures.py --only 4.3 4.5
python3 04_code/scripts/figures/make_result_figures.py          # 附录 B.1、B.2
python3 04_code/scripts/figures/make_result_figures.py --legacy # 旧 skill / 注意力图
```
