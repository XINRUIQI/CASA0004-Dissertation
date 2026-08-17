# 论文插图清单 / Figure index

所有图均为 PNG（300 dpi）+ PDF。图号按正文首次提及顺序编排。

第 3 章图号与生成命令见 `Old/figure_index.md`。下文只记当前第 4 章图。

## 第 4 章 — 结果

| 图号 | 文件 | 内容 | 建议位置 |
| --- | --- | --- | --- |
| 4.1 | `fig_4_1_flat_rmse_improvement` | 横向分组点图：S1–S4 × Ridge/XGBoost 的 RMSE improvement vs M0；x=0 为 M0。八点全在零线左侧 | 4.1 节，表 4.1 之后 |
| 4.2 | `fig_4_2_deep_rmse_improvement` | 横向哑铃图：Gated（主设定，实心圆）与 Cross-attention（空心菱形）；S1 仅 Gated；S3 两点越过 M0 | 4.2 节，表 4.2 之后 |
| 4.3 | `fig_4_1_flat_vs_deep_slope` | 匹配模态集上 Flat XGBoost → Deep gated 配对斜率；S3 高亮并穿过 M0；S1 为路径参照 | 4.3 节，表 4.3 之后 |
| 4.4 | `fig_4_3_shap_modality` | 仅航运：门控权重 ◇ 与绝对 SHAP 份额 ●（0–60%）；金融为互补项不画；两点不连线 | 4.4 节，表 4.4 之后 |
| 4.5 | `fig_4_4_node_shap_map` | 同裁切两面板（2022 \| 2024）。重点节点：中心小圆点 + 半透明比例光环（面积 ∝ 航运内 SHAP 份额，锁定 18%）；其余节点为固定大小浅灰圆点 | 4.4 节，节点段 |
| 4.6 | `fig_4_5_node_shap_heatmap` | 两面板（咽喉 / AOI），颜色 = 6 周向后滚动的航运内 \|SHAP\| 份额；事件窗为统一灰色 ±8 周带；色标 0–18% | 4.4 节，紧随图 4.5 |

## 附录 B

| 图号 | 文件 | 内容 | 建议位置 |
| --- | --- | --- | --- |
| B.1 | `fig_B_1_seed_robustness` | 各 Deep 设定 3 个种子点 + 均值，0 线即相对 M0；seed 42 为菱形 | 附录 B.2，对应表 4.5 |

价格/收益图为第 3 章图 3.2（`fig_3_2_price_returns`），由脚本键 `price` 生成；灰色带为评价样本，不含事件窗。

不画 S1–S4 skill 柱状图、门控变动点图、注意力条形。旧文件仍在盘上，但不入正文。

## 口径

- 图 4.1 / 4.2 的数值与表 4.1 / 4.2 的 Improvement vs M0 (%) 一致，横轴共用 −10% 至 +2%，x=0 即 M0。
- 图 4.3 的 Flat XGB RMSE 读各 `baseline_metrics*.csv`，与表 4.1 一致（S3 = 4.357），不用过期的 `subperiod_summary.csv`。
- 图 4.4 只画航运：◇ 门控权重、● 绝对 SHAP 份额；金融 = 100%−航运，不重复绘制。两点不连线，图注写明门控与 SHAP 不可互换。图 4.5–4.6 的量是节点绝对 SHAP 份额，不是门控、不是 GAT 注意力。
- 图 4.5 / 4.6 同色标（0–18%）、同 17 节点。地图：橙色 = 咽喉，蓝色 = AOI；光环面积 ∝ 航运内份额；只标注裕廊、霍尔木兹、苏伊士、好望角、曼德。裁切 20°W–150°E、44°S–55°N，休斯敦与巴拿马在框外。
- 热图分 (a) 咽喉与 (b) AOI 两面板，组内按完整样本航运内份额从高到低排列。色标为航运模态内部份额（0–18%，与图 4.5 锁定一致），不是全模型归因。红海窗内六周向后滚动均值的最大节点份额为 14.0%（苏伊士）。样本开头不足 6 周的单元格为缺失（浅灰）。

## 重新生成

```bash
python3 04_code/scripts/figures/make_chapter4_figures.py
python3 04_code/scripts/figures/make_chapter4_figures.py --only 4.1 4.2
python3 04_code/scripts/figures/make_chapter4_figures.py --only B.1  # 附录种子稳健性
python3 04_code/scripts/figures/make_result_figures.py          # 旧附录稳健性图
python3 04_code/scripts/figures/make_result_figures.py --legacy # 旧 skill / 注意力图
```
