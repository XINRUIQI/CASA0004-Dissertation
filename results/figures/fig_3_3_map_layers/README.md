# Figure 3.3 — 制图图层（QGIS）

正文第 3.3 节世界地图的输入文件。预测对象是全球 Brent，没有单一研究区；空间范围由这 11 个 AOI 与 6 个咽喉成立。图上不画固定走廊边（`corridor_edges.geojson` 仅作附录/图结构备份，不要加载进此图）。

已有成品：`../fig_3_3_study_sites_map.png` / `.pdf`（`make_method_figures.py`）。本文件夹用于在 QGIS 中重绘或微调。

## 图层（全部 EPSG:4326）

| 文件 | 几何 | 内容 |
| --- | --- | --- |
| `aoi_points.geojson` | 点 11 | 港口 ○ / 炼厂 △ / 码头 □；标注字段 `map_label` |
| `chokepoint_points.geojson` | 点 6 | 咽喉代表点（菱形） |
| `corridor_edges.geojson` | 线 13 | 附录 A.4.2 固定 AOI–咽喉走廊边（单位权重、无向） |
| `persian_gulf_inset.geojson` | 面 1 | 波斯湾放大框 `46.5–60°E, 22.5–32°N` |
| `map_extent.geojson` | 面 1 | 主图图框 `135°W–152°E, 48°S–68°N` |
| `nodes_17.geojson` | 点 17 | AOI + 咽喉合并，按 `node_class` 分类 |
| `ne_110m_land.geojson` | 面 | Natural Earth 110m 陆地 |
| `ne_110m_admin_0_boundary_lines_land.geojson` | 线 | 国界（可选） |

同名 `.csv` 可当属性表；同名 `.qml` 按现图配色（AOI `#2E5A88`，咽喉 `#D1622B`，走廊边 `#8FA8C8`）。

同目录 CSV 也可用「图层 → 添加分隔文本图层」，X=`lon`、Y=`lat`。

## QGIS 建议步骤

1. 新建工程，CRS 先用 `EPSG:4326`；主图可再投到 Equal Earth（`EPSG:8857`）或 Robinson。
2. 自下而上加载：陆地 → 国界 → 走廊边 → inset 框 → AOI → 咽喉。
3. 右键各图层 → 样式 → 加载，选对应 `.qml`。标注字段：AOI / 咽喉用 `map_label`。
4. 主图：用 `map_extent` 设地图范围。波斯湾 inset：Print Layout 里再放一个地图框，范围用 `persian_gulf_inset`，并打开主图的「概览」指向该框。
5. 湾内拥挤点（Al Basrah Terminal / Kharg / Ras Tanura / Fujairah / Hormuz）主图可不标名，只在 inset 标注。`in_gulf_inset=1` 可做筛选。
6. 导出 PDF（矢量）+ 300 dpi PNG。

走廊边是图论示意直线，不是实测航迹。动态 AOI–AOI 航次边不进此图。

## 坐标来源

- AOI：`data/raw/02_sentinel2/aoi_oil_infrastructure.csv`（附录 A.2.1）
- 咽喉：EIA World Oil Transit Chokepoints 代表过境点，与 `make_method_figures.py` 一致
- 边：附录 A.4.2 / `CHOKE_AOI` in `build_m3_graph17.py`

重建：

```bash
python code/scripts/figures/export_fig33_map_layers.py
```
