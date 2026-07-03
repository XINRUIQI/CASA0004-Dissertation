# M2 遥感（Channel A）Sentinel-2 影像 patch 与 embedding — 数据字典

> 原始 patch：`03_data/raw/02_sentinel2/Channel A/s2_patches/`（GeoTIFF，**不入 git**）
> Manifest：`03_data/raw/02_sentinel2/Channel A/s2_patches/S2_patches_manifest_ALL.csv`
> 审计 / 索引 / 嵌入产物：`03_data/processed/M2/outputs/s2_patch_*.csv`、`s2_prithvi_emb_*`
> GEE 导出脚本：`03_data/raw/02_sentinel2/Channel A/export_s2_patches_multimodal_gee{,_bundled}.js`
> 本地脚本：`03_data/processed/M2/py/audit_s2_patches.py`、`s2_patch_utils.py`、`precompute_s2_embeddings.py`
> 配套 Channel B 机制表：`03_data/processed/M2/m2_ChanelB_data_dictionary.md`
> 方案文档：`00_admin/待整理/2026-06-22_research_plan_e2e_multimodal.md` §4.2 / §5.1
> 最后更新：2026-07-03

---

## 0. 定位：M2 = 双通道遥感中的 Channel A（影像表示）

M2 遥感模态采用**双通道设计**：

- **Channel A（本字典）**：11 AOI × 月度 Sentinel-2 **6 波段 GeoTIFF patch** → 冻结 **Prithvi-EO-2.0-300M** → 固定长度 **image embedding**（1024 维）。服务于 **RQ2**「手工指标 vs 表示学习」及贡献层 **模态感知融合** 的 RS 编码器输入。
- **Channel B（机制变量）**：VIIRS 夜光 + S2 光学**表格指数**（`m2_weekly_features.csv`）。服务于 **RQ1** 扁平基线与机制解释。

> ⚠️ **Channel A patch ≠ Channel B AOI**：Channel B 月度指数用 **5 km 圆缓冲** 内像元均值；Channel A 用**以站点为中心的方形 patch**（half-size 因类型/单站而异，见 §4）。二者空间范围不同，不可混为一谈。

> ⚠️ **VIIRS 无影像 patch**：夜光仅 Channel B 表格（`NTL_anom` 等）；不进入 Prithvi encoder（见 Channel B 字典 §0）。

---

## 1. 基本信息

| 项 | 值 |
| --- | --- |
| 传感器 | Copernicus **Sentinel-2 SR Harmonized**（`COPERNICUS/S2_SR_HARMONIZED`） |
| 时间粒度 | **月度**最优合成（月内 SCL 云掩膜后的 **median composite**） |
| 导出窗口 | **2019-01 ~ 2026-06**（与 M0–M4 标准比较窗对齐） |
| 空间分辨率 | **10 m**（导出 `SCALE = 10`） |
| 坐标系 | **逐站 UTM**（由经纬度自动选带，`EPSG:326xx` / `327xx`） |
| 本地 patch 数 | **968** 个 `.tif` 文件（manifest **990** 行，含空月记录） |
| 可用 patch | **963** 个（`valid_mask=1`；已嵌入 Prithvi） |
| 磁盘体积 | 约 **2.0 GB**（`s2_patches/`） |
| EO 模型 | **Prithvi-EO-2.0-300M**（冻结，只做特征提取） |
| Embedding 维数 | **1024**（mean-pool 主用；cls token 备存） |

**963 可用 patch 的 per-site 分布**（`s2_patch_coverage_report.csv`，2026-07-03）：

| `site_id` | 站点 | 类型 | 期望月数 | 可用 | 覆盖率 |
| --- | --- | --- | ---: | ---: | ---: |
| P005 | Houston | port | 90 | 90 | 1.000 |
| P006 | NingboZhoushan | port | 86 | 86 | 1.000 |
| P001 | Rotterdam | port | 83 | 83 | 1.000 |
| P007 | Jamnagar | refinery | 86 | 86 | 1.000 |
| P004 | Jurong | refinery | 84 | 84 | 1.000 |
| P009 | Ulsan | refinery | 87 | 85 | 0.977 |
| P008 | Basra | terminal | 90 | 89 | 0.989 |
| P002 | Fujairah | terminal | 90 | 90 | 1.000 |
| P010 | Kharg | terminal | 90 | 90 | 1.000 |
| P003 | Ras Tanura | terminal | 90 | 90 | 1.000 |
| P011 | Yanbu | terminal | 90 | 90 | 1.000 |

---

## 2. 构建链路概览

```
GEE 导出 (Channel A/*.js)
  └─► 03_data/raw/.../s2_patches/*.tif + S2_patches_manifest_ALL.csv
        └─► audit_s2_patches.py
              └─► s2_patch_index.csv / s2_patch_coverage_report.csv / heatmap
                    └─► precompute_s2_embeddings.py (frozen Prithvi)
                          └─► s2_prithvi_emb_meanpool.npy [963, 1024]
                              s2_prithvi_emb_cls.npy       [963, 1024]
                              s2_prithvi_emb_index.csv
                              s2_prithvi_emb_coverage.csv
        └─► (待建) 月→周 as-of 对齐 + temporal/site attention → z_rs (32 维)
```

| 阶段 | 脚本 | 输入 | 输出 |
| --- | --- | --- | --- |
| **RAW 导出** | `export_s2_patches_multimodal_gee_bundled.js` | GEE S2 SR + SCL | GeoTIFF + manifest |
| **审计** | `audit_s2_patches.py` | manifest + 本地 `.tif` + exclusions | `s2_patch_index.csv` 等 |
| **嵌入** | `precompute_s2_embeddings.py` | `valid_mask=1` 的 patch | `.npy` + `s2_prithvi_emb_index.csv` |

---

## 3. GEE 导出规则（RAW 层）

### 3.1 波段与云处理

| 项 | 规则 |
| --- | --- |
| **导出波段（6）** | `B2, B3, B4, B8A, B11, B12`（蓝/绿/红/窄 NIR/SWIR1/SWIR2） |
| **对齐 Prithvi HLS** | 与 Prithvi-EO-2.0 预训练六波段**物理一致、顺序一致**，无需 remap |
| **SCL** | 用于**云掩膜**（shadow / medium-high cloud / cirrus / snow），**不写入 GeoTIFF** |
| **合成** | 月内有效景 **median**；无有效景则**跳过导出**（不 queue 空任务） |
| **场景预滤** | 整景 `CLOUDY_PIXEL_PERCENTAGE ≤ 60`（`CLOUD_MAX`） |
| **空值约定** | 掩膜边缘 / 无数据像元在 GeoTIFF 中为 **0**；嵌入时映射为 Prithvi `1e-4` |

### 3.2 文件命名

```
S2_{site_id}_{short_name}_{YYYY_MM}.tif
```

示例：`S2_P001_Rotterdam_2019_01.tif`

### 3.3 Manifest 字段（`S2_patches_manifest_ALL.csv`）

| 字段 | 含义 |
| --- | --- |
| `site_id` / `site_name` / `site_type` | 站点标识 |
| `month` / `year` | `YYYY_MM` / 四位年 |
| `patch_half_m` | 半宽（米） |
| `patch_px` | 导出像元边长（@10 m，`≈ 2 × patch_half_m / 10`） |
| `n_scenes` | 该月参与合成的有效 S2 景数 |
| `mean_cloud` / `min_cloud` | 合成景云量统计（%） |
| `crs` | 导出 UTM EPSG |
| `exported` | 是否成功 queue 导出（0/1） |

---

## 4. 各站点 patch 尺寸（最终导出规格）

**默认按类型**（`PATCH_HALF_BY_TYPE`）：port **3200 m** / refinery **2560 m** / terminal **1280 m**。

**单站覆盖微调**（目视校核后写入 GEE bundled AOI 的 `patch_half_m`）：

| 类型 | 站点 | half-size | 全宽 patch | 像元 (@10 m) | 备注 |
| --- | --- | ---: | ---: | ---: | --- |
| port | Rotterdam, Houston, NingboZhoushan | 3200 m | 6.4 km | **640** | 默认 port |
| refinery | Jurong, Jamnagar, Ulsan | 2560 m | 5.12 km | **512** | 默认 refinery |
| terminal | Ras Tanura | 1280 m | 2.56 km | **256** | 默认 terminal |
| terminal | Fujairah, Kharg, Yanbu | **1600 m** | **3.2 km** | **320** | 终端放大（离岸/岛式设施） |
| terminal | Basra | **800 m** | **1.6 km** | **160** | 终端缩小（聚焦装船点） |

> **Resize 策略**：Prithvi 输入固定 **224×224**；各站 patch 双线性 resize（非 tiling），使 port 约 ~29 m/px、terminal 约 ~14 m/px，接近 Prithvi HLS ~30 m 训练 GSD（见 `precompute_s2_embeddings.py` 注释）。

> Channel B 的 **5 km 圆缓冲**仅用于月度 NDVI/NTL **表格**聚合，**不**等于上表 patch 范围。

---

## 5. 有效性审计（`audit_s2_patches.py`）

### 5.1 `valid_mask` 判定

一条 (site, month) 记为可用（`valid_mask=1`）当且仅当：

1. manifest 中有记录且本地 `.tif` **存在**；
2. **不在** `s2_patch_exclusions.csv` 手工排除表；
3. GeoTIFF 为 **6 波段**且 **≥0.5%** 像元非零（`MIN_NONZERO_FRAC=0.005`）。

### 5.2 手工排除（`s2_patch_exclusions.csv`）

| site | month | 原因 |
| --- | --- | --- |
| P008 Basra | 2019_01 | SCL 掩膜后全零（`n_scenes=1`, `mean_cloud≈59.7`） |
| P009 Ulsan | 2022_02 | 掩膜后极稀疏（0.23% 非零像元） |
| P009 Ulsan | 2022_09 | 掩膜后极稀疏（0.03% 非零像元） |

### 5.3 审计产物

| 文件 | 行/形状 | 用途 |
| --- | --- | --- |
| `s2_patch_index.csv` | 966 行 | 每个 (site, month) 的 `file_exists` / `excluded` / `pixel_valid` / `valid_mask` / 云量 / `patch_px` |
| `s2_patch_coverage_report.csv` | 11 行 | 逐站期望月数、可用数、缺失/排除/空文件计数 |
| `s2_patch_validity_heatmap.png` | — | 站点 × 月 可用性热图 |

**运行**：

```bash
python3 03_data/processed/M2/py/audit_s2_patches.py
```

---

## 6. Prithvi 嵌入（`precompute_s2_embeddings.py`）

### 6.1 处理流程（每张可用 patch）

1. 读取 6 波段 GeoTIFF → float32；
2. Prithvi `config.json` 的 **mean/std** 标准化；像元值 **0 → 1e-4**；
3. **双线性 resize** 至 224×224；
4. `forward_features`（**T=1**，单帧）→ 最后一层 `[B, 197, 1024]`；
5. **mean-pool**（196 个 spatial tokens）→ 主 embedding；**cls token** 另存备用；
6. 模型 **eval + requires_grad=False + no_grad**，权重来自 HuggingFace `ibm-nasa-geospatial/Prithvi-EO-2.0-300M`。

### 6.2 产物

| 文件 | 形状 / 规模 | 说明 |
| --- | --- | --- |
| `s2_prithvi_emb_meanpool.npy` | **[963, 1024]** float32 | **主 embedding**（RQ2 RS 编码器默认输入） |
| `s2_prithvi_emb_cls.npy` | **[963, 1024]** float32 | cls token 备选池化 |
| `s2_prithvi_emb_index.csv` | 963 行 | 与 `.npy` 行序 **1:1 对齐**（`emb_row`） |
| `s2_prithvi_emb_coverage.csv` | 11 行 | 每站嵌入月数与首尾月 |
| `s2_prithvi_emb_check.png` | — | 嵌入数值 sanity 图 |

### 6.3 `s2_prithvi_emb_index.csv` 主要字段

| 字段 | 含义 |
| --- | --- |
| `emb_row` | 在 `.npy` 中的行号（0 … N−1） |
| `site_id` / `site_name` / `site_type` | 站点 |
| `month` / `year` / `obs_month_start` | 观测月 |
| `mean_cloud` / `n_scenes` / `patch_px` | 来自 manifest 的质量/尺寸元数据 |
| `filename` | 源 GeoTIFF 文件名 |
| `sensor` | 固定 `S2` |
| `emb_model` | `Prithvi-EO-2.0-300M` |
| `emb_dim` | 1024 |
| `emb_ok` | 嵌入是否成功（1/0） |

**运行**：

```bash
# 冒烟（4 张）
python3 03_data/processed/M2/py/precompute_s2_embeddings.py --limit 4
# 全量（963 张）
python3 03_data/processed/M2/py/precompute_s2_embeddings.py
```

---

## 7. 防泄漏与周频对齐（待建 Part B）

Channel A 的**周频入模表尚未构建**；设计原则与 Channel B 一致（见 `m2_ChanelB_data_dictionary.md` §3）：

| 环节 | 规则（计划） |
| --- | --- |
| 保守可得日 | `availability_date = 月末 + 15 天`（`PUB_LAG_DAYS=15`） |
| 月 → 周 | `merge_asof(direction="backward")` 到每个 `week_fri` |
| 缺失 | **不做 embedding 常数值 ffill**；携带 `days_since_obs` + `valid_mask` + `modality_mask` |
| 训练 | embedding 序列仅作输入；**不对 Prithvi  backbone 微调**（小样本风险） |

下游贡献层计划：同 AOI 多月 **temporal attention** + 11 站 **site attention** → **`z_rs`（32 维）**，再与 `z_fin` / `z_ship` 门控融合。

---

## 8. 与 Channel B / 扁平基线的关系

| 对比项 | Channel A（本字典） | Channel B |
| --- | --- | --- |
| 数据形态 | GeoTIFF patch → 1024-d embedding | 周频表格 `{idx}_anom_{aoi}` |
| 空间定义 | 方形 patch（§4） | 5 km 圆缓冲 AOI 均值 |
| 时间窗 | 2019-01 ~ 2026-06 | 2019-01 ~ 2025-12（标准比较窗） |
| 入模阶段 | RQ2 表示学习 / 门控融合（待建） | RQ1 扁平 M2（**55 anom**，已跑 baseline） |
| VIIRS | **不含** | **含** `NTL_anom` |

**RQ2 核心对照**：M2-flat（Channel B 手工 anom）vs M2-embedding（Channel A Prithvi `z_rs`）vs M4 表示级融合。

---

## 9. 备注与已知缺口

- **RAW patch 不入 git**：`03_data/raw/` gitignore；换机需从 GEE Drive 文件夹 `CASA0004_S2_patches` 重新同步至 `Channel A/s2_patches/`。
- **Rotterdam / Ningbo 期望月数 &lt; 90**：部分月份 GEE 无有效 S2 景，manifest 无导出或本地缺失；覆盖率仍为该站**已期望月**内的比例。
- **SatMAE**：方案中列为 Prithvi **备选** EO encoder；当前仅实现 Prithvi-EO-2.0-300M。
- **周频 RS embedding 矩阵**：Part B（`build_m2_embeddings_weekly.py` 或等价脚本）**待实现**——将 `s2_prithvi_emb_*.npy` + index as-of 对齐到 W-FRI，供贡献层训练。
- **文献**：EO 基础模型方法依据 P094（Prithvi-EO-2.0）、P095（SatMAE）；**无**「image embedding → Brent 周频价格」的直接先例，增量须本项目消融证明。

---

## 10. 相关路径速查

| 路径 | 内容 |
| --- | --- |
| `03_data/raw/02_sentinel2/Channel A/s2_patches/` | GeoTIFF（968 文件，~2 GB） |
| `03_data/raw/02_sentinel2/Channel A/s2_patch_exclusions.csv` | 手工无效 patch 清单 |
| `03_data/raw/02_sentinel2/aoi_oil_infrastructure.csv` | 11 站坐标与类型（Channel B 5 km 缓冲亦用此表） |
| `03_data/processed/M2/outputs/s2_patch_index.csv` | 全量 (site, month) 有效性索引 |
| `03_data/processed/M2/outputs/s2_prithvi_emb_meanpool.npy` | 主 embedding 数组 |
| `03_data/processed/M2/m2_ChanelB_data_dictionary.md` | Channel B 机制特征字典 |
