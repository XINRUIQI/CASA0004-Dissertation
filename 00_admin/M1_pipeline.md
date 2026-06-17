# M1 Pipeline — 数据下载 → 变量构建 → EDA → 模型

> 记录与 M1（金融/市场/宏观基线层）相关的完整数据与建模流程、对应代码文件、输入/输出与复现命令。
> Last updated: 2026-06-17

---

## 0. 总览流程图

```
raw 原始数据 (EIA .xls / FRED .csv)
        │
        ▼
[1] build_weekly_time_index.py ───────────────► weekly_time_index.csv (基础 32 列)
        │  (末尾自动衔接 ↓ 一条龙)
        ├─ [1b] build_m1_to_build.py ─────────► m1_to_build_weekly.csv (8 个新 M1 变量)
        └─ [1c] merge_m1_to_build.py ─────────► weekly_time_index.csv (合并为 40 列)
        │
        ▼
[2] build_feature_matrix.py ──────────────────► weekly_features.csv (282 列, 含 RS/航运/文本/target)
        │                                        feature_groups.json (分组定义)
        ▼
[3] eda_beatrice_variables.py ────────────────► 01_literature/EDA/*.png / *.csv
        │
        ▼
[4] run_all.py → config.py + data_loader.py
        ├─ 01_baselines.py / 02_xgboost_model.py / 03_lstm_model.py / 04_tft_model.py / 05_stgnn_model.py
        └─ evaluation.py ─────────────────────► 01_literature/Test/results/*.csv + 对比热力图
```

---

## 1. 数据下载与变量构建

### 1A. 基础周频锚点（一条龙入口）
- **代码**：`03_data/processed/build_weekly_time_index.py`
- **输入**（raw）：
  - `03_data/raw/01_market_financial/1A Oil Price/` — Brent 日频现货
  - `1B Benchmark comparison/` — WTI 日频
  - `1C EIA Weekly Petroleum Status Report/` — 10 个 EIA 周报序列（库存/产量/进出口/炼厂/成品油）
  - `1D Macro-financial control variables/` — FRED/Yahoo（sp500, vix, dollar_index, treasury_10y, fed_funds_rate）
- **输出**：`03_data/processed/weekly_time_index.csv`（基础 32 列；合并后 40 列）
- **频率对齐**：全部对齐到 **周五截止周频 (W-FRI)**，研究期 **2006-01 ~ 2025-12**
- **运行**：
  - `python build_weekly_time_index.py` — 完整一条龙（含下载+合并 8 个新变量）
  - `python build_weekly_time_index.py --skip-m1-extra` — 只建基础锚点（不联网）

### 1B. 下载 8 个精读后新增的 M1 变量
- **代码**：`03_data/processed/build_m1_to_build.py`
- **输出**：`03_data/processed/m1_to_build_weekly.csv`（8 列，W-FRI）
- **数据源**（无需 API key；FRED 不稳时自动改用 yfinance/本地派生；含重试+增量缓存）：

| 变量 | 数据源 | 标识符 | 处理 |
| --- | --- | --- | --- |
| `ovx` | CBOE (Yahoo) | `^OVX`（备 FRED `OVXCLS`） | 周最后值 |
| `gpr` | Caldara–Iacoviello | `data_gpr_export.xls` 列 `GPR` | 月末 ffill + 1 周发布滞后 |
| `gold_return` | FRED LBMA | `GOLDPMGBD228NLBM`（备 `GC=F`） | 周最后值 → 对数收益率 |
| `global_econ_activity` | Dallas Fed | Kilian REA (igrea) | 月末 ffill + 5 周滞后 |
| `nonoil_industrial_commodity` | FRED IMF | `PINDUINDEXM` | 月末 ffill + 5 周滞后 |
| `futures_spread` | Yahoo + 本地 | `BZ=F` − 本地 `brent_price` | log(fut) − log(spot) |
| `commodity_fx` | Yahoo | `CADUSD=X`, `AUDUSD=X`（备 FRED） | 周变化均值 |
| `dgs10_change` | 本地派生 | 由 `treasury_10y` | 一阶差分 |

> 详细来源见 `03_data/external_sources.md`（M1 To-Build Variables 一节）。

### 1C. 合并进锚点
- **代码**：`03_data/processed/merge_m1_to_build.py`（幂等，可重复运行）
- **作用**：把 8 列按 `week_ending_friday` join 进 `weekly_time_index.csv` → 40 列
- 已被 `build_weekly_time_index.py` 末尾自动调用，单独补跑也可：
  `python build_m1_to_build.py && python merge_m1_to_build.py`

---

## 2. 特征矩阵装配（合并所有模态）
- **代码**：`04_code/scripts/build_feature_matrix.py`
- **输入**：`weekly_time_index.csv`(40) + `weekly_remote_sensing_features.csv` + `weekly_shipping_features.csv` + `weekly_text_features.csv`
- **输出**：
  - `03_data/processed/weekly_features.csv` / `.parquet`（**282 列**，模型/EDA 实际读取）
  - 新增 3 个 target：`target_brent_price_next_1w` / `target_brent_vol_next_1w` / `target_brent_direction_next_1w`
- **运行**：`python build_feature_matrix.py`
- ⚠️ **注意**：此脚本会**自动重写 `feature_groups.json`**（用旧命名 M2_add_text/M3_add_rs/M4_add_shipping，M1=全部 market 列）。当前手工整理的 `feature_groups.json`（M1=10 / M2_rs_clean=44 / M3_add_shipping=119）会被覆盖。**模型/EDA 不读该 json**（变量在 `config.py` 硬编码），故不影响运行，但需注意分组定义文件会被还原。

---

## 3. 当前 M1 变量定义（精读后机制化 10 个）

| # | 变量 | 经济机制 | 主要文献 |
| --- | --- | --- | --- |
| 1 | `brent_price` | 油价自身动态（lags/ma/mom 由 LAG_MA 生成） | P053/P001/P072 |
| 2 | `crude_stocks_change` | 供给 / 市场平衡 | P053【强】 |
| 3 | `global_econ_activity` | 全球需求（Kilian REA） | P053【强】/P052 |
| 4 | `nonoil_industrial_commodity` | 全球需求（工业原料） | P053【短期强】/P054 |
| 5 | `futures_spread` | 市场紧张 / 预期 | P053/P054 |
| 6 | `ovx` | 石油特定不确定性（优先于 VIX） | P052 |
| 7 | `gpr` | 预防性需求（地缘政治风险） | P076 |
| 8 | `dgs10_change` | 利率 / 持有成本（ΔDGS10） | P076 |
| 9 | `gold_return` | 商品联动 / 避险 | P004 |
| 10 | `commodity_fx` | 汇率渠道（CAD/AUD） | P053 |

> 定义位置：`01_literature/Test/config.py` 的 `M1_VARS`（建模权威来源）、
> `01_literature/EDA/eda_beatrice_variables.py` 的 `M1_VARS`（EDA）、
> `03_data/processed/feature_groups.json` 的 `M1_market_macro`（分组文档）。

---

## 4. EDA
- **代码**：`01_literature/EDA/eda_beatrice_variables.py`
- **输入**：`weekly_features.csv`
- **输出**（`01_literature/EDA/`）：摘要统计、缺失热图、时序图、相关矩阵（M1/M2/M3/M4）、与 target 相关性、分布、滚动相关、散点等共 13 图 + 2 csv
- **运行**：`python eda_beatrice_variables.py`

---

## 5. 模型实验
- **配置**：`01_literature/Test/config.py`
  - `M1_VARS`（10 个）、`LAG_MA_SPECS`（仅对 M1 相关基列生成 lag/ma/mom）、`LAYER_FEATURES`（M1–M5）、train/val/test=0.70/0.15/0.15、超参数
- **数据加载**：`01_literature/Test/data_loader.py`
  - `load_weekly()` 读 `weekly_features.csv`
  - `build_lag_ma_features()` 动态生成滞后/均线/动量
  - `prepare_tabular()`（树/线性，含 lag-ma）、`prepare_sequences()`（LSTM/TFT/ST-GNN）
  - 时间顺序切分；`StandardScaler` 仅在训练集 fit；按列 `dropna`
- **模型脚本**：
  - `01_baselines.py` — Ridge / RandomForest / SVM / LogisticRegression（含简单基准）
  - `02_xgboost_model.py` — XGBoost + SHAP
  - `03_lstm_model.py` / `04_tft_model.py` / `05_stgnn_model.py` — 深度/图模型
- **评估**：`01_literature/Test/evaluation.py`（RMSE/MAE/R²；accuracy/macro_f1/directional_acc 等）
- **总控**：`01_literature/Test/run_all.py` — 依次跑 5 个模型 → 聚合 `all_results_combined.csv` → 生成对比热力图（Models × Layers）
- **输出**：`01_literature/Test/results/*.csv` + `*.png`
- **运行**：
  - 冒烟测试：`python 01_baselines.py && python 02_xgboost_model.py`
  - 全量：`python run_all.py`

---

## 6. 一键复现（完整链路）

```bash
# 1) 数据：基础锚点 + 下载8新变量 + 合并 (一条龙)
cd "03_data/processed"
python build_weekly_time_index.py

# 2) 特征矩阵 (合并 RS/航运/文本 + targets)
cd "../../04_code/scripts"
python build_feature_matrix.py

# 3) EDA
cd "../../01_literature/EDA"
python eda_beatrice_variables.py

# 4) 模型
cd "../Test"
python run_all.py
```

---

## 7. 已知 caveat / 待办

1. **缺随机游走/Naive 基准** → price 的高 R²（~0.83）受价格持续性影响，不能直接解读为预测能力（P053/P001 警告）。**待补**：RW 基准 + 相对 RW 的 OOS-R² + DM 检验。
2. **样本窗口**：`ovx`/`futures_spread` 自 2007 起，M1 有效样本 **2007-08 ~ 2025**（957 周）；各层 `dropna` 独立 → M1/M2/M3/M4 测试窗口不完全一致，跨层对比需注明。
3. **`feature_groups.json` 会被 `build_feature_matrix.py` 覆盖**（还原为旧命名/全列 M1）；如需保留手工整理版，建议另存或改造脚本写到独立文件。
4. **`config.py` 仍含 M5（GDELT 文本）**：文本模态已按 Meeting 02 移除，M5 仅作附录/robustness。

---

## 8. M1 相关代码文件清单

| 阶段 | 文件 |
| --- | --- |
| 数据-基础锚点 | `03_data/processed/build_weekly_time_index.py` |
| 数据-新变量下载 | `03_data/processed/build_m1_to_build.py` |
| 数据-合并 | `03_data/processed/merge_m1_to_build.py` |
| 数据-特征矩阵 | `04_code/scripts/build_feature_matrix.py` |
| 数据来源记录 | `03_data/external_sources.md` |
| 分组定义 | `03_data/processed/feature_groups.json` |
| EDA | `01_literature/EDA/eda_beatrice_variables.py` |
| 模型-配置 | `01_literature/Test/config.py` |
| 模型-数据加载 | `01_literature/Test/data_loader.py` |
| 模型-评估 | `01_literature/Test/evaluation.py` |
| 模型-脚本 | `01_literature/Test/0{1..5}_*.py` |
| 模型-总控 | `01_literature/Test/run_all.py` |

| 阶段 | 主要产物 |
| --- | --- |
| 锚点 | `03_data/processed/weekly_time_index.csv` (40 列) |
| 新变量 | `03_data/processed/m1_to_build_weekly.csv` (8 列) |
| 特征矩阵 | `03_data/processed/weekly_features.csv` (282 列) |
| EDA | `01_literature/EDA/*.png`, `*.csv` |
| 模型结果 | `01_literature/Test/results/all_results_combined.csv` + 热力图 |
