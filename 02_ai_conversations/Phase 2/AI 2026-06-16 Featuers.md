## 待构建变量清单及数据源/下载方式


| 变量                                     | 机制          | 文献                         | 数据源                           | 下载方式                                                                                             |
| -------------------------------------- | ----------- | -------------------------- | ----------------------------- | ------------------------------------------------------------------------------------------------ |
| `gpr` 地缘政治风险                           | 预防性需求       | P076【最重要外部】、P072           | Caldara–Iacoviello GPR        | 从 `matteoiacoviello.com/gpr.htm` 下载 `gpr_web.xls`。月频数据，包含全球及国别 GPR 指数                            |
| `ovx` 原油隐含波动率                          | 石油特定不确定性    | P052（OVX > VIX）            | CBOE Crude Oil VIX            | 使用 FRED 代码 `OVXCLS` 下载日频数据，再聚合为周均值                                                               |
| `gold_return` 黄金收益率                    | 商品联动与避险渠道   | P004                       | LBMA / FRED                   | 使用 FRED 代码 `GOLDPMGBD228NLBM` 下载日频黄金价格，再计算对数收益率；也可使用 `yfinance` 下载 `GC=F`                        |
| `global_econ_activity` 全球经济活动          | 全球石油需求      | P053【强】、P052、P004、P072     | Kilian Index / 全球工业生产指数 / PMI | 优先使用 Dallas Fed 发布的 Kilian 全球经济活动指数（月频）；也可使用 FRED 的 OECD CLI，代码为 `OECDLOLITOAASTSAM`             |
| `nonoil_industrial_commodity` 非油工业商品价格 | 全球需求代理      | P053【短期强】、P054             | CRB 工业品指数 / 金属指数              | 可使用 FRED 的 IMF 工业原料价格指数 `PINDUINDEXM`，或全部商品指数 `PALLMINDEXM`；金属数据也可来自 LME 或 World Bank Pink Sheet |
| `futures_spread` 期货－现货价差               | 市场紧张程度与价格预期 | P053、P054、P072             | ICE Brent 期货曲线                | 使用 EIA 的 Brent 现货价格 `RBRTE`，结合 ICE 或 EIA 的 Brent 1–6 月期货价格；也可使用 `yfinance` 的 `BZ=F` 构建近月与远月价差    |
| `commodity_fx` 商品货币汇率                  | 汇率传导渠道      | P053（优于宽口径美元指数）、P004       | FRED                          | 使用 `DEXCAUS`（CAD/USD）和 `DEXUSAL`（AUD/USD）下载日频数据，再计算周度变化率                                         |
| `dgs10_change` 10 年期国债收益率变化            | 利率与持有成本     | P076（收益率水平未通过单位根检验，因此需要差分） | 已有 `treasury_10y`             | 本地派生：使用 `treasury_10y.diff()` 计算一阶差分，无需重新下载                                                      |


## 统一处理原则

1. **按实际发布日期对齐**
  所有变量必须依据其真实发布日期进入模型，避免使用当时尚未发布的数据，从而防止前视偏差。
2. **正确处理月频变量**
  对 GPR、CLI、CRB 等月频指标转换为周频时，应从实际发布日期开始向后进行前向填充。
   不应将月底发布或代表整月的信息反向填充至该月月初。
3. **统一实施滞后处理**
  所有外生变量在进入预测模型前均应进行滞后处理，例如：
   具体滞后阶数应根据预测步长、数据频率和验证集表现确定。
4. **统一周频聚合规则**
  - 日频价格或指数：通常使用周均值或周末值；
  - 日频收益率：先计算日收益率，再根据研究设计聚合为周收益率；
  - 月频指数：按实际发布日期映射至周频，并向后填充；
  - 利率与汇率变量：优先使用周变化或一阶差分，而不是直接使用非平稳水平值。
5. **避免重复信息**
  在加入新变量后，应检查其与现有变量之间的相关性和共线性，尤其包括：
  - `gpr` 与 GDELT 扰动事件特征；
  - `ovx` 与已有波动率变量；
  - `commodity_fx` 与宽口径美元指数；
  - `global_econ_activity` 与工业生产、PMI 或其他需求代理变量；
  - `nonoil_industrial_commodity` 与已有大宗商品价格指数。## 待构建变量清单及数据源/下载方式


| 变量                                     | 机制          | 文献                         | 数据源                           | 下载方式                                                                                             |
| -------------------------------------- | ----------- | -------------------------- | ----------------------------- | ------------------------------------------------------------------------------------------------ |
| `gpr` 地缘政治风险                           | 预防性需求       | P076【最重要外部】、P072           | Caldara–Iacoviello GPR        | 从 `matteoiacoviello.com/gpr.htm` 下载 `gpr_web.xls`。月频数据，包含全球及国别 GPR 指数                            |
| `ovx` 原油隐含波动率                          | 石油特定不确定性    | P052（OVX > VIX）            | CBOE Crude Oil VIX            | 使用 FRED 代码 `OVXCLS` 下载日频数据，再聚合为周均值                                                               |
| `gold_return` 黄金收益率                    | 商品联动与避险渠道   | P004                       | LBMA / FRED                   | 使用 FRED 代码 `GOLDPMGBD228NLBM` 下载日频黄金价格，再计算对数收益率；也可使用 `yfinance` 下载 `GC=F`                        |
| `global_econ_activity` 全球经济活动          | 全球石油需求      | P053【强】、P052、P004、P072     | Kilian Index / 全球工业生产指数 / PMI | 优先使用 Dallas Fed 发布的 Kilian 全球经济活动指数（月频）；也可使用 FRED 的 OECD CLI，代码为 `OECDLOLITOAASTSAM`             |
| `nonoil_industrial_commodity` 非油工业商品价格 | 全球需求代理      | P053【短期强】、P054             | CRB 工业品指数 / 金属指数              | 可使用 FRED 的 IMF 工业原料价格指数 `PINDUINDEXM`，或全部商品指数 `PALLMINDEXM`；金属数据也可来自 LME 或 World Bank Pink Sheet |
| `futures_spread` 期货－现货价差               | 市场紧张程度与价格预期 | P053、P054、P072             | ICE Brent 期货曲线                | 使用 EIA 的 Brent 现货价格 `RBRTE`，结合 ICE 或 EIA 的 Brent 1–6 月期货价格；也可使用 `yfinance` 的 `BZ=F` 构建近月与远月价差    |
| `commodity_fx` 商品货币汇率                  | 汇率传导渠道      | P053（优于宽口径美元指数）、P004       | FRED                          | 使用 `DEXCAUS`（CAD/USD）和 `DEXUSAL`（AUD/USD）下载日频数据，再计算周度变化率                                         |
| `dgs10_change` 10 年期国债收益率变化            | 利率与持有成本     | P076（收益率水平未通过单位根检验，因此需要差分） | 已有 `treasury_10y`             | 本地派生：使用 `treasury_10y.diff()` 计算一阶差分，无需重新下载                                                      |


## 统一处理原则

1. **按实际发布日期对齐**
  所有变量必须依据其真实发布日期进入模型，避免使用当时尚未发布的数据，从而防止前视偏差。
2. **正确处理月频变量**
  对 GPR、CLI、CRB 等月频指标转换为周频时，应从实际发布日期开始向后进行前向填充。
   不应将月底发布或代表整月的信息反向填充至该月月初。
3. **统一实施滞后处理**
  所有外生变量在进入预测模型前均应进行滞后处理，例如：
   具体滞后阶数应根据预测步长、数据频率和验证集表现确定。
4. **统一周频聚合规则**
  - 日频价格或指数：通常使用周均值或周末值；
  - 日频收益率：先计算日收益率，再根据研究设计聚合为周收益率；
  - 月频指数：按实际发布日期映射至周频，并向后填充；
  - 利率与汇率变量：优先使用周变化或一阶差分，而不是直接使用非平稳水平值。
5. **避免重复信息**
  在加入新变量后，应检查其与现有变量之间的相关性和共线性，尤其包括：
  - `gpr` 与 GDELT 扰动事件特征；
  - `ovx` 与已有波动率变量；
  - `commodity_fx` 与宽口径美元指数；
  - `global_econ_activity` 与工业生产、PMI 或其他需求代理变量；
  - `nonoil_industrial_commodity` 与已有大宗商品价格指数。
  ## M1_market_macro 的变量层级构成

虽然这一组命名为 `M1_market_macro`，但在完成文献精读和变量修正后，它刻意混合了多个层面的信息。

这是因为 Kilian（P052）的结构性框架要求 M1 覆盖主要经济机制，包括：

- 石油供给；
- 全球需求；
- 预防性需求；
- 市场与金融条件。

因此，M1 并不只是由纯金融指标组成，而是一个以经济机制为基础的市场、宏观、基本面和风险变量集合。


| 层面          | 变量                                                                                                        | 说明                                                          |
| ----------- | --------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------- |
| 金融市场        | `brent_price`、`brent_log_return`、`futures_spread`、`vix`、`ovx`、`dgs10_change`、`gold_return`、`commodity_fx` | 覆盖价格、期货期限结构、隐含波动率、利率、汇率和避险资产，这些属于真正的市场与金融层面                 |
| 石油基本面（实物供需） | `crude_stocks_change`                                                                                     | EIA 原油库存变化，反映石油供需平衡和市场紧张程度，不属于纯金融指标                         |
| 宏观经济        | `global_econ_activity`                                                                                    | 使用 Kilian 全球经济活动指数、全球工业生产或 PMI，反映全球实体经济和石油需求状况，属于宏观经济层面     |
| 商品市场        | `nonoil_industrial_commodity`                                                                             | 使用 CRB 工业品、工业原材料或金属价格指数，兼具金融资产和实体需求代理属性，可反映全球工业需求           |
| 外部扰动 / 事件指数 | `gpr`                                                                                                     | Caldara–Iacoviello 基于新闻报道构建的风险指数，属于文本聚合和外部扰动风险层面，严格来说不是金融变量 |


## 数据池 vs 实验子集（重要区分）

"旧变量"有两个不同含义，必须区分：

| 层面 | 指什么 | 内容 | 是否"读文献前" |
| --- | --- | --- | --- |
| ① 数据列总池（data inventory） | `feature_groups_old.json` / 自动生成的 `feature_groups.json` 的 `M1_market_macro` | 最初 **27** 个市场/宏观列；精读后补下载 8 个 → 现 **35** 列 | 27 列是读文献前构建；该文件已含新增 8 列，是合并后的快照 |
| ② 旧实验喂给模型的 M1 子集 | 旧 `config.py` / EDA 硬编码的 `M1_VARS` | **9 个**（见下） | ✅ 是，凭经验从 27 列池子里挑的第一版 |
| ③ 新实验的 M1 子集（精读后） | 现 `config.py` / EDA 的 `M1_VARS` | **10 个**机制化（见下） | 精读后按 Kilian 三类机制重选 |

- **旧 M1（9 个，pre-精读）**：`brent_price, brent_log_return, crude_stocks_change, crude_production, refinery_utilisation, vix, dollar_index, sp500_return_pct, treasury_10y`
- **新 M1（10 个，post-精读）**：`brent_price, crude_stocks_change, global_econ_activity, nonoil_industrial_commodity, futures_spread, ovx, gpr, dgs10_change, gold_return, commodity_fx`

精读带来两件事：① 给数据池补了 8 个变量（27 → 35）；② 重选 M1 实验子集（旧 9 个凭经验 → 新 10 个按机制，并用优选项 OVX 替 VIX、ΔDGS10 替水平、商品货币替宽美元，移除被降级的 crude_production/refinery_utilisation/dollar_index/sp500_return_pct）。


## 新旧 M1 实验结果对比（XGBoost / RandomForest / Ridge 为主，深度模型两版均不稳定故略）

> 数据：`01_literature/Test/results/all_results_combined.csv`（新）与 git HEAD 版（旧）。
> ⚠️ 注意：旧组样本自 2006 起、新组自 2007-08 起（因 ovx/futures_spread 起始较晚），测试窗口略有差异，差异中混入少量样本期因素；但提升在"全模型 × 全 target"上高度一致，基本可判定为变量本身更优。

**Price（R²，越高越好）**

| 模型 | 旧 | 新 | Δ |
| --- | --- | --- | --- |
| Ridge | 0.835 | 0.867 | +0.032 |
| XGBoost | 0.748 | 0.835 | +0.087 |
| RandomForest | 0.825 | 0.833 | +0.008 |

**Volatility（R²）— 提升最显著**

| 模型 | 旧 | 新 | Δ |
| --- | --- | --- | --- |
| XGBoost | −0.25 | +0.076 | +0.33 |
| RandomForest | −0.84 | +0.10 | +0.94 |
| Ridge | −1.36 | −0.15 | +1.21 |

XGBoost/RandomForest 从负 R²（不如猜均值）转为正 R²，很可能由新增的 `ovx`/`gpr`/`gold_return` 等不确定性类变量驱动。

**Direction（directional_acc）**

| 模型 | 旧 | 新 | Δ |
| --- | --- | --- | --- |
| RandomForest | 0.448 | 0.529 | +0.081 |
| XGBoost | 0.407 | 0.485 | +0.078 |
| SVM | 0.476 | 0.515 | +0.039 |

**结论**：新 10 变量组在 price / volatility / direction 上对所有可信模型均优于旧 9 变量组。

**遗留 caveat**：当前实验仍缺随机游走/Naive 基准 → price 的高 R² 受价格持续性影响，不能直接解读为预测能力（P053/P001 警告），后续需补 RW 基准并报告相对 RW 的样本外 R² 与 DM 检验。


