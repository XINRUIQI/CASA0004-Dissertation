# Appendix C — Hyperparameter grids & locked settings / 附录 C — 超参数网格与锁定设定

> This appendix pins every value needed to reproduce the reported results.
> §C.1 lists the software environment; §C.2 the shared leakage-safe protocol;
> §C.3 the Flat search grids; §C.4 the locked Deep architecture and training;
> §C.5 the entry-point scripts and output paths.
>
> 本附录锁定复现所需的全部取值。§C.1 软件环境；§C.2 共享无泄漏协议；
> §C.3 扁平搜索网格；§C.4 锁定的深度架构与训练；§C.5 入口脚本与输出路径。

---

## C.1 Software environment / 软件环境

Reproduced with **Python 3.9.6** (CPython, macOS). Full pinned list in
`04_code/requirements.txt`; core packages:

以 **Python 3.9.6** 复现。完整锁定清单见 `04_code/requirements.txt`；核心依赖：

| Package | Version | Role / 用途 |
| --- | --- | --- |
| numpy | 2.0.2 | arrays / 数值 |
| pandas | 2.3.3 | weekly matrix, joins / 周频矩阵与合并 |
| scipy | 1.13.1 | DM / Clark-West test statistics / 检验统计量 |
| scikit-learn | 1.6.1 | Ridge, pipelines, VarianceThreshold, StandardScaler |
| xgboost | 2.1.4 | Flat XGBoost learner / 扁平 XGB |
| torch | 2.8.0 | Deep encoders + fusion (CPU) / 深度编码器与融合 |
| matplotlib | 3.9.4 | figures / 作图 |
| shap | 0.49.1 | Flat feature attribution / 扁平特征归因 |

> **On foundation models / 关于基础模型**：the Deep RS branch consumes
> **pre-computed frozen Prithvi-EO-2.0 embeddings** stored on disk; training and
> evaluation do **not** import `transformers` or fetch any model online (the test
> machine has no `transformers` installed). The one-off embedding-export step is
> separate and not part of this environment. / 深度 RS 分支消费**离线预计算的冻结
> Prithvi-EO-2.0 嵌入**（存盘）；训练与评估**不**在线加载模型、不依赖
> `transformers`（测试机甚至未安装）。嵌入导出为一次性独立步骤，不属于本环境。

Install / 安装：

```bash
python3 -m pip install -r 04_code/requirements.txt
```

---

## C.2 Shared leakage-safe protocol (Flat = Deep) / 共享无泄漏协议

Both families use the identical rolling-origin schedule so architecture
differences are not confounded with protocol differences.

两族使用完全相同的滚动起点日程，使架构差异不与协议差异混淆。

| Item / 项 | Value / 取值 |
| --- | --- |
| Comparison window / 比较窗口 | 2019–2025 (365 weeks in the merged matrix) |
| Lookback / 回看 | **4 weeks** |
| Warm-up `min_train` / 热身 | **104 weeks** (not scored) |
| Refit cadence `retrain_every` / 重训周期 | **13 weeks** |
| Inner-validation `val_weeks` / 内验证 | **52 weeks** (tail of each training fold) |
| Common scored test span / 计分测试区间 | **257 weeks** (2021-01 → 2025-12) |
| Target / 目标 | one-week log return \(r_{t+1}\), reconstructed to price |
| Primary metric / 主指标 | RMSE + MAE on reconstructed price; skill vs M0 |
| Primary test / 主检验 | Diebold–Mariano with HLN small-sample correction, on reconstructed-price squared error, for **every** formal comparison (vs M0, vs S1, and Flat vs Deep) / 全部正式比较统一使用带 HLN 修正的 DM，作用于重构价格平方误差 |
| Test direction / 检验方向 | one-sided where the research question is directional (vs M0, vs S1, Deep vs Flat); two-sided for Deep fusion-mechanism comparisons / 研究问题方向性者单侧，Deep 融合机制比较双侧 |
| Multiplicity / 多重比较 | Holm within three frozen families: benchmark (18), RQ1 (15), RQ2 (14); raw and adjusted p both reported, formal claims on adjusted / 族内 Holm，三族分别 18、15、14 项，原始与调整后 p 并报，正式结论用调整后 |
| Supplementary test / 补充检验 | Clark–West, **Ridge only** (5 comparisons), never for XGBoost or Deep / 仅用于 Ridge 的 5 项，不用于 XGBoost 与 Deep |
| Unified test table / 统一检验表 | `05_outputs/tests/test_table_main.csv` via `04_code/scripts/tools/build_test_tables.py` |
| Seed / 种子 | **42** (main); 1, 2 for robustness |
| Flat leading-gap fill / Flat 前导缺口 | **locked**: RS anomalies → 0; shipping counts → training-fold median. Not varied in Appendix B. / **锁定**：遥感距平填 0，航运计数用训练折中位数。附录 B 不扫描替代填法。 |

**Variance estimation in the DM statistic / DM 统计量的方差估计.** The loss
differential is \(d_t = L_{\text{reference},t} - L_{\text{candidate},t}\), so a
positive statistic means the candidate is the more accurate forecast. Its long-run
variance is estimated with the usual truncation at \(h-1\) autocovariances, where
\(h\) is the forecast horizon. Every comparison in this study is a one-week-ahead
forecast, so \(h = 1\) and no autocovariance term enters: the variance reduces to
\(\hat{\gamma}_0 / T\), where \(\hat{\gamma}_0 = T^{-1}\sum_t (d_t - \bar{d})^2\).
The Harvey–Leybourne–Newbold
finite-sample factor \(\sqrt{[T + 1 - 2h + h(h-1)/T]\,/\,T}\) is then applied,
which at \(h = 1\) equals \(\sqrt{(T-1)/T}\), and the statistic is referred to a
\(t\) distribution with \(T-1\) degrees of freedom. Implementation:
`04_code/src/backtest/metrics.py::dm_test`. / 损失差定义为
\(d_t = L_{\text{参照},t} - L_{\text{候选},t}\)，故统计量为正表示候选更准。其长期
方差按惯例截断至 \(h-1\) 阶自协方差，\(h\) 为预测期。本研究所有比较均为提前一周
预测，故 \(h = 1\)，不引入任何自协方差项：方差退化为 \(\hat{\gamma}_0 / T\)，
其中 \(\hat{\gamma}_0 = T^{-1}\sum_t (d_t - \bar{d})^2\)。
随后施加 Harvey–Leybourne–Newbold 有限样本因子
\(\sqrt{[T + 1 - 2h + h(h-1)/T]\,/\,T}\)，在 \(h = 1\) 时即 \(\sqrt{(T-1)/T}\)，
并以自由度 \(T-1\) 的 \(t\) 分布判读。

---

## C.3 Flat search grids / 扁平搜索网格

Chosen inside each training fold on the past `val_weeks` slice only
(`backtest/rolling.py::tune_hyperparams`); no test-set peeking. Every pipeline
begins with `VarianceThreshold(0.0)` (drops in-fold constant columns), and the
linear model adds `StandardScaler` (fit on the training fold only). Source:
`04_code/src/backtest/models.py`.

在各训练折内、仅用过去 `val_weeks` 切片选取；不看测试集。每条管线先
`VarianceThreshold(0.0)`，线性模型再加 `StandardScaler`（仅在训练折拟合）。

| Learner / 学习器 | Grid / 网格 |
| --- | --- |
| Ridge (α) | {0.1, 1.0, 10.0, 100.0, 1000.0}（default 10.0） |
| XGBoost `max_depth` | {2, 3} |
| XGBoost `learning_rate` | {0.03, 0.05} |
| XGBoost `n_estimators` | {200, 400} |
| XGBoost fixed / 固定 | `subsample=0.8`, `colsample_bytree=0.8`, `reg_lambda=1.0` |

---

## C.4 Locked Deep architecture & training / 锁定的深度架构与训练

Sweeps explored lookback ∈ {4, 8, 12}, representation dim ∈ {32, 64}, fusion ∈
{concat, gated, cross-attention}, dropout / weight-decay, and seeds; the main
specification is **locked to lookback = 4 and d = 32** for protocol parity with
the flat baselines. Sensitivity is reported in Appendix B. Sources:
`deep_rolling.py`, `finance_encoder.py`, `rs_encoder.py`, `shipping_encoder.py`,
`fusion.py`.

扫描覆盖回看、表示维、融合类型、正则与种子；主设定**锁定 lookback = 4、d = 32**
以与扁平基线对齐。敏感性见附录 B。

### C.4.1 Encoders / 编码器

| Encoder / 编码器 | Key settings / 关键设置 | Output / 输出 |
| --- | --- | --- |
| Finance TCN | `d_model=32`, `tcn_layers=2`, `kernel=3`, causal, `dropout=0.1` | z_fin, **32-d** |
| RS (frozen Prithvi) | `emb_dim=1024`, `n_sites=11`, `d_model=64`; temporal-attention + AOI-site-attention pooling | z_rs, **32-d** |
| Shipping graph | 17 nodes (11 AOI + 6 chokepoints); type-specific projection + node-type embedding; **GAT layers = 2, heads = 4**; `log1p(O-D flow)` as attention prior (learned `edge_scale`); adjacency symmetrised + self-looped; then **TCN layers = 2**; node-attention pooling; `d_model=64` | z_ship, **32-d**; ≈ 42k params |

### C.4.2 Fusion / 融合

| Option / 选项 | Role / 角色 |
| --- | --- |
| encoder-concat | fusion-ladder floor / 阶梯下限 |
| **gated** (softmax gate over modality embeddings) | **main reported model / 主报告模型**（gate 权重亦供 RQ3） |
| cross-attention (finance as query, `n_heads=4`) | comparative / 进阶对照（single-seed best but higher variance） |

### C.4.3 Training / 训练

| Item / 项 | Value / 取值 |
| --- | --- |
| Optimiser / 优化器 | Adam |
| Learning rate / 学习率 | `1e-3` |
| Weight decay / 权重衰减 | `1e-4` |
| Dropout | `0.1` |
| Batch size / 批大小 | `32` |
| Epochs (max) / 最大轮数 | `80` |
| Early stopping / 早停 | on last `val_weeks` of fold, `patience=12` |
| Modality dropout / 模态 dropout | `0.0` main（`0.3` robustness arm） |
| Device / 设备 | CPU |
| Seed / 种子 | `42`（robustness: 1, 2） |

After early stopping, the weights from the epoch with the lowest inner-validation
loss are restored and used for the subsequent forecast block. The model is not
refit on the inner-validation weeks. / 早停后恢复内部验证损失最低那一轮的权重，
用于随后的预测块；不在内部验证周上重新拟合。

---

## C.5 Entry points & outputs / 入口脚本与输出

| Purpose / 用途 | Script / 脚本 | Output dir / 输出目录 |
| --- | --- | --- |
| Flat M0–M4 baselines / 扁平基线 | `04_code/scripts/flat/run_baseline.py` (+ `flat/M{1..4}_Flat/*.py`) | `05_outputs/baselines/Flat/M*_Flat/` |
| Table 4.2 / B.2 fusion matrix / 融合矩阵 | `04_code/scripts/deep/run_deep_fusion_matrix.py` | `05_outputs/baselines/Deep/_cross/deep_fusion_matrix.csv` |
| Deep multi-seed (B.4) / 深度多种子 | `04_code/scripts/tools/pool_deep_seeds.py` | `05_outputs/baselines/Deep/_cross/deep_seed_{pooled,summary}.csv` |
| Deep sweeps (B.4.1–B.4.3) / 深度扫描 | `04_code/scripts/deep/run_deep_sweep.py` | `05_outputs/baselines/Deep/_cross/deep_sweep_summary.csv` |
| Advanced ablations (fusion/dropout/sub-period) / 进阶消融 | `run_deep_advanced.py` | `deep_advanced_summary.csv` |
| Sub-period early/late (Flat + Deep, offline) / 早晚子期（离线） | `subperiod_eval.py` | `05_outputs/baselines/subperiod/subperiod_summary.csv` |
| **Frozen comparison families + Holm** / 冻结检验族与 Holm | `04_code/scripts/tools/build_test_tables.py` | `05_outputs/tests/test_table_{main,cw_supplementary,robustness}.csv` |
| Interpretability (gates, attention) / 可解释性 | `run_deep_interpret.py`, `run_deep_interpret_m3.py`, `run_deep_xattn_viz.py` | `deep_interpret*.png`, `deep_*gate*.csv` |
| Feature matrix build / 特征矩阵构建 | `03_data/processed/**/build_*.py`, `merge/py/build_feature_matrix.py` | `03_data/processed/merge/outputs/` |

Reproduce end-to-end / 端到端复现：

```bash
python3 -m pip install -r 04_code/requirements.txt
python3 04_code/scripts/flat/run_baseline.py --modality M3      # Table 4.1
python3 04_code/scripts/deep/run_deep_fusion_matrix.py          # Table 4.2 / B.2
python3 04_code/scripts/tools/pool_deep_seeds.py                # B.4 multi-seed
python3 04_code/scripts/tools/subperiod_eval.py                  # early/late table
python3 04_code/scripts/tools/build_test_tables.py               # all reported p-values
```
