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
| Nested test / 嵌套检验 | Clark–West (vs M1, and vs M0 for "beats random walk") |
| Non-nested test / 非嵌套检验 | Diebold–Mariano, HLN small-sample corrected |
| Seed / 种子 | **42** (main); 1, 2 for robustness |

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

---

## C.5 Entry points & outputs / 入口脚本与输出

| Purpose / 用途 | Script / 脚本 | Output dir / 输出目录 |
| --- | --- | --- |
| Flat M0–M4 baselines / 扁平基线 | `04_code/scripts/flat/run_baseline.py` (+ `flat/M{1..4}_Flat/*.py`) | `05_outputs/baselines/Flat/M*_Flat/` |
| Deep baselines & fusion / 深度基线与融合 | `04_code/scripts/deep/run_deep_baseline.py` | `05_outputs/baselines/Deep/{M*_Deep,_cross}/` |
| Deep sweeps (seed/lookback/dim/reg) / 深度扫描 | `04_code/scripts/deep/run_deep_sweep.py` | `05_outputs/baselines/Deep/_cross/deep_sweep_summary.csv` |
| Fusion matrix (3×3) / 融合矩阵 | `run_deep_fusion_matrix.py` | `deep_fusion_matrix.{csv,png}` |
| Advanced ablations (fusion/dropout/sub-period) / 进阶消融 | `run_deep_advanced.py` | `deep_advanced_summary.csv` |
| Sub-period early/late (Flat + Deep, offline) / 早晚子期（离线） | `subperiod_eval.py` | `05_outputs/baselines/subperiod/subperiod_summary.csv` |
| Interpretability (gates, attention) / 可解释性 | `run_deep_interpret.py`, `run_deep_interpret_m3.py`, `run_deep_xattn_viz.py` | `deep_interpret*.png`, `deep_*gate*.csv` |
| Feature matrix build / 特征矩阵构建 | `03_data/processed/**/build_*.py`, `merge/py/build_feature_matrix.py` | `03_data/processed/merge/outputs/` |

Reproduce end-to-end / 端到端复现：

```bash
python3 -m pip install -r 04_code/requirements.txt
python3 04_code/scripts/flat/run_baseline.py --modality M3      # flat example
python3 04_code/scripts/deep/run_deep_baseline.py               # deep main
python3 04_code/scripts/tools/subperiod_eval.py                  # early/late table
```
