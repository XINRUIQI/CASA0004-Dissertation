# Appendix C — Hyperparameter grids & locked settings / 附录 C — 超参数网格与锁定设定

This appendix records the software versions and locked settings used for the
reported results. Installation commands and scripts are in the GitHub repository.

本附录记录报告结果所用的软件版本与锁定设定。安装命令与脚本见 GitHub 仓库。

---

## C.1 Software environment / 软件环境

Python 3.9.6 (CPython, macOS). Core packages:

| Package | Version | Role / 用途 |
| --- | --- | --- |
| numpy | 2.0.2 | arrays / 数值 |
| pandas | 2.3.3 | weekly matrix / 周频矩阵 |
| scipy | 1.13.1 | *p* values / *p* 值 |
| scikit-learn | 1.6.1 | Ridge, scaling |
| xgboost | 2.1.4 | Flat XGBoost |
| torch | 2.8.0 | Deep encoders and fusion (CPU) |
| matplotlib | 3.9.4 | figures / 作图 |
| shap | 0.49.1 | attribution / 归因 |

The Deep remote-sensing branch uses pre-computed frozen Prithvi-EO-2.0 embeddings.
Training and evaluation do not load the foundation model.

深度遥感分支使用预先计算并冻结的 Prithvi-EO-2.0 嵌入；训练与评价不加载基础模型。

---

## C.2 Shared protocol and seeds / 共用协议与种子

Flat and Deep use the same rolling-origin schedule.

Flat 与 Deep 使用同一滚动起点安排。

| Item / 项 | Value / 取值 |
| --- | --- |
| Sample | 2019–2025 (365 weeks) |
| Lookback | **4 weeks** |
| Initial training (not scored) | **104 weeks** |
| Re-estimation interval | **13 weeks** |
| Inner validation | **52 weeks** (tail of each training fold) |
| Evaluation sample | **257 weeks** |
| Main seed | **42** |
| Robustness seeds | **1, 2** |

---

## C.3 Flat search grids / Flat 搜索网格

Hyperparameters are chosen inside each training fold on the inner-validation
segment only.

超参数仅在各训练折的内部验证段上选择。

| Learner / 学习器 | Grid / 网格 |
| --- | --- |
| Ridge (α) | {0.1, 1.0, 10.0, 100.0, 1000.0} |
| XGBoost `max_depth` | {2, 3} |
| XGBoost `learning_rate` | {0.03, 0.05} |
| XGBoost `n_estimators` | {200, 400} |
| XGBoost fixed | `subsample=0.8`, `colsample_bytree=0.8`, `reg_lambda=1.0` |

---

## C.4 Locked Deep architecture and training / 锁定的 Deep 架构与训练

The main specification is locked to lookback 4 and latent size 32, matching the
Flat lookback. Sensitivity is reported in Appendix B.

主设定锁定为回看 4 周、潜在维 32，与 Flat 回看一致。敏感性见附录 B。

### C.4.1 Encoders / 编码器

| Encoder / 编码器 | Settings / 设定 | Output / 输出 |
| --- | --- | --- |
| Finance TCN | 2 layers, kernel 3, causal, dropout 0.1 | 32-d |
| Remote sensing | frozen Prithvi embeddings (1024-d), temporal then site attention | 32-d |
| Shipping GAT | 17 nodes, 2 GAT layers, 4 heads, then 2-layer TCN | 32-d |

### C.4.2 Fusion / 融合

| Option / 选项 | Role / 角色 |
| --- | --- |
| Concatenation | alternative |
| **Gated fusion** | **main specification** (modality weights for RQ3) |
| Cross-attention (finance as query, 4 heads) | secondary comparison |

### C.4.3 Training / 训练

| Item / 项 | Value / 取值 |
| --- | --- |
| Optimiser | Adam |
| Learning rate | 1e-3 |
| Weight decay | 1e-4 |
| Dropout | 0.1 |
| Batch size | 32 |
| Maximum epochs | 80 |
| Early stopping | inner validation, patience 12 |
| Device | CPU |
| Seed | 42 (robustness: 1, 2) |

After early stopping, the checkpoint with the lowest inner-validation loss is
kept for the subsequent forecast block. The model is not refit on the combined
training and validation sample.

早停后保留内部验证损失最低的 checkpoint，用于随后的预测块；不再用合并后的
训练与验证样本重新拟合。
