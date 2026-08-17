# Appendix C — Hyperparameter grids & locked settings / 附录 C — 超参数网格与锁定设定

This appendix records the locked hyperparameter grids and training settings used
for the reported results. Software versions, installation commands and scripts
are in the GitHub repository.

本附录记录报告结果所用的超参数网格与锁定设定。软件版本、安装命令与脚本见
GitHub 仓库。

---

## C.1 Flat search grids / Flat 搜索网格

Hyperparameters are chosen inside each training fold on the inner-validation
segment only.

超参数仅在各训练折的内部验证段上选择。


| Learner / 学习器           | Grid / 网格                                                 |
| ----------------------- | --------------------------------------------------------- |
| Ridge (α)               | {0.1, 1.0, 10.0, 100.0, 1000.0}                           |
| XGBoost `max_depth`     | {2, 3}                                                    |
| XGBoost `learning_rate` | {0.03, 0.05}                                              |
| XGBoost `n_estimators`  | {200, 400}                                                |
| XGBoost fixed           | `subsample=0.8`, `colsample_bytree=0.8`, `reg_lambda=1.0` |


---



## C.2 Deep architecture and training

The main specification is locked to lookback 4 and latent size 32, matching the
Flat lookback. Sensitivity is reported in Appendix B.

主设定锁定为回看 4 周、潜在维 32，与 Flat 回看一致。敏感性见附录 B。

### C.2.1 Encoders, fusion and regression head / 编码器、融合与回归头

The shared fused latent size is 32. The shipping encoder uses a wider internal
width (`d_model=64`); only the pooled branch output is projected to 32-d before
fusion. Both the finance and shipping TCN stacks reuse the same causal
`TemporalTCN` (kernel 3).

融合后的共享潜在维为 32。航运编码器内部宽为 64，仅在池化后投影到 32 维再进入融合。
金融与航运 TCN 复用同一因果 `TemporalTCN`（kernel 3）。


| Component / 组件 | Settings / 设定 | Output / 输出 |
| --- | --- | --- |
| Finance TCN | internal `d_model=32` (equal to output); 2 layers, kernel 3, causal, dropout 0.1; branch head Linear(32, 32)+ReLU | 32-d |
| Remote sensing | frozen Prithvi embeddings (1024-d), temporal then site attention | 32-d |
| Shipping GAT | 17 nodes; type-specific projection to internal `d_model=64` (≠ fused 32-d); 2 GAT layers, 4 heads, LeakyReLU slope 0.2; 2-layer TCN (kernel 3); branch head Linear(64, 64) → ReLU → Dropout(0.1) → Linear(64, 32) | 32-d |
| Gated fusion (main) / 门控融合（主设定） | Linear(n×32, 32) → ReLU → Linear(32, n) → softmax | 32-d |
| Cross-attention (alternative) / 交叉注意力（备选） | finance as query, `n_heads=4`, `token_dim=64` (matches shipping internal width), γ=0.1 | 32-d |
| Regression head / 回归头 | Linear(32, 32) → ReLU → Dropout(0.1) → Linear(32, 1) | scalar |




### C.2.2 Training / 训练


| Item / 项       | Value / 取值                    |
| -------------- | ----------------------------- |
| Optimiser      | Adam                          |
| Learning rate  | 1e-3                          |
| Weight decay   | 1e-4                          |
| Dropout        | 0.1                           |
| Batch size     | 32                            |
| Maximum epochs | 80                            |
| Early stopping | inner validation, patience 12 |


After early stopping, the checkpoint with the lowest inner-validation loss is
kept for the subsequent forecast block. The model is not refit on the combined
training and validation sample.

早停后保留内部验证损失最低的 checkpoint，用于随后的预测块；不再用合并后的
训练与验证样本重新拟合。
