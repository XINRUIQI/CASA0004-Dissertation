# Appendix B — Extra results & robustness / 附录 B — 额外结果与稳健性

All checks use the same protocol as Chapter 3 (lookback 4, expanding window, 257
scored weeks). 

---

## B.1 Remote-sensing leave-one-AOI-out (LOAO) / 遥感留一站点

Flat S2 (finance + remote sensing). Base RMSE: Ridge 4.414, XGBoost 4.440.
ΔRMSE = dropped site − full S2.

扁平 S2（金融 + 遥感）。ΔRMSE = 去掉该站后的 RMSE − 完整 S2。


| Dropped AOI     | Ridge RMSE | Δ      | XGB RMSE | Δ      |
| --------------- | ---------- | ------ | -------- | ------ |
| (none / full)   | 4.414      | 0      | 4.440    | 0      |
| Al Basrah Terminal | 4.424      | +0.010 | 4.462    | +0.021 |
| Fujairah        | 4.375      | −0.039 | 4.422    | −0.018 |
| Houston         | 4.408      | −0.006 | 4.349    | −0.091 |
| Jamnagar        | 4.410      | −0.004 | 4.402    | −0.038 |
| Jurong          | 4.449      | +0.035 | 4.450    | +0.010 |
| Kharg           | 4.343      | −0.071 | 4.427    | −0.013 |
| Ningbo-Zhoushan | 4.397      | −0.016 | 4.386    | −0.054 |
| Ras Tanura      | 4.406      | −0.008 | 4.401    | −0.039 |
| Rotterdam       | 4.424      | +0.010 | 4.446    | +0.006 |
| Ulsan           | 4.423      | +0.009 | 4.410    | −0.030 |
| Yanbu           | 4.374      | −0.040 | 4.393    | −0.048 |


Removing any single site leaves the ranking unchanged. The largest shift is 0.091
for XGBoost (Houston) and 0.071 for Ridge (Kharg), each about 2% of RMSE.
Dropping a site more often helps than hurts, so no individual location carries
the remote-sensing signal.

去掉任一站点不改变排序。最大变动约为 RMSE 的 2%。去掉站点往往使误差下降，
说明遥感信号并不由单一地点支撑。

---



## B.2 Deep fusion matrix / 深度融合矩阵

Entries are RMSE improvement versus M0 (%).
Positive values indicate lower RMSE than the no-change benchmark.

种子 42。表中为相对 M0 的 RMSE 改善（%）。正值表示优于不变预测。


| Set                          | Concat | Gated     | Cross-attention | *p* vs M0 (best cell) |
| ---------------------------- | ------ | --------- | --------------- | ------------------------- |
| S2 (finance + RS)            | −2.01  | −2.43     | −5.87           | —                         |
| **S3 (finance + shipping)**  | −0.22  | **+0.15** | **+1.00**       | 0.257 (cross-attention)   |
| S4 (finance + RS + shipping) | −8.30  | −0.68     | +0.19           | 0.427 (cross-attention)   |


M0 is cleared only where shipping is present. S2 never beats M0.
Concatenation clears M0 in no set. The three positive cells are descriptive
orderings on one seed: the smallest *p* versus M0 is 0.257. 

本种子下，只有含航运的设定越过 M0；S2 从未优于 M0；拼接在任何集合上都未越过
M0。三个正值是单一种子上的描述性排序，相对 M0 的最小 *p* 为 0.257。跨种子
均值见 B.3。

---



## B.3 Deep multi-seed / Deep 多种子

Seeds 42, 1 and 2 for all ten Deep specifications. Improvement versus M0 (%). S1 has no fusion operator.

种子 42、1、2；十个 Deep 设定全部列入。种子 42 与表 4.2、B.2 为同一运行。S1
无融合算子。


| Set | Model            | Seed 42 | Seed 1 | Seed 2 | Mean ± SD     | Positive runs |
| --- | ---------------- | ------- | ------ | ------ | ------------- | ------------- |
| S1  | Deep             | −2.36   | −0.95  | +0.30  | −1.00% ± 1.33 | 1/3           |
| S2  | Concat           | −2.01   | −2.42  | −0.94  | −1.79% ± 0.77 | 0/3           |
| S2  | Gated            | −2.43   | −1.96  | −5.07  | −3.15% ± 1.67 | 0/3           |
| S2  | Cross-attention  | −5.87   | −0.50  | −4.33  | −3.57% ± 2.77 | 0/3           |
| S3  | Concat           | −0.22   | −0.64  | +0.07  | −0.27% ± 0.35 | 1/3           |
| S3  | **Gated (main)** | +0.15   | −1.39  | −0.29  | −0.51% ± 0.80 | 1/3           |
| S3  | Cross-attention  | +1.00   | −2.87  | −7.14  | −3.01% ± 4.07 | 1/3           |
| S4  | Concat           | −8.30   | −2.09  | −0.98  | −3.79% ± 3.95 | 0/3           |
| S4  | Gated            | −0.68   | −0.86  | −1.19  | −0.91% ± 0.26 | 0/3           |
| S4  | Cross-attention  | +0.19   | −0.87  | −5.01  | −1.90% ± 2.75 | 1/3           |


Every mean is below M0. Five of thirty runs are positive. No S2 fusion is
positive in any seed. S4 concatenation has the weakest mean (−3.79%). The
positive Chapter 4 figures for gated S3 (+0.15%) and cross-attention S3
(+1.00%) are seed-42 outcomes. Each S3 fusion is positive in exactly one of
three seeds. Across seeds the S3 order reverses: concatenation (−0.27%) >
gated (−0.51%) > cross-attention (−3.01%). 

所有跨种子均值均低于 M0。三十次运行中五次为正。S2 三种融合在任一种子上均为
负。S4 拼接均值最差（−3.79%）。第 4 章中门控 S3 与交叉注意力 S3 的正改善是
种子 42 的结果。S3 三种融合各只有 1/3 次为正。跨种子后排序反转。仍以门控为
主设定，是因为它为 RQ3 提供模态权重，而不是因为它更准。

---



## B.4 Matched Flat–Deep comparisons / 匹配 Flat–Deep 比较

Same eight pairs as Table 4.3. Improvement is the Deep RMSE reduction relative
to the matched Flat model, \(100\times(1-\mathrm{RMSE}_\text{Deep}/\mathrm{RMSE}_\text{Flat})\).
Positive values indicate a lower Deep RMSE. *p* is the probability of a
difference at least this large if the two models had the same RMSE. Smaller *p*
indicates stronger evidence that their RMSEs differ. *n* = 257.

与表 4.3 相同的八组配对。改善为 Deep 相对匹配 Flat 的 RMSE 降幅。正值表示
Deep 的 RMSE 更低。*p* 为两模型 RMSE 相同这一假设下，出现至少如表所示差异的
概率。*p* 越小，越不支持二者误差相同。


| Feature set | Flat   | Deep          | Improvement (%) | *p*   |
| ----------- | ------ | ------------- | --------------- | ----- |
| S1          | Ridge  | Deep          | +0.15           | 0.466 |
| S1          | XGB    | Deep          | +2.71           | 0.097 |
| S2          | Ridge  | Deep gated    | +3.64           | 0.096 |
| S2          | XGB    | Deep gated    | +4.22           | 0.042 |
| S3          | Ridge  | Deep gated    | +6.78           | 0.064 |
| S3          | XGB    | Deep gated    | +5.95           | 0.010 |
| S4          | Ridge  | Deep gated    | +7.85           | 0.029 |
| S4          | XGB    | Deep gated    | +7.23           | 0.009 |


All eight pairs favour Deep. Four have *p* below 0.05.

八组均有利于 Deep，其中四组 *p* 低于 0.05。

---



## B.5 Publication-lag sweep / 发布滞后扫描

Locked as-of lags are GFW monthly presence +4 weeks and monthly macro +5 weeks
(Appendix A.3). Alternative lags are an extra calendar shift on already-lagged
series. *n* = 257.

锁定滞后见附录 A.3。备选滞后是在已滞后序列上再平移。

**GFW monthly presence (Flat S3)**


| Lag (weeks)    | Ridge RMSE | Improvement vs M0 (%) | XGB RMSE  | Improvement vs M0 (%) | XGB *p* vs S1 |
| -------------- | ---------- | --------------------- | --------- | --------------------- | ------------- |
| 1              | 4.407      | −6.15     | 4.801     | −15.63    | 0.972             |
| **4 (locked)** | **4.447**  | **−7.11** | **4.408** | **−6.17** | **0.633**         |
| 8              | 4.334      | −4.38     | 4.396     | −5.88     | 0.603             |


**Monthly macro (Flat S1: REA and non-oil commodity)**


| Lag (weeks)    | Ridge RMSE | Improvement vs M0 (%) | XGB RMSE  | Improvement vs M0 (%) |
| -------------- | ---------- | --------------------- | --------- | --------------------- |
| 3              | 4.255      | −2.49     | 4.399     | −5.95     |
| **5 (locked)** | **4.256**  | **−2.52** | **4.368** | **−5.22** |
| 7              | 4.245      | −2.23     | 4.388     | −5.68     |


No lag beats M0. Shortening GFW to +1 week makes XGBoost substantially worse.
Monthly-macro lag moves S1 by at most 0.03 RMSE. The locked buffers are not the
reason Flat models with spatial features fail to clear M0.

没有任何滞后优于 M0。GFW 缩为 +1 周使 XGBoost 明显变差。月度宏观滞后最多移动
0.03 RMSE。锁定缓冲不是 Flat 另类数据未能越过 M0 的原因。