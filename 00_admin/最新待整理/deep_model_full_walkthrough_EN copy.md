# Deep Model

> Three **modality-specific encoders** learn separate **representations (embeddings)** of financial state / remote-sensing activity / shipping network, then **dynamically fuse** them via gating / cross-attention to predict next-week Brent price end-to-end.
>
> **Core question (RQ2)**: on the same data, does **modality-aware representation-level fusion** beat **flat feature fusion** (all columns stacked into one table)?

---

## Table of contents

1. [First: deep model vs flat model](#1)
2. [Raw data](#2)
3. [Three modality encoders](#3)
4. [Fusion module](#4)
5. [Backtest protocol](#5)
6. [Main results](#6)
7. [RQ2 answer](#7)
8. [RQ3 answer](#8)

---



## 1. First: deep model vs flat model

**Prediction target** (identical to the flat model): next-week Brent spot price P_{t+1} (USD/barrel, Friday close).

**The two fusion paradigms (the dissertation's selling point)**:


|                    | Flat fusion (baseline)                             | Representation-level fusion (this deep model)                           |
| ------------------ | -------------------------------------------------- | ----------------------------------------------------------------------- |
| Handling           | flatten all columns into one wide table (212 cols) | each modality goes through its **own encoder** to a 32-d representation |
| Model              | Ridge / XGBoost / LSTM early-fusion                | 3 encoders + gated / cross-attention fusion                             |
| Modality structure | lost (no structure across columns)                 | preserved (temporal, graph, AOI-site structure live inside encoders)    |
| Missing modality   | forced in via fill values                          | modelled explicitly via encoder + mask + modality dropout               |


**The three modality encoders at a glance**:

```text
finance series ──► Finance Encoder (causal TCN) ───────────────────► z_fin  (32-d)
satellite imagery ──► frozen Prithvi-EO-2.0 → embedding
                       └► temporal attention + AOI site attention ──► z_rs   (32-d)
shipping dyn. graph ──► GAT (spatial) → causal TCN (temporal) → node-att pool ─► z_ship (32-d)
      z_fin, z_rs, z_ship
        └► Gated Cross-Modal Fusion ──► z_fused ──► head ──► r̂_{t+1} ──► reconstructed P̂_{t+1}
```

---



## 2. Raw data



### 2.1 M1 finance / macro

Same as flat-model M1.

### 2.2 M2 remote sensing

- **Deep model**: **monthly Sentinel-2 6-band image patches** (GeoTIFF) for 11 oil sites. Bands B2/B3/B4/B8A/B11/B12 match Prithvi's expected HLS bands.
- **Flat model**: **monthly optical-index CSVs** precomputed from imagery (NDVI/NDWI/NDBI/BSI from 2017-04; VIIRS night-time lights NTL from 2014-01).

**The 11 AOI oil sites**

### 2.3 M3 shipping


| Source                     | Content                                                                                                             | Frequency     | Since |
| -------------------------- | ------------------------------------------------------------------------------------------------------------------- | ------------- | ----- |
| IMF PortWatch              | 6 chokepoints transit tanker/cargo counts, capacity; port import/export tanker volume                               | daily         | 2019+ |
| GFW (Global Fishing Watch) | chokepoint AIS vessel-presence hours; AOI port-visit events + dwell; AOI↔AOI voyage O-D; SAR dark-vessel detections | monthly/event | 2012+ |


**The 6 chokepoints**

---



## 3. Three modality encoders

All output a **32-d (d=32)** representation. High dimensions overfit under small samples (a hyperparameter sweep confirms d=64 is always worse).

### 3.1 z_fin — finance TCN encoder

Input `(B, L, 31)` → output `z_fin (B, 32)`.

- **Causal TCN** (`TemporalTCN`): 2 layers of 1-D conv, kernel=3, residual connections, dropout;
- **Literature support**: causal TCN uses dilated convolutions to widen the receptive field while looking only at the past window — suitable for weekly finance series (Bai et al., 2018); Foroutan & Lahmiri (2024) [P001] also report strong TCN performance in crude-oil / precious-metal forecasting, so the finance branch uses a light TCN rather than an RNN.



### 3.2 z_rs — remote-sensing Prithvi-attention encoder

Input `rs (B, L, 11, 1024)` + `mask (B, L, 11)` → output `z_rs (B, 32)` + site attention `(B, 11)`. **The Prithvi backbone is frozen**; only light attention is learned here:

- linear projection 1024→64 + LayerNorm + dropout;
- **Model**: `ibm-nasa-geospatial/Prithvi-EO-2.0-300M` (IBM–NASA remote-sensing foundation model). Weights `Prithvi_EO_V2_300M.pt` from HuggingFace.
- **Why it fits**: exported 6 bands (B2/B3/B4/B8A/B11/B12) **exactly equal** Prithvi's expected HLS bands — same physical bands, same order — no remapping needed.
- **Frozen**: `model.eval()` + `requires_grad_(False)` + `torch.no_grad()` — **not a single backbone parameter is trained**.
- **Temporal attention** (per site over lookback): learnable query scores time steps, masked by availability → softmax weighting → one vector per site;
- **Site attention** (over 11 AOIs): learnable query scores sites → softmax → weighted sum → 32-d. Site-attention weights feed RQ3's "which site the RS branch weights".
- **Literature support**: image representations come from frozen Prithvi-EO-2.0 [P094] (EO foundation model; end-to-end ViT training is infeasible under small samples); temporal/site attention handles **irregular, missing** monthly observations, in the spirit of mTAN [P099] for irregular time series.



### 3.3 z_ship — shipping graph GAT + TCN encoder

Input `aoi (B,L,11,11)`, `choke (B,L,6,20)`, `adj (B,L,17,17)` → `z_ship (B,32)` + node attention `(B,17)`.

- **17-node heterogeneous graph** (`build_m3_graph17.py`): 11 AOI oil sites (P001–P011) + 6 chokepoints, fixed order (AOI 0–10, chokepoints 11–16).
  - **AOI node features (11 per node)**: `pw_portcalls_tanker/cargo`, `pw_import_tanker`, `pw_export_tanker`, `gfw_n_visits`, `gfw_dwell_hrs_mean/median`, `gfw_self_loops`, `sar_detections_total/dark`, `sar_dark_share`.
  - **Chokepoint node features (20 per node)**: GFW 8 + PortWatch 9 + SAR 3.
  - **Edges (adjacency, one 17×17 per week)**: dynamic O-D edges (AOI→AOI from GFW `n_voyages`); static AOI↔chokepoint edges (fixed by geography).
- **Heterogeneous input**: AOI/chokepoint each linearly projected to d_model=64 + node-type embedding (0=AOI, 1=chokepoint) + LayerNorm;
- **Spatial GAT** (`DenseGATLayer`): 2 layers of multi-head (heads=4) dense graph attention (17×17 with boolean mask). Adjacency symmetrised + self-looped. **Innovation P1-4**: **log O-D flow** `log1p(flow)` **as an attention prior** (learnable gain `edge_scale`).
- **Temporal TCN**: per node over lookback via causal TCN (same adaptive dilation);
- **Node-attention pooling**: additive attention over 17 nodes → softmax → 32-d graph representation. Node weights feed RQ3's "which port/chokepoint is attended".
- **Literature support**: spatial layer uses GAT for message passing on the port–chokepoint graph (Veličković et al., 2018); crude-oil shipping already has graph-conv/LSTM flow-forecasting precedents [P062]; spatio-temporal graph modelling see Graph WaveNet [P091]; temporal layer still uses causal TCN over weekly dynamic graphs.

---



## 4. Fusion module

Fuses several 32-d modality representations into one 32-d `z_fused`, then a regression head → r̂.

### 4.1 ConcatFusion (encoder-concat, ladder floor)

Concatenate each modality's 32-d representation into `n_mod × 32` **dims** (64-d for two modalities, 96-d for three), then a fixed MLP `Linear(n_mod·32 → 32) + ReLU` projects back to 32-d. Modalities are still encoded separately, but the fusion layer has **no per-sample gating, no cross-attention** — the RQ2 fusion-ladder floor (Baltrušaitis et al., 2019 [P101]; concat baseline in GMU [P096]).

### 4.2 GatedFusion (gated, main model ✅)

`alpha = softmax(MLP([z_1..z_m]))`; `z = Σ alpha_i · z_i`. A small MLP computes per-modality weights (convex combination) from the current sample — **potentially different every sample, every week**. `alpha` is RQ3's "modality gate weights" (does this week trust finance or shipping more).

### 4.3 CrossModalAttentionFusion (cross-attention, advanced)

Finance `z_fin` acts as **Query** attending over RS/shipping **node tokens** (RS 11 sites + shipping 17 nodes = 28 tokens): `z = LN(z_fin + gamma·CrossAttn(z_fin, tokens))`, 4 heads. Cross-attention weights feed RQ3's "which node/lane the financial state attends to".

---



## 5. Backtest protocol

lookback=**4**, d=**32**, gat_layers=**2**, tcn_layers=**2**, dropout=0.1, seed=42. Rolling-origin walk-forward: `min_train=104`, refit every 13 weeks, strictly leak-free.

---



## 6. Main results



### 6.1 Fusion matrix


| Combo       | Fusion         | RMSE      | skill vs M0 | DirAcc | CW vs M0    |
| ----------- | -------------- | --------- | ----------- | ------ | ----------- |
| M0          | –              | 4.152     | 0.0%        | –      | –           |
| M1 deep     | –              | 4.250     | −2.4%       | 0.494  | 0.315       |
| Mfinship    | Concat         | 4.149     | +0.06%      | 0.525  | 0.195       |
| Mfinship    | Gated          | 4.147     | +0.11%      | 0.529  | 0.166       |
| **M3 deep** | **Cross-Attn** | **4.121** | **+0.74%**  | 0.549  | **0.041** ✅ |
| Mfinrs      | Concat         | 4.232     | −1.93%      | 0.533  | 0.971       |
| Mfinrs      | Gated          | 4.253     | −2.4%       | 0.475  | 0.769       |
| M2 deep     | Cross-Attn     | 4.396     | −5.89%      | 0.455  | 0.898       |
| Mfull       | Concat         | 4.320     | −4.1%       | 0.494  | 0.637       |
| Mfull       | Gated          | 4.205     | −1.3%       | 0.502  | 0.894       |
| **M4 deep** | **Cross-Attn** | **4.147** | **+0.12%**  | 0.564  | **0.018** ✅ |


**New findings from the matrix**:

1. **Finance+shipping (Mfinship) is skill>0 under all three fusions** — the strongest combo; **Cross-Attention is the overall best** (+0.74%, RMSE 4.121, CWvsM0 **0.041** significantly beating the random walk) — even better than three-modality Mfull, showing RS is a net drag.
2. **Finance+RS (Mfinrs) is skill<0 under all three fusions**, with Cross-Attention the worst (−5.89%) — RS noise is amplified by cross-attention.
3. **Gated vs Concat**: gating clearly beats concat for three modalities (−1.28% vs −4.06%); for two modalities they are close (both carried by the shipping signal).

---



## 7. RQ2 answer

**RQ2 asks**: on the same data, does **representation-level fusion** beat **flat concatenation**? The four paired comparisons below hold **modality content fixed** and vary only the fusion paradigm (left = deep representation-level / right = flat wide table):


| Pair         | Modality content (same) | Difference (RQ2 core)                               |
| ------------ | ----------------------- | --------------------------------------------------- |
| M1deep vs M1 | finance only            | TCN encoder vs wide-table Ridge/XGB                 |
| M2deep vs M2 | finance + RS            | gated fusion vs column concat                       |
| M3deep vs M3 | finance + shipping      | GAT+TCN+gated vs 113-column concat                  |
| M4deep vs M4 | all modalities          | 3 encoders+gated vs full-modality wide-table concat |



| Pair         | Flat RMSE | Deep RMSE | skill flat | skill deep | DM_p (XGB) | DM_p (Ridge) |
| ------------ | --------- | --------- | ---------- | ---------- | ---------- | ------------ |
| M1deep vs M1 | 4.368     | 4.250     | −5.2%      | −2.4%      | 0.097      | 0.466        |
| M2deep vs M2 | 4.440     | 4.253     | −7.0%      | −2.4%      | 0.042 ✅    | 0.096        |
| M3deep vs M3 | 4.429     | 4.147     | −6.7%      | +0.11%     | 0.010 ✅    | 0.062        |
| M4deep vs M4 | 4.507     | 4.205     | −8.6%      | −1.3%      | 0.005 ✅    | 0.036 ✅      |


**How to read it**:

1. **M3 and M4 pairs**: representation-level fusion **significantly** beats flat (XGB side p<0.05; M4 significant on the Ridge side too).
2. **M2 pair**: with RS added, deep also **significantly** beats flat (XGB p=0.042).
3. **M1 pair**: finance-only, deep is better but DM does not reach 5% (p=0.097) — swapping the encoder alone is not enough; alternative data + representation-level fusion is what opens the gap.
4. This is much stronger than the earlier "deep vs flat M1, all DM >0.05" because the comparison is **like-for-like**, not "three-modality deep" vs "finance-only flat".

---



## 8. RQ3 answer

RQ3 asks what the model "depends on" — but it is really **two layers of questions** that need **two different kinds of evidence**; do not mix them:


| RQ3 sub-question                     | What can answer it                                                                                                       | What cannot answer it           |
| ------------------------------------ | ------------------------------------------------------------------------------------------------------------------------ | ------------------------------- |
| **Which modality?** (time-varying)   | **Modality gate** α_fin / α_rs / α_ship time series from M4deep (m4rep)                                                  | node attention, flat SHAP       |
| **Which port/chokepoint?** (spatial) | **Node attention** inside the shipping encoder (ship_site_att, 17 nodes); **site attention** in the RS encoder (11 AOIs) | α_shipping in the modality gate |


> ⚠️ **Do not mix the two evidence layers**: a high α_shipping **only** means "the **shipping branch weighs heavily** in this week's fused representation" — it does **not** mean "the model is looking at Hormuz vs Suez". Port/chokepoint information lives in **GAT → node-attention pooling**, **not** in the fusion gate.

