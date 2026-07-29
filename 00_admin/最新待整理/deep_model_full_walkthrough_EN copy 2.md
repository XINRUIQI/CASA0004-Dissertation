# Deep Model (Innovation / Method-Integration Layer) — Full Walkthrough

> **Purpose**: for a reader **completely new to this project**, document **every single step** of the deep (representation-level) model — from downloading data, cloud removal, data cleaning, satellite representation learning, shipping-graph construction, three-encoder fusion, parameter choices, and the walk-forward backtest, through to results, statistical tests, interpretability, and robustness.
>
> **In one line**: the deep model is the **method-integration / empirical-test layer (the contribution)** of the dissertation. Instead of engineering remote sensing / shipping into more numeric columns and stacking them into one wide table (that is the *flat baseline*, see `flat_baseline_full_walkthrough_EN.md`), it uses **three modality-specific encoders** to learn separate **representations (embeddings)** of "financial state / remote-sensing activity / shipping network", then **dynamically fuses** them via **gating / cross-attention** to predict next-week Brent price end-to-end.
>
> **The core question it answers (RQ2)**: on the same data, does **modality-aware representation-level fusion** beat **flat feature fusion** (flatten all columns into one table)?
>
> **Last updated**: 2026-07-28 (adds RQ3 multi-seed stability, Flat–Deep pairing table, path & lag clarifications; main-result numbers remain the 2026-07-07 lookback=4 protocol)
>
> **Code**: `04_code/src/models/` (encoders + fusion + backtest kernel), `04_code/scripts/deep/run_deep_*.py`; results: `05_outputs/baselines/Deep/` (`M*_Deep/` + `_cross/`)  
> **Progress overview**: `00_admin/最新待整理/2026-07-15_研究方案与进度总览.md` §2.4 / §2.4.1 / §2.6  
> **Flat counterpart**: `flat_baseline_full_walkthrough_EN.md` · `flat_baseline_log.md`  
> **Project logic**: `项目逻辑与结果总览_CN.md`

---

## Table of contents

1. [First: deep model vs flat model](#1)
2. [Step 0: raw data download (three modalities)](#2)
3. [Step 1: remote-sensing cloud removal + monthly compositing](#3)
4. [Step 2: Prithvi satellite-representation precompute (frozen foundation model)](#4)
5. [Step 3: 17-node dynamic heterogeneous shipping graph](#5)
6. [Step 4: finance series preparation](#6)
7. [Step 5: deep-dataset alignment (leak-free, sample size)](#7)
8. [Step 6: the three modality encoders (architecture + parameters)](#8)
9. [Step 7: fusion module (gated / concat / cross-attention)](#9)
10. [Step 8: backtest protocol (how it is trained)](#10)
11. [Step 9: main results](#11)
12. [Step 10: how to analyse the results (statistical tests)](#12)
13. [Step 11: robustness](#13)
14. [Step 12: interpretability (RQ3)](#14)
15. [Step 13: advanced ablations](#15)
16. [One-line summary + reproduction commands](#16)
17. [Appendix: full variable & parameter list](#17)

---

<a name="1"></a>
## 1. First: deep model vs flat model

**Prediction target** (identical to the flat model, so the comparison is fair): next-week Brent spot price \(P_{t+1}\) (USD/barrel, Friday close). The model does not predict price directly; it predicts the **log return** \(r_{t+1}=\ln(P_{t+1}/P_t)\) and reconstructs the price \(\hat P_{t+1}=P_t\cdot e^{\hat r}\). Log returns are used because price is non-stationary (trending) while returns are more stationary and easier to model.

**The two fusion paradigms (the dissertation's selling point)**:

| | Flat fusion (baseline) | Representation-level fusion (this deep model) |
|---|---|---|
| Handling | flatten all columns of 3 modalities into one wide table (**199 feature cols** modelled; matrix also has mask/target) | each modality goes through its **own encoder** to a 32-d representation |
| Model | Ridge / XGBoost / LSTM early-fusion | 3 encoders + gated / cross-attention fusion |
| Modality structure | lost (no structure across columns) | preserved (temporal, graph, AOI-site structure live inside encoders) |
| Missing modality | forced in via fill values | modelled explicitly via encoder + mask + modality dropout |

**Contribution type**: not a new fusion operator / network layer / loss, but an **integration** of existing methods (frozen EO foundation model + modality-specific encoders + gated / cross-attention + missing-modality modelling) into one coherent system, and — for the **first time** in weekly crude-oil forecasting — a systematic comparison of "representation-level vs flat fusion" under a unified leak-free protocol with DM / Clark–West tests.

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

<a name="2"></a>
## 2. Step 0: raw data download (three modalities)

Each modality has its own sources, all stored under `03_data/raw/` (raw layer; the whole `raw/` is git-ignored). **The deep and flat models share the exact same raw data**; they differ only in *how* it is used downstream.

### 2.1 M1 finance / macro (`raw/01_market_financial/`)

| Source | Content | Frequency |
|---|---|---|
| EIA (US Energy Info Admin) | Brent/WTI spot, WPSR weekly report (stocks/production/imports-exports/refinery input) | daily/weekly |
| FRED (Federal Reserve) | VIX, dollar index, 10-yr Treasury, fed funds rate, industrial commodity index | daily/monthly |
| Yahoo Finance | S&P500, oil volatility OVX, Brent futures, CAD FX, gold | daily |
| Academic | Kilian global activity index, Caldara–Iacoviello Geopolitical Risk (GPR) | monthly/daily |

### 2.2 M2 remote sensing (`raw/02_sentinel2/`)

**Two channels** (the deep model mainly uses Channel A; the flat model uses Channel B):

- **Channel A** (`Channel A/s2_patches/`): **monthly Sentinel-2 6-band image patches** (GeoTIFF) for the 11 oil sites, exported from Google Earth Engine. Bands B2/B3/B4/B8A/B11/B12 exactly match the 6 HLS bands Prithvi expects. **This is the input for the deep model's z_rs.**
- **Channel B** (`Channel B/*.csv`): **monthly optical-index CSVs** precomputed from imagery (NDVI/NDWI/NDBI/BSI from 2017-04; VIIRS night-time lights NTL from 2014-01). Used by the flat model.

**The 11 AOI oil sites** (P001–P011): Rotterdam, Fujairah, RasTanura, Jurong, Houston, NingboZhoushan, Jamnagar, Basra, Ulsan, Kharg, Yanbu — typed as port / refinery / terminal.

### 2.3 M3 shipping (`raw/03_shipping/`)

| Source | Content | Frequency | Since |
|---|---|---|---|
| IMF PortWatch | 6 chokepoints transit tanker/cargo counts, capacity; port import/export tanker volume | daily | 2019+ |
| GFW (Global Fishing Watch) | chokepoint AIS vessel-presence hours; AOI port-visit events + dwell; AOI↔AOI voyage O-D; SAR dark-vessel detections | monthly/event | 2012+ |

**The 6 chokepoints**: hormuz, suez, malacca, mandeb, panama, cape.

---

<a name="3"></a>
## 3. Step 1: remote-sensing cloud removal + monthly compositing

The biggest enemy of satellite imagery is **cloud**. This step is done in Google Earth Engine (GEE), producing "one clean image per site per month".

### 3.1 Cloud masking + monthly median composite (GEE side)

- **Source**: `COPERNICUS/S2_SR_HARMONIZED` (Sentinel-2 surface reflectance, cross-sensor harmonised).
- **Cloud logic**: for all scenes in a month, mask out cloud / cloud-shadow / snow pixels using the cloud-probability / SCL scene-classification band, then take a **median composite** of the remaining clean pixels to produce one representative monthly image. The median further suppresses residual clouds and outliers.
- **Export**: AOI-differentiated patch sizes (port 6.4 km / refinery 5.12 km / terminal 2.56 km), saved as 6-band GeoTIFF (Channel A); monthly NDVI/NDWI/NDBI/BSI indices are also computed and exported as CSV (Channel B).

### 3.2 Patch-quality validation (`s2_patch_utils.py`)

Not every exported patch is usable; two gates filter them:

1. **Manual exclusion list** `s2_patch_exclusions.csv`: obviously bad patches (heavy residual cloud, misclip) are listed by hand.
2. **Valid-pixel fraction** `has_valid_pixels()`: a patch must be 6-band and have **≥ 0.5% non-zero pixels** (`MIN_NONZERO_FRAC=0.005`), else it is treated as an empty export (GEE sometimes exports all-zeros).

Only patches with `valid_mask==1` (file exists + not excluded + pixels valid) proceed to representation computation. **About 967 usable patches** in total (across 11 sites, from 2017-04).

---

<a name="4"></a>
## 4. Step 2: Prithvi satellite-representation precompute (frozen foundation model)

This is the **first key innovation** distinguishing the deep model from the flat one: instead of hand-computing 5 indices from imagery, a **remote-sensing foundation model** encodes each whole image into a 1024-d vector (embedding), letting the model "look at the picture" itself.

Script: `03_data/processed/M2/py/precompute_s2_embeddings.py`

### 4.1 Which model + why frozen

- **Model**: `ibm-nasa-geospatial/Prithvi-EO-2.0-300M` (an IBM–NASA remote-sensing foundation model, ~**300M parameters**, ViT architecture, pretrained on massive HLS satellite imagery). Weights `Prithvi_EO_V2_300M.pt` downloaded from HuggingFace.
- **Why it fits perfectly**: the exported 6 bands (B2/B3/B4/B8A/B11/B12) **exactly equal** Prithvi's expected 6 HLS bands (Blue/Green/Red/NIR-narrow/SWIR1/SWIR2) — same physical bands, same order — so no band remapping is needed; the model's own mean/std normalisation is reused.
- **Frozen**: `model.eval()` + `requires_grad_(False)` + `torch.no_grad()` — **not a single backbone parameter is trained**. This is the key anti-overfitting measure under small samples: 365 weeks cannot possibly train 300M parameters. Only a light attention head is trained downstream (Step 6).

### 4.2 Pipeline per patch → 1024-d vector

For each (site, month) patch (single frame, T=1):

1. read the 6-band GeoTIFF → `float32`;
2. normalise `(x-mean)/std` (using Prithvi config's mean/std); map 0-values (nodata / patch edge) to Prithvi's convention constant `1e-4`;
3. bilinear resize to **224×224** (done on CPU, to avoid an MPS anti-alias glitch); after resizing, the differentiated patch sizes land near Prithvi's ~30 m training GSD;
4. run Prithvi `forward_features` (**no MAE masking**) → take the last block (post-norm) output `[B, 197, 1024]`;
5. **two poolings**: mean-pool over the 196 patch tokens → **meanpool vector [1024]** (main); also keep the cls token (alternative).

### 4.3 Outputs

- `s2_prithvi_emb_meanpool.npy` `[N, 1024]`: the main representation (used by the deep model).
- `s2_prithvi_emb_cls.npy` `[N, 1024]`: cls-token alternative (robustness comparison; empirically worse).
- `s2_prithvi_emb_index.csv`: N rows aligned to `.npy` row order + metadata (site_id, month, obs_month_start, cloud, n_scenes, …). Missing (site, month) cells stay **explicitly missing** (no ffill) for later as-of alignment.

> Note: the overview mentions `[963, 1024]` — i.e. ~963 successfully embedded patches.

---

<a name="5"></a>
## 5. Step 3: 17-node dynamic heterogeneous shipping graph

The **second key innovation**: instead of flattening shipping into numeric columns, it is built explicitly as a **graph** — nodes are ports and chokepoints, edges are actual ship voyages (O-D flow), and the graph **changes every week**.

### 5.1 Cleaning and lagging (`aggregate_shipping_to_weekly.py` + `build_m3_graph17.py`)

- **Unify to W-FRI (Friday-ending weeks)**: PortWatch daily → weekly sum; GFW monthly ffill → weekly; using a **union index** (not intersection) so neither GFW-early (2012+) nor PortWatch-late samples are lost (fixes an old 727→362 sample-drop bug).
- **Publication lags (key to leak-free; do not mix flat arm vs deep graph)**:


| Stream | Script | Lag | Used for |
| --- | --- | --- | --- |
| GFW **monthly presence** (`gfw_*` in flat 113 cols) | `aggregate_shipping_to_weekly.py` | **+4 weeks** | **Flat M3** (this deep walkthrough does not feed those 113 cols directly) |
| PortWatch | same | **+1 week** | Flat; also used for deep-graph chokepoint nodes |
| GFW **event / port-visit / voyage / O-D edges** | `build_m3_graph17.py` | **+2 weeks** | **Deep 17-node graph** (near-real-time AIS, conservative 2 weeks) |
| GFW SAR dark vessels (monthly) | `build_m3_graph17.py` | **+4 weeks** | Deep graph node features (not in flat main model) |

- Derived quantities (wow_pct/4w_ma) are computed *before* the lag, then the whole block is shifted forward.
- **Data-quality clips**: port-visit dwell capped at 720 h (30 days; longer = AIS long-stay / stitching artefact → NaN); voyage transit capped at 90 days (longer = an unobserved intermediate call → mean set NaN, but the edge still counts).

### 5.2 Assembling the 17-node heterogeneous graph (`build_m3_graph17.py`)

**Nodes (17)** = 11 AOI oil sites (P001–P011) + 6 chokepoints, with fixed order (AOI 0–10, chokepoints 11–16).

**Why "heterogeneous"**: AOI and chokepoint nodes live in **different feature spaces** (F_aoi ≠ F_choke), so the encoder uses **node-type-specific projections** then shared message passing.

- **AOI node features (11 per node)**: `pw_portcalls_tanker/cargo`, `pw_import_tanker`, `pw_export_tanker`, `gfw_n_visits`, `gfw_dwell_hrs_mean/median`, `gfw_self_loops`, `sar_detections_total/dark`, `sar_dark_share`.
- **Chokepoint node features (20 per node)**: GFW 8 (total_hours, total_vessels, cargo/bunker/other_hours, other_share, total_hours_mom_pct, mean_presence) + PortWatch 9 (n_tanker, n_total, capacity, tanker_share, …) + SAR 3.

**Edges (adjacency, one 17×17 per week)**:
- **Dynamic O-D edges** (AOI→AOI): from GFW voyage counts `n_voyages`, **different each week**, reflecting real ship flow.
- **Static AOI↔chokepoint edges**: fixed by geographic association (e.g. hormuz ↔ P002/P003/P008/P010; malacca ↔ P004/P006/P009), present every week.

**Output**: `m3_graph17_tensors.npz` with `aoi_features (T,11,11)`, `choke_features (T,6,20)`, `adjacency (T,17,17)`, where T is the number of weeks.

---

<a name="6"></a>
## 6. Step 4: finance series preparation

The finance modality uses **no graph, no imagery** — just M1's 31 weekly columns (Brent/WTI, EIA stocks/production/imports-exports/refinery, VIX, DXY, rates, S&P500, futures basis, GPR, …).

- Take M1 columns from the merged matrix `weekly_feature_matrix.csv` (`data.select_features(dico, "M1")`).
- **Past-only fill** (`fill_features`): ffill (past-only, no future) + leading residual NaN → 0 (neutral). This yields `fin_df` covering all W-FRI (levels; no differencing — the encoder has LayerNorm + past-only scaling internally).

---

<a name="7"></a>
## 7. Step 5: deep-dataset alignment (leak-free, sample size)

Script: `04_code/src/models/deep_dataset.py` → `build_deep_dataset()`. This aligns the three modalities to the **same weekly target/index** the flat baseline uses (`build_dataset`), which is what makes the deep-vs-flat comparison fair.

### 7.1 Target and index

- Use the flat builder (lookback=1) to get `idx` (usable weeks), target `r_next` (=r_{t+1}), `P_t`, `P_next`, `r_now`. Window **2019-01-01 ~ 2025-12-31**.

### 7.2 Sliding window (lookback)

For each usable date d, build a window over the past `lookback` weeks. **Main model lookback = 4 weeks** (supervisor-set + aligns the flat protocol). Each sample contains:
- `aoi (L, 11, 11)`, `choke (L, 6, 20)`, `adj (L, 17, 17)`: shipping-graph window;
- `fin (L, 31)`: finance-series window;
- `rs (L, 11, 1024)`, `rs_mask (L, 11)`: RS embedding window + availability mask.

Only dates whose **graph window and finance window are both complete** are kept → N aligned samples.

### 7.3 Leak-free as-of alignment + within-site demeaning for RS

- **As-of alignment**: a monthly Prithvi embedding's **availability = month-end + 15 days** (`RS_PUB_LAG_DAYS=15`, a conservative release delay). Each week uses `merge_asof(backward)` to pick the **most-recent already-published** monthly embedding — never the future. Missing → NaN + mask=0.
- **Within-site past-only demeaning** (`meanpool_anom`, optional): per site, in chronological order, expanding (inclusive, no future) demeaning `a_j = e_j - mean(e_1..e_j)`. This strips the static "which-site" scene signature (~80% of frozen-embedding variance), leaving the RS branch to see only the **temporal anomaly**. The main results use raw meanpool.

### 7.4 Past-only standardisation (refit per fold)

**Standardisation is not done here** — it must be fit **on the training slice only** per backtest fold (`fit_scalers`), to avoid leakage:
- aoi/choke/fin/rs are z-scored on the feature axis (nan-aware, std floored);
- adj is not standardised (only its >0 connectivity pattern is used as an attention mask);
- the target `r_next` is also per-fold standardised (`(r-mean)/std`) and reconstructed after prediction.

### 7.5 Sample size

- Merged matrix CSV: **365 weeks × 213 cols** = `week_ending_friday` + **212 data cols** (31 M1 + 55 M2 anom + 113 M3 + 11 mask + 2 target). Deep does not feed anom/shipping wide-table columns directly, but shares the same weekly index and target as flat.
- Aligned deep samples N depend on lookback; backtest `min_train=104` (first 104 weeks warm-up, not tested); **common test span = 257 weeks (2021–2025)**.
- ⚠️ **Historical note**: 2026-07-05 first representation-level run used lookback=**8** (see `flat_baseline_log.md` §13); **the official main protocol locks lookback=4 from 2026-07-07** to align with flat L4. Numbers below are all lb=4.

---

<a name="8"></a>
## 8. Step 6: the three modality encoders (architecture + parameters)

Code: `04_code/src/models/{finance,rs,shipping}_encoder.py`. All output a **32-d (d=32)** representation. Why a small 32-d? High dimensions overfit under small samples (a hyperparameter sweep confirms d=64 is always worse).

### 8.1 z_fin — finance TCN encoder (`finance_encoder.py`)

Input `(B, L, 31)` → output `z_fin (B, 32)`.
- linear projection 31→32 + LayerNorm;
- **causal TCN** (`TemporalTCN`): 2 layers of 1-D conv, kernel=3, residual connections, dropout;
- **adaptive dilation**: lookback ≤ 5 (e.g. main model's 4) uses **dense conv (dilation=1)**; lookback ≥ 6 uses **exponential dilation (1,2,4…)** to widen the receptive field. This is sweep-evidence-driven — short windows prefer dense sampling, long windows need dilation.
- take the last time step → linear head + ReLU → 32-d.

### 8.2 z_rs — remote-sensing Prithvi-attention encoder (`rs_encoder.py`)

Input `rs (B, L, 11, 1024)` + `mask (B, L, 11)` → output `z_rs (B, 32)` + site attention `(B, 11)`. **The Prithvi backbone is already frozen**; only light attention is learned here:
- linear projection 1024→64 + LayerNorm + dropout;
- **temporal attention** (per site over lookback): a learnable query scores time steps, masked by availability → softmax weighting → one vector per site;
- **site attention** (over the 11 AOIs): a learnable query scores sites (masked by "any-valid-in-window") → softmax → weighted sum → 32-d. Site-attention weights feed RQ3's "which site the RS branch weights".

### 8.3 z_ship — shipping graph GAT + TCN encoder (`shipping_encoder.py`)

Input `aoi (B,L,11,11)`, `choke (B,L,6,20)`, `adj (B,L,17,17)` → `z_ship (B,32)` + node attention `(B,17)`. ~**42k parameters**.
- **heterogeneous input**: AOI/chokepoint each linearly projected to d_model=64 + a node-type embedding (0=AOI, 1=chokepoint) + LayerNorm;
- **spatial GAT** (`DenseGATLayer`): 2 layers of multi-head (heads=4) dense graph attention (17×17 with a boolean mask — dense is simpler than sparse torch_geometric for such a tiny, dynamic graph). Adjacency symmetrised + self-looped. **Innovation P1-4**: the **log O-D flow `log1p(flow)` is used as an attention prior** (learnable gain `edge_scale`) — busy lanes naturally get higher attention instead of edge weights being discarded by the boolean adjacency. This strengthens the shipping increment.
- **temporal TCN**: per node over lookback via causal TCN (same adaptive dilation);
- **node-attention pooling**: additive attention scores the 17 nodes → softmax → 32-d graph representation. Node weights feed RQ3's "which port/chokepoint is attended".

---

<a name="9"></a>
## 9. Step 7: fusion module (gated / concat / cross-attention)

Code: `04_code/src/models/fusion.py` → `DeepForecastModel`. Fuses several 32-d modality representations into one 32-d `z_fused`, then a regression head → r̂. The three fusions form the RQ2 "fusion ladder":

### 9.1 GatedFusion (gated, main model ✅)
`alpha = softmax(MLP([z_1..z_m]))`; `z = Σ alpha_i · z_i`. A small MLP computes per-modality weights (a convex combination) from the current sample — **potentially different every sample, every week**. `alpha` is exactly RQ3's "modality gate weights" (does this week trust finance or shipping more).

### 9.2 ConcatFusion (encoder-concat, ladder floor)
`z = ReLU(Linear([z_1..z_m]))`. Modalities are still encoded separately, but mixed by a **fixed MLP** with no per-sample gating, no cross-attention. It is the control: any gain of gating / cross-attention over this arm is attributable to the *fusion mechanism* itself, not merely to per-modality encoding.

### 9.3 CrossModalAttentionFusion (cross-attention, advanced)
Finance `z_fin` acts as **Query** attending over RS/shipping **node tokens** (RS 11 sites + shipping 17 nodes = 28 tokens): `z = LN(z_fin + gamma·CrossAttn(z_fin, tokens))`, 4 heads. Cross-attention weights feed RQ3's "which node/lane the financial state attends to".

### 9.4 Regression head + modality dropout
- **Head**: `Linear(32,32)→ReLU→Dropout→Linear(32,1)` → scalar r̂.
- **Modality dropout** (optional, ModDrop-style): during training randomly drop a whole modality with some probability (keep ≥ 1), improving missing-modality robustness.

**Config table** (`deep_rolling.CONFIGS`): `fin` (finance only), `ship` (shipping only), `rs` (RS only), `fusion` (finance+shipping, gated), `m2_deep_gated` (finance+RS, gated), `m4_deep_gated` (three modalities, gated), `m4_deep_xattn` (three, cross-attention), `m4_deep_concat` (three, concat).

---

<a name="10"></a>
## 10. Step 8: backtest protocol (how it is trained)

Code: `04_code/src/models/deep_rolling.py` → `rolling_origin_deep()`. **Aligned verbatim with the flat baseline** to keep it fair.

### 10.1 Rolling-origin (walk-forward)
- **Expanding window**: `min_train=104` (first 104 weeks train-only, not tested), then test week-by-week forward;
- **Refit every 13 weeks** (`retrain_every=13`): each fold trains only on samples **before** that test week (`slice(0, i)`), strictly no future;
- each test week is predicted by its own fold model, and price reconstructed `P̂ = P_t·exp(r̂)`.

### 10.2 Single-fold training details (`_train_fold`)
- within the training slice, the **last 52 weeks are held out as inner-validation** for early stopping (patience=12);
- optimiser **Adam**, lr=1e-3, weight_decay=1e-4, batch=32, epochs=80 (cap; early stopping usually triggers first);
- loss **MSE** (on standardised r); gradient clipping max_norm=5.0;
- target per-fold standardised; predictions reconstructed via `r̂·r_std + r_mean`.

### 10.3 Key hyperparameters (main model)
lookback=**4**, d=**32**, gat_layers=**2**, tcn_layers=**2**, dropout=0.1, seed=42, fusion=**gated**.

> ⚠️ Engineering caveat: on macOS, xgboost and torch in the same process segfault due to duplicate OpenMP. So the deep scripts **read** the M1_Flat predictions (`baseline_predictions.csv`) rather than re-running xgb in-process.

---

<a name="11"></a>
## 11. Step 9: main results

Source: `04_code/scripts/deep/run_deep_baseline.py` → `05_outputs/baselines/Deep/_cross/deep_metrics.csv` (257 common test weeks 2021–2025, seed=42, lookback=4).

| Model | RMSE | skill vs M0 | DirAcc | CW vs M0 | DM vs M1 |
|---|--:|--:|--:|--:|--:|
| M0 random walk | 4.152 | 0.0% | – | – | – |
| M1_Flat_Ridge / XGB (flat ref) | 4.256 / 4.368 | −2.5% / −5.2% | 0.498 / 0.553 | 0.530 / 0.104 | — |
| M1_Deep (z_fin TCN) | 4.250 | −2.4% | 0.494 | 0.315 | — |
| M_rs_deep (z_rs Prithvi) | 4.247 | −2.3% | 0.459 | 0.928 | 0.457 |
| M_ship_GNN (z_ship graph) | 4.168 | −0.4% | 0.506 | 0.496 | 0.106 |
| **M3_Deep_gated (finance+shipping, gated)** | **4.147** | **+0.11%** | 0.529 | 0.166 | 0.061 |
| M2_Deep_gated (finance+RS) | 4.253 | −2.4% | 0.475 | 0.769 | 0.485 |
| M4_Deep_gated / M4rep (three, gated) | 4.205 | −1.3% | 0.502 | 0.894 | 0.239 |
| M4_Deep_Concat (three, concat) | 4.320 | −4.1% | 0.494 | 0.637 | 0.650 |

**RMSE = root mean squared error on price (lower is better); skill = improvement over the random walk (>0 means beating RW).**

### 11.1 Fusion matrix (RQ2 fusion ladder: 3 modality combos × 3 fusion mechanisms)

Source: `run_deep_fusion_matrix.py` → `deep_fusion_matrix.csv` / `deep_fusion_matrix.png` (257 weeks, seed=42, lookback=4, epochs=80, all 9 cells run once under the same protocol). This fills the full "each fusion model × three fusion mechanisms" matrix (the three mechanisms the project implements: Encoder-Concat / Gated / Cross-Attention).

**skill vs M0 (%), >0 beats the random walk**:

| Modality combo | Encoder-Concat | Gated | Cross-Attention |
|---|--:|--:|--:|
| **M3_Deep_gated** (finance+shipping) | +0.06 | +0.11 | **+0.74** |
| **M2_Deep_gated** (finance+RS) | −1.93 | −2.43 | −5.89 |
| **M4_Deep_gated** (three modalities) | −4.06 | −1.28 | +0.12 |

**Full per-cell metrics**:

| Combo | Fusion | RMSE | skill vs M0 | DirAcc | CW vs M0 | DM vs M1 |
|---|---|--:|--:|--:|--:|--:|
| M3_Deep_gated | Concat | 4.149 | +0.06% | 0.525 | 0.195 | **0.043** ✅ |
| M3_Deep_gated | Gated | 4.147 | +0.11% | 0.529 | 0.166 | 0.061 |
| **M3_Deep_gated** | **Cross-Attn** | **4.121** | **+0.74%** | 0.549 | **0.041** ✅ | 0.055 |
| M2_Deep_gated | Concat | 4.232 | −1.93% | 0.533 | 0.971 | 0.373 |
| M2_Deep_gated | Gated | 4.253 | −2.4% | 0.475 | 0.769 | 0.485 |
| M2_Deep_gated | Cross-Attn | 4.396 | −5.89% | 0.455 | 0.898 | 0.913 |
| M4_Deep_gated | Concat | 4.320 | −4.1% | 0.494 | 0.637 | 0.650 |
| M4_Deep_gated | Gated | 4.205 | −1.3% | 0.502 | 0.894 | 0.239 |
| **M4_Deep_gated** | **Cross-Attn** | **4.147** | **+0.12%** | 0.564 | **0.018** ✅ | 0.088 |

**New findings from the matrix**:

1. **Finance+shipping (M3_Deep_gated) is skill>0 under all three fusions** — the strongest modality combo; its **Cross-Attention is the overall best** (+0.74%, RMSE 4.121, CWvsM0 **0.041**, significantly beating the random walk) — even better than the three-modality M4_Deep_gated, showing that adding RS is a net drag.
2. **Finance+RS (M2_Deep_gated) is skill<0 under all three fusions**, with Cross-Attention the worst (−5.89%) — RS noise is amplified by cross-attention, again confirming the RS modality is intrinsically weak.
3. **Cross-Attention only helps combos that contain shipping** (M3_Deep_gated +0.74, M4_Deep_gated +0.12, both significant CWvsM0); on finance+RS it is a disaster — the value of the fusion mechanism depends on whether the modality carries signal, not on the mechanism itself.
4. **Gated vs Concat**: gating clearly beats concat for three modalities (−1.28% vs −4.06%); for two modalities they are close (both carried by the shipping signal).
5. ⚠️ These are **single-seed (42)** results; Cross-Attention especially must be read with the multi-seed robustness — §13 shows xattn has large multi-seed variance (±2.76, seed2 collapses). So the main model stays **Gated**, with Cross-Attention listed as a "high-ceiling but unstable" advanced result.

---

<a name="12"></a>
## 12. Step 10: how to analyse the results (statistical tests)

A tiny RMSE difference means nothing without **significance testing** (`04_code/src/backtest/metrics.py`). The project is meticulous about **using the correct test**:

### 12.1 Three metrics
- **RMSE skill vs M0**: `1 - RMSE_model/RMSE_M0`, >0 means more accurate than the random walk.
- **Directional accuracy (DirAcc)**: share of correct up/down direction (auxiliary; not in the loss).
- **RMSE, MAE**: absolute price error.

### 12.2 Two tests, don't mix them up (the project's rigour point)
- **Clark–West (CW, for nested)**: when the small model is **nested** in the large one (e.g. "finance" nested in "finance+shipping"; the random walk r̂=0 is nested in any model). CW corrects the bias DM has for nested models.
  - **CW vs M0**: honestly answers "does it beat the random walk?" — M0 is nested in any model, so it is valid.
  - **Deep internal modality increments** (fusion vs fin, m4rep vs fusion, finrs vs fin) are also nested → use CW.
- **Diebold–Mariano (DM, for non-nested, HLN small-sample corrected)**: when two models are **not nested** (different model class + feature set, e.g. deep vs M1_Flat, gated vs concat). Here CW is inflated; DM is correct.

> **Protocol revision on 2026-07-07 (important)**: previously "deep vs M1_Flat" was treated as nested with CW (optimistic); switching to a strict **DM(non-nested)** **downgrades** the RQ2 claim "representation-level significantly beats flat" from a strong result to "directionally consistent but not significant". This is honest science.

### 12.3 How to read the results (`deep_cw.csv`)
- **Shipping increment (M3_Deep_gated vs M1_Deep) nested CW = 0.00057** ✅ — adding a shipping representation on top of finance **significantly** lowers MSE; this is RQ2's **cleanest positive evidence** that representation-level fusion helps.
- **RS increment (M2_Deep_gated vs M1_Deep) CW = 0.019** ✅, but **adds nothing on top of "finance+shipping"** (M4rep vs M3_Deep_gated CW=0.78 ✗, squeezed out by shipping).
- **Deep vs M1_Flat (DM) all >0.05** (M_ship_GNN 0.106 is closest): no longer significant under the strict test.
- **Gated vs concat (DM=0.22)**: gating is better but not significant; concat (M4_Deep_Concat) is the worst overall (−4.1%).

### 12.4 Six core findings
1. Only **M3_Deep_gated (finance+shipping) turns skill positive (+0.11%)**, the only deep model with skill>0; but CWvsM0=0.166 is still not significant — **none significantly beats M0** (the project's honest headline result).
2. **The shipping representation is the hardest positive evidence** (nested CW 0.00057), further strengthened by the GAT's O-D flow prior.
3. Against M1_Flat, switching to strict DM makes it **no longer significant** (a clear downgrade from the earlier optimistic protocol).
4. **Gating > naive concat** (the architecture ablation is directionally consistent).
5. **The RS representation is intrinsically weak**: M_rs_deep −2.3%, DirAcc 0.459 (<0.5), cls even worse (−11%), adds nothing on top of fin+ship — a frozen single-modality Prithvi helps little for **weekly** oil price (consistent with flat M2).
6. All of the above reinforce RQ2: weak increments need modality-aware fusion to add up in a consistent direction.

### 12.5 Flat vs Deep pairing (RQ2 writing main table)

**Fix the information set (modality content), change only the architecture** (deep: M1→`fin`, M2→`m2_deep_gated`, M3→`m3_deep_gated`, M4→`m4_deep_gated`; flat counterpart = same information-set XGB / Ridge):


| Pair | Flat XGB RMSE | Deep RMSE | skill flat | skill deep | DM vs XGB | DM vs Ridge |
| --- | --- | --- | --- | --- | --- | --- |
| M1 | 4.368 | 4.250 | −5.2% | −2.4% | 0.097 | 0.466 |
| M2 | 4.440 | 4.253 | −7.0% | −2.4% | **0.042** ✅ | 0.096 |
| M3 | 4.429 | 4.147 | −6.7% | **+0.11%** | **0.010** ✅ | 0.062 |
| M4 | 4.507 | 4.205 | −8.6% | −1.3% | **0.005** ✅ | **0.036** ✅ |


**How to read**:

- Deep significantly beats flat XGB on **M2/M3/M4 (especially with shipping)**; M1 alone does not reach 5%.
- **Do not claim "deep is always better"** — gains concentrate in multimodal + shipping settings.
- **Coexists with §12.2–12.3**: DM vs flat **M1** is the more conservative test (fusion DM=0.061, not significant); DM vs **matched-information-set flat XGB** is the Ch4 RQ2 main table. Write with the pairing table as primary, DM vs M1 as a supplement.

---

<a name="13"></a>
## 13. Step 11: robustness

Script: `04_code/scripts/deep/run_deep_sweep.py` → `deep_sweep_summary.csv` (lookback=4). Goal: show the main conclusion is not a single-seed / single-hyperparameter fluke.

- **Multi-seed (3 seeds each: 42/1/2)**: fusion −0.47% ± **0.86 (smallest variance = most stable)**; m4rep −0.89% ± 0.60; m4xattn −1.83% ± **2.76 (most unstable, seed2 collapses to −4.98%)**. CWvsM0 mostly 0/3, DMvsM1 fusion 1/3 — **the stable one is gating; the unstable one is cross-attention**.
- **Hyperparameter sweep (fusion, seed42)**: lb=8 d=32 best (+0.34%, DMvsM1 0.041) > lb=4 (+0.11%) > lb=12 (negative); **d=64 is always worse** (evidence that "encoder dimension must be small"). **The main model still locks lb=4** to align the flat protocol for a fair comparison.
- **RS regularisation grid (P1-5)**: meanpool all negative (best −0.90%), cls −11% — **RS weakness is intrinsic, not fixable by tuning**.
- **Main fusion regularisation grid (P1-6)**: the whole grid is skill≈0, dp=0.3 slightly better (+0.29%) — **the main model is robust to regularisation**.
- **Optional, not blocking submission**: deep leave-one-AOI / leave-one-Hormuz-node (see progress overview §3.5).

---

<a name="14"></a>
## 14. Step 12: interpretability (RQ3)

Scripts: `run_deep_interpret.py` (gate + node attention) + `run_deep_xattn_viz.py` (cross-attention). During walk-forward, the fusion info dict is additionally recorded per week — never peeking at the future.

### 14.1 Single seed=42 main figures (narrative starting point)

- **Mean modality gate (gated main model)**: finance **0.44** > RS **0.348** > shipping **0.212**. The gate weights are plotted as a time stackplot overlaid with known supply / geopolitical event lines (Russia–Ukraine 2022-02, EU RU oil ban 2022-06, OPEC+ surprise cut 2023-04, Houthi Red Sea 2023-11) to check temporal alignment.
- **Shipping node attention**: most attended is the **Hormuz chokepoint** + P003/P009/P001/P005 (matching the Red-Sea-rerouting / export-terminal narrative).
- **RS site attention**: most attended P004/P008/P009/P001/P006 (export-terminal AOIs).
- **Cross-attention (finance Query over 28 tokens)**: → shipping **0.575** / → RS **0.425**; top tokens are RS sites but the weights are highly uniform (~0.04), i.e. attention is "flat" → echoes weak/redundant RS information.
- **A discussion point**: the two fusion mechanisms rank "RS vs shipping" **oppositely** (gating RS>shipping; xattn shipping>RS).
- **Caveat**: a high α_shipping does **not** mean the model is "looking at Hormuz" — spatial detail lives in the **node/site attention** layer, not in the fusion gate itself.

Outputs (seed42): `deep_gate_weekly.csv`, `deep_interpret.png`, `deep_xattn_weekly.csv`, `deep_xattn_viz.png` (mostly under `05_outputs/baselines/Deep/M4_Deep/`).

### 14.2 Multi-seed gate / attention stability (2026-07-16, seeds={42,1,2}) ✅

Script: `run_deep_interpret.py --seeds 42,1,2 --lookback 4`. Outputs: `deep_gate_stability.csv` / `deep_gate_corr.csv` / `deep_gate_events.csv` / `deep_gate_band_weekly.csv` / `deep_interpret_stability.png` + `deep_gate_weekly_seed{S}.csv`.

**Mean modality α (across seeds)**: finance ≈ 0.44–0.48 > rs ≈ 0.30–0.35 > shipping ≈ 0.21–0.26 (**rank order stable**).

**Week-level α_shipping correlation across seeds**: Pearson is weak and inconsistent (42↔1 ≈ 0.10, 42↔2 ≈ 0.13, 1↔2 ≈ **−0.31**) — **weekly trajectories are unstable**; do not tell fine-grained event stories from a single seed.

**Event-window Δα_shipping (post−pre, ±8 weeks)**:

| Event | Same sign? | Reading |
| --- | --- | --- |
| Russia–Ukraine (2022-02) | ✅ 3/3 **up** | Only robust "shipping-gate rises" window |
| EU RU oil ban / OPEC+ cut | ✅ 3/3 **down** | Same direction but a *fall* — do not write as supply-shock uplift |
| Houthi Red Sea (2023-11) | ✗ 2↑1↓ | **Unstable** — do not lock in |

**Spatial attention Top-5 frequency (stable foci safe to claim in writing)**:

| Focus | freq | Writing stance |
| --- | ---: | --- |
| **hormuz** | **3/3** | ✅ Only cross-seed-stable chokepoint focus |
| P003 RasTanura / P009 Ulsan / P006 Ningbo | 2/3 | Mention as occasional companions, not mechanisms |
| RS: **P006 Ningbo-Zhoushan, P001 Rotterdam** | **3/3** | ✅ Cross-seed-stable RS sites |

> **Writing rule**: claim only **cross-seed-stable foci** (Hormuz; RS Ningbo / Rotterdam). Weak weekly α_shipping correlation → keep event narrative to the Russia–Ukraine co-rising window only; do not lock the Red Sea window. Keep xattn interpretability in the appendix.

---

<a name="15"></a>
## 15. Step 13: advanced ablations

Script: `run_deep_advanced.py` → `deep_advanced_summary.csv` (lookback=4, seed=42).

| arm | skill vs M0 | DirAcc | CWvsM0 | DMvsM1 |
|---|--:|--:|--:|--:|
| M4concat (concat) | −4.06% | 0.494 | 0.637 | 0.650 |
| M4rep gated | −1.28% | 0.502 | 0.894 | 0.239 |
| **M4 cross-attention (finance Query)** | **+0.12%** | 0.564 | **0.018** | 0.088 |
| M4rep + modality dropout 0.3 | −0.19% | 0.498 | 0.316 | 0.096 |
| **M4-xattn + dropout 0.3** | **+0.62%** | 0.560 | **0.008** | **0.050** |

- **Cross-attention can be the single-seed best**: under seed42+lb4 it turns skill positive and is the **only config that significantly beats the random walk via CWvsM0** (0.018; with dropout 0.008 and DMvsM1 0.050, doubly significant); **but multi-seed it is highly unstable (±2.76) → listed as an "upper bound exists but unstable" advanced result; the main model remains gated.**
- **Sub-periods**: xattn is a stable +0.12% in both early/late; gated is −2.35% early / turns positive +0.49% late (2023–25).

---

<a name="16"></a>
## 16. One-line summary + reproduction commands

**One line**: RQ1 increments are weak (RS especially; all models' CWvsM0 non-significant, still hard to beat M0); **RQ2 partially positive** — hardest evidence is nested CW **0.00057** for adding shipping on finance, gating > concat, and under **matched information-set pairing** deep significantly beats flat XGB on M3/M4, but under strict DM vs flat M1 it is **not significant** (directionally consistent); **RQ3** gates/attention are interpretable but must follow **multi-seed stable foci** (Hormuz; RS Ningbo/Rotterdam) — do not lock the Red Sea window. Cross-attention can be single-seed best but is multi-seed unstable → advanced.

**Reproduction commands**:
```bash
# 0) Prerequisite: run M1_Flat first (deep scripts read its predictions)
python3 04_code/scripts/flat/run_baseline.py --modality M1

# 1) Upstream data (if rebuilding)
python3 03_data/processed/M2/py/build_m2_weekly.py                 # RS monthly -> weekly
python3 03_data/processed/M2/py/precompute_s2_embeddings.py        # Prithvi embeddings
python3 03_data/processed/M3/py/aggregate_shipping_to_weekly.py    # shipping -> weekly (flat GFW +4w)
python3 03_data/processed/M3/py/build_m3_graph_weekly.py           # AOI graph intermediates
python3 03_data/processed/M3/py/build_m3_graph17.py                # 17-node graph tensor (event +2w)

# 2) Deep main results + tests
python3 04_code/scripts/deep/run_deep_baseline.py                  # main results table
python3 04_code/scripts/deep/run_deep_fusion_matrix.py             # 3×3 fusion matrix
python3 04_code/scripts/deep/run_deep_sweep.py                     # robustness sweep
python3 04_code/scripts/deep/run_deep_interpret.py --seeds 42,1,2 --lookback 4   # RQ3 + multi-seed stability
python3 04_code/scripts/deep/run_deep_xattn_viz.py                 # RQ3 cross-attention
python3 04_code/scripts/deep/run_deep_advanced.py                 # advanced ablations
```

---

<a name="17"></a>
## 17. Appendix: full variable & parameter list

### A. Data specs
| Item | Value |
|---|---|
| Unified window | 2019-01-04 ~ 2025-12-26 (365 weeks) |
| Merged matrix | **365 × 213** (incl. `week_ending_friday`); 212 data cols = 31+55+113+11+2 |
| Common test span | 257 weeks (2021–2025) |
| Target | r_{t+1}=ln(P_{t+1}/P_t), reconstruct P̂=P_t·e^r̂ |
| lookback (main) | **4 weeks** (07-05 first run used 8; historical / hyperparameter arm only) |
| RS publication lag | month-end + 15 days (as-of backward) |
| Shipping lags | flat GFW presence **+4w**; deep-graph GFW event/O-D **+2w**; PortWatch **+1w**; SAR **+4w** |

### B. Model / training hyperparameters
| Item | Value |
|---|---|
| Representation dim d | 32 (each encoder's output) |
| GAT | 2 layers, 4 heads, d_model=64, log O-D flow prior |
| TCN | 2 layers, kernel=3, adaptive dilation (dilation=1 at lb≤5) |
| RS encoder | frozen Prithvi-EO-2.0-300M (300M params) + temporal/site attention, proj 1024→64→32 |
| Fusion | gated (main) / concat / cross-attention (4 heads) |
| Optimiser | Adam, lr=1e-3, weight_decay=1e-4 |
| batch / epochs | 32 / 80 (inner-val early stopping, patience=12) |
| Gradient clip | max_norm=5.0 |
| Backtest | min_train=104, retrain_every=13, seed=42 (main table); robustness also seeds {1,2} |

### C. Graph nodes and chokepoints
- **11 AOI**: P001 Rotterdam, P002 Fujairah, P003 RasTanura, P004 Jurong, P005 Houston, P006 NingboZhoushan, P007 Jamnagar, P008 Basra, P009 Ulsan, P010 Kharg, P011 Yanbu.
- **6 chokepoints**: hormuz, suez, malacca, mandeb, panama, cape.

### D. Key files
| Category | Path |
|---|---|
| Encoders | `04_code/src/models/{finance,rs,shipping}_encoder.py` |
| Fusion | `04_code/src/models/fusion.py` |
| Data alignment | `04_code/src/models/deep_dataset.py` |
| Backtest kernel | `04_code/src/models/deep_rolling.py` |
| Entry scripts | `04_code/scripts/deep/run_deep_{baseline,sweep,interpret,advanced,fusion_matrix,xattn_viz}.py` |
| Graph tensor | `03_data/processed/M3/outputs/m3_graph17_tensors.npz` |
| Prithvi embeddings | `03_data/processed/M2/outputs/s2_prithvi_emb_meanpool.npy` |
| Results | `05_outputs/baselines/Deep/{M*_Deep,_cross}/` |
| RQ3 multi-seed | `…/M4_Deep/deep_gate_stability.csv` etc. |
| Progress overview | `00_admin/最新待整理/2026-07-15_研究方案与进度总览.md` |
| Flat walkthrough | `00_admin/最新待整理/flat_baseline_full_walkthrough_EN.md` |
