# Dissertation Draft Structure

**Working title:**  
A Modality-Aware Spatio-Temporal Fusion Framework for Brent Crude Oil Forecasting Using Financial Time Series, Satellite Imagery and Maritime Networks

**Research questions:**

- **RQ1:** Do RS / shipping add incremental OOS value over finance (M1) and random walk (M0)?
- **RQ2:** Does modality-aware representation-level fusion beat flat feature fusion on the same data?
- **RQ3:** Can modality-level interpretability show which signals matter under different market conditions?


| RQ  | Methods (Ch3)                      | Results (Ch4)  | Discussion (Ch5) |
| --- | ---------------------------------- | -------------- | ---------------- |
| RQ1 | M0–M4 ladder; CW vs M1; DM vs M0   | §4.2–4.3       | §5.1             |
| RQ2 | Flat vs deep, same data/protocol   | §4.4 paired Mi | §5.2             |
| RQ3 | SHAP / gates / node–site attention | §4.6           | §5.3             |


## Chapter 1 — Introduction *(~900–1,200)*



### 1.1 Background and motivation

- Weekly Brent hard to beat; no-change / RW is a strong benchmark
- Shipping (AIS / PortWatch) and satellite RS as physical-market proxies
- Most multimodal oil work still flattens sources into one wide table



### 1.2 Research problem

- Incremental value of RS / shipping over finance still unclear
- Unclear if representation-level fusion beats flat concat on same data
- Need leakage-safe protocol + formal forecast tests (DM / CW)



### 1.3 Research gap

- Rare flat vs modality-aware comparison under **one** protocol
- Few studies report both nested increments (vs M1) and absolute skill (vs M0)



### 1.4 Research questions

- State RQ1–RQ3 explicitly



### 1.5 Contributions

- Unified M0–M4 ladder across flat and deep
- Rolling-origin backtest + DM / Clark–West
- Paired Flat–Deep by information set (not best-vs-best)
- Interpretability mainly where predictive value exists (§3.10)



### 1.6 Dissertation structure

- Roadmap Ch2–Ch6

---



## Chapter 2 — Literature Review



---



## Chapter 3 — Data and Methods *(~2,500–3,000)*

*Body stays readable; full dictionaries / grids → Appendix A/C.*

### 3.1 Research design

- Two arms, **same** information sets / timeline / metrics:
  - **Flat:** early concat → Ridge / XGBoost
  - **Deep:** modality encoders → gated / concat / cross-attn fusion
- Nested ablation ladder M0 → M1 → M2 → M3 → M4
- Positioning: integration + fair comparison, not new architecture



### 3.2 Prediction target and timeline

- Target: next-week Brent spot P_{t+1} (USD/bbl)
- Train on log return r_{t+1}=\log(P_{t+1}/P_t); evaluate reconstructed \hat P_{t+1}=P_t e^{\hat r}
- Sample window: **2019–2025**, Friday-ending weeks (W-FRI)
- Merged matrix ≈ **365 weeks**
- Common test span: **257 weeks (~2021–2025)** after warm-up
- Horizon: **1 week ahead** (single-task regression)



### 3.3 Data sources (summary; details → App. A)

**M1 finance / macro (~31 cols)**

- EIA: Brent/WTI, WPSR stocks / production / imports–exports / refinery
- FRED: VIX, DXY, 10y, fed funds, industrial materials
- Yahoo: S&P500, OVX, Brent futures, CADUSD, gold
- Other: Kilian REA, Caldara–Iacoviello GPR

**M2 remote sensing**

- Sentinel-2 monthly indices (NDVI / NDWI / NDBI / BSI) @ 11 AOIs
- VIIRS DNB night lights (NTL)
- Flat arm: **55 within-site seasonally adjusted anomalies (anom)**
- Deep arm: frozen **Prithvi-EO** patch embeddings (1024-d meanpool; cls as alt.)
- 11 AOIs (5 km buffers): Houston, Rotterdam★, NingboZhoushan, Jamnagar, Jurong, Ulsan, Basra, Fujairah★, Kharg, RasTanura★, Yanbu★
- ★ = literature core NTL sites (4-site “literature arm”)

**M3 shipping**

- IMF PortWatch: 6 chokepoints + port tanker flows (daily → weekly)
- GFW: AIS vessel presence / voyages / (optional) SAR dark-vessel
- 6 chokepoints: hormuz, suez, malacca, mandeb, panama, cape
- Flat arm: **113-col full tier** (not 38-col core — core weakest in LOCO)
- Deep arm: **17-node dynamic heterogeneous graph** (11 AOI + 6 choke)



### 3.4 Temporal alignment, lags, missingness

- Unified W-FRI axis; as-of / publication lags (no look-ahead)
- Key lags:
  - EIA WPSR: **+1 week**
  - PortWatch: **+1 week**
  - GFW voyages / port visits: **+2 weeks** (SAR monthly +4 if used)
  - RS monthly: obs mid-month + **PUB_LAG_DAYS=15**; `merge_asof(backward)`; MAX_AGE≈100 days
  - Monthly macros (Kilian / IMF materials): longer lag (e.g. +5 weeks)
- Flat missingness: past-only ffill; residual NaN → 0 in warm-up only
- Deep: explicit **masks** for RS / missing modality (no silent future fill)
- Cloud screening (S2, GEE): CLOUDY_PIXEL≤60; cloud prob<40; SCL masks; monthly **median** composite



### 3.5 M0–M4 information sets


| Set    | Content                           | Flat width (L=4 flatten) |
| ------ | --------------------------------- | ------------------------ |
| **M0** | Random walk \hat r=0 → \hat P=P_t | —                        |
| **M1** | Finance only (31)                 | 31×4 = **124**           |
| **M2** | M1 + RS anom (55)                 | 86×4 = **344**           |
| **M3** | M1 + shipping full (113)          | 144×4 = **576**          |
| **M4** | M1 + RS + shipping                | 199×4 = **796**          |


- Identical modality definitions for flat and deep arms
- Deep configs (examples): `fin`, `rs`, `ship`, `fusion`/`finship`, `finrs`, `m4rep` (gated), `m4xattn`, `m4concat`



### 3.6 Flat models

**M0**

- Rule: \hat r=0; no training
- Anchor RMSE ≈ **4.152** $/bbl on common test

**Ridge**

- Pipeline (fit inside train fold only): `VarianceThreshold(0) → StandardScaler → Ridge(α)`
- Default α = 10
- Grid: `α ∈ {0.1, 1, 10, 100, 1000}`
- High-dim → larger α preferred

**XGBoost**

- Pipeline: `VarianceThreshold(0) → XGBRegressor` (no scaler)
- Defaults: `n_estimators=300, max_depth=3, lr=0.05, subsample=0.8, colsample_bytree=0.8, reg_λ=1`
- Grid (8 combos): depth∈{2,3} × lr∈{0.03,0.05} × n_est∈{200,400}; subsample/colsample/λ fixed
- Design: shallow trees + row/col subsample + L2 (small-n, high-p)

**Inner validation (both)**

- Expanding train fold; last **52 weeks** = val; pick lowest val RMSE
- If train < ~82 weeks → use defaults
- Retune **never** sees test set



### 3.7 Deep / representation-level models

**Shared protocol with flat**

- lookback=**4**, min_train=**104**, retrain_every=**13**, seed=**42**
- Same 257 test weeks

**Encoders (each → 32-d)**

- **z_fin:** causal TCN on (L, 31); proj→LN→TCN (tcn_layers=2, kernel=3); adaptive dilation (L≤5 dense; L≥6 exponential)
- **z_rs:** frozen Prithvi 1024-d → proj 1024→64 → temporal attn (per site) → site attn over 11 AOIs → 32-d (+ site weights for RQ3)
- **z_ship:** heterogeneous GAT (gat_layers=2, heads=4, ~42k params) on 17 nodes + temporal TCN + node-attn pool (+ node weights for RQ3); log1p(O–D flow) as attention prior

**Fusion ladder (RQ2)**

- **Concat:** encode separately → fixed MLP mix (floor / control)
- **Gated (main):** `α = softmax(MLP([z_i]))`; `z = Σ α_i z_i` (per-sample; α → RQ3)
- **Cross-attn:** finance as Q over RS sites + ship nodes (28 tokens); 4 heads

**Head / training**

- Head: `Linear(32,32)→ReLU→Dropout→Linear(32,1)`
- Optional **modality dropout** (e.g. 0.3) during train
- Adam: lr=1e-3, wd=1e-4, batch=32, epochs≤80, early stop patience=12 (val=last 52w)
- Loss: MSE on fold-standardised r; grad clip max_norm=5
- Main HPs: d=**32**, dropout=0.1, gat=2, tcn=2 (sweep: d=64 usually worse)



### 3.8 Hyperparameter selection

- Flat: inner 52w val grids (§3.6)
- Deep: main config fixed from sweep; report seed / lookback / d / dropout / fusion sweeps in §4.5 + App. C
- Selection rule in body; full grids → **Appendix C**



### 3.9 Leakage-free validation protocol

- Expanding-window **rolling-origin** (walk-forward)
- Warm-up: first **104** weeks train-only (≈2y)
- Retrain every **13** weeks (~quarterly); between refits reuse fold model
- Train size grows ~104 → ~360 weeks
- Scalers / filters / SHAP-based selection fit **inside train fold only**
- Flat ≈ 20 folds × 2 models; deep aligned fold structure



### 3.10 Evaluation, tests, interpretability

**Metrics (on reconstructed price)**

- RMSE, MAE, DirAcc
- skill vs M0: 1 - \mathrm{RMSE}/\mathrm{RMSE}_{M0} (>0 beats RW)

**Formal tests**

- **DM vs M0** (non-nested; HLN small-sample correction)
- **Clark–West vs M1** for nested M2/M3/M4 (and deep nested increments)
- Stress in text: **CW significance ≠ beating M0 on RMSE**

**Interpretability rule**

- Primary: best models that beat / approach relevant benchmark (esp. skill>0 vs M0)
- Supplementary: SHAP / gates for models with significant CW gain over M1 even if skill<0 (e.g. Flat M3) — framed as *why shipping helps M1*, not absolute superiority
- Flat: SHAP by feature / modality share
- Deep: modality gates α; shipping **node** attn; RS **site** attn (do not conflate layers)
- Ethics / reproducibility: public data; scripted pipeline; forecasts ≠ investment advice

---



## Chapter 4 — Results *(~2,500–3,200)*

*Organised by experimental logic + RQs — not only Flat-block then Deep-block.*

### 4.1 Descriptive overview

- Window 2019–2025; test **n=257**; target next-week Brent price
- M0 RMSE ≈ **4.152**
- Arms: Flat Ridge/XGB; Deep fin / ship / rs / finship / finrs / m4*
- Metrics + DM/CW plan; point to key tables/figures



### 4.2 Flat-model results *(RQ1)*

**Headline numbers (L4_tuned, n=257, M0=4.152)**


| Model    | RMSE  | skill vs M0 | CW_p vs M1   |
| -------- | ----- | ----------- | ------------ |
| M0_RW    | 4.152 | —           | —            |
| M1_Ridge | 4.256 | −2.5%       | —            |
| M1_XGB   | 4.368 | −5.2%       | —            |
| M2_Ridge | 4.414 | −6.3%       | 0.474        |
| M2_XGB   | 4.440 | −6.9%       | 0.085 ✗      |
| M3_Ridge | 4.430 | −6.7%       | 0.264        |
| M3_XGB   | 4.429 | −6.7%       | **0.0002** ✓ |
| M4_Ridge | 4.525 | −9.0%       | 0.314        |
| M4_XGB   | 4.507 | −8.6%       | **0.009** ✓  |


**Bullet findings**

- **No flat model beats M0** (all skill < 0)
- RS (M2): not significant vs M1 on full 55-col anom
- Shipping (M3 XGB): **highly significant** nested gain vs M1, but RMSE still worse than M1_XGB
- M4 XGB: CW significant, **worst RMSE** — signal exists, flat concat cannot harvest it cleanly
- Ridge vs XGB: Ridge better RMSE on M1; XGB better at detecting nested shipping / M4 increments
- Emphasise: CW ≠ absolute skill vs M0



### 4.3 Deep-model results *(RQ1 within deep arm)*

**Selected deep metrics (seed=42, L=4, n=257)**


| Config             | RMSE      | skill vs M0 | notes                           |
| ------------------ | --------- | ----------- | ------------------------------- |
| Mfin (TCN)         | 4.250     | −2.4%       | ≈ flat M1 Ridge                 |
| Mrs                | 4.247     | −2.3%       | weak                            |
| Mship              | 4.168     | −0.4%       | closer to M0                    |
| **Mfinship gated** | **4.147** | **+0.11%**  | first skill>0                   |
| Mfinrs gated       | 4.253     | −2.4%       | RS drag                         |
| M4rep gated        | 4.205     | −1.3%       |                                 |
| M4concat           | 4.320     | −4.1%       | worst fusion                    |
| **Mfinship xattn** | **4.121** | **+0.74%**  | best overall; CW vs M0 ≈0.041 ✓ |
| M4 xattn           | 4.147     | +0.12%      | CW vs M0 ≈0.018 ✓               |
| M4 xattn + drop0.3 | 4.126     | +0.62%      | CW≈0.008; DM vs M1≈0.050        |


**Fusion matrix (3 combos × 3 fusions) — takeaways**

- Finance+shipping: **all three fusions skill>0**; xattn best
- Finance+RS: **all skill<0**; xattn worst (noise amplification)
- Full M4: concat worst; gated mixed; xattn can beat M0 on CW
- Adding RS to fin+ship often **hurts** absolute RMSE vs finship alone



### 4.4 Flat vs deep *(core RQ2)*

**Paired by information set (hold modality content fixed)**


| Pair         | Flat RMSE | Deep RMSE | skill flat | skill deep | DM vs XGB   | DM vs Ridge |
| ------------ | --------- | --------- | ---------- | ---------- | ----------- | ----------- |
| M1deep vs M1 | 4.368     | 4.250     | −5.2%      | −2.4%      | 0.097       | 0.466       |
| M2deep vs M2 | 4.440     | 4.253     | −7.0%      | −2.4%      | **0.042** ✓ | 0.096       |
| M3deep vs M3 | 4.429     | 4.147     | −6.7%      | +0.11%     | **0.010** ✓ | 0.062       |
| M4deep vs M4 | 4.507     | 4.205     | −8.6%      | −1.3%      | **0.005** ✓ | **0.036** ✓ |


**Bullet findings**

- Separate **architecture effect** from **modality-set effect**
- Deep wins clearest on M3/M4 (and M2 vs XGB); M1 alone DM not at 5%
- Do **not** over-claim “deep always better”
- Prefer: *representation-level fusion outperforms flat counterparts in selected multimodal settings, especially with shipping*



### 4.5 Robustness and sensitivity

**Flat**

- Lookback sweep (esp. M1): Ridge worsens with longer L; XGB more stable
- M2 literature arm (4 NTL cols): XGB CW≈0.022 ✓ (sparse > full 55)
- M2 water-mask: XGB CW 0.085→≈0.028 ✓
- M2 sparsity / LOAO / PCA–ElasticNet (C2)
- M3 channel arms: core(38) **not** sig; full/tanker/portwatch **sig**
- M4 LOMO: M1-only often best RMSE; adding M2 to M1+M3 can hurt

**Deep**

- Seeds {42,1,2}: xattn / gated variance (report mean + range)
- Lookback ∈ {4,8,12}; d ∈ {32,64} — d=64 usually worse
- Modality dropout 0 / 0.3
- Early (≤2022) vs late (≥2023) splits
- RS pool: meanpool vs cls (cls worse)
- Optional min_train=78 longer test window



### 4.6 Interpretability *(RQ3)*

**Flat SHAP (supplementary where CW>M1)**

- M2: NDVI/NDWI often high among RS; finance still dominates Top-5
- M3: hormuz tanker share; suez wow (Red Sea mechanism); derived cols that core dropped
- M4 modality share (illustrative): shipping ≈51.8% > finance ≈33.7% > RS ≈14.5%

**Deep**

- Modality gates α_fin / α_rs / α_ship over time (m4rep)
- Periods when α_shipping rises (stress / disruption windows)
- Shipping **node** attention (17): which ports / chokepoints
- RS **site** attention (11 AOIs)
- Cross-attn maps (finance attending to nodes/sites) if reported
- **Caveat:** high α_shipping ≠ “model looks at Hormuz”; spatial detail lives in node/site attn, not the fusion gate
- Association ≠ causation

---



## Chapter 5 — Discussion *(~1,600–1,900)*

*Answer RQs; do not re-list tables.*

### 5.1 RQ1 — Alternative-data value

- Full RS (M2) under flat concat: high-dim / collinear / cloudy anom columns dilute weak site signals; skill & CW weak unless sparsified (literature arm) or cleaned (water-mask)
- Shipping (M3) more informative than RS: chokepoint tanker flows map more directly to physical oil balance; XGB CW vs M1 highly significant; SHAP hits Hormuz / Suez mechanisms
- M4 CW-significant yet RMSE-worse: nested test detects directional shipping signal after parameter-noise penalty; flat wide table still adds RS noise → absolute error rises
- Deep finship / xattn skill>0 while flat cannot: separate encoders + gated/cross-attn harvest shipping without flattening noise; RS optional and often a drag when forced into M4
- Nested CW vs M1 ≠ practical skill vs M0: report both; economic meaning stays limited while RMSE gains over RW are small



### 5.2 RQ2 — Representation-level vs flat

- Paired Deep vs Flat (§4.4): deep wins clearest on M3/M4 (and M2 vs XGB); M1-only DM not at 5%
- Deep gains tied to temporal encoders, non-linearity, modality-specific structure, and fusion mechanism — not modality content alone
- Finance-only: architecture swap insufficient; gap opens mainly once shipping enters
- Complexity justified for finship xattn; full M4 mixed / often redundant with RS



### 5.3 RQ3 — Modality and spatial dependence

- Gate / SHAP: finance vs shipping vs RS weight patterns over the test window
- Shipping gate / SHAP weight rises in disruption windows (e.g. Red Sea / chokepoint stress)
- Spatial focus: Hormuz, Suez, and related ports/chokepoints in node attention / SHAP
- Attributions economically plausible as physical-market proxies
- Limit: model dependence ≠ causal effect; gate α ≠ which specific node is attended



### 5.4 Implications

- Fair multimodal comparison protocol matters as much as “new” models
- Always report nested (vs M1) **and** absolute (vs M0)
- Practical caution when gains over M0 are small



### 5.5 Limitations

- Weekly horizon; ~365-week sample; proxy noise; reverse causality
- Frozen EO backbone; graph construction choices; missingness
- Seed sensitivity (esp. xattn); compute / coverage



### 5.6 Future research

- Longer history for best deep config
- Richer AIS graphs / dynamic edges
- Stronger missing-modality stress tests
- Other horizons / commodities

---



## Chapter 6 — Conclusion *(~400–700)*



### 6.1 Summary of findings

- RQ1: flat — no beat M0; shipping nested signal; RS weak unless sparsified / cleaned
- RQ1 deep: finance+shipping (esp. xattn) can beat M0; RS often drag
- RQ2: paired deep > flat mainly in multimodal / shipping settings
- RQ3: gates + node/site attn + SHAP as complementary layers



### 6.2 Contributions

- Integration + systematic Flat vs Deep under leakage-safe protocol
- Dual testing lens (CW vs M1 + skill/DM vs M0)



### 6.3 Final conclusion

- Strong baselines first; fusion design second; interpretability where predictive value exists

---



## References



## Appendices

- **A.** Full variable dictionaries (M1–M4); AOI / chokepoint lists; lag table
- **B.** Extra result / robustness tables & figures (lookback, LOAO, LOMO, seeds, early/late)
- **C.** Hyperparameter grids, seeds, software / config paths
- **D.** Supplementary SHAP for incremental-but-not-M0 models (if kept out of main text)

---



## Quick reference — locked experimental knobs


| Knob            | Value                                        |
| --------------- | -------------------------------------------- |
| Target          | next-week Brent P_{t+1} via log-return train |
| Calendar        | W-FRI, 2019–2025                             |
| lookback        | 4 (main)                                     |
| min_train       | 104                                          |
| retrain_every   | 13                                           |
| inner val       | last 52 weeks of train fold                  |
| n_test          | 257                                          |
| M0 RMSE         | ≈4.152                                       |
| Flat models     | Ridge + XGBoost                              |
| Deep d / fusion | 32; gated main; concat / xattn ladder        |
| Seed (main)     | 42                                           |


