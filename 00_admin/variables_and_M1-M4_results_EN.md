# Variables in Use & M1–M4 Model Results

---

## 0. M0 — Baseline (Random Walk / No-Change Prediction)

**M0 is the feature-free "naive baseline" layer**, serving as the **floor** for judging whether M1–M4 have genuine predictive skill. A model only demonstrates real information gain from its features if it significantly beats M0, rather than merely exploiting the series' own persistence.


| target     | M0 baseline                          | definition                                                                                                      |
| ---------- | ------------------------------------ | --------------------------------------------------------------------------------------------------------------- |
| price      | **random walk / no-change**          | `ŷ(t+1) = y(t)` — predict next week's price with this week's price                                              |
| volatility | **naive persistence / rolling mean** | `ŷ(t+1) = vol(t)` (or the mean of the past N weeks)                                                             |
| direction  | **majority class / persistence**     | always predict the training-set majority class; or `ŷ(t+1) = sign(Δy(t))` (carry forward last week's direction) |


**Why M0 is essential**

- **High price R² is misleading:** oil price is highly persistent, so a random walk alone achieves a very high R². M1's price R²≈0.867 looks strong, but without comparison to a random walk one cannot tell whether the model truly "predicts" price or merely copies the previous value.
- **Direction must be compared to random/majority:** for the 3-class problem (down/flat/up) the random baseline is ≈ 0.33–0.45, and the majority-class baseline may be higher. M1–M4 accuracy is mostly 0.42–0.56, so only against M0 can effectiveness be judged.

**Correct comparison metrics (relative to M0)**

- **Out-of-sample R² vs random walk (OOS-R²):** `1 − MSE(model) / MSE(RW)`; >0 means it beats the random walk.
- **Diebold–Mariano (DM) test:** tests whether the difference in forecast errors between the model and M0 is statistically significant.

---

## 1. Prediction Targets (next 1 week)


| key        | column                           | type                          |
| ---------- | -------------------------------- | ----------------------------- |
| direction  | `target_brent_direction_next_1w` | classification (down/flat/up) |
| volatility | `target_brent_vol_next_1w`       | regression                    |
| price      | `target_brent_price_next_1w`     | regression                    |


---

## 2. Layered Variable Definitions

Layers are **cumulative**: `M2 = M1 + remote sensing`, `M3 = M1 + shipping`, `M4 = M1 + remote sensing + shipping`.

### M1 — Financial / Market / Macro baseline (10 variables, mechanism-based)


| #   | variable                      | economic mechanism                            |
| --- | ----------------------------- | --------------------------------------------- |
| 1   | `brent_price`                 | oil-price own dynamics (lags/ma/mom)          |
| 2   | `crude_stocks_change`         | supply / market balance                       |
| 3   | `global_econ_activity`        | global demand (Kilian REA)                    |
| 4   | `nonoil_industrial_commodity` | global demand (industrial materials)          |
| 5   | `futures_spread`              | market tightness / expectations               |
| 6   | `ovx`                         | oil-specific uncertainty (preferred over VIX) |
| 7   | `gpr`                         | precautionary demand (geopolitical risk)      |
| 8   | `dgs10_change`                | rates / carry cost (ΔDGS10)                   |
| 9   | `gold_return`                 | commodity comovement / safe haven             |
| 10  | `commodity_fx`                | FX channel (CAD/AUD)                          |


**Derived features (LAG_MA_SPECS):** three families of time-series features are constructed for selected base variables (all on weekly W-FRI frequency, using past values only, to avoid leakage):

- **lag-k:** value at week t−k, `x_lag{k} = x(t−k)`.
- **ma-w:** mean over the past w weeks, `x_ma{w} = mean(x(t−w+1 … t))`.
- **mom-w:** change relative to w weeks ago, `x_mom{w} = x(t) − x(t−w)`.

> `brent_price` uses 1/4-week lags plus 4/12-week moving averages and momentum to carry the "price lags + returns" information; other variables only add the short-term lags/MAs warranted by their economic nature. Full definition in `LAG_MA_SPECS` in `config.py`.

### M2 increment — Night-time lights remote sensing (12 variables, 4 oil-core AOIs)

**dynamic NTL anomaly** (not raw radiance) + **observation-quality variables**
The four AOIs: Rotterdam (import/refining hub), Fujairah (offshore tanker-dominated storage), Ras Tanura (crude export terminal), US Gulf (Houston / US Gulf storage belt).
The full 11-AOI panel is in `feature_groups.json["M2_rs_clean"]` / `weekly_m2_clean_features.csv`.


| variable (× 4 AOIs)     | count | meaning                                                                                                        |
| ----------------------- | ----- | -------------------------------------------------------------------------------------------------------------- |
| `ntl_anomaly`_*         | 4     | night-time-light **anomaly** (deseasonalised/detrended dynamic deviation, proxy for activity-intensity change) |
| `ntl_valid_obs_count`_* | 4     | VIIRS **valid-observation days** (data-quality / reliability control)                                          |
| `s2_cloud_fraction`_*   | 4     | Sentinel-2 **cloud fraction** (optical information availability; more cloud = less usable signal)              |


> `*` denotes the four AOI suffixes: `rotterdam` / `fujairah` / `ras_tanura` / `us_gulf`.

**Mechanism validation (how to prove the NTL anomaly really reflects oil activity)**

Idea: the NTL anomaly is only a **remote-sensing proxy** for activity; it must be cross-validated against independent "ground-truth" economic quantities — if the NTL anomaly correlates significantly and in the same direction with these physical throughput / inventory / refinery indicators at the corresponding AOI, the signal has a mechanistic basis.


| mechanism dimension                    | ground-truth variable (validation benchmark)                                       | source / layer |
| -------------------------------------- | ---------------------------------------------------------------------------------- | -------------- |
| tanker total capacity                  | `pw_{choke}_capacity_tanker`                                                       | PortWatch (M3) |
| port throughput / import-export volume | PortWatch port-level `export/import_tanker`; EIA `crude_imports` / `crude_exports` | PortWatch; EIA |
| official inventory change              | `crude_stocks_change`                                                              | EIA (M1)       |
| refinery activity                      | `refinery_crude_input` / `refinery_utilisation`                                    | EIA            |


> Validation procedure: compute (lagged) correlation / regression between the NTL anomaly and the above benchmarks at each AOI, and compare high- vs low-valid-observation subsamples (quality control) to confirm the correlation is not driven by missing observations or cloud noise.

**Why 4 AOIs rather than 11?**

- **Clear oil mechanism:** the 4 core AOIs each correspond to a well-defined oil-functional zone — Rotterdam (import/refining hub), Fujairah (offshore tanker-dominated storage), Ras Tanura (crude export terminal), US Gulf (Houston/US Gulf storage belt) — covering the full "import–storage–export–refining" chain, directly tied to the oil-price mechanism.
- **Signal quality vs noise trade-off:** the remaining AOIs are either heavily contaminated by non-oil urban activity, suffer severe missing/cloud issues, or are too small with weak signal; including them dilutes SNR and increases collinearity.
- **Verifiability:** all 4 core AOIs have matching ground truth (PortWatch port throughput / EIA import-export-inventory), enabling the validation above; some AOIs lack comparable throughput data.
- **Full 11-AOI kept for robustness:** the complete panel is in `feature_groups.json["M2_rs_clean"]` / `weekly_m2_clean_features.csv`, used as a robustness check rather than the main specification.

### M3 increment — Shipping (19 variables)

Organised by economic dimension: tanker-specific flow intensity + DWT-capacity weighting + average vessel size + global aggregate + import-export directional asymmetry + GFW dwell-time congestion proxy.
Data sources: PortWatch (2019+), Global Fishing Watch / AIS (2012+).
**Number of chokepoints: 3** — Hormuz, Suez, Malacca; plus two global aggregates across all chokepoints (`pw_all`_* / `gfw_all`_*).

#### (a) PortWatch chokepoint tanker transits (10 variables)

Each chokepoint (hormuz / suez / malacca) has 3 variables + 1 global aggregate:


| variable (× 3 chokepoints) | count | meaning                                                             |
| -------------------------- | ----- | ------------------------------------------------------------------- |
| `pw_*_n_tanker`            | 3     | tanker transit **count**                                            |
| `pw_*_capacity_tanker`     | 3     | **DWT-capacity**-weighted transit volume                            |
| `pw_*_avg_tanker_size`     | 3     | **average vessel size** of tankers                                  |
| `pw_all_n_tanker_sum`      | 1     | **global aggregate** of tanker transit count across all chokepoints |


#### (b) PortWatch import-export directional asymmetry (2 variables, directional)


| variable                 | meaning                                                                  |
| ------------------------ | ------------------------------------------------------------------------ |
| `pw_tanker_exp_imp_asym` | asymmetry index = (export hubs − import hubs) / (sum of both), ∈ [-1, 1] |
| `pw_tanker_exp_imp_net`  | net volume = export-hub loadings − import-hub discharges (tonnes)        |


#### (c) GFW / AIS vessel presence & congestion proxy (7 variables, 2012+)

Each chokepoint (hormuz / suez / malacca) has 2 variables + 1 global aggregate:


| variable (× 3 chokepoints)     | count | meaning                                                                            |
| ------------------------------ | ----- | ---------------------------------------------------------------------------------- |
| `gfw_*_total_hours`            | 3     | AIS vessel **total activity hours** (activity intensity)                           |
| `gfw_*_dwell_hours_per_vessel` | 3     | **dwell hours per vessel** (congestion / dwell proxy = total hours / vessel count) |
| `gfw_all_total_hours_sum`      | 1     | **global aggregate** of AIS total activity hours across all chokepoints            |


**Mechanism validation (how to prove the shipping variables really reflect oil supply/logistics)**

Idea: PortWatch transits and GFW presence are both proxies for vessel activity; they must be cross-validated against independent "ground truth" — seaborne import-export volumes, freight rates, official supply indicators. If the two datasets corroborate each other and correlate significantly and in the same direction with these benchmarks, the signal has a mechanistic basis.


| mechanism dimension                           | ground-truth variable (validation benchmark)                                                                        | source / layer                                   |
| --------------------------------------------- | ------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------ |
| transits vs presence (two-source cross-check) | `pw_{choke}_n_tanker` ↔ `gfw_{choke}_total_hours`                                                                   | PortWatch ↔ GFW (cross-check within M3)          |
| seaborne import-export volume                 | EIA `crude_imports` / `crude_exports`; PortWatch port-level `export/import_tanker`                                  | EIA (in `weekly_time_index.csv`); PortWatch (M3) |
| congestion / freight rates                    | dwell `gfw_{choke}_dwell_hours_per_vessel` ↔ tanker freight-rate index (BDTI/TD3, external, not yet in main matrix) | GFW (M3) ↔ external freight rates                |
| official supply / inventory                   | `crude_stocks_change`, `crude_production` (EIA)                                                                     | EIA (M1 / in `weekly_time_index.csv`)            |


> Validation procedure: compute (lagged) correlation between transits/capacity and EIA seaborne import-export at each chokepoint; use PortWatch ↔ GFW cross-checking to rule out single-source missingness; and test whether dwell time rises together with freight rates during geopolitical events (e.g. Hormuz tensions).

### Per-layer variable counts


| layer | composition                              | base variable count |
| ----- | ---------------------------------------- | ------------------- |
| M1    | financial/macro                          | 10                  |
| M2    | M1 + remote sensing (12)                 | 22                  |
| M3    | M1 + shipping (19)                       | 29                  |
| M4    | M1 + remote sensing (12) + shipping (19) | 41                  |


> ⚠️ **Evaluation-window caveat:** M3/M4 include PortWatch shipping variables (from 2019+); after `dropna`, any layer containing PortWatch is evaluated on **2019+ only** (test set ≈ 2024-11 to 2025-12, N≈54 weeks), whereas M1/M2 use the full sample (2006+). Therefore M3/M4-vs-M1/M2 cross-layer comparisons are confounded by sample-period/test-set differences and must be reported with this caveat.

---

## 3. M1–M4 Model Results (organised by layer, then by the three targets)

Models: classification uses LogisticRegression (linear baseline) / RandomForest / SVM / XGBoost / LSTM / TFT / ST-GNN;
regression uses Ridge (linear baseline) / RandomForest / SVM / XGBoost / LSTM / TFT / ST-GNN.
In each table the **best model is in bold**: Direction by accuracy, Volatility / Price by R² (higher is better).

### 3.1 M1 (financial/macro baseline, 10 variables)

**Direction (classification)**


| model              | accuracy  | macro_f1  | directional_acc |
| ------------------ | --------- | --------- | --------------- |
| LogisticRegression | 0.462     | 0.308     | 0.493           |
| **RandomForest**   | **0.497** | **0.336** | **0.529**       |
| SVM                | 0.483     | 0.331     | 0.515           |
| XGBoost            | 0.455     | 0.288     | 0.485           |
| LSTM               | 0.476     | 0.303     | 0.507           |
| TFT                | 0.434     | 0.261     | 0.463           |
| ST-GNN             | 0.483     | 0.312     | 0.515           |


**Volatility (regression)**


| model            | RMSE        | MAE         | R²        |
| ---------------- | ----------- | ----------- | --------- |
| Ridge            | 0.00895     | 0.00666     | -0.146    |
| **RandomForest** | **0.00793** | **0.00614** | **0.101** |
| SVM              | 0.08924     | 0.08867     | -112.88   |
| XGBoost          | 0.00804     | 0.00617     | 0.076     |
| LSTM             | 0.00900     | 0.00681     | -0.197    |
| TFT              | 0.00936     | 0.00737     | -0.295    |
| ST-GNN           | 0.02640     | 0.02279     | -9.300    |


**Price (regression)**


| model        | RMSE      | MAE       | R²        |
| ------------ | --------- | --------- | --------- |
| **Ridge**    | **3.049** | **2.379** | **0.867** |
| RandomForest | 3.424     | 2.632     | 0.833     |
| SVM          | 5.402     | 4.483     | 0.583     |
| XGBoost      | 3.405     | 2.644     | 0.835     |
| LSTM         | 9.498     | 7.839     | -0.272    |
| TFT          | 15.05     | 12.89     | -2.195    |
| ST-GNN       | 12.58     | 10.55     | -1.231    |


### 3.2 M2 (M1 + remote sensing, 22 variables)

**Direction (classification)**


| model              | accuracy  | macro_f1  | directional_acc |
| ------------------ | --------- | --------- | --------------- |
| LogisticRegression | 0.377     | 0.236     | 0.406           |
| RandomForest       | 0.420     | 0.257     | 0.453           |
| SVM                | 0.449     | 0.226     | 0.484           |
| XGBoost            | 0.435     | 0.301     | 0.469           |
| LSTM               | 0.448     | 0.206     | 0.484           |
| TFT                | 0.448     | 0.206     | 0.484           |
| **ST-GNN**         | **0.469** | **0.322** | **0.500**       |


**Volatility (regression)**


| model        | RMSE        | MAE         | R²         |
| ------------ | ----------- | ----------- | ---------- |
| Ridge        | 0.01187     | 0.00892     | -1.190     |
| RandomForest | 0.00876     | 0.00633     | -0.193     |
| SVM          | 0.08957     | 0.08917     | -123.70    |
| **XGBoost**  | **0.00862** | **0.00681** | **-0.156** |
| LSTM         | 0.01183     | 0.01026     | -1.216     |
| TFT          | 0.02037     | 0.01687     | -5.574     |
| ST-GNN       | 0.02133     | 0.01850     | -5.723     |


**Price (regression)**


| model        | RMSE      | MAE       | R²        |
| ------------ | --------- | --------- | --------- |
| **Ridge**    | **3.160** | **2.429** | **0.643** |
| RandomForest | 3.446     | 2.562     | 0.576     |
| SVM          | 5.716     | 4.969     | -0.167    |
| XGBoost      | 3.809     | 2.673     | 0.482     |
| LSTM         | 5.037     | 4.309     | 0.113     |
| TFT          | 6.458     | 5.674     | -0.458    |
| ST-GNN       | 11.94     | 9.386     | -1.011    |


### 3.3 M3 (M1 + shipping, 29 variables) ⚠️ 2019+ window

**Direction (classification)**


| model              | accuracy  | macro_f1  | directional_acc |
| ------------------ | --------- | --------- | --------------- |
| LogisticRegression | 0.491     | 0.347     | 0.540           |
| RandomForest       | 0.509     | 0.332     | 0.560           |
| SVM                | 0.509     | 0.314     | 0.560           |
| **XGBoost**        | **0.564** | **0.391** | **0.620**       |
| LSTM               | 0.426     | 0.199     | 0.469           |
| TFT                | 0.426     | 0.199     | 0.469           |
| ST-GNN             | 0.441     | 0.284     | 0.470           |


**Volatility (regression)**


| model            | RMSE        | MAE         | R²         |
| ---------------- | ----------- | ----------- | ---------- |
| Ridge            | 0.01302     | 0.01016     | -1.633     |
| **RandomForest** | **0.00895** | **0.00625** | **-0.243** |
| SVM              | 0.09402     | 0.09364     | -136.20    |
| XGBoost          | 0.01045     | 0.00757     | -0.694     |
| LSTM             | 0.01471     | 0.01193     | -2.325     |
| TFT              | 0.06531     | 0.05399     | -64.53     |
| ST-GNN           | 0.01539     | 0.01298     | -2.500     |


**Price (regression)**


| model        | RMSE      | MAE       | R²        |
| ------------ | --------- | --------- | --------- |
| **Ridge**    | **2.804** | **2.243** | **0.704** |
| RandomForest | 3.285     | 2.468     | 0.594     |
| SVM          | 9.195     | 8.025     | -2.178    |
| XGBoost      | 3.204     | 2.398     | 0.614     |
| LSTM         | 70.27     | 69.37     | -183.71   |
| TFT          | 11.57     | 10.39     | -4.005    |
| ST-GNN       | 12.34     | 10.20     | -1.148    |


### 3.4 M4 (M1 + remote sensing + shipping, 41 variables) ⚠️ 2019+ window

**Direction (classification)**


| model              | accuracy  | macro_f1  | directional_acc |
| ------------------ | --------- | --------- | --------------- |
| LogisticRegression | 0.455     | 0.317     | 0.500           |
| RandomForest       | 0.527     | 0.343     | 0.580           |
| SVM                | 0.418     | 0.202     | 0.460           |
| XGBoost            | 0.491     | 0.343     | 0.540           |
| LSTM               | 0.444     | 0.260     | 0.490           |
| **TFT**            | **0.537** | **0.373** | **0.592**       |
| ST-GNN             | 0.455     | 0.208     | 0.485           |


**Volatility (regression)**


| model            | RMSE        | MAE         | R²         |
| ---------------- | ----------- | ----------- | ---------- |
| Ridge            | 0.01332     | 0.01033     | -1.755     |
| **RandomForest** | **0.00895** | **0.00633** | **-0.243** |
| SVM              | 0.09485     | 0.09448     | -138.62    |
| XGBoost          | 0.01094     | 0.00831     | -0.857     |
| LSTM             | 0.01378     | 0.01137     | -1.916     |
| TFT              | 0.01736     | 0.01569     | -3.628     |
| ST-GNN           | 0.01030     | 0.00808     | -0.567     |


**Price (regression)**


| model        | RMSE      | MAE       | R²        |
| ------------ | --------- | --------- | --------- |
| **Ridge**    | **3.029** | **2.511** | **0.655** |
| RandomForest | 3.288     | 2.500     | 0.594     |
| SVM          | 9.975     | 8.800     | -2.740    |
| XGBoost      | 3.143     | 2.363     | 0.629     |
| LSTM         | 68.80     | 66.35     | -176.02   |
| TFT          | 10.33     | 9.233     | -2.990    |
| ST-GNN       | 9.840     | 8.074     | -0.365    |


---

## 4. Summary

- **Direction:** best are XGBoost-M3 (acc 0.564 / dir-acc 0.620) and TFT-M4 (acc 0.537 / dir-acc 0.592); overall close to the random baseline, with generally low macro_f1 (the flat class often has no recall).
- **Volatility:** only a few models reach R²>0 (RandomForest-M1=0.101, XGBoost-M1=0.076); most are negative — hard to predict.
- **Price:** high R² (Ridge-M1=0.867, XGBoost-M1=0.835), but this is **driven by price persistence**; a random-walk/Naive benchmark has not yet been added, so it cannot be read directly as predictive skill (see caveat §7 in `M1_pipeline.md`).
- **Cross-layer comparison:** M3/M4 are evaluated on 2019+ only (N≈54), not directly comparable with the full-sample M1/M2; the window difference must be noted, and DM tests are only valid on the aligned sample.

