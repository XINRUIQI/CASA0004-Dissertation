# A Modality-Aware Spatiotemporal Fusion Framework for Brent Crude Oil Forecasting

**Using Maritime Networks, Satellite Imagery and Financial Time Series**

CASA0010 MSc dissertation · [Centre for Advanced Spatial Analysis (CASA)](https://www.ucl.ac.uk/bartlett/casa), Bartlett Faculty of the Built Environment, UCL

**Xinrui Qi** · Supervisor: Beatrice Taylor · 15 August 2026

---

## Overview

Brent crude is a principal benchmark for internationally traded oil. Its short-term movements affect energy costs, inflation, trade balances and fiscal revenues. Using weekly Friday-ending data from 2019 to 2025, this dissertation asks whether satellite remote sensing and maritime shipping add **incremental predictive value** beyond financial time series for **one-week-ahead Brent price forecasting**, and whether that value depends on **how** the data are combined.

Two modelling pathways share the same information sets, forecast dates and evaluation rules:

- **Flat feature fusion** concatenates all selected inputs into a weekly table and fits Ridge and XGBoost.
- **Deep (representation-level) fusion** encodes each modality separately — a temporal convolutional network for finance, frozen Prithvi-EO-2.0 embeddings with attention for remote sensing, and a graph attention network for shipping — then fuses the representations. Gated fusion is the main Deep specification.

Every learned model is compared with a **no-change benchmark (M0)** that sets next week’s price equal to this week’s price. At the weekly horizon this is a demanding reference: a model is not treated as useful merely because it beats a weaker competitor.

The results show that adding more data or using a more complex model does not guarantee better forecasts. No Flat model outperforms M0. Remote sensing provides no consistent improvement. Shipping provides clearer incremental information, mainly in the Deep pathway, where the finance-plus-shipping specification records a small improvement over M0. Across matched information sets, Deep models outperform Flat models. Predictive value therefore depends more on encoding and fusion than on simply adding spatial features. Forecasts still rely mainly on financial and oil-market information; shipping is a smaller, period-dependent contribution.

The practical contribution is a **reproducible comparison framework**: each new data source is judged both against a finance-only model and against the no-change rule, so incremental information is distinguished from overall forecast skill.

---

## Research questions

**RQ1.** Compared with models using only financial time-series data, do remote-sensing and shipping data improve one-week-ahead Brent price forecasts?

**RQ2.** When using the same information set, does modality-aware representation-level fusion outperform flat feature fusion?

**RQ3.** Which data sources and spatial nodes do the models rely on, and how does this reliance vary across forecast periods?

---

## Research design

Figure 3.1. Research design and forecasting workflow.

*Figure 3.1. Research design and forecasting workflow.*

### Information sets


| Set                | Inputs                                                                      |
| ------------------ | --------------------------------------------------------------------------- |
| **M0** (benchmark) | Last week’s Brent price                                                     |
| **S1**             | Financial time series only (financial, macroeconomic and oil-market series) |
| **S2**             | S1 + remote sensing                                                         |
| **S3**             | S1 + shipping                                                               |
| **S4**             | S1 + remote sensing + shipping                                              |


S2 and S3 are parallel extensions of S1. Comparisons across S1–S4, together with each model versus M0, address RQ1. Matched Flat–Deep comparisons on the same sets address RQ2. Interpretation (RQ3) is reported for Deep specifications that improve on M0.

### Target, sample and protocol

Models predict the one-week log return $r_{t+1}=\log(P_{t+1}/P_t)$ and reconstruct the price forecast as $\hat{P}*{t+1\mid t}=P_t\exp(\hat{r}*{t+1\mid t})$. Under this mapping, M0 is exactly a zero-return forecast. RMSE and $\Delta\mathrm{RMSE}$ (percentage RMSE improvement over M0) are computed from reconstructed prices.


| Item                                                        | Locked value                                                       |
| ----------------------------------------------------------- | ------------------------------------------------------------------ |
| Calendar                                                    | Friday-ending weeks, 4 January 2019 – 26 December 2025 (365 weeks) |
| Lookback / initial train / retrain every / inner validation | 4 / 104 / 13 / 52 weeks                                            |
| Evaluation sample                                           | **257** origins, 22 January 2021 – 19 December 2025                |
| Main seed                                                   | 42 (robustness: 1, 2)                                              |
| Primary metric                                              | RMSE and $\Delta\mathrm{RMSE}$ vs M0                               |
| Secondary tests                                             | Diebold–Mariano (Harvey–Leybourne–Newbold) vs M0; Clark–West vs S1 |


Hyperparameters, encoder sizes and early-stopping settings: [Appendix C](docs/appendix/appendix_C_config_EN.md). Robustness (fusion matrix, seeds): [Appendix B](docs/appendix/appendix_B_robustness_EN.md).

---

## Data

The target is the **global Brent benchmark**, not a local cargo price, so the study does not use a single study region. Spatial inputs come from **11 oil-infrastructure sites** (ports, refineries and export terminals) and **6 maritime chokepoints** (Strait of Hormuz, Suez Canal, Strait of Malacca, Bab el-Mandeb, Panama Canal, Cape of Good Hope). The Deep shipping graph has **17 nodes**, with weekly dynamic voyage edges among the 11 sites and fixed corridor edges from sites to chokepoints on documented oil-trade routes.

Figure 3.3. Spatial distribution of the 11 AOIs and 6 chokepoints.

*Figure 3.3. Spatial distribution of the 11 AOIs and 6 chokepoints.*


| Block          | Flat representation                                                        | Deep representation                                                                                                      | Sources                                                                                                                                                                                                                                               |
| -------------- | -------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Finance        | Weekly tabular series                                                      | Four-week sequences → TCN                                                                                                | [EIA](https://www.eia.gov/petroleum/supply/weekly/), [FRED](https://fred.stlouisfed.org/), [Yahoo Finance](https://finance.yahoo.com/), [Dallas Fed IGREA](https://www.dallasfed.org/research/igrea), [GPR](https://www.matteoiacoviello.com/gpr.htm) |
| Remote sensing | Site-level Sentinel-2 index and VIIRS night-light **anomalies** at 11 AOIs | Frozen [Prithvi-EO-2.0](https://huggingface.co/ibm-nasa-geospatial) embeddings from Sentinel-2 patches (no VIIRS stream) | Sentinel-2 and VIIRS via Google Earth Engine                                                                                                                                                                                                          |
| Shipping       | PortWatch tanker flows and GFW vessel-activity features                    | 17-node graph: node attributes, dynamic voyage edges, fixed corridor edges                                               | [IMF PortWatch](https://portwatch.imf.org/), [Global Fishing Watch](https://globalfishingwatch.org/our-apis/)                                                                                                                                         |


All series are aligned to the Friday weekly calendar with source-specific **publication-lag buffers** so that each forecast uses only information treated as available at the origin. Variable dictionaries, lags, AOIs and graph edges: [Appendix A](docs/appendix/appendix_A_data_EN.md). Source licences and download notes: `[data/sources.md](data/sources.md)`.

Raw downloads live in `data/raw/` (gitignored). Modelling reads the processed weekly products already in the repository.

---

## Main findings

Out-of-sample $n = 257$. Positive $\Delta\mathrm{RMSE}$ means lower RMSE than M0 (RMSE = 4.152 USD/barrel). Gated fusion is the prespecified Deep specification.


| Set                 | Best Flat RMSE  | Main Deep RMSE | $\Delta\mathrm{RMSE}$ (Deep vs M0) |
| ------------------- | --------------- | -------------- | ---------------------------------- |
| S1 finance          | 4.256 (Ridge)   | 4.250          | −2.36%                             |
| S2 + remote sensing | 4.414 (Ridge)   | 4.253          | −2.43%                             |
| S3 + shipping       | 4.357 (XGBoost) | **4.146**      | **+0.15%**                         |
| S4 + both           | 4.412 (XGBoost) | 4.180          | −0.67%                             |


- **RQ1.** No Flat model beats M0. Remote sensing does not improve accuracy when added alone or on top of shipping. Shipping is the more informative spatial source; only Deep S3 (and the secondary cross-attention S3/S4 runs) records a positive \Delta\mathrm{RMSE} in the main tables.
- **RQ2.** On every matched set S1–S4, the main Deep pathway has lower RMSE than both Ridge and XGBoost. The gap is largest when shipping is included. These are comparisons of complete modelling pathways, not of fusion operators in isolation.
- **RQ3.** For gated Deep S3, SHAP still attributes most of the prediction to financial and EIA inputs (about 97%). Shipping’s share is small but not uniform: it rises during the Red Sea window and the model’s spatial focus shifts toward Suez, Bab el-Mandeb and the Cape route in 2024. Gate weights and SHAP describe model attribution, not causal price drivers.

Interpretation: monitoring value and predictive value should be assessed separately. Remote sensing may be better suited to facility- or production-scale monitoring; shipping is better suited to disruptions and rerouting on a global transport network. Spatial data can still inform energy-security and trade monitoring even when they do not beat a weekly no-change forecast.

---

## Repository layout

```text
.
├── README.md
├── code/                     modelling (see code/README.md)
│   ├── requirements.txt
│   ├── src/                  backtest, encoders, fusion
│   └── scripts/              flat / deep / figures / tools
├── data/
│   ├── sources.md            licences and download notes
│   ├── processed/            weekly matrix, Prithvi embeddings, 17-node graph
│   └── raw/                  local downloads (gitignored except scripts/AOI)
├── results/                  metrics, predictions, figures
└── docs/
    ├── reproducibility.md
    └── appendix/             A–D (data, robustness, hyperparameters, log)
```

### Files required to reproduce the main tables


| File                                                                                   | Role                                              |
| -------------------------------------------------------------------------------------- | ------------------------------------------------- |
| `data/processed/merge/outputs/weekly_feature_matrix.csv`                               | Shared weekly matrix for Flat and Deep            |
| `data/processed/merge/outputs/weekly_feature_dictionary.csv`                           | Feature dictionary                                |
| `data/processed/M3/outputs/m3_graph17_tensors.npz`                                     | Deep 17-node shipping graph                       |
| `data/processed/M2/outputs/s2_prithvi_emb_meanpool.npy` and `s2_prithvi_emb_index.csv` | Frozen Prithvi embeddings                         |
| `results/baselines/Flat/M1_Flat/baseline_predictions.csv`                              | Flat S1 predictions (Deep scripts read this file) |


---

## Reproduce the main tables

Processed weekly matrices and Deep tensors are in the repository. Reproducing the reported tables does **not** require re-downloading raw data, installing `transformers`, or re-running Prithvi.

```bash
# 0) Environment (Python 3.9.x; tested 3.9.6 / macOS)
cd "/path/to/casa0004 Dissertation"
python3 -m venv .venv && source .venv/bin/activate
python3 -m pip install -r code/requirements.txt

# 1) Flat S1–S4 (run S1 / M1 first: Deep scripts read its predictions)
python3 code/scripts/flat/run_baseline.py --modality M1
python3 code/scripts/flat/run_baseline.py --modality M2 --m2-features anom
python3 code/scripts/flat/run_baseline.py --modality M3
python3 code/scripts/flat/run_baseline.py --modality M4

# 2) Deep main results (gated fusion; representation-level)
python3 code/scripts/deep/run_deep_baseline.py

# 3) Optional: sub-period tables, fusion matrix, interpretability
python3 code/scripts/tools/subperiod_eval.py
python3 code/scripts/deep/run_deep_fusion_matrix.py
python3 code/scripts/deep/run_deep_interpret.py --seeds 42,1,2 --lookback 4
```

Outputs: `results/baselines/Flat/M*_Flat/` and `results/baselines/Deep/_cross/`.

Full command list, flags, robustness scripts and optional rebuild-from-raw: `[code/README.md](code/README.md)`. Submission-facing checklist: `[docs/reproducibility.md](docs/reproducibility.md)`.

Rebuild from raw is optional and needs local `data/raw/`. Prithvi embedding export is a one-off offline step outside this `requirements.txt` environment.

---

## Software environment

Python **3.9.6** (CPython, macOS). Pinned in `[code/requirements.txt](code/requirements.txt)`. Deep training and evaluation run on **CPU** by default.


| Package      | Version | Role                           |
| ------------ | ------- | ------------------------------ |
| numpy        | 2.0.2   | arrays                         |
| pandas       | 2.3.3   | weekly matrix                  |
| scipy        | 1.13.1  | *p* values                     |
| scikit-learn | 1.6.1   | Ridge, scaling                 |
| xgboost      | 2.1.4   | Flat XGBoost                   |
| torch        | 2.8.0   | Deep encoders and fusion (CPU) |
| matplotlib   | 3.9.4   | figures                        |
| shap         | 0.49.1  | attribution                    |


The Deep remote-sensing branch uses pre-computed frozen Prithvi-EO-2.0 embeddings. Training and evaluation do not load the foundation model (`transformers` is not required). On macOS, run Flat and Deep in separate processes to avoid an OpenMP conflict between `xgboost` and `torch`.

---

## Ethics and data use

The study uses only secondary, aggregate data and does not involve human participants. It received approval through UCL’s low-risk ethics process. Datasets were used under their published licences and terms, including the Copernicus open licence for Sentinel-2 and the research-use terms of IMF PortWatch and Global Fishing Watch. Remote-sensing and vessel-activity variables are analysed only at site or chokepoint level; the code does not identify individual vessels, operators or persons.

---

## Document index


| Purpose                              | Path                                                                                     |
| ------------------------------------ | ---------------------------------------------------------------------------------------- |
| Code runbook                         | `[code/README.md](code/README.md)`                                                       |
| Reproducibility checklist            | `[docs/reproducibility.md](docs/reproducibility.md)`                                     |
| Appendix A — data, AOIs, lags, graph | `[docs/appendix/appendix_A_data_EN.md](docs/appendix/appendix_A_data_EN.md)`             |
| Appendix B — robustness              | `[docs/appendix/appendix_B_robustness_EN.md](docs/appendix/appendix_B_robustness_EN.md)` |
| Appendix C — hyperparameters         | `[docs/appendix/appendix_C_config_EN.md](docs/appendix/appendix_C_config_EN.md)`         |
| External data sources                | `[data/sources.md](data/sources.md)`                                                     |


