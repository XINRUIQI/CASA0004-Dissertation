# CASA0004 Dissertation — Reproduction Entry Point

**Working title**: A Modality-Aware Spatio-Temporal Fusion Framework for Brent Crude Oil Forecasting Using Financial Time Series, Satellite Imagery and Maritime Networks

**One-liner**: Under a shared leakage-safe rolling-origin protocol, compare flat feature concatenation (Flat) vs representation-level multimodal fusion (Deep); test whether remote sensing / shipping add out-of-sample value over a finance baseline and a random-walk benchmark; and interpret modality and spatial-node importance.

---

## Quick start (recommended)

Processed weekly matrices and deep tensors are already in the repo. Reproducing the main tables does **not** require re-downloading raw data or installing `transformers` / re-running Prithvi.

```bash
# 0) Environment (Python 3.9.x; tested 3.9.6 / macOS)
cd "/path/to/casa0004 Dissertation"
python3 -m venv .venv && source .venv/bin/activate
python3 -m pip install -r 04_code/requirements.txt

# 1) Flat baselines M0–M4 (run M1 first: deep scripts read its predictions)
python3 04_code/scripts/flat/run_baseline.py --modality M1
python3 04_code/scripts/flat/run_baseline.py --modality M2 --m2-features anom
python3 04_code/scripts/flat/run_baseline.py --modality M3
python3 04_code/scripts/flat/run_baseline.py --modality M4

# 2) Deep main results (representation-level fusion)
python3 04_code/scripts/deep/run_deep_baseline.py

# 3) Optional: sub-period / fusion matrix / interpretability
python3 04_code/scripts/tools/subperiod_eval.py
python3 04_code/scripts/deep/run_deep_fusion_matrix.py
python3 04_code/scripts/deep/run_deep_interpret.py --seeds 42,1,2 --lookback 4
```

**Full command list, prerequisites, robustness scripts** → `[04_code/README.md](04_code/README.md)`  
**Submission reproducibility pack index** → `[07_submission/reproducibility_pack/README.md](07_submission/reproducibility_pack/README.md)`

---



## Research questions


| RQ      | Question                                                                | Method                            |
| ------- | ----------------------------------------------------------------------- | --------------------------------- |
| **RQ1** | Do RS / shipping add OOS value over M1 and relative to M0?              | Flat M0–M4; CW vs M1; DM vs M0    |
| **RQ2** | Under the same data, does representation-level fusion beat flat concat? | Flat vs Deep paired comparison    |
| **RQ3** | Relative importance of modalities / spatial nodes across regimes?       | Flat SHAP; Deep gate α, attention |


---



## Project layout

```text
casa0004 Dissertation/
├── 00_admin/                 # diaries, meetings, plans, walkthroughs
├── 01_literature/            # matrix, reading notes, PDFs (local)
├── 02_ai_conversations/      # AI usage log
├── 03_data/
│   ├── raw/                  # raw inputs (gitignore; keep locally)
│   ├── processed/            # M1/M2/M3 + merge weekly products (modelling reads these)
│   └── Dataset/              # external_sources.md, etc.
├── 04_code/
│   ├── requirements.txt
│   ├── scripts/flat|deep|tools/
│   └── src/backtest|models/
├── 05_outputs/baselines/     # Flat / Deep / subperiod results
├── 06_writing/               # chapters, outline, appendices
└── 07_submission/            # final submission + reproducibility_pack
```



### Key files required for modelling


| File                                                                                          | Role                                            |
| --------------------------------------------------------------------------------------------- | ----------------------------------------------- |
| `03_data/processed/merge/outputs/weekly_feature_matrix.csv`                                   | Shared weekly matrix (~365×213) for Flat + Deep |
| `03_data/processed/merge/outputs/weekly_feature_dictionary.csv`                               | Feature dictionary (modality / names)           |
| `03_data/processed/M3/outputs/m3_graph17_tensors.npz`                                         | Deep 17-node shipping graph                     |
| `03_data/processed/M2/outputs/s2_prithvi_emb_{meanpool,cls}.npy` + `s2_prithvi_emb_index.csv` | Frozen Prithvi embeddings (precomputed)         |
| `05_outputs/baselines/Flat/M1_Flat/baseline_predictions.csv`                                  | Flat M1 predictions read by deep scripts        |


---



## Locked protocol (Flat = Deep)


| Item                                             | Value                                                      |
| ------------------------------------------------ | ---------------------------------------------------------- |
| Window                                           | 2019–2025 (merged matrix); scored test ≈ **257 weeks**     |
| lookback / min_train / retrain_every / val_weeks | **4 / 104 / 13 / 52**                                      |
| Target                                           | r_{t+1}=\ln(P_{t+1}/P_t), reconstructed to next-week price |
| Seed                                             | Main analysis **42**                                       |
| Metrics / tests                                  | RMSE · MAE · DirAcc · skill vs M0; DM (HLN); Clark–West    |


Details: `[06_writing/Appendix/appendix_C_config.md](06_writing/Appendix/appendix_C_config.md)`.

---



## Document index

| Purpose | Path |
| --- | --- |
| **Code runbook** | [`04_code/README.md`](04_code/README.md) |
| **Repro pack index** | [`07_submission/reproducibility_pack/README.md`](07_submission/reproducibility_pack/README.md) |
| Flat end-to-end walkthrough | [`00_admin/flat_baseline_full_walkthrough.md`](00_admin/flat_baseline_full_walkthrough.md) |
| Deep end-to-end walkthrough | [`00_admin/deep_model_full_walkthrough.md`](00_admin/deep_model_full_walkthrough.md) |
| Flat variable inventory | [`00_admin/flat_baseline_variable_list.md`](00_admin/flat_baseline_variable_list.md) |
| Directory map | [`00_admin/File Structure.md`](00_admin/File%20Structure.md) |
| Data / config appendices | `06_writing/Appendix/appendix_{A_data,C_config}.md` |
| External sources | [`03_data/Dataset/external_sources.md`](03_data/Dataset/external_sources.md) |


---



## Rebuild from raw (optional)

Only if you need to redo feature engineering; requires local `03_data/raw/` (not in git). Order: see `[04_code/README.md](04_code/README.md)` § Data rebuild. Prithvi embedding export is a **one-off offline** step outside this `requirements.txt` environment.

---



## Conventions

- Preferred filenames: `YYYY-MM-DD_topic_version.ext`
- `raw/` is read-only; modelling reads `processed/`; results go to `05_outputs/`
- Claims in the thesis need formal sources

