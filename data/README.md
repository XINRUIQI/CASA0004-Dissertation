# Data

Modelling reads **processed** weekly products already in this folder. Reproducing the main tables does not require `raw/`.

| Path | Role |
| --- | --- |
| `processed/merge/outputs/weekly_feature_matrix.csv` | Shared weekly matrix (Flat and Deep) |
| `processed/merge/outputs/weekly_feature_dictionary.csv` | Feature dictionary |
| `processed/M3/outputs/m3_graph17_tensors.npz` | Deep 17-node shipping graph |
| `processed/M2/outputs/s2_prithvi_emb_meanpool.npy` | Frozen Prithvi embeddings |
| `processed/M2/outputs/s2_prithvi_emb_index.csv` | Embedding index |
| `sources.md` | External sources, licences, download notes |

`M1` / `M2` / `M3` are **information-set labels** (finance, remote sensing, shipping), not folder numbering.

Rebuild-from-raw (optional) uses scripts under `processed/*/py/` and `raw/` (local downloads). Layout: [`raw/README.md`](raw/README.md).
