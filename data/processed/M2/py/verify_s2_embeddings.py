"""
Channel A — sanity-check the frozen Prithvi S2 embeddings before Part B.

Confirms the precomputed embeddings are usable as an RS modality input:
  1. Non-degenerate      : few near-constant dims, sensible norms, no NaN.
  2. Site-separable       : within-site cosine > between-site cosine (embeddings
                            carry geographic structure -> encoder can learn a
                            site-aware representation).
  3. Temporal signal      : same-site adjacent months are MORE similar than
                            random same-site pairs, i.e. embeddings move slowly /
                            seasonally rather than being frozen per site (needed
                            for any activity signal the fusion model could use).

Reads   data/processed/M2/outputs/s2_prithvi_emb_meanpool.npy + _index.csv
Outputs data/processed/M2/outputs/s2_prithvi_emb_check.png + printed report.

Run: python3 data/processed/M2/py/verify_s2_embeddings.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize

ROOT = Path(__file__).resolve().parents[4]
OUT = ROOT / "data/processed/M2/outputs"
TYPE_COLOR = {"port": "tab:blue", "refinery": "tab:orange", "terminal": "tab:green"}


def main() -> None:
    emb = np.load(OUT / "s2_prithvi_emb_meanpool.npy")
    idx = pd.read_csv(OUT / "s2_prithvi_emb_index.csv")
    idx["date"] = pd.to_datetime(idx["obs_month_start"])
    N, D = emb.shape
    print(f"embeddings : {emb.shape}  finite={np.isfinite(emb).all()}")

    # 1. degeneracy ----------------------------------------------------------
    dim_std = emb.std(axis=0)
    norms = np.linalg.norm(emb, axis=1)
    print("\n[1] degeneracy")
    print(f"  near-constant dims (std<1e-4): {int((dim_std < 1e-4).sum())}/{D}")
    print(f"  per-vector L2 norm  min/mean/max = {norms.min():.2f}/{norms.mean():.2f}/{norms.max():.2f}")

    # 2. site separability (cosine) -----------------------------------------
    E = normalize(emb)                      # L2-normalised rows
    S = E @ E.T                              # cosine similarity matrix
    sites = idx["site_name"].to_numpy()
    within, between = [], []
    for i in range(N):
        same = sites == sites[i]
        same[i] = False
        within.append(S[i, same].mean())
        between.append(S[i, ~same].mean())
    within, between = np.array(within), np.array(between)
    print("\n[2] site separability (cosine)")
    print(f"  within-site  mean = {within.mean():.3f}")
    print(f"  between-site mean = {between.mean():.3f}")
    print(f"  gap (within-between) = {within.mean()-between.mean():.3f}  (>0 = sites separable)")

    # 3. temporal signal -----------------------------------------------------
    adj, rnd = [], []
    rng = np.random.default_rng(0)
    for s in np.unique(sites):
        rows = idx.index[idx["site_name"] == s].to_numpy()
        rows = rows[np.argsort(idx.loc[rows, "date"].to_numpy())]
        if len(rows) < 3:
            continue
        for a, b in zip(rows[:-1], rows[1:]):
            adj.append(float(E[a] @ E[b]))
        for _ in range(len(rows)):                # equal # of random same-site pairs
            i, j = rng.choice(rows, size=2, replace=False)
            rnd.append(float(E[i] @ E[j]))
    adj, rnd = np.array(adj), np.array(rnd)
    print("\n[3] temporal signal (same-site cosine)")
    print(f"  adjacent-month mean = {adj.mean():.3f}")
    print(f"  random-pair    mean = {rnd.mean():.3f}")
    print(f"  gap (adj-random) = {adj.mean()-rnd.mean():.3f}  (>0 = smooth/seasonal drift, not frozen)")

    # ---- figure ------------------------------------------------------------
    pca = PCA(n_components=2, random_state=0).fit(emb)
    p2 = pca.transform(emb)
    evr = pca.explained_variance_ratio_

    fig, ax = plt.subplots(2, 3, figsize=(17, 9))

    a = ax[0, 0]
    for s in np.unique(sites):
        m = sites == s
        a.scatter(p2[m, 0], p2[m, 1], s=8, alpha=0.6, label=s)
    a.set_title(f"PCA by site  (EVR={evr[0]:.2f}/{evr[1]:.2f})")
    a.legend(fontsize=6, ncol=2, markerscale=1.5)
    a.set_xlabel("PC1"); a.set_ylabel("PC2")

    a = ax[0, 1]
    for t, c in TYPE_COLOR.items():
        m = idx["site_type"].to_numpy() == t
        a.scatter(p2[m, 0], p2[m, 1], s=8, alpha=0.5, color=c, label=t)
    a.set_title("PCA by site type"); a.legend(fontsize=8); a.set_xlabel("PC1"); a.set_ylabel("PC2")

    a = ax[0, 2]
    a.hist(between, bins=40, alpha=0.6, label="between-site", color="grey", density=True)
    a.hist(within, bins=40, alpha=0.6, label="within-site", color="tab:red", density=True)
    a.set_title("Cosine sim: within vs between site"); a.legend(fontsize=8); a.set_xlabel("cosine")

    a = ax[1, 0]
    a.hist(rnd, bins=40, alpha=0.6, label="random same-site pair", color="grey", density=True)
    a.hist(adj, bins=40, alpha=0.6, label="adjacent month", color="tab:purple", density=True)
    a.set_title("Cosine sim: adjacent vs random month"); a.legend(fontsize=8); a.set_xlabel("cosine")

    a = ax[1, 1]
    for s in ["Houston", "RasTanura"]:
        rows = idx.index[idx["site_name"] == s].to_numpy()
        rows = rows[np.argsort(idx.loc[rows, "date"].to_numpy())]
        a.plot(idx.loc[rows, "date"].to_numpy(), p2[rows, 0], marker=".", ms=3, lw=0.8, label=s)
    a.set_title("PC1 over time (per site)"); a.legend(fontsize=8); a.set_xlabel("date"); a.set_ylabel("PC1")
    a.tick_params(axis="x", rotation=30)

    a = ax[1, 2]
    a.hist(dim_std, bins=50, color="tab:cyan")
    a.axvline(1e-4, color="red", ls="--", lw=0.8)
    a.set_title(f"Per-dim std over {N} patches"); a.set_xlabel("std"); a.set_ylabel("# dims")

    fig.suptitle("Prithvi-EO-2.0-300M frozen S2 embeddings — sanity checks", fontsize=13)
    fig.tight_layout()
    path = OUT / "s2_prithvi_emb_check.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)

    verdict = (within.mean() > between.mean()) and (adj.mean() > rnd.mean()) and \
              (int((dim_std < 1e-4).sum()) < D * 0.2)
    print(f"\nVERDICT: {'PASS' if verdict else 'REVIEW'} "
          f"(separable={within.mean()>between.mean()}, temporal={adj.mean()>rnd.mean()})")
    print(f"Saved: {path}")


if __name__ == "__main__":
    main()
