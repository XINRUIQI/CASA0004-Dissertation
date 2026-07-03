"""
Channel A — precompute FROZEN Prithvi-EO-2.0 image embeddings for S2 patches.

Part A of the modality-aware fusion plan (2026-06-22_research_plan_e2e_multimodal.md
§4.2 / §5.1 EO encoder / 阶段1-1). Turns every usable monthly Sentinel-2 patch into
a single fixed-length image embedding using a frozen EO foundation model, so the
downstream RS temporal encoder (Part B) only trains a light head on top -- never the
600M-parameter backbone (small-sample risk mitigation, plan §8).

Why Prithvi-EO-2.0-300M
  The patches were exported with bands B2/B3/B4/B8A/B11/B12, which are exactly the
  6 HLS bands Prithvi expects (B02 Blue, B03 Green, B04 Red, B05 NIR-narrow,
  B06 SWIR1, B07 SWIR2) -- same physical bands, same order. So no band remapping is
  needed; we reuse the model's own mean/std from config.json.

Pipeline (per patch, single time step T=1):
  read 6-band GeoTIFF -> treat 0 as nodata/mask-edge -> standardise (x-mean)/std with
  nodata->1e-4 (Prithvi convention) -> bilinear resize to 224x224 on CPU (avoids MPS
  antialias gaps) -> Prithvi encoder.forward_features (NO masking) -> last block
  (post-norm) [B, 197, 1024] -> mean-pool over the 196 patch tokens (main embedding)
  + keep the cls token (alt embedding).

Resize (not tiling) rationale: we want ONE global "site activity" embedding per
patch. Differentiated patch sizes (port 6.4km / refinery 5.12km / terminal 2.56km)
mean resize-to-224 lands each site near Prithvi's ~30m HLS training GSD
(port ~29 m/px, terminal ~14 m/px). Tiling is left as a later robustness option.

Frozen: model.eval() + requires_grad_(False) + torch.no_grad(); backbone is never
updated here or in Part B.

Only patches with valid_mask==1 (from s2_patch_index.csv) are embedded; the emb rows
line up 1:1 with the emitted index csv, so missing (site, month) cells stay explicitly
missing for Part B's as-of alignment (no ffill).

Outputs (-> 03_data/processed/M2/outputs/):
  s2_prithvi_emb_meanpool.npy    [N, 1024] float32  (main: mean of patch tokens)
  s2_prithvi_emb_cls.npy         [N, 1024] float32  (alt: cls token)
  s2_prithvi_emb_index.csv       N rows aligned to the .npy row order + metadata
  s2_prithvi_emb_coverage.csv    per-site embedded-month counts

Run:
  # smoke test on 4 patches first:
  python3 03_data/processed/M2/py/precompute_s2_embeddings.py --limit 4
  # full run (967 usable patches):
  python3 03_data/processed/M2/py/precompute_s2_embeddings.py
  # force cpu / change batch:
  python3 03_data/processed/M2/py/precompute_s2_embeddings.py --device cpu --batch-size 8

Requires: torch, timm, einops, rasterio, huggingface_hub, pandas, numpy.
"""
from __future__ import annotations

import os

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")  # let unsupported ops fall back

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
import torch
import torch.nn.functional as F
from huggingface_hub import hf_hub_download

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from s2_patch_utils import PATCH_DIR  # noqa: E402  (shared patch dir)

ROOT = Path(__file__).resolve().parents[4]
OUT = ROOT / "03_data/processed/M2/outputs"
INDEX_CSV = OUT / "s2_patch_index.csv"

REPO = "ibm-nasa-geospatial/Prithvi-EO-2.0-300M"
MODEL_TAG = "Prithvi-EO-2.0-300M"
# 0 in an S2 SR patch = mask-edge / nodata; Prithvi maps nodata to this constant.
NODATA_FLOAT = 1e-4


# ----------------------------------------------------------------------------
# Model loading (standalone, no terratorch): fetch code + weights from HF cache
# ----------------------------------------------------------------------------
def _load_prithvi_class():
    """Import the standalone PrithviMAE class from the HF-cached code file.

    prithvi_mae.py uses PEP 604 ``X | Y`` annotations that Python 3.9 evaluates
    eagerly (TypeError on ``|``). Prepend ``from __future__ import annotations`` to
    the source so annotations stay as strings -- no edit to the cached file.
    """
    code_path = hf_hub_download(REPO, "prithvi_mae.py")
    src = Path(code_path).read_text(encoding="utf-8")
    if "from __future__ import annotations" not in src:
        src = "from __future__ import annotations\n" + src
    mod = importlib.util.module_from_spec(
        importlib.util.spec_from_loader("prithvi_mae", loader=None)
    )
    sys.modules["prithvi_mae"] = mod
    exec(compile(src, code_path, "exec"), mod.__dict__)   # noqa: S102
    return mod.PrithviMAE


def load_config() -> dict:
    with open(hf_hub_download(REPO, "config.json")) as f:
        return json.load(f)["pretrained_cfg"]


def build_model(device: str):
    """Build PrithviMAE for T=1, load pretrained weights, freeze. Returns (model, cfg)."""
    cfg = load_config()
    PrithviMAE = _load_prithvi_class()
    model = PrithviMAE(
        img_size=cfg["img_size"],
        patch_size=tuple(cfg["patch_size"]),
        num_frames=1,                       # each (site, month) is a single-frame patch
        in_chans=cfg["in_chans"],
        embed_dim=cfg["embed_dim"],
        depth=cfg["depth"],
        num_heads=cfg["num_heads"],
        decoder_embed_dim=cfg["decoder_embed_dim"],
        decoder_depth=cfg["decoder_depth"],
        decoder_num_heads=cfg["decoder_num_heads"],
        mlp_ratio=cfg["mlp_ratio"],
        coords_encoding=cfg.get("coords_encoding", []),   # [] -> no time/location coords needed
        coords_scale_learn=cfg.get("coords_scale_learn", False),
    )
    ckpt = hf_hub_download(REPO, "Prithvi_EO_V2_300M.pt")
    sd = torch.load(ckpt, map_location="cpu", weights_only=True)
    for k in list(sd):                      # sincos pos_embed is a recomputed buffer
        if "pos_embed" in k:
            del sd[k]
    missing, unexpected = model.load_state_dict(sd, strict=False)
    bad_missing = [k for k in missing if "pos_embed" not in k]
    if bad_missing:
        print(f"  WARN unexpected MISSING keys (non pos_embed): {bad_missing[:8]}")
    if unexpected:
        print(f"  WARN UNEXPECTED keys: {unexpected[:8]}")

    model.eval().to(device)
    for p in model.parameters():
        p.requires_grad_(False)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  model     : {MODEL_TAG}  ({n_params/1e6:.0f}M params, frozen)  device={device}")
    return model, cfg


# ----------------------------------------------------------------------------
# Patch -> normalised [6, 224, 224] tensor (all on CPU)
# ----------------------------------------------------------------------------
def read_patch(path: Path, mean, std, img_size: int) -> torch.Tensor:
    with rasterio.open(path) as ds:
        raw = ds.read().astype("float32")           # [6, H, W]
    if raw.shape[0] != 6:
        raise ValueError(f"{path.name}: expected 6 bands, got {raw.shape[0]}")
    m = np.asarray(mean, "float32")[:, None, None]
    s = np.asarray(std, "float32")[:, None, None]
    norm = (raw - m) / s
    norm = np.where(raw == 0, NODATA_FLOAT, norm).astype("float32")   # 0 -> nodata
    t = torch.from_numpy(norm).unsqueeze(0)          # [1, 6, H, W]
    t = F.interpolate(t, size=(img_size, img_size), mode="bilinear",
                      align_corners=False, antialias=True)
    return t.squeeze(0)                              # [6, 224, 224]


@torch.no_grad()
def embed_batch(model, batch: torch.Tensor, device: str):
    """[B, 6, 224, 224] -> (meanpool [B,1024], cls [B,1024])."""
    feats = model.forward_features(batch.to(device))     # list of [B, 197, C]
    last = feats[-1]                                     # post-norm block output
    cls = last[:, 0, :]
    meanpool = last[:, 1:, :].mean(dim=1)                # over 196 patch tokens
    return meanpool.float().cpu().numpy(), cls.float().cpu().numpy()


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description="Precompute frozen Prithvi-EO-2.0 S2 patch embeddings.")
    ap.add_argument("--patch-dir", type=Path, default=PATCH_DIR)
    ap.add_argument("--index", type=Path, default=INDEX_CSV)
    ap.add_argument("--out-dir", type=Path, default=OUT)
    ap.add_argument("--device", default=None, help="mps|cpu|cuda (default: auto)")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None, help="only first N usable patches (smoke test)")
    ap.add_argument("--skip-errors", action="store_true", help="skip unreadable patches instead of failing")
    args = ap.parse_args()

    device = args.device or ("mps" if torch.backends.mps.is_available() else "cpu")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    idx = pd.read_csv(args.index, dtype={"month": str, "year": str})
    valid = idx[idx["valid_mask"] == 1].reset_index(drop=True)
    if args.limit:
        valid = valid.head(args.limit).reset_index(drop=True)
    N = len(valid)
    print(f"index     : {args.index}")
    print(f"usable    : {N} patches ({idx['valid_mask'].sum()} total valid; "
          f"{idx['site_id'].nunique()} sites)")

    model, cfg = build_model(device)
    mean, std, img_size = cfg["mean"], cfg["std"], cfg["img_size"]

    emb_mean = np.full((N, cfg["embed_dim"]), np.nan, dtype="float32")
    emb_cls = np.full((N, cfg["embed_dim"]), np.nan, dtype="float32")
    ok = np.zeros(N, dtype=bool)

    t0 = time.time()
    buf, rows = [], []
    done = 0

    def flush():
        nonlocal done
        if not buf:
            return
        batch = torch.stack(buf, dim=0)
        mp, cl = embed_batch(model, batch, device)
        for j, r in enumerate(rows):
            emb_mean[r] = mp[j]
            emb_cls[r] = cl[j]
            ok[r] = True
        done += len(rows)
        rate = done / max(time.time() - t0, 1e-6)
        eta = (N - done) / max(rate, 1e-6)
        print(f"  {done:4d}/{N}  ({rate:4.1f} patch/s, eta {eta/60:4.1f} min)")
        buf.clear()
        rows.clear()

    for i in range(N):
        row = valid.iloc[i]
        path = args.patch_dir / row["filename"]
        try:
            buf.append(read_patch(path, mean, std, img_size))
            rows.append(i)
        except Exception as e:  # noqa: BLE001
            msg = f"{row['filename']}: {type(e).__name__}: {e}"
            if args.skip_errors:
                print(f"  SKIP {msg}")
                continue
            raise RuntimeError(f"failed reading {msg}") from e
        if len(buf) >= args.batch_size:
            flush()
    flush()

    n_ok = int(ok.sum())
    print(f"\nembedded  : {n_ok}/{N} patches in {time.time()-t0:.0f}s "
          f"(dim={cfg['embed_dim']}, pooling=meanpool + cls)")

    out_index = valid.copy()
    out_index.insert(0, "emb_row", np.arange(N))
    out_index["obs_month_start"] = pd.to_datetime(
        out_index["month"].str.replace("_", "-") + "-01", errors="coerce"
    ).dt.date
    out_index["sensor"] = "S2"
    out_index["emb_model"] = MODEL_TAG
    out_index["emb_dim"] = cfg["embed_dim"]
    out_index["emb_ok"] = ok.astype(int)
    keep = ["emb_row", "site_id", "site_name", "site_type", "month", "year",
            "obs_month_start", "mean_cloud", "n_scenes", "patch_px", "filename",
            "sensor", "emb_model", "emb_dim", "emb_ok"]
    out_index = out_index[[c for c in keep if c in out_index.columns]]

    mean_path = args.out_dir / "s2_prithvi_emb_meanpool.npy"
    cls_path = args.out_dir / "s2_prithvi_emb_cls.npy"
    index_path = args.out_dir / "s2_prithvi_emb_index.csv"
    cov_path = args.out_dir / "s2_prithvi_emb_coverage.csv"

    np.save(mean_path, emb_mean)
    np.save(cls_path, emb_cls)
    out_index.to_csv(index_path, index=False)

    cov = (out_index.groupby(["site_id", "site_name", "site_type"])
           .agg(n_emb=("emb_ok", "sum"),
                first_month=("month", "min"),
                last_month=("month", "max"))
           .reset_index()
           .sort_values(["site_type", "site_name"]))
    cov.to_csv(cov_path, index=False)

    # quick numeric sanity on the embeddings that succeeded
    good = emb_mean[ok]
    print("\n[embedding sanity]")
    if good.size:
        print(f"  meanpool: shape={emb_mean.shape}  finite={np.isfinite(good).all()}  "
              f"val[min/mean/max]={good.min():.3f}/{good.mean():.3f}/{good.max():.3f}  "
              f"per-vec L2[mean]={np.linalg.norm(good, axis=1).mean():.2f}")
    print("\n[per-site embedded months]")
    print(cov[["site_name", "site_type", "n_emb", "first_month", "last_month"]].to_string(index=False))
    print(f"\nSaved:\n  {mean_path}\n  {cls_path}\n  {index_path}\n  {cov_path}")


if __name__ == "__main__":
    main()
