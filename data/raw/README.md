# Raw downloads (local only)

This directory is **gitignored** except for scripts, AOI tables, and manifests. Clone the repo, then place downloads here following the layout below. Default modelling **does not** read these files.

```text
data/raw/
??? 00_spatial_anchors/          Natural Earth, port/infrastructure gazetteers
??? 01_market_financial/         EIA, FRED, Yahoo, Dallas Fed, GPR
?   ??? download_m1_raw.py
?   ??? manifest.csv
??? 02_sentinel2/                GEE exports (patches + Channel B indices)
?   ??? aoi_oil_infrastructure.csv
?   ??? aoi_oil_infrastructure_sites.md
?   ??? Channel A|B/             *.js GEE scripts
??? 03_shipping/
    ??? GFW/                     download_gfw_*.py
    ??? IMF Portwatch/           download_portwatch_*.py
```

Do not commit API tokens (`.gfw_token`, `.env`). Source licences: [`../sources.md`](../sources.md).
