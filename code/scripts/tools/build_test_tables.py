"""
Build the FROZEN forecast-comparison tables for Chapters 4-5.

This script is the single place where the study decides which test applies to
which comparison, which alternative is used, and which comparisons form a
multiple-comparison family. Every p-value quoted in the dissertation comes from
here, so the main text, the result tables and the figures cannot drift apart.

Test scheme
  Primary test        Diebold-Mariano with the Harvey-Leybourne-Newbold
                      small-sample correction, on squared errors of the
                      reconstructed price, so that the loss matches the RMSE
                      headline metric. d_t = L(reference) - L(candidate), so a
                      positive statistic favours the candidate.
  Alternative         One-sided where the research question is directional
                      (RQ1 asks whether the added data improve accuracy; RQ2 as
                      stated asks whether the Deep pathway OUTPERFORMS the Flat
                      pathway). Two-sided for the fusion-mechanism contrasts,
                      where a difference in either direction is informative.
  Multiplicity        Holm (1979) within each pre-defined family. Formal
                      significance is judged on the adjusted p-values; raw
                      p-values are reported as nominal.
  Clark-West          Supplementary, Ridge only, nested at the predictor-set
                      level. Never used for XGBoost or Deep comparisons and
                      never used against M0 (see backtest.metrics).

Frozen families
  benchmark  18 rows  every reported specification against M0
  RQ1        15 rows  5 information-set contrasts x 3 learners
  RQ2        14 rows  8 matched-information-set Flat-Deep pairs (one-sided)
                      + 6 fusion-mechanism contrasts vs gated (two-sided)
  Sensitivity and sweep arms are NOT family members; they stay exploratory.

Prediction sources (all on the same 257 forecast origins)
  Flat        results/baselines/Flat/M*_Flat/baseline_predictions*.csv
  Deep S1     results/baselines/Deep/_cross/deep_predictions.csv
  Deep S2-S4  results/baselines/Deep/_cross/deep_fusion_predictions.csv
              (all nine information-set x fusion cells from one run)

Outputs (-> results/tests/):
  test_table_main.csv          the three frozen families, raw + Holm p
  test_table_cw_supplementary.csv   the five Ridge Clark-West rows
  test_table_robustness.csv    appendix checks (absolute-error loss, early/late
                               sub-periods, two-sided sensitivity, leave-out
                               influence diagnostic)

Run:
  python3 code/scripts/tools/build_test_tables.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPTS_DIR = Path(__file__).resolve().parent
SRC_DIR = SCRIPTS_DIR.parent.parent / "src"
sys.path.insert(0, str(SRC_DIR))

from backtest import data, metrics                                  # noqa: E402
from model_naming import deep_out_dir, flat_out_dir                 # noqa: E402

OUT_DIR = data.ROOT / "results" / "tests"
SPLIT = pd.Timestamp("2023-01-01")          # early / late sub-period boundary
INFLUENCE_DROP = 8                          # weeks removed by the diagnostic
SETS = ["S1", "S2", "S3", "S4"]

# Flat tier -> (information set, predictions file). The _anom files carry the
# remote-sensing anomaly features used in the main specification.
FLAT_FILES = {
    "S1": ("M1", "baseline_predictions.csv"),
    "S2": ("M2", "baseline_predictions_anom.csv"),
    "S3": ("M3", "baseline_predictions.csv"),
    "S4": ("M4", "baseline_predictions_anom.csv"),
}
# Deep information-set label in the fusion dump -> study information set.
DEEP_TIER_SET = {"M2_Deep": "S2", "M3_Deep": "S3", "M4_Deep": "S4"}
FUSIONS = ["gated", "concat", "xattn"]
MAIN_FUSION = "gated"                        # pre-specified main Deep design


def _flat_spec(learner: str, s: str) -> str:
    return f"{learner} {s}"


def _deep_spec(fusion: str, s: str) -> str:
    # S1 activates the finance encoder only, so no fusion step applies.
    return f"Deep {s}" if s == "S1" else f"Deep {fusion} {s}"


def deep_main(s: str) -> str:
    return _deep_spec(MAIN_FUSION, s)


# --------------------------------------------------------------------------
# panel assembly
# --------------------------------------------------------------------------
def load_panel() -> pd.DataFrame:
    """One wide frame: P_next_actual + one forecast column per specification."""
    tier, fname = FLAT_FILES["S1"]
    base = pd.read_csv(flat_out_dir(data.ROOT, tier) / fname,
                       index_col=0, parse_dates=True)
    base.index.name = "forecast_origin"
    panel = base[["P_next_actual"]].copy()
    panel["M0"] = base["P_hat_M0"]

    for s, (tier, fname) in FLAT_FILES.items():
        res = pd.read_csv(flat_out_dir(data.ROOT, tier) / fname,
                          index_col=0, parse_dates=True)
        label = "M1_Flat" if s == "S1" else f"{tier}_Flat"
        for learner, col in (("Ridge", "Ridge"), ("XGB", "XGB")):
            panel[_flat_spec(learner, s)] = res[f"P_hat_{label}_{col}"]

    cross = deep_out_dir(data.ROOT, "cross")
    deep1 = pd.read_csv(cross / "deep_predictions.csv",
                        index_col=0, parse_dates=True)
    panel["Deep S1"] = deep1["P_hat_M1_Deep"]

    fus_path = cross / "deep_fusion_predictions.csv"
    if not fus_path.exists():
        raise SystemExit(
            f"Missing {fus_path}.\nRun: python3 "
            f"code/scripts/deep/run_deep_fusion_matrix.py")
    fus = pd.read_csv(fus_path, parse_dates=["forecast_origin"])
    for (tier, fusion), g in fus.groupby(["information_set", "fusion"]):
        s = DEEP_TIER_SET[tier]
        panel[_deep_spec(fusion, s)] = g.set_index("forecast_origin")["P_hat"]

    if panel.isna().any().any():
        bad = panel.columns[panel.isna().any()].tolist()
        raise SystemExit(f"specifications not aligned on the shared calendar: {bad}")
    return panel


# --------------------------------------------------------------------------
# frozen families
# --------------------------------------------------------------------------
def family_rows() -> list[dict]:
    """Every row of the three frozen families, in reporting order."""
    reported = ([_flat_spec("Ridge", s) for s in SETS]
                + [_flat_spec("XGB", s) for s in SETS]
                + ["Deep S1"]
                + [_deep_spec(f, s) for f in FUSIONS for s in SETS[1:]])
    rows = [{"family": "benchmark", "reference": "M0", "candidate": c,
             "information_set": c.split()[-1], "alternative": "greater",
             "note": "beats the no-change benchmark"} for c in reported]

    pairs = [("S1", "S2"), ("S1", "S3"), ("S1", "S4"), ("S2", "S4"), ("S3", "S4")]
    for learner in ("Ridge", "XGB", "Deep"):
        spec = deep_main if learner == "Deep" else (
            lambda s, L=learner: _flat_spec(L, s))
        for a, b in pairs:
            rows.append({"family": "RQ1", "reference": spec(a), "candidate": spec(b),
                         "information_set": f"{a}->{b}", "alternative": "greater",
                         "note": "added modality within one learner"})

    for s in SETS:
        for learner in ("Ridge", "XGB"):
            note = ("model-class control, no fusion at S1" if s == "S1"
                    else "pathway contrast at a matched information set")
            rows.append({"family": "RQ2", "reference": _flat_spec(learner, s),
                         "candidate": deep_main(s), "information_set": s,
                         "alternative": "greater", "note": note})
    for s in SETS[1:]:
        for fusion in ("concat", "xattn"):
            rows.append({"family": "RQ2", "reference": deep_main(s),
                         "candidate": _deep_spec(fusion, s),
                         "information_set": s, "alternative": "two-sided",
                         "note": f"{fusion} against the main gated design"})
    return rows


def cw_rows() -> list[dict]:
    """Supplementary Clark-West rows: Ridge only, predictor-set nested."""
    pairs = [("S1", "S2"), ("S1", "S3"), ("S1", "S4"), ("S2", "S4"), ("S3", "S4")]
    return [{"family": "supplementary_CW", "reference": _flat_spec("Ridge", a),
             "candidate": _flat_spec("Ridge", b), "information_set": f"{a}->{b}",
             "alternative": "greater",
             "note": "nested at the predictor-set level; supplementary only"}
            for a, b in pairs]


# --------------------------------------------------------------------------
# scoring helpers
# --------------------------------------------------------------------------
def _err(panel: pd.DataFrame, spec: str, mask=None) -> np.ndarray:
    e = (panel[spec] - panel["P_next_actual"]).to_numpy()
    return e if mask is None else e[mask]


def _rmse(panel: pd.DataFrame, spec: str, mask=None) -> float:
    return float(np.sqrt(np.mean(_err(panel, spec, mask) ** 2)))


def _mae(panel: pd.DataFrame, spec: str, mask=None) -> float:
    return float(np.mean(np.abs(_err(panel, spec, mask))))


def score(panel: pd.DataFrame, row: dict, mask=None,
          loss: str = "squared_error",
          alternative: "str | None" = None) -> dict:
    ref, cand = row["reference"], row["candidate"]
    alt = alternative or row["alternative"]
    stat, p = metrics.dm_test(_err(panel, ref, mask), _err(panel, cand, mask),
                              alternative=alt, loss=loss)
    rmse_r, rmse_c = _rmse(panel, ref, mask), _rmse(panel, cand, mask)
    return {**row, "test": "DM-HLN", "alternative": alt, "loss": loss,
            "n_test": int(len(_err(panel, ref, mask))),
            "RMSE_reference": rmse_r, "RMSE_candidate": rmse_c,
            "MAE_reference": _mae(panel, ref, mask),
            "MAE_candidate": _mae(panel, cand, mask),
            "RMSE_reduction_pct": 100.0 * (1.0 - rmse_c / rmse_r),
            "statistic": stat, "p_raw": p}


def score_cw(panel: pd.DataFrame, row: dict) -> dict:
    y = panel["P_next_actual"].to_numpy()
    stat, p = metrics.clark_west(y, panel[row["reference"]].to_numpy(),
                                 panel[row["candidate"]].to_numpy())
    rmse_r = _rmse(panel, row["reference"])
    rmse_c = _rmse(panel, row["candidate"])
    return {**row, "test": "Clark-West", "loss": "squared_error",
            "n_test": int(len(y)), "RMSE_reference": rmse_r,
            "RMSE_candidate": rmse_c,
            "RMSE_reduction_pct": 100.0 * (1.0 - rmse_c / rmse_r),
            "statistic": stat, "p_raw": p}


def add_holm(tab: pd.DataFrame, by: tuple[str, ...] = ("family",),
             rows: pd.Series | None = None) -> pd.DataFrame:
    """Holm-adjust p_raw within each `by` group.

    `rows` optionally restricts the adjustment to a subset; excluded rows keep
    a missing adjusted p. That is used for the leave-out influence diagnostic,
    which selects the dropped weeks from the observed loss differential and is
    therefore not a test that a familywise correction can be applied to.
    """
    tab = tab.copy()
    tab["p_holm"] = np.nan
    eligible = pd.Series(True, index=tab.index) if rows is None else rows
    for _, g in tab[eligible].groupby(list(by)):
        tab.loc[g.index, "p_holm"] = metrics.holm(g["p_raw"].to_numpy())
    tab["significant_5pct_holm"] = tab["p_holm"] < 0.05
    tab["family_size"] = tab.groupby(list(by))["p_raw"].transform("size")
    return tab


# --------------------------------------------------------------------------
# appendix robustness
# --------------------------------------------------------------------------
def robustness(panel: pd.DataFrame, main: pd.DataFrame) -> pd.DataFrame:
    early = np.asarray(panel.index < SPLIT)
    late = ~early
    out = []
    for _, r in main.iterrows():
        row = {k: r[k] for k in ("family", "reference", "candidate",
                                 "information_set", "note")}
        row["alternative"] = r["alternative"]
        out.append({**score(panel, row, loss="absolute_error"),
                    "check": "absolute-error loss"})
        out.append({**score(panel, row, mask=early), "check": "early (<=2022)"})
        out.append({**score(panel, row, mask=late), "check": "late (>=2023)"})
        flip = "two-sided" if r["alternative"] == "greater" else "greater"
        out.append({**score(panel, row, alternative=flip),
                    "check": f"{flip} sensitivity"})
        if bool(r["significant_5pct_holm"]):
            # Data-dependent: the dropped weeks are chosen on the observed loss
            # differential, so this is an influence diagnostic, NOT a test.
            d = (_err(panel, r["reference"]) ** 2
                 - _err(panel, r["candidate"]) ** 2)
            keep = np.ones(len(d), bool)
            keep[np.argsort(-np.abs(d))[:INFLUENCE_DROP]] = False
            share = float(d[~keep].sum() / d.sum()) if d.sum() != 0 else np.nan
            out.append({**score(panel, row, mask=keep),
                        "check": f"leave-out influence diagnostic "
                                 f"(drop {INFLUENCE_DROP} most influential weeks; "
                                 f"they carry {share*100:.0f}% of sum d_t)"})
    rob = pd.DataFrame(out)
    # Each check re-runs the same frozen families, so Holm is applied within
    # (check, family) exactly as in the main table, never pooled across checks.
    is_test = ~rob["check"].str.startswith("leave-out influence diagnostic")
    return add_holm(rob, by=("check", "family"), rows=is_test)


# --------------------------------------------------------------------------
def main() -> None:
    panel = load_panel()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"panel: {panel.shape[1]-1} specifications x {len(panel)} forecast "
          f"origins ({panel.index.min().date()} -> {panel.index.max().date()})")

    tab = add_holm(pd.DataFrame([score(panel, r) for r in family_rows()]))
    cw = pd.DataFrame([score_cw(panel, r) for r in cw_rows()])
    cw["p_holm"] = metrics.holm(cw["p_raw"].to_numpy())
    rob = robustness(panel, tab)

    cols = ["family", "family_size", "reference", "candidate", "information_set",
            "test", "alternative", "loss", "n_test", "RMSE_reference",
            "RMSE_candidate", "RMSE_reduction_pct", "MAE_reference",
            "MAE_candidate", "statistic", "p_raw", "p_holm",
            "significant_5pct_holm", "note"]
    tab[cols].to_csv(OUT_DIR / "test_table_main.csv", index=False)
    cw.to_csv(OUT_DIR / "test_table_cw_supplementary.csv", index=False)
    rob.to_csv(OUT_DIR / "test_table_robustness.csv", index=False)

    show = ["reference", "candidate", "alternative", "RMSE_reference",
            "RMSE_candidate", "RMSE_reduction_pct", "statistic", "p_raw",
            "p_holm"]
    for fam, title in (("benchmark", "FAMILY 1 - every specification vs M0"),
                       ("RQ1", "FAMILY 2 - RQ1 information-set increments"),
                       ("RQ2", "FAMILY 3 - RQ2 pathway and fusion contrasts")):
        sub = tab[tab["family"] == fam]
        print("\n" + "=" * 118)
        print(f"{title}  (m = {len(sub)}, Holm within family)")
        print(sub[show].to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("\n" + "=" * 118)
    print("SUPPLEMENTARY Clark-West, Ridge only (not used for formal claims)")
    print(cw[["reference", "candidate", "RMSE_reference", "RMSE_candidate",
              "statistic", "p_raw"]].to_string(
        index=False, float_format=lambda x: f"{x:.4f}"))

    sig = tab[tab["significant_5pct_holm"]]
    print("\n" + "=" * 118)
    if len(sig):
        print("Holm-adjusted significant at 5%:")
        for _, r in sig.iterrows():
            print(f"  [{r['family']}] {r['reference']} -> {r['candidate']}: "
                  f"RMSE {r['RMSE_reference']:.3f} -> {r['RMSE_candidate']:.3f} "
                  f"({r['RMSE_reduction_pct']:+.2f}%), raw p={r['p_raw']:.4f}, "
                  f"Holm p={r['p_holm']:.4f}")
    else:
        print("No comparison is significant at 5% after Holm adjustment.")
    print(f"\nSaved: {OUT_DIR/'test_table_main.csv'}\n"
          f"       {OUT_DIR/'test_table_cw_supplementary.csv'}\n"
          f"       {OUT_DIR/'test_table_robustness.csv'}")


if __name__ == "__main__":
    main()
