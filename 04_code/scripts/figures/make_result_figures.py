"""
Chapter 4 and Appendix B figures.

Reads only committed result CSVs under 05_outputs/ and writes PNG + PDF to
05_outputs/figures/. No model re-fitting.

File names follow the figure numbers used in the dissertation, which run in
order of first mention in Chapter 4.

    python 04_code/scripts/figures/make_result_figures.py [--only 4.1 4.3]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT / "05_outputs" / "baselines"
FLAT = BASE / "Flat"
DEEP = BASE / "deep"
OUT_DIR = ROOT / "05_outputs" / "figures"

MODELS = ["M1", "M2", "M3", "M4"]
MODEL_SUBTITLES = {
    "M1": "finance",
    "M2": "+ RS",
    "M3": "+ shipping",
    "M4": "+ RS + ship",
}

SITE_NAMES = {
    "P001": "Rotterdam", "P002": "Fujairah", "P003": "Ras Tanura",
    "P004": "Jurong", "P005": "Houston", "P006": "Ningbo-Zhoushan",
    "P007": "Jamnagar", "P008": "Basra", "P009": "Ulsan",
    "P010": "Kharg", "P011": "Yanbu",
}

MODEL_NAMES = {
    "M0_RW": "M0 no-change",
    "M1_Flat_Ridge": "M1 Flat (Ridge)",
    "M1_Flat_XGB": "M1 Flat (XGBoost)",
    "M1_Deep": "M1 Deep (finance only)",
    "M_ship_GNN": "Shipping-only Deep",
    "M_rs_deep": "RS-only Deep",
    "M2_Deep_gated": "M2 Deep (gated)",
    "M3_Deep_gated": "M3 Deep (gated)",
    "M4_Deep_gated": "M4 Deep (gated)",
    "M4_Deep_Concat": "M4 Deep (concat)",
}

C = {
    "ridge": "#8FA8C8",
    "xgb": "#2E5A88",
    "gated": "#D1622B",
    "xattn": "#E7A33E",
    "flat": "#2E5A88",
    "deep": "#D1622B",
    "full": "#4A4A4A",
    "early": "#8FA8C8",
    "late": "#2E5A88",
    "pos": "#2E7D5B",
    "neg": "#B23A32",
    "grey": "#9A9A9A",
}

mpl.rcParams.update({
    "figure.dpi": 130,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linewidth": 0.6,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "legend.frameon": False,
    "legend.fontsize": 8,
})


def spread(values, min_gap: float):
    """Nudge label positions apart so text does not overlap, keeping order."""
    vals = np.asarray(values, float)
    order = np.argsort(vals)
    out = vals[order].copy()
    for k in range(1, len(out)):
        out[k] = max(out[k], out[k - 1] + min_gap)
    res = np.empty_like(out)
    res[order] = out
    return res


def save(fig, name: str) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(OUT_DIR / f"{name}.{ext}")
    plt.close(fig)
    print(f"  saved {name}.png / .pdf")


def load_subperiod() -> pd.DataFrame:
    df = pd.read_csv(BASE / "subperiod" / "subperiod_summary.csv")
    df["period"] = df["period"].str.replace(r"\(.*\)", "", regex=True)
    return df.drop_duplicates(subset=["model", "period"])


def load_fusion() -> pd.DataFrame:
    return pd.read_csv(DEEP / "_cross" / "deep_fusion_matrix.csv")


def skill_table() -> pd.DataFrame:
    """Skill vs M0 (%) for every learner x information set, from Tables 4.1-4.3."""
    sub = load_subperiod()
    full = sub[sub["period"] == "full"].set_index("model")["skill_vs_M0"]
    fusion = load_fusion().set_index(["combo", "fusion"])["skill_vs_M0"]

    rows = {}
    for m in MODELS:
        row = {
            "Flat Ridge": full.get(f"{m}_Flat_Ridge", np.nan),
            "Flat XGBoost": full.get(f"{m}_Flat_XGB", np.nan),
        }
        if m == "M1":
            # Only the finance encoder is active, so fusion does not apply.
            row["Deep gated"] = full.get("M1_Deep", np.nan)
            row["Deep cross-attn"] = np.nan
        else:
            row["Deep gated"] = fusion.get((f"{m}_Deep", "gated"), np.nan)
            row["Deep cross-attn"] = fusion.get((f"{m}_Deep", "xattn"), np.nan)
        rows[m] = row
    return pd.DataFrame(rows).T * 100.0


# --------------------------------------------------------------------------
# F1 - grouped skill bars
# --------------------------------------------------------------------------
def fig1_skill_bars() -> None:
    tab = skill_table()
    series = ["Flat Ridge", "Flat XGBoost", "Deep gated", "Deep cross-attn"]
    colors = [C["ridge"], C["xgb"], C["gated"], C["xattn"]]

    fig, ax = plt.subplots(figsize=(7.4, 4.0))
    x = np.arange(len(MODELS))
    width = 0.2

    for i, (name, col) in enumerate(zip(series, colors)):
        vals = tab[name].to_numpy(float)
        pos = x + (i - 1.5) * width
        ax.bar(pos, vals, width * 0.9, label=name, color=col,
               edgecolor="white", linewidth=0.5)
        for xi, v in zip(pos, vals):
            if np.isnan(v):
                ax.text(xi, 0.15, "n/a", ha="center", va="bottom",
                        fontsize=6.5, color=C["grey"], rotation=90)
                continue
            off = 0.22 if v >= 0 else -0.22
            ax.text(xi, v + off, f"{v:+.2f}", ha="center",
                    va="bottom" if v >= 0 else "top", fontsize=6.5)

    ax.axhline(0, color="black", linewidth=1.1, zorder=3)
    ax.text(-0.60, 1.35, "M0 no-change benchmark (skill = 0)", fontsize=7.5,
            color="#444444", style="italic")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{m}\n{MODEL_SUBTITLES[m]}" for m in MODELS])
    ax.set_ylabel("RMSE skill vs M0 (%)")
    ax.set_title("Out-of-sample RMSE skill versus the no-change benchmark "
                 "(n = 257 weeks)\nabove the line = beats M0", loc="left")
    ax.set_ylim(-10.5, 2.4)
    ax.legend(ncol=4, loc="lower left", bbox_to_anchor=(0.0, -0.30))
    ax.grid(axis="x", visible=False)
    save(fig, "fig_4_1_skill_bars")


# --------------------------------------------------------------------------
# F2 - Flat vs Deep paired slope chart
# --------------------------------------------------------------------------
def fig2_slope() -> None:
    sub = load_subperiod()
    full = sub[sub["period"] == "full"].set_index("model")
    fusion = load_fusion().set_index(["combo", "fusion"])

    flat, deep = [], []
    for m in MODELS:
        flat.append(full.loc[f"{m}_Flat_XGB", "RMSE"])
        deep.append(full.loc["M1_Deep", "RMSE"] if m == "M1"
                    else fusion.loc[(f"{m}_Deep", "gated"), "RMSE"])
    m0 = pd.read_csv(DEEP / "_cross" / "deep_metrics.csv").set_index("model").loc["M0_RW", "RMSE"]

    fig, ax = plt.subplots(figsize=(5.8, 4.4))
    gap = 0.024
    flat_lab = spread(flat, gap)
    deep_lab = spread(deep, gap)

    for i, m in enumerate(MODELS):
        beats = deep[i] < m0
        ax.plot([0, 1], [flat[i], deep[i]], "-o", markersize=5,
                color=C["deep"] if beats else C["grey"],
                linewidth=2.0 if beats else 1.3,
                zorder=3 if beats else 2)
        ax.text(-0.05, flat_lab[i], f"{m}  {flat[i]:.3f}", ha="right",
                va="center", fontsize=8)
        ax.text(1.05, deep_lab[i], f"{deep[i]:.3f}  {m}", ha="left", va="center",
                fontsize=8, fontweight="bold" if beats else "normal",
                color=C["deep"] if beats else "black")

    ax.axhline(m0, color="black", linewidth=1.0, linestyle="--")
    ax.text(0.04, m0, f"M0 benchmark = {m0:.3f}", ha="left", va="center",
            fontsize=8, style="italic",
            bbox=dict(facecolor="white", edgecolor="none", pad=1.2))

    ax.set_xlim(-0.45, 1.45)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Flat\n(XGBoost)", "Deep\n(gated fusion)"])
    ax.set_ylabel("Out-of-sample RMSE (USD per barrel)")
    ax.set_title("Deep lowers RMSE in every matched information set;\n"
                 "only M3 (with shipping) also clears M0", loc="left")
    ax.invert_yaxis()
    ax.grid(axis="x", visible=False)
    save(fig, "fig_4_2_flat_vs_deep_slope")


# --------------------------------------------------------------------------
# F3 - sub-period stability
# --------------------------------------------------------------------------
def fig3_subperiod() -> None:
    sub = load_subperiod()
    periods = ["full", "early", "late"]
    plabel = {"full": "Full (n=257)", "early": "Early <2023 (n=102)",
              "late": "Late \u22652023 (n=155)"}

    panels = [
        ("Flat (XGBoost)", [f"{m}_Flat_XGB" for m in MODELS]),
        ("Deep (gated fusion)", ["M1_Deep"] + [f"{m}_Deep_gated" for m in MODELS[1:]]),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.9), sharey=True)
    x = np.arange(len(MODELS))
    width = 0.26

    for ax, (title, names) in zip(axes, panels):
        for j, per in enumerate(periods):
            vals = [sub.loc[(sub["model"] == n) & (sub["period"] == per),
                            "skill_vs_M0"].squeeze() * 100 for n in names]
            ax.bar(x + (j - 1) * width, vals, width * 0.9, label=plabel[per],
                   color=C[per], edgecolor="white", linewidth=0.5)
        ax.axhline(0, color="black", linewidth=1.1, zorder=3)
        ax.set_xticks(x)
        ax.set_xticklabels([f"{m}\n{MODEL_SUBTITLES[m]}" for m in MODELS])
        ax.set_title(title, loc="left")
        ax.grid(axis="x", visible=False)

    axes[0].set_ylabel("RMSE skill vs M0 (%)")
    axes[0].legend(loc="lower left", fontsize=7.5)
    fig.suptitle("Sub-period splits: every Flat set stays below M0 in both windows, "
                 "while Deep M3 stays above it",
                 x=0.005, ha="left", fontsize=10)
    fig.tight_layout()
    save(fig, "fig_4_4_subperiod_skill")


# --------------------------------------------------------------------------
# F4 - event-window gate shifts
# --------------------------------------------------------------------------
def fig4_events() -> None:
    ev = pd.read_csv(DEEP / "M4_Deep" / "deep_gate_events.csv")
    seeds = [("delta_seed42", "seed 42", "o"),
             ("delta_seed1", "seed 1", "s"),
             ("delta_seed2", "seed 2", "^")]
    ev = ev.iloc[::-1].reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(7.4, 3.4))
    y = np.arange(len(ev))

    for i in y:
        agree = bool(ev.loc[i, "same_direction"])
        ax.axhspan(i - 0.42, i + 0.42, color="#F2F2F2" if i % 2 else "white",
                   zorder=0)
        vals = [ev.loc[i, c] for c, _, _ in seeds]
        ax.plot([min(vals), max(vals)], [i, i], color=C["grey"],
                linewidth=1.0, zorder=1)
        for (col, _, marker), v in zip(seeds, vals):
            ax.plot(v, i, marker, markersize=6.5, zorder=3,
                    color=C["pos"] if v > 0 else C["neg"],
                    markeredgecolor="white", markeredgewidth=0.6)
        n_up, n_down = int(ev.loc[i, "n_up"]), int(ev.loc[i, "n_down"])
        note = f"{n_up} up, {n_down} down"
        ax.text(0.088, i, note + ("  (agree)" if agree else "  (mixed)"),
                va="center", fontsize=7.5, style="italic",
                color="black" if agree else C["neg"])

    ax.axvline(0, color="black", linewidth=1.1, zorder=2)
    ax.set_yticks(y)
    ax.set_yticklabels([f"{r.event}\n{r.event_date}" for r in ev.itertuples()],
                       fontsize=8)
    ax.set_ylim(-0.6, len(ev) - 0.4)
    ax.set_xlim(-0.055, 0.135)
    ax.set_xlabel("Change in mean shipping gate, event window (\u00b18 weeks) minus baseline")
    ax.set_title("Only the Russia\u2013Ukraine window raises the shipping gate "
                 "in all three seeds", loc="left")

    handles = [Line2D([], [], marker=mk, linestyle="", color=C["grey"],
                      markersize=6.5, label=lab) for _, lab, mk in seeds]
    handles += [Line2D([], [], marker="o", linestyle="", color=C["pos"],
                       markersize=6.5, label="gate up"),
                Line2D([], [], marker="o", linestyle="", color=C["neg"],
                       markersize=6.5, label="gate down")]
    ax.legend(handles=handles, ncol=5, loc="upper center",
              bbox_to_anchor=(0.5, -0.22))
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    save(fig, "fig_4_5_event_gate_shifts")


# --------------------------------------------------------------------------
# F5 - node-attention rank stability
# --------------------------------------------------------------------------
def fig5_node_attention() -> None:
    st = pd.read_csv(DEEP / "M4_Deep" / "deep_gate_stability.csv")
    chokepoints = {"hormuz", "suez", "mandeb", "malacca", "panama", "cape"}

    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.6))
    for ax, kind, title in [(axes[0], "ship", "Shipping-graph nodes"),
                            (axes[1], "rs", "Remote-sensing sites")]:
        d = st[st["kind"] == kind].sort_values("att_mean").reset_index(drop=True)
        y = np.arange(len(d))
        is_choke = d["name"].isin(chokepoints)
        colors = [C["gated"] if c else C["ridge"] for c in is_choke]

        ax.barh(y, d["att_mean"], xerr=d["att_std"], color=colors, height=0.68,
                error_kw={"elinewidth": 0.9, "ecolor": "#555555", "capsize": 2.5})
        for yi, (name, f5) in enumerate(zip(d["name"], d["freq_top5"])):
            ax.text(0.002, yi, f"{f5}/3", va="center", ha="left", fontsize=6.5,
                    color="white" if f5 >= 2 else "#333333")
        ax.set_yticks(y)
        ax.set_yticklabels([n.capitalize() if n in chokepoints
                            else f"{SITE_NAMES.get(n, n)} ({n})"
                            for n in d["name"]], fontsize=8)
        ax.set_xlim(0, float((d["att_mean"] + d["att_std"]).max()) * 1.35)
        ax.set_xlabel("Mean attention weight across 3 seeds (\u00b11 SD)")
        ax.set_title(title, loc="left")
        ax.grid(axis="y", visible=False)

    axes[0].legend(handles=[
        mpl.patches.Patch(color=C["gated"], label="maritime chokepoint"),
        mpl.patches.Patch(color=C["ridge"], label="port / infrastructure site"),
    ], loc="lower right")
    fig.suptitle("Hormuz is the only chokepoint in the top-5 attention set for all "
                 "three seeds\n(in-bar label = number of seeds ranking the node top-5)",
                 x=0.005, ha="left", fontsize=10)
    fig.tight_layout()
    save(fig, "fig_4_6_node_attention_stability")


# --------------------------------------------------------------------------
# F6 - weekly gate paths across seeds
# --------------------------------------------------------------------------
def fig6_gate_band() -> None:
    band = pd.read_csv(DEEP / "M4_Deep" / "deep_gate_band_weekly.csv",
                       parse_dates=["date"]).set_index("date")
    seeds = {}
    for s in (42, 1, 2):
        p = DEEP / "M4_Deep" / f"deep_gate_weekly_seed{s}.csv"
        if p.exists():
            seeds[s] = pd.read_csv(p, parse_dates=["date"]).set_index("date")

    mods = [("finance", "Financial time series", C["xgb"]),
            ("shipping", "Shipping", C["gated"]),
            ("rs", "Remote sensing", C["pos"])]

    fig, axes = plt.subplots(3, 1, figsize=(8.2, 6.4), sharex=True)
    for ax, (key, label, col) in zip(axes, mods):
        ax.fill_between(band.index, band[f"gate_{key}_lo"], band[f"gate_{key}_hi"],
                        color=col, alpha=0.18, linewidth=0,
                        label="seed range (mean \u00b11 SD)")
        for s, d in seeds.items():
            ax.plot(d.index, d[f"gate_{key}"], linewidth=0.7, alpha=0.55,
                    color=col, label=f"seed {s}" if ax is axes[0] else None)
        ax.plot(band.index, band[f"gate_{key}_mean"], color=col, linewidth=1.8,
                label="cross-seed mean" if ax is axes[0] else None)
        ax.set_ylabel(f"{label}\ngate weight", fontsize=8)
        ax.set_ylim(0, 1)
        ax.margins(x=0.01)

    for ax in axes:
        for d, lab in [("2022-02-24", "Russia\u2013Ukraine"),
                       ("2023-11-19", "Red Sea")]:
            ax.axvline(pd.Timestamp(d), color="#666666", linewidth=0.9,
                       linestyle=":", zorder=1)
    axes[0].text(pd.Timestamp("2022-03-05"), 0.9, "Russia\u2013Ukraine",
                 fontsize=7, color="#444444")
    axes[0].text(pd.Timestamp("2023-11-28"), 0.9, "Red Sea", fontsize=7,
                 color="#444444")
    axes[0].set_title("Weekly modality gates (Deep M4): week-to-week paths differ "
                      "markedly across seeds", loc="left")
    axes[-1].set_xlabel("Evaluation week")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=5, loc="lower center",
               bbox_to_anchor=(0.5, -0.025))
    fig.tight_layout()
    save(fig, "fig_B_1_gate_paths_seeds")


# --------------------------------------------------------------------------
# F7 - Clark-West / Diebold-Mariano p-values
# --------------------------------------------------------------------------
def fig7_cw() -> None:
    cw = pd.read_csv(DEEP / "_cross" / "deep_cw.csv")
    cw["nested"] = cw["valid_test"].str.startswith("CW")
    cw["label"] = (cw["small"].map(lambda s: MODEL_NAMES.get(s, s)) + "  \u2192  "
                   + cw["large"].map(lambda s: MODEL_NAMES.get(s, s)))
    cw = cw.sort_values(["nested", "p_value"], ascending=[False, False])
    cw = cw.reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(7.8, 4.4))
    y = np.arange(len(cw))
    sig = cw["p_value"] < 0.05
    ax.barh(y, cw["p_value"], height=0.66,
            color=[C["pos"] if s else C["ridge"] for s in sig])
    for yi, p, s in zip(y, cw["p_value"], sig):
        ax.text(p + 0.012, yi, f"{p:.3f}", va="center", fontsize=7.5,
                color=C["pos"] if s else "black",
                fontweight="bold" if s else "normal")

    ax.axvline(0.05, color=C["neg"], linewidth=1.1, linestyle="--")
    ax.text(0.062, -0.85, "p = 0.05", color=C["neg"], fontsize=7.5, va="center")
    ax.set_yticks(y)
    ax.set_yticklabels(cw["label"], fontsize=7.5)
    ax.set_xlim(0, 1.05)
    ax.set_ylim(-1.2, len(cw) - 0.4)
    ax.set_xlabel("p-value (one-sided): does the larger model add predictive information?")

    # Nested comparisons occupy the low y positions, non-nested the high ones.
    n_nested = int(cw["nested"].sum())
    ax.axhline(n_nested - 0.5, color="#BBBBBB", linewidth=0.9)
    ax.text(1.02, n_nested - 0.9, "Clark\u2013West (nested)", ha="right",
            va="center", fontsize=7.5, style="italic", color="#555555")
    ax.text(1.02, n_nested - 0.1, "Diebold\u2013Mariano (non-nested)", ha="right",
            va="center", fontsize=7.5, style="italic", color="#555555")
    ax.set_title("Incremental-information tests: shipping adds information over the "
                 "finance-only Deep baseline", loc="left")
    ax.grid(axis="y", visible=False)
    save(fig, "fig_4_3_incremental_tests")


# Keyed by final figure number, in order of first mention in Chapter 4.
FIGURES = {
    "4.1": fig1_skill_bars,
    "4.2": fig2_slope,
    "4.3": fig7_cw,
    "4.4": fig3_subperiod,
    "4.5": fig4_events,
    "4.6": fig5_node_attention,
    "B.1": fig6_gate_band,
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", choices=sorted(FIGURES), default=None)
    args = ap.parse_args()

    keys = args.only or sorted(FIGURES)
    for k in keys:
        print(f"[{k}] {FIGURES[k].__name__}")
        FIGURES[k]()
    print(f"\nOutput: {OUT_DIR}")


if __name__ == "__main__":
    main()
