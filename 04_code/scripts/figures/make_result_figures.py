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
DEEP = BASE / "Deep"
TESTS = ROOT / "05_outputs" / "tests"
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
    # Deep S3 is the M0-clearing arm, so gate diagnostics come from M3 (§3.10).
    ev = pd.read_csv(DEEP / "M3_Deep" / "deep_m3_gate_events.csv")
    seeds = [("delta_seed42", "seed 42", "o"),
             ("delta_seed1", "seed 1", "s"),
             ("delta_seed2", "seed 2", "^")]
    ev = ev.iloc[::-1].reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(7.4, 3.4))
    y = np.arange(len(ev))
    delta_cols = [c for c, _, _ in seeds]
    dmin = float(ev[delta_cols].to_numpy().min())
    dmax = float(ev[delta_cols].to_numpy().max())
    pad = max(dmax - dmin, 1e-3) * 0.12
    note_x = dmax + pad * 0.6

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
        ax.text(note_x, i, note + ("  (agree)" if agree else "  (mixed)"),
                va="center", fontsize=7.5, style="italic",
                color="black" if agree else C["neg"])

    ax.axvline(0, color="black", linewidth=1.1, zorder=2)
    ax.set_yticks(y)
    ax.set_yticklabels([f"{r.event}\n{r.event_date}" for r in ev.itertuples()],
                       fontsize=8)
    ax.set_ylim(-0.6, len(ev) - 0.4)
    ax.set_xlim(dmin - pad, dmax + pad * 4.2)
    ax.set_xlabel("Change in mean shipping gate, event window (\u00b18 weeks) minus baseline")
    agreed = ev[ev["same_direction"].astype(bool)]
    if len(agreed) == 1:
        r = agreed.iloc[0]
        verb = "raises" if int(r["n_up"]) == 3 else "lowers"
        title = f"Only the {r['event']} window {verb} the shipping gate in all three seeds"
    elif len(agreed) == 0:
        title = "No event window moves the shipping gate consistently across all three seeds"
    else:
        title = (f"{len(agreed)} of {len(ev)} event windows move the shipping gate "
                 f"consistently across all three seeds")
    ax.set_title(title, loc="left")

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
def _mean_token_share(path: Path) -> np.ndarray:
    """Mean attention share per token from a weekly cross-attention CSV."""
    d = pd.read_csv(path)
    cols = [c for c in d.columns
            if c not in ("forecast_origin", "target_date", "week", "date")]
    w = d[cols].to_numpy(float)
    w = w / w.sum(axis=1, keepdims=True)
    return w.mean(axis=0)


def fig5_node_attention() -> None:
    """Spatial selectivity of the two RQ3-eligible interpretable mechanisms.

    Only Deep units with positive RMSE skill against M0 qualify for RQ3
    (Deep gated S3, Deep xattn S3, Deep xattn S4). The gated S4 arm does not
    clear M0, so its remote-sensing site attention is an appendix diagnostic.
    """
    st = pd.read_csv(DEEP / "M3_Deep" / "deep_m3_gate_stability.csv")
    chokepoints = {"hormuz", "suez", "mandeb", "malacca", "panama", "cape"}

    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.8))

    # -- left: gated S3 shipping-node attention, mean +-1 SD over 3 seeds --
    ax = axes[0]
    d = st.sort_values("att_mean").reset_index(drop=True)
    y = np.arange(len(d))
    colors = [C["gated"] if c else C["ridge"] for c in d["name"].isin(chokepoints)]
    ax.barh(y, d["att_mean"], xerr=d["att_std"], color=colors, height=0.68,
            error_kw={"elinewidth": 0.9, "ecolor": "#555555", "capsize": 2.5})
    for yi, f5 in enumerate(d["freq_top5"]):
        ax.text(0.002, yi, f"{f5}/3", va="center", ha="left", fontsize=6.5,
                color="white" if f5 >= 2 else "#333333")
    ax.axvline(1 / len(d), color=C["full"], linewidth=1.0, linestyle=":")
    ax.text(1 / len(d) + 0.002, len(d) - 0.6, "uniform", fontsize=7,
            color=C["full"])
    ax.set_yticks(y)
    ax.set_yticklabels([n.capitalize() if n in chokepoints
                        else f"{SITE_NAMES.get(n, n)} ({n})"
                        for n in d["name"]], fontsize=8)
    ax.set_xlim(0, float((d["att_mean"] + d["att_std"]).max()) * 1.35)
    ax.set_xlabel("Mean attention weight across 3 seeds (\u00b11 SD)")
    ax.set_title("Gated S3: shipping-graph nodes", loc="left")
    ax.grid(axis="y", visible=False)
    ax.legend(handles=[
        mpl.patches.Patch(color=C["gated"], label="maritime chokepoint"),
        mpl.patches.Patch(color=C["ridge"], label="port / infrastructure site"),
    ], loc="lower right")

    # -- right: same shares divided by the uniform share, gated against xattn --
    ax = axes[1]
    curves = [
        ("Gated S3, shipping nodes", C["gated"], "o-",
         d["att_mean"].to_numpy()),
        ("Cross-attention S3, 17 tokens", C["xgb"], "s-",
         _mean_token_share(DEEP / "M3_Deep" / "deep_m3_xattn_weekly.csv")),
        ("Cross-attention S4, 28 tokens", C["xattn"], "D-",
         _mean_token_share(DEEP / "M4_Deep" / "deep_xattn_weekly.csv")),
    ]
    for label, color, style, share in curves:
        rel = np.sort(share / (1 / len(share)))[::-1]
        ax.plot(np.linspace(0, 1, len(rel)), rel, style, color=color,
                markersize=3.4, linewidth=1.1,
                label=f"{label}  (spread {rel.max() / rel.min():.2f}\u00d7)")
    ax.axhline(1.0, color=C["full"], linewidth=1.0, linestyle=":")
    ax.text(0.985, 1.0, "uniform", fontsize=7, color=C["full"], ha="right",
            va="bottom")
    ax.set_xlabel("Token rank (normalised, most attended on the left)")
    ax.set_ylabel("Attention share \u00f7 uniform share")
    ax.set_title("Selectivity relative to a uniform allocation", loc="left")
    ax.legend(loc="upper right")

    top = d.loc[d["att_mean"].idxmax()]
    n_all = int((d[d["name"].isin(chokepoints)]["freq_top5"] == 3).sum())
    lead = (f"{str(top['name']).capitalize()} has the highest mean shipping-node "
            "attention, but no chokepoint is top-5 in all three seeds"
            if n_all == 0 else
            f"{str(top['name']).capitalize()} leads shipping-node attention; "
            f"{n_all} chokepoint(s) are top-5 in all three seeds")
    fig.suptitle(f"{lead}; cross-attention is close to uniform\n"
                 "(in-bar label = number of seeds ranking the node top-5)",
                 x=0.005, ha="left", fontsize=10)
    fig.tight_layout()
    save(fig, "fig_4_6_node_attention_stability")


def figB2_rs_site_attention() -> None:
    """Appendix: RS-site attention in gated S4, a unit that does not clear M0."""
    st = pd.read_csv(DEEP / "M4_Deep" / "deep_gate_stability.csv")
    d = (st[st["kind"] == "rs"].sort_values("att_mean").reset_index(drop=True))

    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    y = np.arange(len(d))
    ax.barh(y, d["att_mean"], xerr=d["att_std"], color=C["ridge"], height=0.68,
            error_kw={"elinewidth": 0.9, "ecolor": "#555555", "capsize": 2.5})
    for yi, f5 in enumerate(d["freq_top5"]):
        ax.text(0.002, yi, f"{f5}/3", va="center", ha="left", fontsize=6.5,
                color="white" if f5 >= 2 else "#333333")
    ax.axvline(1 / len(d), color=C["full"], linewidth=1.0, linestyle=":")
    ax.text(1 / len(d) + 0.002, len(d) - 0.6, "uniform", fontsize=7,
            color=C["full"])
    ax.set_yticks(y)
    ax.set_yticklabels([f"{SITE_NAMES.get(n, n)} ({n})" for n in d["name"]],
                       fontsize=8)
    ax.set_xlim(0, float((d["att_mean"] + d["att_std"]).max()) * 1.35)
    ax.set_xlabel("Mean attention weight across 3 seeds (\u00b11 SD)")
    stable = [n for n, f in zip(d["name"], d["freq_top5"]) if f == 3]
    ax.set_title("Remote-sensing site attention, gated S4\n"
                 f"(S4 does not clear M0, so this is a stability diagnostic; "
                 f"top-5 in all seeds: {', '.join(stable) or 'none'})",
                 loc="left", fontsize=9)
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    save(fig, "fig_B_2_rs_site_attention")


# --------------------------------------------------------------------------
# F6 - weekly gate paths across seeds
# --------------------------------------------------------------------------
def fig6_gate_band() -> None:
    # Deep S3 is the M0-clearing arm, so gate paths come from M3 (§3.10).
    band = pd.read_csv(DEEP / "M3_Deep" / "deep_m3_gate_band_weekly.csv",
                       parse_dates=["date"]).set_index("date")
    seeds = {}
    for s in (42, 1, 2):
        p = DEEP / "M3_Deep" / f"deep_m3_gate_weekly_seed{s}.csv"
        if p.exists():
            seeds[s] = pd.read_csv(p, parse_dates=["date"]).set_index("date")

    mods = [("finance", "Financial time series", C["xgb"]),
            ("shipping", "Shipping", C["gated"]),
            ("rs", "Remote sensing", C["pos"])]
    # S3 carries no RS branch; keep only modalities present in the bundle.
    mods = [m for m in mods if f"gate_{m[0]}_mean" in band.columns]

    fig, axes = plt.subplots(len(mods), 1, figsize=(8.2, 2.2 * len(mods)),
                             sharex=True, squeeze=False)
    axes = axes[:, 0]
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
    axes[0].set_title("Weekly modality gates (Deep S3): week-to-week paths differ "
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
    """RQ1 and RQ2 families: raw DM-HLN p against its Holm-adjusted value.

    Every test is DM-HLN on reconstructed-price squared error. Holm runs within
    each frozen family, so the adjustment differs by panel (15 vs 14 tests).
    """
    tab = pd.read_csv(TESTS / "test_table_main.csv")
    panels = [
        ("RQ1", "RQ1: added modality within a learner\n"
                "(reference = same learner at S1)"),
        ("RQ2", "RQ2: Deep against Flat, and fusion mechanisms\n"
                "(reference = Flat at the same information set)"),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(11.6, 5.4))
    for ax, (fam, title) in zip(axes, panels):
        d = (tab[tab["family"] == fam]
             .sort_values("p_raw", ascending=False).reset_index(drop=True))
        y = np.arange(len(d))
        ax.hlines(y, d["p_raw"], d["p_holm"], color=C["grey"], linewidth=0.9,
                  zorder=1)
        ax.scatter(d["p_raw"], y, s=26, color=C["gated"], zorder=3,
                   label="raw p")
        ax.scatter(d["p_holm"], y, s=26, color=C["xgb"], zorder=3,
                   marker="D", label="Holm-adjusted p")
        for yi, p in zip(y, d["p_raw"]):
            if p < 0.05:
                ax.text(p + 0.008, yi + 0.28, f"{p:.4f}", ha="left",
                        fontsize=6.8, color=C["gated"])

        ax.axvline(0.05, color=C["neg"], linewidth=1.1, linestyle="--")
        ax.text(0.055, -1.15, "p = 0.05", color=C["neg"], fontsize=7.5,
                va="center")
        ax.set_yticks(y)
        ax.set_yticklabels(d["reference"] + "  \u2192  " + d["candidate"],
                           fontsize=7.2)
        ax.set_xlim(-0.02, 1.06)
        ax.set_ylim(-1.5, len(d) - 0.4)
        ax.set_xlabel(f"DM-HLN p-value  (Holm within {len(d)} tests)")
        ax.set_title(title, loc="left", fontsize=9)
        ax.grid(axis="y", visible=False)
    axes[0].legend(loc="lower right")

    shown = tab[tab["family"].isin([f for f, _ in panels])]
    n_raw = int((shown["p_raw"] < 0.05).sum())
    n_holm = int((shown["p_holm"] < 0.05).sum())
    tail = ("none survives Holm adjustment" if n_holm == 0
            else f"{n_holm} survives Holm adjustment")
    fig.suptitle(f"Incremental-information tests: {n_raw} of {len(shown)} raw "
                 f"p-values fall below 5%, {tail}",
                 x=0.008, ha="left", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
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
    "B.2": figB2_rs_site_attention,
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
