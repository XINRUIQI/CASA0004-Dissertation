"""
Canonical model display names + output paths.

Data modalities (feature dictionary / processed/M1|M2|M3) stay M1/M2/M3.
Only model labels and baseline output directories use the Flat/Deep scheme.
"""

from __future__ import annotations

from pathlib import Path

# Feature-selection keys (unchanged) -> model display label
FLAT_LABELS = {
    "M1": "M1_Flat",
    "M2": "M2_Flat",
    "M3": "M3_Flat",
    "M4": "M4_Flat",
}

NESTED_BASE_FEAT = "M1"          # select_features key
NESTED_BASE_LABEL = "M1_Flat"    # column / metrics name

# Deep config key -> full model column name (also CONFIGS short name)
DEEP_MODEL_NAMES = {
    "m1_deep": "M1_Deep",
    "m_ship_gnn": "M_ship_GNN",
    "m_rs_deep": "M_rs_deep",
    "m3_deep_gated": "M3_Deep_gated",
    "m3_deep_concat": "M3_Deep_Concat",
    "m3_deep_xattn": "M3_Deep_XAttn",
    "m2_deep_gated": "M2_Deep_gated",
    "m2_deep_concat": "M2_Deep_Concat",
    "m2_deep_xattn": "M2_Deep_XAttn",
    "m4_deep_gated": "M4_Deep_gated",
    "m4_deep_xattn": "M4_Deep_XAttn",
    "m4_deep_concat": "M4_Deep_Concat",
}

# Old CLI / CONFIGS keys -> new keys
DEEP_CONFIG_ALIASES = {
    "fin": "m1_deep",
    "ship": "m_ship_gnn",
    "rs": "m_rs_deep",
    "finship": "m3_deep_gated",
    "fusion": "m3_deep_gated",
    "finship_concat": "m3_deep_concat",
    "finship_xattn": "m3_deep_xattn",
    "finrs": "m2_deep_gated",
    "finrs_concat": "m2_deep_concat",
    "finrs_xattn": "m2_deep_xattn",
    "m4rep": "m4_deep_gated",
    "m4xattn": "m4_deep_xattn",
    "m4concat": "m4_deep_concat",
}


def resolve_deep_config(key: str) -> str:
    """Map old or new deep config key to canonical key."""
    if key in DEEP_MODEL_NAMES:
        return key
    if key in DEEP_CONFIG_ALIASES:
        return DEEP_CONFIG_ALIASES[key]
    raise KeyError(f"Unknown deep config {key!r}; "
                   f"expected one of {list(DEEP_MODEL_NAMES) + list(DEEP_CONFIG_ALIASES)}")


def flat_label(modality: str) -> str:
    """M1 -> M1_Flat. Accepts either form."""
    if modality in FLAT_LABELS:
        return FLAT_LABELS[modality]
    if modality in FLAT_LABELS.values():
        return modality
    raise KeyError(f"Unknown flat modality {modality!r}")


def flat_feat_key(modality: str) -> str:
    """M1_Flat -> M1 for select_features."""
    if modality in FLAT_LABELS:
        return modality
    rev = {v: k for k, v in FLAT_LABELS.items()}
    if modality in rev:
        return rev[modality]
    raise KeyError(f"Unknown flat modality {modality!r}")


def flat_out_dir(root: Path, modality: str) -> Path:
    return root / "results" / "baselines" / "Flat" / flat_label(modality)


# Deep output tiers (mirrors Flat/M*_Flat under baselines/Deep/)
DEEP_CROSS = "_cross"
DEEP_TIER_LABELS = {
    "M1": "M1_Deep",
    "M2": "M2_Deep",
    "M3": "M3_Deep",
    "M4": "M4_Deep",
    "cross": DEEP_CROSS,
}

# CONFIG key -> tier folder
DEEP_CONFIG_TIER: dict[str, str] = {
    "m1_deep": "M1_Deep",
    "m_rs_deep": "M2_Deep",
    "m2_deep_gated": "M2_Deep",
    "m2_deep_concat": "M2_Deep",
    "m2_deep_xattn": "M2_Deep",
    "m_ship_gnn": "M3_Deep",
    "m3_deep_gated": "M3_Deep",
    "m3_deep_concat": "M3_Deep",
    "m3_deep_xattn": "M3_Deep",
    "m4_deep_gated": "M4_Deep",
    "m4_deep_concat": "M4_Deep",
    "m4_deep_xattn": "M4_Deep",
}

# Display model name -> tier folder
DEEP_MODEL_TIER: dict[str, str] = {
    DEEP_MODEL_NAMES[k]: v for k, v in DEEP_CONFIG_TIER.items()
}

# Per-tier model columns for slim baseline_predictions exports
DEEP_TIER_MODELS: dict[str, list[str]] = {
    "M1_Deep": ["M1_Deep"],
    "M2_Deep": ["M2_Deep_gated", "M2_Deep_Concat", "M2_Deep_XAttn", "M_rs_deep"],
    "M3_Deep": ["M3_Deep_gated", "M3_Deep_Concat", "M3_Deep_XAttn", "M_ship_GNN"],
    "M4_Deep": ["M4_Deep_gated", "M4_Deep_Concat", "M4_Deep_XAttn"],
}

# Rows always kept in per-tier metrics tables
DEEP_METRICS_ANCHORS = ["M0_RW", "Naive_DirPersist", "M1_Flat_Ridge", "M1_Flat_XGB"]


def deep_base_dir(root: Path) -> Path:
    return root / "results" / "baselines" / "Deep"


def deep_out_dir(root: Path, tier: str = "cross") -> Path:
    """Output directory for a deep tier.

    ``tier`` accepts M1..M4, cross, folder labels (M1_Deep, _cross), config keys
    (m2_deep_gated), or display model names (M2_Deep_gated).
    """
    if tier in DEEP_TIER_LABELS:
        label = DEEP_TIER_LABELS[tier]
    elif tier in DEEP_TIER_LABELS.values():
        label = tier
    elif tier in DEEP_CONFIG_TIER:
        label = DEEP_CONFIG_TIER[tier]
    elif tier in DEEP_MODEL_NAMES:
        label = DEEP_MODEL_TIER[DEEP_MODEL_NAMES[tier]]
    elif tier in DEEP_MODEL_TIER:
        label = DEEP_MODEL_TIER[tier]
    else:
        raise KeyError(
            f"Unknown deep tier {tier!r}; expected one of "
            f"{list(DEEP_TIER_LABELS) + list(DEEP_TIER_LABELS.values())}"
        )
    return deep_base_dir(root) / label


def deep_cross_predictions(root: Path) -> Path:
    return deep_out_dir(root, "cross") / "deep_predictions.csv"


def m1_flat_predictions(root: Path) -> Path:
    return flat_out_dir(root, "M1") / "baseline_predictions.csv"


# ---------------------------------------------------------------------------
# One-shot CSV migration: old token -> new token (applied to columns, index,
# and selected cell values). Longer keys first to avoid partial replacements.
# ---------------------------------------------------------------------------
CSV_TOKEN_RENAMES = {
    # Flat model names (old -> new)
    "M1_Ridge": "M1_Flat_Ridge",
    "M1_XGB": "M1_Flat_XGB",
    "M2_Ridge": "M2_Flat_Ridge",
    "M2_XGB": "M2_Flat_XGB",
    "M3_Ridge": "M3_Flat_Ridge",
    "M3_XGB": "M3_Flat_XGB",
    "M4_Ridge": "M4_Flat_Ridge",
    "M4_XGB": "M4_Flat_XGB",
    # Deep composed names (label_mname) — order: longest / most specific first
    "Mfinship_FinshipConcat": "M3_Deep_Concat",
    "Mfinship_FinshipXattn": "M3_Deep_XAttn",
    "Mfinship_Finship": "M3_Deep_gated",
    "Mfinrs_FinRSConcat": "M2_Deep_Concat",
    "Mfinrs_FinRSXattn": "M2_Deep_XAttn",
    "Mfinrs_FinRS": "M2_Deep_gated",
    "Mconcat_M4concat": "M4_Deep_Concat",
    "Mxattn_M4xattn": "M4_Deep_XAttn",
    "Mfull_M4rep": "M4_Deep_gated",
    "Mfusion_Fusion": "M3_Deep_gated",
    "Mfin_TCN": "M1_Deep",
    "Mship_GNN": "M_ship_GNN",
    "Mrs_RS": "M_rs_deep",
    # Display-only / combo labels in fusion matrix
    "Mfinship": "M3_Deep",
    "Mfinrs": "M2_Deep",
    "Mfull": "M4_Deep",
    "Mxattn": "M4_Deep_XAttn",
    "Mconcat": "M4_Deep_Concat",
    "Mfin": "M1_Deep",
    "Mship": "M_ship_GNN",
    "Mrs": "M_rs_deep",
    "Mfusion": "M3_Deep_gated",
    # Config keys stored in sweep / matrix CSVs
    "finship_concat": "m3_deep_concat",
    "finship_xattn": "m3_deep_xattn",
    "finrs_concat": "m2_deep_concat",
    "finrs_xattn": "m2_deep_xattn",
    "m4concat": "m4_deep_concat",
    "m4xattn": "m4_deep_xattn",
    "m4rep": "m4_deep_gated",
    "finship": "m3_deep_gated",
    "finrs": "m2_deep_gated",
    # NOTE: do NOT map bare "fusion" — it collides with the fusion-mechanism
    # column name (concat/gated/xattn) in deep_fusion_matrix.csv.
}
