"""Metric groups and impact-model config (shared by app and engine)."""

from __future__ import annotations

COMPARISON_IMPACT_KEYS: tuple[str, ...] = (
    "impact_passes_p90",
    "phi_p90",
    "long_impact_passes",
    "impact_passes",
    "high_impact_passes",
    "aggression_aip",
)

COMPARISON_PROGRESSION_KEYS: tuple[str, ...] = (
    "progressive_passes_p90",
    "final_third_passes_p90",
    "long_balls",
    "progressive_passes",
    "final_third_passes",
    "key_passes",
)

COMPARISON_CARD_GROUPS: dict[str, tuple[str, ...]] = {
    "comparison_impact": COMPARISON_IMPACT_KEYS,
    "comparison_progression": COMPARISON_PROGRESSION_KEYS,
}

# Configuração ativa do app (fixa).
CLASSIFICATION_MODEL_OPT1_SHORT_FT = "opt1_short_ft"
CLASSIFICATION_MODEL_DEFAULT = CLASSIFICATION_MODEL_OPT1_SHORT_FT

CLASSIFICATION_MODEL_LABELS: dict[str, str] = {
    "atual": "Atual (ganho relativo)",
    CLASSIFICATION_MODEL_OPT1_SHORT_FT: "Opção 1 + via curta",
}

TIER_MODEL_PERCENTILE_P65_P85 = "percentile_p65_p85"
TIER_MODEL_DEFAULT = TIER_MODEL_PERCENTILE_P65_P85
TIER_MODEL_FIXED_30_50 = "fixed_30_50"
TIER_MODEL_PERCENTILE_P70_P90 = "percentile_p70_p90"

TIER_MODEL_LABELS: dict[str, str] = {
    TIER_MODEL_DEFAULT: "Atual (0,30 / 0,62)",
    TIER_MODEL_FIXED_30_50: "Fixo (0,30 / 0,50)",
    TIER_MODEL_PERCENTILE_P65_P85: "Percentil (p65 / p85)",
    TIER_MODEL_PERCENTILE_P70_P90: "Percentil (p70 / p90)",
}

TIER_MODEL_PERCENTILES: dict[str, tuple[int, int]] = {
    TIER_MODEL_PERCENTILE_P65_P85: (65, 85),
    TIER_MODEL_PERCENTILE_P70_P90: (70, 90),
}

# Superfície xT (fixa: opção 2 — mapa = passes).
XT_SURFACE_MODE_ALIGNED = "aligned_display"
XT_SURFACE_MODE_DEFAULT = XT_SURFACE_MODE_ALIGNED
XT_SURFACE_MODE_ATUAL = "atual"
XT_SURFACE_MODE_MONOTONIC = "monotonic_fine"

XT_SURFACE_MODE_LABELS: dict[str, str] = {
    XT_SURFACE_MODE_ATUAL: "Atual (mapa ≠ passes)",
    XT_SURFACE_MODE_ALIGNED: "Opção 2 — mapa = passes (pós-processo)",
    XT_SURFACE_MODE_MONOTONIC: "Opção 3 — grid fino monotônico",
}

XT_SURFACE_MODE_DESCRIPTIONS: dict[str, str] = {
    XT_SURFACE_MODE_ATUAL: (
        "Mapa com pós-processamento e boosts; passes usam o grid fino bruto "
        "(comportamento atual — pode divergir do mapa)."
    ),
    XT_SURFACE_MODE_ALIGNED: (
        "Passes interpolam a mesma superfície do mapa 16×12 (pós-processo, "
        "simetria e boosts), upsampled para o campo."
    ),
    XT_SURFACE_MODE_MONOTONIC: (
        "Grid fino com xT monotônico no último terço (x≥80); mapa mostra médias "
        "por quadrante sem pós-processamento de escada."
    ),
}

# Backward-compatible aliases (tier model only).
IMPACT_MODEL_DEFAULT = TIER_MODEL_DEFAULT
IMPACT_MODEL_FIXED_30_50 = TIER_MODEL_FIXED_30_50
IMPACT_MODEL_PERCENTILE_P70_P90 = TIER_MODEL_PERCENTILE_P70_P90
IMPACT_MODEL_PERCENTILE_P65_P85 = TIER_MODEL_PERCENTILE_P65_P85
IMPACT_MODEL_LABELS = TIER_MODEL_LABELS


def normalize_classification_model(model: str | None) -> str:
    key = str(model or CLASSIFICATION_MODEL_DEFAULT).strip().lower()
    return key if key in CLASSIFICATION_MODEL_LABELS else CLASSIFICATION_MODEL_DEFAULT


def normalize_tier_model(model: str | None) -> str:
    key = str(model or TIER_MODEL_DEFAULT).strip().lower()
    return key if key in TIER_MODEL_LABELS else TIER_MODEL_DEFAULT


def normalize_impact_model(model: str | None) -> str:
    """Alias for normalize_tier_model (legacy imports)."""
    return normalize_tier_model(model)


def normalize_xt_surface_mode(mode: str | None) -> str:
    key = str(mode or XT_SURFACE_MODE_DEFAULT).strip().lower()
    return key if key in XT_SURFACE_MODE_LABELS else XT_SURFACE_MODE_DEFAULT
