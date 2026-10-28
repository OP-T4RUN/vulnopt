"""
Configurable Risk-Priority Scoring Model for VulnOpt.

Combines public threat intelligence (CVSS, EPSS, KEV) with enterprise asset context
(Criticality, Exposure) to produce an explainable 0–100 risk-reduction score (R_i).
"""

from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd


DEFAULT_WEIGHTS: Dict[str, float] = {
    "cvss": 0.25,
    "epss": 0.25,
    "kev": 0.25,
    "criticality": 0.15,
    "exposure": 0.10,
}

THREAT_FOCUSED_WEIGHTS: Dict[str, float] = {
    "cvss": 0.15,
    "epss": 0.40,
    "kev": 0.35,
    "criticality": 0.05,
    "exposure": 0.05,
}

ASSET_FOCUSED_WEIGHTS: Dict[str, float] = {
    "cvss": 0.15,
    "epss": 0.10,
    "kev": 0.15,
    "criticality": 0.35,
    "exposure": 0.25,
}

EXPOSURE_WEIGHTS: Dict[str, float] = {
    "internet": 1.0,
    "internal": 0.5,
    "restricted": 0.2,
}


def validate_weights(weights: Dict[str, float]) -> None:
    """Validate that weights contain all required keys, are nonnegative, and sum to 1."""
    required = set(DEFAULT_WEIGHTS.keys())
    if set(weights.keys()) != required:
        raise ValueError(f"Weights dictionary must have exactly keys: {required}")
    if any(not np.isfinite(v) or v < 0 for v in weights.values()):
        raise ValueError("All weights must be finite and nonnegative.")
    if not np.isclose(sum(weights.values()), 1.0, atol=1e-5):
        raise ValueError(f"Weights must sum to 1.0 (current sum: {sum(weights.values()):.5f}).")


def calculate_risk_scores(
    df: pd.DataFrame,
    weights: Optional[Dict[str, float]] = None,
) -> pd.DataFrame:
    """
    Calculate normalized components and the final 0–100 risk score (R_i) for each finding.

    Normalized components:
    - S_i = cvss_base / 10.0 (Severity)
    - E_i = epss_score (Exploit likelihood)
    - K_i = is_kev (Known in-the-wild exploitation)
    - A_i = (criticality - 1) / 4.0 (Asset criticality)
    - X_i = exposure factor (1.0 for internet, 0.5 for internal, 0.2 for restricted)

    Score:
    Q_i = w_s*S_i + w_e*E_i + w_k*K_i + w_a*A_i + w_x*X_i
    R_i = 100 * Q_i
    """
    if weights is None:
        weights = DEFAULT_WEIGHTS
    validate_weights(weights)

    out = df.copy()

    # Required columns check
    req_cols = ["cvss_base", "is_kev", "criticality", "exposure"]
    for col in req_cols:
        if col not in out.columns:
            raise ValueError(f"Missing required column in dataframe: '{col}'")

    cvss = out["cvss_base"].astype(float)
    if cvss.isna().any() or not cvss.between(0.0, 10.0).all():
        raise ValueError("cvss_base must be non-null and in range [0, 10].")

    # EPSS handling: use epss_score if present, or epss with 0.0 imputation
    if "epss_score" in out.columns:
        epss = out["epss_score"].fillna(0.0).astype(float)
    elif "epss" in out.columns:
        epss = out["epss"].fillna(0.0).astype(float)
    else:
        epss = pd.Series(0.0, index=out.index)

    if not epss.between(0.0, 1.0).all():
        raise ValueError("epss must be in range [0, 1].")

    kev = out["is_kev"].astype(float)
    if not kev.isin([0.0, 1.0]).all():
        raise ValueError("is_kev must be 0 or 1.")

    crit = out["criticality"].astype(float)
    if not crit.between(1.0, 5.0).all():
        raise ValueError("criticality must be integer or float in range [1, 5].")

    exposure_num = out["exposure"].map(EXPOSURE_WEIGHTS)
    if exposure_num.isna().any():
        invalid = out.loc[exposure_num.isna(), "exposure"].unique()
        raise ValueError(f"Invalid exposure value(s) encountered: {invalid}")

    # Track imputation flag
    if "epss_missing" not in out.columns:
        out["epss_imputed"] = out.get("epss", pd.Series(np.nan, index=out.index)).isna()
    else:
        out["epss_imputed"] = out["epss_missing"].astype(bool)

    # 1. Normalized components
    out["score_component_cvss"] = cvss / 10.0
    out["score_component_epss"] = epss
    out["score_component_kev"] = kev
    out["score_component_criticality"] = (crit - 1.0) / 4.0
    out["score_component_exposure"] = exposure_num

    # 2. Weighted score Q_i and final priority R_i (0 to 100)
    q_i = (
        weights["cvss"] * out["score_component_cvss"]
        + weights["epss"] * out["score_component_epss"]
        + weights["kev"] * out["score_component_kev"]
        + weights["criticality"] * out["score_component_criticality"]
        + weights["exposure"] * out["score_component_exposure"]
    )

    out["risk_score"] = np.round(100.0 * q_i, 2)
    return out


def generate_dirichlet_weight_samples(n_samples: int = 100, seed: int = 42) -> List[Dict[str, float]]:
    """
    Generate n random valid weight vectors using a Dirichlet distribution centered at default weights.
    Used for sensitivity analysis to test ranking stability across weight variations.
    """
    rng = np.random.default_rng(seed)
    keys = ["cvss", "epss", "kev", "criticality", "exposure"]
    alpha = [DEFAULT_WEIGHTS[k] * 10.0 for k in keys]  # Concentration around defaults
    samples = rng.dirichlet(alpha, size=n_samples)

    weight_configs: List[Dict[str, float]] = []
    for row in samples:
        cfg = {k: float(row[i]) for i, k in enumerate(keys)}
        weight_configs.append(cfg)
    return weight_configs


def build_remediation_candidates(
    findings_df: pd.DataFrame,
    vulnerabilities_df: pd.DataFrame,
    assets_df: pd.DataFrame,
    weights: Optional[Dict[str, float]] = None,
) -> pd.DataFrame:
    """
    Join finding assignments with public vulnerability intelligence and synthetic asset metadata,
    and calculate risk-priority scores.
    """
    # 1. Merge finding with asset details
    merged = findings_df.merge(assets_df, on="asset_id", how="left", suffixes=("", "_asset"))

    # 2. Merge with vulnerability intelligence
    vuln_subset = vulnerabilities_df[[
        "cve_id", "cvss_base", "cvss_version", "cvss_vector", "cvss_severity",
        "epss", "epss_score", "epss_percentile", "epss_missing", "is_kev",
        "kev_ransomware", "cwe_ids"
    ]].drop_duplicates("cve_id")

    merged = merged.merge(vuln_subset, on="cve_id", how="left")

    # 3. Calculate risk scores
    scored_df = calculate_risk_scores(merged, weights=weights)
    return scored_df

