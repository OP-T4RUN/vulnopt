"""
Unit tests for configurable risk-priority scoring model.
"""

import numpy as np
import pandas as pd
import pytest

from vulnopt.risk_model import (
    ASSET_FOCUSED_WEIGHTS,
    DEFAULT_WEIGHTS,
    THREAT_FOCUSED_WEIGHTS,
    calculate_risk_scores,
    generate_dirichlet_weight_samples,
    validate_weights,
)


def test_validate_weights():
    # Valid default weights
    validate_weights(DEFAULT_WEIGHTS)

    # Missing a key
    with pytest.raises(ValueError, match="must have exactly keys"):
        validate_weights({"cvss": 0.5, "epss": 0.5})

    # Negative weight
    with pytest.raises(ValueError, match="nonnegative"):
        validate_weights({"cvss": -0.1, "epss": 0.35, "kev": 0.25, "criticality": 0.25, "exposure": 0.25})

    # Weights don't sum to 1
    with pytest.raises(ValueError, match="must sum to 1"):
        validate_weights({"cvss": 0.2, "epss": 0.2, "kev": 0.2, "criticality": 0.2, "exposure": 0.1})


def test_calculate_risk_scores_bounds_and_components():
    sample = pd.DataFrame([
        {
            "finding_id": "FND-001",
            "cve_id": "CVE-2021-44228",
            "cvss_base": 10.0,
            "epss": 0.97,
            "is_kev": 1,
            "criticality": 5,
            "exposure": "internet",
        },
        {
            "finding_id": "FND-002",
            "cve_id": "CVE-2023-0001",
            "cvss_base": 0.0,
            "epss": 0.0,
            "is_kev": 0,
            "criticality": 1,
            "exposure": "restricted",
        }
    ])

    scored = calculate_risk_scores(sample)

    assert "risk_score" in scored.columns
    assert "score_component_cvss" in scored.columns
    assert "score_component_epss" in scored.columns
    assert "score_component_kev" in scored.columns
    assert "score_component_criticality" in scored.columns
    assert "score_component_exposure" in scored.columns

    # Severe internet finding should have very high score
    assert scored.loc[scored["finding_id"] == "FND-001", "risk_score"].iloc[0] > 90.0

    # Low finding should have low score
    assert scored.loc[scored["finding_id"] == "FND-002", "risk_score"].iloc[0] < 10.0

    # All scores strictly in [0, 100]
    assert scored["risk_score"].between(0.0, 100.0).all()


def test_risk_monotonicity():
    """Higher CVSS, higher EPSS, KEV status, or criticality should strictly increase score."""
    base_case = {
        "finding_id": "FND-BASE",
        "cve_id": "CVE-2023-0001",
        "cvss_base": 5.0,
        "epss": 0.10,
        "is_kev": 0,
        "criticality": 2,
        "exposure": "internal",
    }
    higher_crit = {**base_case, "finding_id": "FND-HIGH-CRIT", "criticality": 5}
    higher_epss = {**base_case, "finding_id": "FND-HIGH-EPSS", "epss": 0.90}
    kev_active = {**base_case, "finding_id": "FND-KEV", "is_kev": 1}

    df = pd.DataFrame([base_case, higher_crit, higher_epss, kev_active])
    scored = calculate_risk_scores(df)

    base_score = scored.loc[scored["finding_id"] == "FND-BASE", "risk_score"].iloc[0]
    crit_score = scored.loc[scored["finding_id"] == "FND-HIGH-CRIT", "risk_score"].iloc[0]
    epss_score = scored.loc[scored["finding_id"] == "FND-HIGH-EPSS", "risk_score"].iloc[0]
    kev_score = scored.loc[scored["finding_id"] == "FND-KEV", "risk_score"].iloc[0]

    assert crit_score > base_score
    assert epss_score > base_score
    assert kev_score > base_score


def test_sensitivity_presets_and_dirichlet():
    sample = pd.DataFrame([{
        "finding_id": "FND-001",
        "cve_id": "CVE-2021-44228",
        "cvss_base": 7.0,
        "epss": 0.85,
        "is_kev": 1,
        "criticality": 2,
        "exposure": "restricted",
    }])

    # Scored with threat-focused weights
    threat_scored = calculate_risk_scores(sample, weights=THREAT_FOCUSED_WEIGHTS)
    # Scored with asset-focused weights
    asset_scored = calculate_risk_scores(sample, weights=ASSET_FOCUSED_WEIGHTS)

    # Because EPSS and KEV are high, threat-focused score should be noticeably higher
    assert threat_scored["risk_score"].iloc[0] > asset_scored["risk_score"].iloc[0]

    # Test Dirichlet sample generator
    dirichlet_weights = generate_dirichlet_weight_samples(n_samples=25, seed=42)
    assert len(dirichlet_weights) == 25
    for w in dirichlet_weights:
        validate_weights(w)

