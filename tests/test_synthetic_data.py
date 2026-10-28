"""
Unit tests for synthetic asset generation and hybrid assignment.
"""

import numpy as np
import pandas as pd
import pytest

from vulnopt.synthetic_data import (
    estimate_remediation_resources,
    generate_finding_assignments,
    generate_synthetic_assets,
    infer_preferred_asset_types,
)


def test_infer_preferred_asset_types_keywords():
    # Database keyword
    db_types = infer_preferred_asset_types("SQL Injection in user login parameter", "CWE-89")
    assert "database" in db_types

    # Web keyword
    web_types = infer_preferred_asset_types("WordPress plugin allows stored XSS", "CWE-79")
    assert "web_server" in web_types

    # Identity keyword
    id_types = infer_preferred_asset_types("Active Directory LDAP credential exposure", "CWE-287")
    assert "identity_service" in id_types

    # Endpoint keyword
    ep_types = infer_preferred_asset_types("Google Chrome browser heap out-of-bounds write", "CWE-787")
    assert "endpoint" in ep_types


def test_infer_preferred_asset_types_fallback():
    # Empty or generic descriptions must fallback cleanly without crashing
    types_empty = infer_preferred_asset_types(None, None)
    assert len(types_empty) > 0
    assert "application_server" in types_empty

    types_generic = infer_preferred_asset_types("Unspecified vulnerability in core library", None)
    assert len(types_generic) > 0


def test_generate_synthetic_assets():
    assets = generate_synthetic_assets(num_assets=50, seed=123)
    assert len(assets) == 50
    assert "asset_id" in assets.columns
    assert "criticality" in assets.columns
    assert "exposure" in assets.columns
    assert "internet_facing" in assets.columns

    # Criticality in [1, 5]
    assert assets["criticality"].between(1, 5).all()

    # Internet-facing boolean consistency
    for _, row in assets.iterrows():
        if row["exposure"] == "internet":
            assert row["internet_facing"] is True
        else:
            assert row["internet_facing"] is False


def test_estimate_remediation_resources():
    rng = np.random.default_rng(42)
    cost, hours, downtime = estimate_remediation_resources(
        criticality=5,
        asset_type="database",
        cvss_base=9.8,
        rng=rng,
    )
    assert cost > 0.0
    assert hours > 0.0
    assert downtime >= 0.0


def test_generate_finding_assignments():
    mock_vulns = pd.DataFrame([
        {"cve_id": "CVE-2023-0001", "cvss_base": 7.5, "description": "WordPress plugin XSS", "cwe_ids": "CWE-79"},
        {"cve_id": "CVE-2023-0002", "cvss_base": 9.0, "description": "PostgreSQL SQL injection", "cwe_ids": "CWE-89"},
    ])
    mock_assets = generate_synthetic_assets(num_assets=20, seed=42)

    findings = generate_finding_assignments(
        vulnerabilities_df=mock_vulns,
        assets_df=mock_assets,
        num_findings=10,
        seed=42,
        use_hybrid_matching=True,
    )

    assert len(findings) == 10
    assert "finding_id" in findings.columns
    assert "cve_id" in findings.columns
    assert "asset_id" in findings.columns
    assert "patch_cost" in findings.columns
    assert "engineer_hours" in findings.columns
    assert "downtime_hours" in findings.columns

    # Foreign keys must exist in asset inventory
    assert findings["asset_id"].isin(mock_assets["asset_id"]).all()
    # Unique finding IDs
    assert findings["finding_id"].nunique() == 10
    # Positive resource values
    assert (findings["patch_cost"] > 0).all()
    assert (findings["engineer_hours"] > 0).all()
    assert (findings["downtime_hours"] >= 0).all()


def test_synthetic_reproducibility():
    assets_1 = generate_synthetic_assets(num_assets=25, seed=999)
    assets_2 = generate_synthetic_assets(num_assets=25, seed=999)
    pd.testing.assert_frame_equal(assets_1, assets_2)

