"""
Synthetic Enterprise Asset & Finding Generator for VulnOpt.

Generates simulated enterprise assets and maps real CVE findings to those assets
with realistic, right-skewed remediation cost, engineer-hours, and downtime estimates.
All generated data is strictly labeled as synthetic.
"""

import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd


ASSET_TYPES = [
    "web_server",
    "application_server",
    "database",
    "endpoint",
    "identity_service",
    "internal_service",
]

BUSINESS_UNITS = [
    "Engineering",
    "Finance",
    "Operations",
    "Sales",
    "Human Resources",
    "Legal",
]

ENVIRONMENTS = ["production", "staging", "development"]

# Keyword patterns for hybrid device matching
KEYWORD_RULES = {
    "database": re.compile(
        r"\b(sql|database|mysql|postgres|oracle|mongodb|redis|sqlite|mariadb|dbms|nosql)\b",
        re.IGNORECASE,
    ),
    "web_server": re.compile(
        r"\b(wordpress|apache|nginx|http|php|xss|csrf|cross-site|django|flask|html|drupal|joomla|web|frontend)\b",
        re.IGNORECASE,
    ),
    "identity_service": re.compile(
        r"\b(active directory|ldap|oauth|saml|jwt|token|privilege escalation|credential|auth|kerberos|sso)\b",
        re.IGNORECASE,
    ),
    "endpoint": re.compile(
        r"\b(windows 10|windows 11|macos|chrome|firefox|safari|office|excel|word|pdf|reader|acrobat|desktop|browser|client)\b",
        re.IGNORECASE,
    ),
}


def infer_preferred_asset_types(description: Optional[str], cwe_ids: Optional[str] = None) -> List[str]:
    """
    Infer preferred asset types using lightweight keyword matching from CVE description and CWEs.
    Returns a list of preferred asset types, or fallback to application/internal servers.
    """
    text = ""
    if isinstance(description, str):
        text += description + " "
    if isinstance(cwe_ids, str):
        text += cwe_ids

    if not text.strip():
        return ["application_server", "internal_service"]

    matches: List[str] = []
    for asset_type, pattern in KEYWORD_RULES.items():
        if pattern.search(text):
            matches.append(asset_type)

    if matches:
        return matches

    # Default fallback for general library/system flaws
    return ["application_server", "internal_service", "web_server"]


def generate_synthetic_assets(num_assets: int = 150, seed: int = 42) -> pd.DataFrame:
    """
    Generate simulated enterprise IT assets with realistic types, criticality, and network exposure.

    Criticality: 1 (lowest) to 5 (mission-critical), centered around 2-4.
    Exposure: internet (1.0), internal (0.5), restricted (0.2).
    """
    rng = np.random.default_rng(seed)

    # Asset type distribution: endpoints and web/app servers are common; DB & Identity are fewer
    type_probs = [0.25, 0.25, 0.12, 0.18, 0.08, 0.12]

    # Criticality distribution: Beta-binomial or categorical centered at 2-4
    crit_probs = [0.10, 0.25, 0.35, 0.20, 0.10]  # 1 to 5

    records: List[Dict[str, Any]] = []
    for i in range(1, num_assets + 1):
        asset_id = f"AST-{i:05d}"
        asset_type = rng.choice(ASSET_TYPES, p=type_probs)
        criticality = int(rng.choice([1, 2, 3, 4, 5], p=crit_probs))
        env = rng.choice(ENVIRONMENTS, p=[0.60, 0.25, 0.15])
        bu = rng.choice(BUSINESS_UNITS)

        # Exposure logic: web servers are frequently internet-facing; DBs and Identity are rarely
        if asset_type == "web_server":
            exposure = rng.choice(["internet", "internal", "restricted"], p=[0.70, 0.25, 0.05])
        elif asset_type == "database":
            exposure = rng.choice(["internet", "internal", "restricted"], p=[0.02, 0.58, 0.40])
        elif asset_type == "identity_service":
            exposure = rng.choice(["internet", "internal", "restricted"], p=[0.05, 0.65, 0.30])
        elif asset_type == "endpoint":
            exposure = rng.choice(["internet", "internal", "restricted"], p=[0.40, 0.55, 0.05])
        else:
            exposure = rng.choice(["internet", "internal", "restricted"], p=[0.15, 0.65, 0.20])

        records.append({
            "asset_id": asset_id,
            "asset_type": asset_type,
            "criticality": criticality,
            "exposure": exposure,
            "internet_facing": (exposure == "internet"),
            "business_unit": bu,
            "environment": env,
            "data_origin": "synthetic",
        })

    return pd.DataFrame(records)


def estimate_remediation_resources(
    criticality: int,
    asset_type: str,
    cvss_base: float,
    rng: np.random.Generator,
) -> Tuple[float, float, float]:
    """
    Generate right-skewed remediation estimates (Cost, Engineer-Hours, Downtime)
    with positive average correlation to asset criticality and CVSS base score.

    Returns:
        (patch_cost, engineer_hours, downtime_hours)
    """
    # Base multipliers from criticality (1 to 5) and CVSS (0 to 10)
    # Higher criticality -> more testing/staging -> higher cost & hours
    crit_mult = 1.0 + 0.15 * (criticality - 3)
    cvss_mult = 1.0 + 0.05 * (cvss_base - 5.0)

    # Asset type factor
    type_factors = {
        "database": 1.4,
        "identity_service": 1.3,
        "application_server": 1.1,
        "web_server": 1.0,
        "internal_service": 0.9,
        "endpoint": 0.7,
    }
    tf = type_factors.get(asset_type, 1.0)
    combined_mult = max(0.5, crit_mult * cvss_mult * tf)

    # 1. Patch Cost (Right-skewed Log-Normal: median ~$400, occasional $2,000+)
    raw_cost = rng.lognormal(mean=5.8, sigma=0.6) * combined_mult
    patch_cost = round(float(np.clip(raw_cost, 50.0, 6000.0)), 2)

    # 2. Engineer Hours (Right-skewed Log-Normal: median ~8 hrs, range 2 to 60+ hrs)
    raw_hours = rng.lognormal(mean=2.1, sigma=0.55) * combined_mult
    engineer_hours = round(float(np.clip(raw_hours, 1.0, 80.0)), 1)

    # 3. Downtime Hours (Most fixes require 0 or low downtime; DB/Infra occasionally 2-6 hrs)
    # Endpoints rarely cause service downtime
    if asset_type in ("endpoint", "internal_service") or rng.random() < 0.45:
        downtime_hours = 0.0
    else:
        raw_downtime = rng.exponential(scale=1.5) * tf
        downtime_hours = round(float(np.clip(raw_downtime, 0.25, 12.0)), 1)

    return patch_cost, engineer_hours, downtime_hours


def generate_finding_assignments(
    vulnerabilities_df: pd.DataFrame,
    assets_df: pd.DataFrame,
    num_findings: int = 1000,
    seed: int = 42,
    use_hybrid_matching: bool = True,
) -> pd.DataFrame:
    """
    Assign public CVEs to synthetic enterprise assets to create candidate remediation findings.
    Uses hybrid keyword matching when enabled, with deterministic random fallback.
    """
    if len(vulnerabilities_df) == 0:
        raise ValueError("Vulnerabilities dataset is empty.")
    if len(assets_df) == 0:
        raise ValueError("Assets dataset is empty.")

    rng = np.random.default_rng(seed)

    # Pre-index assets by type for fast hybrid lookup
    assets_by_type: Dict[str, pd.DataFrame] = {
        atype: assets_df[assets_df["asset_type"] == atype]
        for atype in ASSET_TYPES
    }
    all_asset_ids = assets_df["asset_id"].tolist()

    # Sample CVEs (with replacement if num_findings > len(vulnerabilities_df))
    replace_cves = num_findings > len(vulnerabilities_df)
    sampled_indices = rng.choice(len(vulnerabilities_df), size=num_findings, replace=replace_cves)
    sampled_cves = vulnerabilities_df.iloc[sampled_indices].reset_index(drop=True)

    findings: List[Dict[str, Any]] = []
    for idx, cve_row in sampled_cves.iterrows():
        finding_id = f"FND-{idx+1:05d}"
        cve_id = cve_row["cve_id"]
        cvss_base = float(cve_row.get("cvss_base", 5.0))

        # Determine target asset using hybrid matching or random fallback
        chosen_asset_id: Optional[str] = None
        if use_hybrid_matching:
            desc = cve_row.get("description", "")
            cwes = cve_row.get("cwe_ids", "")
            preferred_types = infer_preferred_asset_types(desc, cwes)

            # Find matching assets in the generated inventory
            candidate_assets: List[str] = []
            for ptype in preferred_types:
                sub_df = assets_by_type.get(ptype)
                if sub_df is not None and not sub_df.empty:
                    candidate_assets.extend(sub_df["asset_id"].tolist())

            if candidate_assets:
                chosen_asset_id = rng.choice(candidate_assets)

        # Fallback to any asset
        if not chosen_asset_id:
            chosen_asset_id = rng.choice(all_asset_ids)

        asset_row = assets_df[assets_df["asset_id"] == chosen_asset_id].iloc[0]
        criticality = int(asset_row["criticality"])
        asset_type = str(asset_row["asset_type"])

        # Estimate resources
        cost, hours, downtime = estimate_remediation_resources(
            criticality=criticality,
            asset_type=asset_type,
            cvss_base=cvss_base,
            rng=rng,
        )

        findings.append({
            "finding_id": finding_id,
            "cve_id": cve_id,
            "asset_id": chosen_asset_id,
            "patch_cost": cost,
            "engineer_hours": hours,
            "downtime_hours": downtime,
            "data_origin": "synthetic",
        })

    return pd.DataFrame(findings)


def build_benchmark_datasets(
    vulnerabilities_path: Path = Path("data/processed/vulnerabilities.csv"),
    output_dir: Path = Path("data/synthetic"),
    seed: int = 42,
) -> Dict[str, Path]:
    """
    Generate the assets catalog and the three benchmark scales:
    100 (Small), 1,000 (Medium), and 5,000 (Large).
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    vuln_df = pd.read_csv(vulnerabilities_path)

    # 1. Generate master enterprise asset inventory (300 assets)
    assets_df = generate_synthetic_assets(num_assets=300, seed=seed)
    assets_path = output_dir / "assets.csv"
    assets_df.to_csv(assets_path, index=False)
    print(f"Generated {len(assets_df)} synthetic assets -> {assets_path}")

    # 2. Generate benchmark finding assignments
    scales = {
        "small_100": (100, output_dir / "finding_assignments_100.csv"),
        "medium_1000": (1000, output_dir / "finding_assignments_1000.csv"),
        "large_5000": (5000, output_dir / "finding_assignments_5000.csv"),
    }

    generated_paths: Dict[str, Path] = {"assets": assets_path}
    for key, (count, path) in scales.items():
        sub_assets = assets_df if count >= 1000 else assets_df.iloc[:50]
        findings_df = generate_finding_assignments(
            vulnerabilities_df=vuln_df,
            assets_df=sub_assets,
            num_findings=count,
            seed=seed,
            use_hybrid_matching=True,
        )
        findings_df.to_csv(path, index=False)
        generated_paths[key] = path
        print(f"Generated benchmark {key} ({len(findings_df)} findings) -> {path}")

    # Also save standard finding_assignments.csv defaulting to medium (1000)
    default_path = output_dir / "finding_assignments.csv"
    shutil_copy = pd.read_csv(generated_paths["medium_1000"])
    shutil_copy.to_csv(default_path, index=False)
    generated_paths["default"] = default_path

    return generated_paths

