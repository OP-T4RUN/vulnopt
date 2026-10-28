"""
Data cleaning and normalization module for VulnOpt.

Standardizes public vulnerability records from NVD, EPSS, and CISA KEV
into a clean, documented dataset ready for synthetic asset assignment and risk scoring.
"""

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd


CVE_PATTERN = re.compile(r"^CVE-\d{4}-\d{4,}$", re.IGNORECASE)


def normalize_cve_id(cve_id: Any) -> Optional[str]:
    """Validate and normalize a CVE ID to standard uppercase format (e.g., CVE-2023-1234)."""
    if not isinstance(cve_id, str):
        return None
    cleaned = cve_id.strip().upper()
    return cleaned if CVE_PATTERN.match(cleaned) else None


def parse_cisa_kev_file(filepath: Path) -> pd.DataFrame:
    """
    Parse a CISA KEV JSON catalog snapshot into a normalized pandas DataFrame.
    """
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    vulnerabilities = data.get("vulnerabilities", [])
    records: List[Dict[str, Any]] = []

    for item in vulnerabilities:
        raw_cve = item.get("cveID")
        norm_cve = normalize_cve_id(raw_cve)
        if not norm_cve:
            continue

        ransomware = item.get("knownRansomwareCampaignUse", "Unknown")
        records.append({
            "cve_id": norm_cve,
            "is_kev": 1,
            "kev_date_added": item.get("dateAdded"),
            "kev_vendor_project": item.get("vendorProject"),
            "kev_product": item.get("product"),
            "kev_vulnerability_name": item.get("vulnerabilityName"),
            "kev_ransomware": ransomware,
        })

    df = pd.DataFrame(records)
    if df.empty:
        df = pd.DataFrame(columns=[
            "cve_id", "is_kev", "kev_date_added", "kev_vendor_project",
            "kev_product", "kev_vulnerability_name", "kev_ransomware"
        ])
    else:
        df = df.drop_duplicates(subset=["cve_id"]).reset_index(drop=True)
    return df


def parse_epss_file(filepath: Path) -> Tuple[pd.DataFrame, Optional[str]]:
    """
    Parse a FIRST EPSS CSV file into a normalized pandas DataFrame.
    Extracts EPSS date/version metadata if present in file headers.
    """
    score_date: Optional[str] = None

    # Check first few lines for header comment
    with open(filepath, "r", encoding="utf-8", errors="replace") as f:
        for _ in range(5):
            line = f.readline()
            if line.startswith("#"):
                date_match = re.search(r"score_date:([0-9T:\+\-]+)", line)
                if date_match:
                    score_date = date_match.group(1)
            else:
                break

    df = pd.read_csv(filepath, comment="#", dtype={"cve": str, "epss": float, "percentile": float})
    df.rename(columns={"cve": "cve_id"}, inplace=True)
    df["cve_id"] = df["cve_id"].apply(normalize_cve_id)
    df = df.dropna(subset=["cve_id"]).drop_duplicates(subset=["cve_id"])

    # Ensure epss and percentile are within [0, 1]
    df["epss"] = pd.to_numeric(df["epss"], errors="coerce")
    df["epss_percentile"] = pd.to_numeric(df["percentile"], errors="coerce")
    df["epss_date"] = score_date or datetime.now(timezone.utc).strftime("%Y-%m-%d")

    return df[["cve_id", "epss", "epss_percentile", "epss_date"]].reset_index(drop=True), score_date


def extract_best_cvss(metrics: Dict[str, Any]) -> Tuple[Optional[float], Optional[str], Optional[str], Optional[str]]:
    """
    Extract CVSS base score, version, vector, and severity prioritizing:
    v3.1 -> v3.0 -> v4.0 -> v2.0.
    """
    priority_keys = ["cvssMetricV31", "cvssMetricV30", "cvssMetricV40", "cvssMetricV2"]

    for key in priority_keys:
        entries = metrics.get(key, [])
        if not entries:
            continue
        # Primary entry preferred if multiple
        primary_entry = next((e for e in entries if e.get("type", "").lower() == "primary"), entries[0])
        cvss_data = primary_entry.get("cvssData", {})
        base_score = cvss_data.get("baseScore")
        version = cvss_data.get("version")
        vector = cvss_data.get("vectorString")
        severity = primary_entry.get("baseSeverity") or cvss_data.get("baseSeverity")

        if base_score is not None:
            try:
                base_score_flt = float(base_score)
                return base_score_flt, str(version), str(vector), str(severity).upper() if severity else None
            except (ValueError, TypeError):
                continue

    return None, None, None, None


def parse_nvd_json_pages(nvd_files: List[Path]) -> pd.DataFrame:
    """
    Parse a list of raw NVD JSON files into a structured DataFrame.
    """
    records: List[Dict[str, Any]] = []

    for file_path in nvd_files:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        vulnerabilities = data.get("vulnerabilities", [])
        for item in vulnerabilities:
            cve_obj = item.get("cve", {})
            norm_cve = normalize_cve_id(cve_obj.get("id"))
            if not norm_cve:
                continue

            published = cve_obj.get("published")
            last_modified = cve_obj.get("lastModified")

            # Extract English description
            descriptions = cve_obj.get("descriptions", [])
            en_desc = next((d.get("value") for d in descriptions if d.get("lang") == "en"), "")

            # Extract CVSS
            metrics = cve_obj.get("metrics", {})
            base_score, version, vector, severity = extract_best_cvss(metrics)

            # Extract CWE IDs
            weaknesses = cve_obj.get("weaknesses", [])
            cwes: List[str] = []
            for w in weaknesses:
                for desc in w.get("description", []):
                    val = desc.get("value", "")
                    if val.upper().startswith("CWE-") and val not in cwes:
                        cwes.append(val)
            cwe_str = ", ".join(cwes) if cwes else None

            records.append({
                "cve_id": norm_cve,
                "published_date": published,
                "last_modified": last_modified,
                "description": en_desc,
                "cvss_base": base_score,
                "cvss_version": version,
                "cvss_vector": vector,
                "cvss_severity": severity,
                "cwe_ids": cwe_str,
            })

    df = pd.DataFrame(records)
    if not df.empty:
        df = df.drop_duplicates(subset=["cve_id"]).reset_index(drop=True)
    return df


def merge_and_normalize_datasets(
    nvd_df: pd.DataFrame,
    epss_df: Optional[pd.DataFrame] = None,
    kev_df: Optional[pd.DataFrame] = None,
    snapshot_date: Optional[str] = None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Merge NVD, EPSS, and KEV records, enforce missing value policies,
    and generate a data quality summary report.
    """
    if snapshot_date is None:
        snapshot_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    raw_total_count = len(nvd_df)

    # 1. Start with NVD
    df = nvd_df.copy()

    # 2. Join KEV
    if kev_df is not None and not kev_df.empty:
        kev_subset = kev_df[["cve_id", "is_kev", "kev_date_added", "kev_ransomware"]].drop_duplicates("cve_id")
        df = df.merge(kev_subset, on="cve_id", how="left")
        df["is_kev"] = df["is_kev"].fillna(0).astype(int)
    else:
        df["is_kev"] = 0
        df["kev_date_added"] = None
        df["kev_ransomware"] = None

    # 3. Join EPSS
    if epss_df is not None and not epss_df.empty:
        epss_subset = epss_df[["cve_id", "epss", "epss_percentile", "epss_date"]].drop_duplicates("cve_id")
        df = df.merge(epss_subset, on="cve_id", how="left")
        df["epss_missing"] = df["epss"].isna()
        # Default convention: fillna with 0.0 for scoring, keeping epss_missing indicator
        df["epss_score"] = df["epss"].fillna(0.0)
    else:
        df["epss"] = None
        df["epss_percentile"] = None
        df["epss_date"] = None
        df["epss_missing"] = True
        df["epss_score"] = 0.0

    # 4. Handle Missing CVSS (Exclude records lacking valid CVSS as per Section 6.3)
    cvss_missing_count = int(df["cvss_base"].isna().sum())
    valid_cvss_mask = df["cvss_base"].notna() & df["cvss_base"].between(0.0, 10.0)
    invalid_cvss_count = int((~valid_cvss_mask).sum()) - cvss_missing_count

    df_clean = df[valid_cvss_mask].copy()
    df_clean["data_snapshot_date"] = snapshot_date

    # Data Quality Summary Report
    summary = {
        "snapshot_date": snapshot_date,
        "nvd_raw_records": raw_total_count,
        "records_excluded_missing_cvss": cvss_missing_count,
        "records_excluded_invalid_cvss": invalid_cvss_count,
        "final_valid_records": len(df_clean),
        "kev_records_matched": int(df_clean["is_kev"].sum()),
        "epss_available_count": int((~df_clean["epss_missing"]).sum()),
        "epss_missing_count": int(df_clean["epss_missing"].sum()),
        "cvss_version_distribution": df_clean["cvss_version"].value_counts().to_dict(),
        "cvss_mean": round(float(df_clean["cvss_base"].mean()), 2) if len(df_clean) else 0.0,
        "epss_mean": round(float(df_clean["epss"].dropna().mean()), 4) if len(df_clean["epss"].dropna()) else 0.0,
    }

    # Format final column order
    final_cols = [
        "cve_id",
        "published_date",
        "last_modified",
        "description",
        "cvss_base",
        "cvss_version",
        "cvss_vector",
        "cvss_severity",
        "epss",
        "epss_score",
        "epss_percentile",
        "epss_missing",
        "epss_date",
        "is_kev",
        "kev_date_added",
        "kev_ransomware",
        "cwe_ids",
        "data_snapshot_date",
    ]
    df_clean = df_clean.reindex(columns=final_cols).reset_index(drop=True)
    return df_clean, summary

