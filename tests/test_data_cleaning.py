"""
Offline unit tests for data cleaning and normalization (Student 1).
"""

import json
import tempfile
from pathlib import Path
import pandas as pd
import pytest

from vulnopt.data_cleaning import (
    extract_best_cvss,
    merge_and_normalize_datasets,
    normalize_cve_id,
    parse_cisa_kev_file,
    parse_epss_file,
    parse_nvd_json_pages,
)


def test_normalize_cve_id():
    assert normalize_cve_id("cve-2023-1234") == "CVE-2023-1234"
    assert normalize_cve_id("  CVE-2021-44228 ") == "CVE-2021-44228"
    assert normalize_cve_id("CVE-2024-9999999") == "CVE-2024-9999999"
    assert normalize_cve_id("INVALID-2023-1") is None
    assert normalize_cve_id("CVE-23-1234") is None
    assert normalize_cve_id(None) is None
    assert normalize_cve_id(1234) is None


def test_extract_best_cvss_priority():
    # Priority: v3.1 over v3.0 over v2.0
    metrics_all = {
        "cvssMetricV2": [{
            "type": "Primary",
            "cvssData": {"version": "2.0", "baseScore": 5.0, "vectorString": "AV:N/AC:L/Au:N/C:N/I:N/A:P"}
        }],
        "cvssMetricV30": [{
            "type": "Primary",
            "cvssData": {"version": "3.0", "baseScore": 7.5, "vectorString": "CVSS:3.0/AV:N/AC:L/PR:N/UI:N/S:U/C:N/I:N/A:H"}
        }],
        "cvssMetricV31": [{
            "type": "Primary",
            "cvssData": {"version": "3.1", "baseScore": 9.8, "vectorString": "CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:U/C:H/I:H/A:H"},
            "baseSeverity": "CRITICAL"
        }]
    }

    score, ver, vector, sev = extract_best_cvss(metrics_all)
    assert score == 9.8
    assert ver == "3.1"
    assert sev == "CRITICAL"

    # Fallback to v3.0 when v3.1 is missing
    metrics_v30 = {"cvssMetricV30": metrics_all["cvssMetricV30"]}
    score, ver, vector, sev = extract_best_cvss(metrics_v30)
    assert score == 7.5
    assert ver == "3.0"

    # Fallback to v2.0
    metrics_v2 = {"cvssMetricV2": metrics_all["cvssMetricV2"]}
    score, ver, vector, sev = extract_best_cvss(metrics_v2)
    assert score == 5.0
    assert ver == "2.0"


def test_parse_cisa_kev_file(tmp_path: Path):
    sample_kev = {
        "title": "CISA Catalog",
        "vulnerabilities": [
            {
                "cveID": "cve-2021-44228",
                "vendorProject": "Apache",
                "product": "Log4j",
                "vulnerabilityName": "Log4Shell",
                "dateAdded": "2021-12-10",
                "knownRansomwareCampaignUse": "Known"
            },
            {
                "cveID": "CVE-2023-9999",
                "vendorProject": "TestVendor",
                "product": "TestProd",
                "vulnerabilityName": "TestName",
                "dateAdded": "2023-01-01",
                "knownRansomwareCampaignUse": "Unknown"
            }
        ]
    }
    kev_path = tmp_path / "mock_kev.json"
    with open(kev_path, "w", encoding="utf-8") as f:
        json.dump(sample_kev, f)

    df_kev = parse_cisa_kev_file(kev_path)
    assert len(df_kev) == 2
    assert "CVE-2021-44228" in df_kev["cve_id"].values
    assert df_kev.loc[df_kev["cve_id"] == "CVE-2021-44228", "is_kev"].iloc[0] == 1
    assert df_kev.loc[df_kev["cve_id"] == "CVE-2021-44228", "kev_ransomware"].iloc[0] == "Known"


def test_parse_epss_file(tmp_path: Path):
    epss_content = (
        "#model_version:v2023.03.01,score_date:2024-04-01T00:00:00+0000\n"
        "cve,epss,percentile\n"
        "CVE-2021-44228,0.97541,0.9998\n"
        "CVE-2023-1234,0.00125,0.4500\n"
    )
    epss_path = tmp_path / "mock_epss.csv"
    with open(epss_path, "w", encoding="utf-8") as f:
        f.write(epss_content)

    df_epss, score_date = parse_epss_file(epss_path)
    assert len(df_epss) == 2
    assert score_date == "2024-04-01T00:00:00+0000"
    log4j_row = df_epss[df_epss["cve_id"] == "CVE-2021-44228"].iloc[0]
    assert pytest.approx(log4j_row["epss"], 0.0001) == 0.97541
    assert pytest.approx(log4j_row["epss_percentile"], 0.0001) == 0.9998


def test_merge_and_normalize_datasets():
    nvd_records = pd.DataFrame([
        {
            "cve_id": "CVE-2021-44228",
            "published_date": "2021-12-10T10:00:00.000",
            "last_modified": "2021-12-15T10:00:00.000",
            "description": "Log4Shell RCE",
            "cvss_base": 10.0,
            "cvss_version": "3.1",
            "cvss_vector": "CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:C/C:H/I:H/A:H",
            "cvss_severity": "CRITICAL",
            "cwe_ids": "CWE-502",
        },
        {
            "cve_id": "CVE-2023-1111",
            "published_date": "2023-01-01T10:00:00.000",
            "last_modified": "2023-01-05T10:00:00.000",
            "description": "Missing CVSS vulnerability",
            "cvss_base": None,  # Should be excluded
            "cvss_version": None,
            "cvss_vector": None,
            "cvss_severity": None,
            "cwe_ids": None,
        },
        {
            "cve_id": "CVE-2023-2222",
            "published_date": "2023-02-01T10:00:00.000",
            "last_modified": "2023-02-05T10:00:00.000",
            "description": "Standard bug",
            "cvss_base": 5.4,
            "cvss_version": "3.1",
            "cvss_vector": "CVSS:3.1/AV:N/AC:L/PR:L/UI:R/S:C/C:L/I:L/A:N",
            "cvss_severity": "MEDIUM",
            "cwe_ids": "CWE-79",
        }
    ])

    epss_df = pd.DataFrame([
        {"cve_id": "CVE-2021-44228", "epss": 0.97, "epss_percentile": 0.99, "epss_date": "2024-04-01"}
        # CVE-2023-2222 intentionally omitted to test epss_missing
    ])

    kev_df = pd.DataFrame([
        {"cve_id": "CVE-2021-44228", "is_kev": 1, "kev_date_added": "2021-12-10", "kev_ransomware": "Known"}
    ])

    clean_df, summary = merge_and_normalize_datasets(nvd_records, epss_df, kev_df, snapshot_date="2024-04-01")

    # Excluded missing CVSS record
    assert len(clean_df) == 2
    assert "CVE-2023-1111" not in clean_df["cve_id"].values
    assert summary["records_excluded_missing_cvss"] == 1

    # Check Log4Shell merged attributes
    log4j = clean_df[clean_df["cve_id"] == "CVE-2021-44228"].iloc[0]
    assert log4j["is_kev"] == 1
    assert log4j["epss_missing"] is False or log4j["epss_missing"] == 0
    assert log4j["epss_score"] == 0.97

    # Check CVE-2023-2222 missing EPSS handling
    cve2 = clean_df[clean_df["cve_id"] == "CVE-2023-2222"].iloc[0]
    assert cve2["is_kev"] == 0
    assert bool(cve2["epss_missing"]) is True
    assert cve2["epss_score"] == 0.0

