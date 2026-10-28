"""
Data collection module for VulnOpt (Student 1).

Retrieves public vulnerability data from:
1. CISA Known Exploited Vulnerabilities (KEV) Catalog JSON feed
2. FIRST Exploit Prediction Scoring System (EPSS) bulk CSV.GZ / API
3. NIST National Vulnerability Database (NVD) API 2.0
"""

import gzip
import json
import os
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
import requests


def load_env_file(env_path: Optional[Path] = None) -> Dict[str, str]:
    """Parse a simple .env file if present and set environment variables."""
    if env_path is None:
        env_path = Path(__file__).resolve().parent.parent / ".env"
    env_vars: Dict[str, str] = {}
    if not env_path.exists():
        return env_vars

    with open(env_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            k = k.strip()
            v = v.strip().strip("'\"")
            env_vars[k] = v
            if k not in os.environ:
                os.environ[k] = v
    return env_vars


# Load environment variables on module import
load_env_file()

CISA_KEV_URL = "https://www.cisa.gov/sites/default/files/feeds/known_exploited_vulnerabilities.json"
EPSS_BULK_URL = "https://epss.empiricalsecurity.com/epss_scores-current.csv.gz"
EPSS_API_URL = "https://api.first.org/data/v1/epss"
NVD_API_URL = "https://services.nvd.nist.gov/rest/json/cves/2.0"


def fetch_cisa_kev(output_dir: Path = Path("data/raw")) -> Path:
    """
    Download the CISA KEV JSON feed and save as a timestamped snapshot.

    Returns the Path to the saved raw file.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_file = output_dir / f"cisa_kev_{timestamp}.json"
    latest_symlink_or_copy = output_dir / "cisa_kev_latest.json"

    print(f"Downloading CISA KEV catalog from {CISA_KEV_URL}...")
    response = requests.get(CISA_KEV_URL, timeout=30)
    response.raise_for_status()

    # Validate JSON payload structure
    data = response.json()
    if "vulnerabilities" not in data or not isinstance(data["vulnerabilities"], list):
        raise ValueError("Invalid CISA KEV response format: missing 'vulnerabilities' array.")

    # Attach retrieval metadata
    data["_retrieval_metadata"] = {
        "source_url": CISA_KEV_URL,
        "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
        "total_records": len(data["vulnerabilities"]),
    }

    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    with open(latest_symlink_or_copy, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    print(f"Saved CISA KEV snapshot with {len(data['vulnerabilities'])} vulnerabilities to {out_file}")
    return latest_symlink_or_copy


def fetch_epss_bulk(output_dir: Path = Path("data/raw")) -> Path:
    """
    Download the daily FIRST EPSS bulk compressed CSV file.

    Returns the Path to the decompressed or saved CSV file.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    gz_file = output_dir / f"epss_{timestamp}.csv.gz"
    out_csv = output_dir / "epss_scores_latest.csv"

    print(f"Downloading bulk EPSS dataset from {EPSS_BULK_URL}...")
    headers = {"User-Agent": "VulnOpt-Research-Project/0.1"}
    response = requests.get(EPSS_BULK_URL, headers=headers, stream=True, timeout=60)
    response.raise_for_status()

    with open(gz_file, "wb") as f:
        for chunk in response.iter_content(chunk_size=65536):
            if chunk:
                f.write(chunk)

    print(f"Decompressing {gz_file} to {out_csv}...")
    with gzip.open(gz_file, "rt", encoding="utf-8", errors="replace") as g_in:
        with open(out_csv, "w", encoding="utf-8") as f_out:
            shutil.copyfileobj(g_in, f_out)

    print(f"EPSS bulk scores saved to {out_csv}")
    return out_csv


def fetch_nvd_batch(
    api_key: Optional[str] = None,
    pub_start_date: Optional[str] = None,
    pub_end_date: Optional[str] = None,
    cve_id: Optional[str] = None,
    results_per_page: int = 2000,
    start_index: int = 0,
    max_retries: int = 5,
    backoff_factor: float = 2.0,
) -> Dict[str, Any]:
    """
    Fetch a single page of CVEs from NVD API 2.0 with retry and rate-limiting handling.
    """
    if api_key is None:
        api_key = os.environ.get("NVD_API_KEY")

    headers = {"User-Agent": "VulnOpt-Research-Project/0.1"}
    if api_key:
        headers["apiKey"] = api_key

    params: Dict[str, Any] = {
        "resultsPerPage": min(results_per_page, 2000),
        "startIndex": start_index,
    }
    if cve_id:
        params["cveId"] = cve_id
    if pub_start_date:
        params["pubStartDate"] = pub_start_date
    if pub_end_date:
        params["pubEndDate"] = pub_end_date

    delay = 1.0 if api_key else 6.0
    for attempt in range(max_retries):
        try:
            time.sleep(delay)
            response = requests.get(NVD_API_URL, headers=headers, params=params, timeout=45)
            if response.status_code == 200:
                return response.json()
            elif response.status_code in (403, 429, 503, 504):
                sleep_time = (backoff_factor ** attempt) * (2.0 if api_key else 8.0)
                print(f"NVD API returned {response.status_code}. Retrying in {sleep_time:.1f}s (attempt {attempt+1}/{max_retries})...")
                time.sleep(sleep_time)
            else:
                response.raise_for_status()
        except requests.RequestException as e:
            if attempt == max_retries - 1:
                raise e
            sleep_time = (backoff_factor ** attempt) * 3.0
            print(f"Request error: {e}. Retrying in {sleep_time:.1f}s...")
            time.sleep(sleep_time)

    raise RuntimeError(f"Failed to fetch NVD data after {max_retries} attempts.")


def fetch_nvd_dataset(
    pub_start_date: str,
    pub_end_date: str,
    max_records: int = 5000,
    output_dir: Path = Path("data/raw/nvd"),
    api_key: Optional[str] = None,
) -> List[Path]:
    """
    Download a window of NVD records via API 2.0 with pagination and save raw JSON pages.
    NVD requires (pubEndDate - pubStartDate) <= 120 days.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    if api_key is None:
        api_key = os.environ.get("NVD_API_KEY")

    start_index = 0
    results_per_page = 2000
    saved_files: List[Path] = []
    total_downloaded = 0

    print(f"Fetching NVD CVEs from {pub_start_date} to {pub_end_date} (target up to {max_records} records)...")

    while total_downloaded < max_records:
        page_size = min(results_per_page, max_records - total_downloaded)
        page_data = fetch_nvd_batch(
            api_key=api_key,
            pub_start_date=pub_start_date,
            pub_end_date=pub_end_date,
            results_per_page=page_size,
            start_index=start_index,
        )

        total_results = page_data.get("totalResults", 0)
        vulnerabilities = page_data.get("vulnerabilities", [])
        if not vulnerabilities:
            print("No more vulnerabilities returned by NVD.")
            break

        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        out_page = output_dir / f"nvd_page_{start_index}_{timestamp}.json"
        with open(out_page, "w", encoding="utf-8") as f:
            json.dump(page_data, f, indent=2)

        saved_files.append(out_page)
        count = len(vulnerabilities)
        total_downloaded += count
        start_index += count

        print(f"Downloaded {count} CVEs (total {total_downloaded}/{total_results}) -> {out_page.name}")

        if start_index >= total_results:
            break

    return saved_files

