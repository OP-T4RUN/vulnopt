"""
Master script to execute the Student 1 data pipeline:
1. Downloads CISA KEV JSON snapshot
2. Downloads FIRST EPSS bulk compressed CSV
3. Fetches real NVD CVE/CVSS records via NVD API 2.0 using the provided API key
4. Merges, normalizes, and filters the datasets
5. Saves clean dataset to data/processed/vulnerabilities.csv
6. Generates data/processed/data_quality_summary.json
"""

import argparse
import json
import os
import sys
from pathlib import Path

# Add project root to path
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from vulnopt.data_collection import (
    fetch_cisa_kev,
    fetch_epss_bulk,
    fetch_nvd_dataset,
    load_env_file,
)
from vulnopt.data_cleaning import (
    merge_and_normalize_datasets,
    parse_cisa_kev_file,
    parse_epss_file,
    parse_nvd_json_pages,
)


def run_pipeline(
    max_nvd_records: int = 5000,
    start_date: str = "2024-01-01T00:00:00.000",
    end_date: str = "2024-04-01T00:00:00.000",
    skip_downloads: bool = False,
):
    print("=" * 60)
    print("VulnOpt Data Collection & Normalization Pipeline (Student 1)")
    print("=" * 60)

    load_env_file()
    api_key = os.environ.get("NVD_API_KEY")
    if not api_key:
        print("WARNING: No NVD_API_KEY found in environment or .env file. NVD requests will be heavily rate-limited.")
    else:
        print(f"Loaded NVD API Key: {api_key[:8]}...{api_key[-4:]}")

    raw_dir = ROOT_DIR / "data" / "raw"
    nvd_dir = raw_dir / "nvd"
    processed_dir = ROOT_DIR / "data" / "processed"
    raw_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)

    kev_file = raw_dir / "cisa_kev_latest.json"
    epss_file = raw_dir / "epss_scores_latest.csv"

    # 1. CISA KEV
    if not skip_downloads or not kev_file.exists():
        print("\n[Step 1/4] Fetching CISA KEV Catalog...")
        kev_file = fetch_cisa_kev(output_dir=raw_dir)
    else:
        print(f"\n[Step 1/4] Using existing CISA KEV snapshot: {kev_file}")

    # 2. FIRST EPSS
    if not skip_downloads or not epss_file.exists():
        print("\n[Step 2/4] Fetching FIRST EPSS bulk scores...")
        epss_file = fetch_epss_bulk(output_dir=raw_dir)
    else:
        print(f"\n[Step 2/4] Using existing EPSS scores: {epss_file}")

    # 3. NVD API 2.0
    existing_nvd_pages = list(nvd_dir.glob("nvd_page_*.json"))
    if not skip_downloads or not existing_nvd_pages:
        print(f"\n[Step 3/4] Fetching up to {max_nvd_records} NVD CVE records from {start_date} to {end_date}...")
        nvd_pages = fetch_nvd_dataset(
            pub_start_date=start_date,
            pub_end_date=end_date,
            max_records=max_nvd_records,
            output_dir=nvd_dir,
            api_key=api_key,
        )
    else:
        print(f"\n[Step 3/4] Using {len(existing_nvd_pages)} existing NVD JSON page(s) in {nvd_dir}")
        nvd_pages = existing_nvd_pages

    # 4. Clean and Merge
    print("\n[Step 4/4] Parsing, normalizing, and merging datasets...")
    df_kev = parse_cisa_kev_file(kev_file)
    print(f" -> Parsed {len(df_kev)} KEV records.")

    df_epss, epss_date = parse_epss_file(epss_file)
    print(f" -> Parsed {len(df_epss)} EPSS records (Score date: {epss_date}).")

    df_nvd = parse_nvd_json_pages(nvd_pages)
    print(f" -> Parsed {len(df_nvd)} NVD CVE records.")

    clean_df, summary = merge_and_normalize_datasets(df_nvd, df_epss, df_kev)

    out_csv = processed_dir / "vulnerabilities.csv"
    clean_df.to_csv(out_csv, index=False)
    print(f"\nSuccessfully wrote {len(clean_df)} normalized vulnerability findings to {out_csv}")

    summary_file = processed_dir / "data_quality_summary.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Data quality summary written to {summary_file}")

    print("\n--- Summary Highlights ---")
    print(f"Total raw NVD records:            {summary['nvd_raw_records']}")
    print(f"Excluded (missing CVSS):          {summary['records_excluded_missing_cvss']}")
    print(f"Final valid records:              {summary['final_valid_records']}")
    print(f"Matched KEV records:              {summary['kev_records_matched']}")
    print(f"EPSS scores available:            {summary['epss_available_count']}")
    print(f"EPSS missing (flagged/imputed):   {summary['epss_missing_count']}")
    print(f"Average CVSS Base:                {summary['cvss_mean']}")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VulnOpt Data Retrieval Pipeline")
    parser.add_argument("--max-records", type=int, default=5000, help="Maximum NVD records to retrieve")
    parser.add_argument("--start-date", type=str, default="2024-01-01T00:00:00.000", help="NVD pubStartDate")
    parser.add_argument("--end-date", type=str, default="2024-04-01T00:00:00.000", help="NVD pubEndDate")
    parser.add_argument("--skip-downloads", action="store_true", help="Skip download if raw files exist")
    args = parser.parse_args()

    run_pipeline(
        max_nvd_records=args.max_records,
        start_date=args.start_date,
        end_date=args.end_date,
        skip_downloads=args.skip_downloads,
    )

