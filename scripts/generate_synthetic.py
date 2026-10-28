"""
Script to generate synthetic enterprise assets, finding assignments,
and baseline-scored remediation candidates across all benchmark scales (100, 1000, 5000).
"""

from pathlib import Path
import sys
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from vulnopt.synthetic_data import build_benchmark_datasets
from vulnopt.risk_model import build_remediation_candidates, DEFAULT_WEIGHTS


def main():
    print("=" * 60)
    print("Generating VulnOpt Synthetic Assets & Benchmark Findings")
    print("=" * 60)

    data_dir = ROOT_DIR / "data"
    synthetic_dir = data_dir / "synthetic"
    processed_vuln_path = data_dir / "processed" / "vulnerabilities.csv"

    if not processed_vuln_path.exists():
        raise FileNotFoundError(f"Processed vulnerabilities file not found: {processed_vuln_path}")

    # 1. Generate assets and raw finding assignments
    paths = build_benchmark_datasets(
        vulnerabilities_path=processed_vuln_path,
        output_dir=synthetic_dir,
        seed=42,
    )

    vuln_df = pd.read_csv(processed_vuln_path)
    assets_df = pd.read_csv(paths["assets"])

    # 2. Score candidates for each benchmark scale with default weights
    scales = ["100", "1000", "5000"]
    for scale in scales:
        fa_path = synthetic_dir / f"finding_assignments_{scale}.csv"
        findings_df = pd.read_csv(fa_path)

        scored_df = build_remediation_candidates(
            findings_df=findings_df,
            vulnerabilities_df=vuln_df,
            assets_df=assets_df,
            weights=DEFAULT_WEIGHTS,
        )

        out_scored_path = synthetic_dir / f"scored_findings_{scale}.csv"
        scored_df.to_csv(out_scored_path, index=False)
        print(f"Scored {len(scored_df)} findings -> {out_scored_path}")

        # Summary statistics
        mean_risk = scored_df["risk_score"].mean()
        total_cost = scored_df["patch_cost"].sum()
        total_hours = scored_df["engineer_hours"].sum()
        kev_count = scored_df["is_kev"].sum()
        print(f"   Scale {scale}: Mean Risk = {mean_risk:.1f}/100, Total Cost = ${total_cost:,.2f}, Total Hours = {total_hours:,.1f}h, KEV Findings = {int(kev_count)}")

    # Default scored_findings.csv pointing to medium (1000)
    default_scored = pd.read_csv(synthetic_dir / "scored_findings_1000.csv")
    default_scored.to_csv(synthetic_dir / "scored_findings.csv", index=False)
    print(f"\nDefault scored candidates (1,000) saved to {synthetic_dir / 'scored_findings.csv'}")
    print("=" * 60)


if __name__ == "__main__":
    main()

