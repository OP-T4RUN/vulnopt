"""
Tests for the VulnOpt experiment runner and output contract.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from vulnopt.experiments import (
    ExperimentConfig,
    apply_epss_strategy,
    build_problem,
    quick_config,
    run_all_experiments,
    run_solver,
)
from vulnopt.metrics import RESULT_COLUMNS
from vulnopt.risk_model import DEFAULT_WEIGHTS, calculate_risk_scores


def _tiny_scored_frame(n: int = 8, seed: int = 0, n_missing_epss: int = 2) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    exposures = ["internet", "internal", "restricted"]
    for i in range(n):
        missing = i < n_missing_epss
        epss = float("nan") if missing else float(rng.uniform(0.0, 0.4))
        rows.append(
            {
                "finding_id": f"FND-{i+1:05d}",
                "cve_id": f"CVE-2024-{1000+i}",
                "cvss_base": float(rng.uniform(2.0, 9.5)),
                "epss": epss,
                "epss_score": 0.0 if missing else epss,
                "epss_missing": missing,
                "is_kev": int(i % 4 == 0),
                "criticality": int(rng.integers(1, 6)),
                "exposure": exposures[i % 3],
                "patch_cost": float(rng.uniform(50.0, 400.0)),
                "engineer_hours": float(rng.uniform(1.0, 8.0)),
                "downtime_hours": float(rng.uniform(0.0, 2.0)),
            }
        )
    return calculate_risk_scores(pd.DataFrame(rows), weights=DEFAULT_WEIGHTS)


@pytest.fixture
def tiny_csv(tmp_path: Path) -> Path:
    path = tmp_path / "tiny_findings.csv"
    _tiny_scored_frame().to_csv(path, index=False)
    return path


def _config_for(tiny_csv: Path, tmp_path: Path) -> ExperimentConfig:
    cfg = quick_config()
    cfg.dataset_sizes = [8]
    cfg.epss_dataset_sizes = [8]
    cfg.sensitivity_dataset_size = 8
    cfg.dataset_files = {8: str(tiny_csv)}
    cfg.bpso_params[8] = {"n_particles": 6, "max_iter": 8}
    cfg.milp_time_limits[8] = 15
    cfg.bpso_seeds = [0, 1]
    cfg.dirichlet_samples = 2
    cfg.data_root = str(tmp_path)
    return cfg


class TestProblemConstruction:
    def test_resource_fractions_match_student3_definition(self):
        df = _tiny_scored_frame()
        problem = build_problem(df, 0.30)
        assert problem.budget == pytest.approx(float(df["patch_cost"].sum()) * 0.30)
        assert problem.max_hours == pytest.approx(float(df["engineer_hours"].sum()) * 0.30)
        assert problem.max_downtime == pytest.approx(float(df["downtime_hours"].sum()) * 0.30)

    def test_all_methods_receive_same_problem(self):
        df = _tiny_scored_frame()
        problem = build_problem(df, 0.50)
        params = {"n_particles": 6, "max_iter": 8}
        r_cvss = run_solver("CVSS-Greedy", problem, df, seed=0, bpso_params=params, milp_time_limit=10)
        r_risk = run_solver("Risk-Greedy", problem, df, seed=0, bpso_params=params, milp_time_limit=10)
        r_milp = run_solver("MILP", problem, df, seed=0, bpso_params=params, milp_time_limit=10)
        r_bpso = run_solver("BPSO", problem, df, seed=0, bpso_params=params, milp_time_limit=10)
        for res in (r_cvss, r_risk, r_milp, r_bpso):
            assert res.feasible
            assert res.budget_used <= problem.budget + 1e-6


class TestEpssPolicies:
    def test_zero_vs_median_vs_exclusion_differ_when_missing(self):
        df = _tiny_scored_frame(n_missing_epss=3)
        zero = apply_epss_strategy(df, "zero_imputation")
        median = apply_epss_strategy(df, "median_imputation")
        excl = apply_epss_strategy(df, "exclusion")
        assert len(excl) == len(df) - 3
        # Imputed rows should not all match unless the observed median happens to be 0.
        missing = df["epss_missing"].astype(bool)
        assert (zero.loc[missing, "epss"] == 0.0).all()
        observed_median = float(df.loc[~missing, "epss"].median())
        assert median.loc[missing, "epss"].eq(observed_median).all()

    def test_unknown_strategy_raises(self):
        with pytest.raises(ValueError, match="Unknown EPSS strategy"):
            apply_epss_strategy(_tiny_scored_frame(), "mean_imputation")


class TestExperimentOutputs:
    def test_writes_contracted_files_and_schema(self, tiny_csv: Path, tmp_path: Path):
        out = tmp_path / "results"
        cfg = _config_for(tiny_csv, tmp_path)
        payload = run_all_experiments(out, cfg, write_figures=True)

        expected = [
            "method_comparison.csv",
            "bpso_seeds.csv",
            "bpso_summary.csv",
            "sensitivity_named_weights.csv",
            "sensitivity_dirichlet.csv",
            "epss_handling.csv",
            "resource_utilization.csv",
            "feasibility.csv",
            "runtime.csv",
            "experiment_failures.csv",
            "schema_example.csv",
            "experiment_config.json",
        ]
        for name in expected:
            assert (out / "tables" / name).exists(), name

        cmp_df = payload["tables"]["method_comparison.csv"]
        assert list(cmp_df.columns) == list(RESULT_COLUMNS)
        assert set(cmp_df["method"].dropna().unique()).issubset(
            {"CVSS-Greedy", "Risk-Greedy", "MILP", "BPSO"}
        )
        assert cmp_df["ran_successfully"].all()
        assert (out / "tables" / "schema_example.csv").stat().st_size > 0

        seeds = payload["tables"]["bpso_seeds.csv"]
        assert sorted(seeds["seed"].dropna().unique().tolist()) == [0, 1]
        summary = payload["tables"]["bpso_summary.csv"]
        assert int(summary["n_successful"].iloc[0]) == 2

        diri = payload["tables"]["sensitivity_dirichlet.csv"]
        assert set(diri["weight_sample_id"].dropna().astype(int).unique()) == {0, 1}

        epss = payload["tables"]["epss_handling.csv"]
        assert set(epss["epss_strategy"].unique()) == {
            "zero_imputation",
            "median_imputation",
            "exclusion",
        }

        failures = payload["tables"]["experiment_failures.csv"]
        assert len(failures) == 0

        assert (out / "figures").exists()
        assert len(list((out / "figures").glob("*.png"))) >= 1

    def test_schema_consistent_across_families(self, tiny_csv: Path, tmp_path: Path):
        cfg = _config_for(tiny_csv, tmp_path)
        payload = run_all_experiments(tmp_path / "results2", cfg, write_figures=False)
        for name in (
            "method_comparison.csv",
            "bpso_seeds.csv",
            "sensitivity_named_weights.csv",
            "sensitivity_dirichlet.csv",
            "epss_handling.csv",
        ):
            df = payload["tables"][name]
            assert list(df.columns) == list(RESULT_COLUMNS)

    def test_bpso_seeds_are_reproducible(self, tiny_csv: Path, tmp_path: Path):
        cfg = _config_for(tiny_csv, tmp_path)
        cfg.run_method_comparison = False
        cfg.run_named_weights = False
        cfg.run_dirichlet = False
        cfg.run_epss = False
        cfg.run_bpso_multiseed = True
        a = run_all_experiments(tmp_path / "rep_a", cfg, write_figures=False)
        b = run_all_experiments(tmp_path / "rep_b", cfg, write_figures=False)
        da = a["tables"]["bpso_seeds.csv"].sort_values("seed").reset_index(drop=True)
        db = b["tables"]["bpso_seeds.csv"].sort_values("seed").reset_index(drop=True)
        pd.testing.assert_series_equal(
            da["total_risk_reduction"], db["total_risk_reduction"], check_names=False
        )
        pd.testing.assert_series_equal(da["n_selected"], db["n_selected"], check_names=False)

    def test_failed_dataset_is_logged_not_fabricated(self, tmp_path: Path):
        cfg = quick_config()
        cfg.dataset_sizes = [100]
        cfg.dataset_files = {100: str(tmp_path / "does_not_exist.csv")}
        cfg.run_bpso_multiseed = False
        cfg.run_named_weights = False
        cfg.run_dirichlet = False
        cfg.run_epss = False
        payload = run_all_experiments(tmp_path / "fail", cfg, write_figures=False)
        cmp_df = payload["tables"]["method_comparison.csv"]
        assert not cmp_df["ran_successfully"].any()
        assert cmp_df["error_message"].str.len().gt(0).all()
        assert cmp_df["total_risk_reduction"].isna().all()
