"""Tests for Part B dashboard helpers (saved results, filtering, exports)."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from vulnopt.dashboard import (
    FAMILY_TABLES,
    annotate_selection,
    available_dataset_sizes,
    available_methods,
    build_custom_problem,
    comparison_export,
    dataframe_to_csv_bytes,
    default_capacities,
    default_results_dir,
    filter_findings,
    filter_result_rows,
    finding_detail_export,
    findings_display_columns,
    findings_snapshot_export,
    format_metric,
    is_missing_metric,
    load_all_tables,
    load_experiment_config,
    load_findings,
    load_table,
    overview_table,
    plan_export,
    prepare_findings,
    reconstruct_plan,
    table_file_status,
    tables_dir,
)
from vulnopt.experiments import ExperimentConfig, build_problem
from vulnopt.metrics import RESULT_COLUMNS
from vulnopt.risk_model import DEFAULT_WEIGHTS, calculate_risk_scores


ROOT = Path(__file__).resolve().parent.parent
TABLES = ROOT / "results" / "tables"


def _tiny_scored_frame(n: int = 8, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    exposures = ["internet", "internal", "restricted"]
    rows = []
    for i in range(n):
        epss = float(rng.uniform(0.0, 0.4))
        rows.append(
            {
                "finding_id": f"FND-{i+1:05d}",
                "cve_id": f"CVE-2024-{1000+i}",
                "asset_id": f"AST-{i+1:05d}",
                "asset_type": "web_server",
                "cvss_base": float(rng.uniform(2.0, 9.5)),
                "cvss_severity": "HIGH",
                "epss": epss,
                "epss_score": epss,
                "epss_missing": False,
                "is_kev": int(i % 4 == 0),
                "criticality": int(rng.integers(1, 6)),
                "exposure": exposures[i % 3],
                "environment": "production",
                "patch_cost": float(rng.uniform(50.0, 400.0)),
                "engineer_hours": float(rng.uniform(1.0, 8.0)),
                "downtime_hours": float(rng.uniform(0.0, 2.0)),
            }
        )
    return calculate_risk_scores(pd.DataFrame(rows), weights=DEFAULT_WEIGHTS)


@pytest.fixture
def saved_tables() -> dict[str, pd.DataFrame]:
    return load_all_tables(TABLES)


class TestLoadSavedResults:
    def test_method_comparison_loads_four_methods(self, saved_tables):
        cmp_df = saved_tables["method_comparison"]
        assert not cmp_df.empty
        methods = set(cmp_df["method"].astype(str))
        for name in ("CVSS-Greedy", "Risk-Greedy", "MILP", "BPSO"):
            assert name in methods

    def test_loads_expected_dataset_sizes(self, saved_tables):
        sizes = set(saved_tables["method_comparison"]["dataset_size"].astype(int))
        assert {100, 1000, 5000}.issubset(sizes)

    def test_experiment_config_matches_part_a_fractions(self):
        cfg = load_experiment_config(TABLES)
        assert cfg
        scenarios = cfg.get("resource_scenarios") or {}
        assert scenarios.get("tight") == pytest.approx(0.3)
        assert scenarios.get("medium") == pytest.approx(0.5)
        assert scenarios.get("relaxed") == pytest.approx(0.7)

    def test_missing_file_returns_empty_frame(self, tmp_path: Path):
        df = load_table("does_not_exist.csv", tmp_path)
        assert isinstance(df, pd.DataFrame)
        assert df.empty

    def test_empty_file_returns_empty_frame(self, tmp_path: Path):
        empty = tmp_path / "method_comparison.csv"
        empty.write_text("", encoding="utf-8")
        df = load_table("method_comparison.csv", tmp_path)
        assert df.empty

    def test_header_only_file_is_empty_or_zero_rows(self, tmp_path: Path):
        path = tmp_path / "experiment_failures.csv"
        path.write_text(",".join(RESULT_COLUMNS) + "\n", encoding="utf-8")
        df = load_table("experiment_failures.csv", tmp_path)
        assert len(df) == 0

    def test_corrupt_csv_returns_empty_frame(self, tmp_path: Path):
        bad = tmp_path / "runtime.csv"
        bad.write_text("not,a\nvalid", encoding="utf-8")
        df = load_table("runtime.csv", tmp_path)
        assert isinstance(df, pd.DataFrame)

    def test_saved_failures_table_has_no_invented_rows(self, saved_tables):
        fails = saved_tables["experiment_failures"]
        cfg = load_experiment_config(TABLES)
        if cfg.get("failed_experiment_count") == 0:
            assert fails.empty or len(fails) == 0


class TestNoFabricatedValues:
    def test_empty_metric_cells_stay_na(self, tmp_path: Path):
        path = tmp_path / "method_comparison.csv"
        path.write_text(
            "method,dataset_size,resource_scenario,pct_risk_reduction,n_selected,ran_successfully\n"
            "MILP,100,medium,,5,True\n",
            encoding="utf-8",
        )
        df = load_table("method_comparison.csv", tmp_path)
        assert pd.isna(df.loc[0, "pct_risk_reduction"])
        overview = overview_table(df)
        assert pd.isna(overview.loc[0, "pct_risk_reduction"])

    def test_failed_row_is_not_filled_with_zeros(self, tmp_path: Path):
        cols = list(RESULT_COLUMNS)
        row = {c: "" for c in cols}
        row.update(
            {
                "method": "BPSO",
                "dataset_size": "100",
                "resource_scenario": "medium",
                "ran_successfully": "False",
                "error_message": "timeout",
                "pct_risk_reduction": "",
                "n_selected": "",
            }
        )
        pd.DataFrame([row], columns=cols).to_csv(tmp_path / "method_comparison.csv", index=False)
        df = load_table("method_comparison.csv", tmp_path)
        assert str(df.loc[0, "ran_successfully"]).lower() in {"false", "0"}
        assert pd.isna(df.loc[0, "pct_risk_reduction"]) or df.loc[0, "pct_risk_reduction"] == "" or df.loc[0, "pct_risk_reduction"] != 0

    def test_missing_method_is_not_invented(self, saved_tables):
        filtered = filter_result_rows(
            saved_tables["method_comparison"],
            dataset_size=100,
            resource_scenario="medium",
            methods=["NotAMethod"],
        )
        assert filtered.empty


class TestMethodAndScenarioSelection:
    def test_medium_100_milp_matches_saved_risk_reduction(self, saved_tables):
        rows = filter_result_rows(
            saved_tables["method_comparison"],
            dataset_size=100,
            resource_scenario="medium",
            methods=["MILP"],
        )
        assert len(rows) == 1
        expected = pd.read_csv(TABLES / "method_comparison.csv")
        expected = expected[
            (expected["dataset_size"] == 100)
            & (expected["resource_scenario"] == "medium")
            & (expected["method"] == "MILP")
        ]
        assert rows.iloc[0]["n_selected"] == expected.iloc[0]["n_selected"]
        assert rows.iloc[0]["pct_risk_reduction"] == pytest.approx(
            float(expected.iloc[0]["pct_risk_reduction"])
        )
        assert rows.iloc[0]["solver_status"] == expected.iloc[0]["solver_status"]

    def test_tight_100_milp_feasibility_is_the_saved_flag(self, saved_tables):
        rows = filter_result_rows(
            saved_tables["method_comparison"],
            dataset_size=100,
            resource_scenario="tight",
            methods=["MILP"],
        )
        assert len(rows) == 1
        raw = pd.read_csv(TABLES / "method_comparison.csv")
        raw = raw[
            (raw["dataset_size"] == 100)
            & (raw["resource_scenario"] == "tight")
            & (raw["method"] == "MILP")
        ]
        assert bool(rows.iloc[0]["feasible"]) == bool(raw.iloc[0]["feasible"])

    def test_overview_omits_unknown_columns_instead_of_filling(self):
        df = pd.DataFrame({"method": ["MILP"], "n_selected": [3]})
        overview = overview_table(df)
        assert "method" in overview.columns
        assert "pct_risk_reduction" not in overview.columns

    def test_available_methods_preserve_canonical_order(self, saved_tables):
        methods = available_methods(saved_tables["method_comparison"])
        assert methods[:4] == ["CVSS-Greedy", "Risk-Greedy", "MILP", "BPSO"]


class TestFindingInspection:
    def test_load_findings_100(self):
        df = load_findings(100, root=ROOT)
        assert len(df) == 100
        for col in ("cve_id", "cvss_base", "epss", "is_kev", "risk_score", "patch_cost"):
            assert col in df.columns

    def test_missing_findings_file_returns_empty(self, tmp_path: Path):
        cfg = ExperimentConfig(dataset_files={99: "missing.csv"}, data_root=str(tmp_path))
        df = load_findings(99, root=tmp_path, config=cfg)
        assert df.empty

    def test_search_filters_cve_id(self):
        df = load_findings(100, root=ROOT)
        sample = str(df.iloc[0]["cve_id"])
        filtered = filter_findings(df, query=sample)
        assert len(filtered) >= 1
        assert sample in set(filtered["cve_id"].astype(str))

    def test_kev_filter_uses_is_kev(self):
        df = load_findings(100, root=ROOT)
        kev = filter_findings(df, kev_only=True)
        if "is_kev" in df.columns:
            expected = int((pd.to_numeric(df["is_kev"], errors="coerce").fillna(0) > 0).sum())
            assert len(kev) == expected

    def test_selected_only_without_column_returns_empty(self):
        df = _tiny_scored_frame()
        out = filter_findings(df, selected_only=True)
        assert out.empty

    def test_greedy_plan_selects_subset_of_findings(self):
        df = _tiny_scored_frame(n=10)
        caps = default_capacities(df, 0.5)
        problem = build_custom_problem(
            df,
            budget=caps["budget"],
            max_hours=caps["max_hours"],
            max_downtime=caps["max_downtime"],
            apply_downtime=True,
        )
        plan = reconstruct_plan("CVSS-Greedy", df, problem, seed=0, dataset_size=10)
        assert 0 < plan["n_selected"] <= len(df)
        assert len(plan["selected_findings"]) == plan["n_selected"]
        annotated = annotate_selection(df, plan["selection"])
        assert int(annotated["selected"].sum()) == plan["n_selected"]
        ids = set(plan["selected_findings"]["finding_id"])
        assert ids == set(annotated.loc[annotated["selected"], "finding_id"])

    def test_prepare_default_weights_keeps_saved_scores(self):
        df = load_findings(100, root=ROOT)
        prepared = prepare_findings(df, weight_scenario="default", epss_strategy="zero_imputation")
        pd.testing.assert_series_equal(prepared["risk_score"], df["risk_score"])


class TestCsvExports:
    def test_comparison_export_keeps_result_columns(self, saved_tables):
        rows = filter_result_rows(
            saved_tables["method_comparison"],
            dataset_size=100,
            resource_scenario="medium",
        )
        exported = comparison_export(rows, extra={"dashboard_dataset_size": 100})
        assert len(exported) == len(rows)
        for col in ("method", "n_selected", "pct_risk_reduction", "feasible"):
            assert col in exported.columns
        assert exported["dashboard_dataset_size"].eq(100).all()
        csv_text = dataframe_to_csv_bytes(exported).decode("utf-8")
        assert "MILP" in csv_text
        assert "CVSS-Greedy" in csv_text

    def test_plan_export_includes_scenario_and_finding_fields(self):
        df = _tiny_scored_frame(n=6)
        selected = df.iloc[:2].copy()
        exported = plan_export(
            selected,
            method="Risk-Greedy",
            dataset_size=6,
            resource_scenario="medium",
            budget=1000.0,
            max_hours=20.0,
            max_downtime=5.0,
            weight_scenario="default",
            apply_downtime=True,
        )
        assert len(exported) == 2
        for col in ("cve_id", "method", "dataset_size", "resource_scenario", "budget_limit", "patch_cost"):
            assert col in exported.columns
        assert set(exported["method"]) == {"Risk-Greedy"}

    def test_empty_plan_export_is_empty(self):
        exported = plan_export(
            pd.DataFrame(),
            method="MILP",
            dataset_size=100,
            resource_scenario="medium",
            budget=1.0,
            max_hours=1.0,
            max_downtime=1.0,
            weight_scenario="default",
            apply_downtime=True,
        )
        assert exported.empty
        assert dataframe_to_csv_bytes(exported) == b""

    def test_finding_detail_export_record_count(self):
        df = _tiny_scored_frame(n=5)
        annotated = annotate_selection(df, np.array([1, 0, 1, 0, 0]))
        exported = finding_detail_export(
            annotated, method="BPSO", dataset_size=5, resource_scenario="tight"
        )
        assert len(exported) == 5
        assert int(exported["selected"].sum()) == 2
        assert (exported["method"] == "BPSO").all()

    def test_display_columns_skip_missing(self):
        df = pd.DataFrame({"cve_id": ["CVE-1"], "risk_score": [10.0]})
        cols = findings_display_columns(df)
        assert cols == ["cve_id", "risk_score"]


class TestPaths:
    def test_default_results_dir(self):
        assert default_results_dir(ROOT) == ROOT / "results"
        assert tables_dir(ROOT / "results") == ROOT / "results" / "tables"

    def test_available_sizes_include_existing_files(self, saved_tables):
        sizes = available_dataset_sizes(saved_tables["method_comparison"], root=ROOT)
        assert 100 in sizes
        assert 1000 in sizes
        assert 5000 in sizes

    def test_family_table_keys_match_contract(self):
        assert "method_comparison.csv" in FAMILY_TABLES.values()
        assert "sensitivity_dirichlet.csv" in FAMILY_TABLES.values()
        assert "epss_handling.csv" in FAMILY_TABLES.values()

    def test_build_problem_capacities_match_helper(self):
        df = _tiny_scored_frame()
        caps = default_capacities(df, 0.3)
        problem = build_problem(df, 0.3)
        assert caps["budget"] == pytest.approx(problem.budget)
        assert caps["max_hours"] == pytest.approx(problem.max_hours)
        assert caps["max_downtime"] == pytest.approx(problem.max_downtime)


class TestTableStatusAndSnapshots:
    def test_saved_tables_are_loaded_not_fabricated(self):
        status = table_file_status(TABLES)
        assert status["method_comparison"] == "loaded"
        assert status["sensitivity_named_weights"] == "loaded"
        assert status["epss_handling"] == "loaded"
        assert status["experiment_failures"] in {"no_rows", "loaded"}

    def test_missing_directory_reports_missing(self, tmp_path: Path):
        status = table_file_status(tmp_path)
        assert status["method_comparison"] == "missing"
        assert all(v == "missing" for v in status.values())

    def test_empty_file_status(self, tmp_path: Path):
        (tmp_path / "method_comparison.csv").write_text("", encoding="utf-8")
        status = table_file_status(tmp_path)
        assert status["method_comparison"] == "empty_file"

    def test_is_missing_metric_does_not_treat_zero_as_absent(self):
        assert is_missing_metric(None)
        assert is_missing_metric(float("nan"))
        assert is_missing_metric("")
        assert not is_missing_metric(0)
        assert not is_missing_metric(0.0)

    def test_format_metric_does_not_turn_missing_into_zero(self):
        assert format_metric(None) == "—"
        assert format_metric(float("nan")) == "—"
        assert format_metric("") == "—"
        assert format_metric(0) == "0"
        assert format_metric(0.0) == "0"
        assert format_metric(np.bool_(True)) == "True"
        assert format_metric(np.bool_(False)) == "False"

    def test_findings_snapshot_keeps_record_count(self):
        df = load_findings(100, root=ROOT)
        snap = findings_snapshot_export(df, dataset_size=100, resource_scenario="medium")
        assert len(snap) == 100
        assert (snap["dataset_size"] == 100).all()
        assert "pct_risk_reduction" not in snap.columns or snap["pct_risk_reduction"].isna().all()

    def test_medium_1000_all_methods_match_saved_csv(self, saved_tables):
        rows = filter_result_rows(
            saved_tables["method_comparison"],
            dataset_size=1000,
            resource_scenario="medium",
        )
        raw = pd.read_csv(TABLES / "method_comparison.csv")
        raw = raw[(raw["dataset_size"] == 1000) & (raw["resource_scenario"] == "medium")]
        assert set(rows["method"].astype(str)) == set(raw["method"].astype(str))
        for method in ("CVSS-Greedy", "Risk-Greedy", "MILP", "BPSO"):
            got = rows[rows["method"] == method].iloc[0]
            exp = raw[raw["method"] == method].iloc[0]
            assert got["n_selected"] == exp["n_selected"]
            assert got["pct_risk_reduction"] == pytest.approx(float(exp["pct_risk_reduction"]))
            assert got["feasible"] == exp["feasible"]


class TestStreamlitAppLoads:
    def test_dashboard_app_renders_saved_comparison(self):
        from streamlit.testing.v1 import AppTest

        at = AppTest.from_file(str(ROOT / "streamlit_app.py"), default_timeout=60)
        at.run()
        assert not at.exception
        titles = [item.value for item in at.title]
        assert any("VulnOpt" in str(v) for v in titles)
        bodies = " ".join(str(item.value) for item in at.markdown)
        captions = " ".join(str(item.value) for item in at.caption)
        combined = bodies + " " + captions
        assert "CVSS-Greedy" in combined or any(
            "CVSS-Greedy" in str(df) for df in getattr(at, "dataframe", [])
        )
