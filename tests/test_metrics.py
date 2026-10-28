"""
Unit tests for VulnOpt evaluation metrics.
"""

import math

import numpy as np
import pandas as pd
import pytest

from vulnopt.baselines import cvss_greedy, risk_greedy
from vulnopt.bpso import solve_bpso
from vulnopt.constraints import OptimizationProblem
from vulnopt.metrics import (
    RESULT_COLUMNS,
    canonicalize_method,
    evaluate_selection,
    evaluate_solver_result,
    failed_result_row,
    records_to_dataframe,
)
from vulnopt.milp_solver import solve_milp


@pytest.fixture
def tiny_problem():
    return OptimizationProblem(
        risk_scores=np.array([80.0, 60.0, 40.0, 20.0, 10.0]),
        patch_costs=np.array([100.0, 80.0, 60.0, 40.0, 20.0]),
        engineer_hours=np.array([4.0, 3.0, 5.0, 2.0, 1.0]),
        downtime_hours=np.array([1.0, 2.0, 0.0, 3.0, 0.0]),
        budget=200.0,
        max_hours=8.0,
        max_downtime=4.0,
    )


class TestPercentageAndZeroHandling:
    def test_manual_risk_percentage(self, tiny_problem):
        # Items 0 and 1: risk 140 of 210 → 66.666...%
        sel = np.array([1, 1, 0, 0, 0])
        rec = evaluate_selection(tiny_problem, sel, method="Risk-Greedy")
        assert rec["total_risk_reduction"] == pytest.approx(140.0)
        assert rec["total_risk_unmitigated"] == pytest.approx(210.0)
        assert rec["pct_risk_reduction"] == pytest.approx(100.0 * 140.0 / 210.0)
        assert rec["n_selected"] == 2

    def test_zero_total_risk_yields_nan_percentage(self):
        problem = OptimizationProblem(
            risk_scores=np.array([0.0, 0.0]),
            patch_costs=np.array([10.0, 10.0]),
            engineer_hours=np.array([1.0, 1.0]),
            downtime_hours=np.array([0.0, 0.0]),
            budget=10.0,
            max_hours=10.0,
            max_downtime=10.0,
        )
        rec = evaluate_selection(problem, np.array([1, 0]), method="MILP")
        assert rec["total_risk_reduction"] == pytest.approx(0.0)
        assert rec["total_risk_unmitigated"] == pytest.approx(0.0)
        assert math.isnan(rec["pct_risk_reduction"])

    def test_zero_budget_limit_yields_nan_pct_used(self):
        problem = OptimizationProblem(
            risk_scores=np.array([10.0]),
            patch_costs=np.array([0.0]),
            engineer_hours=np.array([1.0]),
            downtime_hours=np.array([0.0]),
            budget=0.0,
            max_hours=10.0,
            max_downtime=10.0,
        )
        rec = evaluate_selection(problem, np.array([0]), method="CVSS-Greedy")
        assert math.isnan(rec["pct_budget_used"])
        assert rec["budget_consumed"] == pytest.approx(0.0)

    def test_zero_kev_denominator_is_nan(self, tiny_problem):
        kev = np.zeros(5, dtype=int)
        rec = evaluate_selection(
            tiny_problem, np.array([1, 0, 0, 0, 0]), method="BPSO", kev_flags=kev
        )
        assert rec["n_kev_total"] == 0
        assert rec["n_kev_selected"] == 0
        assert math.isnan(rec["pct_kev_coverage"])

    def test_kev_coverage_manual(self, tiny_problem):
        kev = np.array([1, 0, 1, 0, 0])
        rec = evaluate_selection(
            tiny_problem, np.array([1, 1, 0, 0, 0]), method="MILP", kev_flags=kev
        )
        # 2 KEV findings in the instance, 1 selected
        assert rec["n_kev_total"] == 2
        assert rec["n_kev_selected"] == 1
        assert rec["pct_kev_coverage"] == pytest.approx(50.0)


class TestResourceAndFeasibility:
    def test_consumption_and_remaining(self, tiny_problem):
        sel = np.array([1, 1, 0, 0, 0])
        rec = evaluate_selection(tiny_problem, sel, method="Risk-Greedy")
        assert rec["budget_consumed"] == pytest.approx(180.0)
        assert rec["budget_remaining"] == pytest.approx(20.0)
        assert rec["hours_consumed"] == pytest.approx(7.0)
        assert rec["hours_remaining"] == pytest.approx(1.0)
        assert rec["downtime_consumed"] == pytest.approx(3.0)
        assert rec["downtime_remaining"] == pytest.approx(1.0)
        assert rec["feasible"] is True
        assert rec["pct_budget_used"] == pytest.approx(90.0)

    def test_infeasible_selection_is_flagged(self, tiny_problem):
        rec = evaluate_selection(
            tiny_problem, np.ones(5, dtype=int), method="MILP"
        )
        assert rec["feasible"] is False
        assert rec["budget_remaining"] < 0
        assert rec["n_selected"] == 5

    def test_missing_kev_column_does_not_invent_coverage(self, tiny_problem):
        rec = evaluate_selection(tiny_problem, np.array([1, 0, 0, 0, 0]), method="MILP")
        assert math.isnan(rec["pct_kev_coverage"])
        assert math.isnan(rec["n_kev_total"])


class TestSchemaAndFailures:
    def test_result_schema_complete(self, tiny_problem):
        rec = evaluate_selection(
            tiny_problem, np.array([1, 0, 0, 0, 0]), method="MILP-CBC", runtime_s=0.12
        )
        assert set(RESULT_COLUMNS).issubset(rec.keys())
        assert rec["method"] == "MILP"
        assert rec["runtime_s"] == pytest.approx(0.12)
        assert rec["ran_successfully"] is True

    def test_canonicalize_methods(self):
        assert canonicalize_method("MILP-HiGHS") == "MILP"
        assert canonicalize_method("CVSS-Greedy") == "CVSS-Greedy"
        assert canonicalize_method("BPSO") == "BPSO"

    def test_failed_row_does_not_fill_metrics(self):
        rec = failed_result_row(error_message="solver crashed", method="BPSO")
        assert rec["ran_successfully"] is False
        assert rec["error_message"] == "solver crashed"
        assert math.isnan(rec["total_risk_reduction"])
        assert rec["method"] == "BPSO"

    def test_records_to_dataframe_column_order(self, tiny_problem):
        rec = evaluate_selection(
            tiny_problem, np.array([1, 0, 0, 0, 0]), method="Risk-Greedy"
        )
        df = records_to_dataframe([rec])
        assert list(df.columns) == list(RESULT_COLUMNS)


class TestOptimizerIntegration:
    def test_all_four_methods_share_schema(self, tiny_problem):
        cvss = np.array([9.8, 7.5, 5.0, 3.0, 1.0])
        kev = np.array([1, 0, 0, 0, 1])
        results = [
            evaluate_solver_result(tiny_problem, cvss_greedy(tiny_problem, cvss), kev_flags=kev),
            evaluate_solver_result(tiny_problem, risk_greedy(tiny_problem), kev_flags=kev),
            evaluate_solver_result(tiny_problem, solve_milp(tiny_problem, time_limit=20), kev_flags=kev),
            evaluate_solver_result(
                tiny_problem,
                solve_bpso(tiny_problem, n_particles=8, max_iter=15, seed=0),
                kev_flags=kev,
            ),
        ]
        df = records_to_dataframe(results)
        assert list(df["method"]) == ["CVSS-Greedy", "Risk-Greedy", "MILP", "BPSO"]
        assert df["ran_successfully"].all()
        assert df["feasible"].all()
        # MILP exposes status; greedy does not invent a MIP gap of 0 unless the solver did.
        assert df.loc[df["method"] == "MILP", "solver_status"].iloc[0] != ""
        greedy_gap = df.loc[df["method"] == "CVSS-Greedy", "optimality_gap"].iloc[0]
        assert math.isnan(greedy_gap)

    def test_metrics_match_solver_objective(self, tiny_problem):
        result = risk_greedy(tiny_problem)
        rec = evaluate_solver_result(tiny_problem, result)
        assert rec["total_risk_reduction"] == pytest.approx(result.total_risk_score)
        assert rec["n_selected"] == result.n_selected
        assert rec["budget_consumed"] == pytest.approx(result.budget_used)
        assert rec["hours_consumed"] == pytest.approx(result.hours_used)
        assert rec["downtime_consumed"] == pytest.approx(result.downtime_used)

    def test_bpso_seed_reproducibility_in_metrics(self, tiny_problem):
        r1 = solve_bpso(tiny_problem, n_particles=10, max_iter=20, seed=123)
        r2 = solve_bpso(tiny_problem, n_particles=10, max_iter=20, seed=123)
        m1 = evaluate_solver_result(tiny_problem, r1)
        m2 = evaluate_solver_result(tiny_problem, r2)
        assert m1["total_risk_reduction"] == m2["total_risk_reduction"]
        assert m1["n_selected"] == m2["n_selected"]
        np.testing.assert_array_equal(r1.selection, r2.selection)
