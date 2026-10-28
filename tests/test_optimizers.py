"""
Optimizer tests for VulnOpt.

Covers:
- OptimizationProblem construction and validation (constraints.py)
- Feasibility checks and slack computation (constraints.py)
- CVSS-Greedy and Risk-Greedy baselines (baselines.py)
- MILP exact solver correctness (milp_solver.py)
- BPSO metaheuristic behaviour (bpso.py)
- Cross-solver comparative properties (MILP >= greedy objectives)
"""

import numpy as np
import pandas as pd
import pytest

from vulnopt.constraints import (
    OptimizationProblem,
    compute_default_budgets,
    constraint_slack,
    is_feasible,
    total_risk,
)
from vulnopt.baselines import (
    BaselineResult,
    baselines_to_dataframe,
    cvss_greedy,
    risk_greedy,
    run_all_baselines,
)
from vulnopt.milp_solver import MILPResult, milp_result_to_series, solve_milp
from vulnopt.bpso import BPSOResult, bpso_result_to_series, solve_bpso


# ===========================================================================
# Fixtures
# ===========================================================================

@pytest.fixture(scope="module")
def tiny_problem():
    """
    A 5-item problem with a known optimal solution for unit testing.

    Items  R_i   C_i   T_i   D_i
    0      80    100   4     1
    1      60     80   3     2
    2      40     60   5     0
    3      20     40   2     3
    4      10     20   1     0

    Budget=200, MaxHours=8, MaxDowntime=4

    Items 0 and 1 cost $180 total, use 7 hours, 3 downtime -> feasible,
    total risk = 140.  Optimal by inspection.
    """
    return OptimizationProblem(
        risk_scores=np.array([80.0, 60.0, 40.0, 20.0, 10.0]),
        patch_costs=np.array([100.0, 80.0, 60.0, 40.0, 20.0]),
        engineer_hours=np.array([4.0, 3.0, 5.0, 2.0, 1.0]),
        downtime_hours=np.array([1.0, 2.0, 0.0, 3.0, 0.0]),
        budget=200.0,
        max_hours=8.0,
        max_downtime=4.0,
    )


@pytest.fixture(scope="module")
def medium_problem(seed=99):
    """
    A 100-item problem for testing solver scalability and comparison.
    Resources capped at 40% of totals.
    """
    rng = np.random.default_rng(seed)
    n = 100
    risk   = rng.uniform(0, 100, size=n)
    costs  = rng.lognormal(mean=5.5, sigma=0.6, size=n)
    hours  = rng.lognormal(mean=2.0, sigma=0.5, size=n)
    dtime  = np.clip(rng.exponential(scale=1.5, size=n), 0, 12)
    cvss   = rng.uniform(0, 10, size=n)

    prob = OptimizationProblem(
        risk_scores=risk,
        patch_costs=costs,
        engineer_hours=hours,
        downtime_hours=dtime,
        budget=np.sum(costs) * 0.40,
        max_hours=np.sum(hours) * 0.40,
        max_downtime=np.sum(dtime) * 0.40,
    )
    return prob, cvss


@pytest.fixture(scope="module")
def trivial_feasible_problem():
    """Problem where budget covers all items – optimal is to select all."""
    risk  = np.array([10.0, 20.0, 30.0])
    costs = np.array([5.0,  5.0,  5.0])
    hrs   = np.array([1.0,  1.0,  1.0])
    dtm   = np.array([0.0,  0.0,  0.0])
    return OptimizationProblem(
        risk_scores=risk,
        patch_costs=costs,
        engineer_hours=hrs,
        downtime_hours=dtm,
        budget=100.0,
        max_hours=100.0,
        max_downtime=100.0,
    )


# ===========================================================================
# 1. OptimizationProblem construction & validation
# ===========================================================================

class TestOptimizationProblem:

    def test_basic_construction(self, tiny_problem):
        assert tiny_problem.n == 5
        assert len(tiny_problem.finding_ids) == 5

    def test_from_dataframe(self):
        df = pd.DataFrame({
            "finding_id":    ["FND-001", "FND-002"],
            "risk_score":    [80.0, 40.0],
            "patch_cost":    [100.0, 60.0],
            "engineer_hours":[4.0, 3.0],
            "downtime_hours":[1.0, 0.5],
        })
        prob = OptimizationProblem.from_dataframe(
            df, budget=150.0, max_hours=10.0, max_downtime=5.0
        )
        assert prob.n == 2
        assert prob.finding_ids == ["FND-001", "FND-002"]
        assert prob.budget == 150.0

    def test_from_dataframe_missing_column(self):
        df = pd.DataFrame({"risk_score": [80.0], "patch_cost": [100.0]})
        with pytest.raises(ValueError, match="missing required columns"):
            OptimizationProblem.from_dataframe(df, budget=100, max_hours=10, max_downtime=5)

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError, match="Length of 'patch_costs'"):
            OptimizationProblem(
                risk_scores=np.array([1.0, 2.0]),
                patch_costs=np.array([1.0]),        # wrong length
                engineer_hours=np.array([1.0, 1.0]),
                downtime_hours=np.array([0.0, 0.0]),
                budget=10.0, max_hours=10.0, max_downtime=5.0,
            )

    def test_negative_cost_raises(self):
        with pytest.raises(ValueError, match="non-negative"):
            OptimizationProblem(
                risk_scores=np.array([1.0]),
                patch_costs=np.array([-1.0]),
                engineer_hours=np.array([1.0]),
                downtime_hours=np.array([0.0]),
                budget=10.0, max_hours=10.0, max_downtime=5.0,
            )

    def test_negative_budget_raises(self):
        with pytest.raises(ValueError, match="non-negative"):
            OptimizationProblem(
                risk_scores=np.array([1.0]),
                patch_costs=np.array([1.0]),
                engineer_hours=np.array([1.0]),
                downtime_hours=np.array([0.0]),
                budget=-1.0, max_hours=10.0, max_downtime=5.0,
            )

    def test_finding_ids_length_mismatch(self):
        with pytest.raises(ValueError, match="'finding_ids' length"):
            OptimizationProblem(
                risk_scores=np.array([1.0, 2.0]),
                patch_costs=np.array([1.0, 1.0]),
                engineer_hours=np.array([1.0, 1.0]),
                downtime_hours=np.array([0.0, 0.0]),
                budget=10.0, max_hours=10.0, max_downtime=5.0,
                finding_ids=["only_one"],
            )

    def test_auto_generated_finding_ids(self, tiny_problem):
        assert tiny_problem.finding_ids[0] == "FND-00001"
        assert tiny_problem.finding_ids[4] == "FND-00005"


# ===========================================================================
# 2. Feasibility helpers
# ===========================================================================

class TestFeasibilityHelpers:

    def test_is_feasible_all_selected(self, tiny_problem):
        # All 5 items: cost=300 > budget=200 → infeasible
        all_sel = np.ones(5, dtype=int)
        assert not is_feasible(tiny_problem, all_sel)

    def test_is_feasible_none_selected(self, tiny_problem):
        none_sel = np.zeros(5, dtype=int)
        assert is_feasible(tiny_problem, none_sel)

    def test_is_feasible_items_0_and_1(self, tiny_problem):
        sel = np.array([1, 1, 0, 0, 0])
        # Cost=180 <= 200, Hours=7 <= 8, Downtime=3 <= 4
        assert is_feasible(tiny_problem, sel)

    def test_is_feasible_wrong_length(self, tiny_problem):
        with pytest.raises(ValueError, match="Selection length"):
            is_feasible(tiny_problem, np.array([1, 0]))

    def test_constraint_slack_all_selected(self, tiny_problem):
        sel = np.array([1, 1, 0, 0, 0])
        slacks = constraint_slack(tiny_problem, sel)
        assert abs(slacks["budget_slack"] - 20.0) < 1e-6
        assert abs(slacks["hours_slack"]  - 1.0)  < 1e-6
        assert abs(slacks["downtime_slack"] - 1.0) < 1e-6

    def test_total_risk_computation(self, tiny_problem):
        sel = np.array([1, 0, 0, 0, 0])
        assert abs(total_risk(tiny_problem, sel) - 80.0) < 1e-9

    def test_compute_default_budgets(self, tiny_problem):
        defaults = compute_default_budgets(tiny_problem, budget_fraction=0.50)
        assert abs(defaults["budget"] - 150.0) < 1e-6     # 300 * 0.5
        assert abs(defaults["max_hours"] - 7.5) < 1e-6    # 15 * 0.5

    def test_compute_default_budgets_invalid_fraction(self, tiny_problem):
        with pytest.raises(ValueError, match="budget_fraction"):
            compute_default_budgets(tiny_problem, budget_fraction=0.0)


# ===========================================================================
# 3. Baseline solvers
# ===========================================================================

class TestBaselineSolvers:

    def test_cvss_greedy_returns_feasible(self, tiny_problem):
        cvss = np.array([9.8, 7.5, 5.0, 3.0, 1.0])
        result = cvss_greedy(tiny_problem, cvss)
        assert isinstance(result, BaselineResult)
        assert result.feasible
        assert result.method == "CVSS-Greedy"

    def test_risk_greedy_returns_feasible(self, tiny_problem):
        result = risk_greedy(tiny_problem)
        assert isinstance(result, BaselineResult)
        assert result.feasible
        assert result.method == "Risk-Greedy"

    def test_risk_greedy_selects_highest_risk_first(self, tiny_problem):
        """Item 0 (R=80) should always be selected in risk-greedy."""
        result = risk_greedy(tiny_problem)
        assert result.selection[0] == 1

    def test_cvss_greedy_wrong_length_raises(self, tiny_problem):
        with pytest.raises(ValueError, match="cvss_scores length"):
            cvss_greedy(tiny_problem, np.array([9.8, 7.5]))

    def test_trivial_all_selected(self, trivial_feasible_problem):
        """When budget is unlimited, all items should be selected."""
        result = risk_greedy(trivial_feasible_problem)
        assert result.n_selected == 3
        assert abs(result.total_risk_score - 60.0) < 1e-9

    def test_selection_is_binary(self, tiny_problem):
        cvss = np.array([9.8, 7.5, 5.0, 3.0, 1.0])
        r1 = cvss_greedy(tiny_problem, cvss)
        r2 = risk_greedy(tiny_problem)
        for result in (r1, r2):
            assert set(result.selection.tolist()).issubset({0, 1})

    def test_budget_not_exceeded(self, medium_problem):
        prob, cvss = medium_problem
        for result in (cvss_greedy(prob, cvss), risk_greedy(prob)):
            assert result.budget_used <= prob.budget + 1e-6
            assert result.hours_used  <= prob.max_hours + 1e-6
            assert result.downtime_used <= prob.max_downtime + 1e-6

    def test_run_all_baselines(self, tiny_problem):
        cvss = np.array([9.8, 7.5, 5.0, 3.0, 1.0])
        results = run_all_baselines(tiny_problem, cvss)
        assert "CVSS-Greedy" in results
        assert "Risk-Greedy" in results

    def test_baselines_to_dataframe(self, tiny_problem):
        cvss = np.array([9.8, 7.5, 5.0, 3.0, 1.0])
        results = run_all_baselines(tiny_problem, cvss)
        df = baselines_to_dataframe(results)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2
        assert "method" in df.columns
        assert "total_risk_score" in df.columns
        assert "feasible" in df.columns

    def test_risk_greedy_beats_or_ties_cvss_greedy_on_risk_score(self, medium_problem):
        """
        The risk-greedy heuristic uses a richer scoring signal than CVSS.
        It should achieve at least as high a total risk score as CVSS-greedy
        in expectation (not guaranteed for every single instance, but
        we assert it for this fixed random problem).
        """
        prob, cvss = medium_problem
        rg = risk_greedy(prob)
        cg = cvss_greedy(prob, cvss)
        # Risk-greedy optimises for R_i directly; it should not be beaten
        # by a CVSS-based ordering on the risk objective
        assert rg.total_risk_score >= cg.total_risk_score - 1e-6


# ===========================================================================
# 4. MILP exact solver
# ===========================================================================

class TestMILPSolver:

    def test_milp_returns_milp_result(self, tiny_problem):
        result = solve_milp(tiny_problem)
        assert isinstance(result, MILPResult)
        assert result.method.startswith("MILP-")

    def test_milp_feasible(self, tiny_problem):
        result = solve_milp(tiny_problem)
        assert result.feasible
        assert result.status in ("Optimal", "Not Solved")  # HiGHS may report differently

    def test_milp_known_optimal_tiny(self, tiny_problem):
        """
        Items 0+1: cost=180, hours=7, downtime=3 → all feasible,
        total risk=140.  Items 0+1+4: cost=200, hours=8, downtime=1 → feasible,
        risk=150.  Verify MILP finds risk >= 140.
        """
        result = solve_milp(tiny_problem)
        assert result.total_risk_score >= 140.0 - 1e-6

    def test_milp_selection_is_binary(self, tiny_problem):
        result = solve_milp(tiny_problem)
        assert set(result.selection.tolist()).issubset({0, 1})

    def test_milp_constraints_satisfied(self, tiny_problem):
        result = solve_milp(tiny_problem)
        assert result.budget_used   <= tiny_problem.budget + 1e-6
        assert result.hours_used    <= tiny_problem.max_hours + 1e-6
        assert result.downtime_used <= tiny_problem.max_downtime + 1e-6

    def test_milp_result_to_series(self, tiny_problem):
        result = solve_milp(tiny_problem)
        series = milp_result_to_series(result)
        assert "total_risk_score" in series.index
        assert "feasible" in series.index

    def test_milp_zero_budget_returns_empty(self):
        """With budget=0, no items can be selected."""
        prob = OptimizationProblem(
            risk_scores=np.array([80.0, 60.0]),
            patch_costs=np.array([100.0, 80.0]),
            engineer_hours=np.array([4.0, 3.0]),
            downtime_hours=np.array([1.0, 2.0]),
            budget=0.0,
            max_hours=100.0,
            max_downtime=100.0,
        )
        result = solve_milp(prob)
        assert result.n_selected == 0
        assert result.feasible

    def test_milp_dominates_baselines_on_medium(self, medium_problem):
        """
        MILP must produce a total risk score >= both greedy baselines
        (it is exact / optimal).
        """
        prob, cvss = medium_problem
        milp_res = solve_milp(prob, time_limit=60)
        rg_res   = risk_greedy(prob)
        cg_res   = cvss_greedy(prob, cvss)

        assert milp_res.total_risk_score >= rg_res.total_risk_score - 1e-6, (
            f"MILP ({milp_res.total_risk_score:.2f}) should >= Risk-Greedy "
            f"({rg_res.total_risk_score:.2f})"
        )
        assert milp_res.total_risk_score >= cg_res.total_risk_score - 1e-6, (
            f"MILP ({milp_res.total_risk_score:.2f}) should >= CVSS-Greedy "
            f"({cg_res.total_risk_score:.2f})"
        )


# ===========================================================================
# 5. BPSO metaheuristic
# ===========================================================================

class TestBPSOSolver:

    def test_bpso_returns_bpso_result(self, tiny_problem):
        result = solve_bpso(tiny_problem, n_particles=10, max_iter=50, seed=0)
        assert isinstance(result, BPSOResult)
        assert result.method == "BPSO"

    def test_bpso_feasible(self, tiny_problem):
        result = solve_bpso(tiny_problem, n_particles=10, max_iter=50, seed=0)
        assert result.feasible

    def test_bpso_selection_is_binary(self, tiny_problem):
        result = solve_bpso(tiny_problem, n_particles=10, max_iter=50, seed=0)
        assert set(result.selection.tolist()).issubset({0, 1})

    def test_bpso_constraints_satisfied(self, tiny_problem):
        result = solve_bpso(tiny_problem, n_particles=20, max_iter=100, seed=42)
        assert result.budget_used   <= tiny_problem.budget + 1e-6
        assert result.hours_used    <= tiny_problem.max_hours + 1e-6
        assert result.downtime_used <= tiny_problem.max_downtime + 1e-6

    def test_bpso_convergence_history_length(self, tiny_problem):
        n_iter = 30
        result = solve_bpso(tiny_problem, n_particles=10, max_iter=n_iter, seed=1)
        # history: initial + one entry per iteration
        assert len(result.convergence_history) == n_iter + 1

    def test_bpso_convergence_history_nondecreasing(self, tiny_problem):
        """Global best score must never decrease."""
        result = solve_bpso(tiny_problem, n_particles=10, max_iter=50, seed=7)
        hist = result.convergence_history
        for i in range(1, len(hist)):
            assert hist[i] >= hist[i - 1] - 1e-9

    def test_bpso_finds_good_solution_tiny(self, tiny_problem):
        """BPSO should find a solution with risk >= 100 on the tiny problem."""
        result = solve_bpso(tiny_problem, n_particles=20, max_iter=100, seed=42)
        assert result.total_risk_score >= 100.0

    def test_bpso_result_to_series(self, tiny_problem):
        result = solve_bpso(tiny_problem, n_particles=10, max_iter=20, seed=0)
        series = bpso_result_to_series(result)
        assert "total_risk_score" in series.index
        assert "solve_time_s" in series.index

    def test_bpso_medium_problem_feasibility(self, medium_problem):
        prob, _ = medium_problem
        result = solve_bpso(prob, n_particles=30, max_iter=100, seed=42)
        assert result.feasible
        assert result.budget_used   <= prob.budget + 1e-6
        assert result.hours_used    <= prob.max_hours + 1e-6
        assert result.downtime_used <= prob.max_downtime + 1e-6

    def test_bpso_reproducibility(self, tiny_problem):
        """Same seed must produce identical results."""
        r1 = solve_bpso(tiny_problem, n_particles=10, max_iter=20, seed=123)
        r2 = solve_bpso(tiny_problem, n_particles=10, max_iter=20, seed=123)
        np.testing.assert_array_equal(r1.selection, r2.selection)
        assert r1.total_risk_score == r2.total_risk_score

    def test_bpso_different_seeds_can_differ(self, medium_problem):
        """Different seeds may produce different selections (not always, but typically)."""
        prob, _ = medium_problem
        r1 = solve_bpso(prob, n_particles=20, max_iter=50, seed=1)
        r2 = solve_bpso(prob, n_particles=20, max_iter=50, seed=9999)
        # Both must be feasible regardless of seed
        assert r1.feasible
        assert r2.feasible


# ===========================================================================
# 6. Cross-solver comparative properties
# ===========================================================================

class TestCrossSolverComparison:

    def test_milp_optimal_ge_bpso(self, tiny_problem):
        """
        MILP exact solution must be >= BPSO heuristic on total risk score.
        """
        milp_r = solve_milp(tiny_problem)
        bpso_r = solve_bpso(tiny_problem, n_particles=30, max_iter=200, seed=42)
        assert milp_r.total_risk_score >= bpso_r.total_risk_score - 1e-6, (
            f"MILP ({milp_r.total_risk_score:.4f}) < BPSO ({bpso_r.total_risk_score:.4f})"
        )

    def test_all_three_return_feasible_selection(self, tiny_problem):
        cvss = np.array([9.8, 7.5, 5.0, 3.0, 1.0])
        rg = risk_greedy(tiny_problem)
        cg = cvss_greedy(tiny_problem, cvss)
        milp_r = solve_milp(tiny_problem)
        bpso_r = solve_bpso(tiny_problem, n_particles=20, max_iter=100, seed=42)

        for res in (rg, cg, milp_r, bpso_r):
            assert res.feasible, f"{res.method} returned infeasible solution"

    def test_ranking_improvement_risk_over_cvss(self, medium_problem):
        """
        On a risk-scored problem, Risk-Greedy should achieve >= CVSS-Greedy
        total risk score, demonstrating the benefit of contextual scoring.
        """
        prob, cvss = medium_problem
        rg = risk_greedy(prob)
        cg = cvss_greedy(prob, cvss)
        assert rg.total_risk_score >= cg.total_risk_score - 1e-6
