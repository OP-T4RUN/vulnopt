"""
Greedy Baseline Solvers for VulnOpt.

Two deterministic polynomial-time baselines that establish lower-bound
performance for comparison against the MILP exact solver and the BPSO
metaheuristic:

1. CVSS-Greedy  – ranks findings by raw CVSS base score descending,
                  then adds items one-by-one while all constraints hold.
                  Represents the "sort by severity" industry default.

2. Risk-Greedy  – ranks findings by the composite R_i risk-priority score
                  (which incorporates EPSS, KEV, asset criticality, and
                  exposure), then adds items greedily.

Both use an identical greedy-knapsack loop to ensure fair comparison.
"""

from typing import Any, Dict, List, NamedTuple

import numpy as np
import pandas as pd

from vulnopt.constraints import (
    OptimizationProblem,
    is_feasible,
    total_risk,
    constraint_slack,
)


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

class BaselineResult(NamedTuple):
    """
    Structured result returned by every baseline solver.

    Attributes
    ----------
    method : str
        Human-readable name of the solver.
    selection : np.ndarray
        Binary array of shape (n,); 1 = item selected.
    total_risk_score : float
        Sum of R_i for selected items.
    n_selected : int
        Number of findings selected.
    budget_used : float
        Total patch cost of the selected set.
    hours_used : float
        Total engineer-hours of the selected set.
    downtime_used : float
        Total downtime hours of the selected set.
    feasible : bool
        Whether the selection satisfies all constraints.
    slacks : dict
        Remaining capacity for each constraint.
    metadata : dict
        Solver-specific diagnostics.
    """
    method: str
    selection: np.ndarray
    total_risk_score: float
    n_selected: int
    budget_used: float
    hours_used: float
    downtime_used: float
    feasible: bool
    slacks: dict
    metadata: dict


# ---------------------------------------------------------------------------
# Internal greedy loop (shared by both baselines)
# ---------------------------------------------------------------------------

def _greedy_knapsack(
    problem: OptimizationProblem,
    priority_indices: np.ndarray,
) -> np.ndarray:
    """
    Add items in the given priority order while all constraints remain
    satisfied.  Skips (does not discard) items that would violate any
    constraint.

    Parameters
    ----------
    problem          : OptimizationProblem
    priority_indices : np.ndarray of int
        Indices into problem arrays in decreasing priority order.

    Returns
    -------
    selection : np.ndarray of int, shape (n,)
    """
    selection = np.zeros(problem.n, dtype=int)

    # Running consumed resources
    cost_used     = 0.0
    hours_used    = 0.0
    downtime_used = 0.0

    for idx in priority_indices:
        c_i = problem.patch_costs[idx]
        h_i = problem.engineer_hours[idx]
        d_i = problem.downtime_hours[idx]

        # Check if adding this item would violate any constraint
        if (
            cost_used     + c_i <= problem.budget
            and hours_used    + h_i <= problem.max_hours
            and downtime_used + d_i <= problem.max_downtime
        ):
            selection[idx] = 1
            cost_used     += c_i
            hours_used    += h_i
            downtime_used += d_i

    return selection


def _build_result(
    method: str,
    problem: OptimizationProblem,
    selection: np.ndarray,
    metadata: dict,
) -> BaselineResult:
    """Assemble a BaselineResult from a selection vector."""
    sel = np.asarray(selection, dtype=int)
    slacks = constraint_slack(problem, sel)
    return BaselineResult(
        method=method,
        selection=sel,
        total_risk_score=total_risk(problem, sel),
        n_selected=int(sel.sum()),
        budget_used=float(np.dot(problem.patch_costs,   sel)),
        hours_used=float(np.dot(problem.engineer_hours, sel)),
        downtime_used=float(np.dot(problem.downtime_hours, sel)),
        feasible=is_feasible(problem, sel),
        slacks=slacks,
        metadata=metadata,
    )


# ---------------------------------------------------------------------------
# Baseline 1: CVSS-Greedy
# ---------------------------------------------------------------------------

def cvss_greedy(
    problem: OptimizationProblem,
    cvss_scores: np.ndarray,
) -> BaselineResult:
    """
    Select remediation findings in descending order of CVSS base score.

    This baseline replicates the most common industry practice: simply
    fixing the highest-severity vulnerabilities first until resources run out.

    Parameters
    ----------
    problem     : OptimizationProblem
    cvss_scores : np.ndarray, shape (n,)
        Raw CVSS base scores (0–10) for each finding, in the same order
        as ``problem.risk_scores``.

    Returns
    -------
    BaselineResult
    """
    if len(cvss_scores) != problem.n:
        raise ValueError(
            f"cvss_scores length ({len(cvss_scores)}) must match problem.n ({problem.n})."
        )

    # Sort descending by CVSS, break ties by risk_score
    priority_indices = np.lexsort(
        (problem.risk_scores, cvss_scores)
    )[::-1]

    selection = _greedy_knapsack(problem, priority_indices)
    return _build_result(
        method="CVSS-Greedy",
        problem=problem,
        selection=selection,
        metadata={"sort_key": "cvss_base descending"},
    )


# ---------------------------------------------------------------------------
# Baseline 2: Risk-Greedy (contextual)
# ---------------------------------------------------------------------------

def risk_greedy(problem: OptimizationProblem) -> BaselineResult:
    """
    Select remediation findings in descending order of composite R_i score.

    This baseline uses the full configurable risk-priority score that
    incorporates EPSS, KEV, asset criticality, and network exposure —
    the VulnOpt contextual improvement over CVSS-only ranking.

    Parameters
    ----------
    problem : OptimizationProblem

    Returns
    -------
    BaselineResult
    """
    # Sort descending by risk score; stable sort for reproducibility
    priority_indices = np.argsort(problem.risk_scores)[::-1]
    selection = _greedy_knapsack(problem, priority_indices)
    return _build_result(
        method="Risk-Greedy",
        problem=problem,
        selection=selection,
        metadata={"sort_key": "risk_score descending"},
    )


# ---------------------------------------------------------------------------
# Convenience: run both baselines and return comparison dict
# ---------------------------------------------------------------------------

def run_all_baselines(
    problem: OptimizationProblem,
    cvss_scores: np.ndarray,
) -> Dict[str, BaselineResult]:
    """
    Run both baseline solvers and return a dict keyed by method name.

    Parameters
    ----------
    problem     : OptimizationProblem
    cvss_scores : np.ndarray, shape (n,)

    Returns
    -------
    dict  {"CVSS-Greedy": BaselineResult, "Risk-Greedy": BaselineResult}
    """
    results: Dict[str, BaselineResult] = {}
    results["CVSS-Greedy"] = cvss_greedy(problem, cvss_scores)
    results["Risk-Greedy"] = risk_greedy(problem)
    return results


def baselines_to_dataframe(results: Dict[str, BaselineResult]) -> pd.DataFrame:
    """
    Convert a dict of BaselineResult objects to a tidy summary DataFrame.
    One row per solver method.
    """
    rows = []
    for method, res in results.items():
        rows.append({
            "method":            res.method,
            "total_risk_score":  round(res.total_risk_score, 2),
            "n_selected":        res.n_selected,
            "budget_used":       round(res.budget_used, 2),
            "hours_used":        round(res.hours_used, 2),
            "downtime_used":     round(res.downtime_used, 2),
            "feasible":          res.feasible,
            "budget_slack":      round(res.slacks["budget_slack"], 2),
            "hours_slack":       round(res.slacks["hours_slack"], 2),
            "downtime_slack":    round(res.slacks["downtime_slack"], 2),
        })
    return pd.DataFrame(rows)
