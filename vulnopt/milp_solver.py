"""
Mixed-Integer Linear Programming (MILP) Solver for VulnOpt.

Formulates the vulnerability remediation selection problem as a binary
linear program and solves it to provable optimality using PuLP.

Supports both:
- PuLP >= 4.x  (HiGHS via highspy, model-centric add_variable_dicts API)
- PuLP <  4.x  (CBC bundled solver, LpVariable standalone API)

Mathematical Formulation
------------------------
Decision variables:
    x_i in {0, 1}  for i = 1..n

Objective (maximise):
    max  sum_i  R_i * x_i

Subject to:
    sum_i  C_i * x_i  <=  B    (budget)
    sum_i  T_i * x_i  <=  H    (engineer-hours)
    sum_i  D_i * x_i  <=  W    (downtime window)
    x_i in {0, 1}

Where:
    R_i  = composite risk-priority score (0-100)
    C_i  = patch cost ($)
    T_i  = engineer-hours
    D_i  = downtime hours
    B, H, W = resource capacities

Notes
-----
- This is NP-Hard in general (3-dimensional Multi-dimensional Knapsack).
- HiGHS / CBC are exact for small-to-medium instances (n < ~2000).
- A time_limit parameter caps wall-clock time for large instances.
"""

import time
from typing import Any, Dict, NamedTuple, Optional, Tuple

import numpy as np
import pandas as pd

try:
    import pulp
    _PULP_AVAILABLE = True
    # Detect PuLP major version from __version__ string
    _PULP_MAJOR = int(pulp.__version__.split(".")[0])
except ImportError:  # pragma: no cover
    _PULP_AVAILABLE = False
    _PULP_MAJOR = 0

from vulnopt.constraints import (
    OptimizationProblem,
    constraint_slack,
    is_feasible,
    total_risk,
)


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

class MILPResult(NamedTuple):
    """
    Structured result returned by the MILP solver.

    Attributes
    ----------
    method : str
    selection : np.ndarray, shape (n,)  - binary
    total_risk_score : float
    n_selected : int
    budget_used : float
    hours_used : float
    downtime_used : float
    feasible : bool
    slacks : dict
    status : str
        Solver status string (e.g. 'Optimal', 'Infeasible').
    objective_value : float
        Objective value as reported by the solver.
    solve_time_s : float
        Wall-clock seconds consumed by the solver.
    optimality_gap : float
        Relative MIP gap at termination (0.0 = proven optimal).
    metadata : dict
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
    status: str
    objective_value: float
    solve_time_s: float
    optimality_gap: float
    metadata: dict


# ---------------------------------------------------------------------------
# PuLP 4.x helpers (HiGHS, new model API)
# ---------------------------------------------------------------------------

def _get_solver_v4(time_limit: int, mip_gap: float, verbose: bool):
    """Return (solver_obj, solver_name) for PuLP >= 4.x."""
    # HiGHS Python bindings (preferred)
    try:
        s = pulp.HiGHS(msg=verbose, timeLimit=time_limit, gapRel=mip_gap)
        s.available()   # raises if highspy not installed
        return s, "HiGHS"
    except Exception:
        pass

    # HiGHS command-line binary
    try:
        s = pulp.HiGHS_CMD(msg=verbose, timeLimit=time_limit, gapRel=mip_gap)
        if s.available():
            return s, "HiGHS_CMD"
    except Exception:
        pass

    raise RuntimeError(
        "No MIP solver found for PuLP >= 4.x. "
        "Run: pip install highspy\n"
        f"Available solvers: {pulp.listSolvers(onlyAvailable=True)}"
    )


def _solve_v4(
    problem: OptimizationProblem,
    time_limit: int,
    mip_gap: float,
    verbose: bool,
) -> Tuple[str, np.ndarray, str, float, float, float]:
    """
    Build and solve using the PuLP >= 4.x model-centric API.

    Returns (solver_name, selection, status_str, objective, gap, solve_time)
    """
    n = problem.n
    solver, solver_name = _get_solver_v4(time_limit, mip_gap, verbose)

    model = pulp.LpProblem("VulnOpt_Remediation", pulp.LpMaximize)
    x = model.add_variable_dicts("x", list(range(n)), cat="Binary")

    # Objective
    model += pulp.lpSum(float(problem.risk_scores[i]) * x[i] for i in range(n))

    # Constraints
    model.add(
        pulp.lpSum(float(problem.patch_costs[i]) * x[i] for i in range(n))
        <= problem.budget,
        name="Budget",
    )
    model.add(
        pulp.lpSum(float(problem.engineer_hours[i]) * x[i] for i in range(n))
        <= problem.max_hours,
        name="Hours",
    )
    model.add(
        pulp.lpSum(float(problem.downtime_hours[i]) * x[i] for i in range(n))
        <= problem.max_downtime,
        name="Downtime",
    )

    t0 = time.perf_counter()
    stats = model.solve(solver)
    solve_time = time.perf_counter() - t0

    status_str = str(stats.status_str)
    obj_val    = float(stats.objective or 0.0)
    gap        = float(stats.gap_rel or 0.0)
    has_sol    = bool(stats.has_solution)

    if has_sol:
        selection = np.array(
            [int(round(float(pulp.value(x[i]) or 0))) for i in range(n)],
            dtype=int,
        )
    else:
        selection = np.zeros(n, dtype=int)

    return solver_name, selection, status_str, obj_val, gap, solve_time


# ---------------------------------------------------------------------------
# PuLP < 4.x helpers (CBC, standalone LpVariable API)
# ---------------------------------------------------------------------------

def _get_solver_legacy(time_limit: int, mip_gap: float, verbose: bool):
    """Return (solver_obj, solver_name) for PuLP < 4.x."""
    msg = int(verbose)

    cbc = getattr(pulp, "PULP_CBC_CMD", None)
    if cbc is not None:
        return cbc(timeLimit=time_limit, gapRel=mip_gap, msg=msg), "CBC"

    coin = getattr(pulp, "COIN_CMD", None)
    if coin is not None:
        return coin(maxSeconds=time_limit, msg=msg), "COIN_CMD"

    raise RuntimeError(
        "No MIP solver found for PuLP < 4.x. "
        f"Available solvers: {pulp.listSolvers(onlyAvailable=True)}"
    )


def _solve_legacy(
    problem: OptimizationProblem,
    time_limit: int,
    mip_gap: float,
    verbose: bool,
) -> Tuple[str, np.ndarray, str, float, float, float]:
    """
    Build and solve using the legacy PuLP < 4.x standalone API.

    Returns (solver_name, selection, status_str, objective, gap, solve_time)
    """
    n = problem.n
    solver, solver_name = _get_solver_legacy(time_limit, mip_gap, verbose)

    model = pulp.LpProblem("VulnOpt_Remediation", pulp.LpMaximize)
    x = [pulp.LpVariable(f"x_{i}", cat=pulp.LpBinary) for i in range(n)]  # type: ignore

    model += pulp.lpSum(float(problem.risk_scores[i]) * x[i] for i in range(n))
    model += (
        pulp.lpSum(float(problem.patch_costs[i]) * x[i] for i in range(n))
        <= problem.budget,
        "Budget",
    )
    model += (
        pulp.lpSum(float(problem.engineer_hours[i]) * x[i] for i in range(n))
        <= problem.max_hours,
        "Hours",
    )
    model += (
        pulp.lpSum(float(problem.downtime_hours[i]) * x[i] for i in range(n))
        <= problem.max_downtime,
        "Downtime",
    )

    t0 = time.perf_counter()
    model.solve(solver)
    solve_time = time.perf_counter() - t0

    # Legacy PuLP uses pulp.LpStatus dict and pulp.constants.LpStatusOptimal
    _lp_status = getattr(pulp, "LpStatus", {})
    status_str = _lp_status.get(model.status, str(model.status))  # type: ignore
    _lp_optimal = getattr(
        getattr(pulp, "constants", pulp), "LpStatusOptimal", 1
    )
    is_optimal = model.status == _lp_optimal  # type: ignore
    obj_val = float(pulp.value(model.objective) or 0.0)
    gap = 0.0 if is_optimal else float("nan")

    if is_optimal or any(pulp.value(x[i]) is not None for i in range(n)):
        selection = np.array(
            [int(round(float(pulp.value(x[i]) or 0))) for i in range(n)],
            dtype=int,
        )
    else:
        selection = np.zeros(n, dtype=int)

    return solver_name, selection, status_str, obj_val, gap, solve_time


# ---------------------------------------------------------------------------
# Main public solver
# ---------------------------------------------------------------------------

def solve_milp(
    problem: OptimizationProblem,
    time_limit: int = 120,
    mip_gap: float = 1e-4,
    verbose: bool = False,
) -> MILPResult:
    """
    Solve the VulnOpt 3D-knapsack problem using Binary ILP.

    Automatically selects PuLP 4.x (HiGHS) or legacy (CBC) depending on
    which PuLP version is installed.

    Parameters
    ----------
    problem    : OptimizationProblem
    time_limit : int
        Maximum wall-clock seconds. Default 120 s.
    mip_gap    : float
        Relative optimality gap tolerance. Default 1e-4.
    verbose    : bool
        If True, print solver log to stdout.

    Returns
    -------
    MILPResult

    Raises
    ------
    ImportError
        If the ``pulp`` package is not installed.
    RuntimeError
        If no MIP solver backend is available.
    """
    if not _PULP_AVAILABLE:
        raise ImportError(
            "PuLP is required for MILP solving. "
            "Install with: pip install pulp highspy"
        )

    # Dispatch to correct API version
    if _PULP_MAJOR >= 4:
        solver_name, selection, status_str, obj_val, gap, solve_time = _solve_v4(
            problem, time_limit, mip_gap, verbose
        )
    else:
        solver_name, selection, status_str, obj_val, gap, solve_time = _solve_legacy(
            problem, time_limit, mip_gap, verbose
        )

    slacks  = constraint_slack(problem, selection)
    feasible = is_feasible(problem, selection)

    return MILPResult(
        method=f"MILP-{solver_name}",
        selection=selection,
        total_risk_score=total_risk(problem, selection),
        n_selected=int(selection.sum()),
        budget_used=float(np.dot(problem.patch_costs,    selection)),
        hours_used=float(np.dot(problem.engineer_hours,  selection)),
        downtime_used=float(np.dot(problem.downtime_hours, selection)),
        feasible=feasible,
        slacks=slacks,
        status=status_str,
        objective_value=obj_val,
        solve_time_s=round(solve_time, 4),
        optimality_gap=gap,
        metadata={
            "n_variables":       problem.n,
            "time_limit_s":      time_limit,
            "mip_gap_tolerance": mip_gap,
            "solver":            solver_name,
            "pulp_version":      pulp.__version__,
        },
    )


# ---------------------------------------------------------------------------
# Convenience helpers
# ---------------------------------------------------------------------------

def milp_result_to_series(result: MILPResult) -> pd.Series:
    """Return a named pandas Series summarising the MILP result."""
    return pd.Series({
        "method":            result.method,
        "status":            result.status,
        "total_risk_score":  round(result.total_risk_score, 2),
        "n_selected":        result.n_selected,
        "budget_used":       round(result.budget_used, 2),
        "hours_used":        round(result.hours_used, 2),
        "downtime_used":     round(result.downtime_used, 2),
        "feasible":          result.feasible,
        "optimality_gap":    result.optimality_gap,
        "solve_time_s":      result.solve_time_s,
    })
