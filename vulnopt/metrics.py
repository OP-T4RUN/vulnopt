"""
Consistent evaluation metrics for VulnOpt solvers.

All four methods (CVSS-Greedy, Risk-Greedy, MILP, BPSO) are scored with
the same definitions so comparisons are valid.

Metric definitions
------------------
total_risk_reduction
    Sum of modeled R_i for selected findings: sum_i R_i * x_i.
    This is the objective used by Risk-Greedy, MILP, and BPSO.

total_risk_unmitigated
    Sum of R_i over the full candidate set (the denominator for
    percentage risk reduction).  It is the modeled risk that would remain
    if nothing is remediated.

pct_risk_reduction
    100 * total_risk_reduction / total_risk_unmitigated.
    Undefined (NaN) when total_risk_unmitigated == 0.

budget / hours / downtime consumed
    Dot product of the corresponding cost vector with x.
    Remaining = limit - consumed (may be negative if infeasible).
    Percentage used = 100 * consumed / limit, or NaN if limit == 0.

n_selected
    Number of findings with x_i = 1.

n_kev_total / n_kev_selected / pct_kev_coverage
    Denominator n_kev_total is the number of candidate findings with
    is_kev == 1.  Numerator is how many of those are selected.
    pct_kev_coverage is NaN when n_kev_total == 0 (no KEV findings in
    the instance, so coverage is undefined — not reported as 0%).

feasible
    True iff budget, engineer-hours, and downtime constraints all hold
    (same predicate as vulnopt.constraints.is_feasible).

runtime_s
    Wall-clock seconds of the solver call, measured by the experiment
    runner.  Not invented inside this module.

solver_status / optimality_gap
    Passed through from MILP when present; NaN / empty otherwise.
    BPSO and greedy baselines have no MIP gap.
"""

from __future__ import annotations

import math
from typing import Any, Mapping, Optional, Sequence, Union

import numpy as np
import pandas as pd

from vulnopt.constraints import OptimizationProblem, constraint_slack, is_feasible, total_risk


# Canonical method labels used in result tables (dashboard contract).
CANONICAL_METHODS = ("CVSS-Greedy", "Risk-Greedy", "MILP", "BPSO")

# Unified column order for every experiment CSV.
RESULT_COLUMNS: tuple[str, ...] = (
    "experiment_id",
    "experiment_family",
    "dataset_size",
    "dataset_path",
    "resource_scenario",
    "resource_fraction",
    "method",
    "seed",
    "weight_scenario",
    "weight_sample_id",
    "epss_strategy",
    "n_findings",
    "n_selected",
    "total_risk_reduction",
    "total_risk_unmitigated",
    "pct_risk_reduction",
    "budget_limit",
    "budget_consumed",
    "budget_remaining",
    "pct_budget_used",
    "hours_limit",
    "hours_consumed",
    "hours_remaining",
    "pct_hours_used",
    "downtime_limit",
    "downtime_consumed",
    "downtime_remaining",
    "pct_downtime_used",
    "n_kev_total",
    "n_kev_selected",
    "pct_kev_coverage",
    "feasible",
    "runtime_s",
    "solver_runtime_s",
    "solver_status",
    "optimality_gap",
    "solver_backend",
    "n_particles",
    "max_iter",
    "milp_time_limit",
    "ran_successfully",
    "error_message",
)


def _safe_pct(numerator: float, denominator: float) -> float:
    """Return 100 * num/den, or NaN when the denominator is 0 or non-finite."""
    if denominator is None or not math.isfinite(denominator) or denominator == 0:
        return float("nan")
    if numerator is None or not math.isfinite(numerator):
        return float("nan")
    return 100.0 * float(numerator) / float(denominator)


def _as_float(value: Any, default: float = float("nan")) -> float:
    if value is None:
        return default
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def canonicalize_method(method: str) -> str:
    """Map solver-specific names (e.g. MILP-CBC) onto the four canonical labels."""
    if method is None:
        return ""
    name = str(method).strip()
    if name.startswith("MILP"):
        return "MILP"
    if name in CANONICAL_METHODS:
        return name
    lowered = name.lower()
    if "cvss" in lowered:
        return "CVSS-Greedy"
    if "risk" in lowered or "greedy" in lowered:
        return "Risk-Greedy"
    if "bpso" in lowered:
        return "BPSO"
    return name


def extract_solver_diagnostics(result: Any) -> dict[str, Any]:
    """Pull optional MILP/BPSO fields from a solver NamedTuple without inventing them."""
    status = getattr(result, "status", None)
    gap = getattr(result, "optimality_gap", None)
    solver_time = getattr(result, "solve_time_s", None)
    metadata = getattr(result, "metadata", None) or {}
    backend = metadata.get("solver") if isinstance(metadata, Mapping) else None
    n_particles = metadata.get("n_particles") if isinstance(metadata, Mapping) else None
    max_iter = metadata.get("max_iter") if isinstance(metadata, Mapping) else getattr(result, "n_iterations", None)
    return {
        "solver_status": "" if status is None else str(status),
        "optimality_gap": _as_float(gap) if gap is not None else float("nan"),
        "solver_runtime_s": _as_float(solver_time) if solver_time is not None else float("nan"),
        "solver_backend": "" if backend is None else str(backend),
        "n_particles": n_particles,
        "max_iter": max_iter,
    }


def kev_flags_from_frame(df: pd.DataFrame, n: int) -> np.ndarray:
    """Return a binary is_kev vector aligned with the findings DataFrame."""
    if "is_kev" not in df.columns:
        return np.zeros(n, dtype=int)
    flags = pd.to_numeric(df["is_kev"], errors="coerce").fillna(0).to_numpy(dtype=float)
    return (flags > 0).astype(int)


def evaluate_selection(
    problem: OptimizationProblem,
    selection: Union[np.ndarray, Sequence[int]],
    *,
    method: str,
    kev_flags: Optional[np.ndarray] = None,
    runtime_s: Optional[float] = None,
    solver_status: str = "",
    optimality_gap: Optional[float] = None,
    solver_runtime_s: Optional[float] = None,
    solver_backend: str = "",
    seed: Optional[int] = None,
) -> dict[str, Any]:
    """
    Compute the standard metric dict for a binary selection vector.

    Missing optional fields are stored as empty strings or NaN — never as
    fabricated measurements.
    """
    sel = np.asarray(selection, dtype=int).reshape(-1)
    if sel.shape != (problem.n,):
        raise ValueError(
            f"Selection length {sel.shape} does not match problem size {problem.n}."
        )

    risk_reduction = total_risk(problem, sel)
    risk_total = float(np.sum(problem.risk_scores))
    slacks = constraint_slack(problem, sel)

    budget_consumed = float(np.dot(problem.patch_costs, sel))
    hours_consumed = float(np.dot(problem.engineer_hours, sel))
    downtime_consumed = float(np.dot(problem.downtime_hours, sel))

    if kev_flags is None:
        kev = np.zeros(problem.n, dtype=int)
        n_kev_total = 0
        n_kev_selected = 0
        kev_available = False
    else:
        kev = np.asarray(kev_flags, dtype=int).reshape(-1)
        if kev.shape != (problem.n,):
            raise ValueError("kev_flags length must match the number of findings.")
        kev_available = True
        n_kev_total = int(kev.sum())
        n_kev_selected = int(np.dot(kev, sel))

    pct_kev = _safe_pct(n_kev_selected, n_kev_total) if kev_available else float("nan")

    record = empty_result_row()
    record.update(
        {
            "method": canonicalize_method(method),
            "seed": seed,
            "n_findings": int(problem.n),
            "n_selected": int(sel.sum()),
            "total_risk_reduction": float(risk_reduction),
            "total_risk_unmitigated": risk_total,
            "pct_risk_reduction": _safe_pct(risk_reduction, risk_total),
            "budget_limit": float(problem.budget),
            "budget_consumed": budget_consumed,
            "budget_remaining": float(slacks["budget_slack"]),
            "pct_budget_used": _safe_pct(budget_consumed, problem.budget),
            "hours_limit": float(problem.max_hours),
            "hours_consumed": hours_consumed,
            "hours_remaining": float(slacks["hours_slack"]),
            "pct_hours_used": _safe_pct(hours_consumed, problem.max_hours),
            "downtime_limit": float(problem.max_downtime),
            "downtime_consumed": downtime_consumed,
            "downtime_remaining": float(slacks["downtime_slack"]),
            "pct_downtime_used": _safe_pct(downtime_consumed, problem.max_downtime),
            "n_kev_total": n_kev_total if kev_available else float("nan"),
            "n_kev_selected": n_kev_selected if kev_available else float("nan"),
            "pct_kev_coverage": pct_kev,
            "feasible": bool(is_feasible(problem, sel)),
            "runtime_s": _as_float(runtime_s) if runtime_s is not None else float("nan"),
            "solver_runtime_s": _as_float(solver_runtime_s) if solver_runtime_s is not None else float("nan"),
            "solver_status": solver_status or "",
            "optimality_gap": _as_float(optimality_gap) if optimality_gap is not None else float("nan"),
            "solver_backend": solver_backend or "",
            "ran_successfully": True,
            "error_message": "",
        }
    )
    return record


def evaluate_solver_result(
    problem: OptimizationProblem,
    result: Any,
    *,
    kev_flags: Optional[np.ndarray] = None,
    runtime_s: Optional[float] = None,
    seed: Optional[int] = None,
) -> dict[str, Any]:
    """Evaluate a BaselineResult, MILPResult, or BPSOResult with shared definitions."""
    diag = extract_solver_diagnostics(result)
    method = getattr(result, "method", "")
    selection = getattr(result, "selection")
    if seed is None:
        metadata = getattr(result, "metadata", None) or {}
        if isinstance(metadata, Mapping) and "seed" in metadata:
            seed = metadata["seed"]
    record = evaluate_selection(
        problem,
        selection,
        method=method,
        kev_flags=kev_flags,
        runtime_s=runtime_s,
        solver_status=diag["solver_status"],
        optimality_gap=diag["optimality_gap"] if not math.isnan(diag["optimality_gap"]) else None,
        solver_runtime_s=diag["solver_runtime_s"] if not math.isnan(diag["solver_runtime_s"]) else None,
        solver_backend=diag["solver_backend"],
        seed=seed,
    )
    if diag["n_particles"] is not None:
        record["n_particles"] = diag["n_particles"]
    if diag["max_iter"] is not None:
        record["max_iter"] = diag["max_iter"]
    return record


def empty_result_row() -> dict[str, Any]:
    """A schema-complete row with empty / NaN values (used for failures and padding)."""
    row: dict[str, Any] = {col: "" for col in RESULT_COLUMNS}
    numeric_nan = {
        "seed",
        "weight_sample_id",
        "dataset_size",
        "resource_fraction",
        "n_findings",
        "n_selected",
        "total_risk_reduction",
        "total_risk_unmitigated",
        "pct_risk_reduction",
        "budget_limit",
        "budget_consumed",
        "budget_remaining",
        "pct_budget_used",
        "hours_limit",
        "hours_consumed",
        "hours_remaining",
        "pct_hours_used",
        "downtime_limit",
        "downtime_consumed",
        "downtime_remaining",
        "pct_downtime_used",
        "n_kev_total",
        "n_kev_selected",
        "pct_kev_coverage",
        "runtime_s",
        "solver_runtime_s",
        "optimality_gap",
        "n_particles",
        "max_iter",
        "milp_time_limit",
    }
    for col in numeric_nan:
        row[col] = float("nan")
    row["feasible"] = False
    row["ran_successfully"] = False
    row["error_message"] = ""
    return row


def failed_result_row(*, error_message: str, **overrides: Any) -> dict[str, Any]:
    """Record a failed experiment without substituting sample metric values."""
    row = empty_result_row()
    row["ran_successfully"] = False
    row["error_message"] = str(error_message)
    for key, value in overrides.items():
        if key in row:
            row[key] = value
    if "method" in overrides:
        row["method"] = canonicalize_method(str(overrides["method"]))
    return row


def records_to_dataframe(records: Sequence[Mapping[str, Any]]) -> pd.DataFrame:
    """Build a DataFrame with the contracted column order."""
    if not records:
        return pd.DataFrame(columns=list(RESULT_COLUMNS))
    df = pd.DataFrame(list(records))
    for col in RESULT_COLUMNS:
        if col not in df.columns:
            df[col] = pd.NA
    return df[list(RESULT_COLUMNS)]
