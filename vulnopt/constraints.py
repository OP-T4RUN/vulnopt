"""
Constraint definitions for VulnOpt optimization problems.

Encapsulates the multi-dimensional knapsack problem (MKP) constraints:
  - Budget constraint:          sum(C_i * x_i) <= B
  - Engineer-hours constraint:  sum(T_i * x_i) <= H
  - Downtime constraint:        sum(D_i * x_i) <= W

All feasibility checks are centralised here so that every optimizer
(Baseline, MILP, BPSO) uses the same contract.
"""

from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Problem instance
# ---------------------------------------------------------------------------

@dataclass
class OptimizationProblem:
    """
    Immutable description of a VulnOpt remediation selection problem.

    Attributes
    ----------
    risk_scores : np.ndarray, shape (n,)
        R_i values (0–100) giving the benefit of remediating finding i.
    patch_costs : np.ndarray, shape (n,)
        C_i – monetary cost ($) of remediating finding i.
    engineer_hours : np.ndarray, shape (n,)
        T_i – engineer-hours required for finding i.
    downtime_hours : np.ndarray, shape (n,)
        D_i – maintenance downtime (hours) required for finding i.
    budget : float
        B – total available remediation budget ($).
    max_hours : float
        H – total available engineer-hours.
    max_downtime : float
        W – total available maintenance downtime (hours).
    finding_ids : list[str] or None
        Optional human-readable labels, one per finding.
    """

    risk_scores: np.ndarray
    patch_costs: np.ndarray
    engineer_hours: np.ndarray
    downtime_hours: np.ndarray
    budget: float
    max_hours: float
    max_downtime: float
    finding_ids: Optional[list] = field(default=None)

    # ------------------------------------------------------------------ #

    def __post_init__(self) -> None:
        n = len(self.risk_scores)
        for arr_name in ("patch_costs", "engineer_hours", "downtime_hours"):
            arr = getattr(self, arr_name)
            if len(arr) != n:
                raise ValueError(
                    f"Length of '{arr_name}' ({len(arr)}) must match "
                    f"'risk_scores' ({n})."
                )
            if np.any(arr < 0):
                raise ValueError(
                    f"All values in '{arr_name}' must be non-negative."
                )
        if not np.all(np.isfinite(self.risk_scores)):
            raise ValueError("'risk_scores' must contain only finite values.")
        for cap_name in ("budget", "max_hours", "max_downtime"):
            val = getattr(self, cap_name)
            if val < 0:
                raise ValueError(f"Capacity '{cap_name}' must be non-negative.")

        if self.finding_ids is None:
            self.finding_ids = [f"FND-{i+1:05d}" for i in range(n)]
        elif len(self.finding_ids) != n:
            raise ValueError(
                f"'finding_ids' length ({len(self.finding_ids)}) must match "
                f"number of findings ({n})."
            )

    # ------------------------------------------------------------------ #

    @property
    def n(self) -> int:
        """Number of candidate remediation findings."""
        return len(self.risk_scores)

    # ------------------------------------------------------------------ #

    @classmethod
    def from_dataframe(
        cls,
        df: pd.DataFrame,
        budget: float,
        max_hours: float,
        max_downtime: float,
    ) -> "OptimizationProblem":
        """
        Construct an OptimizationProblem from a scored findings DataFrame.

        Required columns
        ----------------
        risk_score, patch_cost, engineer_hours, downtime_hours

        Optional column
        ---------------
        finding_id  (used as labels when present)
        """
        required = ["risk_score", "patch_cost", "engineer_hours", "downtime_hours"]
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"DataFrame is missing required columns: {missing}")

        fids = df["finding_id"].tolist() if "finding_id" in df.columns else None

        return cls(
            risk_scores=df["risk_score"].to_numpy(dtype=float),
            patch_costs=df["patch_cost"].to_numpy(dtype=float),
            engineer_hours=df["engineer_hours"].to_numpy(dtype=float),
            downtime_hours=df["downtime_hours"].to_numpy(dtype=float),
            budget=float(budget),
            max_hours=float(max_hours),
            max_downtime=float(max_downtime),
            finding_ids=fids,
        )


# ---------------------------------------------------------------------------
# Feasibility helpers
# ---------------------------------------------------------------------------

def is_feasible(problem: OptimizationProblem, selection: np.ndarray) -> bool:
    """
    Return *True* iff the binary selection vector satisfies all three
    capacity constraints simultaneously.

    Parameters
    ----------
    problem   : OptimizationProblem
    selection : np.ndarray, shape (n,), dtype int or bool
        Binary indicator: 1 -> selected for remediation, 0 -> skipped.
    """
    sel = np.asarray(selection, dtype=float)
    if sel.shape != (problem.n,):
        raise ValueError(
            f"Selection length {sel.shape} does not match problem size {problem.n}."
        )
    budget_ok   = float(np.dot(problem.patch_costs,   sel)) <= problem.budget
    hours_ok    = float(np.dot(problem.engineer_hours, sel)) <= problem.max_hours
    downtime_ok = float(np.dot(problem.downtime_hours, sel)) <= problem.max_downtime
    return budget_ok and hours_ok and downtime_ok


def constraint_slack(
    problem: OptimizationProblem, selection: np.ndarray
) -> dict:
    """
    Return a dictionary of remaining capacity (slack) for each constraint.

    Negative slack means the constraint is violated.
    """
    sel = np.asarray(selection, dtype=float)
    return {
        "budget_slack":   problem.budget      - float(np.dot(problem.patch_costs,   sel)),
        "hours_slack":    problem.max_hours   - float(np.dot(problem.engineer_hours, sel)),
        "downtime_slack": problem.max_downtime - float(np.dot(problem.downtime_hours, sel)),
    }


def total_risk(problem: OptimizationProblem, selection: np.ndarray) -> float:
    """Total risk-reduction score for the given selection vector."""
    return float(np.dot(problem.risk_scores, np.asarray(selection, dtype=float)))


def compute_default_budgets(
    problem: OptimizationProblem, budget_fraction: float = 0.40
) -> dict:
    """
    Derive sensible default capacity limits as fractions of the totals.

    Useful for experiments that do not have explicit budget/capacity inputs.

    Parameters
    ----------
    budget_fraction : float
        Fraction of total resource to use as the default cap. Default 0.40.
    """
    if not 0 < budget_fraction <= 1.0:
        raise ValueError("budget_fraction must be in (0, 1].")
    return {
        "budget":       float(np.sum(problem.patch_costs))   * budget_fraction,
        "max_hours":    float(np.sum(problem.engineer_hours)) * budget_fraction,
        "max_downtime": float(np.sum(problem.downtime_hours)) * budget_fraction,
    }
