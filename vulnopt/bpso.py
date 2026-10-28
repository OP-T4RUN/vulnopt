"""
Binary Particle Swarm Optimization (BPSO) for VulnOpt.

Solves the multi-dimensional binary knapsack problem (MKP) using a
population-based metaheuristic.  BPSO is particularly suited to:
  - Large instances (n > 2000) where MILP/CBC runtime becomes prohibitive.
  - Obtaining good-quality feasible solutions quickly.

Algorithm Summary
-----------------
Each particle is a binary vector x in {0,1}^n representing a candidate
remediation selection.  The particle's velocity v in R^n is mapped to
selection probabilities via the sigmoid function.

Update rules (Kennedy & Eberhart, 1997 – binary variant):
    v_i(t+1) = w * v_i(t)
               + c1 * r1 * (pbest_i - x_i(t))
               + c2 * r2 * (gbest_i - x_i(t))

    p_flip   = sigmoid(v_i(t+1))
    x_i(t+1) = 1  if rand() < p_flip  else  0

Feasibility Repair
------------------
After each position update, if the new position violates any constraint,
a greedy repair operator removes the least cost-effective item (lowest
R_i / C_i ratio) until feasibility is restored.  If the repaired
position has a higher total risk score than the current personal best,
it replaces it.

Parameters (defaults)
----------------------
n_particles  : 30
max_iter     : 200
w            : 0.7298  (constriction factor – Clerc & Kennedy, 2002)
c1           : 1.4962  (cognitive component)
c2           : 1.4962  (social component)
v_max        : 6.0     (clamped velocity for numerical stability)
seed         : 42
"""

import time
from typing import Any, Dict, List, NamedTuple, Optional, Tuple

import numpy as np

from vulnopt.constraints import (
    OptimizationProblem,
    constraint_slack,
    is_feasible,
    total_risk,
)


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

class BPSOResult(NamedTuple):
    """
    Structured result returned by the BPSO solver.

    Attributes
    ----------
    method : str
    selection : np.ndarray, shape (n,)  – binary
    total_risk_score : float
    n_selected : int
    budget_used : float
    hours_used : float
    downtime_used : float
    feasible : bool
    slacks : dict
    solve_time_s : float
    n_iterations : int
    convergence_history : list[float]
        Best global risk score at each iteration.
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
    solve_time_s: float
    n_iterations: int
    convergence_history: List[float]
    metadata: dict


# ---------------------------------------------------------------------------
# Sigmoid helper
# ---------------------------------------------------------------------------

def _sigmoid(v: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid: 1 / (1 + exp(-v))."""
    return np.where(
        v >= 0,
        1.0 / (1.0 + np.exp(-v)),
        np.exp(v) / (1.0 + np.exp(v)),
    )


# ---------------------------------------------------------------------------
# Feasibility repair operator
# ---------------------------------------------------------------------------

def _repair(
    problem: OptimizationProblem,
    position: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Greedy repair: iteratively remove the item with the lowest risk/cost ratio
    until all constraints are satisfied.

    If patch_cost is zero for an item, use risk_score alone as the ratio
    to avoid division by zero.

    Parameters
    ----------
    problem  : OptimizationProblem
    position : np.ndarray, shape (n,), dtype int
        Mutable binary position vector (will be modified in-place copy).
    rng      : np.random.Generator (unused currently, kept for extensibility)

    Returns
    -------
    repaired : np.ndarray, shape (n,), dtype int
    """
    pos = position.copy()

    # Efficiency ratio for repair: benefit per unit cost
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(
            problem.patch_costs > 0,
            problem.risk_scores / problem.patch_costs,
            problem.risk_scores,
        )

    max_iters = int(pos.sum()) + 1
    for _ in range(max_iters):
        if is_feasible(problem, pos):
            break
        # Find selected items and remove the worst by ratio
        selected_indices = np.where(pos == 1)[0]
        if len(selected_indices) == 0:
            break
        worst = selected_indices[np.argmin(ratio[selected_indices])]
        pos[worst] = 0

    return pos


# ---------------------------------------------------------------------------
# Main BPSO solver
# ---------------------------------------------------------------------------

def solve_bpso(
    problem: OptimizationProblem,
    n_particles: int = 30,
    max_iter: int = 200,
    w: float = 0.7298,
    c1: float = 1.4962,
    c2: float = 1.4962,
    v_max: float = 6.0,
    seed: int = 42,
    verbose: bool = False,
) -> BPSOResult:
    """
    Solve the VulnOpt 3D binary knapsack problem using BPSO.

    Parameters
    ----------
    problem     : OptimizationProblem
    n_particles : int
        Swarm size. Default 30.
    max_iter    : int
        Maximum number of iterations. Default 200.
    w           : float
        Inertia weight (constriction factor). Default 0.7298.
    c1          : float
        Cognitive acceleration coefficient. Default 1.4962.
    c2          : float
        Social acceleration coefficient. Default 1.4962.
    v_max       : float
        Maximum absolute velocity (clamped). Default 6.0.
    seed        : int
        Random seed for reproducibility. Default 42.
    verbose     : bool
        If True, print iteration progress every 10 iterations.

    Returns
    -------
    BPSOResult
    """
    rng = np.random.default_rng(seed)
    n = problem.n
    t_start = time.perf_counter()

    # ------------------------------------------------------------------ #
    # 1. Initialise particles
    # ------------------------------------------------------------------ #
    # Initialise velocities in [-v_max/2, v_max/2]
    velocities = rng.uniform(-v_max / 2, v_max / 2, size=(n_particles, n))

    # Initialise positions: random binary, then repair for feasibility
    raw_probs = _sigmoid(velocities)
    positions = (rng.random(size=(n_particles, n)) < raw_probs).astype(int)
    for p_idx in range(n_particles):
        positions[p_idx] = _repair(problem, positions[p_idx], rng)

    # Personal bests
    pbest_pos    = positions.copy()
    pbest_scores = np.array(
        [total_risk(problem, positions[p]) for p in range(n_particles)]
    )

    # Global best
    gbest_idx   = int(np.argmax(pbest_scores))
    gbest_pos   = pbest_pos[gbest_idx].copy()
    gbest_score = float(pbest_scores[gbest_idx])

    convergence_history: List[float] = [gbest_score]

    # ------------------------------------------------------------------ #
    # 2. Main iteration loop
    # ------------------------------------------------------------------ #
    for iteration in range(max_iter):
        r1 = rng.random(size=(n_particles, n))
        r2 = rng.random(size=(n_particles, n))

        # Velocity update
        velocities = (
            w * velocities
            + c1 * r1 * (pbest_pos - positions)
            + c2 * r2 * (gbest_pos - positions)
        )

        # Clamp velocities to avoid sigmoid saturation
        velocities = np.clip(velocities, -v_max, v_max)

        # Position update via sigmoid threshold
        flip_probs   = _sigmoid(velocities)
        new_positions = (rng.random(size=(n_particles, n)) < flip_probs).astype(int)

        # Repair each particle for feasibility
        for p_idx in range(n_particles):
            new_positions[p_idx] = _repair(problem, new_positions[p_idx], rng)

        positions = new_positions

        # Update personal and global bests
        scores = np.array(
            [total_risk(problem, positions[p]) for p in range(n_particles)]
        )
        improved_mask = scores > pbest_scores
        pbest_pos[improved_mask]    = positions[improved_mask].copy()
        pbest_scores[improved_mask] = scores[improved_mask]

        current_best_idx = int(np.argmax(pbest_scores))
        if pbest_scores[current_best_idx] > gbest_score:
            gbest_score = float(pbest_scores[current_best_idx])
            gbest_pos   = pbest_pos[current_best_idx].copy()

        convergence_history.append(gbest_score)

        if verbose and (iteration % 10 == 0 or iteration == max_iter - 1):
            print(
                f"  BPSO iter {iteration+1:4d}/{max_iter}  "
                f"best_score={gbest_score:.4f}  "
                f"selected={int(gbest_pos.sum())}"
            )

    # ------------------------------------------------------------------ #
    # 3. Build result
    # ------------------------------------------------------------------ #
    solve_time = time.perf_counter() - t_start
    selection  = gbest_pos.astype(int)
    slacks     = constraint_slack(problem, selection)

    return BPSOResult(
        method="BPSO",
        selection=selection,
        total_risk_score=total_risk(problem, selection),
        n_selected=int(selection.sum()),
        budget_used=float(np.dot(problem.patch_costs,    selection)),
        hours_used=float(np.dot(problem.engineer_hours,  selection)),
        downtime_used=float(np.dot(problem.downtime_hours, selection)),
        feasible=is_feasible(problem, selection),
        slacks=slacks,
        solve_time_s=round(solve_time, 4),
        n_iterations=max_iter,
        convergence_history=convergence_history,
        metadata={
            "n_particles": n_particles,
            "max_iter":    max_iter,
            "w":           w,
            "c1":          c1,
            "c2":          c2,
            "v_max":       v_max,
            "seed":        seed,
        },
    )


# ---------------------------------------------------------------------------
# Convenience helpers
# ---------------------------------------------------------------------------

def bpso_result_to_series(result: BPSOResult) -> "pd.Series":
    """Return a named pandas Series summarising the BPSO result."""
    import pandas as pd  # local import to avoid circular at module level

    return pd.Series({
        "method":           result.method,
        "total_risk_score": round(result.total_risk_score, 2),
        "n_selected":       result.n_selected,
        "budget_used":      round(result.budget_used, 2),
        "hours_used":       round(result.hours_used, 2),
        "downtime_used":    round(result.downtime_used, 2),
        "feasible":         result.feasible,
        "solve_time_s":     result.solve_time_s,
        "n_iterations":     result.n_iterations,
        "final_gbest":      round(result.total_risk_score, 4),
    })
