"""
Reproducible experiment runner for VulnOpt.

Calls the existing baseline, MILP, and BPSO implementations.  Does not
reimplement selection logic.

Resource limits
---------------
For a scored-findings table, each resource capacity is a fraction of that
resource's *total* across all findings:

    B = resource_fraction * sum(patch_cost)
    H = resource_fraction * sum(engineer_hours)
    W = resource_fraction * sum(downtime_hours)

Fractions (applied independently to B, H, and W via
``constraints.compute_default_budgets``):

    tight   = 0.30
    medium  = 0.50
    relaxed = 0.70

Every method on a given (dataset, scenario, weights, EPSS policy) tuple
receives the same OptimizationProblem.

EPSS handling
-------------
The public-data pipeline imputes missing EPSS as 0.0 and flags
``epss_missing``.  Experiments compare:

    zero_imputation    — missing EPSS set to 0.0 (pipeline default)
    median_imputation  — missing EPSS set to the median of observed EPSS
    exclusion          — drop findings with epss_missing before scoring

If a snapshot has zero missing EPSS values, the three policies produce
identical candidate sets; that identity is reported rather than faked.

BPSO
----
Stochastic comparisons use at least 20 seeds (0..19) on the medium
resource scenario.  numpy Generator seeding inside ``solve_bpso`` makes
those runs reproducible.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from vulnopt.baselines import cvss_greedy, risk_greedy
from vulnopt.bpso import solve_bpso
from vulnopt.constraints import OptimizationProblem, compute_default_budgets
from vulnopt.metrics import (
    RESULT_COLUMNS,
    empty_result_row,
    evaluate_solver_result,
    failed_result_row,
    kev_flags_from_frame,
    records_to_dataframe,
)
from vulnopt.milp_solver import solve_milp
from vulnopt.risk_model import (
    ASSET_FOCUSED_WEIGHTS,
    DEFAULT_WEIGHTS,
    THREAT_FOCUSED_WEIGHTS,
    calculate_risk_scores,
    generate_dirichlet_weight_samples,
)


ROOT_DIR = Path(__file__).resolve().parent.parent

RESOURCE_SCENARIOS: Dict[str, float] = {
    "tight": 0.30,
    "medium": 0.50,
    "relaxed": 0.70,
}

NAMED_WEIGHTS: Dict[str, Dict[str, float]] = {
    "default": DEFAULT_WEIGHTS,
    "threat_focused": THREAT_FOCUSED_WEIGHTS,
    "asset_focused": ASSET_FOCUSED_WEIGHTS,
}

DATASET_FILES: Dict[int, str] = {
    100: "data/synthetic/scored_findings_100.csv",
    1000: "data/synthetic/scored_findings_1000.csv",
    5000: "data/synthetic/scored_findings_5000.csv",
}

# BPSO swarm size / iterations scaled by instance size so large benchmarks
# finish in bounded wall-clock time.  Documented in experiment_config.json.
BPSO_PARAMS_BY_SIZE: Dict[int, Dict[str, int]] = {
    100: {"n_particles": 30, "max_iter": 200},
    1000: {"n_particles": 25, "max_iter": 100},
    5000: {"n_particles": 20, "max_iter": 50},
}

MILP_TIME_LIMIT_BY_SIZE: Dict[int, int] = {
    100: 30,
    1000: 90,
    5000: 180,
}

METHODS: tuple[str, ...] = ("CVSS-Greedy", "Risk-Greedy", "MILP", "BPSO")
EPSS_STRATEGIES: tuple[str, ...] = (
    "zero_imputation",
    "median_imputation",
    "exclusion",
)


@dataclass
class ExperimentConfig:
    """Tunable knobs for the experiment grid.  Defaults match the assignment."""

    dataset_sizes: List[int] = field(default_factory=lambda: [100, 1000, 5000])
    resource_scenarios: List[str] = field(default_factory=lambda: ["tight", "medium", "relaxed"])
    methods: List[str] = field(default_factory=lambda: list(METHODS))
    bpso_seeds: List[int] = field(default_factory=lambda: list(range(20)))
    dirichlet_samples: int = 100
    dirichlet_seed: int = 42
    representative_bpso_seed: int = 42
    milp_time_limits: Dict[int, int] = field(default_factory=lambda: dict(MILP_TIME_LIMIT_BY_SIZE))
    bpso_params: Dict[int, Dict[str, int]] = field(default_factory=lambda: {k: dict(v) for k, v in BPSO_PARAMS_BY_SIZE.items()})
    run_method_comparison: bool = True
    run_bpso_multiseed: bool = True
    run_named_weights: bool = True
    run_dirichlet: bool = True
    run_epss: bool = True
    # Dirichlet and EPSS grids are expensive; default them to the small benchmark.
    sensitivity_dataset_size: int = 100
    epss_dataset_sizes: List[int] = field(default_factory=lambda: [100, 1000])
    data_root: str = ""
    dataset_files: Dict[int, str] = field(default_factory=lambda: dict(DATASET_FILES))


def default_config() -> ExperimentConfig:
    return ExperimentConfig()


def quick_config() -> ExperimentConfig:
    """Small grid used by unit tests — still exercises every experiment family."""
    return ExperimentConfig(
        dataset_sizes=[100],
        resource_scenarios=["medium"],
        bpso_seeds=[0, 1],
        dirichlet_samples=3,
        milp_time_limits={100: 15, 1000: 15, 5000: 15},
        bpso_params={
            100: {"n_particles": 8, "max_iter": 12},
            1000: {"n_particles": 8, "max_iter": 12},
            5000: {"n_particles": 8, "max_iter": 12},
        },
        epss_dataset_sizes=[100],
        sensitivity_dataset_size=100,
    )


def _project_root(config: ExperimentConfig) -> Path:
    if config.data_root:
        return Path(config.data_root)
    return ROOT_DIR


def load_scored_findings(dataset_size: int, config: ExperimentConfig) -> pd.DataFrame:
    rel = config.dataset_files.get(dataset_size)
    if rel is None:
        raise FileNotFoundError(f"No benchmark dataset registered for size {dataset_size}.")
    path = Path(rel)
    if not path.is_absolute():
        path = _project_root(config) / rel
    if not path.exists():
        raise FileNotFoundError(f"Benchmark file not found: {path}")
    return pd.read_csv(path)


def dataset_path_str(dataset_size: int, config: ExperimentConfig) -> str:
    rel = config.dataset_files.get(dataset_size, "")
    return str(rel).replace("\\", "/")


def apply_epss_strategy(df: pd.DataFrame, strategy: str) -> pd.DataFrame:
    """
    Apply an EPSS missing-data policy, then recompute R_i with default weights.

    Observed (non-missing) EPSS values are never altered.  Missingness is
    taken from ``epss_missing`` when present, otherwise from NA in ``epss``.
    """
    if strategy not in EPSS_STRATEGIES:
        raise ValueError(f"Unknown EPSS strategy: {strategy}")

    out = df.copy()
    if "epss_missing" in out.columns:
        missing = out["epss_missing"].astype(bool)
    elif "epss" in out.columns:
        missing = out["epss"].isna()
    else:
        missing = pd.Series(False, index=out.index)

    observed = pd.to_numeric(out.loc[~missing, "epss"], errors="coerce") if "epss" in out.columns else pd.Series(dtype=float)

    if strategy == "exclusion":
        out = out.loc[~missing].copy().reset_index(drop=True)
    elif strategy == "zero_imputation":
        if "epss" in out.columns:
            out.loc[missing, "epss"] = 0.0
        if "epss_score" in out.columns:
            out.loc[missing, "epss_score"] = 0.0
    elif strategy == "median_imputation":
        median = float(observed.median()) if observed.notna().any() else 0.0
        if "epss" in out.columns:
            out.loc[missing, "epss"] = median
        if "epss_score" in out.columns:
            out.loc[missing, "epss_score"] = median

    return calculate_risk_scores(out, weights=DEFAULT_WEIGHTS)


def rescore_with_weights(df: pd.DataFrame, weights: Dict[str, float]) -> pd.DataFrame:
    return calculate_risk_scores(df.copy(), weights=weights)


def build_problem(df: pd.DataFrame, resource_fraction: float) -> OptimizationProblem:
    """Construct an OptimizationProblem using Student 3's budget helper."""
    dummy = OptimizationProblem.from_dataframe(
        df, budget=1.0, max_hours=1.0, max_downtime=1.0
    )
    caps = compute_default_budgets(dummy, budget_fraction=resource_fraction)
    return OptimizationProblem.from_dataframe(
        df,
        budget=caps["budget"],
        max_hours=caps["max_hours"],
        max_downtime=caps["max_downtime"],
    )


def _cvss_vector(df: pd.DataFrame, n: int) -> np.ndarray:
    if "cvss_base" not in df.columns:
        raise ValueError("Findings table is missing cvss_base; cannot run CVSS-Greedy.")
    return df["cvss_base"].to_numpy(dtype=float)


def run_solver(
    method: str,
    problem: OptimizationProblem,
    df: pd.DataFrame,
    *,
    seed: int,
    bpso_params: Dict[str, int],
    milp_time_limit: int,
) -> Any:
    """Dispatch to an existing optimizer.  Raises on unknown methods."""
    if method == "CVSS-Greedy":
        return cvss_greedy(problem, _cvss_vector(df, problem.n))
    if method == "Risk-Greedy":
        return risk_greedy(problem)
    if method == "MILP":
        return solve_milp(problem, time_limit=milp_time_limit, verbose=False)
    if method == "BPSO":
        return solve_bpso(
            problem,
            n_particles=int(bpso_params.get("n_particles", 30)),
            max_iter=int(bpso_params.get("max_iter", 200)),
            seed=int(seed),
            verbose=False,
        )
    raise ValueError(f"Unknown method: {method}")


def _annotate(
    record: dict[str, Any],
    *,
    experiment_id: str,
    experiment_family: str,
    dataset_size: int,
    dataset_path: str,
    resource_scenario: str,
    resource_fraction: float,
    weight_scenario: str,
    weight_sample_id: Any,
    epss_strategy: str,
    milp_time_limit: Any,
    bpso_params: Optional[Dict[str, int]] = None,
) -> dict[str, Any]:
    record["experiment_id"] = experiment_id
    record["experiment_family"] = experiment_family
    record["dataset_size"] = int(dataset_size)
    record["dataset_path"] = dataset_path
    record["resource_scenario"] = resource_scenario
    record["resource_fraction"] = float(resource_fraction)
    record["weight_scenario"] = weight_scenario
    record["weight_sample_id"] = weight_sample_id
    record["epss_strategy"] = epss_strategy
    record["milp_time_limit"] = milp_time_limit
    if bpso_params and record.get("method") == "BPSO":
        if pd.isna(record.get("n_particles")):
            record["n_particles"] = bpso_params.get("n_particles")
        if pd.isna(record.get("max_iter")):
            record["max_iter"] = bpso_params.get("max_iter")
    return record


def evaluate_method(
    method: str,
    problem: OptimizationProblem,
    df: pd.DataFrame,
    *,
    seed: Optional[int],
    bpso_params: Dict[str, int],
    milp_time_limit: int,
    context: Dict[str, Any],
) -> dict[str, Any]:
    kev = kev_flags_from_frame(df, problem.n)
    run_seed = context.get("seed", seed)
    print(f"  running {context.get('experiment_id', method)}", flush=True)
    try:
        t0 = time.perf_counter()
        result = run_solver(
            method,
            problem,
            df,
            seed=int(run_seed if run_seed is not None else 42),
            bpso_params=bpso_params,
            milp_time_limit=milp_time_limit,
        )
        runtime = time.perf_counter() - t0
        record = evaluate_solver_result(
            problem,
            result,
            kev_flags=kev,
            runtime_s=runtime,
            seed=run_seed if method == "BPSO" else None,
        )
        record["milp_time_limit"] = milp_time_limit if method == "MILP" else float("nan")
        if method == "BPSO":
            record["n_particles"] = bpso_params.get("n_particles")
            record["max_iter"] = bpso_params.get("max_iter")
            record["seed"] = int(run_seed if run_seed is not None else 42)
    except Exception as exc:  # noqa: BLE001 — experiment isolation
        record = failed_result_row(
            error_message=f"{type(exc).__name__}: {exc}",
            method=method,
            seed=run_seed if method == "BPSO" else float("nan"),
        )
        record["n_findings"] = problem.n
    annotate_kwargs = {
        k: v for k, v in context.items() if k not in {"seed", "milp_time_limit"}
    }
    return _annotate(
        record,
        **annotate_kwargs,
        bpso_params=bpso_params,
        milp_time_limit=milp_time_limit if method == "MILP" else float("nan"),
    )


def _bpso_params_for(size: int, config: ExperimentConfig) -> Dict[str, int]:
    if size in config.bpso_params:
        return dict(config.bpso_params[size])
    # Fall back to the nearest registered size.
    if size <= 100:
        return dict(config.bpso_params.get(100, BPSO_PARAMS_BY_SIZE[100]))
    if size <= 1000:
        return dict(config.bpso_params.get(1000, BPSO_PARAMS_BY_SIZE[1000]))
    return dict(config.bpso_params.get(5000, BPSO_PARAMS_BY_SIZE[5000]))


def _milp_limit_for(size: int, config: ExperimentConfig) -> int:
    if size in config.milp_time_limits:
        return int(config.milp_time_limits[size])
    return int(config.milp_time_limits.get(5000, 180))


def run_method_comparison(config: ExperimentConfig) -> List[dict[str, Any]]:
    rows: List[dict[str, Any]] = []
    for size in config.dataset_sizes:
        try:
            df = load_scored_findings(size, config)
        except Exception as exc:  # noqa: BLE001
            for scenario in config.resource_scenarios:
                for method in config.methods:
                    rows.append(
                        failed_result_row(
                            error_message=f"Failed to load dataset {size}: {exc}",
                            method=method,
                            dataset_size=size,
                        )
                    )
            continue
        path = dataset_path_str(size, config)
        for scenario in config.resource_scenarios:
            frac = RESOURCE_SCENARIOS[scenario]
            problem = build_problem(df, frac)
            bpso_params = _bpso_params_for(size, config)
            milp_limit = _milp_limit_for(size, config)
            for method in config.methods:
                eid = f"cmp_{size}_{scenario}_{method}"
                ctx = dict(
                    experiment_id=eid,
                    experiment_family="method_comparison",
                    dataset_size=size,
                    dataset_path=path,
                    resource_scenario=scenario,
                    resource_fraction=frac,
                    weight_scenario="default",
                    weight_sample_id=float("nan"),
                    epss_strategy="zero_imputation",
                    milp_time_limit=milp_limit,
                    seed=config.representative_bpso_seed,
                )
                rows.append(
                    evaluate_method(
                        method,
                        problem,
                        df,
                        seed=config.representative_bpso_seed,
                        bpso_params=bpso_params,
                        milp_time_limit=milp_limit,
                        context=ctx,
                    )
                )
    return rows


def run_bpso_multiseed(config: ExperimentConfig) -> List[dict[str, Any]]:
    rows: List[dict[str, Any]] = []
    if "BPSO" not in config.methods:
        return rows
    scenario = "medium"
    frac = RESOURCE_SCENARIOS[scenario]
    for size in config.dataset_sizes:
        try:
            df = load_scored_findings(size, config)
            problem = build_problem(df, frac)
        except Exception as exc:  # noqa: BLE001
            for seed in config.bpso_seeds:
                rows.append(
                    failed_result_row(
                        error_message=f"Failed to load dataset {size}: {exc}",
                        method="BPSO",
                        dataset_size=size,
                        seed=seed,
                    )
                )
            continue
        path = dataset_path_str(size, config)
        bpso_params = _bpso_params_for(size, config)
        milp_limit = _milp_limit_for(size, config)
        for seed in config.bpso_seeds:
            eid = f"bpso_{size}_{scenario}_seed{seed}"
            ctx = dict(
                experiment_id=eid,
                experiment_family="bpso_multiseed",
                dataset_size=size,
                dataset_path=path,
                resource_scenario=scenario,
                resource_fraction=frac,
                weight_scenario="default",
                weight_sample_id=float("nan"),
                epss_strategy="zero_imputation",
                milp_time_limit=milp_limit,
                seed=seed,
            )
            rows.append(
                evaluate_method(
                    "BPSO",
                    problem,
                    df,
                    seed=seed,
                    bpso_params=bpso_params,
                    milp_time_limit=milp_limit,
                    context=ctx,
                )
            )
    return rows


def summarize_bpso_seeds(records: Sequence[dict[str, Any]]) -> pd.DataFrame:
    """Mean / std / min / max of BPSO metrics grouped by dataset size."""
    df = records_to_dataframe(records)
    if df.empty:
        return pd.DataFrame(
            columns=[
                "experiment_family",
                "dataset_size",
                "resource_scenario",
                "method",
                "n_runs",
                "n_successful",
                "n_failed",
                "mean_total_risk_reduction",
                "std_total_risk_reduction",
                "min_total_risk_reduction",
                "max_total_risk_reduction",
                "mean_pct_risk_reduction",
                "std_pct_risk_reduction",
                "mean_n_selected",
                "std_n_selected",
                "mean_runtime_s",
                "std_runtime_s",
                "mean_pct_kev_coverage",
                "std_pct_kev_coverage",
                "n_feasible",
            ]
        )
    ok = df[df["ran_successfully"] == True]  # noqa: E712
    grouped = []
    for (size, scenario), g in ok.groupby(["dataset_size", "resource_scenario"], dropna=False):
        n_all = len(df[(df["dataset_size"] == size) & (df["resource_scenario"] == scenario)])
        grouped.append(
            {
                "experiment_family": "bpso_multiseed_summary",
                "dataset_size": size,
                "resource_scenario": scenario,
                "method": "BPSO",
                "n_runs": n_all,
                "n_successful": int(len(g)),
                "n_failed": int(n_all - len(g)),
                "mean_total_risk_reduction": float(g["total_risk_reduction"].mean()),
                "std_total_risk_reduction": float(g["total_risk_reduction"].std(ddof=1)) if len(g) > 1 else 0.0,
                "min_total_risk_reduction": float(g["total_risk_reduction"].min()),
                "max_total_risk_reduction": float(g["total_risk_reduction"].max()),
                "mean_pct_risk_reduction": float(g["pct_risk_reduction"].mean()),
                "std_pct_risk_reduction": float(g["pct_risk_reduction"].std(ddof=1)) if len(g) > 1 else 0.0,
                "mean_n_selected": float(g["n_selected"].mean()),
                "std_n_selected": float(g["n_selected"].std(ddof=1)) if len(g) > 1 else 0.0,
                "mean_runtime_s": float(g["runtime_s"].mean()),
                "std_runtime_s": float(g["runtime_s"].std(ddof=1)) if len(g) > 1 else 0.0,
                "mean_pct_kev_coverage": float(g["pct_kev_coverage"].mean()),
                "std_pct_kev_coverage": float(g["pct_kev_coverage"].std(ddof=1)) if len(g) > 1 else 0.0,
                "n_feasible": int(g["feasible"].astype(bool).sum()),
            }
        )
    return pd.DataFrame(grouped)


def run_named_weight_sensitivity(config: ExperimentConfig) -> List[dict[str, Any]]:
    rows: List[dict[str, Any]] = []
    size = config.sensitivity_dataset_size
    scenario = "medium"
    frac = RESOURCE_SCENARIOS[scenario]
    try:
        base = load_scored_findings(size, config)
    except Exception as exc:  # noqa: BLE001
        for name in NAMED_WEIGHTS:
            for method in config.methods:
                rows.append(
                    failed_result_row(
                        error_message=f"Failed to load dataset {size}: {exc}",
                        method=method,
                        dataset_size=size,
                    )
                )
        return rows
    path = dataset_path_str(size, config)
    bpso_params = _bpso_params_for(size, config)
    milp_limit = _milp_limit_for(size, config)
    for weight_name, weights in NAMED_WEIGHTS.items():
        df = rescore_with_weights(base, weights)
        problem = build_problem(df, frac)
        for method in config.methods:
            eid = f"w_{weight_name}_{size}_{method}"
            ctx = dict(
                experiment_id=eid,
                experiment_family="sensitivity_named",
                dataset_size=size,
                dataset_path=path,
                resource_scenario=scenario,
                resource_fraction=frac,
                weight_scenario=weight_name,
                weight_sample_id=float("nan"),
                epss_strategy="zero_imputation",
                milp_time_limit=milp_limit,
                seed=config.representative_bpso_seed,
            )
            rows.append(
                evaluate_method(
                    method,
                    problem,
                    df,
                    seed=config.representative_bpso_seed,
                    bpso_params=bpso_params,
                    milp_time_limit=milp_limit,
                    context=ctx,
                )
            )
    return rows


def run_dirichlet_sensitivity(config: ExperimentConfig) -> List[dict[str, Any]]:
    rows: List[dict[str, Any]] = []
    size = config.sensitivity_dataset_size
    scenario = "medium"
    frac = RESOURCE_SCENARIOS[scenario]
    try:
        base = load_scored_findings(size, config)
        samples = generate_dirichlet_weight_samples(
            n_samples=config.dirichlet_samples, seed=config.dirichlet_seed
        )
    except Exception as exc:  # noqa: BLE001
        rows.append(
            failed_result_row(
                error_message=f"Dirichlet sensitivity setup failed: {exc}",
                method="Risk-Greedy",
                dataset_size=size,
            )
        )
        return rows
    path = dataset_path_str(size, config)
    bpso_params = _bpso_params_for(size, config)
    milp_limit = _milp_limit_for(size, config)
    for idx, weights in enumerate(samples):
        df = rescore_with_weights(base, weights)
        problem = build_problem(df, frac)
        for method in config.methods:
            eid = f"dirichlet_{idx:03d}_{size}_{method}"
            ctx = dict(
                experiment_id=eid,
                experiment_family="sensitivity_dirichlet",
                dataset_size=size,
                dataset_path=path,
                resource_scenario=scenario,
                resource_fraction=frac,
                weight_scenario="dirichlet",
                weight_sample_id=idx,
                epss_strategy="zero_imputation",
                milp_time_limit=milp_limit,
                seed=config.representative_bpso_seed,
            )
            rows.append(
                evaluate_method(
                    method,
                    problem,
                    df,
                    seed=config.representative_bpso_seed,
                    bpso_params=bpso_params,
                    milp_time_limit=milp_limit,
                    context=ctx,
                )
            )
    return rows


def run_epss_experiments(config: ExperimentConfig) -> List[dict[str, Any]]:
    rows: List[dict[str, Any]] = []
    scenario = "medium"
    frac = RESOURCE_SCENARIOS[scenario]
    for size in config.epss_dataset_sizes:
        try:
            base = load_scored_findings(size, config)
        except Exception as exc:  # noqa: BLE001
            for strategy in EPSS_STRATEGIES:
                for method in config.methods:
                    rows.append(
                        failed_result_row(
                            error_message=f"Failed to load dataset {size}: {exc}",
                            method=method,
                            dataset_size=size,
                        )
                    )
            continue
        path = dataset_path_str(size, config)
        bpso_params = _bpso_params_for(size, config)
        milp_limit = _milp_limit_for(size, config)
        n_missing = int(base["epss_missing"].astype(bool).sum()) if "epss_missing" in base.columns else 0
        for strategy in EPSS_STRATEGIES:
            try:
                df = apply_epss_strategy(base, strategy)
                problem = build_problem(df, frac)
            except Exception as exc:  # noqa: BLE001
                for method in config.methods:
                    rec = failed_result_row(
                        error_message=f"EPSS strategy {strategy} failed (missing={n_missing}): {exc}",
                        method=method,
                        dataset_size=size,
                    )
                    rec["epss_strategy"] = strategy
                    rows.append(rec)
                continue
            for method in config.methods:
                eid = f"epss_{strategy}_{size}_{method}"
                ctx = dict(
                    experiment_id=eid,
                    experiment_family="epss_handling",
                    dataset_size=size,
                    dataset_path=path,
                    resource_scenario=scenario,
                    resource_fraction=frac,
                    weight_scenario="default",
                    weight_sample_id=float("nan"),
                    epss_strategy=strategy,
                    milp_time_limit=milp_limit,
                    seed=config.representative_bpso_seed,
                )
                rec = evaluate_method(
                    method,
                    problem,
                    df,
                    seed=config.representative_bpso_seed,
                    bpso_params=bpso_params,
                    milp_time_limit=milp_limit,
                    context=ctx,
                )
                if n_missing == 0:
                    note = (
                        "Snapshot has zero missing EPSS values; "
                        f"{strategy} is identical to the pipeline default."
                    )
                    rec["error_message"] = note if rec["error_message"] == "" else rec["error_message"]
                rows.append(rec)
    return rows


def _write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def _failures_frame(frames: Sequence[pd.DataFrame]) -> pd.DataFrame:
    if not frames:
        return records_to_dataframe([])
    combined = pd.concat(frames, ignore_index=True)
    if "ran_successfully" not in combined.columns:
        return records_to_dataframe([])
    fails = combined[combined["ran_successfully"] == False]  # noqa: E712
    if fails.empty:
        return records_to_dataframe([])
    return records_to_dataframe(fails.to_dict(orient="records"))


def generate_figures(tables: Dict[str, pd.DataFrame], figures_dir: Path) -> List[str]:
    """Write comparison plots from *successful* rows only.  Skip empty families."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figures_dir.mkdir(parents=True, exist_ok=True)
    written: List[str] = []

    def _save(fig, name: str) -> None:
        out = figures_dir / name
        fig.tight_layout()
        fig.savefig(out, dpi=140)
        plt.close(fig)
        written.append(str(out.as_posix()))

    cmp_df = tables.get("method_comparison.csv")
    if cmp_df is not None and not cmp_df.empty:
        ok = cmp_df[cmp_df["ran_successfully"] == True].copy()  # noqa: E712
        if not ok.empty:
            fig, ax = plt.subplots(figsize=(10, 5))
            pivot = ok.pivot_table(
                index=["dataset_size", "resource_scenario"],
                columns="method",
                values="total_risk_reduction",
                aggfunc="mean",
            )
            pivot.plot(kind="bar", ax=ax)
            ax.set_ylabel("Total modeled risk reduction")
            ax.set_xlabel("Dataset size, resource scenario")
            ax.set_title("Method comparison — risk reduction")
            ax.legend(title="Method")
            _save(fig, "method_comparison_risk.png")

            fig, ax = plt.subplots(figsize=(10, 5))
            pivot_rt = ok.pivot_table(
                index=["dataset_size", "resource_scenario"],
                columns="method",
                values="runtime_s",
                aggfunc="mean",
            )
            pivot_rt.plot(kind="bar", ax=ax)
            ax.set_ylabel("Wall-clock runtime (s)")
            ax.set_xlabel("Dataset size, resource scenario")
            ax.set_title("Method comparison — runtime")
            _save(fig, "runtime_comparison.png")

            fig, ax = plt.subplots(figsize=(10, 5))
            util = ok.melt(
                id_vars=["method", "dataset_size", "resource_scenario"],
                value_vars=["pct_budget_used", "pct_hours_used", "pct_downtime_used"],
                var_name="resource",
                value_name="pct_used",
            )
            util.groupby(["method", "resource"])["pct_used"].mean().unstack().plot(kind="bar", ax=ax)
            ax.set_ylabel("Mean percent of capacity used")
            ax.set_title("Resource utilization by method")
            _save(fig, "resource_utilization.png")

            fig, ax = plt.subplots(figsize=(8, 4.5))
            feas = ok.groupby("method")["feasible"].mean() * 100.0
            feas.plot(kind="bar", ax=ax, color="#3d7a4a")
            ax.set_ylim(0, 110)
            ax.set_ylabel("Percent of runs feasible")
            ax.set_title("Feasibility by method")
            _save(fig, "feasibility.png")

            kev_ok = ok.dropna(subset=["pct_kev_coverage"])
            if not kev_ok.empty:
                fig, ax = plt.subplots(figsize=(10, 5))
                pivot_k = kev_ok.pivot_table(
                    index=["dataset_size", "resource_scenario"],
                    columns="method",
                    values="pct_kev_coverage",
                    aggfunc="mean",
                )
                pivot_k.plot(kind="bar", ax=ax)
                ax.set_ylabel("CISA KEV coverage (%)")
                ax.set_title("KEV coverage (denominator = KEV findings in the instance)")
                _save(fig, "kev_coverage.png")

    bpso = tables.get("bpso_seeds.csv")
    if bpso is not None and not bpso.empty:
        ok = bpso[bpso["ran_successfully"] == True]  # noqa: E712
        if not ok.empty:
            fig, ax = plt.subplots(figsize=(8, 4.5))
            ok.boxplot(column="total_risk_reduction", by="dataset_size", ax=ax)
            ax.set_ylabel("Total modeled risk reduction")
            ax.set_title("BPSO variability across seeds")
            fig.suptitle("")
            _save(fig, "bpso_variability.png")

    named = tables.get("sensitivity_named_weights.csv")
    if named is not None and not named.empty:
        ok = named[named["ran_successfully"] == True]  # noqa: E712
        if not ok.empty:
            fig, ax = plt.subplots(figsize=(9, 5))
            pivot = ok.pivot_table(
                index="weight_scenario",
                columns="method",
                values="total_risk_reduction",
                aggfunc="mean",
            )
            pivot.plot(kind="bar", ax=ax)
            ax.set_ylabel("Total modeled risk reduction")
            ax.set_title("Named risk-weight sensitivity")
            _save(fig, "sensitivity_named_weights.png")

    diri = tables.get("sensitivity_dirichlet.csv")
    if diri is not None and not diri.empty:
        ok = diri[diri["ran_successfully"] == True]  # noqa: E712
        if not ok.empty:
            fig, ax = plt.subplots(figsize=(9, 5))
            methods = sorted(ok["method"].unique())
            data = [ok.loc[ok["method"] == m, "total_risk_reduction"].dropna().to_numpy() for m in methods]
            try:
                ax.boxplot(data, tick_labels=methods)
            except TypeError:
                ax.boxplot(data, labels=methods)
            ax.set_ylabel("Total modeled risk reduction")
            ax.set_title("Dirichlet weight-vector sensitivity")
            _save(fig, "sensitivity_dirichlet.png")

    epss = tables.get("epss_handling.csv")
    if epss is not None and not epss.empty:
        ok = epss[epss["ran_successfully"] == True]  # noqa: E712
        if not ok.empty:
            fig, ax = plt.subplots(figsize=(9, 5))
            pivot = ok.pivot_table(
                index="epss_strategy",
                columns="method",
                values="total_risk_reduction",
                aggfunc="mean",
            )
            pivot.plot(kind="bar", ax=ax)
            ax.set_ylabel("Total modeled risk reduction")
            ax.set_title("EPSS missing-data policy comparison")
            _save(fig, "epss_handling.png")

    return written


def write_schema_example(tables_dir: Path, sample_row: Optional[dict[str, Any]] = None) -> Path:
    """One-row documented example for the Part B dashboard contract."""
    row = sample_row if sample_row is not None else empty_result_row()
    df = records_to_dataframe([row])
    path = tables_dir / "schema_example.csv"
    _write_csv(df, path)
    return path


def run_all_experiments(
    output_dir: Path | str,
    config: Optional[ExperimentConfig] = None,
    *,
    write_figures: bool = True,
) -> Dict[str, Any]:
    """
    Execute the experiment grid and write CSVs under ``output_dir/tables``
    and figures under ``output_dir/figures``.

    Returns a dict of table name -> DataFrame plus manifest metadata.
    Failed runs are stored with ``ran_successfully=False`` and an
    ``error_message``; metric fields are left as NaN rather than filled
    with illustrative numbers.
    """
    config = config or default_config()
    output_dir = Path(output_dir)
    tables_dir = output_dir / "tables"
    figures_dir = output_dir / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    families: Dict[str, List[dict[str, Any]]] = {}
    if config.run_method_comparison:
        families["method_comparison.csv"] = run_method_comparison(config)
    if config.run_bpso_multiseed:
        families["bpso_seeds.csv"] = run_bpso_multiseed(config)
    if config.run_named_weights:
        families["sensitivity_named_weights.csv"] = run_named_weight_sensitivity(config)
    if config.run_dirichlet:
        families["sensitivity_dirichlet.csv"] = run_dirichlet_sensitivity(config)
    if config.run_epss:
        families["epss_handling.csv"] = run_epss_experiments(config)

    tables: Dict[str, pd.DataFrame] = {}
    for name, recs in families.items():
        df = records_to_dataframe(recs)
        tables[name] = df
        _write_csv(df, tables_dir / name)

    bpso_summary = summarize_bpso_seeds(families.get("bpso_seeds.csv", []))
    tables["bpso_summary.csv"] = bpso_summary
    _write_csv(bpso_summary, tables_dir / "bpso_summary.csv")

    # Derived views (same measurements, narrower column sets) for the dashboard.
    if "method_comparison.csv" in tables:
        util_cols = [
            "experiment_id", "dataset_size", "resource_scenario", "method",
            "budget_limit", "budget_consumed", "budget_remaining", "pct_budget_used",
            "hours_limit", "hours_consumed", "hours_remaining", "pct_hours_used",
            "downtime_limit", "downtime_consumed", "downtime_remaining", "pct_downtime_used",
            "feasible", "ran_successfully", "error_message",
        ]
        feas_cols = [
            "experiment_id", "dataset_size", "resource_scenario", "method",
            "feasible", "n_selected", "budget_remaining", "hours_remaining",
            "downtime_remaining", "solver_status", "optimality_gap",
            "ran_successfully", "error_message",
        ]
        rt_cols = [
            "experiment_id", "dataset_size", "resource_scenario", "method",
            "runtime_s", "solver_runtime_s", "n_findings", "n_selected",
            "solver_status", "optimality_gap", "ran_successfully", "error_message",
        ]
        master = tables["method_comparison.csv"]
        tables["resource_utilization.csv"] = master[util_cols].copy()
        tables["feasibility.csv"] = master[feas_cols].copy()
        tables["runtime.csv"] = master[rt_cols].copy()
        _write_csv(tables["resource_utilization.csv"], tables_dir / "resource_utilization.csv")
        _write_csv(tables["feasibility.csv"], tables_dir / "feasibility.csv")
        _write_csv(tables["runtime.csv"], tables_dir / "runtime.csv")

    failures = _failures_frame(list(tables.values()))
    # Derived views share failure rows with method_comparison; keep unique experiment_ids.
    if not failures.empty and "experiment_id" in failures.columns:
        failures = failures.drop_duplicates(subset=["experiment_id"]).reset_index(drop=True)
    tables["experiment_failures.csv"] = failures
    _write_csv(failures, tables_dir / "experiment_failures.csv")

    sample = None
    for df in tables.values():
        ok = df[df.get("ran_successfully", False) == True] if "ran_successfully" in df.columns else df  # noqa: E712
        if not ok.empty:
            sample = ok.iloc[0].to_dict()
            break
    write_schema_example(tables_dir, sample)

    written_figures: List[str] = []
    if write_figures:
        written_figures = generate_figures(tables, figures_dir)

    manifest = {
        "resource_scenarios": RESOURCE_SCENARIOS,
        "resource_limit_definition": (
            "Each capacity is resource_fraction times the sum of that resource "
            "over all findings in the instance (compute_default_budgets)."
        ),
        "pct_risk_reduction_denominator": "sum of R_i over all candidate findings",
        "pct_kev_coverage_denominator": "count of findings with is_kev == 1",
        "epss_pipeline_default": "zero_imputation (epss_score = epss.fillna(0.0))",
        "bpso_params_by_size": config.bpso_params,
        "milp_time_limit_by_size": config.milp_time_limits,
        "bpso_seeds": config.bpso_seeds,
        "dirichlet_samples": config.dirichlet_samples,
        "dirichlet_seed": config.dirichlet_seed,
        "named_weight_scenarios": list(NAMED_WEIGHTS.keys()),
        "result_columns": list(RESULT_COLUMNS),
        "dataset_files": config.dataset_files,
        "figures": written_figures,
        "failed_experiment_count": int(len(failures)),
    }
    config_path = tables_dir / "experiment_config.json"
    config_path.write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")

    return {
        "tables": tables,
        "tables_dir": tables_dir,
        "figures_dir": figures_dir,
        "manifest": manifest,
        "figures": written_figures,
    }


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run VulnOpt evaluation experiments.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT_DIR / "results",
        help="Directory for tables/ and figures/ (default: ./results)",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Reduced grid for tests (n=100, 2 BPSO seeds, 3 Dirichlet samples).",
    )
    parser.add_argument(
        "--no-figures",
        action="store_true",
        help="Skip PNG generation.",
    )
    parser.add_argument(
        "--datasets",
        type=int,
        nargs="*",
        default=None,
        help="Override dataset sizes, e.g. --datasets 100 1000",
    )
    parser.add_argument(
        "--bpso-seeds",
        type=int,
        default=None,
        help="Number of consecutive BPSO seeds starting at 0.",
    )
    parser.add_argument(
        "--dirichlet-samples",
        type=int,
        default=None,
        help="Override Dirichlet sample count.",
    )
    parser.add_argument(
        "--skip-dirichlet",
        action="store_true",
        help="Skip the 100-vector Dirichlet grid.",
    )
    parser.add_argument(
        "--skip-epss",
        action="store_true",
        help="Skip EPSS policy experiments.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    config = quick_config() if args.quick else default_config()
    if args.datasets:
        config.dataset_sizes = list(args.datasets)
        config.epss_dataset_sizes = [s for s in args.datasets if s in (100, 1000)] or list(args.datasets)[:1]
        if config.sensitivity_dataset_size not in config.dataset_sizes:
            config.sensitivity_dataset_size = config.dataset_sizes[0]
    if args.bpso_seeds is not None:
        config.bpso_seeds = list(range(args.bpso_seeds))
    if args.dirichlet_samples is not None:
        config.dirichlet_samples = args.dirichlet_samples
    if args.skip_dirichlet:
        config.run_dirichlet = False
    if args.skip_epss:
        config.run_epss = False

    print("VulnOpt experiments")
    print(f"  output     : {args.output_dir}")
    print(f"  datasets   : {config.dataset_sizes}")
    print(f"  resources  : {config.resource_scenarios}")
    print(f"  bpso seeds : {config.bpso_seeds}")
    print(f"  dirichlet  : {config.dirichlet_samples if config.run_dirichlet else 'skipped'}")

    result = run_all_experiments(
        args.output_dir,
        config,
        write_figures=not args.no_figures,
    )
    fails = result["tables"]["experiment_failures.csv"]
    print(f"Wrote tables to {result['tables_dir']}")
    print(f"Wrote {len(result['figures'])} figures to {result['figures_dir']}")
    print(f"Failed experiment rows: {len(fails)}")
    if len(fails):
        print(fails[["experiment_id", "method", "error_message"]].head(20).to_string(index=False))
    return 0 if len(fails) == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
