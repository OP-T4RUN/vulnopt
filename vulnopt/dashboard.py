"""
Read-only helpers for the VulnOpt Streamlit dashboard.

Loads Part A experiment CSVs and scored-finding tables. Solver plans are
reconstructed by calling the existing experiment dispatcher
(``vulnopt.experiments.run_solver``); metrics are not reimplemented here.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from vulnopt.constraints import OptimizationProblem
from vulnopt.experiments import (
    BPSO_PARAMS_BY_SIZE,
    DATASET_FILES,
    METHODS,
    MILP_TIME_LIMIT_BY_SIZE,
    NAMED_WEIGHTS,
    RESOURCE_SCENARIOS,
    ExperimentConfig,
    apply_epss_strategy,
    build_problem,
    dataset_path_str,
    load_scored_findings,
    rescore_with_weights,
    run_solver,
)
from vulnopt.metrics import RESULT_COLUMNS
from vulnopt.risk_model import DEFAULT_WEIGHTS


ROOT_DIR = Path(__file__).resolve().parent.parent

FAMILY_TABLES: Dict[str, str] = {
    "method_comparison": "method_comparison.csv",
    "resource_utilization": "resource_utilization.csv",
    "feasibility": "feasibility.csv",
    "runtime": "runtime.csv",
    "bpso_seeds": "bpso_seeds.csv",
    "bpso_summary": "bpso_summary.csv",
    "sensitivity_named_weights": "sensitivity_named_weights.csv",
    "sensitivity_dirichlet": "sensitivity_dirichlet.csv",
    "epss_handling": "epss_handling.csv",
    "experiment_failures": "experiment_failures.csv",
    "schema_example": "schema_example.csv",
}

OVERVIEW_COLUMNS: tuple[str, ...] = (
    "method",
    "n_selected",
    "total_risk_reduction",
    "pct_risk_reduction",
    "budget_consumed",
    "pct_budget_used",
    "hours_consumed",
    "pct_hours_used",
    "downtime_consumed",
    "pct_downtime_used",
    "n_kev_selected",
    "pct_kev_coverage",
    "feasible",
    "runtime_s",
    "solver_status",
    "optimality_gap",
    "ran_successfully",
    "error_message",
)

FINDING_DISPLAY_COLUMNS: tuple[str, ...] = (
    "finding_id",
    "cve_id",
    "cvss_base",
    "cvss_severity",
    "epss",
    "epss_percentile",
    "is_kev",
    "kev_ransomware",
    "risk_score",
    "asset_id",
    "asset_type",
    "criticality",
    "exposure",
    "environment",
    "business_unit",
    "patch_cost",
    "engineer_hours",
    "downtime_hours",
)

EXPORT_PLAN_EXTRA: tuple[str, ...] = (
    "selected",
    "method",
    "dataset_size",
    "resource_scenario",
    "budget_limit",
    "hours_limit",
    "downtime_limit",
    "weight_scenario",
    "downtime_constraint_enabled",
)


def project_root() -> Path:
    return ROOT_DIR


def default_results_dir(root: Optional[Path] = None) -> Path:
    return (root or ROOT_DIR) / "results"


def tables_dir(results_dir: Optional[Path] = None) -> Path:
    return (results_dir or default_results_dir()) / "tables"


def figures_dir(results_dir: Optional[Path] = None) -> Path:
    return (results_dir or default_results_dir()) / "figures"


def load_table(filename: str, directory: Optional[Path] = None) -> pd.DataFrame:
    """
    Load a CSV from ``results/tables``.

    Missing, empty, or unreadable files return an empty DataFrame. Values are
    not filled with zeros or sample metrics.
    """
    path = (directory or tables_dir()) / filename
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    try:
        df = pd.read_csv(path)
    except (OSError, pd.errors.EmptyDataError, pd.errors.ParserError, ValueError):
        return pd.DataFrame()
    if df is None or df.empty:
        return pd.DataFrame(columns=list(df.columns) if df is not None else [])
    return df


def load_all_tables(directory: Optional[Path] = None) -> Dict[str, pd.DataFrame]:
    out: Dict[str, pd.DataFrame] = {}
    folder = directory or tables_dir()
    for key, name in FAMILY_TABLES.items():
        out[key] = load_table(name, folder)
    return out


def load_experiment_config(directory: Optional[Path] = None) -> Dict[str, Any]:
    path = (directory or tables_dir()) / "experiment_config.json"
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, ValueError):
        return {}


def table_file_status(directory: Optional[Path] = None) -> Dict[str, str]:
    """
    Classify each family table without inventing rows.

    Values: ``missing`` (file absent), ``empty_file`` (zero bytes),
    ``no_rows`` (readable header only / empty frame), ``loaded``,
    ``unreadable``.
    """
    folder = directory or tables_dir()
    status: Dict[str, str] = {}
    for key, name in FAMILY_TABLES.items():
        path = folder / name
        if not path.exists():
            status[key] = "missing"
            continue
        if path.stat().st_size == 0:
            status[key] = "empty_file"
            continue
        df = load_table(name, folder)
        if df is None:
            status[key] = "unreadable"
        elif df.empty:
            status[key] = "no_rows"
        else:
            status[key] = "loaded"
    return status


def missing_table_names(tables: Dict[str, pd.DataFrame]) -> List[str]:
    missing = []
    for key, name in FAMILY_TABLES.items():
        df = tables.get(key)
        if df is None or df.empty:
            missing.append(name)
    return missing


def is_missing_metric(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str) and value.strip() == "":
        return True
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def format_metric(value: Any) -> str:
    """Human-readable cell for the dashboard. Does not replace missing values with 0."""
    if is_missing_metric(value):
        return "—"
    if isinstance(value, (bool, np.bool_)):
        return "True" if bool(value) else "False"
    if isinstance(value, str):
        return value
    try:
        num = float(value)
    except (TypeError, ValueError):
        return str(value)
    if pd.isna(num):
        return "—"
    if abs(num - round(num)) < 1e-9 and abs(num) >= 1:
        return f"{int(round(num)):,}"
    if abs(num) >= 100:
        return f"{num:,.2f}"
    return f"{num:.4g}"


def available_dataset_sizes(
    comparison: pd.DataFrame,
    root: Optional[Path] = None,
) -> List[int]:
    sizes: set[int] = set()
    if comparison is not None and not comparison.empty and "dataset_size" in comparison.columns:
        for value in comparison["dataset_size"].dropna().unique():
            try:
                sizes.add(int(value))
            except (TypeError, ValueError):
                continue
    base = root or ROOT_DIR
    for size, rel in DATASET_FILES.items():
        if (base / rel).exists():
            sizes.add(int(size))
    return sorted(sizes)


def available_methods(comparison: pd.DataFrame) -> List[str]:
    if comparison is None or comparison.empty or "method" not in comparison.columns:
        return list(METHODS)
    present = [m for m in METHODS if m in set(comparison["method"].astype(str))]
    extras = sorted(
        m for m in comparison["method"].dropna().astype(str).unique() if m not in METHODS
    )
    return present + extras


def _as_optional_int(value: Any) -> Optional[int]:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def filter_result_rows(
    df: pd.DataFrame,
    *,
    dataset_size: Optional[int] = None,
    resource_scenario: Optional[str] = None,
    methods: Optional[Sequence[str]] = None,
    weight_scenario: Optional[str] = None,
    epss_strategy: Optional[str] = None,
    successful_only: bool = False,
) -> pd.DataFrame:
    """Filter saved experiment rows. Returns an empty frame when nothing matches."""
    if df is None or df.empty:
        return pd.DataFrame()
    out = df.copy()
    if dataset_size is not None and "dataset_size" in out.columns:
        out = out[out["dataset_size"].apply(_as_optional_int) == int(dataset_size)]
    if resource_scenario and "resource_scenario" in out.columns:
        out = out[out["resource_scenario"].astype(str) == str(resource_scenario)]
    if methods is not None and "method" in out.columns:
        wanted = set(methods)
        out = out[out["method"].astype(str).isin(wanted)]
    if weight_scenario and "weight_scenario" in out.columns:
        out = out[out["weight_scenario"].astype(str) == str(weight_scenario)]
    if epss_strategy and "epss_strategy" in out.columns:
        out = out[out["epss_strategy"].astype(str) == str(epss_strategy)]
    if successful_only and "ran_successfully" in out.columns:
        flag = out["ran_successfully"]
        if flag.dtype == object:
            out = out[flag.astype(str).str.lower().isin(("true", "1"))]
        else:
            out = out[flag == True]  # noqa: E712
    return out.reset_index(drop=True)


def overview_table(df: pd.DataFrame) -> pd.DataFrame:
    """Narrow comparison view. Missing columns are omitted, not invented."""
    if df is None or df.empty:
        return pd.DataFrame()
    cols = [c for c in OVERVIEW_COLUMNS if c in df.columns]
    return df.loc[:, cols].copy()


def load_findings(
    dataset_size: int,
    *,
    root: Optional[Path] = None,
    config: Optional[ExperimentConfig] = None,
) -> pd.DataFrame:
    cfg = config or ExperimentConfig()
    if root is not None:
        cfg.data_root = str(root)
    try:
        return load_scored_findings(int(dataset_size), cfg)
    except FileNotFoundError:
        return pd.DataFrame()


def findings_display_columns(df: pd.DataFrame) -> List[str]:
    return [c for c in FINDING_DISPLAY_COLUMNS if c in df.columns]


def filter_findings(
    df: pd.DataFrame,
    *,
    query: str = "",
    kev_only: bool = False,
    min_cvss: Optional[float] = None,
    exposures: Optional[Sequence[str]] = None,
    selected_only: bool = False,
) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    out = df.copy()
    text = (query or "").strip().lower()
    if text:
        searchable = [
            c
            for c in ("finding_id", "cve_id", "asset_id", "asset_type", "cwe_ids", "cvss_severity")
            if c in out.columns
        ]
        if searchable:
            mask = pd.Series(False, index=out.index)
            for col in searchable:
                mask = mask | out[col].astype(str).str.lower().str.contains(text, na=False)
            out = out[mask]
    if kev_only and "is_kev" in out.columns:
        kev = pd.to_numeric(out["is_kev"], errors="coerce").fillna(0)
        out = out[kev > 0]
    if min_cvss is not None and "cvss_base" in out.columns:
        cvss = pd.to_numeric(out["cvss_base"], errors="coerce")
        out = out[cvss >= float(min_cvss)]
    if exposures:
        if "exposure" in out.columns:
            out = out[out["exposure"].astype(str).isin(list(exposures))]
    if selected_only:
        if "selected" not in out.columns:
            return pd.DataFrame(columns=out.columns)
        flag = out["selected"]
        if flag.dtype == object:
            out = out[flag.astype(str).str.lower().isin(("true", "1"))]
        else:
            out = out[flag.astype(bool)]
    return out.reset_index(drop=True)


def resource_fraction_for(scenario: str) -> Optional[float]:
    return RESOURCE_SCENARIOS.get(scenario)


def default_capacities(findings: pd.DataFrame, resource_fraction: float) -> Dict[str, float]:
    if findings is None or findings.empty:
        return {"budget": float("nan"), "max_hours": float("nan"), "max_downtime": float("nan")}
    problem = build_problem(findings, resource_fraction)
    return {
        "budget": float(problem.budget),
        "max_hours": float(problem.max_hours),
        "max_downtime": float(problem.max_downtime),
    }


def bpso_params_for(dataset_size: int) -> Dict[str, int]:
    if dataset_size in BPSO_PARAMS_BY_SIZE:
        return dict(BPSO_PARAMS_BY_SIZE[dataset_size])
    if dataset_size <= 100:
        return dict(BPSO_PARAMS_BY_SIZE[100])
    if dataset_size <= 1000:
        return dict(BPSO_PARAMS_BY_SIZE[1000])
    return dict(BPSO_PARAMS_BY_SIZE[5000])


def milp_time_limit_for(dataset_size: int) -> int:
    if dataset_size in MILP_TIME_LIMIT_BY_SIZE:
        return int(MILP_TIME_LIMIT_BY_SIZE[dataset_size])
    return int(MILP_TIME_LIMIT_BY_SIZE.get(5000, 180))


def prepare_findings(
    findings: pd.DataFrame,
    *,
    weight_scenario: str = "default",
    custom_weights: Optional[Dict[str, float]] = None,
    epss_strategy: str = "zero_imputation",
) -> pd.DataFrame:
    if findings is None or findings.empty:
        return pd.DataFrame()
    use_named = custom_weights is None and weight_scenario == "default"
    if use_named and epss_strategy == "zero_imputation":
        return findings.copy()
    df = apply_epss_strategy(findings, epss_strategy)
    if custom_weights is not None:
        weights = custom_weights
    else:
        weights = NAMED_WEIGHTS.get(weight_scenario, DEFAULT_WEIGHTS)
    return rescore_with_weights(df, weights)


def build_custom_problem(
    findings: pd.DataFrame,
    *,
    budget: float,
    max_hours: float,
    max_downtime: float,
    apply_downtime: bool = True,
) -> OptimizationProblem:
    downtime = float(max_downtime) if apply_downtime else float(np.sum(findings["downtime_hours"])) + 1.0
    if not apply_downtime:
        downtime = max(downtime, 1.0)
    return OptimizationProblem.from_dataframe(
        findings,
        budget=float(budget),
        max_hours=float(max_hours),
        max_downtime=downtime,
    )


def reconstruct_plan(
    method: str,
    findings: pd.DataFrame,
    problem: OptimizationProblem,
    *,
    seed: int = 42,
    dataset_size: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Reconstruct a remediation plan with the existing solver implementations.

    Returns selection, solver result, and the findings rows that were selected.
    Does not write experiment CSVs.
    """
    size = int(dataset_size if dataset_size is not None else problem.n)
    result = run_solver(
        method,
        problem,
        findings,
        seed=int(seed),
        bpso_params=bpso_params_for(size),
        milp_time_limit=milp_time_limit_for(size),
    )
    selection = np.asarray(result.selection, dtype=int).reshape(-1)
    selected = findings.iloc[selection.astype(bool)].copy()
    selected.insert(0, "selected", True)
    return {
        "result": result,
        "selection": selection,
        "selected_findings": selected.reset_index(drop=True),
        "n_selected": int(selection.sum()),
    }


def annotate_selection(findings: pd.DataFrame, selection: np.ndarray) -> pd.DataFrame:
    out = findings.copy()
    sel = np.asarray(selection, dtype=bool).reshape(-1)
    if len(sel) != len(out):
        raise ValueError("Selection length does not match findings table.")
    out["selected"] = sel
    return out


def dataframe_to_csv_bytes(df: pd.DataFrame) -> bytes:
    if df is None or df.empty:
        return b""
    return df.to_csv(index=False).encode("utf-8")


def comparison_export(
    df: pd.DataFrame,
    *,
    extra: Optional[Dict[str, Any]] = None,
) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    out = df.copy()
    if extra:
        for key, value in extra.items():
            if key not in out.columns:
                out[key] = value
    preferred = [c for c in RESULT_COLUMNS if c in out.columns]
    rest = [c for c in out.columns if c not in preferred]
    return out.loc[:, preferred + rest]


def plan_export(
    selected_findings: pd.DataFrame,
    *,
    method: str,
    dataset_size: int,
    resource_scenario: str,
    budget: float,
    max_hours: float,
    max_downtime: float,
    weight_scenario: str,
    apply_downtime: bool,
) -> pd.DataFrame:
    if selected_findings is None or selected_findings.empty:
        return pd.DataFrame()
    out = selected_findings.copy()
    out["method"] = method
    out["dataset_size"] = dataset_size
    out["resource_scenario"] = resource_scenario
    out["budget_limit"] = budget
    out["hours_limit"] = max_hours
    out["downtime_limit"] = max_downtime if apply_downtime else pd.NA
    out["weight_scenario"] = weight_scenario
    out["downtime_constraint_enabled"] = bool(apply_downtime)
    if "selected" not in out.columns:
        out["selected"] = True
    cols = [c for c in EXPORT_PLAN_EXTRA if c in out.columns]
    rest = [c for c in out.columns if c not in cols]
    return out.loc[:, cols + rest]


def finding_detail_export(
    findings: pd.DataFrame,
    *,
    method: str,
    dataset_size: int,
    resource_scenario: str,
) -> pd.DataFrame:
    if findings is None or findings.empty:
        return pd.DataFrame()
    out = findings.copy()
    out["method"] = method
    out["dataset_size"] = dataset_size
    out["resource_scenario"] = resource_scenario
    return out


def findings_snapshot_export(
    findings: pd.DataFrame,
    *,
    dataset_size: int,
    resource_scenario: str,
) -> pd.DataFrame:
    """Finding table for the selected dataset, with scenario metadata and no invented metrics."""
    if findings is None or findings.empty:
        return pd.DataFrame()
    out = findings.copy()
    out["dataset_size"] = dataset_size
    out["resource_scenario"] = resource_scenario
    return out


def list_figure_paths(results: Optional[Path] = None) -> List[Path]:
    folder = figures_dir(results)
    if not folder.exists():
        return []
    return sorted(p for p in folder.glob("*.png") if p.is_file())


def figure_path(name: str, results: Optional[Path] = None) -> Optional[Path]:
    path = figures_dir(results) / name
    return path if path.exists() else None


def dataset_label(size: int, root: Optional[Path] = None) -> str:
    rel = dataset_path_str(size, ExperimentConfig(data_root=str(root or ROOT_DIR)))
    return f"{size:,} findings ({rel})"


def weight_choices() -> List[str]:
    return list(NAMED_WEIGHTS.keys())


def method_choices() -> List[str]:
    return list(METHODS)


def scenario_choices() -> List[str]:
    return list(RESOURCE_SCENARIOS.keys())


def coerce_bool_series(series: pd.Series) -> pd.Series:
    if series.dtype == object:
        return series.astype(str).str.lower().isin(("true", "1"))
    return series.astype(bool)


def successful_subset(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty or "ran_successfully" not in df.columns:
        return pd.DataFrame() if df is None else df.copy()
    return df[coerce_bool_series(df["ran_successfully"])].copy()
