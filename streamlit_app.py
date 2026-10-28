"""
VulnOpt Streamlit dashboard (Student 4, Part B).

Reads saved experiment tables under results/tables/ and scored finding CSVs.
Does not reimplement metrics or solvers. Optional live plans call
vulnopt.experiments.run_solver.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import pandas as pd
import plotly.express as px
import streamlit as st

from vulnopt.dashboard import (
    FAMILY_TABLES,
    annotate_selection,
    available_dataset_sizes,
    available_methods,
    bpso_params_for,
    build_custom_problem,
    comparison_export,
    dataframe_to_csv_bytes,
    default_capacities,
    default_results_dir,
    figure_path,
    filter_findings,
    filter_result_rows,
    finding_detail_export,
    findings_display_columns,
    findings_snapshot_export,
    format_metric,
    list_figure_paths,
    load_all_tables,
    load_experiment_config,
    load_findings,
    method_choices,
    milp_time_limit_for,
    overview_table,
    plan_export,
    prepare_findings,
    reconstruct_plan,
    resource_fraction_for,
    scenario_choices,
    successful_subset,
    table_file_status,
    tables_dir,
    weight_choices,
)
from vulnopt.experiments import run_all_experiments, quick_config, default_config
from vulnopt.risk_model import ASSET_FOCUSED_WEIGHTS, DEFAULT_WEIGHTS, THREAT_FOCUSED_WEIGHTS


st.set_page_config(
    page_title="VulnOpt Dashboard",
    page_icon="🛡️",
    layout="wide",
)

WEIGHT_PRESETS = {
    "default": DEFAULT_WEIGHTS,
    "threat_focused": THREAT_FOCUSED_WEIGHTS,
    "asset_focused": ASSET_FOCUSED_WEIGHTS,
}

OVERVIEW_METRIC_LABELS = {
    "pct_risk_reduction": "% risk reduction",
    "total_risk_reduction": "Risk reduction (R_i sum)",
    "n_selected": "Findings selected",
    "pct_kev_coverage": "% KEV coverage",
    "runtime_s": "Runtime (s)",
    "pct_budget_used": "% budget used",
    "pct_hours_used": "% hours used",
    "pct_downtime_used": "% downtime used",
}


@st.cache_data(show_spinner=False)
def _cached_tables(tables_path: str) -> Dict[str, pd.DataFrame]:
    return load_all_tables(Path(tables_path))


@st.cache_data(show_spinner=False)
def _cached_config(tables_path: str) -> dict:
    return load_experiment_config(Path(tables_path))


@st.cache_data(show_spinner=False)
def _cached_findings(size: int) -> pd.DataFrame:
    return load_findings(size)


@st.cache_data(show_spinner="Reconstructing plan with existing solvers…")
def _cached_plan(
    method: str,
    dataset_size: int,
    weight_scenario: str,
    epss_strategy: str,
    budget: float,
    max_hours: float,
    max_downtime: float,
    apply_downtime: bool,
    seed: int,
    w_cvss: float,
    w_epss: float,
    w_kev: float,
    w_crit: float,
    w_exp: float,
    use_custom_weights: bool,
) -> Dict[str, object]:
    findings = _cached_findings(dataset_size)
    custom = None
    if use_custom_weights:
        custom = {
            "cvss": w_cvss,
            "epss": w_epss,
            "kev": w_kev,
            "criticality": w_crit,
            "exposure": w_exp,
        }
    scored = prepare_findings(
        findings,
        weight_scenario=weight_scenario,
        custom_weights=custom,
        epss_strategy=epss_strategy,
    )
    problem = build_custom_problem(
        scored,
        budget=budget,
        max_hours=max_hours,
        max_downtime=max_downtime,
        apply_downtime=apply_downtime,
    )
    plan = reconstruct_plan(method, scored, problem, seed=seed, dataset_size=dataset_size)
    annotated = annotate_selection(scored, plan["selection"])
    return {
        "annotated": annotated,
        "selected": plan["selected_findings"],
        "n_selected": plan["n_selected"],
        "status": getattr(plan["result"], "status", None),
        "gap": getattr(plan["result"], "optimality_gap", None),
        "feasible": getattr(plan["result"], "feasible", None),
        "risk": getattr(plan["result"], "total_risk_score", None),
    }


def _fmt_number(value) -> str:
    return format_metric(value)


def _chart_frame(df: pd.DataFrame, y_col: str) -> Optional[pd.DataFrame]:
    if df is None or df.empty or y_col not in df.columns or "method" not in df.columns:
        return None
    subset = df.dropna(subset=[y_col])
    if subset.empty:
        return None
    return subset


def _bar_chart(df: pd.DataFrame, y_col: str, title: str, y_label: str) -> None:
    frame = _chart_frame(df, y_col)
    if frame is None:
        st.caption(f"{title} unavailable (missing values in saved results).")
        return
    fig = px.bar(
        frame,
        x="method",
        y=y_col,
        color="method",
        title=title,
        labels={"method": "Method", y_col: y_label},
    )
    fig.update_layout(showlegend=False, xaxis_title="", yaxis_title=y_label, margin=dict(t=48, b=16))
    st.plotly_chart(fig, use_container_width=True)


def main() -> None:
    st.title("VulnOpt")
    st.caption(
        "Risk-aware vulnerability remediation under budget, engineer-hour, "
        "and downtime constraints. Decision support only — no scanning or patching."
    )

    results = default_results_dir()
    table_path = tables_dir(results)
    tables = _cached_tables(str(table_path))
    manifest = _cached_config(str(table_path))

    comparison = tables.get("method_comparison", pd.DataFrame())
    sizes = available_dataset_sizes(comparison)
    methods_available = available_methods(comparison)
    scenarios = [s for s in scenario_choices() if s]

    with st.sidebar:
        st.header("Scenario")
        if not sizes:
            st.warning("No scored datasets or comparison tables were found.")
            dataset_size = 100
        else:
            size_labels = {s: f"{s:,} findings" for s in sizes}
            dataset_size = st.selectbox(
                "Dataset size",
                sizes,
                index=min(1, len(sizes) - 1) if 1000 in sizes else 0,
                format_func=lambda s: size_labels.get(s, str(s)),
            )

        if comparison.empty:
            st.info("Saved method-comparison results are missing. Overview tables will be empty.")

        resource_scenario = st.selectbox(
            "Resource scenario",
            scenarios or ["medium"],
            index=(scenarios.index("medium") if "medium" in scenarios else 0),
            help="Tight / medium / relaxed use 30% / 50% / 70% of total patch cost, hours, and downtime.",
        )
        frac = resource_fraction_for(resource_scenario) or 0.5

        findings_preview = _cached_findings(dataset_size)
        caps = default_capacities(findings_preview, frac) if not findings_preview.empty else {
            "budget": 0.0, "max_hours": 0.0, "max_downtime": 0.0
        }

        st.subheader("Capacities")
        budget = st.number_input(
            "Budget ($)",
            min_value=0.0,
            value=float(caps["budget"]) if pd.notna(caps["budget"]) else 0.0,
            step=100.0,
            help="Defaults to the selected resource scenario. Used for plan reconstruction, not for rewriting saved CSVs.",
        )
        hours = st.number_input(
            "Engineer-hours",
            min_value=0.0,
            value=float(caps["max_hours"]) if pd.notna(caps["max_hours"]) else 0.0,
            step=1.0,
        )
        apply_downtime = st.checkbox("Apply downtime constraint", value=True)
        downtime = st.number_input(
            "Downtime (hours)",
            min_value=0.0,
            value=float(caps["max_downtime"]) if pd.notna(caps["max_downtime"]) else 0.0,
            step=0.1,
            disabled=not apply_downtime,
        )

        st.subheader("Risk weights")
        weight_scenario = st.selectbox("Weight scenario", weight_choices())
        use_custom_weights = st.checkbox("Override with custom weights", value=False)
        w_cvss = w_epss = w_kev = w_crit = w_exp = 0.0
        if use_custom_weights:
            preset = WEIGHT_PRESETS.get(weight_scenario, DEFAULT_WEIGHTS)
            w_cvss = st.slider("w_cvss", 0.0, 1.0, float(preset["cvss"]), 0.05)
            w_epss = st.slider("w_epss", 0.0, 1.0, float(preset["epss"]), 0.05)
            w_kev = st.slider("w_kev", 0.0, 1.0, float(preset["kev"]), 0.05)
            w_crit = st.slider("w_criticality", 0.0, 1.0, float(preset["criticality"]), 0.05)
            w_exp = st.slider("w_exposure", 0.0, 1.0, float(preset["exposure"]), 0.05)
            w_sum = w_cvss + w_epss + w_kev + w_crit + w_exp
            if abs(w_sum - 1.0) > 1e-5:
                st.error(f"Custom weights must sum to 1.0 (currently {w_sum:.3f}).")
        else:
            shown = WEIGHT_PRESETS.get(weight_scenario, DEFAULT_WEIGHTS)
            st.caption(", ".join(f"{k}={v:.2f}" for k, v in shown.items()))

        st.subheader("Methods")
        default_methods = methods_available or method_choices()
        selected_methods = st.multiselect(
            "Compare methods",
            options=method_choices(),
            default=default_methods,
        )
        inspector_method = st.selectbox(
            "Inspector / export method",
            selected_methods or method_choices(),
        )

        st.subheader("Results source")
        st.caption(f"Tables directory: `{table_path}`")
        file_status = table_file_status(table_path)
        absent = [
            FAMILY_TABLES[key]
            for key, state in file_status.items()
            if state in {"missing", "empty_file", "unreadable"}
        ]
        header_only = [
            FAMILY_TABLES[key]
            for key, state in file_status.items()
            if state == "no_rows"
        ]
        if absent:
            with st.expander("Missing table files"):
                st.write(", ".join(absent))
                st.caption("Missing files are left empty in the UI; values are not fabricated.")
        if header_only:
            with st.expander("Tables with no data rows"):
                st.write(", ".join(header_only))
                st.caption("Header-only files (for example zero recorded failures) are not filled with sample rows.")

        with st.expander("Re-run experiments (optional)"):
            st.caption(
                "Calls the existing experiment runner. Default output is "
                "`results/dashboard_rerun` so Part A files are not overwritten."
            )
            rerun_quick = st.checkbox("Quick grid only", value=True)
            overwrite = st.checkbox("Write into results/ (overwrite Part A tables)", value=False)
            if st.button("Run experiment grid"):
                out_dir = results if overwrite else results / "dashboard_rerun"
                cfg = quick_config() if rerun_quick else default_config()
                with st.spinner(f"Running experiments into {out_dir}…"):
                    run_all_experiments(out_dir, cfg, write_figures=True)
                st.success(f"Finished. Output: {out_dir}")
                st.cache_data.clear()

    filtered = filter_result_rows(
        comparison,
        dataset_size=dataset_size,
        resource_scenario=resource_scenario,
        methods=selected_methods or None,
    )
    if weight_scenario != "default" and "weight_scenario" in comparison.columns:
        st.info(
            "Saved method-comparison rows use default weights. "
            "Named-weight results are on the Sensitivity tab."
        )

    tab_overview, tab_findings, tab_sensitivity, tab_export = st.tabs(
        ["Results overview", "Finding inspector", "Sensitivity", "CSV export"]
    )

    with tab_overview:
        st.subheader("Saved experiment comparison")
        st.caption(
            f"Dataset n={dataset_size}, resource scenario={resource_scenario} "
            f"(fraction {frac:.0%} of total cost, hours, and downtime)."
        )
        if filtered.empty:
            st.warning("No saved comparison rows match this dataset, scenario, and method selection.")
        else:
            display = overview_table(filtered)
            st.dataframe(display, use_container_width=True, hide_index=True)

            cols = st.columns(4)
            for method, col in zip(method_choices(), cols):
                row = filtered[filtered["method"] == method]
                with col:
                    st.markdown(f"**{method}**")
                    if row.empty:
                        st.caption("No saved result")
                        continue
                    rec = row.iloc[0]
                    if "ran_successfully" in rec and not (
                        rec["ran_successfully"] in (True, "True", "true", 1)
                    ):
                        st.caption("Run did not succeed")
                        err = rec.get("error_message")
                        if isinstance(err, str) and err:
                            st.caption(err)
                        continue
                    st.metric("% risk reduction", _fmt_number(rec.get("pct_risk_reduction")))
                    st.caption(
                        f"Selected {_fmt_number(rec.get('n_selected'))} · "
                        f"KEV {_fmt_number(rec.get('pct_kev_coverage'))}% · "
                        f"feasible={_fmt_number(rec.get('feasible'))}"
                    )

            chart_cols = st.columns(2)
            with chart_cols[0]:
                _bar_chart(
                    filtered,
                    "pct_risk_reduction",
                    "Percentage risk reduction (saved runs)",
                    OVERVIEW_METRIC_LABELS["pct_risk_reduction"],
                )
            with chart_cols[1]:
                _bar_chart(
                    filtered,
                    "runtime_s",
                    "Wall-clock runtime",
                    OVERVIEW_METRIC_LABELS["runtime_s"],
                )

            util_cols = [c for c in ("pct_budget_used", "pct_hours_used", "pct_downtime_used") if c in filtered.columns]
            if util_cols and "method" in filtered.columns:
                util = filtered[["method", *util_cols]].dropna(how="all", subset=util_cols)
                if util.empty:
                    st.caption("Resource-utilization chart unavailable (missing values).")
                else:
                    melted = util.melt(id_vars="method", value_vars=util_cols, var_name="resource", value_name="pct_used")
                    melted = melted.dropna(subset=["pct_used"])
                    melted["resource"] = melted["resource"].map(
                        {
                            "pct_budget_used": "% budget used",
                            "pct_hours_used": "% hours used",
                            "pct_downtime_used": "% downtime used",
                        }
                    )
                    st.markdown("**Resource utilization (% of limit)**")
                    fig = px.bar(
                        melted,
                        x="method",
                        y="pct_used",
                        color="resource",
                        barmode="group",
                        labels={"method": "Method", "pct_used": "% of limit", "resource": "Resource"},
                    )
                    fig.update_layout(xaxis_title="", margin=dict(t=16, b=16))
                    st.plotly_chart(fig, use_container_width=True)

            milp_rows = filtered[filtered["method"] == "MILP"] if "method" in filtered.columns else pd.DataFrame()
            if not milp_rows.empty:
                st.markdown("**MILP solver diagnostics**")
                milp_cols = [c for c in ("solver_status", "optimality_gap", "solver_backend", "solver_runtime_s", "feasible") if c in milp_rows.columns]
                st.dataframe(milp_rows[milp_cols], use_container_width=True, hide_index=True)

            infeas = filtered[filtered["feasible"].isin([False, "False", "false", 0])] if "feasible" in filtered.columns else pd.DataFrame()
            if not infeas.empty:
                st.warning(
                    "At least one saved run is marked infeasible. "
                    "Values are shown as recorded (including floating-point slack)."
                )

        fig = figure_path("method_comparison_risk.png")
        if fig is not None:
            st.image(str(fig), caption="Saved figure: method comparison — risk reduction")

        fail_df = tables.get("experiment_failures", pd.DataFrame())
        if fail_df is not None and not fail_df.empty:
            st.subheader("Failed experiment rows")
            st.dataframe(fail_df, use_container_width=True, hide_index=True)
        elif manifest.get("failed_experiment_count") == 0:
            st.caption("experiment_failures.csv has no data rows (zero recorded failures).")

    with tab_findings:
        st.subheader("Vulnerability findings")
        findings = _cached_findings(dataset_size)
        if findings.empty:
            st.warning(f"No scored findings file is available for n={dataset_size}.")
        else:
            compute = st.checkbox(
                f"Highlight findings selected by {inspector_method}",
                value=False,
                help="Runs the existing solver for the current capacities. Saved overview metrics are not replaced.",
            )
            annotated = findings
            selected_count = None
            solver_note = ""
            custom_ok = True
            if use_custom_weights:
                custom_ok = abs((w_cvss + w_epss + w_kev + w_crit + w_exp) - 1.0) <= 1e-5
            if compute:
                if not custom_ok:
                    st.error("Cannot reconstruct a plan until custom weights sum to 1.0.")
                else:
                    try:
                        plan = _cached_plan(
                            inspector_method,
                            dataset_size,
                            weight_scenario,
                            "zero_imputation",
                            float(budget),
                            float(hours),
                            float(downtime),
                            bool(apply_downtime),
                            42,
                            float(w_cvss),
                            float(w_epss),
                            float(w_kev),
                            float(w_crit),
                            float(w_exp),
                            bool(use_custom_weights),
                        )
                        annotated = plan["annotated"]
                        selected_count = plan["n_selected"]
                        bits = [f"selected={selected_count}"]
                        if plan["feasible"] is not None:
                            bits.append(f"feasible={plan['feasible']}")
                        if plan["status"]:
                            bits.append(f"status={plan['status']}")
                        if plan["gap"] is not None and not (isinstance(plan["gap"], float) and pd.isna(plan["gap"])):
                            bits.append(f"gap={plan['gap']}")
                        solver_note = "Live plan: " + ", ".join(bits)
                        st.session_state["last_plan"] = {
                            "method": inspector_method,
                            "dataset_size": dataset_size,
                            "resource_scenario": resource_scenario,
                            "budget": budget,
                            "hours": hours,
                            "downtime": downtime,
                            "apply_downtime": apply_downtime,
                            "weight_scenario": weight_scenario if not use_custom_weights else "custom",
                            "annotated": annotated,
                            "selected": plan["selected"],
                        }
                    except Exception as exc:  # noqa: BLE001 — surface solver errors in the UI
                        st.error(f"Plan reconstruction failed: {type(exc).__name__}: {exc}")

            if solver_note:
                st.caption(solver_note)
            bpso = bpso_params_for(dataset_size)
            st.caption(
                f"{len(findings):,} findings loaded. "
                f"BPSO params for this size: {bpso}; MILP time limit {milp_time_limit_for(dataset_size)}s."
            )

            q1, q2, q3, q4 = st.columns([2, 1, 1, 1])
            with q1:
                query = st.text_input("Search CVE / finding / asset")
            with q2:
                kev_only = st.checkbox("KEV only")
            with q3:
                min_cvss = st.slider("Min CVSS", 0.0, 10.0, 0.0, 0.5)
            with q4:
                selected_only = st.checkbox("Selected only", value=False, disabled=not compute)

            exposures = []
            if "exposure" in annotated.columns:
                options = sorted(annotated["exposure"].dropna().astype(str).unique())
                exposures = st.multiselect("Exposure", options, default=options)

            viewed = filter_findings(
                annotated,
                query=query,
                kev_only=kev_only,
                min_cvss=min_cvss if min_cvss > 0 else None,
                exposures=exposures or None,
                selected_only=selected_only,
            )
            show_cols = findings_display_columns(viewed)
            if "selected" in viewed.columns:
                show_cols = ["selected"] + [c for c in show_cols if c != "selected"]
            st.dataframe(
                viewed[show_cols] if show_cols else viewed,
                use_container_width=True,
                hide_index=True,
            )
            st.caption(f"Showing {len(viewed):,} of {len(annotated):,} findings.")

    with tab_sensitivity:
        st.subheader("Resource scenarios")
        resource_view = filter_result_rows(comparison, dataset_size=dataset_size, methods=selected_methods or None)
        if resource_view.empty:
            st.caption("No method-comparison rows for this dataset.")
        else:
            keep = [c for c in ("resource_scenario", "method", "pct_risk_reduction", "n_selected", "pct_budget_used", "pct_hours_used", "pct_downtime_used", "pct_kev_coverage", "feasible", "runtime_s") if c in resource_view.columns]
            st.dataframe(resource_view[keep], use_container_width=True, hide_index=True)
            pivot_src = successful_subset(resource_view)
            if not pivot_src.empty and "pct_risk_reduction" in pivot_src.columns:
                melted = pivot_src.dropna(subset=["pct_risk_reduction"])
                if melted.empty:
                    st.caption("Resource-scenario chart unavailable (missing values).")
                else:
                    fig = px.bar(
                        melted,
                        x="resource_scenario",
                        y="pct_risk_reduction",
                        color="method",
                        barmode="group",
                        labels={
                            "resource_scenario": "Resource scenario",
                            "pct_risk_reduction": "% risk reduction",
                            "method": "Method",
                        },
                    )
                    fig.update_layout(margin=dict(t=16, b=16))
                    st.plotly_chart(fig, use_container_width=True)

        st.subheader("Named risk weights")
        st.caption("Saved named-weight runs use n=100 and the medium resource scenario.")
        named = tables.get("sensitivity_named_weights", pd.DataFrame())
        named_f = filter_result_rows(named, methods=selected_methods or None)
        if named_f.empty:
            st.caption("Named-weight sensitivity table is missing or empty.")
        else:
            keep = [c for c in ("weight_scenario", "method", "pct_risk_reduction", "total_risk_reduction", "n_selected", "pct_kev_coverage", "feasible") if c in named_f.columns]
            st.dataframe(named_f[keep], use_container_width=True, hide_index=True)
            ok = successful_subset(named_f)
            if not ok.empty and "pct_risk_reduction" in ok.columns:
                melted = ok.dropna(subset=["pct_risk_reduction"])
                if melted.empty:
                    st.caption("Named-weight chart unavailable (missing values).")
                else:
                    fig = px.bar(
                        melted,
                        x="weight_scenario",
                        y="pct_risk_reduction",
                        color="method",
                        barmode="group",
                        labels={
                            "weight_scenario": "Weight scenario",
                            "pct_risk_reduction": "% risk reduction",
                            "method": "Method",
                        },
                    )
                    fig.update_layout(margin=dict(t=16, b=16))
                    st.plotly_chart(fig, use_container_width=True)
            named_fig = figure_path("sensitivity_named_weights.png")
            if named_fig is not None:
                st.image(str(named_fig), caption="Saved figure: named risk-weight sensitivity")

        st.subheader("Dirichlet weight samples")
        diri = tables.get("sensitivity_dirichlet", pd.DataFrame())
        if diri.empty:
            st.caption("Dirichlet sensitivity table is missing or empty.")
        else:
            ok = successful_subset(filter_result_rows(diri, methods=selected_methods or None))
            if ok.empty:
                st.caption("No successful Dirichlet rows for the selected methods.")
            else:
                summary = (
                    ok.groupby("method")["pct_risk_reduction"]
                    .agg(["count", "mean", "std", "min", "max"])
                    .reset_index()
                )
                st.dataframe(summary, use_container_width=True, hide_index=True)
                st.caption("Statistics are computed from saved Dirichlet rows; empty metrics are omitted.")
                diri_fig = figure_path("sensitivity_dirichlet.png")
                if diri_fig is not None:
                    st.image(str(diri_fig), caption="Saved figure: Dirichlet weight-vector sensitivity")

        st.subheader("EPSS missing-value handling")
        epss = tables.get("epss_handling", pd.DataFrame())
        epss_f = filter_result_rows(epss, methods=selected_methods or None)
        if epss_f.empty:
            st.caption("EPSS handling table is missing or empty.")
        else:
            keep = [c for c in ("dataset_size", "epss_strategy", "method", "n_findings", "pct_risk_reduction", "n_selected", "error_message") if c in epss_f.columns]
            st.dataframe(epss_f[keep], use_container_width=True, hide_index=True)
            notes = epss_f["error_message"].dropna().astype(str).unique() if "error_message" in epss_f.columns else []
            notes = [n for n in notes if n]
            if notes:
                st.info(notes[0])
            epss_fig = figure_path("epss_handling.png")
            if epss_fig is not None:
                st.image(str(epss_fig), caption="Saved figure: EPSS missing-data policy comparison")

        st.subheader("BPSO seed variability")
        bpso_sum = tables.get("bpso_summary", pd.DataFrame())
        if bpso_sum.empty:
            st.caption("BPSO summary table is missing or empty.")
        else:
            st.dataframe(bpso_sum, use_container_width=True, hide_index=True)
            bpso_fig = figure_path("bpso_variability.png")
            if bpso_fig is not None:
                st.image(str(bpso_fig), caption="Saved figure: BPSO variability across seeds")

        extra_figs = [p for p in list_figure_paths() if p.name not in {
            "method_comparison_risk.png",
            "sensitivity_named_weights.png",
            "sensitivity_dirichlet.png",
            "epss_handling.png",
            "bpso_variability.png",
        }]
        if extra_figs:
            with st.expander("Additional saved figures"):
                for path in extra_figs:
                    st.image(str(path), caption=path.name)

    with tab_export:
        st.subheader("Download CSVs")
        extra_meta = {
            "dashboard_dataset_size": dataset_size,
            "dashboard_resource_scenario": resource_scenario,
        }
        cmp_out = comparison_export(filtered, extra=extra_meta)
        st.download_button(
            "Experiment comparison (current filters)",
            data=dataframe_to_csv_bytes(cmp_out),
            file_name=f"vulnopt_comparison_n{dataset_size}_{resource_scenario}.csv",
            mime="text/csv",
            disabled=cmp_out.empty,
        )

        findings_now = _cached_findings(dataset_size)
        snap = findings_snapshot_export(
            findings_now,
            dataset_size=dataset_size,
            resource_scenario=resource_scenario,
        )
        st.download_button(
            "Finding-level details (current dataset, no selection flag)",
            data=dataframe_to_csv_bytes(snap),
            file_name=f"vulnopt_findings_n{dataset_size}_{resource_scenario}.csv",
            mime="text/csv",
            disabled=snap.empty,
        )

        last = st.session_state.get("last_plan")
        if not last:
            st.caption("Reconstruct a plan on the Finding inspector tab to export selected remediations.")
        else:
            plan_df = plan_export(
                last["selected"],
                method=last["method"],
                dataset_size=last["dataset_size"],
                resource_scenario=last["resource_scenario"],
                budget=last["budget"],
                max_hours=last["hours"],
                max_downtime=last["downtime"],
                weight_scenario=last["weight_scenario"],
                apply_downtime=last["apply_downtime"],
            )
            detail_df = finding_detail_export(
                last["annotated"],
                method=last["method"],
                dataset_size=last["dataset_size"],
                resource_scenario=last["resource_scenario"],
            )
            st.download_button(
                f"Selected remediation plan ({last['method']})",
                data=dataframe_to_csv_bytes(plan_df),
                file_name=f"vulnopt_plan_{last['method']}_n{last['dataset_size']}_{last['resource_scenario']}.csv",
                mime="text/csv",
                disabled=plan_df.empty,
            )
            st.download_button(
                f"Finding-level details ({last['method']})",
                data=dataframe_to_csv_bytes(detail_df),
                file_name=f"vulnopt_findings_{last['method']}_n{last['dataset_size']}_{last['resource_scenario']}.csv",
                mime="text/csv",
                disabled=detail_df.empty,
            )

        named = tables.get("sensitivity_named_weights", pd.DataFrame())
        if not named.empty:
            st.download_button(
                "Named-weight sensitivity table",
                data=dataframe_to_csv_bytes(named),
                file_name="sensitivity_named_weights.csv",
                mime="text/csv",
            )


if __name__ == "__main__":
    main()
