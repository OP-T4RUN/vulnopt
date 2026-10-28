# VulnOpt Final Report

Risk-aware vulnerability remediation under resource constraints.

This report uses the implementations in `vulnopt/` and the Part A experiment tables in `results/tables/`. Figures in `results/figures/` were generated from those tables. No metric or solver result in this document is invented; where a table is empty or a cell is missing, that gap is stated.

---

## 1. Problem statement and objectives

Organizations accumulate large volumes of vulnerability findings but cannot remediate all of them. Remediation is limited by:

- a monetary patch budget \(B\)
- available engineer-hours \(H\)
- an acceptable downtime window \(W\)

Ranking by CVSS base score alone ignores exploit likelihood (EPSS), confirmed in-the-wild exploitation (CISA KEV), and asset context (criticality and exposure). Greedy ranking by a single score also fails to solve the underlying multi-dimensional knapsack: a high-score finding that consumes most of a scarce resource can displace a better combination of cheaper findings.

**Objectives**

1. Fuse public CVE intelligence with synthetic enterprise context into an explainable risk score \(R_i\).
2. Select a feasible subset of findings that maximizes modeled risk reduction subject to \(B\), \(H\), and \(W\).
3. Compare four methods on the same instances: CVSS-Greedy, Risk-Greedy, MILP, and BPSO.
4. Report resource use, KEV coverage, feasibility, runtime, and sensitivity to resources, weights, and EPSS missing-data policy.
5. Present results in a local Streamlit dashboard for inspection and CSV export.

VulnOpt is a decision-support prototype. It does not scan networks, probe targets, or deploy patches.

---

## 2. System architecture and data sources

| Layer | Role | Code / artifacts |
|---|---|---|
| Public intelligence | NVD CVSS, FIRST EPSS, CISA KEV | `vulnopt/data_collection.py`, `vulnopt/data_cleaning.py`, `data/processed/` |
| Synthetic enterprise | Assets, finding assignments, patch cost / hours / downtime | `vulnopt/synthetic_data.py`, `data/synthetic/` |
| Risk model | Weighted \(R_i\) in \([0, 100]\) | `vulnopt/risk_model.py` |
| Optimization | Constraints, greedy baselines, CBC MILP, BPSO | `vulnopt/constraints.py`, `baselines.py`, `milp_solver.py`, `bpso.py` |
| Evaluation | Shared metrics and experiment grid | `vulnopt/metrics.py`, `vulnopt/experiments.py`, `scripts/run_experiments.py` |
| Dashboard / report | Saved-result UI, exports, this document | `streamlit_app.py`, `vulnopt/dashboard.py`, `final_report.md` |

**Public sources (joined on `cve_id`)**

- CISA KEV JSON feed
- FIRST EPSS bulk scores
- NVD REST API 2.0 (CVSS, CWE, description)

The current processed snapshot excludes records with missing CVSS. Missing EPSS is flagged (`epss_missing`) and imputed as 0.0 in the scoring pipeline. In the scored benchmark files used here, **every finding has an EPSS value** (`epss_missing` is false throughout).

**Synthetic context**

Asset inventories and patch costs are not taken from a live enterprise. `data/synthetic/assets.csv` and `finding_assignments_*.csv` are labeled synthetic. Benchmark scored tables:

- `data/synthetic/scored_findings_100.csv`
- `data/synthetic/scored_findings_1000.csv`
- `data/synthetic/scored_findings_5000.csv`

---

## 3. Risk model and optimization methods

### Risk score

\[
Q_i = w_s S_i + w_e E_i + w_k K_i + w_a A_i + w_x X_i, \quad R_i = 100 \cdot Q_i
\]

| Component | Definition |
|---|---|
| \(S_i\) | \(\mathrm{CVSS}_i / 10\) |
| \(E_i\) | EPSS in \([0, 1]\) |
| \(K_i\) | 1 if CISA KEV, else 0 |
| \(A_i\) | \((\mathrm{criticality}_i - 1) / 4\) |
| \(X_i\) | 1.0 internet, 0.5 internal, 0.2 restricted |

Default weights: CVSS 0.25, EPSS 0.25, KEV 0.25, criticality 0.15, exposure 0.10. Named alternatives: `threat_focused` and `asset_focused` (see `vulnopt/risk_model.py`).

### Optimization

All methods receive the same `OptimizationProblem`: maximize \(\sum R_i x_i\) subject to budget, hours, and downtime (`vulnopt/constraints.py`).

1. **CVSS-Greedy** — sort by CVSS base score; accept a finding if it fits remaining capacities.
2. **Risk-Greedy** — same knapsack greedy, ordered by \(R_i\).
3. **MILP** — binary linear program solved with PuLP/CBC; time limits 30 / 90 / 180 seconds for n = 100 / 1,000 / 5,000.
4. **BPSO** — binary particle swarm with repair to feasibility. Representative comparison seed is 42. Swarm size and iterations scale down with n (30×200, 25×100, 20×50). Twenty additional seeds (0–19) are recorded on the medium-resource scenario.

Resource capacities for named scenarios are **30% / 50% / 70%** of the instance totals for cost, hours, and downtime independently (`compute_default_budgets`).

---

## 4. Evaluation metrics and experimental methodology

Shared definitions live in `vulnopt/metrics.py` (`RESULT_COLUMNS`). Every method is scored with the same functions.

| Metric | Definition |
|---|---|
| Total risk reduction | \(\sum_i R_i x_i\) |
| % risk reduction | \(100 \times \sum R_i x_i / \sum R_i\) over **all** candidate findings; NaN if the denominator is 0 |
| Resource consumed / remaining / % used | Dot product with \(x\); % used is NaN if the limit is 0 |
| KEV coverage | \(100 \times\) selected KEV / KEV in the instance; NaN if there are no KEV rows |
| Feasibility | Same predicate as `constraints.is_feasible` |
| Runtime | Wall-clock seconds around the solver call |
| MILP status / gap | Passed through from `MILPResult`; empty/NaN for greedy and BPSO |

**Grid (Part A, `experiment_config.json`)**

- Method comparison: n ∈ {100, 1000, 5000} × {tight, medium, relaxed} × 4 methods
- BPSO multi-seed: medium resources, seeds 0–19, all three sizes
- Named weights: n = 100, medium, default / threat_focused / asset_focused
- Dirichlet: 100 weight vectors, seed 42, n = 100, medium
- EPSS policies: zero / median / exclusion on n = 100 and n = 1000, medium

`results/tables/experiment_failures.csv` is header-only. `failed_experiment_count` in the manifest is 0. All comparison rows have `ran_successfully=true`.

---

## 5. Results: comparison of the four methods

Source: `results/tables/method_comparison.csv`. Values below are **percentage risk reduction** unless noted. KEV denominators are small (1, 4, and 19 KEV findings on n = 100 / 1,000 / 5,000).

### n = 100

| Scenario | CVSS-Greedy | Risk-Greedy | MILP | BPSO (seed 42) |
|---|---:|---:|---:|---:|
| Tight | 31.52 | 34.41 | 47.31 | 42.05 |
| Medium | 53.25 | 53.82 | 67.95 | 60.91 |
| Relaxed | 74.13 | 71.99 | 83.23 | 74.27 |

Selected counts (tight / medium / relaxed): CVSS 25 / 47 / 68; Risk 26 / 44 / 63; MILP 43 / 65 / 82; BPSO 38 / 57 / 71.

MILP CBC status is **Optimal** with **gap 0.0** on every comparison row, including n = 5,000.

On **n = 100, tight**, MILP is recorded as **`feasible=False`**. Hours remaining is about \(-1.1 \times 10^{-13}\) (budget and downtime remain non-negative). This is a floating-point boundary case in `is_feasible`, not a missing run. The other 35 comparison rows are feasible.

### n = 1,000

| Scenario | CVSS-Greedy | Risk-Greedy | MILP | BPSO |
|---|---:|---:|---:|---:|
| Tight | 31.97 | 32.37 | 47.26 | 35.80 |
| Medium | 51.93 | 52.17 | 67.72 | 53.75 |
| Relaxed | 71.41 | 71.93 | 84.08 | 58.55 |

KEV coverage on tight: CVSS 75%, Risk 100%, MILP 100%, BPSO 50% (2 of 4 KEV selected). On medium, BPSO covers 50% while the other three cover 100%.

### n = 5,000

| Scenario | CVSS-Greedy | Risk-Greedy | MILP | BPSO |
|---|---:|---:|---:|---:|
| Tight | 31.27 | 31.62 | 47.88 | 32.91 |
| Medium | 52.07 | 52.37 | 68.01 | 51.76 |
| Relaxed | 70.82 | 71.72 | 83.83 | 53.51 |

On the large instance, BPSO does not close the gap to MILP. On medium, BPSO’s risk reduction (51.76%) is slightly **below** both greedy methods. On relaxed, BPSO uses only about 76% of budget and 76% of hours versus nearly 100% for MILP, which is consistent with a weaker feasible solution rather than a tighter constraint set.

### Runtime (wall-clock seconds, same table)

Greedy methods are under 0.01 s at n = 5,000. MILP stays under about 2.2 s in this grid (largest recorded comparison runtime 2.12 s on n = 1,000 tight). BPSO is the slowest on tight n = 5,000 (**47.58 s**) with the configured 20 particles and 50 iterations; other BPSO comparison runtimes are lower (for example 3.32 s on n = 5,000 medium).

---

## 6. Resource utilization and sensitivity

### Utilization

`results/tables/resource_utilization.csv` is a column subset of the method-comparison runs. Typical pattern: MILP fills hours and budget to nearly 100% of the limit. Greedy methods often leave more downtime slack (for example n = 1,000 medium: Risk-Greedy 84.1% downtime vs MILP 99.96%). BPSO on large relaxed instances leaves substantial unused budget (n = 5,000 relaxed: 76.4% budget, 75.6% hours, 74.4% downtime).

### Named weights (n = 100, medium)

Source: `results/tables/sensitivity_named_weights.csv`. Percentage risk reduction:

| Weights | CVSS-Greedy | Risk-Greedy | MILP | BPSO |
|---|---:|---:|---:|---:|
| default | 53.25 | 53.82 | 67.95 | 60.91 |
| threat_focused | 54.94 | 56.00 | 69.95 | 63.17 |
| asset_focused | 50.14 | 59.54 | 68.88 | 60.96 |

Absolute \(R_i\) totals change with the weights (unmitigated risk is 3219.54 default, 1800.35 threat-focused, 4407.01 asset-focused), so percentage reduction is the comparable column. MILP remains highest in each named scenario. Asset-focused BPSO selected **0 of 1** KEV finding (`pct_kev_coverage = 0`); the other methods in that block selected the KEV finding.

### Dirichlet samples (n = 100, medium, 100 vectors)

Source: `results/tables/sensitivity_dirichlet.csv` (400 successful rows). Mean percentage risk reduction by method (computed from that file):

| Method | Mean % | Std | Min | Max |
|---|---:|---:|---:|---:|
| CVSS-Greedy | 53.31 | 1.85 | 48.97 | 57.19 |
| Risk-Greedy | 56.55 | 3.21 | 51.17 | 64.31 |
| MILP | 68.85 | 1.22 | 66.97 | 73.69 |
| BPSO | 62.07 | 1.37 | 59.66 | 67.42 |

On mean percentage risk reduction, MILP > BPSO > Risk-Greedy > CVSS-Greedy. Per-sample rows are in the CSV.

### EPSS missing-value policies

Source: `results/tables/epss_handling.csv`. Every row notes that **the snapshot has zero missing EPSS values**, so zero imputation, median imputation, and exclusion produce the **same candidate set** as the pipeline default. Risk-reduction numbers match the corresponding method-comparison medium rows for n = 100 and n = 1,000. The experiment records that identity rather than fabricating a difference.

### BPSO seed variability (medium resources, 20 seeds)

Source: `results/tables/bpso_summary.csv`.

| n | Mean % risk reduction | Std | Min | Max | Mean selected | Feasible runs |
|---|---:|---:|---:|---:|---:|---:|
| 100 | 61.39 | 0.70 | 59.92 | 62.80 | 58.45 | 20 / 20 |
| 1,000 | 53.87 | 0.23 | 53.46 | 54.24 | 531.8 | 20 / 20 |
| 5,000 | 51.62 | 0.16 | 51.37 | 51.89 | 2569.8 | 20 / 20 |

Seed-42 comparison values sit inside these ranges (n = 100: 60.91%; n = 1,000: 53.75%; n = 5,000: 51.76%). Variability is small relative to the MILP–greedy gap.

---

## 7. Key findings, limitations, assumptions, and conclusion

### Findings supported by the tables

1. **MILP obtained the highest modeled risk reduction** in every method-comparison cell, with CBC reporting Optimal and gap 0.
2. **Risk-Greedy usually beats CVSS-Greedy**, but not always (n = 100 relaxed: 71.99% vs 74.13%). Ordering by \(R_i\) is not a substitute for combinatorial optimization.
3. **BPSO beat both greedy methods on n = 100** in all three resource scenarios, and beat them on n = 1,000 tight/medium, but **fell behind greedy on n = 1,000 relaxed and on n = 5,000 medium/relaxed** with the configured swarm size and iteration caps.
4. **KEV coverage is not the optimization objective.** Methods can diverge on KEV when few KEV rows exist; asset-focused BPSO on n = 100 missed the single KEV finding.
5. **EPSS policy experiments are uninformative on this snapshot** because no EPSS values are missing.
6. One MILP row is flagged infeasible because of a \(10^{-13}\) hours overshoot; treat that flag as numerical, and do not convert it to a fabricated slack of zero.

### Assumptions

- Synthetic costs, hours, downtime, criticality, and exposure stand in for unpublished enterprise data.
- \(R_i\) is a linear weighted score, not an empirical loss or exploit-count model.
- Default EPSS imputation is 0.0 when values are missing; that branch is unused in these scored files.
- BPSO is stochastic; comparison plots use seed 42 plus a 20-seed summary on medium resources.
- Dashboard plan reconstruction re-invokes existing solvers; it does not replay serialized \(x\) vectors from Part A (those vectors were not stored).

### Limitations

- No live asset inventory, ticket history, or change-window calendar.
- CBC with the recorded time limits found proven optimal solutions here; that need not hold for other instances or solvers.
- BPSO hyperparameters were reduced on large n to bound runtime; weaker large-n BPSO solutions should not be read as a general ranking of metaheuristics.
- n = 100 has a single KEV finding, so KEV percentages are coarse.
- Optional NSGA-II multi-objective search was not implemented.
- The dashboard does not deploy patches or contact external APIs at runtime.

### Conclusion

On the committed benchmark, an exact MILP formulation of the three-resource knapsack produced more modeled risk reduction than CVSS-Greedy, Risk-Greedy, and BPSO, at modest CBC runtimes. Contextual greedy is a cheap baseline and sometimes beats CVSS-only greedy, but it is not consistently near-optimal. BPSO is useful on the small instance and as a feasible heuristic, but with the documented particle/iteration limits it does not match MILP on the 1,000- and 5,000-finding tables. Sensitivity to named and Dirichlet weights preserves that method ordering on n = 100 medium. EPSS missing-data handling could not be differentiated on this snapshot.

The Streamlit app exposes these saved tables, reconstructs inspectable plans with the same solvers, and exports CSVs without altering Part A results by default.

---

## Appendix: artifact map

| Path | Contents |
|---|---|
| `results/tables/method_comparison.csv` | Primary four-method grid |
| `results/tables/resource_utilization.csv` | Utilization columns from that grid |
| `results/tables/feasibility.csv` | Feasibility and MILP status |
| `results/tables/runtime.csv` | Runtimes |
| `results/tables/bpso_seeds.csv` / `bpso_summary.csv` | 20-seed BPSO |
| `results/tables/sensitivity_named_weights.csv` | Three named weight vectors |
| `results/tables/sensitivity_dirichlet.csv` | 100 Dirichlet vectors |
| `results/tables/epss_handling.csv` | Three EPSS policies |
| `results/figures/*.png` | Plots generated from the CSVs |
| `streamlit_app.py` | Dashboard |
