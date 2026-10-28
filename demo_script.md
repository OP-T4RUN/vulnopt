# VulnOpt demonstration script

About 10–12 minutes. Use saved Part A results; do not present invented numbers.

## 0. Setup (before the demo)

```bash
cd vulnopt
python -m venv venv
# Windows
.\venv\Scripts\activate
# macOS / Linux
source venv/bin/activate

pip install -r requirements.txt
pytest
```

Expected: the existing suite plus `tests/test_dashboard.py` pass. Experiment CSVs should already be present under `results/tables/`. Regenerating them is optional and slow:

```bash
python scripts/run_experiments.py --quick
# full grid (only if you intend to wait for BPSO × Dirichlet):
# python scripts/run_experiments.py
```

## 1. Introduction (about 1 minute)

VulnOpt is a **decision-support** prototype: given a list of vulnerability findings, it recommends which ones to remediate when budget, engineer-hours, and downtime are limited.

- Public inputs: NVD CVSS, FIRST EPSS, CISA KEV.
- Enterprise inputs: synthetic assets and costs (not a live scan).
- Score \(R_i\) mixes severity, exploit likelihood, KEV, criticality, and exposure.
- Four methods on the same knapsack: CVSS-Greedy, Risk-Greedy, MILP (CBC), BPSO.

State clearly: the tool does **not** scan networks or install patches.

## 2. Launch the dashboard

```bash
streamlit run streamlit_app.py
```

The app reads `results/tables/` and `data/synthetic/scored_findings_*.csv`. No API keys are required for this demo.

## 3. Select a dataset and scenario

In the sidebar:

1. Dataset size **1,000** (default benchmark; 100 and 5,000 are also available).
2. Resource scenario **medium** (50% of total cost, hours, and downtime).
3. Leave downtime enabled; note that budget and hours default to that scenario.
4. Weight scenario **default**.
5. Keep all four methods selected.

Point out that overview numbers come from **saved CSVs**, not from a silent re-solve.

## 4. Compare optimization methods

Open **Results overview**.

Walk through the table:

- **% risk reduction**, findings selected, budget/hours/downtime used, KEV coverage, feasibility, runtime.
- MILP block: solver status **Optimal**, optimality gap **0** on these runs.
- If you switch to n = 100 and **tight**, show that MILP is stored as infeasible with a \(10^{-13}\) hours remainder — report the file, do not “fix” it live.

On n = 1,000 medium (saved):

- CVSS-Greedy ≈ 51.93% risk reduction, 456 selected
- Risk-Greedy ≈ 52.17%, 440 selected
- MILP ≈ 67.72%, 662 selected
- BPSO (seed 42) ≈ 53.75%, 535 selected

MILP is ahead; greedy and BPSO are close to each other. Use the bar charts only as a view of the same table.

## 5. Inspect selected findings

Open **Finding inspector**.

1. Search a CVE from the table (or filter **KEV only** — four KEV rows at n = 1,000).
2. Check **Highlight findings selected by MILP** (or Risk-Greedy for a faster solve).
3. Enable **Selected only** and show CVE, CVSS, EPSS, KEV, risk score, asset context, cost, hours, downtime.
4. Switch inspector method to **CVSS-Greedy** and note that the selected set changes.

Say that this plan is reconstructed with `run_solver` using current sidebar capacities; the overview metrics remain the Part A CSV.

## 6. Sensitivity analysis

Open **Sensitivity**.

1. **Resource scenarios** — tight / medium / relaxed for the current dataset. Risk reduction rises as capacities increase; MILP stays highest in the saved grid.
2. **Named weights** — default vs threat-focused vs asset-focused (n = 100, medium). Percentages move because \(R_i\) is reweighted; MILP remains highest in each block.
3. **Dirichlet** — 100 random weight vectors; show the mean/std table, not a single cherry-picked vector.
4. **EPSS handling** — read the stored note: this snapshot has **zero missing EPSS**, so the three policies match. Do not claim a robustness win.
5. **BPSO seeds** — 20 feasible runs; standard deviation of % risk reduction is small (about 0.7 on n = 100, 0.16 on n = 5,000).

## 7. Export a remediation plan

1. On Finding inspector, reconstruct a plan if you have not already.
2. Open **CSV export**.
3. Download **Experiment comparison (current filters)** — should list the four methods for the selected n and scenario.
4. Download **Selected remediation plan** — selected rows plus method and capacity metadata.
5. Download **Finding-level details** — full instance with a `selected` flag.

Open one CSV in a spreadsheet and confirm row counts match the UI.

## 8. Conclusions and limitations

**Supported by the saved tables**

- MILP achieved the largest modeled risk reduction on every comparison cell, with CBC Optimal / gap 0.
- Contextual greedy is cheap and often slightly better than CVSS greedy, but not always (n = 100 relaxed favors CVSS-Greedy).
- BPSO helps on n = 100; with the documented swarm limits it lags MILP, and on large relaxed instances it also lags greedy.

**Say out loud**

- Scores and costs are modeled, not measured losses.
- Assets and remediation costs are synthetic.
- EPSS-imputation sensitivity is a no-op on this snapshot.
- The dashboard will not patch production systems.

Stop. Take questions from the comparison table and the inspector, not from memory of unpublished runs.
