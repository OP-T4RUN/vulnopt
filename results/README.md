# Part A → Part B result contract

The Streamlit dashboard should load these files. Do not recompute solver metrics in the dashboard.

## Generate

```bash
python scripts/run_experiments.py
```

## Test

```bash
pytest
```

## Tables

See the repository README section “Part B dashboard contract” for column names.

- `tables/method_comparison.csv`
- `tables/resource_utilization.csv`
- `tables/feasibility.csv`
- `tables/runtime.csv`
- `tables/bpso_seeds.csv`
- `tables/bpso_summary.csv`
- `tables/sensitivity_named_weights.csv`
- `tables/sensitivity_dirichlet.csv`
- `tables/epss_handling.csv`
- `tables/experiment_failures.csv`
- `tables/schema_example.csv`
- `tables/experiment_config.json`

## Figures

PNG comparison plots in `figures/`, generated from the CSVs above.
