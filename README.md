# VulnOpt — Risk-Aware Vulnerability Remediation Prioritization

**VulnOpt** is a Python-based decision-support prototype designed to recommend which known software vulnerabilities should be prioritized for remediation when an organization operates under hard resource constraints (limited budget, engineer-hours, and acceptable maintenance downtime).

> [!IMPORTANT]
> **Safety and Scope Boundary:** VulnOpt is strictly an analytical decision-support prototype for human review. It does **not** scan networks, probe live targets, connect to production infrastructure, or deploy patches.

---

## 1. Project Overview & Problem Statement

Organizations routinely identify hundreds or thousands of vulnerability findings across enterprise assets. However, operational resources are inherently constrained:
- A finite remediation budget ($B$).
- A limited number of engineer-hours ($H$).
- Tight maintenance and downtime windows ($W$).

### Why CVSS-only Ranking Fails
Standard vulnerability management commonly prioritizes findings by sorting solely by CVSS base score. This approach has critical limitations:
1. **Ignores Threat Dynamics:** A high CVSS flaw may have near-zero active exploit probability, while a medium CVSS flaw may be actively leveraged in ransomware campaigns (CISA KEV) with a high 30-day exploit likelihood (EPSS).
2. **Ignores Business Context:** Severity alone does not reflect asset criticality (e.g., core database vs. internal development sandbox) or internet exposure.
3. **Suboptimal Combinatorial Selection:** Picking vulnerabilities greedily based on severity does not solve the multi-dimensional knapsack problem (MKP) efficiently, leaving significant modeled risk reduction unrealized within the same resource envelope.

---

## 2. Team Roles & Module Architecture

| Role | Student | Responsibilities | Key Modules |
|---|---|---|---|
| **Student 1** | Public Data Pipeline | NVD, EPSS, KEV collection; cleaning; provenance; data summary | `vulnopt/data_collection.py`, `vulnopt/data_cleaning.py` |
| **Student 2** | Synthetic Data & Risk Model | Simulated enterprise assets; risk scoring formula; sensitivity | `vulnopt/synthetic_data.py`, `vulnopt/risk_model.py` |
| **Student 3** | Optimization Engine | CVSS & contextual greedy baselines; exact MILP; BPSO metaheuristic | `vulnopt/constraints.py`, `vulnopt/baselines.py`, `vulnopt/milp_solver.py`, `vulnopt/bpso.py` |
| **Student 4** | Evaluation & Dashboard | Metrics calculation; experiment runner; Streamlit dashboard; report | `vulnopt/metrics.py`, `vulnopt/experiments.py`, `streamlit_app.py` |

---

## 3. Public Vulnerability Intelligence & Provenance

The project fuses three authoritative, credible public sources using standardized `cve_id` records:

1. **CISA Known Exploited Vulnerabilities (KEV) Catalog:**
   - **Source:** [CISA KEV JSON Feed](https://www.cisa.gov/sites/default/files/feeds/known_exploited_vulnerabilities.json)
   - **Fields:** `cve_id`, `is_kev`, `kev_date_added`, `kev_ransomware`
2. **FIRST Exploit Prediction Scoring System (EPSS):**
   - **Source:** [FIRST EPSS Bulk Scores](https://epss.empiricalsecurity.com/epss_scores-current.csv.gz)
   - **Fields:** `cve_id`, `epss` (probability), `epss_percentile`, `epss_date`
3. **National Vulnerability Database (NVD - NIST):**
   - **Source:** [NVD REST API 2.0](https://services.nvd.nist.gov/rest/json/cves/2.0)
   - **Fields:** `cve_id`, `published_date`, `description`, `cvss_base`, `cvss_version`, `cvss_vector`, `cvss_severity`, `cwe_ids`

### Data Quality & Missing Data Policy
- **Missing CVSS:** Excluded from the scored comparison (175 records in current snapshot), preserving data integrity.
- **Missing EPSS:** Tracked with explicit `epss_missing` boolean flag; imputed as `0.0` for baseline scoring and subjected to sensitivity testing.
- **KEV State:** Evaluated against complete downloaded snapshot; negative match is confirmed only after full parsing.

---

## 4. Setup and Installation

### Prerequisites
- Python 3.11+
- Git

### Installation
```bash
# Clone the repository
git clone <repo-url>
cd vulnopt

# Create and activate virtual environment
python -m venv venv
# On Windows:
.\venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Running Data Collection (Student 1)
```bash
# Set your NVD API key in .env or run with existing snapshot
python scripts/fetch_data.py --max-records 5000
```

### Running Tests
```bash
pytest
```

