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

## 2. System Architecture & Modules

| Subsystem | Responsibilities | Key Modules |
|---|---|---|
| **Public Data Pipeline** | NVD, EPSS, KEV collection; cleaning; provenance; data summary | `vulnopt/data_collection.py`, `vulnopt/data_cleaning.py` |
| **Enterprise Context & Risk Model** | Simulated enterprise assets; risk scoring formula; sensitivity | `vulnopt/synthetic_data.py`, `vulnopt/risk_model.py` |
| **Optimization Engine** | CVSS & contextual greedy baselines; exact MILP; BPSO metaheuristic | `vulnopt/constraints.py`, `vulnopt/baselines.py`, `vulnopt/milp_solver.py`, `vulnopt/bpso.py` |
| **Evaluation & Dashboard** | Metrics calculation; experiment runner; Streamlit dashboard; report | `vulnopt/metrics.py`, `vulnopt/experiments.py`, `streamlit_app.py` |

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

## 4. Synthetic Enterprise Context & Risk Scoring Model

### Synthetic Enterprise Data Strategy
Because operational asset inventories and patching costs are proprietary, all organizational context is modeled synthetically and labeled as such:
* **Assets (`data/synthetic/assets.csv`):** Simulated devices with `criticality` (1–5 scale), `exposure` (`internet`, `internal`, `restricted`), and environment (`production`, `staging`, `development`).
* **Hybrid Device Matching:** Assigns CVEs using keyword patterns (e.g., WordPress/PHP to web servers, SQL/DBMS to databases, Active Directory/OAuth to identity services) with deterministic random fallback.
* **Remediation Resource Estimation:** Patch costs ($C_i$), engineer-hours ($T_i$), and downtime ($D_i$) are generated via right-skewed Log-normal and Exponential distributions with positive average correlation to asset criticality and CVSS severity.
* **Benchmark Scales:**
  * Small: 100 findings
  * Medium: 1,000 findings (Default benchmark)
  * Large: 5,000 findings

### Configurable Risk Priority Score ($R_i$)
The model combines threat intelligence with business impact into an explainable 0–100 priority score:

$$Q_i = w_s \cdot S_i + w_e \cdot E_i + w_k \cdot K_i + w_a \cdot A_i + w_x \cdot X_i$$

$$R_i = 100 \times Q_i$$

Where normalized components are bounded in $[0, 1]$:
* **$S_i$ (Severity):** $\text{CVSS}_i / 10.0$
* **$E_i$ (Exploit Likelihood):** $\text{EPSS}_i \in [0, 1]$
* **$K_i$ (Active Exploitation):** $1.0$ if present in CISA KEV catalog, else $0.0$
* **$A_i$ (Asset Criticality):** $(\text{criticality}_i - 1) / 4.0$
* **$X_i$ (Exposure):** $1.0$ (internet), $0.5$ (internal), $0.2$ (restricted)

**Default Weights:**
* CVSS ($w_s$): `0.25`
* EPSS ($w_e$): `0.25`
* KEV ($w_k$): `0.25`
* Criticality ($w_a$): `0.15`
* Exposure ($w_x$): `0.10`

Alternative sensitivity scenarios (Threat-Focused, Asset-Focused, and Dirichlet random sampling) are supported.

---

## 5. Setup and Installation

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

### Running Data Collection
```bash
# Set your NVD API key in .env or run with existing snapshot
python scripts/fetch_data.py --max-records 5000
```

### Running Tests
```bash
pytest
```

