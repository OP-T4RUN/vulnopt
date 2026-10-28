"""
Generate VulnOpt evaluation tables and figures.

Usage (from the repository root):

    python scripts/run_experiments.py
    python scripts/run_experiments.py --quick
    python scripts/run_experiments.py --datasets 100 1000 --bpso-seeds 20
"""

from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from vulnopt.experiments import main


if __name__ == "__main__":
    raise SystemExit(main())
