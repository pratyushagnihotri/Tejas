"""Single source of truth for filesystem paths, shared across all four
pillars. Replaces the per-file `os.path.dirname(os.path.dirname(...))`
arithmetic the original flat scripts each recomputed independently (which
was depth-dependent and broke if a file moved) with one place to look.
"""

from pathlib import Path

# src/tejas/paths.py -> src/tejas -> src -> repo root
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
RESULTS_DIR = REPO_ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
DATA_DIR = Path(__file__).resolve().parent / "benchmarking" / "data"
COMPANY_DB = REPO_ROOT / "company.db"
