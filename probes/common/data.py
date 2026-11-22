"""SemBench Movie data (scale factor 2000) with the ground-truth sentiment labels."""

from __future__ import annotations

import csv
import os
from pathlib import Path
from typing import Dict, List

csv.field_size_limit(10**8)

DEFAULT_SEMBENCH = Path(
    os.environ.get(
        "SEMBENCH_MOVIE_DIR",
        Path.home() / "AI_Agent/greenQueryLab/third_party/sembench/files/movie/data/sf_2000",
    )
)


def load_reviews(path: Path = DEFAULT_SEMBENCH) -> List[Dict]:
    with open(Path(path) / "Reviews.csv", newline="") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r["label_positive"] = r["scoreSentiment"].upper() == "POSITIVE"
    return rows


# Columns an engine that serialises the whole tuple would put in the prompt.
# reviewState (fresh/rotten) and originalScore are left out on purpose: they leak
# the sentiment label, so including them would change answer quality and the
# projection lever would no longer be a quality-neutral comparison.
ROW_COLUMNS = [
    "id", "reviewId", "creationDate", "criticName", "isTopCritic",
    "publicatioName", "reviewText", "reviewUrl",
]


def row_text(r: Dict, projection: str) -> str:
    """projection='needed' -> only reviewText; 'full_row' -> every column (label excluded)."""
    if projection == "needed":
        return r["reviewText"]
    return "\n".join(f"{c}: {r[c]}" for c in ROW_COLUMNS)
