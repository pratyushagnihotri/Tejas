"""Loads SemBench's Movie scenario demo data (sf_2000: 2000 reviews, already
committed in github.com/SemBench/SemBench at files/movie/data/sf_2000/) and
prepares the pieces sembench_tasks.py needs.

The model only ever sees VISIBLE_FIELDS (no scoreSentiment) — that column is
read here only to compute gold answers, mirroring SemBench's own design where
the hidden ground-truth column is invisible to the system under test.
"""

import csv
import random

from tejas.paths import DATA_DIR

REVIEWS_CSV = str(DATA_DIR / "Reviews.csv")

VISIBLE_FIELDS = ["reviewId", "criticName", "originalScore", "reviewText"]
SAMPLE_SIZE = 24
MOVIE_ID = "taken_3"
REVIEW_TEXT_MAXLEN = 500


def load_reviews_for_movie(movie_id=MOVIE_ID):
    with open(REVIEWS_CSV, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = [r for r in reader if r["id"] == movie_id]
    return rows


def sample_reviews(rows, n=SAMPLE_SIZE, seed=42):
    rows_sorted = sorted(rows, key=lambda r: r["reviewId"])
    rng = random.Random(seed)
    return rng.sample(rows_sorted, min(n, len(rows_sorted)))


def render_reviews_block(sample):
    lines = []
    for r in sample:
        text = r["reviewText"].strip().replace("\n", " ")
        if len(text) > REVIEW_TEXT_MAXLEN:
            text = text[:REVIEW_TEXT_MAXLEN] + "..."
        lines.append(
            f"- reviewId: {r['reviewId']} | critic: {r['criticName']} | "
            f"score: {r['originalScore']} | text: \"{text}\""
        )
    return "\n".join(lines)


def gold_counts(sample):
    positive_ids = [r["reviewId"] for r in sample if r["scoreSentiment"] == "POSITIVE"]
    negative_ids = [r["reviewId"] for r in sample if r["scoreSentiment"] == "NEGATIVE"]
    return {
        "sample_ids": [r["reviewId"] for r in sample],
        "positive_ids": positive_ids,
        "negative_ids": negative_ids,
        "positive_count": len(positive_ids),
        "negative_count": len(negative_ids),
        "ratio": round(len(positive_ids) / len(sample), 4) if sample else 0.0,
    }


def get_movie_sample(movie_id=MOVIE_ID, n=SAMPLE_SIZE, seed=42):
    rows = load_reviews_for_movie(movie_id)
    sample = sample_reviews(rows, n=n, seed=seed)
    return sample, gold_counts(sample)


if __name__ == "__main__":
    sample, gold = get_movie_sample()
    print(f"Sampled {len(sample)} reviews for '{MOVIE_ID}'")
    print(f"positive={gold['positive_count']} negative={gold['negative_count']} ratio={gold['ratio']}")
    print(render_reviews_block(sample[:2]))
