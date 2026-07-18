"""Semantic-reasoning tasks derived from SemBench's Movie scenario
(github.com/SemBench/SemBench), corresponding to their Q2/Q3/Q4/Q8. Unlike
tasks.py's SQL tasks, the model must read actual review text and judge
sentiment itself — there's no column it can filter on.

Widened from the original single-movie version to 2 movies x 2 independent
samples x 4 query types = 16 tasks:
  - taken_3 (120 reviews, 11.7% positive — heavily skewed) and
    ant_man_and_the_wasp_quantumania (256 reviews, 51.6% positive — balanced)
    let us separate "models fail from genuine reasoning limits" from "models
    fail because the sample is an extreme-imbalance needle-in-a-haystack
    problem" — a confound the single-movie version couldn't rule out.
  - 2 independent samples per movie (different seeds) give the semantic tier
    real run-to-run variance to look at, the same role --repeats plays for
    the routing-policy comparison.

Gold answers are computed from the real scoreSentiment label (hidden from the
model) over the same review sample the model is shown, so candidate and gold
are always comparable.
"""

from tejas.benchmarking.load_sembench_movie import get_movie_sample, render_reviews_block

MOVIES = [
    {"movie_id": "taken_3", "short": "taken3"},
    {"movie_id": "ant_man_and_the_wasp_quantumania", "short": "antman"},
]
SEEDS = [42, 43]


def _build_preamble(movie_id, n):
    return (
        f"You are analyzing movie reviews for the movie '{movie_id}'. Below is "
        f"a sample of {n} reviews (reviewId, critic name, original score, and "
        "review text). Judge the sentiment of each review as POSITIVE or "
        "NEGATIVE based on its text and score.\n\n"
    )


def _build_tasks_for_sample(movie_id, short, sample_idx, seed):
    sample, gold = get_movie_sample(movie_id, seed=seed)
    reviews_block = render_reviews_block(sample)
    preamble = _build_preamble(movie_id, len(sample)) + f"Reviews:\n{reviews_block}\n\nTask: "
    prefix = f"sb_{short}_s{sample_idx}"

    return [
        {
            "id": f"{prefix}_q2", "tier": "hard", "type": "semantic",
            "semantic_kind": "id_list_precision",
            "prompt": preamble +
                "List the reviewIds of the reviews that are clearly POSITIVE. "
                "Respond with ONLY valid JSON: a plain array of reviewId strings, "
                "e.g. [\"123\", \"456\"]. If none are positive, respond with []. "
                "No other text, no markdown, no explanation.",
            "gold": gold,
        },
        {
            "id": f"{prefix}_q3", "tier": "hard", "type": "semantic",
            "semantic_kind": "count",
            "prompt": preamble +
                "Count how many of these reviews are POSITIVE. Respond with ONLY "
                "valid JSON: {\"positive_review_cnt\": <integer>}. No other text, "
                "no markdown, no explanation.",
            "gold": {"field": "positive_review_cnt", "value": gold["positive_count"], "tol": 2},
        },
        {
            "id": f"{prefix}_q4", "tier": "hard", "type": "semantic",
            "semantic_kind": "ratio",
            "prompt": preamble +
                "What fraction of these reviews are POSITIVE (positive count "
                "divided by total count)? Respond with ONLY valid JSON: "
                "{\"positivity_ratio\": <number between 0 and 1>}. No other text, "
                "no markdown, no explanation.",
            "gold": {"field": "positivity_ratio", "value": gold["ratio"], "tol": 0.05},
        },
        {
            "id": f"{prefix}_q8", "tier": "hard", "type": "semantic",
            "semantic_kind": "breakdown",
            "prompt": preamble +
                "Count how many reviews are POSITIVE and how many are NEGATIVE. "
                "Respond with ONLY valid JSON: {\"POSITIVE\": <integer>, "
                "\"NEGATIVE\": <integer>}. No other text, no markdown, no explanation.",
            "gold": {"value": {"POSITIVE": gold["positive_count"], "NEGATIVE": gold["negative_count"]}, "tol": 2},
        },
    ]


TASKS = []
for _movie in MOVIES:
    for _idx, _seed in enumerate(SEEDS, start=1):
        TASKS.extend(_build_tasks_for_sample(_movie["movie_id"], _movie["short"], _idx, _seed))

if __name__ == "__main__":
    print(f"{len(TASKS)} semantic tasks generated")
    for t in TASKS:
        print(f"--- {t['id']} ({len(t['prompt'])} chars) ---")
