# Tejas — container for local testing (CPU only).
#
# Everything that doesn't need Ollama or a GPU works here: self-checks,
# reproducing the paper's numbers from the committed CSVs in results/,
# and regenerating figures. Collecting *new* energy measurements needs
# powermetrics (macOS) or nvidia-smi (DGX) directly on the host — that
# part is out of scope for this image, matching the README's own
# "what needs GPU access" split.

FROM python:3.11-slim

WORKDIR /app

# Copy dependency metadata first so `pip install` only re-runs when
# pyproject.toml actually changes, not on every source edit.
COPY pyproject.toml requirements.txt ./
COPY src ./src
COPY results ./results
COPY analysis ./analysis
COPY docs ./docs
COPY README.md ./

RUN pip install --no-cache-dir -e .

# Default: run the same two commands the README's "Verify the
# installation" section describes, so `docker build` + `docker run`
# alone is proof the environment and the committed data both work.
# Override with any other command, e.g.:
#   docker run --rm -it tejas bash
#   docker run --rm tejas python -m tejas.routing.evaluate_policy
CMD ["sh", "-c", "python -m tejas.cost_model.task_features && python -m tejas.routing.oracle"]
