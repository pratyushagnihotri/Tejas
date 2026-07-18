"""Runs all 40 tasks against a model family, logs latency / energy / correctness.

Supports repeated sweeps (--repeats N) for error bars: each invocation
appends new runs to its output CSV rather than overwriting it, so you can
call this multiple times (or once with --repeats 4) and accumulate runs
under one 'run' column. A single sudo/allocation session covers all repeats
in one invocation, so you don't need to re-authenticate per run.

Platform: auto-picks PowerMeter (macOS/powermetrics) or GPUPowerMeter
(Linux/nvidia-smi) based on the OS — no flag needed.

Family: --family selects which model ladder to run. "qwen" is the original
Mac ladder (3b/7b/14b) and writes to results/raw_results.csv. "qwen-dgx"
(3b/7b/32b) and "llama" (3b/8b/70b) are separate ladders for the DGX, each
writing to their own results file so the three experiments never mix.

Usage:
    sudo .venv/bin/python3 -m tejas.benchmarking.run_experiment                        # Mac, one run
    sudo .venv/bin/python3 -m tejas.benchmarking.run_experiment --repeats 4             # Mac, 4 new runs
    .venv/bin/python3 -m tejas.benchmarking.run_experiment --no-energy --limit 3        # smoke test, no sudo
    .venv/bin/python3 -m tejas.benchmarking.run_experiment --family qwen-dgx --num-threads 8   # DGX, inside srun/sbatch
    .venv/bin/python3 -m tejas.benchmarking.run_experiment --family llama --num-threads 8      # DGX, inside srun/sbatch
"""

import argparse
import csv
import json
import os
import platform
import sqlite3
import time
import urllib.request

from tejas.benchmarking import grading, seed_db
from tejas.benchmarking.tasks import TASKS as SQL_CONCEPT_TASKS
from tejas.benchmarking.sembench_tasks import TASKS as SEMANTIC_TASKS
from tejas.paths import RESULTS_DIR

TASKS = SQL_CONCEPT_TASKS + SEMANTIC_TASKS

MODEL_FAMILIES = {
    # Original Mac ladder — unchanged, writes to raw_results.csv as before.
    "qwen": [
        {"tier": "small", "name": "qwen2.5:3b-instruct"},
        {"tier": "medium", "name": "qwen2.5:7b-instruct"},
        {"tier": "large", "name": "qwen2.5:14b-instruct"},
    ],
    # DGX ladder: same family, upgraded "large" tier (fits one A100-40GB at
    # Ollama's default 4-bit quant).
    "qwen-dgx": [
        {"tier": "small", "name": "qwen2.5:3b-instruct"},
        {"tier": "medium", "name": "qwen2.5:7b-instruct"},
        {"tier": "large", "name": "qwen2.5:32b-instruct"},
    ],
    # DGX ladder: second family, for the "does this generalize beyond Qwen"
    # replication (run as its own separate experiment, not cross-routed).
    "llama": [
        {"tier": "small", "name": "llama3.2:3b"},
        {"tier": "medium", "name": "llama3.1:8b"},
        {"tier": "large", "name": "llama3.3:70b"},
    ],
}

OUTPUT_FILENAMES = {
    "qwen": "raw_results.csv",
    "qwen-dgx": "raw_results_dgx_qwen.csv",
    "llama": "raw_results_dgx_llama.csv",
}

_OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
if not _OLLAMA_HOST.startswith(("http://", "https://")):
    _OLLAMA_HOST = f"http://{_OLLAMA_HOST}"
OLLAMA_URL = f"{_OLLAMA_HOST}/api/generate"

FIELDNAMES = ["run", "model_tier", "model_name", "task_id", "task_tier",
              "task_type", "latency_s", "energy_j", "correct", "response", "error"]


def make_meter():
    """Auto-picks the power meter for the current platform."""
    if platform.system() == "Darwin":
        from tejas.measurement.energy_meter import PowerMeter
        return PowerMeter()
    from tejas.measurement.gpu_energy_meter import GPUPowerMeter
    return GPUPowerMeter()


def call_ollama(model, prompt, timeout=300, num_thread=None):
    payload = {"model": model, "prompt": prompt, "stream": False}
    if num_thread:
        # On a Slurm cgroup-limited allocation, llama-server otherwise
        # auto-detects the WHOLE node's CPU count and oversubscribes the
        # cores we actually reserved — this caps it to match.
        payload["options"] = {"num_thread": num_thread}
    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        OLLAMA_URL, data=data, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())


def load_existing_rows(output_csv):
    """Load prior runs, if any. Rows from before the --repeats feature existed
    (no 'run' column) are migrated to run=1 so they aren't discarded."""
    if not os.path.exists(output_csv):
        return []
    with open(output_csv) as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        if not r.get("run"):
            r["run"] = "1"
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--family", choices=list(MODEL_FAMILIES), default="qwen",
                         help="which model ladder to run (default: qwen, the original Mac one)")
    parser.add_argument("--no-energy", action="store_true",
                         help="skip power sampling (no sudo/root needed) — smoke test only")
    parser.add_argument("--limit", type=int, default=None,
                         help="only run the first N tasks (smoke test)")
    parser.add_argument("--repeats", type=int, default=1,
                         help="number of NEW repeat sweeps to run and append (default 1)")
    parser.add_argument("--num-threads", type=int, default=None,
                         help="cap Ollama's CPU thread count (match your Slurm --cpus-per-task "
                              "on DGX; leave unset on the Mac)")
    args = parser.parse_args()

    models = MODEL_FAMILIES[args.family]
    output_csv = os.path.join(str(RESULTS_DIR), OUTPUT_FILENAMES[args.family])

    existing_rows = load_existing_rows(output_csv)
    existing_run_nums = [int(r["run"]) for r in existing_rows if r["run"].isdigit()]
    start_run = (max(existing_run_nums) + 1) if existing_run_nums else 1

    db_path = seed_db.create_db()
    conn = sqlite3.connect(db_path)
    print(f"Seeded {db_path}")
    print(f"Family: {args.family} ({', '.join(m['name'] for m in models)})")

    meter = None
    if not args.no_energy:
        meter = make_meter()
        print(f"Starting power meter ({type(meter).__name__})...")
        meter.start()
        print("Power meter running.")
    else:
        print("Running WITHOUT energy sampling (--no-energy).")

    tasks = TASKS[: args.limit] if args.limit else TASKS

    new_rows = []
    total = len(models) * len(tasks) * args.repeats
    done = 0
    for run_offset in range(args.repeats):
        run_id = str(start_run + run_offset)
        for model in models:
            for task in tasks:
                t0 = time.time()
                try:
                    result = call_ollama(model["name"], task["prompt"], num_thread=args.num_threads)
                    response_text = result.get("response", "")
                    error = ""
                except Exception as exc:  # noqa: BLE001 — log and continue the sweep
                    response_text = ""
                    error = str(exc)
                t1 = time.time()

                energy_j = meter.energy_since(t0, t1) if meter else None
                correct = grading.grade(conn, response_text, task) if not error else False

                row = {
                    "run": run_id,
                    "model_tier": model["tier"],
                    "model_name": model["name"],
                    "task_id": task["id"],
                    "task_tier": task["tier"],
                    "task_type": task["type"],
                    "latency_s": round(t1 - t0, 3),
                    "energy_j": round(energy_j, 3) if energy_j is not None else "",
                    "correct": correct,
                    "response": response_text.strip().replace("\n", " ")[:200],
                    "error": error,
                }
                new_rows.append(row)
                done += 1
                print(f"[{done}/{total}] run={run_id} {model['tier']:<7} {task['id']:<6} "
                      f"{'OK  ' if correct else 'MISS'} "
                      f"{row['latency_s']}s {row['energy_j']}J"
                      + (f"  ERROR: {error}" if error else ""))

    if meter:
        meter.stop()
    conn.close()

    all_rows = existing_rows + new_rows
    os.makedirs(str(RESULTS_DIR), exist_ok=True)
    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"\nWrote {len(all_rows)} total rows ({len(new_rows)} new, "
          f"runs {start_run}-{start_run + args.repeats - 1}) to {output_csv}")


if __name__ == "__main__":
    main()
