#!/bin/bash
# Submit with (note: --gres must be passed here, at submit time — Slurm's
# #SBATCH directives are static and can't branch on FAMILY):
#
#   sbatch --gres=gpu:1 --export=FAMILY=qwen-dgx,REPEATS=4 slurm/run_experiment.slurm
#   sbatch --gres=gpu:2 --export=FAMILY=llama,REPEATS=4    slurm/run_experiment.slurm
#
# qwen-dgx needs 1 GPU (32B fits ~20GB at Ollama's default 4-bit quant).
# llama needs 2 GPUs (70B at 4-bit is ~44GB, doesn't fit one 40GB A100).
# Getting --gres wrong for the family will fail at model-load time, not at
# submission — check nvidia-smi in the log if a job dies partway through.
#
# Slurm does not create the log directory: run `mkdir -p logs` before sbatch,
# otherwise the job fails without writing any output.

#SBATCH --job-name=tejas
#SBATCH --partition=batch
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

REPO="$HOME/pratyush/SustainableAI"
OLLAMA_BIN="$HOME/pratyush/ollama/bin/ollama"
OLLAMA_LIBDIR="$HOME/pratyush/ollama/lib/ollama"
export OLLAMA_MODELS="$HOME/pratyush/ollama-models"
NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
FAMILY="${FAMILY:?set FAMILY=qwen-dgx or FAMILY=llama via sbatch --export}"
REPEATS="${REPEATS:-1}"

mkdir -p "$REPO/logs" "$OLLAMA_MODELS"
cd "$REPO"

echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: $(hostname)"
echo "Family: $FAMILY   Repeats: $REPEATS   Threads: $NUM_THREADS"
echo "Start time: $(date)"
echo "GPU status before run:"
nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv

export LD_LIBRARY_PATH="$OLLAMA_LIBDIR:${LD_LIBRARY_PATH:-}"
"$OLLAMA_BIN" serve > logs/ollama-serve-${SLURM_JOB_ID}.log 2>&1 &
OLLAMA_PID=$!
trap 'kill $OLLAMA_PID 2>/dev/null || true' EXIT

# wait for the server to come up
for i in $(seq 1 30); do
    if curl -s http://localhost:11434 > /dev/null 2>&1; then
        echo "ollama serve is up after ${i}s"
        break
    fi
    sleep 1
done

source .venv/bin/activate

echo "=== pulling models for family=$FAMILY (no-op if already present) ==="
case "$FAMILY" in
    qwen-dgx)
        "$OLLAMA_BIN" pull qwen2.5:3b-instruct
        "$OLLAMA_BIN" pull qwen2.5:7b-instruct
        "$OLLAMA_BIN" pull qwen2.5:32b-instruct
        ;;
    llama)
        "$OLLAMA_BIN" pull llama3.2:3b
        "$OLLAMA_BIN" pull llama3.1:8b
        "$OLLAMA_BIN" pull llama3.3:70b
        ;;
    *)
        echo "unknown FAMILY: $FAMILY" >&2
        exit 1
        ;;
esac

echo "=== running experiment ==="
python3 -m tejas.benchmarking.run_experiment --family "$FAMILY" --repeats "$REPEATS" --num-threads "$NUM_THREADS"

echo "End time: $(date)"
