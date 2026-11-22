#!/usr/bin/env bash
# P1: SemBench Q1 predicate on every review, one vLLM server per model tier (sequentially, same GPU).
#   sbatch probes/slurm/p1_tiers.sbatch
#   then on any machine: python -m probes.p1_tejas_sembench.analyze_p1 <OUT> small,medium,large
#SBATCH --job-name=p1-tejas-sembench
#SBATCH --partition=batch
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x-%j.out
set -euo pipefail
# FlashInfer JIT-compiles its sampler on first use and fails on this node; use vLLM's PyTorch sampler.
export VLLM_USE_FLASHINFER_SAMPLER=0 CUDA_HOME=${CUDA_HOME:-/usr/local/cuda-13.0}
REPO=${REPO:-$HOME/pratyush/sustainableAI}
VENV=${VENV:-$HOME/pratyush/probes-venv}
# Same ladder as Tejas' DGX-Qwen deployment; 32B in AWQ so it fits one 40 GB A100.
TIERS=${TIERS:-"small=Qwen/Qwen2.5-3B-Instruct medium=Qwen/Qwen2.5-7B-Instruct large=Qwen/Qwen2.5-32B-Instruct-AWQ"}
REPS=${REPS:-3}
export SEMBENCH_MOVIE_DIR=${SEMBENCH_MOVIE_DIR:-$HOME/pratyush/greenQueryLab/third_party/sembench/files/movie/data/sf_2000}
export HF_HOME=${HF_HOME:-$HOME/pratyush/hf-cache}
OUT=$REPO/probes/results/p1/${SLURM_JOB_ID:-local}${RUN_TAG:+-$RUN_TAG}
mkdir -p "$OUT" "$REPO/logs"
if nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -q '[0-9]'; then echo "GPU busy" >&2; exit 2; fi
cd "$REPO"
for pair in $TIERS; do
  TIER=${pair%%=*}; MODEL=${pair#*=}
  PORT=$(python3 -c 'import socket;s=socket.socket();s.bind(("",0));print(s.getsockname()[1])')
  VLLM_SERVER_DEV_MODE=1 "$VENV/bin/vllm" serve "$MODEL" --port "$PORT" --enable-prefix-caching --enable-prompt-tokens-details \
    --gpu-memory-utilization 0.90 --max-model-len 4096 --max-logprobs 20 > "$OUT/vllm_$TIER.log" 2>&1 &
  PID=$!
  for i in $(seq 1 900); do curl -sf "http://127.0.0.1:$PORT/v1/models" >/dev/null && break; sleep 2; done
  PROBE_BASE_URL="http://127.0.0.1:$PORT/v1" PROBE_MODEL="$MODEL" PROBE_CONCURRENCY=64 \
    "$VENV/bin/python" -m probes.p1_tejas_sembench.run_p1 --tier "$TIER" --out "$OUT" --reps "$REPS" --idle-seconds 60 ${P1_ARGS:-}
  kill $PID; wait $PID 2>/dev/null || true; sleep 20
done
echo "done: $OUT"
