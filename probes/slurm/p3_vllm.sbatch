#!/usr/bin/env bash
# P3: DBMS-level semantic reuse. Large and small model served by two vLLM servers on one A100.
#   sbatch probes/slurm/p3_vllm.sbatch
#SBATCH --job-name=p3-semantic-reuse
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
MODEL=${MODEL:-Qwen/Qwen2.5-7B-Instruct}
SMALL=${SMALL:-Qwen/Qwen2.5-1.5B-Instruct}
REPS=${REPS:-3}
export SEMBENCH_MOVIE_DIR=${SEMBENCH_MOVIE_DIR:-$HOME/pratyush/greenQueryLab/third_party/sembench/files/movie/data/sf_2000}
export HF_HOME=${HF_HOME:-$HOME/pratyush/hf-cache}
OUT=$REPO/probes/results/p3/${SLURM_JOB_ID:-local}-$(echo "$MODEL" | tr '/:' '__')
mkdir -p "$OUT" "$REPO/logs"

if nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -q '[0-9]'; then
  echo "GPU busy" >&2; exit 2
fi
free_port() { python3 -c 'import socket;s=socket.socket();s.bind(("",0));print(s.getsockname()[1])'; }
P1=$(free_port); P2=$(free_port)
VLLM_SERVER_DEV_MODE=1 "$VENV/bin/vllm" serve "$MODEL" --port "$P1" --enable-prefix-caching --enable-prompt-tokens-details \
  --gpu-memory-utilization 0.60 --max-model-len 4096 > "$OUT/vllm_large.log" 2>&1 &
PID1=$!
for i in $(seq 1 600); do curl -sf "http://127.0.0.1:$P1/v1/models" >/dev/null && break; sleep 2; done
VLLM_SERVER_DEV_MODE=1 "$VENV/bin/vllm" serve "$SMALL" --port "$P2" --enable-prefix-caching --enable-prompt-tokens-details \
  --gpu-memory-utilization 0.25 --max-model-len 4096 > "$OUT/vllm_small.log" 2>&1 &
PID2=$!
trap 'kill $PID1 $PID2 2>/dev/null || true' EXIT
for i in $(seq 1 600); do curl -sf "http://127.0.0.1:$P2/v1/models" >/dev/null && break; sleep 2; done

cd "$REPO"
PROBE_BASE_URL="http://127.0.0.1:$P1/v1" PROBE_MODEL="$MODEL" \
PROBE_SMALL_BASE_URL="http://127.0.0.1:$P2/v1" PROBE_SMALL_MODEL="$SMALL" PROBE_CONCURRENCY=64 \
  "$VENV/bin/python" -m probes.p3_semantic_reuse.run_p3 --out "$OUT" --reps "$REPS" --idle-seconds 60
echo "done: $OUT"
