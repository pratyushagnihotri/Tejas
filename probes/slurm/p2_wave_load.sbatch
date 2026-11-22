#!/usr/bin/env bash
# P2b: wave-size sweep under shared load, work-based attribution (see p2_plan_aware/wave_load.py).
#
#   sbatch probes/slurm/p2_wave_load.sbatch
#
# Needs (once, on the login node):
#   uv venv --python 3.12 ~/pratyush/probes-venv
#   uv pip install --python ~/pratyush/probes-venv/bin/python vllm openai httpx pynvml
#   SemBench Movie data at $SEMBENCH_MOVIE_DIR (defaults to the greenQueryLab checkout)
#
#SBATCH --job-name=p2-wave-load
#SBATCH --partition=batch
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x-%j.out
set -euo pipefail
# FlashInfer JIT-compiles its sampler on first use and fails on this node; use vLLM's PyTorch sampler.
export VLLM_USE_FLASHINFER_SAMPLER=0 CUDA_HOME=${CUDA_HOME:-/usr/local/cuda-13.0}

REPO=${REPO:-$HOME/pratyush/sustainableAI}
VENV=${VENV:-$HOME/pratyush/probes-venv}
MODEL=${MODEL:-Qwen/Qwen2.5-7B-Instruct}
REPS=${REPS:-3}
QUERIES=${QUERIES:-q1_limit,q1_full,q5_join_limit,q5_join_full,q6_twopred_full}
export SEMBENCH_MOVIE_DIR=${SEMBENCH_MOVIE_DIR:-$HOME/pratyush/greenQueryLab/third_party/sembench/files/movie/data/sf_2000}
export HF_HOME=${HF_HOME:-$HOME/pratyush/hf-cache}
OUT=$REPO/probes/results/p2_wave_load/${SLURM_JOB_ID:-local}-$(echo "$MODEL" | tr '/:' '__')
mkdir -p "$OUT" "$REPO/logs"

# Refuse to measure on a GPU someone else is using.
if nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -q '[0-9]'; then
  echo "GPU busy, energy would be contaminated" >&2; exit 2
fi
nvidia-smi --query-gpu=index,name,driver_version,memory.total --format=csv | tee "$OUT/gpu.csv"

PORT=$(python3 -c 'import socket;s=socket.socket();s.bind(("",0));print(s.getsockname()[1])')
# Prefix caching on (fair serving baseline B); dev mode exposes POST /reset_prefix_cache,
# which the driver calls before every execution so runs do not share cached prefixes.
VLLM_SERVER_DEV_MODE=1 "$VENV/bin/vllm" serve "$MODEL" --port "$PORT" \
  --enable-prefix-caching --enable-prompt-tokens-details --gpu-memory-utilization 0.90 --max-model-len 8192 \
  > "$OUT/vllm.log" 2>&1 &
VLLM_PID=$!
trap 'kill $VLLM_PID 2>/dev/null || true' EXIT
for i in $(seq 1 600); do curl -sf "http://127.0.0.1:$PORT/v1/models" >/dev/null && break; sleep 2; done
curl -sf "http://127.0.0.1:$PORT/v1/models" >/dev/null || { echo "vLLM did not start"; tail -50 "$OUT/vllm.log"; exit 3; }
"$VENV/bin/pip" freeze 2>/dev/null | grep -i -E '^(vllm|torch|openai)==' > "$OUT/versions.txt" || true

cd "$REPO"
PROBE_BASE_URL="http://127.0.0.1:$PORT/v1" PROBE_MODEL="$MODEL" PROBE_CONCURRENCY=256 \
  "$VENV/bin/python" -m probes.p2_plan_aware.wave_load --out "$OUT" --reps "$REPS" \
  --idle-seconds 60 ${EXTRA_ARGS:-}
"$VENV/bin/python" -m probes.p2_plan_aware.analyze_wave_load "$OUT" || true
echo "done: $OUT"
