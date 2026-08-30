#!/usr/bin/env bash
# Accuracy for SERE-on-V1 at the three benchmarked configs, on the SAME model,
# harness and engine as the ARES numbers in outputs/accuracy_suite/q3-{dyn,off}.
#
# Without this the Pareto plot has no y-axis: the recorded SERE retention
# figures are Qwen1.5 / OpenCompass, while every speed number here is Qwen3.
#
# GSM8K (n=1319) is the large-N anchor; HumanEval-base (n=164) is where SERE is
# recorded to collapse, so it is the discriminating one despite the small N.
# Reference points already measured, same harness:
#   baseline  GSM8K 85.60   HumanEval-base 30.49
#   ARES      GSM8K 82.94   HumanEval-base 31.71
set -uo pipefail

VENV=/home/PC/new-efficient-moe/.venv
ARES_ROOT=/home/PC/new-efficient-moe
Q3_SERE=/home/PC/SERE/artifacts/calibrated/Qwen3-30B-A3B-sere
OUT=/home/PC/SERE_v1/results/acc
mkdir -p "$OUT/logs"

# cell <tag> <S> <rho> <gpus> <port>
cell() {
  local tag=$1 S=$2 rho=$3 gpus=$4 port=$5
  for bench in gsm8k humaneval_base; do
    local bdir="$OUT/$tag/$bench"
    if find "$bdir" -name eval_results.json -size +0 2>/dev/null | grep -q .; then
      echo "  [$tag/$bench] already scored, skipping"; continue
    fi
    mkdir -p "$bdir"
    # Fresh server per benchmark, matching run_accuracy_suite.sh's protocol.
    env -u PYTHONPATH -u NCCL_NET CUDA_VISIBLE_DEVICES="$gpus" \
      "$VENV/bin/vllm" serve "$Q3_SERE" --served-model-name sere \
      --trust-remote-code --tensor-parallel-size 2 --max-model-len 8192 \
      --gpu-memory-utilization 0.90 --port "$port" \
      --hf-overrides "{\"select_top_k\":$S,\"threshold\":$rho}" \
      > "$bdir/server.log" 2>&1 &
    local pid=$!
    local w=0
    until curl -sf "http://127.0.0.1:$port/health" >/dev/null 2>&1; do
      kill -0 $pid 2>/dev/null || { echo "  [$tag/$bench] SERVER DIED"; break; }
      (( w > 1800 )) && { echo "  [$tag/$bench] TIMEOUT"; kill -9 $pid; break; }
      sleep 10; w=$((w+10))
    done
    if curl -sf "http://127.0.0.1:$port/health" >/dev/null 2>&1; then
      PYTHONPATH="$ARES_ROOT" HF_ALLOW_CODE_EVAL=1 "$VENV/bin/python" \
        "$ARES_ROOT/scripts/serve/run_eval_serve.py" --task "$bench" \
        --model_name "$Q3_SERE" --local_model_path sere \
        --base_url "http://127.0.0.1:$port/v1" \
        --max_concurrency 128 --log_samples --output_dir "$bdir/run" \
        > "$bdir/client.log" 2>&1
      echo "  [$tag/$bench] rc=$? $(grep -oE '\"(pass@1[^\"]*|exact_match,flexible-extract)\": [0-9.]+' "$bdir"/run/*/eval_results.json 2>/dev/null | head -2 | tr '\n' ' ')"
    fi
    kill $pid 2>/dev/null; wait $pid 2>/dev/null; sleep 8
  done
}

echo "=== SERE-on-V1 ACCURACY $(date '+%F %T') ==="
bash "$ARES_ROOT/scripts/drain_gpus.sh" >/dev/null 2>&1
cell sere_s1_r00 1 0.0 0,1 8500 &
cell sere_s1_r03 1 0.3 2,3 8501 &
wait
bash "$ARES_ROOT/scripts/drain_gpus.sh" >/dev/null 2>&1
cell sere_s2_r05 2 0.5 0,1 8500
echo "=== DONE $(date '+%F %T') ==="
