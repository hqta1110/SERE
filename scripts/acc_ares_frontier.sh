#!/usr/bin/env bash
# Accuracy guardrail for the ARES speed-frontier configs.
#
# Speed is the objective; this run only decides which of the fast configs is
# ALLOWED. Pick the fastest one whose accuracy has not dropped significantly.
#
# Same model/harness/engine as every other accuracy number here, so directly
# comparable:  baseline GSM8K 85.60  HumanEval-base 30.49
#              SERE S=1 rho=0.0 (1.456x)  85.06 / 31.71
#
# Configs under test (all keep_share=0, i.e. per-token rescue OFF):
#   f1  frac 0.2  min_active 0.25 -> 1.623x, 19.2% of output mass discarded
#   f2  frac 0.35 min_active 0.25 -> 1.712x, 33.0%
#   f3  frac 0.5  min_active 0.10 -> 1.772x, 47.1%
set -uo pipefail

VENV=/home/PC/new-efficient-moe/.venv
ARES_ROOT=/home/PC/new-efficient-moe
OUT=/home/PC/SERE_v1/results/acc_frontier
mkdir -p "$OUT"
HUB=/home/PC/.cache/huggingface/hub
Q3=$(ls -d "$HUB"/models--Qwen--Qwen3-30B-A3B/snapshots/*/ | head -1); Q3=${Q3%/}

# cell <tag> <frac> <min_active> <gpus> <port>
cell() {
  local tag=$1 frac=$2 ma=$3 gpus=$4 port=$5
  for bench in gsm8k humaneval_base; do
    local bdir="$OUT/$tag/$bench"
    if find "$bdir" -name eval_results.json -size +0 2>/dev/null | grep -q .; then
      echo "  [$tag/$bench] already scored"; continue
    fi
    mkdir -p "$bdir"
    set -a; source "$ARES_ROOT/.env"; set +a
    env CUDA_VISIBLE_DEVICES="$gpus" PYTHONPATH="$ARES_ROOT" \
        EXPERT_SKIP_MODE=dynamic EXPERT_SKIP_DISABLED_LAYERS=0,47 \
        EXPERT_SKIP_ONLINE_THRESHOLD_METHOD=contrib_mass \
        EXPERT_SKIP_ONLINE_CONFIDENCE_KEEP_SHARE=0 \
        EXPERT_SKIP_ONLINE_CONTRIB_MASS_FRACTION="$frac" \
        EXPERT_SKIP_ONLINE_MIN_ACTIVE_FRAC="$ma" \
        EXPERT_SKIP_ONLINE_UNWEIGHTED_NORMS=1 \
        EXPERT_SKIP_ONLINE_REFRESH_EVERY_N_TOKENS=9999999 \
        EXPERT_SKIP_ONLINE_LOG=1 EXPERT_SKIP_ONLINE_LOG_FILE="$bdir/skip.log" \
        EXPERT_SKIP_ONLINE_DRIFT_CHECK_EVERY_N_STEPS=0 \
        EXPERT_SKIP_ONLINE_RISK_TRIGGER=0 \
        "$VENV/bin/vllm" serve "$Q3" --served-model-name ares \
        --trust-remote-code --tensor-parallel-size 2 --max-model-len 8192 \
        --gpu-memory-utilization 0.90 --port "$port" > "$bdir/server.log" 2>&1 &
    local pid=$! w=0
    until curl -sf "http://127.0.0.1:$port/health" >/dev/null 2>&1; do
      kill -0 $pid 2>/dev/null || { echo "  [$tag/$bench] SERVER DIED"; break; }
      (( w > 1800 )) && { echo "  [$tag/$bench] TIMEOUT"; kill -9 $pid; break; }
      sleep 10; w=$((w+10))
    done
    if curl -sf "http://127.0.0.1:$port/health" >/dev/null 2>&1; then
      # --local_model_path is the served alias; --model_name supplies the tokenizer.
      PYTHONPATH="$ARES_ROOT" HF_ALLOW_CODE_EVAL=1 "$VENV/bin/python" \
        "$ARES_ROOT/scripts/serve/run_eval_serve.py" --task "$bench" \
        --model_name "$Q3" --local_model_path ares \
        --base_url "http://127.0.0.1:$port/v1" \
        --max_concurrency 128 --log_samples --output_dir "$bdir/run" \
        > "$bdir/client.log" 2>&1
      echo "  [$tag/$bench] rc=$? $(grep -ohE '"(pass@1,create_test|exact_match,flexible-extract)": [0-9.]+' "$bdir"/run/*/eval_results.json 2>/dev/null | head -1)"
    fi
    kill $pid 2>/dev/null; wait $pid 2>/dev/null; sleep 8
  done
}

echo "=== ARES FRONTIER ACCURACY GUARDRAIL $(date '+%F %T') ==="
bash "$ARES_ROOT/scripts/drain_gpus.sh" >/dev/null 2>&1
cell f1_frac020 0.2  0.25 0,1 8600 &
cell f2_frac035 0.35 0.25 2,3 8601 &
wait
bash "$ARES_ROOT/scripts/drain_gpus.sh" >/dev/null 2>&1
cell f3_frac050 0.5  0.10 0,1 8600
echo "=== DONE $(date '+%F %T') ==="
