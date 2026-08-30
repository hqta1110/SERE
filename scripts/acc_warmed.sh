#!/usr/bin/env bash
# Accuracy with the mask installed BEFORE scoring starts.
#
# Replaces every ARES accuracy number measured so far. In the earlier runs the
# eval itself did the profiling, so the mask landed mid-benchmark (~40 s into a
# ~102 s GSM8K run) and part of the score came from the unmodified model. The
# bias is one-sided: SERE's similarity matrix is static and active from token 0.
#
# Warm data is held out from the scored set:
#   GSM8K test <- GSM8K train      HumanEval <- MBPP
#
# Arms: ARES contrib_mass frac 0.20 (the headline config), the two percentile
# ablation points, and baseline. Baseline gets the same warm traffic so all arms
# see identically warmed caches.
set -uo pipefail

exec 9>/tmp/ares_acc_warmed.lock
flock -n 9 || { echo "[warm-acc] locked"; exit 0; }

VENV=/home/PC/new-efficient-moe/.venv
AR=/home/PC/new-efficient-moe
SV=/home/PC/SERE_v1
OUT=$SV/results/acc_warmed
mkdir -p "$OUT"
HUB=/home/PC/.cache/huggingface/hub
Q3=$(ls -d "$HUB"/models--Qwen--Qwen3-30B-A3B/snapshots/*/ | head -1); Q3=${Q3%/}

# cell <tag> <gpus> <port> <method|base> <frac> <ma> <ks> <mp> <vp>
cell() {
  local tag=$1 gpus=$2 port=$3 method=$4 frac=$5 ma=$6 ks=$7 mp=$8 vp=$9
  for bench in gsm8k humaneval_base; do
    local d="$OUT/$tag/$bench"
    if find "$d" -name eval_results.json -size +0 2>/dev/null | grep -q .; then
      echo "  [$tag/$bench] scored"; continue
    fi
    mkdir -p "$d"
    local skiplog=""
    local -a e=(CUDA_VISIBLE_DEVICES="$gpus")
    if [[ "$method" != base ]]; then
      skiplog="$d/skip.log"
      set -a; source "$AR/.env"; set +a
      e+=(PYTHONPATH="$AR" EXPERT_SKIP_MODE=dynamic
          EXPERT_SKIP_DISABLED_LAYERS=0,47
          EXPERT_SKIP_ONLINE_THRESHOLD_METHOD="$method"
          EXPERT_SKIP_ONLINE_CONFIDENCE_KEEP_SHARE="$ks"
          EXPERT_SKIP_ONLINE_CONTRIB_MASS_FRACTION="$frac"
          EXPERT_SKIP_ONLINE_MIN_ACTIVE_FRAC="$ma"
          EXPERT_SKIP_ONLINE_MEAN_ACT_PERCENTILE="$mp"
          EXPERT_SKIP_ONLINE_VAR_ACT_PERCENTILE="$vp"
          EXPERT_SKIP_ONLINE_UNWEIGHTED_NORMS=1
          EXPERT_SKIP_ONLINE_REFRESH_EVERY_N_TOKENS=9999999
          EXPERT_SKIP_ONLINE_LOG=1 EXPERT_SKIP_ONLINE_LOG_FILE="$skiplog"
          EXPERT_SKIP_ONLINE_DRIFT_CHECK_EVERY_N_STEPS=0
          EXPERT_SKIP_ONLINE_RISK_TRIGGER=0)
    fi
    env "${e[@]}" "$VENV/bin/vllm" serve "$Q3" --served-model-name m \
      --trust-remote-code --tensor-parallel-size 2 --max-model-len 8192 \
      --gpu-memory-utilization 0.90 --port "$port" > "$d/server.log" 2>&1 &
    local pid=$! w=0
    until curl -sf "http://127.0.0.1:$port/health" >/dev/null 2>&1; do
      kill -0 $pid 2>/dev/null || { echo "  [$tag/$bench] SERVER DIED"; break; }
      (( w > 1800 )) && { echo "  [$tag/$bench] TIMEOUT"; kill -9 $pid; break; }
      sleep 10; w=$((w+10))
    done
    if curl -sf "http://127.0.0.1:$port/health" >/dev/null 2>&1; then
      local src=gsm8k_train; [[ "$bench" == humaneval_base ]] && src=mbpp
      "$VENV/bin/python" "$SV/scripts/warm_until_mask.py" --port "$port" --model m \
        --warm-source "$src" ${skiplog:+--skip-log "$skiplog"} > "$d/warm.log" 2>&1
      local wrc=$?
      echo "  [$tag/$bench] warm rc=$wrc $(grep -c 'contrib_mass' "$skiplog" 2>/dev/null || echo 0) finalize lines before scoring"
      if [[ "$method" != base && "$wrc" -ne 0 ]]; then
        echo "  [$tag/$bench] WARNING: mask not confirmed; score would be contaminated"
      fi
      PYTHONPATH="$AR" HF_ALLOW_CODE_EVAL=1 "$VENV/bin/python" \
        "$AR/scripts/serve/run_eval_serve.py" --task "$bench" --model_name "$Q3" \
        --local_model_path m --base_url "http://127.0.0.1:$port/v1" \
        --max_concurrency 128 --log_samples --output_dir "$d/run" \
        > "$d/client.log" 2>&1
      echo "  [$tag/$bench] $(grep -ohE '"(pass@1,create_test|exact_match,flexible-extract)": [0-9.]+' "$d"/run/*/eval_results.json 2>/dev/null | head -1)"
    fi
    kill $pid 2>/dev/null; wait $pid 2>/dev/null; sleep 8
  done
}

echo "=== WARMED ACCURACY (mask installed before scoring) $(date '+%F %T') ==="
echo "    supersedes results/acc_frontier and results/ablation accuracy numbers"
bash "$AR/scripts/drain_gpus.sh" >/dev/null 2>&1
cell base_warm   0,1 8900 base         0.2 0.25 0    50 90 &
cell cm_f020     2,3 8901 contrib_mass 0.2 0.25 0    50 90 &
wait
bash "$AR/scripts/drain_gpus.sh" >/dev/null 2>&1
cell pct_p50p90  0,1 8900 percentile   0.2 0.25 0    50 90 &
cell pct_p90p99  2,3 8901 percentile   0.2 0.25 0    90 99 &
wait
echo "=== DONE $(date '+%F %T') ==="
