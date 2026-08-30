#!/usr/bin/env bash
# Weight-compensation A/B: the cheapest untested accuracy lever.
#
# Every ARES run to date had EXPERT_SKIP_RENORM=0 and
# EXPERT_SKIP_MAGNITUDE_RESCALE=0. But the risk monitor shows the mask
# surrenders ~2-3x more GATE WEIGHT than the output-mass budget suggests
# (frac 0.10 -> 27.7% of gate mass dropped): Qwen3 normalizes top-k weights
# to sum 1, so after masking the kept weights sum to ~0.72 and the MoE
# output is systematically undersized. Two one-env-var fixes, both pure
# tensor ops in the router (CUDA-graph safe, no extra experts run):
#
#   RENORM=1            kept weights / row_sum  -> gate mass restored to 1.
#                       Overshoots output magnitude (dropped experts carried
#                       27% of weight but only 10% of output mass).
#   MAGNITUDE_RESCALE=1 scale by profiled mass ratio sum_all/sum_kept (~1.11
#                       at frac 0.10) -> restores output magnitude exactly;
#                       the theoretically correct compensation for what
#                       contrib_mass removes.
#
# Reference points (warmed, GSM8K strict): base 89.46 | f005 89.61 |
# f010 86.13 | f020 69.75 | SERE S1r0 87.72 @ 1.456x.
# Success = f010/f015 + compensation closes toward baseline at unchanged
# throughput (speed pair at the end confirms the overhead is nil).
set -uo pipefail

exec 8>/tmp/ares_rescale.lock
flock -n 8 || { echo "[rescale] locked"; exit 0; }

VENV=/home/PC/new-efficient-moe/.venv
AR=/home/PC/new-efficient-moe
SV=/home/PC/SERE_v1
OUT=$SV/results/rescale_ab
mkdir -p "$OUT" "$OUT/logs"
HUB=/home/PC/.cache/huggingface/hub
Q3=$(ls -d "$HUB"/models--Qwen--Qwen3-30B-A3B/snapshots/*/ | head -1); Q3=${Q3%/}

# cell <tag> <gpus> <port> <frac> <renorm> <magrescale>
cell() {
  local tag=$1 gpus=$2 port=$3 frac=$4 rn=$5 mr=$6
  for bench in gsm8k humaneval_base; do
    local d="$OUT/$tag/$bench"
    if find "$d" -name eval_results.json -size +0 2>/dev/null | grep -q .; then
      echo "  [$tag/$bench] scored"; continue
    fi
    mkdir -p "$d"
    local skiplog="$d/skip.log"
    set -a; source "$AR/.env"; set +a
    local -a e=(CUDA_VISIBLE_DEVICES="$gpus" PYTHONPATH="$AR"
        EXPERT_SKIP_MODE=dynamic
        EXPERT_SKIP_DISABLED_LAYERS=0,47
        EXPERT_SKIP_RENORM="$rn"
        EXPERT_SKIP_MAGNITUDE_RESCALE="$mr"
        EXPERT_SKIP_ONLINE_THRESHOLD_METHOD=contrib_mass
        EXPERT_SKIP_ONLINE_CONFIDENCE_KEEP_SHARE=0
        EXPERT_SKIP_ONLINE_CONTRIB_MASS_FRACTION="$frac"
        EXPERT_SKIP_ONLINE_MIN_ACTIVE_FRAC=0.25
        EXPERT_SKIP_ONLINE_UNWEIGHTED_NORMS=1
        EXPERT_SKIP_ONLINE_REFRESH_EVERY_N_TOKENS=9999999
        EXPERT_SKIP_ONLINE_LOG=1 EXPERT_SKIP_ONLINE_LOG_FILE="$skiplog"
        EXPERT_SKIP_ONLINE_DRIFT_CHECK_EVERY_N_STEPS=0
        EXPERT_SKIP_ONLINE_RISK_TRIGGER=0)
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
        --warm-source "$src" --skip-log "$skiplog" > "$d/warm.log" 2>&1
      local wrc=$?
      echo "  [$tag/$bench] warm rc=$wrc"
      if [[ "$wrc" -ne 0 ]]; then
        echo "  [$tag/$bench] WARNING: mask not confirmed; score would be contaminated"
      fi
      PYTHONPATH="$AR" HF_ALLOW_CODE_EVAL=1 "$VENV/bin/python" \
        "$AR/scripts/serve/run_eval_serve.py" --task "$bench" --model_name "$Q3" \
        --local_model_path m --base_url "http://127.0.0.1:$port/v1" \
        --max_concurrency 128 --log_samples --output_dir "$d/run" \
        > "$d/client.log" 2>&1
      echo "  [$tag/$bench] $(grep -ohE '"(pass@1,create_test|exact_match,strict-match)": [0-9.]+' "$d"/run/*/eval_results.json 2>/dev/null | head -1)"
    fi
    kill $pid 2>/dev/null; wait $pid 2>/dev/null; sleep 8
  done
}

# speed pair: worst-case-overhead point (frac 0.15) with each compensation on,
# vs contemporaneous baseline. Protocol matches frontier_light exactly
# (in=128 out=1024, 200 prompts saturated, TP=2, max-model-len 4096).
speed_pair() {  # speed_pair <tag> <rn> <mr>
  local tag=$1 rn=$2 mr=$3
  [[ -f "$OUT/$tag.json" && -f "$OUT/base_$tag.json" ]] && { echo "[skip] speed $tag"; return 0; }
  bash "$AR/scripts/drain_gpus.sh" >/dev/null 2>&1
  CUDA_VISIBLE_DEVICES=0,1 "$VENV/bin/vllm" serve "$Q3" --served-model-name bench \
    --trust-remote-code --tensor-parallel-size 2 --max-model-len 4096 \
    --gpu-memory-utilization 0.90 --port 8400 --no-enable-prefix-caching \
    > "$OUT/logs/base_$tag.server.log" 2>&1 &
  local bp=$!
  set -a; source "$AR/.env"; set +a
  env CUDA_VISIBLE_DEVICES=2,3 PYTHONPATH="$AR" EXPERT_SKIP_MODE=dynamic \
      EXPERT_SKIP_DISABLED_LAYERS=0,47 \
      EXPERT_SKIP_RENORM="$rn" EXPERT_SKIP_MAGNITUDE_RESCALE="$mr" \
      EXPERT_SKIP_ONLINE_THRESHOLD_METHOD=contrib_mass \
      EXPERT_SKIP_ONLINE_CONFIDENCE_KEEP_SHARE=0 \
      EXPERT_SKIP_ONLINE_CONTRIB_MASS_FRACTION=0.15 \
      EXPERT_SKIP_ONLINE_MIN_ACTIVE_FRAC=0.25 \
      EXPERT_SKIP_ONLINE_UNWEIGHTED_NORMS=1 \
      EXPERT_SKIP_ONLINE_REFRESH_EVERY_N_TOKENS=9999999 \
      EXPERT_SKIP_ONLINE_LOG=1 EXPERT_SKIP_ONLINE_LOG_FILE="$OUT/logs/$tag.skip.log" \
      EXPERT_SKIP_ONLINE_DRIFT_CHECK_EVERY_N_STEPS=0 \
      EXPERT_SKIP_ONLINE_RISK_TRIGGER=0 \
      "$VENV/bin/vllm" serve "$Q3" --served-model-name bench \
      --trust-remote-code --tensor-parallel-size 2 --max-model-len 4096 \
      --gpu-memory-utilization 0.90 --port 8401 --no-enable-prefix-caching \
      > "$OUT/logs/$tag.server.log" 2>&1 &
  local tp=$!
  local w=0
  until curl -sf http://127.0.0.1:8400/health >/dev/null && curl -sf http://127.0.0.1:8401/health >/dev/null; do
    kill -0 $bp 2>/dev/null || { echo "  [base_$tag] DIED"; return 1; }
    kill -0 $tp 2>/dev/null || { echo "  [$tag] DIED"; return 1; }
    (( w > 1800 )) && { echo "  [$tag] TIMEOUT"; kill -9 $bp $tp; return 1; }
    sleep 10; w=$((w+10))
  done
  local b
  for b in "base_$tag 8400" "$tag 8401"; do
    set -- $b
    ( "$VENV/bin/vllm" bench serve --model bench --tokenizer "$Q3" --trust-remote-code \
        --dataset-name random --random-input-len 128 --random-output-len 1024 \
        --num-prompts 64 --port "$2" > "$OUT/logs/$1.warmup.log" 2>&1
      "$VENV/bin/vllm" bench serve --model bench --tokenizer "$Q3" --trust-remote-code \
        --dataset-name random --random-input-len 128 --random-output-len 1024 \
        --num-prompts 200 --port "$2" --percentile-metrics ttft,tpot \
        --save-result --result-dir "$OUT" --result-filename "$1.json" \
        > "$OUT/logs/$1.bench.log" 2>&1
      printf '  [%s] %s\n' "$1" "$(grep -E 'Output token throughput' "$OUT/logs/$1.bench.log" | tr -s ' ')" ) &
  done
  wait
  kill $bp $tp 2>/dev/null; wait $bp $tp 2>/dev/null; sleep 5
}

echo "=== RESCALE A/B $(date '+%F %T') ==="
bash "$AR/scripts/drain_gpus.sh" >/dev/null 2>&1
cell f010_rn 0,1 8900 0.10 1 0 &
cell f010_mr 2,3 8901 0.10 0 1 &
wait
bash "$AR/scripts/drain_gpus.sh" >/dev/null 2>&1
cell f015_rn 0,1 8900 0.15 1 0 &
cell f015_mr 2,3 8901 0.15 0 1 &
wait
echo "--- speed overhead check (frac 0.15) ---"
speed_pair f015_mr_spd 0 1
speed_pair f015_rn_spd 1 0
echo "=== DONE $(date '+%F %T') ==="
