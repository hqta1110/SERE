#!/usr/bin/env bash
# Two things the frontier run left open.
#
# 1. BASELINE ACCURACY REPLICATES. There are four baseline *speed* replicates
#    (0.34% spread) but only one baseline *accuracy* measurement, so the accuracy
#    noise band is unknown -- and it clearly is not the binomial stderr: configs
#    that should differ modestly spanned 82.2-90.1 on GSM8K. Without this band no
#    small accuracy delta here means anything, in either direction.
#
# 2. frac 0.25. Speed is the objective and the quality cliff sits between frac
#    0.2 (HumanEval 29.27, passes) and frac 0.35 (9.15, fails). If 0.25 holds
#    quality it is free speed; if it fails, the cliff is sharp and 0.2 is the
#    documented edge. Either outcome is worth knowing.
set -uo pipefail

exec 9>/tmp/ares_final.lock
flock -n 9 || { echo "[final] locked"; exit 0; }

VENV=/home/PC/new-efficient-moe/.venv
ARES_ROOT=/home/PC/new-efficient-moe
OUT=/home/PC/SERE_v1/results/final
mkdir -p "$OUT"
HUB=/home/PC/.cache/huggingface/hub
Q3=$(ls -d "$HUB"/models--Qwen--Qwen3-30B-A3B/snapshots/*/ | head -1); Q3=${Q3%/}

# serve <tag> <gpus> <port> <mode:base|ares> [frac] [min_active]
serve() {
  local tag=$1 gpus=$2 port=$3 mode=$4 frac=${5:-} ma=${6:-}
  local log=$OUT/$tag.server.log
  if [[ "$mode" == base ]]; then
    CUDA_VISIBLE_DEVICES=$gpus "$VENV/bin/vllm" serve "$Q3" --served-model-name m \
      --trust-remote-code --tensor-parallel-size 2 --max-model-len 8192 \
      --gpu-memory-utilization 0.90 --port "$port" > "$log" 2>&1 &
  else
    set -a; source "$ARES_ROOT/.env"; set +a
    env CUDA_VISIBLE_DEVICES="$gpus" PYTHONPATH="$ARES_ROOT" \
      EXPERT_SKIP_MODE=dynamic EXPERT_SKIP_DISABLED_LAYERS=0,47 \
      EXPERT_SKIP_ONLINE_THRESHOLD_METHOD=contrib_mass \
      EXPERT_SKIP_ONLINE_CONFIDENCE_KEEP_SHARE=0 \
      EXPERT_SKIP_ONLINE_CONTRIB_MASS_FRACTION="$frac" \
      EXPERT_SKIP_ONLINE_MIN_ACTIVE_FRAC="$ma" \
      EXPERT_SKIP_ONLINE_UNWEIGHTED_NORMS=1 \
      EXPERT_SKIP_ONLINE_REFRESH_EVERY_N_TOKENS=9999999 \
      EXPERT_SKIP_ONLINE_LOG=1 EXPERT_SKIP_ONLINE_LOG_FILE="$OUT/$tag.skip.log" \
      EXPERT_SKIP_ONLINE_DRIFT_CHECK_EVERY_N_STEPS=0 \
      EXPERT_SKIP_ONLINE_RISK_TRIGGER=0 \
      "$VENV/bin/vllm" serve "$Q3" --served-model-name m \
      --trust-remote-code --tensor-parallel-size 2 --max-model-len 8192 \
      --gpu-memory-utilization 0.90 --port "$port" > "$log" 2>&1 &
  fi
  echo $! > "$OUT/$tag.pid"
  local w=0 pid; pid=$(cat "$OUT/$tag.pid")
  until curl -sf "http://127.0.0.1:$port/health" >/dev/null 2>&1; do
    kill -0 "$pid" 2>/dev/null || { echo "  [$tag] DIED"; return 1; }
    (( w > 1800 )) && { echo "  [$tag] TIMEOUT"; kill -9 "$pid"; return 1; }
    sleep 10; w=$((w+10))
  done; echo "  [$tag] up ${w}s"
}
stop() { local p; p=$(cat "$OUT/$1.pid" 2>/dev/null) || return 0; kill "$p" 2>/dev/null; wait "$p" 2>/dev/null; }

acc() {  # acc <tag> <port> <bench>
  # Two statements on purpose: in a single `local a=$1 b="$a"`, every word is
  # expanded before any assignment happens, so $tag would still be unbound here
  # and `set -u` aborts the function.
  local tag=$1 port=$2 bench=$3
  local d="$OUT/$tag/$bench"
  find "$d" -name eval_results.json -size +0 2>/dev/null | grep -q . && { echo "  [$tag/$bench] scored"; return 0; }
  mkdir -p "$d"
  PYTHONPATH="$ARES_ROOT" HF_ALLOW_CODE_EVAL=1 "$VENV/bin/python" \
    "$ARES_ROOT/scripts/serve/run_eval_serve.py" --task "$bench" \
    --model_name "$Q3" --local_model_path m --base_url "http://127.0.0.1:$port/v1" \
    --max_concurrency 128 --log_samples --output_dir "$d/run" > "$d/client.log" 2>&1
  echo "  [$tag/$bench] $(grep -ohE '"(pass@1,create_test|exact_match,flexible-extract)": [0-9.]+' "$d"/run/*/eval_results.json 2>/dev/null | head -1)"
}

speed() {  # speed <tag> <port>
  "$VENV/bin/vllm" bench serve --model m --tokenizer "$Q3" --trust-remote-code \
    --dataset-name random --random-input-len 128 --random-output-len 1024 \
    --num-prompts 64 --port "$2" > "$OUT/$1.warm.log" 2>&1
  "$VENV/bin/vllm" bench serve --model m --tokenizer "$Q3" --trust-remote-code \
    --dataset-name random --random-input-len 128 --random-output-len 1024 \
    --num-prompts 200 --port "$2" --percentile-metrics ttft,tpot,itl,e2el \
    --save-result --result-dir "$OUT" --result-filename "$1.json" > "$OUT/$1.bench.log" 2>&1
  printf '  [%s] %s\n' "$1" "$(grep -E 'Output token throughput' "$OUT/$1.bench.log" | tr -s ' ')"
}

echo "=== FINAL CHECKS $(date '+%F %T') ==="
bash "$ARES_ROOT/scripts/drain_gpus.sh" >/dev/null 2>&1

# Round A: frac 0.25 speed (paired) -- max-model-len 8192 here vs 4096 in the
# frontier run, so the PAIRED baseline is what it must be compared against.
echo "--- A: frac 0.25 speed, paired baseline ---"
serve base_s 0,1 8700 base && serve ares025_s 2,3 8701 ares 0.25 0.25 && {
  speed base_s 8700 & p1=$!; speed ares025_s 8701 & p2=$!; wait $p1 $p2; }
stop base_s; stop ares025_s; sleep 5
bash "$ARES_ROOT/scripts/drain_gpus.sh" >/dev/null 2>&1

# Round B: frac 0.25 accuracy + baseline replicate #2, concurrently
echo "--- B: frac 0.25 accuracy + baseline replicate 2 ---"
serve base_rep2 0,1 8700 base && { for b in gsm8k humaneval_base; do acc base_rep2 8700 "$b"; done; } &
pa=$!
serve ares025 2,3 8701 ares 0.25 0.25 && { for b in gsm8k humaneval_base; do acc ares025 8701 "$b"; done; } &
pb=$!
wait $pa $pb; stop base_rep2; stop ares025; sleep 5
bash "$ARES_ROOT/scripts/drain_gpus.sh" >/dev/null 2>&1

# Round C: baseline replicate #3 -- three points is the minimum for a spread
echo "--- C: baseline replicate 3 ---"
serve base_rep3 0,1 8700 base && { for b in gsm8k humaneval_base; do acc base_rep3 8700 "$b"; done; }
stop base_rep3
echo "=== DONE $(date '+%F %T') ==="
