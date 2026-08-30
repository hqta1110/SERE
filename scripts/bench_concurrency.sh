#!/usr/bin/env bash
# ARES vs SERE across concurrency. This is the axis where the two mechanisms
# should diverge structurally, and every number measured so far is at the single
# most favourable point for SERE.
#
# SERE's primary set is the batch-wide union of each token's top-S experts. At
# batch 1 with S=1 that union is ONE expert, so 7 of 8 slots collapse onto it --
# measured directly: greedy generation degenerates to gibberish at concurrency 1,
# recovers partially by 64. ARES's mask is built once from profiling and does not
# depend on the current batch at all, so its quality should be flat in batch size
# while SERE's improves with it.
#
# If that holds, ARES owns the latency-sensitive / low-concurrency regime
# outright -- which is a defensible niche and directly attacks SERE's
# "efficient batch decoding" premise.
#
# One server per arm, concurrency swept inside it (4 bench runs per server) --
# far cheaper than a server restart per point, and the mask/similarity state is
# identical across the sweep by construction.
set -uo pipefail

exec 9>/tmp/ares_conc.lock
flock -n 9 || { echo "[conc] locked"; exit 0; }

VENV=/home/PC/new-efficient-moe/.venv
AR=/home/PC/new-efficient-moe
OUT=/home/PC/SERE_v1/results/concurrency
mkdir -p "$OUT"
HUB=/home/PC/.cache/huggingface/hub
Q3=$(ls -d "$HUB"/models--Qwen--Qwen3-30B-A3B/snapshots/*/ | head -1); Q3=${Q3%/}
Q3S=/home/PC/SERE/artifacts/calibrated/Qwen3-30B-A3B-sere
CONC="${CONC:-1 4 16 64}"

up() {  # up <tag> <gpus> <port> <base|ares|sere>
  local tag=$1 gpus=$2 port=$3 mode=$4 model=$Q3
  local -a e=(CUDA_VISIBLE_DEVICES="$gpus") extra=()
  case "$mode" in
    ares) set -a; source "$AR/.env"; set +a
          e+=(PYTHONPATH="$AR" EXPERT_SKIP_MODE=dynamic
              EXPERT_SKIP_DISABLED_LAYERS=0,47
              EXPERT_SKIP_ONLINE_THRESHOLD_METHOD=contrib_mass
              EXPERT_SKIP_ONLINE_CONFIDENCE_KEEP_SHARE=0
              EXPERT_SKIP_ONLINE_CONTRIB_MASS_FRACTION=0.2
              EXPERT_SKIP_ONLINE_MIN_ACTIVE_FRAC=0.25
              EXPERT_SKIP_ONLINE_UNWEIGHTED_NORMS=1
              EXPERT_SKIP_ONLINE_REFRESH_EVERY_N_TOKENS=9999999
              EXPERT_SKIP_ONLINE_LOG=1 EXPERT_SKIP_ONLINE_LOG_FILE="$OUT/$tag.skip.log"
              EXPERT_SKIP_ONLINE_DRIFT_CHECK_EVERY_N_STEPS=0
              EXPERT_SKIP_ONLINE_RISK_TRIGGER=0) ;;
    sere) model=$Q3S; extra+=(--hf-overrides '{"select_top_k":1,"threshold":0.0}') ;;
  esac
  env "${e[@]}" "$VENV/bin/vllm" serve "$model" --served-model-name m \
    --trust-remote-code --tensor-parallel-size 2 --max-model-len 4096 \
    --gpu-memory-utilization 0.90 --port "$port" --no-enable-prefix-caching \
    "${extra[@]}" > "$OUT/$tag.server.log" 2>&1 &
  echo $! > "$OUT/$tag.pid"
  local w=0 pid; pid=$(cat "$OUT/$tag.pid")
  until curl -sf "http://127.0.0.1:$port/health" >/dev/null 2>&1; do
    kill -0 "$pid" 2>/dev/null || { echo "  [$tag] DIED"; return 1; }
    (( w > 1800 )) && { echo "  [$tag] TIMEOUT"; kill -9 "$pid"; return 1; }
    sleep 10; w=$((w+10))
  done; echo "  [$tag] up ${w}s"
}
down() { local p; p=$(cat "$OUT/$1.pid" 2>/dev/null) || return 0; kill "$p" 2>/dev/null; wait "$p" 2>/dev/null; }

sweep() {  # sweep <tag> <port> <model>
  local tag=$1 port=$2 model=$3
  # Warmup once, so ARES's mask exists before ANY timed point. Its size is
  # independent of the concurrency swept afterwards.
  "$VENV/bin/vllm" bench serve --model m --tokenizer "$model" --trust-remote-code \
    --dataset-name random --random-input-len 128 --random-output-len 256 \
    --num-prompts 64 --port "$port" > "$OUT/$tag.warm.log" 2>&1
  for c in $CONC; do
    local n=$(( c * 4 )); (( n < 32 )) && n=32
    "$VENV/bin/vllm" bench serve --model m --tokenizer "$model" --trust-remote-code \
      --dataset-name random --random-input-len 128 --random-output-len 256 \
      --num-prompts "$n" --max-concurrency "$c" --port "$port" \
      --percentile-metrics ttft,tpot \
      --save-result --result-dir "$OUT" --result-filename "${tag}_c${c}.json" \
      > "$OUT/${tag}_c${c}.log" 2>&1
    printf '  [%s c=%-3s] %s | %s\n' "$tag" "$c" \
      "$(grep -E 'Output token throughput' "$OUT/${tag}_c${c}.log"|tr -s ' ')" \
      "$(grep -E 'Mean TPOT' "$OUT/${tag}_c${c}.log"|tr -s ' ')"
  done
}

echo "=== CONCURRENCY SWEEP (Qwen3, out=256) $(date '+%F %T') ==="
echo "    concurrency: $CONC   ARES=contrib_mass frac0.2 ks=0   SERE=S1 rho0.0"
for pair in "base_a ares" "base_b sere"; do
  set -- $pair; bt=$1 tm=$2
  bash "$AR/scripts/drain_gpus.sh" >/dev/null 2>&1
  echo "--- $tm vs baseline ---"
  up "$bt" 0,1 8920 base && up "$tm" 2,3 8921 "$tm" && {
    m=$Q3; [[ "$tm" == sere ]] && m=$Q3S
    sweep "$bt" 8920 "$Q3" & p1=$!
    sweep "$tm" 8921 "$m" & p2=$!
    wait $p1 $p2; }
  down "$bt"; down "$tm"; sleep 5
done
echo "=== DONE $(date '+%F %T') ==="
