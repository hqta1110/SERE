#!/usr/bin/env bash
# ARES vs SERE on ONE engine: vLLM 0.18.1 V1, same venv, same GPUs, same protocol.
#
# This is the measurement the cross-stack comparison could not support. SERE was
# previously only measurable on vLLM 0.8.4 V0, whose Qwen3 baseline is 1.54x
# slower than V1's, so neither speedup ratio could be compared to the other.
#
# Protocol: in=128 out=1024, 200 prompts, saturated (no --request-rate), TP=2 --
# identical to the REAP/SERE/ARES saturated runs in /home/PC/SERE/artifacts and
# /home/PC/new-efficient-moe/outputs/ares_ab/passB.
#
# CONTENTION IS PART OF THE MEASUREMENT. This box now has 4 GPUs, so exactly two
# TP=2 cells fit. Every round therefore runs ONE treatment beside ONE
# contemporaneous baseline, rather than reusing a baseline measured under
# different load. Four rounds also yield four independent baseline measurements,
# which is the run-to-run noise floor the earlier ARES numbers lacked.
#
#   round 1: base(0,1) + ARES(2,3)
#   round 2: base(0,1) + SERE S=1 rho=0.0(2,3)
#   round 3: base(0,1) + SERE S=1 rho=0.3(2,3)
#   round 4: base(0,1) + SERE S=2 rho=0.5(2,3)
set -uo pipefail

exec 9>/tmp/sere_v1_h2h.lock
if ! flock -n 9; then echo "[h2h] locked; exiting"; exit 0; fi

VENV=/home/PC/new-efficient-moe/.venv
ARES_ROOT=/home/PC/new-efficient-moe
OUT=/home/PC/SERE_v1/results/h2h
LOGS=$OUT/logs
mkdir -p "$OUT" "$LOGS"

HUB=/home/PC/.cache/huggingface/hub
Q3_BASE=$(ls -d "$HUB"/models--Qwen--Qwen3-30B-A3B/snapshots/*/ | head -1); Q3_BASE=${Q3_BASE%/}
Q3_SERE=/home/PC/SERE/artifacts/calibrated/Qwen3-30B-A3B-sere

NUM_PROMPTS=${NUM_PROMPTS:-200}
OUTLEN=${OUTLEN:-1024}
WARMUP=${WARMUP:-64}

# Refuse to run if ARES's kernel is not the current build: the eager confidence
# fallback is ~27-33x slower per layer and would understate ARES.
PYTHONPATH="$ARES_ROOT" "$VENV/bin/python" - <<'PY' || exit 1
import sys, patches.vllm.fused_skip_ops as m
ok = m.fused_cuda_available() and m.fused_confidence_available() and m.fused_risk_accum_available()
print(f"[h2h] ARES kernel probes: {m.fused_cuda_available()}/{m.fused_confidence_available()}/{m.fused_risk_accum_available()}")
sys.exit(0 if ok else 1)
PY
# Refuse to run if SERE's kernel is not built for THIS interpreter: the patch now
# hard-fails rather than silently using the graph-unsafe eager path, but check up
# front so the failure is not 10 minutes into a server start.
"$VENV/bin/python" -c "
from SERE_vllm.rerouting_cuda_ops import rerouting_ops_cuda; print('[h2h] SERE CUDA reroute: OK')" || exit 1

# serve <tag> <gpus> <port> <mode> [sere_S] [sere_rho]
#   mode: base | ares | sere
serve() {
  local tag=$1 gpus=$2 port=$3 mode=$4 S=${5:-} rho=${6:-}
  local srvlog="$LOGS/${tag}.server.log"
  local model="$Q3_BASE"
  local -a extra=()
  local -a envs=(CUDA_VISIBLE_DEVICES="$gpus")

  case "$mode" in
    base) ;;
    ares)
      # Replicate run_serving_sweep.sh's activation exactly, including the mask
      # freeze -- otherwise a reprofile window can open mid-run and silently mix
      # two configurations into one throughput number.
      set -a; source "$ARES_ROOT/.env"; set +a
      envs+=(PYTHONPATH="$ARES_ROOT" EXPERT_SKIP_MODE=dynamic
             EXPERT_SKIP_DISABLED_LAYERS=0,47
             EXPERT_SKIP_ONLINE_LOG=1
             EXPERT_SKIP_ONLINE_LOG_FILE="$LOGS/${tag}.skip.log"
             EXPERT_SKIP_ONLINE_DRIFT_CHECK_EVERY_N_STEPS=0
             EXPERT_SKIP_ONLINE_RISK_TRIGGER=0)
      for v in $(grep -E "^EXPERT_SKIP" "$ARES_ROOT/.env" | grep -vE "=$" \
                 | grep -vE "^EXPERT_SKIP_MODE=|^EXPERT_SKIP_ONLINE_LOG_FILE=|^EXPERT_SKIP_ONLINE_LOG=|^EXPERT_SKIP_DISABLED_LAYERS="); do
        envs+=("$v")
      done
      ;;
    sere)
      model="$Q3_SERE"
      # select_top_k in the config is what activates SERE (_is_sere_config).
      extra+=(--hf-overrides "{\"select_top_k\":$S,\"threshold\":$rho}")
      ;;
  esac

  echo "=== $tag (mode=$mode gpus=$gpus port=$port ${S:+S=$S rho=$rho}) $(date '+%T') ==="
  env "${envs[@]}" "$VENV/bin/vllm" serve "$model" \
      --served-model-name bench --trust-remote-code \
      --tensor-parallel-size 2 --max-model-len 4096 \
      --gpu-memory-utilization 0.90 --port "$port" \
      --no-enable-prefix-caching "${extra[@]}" > "$srvlog" 2>&1 &
  echo $! > "$LOGS/${tag}.pid"
}

wait_up() {
  local tag=$1 port=$2 waited=0
  local pid; pid=$(cat "$LOGS/${tag}.pid")
  until curl -sf "http://127.0.0.1:$port/health" > /dev/null 2>&1; do
    if ! kill -0 "$pid" 2>/dev/null; then
      echo "  [$tag] SERVER DIED"; grep -oE "(RuntimeError|CUDA error)[:.].{0,110}" "$LOGS/${tag}.server.log" | sort -u | head -3
      return 1
    fi
    (( waited > 1800 )) && { echo "  [$tag] TIMEOUT"; kill -9 "$pid"; return 1; }
    sleep 10; waited=$((waited+10))
  done
  echo "  [$tag] up in ${waited}s"
}

bench() {
  local tag=$1 port=$2 model=$3
  # Warmup: identical for every arm so caches/JIT state match. For ARES it is
  # also what builds the skip mask.
  "$VENV/bin/vllm" bench serve --model bench --tokenizer "$model" --trust-remote-code \
      --dataset-name random --random-input-len 128 --random-output-len "$OUTLEN" \
      --num-prompts "$WARMUP" --port "$port" > "$LOGS/${tag}.warmup.log" 2>&1
  local f="$LOGS/${tag}.bench.log"
  "$VENV/bin/vllm" bench serve --model bench --tokenizer "$model" --trust-remote-code \
      --dataset-name random --random-input-len 128 --random-output-len "$OUTLEN" \
      --num-prompts "$NUM_PROMPTS" --port "$port" \
      --percentile-metrics ttft,tpot,itl,e2el \
      --save-result --result-dir "$OUT" --result-filename "${tag}.json" > "$f" 2>&1
  if grep -q "Successful requests: *0" "$f"; then echo "  [$tag] FAIL: 0 requests"; return 1; fi
  printf '  [%s] %s | %s | %s\n' "$tag" \
    "$(grep -E 'Output token throughput' "$f" | tr -s ' ')" \
    "$(grep -E 'Mean TPOT' "$f" | tr -s ' ')" \
    "$(grep -E 'Mean TTFT' "$f" | tr -s ' ')"
}

stop() { local tag=$1; local p; p=$(cat "$LOGS/${tag}.pid" 2>/dev/null) || return 0
         kill "$p" 2>/dev/null; wait "$p" 2>/dev/null; }

# round <base_tag> <treat_tag> <treat_mode> [S] [rho]
round() {
  local bt=$1 tt=$2 tm=$3 S=${4:-} rho=${5:-}
  if [[ -f "$OUT/${tt}.json" && -f "$OUT/${bt}.json" ]]; then echo "[skip] $tt (done)"; return 0; fi
  bash "$ARES_ROOT/scripts/drain_gpus.sh" > /dev/null 2>&1
  serve "$bt" 0,1 8400 base
  sleep 10
  serve "$tt" 2,3 8401 "$tm" "$S" "$rho"
  local okb=0 okt=0
  wait_up "$bt" 8400 && okb=1
  wait_up "$tt" 8401 && okt=1
  local m="$Q3_BASE"; [[ "$tm" == sere ]] && m="$Q3_SERE"
  if (( okb )); then bench "$bt" 8400 "$Q3_BASE" & local pb=$!; fi
  if (( okt )); then bench "$tt" 8401 "$m" & local pt=$!; fi
  (( okb )) && wait ${pb:-0} 2>/dev/null
  (( okt )) && wait ${pt:-0} 2>/dev/null
  stop "$bt"; stop "$tt"
  sleep 5
}

echo "=== ARES vs SERE HEAD-TO-HEAD ON vLLM 0.18.1 V1  $(date '+%F %T') ==="
echo "    protocol: in=128 out=$OUTLEN prompts=$NUM_PROMPTS saturated TP=2, 2 cells/round"
round base_r1 ares      ares
round base_r2 sere_s1_r00 sere 1 0.0
round base_r3 sere_s1_r03 sere 1 0.3
round base_r4 sere_s2_r05 sere 2 0.5
echo "=== COMPLETE $(date '+%F %T') ==="
