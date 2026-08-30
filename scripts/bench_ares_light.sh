#!/usr/bin/env bash
# How fast can ARES actually go? Map its speed frontier on Qwen3.
#
# Motivation: at CONTRIB_MASS_FRACTION=0.2 ARES prunes 65/128 experts per layer
# (51%) and returns only 1.175x throughput, while REAP deleting 50% of experts
# returns 1.83x and SERE returns 1.46x. Same nominal sparsity, a third of the
# speedup -- so ARES's problem is the conversion from experts-masked to
# wall-clock, not which experts it picks. Two knobs move in the speed direction:
#
#   CONFIDENCE_KEEP_SHARE=0  removes the per-token override kernel entirely AND
#                            stops resurrecting masked experts. On Qwen3 the
#                            rescue only claws back 2-5% of gate mass, so it
#                            should cost little accuracy -- but it runs on every
#                            token of every layer of every step.
#   CONTRIB_MASS_FRACTION up + MIN_ACTIVE_FRAC down  prune harder. MIN_ACTIVE_FRAC
#                            is a hard cap: at 0.25 no layer may lose more than
#                            96/128 experts, so raising the budget past ~0.35 does
#                            nothing until the floor is lowered too.
#
# Protocol identical to bench_v1_headtohead.sh so numbers are directly
# comparable: in=128 out=1024, 200 prompts, saturated, TP=2, one treatment beside
# one contemporaneous baseline per round.
set -uo pipefail

exec 9>/tmp/ares_frontier.lock
if ! flock -n 9; then echo "[frontier] locked; exiting"; exit 0; fi

VENV=/home/PC/new-efficient-moe/.venv
ARES_ROOT=/home/PC/new-efficient-moe
OUT=/home/PC/SERE_v1/results/frontier_light
LOGS=$OUT/logs
mkdir -p "$OUT" "$LOGS"
HUB=/home/PC/.cache/huggingface/hub
Q3=$(ls -d "$HUB"/models--Qwen--Qwen3-30B-A3B/snapshots/*/ | head -1); Q3=${Q3%/}

PYTHONPATH="$ARES_ROOT" "$VENV/bin/python" -c "
import sys,patches.vllm.fused_skip_ops as m
assert m.fused_cuda_available() and m.fused_confidence_available() and m.fused_risk_accum_available()
print('[frontier] ARES kernel: OK')" || exit 1

# serve_base <tag> <gpus> <port>
serve_base() {
  CUDA_VISIBLE_DEVICES=$2 "$VENV/bin/vllm" serve "$Q3" --served-model-name bench \
    --trust-remote-code --tensor-parallel-size 2 --max-model-len 4096 \
    --gpu-memory-utilization 0.90 --port "$3" --no-enable-prefix-caching \
    > "$LOGS/$1.server.log" 2>&1 &
  echo $! > "$LOGS/$1.pid"
}

# serve_ares <tag> <gpus> <port> <keep_share> <frac> <min_active>
serve_ares() {
  local tag=$1 gpus=$2 port=$3 ks=$4 frac=$5 ma=$6
  set -a; source "$ARES_ROOT/.env"; set +a
  env CUDA_VISIBLE_DEVICES="$gpus" PYTHONPATH="$ARES_ROOT" \
      EXPERT_SKIP_MODE=dynamic \
      EXPERT_SKIP_DISABLED_LAYERS=0,47 \
      EXPERT_SKIP_ONLINE_THRESHOLD_METHOD=contrib_mass \
      EXPERT_SKIP_ONLINE_CONFIDENCE_KEEP_SHARE="$ks" \
      EXPERT_SKIP_ONLINE_CONTRIB_MASS_FRACTION="$frac" \
      EXPERT_SKIP_ONLINE_MIN_ACTIVE_FRAC="$ma" \
      EXPERT_SKIP_ONLINE_MIN_TOKENS_FOR_FINALIZE="${EXPERT_SKIP_ONLINE_MIN_TOKENS_FOR_FINALIZE:-4096}" \
      EXPERT_SKIP_ONLINE_MIN_PER_EXPERT="${EXPERT_SKIP_ONLINE_MIN_PER_EXPERT:-8}" \
      EXPERT_SKIP_ONLINE_UNWEIGHTED_NORMS=1 \
      EXPERT_SKIP_ONLINE_REFRESH_EVERY_N_TOKENS=9999999 \
      EXPERT_SKIP_ONLINE_LOG=1 \
      EXPERT_SKIP_ONLINE_LOG_FILE="$LOGS/$tag.skip.log" \
      EXPERT_SKIP_ONLINE_DRIFT_CHECK_EVERY_N_STEPS=0 \
      EXPERT_SKIP_ONLINE_RISK_TRIGGER=0 \
      "$VENV/bin/vllm" serve "$Q3" --served-model-name bench \
      --trust-remote-code --tensor-parallel-size 2 --max-model-len 4096 \
      --gpu-memory-utilization 0.90 --port "$port" --no-enable-prefix-caching \
      > "$LOGS/$tag.server.log" 2>&1 &
  echo $! > "$LOGS/$tag.pid"
}

wait_up() {
  local tag=$1 port=$2 w=0 pid; pid=$(cat "$LOGS/$tag.pid")
  until curl -sf "http://127.0.0.1:$port/health" >/dev/null 2>&1; do
    kill -0 "$pid" 2>/dev/null || { echo "  [$tag] DIED"; grep -oE "(RuntimeError|CUDA error)[:.].{0,100}" "$LOGS/$tag.server.log" | sort -u | head -2; return 1; }
    (( w > 1800 )) && { echo "  [$tag] TIMEOUT"; kill -9 "$pid"; return 1; }
    sleep 10; w=$((w+10))
  done; echo "  [$tag] up ${w}s"
}

bench() {
  local tag=$1 port=$2
  "$VENV/bin/vllm" bench serve --model bench --tokenizer "$Q3" --trust-remote-code \
    --dataset-name random --random-input-len 128 --random-output-len 1024 \
    --num-prompts 64 --port "$port" > "$LOGS/$tag.warmup.log" 2>&1
  "$VENV/bin/vllm" bench serve --model bench --tokenizer "$Q3" --trust-remote-code \
    --dataset-name random --random-input-len 128 --random-output-len 1024 \
    --num-prompts 200 --port "$port" --percentile-metrics ttft,tpot,itl,e2el \
    --save-result --result-dir "$OUT" --result-filename "$tag.json" \
    > "$LOGS/$tag.bench.log" 2>&1
  printf '  [%s] %s | %s\n' "$tag" \
    "$(grep -E 'Output token throughput' "$LOGS/$tag.bench.log" | tr -s ' ')" \
    "$(grep -E 'Mean TPOT' "$LOGS/$tag.bench.log" | tr -s ' ')"
}

stop() { local p; p=$(cat "$LOGS/$1.pid" 2>/dev/null) || return 0; kill "$p" 2>/dev/null; wait "$p" 2>/dev/null; }

# round <n> <keep_share> <frac> <min_active>
round() {
  local n=$1 ks=$2 frac=$3 ma=$4
  local bt="base_f$n" tt="ares_f$n"
  [[ -f "$OUT/$tt.json" && -f "$OUT/$bt.json" ]] && { echo "[skip] $tt"; return 0; }
  bash "$ARES_ROOT/scripts/drain_gpus.sh" >/dev/null 2>&1
  echo "=== $tt (keep_share=$ks frac=$frac min_active=$ma) $(date '+%T') ==="
  serve_base "$bt" 0,1 8400; sleep 10
  serve_ares "$tt" 2,3 8401 "$ks" "$frac" "$ma"
  local ob=0 ot=0
  wait_up "$bt" 8400 && ob=1
  wait_up "$tt" 8401 && ot=1
  (( ob )) && { bench "$bt" 8400 & pb=$!; }
  (( ot )) && { bench "$tt" 8401 & pt=$!; }
  (( ob )) && wait ${pb:-0} 2>/dev/null
  (( ot )) && wait ${pt:-0} 2>/dev/null
  # Mask actually installed, and the runtime gate mass it really surrendered.
  [[ -s "$LOGS/$tt.skip.log" ]] && "$VENV/bin/python" - "$LOGS/$tt.skip.log" <<'PY'
import re,sys,statistics as st
t=open(sys.argv[1],errors="ignore").read()
r=re.findall(r"layer=(\d+) pruned=(\d+)/(\d+) mass_given_up=([0-9.]+) slot_share_masked=([0-9.]+)",t)
if r:
    last={int(a):(int(b),int(c),float(d),float(e)) for a,b,c,d,e in r}
    v=list(last.values())
    print(f"    mask: {st.mean(x[0] for x in v):.1f}/{v[0][1]} pruned/layer  "
          f"mass_budgeted={st.mean(x[2] for x in v):.4f}  slot_share={st.mean(x[3] for x in v):.4f}")
risk=[float(x) for x in re.findall(r"risk=([0-9.]+)",t)]
if risk: print(f"    runtime surrendered gate mass: median={st.median(risk):.4f} n={len(risk)}")
PY
  stop "$bt"; stop "$tt"; sleep 5
}

echo "=== ARES SPEED FRONTIER (Qwen3, saturated) $(date '+%F %T') ==="
echo "    reference: ARES frac0.2 keep0.15 = 1.175x | SERE S=1 rho=0.0 = 1.456x | REAP 0.50 = 1.83x"
round 5 0    0.05 0.25   # light budgets, to compare against percentile p50/p90
round 6 0    0.10 0.25   # at MATCHED accuracy rather than matched speed
round 7 0    0.15 0.25
echo "=== COMPLETE $(date '+%F %T') ==="
