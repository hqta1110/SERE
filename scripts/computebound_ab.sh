#!/usr/bin/env bash
# Does masking WIN where compute binds? This is the one queued experiment that
# could still produce a positive headline for ARES.
#
# The argument, from measured numbers:
#   ARES frac 0.20 masks 55% of experts and leaves 3.60 of 8 routed slots alive
#     -> it performs ~45% of the baseline's expert ARITHMETIC.
#   REAP deletes 50% of experts but still routes top_k=8 slots onto survivors
#     -> it performs ~100% of the baseline's expert arithmetic, from half the
#        weights. Its win is WEIGHT MEMORY TRAFFIC plus the HBM it frees.
#   SERE reroutes secondary slots onto primary experts -> also keeps 8 slots.
#
# Measured: masked ARES gets the SAME KV cache as baseline (5.82 GiB / 127,184
# tokens) because masked weights stay resident. So in a memory-bound regime
# ARES cannot win on footprint, and every number collected so far is
# memory-bound (saturated decode, out=1024).
#
# But arithmetic is the one axis where ARES is strictly ahead. So in a
# COMPUTE-BOUND regime -- prefill-heavy, long input, short output, where each
# expert sees many tokens and its weight load is amortised over them -- ARES's
# 45%-of-FLOPs should beat both SERE and deletion. If that shows up, the paper
# gets a real claim: "masking dominates when compute binds; deletion dominates
# when memory binds", with a mechanism and a measurement for each.
#
# Protocol: identical to bench_v1_headtohead.sh EXCEPT the traffic shape.
# One treatment beside one contemporaneous baseline on the other GPU pair, TP=2.
# Three shapes, prefill share rising left to right:
#   in=1024 out=256   (~4:1 prefill:decode tokens)
#   in=4096 out=128   (~32:1)
#   in=8192 out=64    (~128:1)
# NOTE: no bare `wait` anywhere -- a bare wait also waits on the server PIDs and
# hangs forever (that bug cost 43 minutes of GPU time earlier today).
set -uo pipefail

exec 9>/tmp/ares_computebound.lock
flock -n 9 || { echo "[cb] locked"; exit 0; }

VENV=/home/PC/new-efficient-moe/.venv
AR=/home/PC/new-efficient-moe
OUT=/home/PC/SERE_v1/results/computebound
LOGS=$OUT/logs
mkdir -p "$OUT" "$LOGS"
HUB=/home/PC/.cache/huggingface/hub
Q3=$(ls -d "$HUB"/models--Qwen--Qwen3-30B-A3B/snapshots/*/ | head -1); Q3=${Q3%/}
Q3S=/home/PC/SERE/artifacts/calibrated/Qwen3-30B-A3B-sere

PYTHONPATH="$AR" "$VENV/bin/python" -c "
import patches.vllm.fused_skip_ops as m
assert m.fused_cuda_available() and m.fused_confidence_available() and m.fused_risk_accum_available()
print('[cb] ARES kernels: OK')" || exit 1

# serve <tag> <gpus> <port> <base|ares:FRAC|sere>
serve() {
  local tag=$1 gpus=$2 port=$3 mode=$4 model=$Q3
  local -a e=(CUDA_VISIBLE_DEVICES="$gpus") extra=()
  case "$mode" in
    ares:*) set -a; source "$AR/.env"; set +a
      e+=(PYTHONPATH="$AR" EXPERT_SKIP_MODE=dynamic
          EXPERT_SKIP_DISABLED_LAYERS=0,47
          EXPERT_SKIP_ONLINE_THRESHOLD_METHOD=contrib_mass
          EXPERT_SKIP_ONLINE_CONFIDENCE_KEEP_SHARE=0
          EXPERT_SKIP_ONLINE_CONTRIB_MASS_FRACTION="${mode#ares:}"
          EXPERT_SKIP_ONLINE_MIN_ACTIVE_FRAC=0.25
          EXPERT_SKIP_ONLINE_UNWEIGHTED_NORMS=1
          EXPERT_SKIP_ONLINE_REFRESH_EVERY_N_TOKENS=9999999
          EXPERT_SKIP_ONLINE_LOG=1 EXPERT_SKIP_ONLINE_LOG_FILE="$LOGS/$tag.skip.log"
          EXPERT_SKIP_ONLINE_DRIFT_CHECK_EVERY_N_STEPS=0
          EXPERT_SKIP_ONLINE_RISK_TRIGGER=0) ;;
    sere) model=$Q3S; extra+=(--hf-overrides '{"select_top_k":1,"threshold":0.0}') ;;
  esac
  env "${e[@]}" "$VENV/bin/vllm" serve "$model" --served-model-name bench \
    --trust-remote-code --tensor-parallel-size 2 --max-model-len 16384 \
    --gpu-memory-utilization 0.90 --port "$port" --no-enable-prefix-caching \
    "${extra[@]}" > "$LOGS/$tag.server.log" 2>&1 &
  echo $! > "$LOGS/$tag.pid"
}
wait_up() {
  local tag=$1 port=$2 w=0 pid; pid=$(cat "$LOGS/$tag.pid")
  until curl -sf "http://127.0.0.1:$port/health" >/dev/null 2>&1; do
    kill -0 "$pid" 2>/dev/null || { echo "  [$tag] DIED"; \
      grep -oE "(ValueError|RuntimeError|CUDA error|OutOfMemoryError)[:.].{0,140}" \
        "$LOGS/$tag.server.log" | sort -u | head -2; return 1; }
    (( w > 2400 )) && { echo "  [$tag] TIMEOUT"; kill -9 "$pid"; return 1; }
    sleep 10; w=$((w+10))
  done
  echo "  [$tag] up ${w}s  KV=$(grep -oE 'GPU KV cache size: [0-9,]+ tokens' "$LOGS/$tag.server.log" | tail -1)"
}
stop() { local p; p=$(cat "$LOGS/$1.pid" 2>/dev/null) || return 0
  kill "$p" 2>/dev/null; wait "$p" 2>/dev/null; }

# bench_shapes <tag> <port> <model>: warmup (builds+installs the ARES mask), then
# one measured run per traffic shape. All foreground -- no bare wait.
bench_shapes() {
  local tag=$1 port=$2 model=$3
  "$VENV/bin/vllm" bench serve --model bench --tokenizer "$model" --trust-remote-code \
    --dataset-name random --random-input-len 1024 --random-output-len 128 \
    --num-prompts 64 --port "$port" > "$LOGS/$tag.warmup.log" 2>&1
  local shape
  for shape in "1024 256" "4096 128" "8192 64"; do
    set -- $shape; local il=$1 ol=$2
    local rf="${tag}_i${il}_o${ol}"
    [[ -f "$OUT/$rf.json" ]] && { echo "    [$rf] cached"; continue; }
    "$VENV/bin/vllm" bench serve --model bench --tokenizer "$model" --trust-remote-code \
      --dataset-name random --random-input-len "$il" --random-output-len "$ol" \
      --num-prompts 128 --port "$port" --percentile-metrics ttft,tpot \
      --save-result --result-dir "$OUT" --result-filename "$rf.json" \
      > "$LOGS/$rf.bench.log" 2>&1
    printf '    [%s] %s | %s\n' "$rf" \
      "$(grep -E 'Output token throughput' "$LOGS/$rf.bench.log" | tr -s ' ')" \
      "$(grep -E 'Total Token throughput' "$LOGS/$rf.bench.log" | tr -s ' ')"
  done
}

# round <treat_tag> <mode> : baseline on GPUs 0,1 and treatment on 2,3, together
round() {
  local tt=$1 mode=$2 bt="base_$1"
  bash "$AR/scripts/drain_gpus.sh" >/dev/null 2>&1
  echo "--- $tt ($mode) vs contemporaneous baseline  $(date '+%T')"
  serve "$bt" 0,1 8600 base
  serve "$tt" 2,3 8601 "$mode"
  local ob=0 ot=0
  wait_up "$bt" 8600 && ob=1
  wait_up "$tt" 8601 && ot=1
  local m=$Q3; [[ "$mode" == sere ]] && m=$Q3S
  # Run the two arms' bench suites concurrently, then wait on THOSE pids only.
  local pb="" pt=""
  (( ob )) && { bench_shapes "$bt" 8600 "$Q3" & pb=$!; }
  (( ot )) && { bench_shapes "$tt" 8601 "$m"  & pt=$!; }
  [[ -n "$pb" ]] && wait "$pb"
  [[ -n "$pt" ]] && wait "$pt"
  if [[ "$mode" == ares:* && -s "$LOGS/$tt.skip.log" ]]; then
    "$VENV/bin/python" - "$LOGS/$tt.skip.log" "$tt" <<'PY'
import re,sys,statistics as st
t=open(sys.argv[1],errors='ignore').read()
r=re.findall(r"layer=(\d+) pruned=(\d+)/(\d+)",t); last={}
for a,b,c in r: last[int(a)]=(int(b),int(c))
if last:
    v=list(last.values()); E=v[0][1]; p=st.mean(x[0] for x in v)
    print(f"    [{sys.argv[2]}] pruned {p:.1f}/{E} ({p/E*100:.0f}%) -> slots {8*(1-p/E):.2f}/8")
PY
  fi
  stop "$bt"; stop "$tt"; sleep 5
}

echo "=== COMPUTE-BOUND A/B (Qwen3-30B, TP=2) $(date '+%F %T') ==="
echo "    every prior number was memory-bound decode; these are prefill-heavy"
round ares010 ares:0.10
round ares020 ares:0.20
round sere    sere

echo "=== SUMMARY $(date '+%F %T') ==="
"$VENV/bin/python" - "$OUT" <<'PY'
import json, os, sys
OUT = sys.argv[1]
def g(tag, il, ol, field):
    p = os.path.join(OUT, f"{tag}_i{il}_o{ol}.json")
    if not os.path.exists(p): return None
    d = json.load(open(p))
    if d.get("completed", 0) != d.get("num_prompts", 0):
        print(f"  !! {tag} i{il} o{ol}: completed != num_prompts (INVALID)")
    return d.get(field)
print(f"  {'shape':<14}{'arm':<10}{'out tok/s':>11}{'total tok/s':>13}{'x(total)':>10}")
for il, ol in ((1024,256),(4096,128),(8192,64)):
    for tag, label in (("ares010","ARES.10"), ("ares020","ARES.20"), ("sere","SERE")):
        bt = g(f"base_{tag}", il, ol, "total_token_throughput")
        tv = g(tag, il, ol, "total_token_throughput")
        ov = g(tag, il, ol, "output_throughput")
        if bt and tv:
            print(f"  in{il}/out{ol:<6}{label:<10}{ov:>11.1f}{tv:>13.1f}{tv/bt:>9.3f}x")
PY
echo "=== DONE $(date '+%F %T') ==="
