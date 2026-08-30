#!/usr/bin/env bash
# Concurrency sweep at the QUALITY-SAFE budgets. Plugs a real hole in the paper.
#
# The existing concurrency table (results/concurrency) used ARES frac 0.20 --
# which we now know sits PAST the accuracy cliff (-19.71 GSM8K strict). So it
# compares SERE at -1.74 points against an ARES setting nobody would deploy.
# The shippable settings are frac 0.05 (+0.15 points, i.e. free) and frac 0.10
# (-3.33). This re-runs the same sweep at those, so the paper can state the
# speed-vs-concurrency picture at settings that preserve quality.
#
# Expectation, to be honest about it up front: frac 0.05/0.10 are SLOWER than
# frac 0.20, so ARES should look WORSE against SERE here than the existing table
# shows. Running it anyway -- the current table is not a defensible comparison,
# and a reviewer would ask for exactly this.
#
# Protocol identical to bench_concurrency.sh: out=256, one server per arm,
# concurrency swept inside it, warmup once so the ARES mask exists before any
# timed point. No bare `wait`.
set -uo pipefail

exec 9>/tmp/ares_safeconc.lock
flock -n 9 || { echo "[sc] locked"; exit 0; }

VENV=/home/PC/new-efficient-moe/.venv
AR=/home/PC/new-efficient-moe
OUT=/home/PC/SERE_v1/results/safe_concurrency
LOGS=$OUT/logs
mkdir -p "$OUT" "$LOGS"
HUB=/home/PC/.cache/huggingface/hub
Q3=$(ls -d "$HUB"/models--Qwen--Qwen3-30B-A3B/snapshots/*/ | head -1); Q3=${Q3%/}
CONC="${CONC:-1 4 16 64}"

serve() {  # serve <tag> <gpus> <port> <base|ares:FRAC>
  local tag=$1 gpus=$2 port=$3 mode=$4
  local -a e=(CUDA_VISIBLE_DEVICES="$gpus")
  if [[ "$mode" == ares:* ]]; then
    set -a; source "$AR/.env"; set +a
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
        EXPERT_SKIP_ONLINE_RISK_TRIGGER=0)
  fi
  env "${e[@]}" "$VENV/bin/vllm" serve "$Q3" --served-model-name bench \
    --trust-remote-code --tensor-parallel-size 2 --max-model-len 4096 \
    --gpu-memory-utilization 0.90 --port "$port" --no-enable-prefix-caching \
    > "$LOGS/$tag.server.log" 2>&1 &
  echo $! > "$LOGS/$tag.pid"
}
wait_up() {
  local tag=$1 port=$2 w=0 pid; pid=$(cat "$LOGS/$tag.pid")
  until curl -sf "http://127.0.0.1:$port/health" >/dev/null 2>&1; do
    kill -0 "$pid" 2>/dev/null || { echo "  [$tag] DIED"; return 1; }
    (( w > 2400 )) && { echo "  [$tag] TIMEOUT"; kill -9 "$pid"; return 1; }
    sleep 10; w=$((w+10))
  done; echo "  [$tag] up ${w}s"
}
stop() { local p; p=$(cat "$LOGS/$1.pid" 2>/dev/null) || return 0
  kill "$p" 2>/dev/null; wait "$p" 2>/dev/null; }

sweep() {
  local tag=$1 port=$2
  "$VENV/bin/vllm" bench serve --model bench --tokenizer "$Q3" --trust-remote-code \
    --dataset-name random --random-input-len 128 --random-output-len 256 \
    --num-prompts 64 --port "$port" > "$LOGS/$tag.warmup.log" 2>&1
  local c
  for c in $CONC; do
    local n=$(( c * 4 )); (( n < 32 )) && n=32
    [[ -f "$OUT/${tag}_c${c}.json" ]] && { echo "    [${tag}_c${c}] cached"; continue; }
    "$VENV/bin/vllm" bench serve --model bench --tokenizer "$Q3" --trust-remote-code \
      --dataset-name random --random-input-len 128 --random-output-len 256 \
      --num-prompts "$n" --max-concurrency "$c" --port "$port" \
      --percentile-metrics ttft,tpot --save-result --result-dir "$OUT" \
      --result-filename "${tag}_c${c}.json" > "$LOGS/${tag}_c${c}.log" 2>&1
    printf '    [%s c=%-3s] %s\n' "$tag" "$c" \
      "$(grep -E 'Output token throughput' "$LOGS/${tag}_c${c}.log" | tr -s ' ')"
  done
}

round() {  # round <treat_tag> <mode>
  local tt=$1 mode=$2 bt="base_$1"
  bash "$AR/scripts/drain_gpus.sh" >/dev/null 2>&1
  echo "--- $tt ($mode)  $(date '+%T')"
  serve "$bt" 0,1 8700 base
  serve "$tt" 2,3 8701 "$mode"
  local ob=0 ot=0 pb="" pt=""
  wait_up "$bt" 8700 && ob=1
  wait_up "$tt" 8701 && ot=1
  (( ob )) && { sweep "$bt" 8700 & pb=$!; }
  (( ot )) && { sweep "$tt" 8701 & pt=$!; }
  [[ -n "$pb" ]] && wait "$pb"
  [[ -n "$pt" ]] && wait "$pt"
  stop "$bt"; stop "$tt"; sleep 5
}

echo "=== QUALITY-SAFE CONCURRENCY (Qwen3-30B, out=256) $(date '+%F %T') ==="
round ares005 ares:0.05
round ares010 ares:0.10

echo "=== SUMMARY $(date '+%F %T') ==="
"$VENV/bin/python" - "$OUT" "$CONC" <<'PY'
import json, os, sys
OUT, CONC = sys.argv[1], sys.argv[2].split()
def thr(tag, c):
    p = os.path.join(OUT, f"{tag}_c{c}.json")
    if not os.path.exists(p): return None
    d = json.load(open(p))
    if d.get("completed", 0) != d.get("num_prompts", 0):
        print(f"  !! {tag} c{c}: completed != num_prompts (INVALID)")
    return d.get("output_throughput")
print(f"  {'conc':<8}{'ARES frac0.05':>15}{'ARES frac0.10':>15}")
for c in CONC:
    row = [f"  c={c:<6}"]
    for tag in ("ares005", "ares010"):
        b, t = thr(f"base_{tag}", c), thr(tag, c)
        row.append(f"{t/b:.3f}x".rjust(15) if (b and t) else "--".rjust(15))
    print("".join(row))
print("  (for reference, the past-cliff frac 0.20 table: 1.055/1.236/1.485/1.700x")
print("   and SERE S1 rho0: 1.125/1.390/1.712/1.718x at c=1/4/16/64)")
PY
echo "=== DONE $(date '+%F %T') ==="
