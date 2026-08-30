#!/usr/bin/env bash
# Make Qwen1.5-MoE work. keep_share=0 gave 1.553x but destroyed it
# (GSM8K 60.20 -> 2.27, HumanEval 33.54 -> 2.44).
#
# Diagnosis: the mass budget is model-blind. frac 0.2 discards 20% of layer
# output mass either way, but that is 55% of Qwen3's 128 experts and 66% of
# Qwen1.5's 60 -- and with top_k=4 instead of 8, a Qwen1.5 token is left with
#   4 x (1 - 0.66) = 1.37 of 4 slots      vs Qwen3's 8 x (1 - 0.55) = 3.60 of 8
# One surviving expert per token is not a model. The per-token rescue was the
# only thing keeping Qwen1.5 alive, which is why removing it cost 1.51x of
# throughput there -- it was doing real work, not wasting effort.
#
# Two independent levers, tested separately so the effect is attributable:
#   A. LOWER THE BUDGET (ks=0): frac 0.03 / 0.05 / 0.10 -> prune fewer experts,
#      keep more slots alive, no rescue overhead at all.
#   B. RAISE THE THRESHOLD (frac 0.2): ks=0.40 / 0.55. The rescue fires when a
#      slot carries >= ks of the token's expected MoE mass, so a HIGH threshold
#      rescues only slots that dominate their token -- cheap protection for the
#      worst-hit tokens instead of the blanket rescue that ks=0.15 gave at
#      top_k=4 (where the mean slot share is already 0.25).
#
# TP=1 for Qwen1.5, so four cells fit: one baseline + three treatments per round.
set -uo pipefail

exec 9>/tmp/q15_sweep.lock
flock -n 9 || { echo "[q15] locked"; exit 0; }

VENV=/home/PC/new-efficient-moe/.venv
AR=/home/PC/new-efficient-moe
SV=/home/PC/SERE_v1
OUT=$SV/results/q15_sweep
mkdir -p "$OUT"
HUB=/home/PC/.cache/huggingface/hub
Q15=$(ls -d "$HUB"/models--Qwen--Qwen1.5-MoE-A2.7B/snapshots/*/ | head -1); Q15=${Q15%/}

up() {  # up <tag> <gpu> <port> <base|frac> <ks>
  # ks defaults: baseline cells are called with 4 args, and under `set -u` a bare
  # $5 aborts the function rather than defaulting.
  local tag=$1 gpu=$2 port=$3 frac=$4 ks=${5:-0}
  local -a e=(CUDA_VISIBLE_DEVICES="$gpu")
  if [[ "$frac" != base ]]; then
    set -a; source "$AR/.env"; set +a
    e+=(PYTHONPATH="$AR" EXPERT_SKIP_MODE=dynamic
        EXPERT_SKIP_DISABLED_LAYERS=0,23
        EXPERT_SKIP_ONLINE_THRESHOLD_METHOD=contrib_mass
        EXPERT_SKIP_ONLINE_CONFIDENCE_KEEP_SHARE="$ks"
        EXPERT_SKIP_ONLINE_CONTRIB_MASS_FRACTION="$frac"
        EXPERT_SKIP_ONLINE_MIN_ACTIVE_FRAC=0.25
        EXPERT_SKIP_ONLINE_UNWEIGHTED_NORMS=1
        EXPERT_SKIP_ONLINE_REFRESH_EVERY_N_TOKENS=9999999
        EXPERT_SKIP_ONLINE_LOG=1 EXPERT_SKIP_ONLINE_LOG_FILE="$OUT/$tag.skip.log"
        EXPERT_SKIP_ONLINE_DRIFT_CHECK_EVERY_N_STEPS=0
        EXPERT_SKIP_ONLINE_RISK_TRIGGER=0)
  fi
  env "${e[@]}" "$VENV/bin/vllm" serve "$Q15" --served-model-name m \
    --trust-remote-code --tensor-parallel-size 1 --max-model-len 8192 \
    --gpu-memory-utilization 0.90 --port "$port" --no-enable-prefix-caching \
    > "$OUT/$tag.server.log" 2>&1 &
  echo $! > "$OUT/$tag.pid"
  local w=0 pid; pid=$(cat "$OUT/$tag.pid")
  until curl -sf "http://127.0.0.1:$port/health" >/dev/null 2>&1; do
    kill -0 "$pid" 2>/dev/null || { echo "  [$tag] DIED"; return 1; }
    (( w > 1800 )) && { echo "  [$tag] TIMEOUT"; kill -9 "$pid"; return 1; }
    sleep 10; w=$((w+10))
  done; echo "  [$tag] up ${w}s"
}
down() { local p; p=$(cat "$OUT/$1.pid" 2>/dev/null) || return 0; kill "$p" 2>/dev/null; wait "$p" 2>/dev/null; }

spd() {
  "$VENV/bin/vllm" bench serve --model m --tokenizer "$Q15" --trust-remote-code \
    --dataset-name random --random-input-len 128 --random-output-len 1024 \
    --num-prompts 64 --port "$2" > "$OUT/$1.warm.log" 2>&1
  "$VENV/bin/vllm" bench serve --model m --tokenizer "$Q15" --trust-remote-code \
    --dataset-name random --random-input-len 128 --random-output-len 1024 \
    --num-prompts 200 --port "$2" --percentile-metrics ttft,tpot \
    --save-result --result-dir "$OUT" --result-filename "$1.json" > "$OUT/$1.bench.log" 2>&1
  printf '  [%s] %s\n' "$1" "$(grep -E 'Output token throughput' "$OUT/$1.bench.log"|tr -s ' ')"
}
msk() { [[ -s "$OUT/$1.skip.log" ]] && "$VENV/bin/python" - "$OUT/$1.skip.log" "$1" <<'PY'
import re,sys,statistics as st
t=open(sys.argv[1],errors="ignore").read()
r=re.findall(r"layer=(\d+) pruned=(\d+)/(\d+)",t)
if r:
    last={int(a):(int(b),int(c)) for a,b,c in r}; v=list(last.values())
    p=st.mean(x[0] for x in v); E=v[0][1]
    print(f"    [{sys.argv[2]}] pruned {p:.1f}/{E} ({p/E*100:.0f}%) -> surviving slots/token = {4*(1-p/E):.2f} of 4")
PY
}
acc() {  # acc <tag> <port> <bench> <has_mask>
  local tag=$1 port=$2 bench=$3 hm=$4
  local d="$OUT/$tag/$bench"
  find "$d" -name eval_results.json -size +0 2>/dev/null | grep -q . && return 0
  mkdir -p "$d"
  local src=gsm8k_train; [[ "$bench" == humaneval_base ]] && src=mbpp
  "$VENV/bin/python" "$SV/scripts/warm_until_mask.py" --port "$port" --model m \
    --warm-source "$src" $( [[ "$hm" == 1 ]] && echo --skip-log "$OUT/$tag.skip.log" ) \
    > "$d/warm.log" 2>&1
  PYTHONPATH="$AR" HF_ALLOW_CODE_EVAL=1 "$VENV/bin/python" \
    "$AR/scripts/serve/run_eval_serve.py" --task "$bench" --model_name "$Q15" \
    --local_model_path m --base_url "http://127.0.0.1:$port/v1" \
    --max_concurrency 128 --log_samples --output_dir "$d/run" > "$d/client.log" 2>&1
  echo "  [$tag/$bench] $(grep -ohE '"(pass@1,create_test|exact_match,flexible-extract)": [0-9.]+' "$d"/run/*/eval_results.json 2>/dev/null|head -1)"
}

echo "=== QWEN1.5 RESCUE SWEEP $(date '+%F %T') ==="
echo "    baseline (this box, TP=1): 2162.6 tok/s | GSM8K 60.20 | HumanEval 33.54"
echo "    broken reference: frac0.2 ks=0 -> 1.553x but GSM8K 2.27 / HEval 2.44"

echo "--- A: lower the budget (ks=0) ---"
bash "$AR/scripts/drain_gpus.sh" >/dev/null 2>&1
up q15b_a 0 8910 base & up f003_ks0 1 8911 0.03 0 & up f005_ks0 2 8912 0.05 0 & up f010_ks0 3 8913 0.10 0 &
wait
spd q15b_a 8910 & spd f003_ks0 8911 & spd f005_ks0 8912 & spd f010_ks0 8913 & wait
for t in f003_ks0 f005_ks0 f010_ks0; do msk "$t"; done
for t in f003_ks0 f005_ks0 f010_ks0; do
  port=$(( 8911 + $(echo "f003_ks0 f005_ks0 f010_ks0" | tr ' ' '\n' | grep -n "^$t$" | cut -d: -f1) - 1 ))
  acc "$t" "$port" gsm8k 1; acc "$t" "$port" humaneval_base 1
done
acc q15b_a 8910 gsm8k 0; acc q15b_a 8910 humaneval_base 0
for t in q15b_a f003_ks0 f005_ks0 f010_ks0; do down "$t"; done
sleep 5

echo "--- B: raise the rescue threshold, keep frac 0.2 ---"
bash "$AR/scripts/drain_gpus.sh" >/dev/null 2>&1
up q15b_b 0 8910 base & up f020_ks040 1 8911 0.2 0.40 & up f020_ks055 2 8912 0.2 0.55 & up f010_ks040 3 8913 0.10 0.40 &
wait
spd q15b_b 8910 & spd f020_ks040 8911 & spd f020_ks055 8912 & spd f010_ks040 8913 & wait
for t in f020_ks040 f020_ks055 f010_ks040; do msk "$t"; done
acc f020_ks040 8911 gsm8k 1; acc f020_ks040 8911 humaneval_base 1
acc f020_ks055 8912 gsm8k 1; acc f020_ks055 8912 humaneval_base 1
acc f010_ks040 8913 gsm8k 1; acc f010_ks040 8913 humaneval_base 1
for t in q15b_b f020_ks040 f020_ks055 f010_ks040; do down "$t"; done
echo "=== DONE $(date '+%F %T') ==="
