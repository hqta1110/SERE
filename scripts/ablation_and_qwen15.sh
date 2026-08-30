#!/usr/bin/env bash
# Two experiments.
#
# A. ABLATION (control, not a direction): does the OLD `percentile` rule also
#    reach ~1.6x once the per-token rescue is off? The 1.623x appeared when
#    CONFIDENCE_KEEP_SHARE went 0.15 -> 0, so the win is currently confounded
#    between "better scoring" and "rescue was a bug". contrib_mass's advantage is
#    so far supported only by OFFLINE analysis (Spearman +0.997 vs +0.482) on a
#    different checkpoint, never end-to-end here.
#      percentile has no mass budget; aggressiveness comes from the mean/var
#    percentiles, and the documented claim is that the `mean<pX AND var<pY` gate
#    caps candidates at ~35-49 of 128 -- i.e. it may be unable to reach the
#    70.4/128 that contrib_mass prunes at frac 0.2. That cap is the thing under
#    test, so it is run at the old default AND pushed hard.
#
# B. QWEN1.5 at keep_share=0. It previously showed 0.96x (a net LOSS) with the
#    rescue on, while the rescue was clawing back ~65% of its mask -- the largest
#    single unclaimed speedup available. top_k=4 there, so the 0.15 threshold sat
#    BELOW the mean per-slot share of 0.25, which is why it was so destructive.
#    Qwen1.5 is TP=1, so all four cells run at once.
set -uo pipefail

exec 9>/tmp/ares_ablation.lock
flock -n 9 || { echo "[abl] locked"; exit 0; }

VENV=/home/PC/new-efficient-moe/.venv
AR=/home/PC/new-efficient-moe
OUT=/home/PC/SERE_v1/results/ablation
mkdir -p "$OUT"
HUB=/home/PC/.cache/huggingface/hub
Q3=$(ls -d "$HUB"/models--Qwen--Qwen3-30B-A3B/snapshots/*/ | head -1); Q3=${Q3%/}
Q15=$(ls -d "$HUB"/models--Qwen--Qwen1.5-MoE-A2.7B/snapshots/*/ | head -1); Q15=${Q15%/}

# up <tag> <model> <tp> <gpus> <port> <disabled> [method] [frac] [ma] [ks] [mp] [vp]
up() {
  local tag=$1 model=$2 tp=$3 gpus=$4 port=$5 dis=$6
  local method=${7:-} frac=${8:-} ma=${9:-} ks=${10:-} mp=${11:-} vp=${12:-}
  local -a e=(CUDA_VISIBLE_DEVICES="$gpus")
  if [[ -n "$method" ]]; then
    set -a; source "$AR/.env"; set +a
    e+=(PYTHONPATH="$AR" EXPERT_SKIP_MODE=dynamic
        EXPERT_SKIP_DISABLED_LAYERS="$dis"
        EXPERT_SKIP_ONLINE_THRESHOLD_METHOD="$method"
        EXPERT_SKIP_ONLINE_CONFIDENCE_KEEP_SHARE="$ks"
        EXPERT_SKIP_ONLINE_CONTRIB_MASS_FRACTION="$frac"
        EXPERT_SKIP_ONLINE_MIN_ACTIVE_FRAC="$ma"
        EXPERT_SKIP_ONLINE_MEAN_ACT_PERCENTILE="$mp"
        EXPERT_SKIP_ONLINE_VAR_ACT_PERCENTILE="$vp"
        EXPERT_SKIP_ONLINE_UNWEIGHTED_NORMS=1
        EXPERT_SKIP_ONLINE_REFRESH_EVERY_N_TOKENS=9999999
        EXPERT_SKIP_ONLINE_LOG=1 EXPERT_SKIP_ONLINE_LOG_FILE="$OUT/$tag.skip.log"
        EXPERT_SKIP_ONLINE_DRIFT_CHECK_EVERY_N_STEPS=0
        EXPERT_SKIP_ONLINE_RISK_TRIGGER=0)
  fi
  env "${e[@]}" "$VENV/bin/vllm" serve "$model" --served-model-name m \
    --trust-remote-code --tensor-parallel-size "$tp" --max-model-len 8192 \
    --gpu-memory-utilization 0.90 --port "$port" --no-enable-prefix-caching \
    > "$OUT/$tag.server.log" 2>&1 &
  echo $! > "$OUT/$tag.pid"
  local w=0 pid; pid=$(cat "$OUT/$tag.pid")
  until curl -sf "http://127.0.0.1:$port/health" >/dev/null 2>&1; do
    kill -0 "$pid" 2>/dev/null || { echo "  [$tag] DIED"; grep -oE "(RuntimeError|CUDA error)[:.].{0,90}" "$OUT/$tag.server.log"|sort -u|head -2; return 1; }
    (( w > 1800 )) && { echo "  [$tag] TIMEOUT"; kill -9 "$pid"; return 1; }
    sleep 10; w=$((w+10))
  done; echo "  [$tag] up ${w}s"
}
down() { local p; p=$(cat "$OUT/$1.pid" 2>/dev/null) || return 0; kill "$p" 2>/dev/null; wait "$p" 2>/dev/null; }

spd() {  # spd <tag> <port> <model>
  "$VENV/bin/vllm" bench serve --model m --tokenizer "$3" --trust-remote-code \
    --dataset-name random --random-input-len 128 --random-output-len 1024 \
    --num-prompts 64 --port "$2" > "$OUT/$1.warm.log" 2>&1
  "$VENV/bin/vllm" bench serve --model m --tokenizer "$3" --trust-remote-code \
    --dataset-name random --random-input-len 128 --random-output-len 1024 \
    --num-prompts 200 --port "$2" --percentile-metrics ttft,tpot \
    --save-result --result-dir "$OUT" --result-filename "$1.json" > "$OUT/$1.bench.log" 2>&1
  printf '  [%s] %s\n' "$1" "$(grep -E 'Output token throughput' "$OUT/$1.bench.log"|tr -s ' ')"
}
acc() {  # acc <tag> <port> <bench> <model>
  local tag=$1 port=$2 bench=$3 model=$4
  local d="$OUT/$tag/$bench"
  find "$d" -name eval_results.json -size +0 2>/dev/null | grep -q . && return 0
  mkdir -p "$d"
  PYTHONPATH="$AR" HF_ALLOW_CODE_EVAL=1 "$VENV/bin/python" \
    "$AR/scripts/serve/run_eval_serve.py" --task "$bench" --model_name "$model" \
    --local_model_path m --base_url "http://127.0.0.1:$port/v1" \
    --max_concurrency 128 --log_samples --output_dir "$d/run" > "$d/client.log" 2>&1
  echo "  [$tag/$bench] $(grep -ohE '"(pass@1,create_test|exact_match,flexible-extract)": [0-9.]+' "$d"/run/*/eval_results.json 2>/dev/null|head -1)"
}
mask() { [[ -s "$OUT/$1.skip.log" ]] && "$VENV/bin/python" - "$OUT/$1.skip.log" <<'PY'
import re,sys,statistics as st
t=open(sys.argv[1],errors="ignore").read()
r=re.findall(r"layer=(\d+) pruned=(\d+)/(\d+)",t)
if r:
    last={int(a):(int(b),int(c)) for a,b,c in r}; v=list(last.values())
    print(f"    mask: {st.mean(x[0] for x in v):.1f}/{v[0][1]} pruned/layer over {len(last)} layers")
else: print("    mask: NO FINALIZE LINES (method may not have produced a mask)")
PY
}

echo "=== A. PERCENTILE ABLATION (Qwen3) $(date '+%F %T') ==="
echo "    target to match: contrib_mass frac0.2 = 70.4/128 pruned, 1.623x, GSM8K 90.07, HEval 29.27"
for cfg in "pct_p50p90 50 90" "pct_p90p99 90 99"; do
  set -- $cfg; tag=$1 mp=$2 vp=$3
  [[ -f "$OUT/$tag.json" ]] && { echo "[skip] $tag"; continue; }
  bash "$AR/scripts/drain_gpus.sh" >/dev/null 2>&1
  echo "--- $tag (percentile mean<p$mp AND var<p$vp, keep_share=0) ---"
  up "base_$tag" "$Q3" 2 0,1 8800 0,47 && up "$tag" "$Q3" 2 2,3 8801 0,47 percentile 0.2 0.25 0 "$mp" "$vp" && {
    spd "base_$tag" 8800 "$Q3" & p1=$!; spd "$tag" 8801 "$Q3" & p2=$!; wait $p1 $p2
    mask "$tag"
    acc "$tag" 8801 gsm8k "$Q3"; acc "$tag" 8801 humaneval_base "$Q3"; }
  down "base_$tag"; down "$tag"; sleep 5
done

echo ""
echo "=== B. QWEN1.5 at keep_share=0 (TP=1, 4 cells at once) $(date '+%F %T') ==="
echo "    reference: Qwen1.5 with rescue ON was 0.96x (a net loss)"
bash "$AR/scripts/drain_gpus.sh" >/dev/null 2>&1
up q15_base "$Q15" 1 0 8810 0,23 &
up q15_keep0_f02 "$Q15" 1 1 8811 0,23 contrib_mass 0.2 0.25 0 50 90 &
up q15_keep015_f02 "$Q15" 1 2 8812 0,23 contrib_mass 0.2 0.25 0.15 50 90 &
up q15_keep0_f035 "$Q15" 1 3 8813 0,23 contrib_mass 0.35 0.25 0 50 90 &
wait
spd q15_base 8810 "$Q15" & a=$!
spd q15_keep0_f02 8811 "$Q15" & b=$!
spd q15_keep015_f02 8812 "$Q15" & c=$!
spd q15_keep0_f035 8813 "$Q15" & d=$!
wait $a $b $c $d
for t in q15_keep0_f02 q15_keep015_f02 q15_keep0_f035; do mask "$t"; done
# Accuracy on the two configs that matter, sequentially so the servers are quiet.
acc q15_base 8810 gsm8k "$Q15"; acc q15_base 8810 humaneval_base "$Q15"
acc q15_keep0_f02 8811 gsm8k "$Q15"; acc q15_keep0_f02 8811 humaneval_base "$Q15"
for t in q15_base q15_keep0_f02 q15_keep015_f02 q15_keep0_f035; do down "$t"; done
echo "=== DONE $(date '+%F %T') ==="
