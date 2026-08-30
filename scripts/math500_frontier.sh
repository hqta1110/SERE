#!/usr/bin/env bash
# A SECOND quality benchmark for the Qwen3-30B frontier. This plugs the largest
# remaining hole in the paper.
#
# Right now the entire quality axis rests on ONE metric: GSM8K strict-match.
# That is thin for ICML/ICLR, and it is not for lack of trying -- HumanEval-base
# pass@1,create_test was measured and proved unusable at these effect sizes
# (a config that lost 5.00 GSM8K points scored +10.4 ABOVE baseline; a paired
# analysis on both-short completions gave 76.7% vs 76.7%, identical). So the
# frontier needs a second INDEPENDENT task, and MATH500 is the right one:
# harder, still exact-match scorable, and a different reasoning distribution
# from grade-school arithmetic.
#
# Cells chosen to match the GSM8K frontier exactly so the two tables are
# directly comparable -- baseline plus the three usable budgets:
#
#   arm            speed     GSM8K strict     MATH500
#   baseline       1.000     89.46-89.69      <- this run
#   frac 0.05      1.259x    89.61 (+0.15)    <- this run
#   frac 0.10      1.465x    86.13 (-3.33)    <- this run
#   frac 0.15      1.513x    84.46 (-5.00)    <- this run
#
# If MATH500 reproduces the ordering (0.05 ~ free, a knee by 0.15), the frontier
# claim is properly supported. If it does NOT -- e.g. if frac 0.05 is already
# expensive on harder reasoning -- that is a MORE important finding than any
# speed number here, because it would mean easy benchmarks hide the real cost
# of expert skipping.
#
# max_gen_toks is 1024 for math500 (the script's per-task default), which the
# Qwen3.6 disaster showed matters enormously: too small a cap grades BREVITY
# instead of correctness. 1024 is applied to every arm.
#
# Warm-before-score is mandatory and enforced. No bare `wait`.
set -uo pipefail

exec 9>/tmp/ares_math500.lock
flock -n 9 || { echo "[m500] locked"; exit 0; }

VENV=/home/PC/new-efficient-moe/.venv
AR=/home/PC/new-efficient-moe
SV=/home/PC/SERE_v1
OUT=$SV/results/math500
mkdir -p "$OUT"
HUB=/home/PC/.cache/huggingface/hub
Q3=$(ls -d "$HUB"/models--Qwen--Qwen3-30B-A3B/snapshots/*/ | head -1); Q3=${Q3%/}

PYTHONPATH="$AR" "$VENV/bin/python" -c "
import patches.vllm.fused_skip_ops as m
assert m.fused_cuda_available() and m.fused_confidence_available() and m.fused_risk_accum_available()
print('[m500] ARES kernels: OK')" || exit 1

# Dependency preflight: minerva_math500 needs sympy / math_verify / antlr4.
# A missing one fails the eval AFTER a full server boot, so check it up front.
"$VENV/bin/python" -c "
import importlib, sys
missing = [m for m in ('sympy', 'math_verify', 'antlr4') if not importlib.util.find_spec(m)]
if missing:
    print('[m500] MISSING DEPS:', missing); sys.exit(1)
print('[m500] math deps: OK')" || exit 1

# cell <tag> <gpus> <port> <base|FRAC>
cell() {
  local tag=$1 gpus=$2 port=$3 frac=$4
  local d="$OUT/$tag"
  if find "$d" -name eval_results.json -size +0 2>/dev/null | grep -q .; then
    echo "  [$tag] cached"; return 0
  fi
  mkdir -p "$d"
  local skiplog=""
  local -a e=(CUDA_VISIBLE_DEVICES="$gpus")
  if [[ "$frac" != base ]]; then
    skiplog="$d/skip.log"
    set -a; source "$AR/.env"; set +a
    e+=(PYTHONPATH="$AR" EXPERT_SKIP_MODE=dynamic
        EXPERT_SKIP_DISABLED_LAYERS=0,47
        EXPERT_SKIP_ONLINE_THRESHOLD_METHOD=contrib_mass
        EXPERT_SKIP_ONLINE_CONFIDENCE_KEEP_SHARE=0
        EXPERT_SKIP_ONLINE_CONTRIB_MASS_FRACTION="$frac"
        EXPERT_SKIP_ONLINE_MIN_ACTIVE_FRAC=0.25
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
    kill -0 $pid 2>/dev/null || { echo "  [$tag] SERVER DIED"; return 1; }
    (( w > 2400 )) && { echo "  [$tag] TIMEOUT"; kill -9 $pid; return 1; }
    sleep 10; w=$((w+10))
  done
  echo "  [$tag] up ${w}s"

  # Warm on GSM8K train: held out from MATH500 by construction (different
  # dataset entirely), and long enough to move the prefill budget.
  local -a warg=(--port "$port" --model m --warm-source gsm8k_train)
  [[ -n "$skiplog" ]] && warg+=(--skip-log "$skiplog")
  "$VENV/bin/python" "$SV/scripts/warm_until_mask.py" "${warg[@]}" > "$d/warm.log" 2>&1
  local wrc=$?
  echo "  [$tag] warm rc=$wrc"
  [[ "$frac" != base && $wrc -ne 0 ]] && \
    echo "  [$tag] WARNING: mask unconfirmed -- score INVALID"

  PYTHONPATH="$AR" "$VENV/bin/python" "$AR/scripts/serve/run_eval_serve.py" \
    --task math500 --model_name "$Q3" --local_model_path m \
    --base_url "http://127.0.0.1:$port/v1" --max_concurrency 128 \
    --log_samples --output_dir "$d/run" > "$d/client.log" 2>&1

  local f; f=$(find "$d" -name eval_results.json -size +0 | head -1)
  if [[ -n "$f" ]]; then
    "$VENV/bin/python" -c "
import json
d = json.load(open('$f'))['results']
for task, m in d.items():
    for k, v in m.items():
        if 'exact_match' in k or 'acc' in k:
            if isinstance(v, float): print(f'  [$tag] {task} {k} = {v:.4f}')
" 2>/dev/null | head -4
  else
    echo "  [$tag] NO RESULT -- see $d/client.log"
    grep -oE '(ModuleNotFoundError|ImportError|KeyError|ValueError).{0,120}' "$d/client.log" 2>/dev/null | sort -u | head -2
  fi
  if [[ "$frac" != base && -s "$skiplog" ]]; then
    "$VENV/bin/python" - "$skiplog" "$tag" <<'PY'
import re,sys,statistics as st
t=open(sys.argv[1],errors='ignore').read()
r=re.findall(r"layer=(\d+) pruned=(\d+)/(\d+)",t); last={}
for a,b,c in r: last[int(a)]=(int(b),int(c))
if last:
    v=list(last.values()); E=v[0][1]; p=st.mean(x[0] for x in v)
    print(f"    [{sys.argv[2]}] pruned {p:.1f}/{E} ({p/E*100:.0f}%)")
PY
  fi
  kill $pid 2>/dev/null; wait $pid 2>/dev/null; sleep 5
}

echo "=== MATH500 FRONTIER (Qwen3-30B, warmed) $(date '+%F %T') ==="
echo "    second quality benchmark; GSM8K strict was the only one until now"
bash "$AR/scripts/drain_gpus.sh" >/dev/null 2>&1
cell m_base 0,1 8800 base  & p1=$!
cell m_f005 2,3 8801 0.05  & p2=$!
wait $p1; wait $p2
bash "$AR/scripts/drain_gpus.sh" >/dev/null 2>&1
cell m_f010 0,1 8800 0.10  & p3=$!
cell m_f015 2,3 8801 0.15  & p4=$!
wait $p3; wait $p4

echo "=== SUMMARY $(date '+%F %T') ==="
"$VENV/bin/python" - "$OUT" <<'PY'
import json, glob, os, sys
OUT = sys.argv[1]
def score(tag):
    g = glob.glob(os.path.join(OUT, tag, "**", "eval_results.json"), recursive=True)
    if not g: return None
    res = json.load(open(g[0]))["results"]
    for task, m in res.items():
        for k, v in m.items():
            if ("exact_match" in k or k.startswith("acc")) and isinstance(v, float) \
               and "stderr" not in k:
                return v * 100
    return None
b = score("m_base")
print(f"  {'arm':<12}{'MATH500':>10}{'delta':>9}   (GSM8K strict for reference)")
ref = {"m_base": "89.46-89.69", "m_f005": "89.61 (+0.15)",
       "m_f010": "86.13 (-3.33)", "m_f015": "84.46 (-5.00)"}
for tag, label in (("m_base","baseline"), ("m_f005","frac 0.05"),
                   ("m_f010","frac 0.10"), ("m_f015","frac 0.15")):
    s = score(tag)
    d = f"{s-b:+.2f}" if (s is not None and b) else "--"
    print(f"  {label:<12}{'--' if s is None else f'{s:.2f}':>10}{d:>9}   {ref[tag]}")
PY
echo "=== DONE $(date '+%F %T') ==="
