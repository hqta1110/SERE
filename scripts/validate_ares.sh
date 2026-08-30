#!/usr/bin/env bash
# ADVERSARIAL VALIDATION of the ARES benchmark numbers.
#
# This does not extend the results. It tries to BREAK them. Four specific ways
# the measured speedups and accuracy deltas could be artifacts rather than the
# method, ordered most-damaging first. Every one of these is currently untested,
# and each has a concrete pass criterion.
#
# ---------------------------------------------------------------------------
# TEST 1: ENVIRONMENT ASYMMETRY  (the one that would invalidate everything)
#
# ARES activates only when PYTHONPATH points at the repo, because that is how
# sitecustomize.py loads. But sitecustomize does MORE than install skip patches:
# configure_skip_environment() also performs a "single-node NCCL repair"
# (NCCL_IB_DISABLE, stripping gib from LD_LIBRARY_PATH). Baseline arms never set
# PYTHONPATH, so they never get that repair.
#
# If the NCCL repair changes TP=2 collective performance AT ALL, then every
# ARES-vs-baseline ratio ever measured here is contaminated by an environment
# difference rather than by expert skipping.
#
#   Cell `env_off`:  PYTHONPATH SET, EXPERT_SKIP_MODE=off
#   PASS: env_off throughput == plain baseline within the 0.34-2.1% noise band.
#   FAIL: a systematic gap -> every published ratio must be recomputed against
#         env_off instead of plain baseline.
#
# ---------------------------------------------------------------------------
# TEST 2: CONTROLLER OVERHEAD / NULL-MASK PLACEBO
#
# Every ARES cell ever run had a non-empty mask, so the controller's own cost
# (profiling hooks, apply kernel, risk accumulation, router wrapper) has never
# been separated from the benefit of skipping. MIN_ACTIVE_FRAC=1.0 forces the
# prune limit to zero: the full controller runs, masks nothing.
#
#   PASS: null_mask throughput ~= baseline (overhead is negligible), AND
#         null_mask GSM8K == baseline GSM8K essentially exactly.
#   FAIL (speed): reported speedups are biased -- if null_mask is 0.97x then the
#         true benefit of skipping is larger than published; if it is >1.00x
#         something is wrong, because identical work cannot be faster.
#   FAIL (accuracy): if null_mask GSM8K differs from baseline by more than
#         run-to-run noise, the controller perturbs outputs even with NO mask,
#         which is a correctness bug and would taint every accuracy delta.
#
# ---------------------------------------------------------------------------
# TEST 3: RUN-TO-RUN NOISE FLOOR ON *ACCURACY*
#
# We have exactly ONE GSM8K measurement per config and are reporting deltas as
# small as -3.33 points. Greedy decoding is nominally deterministic, but vLLM
# batches nondeterministically: batch composition changes reduction order,
# which flips borderline tokens. If two identical frac 0.10 runs differ by
# +-2 points, then -3.33 is not a finding.
#
#   Two INDEPENDENT frac 0.10 servers, same config, measured separately.
#   PASS: |rep1 - rep2| well under the deltas being claimed (<1 point).
#   FAIL: the frontier's fine structure (0.05 vs 0.10) is noise.
#
# ---------------------------------------------------------------------------
# TEST 4: IN-DOMAIN PROFILING CONFOUND  (the reviewer's first question)
#
# ARES builds its mask by profiling live traffic, and we warm on GSM8K *train*
# then score GSM8K *test*. A reviewer will immediately say: "you profiled on
# in-domain data, so of course the mask suits the eval." If the mask only works
# when profiled in-domain, ARES is far weaker than claimed -- deployments do not
# get a matching calibration stream.
#
#   Cell `xdomain`: frac 0.10 profiled on MBPP (CODE), then scored on GSM8K.
#   PASS: close to the in-domain 86.13 -> the mask captures general expert
#         importance, which is a genuinely strong result worth its own table.
#   FAIL: a large drop -> ARES's quality depends on in-domain profiling, and
#         that limitation belongs in the paper.
#
# ---------------------------------------------------------------------------
# Reference values being tested against (Qwen3-30B, TP=2, saturated, warmed):
#   baseline    4030-4116 tok/s   GSM8K strict 89.46 / 89.69
#   frac 0.10   1.465x            GSM8K strict 86.13
# No bare `wait` anywhere.
set -uo pipefail

exec 9>/tmp/ares_validate.lock
flock -n 9 || { echo "[val] locked"; exit 0; }

VENV=/home/PC/new-efficient-moe/.venv
AR=/home/PC/new-efficient-moe
SV=/home/PC/SERE_v1
OUT=$SV/results/validate
LOGS=$OUT/logs
mkdir -p "$OUT" "$LOGS"
HUB=/home/PC/.cache/huggingface/hub
Q3=$(ls -d "$HUB"/models--Qwen--Qwen3-30B-A3B/snapshots/*/ | head -1); Q3=${Q3%/}

PYTHONPATH="$AR" "$VENV/bin/python" -c "
import patches.vllm.fused_skip_ops as m
assert m.fused_cuda_available() and m.fused_confidence_available() and m.fused_risk_accum_available()
print('[val] ARES kernels: OK')" || exit 1

# serve <tag> <gpus> <port> <kind> [frac]
#   kind: plain | envoff | nullmask | ares
serve() {
  local tag=$1 gpus=$2 port=$3 kind=$4 frac=${5:-0.10}
  local -a e=(CUDA_VISIBLE_DEVICES="$gpus")
  case "$kind" in
    plain) : ;;                                  # no PYTHONPATH at all
    envoff)                                      # TEST 1
      e+=(PYTHONPATH="$AR" EXPERT_SKIP_MODE=off) ;;
    nullmask)                                    # TEST 2
      set -a; source "$AR/.env"; set +a
      e+=(PYTHONPATH="$AR" EXPERT_SKIP_MODE=dynamic
          EXPERT_SKIP_DISABLED_LAYERS=0,47
          EXPERT_SKIP_ONLINE_THRESHOLD_METHOD=contrib_mass
          EXPERT_SKIP_ONLINE_CONFIDENCE_KEEP_SHARE=0
          EXPERT_SKIP_ONLINE_CONTRIB_MASS_FRACTION=0.0
          EXPERT_SKIP_ONLINE_MIN_ACTIVE_FRAC=1.0
          EXPERT_SKIP_ONLINE_UNWEIGHTED_NORMS=1
          EXPERT_SKIP_ONLINE_REFRESH_EVERY_N_TOKENS=9999999
          EXPERT_SKIP_ONLINE_LOG=1 EXPERT_SKIP_ONLINE_LOG_FILE="$LOGS/$tag.skip.log"
          EXPERT_SKIP_ONLINE_DRIFT_CHECK_EVERY_N_STEPS=0
          EXPERT_SKIP_ONLINE_RISK_TRIGGER=0) ;;
    ares)
      set -a; source "$AR/.env"; set +a
      e+=(PYTHONPATH="$AR" EXPERT_SKIP_MODE=dynamic
          EXPERT_SKIP_DISABLED_LAYERS=0,47
          EXPERT_SKIP_ONLINE_THRESHOLD_METHOD=contrib_mass
          EXPERT_SKIP_ONLINE_CONFIDENCE_KEEP_SHARE=0
          EXPERT_SKIP_ONLINE_CONTRIB_MASS_FRACTION="$frac"
          EXPERT_SKIP_ONLINE_MIN_ACTIVE_FRAC=0.25
          EXPERT_SKIP_ONLINE_UNWEIGHTED_NORMS=1
          EXPERT_SKIP_ONLINE_REFRESH_EVERY_N_TOKENS=9999999
          EXPERT_SKIP_ONLINE_LOG=1 EXPERT_SKIP_ONLINE_LOG_FILE="$LOGS/$tag.skip.log"
          EXPERT_SKIP_ONLINE_DRIFT_CHECK_EVERY_N_STEPS=0
          EXPERT_SKIP_ONLINE_RISK_TRIGGER=0) ;;
  esac
  env "${e[@]}" "$VENV/bin/vllm" serve "$Q3" --served-model-name bench \
    --trust-remote-code --tensor-parallel-size 2 --max-model-len 8192 \
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

# measure <tag> <port> <warm_source> <has_skiplog>
# Warm first (installs+confirms any mask), then speed, then GSM8K -- so the
# mask state is identical for both measurements.
measure() {
  local tag=$1 port=$2 wsrc=$3 hs=$4
  local d="$OUT/$tag"; mkdir -p "$d"
  local -a warg=(--port "$port" --model bench --warm-source "$wsrc")
  [[ "$hs" == 1 ]] && warg+=(--skip-log "$LOGS/$tag.skip.log")
  "$VENV/bin/python" "$SV/scripts/warm_until_mask.py" "${warg[@]}" > "$d/warm.log" 2>&1
  echo "    [$tag] warm rc=$? (source=$wsrc)"

  if [[ ! -f "$OUT/$tag.json" ]]; then
    "$VENV/bin/vllm" bench serve --model bench --tokenizer "$Q3" --trust-remote-code \
      --dataset-name random --random-input-len 128 --random-output-len 1024 \
      --num-prompts 64 --port "$port" > "$LOGS/$tag.warmup.log" 2>&1
    "$VENV/bin/vllm" bench serve --model bench --tokenizer "$Q3" --trust-remote-code \
      --dataset-name random --random-input-len 128 --random-output-len 1024 \
      --num-prompts 200 --port "$port" --percentile-metrics ttft,tpot \
      --save-result --result-dir "$OUT" --result-filename "$tag.json" \
      > "$LOGS/$tag.bench.log" 2>&1
    printf '    [%s] %s\n' "$tag" \
      "$(grep -E 'Output token throughput' "$LOGS/$tag.bench.log" | tr -s ' ')"
  fi

  if ! find "$d/run" -name eval_results.json -size +0 2>/dev/null | grep -q .; then
    PYTHONPATH="$AR" "$VENV/bin/python" "$AR/scripts/serve/run_eval_serve.py" \
      --task gsm8k --model_name "$Q3" --local_model_path bench \
      --base_url "http://127.0.0.1:$port/v1" --max_concurrency 128 \
      --log_samples --output_dir "$d/run" > "$d/client.log" 2>&1
    local f; f=$(find "$d/run" -name eval_results.json -size +0 | head -1)
    [[ -n "$f" ]] && "$VENV/bin/python" -c "
import json; r=json.load(open('$f'))['results']['gsm8k']
print(f\"    [$tag] gsm8k strict={r['exact_match,strict-match']:.4f}\")"
  fi
  if [[ "$hs" == 1 && -s "$LOGS/$tag.skip.log" ]]; then
    "$VENV/bin/python" - "$LOGS/$tag.skip.log" "$tag" <<'PY'
import re,sys,statistics as st
t=open(sys.argv[1],errors='ignore').read()
r=re.findall(r"layer=(\d+) pruned=(\d+)/(\d+)",t); last={}
for a,b,c in r: last[int(a)]=(int(b),int(c))
if last:
    v=list(last.values()); E=v[0][1]; p=st.mean(x[0] for x in v)
    print(f"    [{sys.argv[2]}] pruned {p:.1f}/{E} ({p/E*100:.0f}%)")
else:
    print(f"    [{sys.argv[2]}] mask empty (expected for nullmask)")
PY
  fi
}

# pair <tagA> <kindA> <wsrcA> <hsA> <tagB> <kindB> <wsrcB> <hsB> [fracB]
pair() {
  local ta=$1 ka=$2 wa=$3 ha=$4 tb=$5 kb=$6 wb=$7 hb=$8 fb=${9:-0.10}
  if [[ -f "$OUT/$ta.json" && -f "$OUT/$tb.json" ]] \
     && find "$OUT/$ta/run" -name eval_results.json -size +0 2>/dev/null | grep -q . \
     && find "$OUT/$tb/run" -name eval_results.json -size +0 2>/dev/null | grep -q .; then
    echo "  [$ta | $tb] cached"; return 0
  fi
  bash "$AR/scripts/drain_gpus.sh" >/dev/null 2>&1
  echo "--- $ta ($ka) | $tb ($kb)  $(date '+%T')"
  serve "$ta" 0,1 8850 "$ka"
  serve "$tb" 2,3 8851 "$kb" "$fb"
  local oa=0 ob=0 pa="" pb=""
  wait_up "$ta" 8850 && oa=1
  wait_up "$tb" 8851 && ob=1
  (( oa )) && { measure "$ta" 8850 "$wa" "$ha" & pa=$!; }
  (( ob )) && { measure "$tb" 8851 "$wb" "$hb" & pb=$!; }
  [[ -n "$pa" ]] && wait "$pa"
  [[ -n "$pb" ]] && wait "$pb"
  stop "$ta"; stop "$tb"; sleep 5
}

echo "=== ADVERSARIAL VALIDATION OF ARES $(date '+%F %T') ==="
# TEST 1 + reference baseline
pair base1    plain    gsm8k_train 0   env_off   envoff   gsm8k_train 0
# TEST 2 (paired with a second independent baseline for the noise floor)
pair base2    plain    gsm8k_train 0   nullmask  nullmask gsm8k_train 1
# TEST 3: two identical frac 0.10 runs
pair rep1     ares     gsm8k_train 1   rep2      ares     gsm8k_train 1
# TEST 4: mask profiled on CODE, scored on MATH
pair base3    plain    gsm8k_train 0   xdomain   ares     mbpp        1

echo "=== VERDICT $(date '+%F %T') ==="
"$VENV/bin/python" - "$OUT" <<'PY'
import json, glob, os, statistics as st, sys
OUT = sys.argv[1]
def thr(t):
    p = os.path.join(OUT, f"{t}.json")
    if not os.path.exists(p): return None
    d = json.load(open(p))
    if d.get("completed", 0) != d.get("num_prompts", 0):
        print(f"  !! {t}: completed != num_prompts (INVALID)"); return None
    return d.get("output_throughput")
def acc(t):
    g = glob.glob(os.path.join(OUT, t, "run", "**", "eval_results.json"), recursive=True)
    if not g: return None
    return json.load(open(g[0]))["results"]["gsm8k"]["exact_match,strict-match"] * 100

bases = [(t, thr(t)) for t in ("base1", "base2", "base3")]
bases = [(t, v) for t, v in bases if v]
bacc = [(t, acc(t)) for t in ("base1", "base2", "base3")]
bacc = [(t, v) for t, v in bacc if v is not None]
print("  --- reference baselines (independent replicates) ---")
if bases:
    v = [x for _, x in bases]
    print("   speed: " + ", ".join(f"{t}={x:.1f}" for t, x in bases)
          + f"  spread {max(v)/min(v)-1:+.2%}")
if bacc:
    v = [x for _, x in bacc]
    print("   gsm8k: " + ", ".join(f"{t}={x:.2f}" for t, x in bacc)
          + f"  range {max(v)-min(v):.2f} pts   <-- ACCURACY NOISE FLOOR")
ref = st.mean([x for _, x in bases]) if bases else None
racc = st.mean([x for _, x in bacc]) if bacc else None

def line(tag, label):
    t, a = thr(tag), acc(tag)
    ts = f"{t:.1f} ({t/ref:.3f}x)" if (t and ref) else "--"
    as_ = f"{a:.2f} ({a-racc:+.2f})" if (a is not None and racc) else "--"
    print(f"  {label:<34}{ts:>22}{as_:>20}")

print(f"  {'test':<34}{'throughput (vs base)':>22}{'gsm8k (delta)':>20}")
line("env_off",  "1. PYTHONPATH set, mode=off")
line("nullmask", "2. full controller, NO mask")
line("rep1",     "3. frac 0.10 replicate 1")
line("rep2",     "3. frac 0.10 replicate 2")
line("xdomain",  "4. frac 0.10 profiled on CODE")

print("\n  --- pass/fail ---")
eo, nm = thr("env_off"), thr("nullmask")
if eo and ref:
    d = eo/ref - 1
    print(f"  TEST 1 environment asymmetry: {d:+.2%} "
          + ("PASS (within noise)" if abs(d) < 0.021 else
             "*** FAIL -- ratios are contaminated by the environment ***"))
if nm and ref:
    d = nm/ref - 1
    print(f"  TEST 2 controller overhead:   {d:+.2%} "
          + ("PASS" if abs(d) < 0.021 else
             ("*** FAIL: >1x with identical work -- impossible, investigate ***"
              if d > 0 else "note: speedups are UNDERSTATED by this much")))
na = acc("nullmask")
if na is not None and racc:
    d = na - racc
    print(f"  TEST 2b null-mask accuracy:   {d:+.2f} pts "
          + ("PASS (controller does not perturb outputs)" if abs(d) < 1.0 else
             "*** FAIL -- controller changes outputs with NO mask ***"))
r1, r2 = acc("rep1"), acc("rep2")
if r1 is not None and r2 is not None:
    d = abs(r1 - r2)
    print(f"  TEST 3 accuracy noise floor:  {d:.2f} pts "
          + ("PASS (frontier structure is real)" if d < 1.0 else
             f"*** FAIL -- deltas below {d:.2f} pts are not findings ***"))
xd = acc("xdomain")
if xd is not None:
    print(f"  TEST 4 cross-domain mask:     {xd:.2f} vs 86.13 in-domain "
          + ("-> generalizes" if xd > 84.0 else "-> depends on in-domain profiling"))
PY
echo "=== DONE $(date '+%F %T') ==="
