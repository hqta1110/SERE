#!/usr/bin/env bash
# ARES on Qwen/Qwen3.6-35B-A3B -- third model, and the first test of whether the
# CONTRIB_MASS_FRACTION frontier generalizes across EXPERT GRANULARITY.
#
# Why this model is the interesting one:
#   Qwen3-30B-A3B : 128 experts x inter 768,  top_k 8, no shared expert
#   Qwen1.5-MoE   :  60 experts x inter 1408, top_k 4, shared expert
#   Qwen3.6-35B   : 256 experts x inter 512,  top_k 8, shared expert   <- here
# Experts here are ~2/3 the width of Qwen3-30B's and twice as numerous, so each
# carries far less of the layer's output mass. If the mass budget is the right
# abstraction, the SAME frac should be safe here while pruning many more
# experts. If frac 0.05-0.15 was really a Qwen3-30B constant, this run exposes
# that. Falsifiable either way, which is why it is worth the GPU hours even
# though ARES lost on Qwen1.5.
#
# Confounds to state in any writeup:
#   * SHARED EXPERT always runs and ARES cannot mask it -> hard floor on
#     speedup, and a quality cushion. Qwen3-30B had none, so speedup
#     MAGNITUDES are not comparable across the two models (frontier SHAPE is).
#   * HYBRID ATTENTION (3x GatedDeltaNet : 1x full over 40 layers) makes
#     attention cheaper, so MoE is a larger share of FLOPs -> more headroom.
#
# ===================== TWO DELIBERATE PROTOCOL DEVIATIONS =====================
# 1. TP=4, so baselines are SEQUENTIAL not concurrent. 67 GB of bf16 weights
#    needs 4x A100-40GB (18 GB/GPU, ~18 GB left for KV + mamba state). Every
#    earlier model ran treatment beside a CONTEMPORANEOUS baseline on separate
#    GPU pairs; only one server fits here. Mitigation: an optional second
#    baseline row runs last, and the summary prints the baseline spread so a
#    reader can judge drift directly (prior measured drift: 0.34% within a
#    round, 2.11% across a day).
# 2. ONE SERVER PER FRAC serves BOTH the speed bench and the accuracy eval, at
#    --max-model-len 8192 with prefix caching off for both. Earlier models used
#    4096 for speed and 8192 for accuracy across separate servers. Rationale:
#    server boot dominates cost at 67 GB/TP=4, and this cuts boots from 12 to 5.
#    All Qwen3.6 arms share the setting, so within-model ratios stay valid.
#    Bonus: it is actually BETTER methodology -- warm_until_mask runs FIRST, so
#    the mask is confirmed installed before the speed bench too, where earlier
#    models let the speed warmup pass do double duty as mask construction.
#
# Cells are ordered by information value, and each row is written to disk as it
# completes, so an interrupted run still yields complete usable rows rather than
# all-speed-and-no-accuracy. base and f010/f020 come first because the whole
# question is speed-vs-quality at a usable budget.
set -uo pipefail

exec 9>/tmp/q36_sweep.lock
flock -n 9 || { echo "[q36] locked; another instance is running"; exit 0; }

VENV=/home/PC/new-efficient-moe/.venv
AR=/home/PC/new-efficient-moe
SV=/home/PC/SERE_v1
OUT=$SV/results/q36
LOGS=$OUT/logs
mkdir -p "$OUT" "$LOGS"
HUB=/home/PC/.cache/huggingface/hub
Q36=$(ls -d "$HUB"/models--Qwen--Qwen3.6-35B-A3B/snapshots/*/ 2>/dev/null | head -1); Q36=${Q36%/}
[[ -n "$Q36" && -f "$Q36/config.json" ]] || { echo "[q36] model missing"; exit 1; }
PORT=8500

PYTHONPATH="$AR" "$VENV/bin/python" -c "
import patches.vllm.fused_skip_ops as m
assert m.fused_cuda_available() and m.fused_confidence_available() and m.fused_risk_accum_available()
print('[q36] ARES kernels: OK')" || exit 1

mask() {  # mask <tag> -- what got pruned and what it cost at runtime
  [[ -s "$LOGS/$1.skip.log" ]] || return 0
  "$VENV/bin/python" - "$LOGS/$1.skip.log" "$1" <<'PY'
import re, sys, statistics as st
t = open(sys.argv[1], errors="ignore").read()
rows = re.findall(r"layer=(\d+) pruned=(\d+)/(\d+)", t)
if not rows:
    print(f"    [{sys.argv[2]}] NO MASK LINES -- hook failed"); sys.exit(0)
last = {}
for a, b, c in rows: last[int(a)] = (int(b), int(c))
v = list(last.values()); E = v[0][1]; p = st.mean(x[0] for x in v)
risk = [float(x) for x in re.findall(r"risk=([0-9.]+)", t)]
print(f"    [{sys.argv[2]}] pruned {p:.1f}/{E} ({p/E*100:.0f}%) over {len(last)} layers"
      f" | routed slots surviving {8*(1-p/E):.2f}/8 (+shared, unmaskable)"
      + (f" | gate mass surrendered median={st.median(risk):.4f}" if risk else ""))
PY
}

# row <tag> <base|FRAC>  -- boot once, then warm -> speed -> accuracy -> teardown
row() {
  local tag=$1 frac=$2
  local d="$OUT/$tag/gsm8k"
  local have_spd=0 have_acc=0
  [[ -f "$OUT/$tag.json" ]] && have_spd=1
  find "$d" -name eval_results.json -size +0 2>/dev/null | grep -q . && have_acc=1
  (( have_spd && have_acc )) && { echo "  [$tag] cached (speed+accuracy)"; return 0; }
  mkdir -p "$d"

  echo "--- [$tag] frac=$frac  $(date '+%T')"
  bash "$AR/scripts/drain_gpus.sh" >/dev/null 2>&1
  local -a e=(CUDA_VISIBLE_DEVICES=0,1,2,3)
  local skiplog=""
  if [[ "$frac" != base ]]; then
    skiplog="$LOGS/$tag.skip.log"
    set -a; source "$AR/.env"; set +a
    e+=(PYTHONPATH="$AR" EXPERT_SKIP_MODE=dynamic
        EXPERT_SKIP_DISABLED_LAYERS=0,39
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
  env "${e[@]}" "$VENV/bin/vllm" serve "$Q36" --served-model-name m \
    --trust-remote-code --tensor-parallel-size 4 --max-model-len 8192 \
    --gpu-memory-utilization 0.90 --port "$PORT" --no-enable-prefix-caching \
    > "$LOGS/$tag.server.log" 2>&1 &
  local pid=$! w=0
  until curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; do
    kill -0 "$pid" 2>/dev/null || {
      echo "  [$tag] SERVER DIED:"
      grep -oE "(ValueError|RuntimeError|NotImplementedError|CUDA error|OutOfMemoryError)[:.].{0,160}" \
        "$LOGS/$tag.server.log" | sort -u | head -3
      return 1; }
    (( w > 2400 )) && { echo "  [$tag] TIMEOUT"; kill -9 "$pid"; return 1; }
    sleep 15; w=$((w+15))
  done
  echo "  [$tag] up ${w}s"

  # 1. Install + CONFIRM the mask before ANY measurement (speed included).
  local -a warg=(--port "$PORT" --model m --warm-source gsm8k_train)
  [[ -n "$skiplog" ]] && warg+=(--skip-log "$skiplog")
  "$VENV/bin/python" "$SV/scripts/warm_until_mask.py" "${warg[@]}" > "$d/warm.log" 2>&1
  local wrc=$?
  echo "  [$tag] warm rc=$wrc"
  if [[ "$frac" != base && $wrc -ne 0 ]]; then
    echo "  [$tag] WARNING: mask unconfirmed -- BOTH speed and accuracy are invalid"
  fi

  # 2. Speed: in=128 out=1024, 200 prompts, saturated. Same as every other model.
  if (( ! have_spd )); then
    "$VENV/bin/vllm" bench serve --model m --tokenizer "$Q36" --trust-remote-code \
      --dataset-name random --random-input-len 128 --random-output-len 1024 \
      --num-prompts 64 --port "$PORT" > "$LOGS/$tag.warmup.log" 2>&1
    "$VENV/bin/vllm" bench serve --model m --tokenizer "$Q36" --trust-remote-code \
      --dataset-name random --random-input-len 128 --random-output-len 1024 \
      --num-prompts 200 --port "$PORT" --percentile-metrics ttft,tpot \
      --save-result --result-dir "$OUT" --result-filename "$tag.json" \
      > "$LOGS/$tag.bench.log" 2>&1
    printf '  [%s] %s | %s\n' "$tag" \
      "$(grep -E 'Output token throughput' "$LOGS/$tag.bench.log" | tr -s ' ')" \
      "$(grep -E 'Mean TPOT' "$LOGS/$tag.bench.log" | tr -s ' ')"
  fi

  # 3. Accuracy: GSM8K strict-match at max_gen_toks=1024, NOT the 256 used for
  # Qwen3-30B/Qwen1.5. This deviation is mandatory on this model, and the first
  # attempt proved why:
  #
  #     max_gen_toks=256   ->   base 29.49   f010 (masked!) 62.17
  #
  # A masked model cannot be twice as capable as the model it was derived from.
  # Qwen3.6 is a reasoning-style model that emits a <think></think> preamble and
  # verbose prose, so at a 256-token cap the metric scores BREVITY, not
  # correctness: the baseline rambles past the cap and never reaches `#### N`,
  # while masking makes generations terser and therefore *more* likely to finish.
  # Both numbers measure the cap. Evidence preserved in
  # results/q36/{base,f010}/gsm8k_INVALID_maxtok256/.
  #
  # This is the same length/termination confound already documented for
  # HumanEval-base pass@1 on Qwen3-30B, except here it hits GSM8K because the
  # cap binds. 1024 is applied to EVERY arm, so within-model deltas stay valid;
  # absolute scores are not comparable to the 256-cap numbers on other models.
  if (( ! have_acc )); then
    PYTHONPATH="$AR" "$VENV/bin/python" "$AR/scripts/serve/run_eval_serve.py" \
      --task gsm8k --model_name "$Q36" --local_model_path m \
      --base_url "http://127.0.0.1:$PORT/v1" --max_concurrency 128 \
      --max_tokens 1024 \
      --log_samples --output_dir "$d/run" > "$d/client.log" 2>&1
    local f; f=$(find "$d" -name eval_results.json -size +0 | head -1)
    if [[ -n "$f" ]]; then
      "$VENV/bin/python" -c "
import json; r=json.load(open('$f'))['results']['gsm8k']
print(f\"  [$tag] gsm8k strict={r['exact_match,strict-match']:.4f} flexible={r['exact_match,flexible-extract']:.4f}\")"
    else
      echo "  [$tag] NO EVAL RESULT -- see $d/client.log"
    fi
  fi

  [[ "$frac" != base ]] && mask "$tag"
  kill "$pid" 2>/dev/null; wait "$pid" 2>/dev/null
  bash "$AR/scripts/drain_gpus.sh" >/dev/null 2>&1; sleep 5
}

summary() {
"$VENV/bin/python" - "$OUT" <<'PY'
import json, glob, os, statistics as st, sys
OUT = sys.argv[1]
def thr(tag):
    p = os.path.join(OUT, f"{tag}.json")
    if not os.path.exists(p): return None
    d = json.load(open(p))
    c, n = d.get("completed", 0), d.get("num_prompts", 0)
    if n and c != n: print(f"  !! {tag}: completed={c} != num_prompts={n} (INVALID)")
    return d.get("output_throughput")
def acc(tag):
    g = glob.glob(os.path.join(OUT, tag, "gsm8k", "**", "eval_results.json"), recursive=True)
    if not g: return None
    return json.load(open(g[0]))["results"]["gsm8k"]["exact_match,strict-match"] * 100
bt = [(t, thr(t)) for t in ("base", "base2")]
bt = [(t, v) for t, v in bt if v]
if not bt:
    print("  no baseline yet"); sys.exit(0)
vals = [v for _, v in bt]
ref = st.mean(vals)
print("  baseline throughput: " + ", ".join(f"{t}={v:.1f}" for t, v in bt)
      + (f"  spread {max(vals)/min(vals)-1:+.2%}" if len(vals) > 1 else "  (single replicate)"))
ba = acc("base") or 0.0
print(f"  baseline gsm8k strict: {ba:.2f}")
print(f"  {'frac':<8}{'tok/s':>10}{'xthr':>9}{'gsm8k':>9}{'delta':>9}")
for tag, frac in (("base","0 (ref)"), ("f005","0.05"), ("f010","0.10"),
                  ("f020","0.20"), ("f035","0.35")):
    v, s = thr(tag), acc(tag)
    print(f"  {frac:<8}{'--' if v is None else f'{v:.1f}':>10}"
          f"{'--' if v is None else f'{v/ref:.3f}x':>9}"
          f"{'--' if s is None else f'{s:.2f}':>9}"
          f"{'--' if (s is None or not ba) else f'{s-ba:+.2f}':>9}")
PY
}

echo "=== Qwen3.6-35B-A3B: ARES sweep (TP=4, one server per frac) $(date '+%F %T') ==="
echo "    model: $Q36"
# Priority order: baseline and the two most informative budgets first, so an
# interrupted run still has complete speed+accuracy rows to compare.
row base base
row f010 0.10
row f020 0.20
summary
row f005 0.05
row f035 0.35
row base2 base          # drift control for the sequential-baseline deviation
echo "=== SUMMARY $(date '+%F %T') ==="
summary
echo "=== DONE $(date '+%F %T') ==="
