#!/usr/bin/env bash
# GO/NO-GO smoke test for Qwen3.6-35B-A3B (Qwen3_5MoeForConditionalGeneration).
#
# This architecture differs from everything ARES has been tested on, in four
# ways that could each break the hook or invalidate the comparison:
#
#   1. FINE-GRAINED EXPERTS: 256 experts x moe_intermediate_size 512, top_k 8
#      (Qwen3-30B-A3B: 128 x 768). Each expert carries roughly half the mass,
#      so a given CONTRIB_MASS_FRACTION should prune far MORE experts here.
#      This is the real scientific question: does the mass budget generalize
#      across expert granularity, or was frac 0.05-0.15 a Qwen3-30B constant?
#   2. HYBRID ATTENTION: 40 layers, 3x linear_attention (GatedDeltaNet) per
#      1x full_attention. Linear attention is cheaper than full, so MoE is a
#      LARGER share of total FLOPs -> more headroom for expert skipping.
#      Also means vLLM runs the IsHybrid path with a mamba-style state cache.
#   3. SHARED EXPERT: shared_expert_intermediate_size 512 always executes and
#      is NOT skippable by ARES -> a floor on achievable speedup and a cushion
#      on quality degradation. Both must be stated when comparing to Qwen3-30B
#      (which has no shared expert).
#   4. VLM: Qwen3_5MoeForConditionalGeneration wraps the LM as `language_model`.
#      Layer names become `language_model.model.layers.N.mlp.experts`, and
#      ARES's _LAYER_RE (`\.layers\.(\d+)`) must still resolve the index. The
#      vision tower uses `visual.blocks.N` so it should not collide -- but
#      "should" is why this script exists. Text-only prompts throughout.
#
# TP=4 is mandatory: 72 GB of bf16 weights over 4x A100-40GB is 18 GB/GPU,
# leaving ~18 GB for KV + mamba state at util 0.90. TP=2 would need 36 GB/GPU
# against a 36.9 GB budget -> no room for cache. This is why the Qwen3.6 speed
# protocol CANNOT use the concurrent-baseline layout every earlier model used.
#
# Checks, in order of what would kill the experiment soonest:
#   A. server boots at all on the hybrid+VLM path
#   B. ARES resolves layer indices (skip log shows `layer=N pruned=X/256`)
#   C. the mask installs before the eval would start (initial BUILD complete)
#   D. generation is COHERENT with the mask on (greedy, known-answer prompt)
#   E. what the mask actually costs at runtime (surrendered gate mass)
# Exits nonzero on the first hard failure so the caller can stop.
set -uo pipefail

VENV=/home/PC/new-efficient-moe/.venv
AR=/home/PC/new-efficient-moe
SV=/home/PC/SERE_v1
OUT=$SV/results/q36_smoke
mkdir -p "$OUT"
HUB=/home/PC/.cache/huggingface/hub
Q36=$(ls -d "$HUB"/models--Qwen--Qwen3.6-35B-A3B/snapshots/*/ 2>/dev/null | head -1); Q36=${Q36%/}
[[ -n "$Q36" && -f "$Q36/config.json" ]] || { echo "[smoke] model not downloaded"; exit 1; }
FRAC="${FRAC:-0.10}"
PORT="${PORT:-8500}"

echo "=== Qwen3.6-35B-A3B SMOKE $(date '+%F %T') ==="
echo "    model: $Q36"
echo "    frac=$FRAC  TP=4  DISABLED_LAYERS=0,39"

PYTHONPATH="$AR" "$VENV/bin/python" -c "
import patches.vllm.fused_skip_ops as m
assert m.fused_cuda_available() and m.fused_confidence_available() and m.fused_risk_accum_available()
print('[smoke] ARES kernels: OK')" || { echo "[smoke] KERNEL FAIL"; exit 1; }

set -a; source "$AR/.env"; set +a
env CUDA_VISIBLE_DEVICES=0,1,2,3 PYTHONPATH="$AR" \
    EXPERT_SKIP_MODE=dynamic \
    EXPERT_SKIP_DISABLED_LAYERS=0,39 \
    EXPERT_SKIP_ONLINE_THRESHOLD_METHOD=contrib_mass \
    EXPERT_SKIP_ONLINE_CONFIDENCE_KEEP_SHARE=0 \
    EXPERT_SKIP_ONLINE_CONTRIB_MASS_FRACTION="$FRAC" \
    EXPERT_SKIP_ONLINE_MIN_ACTIVE_FRAC=0.25 \
    EXPERT_SKIP_ONLINE_UNWEIGHTED_NORMS=1 \
    EXPERT_SKIP_ONLINE_REFRESH_EVERY_N_TOKENS=9999999 \
    EXPERT_SKIP_ONLINE_LOG=1 EXPERT_SKIP_ONLINE_LOG_FILE="$OUT/skip.log" \
    EXPERT_SKIP_ONLINE_DRIFT_CHECK_EVERY_N_STEPS=0 \
    EXPERT_SKIP_ONLINE_RISK_TRIGGER=0 \
    "$VENV/bin/vllm" serve "$Q36" --served-model-name m \
    --trust-remote-code --tensor-parallel-size 4 --max-model-len 8192 \
    --gpu-memory-utilization 0.90 --port "$PORT" --no-enable-prefix-caching \
    > "$OUT/server.log" 2>&1 &
PID=$!
trap 'kill $PID 2>/dev/null' EXIT

w=0
until curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; do
  kill -0 $PID 2>/dev/null || {
    echo "[smoke] A. SERVER DIED -- root cause:"
    grep -oE "(ValueError|RuntimeError|NotImplementedError|KeyError|CUDA error|torch.OutOfMemoryError)[:.].{0,200}" \
      "$OUT/server.log" | sort -u | head -5
    exit 2; }
  (( w > 2400 )) && { echo "[smoke] A. TIMEOUT after ${w}s"; exit 2; }
  sleep 15; w=$((w+15))
done
echo "[smoke] A. server up in ${w}s -- hybrid+VLM path OK"

# B+C: drive prefill until the mask installs. Held-out GSM8K train, same gate
# the real accuracy runs use, so a pass here means the accuracy protocol works.
"$VENV/bin/python" "$SV/scripts/warm_until_mask.py" --port "$PORT" --model m \
  --warm-source gsm8k_train --skip-log "$OUT/skip.log" 2>&1 | tail -4
WRC=${PIPESTATUS[0]}

"$VENV/bin/python" - "$OUT/skip.log" <<'PY'
import re, sys, statistics as st
try:
    t = open(sys.argv[1], errors="ignore").read()
except OSError:
    print("[smoke] B. NO SKIP LOG -- ARES never logged; hook did not attach"); sys.exit(3)
rows = re.findall(r"layer=(\d+) pruned=(\d+)/(\d+)", t)
if not rows:
    print("[smoke] B. FAIL -- no `layer=N pruned=X/E` lines. Layer indices did not")
    print("          resolve (VLM prefix?) or finalize never ran.")
    sys.exit(3)
last = {}
for a, b, c in rows:
    last[int(a)] = (int(b), int(c))
vals = list(last.values())
E = vals[0][1]
pruned = st.mean(v[0] for v in vals)
print(f"[smoke] B. OK -- {len(last)} layers masked, experts/layer={E}, "
      f"mean pruned={pruned:.1f}/{E} ({pruned/E*100:.0f}%)")
print(f"          layer indices seen: min={min(last)} max={max(last)}")
print(f"          surviving routed slots/token = {8*(1-pruned/E):.2f} of 8 "
      f"(+ shared expert, always on)")
if E != 256:
    print(f"          WARNING: expected 256 experts, saw {E}")
risk = [float(x) for x in re.findall(r"risk=([0-9.]+)", t)]
if risk:
    print(f"[smoke] E. runtime surrendered gate mass: median={st.median(risk):.4f} "
          f"max={max(risk):.4f} n={len(risk)}")
else:
    print("[smoke] E. no risk telemetry yet (needs decode traffic)")
print("[smoke] C. mask installed" if "initial BUILD complete" in t
      else "[smoke] C. WARNING: no `initial BUILD complete` line")
PY
BRC=$?
[[ $BRC -ne 0 ]] && exit $BRC
echo "[smoke] warm gate rc=$WRC (0 = mask confirmed before scoring)"

# D: coherence. A mask that installs but produces garbage is a silent failure --
# this is exactly how Qwen1.5 at frac 0.20 looked (1.5x faster, GSM8K 2.27).
echo "[smoke] D. coherence check (greedy, mask ON):"
for q in "Question: Natalia sold clips to 48 friends in April, and half as many in May. How many clips did she sell altogether?\nAnswer:" \
         "Question: What is 17 multiplied by 4?\nAnswer:"; do
  curl -sf "http://127.0.0.1:$PORT/v1/completions" -H 'Content-Type: application/json' \
    -d "$(printf '{"model":"m","prompt":"%s","max_tokens":80,"temperature":0}' "$q")" \
    | "$VENV/bin/python" -c "
import json,sys
try: print('   ->', repr(json.load(sys.stdin)['choices'][0]['text'][:180]))
except Exception as e: print('   -> REQUEST FAILED', e)"
done

echo "=== SMOKE DONE $(date '+%F %T') -- inspect D above for coherence ==="
