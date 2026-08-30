# ARES vs SERE — consolidated results & reproduction guide

Date: 2026-08-08. Every number in this document was re-derived from raw artifacts by an
independent audit pass on this date (speed ratios reproduce to ≤0.1%; accuracy scores to the
decimal; no incomplete or truncated runs). Companion analysis: `RESULTS_ARES_VS_SERE.md`.

---

## 1. Environment

| component | value |
|---|---|
| GPU | 4× NVIDIA A100-SXM4-40GB (driver 580.126.20) |
| CPU / RAM | 48 vCPU, 334 GB (GCP VM, hostname `moe-a10-dm`) |
| Python | 3.13.12 (venv: `/home/PC/new-efficient-moe/.venv` — **all arms, incl. SERE, run from this one venv**) |
| PyTorch | 2.10.0+cu128 (CUDA 12.8) |
| vLLM | **0.18.1 (V1 engine)** — all head-to-head numbers; SERE's official stack (vLLM 0.8.4, V0 engine, separate env) used only for the port-validation cross-check |
| lm-eval | 0.4.11 (`local-completions` API mode) |
| ARES kernels | `efficient_moe_fused_skip_cuda` 0.1.0, built via `uv pip install --no-build-isolation ./patches/vllm/fused_skip_ops` (torch 2.10 breaks the JIT path — **verify `fused_cuda_available() and fused_confidence_available() and fused_risk_accum_available()` all True before benchmarking**; earlier results with the eager fallback were 27–33× slower per layer and are excluded) |
| SERE kernel | SERE's own `rerouting_kernel.cu`, compiled unmodified; installed as editable pkg `sere_vllm` 0.1 (vLLM `general_plugins` entry point) |

## 2. Models

| model | snapshot | MoE shape |
|---|---|---|
| Qwen/Qwen3-30B-A3B | `ad44e777bcd18fa416d9da3bd8f70d33ebb85d39` | 48 layers, 128 experts, top-k 8, `norm_topk_prob=true` |
| Qwen/Qwen1.5-MoE-A2.7B | `1a758c50ecb6350748b9ce0a99d2352fd9fc11c9` | 24 layers, 60 experts, top-k 4, `norm_topk_prob=false` |
| SERE calibrated | `/home/PC/SERE/artifacts/calibrated/Qwen3-30B-A3B-sere` (Qwen3 weights + per-layer expert similarity matrices from SERE's own calibration) | as Qwen3 |

## 3. The two methods, exactly as launched

### ARES (dynamic online expert skipping, repo `/home/PC/new-efficient-moe`, branch `feat/new_method`)

Activation: the controller loads via `sitecustomize.py` **only when `PYTHONPATH=/home/PC/new-efficient-moe`
is set**. Servers without it are stock vLLM regardless of `EXPERT_SKIP_*` env vars (this is what
keeps baselines clean even though `.env` is sourced broadly).

Canonical treatment-arm launch (Qwen3):

```bash
set -a; source /home/PC/new-efficient-moe/.env; set +a   # defaults; overridden below
env CUDA_VISIBLE_DEVICES=2,3 PYTHONPATH=/home/PC/new-efficient-moe \
    EXPERT_SKIP_MODE=dynamic \
    EXPERT_SKIP_DISABLED_LAYERS=0,47 \                    # Qwen1.5: 0,23
    EXPERT_SKIP_ONLINE_THRESHOLD_METHOD=contrib_mass \
    EXPERT_SKIP_ONLINE_CONFIDENCE_KEEP_SHARE=0 \          # rescue OFF (see §7)
    EXPERT_SKIP_ONLINE_CONTRIB_MASS_FRACTION=<FRAC> \     # the one swept knob
    EXPERT_SKIP_ONLINE_MIN_ACTIVE_FRAC=0.25 \
    EXPERT_SKIP_ONLINE_MIN_TOKENS_FOR_FINALIZE=4096 \
    EXPERT_SKIP_ONLINE_MIN_PER_EXPERT=8 \
    EXPERT_SKIP_ONLINE_UNWEIGHTED_NORMS=1 \
    EXPERT_SKIP_ONLINE_REFRESH_EVERY_N_TOKENS=9999999 \   # one mask, never refreshed
    EXPERT_SKIP_ONLINE_DRIFT_CHECK_EVERY_N_STEPS=0 \
    EXPERT_SKIP_ONLINE_RISK_TRIGGER=0 \                   # risk monitor logs only
    EXPERT_SKIP_ONLINE_LOG=1 EXPERT_SKIP_ONLINE_LOG_FILE=<skip.log> \
    .venv/bin/vllm serve Qwen/Qwen3-30B-A3B --served-model-name m \
    --trust-remote-code --tensor-parallel-size 2 \
    --max-model-len 4096 \                                # speed runs; accuracy runs use 8192
    --gpu-memory-utilization 0.90 --port <P> --no-enable-prefix-caching
```

Semantics: during profiling (prefill tokens only, budget 4096 tokens) each expert accumulates
contribution mass `Σ w_k·‖E_k(x)‖`; at finalize, experts are pruned cheapest-first per layer
while cumulative pruned mass ≤ `CONTRIB_MASS_FRACTION`, hard-capped so ≥ `MIN_ACTIVE_FRAC`
(25%) of experts survive per layer. First and last layers never skip. The mask is installed once
and never refreshed. `EXPERT_SKIP_RENORM=0` and `EXPERT_SKIP_MAGNITUDE_RESCALE=0` (defaults —
turning either on *hurts*, §6d). `UNWEIGHTED_NORMS=1` affects only the magnitude-method scores;
`contrib_mass` snapshots the weighted mass before the division, verified in code.

Ablation arm (`percentile`, ARES's pre-Aug-3 rule): same block with
`THRESHOLD_METHOD=percentile`, `MEAN_ACT_PERCENTILE=50`, `VAR_ACT_PERCENTILE=90` (or 90/99).

### SERE (batch-wide expert rerouting, V1 port at `/home/PC/SERE_v1/vllm/SERE_vllm`)

Activation: `sere_vllm` is a vLLM plugin in the venv; it no-ops unless the model config has a
SERE architecture or a `select_top_k` key. The stock-architecture calibrated checkpoint is
switched on via `--hf-overrides`:

```bash
CUDA_VISIBLE_DEVICES=2,3 .venv/bin/vllm serve \
    /home/PC/SERE/artifacts/calibrated/Qwen3-30B-A3B-sere \
    --served-model-name m --trust-remote-code --tensor-parallel-size 2 \
    --max-model-len 4096 --gpu-memory-utilization 0.90 --port <P> \
    --no-enable-prefix-caching \
    --hf-overrides '{"select_top_k": 1, "threshold": 0.0}'   # = S, ρ
```

Port validation (details + scripts in `PLAN_SERE_V1_PORT.md`, `scripts/final_checks.sh`):
SERE's CUDA kernel compiled from unmodified source; kernel bit-exact vs the torch reference
across E∈{60,128} × S∈{1,2} × ρ∈{0,0.3,0.5}; identity-similarity-matrix misload hard-fails;
S=2 ρ=0.5 throughput ratio 1.097× on official V0 vs 1.116× on this port (1.7% apart).

### Baseline

Identical `vllm serve` line, stock model, **no PYTHONPATH, no hf-overrides**. Every treatment
number is paired with a *contemporaneous* baseline on the other 2 GPUs in the same round
(never a baseline from another day/file). 15 Qwen3 TP=2 saturated baseline replicates:
4030–4116 tok/s (2.11% total spread; ≤0.34% within a round).

## 4. Speed harness

Warmup pass then measured pass, per server:

```bash
vllm bench serve --model m --tokenizer <model_path> --trust-remote-code \
  --dataset-name random --random-input-len 128 --random-output-len 1024 \
  --num-prompts 64  --port <P>                       # warmup (also builds the ARES mask)
vllm bench serve ... --random-output-len 1024 --num-prompts 200 \
  --percentile-metrics ttft,tpot --save-result ...   # measured, request_rate=inf (saturated)
```

- **Saturated protocol** (headline): in=128, out=1024, 200 prompts, no rate limit, TP=2.
  Metric: `Output token throughput (tok/s)`, ratio vs paired baseline.
- **Concurrency sweep**: out=256, `--max-concurrency c` for c∈{1,4,16,64},
  `num_prompts = max(32, 4c)`, one server per arm, warmup once before the sweep so the ARES
  mask exists before any timed point.
- Qwen1.5 runs: TP=1 (fits one GPU), otherwise identical.

## 5. Accuracy harness

lm-eval 0.4.11 through `scripts/serve/run_eval_serve.py` (repo `new-efficient-moe`) against the
OpenAI-compatible server: `local-completions`, `--max_concurrency 128`, temperature 0,
`--log_samples`. Tasks: `gsm8k` (n=1319, lm-eval default few-shot, `max_gen_toks=256`,
stop=`["Question:", "</s>", "<|im_end|>"]`) and `humaneval_base` (n=164, pass@1,create_test,
`max_gen_toks=1024`). Server: TP=2, `--max-model-len 8192` (prefix caching left at default —
identical across arms).

**Warm-before-score protocol (mandatory for ARES arms).** The profiling budget advances on
prefill only; without warming, the mask lands mid-eval and part of the score comes from the
unmodified model (measured inflation: up to ~+15 points, one-sided in ARES's favour since
SERE's matrix is active from token 0). `scripts/warm_until_mask.py` sends held-out same-domain
traffic until the skip log confirms the mask is installed (`initial BUILD complete` /
finalize lines / `risk=` telemetry), exits nonzero otherwise:

- GSM8K test ← warmed on **GSM8K train** (disjoint)
- HumanEval ← warmed on **MBPP** with `code`+`test_list` appended (prompt-length matched;
  disjoint problems)
- Baseline and SERE arms receive the same warm traffic budget for cache parity (SERE cells in
  `results/acc` predate this; static matrix ⇒ no contamination risk, noted for completeness).

**Metric: GSM8K `exact_match,strict-match`.** Flexible-extract is unreliable here: it takes the
*last* number, and Qwen3 sometimes emits the correct `#### N` then relapses into a fresh
chain-of-thought (baseline: ~61/1319 such docs; ARES frac 0.05: 188 — light masks perturb
stopping before arithmetic). Exception: Qwen1.5 tables use flexible-extract because that model
rarely emits `####` at all (base strict = 15.77 vs flexible 60.20) — label accordingly.
**HumanEval-base pass@1,create_test is reported but is NOT a usable quality signal** at these
deltas: a config that loses 5 GSM8K points scores +10.4 *above* baseline (termination/length
confound; verified twice).

## 6. Results (all audited)

### a) Qwen3-30B-A3B frontier — saturated speed × warmed GSM8K strict

Baselines: 89.46 / 89.69 strict (two independent warmed baseline cells).

| arm | ×throughput | GSM8K strict | Δ | HumanEval* |
|---|---|---|---|---|
| SERE S=2 ρ=0.5 | 1.116× | 89.46 | 0.00 | 29.27 |
| percentile p50/p90 | 1.226× | 88.86 | −0.61 | 42.07* |
| **ARES frac 0.05** | **1.259×** | **89.61** | **+0.15** | 35.37 |
| SERE S=1 ρ=0.3 | 1.414× | 88.55 | −0.91 | 28.66 |
| SERE S=1 ρ=0.0 | 1.456× | 87.72 | −1.74 | 31.71 |
| **ARES frac 0.10** | **1.465×** | 86.13 | −3.33 | 31.71 |
| **ARES frac 0.15** | **1.513×** | 84.46 | −5.00 | 43.29* |
| ARES frac 0.20 | 1.623× | 69.75 | −19.71 | 35.37 |
| ARES frac 0.25 | 1.706× | (invalid cell — mask landed mid-eval) | — | — |
| ARES frac 0.35 / 0.50 / 0.70 | 1.712× / 1.772× / 1.859× | not measured warmed (expect ≤ frac 0.20) | — | — |
| percentile p90/p99 | 1.825× | 0.38 (destroyed) | — | 0.00 |

\* HumanEval values with * exceed baseline (32.93) while GSM8K drops — the Bug-2 confound; do not
interpret. Headline conclusions: the frontier is **interleaved** (ARES wins ≤1.26× and ≥1.51×,
SERE wins the 1.41–1.46× band); there is an accuracy **cliff between frac 0.15 and 0.20**
(runtime surrendered gate mass 18/28/34/43% at frac 0.05/0.10/0.15/0.20).

### b) Concurrency (Qwen3, out=256, ARES frac 0.20 ks=0, SERE S=1 ρ=0)

| c | ARES ×thr | SERE ×thr |
|---|---|---|
| 1 | 1.055× | **1.125×** |
| 4 | 1.236× | **1.390×** |
| 16 | 1.485× | **1.712×** |
| 64 | 1.700× | 1.718× |
| saturated, out=1024 | **1.623×** | 1.456× |

SERE wins everywhere except the saturated extreme. (ARES at frac 0.10–0.15 is slower than
frac 0.20, so this ordering can only shift further toward SERE at the quality-safe budgets.)

### c) Qwen1.5-MoE-A2.7B — no viable ARES config found

Baseline TP=1: 2162 tok/s, GSM8K flexible 60.20, HumanEval 33.54. Every config >1.1× loses
≥29 GSM8K points (best trade: frac 0.10 + keep_share 0.40 → 1.278× at 31.31 flexible).
Diagnosis: at top_k=4 with 60 experts, a 20%-mass budget leaves ~1.4 surviving slots/token vs
Qwen3's 3.6/8. Sweep table in `RESULTS_ARES_VS_SERE.md`; raw in `results/q15_sweep`.

### d) Weight compensation after masking: **hurts — keep both OFF** (negative result)

Motivation: at frac 0.10 the mask drops 27.7% of *gate weight* (budget counts *output mass*,
only 10%), so kept weights sum to ~0.72. Both built-in fixes were tested warmed:

| arm | GSM8K strict | vs plain mask |
|---|---|---|
| frac 0.10 plain | 86.13 | — |
| + `EXPERT_SKIP_RENORM=1` | 82.18 | −3.95 |
| + `EXPERT_SKIP_MAGNITUDE_RESCALE=1` | 67.25 | −18.88 |
| frac 0.15 plain | 84.46 | — |
| + RENORM | 69.67 | −14.79 |
| + MAGNITUDE_RESCALE | 22.74 | −61.72 |

Rescaling adds energy along the *kept* experts' directions while the dropped directions stay
missing — undersized-but-right-direction beats resized-wrong-direction. Consistent with §e.
(Magnitude-rescale also costs ~4% throughput: 1.456× vs 1.512× plain at frac 0.15.)
Raw: `results/rescale_ab`.

### e) Per-token confidence rescue: OFF for speed (Qwen3), load-bearing on Qwen1.5

`CONFIDENCE_KEEP_SHARE=0.15 → 0` took Qwen3 frac 0.20 from 1.175× to 1.623× (the entire
speed win). On Qwen1.5 the rescue was the only thing keeping the model alive (removing it:
GSM8K 60.20 → 2.27).

### f) V0↔V1 cross-check (baseline-legitimacy answer)

Qwen3 V1 baseline is 1.54× the V0 baseline. SERE S=2 ρ=0.5: 1.097× on official V0 stack vs
1.116× on the V1 port — ratios agree within 1.7%. Accuracy is stack-invariant, so SERE quality
can also be quoted from the authors' own V0 stack directly.

## 7. Pitfalls a reproducer must avoid (each one bit us)

1. **Unbuilt fused kernel** → eager fallback, 27–33× slower/layer; assert all three
   `fused_*_available()` probes before trusting any speed number.
2. **Unwarmed accuracy** → mask lands mid-eval, score is a blend (up to +15 points, one-sided).
   Always run `warm_until_mask.py` and require rc=0 (or `initial BUILD complete` before the
   eval's first request timestamp).
3. **flexible-extract on Qwen3** → understates every arm (post-`####` relapse); use strict.
4. **HumanEval pass@1,create_test** → blind to 20-point GSM8K losses; never use as the quality
   gate.
5. **Non-contemporaneous baselines** → 2.1% cross-day drift ≈ half of some effect sizes; pair
   every treatment with a same-round baseline.
6. **`completed < num_prompts` / all-zero result JSONs** (wrong port) → check `completed` and
   total token counts in every bench JSON.
7. Keep `--no-enable-prefix-caching` on speed runs (random prompts still share a template
   prefix).

## 8. Artifact map

| what | where |
|---|---|
| drivers (speed) | `SERE_v1/scripts/bench_v1_headtohead.sh`, `bench_ares_frontier.sh`, `bench_ares_light.sh`, `bench_concurrency.sh` |
| drivers (accuracy) | `SERE_v1/scripts/acc_warmed.sh`, `acc_light.sh`, `rescale_ab.sh`, `qwen15_rescue_sweep.sh`; warm gate: `warm_until_mask.py` |
| raw speed JSONs | `SERE_v1/results/{h2h,frontier,frontier_light,concurrency,q15_sweep,ablation}` |
| raw accuracy | `SERE_v1/results/{acc_warmed,acc_light,rescale_ab,q15_sweep,acc}` (per-cell `warm.log`, `skip.log`, `client.log`, lm-eval `eval_results.json` + samples) |
| SERE port + validation | `SERE_v1/vllm/SERE_vllm/`, `SERE_v1/scripts/final_checks.sh`, `SERE_v1/PLAN_SERE_V1_PORT.md` |
| analysis | `SERE_v1/RESULTS_ARES_VS_SERE.md` (verdicts, bugs, audit note), `new-efficient-moe/RESULTS_ARES.md` |
| known-invalid cells (do not cite) | `results/final/ares025` (mask mid-eval), everything under `results/acc_frontier` + `results/ablation/pct_*` (unwarmed), `acc_warmed/cm_f020/humaneval_base_UNWARMED_INVALID` |
