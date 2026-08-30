# ARES vs SERE on one engine (vLLM 0.18.1 V1)

Measured 2026-08-08, 4× A100-SXM4-40GB, Qwen3-30B-A3B, TP=2.
Protocol: in=128 out=1024, 200 prompts, saturated — identical to the REAP/SERE/ARES
saturated runs in `/home/PC/SERE/artifacts` and `outputs/ares_ab/passB`.

## BOTTOM LINE (2026-08-08, rev. 2 — light-budget sweep landed; earlier "SERE beats ARES" verdict revised)

**On Qwen3-30B-A3B the frontier is interleaved: neither method dominates.** One engine, one
harness, one machine. GSM8K **strict-match** (flexible-extract understates every Qwen3 arm; see
Bug 2 and the relapse analysis below). Warmed protocol throughout; baselines 89.46 (n=4 acc_warmed)
and 89.69 (acc_light) — 0.23 apart.

| method | ×throughput | GSM8K strict | Δ |
|---|---|---|---|
| baseline | 1.000 | 89.46–89.69 | — |
| SERE S=2 ρ=0.5 | 1.116× | 89.46 | 0.00 |
| `percentile` p50/p90 *(ARES's OLD rule)* | 1.226× | 88.86 | −0.61 |
| **ARES `contrib_mass` frac 0.05, ks=0** | **1.259×** | **89.61** | **+0.15** |
| SERE S=1 ρ=0.3 | 1.414× | 88.55 | −0.91 |
| **SERE S=1 ρ=0.0** | **1.456×** | **87.72** | **−1.74** |
| **ARES `contrib_mass` frac 0.10, ks=0** | **1.465×** | **86.13** | −3.33 |
| **ARES `contrib_mass` frac 0.15, ks=0** | **1.513×** | 84.46 | −5.00 |
| ARES `contrib_mass` frac 0.20, ks=0 | 1.623× | 69.75 | −19.71 |

Reading the frontier:
* **ARES wins the low-loss end**: frac 0.05 is quality-neutral on strict-match (+0.15, i.e. noise)
  at 1.259× — strictly dominating SERE S=2 ρ=0.5 (1.116×, 0.00) *and* the old `percentile` rule
  (1.226×, −0.61). The earlier conclusion that `percentile` was on a better frontier than
  `contrib_mass` was an artifact of only having the frac-0.20 point.
* **SERE wins the middle band** (~1.41–1.46×): at ARES's speed-matched point (frac 0.10, 1.465×)
  SERE gives up 1.6 fewer points.
* **ARES extends past SERE's ceiling**: SERE tops out at 1.456×; ARES frac 0.15 reaches 1.513× at
  −5.00.
* **There is a cliff between frac 0.15 and 0.20** (−5.0 → −19.7). The old headline config sat past
  the cliff edge; frac 0.05–0.15 is the usable range. Runtime surrendered gate mass tracks it:
  18% / 28% / 34% / 43% at frac 0.05/0.10/0.15/0.20.
* Caveat: the concurrency sweep below used frac 0.20; at the now-preferred frac 0.10–0.15 ARES is
  slower still at low concurrency, so SERE's low-batch advantage only grows.

Pending (`rescale_ab.sh`, running): whether `EXPERT_SKIP_RENORM=1` or
`EXPERT_SKIP_MAGNITUDE_RESCALE=1` — both free at runtime and OFF in every run above — claw back
part of the −3.3/−5.0. The mask drops ~28% of *gate weight* at frac 0.10 while its budget only
accounts for ~10% of *output mass*; the kept weights sum to ~0.72 instead of 1, which these
compensate.

**What survives from the earlier draft of this document:**
* ARES 1.623× > SERE 1.456× **on raw speed** — solid, 6 baseline replicates at 0.34–0.68%.
* `CONFIDENCE_KEEP_SHARE=0` is worth 1.38× on Qwen3 / 1.51× on Qwen1.5 — solid, but on Qwen1.5
  it destroys the model.
* The SERE V1 port and its validation — solid.

**What is withdrawn:** every "matched quality" claim. They rested on contaminated accuracy.

### Concurrency: SERE is faster at every point except the saturated extreme

Qwen3, out=256, one server per arm with concurrency swept inside it. Baselines reproduce to
0.03–0.75% between the two arms.

| concurrency | ARES ×thr | SERE ×thr | winner |
|---|---|---|---|
| 1 | 1.055× | **1.125×** | SERE +6.7% |
| 4 | 1.236× | **1.390×** | SERE +12.5% |
| 16 | 1.485× | **1.712×** | SERE +15.4% |
| 64 | 1.700× | 1.718× | tied (+1.1%) |
| saturated (out=1024) | **1.623×** | 1.456× | ARES |

**This kills the low-concurrency niche hypothesis.** The expectation was that SERE degenerates at
small batch — its primary set is the batch-wide union of top-S, which at batch 1 is a *single*
expert (and indeed produces gibberish there). But it is *faster* at small batch, and by the
largest margins. The reason neither method gains much at batch 1 is that MoE weight loading is
not the bottleneck there at all; attention, dense layers and launch overhead dominate, so
eliminating experts buys little regardless of how many you eliminate.

ARES's relative position improves monotonically with batch size (1.055× → 1.700×) because the
baseline touches more distinct experts as the batch grows, giving a fixed mask more to remove.
The crossover is beyond c=64. So ARES's only winning regime is *saturated, long-output* serving
— exactly where it costs 19.71 GSM8K points and SERE costs 1.74.

**Answered (2026-08-08):** the light-budget sweep landed — frac 0.05 → 1.259× at +0.15 strict,
frac 0.10 → 1.465× at −3.33, frac 0.15 → 1.513× at −5.00. Frac 0.10 misses the "≤2 points at
≥1.45×" bar by 1.3 points *before* weight compensation; see BOTTOM LINE rev. 2 and the pending
`rescale_ab.sh` run.

## Full-audit note (2026-08-08)

Every published number was re-derived from the raw artifacts by two independent audit passes:
* **Speed**: all ~30 ratios reproduce to ≤0.1% of the values in this file; all 57 bench JSONs have
  `completed == num_prompts` and exact `num_prompts × output_len` output tokens (no silent
  failures, no truncation); configs matched within every pair; 15 Qwen3 TP=2 saturated baseline
  replicates span 2.11% total (0.34% within the frontier round). The phrase "six replicates at
  0.34–0.68%" earlier in this file is not traceable to a specific six — magnitude right, endpoint
  cosmetic. Unreported-but-valid ratios: frac 0.25 → 1.706×, pct p90/p99 → 1.825×, Qwen1.5
  keep0 frac 0.35 → 1.658×.
* **Accuracy**: every quoted score reproduces to the decimal; all n = 1319/164; zero client-side
  errors in all 58 current-generation runs. Contaminated cells are correctly quarantined — plus
  one more found: `results/final/ares025` (88.93 strict) had its mask installed 13 s *after* the
  GSM8K run started → INVALID, never cite it. Two `acc_warmed/pct_*` gsm8k cells had the warm
  confirmation fall through to the token-budget proxy, but their skip.logs show `initial BUILD
  complete` before eval start → valid.
* **Cross-contamination ruled out**: `.env` leaks `EXPERT_SKIP_MODE=dynamic` into later baseline
  launches in `bench_concurrency.sh` (main-shell `source`), but ARES only activates via
  `sitecustomize.py` on `PYTHONPATH`, which baselines never set; the `sere_vllm` plugin in the
  shared venv no-ops unless the checkpoint's architecture is a SERE one. `UNWEIGHTED_NORMS=1`
  does not corrupt `contrib_mass` (weighted mass snapshotted before the division).
* **Metric-mixing warning**: the headline tables are strict-match, but the Bug-1 magnitude table,
  the superseded frontier table, and the **entire Qwen1.5 sweep** are flexible-extract. For
  Qwen1.5 flexible is the defensible choice (base strict is 15.77 — the model rarely emits
  `####`) but it must be labeled as such in the paper.
* **cm_f005's flexible score (75.74 vs 89.61 strict) is a stopping artifact, not capability
  loss**: 188/1319 responses (vs ~61 baseline) emit the correct `#### N` then relapse into a
  fresh chain-of-thought without hitting a stop sequence; flexible-extract grabs the last number
  of the relapse. Light masks perturb termination behaviour before arithmetic. Related: HumanEval
  `pass@1,create_test` remains unusable as a quality signal (cm_f015 scores +10.4 *above*
  baseline while losing 5 GSM8K points — Bug 2 again).

## Why this run exists

SERE was previously only measurable on vLLM 0.8.4 **V0**, ARES on 0.18.1 **V1**. On identical
hardware and protocol the V0 Qwen3 baseline is **1.54× slower** than V1's (2658.8 vs 4086.9
tok/s), so the two speedup ratios could not be compared in either direction. This run puts both
methods on the same engine, same venv, same GPUs, same protocol.

## The port

`/home/PC/SERE_v1` — a copy; `SERE/` and `SERE_customized/` are untouched. Built on the user's
existing `vllm_v1_patch.py`, which was structurally correct but had two defects that made it
unusable for measurement.

**Defect 1 — the reroute never ran.** vLLM 0.18.1 ends `FusedMoE.__init__` with
`self.runner = DefaultMoERunner(..., router=self.router, ...)`, capturing the router **by
reference**. The patch replaced `experts.router` after `__init__`, so the runner kept the
original and `custom_routing_function` was never called. The failure was silent and
convincing: server healthy, "Enabled SERE routing" logged 96×, correct code generated — while
running stock Qwen3. It would have reported SERE as having *no speedup and no accuracy loss*.
Fix: rebuild the runner, as vLLM's own `_replace_quant_method` does.

**Defect 2 — silent fallback to a graph-unsafe path.** The shipped `.so` was cpython-310 only;
under the py3.13 V1 venv `_reroute` fell back to `reroute_torch`, which uses `torch.nonzero`
and `.any()` host syncs and so cannot be CUDA-graph captured. That path would have badly
understated SERE. It now hard-fails with a build instruction (`SERE_ALLOW_TORCH_REROUTE=1`
overrides).

Also added: similarity-matrix load assertion (identity ⇒ calibration didn't load ⇒ silent
no-op at ρ>0 or remap-everything at ρ=0), device-side reroute telemetry, and `sorted=True` in
`_torch_topk` (the top-S slice is meaningless without it).

### Port validation

| check | result |
|---|---|
| Kernel builds on torch 2.10 / py3.13 | ✅ zero source changes |
| CUDA kernel == torch reference | ✅ exact, E∈{60,128} × S∈{1,2} × ρ∈{0,0.3,0.5} |
| CUDA-graph capture + replay | ✅ safe, replay == eager |
| `topk_ids[:, :S]` is the true top-S | ✅ `fused_topk` is score-descending |
| Reroute actually executes | ✅ greedy output changes vs the inert control |
| **V0/V1 agreement, S=2 ρ=0.5** | ✅ **1.097× vs 1.116%, inside the noise floor** |

## Speed

Every treatment ran beside a **contemporaneous baseline** on the same 4 GPUs (2 TP=2 cells per
round), so contention is matched rather than imported. Four rounds ⇒ four baseline replicates.

**Baseline replicates:** 4030.4 / 4101.0 / 4115.5 / 4090.5 tok/s — mean 4084.3,
**stdev 0.9%, spread 2.1%**. That is the noise floor.

| method | out tok/s | ×throughput | mean TPOT | ×TPOT | mean TTFT |
|---|---|---|---|---|---|
| **ARES** (contrib_mass, frac 0.2) | 4736.5 | **1.175×** | 33.19 | 1.166× | 786.2 |
| SERE S=1 ρ=0.0 | 5971.4 | **1.456×** | 26.20 | 1.448× | 865.8 |
| SERE S=1 ρ=0.3 | 5817.9 | **1.414×** | 27.01 | 1.398× | 787.8 |
| SERE S=2 ρ=0.5 | 4564.1 | **1.116×** | 34.40 | 1.099× | 832.9 |

All cells: `completed=200`, `total_output_tokens=204800` — identical work, no EOS confound.

**ARES beats SERE at SERE's least-aggressive setting (1.175× vs 1.116×) and loses to its
aggressive ones (1.41–1.46×).** The 5.3% ARES-vs-S2ρ0.5 gap is ~6× the baseline stdev.

## ⚠ ALL ACCURACY NUMBERS BELOW ARE INVALID — two independent measurement bugs

Both were found on 2026-08-08 after the speed results were complete. **The speed results are
unaffected**; every accuracy comparison in this document is affected.

### Bug 1 — the mask arrived partway through scoring

There was no warm-up before the eval, so the eval itself did the profiling and the mask
installed mid-benchmark. Measured: on a ~102 s GSM8K run the mask finalized ~40 s in, so a large
share of scored problems ran as the **unmodified model**. The bias is one-sided — SERE loads a
static similarity matrix and is active from token 0, so SERE was never contaminated while ARES
always was.

Magnitude, `contrib_mass` frac 0.20 on GSM8K:

| | GSM8K |
|---|---|
| baseline (n=3) | 85.19 |
| contaminated (mask mid-run) | **90.07** (+4.88) |
| warmed (mask installed first) | **70.13** (−15.06) |

A 15-point collapse was reported as a 5-point gain. Fixed by
`scripts/warm_until_mask.py`, which sends held-out same-domain prompts (GSM8K test ← GSM8K
*train*; HumanEval ← MBPP) and refuses to score until the controller logs
`initial BUILD complete`.

### Bug 2 — HumanEval-base `pass@1` ranks arms by termination, not correctness

Paired analysis over the same 164 problems, baseline vs `percentile` p50/p90 (which appeared to
beat baseline by +10.8):

| subset | n | baseline | `pct_p50p90` |
|---|---|---|---|
| **both arms emit short code (<1000 ch)** | 60 | **76.7%** | **76.7%** |
| either arm runs long (>=2000 ch) | 88 | 5.7% | **25.0%** |

On problems where both produce concise code the arms are **identical**. The whole gap comes from
the long-generation subset: HumanEval-base is bare completion, so the model keeps writing past
the solution, hits the 1024-token cap, and is scored on truncated code. A mask that damages the
model's tendency to continue makes it terminate more often, which raises pass@1 without
improving any code it actually finishes.

Length distribution confirms the mechanism (bimodal: clean short functions vs capped runs):

| arm | p25 | median | p75 | >=2000 ch | pass@1 |
|---|---|---|---|---|---|
| baseline | 297 | 1678 | 3161 | 72/164 | 31.71 |
| `pct_p50p90` | 133 | 430 | 2630 | 51/164 | 42.07 |
| `pct_p90p99` | 1834 | 2974 | 3831 | 116/164 | 0.00 |

pass@1 tracks the *fraction of capped generations* across all three arms. Fix: score the
completed-generation subset, raise the cap, or use a harness with real stop handling. Otherwise
configurations get ranked by termination behaviour.

### Bug 2b — original framing (superseded by the paired test above)

pass@1 on this metric tracks *output length*, not model quality:

| arm | pass@1 | median output chars |
|---|---|---|
| baseline (×2) | 31.71 | 1678 |
| `percentile` p50/p90 (warmed) | **42.07** | **430** |
| `percentile` p90/p99 (warmed) | 0.00 | 2975 |
| `cm_f020` unwarmed (≈ no mask) | 31.10 | 1597 |

HumanEval-base is bare completion, so the model normally keeps writing extra functions and prose
after the target solution and that trailing content breaks extraction/execution. A mask that
damages the model's tendency to ramble makes it stop earlier, which **raises** the score without
improving the code. The arm that "beat" baseline by 10.8 points emits generations 4× shorter.

Consequence: this metric systematically mis-ranks expert-skipping configurations, and any tuning
done against it is unsound. The protocol must move to **HumanEval+/MBPP via evalplus with stop
sequences** and **GSM8K strict-match** (not flexible-extract), with **mean generation length
reported per arm** as a standing confound check.

---

## Accuracy — same model, harness, engine, concurrency 128 (SUPERSEDED, see above)

| method | ×thr | GSM8K (n=1319) | Δ | σ | HumanEval-base (n=164) | Δ | σ |
|---|---|---|---|---|---|---|---|
| baseline | 1.000 | 85.60 | — | — | 30.49 | — | — |
| **ARES** (frac 0.2) | 1.175 | 82.94 | **−2.66** | 1.9 | 31.71 | +1.22 | 0.2 |
| SERE S=1 ρ=0.0 | **1.456** | 85.06 | −0.54 | 0.4 | 31.71 | +1.22 | 0.2 |
| SERE S=1 ρ=0.3 | 1.414 | 86.58 | +0.98 | 0.7 | 28.66 | −1.83 | 0.4 |
| SERE S=2 ρ=0.5 | 1.116 | 88.10 | +2.50 | 1.9 | 29.27 | −1.22 | 0.2 |

GSM8K is the discriminating benchmark; at n=164 every HumanEval delta is inside 0.5σ.

## The per-token rescue was costing ARES its speed

At `CONFIDENCE_KEEP_SHARE=0.15` ARES returns 1.175×. **Setting it to 0 returns 1.623×** — same
selection rule, same mask, same budget. The rescue was consuming ~28% of throughput.

It was also undoing most of the mask. The two numbers must be compared in the *same* units;
an earlier version of this analysis wrongly compared runtime gate mass against budgeted output
mass and concluded the rescue was nearly free on Qwen3:

| `keep_share` | runtime surrendered **gate** mass |
|---|---|
| 0.15 | ~18–19% |
| **0** | **43.0%** |

So the rescue clawed back more than half the mask on Qwen3 — the same failure diagnosed on
Qwen1.5, not a Qwen1.5-only problem.

## ARES speed frontier (Qwen3, saturated, rescue off)

Four baseline replicates: 4073.2 / 4078.5 / 4076.8 / 4064.7 tok/s — **0.34% spread**.

| config | pruned/layer | output mass discarded | ×thr | ×TPOT | GSM8K | HumanEval |
|---|---|---|---|---|---|---|
| baseline | — | — | 1.000 | 1.000 | 85.60 | 30.49 |
| **frac 0.20, floor 0.25** | 70.4/128 | 19.2% | **1.623×** | 1.657× | 90.07 | **29.27** |
| frac 0.35, floor 0.25 | 74.4/128 | 33.0% | 1.712× | 1.748× | 82.18 | **9.15** |
| frac 0.50, floor 0.10 | 78.4/128 | 47.1% | 1.772× | 1.831× | 61.49 | **1.83** |
| frac 0.70, floor 0.10 | 88.8/128 | 63.5% | 1.859× | 1.902× | — | — |

**There is a hard quality cliff between frac 0.2 and 0.35**: +0.09× of throughput costs a 70%
relative collapse in HumanEval. frac 0.2 is not a comfortable operating point, it is the last
config before the edge. HumanEval is far more discriminating than GSM8K here — at frac 0.35
GSM8K slips 3.4 points while HumanEval loses 21.

## Conclusion (SPEED ONLY — the quality half is withdrawn)

| method | ×throughput | HumanEval | GSM8K |
|---|---|---|---|
| **ARES frac 0.2, rescue off** | **1.623×** | 29.27 | 90.07 |
| SERE S=1 ρ=0.0 (its best) | 1.456× | 31.71 | 85.06 |
| SERE S=1 ρ=0.3 | 1.414× | 28.66 | 86.58 |
| SERE S=2 ρ=0.5 | 1.116× | 29.27 | 88.10 |
| REAP 0.50 (deletes weights) | 1.83× | — | — |

ARES is **11% faster than SERE's best configuration** on an identical stack — that part stands,
backed by six baseline replicates at 0.34–0.68% spread and verified-identical work. At frac 0.7
ARES reaches 1.859×, above REAP's 1.83× at comparable sparsity, so masking converts to
wall-clock at least as well as physically deleting weights. An earlier hypothesis in this
document that it does not is disproved.

**The "at matched quality" half of this claim is WITHDRAWN.** It rested on ARES HumanEval 29.27
vs SERE 31.71, and both bugs above attack that comparison: ARES's number was contaminated by a
mid-run mask install (SERE's was not), and the metric itself rewards shorter generations rather
than better code. Warmed GSM8K for frac 0.20 came in at 70.13 against an 85.19 baseline, so the
current evidence points to ARES's fast configuration costing **real** accuracy. Whether any ARES
operating point is both faster than SERE and quality-neutral is **open**, and is what the
light-budget sweep (frac 0.05/0.10/0.15) plus a strict-metric protocol is meant to answer.

### What this still does not establish

1. **The accuracy noise band is unmeasured.** Four baseline *speed* replicates exist (0.34%
   spread) but only one baseline *accuracy* point, while configs that should differ modestly
   spanned 82.2–90.1 on GSM8K. Until replicates land, no small accuracy delta here means
   anything — including ARES's apparent +4.47 GSM8K, which should not be reported as a gain.
2. **Two benchmarks, one of them n=164.** Not enough to claim matched quality for a paper.
3. **Low concurrency untested.** Every number is at concurrency 128 — SERE's best case, since
   its primary set is the batch-wide union of top-S and collapses to a single expert at batch 1
   (measured: gibberish). ARES's mask has no such dependence, so the gap should widen at low
   concurrency. This is the strongest untested claim available.
4. **Qwen1.5 does NOT work at `keep_share=0` — the fix is Qwen3-specific.** Tested, and the
   rescue turns out to be load-bearing on Qwen1.5 rather than overhead:

   Full sweep, both levers (lower the budget; raise the rescue threshold so only
   token-dominating slots are protected). Baseline 2161.6 tok/s, GSM8K 60.20, HumanEval 33.54:

   | config | ×thr | GSM8K | Δ | HumanEval | Δ |
   |---|---|---|---|---|---|
   | frac 0.2, ks=0.15 *(shipped default)* | **1.029×** | intact | ~0 | intact | ~0 |
   | frac 0.03, ks=0 | 1.143× | 9.55 | −50.6 | 15.24 | −18.3 |
   | frac 0.05, ks=0 | 1.233× | 7.51 | −52.7 | 10.37 | −23.2 |
   | **frac 0.10, ks=0.40** *(best trade)* | **1.278×** | **31.31** | **−28.9** | 26.22 | −7.3 |
   | frac 0.20, ks=0.40 | 1.384× | 19.33 | −40.9 | 22.56 | −11.0 |
   | frac 0.10, ks=0 | 1.370× | 2.27 | −57.9 | 9.15 | −24.4 |
   | frac 0.20, ks=0.55 | 1.509× | 2.50 | −57.7 | 10.37 | −23.2 |
   | frac 0.20, ks=0 | 1.553× | 2.27 | −57.9 | 2.44 | −31.1 |

   **No configuration above 1.1× keeps quality.** The best point in the sweep still gives up 29
   GSM8K points off a 60-point baseline. `keep_share` behaves monotonically and as designed
   (0.55→2.50, 0.40→19.33, 0.15→intact), so this is not a tuning failure: Qwen1.5-MoE has no
   exploitable redundancy at 60 experts / top_k=4.

   This also **refutes the surviving-slots-per-token rule** proposed earlier in this session:
   Qwen3 is healthy at 3.60 of 8 surviving, Qwen1.5 is destroyed at 3.02 of 4. Slot count is
   not the invariant.

   **Scope implication: ARES is a large-MoE method.** It should be claimed for `top_k>=8`
   architectures (Qwen3-30B/Coder, Mixtral, DeepSeek-V2/V3, OLMoE), not for MoEs generally.

## A structural finding about SERE

SERE's quality is **batch-size dependent** in a way ARES's is not. At S=1 the primary set is the
union of each token's top-1 expert, so at batch 1 it is a *single* expert and 7 of 8 slots
collapse onto it. Measured, greedy, same prompt:

| concurrency | correct completions |
|---|---|
| 1 | 0/1 (gibberish) |
| 8 | 2/8 |
| 64 | 1/64 (correct text begins to appear) |

This matches SERE's recorded V0 retention (S=1 ρ=0.3 → 80% 8-task average but **22% on
HumanEval**), which is further evidence the port is faithful. It also means "SERE's accuracy" is
undefined without stating concurrency — a mask under a mass budget does not degenerate this way
as the batch shrinks. Worth making an explicit axis in the paper.

## Corrections to earlier analysis

- I previously argued V1's faster baseline leaves less headroom, partly excusing ARES's smaller
  ratio. **That is wrong** — SERE's ratio *rose* on V1 (1.456× vs 1.275× on V0). Do not use it.
- I earlier quoted SERE at 1.60× on "the same saturated protocol". That figure came from a
  fixed-batch TPOT cell; the matched saturated V0 number is 1.275×.

## What this supports, and what it does not

**Supported:** at SERE's least-aggressive configuration, ARES is faster (1.175× vs 1.116×) on an
identical stack — and, pending the accuracy run, at better quality.

**Not supported:** "ARES is faster than SERE" without qualification. At S=1 SERE is decisively
faster. The defensible claim is a Pareto one, and it needs the accuracy axis to stand up.

## Reproduce

```
bash /home/PC/SERE_v1/scripts/bench_v1_headtohead.sh   # speed, 4 rounds
bash /home/PC/SERE_v1/scripts/acc_v1_sere.sh           # accuracy, 3 configs
```
Results in `results/h2h/*.json` and `results/acc/`.
