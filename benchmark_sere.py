#!/usr/bin/env python3
"""
Standalone throughput benchmark for SERE (vLLM 0.8.4).

Matches the measurement protocol of new-efficient-moe's
benchmark_throughput_sere_style.py so results are directly comparable:

  - Random token-ID prompts (seed=42, avoids prefix caching)
  - ignore_eos=True  →  every request generates exactly output_len tokens
  - temperature=1.0  (configurable)
  - Per-step hook for TTFT and ITL, plus E2E wall clock for throughput

Primary metrics (identical definition in both scripts):
  output_throughput_tps  =  total_output_tokens / elapsed_s
  TPOT_e2e               =  elapsed_ms / total_output_tokens
  requests_per_s         =  num_requests / elapsed_s

Usage
-----
# Baseline (no SERE, plain Qwen3 Coder)
VLLM_USE_V1=0 python benchmark_sere.py \\
    --model Qwen/Qwen3-Coder-30B-A3B-Instruct \\
    --input_len 512 --output_len 256 --batch_size 32

# SERE-accelerated (after calibration)
VLLM_USE_V1=0 python benchmark_sere.py \\
    --model ./calibration/output/qwen3_coder_sere \\
    --sere \\
    --select_top_k 2 --threshold 0.0 \\
    --input_len 512 --output_len 256 --batch_size 32

# Sweep batch sizes
VLLM_USE_V1=0 python benchmark_sere.py \\
    --model ./calibration/output/qwen3_coder_sere \\
    --sere --select_top_k 2 --threshold 0.0 \\
    --sweep_batch_sizes 1,2,4,8,16,32 \\
    --input_len 512 --output_len 256
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

os.environ.setdefault("VLLM_USE_V1", "0")


# ---------------------------------------------------------------------------
# Per-step timing (mirrors timing_utils.py in new-efficient-moe)
# ---------------------------------------------------------------------------

@dataclass
class StepRecord:
    step_idx: int
    t_start_ns: int
    t_end_ns: int

    @property
    def duration_ms(self) -> float:
        return (self.t_end_ns - self.t_start_ns) / 1e6


def _wrap_engine_step(engine: Any):
    records: list[StepRecord] = []
    original_step = engine.step

    def _timed():
        idx = len(records)
        t0 = time.perf_counter_ns()
        result = original_step()
        t1 = time.perf_counter_ns()
        records.append(StepRecord(step_idx=idx, t_start_ns=t0, t_end_ns=t1))
        return result

    engine.step = _timed
    return records, original_step


@dataclass
class IterMetrics:
    batch_size: int
    input_len: int
    output_len: int
    total_input_tokens: int
    total_output_tokens: int
    ttft_ms: float
    mean_itl_ms: float
    p95_itl_ms: float
    e2e_latency_ms: float
    decode_throughput_tps: float
    prefill_throughput_tps: float
    e2e_throughput_tps: float
    prefill_wall_ms: float
    decode_wall_ms: float
    total_wall_ms: float
    num_prefill_steps: int
    num_decode_steps: int


def _compute_metrics(
    records: list[StepRecord],
    total_input_tokens: int,
    total_output_tokens: int,
    batch_size: int,
    input_len: int,
    output_len: int,
    wall_s: float,
) -> IterMetrics:
    prefill_steps = records[:1]
    decode_steps  = records[1:]

    prefill_wall_ms = sum(s.duration_ms for s in prefill_steps)
    decode_wall_ms  = sum(s.duration_ms for s in decode_steps)
    total_wall_ms   = wall_s * 1000.0

    if decode_steps:
        itl_samples = [s.duration_ms / batch_size for s in decode_steps]
        mean_itl = statistics.mean(itl_samples)
        sorted_itl = sorted(itl_samples)
        p95_itl = sorted_itl[max(0, int(len(sorted_itl) * 0.95) - 1)]
    else:
        mean_itl = p95_itl = 0.0

    return IterMetrics(
        batch_size=batch_size,
        input_len=input_len,
        output_len=output_len,
        total_input_tokens=total_input_tokens,
        total_output_tokens=total_output_tokens,
        ttft_ms=prefill_wall_ms / batch_size,
        mean_itl_ms=mean_itl,
        p95_itl_ms=p95_itl,
        e2e_latency_ms=total_wall_ms / batch_size,
        decode_throughput_tps=(total_output_tokens / (decode_wall_ms / 1000)
                               if decode_wall_ms > 0 else 0.0),
        prefill_throughput_tps=(total_input_tokens / (prefill_wall_ms / 1000)
                                if prefill_wall_ms > 0 else 0.0),
        e2e_throughput_tps=(total_output_tokens / wall_s if wall_s > 0 else 0.0),
        prefill_wall_ms=prefill_wall_ms,
        decode_wall_ms=decode_wall_ms,
        total_wall_ms=total_wall_ms,
        num_prefill_steps=len(prefill_steps),
        num_decode_steps=len(decode_steps),
    )


@dataclass
class AggMetrics:
    config: dict
    n_iters: int
    mode: str
    e2e_throughput_mean: float
    e2e_throughput_std: float
    decode_throughput_mean: float
    decode_throughput_std: float
    prefill_throughput_mean: float
    prefill_throughput_std: float
    ttft_mean_ms: float
    ttft_std_ms: float
    mean_itl_mean_ms: float
    mean_itl_std_ms: float
    p95_itl_mean_ms: float
    p95_itl_std_ms: float
    e2e_latency_mean_ms: float
    e2e_latency_std_ms: float
    total_output_tokens: int
    total_input_tokens: int
    # SERE-style top-level keys (identical names to benchmark_throughput_sere_style.py)
    sere_output_throughput_tps: float = 0.0
    sere_tpot_ms: float = 0.0
    sere_requests_per_s: float = 0.0


def _aggregate(iters: list[IterMetrics], mode: str, output_len: int) -> AggMetrics:
    def ms(vals):
        mean = statistics.mean(vals)
        std = statistics.stdev(vals) if len(vals) > 1 else 0.0
        return mean, std

    e2e   = ms([m.e2e_throughput_tps for m in iters])
    dec   = ms([m.decode_throughput_tps for m in iters])
    pre   = ms([m.prefill_throughput_tps for m in iters])
    ttft  = ms([m.ttft_ms for m in iters])
    itl   = ms([m.mean_itl_ms for m in iters])
    p95   = ms([m.p95_itl_ms for m in iters])
    e2e_l = ms([m.e2e_latency_ms for m in iters])

    m0 = iters[0]
    tput_mean = e2e[0]
    agg = AggMetrics(
        config={"batch_size": m0.batch_size, "input_len": m0.input_len, "output_len": m0.output_len},
        n_iters=len(iters),
        mode=mode,
        e2e_throughput_mean=tput_mean,   e2e_throughput_std=e2e[1],
        decode_throughput_mean=dec[0],   decode_throughput_std=dec[1],
        prefill_throughput_mean=pre[0],  prefill_throughput_std=pre[1],
        ttft_mean_ms=ttft[0],            ttft_std_ms=ttft[1],
        mean_itl_mean_ms=itl[0],         mean_itl_std_ms=itl[1],
        p95_itl_mean_ms=p95[0],          p95_itl_std_ms=p95[1],
        e2e_latency_mean_ms=e2e_l[0],    e2e_latency_std_ms=e2e_l[1],
        total_output_tokens=sum(m.total_output_tokens for m in iters),
        total_input_tokens=sum(m.total_input_tokens for m in iters),
        sere_output_throughput_tps=tput_mean,
        sere_tpot_ms=(1000.0 / tput_mean if tput_mean > 0 else 0.0),
        sere_requests_per_s=(tput_mean / output_len if output_len > 0 else 0.0),
    )
    return agg


def _print_summary(agg: AggMetrics) -> str:
    W = 68
    c = agg.config
    tpot = 1000.0 / agg.e2e_throughput_mean if agg.e2e_throughput_mean > 0 else float("inf")
    lines = [
        "=" * W,
        "  SERE Throughput Summary",
        "=" * W,
        f"  Mode   : {agg.mode}",
        f"  Config : batch={c['batch_size']}, input={c['input_len']} tok, output={c['output_len']} tok",
        f"  Iters  : {agg.n_iters}",
        "-" * W,
        "  PRIMARY METRICS",
        "-" * W,
    ]

    def row(label, mean, std, unit=""):
        return f"  {label:<38}  {mean:>9.2f} {unit}  ±{std:.2f}"

    lines += [
        row("Output throughput (tok/s)",    agg.e2e_throughput_mean,  agg.e2e_throughput_std,  "tok/s"),
        f"  {'TPOT (E2E, ms/tok)':<38}  {tpot:>9.3f} ms/tok",
        row("Requests/s",                   agg.sere_requests_per_s,  0.0,                     "req/s"),
        "-" * W,
        "  DETAILED METRICS",
        "-" * W,
        row("Decode throughput (tok/s)",    agg.decode_throughput_mean,  agg.decode_throughput_std,  "tok/s"),
        row("Prefill throughput (tok/s)",   agg.prefill_throughput_mean, agg.prefill_throughput_std, "tok/s"),
        row("TTFT (ms/req)",                agg.ttft_mean_ms,          agg.ttft_std_ms,          "ms"),
        row("Mean ITL / decode TPOT (ms/tok)", agg.mean_itl_mean_ms,  agg.mean_itl_std_ms,     "ms"),
        row("P95 ITL (ms/tok)",             agg.p95_itl_mean_ms,       agg.p95_itl_std_ms,      "ms"),
        row("E2E latency (ms/req)",         agg.e2e_latency_mean_ms,   agg.e2e_latency_std_ms,  "ms"),
        "=" * W,
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------

def _build_prompts(batch_size: int, input_len: int) -> list[dict]:
    rng = np.random.default_rng(seed=42)
    token_ids = rng.integers(100, 32000, size=(batch_size, input_len)).tolist()
    return [{"prompt_token_ids": ids} for ids in token_ids]


def _run_one_iter(llm, prompts, sampling_params, input_len, output_len) -> IterMetrics:
    records, orig = _wrap_engine_step(llm.llm_engine)
    try:
        t0 = time.perf_counter()
        outputs = llm.generate(prompts, sampling_params)
        wall_s = time.perf_counter() - t0
    finally:
        llm.llm_engine.step = orig

    total_in  = sum(len(o.prompt_token_ids) for o in outputs if o.prompt_token_ids)
    total_out = sum(len(o.outputs[0].token_ids) for o in outputs if o.outputs)
    if total_in == 0:
        total_in = len(prompts) * input_len

    return _compute_metrics(records, total_in, total_out,
                            len(prompts), input_len, output_len, wall_s)


def _run_batch_size(llm, batch_size: int, args, out_dir: Path, mode: str) -> AggMetrics:
    from vllm import SamplingParams

    prompts = _build_prompts(batch_size, args.input_len)
    sp = SamplingParams(
        temperature=args.temperature,
        top_p=1.0,
        ignore_eos=True,
        max_tokens=args.output_len,
    )

    print(f"  Warming up ({args.warmup_iters} iter)...", flush=True)
    for _ in range(args.warmup_iters):
        llm.generate(prompts, sp)

    iters: list[IterMetrics] = []
    for i in range(args.num_iters):
        print(f"  iter {i + 1}/{args.num_iters} ...", end=" ", flush=True)
        m = _run_one_iter(llm, prompts, sp, args.input_len, args.output_len)
        iters.append(m)
        tpot = 1000.0 / m.e2e_throughput_tps if m.e2e_throughput_tps > 0 else float("inf")
        print(
            f"output_tput={m.e2e_throughput_tps:.1f} tok/s  "
            f"TPOT={tpot:.2f} ms/tok  "
            f"ttft={m.ttft_ms:.0f} ms  "
            f"itl={m.mean_itl_ms:.2f} ms  "
            f"out_tok={m.total_output_tokens}"
        )

    agg = _aggregate(iters, mode, args.output_len)
    summary = _print_summary(agg)
    print("\n" + summary + "\n")

    bs_dir = out_dir / f"bs{batch_size}"
    bs_dir.mkdir(exist_ok=True)
    (bs_dir / "throughput.json").write_text(
        json.dumps({"aggregated": asdict(agg), "per_iter": [asdict(m) for m in iters]}, indent=2),
        encoding="utf-8",
    )
    (bs_dir / "summary.txt").write_text(summary, encoding="utf-8")
    return agg


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(
        description="SERE standalone throughput benchmark (vLLM 0.8.4).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--model", required=True,
                    help="Path to model or calibrated SERE model dir.")
    ap.add_argument("--sere", action="store_true",
                    help="Enable SERE routing (Qwen3MoeForCausalLMSERE architecture).")
    ap.add_argument("--select_top_k", type=int, default=2,
                    help="SERE: number of primary experts to retain.")
    ap.add_argument("--threshold", type=float, default=0.0,
                    help="SERE: similarity threshold for critical-expert preservation.")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--kv_cache_dtype", default="auto",
                    choices=["auto", "fp8", "fp8_e5m2", "fp8_e4m3"],
                    help="KV cache dtype. Use 'fp8' to halve KV memory and fit 262k context "
                         "on 2×H100 without changing max_model_len.")
    ap.add_argument("--tensor_parallel_size", type=int, default=1)
    ap.add_argument("--gpu_memory_utilization", type=float, default=0.90)
    ap.add_argument("--max_model_len", type=int, default=None)
    ap.add_argument("--enforce_eager", action="store_true", default=False,
                    help="Disable CUDA graph compilation. Eliminates 20+ min silent startup "
                         "when vLLM captures graphs for all 32 cudagraph_capture_sizes.")
    ap.add_argument("--input_len", type=int, default=512)
    ap.add_argument("--output_len", type=int, default=256)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--sweep_batch_sizes", default=None,
                    help="Comma-separated, e.g. '1,2,4,8,16,32'. Overrides --batch_size.")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--num_iters", type=int, default=3)
    ap.add_argument("--warmup_iters", type=int, default=1)
    ap.add_argument("--output_dir", default="outputs/throughput_benchmarks")
    args = ap.parse_args()

    from vllm import LLM

    hf_overrides = {}
    if args.sere:
        hf_overrides = {
            "architectures": ["Qwen3MoeForCausalLMSERE"],
            "select_top_k": args.select_top_k,
            "threshold": args.threshold,
        }

    mode = (f"SERE(top_k={args.select_top_k}, thr={args.threshold})"
            if args.sere else "baseline")

    vllm_kwargs: dict[str, Any] = dict(
        model=args.model,
        dtype=args.dtype,
        kv_cache_dtype=args.kv_cache_dtype,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        trust_remote_code=True,
    )
    if args.max_model_len is not None:
        vllm_kwargs["max_model_len"] = args.max_model_len
    if args.enforce_eager:
        vllm_kwargs["enforce_eager"] = True
    if hf_overrides:
        vllm_kwargs["hf_overrides"] = hf_overrides

    stamp = time.strftime("%Y%m%d_%H%M%S")
    safe_model = Path(args.model).name.replace("/", "_")
    run_name = f"{safe_model}_{mode.replace('/', '_')}_{stamp}"
    out_dir = Path(args.output_dir) / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nMode   : {mode}")
    print(f"Model  : {args.model}")
    if hf_overrides:
        print(f"hf_overrides: {hf_overrides}")
    llm = LLM(**vllm_kwargs)

    batch_sizes = (
        [int(x.strip()) for x in args.sweep_batch_sizes.split(",")]
        if args.sweep_batch_sizes else [args.batch_size]
    )

    all_agg: dict[int, AggMetrics] = {}
    for bs in batch_sizes:
        print(f"\n{'='*60}")
        print(f"Batch: {bs}  input_len={args.input_len}  output_len={args.output_len}  temp={args.temperature}")
        print("=" * 60)
        all_agg[bs] = _run_batch_size(llm, bs, args, out_dir, mode)

    combined = {
        "model": args.model,
        "mode": mode,
        "hf_overrides": hf_overrides,
        "args": vars(args),
        "sere_metric_note": (
            "sere_output_throughput_tps == 'output tokens/s'; "
            "sere_tpot_ms == 1000/output_throughput_tps"
        ),
        "results_by_batch_size": {str(bs): asdict(agg) for bs, agg in all_agg.items()},
    }
    (out_dir / "results.json").write_text(json.dumps(combined, indent=2), encoding="utf-8")

    print("\n" + "=" * 68)
    print("  FINAL TABLE  (compare directly with new-efficient-moe output)")
    print("=" * 68)
    print(f"  {'batch':>5}  {'output_tok/s':>13}  {'TPOT_e2e(ms)':>13}  "
          f"{'req/s':>8}  {'TTFT(ms)':>9}  {'ITL(ms)':>8}")
    print(f"  {'-'*5}  {'-'*13}  {'-'*13}  {'-'*8}  {'-'*9}  {'-'*8}")
    for bs, agg in all_agg.items():
        tpot = 1000.0 / agg.e2e_throughput_mean if agg.e2e_throughput_mean > 0 else float("inf")
        print(f"  {bs:>5}  {agg.e2e_throughput_mean:>13.1f}  {tpot:>13.3f}  "
              f"{agg.sere_requests_per_s:>8.2f}  {agg.ttft_mean_ms:>9.1f}  "
              f"{agg.mean_itl_mean_ms:>8.3f}")
    print("=" * 68)
    print(f"\nDone. Results saved to: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
