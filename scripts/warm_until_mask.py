#!/usr/bin/env python3
"""Send held-out, same-domain prompts until the skip mask is actually installed.

Why this exists: without it the eval itself does the profiling, so the mask
arrives partway through scoring and part of the benchmark measures the
*unmodified* model. Measured on a previous run: the mask finalized ~40 s into a
~102 s GSM8K run. That inflates ARES's accuracy, and the bias is one-sided --
SERE loads a static similarity matrix and is active from the first token, so it
never gets the same free head start.

Warm data is HELD OUT from the scored set on purpose:
  * GSM8K test  <- warmed on GSM8K *train*
  * HumanEval   <- warmed on MBPP (same domain, disjoint problems)
Routing statistics are not labels, so reusing the test items would not leak
answers -- but it invites an objection for no benefit.

Exits 0 once the mask is confirmed installed, 3 if it never appears.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
from pathlib import Path

GSM8K_TRAIN = ("/home/PC/.cache/huggingface/hub/datasets--openai--gsm8k/"
               "snapshots/740312add88f781978c0658806c59bc2815b9866/main/"
               "train-00000-of-00001.parquet")
MBPP = "/home/PC/.cache/opencompass/data/mbpp/mbpp.jsonl"


def load_prompts(source: str, limit: int) -> list[str]:
    out: list[str] = []
    if source == "gsm8k_train":
        import pandas as pd
        df = pd.read_parquet(GSM8K_TRAIN)
        for q in df["question"].tolist():
            out.append(f"Question: {q}\nAnswer:")
    elif source == "mbpp":
        # MBPP's ``text`` alone is ~16 tokens, and the profiling budget only
        # advances on PREFILL (decode-only steps deliberately do not burn it), so
        # short prompts starve the finalize gate: an earlier version sent 12,415
        # prompt tokens across 6 rounds and never reached the 4096-token
        # threshold. Append the reference solution and tests to reach a
        # HumanEval-like prompt length, which is also closer to the traffic being
        # scored. Still disjoint from HumanEval's problems.
        with open(MBPP) as f:
            for line in f:
                try:
                    d = json.loads(line)
                except Exception:
                    continue
                t = d.get("text") or d.get("prompt")
                if not t:
                    continue
                parts = [str(t)]
                if d.get("code"):
                    parts.append(str(d["code"]))
                tests = d.get("test_list") or []
                if tests:
                    parts.append("\n".join(str(x) for x in tests))
                out.append("\n".join(parts) + "\n")
    else:
        raise SystemExit(f"unknown warm source: {source}")
    if not out:
        raise SystemExit(f"no prompts loaded from {source}")
    return out[:limit]


def mask_installed(skip_log: str) -> bool:
    """True once the mask is installed AND actively skipping.

    Two independent signals, because the first is method-specific:
      1. ``[contrib_mass] layer=`` / ``[per_slot_contrib] layer=`` -- one line per
         masked layer at finalize. The magnitude-only methods (percentile /
         dynamic / std / absolute) log nothing at finalize, so this alone would
         silently fall through to a token budget for them.
      2. ``risk=`` -- the tier-0 monitor only reports during the SKIPPING phase,
         so its presence is positive evidence that a mask is installed and being
         applied. This works for every threshold method, which makes it the
         stronger check.
    """
    if not skip_log or not os.path.exists(skip_log):
        return False
    try:
        text = Path(skip_log).read_text(errors="ignore")
    except OSError:
        return False
    # Strongest and method-agnostic: the controller logs this once the initial
    # mask is installed and skip is enabled, whatever the threshold method.
    if "initial BUILD complete" in text:
        return True
    if re.search(r"\[(contrib_mass|per_slot_contrib)\] layer=", text):
        return True
    return bool(re.search(r"\brisk=[0-9.]", text))


async def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--port", type=int, required=True)
    p.add_argument("--model", default="m")
    p.add_argument("--warm-source", required=True, choices=("gsm8k_train", "mbpp"))
    p.add_argument("--skip-log", default="")
    # Short generations on purpose: the profiling budget advances on prefill, so
    # long decodes cost wall-clock without moving the gate.
    p.add_argument("--max-tokens", type=int, default=16)
    p.add_argument("--concurrency", type=int, default=32)
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--rounds", type=int, default=10)
    # Fallback for methods that do not log a finalize: MIN_TOKENS_FOR_FINALIZE is
    # 4096 prefill tokens, so one batch already clears it with margin.
    p.add_argument("--min-tokens", type=int, default=20000)
    args = p.parse_args()

    import httpx

    prompts = load_prompts(args.warm_source, args.batch * args.rounds)
    url = f"http://127.0.0.1:{args.port}/v1/completions"
    sent = 0
    print(f"[warm] source={args.warm_source} pool={len(prompts)} "
          f"target: mask installed or >={args.min_tokens} prompt tokens")

    async with httpx.AsyncClient(timeout=600) as client:
        sem = asyncio.Semaphore(args.concurrency)

        async def one(pr: str) -> int:
            async with sem:
                try:
                    r = await client.post(url, json={
                        "model": args.model, "prompt": pr,
                        "max_tokens": args.max_tokens, "temperature": 0.0})
                    return int((r.json().get("usage") or {}).get("prompt_tokens", 0))
                except Exception:
                    return 0

        for rnd in range(args.rounds):
            chunk = prompts[rnd * args.batch:(rnd + 1) * args.batch]
            if not chunk:
                break
            sent += sum(await asyncio.gather(*(one(x) for x in chunk)))
            ok = mask_installed(args.skip_log)
            print(f"[warm] round {rnd+1}: prompt_tokens={sent} mask_installed={ok}")
            if ok:
                print("[warm] mask confirmed installed -- safe to score")
                return 0
            if not args.skip_log and sent >= args.min_tokens:
                print("[warm] no skip log (baseline arm); token budget met")
                return 0
            if sent >= args.min_tokens and rnd + 1 >= 2:
                # Method logs no finalize (percentile et al). Budget is the proxy.
                print("[warm] token budget met; method logs no finalize line")
                return 0

    print(f"[warm] FAILED: mask not confirmed after {sent} prompt tokens",
          file=sys.stderr)
    return 3


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
