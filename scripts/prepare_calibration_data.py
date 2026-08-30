#!/usr/bin/env python3
"""Build a SERE calibration parquet from a Hugging Face text dataset."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Iterable


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create a parquet file with a `text` column for "
            "calibration/cal_expert_similarity.py."
        )
    )
    parser.add_argument(
        "--dataset",
        default="wikitext",
        help="Dataset name/path. Examples: wikitext, allenai/c4, c4.",
    )
    parser.add_argument(
        "--config",
        default="wikitext-2-raw-v1",
        help="Dataset config/subset. Examples: wikitext-103-raw-v1, en.",
    )
    parser.add_argument("--split", default="train")
    parser.add_argument("--text_column", default="text")
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--model_path", default=None, help="Optional tokenizer for token-length filtering.")
    parser.add_argument("--max_samples", type=int, default=2048)
    parser.add_argument(
        "--max_text_bytes",
        type=int,
        default=95_000_000,
        help="Stop adding examples before selected UTF-8 text exceeds this budget. Default: 95 MB.",
    )
    parser.add_argument(
        "--max_scan_samples",
        type=int,
        default=None,
        help="Maximum source rows to inspect. Default: max_samples * 20.",
    )
    parser.add_argument("--min_chars", type=int, default=32)
    parser.add_argument(
        "--min_tokens",
        type=int,
        default=None,
        help="Keep only examples with at least this many tokenizer tokens.",
    )
    streaming = parser.add_mutually_exclusive_group()
    streaming.add_argument(
        "--streaming",
        action="store_true",
        dest="streaming",
        default=True,
        help="Use Hugging Face streaming mode. Default and recommended for large corpora.",
    )
    streaming.add_argument(
        "--no_streaming",
        action="store_false",
        dest="streaming",
        help="Disable streaming for known-small local or Hugging Face datasets.",
    )
    parser.add_argument("--trust_remote_code", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def _iter_rows(dataset: Any) -> Iterable[dict[str, Any]]:
    if hasattr(dataset, "__iter__"):
        return iter(dataset)
    return (dataset[i] for i in range(len(dataset)))


def main() -> int:
    args = parse_args()

    import pandas as pd
    from datasets import load_dataset
    from transformers import AutoTokenizer

    tokenizer = None
    if args.min_tokens is not None:
        if not args.model_path:
            raise ValueError("--model_path is required when --min_tokens is set.")
        tokenizer = AutoTokenizer.from_pretrained(
            args.model_path,
            trust_remote_code=args.trust_remote_code,
        )

    load_kwargs: dict[str, Any] = {"split": args.split, "streaming": args.streaming}
    if args.config:
        ds = load_dataset(args.dataset, args.config, **load_kwargs)
    else:
        ds = load_dataset(args.dataset, **load_kwargs)

    if not args.streaming:
        ds = ds.shuffle(seed=args.seed)

    max_scan_samples = args.max_scan_samples or max(args.max_samples * 20, args.max_samples)
    if args.max_samples <= 0:
        raise ValueError("--max_samples must be > 0.")
    if args.max_text_bytes <= 0:
        raise ValueError("--max_text_bytes must be > 0.")
    if max_scan_samples <= 0:
        raise ValueError("--max_scan_samples must be > 0.")

    texts: list[str] = []
    selected_text_bytes = 0
    scanned = 0
    for row in _iter_rows(ds):
        scanned += 1
        if scanned > max_scan_samples:
            break
        text = str(row.get(args.text_column, "")).strip()
        if len(text) < args.min_chars:
            continue
        if tokenizer is not None and len(tokenizer.encode(text, add_special_tokens=True)) < args.min_tokens:
            continue
        text_bytes = len(text.encode("utf-8"))
        if selected_text_bytes + text_bytes > args.max_text_bytes:
            if texts:
                break
            raise ValueError(
                "The first matching text exceeds --max_text_bytes; increase the budget or add tighter filters."
            )
        texts.append(text)
        selected_text_bytes += text_bytes
        if len(texts) >= args.max_samples:
            break

    if not texts:
        raise ValueError("No calibration texts matched the requested filters.")

    out = Path(args.output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"text": texts}).to_parquet(out, index=False)
    parquet_bytes = out.stat().st_size
    print(
        f"Wrote {len(texts)} calibration samples to {out} "
        f"(selected_text={selected_text_bytes / 1_000_000:.2f} MB, "
        f"parquet={parquet_bytes / 1_000_000:.2f} MB, scanned={scanned})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
