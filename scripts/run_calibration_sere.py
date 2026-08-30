#!/usr/bin/env python3
"""End-to-end SERE expert-similarity calibration."""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent


def _resolve_user_path(value: str) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    return (Path.cwd() / path).resolve()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Optionally prepare a C4/WikiText-style calibration parquet and run "
            "calibration/cal_expert_similarity.py."
        )
    )
    parser.add_argument("--model_type", required=True, choices=["qwen2_moe", "qwen3_moe", "deepseek_v2"])
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--data_path", default=None, help="Existing parquet with a `text` column.")
    parser.add_argument("--dataset", default="wikitext")
    parser.add_argument("--config", default="wikitext-2-raw-v1")
    parser.add_argument("--split", default="train")
    parser.add_argument("--text_column", default="text")
    parser.add_argument("--num_calibration_samples", type=int, default=2048)
    parser.add_argument(
        "--max_calibration_bytes",
        type=int,
        default=95_000_000,
        help="Selected calibration text byte budget forwarded to prepare_calibration_data.py.",
    )
    parser.add_argument(
        "--max_scan_samples",
        type=int,
        default=None,
        help="Maximum source rows to inspect while building calibration data.",
    )
    parser.add_argument("--min_chars", type=int, default=32)
    streaming = parser.add_mutually_exclusive_group()
    streaming.add_argument(
        "--streaming",
        action="store_true",
        dest="streaming",
        default=True,
        help="Use Hugging Face streaming mode while preparing calibration data. Default.",
    )
    streaming.add_argument(
        "--no_streaming",
        action="store_false",
        dest="streaming",
        help="Disable streaming for known-small datasets.",
    )
    parser.add_argument("--max_len", type=int, default=128)
    parser.add_argument("--similarity_method", default="frobenius", choices=["cka", "cosine", "frobenius"])
    parser.add_argument("--kernel", default="linear", choices=["linear", "rbf", "polynomial"])
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--trust_remote_code", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_path = _resolve_user_path(args.output_path)
    data_path = _resolve_user_path(args.data_path) if args.data_path else output_path / "calibration_data.parquet"

    if args.data_path is None:
        prep_cmd = [
            sys.executable,
            str(ROOT / "scripts" / "prepare_calibration_data.py"),
            "--dataset",
            args.dataset,
            "--config",
            args.config,
            "--split",
            args.split,
            "--text_column",
            args.text_column,
            "--output_path",
            str(data_path),
            "--model_path",
            args.model_path,
            "--max_samples",
            str(args.num_calibration_samples),
            "--max_text_bytes",
            str(args.max_calibration_bytes),
            "--min_chars",
            str(args.min_chars),
            "--min_tokens",
            str(args.max_len + 1),
        ]
        if args.max_scan_samples is not None:
            prep_cmd.extend(["--max_scan_samples", str(args.max_scan_samples)])
        if args.streaming:
            prep_cmd.append("--streaming")
        else:
            prep_cmd.append("--no_streaming")
        if args.trust_remote_code:
            prep_cmd.append("--trust_remote_code")
        subprocess.run(prep_cmd, cwd=ROOT, check=True)

    cal_cmd = [
        sys.executable,
        str(ROOT / "calibration" / "cal_expert_similarity.py"),
        "--model_type",
        args.model_type,
        "--model_path",
        args.model_path,
        "--output_path",
        str(output_path),
        "--data_path",
        str(data_path),
        "--max_len",
        str(args.max_len),
        "--similarity_method",
        args.similarity_method,
        "--kernel",
        args.kernel,
        "--batch_size",
        str(args.batch_size),
    ]
    subprocess.run(cal_cmd, cwd=ROOT / "calibration", check=True)
    print(f"Calibrated SERE model written to {args.output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
