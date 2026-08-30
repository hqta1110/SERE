#!/usr/bin/env python3
"""Evaluate a SERE-enabled vLLM model on common accuracy benchmarks.

The script is intentionally self-contained and uses vLLM directly with
``hf_overrides`` so the calibrated model is evaluated through the SERE model
class instead of a vanilla architecture.

Examples:
  VLLM_USE_V1=0 python scripts/eval_sere_benchmarks.py \
      --model ./calibration/output/qwen3_sere_wikitext \
      --model_type qwen3_moe \
      --benchmarks mmlu boolq truthfulqa humaneval lcb \
      --tensor_parallel_size 1 \
      --select_top_k 2 \
      --threshold 0.1

  # Smoke test without running full datasets.
  VLLM_USE_V1=0 python scripts/eval_sere_benchmarks.py \
      --model ./calibration/output/qwen2_moe_similarity \
      --model_type qwen2_moe \
      --benchmarks boolq truthfulqa \
      --limit 20
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable

os.environ.setdefault("VLLM_USE_V1", "0")

_ROOT = Path(__file__).resolve().parent.parent
_LOCAL_SERE_VLLM = _ROOT / "vllm"
if _LOCAL_SERE_VLLM.exists():
    sys.path.insert(0, str(_LOCAL_SERE_VLLM))


ARCHITECTURES = {
    "qwen2_moe": "Qwen2MoeForCausalLMSERE",
    "qwen3_moe": "Qwen3MoeForCausalLMSERE",
    "deepseek_v2": "DeepseekV2ForCausalLMSERE",
}


@dataclass(frozen=True)
class RunMeta:
    created_at: str
    model: str
    model_type: str
    architecture: str
    benchmarks: list[str]
    select_top_k: int
    threshold: float
    dtype: str
    tensor_parallel_size: int
    gpu_memory_utilization: float
    max_model_len: int | None
    limit: int | None
    output_dir: str
    cwd: str


def _stamp() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _batched(items: list[Any], batch_size: int) -> Iterable[list[Any]]:
    for i in range(0, len(items), batch_size):
        yield items[i : i + batch_size]


def _load_dataset(*args: Any, **kwargs: Any) -> Any:
    from datasets import load_dataset

    return load_dataset(*args, **kwargs)


def _letters(n: int) -> list[str]:
    alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    if n > len(alphabet):
        raise ValueError(f"Too many answer choices ({n}); max supported is {len(alphabet)}.")
    return list(alphabet[:n])


def _extract_choice(text: str, choices: list[str]) -> str | None:
    valid = {c.upper() for c in choices}
    upper = text.strip().upper()
    if not upper:
        return None
    m = re.search(r"\b([A-Z])\b", upper)
    if m and m.group(1) in valid:
        return m.group(1)
    first = upper[0]
    if first in valid:
        return first
    return None


def _extract_bool(text: str) -> bool | None:
    lower = text.strip().lower()
    if re.search(r"\btrue\b|\byes\b", lower):
        return True
    if re.search(r"\bfalse\b|\bno\b", lower):
        return False
    if lower.startswith("t"):
        return True
    if lower.startswith("f"):
        return False
    return None


def _generate_texts(llm: Any, prompts: list[str], *, max_tokens: int, batch_size: int) -> list[str]:
    from vllm import SamplingParams

    sampling = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=max_tokens)
    outputs: list[str] = []
    for batch in _batched(prompts, batch_size):
        for out in llm.generate(batch, sampling):
            outputs.append(out.outputs[0].text if out.outputs else "")
    return outputs


def _format_mc_prompt(question: str, choices: list[str], instruction: str) -> str:
    labels = _letters(len(choices))
    lines = [instruction, "", f"Question: {question.strip()}"]
    lines.extend(f"{label}. {choice}" for label, choice in zip(labels, choices))
    lines.append("")
    lines.append("Answer with only the letter.")
    lines.append("Answer:")
    return "\n".join(lines)


def run_mmlu(llm: Any, out_dir: Path, args: argparse.Namespace) -> dict[str, Any]:
    from datasets import get_dataset_config_names

    subjects = [s.strip() for s in args.mmlu_subjects.split(",") if s.strip()]
    if subjects == ["all"]:
        configs = get_dataset_config_names(args.mmlu_dataset)
        subjects = [c for c in configs if c != "all"]

    samples: list[dict[str, Any]] = []
    fewshot_by_subject: dict[str, list[dict[str, Any]]] = {}
    for subject in subjects:
        if args.mmlu_num_fewshot:
            dev = _load_dataset(args.mmlu_dataset, subject, split="dev")
            fewshot_by_subject[subject] = list(dev)[: args.mmlu_num_fewshot]
        ds = _load_dataset(args.mmlu_dataset, subject, split=args.mmlu_split)
        for row in ds:
            samples.append({"subject": subject, "row": row})
            if args.limit is not None and len(samples) >= args.limit:
                break
        if args.limit is not None and len(samples) >= args.limit:
            break

    prompts: list[str] = []
    golds: list[str] = []
    for sample in samples:
        subject = sample["subject"]
        row = sample["row"]
        prefix_parts: list[str] = []
        for ex in fewshot_by_subject.get(subject, []):
            ex_gold = _letters(len(ex["choices"]))[int(ex["answer"])]
            prefix_parts.append(
                _format_mc_prompt(
                    ex["question"],
                    list(ex["choices"]),
                    "Choose the correct answer.",
                )
                + f" {ex_gold}\n"
            )
        prompt = "".join(prefix_parts) + _format_mc_prompt(
            row["question"],
            list(row["choices"]),
            f"Choose the correct answer for this MMLU subject: {subject}.",
        )
        prompts.append(prompt)
        golds.append(_letters(len(row["choices"]))[int(row["answer"])])

    raw = _generate_texts(llm, prompts, max_tokens=8, batch_size=args.eval_batch_size)
    rows: list[dict[str, Any]] = []
    correct = 0
    by_subject: dict[str, dict[str, int]] = {}
    for sample, prompt, gold, pred_text in zip(samples, prompts, golds, raw):
        subject = sample["subject"]
        pred = _extract_choice(pred_text, _letters(len(sample["row"]["choices"])))
        is_correct = pred == gold
        correct += int(is_correct)
        stat = by_subject.setdefault(subject, {"correct": 0, "total": 0})
        stat["correct"] += int(is_correct)
        stat["total"] += 1
        rows.append(
            {
                "subject": subject,
                "question": sample["row"]["question"],
                "gold": gold,
                "prediction": pred,
                "raw_prediction": pred_text,
                "correct": is_correct,
                "prompt": prompt if args.save_prompts else None,
            }
        )

    for stat in by_subject.values():
        stat["accuracy"] = stat["correct"] / stat["total"] if stat["total"] else 0.0

    result = {
        "accuracy": correct / len(rows) if rows else 0.0,
        "correct": correct,
        "total": len(rows),
        "by_subject": by_subject,
    }
    samples_path = out_dir / "mmlu_samples.jsonl"
    with samples_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    result["samples_path"] = str(samples_path)
    return result


def run_boolq(llm: Any, out_dir: Path, args: argparse.Namespace) -> dict[str, Any]:
    ds = _load_dataset(args.boolq_dataset, split=args.boolq_split)
    rows = list(ds)
    if args.limit is not None:
        rows = rows[: args.limit]

    prompts = [
        (
            "Read the passage and answer the question with only True or False.\n\n"
            f"Passage: {r['passage']}\n\n"
            f"Question: {r['question']}\n"
            "Answer:"
        )
        for r in rows
    ]
    raw = _generate_texts(llm, prompts, max_tokens=8, batch_size=args.eval_batch_size)

    out_rows: list[dict[str, Any]] = []
    correct = 0
    for row, prompt, pred_text in zip(rows, prompts, raw):
        pred = _extract_bool(pred_text)
        gold = bool(row["answer"])
        is_correct = pred == gold
        correct += int(is_correct)
        out_rows.append(
            {
                "question": row["question"],
                "gold": gold,
                "prediction": pred,
                "raw_prediction": pred_text,
                "correct": is_correct,
                "prompt": prompt if args.save_prompts else None,
            }
        )

    samples_path = out_dir / "boolq_samples.jsonl"
    with samples_path.open("w", encoding="utf-8") as f:
        for row in out_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return {
        "accuracy": correct / len(out_rows) if out_rows else 0.0,
        "correct": correct,
        "total": len(out_rows),
        "samples_path": str(samples_path),
    }


def run_truthfulqa(llm: Any, out_dir: Path, args: argparse.Namespace) -> dict[str, Any]:
    ds = _load_dataset(args.truthfulqa_dataset, args.truthfulqa_config, split=args.truthfulqa_split)
    rows = list(ds)
    if args.limit is not None:
        rows = rows[: args.limit]

    prompts: list[str] = []
    golds: list[str] = []
    choices_by_row: list[list[str]] = []
    for row in rows:
        targets = row.get("mc1_targets") or row.get("mc2_targets")
        choices = list(targets["choices"])
        labels = list(targets["labels"])
        gold_idx = labels.index(1)
        prompts.append(
            _format_mc_prompt(
                row["question"],
                choices,
                "Choose the truthful answer.",
            )
        )
        golds.append(_letters(len(choices))[gold_idx])
        choices_by_row.append(choices)

    raw = _generate_texts(llm, prompts, max_tokens=8, batch_size=args.eval_batch_size)
    out_rows: list[dict[str, Any]] = []
    correct = 0
    for row, prompt, choices, gold, pred_text in zip(rows, prompts, choices_by_row, golds, raw):
        pred = _extract_choice(pred_text, _letters(len(choices)))
        is_correct = pred == gold
        correct += int(is_correct)
        out_rows.append(
            {
                "question": row["question"],
                "choices": choices,
                "gold": gold,
                "prediction": pred,
                "raw_prediction": pred_text,
                "correct": is_correct,
                "prompt": prompt if args.save_prompts else None,
            }
        )

    samples_path = out_dir / "truthfulqa_samples.jsonl"
    with samples_path.open("w", encoding="utf-8") as f:
        for row in out_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return {
        "accuracy": correct / len(out_rows) if out_rows else 0.0,
        "correct": correct,
        "total": len(out_rows),
        "samples_path": str(samples_path),
    }


HUMANEVAL_PREFIX = (
    "Complete the following Python function. "
    "Return ONLY valid Python code that continues the prompt.\n\n"
)


def _truncate_humaneval_completion(text: str) -> str:
    cut: int | None = None
    for marker in ["```", "Human:", "Assistant:", "\nif __name__ ==", "\n# Explanation", "\n## "]:
        idx = text.find(marker)
        if idx != -1:
            cut = idx if cut is None else min(cut, idx)
    if cut is not None:
        text = text[:cut]
    return text.rstrip() + "\n"


def _run_humaneval_checker(sample_path: Path, *, k: str, workers: int, timeout: float) -> dict[str, float]:
    from human_eval.data import HUMAN_EVAL, read_problems
    from human_eval.evaluation import evaluate_functional_correctness

    attempted: set[str] = set()
    with sample_path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                attempted.add(json.loads(line)["task_id"])

    problems = read_problems(HUMAN_EVAL)
    fd, problem_file = tempfile.mkstemp(prefix="humaneval_subset_", suffix=".jsonl")
    os.close(fd)
    try:
        with open(problem_file, "w", encoding="utf-8") as f:
            for task_id in sorted(attempted):
                row = dict(problems[task_id])
                row["task_id"] = task_id
                f.write(json.dumps(row) + "\n")
        result = evaluate_functional_correctness(
            str(sample_path),
            k=[int(x) for x in k.split(",") if x.strip()],
            n_workers=workers,
            timeout=timeout,
            problem_file=problem_file,
        )
        return {str(key): float(value) for key, value in dict(result).items()}
    finally:
        try:
            os.unlink(problem_file)
        except OSError:
            pass


def run_humaneval(llm: Any, out_dir: Path, args: argparse.Namespace) -> dict[str, Any]:
    ds = _load_dataset(args.humaneval_dataset, split=args.humaneval_split)
    rows = list(ds)
    if args.limit is not None:
        rows = rows[: args.limit]

    prompts = [HUMANEVAL_PREFIX + row["prompt"] for row in rows]
    raw = _generate_texts(
        llm,
        prompts,
        max_tokens=args.humaneval_max_tokens,
        batch_size=args.eval_batch_size,
    )

    samples_path = out_dir / "humaneval_samples.jsonl"
    with samples_path.open("w", encoding="utf-8") as f:
        for row, prompt, pred_text in zip(rows, prompts, raw):
            completion = pred_text
            if completion.startswith(prompt):
                completion = completion[len(prompt) :]
            completion = _truncate_humaneval_completion(completion)
            f.write(
                json.dumps(
                    {
                        "task_id": row["task_id"],
                        "completion": completion,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    result: dict[str, Any] = {
        "total": len(rows),
        "samples_path": str(samples_path),
        "pass_at_k": None,
    }
    if args.humaneval_run_eval:
        result["pass_at_k"] = _run_humaneval_checker(
            samples_path,
            k=args.humaneval_eval_k,
            workers=args.humaneval_eval_workers,
            timeout=args.humaneval_eval_timeout,
        )
    return result


def run_lcb(llm: Any, out_dir: Path, args: argparse.Namespace) -> dict[str, Any]:
    lcb_root = Path(args.lcb_root).resolve()
    if not lcb_root.exists():
        raise FileNotFoundError(
            f"LiveCodeBench checkout not found at {lcb_root}. "
            "Pass --lcb_root /path/to/LiveCodeBench."
        )
    sys.path.insert(0, str(lcb_root))

    old_cwd = os.getcwd()
    os.chdir(lcb_root)
    try:
        from lcb_runner.evaluation import extract_instance_results
        from lcb_runner.lm_styles import LMStyle
        from lcb_runner.runner.scenario_router import combine_results, get_metrics
        from lcb_runner.utils.scenarios import Scenario
        from lcb_runner.prompts import format_prompt_generation
        from lcb_runner.benchmarks import load_code_generation_dataset, load_code_generation_dataset_not_fast
        from vllm import SamplingParams

        if args.lcb_not_fast:
            benchmark = load_code_generation_dataset_not_fast(args.lcb_release_version)
        else:
            benchmark = load_code_generation_dataset(
                args.lcb_release_version,
                start_date=args.lcb_start_date,
                end_date=args.lcb_end_date,
            )
        benchmark = sorted(benchmark, key=lambda x: x.question_id)
        if args.limit is not None:
            benchmark = benchmark[: args.limit]

        prompts = [format_prompt_generation(problem, LMStyle.CodeQwenInstruct) for problem in benchmark]
        sampling = SamplingParams(
            n=args.lcb_n,
            temperature=0.0 if args.lcb_greedy else args.lcb_temperature,
            top_p=1.0 if args.lcb_greedy else args.lcb_top_p,
            max_tokens=args.lcb_max_tokens,
            stop=args.lcb_stop,
        )
        generated: list[list[str]] = []
        for batch in _batched(prompts, args.lcb_batch_size):
            for out in llm.generate(batch, sampling):
                generated.append([choice.text for choice in out.outputs])

        model = SimpleNamespace(model_style=LMStyle.CodeQwenInstruct)
        scenario = Scenario(args.lcb_scenario)
        combined = combine_results(scenario, generated, model)
        save_rows = [
            problem.insert_output(outputs, extracted)
            for problem, (outputs, extracted) in zip(benchmark, combined)
        ]
        output_path = out_dir / "lcb_output.json"
        _write_json(output_path, save_rows)

        result: dict[str, Any] = {
            "total": len(benchmark),
            "output_path": str(output_path),
            "metrics": None,
        }
        if args.lcb_run_eval:
            eval_args = SimpleNamespace(
                scenario=scenario,
                num_process_evaluate=args.lcb_num_process_evaluate,
                timeout=args.lcb_timeout,
            )
            metrics = get_metrics(scenario, eval_args, benchmark, combined)
            graded = extract_instance_results(metrics[1])
            eval_rows = [
                problem.insert_output_evaluation(outputs, extracted, graded_list)
                for problem, (outputs, extracted), graded_list in zip(benchmark, combined, graded)
            ]
            eval_path = out_dir / "lcb_eval_all.json"
            metrics_path = out_dir / "lcb_eval.json"
            _write_json(eval_path, eval_rows)
            _write_json(metrics_path, metrics)
            result["metrics"] = metrics[0]
            result["eval_path"] = str(eval_path)
            result["metrics_path"] = str(metrics_path)
        return result
    finally:
        os.chdir(old_cwd)


def build_llm(args: argparse.Namespace) -> Any:
    import SERE_vllm
    from vllm import LLM

    SERE_vllm.register()
    os.environ.setdefault("VLLM_DISABLE_PYNCCL", "1")
    if os.environ.get("NCCL_NET") == "gIB":
        os.environ.pop("NCCL_NET", None)
    if args.force_torch_sere_reroute:
        _patch_sere_rerouting_to_torch()
    if args.force_triton_moe_align:
        _patch_vllm_moe_align_block_size_to_triton()

    hf_overrides = {
        "architectures": [args.architecture],
        "select_top_k": int(args.select_top_k),
        "threshold": float(args.threshold),
    }
    llm_kwargs: dict[str, Any] = {
        "model": args.model,
        "tensor_parallel_size": args.tensor_parallel_size,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "dtype": args.dtype,
        "trust_remote_code": True,
        "hf_overrides": hf_overrides,
    }
    if args.enforce_eager:
        llm_kwargs["enforce_eager"] = True
    if args.disable_custom_all_reduce:
        llm_kwargs["disable_custom_all_reduce"] = True
    if args.max_model_len is not None:
        llm_kwargs["max_model_len"] = args.max_model_len
    return LLM(**llm_kwargs)


def _patch_sere_rerouting_to_torch() -> None:
    """Use a pure PyTorch SERE rerouting implementation.

    This is slower than the custom CUDA extension but avoids startup failures
    when ``SERE_vllm.rerouting_cuda_ops.rerouting_ops`` was compiled without a
    device-compatible cubin. It preserves the same routing semantics: keep the
    first ``select_top_k`` experts as the primary set, and reroute later expert
    slots to the most similar primary expert unless the similarity threshold
    says to preserve the original expert.
    """
    import importlib
    import sys

    import torch

    def rerouting_ops_torch(
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        similarity_matrix: torch.Tensor,
        select_top_k: int = 1,
        high_mask_cache: torch.Tensor | None = None,
        expert_mapping_cache: torch.Tensor | None = None,
        threshold: float = 0.0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del high_mask_cache, expert_mapping_cache

        if topk_ids.dtype is not torch.long:
            topk_ids = topk_ids.to(dtype=torch.long)

        if topk_ids.ndim != 2:
            return topk_weights, topk_ids
        num_tokens, top_k = topk_ids.shape
        if num_tokens == 0 or select_top_k <= 0 or select_top_k >= top_k:
            return topk_weights, topk_ids

        num_experts = int(similarity_matrix.shape[0])
        primary = topk_ids[:, :select_top_k].reshape(-1)
        primary = primary[(primary >= 0) & (primary < num_experts)]
        if primary.numel() == 0:
            return topk_weights, topk_ids

        high_mask = torch.zeros(num_experts, dtype=torch.bool, device=topk_ids.device)
        high_mask[primary] = True
        high_experts = torch.nonzero(high_mask, as_tuple=False).flatten()

        reroute = topk_ids[:, select_top_k:]
        flat_orig = reroute.reshape(-1)
        flat_new = flat_orig.clone()

        valid = (flat_orig >= 0) & (flat_orig < num_experts)
        flat_new[~valid] = 0
        if valid.any():
            valid_orig = flat_orig[valid]
            already_primary = high_mask[valid_orig]
            needs_route = ~already_primary
            if needs_route.any():
                route_positions = torch.nonzero(valid, as_tuple=False).flatten()[needs_route]
                route_orig = valid_orig[needs_route]
                sims = similarity_matrix[route_orig][:, high_experts]
                best_idx = sims.argmax(dim=1)
                best_sim = sims.gather(1, best_idx[:, None]).flatten()
                best_expert = high_experts[best_idx]
                if threshold > 0.0:
                    best_expert = torch.where(
                        best_sim < threshold,
                        route_orig,
                        best_expert,
                    )
                flat_new[route_positions] = best_expert

        patched_topk_ids = topk_ids.clone()
        patched_topk_ids[:, select_top_k:] = flat_new.view_as(reroute)
        return topk_weights, patched_topk_ids

    reroute_pkg = importlib.import_module("SERE_vllm.rerouting_cuda_ops")
    reroute_pkg.rerouting_ops_cuda = rerouting_ops_torch

    # If any model module has already imported the function by value, patch
    # that alias too. Usually this loop is not needed because this function runs
    # before vLLM imports the model class.
    for name in (
        "SERE_vllm.sere_qwen2_moe",
        "SERE_vllm.sere_qwen3_moe",
        "SERE_vllm.sere_deepseek_v2",
    ):
        mod = sys.modules.get(name)
        if mod is not None:
            setattr(mod, "rerouting_ops_cuda", rerouting_ops_torch)


def _patch_vllm_moe_align_block_size_to_triton() -> None:
    """Avoid vLLM's precompiled moe_align_block_size op on incompatible wheels.

    Some vLLM 0.8.4 CUDA wheels miss an A100-compatible image for
    ``ops.moe_align_block_size``. vLLM already ships a Triton implementation,
    but only uses it for large expert counts. Qwen2-MoE has 60 experts, so we
    patch both import sites to use the Triton implementation for all MoE sizes.
    """
    import importlib

    import torch
    import triton

    from vllm.utils import round_up

    fused_moe_mod = importlib.import_module(
        "vllm.model_executor.layers.fused_moe.fused_moe"
    )
    align_mod = importlib.import_module(
        "vllm.model_executor.layers.fused_moe.moe_align_block_size"
    )

    def moe_align_block_size_triton_all(
        topk_ids: torch.Tensor,
        block_size: int,
        num_experts: int,
        expert_map: torch.Tensor | None = None,
        pad_sorted_ids: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        max_num_tokens_padded = topk_ids.numel() + num_experts * (block_size - 1)
        if pad_sorted_ids:
            max_num_tokens_padded = round_up(max_num_tokens_padded, block_size)
        sorted_ids = torch.empty(
            (max_num_tokens_padded,), dtype=torch.int32, device=topk_ids.device
        )
        sorted_ids.fill_(topk_ids.numel())
        max_num_m_blocks = triton.cdiv(max_num_tokens_padded, block_size)
        expert_ids = torch.zeros(
            (max_num_m_blocks,), dtype=torch.int32, device=topk_ids.device
        )
        num_tokens_post_pad = torch.empty((1,), dtype=torch.int32, device=topk_ids.device)

        align_mod.moe_align_block_size_triton(
            topk_ids,
            num_experts,
            block_size,
            sorted_ids,
            expert_ids,
            num_tokens_post_pad,
        )
        if expert_map is not None:
            expert_ids = expert_map[expert_ids]
        return sorted_ids, expert_ids, num_tokens_post_pad

    align_mod.moe_align_block_size = moe_align_block_size_triton_all
    fused_moe_mod.moe_align_block_size = moe_align_block_size_triton_all


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="./calibration/output/qwen3_sere_wikitext")
    ap.add_argument("--model_type", choices=sorted(ARCHITECTURES), default="qwen3_moe")
    ap.add_argument("--architecture", default=None, help="Override SERE architecture class.")
    ap.add_argument(
        "--benchmarks",
        nargs="+",
        default=["mmlu", "boolq", "truthfulqa", "humaneval", "lcb"],
        choices=["mmlu", "boolq", "truthfulqa", "humaneval", "lcb"],
    )
    ap.add_argument("--select_top_k", type=int, default=2)
    ap.add_argument("--threshold", type=float, default=0.1)
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--tensor_parallel_size", type=int, default=1)
    ap.add_argument("--gpu_memory_utilization", type=float, default=0.9)
    ap.add_argument("--max_model_len", type=int, default=None)
    ap.add_argument("--enforce_eager", action="store_true")
    ap.add_argument("--disable_custom_all_reduce", action="store_true")
    ap.add_argument(
        "--no_force_torch_sere_reroute",
        dest="force_torch_sere_reroute",
        action="store_false",
        help="Use the SERE custom CUDA rerouting extension instead of the PyTorch fallback.",
    )
    ap.add_argument(
        "--no_force_triton_moe_align",
        dest="force_triton_moe_align",
        action="store_false",
        help="Use vLLM's precompiled moe_align_block_size op instead of the Triton fallback.",
    )
    ap.set_defaults(force_torch_sere_reroute=True, force_triton_moe_align=True)
    ap.add_argument("--eval_batch_size", type=int, default=32)
    ap.add_argument("--limit", type=int, default=None, help="Optional smoke-test cap per benchmark.")
    ap.add_argument("--output_dir", default="outputs/sere_eval_benchmarks")
    ap.add_argument("--save_prompts", action="store_true")

    ap.add_argument("--mmlu_dataset", default="cais/mmlu")
    ap.add_argument("--mmlu_subjects", default="all", help="'all' or comma-separated MMLU configs.")
    ap.add_argument("--mmlu_split", default="test")
    ap.add_argument("--mmlu_num_fewshot", type=int, default=5)

    ap.add_argument("--boolq_dataset", default="google/boolq")
    ap.add_argument("--boolq_split", default="validation")

    ap.add_argument("--truthfulqa_dataset", default="truthful_qa")
    ap.add_argument("--truthfulqa_config", default="multiple_choice")
    ap.add_argument("--truthfulqa_split", default="validation")

    ap.add_argument("--humaneval_dataset", default="openai_humaneval")
    ap.add_argument("--humaneval_split", default="test")
    ap.add_argument("--humaneval_max_tokens", type=int, default=512)
    ap.add_argument("--humaneval_run_eval", action="store_true")
    ap.add_argument("--humaneval_eval_k", default="1")
    ap.add_argument("--humaneval_eval_workers", type=int, default=4)
    ap.add_argument("--humaneval_eval_timeout", type=float, default=10.0)

    ap.add_argument("--lcb_root", default=str(_ROOT.parent / "new-efficient-moe" / "LiveCodeBench"))
    ap.add_argument("--lcb_scenario", default="codegeneration", choices=["codegeneration"])
    ap.add_argument("--lcb_release_version", default="release_latest")
    ap.add_argument("--lcb_n", type=int, default=1)
    ap.add_argument("--lcb_temperature", type=float, default=0.2)
    ap.add_argument("--lcb_top_p", type=float, default=0.95)
    ap.add_argument("--lcb_max_tokens", type=int, default=2000)
    ap.add_argument("--lcb_stop", default="###")
    ap.add_argument("--lcb_greedy", action="store_true")
    ap.add_argument("--lcb_batch_size", type=int, default=16)
    ap.add_argument("--lcb_run_eval", action="store_true")
    ap.add_argument("--lcb_num_process_evaluate", type=int, default=12)
    ap.add_argument("--lcb_timeout", type=int, default=6)
    ap.add_argument("--lcb_start_date", default=None)
    ap.add_argument("--lcb_end_date", default=None)
    ap.add_argument("--lcb_not_fast", action="store_true")

    args = ap.parse_args()
    args.architecture = args.architecture or ARCHITECTURES[args.model_type]
    return args


def main() -> int:
    args = parse_args()
    run_name = f"{_safe_name(Path(args.model).name or args.model)}_{_stamp()}"
    out_dir = Path(args.output_dir) / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    meta = RunMeta(
        created_at=_stamp(),
        model=args.model,
        model_type=args.model_type,
        architecture=args.architecture,
        benchmarks=list(args.benchmarks),
        select_top_k=int(args.select_top_k),
        threshold=float(args.threshold),
        dtype=args.dtype,
        tensor_parallel_size=int(args.tensor_parallel_size),
        gpu_memory_utilization=float(args.gpu_memory_utilization),
        max_model_len=args.max_model_len,
        limit=args.limit,
        output_dir=str(out_dir),
        cwd=os.getcwd(),
    )
    _write_json(out_dir / "meta.json", asdict(meta))

    llm = build_llm(args)
    runners = {
        "mmlu": run_mmlu,
        "boolq": run_boolq,
        "truthfulqa": run_truthfulqa,
        "humaneval": run_humaneval,
        "lcb": run_lcb,
    }
    results: dict[str, Any] = {}
    for benchmark in args.benchmarks:
        print(f"\n=== Running {benchmark} ===", flush=True)
        started = time.perf_counter()
        results[benchmark] = runners[benchmark](llm, out_dir, args)
        results[benchmark]["elapsed_seconds"] = time.perf_counter() - started
        _write_json(out_dir / "results.json", results)
        print(json.dumps(results[benchmark], indent=2), flush=True)

    print(f"\nDone. Outputs in: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
