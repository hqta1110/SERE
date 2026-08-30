"""
High-performance CUDA kernels for SERE in MoE models.
"""

import torch
from typing import Tuple

from . import rerouting_ops

_EMPTY = torch.Tensor()   # single sentinel to avoid creating new empty tensors


def _high_mask_for(topk_ids: torch.Tensor, similarity_matrix: torch.Tensor) -> torch.Tensor:
    num_experts = int(similarity_matrix.shape[0])
    return torch.empty(num_experts, dtype=torch.bool, device=topk_ids.device)


# ---------------------------------------------------------------------------
# Register as a proper torch.library custom op so that torch.compile / vLLM
# V1 CUDA graphs can capture our kernel:
#   - During CUDA graph capture: dynamo calls register_fake (abstract) impl
#     for shape inference and records the kernel launch in the graph.
#   - During CUDA graph replay: the graph is replayed without any Python.
#     → per-step overhead drops to ~0 ms.
#   - During eager execution (V0 / no graph): is_compiling() == False, so
#     we bypass the dispatcher entirely with a direct C++ call (~0.005 ms).
# ---------------------------------------------------------------------------

@torch.library.custom_op("sere_ops::fused_reroute", mutates_args={"topk_ids"})
def _fused_reroute_op(
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    similarity_matrix: torch.Tensor,
    select_top_k: int,
    threshold: float,
) -> None:
    """CUDA impl — only runs during CUDA graph capture (warmup), never during replay."""
    high_mask = _high_mask_for(topk_ids, similarity_matrix)
    rerouting_ops.fused_reroute(
        topk_weights, topk_ids, similarity_matrix,
        select_top_k, high_mask, _EMPTY, threshold,
    )


@_fused_reroute_op.register_fake
def _fused_reroute_fake(
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    similarity_matrix: torch.Tensor,
    select_top_k: int,
    threshold: float,
) -> None:
    """Abstract impl: tells the compiler topk_ids is mutated, returns nothing."""
    pass


def rerouting_ops_cuda(
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    similarity_matrix: torch.Tensor,
    select_top_k: int = 1,
    high_mask_cache: torch.Tensor = None,
    expert_mapping_cache: torch.Tensor = None,
    threshold: float = 0.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Fused SERE re-routing.

    Two code paths:
      • torch.compile / CUDA graph capture (is_compiling == True):
            uses the registered custom op so the kernel is captured in the
            CUDA graph; subsequent replays have zero Python overhead.
      • Eager / V0 mode (is_compiling == False):
            calls the C++ extension directly, bypassing the torch.library
            dispatcher to keep per-call overhead at ~0.005 ms.

    For num_tokens <= 256 (all practical decode batches) the C++ code uses
    a single fused kernel (clear + mark + reroute).  Larger prefill batches
    fall back to the original 3-op path automatically.
    """
    if topk_ids.dtype is not torch.long:
        topk_ids = topk_ids.to(dtype=torch.long)

    if torch.compiler.is_compiling():
        # Inside torch.compile — use the registered op so the kernel is
        # captured in the CUDA graph.
        torch.ops.sere_ops.fused_reroute(
            topk_weights, topk_ids, similarity_matrix, select_top_k, threshold,
        )
    else:
        # Eager mode — direct C++ call, no dispatcher overhead.
        hm = (
            high_mask_cache
            if high_mask_cache is not None
            else _high_mask_for(topk_ids, similarity_matrix)
        )
        em = expert_mapping_cache if expert_mapping_cache is not None else _EMPTY
        rerouting_ops.fused_reroute(
            topk_weights, topk_ids, similarity_matrix,
            select_top_k, hm, em, threshold,
        )

    return topk_weights, topk_ids
