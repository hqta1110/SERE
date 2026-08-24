# SPDX-License-Identifier: Apache-2.0
"""Compatibility hooks for SERE on modern vLLM MoE models.

vLLM 0.18.x builds MoE execution around native model classes plus
SharedFusedMoE/FusedMoE routers.  SERE only needs to replace the top-k router
ids, so this module patches native MoE block constructors to attach a SERE
custom routing function while leaving vLLM's TP/EP dispatch machinery intact.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from typing import Any

import torch
from torch import nn

from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.router.fused_topk_bias_router import (
    fused_topk_bias,
)
from vllm.model_executor.layers.fused_moe.router.fused_topk_router import fused_topk
from vllm.model_executor.layers.fused_moe.router.router_factory import (
    create_fused_moe_router,
)

logger = init_logger("vllm.SERE_vllm")

_SERE_ARCHS = {
    "Qwen2MoeForCausalLMSERE",
    "Qwen3MoeForCausalLMSERE",
    "DeepseekV2ForCausalLMSERE",
}

_PATCHED = False


def _is_sere_config(config: Any) -> bool:
    archs = getattr(config, "architectures", None) or []
    return any(arch in _SERE_ARCHS for arch in archs) or hasattr(
        config, "select_top_k"
    )


def _num_experts(config: Any) -> int:
    for attr in ("num_experts", "n_routed_experts"):
        value = getattr(config, attr, None)
        if value is not None:
            return int(value)
    raise AttributeError("SERE requires config.num_experts or config.n_routed_experts")


def _torch_topk(
    gating_output: torch.Tensor,
    topk: int,
    renormalize: bool,
    scoring_func: str,
    e_score_correction_bias: torch.Tensor | None,
    routed_scaling_factor: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    if scoring_func == "softmax":
        scores = gating_output.softmax(dim=-1)
    elif scoring_func == "sigmoid":
        scores = gating_output.sigmoid()
    else:
        raise ValueError(f"Unsupported SERE scoring function: {scoring_func}")

    scores_for_choice = scores
    if e_score_correction_bias is not None:
        scores_for_choice = scores_for_choice + e_score_correction_bias.unsqueeze(0)

    # ``sorted=True`` is REQUIRED, not cosmetic: SERE's primary set is
    # ``topk_ids[:, :select_top_k]``, i.e. the S highest-scoring experts. With
    # sorted=False torch.topk may return the k winners in any order, which makes
    # that slice an arbitrary S-subset of the top-k and silently changes the
    # policy. vLLM's CUDA ``fused_topk`` already returns score-descending order,
    # so this only fixes the CPU path -- but it makes the two paths agree.
    topk_ids = torch.topk(scores_for_choice, k=topk, dim=-1, sorted=True).indices
    topk_weights = scores.gather(1, topk_ids)
    if renormalize:
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
    if routed_scaling_factor != 1.0:
        topk_weights = topk_weights * routed_scaling_factor
    return topk_weights.to(torch.float32), topk_ids.to(torch.int32)


def _standard_topk(
    hidden_states: torch.Tensor,
    gating_output: torch.Tensor,
    topk: int,
    renormalize: bool,
    scoring_func: str,
    e_score_correction_bias: torch.Tensor | None,
    routed_scaling_factor: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    if hidden_states.is_cuda:
        if e_score_correction_bias is not None:
            topk_weights, topk_ids = fused_topk_bias(
                hidden_states=hidden_states,
                gating_output=gating_output,
                e_score_correction_bias=e_score_correction_bias,
                topk=topk,
                renormalize=renormalize,
                scoring_func=scoring_func,
                indices_type=torch.int32,
            )
        else:
            topk_weights, topk_ids, _ = fused_topk(
                hidden_states=hidden_states,
                gating_output=gating_output,
                topk=topk,
                renormalize=renormalize,
                indices_type=torch.int32,
                scoring_func=scoring_func,
            )
        if routed_scaling_factor != 1.0:
            topk_weights = topk_weights * routed_scaling_factor
        return topk_weights, topk_ids

    return _torch_topk(
        gating_output,
        topk,
        renormalize,
        scoring_func,
        e_score_correction_bias,
        routed_scaling_factor,
    )


def reroute_torch(
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    similarity_matrix: torch.Tensor,
    select_top_k: int,
    threshold: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    if topk_ids.ndim != 2:
        return topk_weights, topk_ids
    num_tokens, top_k = topk_ids.shape
    if num_tokens == 0 or select_top_k <= 0 or select_top_k >= top_k:
        return topk_weights, topk_ids

    num_experts = int(similarity_matrix.shape[0])
    topk_ids_i64 = topk_ids.to(torch.long)
    primary = topk_ids_i64[:, :select_top_k].reshape(-1)
    primary = primary[(primary >= 0) & (primary < num_experts)]
    if primary.numel() == 0:
        return topk_weights, topk_ids

    high_mask = torch.zeros(num_experts, dtype=torch.bool, device=topk_ids.device)
    high_mask[primary] = True
    high_experts = torch.nonzero(high_mask, as_tuple=False).flatten()

    reroute = topk_ids_i64[:, select_top_k:]
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
                best_expert = torch.where(best_sim < threshold, route_orig, best_expert)
            flat_new[route_positions] = best_expert

    patched = topk_ids_i64.clone()
    patched[:, select_top_k:] = flat_new.view_as(reroute)
    return topk_weights, patched.to(topk_ids.dtype)


# Set SERE_ALLOW_TORCH_REROUTE=1 to permit the eager fallback on GPU. Off by
# default because it is a BENCHMARK HAZARD, not merely slower: reroute_torch
# uses torch.nonzero (data-dependent output shape) and `.any()` host syncs, so
# it breaks CUDA-graph capture and serialises every step. A silent downgrade to
# that path makes SERE look far slower than it is -- the worst possible bias in
# a baseline -- and the original code announced it only via warning_once.
_ALLOW_TORCH_REROUTE = os.environ.get("SERE_ALLOW_TORCH_REROUTE", "0") == "1"

# Telemetry: proves the hook actually fires. vLLM V1 has several MoE execution
# paths and not all of them are guaranteed to honour custom_routing_function, so
# "the mask/remap is installed" and "the remap ran" are different claims.
_STATS = {"checked_similarity": False}
# Per-device int64 [calls, slots, changed]. Device-side so a captured graph
# replays the updates along with the kernel it is counting.
_STATS_DEV: dict = {}
_COUNT_REROUTE = os.environ.get("SERE_COUNT_REROUTE", "0") == "1"


def sere_stats() -> dict:
    """Reroute counters, read to host. Call only OUTSIDE CUDA-graph capture."""
    out = {"calls": 0, "slots": 0, "changed": 0}
    for acc in _STATS_DEV.values():
        c, s, ch = acc.tolist()
        out["calls"] += c
        out["slots"] += s
        out["changed"] += ch
    out["changed_frac"] = out["changed"] / out["slots"] if out["slots"] else 0.0
    return out


def _check_similarity_once(similarity_matrix: torch.Tensor) -> None:
    """Fail loudly if the similarity matrix never loaded.

    It is initialised to ``torch.eye``. If the checkpoint's
    ``similarity_matrix`` fails to load, an identity matrix makes every
    off-diagonal similarity 0, so with threshold>0 SERE degenerates to a no-op
    and with threshold=0 every secondary slot remaps to whichever primary sorts
    first -- both silent, and both invalidate the run.
    """
    if _STATS["checked_similarity"]:
        return
    try:
        with torch.no_grad():
            n = similarity_matrix.shape[0]
            off = similarity_matrix - torch.diag(torch.diagonal(similarity_matrix))
            off_mean = float(off.abs().sum() / max(n * (n - 1), 1))
    except RuntimeError:
        # Reading to host is illegal mid CUDA-graph capture. Defer to the next
        # eager call rather than aborting capture; in practice the first forward
        # is eager (profiling / warm-up) so the check lands there.
        return
    _STATS["checked_similarity"] = True
    if off_mean < 1e-6:
        raise RuntimeError(
            "SERE: similarity_matrix is (near-)identity -- the calibrated matrix "
            "did not load. Serve a calibrated checkpoint dir containing "
            "model.layers.N.mlp.similarity_matrix. Refusing to run, because this "
            "silently degenerates to no-op (threshold>0) or to remap-everything "
            "(threshold=0)."
        )
    logger.info("SERE similarity matrix loaded: mean |off-diagonal| = %.4f", off_mean)


def _reroute(
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    similarity_matrix: torch.Tensor,
    select_top_k: int,
    high_mask_cache: torch.Tensor,
    expert_mapping_cache: torch.Tensor,
    threshold: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    _check_similarity_once(similarity_matrix)
    if topk_ids.is_cuda and similarity_matrix.is_cuda:
        try:
            from SERE_vllm.rerouting_cuda_ops import rerouting_ops_cuda
        except Exception as exc:
            if not _ALLOW_TORCH_REROUTE:
                raise RuntimeError(
                    "SERE CUDA reroute kernel unavailable "
                    f"({type(exc).__name__}: {exc}). Build it for THIS "
                    "interpreter with 'python setup.py build_ext --inplace' in "
                    "the vllm/ dir. Refusing to fall back to the eager path, "
                    "which is CUDA-graph-unsafe and would understate SERE's "
                    "speed. Set SERE_ALLOW_TORCH_REROUTE=1 to override."
                ) from exc
            logger.warning_once("SERE CUDA reroute unavailable, using torch: %s", exc)
        else:
            weights, new_ids = rerouting_ops_cuda(
                topk_weights,
                topk_ids,
                similarity_matrix,
                select_top_k,
                high_mask_cache,
                expert_mapping_cache,
                threshold,
            )
            _record(topk_ids, new_ids)
            return weights, new_ids
    weights, new_ids = reroute_torch(
        topk_weights, topk_ids, similarity_matrix, select_top_k, threshold
    )
    _record(topk_ids, new_ids)
    return weights, new_ids


def _record(old_ids: torch.Tensor, new_ids: torch.Tensor) -> None:
    """Accumulate reroute activity **device-side only**.

    Every counter update must be a tensor op. An earlier version read the
    changed-slot count to host with ``int(...)`` inside this call, which aborted
    vLLM's CUDA-graph capture outright ("operation not permitted when stream is
    capturing"). Host-side Python counters are equally wrong in the opposite
    direction: under graph replay the Python never runs, so they silently
    undercount while the captured tensor ops keep executing.

    Read the counters with :func:`sere_stats`, and only from outside capture.
    """
    if not _COUNT_REROUTE:
        return
    dev = new_ids.device
    acc = _STATS_DEV.get(dev)
    if acc is None:
        acc = torch.zeros(3, dtype=torch.int64, device=dev)  # [calls, slots, changed]
        _STATS_DEV[dev] = acc
    acc[0] += 1
    acc[1] += old_ids.numel()
    acc[2] += (old_ids.reshape(-1) != new_ids.reshape(-1)).sum()


def _make_sere_routing_function(
    module: nn.Module,
    scoring_func: str,
    e_score_correction_bias: torch.Tensor | None,
    routed_scaling_factor: float,
) -> Callable[..., tuple[torch.Tensor, torch.Tensor]]:
    def rerouting_function(
        hidden_states: torch.Tensor,
        gating_output: torch.Tensor,
        topk: int,
        renormalize: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        topk_weights, topk_ids = _standard_topk(
            hidden_states,
            gating_output,
            topk,
            renormalize,
            scoring_func,
            e_score_correction_bias,
            routed_scaling_factor,
        )
        return _reroute(
            topk_weights,
            topk_ids,
            module.similarity_matrix,
            int(module.select_top_k),
            module._high_mask_cache,
            module._expert_mapping_cache,
            float(module.threshold),
        )

    return rerouting_function


def _rebuild_router(experts: nn.Module, custom_routing_function: Callable) -> None:
    experts.custom_routing_function = custom_routing_function
    experts.router = create_fused_moe_router(
        top_k=experts.top_k,
        global_num_experts=experts.global_num_experts,
        eplb_state=experts.eplb_state,
        renormalize=experts.renormalize,
        use_grouped_topk=False,
        custom_routing_function=custom_routing_function,
        enable_eplb=experts.enable_eplb,
        indices_type_getter=lambda: experts.quant_method.topk_indices_dtype,
    )
    experts.routing_method_type = experts.router.routing_method_type

    # Rebuilding the runner is MANDATORY, not defensive. vLLM 0.18.1 ends
    # FusedMoE.__init__ with
    #     self.runner = DefaultMoERunner(..., router=self.router, ...)
    # which captures the router BY REFERENCE. Since this patch runs after
    # __init__, assigning experts.router alone leaves the runner holding the
    # original router, the forward path never calls custom_routing_function, and
    # SERE silently degenerates to the unmodified baseline -- identical accuracy
    # AND identical speed, which reads as "the method does nothing" rather than
    # as a bug. vLLM's own _replace_quant_method rebuilds the runner for the
    # same reason.
    if hasattr(experts, "_init_runner"):
        experts.runner = experts._init_runner()


def _enable_sere_on_block(module: nn.Module, config: Any) -> None:
    if not _is_sere_config(config) or not hasattr(module, "experts"):
        return

    num_experts = _num_experts(config)
    module.select_top_k = int(getattr(config, "select_top_k", 1))
    module.threshold = float(getattr(config, "threshold", 0.0))
    module.register_parameter(
        "similarity_matrix",
        nn.Parameter(torch.eye(num_experts), requires_grad=False),
    )
    module.register_buffer("_high_mask_cache", torch.zeros(num_experts, dtype=torch.bool))
    module.register_buffer(
        "_expert_mapping_cache", torch.zeros(num_experts, dtype=torch.long)
    )

    scoring_func = getattr(config, "scoring_func", "softmax")
    routed_scaling_factor = float(getattr(module, "routed_scaling_factor", 1.0))
    gate = getattr(module, "gate", None)
    e_score_correction_bias = getattr(gate, "e_score_correction_bias", None)
    if isinstance(e_score_correction_bias, nn.Parameter):
        e_score_correction_bias = e_score_correction_bias.data

    custom_routing_function = _make_sere_routing_function(
        module,
        scoring_func=scoring_func,
        e_score_correction_bias=e_score_correction_bias,
        routed_scaling_factor=routed_scaling_factor,
    )
    _rebuild_router(module.experts, custom_routing_function)
    logger.info(
        "Enabled SERE routing for %s: select_top_k=%s threshold=%s",
        module.__class__.__name__,
        module.select_top_k,
        module.threshold,
    )


def _patch_block_init(block_cls: type[nn.Module]) -> None:
    if getattr(block_cls, "_sere_v1_patched", False):
        return
    original_init = block_cls.__init__

    def patched_init(self: nn.Module, *args: Any, **kwargs: Any) -> None:
        original_init(self, *args, **kwargs)
        vllm_config = kwargs.get("vllm_config")
        if vllm_config is None and args:
            vllm_config = args[0]
        model_config = getattr(vllm_config, "model_config", None)
        config = None
        if model_config is not None:
            config = getattr(model_config, "hf_text_config", None)
            if config is None:
                config = getattr(model_config, "hf_config", None)
        elif args:
            # DeepSeek's native vLLM MoE block receives the HF config directly.
            config = args[0]
        elif "config" in kwargs:
            config = kwargs["config"]
        if config is not None:
            _enable_sere_on_block(self, config)

    block_cls.__init__ = patched_init
    block_cls._sere_v1_patched = True


import os as _os
import re as _re

_Q35_SIM_CACHE: dict = {}


def _load_qwen3_5_similarity(layer_idx: int) -> "torch.Tensor | None":
    """Load the per-layer similarity matrix for Qwen3.5-MoE from the .pt named by
    $SERE_SIMILARITY_PT (disk-clean: we serve the base checkpoint and inject the
    calibrated matrices here instead of baking a 64GB copy)."""
    path = _os.environ.get("SERE_SIMILARITY_PT")
    if not path:
        return None
    if not _Q35_SIM_CACHE:
        _Q35_SIM_CACHE.update(torch.load(path, map_location="cpu"))
    return _Q35_SIM_CACHE.get(layer_idx)


def _enable_sere_on_qwen3_5_block(module: nn.Module, prefix: str) -> None:
    """Qwen3.5-MoE (Qwen3NextSparseMoeBlock) SERE enable: like _enable_sere_on_block
    but the similarity matrix is loaded from the calibration .pt by layer index
    (parsed from the module prefix, e.g. model.language_model.layers.7.mlp), and
    select_top_k/threshold come from env (SERE_SELECT_TOP_K / SERE_THRESHOLD)."""
    if not hasattr(module, "experts"):
        return
    m = _re.search(r"layers\.(\d+)\.", prefix or "")
    if m is None:
        return
    layer_idx = int(m.group(1))
    sim = _load_qwen3_5_similarity(layer_idx)
    if sim is None:
        logger.warning("SERE qwen3_5: no similarity for layer %s (SERE_SIMILARITY_PT set?)", layer_idx)
        return
    num_experts = sim.shape[0]
    module.select_top_k = int(_os.environ.get("SERE_SELECT_TOP_K", "2"))
    module.threshold = float(_os.environ.get("SERE_THRESHOLD", "0.0"))
    # Register as NON-PERSISTENT BUFFERS (not Parameters): we inject the calibrated
    # value here rather than loading from the checkpoint, and vLLM's weight-loader
    # completeness check flags uninitialized Parameters but ignores buffers. Place
    # them on the block's own device so the rerouting op's indexing doesn't hit a
    # CPU/GPU device mismatch.
    dev = next((p.device for p in module.parameters()), torch.device("cpu"))
    module.register_buffer(
        "similarity_matrix", sim.to(torch.float32).clone().to(dev), persistent=False,
    )
    module.register_buffer("_high_mask_cache", torch.zeros(num_experts, dtype=torch.bool, device=dev), persistent=False)
    module.register_buffer("_expert_mapping_cache", torch.zeros(num_experts, dtype=torch.long, device=dev), persistent=False)
    custom_routing_function = _make_sere_routing_function(
        module, scoring_func="softmax", e_score_correction_bias=None,
        routed_scaling_factor=1.0,
    )
    _rebuild_router(module.experts, custom_routing_function)
    logger.info(
        "Enabled SERE routing for Qwen3.5-MoE layer %s: select_top_k=%s threshold=%s",
        layer_idx, module.select_top_k, module.threshold,
    )


def _patch_qwen3_5_block_init(block_cls: type[nn.Module]) -> None:
    if getattr(block_cls, "_sere_v1_patched", False):
        return
    original_init = block_cls.__init__

    def patched_init(self: nn.Module, *args: Any, **kwargs: Any) -> None:
        original_init(self, *args, **kwargs)
        prefix = kwargs.get("prefix")
        if prefix is None and len(args) >= 2 and isinstance(args[1], str):
            prefix = args[1]
        _enable_sere_on_qwen3_5_block(self, prefix or "")

    block_cls.__init__ = patched_init
    block_cls._sere_v1_patched = True


def patch_vllm_v1_sere() -> None:
    global _PATCHED
    if _PATCHED:
        return

    from vllm.model_executor.models import deepseek_v2, qwen2_moe, qwen3_moe

    _patch_block_init(qwen2_moe.Qwen2MoeSparseMoeBlock)
    _patch_block_init(qwen3_moe.Qwen3MoeSparseMoeBlock)
    _patch_block_init(deepseek_v2.DeepseekV2MoE)
    # Qwen3.5-MoE uses qwen3_next's SparseMoeBlock; enable only when calibrated
    # matrices are provided (SERE_SIMILARITY_PT), so non-SERE qwen3_next serves
    # are unaffected.
    if _os.environ.get("SERE_SIMILARITY_PT"):
        try:
            from vllm.model_executor.models import qwen3_next
            _patch_qwen3_5_block_init(qwen3_next.Qwen3NextSparseMoeBlock)
        except Exception as exc:  # pragma: no cover
            logger.warning("SERE qwen3_5 block patch skipped: %r", exc)
    _PATCHED = True


V1_MODEL_REFS = {
    "Qwen2MoeForCausalLMSERE": (
        "vllm.model_executor.models.qwen2_moe:Qwen2MoeForCausalLM"
    ),
    "Qwen3MoeForCausalLMSERE": (
        "vllm.model_executor.models.qwen3_moe:Qwen3MoeForCausalLM"
    ),
    "DeepseekV2ForCausalLMSERE": (
        "vllm.model_executor.models.deepseek_v2:DeepseekV2ForCausalLM"
    ),
}
