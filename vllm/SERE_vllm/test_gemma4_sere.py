"""CPU self-check for the gemma4 SERE routing wrapper.

The one thing worth testing here is the per_expert_scale A->B fold: if SERE
reroutes a slot from expert A to expert B, the slot must carry B's scale. Doing
it wrong produces no crash and no NaN -- just a quietly wrong model.

Read this before loosening any assertion below. An earlier version of this file
PASSED while the routing wrapper was wrong: it used allclose(atol=1e-6) on
float32 inputs, and the bug was a scale fold performed one cast too late, which
is invisible at both that tolerance and that dtype. The equivalence gate then
failed on 58/60 aime rows and cost a full GPU run to diagnose. A self-check that
cannot fail on the bug it exists to catch is worse than no self-check: it
manufactures confidence. Hence torch.equal, not allclose, and bfloat16, not
float32 -- the assertions are tight and the dtype is adversarial on purpose.

Run: python test_gemma4_sere.py
"""
import torch
from torch import nn

from vllm.model_executor.models.gemma4 import gemma4_routing_function_torch
from SERE_vllm.vllm_v1_patch import _make_gemma4_sere_routing_function

E, TOPK, T = 8, 4, 3  # few tokens: with select_top_k=1 the primary union must not cover all E


def _module(select_top_k):
    m = nn.Module()
    m.per_expert_scale = nn.Parameter(
        (torch.rand(E) + 0.5).to(torch.bfloat16), requires_grad=False)
    # Deliberately NOT identity: _check_similarity_once refuses identity.
    sim = torch.rand(E, E)
    sim = (sim + sim.T) / 2
    sim.fill_diagonal_(1.0)
    m.register_buffer("similarity_matrix", sim)
    m.register_buffer("_high_mask_cache", torch.zeros(E, dtype=torch.bool))
    m.register_buffer("_expert_mapping_cache", torch.zeros(E, dtype=torch.long))
    m.select_top_k = select_top_k
    m.threshold = 0.0
    return m


def main():
    torch.manual_seed(42)
    # bfloat16: the equivalence gate lives or dies on the scale fold happening in
    # the GATING dtype, and in float32 a wrong fold order is invisible.
    gating = torch.randn(T, E, dtype=torch.bfloat16)

    # 1. No-op reroute (select_top_k == topk) must be BIT-identical to vLLM's own
    #    routing function with the real scale -- allclose is not enough: an arm
    #    that is merely close diverges from the baseline at the first token under
    #    greedy decoding, and the threshold gate can never pass.
    m = _module(TOPK)
    w, ids = _make_gemma4_sere_routing_function(m)(None, gating, TOPK, True)
    ref_w, ref_ids = gemma4_routing_function_torch(gating, TOPK, m.per_expert_scale)
    assert torch.equal(ids, ref_ids), "no-op reroute changed ids"
    assert torch.equal(w, ref_w), f"not bit-identical: max {(w - ref_w).abs().max()}"

    # 2. Real reroute: every slot carries its NEW expert's scale.
    m = _module(1)
    w, ids = _make_gemma4_sere_routing_function(m)(None, gating, TOPK, True)
    unscaled, old_ids = gemma4_routing_function_torch(
        gating, TOPK, torch.ones(E, dtype=gating.dtype))
    changed = (ids != old_ids)
    assert changed.any(), "test is vacuous: nothing rerouted"
    scale = m.per_expert_scale
    fold = lambda i: (unscaled.to(gating.dtype) * scale[i.long()]).to(torch.float32)
    assert torch.equal(w, fold(ids))
    # and the A-scale bug would actually be caught by that assertion:
    assert not torch.equal(w, fold(old_ids)), "scale fold is unobservable"

    print(f"ok: {int(changed.sum())}/{changed.numel()} slots rerouted, scale follows new id")


if __name__ == "__main__":
    main()
