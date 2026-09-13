"""SERE calibration adapter for GLM-4.7-Flash (glm4_moe_lite).

Like Qwen3.5-MoE (and UNLIKE qwen2/qwen3-moe's ModuleList), GLM-4.7-Flash-Lite
stores experts fused in a single ``Glm4MoeLiteExperts`` module: stacked
``gate_up_proj`` (E, 2I, H) and ``down_proj`` (E, H, I). So the per-expert
unweighted-output math is identical to the qwen3_5 adapter (batched einsum over the
stacked weights) — this file is that adapter with GLM class names.

Requires a transformers that ships ``glm4_moe_lite`` (>=5.16-dev; the pinned serve
transformers 4.57.6 does NOT have it, so calibration runs in the reap/.venv_q36
env). The model is the text-only ``Glm4MoeLiteForCausalLM`` (decoder layers at
``model.model.layers``; dense layer 0 = ``first_k_dense_replace`` has no MoE block
and is skipped by the ``hasattr(mlp, 'enable_similarity_computation')`` guard).
Exported name ``Glm4MoeLiteForCausalLM`` is what cal_expert_similarity.py imports.
"""
import torch
from transformers import Glm4MoeLiteForCausalLM as _GlmLite
from transformers.models.glm4_moe_lite import modeling_glm4_moe_lite as _M

from utils import compute_similarity_matrix_gpu

_Block = _M.Glm4MoeLiteMoE


def enable_similarity_computation(self, method="frobenius", kernel="linear"):
    self.compute_similarity = True
    self.similarity_method = method
    self.kernel = kernel


def disable_similarity_computation(self):
    self.compute_similarity = False


def set_similarity_matrix(self, m):
    self.similarity_matrix = (m.clone().detach() if torch.is_tensor(m)
                              else torch.tensor(m, dtype=torch.float32))


def get_similarity_matrix(self):
    return getattr(self, "similarity_matrix", None)


def reset_similarity_matrix(self):
    self.similarity_matrix = None
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


@torch.no_grad()
def _all_expert_outputs(block, x):
    """UNWEIGHTED output of every expert on every token: (E, T, H).
    Chunked over tokens to bound the transient (E, chunk, .) tensors."""
    exp = block.experts
    W_gu, W_dn = exp.gate_up_proj, exp.down_proj          # (E,2I,H),(E,H,I)
    x = x.to(W_gu.dtype).to(W_gu.device)
    E = W_gu.shape[0]
    outs = []
    chunk = max(1, 4096 // max(1, E // 32))
    for s in range(0, x.shape[0], chunk):
        xc = x[s:s + chunk]                               # (c, H)
        gu = torch.einsum("th,eoh->eto", xc, W_gu)        # (E, c, 2I)
        g, u = gu.chunk(2, dim=-1)
        h = exp.act_fn(g) * u                             # (E, c, I)
        outs.append(torch.einsum("eti,ehi->eth", h, W_dn))  # (E, c, H)
    return torch.cat(outs, dim=1)                          # (E, T, H)


_orig_forward = _Block.forward


def _sim_forward(self, hidden_states):
    if getattr(self, "compute_similarity", False):
        x = hidden_states.view(-1, hidden_states.shape[-1])
        acts = _all_expert_outputs(self, x)               # (E, T, H)
        expert_outputs = [acts[e] for e in range(acts.shape[0])]
        sim = compute_similarity_matrix_gpu(
            expert_outputs, method=self.similarity_method,
            kernel=getattr(self, "kernel", "linear"), device=x.device)
        self.set_similarity_matrix(sim.detach().cpu())
        del acts, expert_outputs, sim
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return _orig_forward(self, hidden_states)


for _name, _fn in [
    ("enable_similarity_computation", enable_similarity_computation),
    ("disable_similarity_computation", disable_similarity_computation),
    ("set_similarity_matrix", set_similarity_matrix),
    ("get_similarity_matrix", get_similarity_matrix),
    ("reset_similarity_matrix", reset_similarity_matrix),
    ("forward", _sim_forward),
]:
    setattr(_Block, _name, _fn)


# cal_expert_similarity.py does ``model_class.from_pretrained(...)``.
Glm4MoeLiteForCausalLM = _GlmLite
