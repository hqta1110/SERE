"""SERE calibration adapter for Qwen3.5-MoE (qwen3_5_moe).

Unlike qwen2/qwen3-moe (ModuleList experts), Qwen3.5-MoE stores experts fused in a
single ``Qwen3_5MoeExperts`` module (stacked gate_up_proj/down_proj). This module
monkeypatches the transformers ``Qwen3_5MoeSparseMoeBlock`` to add SERE's
similarity-computation hooks, computing each expert's UNWEIGHTED output on all
tokens by batched matmul over the stacked weights (faithful to the model's own
per-expert math) and feeding them to ``compute_similarity_matrix_gpu``.

The model itself is the FULL multimodal ``Qwen3_5MoeForConditionalGeneration`` (the
text tower lives at ``model.model.language_model.layers``); the checkpoint has no
text-only variant and loading the whole thing avoids key-remapping. Exported name
``Qwen3_5MoeForCausalLM`` is what cal_expert_similarity.py imports.
"""
import torch
from transformers import Qwen3_5MoeForConditionalGeneration as _Q35Cond
from transformers.models.qwen3_5_moe import modeling_qwen3_5_moe as _M

from utils import compute_similarity_matrix_gpu

_Block = _M.Qwen3_5MoeSparseMoeBlock


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
    Chunked over tokens to bound memory (256 experts * T * H can be large)."""
    exp = block.experts
    W_gu, W_dn = exp.gate_up_proj, exp.down_proj          # (E,2I,H),(E,H,I)
    x = x.to(W_gu.dtype).to(W_gu.device)
    E = W_gu.shape[0]
    outs = []
    # keep each chunk's (E, chunk, H) transient modest
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
Qwen3_5MoeForCausalLM = _Q35Cond
