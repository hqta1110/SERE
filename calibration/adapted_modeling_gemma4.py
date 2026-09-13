"""Gemma4 (26B-A4B) expert-similarity calibration adapter.

Templated on ``adapted_modeling_qwen3_5_moe.py``: gemma4 stores its experts
FUSED as 3D tensors (``Gemma4TextExperts.gate_up_proj`` (E, 2I, H) and
``down_proj`` (E, H, I)), the same layout as Qwen3.5-MoE, so
``_all_expert_outputs`` carries over unchanged.

It could NOT be templated line-for-line on the hook site, though. In qwen3_5 the
MoE is a submodule at ``layer.mlp`` and the hook wraps that block's
``forward(hidden_states)``. Gemma4 has no MoE block: the decoder layer owns
``router`` and ``experts`` directly, and ``layer.mlp`` is the PARALLEL DENSE MLP
that runs alongside the MoE on every layer -- hooking it would calibrate the
wrong module. So the hook goes on ``Gemma4TextExperts.forward`` instead, which
receives the MoE input (already flattened to (T, H) and already through
``pre_feedforward_layernorm_2``) and owns the expert weights.

That choice also makes two of the required exclusions automatic:
  * the parallel dense MLP is a different class (``Gemma4TextMLP``) -> skipped;
  * the vision tower uses ``Gemma4VisionMLP`` -> skipped.
All 30 text layers are MoE (no ``first_k_dense_replace``), so all 30 calibrate.

Activation is ``gelu_pytorch_tanh``, read off the module rather than hardcoded.
"""

import torch
from transformers.models.gemma4.modeling_gemma4 import (
    Gemma4TextExperts as _Experts,
)
from transformers.models.gemma4.modeling_gemma4 import (
    Gemma4ForConditionalGeneration as _G4Cond,
)

from utils import compute_similarity_matrix_gpu


def enable_similarity_computation(self, method="frobenius", kernel="linear"):
    self.compute_similarity = True
    self.similarity_method = method
    self.kernel = kernel


def disable_similarity_computation(self):
    self.compute_similarity = False


def set_similarity_matrix(self, m):
    self.similarity_matrix = (m.clone().detach() if torch.is_tensor(m)
                              else torch.tensor(m))


def get_similarity_matrix(self):
    return getattr(self, "similarity_matrix", None)


def reset_similarity_matrix(self):
    self.similarity_matrix = None
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


@torch.no_grad()
def _all_expert_outputs(experts, x):
    """UNWEIGHTED output of every expert on every token: (E, T, H).

    Unweighted on purpose: the similarity basis must be the expert function,
    not the expert function scaled by how often the router happened to pick it.
    Chunked over tokens to bound the transient (128 experts * T * H).
    """
    W_gu, W_dn = experts.gate_up_proj, experts.down_proj   # (E,2I,H),(E,H,I)
    x = x.to(W_gu.dtype).to(W_gu.device)
    E = W_gu.shape[0]
    outs = []
    chunk = max(1, 4096 // max(1, E // 32))
    for s in range(0, x.shape[0], chunk):
        xc = x[s:s + chunk]                                # (c, H)
        gu = torch.einsum("th,eoh->eto", xc, W_gu)         # (E, c, 2I)
        g, u = gu.chunk(2, dim=-1)
        h = experts.act_fn(g) * u                          # (E, c, I)
        outs.append(torch.einsum("eti,ehi->eth", h, W_dn))  # (E, c, H)
    return torch.cat(outs, dim=1)                          # (E, T, H)


_orig_forward = _Experts.forward


def _sim_forward(self, hidden_states, top_k_index, top_k_weights):
    if getattr(self, "compute_similarity", False):
        x = hidden_states.view(-1, hidden_states.shape[-1])
        acts = _all_expert_outputs(self, x)                # (E, T, H)
        expert_outputs = [acts[e] for e in range(acts.shape[0])]
        sim = compute_similarity_matrix_gpu(
            expert_outputs, method=self.similarity_method,
            kernel=getattr(self, "kernel", "linear"), device=x.device)
        self.set_similarity_matrix(sim.detach().cpu())
        del acts, expert_outputs, sim
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return _orig_forward(self, hidden_states, top_k_index, top_k_weights)


for _name, _fn in [
    ("enable_similarity_computation", enable_similarity_computation),
    ("disable_similarity_computation", disable_similarity_computation),
    ("set_similarity_matrix", set_similarity_matrix),
    ("get_similarity_matrix", get_similarity_matrix),
    ("reset_similarity_matrix", reset_similarity_matrix),
    ("forward", _sim_forward),
]:
    setattr(_Experts, _name, _fn)


# cal_expert_similarity.py does ``model_class.from_pretrained(...)``.
Gemma4ForCausalLM = _G4Cond
