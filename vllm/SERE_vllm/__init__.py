# SPDX-License-Identifier: Apache-2.0

import os
from importlib import metadata


def _vllm_version_tuple() -> tuple[int, int, int]:
    try:
        version = metadata.version("vllm")
    except metadata.PackageNotFoundError:
        return (0, 0, 0)
    parts = []
    for piece in version.split(".")[:3]:
        number = ""
        for char in piece:
            if not char.isdigit():
                break
            number += char
        parts.append(int(number or 0))
    while len(parts) < 3:
        parts.append(0)
    return tuple(parts)


def _use_vllm_v1_compat() -> bool:
    return _vllm_version_tuple() >= (0, 9, 0)


# The host profile exports NCCL_NET=gIB without the matching net plugin
# loaded in either conda env; leaving it set fails NCCL communicator
# creation. Undo it regardless of vLLM version, before vLLM imports its
# distributed layer.
if os.environ.get("NCCL_NET") == "gIB":
    os.environ.pop("NCCL_NET", None)

# vLLM 0.8.4-only quirks:
#   - PyNCCL communicator can fail before model init in this environment;
#     VLLM_DISABLE_PYNCCL=1 forces vLLM to use torch/custom all-reduce.
#   - The SERE model classes registered by this plugin
#     (Qwen2MoeForCausalLMSERE etc.) are written against vLLM 0.8.4's V0
#     engine APIs. vLLM 0.8.4 silently picks its experimental V1 engine when
#     VLLM_USE_V1 is unset, which breaks SERE weight loading. Force V0 on
#     0.8.x so `vllm serve` users don't have to remember VLLM_USE_V1=0.
if not _use_vllm_v1_compat():
    os.environ.setdefault("VLLM_USE_V1", "0")
    os.environ["VLLM_DISABLE_PYNCCL"] = "1"


from vllm import ModelRegistry  # noqa: E402  (must follow env-var setup above)

_SERE_MODELS = {
    "Qwen2MoeForCausalLMSERE":
        "SERE_vllm.sere_qwen2_moe:Qwen2MoeForCausalLMSERE",
    "DeepseekV2ForCausalLMSERE":
        "SERE_vllm.sere_deepseek_v2:DeepseekV2ForCausalLMSERE",
    "Qwen3MoeForCausalLMSERE":
        "SERE_vllm.sere_qwen3_moe:Qwen3MoeForCausalLMSERE",
}


def register():
    if _use_vllm_v1_compat():
        from .vllm_v1_patch import V1_MODEL_REFS, patch_vllm_v1_sere

        patch_vllm_v1_sere()
        model_refs = V1_MODEL_REFS
    else:
        model_refs = _SERE_MODELS

    supported_archs = ModelRegistry.get_supported_archs()
    for arch, model_ref in model_refs.items():
        if arch not in supported_archs:
            ModelRegistry.register_model(arch, model_ref)
