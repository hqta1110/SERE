# Registers Glm4MoeLiteConfig into transformers 4.57.6 so vLLM 0.18.1 can serve
# zai-org/GLM-4.7-Flash (model_type=glm4_moe_lite) using its own glm4_moe_lite executor.
# The Lite arch mirrors glm4_moe (vLLM's Glm4MoeLite subclasses Glm4MoE), so the config
# is Glm4MoeConfig with a different model_type. Idempotent + fail-safe.
try:
    from transformers import AutoConfig
    from transformers.models.glm4_moe.configuration_glm4_moe import Glm4MoeConfig

    class Glm4MoeLiteConfig(Glm4MoeConfig):
        model_type = "glm4_moe_lite"

    try:
        AutoConfig.register("glm4_moe_lite", Glm4MoeLiteConfig)
    except Exception:
        pass  # already registered
except Exception:
    pass
