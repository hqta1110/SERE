#!/usr/bin/env bash
# Build the dedicated, isolated SERE-glm serve venv.
# Reproducible: vLLM stack from moe-eval-unified/repro/env/v1 lock (same as the
# working SERE-v1), + this tree's SERE rerouting kernel, + glm4_moe_lite config shim.
# Does NOT touch new-efficient-moe/.venv or SERE_v1 (the working setup).
set -euo pipefail
ROOT=/home/PC/SERE-glm
ENVDIR=$ROOT/_env
VENV=$ROOT/.venv
SPEC=/home/PC/moe-eval-unified/repro/env/v1

echo "=== [1/4] sync vLLM stack from vendored v1 lock -> $VENV  $(date +%H:%M:%S) ==="
mkdir -p "$ENVDIR"
cp "$SPEC/pyproject.toml" "$SPEC/uv.lock" "$ENVDIR/"
( cd "$ENVDIR" && UV_PROJECT_ENVIRONMENT="$VENV" uv sync --locked )

echo "=== [2/4] build SERE rerouting kernel into the venv (no-build-isolation)  $(date +%H:%M:%S) ==="
uv pip install --python "$VENV/bin/python" setuptools wheel
uv pip install --python "$VENV/bin/python" --no-build-isolation -e "$ROOT/vllm"

echo "=== [3/4] install glm4_moe_lite config shim (transformers 4.57.6 lacks it)  $(date +%H:%M:%S) ==="
SP=$("$VENV/bin/python" -c "import site;print(site.getsitepackages()[0])")
printf '/home/PC/glm_lite_shim\nimport glm4_moe_lite_register\n' > "$SP/zz_glm_lite.pth"

echo "=== [4/4] verify  $(date +%H:%M:%S) ==="
"$VENV/bin/python" - <<'PY'
import importlib.metadata as m
eps=[e.name for e in m.entry_points(group='vllm.general_plugins')]
print("plugins:", eps, "-> register_SERE_vllm present:", 'register_SERE_vllm' in eps)
import SERE_vllm.rerouting_cuda_ops.rerouting_ops as k
print("rerouting kernel import: OK")
from transformers import AutoConfig
snap="/home/PC/.cache/huggingface/hub/models--zai-org--GLM-4.7-Flash/snapshots/7dd20894a642a0aa287e9827cb1a1f7f91386b67"
c=AutoConfig.from_pretrained(snap)
print("glm4_moe_lite config:", type(c).__name__, c.model_type, "experts", c.n_routed_experts, "topk", c.num_experts_per_tok)
import vllm; print("vllm", vllm.__version__)
PY
echo "=== BUILD_OK $(date +%H:%M:%S) ==="
