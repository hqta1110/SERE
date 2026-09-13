#!/usr/bin/env bash
# Build a SERE serve venv and compile the rerouting CUDA kernel into it.
#
#   scripts/build_venv.sh [--family qwen|glm|gemma4] [--venv DIR] [--glm-shim-dir DIR]
#
# The MODEL FAMILY decides the whole stack, because the serve-side plugin is a
# monkeypatch against a specific vLLM MoE layer:
#
#   qwen   (Qwen3-30B-A3B, Qwen3.6-35B-A3B)  branch accuracy-bench-repro  vLLM 0.18.1
#   glm    (GLM-4.7-Flash)                    branch glm-4.7-flash         vLLM 0.18.1
#   gemma4 (gemma-4-26B-A4B-it)               branch gemma4-support        vLLM 0.29.0
#
# One venv per family. They are NOT interchangeable: the trees carry different
# vllm_v1_patch.py files and gemma4 needs a different vLLM major entirely.
#
# Checkout the branch for your family FIRST, then run this from the repo root.
#
# Prereqs: uv, nvcc on PATH (CUDA toolkit matching the torch build), a GPU whose
# arch the kernel is compiled for. NEVER copy a built .so between machines --
# rebuild. Kernel build is ~3-6 min.
set -euo pipefail

FAMILY=qwen; VENV=""; SHIM_DIR=""
while [ $# -gt 0 ]; do
  case "$1" in
    --family) FAMILY=$2; shift 2 ;;
    --venv) VENV=$2; shift 2 ;;
    --glm-shim-dir) SHIM_DIR=$2; shift 2 ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
VENV=${VENV:-$ROOT/.venv}
SHIM_DIR=${SHIM_DIR:-$ROOT/.glm_shim}

command -v uv   >/dev/null || { echo "uv not on PATH"   >&2; exit 1; }
command -v nvcc >/dev/null || { echo "nvcc not on PATH -- the kernel build needs it" >&2; exit 1; }

echo "=== [1/4] python stack -> $VENV  ($FAMILY)  $(date +%H:%M:%S) ==="
case "$FAMILY" in
  qwen|glm)
    # vLLM 0.18.1 / transformers 4.57.6 / torch 2.10.0, fully locked.
    SPEC=$ROOT/repro/envs/v1
    WORK=$(mktemp -d); trap 'rm -rf "$WORK"' EXIT
    cp "$SPEC/pyproject.toml" "$SPEC/uv.lock" "$WORK/"
    ( cd "$WORK" && UV_PROJECT_ENVIRONMENT="$VENV" uv sync --locked )
    ;;
  gemma4)
    # vLLM 0.29.0 / transformers 5.17.0 / torch 2.13.0. Pinned, not locked:
    # a frozen requirements set, so resolution is reproducible but not hash-pinned.
    uv venv --python 3.12 "$VENV"
    uv pip install --python "$VENV/bin/python" -r "$ROOT/repro/envs/gemma4/requirements.txt"
    ;;
  *) echo "unknown family: $FAMILY (want qwen|glm|gemma4)" >&2; exit 2 ;;
esac

echo "=== [2/4] build the SERE rerouting kernel into the venv  $(date +%H:%M:%S) ==="
# --no-build-isolation because setup.py imports torch at build time.
uv pip install --python "$VENV/bin/python" setuptools wheel
uv pip install --python "$VENV/bin/python" --no-build-isolation -e "$ROOT/vllm"

if [ "$FAMILY" = glm ]; then
  echo "=== [3/4] glm4_moe_lite config shim (transformers 4.57.6 has no such model_type) ==="
  mkdir -p "$SHIM_DIR"
  cp "$ROOT/glm4_moe_lite_register.py" "$SHIM_DIR/"
  SP=$("$VENV/bin/python" -c "import site;print(site.getsitepackages()[0])")
  printf '%s\nimport glm4_moe_lite_register\n' "$SHIM_DIR" > "$SP/zz_glm_lite.pth"
else
  echo "=== [3/4] no shim needed for $FAMILY ==="
fi

echo "=== [4/4] verify  $(date +%H:%M:%S) ==="
"$VENV/bin/python" - <<'PY'
import importlib.metadata as m
eps = [e.name for e in m.entry_points(group='vllm.general_plugins')]
assert 'register_SERE_vllm' in eps, f"plugin entry point missing, got {eps}"
print("plugin entry point: register_SERE_vllm OK")

# The kernel lives in a SUBMODULE. `import rerouting_ops_cuda` at top level fails
# even on a perfectly good install -- that false negative has cost real hours.
from SERE_vllm.rerouting_cuda_ops import rerouting_ops_cuda   # noqa: F401
print("rerouting kernel import: OK")

import SERE_vllm, os, vllm
print("plugin tree:", os.path.dirname(SERE_vllm.__file__))
print("vllm:", vllm.__version__)
PY
echo "=== BUILD_OK  venv=$VENV  family=$FAMILY  $(date +%H:%M:%S) ==="
