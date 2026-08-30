#!/usr/bin/env bash
# Serve GLM-4.7-Flash with SERE (injection path) from the ISOLATED SERE-glm venv.
# Usage: serve_sere_glm.sh <gpus> <port> <select_top_k> <threshold>
# Untouched: SERE_v1 + new-efficient-moe/.venv (the working qwen SERE setup).
set -uo pipefail
GPUS="$1"; PORT="$2"; TOPK="$3"; THR="$4"
ROOT=/home/PC/SERE-glm
VENV=$ROOT/.venv
SNAP=/home/PC/.cache/huggingface/hub/models--zai-org--GLM-4.7-Flash/snapshots/7dd20894a642a0aa287e9827cb1a1f7f91386b67

# NCCL single-node repair (GCP gIB)
export NCCL_NET=Socket NCCL_IB_DISABLE=1; unset NCCL_TUNER_CONFIG_PATH
export LD_LIBRARY_PATH="$(printf '%s' "${LD_LIBRARY_PATH:-}" | tr ':' '\n' | grep -v gib | paste -sd':' -)"

# SERE injection: plugin + per-layer similarity .pt + routing knobs
export VLLM_PLUGINS=register_SERE_vllm
export SERE_SIMILARITY_PT=$ROOT/artifacts/glm-4.7-flash-sere/similarity_matrices.pt
export SERE_SELECT_TOP_K="$TOPK"
export SERE_THRESHOLD="$THR"

export CUDA_VISIBLE_DEVICES="$GPUS"
export PATH="$VENV/bin:$PATH"
exec "$VENV/bin/vllm" serve "$SNAP" \
  --served-model-name glm-4.7-flash --trust-remote-code \
  --tensor-parallel-size 2 --max-model-len 16384 \
  --gpu-memory-utilization 0.90 --no-enable-prefix-caching \
  --port "$PORT"
