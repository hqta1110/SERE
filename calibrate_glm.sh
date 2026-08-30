#!/usr/bin/env bash
# Calibrate SERE similarity matrices for GLM-4.7-Flash (glm4_moe_lite).
# Runs in the tf5.16 env (reap/.venv_q36) because glm4_moe_lite isn't in the
# pinned serve transformers 4.57.6. Model sharded across all GPUs (device_map=auto).
# Output: artifacts/glm-4.7-flash-sere/similarity_matrices.pt  (the injection .pt).
set -euo pipefail
ROOT=/home/PC/SERE-glm
PY=/home/PC/reap/.venv_q36/bin/python
SNAP=/home/PC/.cache/huggingface/hub/models--zai-org--GLM-4.7-Flash/snapshots/7dd20894a642a0aa287e9827cb1a1f7f91386b67
DATA=/home/PC/SERE/calibration/data/fineweb_edu_calibration.parquet
OUT=$ROOT/artifacts/glm-4.7-flash-sere

# token budget sized to a 40GB shard: the calibration adapter materializes the
# unweighted (E=64, T, H=2048) expert-output tensor per MoE layer, so keep
# batch_size*max_len modest. 128*512 ~= 65k tokens is ample for a [64,64] matrix.
cd "$ROOT/calibration"
exec "$PY" cal_expert_similarity.py \
  --model_type glm4_moe_lite \
  --model_path "$SNAP" \
  --data_path "$DATA" \
  --output_path "$OUT" \
  --similarity_method frobenius --kernel linear \
  --batch_size 32 --max_len 512
