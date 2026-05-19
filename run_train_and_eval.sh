#!/usr/bin/env bash
set -euo pipefail

cd /home/tap/TRO_lang

export MPLCONFIGDIR=/tmp/matplotlib-cache
export PYTHONPYCACHEPREFIX=/tmp/tro_lang_pycache
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
export NCCL_ASYNC_ERROR_HANDLING=1

PYTHON=/home/tap/miniconda3/envs/TRO/bin/python
ACCELERATE=/home/tap/miniconda3/envs/TRO/bin/accelerate
RUN_DIR=/home/tap/TRO_lang/results/shadow_hand_filtered_0519
LOG_DIR="${RUN_DIR}/logs"
mkdir -p "${LOG_DIR}"

echo "[start] $(date -Is)"
echo "[train] config/train_func_float_lang.yaml"

"${ACCELERATE}" launch \
  --multi_gpu \
  --num_processes 8 \
  --mixed_precision bf16 \
  train_ddp_func_lang.py \
  --config config/train_func_float_lang.yaml

echo "[train done] $(date -Is)"
echo "[eval] test05 latest checkpoint"

"${PYTHON}" eval_func_lang_loss.py \
  --config config/train_func_float_lang.yaml \
  --ckpt "${RUN_DIR}/ckpt/latest.pth" \
  --data-root /home/tap/Data-Filter/new_log/filtered_current_test05.jsonl \
  --max-batches 200 \
  | tee "${LOG_DIR}/test05_latest_eval.json"

echo "[done] $(date -Is)"
