#!/usr/bin/env bash
set -euo pipefail

# Run this file on the Linux GPU server that contains the model.
MODEL_PATH="/home/user/gptdata/CZH/model/Qwen/Qwen3-8B"
SERVED_MODEL_NAME="qwen3-8b"
PORT="${VLLM_PORT:-8000}"

if [ ! -d "$MODEL_PATH" ]; then
  echo "Model directory not found: $MODEL_PATH" >&2
  exit 1
fi

vllm serve "$MODEL_PATH" \
  --served-model-name "$SERVED_MODEL_NAME" \
  --host 0.0.0.0 \
  --port "$PORT" \
  --dtype auto \
  --max-model-len 8192
