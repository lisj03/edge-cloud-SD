#!/usr/bin/env bash
set -euo pipefail

# 默认参数
INPUT="On a freezing New Year’s Eve, a poor little girl wandered the streets barefoot, clutching a handful of matches she had failed to sell. Afraid to go home to her harsh father, she crouched in a corner and struck a match for warmth."
DRAFT_MODEL="/Users/aoliliaoao/Downloads/DSSD-Efficient-Edge-Computing/gRPC/LLM/opt-125m"
MAX_LEN=128
SEED=321
TEMPERATURE=1
TOP_K=10
TOP_P=0
GAMMA=2
DEVICE="mps"
SERVER_ADDR=127.0.0.1:8000


# Python解释器
PYTHON_BIN=${PYTHON_BIN:-python3}

# 循环运行gamma
for GAMMA in {4..4}; do
  echo ""
  echo "========================================"
  echo "Running with GAMMA=${GAMMA}"
  echo "========================================"
  
  "${PYTHON_BIN}" ./client.py \
    --input "${INPUT}" \
    --draft_model_name "${DRAFT_MODEL}" \
    --server_addr "${SERVER_ADDR}" \
    --gamma "${GAMMA}" \
    --max_len "${MAX_LEN}" \
    --seed "${SEED}" \
    --device "${DEVICE}" \
    --temperature $TEMPERATURE \
    --top_k $TOP_K \
    --top_p $TOP_P \

    "$@"
done

echo ""
echo "========================================"
echo "All runs completed"
echo "========================================"




