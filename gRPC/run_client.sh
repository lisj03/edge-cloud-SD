#!/usr/bin/env bash
set -euo pipefail

# 默认参数
INPUT="I have 10 apples. I find 3 gold coins in the bottom of a river. The river runs near a big city that has something to do with what I can spend the coins on. What do I spend them on?"
DRAFT_MODEL="/Users/aoliliaoao/Downloads/DSSD-Efficient-Edge-Computing/gRPC/LLM/opt-125m"
MAX_LEN=50
SEED=321
TEMPERATURE=1
TOP_K=10
TOP_P=0
GAMMA=4
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




