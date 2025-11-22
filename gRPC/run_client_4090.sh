#!/usr/bin/env bash
set -euo pipefail

# 默认参数
INPUT="The Little Match Girl tells the tragic and haunting story of a young girl wandering the dark, icy streets on New Year’s Eve. Snow falls steadily, covering the city in white, and the wind bites through the narrow alleys. The girl, poorly dressed in a thin apron and wearing no shoes—she lost them in the street earlier—walks slowly, clutching a bundle of matches in her small, trembling hands. Her feet are numb, her fingers stiff, and her stomach empty. She has not sold a single match all day, but she knows she cannot go home. Her father, harsh and quick-tempered, will beat her for failing to bring in money, and the house is no warmer than the streets."
DRAFT_MODEL="/home/lym/sijia/DSSD/gRPC/LLM/opt-125m"
MAX_LEN=256
SEED=321
TEMPERATURE=1
TOP_K=10
TOP_P=0
GAMMA=6
DEVICE="cuda:0"
SERVER_ADDR=127.0.0.1:8000


# Python解释器
PYTHON_BIN=${PYTHON_BIN:-python3}

# 循环运行gamma
for GAMMA in {3..3}; do
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




