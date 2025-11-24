#!/usr/bin/env bash
set -euo pipefail

INPUT="On a freezing New Year’s Eve, a poor little girl wandered the streets barefoot, clutching a handful of matches she had failed to sell. Afraid to go home to her harsh father, she crouched in a corner and struck a match for warmth."
DRAFT_MODEL="/home/lym/sijia/DSSD/gRPC/LLM/opt-125m"
MAX_LEN=256
SEED=321
TEMPERATURE=1
TOP_K=10
TOP_P=0
DEVICE="cuda:0"
SERVER_ADDR="127.0.0.1:8000"
GAMMAS="2 3 4 5 6 7 8 9 10 11 12 13 14 15"
# GAMMAS="2"
COOLDOWN=1

PYTHON_BIN=${PYTHON_BIN:-python3}

"${PYTHON_BIN}" ./test_gamma_sweep.py \
  --input "${INPUT}" \
  --draft_model_name "${DRAFT_MODEL}" \
  --server_addr "${SERVER_ADDR}" \
  --max_len "${MAX_LEN}" \
  --seed "${SEED}" \
  --device "${DEVICE}" \
  --temperature "${TEMPERATURE}" \
  --top_k "${TOP_K}" \
  --top_p "${TOP_P}" \
  --cooldown "${COOLDOWN}" \
  --gamma_values ${GAMMAS} \
  "$@"
