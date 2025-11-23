#!/bin/bash

# 快速测试脚本 - 使用少量样本快速验证
# Quick test with small sample size

# 服务器地址（通过SSH隧道）
SERVER_ADDR="127.0.0.1:8000"

# 只测试5个样本进行快速验证
python test_spider.py \
    --server_addr "$SERVER_ADDR" \
    --device mps \
    --n_samples 10 \
    --max_len 80 \
    --gamma 4 \
    --test_function both \
    --seed 42

echo ""
echo "✓ Quick test completed!"
