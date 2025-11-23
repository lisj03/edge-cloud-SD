#!/bin/bash

# Spider数据集测试脚本
# 用法示例：
# ./run_spider_test.sh

# 服务器地址（修改为你的实际服务器地址）
SERVER_ADDR="127.0.0.1:50051"

# 设备（根据你的硬件选择：mps, cuda, cpu）
DEVICE="mps"

# 测试参数
N_SAMPLES=20  # 测试样本数量
MAX_LEN=100   # 最大生成长度
GAMMA=4       # 推测token数量

# 运行测试
echo "====================================="
echo "Spider Dataset Test"
echo "====================================="
echo "Server: $SERVER_ADDR"
echo "Device: $DEVICE"
echo "Samples: $N_SAMPLES"
echo "====================================="

python test_spider.py \
    --server_addr "$SERVER_ADDR" \
    --device "$DEVICE" \
    --n_samples "$N_SAMPLES" \
    --max_len "$MAX_LEN" \
    --gamma "$GAMMA" \
    --test_function both \
    --seed 42

echo ""
echo "✓ Test completed!"
echo "Results saved to: spider_test_results.json"
