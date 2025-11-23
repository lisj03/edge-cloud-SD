#!/bin/bash

# Confidence阈值扫描实验脚本
# 测试confidence = 0.005, 0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6
# 生成max_len=1000的文段

echo "========================================"
echo "Confidence Threshold Sweep Experiment"
echo "========================================"

# 配置
SERVER_ADDR="127.0.0.1:8000"
DRAFT_MODEL="./LLM/opt-125m"
DEVICE="mps"
MAX_LEN=1000
INPUT_TEXT="Alan Turing theorized that computers would one day become the most powerful tools for solving complex problems. In his groundbreaking 1950 paper 'Computing Machinery and Intelligence', he proposed what is now known as the Turing Test, a criterion of intelligence in a machine."

echo ""
echo "Configuration:"
echo "  Server: $SERVER_ADDR"
echo "  Draft Model: $DRAFT_MODEL"
echo "  Device: $DEVICE"
echo "  Max Length: $MAX_LEN"
echo "  Testing 9 confidence thresholds"
echo "========================================"
echo ""

# 检查SSH隧道
echo "Checking SSH tunnel to server..."
if ! nc -z 127.0.0.1 8000 2>/dev/null; then
    echo "⚠️  Warning: Cannot connect to 127.0.0.1:8000"
    echo "   Please ensure SSH tunnel is running:"
    echo "   ssh -L 8000:127.0.0.1:50051 lym@121.237.183.19"
    echo ""
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# 运行实验
python test_confidence_sweep.py \
    --server_addr "$SERVER_ADDR" \
    --draft_model_name "$DRAFT_MODEL" \
    --device "$DEVICE" \
    --max_len $MAX_LEN \
    --input "$INPUT_TEXT" \
    --temperature 0.7 \
    --top_k 10 \
    --top_p 0 \
    --seed 321

# 检查是否成功
if [ $? -eq 0 ]; then
    echo ""
    echo "========================================"
    echo "✅ Experiment completed successfully!"
    echo "========================================"
    echo ""
    echo "Results saved to:"
    echo "  - confidence_sweep_results_*.json"
    echo "  - confidence_sweep_results_*.csv"
    echo ""
    echo "To view results:"
    echo "  cat confidence_sweep_results_*.csv"
    echo "  python -m json.tool confidence_sweep_results_*.json"
else
    echo ""
    echo "❌ Experiment failed!"
    exit 1
fi
