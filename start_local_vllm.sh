#!/bin/bash
# 快速启动本地 vLLM 服务

MODEL_PATH="models/Qwen--Qwen2.5-7B-Instruct"
PORT=8000

echo "🚀 启动本地 vLLM 服务"
echo "   模型: $MODEL_PATH"
echo "   端口: $PORT"
echo ""

# 检查模型是否存在
if [ ! -d "$MODEL_PATH" ]; then
    echo "❌ 模型不存在: $MODEL_PATH"
    echo ""
    echo "请先下载模型："
    echo "  python scripts/download_model.py --model Qwen/Qwen2.5-7B-Instruct --output models/"
    exit 1
fi

# 启动服务（使用 --enforce-eager 避免 duplicate template name 错误）
# 使用 70% GPU 内存利用率以确保有足够内存用于 KV cache
echo "⚠️  使用 --enforce-eager 模式（避免 duplicate template name 错误）"
echo "   使用 70% GPU 内存利用率（确保有足够内存用于 KV cache）"
echo ""

python scripts/launch_vllm_server.py --model "$MODEL_PATH" --port $PORT --enforce-eager --gpu-memory-utilization 0.7

