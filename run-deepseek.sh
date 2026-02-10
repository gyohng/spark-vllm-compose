#!/bin/bash

# DeepSeek-V3/R1 Optimized Mode with MLA
# Usage: ./run-deepseek.sh [deepseek_model]

set -e

MODEL="${1:-deepseek-ai/DeepSeek-V3}"

echo "========================================="
echo "DeepSeek-V3/R1 MLA Optimized Mode"
echo "========================================="
echo "Model: $MODEL"
echo ""
echo "MLA (Multi-head Latent Attention) provides:"
echo "  - 9.6x more memory capacity for KV caches"
echo "  - Up to 3x throughput improvement"
echo "  - Matrix absorption algorithm"
echo ""

mkdir -p models state
chmod -R 777 state 2>/dev/null || true

# DeepSeek specific optimizations
# MLA is auto-detected for DeepSeek models

docker compose run --rm --service-ports \
    -e VLLM_ATTENTION_BACKEND=FLASHINFER \
    vllm serve "$MODEL" \
    --host 0.0.0.0 \
    --port 8000 \
    --max-model-len 32768 \
    --max-num-batched-tokens 8192 \
    --tensor-parallel-size 1 \
    --pipeline-parallel-size 1 \
    --quantization fp8 \
    --kv-cache-dtype fp8 \
    --gpu-memory-utilization 0.90 \
    --enable-prefix-caching \
    --enable-chunked-prefill \
    --async-scheduling
