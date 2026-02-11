#!/bin/bash

# Qwen3-Coder-Next with Prefix Caching Mode
# Usage: ./run-prefix-caching.sh [model_name]
# 
# This mode enables prefix caching for better performance with repeated prompts
# (e.g., coding workflows). Async scheduling is disabled to allow this.
# Based on: https://forums.developer.nvidia.com/t/how-to-run-qwen3-coder-next-on-spark/...

set -e

# Export current user's UID/GID for container permissions
export HOST_UID=$(id -u)
export HOST_GID=$(id -g)

MODEL="${1:-Qwen/Qwen3-Coder-Next-FP8}"

echo "========================================="
echo "Qwen3-Coder-Next with Prefix Caching"
echo "========================================="
echo "Model: $MODEL"
echo ""
echo "Features:"
echo "  ✓ Prefix caching enabled (3-5x speedup for repeated prompts)"
echo "  ✓ FlashInfer attention backend (~170K tokens context)"
echo "  ✓ Tool calling support (qwen3_coder parser)"
echo "  ✓ Fast safetensors loading"
echo ""
echo "Note: Async scheduling disabled (required for prefix caching + Mamba)"
echo ""

mkdir -p models state
chmod -R 777 state 2>/dev/null || true

# Use the host network mode for better performance
docker compose run --rm --service-ports \
    -e HOST_UID="$HOST_UID" \
    -e HOST_GID="$HOST_GID" \
    -e VLLM_USE_V1=1 \
    -e VLLM_ENABLE_ASYNC_SCHEDULING=0 \
    -e VLLM_USE_FLASHINFER_MOE_FP8=0 \
    vllm serve "$MODEL" \
    --host 0.0.0.0 \
    --port 8000 \
    --max-model-len 32768 \
    --gpu-memory-utilization 0.8 \
    --kv-cache-dtype fp8 \
    --attention-backend flashinfer \
    --enable-prefix-caching \
    --enable-auto-tool-choice \
    --tool-call-parser qwen3_coder \
    --load-format auto
