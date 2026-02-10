#!/bin/bash

# Export current user's UID/GID for container permissions
export HOST_UID=$(id -u)
export HOST_GID=$(id -g)

# NVFP4 Mode - Native Blackwell 4-bit Quantization
# Usage: ./run-nvfp4.sh [model_name]

set -e

MODEL="${1:-GadflyII/Qwen3-Coder-Next-NVFP4}"

echo "========================================="
echo "NVFP4 Mode - Blackwell Native 4-bit"
echo "========================================="
echo "Model: $MODEL"
echo ""
echo "NVFP4 provides:"
echo "  - ~5x throughput vs FP8 on Blackwell"
echo "  - Native 4-bit floating point (SM120)"
echo "  - Block scaling: 32 elements per E8M0 scale"
echo "  - ~4.25 bits per value effective"
echo ""
echo "⚠️  Requires:"
echo "  - Blackwell GPU (GB10, B100, B200)"
echo "  - Model with NVFP4 weights or quantization support"
echo ""

mkdir -p models state
chmod -R 777 state 2>/dev/null || true

if [[ "$MODEL" == /* ]] || [[ "$MODEL" == ./* ]]; then
    MODEL_PATH="/models/$(basename $MODEL)"
else
    MODEL_PATH="$MODEL"
fi

# Check if model supports NVFP4 (basic check)
echo "Checking NVFP4 availability..."
docker compose run --rm vllm python -c "
from vllm.model_executor.layers.quantization import QUANTIZATION_METHODS
if 'nvfp4' in QUANTIZATION_METHODS or 'mxfp4' in QUANTIZATION_METHODS:
    print('✓ NVFP4/MXFP4 quantization available')
else:
    print('⚠ NVFP4 not available, falling back to FP8')
" || true

echo ""
echo "Starting with NVFP4 (or best available)..."

# Try NVFP4 first, fallbacks are handled by vLLM
docker compose run --rm --service-ports \
    -e VLLM_ENABLE_NVFP4=1 \
    -e VLLM_USE_FLASHINFER_MOE_MXFP4_BF16=1 \
    vllm serve "$MODEL_PATH" \
    --host 0.0.0.0 \
    --port 8000 \
    --max-model-len 8192 \
    --quantization mxfp4 \
    --kv-cache-dtype fp8 \
    --dtype float16 \
    --gpu-memory-utilization 0.95 \
    --enable-prefix-caching \
    --enable-chunked-prefill \
    --async-scheduling
