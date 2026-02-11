#!/bin/bash

# Export current user's UID/GID for container permissions
export HOST_UID=$(id -u)
export HOST_GID=$(id -g)

# NVFP4 Mode - Native Blackwell 4-bit Quantization
# Usage: ./run-nvfp4.sh [model_name]

set -e

MODEL="${1:-GadflyII/Qwen3-Coder-Next-NVFP4}"

# Context length: Maximum 128K for NVFP4 (tons of memory on GB10!)
MAX_MODEL_LEN="${MAX_MODEL_LEN:-131072}"

# Detect model architecture for compatibility
IS_MAMBA=false
if [[ "$MODEL" =~ (qwen3|Qwen3|mamba|Mamba) ]]; then
    IS_MAMBA=true
fi

# Build optimal flags based on model type
if [ "$IS_MAMBA" = true ]; then
    # Mamba: Prefix caching >> Async scheduling
    SCHEDULING_FLAGS="--enable-prefix-caching --enable-chunked-prefill"
    MODEL_TYPE="Mamba (Qwen3)"
else
    # Transformer: Both work
    SCHEDULING_FLAGS="--enable-prefix-caching --enable-chunked-prefill --async-scheduling"
    MODEL_TYPE="Transformer"
fi

echo "========================================="
echo "NVFP4 Mode - Blackwell Native 4-bit"
echo "========================================="
echo "Model: $MODEL"
echo "Max Context: ${MAX_MODEL_LEN} tokens"
echo "Architecture: $MODEL_TYPE"
echo ""
echo "NVFP4 provides:"
echo "  - ~5x throughput vs FP8 on Blackwell"
echo "  - Native 4-bit floating point (SM120)"
echo "  - Block scaling: 32 elements per E8M0 scale"
echo "  - ~4.25 bits per value effective"
echo ""
echo "✓ Enabled:"
echo "  - NVFP4 weights (compressed-tensors)"
echo "  - Prefix caching"
echo "  - Chunked prefill"
echo ""
echo "✗ Disabled (NVFP4 compatibility):"
echo "  - torch.compile (dtype mismatch issue)"
echo ""

mkdir -p models state
chmod -R 777 state 2>/dev/null || true

if [[ "$MODEL" == /* ]] || [[ "$MODEL" == ./* ]]; then
    MODEL_PATH="/models/$(basename $MODEL)"
else
    MODEL_PATH="$MODEL"
fi

# Note: NVFP4/MXFP4 support is enabled via VLLM_ENABLE_NVFP4=1
# The model config will be auto-detected from config.json

echo ""
echo "Starting with NVFP4 (or best available)..."
echo ""
echo "⚠️  NVFP4 currently disables torch.compile due to dtype compatibility"
echo "   This is a known issue with torch.compile + NVFP4 on Blackwell"
echo ""

# Try NVFP4 first, fallbacks are handled by vLLM
# Note: torch.compile disabled for NVFP4 due to dtype mismatch (float vs c10::Half)
docker compose run --rm --service-ports \
    -e VLLM_ENABLE_NVFP4=1 \
    -e VLLM_USE_FLASHINFER_MOE_MXFP4_BF16=1 \
    -e VLLM_TORCH_COMPILE=0 \
    -e VLLM_USE_V1=1 \
    vllm serve "$MODEL_PATH" \
    --host 0.0.0.0 \
    --port 8000 \
    --max-model-len $MAX_MODEL_LEN \
    --kv-cache-dtype fp8 \
    --dtype float16 \
    --gpu-memory-utilization 0.95 \
    --enforce-eager \
    $SCHEDULING_FLAGS
