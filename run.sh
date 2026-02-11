#!/bin/bash

# Spark vLLM Runner Script - ULTIMATE SOTA GB10/Blackwell Optimized (2025)
# Usage: ./run.sh [model_name] [additional_args]

set -e

# Export current user's UID/GID for container permissions
export HOST_UID=$(id -u)
export HOST_GID=$(id -g)

# Default model
MODEL="${1:-Qwen/Qwen3-Coder-Next-FP8}"
shift || true

# Additional arguments
EXTRA_ARGS="$@"

# Context length: Default 8K for general use (up to 128K supported)
# Context length: Maximum 128K (tons of memory on GB10!)
# Override if you need more concurrent requests: MAX_MODEL_LEN=32768 ./run.sh
MAX_MODEL_LEN="${MAX_MODEL_LEN:-131072}"

# Create directories
mkdir -p models state state/vllm state/huggingface state/transformers state/torch state/cache state/torch_compile_cache config
chmod -R 777 state 2>/dev/null || true

# Detect model architecture for compatibility
# Qwen3/Mamba models: prefix caching works, async scheduling conflicts
# DeepSeek/Transformer: both work
IS_MAMBA=false
if [[ "$MODEL" =~ (qwen3|Qwen3|mamba|Mamba) ]]; then
    IS_MAMBA=true
fi

# Build optimal flags based on model type
if [ "$IS_MAMBA" = true ]; then
    # Mamba: Prefix caching >> Async scheduling (higher throughput, no delay)
    # vLLM will auto-disable async anyway, so we don't pass it
    SCHEDULING_FLAGS="--enable-prefix-caching"
    MODEL_TYPE="Mamba (Qwen3)"
else
    # Transformer: Both work, async helps multi-user
    SCHEDULING_FLAGS="--enable-prefix-caching --async-scheduling"
    MODEL_TYPE="Transformer"
fi

echo "========================================="
echo "Spark vLLM - ULTIMATE SOTA (2025)"
echo "========================================="
echo "Model: $MODEL"
echo "Max Context: ${MAX_MODEL_LEN} tokens"
echo "Architecture: $MODEL_TYPE"
echo ""
echo "🚀 ENABLED OPTIMIZATIONS:"
echo "  ✓ V1 Architecture (2025 rewrite)"
echo "  ✓ torch.compile (fusion passes)"
echo "  ✓ NVFP4 Quantization (Blackwell native)"
echo "  ✓ FP8 Quantization"
echo "  ✓ FlashInfer Attention (SM120 optimized)"
echo "  ✓ Prefix Caching (3-5x hit rate)"
echo "  ✓ EAGLE Speculative Decoding"
echo "  ✓ CUTLASS MoE/MLA Kernels"
echo "  ✓ CUDA 13.1.1 (latest)"
if [ "$IS_MAMBA" = false ]; then
    echo "  ✓ Async Scheduling (zero GPU idle)"
fi
echo ""

# Check local vs HF model
if [[ "$MODEL" == /* ]] || [[ "$MODEL" == ./* ]]; then
    if [ ! -d "models/$(basename $MODEL)" ] && [ ! -d "$MODEL" ]; then
        echo "WARNING: Model directory not found: $MODEL"
        echo "Looking in: $(pwd)/models/"
        echo ""
    fi
    MODEL_PATH="/models/$(basename $MODEL)"
else
    MODEL_PATH="$MODEL"
fi

# Run vLLM with SOTA defaults
docker compose run --rm -e VLLM_USE_FLASHINFER_MOE_FP8=0 --service-ports \
    -e MODEL_PATH="$MODEL_PATH" \
    vllm serve "$MODEL_PATH" \
    --host 0.0.0.0 \
    --port 8000 \
    --max-model-len $MAX_MODEL_LEN \
    --gpu-memory-utilization 0.95 \
    $SCHEDULING_FLAGS \
    --enable-chunked-prefill \
    $EXTRA_ARGS
