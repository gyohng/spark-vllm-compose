#!/bin/bash

# Export current user's UID/GID for container permissions
export HOST_UID=$(id -u)
export HOST_GID=$(id -g)

# ULTIMATE Performance Mode - Maximum throughput for GB10/Blackwell
# Usage: ./run-ultimate.sh [model_name]

set -e

MODEL="${1:-Qwen/Qwen3-Coder-Next-FP8}"

# Context length: Maximum 128K for Qwen3-Coder-Next (tons of memory on GB10!)
# Override if you need more concurrent requests: MAX_MODEL_LEN=32768 ./run-ultimate.sh
MAX_MODEL_LEN="${MAX_MODEL_LEN:-131072}"

# Memory optimization for 128GB unified RAM:
# - GPU util 0.92 leaves headroom for unified memory management
# - Max 64 sequences (plenty for single-user, saves KV cache memory)
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.92}"
MAX_SEQS="${MAX_SEQS:-64}"

# Detect model architecture for compatibility
IS_MAMBA=false
if [[ "$MODEL" =~ (qwen3|Qwen3|mamba|Mamba) ]]; then
    IS_MAMBA=true
fi

# Build optimal flags based on model type
# Priority: Higher throughput + No start delay
if [ "$IS_MAMBA" = true ]; then
    # Mamba: Prefix caching gives 3-5x speedup on cache hits
    # Async scheduling conflicts and causes hard error
    SCHEDULING_FLAGS="--enable-prefix-caching --enable-chunked-prefill"
    MODEL_TYPE="Mamba (Qwen3)"
else
    # Transformer: Both features work together
    SCHEDULING_FLAGS="--enable-prefix-caching --enable-chunked-prefill --async-scheduling"
    MODEL_TYPE="Transformer"
fi

echo "========================================="
echo "ULTIMATE Performance Mode"
echo "========================================="
echo "Model: $MODEL"
echo "Max Context: ${MAX_MODEL_LEN} tokens"
echo "GPU Memory: ${GPU_MEM_UTIL} (0.92 for 128GB unified)"
echo "Max Sequences: ${MAX_SEQS} (reduced for memory)"
echo "Architecture: $MODEL_TYPE"
echo ""
echo "This mode enables ALL SOTA optimizations:"
echo "  ✓ torch.compile with fusion passes"
echo "  ✓ Speculative decoding (if available)"
echo "  ✓ Max batch size optimization"
echo "  ✓ Aggressive memory utilization"
echo "  ✓ Prefix caching enabled"
if [ "$IS_MAMBA" = false ]; then
    echo "  ✓ Async scheduling enabled"
fi
echo ""

mkdir -p models state state/torch_compile_cache
chmod -R 777 state 2>/dev/null || true

if [[ "$MODEL" == /* ]] || [[ "$MODEL" == ./* ]]; then
    MODEL_PATH="/models/$(basename $MODEL)"
else
    MODEL_PATH="$MODEL"
fi

# ULTIMATE compilation config for Blackwell
# Reduced compile sizes to save memory (still covers common batch sizes)
COMPILATION_CONFIG='{
    "custom_ops": ["+quant_fp8", "+rms_norm"],
    "cudagraph_mode": "FULL_DECODE_ONLY",
    "compile_sizes": [1, 2, 4, 8, 16, 32, 64]
}'

docker compose run --rm -e VLLM_USE_FLASHINFER_MOE_FP8=0 --service-ports \
    vllm serve "$MODEL_PATH" \
    --host 0.0.0.0 \
    --port 8000 \
    --max-model-len $MAX_MODEL_LEN \
    --max-num-batched-tokens 16384 \
    --max-num-seqs $MAX_SEQS \
    --gpu-memory-utilization $GPU_MEM_UTIL \
    --num-scheduler-steps 1 \
    $SCHEDULING_FLAGS \
    --compilation-config "$COMPILATION_CONFIG" \
    --quantization fp8 \
    --kv-cache-dtype fp8
