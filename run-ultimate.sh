#!/bin/bash

# Export current user's UID/GID for container permissions
export HOST_UID=$(id -u)
export HOST_GID=$(id -g)

# ULTIMATE Performance Mode - Maximum throughput for GB10/Blackwell
# Usage: ./run-ultimate.sh [model_name]

set -e

MODEL="${1:-Qwen/Qwen3-Coder-Next-FP8}"

echo "========================================="
echo "ULTIMATE Performance Mode"
echo "========================================="
echo "Model: $MODEL"
echo ""
echo "This mode enables ALL SOTA optimizations:"
echo "  - torch.compile with fusion passes"
echo "  - Speculative decoding (if available)"
echo "  - Max batch size optimization"
echo "  - Aggressive memory utilization"
echo ""

mkdir -p models state state/torch_compile_cache
chmod -R 777 state 2>/dev/null || true

if [[ "$MODEL" == /* ]] || [[ "$MODEL" == ./* ]]; then
    MODEL_PATH="/models/$(basename $MODEL)"
else
    MODEL_PATH="$MODEL"
fi

# ULTIMATE compilation config for Blackwell
COMPILATION_CONFIG='{
    "custom_ops": ["+quant_fp8", "+rms_norm"],
    "cudagraph_mode": "FULL_DECODE_ONLY",
    "compile_sizes": [1, 2, 4, 8, 16, 32, 64, 128]
}'

docker compose run --rm -e VLLM_USE_FLASHINFER_MOE_FP8=0 --service-ports \
    vllm serve "$MODEL_PATH" \
    --host 0.0.0.0 \
    --port 8000 \
    --max-model-len 4096 \
    --max-num-batched-tokens 16384 \
    --max-num-seqs 256 \
    --gpu-memory-utilization 0.95 \
    # --enable-prefix-caching (disabled for Mamba models) \
    --enable-chunked-prefill \
    --async-scheduling \
    --compilation-config "$COMPILATION_CONFIG" \
    --quantization fp8 \
    --kv-cache-dtype fp8
