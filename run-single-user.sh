#!/bin/bash

# Export current user's UID/GID for container permissions
export HOST_UID=$(id -u)
export HOST_GID=$(id -g)

# SINGLE-USER Maximum Throughput Mode
# Optimized for ONE sequential connection (not concurrent users)
# 
# Why this is different from multi-user modes:
# - Async scheduling → NO benefit (only 1 request to schedule)
# - Speculative decoding → HUGE benefit (2-3x tokens/sec)
# - Prefix caching → Only helps if repeating prompts
#
# Usage: ./run-single-user.sh [model_name]

set -e

MODEL="${1:-Qwen/Qwen3-Coder-Next-FP8}"

# Context length: Maximum 128K for single-user (tons of memory on GB10!)
MAX_MODEL_LEN="${MAX_MODEL_LEN:-131072}"

# Detect model architecture for compatibility
IS_MAMBA=false
if [[ "$MODEL" =~ (qwen3|Qwen3|mamba|Mamba) ]]; then
    IS_MAMBA=true
fi

# For single-user, async scheduling never helps (only 1 request)
# So we never enable it regardless of model type
if [ "$IS_MAMBA" = true ]; then
    MODEL_TYPE="Mamba (Qwen3)"
else
    MODEL_TYPE="Transformer"
fi

echo "========================================="
echo "SINGLE-USER Maximum Throughput Mode"
echo "========================================="
echo "Model: $MODEL"
echo "Max Context: ${MAX_MODEL_LEN} tokens"
echo "Architecture: $MODEL_TYPE"
echo ""
echo "Optimized for ONE connection:"
echo "  ✓ N-Gram Speculative Decoding (5 tokens ahead)"
echo "  ✓ torch.compile (fusion passes)"
echo "  ✓ CUDA Graphs (zero CPU overhead)"
echo "  ✓ FP8/NVFP4 quantization"
echo "  ✓ Prefix caching (if repeating prompts)"
echo ""
echo "NOT enabled (no benefit for single user):"
echo "  ✗ Async scheduling (needs multiple requests)"
echo ""

mkdir -p models state state/torch_compile_cache
chmod -R 777 state 2>/dev/null || true

if [[ "$MODEL" == /* ]] || [[ "$MODEL" == ./* ]]; then
    MODEL_PATH="/models/$(basename $MODEL)"
else
    MODEL_PATH="$MODEL"
fi

# Single-user optimized compilation config
# Focus on decode speed, not batching
COMPILATION_CONFIG='{
    "custom_ops": ["+quant_fp8", "+rms_norm"],
    "cudagraph_mode": "FULL_DECODE_ONLY",
    "compile_sizes": [1, 2, 4, 8, 16, 32, 64]
}'

# N-Gram speculative config for single-user speed
SPEC_CONFIG='{"method": "ngram", "num_speculative_tokens": 5}'

docker compose run --rm -e VLLM_USE_FLASHINFER_MOE_FP8=0 --service-ports \
    vllm serve "$MODEL_PATH" \
    --host 0.0.0.0 \
    --port 8000 \
    --max-model-len $MAX_MODEL_LEN \
    --max-num-batched-tokens 8192 \
    --gpu-memory-utilization 0.95 \
    --enable-prefix-caching \
    --enable-chunked-prefill \
    --compilation-config "$COMPILATION_CONFIG" \
    --speculative-config "$SPEC_CONFIG" \
    --quantization fp8 \
    --kv-cache-dtype fp8
