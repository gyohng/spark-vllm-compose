#!/bin/bash

# Export current user's UID/GID for container permissions
export HOST_UID=$(id -u)
export HOST_GID=$(id -g)

# EAGLE-3 Speculative Decoding Mode
# Usage: ./run-speculative.sh [model_name] [speculator_model]

set -e

MODEL="${1:-Qwen/Qwen3-Coder-Next-FP8}"
SPECULATOR="${2:-}"  # Optional speculator model

# Context length: Maximum 128K (tons of memory on GB10!)
MAX_MODEL_LEN="${MAX_MODEL_LEN:-131072}"

# Detect model architecture for compatibility
IS_MAMBA=false
if [[ "$MODEL" =~ (qwen3|Qwen3|mamba|Mamba) ]]; then
    IS_MAMBA=true
fi

# Build optimal flags based on model type
if [ "$IS_MAMBA" = true ]; then
    # Mamba: Async scheduling conflicts
    SCHEDULING_FLAGS="--enable-prefix-caching"
    MODEL_TYPE="Mamba (Qwen3)"
else
    # Transformer: Both work
    SCHEDULING_FLAGS="--enable-prefix-caching --async-scheduling"
    MODEL_TYPE="Transformer"
fi

echo "========================================="
echo "EAGLE-3 Speculative Decoding Mode"
echo "========================================="
echo "Target Model: $MODEL"
echo "Max Context: ${MAX_MODEL_LEN} tokens"
echo "Architecture: $MODEL_TYPE"
[ -n "$SPECULATOR" ] && echo "Speculator: $SPECULATOR"
echo ""
echo "EAGLE-3 can provide 2-3x throughput improvement"
echo "by predicting multiple tokens ahead."
echo ""

mkdir -p models state
chmod -R 777 state 2>/dev/null || true

if [[ "$MODEL" == /* ]] || [[ "$MODEL" == ./* ]]; then
    MODEL_PATH="/models/$(basename $MODEL)"
else
    MODEL_PATH="$MODEL"
fi

# Build speculative config
if [ -n "$SPECULATOR" ]; then
    SPEC_CONFIG="{\"model\": \"$SPECULATOR\", \"num_speculative_tokens\": 5, \"method\": \"eagle3\"}"
    echo "Using EAGLE-3 with speculator model..."
else
    # Use n-gram speculative decoding as fallback
    SPEC_CONFIG="{\"method\": \"ngram\", \"num_speculative_tokens\": 5}"
    echo "Using N-Gram speculative decoding..."
fi

docker compose run --rm --service-ports \
    -e VLLM_ENABLE_EAGLE_SPECULATIVE=1 \
    vllm serve "$MODEL_PATH" \
    --host 0.0.0.0 \
    --port 8000 \
    --max-model-len $MAX_MODEL_LEN \
    --gpu-memory-utilization 0.90 \
    $SCHEDULING_FLAGS \
    --speculative-config "$SPEC_CONFIG"
