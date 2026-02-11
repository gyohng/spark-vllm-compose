#!/bin/bash

# Export current user's UID/GID for container permissions
export HOST_UID=$(id -u)
export HOST_GID=$(id -g)

# SINGLE-USER Total Completion Time Optimized
# 
# Optimized for minimum TOTAL time: TTFT + generation time
# NOT just tok/s - we factor in ALL delays (compile, graphs, etc.)
#
# Usage: ./run-single-user.sh [model_name]
# Modes: MODE=fastest|balanced|maxspeed ./run-single-user.sh
#
# ℹ️  IMPORTANT: All modes produce IDENTICAL output quality!
#    The modes only affect SPEED, not token quality.

set -e

MODEL="${1:-Qwen/Qwen3-Coder-Next-FP8}"

# Mode selection:
# - fastest: Minimize TTFT, no compile/graph overhead (short prompts/outputs)
# - balanced: Cached compile + graphs (default, best overall)
# - maxspeed: Full optimizations for long generations (accept warmup delay)
MODE="${MODE:-balanced}"

# Context length: Maximum 128K
MAX_MODEL_LEN="${MAX_MODEL_LEN:-131072}"

# Detect model architecture
IS_MAMBA=false
if [[ "$MODEL" =~ (qwen3|Qwen3|mamba|Mamba) ]]; then
    IS_MAMBA=true
    MODEL_TYPE="Mamba (Qwen3)"
else
    MODEL_TYPE="Transformer"
fi

echo "========================================="
echo "SINGLE-USER Total Time Optimized"
echo "========================================="
echo "Model: $MODEL"
echo "Mode: $MODE"
echo "Max Context: ${MAX_MODEL_LEN} tokens"
echo "Architecture: $MODEL_TYPE"
echo ""

# Build config based on MODE
case "$MODE" in
  fastest)
    # MINIMUM delay mode - best for short prompts + short outputs (<100 tokens)
    # No compile overhead, no graph capture, no speculative verification
    echo "⚡ FASTEST MODE: Minimum TTFT, no initial delay"
    echo "  ✗ No torch.compile (saves 10-30s warmup)"
    echo "  ✗ No CUDA graphs (saves 30s capture)"
    echo "  ✗ No speculative decoding (saves verification overhead)"
    echo "  ✓ Immediate response"
    echo ""
    COMPILATION_CONFIG='{"level": 0}'
    SPEC_CONFIG=''
    CHUNKED_PREFILL=''
    ;;
  
  maxspeed)
    # MAXIMUM throughput for long generations (>500 tokens)
    # Accept initial delay for 2-3x faster generation
    echo "🚀 MAX SPEED MODE: Maximum throughput for long outputs"
    echo "  ✓ torch.compile (fusion passes)"
    echo "  ✓ CUDA graphs (zero CPU overhead)"
    echo "  ✓ Speculative decoding (5 tokens ahead)"
    echo "  ✓ Chunked prefill (better interactivity)"
    echo "  ⚠  30-60s initial warmup"
    echo ""
    COMPILATION_CONFIG='{
      "custom_ops": ["+quant_fp8", "+rms_norm"],
      "cudagraph_mode": "FULL_DECODE_ONLY",
      "compile_sizes": [1, 2, 4, 8, 16, 32, 64]
    }'
    SPEC_CONFIG='--speculative-config {"method": "ngram", "num_speculative_tokens": 5}'
    CHUNKED_PREFILL='--enable-chunked-prefill'
    ;;
  
  balanced|*)
    # Best overall - cached compile, graphs, speculative for long outputs
    # Subsequent requests are fast
    echo "⚖️  BALANCED MODE: Best total time (cached)"
    echo "  ✓ torch.compile with cache"
    echo "  ✓ CUDA graphs (after first capture)"
    echo "  ✓ Speculative decoding"
    echo "  ✓ First request: ~30s warmup"
    echo "  ✓ Follow-up: maximum speed"
    echo ""
    MODE="balanced"
    COMPILATION_CONFIG='{
      "custom_ops": ["+quant_fp8", "+rms_norm"],
      "cudagraph_mode": "FULL_DECODE_ONLY",
      "compile_sizes": [1, 2, 4, 8, 16, 32, 64]
    }'
    SPEC_CONFIG='--speculative-config {"method": "ngram", "num_speculative_tokens": 5}'
    CHUNKED_PREFILL='--enable-chunked-prefill'
    ;;
esac

mkdir -p models state state/torch_compile_cache
chmod -R 777 state 2>/dev/null || true

if [[ "$MODEL" == /* ]] || [[ "$MODEL" == ./* ]]; then
    MODEL_PATH="/models/$(basename $MODEL)"
else
    MODEL_PATH="$MODEL"
fi

# Base command
CMD="docker compose run --rm -e VLLM_USE_FLASHINFER_MOE_FP8=0 --service-ports \
    vllm serve \"$MODEL_PATH\" \
    --host 0.0.0.0 \
    --port 8000 \
    --max-model-len $MAX_MODEL_LEN \
    --max-num-batched-tokens 8192 \
    --gpu-memory-utilization 0.95 \
    --enable-prefix-caching \
    --quantization fp8 \
    --kv-cache-dtype fp8"

# Add mode-specific flags
if [ "$MODE" != "fastest" ]; then
    CMD="$CMD --compilation-config '$COMPILATION_CONFIG'"
fi

if [ -n "$SPEC_CONFIG" ]; then
    CMD="$CMD $SPEC_CONFIG"
fi

if [ -n "$CHUNKED_PREFILL" ]; then
    CMD="$CMD $CHUNKED_PREFILL"
fi

# Execute
eval $CMD
