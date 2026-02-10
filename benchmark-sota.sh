#!/bin/bash
# Comprehensive SOTA Benchmark for GB10/Blackwell
# Export current user's UID/GID for container permissions
export UID=$(id -u)
export GID=$(id -g)


set -e

MODEL="${1:-Qwen/Qwen3-Coder-Next-FP8}"

echo "========================================="
echo "ULTIMATE SOTA Benchmark - GB10/Blackwell"
echo "========================================="
echo "Model: $MODEL"
echo ""

mkdir -p models state state/torch_compile_cache

# Check if image exists
if ! docker image inspect spark-vllm:latest >/dev/null 2>&1; then
    echo "Error: Image spark-vllm:latest not found. Run ./build.sh first"
    exit 1
fi

echo "1. Checking GPU and ALL SOTA features..."
echo ""
docker compose run --rm --entrypoint python vllm -c "
import torch
import os

try:
    from vllm.model_executor.layers.quantization import QUANTIZATION_METHODS
    quant_methods = list(QUANTIZATION_METHODS.keys()) if hasattr(QUANTIZATION_METHODS, 'keys') else QUANTIZATION_METHODS
except:
    quant_methods = ['nvfp4', 'fp8', 'awq', 'gptq', 'compressed-tensors']

print('='*60)
print('GPU & CUDA INFORMATION')
print('='*60)
print(f'GPU: {torch.cuda.get_device_name(0)}')
print(f'CUDA Available: {torch.cuda.is_available()}')
print(f'CUDA Version: {torch.version.cuda}')
print(f'PyTorch Version: {torch.__version__}')
print()
print('='*60)
print('AVAILABLE QUANTIZATION METHODS')
print('='*60)
for method in sorted(quant_methods):
    marker = '✓' if method in ['nvfp4', 'mxfp4', 'fp8', 'awq', 'gptq', 'compressed-tensors'] else ' '
    print(f'  [{marker}] {method}')
print()
print('='*60)
print('SOTA FEATURES STATUS')
print('='*60)
print(f'  V1 Architecture: {os.environ.get(\"VLLM_USE_V1\", \"not set\")}')
print(f'  Async Scheduling: {os.environ.get(\"VLLM_ENABLE_ASYNC_SCHEDULING\", \"not set\")}')
print(f'  torch.compile: {os.environ.get(\"VLLM_TORCH_COMPILE\", \"not set\")}')
print(f'  NVFP4 Enabled: {os.environ.get(\"VLLM_ENABLE_NVFP4\", \"not set\")}')
print(f'  FP8 Enabled: {os.environ.get(\"VLLM_FP8_ENABLED\", \"not set\")}')
print(f'  Prefix Caching: {os.environ.get(\"VLLM_PREFIX_CACHING\", \"not set\")}')
print(f'  EAGLE Speculative: {os.environ.get(\"VLLM_ENABLE_EAGLE_SPECULATIVE\", \"not set\")}')
print()
"

echo ""
echo "2. Throughput Benchmark (Standard)..."
docker compose run --rm -e VLLM_USE_FLASHINFER_MOE_FP8=0 vllm bench throughput \
    --model "$MODEL" \
    --max-model-len 1024 \
    --input-len 512 \
    --output-len 128 \
    --num-prompts 100 \
    --dtype bfloat16 \
    --gpu-memory-utilization 0.95

echo ""
echo "3. Latency Benchmark (Single Request)..."
docker compose run --rm -e VLLM_USE_FLASHINFER_MOE_FP8=0 vllm bench latency \
    --model "$MODEL" \
    --max-model-len 1024 \
    --input-len 512 \
    --output-len 128 \
    --batch-size 1 \
    --dtype bfloat16 \
    --gpu-memory-utilization 0.95

echo ""
echo "========================================="
echo "Benchmark Complete!"
echo "========================================="
echo ""
echo "For ULTIMATE performance, try:"
echo "  ./run-ultimate.sh Qwen/Qwen3-Coder-Next-FP8"
echo ""
echo "For NVFP4 quantization (Blackwell):"
echo "  ./run-nvfp4.sh GadflyII/Qwen3-Coder-Next-NVFP4"
echo ""
echo "For EAGLE speculative decoding:"
echo "  ./run-speculative.sh Qwen/Qwen3-Coder-Next-FP8"
echo ""
echo "For DeepSeek with MLA:"
echo "  ./run-deepseek.sh deepseek-ai/DeepSeek-V3"
echo ""
