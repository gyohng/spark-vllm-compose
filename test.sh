#!/bin/bash
# Test script to verify vLLM Docker setup

set -e

echo "========================================="
echo "Spark vLLM Test Suite"
echo "========================================="

# Test 1: Check image exists
echo ""
echo "Test 1: Checking if Docker image exists..."
if docker image inspect spark-vllm:latest >/dev/null 2>&1; then
    echo "✓ Image spark-vllm:latest found"
else
    echo "✗ Image not found. Run ./build.sh first"
    exit 1
fi

# Test 2: Check GPU access
echo ""
echo "Test 2: Checking GPU access..."
if docker run --rm --gpus all spark-vllm:latest python -c "import torch; print(f'GPUs: {torch.cuda.device_count()}')" 2>/dev/null | grep -q "GPUs:"; then
    GPU_COUNT=$(docker run --rm --gpus all spark-vllm:latest python -c "import torch; print(torch.cuda.device_count())" 2>/dev/null)
    echo "✓ GPUs detected: $GPU_COUNT"
else
    echo "✗ GPU access failed. Check NVIDIA Container Toolkit"
    exit 1
fi

# Test 3: Check vLLM imports
echo ""
echo "Test 3: Checking vLLM imports..."
if docker run --rm --gpus all spark-vllm:latest python -c "import vllm; print(vllm.__version__)" >/dev/null 2>&1; then
    VLLM_VERSION=$(docker run --rm --gpus all spark-vllm:latest python -c "import vllm; print(vllm.__version__)" 2>/dev/null)
    echo "✓ vLLM version: $VLLM_VERSION"
else
    echo "✗ vLLM import failed"
    exit 1
fi

# Test 4: Check flash attention
echo ""
echo "Test 4: Checking Flash Attention..."
if docker run --rm --gpus all spark-vllm:latest python -c "from vllm.vllm_flash_attn import flash_attn_varlen_func" 2>/dev/null; then
    echo "✓ Flash Attention available"
else
    echo "✗ Flash Attention import failed"
    exit 1
fi

# Test 5: Check state directories
echo ""
echo "Test 5: Checking state directories..."
mkdir -p models state
if [ -d "models" ] && [ -d "state" ]; then
    echo "✓ Directories exist"
else
    echo "✗ Directory creation failed"
    exit 1
fi

# Test 6: Quick inference test (optional, requires model)
echo ""
echo "Test 6: Quick API test (requires running container)..."
if curl -s http://localhost:8000/health >/dev/null 2>&1; then
    echo "✓ vLLM API is responding"
else
    echo "⚠ vLLM not running (this is OK if you haven't started it yet)"
fi

echo ""
echo "========================================="
echo "All tests passed!"
echo "========================================="
echo ""
echo "To start vLLM:"
echo "  ./run.sh Qwen/Qwen2.5-1.5B-Instruct"
echo ""
