#!/bin/bash
set -e

echo "========================================="
echo "Building Spark vLLM Docker Image"
echo "========================================="

# Check for NVIDIA Docker runtime
echo "Checking NVIDIA Docker runtime..."
if ! docker info | grep -q nvidia; then
    echo "WARNING: NVIDIA runtime not detected in Docker."
    echo "Make sure nvidia-container-toolkit is installed:"
    echo "  https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html"
    echo ""
fi

# Create necessary directories
echo "Creating directories..."
mkdir -p models state state/vllm state/huggingface state/transformers state/torch state/cache

# Build the Docker image
echo ""
echo "Building Docker image (this may take 30-60 minutes)..."
echo "Note: The build stage compiles vLLM and Flash Attention from source"
echo ""

export DOCKER_BUILDKIT=1
export COMPOSE_DOCKER_CLI_BUILD=1

#docker compose build --no-cache --progress=plain 2>&1 | tee build.log
docker compose build --progress=plain 2>&1 | tee build.log

echo ""
echo "========================================="
echo "Build Complete!"
echo "========================================="
echo ""
echo "Image: spark-vllm:latest"
echo ""
echo "To run vLLM with a model:"
echo "  docker compose up -d"
echo ""
echo "Or with a specific model:"
echo "  docker compose run --rm vllm serve Qwen/Qwen2.5-1.5B-Instruct --max-model-len 2048"
echo ""
echo "To use local models from ./models:"
echo "  docker compose run --rm vllm serve /models/your-model --max-model-len 2048"
echo ""
echo "Directories:"
echo "  ./models    - Place your model files here"
echo "  ./state     - Cache, HuggingFace hub, etc."
echo ""
