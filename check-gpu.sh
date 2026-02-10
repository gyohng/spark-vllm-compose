#!/bin/bash
# GPU/NVIDIA Container Toolkit Diagnostic Script

echo "========================================="
echo "GPU/NVIDIA Diagnostic"
echo "========================================="
echo ""

echo "1. Checking NVIDIA driver..."
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader 2>/dev/null || echo "❌ nvidia-smi failed - NVIDIA driver not installed?"
echo ""

echo "2. Checking Docker..."
docker --version
docker info --format '{{json .}}' 2>/dev/null | grep -o '"nvidia"' > /dev/null && echo "✓ NVIDIA runtime found in Docker" || echo "❌ NVIDIA runtime NOT found in Docker"
echo ""

echo "3. Checking NVIDIA Container Toolkit..."
if command -v nvidia-ctk &> /dev/null; then
    echo "✓ nvidia-ctk found"
    nvidia-ctk --version
else
    echo "❌ nvidia-ctk not found"
fi

if [ -f /etc/docker/daemon.json ]; then
    echo ""
    echo "Docker daemon config:"
    cat /etc/docker/daemon.json
fi
echo ""

echo "4. Testing GPU in Docker..."
docker run --rm --gpus all ubuntu:24.04 nvidia-smi 2>&1 | head -10 || echo "❌ GPU test failed"
echo ""

echo "5. Checking docker compose GPU support..."
docker compose version
echo ""

echo "========================================="
echo "Fix Instructions (if needed):"
echo "========================================="
echo ""
echo "If NVIDIA runtime is missing, install it:"
echo ""
echo "  # Ubuntu/Debian"
echo "  sudo apt-get update"
echo "  sudo apt-get install -y nvidia-container-toolkit"
echo "  sudo nvidia-ctk runtime configure --runtime=docker"
echo "  sudo systemctl restart docker"
echo ""
echo "  # Verify installation"
echo "  docker run --rm --gpus all ubuntu:24.04 nvidia-smi"
echo ""
