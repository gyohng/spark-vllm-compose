#!/bin/bash
# Fix NVIDIA Runtime for Docker
# Run this script with sudo to configure Docker for GPU support

echo "========================================="
echo "Fixing NVIDIA Runtime for Docker"
echo "========================================="
echo ""

# Check if running as root
if [ "$EUID" -ne 0 ]; then 
    echo "Please run as root or with sudo:"
    echo "  sudo $0"
    exit 1
fi

echo "1. Creating Docker daemon.json with nvidia runtime..."
mkdir -p /etc/docker

cat > /etc/docker/daemon.json << 'EOF'
{
    "runtimes": {
        "nvidia": {
            "path": "nvidia-container-runtime",
            "runtimeArgs": []
        }
    },
    "default-runtime": "nvidia"
}
EOF

echo "   Created /etc/docker/daemon.json"
cat /etc/docker/daemon.json
echo ""

echo "2. Restarting Docker..."
systemctl restart docker

# Wait for Docker to be ready
sleep 3

echo ""
echo "3. Testing GPU in Docker..."
docker run --rm --runtime=nvidia ubuntu:24.04 nvidia-smi

if [ $? -eq 0 ]; then
    echo ""
    echo "========================================="
    echo "✓ SUCCESS! GPU is now accessible in Docker"
    echo "========================================="
    echo ""
    echo "You can now run:"
    echo "  ./run-nvfp4.sh"
else
    echo ""
    echo "========================================="
    echo "❌ FAILED - GPU still not accessible"
    echo "========================================="
    echo ""
    echo "Check logs: journalctl -u docker -n 50"
fi
