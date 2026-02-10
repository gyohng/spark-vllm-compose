#!/bin/bash
# Helper script to download models to the local ./models directory

set -e

MODEL_ID="${1:-}"

if [ -z "$MODEL_ID" ]; then
    echo "Usage: $0 <model_id>"
    echo ""
    echo "Examples:"
    echo "  $0 Qwen/Qwen2.5-1.5B-Instruct"
    echo "  $0 meta-llama/Llama-2-7b-chat-hf"
    echo "  $0 microsoft/Phi-3-mini-4k-instruct"
    echo ""
    exit 1
fi

MODEL_NAME=$(basename "$MODEL_ID")
echo "========================================="
echo "Downloading model: $MODEL_ID"
echo "Target: ./models/$MODEL_NAME"
echo "========================================="

mkdir -p models

# Run huggingface-cli in the vLLM container to download
docker compose run --rm \
    -v "$(pwd)/models:/models" \
    -e HF_HOME=/state/huggingface \
    vllm huggingface-cli download \
    "$MODEL_ID" \
    --local-dir "/models/$MODEL_NAME" \
    --local-dir-use-symlinks False

echo ""
echo "========================================="
echo "Download complete!"
echo "========================================="
echo "Model location: ./models/$MODEL_NAME"
echo ""
echo "To run:"
echo "  ./run.sh $MODEL_NAME --max-model-len 2048"
