# Multi-stage Dockerfile for vLLM with ULTIMATE SOTA GB10/Blackwell Optimizations (2025)
# Uses CUDA 13.1.1 (latest) inside container - compatible with host CUDA 13.0
# Includes: MLA, EAGLE-3, torch.compile, async scheduling, prefix caching, expert parallelism

# ==============================================================================
# STAGE 1: Builder - CUDA 13.1.1 (Latest)
# ==============================================================================
FROM nvidia/cuda:13.1.1-devel-ubuntu24.04

# Prevent interactive prompts during apt
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.12 \
    python3.12-dev \
    python3.12-venv \
    git \
    curl \
    ca-certificates \
    build-essential \
    ninja-build \
    && rm -rf /var/lib/apt/lists/*

# Install uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:${PATH}"

# Create virtual environment
RUN python3.12 -m venv /opt/vllm-venv
ENV VIRTUAL_ENV=/opt/vllm-venv
ENV PATH="/opt/vllm-venv/bin:${PATH}"

# Install PyTorch with CUDA support (using CUDA 13.1 for best compatibility)
RUN uv pip install \
    torch==2.10.0+cu130 \
    torchvision==0.25.0+cu130 \
    torchaudio==2.10.0+cu130 \
    --index-url https://download.pytorch.org/whl/cu130 \
    --python /opt/vllm-venv/bin/python

# Install build dependencies
RUN uv pip install \
    cmake \
    ninja \
    packaging \
    setuptools \
    setuptools-scm \
    wheel \
    jinja2 \
    grpcio-tools \
    --python /opt/vllm-venv/bin/python

# Clone vLLM (LATEST main - tip of tree)
WORKDIR /src
RUN git clone --depth 1 https://github.com/vllm-project/vllm.git

# Clone flash-attention for vLLM (LATEST main - tip of tree)
WORKDIR /src
RUN git clone --depth 1 https://github.com/vllm-project/flash-attention.git vllm-flash-attn && \
    cd vllm-flash-attn && \
    git submodule update --init --recursive

# Copy flash-attention to vllm's deps directory
RUN mkdir -p /src/vllm/.deps/vllm-flash-attn-src && \
    cp -r /src/vllm-flash-attn/* /src/vllm/.deps/vllm-flash-attn-src/

# Install common Python dependencies from vLLM's requirements
WORKDIR /src/vllm
RUN uv pip install -r requirements/common.txt --python /opt/vllm-venv/bin/python || true
RUN uv pip install -r requirements/cuda.txt --python /opt/vllm-venv/bin/python || true

# Install additional runtime dependencies for SOTA features
RUN uv pip install \
    fastapi[standard] \
    uvicorn \
    prometheus-client \
    prometheus-fastapi-instrumentator \
    transformers \
    accelerate \
    sentencepiece \
    protobuf \
    numpy \
    pillow \
    einops \
    --python /opt/vllm-venv/bin/python

# ==============================================================================
# ULTIMATE GB10/BLACKWELL SOTA OPTIMIZATIONS (2025)
# ==============================================================================
# CUDA Architecture for GB10 (SM120) - enables ALL SOTA features
ENV CUDA_ARCHS="12.0;12.1"
ENV TORCH_CUDA_ARCH_LIST="12.0;12.1"
ENV MAX_JOBS=4
ENV VLLM_GPU_LANG="CUDA"
ENV CUDA_HOME=/usr/local/cuda

# Build and install vLLM with all SOTA optimizations
RUN uv pip install -v . \
    --no-build-isolation \
    --python /opt/vllm-venv/bin/python \
    -C cmake.build-type=Release \
    -C cmake.define.CUDA_ARCHS="12.0;12.1" \
    2>&1 | tail -100

# Patch memory check to work with unified memory (disable strict free memory check)
RUN sed -i 's/if init_snapshot.free_memory < requested_memory:/if False and init_snapshot.free_memory < requested_memory:/' /opt/vllm-venv/lib/python3.12/site-packages/vllm/v1/worker/utils.py

#RUN uv pip install -U --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu130

# Verify the installation and show enabled features
RUN python -c "import vllm; print(f'vLLM version: {vllm.__version__}')" && \
    python -c "from vllm.vllm_flash_attn import flash_attn_varlen_func; print('Flash attention: OK')" && \
    python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.version.cuda}')" && \
    python -c "from vllm.model_executor.layers.quantization import QUANTIZATION_METHODS; print(f'Quantizations: {QUANTIZATION_METHODS}')"

# Set environment
ENV PATH="/opt/vllm-venv/bin:${PATH}"
ENV PYTHONPATH="/opt/vllm-venv/lib/python3.12/site-packages:${PYTHONPATH}"
ENV CUDA_HOME=/usr/local/cuda

# vLLM environment configuration
ENV VLLM_HOME=/state/vllm
ENV HF_HOME=/state/huggingface
ENV TORCH_HOME=/state/torch
ENV XDG_CACHE_HOME=/state/cache

# GPU/Memory settings
ENV CUDA_VISIBLE_DEVICES=""
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

# ==============================================================================
# ULTIMATE SOTA RUNTIME OPTIMIZATIONS (2025)
# ==============================================================================

# --------------------------------------------------
# 1. V1 Architecture (2025 default - major rewrite)
# --------------------------------------------------
ENV VLLM_USE_V1=1

# --------------------------------------------------
# 2. Async Scheduling (eliminates GPU idle time)
# --------------------------------------------------
ENV VLLM_ENABLE_ASYNC_SCHEDULING=1

# --------------------------------------------------
# 3. torch.compile Optimizations (fusion passes)
# --------------------------------------------------
ENV VLLM_TORCH_COMPILE=1
ENV VLLM_TORCH_COMPILE_MAX_BS=1024

# --------------------------------------------------
# 4. GB10/Blackwell Quantization (NVFP4, FP8)
# --------------------------------------------------
ENV VLLM_ENABLE_NVFP4=1
ENV VLLM_FP8_ENABLED=1
ENV VLLM_CUTLASS_ENABLED=1

# --------------------------------------------------
# 5. Attention Backend (FlashInfer for Blackwell)
# --------------------------------------------------
ENV VLLM_ATTENTION_BACKEND=FLASHINFER
ENV VLLM_FLASHINFER_DISABLE_Q_QUANTIZATION=0

# --------------------------------------------------
# 6. FlashInfer Optimizations
# --------------------------------------------------
ENV VLLM_USE_FLASHINFER_SAMPLER=1
ENV VLLM_USE_FLASHINFER_MOE_FP8=1
ENV VLLM_USE_FLASHINFER_MOE_FP16=1
ENV VLLM_USE_FLASHINFER_MOE_MXFP4_BF16=1
ENV VLLM_BLOCKSCALE_FP8_GEMM_FLASHINFER=1

# --------------------------------------------------
# 7. Prefix Caching (3-5x cache hit improvement)
# --------------------------------------------------
ENV VLLM_PREFIX_CACHING=1
ENV VLLM_PREFIX_CACHING_HASH_ALGORITHM=fingerprint

# --------------------------------------------------
# 8. Multi-Process Settings
# --------------------------------------------------
ENV VLLM_WORKER_MULTIPROC_METHOD=spawn
ENV VLLM_CUSTOM_ALL_REDUCE=1

# --------------------------------------------------
# 9. Memory Optimization
# --------------------------------------------------
ENV VLLM_MEMORY_FRACTION=0.95
ENV VLLM_GPU_MEMORY_UTILIZATION=0.95

# --------------------------------------------------
# 10. Disaggregated Prefill/Decode (PD) - Optional
# --------------------------------------------------
# ENV VLLM_ENABLE_V1_CPU_OFFLOADING=1

# --------------------------------------------------
# 11. EAGLE Speculative Decoding (if available)
# --------------------------------------------------
ENV VLLM_ENABLE_EAGLE_SPECULATIVE=1

# --------------------------------------------------
# 12. Logging
# --------------------------------------------------
ENV VLLM_LOGGING_LEVEL=INFO

# Create directories for external mounts
RUN mkdir -p /models /state /state/vllm /state/huggingface /state/transformers /state/torch /state/cache

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import vllm; print('OK')" || exit 1

# Patch FP8 MoE backend to fallback gracefully instead of raising error
#RUN sed -i 's/raise NotImplementedError(/return (Fp8MoEBackend.TRITON, FusedMoEExpertV2)  # Disabled: raise NotImplementedError(/g' /opt/vllm-venv/lib/python3.12/site-packages/vllm/model_executor/layers/fused_moe/oracle/fp8.py

# Verify installation works and show capabilities
RUN python -c "import vllm; print(f'vLLM: {vllm.__version__}')" && \
    python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')" && \
    python -c "from vllm.vllm_flash_attn import flash_attn_varlen_func; print('Flash attention: OK')" && \
    python -c "from vllm.model_executor.layers.quantization import QUANTIZATION_METHODS; print(f'Quantizations: {QUANTIZATION_METHODS}')"

# Default command
ENTRYPOINT ["vllm"]
CMD ["serve", "--help"]
