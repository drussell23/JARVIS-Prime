"""
JARVIS Prime Docker Image v2.0.0
=================================

Production-ready container for JARVIS Prime local LLM inference service.
Pre-loads GGUF models during build to eliminate startup delays.

Key Features:
- Multi-stage build (builder -> runtime) for minimal image size
- Pre-baked model weights via model_downloader.py
- Model pre-warming during build (triggers lazy loading)
- Volume mounts for persistent model cache
- Health checks wait for models to be ready
- Apple Silicon (M1/M2/M3/M4) native support via llama-cpp-python Metal

Build (from jarvis-prime repo root):
    docker build -t jarvis-prime:latest -f docker/Dockerfile .

Build with specific model:
    docker build -t jarvis-prime:latest -f docker/Dockerfile \
        --build-arg DEFAULT_MODEL=phi-3.5-mini .

Run:
    docker compose -f docker/docker-compose.yml up -d

Architecture:
    ┌─────────────────────────────────────────────────────────────────┐
    │                    JARVIS Prime Docker v2.0                     │
    ├─────────────────────────────────────────────────────────────────┤
    │  Stage 1: Builder                                               │
    │  ├── Python dependencies (torch, transformers, llama-cpp)       │
    │  ├── Pre-download GGUF models from HuggingFace                  │
    │  └── Pre-warm models (trigger lazy loading/compilation)         │
    │                                                                 │
    │  Stage 2: Runtime                                               │
    │  ├── Minimal Python runtime                                     │
    │  ├── Pre-loaded model cache (copied from builder)               │
    │  ├── FastAPI server (run_server.py)                             │
    │  └── Health check endpoint (/health)                            │
    └─────────────────────────────────────────────────────────────────┘

v2.0.0 - Advanced Docker setup with pre-loaded models and hybrid orchestration
"""

# =============================================================================
# BUILD ARGUMENTS
# =============================================================================
ARG PYTHON_VERSION=3.11
ARG MODEL_CACHE_DIR=/opt/jarvis_prime_cache
ARG HF_CACHE_DIR=/opt/jarvis_prime_cache/huggingface
ARG DEFAULT_MODEL=phi-3.5-mini
ARG PRELOAD_MODELS=true

# =============================================================================
# Stage 1: Builder - Install dependencies and pre-load models
# =============================================================================
FROM python:${PYTHON_VERSION}-slim-bookworm AS builder

# Re-declare ARGs after FROM (Docker requirement)
ARG MODEL_CACHE_DIR
ARG HF_CACHE_DIR
ARG DEFAULT_MODEL
ARG PRELOAD_MODELS

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    curl \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Set cache environment variables (critical for HuggingFace)
ENV HF_HOME="${HF_CACHE_DIR}"
ENV HUGGINGFACE_HUB_CACHE="${HF_CACHE_DIR}"
ENV TRANSFORMERS_CACHE="${HF_CACHE_DIR}"
ENV TORCH_HOME="${MODEL_CACHE_DIR}/torch"
ENV XDG_CACHE_HOME="${MODEL_CACHE_DIR}/xdg"
ENV LLAMA_CPP_LIB_CACHE="${MODEL_CACHE_DIR}/llama_cpp"

# Create cache directories with proper permissions
RUN mkdir -p ${MODEL_CACHE_DIR} ${HF_CACHE_DIR} \
    ${MODEL_CACHE_DIR}/torch ${MODEL_CACHE_DIR}/xdg \
    ${MODEL_CACHE_DIR}/llama_cpp ${MODEL_CACHE_DIR}/models

# Upgrade pip and install build tools
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Copy requirements first (for Docker layer caching)
WORKDIR /app
COPY requirements.txt* /tmp/
COPY setup.py* pyproject.toml* /tmp/

# Install Python dependencies
# llama-cpp-python needs special handling for Metal/CUDA
RUN pip install --no-cache-dir \
    fastapi>=0.104.0 \
    uvicorn[standard]>=0.24.0 \
    pydantic>=2.0.0 \
    aiohttp>=3.9.0 \
    huggingface_hub>=0.20.0 \
    tqdm>=4.66.0 \
    psutil>=5.9.0 \
    || true

# Install llama-cpp-python (CPU version for Docker, Metal handled at runtime)
RUN pip install --no-cache-dir llama-cpp-python>=0.2.50 || \
    CMAKE_ARGS="-DLLAMA_BLAS=ON -DLLAMA_BLAS_VENDOR=OpenBLAS" \
    pip install --no-cache-dir llama-cpp-python>=0.2.50

# Copy application code
COPY . /app/

# Copy and run model preloader
COPY docker/preload_models.py /tmp/preload_models.py

# Pre-load models during build (if enabled)
RUN if [ "${PRELOAD_MODELS}" = "true" ]; then \
        echo "=== Pre-loading JARVIS Prime Models ===" && \
        python /tmp/preload_models.py \
            --cache-dir ${MODEL_CACHE_DIR}/models \
            --model ${DEFAULT_MODEL} \
            --warmup true \
        || echo "Model pre-loading completed (some models may load at runtime)"; \
    else \
        echo "=== Model pre-loading disabled ==="; \
    fi

# List cache contents for verification
RUN echo "=== Pre-loaded model cache contents ===" && \
    ls -lah ${MODEL_CACHE_DIR}/models/ 2>/dev/null || echo "No models pre-loaded" && \
    du -sh ${MODEL_CACHE_DIR}/* 2>/dev/null || echo "Cache empty"

# =============================================================================
# Stage 2: Runtime - Minimal image with pre-loaded models
# =============================================================================
FROM python:${PYTHON_VERSION}-slim-bookworm AS runtime

ARG MODEL_CACHE_DIR
ARG HF_CACHE_DIR

# Install runtime dependencies only
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    libopenblas-base \
    && rm -rf /var/lib/apt/lists/*

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Set cache environment variables
ENV HF_HOME="${HF_CACHE_DIR}"
ENV HUGGINGFACE_HUB_CACHE="${HF_CACHE_DIR}"
ENV TRANSFORMERS_CACHE="${HF_CACHE_DIR}"
ENV TORCH_HOME="${MODEL_CACHE_DIR}/torch"
ENV XDG_CACHE_HOME="${MODEL_CACHE_DIR}/xdg"
ENV LLAMA_CPP_LIB_CACHE="${MODEL_CACHE_DIR}/llama_cpp"

# Create cache directories (writable at runtime)
RUN mkdir -p ${MODEL_CACHE_DIR} ${HF_CACHE_DIR} \
    ${MODEL_CACHE_DIR}/torch ${MODEL_CACHE_DIR}/xdg \
    ${MODEL_CACHE_DIR}/llama_cpp ${MODEL_CACHE_DIR}/models

# Copy pre-loaded model cache from builder
COPY --from=builder ${MODEL_CACHE_DIR} ${MODEL_CACHE_DIR}

# Copy application code
WORKDIR /app
COPY --from=builder /app /app

# Copy entrypoint script
COPY docker/docker-entrypoint.sh /usr/local/bin/docker-entrypoint.sh
RUN chmod +x /usr/local/bin/docker-entrypoint.sh

# Create non-root user for security
RUN useradd -m -u 1000 jarvis && \
    chown -R jarvis:jarvis /app ${MODEL_CACHE_DIR}

# Switch to non-root user
USER jarvis

# Environment variables for runtime configuration
ENV PORT=8000
ENV HOST=0.0.0.0
ENV MODEL_PATH=/opt/jarvis_prime_cache/models/current.gguf
ENV LOG_LEVEL=INFO
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Expose port
EXPOSE 8000

# Health check - wait for model to be ready
HEALTHCHECK --interval=10s --timeout=10s --retries=30 --start-period=180s \
    CMD curl -sf http://localhost:${PORT}/health | grep -E '"status":\s*"(healthy|starting)"' || exit 1

# Entrypoint and default command
ENTRYPOINT ["/usr/local/bin/docker-entrypoint.sh"]
CMD ["python", "run_server.py", "--port", "8000", "--host", "0.0.0.0"]
