# ==============================================================================
# JARVIS-Prime Docker Image - Cloud Run Optimized
# ==============================================================================
# Lightweight build using llama-cpp-python for efficient GGUF model serving
# Optimized for Google Cloud Run with fast cold starts
#
# Build:   docker build -t jarvis-prime:latest .
# Run:     docker run -p 8000:8000 -v ./models:/app/models jarvis-prime:latest
#
# Cloud Run Deploy:
#   docker tag jarvis-prime:latest us-central1-docker.pkg.dev/PROJECT/jarvis-prime/jarvis-prime:latest
#   docker push us-central1-docker.pkg.dev/PROJECT/jarvis-prime/jarvis-prime:latest
# ==============================================================================

# Stage 1: Build dependencies (includes llama-cpp-python compilation)
FROM python:3.11-slim as builder

WORKDIR /build

# Install build dependencies for llama-cpp-python
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip and install wheel
RUN pip install --no-cache-dir --upgrade pip wheel setuptools

# Install llama-cpp-python (compiles from source for optimal performance)
# Use CPU-only build for Cloud Run compatibility
RUN pip wheel --no-cache-dir --wheel-dir=/build/wheels llama-cpp-python==0.2.90

# Install other dependencies
COPY requirements.txt .
RUN pip wheel --no-cache-dir --wheel-dir=/build/wheels -r requirements.txt

# Stage 2: Final runtime image (minimal)
FROM python:3.11-slim as runtime

# Labels
LABEL maintainer="JARVIS-Prime Team"
LABEL description="JARVIS-Prime Tier-0 Muscle Memory Brain"
LABEL version="2.0.0"

# Environment - Cloud Run optimized
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app \
    # Server settings
    JARVIS_PRIME_HOST=0.0.0.0 \
    JARVIS_PRIME_PORT=8000 \
    # Model settings
    MODEL_PATH=/app/models/current.gguf \
    MODELS_DIR=/app/models \
    # llama-cpp settings
    LLAMA_N_CTX=2048 \
    LLAMA_N_THREADS=4 \
    LLAMA_N_GPU_LAYERS=0 \
    # Telemetry
    TELEMETRY_DIR=/app/telemetry \
    LOG_LEVEL=INFO \
    # Cloud Run settings
    PORT=8000

# Create non-root user for security
RUN groupadd -r jarvis && useradd -r -g jarvis jarvis

# Install minimal runtime dependencies + GCS client
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean \
    && pip install --no-cache-dir google-cloud-storage

WORKDIR /app

# Copy pre-built wheels and install
COPY --from=builder /build/wheels /wheels
RUN pip install --no-cache-dir /wheels/*.whl && rm -rf /wheels

# Copy application code
COPY jarvis_prime/ /app/jarvis_prime/
COPY run_server.py /app/run_server.py

# Create directories with proper permissions
RUN mkdir -p /app/models /app/telemetry /app/logs && \
    chown -R jarvis:jarvis /app

# Switch to non-root user
USER jarvis

# Health check optimized for Cloud Run (faster intervals)
HEALTHCHECK --interval=10s --timeout=5s --start-period=30s --retries=3 \
    CMD curl -sf http://localhost:${PORT}/health || exit 1

# Expose port (Cloud Run uses PORT env var)
EXPOSE 8000

# Entry point - use the lightweight run_server.py
CMD ["python", "run_server.py", "--host", "0.0.0.0", "--port", "8000"]
