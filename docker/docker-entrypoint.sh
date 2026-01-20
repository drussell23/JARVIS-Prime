#!/bin/bash
# =============================================================================
# JARVIS Prime Docker Entrypoint v2.0.0
# =============================================================================
#
# Intelligent entrypoint script for JARVIS Prime container.
#
# Features:
# - Model cache verification
# - Pre-warming models on startup
# - Environment variable configuration
# - Graceful shutdown handling
# - Health check readiness
#
# v2.0.0 - Advanced entrypoint with model pre-warming
# =============================================================================

set -e

echo ""
echo "============================================================"
echo "🚀 JARVIS Prime Docker Entrypoint v2.0.0"
echo "============================================================"
echo ""

# =============================================================================
# Configuration from environment
# =============================================================================
MODEL_CACHE_DIR="${MODEL_CACHE_DIR:-/opt/jarvis_prime_cache}"
MODEL_PATH="${MODEL_PATH:-/opt/jarvis_prime_cache/models/current.gguf}"
WARMUP_ON_START="${WARMUP_ON_START:-true}"
PORT="${PORT:-8000}"
HOST="${HOST:-0.0.0.0}"
LOG_LEVEL="${LOG_LEVEL:-INFO}"

echo "📋 Configuration:"
echo "   MODEL_CACHE_DIR: ${MODEL_CACHE_DIR}"
echo "   MODEL_PATH:      ${MODEL_PATH}"
echo "   WARMUP_ON_START: ${WARMUP_ON_START}"
echo "   PORT:            ${PORT}"
echo "   HOST:            ${HOST}"
echo "   LOG_LEVEL:       ${LOG_LEVEL}"
echo ""

# =============================================================================
# Verify model cache directory
# =============================================================================
echo "📂 Checking model cache directory..."

if [ -d "${MODEL_CACHE_DIR}" ]; then
    echo "   ✅ Cache directory exists"
    CACHE_SIZE=$(du -sh "${MODEL_CACHE_DIR}" 2>/dev/null | cut -f1 || echo "unknown")
    echo "   📊 Cache size: ${CACHE_SIZE}"
else
    echo "   ⚠️ Cache directory not found, creating..."
    mkdir -p "${MODEL_CACHE_DIR}/models"
    mkdir -p "${MODEL_CACHE_DIR}/huggingface"
    mkdir -p "${MODEL_CACHE_DIR}/torch"
    mkdir -p "${MODEL_CACHE_DIR}/xdg"
fi
echo ""

# =============================================================================
# Verify model file
# =============================================================================
echo "🔍 Checking model file..."

if [ -f "${MODEL_PATH}" ]; then
    MODEL_SIZE=$(ls -lh "${MODEL_PATH}" 2>/dev/null | awk '{print $5}' || echo "unknown")
    MODEL_NAME=$(basename "${MODEL_PATH}")
    echo "   ✅ Model found: ${MODEL_NAME}"
    echo "   📊 Model size: ${MODEL_SIZE}"
elif [ -L "${MODEL_PATH}" ]; then
    # It's a symlink, check if target exists
    SYMLINK_TARGET=$(readlink -f "${MODEL_PATH}" 2>/dev/null || echo "broken")
    if [ -f "${SYMLINK_TARGET}" ]; then
        MODEL_SIZE=$(ls -lh "${SYMLINK_TARGET}" 2>/dev/null | awk '{print $5}' || echo "unknown")
        MODEL_NAME=$(basename "${SYMLINK_TARGET}")
        echo "   ✅ Model symlink found: current.gguf -> ${MODEL_NAME}"
        echo "   📊 Model size: ${MODEL_SIZE}"
    else
        echo "   ⚠️ Model symlink broken, will download at runtime"
    fi
else
    echo "   ⚠️ Model not found at ${MODEL_PATH}"
    echo "   ℹ️ Model will be downloaded at runtime (slower startup)"

    # Check if there are any .gguf files in the models directory
    MODELS_DIR="${MODEL_CACHE_DIR}/models"
    if [ -d "${MODELS_DIR}" ]; then
        GGUF_FILES=$(find "${MODELS_DIR}" -name "*.gguf" -type f 2>/dev/null | head -1)
        if [ -n "${GGUF_FILES}" ]; then
            echo "   🔄 Found existing model, creating symlink..."
            FIRST_MODEL=$(basename "${GGUF_FILES}")
            ln -sf "${FIRST_MODEL}" "${MODEL_PATH}" 2>/dev/null || true
            echo "   ✅ Symlink created: current.gguf -> ${FIRST_MODEL}"
        fi
    fi
fi
echo ""

# =============================================================================
# List available models
# =============================================================================
echo "📦 Available models in cache:"
MODELS_DIR="${MODEL_CACHE_DIR}/models"
if [ -d "${MODELS_DIR}" ]; then
    MODEL_COUNT=$(find "${MODELS_DIR}" -name "*.gguf" -type f 2>/dev/null | wc -l | tr -d ' ')
    if [ "${MODEL_COUNT}" -gt 0 ]; then
        find "${MODELS_DIR}" -name "*.gguf" -type f -exec ls -lh {} \; 2>/dev/null | while read -r line; do
            SIZE=$(echo "${line}" | awk '{print $5}')
            NAME=$(echo "${line}" | awk '{print $NF}' | xargs basename)
            echo "   - ${NAME} (${SIZE})"
        done
    else
        echo "   (no models found)"
    fi
else
    echo "   (models directory not found)"
fi
echo ""

# =============================================================================
# Set up environment variables for HuggingFace caching
# =============================================================================
export HF_HOME="${MODEL_CACHE_DIR}/huggingface"
export HUGGINGFACE_HUB_CACHE="${MODEL_CACHE_DIR}/huggingface"
export TRANSFORMERS_CACHE="${MODEL_CACHE_DIR}/huggingface"
export TORCH_HOME="${MODEL_CACHE_DIR}/torch"
export XDG_CACHE_HOME="${MODEL_CACHE_DIR}/xdg"

echo "🌐 Environment configured:"
echo "   HF_HOME:            ${HF_HOME}"
echo "   TRANSFORMERS_CACHE: ${TRANSFORMERS_CACHE}"
echo "   TORCH_HOME:         ${TORCH_HOME}"
echo ""

# =============================================================================
# Optional: Pre-warm model on startup
# =============================================================================
if [ "${WARMUP_ON_START}" = "true" ] && [ -f "${MODEL_PATH}" ]; then
    echo "🔥 Pre-warming model (optional)..."
    echo "   This triggers lazy loading for faster first request..."

    python3 -c "
import os
import sys
import time

# Suppress warnings
import warnings
warnings.filterwarnings('ignore')

try:
    start = time.time()
    print('   Loading llama-cpp-python...')

    from llama_cpp import Llama

    model_path = os.environ.get('MODEL_PATH', '/opt/jarvis_prime_cache/models/current.gguf')

    print(f'   Loading model: {os.path.basename(model_path)}')

    # Load with minimal context for quick warmup
    llm = Llama(
        model_path=model_path,
        n_ctx=256,
        n_threads=2,
        n_gpu_layers=int(os.environ.get('N_GPU_LAYERS', '0')),
        verbose=False,
    )

    print('   Running test inference...')
    output = llm('Hello', max_tokens=3, temperature=0.1, echo=False)

    # Cleanup
    del llm

    elapsed = time.time() - start
    print(f'   ✅ Model pre-warmed in {elapsed:.1f}s')

except ImportError as e:
    print(f'   ⚠️ llama-cpp-python not available: {e}')
    print('   Model will load on first request')
except Exception as e:
    print(f'   ⚠️ Pre-warm failed (non-fatal): {e}')
    print('   Model will load on first request')
" 2>&1 || echo "   ⚠️ Pre-warm skipped (model will load on first request)"
    echo ""
fi

# =============================================================================
# Graceful shutdown handling
# =============================================================================
cleanup() {
    echo ""
    echo "🛑 Shutdown signal received, cleaning up..."
    # Send SIGTERM to child processes
    kill -TERM "${SERVER_PID}" 2>/dev/null || true
    wait "${SERVER_PID}" 2>/dev/null || true
    echo "✅ Shutdown complete"
    exit 0
}

trap cleanup SIGTERM SIGINT SIGQUIT

# =============================================================================
# Start the server
# =============================================================================
echo "============================================================"
echo "🎯 Starting JARVIS Prime server..."
echo "   Command: $@"
echo "   URL: http://${HOST}:${PORT}"
echo "   Health: http://${HOST}:${PORT}/health"
echo "============================================================"
echo ""

# Start server in background and capture PID
exec "$@" &
SERVER_PID=$!

# Wait for server
wait "${SERVER_PID}"
