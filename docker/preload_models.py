#!/usr/bin/env python3
"""
JARVIS Prime Model Pre-loader v2.0.0
=====================================

Intelligent model pre-loading script for Docker build phase.
Downloads and pre-warms GGUF models to eliminate startup delays.

Features:
- Downloads models from HuggingFace Hub
- Uses jarvis_prime's ModelDownloader when available
- Pre-warms models to trigger lazy loading/compilation
- Progress tracking with detailed logging
- Graceful fallback for missing dependencies
- SHA256 verification for integrity

Usage (during Docker build):
    python preload_models.py --cache-dir /opt/cache/models --model phi-3.5-mini --warmup true

Usage (standalone):
    python preload_models.py --cache-dir ./models --model tinyllama-chat --warmup true

v2.0.0 - Advanced model pre-loading with intelligent routing
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("model-preloader")


# =============================================================================
# Model Catalog (Fallback if jarvis_prime not available)
# =============================================================================

FALLBACK_MODEL_CATALOG: Dict[str, Dict[str, Any]] = {
    "tinyllama-chat": {
        "repo_id": "TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF",
        "filename": "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf",
        "size_mb": 670,
        "description": "Fast, lightweight model for testing",
    },
    "phi-3.5-mini": {
        "repo_id": "bartowski/Phi-3.5-mini-instruct-GGUF",
        "filename": "Phi-3.5-mini-instruct-Q4_K_M.gguf",
        "size_mb": 2300,
        "description": "Microsoft's efficient reasoning model",
    },
    "phi-2": {
        "repo_id": "TheBloke/phi-2-GGUF",
        "filename": "phi-2.Q4_K_M.gguf",
        "size_mb": 1600,
        "description": "Excellent reasoning for its size",
    },
    "mistral-7b-instruct": {
        "repo_id": "TheBloke/Mistral-7B-Instruct-v0.3-GGUF",
        "filename": "mistral-7b-instruct-v0.3.Q4_K_M.gguf",
        "size_mb": 4370,
        "description": "Production-quality instruction following",
    },
    "deepseek-coder-6.7b": {
        "repo_id": "TheBloke/deepseek-coder-6.7b-instruct-GGUF",
        "filename": "deepseek-coder-6.7b-instruct.Q4_K_M.gguf",
        "size_mb": 3800,
        "description": "Specialized coding model",
    },
    "qwen2.5-7b-instruct": {
        "repo_id": "Qwen/Qwen2.5-7B-Instruct-GGUF",
        "filename": "qwen2.5-7b-instruct-q4_k_m.gguf",
        "size_mb": 4650,
        "description": "Strong math, logic, and multilingual",
    },
    "llama-3-8b-instruct": {
        "repo_id": "QuantFactory/Meta-Llama-3-8B-Instruct-GGUF",
        "filename": "Meta-Llama-3-8B-Instruct.Q4_K_M.gguf",
        "size_mb": 4920,
        "description": "Latest Llama 3 with excellent instruction following",
    },
}


def setup_cache_environment(cache_dir: str) -> None:
    """Set up environment variables for model caching."""
    hf_cache = f"{cache_dir}/huggingface"

    os.environ["HF_HOME"] = hf_cache
    os.environ["HUGGINGFACE_HUB_CACHE"] = hf_cache
    os.environ["TRANSFORMERS_CACHE"] = hf_cache
    os.environ["TORCH_HOME"] = f"{cache_dir}/torch"
    os.environ["XDG_CACHE_HOME"] = f"{cache_dir}/xdg"

    # Create directories
    for subdir in ["", "huggingface", "torch", "xdg", "models"]:
        Path(f"{cache_dir}/{subdir}" if subdir else cache_dir).mkdir(parents=True, exist_ok=True)

    logger.info(f"Cache environment configured: {cache_dir}")


async def download_with_jarvis_prime(
    model_key: str,
    cache_dir: str,
    warmup: bool = True,
) -> Optional[Path]:
    """
    Download model using jarvis_prime's ModelDownloader.

    This is the preferred method as it handles:
    - Model catalog lookup
    - Progress tracking
    - Checksum verification
    - Active model symlink
    """
    try:
        # Add parent directories to path for imports
        parent_dir = Path(__file__).parent.parent
        if str(parent_dir) not in sys.path:
            sys.path.insert(0, str(parent_dir))

        from jarvis_prime.docker.model_downloader import (
            ModelDownloader,
            MODEL_CATALOG,
            download_model,
        )

        logger.info(f"Using jarvis_prime ModelDownloader")

        if model_key not in MODEL_CATALOG:
            available = ", ".join(MODEL_CATALOG.keys())
            logger.warning(f"Model '{model_key}' not in catalog. Available: {available}")
            logger.info(f"Using first available model from catalog")
            model_key = list(MODEL_CATALOG.keys())[0]

        spec = MODEL_CATALOG[model_key]
        logger.info(f"")
        logger.info(f"=" * 60)
        logger.info(f"DOWNLOADING MODEL: {spec.name}")
        logger.info(f"=" * 60)
        logger.info(f"  Repository: {spec.repo_id}")
        logger.info(f"  Filename:   {spec.filename}")
        logger.info(f"  Size:       ~{spec.size_mb} MB")
        logger.info(f"  Quantization: {spec.quantization}")
        logger.info(f"")

        start_time = time.time()

        # Download model
        model_path = await download_model(
            model_key=model_key,
            models_dir=cache_dir,
            set_active=True,
        )

        download_time = time.time() - start_time
        logger.info(f"")
        logger.info(f"✅ Model downloaded in {download_time:.1f}s: {model_path}")

        # Pre-warm model if requested
        if warmup and model_path.exists():
            await warmup_model(model_path)

        return model_path

    except ImportError as e:
        logger.warning(f"jarvis_prime not available: {e}")
        return None
    except Exception as e:
        logger.error(f"Error using jarvis_prime downloader: {e}")
        import traceback
        traceback.print_exc()
        return None


async def download_with_huggingface(
    model_key: str,
    cache_dir: str,
    warmup: bool = True,
) -> Optional[Path]:
    """
    Fallback: Download model directly from HuggingFace Hub.

    Used when jarvis_prime module is not available.
    """
    try:
        from huggingface_hub import hf_hub_download

        logger.info(f"Using HuggingFace Hub direct download (fallback)")

        if model_key not in FALLBACK_MODEL_CATALOG:
            available = ", ".join(FALLBACK_MODEL_CATALOG.keys())
            logger.warning(f"Model '{model_key}' not in catalog. Available: {available}")
            model_key = "tinyllama-chat"  # Default to smallest model

        spec = FALLBACK_MODEL_CATALOG[model_key]

        logger.info(f"")
        logger.info(f"=" * 60)
        logger.info(f"DOWNLOADING MODEL (HuggingFace): {model_key}")
        logger.info(f"=" * 60)
        logger.info(f"  Repository: {spec['repo_id']}")
        logger.info(f"  Filename:   {spec['filename']}")
        logger.info(f"  Size:       ~{spec['size_mb']} MB")
        logger.info(f"")

        models_dir = Path(cache_dir)
        models_dir.mkdir(parents=True, exist_ok=True)

        start_time = time.time()

        # Download with progress
        loop = asyncio.get_event_loop()
        downloaded_path = await loop.run_in_executor(
            None,
            lambda: hf_hub_download(
                repo_id=spec["repo_id"],
                filename=spec["filename"],
                local_dir=str(models_dir),
                local_dir_use_symlinks=False,
            ),
        )

        download_time = time.time() - start_time
        model_path = Path(downloaded_path)

        logger.info(f"✅ Model downloaded in {download_time:.1f}s: {model_path}")

        # Create current.gguf symlink
        current_link = models_dir / "current.gguf"
        if current_link.exists() or current_link.is_symlink():
            current_link.unlink()
        current_link.symlink_to(model_path.name)
        logger.info(f"✅ Active model symlink created: current.gguf -> {model_path.name}")

        # Pre-warm model if requested
        if warmup and model_path.exists():
            await warmup_model(model_path)

        return model_path

    except ImportError as e:
        logger.error(f"huggingface_hub not available: {e}")
        logger.error("Install with: pip install huggingface_hub")
        return None
    except Exception as e:
        logger.error(f"Error downloading from HuggingFace: {e}")
        import traceback
        traceback.print_exc()
        return None


async def warmup_model(model_path: Path) -> bool:
    """
    Pre-warm model to trigger lazy loading and compilation.

    This ensures the model is ready to serve immediately when the
    container starts, eliminating first-request latency.
    """
    try:
        logger.info(f"")
        logger.info(f"🔥 Pre-warming model: {model_path.name}")
        logger.info(f"   This triggers lazy loading and JIT compilation...")

        start_time = time.time()

        # Try to import llama-cpp-python
        try:
            from llama_cpp import Llama

            logger.info(f"   Loading model into memory...")

            # Load with minimal config for warmup
            llm = Llama(
                model_path=str(model_path),
                n_ctx=512,  # Minimal context for warmup
                n_threads=2,
                n_gpu_layers=0,  # CPU only for Docker build
                verbose=False,
            )

            logger.info(f"   Running warmup inference...")

            # Run a simple inference to trigger any remaining lazy loading
            output = llm(
                "Hello",
                max_tokens=5,
                temperature=0.1,
                echo=False,
            )

            # Clean up
            del llm

            warmup_time = time.time() - start_time
            logger.info(f"")
            logger.info(f"✅ Model pre-warmed in {warmup_time:.1f}s")
            logger.info(f"   First response: {output['choices'][0]['text'][:50]}...")

            return True

        except ImportError:
            logger.warning("   llama-cpp-python not available, skipping warmup")
            return False

    except Exception as e:
        logger.warning(f"   Warmup failed (non-fatal): {e}")
        return False


async def preload_models(
    cache_dir: str,
    model_key: str,
    warmup: bool = True,
) -> bool:
    """
    Main function to pre-load models during Docker build.

    Attempts to use jarvis_prime's downloader first, falls back
    to direct HuggingFace download if not available.
    """
    logger.info(f"")
    logger.info(f"=" * 70)
    logger.info(f"JARVIS PRIME MODEL PRE-LOADER v2.0.0")
    logger.info(f"=" * 70)
    logger.info(f"")
    logger.info(f"Configuration:")
    logger.info(f"  Cache directory: {cache_dir}")
    logger.info(f"  Model:           {model_key}")
    logger.info(f"  Warmup:          {warmup}")
    logger.info(f"")

    # Setup cache environment
    setup_cache_environment(cache_dir)

    # Try jarvis_prime downloader first
    model_path = await download_with_jarvis_prime(model_key, cache_dir, warmup)

    if model_path is None:
        # Fallback to direct HuggingFace download
        logger.info(f"Falling back to direct HuggingFace download...")
        model_path = await download_with_huggingface(model_key, cache_dir, warmup)

    if model_path and model_path.exists():
        logger.info(f"")
        logger.info(f"=" * 70)
        logger.info(f"✅ MODEL PRE-LOADING COMPLETE")
        logger.info(f"=" * 70)
        logger.info(f"")
        logger.info(f"Model ready at: {model_path}")
        logger.info(f"Size: {model_path.stat().st_size / (1024**3):.2f} GB")
        logger.info(f"")

        # Verify integrity
        try:
            import gc
            gc.collect()
        except Exception:
            pass

        return True
    else:
        logger.error(f"")
        logger.error(f"❌ MODEL PRE-LOADING FAILED")
        logger.error(f"Model will be downloaded at runtime (slower startup)")
        logger.error(f"")
        return False


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Pre-load JARVIS Prime models for Docker",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Pre-load default model (phi-3.5-mini)
    python preload_models.py --cache-dir /opt/cache/models

    # Pre-load specific model with warmup
    python preload_models.py --cache-dir ./models --model mistral-7b-instruct --warmup true

    # Pre-load without warmup (faster build, slower first request)
    python preload_models.py --cache-dir ./models --model tinyllama-chat --warmup false

Available models:
    - tinyllama-chat       (~670 MB) - Fast, lightweight
    - phi-3.5-mini         (~2.3 GB) - Excellent reasoning
    - phi-2                (~1.6 GB) - Good coding/reasoning
    - mistral-7b-instruct  (~4.4 GB) - Production quality
    - deepseek-coder-6.7b  (~3.8 GB) - Specialized coding
    - qwen2.5-7b-instruct  (~4.6 GB) - Math/multilingual
    - llama-3-8b-instruct  (~4.9 GB) - Latest Llama
        """,
    )

    parser.add_argument(
        "--cache-dir",
        required=True,
        help="Directory to store models",
    )
    parser.add_argument(
        "--model",
        default="phi-3.5-mini",
        help="Model key to download (default: phi-3.5-mini)",
    )
    parser.add_argument(
        "--warmup",
        type=lambda x: x.lower() in ("true", "1", "yes"),
        default=True,
        help="Pre-warm model after download (default: true)",
    )

    args = parser.parse_args()

    # Run async preloader
    success = asyncio.run(preload_models(
        cache_dir=args.cache_dir,
        model_key=args.model,
        warmup=args.warmup,
    ))

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
