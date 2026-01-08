"""
JARVIS-Prime Server - Tier-0 Muscle Memory Brain
=================================================

v80.0 - M1/Apple Silicon Optimized with Metal GPU Acceleration

Entry point for running JARVIS-Prime as an OpenAI-compatible API server.
Uses llama-cpp-python with GGUF models for efficient local inference.

Usage:
    # Start server (auto-detects Metal GPU)
    python -m jarvis_prime.server

    # With custom settings
    python -m jarvis_prime.server --port 8000 --models-dir ./models

    # Use specific executor backend
    python -m jarvis_prime.server --executor llama-cpp

    # CPU-only mode (no GPU)
    python -m jarvis_prime.server --cpu-only

    # Test endpoint
    curl http://localhost:8000/v1/chat/completions \
        -H "Content-Type: application/json" \
        -d '{"messages": [{"role": "user", "content": "Hello!"}]}'

Hardware Detection:
    - Apple Silicon (M1/M2/M3/M4): Full Metal GPU acceleration
    - NVIDIA GPU: CUDA acceleration
    - CPU: Optimized multi-threaded inference
"""

import argparse
import asyncio
import logging
import os
import signal
import sys
from pathlib import Path
from typing import Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="JARVIS-Prime Tier-0 Brain Server (M1/Metal Optimized)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Server settings
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host to bind to",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to listen on",
    )

    # Model settings
    parser.add_argument(
        "--models-dir",
        type=str,
        default=None,
        help="Directory containing model files (default: ~/.jarvis/prime/models)",
    )
    parser.add_argument(
        "--initial-model",
        type=str,
        default=None,
        help="Path to initial model to load (default: auto-detect)",
    )

    # Executor settings
    parser.add_argument(
        "--executor",
        type=str,
        choices=["llama-cpp", "transformers", "auto"],
        default="llama-cpp",
        help="Model executor backend",
    )
    parser.add_argument(
        "--cpu-only",
        action="store_true",
        help="Disable GPU acceleration (CPU only)",
    )
    parser.add_argument(
        "--n-gpu-layers",
        type=int,
        default=-1,
        help="Number of layers to offload to GPU (-1 = all)",
    )
    parser.add_argument(
        "--context-size",
        type=int,
        default=4096,
        help="Context window size",
    )

    # Monitoring
    parser.add_argument(
        "--telemetry-dir",
        type=str,
        default="./telemetry",
        help="Directory for telemetry logs",
    )
    parser.add_argument(
        "--reactor-core-dir",
        type=str,
        default=None,
        help="Directory to watch for reactor-core model updates",
    )

    # Server options
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of worker processes",
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable auto-reload for development",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging",
    )

    # Auto-download
    parser.add_argument(
        "--auto-download",
        action="store_true",
        help="Auto-download recommended model if not found",
    )

    return parser.parse_args()


async def main():
    """Main entry point"""
    args = parse_args()

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    # Default models directory
    models_dir = args.models_dir
    if models_dir is None:
        models_dir = str(Path.home() / ".jarvis" / "prime" / "models")

    # Print banner
    logger.info("=" * 70)
    logger.info("JARVIS-Prime Tier-0 Brain Server")
    logger.info("v80.0 - M1/Apple Silicon Optimized")
    logger.info("=" * 70)

    # Detect hardware
    try:
        from jarvis_prime.core.llama_cpp_executor import HardwareDetector, LlamaCppConfig
        hw = HardwareDetector.detect()
        logger.info(f"Hardware: {hw.gpu_name or 'CPU'}")
        logger.info(f"Backend: {hw.backend.name}")
        logger.info(f"Memory: {hw.total_memory_gb:.1f} GB")
        if hw.metal_supported:
            logger.info("Metal GPU: Enabled")
    except Exception as e:
        logger.warning(f"Hardware detection failed: {e}")
        hw = None

    logger.info("-" * 70)
    logger.info(f"Models directory: {models_dir}")
    logger.info(f"Executor: {args.executor}")
    logger.info(f"GPU Layers: {'CPU only' if args.cpu_only else args.n_gpu_layers}")
    logger.info(f"Context Size: {args.context_size}")
    logger.info(f"Host: {args.host}:{args.port}")
    logger.info("-" * 70)

    try:
        # Import components
        from jarvis_prime.core.model_manager import PrimeModelManager, create_api_app

        # Select executor based on args
        executor_class = None
        executor_config = None

        if args.executor == "llama-cpp":
            from jarvis_prime.core.llama_cpp_executor import (
                LlamaCppExecutor,
                LlamaCppConfig,
                GGUFModelDownloader,
                get_default_model_path,
            )

            # Configure for hardware
            if args.cpu_only:
                executor_config = LlamaCppConfig.for_cpu(args.context_size)
            else:
                executor_config = LlamaCppConfig.auto_detect(args.context_size)
                executor_config.n_gpu_layers = args.n_gpu_layers

            executor_class = LlamaCppExecutor
            logger.info(f"Using LlamaCppExecutor (n_gpu_layers={executor_config.n_gpu_layers})")

            # Find or download model
            initial_model = args.initial_model
            if initial_model is None:
                # Look for existing model
                model_path = get_default_model_path()
                if model_path.exists():
                    initial_model = str(model_path)
                    logger.info(f"Found model: {model_path.name}")
                elif args.auto_download:
                    logger.info("No model found, downloading recommended model...")
                    downloader = GGUFModelDownloader(models_dir=Path(models_dir))
                    recommended = downloader.get_recommended_model()
                    if recommended:
                        model_path = await downloader.download(
                            f"{recommended.repo_id}/{recommended.filename}"
                        )
                        initial_model = str(model_path)
                    else:
                        logger.error("No suitable model found for your hardware")
                        logger.error("Run: python -m jarvis_prime.scripts.download_brain")
                        sys.exit(1)
                else:
                    logger.warning("No model found. Run with --auto-download or:")
                    logger.warning("  python -m jarvis_prime.scripts.download_brain")

        # Create manager with selected executor
        manager = PrimeModelManager(
            models_dir=models_dir,
            telemetry_dir=args.telemetry_dir,
            reactor_core_watch_dir=args.reactor_core_dir,
            executor_class=executor_class,
        )

        # Start manager
        model_path = Path(initial_model) if initial_model else None
        await manager.start(initial_model_path=model_path)

        # Create FastAPI app
        app = create_api_app(manager)

        # Run with uvicorn
        import uvicorn

        config = uvicorn.Config(
            app,
            host=args.host,
            port=args.port,
            workers=args.workers,
            reload=args.reload,
            log_level="debug" if args.debug else "info",
        )

        server = uvicorn.Server(config)

        # Handle shutdown
        loop = asyncio.get_event_loop()

        def signal_handler():
            logger.info("Shutdown signal received")
            loop.create_task(shutdown(manager, server))

        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, signal_handler)

        logger.info("=" * 70)
        logger.info("Server starting...")
        logger.info(f"API endpoint: http://{args.host}:{args.port}/v1/chat/completions")
        logger.info("=" * 70)
        await server.serve()

    except ImportError as e:
        logger.error(f"Missing dependency: {e}")
        logger.error("Install with: pip install fastapi uvicorn llama-cpp-python")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Server error: {e}")
        raise


async def shutdown(manager, server):
    """Graceful shutdown"""
    logger.info("Shutting down...")
    await manager.stop()
    server.should_exit = True


if __name__ == "__main__":
    asyncio.run(main())
