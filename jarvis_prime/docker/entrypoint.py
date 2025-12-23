"""
Docker Entrypoint - Container Command Dispatcher
=================================================

Entry point for JARVIS-Prime Docker container with support for:
- serve: Run the API server (default)
- download: Download models from HuggingFace
- convert: Convert HF models to GGUF
- shell: Interactive Python shell

Usage:
    docker run jarvis-prime:latest serve              # Start server
    docker run jarvis-prime:latest download           # Download default model
    docker run jarvis-prime:latest download --repo X  # Download specific model
    docker run jarvis-prime:latest convert --input X  # Convert model to GGUF
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import signal
import sys
from pathlib import Path
from typing import List, Optional

# Configure logging
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("jarvis-prime.entrypoint")


# ============================================================================
# Serve Command
# ============================================================================

async def cmd_serve(args: argparse.Namespace) -> int:
    """Run the JARVIS-Prime API server."""
    from jarvis_prime.docker.llama_server_executor import LlamaServerExecutor
    from jarvis_prime.core.model_manager import PrimeModelManager, create_api_app

    logger.info("=" * 60)
    logger.info("JARVIS-Prime Tier-0 Brain Server")
    logger.info("=" * 60)

    # Get config from environment
    host = os.getenv("JARVIS_PRIME_HOST", "0.0.0.0")
    port = int(os.getenv("JARVIS_PRIME_PORT", "8000"))
    model_path = Path(os.getenv("MODEL_PATH", "/app/models/current.gguf"))
    telemetry_dir = Path(os.getenv("TELEMETRY_DIR", "/app/telemetry"))
    reactor_watch = os.getenv("REACTOR_CORE_WATCH_DIR")

    # Verify model exists
    if not model_path.exists():
        logger.error(f"Model not found: {model_path}")
        logger.info("Run 'download' command first to get a model")
        return 1

    logger.info(f"Model: {model_path}")
    logger.info(f"Host: {host}:{port}")
    logger.info(f"Telemetry: {telemetry_dir}")

    try:
        # Import and patch model executor to use Docker-optimized version
        from jarvis_prime.docker.llama_server_executor import LlamaServerExecutor

        # Create custom executor class for PrimeModelManager
        class DockerLlamaExecutor:
            """Adapter for PrimeModelManager using LlamaServerExecutor."""

            def __init__(self):
                self._executor = LlamaServerExecutor()

            async def load(self, model_path: Path, **kwargs):
                await self._executor.load(model_path, **kwargs)

            async def unload(self):
                await self._executor.unload()

            async def validate(self) -> bool:
                return await self._executor.validate()

            def is_loaded(self) -> bool:
                return self._executor.is_loaded()

            async def generate(self, prompt: str, max_tokens: int = 2048, temperature: float = 0.7, **kwargs) -> str:
                return await self._executor.generate(prompt, max_tokens, temperature, **kwargs)

            async def generate_stream(self, prompt: str, max_tokens: int = 2048, temperature: float = 0.7, **kwargs):
                async for token in self._executor.generate_stream(prompt, max_tokens, temperature, **kwargs):
                    yield token

        # Create manager with Docker executor
        manager = PrimeModelManager(
            models_dir=model_path.parent,
            telemetry_dir=telemetry_dir,
            reactor_core_watch_dir=reactor_watch,
            executor_class=DockerLlamaExecutor,
        )

        # Start manager
        await manager.start(initial_model_path=model_path)

        # Create FastAPI app
        app = create_api_app(manager)

        # Run with uvicorn
        import uvicorn

        config = uvicorn.Config(
            app,
            host=host,
            port=port,
            log_level="info",
            access_log=True,
        )

        server = uvicorn.Server(config)

        # Handle shutdown
        loop = asyncio.get_event_loop()
        shutdown_event = asyncio.Event()

        def signal_handler():
            logger.info("Shutdown signal received")
            shutdown_event.set()

        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, signal_handler)

        logger.info("Starting server...")

        # Start server task
        server_task = asyncio.create_task(server.serve())

        # Wait for shutdown signal
        await shutdown_event.wait()

        # Graceful shutdown
        logger.info("Shutting down...")
        await manager.stop()
        server.should_exit = True
        await server_task

        return 0

    except ImportError as e:
        logger.error(f"Missing dependency: {e}")
        return 1
    except Exception as e:
        logger.error(f"Server error: {e}")
        raise


# ============================================================================
# Download Command
# ============================================================================

async def cmd_download(args: argparse.Namespace) -> int:
    """Download a model from HuggingFace."""
    from jarvis_prime.docker.model_downloader import (
        ModelDownloader,
        MODEL_CATALOG,
        recommend_model,
    )

    models_dir = Path(os.getenv("MODEL_PATH", "/app/models")).parent

    downloader = ModelDownloader(
        models_dir=models_dir,
        hf_token=os.getenv("HF_TOKEN"),
    )

    try:
        if args.list:
            # List available models
            print("\nAvailable models in catalog:\n")
            for key, spec in MODEL_CATALOG.items():
                print(f"  {key}")
                print(f"    Name: {spec.name}")
                print(f"    Size: {spec.size_mb} MB")
                print(f"    Desc: {spec.description}")
                print(f"    Good for: {', '.join(spec.recommended_for)}")
                print()
            return 0

        if args.recommend:
            # Recommend a model
            max_mem = float(args.max_memory or 10.0)
            use_case = args.use_case or "balanced"
            recommended = recommend_model(use_case=use_case, max_memory_gb=max_mem)
            if recommended:
                print(f"\nRecommended model: {recommended}")
                print(f"  For: {use_case}")
                print(f"  Memory limit: {max_mem} GB")
                spec = MODEL_CATALOG[recommended]
                print(f"  Size: {spec.size_mb} MB")
                print(f"  Description: {spec.description}")
            else:
                print("No suitable model found for your constraints")
            return 0

        # Download model
        if args.catalog:
            # Download from catalog
            logger.info(f"Downloading catalog model: {args.catalog}")
            path = await downloader.download_catalog_model(
                model_key=args.catalog,
                force=args.force,
            )
        elif args.repo and args.file:
            # Download custom model
            logger.info(f"Downloading {args.file} from {args.repo}")
            path = await downloader.download(
                repo_id=args.repo,
                filename=args.file,
                force=args.force,
            )
        else:
            # Download default (TinyLlama for testing)
            logger.info("Downloading default model (tinyllama-chat)")
            path = await downloader.download_catalog_model("tinyllama-chat")

        # Set as active
        if args.set_active or args.catalog or (not args.repo):
            model_name = args.catalog or "tinyllama-chat"
            await downloader.set_active_model(model_name)
            logger.info(f"Set {model_name} as active model")

        logger.info(f"Model downloaded to: {path}")
        return 0

    except Exception as e:
        logger.error(f"Download failed: {e}")
        return 1


# ============================================================================
# Convert Command
# ============================================================================

async def cmd_convert(args: argparse.Namespace) -> int:
    """Convert HuggingFace model to GGUF format."""
    import subprocess

    llama_cpp = Path(os.getenv("LLAMA_CPP_PATH", "/app/llama.cpp"))
    convert_script = llama_cpp / "convert_hf_to_gguf.py"
    quantize_bin = llama_cpp / "llama-quantize"

    input_path = Path(args.input)
    output_path = Path(args.output)
    quantize_method = args.quantize

    if not input_path.exists():
        logger.error(f"Input model not found: {input_path}")
        return 1

    if not convert_script.exists():
        logger.error(f"Convert script not found: {convert_script}")
        return 1

    try:
        # Step 1: Convert to F16 GGUF
        f16_output = output_path.with_suffix(".f16.gguf")
        logger.info(f"Converting {input_path} to F16 GGUF...")

        result = subprocess.run(
            [
                "python3",
                str(convert_script),
                str(input_path),
                "--outfile", str(f16_output),
                "--outtype", "f16",
            ],
            capture_output=True,
            text=True,
        )

        if result.returncode != 0:
            logger.error(f"Conversion failed: {result.stderr}")
            return 1

        # Step 2: Quantize
        if quantize_method.lower() != "f16":
            logger.info(f"Quantizing to {quantize_method}...")

            result = subprocess.run(
                [
                    str(quantize_bin),
                    str(f16_output),
                    str(output_path),
                    quantize_method,
                ],
                capture_output=True,
                text=True,
            )

            if result.returncode != 0:
                logger.error(f"Quantization failed: {result.stderr}")
                return 1

            # Remove F16 intermediate file
            f16_output.unlink()
        else:
            # Just rename F16 file
            f16_output.rename(output_path)

        logger.info(f"Conversion complete: {output_path}")
        return 0

    except Exception as e:
        logger.error(f"Conversion error: {e}")
        return 1


# ============================================================================
# Health Command
# ============================================================================

async def cmd_health(args: argparse.Namespace) -> int:
    """Check server health."""
    import httpx

    host = os.getenv("JARVIS_PRIME_HOST", "localhost")
    port = int(os.getenv("JARVIS_PRIME_PORT", "8000"))

    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(f"http://{host}:{port}/health", timeout=10)

            if response.status_code == 200:
                print("Server is healthy")
                print(response.json())
                return 0
            else:
                print(f"Server unhealthy: {response.status_code}")
                return 1

    except Exception as e:
        print(f"Health check failed: {e}")
        return 1


# ============================================================================
# Main Entry Point
# ============================================================================

def main(argv: Optional[List[str]] = None) -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="JARVIS-Prime Docker Entrypoint",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Commands:
  serve     Start the API server (default)
  download  Download models from HuggingFace
  convert   Convert HF models to GGUF format
  health    Check server health

Examples:
  jarvis-prime serve
  jarvis-prime download --catalog mistral-7b-instruct
  jarvis-prime download --list
  jarvis-prime convert --input ./hf-model --output ./model.gguf
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Serve command
    serve_parser = subparsers.add_parser("serve", help="Start API server")

    # Download command
    download_parser = subparsers.add_parser("download", help="Download models")
    download_parser.add_argument("--catalog", help="Model key from catalog")
    download_parser.add_argument("--repo", help="HuggingFace repo ID")
    download_parser.add_argument("--file", help="Model filename")
    download_parser.add_argument("--list", action="store_true", help="List catalog models")
    download_parser.add_argument("--recommend", action="store_true", help="Get model recommendation")
    download_parser.add_argument("--use-case", choices=["testing", "coding", "production", "balanced"])
    download_parser.add_argument("--max-memory", help="Max memory in GB")
    download_parser.add_argument("--force", action="store_true", help="Re-download if exists")
    download_parser.add_argument("--set-active", action="store_true", help="Set as active model")

    # Convert command
    convert_parser = subparsers.add_parser("convert", help="Convert to GGUF")
    convert_parser.add_argument("--input", required=True, help="Input HF model path")
    convert_parser.add_argument("--output", required=True, help="Output GGUF path")
    convert_parser.add_argument("--quantize", default="q4_k_m", help="Quantization method")

    # Health command
    health_parser = subparsers.add_parser("health", help="Check server health")

    args = parser.parse_args(argv)

    # Default to serve
    if not args.command:
        args.command = "serve"

    # Dispatch command
    if args.command == "serve":
        return asyncio.run(cmd_serve(args))
    elif args.command == "download":
        return asyncio.run(cmd_download(args))
    elif args.command == "convert":
        return asyncio.run(cmd_convert(args))
    elif args.command == "health":
        return asyncio.run(cmd_health(args))
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())
