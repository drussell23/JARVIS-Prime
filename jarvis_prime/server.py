"""
JARVIS-Prime Server - Tier-0 Muscle Memory Brain
=================================================

Entry point for running JARVIS-Prime as an OpenAI-compatible API server.

Usage:
    # Start server
    python -m jarvis_prime.server

    # With custom settings
    python -m jarvis_prime.server --port 8000 --models-dir ./models

    # Test endpoint
    curl http://localhost:8000/v1/chat/completions \
        -H "Content-Type: application/json" \
        -d '{"messages": [{"role": "user", "content": "Hello!"}]}'
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
        description="JARVIS-Prime Tier-0 Brain Server",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

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
    parser.add_argument(
        "--models-dir",
        type=str,
        default="./models",
        help="Directory containing model files",
    )
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
    parser.add_argument(
        "--initial-model",
        type=str,
        default=None,
        help="Path to initial model to load",
    )
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

    return parser.parse_args()


async def main():
    """Main entry point"""
    args = parse_args()

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    logger.info("=" * 60)
    logger.info("JARVIS-Prime Tier-0 Brain Server")
    logger.info("=" * 60)
    logger.info(f"Models directory: {args.models_dir}")
    logger.info(f"Telemetry directory: {args.telemetry_dir}")
    logger.info(f"Reactor-core watch: {args.reactor_core_dir or 'disabled'}")
    logger.info(f"Host: {args.host}:{args.port}")

    try:
        # Import components
        from jarvis_prime.core.model_manager import PrimeModelManager, create_api_app

        # Create manager
        manager = PrimeModelManager(
            models_dir=args.models_dir,
            telemetry_dir=args.telemetry_dir,
            reactor_core_watch_dir=args.reactor_core_dir,
        )

        # Start manager
        initial_model = Path(args.initial_model) if args.initial_model else None
        await manager.start(initial_model_path=initial_model)

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

        logger.info("Starting server...")
        await server.serve()

    except ImportError as e:
        logger.error(f"Missing dependency: {e}")
        logger.error("Install with: pip install fastapi uvicorn")
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
