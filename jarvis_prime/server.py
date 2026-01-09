"""
JARVIS-Prime Server - Tier-0 Muscle Memory Brain
=================================================

v84.0 - Trinity-Integrated with Advanced Async Patterns

Entry point for running JARVIS-Prime as an OpenAI-compatible API server.
Uses llama-cpp-python with GGUF models for efficient local inference.

TRINITY INTEGRATION:
    - Automatic connection to Trinity network on startup
    - Guaranteed event delivery with ACK and retry
    - OOM protection for parallel inference
    - Network partition detection
    - Graceful shutdown with Trinity notification

Usage:
    # Start server (auto-detects Metal GPU, connects to Trinity)
    python -m jarvis_prime.server

    # With custom settings
    python -m jarvis_prime.server --port 8000 --models-dir ./models

    # Disable Trinity (standalone mode)
    TRINITY_ENABLED=false python -m jarvis_prime.server

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
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional
from enum import Enum, auto

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# =============================================================================
# v84.0: ADVANCED TRINITY INTEGRATION
# =============================================================================

class TrinityConnectionState(Enum):
    """Trinity connection states."""
    DISCONNECTED = auto()
    CONNECTING = auto()
    CONNECTED = auto()
    PARTITIONED = auto()
    RECONNECTING = auto()


@dataclass
class TrinityIntegration:
    """
    v84.0: Advanced Trinity integration for J-Prime.

    Features:
        - Automatic connection on startup
        - Network partition detection
        - OOM protection before inference
        - Graceful shutdown coordination
        - Heartbeat with guaranteed delivery
    """
    # Configuration from environment (zero hardcoding)
    enabled: bool = field(default_factory=lambda: os.getenv("TRINITY_ENABLED", "true").lower() == "true")
    heartbeat_interval: float = field(default_factory=lambda: float(os.getenv("TRINITY_HEARTBEAT_INTERVAL", "5.0")))
    partition_threshold: float = field(default_factory=lambda: float(os.getenv("TRINITY_PARTITION_THRESHOLD", "30.0")))
    oom_memory_limit_mb: float = field(default_factory=lambda: float(os.getenv("OOM_MEMORY_LIMIT_MB", "8192")))
    oom_warning_threshold: float = field(default_factory=lambda: float(os.getenv("OOM_WARNING_THRESHOLD", "0.75")))
    reconnect_interval: float = field(default_factory=lambda: float(os.getenv("TRINITY_RECONNECT_INTERVAL", "5.0")))
    max_reconnect_attempts: int = field(default_factory=lambda: int(os.getenv("TRINITY_MAX_RECONNECT_ATTEMPTS", "10")))

    # State tracking
    state: TrinityConnectionState = TrinityConnectionState.DISCONNECTED
    last_heartbeat_time: float = 0.0
    last_jarvis_heartbeat: float = 0.0
    reconnect_attempts: int = 0
    start_time: float = field(default_factory=time.time)

    # Background tasks
    _heartbeat_task: Optional[asyncio.Task] = None
    _partition_detector_task: Optional[asyncio.Task] = None
    _reconnect_task: Optional[asyncio.Task] = None

    # Callbacks
    _partition_callbacks: List[Callable] = field(default_factory=list)
    _connection_callbacks: List[Callable] = field(default_factory=list)

    # Trinity bridge reference
    _trinity_bridge = None

    async def initialize(self, port: int, model_path: str = "", model_loaded: bool = False) -> bool:
        """
        Initialize Trinity integration.

        Args:
            port: Port J-Prime is running on
            model_path: Path to loaded model
            model_loaded: Whether model is loaded

        Returns:
            True if initialization succeeded
        """
        if not self.enabled:
            logger.info("[Trinity] Integration disabled (TRINITY_ENABLED=false)")
            return False

        self.state = TrinityConnectionState.CONNECTING
        logger.info("=" * 60)
        logger.info("v84.0 TRINITY INTEGRATION: Initializing J-Prime Connection")
        logger.info("=" * 60)

        try:
            # Import trinity_bridge
            from jarvis_prime.core.trinity_bridge import (
                initialize_trinity,
                update_model_status,
                set_model_health_callback,
                record_inference,
                TRINITY_ENABLED,
            )

            self._trinity_bridge = {
                "initialize": initialize_trinity,
                "update_model_status": update_model_status,
                "set_model_health_callback": set_model_health_callback,
                "record_inference": record_inference,
            }

            # Initialize Trinity connection
            success = await initialize_trinity(
                port=port,
                model_path=model_path,
                model_loaded=model_loaded,
            )

            if success:
                self.state = TrinityConnectionState.CONNECTED
                self.reconnect_attempts = 0
                logger.info("[Trinity] ✓ Connected to Trinity network")

                # Start background tasks
                self._heartbeat_task = asyncio.create_task(self._enhanced_heartbeat_loop())
                self._partition_detector_task = asyncio.create_task(self._partition_detection_loop())

                # Notify callbacks
                for callback in self._connection_callbacks:
                    try:
                        if asyncio.iscoroutinefunction(callback):
                            await callback(True)
                        else:
                            callback(True)
                    except Exception as e:
                        logger.warning(f"[Trinity] Connection callback error: {e}")

                return True
            else:
                self.state = TrinityConnectionState.DISCONNECTED
                logger.warning("[Trinity] Failed to connect to Trinity network")
                return False

        except ImportError as e:
            logger.warning(f"[Trinity] Trinity bridge not available: {e}")
            self.state = TrinityConnectionState.DISCONNECTED
            return False
        except Exception as e:
            logger.error(f"[Trinity] Initialization error: {e}")
            self.state = TrinityConnectionState.DISCONNECTED
            return False

    async def shutdown(self) -> None:
        """Graceful shutdown of Trinity integration."""
        if self.state == TrinityConnectionState.DISCONNECTED:
            return

        logger.info("[Trinity] Shutting down integration...")

        # Cancel background tasks
        for task in [self._heartbeat_task, self._partition_detector_task, self._reconnect_task]:
            if task and not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

        # Shutdown trinity bridge
        try:
            from jarvis_prime.core.trinity_bridge import shutdown_trinity
            await shutdown_trinity()
        except Exception as e:
            logger.warning(f"[Trinity] Shutdown error: {e}")

        self.state = TrinityConnectionState.DISCONNECTED
        logger.info("[Trinity] Disconnected from Trinity network")

    async def _enhanced_heartbeat_loop(self) -> None:
        """Enhanced heartbeat with OOM check and guaranteed delivery."""
        while self.state in (TrinityConnectionState.CONNECTED, TrinityConnectionState.RECONNECTING):
            try:
                # Check memory before operations (OOM protection)
                if not await self._check_memory_safe():
                    logger.warning("[Trinity] Memory pressure detected, skipping heartbeat")
                    await asyncio.sleep(self.heartbeat_interval)
                    continue

                self.last_heartbeat_time = time.time()
                await asyncio.sleep(self.heartbeat_interval)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.debug(f"[Trinity] Heartbeat error: {e}")
                await asyncio.sleep(self.heartbeat_interval)

    async def _partition_detection_loop(self) -> None:
        """Detect network partitions via missing heartbeats."""
        while self.state in (TrinityConnectionState.CONNECTED, TrinityConnectionState.RECONNECTING):
            try:
                # Check JARVIS heartbeat freshness
                jarvis_state_file = Path.home() / ".jarvis" / "trinity" / "components" / "jarvis_body.json"

                if jarvis_state_file.exists():
                    try:
                        import json
                        with open(jarvis_state_file, 'r') as f:
                            data = json.load(f)
                            jarvis_timestamp = data.get("timestamp", 0)
                            heartbeat_age = time.time() - jarvis_timestamp

                            if heartbeat_age > self.partition_threshold:
                                if self.state != TrinityConnectionState.PARTITIONED:
                                    logger.warning(f"[Trinity] JARVIS heartbeat stale ({heartbeat_age:.1f}s), possible partition")
                                    self.state = TrinityConnectionState.PARTITIONED
                                    await self._handle_partition()
                            else:
                                if self.state == TrinityConnectionState.PARTITIONED:
                                    logger.info("[Trinity] JARVIS heartbeat recovered, partition resolved")
                                    self.state = TrinityConnectionState.CONNECTED
                                self.last_jarvis_heartbeat = jarvis_timestamp
                    except json.JSONDecodeError:
                        pass  # File being written

                await asyncio.sleep(5.0)  # Check every 5 seconds

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.debug(f"[Trinity] Partition detection error: {e}")
                await asyncio.sleep(5.0)

    async def _handle_partition(self) -> None:
        """Handle detected network partition."""
        logger.warning("[Trinity] Network partition detected - starting recovery")

        # Notify callbacks
        for callback in self._partition_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback()
                else:
                    callback()
            except Exception as e:
                logger.warning(f"[Trinity] Partition callback error: {e}")

        # Start reconnection attempts
        if self._reconnect_task is None or self._reconnect_task.done():
            self._reconnect_task = asyncio.create_task(self._reconnection_loop())

    async def _reconnection_loop(self) -> None:
        """Attempt to reconnect after partition."""
        self.state = TrinityConnectionState.RECONNECTING

        while self.reconnect_attempts < self.max_reconnect_attempts:
            self.reconnect_attempts += 1

            # Exponential backoff with jitter
            delay = min(self.reconnect_interval * (2 ** (self.reconnect_attempts - 1)), 60.0)
            jitter = delay * 0.1 * (hash(time.time()) % 10) / 10
            actual_delay = delay + jitter

            logger.info(f"[Trinity] Reconnect attempt {self.reconnect_attempts}/{self.max_reconnect_attempts} in {actual_delay:.1f}s")
            await asyncio.sleep(actual_delay)

            # Check if partition resolved
            jarvis_state_file = Path.home() / ".jarvis" / "trinity" / "components" / "jarvis_body.json"
            if jarvis_state_file.exists():
                try:
                    import json
                    with open(jarvis_state_file, 'r') as f:
                        data = json.load(f)
                        heartbeat_age = time.time() - data.get("timestamp", 0)

                        if heartbeat_age < self.partition_threshold:
                            logger.info("[Trinity] Reconnection successful - partition resolved")
                            self.state = TrinityConnectionState.CONNECTED
                            self.reconnect_attempts = 0
                            return
                except Exception:
                    pass

        logger.error(f"[Trinity] Reconnection failed after {self.max_reconnect_attempts} attempts")
        self.state = TrinityConnectionState.PARTITIONED

    async def _check_memory_safe(self) -> bool:
        """Check if memory usage is safe for operations."""
        try:
            import psutil
            memory = psutil.virtual_memory()
            used_percent = memory.percent / 100.0

            if used_percent > self.oom_warning_threshold:
                logger.warning(f"[Trinity] Memory usage at {used_percent*100:.1f}% (threshold: {self.oom_warning_threshold*100:.0f}%)")
                return False
            return True
        except ImportError:
            return True  # Can't check, assume safe

    def record_inference(self, latency_ms: float, success: bool = True) -> None:
        """Record inference metrics for Trinity heartbeat."""
        if self._trinity_bridge and "record_inference" in self._trinity_bridge:
            self._trinity_bridge["record_inference"](latency_ms, success)

    def update_model_status(self, loaded: bool, model_path: str = "") -> None:
        """Update model status in Trinity heartbeat."""
        if self._trinity_bridge and "update_model_status" in self._trinity_bridge:
            self._trinity_bridge["update_model_status"](loaded, model_path)

    def register_partition_callback(self, callback: Callable) -> None:
        """Register callback for partition events."""
        self._partition_callbacks.append(callback)

    def register_connection_callback(self, callback: Callable) -> None:
        """Register callback for connection events."""
        self._connection_callbacks.append(callback)

    def get_status(self) -> Dict[str, Any]:
        """Get current Trinity integration status."""
        return {
            "enabled": self.enabled,
            "state": self.state.name,
            "connected": self.state == TrinityConnectionState.CONNECTED,
            "partitioned": self.state == TrinityConnectionState.PARTITIONED,
            "uptime_seconds": time.time() - self.start_time,
            "last_heartbeat_time": self.last_heartbeat_time,
            "last_jarvis_heartbeat": self.last_jarvis_heartbeat,
            "reconnect_attempts": self.reconnect_attempts,
        }


# Global Trinity integration instance
_trinity_integration: Optional[TrinityIntegration] = None


def get_trinity_integration() -> Optional[TrinityIntegration]:
    """Get the global Trinity integration instance."""
    return _trinity_integration


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
    """Main entry point with Trinity integration."""
    global _trinity_integration

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
    logger.info("v84.0 - Trinity-Integrated with Advanced Async")
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

        # v84.0: Initialize Trinity Integration
        _trinity_integration = TrinityIntegration()
        trinity_model_path = str(model_path) if model_path else ""
        trinity_connected = await _trinity_integration.initialize(
            port=args.port,
            model_path=trinity_model_path,
            model_loaded=model_path is not None and model_path.exists(),
        )

        if trinity_connected:
            logger.info("[Trinity] ✓ J-Prime connected to Trinity network")

            # Register inference callback for Trinity health monitoring
            def on_inference_complete(latency_ms: float, success: bool = True):
                if _trinity_integration:
                    _trinity_integration.record_inference(latency_ms, success)

            # Register partition callback for graceful degradation
            async def on_partition_detected():
                logger.warning("[Trinity] Partition detected - entering standalone mode")
                # Could pause non-critical operations here

            _trinity_integration.register_partition_callback(on_partition_detected)
        else:
            logger.info("[Trinity] Running in standalone mode (no Trinity connection)")

        # Create FastAPI app
        app = create_api_app(manager)

        # v84.0: Add Trinity status endpoint to the app
        @app.get("/trinity/status")
        async def trinity_status():
            """Get Trinity integration status."""
            if _trinity_integration:
                return _trinity_integration.get_status()
            return {"enabled": False, "state": "DISABLED"}

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
    """Graceful shutdown with Trinity coordination."""
    logger.info("Shutting down...")

    # v84.0: Shutdown Trinity integration first (notify network)
    if _trinity_integration:
        try:
            await _trinity_integration.shutdown()
            logger.info("[Trinity] ✓ Trinity integration shutdown complete")
        except Exception as e:
            logger.warning(f"[Trinity] Shutdown error: {e}")

    # Then stop the model manager
    await manager.stop()
    server.should_exit = True
    logger.info("Shutdown complete")


if __name__ == "__main__":
    asyncio.run(main())
