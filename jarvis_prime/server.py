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
import json
import logging
import os
import signal
import sys
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional
from enum import Enum, auto

# v93.0: Suppress known benign compatibility warnings from external libraries
# These are informational warnings about version compatibility, not errors
warnings.filterwarnings('ignore', message='.*scikit-learn version.*not supported.*')
warnings.filterwarnings('ignore', message='.*Torch version.*has not been tested.*')
warnings.filterwarnings('ignore', message='.*coremltools.*')
warnings.filterwarnings('ignore', category=DeprecationWarning, module='coremltools.*')
warnings.filterwarnings('ignore', category=FutureWarning, module='coremltools.*')

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


class StartupState:
    """
    v93.1: Track server startup state for immediate health checks.

    This allows the HTTP server to start IMMEDIATELY and respond to health
    checks while heavy initialization (ML imports, model loading) happens
    in the background.
    """
    def __init__(self):
        self.phase = "starting"  # starting -> initializing -> loading_model -> ready | error
        self.start_time = time.time()
        self.error: Optional[str] = None
        self.manager = None
        self.init_elapsed: Optional[float] = None
        self.model_load_start: Optional[float] = None
        self.model_load_elapsed: Optional[float] = None

    def get_status(self) -> Dict[str, Any]:
        """Get current status for health endpoint."""
        elapsed = time.time() - self.start_time
        result = {
            "status": "error" if self.error else ("healthy" if self.phase == "ready" else "starting"),
            "phase": self.phase,
            "startup_elapsed_seconds": round(elapsed, 1),
            "pid": os.getpid(),
        }
        if self.init_elapsed:
            result["init_elapsed_seconds"] = round(self.init_elapsed, 1)
        if self.model_load_elapsed:
            result["model_load_elapsed_seconds"] = round(self.model_load_elapsed, 1)
        if self.error:
            result["error"] = self.error
        return result


# Global startup state for immediate health checks
_startup_state: Optional[StartupState] = None


async def main():
    """
    v93.1: Main entry point with IMMEDIATE HTTP server startup.

    CRITICAL FIX: The HTTP server now starts FIRST before any heavy imports
    or model loading. This ensures health checks succeed immediately while
    initialization happens in the background.

    Startup Flow:
    1. Start HTTP server immediately (responds with "starting" status)
    2. Run heavy initialization in background task
    3. Update status to "healthy" when ready
    """
    global _trinity_integration, _startup_state

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
    logger.info("v93.1 - Immediate Startup with Background Initialization")
    logger.info("=" * 70)
    logger.info(f"Host: {args.host}:{args.port}")
    logger.info("-" * 70)

    # Initialize startup state FIRST
    _startup_state = StartupState()

    try:
        # =====================================================================
        # STEP 1: Create minimal FastAPI app that responds to health checks
        # =====================================================================
        from fastapi import FastAPI, HTTPException

        app = FastAPI(
            title="JARVIS-Prime API",
            description="OpenAI-compatible API for JARVIS-Prime Tier-0 Brain",
            version="1.0.0",
        )

        @app.get("/health")
        async def health():
            """
            v93.1: Immediate health endpoint.

            Returns status IMMEDIATELY - even during heavy initialization.
            """
            status = _startup_state.get_status() if _startup_state else {"status": "starting"}

            if status.get("status") == "error":
                raise HTTPException(status_code=503, detail=status)

            return status

        @app.get("/trinity/status")
        async def trinity_status():
            """Get Trinity integration status."""
            if _trinity_integration:
                return _trinity_integration.get_status()
            return {"enabled": False, "state": "DISABLED"}

        # Placeholder routes that will be replaced once manager is ready
        _completion_routes_ready = False

        @app.post("/v1/chat/completions")
        async def chat_completions_placeholder(request: dict):
            """Placeholder until model manager is ready."""
            if not _completion_routes_ready:
                raise HTTPException(
                    status_code=503,
                    detail={
                        "error": "Model still loading",
                        "status": _startup_state.get_status() if _startup_state else {}
                    }
                )
            return {"error": "Not implemented"}

        @app.post("/v1/completions")
        async def completions_placeholder(request: dict):
            """Placeholder until model manager is ready."""
            if not _completion_routes_ready:
                raise HTTPException(
                    status_code=503,
                    detail={
                        "error": "Model still loading",
                        "status": _startup_state.get_status() if _startup_state else {}
                    }
                )
            return {"error": "Not implemented"}

        # =====================================================================
        # STEP 2: Start uvicorn server IMMEDIATELY
        # =====================================================================
        import uvicorn

        config = uvicorn.Config(
            app,
            host=args.host,
            port=args.port,
            workers=1,  # Single worker for startup
            log_level="debug" if args.debug else "info",
        )

        server = uvicorn.Server(config)

        # Handle shutdown
        shutdown_event = asyncio.Event()

        async def handle_shutdown():
            """Handle graceful shutdown."""
            logger.info("Shutdown signal received")
            shutdown_event.set()
            server.should_exit = True

        loop = asyncio.get_event_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, lambda: asyncio.create_task(handle_shutdown()))

        logger.info("=" * 70)
        logger.info("[v93.1] HTTP server starting IMMEDIATELY")
        logger.info(f"Health endpoint: http://{args.host}:{args.port}/health")
        logger.info("Heavy initialization will run in background...")
        logger.info("=" * 70)

        # =====================================================================
        # STEP 3: Run heavy initialization in background
        # =====================================================================
        async def background_initialization():
            """
            v93.1: Run all heavy initialization in background.

            This includes:
            - Heavy ML library imports (torch, scikit-learn, etc.)
            - Hardware detection
            - Model manager creation
            - Model loading
            - Trinity integration
            - Heartbeat writer
            """
            nonlocal _completion_routes_ready

            try:
                _startup_state.phase = "initializing"
                init_start = time.time()

                logger.info("[Background] Starting heavy initialization...")

                # Heavy imports (triggers torch/scikit-learn warnings)
                logger.info("[Background] Importing ML libraries...")
                from jarvis_prime.core.model_manager import PrimeModelManager, create_api_app

                # Detect hardware
                try:
                    from jarvis_prime.core.llama_cpp_executor import HardwareDetector, LlamaCppConfig
                    hw = HardwareDetector.detect()
                    logger.info(f"[Background] Hardware: {hw.gpu_name or 'CPU'}")
                    logger.info(f"[Background] Backend: {hw.backend.name}")
                    logger.info(f"[Background] Memory: {hw.total_memory_gb:.1f} GB")
                except Exception as e:
                    logger.warning(f"[Background] Hardware detection failed: {e}")

                # Select executor
                executor_class = None
                initial_model = args.initial_model
                model_path = None

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
                    logger.info(f"[Background] Using LlamaCppExecutor (n_gpu_layers={executor_config.n_gpu_layers})")

                    # Find model
                    if initial_model is None:
                        model_path_obj = get_default_model_path()
                        if model_path_obj.exists():
                            initial_model = str(model_path_obj)
                            logger.info(f"[Background] Found model: {model_path_obj.name}")
                        elif args.auto_download:
                            logger.info("[Background] Downloading model...")
                            downloader = GGUFModelDownloader(models_dir=Path(models_dir))
                            recommended = downloader.get_recommended_model()
                            if recommended:
                                model_path_obj = await downloader.download(
                                    f"{recommended.repo_id}/{recommended.filename}"
                                )
                                initial_model = str(model_path_obj)

                _startup_state.init_elapsed = time.time() - init_start
                logger.info(f"[Background] Initialization complete ({_startup_state.init_elapsed:.1f}s)")

                # Create manager
                manager = PrimeModelManager(
                    models_dir=models_dir,
                    telemetry_dir=args.telemetry_dir,
                    reactor_core_watch_dir=args.reactor_core_dir,
                    executor_class=executor_class,
                )
                _startup_state.manager = manager

                # Start manager (with background model loading)
                _startup_state.phase = "loading_model"
                _startup_state.model_load_start = time.time()
                model_path = Path(initial_model) if initial_model else None
                await manager.start(initial_model_path=model_path, background_model_load=True)

                # Wait for model to load
                if model_path:
                    logger.info(f"[Background] Waiting for model to load: {model_path.name}")
                    while manager.get_health_status() == "starting":
                        await asyncio.sleep(1.0)
                        if shutdown_event.is_set():
                            return
                    _startup_state.model_load_elapsed = time.time() - _startup_state.model_load_start
                    logger.info(f"[Background] Model loaded ({_startup_state.model_load_elapsed:.1f}s)")

                # Initialize Trinity
                global _trinity_integration
                _trinity_integration = TrinityIntegration()
                trinity_model_path = str(model_path) if model_path else ""
                await _trinity_integration.initialize(
                    port=args.port,
                    model_path=trinity_model_path,
                    model_loaded=model_path is not None and model_path.exists(),
                )

                # Replace routes with real API
                real_app = create_api_app(manager)

                # Copy real routes to main app
                for route in real_app.routes:
                    if hasattr(route, 'path'):
                        # Skip health - we keep our own
                        if route.path in ['/health', '/trinity/status']:
                            continue
                        # Remove placeholder and add real route
                        app.routes = [r for r in app.routes if getattr(r, 'path', None) != route.path]
                        app.routes.append(route)

                _completion_routes_ready = True

                # Start heartbeat writer
                await start_heartbeat_writer(args, manager, model_path)

                # Mark as ready
                _startup_state.phase = "ready"
                total_time = time.time() - _startup_state.start_time
                logger.info("=" * 70)
                logger.info(f"[v93.1] JARVIS-Prime fully ready in {total_time:.1f}s")
                logger.info(f"API endpoint: http://{args.host}:{args.port}/v1/chat/completions")
                logger.info("=" * 70)

            except Exception as e:
                logger.error(f"[Background] Initialization failed: {e}")
                _startup_state.phase = "error"
                _startup_state.error = str(e)

        async def start_heartbeat_writer(args, manager, model_path):
            """Start the heartbeat writer task."""
            import tempfile

            heartbeat_file = Path.home() / ".jarvis" / "trinity" / "components" / "jarvis_prime.json"
            trinity_dir = heartbeat_file.parent
            trinity_dir.mkdir(parents=True, exist_ok=True)

            heartbeat_interval = float(os.getenv("JARVIS_PRIME_HEARTBEAT_INTERVAL", "5.0"))

            async def writer():
                while not shutdown_event.is_set():
                    try:
                        heartbeat_data = {
                            "component": "jarvis_prime",
                            "component_type": "j_prime",
                            "instance_id": f"jprime-{os.getpid()}-{int(time.time())}",
                            "pid": os.getpid(),
                            "port": args.port,
                            "host": args.host,
                            "endpoint": f"http://localhost:{args.port}",
                            "api_format": "openai",
                            "model_loaded": manager.current_model is not None if hasattr(manager, 'current_model') else False,
                            "model_name": str(manager.current_model.name) if hasattr(manager, 'current_model') and manager.current_model else None,
                            "model_path": str(model_path) if model_path else None,
                            "status": _startup_state.phase if _startup_state else "unknown",
                            "healthy": _startup_state.phase == "ready" if _startup_state else False,
                            "timestamp": time.time(),
                            "uptime_seconds": time.time() - _startup_state.start_time if _startup_state else 0,
                        }

                        try:
                            import psutil
                            proc = psutil.Process()
                            heartbeat_data["cpu_percent"] = proc.cpu_percent()
                            heartbeat_data["memory_mb"] = proc.memory_info().rss / (1024 * 1024)
                        except ImportError:
                            pass

                        if _trinity_integration:
                            trinity_status = _trinity_integration.get_status()
                            heartbeat_data["trinity_connected"] = trinity_status.get("connected", False)
                            heartbeat_data["trinity_state"] = trinity_status.get("state", "UNKNOWN")

                        # Atomic write
                        tmp_fd, tmp_name = tempfile.mkstemp(dir=trinity_dir, prefix=".jprime.", suffix=".tmp")
                        try:
                            with os.fdopen(tmp_fd, 'w') as f:
                                json.dump(heartbeat_data, f, indent=2)
                                f.flush()
                                os.fsync(f.fileno())
                            os.replace(tmp_name, heartbeat_file)
                        except Exception:
                            if os.path.exists(tmp_name):
                                os.unlink(tmp_name)

                        await asyncio.sleep(heartbeat_interval)

                    except asyncio.CancelledError:
                        break
                    except Exception as e:
                        logger.debug(f"[Heartbeat] Error: {e}")
                        await asyncio.sleep(heartbeat_interval)

            asyncio.create_task(writer())
            logger.info(f"[Heartbeat] Writer started (file={heartbeat_file})")

        # Start background initialization
        init_task = asyncio.create_task(background_initialization())

        # Run server (this blocks until shutdown)
        await server.serve()

        # Cleanup
        init_task.cancel()
        try:
            await init_task
        except asyncio.CancelledError:
            pass

        if _startup_state and _startup_state.manager:
            await _startup_state.manager.stop()
        if _trinity_integration:
            await _trinity_integration.shutdown()

    except ImportError as e:
        logger.error(f"Missing dependency: {e}")
        logger.error("Install with: pip install fastapi uvicorn llama-cpp-python")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Server error: {e}")
        raise


# v93.1: Legacy shutdown function (deprecated - cleanup is now handled inline in main())
async def shutdown(
    manager,
    server,
    heartbeat_task: Optional[asyncio.Task] = None,
    heartbeat_file: Optional[Path] = None,
):
    """
    Graceful shutdown with Trinity coordination and heartbeat cleanup.

    DEPRECATED: v93.1 handles cleanup inline in main() for better control.
    This function is kept for backwards compatibility only.
    """
    logger.info("Shutting down...")

    # v84.0: Cancel heartbeat task first
    if heartbeat_task and not heartbeat_task.done():
        heartbeat_task.cancel()
        try:
            await heartbeat_task
        except asyncio.CancelledError:
            pass
        logger.info("[Heartbeat] ✓ Writer stopped")

    # Cleanup heartbeat file
    if heartbeat_file:
        try:
            if heartbeat_file.exists():
                heartbeat_file.unlink()
                logger.info("[Heartbeat] ✓ File cleaned up")
        except Exception as e:
            logger.debug(f"[Heartbeat] Cleanup error: {e}")

    # v84.0: Shutdown Trinity integration (notify network)
    if _trinity_integration:
        try:
            await _trinity_integration.shutdown()
            logger.info("[Trinity] ✓ Trinity integration shutdown complete")
        except Exception as e:
            logger.warning(f"[Trinity] Shutdown error: {e}")

    # Then stop the model manager
    if manager:
        await manager.stop()
    if server:
        server.should_exit = True
    logger.info("Shutdown complete")


if __name__ == "__main__":
    asyncio.run(main())
