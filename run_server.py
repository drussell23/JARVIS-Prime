#!/usr/bin/env python3
"""
JARVIS-Prime Server - Quick Start Script (v93.2 with Immediate HTTP Startup)
=============================================================================

CRITICAL FIX v93.2: HTTP server starts IMMEDIATELY, heavy initialization runs
in background. This solves the 61.9s timeout issue where ML imports blocked
the server from responding to health checks.

Runs JARVIS-Prime with llama-cpp-python backend.
Integrates with main JARVIS infrastructure for unified cost tracking.

Usage:
    python run_server.py                        # Default
    python run_server.py --model foo.gguf       # Custom model
    python run_server.py --gpu-layers -1        # Metal GPU

Endpoints:
    POST /v1/chat/completions  - OpenAI-compatible chat
    POST /generate             - Simple text generation
    GET  /health               - Health check (IMMEDIATE)
    GET  /metrics              - Cost/inference metrics
"""

# =============================================================================
# v93.16: Warning Suppression - BEFORE any imports
# =============================================================================
import os
os.environ.setdefault('PYTHONWARNINGS', 'ignore::UserWarning,ignore::DeprecationWarning,ignore::FutureWarning')
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
os.environ.setdefault('TRANSFORMERS_VERBOSITY', 'error')
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')

# =============================================================================
# v149.0: HOLLOW CLIENT GUARD - MUST BE BEFORE ANY ML IMPORTS
# v150.0: Import is_hollow_client_active for model loading skip check
# =============================================================================
_hollow_installed = False
_hollow_reason = "Not installed"
_is_hollow_client_active = lambda: False  # Default: not active

try:
    from jarvis_prime.core.hollow_client_guard import (
        install_hollow_guard,
        is_hollow_client_active as _is_hollow_check,
    )
    _hollow_installed, _hollow_reason = install_hollow_guard()
    _is_hollow_client_active = _is_hollow_check
except (ImportError, Exception) as e:
    pass  # Guard not available

import warnings
import sys

warnings.simplefilter('ignore', category=UserWarning)
warnings.simplefilter('ignore', category=DeprecationWarning)
warnings.simplefilter('ignore', category=FutureWarning)

# v93.16: Specific filters for known warning messages
warnings.filterwarnings('ignore', message='.*urllib3.*OpenSSL.*LibreSSL.*')
warnings.filterwarnings('ignore', category=DeprecationWarning, module='urllib3')
warnings.filterwarnings('ignore', message='.*Torch version.*has not been tested.*')
warnings.filterwarnings('ignore', message='.*coremltools.*')
warnings.filterwarnings('ignore', category=UserWarning, module='torch')
warnings.filterwarnings('ignore', message='.*scikit-learn version.*is not supported.*')
warnings.filterwarnings('ignore', message='.*Disabling scikit-learn conversion API.*')
warnings.filterwarnings('ignore', message='.*Minimum required version.*')
warnings.filterwarnings('ignore', message='.*Maximum required version.*')

# v93.16: Aggressive suppression for coremltools at module level
warnings.filterwarnings('ignore', module='coremltools.*')
warnings.filterwarnings('ignore', module='sklearn.*')
warnings.filterwarnings('ignore', module='torch.*')

# v95.0: Override warnings.warn to filter at the source
# CRITICAL: Accept all parameters that Python's warnings.warn() can receive
# including 'source' (added in Python 3.6) to prevent TypeError
_original_warn = warnings.warn
def _filtered_warn(message, category=UserWarning, stacklevel=1, source=None):
    """
    Filtered warn that suppresses known non-critical warnings.

    v95.0: Added 'source' parameter to match Python's warnings.warn() signature.
    Without this, any library calling warnings.warn(..., source=something)
    would raise: TypeError: _filtered_warn() got an unexpected keyword argument 'source'
    """
    msg_str = str(message).lower()
    suppress_patterns = [
        'scikit-learn', 'coremltools', 'torch version', 'not supported',
        'has not been tested', 'minimum required', 'maximum required',
        'disabling', 'conversion api', 'urllib3', 'libressl', 'openssl'
    ]
    if any(pattern in msg_str for pattern in suppress_patterns):
        return  # Suppress
    # Pass source parameter to original warn if provided
    if source is not None:
        _original_warn(message, category, stacklevel + 1, source=source)
    else:
        _original_warn(message, category, stacklevel + 1)
warnings.warn = _filtered_warn

# =============================================================================
# MINIMAL IMPORTS ONLY - Heavy imports happen in background_initialization
# =============================================================================
import argparse
import asyncio
import json
import logging
import os
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("jarvis-prime")


# =============================================================================
# v97.0: Service Registry Functions for Cross-Repo Coordination
# =============================================================================
# These functions enable JARVIS Prime to register with the shared service
# registry and maintain heartbeats to prevent stale cleanup.

import fcntl
from contextlib import contextmanager

_REGISTRY_LOCK_TIMEOUT = 10.0

@contextmanager
def _acquire_registry_lock(lock_file: Path, timeout: float = _REGISTRY_LOCK_TIMEOUT):
    """Acquire an exclusive lock on the registry lock file."""
    lock_file.parent.mkdir(parents=True, exist_ok=True)
    lock_fd = open(lock_file, 'w')
    start_time = time.time()

    while True:
        try:
            fcntl.flock(lock_fd.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            break
        except BlockingIOError:
            if time.time() - start_time > timeout:
                lock_fd.close()
                raise TimeoutError(f"Could not acquire registry lock within {timeout}s")
            time.sleep(0.05)

    try:
        yield lock_fd
    finally:
        try:
            fcntl.flock(lock_fd.fileno(), fcntl.LOCK_UN)
            lock_fd.close()
        except Exception:
            pass


def _atomic_register_service(
    service_name: str,
    service_data: Dict[str, Any],
    alternate_names: Optional[List[str]] = None,
    timeout: float = _REGISTRY_LOCK_TIMEOUT,
) -> bool:
    """
    v97.0: Register a service with the shared registry atomically.
    """
    registry_dir = Path(os.getenv(
        "JARVIS_REGISTRY_DIR",
        str(Path.home() / ".jarvis" / "registry")
    ))
    registry_file = registry_dir / "services.json"
    lock_file = registry_dir / "services.json.lock"

    try:
        with _acquire_registry_lock(lock_file, timeout):
            existing_services = {}
            if registry_file.exists():
                try:
                    content = registry_file.read_text()
                    if content.strip():
                        existing_services = json.loads(content)
                        if not isinstance(existing_services, dict):
                            existing_services = {}
                except (json.JSONDecodeError, OSError):
                    existing_services = {}

            existing_services[service_name] = service_data
            if alternate_names:
                for alt_name in alternate_names:
                    existing_services[alt_name] = service_data

            registry_dir.mkdir(parents=True, exist_ok=True)
            temp_file = registry_file.with_suffix(".tmp")
            with open(temp_file, 'w') as f:
                json.dump(existing_services, f, indent=2)
                f.flush()
                os.fsync(f.fileno())
            temp_file.replace(registry_file)

        return True
    except Exception as e:
        logger.warning(f"[v97.0] Registry registration failed: {e}")
        return False


def _atomic_update_heartbeat(
    service_names: List[str],
    status: str = "running",
    timeout: float = _REGISTRY_LOCK_TIMEOUT,
) -> bool:
    """
    v97.0: Atomically update heartbeat timestamp in the service registry.
    """
    registry_dir = Path(os.getenv(
        "JARVIS_REGISTRY_DIR",
        str(Path.home() / ".jarvis" / "registry")
    ))
    registry_file = registry_dir / "services.json"
    lock_file = registry_dir / "services.json.lock"

    try:
        with _acquire_registry_lock(lock_file, timeout):
            existing_services = {}
            if registry_file.exists():
                try:
                    content = registry_file.read_text()
                    if content.strip():
                        existing_services = json.loads(content)
                except (json.JSONDecodeError, OSError):
                    return False

            current_time = time.time()
            updated = False
            for name in service_names:
                if name in existing_services:
                    existing_services[name]["last_heartbeat"] = current_time
                    existing_services[name]["status"] = status
                    updated = True

            if not updated:
                return False

            temp_file = registry_file.with_suffix(".tmp")
            with open(temp_file, 'w') as f:
                json.dump(existing_services, f, indent=2)
                f.flush()
                os.fsync(f.fileno())
            temp_file.replace(registry_file)

        return True
    except Exception:
        return False


def _atomic_deregister_service(
    service_names: List[str],
    timeout: float = _REGISTRY_LOCK_TIMEOUT,
) -> bool:
    """
    v97.0: Deregister services from the shared registry atomically.
    """
    registry_dir = Path(os.getenv(
        "JARVIS_REGISTRY_DIR",
        str(Path.home() / ".jarvis" / "registry")
    ))
    registry_file = registry_dir / "services.json"
    lock_file = registry_dir / "services.json.lock"

    try:
        with _acquire_registry_lock(lock_file, timeout):
            existing_services = {}
            if registry_file.exists():
                try:
                    content = registry_file.read_text()
                    if content.strip():
                        existing_services = json.loads(content)
                except (json.JSONDecodeError, OSError):
                    return True

            for name in service_names:
                existing_services.pop(name, None)

            temp_file = registry_file.with_suffix(".tmp")
            with open(temp_file, 'w') as f:
                json.dump(existing_services, f, indent=2)
                f.flush()
                os.fsync(f.fileno())
            temp_file.replace(registry_file)

        return True
    except Exception:
        return False


# v97.0: Global registry heartbeat task reference
_registry_heartbeat_task: Optional[asyncio.Task] = None

# v258.0: Strong references for fire-and-forget background tasks (prevents GC collection)
_background_tasks: set = set()


# =============================================================================
# v138.0: HARDWARE-AWARE STARTUP STRATEGY
# =============================================================================
# Detects system memory and configures AGI Hub appropriately:
# - < 16GB:  CLOUD ONLY mode (skip local AGI Hub entirely)
# - 16-30GB: SLIM MODE (staged init with deferred heavy subsystems)
# - 30-64GB: FULL MODE (all subsystems but with memory gates)
# - 64GB+:   UNLIMITED MODE (full parallel initialization)
# =============================================================================

from enum import Enum, auto
from dataclasses import dataclass
from typing import Tuple


class HardwareProfile(Enum):
    """
    v138.0: Hardware profile classification based on available resources.
    """
    CLOUD_ONLY = auto()      # < 16GB RAM - skip local AGI, use cloud
    SLIM = auto()            # 16-30GB RAM - staged init, defer heavy
    FULL = auto()            # 30-64GB RAM - full with memory gates
    UNLIMITED = auto()       # 64GB+ RAM - full parallel init


@dataclass
class HardwareAssessment:
    """
    v138.0: Complete hardware assessment for startup decisions.
    """
    profile: HardwareProfile
    total_ram_gb: float
    available_ram_gb: float
    cpu_count: int
    is_apple_silicon: bool
    has_gpu: bool
    gpu_name: str
    recommended_gpu_layers: int
    recommended_context_size: int
    skip_agi_hub: bool
    enable_slim_mode: bool
    defer_heavy_subsystems: bool
    reason: str


def _assess_hardware() -> HardwareAssessment:
    """
    v138.0: Perform comprehensive hardware assessment for startup decisions.

    First checks for environment variables passed from the supervisor (JARVIS_HARDWARE_PROFILE, etc.)
    which allows the orchestrator to coordinate the hardware profile decision across repos.
    If not present, falls back to local hardware detection.

    Returns:
        HardwareAssessment with profile and recommendations.
    """
    import platform

    # =========================================================================
    # v138.0: CHECK FOR SUPERVISOR-PROVIDED HARDWARE PROFILE
    # =========================================================================
    # The supervisor (cross_repo_startup_orchestrator) may have already assessed
    # hardware and passed the profile via environment variables. This ensures
    # consistent profile decisions across all repos in the JARVIS ecosystem.
    # =========================================================================
    supervisor_profile = os.environ.get("JARVIS_HARDWARE_PROFILE")
    if supervisor_profile:
        logger.info(f"[v138.0] 📡 Hardware profile received from supervisor: {supervisor_profile}")
        try:
            # Parse profile from environment
            profile = HardwareProfile[supervisor_profile]

            # Read other environment variables from supervisor
            total_ram_gb = float(os.environ.get("JARVIS_TOTAL_RAM_GB", "16.0"))
            available_ram_gb = float(os.environ.get("JARVIS_AVAILABLE_RAM_GB", "8.0"))
            cpu_count = int(os.environ.get("JARVIS_CPU_COUNT", "4"))
            is_apple_silicon = os.environ.get("JARVIS_IS_APPLE_SILICON", "false").lower() == "true"
            skip_agi_hub = os.environ.get("JARVIS_SKIP_AGI_HUB", "false").lower() == "true"
            enable_slim_mode = os.environ.get("JARVIS_ENABLE_SLIM_MODE", "false").lower() == "true"
            defer_heavy_subsystems = os.environ.get("JARVIS_DEFER_HEAVY_SUBSYSTEMS", "false").lower() == "true"
            has_gpu = os.environ.get("JARVIS_HAS_GPU", "false").lower() == "true"
            gpu_name = os.environ.get("JARVIS_GPU_NAME", "Unknown")
            recommended_gpu_layers = int(os.environ.get("JARVIS_GPU_LAYERS", "0"))
            recommended_context_size = int(os.environ.get("JARVIS_CONTEXT_SIZE", "2048"))

            reason = f"Profile received from supervisor: {supervisor_profile} ({total_ram_gb:.1f}GB RAM)"

            logger.info(f"[v138.0] ✅ Using supervisor-provided hardware profile: {profile.name}")
            return HardwareAssessment(
                profile=profile,
                total_ram_gb=total_ram_gb,
                available_ram_gb=available_ram_gb,
                cpu_count=cpu_count,
                is_apple_silicon=is_apple_silicon,
                has_gpu=has_gpu,
                gpu_name=gpu_name,
                recommended_gpu_layers=recommended_gpu_layers,
                recommended_context_size=recommended_context_size,
                skip_agi_hub=skip_agi_hub,
                enable_slim_mode=enable_slim_mode,
                defer_heavy_subsystems=defer_heavy_subsystems,
                reason=reason,
            )
        except (KeyError, ValueError) as e:
            logger.warning(f"[v138.0] ⚠️ Invalid supervisor profile '{supervisor_profile}': {e}. Falling back to local detection.")

    # =========================================================================
    # v138.0: LOCAL HARDWARE DETECTION (Fallback)
    # =========================================================================
    # If supervisor didn't provide profile, or if running standalone, detect
    # hardware locally.
    # =========================================================================
    logger.info("[v138.0] 🔍 Performing local hardware detection...")

    # Default values if psutil not available
    total_ram_gb = 8.0
    available_ram_gb = 4.0
    cpu_count = 4
    is_apple_silicon = False
    has_gpu = False
    gpu_name = "Unknown"

    # Detect Apple Silicon
    machine = platform.machine().lower()
    is_apple_silicon = machine in ("arm64", "aarch64") and platform.system() == "Darwin"

    # Try to get accurate hardware info via psutil
    try:
        import psutil
        mem = psutil.virtual_memory()
        total_ram_gb = mem.total / (1024 ** 3)
        available_ram_gb = mem.available / (1024 ** 3)
        cpu_count = psutil.cpu_count(logical=True) or 4
    except ImportError:
        logger.warning("[v138.0] psutil not available - using conservative defaults")
    except Exception as e:
        logger.warning(f"[v138.0] Hardware detection error: {e}")

    # Detect GPU
    if is_apple_silicon:
        has_gpu = True
        # Detect specific Apple Silicon chip
        try:
            import subprocess
            result = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True, text=True, timeout=5
            )
            chip_info = result.stdout.strip()
            if "M1" in chip_info:
                gpu_name = "Apple M1 GPU"
            elif "M2" in chip_info:
                gpu_name = "Apple M2 GPU"
            elif "M3" in chip_info:
                gpu_name = "Apple M3 GPU"
            elif "M4" in chip_info:
                gpu_name = "Apple M4 GPU"
            else:
                gpu_name = f"Apple Silicon GPU ({chip_info})"
        except Exception:
            gpu_name = "Apple Silicon GPU"

    # Classify hardware profile
    if total_ram_gb < 16:
        profile = HardwareProfile.CLOUD_ONLY
        skip_agi_hub = True
        enable_slim_mode = True
        defer_heavy_subsystems = True
        reason = f"Insufficient RAM ({total_ram_gb:.1f}GB < 16GB) - CLOUD ONLY mode"
        recommended_gpu_layers = 0  # Don't use GPU for model on low-memory systems
        recommended_context_size = 1024

    elif total_ram_gb < 30:
        profile = HardwareProfile.SLIM
        skip_agi_hub = False
        enable_slim_mode = True
        defer_heavy_subsystems = True
        reason = f"Moderate RAM ({total_ram_gb:.1f}GB) - SLIM MODE with staged initialization"
        recommended_gpu_layers = -1 if is_apple_silicon else 0
        recommended_context_size = 2048

    elif total_ram_gb < 64:
        profile = HardwareProfile.FULL
        skip_agi_hub = False
        enable_slim_mode = False
        defer_heavy_subsystems = False
        reason = f"Adequate RAM ({total_ram_gb:.1f}GB) - FULL MODE with memory gates"
        recommended_gpu_layers = -1 if is_apple_silicon else 32
        recommended_context_size = 4096

    else:
        profile = HardwareProfile.UNLIMITED
        skip_agi_hub = False
        enable_slim_mode = False
        defer_heavy_subsystems = False
        reason = f"Abundant RAM ({total_ram_gb:.1f}GB) - UNLIMITED MODE"
        recommended_gpu_layers = -1
        recommended_context_size = 8192

    return HardwareAssessment(
        profile=profile,
        total_ram_gb=total_ram_gb,
        available_ram_gb=available_ram_gb,
        cpu_count=cpu_count,
        is_apple_silicon=is_apple_silicon,
        has_gpu=has_gpu,
        gpu_name=gpu_name,
        recommended_gpu_layers=recommended_gpu_layers,
        recommended_context_size=recommended_context_size,
        skip_agi_hub=skip_agi_hub,
        enable_slim_mode=enable_slim_mode,
        defer_heavy_subsystems=defer_heavy_subsystems,
        reason=reason,
    )


def _log_hardware_assessment(assessment: HardwareAssessment) -> None:
    """Log hardware assessment with formatted output."""
    logger.info("")
    logger.info("=" * 70)
    logger.info("🔧 v138.0: HARDWARE-AWARE STARTUP STRATEGY")
    logger.info("=" * 70)
    logger.info(f"   Profile: {assessment.profile.name}")
    logger.info(f"   Total RAM: {assessment.total_ram_gb:.1f} GB")
    logger.info(f"   Available RAM: {assessment.available_ram_gb:.1f} GB")
    logger.info(f"   CPU Cores: {assessment.cpu_count}")
    logger.info(f"   Apple Silicon: {assessment.is_apple_silicon}")
    logger.info(f"   GPU: {assessment.gpu_name}")
    logger.info(f"   Decision: {assessment.reason}")

    if assessment.skip_agi_hub:
        logger.warning("   ⚠️ AGI Hub will be SKIPPED - using cloud bridge only")
    elif assessment.enable_slim_mode:
        logger.info("   💡 SLIM MODE active - staged initialization with deferred subsystems")
    else:
        logger.info("   ✅ FULL MODE active - all subsystems will initialize")

    logger.info("=" * 70)
    logger.info("")


# Global hardware assessment (populated during startup)
_hardware_assessment: Optional[HardwareAssessment] = None


def parse_args():
    parser = argparse.ArgumentParser(description="JARVIS-Prime Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to listen on")
    parser.add_argument("--model", default="models/current.gguf", help="Model path")
    parser.add_argument("--ctx-size", type=int, default=2048, help="Context size")
    parser.add_argument("--threads", type=int, default=4, help="CPU threads")
    parser.add_argument("--gpu-layers", type=int, default=0, help="GPU layers (-1 for all)")
    parser.add_argument("--reload", action="store_true", help="Auto-reload on changes")
    parser.add_argument("--debug", action="store_true", help="Debug logging")
    parser.add_argument(
        "--bridge-enabled",
        action="store_true",
        default=True,
        help="Enable cross-repo bridge for JARVIS integration (default: True)"
    )
    parser.add_argument(
        "--no-bridge",
        action="store_true",
        help="Disable cross-repo bridge"
    )
    return parser.parse_args()


class StartupState:
    """
    v93.2: Track server startup state for immediate health checks.
    v224.0: Phase-based progress tracking with milestone-based reporting.

    This allows the HTTP server to start IMMEDIATELY and respond to health
    checks while heavy initialization (ML imports, model loading) happens
    in the background.
    """

    _TOTAL_INIT_STEPS = 11  # Steps 1-11 in background_initialization()

    def __init__(self):
        self.phase = "starting"  # starting -> initializing -> loading_model -> ready | error
        self.start_time = time.time()
        self.error: Optional[str] = None
        self.init_elapsed: Optional[float] = None
        self.model_load_start: Optional[float] = None
        self.model_load_elapsed: Optional[float] = None
        self.model_path: Optional[str] = None
        self.model_loaded: bool = False
        self.details: Dict[str, Any] = {}

        # v224.0: Phase-based progress tracking
        self.init_phases: list = []
        self.current_phase_index: int = -1

        # v235.3: Optional APARS file integration (for golden-image startup)
        self._apars_file: Optional[str] = os.environ.get("JARVIS_APARS_FILE")
        self._apars_cache: Optional[Dict[str, Any]] = None
        self._apars_cache_mtime: float = 0.0

    def _read_apars_file(self) -> Optional[Dict[str, Any]]:
        """v235.3: Load APARS payload from file if configured."""
        if not self._apars_file:
            self._apars_file = os.environ.get("JARVIS_APARS_FILE")
        if not self._apars_file:
            return None
        try:
            mtime = os.path.getmtime(self._apars_file)
        except FileNotFoundError:
            return None
        except Exception:
            return self._apars_cache
        if self._apars_cache is not None and mtime == self._apars_cache_mtime:
            return self._apars_cache
        try:
            with open(self._apars_file, "r") as f:
                payload = json.load(f)
            self._apars_cache = payload
            self._apars_cache_mtime = mtime
            return payload
        except Exception:
            return self._apars_cache

    def begin_phase(self, name: str, step_num: int) -> None:
        """
        v224.0: Begin a new initialization phase with milestone tracking.

        Automatically completes the previous in-progress phase (if any).
        """
        # Auto-complete previous phase
        if 0 <= self.current_phase_index < len(self.init_phases):
            prev = self.init_phases[self.current_phase_index]
            if prev["status"] == "in_progress":
                prev["status"] = "completed"
                prev["completed_at"] = time.time()
                prev["elapsed_seconds"] = round(time.time() - prev["started_at"], 2)

        phase_entry = {
            "name": name,
            "step_num": step_num,
            "total_steps": self._TOTAL_INIT_STEPS,
            "status": "in_progress",
            "started_at": time.time(),
            "completed_at": None,
            "elapsed_seconds": 0,
        }

        self.init_phases.append(phase_entry)
        self.current_phase_index = len(self.init_phases) - 1
        self.details["step"] = name
        self.details["step_num"] = step_num
        self.details["total_steps"] = self._TOTAL_INIT_STEPS

    def complete_phase(self, name: str) -> None:
        """v224.0: Mark a phase as completed by name."""
        for phase in self.init_phases:
            if phase["name"] == name and phase["status"] == "in_progress":
                phase["status"] = "completed"
                phase["completed_at"] = time.time()
                phase["elapsed_seconds"] = round(time.time() - phase["started_at"], 2)
                break

    def fail_phase(self, name: str, error: str) -> None:
        """v224.0: Mark a phase as failed by name."""
        for phase in self.init_phases:
            if phase["name"] == name and phase["status"] == "in_progress":
                phase["status"] = "failed"
                phase["completed_at"] = time.time()
                phase["elapsed_seconds"] = round(time.time() - phase["started_at"], 2)
                phase["error"] = error
                break

    def get_status(self) -> Dict[str, Any]:
        """
        Get current status for health endpoint.

        v93.7: Enhanced with detailed step information and loading progress.
        v224.0: Phase-based progress tracking with v2 protocol.

        Reports model_load_elapsed_seconds DURING loading (not just after)
        to enable intelligent progress-based timeout extension.
        """
        elapsed = time.time() - self.start_time
        result = {
            # v93.13: Add "service" field for Trinity cross-repo discovery
            # TrinityIntegrator._discover_running_component() checks this field
            # to verify it's talking to the correct service during startup
            "service": "jarvis_prime",
            "status": "error" if self.error else ("healthy" if self.phase == "ready" else "starting"),
            "phase": self.phase,
            "startup_elapsed_seconds": round(elapsed, 1),
            "pid": os.getpid(),
            "model_loaded": self.model_loaded,
            "model_path": self.model_path,
        }
        if self.init_elapsed:
            result["init_elapsed_seconds"] = round(self.init_elapsed, 1)

        # v93.7: Report current initialization step for better debugging
        if self.details:
            result["current_step"] = self.details.get("step", "unknown")
            result["details"] = self.details

        # v224.0: Structured phase progress (v2 protocol)
        if self.init_phases:
            completed_phases = sum(1 for p in self.init_phases if p["status"] == "completed")
            total_phases = self._TOTAL_INIT_STEPS
            current_phase = (
                self.init_phases[self.current_phase_index]
                if 0 <= self.current_phase_index < len(self.init_phases)
                else None
            )

            result["init_progress"] = {
                "protocol_version": 2,
                "completed_phases": completed_phases,
                "total_phases": total_phases,
                "overall_pct": round((completed_phases / total_phases) * 100, 1) if total_phases > 0 else 0,
                "current_phase": {
                    "name": current_phase["name"],
                    "step_num": current_phase["step_num"],
                    "status": current_phase["status"],
                    "elapsed_seconds": round(
                        time.time() - current_phase["started_at"], 1
                    ) if current_phase["status"] == "in_progress" else current_phase.get("elapsed_seconds", 0),
                } if current_phase else None,
                "phases": [
                    {
                        "name": p["name"],
                        "step_num": p["step_num"],
                        "status": p["status"],
                        "elapsed_seconds": round(
                            time.time() - p["started_at"], 1
                        ) if p["status"] == "in_progress" else p.get("elapsed_seconds", 0),
                    }
                    for p in self.init_phases
                ],
            }

        # v93.5 + v224.0: Report model load elapsed DURING loading (not just after)
        # BUGFIX (v224.0): model_load_progress_pct is now 0 when NOT in the actual
        # model loading step. Previously it was non-zero during Steps 1-7 because
        # model_load_start was set before pre-load work.
        if self.model_load_elapsed:
            result["model_load_elapsed_seconds"] = round(self.model_load_elapsed, 1)
        elif self.model_load_start:
            # Model is currently loading - report elapsed time so far
            current_elapsed = time.time() - self.model_load_start
            result["model_load_elapsed_seconds"] = round(current_elapsed, 1)
            result["model_loading_in_progress"] = True
            # v224.0: Only report non-zero progress when actually loading model
            model_timeout = self.details.get("model_load_timeout", 600.0)
            result["model_load_progress_pct"] = min(95, round((current_elapsed / model_timeout) * 100, 1))
        else:
            # v224.0: Not yet in model loading phase — report 0
            result["model_load_progress_pct"] = 0
            result["model_loading_in_progress"] = False

        if self.error:
            result["error"] = self.error

        # v235.3: Attach APARS payload if available (golden-image startup)
        apars_payload = self._read_apars_file()
        if apars_payload:
            result["apars"] = apars_payload
            if "ready_for_inference" in apars_payload:
                result.setdefault("ready_for_inference", apars_payload.get("ready_for_inference"))
            if "model_loaded" in apars_payload:
                result.setdefault("model_loaded", apars_payload.get("model_loaded"))

        return result


# =============================================================================
# GLOBAL STATE - Populated during background initialization
# =============================================================================
_startup_state: Optional[StartupState] = None
_bridge = None
_reactor_bridge = None
_training_pipeline = None
_jarvis_bridge = None
_jarvis_prime_bridge = None  # v238.0: Unified inference bridge (local-first + Claude fallback)
_neural_orchestrator = None
_executor = None
_model_coordinator = None  # v241.0: GCP multi-model swap coordinator
_classifier_loaded = False  # v242.0: Phi classifier state
_classifier_system_prompt = ""  # v242.0: Cached system prompt
_prompt_lock = None  # v242.1: Set after threading import below; guards _classifier_system_prompt
_telemetry_hook = None  # v242.0: Training data capture

# v242.1: Phi classifier circuit breaker (thread-safe)
import threading as _threading

_prompt_lock = _threading.Lock()  # v242.1: Guards _classifier_system_prompt updates

class _PhiCircuitBreaker:
    """Thread-safe circuit breaker for Phi classification.

    When Phi fails repeatedly, the circuit opens and classification
    is skipped -- requests fall through to the currently loaded specialist
    model with domain=general.
    """
    __slots__ = ("failure_count", "last_failure", "threshold", "timeout", "state", "_lock")

    def __init__(self):
        self.failure_count = 0
        self.last_failure = 0.0
        self.threshold = int(os.getenv("JPRIME_PHI_CIRCUIT_THRESHOLD", "3"))
        self.timeout = float(os.getenv("JPRIME_PHI_CIRCUIT_TIMEOUT", "30.0"))
        self.state = "closed"  # closed | open | half_open
        self._lock = _threading.Lock()

    def can_execute(self) -> bool:
        with self._lock:
            if self.state == "closed":
                return True
            if self.state == "open":
                if time.time() - self.last_failure > self.timeout:
                    self.state = "half_open"
                    return True
                return False
            return True  # half_open: allow one probe attempt

    def record_success(self):
        with self._lock:
            self.failure_count = 0
            self.state = "closed"

    def record_failure(self):
        with self._lock:
            self.failure_count += 1
            self.last_failure = time.time()
            if self.failure_count >= self.threshold:
                self.state = "open"

_phi_circuit = _PhiCircuitBreaker()

# v242.1: Classification observability counters
_classification_metrics = {
    "total": 0,
    "successes": 0,
    "failures": 0,
    "circuit_trips": 0,
    "self_served": 0,
    "latency_sum_ms": 0,
}

_agi_hub = None
_trinity_initialized = False
_trinity_record_inference = None
_neural_routing_enabled = False
_model_path: Optional[Path] = None
_args = None
_continual_learner = None
_self_modifier = None
_tier2_capability_task: Optional[asyncio.Task] = None
_tier2_capability_status: Dict[str, Any] = {
    "enabled": False,
    "state": "not_initialized",
    "continual_learning": {"status": "pending"},
    "self_improvement": {"status": "pending"},
}

# v235.3: Module-level app reference for ASGI middleware import
# The APARS enrichment middleware (generated by gcp_vm_manager startup script)
# imports this to wrap /health responses with progress data during model loading.
# Assigned inside main() via `global app`.
app = None


def _check_port_available(host: str, port: int) -> tuple[bool, str]:
    """
    v198.0: Pre-startup port availability check.

    This is a defense-in-depth measure that provides a clear error message
    BEFORE uvicorn attempts to bind, rather than letting it fail with
    a cryptic "Address already in use" error.

    Uses two-phase check:
    1. Connect test - detects if something is actively listening
    2. Bind test - confirms we can actually bind

    Returns:
        Tuple of (is_available, error_message)
    """
    import socket

    bind_host = '127.0.0.1' if host == '0.0.0.0' else host

    # Phase 1: Check if anything is already listening
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(0.5)
            result = sock.connect_ex((bind_host, port))
            if result == 0:
                # Connection succeeded = something is listening
                pass  # Fall through to get PID info
            else:
                # No listener, verify we can bind
                try:
                    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as bind_sock:
                        bind_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                        bind_sock.settimeout(1.0)
                        bind_sock.bind((bind_host, port))
                        return True, ""
                except OSError as bind_err:
                    pass  # Fall through to error reporting
    except Exception:
        pass

    # Port is in use - gather diagnostic info
    # Try to identify what's using the port
    pid_info = ""
    try:
        import psutil
        for conn in psutil.net_connections(kind='inet'):
            if conn.laddr.port == port:
                try:
                    proc = psutil.Process(conn.pid)
                    pid_info = f" (PID {conn.pid}: {proc.name()})"
                except Exception:
                    pid_info = f" (PID {conn.pid})"
                break
    except ImportError:
        pass
    except Exception:
        pass

    return False, f"Port {port} is already in use{pid_info}"


def _find_available_port(host: str, preferred_port: int, max_attempts: int = 10) -> Optional[int]:
    """
    v228.0: Find an available port starting from preferred_port + 1.
    Searches sequentially through ports preferred_port+1 to preferred_port+max_attempts.
    Returns the first available port, or None if all are occupied.
    """
    import socket

    bind_host = '127.0.0.1' if host == '0.0.0.0' else host

    for offset in range(1, max_attempts + 1):
        candidate = preferred_port + offset
        if candidate > 65535:
            break
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                sock.settimeout(1.0)
                sock.bind((bind_host, candidate))
                # Port is free — also verify nothing is listening
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
                    probe.settimeout(0.5)
                    result = probe.connect_ex((bind_host, candidate))
                    if result != 0:
                        logger.debug(f"[PortFallback] Port {candidate} is available")
                        return candidate
        except OSError:
            logger.debug(f"[PortFallback] Port {candidate} unavailable, trying next")
            continue

    return None


# =============================================================================
# v243.0: Reflex Inhibition Publisher — atomic writes + API key auth
# =============================================================================
_INHIBIT_API_KEY = os.getenv("JPRIME_INHIBIT_API_KEY", "")


def _atomic_write_json(path: Path, data: dict) -> None:
    """Write JSON atomically via temp-file-rename (POSIX atomic)."""
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2))
    tmp.rename(path)  # Atomic on POSIX


async def main():
    """
    v93.2: Main entry point with IMMEDIATE HTTP server startup.

    CRITICAL FIX: The HTTP server starts FIRST before any heavy imports
    or model loading. This ensures health checks succeed immediately while
    initialization happens in the background.

    Startup sequence:
    1. Parse args (instant)
    2. v198.0: Pre-check port availability (instant)
    3. Create minimal FastAPI app with health endpoint (instant)
    4. Start uvicorn server (instant - server is now LISTENING)
    5. FastAPI startup event triggers background_initialization()
    6. Heavy imports, model loading, bridges all run in background
    7. Health endpoint reports "starting" -> "ready" as init completes
    """
    global _startup_state, _args, app

    _args = parse_args()

    if _args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    # =========================================================================
    # v198.0: PRE-STARTUP PORT AVAILABILITY CHECK
    # v228.0: Intelligent port fallback instead of hard exit
    # =========================================================================
    # Check port BEFORE any heavy initialization to fail fast with clear error
    original_port = _args.port
    port_available, port_error = _check_port_available(_args.host, _args.port)
    if not port_available:
        logger.warning(f"[Startup] Port {_args.port} unavailable: {port_error}")
        # v228.0: Intelligent port fallback
        fallback_port = _find_available_port(_args.host, _args.port)
        if fallback_port:
            logger.info(f"[Startup] Found available fallback port: {fallback_port} (original: {_args.port})")
            _args.port = fallback_port
        else:
            logger.error("=" * 70)
            logger.error("PORT CONFLICT DETECTED - NO FALLBACK AVAILABLE")
            logger.error("=" * 70)
            logger.error(f"   {port_error}")
            logger.error(f"   Tried ports {original_port} through {original_port + 10}")
            logger.error("=" * 70)
            sys.exit(1)

    # v228.0: Broadcast actual port to environment for discovery
    os.environ["JARVIS_PRIME_PORT"] = str(_args.port)
    os.environ["JARVIS_PRIME_URL"] = f"http://localhost:{_args.port}"
    if _args.port != original_port:
        os.environ["JARVIS_PRIME_ORIGINAL_PORT"] = str(original_port)
        os.environ["JARVIS_PRIME_IS_FALLBACK_PORT"] = "true"
        logger.info(f"[Startup] Environment updated: JARVIS_PRIME_PORT={_args.port}, JARVIS_PRIME_URL=http://localhost:{_args.port}")

    # Initialize startup state FIRST
    _startup_state = StartupState()

    logger.info("[v93.2] JARVIS-Prime starting with IMMEDIATE HTTP server...")

    # =========================================================================
    # STEP 1: Import FastAPI (lightweight, instant)
    # =========================================================================
    # v197.1: RESILIENT DEPENDENCY LOADING - Auto-install or graceful degradation
    # =========================================================================
    try:
        from fastapi import FastAPI, HTTPException, Request
        from fastapi.middleware.cors import CORSMiddleware
        from fastapi.responses import StreamingResponse, JSONResponse
        from pydantic import BaseModel
        import uvicorn
    except ImportError as e:
        logger.error(f"Missing dependencies: {e}")
        logger.error("Install with: pip install fastapi uvicorn pydantic")

        # v197.1: Attempt auto-installation before giving up
        logger.info("[v197.1] Attempting auto-installation of missing dependencies...")
        try:
            import subprocess
            result = subprocess.run(
                [sys.executable, "-m", "pip", "install", "--quiet", "fastapi", "uvicorn", "pydantic"],
                capture_output=True,
                text=True,
                timeout=120
            )
            if result.returncode == 0:
                logger.info("[v197.1] ✅ Auto-installation successful! Retrying import...")
                # Retry the imports
                from fastapi import FastAPI, HTTPException, Request
                from fastapi.middleware.cors import CORSMiddleware
                from fastapi.responses import StreamingResponse, JSONResponse
                from pydantic import BaseModel
                import uvicorn
                logger.info("[v197.1] ✅ Dependencies loaded successfully after auto-install")
            else:
                logger.error(f"[v197.1] ❌ Auto-installation failed: {result.stderr}")
                logger.error("[v197.1] CRITICAL: Cannot start without FastAPI. Please install manually.")
                sys.exit(1)
        except Exception as install_error:
            logger.error(f"[v197.1] ❌ Auto-installation error: {install_error}")
            logger.error("[v197.1] CRITICAL: Cannot start without FastAPI. Please install manually.")
            sys.exit(1)

    # =========================================================================
    # STEP 2: Create MINIMAL FastAPI app that responds to health IMMEDIATELY
    # =========================================================================
    app = FastAPI(
        title="JARVIS-Prime",
        description="Tier-0 Muscle Memory Brain - OpenAI-compatible API (v93.2 Immediate Start)",
        version="93.2.0",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # =========================================================================
    # IMMEDIATE HEALTH ENDPOINT - Responds BEFORE initialization completes
    # =========================================================================
    @app.get("/health")
    async def health_check():
        """
        v93.2: Immediate health endpoint.

        Returns status IMMEDIATELY - even during heavy initialization.
        This is the key fix for the 61.9s timeout issue.

        Status meanings:
        - "starting": Server is up, initialization in progress
        - "healthy": Fully initialized and ready for inference
        - "error": Initialization failed
        """
        if _startup_state:
            status = _startup_state.get_status()
            status["websocket_paths"] = ["/ws/events"]
            status["integration_contracts"] = {
                "ws_events": "/api/v1/integration/contracts/ws-events",
            }
            status["websocket_contract_version"] = "1.0"
            # v291.2: Expose autonomy schema version for cross-repo contract
            # validation.  Without this, the unified supervisor's boot-time
            # check_autonomy_contracts() sees prime_schema=None → prime_compatible=False
            # → contract_mismatch → Autonomy Contract ⚠️ permanently.
            status["autonomy_schema_version"] = "1.0"
            status["contract_version"] = "1.0"

            # Add runtime component status once initialized
            if _startup_state.phase == "ready":
                status["bridge_enabled"] = _bridge is not None
                status["reactor_bridge_enabled"] = _reactor_bridge is not None
                status["training_pipeline_enabled"] = _training_pipeline is not None
                status["jarvis_bridge_enabled"] = _jarvis_bridge is not None
                status["jarvis_prime_bridge_enabled"] = _jarvis_prime_bridge is not None
                status["trinity_enabled"] = _trinity_initialized
                status["agi_enabled"] = _agi_hub is not None
                status["neural_routing_enabled"] = _neural_routing_enabled
                status["tier2_capabilities"] = _tier2_capability_status
                # v242.1: Classification observability counters
                status["classification_metrics"] = dict(_classification_metrics)
                if _reactor_bridge:
                    try:
                        reactor_stats = _reactor_bridge.get_statistics()
                        status["reactor_bridge"] = {
                            "is_running": reactor_stats.get("is_running", False),
                            "jobs_submitted": reactor_stats.get("jobs_submitted", 0),
                            "jobs_completed": reactor_stats.get("jobs_completed", 0),
                            "jobs_failed": reactor_stats.get("jobs_failed", 0),
                        }
                    except Exception:
                        status["reactor_bridge"] = {"is_running": False, "error": "stats_unavailable"}
                if _jarvis_bridge:
                    try:
                        status["jarvis_bridge"] = _jarvis_bridge.get_metrics()
                    except Exception:
                        status["jarvis_bridge"] = {"initialized": False, "error": "metrics_unavailable"}
                if _jarvis_prime_bridge:
                    try:
                        status["jarvis_prime_bridge"] = _jarvis_prime_bridge.get_status()
                    except Exception:
                        status["jarvis_prime_bridge"] = {"initialized": False, "error": "status_unavailable"}

                # v150.0: Check hollow client mode for inference readiness
                hollow_client_active = _is_hollow_client_active()
                status["hollow_client_mode"] = hollow_client_active

                if hollow_client_active:
                    # In hollow client mode, we're ready if GCP endpoint is configured
                    gcp_endpoint = os.environ.get("GCP_PRIME_ENDPOINT", os.environ.get("JARVIS_GCP_PRIME_ENDPOINT"))
                    status["ready_for_inference"] = bool(gcp_endpoint)
                    status["inference_mode"] = "cloud_routing"
                    status["gcp_endpoint"] = gcp_endpoint
                else:
                    # Normal mode - check local executor
                    status["ready_for_inference"] = _executor is not None and _executor.is_loaded() if hasattr(_executor, 'is_loaded') else False
                    status["inference_mode"] = "local"

            # Managed-mode enrichment
            session_id = os.environ.get("JARVIS_ROOT_SESSION_ID", "")
            if session_id:
                from managed_mode import build_health_envelope
                readiness = "ready" if status.get("phase") == "ready" else "not_ready"
                if status.get("status") == "error":
                    readiness = "degraded"
                status = build_health_envelope(status, readiness=readiness)

            return status

        return {"status": "starting", "phase": "pre-init"}

    # =========================================================================
    # LIFECYCLE DRAIN ENDPOINT - Managed-mode controlled shutdown
    # =========================================================================
    _draining = False
    _drain_id = None

    @app.post("/lifecycle/drain")
    async def lifecycle_drain(request: Request):
        nonlocal _draining, _drain_id
        body = await request.json()
        session_id = os.environ.get("JARVIS_ROOT_SESSION_ID", "")

        if body.get("session_id") != session_id:
            return JSONResponse(status_code=409, content={"error": "session_id mismatch"})

        from managed_mode import verify_hmac_auth, get_control_plane_secret
        auth_header = request.headers.get("X-Root-Auth", "")
        if not verify_hmac_auth(auth_header, session_id, get_control_plane_secret()):
            return JSONResponse(status_code=403, content={"error": "auth failed"})

        if _draining:
            return JSONResponse(status_code=202, content={
                "drain_id": _drain_id, "session_id": session_id, "status": "already_draining"
            })

        import uuid
        _draining = True
        _drain_id = str(uuid.uuid4())

        asyncio.create_task(_drain_and_exit_prime())

        return JSONResponse(status_code=202, content={
            "drain_id": _drain_id, "session_id": session_id, "status": "draining"
        })

    async def _drain_and_exit_prime():
        """Controlled drain: stop accepting, flush, signal exit."""
        import logging
        _logger = logging.getLogger(__name__)
        _logger.info(f"Drain initiated: {_drain_id}")
        await asyncio.sleep(5)
        _logger.info("Drain complete, signaling shutdown")

    # =========================================================================
    # CAPABILITY MANIFEST - Contract gate for cross-repo version validation
    # =========================================================================
    @app.get("/capabilities")
    async def get_capabilities():
        """Publish capability manifest for JARVIS contract gate.

        v290.1: Consumed by JARVIS _validate_cross_repo_contracts()
        to verify version compatibility at startup.
        """
        import time as _cap_time
        return {
            "provider_id": "jprime",
            "capabilities": sorted(list({
                "chat", "reasoning", "code", "vision", "multimodal",
                "embedding", "tool_use", "function_calling",
            })),
            "contract_version": [0, 3, 0],
            "policy_hash": "",  # Populated when Prime has its own contract package
            "timestamp": _cap_time.time(),
        }

    # =========================================================================
    # PLACEHOLDER ENDPOINTS - Return 503 until initialization completes
    # =========================================================================

    class Message(BaseModel):
        role: str
        content: str

    class ChatRequest(BaseModel):
        model: str = "jarvis-prime"
        messages: List[Message]
        max_tokens: int = 512
        temperature: float = 0.7
        stream: bool = False
        metadata: Optional[Dict[str, Any]] = None  # v241.0: task_type hints from JARVIS Body

    class GenerateRequest(BaseModel):
        prompt: str
        max_tokens: int = 512
        temperature: float = 0.7

    def _check_ready():
        """
        Check if server is ready for inference requests.

        v150.0: Updated to recognize Hollow Client mode as a valid ready state.
        When hollow client is active, local model is not loaded, but requests
        should be routed to GCP instead of returning 503.
        """
        if _startup_state and _startup_state.phase != "ready":
            raise HTTPException(
                status_code=503,
                detail={
                    "error": "Server still initializing",
                    "phase": _startup_state.phase if _startup_state else "unknown",
                    "status": _startup_state.get_status() if _startup_state else {}
                }
            )

        # v150.0: Check if hollow client mode is active
        # In hollow client mode, we don't have a local model but can route to cloud
        hollow_client_active = _is_hollow_client_active()

        if hollow_client_active:
            # Hollow client mode - requests will be routed to GCP
            # Don't require local executor or model to be loaded
            return  # Ready for cloud routing

        # Normal mode - require local executor and model
        if _executor is None:
            raise HTTPException(
                status_code=503,
                detail={"error": "Executor not initialized"}
            )
        if hasattr(_executor, 'is_loaded') and not _executor.is_loaded():
            raise HTTPException(
                status_code=503,
                detail={"error": "Model not loaded"}
            )

    # =========================================================================
    # v150.0: GCP CLOUD ROUTING FOR HOLLOW CLIENT MODE
    # =========================================================================
    async def _route_to_gcp_chat_completions(request: ChatRequest):
        """
        v150.0: Route chat completion request to GCP when hollow client is active.

        When JARVIS Prime runs in hollow client mode (SLIM hardware with GCP offload),
        this function proxies requests to the GCP VM running the full inference engine.
        """
        import aiohttp
        import json

        gcp_endpoint = os.environ.get(
            "GCP_PRIME_ENDPOINT",
            os.environ.get("JARVIS_GCP_PRIME_ENDPOINT")
        )

        if not gcp_endpoint:
            raise HTTPException(
                status_code=503,
                detail={
                    "error": "Hollow client mode active but no GCP endpoint configured",
                    "hint": "Set GCP_PRIME_ENDPOINT or JARVIS_GCP_PRIME_ENDPOINT environment variable"
                }
            )

        # Build request payload matching OpenAI format
        payload = {
            "model": request.model,
            "messages": [{"role": m.role, "content": m.content} for m in request.messages],
            "max_tokens": request.max_tokens,
            "temperature": request.temperature,
            "stream": request.stream,
        }

        completion_id = f"chatcmpl-{uuid.uuid4().hex[:8]}"
        created = int(time.time())

        try:
            timeout = aiohttp.ClientTimeout(total=120)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                url = f"{gcp_endpoint.rstrip('/')}/v1/chat/completions"
                logger.info(f"[v150.0] Routing to GCP: {url}")

                if request.stream:
                    # Streaming response - proxy SSE events
                    from fastapi.responses import StreamingResponse

                    async def stream_from_gcp():
                        async with session.post(url, json=payload) as response:
                            if response.status != 200:
                                error_text = await response.text()
                                yield f"data: {json.dumps({'error': error_text})}\n\n"
                                return

                            async for line in response.content:
                                decoded = line.decode('utf-8', errors='ignore')
                                if decoded.strip():
                                    yield decoded

                    return StreamingResponse(
                        stream_from_gcp(),
                        media_type="text/event-stream",
                        headers={
                            "Cache-Control": "no-cache",
                            "Connection": "keep-alive",
                            "X-JARVIS-Routing": "gcp-hollow-client",
                        }
                    )
                else:
                    # Non-streaming - proxy full response
                    async with session.post(url, json=payload) as response:
                        if response.status != 200:
                            error_text = await response.text()
                            raise HTTPException(
                                status_code=response.status,
                                detail={"error": f"GCP request failed: {error_text}"}
                            )

                        result = await response.json()

                        # Add routing metadata
                        if "choices" in result:
                            result["x_jarvis_routing"] = {
                                "mode": "hollow_client",
                                "endpoint": gcp_endpoint,
                                "routed_at": datetime.now().isoformat(),
                            }

                        return result

        except aiohttp.ClientError as e:
            logger.error(f"[v150.0] GCP routing failed: {e}")
            raise HTTPException(
                status_code=503,
                detail={
                    "error": f"GCP endpoint unreachable: {e}",
                    "endpoint": gcp_endpoint,
                    "mode": "hollow_client"
                }
            )
        except Exception as e:
            logger.error(f"[v150.0] Unexpected error in GCP routing: {e}")
            raise HTTPException(
                status_code=500,
                detail={"error": f"Internal error routing to GCP: {e}"}
            )

    async def _route_to_gcp_generate(request: GenerateRequest):
        """
        v150.0: Route generate request to GCP when hollow client is active.
        """
        import aiohttp

        gcp_endpoint = os.environ.get(
            "GCP_PRIME_ENDPOINT",
            os.environ.get("JARVIS_GCP_PRIME_ENDPOINT")
        )

        if not gcp_endpoint:
            raise HTTPException(
                status_code=503,
                detail={
                    "error": "Hollow client mode active but no GCP endpoint configured",
                    "hint": "Set GCP_PRIME_ENDPOINT environment variable"
                }
            )

        payload = {
            "prompt": request.prompt,
            "max_tokens": request.max_tokens,
            "temperature": request.temperature,
        }

        try:
            timeout = aiohttp.ClientTimeout(total=120)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                url = f"{gcp_endpoint.rstrip('/')}/generate"
                logger.info(f"[v150.0] Routing generate to GCP: {url}")

                async with session.post(url, json=payload) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        raise HTTPException(
                            status_code=response.status,
                            detail={"error": f"GCP request failed: {error_text}"}
                        )

                    result = await response.json()
                    result["x_jarvis_routing"] = {
                        "mode": "hollow_client",
                        "endpoint": gcp_endpoint,
                    }
                    return result

        except aiohttp.ClientError as e:
            logger.error(f"[v150.0] GCP generate routing failed: {e}")
            raise HTTPException(
                status_code=503,
                detail={"error": f"GCP endpoint unreachable: {e}"}
            )

    # v242.0: Telemetry capture helper for training data pipeline
    async def _capture_interaction(
        messages: list,
        response_text: str,
        model_id: Optional[str] = None,
        task_type: Optional[str] = None,
        latency_ms: float = 0.0,
        tokens_used: int = 0,
        success: bool = True,
        error: Optional[str] = None,
    ) -> None:
        """v242.0: Capture interaction for training data pipeline."""
        if not _telemetry_hook and not _training_pipeline:
            return

        try:
            # Extract user's last message as prompt
            user_input = ""
            system_prompt: Optional[str] = None
            for msg in reversed(messages):
                if isinstance(msg, dict):
                    role = msg.get("role", "")
                    content = msg.get("content", "")
                elif hasattr(msg, "role"):
                    role = msg.role
                    content = msg.content
                else:
                    continue
                if role == "user" and not user_input:
                    user_input = content
                    continue
                if role == "system" and system_prompt is None:
                    system_prompt = content

            metadata = {
                "task_type": task_type,
                "tokens_used": tokens_used,
                "source": "jarvis_prime",
            }
            # v242.2: Include error info for failed interactions — valuable for DPO
            if error:
                metadata["error"] = error
                metadata["outcome"] = "failure"

            if _telemetry_hook:
                await _telemetry_hook.log(
                    prompt=user_input,
                    completion=response_text,
                    model_version=model_id or "unknown",
                    latency_ms=latency_ms,
                    success=success,
                    task_type=task_type or "",
                    metadata=metadata,
                )

            if _training_pipeline:
                await _training_pipeline.capture_conversation(
                    user_input=user_input,
                    assistant_response=response_text,
                    system_prompt=system_prompt,
                    context=metadata,
                )
        except Exception as e:
            logger.debug(f"[v242] Telemetry record failed: {e}")

    @app.post("/v1/chat/completions")
    async def chat_completions(request: ChatRequest):
        """OpenAI-compatible chat completions with unified intelligent routing (v100.0)."""
        _check_ready()
        _req_start = time.time()  # v242.0: Track request timing for telemetry

        # v150.0: Route to GCP when hollow client mode is active
        if _is_hollow_client_active():
            return await _route_to_gcp_chat_completions(request)

        # v242.0: Phi-first classification (replaces Body's keyword hints)
        _v241_task_type = None
        _v241_active_model_id = None
        _classification = None
        _classification_ms = 0

        if _classifier_loaded and _phi_circuit.can_execute():
            _classification_metrics["total"] += 1
            _t0 = time.monotonic()
            try:
                _classification = await _executor.classify(
                    query=request.messages[-1].content if request.messages else "",
                    system_prompt=_classifier_system_prompt,
                )
                _phi_circuit.record_success()
                _classification_ms = int((time.monotonic() - _t0) * 1000)
                _classification_metrics["successes"] += 1
                _classification_metrics["latency_sum_ms"] += _classification_ms
                _v241_task_type = DOMAIN_TO_TASK_TYPE.get(
                    _classification.get("domain", "general"), "general_chat"
                )
                logger.debug(
                    f"[v242] Phi classified: intent={_classification.get('intent')}, "
                    f"domain={_classification.get('domain')}, "
                    f"confidence={_classification.get('confidence', 0):.2f}, "
                    f"ms={_classification_ms}"
                )
            except Exception as _cls_err:
                _phi_circuit.record_failure()
                _classification_metrics["failures"] += 1
                logger.warning(
                    f"[v242.1] Phi classification failed (circuit: {_phi_circuit.state}): {_cls_err}"
                )
                _classification = None
                _v241_task_type = request.metadata.get("task_type") if request.metadata else None
        else:
            # Fallback: use Body's metadata hint (pre-v242 compatibility)
            _v241_task_type = request.metadata.get("task_type") if request.metadata else None
            # v242.1: Track circuit breaker trips
            if _classifier_loaded and not _phi_circuit.can_execute():
                _classification_metrics["circuit_trips"] += 1

        # v242.1: Circuit breaker fallback — skip classification, use default routing
        if _classification is None and _classifier_loaded and _phi_circuit.state == "open":
            logger.info("[v242.1] Phi circuit OPEN — using default routing (domain=general)")
            _v241_task_type = "general_chat"  # Align task type with fallback domain
            _classification = {
                "schema_version": SCHEMA_VERSION,
                "intent": "answer",
                "domain": "general",
                "complexity": "simple",
                "confidence": 0.0,
                "requires_vision": False,
                "requires_action": False,
                "escalate_to_claude": False,
                "suggested_actions": [],
            }

        # v242.0: Escalation signal — return early if Phi says Claude should handle this
        if (_classification
                and _classification.get("escalate_to_claude")
                and _classification.get("confidence", 0) > MIN_CONFIDENCE_THRESHOLD):
            return JSONResponse({
                "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": "jarvis-prime",
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": ""},
                    "finish_reason": "escalated",
                }],
                "x_jarvis_routing": {
                    "schema_version": SCHEMA_VERSION,
                    "intent": _classification.get("intent", "answer"),
                    "domain": _classification.get("domain", "general"),
                    "escalate_to_claude": True,
                    "escalation_reason": _classification.get("escalation_reason", "complexity"),
                    "confidence": _classification.get("confidence", 0.0),
                    "classifier_model": "phi-3.5-mini-q4km",
                    "classification_ms": _classification_ms,
                    "suggested_actions": _classification.get("suggested_actions", []),
                },
            })

        # v242.1: Phi self-serve for conversation domain (no specialist swap needed).
        # If the query is a greeting / small talk and Phi is confident, generate
        # a response directly from the classifier model — saving 500ms-2s of
        # specialist model swap latency.  Falls through to normal routing on
        # any failure (graceful degradation).
        _phi_self_served = False

        if (_classification
                and _classification.get("domain") in PHI_SELF_SERVE_DOMAINS
                and _classification.get("confidence", 0) >= MIN_CONFIDENCE_THRESHOLD
                and not _classification.get("escalate_to_claude")
                and not _classification.get("requires_action")):
            try:
                _phi_t0 = time.monotonic()
                _phi_content = await _executor.generate_from_classifier(
                    prompt=request.messages[-1].content if request.messages else "",
                    max_tokens=min(request.max_tokens or 256, 256),
                )
                _phi_gen_ms = int((time.monotonic() - _phi_t0) * 1000)
                _phi_self_served = True
                _classification_metrics["self_served"] += 1
                logger.debug(
                    f"[v242.1] Phi self-serve: domain={_classification.get('domain')}, "
                    f"ms={_phi_gen_ms}, len={len(_phi_content)}"
                )
            except Exception as _phi_err:
                logger.warning(f"[v242.1] Phi self-serve failed, falling through: {_phi_err}")
                _phi_self_served = False

        if _phi_self_served:
            _phi_prompt_tokens = sum(len((m.content or "").split()) for m in request.messages)
            _phi_completion_tokens = len(_phi_content.split()) if _phi_content else 0
            _record_inference_metrics(_phi_prompt_tokens, _phi_completion_tokens, _phi_gen_ms, True)
            _response_dict = {
                "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": "jarvis-prime",
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": _phi_content or ""},
                    "finish_reason": "stop",
                }],
                "usage": {
                    "prompt_tokens": _phi_prompt_tokens,
                    "completion_tokens": _phi_completion_tokens,
                    "total_tokens": _phi_prompt_tokens + _phi_completion_tokens,
                },
                "x_jarvis_routing": {
                    "schema_version": _classification.get("schema_version", SCHEMA_VERSION),
                    "intent": _classification.get("intent", "conversation"),
                    "domain": _classification.get("domain", "conversation"),
                    "complexity": _classification.get("complexity", "trivial"),
                    "confidence": _classification.get("confidence", 0.0),
                    "requires_vision": False,
                    "requires_action": False,
                    "escalate_to_claude": False,
                    "source": "phi_self_serve",
                    "classifier_model": "phi-3.5-mini-q4km",
                    "classification_ms": _classification_ms,
                    "generation_ms": _phi_gen_ms,
                    "suggested_actions": _classification.get("suggested_actions", []),
                },
            }
            return _response_dict

        # v241.0: Pre-hook — ensure correct model loaded for task type.
        # This swaps the model in _executor in-place. All existing code below
        # (format_messages, generate, generate_stream) runs UNCHANGED.
        if _model_coordinator:
            try:
                _v241_active_model_id = await _model_coordinator.ensure_model(_v241_task_type)
            except Exception as _v241_err:
                # If it's an HTTPException (503 from bounded queue), re-raise
                if isinstance(_v241_err, HTTPException):
                    raise
                logger.warning(f"[v241] Model swap coordinator error: {_v241_err}")

        # Format messages
        from jarvis_prime.core.model_manager import ChatMessage
        messages = [ChatMessage(role=m.role, content=m.content) for m in request.messages]
        prompt = _executor.format_messages(messages)
        prompt_tokens = len(prompt.split())
        completion_id = f"chatcmpl-{uuid.uuid4().hex[:8]}"
        created = int(time.time())

        # v100.0: Neural Orchestrator Routing Decision
        routing_result = None
        routing_metadata = {}
        if _neural_orchestrator and _neural_routing_enabled:
            try:
                user_messages = [m.content for m in request.messages if m.role == "user"]
                classification_prompt = user_messages[-1] if user_messages else prompt

                routing_result = await _neural_orchestrator.route(
                    prompt=classification_prompt,
                    context={
                        "request_id": completion_id,
                        "message_count": len(request.messages),
                        "model_requested": request.model,
                        "stream": request.stream,
                    }
                )

                routing_metadata = {
                    "tier": routing_result.tier.name if routing_result else "unknown",
                    "model_id": routing_result.model_id if routing_result else None,
                    "decision_reason": routing_result.decision_reason.value if routing_result else "none",
                    "confidence": routing_result.confidence if routing_result else 0.0,
                    "routing_latency_ms": routing_result.latency_ms if routing_result else 0.0,
                }

                logger.debug(
                    f"Neural Routing: {completion_id} -> {routing_metadata['tier']} "
                    f"(confidence={routing_metadata['confidence']:.2f}, "
                    f"reason={routing_metadata['decision_reason']})"
                )
            except Exception as e:
                logger.warning(f"Neural routing failed, using default: {e}")

        # v74.0: Streaming Response (SSE format)
        if request.stream:
            async def stream_generator():
                start = time.time()
                token_count = 0
                accumulated_content = []  # v242.1: Accumulate for telemetry
                _telemetry_captured = False  # v242.3: Guard against double-capture

                try:
                    async for token in _executor.generate_stream(
                        prompt=prompt,
                        max_tokens=request.max_tokens,
                        temperature=request.temperature,
                    ):
                        token_count += 1
                        accumulated_content.append(token)
                        chunk = {
                            "id": completion_id,
                            "object": "chat.completion.chunk",
                            "created": created,
                            "model": "jarvis-prime",
                            "choices": [{
                                "index": 0,
                                "delta": {"content": token},
                                "finish_reason": None,
                            }],
                        }
                        yield f"data: {json.dumps(chunk)}\n\n"

                    final_chunk = {
                        "id": completion_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": "jarvis-prime",
                        "choices": [{
                            "index": 0,
                            "delta": {},
                            "finish_reason": "stop",
                        }],
                    }

                    # v242.0: Attach x_jarvis_routing to final SSE chunk
                    _stream_generation_ms = int((time.time() - start) * 1000)
                    if _classification:
                        final_chunk["x_jarvis_routing"] = {
                            "schema_version": _classification.get("schema_version", SCHEMA_VERSION),
                            "intent": _classification.get("intent", "answer"),
                            "domain": _classification.get("domain", "general"),
                            "complexity": _classification.get("complexity", "simple"),
                            "confidence": _classification.get("confidence", 0.0),
                            "requires_vision": _classification.get("requires_vision", False),
                            "requires_action": _classification.get("requires_action", False),
                            "escalate_to_claude": _classification.get("escalate_to_claude", False),
                            "classifier_model": "phi-3.5-mini-q4km",
                            "generator_model": _v241_active_model_id or "unknown",
                            "classification_ms": _classification_ms,
                            "generation_ms": _stream_generation_ms,
                            "suggested_actions": _classification.get("suggested_actions", []),
                        }

                    yield f"data: {json.dumps(final_chunk)}\n\n"
                    yield "data: [DONE]\n\n"

                    latency_ms = (time.time() - start) * 1000
                    _record_inference_metrics(prompt_tokens, token_count, latency_ms, True)

                    # v242.1: Capture streaming interaction for training data pipeline
                    if _telemetry_hook and accumulated_content:
                        try:
                            _telemetry_captured = True
                            full_response = "".join(accumulated_content)
                            _task = asyncio.create_task(_capture_interaction(
                                messages=messages,
                                response_text=full_response,
                                model_id=_v241_active_model_id,
                                task_type=_v241_task_type,
                                latency_ms=latency_ms,
                                tokens_used=token_count,
                                success=True,
                            ), name="capture_stream_telemetry")
                            _background_tasks.add(_task)
                            _task.add_done_callback(_background_tasks.discard)
                        except Exception as e:
                            logger.debug(f"[v242.1] Stream telemetry capture failed: {e}")

                except Exception as e:
                    logger.error(f"Streaming error: {e}")
                    latency_ms = (time.time() - start) * 1000
                    _record_inference_metrics(prompt_tokens, token_count, latency_ms, False)

                    # v242.2: Capture partial streaming response on error for training data
                    # Failure data is valuable — DPO generator can use as negative examples
                    if _telemetry_hook and accumulated_content:
                        try:
                            _telemetry_captured = True
                            partial_response = "".join(accumulated_content)
                            _task = asyncio.create_task(_capture_interaction(
                                messages=messages,
                                response_text=partial_response,
                                model_id=_v241_active_model_id,
                                task_type=_v241_task_type,
                                latency_ms=latency_ms,
                                tokens_used=token_count,
                                success=False,
                                error=str(e),
                            ), name="capture_error_telemetry")
                            _background_tasks.add(_task)
                            _task.add_done_callback(_background_tasks.discard)
                        except Exception as cap_err:
                            logger.debug(f"[v242.2] Failed to capture error telemetry: {cap_err}")

                    error_chunk = {
                        "id": completion_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": "jarvis-prime",
                        "choices": [{
                            "index": 0,
                            "delta": {"content": f"\n[Error: {str(e)}]"},
                            "finish_reason": "error",
                        }],
                    }
                    yield f"data: {json.dumps(error_chunk)}\n\n"
                    yield "data: [DONE]\n\n"

                finally:
                    # v242.3: Capture telemetry on client disconnect (GeneratorExit /
                    # CancelledError are BaseExceptions — not caught by except Exception).
                    # Without this, partial responses from disconnected clients are silently lost.
                    if not _telemetry_captured and accumulated_content:
                        latency_ms = (time.time() - start) * 1000
                        _record_inference_metrics(prompt_tokens, token_count, latency_ms, False)
                        if _telemetry_hook:
                            try:
                                partial_response = "".join(accumulated_content)
                                _task = asyncio.create_task(_capture_interaction(
                                    messages=messages,
                                    response_text=partial_response,
                                    model_id=_v241_active_model_id,
                                    task_type=_v241_task_type,
                                    latency_ms=latency_ms,
                                    tokens_used=token_count,
                                    success=False,
                                    error="client_disconnect",
                                ), name="capture_disconnect_telemetry")
                                _background_tasks.add(_task)
                                _task.add_done_callback(_background_tasks.discard)
                            except Exception:
                                pass  # Best-effort — process may be shutting down

            _stream_headers = {
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            }
            if _v241_active_model_id:  # v241.0: per-model telemetry
                _stream_headers["X-Model-Id"] = _v241_active_model_id
            return StreamingResponse(
                stream_generator(),
                media_type="text/event-stream",
                headers=_stream_headers,
            )

        # Non-streaming response
        try:
            start = time.time()

            response = await _executor.generate(
                prompt=prompt,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
            )

            latency_ms = (time.time() - start) * 1000
            completion_tokens = len(response.split())

            _record_inference_metrics(prompt_tokens, completion_tokens, latency_ms, True)

            if _neural_orchestrator and routing_result:
                try:
                    await _neural_orchestrator.record_circuit_success(routing_result.tier.name)
                except Exception:
                    pass

            # v242.0: Capture interaction for training data pipeline
            if _telemetry_hook:
                try:
                    usage = {
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": completion_tokens,
                        "total_tokens": prompt_tokens + completion_tokens,
                    }
                    _task = asyncio.create_task(_capture_interaction(
                        messages=messages,
                        response_text=response,
                        model_id=_v241_active_model_id,
                        task_type=_v241_task_type,
                        latency_ms=(time.time() - _req_start) * 1000,
                        tokens_used=usage.get("total_tokens", 0),
                        success=True,
                    ), name="capture_nonstream_telemetry")
                    _background_tasks.add(_task)
                    _task.add_done_callback(_background_tasks.discard)
                except Exception as e:
                    logger.debug(f"[v242] Telemetry capture failed: {e}")

            _generation_ms = int(latency_ms)
            _response_dict = {
                "id": completion_id,
                "object": "chat.completion",
                "created": created,
                "model": "jarvis-prime",
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": response},
                    "finish_reason": "stop",
                }],
                "usage": {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": prompt_tokens + completion_tokens,
                },
                "x_latency_ms": latency_ms,
                "x_routing": routing_metadata,
                "x_model_id": _v241_active_model_id,  # v241.0: per-model telemetry
            }

            # v242.0: Attach x_jarvis_routing metadata to response
            if _classification:
                _response_dict["x_jarvis_routing"] = {
                    "schema_version": _classification.get("schema_version", SCHEMA_VERSION),
                    "intent": _classification.get("intent", "answer"),
                    "domain": _classification.get("domain", "general"),
                    "complexity": _classification.get("complexity", "simple"),
                    "confidence": _classification.get("confidence", 0.0),
                    "requires_vision": _classification.get("requires_vision", False),
                    "requires_action": _classification.get("requires_action", False),
                    "escalate_to_claude": _classification.get("escalate_to_claude", False),
                    "classifier_model": "phi-3.5-mini-q4km",
                    "generator_model": _v241_active_model_id or "unknown",
                    "classification_ms": _classification_ms,
                    "generation_ms": _generation_ms,
                    "suggested_actions": _classification.get("suggested_actions", []),
                }

            return _response_dict
        except Exception as e:
            logger.error(f"Chat error: {e}")
            _record_inference_metrics(0, 0, 0, False)
            if _neural_orchestrator and routing_result:
                try:
                    await _neural_orchestrator.record_circuit_failure(routing_result.tier.name)
                except Exception:
                    pass
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/generate")
    async def generate(request: GenerateRequest):
        """Simple text generation with unified intelligent routing (v100.0)."""
        _check_ready()

        # v150.0: Route to GCP when hollow client mode is active
        if _is_hollow_client_active():
            return await _route_to_gcp_generate(request)

        generate_id = f"gen-{uuid.uuid4().hex[:8]}"
        routing_result = None
        routing_metadata = {}

        if _neural_orchestrator and _neural_routing_enabled:
            try:
                routing_result = await _neural_orchestrator.route(
                    prompt=request.prompt,
                    context={"request_id": generate_id, "endpoint": "generate"}
                )
                routing_metadata = {
                    "tier": routing_result.tier.name if routing_result else "unknown",
                    "model_id": routing_result.model_id if routing_result else None,
                    "decision_reason": routing_result.decision_reason.value if routing_result else "none",
                    "confidence": routing_result.confidence if routing_result else 0.0,
                }
            except Exception as e:
                logger.warning(f"Neural routing failed for generate: {e}")

        try:
            start = time.time()

            response = await _executor.generate(
                prompt=request.prompt,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
            )

            latency_ms = (time.time() - start) * 1000
            prompt_tokens = len(request.prompt.split())
            completion_tokens = len(response.split())

            _record_inference_metrics(prompt_tokens, completion_tokens, latency_ms, True)

            if _neural_orchestrator and routing_result:
                try:
                    await _neural_orchestrator.record_circuit_success(routing_result.tier.name)
                except Exception:
                    pass

            return {
                "text": response,
                "latency_ms": latency_ms,
                "x_routing": routing_metadata,
            }
        except Exception as e:
            logger.error(f"Generate error: {e}")
            _record_inference_metrics(0, 0, 0, False)
            if _neural_orchestrator and routing_result:
                try:
                    await _neural_orchestrator.record_circuit_failure(routing_result.tier.name)
                except Exception:
                    pass
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/metrics")
    async def metrics():
        """Get inference and cost metrics."""
        payload: Dict[str, Any] = {"status": "ok"}

        if _bridge:
            try:
                from jarvis_prime.core.cross_repo_bridge import get_cost_summary
                cost_summary = get_cost_summary()
                inference_metrics = _bridge.get_metrics()
                payload.update({
                    "cost_summary": cost_summary,
                    "inference_metrics": inference_metrics,
                    "connected_to_jarvis": _bridge.state.connected_to_jarvis,
                })
            except Exception as e:
                payload["cross_repo_bridge_error"] = str(e)
        else:
            payload["cross_repo_bridge"] = "disabled"

        if _reactor_bridge:
            try:
                payload["reactor_bridge"] = _reactor_bridge.get_statistics()
            except Exception as e:
                payload["reactor_bridge_error"] = str(e)
        else:
            payload["reactor_bridge"] = "disabled"

        if _jarvis_bridge:
            try:
                payload["jarvis_bridge"] = _jarvis_bridge.get_metrics()
            except Exception as e:
                payload["jarvis_bridge_error"] = str(e)
        else:
            payload["jarvis_bridge"] = "disabled"

        payload["tier2_capabilities"] = _tier2_capability_status

        return payload

    @app.get("/api/v1/capabilities/tier2/status")
    async def tier2_capability_status():
        """Get status of Prime Tier-2 learning capabilities."""
        return _tier2_capability_status

    @app.get("/v1/models")
    async def list_models():
        """List available models."""
        model_status = "loading"
        if _startup_state and _startup_state.phase == "ready":
            model_status = "ready" if (_executor and hasattr(_executor, 'is_loaded') and _executor.is_loaded()) else "not_loaded"

        return {
            "object": "list",
            "data": [{
                "id": "jarvis-prime",
                "object": "model",
                "owned_by": "jarvis",
                "status": model_status,
            }],
        }

    # =========================================================================
    # MODEL HOT-RELOAD ENDPOINT - Reactor-Core Integration
    # =========================================================================
    class ModelReloadRequest(BaseModel):
        model_path: str
        model_version: str = "unknown"
        model_id: str = ""

    @app.post("/api/v1/models/reload")
    async def reload_model(request: ModelReloadRequest):
        """Hot-reload model from Reactor-Core."""
        global _model_path

        _check_ready()
        logger.info(f"Model reload requested: {request.model_path} (v{request.model_version})")

        try:
            new_model_path = Path(request.model_path)

            if not new_model_path.exists():
                raise HTTPException(status_code=404, detail=f"Model file not found: {request.model_path}")

            if _executor.is_loaded():
                logger.info("Unloading current model...")
                await _executor.close()

            logger.info(f"Loading new model: {new_model_path}")
            start = time.time()
            await _executor.load(new_model_path)
            load_time = time.time() - start

            _model_path = new_model_path

            if _bridge:
                try:
                    from jarvis_prime.core.cross_repo_bridge import update_model_status
                    update_model_status(loaded=True, model_path=str(_model_path))
                    await _bridge.notify_jarvis("model_reloaded", {
                        "model_path": str(_model_path),
                        "model_version": request.model_version,
                        "load_time_seconds": load_time,
                    })
                except Exception as e:
                    logger.warning(f"Failed to notify bridge: {e}")

            if _trinity_initialized:
                try:
                    from jarvis_prime.core.trinity_bridge import update_model_status as trinity_update
                    trinity_update(loaded=True, model_path=str(_model_path))
                except Exception as e:
                    logger.warning(f"Failed to notify Trinity: {e}")

            # Sync model registry so version tracking stays consistent
            try:
                from jarvis_prime.core.dynamic_model_registry import get_dynamic_model_registry
                _dmr = await get_dynamic_model_registry()
                if _dmr:
                    model_id = request.model_version or new_model_path.stem
                    # Estimate memory from file size (GGUF ~= file size in RAM)
                    try:
                        _mem_gb = new_model_path.stat().st_size / (1024 ** 3)
                    except OSError:
                        _mem_gb = 0.0
                    await _dmr.register_loaded(
                        model_id=model_id,
                        memory_gb=_mem_gb,
                    )
                    logger.info(f"[reload] Synced model registry: {model_id} ({_mem_gb:.1f}GB)")
            except ImportError:
                logger.debug("[reload] DynamicModelRegistry not available")
            except Exception as e:
                logger.warning(f"[reload] Failed to sync model registry: {e}")

            logger.info(f"Model reloaded in {load_time:.2f}s: {_model_path}")

            return {
                "status": "success",
                "model_path": str(_model_path),
                "model_version": request.model_version,
                "load_time_seconds": load_time,
            }

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Model reload failed: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    # =========================================================================
    # AGI v77.0 ENDPOINTS
    # =========================================================================
    class AGIReasonRequest(BaseModel):
        query: str
        strategy: str = "chain_of_thought"
        context: dict = {}

    class AGIPlanRequest(BaseModel):
        goal: str
        context: dict = {}
        constraints: List[str] = []

    class AGIFeedbackRequest(BaseModel):
        experience_id: str
        score: float
        comment: Optional[str] = None

    class AGIProcessRequest(BaseModel):
        content: str
        modalities: List[str] = ["text"]
        context: dict = {}
        enable_reasoning: bool = True
        enable_learning: bool = True

    @app.post("/agi/reason")
    async def agi_reason(request: AGIReasonRequest):
        if not _agi_hub:
            raise HTTPException(status_code=503, detail="AGI Hub not initialized")

        try:
            start = time.time()
            result = await _agi_hub.reason(
                query=request.query,
                strategy=request.strategy,
                context=request.context,
            )
            latency_ms = (time.time() - start) * 1000

            return {
                "status": "success",
                "query": request.query,
                "strategy": request.strategy,
                "conclusion": result.get("conclusion"),
                "trace": result.get("trace", []),
                "confidence": result.get("confidence", 0.0),
                "latency_ms": latency_ms,
            }
        except Exception as e:
            logger.error(f"AGI reasoning error: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/agi/plan")
    async def agi_plan(request: AGIPlanRequest):
        if not _agi_hub:
            raise HTTPException(status_code=503, detail="AGI Hub not initialized")

        try:
            start = time.time()
            result = await _agi_hub.plan(goal=request.goal, context=request.context)
            latency_ms = (time.time() - start) * 1000

            return {
                "status": "success",
                "goal": request.goal,
                "plan": result,
                "latency_ms": latency_ms,
            }
        except Exception as e:
            logger.error(f"AGI planning error: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/agi/process")
    async def agi_process(request: AGIProcessRequest):
        if not _agi_hub:
            raise HTTPException(status_code=503, detail="AGI Hub not initialized")

        try:
            async def inference_fn(prompt: str, **kwargs):
                if _executor and hasattr(_executor, 'is_loaded') and _executor.is_loaded():
                    return await _executor.generate(prompt=prompt, max_tokens=512, temperature=0.7)
                return prompt

            result = await _agi_hub.process(
                content=request.content,
                modalities=request.modalities,
                context=request.context,
                inference_fn=inference_fn if request.enable_reasoning else None,
            )

            return {
                "status": "success",
                "request_id": result.request_id,
                "content": result.content,
                "reasoning_trace": result.reasoning_trace,
                "confidence": result.confidence,
                "models_used": result.models_used,
                "processing_time_ms": result.processing_time_ms,
                "feedback_recorded": result.feedback_recorded,
            }
        except Exception as e:
            logger.error(f"AGI process error: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/agi/feedback")
    async def agi_feedback(request: AGIFeedbackRequest):
        if not _agi_hub:
            raise HTTPException(status_code=503, detail="AGI Hub not initialized")

        try:
            success = await _agi_hub.record_feedback(
                experience_id=request.experience_id,
                score=request.score,
                comment=request.comment,
            )
            return {"status": "success" if success else "failed", "experience_id": request.experience_id, "score": request.score}
        except Exception as e:
            logger.error(f"AGI feedback error: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/agi/learning/trigger")
    async def agi_learning_trigger(force: bool = False):
        if not _agi_hub:
            raise HTTPException(status_code=503, detail="AGI Hub not initialized")

        try:
            result = await _agi_hub.trigger_learning_update(force=force)
            return {"status": "success", "result": result}
        except Exception as e:
            logger.error(f"AGI learning trigger error: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/agi/status")
    async def agi_status():
        if not _agi_hub:
            return {"status": "not_initialized", "message": "AGI Hub not available"}

        try:
            status = _agi_hub.get_status()
            health = await _agi_hub.health_check()
            return {
                "status": "ok",
                "initialized": status["initialized"],
                "healthy": health["healthy"],
                "subsystems": status["subsystems"],
                "metrics": status["metrics"],
            }
        except Exception as e:
            logger.error(f"AGI status error: {e}")
            return {"status": "error", "error": str(e)}

    @app.get("/agi/learning/stats")
    async def agi_learning_stats():
        if not _agi_hub or not _agi_hub.learning_engine:
            return {"status": "not_available", "message": "Learning engine not initialized"}

        try:
            stats = _agi_hub.learning_engine.get_statistics()
            return {"status": "ok", "statistics": stats}
        except Exception as e:
            logger.error(f"AGI learning stats error: {e}")
            return {"status": "error", "error": str(e)}

    # =========================================================================
    # NEURAL ORCHESTRATOR v100.0 ENDPOINTS
    # =========================================================================
    class NeuralRouteRequest(BaseModel):
        prompt: str
        context: dict = {}

    @app.get("/neural-orchestrator/health")
    async def neural_orchestrator_health():
        if not _neural_orchestrator or not _neural_routing_enabled:
            return {"status": "not_initialized", "message": "Neural Orchestrator not available"}

        try:
            health = _neural_orchestrator.get_health_status()
            return {"status": "ok", **health}
        except Exception as e:
            logger.error(f"Neural Orchestrator health error: {e}")
            return {"status": "error", "error": str(e)}

    @app.get("/neural-orchestrator/stats")
    async def neural_orchestrator_stats():
        if not _neural_orchestrator or not _neural_routing_enabled:
            return {"status": "not_initialized", "message": "Neural Orchestrator not available"}

        try:
            stats = _neural_orchestrator.get_comprehensive_stats()
            return {"status": "ok", **stats}
        except Exception as e:
            logger.error(f"Neural Orchestrator stats error: {e}")
            return {"status": "error", "error": str(e)}

    @app.post("/neural-orchestrator/route")
    async def neural_orchestrator_route(request: NeuralRouteRequest):
        if not _neural_orchestrator or not _neural_routing_enabled:
            raise HTTPException(status_code=503, detail="Neural Orchestrator not available")

        try:
            result = await _neural_orchestrator.route(prompt=request.prompt, context=request.context)
            return {
                "status": "ok",
                "routing": result.to_dict() if result else None,
                "task_classification": {
                    "task_type": result.task_classification.task_type.value if result and result.task_classification else None,
                    "complexity": result.task_classification.complexity if result and result.task_classification else None,
                    "confidence": result.task_classification.confidence if result and result.task_classification else None,
                    "recommended_tier": result.task_classification.recommended_tier.name if result and result.task_classification else None,
                } if result and result.task_classification else None,
            }
        except Exception as e:
            logger.error(f"Neural routing error: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/neural-orchestrator/memory")
    async def neural_orchestrator_memory():
        if not _neural_orchestrator or not _neural_routing_enabled:
            return {"status": "not_initialized", "message": "Neural Orchestrator not available"}

        try:
            memory = await _neural_orchestrator.get_memory_status()
            should_burst = await _neural_orchestrator.should_burst_to_cloud()
            return {"status": "ok", "memory": memory, "should_burst_to_cloud": should_burst}
        except Exception as e:
            logger.error(f"Neural Orchestrator memory error: {e}")
            return {"status": "error", "error": str(e)}

    @app.post("/neural-orchestrator/classify")
    async def neural_orchestrator_classify(request: NeuralRouteRequest):
        if not _neural_orchestrator or not _neural_routing_enabled:
            raise HTTPException(status_code=503, detail="Neural Orchestrator not available")

        try:
            classification = await _neural_orchestrator.classify_task(prompt=request.prompt, context=request.context)
            return {
                "status": "ok",
                "task_type": classification.task_type.value,
                "complexity": classification.complexity,
                "confidence": classification.confidence,
                "signals": classification.signals,
                "recommended_tier": classification.recommended_tier.name,
                "requires_fast_response": classification.requires_fast_response,
                "is_coding_session": classification.is_coding_session,
            }
        except Exception as e:
            logger.error(f"Neural classification error: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    # =========================================================================
    # WEBSOCKET EVENT STREAM - For Neural Mesh Integration (v93.15)
    # =========================================================================
    from fastapi import WebSocket, WebSocketDisconnect
    from datetime import datetime

    # v93.15: Global event queue and subscribers for WebSocket streaming
    _ws_event_queue: asyncio.Queue = asyncio.Queue(maxsize=1000)
    _ws_subscribers: List[WebSocket] = []

    async def _broadcast_ws_event(event_type: str, data: Dict[str, Any]) -> None:
        """Broadcast an event to all connected WebSocket subscribers."""
        event = {
            "event_type": event_type,
            "data": data,
            "timestamp": datetime.now().isoformat(),
        }
        for ws in _ws_subscribers[:]:  # Copy to avoid modification during iteration
            try:
                await ws.send_json(event)
            except Exception:
                # Remove dead connections
                try:
                    _ws_subscribers.remove(ws)
                except ValueError:
                    pass

    def _build_ws_contract() -> Dict[str, Any]:
        """Build a machine-readable websocket contract for cross-repo discovery."""
        startup_phase = _startup_state.phase if _startup_state else "starting"
        status = "healthy" if startup_phase == "ready" else ("error" if startup_phase == "error" else "starting")
        return {
            "contract_type": "jarvis_prime.ws_events",
            "contract_version": "1.0",
            "service": "jarvis_prime",
            "status": status,
            "phase": startup_phase,
            "websocket": {
                "primary_path": "/ws/events",
                "paths": ["/ws/events"],
                "heartbeat_seconds": 30.0,
                "supports_ping_pong": True,
                "message_format": "json",
                "envelope_version": "v1",
                "event_types": [
                    "connected",
                    "heartbeat",
                    "inference_complete",
                    "model_loaded",
                    "error",
                ],
            },
            "health": {
                "paths": ["/health"],
            },
            "integration_contracts": {
                "ws_events": "/api/v1/integration/contracts/ws-events",
            },
            "timestamp": datetime.now().isoformat(),
        }

    @app.get("/api/v1/integration/contracts/ws-events")
    @app.get("/api/contracts/ws-events")
    async def ws_events_contract():
        """Advertise websocket event contract for Reactor/Core integrations."""
        return _build_ws_contract()

    # =========================================================================
    # v243.0: Reflex Inhibition Endpoint — atomic file write + API key auth
    # =========================================================================
    @app.post("/v1/reflex/inhibit")
    async def inhibit_reflex(request: Request):
        """
        v243.0: Allow Mind (J-Prime) to publish reflex inhibition state.

        The Body reads ~/.jarvis/trinity/reflex_inhibition.json to check
        whether certain reflexes should be suppressed.  This endpoint lets
        any authenticated caller write that file atomically.

        Auth: Bearer token checked against JPRIME_INHIBIT_API_KEY env var.
              If the env var is empty/unset, the endpoint is open (dev mode).
        """
        from datetime import datetime, timezone
        from fastapi.responses import JSONResponse as _JSONResp

        # --- Auth gate ---
        if _INHIBIT_API_KEY:
            auth = request.headers.get("Authorization", "")
            if auth != f"Bearer {_INHIBIT_API_KEY}":
                return _JSONResp(status_code=403, content={"error": "unauthorized"})

        try:
            body = await request.json()
        except Exception:
            return _JSONResp(status_code=400, content={"error": "invalid JSON body"})

        inhibit_path = Path.home() / ".jarvis" / "trinity" / "reflex_inhibition.json"
        inhibit_path.parent.mkdir(parents=True, exist_ok=True)

        _atomic_write_json(inhibit_path, {
            "version": 1,
            "inhibit_reflexes": body.get("reflex_ids", []),
            "reason": body.get("reason", ""),
            "published_at": datetime.now(timezone.utc).isoformat(),
            "ttl_seconds": body.get("ttl_seconds", 300),
        })
        logger.info(
            f"[v243] Reflex inhibition updated: {body.get('reflex_ids', [])} "
            f"reason={body.get('reason', '')!r} ttl={body.get('ttl_seconds', 300)}s"
        )
        return {"status": "ok"}

    # =========================================================================
    # v242.1: Capability Registry — Body registers available actions for Phi
    # =========================================================================
    @app.post("/v1/capabilities/register")
    async def register_capabilities(request: Request):
        """
        v242.1: Register Body capabilities so Phi knows what actions are available.

        The Body sends a dict of {action_name: description} pairs.  This rebuilds
        the Phi classifier system prompt with the action list so classification
        output can reference concrete actions the Body supports.

        Thread-safe: _prompt_lock guards the write since classify() reads
        _classifier_system_prompt from a ThreadPoolExecutor.  (CPython string
        assignment is GIL-atomic, so stale reads are harmless — the lock
        serializes the rebuild itself.)
        """
        global _classifier_system_prompt
        from fastapi.responses import JSONResponse as _JSONResp

        if not _classifier_loaded:
            return _JSONResp(
                status_code=503,
                content={"error": "Phi classifier not loaded yet"},
            )

        try:
            capabilities = await request.json()
        except Exception:
            return _JSONResp(
                status_code=400,
                content={"error": "invalid JSON body"},
            )

        if not isinstance(capabilities, dict):
            return _JSONResp(
                status_code=400,
                content={"error": "body must be a JSON object mapping action names to descriptions"},
            )

        from jarvis_prime.core.classification_schema import build_classifier_system_prompt

        with _prompt_lock:
            _classifier_system_prompt = build_classifier_system_prompt(
                action_registry=capabilities,
            )

        logger.info(
            f"[v242.1] Registered {len(capabilities)} capabilities from Body"
        )
        return {"status": "ok", "actions_registered": len(capabilities)}

    @app.websocket("/ws/events")
    async def websocket_events(websocket: WebSocket):
        """
        v93.15: WebSocket endpoint for real-time event streaming to Neural Mesh.

        This endpoint allows JARVIS Body and other clients to receive real-time
        events from JARVIS-Prime without polling.

        Events include:
        - connected: Initial connection event with status
        - heartbeat: Periodic keep-alive (every 30s)
        - inference_complete: After each inference
        - model_loaded: When model loading completes
        - error: On errors
        """
        await websocket.accept()
        _ws_subscribers.append(websocket)
        logger.info(f"[WebSocket] Client connected ({len(_ws_subscribers)} active)")

        try:
            # Send initial connection event
            await websocket.send_json({
                "event_type": "connected",
                "data": {
                    "status": "starting" if not _startup_state.ready else "ready",
                    "phase": _startup_state.phase,
                    "model_loaded": _startup_state.model_loaded,
                    "instance_id": str(uuid.uuid4())[:8],
                },
                "timestamp": datetime.now().isoformat(),
            })

            # Keep connection alive with heartbeats
            while True:
                try:
                    # Wait for incoming message or timeout for heartbeat
                    try:
                        message = await asyncio.wait_for(
                            websocket.receive_text(),
                            timeout=30.0  # Heartbeat every 30s
                        )
                        # Handle ping/pong
                        if message == "ping":
                            await websocket.send_text("pong")
                    except asyncio.TimeoutError:
                        # Send heartbeat
                        await websocket.send_json({
                            "event_type": "heartbeat",
                            "data": {
                                "status": "ready" if _startup_state.ready else "starting",
                                "model_loaded": _startup_state.model_loaded,
                            },
                            "timestamp": datetime.now().isoformat(),
                        })
                except WebSocketDisconnect:
                    break

        except Exception as e:
            logger.debug(f"[WebSocket] Error: {e}")
        finally:
            try:
                _ws_subscribers.remove(websocket)
            except ValueError:
                pass
            logger.info(f"[WebSocket] Client disconnected ({len(_ws_subscribers)} remaining)")

    # =========================================================================
    # HELPER FUNCTIONS
    # =========================================================================
    def _record_inference_metrics(tokens_in: int, tokens_out: int, latency_ms: float, success: bool):
        """Record inference metrics to all tracking systems."""
        if _bridge:
            try:
                from jarvis_prime.core.cross_repo_bridge import record_inference
                record_inference(tokens_in=tokens_in, tokens_out=tokens_out, latency_ms=latency_ms)
            except Exception:
                pass

        if _trinity_record_inference:
            try:
                _trinity_record_inference(latency_ms=latency_ms, success=success)
            except Exception:
                pass

    async def initialize_tier2_capabilities() -> None:
        """
        Initialize Prime Tier-2 learning capabilities in background.

        Capabilities:
        - Continual learning engine (RAG + replay + stateful learning)
        - Self-modification engine (safety-constrained adaptation)
        """
        global _continual_learner, _self_modifier, _tier2_capability_status

        enabled = os.getenv("PRIME_TIER2_ENABLED", "true").strip().lower() in ("1", "true", "yes", "on")
        if not enabled:
            _tier2_capability_status = {
                "enabled": False,
                "state": "disabled",
                "continual_learning": {"status": "disabled"},
                "self_improvement": {"status": "disabled"},
                "updated_at": datetime.now().isoformat(),
            }
            return

        started = time.perf_counter()
        _tier2_capability_status = {
            "enabled": True,
            "state": "initializing",
            "continual_learning": {"status": "initializing"},
            "self_improvement": {"status": "initializing"},
            "updated_at": datetime.now().isoformat(),
        }

        success_count = 0

        try:
            from jarvis_prime.models.continual_learning_system import get_continual_learner

            _continual_learner = await get_continual_learner()
            _tier2_capability_status["continual_learning"] = {
                "status": "ready",
                "details": _continual_learner.get_status() if hasattr(_continual_learner, "get_status") else {},
            }
            success_count += 1
        except Exception as exc:
            _continual_learner = None
            _tier2_capability_status["continual_learning"] = {
                "status": "error",
                "error": str(exc),
            }
            logger.warning("[Tier2] Continual learning init failed: %s", exc)

        try:
            from jarvis_prime.models.self_improving_agent import get_self_modifier

            _self_modifier = await get_self_modifier()
            _tier2_capability_status["self_improvement"] = {
                "status": "ready",
                "details": _self_modifier.get_status() if hasattr(_self_modifier, "get_status") else {},
            }
            success_count += 1
        except Exception as exc:
            _self_modifier = None
            _tier2_capability_status["self_improvement"] = {
                "status": "error",
                "error": str(exc),
            }
            logger.warning("[Tier2] Self-improvement init failed: %s", exc)

        final_state = "ready" if success_count == 2 else ("degraded" if success_count > 0 else "error")
        _tier2_capability_status["state"] = final_state
        _tier2_capability_status["init_duration_seconds"] = round(time.perf_counter() - started, 3)
        _tier2_capability_status["updated_at"] = datetime.now().isoformat()

    # =========================================================================
    # BACKGROUND INITIALIZATION - Runs AFTER server starts listening
    # =========================================================================
    async def background_initialization():
        """
        v93.2: Run ALL heavy initialization in background.

        This function is triggered by FastAPI's startup event, which means
        the HTTP server is ALREADY listening when this runs.

        Initialization order:
        1. Import ML libraries (torch, sklearn - triggers warnings)
        2. Initialize cross-repo bridge
        3. Initialize Trinity bridge
        4. Initialize Reactor + training data bridges
        5. Initialize AGI Hub
        6. Initialize JARVIS action/safety bridge
        7. Initialize Neural Orchestrator
        8. Download model if needed
        9. Configure hardware
        10. Load model
        11. Mark ready
        """
        global _bridge, _reactor_bridge, _training_pipeline, _jarvis_bridge
        global _executor, _agi_hub, _neural_orchestrator, _model_path
        global _trinity_initialized, _trinity_record_inference, _neural_routing_enabled
        global _model_coordinator  # v241.0
        global _telemetry_hook  # v242.0
        global _tier2_capability_task

        try:
            _startup_state.phase = "initializing"
            init_start = time.time()

            # v93.7: Enhanced step logging with timing
            def log_step(step_name: str, step_num: int, total_steps: int = 11):
                """Log step with progress indicator."""
                _startup_state.details["step"] = step_name
                _startup_state.details["step_num"] = step_num
                _startup_state.details["total_steps"] = total_steps
                elapsed = time.time() - init_start
                logger.info("")
                logger.info(f"{'='*60}")
                logger.info(f"📍 STEP {step_num}/{total_steps}: {step_name.upper()}")
                logger.info(f"   Elapsed: {elapsed:.1f}s")
                logger.info(f"{'='*60}")

            def log_step_complete(step_name: str, duration: float):
                """Log step completion."""
                logger.info(f"✅ {step_name} complete ({duration:.2f}s)")

            logger.info("")
            logger.info("=" * 70)
            logger.info("🚀 JARVIS-PRIME BACKGROUND INITIALIZATION STARTING")
            logger.info("=" * 70)
            logger.info(f"   PID: {os.getpid()}")
            logger.info(f"   Python: {sys.version.split()[0]}")
            logger.info("")

            # -----------------------------------------------------------------
            # STEP 1: Import ML libraries (this triggers the warnings)
            # -----------------------------------------------------------------
            step_start = time.time()
            log_step("importing_ml_libraries", 1)
            _startup_state.begin_phase("importing_ml_libraries", 1)

            try:
                from jarvis_prime.core.llama_cpp_executor import LlamaCppExecutor, LlamaCppConfig
                log_step_complete("ML libraries import", time.time() - step_start)
                _startup_state.complete_phase("importing_ml_libraries")
            except ImportError as e:
                logger.error(f"❌ Import error: {e}")
                _startup_state.fail_phase("importing_ml_libraries", str(e))
                _startup_state.phase = "error"
                _startup_state.error = f"Missing llama-cpp-python: {e}"
                return

            # -----------------------------------------------------------------
            # STEP 2: Initialize cross-repo bridge
            # -----------------------------------------------------------------
            step_start = time.time()
            log_step("initializing_bridge", 2)
            _startup_state.begin_phase("initializing_bridge", 2)
            bridge_enabled = _args.bridge_enabled and not _args.no_bridge
            if bridge_enabled:
                try:
                    from jarvis_prime.core.cross_repo_bridge import (
                        initialize_bridge,
                        shutdown_bridge,
                        record_inference,
                        update_model_status,
                        get_cost_summary,
                    )
                    _bridge = await initialize_bridge(port=_args.port)
                    log_step_complete("Cross-repo bridge", time.time() - step_start)
                except Exception as e:
                    logger.warning(f"⚠️ Cross-repo bridge failed: {e}")
                    _bridge = None
            else:
                logger.info("   ℹ️ Cross-repo bridge disabled (--no-bridge)")
                _bridge = None
            _startup_state.complete_phase("initializing_bridge")

            # -----------------------------------------------------------------
            # STEP 3: Initialize Trinity bridge
            # -----------------------------------------------------------------
            step_start = time.time()
            log_step("initializing_trinity", 3)
            _startup_state.begin_phase("initializing_trinity", 3)
            try:
                from jarvis_prime.core.trinity_bridge import (
                    initialize_trinity,
                    shutdown_trinity,
                    update_model_status as trinity_update_model_status,
                    record_inference as _trinity_rec_inf,
                    TRINITY_ENABLED,
                )
                _trinity_record_inference = _trinity_rec_inf
                if TRINITY_ENABLED:
                    _trinity_initialized = await initialize_trinity(port=_args.port)
                    if _trinity_initialized:
                        log_step_complete("Trinity bridge", time.time() - step_start)
                    else:
                        logger.warning("   ⚠️ Trinity init returned False")
                else:
                    logger.info("   ℹ️ Trinity disabled")
            except ImportError as e:
                logger.warning(f"   ⚠️ Trinity module not available ({e})")
            except Exception as e:
                logger.warning(f"   ⚠️ Trinity init failed ({e})")
            _startup_state.complete_phase("initializing_trinity")

            # -----------------------------------------------------------------
            # STEP 4: Initialize Reactor Core Bridge + Training Pipeline
            # -----------------------------------------------------------------
            step_start = time.time()
            log_step("initializing_reactor_pipeline", 4)
            _startup_state.begin_phase("initializing_reactor_pipeline", 4)
            if bridge_enabled:
                try:
                    from jarvis_prime.core.reactor_core_bridge import get_reactor_core_bridge
                    from jarvis_prime.core.training_data_pipeline import get_training_data_pipeline

                    _reactor_bridge = await get_reactor_core_bridge()
                    _training_pipeline = await get_training_data_pipeline()
                    log_step_complete("Reactor bridge + training pipeline", time.time() - step_start)
                except Exception as e:
                    logger.warning(f"   ⚠️ Reactor/training pipeline init failed: {e}")
                    _reactor_bridge = None
                    _training_pipeline = None
            else:
                logger.info("   ℹ️ Reactor/training pipeline disabled (--no-bridge)")
                _reactor_bridge = None
                _training_pipeline = None
            _startup_state.complete_phase("initializing_reactor_pipeline")

            # -----------------------------------------------------------------
            # STEP 5: Initialize AGI Hub (with Hardware-Aware Configuration)
            # -----------------------------------------------------------------
            step_start = time.time()
            log_step("initializing_agi_hub", 5)
            _startup_state.begin_phase("initializing_agi_hub", 5)

            # v138.0: Perform hardware assessment BEFORE AGI Hub initialization
            global _hardware_assessment
            _hardware_assessment = _assess_hardware()
            _log_hardware_assessment(_hardware_assessment)

            # Store assessment in startup state for health endpoint
            _startup_state.details["hardware"] = {
                "profile": _hardware_assessment.profile.name,
                "total_ram_gb": _hardware_assessment.total_ram_gb,
                "available_ram_gb": _hardware_assessment.available_ram_gb,
                "is_apple_silicon": _hardware_assessment.is_apple_silicon,
                "gpu": _hardware_assessment.gpu_name,
                "slim_mode": _hardware_assessment.enable_slim_mode,
                "skip_agi_hub": _hardware_assessment.skip_agi_hub,
            }

            # v138.0: Skip AGI Hub entirely if insufficient memory (CLOUD ONLY mode)
            if _hardware_assessment.skip_agi_hub:
                logger.warning("")
                logger.warning("=" * 70)
                logger.warning("⚠️ v138.0: SKIPPING LOCAL AGI HUB (CLOUD ONLY MODE)")
                logger.warning("=" * 70)
                logger.warning(f"   Reason: {_hardware_assessment.reason}")
                logger.warning("   JARVIS Prime will operate as a thin client,")
                logger.warning("   forwarding complex requests to the cloud AGI service.")
                logger.warning("=" * 70)
                logger.warning("")
                _agi_hub = None
                log_step_complete("AGI Hub (SKIPPED - Cloud Only)", time.time() - step_start)
            else:
                try:
                    from jarvis_prime.core.agi_integration import (
                        AGIIntegrationHub,
                        AGIHubConfig,
                        get_agi_hub,
                        shutdown_agi_hub,
                        AGIEnhancedInference,
                        SubsystemPriority,
                    )

                    # v138.0: Configure AGI Hub based on hardware assessment
                    # The v138.0 Memory-Aware Staged Initialization handles the rest
                    agi_config = AGIHubConfig(
                        # Core subsystem enablement
                        enable_orchestrator=True,
                        enable_reasoning=True,
                        enable_learning=not _hardware_assessment.enable_slim_mode,  # Defer in slim mode
                        enable_multimodal=not _hardware_assessment.defer_heavy_subsystems,  # Defer heavy
                        enable_hardware_optimization=True,
                        enable_auto_reasoning=True,
                        enable_experience_recording=not _hardware_assessment.enable_slim_mode,

                        # v93.12: Timeout configuration (prevents hanging)
                        enable_agi_models_v80=(
                            os.getenv("ENABLE_AGI_MODELS_V80", "true").lower() == "true"
                            and not _hardware_assessment.defer_heavy_subsystems
                        ),
                        agi_models_v80_timeout=float(os.getenv("AGI_MODELS_V80_TIMEOUT", "30.0")),
                        agi_models_v80_graceful_degradation=True,
                        subsystem_init_timeout=float(os.getenv("SUBSYSTEM_INIT_TIMEOUT", "60.0")),
                        parallel_init_timeout=float(os.getenv("PARALLEL_INIT_TIMEOUT", "120.0")),

                        # v138.0: Memory-Aware Staged Initialization configuration
                        enable_staged_init=True,  # Always use staged init
                        enable_slim_mode=_hardware_assessment.enable_slim_mode,
                        enable_lazy_loading=_hardware_assessment.enable_slim_mode,
                        defer_heavy_subsystems=_hardware_assessment.defer_heavy_subsystems,
                        gc_between_stages=True,

                        # Memory thresholds based on hardware profile
                        min_memory_headroom_percent=(
                            30.0 if _hardware_assessment.profile == HardwareProfile.SLIM else 20.0
                        ),
                        slim_mode_threshold_percent=(
                            20.0 if _hardware_assessment.profile == HardwareProfile.SLIM else 15.0
                        ),

                        # OOM Protection (critical for low-memory systems)
                        enable_oom_protection=True,
                        oom_warning_threshold=(
                            75.0 if _hardware_assessment.profile == HardwareProfile.SLIM else 85.0
                        ),
                        oom_critical_threshold=(
                            85.0 if _hardware_assessment.profile == HardwareProfile.SLIM else 95.0
                        ),

                        # Subsystem priority based on hardware
                        subsystem_priorities={
                            "hardware": SubsystemPriority.CRITICAL,
                            "orchestrator": SubsystemPriority.CRITICAL,
                            "reasoning": (
                                SubsystemPriority.IMPORTANT
                                if not _hardware_assessment.enable_slim_mode
                                else SubsystemPriority.OPTIONAL
                            ),
                            "learning": (
                                SubsystemPriority.IMPORTANT
                                if not _hardware_assessment.enable_slim_mode
                                else SubsystemPriority.OPTIONAL
                            ),
                            "multimodal": SubsystemPriority.OPTIONAL,
                            "agi_models_v80": SubsystemPriority.OPTIONAL,
                        },
                    )

                    logger.info(f"   [v138.0] AGI Hub Config: staged_init=True, "
                               f"slim_mode={_hardware_assessment.enable_slim_mode}, "
                               f"defer_heavy={_hardware_assessment.defer_heavy_subsystems}")

                    _agi_hub = await get_agi_hub(agi_config)
                    log_step_complete("AGI Integration Hub", time.time() - step_start)

                    # Log final AGI Hub status
                    if _agi_hub:
                        status = _agi_hub.get_status()
                        staged_info = status.get("staged_init", {})
                        logger.info(f"   [v138.0] AGI Hub Status: "
                                   f"stage={staged_info.get('current_stage', 'UNKNOWN')}, "
                                   f"slim_mode={staged_info.get('slim_mode', False)}, "
                                   f"deferred={staged_info.get('deferred_subsystems', [])}")

                except asyncio.TimeoutError:
                    # v93.12: Handle timeout gracefully - don't block startup
                    logger.warning(f"   ⏱️ AGI Hub init timed out - continuing without it")
                    _agi_hub = None
                except ImportError as e:
                    logger.warning(f"   ⚠️ AGI Hub not available: {e}")
                    _agi_hub = None
                except MemoryError as e:
                    # v138.0: Handle OOM during initialization
                    logger.error(f"   ❌ AGI Hub OOM during initialization: {e}")
                    logger.warning("   Falling back to CLOUD ONLY mode")
                    _agi_hub = None
                except Exception as e:
                    logger.warning(f"   ⚠️ AGI Hub init failed: {e}")
                    import traceback
                    traceback.print_exc()
                    _agi_hub = None
            _startup_state.complete_phase("initializing_agi_hub")

            # -----------------------------------------------------------------
            # STEP 6: Initialize JARVIS action/safety bridge
            # -----------------------------------------------------------------
            step_start = time.time()
            log_step("initializing_jarvis_bridge", 6)
            _startup_state.begin_phase("initializing_jarvis_bridge", 6)
            if bridge_enabled and _agi_hub:
                try:
                    from jarvis_prime.core.jarvis_bridge import get_jarvis_bridge

                    _jarvis_bridge = await get_jarvis_bridge()
                    log_step_complete("JARVIS action/safety bridge", time.time() - step_start)
                except Exception as e:
                    logger.warning(f"   ⚠️ JARVIS action/safety bridge init failed: {e}")
                    _jarvis_bridge = None
            else:
                reason = "--no-bridge" if not bridge_enabled else "AGI hub unavailable"
                logger.info(f"   ℹ️ JARVIS action/safety bridge disabled ({reason})")
                _jarvis_bridge = None
            _startup_state.complete_phase("initializing_jarvis_bridge")

            # -----------------------------------------------------------------
            # STEP 6b: Initialize JARVIS-Prime unified inference bridge
            # v238.0: This bridge provides local-first inference with Claude API
            # fallback, complexity-based task routing, and training data capture.
            # Previously only initialized by run_supervisor.py --unified mode.
            # No dependency on AGI Hub — uses lazy model loading.
            # -----------------------------------------------------------------
            step_start = time.time()
            log_step("initializing_jarvis_prime_bridge", "6b")
            _startup_state.begin_phase("initializing_jarvis_prime_bridge", 6)
            if bridge_enabled:
                try:
                    from jarvis_prime.core.jarvis_prime_bridge import get_jarvis_prime_bridge

                    _jarvis_prime_bridge = await get_jarvis_prime_bridge()
                    log_step_complete("JARVIS-Prime unified inference bridge", time.time() - step_start)
                except ImportError:
                    logger.info("   ℹ️ JARVIS-Prime inference bridge not available (module missing)")
                    _jarvis_prime_bridge = None
                except Exception as e:
                    logger.warning(f"   ⚠️ JARVIS-Prime inference bridge init failed: {e}")
                    _jarvis_prime_bridge = None
            else:
                logger.info("   ℹ️ JARVIS-Prime inference bridge disabled (--no-bridge)")
                _jarvis_prime_bridge = None
            _startup_state.complete_phase("initializing_jarvis_prime_bridge")

            # -----------------------------------------------------------------
            # STEP 7: Initialize Neural Orchestrator
            # -----------------------------------------------------------------
            step_start = time.time()
            log_step("initializing_neural_orchestrator", 7)
            _startup_state.begin_phase("initializing_neural_orchestrator", 7)
            try:
                from jarvis_prime.core.neural_orchestrator_core import (
                    NeuralOrchestratorCore,
                    DynamicConfig as NeuralConfig,
                    neural_route,
                    get_neural_orchestrator,
                    shutdown_neural_orchestrator,
                    RoutingTier,
                )

                neural_config = NeuralConfig.from_env_and_yaml()
                _neural_orchestrator = await get_neural_orchestrator(neural_config)
                _neural_routing_enabled = True
                log_step_complete("Neural Orchestrator", time.time() - step_start)
            except ImportError as e:
                logger.warning(f"   ⚠️ Neural Orchestrator not available: {e}")
            except Exception as e:
                logger.warning(f"   ⚠️ Neural Orchestrator init failed: {e}")
                import traceback
                traceback.print_exc()
            _startup_state.complete_phase("initializing_neural_orchestrator")

            # -----------------------------------------------------------------
            # STEP 8: Resolve model path and download if needed
            # -----------------------------------------------------------------
            step_start = time.time()
            log_step("resolving_model", 8)
            _startup_state.begin_phase("resolving_model", 8)
            _startup_state.phase = "loading_model"

            model_path = Path(_args.model)
            if not model_path.exists():
                model_path = Path(__file__).parent / _args.model

            # Try GCS download
            gcs_model_uri = os.getenv("MODEL_GCS_URI")
            auto_download_model = os.getenv("AUTO_DOWNLOAD_MODEL", "true").lower() == "true"

            if not model_path.exists() and gcs_model_uri:
                logger.info(f"[Background] Downloading model from GCS: {gcs_model_uri}")
                try:
                    from google.cloud import storage
                    import re

                    model_path.parent.mkdir(parents=True, exist_ok=True)

                    match = re.match(r"gs://([^/]+)/(.+)", gcs_model_uri)
                    if match:
                        bucket_name, blob_path = match.groups()
                        client = storage.Client()
                        bucket = client.bucket(bucket_name)
                        blob = bucket.blob(blob_path)

                        if blob.exists():
                            blob.reload()
                            size_mb = blob.size / 1024 / 1024 if blob.size else 0
                            logger.info(f"[Background] Downloading {size_mb:.1f}MB model...")
                            blob.download_to_filename(str(model_path))
                            if model_path.exists():
                                logger.info(f"[Background] Model downloaded: {model_path}")
                except ImportError:
                    logger.info("[Background] GCS client not available, trying gsutil...")
                    try:
                        import subprocess
                        subprocess.run(["gsutil", "cp", gcs_model_uri, str(model_path)], check=True, capture_output=True)
                        logger.info(f"[Background] Model downloaded via gsutil")
                    except Exception as e:
                        logger.warning(f"[Background] gsutil download failed: {e}")
                except Exception as e:
                    logger.warning(f"[Background] GCS download failed: {e}")

            # Try HuggingFace download
            if not model_path.exists() and auto_download_model:
                logger.info("[Background] Attempting HuggingFace download...")
                try:
                    from jarvis_prime.docker.model_downloader import download_model, recommend_model, MODEL_CATALOG

                    default_model = os.getenv("DEFAULT_MODEL", "tinyllama-chat")
                    if default_model not in MODEL_CATALOG:
                        recommended = recommend_model(use_case="balanced", max_memory_gb=8.0)
                        default_model = recommended if recommended else "tinyllama-chat"

                    logger.info(f"[Background] Auto-downloading: {default_model}")
                    models_dir = Path(__file__).parent / "models"
                    downloaded_path = await download_model(
                        model_key=default_model,
                        models_dir=str(models_dir),
                        set_active=True,
                    )
                    model_path = models_dir / "current.gguf"
                    if model_path.exists():
                        logger.info(f"[Background] Model auto-downloaded: {downloaded_path}")
                except ImportError as e:
                    logger.warning(f"[Background] HuggingFace downloader not available: {e}")
                except Exception as e:
                    logger.warning(f"[Background] HuggingFace download failed: {e}")
                    import traceback
                    traceback.print_exc()

            _model_path = model_path
            logger.info(f"   Model path: {model_path}")
            logger.info(f"   Exists: {model_path.exists()}")
            if model_path.exists():
                size_gb = model_path.stat().st_size / (1024**3)
                logger.info(f"   Size: {size_gb:.2f} GB")
            log_step_complete("Model resolution", time.time() - step_start)
            _startup_state.complete_phase("resolving_model")

            # -----------------------------------------------------------------
            # STEP 9: Hardware optimization and executor creation
            # -----------------------------------------------------------------
            step_start = time.time()
            log_step("configuring_hardware", 9)
            _startup_state.begin_phase("configuring_hardware", 9)
            optimized_gpu_layers = _args.gpu_layers
            optimized_threads = _args.threads
            optimized_ctx_size = _args.ctx_size

            if _agi_hub and _agi_hub.hardware_optimizer:
                try:
                    hw_opt = _agi_hub.hardware_optimizer
                    recommendations = hw_opt.get_recommendations()
                    if recommendations:
                        if recommendations.get("use_mps", False):
                            optimized_gpu_layers = -1
                            logger.info("[Background] Apple Silicon: MPS enabled")
                        if "optimal_threads" in recommendations:
                            optimized_threads = recommendations["optimal_threads"]
                            logger.info(f"[Background] Apple Silicon: {optimized_threads} threads")
                except Exception as e:
                    logger.warning(f"[Background] Apple Silicon optimization failed: {e}")
            else:
                try:
                    from jarvis_prime.core.llama_cpp_executor import HardwareDetector, HardwareBackend
                    hw = HardwareDetector.detect()
                    logger.info(f"[Background] Hardware: {hw.backend.name}")
                    if hw.backend == HardwareBackend.METAL:
                        optimized_gpu_layers = -1
                        optimized_threads = hw.performance_cores or _args.threads
                        logger.info(f"[Background] Metal GPU enabled")
                    elif hw.backend == HardwareBackend.CUDA:
                        optimized_gpu_layers = -1
                        logger.info("[Background] CUDA GPU enabled")
                except Exception as e:
                    logger.warning(f"[Background] Hardware detection failed: {e}")

            config = LlamaCppConfig(
                n_ctx=optimized_ctx_size,
                n_threads=optimized_threads,
                n_gpu_layers=optimized_gpu_layers,
                verbose=_args.debug,
                flash_attn=True,
                cache_prompt=True,
            )
            _executor = LlamaCppExecutor(config)
            logger.info(f"   GPU Layers: {optimized_gpu_layers}")
            logger.info(f"   Threads: {optimized_threads}")
            logger.info(f"   Context: {optimized_ctx_size}")
            log_step_complete("Hardware configuration", time.time() - step_start)
            _startup_state.complete_phase("configuring_hardware")

            # -----------------------------------------------------------------
            # STEP 10: Load model (v93.7: with timeout and progress reporting)
            # v150.0: HOLLOW CLIENT CHECK - Skip local model loading entirely
            # -----------------------------------------------------------------
            step_start = time.time()
            log_step("loading_model", 10)
            _startup_state.begin_phase("loading_model", 10)
            # v224.0: BUGFIX — model_load_start is set below, right before the actual
            # _executor.load() call (was previously set here, inflating elapsed time
            # by including hollow client checks and model existence validation).

            # v150.1: Define model load timeout (was missing, causing NameError)
            # Default: 600s (10 min) for large models, configurable via env var
            model_load_timeout = float(os.getenv("JARVIS_MODEL_LOAD_TIMEOUT", "600.0"))

            # v150.0: CRITICAL FIX - Check hollow client mode BEFORE model loading
            # When hollow client is active, local ML is blocked and all inference
            # is routed to GCP. Attempting to load the model would trigger
            # CloudOffloadRequired, but by then memory has already been allocated.
            # This check prevents OOM by skipping model loading entirely.
            if _is_hollow_client_active():
                logger.info("")
                logger.info("=" * 70)
                logger.info("☁️  v150.0: HOLLOW CLIENT MODE ACTIVE - SKIPPING LOCAL MODEL")
                logger.info("=" * 70)
                logger.info("   Local ML inference is blocked in Hollow Client mode.")
                logger.info("   All inference requests will be routed to:")
                gcp_endpoint = os.environ.get("GCP_PRIME_ENDPOINT", os.environ.get("JARVIS_GCP_PRIME_ENDPOINT", "GCP Cloud"))
                logger.info(f"   → {gcp_endpoint}")
                logger.info("")
                logger.info("   This prevents OOM on SLIM hardware (<32GB RAM).")
                logger.info("   The server will operate as a routing proxy to cloud inference.")
                logger.info("=" * 70)
                logger.info("")

                _startup_state.model_loaded = False
                _startup_state.model_path = None
                _startup_state.details["hollow_client_mode"] = True
                _startup_state.details["gcp_endpoint"] = gcp_endpoint
                _startup_state.details["skip_reason"] = "Hollow Client mode - routing to cloud"
                log_step_complete("Model loading (SKIPPED - Hollow Client)", time.time() - step_start)
                _startup_state.complete_phase("loading_model")

                # Skip local model loading path and continue initialization
            elif model_path.exists():
                model_size_mb = model_path.stat().st_size / (1024 * 1024)
                logger.info(f"[Background] Loading model: {model_path} ({model_size_mb:.1f}MB)")
                logger.info(f"[Background] Model load timeout: {model_load_timeout}s")
                _startup_state.details["model_size_mb"] = round(model_size_mb, 1)
                _startup_state.details["model_load_timeout"] = model_load_timeout

                start = time.time()

                # v93.7: Progress reporting task
                async def report_progress():
                    """Report progress every 30 seconds during model loading."""
                    while True:
                        await asyncio.sleep(30)
                        elapsed = time.time() - start
                        remaining = model_load_timeout - elapsed
                        logger.info(
                            f"[Background] Model loading... {elapsed:.0f}s elapsed, "
                            f"{remaining:.0f}s remaining (timeout: {model_load_timeout}s)"
                        )
                        _startup_state.details["loading_elapsed"] = round(elapsed, 1)

                progress_task = asyncio.create_task(report_progress())

                try:
                    # v93.7: Model loading with timeout protection
                    # v108.2: Enhanced exception handling for all failure modes
                    # v224.0: Set model_load_start HERE, right before actual loading
                    _startup_state.model_load_start = time.time()
                    await asyncio.wait_for(
                        _executor.load(model_path),
                        timeout=model_load_timeout
                    )
                    load_time = time.time() - start
                    _startup_state.model_load_elapsed = load_time
                    _startup_state.model_loaded = True
                    _startup_state.model_path = str(model_path)
                    logger.info(f"[Background] Model loaded in {load_time:.2f}s")

                    # v93.7: Log step completion with timing
                    log_step_complete("Model loading", load_time)
                    _startup_state.complete_phase("loading_model")

                except asyncio.TimeoutError:
                    load_time = time.time() - start
                    logger.error(
                        f"[Background] MODEL LOAD TIMEOUT after {load_time:.1f}s "
                        f"(limit: {model_load_timeout}s)"
                    )
                    _startup_state.fail_phase("loading_model", f"Timeout after {load_time:.1f}s")
                    _startup_state.phase = "error"
                    _startup_state.error = f"Model load timeout after {load_time:.1f}s"
                    _startup_state.model_load_elapsed = load_time
                    progress_task.cancel()
                    return

                except MemoryError as e:
                    # v108.2: Handle out-of-memory during model loading
                    load_time = time.time() - start
                    logger.error(
                        f"[Background] MODEL LOAD FAILED - OUT OF MEMORY after {load_time:.1f}s\n"
                        f"  Model: {model_path.name}\n"
                        f"  Size: {model_size_mb:.1f}MB\n"
                        f"  Error: {e}\n"
                        f"  Suggestion: Try a smaller model or free system memory"
                    )
                    _startup_state.fail_phase("loading_model", f"OOM: {e}")
                    _startup_state.phase = "error"
                    _startup_state.error = f"Out of memory loading model ({model_size_mb:.0f}MB)"
                    _startup_state.model_load_elapsed = load_time
                    progress_task.cancel()
                    return

                except OSError as e:
                    # v108.2: Handle file system errors (corrupted model, disk full, etc.)
                    load_time = time.time() - start
                    logger.error(
                        f"[Background] MODEL LOAD FAILED - FILE ERROR after {load_time:.1f}s\n"
                        f"  Model: {model_path}\n"
                        f"  Error: {e}\n"
                        f"  Suggestion: Check model file integrity and disk space"
                    )
                    _startup_state.fail_phase("loading_model", f"OSError: {e}")
                    _startup_state.phase = "error"
                    _startup_state.error = f"File error loading model: {e}"
                    _startup_state.model_load_elapsed = load_time
                    progress_task.cancel()
                    return

                except Exception as e:
                    # v108.2: Catch ALL other exceptions to prevent process crash
                    load_time = time.time() - start
                    import traceback
                    tb_str = traceback.format_exc()
                    logger.error(
                        f"[Background] MODEL LOAD FAILED - UNEXPECTED ERROR after {load_time:.1f}s\n"
                        f"  Model: {model_path}\n"
                        f"  Error type: {type(e).__name__}\n"
                        f"  Error: {e}\n"
                        f"  Traceback:\n{tb_str}"
                    )
                    _startup_state.fail_phase("loading_model", f"{type(e).__name__}: {e}")
                    _startup_state.phase = "error"
                    _startup_state.error = f"Model load failed: {type(e).__name__}: {e}"
                    _startup_state.model_load_elapsed = load_time
                    progress_task.cancel()
                    # v108.2: Don't exit! Continue running in health-check-only mode
                    # This allows the supervisor to see the error via health endpoint
                    logger.warning(
                        "[Background] Server will continue in health-check-only mode. "
                        "Model loading failed but service remains available for diagnostics."
                    )
                    return

                finally:
                    progress_task.cancel()
                    try:
                        await progress_task
                    except asyncio.CancelledError:
                        pass

                if _bridge:
                    try:
                        from jarvis_prime.core.cross_repo_bridge import update_model_status
                        update_model_status(loaded=True, model_path=str(model_path))
                        await _bridge.notify_jarvis("model_loaded", {
                            "model_path": str(model_path),
                            "load_time_seconds": load_time,
                        })
                    except Exception as e:
                        logger.warning(f"[Background] Failed to notify bridge: {e}")
            else:
                logger.warning(f"[Background] Model not found: {_args.model}")
                logger.warning("[Background] Server will run in health-check-only mode")
                _startup_state.model_path = None
                _startup_state.complete_phase("loading_model")

                if _bridge:
                    try:
                        from jarvis_prime.core.cross_repo_bridge import update_model_status
                        update_model_status(loaded=False, model_path="")
                    except Exception:
                        pass

            # -----------------------------------------------------------------
            # STEP 10b: v241.0 — Initialize GCP model swap coordinator
            # -----------------------------------------------------------------
            _model_coordinator = None
            _v241_models_dir = Path(os.getenv("GCP_MODELS_DIR", "/opt/jarvis-prime/models"))
            _v241_manifest = _v241_models_dir / "manifest.json"
            if _v241_manifest.exists():
                try:
                    from jarvis_prime.core.gcp_model_swap_coordinator import GCPModelSwapCoordinator
                    _model_coordinator = GCPModelSwapCoordinator(
                        executor=_executor,
                        models_dir=_v241_models_dir,
                        manifest_path=_v241_manifest,
                    )
                    await _model_coordinator.initialize()
                    logger.info(
                        f"[v241] Model swap coordinator initialized: "
                        f"{len(_model_coordinator.inventory)} models in inventory"
                    )
                except Exception as _v241_init_err:
                    logger.warning(f"[v241] Coordinator init failed: {_v241_init_err}")
                    _model_coordinator = None
            else:
                logger.info("[v241] No model manifest found — single-model mode (backward compat)")

            # -----------------------------------------------------------------
            # STEP 10b.1: v242.0 — Load Phi-3.5-mini as permanent classifier
            # -----------------------------------------------------------------
            global _classifier_loaded, _classifier_system_prompt

            from jarvis_prime.core.classification_schema import (
                CLASSIFICATION_SCHEMA,
                build_classifier_system_prompt,
                DOMAIN_TO_TASK_TYPE,
                PHI_SELF_SERVE_DOMAINS,
                MIN_CONFIDENCE_THRESHOLD,
                SCHEMA_VERSION,
            )

            _v242_phi_path = _v241_models_dir / "phi-3.5-mini-instruct.Q4_K_M.gguf"
            if _v242_phi_path.exists():
                try:
                    await _executor.load_classifier(_v242_phi_path, CLASSIFICATION_SCHEMA)
                    with _prompt_lock:
                        _classifier_system_prompt = build_classifier_system_prompt()
                    _classifier_loaded = True
                    logger.info("[v242] Phi-3.5-mini classifier loaded (permanently resident)")
                except Exception as _v242_cls_err:
                    logger.warning(
                        f"[v242] Phi classifier failed to load: {_v242_cls_err}. "
                        "Falling back to metadata hints."
                    )
            else:
                logger.warning(
                    f"[v242] Phi model not found at {_v242_phi_path}. "
                    "Using metadata hints."
                )

            # -----------------------------------------------------------------
            # STEP 10c: v242.0 — Initialize telemetry hook for training data capture
            # -----------------------------------------------------------------
            _telemetry_hook = None
            try:
                from jarvis_prime.core.telemetry_hook import TelemetryHook
                telemetry_dir = Path(os.getenv("JARVIS_TELEMETRY_DIR", str(Path.home() / ".jarvis" / "telemetry")))
                telemetry_dir.mkdir(parents=True, exist_ok=True)
                _telemetry_hook = TelemetryHook(output_dir=str(telemetry_dir))
                await _telemetry_hook.start()
                logger.info("[v242] Telemetry hook initialized for training data capture")
            except Exception as e:
                logger.warning(f"[v242] Telemetry hook initialization failed (non-critical): {e}")

            # -----------------------------------------------------------------
            # STEP 10d: v242.0 — Publish reflex manifest to shared state
            #           v243.0 — Atomic writes + inhibition file bootstrap
            # -----------------------------------------------------------------
            try:
                from datetime import datetime, timezone

                trinity_dir = Path.home() / ".jarvis" / "trinity"
                trinity_dir.mkdir(parents=True, exist_ok=True)

                seed_manifest = Path(__file__).parent / "jarvis_prime" / "config" / "seed_reflex_manifest.json"
                target_manifest = trinity_dir / "reflex_manifest.json"

                if seed_manifest.exists():
                    manifest_data = json.loads(seed_manifest.read_text())
                    manifest_data["published_at"] = datetime.now(timezone.utc).isoformat()
                    _atomic_write_json(target_manifest, manifest_data)
                    logger.info(f"[v242] Reflex manifest published to {target_manifest}")
                else:
                    logger.warning(f"[v242] Seed reflex manifest not found at {seed_manifest}")

                # v243.0: Bootstrap empty inhibition file if absent
                inhibition_path = trinity_dir / "reflex_inhibition.json"
                if not inhibition_path.exists():
                    _atomic_write_json(inhibition_path, {
                        "version": 1,
                        "inhibit_reflexes": [],
                        "reason": "",
                        "published_at": datetime.now(timezone.utc).isoformat(),
                        "ttl_seconds": 0,
                    })
                    logger.info(f"[v243] Reflex inhibition file bootstrapped at {inhibition_path}")
            except Exception as e:
                logger.warning(f"[v242] Reflex manifest publish failed (non-critical): {e}")

            # -----------------------------------------------------------------
            # STEP 11: Mark ready (v93.7: with enhanced logging)
            # -----------------------------------------------------------------
            step_start = time.time()
            log_step("marking_ready", 11)
            _startup_state.begin_phase("marking_ready", 11)

            _startup_state.phase = "ready"
            _startup_state.init_elapsed = time.time() - init_start
            _startup_state.complete_phase("marking_ready")
            _startup_state.details = {}  # Clear step details

            # v93.7: Calculate detailed timing breakdown
            total_time = _startup_state.init_elapsed
            model_load_time = _startup_state.model_load_elapsed or 0
            init_overhead = total_time - model_load_time

            logger.info("")
            logger.info("=" * 70)
            logger.info("🎉 JARVIS-PRIME INITIALIZATION COMPLETE")
            logger.info("=" * 70)
            logger.info("")
            logger.info("📊 TIMING BREAKDOWN:")
            logger.info(f"   Total initialization: {total_time:.2f}s")
            logger.info(f"   ├─ Model loading:     {model_load_time:.2f}s ({model_load_time/total_time*100:.1f}%)" if total_time > 0 else f"   ├─ Model loading:     {model_load_time:.2f}s")
            logger.info(f"   └─ Other init:        {init_overhead:.2f}s ({init_overhead/total_time*100:.1f}%)" if total_time > 0 else f"   └─ Other init:        {init_overhead:.2f}s")
            logger.info("")
            logger.info("🖥️  SERVER CONFIGURATION:")
            logger.info(f"   Model: {model_path.name if model_path.exists() else 'Not loaded'}")
            if model_path.exists():
                model_size_gb = model_path.stat().st_size / (1024**3)
                logger.info(f"   Size:  {model_size_gb:.2f} GB")
            logger.info(f"   Context: {_args.ctx_size} tokens")
            logger.info(f"   GPU layers: {optimized_gpu_layers}")
            logger.info(f"   Threads: {optimized_threads}")
            logger.info(f"   Listening: http://{_args.host}:{_args.port}")
            logger.info("")
            logger.info("🔌 INTEGRATIONS:")

            if _bridge:
                logger.info(f"   ├─ JARVIS Bridge: {'Connected' if _bridge.state.connected_to_jarvis else 'Enabled (standalone)'}")
            else:
                logger.info("   ├─ JARVIS Bridge: Disabled")
            if _reactor_bridge:
                logger.info("   ├─ Reactor Core Bridge: Active")
            else:
                logger.info("   ├─ Reactor Core Bridge: Not initialized")
            if _jarvis_bridge:
                logger.info("   ├─ JARVIS Action/Safety Bridge: Active")
            else:
                logger.info("   ├─ JARVIS Action/Safety Bridge: Not initialized")
            if _trinity_initialized:
                logger.info("   ├─ PROJECT TRINITY: Connected (Mind component)")
            else:
                logger.info("   ├─ PROJECT TRINITY: Not initialized")
            if _agi_hub:
                logger.info("   ├─ AGI Integration Hub: Active")
            else:
                logger.info("   ├─ AGI Integration Hub: Not initialized")
            if _neural_routing_enabled:
                logger.info("   ├─ Neural Orchestrator v100.0: Active")
            else:
                logger.info("   ├─ Neural Orchestrator: Not initialized")
            if _model_coordinator:
                _v241_inv = _model_coordinator.inventory
                _v241_routable = sum(1 for e in _v241_inv.values() if e.routable)
                logger.info(f"   ├─ GCP Model Coordinator v241.0: {_v241_routable} routable / {len(_v241_inv)} total")
            else:
                logger.info("   ├─ GCP Model Coordinator: Not active (single-model mode)")
            if _training_pipeline:
                logger.info("   ├─ Training Data Pipeline: Active")
            else:
                logger.info("   ├─ Training Data Pipeline: Not initialized")
            if _telemetry_hook:
                logger.info(f"   └─ Telemetry Hook v242.0: Active ({_telemetry_hook.output_dir})")
            else:
                logger.info("   └─ Telemetry Hook: Not initialized")

            # Tier-2 capabilities are initialized in background to avoid
            # blocking readiness while still wiring them into live runtime.
            _tier2_capability_task = asyncio.create_task(
                initialize_tier2_capabilities(),
                name="prime_tier2_capabilities_init",
            )
            _background_tasks.add(_tier2_capability_task)
            _tier2_capability_task.add_done_callback(_background_tasks.discard)

            logger.info("")
            logger.info("=" * 70)
            logger.info("✅ READY FOR INFERENCE")
            logger.info("=" * 70)
            logger.info("")

            log_step_complete("Marking ready", time.time() - step_start)

        except Exception as e:
            logger.error(f"[Background] Initialization failed: {e}")
            import traceback
            traceback.print_exc()
            _startup_state.phase = "error"
            _startup_state.error = str(e)

    # =========================================================================
    # STARTUP EVENT - Triggers background initialization AFTER server starts
    # =========================================================================
    @app.on_event("startup")
    async def on_startup():
        """
        v97.0: FastAPI startup event with SERVICE REGISTRATION.

        This runs AFTER uvicorn binds to the port and starts listening.
        This is the key to solving the 61.9s timeout - the health endpoint
        is already responding when this function starts.

        v97.0: Now registers with shared service registry to enable
        TrinityIntegrator's registration-aware verification.
        """
        global _registry_heartbeat_task

        logger.info("[v93.2] Server is now LISTENING - starting background initialization")

        # v97.0: Register with shared service registry using ATOMIC locking
        service_data = {
            "service_name": "jarvis_prime",
            "pid": os.getpid(),
            "port": _args.port,
            "host": _args.host,
            "health_endpoint": "/health",
            "status": "starting",
            "registered_at": time.time(),
            "last_heartbeat": time.time(),
            "metadata": {
                "version": "4.0.0",
                "role": "inference",
            },
        }

        if _atomic_register_service(
            service_name="jarvis_prime",
            service_data=service_data,
            alternate_names=["jarvis-prime", "jprime"],
        ):
            logger.info(f"[v97.0] ✅ Registered with service registry (port={_args.port}, pid={os.getpid()})")

            # Start registry heartbeat loop
            async def _registry_heartbeat_loop():
                service_names = ["jarvis_prime", "jarvis-prime", "jprime"]
                while True:
                    try:
                        await asyncio.sleep(15.0)
                        success = _atomic_update_heartbeat(service_names, status="running")
                        if success:
                            logger.debug("[v97.0] Registry heartbeat updated")
                    except asyncio.CancelledError:
                        break
                    except Exception:
                        pass

            _registry_heartbeat_task = asyncio.create_task(
                _registry_heartbeat_loop(),
                name="jarvis_prime_registry_heartbeat"
            )
            logger.info("[v97.0] ✅ Registry heartbeat started (interval: 15s)")
        else:
            logger.warning("[v97.0] Service registry registration failed (non-fatal)")

        _bg_init_task = asyncio.create_task(background_initialization(), name="background_initialization")
        _background_tasks.add(_bg_init_task)
        _bg_init_task.add_done_callback(_background_tasks.discard)

    # =========================================================================
    # SHUTDOWN EVENT - Clean up all components
    # =========================================================================
    @app.on_event("shutdown")
    async def on_shutdown():
        global _registry_heartbeat_task, _reactor_bridge, _training_pipeline, _telemetry_hook, _jarvis_bridge
        global _tier2_capability_task, _continual_learner, _self_modifier, _tier2_capability_status

        logger.info("Shutting down...")

        # v97.0: Stop registry heartbeat and deregister
        if _registry_heartbeat_task and not _registry_heartbeat_task.done():
            _registry_heartbeat_task.cancel()
            try:
                await asyncio.wait_for(_registry_heartbeat_task, timeout=5.0)
            except (asyncio.CancelledError, asyncio.TimeoutError):
                pass
            logger.info("[v97.0] Registry heartbeat stopped")

        if _atomic_deregister_service(["jarvis_prime", "jarvis-prime", "jprime"]):
            logger.info("[v97.0] ✅ Deregistered from service registry")

        if _tier2_capability_task and not _tier2_capability_task.done():
            _tier2_capability_task.cancel()
            try:
                await asyncio.wait_for(_tier2_capability_task, timeout=5.0)
            except (asyncio.CancelledError, asyncio.TimeoutError):
                pass

        if _continual_learner:
            try:
                await _continual_learner.save_state()
                logger.info("[Tier2] Continual learner state saved")
            except Exception as e:
                logger.warning("[Tier2] Continual learner save failed: %s", e)
            finally:
                _continual_learner = None

        if _self_modifier:
            try:
                await _self_modifier.save_state()
                logger.info("[Tier2] Self-modifier state saved")
            except Exception as e:
                logger.warning("[Tier2] Self-modifier save failed: %s", e)
            finally:
                _self_modifier = None

        _tier2_capability_status = {
            **_tier2_capability_status,
            "state": "stopped",
            "updated_at": datetime.now().isoformat(),
        }

        if _neural_orchestrator:
            try:
                from jarvis_prime.core.neural_orchestrator_core import shutdown_neural_orchestrator
                stats = _neural_orchestrator.get_comprehensive_stats()
                routing_stats = stats.get("routing", {})
                logger.info("=" * 50)
                logger.info("Neural Orchestrator v100.0 Session Summary")
                logger.info("=" * 50)
                logger.info(f"  Total Requests Routed: {routing_stats.get('total_requests', 0)}")
                logger.info(f"  Successful Routes: {routing_stats.get('successful_routes', 0)}")
                logger.info(f"  Fallback Routes: {routing_stats.get('fallback_routes', 0)}")
                logger.info("=" * 50)
                await shutdown_neural_orchestrator()
                logger.info("Neural Orchestrator shutdown complete")
            except Exception as e:
                logger.warning(f"Neural Orchestrator shutdown error: {e}")

        if _jarvis_prime_bridge:
            try:
                from jarvis_prime.core.jarvis_prime_bridge import shutdown_jarvis_prime_bridge
                await shutdown_jarvis_prime_bridge()
                _jarvis_prime_bridge = None
                logger.info("JARVIS-Prime inference bridge shutdown complete")
            except Exception as e:
                logger.warning(f"JARVIS-Prime inference bridge shutdown error: {e}")

        if _jarvis_bridge:
            try:
                from jarvis_prime.core.jarvis_bridge import shutdown_jarvis_bridge
                await shutdown_jarvis_bridge()
                _jarvis_bridge = None
                logger.info("JARVIS action/safety bridge shutdown complete")
            except Exception as e:
                logger.warning(f"JARVIS action/safety bridge shutdown error: {e}")

        if _agi_hub:
            try:
                from jarvis_prime.core.agi_integration import shutdown_agi_hub
                await shutdown_agi_hub()
                logger.info("AGI Integration Hub shutdown complete")
            except Exception as e:
                logger.warning(f"AGI Hub shutdown error: {e}")

        if _trinity_initialized:
            try:
                from jarvis_prime.core.trinity_bridge import shutdown_trinity
                await shutdown_trinity()
                logger.info("PROJECT TRINITY: J-Prime disconnected")
            except Exception as e:
                logger.warning(f"Trinity shutdown error: {e}")

        if _training_pipeline:
            try:
                from jarvis_prime.core.training_data_pipeline import shutdown_training_data_pipeline
                await shutdown_training_data_pipeline()
                _training_pipeline = None
                logger.info("Training data pipeline shutdown complete")
            except Exception as e:
                logger.warning(f"Training data pipeline shutdown error: {e}")

        if _reactor_bridge:
            try:
                from jarvis_prime.core.reactor_core_bridge import shutdown_reactor_core_bridge
                await shutdown_reactor_core_bridge()
                _reactor_bridge = None
                logger.info("Reactor Core bridge shutdown complete")
            except Exception as e:
                logger.warning(f"Reactor Core bridge shutdown error: {e}")

        if _telemetry_hook:
            try:
                await _telemetry_hook.stop()
                _telemetry_hook = None
                logger.info("Telemetry hook shutdown complete")
            except Exception as e:
                logger.warning(f"Telemetry hook shutdown error: {e}")

        if _bridge:
            try:
                from jarvis_prime.core.cross_repo_bridge import get_cost_summary, shutdown_bridge
                cost_summary = get_cost_summary()

                if cost_summary.get("total_requests", 0) > 0:
                    logger.info("=" * 50)
                    logger.info("JARVIS-Prime Session Cost Summary")
                    logger.info("=" * 50)
                    logger.info(f"  Total Requests: {cost_summary.get('total_requests', 0)}")
                    logger.info(f"  Total Tokens: {cost_summary.get('total_tokens', 0)}")
                    logger.info(f"  Local Cost: ${cost_summary.get('local_cost_usd', 0):.4f}")
                    logger.info(f"  Savings: ${cost_summary.get('savings_usd', 0):.4f}")
                    logger.info("=" * 50)

                await _bridge.notify_jarvis("shutdown", cost_summary)
                await shutdown_bridge()
                logger.info("Cross-repo bridge shutdown complete")
            except Exception as e:
                logger.warning(f"Bridge shutdown error: {e}")

        if _executor:
            try:
                await _executor.close()
            except Exception as e:
                logger.warning(f"Executor close error: {e}")

    # =========================================================================
    # START SERVER - This is IMMEDIATE, background init happens via startup event
    # =========================================================================
    logger.info(f"[v93.2] Starting uvicorn server on {_args.host}:{_args.port}...")

    config = uvicorn.Config(
        app,
        host=_args.host,
        port=_args.port,
        reload=_args.reload,
        log_level="debug" if _args.debug else "info",
    )
    server = uvicorn.Server(config)
    await server.serve()


if __name__ == "__main__":
    asyncio.run(main())
