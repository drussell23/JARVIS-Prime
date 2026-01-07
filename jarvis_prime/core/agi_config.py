"""
AGI Configuration Management
=============================

v78.0 - Unified configuration for all AGI components

Provides centralized, validated, and dynamic configuration:
- Environment variable overrides
- YAML/JSON file support
- Runtime hot-reloading
- Validation with Pydantic-like checks
- Secrets management

CONFIGURATION HIERARCHY (highest to lowest priority):
    1. Environment variables (JARVIS_PRIME_*)
    2. Runtime overrides
    3. User config file (~/.jarvis/prime/config.yaml)
    4. Project config file (./config.yaml)
    5. Default values
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import secrets
from dataclasses import dataclass, field, fields
from enum import Enum
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    Generic,
    List,
    Optional,
    Set,
    Type,
    TypeVar,
    Union,
    get_type_hints,
)

logger = logging.getLogger(__name__)

T = TypeVar("T")


# =============================================================================
# CONFIGURATION SECTIONS
# =============================================================================


@dataclass
class ModelConfig:
    """Configuration for model loading."""

    # Model paths
    models_dir: Path = field(default_factory=lambda: Path.home() / ".jarvis" / "prime" / "models")
    default_model: str = "tinyllama-chat"
    auto_download: bool = True

    # Quantization
    default_quantization: str = "Q4_K_M"
    enable_quantization: bool = True

    # Memory
    context_size: int = 2048
    batch_size: int = 512
    max_tokens: int = 1024

    # Performance
    n_threads: int = 4
    n_gpu_layers: int = -1  # -1 for auto

    # Caching
    enable_kv_cache: bool = True
    cache_size_mb: int = 512


@dataclass
class ReasoningConfig:
    """Configuration for reasoning engine."""

    # Strategy defaults
    default_strategy: str = "chain_of_thought"
    enable_auto_strategy: bool = True

    # Limits
    max_reasoning_depth: int = 10
    max_tree_branches: int = 5
    max_thoughts_per_branch: int = 10

    # Timeouts
    reasoning_timeout_seconds: float = 30.0
    thought_generation_timeout: float = 5.0

    # Quality thresholds
    min_confidence_threshold: float = 0.3
    beam_width: int = 3


@dataclass
class LearningConfig:
    """Configuration for continuous learning."""

    # Experience buffer
    buffer_size: int = 10000
    min_samples_for_update: int = 100

    # EWC settings
    ewc_lambda: float = 0.4
    fisher_sample_size: int = 200

    # Synaptic Intelligence
    si_c: float = 0.1
    si_epsilon: float = 1e-7

    # Updates
    auto_update_interval_hours: float = 24.0
    enable_auto_update: bool = True

    # A/B Testing
    enable_ab_testing: bool = True
    ab_test_sample_size: int = 100


@dataclass
class HardwareConfig:
    """Configuration for hardware optimization."""

    # Platform
    prefer_gpu: bool = True
    prefer_neural_engine: bool = True  # Apple Neural Engine

    # Apple Silicon
    enable_mps: bool = True
    enable_coreml: bool = True
    enable_uma_optimization: bool = True

    # Memory
    enable_memory_mapping: bool = True
    max_memory_percent: float = 0.8  # Don't use more than 80% of available

    # Threading
    auto_detect_threads: bool = True
    performance_cores_only: bool = False


@dataclass
class MultiModalConfig:
    """Configuration for multi-modal fusion."""

    # Encoders
    enable_text_encoder: bool = True
    enable_image_encoder: bool = True
    enable_screen_encoder: bool = True
    enable_audio_encoder: bool = False
    enable_gesture_encoder: bool = False

    # Fusion
    default_fusion_strategy: str = "attention"  # early, late, attention, hierarchical
    fusion_hidden_dim: int = 512

    # Screen understanding
    screen_resize_max: int = 1024
    enable_spatial_reasoning: bool = True
    enable_temporal_reasoning: bool = True


@dataclass
class TrinityConfig:
    """Configuration for Trinity Protocol."""

    # Transport
    transport: str = "file_ipc"  # file_ipc, websocket, http2
    host: str = "localhost"
    port: int = 9100

    # IPC paths
    ipc_base_dir: Path = field(default_factory=lambda: Path.home() / ".jarvis" / "trinity")

    # Security
    enable_encryption: bool = False
    enable_signatures: bool = True
    secret_key: str = field(default_factory=lambda: secrets.token_hex(32))

    # Timing
    heartbeat_interval_seconds: float = 5.0
    connection_timeout_seconds: float = 10.0
    request_timeout_seconds: float = 30.0


@dataclass
class SafetyConfig:
    """Configuration for safety features."""

    # Risk thresholds
    auto_confirm_risk_level: str = "low"  # safe, low, medium
    require_confirmation_above: str = "medium"

    # Kill switch
    enable_kill_switch: bool = True
    kill_switch_timeout_seconds: float = 300.0  # 5 minutes

    # Trust
    default_trust_level: float = 1.0
    min_trust_level: float = 0.5

    # Logging
    log_all_actions: bool = True
    log_risky_actions: bool = True


@dataclass
class PersistenceConfig:
    """Configuration for model persistence."""

    # Storage
    storage_dir: Path = field(default_factory=lambda: Path.home() / ".jarvis" / "prime" / "storage")

    # Versioning
    max_versions_per_component: int = 10
    auto_cleanup: bool = True

    # Compression
    enable_compression: bool = True
    compression_level: int = 6

    # Checksums
    verify_checksums: bool = True


@dataclass
class MetricsConfig:
    """Configuration for metrics and monitoring."""

    # Collection
    enable_metrics: bool = True
    collection_interval_seconds: float = 10.0

    # Export
    enable_prometheus: bool = False
    prometheus_port: int = 9090

    # History
    max_history_size: int = 10000
    history_retention_hours: float = 168.0  # 1 week


@dataclass
class ServerConfig:
    """Configuration for server."""

    # Binding
    host: str = "0.0.0.0"
    port: int = 8000

    # CORS
    enable_cors: bool = True
    cors_origins: List[str] = field(default_factory=lambda: ["*"])

    # Timeouts
    request_timeout_seconds: float = 60.0
    keepalive_timeout_seconds: float = 5.0

    # Limits
    max_request_size_mb: int = 100
    max_concurrent_requests: int = 100


# =============================================================================
# MAIN CONFIGURATION
# =============================================================================


@dataclass
class AGIConfig:
    """
    Master configuration for all AGI components.

    Aggregates all sub-configurations into a single object
    that can be loaded, validated, and shared across components.
    """

    # Sub-configurations
    model: ModelConfig = field(default_factory=ModelConfig)
    reasoning: ReasoningConfig = field(default_factory=ReasoningConfig)
    learning: LearningConfig = field(default_factory=LearningConfig)
    hardware: HardwareConfig = field(default_factory=HardwareConfig)
    multimodal: MultiModalConfig = field(default_factory=MultiModalConfig)
    trinity: TrinityConfig = field(default_factory=TrinityConfig)
    safety: SafetyConfig = field(default_factory=SafetyConfig)
    persistence: PersistenceConfig = field(default_factory=PersistenceConfig)
    metrics: MetricsConfig = field(default_factory=MetricsConfig)
    server: ServerConfig = field(default_factory=ServerConfig)

    # Version
    config_version: str = "78.0"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result = {}
        for f in fields(self):
            value = getattr(self, f.name)
            if hasattr(value, "to_dict"):
                result[f.name] = value.to_dict()
            elif hasattr(value, "__dataclass_fields__"):
                result[f.name] = {
                    k: str(v) if isinstance(v, Path) else v
                    for k, v in value.__dict__.items()
                }
            else:
                result[f.name] = value
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AGIConfig":
        """Create from dictionary."""
        config = cls()

        for section_name, section_data in data.items():
            if hasattr(config, section_name) and isinstance(section_data, dict):
                section = getattr(config, section_name)
                for key, value in section_data.items():
                    if hasattr(section, key):
                        # Handle Path conversion
                        field_value = getattr(section, key)
                        if isinstance(field_value, Path):
                            value = Path(value)
                        setattr(section, key, value)

        return config


# =============================================================================
# CONFIGURATION MANAGER
# =============================================================================


class ConfigManager:
    """
    Manages AGI configuration with support for:
    - Multiple config sources (files, env vars)
    - Validation
    - Hot reloading
    - Change notifications

    Usage:
        manager = ConfigManager()
        await manager.load()

        # Access config
        ctx_size = manager.config.model.context_size

        # Override at runtime
        manager.set("model.context_size", 4096)

        # Watch for changes
        @manager.on_change("model")
        def handle_model_change(new_config):
            reload_model(new_config)
    """

    ENV_PREFIX = "JARVIS_PRIME_"

    def __init__(
        self,
        config_paths: Optional[List[Path]] = None,
        auto_reload: bool = True,
    ) -> None:
        self._config_paths = config_paths or self._default_paths()
        self._auto_reload = auto_reload
        self._config = AGIConfig()
        self._change_callbacks: Dict[str, List[Callable]] = {}
        self._watch_task: Optional[asyncio.Task] = None
        self._file_mtimes: Dict[Path, float] = {}
        self._lock = asyncio.Lock()

    def _default_paths(self) -> List[Path]:
        """Get default config file paths."""
        return [
            Path("config.yaml"),
            Path("config.json"),
            Path.home() / ".jarvis" / "prime" / "config.yaml",
            Path.home() / ".jarvis" / "prime" / "config.json",
        ]

    @property
    def config(self) -> AGIConfig:
        """Get current configuration."""
        return self._config

    async def load(self) -> AGIConfig:
        """Load configuration from all sources."""
        async with self._lock:
            # Start with defaults
            self._config = AGIConfig()

            # Load from files (in order of priority)
            for path in reversed(self._config_paths):
                if path.exists():
                    await self._load_file(path)
                    self._file_mtimes[path] = path.stat().st_mtime

            # Apply environment variables
            self._apply_env_vars()

            # Validate
            self._validate()

            # Start watch task if auto-reload enabled
            if self._auto_reload and self._watch_task is None:
                self._watch_task = asyncio.create_task(self._watch_files())

            logger.info(f"Configuration loaded (version {self._config.config_version})")
            return self._config

    async def _load_file(self, path: Path) -> None:
        """Load configuration from a file."""
        try:
            content = path.read_text()

            if path.suffix in (".yaml", ".yml"):
                try:
                    import yaml
                    data = yaml.safe_load(content)
                except ImportError:
                    logger.warning("PyYAML not installed, skipping YAML config")
                    return
            else:
                data = json.loads(content)

            if data:
                self._config = AGIConfig.from_dict(data)
                logger.debug(f"Loaded config from {path}")

        except Exception as e:
            logger.warning(f"Failed to load config from {path}: {e}")

    def _apply_env_vars(self) -> None:
        """Apply environment variable overrides."""
        for key, value in os.environ.items():
            if not key.startswith(self.ENV_PREFIX):
                continue

            # Parse key: JARVIS_PRIME_MODEL_CONTEXT_SIZE -> model.context_size
            config_key = key[len(self.ENV_PREFIX):].lower().replace("_", ".", 1)

            try:
                self.set(config_key, self._parse_env_value(value))
            except Exception as e:
                logger.warning(f"Failed to apply env var {key}: {e}")

    def _parse_env_value(self, value: str) -> Any:
        """Parse environment variable value."""
        # Try bool
        if value.lower() in ("true", "yes", "1"):
            return True
        if value.lower() in ("false", "no", "0"):
            return False

        # Try int
        try:
            return int(value)
        except ValueError:
            pass

        # Try float
        try:
            return float(value)
        except ValueError:
            pass

        # Return as string
        return value

    def _validate(self) -> None:
        """Validate configuration."""
        # Validate paths exist or can be created
        for path_attr in ["models_dir"]:
            path = getattr(self._config.model, path_attr, None)
            if path:
                path.mkdir(parents=True, exist_ok=True)

        # Validate numeric ranges
        if self._config.model.context_size < 128:
            logger.warning("context_size too small, setting to 128")
            self._config.model.context_size = 128

        if self._config.model.context_size > 131072:
            logger.warning("context_size too large, setting to 131072")
            self._config.model.context_size = 131072

        if not 0 <= self._config.safety.default_trust_level <= 1:
            logger.warning("default_trust_level out of range, setting to 1.0")
            self._config.safety.default_trust_level = 1.0

    async def _watch_files(self) -> None:
        """Watch config files for changes."""
        while True:
            try:
                await asyncio.sleep(5.0)  # Check every 5 seconds

                for path in self._config_paths:
                    if not path.exists():
                        continue

                    mtime = path.stat().st_mtime
                    if path in self._file_mtimes and mtime != self._file_mtimes[path]:
                        logger.info(f"Config file changed: {path}")
                        self._file_mtimes[path] = mtime
                        await self.reload()
                        break

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.warning(f"Config watch error: {e}")

    async def reload(self) -> AGIConfig:
        """Reload configuration from files."""
        old_config = self._config
        await self.load()

        # Notify change callbacks
        for section, callbacks in self._change_callbacks.items():
            old_section = getattr(old_config, section, None)
            new_section = getattr(self._config, section, None)

            if old_section != new_section:
                for callback in callbacks:
                    try:
                        if asyncio.iscoroutinefunction(callback):
                            await callback(new_section)
                        else:
                            callback(new_section)
                    except Exception as e:
                        logger.warning(f"Config change callback error: {e}")

        return self._config

    def set(self, key: str, value: Any) -> None:
        """Set a configuration value at runtime."""
        parts = key.split(".")

        if len(parts) == 1:
            if hasattr(self._config, parts[0]):
                setattr(self._config, parts[0], value)
        elif len(parts) == 2:
            section = getattr(self._config, parts[0], None)
            if section and hasattr(section, parts[1]):
                # Handle Path conversion
                if isinstance(getattr(section, parts[1]), Path):
                    value = Path(value)
                setattr(section, parts[1], value)

    def get(self, key: str, default: Any = None) -> Any:
        """Get a configuration value."""
        parts = key.split(".")

        try:
            if len(parts) == 1:
                return getattr(self._config, parts[0], default)
            elif len(parts) == 2:
                section = getattr(self._config, parts[0], None)
                if section:
                    return getattr(section, parts[1], default)
        except Exception:
            pass

        return default

    def on_change(self, section: str) -> Callable:
        """Decorator to register change callback for a section."""
        def decorator(func: Callable) -> Callable:
            if section not in self._change_callbacks:
                self._change_callbacks[section] = []
            self._change_callbacks[section].append(func)
            return func
        return decorator

    async def save(self, path: Optional[Path] = None) -> bool:
        """Save current configuration to file."""
        path = path or (Path.home() / ".jarvis" / "prime" / "config.json")

        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            data = self._config.to_dict()
            path.write_text(json.dumps(data, indent=2, default=str))
            logger.info(f"Configuration saved to {path}")
            return True

        except Exception as e:
            logger.error(f"Failed to save config: {e}")
            return False

    async def shutdown(self) -> None:
        """Shutdown config manager."""
        if self._watch_task:
            self._watch_task.cancel()
            try:
                await self._watch_task
            except asyncio.CancelledError:
                pass


# =============================================================================
# SINGLETON
# =============================================================================


_config_manager: Optional[ConfigManager] = None
_config_lock = asyncio.Lock()


async def get_config_manager() -> ConfigManager:
    """Get or create global config manager."""
    global _config_manager

    async with _config_lock:
        if _config_manager is None:
            _config_manager = ConfigManager()
            await _config_manager.load()

        return _config_manager


async def get_config() -> AGIConfig:
    """Get current AGI configuration."""
    manager = await get_config_manager()
    return manager.config
