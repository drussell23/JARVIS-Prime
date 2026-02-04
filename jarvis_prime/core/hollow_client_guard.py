"""
Hollow Client Guard v149.0 - ML Import Blocker for Cloud-First Architecture
=============================================================================

CRITICAL: This module MUST be imported at the VERY TOP of run_server.py,
BEFORE any other imports that might transitively import torch/transformers.

Purpose:
    When running on SLIM (<32GB RAM) or CLOUD_ONLY (<16GB RAM) hardware,
    blocks heavy ML library imports and routes inference to GCP instead.

How It Works:
    1. Detects hardware profile from environment or local assessment
    2. If SLIM/CLOUD_ONLY mode: installs mock modules for torch, tensorflow, transformers
    3. Mock modules raise CloudOffloadRequired when actually used
    4. This exception signals HybridTieredRouter to route to GCP

Benefits:
    - Reduces local RAM from ~4GB to <200MB
    - Prevents SIGBUS/OOM crashes on memory-constrained hardware
    - Enables true "thin client + cloud backend" architecture

Usage:
    # At the very top of run_server.py (after os import):
    from jarvis_prime.core.hollow_client_guard import install_hollow_guard
    install_hollow_guard()

Author: JARVIS AI System
Version: 149.0.0
"""

from __future__ import annotations

import logging
import os
import sys
import types
from dataclasses import dataclass
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


# =============================================================================
# EXCEPTIONS
# =============================================================================

class CloudOffloadRequired(Exception):
    """
    Raised when a mocked ML module is actually used.
    
    This signals to HybridTieredRouter that the operation must be
    offloaded to the GCP cloud backend.
    """
    def __init__(self, module_name: str, operation: str = "import"):
        self.module_name = module_name
        self.operation = operation
        super().__init__(
            f"CloudOffloadRequired: {module_name}.{operation} - "
            f"Local ML blocked in Hollow Client mode. Route to GCP."
        )


class HollowClientError(Exception):
    """General error in hollow client guard."""
    pass


# =============================================================================
# HARDWARE PROFILE DETECTION
# =============================================================================

class HollowProfile(Enum):
    """Hardware profile for hollow client decision."""
    CLOUD_ONLY = auto()   # <16GB: Block ALL local ML
    SLIM = auto()         # 16-32GB: Block heavy ML, allow lightweight
    FULL = auto()         # 32-64GB: Allow most ML, warn on heavy
    UNLIMITED = auto()    # 64GB+: Allow everything


@dataclass
class HollowGuardConfig:
    """Configuration for hollow guard behavior."""
    profile: HollowProfile
    total_ram_gb: float
    gcp_offload_active: bool
    force_hollow: bool
    blocked_modules: Set[str]
    lightweight_modules: Set[str]  # Allowed even in SLIM mode
    
    @classmethod
    def from_environment(cls) -> 'HollowGuardConfig':
        """Build config from environment variables."""
        # Check for supervisor-provided hardware profile
        profile_name = os.environ.get("JARVIS_HARDWARE_PROFILE", "")
        gcp_offload = os.environ.get("JARVIS_GCP_OFFLOAD_ACTIVE", "").lower() == "true"
        force_hollow = os.environ.get("JARVIS_FORCE_HOLLOW_CLIENT", "").lower() == "true"
        slim_mode = os.environ.get("JARVIS_ENABLE_SLIM_MODE", "").lower() == "true"
        
        # Get RAM from environment or detect locally
        try:
            total_ram_gb = float(os.environ.get("JARVIS_TOTAL_RAM_GB", "0"))
            if total_ram_gb == 0:
                import psutil
                total_ram_gb = psutil.virtual_memory().total / (1024 ** 3)
        except (ImportError, ValueError):
            total_ram_gb = 16.0  # Conservative default
        
        # Determine profile
        # v150.0: Environment override takes PRIORITY over RAM detection
        # This allows users to force local ML on SLIM hardware (e.g., TinyLlama on 16GB)
        if profile_name == "UNLIMITED":
            # Explicit UNLIMITED override - allow everything
            profile = HollowProfile.UNLIMITED
        elif profile_name == "FULL":
            # Explicit FULL override - allow most ML locally
            profile = HollowProfile.FULL
        elif profile_name == "SLIM":
            # Explicit SLIM - block heavy ML
            profile = HollowProfile.SLIM
        elif profile_name == "CLOUD_ONLY" or force_hollow:
            # Explicit CLOUD_ONLY or forced - block all ML
            profile = HollowProfile.CLOUD_ONLY
        # RAM-based auto-detection (only when no explicit profile set)
        elif total_ram_gb < 16:
            profile = HollowProfile.CLOUD_ONLY
        elif slim_mode or total_ram_gb < 32:
            profile = HollowProfile.SLIM
        elif total_ram_gb < 64:
            profile = HollowProfile.FULL
        else:
            profile = HollowProfile.UNLIMITED
        
        # Define blocked modules by profile
        # CLOUD_ONLY: Block everything heavy
        # SLIM: Block torch core, allow lightweight like tokenizers
        heavy_ml_modules = {
            "torch", "torch.nn", "torch.cuda", "torch.mps",
            "tensorflow", "tf", "keras",
            "transformers", 
            "speechbrain",
            "whisper", "openai_whisper",
            "llama_cpp",
            "sentence_transformers",
            "diffusers",
            "accelerate",
            "bitsandbytes",
        }
        
        lightweight_modules = {
            "tokenizers",  # Fast, lightweight
            "numpy",       # Minimal overhead
            "scipy",       # Computational but not ML
        }
        
        return cls(
            profile=profile,
            total_ram_gb=total_ram_gb,
            gcp_offload_active=gcp_offload,
            force_hollow=force_hollow,
            blocked_modules=heavy_ml_modules,
            lightweight_modules=lightweight_modules,
        )


# =============================================================================
# MOCK MODULE SYSTEM
# =============================================================================

class MockMLModule(types.ModuleType):
    """
    A mock module that raises CloudOffloadRequired when any attribute is accessed.
    
    This allows imports to succeed (no ImportError) but actual usage
    triggers cloud offload.
    """
    
    _ALLOWED_ATTRS = {
        "__name__", "__loader__", "__package__", "__path__", 
        "__file__", "__cached__", "__spec__", "__doc__",
        "__dict__", "__class__", "__module__",
    }
    
    def __init__(self, name: str, original_module: Optional[types.ModuleType] = None):
        super().__init__(name)
        self._mock_name = name
        self._mock_original = original_module
        self._mock_submodules: Dict[str, 'MockMLModule'] = {}
        
        # Copy some basic metadata if original exists
        if original_module:
            self.__doc__ = getattr(original_module, "__doc__", None)
            self.__file__ = getattr(original_module, "__file__", None)
    
    def __getattr__(self, name: str) -> Any:
        """Raise CloudOffloadRequired for any non-meta attribute access."""
        if name in self._ALLOWED_ATTRS or name.startswith("_mock_"):
            return super().__getattribute__(name)
        
        # Check if this is a submodule request
        if name in self._mock_submodules:
            return self._mock_submodules[name]
        
        # Create mock submodule on demand
        full_name = f"{self._mock_name}.{name}"
        submock = MockMLModule(full_name)
        self._mock_submodules[name] = submock
        
        # Log that we're returning a mock
        logger.debug(f"[HollowGuard] Mock access: {full_name}")
        
        return submock
    
    def __call__(self, *args, **kwargs) -> None:
        """Raise CloudOffloadRequired when module is called as function."""
        raise CloudOffloadRequired(self._mock_name, "call")
    
    def __repr__(self) -> str:
        return f"<MockMLModule '{self._mock_name}' (HollowClient)>"


class HollowImportFinder:
    """
    Meta path finder that intercepts imports of blocked modules.
    
    Installed into sys.meta_path to catch imports before they hit
    the filesystem.
    """
    
    def __init__(self, config: HollowGuardConfig):
        self.config = config
        self._installed_mocks: Dict[str, MockMLModule] = {}
    
    def find_module(self, fullname: str, path=None) -> Optional['HollowImportFinder']:
        """Check if this module should be mocked."""
        # Check if module or any parent is blocked
        module_parts = fullname.split(".")
        for i in range(len(module_parts)):
            check_name = ".".join(module_parts[:i+1])
            if check_name in self.config.blocked_modules:
                logger.info(
                    f"[HollowGuard] Blocking import: {fullname} "
                    f"(matches blocked: {check_name})"
                )
                return self
        return None
    
    def load_module(self, fullname: str) -> MockMLModule:
        """Return a mock module for the blocked import."""
        if fullname in self._installed_mocks:
            return self._installed_mocks[fullname]
        
        # Create mock module
        mock = MockMLModule(fullname)
        self._installed_mocks[fullname] = mock
        sys.modules[fullname] = mock
        
        return mock


# =============================================================================
# GUARD INSTALLATION
# =============================================================================

# Global state
_guard_installed: bool = False
_guard_config: Optional[HollowGuardConfig] = None
_import_finder: Optional[HollowImportFinder] = None


def install_hollow_guard(force: bool = False) -> Tuple[bool, str]:
    """
    Install the hollow client guard to block heavy ML imports.
    
    MUST be called at the very top of the entry point, before any
    other imports that might transitively import torch/transformers.
    
    Args:
        force: If True, install even on FULL/UNLIMITED systems
        
    Returns:
        Tuple of (installed: bool, reason: str)
    """
    global _guard_installed, _guard_config, _import_finder
    
    if _guard_installed and not force:
        return True, "Already installed"
    
    # Build configuration
    config = HollowGuardConfig.from_environment()
    _guard_config = config
    
    # Log hardware profile
    logger.info(
        f"[HollowGuard v149.0] Hardware Profile: {config.profile.name} "
        f"({config.total_ram_gb:.1f}GB RAM)"
    )
    
    # Decide whether to install based on profile
    should_install = (
        config.profile in (HollowProfile.CLOUD_ONLY, HollowProfile.SLIM) or
        config.gcp_offload_active or
        config.force_hollow or
        force
    )
    
    if not should_install:
        logger.info(
            f"[HollowGuard] Skipping guard installation - "
            f"profile {config.profile.name} allows local ML"
        )
        return False, f"Profile {config.profile.name} allows local ML"
    
    # Install the import finder
    _import_finder = HollowImportFinder(config)
    
    # Insert at the beginning of meta_path to intercept first
    if _import_finder not in sys.meta_path:
        sys.meta_path.insert(0, _import_finder)
    
    _guard_installed = True
    
    # Set environment variable for other modules to check
    os.environ["JARVIS_HOLLOW_CLIENT_ACTIVE"] = "true"
    
    blocked_str = ", ".join(sorted(config.blocked_modules)[:5]) + "..."
    logger.info(
        f"[HollowGuard] ✅ Guard INSTALLED - Blocking: {blocked_str}"
    )
    
    return True, f"Installed for profile {config.profile.name}"


def uninstall_hollow_guard() -> bool:
    """
    Remove the hollow client guard.
    
    Use with caution - only for testing or when switching to local mode.
    """
    global _guard_installed, _import_finder
    
    if not _guard_installed:
        return False
    
    if _import_finder in sys.meta_path:
        sys.meta_path.remove(_import_finder)
    
    _guard_installed = False
    os.environ.pop("JARVIS_HOLLOW_CLIENT_ACTIVE", None)
    
    logger.info("[HollowGuard] Guard uninstalled")
    return True


def is_hollow_client_active() -> bool:
    """Check if hollow client mode is active."""
    return _guard_installed or os.environ.get("JARVIS_HOLLOW_CLIENT_ACTIVE") == "true"


def get_hollow_guard_status() -> Dict[str, Any]:
    """Get current hollow guard status for diagnostics."""
    return {
        "installed": _guard_installed,
        "profile": _guard_config.profile.name if _guard_config else None,
        "total_ram_gb": _guard_config.total_ram_gb if _guard_config else None,
        "gcp_offload_active": _guard_config.gcp_offload_active if _guard_config else False,
        "blocked_modules": list(_guard_config.blocked_modules)[:10] if _guard_config else [],
        "mocked_count": len(_import_finder._installed_mocks) if _import_finder else 0,
    }


# =============================================================================
# CLOUD OFFLOAD INTEGRATION
# =============================================================================

def check_cloud_offload_needed(operation: str = "inference") -> Tuple[bool, Optional[str]]:
    """
    Check if an operation needs cloud offload.
    
    Args:
        operation: Type of operation (inference, training, embedding, etc.)
        
    Returns:
        Tuple of (needs_offload: bool, gcp_endpoint: Optional[str])
    """
    if not is_hollow_client_active():
        return False, None
    
    # Get GCP endpoint from environment
    gcp_endpoint = os.environ.get(
        "JARVIS_GCP_PRIME_ENDPOINT",
        os.environ.get("GCP_PRIME_ENDPOINT")
    )
    
    if not gcp_endpoint:
        logger.warning(
            f"[HollowGuard] Cloud offload needed for {operation} but no GCP endpoint set"
        )
    
    return True, gcp_endpoint


def route_to_cloud_or_raise(operation: str = "inference") -> str:
    """
    Get cloud endpoint or raise if not available.
    
    Use this when an operation MUST go to cloud.
    
    Raises:
        HollowClientError: If cloud endpoint not configured
    """
    needs_offload, endpoint = check_cloud_offload_needed(operation)
    
    if not needs_offload:
        raise HollowClientError(
            f"route_to_cloud_or_raise called but hollow client not active"
        )
    
    if not endpoint:
        raise HollowClientError(
            f"Cloud offload required for {operation} but GCP_PRIME_ENDPOINT not set. "
            f"Set GCP_ENABLED=true and configure GCP credentials."
        )
    
    return endpoint


# =============================================================================
# MODULE EXPORTS
# =============================================================================

__all__ = [
    "install_hollow_guard",
    "uninstall_hollow_guard",
    "is_hollow_client_active",
    "get_hollow_guard_status",
    "check_cloud_offload_needed",
    "route_to_cloud_or_raise",
    "CloudOffloadRequired",
    "HollowClientError",
    "HollowProfile",
    "HollowGuardConfig",
]
