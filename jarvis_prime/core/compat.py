"""
Python Compatibility Layer v1.0 - Cross-Version Support
========================================================

Provides compatibility shims for Python 3.9+ features and handles
version-specific issues gracefully.

KEY ISSUES ADDRESSED:
    - importlib.metadata.packages_distributions() (Python 3.10+)
    - asyncio.TaskGroup (Python 3.11+)
    - typing.Self (Python 3.11+)
    - ExceptionGroup (Python 3.11+)
    - tomllib (Python 3.11+)

USAGE:
    from jarvis_prime.core.compat import (
        safe_import,
        packages_distributions,
        create_task_group,
        suppress_warnings,
    )
"""

from __future__ import annotations

import asyncio
import functools
import importlib
import logging
import sys
import warnings
from contextlib import contextmanager, asynccontextmanager
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Set,
    Tuple,
    Type,
    TypeVar,
    Union,
)

logger = logging.getLogger(__name__)

# =============================================================================
# VERSION INFO
# =============================================================================

PYTHON_VERSION = sys.version_info
PYTHON_39 = PYTHON_VERSION >= (3, 9)
PYTHON_310 = PYTHON_VERSION >= (3, 10)
PYTHON_311 = PYTHON_VERSION >= (3, 11)
PYTHON_312 = PYTHON_VERSION >= (3, 12)

VERSION_STR = f"{PYTHON_VERSION.major}.{PYTHON_VERSION.minor}.{PYTHON_VERSION.micro}"

logger.debug(f"Python version: {VERSION_STR}")


# =============================================================================
# SAFE IMPORT UTILITIES
# =============================================================================

T = TypeVar('T')


def safe_import(
    module_name: str,
    attribute: Optional[str] = None,
    fallback: Any = None,
    warn: bool = True,
) -> Any:
    """
    Safely import a module or attribute with fallback.

    Args:
        module_name: Full module path (e.g., 'opentelemetry.trace')
        attribute: Optional attribute to get from module
        fallback: Value to return if import fails
        warn: Whether to log a warning on failure

    Returns:
        The imported module/attribute or fallback

    USAGE:
        # Import module with fallback
        aiohttp = safe_import('aiohttp', fallback=None)

        # Import specific attribute
        trace = safe_import('opentelemetry', 'trace', fallback=None)
    """
    try:
        module = importlib.import_module(module_name)
        if attribute:
            return getattr(module, attribute, fallback)
        return module
    except ImportError as e:
        if warn:
            logger.debug(f"Optional import failed: {module_name} - {e}")
        return fallback
    except Exception as e:
        if warn:
            logger.debug(f"Import error: {module_name} - {e}")
        return fallback


def safe_import_from(
    module_name: str,
    *attributes: str,
    fallbacks: Optional[Dict[str, Any]] = None,
) -> Tuple[Any, ...]:
    """
    Safely import multiple attributes from a module.

    Args:
        module_name: Module to import from
        *attributes: Attribute names to import
        fallbacks: Dict mapping attribute names to fallback values

    Returns:
        Tuple of imported values (or fallbacks)

    USAGE:
        trace, metrics = safe_import_from(
            'opentelemetry',
            'trace', 'metrics',
            fallbacks={'trace': None, 'metrics': None}
        )
    """
    fallbacks = fallbacks or {}
    results = []

    try:
        module = importlib.import_module(module_name)
        for attr in attributes:
            results.append(getattr(module, attr, fallbacks.get(attr)))
    except ImportError:
        for attr in attributes:
            results.append(fallbacks.get(attr))
    except Exception as e:
        logger.debug(f"Import error: {module_name} - {e}")
        for attr in attributes:
            results.append(fallbacks.get(attr))

    return tuple(results)


# =============================================================================
# IMPORTLIB.METADATA COMPATIBILITY
# =============================================================================

def packages_distributions() -> Dict[str, List[str]]:
    """
    Get mapping of package names to distribution names.

    This is a compatibility shim for importlib.metadata.packages_distributions()
    which is only available in Python 3.10+.

    Returns:
        Dict mapping package names to list of distribution names
    """
    if PYTHON_310:
        try:
            from importlib.metadata import packages_distributions as _pd
            return _pd()
        except AttributeError:
            pass

    # Fallback for Python 3.9
    try:
        from importlib.metadata import distributions
        result: Dict[str, List[str]] = {}
        for dist in distributions():
            if dist.files:
                for file in dist.files:
                    parts = str(file).split('/')
                    if parts and not parts[0].startswith('_'):
                        pkg = parts[0].replace('.py', '')
                        if pkg not in result:
                            result[pkg] = []
                        if dist.metadata['Name'] not in result[pkg]:
                            result[pkg].append(dist.metadata['Name'])
        return result
    except Exception:
        return {}


# =============================================================================
# ASYNCIO COMPATIBILITY
# =============================================================================

class TaskGroupCompat:
    """
    Compatibility class for asyncio.TaskGroup (Python 3.11+).

    Provides similar functionality for Python 3.9/3.10.
    """

    def __init__(self):
        self._tasks: List[asyncio.Task] = []
        self._errors: List[BaseException] = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self._tasks:
            results = await asyncio.gather(*self._tasks, return_exceptions=True)
            for result in results:
                if isinstance(result, Exception):
                    self._errors.append(result)

        if self._errors:
            # In Python 3.11+ this would be ExceptionGroup
            if len(self._errors) == 1:
                raise self._errors[0]
            raise RuntimeError(f"Multiple errors: {self._errors}")

        return False

    def create_task(self, coro) -> asyncio.Task:
        """Create a task within the group."""
        task = asyncio.create_task(coro)
        self._tasks.append(task)
        return task


@asynccontextmanager
async def create_task_group():
    """
    Create a task group compatible with Python 3.9+.

    USAGE:
        async with create_task_group() as tg:
            tg.create_task(async_operation_1())
            tg.create_task(async_operation_2())
    """
    if PYTHON_311:
        async with asyncio.TaskGroup() as tg:
            yield tg
    else:
        async with TaskGroupCompat() as tg:
            yield tg


async def gather_with_concurrency(
    limit: int,
    *coros,
    return_exceptions: bool = False,
) -> List[Any]:
    """
    Run coroutines with a concurrency limit.

    Args:
        limit: Maximum concurrent coroutines
        *coros: Coroutines to run
        return_exceptions: If True, return exceptions instead of raising

    Returns:
        List of results in order
    """
    semaphore = asyncio.Semaphore(limit)

    async def limited(coro):
        async with semaphore:
            return await coro

    return await asyncio.gather(
        *[limited(c) for c in coros],
        return_exceptions=return_exceptions,
    )


# =============================================================================
# EXCEPTION HANDLING
# =============================================================================

class ExceptionGroupCompat(Exception):
    """
    Compatibility class for ExceptionGroup (Python 3.11+).
    """

    def __init__(self, message: str, exceptions: List[BaseException]):
        super().__init__(message)
        self.exceptions = exceptions

    def __str__(self):
        details = '\n'.join(f"  - {type(e).__name__}: {e}" for e in self.exceptions)
        return f"{self.args[0]}:\n{details}"


def create_exception_group(
    message: str,
    exceptions: List[BaseException],
) -> BaseException:
    """
    Create an exception group compatible with Python 3.9+.
    """
    if PYTHON_311:
        return ExceptionGroup(message, exceptions)
    return ExceptionGroupCompat(message, exceptions)


# =============================================================================
# WARNING SUPPRESSION
# =============================================================================

@contextmanager
def suppress_warnings(*categories: Type[Warning]):
    """
    Context manager to suppress specific warning categories.

    USAGE:
        with suppress_warnings(DeprecationWarning, FutureWarning):
            import problematic_library
    """
    with warnings.catch_warnings():
        for category in categories:
            warnings.filterwarnings('ignore', category=category)
        yield


@contextmanager
def suppress_all_warnings():
    """Suppress all warnings."""
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        yield


# =============================================================================
# RESILIENT IMPORT CONTEXT
# =============================================================================

@contextmanager
def resilient_import(module_name: str, critical: bool = False):
    """
    Context manager for resilient imports.

    Catches and logs import errors without crashing.

    Args:
        module_name: Name of module being imported (for logging)
        critical: If True, re-raise after logging

    USAGE:
        with resilient_import('google.cloud'):
            from google.cloud import storage
    """
    try:
        with suppress_warnings(DeprecationWarning, FutureWarning):
            yield
    except ImportError as e:
        logger.warning(f"Import failed for {module_name}: {e}")
        if critical:
            raise
    except AttributeError as e:
        # Handles things like 'module has no attribute X'
        logger.warning(f"Attribute error importing {module_name}: {e}")
        if critical:
            raise
    except Exception as e:
        logger.warning(f"Unexpected error importing {module_name}: {e}")
        if critical:
            raise


# =============================================================================
# DEFENSIVE DECORATOR
# =============================================================================

def defensive(
    fallback: Any = None,
    log_errors: bool = True,
    reraise: bool = False,
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """
    Decorator that catches and handles exceptions gracefully.

    Args:
        fallback: Value to return on error
        log_errors: Whether to log errors
        reraise: Whether to re-raise after logging

    USAGE:
        @defensive(fallback=None)
        def risky_operation():
            return external_lib.do_something()
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> T:
            try:
                return func(*args, **kwargs)
            except Exception as e:
                if log_errors:
                    logger.warning(f"{func.__name__} failed: {e}")
                if reraise:
                    raise
                return fallback
        return wrapper
    return decorator


def async_defensive(
    fallback: Any = None,
    log_errors: bool = True,
    reraise: bool = False,
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """
    Async version of defensive decorator.
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs) -> T:
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                if log_errors:
                    logger.warning(f"{func.__name__} failed: {e}")
                if reraise:
                    raise
                return fallback
        return wrapper
    return decorator


# =============================================================================
# FEATURE FLAGS
# =============================================================================

class FeatureFlags:
    """
    Runtime feature detection and flags.

    Automatically detects what features are available.
    """

    def __init__(self):
        self._cache: Dict[str, bool] = {}

    def _check(self, name: str, checker: Callable[[], bool]) -> bool:
        if name not in self._cache:
            try:
                self._cache[name] = checker()
            except Exception:
                self._cache[name] = False
        return self._cache[name]

    @property
    def has_aiohttp(self) -> bool:
        return self._check('aiohttp', lambda: safe_import('aiohttp') is not None)

    @property
    def has_aiofiles(self) -> bool:
        return self._check('aiofiles', lambda: safe_import('aiofiles') is not None)

    @property
    def has_opentelemetry(self) -> bool:
        return self._check('opentelemetry', lambda: safe_import('opentelemetry') is not None)

    @property
    def has_langfuse(self) -> bool:
        return self._check('langfuse', lambda: safe_import('langfuse') is not None)

    @property
    def has_google_cloud(self) -> bool:
        return self._check('google.cloud', lambda: safe_import('google.cloud') is not None)

    @property
    def has_grpc(self) -> bool:
        return self._check('grpc', lambda: safe_import('grpc') is not None)

    @property
    def has_yaml(self) -> bool:
        return self._check('yaml', lambda: safe_import('yaml') is not None)

    def get_status(self) -> Dict[str, bool]:
        """Get all feature flags status."""
        # Trigger all checks
        _ = (
            self.has_aiohttp, self.has_aiofiles, self.has_opentelemetry,
            self.has_langfuse, self.has_google_cloud, self.has_grpc, self.has_yaml
        )
        return dict(self._cache)


# Global feature flags instance
features = FeatureFlags()


# =============================================================================
# STDOUT/STDERR SUPPRESSION
# =============================================================================

@contextmanager
def suppress_output():
    """
    Context manager to suppress stdout/stderr temporarily.

    Useful for imports that print error messages we can't control.

    USAGE:
        with suppress_output():
            import noisy_library
    """
    import io
    import sys

    # Save original
    old_stdout = sys.stdout
    old_stderr = sys.stderr

    try:
        # Redirect to null
        sys.stdout = io.StringIO()
        sys.stderr = io.StringIO()
        yield
    finally:
        # Restore
        sys.stdout = old_stdout
        sys.stderr = old_stderr


@contextmanager
def quiet_import():
    """
    Context manager for completely quiet imports.

    Suppresses warnings, stdout, stderr - everything.

    USAGE:
        with quiet_import():
            import google.cloud  # No output at all
    """
    with suppress_output():
        with suppress_all_warnings():
            yield


# =============================================================================
# GOOGLE CLOUD COMPATIBILITY FIX
# =============================================================================

def patch_google_api_core():
    """
    Patch Google API core to avoid Python 3.9 compatibility issues.

    The google.api_core._python_version_support module calls
    importlib.metadata.packages_distributions() which doesn't exist in Python 3.9.
    This patch provides a fallback.

    Call this BEFORE importing any Google Cloud libraries.
    """
    if PYTHON_310:
        return  # No patch needed for Python 3.10+

    try:
        from importlib import metadata

        # If packages_distributions doesn't exist, add a stub
        if not hasattr(metadata, 'packages_distributions'):
            def packages_distributions():
                """Compatibility stub for Python 3.9"""
                return {}

            metadata.packages_distributions = packages_distributions
            logger.debug("Patched importlib.metadata.packages_distributions for Python 3.9")
    except Exception as e:
        logger.debug(f"Could not patch importlib.metadata: {e}")


# Auto-apply patch on import
patch_google_api_core()


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Version info
    "PYTHON_VERSION",
    "PYTHON_39",
    "PYTHON_310",
    "PYTHON_311",
    "PYTHON_312",
    "VERSION_STR",
    # Safe imports
    "safe_import",
    "safe_import_from",
    "resilient_import",
    # Compat functions
    "packages_distributions",
    "create_task_group",
    "gather_with_concurrency",
    "TaskGroupCompat",
    # Exception handling
    "ExceptionGroupCompat",
    "create_exception_group",
    # Warnings
    "suppress_warnings",
    "suppress_all_warnings",
    # Output suppression
    "suppress_output",
    "quiet_import",
    # Google Cloud fix
    "patch_google_api_core",
    # Decorators
    "defensive",
    "async_defensive",
    # Feature flags
    "FeatureFlags",
    "features",
]
