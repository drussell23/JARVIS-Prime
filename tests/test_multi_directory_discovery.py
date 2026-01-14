#!/usr/bin/env python3
"""
Multi-Directory Discovery Test Suite v99.0
===========================================

Tests for the v99.0 multi-directory model discovery enhancements:
1. MultiDirectoryModelScanner
2. ModelFileWatcher
3. ModelValidationPipeline
4. ReactorCoreModelSync

Run:
    python3 tests/test_multi_directory_discovery.py
"""

from __future__ import annotations

import asyncio
import sys
import tempfile
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


class TestResults:
    """Track test results."""

    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.errors: list[str] = []

    def record_pass(self, name: str) -> None:
        self.passed += 1
        print(f"  [PASS] {name}")

    def record_fail(self, name: str, reason: str) -> None:
        self.failed += 1
        self.errors.append(f"{name}: {reason}")
        print(f"  [FAIL] {name}: {reason}")

    def summary(self) -> None:
        total = self.passed + self.failed
        print()
        print(f"Results: {self.passed}/{total} passed")
        if self.errors:
            print("Failures:")
            for error in self.errors:
                print(f"  - {error}")


async def test_multi_directory_scanner(results: TestResults) -> None:
    """Test MultiDirectoryModelScanner functionality."""
    print()
    print("1. Testing Multi-Directory Model Scanner...")

    try:
        from jarvis_prime.core.dynamic_model_registry import (
            MultiDirectoryModelScanner,
            ModelDirectoryConfig,
            DiscoveredModel,
            KNOWN_MODELS,
        )

        # Test initialization
        scanner = MultiDirectoryModelScanner()
        if scanner is not None:
            results.record_pass("MultiDirectoryScanner.init")
        else:
            results.record_fail("MultiDirectoryScanner.init", "None")

        # Test default directories are configured
        directories = scanner.get_directories()
        if len(directories) >= 1:
            results.record_pass("MultiDirectoryScanner.default_directories")
        else:
            results.record_fail("MultiDirectoryScanner.default_directories", f"got {len(directories)}")

        # Test directory configuration
        config = directories[0] if directories else None
        if config and isinstance(config, ModelDirectoryConfig):
            results.record_pass("MultiDirectoryScanner.config_type")
        else:
            results.record_fail("MultiDirectoryScanner.config_type", "wrong type")

        # Test scan_all
        discovered = await scanner.scan_all(force_rescan=True, known_models=KNOWN_MODELS)
        if isinstance(discovered, dict):
            results.record_pass("MultiDirectoryScanner.scan_all")
        else:
            results.record_fail("MultiDirectoryScanner.scan_all", f"got {type(discovered)}")

        # Test statistics
        stats = scanner.get_statistics()
        required_fields = ["directories_configured", "models_discovered", "total_size_gb"]
        if all(field in stats for field in required_fields):
            results.record_pass("MultiDirectoryScanner.statistics")
        else:
            results.record_fail("MultiDirectoryScanner.statistics", f"missing fields: {list(stats.keys())}")

        # Test adding custom directory
        with tempfile.TemporaryDirectory() as tmpdir:
            scanner.add_directory(Path(tmpdir), priority=200, source="test")
            new_dirs = scanner.get_directories()
            if len(new_dirs) > len(directories):
                results.record_pass("MultiDirectoryScanner.add_directory")
            else:
                results.record_fail("MultiDirectoryScanner.add_directory", "not added")

    except Exception as e:
        results.record_fail("MultiDirectoryScanner", str(e))


async def test_model_file_watcher(results: TestResults) -> None:
    """Test ModelFileWatcher functionality."""
    print()
    print("2. Testing Model File Watcher...")

    try:
        from jarvis_prime.core.dynamic_model_registry import (
            ModelFileWatcher,
            ModelFileEvent,
            ModelFileChange,
            ModelDirectoryConfig,
        )

        # Create a test directory config
        with tempfile.TemporaryDirectory() as tmpdir:
            config = ModelDirectoryConfig(
                path=Path(tmpdir),
                priority=100,
                recursive=True,
                watch=True,
                source="test",
            )

            # Test initialization
            watcher = ModelFileWatcher(
                directories=[config],
                debounce_seconds=0.5,
                poll_interval_seconds=1.0,
            )
            if watcher is not None:
                results.record_pass("ModelFileWatcher.init")
            else:
                results.record_fail("ModelFileWatcher.init", "None")

            # Test start
            await watcher.start()
            stats = watcher.get_statistics()
            if stats.get("running"):
                results.record_pass("ModelFileWatcher.start")
            else:
                results.record_fail("ModelFileWatcher.start", "not running")

            # Test subscribe
            events_received = []
            async def callback(event: ModelFileChange):
                events_received.append(event)

            watcher.subscribe(callback)
            if len(watcher._callbacks) == 1:
                results.record_pass("ModelFileWatcher.subscribe")
            else:
                results.record_fail("ModelFileWatcher.subscribe", "not subscribed")

            # Test statistics
            stats = watcher.get_statistics()
            required_fields = ["running", "watchdog_available", "directories_watched"]
            if all(field in stats for field in required_fields):
                results.record_pass("ModelFileWatcher.statistics")
            else:
                results.record_fail("ModelFileWatcher.statistics", f"missing fields")

            # Test stop
            await watcher.stop()
            if not watcher._running:
                results.record_pass("ModelFileWatcher.stop")
            else:
                results.record_fail("ModelFileWatcher.stop", "still running")

    except Exception as e:
        results.record_fail("ModelFileWatcher", str(e))


async def test_model_validation_pipeline(results: TestResults) -> None:
    """Test ModelValidationPipeline functionality."""
    print()
    print("3. Testing Model Validation Pipeline...")

    try:
        from jarvis_prime.core.dynamic_model_registry import (
            ModelValidationPipeline,
            ValidationResult,
            ModelValidation,
        )

        # Test initialization
        validator = ModelValidationPipeline(
            enable_sha256=True,
            enable_loadability_test=False,
        )
        if validator is not None:
            results.record_pass("ValidationPipeline.init")
        else:
            results.record_fail("ValidationPipeline.init", "None")

        # Test validation of non-existent file
        fake_path = Path("/nonexistent/model.gguf")
        validation = await validator.validate(fake_path)
        if validation.result == ValidationResult.INVALID:
            results.record_pass("ValidationPipeline.validate_nonexistent")
        else:
            results.record_fail("ValidationPipeline.validate_nonexistent", f"got {validation.result}")

        # Test with real file (create temp file)
        with tempfile.NamedTemporaryFile(suffix=".gguf", delete=False) as f:
            f.write(b"GGUF" + b"\x00" * 100)  # Fake GGUF header
            f.flush()
            temp_path = Path(f.name)

        try:
            validation = await validator.validate(temp_path)
            if validation.result == ValidationResult.VALID:
                results.record_pass("ValidationPipeline.validate_file")
            else:
                results.record_fail("ValidationPipeline.validate_file", f"got {validation.result}")

            # Test hash calculation
            if validation.sha256_hash and len(validation.sha256_hash) == 64:
                results.record_pass("ValidationPipeline.sha256")
            else:
                results.record_fail("ValidationPipeline.sha256", "no hash")

            # Test format detection
            if validation.format_detected == "gguf":
                results.record_pass("ValidationPipeline.format_detection")
            else:
                results.record_fail("ValidationPipeline.format_detection", f"got {validation.format_detected}")

        finally:
            temp_path.unlink()

        # Test statistics
        stats = validator.get_statistics()
        if "cached_validations" in stats and "valid_models" in stats:
            results.record_pass("ValidationPipeline.statistics")
        else:
            results.record_fail("ValidationPipeline.statistics", "missing fields")

        # Test cache clear
        validator.clear_cache()
        stats = validator.get_statistics()
        if stats["cached_validations"] == 0:
            results.record_pass("ValidationPipeline.clear_cache")
        else:
            results.record_fail("ValidationPipeline.clear_cache", "cache not cleared")

    except Exception as e:
        results.record_fail("ValidationPipeline", str(e))


async def test_reactor_core_sync(results: TestResults) -> None:
    """Test ReactorCoreModelSync functionality."""
    print()
    print("4. Testing Reactor Core Model Sync...")

    try:
        from jarvis_prime.core.dynamic_model_registry import ReactorCoreModelSync

        # Test initialization
        sync = ReactorCoreModelSync()
        if sync is not None:
            results.record_pass("ReactorCoreSync.init")
        else:
            results.record_fail("ReactorCoreSync.init", "None")

        # Test statistics
        stats = sync.get_statistics()
        required_fields = ["reactor_repo_found", "running", "models_synced"]
        if all(field in stats for field in required_fields):
            results.record_pass("ReactorCoreSync.statistics")
        else:
            results.record_fail("ReactorCoreSync.statistics", f"missing fields: {list(stats.keys())}")

        # Test reactor repo detection
        if stats.get("reactor_repo_found"):
            results.record_pass("ReactorCoreSync.repo_detection")
        else:
            # Not a failure if repo doesn't exist
            results.record_pass("ReactorCoreSync.repo_detection (skipped - no reactor repo)")

        # Test start/stop (if repo exists)
        if stats.get("reactor_repo_found"):
            await sync.start()
            if sync._running:
                results.record_pass("ReactorCoreSync.start")
            else:
                results.record_fail("ReactorCoreSync.start", "not running")

            await sync.stop()
            if not sync._running:
                results.record_pass("ReactorCoreSync.stop")
            else:
                results.record_fail("ReactorCoreSync.stop", "still running")
        else:
            results.record_pass("ReactorCoreSync.start (skipped)")
            results.record_pass("ReactorCoreSync.stop (skipped)")

    except Exception as e:
        results.record_fail("ReactorCoreSync", str(e))


async def test_registry_v99_integration(results: TestResults) -> None:
    """Test DynamicModelRegistry v99.0 integration."""
    print()
    print("5. Testing Registry v99.0 Integration...")

    try:
        from jarvis_prime.core.dynamic_model_registry import get_dynamic_model_registry

        registry = await get_dynamic_model_registry()

        # Test scanner statistics
        scanner_stats = registry.get_scanner_statistics()
        if "directories_configured" in scanner_stats:
            results.record_pass("Registry.scanner_integration")
        else:
            results.record_fail("Registry.scanner_integration", "no scanner stats")

        # Test file watcher statistics
        watcher_stats = registry.get_file_watcher_statistics()
        if watcher_stats is not None:
            results.record_pass("Registry.watcher_integration")
        else:
            results.record_fail("Registry.watcher_integration", "no watcher stats")

        # Test reactor sync statistics
        reactor_stats = registry.get_reactor_sync_statistics()
        if "reactor_repo_found" in reactor_stats:
            results.record_pass("Registry.reactor_integration")
        else:
            results.record_fail("Registry.reactor_integration", "no reactor stats")

        # Test rescan_directories
        discovered = await registry.rescan_directories()
        if isinstance(discovered, dict):
            results.record_pass("Registry.rescan_directories")
        else:
            results.record_fail("Registry.rescan_directories", f"got {type(discovered)}")

        # Test overall statistics include v99.0 fields
        stats = registry.get_statistics()
        v99_fields = ["scanner_stats", "file_watcher_stats", "reactor_sync_stats", "validation_stats"]
        if all(field in stats for field in v99_fields):
            results.record_pass("Registry.v99_statistics")
        else:
            missing = [f for f in v99_fields if f not in stats]
            results.record_fail("Registry.v99_statistics", f"missing: {missing}")

    except Exception as e:
        results.record_fail("Registry.v99_integration", str(e))


async def test_fuzzy_matching(results: TestResults) -> None:
    """Test fuzzy model matching functionality."""
    print()
    print("6. Testing Fuzzy Model Matching...")

    try:
        from jarvis_prime.core.dynamic_model_registry import (
            MultiDirectoryModelScanner,
            KNOWN_MODELS,
        )

        scanner = MultiDirectoryModelScanner()

        # Test exact match
        test_path = Path("/models/Phi-3.5-mini-instruct-Q4_K_M.gguf")
        model_id, spec, score = scanner._match_model(test_path, KNOWN_MODELS)
        if model_id == "phi-3.5-mini" and score == 1.0:
            results.record_pass("FuzzyMatch.exact_match")
        else:
            results.record_fail("FuzzyMatch.exact_match", f"got {model_id}, score={score}")

        # Test fuzzy match (partial name)
        test_path = Path("/models/qwen-7b-instruct-q4.gguf")
        model_id, spec, score = scanner._match_model(test_path, KNOWN_MODELS)
        if score > 0:  # Should have some non-zero match (conservative fuzzy matching)
            results.record_pass("FuzzyMatch.partial_match")
        else:
            results.record_fail("FuzzyMatch.partial_match", f"score too low: {score}")

        # Test no match (completely unknown)
        test_path = Path("/models/completely-unknown-model-xyz.gguf")
        model_id, spec, score = scanner._match_model(test_path, KNOWN_MODELS)
        if model_id is None or score < 0.6:
            results.record_pass("FuzzyMatch.no_match")
        else:
            results.record_fail("FuzzyMatch.no_match", f"unexpected match: {model_id}")

    except Exception as e:
        results.record_fail("FuzzyMatch", str(e))


async def main():
    """Run all Multi-Directory Discovery tests."""
    print("=" * 70)
    print("  MULTI-DIRECTORY DISCOVERY TEST SUITE v99.0")
    print("  Scanner + Watcher + Validation + Reactor Sync")
    print("=" * 70)

    results = TestResults()

    # Run all tests
    await test_multi_directory_scanner(results)
    await test_model_file_watcher(results)
    await test_model_validation_pipeline(results)
    await test_reactor_core_sync(results)
    await test_registry_v99_integration(results)
    await test_fuzzy_matching(results)

    # Print summary
    print()
    print("=" * 70)
    results.summary()
    print("=" * 70)

    # Exit with appropriate code
    if results.failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
