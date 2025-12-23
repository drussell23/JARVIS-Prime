#!/usr/bin/env python3
"""
JARVIS-Prime Docker Integration Tests
======================================

Tests all Docker-based model serving enhancements:
1. Docker module imports
2. Model downloader functionality
3. Reactor-core watcher integration
4. Hot-swap bridge
5. Cross-repo integration readiness

Run: python test_docker_integration.py
"""

import asyncio
import sys
import tempfile
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))


def print_header(title: str) -> None:
    """Print a formatted test header."""
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}\n")


def print_result(test_name: str, passed: bool, message: str = "") -> None:
    """Print a test result."""
    status = "PASS" if passed else "FAIL"
    icon = "[OK]" if passed else "[X]"
    print(f"  {icon} {test_name}")
    if message:
        print(f"      {message}")


async def test_docker_imports() -> bool:
    """Test that all Docker module imports work."""
    print_header("Test 1: Docker Module Imports")

    tests_passed = 0
    tests_total = 0

    # Test LlamaServerExecutor
    tests_total += 1
    try:
        from jarvis_prime.docker.llama_server_executor import (
            LlamaServerExecutor,
            LlamaServerConfig,
        )
        print_result("LlamaServerExecutor import", True)
        tests_passed += 1
    except Exception as e:
        print_result("LlamaServerExecutor import", False, str(e))

    # Test ModelDownloader
    tests_total += 1
    try:
        from jarvis_prime.docker.model_downloader import (
            ModelDownloader,
            MODEL_CATALOG,
            download_model,
            recommend_model,
        )
        print_result("ModelDownloader import", True)
        tests_passed += 1
    except Exception as e:
        print_result("ModelDownloader import", False, str(e))

    # Test ReactorCoreWatcher
    tests_total += 1
    try:
        from jarvis_prime.docker.reactor_core_watcher import (
            ReactorCoreWatcher,
            ReactorCoreModelManifest,
            DeploymentResult,
        )
        print_result("ReactorCoreWatcher import", True)
        tests_passed += 1
    except Exception as e:
        print_result("ReactorCoreWatcher import", False, str(e))

    # Test entrypoint
    tests_total += 1
    try:
        from jarvis_prime.docker.entrypoint import main, cmd_download, cmd_serve
        print_result("Entrypoint import", True)
        tests_passed += 1
    except Exception as e:
        print_result("Entrypoint import", False, str(e))

    # Test package __init__
    tests_total += 1
    try:
        from jarvis_prime.docker import (
            LlamaServerExecutor,
            ModelDownloader,
            ReactorCoreWatcher,
        )
        print_result("Package __init__ import", True)
        tests_passed += 1
    except Exception as e:
        print_result("Package __init__ import", False, str(e))

    print(f"\n  Results: {tests_passed}/{tests_total} passed")
    return tests_passed == tests_total


async def test_model_catalog() -> bool:
    """Test model catalog functionality."""
    print_header("Test 2: Model Catalog")

    tests_passed = 0
    tests_total = 0

    from jarvis_prime.docker.model_downloader import (
        MODEL_CATALOG,
        recommend_model,
        list_available_models,
    )

    # Test catalog has models
    tests_total += 1
    catalog_count = len(MODEL_CATALOG)
    if catalog_count >= 3:
        print_result(f"Catalog populated", True, f"{catalog_count} models available")
        tests_passed += 1
    else:
        print_result(f"Catalog populated", False, f"Only {catalog_count} models")

    # Test model recommendation
    tests_total += 1
    recommended = recommend_model(use_case="testing", max_memory_gb=4.0)
    if recommended:
        print_result("Model recommendation", True, f"Recommended: {recommended}")
        tests_passed += 1
    else:
        print_result("Model recommendation", False, "No recommendation returned")

    # Test production recommendation
    tests_total += 1
    prod_recommended = recommend_model(use_case="production", max_memory_gb=12.0)
    if prod_recommended:
        spec = MODEL_CATALOG.get(prod_recommended)
        print_result("Production recommendation", True,
                     f"{prod_recommended} ({spec.size_mb}MB)")
        tests_passed += 1
    else:
        print_result("Production recommendation", False)

    # Test catalog listing
    tests_total += 1
    models = list_available_models()
    expected_models = ["tinyllama-chat", "phi-2", "mistral-7b-instruct"]
    found = [m for m in expected_models if m in models]
    if len(found) >= 2:
        print_result("Catalog listing", True, f"Found: {', '.join(found)}")
        tests_passed += 1
    else:
        print_result("Catalog listing", False, f"Only found: {found}")

    print(f"\n  Results: {tests_passed}/{tests_total} passed")
    return tests_passed == tests_total


async def test_model_downloader() -> bool:
    """Test ModelDownloader functionality (without actual downloads)."""
    print_header("Test 3: Model Downloader")

    tests_passed = 0
    tests_total = 0

    from jarvis_prime.docker.model_downloader import ModelDownloader

    with tempfile.TemporaryDirectory() as temp_dir:
        # Test initialization
        tests_total += 1
        try:
            downloader = ModelDownloader(models_dir=temp_dir)
            print_result("Downloader initialization", True)
            tests_passed += 1
        except Exception as e:
            print_result("Downloader initialization", False, str(e))
            return False

        # Test status
        tests_total += 1
        status = downloader.get_status()
        if "models_dir" in status and "catalog_models" in status:
            print_result("Status retrieval", True,
                         f"Catalog: {status['catalog_models']} models")
            tests_passed += 1
        else:
            print_result("Status retrieval", False)

        # Test auto-select
        tests_total += 1
        try:
            result = asyncio.create_task(downloader.auto_select_model(max_memory_gb=8.0))
            selected = await asyncio.wait_for(result, timeout=5.0)
            if selected:
                print_result("Auto model selection", True, f"Selected: {selected}")
                tests_passed += 1
            else:
                print_result("Auto model selection", False, "No selection")
        except Exception as e:
            print_result("Auto model selection", False, str(e))

        # Test list local models (should be empty)
        tests_total += 1
        local_models = downloader.list_local_models()
        if len(local_models) == 0:
            print_result("List local models", True, "Empty (expected)")
            tests_passed += 1
        else:
            print_result("List local models", False, f"Found {len(local_models)}")

    print(f"\n  Results: {tests_passed}/{tests_total} passed")
    return tests_passed == tests_total


async def test_reactor_core_watcher() -> bool:
    """Test ReactorCoreWatcher functionality."""
    print_header("Test 4: Reactor-Core Watcher")

    tests_passed = 0
    tests_total = 0

    from jarvis_prime.docker.reactor_core_watcher import (
        ReactorCoreWatcher,
        ReactorCoreModelManifest,
        DeploymentResult,
    )

    with tempfile.TemporaryDirectory() as watch_dir:
        with tempfile.TemporaryDirectory() as models_dir:
            # Test initialization
            tests_total += 1
            try:
                watcher = ReactorCoreWatcher(
                    watch_dir=watch_dir,
                    models_dir=models_dir,
                    auto_deploy=False,
                )
                print_result("Watcher initialization", True)
                tests_passed += 1
            except Exception as e:
                print_result("Watcher initialization", False, str(e))
                return False

            # Test directory creation
            tests_total += 1
            pending_dir = Path(watch_dir) / "pending"
            deployed_dir = Path(watch_dir) / "deployed"
            failed_dir = Path(watch_dir) / "failed"

            if pending_dir.exists() and deployed_dir.exists() and failed_dir.exists():
                print_result("Directory creation", True)
                tests_passed += 1
            else:
                print_result("Directory creation", False)

            # Test status
            tests_total += 1
            status = watcher.get_status()
            if "watch_dir" in status and "pending_models" in status:
                print_result("Status retrieval", True)
                tests_passed += 1
            else:
                print_result("Status retrieval", False)

            # Test manifest creation
            tests_total += 1
            manifest = ReactorCoreModelManifest(
                model_id="test-model",
                version="v1.0.0",
                training_run_id="run-123",
                base_model="TinyLlama-1.1B",
                quantization_method="Q4_K_M",
                training_config={"epochs": 3},
                metrics={"loss": 0.5},
                file_path="/test/model.gguf",
                sha256="abc123",
                created_at=datetime.now(),
            )
            manifest_dict = manifest.to_dict()
            if manifest_dict["model_id"] == "test-model":
                print_result("Manifest creation", True)
                tests_passed += 1
            else:
                print_result("Manifest creation", False)

            # Test deployment result
            tests_total += 1
            result = DeploymentResult(
                success=True,
                model_id="test-model",
                version="v1.0.0",
                deployed_at=datetime.now(),
            )
            result_dict = result.to_dict()
            if result_dict["success"]:
                print_result("DeploymentResult creation", True)
                tests_passed += 1
            else:
                print_result("DeploymentResult creation", False)

    print(f"\n  Results: {tests_passed}/{tests_total} passed")
    return tests_passed == tests_total


async def test_hot_swap_integration() -> bool:
    """Test hot-swap manager Docker enhancements."""
    print_header("Test 5: Hot-Swap Docker Integration")

    tests_passed = 0
    tests_total = 0

    # Test imports
    tests_total += 1
    try:
        from jarvis_prime.core.hot_swap_manager import (
            HotSwapManager,
            DockerAwareModelLoader,
            ReactorCoreHotSwapBridge,
            SwapState,
            SwapResult,
        )
        print_result("Hot-swap imports", True)
        tests_passed += 1
    except Exception as e:
        print_result("Hot-swap imports", False, str(e))
        return False

    # Test SwapState enum
    tests_total += 1
    states = [SwapState.IDLE, SwapState.LOADING_BACKGROUND, SwapState.COMPLETED]
    if all(hasattr(SwapState, s.name) for s in states):
        print_result("SwapState enum", True)
        tests_passed += 1
    else:
        print_result("SwapState enum", False)

    # Test SwapResult
    tests_total += 1
    result = SwapResult(
        success=True,
        state=SwapState.COMPLETED,
        old_version="v1.0",
        new_version="v1.1",
        duration_seconds=5.5,
    )
    result_dict = result.to_dict()
    if result_dict["success"] and result_dict["state"] == "completed":
        print_result("SwapResult creation", True)
        tests_passed += 1
    else:
        print_result("SwapResult creation", False)

    print(f"\n  Results: {tests_passed}/{tests_total} passed")
    return tests_passed == tests_total


async def test_core_components() -> bool:
    """Test core component imports and functionality."""
    print_header("Test 6: Core Components")

    tests_passed = 0
    tests_total = 0

    # Test model_manager
    tests_total += 1
    try:
        from jarvis_prime.core.model_manager import (
            PrimeModelManager,
            create_api_app,
            ChatMessage,
            CompletionRequest,
        )
        print_result("model_manager imports", True)
        tests_passed += 1
    except Exception as e:
        print_result("model_manager imports", False, str(e))

    # Test model_registry
    tests_total += 1
    try:
        from jarvis_prime.core.model_registry import (
            ModelRegistry,
            ModelVersion,
            ModelLineage,
        )
        print_result("model_registry imports", True)
        tests_passed += 1
    except Exception as e:
        print_result("model_registry imports", False, str(e))

    # Test hybrid_router
    tests_total += 1
    try:
        from jarvis_prime.core.hybrid_router import (
            HybridRouter,
            TierClassification,
            TaskType,
        )
        print_result("hybrid_router imports", True)
        tests_passed += 1
    except Exception as e:
        print_result("hybrid_router imports", False, str(e))

    # Test telemetry_hook
    tests_total += 1
    try:
        from jarvis_prime.core.telemetry_hook import (
            TelemetryHook,
            PIIAnonymizer,
        )
        print_result("telemetry_hook imports", True)
        tests_passed += 1
    except Exception as e:
        print_result("telemetry_hook imports", False, str(e))

    # Test ChatMessage
    tests_total += 1
    msg = ChatMessage(role="user", content="Hello")
    if msg.role == "user" and msg.content == "Hello":
        print_result("ChatMessage creation", True)
        tests_passed += 1
    else:
        print_result("ChatMessage creation", False)

    print(f"\n  Results: {tests_passed}/{tests_total} passed")
    return tests_passed == tests_total


async def test_dockerfile_exists() -> bool:
    """Test that Dockerfile and docker-compose.yml exist."""
    print_header("Test 7: Docker Files")

    tests_passed = 0
    tests_total = 0

    project_root = Path(__file__).parent

    # Test Dockerfile
    tests_total += 1
    dockerfile = project_root / "Dockerfile"
    if dockerfile.exists():
        # Check for key contents
        content = dockerfile.read_text()
        has_llama = "llama" in content.lower()
        has_python = "python" in content.lower()
        if has_llama and has_python:
            print_result("Dockerfile", True, "Contains llama and Python setup")
            tests_passed += 1
        else:
            print_result("Dockerfile", False, "Missing expected content")
    else:
        print_result("Dockerfile", False, "File not found")

    # Test docker-compose.yml
    tests_total += 1
    compose = project_root / "docker-compose.yml"
    if compose.exists():
        content = compose.read_text()
        has_service = "jarvis-prime:" in content
        has_volume = "volumes:" in content
        if has_service and has_volume:
            print_result("docker-compose.yml", True, "Contains service and volumes")
            tests_passed += 1
        else:
            print_result("docker-compose.yml", False, "Missing expected content")
    else:
        print_result("docker-compose.yml", False, "File not found")

    # Test requirements.txt
    tests_total += 1
    requirements = project_root / "requirements.txt"
    if requirements.exists():
        content = requirements.read_text()
        has_fastapi = "fastapi" in content.lower()
        has_httpx = "httpx" in content.lower()
        if has_fastapi and has_httpx:
            print_result("requirements.txt", True, "Contains core dependencies")
            tests_passed += 1
        else:
            print_result("requirements.txt", False, "Missing expected dependencies")
    else:
        print_result("requirements.txt", False, "File not found")

    print(f"\n  Results: {tests_passed}/{tests_total} passed")
    return tests_passed == tests_total


async def main() -> int:
    """Run all integration tests."""
    print("\n" + "=" * 60)
    print("  JARVIS-Prime Docker Integration Tests")
    print("=" * 60)

    results = []

    # Run all tests
    results.append(("Docker Imports", await test_docker_imports()))
    results.append(("Model Catalog", await test_model_catalog()))
    results.append(("Model Downloader", await test_model_downloader()))
    results.append(("Reactor-Core Watcher", await test_reactor_core_watcher()))
    results.append(("Hot-Swap Integration", await test_hot_swap_integration()))
    results.append(("Core Components", await test_core_components()))
    results.append(("Docker Files", await test_dockerfile_exists()))

    # Summary
    print_header("Test Summary")

    passed = sum(1 for _, p in results if p)
    total = len(results)

    for name, result in results:
        status = "[OK]" if result else "[X]"
        print(f"  {status} {name}")

    print(f"\n  Overall: {passed}/{total} test suites passed")
    print("=" * 60)

    if passed == total:
        print("\n  All Docker integration tests PASSED!")
        print("\n  Next steps:")
        print("  1. Build Docker image: docker build -t jarvis-prime:latest .")
        print("  2. Download a model: docker-compose run model-downloader download --catalog tinyllama-chat")
        print("  3. Start server: docker-compose up -d jarvis-prime")
        print("  4. Test: curl http://localhost:8000/health")
        return 0
    else:
        print("\n  Some tests FAILED. Please check the output above.")
        return 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
