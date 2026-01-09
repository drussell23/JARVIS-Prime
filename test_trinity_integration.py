#!/usr/bin/env python3
"""
Trinity Integration Test - v84.0
=================================

Comprehensive test script for verifying Trinity cross-repo integration.

Usage:
    python test_trinity_integration.py              # Full test
    python test_trinity_integration.py --health     # Health check only
    python test_trinity_integration.py --repos      # Repo detection only
    python test_trinity_integration.py --inference  # Test inference endpoint
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path

# Add jarvis-prime to path
sys.path.insert(0, str(Path(__file__).parent))


async def test_health_checker():
    """Test TrinityHealthChecker."""
    print()
    print("=" * 60)
    print("TEST: TrinityHealthChecker")
    print("=" * 60)

    from jarvis_prime.core.cross_repo_orchestrator import (
        TrinityHealthChecker,
        check_trinity_health,
        print_trinity_status,
    )

    # Create health checker
    checker = TrinityHealthChecker()

    # Check all components
    print("\n1. Checking all Trinity components...")
    health_map = await checker.check_all()

    for name, health in health_map.items():
        status_icon = {
            "HEALTHY": "✅",
            "DEGRADED": "⚠️",
            "STALE": "🔶",
            "OFFLINE": "❌",
        }.get(health.status, "❓")

        print(f"  {status_icon} {name:20s} {health.status:10s}", end="")
        if health.online:
            print(f" (age: {health.heartbeat_age:.1f}s)", end="")
            if health.http_healthy:
                print(f" [HTTP OK ::{health.port}]", end="")
        elif health.error:
            print(f" ({health.error})", end="")
        print()

    # Print unified status
    print("\n2. Unified status:")
    status = checker.get_unified_status()
    print(f"  Overall: {status['status'].upper()}")
    print(f"  Components online: {sum(1 for c in status['components'].values() if c['online'])}")

    return all(h.online for h in health_map.values())


async def test_repo_detection():
    """Test repository auto-detection."""
    print()
    print("=" * 60)
    print("TEST: Repository Detection")
    print("=" * 60)

    from jarvis_prime.core.cross_repo_orchestrator import (
        CrossRepoOrchestrator,
        OrchestratorConfig,
    )

    # Create orchestrator
    config = OrchestratorConfig()
    orchestrator = CrossRepoOrchestrator(config)

    # Detect repos
    repos = orchestrator._detect_repos()

    print(f"\nFound {len(repos)} repositories:")
    for name, repo_config in repos.items():
        exists = repo_config.path.exists()
        status = "✅" if exists else "❌"
        print(f"\n  {status} {repo_config.name}")
        print(f"     Path: {repo_config.path}")
        print(f"     Entry: {repo_config.entry_point}")
        print(f"     Health: {repo_config.health_url}")
        print(f"     Deps: {repo_config.dependencies or 'none'}")

    return len(repos) > 0


async def test_heartbeat_files():
    """Test heartbeat file system."""
    print()
    print("=" * 60)
    print("TEST: Heartbeat File System")
    print("=" * 60)

    trinity_dir = Path.home() / ".jarvis" / "trinity"
    components_dir = trinity_dir / "components"

    print(f"\n1. Trinity directory: {trinity_dir}")
    print(f"   Exists: {'✅' if trinity_dir.exists() else '❌'}")

    print(f"\n2. Components directory: {components_dir}")
    print(f"   Exists: {'✅' if components_dir.exists() else '❌'}")

    if components_dir.exists():
        print("\n3. Heartbeat files:")
        heartbeat_files = list(components_dir.glob("*.json"))

        if not heartbeat_files:
            print("   ⚠️ No heartbeat files found")
        else:
            for hb_file in heartbeat_files:
                try:
                    with open(hb_file) as f:
                        data = json.load(f)

                    age = time.time() - data.get("timestamp", 0)
                    status = "🟢" if age < 30 else "🟡" if age < 60 else "🔴"

                    print(f"\n   {status} {hb_file.name}")
                    print(f"      Age: {age:.1f}s")
                    print(f"      Port: {data.get('port', 'N/A')}")
                    print(f"      PID: {data.get('pid', 'N/A')}")
                    print(f"      Model: {'✅' if data.get('model_loaded') else '❌'}")

                except Exception as e:
                    print(f"   ❌ {hb_file.name}: {e}")

    return components_dir.exists()


async def test_inference_endpoint():
    """Test J-Prime inference endpoint."""
    print()
    print("=" * 60)
    print("TEST: Inference Endpoint")
    print("=" * 60)

    port = int(os.getenv("JARVIS_PRIME_PORT", "8000"))
    url = f"http://localhost:{port}/v1/chat/completions"

    print(f"\n1. Testing: {url}")

    try:
        import aiohttp

        payload = {
            "messages": [{"role": "user", "content": "Say hello in exactly 3 words."}],
            "max_tokens": 20,
        }

        print("   Sending request...")
        start = time.time()

        async with aiohttp.ClientSession() as session:
            async with session.post(
                url,
                json=payload,
                timeout=aiohttp.ClientTimeout(total=60),
            ) as response:
                elapsed = time.time() - start

                if response.status == 200:
                    data = await response.json()
                    content = data["choices"][0]["message"]["content"]
                    tokens = data.get("usage", {}).get("total_tokens", "?")

                    print(f"   ✅ Response received in {elapsed:.2f}s")
                    print(f"   Content: \"{content}\"")
                    print(f"   Tokens: {tokens}")
                    return True
                else:
                    print(f"   ❌ HTTP {response.status}")
                    return False

    except ImportError:
        print("   ⚠️ aiohttp not installed: pip install aiohttp")
        return False
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False


async def test_full_orchestrator():
    """Test full orchestrator lifecycle."""
    print()
    print("=" * 60)
    print("TEST: Full Orchestrator Lifecycle")
    print("=" * 60)

    from jarvis_prime.core.cross_repo_orchestrator import get_orchestrator

    print("\n1. Getting orchestrator...")
    orchestrator = await get_orchestrator()
    print(f"   ✅ Orchestrator initialized")
    print(f"   Repos: {list(orchestrator._repos.keys())}")
    print(f"   Startup order: {orchestrator._startup_order}")

    print("\n2. Health checker status:")
    await orchestrator._health_checker.print_status()

    return True


async def main():
    """Run all tests."""
    parser = argparse.ArgumentParser(description="Trinity Integration Tests")
    parser.add_argument("--health", action="store_true", help="Health check only")
    parser.add_argument("--repos", action="store_true", help="Repo detection only")
    parser.add_argument("--heartbeat", action="store_true", help="Heartbeat files only")
    parser.add_argument("--inference", action="store_true", help="Inference test only")
    parser.add_argument("--orchestrator", action="store_true", help="Orchestrator test only")
    args = parser.parse_args()

    print()
    print("╔" + "═" * 58 + "╗")
    print("║" + " TRINITY INTEGRATION TEST v84.0 ".center(58) + "║")
    print("╚" + "═" * 58 + "╝")

    results = {}

    # Run specific tests or all
    if args.health or not any([args.repos, args.heartbeat, args.inference, args.orchestrator]):
        results["health"] = await test_health_checker()

    if args.repos or not any([args.health, args.heartbeat, args.inference, args.orchestrator]):
        results["repos"] = await test_repo_detection()

    if args.heartbeat or not any([args.health, args.repos, args.inference, args.orchestrator]):
        results["heartbeat"] = await test_heartbeat_files()

    if args.inference:
        results["inference"] = await test_inference_endpoint()

    if args.orchestrator:
        results["orchestrator"] = await test_full_orchestrator()

    # Summary
    print()
    print("=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)

    passed = sum(1 for v in results.values() if v)
    total = len(results)

    for name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {status}  {name}")

    print()
    print(f"Results: {passed}/{total} passed")

    if passed == total:
        print("\n🎉 All tests passed!")
        return 0
    else:
        print("\n⚠️ Some tests failed")
        return 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
