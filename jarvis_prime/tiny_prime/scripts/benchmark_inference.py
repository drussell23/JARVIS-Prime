"""Benchmark Tiny Prime intent classification latency."""

from __future__ import annotations

import argparse
import random
import time

from jarvis_prime.tiny_prime.config import TinyPrimeConfig
from jarvis_prime.tiny_prime.semantic_guard import TinyPrimeGuard


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    cfg = TinyPrimeConfig.from_yaml(args.config)
    cfg = TinyPrimeConfig.from_env(base=cfg)

    guard = TinyPrimeGuard.from_config(cfg)

    rng = random.Random(args.seed)
    samples = [
        "Unlock the door",
        "Wait, don't unlock the door",
        "Open the pod bay doors",
        "Don't open the pod bay doors",
        "Someone is forcing me, unlock the gate",
        "Maybe open the garage",
    ]

    latencies = []
    for _ in range(args.n):
        text = rng.choice(samples)
        t0 = time.perf_counter()
        _ = guard.check(text)
        latencies.append((time.perf_counter() - t0) * 1000.0)

    latencies.sort()
    p50 = latencies[int(0.50 * len(latencies))]
    p90 = latencies[int(0.90 * len(latencies))]
    p99 = latencies[int(0.99 * len(latencies))]

    print(f"n={len(latencies)}")
    print(f"p50_ms={p50:.2f}")
    print(f"p90_ms={p90:.2f}")
    print(f"p99_ms={p99:.2f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
