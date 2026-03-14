"""
Quality Regression Tester — A/B Benchmarking
=============================================

Measures quality/speed of quantization variants for calibration data.
Runs asynchronously in background during idle periods.

Preemption rules:
- Always preemptible by production traffic
- Max 30s per prompt
- Results used for calibration only, never for pass/fail gating
"""
from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from jarvis_prime.core.quantization_intelligence import CalibrationData, CalibrationPoint

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class BenchmarkPrompt:
    name: str
    prompt: str
    expected_patterns: tuple
    max_tokens: int
    temperature: float = 0.1


@dataclass(frozen=True)
class BenchmarkSuite:
    prompts: tuple
    version: str


@dataclass(frozen=True)
class BenchmarkResult:
    variant_name: str
    suite_version: str
    mean_tok_s: float
    p50_tok_s: float
    p95_first_token_ms: float
    quality_score: float
    vram_peak_bytes: int
    context_tested: int
    timestamp: float


DEFAULT_SUITE = BenchmarkSuite(
    version="1.0",
    prompts=(
        BenchmarkPrompt(
            name="python_function",
            prompt="Write a Python function that checks if a number is prime. Return only the code.",
            expected_patterns=("def ", "return"),
            max_tokens=100,
        ),
        BenchmarkPrompt(
            name="explain_concept",
            prompt="Explain what a binary search tree is in 2-3 sentences.",
            expected_patterns=("tree", "node"),
            max_tokens=80,
        ),
        BenchmarkPrompt(
            name="code_review",
            prompt="Review this code and suggest improvements:\ndef add(a,b): return a+b",
            expected_patterns=("type", "def "),
            max_tokens=100,
        ),
        BenchmarkPrompt(
            name="fibonacci",
            prompt="Write a Python function to compute the nth Fibonacci number iteratively.",
            expected_patterns=("def ", "fib"),
            max_tokens=80,
        ),
    ),
)


class QualityRegressionTester:
    """Background A/B benchmarking for calibration."""

    def __init__(
        self,
        executor: Any = None,
        calibration_dir: Optional[Path] = None,
    ):
        self._executor = executor
        self._calibration_dir = calibration_dir or Path("models/calibration")

    async def run_single(
        self,
        prompt: BenchmarkPrompt,
        model_name: str,
    ) -> Optional[BenchmarkResult]:
        """Run a single benchmark prompt. Preemptible."""
        if not self._executor or not self._executor.is_loaded():
            return None

        start = time.monotonic()
        try:
            output = await self._executor.generate(
                prompt=prompt.prompt,
                max_tokens=prompt.max_tokens,
                temperature=prompt.temperature,
            )
        except Exception as e:
            logger.warning(f"[QualityTester] Benchmark failed: {e}")
            return None

        elapsed = time.monotonic() - start
        tokens_approx = len(output.split()) * 1.3  # rough token estimate
        tok_s = tokens_approx / elapsed if elapsed > 0 else 0

        # Quality: ratio of expected patterns found
        matches = sum(1 for p in prompt.expected_patterns if re.search(p, output, re.IGNORECASE))
        quality = matches / len(prompt.expected_patterns) if prompt.expected_patterns else 1.0

        return BenchmarkResult(
            variant_name=model_name,
            suite_version=DEFAULT_SUITE.version,
            mean_tok_s=tok_s,
            p50_tok_s=tok_s,
            p95_first_token_ms=elapsed * 1000,
            quality_score=quality,
            vram_peak_bytes=0,
            context_tested=prompt.max_tokens,
            timestamp=time.time(),
        )

    def save_result(self, model_family: str, result: BenchmarkResult) -> None:
        """Persist benchmark result to calibration file."""
        self._calibration_dir.mkdir(parents=True, exist_ok=True)
        path = self._calibration_dir / f"{model_family}.json"

        data: Dict[str, Any] = {}
        if path.exists():
            try:
                data = json.loads(path.read_text())
            except (json.JSONDecodeError, OSError):
                pass

        data.setdefault("measurements", {})[result.variant_name] = {
            "measured_tok_s": result.mean_tok_s,
            "measured_perplexity": None,
            "measured_vram_bytes": result.vram_peak_bytes,
            "context_size": result.context_tested,
            "quality_score": result.quality_score,
            "timestamp": result.timestamp,
            "suite_version": result.suite_version,
        }
        path.write_text(json.dumps(data, indent=2))

    def load_calibration(self, model_family: str) -> Optional[CalibrationData]:
        """Load calibration data from disk."""
        path = self._calibration_dir / f"{model_family}.json"
        if not path.exists():
            return None

        try:
            data = json.loads(path.read_text())
        except (json.JSONDecodeError, OSError):
            return None

        measurements: Dict[str, CalibrationPoint] = {}
        for name, m in data.get("measurements", {}).items():
            measurements[name] = CalibrationPoint(
                quant_name=name,
                measured_tok_s=m["measured_tok_s"],
                measured_perplexity=m.get("measured_perplexity"),
                measured_vram_bytes=m["measured_vram_bytes"],
                context_size=m["context_size"],
                timestamp=m["timestamp"],
            )

        return CalibrationData(model_family=model_family, measurements=measurements)
