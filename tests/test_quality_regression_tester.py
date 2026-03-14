"""Tests for quality_regression_tester.py — A/B benchmarking."""
from __future__ import annotations

import json
import pytest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

from jarvis_prime.core.quality_regression_tester import (
    BenchmarkPrompt,
    BenchmarkSuite,
    BenchmarkResult,
    QualityRegressionTester,
    DEFAULT_SUITE,
)


class TestBenchmarkSuite:

    def test_default_suite_has_prompts(self):
        assert len(DEFAULT_SUITE.prompts) >= 3

    def test_prompts_have_low_temperature(self):
        for p in DEFAULT_SUITE.prompts:
            assert p.temperature <= 0.2


class TestQualityRegressionTester:

    @pytest.mark.asyncio
    async def test_run_benchmark_returns_result(self):
        executor = MagicMock()
        executor.generate = AsyncMock(return_value="def hello():\n    print('hello')\n")
        executor.is_loaded = MagicMock(return_value=True)

        tester = QualityRegressionTester(executor=executor)
        prompt = BenchmarkPrompt(
            name="simple_func",
            prompt="Write a Python hello world function",
            expected_patterns=("def ",),
            max_tokens=50,
        )
        result = await tester.run_single(prompt, model_name="test-model")
        assert result is not None
        assert result.mean_tok_s >= 0
        assert result.quality_score >= 0

    @pytest.mark.asyncio
    async def test_save_and_load_calibration(self, tmp_path):
        cal_dir = tmp_path / "calibration"
        cal_dir.mkdir()

        tester = QualityRegressionTester(
            executor=MagicMock(),
            calibration_dir=cal_dir,
        )
        result = BenchmarkResult(
            variant_name="test-model-Q4_K_M",
            suite_version="1.0",
            mean_tok_s=45.0,
            p50_tok_s=44.5,
            p95_first_token_ms=210.0,
            quality_score=0.92,
            vram_peak_bytes=9_000_000_000,
            context_tested=8192,
            timestamp=1710400000.0,
        )
        tester.save_result("qwen2.5-coder-7b", result)

        loaded = tester.load_calibration("qwen2.5-coder-7b")
        assert loaded is not None
        assert "test-model-Q4_K_M" in loaded.measurements
