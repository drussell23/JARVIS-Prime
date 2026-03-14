"""Tests for adaptive_model_selector.py — inventory + proposals."""
from __future__ import annotations

import pytest
from pathlib import Path

from jarvis_prime.core.adaptive_model_selector import (
    ModelVariant,
    ModelFamily,
    ModelSelectionProposal,
    parse_gguf_filename,
    scan_inventory,
    propose_optimal,
)
from jarvis_prime.core.quantization_intelligence import KNOWN_PROFILES


class TestParseGgufFilename:

    def test_standard_format(self):
        base, quant = parse_gguf_filename("Qwen2.5-Coder-32B-Instruct-Q4_K_M.gguf")
        assert base == "qwen2.5-coder-32b-instruct"
        assert quant == "Q4_K_M"

    def test_iq_format(self):
        base, quant = parse_gguf_filename("Qwen2.5-Coder-32B-Instruct-IQ2_M.gguf")
        assert base == "qwen2.5-coder-32b-instruct"
        assert quant == "IQ2_M"

    def test_llama_format(self):
        base, quant = parse_gguf_filename("Llama-3.2-1B-Instruct-Q4_K_M.gguf")
        assert base == "llama-3.2-1b-instruct"
        assert quant == "Q4_K_M"

    def test_unknown_quant_returns_none(self):
        base, quant = parse_gguf_filename("some-random-model.gguf")
        assert quant is None


class TestScanInventory:

    @pytest.mark.asyncio
    async def test_groups_by_family(self, fake_gguf_files, tmp_models_dir):
        families = await scan_inventory(tmp_models_dir)
        family_names = {f.base_model for f in families}
        assert "qwen2.5-coder-32b-instruct" in family_names

    @pytest.mark.asyncio
    async def test_32b_has_two_variants(self, fake_gguf_files, tmp_models_dir):
        families = await scan_inventory(tmp_models_dir)
        family_32b = next(f for f in families if "32b" in f.base_model)
        assert len(family_32b.variants) == 2  # IQ2_M + Q4_K_M

    @pytest.mark.asyncio
    async def test_empty_dir(self, tmp_models_dir):
        families = await scan_inventory(tmp_models_dir)
        assert families == []


class TestProposeOptimal:

    @pytest.mark.asyncio
    async def test_proposes_model_that_fits(self, fake_gguf_files, tmp_models_dir):
        families = await scan_inventory(tmp_models_dir)
        proposal = await propose_optimal(
            families=families,
            vram_budget_bytes=23_034 * 1024 * 1024,
            target_context=8192,
            task_complexity="medium",
            current_model=None,
        )
        assert proposal is not None
        assert proposal.selected_variant.size_bytes < 23_034 * 1024 * 1024

    @pytest.mark.asyncio
    async def test_proposal_has_reason(self, fake_gguf_files, tmp_models_dir):
        families = await scan_inventory(tmp_models_dir)
        proposal = await propose_optimal(
            families=families,
            vram_budget_bytes=23_034 * 1024 * 1024,
            target_context=8192,
            task_complexity="medium",
            current_model=None,
        )
        assert proposal is not None
        assert len(proposal.reason) > 0


from jarvis_prime.core.adaptive_model_selector import verify_model_integrity


class TestVerifyModelIntegrity:

    def test_file_not_found(self, tmp_path):
        ok, reason = verify_model_integrity(tmp_path / "nonexistent.gguf")
        assert ok is False
        assert "not found" in reason

    def test_size_mismatch(self, tmp_path):
        f = tmp_path / "test.gguf"
        f.write_bytes(b"\0" * 1000)
        ok, reason = verify_model_integrity(f, expected_size_bytes=5000)
        assert ok is False
        assert "Size mismatch" in reason

    def test_size_within_tolerance(self, tmp_path):
        f = tmp_path / "test.gguf"
        f.write_bytes(b"\0" * 1000)
        ok, _ = verify_model_integrity(f, expected_size_bytes=1005)  # <1% off
        assert ok is True

    def test_sha256_match(self, tmp_path):
        import hashlib
        data = b"hello model data"
        f = tmp_path / "test.gguf"
        f.write_bytes(data)
        expected = hashlib.sha256(data).hexdigest()
        ok, _ = verify_model_integrity(f, expected_sha256=expected)
        assert ok is True

    def test_sha256_mismatch(self, tmp_path):
        f = tmp_path / "test.gguf"
        f.write_bytes(b"some data")
        ok, reason = verify_model_integrity(f, expected_sha256="0000" * 16)
        assert ok is False
        assert "SHA256 mismatch" in reason
