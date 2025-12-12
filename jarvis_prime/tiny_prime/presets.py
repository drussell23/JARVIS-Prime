"""Tiny Prime preset configs."""

from __future__ import annotations

from importlib import resources

from jarvis_prime.tiny_prime.config import TinyPrimeConfig


class TinyPrimePresets:
    @staticmethod
    def tiny_prime_v0_1_0() -> TinyPrimeConfig:
        with resources.files("jarvis_prime.tiny_prime").joinpath(
            "config/model_config.yaml"
        ).open("r", encoding="utf-8") as f:
            try:
                import yaml  # type: ignore
            except ImportError as e:
                raise ImportError(
                    "PyYAML is required to load the bundled Tiny Prime YAML config. "
                    "Install with: pip install pyyaml"
                ) from e

            raw = yaml.safe_load(f) or {}
        return TinyPrimeConfig(raw=raw, source_path=None)
