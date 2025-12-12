"""Train Tiny Prime from scratch (decoder-only Llama-style)."""

from __future__ import annotations

import argparse

from jarvis_prime.tiny_prime.config import TinyPrimeConfig
from jarvis_prime.tiny_prime.trainer import TinyPrimeTrainer


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="Path to model_config.yaml")
    ap.add_argument("--resume", default=None, help="Path to HF checkpoint dir")
    args = ap.parse_args()

    cfg = TinyPrimeConfig.from_yaml(args.config)
    cfg = TinyPrimeConfig.from_env(base=cfg)

    trainer = TinyPrimeTrainer(cfg)
    trainer.train(resume_from_checkpoint=args.resume)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
