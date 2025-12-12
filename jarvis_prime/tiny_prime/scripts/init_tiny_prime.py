"""Initialize a random-weight Tiny Prime model (no training).

This is useful to validate:
- tokenizer compatibility
- model config shapes
- save/load round-trips
"""

from __future__ import annotations

import argparse
from pathlib import Path

from transformers import AutoTokenizer, LlamaConfig, LlamaForCausalLM

from jarvis_prime.tiny_prime.config import TinyPrimeConfig


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", default=None, help="Override output dir (defaults to training.output_dir)")
    args = ap.parse_args()

    cfg = TinyPrimeConfig.from_yaml(args.config)
    cfg = TinyPrimeConfig.from_env(base=cfg)

    tok_dir = cfg.resolve_path_for("tokenizer.output_dir")
    tokenizer = AutoTokenizer.from_pretrained(tok_dir, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or "</s>"

    model_cfg = LlamaConfig(
        vocab_size=int(cfg.get("model.vocab_size", 0) or len(tokenizer)),
        hidden_size=int(cfg.get("model.hidden_size", 768)),
        intermediate_size=int(cfg.get("model.intermediate_size", 3072)),
        num_hidden_layers=int(cfg.get("model.num_hidden_layers", 12)),
        num_attention_heads=int(cfg.get("model.num_attention_heads", 12)),
        max_position_embeddings=int(cfg.get("model.context_window", 512)),
        rms_norm_eps=float(cfg.get("model.rms_norm_eps", 1e-5)),
        rope_theta=float(cfg.get("model.rope_theta", 10000)),
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )

    model = LlamaForCausalLM(model_cfg)

    out_dir = args.out or cfg.resolve_path_for("training.output_dir")
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    model.save_pretrained(str(out))
    tokenizer.save_pretrained(str(out / "tokenizer"))

    print(f"✅ Initialized random Tiny Prime model at {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
