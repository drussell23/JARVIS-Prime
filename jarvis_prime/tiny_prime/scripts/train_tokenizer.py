"""Train a BPE tokenizer (fast) for Tiny Prime."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.normalizers import NFKC
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import BpeTrainer
from transformers import PreTrainedTokenizerFast

from jarvis_prime.tiny_prime.config import TinyPrimeConfig


def _iter_jsonl_texts(path: str) -> Iterator[str]:
    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            text = obj.get("text")
            if isinstance(text, str) and text.strip():
                yield text


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--dataset", default=None, help="Override dataset path (JSONL)")
    ap.add_argument("--output-dir", default=None)
    args = ap.parse_args()

    cfg = TinyPrimeConfig.from_yaml(args.config)
    cfg = TinyPrimeConfig.from_env(base=cfg)

    ds_path = args.dataset or cfg.get("training.dataset_path") or cfg.get("data.output_path")
    if not ds_path:
        raise ValueError("No dataset path found (training.dataset_path or data.output_path)")
    ds_path = cfg.resolve_path(str(ds_path))

    out_dir = args.output_dir or cfg.get("tokenizer.output_dir", "./artifacts/tiny_prime/tokenizer")
    out_dir = cfg.resolve_path(str(out_dir))
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    vocab_size = int(cfg.get("tokenizer.vocab_size", 16000))
    min_freq = int(cfg.get("tokenizer.min_frequency", 2))
    special_tokens = cfg.special_tokens()

    tok = Tokenizer(BPE(unk_token="<unk>"))
    tok.normalizer = NFKC()
    tok.pre_tokenizer = Whitespace()

    trainer = BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=min_freq,
        special_tokens=special_tokens,
    )

    # Train from JSONL iterator (no hardcoded file format beyond `text` field)
    tok.train_from_iterator(_iter_jsonl_texts(ds_path), trainer=trainer)

    # Save raw tokenizer.json
    tok_json_path = Path(out_dir) / "tokenizer.json"
    tok.save(str(tok_json_path))

    # Wrap in transformers fast tokenizer for compatibility
    fast = PreTrainedTokenizerFast(tokenizer_object=tok)

    # Core tokens (if present in special_tokens, these are single ids)
    if "<pad>" in special_tokens:
        fast.pad_token = "<pad>"
    if "<unk>" in special_tokens:
        fast.unk_token = "<unk>"
    if "<s>" in special_tokens:
        fast.bos_token = "<s>"
    if "</s>" in special_tokens:
        fast.eos_token = "</s>"

    # Anything not core gets added as additional specials
    core = {fast.pad_token, fast.unk_token, fast.bos_token, fast.eos_token}
    core = {t for t in core if t}
    additional = [t for t in special_tokens if t not in core]
    fast.add_special_tokens({"additional_special_tokens": additional})

    fast.save_pretrained(out_dir)

    print(f"✅ Tokenizer saved to {out_dir}")
    print(f"   vocab_size={len(fast)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
