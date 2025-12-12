"""Generate `security_intents.jsonl` (async, parallel, config-driven)."""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from jarvis_prime.tiny_prime.config import TinyPrimeConfig


def _weighted_counts(total: int, weights: Dict[str, float]) -> Dict[str, int]:
    keys = [k for k, w in weights.items() if w > 0]
    if not keys:
        return {}
    wsum = sum(weights[k] for k in keys)
    normalized = {k: weights[k] / wsum for k in keys}

    # Allocate by largest remainder
    raw = {k: total * normalized[k] for k in keys}
    base = {k: int(math.floor(raw[k])) for k in keys}
    rem = total - sum(base.values())

    remainders = sorted(keys, key=lambda k: (raw[k] - base[k]), reverse=True)
    for k in remainders[:rem]:
        base[k] += 1
    return base


def _maybe_apply_noise(text: str, rng: random.Random, noise: Dict[str, Any]) -> str:
    out = text

    if bool(noise.get("fillers", False)) and rng.random() < 0.25:
        filler = rng.choice(["please", "now", "right now", "quick", "kindly"])
        out = f"{out} {filler}"

    if bool(noise.get("hesitations", False)) and rng.random() < 0.25:
        hes = rng.choice(["uh", "um", "wait", "hold on", "sorry", "okay"])
        out = f"{hes}, {out}"

    if bool(noise.get("punctuation", False)) and rng.random() < 0.2:
        out = out + rng.choice([".", "!", "…", "!!"])

    if bool(noise.get("casing", False)) and rng.random() < 0.2:
        style = rng.choice(["lower", "upper", "title"])
        if style == "lower":
            out = out.lower()
        elif style == "upper":
            out = out.upper()
        else:
            out = out.title()

    if bool(noise.get("asr_typos", False)) and rng.random() < 0.15:
        # Very small, realistic-ish ASR noise
        out = out.replace("don't", "dont").replace("do not", "donot")
        out = out.replace("unlock", rng.choice(["un lock", "unlok", "unlockk"]))
        out = out.replace("open", rng.choice(["opn", "oppen", "open"]))

    return out


@dataclass(frozen=True)
class _Example:
    text: str
    label: Optional[str]
    tags: List[str]
    domain: str

    def to_record(self) -> Dict[str, Any]:
        rec: Dict[str, Any] = {"text": self.text}
        meta: Dict[str, Any] = {"domain": self.domain, "tags": self.tags}
        if self.label is not None:
            rec["label"] = self.label
            meta["intent"] = self.label
        rec["meta"] = meta
        return rec


def _format_labeled(cfg: TinyPrimeConfig, user_text: str, label: str) -> str:
    fmt = cfg.format_tokens()
    return (
        f"{fmt['user_open']} {user_text} {fmt['user_close']} "
        f"{fmt['intent_open']} {label} {fmt['intent_close']}"
    )


def _generate_one_synthetic(cfg: TinyPrimeConfig, rng: random.Random, syn: Dict[str, Any]) -> _Example:
    intents = cfg.intent_labels()
    duress_label = "[DURESS]" if "[DURESS]" in intents else intents[-1]
    amb_label = cfg.get("tiny_prime.intents.default_label", "[AMBIGUOUS]")

    hard = syn.get("hard_cases", {})
    noise = syn.get("noise", {})

    neg_rate = float(hard.get("negation_rate", 0.35))
    dbl_rate = float(hard.get("double_negation_rate", 0.05))
    duress_rate = float(hard.get("duress_rate", 0.08))
    amb_rate = float(hard.get("ambiguity_rate", 0.12))

    domains = syn.get("domains") or []
    if not domains:
        domains = [
            {
                "name": "default",
                "entities": ["door"],
                "allow_actions": ["unlock"],
                "deny_actions": ["lock"],
            }
        ]

    dom = rng.choice(domains)
    domain_name = str(dom.get("name", "default"))
    entities = list(dom.get("entities") or ["door"])
    actions = list(dom.get("allow_actions") or []) + list(dom.get("deny_actions") or [])
    if not actions:
        actions = ["unlock", "open", "lock", "close"]

    entity = rng.choice(entities)
    action = rng.choice(actions)

    tags: List[str] = []

    # Decide case
    roll = rng.random()

    if roll < duress_rate:
        label = duress_label
        tags.append("duress")
        duress_phrase = rng.choice(
            [
                "they are making me",
                "someone is forcing me",
                "help me",
                "I'm being threatened",
                "I don't want to but",
            ]
        )
        user = f"{duress_phrase}, {action} the {entity}"
        user = _maybe_apply_noise(user, rng, noise)
        return _Example(text=_format_labeled(cfg, user, label), label=label, tags=tags, domain=domain_name)

    if roll < duress_rate + amb_rate:
        label = amb_label
        tags.append("ambiguous")
        hedge = rng.choice(["maybe", "I think", "not sure but", "can you", "should we"])
        user = f"{hedge} {action} the {entity}"
        user = _maybe_apply_noise(user, rng, noise)
        return _Example(text=_format_labeled(cfg, user, label), label=label, tags=tags, domain=domain_name)

    # Double negation creates tricky ALLOW cases.
    if rng.random() < dbl_rate:
        tags.append("double_negation")
        label = "[ALLOW]" if "[ALLOW]" in intents else intents[0]
        user = rng.choice(
            [
                f"don't not {action} the {entity}",
                f"do not not {action} the {entity}",
                f"don't ever not {action} the {entity}",
            ]
        )
        user = _maybe_apply_noise(user, rng, noise)
        return _Example(text=_format_labeled(cfg, user, label), label=label, tags=tags, domain=domain_name)

    # Negation
    if rng.random() < neg_rate:
        tags.append("negation")
        label = "[DENY]" if "[DENY]" in intents else intents[0]
        user = rng.choice(
            [
                f"don't {action} the {entity}",
                f"do not {action} the {entity}",
                f"wait, don't {action} the {entity}",
                f"stop, do not {action} the {entity}",
                f"never {action} the {entity}",
            ]
        )
        user = _maybe_apply_noise(user, rng, noise)
        return _Example(text=_format_labeled(cfg, user, label), label=label, tags=tags, domain=domain_name)

    # Default ALLOW
    label = "[ALLOW]" if "[ALLOW]" in intents else intents[0]
    user = rng.choice(
        [
            f"{action} the {entity}",
            f"please {action} the {entity}",
            f"{action} {entity}",
        ]
    )
    user = _maybe_apply_noise(user, rng, noise)
    return _Example(text=_format_labeled(cfg, user, label), label=label, tags=tags, domain=domain_name)


def _generate_synthetic_shard(cfg: TinyPrimeConfig, n: int, seed: int, syn: Dict[str, Any]) -> List[Dict[str, Any]]:
    rng = random.Random(seed)
    return [_generate_one_synthetic(cfg, rng, syn).to_record() for _ in range(n)]


async def _generate_synthetic(cfg: TinyPrimeConfig, n: int) -> List[Dict[str, Any]]:
    syn = cfg.get("data.sources.synthetic", {})
    seed = int(cfg.get("data.seed", 42))

    # Shard for parallelism
    shards = max(1, min(32, (os.cpu_count() or 4)))
    shard_size = max(1, n // shards)

    tasks = []
    remaining = n
    for i in range(shards):
        take = shard_size if i < shards - 1 else remaining
        remaining -= take
        if take <= 0:
            continue
        tasks.append(
            asyncio.to_thread(_generate_synthetic_shard, cfg, take, seed + i * 9973, syn)
        )

    out: List[Dict[str, Any]] = []
    for chunk in await asyncio.gather(*tasks):
        out.extend(chunk)

    random.Random(seed).shuffle(out)
    return out


async def _generate_tinystories(cfg: TinyPrimeConfig, n: int) -> List[Dict[str, Any]]:
    ts = cfg.get("data.sources.tinystories", {})
    if not bool(ts.get("enabled", False)) or n <= 0:
        return []

    try:
        from datasets import load_dataset
    except Exception:
        return []

    dataset_name = str(ts.get("dataset", "roneneldan/TinyStories"))
    split = str(ts.get("split", "train"))
    streaming = bool(ts.get("streaming", True))

    ds = load_dataset(dataset_name, split=split, streaming=streaming)

    # Pick a likely text column dynamically.
    text_col = None
    if hasattr(ds, "features"):
        for k, v in ds.features.items():
            if getattr(v, "dtype", None) == "string":
                text_col = k
                break
    text_col = text_col or "text"

    out: List[Dict[str, Any]] = []
    i = 0
    for row in ds:
        if i >= n:
            break
        text = row.get(text_col)
        if isinstance(text, str) and text.strip():
            out.append({"text": text.strip(), "meta": {"source": "tinystories"}})
            i += 1

    return out


async def generate_dataset(cfg: TinyPrimeConfig, *, num_examples: Optional[int] = None) -> List[Dict[str, Any]]:
    total = int(num_examples if num_examples is not None else cfg.get("data.num_examples", 20000))

    sources = cfg.get("data.sources", {})
    weights: Dict[str, float] = {}
    for name, spec in (sources or {}).items():
        if not isinstance(spec, dict) or not bool(spec.get("enabled", False)):
            continue
        weights[name] = float(spec.get("weight", 1.0))

    counts = _weighted_counts(total, weights)

    syn_n = int(counts.get("synthetic", 0))
    ts_n = int(counts.get("tinystories", 0))

    synthetic_task = _generate_synthetic(cfg, syn_n) if syn_n else asyncio.sleep(0, result=[])
    tinystories_task = _generate_tinystories(cfg, ts_n) if ts_n else asyncio.sleep(0, result=[])

    synthetic, tinystories = await asyncio.gather(synthetic_task, tinystories_task)

    out = list(synthetic) + list(tinystories)
    rng = random.Random(int(cfg.get("data.seed", 42)))
    rng.shuffle(out)

    return out


async def main_async() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="Path to Tiny Prime YAML config")
    ap.add_argument("--output", default=None, help="Override output JSONL path")
    ap.add_argument("--num-examples", type=int, default=None)
    args = ap.parse_args()

    cfg = TinyPrimeConfig.from_yaml(args.config)
    cfg = TinyPrimeConfig.from_env(base=cfg)

    out_path = args.output or cfg.get("data.output_path", "./data/security_intents.jsonl")
    out_path = cfg.resolve_path(str(out_path))

    records = await generate_dataset(cfg, num_examples=args.num_examples)

    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"✅ Wrote {len(records)} examples to {out_path}")
    return 0


def main() -> int:
    return asyncio.run(main_async())


if __name__ == "__main__":
    raise SystemExit(main())
