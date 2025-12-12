"""Evaluate Tiny Prime guard accuracy (overall + negation slice)."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, Optional, Tuple

from jarvis_prime.tiny_prime.config import TinyPrimeConfig
from jarvis_prime.tiny_prime.semantic_guard import TinyPrimeGuard


_INTENT_RE = re.compile(r"<intent>\s*(\[[A-Z_]+\])\s*</intent>")


def _iter_records(path: str) -> Iterator[Dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            if isinstance(obj, dict) and isinstance(obj.get("text"), str):
                yield obj


def _extract_label(rec: Dict[str, Any]) -> Optional[str]:
    lab = rec.get("label")
    if isinstance(lab, str):
        return lab
    m = _INTENT_RE.search(rec.get("text", ""))
    if m:
        return m.group(1)
    return None


def _extract_user_text(text: str) -> str:
    # Best-effort: strip tags if present
    text = text.replace("<user>", "").replace("</user>", "")
    text = _INTENT_RE.sub("", text)
    text = text.replace("</intent>", "")
    return " ".join(text.split()).strip()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--dataset", default=None)
    ap.add_argument("--limit", type=int, default=2000)
    args = ap.parse_args()

    cfg = TinyPrimeConfig.from_yaml(args.config)
    cfg = TinyPrimeConfig.from_env(base=cfg)

    ds_path = args.dataset or cfg.get("training.dataset_path")
    if not ds_path:
        raise ValueError("No dataset path provided")
    ds_path = cfg.resolve_path(str(ds_path))

    guard = TinyPrimeGuard.from_config(cfg)

    total = 0
    correct = 0

    neg_total = 0
    neg_correct = 0

    conf_sum = 0.0

    per_label = Counter()
    per_label_correct = Counter()

    for rec in _iter_records(ds_path):
        if total >= args.limit:
            break
        gold = _extract_label(rec)
        if gold is None:
            continue

        text = _extract_user_text(rec["text"])
        pred = guard.check(text)

        total += 1
        conf_sum += float(pred.confidence)

        per_label[gold] += 1
        if pred.label == gold:
            correct += 1
            per_label_correct[gold] += 1

        tags = rec.get("meta", {}).get("tags", [])
        if isinstance(tags, list) and "negation" in tags:
            neg_total += 1
            if pred.label == gold:
                neg_correct += 1

    acc = correct / total if total else 0.0
    neg_acc = neg_correct / neg_total if neg_total else 0.0
    avg_conf = conf_sum / total if total else 0.0

    print(f"examples={total}")
    print(f"accuracy={acc:.4f}")
    print(f"avg_confidence={avg_conf:.4f}")
    if neg_total:
        print(f"negation_examples={neg_total}")
        print(f"negation_accuracy={neg_acc:.4f}")

    print("\nper_label:")
    for k in sorted(per_label.keys()):
        denom = per_label[k]
        num = per_label_correct[k]
        print(f"  {k}: {num}/{denom} = {(num/denom if denom else 0.0):.4f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
