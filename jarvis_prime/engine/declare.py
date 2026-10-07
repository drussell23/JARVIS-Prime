"""Declare a model J-Prime serves by name -- verified, through the registry.

    python -m jarvis_prime.engine.declare --name jarvis-vision:8b \
        --model <dir>/Qwen3VL-8B-Instruct-Q4_K_M.gguf \
        --projector <dir>/mmproj-Qwen3VL-8B-Instruct-F16.gguf \
        --sha256 <model-digest> --sha256 <projector-digest> --inherit-defaults

``--inherit-defaults`` keeps the request defaults the store ALREADY resolves
for that name (e.g. an Ollama manifest's params layer), so moving a model to
llama.cpp-format files does not silently change how it is sampled. Takes
effect on the next load: the store re-reads the declared file.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from .adapter_registry import AdapterRegistry, AdapterRejected
from .model_store import ModelStore


def _value(raw: str) -> Any:
    try:
        return json.loads(raw)
    except ValueError:
        return raw


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="jarvis_prime.engine.declare", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", required=True)
    ap.add_argument("--model", required=True, type=Path)
    ap.add_argument("--projector", type=Path)
    ap.add_argument("--sha256", action="append", default=[],
                    help="expected digest, in the order the files were given (model, projector)")
    ap.add_argument("--default", action="append", default=[], metavar="KEY=VALUE",
                    help="a request default (JSON value), e.g. temperature=0.1")
    ap.add_argument("--inherit-defaults", action="store_true",
                    help="start from the defaults the store already resolves for --name")
    args = ap.parse_args(argv)
    store = ModelStore()
    defaults: Dict[str, Any] = {}
    if args.inherit_defaults:
        prior = store.resolve(args.name)
        defaults.update(dict(prior.defaults) if prior is not None else {})
    for kv in args.default:
        k, _, v = kv.partition("=")
        defaults[k.strip()] = _value(v.strip())
    files = [str(args.model)] + ([str(args.projector)] if args.projector else [])
    if len(args.sha256) > len(files):
        ap.error("more --sha256 values than files")
    try:
        entry = AdapterRegistry(store).declare(
            name=args.name, model=args.model, projector=args.projector, defaults=defaults,
            sha256=dict(zip(files, args.sha256)))
    except AdapterRejected as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(entry, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
