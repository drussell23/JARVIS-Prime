"""Export Tiny Prime for fast inference.

Supports:
- GGUF export (llama.cpp) if you have a local clone of llama.cpp

Example:
  python -m jarvis_prime.tiny_prime.scripts.export_tiny_prime \
    --config jarvis_prime/tiny_prime/config/model_config.yaml \
    --format gguf \
    --llama-cpp-dir ~/src/llama.cpp \
    --out ./artifacts/tiny_prime/tiny_prime.gguf
"""

from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path

from jarvis_prime.tiny_prime.config import TinyPrimeConfig


def _run(cmd: list[str]) -> None:
    print("$ " + " ".join(cmd))
    subprocess.run(cmd, check=True)


def export_gguf(*, model_dir: str, out_path: str, llama_cpp_dir: str, outtype: str) -> None:
    llama_cpp_dir_p = Path(llama_cpp_dir).expanduser().resolve()
    converter = llama_cpp_dir_p / "convert_hf_to_gguf.py"
    if not converter.exists():
        raise FileNotFoundError(f"Could not find convert_hf_to_gguf.py in {llama_cpp_dir_p}")

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "python3",
        str(converter),
        str(Path(model_dir).resolve()),
        "--outfile",
        str(Path(out_path).resolve()),
        "--outtype",
        outtype,
    ]
    _run(cmd)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--format", choices=["gguf"], required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--llama-cpp-dir", default=os.environ.get("LLAMA_CPP_DIR"))
    ap.add_argument("--outtype", default="q8_0", help="GGUF outtype (e.g. f16, q8_0, q4_0)")
    args = ap.parse_args()

    cfg = TinyPrimeConfig.from_yaml(args.config)
    cfg = TinyPrimeConfig.from_env(base=cfg)

    model_dir = cfg.resolve_path_for("training.output_dir")

    if args.format == "gguf":
        if not args.llama_cpp_dir:
            raise ValueError("--llama-cpp-dir is required (or set LLAMA_CPP_DIR)")
        export_gguf(model_dir=model_dir, out_path=args.out, llama_cpp_dir=args.llama_cpp_dir, outtype=args.outtype)

    print(f"✅ Export complete: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
