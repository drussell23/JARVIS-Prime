"""``python -m jarvis_prime.engine`` -- run the J-Prime Engine.

    python -m jarvis_prime.engine --port 8000 --preload qwen3-coder-ov:30b

Environment (all optional):
  JPRIME_ENGINE_HOST / JPRIME_ENGINE_PORT     bind address (127.0.0.1:8000)
  JPRIME_ENGINE_PRELOAD                       model to load at startup, resident until evicted
  JPRIME_ENGINE_DEFAULT_MODEL                 model for requests naming none / "jarvis-prime"
  JPRIME_LLAMA_SERVER_BIN / JPRIME_ENGINE_HOME   engine binary, or where builds live
  JPRIME_ENGINE_MODELS                        YAML of declared models (else the Ollama store)
  JPRIME_ENGINE_DEFAULT_CTX / _MAX_CTX        context when a request names none / ceiling
  JPRIME_ENGINE_KEEP_ALIVE                    default residency after last request (5m)
"""
from __future__ import annotations

import argparse
import logging
import os


def main() -> None:
    ap = argparse.ArgumentParser(prog="jarvis_prime.engine", description="J-Prime Engine")
    ap.add_argument("--host", default=os.environ.get("JPRIME_ENGINE_HOST", "127.0.0.1"))
    ap.add_argument("--port", type=int, default=int(os.environ.get("JPRIME_ENGINE_PORT", "8000")))
    ap.add_argument("--preload", default=None, help="model to load at startup (resident until evicted)")
    ap.add_argument("--default-model", default=None)
    ap.add_argument("--ctx", type=int, default=None, help="context for a model loaded with no num_ctx")
    ap.add_argument("--log-level", default=os.environ.get("JPRIME_ENGINE_LOG_LEVEL", "info"))
    ap.add_argument("--log-file", default=os.environ.get("JPRIME_ENGINE_LOG_FILE", ""),
                    help="write logs here instead of stderr (a detached engine has no console)")
    args = ap.parse_args()

    if args.ctx:
        os.environ["JPRIME_ENGINE_DEFAULT_CTX"] = str(args.ctx)
    if args.preload:
        os.environ["JPRIME_ENGINE_PRELOAD"] = args.preload
    if args.default_model:
        os.environ["JPRIME_ENGINE_DEFAULT_MODEL"] = args.default_model
    elif args.preload and not os.environ.get("JPRIME_ENGINE_DEFAULT_MODEL"):
        os.environ["JPRIME_ENGINE_DEFAULT_MODEL"] = args.preload

    logging.basicConfig(level=args.log_level.upper(),
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
                        **({"filename": args.log_file} if args.log_file else {}))
    # httpx logs every upstream request at INFO -- including the 0.5 s health
    # polls during a load. The engine's own lines carry what matters.
    logging.getLogger("httpx").setLevel(logging.WARNING)
    import uvicorn

    from .app import create_app

    # log_config=None: uvicorn keeps the root handler configured above instead
    # of installing its own stderr handlers (which a detached engine lacks).
    uvicorn.run(create_app(), host=args.host, port=args.port, log_level=args.log_level.lower(),
                access_log=False, log_config=None)


if __name__ == "__main__":
    main()
