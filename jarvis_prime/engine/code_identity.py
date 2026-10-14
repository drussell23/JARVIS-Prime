"""What code this engine process is running -- so a launcher can tell.

JARVIS's boot adopts a J-Prime that already answers instead of starting
another. After a merge or pull in this checkout, that meant the engine
started before the change kept serving, and the change never ran: a fix to
model residency merged at noon was not live in an engine started at 01:32.

``/api/version`` therefore carries the engine's code identity: the checkout
it runs from, when its process started, and the checkout's Python sources it
has loaded. A launcher compares those files' modification times with the
start time; one modified after the start (or gone) means the engine runs code
the checkout no longer has.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

#: The checkout this package runs from (jarvis_prime/engine/ -> repo root).
REPO_ROOT = Path(__file__).resolve().parents[2]


def _process_started_at() -> Optional[float]:
    try:
        import psutil
        return psutil.Process(os.getpid()).create_time()
    except Exception:  # noqa: BLE001 -- psutil absent: the launcher asks the OS itself
        return None


def loaded_sources(root: Path = REPO_ROOT) -> List[str]:
    """Repo-relative paths of the checkout's Python sources this process loaded."""
    seen = set()
    for module in list(sys.modules.values()):
        path = getattr(module, "__file__", None)
        if not path or not path.endswith(".py"):
            continue
        try:
            seen.add(Path(path).resolve().relative_to(root).as_posix())
        except (ValueError, OSError):
            continue
    return sorted(seen)


def code_identity(root: Path = REPO_ROOT) -> Dict[str, Any]:
    return {
        "root": str(root),
        "pid": os.getpid(),
        "started_at": _process_started_at(),
        "sources": loaded_sources(root),
    }


__all__ = ["REPO_ROOT", "code_identity", "loaded_sources"]
