"""GGUF metadata reader -- the model's own physics, read from its header.

O+V sizes its context window from ``<arch>.context_length``,
``<arch>.block_count``, ``<arch>.attention.head_count_kv`` and the key/value
lengths (``/api/show`` -> ``model_info``). Those numbers live in the GGUF
header; reading them from the file is the one source that cannot drift from
the weights actually being served.

Only the key/value section is read -- never tensor data -- and array values
are skipped by seeking (a tokenizer vocabulary is ~150k strings). Results are
cached per (path, mtime). Never raises: an unreadable file yields ``{}``.
"""
from __future__ import annotations

import logging
import struct
from pathlib import Path
from typing import Any, BinaryIO, Dict, Tuple

logger = logging.getLogger(__name__)

_MAGIC = b"GGUF"

# GGUF value types -> struct format (scalars only).
_SCALAR = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f", 7: "<?",
           10: "<Q", 11: "<q", 12: "<d"}
_STRING, _ARRAY = 8, 9

_cache: Dict[Tuple[str, float], Dict[str, Any]] = {}


def _read(f: BinaryIO, fmt: str) -> Any:
    size = struct.calcsize(fmt)
    data = f.read(size)
    if len(data) != size:
        raise EOFError("truncated GGUF header")
    return struct.unpack(fmt, data)[0]


def _read_str(f: BinaryIO) -> str:
    n = _read(f, "<Q")
    return f.read(n).decode("utf-8", errors="replace")


def _skip_value(f: BinaryIO, vtype: int) -> None:
    if vtype in _SCALAR:
        f.seek(struct.calcsize(_SCALAR[vtype]), 1)
    elif vtype == _STRING:
        f.seek(_read(f, "<Q"), 1)
    elif vtype == _ARRAY:
        itype, count = _read(f, "<I"), _read(f, "<Q")
        if itype in _SCALAR:
            f.seek(struct.calcsize(_SCALAR[itype]) * count, 1)
        else:
            for _ in range(count):
                _skip_value(f, itype)
    else:
        raise ValueError(f"unknown GGUF value type {vtype}")


def read_metadata(path: Path) -> Dict[str, Any]:
    """Scalar and string metadata of a GGUF file (arrays omitted)."""
    path = Path(path)
    try:
        key = (str(path), path.stat().st_mtime)
    except OSError:
        return {}
    if key in _cache:
        return _cache[key]
    meta: Dict[str, Any] = {}
    try:
        with path.open("rb") as f:
            if f.read(4) != _MAGIC:
                return {}
            version = _read(f, "<I")
            if version < 2:
                return {}
            _read(f, "<Q")  # tensor count
            kv_count = _read(f, "<Q")
            for _ in range(kv_count):
                k = _read_str(f)
                vtype = _read(f, "<I")
                if vtype in _SCALAR:
                    meta[k] = _read(f, _SCALAR[vtype])
                elif vtype == _STRING:
                    meta[k] = _read_str(f)
                else:
                    _skip_value(f, vtype)
    except (OSError, EOFError, ValueError, struct.error) as exc:
        logger.warning("[GGUF] could not read %s: %s", path, exc)
        return meta
    _cache[key] = meta
    return meta


def kv_bytes_per_token(meta: Dict[str, Any]) -> int:
    """f16 KV cache bytes per token of context, from the model's own geometry.

    0 when the geometry is not declared (the caller then estimates without
    it rather than inventing one).
    """
    arch = meta.get("general.architecture", "")
    blocks = int(meta.get(f"{arch}.block_count", 0) or 0)
    heads_kv = int(meta.get(f"{arch}.attention.head_count_kv", 0) or 0)
    k_len = int(meta.get(f"{arch}.attention.key_length", 0) or 0)
    v_len = int(meta.get(f"{arch}.attention.value_length", 0) or 0)
    if not (blocks and heads_kv and k_len and v_len):
        emb = int(meta.get(f"{arch}.embedding_length", 0) or 0)
        heads = int(meta.get(f"{arch}.attention.head_count", 0) or 0)
        if blocks and heads_kv and emb and heads:
            k_len = v_len = emb // heads
        else:
            return 0
    return blocks * heads_kv * (k_len + v_len) * 2
