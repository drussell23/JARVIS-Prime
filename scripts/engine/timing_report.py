"""Summarise llama-server timing lines into the figures BENCHMARKS.md publishes.

    python scripts/engine/timing_report.py --head-lines 3505 %LOCALAPPDATA%\\JARVIS\\jprime\\state\\logs\\qwen3-coder-ov_30b.log

Every request llama-server finishes logs a ``prompt eval time`` line (prefill:
only the tokens NOT served from the prompt cache) and an ``eval time`` line
(decode). This reads both, so a published number can always be regenerated
from the log it came from rather than transcribed by hand. Stdlib only.

The engine log is append-only and carries no wall-clock timestamps, so a
published snapshot is pinned by line count: ``--head-lines N`` reads only the
first N lines, the traffic that existed when the figures were taken.
"""
from __future__ import annotations

import argparse
import re
import sys
from typing import List, Optional, Sequence, Tuple

_PREFILL = re.compile(r"prompt eval time =\s+([\d.]+) ms /\s+(\d+) tokens.*?([\d.]+) tokens per second")
_DECODE = re.compile(r"\|\s+eval time =\s+([\d.]+) ms /\s+(\d+) tokens.*?([\d.]+) tokens per second")

#: Decodes shorter than this are dominated by per-request overhead, not throughput.
_MIN_DECODE_TOKENS = 32
_PREFILL_BUCKETS = ((1_000, 4_000), (4_000, 8_000), (8_000, 16_000), (16_000, 10**9))

Timing = Tuple[float, int, float]  # (ms, tokens, tokens/s)


def _quantile(values: Sequence[float], q: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(q * len(ordered)))]


def parse(path: str, head_lines: Optional[int] = None) -> Tuple[List[Timing], List[Timing]]:
    prefill: List[Timing] = []
    decode: List[Timing] = []
    with open(path, encoding="utf-8", errors="replace") as fh:
        for lineno, line in enumerate(fh, 1):
            if head_lines is not None and lineno > head_lines:
                break
            m = _PREFILL.search(line)
            if m:
                prefill.append((float(m[1]), int(m[2]), float(m[3])))
                continue
            m = _DECODE.search(line)
            if m:
                decode.append((float(m[1]), int(m[2]), float(m[3])))
    return prefill, decode


def report(path: str, head_lines: Optional[int] = None) -> str:
    prefill, decode = parse(path, head_lines)
    scope = f" (first {head_lines} lines)" if head_lines is not None else ""
    out = [f"{path}{scope}: {len(decode)} generations"]
    rates = [tps for _, n, tps in decode if n >= _MIN_DECODE_TOKENS]
    if rates:
        out.append(
            f"  decode (>= {_MIN_DECODE_TOKENS} tokens, n={len(rates)}): "
            f"p10 {_quantile(rates, .1):.0f}  p50 {_quantile(rates, .5):.0f}  "
            f"p90 {_quantile(rates, .9):.0f}  max {max(rates):.0f} tok/s")
    for lo, hi in _PREFILL_BUCKETS:
        bucket = [t for t in prefill if lo <= t[1] < hi]
        if bucket:
            out.append(
                f"  prefill {lo}-{hi} uncached tokens (n={len(bucket)}): "
                f"p50 {_quantile([t[0] for t in bucket], .5):.0f} ms, "
                f"p50 {_quantile([t[2] for t in bucket], .5):.0f} tok/s")
    return "\n".join(out)


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("logs", nargs="+", help="llama-server log files")
    ap.add_argument("--head-lines", type=int, default=None, metavar="N",
                    help="read only the first N lines of each log (pins a published snapshot)")
    args = ap.parse_args(argv)
    if args.head_lines is not None and args.head_lines < 1:
        ap.error("--head-lines must be >= 1")
    for path in args.logs:
        print(report(path, args.head_lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
