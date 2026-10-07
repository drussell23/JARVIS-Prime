"""GPU memory probe -- measured, never assumed.

Admission and ``/api/ps`` both need real numbers: how much VRAM is free now,
and how much a model actually took when it loaded. A hardcoded card profile
is exactly the "confidently wrong number" class that once sized this 5090 as
a 24 GiB L4. ``nvidia-smi`` is the instrument; when it is absent the probe
says so (``None``) and callers fall back to file-size estimates explicitly.
"""
from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class GpuMemory:
    name: str
    total_mib: int
    used_mib: int

    @property
    def free_mib(self) -> int:
        return max(self.total_mib - self.used_mib, 0)


def query(index: int = 0, timeout_s: float = 5.0) -> Optional[GpuMemory]:
    exe = shutil.which("nvidia-smi")
    if not exe:
        return None
    try:
        out = subprocess.run(
            [exe, f"--id={index}", "--query-gpu=name,memory.total,memory.used",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=timeout_s, check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if out.returncode != 0 or not out.stdout.strip():
        return None
    try:
        name, total, used = [p.strip() for p in out.stdout.strip().splitlines()[0].split(",")]
        return GpuMemory(name=name, total_mib=int(float(total)), used_mib=int(float(used)))
    except ValueError:
        return None
