"""Runtime CUDA auto-detect so training defaults to GPU when present, CPU otherwise."""
from __future__ import annotations

import functools
import shutil
import subprocess


@functools.lru_cache(maxsize=1)
def has_gpu() -> bool:
    if not shutil.which("nvidia-smi"):
        return False
    try:
        subprocess.run(
            ["nvidia-smi"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            check=True, timeout=5,
        )
        return True
    except Exception:
        return False
