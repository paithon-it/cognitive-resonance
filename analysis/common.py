"""Helpers shared by the analysis scripts: clean JSON, a small process pool, medians."""
from __future__ import annotations

import json
import math
import os
import platform
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import scipy

ROOT = Path(__file__).resolve().parents[1]
MAX_WORKERS = 3


def clean(x):
    """Numpy types to Python, non-finite floats to None."""
    if isinstance(x, dict):
        return {str(k): clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [clean(v) for v in x]
    if isinstance(x, np.ndarray):
        return [clean(v) for v in x.tolist()]
    if isinstance(x, (np.bool_, bool)):
        return bool(x)
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating, float)):
        x = float(x)
        return x if math.isfinite(x) else None
    return x


def write_json(path, obj) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(clean(obj), indent=1, ensure_ascii=False) + "\n")


def environment() -> dict:
    return {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__,
            "code_commit": git_commit(ROOT)}


def git_commit(path) -> str | None:
    try:
        return subprocess.run(["git", "-C", str(path), "rev-parse", "HEAD"], capture_output=True,
                              text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def run_pool(fn, jobs, workers: int = MAX_WORKERS):
    """Map in at most `workers` processes, lowering their priority; order is kept."""
    jobs = list(jobs)
    if workers <= 1:
        return [fn(j) for j in jobs]
    with ProcessPoolExecutor(max_workers=min(workers, MAX_WORKERS), initializer=_nice) as ex:
        return list(ex.map(fn, jobs, chunksize=max(1, len(jobs) // (8 * workers))))


def _nice():
    try:
        os.nice(10)
    except OSError:
        pass


def med(values):
    v = np.asarray([x for x in values if x is not None], float)
    v = v[np.isfinite(v)]
    return float(np.median(v)) if v.size else None


class Timer:
    def __init__(self):
        self.t0 = time.time()

    def minutes(self) -> float:
        return round((time.time() - self.t0) / 60.0, 2)


def verdict(passed, **numbers) -> dict:
    """An outcome with the numbers that decide it."""
    if passed is None:
        return {"outcome": "not evaluable", **numbers}
    return {"outcome": "pass" if passed else "fail", **numbers}
