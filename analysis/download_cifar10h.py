"""Download the CIFAR-10H vote counts into data/ (CC BY-NC-SA 4.0: not redistributed here).

Peterson, Battleday, Griffiths and Russakovsky (2019), https://github.com/jcpeterson/cifar-10h.

    python analysis/download_cifar10h.py [--dest data]
"""
from __future__ import annotations

import argparse
import hashlib
import urllib.request
from pathlib import Path

import numpy as np

URL = "https://raw.githubusercontent.com/jcpeterson/cifar-10h/master/data/cifar10h-counts.npy"
ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dest", default=str(ROOT / "data"))
    args = ap.parse_args()
    dest = Path(args.dest)
    dest.mkdir(parents=True, exist_ok=True)
    target = dest / "cifar10h-counts.npy"
    tmp = dest / "cifar10h-counts.npy.part"
    req = urllib.request.Request(URL, headers={"User-Agent": "cognitive-resonance-download"})
    with urllib.request.urlopen(req, timeout=120) as r:
        tmp.write_bytes(r.read())
    counts = np.load(tmp, allow_pickle=False)
    if counts.shape != (10_000, 10) or counts.min() < 0:
        tmp.unlink()
        raise SystemExit(f"unexpected content: shape {counts.shape}")
    tmp.replace(target)
    digest = hashlib.sha256(target.read_bytes()).hexdigest()
    print(f"{target.name}: {counts.shape}, {int(counts.sum())} votes, sha256 {digest}")


if __name__ == "__main__":
    main()
