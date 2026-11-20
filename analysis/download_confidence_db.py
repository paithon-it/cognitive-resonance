"""Download the Confidence Database into data/confidence_db/ (CC0; not copied into this repository).

Rahnev, Desender, Lee and colleagues (2020), The Confidence Database, Nature Human Behaviour 4, 317-325,
https://osf.io/s46pr/. Only the files named data_*.csv of the folder "Confidence Database" are taken. A manifest
with size and SHA-256 of every file is written next to them.

    python analysis/download_confidence_db.py [--dest data/confidence_db]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

NODE = "https://api.osf.io/v2/nodes/s46pr/files/osfstorage/"
FOLDER = "Confidence Database"
ROOT = Path(__file__).resolve().parents[1]
NAME = re.compile(r"data_[A-Za-z0-9_.\-]+\.csv")
AGENT = {"User-Agent": "cognitive-resonance-download"}


def pages(url: str):
    while url:
        with urllib.request.urlopen(urllib.request.Request(url, headers=AGENT), timeout=120) as r:
            page = json.load(r)
        yield from page["data"]
        url = page["links"].get("next")


def listing() -> list[dict]:
    folder = next(it for it in pages(NODE) if it["attributes"]["name"] == FOLDER)
    url = folder["relationships"]["files"]["links"]["related"]["href"] + "?page%5Bsize%5D=100"
    out = []
    for it in pages(url):
        a = it["attributes"]
        if a["kind"] == "file" and NAME.fullmatch(a["name"]):
            out.append({"name": a["name"], "size": a["size"], "url": it["links"]["download"]})
    return sorted(out, key=lambda f: f["name"])


def fetch(dest: Path, f: dict) -> dict:
    target = dest / f["name"]
    if not (target.exists() and target.stat().st_size == f["size"]):
        tmp = dest / (f["name"] + ".part")
        with urllib.request.urlopen(urllib.request.Request(f["url"], headers=AGENT), timeout=600) as r:
            tmp.write_bytes(r.read())
        if tmp.stat().st_size != f["size"]:
            tmp.unlink()
            raise SystemExit(f"{f['name']}: size differs from the listing")
        tmp.replace(target)
    return {"name": f["name"], "size": f["size"], "sha256": hashlib.sha256(target.read_bytes()).hexdigest()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dest", default=str(ROOT / "data" / "confidence_db"))
    dest = Path(ap.parse_args().dest)
    dest.mkdir(parents=True, exist_ok=True)
    files = listing()
    with ThreadPoolExecutor(6) as pool:
        manifest = list(pool.map(lambda f: fetch(dest, f), files))
    (dest / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
    print(f"{len(manifest)} files, {sum(m['size'] for m in manifest) / 1e6:.0f} MB in {dest}")


if __name__ == "__main__":
    main()
