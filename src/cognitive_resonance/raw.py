"""Readers of the raw files of § 11. They check the contract and refuse what does not fit.

Files are loaded with `allow_pickle=False`: a raw file holds arrays, never objects.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np

BENCH_RE = re.compile(r"^(?P<task>.+)_mu(?P<mu>[0-9][0-9.p]*)_s(?P<seed>\d+)\.npz$")
REPHASE_RE = re.compile(r"^(?P<task>.+)_s(?P<seed>\d+)\.npz$")
FOLD_RE = re.compile(r"^fold(?P<fold>\d+)_s(?P<seed>\d+)\.npz$")
BASE_RE = re.compile(r"^base_s(?P<seed>\d+)\.npz$")

BENCH_KEYS = ("y", "image", "a", "z_plus", "z_minus", "z_plus2")
REPHASE_KEYS = ("y", "image", "a", "z_H", "z_H2", "z_M")


class ContractError(ValueError):
    """A raw file that does not follow § 11."""


def _load(path) -> dict:
    with np.load(path, allow_pickle=False) as f:
        return {k: f[k] for k in f.files}


def _need(d: dict, keys, path):
    missing = [k for k in keys if k not in d]
    if missing:
        raise ContractError(f"{Path(path).name}: missing {missing}")


def _same_length(d: dict, keys, path, axis_of=None):
    axis_of = axis_of or {}
    n = {k: d[k].shape[axis_of.get(k, 0)] for k in keys}
    if len(set(n.values())) != 1:
        raise ContractError(f"{Path(path).name}: lengths differ {n}")
    return next(iter(n.values()))


def _binary(y, path):
    if not np.isin(np.unique(y), [0, 1]).all():
        raise ContractError(f"{Path(path).name}: y is not 0/1")


def _finite(d: dict, keys, path):
    for k in keys:
        if not np.all(np.isfinite(np.asarray(d[k], float))):
            raise ContractError(f"{Path(path).name}: {k} has non-finite values")


def _mu(text: str) -> float:
    return float(text.replace("p", "."))


def read_bench(path) -> dict:
    """bench/{task}_mu{mu}_s{seed}.npz, test and validation."""
    path = Path(path)
    m = BENCH_RE.match(path.name)
    if not m:
        raise ContractError(f"{path.name}: name does not match {{task}}_mu{{mu}}_s{{seed}}.npz")
    d = _load(path)
    _need(d, BENCH_KEYS + tuple("val_" + k for k in BENCH_KEYS), path)
    for prefix in ("", "val_"):
        keys = tuple(prefix + k for k in BENCH_KEYS)
        _same_length(d, keys, path)
        _binary(d[prefix + "y"], path)
        _finite(d, keys, path)
    d.update(task=m["task"], mu=_mu(m["mu"]), seed=int(m["seed"]), file=path.name)
    return d


def read_rephase(path) -> dict:
    """rephase/{task}_s{seed}.npz: z_M has one row per lambda."""
    path = Path(path)
    m = REPHASE_RE.match(path.name)
    if not m:
        raise ContractError(f"{path.name}: name does not match {{task}}_s{{seed}}.npz")
    d = _load(path)
    _need(d, REPHASE_KEYS + ("lambdas",) + tuple("val_" + k for k in REPHASE_KEYS), path)
    for prefix in ("", "val_"):
        keys = tuple(prefix + k for k in REPHASE_KEYS)
        if d[prefix + "z_M"].ndim != 2 or d[prefix + "z_M"].shape[0] != d["lambdas"].size:
            raise ContractError(f"{path.name}: {prefix}z_M must be (number of lambdas, N)")
        _same_length(d, keys, path, axis_of={prefix + "z_M": 1})
        _binary(d[prefix + "y"], path)
        _finite(d, keys, path)
    d.update(task=m["task"], seed=int(m["seed"]), file=path.name)
    return d


REPHASE_BIS_KEYS = ("y", "image", "a", "z_H", "z_M")
REPHASE_BIS_MACHINES = ("M0", "Mpair", "Mframe")


def read_rephase_bis(path) -> dict:
    """rephase_bis/{task}_s{seed}.npz (§ 6 bis): z_M has one row per machine, named in `machines`."""
    path = Path(path)
    m = REPHASE_RE.match(path.name)
    if not m:
        raise ContractError(f"{path.name}: name does not match {{task}}_s{{seed}}.npz")
    d = _load(path)
    _need(d, REPHASE_BIS_KEYS + ("machines",) + tuple("val_" + k for k in REPHASE_BIS_KEYS), path)
    names = [str(x) for x in d["machines"]]
    if sorted(names) != sorted(REPHASE_BIS_MACHINES):
        raise ContractError(f"{path.name}: machines must be {list(REPHASE_BIS_MACHINES)}, got {names}")
    for prefix in ("", "val_"):
        keys = tuple(prefix + k for k in REPHASE_BIS_KEYS)
        if d[prefix + "z_M"].ndim != 2 or d[prefix + "z_M"].shape[0] != len(names):
            raise ContractError(f"{path.name}: {prefix}z_M must be (number of machines, N)")
        _same_length(d, keys, path, axis_of={prefix + "z_M": 1})
        _binary(d[prefix + "y"], path)
        _finite(d, keys, path)
    d.update(task=m["task"], seed=int(m["seed"]), file=path.name, machine_names=names)
    return d


def read_people_fold(path) -> dict:
    """people/fold{f}_s{seed}.npz: logits (number of lambdas, N, 10) on the unseen half."""
    path = Path(path)
    m = FOLD_RE.match(path.name)
    if not m:
        raise ContractError(f"{path.name}: name does not match fold{{f}}_s{{seed}}.npz")
    d = _load(path)
    _need(d, ("image", "y", "logits", "lambdas"), path)
    lg = d["logits"]
    if lg.ndim != 3 or lg.shape[0] != d["lambdas"].size or lg.shape[2] != 10:
        raise ContractError(f"{path.name}: logits must be (number of lambdas, N, 10)")
    if lg.shape[1] != d["image"].size or d["y"].size != d["image"].size:
        raise ContractError(f"{path.name}: lengths differ")
    if d["y"].min() < 0 or d["y"].max() > 9:
        raise ContractError(f"{path.name}: y outside 0-9")
    _finite(d, ("logits",), path)
    d.update(fold=int(m["fold"]), seed=int(m["seed"]), file=path.name)
    return d


def read_people_base(path) -> dict:
    """people/base_s{seed}.npz: logits of the base machine on the 10,000 test images."""
    path = Path(path)
    m = BASE_RE.match(path.name)
    if not m:
        raise ContractError(f"{path.name}: name does not match base_s{{seed}}.npz")
    d = _load(path)
    _need(d, ("logits",), path)
    if d["logits"].ndim != 2 or d["logits"].shape[1] != 10:
        raise ContractError(f"{path.name}: logits must be (N, 10)")
    _finite(d, ("logits",), path)
    if "image" not in d:
        d["image"] = np.arange(d["logits"].shape[0])
    d.update(seed=int(m["seed"]), file=path.name)
    return d


def list_units(folder, pattern: re.Pattern) -> list[Path]:
    folder = Path(folder)
    if not folder.is_dir():
        return []
    return sorted(p for p in folder.iterdir() if p.is_file() and pattern.match(p.name))


def read_manifest(folder) -> dict | None:
    p = Path(folder) / "manifest.json"
    if not p.is_file():
        return None
    return json.loads(p.read_text())
