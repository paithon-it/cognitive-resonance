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
REPHASE_TER_MACHINES = ("M0", "M0plus", "Mdue", "Mramo")


def read_rephase_bis(path, machines=REPHASE_BIS_MACHINES) -> dict:
    """rephase_bis/{task}_s{seed}.npz (§ 6 bis): z_M has one row per machine, named in `machines`."""
    path = Path(path)
    m = REPHASE_RE.match(path.name)
    if not m:
        raise ContractError(f"{path.name}: name does not match {{task}}_s{{seed}}.npz")
    d = _load(path)
    _need(d, REPHASE_BIS_KEYS + ("machines",) + tuple("val_" + k for k in REPHASE_BIS_KEYS), path)
    names = [str(x) for x in d["machines"]]
    if sorted(names) != sorted(machines):
        raise ContractError(f"{path.name}: machines must be {list(machines)}, got {names}")
    for prefix in ("", "val_"):
        keys = tuple(prefix + k for k in REPHASE_BIS_KEYS)
        if d[prefix + "z_M"].ndim != 2 or d[prefix + "z_M"].shape[0] != len(names):
            raise ContractError(f"{path.name}: {prefix}z_M must be (number of machines, N)")
        _same_length(d, keys, path, axis_of={prefix + "z_M": 1})
        _binary(d[prefix + "y"], path)
        _finite(d, keys, path)
    d.update(task=m["task"], seed=int(m["seed"]), file=path.name, machine_names=names)
    return d


def read_rephase_ter(path) -> dict:
    """rephase_ter/{task}_s{seed}.npz (§ 6 ter): as rephase_bis, with the machines M0, M0plus, Mdue, Mramo."""
    return read_rephase_bis(path, REPHASE_TER_MACHINES)


TRUST_WEIGHTS = (0.5, 1.0, 2.0, 4.0)
TRUST_BRANCHES = tuple(f"Mramo_w{w:g}" for w in TRUST_WEIGHTS)
TRUST_MACHINES = ("M0", "M0plus", "Mdue") + TRUST_BRANCHES
VIEWS_MACHINES = ("M0", "Mramo", "Mpair")
VIEWS_OVERLAPS = (100, 50, 0)
VIEWS_RE = re.compile(r"^(?P<task>.+)_o(?P<overlap>\d+)_s(?P<seed>\d+)\.npz$")


def read_trust(path) -> dict:
    """trust/{task}_s{seed}.npz (§ 12): as rephase_bis, with M0, M0plus, Mdue and one Mramo_w per weight."""
    d = read_rephase_bis(path, TRUST_MACHINES)
    if "built_for_weight" not in d or tuple(float(w) for w in d["built_for_weight"]) != TRUST_WEIGHTS:
        raise ContractError(f"{Path(path).name}: built_for_weight must be {list(TRUST_WEIGHTS)}")
    return d


def read_views(path) -> dict:
    """views/{task}_o{overlap}_s{seed}.npz (§ 13): as rephase_bis, with M0, Mramo, Mpair and the overlap."""
    path = Path(path)
    m = VIEWS_RE.match(path.name)
    if not m:
        raise ContractError(f"{path.name}: name does not match {{task}}_o{{overlap}}_s{{seed}}.npz")
    d = _load(path)
    _need(d, REPHASE_BIS_KEYS + ("machines", "overlap") + tuple("val_" + k for k in REPHASE_BIS_KEYS), path)
    names = [str(x) for x in d["machines"]]
    if sorted(names) != sorted(VIEWS_MACHINES):
        raise ContractError(f"{path.name}: machines must be {list(VIEWS_MACHINES)}, got {names}")
    overlap = int(m["overlap"])
    if overlap not in VIEWS_OVERLAPS or int(d["overlap"]) != overlap:
        raise ContractError(f"{path.name}: overlap must be one of {list(VIEWS_OVERLAPS)} and match the name")
    for prefix in ("", "val_"):
        keys = tuple(prefix + k for k in REPHASE_BIS_KEYS)
        if d[prefix + "z_M"].ndim != 2 or d[prefix + "z_M"].shape[0] != len(names):
            raise ContractError(f"{path.name}: {prefix}z_M must be (number of machines, N)")
        _same_length(d, keys, path, axis_of={prefix + "z_M": 1})
        _binary(d[prefix + "y"], path)
        _finite(d, keys, path)
    d.update(task=m["task"], seed=int(m["seed"]), overlap=overlap, file=path.name, machine_names=names)
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


PEOPLE_BIS_RECIPES = ("controllo", "somma", "mista")


def read_people_bis_fold(path) -> dict:
    """people_bis/fold{f}_s{seed}.npz (§ 7 bis): logits (3, N, 10), one row per recipe, named in `recipes`."""
    path = Path(path)
    m = FOLD_RE.match(path.name)
    if not m:
        raise ContractError(f"{path.name}: name does not match fold{{f}}_s{{seed}}.npz")
    d = _load(path)
    _need(d, ("image", "y", "logits", "recipes"), path)
    names = [str(x) for x in d["recipes"]]
    if sorted(names) != sorted(PEOPLE_BIS_RECIPES):
        raise ContractError(f"{path.name}: recipes must be {list(PEOPLE_BIS_RECIPES)}, got {names}")
    lg = d["logits"]
    if lg.ndim != 3 or lg.shape[0] != len(names) or lg.shape[2] != 10:
        raise ContractError(f"{path.name}: logits must be (number of recipes, N, 10)")
    if lg.shape[1] != d["image"].size or d["y"].size != d["image"].size:
        raise ContractError(f"{path.name}: lengths differ")
    if d["y"].min() < 0 or d["y"].max() > 9:
        raise ContractError(f"{path.name}: y outside 0-9")
    _finite(d, ("logits",), path)
    d.update(fold=int(m["fold"]), seed=int(m["seed"]), file=path.name, recipe_names=names)
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


SHAPES = ("gauss", "two", "uniform", "t3")
SHAPE_KEYS = ("y", "image", "a", "z_plus", "z_minus")
CHART_MACHINES = ("M", "Mramo", "Mblind")
CHART_VIEWS = (100, 50, 25)
CHART_KEYS = REPHASE_BIS_KEYS + ("z_H0",)
CHART_RE = re.compile(r"^(?P<task>.+)_v(?P<v>\d+)_s(?P<seed>\d+)\.npz$")


def read_shape(path) -> dict:
    """shape/{task}_mu{mu}_s{seed}.npz (§ 15): a, z_plus and z_minus have one row per shape of the test frame."""
    path = Path(path)
    m = BENCH_RE.match(path.name)
    if not m:
        raise ContractError(f"{path.name}: name does not match {{task}}_mu{{mu}}_s{{seed}}.npz")
    d = _load(path)
    _need(d, SHAPE_KEYS + ("shapes",) + tuple("val_" + k for k in SHAPE_KEYS), path)
    names = tuple(str(x) for x in d["shapes"])
    if names != SHAPES:
        raise ContractError(f"{path.name}: shapes must be {list(SHAPES)}, got {list(names)}")
    for prefix in ("", "val_"):
        keys = tuple(prefix + k for k in SHAPE_KEYS)
        rows = {prefix + k: 1 for k in ("a", "z_plus", "z_minus")}
        for k in rows:
            if d[k].ndim != 2 or d[k].shape[0] != len(SHAPES):
                raise ContractError(f"{path.name}: {k} must be (number of shapes, N)")
        _same_length(d, keys, path, axis_of=rows)
        _binary(d[prefix + "y"], path)
        _finite(d, keys, path)
    d.update(task=m["task"], mu=_mu(m["mu"]), seed=int(m["seed"]), file=path.name, shape_names=list(names))
    return d


def read_chart(path) -> dict:
    """chart/{task}_v{v}_s{seed}.npz (§ 14): as rephase_bis, with z_H0 (the judge without the bias), the machines
    M, Mramo, Mblind and v, the share of the inside of the image that the machine sees (100, 50 or 25)."""
    path = Path(path)
    m = CHART_RE.match(path.name)
    if not m:
        raise ContractError(f"{path.name}: name does not match {{task}}_v{{v}}_s{{seed}}.npz")
    d = _load(path)
    _need(d, CHART_KEYS + ("machines", "v") + tuple("val_" + k for k in CHART_KEYS), path)
    names = [str(x) for x in d["machines"]]
    if sorted(names) != sorted(CHART_MACHINES):
        raise ContractError(f"{path.name}: machines must be {list(CHART_MACHINES)}, got {names}")
    v = int(m["v"])
    if v not in CHART_VIEWS or int(d["v"]) != v:
        raise ContractError(f"{path.name}: v must be one of {list(CHART_VIEWS)} and match the name")
    for prefix in ("", "val_"):
        keys = tuple(prefix + k for k in CHART_KEYS)
        if d[prefix + "z_M"].ndim != 2 or d[prefix + "z_M"].shape[0] != len(names):
            raise ContractError(f"{path.name}: {prefix}z_M must be (number of machines, N)")
        _same_length(d, keys, path, axis_of={prefix + "z_M": 1})
        _binary(d[prefix + "y"], path)
        _finite(d, keys, path)
    d.update(task=m["task"], seed=int(m["seed"]), v=v, file=path.name, machine_names=names)
    return d


# --- E10 to E12: the bench of the informative chart (protocol §§ 16-18) -------------------------------------------------

TRACE_VIEWS = (25, 100)
TRACE_LEVELS = (0.0, 0.5, 1.0, 1.5)
TRACE_KEYS = ("y", "image", "eps", "a", "z_H", "z_C", "z_M", "z_R")
TRACE_RE = CHART_RE


def read_trace(path) -> dict:
    """trace/{task}_v{v}_s{seed}.npz (§ 16). The frame has four levels of sensitivity: `a`, `z_H` (the judge) and
    `z_C` (the judge trained at that level) have one row per level; `z_M` is the soloist, which does not see the
    frame; `z_R[t][d]` is the branch built for level t read at level d. v is 25 or 100."""
    path = Path(path)
    m = TRACE_RE.match(path.name)
    if not m:
        raise ContractError(f"{path.name}: name does not match {{task}}_v{{v}}_s{{seed}}.npz")
    d = _load(path)
    _need(d, TRACE_KEYS + ("levels", "v") + tuple("val_" + k for k in TRACE_KEYS), path)
    levels = [float(x) for x in d["levels"]]
    if levels != list(TRACE_LEVELS):
        raise ContractError(f"{path.name}: levels must be {list(TRACE_LEVELS)}, got {levels}")
    v = int(m["v"])
    if v not in TRACE_VIEWS or int(d["v"]) != v:
        raise ContractError(f"{path.name}: v must be one of {list(TRACE_VIEWS)} and match the name")
    n_levels = len(levels)
    for prefix in ("", "val_"):
        n = d[prefix + "y"].shape[0]
        want = {"image": (n,), "eps": (n,), "z_M": (n,), "a": (n_levels, n), "z_H": (n_levels, n),
                "z_C": (n_levels, n), "z_R": (n_levels, n_levels, n)}
        for k, shape in want.items():
            if d[prefix + k].shape != shape:
                raise ContractError(f"{path.name}: {prefix}{k} must have shape {shape}, got {d[prefix + k].shape}")
        _binary(d[prefix + "y"], path)
        _finite(d, tuple(prefix + k for k in ("eps", "a", "z_H", "z_C", "z_M", "z_R")), path)
    d.update(task=m["task"], seed=int(m["seed"]), v=v, file=path.name)
    return d
