"""E14, what a shortcut leaves unlearned (protocol § 20).

For every network of gpu/shortcut.py: the sensitivity as it is, with the frame removed from the output, and with the
last layer trained again where the frame says nothing; all over the network of the same task, seed and length of
training that never had the shortcut. "The best" is the higher of the two corrections. Counts by task, on the median
of the seeds.

    python analysis/unlearned.py
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ROOT, med, write_json  # noqa: E402

from cognitive_resonance import estimators as est  # noqa: E402

NAME = re.compile(r"(?P<task>[a-z-]+)_e(?P<epochs>\d+)_s(?P<seed>\d+)\.npz")
MORE, LEFT, NEED = 0.05, 0.90, 4


def removed(z, a, y) -> float:
    return float(est.d_a(z - float(est.slope_within(z, a, y)) * a, y))


def read(path: Path) -> list:
    m = NAME.fullmatch(path.name)
    d = np.load(path)
    y, rows = d["y"], []
    for i, mu in enumerate(d["mus"]):
        z, zl, a = d["z"][i].astype(float), d["z_last"][i].astype(float), d["a"][i].astype(float)
        rows.append({"task": m["task"], "seed": int(m["seed"]), "epochs": int(m["epochs"]), "mu": float(mu),
                     "d_as_it_is": float(est.d_a(z, y)), "d_out": removed(z, a, y), "d_last": float(est.d_a(zl, y)),
                     "d_last_out": removed(zl, a, y),
                     "right_as_it_is": float(((z > 0) == (y == 1)).mean()), "right_last": float(((zl > 0) == (y == 1)).mean())})
    ref = next(r for r in rows if r["mu"] == 0.0)["d_out"]
    for r in rows:
        r["d_reference"] = ref
        for k in ("as_it_is", "out", "last", "last_out"):
            r["kept_" + k] = r["d_" + k] / ref
        r["kept_best"] = max(r["kept_out"], r["kept_last"])
    return rows


def by_task(rows, epochs, mu, key) -> dict:
    out = {}
    for t in sorted({r["task"] for r in rows}):
        v = [r[key] for r in rows if r["task"] == t and r["epochs"] == epochs and r["mu"] == mu]
        if v:
            out[t] = float(np.median(v))
    return out


def main() -> None:
    files = sorted((ROOT / "results" / "raw" / "shortcut").glob("*.npz"))
    rows = [r for f in files for r in read(f)]
    lengths, mus = sorted({r["epochs"] for r in rows}), sorted({r["mu"] for r in rows})
    table = []
    for e in lengths:
        for mu in mus:
            rs = [r for r in rows if r["epochs"] == e and r["mu"] == mu]
            row = {"epochs": e, "mu": mu, "networks": len(rs)}
            for k in ("kept_as_it_is", "kept_out", "kept_last", "kept_last_out", "kept_best", "d_reference"):
                tasks = by_task(rows, e, mu, k)
                row[k] = med(list(tasks.values()))
                row[k + "_min_task"], row[k + "_max_task"] = min(tasks.values()), max(tasks.values())
            table.append(row)
    short, long, top = lengths[0], lengths[-1], mus[-1]
    out_s, last_s = by_task(rows, short, top, "kept_out"), by_task(rows, short, top, "kept_last")
    l1 = {t: last_s[t] - out_s[t] > MORE for t in out_s}
    best_l = by_task(rows, long, top, "kept_best")
    l2 = {t: best_l[t] < LEFT for t in best_l}
    l3 = {}
    for e in lengths:
        for t in sorted({r["task"] for r in rows}):
            kept = [by_task(rows, e, mu, "kept_best").get(t) for mu in mus]
            l3[f"{t}|{e}"] = float(spearmanr(mus, kept).statistic) <= -0.9
    tasks_l3 = {t: all(v for k, v in l3.items() if k.startswith(t + "|")) for t in sorted({r["task"] for r in rows})}
    best_s = by_task(rows, short, top, "kept_best")
    criteria = {
        "L1": {"outcome": "pass" if sum(l1.values()) >= NEED else "fail", "tasks_passing": sum(l1.values()), "by_task": l1,
               "median_kept_out": med(list(out_s.values())), "median_kept_last": med(list(last_s.values())),
               "rule": f"at mu = {top:g} and {short} epochs, kept with the last layer retrained exceeds kept with the frame "
                       f"removed from the output by more than {MORE}, in at least {NEED} tasks of 5"},
        "L2": {"outcome": "pass" if sum(l2.values()) >= NEED else "fail", "tasks_passing": sum(l2.values()), "by_task": l2,
               "median_kept_best": med(list(best_l.values())), "kept_best_by_task": best_l,
               "rule": f"at mu = {top:g} and {long} epochs, the best of the two corrections keeps less than {LEFT}, in at "
                       f"least {NEED} tasks of 5"},
        "L3": {"outcome": "pass" if sum(tasks_l3.values()) >= NEED else "fail", "tasks_passing": sum(tasks_l3.values()),
               "by_task": tasks_l3,
               "rule": "the best of the two corrections falls with mu (rank correlation not above -0.9) at both lengths, "
                       f"in at least {NEED} tasks of 5"}}
    change = {t: best_l[t] - best_s[t] for t in best_l}
    out = {"protocol": "§ 20, E14", "files": len(files), "lengths": lengths, "mus": mus, "table": table,
           "criteria": criteria,
           "from_short_to_long_at_strongest": {"by_task": change, "median": med(list(change.values()))},
           "units": rows}
    write_json(ROOT / "results" / "e14_unlearned.json", out)
    for r in table:
        print(r["epochs"], r["mu"], r["networks"], "as", round(r["kept_as_it_is"], 3), "out", round(r["kept_out"], 3), "last",
              round(r["kept_last"], 3), "last+out", round(r["kept_last_out"], 3), "best", round(r["kept_best"], 3),
              (round(r["kept_best_min_task"], 2), round(r["kept_best_max_task"], 2)), "ref d", round(r["d_reference"], 2))
    print({k: (v["outcome"], v["tasks_passing"]) for k, v in criteria.items()}, "change 30->90:", round(out["from_short_to_long_at_strongest"]["median"], 3))


if __name__ == "__main__":
    main()
