"""Removing the whole push or only the excess, at face value: a look at old data (not a test).

Written on 20 November 2026 to size the expectations of protocol § 17, before the run of E10 to
E12. On the files of E8, resampled as in `informative.py` so that the frame has a sensitivity
of its own (0, 0.5, 1 or 1.5), the two cues are summed as they are (raw log-odds, weight 1):

- with the soloist M;
- with the partner of E8, whose branch was trained when the frame said nothing ("old");
- with M plus a branch linear in the frame, trained on the error of the sum on the resampled
  validation set ("new").

    python analysis/informative_face.py [--raw results/raw/chart] [--out results/e8_informative_face.json]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from common import ROOT, Timer, environment, med, run_pool, write_json
from cognitive_resonance import estimators as est
from cognitive_resonance import raw
from fixed import accuracy
from informative import keep, sum_branch

TARGETS = (0.0, 0.5, 1.0, 1.5)
VIEWS = (25, 100)
SEED = 20261122


def unit(path) -> dict | None:
    d = raw.read_chart(path)
    if d["v"] not in VIEWS:
        return None
    gen = np.random.default_rng([SEED, d["seed"], d["v"], sum(map(ord, d["task"]))])
    names = d["machine_names"]
    i_m, i_old = names.index("M"), names.index("Mramo")
    out = {}
    for t in TARGETS:
        kt = keep(gen, np.asarray(d["a"], float), d["y"], t)
        kv = keep(gen, np.asarray(d["val_a"], float), d["val_y"], t)
        y, a = d["y"][kt], np.asarray(d["a"], float)[kt]
        vy, va = d["val_y"][kv], np.asarray(d["val_a"], float)[kv]
        zh, zm, zo = (np.asarray(x, float)[kt] for x in (d["z_H"], d["z_M"][i_m], d["z_M"][i_old]))
        c, c0 = sum_branch(np.asarray(d["val_z_H"], float)[kv], np.asarray(d["val_z_M"][i_m], float)[kv], va, vy)
        sums = {"soloist": zh + zm, "old": zh + zo, "new": zh + zm + c * a + c0}
        out[f"{t:g}"] = {"kept_test": int(kt.sum()),
                         **{f"d_{k}": float(est.d_a(s, y)) for k, s in sums.items()},
                         **{f"right_{k}": accuracy(s, y) for k, s in sums.items()}}
    return {"task": d["task"], "seed": d["seed"], "v": d["v"], "by_d_a": out}


def summary(units) -> list:
    rows = []
    for v in VIEWS:
        for t in TARGETS:
            us = [u["by_d_a"][f"{t:g}"] for u in units if u["v"] == v]
            diff = lambda a, b: [u[f"d_{a}"] - u[f"d_{b}"] for u in us]                    # noqa: E731
            row = {"v": v, "target_d_a": t, "units": len(us)}
            for k in us[0]:
                row[k] = med([u[k] for u in us])
            for a, b in (("new", "old"), ("old", "soloist"), ("new", "soloist")):
                row[f"{a}_minus_{b}"] = med(diff(a, b))
                row[f"units_where_{a}_beats_{b}"] = sum(x > 0 for x in diff(a, b))
            rows.append(row)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", type=Path, default=ROOT / "results" / "raw" / "chart")
    ap.add_argument("--out", type=Path, default=ROOT / "results" / "e8_informative_face.json")
    ap.add_argument("--workers", type=int, default=3)
    args = ap.parse_args()
    timer = Timer()
    files = raw.list_units(args.raw, raw.CHART_RE)
    if not files:
        raise SystemExit(f"no chart files in {args.raw}")
    units = [u for u in run_pool(unit, files, args.workers) if u is not None]
    out = {"experiment": "E8 resampled, the pairs at face value with the old partner and a new linear branch",
           "protocol": "§ 17 (what was known), on the files of E8",
           "status": "a look at old data made on 20 November 2026 to write the expectations of § 17: not a test",
           "how": "presentations resampled as in informative.py; raw log-odds summed at weight 1; d_a and right answers "
                  "of z_H + z_M, z_H + z_Mramo (old) and z_H + z_M + c a + c0 (new, fitted on the resampled validation set)",
           "settings": {"targets": list(TARGETS), "views": list(VIEWS)}, "by_v_and_d_a": summary(units), "units": units,
           "minutes": timer.minutes(), "environment": environment()}
    write_json(args.out, out)
    for r in out["by_v_and_d_a"]:
        print("v", r["v"], "d_a", r["target_d_a"], "new-old", round(r["new_minus_old"], 3), r["units_where_new_beats_old"],
              "| old-soloist", round(r["old_minus_soloist"], 3), r["units_where_old_beats_soloist"],
              "| new-soloist", round(r["new_minus_soloist"], 3), r["units_where_new_beats_soloist"])
    print("written", args.out)


if __name__ == "__main__":
    main()
