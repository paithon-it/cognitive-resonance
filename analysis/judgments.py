"""How many judgments a correction needs (a reading added after the results: not a test).

The theorem of protocol § 14 takes the bias as known. A machine has to learn it from judgments of
the judge whose right answer is known. With n of them, the slope of the judge's cue on the frame
is known with a variance of 1 / n in units of the judge's clean noise, whatever the size of the
bias: the bias left in the pair is 1 / n, and the share of the gain that is kept follows
(`cognitive_resonance.chart.share_kept_after`).

This script measures it on the files of E8. Per unit and per n: n presentations are drawn from
the validation set, the slope is estimated on them, the judge's test cue is corrected with that
slope, and the corrected judge is put beside the soloist with the best linear use of the two
cues (weights from one half of the test images, sensitivity read on the other, swapped).

    python analysis/judgments.py [--raw results/raw/chart] [--out results/e8_judgments.json]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from common import ROOT, Timer, environment, med, run_pool, write_json
from cognitive_resonance import chart as theorem
from cognitive_resonance import estimators as est
from cognitive_resonance import raw

NS = (5, 10, 20, 50, 100, 200, 500, 1000, 5000)
DRAWS = 40
SEED = 20261120


def best_linear(X, y, half) -> float:
    """Sensitivity of the best linear use of the columns of X: weights from one half, d_a on the other, swapped."""
    out = []
    for fit, read in ((0, 1), (1, 0)):
        f, r = half == fit, half == read
        mean1, mean0 = X[f & (y == 1)].mean(axis=0), X[f & (y == 0)].mean(axis=0)
        res = np.vstack([X[f & (y == 1)] - mean1, X[f & (y == 0)] - mean0])
        w = np.linalg.solve(res.T @ res / (res.shape[0] - 2), mean1 - mean0)
        out.append(float(est.d_a(X[r] @ w, y[r])))
    return float(np.mean(out))


def unit(path) -> dict:
    d = raw.read_chart(path)
    y, a, vy, va = d["y"], np.asarray(d["a"], float), d["val_y"], np.asarray(d["val_a"], float)
    names = d["machine_names"]
    sd_h = est.within_class_sd(d["val_z_H"], vy)
    zh, vzh = np.asarray(d["z_H"], float) / sd_h, np.asarray(d["val_z_H"], float) / sd_h
    i = names.index("M")
    zm = est.standardize(d["z_M"][i], y, d["val_z_M"][i], vy)
    half = est.split_halves(d["image"], y)
    pair = lambda slope: best_linear(np.column_stack([zh - slope * a, zm]), y, half)       # noqa: E731
    slope_all = float(est.slope_within(vzh, va, vy))
    d1, d_all = pair(0.0), pair(slope_all)
    h = zh - float(est.slope_within(zh, a, y)) * a
    b = (float(est.slope_within(zh, a, y)) * est.within_class_sd(a, y)) ** 2 / est.within_class_sd(h, y) ** 2
    r = float(est.pearson_within(h, zm, y))
    gen = np.random.default_rng([SEED, d["seed"], d["v"], sum(map(ord, d["task"]))])
    by_n = {}
    for n in NS:
        kept = []
        for _ in range(DRAWS if n < vy.size else 1):
            pick = gen.choice(vy.size, size=min(n, vy.size), replace=False)
            if len(set(vy[pick].tolist())) < 2 or min(np.sum(vy[pick] == 0), np.sum(vy[pick] == 1)) < 2:
                continue                                                 # a slope within the class needs both classes
            slope = float(est.slope_within(vzh[pick], va[pick], vy[pick]))
            kept.append((pair(slope) ** 2 - d1 ** 2) / (d_all ** 2 - d1 ** 2))
        by_n[str(n)] = {"share_kept": med(kept), "draws": len(kept), "predicted": theorem.share_kept_after(n, r, b)}
    return {"task": d["task"], "seed": d["seed"], "v": d["v"], "b": b, "r": r, "d1": d1, "d_all_judgments": d_all,
            "gain_squared": d_all ** 2 - d1 ** 2, "by_n": by_n}


def summary(units) -> list:
    rows = []
    for v in raw.CHART_VIEWS:
        us = [u for u in units if u["v"] == v and u["gain_squared"] > 0.2]      # a gain to keep: not where it is nil
        if not us:
            continue
        rows.append({"v": v, "units": len(us), "median_b": med([u["b"] for u in us]), "median_r": med([u["r"] for u in us]),
                     "by_n": {str(n): {"share_kept": med([u["by_n"][str(n)]["share_kept"] for u in us]),
                                       "lowest": min(u["by_n"][str(n)]["share_kept"] for u in us),
                                       "highest": max(u["by_n"][str(n)]["share_kept"] for u in us),
                                       "predicted": med([u["by_n"][str(n)]["predicted"] for u in us])} for n in NS}})
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", type=Path, default=ROOT / "results" / "raw" / "chart")
    ap.add_argument("--out", type=Path, default=ROOT / "results" / "e8_judgments.json")
    ap.add_argument("--workers", type=int, default=3)
    args = ap.parse_args()
    timer = Timer()
    files = raw.list_units(args.raw, raw.CHART_RE)
    if not files:
        raise SystemExit(f"no chart files in {args.raw}")
    units = run_pool(unit, files, args.workers)
    out = {"experiment": "E8, how many judgments a correction needs", "protocol": "§ 14, on the files of E8",
           "status": "a reading added on 20 November 2026, after the results of E8 were seen: not a test",
           "how": "n presentations drawn from the validation set give the slope of the judge's cue on the frame; the "
                  "test cue of the judge is corrected with it and put beside the soloist with the best linear use of the "
                  "two cues. Share kept = (D_n^2 - D_1^2) / (D_all^2 - D_1^2), median over the draws; predicted from "
                  "a bias left of 1 / n. Units where the gain is below 0.2 are left out",
           "settings": {"ns": list(NS), "draws": DRAWS}, "by_v": summary(units), "units": units,
           "minutes": timer.minutes(), "environment": environment()}
    write_json(args.out, out)
    for row in out["by_v"]:
        print("v", row["v"], "units", row["units"], {n: (round(x["share_kept"], 3), round(x["predicted"], 3)) for n, x in row["by_n"].items()})
    print("written", args.out)


if __name__ == "__main__":
    main()
