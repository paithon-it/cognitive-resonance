"""E12 (protocol § 18): the push of the judge estimated from yes-or-no answers.

Reads results/raw/trace/*_v25_*.npz at level 0, where the frame says nothing (the case of § 14).
`judgments.py` estimates the slope of the judge's graded cue on the frame; a person gives an
answer, not a cue. Here the same draws of n validation presentations feed two estimators:

- graded: the slope of the standardized cue on the frame, within class;
- yes and no: a probit regression of the judge's answer (the sign of the raw cue) on the class
  (two intercepts) and on the frame, with a small quadratic penalty. Its slope is in units of
  the clean noise; divided by sqrt(1 + beta^2 var(a)) it is in the units of the standardized cue.

The judge's test cue is corrected with each estimate and put beside the soloist with the best
linear use of the two cues. Share kept = (D_n^2 - D_1^2) / (D_all^2 - D_1^2), median over the
draws. The prediction: n yes-or-no answers are worth n / kappa graded judgments, with kappa
from the Fisher information (`cognitive_resonance.chart.yes_no_factor`). Writes Y1-Y4.

Added after the results (not part of the registered test): the registered prediction left out a
factor. The probit slope is in units of the clean noise, and bringing it to the units of the
judge's whole cue with the same estimate shrinks its error by (1 + B)^(3/2); the bias left is
kappa / (n (1 + B)^2), not kappa / n. The fields `kappa_corrected` and `predicted_yes_no_corrected`
carry that reading; the criteria are untouched and still use the registered prediction.

    python analysis/yesno.py [--raw results/raw/trace] [--out results/e12_yesno.json]
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm

from common import ROOT, Timer, environment, med, run_pool, verdict, write_json
from cognitive_resonance import chart as theorem
from cognitive_resonance import estimators as est
from cognitive_resonance import raw
from judgments import best_linear

NS = (5, 10, 20, 50, 100, 200, 500, 1000, 5000)
DRAWS = 40
SEED = 20261123
PENALTY = 0.05                    # on the squares of the three coefficients of the probit regression
TARGET = 0.90                     # Y2: the share kept that counts as "enough"
VIEW, LEVEL = 25, 0               # the files and the level this test reads
WORSE_IN = 0.8                    # Y1: share of the units where yes-or-no keeps less than the graded cue at n = 50
FACTOR = (3.0, 9.0)               # Y2: judgments needed with yes-or-no over judgments needed with the graded cue
GAP = 0.07                        # Y3: |measured - predicted| share kept at n = 100 and 200
ENOUGH = 0.90                     # Y4: share kept with all the validation answers


def probit_slope(answer, y, a) -> float:
    """Slope on the frame of a probit regression of the answer on the class (two intercepts) and the frame."""
    q = 2.0 * np.asarray(answer, float) - 1.0
    y = np.asarray(y, int)
    a = np.asarray(a, float)

    def loss(p):
        eta = p[y] + p[2] * a
        g = -q * np.exp(norm.logpdf(q * eta) - norm.logcdf(q * eta))
        grad = np.array([g[y == 0].sum(), g[y == 1].sum(), float(g @ a)]) + 2.0 * PENALTY * p
        return float(-norm.logcdf(q * eta).sum() + PENALTY * float(p @ p)), grad

    res = minimize(loss, x0=np.zeros(3), jac=True, method="BFGS")
    return float(res.x[2])


def needed(ns, shares, target: float = TARGET) -> float:
    """The number of judgments at which the share kept first reaches `target`: linear in ln n between grid points.
    Infinite when it never does on the grid."""
    for i, (n, s) in enumerate(zip(ns, shares)):
        if s is None or s < target:
            continue
        if i == 0 or shares[i - 1] is None:
            return float(n)
        t = (target - shares[i - 1]) / (s - shares[i - 1])
        return float(math.exp(math.log(ns[i - 1]) + t * (math.log(n) - math.log(ns[i - 1]))))
    return math.inf


def unit(path) -> dict | None:
    d = raw.read_trace(path)
    if d["v"] != VIEW:
        return None
    k = list(raw.TRACE_LEVELS).index(float(LEVEL))
    y, a = d["y"], np.asarray(d["a"][k], float)
    vy, va = d["val_y"], np.asarray(d["val_a"][k], float)
    sd_h = est.within_class_sd(d["val_z_H"][k], vy)
    zh, vzh = np.asarray(d["z_H"][k], float) / sd_h, np.asarray(d["val_z_H"][k], float) / sd_h
    vans = np.asarray(d["val_z_H"][k], float) > 0
    zm = est.standardize(d["z_M"], y, d["val_z_M"], vy)
    half = est.split_halves(d["image"], y)
    pair = lambda slope: best_linear(np.column_stack([zh - slope * a, zm]), y, half)       # noqa: E731
    var_a = est.within_class_sd(va, vy) ** 2
    to_std = lambda beta: beta / math.sqrt(1.0 + beta ** 2 * var_a)                         # noqa: E731
    d1, d_all = pair(0.0), pair(float(est.slope_within(vzh, va, vy)))
    slope = float(est.slope_within(zh, a, y))
    h = zh - slope * a
    sd_clean = est.within_class_sd(h, y)
    b = (slope * est.within_class_sd(a, y)) ** 2 / sd_clean ** 2
    r = float(est.pearson_within(h, zm, y))
    means = [float(h[y == c].mean()) / sd_clean for c in (0, 1)]
    kappa = theorem.yes_no_factor(means[0], means[1], slope / sd_clean, est.within_class_sd(a, y) ** 2)
    gen = np.random.default_rng([SEED, d["seed"], d["v"], sum(map(ord, d["task"]))])
    by_n = {}
    for n in NS:
        graded, yesno = [], []
        for _ in range(DRAWS if n < vy.size else 1):
            pick = gen.choice(vy.size, size=min(n, vy.size), replace=False)
            if min(np.sum(vy[pick] == 0), np.sum(vy[pick] == 1)) < 2:
                continue                                                 # a slope within the class needs both classes
            share = lambda s: (pair(s) ** 2 - d1 ** 2) / (d_all ** 2 - d1 ** 2)             # noqa: E731
            graded.append(share(float(est.slope_within(vzh[pick], va[pick], vy[pick]))))
            yesno.append(share(to_std(probit_slope(vans[pick], vy[pick], va[pick]))))
        by_n[str(n)] = {"graded": med(graded), "yes_no": med(yesno), "draws": len(graded),
                        "predicted_graded": theorem.share_kept_after(n, r, b),
                        "predicted_yes_no": theorem.share_kept_after(n / kappa, r, b),
                        "predicted_yes_no_corrected": theorem.share_kept_after(n * (1.0 + b) ** 2 / kappa, r, b)}
    need_g = needed(NS, [by_n[str(n)]["graded"] for n in NS])
    need_y = needed(NS, [by_n[str(n)]["yes_no"] for n in NS])
    return {"task": d["task"], "seed": d["seed"], "v": d["v"], "b": b, "r": r, "kappa": kappa,
            "kappa_corrected": kappa / (1.0 + b) ** 2, "class_means": means,
            "beta_clean": slope / sd_clean, "d1": d1, "d_all_judgments": d_all, "gain_squared": d_all ** 2 - d1 ** 2,
            "right_answers_of_the_judge_on_validation": float(np.mean(vans == (vy == 1))),
            "needed_graded": need_g, "needed_yes_no": need_y,
            "factor": (need_y / need_g) if math.isfinite(need_g) else None, "by_n": by_n}


def criteria(units) -> dict:
    us = [u for u in units if u["gain_squared"] > 0.2]
    out = {}
    if not us:
        return {k: verdict(None, reason="no unit with a gain to keep") for k in ("Y1", "Y2", "Y3", "Y4")}
    worse = sum(u["by_n"]["50"]["yes_no"] < u["by_n"]["50"]["graded"] for u in us)
    out["Y1"] = verdict(worse >= math.ceil(WORSE_IN * len(us)), units_where_yes_no_keeps_less=worse, units=len(us),
                        needed=math.ceil(WORSE_IN * len(us)),
                        median_graded=med([u["by_n"]["50"]["graded"] for u in us]),
                        median_yes_no=med([u["by_n"]["50"]["yes_no"] for u in us]),
                        rule="with 50 judgments the share kept with yes-or-no answers is lower than with the graded "
                             f"cue in at least {WORSE_IN:.0%} of the units")
    factors = [u["factor"] if u["factor"] is not None else math.inf for u in us]
    middle = float(np.median(np.asarray(factors, float)))
    out["Y2"] = verdict(FACTOR[0] <= middle <= FACTOR[1], median_factor=middle if math.isfinite(middle) else None,
                        band=list(FACTOR), units_never_reaching_target=sum(not math.isfinite(f) for f in factors),
                        median_needed_graded=med([u["needed_graded"] for u in us]),
                        median_needed_yes_no=med([u["needed_yes_no"] for u in us]),
                        median_kappa=med([u["kappa"] for u in us]),
                        rule=f"judgments needed to keep {TARGET:.0%} of the gain, yes-or-no over graded, median over "
                             f"the units, within {list(FACTOR)}; a unit that never reaches it counts as outside")
    gap = {n: med([abs(u["by_n"][n]["yes_no"] - u["by_n"][n]["predicted_yes_no"]) for u in us]) for n in ("100", "200")}
    out["Y3"] = verdict(all(g is not None and g <= GAP for g in gap.values()), median_abs_gap=gap, at_most=GAP,
                        median_measured={n: med([u["by_n"][n]["yes_no"] for u in us]) for n in gap},
                        median_predicted={n: med([u["by_n"][n]["predicted_yes_no"] for u in us]) for n in gap},
                        rule="with 100 and with 200 yes-or-no answers, |measured - predicted| share kept, median over "
                             f"the units, is at most {GAP}")
    last = med([u["by_n"]["5000"]["yes_no"] for u in us])
    out["Y4"] = verdict(last is not None and last >= ENOUGH, median_share_kept=last, at_least=ENOUGH,
                        rule=f"with 5,000 yes-or-no answers the share kept is in the median at least {ENOUGH}")
    return out


def summary(units) -> dict:
    us = [u for u in units if u["gain_squared"] > 0.2]
    return {"units": len(us), "left_out": len(units) - len(us), "median_b": med([u["b"] for u in us]),
            "median_r": med([u["r"] for u in us]), "median_kappa": med([u["kappa"] for u in us]),
            "added_after_the_results": {
                "median_kappa_corrected": med([u["kappa_corrected"] for u in us]),
                "median_factor_measured": med([u["factor"] for u in us]),
                "factor_by_target": {f"{t:g}": med([needed(NS, [u["by_n"][str(n)]["yes_no"] for n in NS], t)
                                                    / needed(NS, [u["by_n"][str(n)]["graded"] for n in NS], t) for u in us])
                                     for t in (0.8, 0.9, 0.95, 0.975)},
                "why": "bias left with yes and no = kappa / (n (1 + B)^2): the slope is needed in units of the "
                       "judge's whole noise, and the same estimate sets the unit"},
            "kappa_range": [min(u["kappa"] for u in us), max(u["kappa"] for u in us)] if us else None,
            "by_n": {str(n): {k: med([u["by_n"][str(n)][k] for u in us])
                              for k in ("graded", "yes_no", "predicted_graded", "predicted_yes_no",
                                        "predicted_yes_no_corrected")}
                     | {"yes_no_range": [min(u["by_n"][str(n)]["yes_no"] for u in us),
                                         max(u["by_n"][str(n)]["yes_no"] for u in us)] if us else None}
                     for n in NS}}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", type=Path, default=ROOT / "results" / "raw" / "trace")
    ap.add_argument("--out", type=Path, default=ROOT / "results" / "e12_yesno.json")
    ap.add_argument("--workers", type=int, default=3)
    args = ap.parse_args()
    timer = Timer()
    files = [f for f in raw.list_units(args.raw, raw.TRACE_RE) if f"_v{VIEW}_" in f.name]
    if not files:
        raise SystemExit(f"no trace files of v = {VIEW} in {args.raw}")
    units = [u for u in run_pool(unit, files, args.workers) if u is not None]
    out = {"experiment": "E12, the push estimated from yes-or-no answers", "protocol": "§ 18 (v2.3)",
           "status": "registered before the run",
           "settings": {"ns": list(NS), "draws": DRAWS, "penalty": PENALTY, "target": TARGET, "view": VIEW, "level": LEVEL},
           "criteria": criteria(units), "summary": summary(units), "units": units,
           "minutes": timer.minutes(), "environment": environment()}
    write_json(args.out, out)
    for name, c in out["criteria"].items():
        print(name, c["outcome"])
    print("minutes", out["minutes"], "written", args.out)


if __name__ == "__main__":
    main()
