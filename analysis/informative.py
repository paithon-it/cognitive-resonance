"""When the trace says something true (a reading added after the results: not a test).

In the theorem of protocol § 14 the bias does not depend on the class. A real bias often leans
on something that is informative and gets too much weight: a patient's chart tells something
about the disease, and the doctor trusts it more than it deserves. `cognitive_resonance.chart`
has the theorem for that case: only the excess over the deserved weight is a bias, and the gain
of a machine built on the judge is B / (1 - r^2 + B) * P * (1 - beta_0 / beta)^2.

No run of E8 has an informative frame at test: there the frame is drawn independently of the
class. This script makes one by resampling. Every presentation of the files of E8 is kept with a
probability that depends on its frame and on its class, so that among the kept ones the frame
has the same spread in the two classes and a chosen sensitivity d_a (0, 0.5 or 1). The cues of
the networks are untouched. On the resampled files, per unit:

- the theorem, fed with quantities measured on the judge and on the soloist;
- the pair with the soloist and the best linear use of judge, soloist and frame;
- the pair with the partner of E8, whose branch was trained when the frame said nothing;
- the pair with a new branch, linear in the frame, trained on the error of the sum on the
  resampled validation set (two numbers fitted: a slope and an offset).

    python analysis/informative.py [--raw results/raw/chart] [--out results/e8_informative.json]
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

from common import ROOT, Timer, environment, med, run_pool, write_json
from cognitive_resonance import chart as theorem
from cognitive_resonance import estimators as est
from cognitive_resonance import raw
from judgments import best_linear

TARGETS = (0.0, 0.5, 1.0)        # sensitivity of the frame after the resampling
SPREAD = 1.0                     # standard deviation of the kept frames within a class (the frames drawn have variance 2)
SEED = 20261121


def keep(gen, a, y, d_a: float) -> np.ndarray:
    """Which presentations stay, so that among them a | class is normal with mean +-d_a * SPREAD / 2 and sd SPREAD."""
    mu = d_a * SPREAD / 2.0
    s = 2.0 * np.asarray(y, float) - 1.0
    var0 = 2.0
    log_ratio = -(a - s * mu) ** 2 / (2 * SPREAD ** 2) + a ** 2 / (2 * var0)
    top = (mu ** 2) * var0 / (2 * SPREAD ** 2 * (var0 - SPREAD ** 2)) if mu else 0.0     # the largest value of log_ratio
    return gen.random(a.size) < np.exp(log_ratio - top)


def sum_branch(zh, zm, a, y) -> tuple[float, float]:
    """A branch linear in the frame, trained on the error of the sum z_H + z_M + c a + c0 (raw log-odds, weight 1)."""
    t = np.asarray(y, float)

    def loss(p):
        z = zh + zm + p[0] * a + p[1]
        return float(np.mean(np.logaddexp(0.0, z) - t * z))

    res = minimize(loss, x0=np.zeros(2), method="BFGS")
    return float(res.x[0]), float(res.x[1])


def one(d: dict, d_a: float, gen) -> dict:
    names = d["machine_names"]
    kt, kv = keep(gen, np.asarray(d["a"], float), d["y"], d_a), keep(gen, np.asarray(d["val_a"], float), d["val_y"], d_a)
    y, vy = d["y"][kt], d["val_y"][kv]
    a, va = np.asarray(d["a"], float)[kt], np.asarray(d["val_a"], float)[kv]
    zh_raw, vzh_raw = np.asarray(d["z_H"], float)[kt], np.asarray(d["val_z_H"], float)[kv]
    zm_raw = {n: np.asarray(d["z_M"][i], float)[kt] for i, n in enumerate(names)}
    vzm_raw = {n: np.asarray(d["val_z_M"][i], float)[kv] for i, n in enumerate(names)}
    half = est.split_halves(d["image"][kt], y)
    std = lambda z, vz: z / est.within_class_sd(vz, vy)                                     # noqa: E731
    zh, zm, zp = std(zh_raw, vzh_raw), std(zm_raw["M"], vzm_raw["M"]), std(zm_raw["Mramo"], vzm_raw["Mramo"])
    a1 = a / est.within_class_sd(a, y)
    slope = float(est.slope_within(zh, a1, y))
    h = zh - slope * a1
    sd_h = est.within_class_sd(h, y)
    beta, d_h, d_m = slope / sd_h, float(est.d_a(h, y)), float(est.d_a(zm, y))
    d_tr, r = float(est.d_a(a1, y)), float(est.pearson_within(h, zm, y))
    th1 = theorem.with_soloist_informative(d_h, d_m, r, beta, d_tr)
    th2 = theorem.with_partner_informative(d_h, d_m, r, d_tr)
    c, c0 = sum_branch(vzh_raw, vzm_raw["M"], va, vy)
    new = std(zm_raw["M"] + c * a, vzm_raw["M"] + c * va)
    b_raw = float(est.slope_within(zh_raw, a, y))                       # the whole push of the judge, raw log-odds per unit of frame
    d1 = best_linear(np.column_stack([zh, zm]), y, half)
    out = {"kept_test": int(kt.sum()), "kept_validation": int(kv.sum()), "d_a_frame": d_tr, "beta": beta,
           "beta_deserved": theorem.deserved_weight(d_h, d_m, r, d_tr), "d_h_clean": d_h, "d_m": d_m, "r": r,
           "theorem_d1": th1, "theorem_d2": th2, "measured_d1": d1,
           "measured_d2_three_cues": best_linear(np.column_stack([zh, zm, a1]), y, half),
           "pair_with_partner_of_e8": best_linear(np.column_stack([zh, zp]), y, half),
           "pair_with_new_branch": best_linear(np.column_stack([zh, new]), y, half),
           "push_of_the_judge_raw": b_raw, "push_removed_by_new_branch_raw": -c,
           "share_of_the_push_removed_by_new_branch": -c / b_raw}
    out["excess_share_of_the_push"] = 1.0 - out["beta_deserved"] / beta
    gain = th2 ** 2 - th1 ** 2
    for k in ("measured_d2_three_cues", "pair_with_partner_of_e8", "pair_with_new_branch"):
        out["share_of_gain_" + k] = (out[k] ** 2 - d1 ** 2) / gain if gain > 0.05 else None
    return out


def unit(path) -> dict:
    d = raw.read_chart(path)
    gen = np.random.default_rng([SEED, d["seed"], d["v"], sum(map(ord, d["task"]))])
    return {"task": d["task"], "seed": d["seed"], "v": d["v"], "by_d_a": {f"{t:g}": one(d, t, gen) for t in TARGETS}}


def summary(units) -> list:
    rows = []
    for v in raw.CHART_VIEWS:
        for t in TARGETS:
            us = [u["by_d_a"][f"{t:g}"] for u in units if u["v"] == v]
            if not us:
                continue
            row = {"v": v, "target_d_a": t, "units": len(us)}
            for k in us[0]:
                row[k] = med([u[k] for u in us])
            row["units_where_new_branch_beats_partner_of_e8"] = sum(u["pair_with_new_branch"] > u["pair_with_partner_of_e8"] for u in us)
            row["units_where_new_branch_beats_soloist"] = sum(u["pair_with_new_branch"] > u["measured_d1"] for u in us)
            rows.append(row)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", type=Path, default=ROOT / "results" / "raw" / "chart")
    ap.add_argument("--out", type=Path, default=ROOT / "results" / "e8_informative.json")
    ap.add_argument("--workers", type=int, default=3)
    args = ap.parse_args()
    timer = Timer()
    files = raw.list_units(args.raw, raw.CHART_RE)
    if not files:
        raise SystemExit(f"no chart files in {args.raw}")
    units = run_pool(unit, files, args.workers)
    out = {"experiment": "E8, when the trace says something true", "protocol": "§ 14, on the files of E8, resampled",
           "status": "a reading added on 20 November 2026, after the results of E8 were seen: not a test. The frame is made "
                     "informative by resampling the presentations, and the new branch is linear",
           "how": "presentations kept with a probability that depends on frame and class, so that the kept frames have "
                  f"sd {SPREAD:g} within a class and sensitivity 0, 0.5 or 1; the theorem of "
                  "cognitive_resonance.chart (informative trace) from the judge and the soloist; pairs as the best "
                  "linear use of the cues, weights from one half of the images and d_a on the other",
           "settings": {"targets": list(TARGETS), "spread": SPREAD}, "by_v_and_d_a": summary(units), "units": units,
           "minutes": timer.minutes(), "environment": environment()}
    write_json(args.out, out)
    for r in out["by_v_and_d_a"]:
        print("v", r["v"], "d_a", r["target_d_a"], "| d_a meas", round(r["d_a_frame"], 2), "beta", round(r["beta"], 2), "deserved", round(r["beta_deserved"], 2),
              "| D1", round(r["measured_d1"], 2), "th", round(r["theorem_d1"], 2), "| D2 three", round(r["measured_d2_three_cues"], 2), "th", round(r["theorem_d2"], 2),
              "| old partner", round(r["pair_with_partner_of_e8"], 2), "new branch", round(r["pair_with_new_branch"], 2),
              "| push removed", round(r["share_of_the_push_removed_by_new_branch"], 2), "excess", round(r["excess_share_of_the_push"], 2),
              "| new>old", r["units_where_new_branch_beats_partner_of_e8"], "/", r["units"])
    print("written", args.out)


if __name__ == "__main__":
    main()
