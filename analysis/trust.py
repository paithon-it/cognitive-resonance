"""E6 (protocol § 12): trust made to measure.

Reads results/raw/trust/*.npz: the person H (plus, mu = 1) and seven machines: M0, M0plus (the
control soloist), Mdue, and four Mramo_w, the frozen M0 plus a frame-only branch built for the
pair that gives the machine w times the weight of the person (w = 0.5, 1, 2, 4, raw log-odds).
Estimators, grid and bootstrap of § 3, plus the measures at a fixed weight of `fixed.py`.
Writes every measure, the paired differences from M0plus and F1-F7.

    python analysis/trust.py [--raw results/raw/trust] [--out results/e6_trust.json]

With `--ter` it reads the files of E2 ter instead (results/raw/rephase_ter): that is where the
hypotheses of E6 come from. It is a look at old data, not a test: only F1-F4 can be computed,
and the output says so.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from common import ROOT, Timer, environment, med, run_pool, verdict, write_json
from cognitive_resonance import estimators as est
from cognitive_resonance import model, raw
from fixed import (STD_WEIGHTS, fixed_measures, lower_above_zero, paired, std_key, upper_below_zero)
from rephase import _need
from rephase_bis import measure, per_task

CONTROL = "M0plus"
PEAK_FACTOR = 1.5                 # F5: the measured peak within this factor of the weight the branch is built for
PEAK_WEIGHTS = (0.5, 1.0, 2.0)    # F5 judges the place of the peak at these weights; at 4 the curve is nearly flat
SAME_PEAK = (0.97, 1.03)          # F6: tuned turing over the control's
FACE_TOLERANCE = 0.05             # F4: the pair at face value may stay this much (d_a) under the soloist alone
HIGH = 8.0                        # F3: the weight of someone who trusts the machine too much
SCORES = tuple(std_key(w) for w in STD_WEIGHTS) + ("face", "alone")


def branches_of(d) -> dict:
    """Branch machine -> the weight it is built for. The files of E2 ter have one branch, built for 1."""
    if "built_for_weight" in d:
        return {f"Mramo_w{float(w):g}": float(w) for w in d["built_for_weight"]}
    return {"Mramo": 1.0}


def unit(job):
    path, reps, old = job
    d = raw.read_rephase_ter(path) if old else raw.read_trust(path)
    machines = list(d["machine_names"])
    design, point, judges, boot = measure(d, machines, reps)
    fpoint, fboot, dh = fixed_measures(d, design.cues, design, machines, reps)
    for name in machines:
        p = fpoint[name]
        p["tu"] = {k: p["d_a"][k] / dh for k in SCORES if k != "alone"}
        p["d_a_ci"] = {k: est.ci(fboot[(name, k)]) for k in SCORES}
        # the pair at face value against the control soloist alone, and against the person alone
        p["face_minus_control_alone"] = paired(p["d_a"]["face"], fpoint[CONTROL]["d_a"]["alone"],
                                               fboot[(name, "face")], fboot[(CONTROL, "alone")])
        p["tuned_over_control"] = point[name]["tu_tuned"] / point[CONTROL]["tu_tuned"]
    diffs = {}
    for name in machines:
        if name == CONTROL:
            continue
        diffs[name] = {k: paired(fpoint[name]["d_a"][k], fpoint[CONTROL]["d_a"][k],
                                 fboot[(name, k)], fboot[(CONTROL, k)]) for k in SCORES}
        diffs[name]["pair_tuned"] = paired(point[name]["d_pair_tuned"], point[CONTROL]["d_pair_tuned"],
                                           boot[(name, "d_pair_tuned")], boot[(CONTROL, "d_pair_tuned")])
        diffs[name]["phi"] = paired(point[name]["phi"], point[CONTROL]["phi"],
                                    boot[(name, "phi")], boot[(CONTROL, "phi")])
    return {"task": d["task"], "seed": d["seed"], "file": d["file"], "machines": machines,
            "branches": branches_of(d), "n": int(d["y"].size), "n_images": design.n_images,
            "d_h": dh, "judges": judges, "pairs": point, "fixed": fpoint, "differences": diffs}


def _spearman(x, y):
    return est.spearman(x, y)


def criteria(units) -> dict:
    tasks = sorted({u["task"] for u in units})
    need = _need(len(tasks), 3)
    branches = units[0]["branches"]
    one = next(n for n, w in branches.items() if w == 1.0)          # the branch built for weight one
    out = {}

    def by_task(test):
        res = per_task(units, test)
        return res, sum(res.values())

    k1 = std_key(1.0)
    f1, n1 = by_task(lambda u: lower_above_zero(u["differences"][one][k1]))
    out["F1"] = verdict(n1 >= need, tasks_passing=n1, tasks_needed=need, by_task=f1,
                        units_with_positive_difference=sum(u["differences"][one][k1]["diff"] > 0 for u in units),
                        units=len(units), median_difference=med([u["differences"][one][k1]["diff"] for u in units]),
                        median_tu_partner=med([u["fixed"][one]["tu"][k1] for u in units]),
                        median_tu_soloist=med([u["fixed"][CONTROL]["tu"][k1] for u in units]),
                        rule="at standardized weight 1, d_a of the pair with the branch built for weight 1 minus d_a "
                             "of the pair with M0plus: interval above zero in every seed, in >= 3 tasks of 5")

    f2, n2 = by_task(lambda u: lower_above_zero(u["differences"][one]["face"]))
    out["F2"] = verdict(n2 >= need, tasks_passing=n2, tasks_needed=need, by_task=f2,
                        median_difference=med([u["differences"][one]["face"]["diff"] for u in units]),
                        median_tu_partner=med([u["fixed"][one]["tu"]["face"] for u in units]),
                        median_tu_soloist=med([u["fixed"][CONTROL]["tu"]["face"] for u in units]),
                        rule="the same at face value (raw log-odds added, weight 1)")

    kh = std_key(HIGH)
    f3, n3 = by_task(lambda u: upper_below_zero(u["differences"][one][kh]))
    out["F3"] = verdict(n3 >= need, tasks_passing=n3, tasks_needed=need, by_task=f3,
                        median_difference=med([u["differences"][one][kh]["diff"] for u in units]),
                        median_tu_partner=med([u["fixed"][one]["tu"][kh] for u in units]),
                        median_tu_soloist=med([u["fixed"][CONTROL]["tu"][kh] for u in units]),
                        rule=f"at standardized weight {HIGH:g} the pair with M0plus is the more sensitive: interval of "
                             "(partner minus soloist) below zero in every seed, in >= 3 tasks of 5")

    f4, n4 = by_task(lambda u: upper_below_zero(u["fixed"][CONTROL]["face_minus_control_alone"]))
    m4 = med([u["fixed"][one]["face_minus_control_alone"]["diff"] for u in units])
    out["F4"] = verdict(n4 >= need and m4 is not None and m4 >= -FACE_TOLERANCE, tasks_passing=n4, tasks_needed=need,
                        by_task=f4,
                        median_soloist_pair_minus_soloist_alone=med(
                            [u["fixed"][CONTROL]["face_minus_control_alone"]["diff"] for u in units]),
                        median_partner_pair_minus_soloist_alone=m4, tolerance=FACE_TOLERANCE,
                        rule="at face value the pair (H, M0plus) is less sensitive than M0plus alone (interval below "
                             "zero in every seed, in >= 3 tasks of 5), and the pair with the branch built for weight 1 "
                             f"is not: median of (pair minus M0plus alone) >= -{FACE_TOLERANCE}")

    if len(branches) < 4:
        for h in ("F5", "F6", "F7"):
            out[h] = verdict(None, reason="needs the four branches of E6: these files have one")
        return out

    names = sorted(branches, key=branches.get)
    ws = [branches[n] for n in names]
    centre = {n: [u["fixed"][n]["raw_peak"]["estimate"] for u in units] for n in names}
    open_ = {n: sum(u["fixed"][n]["raw_peak"]["centre"] is None for u in units) for n in names}
    ratio = {n: med([c / branches[n] for c in centre[n]]) for n in names}
    in_place = {n: 1.0 / PEAK_FACTOR <= ratio[n] <= PEAK_FACTOR for n in names if branches[n] in PEAK_WEIGHTS}
    rank = [_spearman(ws, [u["fixed"][n]["raw_peak"]["estimate"] for n in names]) for u in units]
    mr = med(rank)
    out["F5"] = verdict(all(in_place.values()) and mr is not None and mr >= 0.9,
                        median_peak_over_built_weight=ratio, within_factor=in_place, factor=PEAK_FACTOR,
                        median_peak={n: med(centre[n]) for n in names}, units_with_open_centre=open_,
                        median_spearman_weight_peak=mr, spearman_by_unit=rank,
                        rule=f"the raw weight at which the pair (H, Mramo_w) peaks is, in median over the units, within "
                             f"a factor {PEAK_FACTOR} of w for w = 0.5, 1 and 2, and the median over the units of the "
                             "rank correlation between w and the peak, over the four branches, is >= 0.9")

    same = {n: med([u["fixed"][n]["tuned_over_control"] for u in units]) for n in names}
    out["F6"] = verdict(all(v is not None and SAME_PEAK[0] <= v <= SAME_PEAK[1] for v in same.values()),
                        median_tuned_over_control=same, band=list(SAME_PEAK),
                        rule="tuned turing of (H, Mramo_w) over that of (H, M0plus), median over the units, within "
                             f"[{SAME_PEAK[0]}, {SAME_PEAK[1]}] for each of the four weights")

    alone_rank = [_spearman(ws, [u["fixed"][n]["d_a"]["alone"] for n in names]) for u in units]
    ma = med(alone_rank)
    low = names[0]
    f7, n7 = by_task(lambda u: upper_below_zero(u["differences"][low]["alone"]))
    out["F7"] = verdict(ma is not None and ma >= 0.9 and n7 >= need, median_spearman_weight_alone=ma,
                        tasks_passing=n7, tasks_needed=need, by_task=f7,
                        median_alone={n: med([u["fixed"][n]["d_a"]["alone"] for u in units]) for n in names},
                        median_control_alone=med([u["fixed"][CONTROL]["d_a"]["alone"] for u in units]),
                        rule="the machine alone is worth less the lower the weight it is built for: median rank "
                             "correlation between w and d_a alone >= 0.9, and the branch built for 0.5 is less "
                             "sensitive alone than M0plus, interval below zero in every seed, in >= 3 tasks of 5")
    return out


def summary_by_machine(units) -> list:
    rows = []
    for name in units[0]["machines"]:
        row = {"machine": name, "built_for_weight": units[0]["branches"].get(name),
               "d_a_alone": med([u["fixed"][name]["d_a"]["alone"] for u in units]),
               "k": med([u["pairs"][name]["k"] for u in units]), "phi": med([u["pairs"][name]["phi"] for u in units]),
               "w_star_model": med([u["pairs"][name]["w_star"] for u in units]),
               "tu_tuned": med([u["pairs"][name]["tu_tuned"] for u in units]),
               "raw_peak": med([u["fixed"][name]["raw_peak"]["estimate"] for u in units]),
               "sd_ratio_m_over_h": med([u["fixed"][name]["sd_ratio_m_over_h"] for u in units]),
               "face_minus_control_alone": med([u["fixed"][name]["face_minus_control_alone"]["diff"] for u in units])}
        for k in SCORES:
            if k != "alone":
                row["tu_" + k] = med([u["fixed"][name]["tu"][k] for u in units])
        rows.append(row)
    return rows


def crossings(units, name) -> list:
    """Standardized weight at which the pair with `name` and the pair with the control are worth the same."""
    out = []
    lw = np.log(model.GRID)
    for u in units:
        a = np.array(u["pairs"][name]["curve_measured"]) - np.array(u["pairs"][CONTROL]["curve_measured"])
        idx = np.flatnonzero(np.diff(np.sign(a)) != 0)
        out.append([float(np.exp(lw[i] + (0 - a[i]) / (a[i + 1] - a[i]) * (lw[i + 1] - lw[i]))) for i in idx])
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--reps", type=int, default=1000)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--ter", action="store_true", help="read the files of E2 ter: a look at old data, not a test")
    args = ap.parse_args()
    folder = args.raw or ROOT / "results" / "raw" / ("rephase_ter" if args.ter else "trust")
    target = args.out or ROOT / "results" / ("e6_trust_from_e2ter.json" if args.ter else "e6_trust.json")
    files = sorted(folder.glob("*.npz"))
    if not files:
        raise SystemExit(f"no raw files in {folder}")
    timer = Timer()
    units = run_pool(unit, [(f, args.reps, args.ter) for f in files], args.workers)
    one = next(n for n, w in units[0]["branches"].items() if w == 1.0)
    out = {"experiment": "E6", "protocol": "§ 12 (v2.0)",
           "status": ("a look at the data of E2 ter, from which the hypotheses of E6 were written: not a test"
                      if args.ter else "test registered before the run"),
           "raw": Path(folder).name, "manifest": raw.read_manifest(folder),
           "settings": {"reps": args.reps, "grid": "49 weights, 1/64 to 64",
                        "standardized_weights": list(STD_WEIGHTS), "control": CONTROL,
                        "peak": "middle, in ln w, of the weights where Tu^2 - 1 is at least 0.9 of its maximum; the "
                                "weight of the maximum where a crossing is outside the grid",
                        "differences": "machine minus M0plus, paired on the same resamples"},
           "criteria": criteria(units), "by_machine": summary_by_machine(units),
           "crossing_weight_partner_vs_soloist": crossings(units, one), "units": units,
           "minutes": timer.minutes(), "environment": environment()}
    write_json(target, out)
    for h, v in out["criteria"].items():
        print(h, v["outcome"])
    print("minutes", out["minutes"])


if __name__ == "__main__":
    main()
