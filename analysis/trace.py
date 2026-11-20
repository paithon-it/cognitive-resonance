"""E10 and E11 (protocol §§ 16-17): the chart that tells the truth, and what a partner should remove.

Reads results/raw/trace/*.npz. At test the frame has a sensitivity of its own (0, 0.5, 1 or 1.5).
The judge H was trained where it was worth 2 and gives it too much weight. Beside it: the soloist
M (it does not see the frame), M plus the branch built for the level of the test ("new") and M
plus the branch built where the frame said nothing ("old": the partner of E8, which removes the
whole push). C is the judge trained with frames of the level of the test.

Per unit and per level: the share of the judge's push that each branch removes, the excess the
theorem says is there (`cognitive_resonance.chart`, quantities measured as in `informative.py`),
the tuned pairs of § 3 and the pairs at face value (raw log-odds summed, weight 1), with paired
bootstrap differences. Writes every measure, T1-T6 and X1-X5.

    python analysis/trace.py [--raw results/raw/trace] [--out results/e10_trace.json]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from common import ROOT, Timer, environment, med, run_pool, verdict, write_json
from cognitive_resonance import chart as theorem
from cognitive_resonance import estimators as est
from cognitive_resonance import raw
from fixed import accuracy, fixed_measures, lower_above_zero, paired, upper_below_zero
from judgments import best_linear
from rephase import _need
from rephase_bis import measure, per_task

VIEWS = raw.TRACE_VIEWS                       # 25, 100
LEVELS = raw.TRACE_LEVELS                     # 0, 0.5, 1, 1.5
MACHINES = ("M", "new", "old")
REMOVES_ALL = (0.9, 1.1)                      # T1: share of the push removed at level 0
CLOSE = 0.10                                  # T2: |share removed - excess of the theorem|
FAR = 0.20                                    # T3: share removed minus excess, with the whole view
NOTHING = 0.03                                # T5: relative difference at face value where the weight is almost right
NOTHING_NEW = 1.05                            # T6: the pair with the partner over the pair of the calibrated judge
MADE_UP = 0.02                                # X4: relative difference between the tuned pairs with new and old


def key(level: float) -> str:
    return f"{level:g}"


def level_view(d: dict, k: int) -> dict:
    """One level of a trace file in the layout of the other benches: H, and the machines M, new and old."""
    out = {"task": d["task"], "seed": d["seed"], "v": d["v"], "machine_names": list(MACHINES)}
    for p in ("", "val_"):
        zm = np.asarray(d[p + "z_M"], float)
        out[p + "y"], out[p + "image"], out[p + "a"] = d[p + "y"], d[p + "image"], np.asarray(d[p + "a"][k], float)
        out[p + "z_H"] = np.asarray(d[p + "z_H"][k], float)
        out[p + "z_M"] = np.stack([zm, zm + d[p + "z_R"][k][k], zm + d[p + "z_R"][0][k]])
    return out


def theorem_quantities(zh, zm, a, y) -> dict:
    """The quantities of the theorem with an informative trace, from standardized cues (as informative.py)."""
    a1 = a / est.within_class_sd(a, y)
    slope = float(est.slope_within(zh, a1, y))
    h = zh - slope * a1
    beta = slope / est.within_class_sd(h, y)
    d_h, d_m, d_tr = float(est.d_a(h, y)), float(est.d_a(zm, y)), float(est.d_a(a1, y))
    r = float(est.pearson_within(h, zm, y))
    deserved = theorem.deserved_weight(d_h, d_m, r, d_tr)
    d1 = theorem.with_soloist_informative(d_h, d_m, r, beta, d_tr)
    d2 = theorem.with_partner_informative(d_h, d_m, r, d_tr)
    return {"beta": beta, "beta_deserved": deserved, "excess": 1.0 - deserved / beta, "d_h_clean": d_h, "d_m": d_m,
            "d_a_frame": d_tr, "r": r, "d1": d1, "d2": d2, "gain_squared": d2 ** 2 - d1 ** 2}


def one_level(d: dict, k: int, reps: int) -> dict:
    lv = level_view(d, k)
    y, a = lv["y"], lv["a"]
    design, point, judges, boot = measure(lv, list(MACHINES), reps)
    fpoint, fboot, _ = fixed_measures(lv, design.cues, design, list(MACHINES), reps)
    zh, zm = design.cues["H"], design.cues["M"]
    th = theorem_quantities(zh, zm, a, y)
    push = float(est.slope_within(lv["z_H"], a, y))                      # raw log-odds per unit of frame
    removed = {name: -float(est.slope_within(np.asarray(d["z_R"][t][k], float), a, y)) / push
               for name, t in (("new", k), ("old", 0))}
    zc = est.standardize(d["z_C"][k], y, d["val_z_C"][k], d["val_y"])
    with_c = est.measure_pair(zc, zm, y, design.half)["d_pair_tuned"]
    tuned = {n: point[n]["d_pair_tuned"] for n in MACHINES}
    face = {n: fpoint[n]["d_a"]["face"] for n in MACHINES}
    three = best_linear(np.column_stack([zh, zm, a / est.within_class_sd(a, y)]), y, design.half)
    diff_t = lambda p, q: paired(tuned[p], tuned[q], boot[(p, "d_pair_tuned")], boot[(q, "d_pair_tuned")])   # noqa: E731
    diff_f = lambda p, q: paired(face[p], face[q], fboot[(p, "face")], fboot[(q, "face")])                    # noqa: E731
    gain = tuned["new"] ** 2 - tuned["M"] ** 2
    return {"theorem": th, "share_removed": removed, "push_raw": push,
            "tuned": {**tuned, "calibrated_judge_with_soloist": with_c, "three_cues": three},
            "face": face,
            "right": {"H": accuracy(lv["z_H"], y), "C": accuracy(d["z_C"][k], y),
                      "M": fpoint["M"]["accuracy"]["alone"], **{f"H+{n}": fpoint[n]["accuracy"]["face"] for n in MACHINES}},
            "alone": {n: judges[n]["d_a"] for n in MACHINES} | {"H": judges["H"]["d_a"], "C": float(est.d_a(zc, y))},
            "share_of_predicted_gain": gain / th["gain_squared"] if th["gain_squared"] > 0.2 else None,
            "tuned_new_minus_soloist": diff_t("new", "M"), "tuned_new_minus_old": diff_t("new", "old"),
            "face_new_minus_old": diff_f("new", "old"), "face_old_minus_soloist": diff_f("old", "M"),
            "face_new_minus_soloist": diff_f("new", "M")}


def unit(job) -> dict:
    path, reps = job
    d = raw.read_trace(path)
    return {"task": d["task"], "seed": d["seed"], "v": d["v"], "file": d["file"], "n": int(d["y"].size),
            "levels": {key(level): one_level(d, k, reps) for k, level in enumerate(LEVELS)}}


def criteria(units) -> dict:
    tasks = sorted({u["task"] for u in units})
    need = _need(len(tasks), 3)
    at = {v: [u for u in units if u["v"] == v] for v in VIEWS}
    L = lambda u, level: u["levels"][key(level)]                                                  # noqa: E731
    m = lambda v, level, f: med([f(L(u, level)) for u in at[v]])                                 # noqa: E731
    out = {}

    s = {key(x): m(25, x, lambda r: r["share_removed"]["new"]) for x in LEVELS}
    ordered = all(x is not None for x in s.values()) and s["0"] > s["0.5"] > s["1"] > s["1.5"]
    out["T1"] = verdict(ordered and REMOVES_ALL[0] <= s["0"] <= REMOVES_ALL[1], median_share_removed=s,
                        band_at_level_0=list(REMOVES_ALL),
                        rule="at v = 25 the median share of the push the branch removes is within "
                             f"{list(REMOVES_ALL)} at level 0 and falls as the level grows")

    gap = {key(x): m(25, x, lambda r: abs(r["share_removed"]["new"] - r["theorem"]["excess"])) for x in (0.5, 1.0)}
    out["T2"] = verdict(all(g is not None and g <= CLOSE for g in gap.values()), median_abs_gap=gap, at_most=CLOSE,
                        median_share_removed={k_: s[k_] for k_ in gap},
                        median_excess={key(x): m(25, x, lambda r: r["theorem"]["excess"]) for x in (0.5, 1.0)},
                        rule="at v = 25 and levels 0.5 and 1, |share removed - excess of the theorem|, median over "
                             f"the units, is at most {CLOSE}")

    far = m(100, 0.5, lambda r: r["share_removed"]["new"] - r["theorem"]["excess"])
    out["T3"] = verdict(far is not None and far > FAR, median_removed_minus_excess=far, above=FAR,
                        median_share_removed=m(100, 0.5, lambda r: r["share_removed"]["new"]),
                        median_excess=m(100, 0.5, lambda r: r["theorem"]["excess"]),
                        rule=f"at v = 100 and level 0.5 the median of share removed minus excess is above {FAR}: the "
                             "theorem's excess does not describe the branch there (an expectation taken from old data)")

    t4 = per_task(at[25], lambda u: lower_above_zero(L(u, 0.5)["tuned_new_minus_soloist"]))
    out["T4"] = verdict(sum(t4.values()) >= need, tasks_passing=sum(t4.values()), tasks_needed=need, by_task=t4,
                        median_gain=m(25, 0.5, lambda r: r["tuned_new_minus_soloist"]["diff"]),
                        rule="at v = 25 and level 0.5 the tuned pair with the partner built for the level is more "
                             "sensitive than the tuned pair with the soloist: interval above zero in every seed, in "
                             ">= 3 tasks of 5")

    rel = m(25, 1.5, lambda r: r["face_new_minus_soloist"]["diff"] / r["face"]["M"])
    out["T5"] = verdict(rel is not None and abs(rel) <= NOTHING, median_relative_difference=rel, within=NOTHING,
                        rule="at v = 25 and level 1.5, at face value, (pair with the partner - pair with the soloist) "
                             f"/ pair with the soloist is in the median within +-{NOTHING}")

    ratio = {key(x): m(25, x, lambda r: r["tuned"]["new"] / r["tuned"]["calibrated_judge_with_soloist"]) for x in LEVELS}
    out["T6"] = verdict(all(x is not None and x <= NOTHING_NEW for x in ratio.values()),
                        median_partner_pair_over_calibrated_pair=ratio, at_most=NOTHING_NEW,
                        rule="at v = 25 and every level the tuned pair (H, M + R_d) over the tuned pair (C_d, M) is "
                             f"in the median at most {NOTHING_NEW}")

    def new_beats_old(v):
        by = per_task(at[v], lambda u: lower_above_zero(L(u, 1.0)["face_new_minus_old"]))
        return verdict(sum(by.values()) >= need, tasks_passing=sum(by.values()), tasks_needed=need, by_task=by,
                       median_difference=m(v, 1.0, lambda r: r["face_new_minus_old"]["diff"]),
                       units_where_new_is_ahead=sum(L(u, 1.0)["face_new_minus_old"]["diff"] > 0 for u in at[v]),
                       units=len(at[v]),
                       rule=f"at v = {v} and level 1, at face value, the pair with M + R_1 is more sensitive than "
                            "the pair with M + R_0: interval above zero in every seed, in >= 3 tasks of 5")

    out["X1"] = new_beats_old(25)
    x2 = per_task(at[25], lambda u: upper_below_zero(L(u, 1.5)["face_old_minus_soloist"]))
    out["X2"] = verdict(sum(x2.values()) >= need, tasks_passing=sum(x2.values()), tasks_needed=need, by_task=x2,
                        median_difference=m(25, 1.5, lambda r: r["face_old_minus_soloist"]["diff"]),
                        units_where_old_is_behind=sum(L(u, 1.5)["face_old_minus_soloist"]["diff"] < 0 for u in at[25]),
                        units=len(at[25]),
                        rule="at v = 25 and level 1.5, at face value, the pair with M + R_0 is less sensitive than "
                             "the pair with M: interval below zero in every seed, in >= 3 tasks of 5")

    dose = {key(x): m(25, x, lambda r: r["face_new_minus_old"]["diff"]) for x in (0.5, 1.0, 1.5)}
    grows = all(x is not None for x in dose.values()) and 0 < dose["0.5"] < dose["1"] < dose["1.5"]
    out["X3"] = verdict(grows, median_new_minus_old=dose,
                        rule="at v = 25 the median difference at face value between new and old is positive at "
                             "levels 0.5, 1 and 1.5 and grows with the level")

    made_up = {key(x): m(25, x, lambda r: abs(r["tuned_new_minus_old"]["diff"]) / r["tuned"]["old"]) for x in (0.5, 1.0)}
    out["X4"] = verdict(all(x is not None and x <= MADE_UP for x in made_up.values()),
                        median_abs_relative_difference=made_up, at_most=MADE_UP,
                        median_signed_difference={key(x): m(25, x, lambda r: r["tuned_new_minus_old"]["diff"])
                                                  for x in (0.5, 1.0)},
                        rule="at v = 25 and levels 0.5 and 1 the tuned pairs with new and old differ in the median, "
                             f"in absolute value, by at most {MADE_UP} of the second")
    out["X5"] = new_beats_old(100)
    return out


def summary(units) -> list:
    rows = []
    for v in VIEWS:
        for level in LEVELS:
            us = [u["levels"][key(level)] for u in units if u["v"] == v]
            if not us:
                continue
            m = lambda f: med([f(u) for u in us])                                                 # noqa: E731
            cnt = lambda f: sum(bool(f(u)) for u in us)                                           # noqa: E731
            rng = lambda f: [min(f(u) for u in us), max(f(u) for u in us)]                        # noqa: E731
            rows.append({
                "v": v, "level": level, "units": len(us),
                **{f"theorem_{k_}": m(lambda u, k_=k_: u["theorem"][k_]) for k_ in us[0]["theorem"]},
                "share_removed_new": m(lambda u: u["share_removed"]["new"]),
                "share_removed_new_range": rng(lambda u: u["share_removed"]["new"]),
                "share_removed_old": m(lambda u: u["share_removed"]["old"]),
                **{f"tuned_{k_}": m(lambda u, k_=k_: u["tuned"][k_]) for k_ in us[0]["tuned"]},
                **{f"face_{k_}": m(lambda u, k_=k_: u["face"][k_]) for k_ in us[0]["face"]},
                **{f"right_{k_}": m(lambda u, k_=k_: u["right"][k_]) for k_ in us[0]["right"]},
                **{f"alone_{k_}": m(lambda u, k_=k_: u["alone"][k_]) for k_ in us[0]["alone"]},
                "share_of_predicted_gain": m(lambda u: u["share_of_predicted_gain"]),
                **{k_: m(lambda u, k_=k_: u[k_]["diff"]) for k_ in ("tuned_new_minus_soloist", "tuned_new_minus_old",
                                                                    "face_new_minus_old", "face_old_minus_soloist",
                                                                    "face_new_minus_soloist")},
                "face_new_minus_old_range": rng(lambda u: u["face_new_minus_old"]["diff"]),
                "face_old_minus_soloist_range": rng(lambda u: u["face_old_minus_soloist"]["diff"]),
                "units_where_new_beats_old_at_face_value": cnt(lambda u: u["face_new_minus_old"]["diff"] > 0),
                "units_where_old_beats_soloist_at_face_value": cnt(lambda u: u["face_old_minus_soloist"]["diff"] > 0),
                "units_where_new_beats_soloist_at_face_value": cnt(lambda u: u["face_new_minus_soloist"]["diff"] > 0),
                "units_where_new_beats_old_tuned": cnt(lambda u: u["tuned_new_minus_old"]["diff"] > 0),
                "right_new_minus_old_points": m(lambda u: 100 * (u["right"]["H+new"] - u["right"]["H+old"])),
                "right_new_minus_old_points_range": rng(lambda u: 100 * (u["right"]["H+new"] - u["right"]["H+old"])),
                "units_where_calibrated_judge_alone_beats_pair_with_new": cnt(lambda u: u["right"]["C"] > u["right"]["H+new"]),
            })
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", type=Path, default=ROOT / "results" / "raw" / "trace")
    ap.add_argument("--out", type=Path, default=ROOT / "results" / "e10_trace.json")
    ap.add_argument("--reps", type=int, default=1000)
    ap.add_argument("--workers", type=int, default=3)
    args = ap.parse_args()
    timer = Timer()
    files = raw.list_units(args.raw, raw.TRACE_RE)
    if not files:
        raise SystemExit(f"no trace files in {args.raw}")
    units = run_pool(unit, [(f, args.reps) for f in files], args.workers)
    out = {"experiment": "E10 and E11, the chart that tells the truth", "protocol": "§§ 16-17 (v2.3)",
           "status": "registered before the run",
           "manifest": raw.read_manifest(args.raw) is not None,
           "settings": {"reps": args.reps, "levels": list(LEVELS), "views": list(VIEWS),
                        "machines": {"M": "the soloist, no frame", "new": "M + the branch built for the level of the test",
                                     "old": "M + the branch built at level 0 (the partner of E8)"}},
           "criteria": criteria(units), "by_v_and_level": summary(units), "units": units,
           "minutes": timer.minutes(), "environment": environment()}
    write_json(args.out, out)
    for name, c in out["criteria"].items():
        print(name, c["outcome"])
    print("minutes", out["minutes"], "written", args.out)


if __name__ == "__main__":
    main()
