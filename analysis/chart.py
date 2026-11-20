"""E8 (protocol § 14): the bench of the chart, and the theorem that goes with it.

Reads results/raw/chart/*.npz. The judge H (plus, mu = 1) sees the whole image; the machine sees
the frame and all of the inside of the image (v = 100), its left half (50) or its left quarter
(25). Three partners for the same judge: M (the soloist), Mramo (M plus a frame-only branch
trained on the sum with H) and Mblind (sees only the frame, trained on the sum with H). H0 is
the judge without the bias. Estimators, grid and bootstrap of § 3.

For every unit the theorem of `cognitive_resonance.chart` is fed with quantities measured on
that unit: the slope of H's cue on the frame gives the bias, what is left of the cue once the
bias is taken out is the clean cue h, and from h and M come d_h, d_m, r and B. Its predictions
are compared with the pairs actually measured. Writes every measure and K1-K7.

    python analysis/chart.py [--raw results/raw/chart] [--out results/e8_chart.json]

With `--e2` it reads instead the files of E2 bis and E2 ter (results/raw/rephase_bis and
rephase_ter): the theorem on runs made before it was written. A look at old data, not a test.

With `--views` it reads the files of E7 (results/raw/views): the same theorem on a bench it was
not registered for, where the machine too sees things the judge does not. Added after the
results of E7 were seen: not a test.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from common import ROOT, Timer, environment, med, run_pool, verdict, write_json
from cognitive_resonance import chart as theorem
from cognitive_resonance import estimators as est
from cognitive_resonance import raw
from fixed import accuracy, lower_above_zero, paired, upper_below_zero
from rephase import _need
from rephase_bis import measure, per_task

VIEWS = raw.CHART_VIEWS                       # 100, 50, 25
SMALL = 0.03                                  # K1: with the whole view, |gain| below this share of the pair with M
LAW = (0.8, 1.25)                             # K4: measured over predicted gain
AGREE = 0.8                                   # K6: share of units where the verdict of the threshold is the measured one
NOTHING_NEW = 1.05                            # K7: the pair with the partner over the pair of the judge without bias


def clean_judge(zh, a, y):
    """The judge's cue split in two: the bias (its slope on the frame) and what is left, h."""
    slope = float(est.slope_within(zh, a, y))
    h = zh - slope * a
    var_h = est.within_class_sd(h, y) ** 2
    var_bias = (slope * est.within_class_sd(a, y)) ** 2
    return h, slope, var_bias / var_h


def ideal_three(zh, zm, a, y, half) -> float:
    """The best linear use of (z_H, z_M, a): weights from one half of the images, d_a read on the other, swapped."""
    X = np.column_stack([zh, zm, a])
    out = []
    for fit, read in ((0, 1), (1, 0)):
        f, r = half == fit, half == read
        mean1, mean0 = X[f & (y == 1)].mean(axis=0), X[f & (y == 0)].mean(axis=0)
        res = np.vstack([X[f & (y == 1)] - mean1, X[f & (y == 0)] - mean0])
        w = np.linalg.solve(res.T @ res / (res.shape[0] - 2), mean1 - mean0)
        out.append(float(est.d_a(X[r] @ w, y[r])))
    return float(np.mean(out))


def ideal_two(zh, a, y, half) -> float:
    """The judge with an ideal blind corrector: the best linear use of (z_H, a), fitted on one half, read on the other."""
    X = np.column_stack([zh, a])
    out = []
    for fit, read in ((0, 1), (1, 0)):
        f, r = half == fit, half == read
        mean1, mean0 = X[f & (y == 1)].mean(axis=0), X[f & (y == 0)].mean(axis=0)
        res = np.vstack([X[f & (y == 1)] - mean1, X[f & (y == 0)] - mean0])
        w = np.linalg.solve(res.T @ res / (res.shape[0] - 2), mean1 - mean0)
        out.append(float(est.d_a(X[r] @ w, y[r])))
    return float(np.mean(out))


def predictions(zh, zm, a, y) -> dict:
    """The theorem with the quantities of one unit."""
    h, slope, b = clean_judge(zh, a, y)
    d_h, d_m = float(est.d_a(h, y)), float(est.d_a(zm, y))
    r = float(est.pearson_within(h, zm, y))
    lo, hi = theorem.blind_band(r, b)
    return {"slope_of_h_on_a": slope, "b": b, "d_h_clean": d_h, "d_m": d_m, "r": r,
            "d1": theorem.with_soloist(d_h, d_m, r, b), "d2": theorem.with_partner(d_h, d_m, r),
            "d_blind": d_h, "private": theorem.private(d_h, d_m, r), "gain_squared": theorem.gain(d_h, d_m, r, b),
            "blind_band": [lo, hi], "strength": d_m / d_h, "blind_beats_soloist": bool(lo < d_m / d_h < hi)}


def unit(job):
    path, reps = job
    d = raw.read_chart(path)
    machines = list(d["machine_names"])
    design, point, judges, boot = measure(d, machines, reps)
    y, a = d["y"], np.asarray(d["a"], float)
    zh, zm = design.cues["H"], design.cues["M"]
    zh0 = est.standardize(d["z_H0"], y, d["val_z_H0"], d["val_y"])
    pred = predictions(zh, zm, a, y)
    d1, d2, dc = (point[n]["d_pair_tuned"] for n in ("M", "Mramo", "Mblind"))
    three = ideal_three(zh, zm, a, y, design.half)
    without_bias = est.measure_pair(zh0, zm, y, design.half)
    gain = paired(d2, d1, boot[("Mramo", "d_pair_tuned")], boot[("M", "d_pair_tuned")])
    return {"task": d["task"], "seed": d["seed"], "v": d["v"], "file": d["file"], "machines": machines,
            "n": int(y.size), "n_images": design.n_images, "judges": judges, "pairs": point, "theorem": pred,
            "accuracy": {"H": accuracy(d["z_H"], y), "H0": accuracy(d["z_H0"], y),
                         **{n: accuracy(d["z_M"][machines.index(n)], y) for n in machines},
                         **{f"H+{n}": accuracy(d["z_H"] + d["z_M"][machines.index(n)], y) for n in machines}},
            "measured": {"d1": d1, "d2_partner": d2, "d2_three_cues": three, "d_blind": dc,
                         "d_pair_without_bias": without_bias["d_pair_tuned"], "d_h0": float(est.d_a(zh0, y)),
                         "gain_squared_three_cues": three ** 2 - d1 ** 2, "gain_squared_partner": d2 ** 2 - d1 ** 2},
            "gain": gain,
            "partner_alone_minus_soloist": paired(judges["Mramo"]["d_a"], judges["M"]["d_a"],
                                                  boot[("Mramo", "d_a")], boot[("M", "d_a")]),
            "blind_minus_soloist_pair": paired(dc, d1, boot[("Mblind", "d_pair_tuned")], boot[("M", "d_pair_tuned")])}


def criteria(units) -> dict:
    tasks = sorted({u["task"] for u in units})
    need = _need(len(tasks), 3)
    at = {v: [u for u in units if u["v"] == v] for v in VIEWS}
    out = {}
    gain = {v: med([u["gain"]["diff"] for u in at[v]]) for v in VIEWS}
    share = med([abs(u["gain"]["diff"]) / u["measured"]["d1"] for u in at[100]])
    out["K1"] = verdict(share is not None and share <= SMALL, median_abs_gain_over_pair=share, at_most=SMALL,
                        median_gain=gain[100],
                        rule="with the whole view (v = 100), |d_pair(H, Mramo) - d_pair(H, M)| / d_pair(H, M), median "
                             f"over the units, is at most {SMALL}")

    k2 = per_task(at[25], lambda u: lower_above_zero(u["gain"]))
    out["K2"] = verdict(sum(k2.values()) >= need, tasks_passing=sum(k2.values()), tasks_needed=need, by_task=k2,
                        median_gain=gain[25],
                        median_pair_with_partner=med([u["measured"]["d2_partner"] for u in at[25]]),
                        median_pair_with_soloist=med([u["measured"]["d1"] for u in at[25]]),
                        rule="at v = 25 the tuned pair (H, Mramo) is more sensitive than the tuned pair (H, M): interval "
                             "of the difference above zero in every seed, in >= 3 tasks of 5")

    grows = all(g is not None for g in gain.values()) and gain[25] > gain[50] > gain[100]
    out["K3"] = verdict(grows, median_gain_by_v={str(v): gain[v] for v in VIEWS},
                        rule="the median of that difference grows as the machine sees less: v = 25 > 50 > 100")

    law = {v: med([u["measured"]["gain_squared_three_cues"] / u["theorem"]["gain_squared"] for u in at[v]
                   if u["theorem"]["gain_squared"] > 0]) for v in (50, 25)}
    out["K4"] = verdict(all(x is not None and LAW[0] <= x <= LAW[1] for x in law.values()),
                        median_measured_over_predicted={str(v): x for v, x in law.items()}, band=list(LAW),
                        median_predicted_d2={str(v): med([u["theorem"]["d2"] for u in at[v]]) for v in VIEWS},
                        median_measured_d2_three_cues={str(v): med([u["measured"]["d2_three_cues"] for u in at[v]])
                                                       for v in VIEWS},
                        median_measured_d2_partner={str(v): med([u["measured"]["d2_partner"] for u in at[v]])
                                                    for v in VIEWS},
                        rule="at v = 50 and v = 25, (D2^2 - D1^2) measured with the best linear use of (z_H, z_M, a) "
                             f"over the gain predicted by the theorem, median over the units, within {list(LAW)}")

    k5 = per_task(at[25], lambda u: upper_below_zero(u["partner_alone_minus_soloist"]) and lower_above_zero(u["gain"]))
    out["K5"] = verdict(sum(k5.values()) >= need, tasks_passing=sum(k5.values()), tasks_needed=need, by_task=k5,
                        median_alone_difference=med([u["partner_alone_minus_soloist"]["diff"] for u in at[25]]),
                        rule="at v = 25 Mramo alone is less sensitive than M alone and the pair with it more sensitive, "
                             "both intervals excluding zero in every seed, in >= 3 tasks of 5")

    same = [(u["blind_minus_soloist_pair"]["diff"] > 0) == u["theorem"]["blind_beats_soloist"] for u in units]
    out["K6"] = verdict(np.mean(same) >= AGREE, share_of_units=float(np.mean(same)), at_least=AGREE, units=len(same),
                        blind_wins_by_v={str(v): sum(u["blind_minus_soloist_pair"]["diff"] > 0 for u in at[v])
                                         for v in VIEWS},
                        predicted_by_v={str(v): sum(u["theorem"]["blind_beats_soloist"] for u in at[v]) for v in VIEWS},
                        units_by_v={str(v): len(at[v]) for v in VIEWS},
                        rule="which of (H, Mblind) and (H, M) is the more sensitive pair is what the threshold of the "
                             f"theorem says, with the unit's own quantities, in at least {AGREE:.0%} of the units")

    ratio = {v: med([u["measured"]["d2_partner"] / u["measured"]["d_pair_without_bias"] for u in at[v]]) for v in VIEWS}
    out["K7"] = verdict(all(x is not None and x <= NOTHING_NEW for x in ratio.values()),
                        median_partner_pair_over_pair_without_bias={str(v): x for v, x in ratio.items()},
                        at_most=NOTHING_NEW,
                        rule="the tuned pair (H, Mramo) over the tuned pair (H0, M), median over the units, is at most "
                             f"{NOTHING_NEW} at every v")
    return out


def summary(units) -> list:
    rows = []
    for v in VIEWS:
        us = [u for u in units if u["v"] == v]
        if not us:
            continue
        m = lambda f: med([f(u) for u in us])                                           # noqa: E731
        rows.append({"v": v, "units": len(us), "accuracy_h": m(lambda u: u["accuracy"]["H"]),
                     "accuracy_h0": m(lambda u: u["accuracy"]["H0"]), "accuracy_m": m(lambda u: u["accuracy"]["M"]),
                     "accuracy_mramo": m(lambda u: u["accuracy"]["Mramo"]),
                     "accuracy_mblind": m(lambda u: u["accuracy"]["Mblind"]),
                     "accuracy_h_plus_m": m(lambda u: u["accuracy"]["H+M"]),
                     "accuracy_h_plus_mramo": m(lambda u: u["accuracy"]["H+Mramo"]),
                     "accuracy_h_plus_mblind": m(lambda u: u["accuracy"]["H+Mblind"]),
                     "d_h": m(lambda u: u["judges"]["H"]["d_a"]), "d_m": m(lambda u: u["judges"]["M"]["d_a"]),
                     "d_mramo": m(lambda u: u["judges"]["Mramo"]["d_a"]), "d_h0": m(lambda u: u["measured"]["d_h0"]),
                     "phi_m": m(lambda u: u["pairs"]["M"]["phi"]), "phi_mramo": m(lambda u: u["pairs"]["Mramo"]["phi"]),
                     "d1": m(lambda u: u["measured"]["d1"]), "d2_partner": m(lambda u: u["measured"]["d2_partner"]),
                     "d2_three_cues": m(lambda u: u["measured"]["d2_three_cues"]),
                     "d_blind": m(lambda u: u["measured"]["d_blind"]),
                     "d_pair_without_bias": m(lambda u: u["measured"]["d_pair_without_bias"]),
                     "gain": m(lambda u: u["gain"]["diff"]),
                     "theorem_d1": m(lambda u: u["theorem"]["d1"]), "theorem_d2": m(lambda u: u["theorem"]["d2"]),
                     "theorem_d_blind": m(lambda u: u["theorem"]["d_blind"]), "b": m(lambda u: u["theorem"]["b"]),
                     "r": m(lambda u: u["theorem"]["r"]), "strength": m(lambda u: u["theorem"]["strength"]),
                     "blind_threshold": m(lambda u: u["theorem"]["blind_band"][1]),
                     "private": m(lambda u: u["theorem"]["private"])})
    return rows


# --- the theorem on the runs of E2 bis and E2 ter (a look at old data) -----------------------------------------

def old_unit(job):
    path, kind = job
    d = raw.read_rephase_bis(path) if kind == "bis" else raw.read_rephase_ter(path)
    y, a = d["y"], np.asarray(d["a"], float)
    names = d["machine_names"]
    cue = lambda z, vz: est.standardize(z, y, vz, d["val_y"])                                # noqa: E731
    zh = cue(d["z_H"], d["val_z_H"])
    zm = {n: cue(d["z_M"][i], d["val_z_M"][i]) for i, n in enumerate(names)}
    half = est.split_halves(d["image"], y)
    soloist = "M0plus" if kind == "ter" else "M0"
    pred = predictions(zh, zm[soloist], a, y)
    tuned = {n: est.tuned_turing(zh, zm[n], y, half)["d_pair_tuned"] for n in names}
    return {"task": d["task"], "seed": d["seed"], "experiment": "E2 " + kind, "soloist": soloist, "theorem": pred,
            "measured": {"d1": tuned[soloist], **{f"pair_with_{n}": v for n, v in tuned.items()},
                         "d2_three_cues": ideal_three(zh, zm[soloist], a, y, half)}}


def old_summary(units) -> dict:
    out = {}
    for kind, partner, blind in (("E2 bis", None, "Mframe"), ("E2 ter", "Mramo", None)):
        us = [u for u in units if u["experiment"] == kind]
        m = lambda f: med([f(u) for u in us])                                           # noqa: E731
        row = {"units": len(us), "b": m(lambda u: u["theorem"]["b"]), "r": m(lambda u: u["theorem"]["r"]),
               "d_h_clean": m(lambda u: u["theorem"]["d_h_clean"]), "d_m": m(lambda u: u["theorem"]["d_m"]),
               "theorem_d1": m(lambda u: u["theorem"]["d1"]), "measured_d1": m(lambda u: u["measured"]["d1"]),
               "theorem_d2": m(lambda u: u["theorem"]["d2"]),
               "measured_d2_three_cues": m(lambda u: u["measured"]["d2_three_cues"]),
               "strength": m(lambda u: u["theorem"]["strength"]),
               "blind_threshold": m(lambda u: u["theorem"]["blind_band"][1]),
               "gain_over_d1_percent": m(lambda u: 100 * (u["theorem"]["d2"] / u["theorem"]["d1"] - 1))}
        if partner:
            row["measured_d2_partner"] = m(lambda u: u["measured"][f"pair_with_{partner}"])
            row["median_partner_over_theorem"] = m(lambda u: u["measured"][f"pair_with_{partner}"] / u["theorem"]["d2"])
        if blind:
            row["theorem_d_blind"] = m(lambda u: u["theorem"]["d_blind"])
            row["measured_d_blind"] = m(lambda u: u["measured"][f"pair_with_{blind}"])
            row["median_blind_over_theorem"] = m(lambda u: u["measured"][f"pair_with_{blind}"] / u["theorem"]["d_blind"])
            row["units_where_soloist_wins"] = sum(u["measured"]["d1"] > u["measured"][f"pair_with_{blind}"] for u in us)
            row["units_where_theorem_says_soloist"] = sum(not u["theorem"]["blind_beats_soloist"] for u in us)
        out[kind] = row
    return out


def views_unit(path):
    d = raw.read_views(path)
    y, a = d["y"], np.asarray(d["a"], float)
    names = d["machine_names"]
    cue = lambda z, vz: est.standardize(z, y, vz, d["val_y"])                                # noqa: E731
    zh = cue(d["z_H"], d["val_z_H"])
    zm = {n: cue(d["z_M"][i], d["val_z_M"][i]) for i, n in enumerate(names)}
    half = est.split_halves(d["image"], y)
    tuned = {n: est.tuned_turing(zh, zm[n], y, half)["d_pair_tuned"] for n in names}
    return {"task": d["task"], "seed": d["seed"], "overlap": d["overlap"], "theorem": predictions(zh, zm["M0"], a, y),
            "measured": {"d_h": float(est.d_a(zh, y)), "d1": tuned["M0"], "pair_with_Mramo": tuned["Mramo"],
                         "pair_with_Mpair": tuned["Mpair"], "d2_three_cues": ideal_three(zh, zm["M0"], a, y, half),
                         "d_blind_two_cues": ideal_two(zh, a, y, half)}}


def views_summary(units) -> list:
    rows = []
    for o in raw.VIEWS_OVERLAPS:
        us = [u for u in units if u["overlap"] == o]
        if not us:
            continue
        m = lambda f: med([f(u) for u in us])                                           # noqa: E731
        t_, me = (lambda u: u["theorem"]), (lambda u: u["measured"])
        taken = [(me(u)["pair_with_Mramo"] - me(u)["d1"]) / (t_(u)["d2"] - t_(u)["d1"]) for u in us
                 if t_(u)["d2"] - t_(u)["d1"] > 0.05]
        rows.append({
            "overlap": o, "units": len(us), "b": m(lambda u: t_(u)["b"]), "r": m(lambda u: t_(u)["r"]),
            "d_h": m(lambda u: me(u)["d_h"]), "d_h_clean": m(lambda u: t_(u)["d_h_clean"]), "d_m": m(lambda u: t_(u)["d_m"]),
            "theorem_d1": m(lambda u: t_(u)["d1"]), "measured_d1": m(lambda u: me(u)["d1"]),
            "theorem_d2": m(lambda u: t_(u)["d2"]), "measured_d2_three_cues": m(lambda u: me(u)["d2_three_cues"]),
            "measured_d2_partner": m(lambda u: me(u)["pair_with_Mramo"]),
            "measured_pair_from_scratch": m(lambda u: me(u)["pair_with_Mpair"]),
            "theorem_d_blind": m(lambda u: t_(u)["d_blind"]), "measured_d_blind_two_cues": m(lambda u: me(u)["d_blind_two_cues"]),
            "strength": m(lambda u: t_(u)["strength"]), "blind_threshold": m(lambda u: t_(u)["blind_band"][1]),
            "median_partner_over_theorem": m(lambda u: me(u)["pair_with_Mramo"] / t_(u)["d2"]),
            "median_share_of_predicted_gain_taken": med(taken) if taken else None,
            "units_with_a_predicted_gain_above_0.05": len(taken),
            "median_partner_minus_blind": m(lambda u: me(u)["pair_with_Mramo"] - me(u)["d_blind_two_cues"]),
            "units_where_partner_beats_blind": sum(me(u)["pair_with_Mramo"] > me(u)["d_blind_two_cues"] for u in us),
            "units_where_partner_beats_soloist": sum(me(u)["pair_with_Mramo"] > me(u)["d1"] for u in us),
            "units_where_blind_beats_soloist": sum(me(u)["d_blind_two_cues"] > me(u)["d1"] for u in us),
            "units_where_theorem_says_blind": sum(bool(t_(u)["blind_beats_soloist"]) for u in us)})
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--reps", type=int, default=1000)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--e2", action="store_true", help="the theorem on the files of E2 bis and E2 ter: not a test")
    ap.add_argument("--views", action="store_true", help="the theorem on the files of E7, added after its results: not a test")
    args = ap.parse_args()
    timer = Timer()
    if args.views:
        folder = args.raw or ROOT / "results" / "raw" / "views"
        files = raw.list_units(folder, raw.VIEWS_RE)
        if not files:
            raise SystemExit(f"no views files in {folder}")
        units = run_pool(views_unit, files, args.workers)
        out = {"experiment": "E7, read with the theorem of the chart", "protocol": "§ 14 (v2.1), on the files of § 13",
               "status": "a reading added on 20 November 2026, after the results of E7 were seen: not a test",
               "how": "per unit: the slope of H's cue on the frame gives the bias B, the cue without it is h; d_h, d_m "
                      "and r from h and the soloist M0; D1, D2 and the threshold from cognitive_resonance.chart. The "
                      "blind corrector is not a trained network here: it is the best linear use of (z_H, a), fitted "
                      "on one half of the images and read on the other",
               "by_overlap": views_summary(units), "units": units, "minutes": timer.minutes(), "environment": environment()}
        write_json(args.out or ROOT / "results" / "chart_from_e7.json", out)
        for row in out["by_overlap"]:
            print({a: (round(b, 3) if isinstance(b, float) else b) for a, b in row.items()})
        return
    if args.e2:
        root = args.raw or ROOT / "results" / "raw"
        jobs = [(f, "bis") for f in raw.list_units(root / "rephase_bis", raw.REPHASE_RE)]
        jobs += [(f, "ter") for f in raw.list_units(root / "rephase_ter", raw.REPHASE_RE)]
        if not jobs:
            raise SystemExit(f"no rephase_bis or rephase_ter files in {root}")
        units = run_pool(old_unit, jobs, args.workers)
        out = {"experiment": "E2 bis and E2 ter, read with the theorem of the chart", "protocol": "§ 14 (v2.1)",
               "status": "a look at runs made before the theorem was written: not a test",
               "how": "per unit: the slope of H's cue on the frame gives the bias B, the cue without it is h; d_h, d_m "
                      "and r from h and the soloist; D1, D2 and the threshold from cognitive_resonance.chart",
               "summary": old_summary(units), "units": units, "minutes": timer.minutes(), "environment": environment()}
        write_json(args.out or ROOT / "results" / "chart_from_e2.json", out)
        for k, v in out["summary"].items():
            print(k, {a: (round(b, 3) if isinstance(b, float) else b) for a, b in v.items()})
        return
    folder = args.raw or ROOT / "results" / "raw" / "chart"
    files = raw.list_units(folder, raw.CHART_RE)
    if not files:
        raise SystemExit(f"no chart files in {folder}")
    units = run_pool(unit, [(f, args.reps) for f in files], args.workers)
    out = {"experiment": "E8", "protocol": "§ 14 (v2.1)", "status": "test registered before the run",
           "raw": Path(folder).name, "manifest": raw.read_manifest(folder),
           "settings": {"reps": args.reps, "grid": "49 weights, 1/64 to 64", "views": list(VIEWS),
                        "clean_cue": "H's standardized cue minus its within-class slope on the frame times the frame",
                        "three_cues": "weights of the best linear use of (z_H, z_M, a) from one half of the images, "
                                      "d_a read on the other, swapped"},
           "criteria": criteria(units), "by_v": summary(units), "units": units,
           "minutes": timer.minutes(), "environment": environment()}
    write_json(args.out or ROOT / "results" / "e8_chart.json", out)
    for h, v in out["criteria"].items():
        print(h, v["outcome"])
    print("minutes", out["minutes"])


if __name__ == "__main__":
    main()
