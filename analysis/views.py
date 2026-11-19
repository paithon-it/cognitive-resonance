"""E7 (protocol § 13): different views.

Reads results/raw/views/*.npz. Each judge sees a vertical strip of the image, half of it, and
both see the frame; the machine's strip overlaps the person's by 100, 50 or 0 per cent. The
person H (plus, mu = 1) is the same network at the three overlaps. Three machines per overlap:
M0 (the soloist), Mramo (M0 frozen plus a frame-only branch trained on the sum with H) and
Mpair (trained from scratch on the sum with H). Estimators, grid and bootstrap of § 3, plus
the pair at face value of `fixed.py`. Writes every measure, the paired differences from M0
and V1-V6.

    python analysis/views.py [--raw results/raw/views] [--out results/e7_views.json]
"""
from __future__ import annotations

import argparse
from pathlib import Path

from common import ROOT, Timer, environment, med, run_pool, verdict, write_json
from cognitive_resonance import estimators as est
from cognitive_resonance import raw
from fixed import STD_WEIGHTS, fixed_measures, lower_above_zero, paired, std_key, upper_below_zero
from rephase import _need
from rephase_bis import measure, per_task

SOLOIST = "M0"
OVERLAPS = raw.VIEWS_OVERLAPS                 # 100, 50, 0
LEARNS = 0.7                                  # V4: Mpair alone over M0 alone where the views do not overlap, at least
DOES_NOT = 0.3                                # V4: the same where the views are the same, at most
SAME_VIEW_SHARE = 1 / 3                       # V2: the gain with the same view, as a share of the gain without overlap
SCORES = tuple(std_key(w) for w in STD_WEIGHTS) + ("face", "alone")


def unit(job):
    path, reps = job
    d = raw.read_views(path)
    machines = list(d["machine_names"])
    design, point, judges, boot = measure(d, machines, reps)
    fpoint, fboot, dh = fixed_measures(d, design.cues, design, machines, reps)
    for name in machines:
        fpoint[name]["tu"] = {k: fpoint[name]["d_a"][k] / dh for k in SCORES if k != "alone"}
    diffs, against_soloist = {}, {}
    for name in machines:
        # the tuned pair with this machine against the soloist alone (V5); for M0 it is the pair against itself alone
        draws = boot[(name, "d_pair_tuned")] - boot[(SOLOIST, "d_a")]
        against_soloist[name] = {"diff": point[name]["d_pair_tuned"] - judges[SOLOIST]["d_a"], "ci": est.ci(draws)}
        if name == SOLOIST:
            continue
        diffs[name] = {
            "pair_tuned": paired(point[name]["d_pair_tuned"], point[SOLOIST]["d_pair_tuned"],
                                 boot[(name, "d_pair_tuned")], boot[(SOLOIST, "d_pair_tuned")]),
            "alone": paired(judges[name]["d_a"], judges[SOLOIST]["d_a"], boot[(name, "d_a")], boot[(SOLOIST, "d_a")]),
            "phi": paired(point[name]["phi"], point[SOLOIST]["phi"], boot[(name, "phi")], boot[(SOLOIST, "phi")]),
            "face": paired(fpoint[name]["d_a"]["face"], fpoint[SOLOIST]["d_a"]["face"],
                           fboot[(name, "face")], fboot[(SOLOIST, "face")])}
    ratio = judges["Mpair"]["d_a"] / judges[SOLOIST]["d_a"] if judges[SOLOIST]["d_a"] > 0 else None
    return {"task": d["task"], "seed": d["seed"], "overlap": d["overlap"], "file": d["file"], "machines": machines,
            "n": int(d["y"].size), "n_images": design.n_images, "d_h": dh, "judges": judges, "pairs": point,
            "fixed": fpoint, "differences": diffs, "pair_against_soloist_alone": against_soloist,
            "mpair_alone_over_soloist_alone": ratio}


def _ordered(values) -> bool:
    """True if the values, given in the order 0, 50, 100 per cent of overlap, strictly decrease."""
    return all(v is not None for v in values) and values[0] > values[1] > values[2]


def criteria(units) -> dict:
    tasks = sorted({u["task"] for u in units})
    need = _need(len(tasks), 3)
    at = {o: [u for u in units if u["overlap"] == o] for o in OVERLAPS}
    out = {}

    def by_task(o, test):
        res = per_task(at[o], test)
        return res, sum(res.values())

    gain = {o: med([u["differences"]["Mramo"]["pair_tuned"]["diff"] for u in at[o]]) for o in OVERLAPS}
    v1 = {o: by_task(o, lambda u: lower_above_zero(u["differences"]["Mramo"]["pair_tuned"])) for o in OVERLAPS}
    out["V1"] = verdict(v1[0][1] >= need, tasks_passing=v1[0][1], tasks_needed=need, by_task=v1[0][0],
                        median_difference=gain[0],
                        median_pair_with_partner=med([u["pairs"]["Mramo"]["d_pair_tuned"] for u in at[0]]),
                        median_pair_with_soloist=med([u["pairs"][SOLOIST]["d_pair_tuned"] for u in at[0]]),
                        rule="where the views do not overlap, the tuned pair (H, Mramo) is more sensitive than the tuned "
                             "pair (H, M0): interval of the difference above zero in every seed, in >= 3 tasks of 5")

    out["V2"] = verdict(_ordered([gain[0], gain[50], gain[100]]) and gain[100] <= SAME_VIEW_SHARE * gain[0],
                        median_difference_by_overlap={str(o): gain[o] for o in OVERLAPS},
                        same_view_over_no_overlap=gain[100] / gain[0] if gain[0] else None,
                        at_most=SAME_VIEW_SHARE,
                        tasks_passing_by_overlap={str(o): v1[o][1] for o in OVERLAPS}, tasks_needed=need,
                        rule="the median of that difference grows as the overlap falls (0 > 50 > 100 per cent), and "
                             f"where the views are the same it is at most {SAME_VIEW_SHARE:g} of what it is where "
                             "they do not overlap")

    v3, n3 = by_task(0, lambda u: upper_below_zero(u["differences"]["Mramo"]["alone"]))
    out["V3"] = verdict(n3 >= need and out["V1"]["outcome"] == "pass", tasks_passing=n3, tasks_needed=need, by_task=v3,
                        median_alone_difference=med([u["differences"]["Mramo"]["alone"]["diff"] for u in at[0]]),
                        v1=out["V1"]["outcome"],
                        rule="where the views do not overlap, Mramo alone is less sensitive than M0 alone (interval "
                             "below zero in every seed, in >= 3 tasks of 5) and V1 passes")

    learn = {o: med([u["mpair_alone_over_soloist_alone"] for u in at[o]]) for o in OVERLAPS}
    out["V4"] = verdict(learn[0] is not None and learn[100] is not None and learn[0] >= LEARNS
                        and learn[100] <= DOES_NOT and _ordered([learn[0], learn[50], learn[100]]),
                        median_mpair_alone_over_soloist_alone={str(o): learn[o] for o in OVERLAPS},
                        at_least_without_overlap=LEARNS, at_most_with_the_same_view=DOES_NOT,
                        median_mpair_alone={str(o): med([u["judges"]["Mpair"]["d_a"] for u in at[o]]) for o in OVERLAPS},
                        rule=f"d_a(Mpair alone) / d_a(M0 alone), median over the units: >= {LEARNS} where the views do "
                             f"not overlap, <= {DOES_NOT} where they are the same, and decreasing with the overlap")

    v5 = {o: by_task(o, lambda u: u["pair_against_soloist_alone"]["Mramo"]["ci"] is not None
                     and u["pair_against_soloist_alone"]["Mramo"]["ci"][0] > 0) for o in OVERLAPS}
    out["V5"] = verdict(v5[0][1] >= need, tasks_passing=v5[0][1], tasks_needed=need, by_task=v5[0][0],
                        tasks_passing_by_overlap={str(o): v5[o][1] for o in OVERLAPS},
                        median_difference_by_overlap={
                            str(o): med([u["pair_against_soloist_alone"]["Mramo"]["diff"] for u in at[o]])
                            for o in OVERLAPS},
                        rule="where the views do not overlap, the tuned pair (H, Mramo) is more sensitive than M0 "
                             "alone: interval above zero in every seed, in >= 3 tasks of 5")

    v6, n6 = by_task(0, lambda u: lower_above_zero(u["differences"]["Mramo"]["face"]))
    out["V6"] = verdict(n6 >= need, tasks_passing=n6, tasks_needed=need, by_task=v6,
                        median_difference_by_overlap={
                            str(o): med([u["differences"]["Mramo"]["face"]["diff"] for u in at[o]]) for o in OVERLAPS},
                        rule="where the views do not overlap, at face value (raw log-odds added) the pair (H, Mramo) "
                             "is more sensitive than the pair (H, M0): interval above zero in every seed, in >= 3 "
                             "tasks of 5")
    return out


def summary(units) -> list:
    rows = []
    for o in OVERLAPS:
        us = [u for u in units if u["overlap"] == o]
        for name in us[0]["machines"] if us else []:
            rows.append({"overlap": o, "machine": name, "d_h": med([u["d_h"] for u in us]),
                         "d_a_alone": med([u["judges"][name]["d_a"] for u in us]),
                         "slope_on_a": med([u["judges"][name]["slope_on_a"] for u in us]),
                         "k": med([u["pairs"][name]["k"] for u in us]), "phi": med([u["pairs"][name]["phi"] for u in us]),
                         "tu_app": med([u["pairs"][name]["tu_app_nonneg"] for u in us]),
                         "tu_tuned": med([u["pairs"][name]["tu_tuned"] for u in us]),
                         "d_pair_tuned": med([u["pairs"][name]["d_pair_tuned"] for u in us]),
                         "w_star_model": med([u["pairs"][name]["w_star"] for u in us]),
                         "tu_face": med([u["fixed"][name]["tu"]["face"] for u in us]),
                         "tu_std1": med([u["fixed"][name]["tu"][std_key(1.0)] for u in us]),
                         "pair_minus_soloist_alone": med([u["pair_against_soloist_alone"][name]["diff"] for u in us]),
                         "accuracy_alone": med([u["fixed"][name]["accuracy"]["alone"] for u in us]),
                         "accuracy_face": med([u["fixed"][name]["accuracy"]["face"] for u in us]),
                         "accuracy_person_alone": med([u["fixed"][name]["accuracy"]["person_alone"] for u in us])})
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", type=Path, default=ROOT / "results" / "raw" / "views")
    ap.add_argument("--out", type=Path, default=ROOT / "results" / "e7_views.json")
    ap.add_argument("--reps", type=int, default=1000)
    ap.add_argument("--workers", type=int, default=3)
    args = ap.parse_args()
    files = sorted(args.raw.glob("*.npz"))
    if not files:
        raise SystemExit(f"no raw files in {args.raw}")
    timer = Timer()
    units = run_pool(unit, [(f, args.reps) for f in files], args.workers)
    out = {"experiment": "E7", "protocol": "§ 13 (v2.0)", "status": "test registered before the run",
           "raw": Path(args.raw).name, "manifest": raw.read_manifest(args.raw),
           "settings": {"reps": args.reps, "grid": "49 weights, 1/64 to 64", "overlaps": list(OVERLAPS),
                        "soloist": SOLOIST, "differences": "machine minus M0 at the same overlap, paired on the same "
                                                           "resamples"},
           "criteria": criteria(units), "by_overlap_and_machine": summary(units), "units": units,
           "minutes": timer.minutes(), "environment": environment()}
    write_json(args.out, out)
    for h, v in out["criteria"].items():
        print(h, v["outcome"])
    print("minutes", out["minutes"])


if __name__ == "__main__":
    main()
