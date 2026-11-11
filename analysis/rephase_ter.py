"""E2 ter (protocol § 6 ter, v1.4, exploratory): the soloist that learns the person.

Reads results/raw/rephase_ter/*.npz: the person H (plus, mu = 1) and four machines, in the
order given by `machines`: M0 (normal training, as in E2 and E2 bis), M0plus (M0 fine-tuned
for 10 more epochs with its own loss: the control), Mdue (M0 fine-tuned the same way on its
own cross-entropy plus that of the sum z_H + z_M) and Mramo (M0 frozen plus a branch that
sees only the frame, trained on the sum z_H + z_M0 + z_branch; its cue is z_M0 + z_branch).
Same estimators, grid and bootstrap as E2 bis, whose functions are reused. Writes every
measure, the paired differences from M0plus (and from M0), the tuned pair against M0plus
alone, and T1-T5.

    python analysis/rephase_ter.py [--raw results/raw/rephase_ter] [--out results/e2ter_rephase.json]
"""
from __future__ import annotations

import argparse
from pathlib import Path

from common import ROOT, Timer, environment, med, run_pool, verdict, write_json
from cognitive_resonance import estimators as est
from cognitive_resonance import raw
from rephase import _need
from rephase_bis import _opposite, by_machine, differences, measure, per_task

MACHINES = raw.REPHASE_TER_MACHINES
CONTROL = "M0plus"
TUNED = ("Mdue", "Mramo")          # the two machines judged by T2-T4


def unit(job):
    path, reps = job
    d = raw.read_rephase_ter(path)
    design, point, judges, boot = measure(d, MACHINES, reps)
    diffs = differences(point, judges, boot, [m for m in MACHINES if m != CONTROL], CONTROL)
    diffs_m0 = differences(point, judges, boot, [m for m in MACHINES if m != "M0"], "M0")
    alone = {}
    for name in MACHINES:              # the tuned pair (H, M) against the control machine alone (T5)
        draws = boot[(name, "d_pair_tuned")] - boot[(CONTROL, "d_a")]
        alone[name] = {"pair_minus_control_alone": point[name]["d_pair_tuned"] - judges[CONTROL]["d_a"],
                       "pair_minus_control_alone_ci": est.ci(draws)}
    return {"task": d["task"], "seed": d["seed"], "file": d["file"], "machines": list(MACHINES),
            "n": int(d["y"].size), "n_images": design.n_images, "judges": judges, "pairs": point,
            "differences": diffs, "differences_from_M0": diffs_m0, "pair_against_control_alone": alone}


def _lower_above_zero(ci) -> bool:
    return ci is not None and ci[0] > 0


def criteria(units) -> dict:
    tasks = sorted({u["task"] for u in units})
    need = _need(len(tasks), 3)
    out = {}

    rows = []
    for u in units:
        j = u["judges"]
        ratio = j["Mdue"]["d_a"] / j[CONTROL]["d_a"] if j[CONTROL]["d_a"] > 0 else None
        rows.append({"unit": f"{u['task']} s{u['seed']}", "d_a_Mdue": j["Mdue"]["d_a"], "d_a_M0plus": j[CONTROL]["d_a"],
                     "ratio": ratio, "slope_Mdue": j["Mdue"]["slope_on_a"], "slope_H": j["H"]["slope_on_a"],
                     "slope_opposite_to_h": _opposite(j["Mdue"]["slope_on_a"], j["H"]["slope_on_a"])})
    r1 = med([r["ratio"] for r in rows])
    opposite = sum(r["slope_opposite_to_h"] for r in rows)
    defined = all(r["ratio"] is not None for r in rows)          # M0plus at or below chance: the ratio has no sense
    out["T1"] = verdict(defined and r1 is not None and r1 >= 0.9 and opposite == len(rows),
                        median_ratio_d_a_Mdue_over_M0plus=r1, units=len(rows), slope_opposite_to_h=opposite,
                        detail=rows,
                        rule="median over units of d_a(Mdue) / d_a(M0plus) >= 0.9, and the slope of Mdue's cue on a "
                             "opposite in sign to H's in every unit (point estimates)")

    for m in TUNED:
        t2 = per_task(units, lambda u, m=m: _lower_above_zero(u["differences"][m]["phi_ci"]))
        out[f"T2 {m}"] = verdict(sum(t2.values()) >= need, tasks_passing=sum(t2.values()), tasks_needed=need,
                                 by_task=t2, median_phi_difference=med([u["differences"][m]["phi"] for u in units]),
                                 rule=f"phi(H, {m}) - phi(H, M0plus) with its interval above zero in every seed, "
                                      "in >= 3 tasks of 5")
    for m in TUNED:
        t3 = per_task(units, lambda u, m=m: _lower_above_zero(u["differences"][m]["pair_ci"]))
        out[f"T3 {m}"] = verdict(sum(t3.values()) >= need, tasks_passing=sum(t3.values()), tasks_needed=need,
                                 by_task=t3, median_pair_difference=med([u["differences"][m]["pair"] for u in units]),
                                 median_d_pair_M=med([u["pairs"][m]["d_pair_tuned"] for u in units]),
                                 median_d_pair_M0plus=med([u["pairs"][CONTROL]["d_pair_tuned"] for u in units]),
                                 rule=f"d_a of the tuned pair (H, {m}) minus that of (H, M0plus), with its interval "
                                      "above zero in every seed, in >= 3 tasks of 5")
    t4 = {}
    for m in TUNED:
        r = med([u["pairs"][m]["ratio_tuned_app_nonneg"] for u in units])
        t4[m] = {"median_ratio_tuned_over_tu_app": r, "within_10_percent": r is not None and 0.9 <= r <= 1.1,
                 "median_tuned_turing": med([u["pairs"][m]["tu_tuned"] for u in units]),
                 "median_tu_app": med([u["pairs"][m]["tu_app_nonneg"] for u in units])}
    out["T4"] = verdict(all(v["within_10_percent"] for v in t4.values()), by_machine=t4,
                        rule="for Mdue and for Mramo, median tuned turing / Tu_app from k and phi in [0.9, 1.1] "
                             "(Tu_app: maximum over w >= 0); both must hold")
    t5 = per_task(units, lambda u: _lower_above_zero(u["pair_against_control_alone"]["Mdue"]
                                                     ["pair_minus_control_alone_ci"]))
    out["T5"] = verdict(sum(t5.values()) >= need, tasks_passing=sum(t5.values()), tasks_needed=need, by_task=t5,
                        median_difference=med([u["pair_against_control_alone"]["Mdue"]["pair_minus_control_alone"]
                                               for u in units]),
                        median_difference_Mramo=med([u["pair_against_control_alone"]["Mramo"]
                                                     ["pair_minus_control_alone"] for u in units]),
                        rule="d_a of the tuned pair (H, Mdue) minus d_a of M0plus alone, with its interval above zero "
                             "in every seed, in >= 3 tasks of 5 (Mramo reported, not judged)")
    return out


def summary_by_machine(units) -> dict:
    out = by_machine(units, MACHINES, CONTROL)
    for m in MACHINES:
        out[m]["pair_minus_M0plus_alone"] = med([u["pair_against_control_alone"][m]["pair_minus_control_alone"]
                                                 for u in units])
        if m != "M0":
            out[m]["difference_from_M0"] = {k: med([u["differences_from_M0"][m][k] for u in units])
                                            for k in ("solo", "pair", "phi")}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default=str(ROOT / "results" / "raw" / "rephase_ter"))
    ap.add_argument("--out", default=str(ROOT / "results" / "e2ter_rephase.json"))
    ap.add_argument("--reps", type=int, default=1000)
    ap.add_argument("--workers", type=int, default=3)
    args = ap.parse_args()
    timer = Timer()
    files = raw.list_units(args.raw, raw.REPHASE_RE)
    if not files:
        raise SystemExit(f"no rephase_ter files in {args.raw}")
    units = run_pool(unit, [(f, args.reps) for f in files], args.workers)
    units.sort(key=lambda u: (u["task"], u["seed"]))
    out = {"experiment": "E2 ter", "exploratory": True,
           "protocol": "PROTOCOL.md § 6 ter (v1.4): written after seeing all the results, before this run",
           "raw": Path(args.raw).name, "manifest": raw.read_manifest(args.raw),
           "settings": {"resamples": args.reps, "split_seed": est.SPLIT_SEED, "bootstrap_seed": est.BOOT_SEED,
                        "bootstrap_weight": "chosen again in each resample on 13 weights",
                        "control": CONTROL, "differences": "machine minus M0plus, paired on the same resamples; "
                                                           "differences_from_M0 likewise from M0"},
           "criteria": criteria(units), "by_machine": summary_by_machine(units), "units": units,
           "minutes": timer.minutes(), "environment": environment()}
    write_json(args.out, out)
    for h, v in out["criteria"].items():
        print(h, v["outcome"])
    print("minutes", out["minutes"])


if __name__ == "__main__":
    main()
