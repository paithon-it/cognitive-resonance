"""E2 bis (protocol § 6 bis, v1.3, exploratory): the partner trained for the pair.

Reads results/raw/rephase_bis/*.npz: the person H (plus, mu = 1) and three machines, in the
order given by `machines`: M0 (normal training), Mpair (trained on the error of the sum
z_H + z_M, H frozen) and Mframe (the same, seeing only the frame). Same estimators, grid
and bootstrap as E2. Writes every measure, the paired differences from M0 and B1-B5.

    python analysis/rephase_bis.py [--raw results/raw/rephase_bis] [--out results/e2bis_rephase.json]
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np

from common import ROOT, Timer, environment, med, run_pool, verdict, write_json
from cognitive_resonance import estimators as est
from cognitive_resonance import model, raw
from rephase import _need

MACHINES = raw.REPHASE_BIS_MACHINES


def tu_app_nonneg(k, rho) -> float | None:
    """Maximum of Tu(w) over w >= 0 for k of any sign (model.tu_app_nonneg)."""
    return model.tu_app_nonneg(k, rho) if abs(rho) < 1 else None


def measure(d, machines, reps):
    """Every pair (H, M) of one unit: point estimates with their intervals, the judges alone, and the
    bootstrap arrays (the same resamples for every pair), for paired differences."""
    y = d["y"]
    cues = {"H": est.standardize(d["z_H"], y, d["val_z_H"], d["val_y"])}
    for i, name in enumerate(d["machine_names"]):
        cues[name] = est.standardize(d["z_M"][i], y, d["val_z_M"][i], d["val_y"])
    pairs = {name: ("H", name) for name in machines}
    design = est.Design(y, d["image"], cues, a=d["a"])
    point = {name: est.measure_pair(cues["H"], cues[name], y, design.half) for name in machines}
    boot = est.bootstrap(design, pairs, reps=reps, slopes=True)
    for name in machines:
        p = est.attach_intervals(point[name], boot, name)
        sin_phi = math.sin(math.radians(p["phi"]))
        p["tu_app_signed"] = model.tu_peak_signed(p["k"], p["rho"])
        p["tu_app_nonneg"] = tu_app_nonneg(p["k"], p["rho"])
        p["inv_sin_phi"] = 1.0 / sin_phi
        p["ratio_tuned_app_nonneg"] = p["tu_tuned"] / p["tu_app_nonneg"] if p["tu_app_nonneg"] else None
        p["ratio_tuned_inv_sin_phi"] = p["tu_tuned"] * sin_phi
    judges = {j: {"d_a": float(est.d_a(z, y)), "d_a_ci": est.ci(boot[(j, "d_a")]),
                  "d_yesno": float(est.d_yesno(z, y)),
                  "slope_on_a": float(est.slope_within(z, d["a"], y)), "slope_on_a_ci": est.ci(boot[(j, "slope")])}
              for j, z in cues.items()}
    return design, point, judges, boot


def differences(point, judges, boot, names, reference) -> dict:
    """Paired differences from the reference machine: alone, in the tuned pair, and of the angle."""
    diffs = {}
    for name in names:
        solo = boot[(name, "d_a")] - boot[(reference, "d_a")]
        pair = boot[(name, "d_pair_tuned")] - boot[(reference, "d_pair_tuned")]
        phi = boot[(name, "phi")] - boot[(reference, "phi")]
        diffs[name] = {"solo": judges[name]["d_a"] - judges[reference]["d_a"], "solo_ci": est.ci(solo),
                       "pair": point[name]["d_pair_tuned"] - point[reference]["d_pair_tuned"], "pair_ci": est.ci(pair),
                       "phi": point[name]["phi"] - point[reference]["phi"], "phi_ci": est.ci(phi)}
    return diffs


def unit(job):
    path, reps = job
    d = raw.read_rephase_bis(path)
    design, point, judges, boot = measure(d, MACHINES, reps)
    diffs = differences(point, judges, boot, MACHINES[1:], "M0")
    return {"task": d["task"], "seed": d["seed"], "file": d["file"], "machines": list(MACHINES),
            "n": int(d["y"].size), "n_images": design.n_images, "judges": judges, "pairs": point,
            "differences": diffs}


def _opposite(a, b):
    return bool(np.sign(a) == -np.sign(b) and a != 0 and b != 0)


def per_task(units, test) -> dict:
    """Per task: True when the test holds in every seed (unit) of that task."""
    res = {}
    for t in sorted({u["task"] for u in units}):
        us = [u for u in units if u["task"] == t]
        res[t] = bool(us) and all(test(u) for u in us)
    return res


def criteria(units) -> dict:
    tasks = sorted({u["task"] for u in units})
    need = _need(len(tasks), 3)
    out = {}
    rows = []
    for u in units:
        j = u["judges"]
        rows.append({"unit": f"{u['task']} s{u['seed']}",
                     "mpair_above_chance": j["Mpair"]["d_a_ci"][0] > 0,
                     "mframe_contains_zero": j["Mframe"]["d_a_ci"][0] <= 0 <= j["Mframe"]["d_a_ci"][1],
                     "mpair_slope_opposite_to_h": _opposite(j["Mpair"]["slope_on_a"], j["H"]["slope_on_a"]),
                     "mframe_slope_opposite_to_h": _opposite(j["Mframe"]["slope_on_a"], j["H"]["slope_on_a"]),
                     "d_a_mpair_ci": j["Mpair"]["d_a_ci"], "d_a_mframe_ci": j["Mframe"]["d_a_ci"]})
    keys = ("mpair_above_chance", "mframe_contains_zero", "mpair_slope_opposite_to_h", "mframe_slope_opposite_to_h")
    out["B1"] = verdict(all(all(r[k] for k in keys) for r in rows), units=len(rows),
                        **{k: sum(r[k] for r in rows) for k in keys}, detail=rows,
                        rule="in every unit: Mpair's d_a interval above zero, Mframe's containing zero, and the slope "
                             "of both machines' cue on a opposite in sign to H's")

    b2 = per_task(units, lambda u: u["differences"]["Mpair"]["phi_ci"][0] > 0)
    out["B2"] = verdict(sum(b2.values()) >= need, tasks_passing=sum(b2.values()), tasks_needed=need, by_task=b2,
                        median_phi_difference=med([u["differences"]["Mpair"]["phi"] for u in units]),
                        rule="phi(H, Mpair) - phi(H, M0) with its interval above zero in every seed, in >= 3 tasks of 5")
    b3 = per_task(units, lambda u: u["differences"]["Mpair"]["solo_ci"][1] < 0
                  and u["differences"]["Mpair"]["pair_ci"][0] > 0)
    out["B3"] = verdict(sum(b3.values()) >= need, tasks_passing=sum(b3.values()), tasks_needed=need, by_task=b3,
                        median_solo_difference=med([u["differences"]["Mpair"]["solo"] for u in units]),
                        median_pair_difference=med([u["differences"]["Mpair"]["pair"] for u in units]),
                        rule="Mpair alone less sensitive than M0 and (H, Mpair) more sensitive than (H, M0), both "
                             "intervals excluding zero in every seed, in >= 3 tasks of 5")
    b4 = per_task(units, lambda u: u["pairs"]["Mframe"]["tu_tuned_ci"][0] > 1.0)
    r4 = med([u["pairs"]["Mframe"]["ratio_tuned_inv_sin_phi"] for u in units])
    out["B4"] = verdict(sum(b4.values()) >= need and r4 is not None and 0.9 <= r4 <= 1.1,
                        tasks_with_interval_above_1=sum(b4.values()), tasks_needed=need, by_task=b4,
                        median_ratio_tuned_over_inv_sin_phi=r4,
                        median_tuned_turing=med([u["pairs"]["Mframe"]["tu_tuned"] for u in units]),
                        median_inv_sin_phi=med([u["pairs"]["Mframe"]["inv_sin_phi"] for u in units]),
                        rule="tuned turing of (H, Mframe) with its interval above 1 in every seed in >= 3 tasks of 5, "
                             "and median tuned / (1/sin phi) in [0.9, 1.1]")
    r5 = med([u["pairs"]["Mpair"]["ratio_tuned_app_nonneg"] for u in units])
    out["B5"] = verdict(r5 is not None and 0.9 <= r5 <= 1.1, median_ratio_tuned_over_tu_app=r5,
                        median_tuned_turing=med([u["pairs"]["Mpair"]["tu_tuned"] for u in units]),
                        median_tu_app=med([u["pairs"]["Mpair"]["tu_app_nonneg"] for u in units]),
                        rule="median tuned turing of (H, Mpair) / Tu_app from k and phi in [0.9, 1.1] (Tu_app: maximum "
                             "over w >= 0)")
    return out


def by_machine(units, machines=MACHINES, reference="M0") -> dict:
    out = {"H": {"d_a": med([u["judges"]["H"]["d_a"] for u in units]),
                 "slope_on_a": med([u["judges"]["H"]["slope_on_a"] for u in units])}}
    for m in machines:
        ps = [u["pairs"][m] for u in units]
        out[m] = {"d_a": med([u["judges"][m]["d_a"] for u in units]),
                  "slope_on_a": med([u["judges"][m]["slope_on_a"] for u in units]),
                  **{k: med([p[k] for p in ps]) for k in ("phi", "phi_tetrachoric", "k", "q", "tu_tuned",
                                                          "tu_app_nonneg", "tu_app_signed", "inv_sin_phi",
                                                          "d_pair_tuned")}}
        if m != reference:
            out[m][f"difference_from_{reference}"] = {k: med([u["differences"][m][k] for u in units])
                                                      for k in ("solo", "pair", "phi")}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default=str(ROOT / "results" / "raw" / "rephase_bis"))
    ap.add_argument("--out", default=str(ROOT / "results" / "e2bis_rephase.json"))
    ap.add_argument("--reps", type=int, default=1000)
    ap.add_argument("--workers", type=int, default=3)
    args = ap.parse_args()
    timer = Timer()
    files = raw.list_units(args.raw, raw.REPHASE_RE)
    if not files:
        raise SystemExit(f"no rephase_bis files in {args.raw}")
    units = run_pool(unit, [(f, args.reps) for f in files], args.workers)
    units.sort(key=lambda u: (u["task"], u["seed"]))
    out = {"experiment": "E2 bis", "exploratory": True,
           "protocol": "PROTOCOL.md § 6 bis (v1.3): written after seeing E2 and E3, before this run",
           "raw": Path(args.raw).name, "manifest": raw.read_manifest(args.raw),
           "settings": {"resamples": args.reps, "split_seed": est.SPLIT_SEED, "bootstrap_seed": est.BOOT_SEED,
                        "bootstrap_weight": "chosen again in each resample on 13 weights"},
           "criteria": criteria(units), "by_machine": by_machine(units), "units": units,
           "minutes": timer.minutes(), "environment": environment()}
    write_json(args.out, out)
    for h, v in out["criteria"].items():
        print(h, v["outcome"])
    print("minutes", out["minutes"])


if __name__ == "__main__":
    main()
