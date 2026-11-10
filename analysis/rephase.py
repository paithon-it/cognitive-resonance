"""E2 (protocol § 6): rephasing, a machine trained on the errors of a mind.

Reads results/raw/rephase/*.npz: the person H (plus, mu = 1), its control H' (H2) and the
machines M_lambda. Writes all the measures and the outcome of H2a-H2c.

    python analysis/rephase.py [--raw results/raw/rephase] [--out results/e2_rephase.json]
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from common import ROOT, Timer, environment, med, run_pool, verdict, write_json
from cognitive_resonance import estimators as est
from cognitive_resonance import model, raw


def unit(job):
    path, reps, fixed = job
    d = raw.read_rephase(path)
    y, lams = d["y"], [float(x) for x in d["lambdas"]]
    cues = {"H": est.standardize(d["z_H"], y, d["val_z_H"], d["val_y"]),
            "H2": est.standardize(d["z_H2"], y, d["val_z_H2"], d["val_y"])}
    for i in range(len(lams)):
        cues[f"M{i}"] = est.standardize(d["z_M"][i], y, d["val_z_M"][i], d["val_y"])
    pairs = {f"M{i}": ("H", f"M{i}") for i in range(len(lams))}
    pairs["H2"] = ("H", "H2")
    design = est.Design(y, d["image"], cues, a=d["a"])
    point = {name: est.measure_pair(cues[h], cues[m], y, design.half) for name, (h, m) in pairs.items()}
    weights = {name: p["w_tuned_halves"] for name, p in point.items()} if fixed else None
    boot = est.bootstrap(design, pairs, reps=reps, slopes=True, fixed_weights=weights)
    for name in pairs:
        est.attach_intervals(point[name], boot, name)
    judges = {j: {"d_a": float(est.d_a(z, y)), "d_a_ci": est.ci(boot[(j, "d_a")]),
                  "d_yesno": float(est.d_yesno(z, y)),
                  "slope_on_a": float(est.slope_within(z, d["a"], y)), "slope_on_a_ci": est.ci(boot[(j, "slope")])}
              for j, z in cues.items()}
    diffs = {}
    for i in range(1, len(lams)):
        solo = boot[(f"M{i}", "d_a")] - boot[("M0", "d_a")]
        pair = boot[(f"M{i}", "d_pair_tuned")] - boot[("M0", "d_pair_tuned")]
        diffs[f"M{i}"] = {"lambda": lams[i],
                          "solo": judges[f"M{i}"]["d_a"] - judges["M0"]["d_a"], "solo_ci": est.ci(solo),
                          "pair": point[f"M{i}"]["d_pair_tuned"] - point["M0"]["d_pair_tuned"], "pair_ci": est.ci(pair)}
    return {"task": d["task"], "seed": d["seed"], "file": d["file"], "lambdas": lams, "n": int(y.size),
            "n_images": design.n_images, "judges": judges, "pairs": point, "differences": diffs}


def _above90(p):
    return p.get("phi_ci") is not None and p["phi_ci"][0] > 90.0


def _need(n_tasks, of_five):
    return math.ceil(of_five / 5 * n_tasks)


def criteria(units) -> dict:
    tasks = sorted({u["task"] for u in units})
    out = {}
    lams = units[0]["lambdas"]
    machines = [f"M{i}" for i in range(len(lams))]

    rank = {f"{u['task']} s{u['seed']}": est.spearman(lams, [u["pairs"][m]["phi"] for m in machines]) for u in units}
    per_task = {}
    for t in tasks:
        us = [u for u in units if u["task"] == t]
        ok = [lams[i] for i, m in enumerate(machines) if all(_above90(u["pairs"][m]) for u in us)]
        per_task[t] = {"lambdas_with_phi_interval_above_90_in_every_seed": ok,
                       "phi_by_seed": {f"s{u['seed']}": [u["pairs"][m]["phi"] for m in machines] for u in us}}
    n_ok = sum(bool(v["lambdas_with_phi_interval_above_90_in_every_seed"]) for v in per_task.values())
    mr = med(rank.values())
    h2a_pass = mr is not None and mr >= 0.9 and n_ok >= _need(len(tasks), 3)
    moved = [u["pairs"][machines[-1]]["phi"] - u["pairs"]["M0"]["phi"] for u in units]
    out["H2a"] = verdict(h2a_pass, median_spearman_lambda_phi=mr, spearman_by_task_seed=rank,
                         tasks_with_phi_interval_above_90=n_ok, tasks_needed=_need(len(tasks), 3), by_task=per_task,
                         phi_moved_from_lambda0_to_max={"median": med(moved), "min": float(np.min(moved)),
                                                        "max": float(np.max(moved))},
                         rule="median Spearman(lambda, phi) >= 0.9 and, for at least one lambda, the phi interval "
                              "above 90 degrees in every seed, in at least 3 tasks of 5")

    per_task_b = {}
    for t in tasks:
        us = [u for u in units if u["task"] == t]
        good = []
        for m in machines[1:]:
            if all(u["differences"][m]["solo_ci"][1] < 0 and u["differences"][m]["pair_ci"][0] > 0 for u in us):
                good.append(us[0]["differences"][m]["lambda"])
        per_task_b[t] = {"lambdas": good,
                         "differences_by_seed": {f"s{u['seed']}": u["differences"] for u in us}}
    n_b = sum(bool(v["lambdas"]) for v in per_task_b.values())
    out["H2b"] = verdict(n_b >= _need(len(tasks), 3), tasks_passing=n_b, tasks_needed=_need(len(tasks), 3),
                         by_task={t: v["lambdas"] for t, v in per_task_b.items()},
                         rule="some lambda > 0 with d_a(M_lambda) - d_a(M_0) < 0 and d_pair(H, M_lambda) - "
                              "d_pair(H, M_0) > 0, both intervals excluding zero, in every seed; in >= 3 tasks of 5")

    cases = []
    for u in units:
        sh = u["judges"]["H"]["slope_on_a"]
        for i, m in enumerate(machines):
            if _above90(u["pairs"][m]):
                sm = u["judges"][m]["slope_on_a"]
                cases.append({"task": u["task"], "seed": u["seed"], "lambda": lams[i], "slope_H": sh,
                              "slope_M": sm, "opposite": bool(np.sign(sm) == -np.sign(sh))})
    out["H2c"] = verdict(None if not cases else all(c["opposite"] for c in cases),
                         cases=len(cases), opposite_sign=sum(c["opposite"] for c in cases),
                         h2a_passed=h2a_pass, detail=cases,
                         rule="where the phi interval is above 90 degrees, the slope of M_lambda's cue on a has the "
                              "sign opposite to H's")
    return out


def by_lambda(units) -> dict:
    lams = units[0]["lambdas"]
    rows = {}
    for i, lam in enumerate(lams):
        m = f"M{i}"
        rows[f"{lam:g}"] = {
            "phi": med([u["pairs"][m]["phi"] for u in units]),
            "phi_tetrachoric": med([u["pairs"][m]["phi_tetrachoric"] for u in units]),
            "k": med([u["pairs"][m]["k"] for u in units]), "q": med([u["pairs"][m]["q"] for u in units]),
            "d_m": med([u["judges"][m]["d_a"] for u in units]), "d_h": med([u["judges"]["H"]["d_a"] for u in units]),
            "tu_tuned": med([u["pairs"][m]["tu_tuned"] for u in units]),
            "tu_app": med([u["pairs"][m]["tu_app"] for u in units]),
            "tu_app_nonneg": med([model.tu_app_nonneg(u["pairs"][m]["k"], u["pairs"][m]["rho"]) for u in units]),
            "d_pair_tuned": med([u["pairs"][m]["d_pair_tuned"] for u in units]),
            "slope_m_on_a": med([u["judges"][m]["slope_on_a"] for u in units]),
            "slope_h_on_a": med([u["judges"]["H"]["slope_on_a"] for u in units])}
    rows["control_H2"] = {k: med([u["pairs"]["H2"][k] for u in units])
                          for k in ("phi", "phi_tetrachoric", "k", "tu_tuned", "tu_app", "d_pair_tuned")}
    return rows


# --- exploratory, not in the protocol: the null at positive weight of the reversed machines ----------

def _null_positive(job):
    """For the machines below chance with their interval, the signed curve of (H, M) at w > 0."""
    path, cases = job
    d = raw.read_rephase(path)
    y = d["y"]
    zh = est.standardize(d["z_H"], y, d["val_z_H"], d["val_y"])
    half = est.split_halves(d["image"], y)
    rows = []
    for i, lam in cases:
        zm = est.standardize(d["z_M"][i], y, d["val_z_M"][i], d["val_y"])
        dh, dm = float(est.d_a(zh, y)), float(est.d_a(zm, y))
        k = dm / dh
        rho = float(est.pearson_within(zh, zm, y))
        t = est.signed_curve(zh, zm, y)
        w0 = est.null_from_curve(model.GRID_SIGNED, t, side=+1)
        depth = est.null_depth_split(zh, zm, y, half)
        # the same pair with the machine's cue reversed: measured again, not assumed
        k_rev = float(est.d_a(-zm, y)) / dh
        rho_rev = float(est.pearson_within(zh, -zm, y))
        rows.append({"task": d["task"], "seed": d["seed"], "lambda": lam, "k": k, "rho": rho,
                     "phi": float(model.angle_deg(rho)), "w0_positive": w0,
                     "w0_predicted": 1.0 / abs(k), "w0_ratio": w0 * abs(k) if w0 is not None else None,
                     "abs_tu_at_predicted_null": depth["abs_tu_at_null"],
                     "k_reversed": k_rev, "rho_reversed": rho_rev,
                     "tu_app": model.tu_peak_signed(k, rho), "tu_app_reversed": model.tu_peak_signed(k_rev, rho_rev),
                     "max_abs_tu_measured": float(np.max(np.abs(t)))})
    return rows


def exploratory_null(units, raw_dir, workers) -> dict:
    files = {Path(p).name: p for p in raw.list_units(raw_dir, raw.REPHASE_RE)}
    jobs = []
    for u in units:
        cases = [(i, lam) for i, lam in enumerate(u["lambdas"]) if lam > 0
                 and u["judges"][f"M{i}"]["d_a"] < 0 and u["judges"][f"M{i}"]["d_a_ci"][1] < 0]
        if cases:
            jobs.append((files[u["file"]], cases))
    rows = [r for rs in run_pool(_null_positive, jobs, workers) for r in rs]
    crossed = [r for r in rows if r["w0_positive"] is not None]
    gap = [abs(r["tu_app"] - r["tu_app_reversed"]) for r in rows]
    return {"exploratory": True,
            "note": "Not in the protocol, added after seeing the data. Machines with lambda > 0 whose d_a is negative "
                    "with its interval below zero: the signed turing of (H, M_lambda) at positive weights, where § 8 bis "
                    "puts the null at w0 = 1/|k|. Interpolation as in E5; |Tu| at the predicted null with k from one "
                    "half and the turing read on the other, as in H5b. The pair with the machine's cue reversed, "
                    "(-k, -rho), is measured again: its apparent turing must be the same.",
            "pairs": len(rows), "crossing_at_positive_weight": len(crossed),
            "median_ratio_w0_over_predicted": med([r["w0_ratio"] for r in crossed]),
            "ratio_quartiles": [float(np.percentile([r["w0_ratio"] for r in crossed], q)) for q in (25, 75)]
            if crossed else None,
            "median_abs_tu_at_predicted_null": med([r["abs_tu_at_predicted_null"] for r in rows]),
            "median_tu_app": med([r["tu_app"] for r in rows]),
            "median_tu_app_reversed": med([r["tu_app_reversed"] for r in rows]),
            "max_abs_difference_tu_app_reversed": float(max(gap)) if gap else None,
            "pairs_detail": rows}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default=str(ROOT / "results" / "raw" / "rephase"))
    ap.add_argument("--out", default=str(ROOT / "results" / "e2_rephase.json"))
    ap.add_argument("--reps", type=int, default=1000)
    ap.add_argument("--fixed-weight", action="store_true")
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--only-exploratory", action="store_true",
                    help="add the exploratory block to an existing results file and refresh its medians, "
                         "leaving the measures as they are")
    args = ap.parse_args()
    timer = Timer()
    if args.only_exploratory:
        out = json.loads(Path(args.out).read_text())
        out["by_lambda"] = by_lambda(out["units"])
        out["exploratory_null_positive_weight"] = exploratory_null(out["units"], args.raw, args.workers)
        write_json(args.out, out)
        e = out["exploratory_null_positive_weight"]
        print("exploratory:", e["pairs"], "pairs,", e["crossing_at_positive_weight"], "cross at w > 0")
        return
    files = raw.list_units(args.raw, raw.REPHASE_RE)
    if not files:
        raise SystemExit(f"no rephase files in {args.raw}")
    units = run_pool(unit, [(f, args.reps, args.fixed_weight) for f in files], args.workers)
    units.sort(key=lambda u: (u["task"], u["seed"]))
    out = {"experiment": "E2", "protocol": "PROTOCOL.md § 6 (v1.2)", "raw": Path(args.raw).name,
           "manifest": raw.read_manifest(args.raw),
           "settings": {"resamples": args.reps, "split_seed": est.SPLIT_SEED, "bootstrap_seed": est.BOOT_SEED,
                        "bootstrap_weight": "fixed at the whole-data choice" if args.fixed_weight
                        else "chosen again in each resample on 13 weights"},
           "criteria": criteria(units), "by_lambda": by_lambda(units), "units": units,
           "exploratory_null_positive_weight": exploratory_null(units, args.raw, args.workers),
           "minutes": timer.minutes(), "environment": environment()}
    write_json(args.out, out)
    for h, v in out["criteria"].items():
        print(h, v["outcome"])
    print("minutes", out["minutes"])


if __name__ == "__main__":
    main()
