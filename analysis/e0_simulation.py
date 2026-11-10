"""E0 (protocol § 4): the instrument on data generated from the model.

Bias of the tuned turing on 3 x 8 situations, 200 experiments each, and coverage of its
bootstrap interval on the reduced set (k = 1, rho in {0, -0.6, -0.9}, 100 experiments,
300 resamples with the weight chosen again on the 13-weight grid).

    python analysis/e0_simulation.py [--out results/e0_simulation.json] [--quick]
"""
from __future__ import annotations

import argparse
import math

import numpy as np

from common import ROOT, Timer, environment, run_pool, verdict, write_json
from cognitive_resonance import estimators as est
from cognitive_resonance import model
from cognitive_resonance.simulate import IMAGE_SHARE, model_experiment

KS = (0.7, 1.0, 1.5)
RHOS = (0.6, 0.3, 0.0, -0.3, -0.6, -0.8, -0.9, -0.95)
EXPERIMENTS = 200
COVER_RHOS = (0.0, -0.6, -0.9)
COVER_EXPERIMENTS = 100
COVER_REPS = 300
DH = 1.0                 # the person's sensitivity in the simulation (not fixed by the protocol)
DH_CHECK = 2.0           # added: the same bias at a higher sensitivity, without a criterion
CHECK_EXPERIMENTS = 50
SEED = 534


def _standardized(d):
    zh = est.standardize(d["z_h"], d["y"], d["val_z_h"], d["val_y"])
    zm = est.standardize(d["z_m"], d["y"], d["val_z_m"], d["val_y"])
    return d["y"], d["image"], zh, zm


def one(job):
    k, rho, e, dh = job
    gen = np.random.default_rng([SEED, round(k * 100), round((rho + 1) * 1000), e, round(dh * 10)])
    y, image, zh, zm = _standardized(model_experiment(gen, k, rho, dh=dh))
    half = est.split_halves(image, y)
    t = est.tuned_turing(zh, zm, y, half)
    edge = any(bool(est.at_edge(zh[half == b] + w * zm[half == b], y[half == b]))
               for b, w in ((1, t["w_tuned_halves"][0]), (0, t["w_tuned_halves"][1])))
    dh_hat, dm_hat = float(est.d_a(zh, y)), float(est.d_a(zm, y))
    rho_hat = float(est.pearson_within(zh, zm, y))
    return {"tu_tuned": t["tu_tuned"], "k": dm_hat / dh_hat, "d_h": dh_hat, "rho": rho_hat,
            "rho_tetrachoric": float(est.tetrachoric_within(zh, zm, y)),
            "tu_app_plugin": model.tu_app(dm_hat / dh_hat, rho_hat), "edge": edge}


def cover(job):
    rho, e = job
    k = 1.0
    gen = np.random.default_rng([SEED, 7, round((rho + 1) * 1000), e])
    y, image, zh, zm = _standardized(model_experiment(gen, k, rho, dh=DH))
    design = est.Design(y, image, {"h": zh, "m": zm})
    reps = est.bootstrap(design, {"pair": ("h", "m")}, reps=COVER_REPS, seed=SEED + e)
    tu_ci = est.ci(reps[("pair", "tu_tuned")])
    phi_ci = est.ci(reps[("pair", "phi")])
    truth, phi = model.tu_app(k, rho), float(model.angle_deg(rho))
    return {"tu_ci": tu_ci, "covered": tu_ci[0] <= truth <= tu_ci[1],
            "phi_covered": phi_ci[0] <= phi <= phi_ci[1]}


def summarize(k, rho, rows, dh) -> dict:
    truth = model.tu_app(k, rho)
    tu = np.array([r["tu_tuned"] for r in rows])
    return {"k": k, "rho": rho, "phi": float(model.angle_deg(rho)), "d_h": dh, "experiments": len(rows),
            "tu_true": truth, "tu_tuned_mean": float(tu.mean()), "tu_tuned_median": float(np.median(tu)),
            "tu_tuned_sd": float(tu.std(ddof=1)),
            "bias_rel": float(tu.mean() / truth - 1.0), "bias_rel_median": float(np.median(tu) / truth - 1.0),
            "k_mean": float(np.mean([r["k"] for r in rows])),
            "rho_mean": float(np.mean([r["rho"] for r in rows])),
            "rho_tetrachoric_mean": float(np.mean([r["rho_tetrachoric"] for r in rows])),
            "tu_app_plugin_bias_rel": float(np.mean([r["tu_app_plugin"] for r in rows]) / truth - 1.0),
            "share_at_edge": float(np.mean([r["edge"] for r in rows]))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "results" / "e0_simulation.json"))
    ap.add_argument("--quick", action="store_true", help="a few experiments, to try the script")
    ap.add_argument("--workers", type=int, default=3)
    args = ap.parse_args()
    n_exp, n_cov, n_check = (EXPERIMENTS, COVER_EXPERIMENTS, CHECK_EXPERIMENTS) if not args.quick else (4, 3, 2)
    timer = Timer()

    jobs = [(k, rho, e, DH) for k in KS for rho in RHOS for e in range(n_exp)]
    rows = run_pool(one, jobs, args.workers)
    bias = []
    for k in KS:
        for rho in RHOS:
            sel = [r for j, r in zip(jobs, rows) if j[0] == k and j[1] == rho]
            bias.append(summarize(k, rho, sel, DH))

    cjobs = [(rho, e) for rho in COVER_RHOS for e in range(n_cov)]
    crow = run_pool(cover, cjobs, args.workers)
    coverage = []
    for rho in COVER_RHOS:
        sel = [r for j, r in zip(cjobs, crow) if j[0] == rho]
        coverage.append({"k": 1.0, "rho": rho, "experiments": len(sel), "resamples": COVER_REPS,
                         "covered": int(sum(r["covered"] for r in sel)),
                         "covered_share": float(np.mean([r["covered"] for r in sel])),
                         "phi_covered_share": float(np.mean([r["phi_covered"] for r in sel])),
                         "median_ci_width": float(np.median([r["tu_ci"][1] - r["tu_ci"][0] for r in sel]))})

    check_jobs = [(k, rho, e, DH_CHECK) for k in KS for rho in RHOS for e in range(n_check)]
    check_rows = run_pool(one, check_jobs, args.workers)
    check = []
    for k in KS:
        for rho in RHOS:
            sel = [r for j, r in zip(check_jobs, check_rows) if j[0] == k and j[1] == rho]
            check.append(summarize(k, rho, sel, DH_CHECK))

    judged = [b for b in bias if b["rho"] >= -0.9]
    worst = max(judged, key=lambda b: abs(b["bias_rel"]))
    e0a = verdict(all(abs(b["bias_rel"]) <= 0.05 for b in judged),
                  rule="|mean tuned turing / true - 1| <= 0.05 in every situation with rho >= -0.9",
                  situations=len(judged), within=sum(abs(b["bias_rel"]) <= 0.05 for b in judged),
                  worst={"k": worst["k"], "rho": worst["rho"], "bias_rel": worst["bias_rel"]})
    e0b = verdict(all(90 <= c["covered"] * 100 / c["experiments"] <= 98 for c in coverage),
                  rule="the 95% interval contains the true value between 90 and 98 times in 100, for each rho",
                  covered={str(c["rho"]): c["covered"] for c in coverage},
                  pooled_share=float(np.mean([r["covered"] for r in crow])))
    out = {
        "experiment": "E0", "protocol": "PROTOCOL.md § 4 (v1.2)",
        "design": {"n_images": 2000, "presentations": 5, "n_val_images": 1000, "d_h": DH,
                   "image_share_of_noise": IMAGE_SHARE, "k": KS, "rho": RHOS, "experiments": n_exp,
                   "coverage": {"k": 1.0, "rho": COVER_RHOS, "experiments": n_cov, "resamples": COVER_REPS,
                                "weight": "chosen again in every resample on the 13-weight grid"},
                   "split_seed": est.SPLIT_SEED, "seed": SEED,
                   "notes": ["the protocol does not fix dH; the simulation uses dH = 1, and the same bias at dH = 2 "
                             "is reported apart, without a criterion",
                             "noise: half of the variance belongs to the image (shared by its 5 presentations), "
                             "with the same correlation rho at both levels",
                             "an AUC of exactly 1 is moved half a pair from the edge (share_at_edge counts it)"]},
        "bias": bias, "coverage": coverage,
        "criteria": {"E0a": e0a, "E0b": e0b},
        "rho_minus_0_95": [b for b in bias if b["rho"] == -0.95],
        "added_check_dh2": check,
        "minutes": timer.minutes(), "environment": environment(),
    }
    write_json(args.out, out)
    print("E0a", e0a["outcome"], "E0b", e0b["outcome"], "minutes", out["minutes"])


if __name__ == "__main__":
    main()
