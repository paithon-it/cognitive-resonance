"""E5 (protocol § 8 bis, v1.2): the null, on the files of E1.

For every bench pair: the signed turing on the extended grid (49 weights of each sign),
the measured null and its width against the prediction from the measured k and rho, the
depth at the predicted null with k from the other half, and the error over the whole
curve. For H5d, in the opposite pairs with mu >= 1, the presentations where the frame
points against the truth for the machine: there the machine is below chance and the null
moves to a positive weight.

    python analysis/null.py [--raw results/raw/bench] [--out results/e5_null.json]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from bench import JUDGES, PAIRS, standardized
from common import ROOT, Timer, environment, med, run_pool, verdict, write_json
from cognitive_resonance import estimators as est
from cognitive_resonance import model, raw

H5D_MU = 1.0


def against_frame(y, a, sign, sd):
    """Presentations where the frame pushes the machine toward the wrong class, beyond one SD."""
    push = sign * a
    return np.where(y == 1, push < -sd, push > sd)


def positive_null(d, reps) -> dict:
    y, a = d["y"], d["a"]
    slope = float(est.slope_within(d["z_minus"], a, y))
    sign = float(np.sign(slope))
    sel = against_frame(y, a, sign, float(np.std(a)))
    vsel = against_frame(d["val_y"], d["val_a"], sign, float(np.std(d["val_a"])))
    out = {"machine_slope_on_a": slope, "n_selected": int(sel.sum()), "n_val_selected": int(vsel.sum())}
    if min(np.bincount(y[sel], minlength=2)) < 2 or min(np.bincount(d["val_y"][vsel], minlength=2)) < 2:
        out["skipped"] = "a class without selected presentations"
        return out
    ys = y[sel]
    zh = est.standardize(d["z_plus"][sel], ys, d["val_z_plus"][vsel], d["val_y"][vsel])
    zm = est.standardize(d["z_minus"][sel], ys, d["val_z_minus"][vsel], d["val_y"][vsel])
    image = d["image"][sel]
    ids, idx = np.unique(image, return_inverse=True)
    sm = est.SortedScore(zm, ys)
    gen = np.random.default_rng(est.BOOT_SEED)
    boot = []
    for start in range(0, reps, 100):
        W = est.resample_counts(gen, ids.size, min(100, reps - start))[:, idx]
        boot.append(est.d_a_from_auc(*sm.auc(W)))
    dm_boot = np.concatenate(boot)
    dh, dm = float(est.d_a(zh, ys)), float(est.d_a(zm, ys))
    k = dm / dh
    rho = float(est.pearson_within(zh, zm, ys))
    t = est.signed_curve(zh, zm, ys)
    w0 = est.null_from_curve(model.GRID_SIGNED, t, side=+1)
    pred = model.tu_signed(model.GRID_SIGNED, k, rho)
    out.update({"d_h": dh, "d_m": dm, "d_m_ci": est.ci(dm_boot), "k": k, "rho": rho,
                "person_at_auc_edge": bool(est.at_edge(zh, ys)),
                "phi": float(model.angle_deg(rho)), "w0_positive": w0,
                "w0_predicted": 1.0 / abs(k) if k < 0 else None,
                "w0_ratio": w0 * abs(k) if (w0 is not None and k < 0) else None,
                "null_measured": est.null_width_from_curve(model.GRID_SIGNED, t, w0),
                "null_predicted": est.null_predicted(k, rho),
                "curve_signed_measured": [float(x) for x in t], "curve_signed_predicted": [float(x) for x in pred]})
    return out


def unit(job):
    path, reps = job
    d = raw.read_bench(path)
    y = d["y"]
    cues = standardized(d, JUDGES)
    half = est.split_halves(d["image"], y)
    pairs = {kind: est.measure_null(cues[h], cues[m], y, half, side=-1) for kind, (h, m) in PAIRS.items()}
    out = {"task": d["task"], "mu": d["mu"], "seed": d["seed"], "file": d["file"], "pairs": pairs}
    if d["mu"] >= H5D_MU:
        out["positive"] = positive_null(d, reps)
    return out


def criteria(units) -> dict:
    allp = [(u, kind, u["pairs"][kind]) for u in units for kind in PAIRS]
    crossed = [p for _, _, p in allp if p["w0"] is not None]
    mk = med([p["w0_k"] for p in crossed])
    out = {"H5a": verdict(len(crossed) == len(allp) and mk is not None and -1.1 <= mk <= -0.9,
                          pairs=len(allp), crossing_at_negative_weight=len(crossed),
                          missing=[f"{u['task']} mu{u['mu']:g} s{u['seed']} {kind}" for u, kind, p in allp
                                   if p["w0"] is None],
                          median_w0_k=mk,
                          w0_k_quartiles=[float(np.percentile([p["w0_k"] for p in crossed], q)) for q in (25, 75)]
                          if crossed else None,
                          rule="every pair crosses zero at a negative weight and median w0*k in [-1.1, -0.9]")}
    depth = med([p["abs_tu_at_null"] for _, _, p in allp])
    out["H5b"] = verdict(depth < 0.1, pairs=len(allp), median_abs_tu_at_predicted_null=depth,
                         quartiles=[float(np.percentile([p["abs_tu_at_null"] for _, _, p in allp], q)) for q in (25, 75)],
                         by_kind={k: med([p["abs_tu_at_null"] for _, kk, p in allp if kk == k]) for k in PAIRS},
                         rule="|Tu| at -1/k (k from one half, read on the other) has a median below 0.1")
    both = [p for _, _, p in allp if p["null_measured"]["log_width"] is not None
            and p["null_predicted"]["log_width"] is not None]
    lr = [p["null_measured"]["log_width"] / p["null_predicted"]["log_width"] for p in both]
    meas = [p for _, _, p in allp if p["null_measured"]["width"] is not None]
    rk = est.spearman([p["phi"] for p in meas], [p["null_measured"]["width"] for p in meas])
    both_d = [p for _, _, p in allp if p["null_measured"]["width_diff"] is not None
              and p["null_predicted"]["width_diff"] is not None]
    lr_d = [np.log(p["null_measured"]["width_diff"]) / np.log(p["null_predicted"]["width_diff"]) for p in both_d]
    rk_d = est.spearman([p["phi"] for p in meas], [p["null_measured"]["width_diff"] for p in meas])
    mlr = med(lr)
    out["H5c"] = verdict(None if mlr is None or rk is None else (0.7 <= mlr <= 1.3 and rk >= 0.7),
                         pairs=len(allp), pairs_with_both_widths=len(both),
                         median_ratio_log_width_measured_over_predicted=mlr, spearman_phi_width=rk,
                         width="ratio far/near of the edges of |Tu| < 1/2, the convention of the band",
                         as_difference={"median_ratio_log_width": med(lr_d), "spearman_phi_width": rk_d,
                                        "pairs": len(both_d),
                                        "note": "ln of a difference of weights changes sign around 1; reported, "
                                                "not used for the outcome"},
                         rule="median ln(measured width)/ln(predicted width) in [0.7, 1.3] and Spearman(phi, width) "
                              ">= 0.7 over all the pairs")
    pos = [u["positive"] for u in units if "positive" in u and "d_m_ci" in u["positive"]]
    cases = [p for p in pos if p["d_m_ci"] is not None and p["d_m_ci"][1] < 0]
    ok = [p for p in cases if p["w0_positive"] is not None]
    mr = med([p["w0_ratio"] for p in ok])
    out["H5d"] = verdict(None if not cases else (len(ok) == len(cases) and mr is not None and 0.8 <= mr <= 1.25),
                         opposite_pairs_mu_ge_1=len([u for u in units if "positive" in u]),
                         with_selection=len(pos), machine_negative_with_interval=len(cases),
                         crossing_at_positive_weight=len(ok), median_ratio_measured_over_predicted=mr,
                         median_k=med([p["k"] for p in cases]), median_d_m=med([p["d_m"] for p in cases]),
                         median_n_selected=med([p["n_selected"] for p in cases]),
                         person_at_auc_edge=sum(p.get("person_at_auc_edge", False) for p in cases),
                         rule="where d_a(machine) < 0 with its interval, the signed turing crosses zero at a positive "
                              "weight, and median w0 / (1/|k|) in [0.8, 1.25]")
    mae = med([p["mae_over_peak"] for _, _, p in allp])
    out["H5e"] = verdict(mae < 0.10, pairs=len(allp), median_mae_over_tu_app=mae,
                         by_kind={k: med([p["mae_over_peak"] for _, kk, p in allp if kk == k]) for k in PAIRS},
                         quartiles=[float(np.percentile([p["mae_over_peak"] for _, _, p in allp], q)) for q in (25, 75)],
                         rule="median over pairs of mean |measured - predicted| / Tu_app on the extended grid below "
                              "0.10; Tu_app = sqrt(1 + k^2 - 2 k rho)/sin(phi), the peak of the Fano profile")
    return out


def by_mu(units) -> dict:
    out = {}
    for kind in PAIRS:
        rows = {}
        for m in sorted({u["mu"] for u in units}):
            ps = [u["pairs"][kind] for u in units if u["mu"] == m]
            rows[f"{m:g}"] = {"w0_k": med([p["w0_k"] for p in ps]),
                              "abs_tu_at_predicted_null": med([p["abs_tu_at_null"] for p in ps]),
                              "null_width_measured": med([p["null_measured"]["width"] for p in ps]),
                              "null_width_predicted": med([p["null_predicted"]["width"] for p in ps]),
                              "null_width_diff_measured": med([p["null_measured"]["width_diff"] for p in ps]),
                              "null_width_diff_predicted": med([p["null_predicted"]["width_diff"] for p in ps]),
                              "q": med([p["q"] for p in ps]), "phi": med([p["phi"] for p in ps]),
                              "k": med([p["k"] for p in ps]), "mae_over_peak": med([p["mae_over_peak"] for p in ps]),
                              "n_pairs": len(ps)}
        out[kind] = rows
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default=str(ROOT / "results" / "raw" / "bench"))
    ap.add_argument("--out", default=str(ROOT / "results" / "e5_null.json"))
    ap.add_argument("--reps", type=int, default=1000)
    ap.add_argument("--workers", type=int, default=3)
    args = ap.parse_args()
    timer = Timer()
    files = raw.list_units(args.raw, raw.BENCH_RE)
    if not files:
        raise SystemExit(f"no bench files in {args.raw}")
    units = run_pool(unit, [(f, args.reps) for f in files], args.workers)
    units.sort(key=lambda u: (u["task"], u["mu"], u["seed"]))
    out = {"experiment": "E5", "protocol": "PROTOCOL.md § 8 bis (v1.2)", "raw": Path(args.raw).name,
           "settings": {"grid": "49 weights of § 3 with both signs (98 points)", "resamples_h5d": args.reps,
                        "null": "first zero crossing going out from w = 0, linear in w",
                        "width": "|Tu| < 1/2 around the null, linear in w; ratio far/near and difference",
                        "h5d_selection": "frame against the truth for the machine beyond one SD of a; direction "
                                         "from the sign of the machine's within-class slope on a (test set)"},
           "criteria": criteria(units), "by_mu": by_mu(units), "units": units,
           "minutes": timer.minutes(), "environment": environment()}
    write_json(args.out, out)
    for h, v in out["criteria"].items():
        print(h, v["outcome"])
    print("minutes", out["minutes"])


if __name__ == "__main__":
    main()
