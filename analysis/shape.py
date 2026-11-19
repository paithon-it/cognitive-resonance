"""E9 (protocol § 15): the shape of the noise.

Reads results/raw/shape/*.npz: the two opposite networks of E1 ("plus" as H, "minus" as M) on
new seeds, with the test frame drawn from four distributions of the same variance: gauss (as
in E1), two (two values), uniform, t3 (Student's t, 3 degrees of freedom).

With sensitivity as a signal-to-noise ratio the formula of § 2 is an identity (`identity.py`):
no shape can break it. With d_a, which comes from the AUC, it holds for Gaussian cues. Here the
part of the noise that we plant ourselves stops being Gaussian, and the question is how far
the measured pair moves from what the formula promises. Writes every measure and S1-S5.

    python analysis/shape.py [--raw results/raw/shape] [--out results/e9_shape.json]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from common import ROOT, Timer, environment, med, run_pool, verdict, write_json
from cognitive_resonance import estimators as est
from cognitive_resonance import raw
from identity import excess_kurtosis, snr

SHAPES = raw.SHAPES
WINDOW = (0.9, 1.1)               # S1: the window of H1c
BROKEN = 0.9                      # S2: below this the formula promises more than the pair gives
SAME_PAIR = (0.95, 1.05)          # S5: the pair with a shape over the pair with the Gaussian frame
UNIFORM = (0.95, 1.10)            # S4


def unit(path) -> dict:
    d = raw.read_shape(path)
    y = d["y"]
    half = est.split_halves(d["image"], y)
    out = {"task": d["task"], "mu": d["mu"], "seed": d["seed"], "file": d["file"], "n": int(y.size), "shapes": {}}
    for i, shape in enumerate(d["shape_names"]):
        zh = est.standardize(d["z_plus"][i], y, d["val_z_plus"][i], d["val_y"])
        zm = est.standardize(d["z_minus"][i], y, d["val_z_minus"][i], d["val_y"])
        p = est.measure_pair(zh, zm, y, half)
        w = float(np.sqrt(np.prod(p["w_tuned_halves"])))
        pair = zh + w * zm
        sh, sm = est.within_class_sd(zh, y), est.within_class_sd(zm, y)
        wu = w * sm / sh
        identity = (1 + wu * snr(zm, y) / snr(zh, y)) / np.sqrt(1 + wu * wu + 2 * p["rho"] * wu)
        out["shapes"][shape] = {
            "phi": p["phi"], "k": p["k"], "d_h": p["d_h"], "d_m": p["d_m"], "tu_app": p["tu_app"],
            "tu_tuned": p["tu_tuned"], "d_pair_tuned": p["d_pair_tuned"], "ratio_tuned_app": p["ratio_tuned_app"],
            "d_pair_predicted": p["tu_app"] * p["d_h"],
            "snr_h": snr(zh, y), "snr_pair": snr(pair, y), "d_a_over_snr_single": p["d_h"] / snr(zh, y),
            "d_a_over_snr_pair": float(est.d_a(pair, y)) / snr(pair, y),
            "snr_identity_relative_error": abs(snr(pair, y) / snr(zh, y) / identity - 1.0),
            "excess_kurtosis_single": excess_kurtosis(zh, y), "excess_kurtosis_pair": excess_kurtosis(pair, y),
            "accuracy_single": float(np.mean((d["z_plus"][i] > 0) == (y == 1)))}
    g = out["shapes"]["gauss"]["d_pair_tuned"]
    for shape in SHAPES:
        out["shapes"][shape]["pair_over_gauss"] = out["shapes"][shape]["d_pair_tuned"] / g
    return out


def criteria(units) -> dict:
    out = {}
    at = lambda mu: [u for u in units if u["mu"] == mu]                                     # noqa: E731
    ratio = lambda mu, shape: med([u["shapes"][shape]["ratio_tuned_app"] for u in at(mu)])  # noqa: E731
    r = {f"{mu:g}": {s: ratio(mu, s) for s in SHAPES} for mu in sorted({u["mu"] for u in units})}

    high = [u for u in units if u["mu"] in (1.5, 2.0)]
    beyond = all(u["shapes"]["gauss"]["phi"] > 90 for u in high)
    ok = all(r[m]["gauss"] is not None and WINDOW[0] <= r[m]["gauss"] <= WINDOW[1] for m in ("1.5", "2"))
    out["S1"] = verdict(beyond and ok, median_ratio_gauss={m: r[m]["gauss"] for m in r}, window=list(WINDOW),
                        units_beyond_90=sum(u["shapes"]["gauss"]["phi"] > 90 for u in high), units=len(high),
                        rule="with the Gaussian frame, at mu = 1.5 and 2: the angle is above 90 degrees in every unit "
                             f"and the median of tuned turing / Tu_app is within {list(WINDOW)} (H1a and H1c of E1, on "
                             "seeds E1 never used)")
    out["S2"] = verdict(r["2"]["two"] < BROKEN and r["2"]["t3"] < BROKEN,
                        median_ratio_at_mu_2={s: r["2"][s] for s in SHAPES}, below=BROKEN,
                        rule=f"at mu = 2 the median ratio is below {BROKEN} both with the two-valued frame and with "
                             "the t frame: the formula promises more than the pair gives")
    out["S3"] = verdict(r["1.5"]["two"] > 1.0 and r["1.5"]["t3"] < 0.95,
                        median_ratio_at_mu_1_5={s: r["1.5"][s] for s in SHAPES},
                        rule="at mu = 1.5 the two shapes go opposite ways: the median ratio is above 1 with the "
                             "two-valued frame and below 0.95 with the t frame")
    out["S4"] = verdict(all(UNIFORM[0] <= r[m]["uniform"] <= UNIFORM[1] for m in ("1.5", "2")),
                        median_ratio_uniform={m: r[m]["uniform"] for m in r}, window=list(UNIFORM),
                        rule=f"with the uniform frame the median ratio stays within {list(UNIFORM)} at mu = 1.5 and 2")
    same = {f"{mu:g}": {s: med([u["shapes"][s]["pair_over_gauss"] for u in at(mu)]) for s in SHAPES if s != "gauss"}
            for mu in (1.5, 2.0)}
    out["S5"] = verdict(all(SAME_PAIR[0] <= x <= SAME_PAIR[1] for v in same.values() for x in v.values()),
                        median_pair_over_gauss=same, window=list(SAME_PAIR),
                        rule="the pair itself does not depend on the shape, because in the pair the frame cancels: "
                             f"d of the tuned pair over that with the Gaussian frame, median within {list(SAME_PAIR)} "
                             "for every shape at mu = 1.5 and 2")
    return out


def by_mu(units) -> list:
    rows = []
    for mu in sorted({u["mu"] for u in units}):
        us = [u for u in units if u["mu"] == mu]
        for shape in SHAPES:
            m = lambda key: med([u["shapes"][shape][key] for u in us])                    # noqa: E731
            rows.append({"mu": mu, "shape": shape, "units": len(us), **{k: m(k) for k in (
                "phi", "d_h", "snr_h", "d_a_over_snr_single", "d_pair_tuned", "d_pair_predicted", "ratio_tuned_app",
                "snr_pair", "d_a_over_snr_pair", "pair_over_gauss", "tu_tuned", "tu_app", "excess_kurtosis_single",
                "excess_kurtosis_pair", "accuracy_single")},
                         "snr_identity_max_relative_error": max(u["shapes"][shape]["snr_identity_relative_error"]
                                                                for u in us)})
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", type=Path, default=ROOT / "results" / "raw" / "shape")
    ap.add_argument("--out", type=Path, default=ROOT / "results" / "e9_shape.json")
    ap.add_argument("--workers", type=int, default=3)
    args = ap.parse_args()
    files = raw.list_units(args.raw, raw.BENCH_RE)
    if not files:
        raise SystemExit(f"no shape files in {args.raw}")
    timer = Timer()
    units = run_pool(unit, files, args.workers)
    out = {"experiment": "E9", "protocol": "§ 15 (v2.1)", "status": "test registered before the run",
           "raw": Path(args.raw).name, "manifest": raw.read_manifest(args.raw),
           "settings": {"grid": "49 weights, 1/64 to 64", "shapes": list(SHAPES),
                        "ratio": "tuned turing over Tu_app from k and rho of the same unit and shape, as in H1c",
                        "pair": "z_H + w z_M at the geometric mean of the two tuned weights, for the kurtosis and the "
                                "signal-to-noise ratio"},
           "criteria": criteria(units), "by_mu_and_shape": by_mu(units),
           "snr_identity_max_relative_error": max(s["snr_identity_relative_error"] for u in units
                                                  for s in u["shapes"].values()),
           "units": units, "minutes": timer.minutes(), "environment": environment()}
    write_json(args.out, out)
    for h, v in out["criteria"].items():
        print(h, v["outcome"])
    print("minutes", out["minutes"])


if __name__ == "__main__":
    main()
