"""How much of E1 is an identity? (added with protocol v2.0, descriptive, no criterion)

If sensitivity is the signal-to-noise ratio of a cue (difference of the class means over the
pooled within-class standard deviation) and rho is the within-class correlation, then for the
weighted sum of two cues

    SNR(z_H + w z_M) / SNR(z_H) = (1 + w k) / sqrt(1 + w^2 + 2 rho w),      k = SNR(z_M) / SNR(z_H)

holds for any data: it is algebra on the sample moments, with w in units of the two standard
deviations. E1 measures sensitivity as d_a = sqrt(2) z(AUC), which equals the SNR only for
Gaussian cues with the same variance in the two classes. So what H1c and H1d test is how far
the networks' cues are from that model. This script shows both sides on the files of E1:
the identity (to rounding) and the distance between d_a and the SNR, with the excess kurtosis
of the summed cue as a description of how Gaussian it is.

    python analysis/identity.py [--raw results/raw/bench] [--out results/identity.json]
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np

from common import ROOT, environment, med, write_json
from cognitive_resonance import estimators as est
from cognitive_resonance import model, raw

WEIGHTS = (0.25, 0.5, 1.0, 2.0, 4.0)


def snr(z, y) -> float:
    z, y = np.asarray(z, float), np.asarray(y)
    return float((z[y == 1].mean() - z[y == 0].mean()) / est.within_class_sd(z, y))


def excess_kurtosis(z, y) -> float:
    z, y = np.asarray(z, float), np.asarray(y)
    res = np.concatenate([z[y == c] - z[y == c].mean() for c in (0, 1)])
    return float(np.mean(res ** 4) / np.mean(res ** 2) ** 2 - 3.0)


def unit(path) -> dict:
    d = raw.read_bench(path)
    y = d["y"]
    zh = est.standardize(d["z_plus"], y, d["val_z_plus"], d["val_y"])
    zm = est.standardize(d["z_minus"], y, d["val_z_minus"], d["val_y"])
    rho = float(est.pearson_within(zh, zm, y))
    sh, sm = est.within_class_sd(zh, y), est.within_class_sd(zm, y)
    dh, dm = snr(zh, y), snr(zm, y)
    worst = 0.0
    for w in WEIGHTS:
        measured = snr(zh + w * zm, y) / dh
        wu = w * sm / sh                                   # the weight in units of the two standard deviations
        predicted = (1.0 + wu * dm / dh) / math.sqrt(1.0 + wu * wu + 2.0 * rho * wu)
        worst = max(worst, abs(measured / predicted - 1.0))
    da_h, da_m = float(est.d_a(zh, y)), float(est.d_a(zm, y))
    k = da_m / da_h
    auc_gap = float(est.d_a(zh + zm, y)) / da_h / float(model.tu(1.0, k, rho)) - 1.0 if k > 0 else None
    return {"task": d["task"], "mu": d["mu"], "seed": d["seed"], "phi": float(model.angle_deg(rho)),
            "snr_identity_max_relative_error": worst, "auc_measured_over_predicted_minus_1_at_w1": auc_gap,
            "snr_over_d_a_minus_1_person": dh / da_h - 1.0, "excess_kurtosis_of_the_sum": excess_kurtosis(zh + zm, y)}


def by_mu(units) -> list:
    rows = []
    for mu in sorted({u["mu"] for u in units}):
        us = [u for u in units if u["mu"] == mu]
        gaps = [abs(u["auc_measured_over_predicted_minus_1_at_w1"]) for u in us
                if u["auc_measured_over_predicted_minus_1_at_w1"] is not None]
        rows.append({"mu": mu, "pairs": len(us), "phi": med([u["phi"] for u in us]),
                     "auc_gap_median": med([u["auc_measured_over_predicted_minus_1_at_w1"] for u in us]),
                     "auc_gap_max_abs": max(gaps) if gaps else None,
                     "excess_kurtosis_of_the_sum": med([u["excess_kurtosis_of_the_sum"] for u in us])})
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", type=Path, default=ROOT / "results" / "raw" / "bench")
    ap.add_argument("--out", type=Path, default=ROOT / "results" / "identity.json")
    args = ap.parse_args()
    files = raw.list_units(args.raw, raw.BENCH_RE)
    if not files:
        raise SystemExit(f"no bench files in {args.raw}")
    units = [unit(f) for f in files]
    out = {"experiment": "E1, read again", "status": "added with protocol v2.0: descriptive, no criterion",
           "raw": Path(args.raw).name, "pair": "the two opposite networks of E1 (plus as H, minus as M)",
           "weights": list(WEIGHTS), "pairs": len(units),
           "snr_identity_max_relative_error": max(u["snr_identity_max_relative_error"] for u in units),
           "auc_gap_max_abs": max(abs(u["auc_measured_over_predicted_minus_1_at_w1"]) for u in units
                                  if u["auc_measured_over_predicted_minus_1_at_w1"] is not None),
           "by_mu": by_mu(units), "units": units, "environment": environment()}
    write_json(args.out, out)
    print("pairs", out["pairs"], "| identity, worst relative error", f"{out['snr_identity_max_relative_error']:.1e}",
          "| largest gap with d_a at w = 1", f"{out['auc_gap_max_abs']:.3f}")


if __name__ == "__main__":
    main()
