"""The data behind the article's figures, from the results files (none of it is recomputed
from raw data except the theoretical curves).

    python analysis/figure_data.py [--results results] [--out results/figure_data.json]
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from common import ROOT, med, write_json
from cognitive_resonance import model

PHIS = (30, 60, 90, 120, 150)
FINE = np.exp(np.linspace(math.log(1 / 64), math.log(64), 241))


def theory() -> dict:
    out = []
    for phi in PHIS:
        rho = math.cos(math.radians(phi))
        low, high = model.band_edges_k1(phi)
        out.append({"phi": phi, "q": float(model.q_factor(phi)), "tu_peak": model.tu_app(1.0, rho),
                    "peak_tu2_minus_1": math.tan(math.radians(phi) / 2) ** 2, "band": [low, high],
                    "tu": [float(x) for x in model.tu(FINE, 1.0, rho)],
                    "tu2_minus_1": [float(x) for x in model.resonance(FINE, phi)]})
    return {"k": 1.0, "w": [float(x) for x in FINE], "omega": [float(math.sqrt(x)) for x in FINE], "curves": out,
            "note": "Tu(w)^2 - 1 = tan^2(phi/2) / (1 + Q^2 (Omega - 1/Omega)^2), Omega = sqrt(w)"}


def load(path):
    p = Path(path)
    return json.loads(p.read_text()) if p.is_file() else None


def _r(values, nd=5):
    return [None if v is None else round(float(v), nd) for v in values]


def bench(e1) -> dict | None:
    if e1 is None:
        return None
    mus = sorted({u["mu"] for u in e1["units"]})
    pick = mus
    by_unit = [{"task": u["task"], "seed": u["seed"], "mu": u["mu"],
                **{kind: {"phi": u["pairs"][kind]["phi"], "k": u["pairs"][kind]["k"],
                          "tu_tuned": u["pairs"][kind]["tu_tuned"], "tu_app": u["pairs"][kind]["tu_app"],
                          "measured": _r(u["pairs"][kind]["curve_measured"]),
                          "predicted": _r(u["pairs"][kind]["curve_predicted"])} for kind in ("opposite", "same")}}
               for u in e1["units"]]
    curves = {}
    for kind in ("opposite", "same"):
        curves[kind] = {}
        for m in pick:
            us = [u["pairs"][kind] for u in e1["units"] if u["mu"] == m]
            meas = np.array([u["curve_measured"] for u in us], float)
            pred = np.array([u["curve_predicted"] for u in us], float)
            curves[kind][f"{m:g}"] = {"measured_median": np.nanmedian(meas, 0).tolist(),
                                      "predicted_median": np.nanmedian(pred, 0).tolist(),
                                      "measured_q25": np.nanpercentile(meas, 25, 0).tolist(),
                                      "measured_q75": np.nanpercentile(meas, 75, 0).tolist(),
                                      "n_pairs": len(us)}
    return {"by_mu": e1["by_mu"], "grid": [float(x) for x in model.GRID], "curves": curves,
            "curves_by_unit": by_unit,
            "note": "by_mu and curves: medians over tasks and seeds (two seeds); tu_app is the prediction from the "
                    "measured k and phi; curves_by_unit: Tu(w) measured and predicted for every task, seed and mu, "
                    "on the 49 weights of `grid`; phi from Pearson (primary) and tetrachoric (bridge)"}


def null_examples(e5) -> dict | None:
    if e5 is None:
        return None
    units = e5["units"]
    same = min(units, key=lambda u: u["pairs"]["same"]["phi"])
    zero = sorted([u for u in units if u["mu"] == min(x["mu"] for x in units)], key=lambda u: (u["task"], u["seed"]))[0]
    wide = max(units, key=lambda u: u["pairs"]["opposite"]["phi"])

    def ex(label, u, kind):
        p = u["pairs"][kind]
        return {"label": label, "task": u["task"], "mu": u["mu"], "seed": u["seed"], "pair": kind,
                "phi": p["phi"], "k": p["k"], "q": p["q"], "w0_measured": p["w0"],
                "w0_predicted": -1.0 / p["k"] if p["k"] else None,
                "null_width_measured": p["null_measured"], "null_width_predicted": p["null_predicted"],
                "curve_measured": p["curve_signed_measured"], "curve_predicted": p["curve_signed_predicted"]}

    out = [ex("same bias, smallest angle", same, "same"), ex("no bias (lowest mu)", zero, "opposite"),
           ex("opposite biases, largest angle", wide, "opposite")]
    pos = [u for u in units if "positive" in u and u["positive"].get("d_m_ci") and u["positive"]["d_m_ci"][1] < 0]
    if pos:
        u = min(pos, key=lambda x: x["positive"]["d_m"])
        p = u["positive"]
        out.append({"label": "machine below chance (frame against it), null at positive weight", "task": u["task"],
                    "mu": u["mu"], "seed": u["seed"], "pair": "opposite, selected presentations",
                    "phi": p["phi"], "k": p["k"], "q": float(model.fano_q(p["k"], p["phi"])),
                    "w0_measured": p["w0_positive"], "w0_predicted": p["w0_predicted"],
                    "null_width_measured": p["null_measured"], "null_width_predicted": p["null_predicted"],
                    "curve_measured": p["curve_signed_measured"], "curve_predicted": p["curve_signed_predicted"]})
    return {"grid_signed": [float(x) for x in model.GRID_SIGNED], "examples": out, "by_mu": e5["by_mu"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=str(ROOT / "results"))
    ap.add_argument("--out", default=str(ROOT / "results" / "figure_data.json"))
    args = ap.parse_args()
    r = Path(args.results)
    e1, e2, e3, e4, e5, e2bis = (load(r / f) for f in ("e1_bench.json", "e2_rephase.json", "e3_people.json",
                                                       "e4_reread.json", "e5_null.json", "e2bis_rephase.json"))
    out = {"a_theory": theory(), "b_bench": bench(e1),
           "c_rephase": ({"by_lambda": e2["by_lambda"],
                          "exploratory_null_positive_weight": {k: v for k, v in
                                                               e2.get("exploratory_null_positive_weight", {}).items()
                                                               if k != "pairs_detail"} or None,
                          "e2bis_by_machine": e2bis["by_machine"] if e2bis else None} if e2 else None),
           "d_people": ({"by_lambda": e3["by_lambda"],
                         "base": {s: {"accuracy": v["base"]["accuracy"], "phi": v["base"]["angle"].get("phi"),
                                      "phi_ci": v["base"]["angle"].get("phi_ci"),
                                      "tu_app": (v["base"]["angle"]["ideal_phi"] / v["base"]["angle"]["dh"]
                                                 if v["base"]["angle"].get("ideal_phi") else None),
                                      "turing_group": v["base"]["group"]["turing_group"]}
                                  for s, v in e3["seeds"].items() if "base" in v}} if e3 else None),
           "e_reread": e4["pairs"] if e4 else None,
           "f_null": null_examples(e5),
           "missing": [n for n, x in (("e1", e1), ("e2", e2), ("e3", e3), ("e4", e4), ("e5", e5),
                                      ("e2bis", e2bis)) if x is None]}
    write_json(args.out, out)
    print("missing:", out["missing"])


if __name__ == "__main__":
    main()
