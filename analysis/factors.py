"""The turing of the pair in three factors (a reading added after the results: not a test).

`cognitive_resonance.chart.pair_factors` rearranges the theorem of the three machines:
with the partner Tu^2 = (1 + B) (q^2 + p). 1 + B is what cancelling the readable bias is worth
(the factor of resonance), q^2 what the machine knows of the task, p what the judge knows and the
machine does not. This script reads the three factors on the files already collected and puts the
turings they give beside the measured ones:

- the bench of the chart (E8, results/raw/chart) and the same bench on new seeds (the files of
  E10 at level 0, results/raw/trace): judge H, soloist M, partner, blind machine;
- the bench of the curve (E1, results/raw/bench): two networks with opposite biases. There the
  second network carries the opposite push itself, so the pair is read with the factors of the
  first network and compared with the tuned pair of the two.

Measured turings are tuned pairs (§ 3) over the sensitivity of the judge alone.

    python analysis/factors.py [--chart results/raw/chart] [--trace results/raw/trace]
                               [--bench results/raw/bench] [--out results/pair_factors.json]
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np

from common import ROOT, Timer, environment, med, run_pool, write_json
from chart import predictions
from cognitive_resonance import chart as theorem
from cognitive_resonance import estimators as est
from cognitive_resonance import raw


def _factors(zh, zm, a, y) -> dict:
    pr = predictions(zh, zm, a, y)
    f = theorem.pair_factors(pr["d_h_clean"], pr["d_m"], pr["r"], pr["b"])
    return {"b": pr["b"], "r": pr["r"], "d_h_clean": pr["d_h_clean"], "d_m": pr["d_m"], **f}


def _measured(zh, others: dict, y, half) -> dict:
    d_alone = float(est.d_a(zh, y))
    out = {"judge_alone": d_alone}
    for name, z in others.items():
        m = est.measure_pair(zh, z, y, half)
        out[f"tu_{name}"] = m["d_pair_tuned"] / d_alone
        out[f"angle_{name}"] = m["phi"]
    return out


def chart_unit(path) -> dict:
    d = raw.read_chart(path)
    y, a = d["y"], np.asarray(d["a"], float)
    std = lambda z, vz: est.standardize(z, y, vz, d["val_y"])                                # noqa: E731
    i = d["machine_names"].index
    zh = std(d["z_H"], d["val_z_H"])
    z = {n: std(d["z_M"][i(k)], d["val_z_M"][i(k)]) for n, k in (("soloist", "M"), ("partner", "Mramo"), ("blind", "Mblind"))}
    half = est.split_halves(d["image"], y)
    return {"bench": "chart", "task": d["task"], "seed": d["seed"], "condition": f"v{d['v']}",
            "factors": _factors(zh, z["soloist"], a, y), "measured": _measured(zh, z, y, half)}


def trace_unit(path) -> dict:
    d = raw.read_trace(path)                                    # level 0: the frame says nothing, as in E8
    y, a = d["y"], np.asarray(d["a"][0], float)
    std = lambda z, vz: est.standardize(z, y, vz, d["val_y"])                                # noqa: E731
    zh = std(d["z_H"][0], d["val_z_H"][0])
    zm, vzm = np.asarray(d["z_M"], float), np.asarray(d["val_z_M"], float)
    z = {"soloist": std(zm, vzm), "partner": std(zm + d["z_R"][0][0], vzm + d["val_z_R"][0][0]),
         "blind": std(d["z_R"][0][0], d["val_z_R"][0][0])}
    half = est.split_halves(d["image"], y)
    return {"bench": "trace", "task": d["task"], "seed": d["seed"], "condition": f"v{d['v']}",
            "factors": _factors(zh, z["soloist"], a, y), "measured": _measured(zh, z, y, half)}


def bench_unit(path) -> dict:
    d = raw.read_bench(path)
    y, a = d["y"], np.asarray(d["a"], float)
    std = lambda z, vz: est.standardize(z, y, vz, d["val_y"])                                # noqa: E731
    zh, zo = std(d["z_plus"], d["val_z_plus"]), std(d["z_minus"], d["val_z_minus"])
    slope_o = float(est.slope_within(zo, a, y))
    clean_other = zo - slope_o * a                              # the second network without its own push
    f = _factors(zh, clean_other / est.within_class_sd(clean_other, y), a, y)
    half = est.split_halves(d["image"], y)
    return {"bench": "curve", "task": d["task"], "seed": d["seed"], "condition": f"mu{d['mu']:g}",
            "factors": f, "measured": _measured(zh, {"partner": zo}, y, half)}


def summary(units) -> list:
    rows = []
    for bench in ("curve", "chart", "trace"):
        conds = sorted({u["condition"] for u in units if u["bench"] == bench}, key=lambda c: float(c.lstrip("vmu")))
        for c in conds:
            us = [u for u in units if u["bench"] == bench and u["condition"] == c]
            row = {"bench": bench, "condition": c, "units": len(us)}
            for k in us[0]["factors"]:
                row[k] = med([u["factors"][k] for u in us])
            for k in us[0]["measured"]:
                row["measured_" + k] = med([u["measured"][k] for u in us])
            for name in ("partner", "soloist", "blind"):
                if f"tu_{name}" in us[0]["measured"]:
                    row[f"measured_over_equation_{name}"] = med([u["measured"][f"tu_{name}"] / u["factors"][f"tu_{name}"] for u in us])
            total = lambda u: math.log(u["factors"]["resonance"]) + math.log(u["factors"]["strength"] + u["factors"]["private"])  # noqa: E731
            row["share_of_log_turing_from_resonance"] = med([math.log(u["factors"]["resonance"]) / total(u) for u in us
                                                             if total(u) > 0])
            rows.append(row)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--chart", type=Path, default=ROOT / "results" / "raw" / "chart")
    ap.add_argument("--trace", type=Path, default=ROOT / "results" / "raw" / "trace")
    ap.add_argument("--bench", type=Path, default=ROOT / "results" / "raw" / "bench")
    ap.add_argument("--out", type=Path, default=ROOT / "results" / "pair_factors.json")
    ap.add_argument("--workers", type=int, default=3)
    args = ap.parse_args()
    timer = Timer()
    units = []
    for folder, pattern, fn in ((args.bench, raw.BENCH_RE, bench_unit), (args.chart, raw.CHART_RE, chart_unit),
                                (args.trace, raw.TRACE_RE, trace_unit)):
        files = raw.list_units(folder, pattern)
        units += run_pool(fn, files, args.workers) if files else []
    if not units:
        raise SystemExit("no raw files found")
    out = {"experiment": "the turing of the pair in three factors", "protocol": "§ 14, on files already collected",
           "status": "a reading added on 20 November 2026: the algebra of the theorem rearranged, not a test",
           "how": "per unit: B, r, d_h and d_m measured as in analysis/chart.py; Tu^2 = (1 + B)(q^2 + p) with the partner, "
                  "(1 + B)(q^2 + p (1 - r^2) / (1 - r^2 + B)) with the soloist, 1 + B with the blind machine; beside them the "
                  "tuned pairs over the sensitivity of the judge alone. On the bench of the curve the second network carries "
                  "the opposite push itself and is read as a partner",
           "by_bench_and_condition": summary(units), "units": units, "minutes": timer.minutes(), "environment": environment()}
    write_json(args.out, out)
    for r in out["by_bench_and_condition"]:
        print(r["bench"], r["condition"], "units", r["units"], "| 1+B", round(r["resonance"], 2), "q^2", round(r["strength"], 2), "p", round(r["private"], 2),
              "| partner: equation", round(r["tu_partner"], 2), "measured", round(r["measured_tu_partner"], 2),
              "| blind:", round(r["tu_blind"], 2), round(r.get("measured_tu_blind") or float("nan"), 2),
              "| soloist:", round(r["tu_soloist"], 2), round(r.get("measured_tu_soloist") or float("nan"), 2),
              "| angle partner", round(r["angle_partner"]), round(r["measured_angle_partner"]))
    print("written", args.out)


if __name__ == "__main__":
    main()
