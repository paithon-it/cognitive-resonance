"""E6, the push of the branch (a reading added after the results: not a test).

The model of the knob says that a branch built for the weight w_c takes away a push b / w_c,
where b is the slope of the judge's cue on the frame: the pair then peaks where the push
cancels, at b / c, and beats the same machine without the branch up to 2 b / c. The design
reading of protocol § 12 bis found the break-even against the control soloist M0plus earlier
than twice the built weight (0.96, 1.83 and 3.03 against 1, 2 and 4). Two things could do that:
a branch that does not cancel the push exactly, or a control that is not the same machine
without the branch (M0plus is M0 trained ten more epochs).

This script measures both on the raw files of E6. Per unit, with the log-odds as they come out
of the networks: b, the within-class slope of z_H on the frame; c, minus the slope of the
branch (z_Mramo_w - z_M0) on the frame; the break-even of each partner against M0plus (the
registered comparison) and against M0 (the same machine without the branch), on the measured
curves of results/e6_trust.json.

    python analysis/push.py [--raw results/raw/trust] [--trust results/e6_trust.json] [--out results/e6_push.json]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from common import ROOT, environment, med, write_json
from cognitive_resonance import estimators as est
from cognitive_resonance import raw
from design import break_even

SAME, CONTROL = "M0", "M0plus"


def slopes(path) -> dict:
    """The push of the judge and of every branch in one raw file, in raw log-odds per unit of frame."""
    d = raw.read_trust(path)
    y, a = d["y"], np.asarray(d["a"], float)
    z = {n: np.asarray(d["z_M"][i], float) for i, n in enumerate(d["machine_names"])}
    return {"task": d["task"], "seed": d["seed"], "b": float(est.slope_within(np.asarray(d["z_H"], float), a, y)),
            "slope_of_M0": float(est.slope_within(z[SAME], a, y)),
            "slope_of_M0plus": float(est.slope_within(z[CONTROL], a, y)),
            "c": {n: -float(est.slope_within(z[n] - z[SAME], a, y)) for n in z if n.startswith("Mramo")}}


def reading(trust: dict, rows: list) -> dict:
    """Join the slopes with the measured curves of e6_trust.json and summarise, branch by branch."""
    units = {(u["task"], u["seed"]): u for u in trust["units"]}
    per, out = {}, {"units": len(rows), "median_b": med([r["b"] for r in rows]),
                    "median_abs_slope_of_M0": med([abs(r["slope_of_M0"]) for r in rows]),
                    "median_abs_slope_of_M0plus": med([abs(r["slope_of_M0plus"]) for r in rows])}
    for r in rows:
        u = units[(r["task"], r["seed"])]
        for name, w in u["branches"].items():
            c = r["c"][name]
            curve = lambda m: u["fixed"][m]["raw_curve"]                                  # noqa: E731
            per.setdefault(name, []).append({
                "task": r["task"], "seed": r["seed"], "built_for_weight": float(w), "b": r["b"], "c": c,
                "b_over_c": r["b"] / c if c > 0 else None,
                "push_left_at_the_built_weight": 1.0 - float(w) * c / r["b"],
                "measured_peak": u["fixed"][name]["raw_peak"]["estimate"],
                "break_even_against_control": break_even(curve(name), curve(CONTROL), float(w)),
                "break_even_against_same_machine": break_even(curve(name), curve(SAME), float(w))})
    by_branch = {}
    for name in sorted(per, key=lambda n: per[n][0]["built_for_weight"]):
        us, w = per[name], per[name][0]["built_for_weight"]
        closed = lambda k: [x[k] for x in us if x[k] is not None]                           # noqa: E731
        ratio = [x["b_over_c"] / w for x in us if x["b_over_c"] is not None]
        same, control = closed("break_even_against_same_machine"), closed("break_even_against_control")
        by_branch[name] = {
            "built_for_weight": w, "median_c": med([x["c"] for x in us]), "median_b_over_c": med(closed("b_over_c")),
            "median_b_over_c_over_built_weight": med(ratio), "range_b_over_c_over_built_weight": [min(ratio), max(ratio)],
            "median_push_left_at_the_built_weight": med([x["push_left_at_the_built_weight"] for x in us]),
            "median_abs_push_left": med([abs(x["push_left_at_the_built_weight"]) for x in us]),
            "largest_abs_push_left": max(abs(x["push_left_at_the_built_weight"]) for x in us),
            "median_measured_peak": med([x["measured_peak"] for x in us]),
            "predicted_break_even": 2.0 * w,
            "break_even_against_same_machine": {"median": med(same) if same else None, "closed_units": len(same),
                                                "range": [min(same), max(same)] if same else None,
                                                "median_over_predicted": med(same) / (2.0 * w) if same else None},
            "break_even_against_control": {"median": med(control) if control else None, "closed_units": len(control),
                                           "range": [min(control), max(control)] if control else None,
                                           "median_over_predicted": med(control) / (2.0 * w) if control else None},
            "units": len(us), "by_unit": us}
    out["by_branch"] = by_branch
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", type=Path, default=ROOT / "results" / "raw" / "trust")
    ap.add_argument("--trust", type=Path, default=ROOT / "results" / "e6_trust.json")
    ap.add_argument("--out", type=Path, default=ROOT / "results" / "e6_push.json")
    args = ap.parse_args()
    files = raw.list_units(args.raw, raw.REPHASE_RE)
    if not files:
        raise SystemExit(f"no trust files in {args.raw}")
    trust = json.loads(args.trust.read_text())
    out = {"experiment": "E6, the push of the branch", "protocol": "§ 12 and § 12 bis, on the files of E6",
           "status": "a reading added on 20 November 2026, after the results of E6 were seen: not a test",
           "how": "raw log-odds; b = within-class slope of z_H on the frame, c = minus the slope of (z_Mramo_w - z_M0); "
                  "break-evens on the measured raw curves of e6_trust.json, as in analysis/design.py",
           **reading(trust, [slopes(f) for f in files]), "environment": environment()}
    write_json(args.out, out)
    for name, s in out["by_branch"].items():
        print(f"{name}: b/c {s['median_b_over_c']:.3f} (built for {s['built_for_weight']:g}), push left "
              f"{s['median_push_left_at_the_built_weight']:+.3f}; break-even against M0 "
              f"{s['break_even_against_same_machine']['median']:.2f} in {s['break_even_against_same_machine']['closed_units']}"
              f"/{s['units']}, against M0plus {s['break_even_against_control']['median']:.2f} in "
              f"{s['break_even_against_control']['closed_units']}/{s['units']}, predicted {s['predicted_break_even']:g}")
    print("written", args.out)


if __name__ == "__main__":
    main()
