"""E15, the threshold crossed (protocol § 21).

The judge and the trained corrector are those of E8 (results/raw/chart); the generic machine is trained again at
seven widths (results/raw/crossing). For every task, seed and width: the pair of the judge with the generic machine
and with the corrector, at the best weights as in E8, and what the proposition predicts from the measures on the
judge and on the generic machine alone. Then, task by task on the median of the seeds, the width at which the two
pairs cross, measured and predicted.

    python analysis/crossing.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from chart import predictions  # noqa: E402
from common import ROOT, med, write_json  # noqa: E402

from cognitive_resonance import estimators as est  # noqa: E402
from cognitive_resonance import raw  # noqa: E402

E8_WIDTH = {25: 7, 50: 14, 100: 28}
TOLERANCE, NEED, SAME = 3.0, 4, 0.03


def one(zh, zm_raw, val_zm, d, half) -> dict:
    y, a = d["y"], np.asarray(d["a"], float)
    zm = est.standardize(zm_raw, y, val_zm, d["val_y"])
    pred = predictions(zh, zm, a, y)
    return {"d_m": float(est.d_a(zm, y)), "d1": est.measure_pair(zh, zm, y, half)["d_pair_tuned"],
            "predicted_d1": pred["d1"], "predicted_d_blind": pred["d_blind"], "r": pred["r"], "b": pred["b"]}


def crossing(widths, diff) -> float | None:
    """The first width at which the difference goes from below to above zero, by linear interpolation."""
    for (w0, d0), (w1, d1) in zip(zip(widths, diff), zip(widths[1:], diff[1:])):
        if d0 < 0 <= d1:
            return float(w0 + (w1 - w0) * (-d0) / (d1 - d0))
    return None


def main() -> None:
    units = []
    for new in sorted((ROOT / "results" / "raw" / "crossing").glob("*.npz")):
        task, seed = new.stem.rsplit("_s", 1)
        n = np.load(new)
        rows = {}
        for v, width in E8_WIDTH.items():
            d = raw.read_chart(ROOT / "results" / "raw" / "chart" / f"{task}_v{v}_s{seed}.npz")
            assert np.array_equal(d["image"], n["image"]) and np.allclose(d["a"], n["a"]), new.name
            y = d["y"]
            names = list(d["machine_names"])
            zh = est.standardize(d["z_H"], y, d["val_z_H"], d["val_y"])
            half = est.split_halves(d["image"], y)
            i, j = names.index("M"), names.index("Mblind")
            rows[width] = one(zh, d["z_M"][i], d["val_z_M"][i], d, half)
            rows[width]["from"] = "E8"
            zb = est.standardize(d["z_M"][j], y, d["val_z_M"][j], d["val_y"])
            d_blind = est.measure_pair(zh, zb, y, half)["d_pair_tuned"]
        for k, width in enumerate(n["columns"]):
            r = one(zh, n["z_M"][k], n["val_z_M"][k], d, half)
            r["from"] = "E15"
            if int(width) in rows:                                   # 14: the half of E8 trained again
                r["d_m_of_e8"] = rows[int(width)]["d_m"]
                rows[f"{int(width)}_again"] = r
            else:
                rows[int(width)] = r
        units.append({"task": task, "seed": int(seed), "d_blind": d_blind, "widths": rows})
    tasks = sorted({u["task"] for u in units})
    widths = sorted(w for w in units[0]["widths"] if isinstance(w, int) and w >= 14) if units else []
    by_task, x1, x2 = [], {}, {}
    for t in tasks:
        us = [u for u in units if u["task"] == t]
        measured = [med([u["widths"][w]["d1"] - u["d_blind"] for u in us]) for w in widths]
        predicted = [med([u["widths"][w]["predicted_d1"] - u["widths"][w]["predicted_d_blind"] for u in us]) for w in widths]
        cm, cp = crossing(widths, measured), crossing(widths, predicted)
        changes = sum((a < 0) != (b < 0) for a, b in zip(measured, measured[1:]))
        x1[t] = bool(changes == 1 and measured[0] < 0 <= measured[-1])
        x2[t] = bool(cm is not None and cp is not None and abs(cm - cp) <= TOLERANCE)
        by_task.append({"task": t, "seeds": len(us), "measured": measured, "predicted": predicted,
                        "crossing_measured": cm, "crossing_predicted": cp})
    again = [abs(u["widths"]["14_again"]["d_m"] / u["widths"]["14_again"]["d_m_of_e8"] - 1.0) for u in units
             if "14_again" in u["widths"]]
    gaps = [abs(b["crossing_measured"] - b["crossing_predicted"]) for b in by_task
            if b["crossing_measured"] is not None and b["crossing_predicted"] is not None]
    criteria = {
        "X1": {"outcome": "pass" if all(x1.values()) and x1 else "fail", "tasks_passing": sum(x1.values()), "by_task": x1,
               "rule": "for every task, on the median of the seeds, D1 minus the pair with the corrector changes sign "
                       "only once between 14 and 28 columns, from negative to positive"},
        "X2": {"outcome": "pass" if sum(x2.values()) >= NEED else "fail", "tasks_passing": sum(x2.values()), "by_task": x2,
               "median_gap_in_columns": med(gaps) if gaps else None,
               "rule": f"the measured crossing is at most {TOLERANCE:g} columns from the predicted one, in at least {NEED} "
                       "tasks of 5"},
        "X3": {"outcome": "pass" if again and float(np.median(again)) < SAME else "fail",
               "median_relative_difference": float(np.median(again)) if again else None,
               "rule": f"at 14 columns the generic machine trained again is within {SAME:g} of the sensitivity of that of "
                       "E8, median over the units"}}
    out = {"protocol": "§ 21, E15", "units_read": len(units), "widths": widths, "by_task": by_task, "criteria": criteria,
           "median_measured_by_width": {str(w): med([u["widths"][w]["d1"] - u["d_blind"] for u in units]) for w in widths},
           "median_predicted_by_width": {str(w): med([u["widths"][w]["predicted_d1"] - u["widths"][w]["predicted_d_blind"]
                                                      for u in units]) for w in widths},
           "units": units}
    write_json(ROOT / "results" / "e15_crossing.json", out)
    for b in by_task:
        print(b["task"], "measured", [round(x, 3) for x in b["measured"]], "| crossing", b["crossing_measured"] and round(b["crossing_measured"], 1),
              "predicted", b["crossing_predicted"] and round(b["crossing_predicted"], 1))
    print({k: (v["outcome"], v.get("tasks_passing")) for k, v in criteria.items()}, criteria["X3"]["median_relative_difference"])


if __name__ == "__main__":
    main()
