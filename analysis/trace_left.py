"""What the trained branch leaves to the frame (a reading added after the results of E10: not a test).

T3 of protocol § 16 expected, from old data, that with the whole view the excess of the theorem would not describe
what the branch removes. It did not. This script reads `results/e10_trace.json` and asks what the branch leaves
instead: the coefficient of the frame that stays in the sum, (1 - share removed) x push of the judge (raw log-odds
per unit of frame), beside the slope of the frame's own log-likelihood ratio at that level, 2 mu_d / sd_d^2.

    python analysis/trace_left.py [--in results/e10_trace.json] [--out results/e10_trace_left.json]
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from common import ROOT, environment, med, write_json


def own_slope(level: float) -> float:
    """Slope of the log-likelihood ratio of a frame of sensitivity `level` and total variance 2 (protocol § 16)."""
    mu = level * math.sqrt(2.0 / (4.0 + level ** 2))
    return 2.0 * mu / (2.0 - mu ** 2)


def rows(units) -> list:
    out = []
    for v in sorted({u["v"] for u in units}):
        for key in sorted(units[0]["levels"], key=float):
            us = [u["levels"][key] for u in units if u["v"] == v]
            left = [(1.0 - u["share_removed"]["new"]) * u["push_raw"] for u in us]
            out.append({"v": v, "level": float(key), "units": len(us), "push_raw": med([u["push_raw"] for u in us]),
                        "left_to_the_frame": med(left), "left_range": [min(left), max(left)],
                        "frame_own_slope": own_slope(float(key)),
                        "share_removed": med([u["share_removed"]["new"] for u in us]),
                        "excess_of_the_theorem": med([u["theorem"]["excess"] for u in us]),
                        "units_where_old_pair_is_below_the_judge_alone":
                            sum(u["right"]["H+old"] < u["right"]["H"] for u in us),
                        "right_old_minus_soloist_points": med([100 * (u["right"]["H+old"] - u["right"]["H+M"]) for u in us])})
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="src", type=Path, default=ROOT / "results" / "e10_trace.json")
    ap.add_argument("--out", type=Path, default=ROOT / "results" / "e10_trace_left.json")
    args = ap.parse_args()
    d = json.loads(args.src.read_text())
    out = {"experiment": "E10, what the branch leaves to the frame", "protocol": "§ 16, on the results of E10",
           "status": "a reading added on 20 November 2026, after the results of E10 were seen: not a test",
           "how": "per unit and level: (1 - share of the push removed by the branch built for the level) x raw push of "
                  "the judge, beside the slope of the frame's own log-likelihood ratio, 2 mu_d / sd_d^2",
           "by_v_and_level": rows(d["units"]), "environment": environment()}
    write_json(args.out, out)
    for r in out["by_v_and_level"]:
        print("v", r["v"], "level", r["level"], "left", round(r["left_to_the_frame"], 2), "own", round(r["frame_own_slope"], 2),
              "removed", round(r["share_removed"], 2), "excess", round(r["excess_of_the_theorem"], 2))
    print("written", args.out)


if __name__ == "__main__":
    main()
