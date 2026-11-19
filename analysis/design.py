"""Which weight should a partner be built for? (a design aid on the curves of E6; not a test of the protocol)

E6 builds four partners, each for a weight, and measures the turing of the pair with each of them, and with the
soloist, at every weight on the grid. Whoever builds a machine for other people does not know the weight they
will give it. This script takes a belief about that weight, as shares over a few weights, and reports for
every machine the expected turing of the pair under that belief, the worst case over the same weights, and the
share of right answers of the machine used alone. Which machine to choose is then a matter of policy, and the
policy is the function `choose` below.

    python analysis/design.py [--results results] [--file e6_trust_from_e2ter.json] [--belief 0.25:0.5,1:0.3,4:0.2]
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path

from common import ROOT

GRID = [math.exp(math.log(1 / 64) + i * math.log(64 * 64) / 48) for i in range(49)]
# Kong et al. (2026) estimate that the participants of ChatBench give the machine about 0.285 of the weight it
# deserves; the default belief puts most people there and a few at face value and beyond.
DEFAULT_BELIEF = {0.285: 0.6, 1.0: 0.3, 4.0: 0.1}


def at(curve, w: float) -> float:
    """The curve at weight w, by linear interpolation in ln w."""
    x = (math.log(w) - math.log(GRID[0])) / (math.log(GRID[-1]) - math.log(GRID[0])) * (len(GRID) - 1)
    i = min(max(int(math.floor(x)), 0), len(GRID) - 2)
    return curve[i] + (x - i) * (curve[i + 1] - curve[i])


def table(d: dict, belief: dict) -> list:
    """Per machine: expected and worst turing of the pair under the belief, and right answers alone."""
    total = sum(belief.values())
    rows = []
    for row in d["by_machine"]:
        name = row["machine"]
        curve = [statistics.median(u["fixed"][name]["raw_curve"][i] for u in d["units"]) for i in range(len(GRID))]
        values = {w: at(curve, w) for w in belief}
        rows.append({"machine": name, "built_for_weight": row.get("built_for_weight"),
                     "expected_turing": sum(values[w] * share for w, share in belief.items()) / total,
                     "worst_turing": min(values.values()), "accuracy_alone": row["accuracy_alone"],
                     "accuracy_judge": row["accuracy_person_alone"], "turing_by_weight": values})
    return rows


def choose(rows: list) -> str:
    """The machine to build, given the table.

    TODO(author): this is the policy, and it is a choice, not a computation. Each row has `expected_turing` (the
    pair, averaged over the weights people are believed to give), `worst_turing` (the pair with the user who
    gives the least favourable weight), `accuracy_alone` (the machine used without its person) and
    `accuracy_judge` (the judge without the machine). Options: the highest expected turing; the highest worst
    case (nobody is harmed much); either of them among the machines that stay above a floor of accuracy alone.
    A floor with no free number: the machine alone is worth at least the judge alone. In the model the pair
    then stays at one turing or above for every positive weight, since the curve of equation (2) has its
    minimum over positive weights at one of the two ends, 1 at weight zero and k at very large weights.
    Return the name of the chosen machine.
    """
    return max(rows, key=lambda r: r["expected_turing"])["machine"]          # a placeholder: expected value only


def parse_belief(text: str) -> dict:
    return {float(w): float(s) for w, s in (pair.split(":") for pair in text.split(","))}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=Path, default=ROOT / "results")
    ap.add_argument("--belief", type=parse_belief, default=DEFAULT_BELIEF)
    ap.add_argument("--file", default="e6_trust.json", help="e6_trust_from_e2ter.json for the reading of E2 ter")
    args = ap.parse_args()
    d = json.loads((args.results / args.file).read_text())
    rows = table(d, args.belief)
    print("belief (weight: share):", args.belief)
    print(f"the judge alone: {100 * rows[0]['accuracy_judge']:.1f}% right")
    print(f"{'machine':14s} {'built for':>9s} {'expected Tu':>12s} {'worst Tu':>9s} {'right alone':>12s}")
    for r in rows:
        built = "" if r["built_for_weight"] is None else f"{r['built_for_weight']:g}"
        print(f"{r['machine']:14s} {built:>9s} {r['expected_turing']:12.3f} {r['worst_turing']:9.3f} "
              f"{100 * r['accuracy_alone']:11.1f}%")
    print("chosen:", choose(rows))


if __name__ == "__main__":
    main()
