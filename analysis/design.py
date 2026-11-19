"""E6, the design reading (protocol section 12 bis): who a partner is worth it for.

No new measure on the raw cues: it reads the result file of E6 (analysis/trust.py), which holds for every unit
the raw curve Tu(w) of the pair with each machine on the 49 weights, the peak weight of each pair and the
sensitivities alone, and writes results/e6_design.json.

- The weight due, w_hat_u: the peak weight of the pair (H, M0plus) in unit u. A person who gives the soloist a
  fraction f of it uses the raw weight f * w_hat_u with every machine.
- The partner for f: the branch built for the weight nearest, in ln w, to f * median(w_hat_u). Its built weight
  decides, not its result.
- The break-even of a branch: going up from the weight it is built for, the first weight at which its pair
  falls below the pair with the control soloist. The model puts it at twice the built weight.
- The floor: k = d_a(machine) / d_a(H) >= 1. In the model the pair is then at one turing or above at every
  positive weight.
- The rule of choice for the people between f1 and f2: among the machines that hold the floor, the highest
  minimum of the median curve between f1 * w_hat and f2 * w_hat; if equal, the highest mean on the same points.

    python analysis/design.py [--results results] [--ter]
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from common import ROOT, environment, med, verdict, write_json
from cognitive_resonance import model

CONTROL = "M0plus"
KONG = 0.285                                # Kong et al. (2026): the share of the weight due that people give
FRACTIONS = (1 / 16, 1 / 8, KONG, 1 / 2, 1.0)
RANGE = (KONG, 1.0)                         # the registered interval of the rule of choice
JUDGED_WEIGHTS = (0.5, 1.0, 2.0)            # G1; at 4 the two curves are almost flat
FACTOR = 1.5                                # G1: the break-even within this factor of twice the built weight
CLOSED_SHARE = 15 / 25                      # G1: units in which the break-even is closed
WINS_SHARE = 18 / 25                        # G2
BAND = 0.03                                 # G3
EXPECTED_MARGIN = 0.05                      # the expectation on the chosen machine, not a criterion
LN = np.log(model.GRID)


def at(curve, w: float) -> float:
    """The curve at weight w, by linear interpolation in ln w; outside the grid the end value holds."""
    return float(np.interp(math.log(w), LN, np.asarray(curve, float)))


def due(u: dict) -> float:
    return float(u["fixed"][CONTROL]["raw_peak"]["estimate"])


def partner_for(f: float, w_hat: float, branches: dict) -> str:
    """The branch built for the weight nearest, in ln w, to f * w_hat."""
    return min(sorted(branches, key=branches.get), key=lambda n: abs(math.log(branches[n]) - math.log(f * w_hat)))


def break_even(partner, control, built_for: float):
    """Going up from the grid weight nearest to `built_for`, the first weight at which partner - control turns
    negative (linear interpolation in ln w). None if the difference is not positive at the start, or never
    turns negative: the break-even is open."""
    diff = np.asarray(partner, float) - np.asarray(control, float)
    i0 = int(np.argmin(np.abs(LN - math.log(built_for))))
    if not diff[i0] > 0:
        return None
    for j in range(i0 + 1, diff.size):
        if diff[j] < 0:
            t = diff[j - 1] / (diff[j - 1] - diff[j])
            return float(math.exp(LN[j - 1] + t * (LN[j] - LN[j - 1])))
    return None


def points(lo: float, hi: float) -> list:
    """The two ends and the grid weights between them."""
    return [lo, *[float(w) for w in model.GRID if lo < w < hi], hi]


def choose(curves: dict, k: dict, lo: float, hi: float):
    """The rule of choice. curves: machine -> the 49 values of Tu(w); k: machine -> d_a alone over d_a of H.
    Returns the chosen machine (None if no machine holds the floor) and the table the choice was made on."""
    ws = points(lo, hi)
    table = {}
    for name, curve in curves.items():
        values = [at(curve, w) for w in ws]
        table[name] = {"k": k[name], "holds_floor": bool(k[name] >= 1.0), "worst": min(values),
                       "mean": sum(values) / len(values), "minimum_on_grid": float(np.min(curve))}
    ok = [n for n in table if table[n]["holds_floor"]]
    chosen = max(ok, key=lambda n: (table[n]["worst"], table[n]["mean"])) if ok else None
    return chosen, table


def reading(d: dict) -> dict:
    units = d["units"]
    n = len(units)
    names = list(units[0]["machines"])
    branches = dict(units[0]["branches"])
    full = len(branches) == 4
    w_hats = [due(u) for u in units]
    w_hat = med(w_hats)
    curves = [{m: u["fixed"][m]["raw_curve"] for m in names} for u in units]
    ks = [{m: u["pairs"][m]["k"] for m in names} for u in units]
    median_curve = {m: [med([curves[i][m][j] for i in range(n)]) for j in range(LN.size)] for m in names}
    median_k = {m: med([ks[i][m] for i in range(n)]) for m in names}

    # --- the turing at a fraction of the weight due
    by_fraction = []
    for f in FRACTIONS:
        row = {"fraction": f, "raw_weight": f * w_hat,
               "control_turing": med([at(curves[i][CONTROL], f * w_hats[i]) for i in range(n)])}
        if branches:
            p = partner_for(f, w_hat, branches)
            gains = [at(curves[i][p], f * w_hats[i]) - at(curves[i][CONTROL], f * w_hats[i]) for i in range(n)]
            own = [at(curves[i][p], f * units[i]["fixed"][p]["raw_peak"]["estimate"])
                   - at(curves[i][CONTROL], f * w_hats[i]) for i in range(n)]
            best = max((m for m in names if median_k[m] >= 1.0), key=lambda m: at(median_curve[m], f * w_hat),
                       default=None)
            row.update({"partner": p, "built_for_weight": branches[p],
                        "partner_turing": med([at(curves[i][p], f * w_hats[i]) for i in range(n)]),
                        "gain": med(gains), "units_with_gain": int(sum(g > 0 for g in gains)), "units": n,
                        "gain_if_the_fraction_is_of_its_own_due_weight": med(own),
                        "best_on_median_curves_seen_after": best})
        by_fraction.append(row)

    # --- the break-even of every branch
    crossing = {}
    for name, w_b in sorted(branches.items(), key=lambda kv: kv[1]):
        per_unit = [break_even(curves[i][name], curves[i][CONTROL], w_b) for i in range(n)]
        closed = [x for x in per_unit if x is not None]
        m = med(closed)
        crossing[name] = {"built_for_weight": w_b, "predicted": 2 * w_b, "closed_units": len(closed), "units": n,
                          "median": m, "median_over_predicted": None if m is None else m / (2 * w_b),
                          "on_median_curves": break_even(median_curve[name], median_curve[CONTROL], w_b),
                          "by_unit": per_unit}

    # --- the rule of choice, on the median curves and unit by unit
    lo, hi = RANGE[0] * w_hat, RANGE[1] * w_hat
    chosen, table = choose(median_curve, median_k, lo, hi)
    per_unit_choice = [choose(curves[i], ks[i], RANGE[0] * w_hats[i], RANGE[1] * w_hats[i])[0] for i in range(n)]
    choice = {"interval_of_fractions": list(RANGE), "raw_weights": [lo, hi], "chosen": chosen, "table": table,
              "units_with_the_same_choice": int(sum(c == chosen for c in per_unit_choice)), "units": n,
              "by_unit": per_unit_choice,
              "worst_minus_control": None if chosen is None else table[chosen]["worst"] - table[CONTROL]["worst"]}
    four = next((m for m, w in branches.items() if w == 4.0), None)
    choice["expectation"] = {
        "text": f"the rule chooses the branch built for 4, with an advantage over {CONTROL} in the worst case "
                f"below {EXPECTED_MARGIN} turing; not a criterion",
        "met": bool(chosen == four and choice["worst_minus_control"] < EXPECTED_MARGIN) if full else None}

    # --- the criteria
    why = None if full else "needs the four branches of E6: these files do not have them"
    need_closed, need_wins = math.ceil(CLOSED_SHARE * n - 1e-9), math.ceil(WINS_SHARE * n - 1e-9)
    judged = [m for m, w in sorted(branches.items(), key=lambda kv: kv[1]) if w in JUDGED_WEIGHTS]
    g1_ok = all(crossing[m]["closed_units"] >= need_closed and crossing[m]["median"] is not None
                and 1 / FACTOR <= crossing[m]["median_over_predicted"] <= FACTOR for m in judged)
    at_kong = next(r for r in by_fraction if r["fraction"] == KONG)
    at_due = next(r for r in by_fraction if r["fraction"] == 1.0)
    low = next((m for m, w in branches.items() if w == 0.5), None)
    g4_ok = low is not None and median_k[low] < 1.0 and all(median_k[m] >= 1.0 for m, w in branches.items()
                                                             if w != 0.5)
    criteria = {
        "G1": verdict(g1_ok if full else None, reason=why, needed_closed_units=need_closed, factor=FACTOR,
                      median_over_twice_the_built_weight={m: crossing[m]["median_over_predicted"] for m in judged},
                      closed_units={m: crossing[m]["closed_units"] for m in judged},
                      rule="the break-even of the branch built for w_b is closed in >= 15 units of 25 and its "
                           "median over the closed units is within a factor 1.5 of 2 w_b, for w_b = 0.5, 1 and 2"),
        "G2": verdict(at_kong.get("units_with_gain", 0) >= need_wins if full else None, reason=why,
                      partner=at_kong.get("partner"), units_with_gain=at_kong.get("units_with_gain"),
                      needed_units=need_wins, median_gain=at_kong.get("gain"),
                      rule=f"at f = {KONG} of the weight due the pair with the partner for f has a higher turing "
                           f"than the pair with {CONTROL} in >= 18 units of 25"),
        "G3": verdict(abs(at_due["gain"]) <= BAND if full else None, reason=why, partner=at_due.get("partner"),
                      median_gain=at_due.get("gain"), band=BAND,
                      rule="at f = 1 the median difference between the pair with the partner for f and the pair "
                           f"with {CONTROL} is within {BAND} turing"),
        "G4": verdict(g4_ok if full else None, reason=why,
                      median_k={m: median_k[m] for m in sorted(branches, key=branches.get)},
                      rule="k = d_a(machine) / d_a(H) is, in median, below 1 for the branch built for 0.5 and at "
                           "least 1 for those built for 1, 2 and 4"),
    }
    for c in criteria.values():
        if c.get("reason") is None:
            c.pop("reason", None)
    return {"due_weight": {"median": w_hat, "by_unit": w_hats}, "median_k": median_k, "by_fraction": by_fraction,
            "break_even": crossing, "choice": choice, "criteria": criteria}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=Path, default=ROOT / "results")
    ap.add_argument("--ter", action="store_true", help="the reading of E2 ter: old data, one branch, not a test")
    args = ap.parse_args()
    source = "e6_trust_from_e2ter.json" if args.ter else "e6_trust.json"
    d = json.loads((args.results / source).read_text())
    out = {"experiment": "E6, design reading", "protocol": "§ 12 bis (v2.2)",
           "status": ("a look at the data of E2 ter, with one branch: not a test" if args.ter else
                      "reading registered when the runs of E6 had just started, before any of their files was read"),
           "source": source,
           "settings": {"control": CONTROL, "fractions": list(FRACTIONS), "kong": KONG,
                        "interval_of_the_choice": list(RANGE), "grid": "49 weights, 1/64 to 64",
                        "weight_due": f"the peak weight of the pair (H, {CONTROL}), on raw log-odds",
                        "turing_at_a_fraction": "linear interpolation in ln w of the raw curve of the unit"},
           **reading(d), "environment": environment()}
    target = args.results / ("e6_design_from_e2ter.json" if args.ter else "e6_design.json")
    write_json(target, out)
    print(f"weight due (median): {out['due_weight']['median']:.2f}")
    for r in out["by_fraction"]:
        if "partner" in r:
            print(f"f = {r['fraction']:.3f}  raw w = {r['raw_weight']:5.2f}  {r['partner']:11s} gain {r['gain']:+.3f} "
                  f"in {r['units_with_gain']}/{r['units']} units")
    for name, c in out["break_even"].items():
        m = "open" if c["median"] is None else f"{c['median']:.2f}"
        print(f"break-even of {name}: {m} (predicted {c['predicted']:g}), closed in {c['closed_units']}/{c['units']}")
    ch = out["choice"]
    print(f"chosen for f in {RANGE}: {ch['chosen']}, the same in {ch['units_with_the_same_choice']}/{ch['units']} units")
    for name, row in ch["table"].items():
        print(f"   {name:11s} k {row['k']:.2f}  worst {row['worst']:.3f}  mean {row['mean']:.3f}")
    for h, c in out["criteria"].items():
        print(h, c["outcome"])
    print("written", target)


if __name__ == "__main__":
    main()
