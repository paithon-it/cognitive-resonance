"""E13, two readings added after the confirmation run (not tests).

1. The curve of personalisation. The corrector of a person learns from the n trials just before the second half
   (n = 50, 100, 200, 400, 800), the corrector of the group from the first halves of the other people of the file,
   the corrector in between pulls the first towards the second with the weight chosen for that file by E13. All
   three are measured on the second half of the person. It says after how many trials knowing the person gives more
   than knowing people like them, if ever.
2. What the definition of bias leaves out. The readable bias of E13 is a push that changes from trial to trial; a
   threshold in the wrong place and a confidence that orders the cases badly are outside it. Here, for every
   person: the best threshold learned on the first half and applied to the second, and a map from each step of
   the judgment (answer and confidence) to the log-odds of the class measured on the first half, read on the
   second.

    python analysis/people_more.py [--data data/confidence_db]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import people_bias as pb  # noqa: E402
from common import ROOT, write_json  # noqa: E402

from cognitive_resonance import readable  # noqa: E402
from cognitive_resonance.estimators import d_a  # noqa: E402

SIZES = (50, 100, 200, 400, 800)
RESAMPLES = 2000


def med_ci(values_by_file: dict, rng) -> dict:
    groups = [np.asarray(v, float) for v in values_by_file.values() if len(v)]
    if not groups:
        return {"median": None, "lo": None, "hi": None, "people": 0, "files": 0}
    pooled = np.concatenate(groups)
    draws = [np.median(np.concatenate([groups[j] for j in rng.integers(0, len(groups), len(groups))]))
             for _ in range(RESAMPLES)]
    lo, hi = np.quantile(draws, [0.025, 0.975])
    return {"median": float(np.median(pooled)), "mean": float(pooled.mean()), "lo": float(lo), "hi": float(hi),
            "people": int(pooled.size), "files": len(groups)}


def prepared(p: dict) -> dict:
    g_all = readable.graded_judgment(p["response"], p["confidence"])
    a, g, y = readable.history(p["response"], g_all), g_all[1:], p["stimulus"][1:]
    half = g.size // 2
    return {"g_all": g_all, "a": a, "g": g, "y": y, "half": half, "response": p["response"],
            "first": readable.moments(g[:half], a[:half], y[:half])}


def curve_of_file(people: list, weight: float) -> list:
    """For every person and size: gain of the three correctors on the second half."""
    xtx = sum(m["first"][0] for m in people)
    xty = sum(m["first"][1] for m in people)
    n_all = sum(m["half"] for m in people)
    out = []
    for m in people:
        others = (xtx - m["first"][0], xty - m["first"][1], n_all - m["half"])
        h, second = m["half"], slice(m["half"], None)
        test = (m["g"][second], m["a"][second], m["y"][second])
        row = {"n_first": h, "group": readable.gain(*test, readable.slopes(others[0], others[1])),
               "group_moments": readable.gain_moments(*test, readable.slopes(others[0], others[1])), "sizes": {}}
        for n in SIZES:
            if n > h or row["group"] is None:
                continue
            last = slice(h - n, h)
            own = readable.moments(m["g"][last], m["a"][last], m["y"][last])[:2]
            mine, shrunk = readable.slopes(*own), readable.shrunk_slopes(own, *others, weight)
            # the same gains with the sensitivity that does not see the order inside one step of the scale (added
            # after a fourth review: the two measures tell the variance removed from the ties the trace breaks)
            row["sizes"][n] = {"personal": readable.gain(*test, mine), "between": readable.gain(*test, shrunk),
                               "personal_moments": readable.gain_moments(*test, mine),
                               "between_moments": readable.gain_moments(*test, shrunk)}
        out.append(row)
    return out


def threshold_and_map(m: dict) -> dict:
    """The best threshold and the map of the steps, learned on the first half and read on the second."""
    g, y, half = m["g"], m["y"], m["half"]
    lows, highs = m["g_all"][m["response"] == 0], m["g_all"][m["response"] == 1]
    if not (lows.size and highs.size):
        return {}
    own = 0.5 * (lows.max() + highs.min())
    first, second = slice(0, half), slice(half, None)
    steps = np.unique(m["g_all"])
    cuts = np.concatenate([[steps[0] - 1.0], 0.5 * (steps[:-1] + steps[1:]), [steps[-1] + 1.0]])
    right = lambda cut, part: float(((g[part] > cut) == (y[part] == 1)).mean())               # noqa: E731
    best = max(cuts, key=lambda c: (right(c, first), -abs(c - own)))                           # ties: the nearest to its own
    # the map: log-odds of class 1 on every step, with half a case added to each cell
    value = {}
    for s in steps:
        on = g[first] == s
        value[s] = float(np.log((np.sum(y[first][on] == 1) + 0.5) / (np.sum(y[first][on] == 0) + 0.5)))
    mapped = np.array([value[s] for s in g[second]])
    before = float(d_a(g[second], y[second]))
    return {"accuracy_own": right(own, second), "accuracy_best_threshold": right(best, second),
            "accuracy_map": float(((mapped > 0) == (y[second] == 1)).mean()),
            "gain_map": float(d_a(mapped, y[second])) / before if before >= readable.MIN_D else None,
            "threshold_moved": bool(best != own),
            "steps_out_of_order": int(np.sum(np.diff([value[s] for s in steps]) < 0))}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT / "data" / "confidence_db"))
    args = ap.parse_args()
    e13 = json.loads((ROOT / "results" / "e13_people_bias.json").read_text())
    weights = {}
    for r in e13["people"]:
        weights[r["file"]] = r["weight_of_the_group"]
    rng = np.random.default_rng(1314)
    out = {"what": "readings added after the confirmation run of E13; no criterion", "sizes": list(SIZES)}
    for part in ("confirmation", "exploration"):
        curves, extra = {}, {}
        for entry in e13["files"]:
            if entry["status"] != "included" or entry["split"] != part:
                continue
            people, _ = pb.read_file(Path(args.data) / entry["file"])
            ms = [prepared(people[k]) for k in people if pb.included(people[k])]
            curves[entry["file"]] = curve_of_file(ms, weights[entry["file"]])
            extra[entry["file"]] = [t for t in (threshold_and_map(m) for m in ms) if t]
        rows, long = [], []
        for n in SIZES:
            by = lambda key, n=n, only=None: {f: [r["sizes"][n][key] for r in rs if n in r["sizes"] and r["sizes"][n][key] is not None  # noqa: E731
                                                  and (only is None or r["n_first"] >= only)] for f, rs in curves.items()}
            group = lambda n=n, only=None: {f: [r["group"] for r in rs if n in r["sizes"] and (only is None or r["n_first"] >= only)]  # noqa: E731
                                            for f, rs in curves.items()}
            rows.append({"n": n, "personal": med_ci(by("personal"), rng), "between": med_ci(by("between"), rng),
                         "group": med_ci(group(), rng)})
            long.append({"n": n, "personal": med_ci(by("personal", only=SIZES[-1]), rng),
                         "between": med_ci(by("between", only=SIZES[-1]), rng),
                         "group": med_ci(group(only=SIZES[-1]), rng)})
        rng2 = np.random.default_rng(1315)                    # its own stream: the intervals above stay what they were
        for row, lrow in zip(rows, long):
            n = row["n"]
            for target, only in ((row, None), (lrow, SIZES[-1])):
                for key in ("personal_moments", "between_moments"):
                    target[key] = med_ci({f: [r["sizes"][n][key] for r in rs if n in r["sizes"] and r["sizes"][n][key] is not None
                                              and (only is None or r["n_first"] >= only)] for f, rs in curves.items()}, rng2)
                target["group_moments"] = med_ci({f: [r["group_moments"] for r in rs if n in r["sizes"] and r["group_moments"] is not None
                                                      and (only is None or r["n_first"] >= only)] for f, rs in curves.items()}, rng2)
        # paired, person by person: how often the personal corrector beats the group one at each size
        for row in rows:
            n = row["n"]
            pairs = [(r["sizes"][n]["personal"], r["group"]) for rs in curves.values() for r in rs
                     if n in r["sizes"] and r["sizes"][n]["personal"] is not None]
            row["share_personal_above_group"] = float(np.mean([a > b for a, b in pairs])) if pairs else None
            row["median_personal_minus_group"] = float(np.median([a - b for a, b in pairs])) if pairs else None
        t = {k: {f: [x[k] for x in xs if x.get(k) is not None] for f, xs in extra.items()}
             for k in ("accuracy_own", "accuracy_best_threshold", "accuracy_map", "gain_map")}
        diff = lambda a, b: {f: [x[a] - x[b] for x in xs] for f, xs in extra.items()}                 # noqa: E731
        flat = [x for xs in extra.values() for x in xs]
        out[part] = {"curve": rows, "curve_people_with_800": long,
                     "threshold": {"accuracy_own": med_ci(t["accuracy_own"], rng),
                                   "accuracy_best_threshold": med_ci(t["accuracy_best_threshold"], rng),
                                   "change": med_ci(diff("accuracy_best_threshold", "accuracy_own"), rng),
                                   "share_threshold_moved": float(np.mean([x["threshold_moved"] for x in flat])),
                                   "share_up": float(np.mean([x["accuracy_best_threshold"] > x["accuracy_own"] for x in flat])),
                                   "share_down": float(np.mean([x["accuracy_best_threshold"] < x["accuracy_own"] for x in flat]))},
                     "map": {"accuracy_map": med_ci(t["accuracy_map"], rng),
                             "change": med_ci(diff("accuracy_map", "accuracy_own"), rng),
                             "gain": med_ci(t["gain_map"], rng),
                             "share_with_steps_out_of_order": float(np.mean([x["steps_out_of_order"] > 0 for x in flat])),
                             "share_up": float(np.mean([x["accuracy_map"] > x["accuracy_own"] for x in flat])),
                             "share_down": float(np.mean([x["accuracy_map"] < x["accuracy_own"] for x in flat]))}}
    write_json(ROOT / "results" / "e13_people_more.json", out)
    c = out["confirmation"]
    for row in c["curve"]:
        print(row["n"], row["personal"]["people"], "personal", round(row["personal"]["median"], 4), "between",
              round(row["between"]["median"], 4), "group", round(row["group"]["median"], 4), "| personal>group",
              round(row["share_personal_above_group"], 3))
    print("with 800:", [(r["n"], r["personal"]["people"], round(r["personal"]["median"], 4), round(r["between"]["median"], 4),
                         round(r["group"]["median"], 4)) for r in c["curve_people_with_800"]])
    print("with 800, not seeing the order inside a step:", [(r["n"], round(r["personal_moments"]["median"], 4),
          round(r["between_moments"]["median"], 4), round(r["group_moments"]["median"], 4)) for r in c["curve_people_with_800"]])
    th, mp = c["threshold"], c["map"]
    print("threshold: own", round(th["accuracy_own"]["median"], 4), "best", round(th["accuracy_best_threshold"]["median"], 4),
          "change median", round(th["change"]["median"], 4), "mean", round(th["change"]["mean"], 4), "moved",
          round(th["share_threshold_moved"], 3), "up/down", round(th["share_up"], 3), round(th["share_down"], 3))
    print("map: change median", round(mp["change"]["median"], 4), "mean", round(mp["change"]["mean"], 4), "gain",
          round(mp["gain"]["median"], 4), (round(mp["gain"]["lo"], 4), round(mp["gain"]["hi"], 4)), "out of order",
          round(mp["share_with_steps_out_of_order"], 3), "up/down", round(mp["share_up"], 3), round(mp["share_down"], 3))


if __name__ == "__main__":
    main()
