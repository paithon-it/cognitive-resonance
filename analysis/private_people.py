"""The theorem read on people already measured (a reading added after the results: not a test).

turing-meter's experiment `angle-between-errors` measured, for comparisons of people with a
language model on four sources, the sensitivity of the people (d_H), of the machine (d_m) and
the correlation of their errors (rho). Those numbers have the people's biases still in them.
The 74 comparisons read here are those of its first hypothesis: ChatBench on all the tasks,
alphaNLI, and HAIID without the answers given by the dataset (text: sarcasm and census; images:
cities and art).
For such a pair the theorem of protocol § 14 says two things:

- what the person adds to the machine today is P_obs = (d_H - rho d_m)^2 / (1 - rho^2);
- if a share B / (1 + B) of the person's noise were a bias that a machine can read, a machine
  built on the person would add P_obs * B / (1 - rho^2 (1 + B)) to the squared sensitivity of
  the pair.

Nothing here measures B: no trace of the people's errors is in those data. The script says how
much there would be to gain for a given B, and that is all.

    python analysis/private_people.py --turing-meter PATH_TO_CLONE [--out results/e4_private.json]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

from common import ROOT, environment, git_commit, med, write_json
from cognitive_resonance import chart as theorem

BIASES = (0.002, 0.01, 0.25, 1.0, 3.0)   # B. The first two are the size of a documented bias (see SHIFT); the others
                                          # are scenarios: a fifth, a half, three quarters of the person's noise
# Chen, Moskowitz and Shue (2016): asylum judges are up to 3.3 percentage points more likely to reject a case after
# granting the previous one, around a grant rate of 0.29. `chart.bias_from_shift` turns it into B = 0.002 when the
# noise is everything that makes the answer vary, differences between cases included; five times as much, 0.01,
# stands for a judge whose own noise is a fifth of that.
SHIFT = {"source": "Chen, Moskowitz and Shue (2016), asylum judges", "shift": 0.033, "base_rate": 0.29}
MAP = Path("experiments") / "angle-between-errors" / "results" / "map.json"


HAIID = {"sarcasm": "haiid-text", "census": "haiid-text", "cities": "haiid-images", "art": "haiid-images"}


def group(row: dict) -> str | None:
    """The family of a comparison, or None when it is not one of the 74 of turing-meter's first hypothesis."""
    if "dh" not in row:
        return None                                        # excluded there: one of the two at or below chance
    if row["source"] == "chatbench":
        return "chatbench" if row.get("task") == "all" else None
    if row["source"] == "alphanli":
        return "alphanli"
    if row["source"] == "haiid" and row.get("mode") != "given":
        return HAIID.get(row.get("task"))
    return None


def read(row: dict) -> dict:
    d_h, d_m, rho = float(row["dh"]), float(row["dm"]), float(row["rho"])
    p_obs = theorem.private_observed(d_h, d_m, rho)
    d1 = math.sqrt(d_m ** 2 + p_obs)                       # the best pair as it is, bias in
    out = {"source": group(row), "task": row.get("task"), "machine": row.get("machine"), "mode": row.get("mode"),
           "d_h": d_h, "d_m": d_m, "rho": rho, "k": d_m / d_h, "private_observed": p_obs, "pair_now": d1,
           "pair_now_over_machine": d1 / d_m, "pair_now_over_best_alone": d1 / max(d_h, d_m), "if_bias": {}}
    for b in BIASES:
        g = theorem.gain_if_bias(d_h, d_m, rho, b)
        out["if_bias"][f"{b:g}"] = None if g is None else {
            "gain_squared": g, "pair": math.sqrt(d1 ** 2 + g), "pair_over_pair_now": math.sqrt(d1 ** 2 + g) / d1,
            "pair_over_machine": math.sqrt(d1 ** 2 + g) / d_m,
            "person_alone_without_the_bias_over_person": math.sqrt(1.0 + b)}
    return out


def summary(rows: list) -> list:
    out = []
    for source in sorted({r["source"] for r in rows}) + ["all"]:
        us = rows if source == "all" else [r for r in rows if r["source"] == source]
        m = lambda f: med([f(u) for u in us])                                                # noqa: E731
        row = {"source": source, "comparisons": len(us), "d_h": m(lambda u: u["d_h"]), "d_m": m(lambda u: u["d_m"]),
               "rho": m(lambda u: u["rho"]), "k": m(lambda u: u["k"]), "private_observed": m(lambda u: u["private_observed"]),
               "pair_now_over_machine": m(lambda u: u["pair_now_over_machine"]),
               "pair_now_over_best_alone": m(lambda u: u["pair_now_over_best_alone"]),
               "people_stronger_than_machine": sum(u["d_h"] > u["d_m"] for u in us), "if_bias": {}}
        for b in BIASES:
            ok = [u["if_bias"][f"{b:g}"] for u in us if u["if_bias"][f"{b:g}"] is not None]
            gains = [x["pair_over_pair_now"] - 1 for x in ok]
            row["if_bias"][f"{b:g}"] = {"comparisons": len(ok), "median_gain_over_pair_now": med(gains),
                                        "lowest": min(gains) if gains else None, "highest": max(gains) if gains else None,
                                        "median_pair_over_machine": med([x["pair_over_machine"] for x in ok])}
        out.append(row)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--turing-meter", required=True, help="path of a local turing-meter clone")
    ap.add_argument("--out", type=Path, default=ROOT / "results" / "e4_private.json")
    args = ap.parse_args()
    f = Path(args.turing_meter) / MAP
    if not f.is_file():
        raise SystemExit(f"not a turing-meter clone with the results of angle-between-errors: {args.turing_meter}")
    rows = [read(r) for r in json.loads(f.read_text())["comparisons"] if group(r)]
    out = {"experiment": "E4, the theorem on people already measured", "protocol": "§ 14, on the comparisons of turing-meter",
           "status": "a reading added on 20 November 2026: not a test, and no measure of the people's bias",
           "source": {"repository": "https://github.com/paithon-it/turing-meter", "commit": git_commit(args.turing_meter),
                      "file": str(MAP), "sha256": hashlib.sha256(f.read_bytes()).hexdigest()},
           "how": "per comparison, from d_H, d_m and rho as published (the people's biases are in them): P_obs, the "
                  "best pair as it is, and the best pair if a share B / (1 + B) of the person's noise were a bias a "
                  "machine can read, for B = " + ", ".join(f"{b:g}" for b in BIASES),
           "documented_bias": {**SHIFT, "b": theorem.bias_from_shift(SHIFT["shift"], SHIFT["base_rate"]),
                               "how": "probit: push = shift / phi(Phi^-1(base rate)), B = push^2 q (1 - q); the noise "
                                      "is everything that varies at a given trace"},
           "by_source": summary(rows), "comparisons": rows, "environment": environment()}
    write_json(args.out, out)
    for r in out["by_source"]:
        print(r["source"], r["comparisons"], "d_H", round(r["d_h"], 2), "d_m", round(r["d_m"], 2), "rho", round(r["rho"], 2),
              "pair now / machine", round(r["pair_now_over_machine"], 3),
              {b: (round(100 * x["median_gain_over_pair_now"], 1), x["comparisons"]) for b, x in r["if_bias"].items()})
    print("written", args.out)


if __name__ == "__main__":
    main()
