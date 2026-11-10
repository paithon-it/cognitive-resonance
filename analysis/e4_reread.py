"""E4 (protocol § 8): the pairs already measured in "Intelligenza apparente", reread.

From each published angle: Q = 1/(2 cos(phi/2)) and the apparent turing the pair would have
at equal sensitivity (k = 1), which is 2Q; with the band and, from v1.2, the null. No criteria.

    python analysis/e4_reread.py [--out results/e4_reread.json]
"""
from __future__ import annotations

import argparse
import math

from common import ROOT, environment, write_json
from cognitive_resonance import model

PAIRS = [
    ("ChatBench (people and machine)", 81.0),
    ("αNLI (people and machine)", 64.0),
    ("HAIID text (people and machine)", 65.0),
    ("HAIID images (people and machine)", 78.0),
    ("objects, model-vs-human (people and machine)", 59.0),
    ("two people, with each other (higher)", 66.0),
    ("two people, with each other (lower)", 54.0),
    ("two networks, with each other (upper end)", 43.0),
    ("two networks, with each other (lower end)", 22.0),
    ("reference: independent errors", 90.0),
]


def row(name, phi):
    rho = math.cos(math.radians(phi))
    q = float(model.q_factor(phi))
    low, high = model.band_edges_k1(phi)
    near, far = model.null_edges(1.0, rho)
    return {"pair": name, "phi": phi, "rho": rho, "q": q, "tu_app_k1": model.tu_app(1.0, rho),
            "two_q": 2 * q, "over_pythagoras": model.tu_app(1.0, rho) / math.sqrt(2.0),
            "peak_tu2_minus_1": math.tan(math.radians(phi) / 2) ** 2,
            "band_low": low, "band_high": high, "bandwidth": high / low,
            "null_near": near, "null_far": far, "null_width": far / near, "null_width_diff": abs(far - near)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "results" / "e4_reread.json"))
    args = ap.parse_args()
    rows = [row(n, p) for n, p in PAIRS]
    write_json(args.out, {
        "experiment": "E4", "protocol": "PROTOCOL.md § 8 (v1.2)",
        "source": "angles published in 'Intelligenza apparente' (https://paithon.it/intelligenza-apparente/)",
        "notes": ["k = 1: the apparent turing at equal sensitivity, Tu_app = 2Q = 1/cos(phi/2)",
                  "bandwidth = w_high / w_low of the band where Tu^2 - 1 stays above half its peak",
                  "null width (v1.2): ratio far/near of the edges of |Tu| < 1/2 around w0 = -1, and the difference"],
        "pairs": rows, "environment": environment()})
    for r in rows:
        print(f"{r['pair']:48s} phi {r['phi']:5.1f}  Q {r['q']:.3f}  Tu_app {r['tu_app_k1']:.3f}  band {r['bandwidth']:.1f}")


if __name__ == "__main__":
    main()
