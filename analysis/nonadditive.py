"""A bias that depends on how unsure the person is: a simulation added after the reviews (not a test).

In the proposition the push is beta * a, the same on every case. Here the person leans on the trace more when the
case is doubtful: the push is beta * a * k(h), with k(h) = c exp(-h^2 / (2 tau^2)) and c such that the push has the
same variance beta^2 for every tau. tau large: the additive case. tau small: the trace weighs only where the clean
cue h is close to zero. Three ways of deciding are compared: the judgment as it is, a linear corrector that removes
the slope of the judgment on the trace (what the bench trains), and a corrector that also knows how the push changes
with the judgment, and removes beta * a * k(H).

    python analysis/nonadditive.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ROOT, write_json  # noqa: E402

from cognitive_resonance.estimators import d_a, slope_within  # noqa: E402

N, D = 400_000, 2.0
TAUS = (None, 2.0, 1.0, 0.5)
BETAS = (0.5, 0.85)


def one(beta: float, tau: float | None, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, N)
    h = D * (y - 0.5) + rng.standard_normal(N)
    a = rng.standard_normal(N)
    shape = (lambda x: np.ones_like(x)) if tau is None else (lambda x: np.exp(-x * x / (2 * tau * tau)))
    c = 1.0 / np.sqrt(np.mean(shape(h) ** 2))
    k = lambda x: c * shape(x)                                                                    # noqa: E731
    judgment = h + beta * a * k(h)
    slope = float(slope_within(judgment, a, y))
    linear = judgment - slope * a
    shaped = judgment - beta * a * k(judgment)
    right = lambda z, where: float(((z[where] > 0) == (y[where] == 1)).mean())                     # noqa: E731
    clear, doubtful = np.abs(h) > 1.5, np.abs(h) < 0.5
    base = float(d_a(h, y))
    out = {"beta": beta, "tau": tau, "slope_removed": slope, "d_clean": base}
    for name, z in (("as_it_is", judgment), ("linear", linear), ("shaped", shaped)):
        out[name] = {"d": float(d_a(z, y)), "kept": float(d_a(z, y)) / base, "right": right(z, slice(None)),
                     "right_clear": right(z, clear), "right_doubtful": right(z, doubtful)}
    lost = base - out["as_it_is"]["d"]
    out["linear_gives_back"] = (out["linear"]["d"] - out["as_it_is"]["d"]) / lost if lost > 1e-3 else None
    out["shaped_gives_back"] = (out["shaped"]["d"] - out["as_it_is"]["d"]) / lost if lost > 1e-3 else None
    out["share_clear"], out["share_doubtful"] = float(clear.mean()), float(doubtful.mean())
    return out


def main() -> None:
    rows = [one(b, t, 2026) for b in BETAS for t in TAUS]
    write_json(ROOT / "results" / "nonadditive.json", {"what": "simulation, no data", "n": N, "d": D, "rows": rows})
    for r in rows:
        print(r["beta"], r["tau"], "kept:", round(r["as_it_is"]["kept"], 3), round(r["linear"]["kept"], 3),
              round(r["shaped"]["kept"], 3), "| linear gives back", None if r["linear_gives_back"] is None else round(r["linear_gives_back"], 2),
              "| clear:", round(100 * r["as_it_is"]["right_clear"], 2), round(100 * r["linear"]["right_clear"], 2),
              "| doubtful:", round(100 * r["as_it_is"]["right_doubtful"], 1), round(100 * r["linear"]["right_doubtful"], 1),
              round(100 * r["shaped"]["right_doubtful"], 1), "| right:", round(100 * r["as_it_is"]["right"], 2),
              round(100 * r["linear"]["right"], 2), round(100 * r["shaped"]["right"], 2))


if __name__ == "__main__":
    main()
