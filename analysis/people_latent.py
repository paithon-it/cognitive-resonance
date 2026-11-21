"""E13, the readable bias estimated on the latent noise (a reading added after a fifth review; not a test).

E13 estimates B from a linear regression of the graded judgment, which lives on a scale with few steps. Here the same
three traces enter a logistic model of the answer alone, with the stimulus, as in the history-dependent psychometric
function of Fründ, Wichmann and Macke (2014): P(answer = 1) = logistic(b0 + b1 s + w . a). B is the variance of the
history term over the variance of the latent noise of the model (pi^2 / 3), with the part due to the error of the
estimate removed, so that with no history it is zero on average. The confidence is used only to build two of the
three traces, as in E13.

    python analysis/people_latent.py [--data data/confidence_db]
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import people_bias as pb  # noqa: E402
from common import ROOT, write_json  # noqa: E402
from people_more import med_ci  # noqa: E402

from cognitive_resonance import readable  # noqa: E402

NOISE = math.pi ** 2 / 3


def logistic_fit(x: np.ndarray, y: np.ndarray, steps: int = 50):
    """Maximum likelihood by Newton steps; returns the coefficients and their covariance, or None if it does not settle."""
    w = np.zeros(x.shape[1])
    for _ in range(steps):
        p = 1.0 / (1.0 + np.exp(-np.clip(x @ w, -30, 30)))
        info = (x * (p * (1 - p))[:, None]).T @ x + 1e-8 * np.eye(x.shape[1])
        step = np.linalg.solve(info, x.T @ (y - p))
        w = w + step
        if not np.all(np.isfinite(w)) or np.abs(w).max() > 50:
            return None
        if np.abs(step).max() < 1e-8:
            return w, np.linalg.inv(info)
    return None


def latent_bias(stimulus: np.ndarray, response: np.ndarray, a: np.ndarray) -> float | None:
    """B on the latent scale: variance of the history term, less what the error of the estimate adds, over pi^2 / 3."""
    ac = a - a.mean(axis=0)
    x = np.column_stack([np.ones(len(response)), 2.0 * stimulus - 1.0, ac])
    fit = logistic_fit(x, response.astype(float))
    if fit is None:
        return None
    w, cov = fit
    sigma = np.cov(ac, rowvar=False)
    return float((w[2:] @ sigma @ w[2:] - np.trace(sigma @ cov[2:, 2:])) / NOISE)


def person(p: dict, rng) -> dict:
    g_all = readable.graded_judgment(p["response"], p["confidence"])
    a, s, r = readable.history(p["response"], g_all), p["stimulus"][1:], p["response"][1:]
    order = rng.permutation(len(p["response"]))
    g_sh = readable.graded_judgment(p["response"][order], p["confidence"][order])
    a_sh = readable.history(p["response"][order], g_sh)
    return {"b_latent": latent_bias(s, r, a), "b_latent_shuffled": latent_bias(p["stimulus"][order][1:], p["response"][order][1:], a_sh)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT / "data" / "confidence_db"))
    args = ap.parse_args()
    e13 = json.loads((ROOT / "results" / "e13_people_bias.json").read_text())
    linear = {(r["file"], str(r["person"])): r["b"] for r in e13["people"]}
    rng = np.random.default_rng(1316)
    out = {"what": "a reading added after a fifth review; no criterion", "noise_variance_of_the_model": NOISE}
    for part in ("confirmation", "exploration"):
        real, shuffled, pairs, failed = {}, {}, [], 0
        for entry in e13["files"]:
            if entry["status"] != "included" or entry["split"] != part:
                continue
            people, _ = pb.read_file(Path(args.data) / entry["file"])
            for k in people:
                if not pb.included(people[k]):
                    continue
                m = person(people[k], rng)
                if m["b_latent"] is None or m["b_latent_shuffled"] is None:
                    failed += 1
                    continue
                real.setdefault(entry["file"], []).append(m["b_latent"])
                shuffled.setdefault(entry["file"], []).append(m["b_latent_shuffled"])
                if (entry["file"], str(k)) in linear:
                    pairs.append((linear[(entry["file"], str(k))], m["b_latent"]))
        from scipy.stats import spearmanr
        out[part] = {"b_latent": med_ci(real, rng), "b_latent_shuffled": med_ci(shuffled, rng), "not_fitted": failed,
                     "spearman_with_the_linear_estimate": float(spearmanr(*zip(*pairs)).statistic) if len(pairs) > 10 else None,
                     "people_matched_to_e13": len(pairs)}
    write_json(ROOT / "results" / "e13_people_latent.json", out)
    for part in ("confirmation", "exploration"):
        c = out[part]
        print(part, "B latent", round(c["b_latent"]["median"], 4), (round(c["b_latent"]["lo"], 4), round(c["b_latent"]["hi"], 4)),
              "people", c["b_latent"]["people"], "| shuffled", round(c["b_latent_shuffled"]["median"], 4), "| not fitted", c["not_fitted"],
              "| rank correlation with the linear B", c["spearman_with_the_linear_estimate"], c["people_matched_to_e13"])


if __name__ == "__main__":
    main()
