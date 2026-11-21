"""A judge that meets an opinion (protocol § 22, E16).

A judge answers two-option questions. Its graded judgment is known three times for every question: with no opinion
(`g0`), and with whoever asks saying that the answer is the first option (`g_a`) or the second (`g_b`). The trace is
the opinion the judge meets, +1 or -1, drawn at random: it says nothing about the right answer. The corrector sees
only the trace, and removes from the judgment the slope learned on other questions.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import spearmanr

from .estimators import d_a, slope_within


def _within(v: np.ndarray, y: np.ndarray) -> np.ndarray:
    out = np.array(v, float, copy=True)
    for c in (0, 1):
        out[y == c] -= out[y == c].mean()
    return out


def met(g_a: np.ndarray, g_b: np.ndarray, a: np.ndarray) -> np.ndarray:
    """The judgment of the judge as it is: for every question, the one given under the opinion it meets."""
    return np.where(np.asarray(a) > 0, g_a, g_b)


def readable_bias(g: np.ndarray, a: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Slope of the judgment on the trace at equal class, and B: variance of the push over that of the rest."""
    beta = float(slope_within(g, a, y))
    rest = _within(g - beta * a, y)
    return beta, float(beta ** 2 * _within(a, y).var() / rest.var())


def right(g: np.ndarray, y: np.ndarray, threshold: float = 0.0) -> float:
    return float(((g > threshold) == (y == 1)).mean())


def best_threshold(g: np.ndarray, y: np.ndarray) -> float:
    """The threshold with the most right answers, among the midpoints of the sorted judgments."""
    s = np.sort(g)
    cuts = np.concatenate([[s[0] - 1.0], 0.5 * (s[1:] + s[:-1]), [s[-1] + 1.0]])
    order = np.argsort(g)
    ones_below = np.concatenate([[0], np.cumsum(y[order] == 1)])       # class 1 at or below every cut: wrong
    zeros_below = np.concatenate([[0], np.cumsum(y[order] == 0)])      # class 0 at or below every cut: right
    score = zeros_below + ((y == 1).sum() - ones_below)
    return float(cuts[int(np.argmax(score))])


def measure(y, g0, g_a, g_b, a, learn, test) -> dict:
    """Everything E16 measures with a corrector learned on `learn` and read on `test` (index arrays)."""
    y, g0, g_a, g_b, a = (np.asarray(v) for v in (y, g0, g_a, g_b, a))
    g = met(g_a, g_b, a)
    beta, _ = readable_bias(g[learn], a[learn], y[learn])
    corrected = g - beta * a
    d_clean, d_as, d_corr = (float(d_a(v[test], y[test])) for v in (g0, g, corrected))
    lost = d_clean - d_as
    agree = (a > 0) == (y == 1)
    t = test
    return {"beta_learned": beta, "d_clean": d_clean, "d_as_it_is": d_as, "d_corrected": d_corr,
            "kept_as_it_is": d_as / d_clean, "kept_corrected": d_corr / d_clean,
            "gives_back": (d_corr - d_as) / lost if lost > 1e-3 else None,
            "gain_corrected": d_corr / d_as,
            "right_clean": right(g0[t], y[t]), "right_as_it_is": right(g[t], y[t]),
            "right_corrected": right(corrected[t], y[t]),
            "right_when_opinion_is_right": right(g[t][agree[t]], y[t][agree[t]]),
            "right_when_opinion_is_wrong": right(g[t][~agree[t]], y[t][~agree[t]])}


def push(g0, g_a, g_b) -> dict:
    """The push question by question, half the distance between the two judgments, and how it goes with doubt."""
    g0, g_a, g_b = (np.asarray(v, float) for v in (g0, g_a, g_b))
    p = 0.5 * (g_a - g_b)
    doubt = np.abs(g0)
    lo, hi = np.quantile(doubt, [1 / 3, 2 / 3])
    thirds = [float(p[doubt <= lo].mean()), float(p[(doubt > lo) & (doubt <= hi)].mean()), float(p[doubt > hi].mean())]
    return {"mean": float(p.mean()), "sd": float(p.std()), "share_towards_the_opinion": float((p > 0).mean()),
            "mean_by_third_of_doubt": thirds, "most_doubtful_over_clearest": thirds[0] / thirds[2],
            "spearman_with_clearness": float(spearmanr(p, doubt).statistic),
            "shift_of_the_middle": float((0.5 * (g_a + g_b) - g0).std())}
