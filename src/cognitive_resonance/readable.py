"""The readable bias of one person, from their own trials (protocol § 19, E13).

A person gives a response and a confidence on every trial. The graded judgment is the normal score of the response
with its confidence; the trace is the history of the previous trials, which says nothing about the stimulus about to
arrive. B is the variance of what the trace predicts of the judgment, at equal class, over the variance of the rest.
The corrector sees no stimulus: it removes from the judgment what the trace predicts.
"""
from __future__ import annotations

import math

import numpy as np
from scipy.stats import norm, rankdata

from .estimators import d_a

TRACES = 3          # sign of the previous response, previous graded judgment, mean of the twenty before
WINDOW = 20
MIN_D = 0.2         # below this sensitivity in the second half the gain is not computed


def graded_judgment(response: np.ndarray, confidence: np.ndarray) -> np.ndarray:
    """Normal scores of the trials ordered from the surest answer for class 0 to the surest for class 1.

    `response` is 0 or 1; `confidence` any numbers, higher is surer. Ties keep the mean rank.
    """
    response, confidence = np.asarray(response), np.asarray(confidence, float)
    sign = np.where(response == 1, 1.0, -1.0)
    level = rankdata(confidence, method="dense")            # 1, 2, ... within the person
    rank = rankdata(sign * level, method="average")
    return norm.ppf((rank - 0.5) / rank.size)


def history(response: np.ndarray, g: np.ndarray) -> np.ndarray:
    """The three traces of every trial from the second on: rows are trials 1 .. n-1, known before the stimulus."""
    sign = np.where(np.asarray(response) == 1, 1.0, -1.0)
    csum = np.concatenate([[0.0], np.cumsum(g)])
    t = np.arange(1, g.size)
    start = np.maximum(0, t - WINDOW)
    mean = (csum[t] - csum[start]) / (t - start)
    return np.column_stack([sign[:-1], g[:-1], mean])


def _centred(v: np.ndarray, y: np.ndarray) -> np.ndarray:
    out = np.array(v, float, copy=True)
    for c in (0, 1):
        out[y == c] -= out[y == c].mean(axis=0)
    return out


def moments(g: np.ndarray, a: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray, float, int]:
    """What the least squares of g on the traces need, at equal class: A'A, A'g, g'g and the number of trials."""
    gc, ac = _centred(g, y), _centred(a, y)
    return ac.T @ ac, ac.T @ gc, float(gc @ gc), int(g.size)


def slopes(xtx: np.ndarray, xty: np.ndarray) -> np.ndarray:
    return np.linalg.lstsq(xtx, xty, rcond=None)[0]


def readable_bias(g: np.ndarray, a: np.ndarray, y: np.ndarray) -> float:
    """B = R2 / (1 - R2), with the partial R2 of the traces adjusted for the degrees of freedom. Can be negative."""
    xtx, xty, yty, n = moments(g, a, y)
    beta = slopes(xtx, xty)
    r2 = float(beta @ xty) / yty
    adjusted = 1.0 - (1.0 - r2) * (n - 2) / (n - 2 - a.shape[1])
    return adjusted / (1.0 - adjusted)


def gain(g: np.ndarray, a: np.ndarray, y: np.ndarray, beta: np.ndarray) -> float | None:
    """Sensitivity of the corrected judgment over that of the judgment as it is, both from the AUC."""
    before = float(d_a(g, y))
    if not before >= MIN_D:
        return None
    return float(d_a(g - a @ beta, y)) / before


def _d_moments(score: np.ndarray, y: np.ndarray) -> float:
    one, zero = score[y == 1], score[y == 0]
    pooled = ((one.size - 1) * one.var(ddof=1) + (zero.size - 1) * zero.var(ddof=1)) / (score.size - 2)
    return float((one.mean() - zero.mean()) / math.sqrt(pooled))


def gain_moments(g: np.ndarray, a: np.ndarray, y: np.ndarray, beta: np.ndarray) -> float | None:
    """The same gain with the sensitivity as difference of the class means over the within-class deviation.

    It does not see the order of the trials inside one step of the scale: with a coarse scale it separates the
    variance removed from what the trace adds by reordering tied trials.
    """
    before = _d_moments(g, y)
    if not before >= MIN_D:
        return None
    return _d_moments(g - a @ beta, y) / before


def person(stimulus: np.ndarray, response: np.ndarray, confidence: np.ndarray) -> dict:
    """Everything E13 measures on one person. `stimulus` and `response` are 0 or 1, in the order of the trials."""
    stimulus = np.asarray(stimulus).astype(int)
    g_all = graded_judgment(response, confidence)
    a, g, y = history(response, g_all), g_all[1:], stimulus[1:]
    half = g.size // 2
    first, second = slice(0, half), slice(half, None)
    m1 = moments(g[first], a[first], y[first])
    beta = slopes(m1[0], m1[1])
    return {"n": int(g.size), "n_first": int(half), "b": readable_bias(g, a, y),
            "b_first": readable_bias(g[first], a[first], y[first]),
            "gain_personal": gain(g[second], a[second], y[second], beta),
            "gain_moments": gain_moments(g[second], a[second], y[second], beta),
            "d_second": float(d_a(g[second], y[second])),
            "_moments_first": (m1[0], m1[1]), "_second": (g[second], a[second], y[second])}


def predicted_gain_sq(b_first: float, n_first: int, traces: int = TRACES) -> float:
    """(1 + B) / (1 + p / n): the gain of removing a bias B learned from n judgments with p traces."""
    return (1.0 + b_first) / (1.0 + traces / n_first)


def corrector_angle(b: float) -> float | None:
    """Degrees between the error of the person and that of the corrector that removes a bias B."""
    if not b > 0:
        return None
    return math.degrees(math.acos(-math.sqrt(b / (1.0 + b))))
