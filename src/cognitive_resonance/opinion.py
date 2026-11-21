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
        if (y == c).any():
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
    """Everything E16 measures with a corrector learned on `learn` and read on `test` (index arrays).

    Right answers are counted twice: at threshold zero, as the judge would answer, and with the threshold that gives
    the most right answers on `learn` (added after the exploration: the judge prefers one letter, and the sentence
    with the opinion moves that preference).
    """
    y, g0, g_a, g_b, a = (np.asarray(v) for v in (y, g0, g_a, g_b, a))
    g = met(g_a, g_b, a)
    beta, _ = readable_bias(g[learn], a[learn], y[learn])
    corrected = g - beta * a
    d_clean, d_as, d_corr = (float(d_a(v[test], y[test])) for v in (g0, g, corrected))
    lost = d_clean - d_as
    agree = (a > 0) == (y == 1)
    t = test
    cut = {k: best_threshold(v[learn], y[learn]) for k, v in (("clean", g0), ("as_it_is", g), ("corrected", corrected))}
    return {"beta_learned": beta, "d_clean": d_clean, "d_as_it_is": d_as, "d_corrected": d_corr,
            "kept_as_it_is": d_as / d_clean, "kept_corrected": d_corr / d_clean,
            "gives_back": (d_corr - d_as) / lost if lost > 1e-3 else None,
            "gain_corrected": d_corr / d_as,
            "right_clean": right(g0[t], y[t]), "right_as_it_is": right(g[t], y[t]),
            "right_corrected": right(corrected[t], y[t]),
            "right_clean_with_threshold": right(g0[t], y[t], cut["clean"]),
            "right_as_it_is_with_threshold": right(g[t], y[t], cut["as_it_is"]),
            "right_corrected_with_threshold": right(corrected[t], y[t], cut["corrected"]),
            "threshold_as_it_is": cut["as_it_is"], "threshold_corrected": cut["corrected"],
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


# --- readings added after a fourth review (21 November 2026): a corrector that knows more, and the shape of the push ----

def _monotone(v: np.ndarray, w: np.ndarray) -> np.ndarray:
    """The closest non-decreasing sequence, in weighted least squares (pool adjacent violators)."""
    blocks = [[float(x), float(k), 1] for x, k in zip(v, w)]
    i = 0
    while i < len(blocks) - 1:
        if blocks[i][0] > blocks[i + 1][0]:
            a, b = blocks[i], blocks.pop(i + 1)
            blocks[i] = [(a[0] * a[1] + b[0] * b[1]) / (a[1] + b[1]), a[1] + b[1], a[2] + b[2]]
            i = max(i - 1, 0)
        else:
            i += 1
    return np.concatenate([[x] * n for x, _, n in blocks])


def state_corrector(g: np.ndarray, a: np.ndarray, y: np.ndarray, learn: np.ndarray, bins: int = 15) -> np.ndarray:
    """A corrector that knows the trace and the judgment it corrects, and nothing of the question.

    For each of the two opinions, a map from the judgment to the log-odds of the right answer, learned on `learn`:
    bins of equal count, half a case added to every cell, made non-decreasing, read by linear interpolation. Where
    the linear corrector moves the two groups by the same amount everywhere, this one moves them by an amount that
    depends on how far the judgment is from the doubt.
    """
    g, a, y = np.asarray(g, float), np.asarray(a), np.asarray(y)
    out = np.empty_like(g)
    for s in (1, -1):
        gl, yl = g[learn][a[learn] == s], y[learn][a[learn] == s]
        order = np.argsort(gl)
        parts = np.array_split(order, bins)
        centre = np.array([gl[k].mean() for k in parts])
        value = np.array([np.log(((yl[k] == 1).sum() + 0.5) / ((yl[k] == 0).sum() + 0.5)) for k in parts])
        value = _monotone(value, np.array([len(k) for k in parts], float))
        out[a == s] = np.interp(g[a == s], centre, value)
    return out


def shape_of_the_push(g0: np.ndarray, g_a: np.ndarray, g_b: np.ndarray, y: np.ndarray, centre: float, parts: int = 5) -> dict:
    """The push by how far the judgment with no opinion is from the doubt, and the width of the bell that fits it.

    The model is that of the simulation: push = beta k(h), k(h) = exp(-h^2 / (2 tau^2)), with h the distance of the
    judgment with no opinion from the point of doubt (`centre`), in standard deviations of its noise at equal answer.
    tau is chosen on a grid by least squares; `additive` is the same fit with a constant push.
    """
    g0, y = np.asarray(g0, float), np.asarray(y)
    push_ = 0.5 * (np.asarray(g_a, float) - np.asarray(g_b, float))
    sd = float(np.sqrt(_within(g0, y).var()))
    h = (g0 - centre) / sd
    far = np.abs(h)
    edges = np.quantile(far, np.linspace(0, 1, parts + 1))
    which = np.clip(np.searchsorted(edges, far, side="right") - 1, 0, parts - 1)
    profile = [{"distance": float(far[which == k].mean()), "push": float(push_[which == k].mean() / sd)} for k in range(parts)]
    best = None
    for tau in np.round(np.exp(np.linspace(np.log(0.2), np.log(20.0), 80)), 4):
        k = np.exp(-h * h / (2 * tau * tau))
        beta = float(push_ @ k / (k @ k))
        sse = float(((push_ - beta * k) ** 2).sum())
        if best is None or sse < best[0]:
            best = (sse, float(tau), beta / sd)
    flat = float(((push_ - push_.mean()) ** 2).sum())
    return {"noise_sd": sd, "profile": profile, "tau": best[1], "beta_at_the_doubt": best[2],
            "rms_push": float(np.sqrt((push_ ** 2).mean()) / sd),
            "variance_explained_by_the_bell": 1.0 - best[0] / flat if flat > 0 else None}
