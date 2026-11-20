"""The theorem of the chart (protocol § 14): a judge with a bias, and three partners.

The judge's cue is `h + beta a`: a clean cue `h` with sensitivity `d_h` and unit noise, plus a
bias that does not depend on the class, with variance `B` in units of the noise of `h`. The
machine's cue `m` has sensitivity `d_m` and unit noise, correlated `r` with the noise of `h`.
All the values below are those of the best linear combination, with weights of any sign.

- `with_soloist`: the judge with the machine `m` (the bias stays in the pair);
- `with_partner`: the judge with `m` and something that reads `a` (the bias can be removed);
- the judge with a blind corrector, which reads only `a`, is worth `d_h`.
"""
from __future__ import annotations

import math


def with_soloist(d_h: float, d_m: float, r: float, b: float) -> float:
    """D1: sensitivity of the best pair (h + beta a, m)."""
    return math.sqrt((d_h ** 2 - 2 * r * d_h * d_m + (1 + b) * d_m ** 2) / (1 + b - r ** 2))


def with_partner(d_h: float, d_m: float, r: float) -> float:
    """D2: sensitivity of the best combination of (h + beta a, m, a), that is of (h, m)."""
    return math.sqrt((d_h ** 2 - 2 * r * d_h * d_m + d_m ** 2) / (1 - r ** 2))


def private(d_h: float, d_m: float, r: float) -> float:
    """P = D2^2 - d_m^2: what the clean judge knows and the machine does not. Zero at the useful angle, r = d_h / d_m."""
    return (d_h - r * d_m) ** 2 / (1 - r ** 2)


def gain(d_h: float, d_m: float, r: float, b: float) -> float:
    """D2^2 - D1^2 = B / (1 - r^2 + B) * P: the share of P that the bias was covering."""
    return b / (1 - r ** 2 + b) * private(d_h, d_m, r)


def blind_band(r: float, b: float) -> tuple[float, float]:
    """The blind corrector (pair worth d_h) beats the soloist as a partner when d_m / d_h lies between these two
    values. The upper one is the threshold that matters: a machine stronger than that is the better partner. The
    lower one is above zero only when r^2 > B: there a very weak machine, used with a negative weight, serves as a
    reference for the noise it shares with the judge, and the pair with it is again worth more than d_h."""
    root = math.sqrt(b * (1 + b - r ** 2))
    return max(0.0, (r - root) / (1 + b)), (r + root) / (1 + b)


def blind_threshold(r: float, b: float) -> float:
    """The upper value of `blind_band`."""
    return blind_band(r, b)[1]


# --- the trace says something true (added with the third review, 20 November 2026) -------------------------------

def deserved_weight(d_h: float, d_m: float, r: float, d_a: float) -> float:
    """The weight the trace `a` deserves in the judge's cue, next to the machine: with it there is nothing to correct.
    Zero when the trace does not depend on the class; d_a / d_h when the machine's noise is independent (r = 0)."""
    return d_a * (1 - r ** 2) / (d_h - r * d_m)


def with_soloist_informative(d_h: float, d_m: float, r: float, beta: float, d_a: float) -> float:
    """D1 when the trace has sensitivity d_a (unit noise, independent of h and m): best pair (h + beta a, m)."""
    s = d_h + beta * d_a
    return math.sqrt((s ** 2 - 2 * r * s * d_m + (1 + beta ** 2) * d_m ** 2) / (1 + beta ** 2 - r ** 2))


def with_partner_informative(d_h: float, d_m: float, r: float, d_a: float) -> float:
    """D2 when the trace has sensitivity d_a: the best use of (h, m, a)."""
    return math.sqrt(with_partner(d_h, d_m, r) ** 2 + d_a ** 2)


def gain_informative(d_h: float, d_m: float, r: float, beta: float, d_a: float) -> float:
    """D2^2 - D1^2 when the trace is informative: B / (1 - r^2 + B) * P * (1 - beta_0 / beta)^2, with B = beta^2 and
    beta_0 the deserved weight. With d_a = 0 it is `gain`; with beta = beta_0 it is zero: only the excess is a bias."""
    return (beta * (d_h - r * d_m) - d_a * (1 - r ** 2)) ** 2 / ((1 - r ** 2) * (1 + beta ** 2 - r ** 2))


# --- how well the machine has to know the judge ---------------------------------------------------------------------

def bias_left_after(n: float) -> float:
    """Expected bias left, in units of the noise of h, when the slope of the judge on the trace is estimated from n
    judgments with known answers: the variance of a regression slope, 1 / n, whatever the size of the bias."""
    return 1.0 / n


def share_kept_after(n: float, r: float, b: float) -> float:
    """Share of the gain of the theorem that a correction learned from n judgments keeps."""
    left = bias_left_after(n)
    return 1.0 - (left / (1 - r ** 2 + left)) / (b / (1 - r ** 2 + b))


# --- the theorem read on a pair measured with its bias still in (people) ---------------------------------------------

def private_observed(d_big_h: float, d_m: float, rho: float) -> float:
    """What the judge adds to the machine today, bias included: (d_H - rho d_m)^2 / (1 - rho^2). It is the true
    private part P shrunk by (1 - r^2) / (1 - r^2 + B)."""
    return (d_big_h - rho * d_m) ** 2 / (1 - rho ** 2)


def gain_if_bias(d_big_h: float, d_m: float, rho: float, b: float) -> float | None:
    """D2^2 - D1^2 for a pair measured with d_H, d_m and rho, if a share B / (1 + B) of the judge's noise were a
    bias a machine can read: P_obs * B / (1 - rho^2 (1 + B)). None when rho^2 (1 + B) >= 1 (no such judge)."""
    den = 1 - rho ** 2 * (1 + b)
    return private_observed(d_big_h, d_m, rho) * b / den if den > 0 else None
