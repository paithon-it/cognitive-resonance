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
