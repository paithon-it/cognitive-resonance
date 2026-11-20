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


# --- the turing of the pair, in three factors -------------------------------------------------------------------------

def pair_factors(d_h: float, d_m: float, r: float, b: float) -> dict:
    """The turing of a pair, measured against the judge alone with its bias in (d_H = d_h / sqrt(1 + B)), as a product.

    With q = d_m / d_h (how many times the machine is worth the judge without its bias) and
    p = (1 - r q)^2 / (1 - r^2) (the private part, in units of d_h^2):

        with the partner         Tu^2 = (1 + B) (q^2 + p)
        with the soloist         Tu^2 = (1 + B) (q^2 + p (1 - r^2) / (1 - r^2 + B))
        with the blind machine   Tu^2 = 1 + B

    1 + B is the factor of resonance: what cancelling the readable bias is worth. q^2 is what the machine knows of
    the task, p what the judge knows and the machine does not. It is the theorem of the three machines divided by
    d_H^2: the same algebra, rearranged. q^2 + p ("together") is what the two know together, in units of the judge
    without its bias: it is 1 when the machine sees a part of what the judge sees (r = q), q^2 when the judge sees a
    part of what the machine sees (r = 1 / q), 1 + q^2 when their errors are independent. Also returned: what the
    judge adds to the machine alone with the partner (sqrt(1 + p / q^2) - 1), and the angles between the errors of
    the judge and of a blind machine or of a partner built for weight 1 (noise correlations -sqrt(B / (1 + B)) and
    (r - B) / (1 + B))."""
    q = d_m / d_h
    p = (1.0 - r * q) ** 2 / (1.0 - r * r)
    seen = p * (1.0 - r * r) / (1.0 - r * r + b)
    out = {"resonance": 1.0 + b, "strength": q * q, "private": p, "together": q * q + p, "private_with_soloist": seen,
           "tu_partner": math.sqrt((1.0 + b) * (q * q + p)), "tu_soloist": math.sqrt((1.0 + b) * (q * q + seen)),
           "tu_blind": math.sqrt(1.0 + b),
           "angle_blind": math.degrees(math.acos(-math.sqrt(b / (1.0 + b)))),
           "angle_partner": math.degrees(math.acos(max(-1.0, min(1.0, (r - b) / (1.0 + b)))))}
    out["judge_adds_to_machine"] = math.sqrt(1.0 + p / (q * q)) - 1.0 if q > 0 else None
    return out


# --- how well the machine has to know the judge ---------------------------------------------------------------------

def bias_left_after(n: float) -> float:
    """Expected bias left, in units of the noise of h, when the slope of the judge on the trace is estimated from n
    judgments with known answers: the variance of a regression slope, 1 / n, whatever the size of the bias."""
    return 1.0 / n


def share_kept_after(n: float, r: float, b: float) -> float:
    """Share of the gain of the theorem that a correction learned from n judgments keeps."""
    left = bias_left_after(n)
    return 1.0 - (left / (1 - r ** 2 + left)) / (b / (1 - r ** 2 + b))


def yes_no_factor(mean_0: float, mean_1: float, beta: float, var_a: float, points: int = 20001) -> float:
    """How many yes-or-no answers are worth one graded judgment when the slope of the judge on the trace is estimated.

    The answer is yes when h + beta a > 0, with h of unit noise and class means `mean_0` and `mean_1`, and the trace
    a normal within class with variance `var_a` (classes equally frequent). A probit regression on the class (two
    intercepts) and on a has, per answer, the Fisher information on the slope
    E[lambda a^2] - sum_y p_y E_y[lambda a]^2 / E_y[lambda], lambda = phi(eta)^2 / (Phi(eta) (1 - Phi(eta))),
    eta = mean_y + beta a. A graded cue has `var_a`. The factor is their ratio, always above pi / 2.
    """
    sd = math.sqrt(var_a)
    grid = [sd * (-8.0 + 16.0 * i / (points - 1)) for i in range(points)]
    dens = [math.exp(-0.5 * (x / sd) ** 2) for x in grid]
    total = sum(dens)
    info = 0.0
    for mean in (mean_0, mean_1):
        e_l = e_la = e_laa = 0.0
        for x, w in zip(grid, dens):
            eta = mean + beta * x
            p = 0.5 * math.erfc(-eta / math.sqrt(2.0))
            q = 0.5 * math.erfc(eta / math.sqrt(2.0))
            if p <= 0.0 or q <= 0.0:
                continue
            lam = math.exp(-eta * eta) / (2.0 * math.pi) / (p * q) * (w / total)
            e_l, e_la, e_laa = e_l + lam, e_la + lam * x, e_laa + lam * x * x
        info += 0.5 * (e_laa - e_la ** 2 / e_l)
    return var_a / info


def bias_from_shift(shift: float, base_rate: float, trace_rate: float | None = None) -> float:
    """The size B of a bias, in units of the judge's noise, from a shift measured in probabilities.

    A yes-or-no trace (the previous decision was a yes, with probability `trace_rate`, default `base_rate`) moves
    the probability of a yes by `shift` around `base_rate`. In a probit model that is a push of
    shift / phi(Phi^-1(base_rate)) between the two values of the trace, and its variance is that push squared times
    trace_rate (1 - trace_rate). The noise is all that makes the answer vary at a given trace, differences between
    cases included: for the judge's own noise alone B is larger, by as much as the cases differ."""
    q = base_rate if trace_rate is None else trace_rate
    z = _probit(base_rate)
    push = shift / (math.exp(-0.5 * z * z) / math.sqrt(2.0 * math.pi))
    return push ** 2 * q * (1.0 - q)


def _probit(p: float) -> float:
    """Phi^-1 by bisection (no scipy in the library)."""
    lo, hi = -8.0, 8.0
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if 0.5 * math.erfc(-mid / math.sqrt(2.0)) < p:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


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
