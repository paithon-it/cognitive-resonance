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


# --- the turing of the pair: one factor and two addends -------------------------------------------------------------------------

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
    part of what the machine sees (r = 1 / q), 1 + q^2 when their errors are independent. Also returned: the pair
    seen from the two sides once the bias is gone, the turing sqrt(q^2 + p) and the yield on the machine
    sqrt(1 + p / q^2) (what the judge adds to the machine is that, minus one). With independent errors the two are
    tied: (Tu^2 - 1)(yield^2 - 1) = 1, and in general the product is ((1 + r^2 - r (q + 1 / q)) / (1 - r^2))^2
    ("exchange"). And the angles between the errors of the judge and of a blind machine or of a partner built for
    weight 1 (noise correlations -sqrt(B / (1 + B)) and (r - B) / (1 + B))."""
    q = d_m / d_h
    p = (1.0 - r * q) ** 2 / (1.0 - r * r)
    seen = p * (1.0 - r * r) / (1.0 - r * r + b)
    out = {"resonance": 1.0 + b, "strength": q * q, "private": p, "together": q * q + p, "private_with_soloist": seen,
           "tu_partner": math.sqrt((1.0 + b) * (q * q + p)), "tu_soloist": math.sqrt((1.0 + b) * (q * q + seen)),
           "tu_blind": math.sqrt(1.0 + b),
           "angle_blind": math.degrees(math.acos(-math.sqrt(b / (1.0 + b)))),
           "angle_partner": math.degrees(math.acos(max(-1.0, min(1.0, (r - b) / (1.0 + b)))))}
    out["tu_without_bias"] = math.sqrt(q * q + p)
    out["yield_on_machine"] = math.sqrt(1.0 + p / (q * q)) if q > 0 else None
    out["judge_adds_to_machine"] = out["yield_on_machine"] - 1.0 if q > 0 else None
    out["exchange"] = (q * q + p - 1.0) * (p / (q * q)) if q > 0 else None
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


def _phi(x: float) -> float:
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


def _Phi(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def pair_information(d_h: float, d_m: float, r: float, b: float) -> dict:
    """The four pairs of the proposition in d'^2, where the parts add up.

    The person alone is worth d_h^2 / (1 + B); with a corrector that reads only the bias, d_h^2; with a machine that
    knows the task, d_m^2 plus the private part P shrunk by (1 - r^2) / (1 - r^2 + B); with a machine that also reads
    the bias, d_m^2 + P. The bias never adds anything: it only covers a share of P.
    """
    private = (d_h - r * d_m) ** 2 / (1.0 - r * r)
    reaching = private * (1.0 - r * r) / (1.0 - r * r + b)
    return {"person_alone": d_h ** 2 / (1.0 + b), "person_with_corrector": d_h ** 2, "machine_alone": d_m ** 2,
            "private": private, "private_reaching_the_pair": reaching, "covered_by_the_bias": private - reaching,
            "person_with_generic": d_m ** 2 + reaching, "person_with_tailored": d_m ** 2 + private}


def opposite_errors_as_a_common_cause(r: float) -> tuple[float, float, float]:
    """Two unit errors with correlation r < 0 as one hidden cause Z entering with opposite signs.

    e_h = c Z + s n_h and e_m = -c Z + s n_m, with c = sqrt(|r|) and s = sqrt(1 - |r|): returns (c, -c, s). Errors in
    antiphase are a shared disturbance that nobody has labelled, the corrector's case with the trace unknown.
    """
    if not -1.0 < r < 0.0:
        raise ValueError("r must be negative")
    c = math.sqrt(-r)
    return c, -c, math.sqrt(1.0 + r)


def left_under_a_ceiling(ceiling: float) -> float:
    """Share of a Gaussian bias that stays when no correction larger than `ceiling` gets through.

    The push is beta a with a ~ N(0, 1); `ceiling` is in units of beta. What stays is (|a| - ceiling)+, and its mean
    square is 2 [(1 + s^2)(1 - Phi(s)) - s phi(s)].
    """
    s = max(0.0, ceiling)
    return 2.0 * ((1.0 + s * s) * (1.0 - _Phi(s)) - s * _phi(s))


def left_by_an_open_warning(weight: float) -> float:
    """Share of the bias that stays when the correction is shown apart and the person applies a share of it."""
    return (1.0 - weight) ** 2


# --- added after a fourth review (21 November 2026): a bias the two share, and errors that do not cost the same ----------

def with_shared_bias(d_h: float, d_m: float, r: float, beta: float, gamma: float) -> float:
    """Squared sensitivity of the pair when the same trace pushes both: cues h + beta a and m + gamma a, best weights.

    gamma = 0 is the pair with the generic machine of the proposition. With the same sign the trace adds to the
    correlation of the two errors, (r + beta gamma) / sqrt((1 + beta^2)(1 + gamma^2)), and the pair cannot cancel it;
    with opposite signs the pair cancels part of it by itself. A machine that reads the trace removes it from both:
    the pair is again worth d_m^2 + P.
    """
    num = d_h ** 2 * (1 + gamma ** 2) - 2 * d_h * d_m * (r + beta * gamma) + d_m ** 2 * (1 + beta ** 2)
    return num / ((1 + beta ** 2) * (1 + gamma ** 2) - (r + beta * gamma) ** 2)


def angle_with_shared_bias(r: float, beta: float, gamma: float) -> float:
    """Degrees between the two errors when the same trace pushes both."""
    return math.degrees(math.acos((r + beta * gamma) / math.sqrt((1 + beta ** 2) * (1 + gamma ** 2))))


def criterion_for_costs(cost_miss: float, cost_false_alarm: float, prior: float = 0.5) -> float:
    """The threshold on the log-odds that minimises the expected cost: say "yes" above it. Zero when the two errors
    cost the same and the two answers are equally likely."""
    return math.log(cost_false_alarm * (1 - prior) / (cost_miss * prior))


def decisions_at(d: float, criterion: float, cost_miss: float = 1.0, cost_false_alarm: float = 1.0, prior: float = 0.5) -> dict:
    """Right answers and expected cost of a Gaussian judge of sensitivity d that says "yes" above `criterion` (log-odds)."""
    from statistics import NormalDist
    phi = NormalDist().cdf
    miss, false_alarm = phi(criterion / d - d / 2), 1.0 - phi(criterion / d + d / 2)
    return {"miss": miss, "false_alarm": false_alarm, "right": 1.0 - prior * miss - (1 - prior) * false_alarm,
            "cost": prior * cost_miss * miss + (1 - prior) * cost_false_alarm * false_alarm}


# --- added after a fifth review (21 November 2026): what a bias is worth in right answers, and the other way to a pair ----

def right_answers_gained(right_as_it_is: float, b: float, traces: int = 0, judgments: float | None = None) -> float:
    """Right answers gained (as a share) by removing a readable bias B from a Gaussian judge that is right
    `right_as_it_is` of the times, at its best threshold. With `traces` and `judgments`, the correction is learned and
    leaves a bias of traces / judgments: the sensitivity grows by sqrt((1 + B) / (1 + p / n))."""
    from statistics import NormalDist
    nd = NormalDist()
    d = 2.0 * nd.inv_cdf(right_as_it_is)
    learned = traces / judgments if judgments else 0.0
    return nd.cdf(0.5 * d * math.sqrt((1.0 + b) / (1.0 + learned))) - right_as_it_is


def from_repetition(extra_repeats: float, right: float) -> float:
    """The readable bias implied by a tendency to repeat the last answer: `extra_repeats` is how much the probability
    of repeating exceeds one half, `right` the share of right answers. A push c towards the last answer raises the
    probability of repeating by c phi(d / 2); B is c^2 in units of the noise."""
    from statistics import NormalDist
    nd = NormalDist()
    return (extra_repeats / nd.pdf(nd.inv_cdf(right))) ** 2


def specialisation(strong: float, weak: float) -> dict:
    """Two judges, each right `strong` of the times on its own half of the cases and `weak` on the other half.

    Alone both are right (strong + weak) / 2; a pair that knows whom to listen to on each case is right `strong`. The
    errors are negatively correlated although nobody's bias has been cancelled: the other way to a pair that beats both.
    """
    alone = 0.5 * (strong + weak)
    both_wrong = (1 - strong) * (1 - weak)
    corr = (both_wrong - (1 - alone) ** 2) / (alone * (1 - alone))
    return {"alone": alone, "with_deferral": strong, "error_correlation": corr, "angle": math.degrees(math.acos(corr))}
