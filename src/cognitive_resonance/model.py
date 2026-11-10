"""The model of the protocol, section 2: two judges with Gaussian cues, summed with a weight.

Pure functions. The person has cue x_H = s dH/2 + n_H, the machine x_M = s dM/2 + n_M,
with s = +1 or -1 by class, noises with unit variance and correlation rho = cos(phi).
The pair reads x_H + w x_M, w >= 0. Everything is in the person's unit: k = dM / dH.
Angles are in degrees, as in turing-meter.
"""
from __future__ import annotations

import math

import numpy as np
from scipy.optimize import brentq

# The 49 weights of the protocol (§ 3), evenly spaced in ln w between 1/64 and 64,
# and the 13 of the bootstrap: every fourth one, so that the sparse grid is a subset.
GRID = np.exp(np.linspace(math.log(1 / 64), math.log(64), 49))
COARSE = GRID[::4]


def angle_deg(rho):
    """phi = arccos(rho), in degrees."""
    return np.degrees(np.arccos(np.clip(rho, -1.0, 1.0)))


def rho_of(phi_deg):
    return np.cos(np.radians(phi_deg))


def _check(k, rho):
    if np.any(np.asarray(k) < 0):
        raise ValueError("k must be non-negative")
    if np.any(np.abs(np.asarray(rho)) >= 1.0):
        raise ValueError("rho must be strictly between -1 and 1")


def tu(w, k, rho):
    """Tu(w) = (1 + w k) / sqrt(1 + w^2 + 2 rho w): the pair at weight w, in the person's unit."""
    _check(k, rho)
    w = np.asarray(w, float)
    return (1.0 + w * k) / np.sqrt(1.0 + w * w + 2.0 * rho * w)


def tu2_minus_1(w, k, rho):
    """Tu(w)^2 - 1, the quantity whose half maximum defines the band."""
    w = np.asarray(w, float)
    return (w * (2.0 * k - 2.0 * rho) + w * w * (k * k - 1.0)) / (1.0 + w * w + 2.0 * rho * w)


def has_interior_peak(k, rho) -> bool:
    """True when both judges get a positive weight at the optimum: rho < min(k, 1/k)."""
    return bool(k > 0 and rho < min(k, 1.0 / k))


def w_star(k, rho) -> float:
    """The weight of the maximum, (k - rho) / (1 - rho k), restricted to w >= 0.

    Below the useful angle the optimum sits at an end: 0 (the person alone) when the
    machine is the weaker partner, infinity (the machine alone) when it is the stronger.
    """
    _check(k, rho)
    if has_interior_peak(k, rho):
        return (k - rho) / (1.0 - rho * k)
    return 0.0 if k <= 1.0 else math.inf


def tu_app(k, rho) -> float:
    """The apparent turings: the maximum of Tu(w) over w >= 0.

    sqrt(1 + k^2 - 2 rho k) / sqrt(1 - rho^2) when both weights are positive, otherwise the
    stronger partner alone, max(1, k), as in turing-meter (non-negative weights by choice).
    """
    _check(k, rho)
    if has_interior_peak(k, rho):
        return math.sqrt((1.0 + k * k - 2.0 * rho * k) / (1.0 - rho * rho))
    return max(1.0, float(k))


def tu_app_phi(k, phi_deg) -> float:
    return tu_app(k, float(rho_of(phi_deg)))


def pythagoras(k) -> float:
    """The prediction at 90 degrees, sqrt(1 + k^2)."""
    return math.sqrt(1.0 + k * k)


def q_factor(phi_deg):
    """Quality factor of the pair, Q = 1 / (2 cos(phi/2)), defined for every k."""
    return 1.0 / (2.0 * np.cos(np.radians(phi_deg) / 2.0))


def resonance(w, phi_deg):
    """Right-hand side of the identity: tan^2(phi/2) / (1 + Q^2 (Omega - 1/Omega)^2), Omega = sqrt(w).

    Equal to Tu(w)^2 - 1 for k = 1: the response of a series RLC circuit.
    """
    w = np.asarray(w, float)
    half = np.radians(phi_deg) / 2.0
    q = 1.0 / (2.0 * np.cos(half))
    om = np.sqrt(w)
    return np.tan(half) ** 2 / (1.0 + q * q * (om - 1.0 / om) ** 2)


def band_edges_k1(phi_deg) -> tuple[float, float]:
    """Band edges for k = 1, in closed form: Omega - 1/Omega = -+1/Q, w = Omega^2."""
    q = float(q_factor(phi_deg))
    b = 1.0 / q
    om_high = (b + math.sqrt(b * b + 4.0)) / 2.0
    return 1.0 / om_high ** 2, om_high ** 2


def band_edges(k, rho) -> tuple[float, float]:
    """Weights where Tu^2 - 1 falls to half its maximum, by root finding in ln w.

    (nan, nan) when there is no interior peak. The high edge is infinite when Tu^2 - 1 never
    falls below the half maximum on the machine's side (k > 1 and the machine alone already
    above it).
    """
    _check(k, rho)
    if not has_interior_peak(k, rho):
        return math.nan, math.nan
    ws = w_star(k, rho)
    half = (tu_app(k, rho) ** 2 - 1.0) / 2.0
    f = lambda u: float(tu2_minus_1(math.exp(u), k, rho)) - half  # noqa: E731
    u0 = math.log(ws)
    lo = u0 - 1.0
    while f(lo) > 0:
        lo -= 1.0
    low = math.exp(brentq(f, lo, u0, xtol=1e-12))
    if k * k - 1.0 >= half:
        return low, math.inf
    hi = u0 + 1.0
    while f(hi) > 0:
        hi += 1.0
    high = math.exp(brentq(f, u0, hi, xtol=1e-12))
    return low, high


def bandwidth(k, rho) -> float:
    """Width of the band as the ratio w_high / w_low (nan without a peak, inf if open)."""
    low, high = band_edges(k, rho)
    return high / low


def log_bandwidth(k, rho) -> float:
    return math.log(bandwidth(k, rho))


def useful_angle(k) -> float:
    """arccos(min(k, 1/k)) in degrees: below it the weaker judge adds nothing to the ideal."""
    if k <= 0:
        raise ValueError("k must be positive")
    return math.degrees(math.acos(min(k, 1.0 / k)))


def curve(k, rho, grid=GRID) -> np.ndarray:
    """The predicted Tu(w) on a grid of weights: no free parameter once k and rho are known."""
    return tu(grid, k, rho)


# --- the signed turing and the null (§ 8 bis, protocol v1.2) ---------------------------
#
# With weights of any sign, Tu(w) = (1 + w k) / sqrt(1 + w^2 + 2 rho w) keeps its sign: it is
# negative where the pair errs systematically. Its square is a Fano profile in
# eps = (w + cos phi) / sin phi, with asymmetry q = (1 - k cos phi) / (k sin phi).

GRID_SIGNED = np.r_[-GRID[::-1], GRID]       # the 49 weights with both signs (98 points, no zero)


def tu_signed(w, k, rho):
    """The signed turing, for weights and k of any sign (|rho| < 1)."""
    if np.any(np.abs(np.asarray(rho)) >= 1.0):
        raise ValueError("rho must be strictly between -1 and 1")
    w = np.asarray(w, float)
    return (1.0 + w * k) / np.sqrt(1.0 + w * w + 2.0 * rho * w)


def fano_eps(w, phi_deg):
    """eps = (w + cos phi) / sin phi: the weight in the Fano variable."""
    p = np.radians(phi_deg)
    return (np.asarray(w, float) + np.cos(p)) / np.sin(p)


def fano_q(k, phi_deg):
    """q = (1 - k cos phi) / (k sin phi): zero at the useful angle, cos phi = 1/k."""
    p = np.radians(phi_deg)
    return (1.0 - k * np.cos(p)) / (k * np.sin(p))


def fano(w, k, phi_deg):
    """Tu(w)^2 written as a Fano profile: k^2 (q + eps)^2 / (1 + eps^2)."""
    e = fano_eps(w, phi_deg)
    q = fano_q(k, phi_deg)
    return k * k * (q + e) ** 2 / (1.0 + e * e)


def fano_signed(w, k, phi_deg):
    """The signed turing in Fano form: k (q + eps) / sqrt(1 + eps^2)."""
    e = fano_eps(w, phi_deg)
    q = fano_q(k, phi_deg)
    return k * (q + e) / np.sqrt(1.0 + e * e)


def null_weight(k) -> float:
    """w0 = -1/k: the signals cancel at every angle. Positive when k < 0."""
    if k == 0:
        return math.inf
    return -1.0 / k


def tu_peak_signed(k, rho) -> float:
    """The extremum of the Fano profile, |k| sqrt(1 + q^2) = sqrt(1 + k^2 - 2 k rho) / sin phi.

    It is the formula of Tu_app without the restriction to non-negative weights. At the
    extremum Tu has the sign of k q: below the useful angle (q < 0, k > 0) it is reached
    with Tu < 0, that is with the sign of the person's weight reversed.
    """
    return math.sqrt((1.0 + k * k - 2.0 * k * rho) / (1.0 - rho * rho))


def peak_weight_signed(k, rho) -> float:
    """The weight of the Fano peak, eps = 1/q; infinite at the useful angle (q = 0)."""
    phi = float(angle_deg(rho))
    q = float(fano_q(k, phi))
    if q == 0:
        return math.inf
    s, c = math.sin(math.radians(phi)), math.cos(math.radians(phi))
    return s / q - c


def null_edges(k, rho, level: float = 0.5) -> tuple[float, float]:
    """The interval around w0 = -1/k in which |Tu| < level, found numerically.

    Returned as (near, far): `near` is the edge closer to w = 0, `far` the other one; an
    edge the curve never reaches (|k| < level on that side) is infinite with its sign.
    """
    if k == 0 or abs(rho) >= 1.0:
        return math.nan, math.nan
    w0 = -1.0 / k
    g = lambda w: abs(float(tu_signed(w, k, rho))) - level  # noqa: E731
    sign = 1.0 if w0 > 0 else -1.0
    # toward zero: |Tu| goes from 0 at w0 to 1 at w = 0 with at most one extremum, above 1
    out = [brentq(g, min(w0, 0.0), max(w0, 0.0), xtol=1e-13)]
    for direction in (sign,):                # away from zero, stepping out to bracket it
        step = abs(w0) / 8.0
        lo, hi = w0, w0 + direction * step
        found = False
        for _ in range(200):
            if g(hi) >= 0:
                found = True
                break
            lo, hi = hi, hi + direction * step
            step *= 1.3
            if abs(hi) > 1e9:
                break
        if not found:
            out.append(direction * math.inf)
            continue
        a, b = (lo, hi) if lo < hi else (hi, lo)
        out.append(brentq(g, a, b, xtol=1e-13))
    return out[0], out[1]


def null_width(k, rho, level: float = 0.5) -> float:
    """Width of the null as the ratio far / near of its edges (both on the side of w0),
    the same convention as the band; infinite if the null is open."""
    near, far = null_edges(k, rho, level)
    return far / near


def null_width_diff(k, rho, level: float = 0.5) -> float:
    """Width of the null as a difference of weights, |far - near|."""
    near, far = null_edges(k, rho, level)
    return abs(far - near)
