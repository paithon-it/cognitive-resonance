import math

import numpy as np
import pytest
from scipy.optimize import minimize_scalar

from cognitive_resonance import model


W = np.exp(np.linspace(-5, 5, 201))


@pytest.mark.parametrize("phi", [10, 45, 90, 120, 150, 175])
def test_resonance_identity_k1(phi):
    rho = math.cos(math.radians(phi))
    lhs = model.tu(W, 1.0, rho) ** 2 - 1.0
    assert np.allclose(lhs, model.resonance(W, phi), rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("phi", [30, 90, 150])
def test_peak_is_2q_at_w1(phi):
    rho = math.cos(math.radians(phi))
    assert model.w_star(1.0, rho) == pytest.approx(1.0)
    assert model.tu(1.0, 1.0, rho) == pytest.approx(2 * model.q_factor(phi))
    assert model.tu_app(1.0, rho) == pytest.approx(1 / math.cos(math.radians(phi) / 2))


@pytest.mark.parametrize("k", [0.3, 0.7, 1.0, 1.5, 4.0])
@pytest.mark.parametrize("rho", [0.2, 0.0, -0.5, -0.9, -0.99])
def test_w_star_is_the_maximum(k, rho):
    if not model.has_interior_peak(k, rho):
        pytest.skip("no interior peak")
    res = minimize_scalar(lambda u: -float(model.tu(math.exp(u), k, rho)), bounds=(-12, 12),
                          method="bounded", options={"xatol": 1e-10})
    assert math.exp(res.x) == pytest.approx(model.w_star(k, rho), rel=1e-4)
    assert -res.fun == pytest.approx(model.tu_app(k, rho), rel=1e-9)


def test_below_useful_angle_stronger_alone():
    assert model.tu_app(0.5, 0.6) == 1.0          # rho > k: the person alone
    assert model.tu_app(2.0, 0.6) == 2.0          # rho > 1/k: the machine alone
    assert model.w_star(2.0, 0.6) == math.inf
    assert model.useful_angle(2.0) == pytest.approx(60.0)


@pytest.mark.parametrize("phi", [20, 60, 90, 130, 170])
def test_band_closed_form_matches_numeric(phi):
    rho = math.cos(math.radians(phi))
    lo, hi = model.band_edges_k1(phi)
    nlo, nhi = model.band_edges(1.0, rho)
    assert lo == pytest.approx(nlo, rel=1e-9)
    assert hi == pytest.approx(nhi, rel=1e-9)
    q = model.q_factor(phi)
    om = math.sqrt(hi)
    assert om - 1 / om == pytest.approx(1 / q)


@pytest.mark.parametrize("k", [0.6, 1.0, 1.3])
@pytest.mark.parametrize("rho", [0.3, -0.4, -0.9])
def test_band_edges_are_half_maximum(k, rho):
    lo, hi = model.band_edges(k, rho)
    half = (model.tu_app(k, rho) ** 2 - 1) / 2
    assert model.tu2_minus_1(lo, k, rho) == pytest.approx(half, rel=1e-8)
    if math.isfinite(hi):
        assert model.tu2_minus_1(hi, k, rho) == pytest.approx(half, rel=1e-8)


def test_band_narrows_with_angle():
    widths = [model.bandwidth(1.0, math.cos(math.radians(p))) for p in (60, 90, 120, 150, 170)]
    assert all(a > b for a, b in zip(widths, widths[1:]))


def test_open_band_when_machine_alone_is_above_half():
    lo, hi = model.band_edges(1.5, 0.0)
    assert math.isfinite(lo) and hi == math.inf


# --- the signed turing and the null --------------------------------------------------------

WS = np.linspace(-20, 20, 4001)


@pytest.mark.parametrize("k", [0.4, 1.0, 2.5, -0.6, -1.7])
@pytest.mark.parametrize("phi", [25, 90, 140])
def test_fano_identity_any_sign(k, phi):
    rho = math.cos(math.radians(phi))
    assert np.allclose(model.tu_signed(WS, k, rho) ** 2, model.fano(WS, k, phi), rtol=1e-10, atol=1e-12)
    assert np.allclose(model.tu_signed(WS, k, rho), model.fano_signed(WS, k, phi), rtol=1e-10, atol=1e-12)


@pytest.mark.parametrize("k", [0.4, 1.0, 2.5, -0.6, -1.7])
@pytest.mark.parametrize("phi", [25, 90, 140])
def test_null_at_minus_one_over_k(k, phi):
    rho = math.cos(math.radians(phi))
    w0 = model.null_weight(k)
    assert w0 == pytest.approx(-1 / k)
    assert model.tu_signed(w0, k, rho) == pytest.approx(0.0, abs=1e-12)
    assert model.fano_eps(w0, phi) == pytest.approx(-model.fano_q(k, phi))
    assert (w0 > 0) == (k < 0)


@pytest.mark.parametrize("k", [1.2, 2.0, 5.0])
def test_q_zero_at_useful_angle(k):
    phi = model.useful_angle(k)                 # cos(phi) = 1/k for k > 1
    assert model.fano_q(k, phi) == pytest.approx(0.0, abs=1e-12)


@pytest.mark.parametrize("k", [0.7, 1.0, 1.6])
@pytest.mark.parametrize("phi", [40, 100, 160])
def test_fano_peak_is_tu_app(k, phi):
    rho = math.cos(math.radians(phi))
    w = model.peak_weight_signed(k, rho)
    q = model.fano_q(k, phi)
    # below the useful angle (q < 0) the extremum of Tu^2 has Tu < 0: the person's weight flips
    assert model.tu_signed(w, k, rho) == pytest.approx(np.sign(q) * model.tu_peak_signed(k, rho), rel=1e-10)
    assert np.max(np.abs(model.tu_signed(WS, k, rho))) <= model.tu_peak_signed(k, rho) + 1e-12
    if model.has_interior_peak(k, rho):
        assert model.tu_peak_signed(k, rho) == pytest.approx(model.tu_app(k, rho))


@pytest.mark.parametrize("k", [0.8, 1.0, 1.9, -0.9, -2.0])
@pytest.mark.parametrize("phi", [30, 90, 150])
def test_null_edges_at_half(k, phi):
    rho = math.cos(math.radians(phi))
    near, far = model.null_edges(k, rho)
    assert abs(model.tu_signed(near, k, rho)) == pytest.approx(0.5, abs=1e-9)
    assert abs(model.tu_signed(far, k, rho)) == pytest.approx(0.5, abs=1e-9)
    assert abs(near) < abs(model.null_weight(k)) < abs(far)
    inside = np.linspace(min(near, far), max(near, far), 501)[1:-1]
    assert np.all(np.abs(model.tu_signed(inside, k, rho)) < 0.5)


def test_null_width_k1_closed_form_and_grows_with_angle():
    for phi in (20, 60, 90, 150):
        t = 1 - math.cos(math.radians(phi))
        assert model.null_width_diff(1.0, 1 - t) == pytest.approx(math.sqrt(4 * t * t + 24 * t) / 3, rel=1e-9)
    ratios = [model.null_width(1.0, math.cos(math.radians(p))) for p in (20, 60, 90, 120, 160)]
    diffs = [model.null_width_diff(1.0, math.cos(math.radians(p))) for p in (20, 60, 90, 120, 160)]
    assert all(a < b for a, b in zip(ratios, ratios[1:]))
    assert all(a < b for a, b in zip(diffs, diffs[1:]))


def test_open_null_when_k_small():
    near, far = model.null_edges(0.4, 0.1)
    assert math.isfinite(near) and far == -math.inf


@pytest.mark.parametrize("k", [-0.8, -0.2, 0.0, 0.5, 1.3])
@pytest.mark.parametrize("rho", [-0.95, -0.4, 0.3])
def test_tu_app_nonneg_is_the_maximum_over_positive_weights(k, rho):
    grid = np.linspace(0, 200, 400001)
    best = max(1.0, float(np.max(model.tu_signed(grid, k, rho))))
    assert model.tu_app_nonneg(k, rho) == pytest.approx(best, rel=1e-5)
