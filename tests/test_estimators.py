import math

import numpy as np
import pytest
from scipy.stats import mannwhitneyu

from cognitive_resonance import estimators as est
from cognitive_resonance import model
from cognitive_resonance.simulate import model_experiment


def _data(k, rho, seed=1, dh=1.0, n_images=2000):
    gen = np.random.default_rng(seed)
    d = model_experiment(gen, k, rho, dh=dh, n_images=n_images)
    zh = est.standardize(d["z_h"], d["y"], d["val_z_h"], d["val_y"])
    zm = est.standardize(d["z_m"], d["y"], d["val_z_m"], d["val_y"])
    return d["y"], d["image"], zh, zm


def test_auc_matches_mann_whitney_with_ties():
    gen = np.random.default_rng(0)
    s = np.round(gen.normal(size=800), 1)
    y = gen.integers(0, 2, 800)
    u = mannwhitneyu(s[y == 1], s[y == 0]).statistic
    assert est.auc(s, y) == pytest.approx(u / ((y == 1).sum() * (y == 0).sum()))


def test_weighted_auc_equals_repeated_rows():
    gen = np.random.default_rng(1)
    s = np.round(gen.normal(size=300), 1)
    y = gen.integers(0, 2, 300)
    w = gen.integers(0, 4, 300)
    rep = np.repeat(np.arange(300), w)
    assert est.auc(s, y, w.astype(float)) == pytest.approx(est.auc(s[rep], y[rep]))
    W = np.stack([w, np.ones(300, int)]).astype(float)
    a = est.SortedScore(s, y).auc(W)[0]
    assert a[0] == pytest.approx(est.auc(s[rep], y[rep]))
    assert a[1] == pytest.approx(est.auc(s, y))


def test_d_a_is_signed():
    gen = np.random.default_rng(2)
    y = np.repeat([0, 1], 5000)
    s = (2 * y - 1) * 0.5 + gen.normal(size=y.size)
    assert est.d_a(s, y) == pytest.approx(1.0, abs=0.06)
    assert est.d_a(-s, y) == pytest.approx(-1.0, abs=0.06)


def test_estimators_recover_model():
    y, image, zh, zm = _data(1.3, -0.6, seed=3, n_images=6000)
    dh, dm = est.d_a(zh, y), est.d_a(zm, y)
    assert dh == pytest.approx(1.0, abs=0.06)
    assert dm / dh == pytest.approx(1.3, abs=0.06)
    assert est.pearson_within(zh, zm, y) == pytest.approx(-0.6, abs=0.03)
    assert est.tetrachoric_within(zh, zm, y) == pytest.approx(-0.6, abs=0.06)
    assert est.d_yesno(zh, y) == pytest.approx(1.0, abs=0.08)


def test_within_class_sd():
    y = np.repeat([0, 1], 4)
    z = np.array([0, 1, 2, 3, 10, 11, 12, 13], float)
    assert est.within_class_sd(z, y) == pytest.approx(math.sqrt(10 / 6))


def test_tuned_turing_close_to_tu_app():
    y, image, zh, zm = _data(1.0, -0.6, seed=4)
    half = est.split_halves(image, y)
    assert np.all(np.bincount(half) == y.size // 2)
    r = est.tuned_turing(zh, zm, y, half)
    assert r["tu_tuned"] == pytest.approx(model.tu_app(1.0, -0.6), rel=0.08)


def test_halves_keep_images_whole():
    image = np.repeat(np.arange(100), 5)
    y = (image >= 50).astype(int)
    half = est.split_halves(image, y)
    for i in range(100):
        assert np.unique(half[image == i]).size == 1
    assert (half[y == 1] == 0).sum() == (half[y == 1] == 1).sum()


def test_measured_band_on_predicted_curve():
    for k, rho in ((1.0, -0.5), (0.8, -0.2), (1.2, 0.1)):
        t = model.curve(k, rho)
        meas = est.band_from_curve(model.GRID, t)
        pred = est.band_predicted(k, rho)
        assert meas["log_width"] == pytest.approx(pred["log_width"], rel=0.03)


def test_measure_pair_and_bootstrap():
    y, image, zh, zm = _data(1.0, -0.6, seed=5)
    design = est.Design(y, image, {"h": zh, "m": zm})
    p = est.measure_pair(zh, zm, y, design.half)
    reps = est.bootstrap(design, {"pair": ("h", "m")}, reps=200)
    p = est.attach_intervals(p, reps, "pair")
    truth = model.tu_app(1.0, -0.6)
    assert p["tu_tuned_ci"][0] < truth < p["tu_tuned_ci"][1]
    assert p["phi_ci"][0] < 126.87 < p["phi_ci"][1]
    assert reps[("pair", "tu_tuned")].shape == (200,)
    fixed = est.bootstrap(design, {"pair": ("h", "m")}, reps=50,
                          fixed_weights={"pair": p["w_tuned_halves"]})
    assert fixed[("pair", "tu_tuned")].shape == (50,)


def test_resample_counts():
    gen = np.random.default_rng(0)
    c = est.resample_counts(gen, 50, 7)
    assert c.shape == (7, 50) and np.all(c.sum(1) == 50)


def test_slope_within():
    gen = np.random.default_rng(6)
    y = np.repeat([0, 1], 3000)
    a = gen.normal(size=y.size)
    z = 2.0 * y - 0.7 * a + gen.normal(size=y.size)
    assert est.slope_within(z, a, y) == pytest.approx(-0.7, abs=0.05)


# --- the null ---------------------------------------------------------------------------------

def test_null_on_predicted_curve():
    for k, rho in ((1.0, 0.0), (1.4, -0.7), (0.8, 0.5)):
        t = model.tu_signed(model.GRID_SIGNED, k, rho)
        w0 = est.null_from_curve(model.GRID_SIGNED, t, side=-1)
        assert w0 == pytest.approx(-1 / k, rel=0.01)
        meas = est.null_width_from_curve(model.GRID_SIGNED, t, w0)
        pred = est.null_predicted(k, rho)
        assert meas["width"] == pytest.approx(pred["width"], rel=0.05)
        assert meas["width_diff"] == pytest.approx(pred["width_diff"], rel=0.05)


def test_null_at_positive_weight_when_k_negative():
    k, rho = -0.8, 0.2
    t = model.tu_signed(model.GRID_SIGNED, k, rho)
    w0 = est.null_from_curve(model.GRID_SIGNED, t, side=+1)
    assert w0 == pytest.approx(1 / 0.8, rel=0.01)
    assert est.null_from_curve(model.GRID_SIGNED, t, side=-1) is None


def test_measure_null_on_model_data():
    y, image, zh, zm = _data(1.0, -0.5, seed=7)
    half = est.split_halves(image, y)
    r = est.measure_null(zh, zm, y, half)
    assert r["w0_k"] == pytest.approx(-1.0, abs=0.06)
    assert r["abs_tu_at_null"] < 0.15          # typical value under the model: about 0.05
    assert r["mae_over_peak"] < 0.05
