"""E6 and E7 (protocol §§ 12-13): the readers, the measures at a fixed weight, and the two analyses on
synthetic files whose answer is known."""
import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

from cognitive_resonance import estimators as est
from cognitive_resonance import model, raw, synthetic

ANALYSIS = Path(__file__).resolve().parents[1] / "analysis"
sys.path.insert(0, str(ANALYSIS))

import fixed  # noqa: E402
import trust  # noqa: E402
import views  # noqa: E402
from rephase_bis import measure  # noqa: E402

SMALL = dict(tasks=("cat_dog", "deer_horse", "bird_frog"), seeds=(2, 3), n_images=400, n_val=400)


# --- readers ---------------------------------------------------------------------------------

def test_trust_reader(tmp_path):
    synthetic.write_trust(tmp_path, tasks=("cat_dog",), seeds=(2,), n_images=40, n_val=20)
    d = raw.read_trust(tmp_path / "cat_dog_s2.npz")
    assert d["machine_names"] == list(raw.TRUST_MACHINES)
    assert d["z_M"].shape == (7, 200) and d["task"] == "cat_dog" and d["seed"] == 2
    assert trust.branches_of(d) == {"Mramo_w0.5": 0.5, "Mramo_w1": 1.0, "Mramo_w2": 2.0, "Mramo_w4": 4.0}


def test_trust_reader_refuses_other_weights(tmp_path):
    synthetic.write_trust(tmp_path, tasks=("cat_dog",), seeds=(2,), n_images=40, n_val=20)
    p = tmp_path / "cat_dog_s2.npz"
    with np.load(p) as f:
        arrays = {k: f[k] for k in f.files}
    arrays["built_for_weight"] = np.array([1, 2, 3, 4], dtype=np.float32)
    np.savez(p, **arrays)
    with pytest.raises(raw.ContractError, match="built_for_weight"):
        raw.read_trust(p)


def test_views_reader(tmp_path):
    synthetic.write_views(tmp_path, tasks=("cat_dog",), seeds=(2,), n_images=40, n_val=20)
    assert sorted(p.name for p in tmp_path.glob("*.npz")) == ["cat_dog_o0_s2.npz", "cat_dog_o100_s2.npz",
                                                             "cat_dog_o50_s2.npz"]
    d = raw.read_views(tmp_path / "cat_dog_o50_s2.npz")
    assert d["overlap"] == 50 and d["task"] == "cat_dog" and d["seed"] == 2
    assert d["machine_names"] == list(raw.VIEWS_MACHINES)


def test_views_reader_refuses_a_name_that_lies(tmp_path):
    synthetic.write_views(tmp_path, tasks=("cat_dog",), seeds=(2,), n_images=40, n_val=20)
    (tmp_path / "cat_dog_o50_s2.npz").rename(tmp_path / "cat_dog_o0_s9.npz")
    with pytest.raises(raw.ContractError, match="overlap"):
        raw.read_views(tmp_path / "cat_dog_o0_s9.npz")
    (tmp_path / "cat_dog_o100_s2.npz").rename(tmp_path / "cat_dog_s2.npz")
    with pytest.raises(raw.ContractError, match="name"):
        raw.read_views(tmp_path / "cat_dog_s2.npz")


# --- the measures at a fixed weight ------------------------------------------------------------

def test_peak_centre_on_the_model():
    grid = model.GRID
    for phi in (100.0, 130.0, 150.0):                       # k = 1: the curve is symmetric in ln w around 1
        p = fixed.peak_centre(grid, model.curve(1.0, model.rho_of(phi), grid))
        assert p["centre"] == pytest.approx(1.0, rel=0.02)
        assert p["estimate"] == p["centre"]
    k, rho = 1.2, model.rho_of(120.0)
    p = fixed.peak_centre(grid, model.curve(k, rho, grid))
    assert p["centre"] == pytest.approx(model.w_star(k, rho), rel=0.1)


def test_peak_centre_open_when_the_curve_only_rises():
    grid = model.GRID
    p = fixed.peak_centre(grid, np.linspace(1.0, 1.4, grid.size))
    assert p["centre"] is None and p["high"] is None
    assert p["estimate"] == p["argmax"] == pytest.approx(grid[-1])


def test_boot_d_a_draws_the_same_images_as_the_bootstrap(tmp_path):
    synthetic.write_trust(tmp_path, tasks=("cat_dog",), seeds=(2,), n_images=200, n_val=200)
    d = raw.read_trust(tmp_path / "cat_dog_s2.npz")
    design, point, judges, boot = measure(d, list(d["machine_names"]), 30)
    mine = fixed.boot_d_a(design, {"alone": d["z_M"][1].astype(float), "h": d["z_H"].astype(float)}, 30)
    assert np.allclose(mine["alone"], boot[("M0plus", "d_a")])      # d_a does not depend on the scale of the cue
    assert np.allclose(mine["h"], boot[("H", "d_a")])


def test_face_value_is_the_raw_sum(tmp_path):
    synthetic.write_trust(tmp_path, tasks=("cat_dog",), seeds=(2,), n_images=200, n_val=200)
    d = raw.read_trust(tmp_path / "cat_dog_s2.npz")
    design, point, judges, boot = measure(d, list(d["machine_names"]), 20)
    fpoint, fboot, dh = fixed.fixed_measures(d, design.cues, design, list(d["machine_names"]), 20)
    y = d["y"]
    assert dh == pytest.approx(float(est.d_a(d["z_H"], y)))
    assert fpoint["M0"]["d_a"]["face"] == pytest.approx(float(est.d_a(d["z_H"] + d["z_M"][0], y)))
    one = design.cues["H"] + design.cues["M0"]
    assert fpoint["M0"]["d_a"]["std1"] == pytest.approx(float(est.d_a(one, y)))
    assert fboot[("M0", "face")].shape == (20,)
    acc = fpoint["M0"]["accuracy"]
    assert acc["face"] == pytest.approx(float(np.mean((d["z_H"] + d["z_M"][0] > 0) == (y == 1))))
    assert acc["alone"] == pytest.approx(float(np.mean((d["z_M"][0] > 0) == (y == 1))))
    assert acc["person_alone"] == fpoint["Mdue"]["accuracy"]["person_alone"]


# --- E6 --------------------------------------------------------------------------------------------

@pytest.fixture(scope="module")
def trust_units(tmp_path_factory):
    folder = tmp_path_factory.mktemp("trust")
    synthetic.write_trust(folder, **SMALL)
    return [trust.unit((f, 40, False)) for f in sorted(folder.glob("*.npz"))]


def test_trust_on_matched_branches(trust_units):
    c = trust.criteria(trust_units)
    assert set(c) == {"F1", "F2", "F3", "F4", "F5", "F6", "F7"}
    assert c["F1"]["outcome"] == "pass"            # at equal weight the frame of H is cancelled only by the branch
    assert c["F3"]["outcome"] == "pass"            # and the branch is paid for by those who trust too much
    assert c["F5"]["outcome"] == "pass"            # the toy puts the peak where the branch is built for
    assert c["F7"]["outcome"] == "pass"
    ratio = c["F5"]["median_peak_over_built_weight"]
    assert all(1 / 1.5 <= ratio[n] <= 1.5 for n in ("Mramo_w0.5", "Mramo_w1", "Mramo_w2"))
    rows = {r["machine"]: r for r in trust.summary_by_machine(trust_units)}
    assert rows["Mramo_w1"]["built_for_weight"] == 1.0 and rows["M0plus"]["built_for_weight"] is None
    assert rows["Mramo_w1"]["tu_std1"] > rows["M0plus"]["tu_std1"]


def test_trust_fails_when_the_branches_ignore_the_weight(tmp_path):
    synthetic.write_trust(tmp_path, matched=False, **SMALL)
    units = [trust.unit((f, 40, False)) for f in sorted(tmp_path.glob("*.npz"))]
    c = trust.criteria(units)
    assert c["F5"]["outcome"] == "fail"            # four identical branches: the peak does not follow the weight
    assert c["F7"]["outcome"] == "fail"


def test_trust_reads_the_files_of_e2_ter(tmp_path):
    synthetic.write_rephase_ter(tmp_path, tasks=("cat_dog", "deer_horse"), seeds=(0, 1), n_images=300, n_val=300)
    units = [trust.unit((f, 30, True)) for f in sorted(tmp_path.glob("*.npz"))]
    assert units[0]["branches"] == {"Mramo": 1.0}
    c = trust.criteria(units)
    assert c["F1"]["outcome"] in ("pass", "fail")
    assert all(c[h]["outcome"] == "not evaluable" for h in ("F5", "F6", "F7"))


def test_trust_verdicts_on_constructed_units(trust_units):
    import copy
    units = copy.deepcopy(trust_units)
    for u in units:                                # the partner no longer beats the soloist at equal weight
        u["differences"]["Mramo_w1"]["std1"] = {"diff": -0.1, "ci": [-0.2, 0.0]}
    assert trust.criteria(units)["F1"]["outcome"] == "fail"
    units = copy.deepcopy(trust_units)
    for u in units:                                # the pair with the partner sinks under the soloist alone
        u["fixed"]["Mramo_w1"]["face_minus_control_alone"]["diff"] = -0.5
    assert trust.criteria(units)["F4"]["outcome"] == "fail"
    units = copy.deepcopy(trust_units)
    for u in units:                                # a branch whose pair is worth much less at its peak
        u["fixed"]["Mramo_w0.5"]["tuned_over_control"] = 0.8
    assert trust.criteria(units)["F6"]["outcome"] == "fail"
    units = copy.deepcopy(trust_units)
    for u in units:                                # the peak of the branch built for 2 sits at 6
        u["fixed"]["Mramo_w2"]["raw_peak"]["estimate"] = 6.0
    assert trust.criteria(units)["F5"]["outcome"] == "fail"


# --- E7 --------------------------------------------------------------------------------------------

@pytest.fixture(scope="module")
def views_units(tmp_path_factory):
    folder = tmp_path_factory.mktemp("views")
    synthetic.write_views(folder, **SMALL)
    return [views.unit((f, 40)) for f in sorted(folder.glob("*.npz"))]


def test_views_where_the_person_knows_something_of_its_own(views_units):
    c = views.criteria(views_units)
    assert set(c) == {"V1", "V2", "V3", "V4", "V5", "V6"}
    assert c["V1"]["outcome"] == "pass"            # without overlap, cancelling the frame frees what H sees
    assert c["V2"]["outcome"] == "pass"            # and the gain dies out as the views become the same
    assert c["V3"]["outcome"] == "pass"
    assert c["V4"]["outcome"] == "pass"            # the toy's Mpair learns only what H does not see
    assert c["V5"]["outcome"] == "pass"
    g = c["V2"]["median_difference_by_overlap"]
    assert g["0"] > g["50"] > g["100"]
    rows = views.summary(views_units)
    assert {(r["overlap"], r["machine"]) for r in rows} == {(o, m) for o in (100, 50, 0) for m in raw.VIEWS_MACHINES}


def test_views_fail_when_the_overlap_changes_nothing(tmp_path):
    synthetic.write_views(tmp_path, flat=True, **SMALL)
    units = [views.unit((f, 40)) for f in sorted(tmp_path.glob("*.npz"))]
    c = views.criteria(units)
    assert c["V1"]["outcome"] == "fail"            # the machine sees what H sees: nothing to free
    assert c["V2"]["outcome"] == "fail"
    assert c["V4"]["outcome"] == "fail"
    assert c["V5"]["outcome"] == "fail"


def test_views_verdicts_on_constructed_units(views_units):
    import copy
    units = copy.deepcopy(views_units)
    for u in units:
        if u["overlap"] == 0:                      # the partner is not worse alone: V3 cannot pass
            u["differences"]["Mramo"]["alone"] = {"diff": 0.1, "ci": [0.0, 0.2]}
    c = views.criteria(units)
    assert c["V3"]["outcome"] == "fail" and c["V1"]["outcome"] == "pass"
    units = copy.deepcopy(views_units)
    for u in units:
        if u["overlap"] == 100:                    # with the same view Mpair learns the task all the same
            u["mpair_alone_over_soloist_alone"] = 0.9
    assert views.criteria(units)["V4"]["outcome"] == "fail"
    units = copy.deepcopy(views_units)
    g0 = views.criteria(views_units)["V2"]["median_difference_by_overlap"]["0"]
    for u in units:
        if u["overlap"] == 100:                    # with the same view the gain is half of the gain without overlap
            u["differences"]["Mramo"]["pair_tuned"] = {"diff": 0.5 * g0, "ci": [0.4 * g0, 0.6 * g0]}
        if u["overlap"] == 50:
            u["differences"]["Mramo"]["pair_tuned"] = {"diff": 0.7 * g0, "ci": [0.6 * g0, 0.8 * g0]}
    c = views.criteria(units)
    assert c["V2"]["outcome"] == "fail" and c["V2"]["same_view_over_no_overlap"] == pytest.approx(0.5)


def test_std_key_names():
    assert [fixed.std_key(w) for w in fixed.STD_WEIGHTS] == ["std0.5", "std1", "std2", "std8"]
    assert math.isclose(fixed.PEAK_LEVEL, 0.9)


# --- E1 read again: the identity of the sample moments -------------------------------------------------

def test_the_formula_is_an_identity_for_the_signal_to_noise_ratio(tmp_path):
    import identity
    synthetic.write_bench(tmp_path, tasks=("cat_dog",), mus=(0.0, 1.0, 2.0), seeds=(0,), n_images=300, n_val=300)
    units = [identity.unit(f) for f in sorted(tmp_path.glob("*.npz"))]
    assert all(u["snr_identity_max_relative_error"] < 1e-10 for u in units)
    rows = identity.by_mu(units)
    assert [r["mu"] for r in rows] == [0.0, 1.0, 2.0] and all(r["pairs"] == 1 for r in rows)


def test_the_identity_does_not_need_gaussian_cues():
    import identity
    gen = np.random.default_rng(3)
    n = 4000
    y = (np.arange(n) >= n // 2).astype(int)
    a = gen.exponential(1.0, n) * np.where(gen.random(n) < 0.5, 1.0, -3.0)        # skewed, heavy-tailed
    zh = 0.8 * (2 * y - 1) + gen.standard_t(3, n) + a
    zm = 0.5 * (2 * y - 1) + gen.laplace(0, 1, n) - 0.7 * a
    rho = float(est.pearson_within(zh, zm, y))
    sh, sm = est.within_class_sd(zh, y), est.within_class_sd(zm, y)
    for w in (0.3, 1.0, 5.0):
        wu = w * sm / sh
        predicted = (1 + wu * identity.snr(zm, y) / identity.snr(zh, y)) / math.sqrt(1 + wu * wu + 2 * rho * wu)
        assert identity.snr(zh + w * zm, y) / identity.snr(zh, y) == pytest.approx(predicted, rel=1e-10)
    assert abs(identity.excess_kurtosis(zh + zm, y)) > 1.0                         # and these cues are far from Gaussian


# --- E8: the bench of the chart and its theorem -----------------------------------------------------------

def test_the_theorem_of_the_chart_against_the_matrices():
    from cognitive_resonance import chart as theorem
    gen = np.random.default_rng(7)
    for _ in range(2000):
        dh, dm = gen.uniform(0.2, 4, 2)
        r, b = gen.uniform(-0.95, 0.95), gen.uniform(0, 5)
        mean = np.array([dh, dm])
        d1 = math.sqrt(mean @ np.linalg.solve(np.array([[1 + b, r], [r, 1]]), mean))
        d2 = math.sqrt(mean @ np.linalg.solve(np.array([[1, r], [r, 1]]), mean))
        assert theorem.with_soloist(dh, dm, r, b) == pytest.approx(d1, rel=1e-10)
        assert theorem.with_partner(dh, dm, r) == pytest.approx(d2, rel=1e-10)
        assert theorem.gain(dh, dm, r, b) == pytest.approx(d2 ** 2 - d1 ** 2, abs=1e-9)
        assert theorem.private(dh, dm, r) == pytest.approx(d2 ** 2 - dm ** 2, abs=1e-9)
        lo, hi = theorem.blind_band(r, b)
        if abs(dh - d1) > 1e-9:                         # the blind corrector wins exactly inside the band
            assert (dh > d1) == (lo < dm / dh < hi)


def test_the_private_part_vanishes_at_the_useful_angle():
    from cognitive_resonance import chart as theorem
    assert theorem.private(1.0, 2.0, 0.5) == pytest.approx(0.0)           # r = d_h / d_m
    assert theorem.gain(1.0, 2.0, 0.5, 3.0) == pytest.approx(0.0)
    assert theorem.blind_band(0.9, 0.1)[0] > 0                            # r^2 > B: a lower threshold appears
    assert theorem.blind_band(0.5, 1.0)[0] == 0.0


def test_chart_reader(tmp_path):
    synthetic.write_chart(tmp_path, tasks=("cat_dog",), seeds=(2,), n_images=40, n_val=20)
    d = raw.read_chart(tmp_path / "cat_dog_v25_s2.npz")
    assert d["v"] == 25 and d["machine_names"] == list(raw.CHART_MACHINES) and d["z_H0"].shape == (200,)
    (tmp_path / "cat_dog_v50_s2.npz").rename(tmp_path / "cat_dog_v25_s9.npz")
    with pytest.raises(raw.ContractError, match="v must"):
        raw.read_chart(tmp_path / "cat_dog_v25_s9.npz")


@pytest.fixture(scope="module")
def chart_units(tmp_path_factory):
    import chart
    folder = tmp_path_factory.mktemp("chart")
    synthetic.write_chart(folder, tasks=("cat_dog", "deer_horse", "bird_frog"), seeds=(2, 3), n_images=500, n_val=500)
    return [chart.unit((f, 40)) for f in sorted(folder.glob("*.npz"))]


def test_chart_where_the_judge_sees_more_than_the_machine(chart_units):
    import chart
    c = chart.criteria(chart_units)
    assert set(c) == {"K1", "K2", "K3", "K4", "K5", "K6", "K7"}
    assert all(c[k]["outcome"] == "pass" for k in ("K1", "K2", "K3", "K4", "K5", "K7"))
    g = c["K3"]["median_gain_by_v"]
    assert g["25"] > g["50"] > g["100"]
    u = next(u for u in chart_units if u["v"] == 25 and u["task"] == "cat_dog")
    d = 1.8                                             # the toy's cat_dog: the clean cue has noise variance d^2 + 0.16
    assert u["theorem"]["b"] == pytest.approx(1.1 ** 2 * 2 / (d ** 2 + 0.16), rel=0.1)
    assert u["theorem"]["slope_of_h_on_a"] > 0
    assert u["measured"]["d2_three_cues"] >= u["measured"]["d1"]
    rows = chart.summary(chart_units)
    assert [r["v"] for r in rows] == [100, 50, 25]


def test_chart_fails_when_the_machine_sees_everything(tmp_path):
    import chart
    synthetic.write_chart(tmp_path, tasks=("cat_dog", "deer_horse", "bird_frog"), seeds=(2, 3), n_images=500, n_val=500,
                          flat=True)
    units = [chart.unit((f, 40)) for f in sorted(tmp_path.glob("*.npz"))]
    c = chart.criteria(units)
    assert c["K2"]["outcome"] == "fail" and c["K3"]["outcome"] == "fail" and c["K5"]["outcome"] == "fail"


def test_chart_reads_the_files_of_e2(tmp_path):
    import chart
    synthetic.write_rephase_ter(tmp_path / "rephase_ter", tasks=("cat_dog",), seeds=(0, 1), n_images=300, n_val=300)
    synthetic.write_rephase_bis(tmp_path / "rephase_bis", tasks=("cat_dog",), seeds=(0, 1), n_images=300, n_val=300)
    units = [chart.old_unit((f, "bis")) for f in sorted((tmp_path / "rephase_bis").glob("*.npz"))]
    units += [chart.old_unit((f, "ter")) for f in sorted((tmp_path / "rephase_ter").glob("*.npz"))]
    s = chart.old_summary(units)
    assert s["E2 bis"]["units"] == 2 and s["E2 ter"]["units"] == 2
    assert s["E2 ter"]["theorem_d2"] >= s["E2 ter"]["theorem_d1"]


# --- E9: the shape of the noise -----------------------------------------------------------------------------

def test_shape_reader(tmp_path):
    synthetic.write_shape(tmp_path, tasks=("cat_dog",), mus=(2.0,), seeds=(2,), n_images=40, n_val=20)
    d = raw.read_shape(tmp_path / "cat_dog_mu2_s2.npz")
    assert d["shape_names"] == list(raw.SHAPES) and d["z_plus"].shape == (4, 200) and d["mu"] == 2.0
    with np.load(tmp_path / "cat_dog_mu2_s2.npz") as f:
        arrays = {k: f[k] for k in f.files}
    arrays["shapes"] = np.array(["gauss", "two", "t3", "uniform"])
    np.savez(tmp_path / "cat_dog_mu2_s2.npz", **arrays)
    with pytest.raises(raw.ContractError, match="shapes"):
        raw.read_shape(tmp_path / "cat_dog_mu2_s2.npz")


def test_the_frames_of_the_four_shapes_have_the_same_variance():
    gen = np.random.default_rng(5)
    for shape in synthetic.SHAPES:
        a = synthetic._unit_noise(gen, shape, 400_000)
        assert abs(a.mean()) < 0.02
        assert a.var() == pytest.approx(1.0, rel=0.05 if shape != "t3" else 0.5)     # the t has heavy tails
    assert set(np.unique(synthetic._unit_noise(gen, "two", 100))) == {-1.0, 1.0}


def test_shape_where_the_formula_breaks(tmp_path):
    import shape
    synthetic.write_shape(tmp_path, tasks=("cat_dog", "deer_horse", "bird_frog"), seeds=(2, 3), n_images=1500, n_val=1000)
    units = [shape.unit(f) for f in sorted(tmp_path.glob("*.npz"))]
    c = shape.criteria(units)
    assert set(c) == {"S1", "S2", "S3", "S4", "S5"}
    assert c["S1"]["outcome"] == "pass" and c["S2"]["outcome"] == "pass" and c["S5"]["outcome"] == "pass"
    assert max(s["snr_identity_relative_error"] for u in units for s in u["shapes"].values()) < 1e-9
    rows = shape.by_mu(units)
    two = next(r for r in rows if r["mu"] == 2.0 and r["shape"] == "two")
    assert two["d_a_over_snr_single"] > 1.1 and two["d_a_over_snr_pair"] == pytest.approx(1.0, abs=0.05)


def test_shape_changes_nothing_when_every_frame_is_gaussian(tmp_path):
    import shape
    synthetic.write_shape(tmp_path, tasks=("cat_dog", "deer_horse", "bird_frog"), seeds=(2, 3), n_images=1500, n_val=1000,
                          all_gauss=True)
    units = [shape.unit(f) for f in sorted(tmp_path.glob("*.npz"))]
    c = shape.criteria(units)
    assert c["S1"]["outcome"] == "pass" and c["S2"]["outcome"] == "fail" and c["S3"]["outcome"] == "fail"


# --- E6, the design reading (protocol section 12 bis) ---------------------------------------------------------

def test_floor_of_the_model():
    """With k >= 1 the pair is at one turing or above at every positive weight; with k < 1 it is not."""
    gen = np.random.default_rng(5)
    for _ in range(200):
        rho, w = gen.uniform(-0.95, 0.95), math.exp(gen.uniform(-5, 5))
        assert model.tu(w, 1.0 + gen.exponential(0.5), rho) >= 1.0 - 1e-12
        k = gen.uniform(0.2, 0.99)
        assert model.tu(1e4, k, rho) < 1.0


def _pair_with_bias(w, rest):
    """The turing of a pair whose sum keeps `rest` times the bias of the judge: signal 1 + 1.4 w, the noise of the
    judge 1, of the machine 1 (uncorrelated), the bias of the judge with variance 0.8."""
    return (1 + 1.4 * w) / np.sqrt(1 + w * w + 0.8 * rest ** 2) * math.sqrt(1.8)


def test_break_even_at_twice_the_built_weight():
    import design
    g = model.GRID
    soloist = _pair_with_bias(g, 1.0)
    for w_b in (0.5, 1.0, 2.0, 4.0):
        partner = _pair_with_bias(g, 1.0 - g / w_b)
        assert design.break_even(partner, soloist, w_b) == pytest.approx(2 * w_b, rel=0.03)
    assert design.break_even(soloist - 0.01, soloist, 1.0) is None          # below from the start: open
    assert design.break_even(soloist + 0.01, soloist, 1.0) is None          # never below: open


def test_rule_of_choice():
    import design
    g = model.GRID
    curves = {"flat": np.full(g.size, 1.30), "peaked": 1.2 + 0.3 * np.exp(-np.log(g / 2.0) ** 2),
              "harmful": np.full(g.size, 1.60)}
    k = {"flat": 1.4, "peaked": 1.2, "harmful": 0.8}
    chosen, table = design.choose(curves, k, 1.0, 4.0)
    assert chosen == "peaked" and not table["harmful"]["holds_floor"]       # the best worst case above the floor
    assert design.choose(curves, k, 1 / 32, 32.0)[0] == "flat"              # a wide range of people: the flat one
    assert design.choose(curves, {n: 0.5 for n in curves}, 1.0, 4.0)[0] is None
    assert design.points(1.0, 2.0)[0] == 1.0 and design.points(1.0, 2.0)[-1] == 2.0
    assert design.partner_for(0.285, 8.0, {"a": 0.5, "b": 1.0, "c": 2.0, "d": 4.0}) == "c"
    assert design.at(list(range(49)), g[10]) == pytest.approx(10.0)
    assert design.at(list(range(49)), 1e6) == 48.0                          # outside the grid: the end value


def test_design_reading_on_the_toy(trust_units):
    import design
    r = design.reading({"units": trust_units})
    assert set(r["criteria"]) == {"G1", "G2", "G3", "G4"}
    assert all(c["outcome"] in ("pass", "fail") for c in r["criteria"].values())
    assert [row["fraction"] for row in r["by_fraction"]] == list(design.FRACTIONS)
    one = r["break_even"]["Mramo_w1"]
    assert one["predicted"] == 2.0 and one["closed_units"] == len(trust_units)
    assert 1 / 1.5 <= one["median_over_predicted"] <= 1.5                   # the toy is additive: twice the weight
    assert r["choice"]["chosen"] in raw.TRUST_MACHINES and r["choice"]["table"][r["choice"]["chosen"]]["holds_floor"]
    assert 0 <= r["choice"]["units_with_the_same_choice"] <= len(trust_units)


def test_design_reading_with_one_branch_is_not_judged(trust_units):
    import copy
    import design
    units = copy.deepcopy(trust_units)
    for u in units:
        u["machines"] = ["M0", "M0plus", "Mdue", "Mramo_w1"]
        u["branches"] = {"Mramo_w1": 1.0}
    r = design.reading({"units": units})
    assert all(c["outcome"] == "not evaluable" for c in r["criteria"].values())
    assert r["by_fraction"][2]["partner"] == "Mramo_w1" and r["choice"]["expectation"]["met"] is None


# --- readings added after the results of E6 and E7 (not tests of the hypotheses) ---------------------------------

def test_chart_reads_the_files_of_e7(tmp_path):
    import chart
    synthetic.write_views(tmp_path, **SMALL)
    units = [chart.views_unit(f) for f in sorted(tmp_path.glob("*.npz"))]
    rows = chart.views_summary(units)
    assert [r["overlap"] for r in rows] == list(raw.VIEWS_OVERLAPS)
    for r in rows:
        assert r["units"] == 6 and r["theorem_d2"] >= r["theorem_d1"] - 1e-9
        assert r["measured_d_blind_two_cues"] == pytest.approx(r["theorem_d_blind"], rel=0.1)   # the same thing, two ways
        assert r["measured_d1"] == pytest.approx(r["theorem_d1"], rel=0.1)
    apart, same = rows[-1], rows[0]                    # nothing in common against the same view
    assert apart["theorem_d2"] - apart["theorem_d1"] > same["theorem_d2"] - same["theorem_d1"]
    assert apart["units_where_partner_beats_blind"] == apart["units"]


def test_the_push_of_the_branch_on_the_toy(tmp_path, trust_units):
    import push
    synthetic.write_trust(tmp_path, **SMALL)                       # the same files as the fixture: same seeds
    rows = [push.slopes(f) for f in sorted(tmp_path.glob("*.npz"))]
    r = push.reading({"units": trust_units}, rows)
    assert r["units"] == len(trust_units) and r["median_b"] > 0
    assert list(r["by_branch"]) == ["Mramo_w0.5", "Mramo_w1", "Mramo_w2", "Mramo_w4"]
    for name, s in r["by_branch"].items():
        assert s["predicted_break_even"] == 2 * s["built_for_weight"]
        assert s["median_b_over_c_over_built_weight"] == pytest.approx(1.0, abs=0.1)   # the toy cancels the push exactly
        assert abs(s["median_push_left_at_the_built_weight"]) < 0.1
    one = r["by_branch"]["Mramo_w1"]["break_even_against_same_machine"]
    assert one["closed_units"] == len(trust_units) and 1 / 1.5 <= one["median_over_predicted"] <= 1.5


# --- third review: the informative trace, the judgments needed, the theorem on pairs with the bias in -----------------

def _d2(delta, sigma):
    return float(np.asarray(delta) @ np.linalg.solve(np.asarray(sigma), np.asarray(delta)))


def test_the_theorem_with_an_informative_trace_against_the_matrices():
    from cognitive_resonance import chart as th
    gen = np.random.default_rng(7)
    for _ in range(200):
        d_h, d_m, d_a = gen.uniform(0.3, 3.0, 3)
        r, beta = gen.uniform(-0.9, 0.9), gen.uniform(-2.0, 2.0)
        d1 = _d2([d_h + beta * d_a, d_m], [[1 + beta ** 2, r], [r, 1]])
        d2 = _d2([d_h, d_m, d_a], [[1, r, 0], [r, 1, 0], [0, 0, 1]])
        assert th.with_soloist_informative(d_h, d_m, r, beta, d_a) ** 2 == pytest.approx(d1)
        assert th.with_partner_informative(d_h, d_m, r, d_a) ** 2 == pytest.approx(d2)
        assert th.gain_informative(d_h, d_m, r, beta, d_a) == pytest.approx(d2 - d1, abs=1e-9)
        if abs(beta) > 1e-3 and abs(d_h - r * d_m) > 1e-3:               # the form of the article
            b, b0 = beta ** 2, th.deserved_weight(d_h, d_m, r, d_a)
            form = b / (1 - r ** 2 + b) * th.private(d_h, d_m, r) * (1 - b0 / beta) ** 2
            assert th.gain_informative(d_h, d_m, r, beta, d_a) == pytest.approx(form, rel=1e-9, abs=1e-9)
    assert th.gain_informative(2.0, 1.5, 0.4, 0.8, 0.0) == pytest.approx(th.gain(2.0, 1.5, 0.4, 0.64))    # d_a = 0
    b0 = th.deserved_weight(2.0, 1.5, 0.4, 0.7)
    assert th.gain_informative(2.0, 1.5, 0.4, b0, 0.7) == pytest.approx(0.0, abs=1e-12)                    # deserved


def test_the_judgments_needed_and_the_pair_with_its_bias_in():
    from cognitive_resonance import chart as th
    d_h, d_m, r, b = 2.9, 1.9, 0.5, 0.7
    assert th.share_kept_after(1e9, r, b) == pytest.approx(1.0, abs=1e-6)
    assert 0 < th.share_kept_after(20, r, b) < th.share_kept_after(200, r, b) < 1
    # the same pair, as it would be measured with the bias in: d_H, rho; the gain from B must be the gain of the theorem
    d_big, rho = d_h / math.sqrt(1 + b), r / math.sqrt(1 + b)
    assert th.private_observed(d_big, d_m, rho) == pytest.approx(th.private(d_h, d_m, r) * (1 - r ** 2) / (1 - r ** 2 + b))
    assert th.gain_if_bias(d_big, d_m, rho, b) == pytest.approx(th.gain(d_h, d_m, r, b))
    assert th.gain_if_bias(1.0, 1.0, 0.8, 1.0) is None


def test_the_ceiling_at_any_k_and_two_systems_that_adapt():
    for k in (0.5, 1.0, 2.0):
        for w in (0.25, 0.5, 2.0, 4.0):
            assert all(model.tu(w, k, rho) <= model.ceiling_at_weight(w, k) + 1e-9 for rho in np.linspace(-0.999, 0.999, 41))
    assert model.ceiling_at_weight(2.0) == pytest.approx(3.0) and model.ceiling_at_weight(0.5) == pytest.approx(3.0)
    for gamma, eta in ((0.2, 0.3), (0.5, 0.9), (0.1, 1.2), (0.9, 0.05)):
        r = [1.0, 1.0 - gamma]
        for _ in range(60):
            r.append((1 - gamma) * r[-1] - eta * r[-2])
        modulus = max(abs(x) for x in model.adaptation_roots(gamma, eta))
        assert (abs(r[-1]) < 1e-3) == (modulus < 0.89) or 0.89 <= modulus <= 1.0      # it dies out iff the roots are inside
        if (1 - gamma) ** 2 < 4 * eta:
            assert modulus == pytest.approx(math.sqrt(eta))                              # complex roots: modulus sqrt(eta)


def test_the_three_readings_of_the_third_review(tmp_path):
    import json as _json
    import informative
    import judgments
    import private_people
    synthetic.write_chart(tmp_path / "chart", tasks=("cat_dog", "deer_horse"), seeds=(2, 3), n_images=600, n_val=600)
    files = sorted((tmp_path / "chart").glob("*_v25_*.npz"))
    us = [judgments.unit(f) for f in files]
    for u in us:
        few, many = u["by_n"]["10"], u["by_n"]["1000"]
        assert 0.0 <= many["predicted"] <= 1.0 and many["predicted"] > few["predicted"]
        assert many["share_kept"] > 0.9                                   # with a thousand judgments the bias is known
    rows = judgments.summary(us)
    assert rows and rows[0]["v"] == 25 and set(rows[0]["by_n"]) == {str(n) for n in judgments.NS}
    ui = [informative.unit(f) for f in files]
    for u in ui:
        for target, x in u["by_d_a"].items():
            assert x["d_a_frame"] == pytest.approx(float(target), abs=0.25)   # the resampling gives the frame that sensitivity
            assert x["theorem_d2"] >= x["theorem_d1"] - 1e-9
        removed = [u["by_d_a"][k]["share_of_the_push_removed_by_new_branch"] for k in ("0", "0.5", "1")]
        assert removed[0] > removed[1] > removed[2]                       # the more the frame says, the less is taken away
    assert informative.summary(ui)[0]["units"] == len(files)
    clone = tmp_path / "tm" / private_people.MAP.parent
    clone.mkdir(parents=True)
    rows = [{"source": "chatbench", "task": "all", "mode": "letter", "machine": "m", "dh": 1.1, "dm": 1.7, "rho": 0.2},
            {"source": "chatbench", "task": "math", "mode": "letter", "machine": "m", "dh": 1.0, "dm": 1.5, "rho": 0.2},
            {"source": "haiid", "task": "art", "mode": "given", "machine": "advice", "dh": 1.0, "dm": 1.5, "rho": 0.2},
            {"source": "haiid", "task": "art", "mode": "letter", "machine": "m", "dh": 0.9, "dm": 2.5, "rho": 0.3},
            {"source": "alphanli", "task": "alphanli", "mode": "reason", "machine": "m", "excluded": "at chance"}]
    (clone / private_people.MAP.name).write_text(_json.dumps({"comparisons": rows}))
    kept = [private_people.read(r) for r in rows if private_people.group(r)]
    assert [k["source"] for k in kept] == ["chatbench", "haiid-images"]
    one = kept[0]
    assert one["pair_now"] ** 2 == pytest.approx(one["d_m"] ** 2 + one["private_observed"])
    assert one["if_bias"]["1"]["pair_over_pair_now"] > one["if_bias"]["0.25"]["pair_over_pair_now"] > 1.0
    assert private_people.summary(kept)[-1]["source"] == "all"



def test_the_right_answers_of_e8_pair_by_pair():
    """The table added to the results page reads the units: medians, the paired gain with its range, two counts."""
    import results_v2
    acc = lambda h, hm, hb, hr, h0: {"H": h, "H+M": hm, "H+Mblind": hb, "H+Mramo": hr, "H0": h0}
    d = {"units": [{"v": 25, "accuracy": acc(0.80, 0.85, 0.90, 0.88, 0.93)},
                   {"v": 25, "accuracy": acc(0.82, 0.86, 0.90, 0.91, 0.90)},
                   {"v": 25, "accuracy": acc(0.84, 0.87, 0.91, 0.93, 0.95)}]}
    lines = results_v2._e8_right_answers(d).splitlines()
    cells = [c.strip() for c in lines[2].strip("|").split("|")]
    assert len(lines) == 3 and cells[:2] == ["25", "3"]
    assert cells[2:7] == ["0.820", "0.860", "0.900", "0.910", "0.930"]
    assert cells[7] == "0.050 (0.030 to 0.060)"
    assert cells[8] == "2 of 3 (-0.020 to 0.020)" and cells[9] == "2 of 3"


def test_yes_and_no_are_worth_less_than_a_graded_judgment():
    """The Fisher factor of protocol § 18: pi / 2 at best, larger for a sensitive judge, and checked by simulation."""
    from cognitive_resonance import chart
    assert chart.yes_no_factor(0.0, 0.0, 0.0, 1.0) == pytest.approx(math.pi / 2, rel=1e-3)
    easy, hard = chart.yes_no_factor(-0.5, 0.5, 0.3, 2.0), chart.yes_no_factor(-1.45, 1.45, 0.6, 2.0)
    assert math.pi / 2 < easy < hard and hard == pytest.approx(5.28, abs=0.05)
    # simulation: the variance of the probit slope against the variance of the graded slope, one class, intercept known
    gen = np.random.default_rng(5)
    a = gen.normal(0.0, math.sqrt(2.0), 400000)
    from scipy.stats import norm
    eta = 0.6 * a
    lam = norm.pdf(eta) ** 2 / (norm.cdf(eta) * norm.sf(eta))
    by_hand = 2.0 / (np.mean(lam * a * a) - np.mean(lam * a) ** 2 / np.mean(lam))
    assert chart.yes_no_factor(0.0, 0.0, 0.6, 2.0) == pytest.approx(by_hand, rel=0.02)


# --- E10 to E12: the bench of the informative chart ----------------------------------------------------------

def test_trace_reader_and_contract(tmp_path):
    synthetic.write_trace(tmp_path, tasks=("cat_dog",), seeds=(7,), n_images=60, n_val=40)
    d = raw.read_trace(tmp_path / "cat_dog_v25_s7.npz")
    assert d["z_R"].shape == (4, 4, 300) and d["z_H"].shape == (4, 300) and d["z_M"].shape == (300,)
    assert d["val_a"].shape == (4, 200) and d["v"] == 25 and d["seed"] == 7
    bad = dict(np.load(tmp_path / "cat_dog_v25_s7.npz"))
    bad["z_R"] = bad["z_R"][:, :3]
    np.savez(tmp_path / "cat_dog_v25_s9.npz", **bad)
    with pytest.raises(raw.ContractError):
        raw.read_trace(tmp_path / "cat_dog_v25_s9.npz")


def test_trace_analysis_on_synthetic_files(tmp_path):
    """Where the branch built for the level leaves what the frame is worth, new beats old and old hurts."""
    import trace
    synthetic.write_trace(tmp_path, tasks=("cat_dog", "deer_horse", "bird_frog"), seeds=(7, 8), n_images=1200, n_val=400)
    units = [trace.unit((f, 60)) for f in raw.list_units(tmp_path, raw.TRACE_RE)]
    assert len(units) == 12 and set(units[0]["levels"]) == {"0", "0.5", "1", "1.5"}
    u = next(x for x in units if x["v"] == 25)
    zero = u["levels"]["0"]
    assert zero["share_removed"]["new"] == pytest.approx(1.0, abs=0.05)
    assert zero["share_removed"]["old"] == pytest.approx(zero["share_removed"]["new"], abs=1e-9)   # the same branch
    assert zero["face_new_minus_old"]["diff"] == pytest.approx(0.0, abs=1e-9)
    assert u["levels"]["1"]["share_removed"]["new"] < u["levels"]["0.5"]["share_removed"]["new"] < 1.0
    c = trace.criteria(units)
    assert set(c) == {"T1", "T2", "T3", "T4", "T5", "T6", "X1", "X2", "X3", "X4", "X5"}
    assert c["X1"]["outcome"] == "pass" and c["X3"]["outcome"] == "pass" and c["X5"]["outcome"] == "pass"
    assert c["X2"]["outcome"] == "pass"
    rows = trace.summary(units)
    assert len(rows) == 8 and rows[0]["units"] == 6


def test_probit_slope_and_judgments_needed():
    import yesno
    gen = np.random.default_rng(3)
    n = 60000
    y = (np.arange(n) % 2).astype(int)
    a = gen.normal(0.0, math.sqrt(2.0), n)
    answer = (1.2 * (2 * y - 1) + 0.5 * a + gen.standard_normal(n)) > 0
    assert yesno.probit_slope(answer, y, a) == pytest.approx(0.5, abs=0.03)
    assert yesno.probit_slope(answer[:6], y[:6], a[:6]) == pytest.approx(yesno.probit_slope(answer[:6], y[:6], a[:6]))
    assert abs(yesno.probit_slope(np.array([True, True, False, False]), np.array([1, 1, 0, 0]), np.array([1.0, 2.0, -1.0, -2.0]))) < 5
    assert yesno.needed((10, 100, 1000), (0.5, 0.9, 1.0)) == pytest.approx(100.0)
    assert yesno.needed((10, 100, 1000), (0.8, 1.0, 1.0)) == pytest.approx(math.sqrt(1000.0), rel=1e-6)
    assert yesno.needed((10, 100), (0.5, 0.6)) == math.inf and yesno.needed((10, 100), (0.95, 0.99)) == 10.0


def test_yes_or_no_needs_more_judgments_on_synthetic_files(tmp_path):
    import yesno
    synthetic.write_trace(tmp_path, tasks=("cat_dog", "deer_horse"), seeds=(7,), n_images=1500, n_val=1000)
    units = [u for u in (yesno.unit(f) for f in raw.list_units(tmp_path, raw.TRACE_RE)) if u is not None]
    assert len(units) == 2 and all(u["v"] == 25 for u in units)
    u = units[0]
    assert u["kappa"] > math.pi / 2 and u["by_n"]["5000"]["draws"] == 1
    assert u["by_n"]["5000"]["yes_no"] > 0.8 and u["by_n"]["20"]["yes_no"] < u["by_n"]["20"]["graded"] + 0.05
    assert u["needed_yes_no"] > u["needed_graded"]
    c = yesno.criteria(units)
    assert set(c) == {"Y1", "Y2", "Y3", "Y4"} and c["Y4"]["outcome"] == "pass"


def test_the_size_of_a_documented_bias():
    """3.3 percentage points around a rate of 0.29 are a bias of about two thousandths of the noise."""
    from cognitive_resonance import chart
    b = chart.bias_from_shift(0.033, 0.29)
    assert b == pytest.approx(0.0019, abs=0.0003)
    assert chart.bias_from_shift(0.066, 0.29) == pytest.approx(4 * b, rel=1e-9)
    assert chart.bias_from_shift(0.033, 0.5) < b                      # the same shift is a smaller push where phi is largest
    assert chart._probit(0.5) == pytest.approx(0.0, abs=1e-9) and chart._probit(0.975) == pytest.approx(1.96, abs=1e-3)


def test_what_the_branch_leaves_to_the_frame(tmp_path):
    """The reading added after E10: on synthetic files the branch leaves the frame's own log-likelihood slope."""
    import trace
    import trace_left
    assert trace_left.own_slope(0.0) == 0.0 and trace_left.own_slope(1.0) == pytest.approx(0.7906, abs=1e-3)
    synthetic.write_trace(tmp_path, tasks=("cat_dog",), seeds=(7,), n_images=1200, n_val=400)
    units = [trace.unit((f, 20)) for f in raw.list_units(tmp_path, raw.TRACE_RE)]
    rows = trace_left.rows(units)
    assert len(rows) == 8
    for r in rows:
        assert r["left_to_the_frame"] == pytest.approx(r["frame_own_slope"], abs=0.05)


def test_the_turing_of_the_pair_in_three_factors(tmp_path):
    """The equation is the theorem divided by the judge alone; a blind machine is worth sqrt(1 + B)."""
    from cognitive_resonance import chart
    for d_h, d_m, r, b in ((2.9, 1.9, 0.5, 0.7), (2.0, 3.0, 0.8, 0.2), (1.2, 2.5, -0.3, 1.5)):
        f = chart.pair_factors(d_h, d_m, r, b)
        alone = d_h / math.sqrt(1 + b)
        assert f["tu_partner"] == pytest.approx(chart.with_partner(d_h, d_m, r) / alone, rel=1e-12)
        assert f["tu_soloist"] == pytest.approx(chart.with_soloist(d_h, d_m, r, b) / alone, rel=1e-12)
        assert f["tu_blind"] == pytest.approx(math.sqrt(1 + b)) and f["tu_partner"] >= f["tu_soloist"]
        assert f["judge_adds_to_machine"] == pytest.approx(chart.with_partner(d_h, d_m, r) / d_m - 1, rel=1e-12)
    same = chart.pair_factors(2.0, 2.0, -0.5, 0.0)          # two equal judges, opposite errors: the peak of the curve
    assert same["tu_partner"] == pytest.approx(1 / math.cos(math.radians(120) / 2), rel=1e-9)
    assert chart.pair_factors(2.0, 2.0, 0.3, 3.0)["angle_blind"] == pytest.approx(150.0, abs=1e-6)
    import factors
    synthetic.write_chart(tmp_path, tasks=("cat_dog",), seeds=(2,), n_images=600, n_val=300)
    u = factors.chart_unit(tmp_path / "cat_dog_v25_s2.npz")
    assert u["bench"] == "chart" and u["measured"]["tu_partner"] == pytest.approx(u["factors"]["tu_partner"], rel=0.1)
    assert u["measured"]["tu_blind"] == pytest.approx(u["factors"]["tu_blind"], rel=0.1)
    assert u["factors"]["together"] == pytest.approx(u["factors"]["strength"] + u["factors"]["private"])
    assert u["measured"]["judge_without_bias_over_judge"] > 1 and 0.5 < u["measured"]["accuracy_without_bias_plus_soloist"] <= 1
    part = chart.pair_factors(2.0, 1.2, 0.6, 0.5)           # the machine sees a part of what the judge sees: r = q
    assert part["together"] == pytest.approx(1.0) and part["tu_partner"] == pytest.approx(part["tu_blind"])
    assert chart.pair_factors(2.0, 3.0, 2.0 / 3.0, 0.5)["private"] == pytest.approx(0.0, abs=1e-12)   # r = 1 / q
    assert chart.pair_factors(2.0, 3.0, 0.0, 0.5)["together"] == pytest.approx(1 + 1.5 ** 2)
    for q in (0.3, 1.0, 2.5):                               # independent errors: what one adds, the other loses
        g = chart.pair_factors(1.0, q, 0.0, 0.4)
        assert (g["tu_without_bias"] ** 2 - 1) * (g["yield_on_machine"] ** 2 - 1) == pytest.approx(1.0)
        assert g["exchange"] == pytest.approx(1.0)
    for q, r in ((0.7, 0.5), (1.0, -0.5), (2.0, 0.3)):
        g = chart.pair_factors(1.0, q, r, 0.0)
        assert g["exchange"] == pytest.approx(((1 + r * r - r * (q + 1 / q)) / (1 - r * r)) ** 2)
        assert g["tu_partner"] == pytest.approx(g["tu_without_bias"])
    assert chart.pair_factors(1.0, 1.0, -0.5, 0.0)["yield_on_machine"] == pytest.approx(2.0)
    people = tmp_path / "people.json"
    people.write_text(json.dumps({"comparisons": [
        {"source": "x", "task": "t", "machine": "m", "mode": "letter", "d_h": 1.0, "d_m": 2.0, "rho": 0.0, "k": 2.0},
        {"source": "x", "task": "t", "machine": "n", "mode": "letter", "d_h": 2.0, "d_m": 1.0, "rho": 0.5, "k": 0.5}]}))
    read = factors.people(people)
    assert read["comparisons"][0]["exchange"] == pytest.approx(1.0) and read["comparisons"][0]["tu"] == pytest.approx(math.sqrt(5))
    assert read["comparisons"][1]["tu"] == pytest.approx(1.0) and read["by_source"][-1]["comparisons"] == 2   # r = q: nothing to add
    assert factors.people(tmp_path / "missing.json") is None


# --- E13: the readable bias of a person --------------------------------------------------------

def _simulated_person(push: float, n: int = 6000, seed: int = 0, d: float = 1.5, fine: bool = False):
    """A person whose judgment leans towards their previous answer by `push` (in units of the clean noise)."""
    rng = np.random.default_rng(seed)
    stimulus = rng.integers(0, 2, n)
    clean = d * (stimulus - 0.5) + rng.standard_normal(n)
    judgment, previous = np.empty(n), 0.0
    for t in range(n):
        judgment[t] = clean[t] + push * previous
        previous = 1.0 if judgment[t] > 0 else -1.0
    response = (judgment > 0).astype(int)
    confidence = np.abs(judgment) if fine else np.digitize(np.abs(judgment), [0.5, 1.2, 2.0])   # four steps
    return stimulus, response, confidence


def test_the_graded_judgment_orders_the_trials_from_one_sure_answer_to_the_other():
    from cognitive_resonance import readable
    g = readable.graded_judgment(np.array([0, 0, 1, 1]), np.array([3.0, 1.0, 1.0, 3.0]))
    assert np.all(np.diff(g) > 0) and abs(g.sum()) < 1e-12
    a = readable.history(np.array([0, 1, 1]), np.array([-1.0, 0.5, 2.0]))
    assert a.shape == (2, 3) and a[0].tolist() == [-1.0, -1.0, -1.0] and a[1].tolist() == [1.0, 0.5, -0.25]


def test_a_person_without_a_readable_bias_has_nothing_to_give_back():
    from cognitive_resonance import readable
    m = readable.person(*_simulated_person(0.0, seed=1))
    assert abs(m["b"]) < 0.01 and abs(m["gain_personal"] - 1.0) < 0.01


def test_a_planted_push_is_read_and_removing_it_gives_what_the_formula_says():
    from cognitive_resonance import readable
    weak, strong = (readable.person(*_simulated_person(p, seed=2, fine=True)) for p in (0.15, 0.3))
    assert 0.01 < weak["b"] < strong["b"] < 0.15                       # planted: 0.0225 and 0.09
    assert 1.0 < weak["gain_personal"] < strong["gain_personal"]
    for m in (weak, strong):
        predicted = readable.predicted_gain_sq(m["b_first"], m["n_first"])
        assert abs(m["gain_personal"] ** 2 - predicted) < 0.03
        assert abs(m["gain_moments"] ** 2 - predicted) < 0.03
    assert 90 < readable.corrector_angle(weak["b"]) < readable.corrector_angle(strong["b"]) < 180


def test_with_a_scale_of_a_few_steps_the_corrector_gives_more_than_the_formula():
    """The trace also reorders the trials that fell on the same step: a gain the Gaussian count does not have."""
    from cognitive_resonance import readable
    gaps = []
    for seed in range(4):
        m = readable.person(*_simulated_person(0.15, seed=seed))
        gaps.append(m["gain_personal"] ** 2 - readable.predicted_gain_sq(m["b_first"], m["n_first"]))
    assert 0.0 < float(np.mean(gaps)) < 0.08


def test_the_files_of_e13_are_split_by_their_name_and_the_split_does_not_move():
    import importlib.util
    spec = importlib.util.spec_from_file_location("people_bias", ANALYSIS / "people_bias.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert mod.split_of("data_Adler_2018_Expt1.csv") == mod.split_of("DATA_ADLER_2018_EXPT1.CSV")
    parts = [mod.split_of(f"data_{i}.csv") for i in range(400)]
    assert 0.12 < parts.count("exploration") / 400 < 0.26


def test_a_file_of_e13_is_read_with_text_labels_a_byte_order_mark_and_any_case(tmp_path):
    import importlib.util
    spec = importlib.util.spec_from_file_location("people_bias", ANALYSIS / "people_bias.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    rows = ["﻿subj_idx,STIMULUS,Response,confidence"]
    for i in range(60):
        stimulus = "old" if i % 3 else "new"
        wrong = i % 5 == 0
        response = ("new" if stimulus == "old" else "old") if wrong else stimulus
        rows.append(f"s{i % 2},{stimulus},{response},{1 if wrong else 2 + i % 2}")
    rows += ["s0,old,NaN,2", "s0,old,old,", ",old,old,2"]
    path = tmp_path / "data_test.csv"
    path.write_text("\n".join(rows), encoding="utf-8")
    people, status = mod.read_file(path)
    assert status == "read" and sorted(people) == ["s0", "s1"]
    assert sum(p["stimulus"].size for p in people.values()) == 60
    assert set(np.unique(people["s0"]["stimulus"])) == {0, 1}
    path.write_text("Subj_idx,Stimulus,Response,Confidence\n1,1,1,2\n1,2,3,1\n1,3,1,3\n", encoding="utf-8")
    assert mod.read_file(path) == (None, "not a two-answer task")


def test_in_information_the_parts_of_the_pair_add_up_and_the_bias_only_covers():
    from cognitive_resonance import chart
    d_h, d_m, r, b = 2.0, 1.5, 0.4, 0.7
    info, old = chart.pair_information(d_h, d_m, r, b), chart.pair_factors(d_h, d_m, r, b)
    alone = info["person_alone"]
    assert info["person_with_tailored"] / alone == pytest.approx(old["tu_partner"] ** 2)
    assert info["person_with_generic"] / alone == pytest.approx(old["tu_soloist"] ** 2)
    assert info["person_with_corrector"] / alone == pytest.approx(old["tu_blind"] ** 2)
    assert info["person_with_tailored"] - info["person_with_generic"] == pytest.approx(info["covered_by_the_bias"])
    assert info["covered_by_the_bias"] == pytest.approx(b / (1 - r * r + b) * info["private"])
    assert chart.pair_information(d_h, d_m, r, 0.0)["covered_by_the_bias"] == 0.0


def test_opposite_errors_are_a_hidden_common_cause_with_opposite_signs():
    from cognitive_resonance import chart
    rng = np.random.default_rng(0)
    c_h, c_m, s = chart.opposite_errors_as_a_common_cause(-0.6)
    z, n_h, n_m = rng.standard_normal((3, 400_000))
    e_h, e_m = c_h * z + s * n_h, c_m * z + s * n_m
    assert e_h.var() == pytest.approx(1.0, abs=0.01) and e_m.var() == pytest.approx(1.0, abs=0.01)
    assert np.corrcoef(e_h, e_m)[0, 1] == pytest.approx(-0.6, abs=0.01)
    # what is left of e_h once Z is known does not go with anything the machine computes from Z and its own noise
    assert abs(np.corrcoef(e_h - c_h * z, e_m)[0, 1]) < 0.01


def test_a_correction_with_a_ceiling_leaves_the_large_pushes():
    from cognitive_resonance import chart
    rng = np.random.default_rng(1)
    a = rng.standard_normal(1_000_000)
    for s in (0.0, 0.5, 1.0, 2.0):
        left = np.clip(np.abs(a) - s, 0.0, None)
        assert chart.left_under_a_ceiling(s) == pytest.approx(float((left ** 2).mean()), abs=0.005)
    assert chart.left_under_a_ceiling(0.0) == pytest.approx(1.0)
    assert chart.left_by_an_open_warning(0.5) == 0.25


def test_the_shortcut_reading_says_that_what_is_learned_falls_with_the_shortcut():
    path = ANALYSIS.parent / "results" / "shortcut.json"
    if not path.exists():
        pytest.skip("results/shortcut.json not there")
    d = json.loads(path.read_text())
    kept = [r["kept_frame_removed"] for r in d["by_mu"]]
    assert kept[0] == pytest.approx(1.0) and all(x >= y - 0.01 for x, y in zip(kept, kept[1:]))
    assert all(r["kept_frame_removed"] >= r["kept_as_it_is"] - 1e-9 for r in d["by_mu"])


def test_the_answers_are_counted_before_and_after_the_correction():
    from cognitive_resonance import readable
    m = readable.person(*_simulated_person(0.7, seed=3, fine=True))
    assert m["accuracy_after"] > m["accuracy_before"]                   # a large push, a fine scale: answers improve
    assert m["changed"] == pytest.approx(m["changed_to_right"] + m["changed_to_wrong"])
    assert m["accuracy_after"] - m["accuracy_before"] == pytest.approx(m["changed_to_right"] - m["changed_to_wrong"])
    still = readable.person(*_simulated_person(0.0, seed=3))
    assert abs(still["accuracy_after"] - still["accuracy_before"]) < 0.01


def test_the_slopes_pulled_towards_the_group_go_from_the_person_to_the_group():
    from cognitive_resonance import readable
    a, b = (readable.person(*_simulated_person(p, seed=s)) for p, s in ((0.3, 4), (0.6, 5)))
    own, (ox, oy), n = a["_moments_first"], b["_moments_first"], b["n_first"]
    personal, group = readable.slopes(*own), readable.slopes(ox, oy)
    assert np.allclose(readable.shrunk_slopes(own, ox, oy, n, 0.0), personal)
    assert np.allclose(readable.shrunk_slopes(own, ox, oy, n, 1e9), group, atol=1e-4)
    middle = readable.shrunk_slopes(own, ox, oy, n, float(a["n_first"]))
    assert np.linalg.norm(middle - group) < np.linalg.norm(personal - group)


def test_the_second_implementation_of_e13_gives_the_same_numbers():
    first, second = (ANALYSIS.parent / "results" / n for n in ("e13_people_bias.json", "e13_second_implementation.json"))
    if not (first.exists() and second.exists()):
        pytest.skip("results of E13 not there")
    a, b = json.loads(first.read_text()), json.loads(second.read_text())
    for part, short in (("confirmation", "conf"), ("exploration", "expl")):
        assert a[part]["people"] == b["counts"][short]["people"] and a[part]["files"] == b["counts"][short]["files_included"]
        assert a[part]["b"]["median"] == pytest.approx(b[short]["median_B"], abs=1e-9)
        assert a[part]["gain_personal"]["median"] == pytest.approx(b[short]["median_G_personal_auc"], abs=1e-4)
        assert a[part]["gain_everyone"]["median"] == pytest.approx(b[short]["median_G_everyone"], abs=1e-4)
        assert a[part]["gain_moments"]["median"] == pytest.approx(b[short]["median_G_personal_moments"], abs=1e-3)


def test_the_curve_of_personalisation_has_the_personal_corrector_below_the_group_with_few_trials():
    path = ANALYSIS.parent / "results" / "e13_people_more.json"
    if not path.exists():
        pytest.skip("results/e13_people_more.json not there")
    rows = json.loads(path.read_text())["confirmation"]["curve_people_with_800"]
    personal = [r["personal"]["median"] for r in rows]
    assert personal[0] < 1.0 < personal[-1] and all(a <= b + 1e-3 for a, b in zip(personal, personal[1:]))
    assert rows[0]["personal"]["median"] < rows[0]["group"]["median"]


def test_a_push_on_the_doubtful_cases_is_given_back_only_in_part_by_a_linear_corrector():
    path = ANALYSIS.parent / "results" / "nonadditive.json"
    if not path.exists():
        pytest.skip("results/nonadditive.json not there")
    rows = json.loads(path.read_text())["rows"]
    for beta in {r["beta"] for r in rows}:
        back = [r["linear_gives_back"] for r in rows if r["beta"] == beta]        # additive first, then tighter and tighter
        assert back[0] == pytest.approx(1.0, abs=0.02) and all(a > b for a, b in zip(back, back[1:]))


def _opinion_data(n, beta, tau, seed=0):
    """A judge with a clean cue of sensitivity 1.5 and a push beta a k(g0): the same for every question, or only on the doubtful ones."""
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, n)
    g0 = 1.5 * (y - 0.5) + rng.standard_normal(n)
    k = np.ones(n) if tau is None else np.exp(-g0 ** 2 / (2 * tau ** 2))
    a = rng.choice([-1.0, 1.0], n)
    return y, g0, g0 + beta * k, g0 - beta * k, a


def test_an_opinion_that_only_adds_is_given_back_whole_by_a_linear_corrector():
    from cognitive_resonance import opinion
    y, g0, ga, gb, a = _opinion_data(40000, 1.0, None)
    beta, b = opinion.readable_bias(opinion.met(ga, gb, a), a, y)
    assert beta == pytest.approx(1.0, abs=0.02) and b == pytest.approx(1.0, abs=0.05)
    m = opinion.measure(y, g0, ga, gb, a, np.arange(20000), np.arange(20000, 40000))
    assert m["kept_as_it_is"] == pytest.approx(1 / np.sqrt(2), abs=0.03) and m["gives_back"] == pytest.approx(1.0, abs=0.03)
    assert opinion.push(g0, ga, gb)["most_doubtful_over_clearest"] == pytest.approx(1.0, abs=1e-9)


def test_an_opinion_that_weighs_on_the_doubtful_questions_is_given_back_in_part():
    from cognitive_resonance import opinion
    y, g0, ga, gb, a = _opinion_data(40000, 3.0, 0.7)
    m = opinion.measure(y, g0, ga, gb, a, np.arange(20000), np.arange(20000, 40000))
    assert m["kept_as_it_is"] < m["kept_corrected"] < 0.97 and 0.05 < m["gives_back"] < 0.9
    assert opinion.push(g0, ga, gb)["most_doubtful_over_clearest"] > 2


def test_the_best_threshold_finds_a_criterion_that_was_moved():
    from cognitive_resonance import opinion
    rng = np.random.default_rng(3)
    y = rng.integers(0, 2, 20000)
    g = 2.0 * (y - 0.5) + rng.standard_normal(20000) + 0.8
    cut = opinion.best_threshold(g, y)
    assert cut == pytest.approx(0.8, abs=0.15) and opinion.right(g, y, cut) > opinion.right(g, y) + 0.02


def _result(name):
    path = ANALYSIS.parent / "results" / name
    if not path.exists():
        pytest.skip(f"results/{name} not there")
    return json.loads(path.read_text())


def test_the_threshold_is_crossed_once_and_where_the_proposition_puts_it():
    d = _result("e15_crossing.json")
    gaps = [abs(b["crossing_measured"] - b["crossing_predicted"]) for b in d["by_task"]
            if b["crossing_measured"] is not None and b["crossing_predicted"] is not None]
    assert len(gaps) >= 4 and max(gaps) <= 3.0
    for b in d["by_task"]:                                   # the generic machine overtakes the corrector as it sees more
        assert b["measured"][0] < 0 < b["measured"][-1]


def test_a_new_last_layer_and_a_longer_training_give_back_part_of_what_a_strong_shortcut_left():
    d = _result("e14_unlearned.json")
    row = {(r["epochs"], r["mu"]): r for r in d["table"]}
    short, long = min(d["lengths"]), max(d["lengths"])
    top = max(d["mus"])
    assert row[(short, top)]["kept_last"] > row[(short, top)]["kept_out"]          # learned and not used
    assert row[(long, top)]["kept_best"] > row[(short, top)]["kept_best"]          # learned later
    assert row[(long, top)]["kept_best"] < 0.95                                    # and something is still missing
    assert row[(short, 0.0)]["kept_out"] == pytest.approx(1.0)                     # the reference is the network itself


def test_the_language_model_follows_the_opinion_and_a_linear_corrector_gives_back_part():
    d = _result("e16_llm_judge.json")
    assert d["judges"]["qwen2.5-1.5b-instruct"]["b"] > 0.2                       # the primary judge; a larger one gives way less
    for j in d["judges"].values():
        assert j["b"] > 0 and j["kept_as_it_is"] < j["kept_corrected"] < 1.0
        assert j["right_as_it_is"] < j["right_corrected"] < j["right_clean"]
        assert j["right_when_opinion_is_wrong"] < j["right_clean"] < j["right_when_opinion_is_right"]
        assert j["push"]["mean_by_third_of_doubt"][0] > j["push"]["mean_by_third_of_doubt"][2]


def test_the_corrector_beats_the_generic_machine_inside_a_band_and_not_below_it():
    from cognitive_resonance import chart
    lo, hi = chart.blind_band(0.31, 0.015)                      # noises that resemble each other more than the bias weighs
    assert lo == pytest.approx(0.190, abs=2e-3) and hi == pytest.approx(0.421, abs=2e-3)
    d1 = lambda x: chart.pair_information(1.0, x, 0.31, 0.015)["person_with_generic"]                    # noqa: E731
    assert d1(0.05) > 1.0 > d1(0.30) and d1(0.50) > 1.0          # a very weak machine wins again: a reference for the shared noise
    assert chart.blind_band(0.0, 1.0) == pytest.approx((0.0, 1 / np.sqrt(2)))                     # B above r^2: only the upper edge
    assert chart.blind_band(-0.3, 0.05)[1] < 0                   # opposite noises and a small bias: the corrector never wins


def test_a_bias_the_two_share_cannot_be_cancelled_by_the_pair():
    from cognitive_resonance import chart
    d_h, d_m, r, beta = 2.0, 1.5, 0.3, 0.8
    for gamma in (0.0, 0.8, -0.8):
        sigma = np.array([[1 + beta ** 2, r + beta * gamma], [r + beta * gamma, 1 + gamma ** 2]])
        delta = np.array([d_h, d_m])
        assert chart.with_shared_bias(d_h, d_m, r, beta, gamma) == pytest.approx(float(delta @ np.linalg.solve(sigma, delta)))
    private = chart.with_shared_bias(d_h, d_m, r, beta, 0.0)
    assert private == pytest.approx(chart.pair_information(d_h, d_m, r, beta ** 2)["person_with_generic"])
    assert chart.with_shared_bias(d_h, d_m, r, beta, 0.8) < private < chart.pair_information(d_h, d_m, r, beta ** 2)["person_with_tailored"]
    assert chart.angle_with_shared_bias(0.0, 1.0, 1.0) == pytest.approx(60.0)                      # the same push closes the angle
    assert chart.with_shared_bias(1.0, 1.0, 0.0, 1.0, 1.0) == pytest.approx(2 / 3)                # two judges with the same bias: less than one without


def test_caution_under_unequal_costs_is_not_an_error_and_removing_it_costs():
    from cognitive_resonance import chart
    lam = chart.criterion_for_costs(5.0, 1.0)
    assert lam == pytest.approx(-np.log(5.0))
    grid = np.linspace(-4, 4, 801)
    costs = [chart.decisions_at(2.0, x, 5.0, 1.0)["cost"] for x in grid]
    assert grid[int(np.argmin(costs))] == pytest.approx(lam, abs=0.02)                             # the criterion is the best one
    cautious, corrected = chart.decisions_at(2.0, lam, 5.0, 1.0), chart.decisions_at(2.0, 0.0, 5.0, 1.0)
    assert corrected["right"] > cautious["right"] + 0.05 and corrected["cost"] > 1.5 * cautious["cost"]


def test_a_corrector_that_knows_the_judgment_follows_a_push_that_weighs_on_the_doubt():
    from cognitive_resonance import opinion
    from cognitive_resonance.estimators import d_a
    y, g0, ga, gb, a = _opinion_data(40000, 3.0, 0.7)
    g = opinion.met(ga, gb, a)
    learn, test = np.arange(20000), np.arange(20000, 40000)
    beta, _ = opinion.readable_bias(g[learn], a[learn], y[learn])
    state = opinion.state_corrector(g, a, y, learn)
    assert d_a(state[test], y[test]) > d_a((g - beta * a)[test], y[test]) > d_a(g[test], y[test])
    shape = opinion.shape_of_the_push(g0, ga, gb, y, 0.0)
    assert shape["tau"] == pytest.approx(0.7, rel=0.1) and shape["variance_explained_by_the_bell"] > 0.99
