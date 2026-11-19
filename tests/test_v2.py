"""E6 and E7 (protocol §§ 12-13): the readers, the measures at a fixed weight, and the two analyses on
synthetic files whose answer is known."""
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
