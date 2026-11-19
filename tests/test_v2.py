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
