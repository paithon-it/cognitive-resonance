"""The analysis scripts on small synthetic raw files: they run, and every criterion has an outcome."""
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from cognitive_resonance import synthetic

ANALYSIS = Path(__file__).resolve().parents[1] / "analysis"
sys.path.insert(0, str(ANALYSIS))

import bench  # noqa: E402
import null  # noqa: E402
import people  # noqa: E402
import rephase  # noqa: E402


@pytest.fixture(scope="module")
def bench_files(tmp_path_factory):
    d = tmp_path_factory.mktemp("bench")
    synthetic.write_bench(d, tasks=("cat_dog", "deer_horse"), mus=(0.0, 1.0, 2.0), seeds=(0, 1),
                          n_images=300, n_val=300)
    return sorted(d.glob("*.npz"))


def test_bench_criteria(bench_files):
    units = [bench.unit((f, 40, False)) for f in bench_files]
    c = bench.criteria(units)
    assert set(c) == {"H1a", "H1b", "H1c", "H1d", "H1e", "H1f"}
    assert all(v["outcome"] in ("pass", "fail", "not evaluable") for v in c.values())
    top = [u for u in units if u["mu"] == 2.0]
    assert all(u["pairs"]["opposite"]["phi"] > 90 for u in top)        # the toy opens the angle
    assert all(u["judges"]["plus"]["slope_on_a"] > 0 > u["judges"]["minus"]["slope_on_a"] for u in top)
    rows = bench.by_mu(units)
    assert set(rows["opposite"]) == {"0", "1", "2"}


def test_null_criteria(bench_files):
    units = [null.unit((f, 40)) for f in bench_files]
    c = null.criteria(units)
    assert set(c) == {"H5a", "H5b", "H5c", "H5d", "H5e"}
    pos = [u["positive"] for u in units if "positive" in u]
    assert pos and all(p["machine_slope_on_a"] < 0 for p in pos)
    assert any(p.get("d_m", 0) < 0 for p in pos)                     # frame against it: below chance


def test_rephase_criteria(tmp_path):
    synthetic.write_rephase(tmp_path, tasks=("cat_dog", "deer_horse"), seeds=(0, 1), n_images=300, n_val=300)
    units = [rephase.unit((f, 40, False)) for f in sorted(tmp_path.glob("*.npz"))]
    c = rephase.criteria(units)
    assert set(c) == {"H2a", "H2b", "H2c"}
    assert units[0]["differences"]["M1"]["solo_ci"] is not None
    assert "control_H2" in rephase.by_lambda(units)


def test_people_helpers():
    y, other, hard, counts = synthetic.people_world()
    n = counts.sum(1).astype(float)
    pc = people.plurality_correct(counts, y)
    assert 0.9 < pc.mean() <= 1.0
    gen = np.random.default_rng(0)
    lg = gen.normal(size=(y.size, 10))
    lg[np.arange(y.size), y] += 3.0
    X = people.features(counts.astype(float), n, lg)
    ok, coefs = people.combine_cv(X, y)
    assert ok.mean() >= pc.mean() - 0.01 and len(coefs) == people.CV_FOLDS
    with pytest.raises(SystemExit):
        people.labels([{"image": np.arange(3), "y": np.zeros(3, int)}], [], size=4)


@pytest.mark.skipif(not os.environ.get("TURING_METER"), reason="set TURING_METER to a turing-meter clone")
def test_people_end_to_end(tmp_path):
    synthetic.write_people(tmp_path / "people", tmp_path / "counts.npy")
    out = tmp_path / "e3.json"
    subprocess.run([sys.executable, str(ANALYSIS / "people.py"), "--raw", str(tmp_path / "people"),
                    "--counts", str(tmp_path / "counts.npy"), "--turing-meter", os.environ["TURING_METER"],
                    "--out", str(out), "--reps", "30"], check=True)
    r = json.loads(out.read_text())
    assert set(r["criteria"]) == {"H3a", "H3b"}
    assert r["turing_meter"]["commit"]


def test_rephase_bis_criteria(tmp_path):
    import rephase_bis
    synthetic.write_rephase_bis(tmp_path, tasks=("cat_dog", "deer_horse", "bird_frog"), seeds=(0, 1),
                                n_images=400, n_val=400)
    units = [rephase_bis.unit((f, 60)) for f in sorted(tmp_path.glob("*.npz"))]
    c = rephase_bis.criteria(units)
    assert set(c) == {"B1", "B2", "B3", "B4", "B5"}
    assert all(v["outcome"] in ("pass", "fail") for v in c.values())
    u = units[0]
    assert u["judges"]["H"]["slope_on_a"] > 0 > u["judges"]["Mframe"]["slope_on_a"]
    assert u["differences"]["Mpair"]["phi_ci"] is not None
    # the toy is built so that the frame-only machine is at chance and helps as 1/sin(phi) says
    assert c["B1"]["mframe_contains_zero"] >= 4
    assert c["B4"]["median_ratio_tuned_over_inv_sin_phi"] == pytest.approx(1.0, abs=0.1)
    rows = rephase_bis.by_machine(units)
    assert set(rows) == {"H", "M0", "Mpair", "Mframe"}


def test_tu_app_nonneg_any_k():
    import rephase_bis
    from cognitive_resonance import model
    for k, rho in ((-0.8, -0.95), (0.0, -0.5), (0.9, -0.3)):
        grid = np.linspace(0, 50, 200001)
        best = max(1.0, float(np.max(model.tu_signed(grid, k, rho))))
        assert rephase_bis.tu_app_nonneg(k, rho) == pytest.approx(best, rel=1e-6)
    assert rephase_bis.tu_app_nonneg(-0.5, 0.3) == 1.0


def test_exploratory_null_positive_weight(tmp_path):
    synthetic.write_rephase(tmp_path, tasks=("cat_dog",), seeds=(0,), n_images=400, n_val=400)
    f = tmp_path / "cat_dog_s0.npz"
    with np.load(f, allow_pickle=False) as z:
        arrays = {k: z[k] for k in z.files}
    for key in ("z_M", "val_z_M"):                     # machines with lambda > 0 reversed, as in the real E2
        arrays[key][1:] = -arrays[key][1:]
    np.savez(f, **arrays)
    units = [rephase.unit((f, 40, False))]
    e = rephase.exploratory_null(units, tmp_path, workers=1)
    assert e["exploratory"] and e["pairs"] == 4
    assert e["crossing_at_positive_weight"] == 4
    assert e["median_ratio_w0_over_predicted"] == pytest.approx(1.0, abs=0.1)
    assert e["max_abs_difference_tu_app_reversed"] < 1e-9


# --- E2 ter --------------------------------------------------------------------------------------------

def test_rephase_ter_on_synthetic_files(tmp_path):
    import rephase_ter
    synthetic.write_rephase_ter(tmp_path, tasks=("cat_dog", "deer_horse", "bird_frog"), seeds=(0, 1),
                                n_images=400, n_val=400)
    units = [rephase_ter.unit((f, 60)) for f in sorted(tmp_path.glob("*.npz"))]
    c = rephase_ter.criteria(units)
    assert set(c) == {"T1", "T2 Mdue", "T2 Mramo", "T3 Mdue", "T3 Mramo", "T4", "T5"}
    assert all(v["outcome"] in ("pass", "fail") for v in c.values())
    u = units[0]
    assert set(u["differences"]) == {"M0", "Mdue", "Mramo"}                 # all from the control, M0plus
    assert set(u["differences_from_M0"]) == {"M0plus", "Mdue", "Mramo"}
    assert u["differences"]["Mdue"]["phi_ci"] is not None
    alone = u["pair_against_control_alone"]["Mdue"]
    assert alone["pair_minus_control_alone"] == pytest.approx(
        u["pairs"]["Mdue"]["d_pair_tuned"] - u["judges"]["M0plus"]["d_a"])
    # the toy: both machines learn the frame against H, and the angle opens from the control
    assert u["judges"]["H"]["slope_on_a"] > 0 > u["judges"]["Mramo"]["slope_on_a"]
    assert c["T2 Mdue"]["outcome"] == c["T2 Mramo"]["outcome"] == "pass"
    assert c["T1"]["slope_opposite_to_h"] == len(units)
    rows = rephase_ter.summary_by_machine(units)
    assert set(rows) == {"H", "M0", "M0plus", "Mdue", "Mramo"}
    assert "difference_from_M0plus" in rows["Mdue"] and "difference_from_M0" in rows["M0plus"]


def _ter_unit(task, seed, *, phi_lo=(1.0, 1.0), pair_lo=(0.1, 0.1), alone_lo=0.1, ratio=1.0, slope_due=-0.5,
              tu_ratio=(1.0, 1.0)):
    """A unit of E2 ter reduced to what the criteria read; the lower ends of the intervals are set."""
    def diff(lo_phi, lo_pair):
        return {"phi": lo_phi + 1, "phi_ci": [lo_phi, lo_phi + 2], "pair": lo_pair + 0.1, "pair_ci": [lo_pair, 1.0],
                "solo": 0.0, "solo_ci": [-1.0, 1.0]}
    judges = {"H": {"d_a": 1.5, "slope_on_a": 0.7}, "M0plus": {"d_a": 2.0, "slope_on_a": 0.0},
              "Mdue": {"d_a": 2.0 * ratio, "slope_on_a": slope_due}, "Mramo": {"d_a": 1.0, "slope_on_a": -0.6},
              "M0": {"d_a": 1.9, "slope_on_a": 0.0}}
    pairs = {m: {"ratio_tuned_app_nonneg": r, "tu_tuned": 1.5, "tu_app_nonneg": 1.5 / r, "d_pair_tuned": 2.2}
             for m, r in zip(("Mdue", "Mramo"), tu_ratio)}
    pairs["M0plus"] = {"d_pair_tuned": 2.1}
    alone = {m: {"pair_minus_control_alone": alone_lo + 0.1, "pair_minus_control_alone_ci": [alone_lo, 1.0]}
             for m in ("Mdue", "Mramo")}
    return {"task": task, "seed": seed, "judges": judges, "pairs": pairs,
            "differences": {"Mdue": diff(phi_lo[0], pair_lo[0]), "Mramo": diff(phi_lo[1], pair_lo[1])},
            "pair_against_control_alone": alone}


def test_rephase_ter_verdicts_on_constructed_units():
    import rephase_ter
    tasks = ("t1", "t2", "t3", "t4", "t5")
    good = [_ter_unit(t, s) for t in tasks for s in (0, 1)]
    c = rephase_ter.criteria(good)
    assert all(v["outcome"] == "pass" for v in c.values())
    # T2 and T3 are judged per machine; one seed with the interval touching zero takes its task out
    units = [_ter_unit(t, s, phi_lo=(-0.1 if (t in tasks[:3] and s == 1) else 1.0, 1.0))
             for t in tasks for s in (0, 1)]
    c = rephase_ter.criteria(units)
    assert c["T2 Mdue"]["outcome"] == "fail" and c["T2 Mdue"]["tasks_passing"] == 2
    assert c["T2 Mramo"]["outcome"] == "pass"
    units = [_ter_unit(t, s, pair_lo=(0.1, 0.0 if t in tasks[:2] else 0.1)) for t in tasks for s in (0, 1)]
    c = rephase_ter.criteria(units)
    assert c["T3 Mdue"]["outcome"] == "pass" and c["T3 Mramo"]["outcome"] == "pass"   # 3 tasks of 5 are enough
    units = [_ter_unit(t, s, pair_lo=(0.1, 0.0 if t in tasks[:3] else 0.1)) for t in tasks for s in (0, 1)]
    assert rephase_ter.criteria(units)["T3 Mramo"]["outcome"] == "fail"
    # T1: the median ratio, and the slope in every unit
    units = [_ter_unit(t, s, ratio=0.85 if t in tasks[:3] else 1.0) for t in tasks for s in (0, 1)]
    assert rephase_ter.criteria(units)["T1"]["outcome"] == "fail"
    units = [_ter_unit(t, s, ratio=0.85 if t in tasks[:2] else 0.95) for t in tasks for s in (0, 1)]
    assert rephase_ter.criteria(units)["T1"]["outcome"] == "pass"
    units = good[:-1] + [_ter_unit("t5", 1, slope_due=0.2)]
    assert rephase_ter.criteria(units)["T1"]["outcome"] == "fail"
    # T4 needs both machines; T5 looks at Mdue only
    units = [_ter_unit(t, s, tu_ratio=(1.0, 1.2)) for t in tasks for s in (0, 1)]
    assert rephase_ter.criteria(units)["T4"]["outcome"] == "fail"
    units = [_ter_unit(t, s, alone_lo=-0.05 if t in tasks[:3] else 0.1) for t in tasks for s in (0, 1)]
    assert rephase_ter.criteria(units)["T5"]["outcome"] == "fail"


# --- E3 bis --------------------------------------------------------------------------------------------

def test_people_bis_cue():
    import people_bis
    counts = np.array([[50, 0, 0, 0, 0, 0, 0, 0, 0, 1], [5] * 10, [0] * 10])
    u = people_bis.people_cue(counts)
    assert u.shape == (3, 10)
    assert np.exp(u).sum(1) == pytest.approx(np.ones(3))                  # a log probability over ten classes
    assert u[0, 0] == pytest.approx(np.log(50.5 / 56)) and u[0, 1] == pytest.approx(np.log(0.5 / 56))
    assert np.allclose(u[1], np.log(0.1)) and np.allclose(u[2], np.log(0.1))
    # argmax(u + z): the people decide where the machine is unsure, the machine where people have no votes
    y = np.array([0, 3, 7])
    lg = np.zeros((3, 10))
    lg[0, 9] = 1.0
    lg[1, 3] = 1.0
    lg[2, 7] = 0.5
    assert people_bis.sum_correct(counts, lg, y).tolist() == [1.0, 1.0, 1.0]


def test_people_bis_out_of_fold_order():
    import people_bis
    y = np.arange(10) % 10
    lg = np.random.default_rng(0).normal(size=(3, 5, 10))
    folds = [{"image": np.arange(5), "logits": lg, "recipe_names": ["mista", "controllo", "somma"]},
             {"image": np.arange(5, 10), "logits": lg, "recipe_names": ["controllo", "somma", "mista"]}]
    oof, covered = people_bis.out_of_fold(folds, y.size)
    assert covered.all() and oof.shape == (3, 10, 10)
    assert np.array_equal(oof[0, :5], lg[1]) and np.array_equal(oof[2, :5], lg[0])      # rows by name
    assert np.array_equal(oof[1, 5:], lg[1])


def test_people_bis_paired_group():
    import people_bis
    runs = {"controllo": [{"turing_group": 1.0}, {"turing_group": 1.1}, {"turing_group": 0.9}],
            "mista": [{"turing_group": 1.2}, {"turing_group": 1.2}, {"turing_group": 1.1}]}
    points = {"controllo": {"turing_group": 1.0}, "mista": {"turing_group": 1.15}}
    g = people_bis.paired_group(points, runs)
    assert set(g) == {"mista"} and g["mista"]["turing_group"] == pytest.approx(0.15)
    assert g["mista"]["turing_group_ci"] == pytest.approx([0.105, 0.2])          # draws 0.2, 0.1, 0.2


def _bis_seed(*, acc=None, phi=None, tu=None, group=None, angles=None):
    """A seed of E3 bis reduced to what the criteria read: lower ends of the paired intervals are set."""
    acc = acc or {"fold0": (0.94, 0.90, 0.93), "fold1": (0.94, 0.90, 0.93)}
    phi, tu, group = phi or (1.0, 1.0), tu or (0.01, 0.01), group or (0.01, 0.01)
    angles = angles or (68.0, 75.0, 72.0)
    names = ("controllo", "somma", "mista")
    paired = {r: {"phi": phi[i] + 1, "phi_ci": [phi[i], phi[i] + 2], "tu_app": tu[i] + 0.01,
                  "tu_app_ci": [tu[i], tu[i] + 0.02], "turing_group": group[i] + 0.01,
                  "turing_group_ci": [group[i], group[i] + 0.02]} for i, r in enumerate(names[1:])}
    return {"accuracy_by_fold": {f: dict(zip(names, a)) for f, a in acc.items()},
            "recipes": {r: {"angle": {} if angles[i] is None else {"phi": angles[i]}} for i, r in enumerate(names)},
            "paired": paired}


def test_people_bis_verdicts_on_constructed_seeds():
    import people_bis
    c = people_bis.criteria({"s0": _bis_seed(), "s1": _bis_seed()})
    assert set(c) == {"P1", "P2", "P3", "P4", "P5"}
    assert all(v["outcome"] == "pass" for v in c.values())
    assert c["P2"]["recipes_opening_in_every_seed"] == ["somma", "mista"]
    # P1: exactly 2 points below the control passes, a little more does not; somma is not judged
    edge = {"fold0": (0.95, 0.80, 0.93), "fold1": (0.94, 0.80, 0.92)}
    assert people_bis.criteria({"s0": _bis_seed(acc=edge)})["P1"]["outcome"] == "pass"
    over = {"fold0": (0.95, 0.95, 0.9298), "fold1": (0.94, 0.94, 0.94)}
    assert people_bis.criteria({"s0": _bis_seed(), "s1": _bis_seed(acc=over)})["P1"]["outcome"] == "fail"
    one_half = {"fold0": (0.95, 0.95, 0.95)}
    assert people_bis.criteria({"s0": _bis_seed(acc=one_half)})["P1"]["outcome"] == "not evaluable"
    # P2: the same recipe in every seed
    c = people_bis.criteria({"s0": _bis_seed(phi=(1.0, -1.0)), "s1": _bis_seed(phi=(-1.0, 1.0))})
    assert c["P2"]["outcome"] == "fail" and c["P3"]["outcome"] == "fail"
    # P3: the apparent turings must rise for a recipe that passes P2, not for another one
    c = people_bis.criteria({"s0": _bis_seed(phi=(1.0, -1.0), tu=(-0.01, 0.01)),
                             "s1": _bis_seed(phi=(1.0, -1.0), tu=(-0.01, 0.01))})
    assert c["P2"]["recipes_opening_in_every_seed"] == ["somma"]
    assert c["P3"]["recipes_raising_tu_app_in_every_seed"] == ["mista"]
    assert c["P3"]["outcome"] == "fail"
    # P4: point estimates of somma and mista; an angle that cannot be measured leaves it open
    assert people_bis.criteria({"s0": _bis_seed(angles=(95.0, 80.0, 85.0))})["P4"]["outcome"] == "pass"
    assert people_bis.criteria({"s0": _bis_seed(angles=(60.0, 90.5, 85.0))})["P4"]["outcome"] == "fail"
    assert people_bis.criteria({"s0": _bis_seed(angles=(60.0, None, 85.0))})["P4"]["outcome"] == "not evaluable"
    assert people_bis.criteria({"s0": _bis_seed(angles=(60.0, None, 95.0))})["P4"]["outcome"] == "fail"
    # P5: mista only, in every seed
    c = people_bis.criteria({"s0": _bis_seed(group=(-0.1, 0.01)), "s1": _bis_seed(group=(-0.1, 0.0))})
    assert c["P5"]["outcome"] == "fail"
    assert people_bis.criteria({"s0": _bis_seed(group=(-0.1, 0.01))})["P5"]["outcome"] == "pass"
    assert people_bis.criteria({"s0": {"base": {}}})["P1"]["outcome"] == "not evaluable"


@pytest.mark.skipif(not os.environ.get("TURING_METER"), reason="set TURING_METER to a turing-meter clone")
def test_people_bis_end_to_end(tmp_path):
    synthetic.write_people_bis(tmp_path / "people_bis", tmp_path / "counts.npy")
    out = tmp_path / "e3bis.json"
    subprocess.run([sys.executable, str(ANALYSIS / "people_bis.py"), "--raw", str(tmp_path / "people_bis"),
                    "--counts", str(tmp_path / "counts.npy"), "--turing-meter", os.environ["TURING_METER"],
                    "--out", str(out), "--reps", "30"], check=True)
    r = json.loads(out.read_text())
    assert set(r["criteria"]) == {"P1", "P2", "P3", "P4", "P5"}
    for s in r["seeds"].values():
        rec = s["recipes"]
        for nm in ("somma", "mista"):                 # the paired step reproduces compare_votes on the whole data
            assert s["paired"][nm]["tu_app"] == pytest.approx(rec[nm]["angle"]["tu_app"]
                                                              - rec["controllo"]["angle"]["tu_app"])
            assert s["paired"][nm]["phi"] == pytest.approx(rec[nm]["angle"]["phi"] - rec["controllo"]["angle"]["phi"])
            assert s["paired"][nm]["turing_group"] == pytest.approx(rec[nm]["group"]["turing_group"]
                                                                    - rec["controllo"]["group"]["turing_group"])
        assert rec["controllo"]["accuracy_majority"] == pytest.approx(r["cifar10h"]["plurality_accuracy"])
