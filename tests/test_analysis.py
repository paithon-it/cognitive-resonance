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
