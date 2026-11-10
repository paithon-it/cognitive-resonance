import numpy as np
import pytest

from cognitive_resonance import raw, synthetic


def test_bench_reader(tmp_path):
    synthetic.write_bench(tmp_path, tasks=("cat_dog",), mus=(0.0, 1.5), seeds=(0,), n_images=40, n_val=20)
    files = raw.list_units(tmp_path, raw.BENCH_RE)
    assert [f.name for f in files] == ["cat_dog_mu0_s0.npz", "cat_dog_mu1.5_s0.npz"]
    d = raw.read_bench(files[1])
    assert d["task"] == "cat_dog" and d["mu"] == 1.5 and d["seed"] == 0
    assert d["z_plus"].shape == (200,) and d["val_z_minus"].shape == (20,)
    assert raw.read_manifest(tmp_path) == {"synthetic": True}


def test_bench_reader_refuses_broken_files(tmp_path):
    p = tmp_path / "cat_dog_mu1_s0.npz"
    np.savez(p, y=np.array([0, 1]), image=np.array([0, 1]))
    with pytest.raises(raw.ContractError):
        raw.read_bench(p)
    q = tmp_path / "wrongname.npz"
    np.savez(q, y=np.array([0, 1]))
    with pytest.raises(raw.ContractError):
        raw.read_bench(q)
    keys = {k: np.zeros(3) for k in raw.BENCH_KEYS}
    keys["y"] = np.array([0, 1, 2])
    keys.update({"val_" + k: np.zeros(3) for k in raw.BENCH_KEYS})
    np.savez(p, **keys)
    with pytest.raises(raw.ContractError, match="0/1"):
        raw.read_bench(p)


def test_mu_with_p_separator(tmp_path):
    synthetic.write_bench(tmp_path, tasks=("a",), mus=(0.25,), seeds=(1,), n_images=20, n_val=10)
    (tmp_path / "a_mu0.25_s1.npz").rename(tmp_path / "a_mu0p25_s1.npz")
    assert raw.read_bench(tmp_path / "a_mu0p25_s1.npz")["mu"] == 0.25


def test_rephase_reader(tmp_path):
    synthetic.write_rephase(tmp_path, tasks=("cat_dog",), seeds=(0,), n_images=40, n_val=20)
    d = raw.read_rephase(tmp_path / "cat_dog_s0.npz")
    assert d["z_M"].shape == (5, 200) and d["val_z_M"].shape == (5, 20)
    assert list(d["lambdas"]) == [0.0, 0.3, 1.0, 3.0, 10.0]


def test_people_readers(tmp_path):
    synthetic.write_people(tmp_path, tmp_path / "counts.npy", seeds=(0,))
    f = raw.read_people_fold(tmp_path / "fold0_s0.npz")
    assert f["logits"].shape == (4, 5000, 10) and f["fold"] == 0
    b = raw.read_people_base(tmp_path / "base_s0.npz")
    assert b["logits"].shape == (10000, 10)
    counts = np.load(tmp_path / "counts.npy")
    assert counts.shape == (10000, 10)
    bad = tmp_path / "fold1_s0.npz"
    np.savez(bad, image=np.arange(3), y=np.arange(3), logits=np.zeros((2, 3, 9)), lambdas=np.zeros(2))
    with pytest.raises(raw.ContractError):
        raw.read_people_fold(bad)


def test_rephase_bis_reader(tmp_path):
    synthetic.write_rephase_bis(tmp_path, tasks=("cat_dog",), seeds=(0,), n_images=40, n_val=20)
    d = raw.read_rephase_bis(tmp_path / "cat_dog_s0.npz")
    assert d["machine_names"] == ["M0", "Mpair", "Mframe"]
    assert d["z_M"].shape == (3, 200) and d["val_z_M"].shape == (3, 20)
    bad = tmp_path / "x_s0.npz"
    with np.load(tmp_path / "cat_dog_s0.npz", allow_pickle=False) as f:
        arrays = {k: f[k] for k in f.files}
    arrays["machines"] = np.array(["M0", "Mpair", "Mother"])
    np.savez(bad, **arrays)
    with pytest.raises(raw.ContractError, match="machines"):
        raw.read_rephase_bis(bad)
