"""E1, the bench (protocol section 5): two opposite biases planted with a frame.

For every task, mu and seed, three small networks learn the binary task with a spurious frame whose value a
depends on the class in training: "plus" sees a ~ N(+mu, 1) on class 1 and N(-mu, 1) on class 0, "minus" the
opposite, "plus2" is a second "plus" with another seed. On validation and test each image is shown 5 times with
a ~ N(0, 1 + mu^2), independent of the class and identical for the three judges.

    python gpu/bench.py OUT CONFIG WORKER WORKERS [SEEDS] [TASKS]

Each unit is written to OUT/{task}_mu{mu}_s{seed}.npz (resumable), with a sidecar JSON in OUT/units/.
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

from common import (DEV, PRESENTATIONS, TASKS, Clock, Net, cifar, log_odds, logits, new_model, save_npz, seed_of,
                    setup_torch, task_split, train, versions, write_json)

MUS = (0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0)
ROLES = {"plus": 1.0, "minus": -1.0, "plus2": 1.0}


def mu_name(mu: float) -> str:
    return f"{mu:g}"


def units(seeds, tasks):
    """Seed-major order: if time runs out, the units that are missing are those of the last seed."""
    return [(s, t, mu) for s in seeds for t in tasks for mu in MUS]


def run_unit(D, cfg, out: Path, seed: int, task: str, mu: float) -> None:
    Xtr, Ytr, Xte, Yte, ytr, yte = D
    name = f"{task}_mu{mu_name(mu)}_s{seed}"
    ids = task_split(task, ytr, yte)
    c1 = TASKS[task][1]
    X, Y = Xtr[ids["train"]], (Ytr[ids["train"]] == c1).long()
    sets = {}
    for part, Xall, Yall, idx in (("val", Xtr, Ytr, ids["val"]), ("test", Xte, Yte, ids["test"])):
        image = np.repeat(idx, PRESENTATIONS)
        # common random numbers: the same standard normals for every mu, scaled to variance 1 + mu^2
        eps = np.random.default_rng(seed_of("bench", task, f"s{seed}", f"{part}-a")).standard_normal(len(image))
        a = (np.sqrt(1 + mu ** 2) * eps).astype(np.float32)
        y = (yall_np(Yall)[image] == c1).astype(np.int8)
        sets[part] = {"image": image.astype(np.int32), "a": a, "y": y,
                      "X": Xall[torch.as_tensor(image, device=DEV)], "a_t": torch.as_tensor(a, device=DEV)}
    clock = Clock()
    z, rec = {}, {"unit": name, "task": task, "mu": mu, "seed": seed, "nets": {}}
    for role, sign in ROLES.items():
        s = seed_of("bench", task, f"s{seed}", role)          # same seed for every mu (common random numbers)

        def frame(y, gen, sign=sign):
            return sign * mu * (2.0 * y.float() - 1.0) + torch.randn(y.shape[0], device=DEV, generator=gen)

        t0 = clock()
        m = new_model(Net, s, n_out=2)
        train(m, X, Y, epochs=cfg["epochs"], lr=cfg["lr"], bs=cfg["batch_size"], seed=s, frame=frame,
              delta=cfg["delta"])
        info = {"torch_seed": s, "train_seconds": round(clock() - t0, 1)}
        for part, d in sets.items():
            zz = log_odds(logits(m, d["X"], d["a_t"], cfg["delta"]))
            z[(part, role)] = zz
            info[f"{part}_acc"] = round(float(((zz > 0) == (d["y"] == 1)).mean()), 4)
        rec["nets"][role] = info
        del m
    arrays = {}
    for part, d in sets.items():
        p = "" if part == "test" else "val_"
        arrays[p + "y"], arrays[p + "image"], arrays[p + "a"] = d["y"], d["image"], d["a"]
        for role in ROLES:
            arrays[f"{p}z_{role}"] = z[(part, role)]
    save_npz(out / f"{name}.npz", **arrays)
    rec["seconds"] = clock()
    write_json(out / "units" / f"{name}.json", rec)
    print(json.dumps({"unit": name, "s": rec["seconds"],
                      **{r: rec["nets"][r]["test_acc"] for r in ROLES}}), flush=True)


_YNP = {}


def yall_np(Y: torch.Tensor) -> np.ndarray:
    k = id(Y)
    if k not in _YNP:
        _YNP[k] = Y.cpu().numpy()
    return _YNP[k]


def main() -> None:
    setup_torch()
    out, cfg = Path(sys.argv[1]), json.loads(Path(sys.argv[2]).read_text())
    worker, workers = int(sys.argv[3]), int(sys.argv[4])
    seeds = [int(s) for s in sys.argv[5].split(",")] if len(sys.argv) > 5 else [0, 1, 2]
    tasks = sys.argv[6].split(",") if len(sys.argv) > 6 else list(TASKS)
    (out / "units").mkdir(parents=True, exist_ok=True)
    if worker == 0:
        write_json(out / "units" / "_versions.json", versions())
    Xtr, Ytr, Xte, Yte = cifar()
    D = (Xtr, Ytr, Xte, Yte, Ytr.cpu().numpy(), Yte.cpu().numpy())
    for seed, task, mu in units(seeds, tasks)[worker::workers]:
        if (out / f"{task}_mu{mu_name(mu)}_s{seed}.npz").exists():
            continue
        run_unit(D, cfg, out, seed, task, mu)
    print("END", worker, flush=True)


if __name__ == "__main__":
    main()
