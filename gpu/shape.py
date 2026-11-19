"""E9, the shape of the noise (protocol section 15).

The networks are those of E1 (bench.py): for every task, mu and seed, "plus" and "minus" learn the binary task
with a frame whose value depends on the class in training, a ~ N(+-mu, 1), with opposite signs. What changes is
the test. On validation and test each image is shown 5 times, and the frame is drawn from four distributions
with the same variance 1 + mu^2 and zero mean, independent of the class:

- gauss:   a normal, as in E1 (on seeds E1 never used: it is also a replication);
- two:     two values, +-sqrt(1 + mu^2), a light frame or a dark one;
- uniform: uniform between +-sqrt(3 (1 + mu^2));
- t3:      Student's t with 3 degrees of freedom, rescaled to the same variance.

    python gpu/shape.py OUT CONFIG WORKER WORKERS [SEEDS] [TASKS]

Each unit is written to OUT/{task}_mu{mu}_s{seed}.npz (resumable), with a sidecar JSON in OUT/units/.
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

from bench import mu_name, yall_np
from common import (DEV, PRESENTATIONS, TASKS, Clock, Net, cifar, log_odds, logits, new_model, save_npz, seed_of,
                    setup_torch, task_split, train, versions, write_json)

MUS = (1.0, 1.5, 2.0)
SEEDS = (2, 3, 4, 5, 6)                 # E1 used 0 and 1
ROLES = {"plus": 1.0, "minus": -1.0}
SHAPES = ("gauss", "two", "uniform", "t3")


def draw(shape: str, n: int, gen: np.random.Generator) -> np.ndarray:
    """n numbers with zero mean and unit variance, of the given shape."""
    if shape == "gauss":
        return gen.standard_normal(n)
    if shape == "two":
        return np.where(gen.random(n) < 0.5, 1.0, -1.0)
    if shape == "uniform":
        return gen.uniform(-np.sqrt(3.0), np.sqrt(3.0), n)
    if shape == "t3":
        return gen.standard_t(3, n) / np.sqrt(3.0)
    raise ValueError(shape)


def units(seeds, tasks):
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
        frames = {}
        for shape in SHAPES:
            gen = np.random.default_rng(seed_of("shape", task, f"s{seed}", f"{part}-{shape}"))
            frames[shape] = (np.sqrt(1 + mu ** 2) * draw(shape, len(image), gen)).astype(np.float32)
        sets[part] = {"image": image.astype(np.int32), "a": frames, "y": (yall_np(Yall)[image] == c1).astype(np.int8),
                      "X": Xall[torch.as_tensor(image, device=DEV)]}
    clock = Clock()
    z, rec = {}, {"unit": name, "task": task, "mu": mu, "seed": seed, "nets": {}}
    for role, sign in ROLES.items():
        s = seed_of("bench", task, f"s{seed}", role)          # the seed labels of E1, on new seeds

        def frame(y, gen, sign=sign):
            return sign * mu * (2.0 * y.float() - 1.0) + torch.randn(y.shape[0], device=DEV, generator=gen)

        t0 = clock()
        m = new_model(Net, s, n_out=2)
        train(m, X, Y, epochs=cfg["epochs"], lr=cfg["lr"], bs=cfg["batch_size"], seed=s, frame=frame,
              delta=cfg["delta"])
        info = {"torch_seed": s, "train_seconds": round(clock() - t0, 1)}
        for part, d in sets.items():
            for shape in SHAPES:
                zz = log_odds(logits(m, d["X"], torch.as_tensor(d["a"][shape], device=DEV), cfg["delta"]))
                z[(part, role, shape)] = zz
                info[f"{part}_acc_{shape}"] = round(float(((zz > 0) == (d["y"] == 1)).mean()), 4)
        rec["nets"][role] = info
        del m
    arrays = {"shapes": np.array(SHAPES)}
    for part, d in sets.items():
        p = "" if part == "test" else "val_"
        arrays[p + "y"], arrays[p + "image"] = d["y"], d["image"]
        arrays[p + "a"] = np.stack([d["a"][shape] for shape in SHAPES])
        for role in ROLES:
            arrays[f"{p}z_{role}"] = np.stack([z[(part, role, shape)] for shape in SHAPES])
    save_npz(out / f"{name}.npz", **arrays)
    rec["seconds"] = clock()
    write_json(out / "units" / f"{name}.json", rec)
    print(json.dumps({"unit": name, "s": rec["seconds"],
                      **{f"{r}_{sh}": rec["nets"][r][f"test_acc_{sh}"] for r in ROLES for sh in SHAPES}}), flush=True)


def main() -> None:
    setup_torch()
    out, cfg = Path(sys.argv[1]), json.loads(Path(sys.argv[2]).read_text())
    worker, workers = int(sys.argv[3]), int(sys.argv[4])
    seeds = [int(s) for s in sys.argv[5].split(",")] if len(sys.argv) > 5 else list(SEEDS)
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
