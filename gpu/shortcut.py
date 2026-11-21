"""E14, what a shortcut leaves unlearned (protocol section 20).

The "plus" networks of E1 again: in training the frame says something about the class (strength mu), at validation
and at test it says nothing. Two things E1 did not have: a training three times as long, and the last layer trained
again on data where the frame says nothing (Kirichenko, Izmailov and Wilson, 2023). For every task, seed, length of
training and mu, one network; saved for validation and test:
- z: its log-odds as they are;
- z_last: the log-odds of a new last layer, a logistic regression on the 192 features before the last layer, fitted
  on the validation presentations (5,000: the frame there is independent of the class) and read on the test ones.
The frames at validation and at test are those of E1 (same seeds): a = sqrt(1 + mu^2) eps, the same eps for every mu.

    python gpu/shortcut.py OUT CONFIG WORKER WORKERS [SEEDS] [TASKS]
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from common import (DEV, PRESENTATIONS, TASKS, Clock, Net, apply_frame, cifar, log_odds, logits, new_model, save_npz,
                    seed_of, setup_torch, task_split, train, versions, write_json)

MUS = (0.0, 1.0, 1.5, 2.0)
LENGTHS = (1, 3)                 # times the epochs of the pilot: 30 and 90
SEEDS = (0, 1, 2)
RIDGE = 1e-3                     # on the standardised features, fixed beforehand


@torch.no_grad()
def features(model, X, a, delta, bs=2000) -> torch.Tensor:
    """The 192 numbers before the last layer, in eval mode, with the frame a * delta."""
    model.eval()
    out = []
    for k in range(0, len(X), bs):
        x = apply_frame(X[k:k + bs], a[k:k + bs], delta).contiguous(memory_format=torch.channels_last)
        out.append(model.f(x).float())
    return torch.cat(out)


def new_last_layer(f_val, y_val, f_test):
    """Log-odds at validation and test of a logistic regression fitted on the validation features."""
    mean, sd = f_val.mean(0, keepdim=True), f_val.std(0, keepdim=True).clamp(min=1e-6)
    a, b = (f_val - mean) / sd, (f_test - mean) / sd
    w = torch.zeros(a.shape[1], device=DEV, requires_grad=True)
    c = torch.zeros(1, device=DEV, requires_grad=True)
    opt = torch.optim.LBFGS([w, c], lr=1.0, max_iter=300, tolerance_grad=1e-7, line_search_fn="strong_wolfe")
    target = y_val.float()

    def closure():
        opt.zero_grad()
        loss = F.binary_cross_entropy_with_logits(a @ w + c, target) + RIDGE * (w * w).sum()
        loss.backward()
        return loss

    opt.step(closure)
    with torch.no_grad():
        return (a @ w + c).cpu().numpy().astype(np.float32), (b @ w + c).cpu().numpy().astype(np.float32)


def run_unit(D, cfg, out: Path, seed: int, task: str, length: int) -> None:
    Xtr, Ytr, Xte, Yte, ytr, yte = D
    epochs = cfg["epochs"] * length
    name = f"{task}_e{epochs}_s{seed}"
    ids = task_split(task, ytr, yte)
    c1 = TASKS[task][1]
    X, Y = Xtr[ids["train"]], (Ytr[ids["train"]] == c1).long()
    sets = {}
    for part, Xall, yall, idx in (("val", Xtr, ytr, ids["val"]), ("test", Xte, yte, ids["test"])):
        image = np.repeat(idx, PRESENTATIONS)
        eps = np.random.default_rng(seed_of("bench", task, f"s{seed}", f"{part}-a")).standard_normal(len(image))
        y = (yall[image] == c1).astype(np.int8)
        sets[part] = {"image": image.astype(np.int32), "eps": eps.astype(np.float32), "y": y,
                      "X": Xall[torch.as_tensor(image, device=DEV)], "y_t": torch.as_tensor(y, device=DEV)}
    clock = Clock()
    s = seed_of("bench", task, f"s{seed}", "plus")            # the seed of the "plus" network of E1
    z, z_last, frames, rec = {}, {}, {}, {"unit": name, "task": task, "seed": seed, "epochs": epochs, "nets": {}}
    for mu in MUS:

        def frame(y, gen, mu=mu):
            return mu * (2.0 * y.float() - 1.0) + torch.randn(y.shape[0], device=DEV, generator=gen)

        t0 = clock()
        m = new_model(Net, s, n_out=2)
        train(m, X, Y, epochs=epochs, lr=cfg["lr"], bs=cfg["batch_size"], seed=s, frame=frame, delta=cfg["delta"])
        info = {"torch_seed": s, "train_seconds": round(clock() - t0, 1)}
        feats = {}
        for part, d in sets.items():
            a = (np.sqrt(1 + mu ** 2) * d["eps"]).astype(np.float32)
            a_t = torch.as_tensor(a, device=DEV)
            frames[(part, mu)] = a
            zz = log_odds(logits(m, d["X"], a_t, cfg["delta"]))
            z[(part, mu)] = zz
            info[f"{part}_acc"] = round(float(((zz > 0) == (d["y"] == 1)).mean()), 4)
            feats[part] = features(m, d["X"], a_t, cfg["delta"])
        z_last[("val", mu)], z_last[("test", mu)] = new_last_layer(feats["val"], sets["val"]["y_t"], feats["test"])
        info["test_acc_last"] = round(float(((z_last[("test", mu)] > 0) == (sets["test"]["y"] == 1)).mean()), 4)
        rec["nets"][f"mu{mu:g}"] = info
        del m, feats
    arrays = {"mus": np.array(MUS, dtype=np.float32), "epochs": np.array(epochs, dtype=np.int32)}
    for part, d in sets.items():
        p = "" if part == "test" else "val_"
        arrays[p + "y"], arrays[p + "image"] = d["y"], d["image"]
        arrays[p + "a"] = np.stack([frames[(part, mu)] for mu in MUS])
        arrays[p + "z"] = np.stack([z[(part, mu)] for mu in MUS])
        arrays[p + "z_last"] = np.stack([z_last[(part, mu)] for mu in MUS])
    save_npz(out / f"{name}.npz", **arrays)
    rec["seconds"] = clock()
    write_json(out / "units" / f"{name}.json", rec)
    print(json.dumps({"unit": name, "s": rec["seconds"],
                      **{k: (v["test_acc"], v["test_acc_last"]) for k, v in rec["nets"].items()}}), flush=True)


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
    # the short trainings first: if time runs out, what is missing is the long ones of the last seed
    todo = [(s, t, n) for n in LENGTHS for s in seeds for t in tasks]
    for seed, task, length in todo[worker::workers]:
        if (out / f"{task}_e{cfg['epochs'] * length}_s{seed}.npz").exists():
            continue
        run_unit(D, cfg, out, seed, task, length)
    print("END", worker, flush=True)


if __name__ == "__main__":
    main()
