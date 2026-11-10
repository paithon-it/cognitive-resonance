"""Pilot, before the runs: throughput, frame probe, and the choice of epochs and learning rate at mu = 0.

    python gpu/pilot.py timing OUT                 # images per second of one Net and of the ResNet-18
    python gpu/pilot.py probe OUT                  # linear probe: can the frame value a be read off the image?
    python gpu/pilot.py grid OUT WORKER WORKERS    # one share of the epochs x learning-rate grid, at mu = 0
    python gpu/pilot.py choose OUT                 # pick epochs and learning rate from the validation accuracies

The grid uses the validation images only (protocol section 5): two tasks, one seed, mu = 0, frame a ~ N(0, 1).
Rule fixed before looking: the smallest number of epochs whose mean validation accuracy is within 0.5 points
of the best cell, with the learning rate that is best at that number of epochs.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

from common import (DEV, PRESENTATIONS, TASKS, Clock, Machine, Net, apply_frame, cifar, logits, new_model, seed_of,
                    setup_torch, task_split, train, versions, write_json)

GRID_TASKS = ("cat-dog", "automobile-truck")
GRID_EPOCHS = (10, 15, 20, 30)
GRID_LR = (0.05, 0.1)
BS = 256
DELTAS = (0.25, 0.5, 1.0)
DELTA_DEFAULT = 1.0
TOLERANCE = 0.005


def data(out: Path):
    Xtr, Ytr, Xte, Yte = cifar()
    return Xtr, Ytr, Xte, Yte, Ytr.cpu().numpy(), Yte.cpu().numpy()


def frame_mu0(y, gen):
    return torch.randn(y.shape[0], device=DEV, generator=gen)


def timing(out: Path) -> None:
    Xtr, Ytr, Xte, Yte, ytr, yte = data(out)
    ids = task_split("cat-dog", ytr, yte)
    X, Y = Xtr[ids["train"]], (Ytr[ids["train"]] == TASKS["cat-dog"][1]).long()
    res = {}
    for epochs in (1, 3):                           # the first includes cudnn autotuning
        m = new_model(Net, 1, n_out=2)
        torch.cuda.synchronize(); t = time.time()
        train(m, X, Y, epochs=epochs, lr=0.1, bs=BS, seed=1, frame=frame_mu0)
        torch.cuda.synchronize()
        res[f"net_{epochs}ep_s"] = round(time.time() - t, 2)
    res["net_img_per_s"] = round(3 * len(X) / res["net_3ep_s"])      # the 3-epoch run, after autotuning
    m = new_model(Machine, 1, n_out=10)
    sub = torch.arange(10240, device=DEV)
    torch.cuda.synchronize(); t = time.time()
    train(m, Xtr[sub], Ytr[sub], epochs=1, lr=0.1, bs=BS, seed=1)
    torch.cuda.synchronize(); t1 = time.time() - t; t = time.time()
    train(m, Xtr[sub], Ytr[sub], epochs=2, lr=0.1, bs=BS, seed=1)
    torch.cuda.synchronize(); t2 = time.time() - t
    res["resnet_img_per_s"] = round(2 * 10240 / t2)
    res["resnet_first_epoch_s"] = round(t1, 2)
    res["versions"] = versions()
    print(json.dumps(res))
    write_json(out / "timing.json", res)


def probe(out: Path) -> None:
    """Ridge regression from the framed image (3072 pixels) to a, fitted on training images, R^2 on validation."""
    Xtr, Ytr, Xte, Yte, ytr, yte = data(out)
    ids = task_split("cat-dog", ytr, yte)
    g = torch.Generator(device=DEV).manual_seed(seed_of("pilot", "probe"))
    Xf = Xtr[ids["train"]]
    Xv = Xtr[ids["val"]].repeat_interleave(PRESENTATIONS, 0)
    res = {}
    for delta in DELTAS:
        a = torch.randn(len(Xf), device=DEV, generator=g)
        av = torch.randn(len(Xv), device=DEV, generator=g)
        A = apply_frame(Xf, a, delta).flatten(1).double()
        V = apply_frame(Xv, av, delta).flatten(1).double()
        mu = A.mean(0)
        A, V = A - mu, V - mu
        w = torch.linalg.solve(A.T @ A + 1.0 * torch.eye(A.shape[1], device=DEV, dtype=A.dtype), A.T @ (a.double() - a.mean()))
        pred = V @ w + a.mean()
        r2 = 1 - ((av.double() - pred) ** 2).sum() / ((av.double() - av.double().mean()) ** 2).sum()
        res[str(delta)] = round(float(r2), 5)
    print(json.dumps(res))
    write_json(out / "probe.json", {"ridge_alpha": 1.0, "fit": "9,000 cat-dog training images, a ~ N(0,1)",
                                     "test": "1,000 validation images x 5 presentations, a ~ N(0,1)", "r2_by_delta": res})


def grid(out: Path, worker: int, workers: int) -> None:
    Xtr, Ytr, Xte, Yte, ytr, yte = data(out)
    cells = [(t, e, lr) for t in GRID_TASKS for e in GRID_EPOCHS for lr in GRID_LR]
    cells.sort(key=lambda c: -c[1])                  # longest first, for balance
    for t, e, lr in cells[worker::workers]:
        dest = out / "grid" / f"{t}_e{e}_lr{lr}.json"
        if dest.exists():
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        ids = task_split(t, ytr, yte)
        c1 = TASKS[t][1]
        X, Y = Xtr[ids["train"]], (Ytr[ids["train"]] == c1).long()
        Xv = Xtr[ids["val"]].repeat_interleave(PRESENTATIONS, 0)
        yv = (Ytr[ids["val"]] == c1).long().repeat_interleave(PRESENTATIONS, 0)
        av = torch.tensor(np.random.default_rng(seed_of("pilot", t, "val-a")).standard_normal(len(Xv)),
                          device=DEV, dtype=torch.float32)
        clock = Clock()
        m = new_model(Net, seed_of("pilot", t, "net"), n_out=2)
        train(m, X, Y, epochs=e, lr=lr, bs=BS, seed=seed_of("pilot", t, "net"), frame=frame_mu0, delta=DELTA_DEFAULT)
        torch.cuda.synchronize()
        lg = logits(m, Xv, av, DELTA_DEFAULT)
        acc = float(((lg[:, 1] > lg[:, 0]).long() == yv).float().mean())
        rec = {"task": t, "epochs": e, "lr": lr, "val_acc": round(acc, 4), "seconds": clock()}
        print(json.dumps(rec), flush=True)
        write_json(dest, rec)


def choose(out: Path) -> None:
    recs = [json.loads(p.read_text()) for p in sorted((out / "grid").glob("*.json"))]
    mean = {}
    for e in GRID_EPOCHS:
        for lr in GRID_LR:
            v = [r["val_acc"] for r in recs if r["epochs"] == e and r["lr"] == lr]
            mean[(e, lr)] = float(np.mean(v))
    best = max(mean.values())
    epochs = min(e for (e, lr), v in mean.items() if v >= best - TOLERANCE)
    lr = max(GRID_LR, key=lambda l: mean[(epochs, l)])
    probe_res = json.loads((out / "probe.json").read_text())
    timing_res = json.loads((out / "timing.json").read_text())
    cfg = {"epochs": epochs, "lr": lr, "batch_size": BS, "delta": DELTA_DEFAULT,
           "rule": "smallest epochs within 0.5 accuracy points of the best grid cell (mean over the two pilot "
                   "tasks), then the best learning rate at those epochs",
           "grid_mean_val_acc": {f"e{e}_lr{l}": round(v, 4) for (e, l), v in mean.items()},
           "grid": recs, "probe": probe_res, "timing": timing_res}
    write_json(out / "config.json", cfg)
    print(json.dumps({k: cfg[k] for k in ("epochs", "lr", "batch_size", "delta", "grid_mean_val_acc")}, indent=1))


def main() -> None:
    setup_torch()
    mode, out = sys.argv[1], Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)
    if mode == "timing":
        timing(out)
    elif mode == "probe":
        probe(out)
    elif mode == "grid":
        grid(out, int(sys.argv[3]), int(sys.argv[4]))
    elif mode == "choose":
        choose(out)
    else:
        raise SystemExit(f"unknown mode {mode}")


if __name__ == "__main__":
    main()
