"""E2 bis, the partner trained for the pair (protocol section 6 bis, exploratory).

For every task and seed, with the tasks, seeds, halves, frame and seeds of E2 (rephase.py):
- H, the person: a "plus" network with mu = 1 trained on half A, then frozen (eval mode);
- M0: trained on half B with cross-entropy, frame a ~ N(0, 2) independent of the class (as M_0 of E2);
- Mpair: the same network, half B, same frame; the loss is the cross-entropy of the summed cue: total log-odds
  = (out1 - out0) + z_H, with z_H the log-odds of H on the same augmented, framed presentation, without gradient,
  weight one, no standardisation;
- Mframe: as Mpair, but the interior of the image is set to zero (normalised units), in training and in test: it
  sees only the 2-pixel frame, while H still sees the whole image.
All machines of a unit share seed, initialisation, batch order, crops, flips and frames.
On validation and test every image is shown 5 times with a ~ N(0, 2), the same eps as E2.
z_M holds the log-odds of each machine on its own, not the sum.

    python gpu/rephase_bis.py OUT CONFIG WORKER WORKERS [SEEDS] [TASKS]
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from common import (DEV, PRESENTATIONS, TASKS, Clock, Net, cifar, frame_only, log_odds, logits, new_model, save_npz,
                    seed_of, setup_torch, task_split, train, versions, write_json)

MU_H = 1.0
VAR_A = 1 + MU_H ** 2
MACHINES = ("M0", "Mpair", "Mframe")


def run_unit(D, cfg, out: Path, seed: int, task: str) -> None:
    Xtr, Ytr, Xte, Yte, ytr, yte = D
    name = f"{task}_s{seed}"
    ids = task_split(task, ytr, yte)
    c1 = TASKS[task][1]
    delta, ep, lr, bs = cfg["delta"], cfg["epochs"], cfg["lr"], cfg["batch_size"]
    XA, YA = Xtr[ids["half_a"]], (Ytr[ids["half_a"]] == c1).long()
    XB, YB = Xtr[ids["half_b"]], (Ytr[ids["half_b"]] == c1).long()
    sets = {}
    for part, Xall, yall, idx in (("val", Xtr, ytr, ids["val"]), ("test", Xte, yte, ids["test"])):
        image = np.repeat(idx, PRESENTATIONS)
        eps = np.random.default_rng(seed_of("rephase", task, f"s{seed}", f"{part}-a")).standard_normal(len(image))
        a = (np.sqrt(VAR_A) * eps).astype(np.float32)
        sets[part] = {"image": image.astype(np.int32), "a": a, "y": (yall[image] == c1).astype(np.int8),
                      "X": Xall[torch.as_tensor(image, device=DEV)], "a_t": torch.as_tensor(a, device=DEV)}

    def plus(y, gen):
        return MU_H * (2.0 * y.float() - 1.0) + torch.randn(y.shape[0], device=DEV, generator=gen)

    def flat(y, gen):
        return np.sqrt(VAR_A) * torch.randn(y.shape[0], device=DEV, generator=gen)

    clock = Clock()
    rec = {"unit": name, "task": task, "seed": seed, "nets": {}}
    z = {}

    def evaluate(m, key, info, view=None):
        for part, d in sets.items():
            zz = log_odds(logits(m, d["X"], d["a_t"], delta, view=view))
            z[(part, key)] = zz
            info[f"{part}_acc"] = round(float(((zz > 0) == (d["y"] == 1)).mean()), 4)

    s_h = seed_of("rephase", task, f"s{seed}", "H")
    t0 = clock()
    H = new_model(Net, s_h, n_out=2)
    train(H, XA, YA, epochs=ep, lr=lr, bs=bs, seed=s_h, frame=plus, delta=delta)
    info = {"torch_seed": s_h, "train_seconds": round(clock() - t0, 1), "half": "A"}
    evaluate(H, "H", info)
    rec["nets"]["H"] = info
    H.eval()
    for p in H.parameters():
        p.requires_grad_(False)

    def pair_loss(x, y, out, idx):
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            zh = H(x)                                        # H sees the whole framed image
        total = (out[:, 1] - out[:, 0]).float() + (zh[:, 1] - zh[:, 0]).float()
        return F.binary_cross_entropy_with_logits(total, y.float())

    s_m = seed_of("rephase", task, f"s{seed}", "M")
    for key, loss_fn, view in (("M0", None, None), ("Mpair", pair_loss, None), ("Mframe", pair_loss, frame_only)):
        t0 = clock()
        m = new_model(Net, s_m, n_out=2)
        train(m, XB, YB, epochs=ep, lr=lr, bs=bs, seed=s_m, frame=flat, delta=delta, loss_fn=loss_fn, view=view)
        info = {"torch_seed": s_m, "train_seconds": round(clock() - t0, 1), "half": "B"}
        evaluate(m, key, info, view=view)
        for part, d in sets.items():
            s = z[(part, "H")] + z[(part, key)]
            info[f"{part}_acc_sum_raw"] = round(float(((s > 0) == (d["y"] == 1)).mean()), 4)
        rec["nets"][key] = info
        del m
    arrays = {"machines": np.array(MACHINES)}
    for part, d in sets.items():
        p = "" if part == "test" else "val_"
        arrays[p + "y"], arrays[p + "image"], arrays[p + "a"] = d["y"], d["image"], d["a"]
        arrays[p + "z_H"] = z[(part, "H")]
        arrays[p + "z_M"] = np.stack([z[(part, k)] for k in MACHINES])
    save_npz(out / f"{name}.npz", **arrays)
    rec["seconds"] = clock()
    write_json(out / "units" / f"{name}.json", rec)
    print(json.dumps({"unit": name, "s": rec["seconds"], **{k: v["test_acc"] for k, v in rec["nets"].items()},
                      **{f"{k}_sum": v["test_acc_sum_raw"] for k, v in rec["nets"].items() if k != "H"}}), flush=True)


def main() -> None:
    setup_torch()
    out, cfg = Path(sys.argv[1]), json.loads(Path(sys.argv[2]).read_text())
    worker, workers = int(sys.argv[3]), int(sys.argv[4])
    seeds = [int(s) for s in sys.argv[5].split(",")] if len(sys.argv) > 5 else [0, 1]
    tasks = sys.argv[6].split(",") if len(sys.argv) > 6 else list(TASKS)
    (out / "units").mkdir(parents=True, exist_ok=True)
    if worker == 0:
        write_json(out / "units" / "_versions.json", versions())
    Xtr, Ytr, Xte, Yte = cifar()
    D = (Xtr, Ytr, Xte, Yte, Ytr.cpu().numpy(), Yte.cpu().numpy())
    for seed, task in [(s, t) for s in seeds for t in tasks][worker::workers]:
        if (out / f"{task}_s{seed}.npz").exists():
            continue
        run_unit(D, cfg, out, seed, task)
    print("END", worker, flush=True)


if __name__ == "__main__":
    main()
