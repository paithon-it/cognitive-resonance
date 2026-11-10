"""E2, rephasing (protocol section 6): a machine trained on the errors of a mind.

For every task and seed:
- H, the person: a "plus" network with mu = 1 (frame a ~ N(+1, 1) on class 1, N(-1, 1) on class 0), trained on
  half A of the task's training images;
- H2, the control H': another "plus" network with mu = 1, trained on half B;
- M_lambda, the machines: trained on half B with a frame that does not depend on the class, a ~ N(0, 2). To the
  cross-entropy one adds lambda times the Pearson correlation, in the batch, between the cue of M and the cue of
  H towards the true class, after removing the class means in the batch. H is frozen (eval mode) and sees the
  same augmented, framed batch. All machines of a unit share seed, initialisation and batch order.
On validation and test every image is shown 5 times with a ~ N(0, 2), the same for all judges of a unit.

    python gpu/rephase.py OUT CONFIG WORKER WORKERS [SEEDS] [TASKS]
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

from common import (DEV, PRESENTATIONS, TASKS, Clock, Net, batch_corr, cifar, log_odds, logits, new_model, save_npz,
                    seed_of, setup_torch, task_split, train, versions, write_json)

MU_H = 1.0
LAMBDAS = (0.0, 0.3, 1.0, 3.0, 10.0)
VAR_A = 1 + MU_H ** 2                     # 2: the variance of a seen by H in training


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

    def evaluate(m, key, info):
        for part, d in sets.items():
            zz = log_odds(logits(m, d["X"], d["a_t"], delta))
            z[(part, key)] = zz
            info[f"{part}_acc"] = round(float(((zz > 0) == (d["y"] == 1)).mean()), 4)

    nets = {}
    for role, Xh, Yh in (("H", XA, YA), ("H2", XB, YB)):
        s = seed_of("rephase", task, f"s{seed}", role)
        t0 = clock()
        m = new_model(Net, s, n_out=2)
        train(m, Xh, Yh, epochs=ep, lr=lr, bs=bs, seed=s, frame=plus, delta=delta)
        info = {"torch_seed": s, "train_seconds": round(clock() - t0, 1), "half": "A" if role == "H" else "B"}
        evaluate(m, role, info)
        rec["nets"][role] = info
        nets[role] = m
    H = nets.pop("H")
    H.eval()
    for p in H.parameters():
        p.requires_grad_(False)
    del nets
    s_m = seed_of("rephase", task, f"s{seed}", "M")
    for lam in LAMBDAS:
        corr_log = []

        def extra(x, y, out, idx, lam=lam, corr_log=corr_log):
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                zh = H(x)
            sgn = 2.0 * y.float() - 1.0
            c_m = sgn * (out[:, 1] - out[:, 0]).float()
            c_h = sgn * (zh[:, 1] - zh[:, 0]).float()
            r = batch_corr(c_m, c_h, y, 2)
            corr_log.append(r.detach())
            return lam * r if lam > 0 else torch.zeros((), device=DEV)

        t0 = clock()
        m = new_model(Net, s_m, n_out=2)
        train(m, XB, YB, epochs=ep, lr=lr, bs=bs, seed=s_m, frame=flat, delta=delta, extra=extra)
        per_epoch = len(XB) // bs
        info = {"torch_seed": s_m, "train_seconds": round(clock() - t0, 1), "half": "B",
                "batch_corr_last_epoch": round(float(torch.stack(corr_log[-per_epoch:]).mean()), 4)}
        evaluate(m, f"M{lam:g}", info)
        rec["nets"][f"M{lam:g}"] = info
        del m
    arrays = {"lambdas": np.array(LAMBDAS, dtype=np.float32)}
    for part, d in sets.items():
        p = "" if part == "test" else "val_"
        arrays[p + "y"], arrays[p + "image"], arrays[p + "a"] = d["y"], d["image"], d["a"]
        arrays[p + "z_H"], arrays[p + "z_H2"] = z[(part, "H")], z[(part, "H2")]
        arrays[p + "z_M"] = np.stack([z[(part, f"M{lam:g}")] for lam in LAMBDAS])
    save_npz(out / f"{name}.npz", **arrays)
    rec["seconds"] = clock()
    write_json(out / "units" / f"{name}.json", rec)
    print(json.dumps({"unit": name, "s": rec["seconds"],
                      **{k: v["test_acc"] for k, v in rec["nets"].items()}}), flush=True)


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
