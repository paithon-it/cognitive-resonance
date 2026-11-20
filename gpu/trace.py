"""E10 to E12, the bench of the informative chart (protocol sections 16 to 18).

In E8 the frame says nothing about the class when the judge is tested. Here it does. At validation, at test and in
the training of the machines the frame is a = mu_d (2y - 1) + sd_d eps, with total variance 2 as in E8 and a
sensitivity d of its own: 0 (the bench of E8), 0.5, 1 or 1.5. The judge was trained where the frame was worth 2:
it gives it more weight than it deserves.

For every task and seed:
- H: the judge of E8, a "plus" network with mu = 1 trained on half A on the whole image, then frozen (eval mode);
- C_d, one per level d: the same network and seed trained with frames of level d, the judge that gives the frame
  the weight it has at test (C_0 is the judge without the bias of E8);
and, for every view v of the machine (25: the left quarter of the inside; 100: all of it):
- M: the soloist, trained on half B with cross-entropy on its part of the inside. It does not see the frame;
- R_t, one per level t: a branch that sees only the frame, trained from scratch on half B on the binary
  cross-entropy of z_H + z_M + z_branch, with training frames of level t. The partner built for level t is
  M + R_t. R_0 is the branch of E8: it was trained where the frame said nothing.
The four branches of a unit share seed, order and noise of the frames: only the level changes.
On validation and test every image is shown 5 times; eps is the same at the four levels and at the two views.
Every branch is read at every level: z_R has shape (trained for, tested at, N).

    python gpu/trace.py OUT CONFIG WORKER WORKERS [SEEDS] [TASKS]
"""
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from common import (DEV, FRAME_WIDTH, PRESENTATIONS, TASKS, Clock, Net, cifar, frame_only, log_odds, logits,
                    new_model, save_npz, seed_of, setup_torch, task_split, train, versions, write_json)
from rephase_ter import MU_H, VAR_A, frozen
from trust import frozen_log_odds

LO, HI = FRAME_WIDTH, 32 - FRAME_WIDTH           # the inside of the image is rows and columns 2 to 29
LAST_COLUMN = {25: LO + 7, 100: HI}              # the machine sees the columns LO .. LAST_COLUMN - 1 of the inside
LEVELS = (0.0, 0.5, 1.0, 1.5)                    # sensitivity of the frame at validation, at test, in the machines' training
SEEDS = (7, 8, 9, 10, 11)                        # never used by an earlier run


def level_frame(level: float) -> tuple[float, float]:
    """Mean shift and within-class sd of a frame of sensitivity `level` and total variance VAR_A: d = 2 mu / sd."""
    mu = level * math.sqrt(VAR_A / (4.0 + level ** 2))
    return mu, math.sqrt(VAR_A - mu ** 2)


def inside_view(last: int):
    """What the soloist sees: the columns of the inside up to `last` (excluded), and no frame."""
    mask = torch.zeros(1, 1, 32, 32, device=DEV)
    mask[..., LO:HI, LO:last] = 1
    return lambda x: x * mask


def run_unit(D, cfg, out: Path, seed: int, task: str) -> None:
    Xtr, Ytr, Xte, Yte, ytr, yte = D
    ids = task_split(task, ytr, yte)
    c1 = TASKS[task][1]
    delta, ep, lr, bs = cfg["delta"], cfg["epochs"], cfg["lr"], cfg["batch_size"]
    XA, YA = Xtr[ids["half_a"]], (Ytr[ids["half_a"]] == c1).long()
    XB, YB = Xtr[ids["half_b"]], (Ytr[ids["half_b"]] == c1).long()
    sets = {}
    for part, Xall, yall, idx in (("val", Xtr, ytr, ids["val"]), ("test", Xte, yte, ids["test"])):
        image = np.repeat(idx, PRESENTATIONS)
        eps = np.random.default_rng(seed_of("trace", task, f"s{seed}", f"{part}-eps")).standard_normal(len(image))
        y = (yall[image] == c1).astype(np.int8)
        a = np.stack([mu * (2.0 * y - 1.0) + sd * eps for mu, sd in map(level_frame, LEVELS)]).astype(np.float32)
        sets[part] = {"image": image.astype(np.int32), "eps": eps.astype(np.float32), "a": a, "y": y,
                      "X": Xall[torch.as_tensor(image, device=DEV)], "a_t": torch.as_tensor(a, device=DEV)}

    def plus(y, gen):
        return MU_H * (2.0 * y.float() - 1.0) + torch.randn(y.shape[0], device=DEV, generator=gen)

    def at(level):
        mu, sd = level_frame(level)
        return lambda y, gen: mu * (2.0 * y.float() - 1.0) + sd * torch.randn(y.shape[0], device=DEV, generator=gen)

    def cue(m, k, view=None):
        """Log-odds of a network on validation and test, with the frames of the k-th level."""
        return {part: log_odds(logits(m, d["X"], d["a_t"][k], delta, view=view)) for part, d in sets.items()}

    def accuracy(zz, part):
        return round(float(((zz > 0) == (sets[part]["y"] == 1)).mean()), 4)

    def described(by_level, info, with_h=None, extra=None):
        """Right answers of a cue at every level it is read at (test and validation)."""
        for k, level in enumerate(LEVELS):
            if k not in by_level:
                continue
            for part in sets:
                info[f"{part}_acc_d{level:g}"] = accuracy(by_level[k][part], part)
                if with_h is not None:
                    more = extra[k][part] if extra is not None else 0.0
                    info[f"{part}_acc_sum_raw_d{level:g}"] = accuracy(with_h[k][part] + by_level[k][part] + more, part)
        return info

    clock = Clock()
    s_h = seed_of("trace", task, f"s{seed}", "H")
    t0 = clock()
    H = new_model(Net, s_h, n_out=2)
    train(H, XA, YA, epochs=ep, lr=lr, bs=bs, seed=s_h, frame=plus, delta=delta)
    z_h = {k: cue(H, k) for k in range(len(LEVELS))}
    info_h = described(z_h, {"torch_seed": s_h, "train_seconds": round(clock() - t0, 1), "half": "A"})
    frozen(H)

    z_c, info_c = {}, {}
    for k, level in enumerate(LEVELS):
        t0 = clock()
        C = new_model(Net, s_h, n_out=2)
        train(C, XA, YA, epochs=ep, lr=lr, bs=bs, seed=s_h, frame=at(level), delta=delta)
        z_c[k] = cue(C, k)
        info_c[f"C_{level:g}"] = described({k: z_c[k]}, {"torch_seed": s_h, "train_seconds": round(clock() - t0, 1),
                                                         "half": "A", "training_frame_level": level})
        del C

    s_m = seed_of("trace", task, f"s{seed}", "M")
    s_b = seed_of("trace", task, f"s{seed}", "ramo")
    for v, last in LAST_COLUMN.items():
        name = f"{task}_v{v}_s{seed}"
        if (out / f"{name}.npz").exists():
            continue
        t_unit = clock()
        view_m = inside_view(last)
        rec = {"unit": name, "task": task, "seed": seed, "v": v, "nets": {"H": info_h, **info_c}}

        t0 = clock()
        M = new_model(Net, s_m, n_out=2)
        train(M, XB, YB, epochs=ep, lr=lr, bs=bs, seed=s_m, frame=None, delta=delta, view=view_m)
        z_m = cue(M, 0, view_m)                              # the soloist does not see the frame: one cue for every level
        same = {k: z_m for k in range(len(LEVELS))}
        rec["nets"]["M"] = described(same, {"torch_seed": s_m, "train_seconds": round(clock() - t0, 1), "half": "B",
                                            "last_column": last, "sees_the_frame": False}, with_h=z_h)
        frozen(M)

        def branch_loss(x, y, out_, idx):
            mine = (out_[:, 1] - out_[:, 0]).float()
            total = frozen_log_odds(H, x) + frozen_log_odds(M, view_m(x)) + mine
            return F.binary_cross_entropy_with_logits(total, y.float())

        z_r = {}
        for t, level in enumerate(LEVELS):
            t0 = clock()
            branch = new_model(Net, s_b, n_out=2)
            train(branch, XB, YB, epochs=ep, lr=lr, bs=bs, seed=s_b, frame=at(level), delta=delta, loss_fn=branch_loss,
                  view=frame_only)
            z_r[t] = {k: cue(branch, k, frame_only) for k in range(len(LEVELS))}
            rec["nets"][f"R_{level:g}"] = described(
                z_r[t], {"torch_seed": s_b, "train_seconds": round(clock() - t0, 1), "half": "B",
                         "training_frame_level": level, "sees": "only the frame",
                         "sum": "right answers of z_H + z_M + z_branch at every level"}, with_h=z_h, extra=same)
            del branch
        del M

        arrays = {"levels": np.array(LEVELS, dtype=np.float32), "v": np.array(v, dtype=np.int32)}
        for part, d in sets.items():
            p = "" if part == "test" else "val_"
            arrays[p + "y"], arrays[p + "image"], arrays[p + "eps"], arrays[p + "a"] = d["y"], d["image"], d["eps"], d["a"]
            arrays[p + "z_H"] = np.stack([z_h[k][part] for k in range(len(LEVELS))])
            arrays[p + "z_C"] = np.stack([z_c[k][part] for k in range(len(LEVELS))])
            arrays[p + "z_M"] = z_m[part]
            arrays[p + "z_R"] = np.stack([np.stack([z_r[t][k][part] for k in range(len(LEVELS))])
                                          for t in range(len(LEVELS))])
        save_npz(out / f"{name}.npz", **arrays)
        rec["seconds"] = round(clock() - t_unit, 1)
        write_json(out / "units" / f"{name}.json", rec)
        print(json.dumps({"unit": name, "s": rec["seconds"], "H_d1": info_h["test_acc_d1"],
                          "M": rec["nets"]["M"]["test_acc_d0"],
                          "sum_d1": {"M": rec["nets"]["M"]["test_acc_sum_raw_d1"],
                                     "R_0": rec["nets"]["R_0"]["test_acc_sum_raw_d1"],
                                     "R_1": rec["nets"]["R_1"]["test_acc_sum_raw_d1"]}}), flush=True)


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
    for seed, task in [(s, t) for s in seeds for t in tasks][worker::workers]:
        if all((out / f"{task}_v{v}_s{seed}.npz").exists() for v in LAST_COLUMN):
            continue
        run_unit(D, cfg, out, seed, task)
    print("END", worker, flush=True)


if __name__ == "__main__":
    main()
