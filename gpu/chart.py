"""E8, the bench of the chart (protocol section 14).

The judge in the person's seat sees the whole image, with the bias of the frame. The machine sees the frame and
only a part of the inside of the image: all of it (v = 100: the bench of E2 ter), the left half (v = 50) or the
left quarter (v = 25). Below 100 the judge knows something the machine does not see.

For every task and seed:
- H: a "plus" network with mu = 1 trained on half A on the whole image, then frozen (eval mode);
- H0: the same network without the bias: same seed and half, frame a ~ N(0, 2) independent of the class;
- Mblind: a network that sees only the frame, trained from scratch on half B on the binary cross-entropy of
  z_H + z_blind (the Mframe of E2 bis). It knows the judge and nothing of the task;
and, for every v:
- M: the soloist, trained on half B with cross-entropy on its part of the image, frame independent of the class;
- Mramo: M frozen plus a branch that sees only the frame, trained from scratch on the binary cross-entropy of
  z_H + z_M + z_branch. The machine's cue is z_M + z_branch.
M and the branch use the same seeds at the three values of v: only what the machine sees changes.
On validation and test every image is shown 5 times with a ~ N(0, 2), the same frames at the three values of v.

    python gpu/chart.py OUT CONFIG WORKER WORKERS [SEEDS] [TASKS]
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from common import (DEV, FRAME_WIDTH, PRESENTATIONS, TASKS, Clock, Net, border_mask, cifar, frame_only, log_odds,
                    logits, new_model, save_npz, seed_of, setup_torch, task_split, train, versions, write_json)
from rephase_ter import MU_H, VAR_A, frozen
from trust import frozen_log_odds

LO, HI = FRAME_WIDTH, 32 - FRAME_WIDTH           # the inside of the image is rows and columns 2 to 29
LAST_COLUMN = {100: HI, 50: LO + 14, 25: LO + 7}  # the machine sees the columns LO .. LAST_COLUMN - 1 of the inside
SEEDS = (2, 3, 4, 5, 6)
MACHINES = ("M", "Mramo", "Mblind")


def left_view(last: int):
    """What the machine sees: the whole frame and the columns of the inside up to `last` (excluded)."""
    mask = border_mask().clone()
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
        eps = np.random.default_rng(seed_of("chart", task, f"s{seed}", f"{part}-a")).standard_normal(len(image))
        a = (np.sqrt(VAR_A) * eps).astype(np.float32)
        sets[part] = {"image": image.astype(np.int32), "a": a, "y": (yall[image] == c1).astype(np.int8),
                      "X": Xall[torch.as_tensor(image, device=DEV)], "a_t": torch.as_tensor(a, device=DEV)}

    def plus(y, gen):
        return MU_H * (2.0 * y.float() - 1.0) + torch.randn(y.shape[0], device=DEV, generator=gen)

    def flat(y, gen):
        return np.sqrt(VAR_A) * torch.randn(y.shape[0], device=DEV, generator=gen)

    def cue(m, view=None):
        return {part: log_odds(logits(m, d["X"], d["a_t"], delta, view=view)) for part, d in sets.items()}

    def accuracy(zz, part):
        return round(float(((zz > 0) == (sets[part]["y"] == 1)).mean()), 4)

    def described(zz, info, with_h=None):
        for part in sets:
            info[f"{part}_acc"] = accuracy(zz[part], part)
            if with_h is not None:
                info[f"{part}_acc_sum_raw"] = accuracy(with_h[part] + zz[part], part)
        return info

    clock = Clock()
    s_h = seed_of("chart", task, f"s{seed}", "H")
    t0 = clock()
    H = new_model(Net, s_h, n_out=2)
    train(H, XA, YA, epochs=ep, lr=lr, bs=bs, seed=s_h, frame=plus, delta=delta)
    z_h = cue(H)
    info_h = described(z_h, {"torch_seed": s_h, "train_seconds": round(clock() - t0, 1), "half": "A"})
    frozen(H)

    t0 = clock()
    H0 = new_model(Net, s_h, n_out=2)
    train(H0, XA, YA, epochs=ep, lr=lr, bs=bs, seed=s_h, frame=flat, delta=delta)
    z_h0 = cue(H0)
    info_h0 = described(z_h0, {"torch_seed": s_h, "train_seconds": round(clock() - t0, 1), "half": "A",
                               "frame": "independent of the class"})
    del H0

    def blind_loss(x, y, out_, idx):
        total = frozen_log_odds(H, x) + (out_[:, 1] - out_[:, 0]).float()
        return F.binary_cross_entropy_with_logits(total, y.float())

    s_c = seed_of("chart", task, f"s{seed}", "blind")
    t0 = clock()
    blind = new_model(Net, s_c, n_out=2)
    train(blind, XB, YB, epochs=ep, lr=lr, bs=bs, seed=s_c, frame=flat, delta=delta, loss_fn=blind_loss,
          view=frame_only)
    z_blind = cue(blind, frame_only)
    info_blind = described(z_blind, {"torch_seed": s_c, "train_seconds": round(clock() - t0, 1), "half": "B",
                                     "sees": "only the frame"}, with_h=z_h)
    del blind

    s_m = seed_of("chart", task, f"s{seed}", "M")
    s_b = seed_of("chart", task, f"s{seed}", "ramo")
    for v, last in LAST_COLUMN.items():
        name = f"{task}_v{v}_s{seed}"
        if (out / f"{name}.npz").exists():
            continue
        t_unit = clock()
        view_m = left_view(last)
        rec = {"unit": name, "task": task, "seed": seed, "v": v,
               "nets": {"H": info_h, "H0": info_h0, "Mblind": info_blind}}

        t0 = clock()
        M = new_model(Net, s_m, n_out=2)
        train(M, XB, YB, epochs=ep, lr=lr, bs=bs, seed=s_m, frame=flat, delta=delta, view=view_m)
        z_m = cue(M, view_m)
        rec["nets"]["M"] = described(z_m, {"torch_seed": s_m, "train_seconds": round(clock() - t0, 1), "half": "B",
                                           "last_column": last}, with_h=z_h)
        frozen(M)

        def branch_loss(x, y, out_, idx):
            mine = (out_[:, 1] - out_[:, 0]).float()
            total = frozen_log_odds(H, x) + frozen_log_odds(M, view_m(x)) + mine
            return F.binary_cross_entropy_with_logits(total, y.float())

        t0 = clock()
        branch = new_model(Net, s_b, n_out=2)
        train(branch, XB, YB, epochs=ep, lr=lr, bs=bs, seed=s_b, frame=flat, delta=delta, loss_fn=branch_loss,
              view=frame_only)
        z_br = cue(branch, frame_only)
        z_ramo = {part: z_m[part] + z_br[part] for part in sets}
        rec["nets"]["Mramo"] = described(z_ramo, {"torch_seed": s_b, "train_seconds": round(clock() - t0, 1),
                                                  "half": "B", "cue": "z_M + z_branch; the branch sees only the frame"},
                                         with_h=z_h)
        del branch, M

        cues = {"M": z_m, "Mramo": z_ramo, "Mblind": z_blind}
        arrays = {"machines": np.array(MACHINES), "v": np.array(v, dtype=np.int32)}
        for part, d in sets.items():
            p = "" if part == "test" else "val_"
            arrays[p + "y"], arrays[p + "image"], arrays[p + "a"] = d["y"], d["image"], d["a"]
            arrays[p + "z_H"], arrays[p + "z_H0"] = z_h[part], z_h0[part]
            arrays[p + "z_M"] = np.stack([cues[k][part] for k in MACHINES])
        save_npz(out / f"{name}.npz", **arrays)
        rec["seconds"] = round(clock() - t_unit, 1)
        write_json(out / "units" / f"{name}.json", rec)
        print(json.dumps({"unit": name, "s": rec["seconds"], **{k: i["test_acc"] for k, i in rec["nets"].items()},
                          **{f"{k}_sum": i["test_acc_sum_raw"] for k, i in rec["nets"].items()
                             if "test_acc_sum_raw" in i}}), flush=True)


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
