"""E7, different views (protocol section 13).

In E1 to E6 the machine sees everything the person sees. Here each judge sees half of the image inside the
frame, and both see the whole frame. The person H always sees the left half (columns 2 to 15). The machine sees
the left half too (overlap 100 per cent), the top half (rows 2 to 15: it shares the top-left quarter with H,
overlap 50 per cent) or the right half (columns 16 to 29, overlap 0). Every half goes through the centre of the
image, so that none is the better half by construction. What a judge does not see is set to zero in normalised
units, after the frame has been drawn.

For every task and seed:
- H, the person: a "plus" network with mu = 1 trained on half A of the training images, on its half of the image,
  then frozen (eval mode). It is the same network at the three overlaps;
and, for every overlap:
- M0, the soloist: trained on half B on its half of the image with cross-entropy, frame a ~ N(0, 2) independent
  of the class;
- Mramo: M0 frozen (eval mode) plus a branch that sees only the frame, trained from scratch on the binary
  cross-entropy of z_H + z_M0 + z_branch (weight one, raw log-odds). The machine's cue is z_M0 + z_branch;
- Mpair: a new network on the machine's half of the image, trained from scratch on the binary cross-entropy of
  z_H + z_M (the recipe of Mpair in E2 bis). Same seed, batch order, crops, flips and frames as M0.
M0, the branch and Mpair use the same seeds at the three overlaps: only what the machine sees changes.
On validation and test every image is shown 5 times with a ~ N(0, 2), the same frames at the three overlaps.

    python gpu/views.py OUT CONFIG WORKER WORKERS [SEEDS] [TASKS]
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

LO, MID, HI = FRAME_WIDTH, 16, 32 - FRAME_WIDTH       # the interior is rows and columns 2 to 29; 16 halves it
HALVES = {"left": (LO, HI, LO, MID), "top": (LO, MID, LO, HI), "right": (LO, HI, MID, HI)}   # rows, then columns
VIEW_H = "left"
VIEW_M = {100: "left", 50: "top", 0: "right"}         # overlap with H's half, per cent -> the machine's half
SEEDS = (2, 3, 4)
MACHINES = ("M0", "Mramo", "Mpair")


def half_view(name: str):
    """What a judge sees: the whole frame and one half of the interior; the rest is zero."""
    r0, r1, c0, c1 = HALVES[name]
    mask = border_mask().clone()
    mask[..., r0:r1, c0:c1] = 1
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
        eps = np.random.default_rng(seed_of("views", task, f"s{seed}", f"{part}-a")).standard_normal(len(image))
        a = (np.sqrt(VAR_A) * eps).astype(np.float32)
        sets[part] = {"image": image.astype(np.int32), "a": a, "y": (yall[image] == c1).astype(np.int8),
                      "X": Xall[torch.as_tensor(image, device=DEV)], "a_t": torch.as_tensor(a, device=DEV)}

    def plus(y, gen):
        return MU_H * (2.0 * y.float() - 1.0) + torch.randn(y.shape[0], device=DEV, generator=gen)

    def flat(y, gen):
        return np.sqrt(VAR_A) * torch.randn(y.shape[0], device=DEV, generator=gen)

    def cue(m, view):
        return {part: log_odds(logits(m, d["X"], d["a_t"], delta, view=view)) for part, d in sets.items()}

    def accuracy(zz, part):
        return round(float(((zz > 0) == (sets[part]["y"] == 1)).mean()), 4)

    clock = Clock()
    view_h = half_view(VIEW_H)
    s_h = seed_of("views", task, f"s{seed}", "H")
    t0 = clock()
    H = new_model(Net, s_h, n_out=2)
    train(H, XA, YA, epochs=ep, lr=lr, bs=bs, seed=s_h, frame=plus, delta=delta, view=view_h)
    z_h = cue(H, view_h)
    info_h = {"torch_seed": s_h, "train_seconds": round(clock() - t0, 1), "half": "A", "sees": VIEW_H,
              **{f"{part}_acc": accuracy(z_h[part], part) for part in sets}}
    frozen(H)

    s_m = seed_of("views", task, f"s{seed}", "M")
    s_b = seed_of("views", task, f"s{seed}", "ramo")
    for overlap, sees in VIEW_M.items():
        name = f"{task}_o{overlap}_s{seed}"
        if (out / f"{name}.npz").exists():
            continue
        t_unit = clock()
        view_m = half_view(sees)
        rec = {"unit": name, "task": task, "seed": seed, "overlap": overlap, "nets": {"H": info_h}}
        z = {}

        def record(key, zz, info):
            for part in sets:
                z[(part, key)] = zz[part]
                info[f"{part}_acc"] = accuracy(zz[part], part)
                info[f"{part}_acc_sum_raw"] = accuracy(z_h[part] + zz[part], part)
            rec["nets"][key] = info

        t0 = clock()
        M0 = new_model(Net, s_m, n_out=2)
        train(M0, XB, YB, epochs=ep, lr=lr, bs=bs, seed=s_m, frame=flat, delta=delta, view=view_m)
        z_m0 = cue(M0, view_m)
        record("M0", z_m0, {"torch_seed": s_m, "train_seconds": round(clock() - t0, 1), "half": "B",
                            "sees": sees})
        frozen(M0)

        def branch_loss(x, y, out_, idx):
            mine = (out_[:, 1] - out_[:, 0]).float()
            total = frozen_log_odds(H, view_h(x)) + frozen_log_odds(M0, view_m(x)) + mine
            return F.binary_cross_entropy_with_logits(total, y.float())

        t0 = clock()
        branch = new_model(Net, s_b, n_out=2)
        train(branch, XB, YB, epochs=ep, lr=lr, bs=bs, seed=s_b, frame=flat, delta=delta, loss_fn=branch_loss,
              view=frame_only)
        z_br = cue(branch, frame_only)
        info = {"torch_seed": s_b, "train_seconds": round(clock() - t0, 1), "half": "B",
                "cue": "z_M0 + z_branch; the branch sees only the frame",
                **{f"{part}_acc_branch_alone": accuracy(z_br[part], part) for part in sets}}
        record("Mramo", {part: z_m0[part] + z_br[part] for part in sets}, info)
        del branch

        def pair_loss(x, y, out_, idx):
            total = frozen_log_odds(H, view_h(x)) + (out_[:, 1] - out_[:, 0]).float()
            return F.binary_cross_entropy_with_logits(total, y.float())

        t0 = clock()
        Mpair = new_model(Net, s_m, n_out=2)
        train(Mpair, XB, YB, epochs=ep, lr=lr, bs=bs, seed=s_m, frame=flat, delta=delta, loss_fn=pair_loss,
              view=view_m)
        record("Mpair", cue(Mpair, view_m), {"torch_seed": s_m, "train_seconds": round(clock() - t0, 1), "half": "B",
                                             "sees": sees})
        del Mpair, M0

        arrays = {"machines": np.array(MACHINES), "overlap": np.array(overlap, dtype=np.int32)}
        for part, d in sets.items():
            p = "" if part == "test" else "val_"
            arrays[p + "y"], arrays[p + "image"], arrays[p + "a"] = d["y"], d["image"], d["a"]
            arrays[p + "z_H"] = z_h[part]
            arrays[p + "z_M"] = np.stack([z[(part, k)] for k in MACHINES])
        save_npz(out / f"{name}.npz", **arrays)
        rec["seconds"] = round(clock() - t_unit, 1)
        write_json(out / "units" / f"{name}.json", rec)
        print(json.dumps({"unit": name, "s": rec["seconds"], **{k: v["test_acc"] for k, v in rec["nets"].items()},
                          **{f"{k}_sum": v["test_acc_sum_raw"] for k, v in rec["nets"].items() if k != "H"}}),
              flush=True)


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
        if all((out / f"{task}_o{o}_s{seed}.npz").exists() for o in VIEW_M):
            continue
        run_unit(D, cfg, out, seed, task)
    print("END", worker, flush=True)


if __name__ == "__main__":
    main()
