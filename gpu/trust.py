"""E6, trust made to measure (protocol section 12).

For every task and seed, with the construction of E2 ter (rephase_ter.py) on seeds that no earlier run has used:
- H, the person: a "plus" network with mu = 1 trained on half A, then frozen (eval mode);
- M0: trained on half B with cross-entropy, frame a ~ N(0, 2) independent of the class;
- M0plus, the control soloist: a copy of M0 fine-tuned on half B with its own cross-entropy, 10 epochs, learning
  rate one fifth of the pilot's, cosine decay to zero;
- Mdue: a copy of M0 fine-tuned in the same way (same seed, batch order, crops, flips and frames) on its own
  cross-entropy plus the binary cross-entropy of the summed log-odds z_H + z_M;
- Mramo_w, for w in WEIGHTS: M0 frozen (eval mode) plus a branch, a new network that sees only the frame (the
  interior of the image set to zero), trained from scratch with the pilot's hyper-parameters on the binary
  cross-entropy of z_H + w * (z_M0 + z_branch). w is the weight, in raw log-odds, that the pair the branch is
  built for gives to the machine. The four branches share seed, batch order, crops, flips and frames: they differ
  only in w. The machine's cue is z_M0 + z_branch. Mramo_w1 is the Mramo of E2 ter.
On validation and test every image is shown 5 times with a ~ N(0, 2). z_M holds the log-odds of each machine on
its own, not the sum with H.

    python gpu/trust.py OUT CONFIG WORKER WORKERS [SEEDS] [TASKS]
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from common import (DEV, PRESENTATIONS, TASKS, Clock, Net, cifar, frame_only, log_odds, logits, new_model, save_npz,
                    seed_of, setup_torch, task_split, train, versions, write_json)
from rephase_ter import FT_EPOCHS, FT_LR_DIVISOR, MU_H, VAR_A, frozen, summed_log_odds

WEIGHTS = (0.5, 1.0, 2.0, 4.0)          # the weight each branch is built for
SEEDS = (2, 3, 4)                       # E1 to E5 used 0 and 1


def branch_name(w: float) -> str:
    return f"Mramo_w{w:g}"


MACHINES = ("M0", "M0plus", "Mdue") + tuple(branch_name(w) for w in WEIGHTS)


def frozen_log_odds(net, x):
    """Log-odds of a frozen network on x, without gradient, in float32."""
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        z = net(x)
    return (z[:, 1] - z[:, 0]).float()


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

    def evaluate(m, key, info, view=None, add=None):
        """Log-odds of m on validation and test; `add` is a key of z to add (a branch: M0's log-odds)."""
        for part, d in sets.items():
            zz = log_odds(logits(m, d["X"], d["a_t"], delta, view=view))
            if add is not None:
                info[f"{part}_acc_branch_alone"] = round(float(((zz > 0) == (d["y"] == 1)).mean()), 4)
                zz = z[(part, add)] + zz
            z[(part, key)] = zz
            info[f"{part}_acc"] = round(float(((zz > 0) == (d["y"] == 1)).mean()), 4)

    def sum_accuracy(key, info):
        for part, d in sets.items():
            s = z[(part, "H")] + z[(part, key)]
            info[f"{part}_acc_sum_raw"] = round(float(((s > 0) == (d["y"] == 1)).mean()), 4)

    # H and M0: the seed labels of E2, E2 bis and E2 ter, on seeds they never used
    s_h = seed_of("rephase", task, f"s{seed}", "H")
    t0 = clock()
    H = new_model(Net, s_h, n_out=2)
    train(H, XA, YA, epochs=ep, lr=lr, bs=bs, seed=s_h, frame=plus, delta=delta)
    info = {"torch_seed": s_h, "train_seconds": round(clock() - t0, 1), "half": "A"}
    evaluate(H, "H", info)
    rec["nets"]["H"] = info
    frozen(H)

    s_m = seed_of("rephase", task, f"s{seed}", "M")
    t0 = clock()
    M0 = new_model(Net, s_m, n_out=2)
    train(M0, XB, YB, epochs=ep, lr=lr, bs=bs, seed=s_m, frame=flat, delta=delta)
    info = {"torch_seed": s_m, "train_seconds": round(clock() - t0, 1), "half": "B"}
    evaluate(M0, "M0", info)
    sum_accuracy("M0", info)
    rec["nets"]["M0"] = info
    state_m0 = {k: v.detach().clone() for k, v in M0.state_dict().items()}
    frozen(M0)

    # M0plus and Mdue, as in E2 ter
    def mixed_loss(x, y, out, idx):
        return F.cross_entropy(out.float(), y) + F.binary_cross_entropy_with_logits(summed_log_odds(out, (H, x)),
                                                                                     y.float())

    s_ft = seed_of("rephase", task, f"s{seed}", "ter")
    ft_lr = lr / FT_LR_DIVISOR
    for key, loss_fn in (("M0plus", None), ("Mdue", mixed_loss)):
        t0 = clock()
        m = new_model(Net, s_m, n_out=2)
        m.load_state_dict(state_m0)
        train(m, XB, YB, epochs=FT_EPOCHS, lr=ft_lr, bs=bs, seed=s_ft, frame=flat, delta=delta, loss_fn=loss_fn,
              schedule="cosine")
        info = {"torch_seed": s_ft, "train_seconds": round(clock() - t0, 1), "half": "B", "start": "M0",
                "epochs": FT_EPOCHS, "lr": ft_lr, "schedule": "cosine to zero"}
        evaluate(m, key, info)
        sum_accuracy(key, info)
        rec["nets"][key] = info
        del m

    # the branches: one per weight, the same seed for all
    s_b = seed_of("rephase", task, f"s{seed}", "ramo")
    for w in WEIGHTS:
        def branch_loss(x, y, out, idx, w=w):
            mine = (out[:, 1] - out[:, 0]).float()
            total = frozen_log_odds(H, x) + w * (frozen_log_odds(M0, x) + mine)
            return F.binary_cross_entropy_with_logits(total, y.float())

        key = branch_name(w)
        t0 = clock()
        branch = new_model(Net, s_b, n_out=2)
        train(branch, XB, YB, epochs=ep, lr=lr, bs=bs, seed=s_b, frame=flat, delta=delta, loss_fn=branch_loss,
              view=frame_only)
        info = {"torch_seed": s_b, "train_seconds": round(clock() - t0, 1), "half": "B", "built_for_weight": w,
                "cue": "z_M0 + z_branch; the branch sees only the frame"}
        evaluate(branch, key, info, view=frame_only, add="M0")
        sum_accuracy(key, info)
        rec["nets"][key] = info
        del branch

    arrays = {"machines": np.array(MACHINES), "built_for_weight": np.array(WEIGHTS, dtype=np.float32)}
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
    seeds = [int(s) for s in sys.argv[5].split(",")] if len(sys.argv) > 5 else list(SEEDS)
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
