"""E2 ter, the soloist that learns the person (protocol section 6 ter, exploratory).

For every task and seed, with the tasks, seeds, halves and frames of E2 and E2 bis (rephase.py, rephase_bis.py):
- H, the person: a "plus" network with mu = 1 trained on half A, then frozen (eval mode), as in E2 bis;
- M0: trained on half B with cross-entropy, frame a ~ N(0, 2) independent of the class, as in E2 bis;
- M0plus, the control: a copy of M0's weights fine-tuned on half B with its normal cross-entropy, same frame,
  10 epochs, learning rate one fifth of the pilot's, cosine decay to zero;
- Mdue: a copy of M0's weights fine-tuned with the same schedule, seed, batch order, crops, flips and frames as
  M0plus; the loss is its own cross-entropy plus the binary cross-entropy of the summed log-odds
  (out1 - out0) + z_H, with z_H from the frozen H on the same presentation, without gradient, weight one, no
  standardisation (the second term is the loss of Mpair in E2 bis);
- Mramo: M0 frozen (eval mode), plus a branch: a new network that sees only the frame (the interior of the image
  set to zero, as Mframe of E2 bis), trained from scratch with the pilot's hyper-parameters on the binary
  cross-entropy of z_H + z_M0 + z_branch; H and M0 see the whole framed presentation, without gradient. The
  machine's cue is z_M0 + z_branch.
On validation and test every image is shown 5 times with a ~ N(0, 2), the same eps as E2 and E2 bis.
z_M holds the log-odds of each machine on its own (for Mramo, z_M0 + z_branch), not the sum with H.

    python gpu/rephase_ter.py OUT CONFIG WORKER WORKERS [SEEDS] [TASKS]
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
MACHINES = ("M0", "M0plus", "Mdue", "Mramo")
FT_EPOCHS = 10                  # fine-tuning of M0plus and Mdue: 10 epochs ...
FT_LR_DIVISOR = 5               # ... at the pilot's learning rate divided by 5, cosine decay to zero


def frozen(m):
    m.eval()
    for p in m.parameters():
        p.requires_grad_(False)
    return m


def summed_log_odds(out, *nets_and_x):
    """(out1 - out0) plus the log-odds of each frozen network on its input, in float32."""
    total = (out[:, 1] - out[:, 0]).float()
    for net, x in nets_and_x:
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            z = net(x)
        total = total + (z[:, 1] - z[:, 0]).float()
    return total


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
        """Log-odds of m on validation and test; `add` is a key of z to add (Mramo: M0's log-odds)."""
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

    # H and M0: the same seeds, data and training as in E2 bis
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

    # M0plus and Mdue: copies of M0 fine-tuned with one seed, so with the same batches, crops, flips and frames
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

    # Mramo: frozen M0 plus a frame-only branch trained from scratch on the sum of the three
    def branch_loss(x, y, out, idx):
        return F.binary_cross_entropy_with_logits(summed_log_odds(out, (H, x), (M0, x)), y.float())

    s_b = seed_of("rephase", task, f"s{seed}", "ramo")
    t0 = clock()
    branch = new_model(Net, s_b, n_out=2)
    train(branch, XB, YB, epochs=ep, lr=lr, bs=bs, seed=s_b, frame=flat, delta=delta, loss_fn=branch_loss,
          view=frame_only)
    info = {"torch_seed": s_b, "train_seconds": round(clock() - t0, 1), "half": "B",
            "cue": "z_M0 + z_branch; the branch sees only the frame"}
    evaluate(branch, "Mramo", info, view=frame_only, add="M0")
    sum_accuracy("Mramo", info)
    rec["nets"]["Mramo"] = info
    del branch

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
