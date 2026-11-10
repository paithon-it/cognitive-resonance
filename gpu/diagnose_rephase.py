"""Diagnostic for E2 (not part of the raw-data contract): how M_lambda behaves on its own training images.

Retrains H and a few M_lambda of one unit exactly as rephase.py does (same seeds) and reports, on the training
half B (frames a ~ N(0, 2), no augmentation) and on the test images, the accuracy of M and the within-class
correlation of its cue with H's cue.

    python gpu/diagnose_rephase.py OUT CONFIG TASK SEED [LAMBDAS]
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

from common import (DEV, TASKS, Net, batch_corr, cifar, log_odds, logits, new_model, seed_of, setup_torch,
                    task_split, train, write_json)


def main() -> None:
    setup_torch()
    out, cfg = Path(sys.argv[1]), json.loads(Path(sys.argv[2]).read_text())
    task, seed = sys.argv[3], int(sys.argv[4])
    lambdas = [float(x) for x in sys.argv[5].split(",")] if len(sys.argv) > 5 else [0.0, 0.3, 3.0]
    delta, ep, lr, bs = cfg["delta"], cfg["epochs"], cfg["lr"], cfg["batch_size"]
    Xtr, Ytr, Xte, Yte = cifar()
    ids = task_split(task, Ytr.cpu().numpy(), Yte.cpu().numpy())
    c1 = TASKS[task][1]
    XA, YA = Xtr[ids["half_a"]], (Ytr[ids["half_a"]] == c1).long()
    XB, YB = Xtr[ids["half_b"]], (Ytr[ids["half_b"]] == c1).long()
    XT, YT = Xte[torch.as_tensor(ids["test"], device=DEV)], (Yte[torch.as_tensor(ids["test"], device=DEV)] == c1).long()
    g = torch.Generator(device=DEV).manual_seed(seed_of("diagnose", task, seed))
    aB = np.sqrt(2.0) * torch.randn(len(XB), device=DEV, generator=g)
    aT = np.sqrt(2.0) * torch.randn(len(XT), device=DEV, generator=g)

    def plus(y, gen):
        return (2.0 * y.float() - 1.0) + torch.randn(y.shape[0], device=DEV, generator=gen)

    def flat(y, gen):
        return np.sqrt(2.0) * torch.randn(y.shape[0], device=DEV, generator=gen)

    sH = seed_of("rephase", task, f"s{seed}", "H")
    H = new_model(Net, sH, n_out=2)
    train(H, XA, YA, epochs=ep, lr=lr, bs=bs, seed=sH, frame=plus, delta=delta)
    H.eval()
    for p in H.parameters():
        p.requires_grad_(False)
    zH = {"B": torch.as_tensor(log_odds(logits(H, XB, aB, delta)), device=DEV),
          "test": torch.as_tensor(log_odds(logits(H, XT, aT, delta)), device=DEV)}
    res = {"task": task, "seed": seed}
    s_m = seed_of("rephase", task, f"s{seed}", "M")
    for lam in lambdas:
        def extra(x, y, out, idx, lam=lam):
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                zh = H(x)
            sgn = 2.0 * y.float() - 1.0
            r = batch_corr(sgn * (out[:, 1] - out[:, 0]).float(), sgn * (zh[:, 1] - zh[:, 0]).float(), y, 2)
            return lam * r if lam > 0 else torch.zeros((), device=DEV)

        m = new_model(Net, s_m, n_out=2)
        train(m, XB, YB, epochs=ep, lr=lr, bs=bs, seed=s_m, frame=flat, delta=delta, extra=extra)
        row = {}
        for part, X, Y, a in (("train_half_B", XB, YB, aB), ("test", XT, YT, aT)):
            lg = logits(m, X, a, delta)
            z = lg[:, 1] - lg[:, 0]
            key = "B" if part == "train_half_B" else "test"
            row[part] = {"acc": round(float(((z > 0).long() == Y).float().mean()), 4),
                         "cross_entropy": round(float(torch.nn.functional.cross_entropy(lg, Y)), 4),
                         "r_with_H": round(float(batch_corr(z, zH[key], Y, 2)), 4)}
        res[f"lambda{lam:g}"] = row
        print(json.dumps({f"lambda{lam:g}": row}), flush=True)
    write_json(out / f"diagnose_{task}_s{seed}.json", res)


if __name__ == "__main__":
    main()
