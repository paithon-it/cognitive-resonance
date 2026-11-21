"""E15, the threshold crossed (protocol section 21).

In E8 the generic machine sees a quarter, a half or the whole of the inside of the image: with a quarter and a half
the corrector alone beats it beside the judge, with the whole it does not. Here the same machine is trained at the
widths in between, so that the pairs cross the threshold of the proposition. Judge, corrector, frames and seeds are
those of E8: this script only trains the generic machine M of every task and seed at seven widths (the columns of the
inside it sees, of 28) and saves its log-odds on the validation and test presentations of E8.

    python gpu/crossing.py OUT CONFIG WORKER WORKERS [SEEDS] [TASKS]
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

from chart import LO, SEEDS, left_view
from common import (DEV, PRESENTATIONS, TASKS, Clock, Net, cifar, log_odds, logits, new_model, save_npz, seed_of,
                    setup_torch, task_split, train, versions, write_json)
from rephase_ter import VAR_A

COLUMNS = (14, 16, 18, 20, 22, 24, 26)        # 14 is the half of E8, trained again as a check; 28 would be the whole


def run_unit(D, cfg, out: Path, seed: int, task: str) -> None:
    Xtr, Ytr, Xte, Yte, ytr, yte = D
    name = f"{task}_s{seed}"
    ids = task_split(task, ytr, yte)
    c1 = TASKS[task][1]
    delta, ep, lr, bs = cfg["delta"], cfg["epochs"], cfg["lr"], cfg["batch_size"]
    XB, YB = Xtr[ids["half_b"]], (Ytr[ids["half_b"]] == c1).long()
    sets = {}
    for part, Xall, yall, idx in (("val", Xtr, ytr, ids["val"]), ("test", Xte, yte, ids["test"])):
        image = np.repeat(idx, PRESENTATIONS)
        eps = np.random.default_rng(seed_of("chart", task, f"s{seed}", f"{part}-a")).standard_normal(len(image))
        a = (np.sqrt(VAR_A) * eps).astype(np.float32)
        sets[part] = {"image": image.astype(np.int32), "a": a, "y": (yall[image] == c1).astype(np.int8),
                      "X": Xall[torch.as_tensor(image, device=DEV)], "a_t": torch.as_tensor(a, device=DEV)}

    def flat(y, gen):
        return np.sqrt(VAR_A) * torch.randn(y.shape[0], device=DEV, generator=gen)

    clock = Clock()
    s_m = seed_of("chart", task, f"s{seed}", "M")             # the seed of the generic machine of E8
    z, rec = {}, {"unit": name, "task": task, "seed": seed, "nets": {}}
    for cols in COLUMNS:
        view = left_view(LO + cols)
        t0 = clock()
        M = new_model(Net, s_m, n_out=2)
        train(M, XB, YB, epochs=ep, lr=lr, bs=bs, seed=s_m, frame=flat, delta=delta, view=view)
        info = {"torch_seed": s_m, "train_seconds": round(clock() - t0, 1), "columns": cols}
        for part, d in sets.items():
            zz = log_odds(logits(M, d["X"], d["a_t"], delta, view=view))
            z[(part, cols)] = zz
            info[f"{part}_acc"] = round(float(((zz > 0) == (d["y"] == 1)).mean()), 4)
        rec["nets"][f"c{cols}"] = info
        del M
    arrays = {"columns": np.array(COLUMNS, dtype=np.int32)}
    for part, d in sets.items():
        p = "" if part == "test" else "val_"
        arrays[p + "y"], arrays[p + "image"], arrays[p + "a"] = d["y"], d["image"], d["a"]
        arrays[p + "z_M"] = np.stack([z[(part, cols)] for cols in COLUMNS])
    save_npz(out / f"{name}.npz", **arrays)
    rec["seconds"] = clock()
    write_json(out / "units" / f"{name}.json", rec)
    print(json.dumps({"unit": name, "s": rec["seconds"], **{k: v["test_acc"] for k, v in rec["nets"].items()}}),
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
        if (out / f"{task}_s{seed}.npz").exists():
            continue
        run_unit(D, cfg, out, seed, task)
    print("END", worker, flush=True)


if __name__ == "__main__":
    main()
