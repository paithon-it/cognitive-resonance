"""E3 bis, the recipe of the sum with real people (protocol section 7 bis, exploratory).

    python gpu/people_bis.py base OUT CONFIG SEED       # the base machine: people.py's base (same seed and settings)
    python gpu/people_bis.py finetune OUT CONFIG SEED   # both folds, the three recipes, starting from base SEED

Halves, fine-tuning schedule and seeds are those of E3 (people.py). Fold f: the base machine is fine-tuned on the
other half, then its 10-class logits are recorded on half f, once per recipe:
  - controllo: cross-entropy of the machine's logits (lambda = 0 of E3, again);
  - somma: cross-entropy of u + z_M, with u the people's cue and z_M the machine's logits;
  - mista: the sum of the two losses.
The people's cue on ten classes is u_c = log((v_c + 0.5) / (n + 5)), with v_c the votes for class c out of n.
The three recipes share seed, batch order and augmentation.

The CIFAR-10H counts (cifar10h-counts.npy, CC BY-NC-SA 4.0) are read from the data folder and never written out.
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from common import (DEV, Clock, Machine, cifar, data_root, logits, new_model, save_npz, seed_of, setup_torch, train,
                    versions, write_json)
from people import base, halves, params

RECIPES = ("controllo", "somma", "mista")


def people_cue_classes(counts: np.ndarray) -> np.ndarray:
    """u_c = log((v_c + 0.5) / (n + 5)) for every image and class: the log of a smoothed vote share."""
    counts = np.asarray(counts, dtype=np.float64)
    n = counts.sum(1, keepdims=True)
    return np.log((counts + 0.5) / (n + 5.0)).astype(np.float32)


def read_counts() -> np.ndarray:
    counts = np.load(Path(data_root()) / "cifar10h-counts.npy", allow_pickle=False)
    assert counts.shape == (10000, 10), counts.shape
    return counts


def finetune(out: Path, cfg: dict, seed: int) -> None:
    P = params(cfg)
    Xtr, Ytr, Xte, Yte = cifar()
    del Xtr, Ytr
    yte = Yte.cpu().numpy()
    counts = read_counts()
    U = torch.as_tensor(people_cue_classes(counts), device=DEV)
    hv = halves(yte)
    state = torch.load(out / "ckpt" / f"base_s{seed}.pt", map_location=DEV, weights_only=True)
    clock = Clock()
    rec = {"unit": f"finetune_s{seed}", "recipes": list(RECIPES), "folds": {}}
    for f in (0, 1):
        measure, tune = hv[f], hv[1 - f]
        tune_t = torch.as_tensor(tune, device=DEV)
        Xf, Yf, Uf = Xte[tune_t], Yte[tune_t], U[tune_t]
        measure_t = torch.as_tensor(measure, device=DEV)
        Xm, Um = Xte[measure_t], U[measure_t].cpu().numpy()

        def ce_sum(x, y, lg, idx):
            return F.cross_entropy(lg.float() + Uf[idx], y)

        def mixed(x, y, lg, idx):
            return F.cross_entropy(lg.float(), y) + ce_sum(x, y, lg, idx)

        losses = {"controllo": None, "somma": ce_sum, "mista": mixed}
        s = seed_of("people", f"s{seed}", f"fold{f}", "ft")          # E3's seed for this fold
        out_logits, frec = [], {"torch_seed": s, "tuned_on": f"half {1 - f}", "measured_on": f"half {f}"}
        for recipe in RECIPES:
            m = new_model(Machine, 0, n_out=10)
            m.load_state_dict(state)
            t0 = clock()
            train(m, Xf, Yf, epochs=P["ft_epochs"], lr=P["ft_lr"], bs=P["ft_batch_size"], seed=s,
                  wd=P["weight_decay"], schedule="cosine", loss_fn=losses[recipe])
            lg = logits(m, Xm).cpu().numpy().astype(np.float32)
            frec[recipe] = {
                "train_seconds": round(clock() - t0, 1),
                "acc_measured_half": round(float((lg.argmax(1) == yte[measure]).mean()), 4),
                "acc_sum_measured_half": round(float(((lg + Um).argmax(1) == yte[measure]).mean()), 4)}
            out_logits.append(lg)
            del m
        save_npz(out / f"fold{f}_s{seed}.npz", image=measure.astype(np.int32), y=yte[measure].astype(np.int8),
                 logits=np.stack(out_logits), recipes=np.array(RECIPES))
        rec["folds"][f"fold{f}"] = frec
        print(json.dumps({"fold": f, "seed": seed, **{k: v for k, v in frec.items() if k in RECIPES}}), flush=True)
    rec["seconds"] = clock()
    rec["cifar10h"] = {"votes_per_image_mean": round(float(counts.sum(1).mean()), 2),
                       "majority_agrees_with_label": round(float((counts.argmax(1) == yte).mean()), 4)}
    write_json(out / "units" / f"finetune_s{seed}.json", rec)


def main() -> None:
    setup_torch()
    mode, out, cfg, seed = sys.argv[1], Path(sys.argv[2]), json.loads(Path(sys.argv[3]).read_text()), int(sys.argv[4])
    (out / "units").mkdir(parents=True, exist_ok=True)
    write_json(out / "units" / "_versions.json", versions())
    if mode == "base":
        base(out, cfg, seed)
    elif mode == "finetune":
        finetune(out, cfg, seed)
    else:
        raise SystemExit(f"unknown mode {mode}")


if __name__ == "__main__":
    main()
