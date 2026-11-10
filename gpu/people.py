"""E3, real people (protocol section 7): CIFAR-10H votes against a ResNet-18, and a machine tuned into antiphase.

    python gpu/people.py base OUT CONFIG SEED       # the base machine: ResNet-18 on the 50,000 training images
    python gpu/people.py finetune OUT CONFIG SEED   # both folds, every lambda, starting from base SEED

The 10,000 test images are split in two halves, stratified by class (fixed split). Fold f: the base machine is
fine-tuned on the other half with cross-entropy plus lambda times the Pearson correlation, in the batch and after
removing the means of the true classes in the batch, between
  - the machine's cue towards the true class: true logit minus logsumexp of the other nine, and
  - the people's cue: logit((h + 0.5) / (n + 1)), with h correct votes out of n;
then its 10-class logits are recorded on half f. The fine-tuning is the same for every lambda (same seed, batch
order and augmentation); lambda = 0 is the control.

The CIFAR-10H counts (cifar10h-counts.npy, CC BY-NC-SA 4.0) are read from the data folder and never written out.
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

from common import (DEV, Clock, Machine, batch_corr, cifar, data_root, logits, new_model, save_npz, seed_of,
                    setup_torch, train, versions, write_json)

LAMBDAS = (0.0, 1.0, 3.0, 10.0)
DEFAULTS = {"base_epochs": 40, "base_lr": 0.2, "base_batch_size": 256,
            "ft_epochs": 8, "ft_lr": 0.01, "ft_batch_size": 256, "ft_schedule": "cosine to zero", "weight_decay": 5e-4}


def params(cfg: dict) -> dict:
    p = dict(DEFAULTS)
    p.update(cfg.get("people", {}))
    return p


def halves(yte: np.ndarray):
    rng = np.random.default_rng(seed_of("people", "halves"))
    h = ([], [])
    for c in range(10):
        idx = rng.permutation(np.flatnonzero(yte == c))
        h[0].append(idx[: len(idx) // 2])
        h[1].append(idx[len(idx) // 2:])
    return [np.sort(np.concatenate(x)) for x in h]


def people_cue(yte: np.ndarray) -> np.ndarray:
    counts = np.load(Path(data_root()) / "cifar10h-counts.npy")
    assert counts.shape == (10000, 10), counts.shape
    n = counts.sum(1)
    h = counts[np.arange(10000), yte]
    p = (h + 0.5) / (n + 1)
    return np.log(p / (1 - p)).astype(np.float32), counts


def base(out: Path, cfg: dict, seed: int) -> None:
    P = params(cfg)
    Xtr, Ytr, Xte, Yte = cifar()
    clock = Clock()
    s = seed_of("people", f"s{seed}", "base")
    m = new_model(Machine, s, n_out=10)
    train(m, Xtr, Ytr, epochs=P["base_epochs"], lr=P["base_lr"], bs=P["base_batch_size"], seed=s,
          wd=P["weight_decay"])
    seconds = clock()
    lg = logits(m, Xte).cpu().numpy().astype(np.float32)
    yte = Yte.cpu().numpy()
    (out / "ckpt").mkdir(parents=True, exist_ok=True)
    torch.save(m.state_dict(), out / "ckpt" / f"base_s{seed}.pt")
    save_npz(out / f"base_s{seed}.npz", image=np.arange(10000, dtype=np.int32), y=yte.astype(np.int8), logits=lg)
    acc = float((lg.argmax(1) == yte).mean())
    rec = {"unit": f"base_s{seed}", "torch_seed": s, "train_seconds": seconds, "test_acc": round(acc, 4)}
    write_json(out / "units" / f"base_s{seed}.json", rec)
    print(json.dumps(rec), flush=True)


def finetune(out: Path, cfg: dict, seed: int) -> None:
    P = params(cfg)
    Xtr, Ytr, Xte, Yte = cifar()
    del Xtr, Ytr
    yte = Yte.cpu().numpy()
    cue_np, counts = people_cue(yte)
    cue = torch.as_tensor(cue_np, device=DEV)
    hv = halves(yte)
    state = torch.load(out / "ckpt" / f"base_s{seed}.pt", map_location=DEV, weights_only=True)
    clock = Clock()
    rec = {"unit": f"finetune_s{seed}", "folds": {}}
    for f in (0, 1):
        measure, tune = hv[f], hv[1 - f]
        tune_t = torch.as_tensor(tune, device=DEV)
        Xf, Yf, cf = Xte[tune_t], Yte[tune_t], cue[tune_t]
        Xm = Xte[torch.as_tensor(measure, device=DEV)]
        s = seed_of("people", f"s{seed}", f"fold{f}", "ft")
        out_logits, frec = [], {"torch_seed": s, "tuned_on": f"half {1 - f}", "measured_on": f"half {f}"}
        for lam in LAMBDAS:
            corr_log = []

            def extra(x, y, lg, idx, lam=lam, corr_log=corr_log):
                lgf = lg.float()
                true = lgf.gather(1, y[:, None])[:, 0]
                others = torch.logsumexp(lgf.scatter(1, y[:, None], float("-inf")), dim=1)
                r = batch_corr(true - others, cf[idx], y, 10)
                corr_log.append(r.detach())
                return lam * r if lam > 0 else torch.zeros((), device=DEV)

            m = new_model(Machine, 0, n_out=10)
            m.load_state_dict(state)
            t0 = clock()
            train(m, Xf, Yf, epochs=P["ft_epochs"], lr=P["ft_lr"], bs=P["ft_batch_size"], seed=s,
                  wd=P["weight_decay"], schedule="cosine", extra=extra)
            lg = logits(m, Xm).cpu().numpy().astype(np.float32)
            per_epoch = len(Xf) // P["ft_batch_size"]
            frec[f"lambda{lam:g}"] = {
                "train_seconds": round(clock() - t0, 1),
                "batch_corr_last_epoch": round(float(torch.stack(corr_log[-per_epoch:]).mean()), 4),
                "acc_measured_half": round(float((lg.argmax(1) == yte[measure]).mean()), 4)}
            out_logits.append(lg)
            del m
        save_npz(out / f"fold{f}_s{seed}.npz", image=measure.astype(np.int32), y=yte[measure].astype(np.int8),
                 logits=np.stack(out_logits), lambdas=np.array(LAMBDAS, dtype=np.float32))
        rec["folds"][f"fold{f}"] = frec
        print(json.dumps({"fold": f, "seed": seed, **{k: v for k, v in frec.items() if k.startswith("lambda")}}),
              flush=True)
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
