"""Writes manifest.json for one experiment folder from the per-unit records (protocol section 11).

    python gpu/manifest.py OUT_ROOT bench|rephase|people [DEVIATION ...]

OUT_ROOT is the folder with pilot/ and the experiment folders; the manifest lands in OUT_ROOT/<experiment>/.
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import CLASSES, FRAME_WIDTH, MEAN, N_VAL_PER_CLASS, PRESENTATIONS, SPLIT_SEED, STD, TASKS  # noqa: E402

PROTOCOL = "PROTOCOL.it.md v1.1 (2026-11-10)"
NET = ("Net of turing-meter experiments/angle-between-errors/gpu/silicon.py with a 2-way head: conv3x3 48-96, "
       "maxpool, 96-192, maxpool, 192, global average pool, linear; BatchNorm and ReLU after every convolution")
OPTIM = "SGD, Nesterov momentum 0.9, weight decay 5e-4, one-cycle schedule (PyTorch OneCycleLR defaults), bf16 autocast"
AUG = "random 32x32 crop of the image reflect-padded by 4 pixels, random horizontal flip"
FRAME = (f"the outer {FRAME_WIDTH}-pixel border of all three channels is set to a * delta, in normalised units "
         "(after per-channel normalisation), applied after crop and flip, without clipping")


def load_units(folder: Path):
    units = {}
    for p in sorted((folder / "units").glob("*.json")):
        units[p.stem] = json.loads(p.read_text())
    return units


def common_block(root: Path, units: dict) -> dict:
    cfg = json.loads((root / "pilot" / "config.json").read_text())
    return {"protocol": PROTOCOL, "versions": units.get("_versions"), "stage_time": units.get("_stage"),
            "normalisation": {"mean": MEAN, "std": STD}, "cifar10_classes": list(CLASSES),
            "pilot": {k: cfg[k] for k in ("epochs", "lr", "batch_size", "delta", "rule", "grid_mean_val_acc")}
            | {"probe": cfg["probe"], "timing": cfg["timing"]}}


def bench(root: Path, deviations) -> dict:
    folder = root / "bench"
    units = load_units(folder)
    cfg = json.loads((root / "pilot" / "config.json").read_text())
    rows = [u for k, u in units.items() if not k.startswith("_")]
    table = {}
    for u in rows:
        for role, info in u["nets"].items():
            key = f"{u['task']}|mu{u['mu']:g}|{role}"
            table.setdefault(key, {"test_acc": [], "val_acc": [], "train_seconds": []})
            for f in ("test_acc", "val_acc", "train_seconds"):
                table[key][f].append(info[f])
    acc = {k: {"test_acc_mean": round(float(np.mean(v["test_acc"])), 4), "test_acc_by_seed": v["test_acc"],
               "val_acc_mean": round(float(np.mean(v["val_acc"])), 4)} for k, v in sorted(table.items())}
    m = common_block(root, units)
    m.update({
        "experiment": "E1, the bench (protocol section 5)",
        "files": sorted(p.name for p in folder.glob("*.npz")),
        "file_pattern": "{task}_mu{mu}_s{seed}.npz",
        "keys": {"y": "0/1, 1 = second class of the pair", "image": "index in the CIFAR-10 test set",
                 "a": "frame value shown, same for the three judges", "z_plus|z_minus|z_plus2":
                 "log-odds towards class 1 = logit of class 1 minus logit of class 0",
                 "val_*": "the same on validation; val_image is an index in the CIFAR-10 training set"},
        "layout": f"each image appears {PRESENTATIONS} times in a row (np.repeat), with independent frames",
        "tasks": {t: {"class0": CLASSES[c[0]], "class1": CLASSES[c[1]], "cifar_ids": list(c)} for t, c in TASKS.items()},
        "split": {"validation": f"{N_VAL_PER_CLASS} training images per class, drawn once per task with "
                                f"numpy default_rng(crc32('{SPLIT_SEED}/<task>')); the other 9,000 train",
                  "test": "all 2,000 CIFAR-10 test images of the two classes"},
        "mus": [0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0],
        "seeds": sorted({u["seed"] for u in rows}),
        "judges": {"plus": "training frame a ~ N(+mu, 1) on class 1, N(-mu, 1) on class 0",
                   "minus": "training frame a ~ N(-mu, 1) on class 1, N(+mu, 1) on class 0",
                   "plus2": "as plus, another seed"},
        "frame": FRAME, "delta": cfg["delta"],
        "test_and_validation_frames": "a = sqrt(1 + mu^2) * eps, eps ~ N(0, 1) from numpy default_rng(crc32("
                                      "'bench/<task>/s<seed>/<test|val>-a')): the same eps for every mu",
        "torch_seeds": "crc32('bench/<task>/s<seed>/<role>') & 0x7fffffff, the same for every mu; it seeds the "
                       "initialisation and one CUDA generator for batch order, crops, flips and training frames",
        "hyperparameters": {"network": NET, "optimiser": OPTIM, "augmentation": AUG, "epochs": cfg["epochs"],
                            "max_lr": cfg["lr"], "batch_size": cfg["batch_size"], "train_images": 9000,
                            "evaluation": "eval mode, fp32 with TF32 convolutions, no augmentation"},
        "control_accuracy": acc,
        "unit_seconds": {k: u["seconds"] for k, u in units.items() if not k.startswith("_")},
        "deviations": deviations,
    })
    return m


def rephase(root: Path, deviations) -> dict:
    folder = root / "rephase"
    units = load_units(folder)
    cfg = json.loads((root / "pilot" / "config.json").read_text())
    rows = [u for k, u in units.items() if not k.startswith("_")]
    acc = {}
    for u in rows:
        for role, info in u["nets"].items():
            acc.setdefault(f"{u['task']}|{role}", []).append(
                {"seed": u["seed"], "test_acc": info["test_acc"], "val_acc": info["val_acc"],
                 **({"batch_corr_last_epoch": info["batch_corr_last_epoch"]} if "batch_corr_last_epoch" in info else {})})
    m = common_block(root, units)
    m.update({
        "experiment": "E2, rephasing (protocol section 6)",
        "files": sorted(p.name for p in folder.glob("*.npz")),
        "file_pattern": "{task}_s{seed}.npz",
        "keys": {"y": "0/1", "image": "index in the CIFAR-10 test set", "a": "frame value shown, a ~ N(0, 2)",
                 "z_H": "person H: plus network, mu = 1, half A", "z_H2": "control H': plus network, mu = 1, half B",
                 "z_M": "machines, shape (len(lambdas), N), half B, frame independent of class",
                 "lambdas": "penalty weights", "val_*": "the same on validation (val_image: CIFAR-10 training index)"},
        "layout": f"each image appears {PRESENTATIONS} times in a row (np.repeat), with independent frames",
        "halves": "the 9,000 training images of the task (validation excluded) split per class in two halves of "
                  "2,250 + 2,250 (A = first half of the same permutation that draws the validation set)",
        "machine_training_frame": "a ~ N(0, 2), independent of the class",
        "penalty": "lambda * Pearson correlation in the batch between sgn*z_M and sgn*z_H (sgn = +1 for class 1, -1 "
                   "for class 0), after removing the mean of each true class in the batch; H frozen in eval mode, "
                   "forward on the same augmented and framed batch (bf16); lambda = 0 trains without the term",
        "frame": FRAME, "delta": cfg["delta"],
        "torch_seeds": "crc32('rephase/<task>/s<seed>/<H|H2|M>') & 0x7fffffff; every M_lambda of a unit shares "
                       "seed, initialisation, batch order, crops, flips and frames",
        "test_and_validation_frames": "a = sqrt(2) * eps, eps from numpy default_rng(crc32('rephase/<task>/s<seed>/"
                                      "<test|val>-a'))",
        "hyperparameters": {"network": NET, "optimiser": OPTIM, "augmentation": AUG, "epochs": cfg["epochs"],
                            "max_lr": cfg["lr"], "batch_size": cfg["batch_size"], "train_images": 4500},
        "control_accuracy": acc,
        "unit_seconds": {k: u["seconds"] for k, u in units.items() if not k.startswith("_")},
        "diagnostics": {p.stem: json.loads(p.read_text()) for p in sorted(folder.glob("diagnose_*.json"))} or None,
        "deviations": deviations,
    })
    return m


def people(root: Path, deviations) -> dict:
    folder = root / "people"
    units = load_units(folder)
    cfg = json.loads((root / "pilot" / "config.json").read_text())
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from people import DEFAULTS, LAMBDAS
    P = dict(DEFAULTS); P.update(cfg.get("people", {}))
    m = {"protocol": PROTOCOL, "versions": units.get("_versions"), "stage_time": units.get("_stage"),
         "experiment": "E3, real people (protocol section 7)",
         "files": sorted(p.name for p in folder.glob("*.npz")),
         "keys": {"base_s{seed}.npz": "image (0..9999, CIFAR-10 test index), y (0-9), logits (10000, 10) of the base",
                  "fold{f}_s{seed}.npz": "image (the CIFAR-10 test indices of half f), y, logits (len(lambdas), "
                                         "5000, 10) measured on half f after fine-tuning on the other half, lambdas"},
         "halves": "the 10,000 test images split per class 500 + 500 with numpy default_rng(crc32('people/halves'))",
         "machine": "ResNet-18 for CIFAR of silicon.py (Machine), trained on the 50,000 training images",
         "people_cue": "logit((h + 0.5) / (n + 1)), h correct votes out of n, from cifar10h-counts.npy",
         "machine_cue": "true logit minus logsumexp of the other nine",
         "penalty": "lambda * Pearson correlation in the batch between machine and people cues, after removing "
                    "the mean of each true class in the batch; lambda = 0 trains without the term",
         "lambdas": list(LAMBDAS),
         "hyperparameters": P | {"optimiser": "SGD Nesterov momentum 0.9, bf16 autocast; base: one-cycle; "
                                              "fine-tuning: cosine from ft_lr to zero, BatchNorm in training mode",
                                 "augmentation": AUG},
         "torch_seeds": "base: crc32('people/s<seed>/base'); fine-tuning: crc32('people/s<seed>/fold<f>/ft'), "
                        "shared by every lambda",
         "units": {k: v for k, v in units.items() if not k.startswith("_")},
         "cifar10h_licence": "CC BY-NC-SA 4.0; the counts are not included here",
         "deviations": deviations}
    return m


def main() -> None:
    root, exp = Path(sys.argv[1]), sys.argv[2]
    deviations = list(sys.argv[3:])
    dev_file = root / "deviations.json"           # optional: {"bench": ["..."], "rephase": [...], "people": [...]}
    if dev_file.exists():
        deviations += json.loads(dev_file.read_text()).get(exp, [])
    m = {"bench": bench, "rephase": rephase, "people": people}[exp](root, deviations)
    (root / exp / "manifest.json").write_text(json.dumps(m, indent=1))
    print("manifest", exp, len(m["files"]), "files")


if __name__ == "__main__":
    main()
