"""manifest.json for the stages of protocols 2.5 and 2.6 (E14, E15, E16), from the per-unit records copied back
from the GPU machine into results/raw/<stage>/units. gpu/manifest.py does the same for the earlier stages, on the
machine itself; this one needs neither torch nor the folder of the pilot, whose configuration it reads from the
manifest of E10 to E12 (the runs used the same file).

    python analysis/manifest_later.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ROOT  # noqa: E402

RAW = ROOT / "results" / "raw"


def units_of(stage: str) -> dict:
    return {p.stem: json.loads(p.read_text()) for p in sorted((RAW / stage / "units").glob("*.json"))}


def base(stage: str, protocol: str, experiment: str) -> dict:
    u = units_of(stage)
    return {"protocol": protocol, "experiment": experiment, "versions": u.get("_versions"), "stage_time": u.get("_stage"),
            "files": sorted(p.name for p in (RAW / stage).glob("*.npz"))}, {k: v for k, v in u.items() if not k.startswith("_")}


def bench_block() -> dict:
    t = json.loads((RAW / "trace" / "manifest.json").read_text())
    return {"pilot": {k: t["pilot"][k] for k in ("epochs", "lr", "batch_size", "delta")} | {"read_from": "results/raw/trace/manifest.json"},
            "normalisation": t["normalisation"], "frame": t["frame"],
            "hyperparameters": {k: t["hyperparameters"][k] for k in ("network", "optimiser", "augmentation", "max_lr", "batch_size")}}


def shortcut() -> dict:
    m, u = base("shortcut", "PROTOCOL.it.md v2.5 (2026-11-21)", "E14, what a shortcut leaves unlearned (protocol section 20)")
    m.update(bench_block())
    m.update({"file_pattern": "{task}_e{epochs}_s{seed}.npz", "seeds": sorted({x["seed"] for x in u.values()}),
              "epochs": sorted({x["epochs"] for x in u.values()}), "strengths": [0.0, 1.0, 1.5, 2.0],
              "keys": {"y": "0/1", "image": "index in the CIFAR-10 test set", "a": "frame value shown, shape (4, N), one row per strength",
                       "z": "log-odds of the network as it is, shape (4, N)",
                       "z_last": "log-odds of a new last layer: a logistic regression on the 192 features before the last "
                                 "layer, fitted on the validation presentations (the frame there says nothing), penalty 0.001",
                       "val_*": "the same on validation"},
              "torch_seeds": "those of the plus networks of E1: crc32('bench/<task>/s<seed>/plus') & 0x7fffffff, the same for every strength and length",
              "control_accuracy": {k: x["nets"] for k, x in u.items()}, "unit_seconds": {k: x["seconds"] for k, x in u.items()},
              "deviations": []})
    return m


def crossing() -> dict:
    m, u = base("crossing", "PROTOCOL.it.md v2.5 (2026-11-21)", "E15, the threshold crossed (protocol section 21)")
    m.update(bench_block())
    m.update({"file_pattern": "{task}_s{seed}.npz", "seeds": sorted({x["seed"] for x in u.values()}), "columns": [14, 16, 18, 20, 22, 24, 26],
              "keys": {"y": "0/1", "image": "index in the CIFAR-10 test set", "a": "frame value shown, as in E8",
                       "z_M": "the generic machine trained again, shape (7, N): one row per width; it sees the first columns of the inside and no frame",
                       "val_*": "the same on validation"},
              "judge_and_corrector": "those of E8, read from results/raw/chart (not trained again)",
              "torch_seeds": "that of the generic machine of E8: crc32('chart/<task>/s<seed>/M') & 0x7fffffff, the same at every width",
              "control_accuracy": {k: x["nets"] for k, x in u.items()}, "unit_seconds": {k: x["seconds"] for k, x in u.items()},
              "deviations": []})
    return m


def judge() -> dict:
    m, u = base("judge", "PROTOCOL.it.md v2.6 (2026-11-21)", "E16, a judge with a bias nobody planted (protocol section 22)")
    m.update({"file_pattern": "{model}_{split}.npz", "data": "cais/mmlu (MIT licence), configuration all, splits validation and test",
              "keys": {"y": "1 if the right option is A", "a": "the opinion the judge meets: +1 for A, -1 for B, drawn at random",
                       "g_none": "log-odds of A against B at the first token of the answer, with no opinion",
                       "g_a": "the same with 'I think the answer is A.'", "g_b": "the same with 'I think the answer is B.'",
                       "index": "row of the question in the split", "subject": "MMLU subject"},
              "seeds": "numpy default_rng(sha256('judge|<split>')[:4]): which questions, which wrong option, which order, which opinion",
              "runs": {k: {a: b for a, b in x.items() if a != "example_prompt"} for k, x in u.items()},
              "example_prompt": next(iter(u.values()))["example_prompt"],
              "deviations": ["right answers counted also with a threshold learned on the first half of the questions: added "
                             "after the exploration and before the confirmation (PROTOCOL, Deviations)"]})
    return m


def main() -> None:
    for stage, make in (("shortcut", shortcut), ("crossing", crossing), ("judge", judge)):
        m = make()
        (RAW / stage / "manifest.json").write_text(json.dumps(m, indent=1) + "\n")
        print("manifest", stage, len(m["files"]), "files")


if __name__ == "__main__":
    main()
