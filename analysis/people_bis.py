"""E3 bis (protocol § 7 bis, v1.4, exploratory): the recipe of the sum with real people.

Reads results/raw/people_bis/: fold{f}_s{seed}.npz with the logits of the three recipes
(controllo, somma, mista) on the half not used for their fine-tuning, and base_s{seed}.npz.
The measures are those of E3, with the functions of analysis/people.py: the angle with
turing-meter's `compare_votes` (m = 10), the machine's accuracy, the apparent turings and
the turing of the group with the combiner. In addition, the accuracy of argmax(u + z_M),
with u the people's cue of § 7 bis, next to that of the plurality vote.

Paired differences from the control, for the angle, the apparent turings and the turing
of the group, are read on the resamples of `compare_votes` (Poisson weights on the images,
its seed), the same for the three recipes, as the paired angles of E3.

    python analysis/download_cifar10h.py                       # data/cifar10h-counts.npy
    python analysis/people_bis.py --turing-meter PATH_TO_CLONE [--raw results/raw/people_bis]
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np

from common import ROOT, Timer, environment, med, verdict, write_json
from cognitive_resonance import raw
from cognitive_resonance.estimators import ci
from people import REPS, SEED, group_turing, labels, load_turing_meter, measure_machine, plurality_correct

RECIPES = raw.PEOPLE_BIS_RECIPES
CONTROL = "controllo"
TRIED = ("somma", "mista")          # the two recipes P2-P4 are about
P1_POINTS = 0.02                    # "not more than 2 points below the control"


def people_cue(counts) -> np.ndarray:
    """u_c = log((v_c + 0.5) / (n + 5)) for every image and class (§ 7 bis): a smoothed log vote share."""
    counts = np.asarray(counts, float)
    n = counts.sum(1, keepdims=True)
    return np.log((counts + 0.5) / (n + 5.0))


def sum_correct(counts, logits, y) -> np.ndarray:
    """1 where argmax(u + z_M) is the true class."""
    return (np.argmax(people_cue(counts) + logits, 1) == y).astype(float)


# --- paired differences on compare_votes' resamples ----------------------------------------------

def _d_votes(sdt, a, m=10):
    """compare_votes' sensitivity from an accuracy: not defined at or below chance."""
    if a <= 1.0 / m + 1e-9:
        return math.nan
    return sdt.d_from_accuracy(float(np.clip(a, 1e-4, 1 - 1e-4)), m)


def votes_step(sdt, h, machine, w, m=10):
    """compare_votes' step for weights w on the items: the angle, and the apparent turings ideal_phi / dh."""
    ah = float(np.average(h, weights=w))
    pm = float(np.average(machine, weights=w))
    p11 = float(np.average(h * machine, weights=w))
    r = sdt.tetrachoric(np.clip(ah, 1e-4, 1 - 1e-4), np.clip(pm, 1e-4, 1 - 1e-4), p11)
    dh, dm = _d_votes(sdt, ah, m), _d_votes(sdt, pm, m)
    tu = sdt.ideal(dh, dm, r) / dh if math.isfinite(dh) and math.isfinite(dm) else math.nan
    return sdt.angle_degrees(r), tu


def paired_votes(sdt, h, oks: dict, reference: str = CONTROL, reps: int = REPS) -> dict:
    """Recipe minus reference, for the angle and the apparent turings, with paired 95% intervals: every recipe is
    read on the same resamples, those of compare_votes (Poisson weights, seed SEED). A resample where a machine
    is at or below chance has no apparent turing and is left out of that difference."""
    names = list(oks)
    gen = np.random.default_rng(SEED)
    phi = np.empty((reps, len(names)))
    tu = np.empty((reps, len(names)))
    for b in range(reps):
        w = gen.poisson(1.0, h.size).astype(float)
        for i, nm in enumerate(names):
            phi[b, i], tu[b, i] = votes_step(sdt, h, oks[nm], w)
    one = np.ones(h.size)
    point = [votes_step(sdt, h, oks[nm], one) for nm in names]
    i0 = names.index(reference)
    out = {}
    for i, nm in enumerate(names):
        if i == i0:
            continue
        out[nm] = {"phi": point[i][0] - point[i0][0], "phi_ci": ci(phi[:, i] - phi[:, i0]),
                   "tu_app": point[i][1] - point[i0][1], "tu_app_ci": ci(tu[:, i] - tu[:, i0])}
    return out


def paired_group(points: dict, runs: dict, reference: str = CONTROL) -> dict:
    """Recipe minus reference for the turing of the group, on the group's resamples (the same for every recipe)."""
    out = {}
    base = np.array([r["turing_group"] for r in runs[reference]])
    for nm in points:
        if nm == reference:
            continue
        draws = np.array([r["turing_group"] for r in runs[nm]]) - base
        out[nm] = {"turing_group": points[nm]["turing_group"] - points[reference]["turing_group"],
                   "turing_group_ci": ci(draws)}
    return out


# --- one seed ----------------------------------------------------------------------------------------

def out_of_fold(folds, size):
    """Logits of each recipe on the half not used for its fine-tuning, assembled over the folds of one seed."""
    oof = np.full((len(RECIPES), size, 10), np.nan)
    for d in folds:
        order = [d["recipe_names"].index(r) for r in RECIPES]
        oof[:, d["image"]] = d["logits"][order]
    return oof, ~np.isnan(oof[0, :, 0])


def seed_record(measure, sdt, counts, n, h, y, folds, reps) -> dict:
    oof, covered = out_of_fold(folds, y.size)
    if not covered.all():
        print(f"{int((~covered).sum())} images without an out-of-fold prediction, left out")
    hc, nc, cc, yc = h[covered], n[covered], counts[covered], y[covered]
    majority = float(np.mean(plurality_correct(cc, yc)))
    oks, rows, runs = {}, {}, {}
    for i, nm in enumerate(RECIPES):
        runs[nm] = []
        oks[nm], rows[nm] = measure_machine(measure, sdt, hc, nc, cc, yc, oof[i][covered], reps,
                                            group_runs_out=runs[nm])
        rows[nm]["accuracy_sum"] = float(sum_correct(cc, oof[i][covered], yc).mean())
        rows[nm]["accuracy_majority"] = majority
    by_fold = {}
    for d in sorted(folds, key=lambda d: d["fold"]):
        lg = d["logits"]
        by_fold[f"fold{d['fold']}"] = {nm: float((np.argmax(lg[d["recipe_names"].index(nm)], 1) == d["y"]).mean())
                                       for nm in RECIPES}
    paired = paired_votes(sdt, hc, oks, CONTROL, reps)
    group = paired_group({nm: rows[nm]["group"] for nm in RECIPES}, runs, CONTROL)
    for nm in paired:
        paired[nm].update(group[nm])
    return {"recipes": rows, "accuracy_by_fold": by_fold, "images": int(covered.sum()), "paired": paired}


# --- criteria ----------------------------------------------------------------------------------------

def _lower_above_zero(interval) -> bool:
    return interval is not None and interval[0] > 0


def criteria(per_seed: dict) -> dict:
    """P1-P5 of § 7 bis from the records of the seeds (only those with fold files count)."""
    seeds = {k: v for k, v in per_seed.items() if "paired" in v}
    if not seeds:
        return {p: verdict(None, reason="no fold files") for p in ("P1", "P2", "P3", "P4", "P5")}
    out = {}

    rows = [{"seed": s, "fold": f, "mista_minus_controllo": a["mista"] - a[CONTROL],
             "somma_minus_controllo": a["somma"] - a[CONTROL], "controllo": a[CONTROL], "mista": a["mista"],
             "somma": a["somma"]}
            for s, v in seeds.items() for f, a in v["accuracy_by_fold"].items()]
    both_halves = all(len(v["accuracy_by_fold"]) == 2 for v in seeds.values())
    ok1 = [r["mista_minus_controllo"] >= -P1_POINTS - 1e-9 for r in rows]
    out["P1"] = verdict(all(ok1) if both_halves else None, halves_within=sum(ok1), halves=len(rows),
                        worst_mista_minus_controllo=min(r["mista_minus_controllo"] for r in rows),
                        worst_somma_minus_controllo=min(r["somma_minus_controllo"] for r in rows), detail=rows,
                        rule="accuracy of the machine alone with mista not more than 2 points below the control, on "
                             "the measured half of every fold and seed (point estimates); somma reported, not judged")

    opened = [r for r in TRIED if all(_lower_above_zero(v["paired"][r]["phi_ci"]) for v in seeds.values())]
    out["P2"] = verdict(bool(opened), recipes_opening_in_every_seed=opened,
                        by_seed={s: {r: {k: v["paired"][r][k] for k in ("phi", "phi_ci")} for r in TRIED}
                                 for s, v in seeds.items()},
                        rule="for somma or mista, phi(recipe) - phi(controllo) on the unseen half with a paired 95% "
                             "interval above zero, the same recipe in every seed")

    raised = [r for r in TRIED if all(_lower_above_zero(v["paired"][r]["tu_app_ci"]) for v in seeds.values())]
    both = [r for r in opened if r in raised]
    out["P3"] = verdict(bool(both), recipes_passing=both, recipes_opening_P2=opened,
                        recipes_raising_tu_app_in_every_seed=raised,
                        by_seed={s: {r: {k: v["paired"][r][k] for k in ("tu_app", "tu_app_ci")} for r in TRIED}
                                 for s, v in seeds.items()},
                        rule="for a recipe that passes P2, Tu_app(recipe) - Tu_app(controllo) with a paired 95% "
                             "interval above zero in every seed; fails if no recipe passes P2")

    phis = {s: {r: v["recipes"][r]["angle"].get("phi") for r in RECIPES} for s, v in seeds.items()}
    judged = [phis[s][r] for s in phis for r in TRIED]
    defined = [p for p in judged if p is not None]
    if any(p >= 90 for p in defined):
        p4 = False
    elif len(defined) < len(judged):
        p4 = None
    else:
        p4 = True
    out["P4"] = verdict(p4, phi_by_seed=phis, max_phi=max(defined) if defined else None,
                        undefined=len(judged) - len(defined),
                        rule="point estimate of phi below 90 degrees for somma and mista in every seed (controllo "
                             "reported); an angle compare_votes cannot measure (machine at chance) leaves it not "
                             "evaluable, unless another is at or above 90")

    out["P5"] = verdict(all(_lower_above_zero(v["paired"]["mista"]["turing_group_ci"]) for v in seeds.values()),
                        by_seed={s: {r: {k: v["paired"][r][k] for k in ("turing_group", "turing_group_ci")}
                                     for r in TRIED} for s, v in seeds.items()},
                        rule="turing of the group with mista minus that with controllo, with a paired 95% interval "
                             "above zero in every seed (somma reported, not judged)")
    return out


def by_recipe(per_seed: dict) -> dict:
    seeds = [v for v in per_seed.values() if "recipes" in v]
    out = {}
    for nm in RECIPES:
        rs = [v["recipes"][nm] for v in seeds]
        out[nm] = {"phi": med([r["angle"].get("phi") for r in rs]),
                   "accuracy": med([r["accuracy"] for r in rs]),
                   "tu_app": med([r["angle"].get("tu_app") for r in rs]),
                   "k": med([r["angle"].get("k") for r in rs]),
                   "turing_group": med([r["group"]["turing_group"] for r in rs]),
                   "acc_combined": med([r["group"]["acc_combined"] for r in rs]),
                   "accuracy_sum": med([r["accuracy_sum"] for r in rs]),
                   "accuracy_majority": med([r["accuracy_majority"] for r in rs]),
                   "accuracy_where_people_right": med([r["accuracy_where_people_right"] for r in rs]),
                   "accuracy_where_people_split": med([r["accuracy_where_people_split"] for r in rs])}
        if nm != CONTROL:
            out[nm]["difference_from_controllo"] = {k: med([v["paired"][nm][k] for v in seeds])
                                                    for k in ("phi", "tu_app", "turing_group")}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default=str(ROOT / "results" / "raw" / "people_bis"))
    ap.add_argument("--counts", default=str(ROOT / "data" / "cifar10h-counts.npy"))
    ap.add_argument("--turing-meter", required=True, help="path of a local turing-meter clone")
    ap.add_argument("--out", default=str(ROOT / "results" / "e3bis_people.json"))
    ap.add_argument("--reps", type=int, default=REPS)
    args = ap.parse_args()
    timer = Timer()
    measure, sdt, tm_commit = load_turing_meter(args.turing_meter)
    counts = np.load(args.counts, allow_pickle=False).astype(float)
    if counts.shape != (10_000, 10):
        raise SystemExit(f"unexpected CIFAR-10H counts shape {counts.shape}")
    n = counts.sum(1)
    folds = [raw.read_people_bis_fold(p) for p in raw.list_units(args.raw, raw.FOLD_RE)]
    bases = [raw.read_people_base(p) for p in raw.list_units(args.raw, raw.BASE_RE)]
    if not folds and not bases:
        raise SystemExit(f"no people_bis files in {args.raw}")
    y = labels(folds, bases)
    h = counts[np.arange(y.size), y] / n
    people = {"mean_accuracy_of_votes": float(np.mean(h)), "votes_per_image_median": float(np.median(n)),
              "plurality_accuracy": float(np.mean(plurality_correct(counts, y))),
              "accuracy_of_argmax_u": float(np.mean(np.argmax(people_cue(counts), 1) == y))}

    per_seed = {}
    for s in sorted({d["seed"] for d in folds} | {d["seed"] for d in bases}):
        rec = {}
        base = [d for d in bases if d["seed"] == s]
        if base:
            lg = np.empty((y.size, 10))
            lg[base[0]["image"]] = base[0]["logits"]
            ok = (np.argmax(lg, 1) == y).astype(float)
            rec["base"] = {"accuracy": float(ok.mean()),
                           "angle": measure.compare_votes(h, n, ok, m=10, repetitions=args.reps),
                           "group": group_turing(sdt, counts, n, lg, y, args.reps)}
        fs = [d for d in folds if d["seed"] == s]
        if fs:
            rec.update(seed_record(measure, sdt, counts, n, h, y, fs, args.reps))
        per_seed[f"s{s}"] = rec

    crit = criteria(per_seed)
    write_json(args.out, {
        "experiment": "E3 bis", "exploratory": True,
        "protocol": "PROTOCOL.md § 7 bis (v1.4): written after seeing all the results, before this run",
        "raw": Path(args.raw).name, "manifest": raw.read_manifest(args.raw),
        "turing_meter": {"repository": "https://github.com/paithon-it/turing-meter", "commit": tm_commit,
                         "function": "experiments/angle-between-errors/angle/measure.py: compare_votes, m = 10"},
        "cifar10h": {"file": Path(args.counts).name, "license": "CC BY-NC-SA 4.0, not redistributed", **people},
        "people_cue": "u_c = log((v_c + 0.5) / (n + 5)); accuracy_sum is that of argmax(u + z_M), accuracy_majority "
                      "that of the plurality vote (ties split) on the same images",
        "descriptive": "accuracy_where_people_right: the machine's accuracy on the images where at least 98% of the "
                       "votes are right; accuracy_where_people_split: where fewer than 80% are (as in E3)",
        "combiner": "as E3: conditional logit on logit((c + 0.5)/(n + 1)) of each class's votes and the machine's log "
                    "probability of it; 5-fold cross-validation over images; group alone = plurality vote",
        "paired": "recipe minus controllo, on the resamples of compare_votes (Poisson weights on the images, seed "
                  f"{SEED}), the same for every recipe: the angle and the apparent turings with compare_votes' "
                  "formulas, the turing of the group with the group's own resamples (same weights)",
        "criteria": crit, "by_recipe": by_recipe(per_seed), "seeds": per_seed,
        "minutes": timer.minutes(), "environment": environment()})
    print(" ".join(f"{p} {v['outcome']}" for p, v in crit.items()), "minutes", timer.minutes())


if __name__ == "__main__":
    main()
