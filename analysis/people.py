"""E3 (protocol § 7): real people, CIFAR-10H against a ResNet-18 and its antiphase versions.

The angle is measured with turing-meter's `compare_votes` (m = 10), imported from a local
clone whose commit is written in the results. The combiner for the turing of the group is
a conditional logistic regression on the vote share and the machine's probability of each
class, with 5-fold cross-validation over images.

    python analysis/download_cifar10h.py                       # data/cifar10h-counts.npy
    python analysis/people.py --turing-meter PATH_TO_CLONE [--raw results/raw/people]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.special import log_softmax, logsumexp

from common import ROOT, Timer, environment, git_commit, med, verdict, write_json
from cognitive_resonance import raw

REPS = 1000
SEED = 534               # the same seed as compare_votes, for the paired bootstrap
CV_FOLDS = 5


def load_turing_meter(path):
    exp = Path(path) / "experiments" / "angle-between-errors"
    if not (exp / "angle" / "measure.py").is_file():
        raise SystemExit(f"not a turing-meter clone: {path}")
    sys.path.insert(0, str(exp))
    from angle import measure, sdt  # noqa: E402
    return measure, sdt, git_commit(path)


# --- the combiner ------------------------------------------------------------------------------

def features(counts, n, logits):
    p = (counts + 0.5) / (n[:, None] + 1.0)                 # the people's cue of § 7, for every class
    share = np.log(p) - np.log1p(-p)
    return np.stack([share, log_softmax(logits, axis=1)], axis=-1)        # (N, 10, 2)


def _nll(beta, X, y):
    s = X @ beta
    return float(np.sum(logsumexp(s, axis=1) - s[np.arange(y.size), y]))


def _grad(beta, X, y):
    s = X @ beta
    p = np.exp(s - logsumexp(s, axis=1, keepdims=True))
    return np.einsum("nc,nck->k", p, X) - X[np.arange(y.size), y].sum(0)


def combine_cv(X, y, seed=SEED) -> tuple[np.ndarray, list]:
    """Out-of-fold choice of the class, and the coefficients of each fold."""
    gen = np.random.default_rng(seed)
    fold = gen.permutation(y.size) % CV_FOLDS
    pred = np.empty(y.size, int)
    coefs = []
    for f in range(CV_FOLDS):
        tr, te = fold != f, fold == f
        res = minimize(_nll, np.ones(X.shape[-1]), args=(X[tr], y[tr]), jac=_grad, method="L-BFGS-B")
        coefs.append([float(b) for b in res.x])
        pred[te] = np.argmax(X[te] @ res.x, axis=1)
    return pred == y, coefs


def plurality_correct(counts, y):
    top = counts.max(1, keepdims=True)
    ties = (counts == top).sum(1)
    return (counts[np.arange(y.size), y] == top[:, 0]) / ties


def group_turing(sdt, counts, n, logits, y, reps=REPS) -> dict:
    ok_comb, coefs = combine_cv(features(counts, n, logits), y)
    ok_group = plurality_correct(counts, y)
    ok_m = (np.argmax(logits, 1) == y).astype(float)

    def d(acc, size):
        return sdt.d_from_accuracy(float(np.clip(acc, 0.1 + 1e-6, 1 - 0.5 / size)), 10)

    def step(w):
        a = [float(np.average(v, weights=w)) for v in (ok_comb, ok_group, ok_m)]
        size = float(w.sum())
        dc, dg, dm = d(a[0], size), d(a[1], size), d(a[2], size)
        return {"acc_combined": a[0], "acc_group": a[1], "acc_machine": a[2], "d_combined": dc,
                "d_group": dg, "d_machine": dm, "turing_group": dc / dg, "yield_on_machine": dc / dm}

    point = step(np.ones(y.size))
    gen = np.random.default_rng(SEED)
    runs = [step(gen.poisson(1.0, y.size).astype(float)) for _ in range(reps)]
    for key in list(point):
        s = np.array([r[key] for r in runs])
        point[key + "_ci"] = [float(np.percentile(s, 2.5)), float(np.percentile(s, 97.5))]
    point["coefficients_by_fold"] = coefs
    return point


# --- the paired angle ----------------------------------------------------------------------------

def phi_votes(sdt, h, machine, w):
    """The angle of compare_votes' step, for weights w on the items (same formulas)."""
    ah = float(np.average(h, weights=w))
    pm = float(np.average(machine, weights=w))
    p11 = float(np.average(h * machine, weights=w))
    r = sdt.tetrachoric(np.clip(ah, 1e-4, 1 - 1e-4), np.clip(pm, 1e-4, 1 - 1e-4), p11)
    return sdt.angle_degrees(r)


def paired_differences(sdt, h, oks: dict, reps=REPS) -> dict:
    gen = np.random.default_rng(SEED)
    names = list(oks)
    runs = np.empty((reps, len(names)))
    for b in range(reps):
        w = gen.poisson(1.0, h.size).astype(float)
        runs[b] = [phi_votes(sdt, h, oks[nm], w) for nm in names]
    point = np.array([phi_votes(sdt, h, oks[nm], np.ones(h.size)) for nm in names])
    out = {}
    for i, nm in enumerate(names[1:], start=1):
        diff = runs[:, i] - runs[:, 0]
        out[nm] = {"phi_minus_phi_lambda0": float(point[i] - point[0]),
                   "ci": [float(np.percentile(diff, 2.5)), float(np.percentile(diff, 97.5))]}
    return out


def labels(folds, bases, size=10_000):
    y = np.full(size, -1)
    for d in list(folds) + list(bases):
        if "y" in d:
            img, yy = d["image"], d["y"]
            if np.any((y[img] >= 0) & (y[img] != yy)):
                raise SystemExit("inconsistent labels between people files")
            y[img] = yy
    if np.any(y < 0):
        raise SystemExit("labels missing for some images")
    return y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default=str(ROOT / "results" / "raw" / "people"))
    ap.add_argument("--counts", default=str(ROOT / "data" / "cifar10h-counts.npy"))
    ap.add_argument("--turing-meter", required=True, help="path of a local turing-meter clone")
    ap.add_argument("--out", default=str(ROOT / "results" / "e3_people.json"))
    ap.add_argument("--reps", type=int, default=REPS)
    args = ap.parse_args()
    timer = Timer()
    measure, sdt, tm_commit = load_turing_meter(args.turing_meter)
    counts = np.load(args.counts, allow_pickle=False).astype(float)
    if counts.shape != (10_000, 10):
        raise SystemExit(f"unexpected CIFAR-10H counts shape {counts.shape}")
    n = counts.sum(1)
    folds = [raw.read_people_fold(p) for p in raw.list_units(args.raw, raw.FOLD_RE)]
    bases = [raw.read_people_base(p) for p in raw.list_units(args.raw, raw.BASE_RE)]
    if not folds and not bases:
        raise SystemExit(f"no people files in {args.raw}")
    y = labels(folds, bases)
    h = counts[np.arange(y.size), y] / n
    people = {"mean_accuracy_of_votes": float(np.mean(h)), "votes_per_image_median": float(np.median(n)),
              "plurality_accuracy": float(np.mean(plurality_correct(counts, y)))}

    seeds = sorted({d["seed"] for d in folds} | {d["seed"] for d in bases})
    per_seed = {}
    for s in seeds:
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
            lams = [float(x) for x in fs[0]["lambdas"]]
            oof = np.full((len(lams), y.size, 10), np.nan)
            for d in fs:
                if [float(x) for x in d["lambdas"]] != lams:
                    raise SystemExit("folds with different lambdas")
                oof[:, d["image"]] = d["logits"]
            covered = ~np.isnan(oof[0, :, 0])
            if not covered.all():
                print(f"seed {s}: {int((~covered).sum())} images without an out-of-fold prediction, left out")
            oks, rows = {}, {}
            for i, lam in enumerate(lams):
                ok = (np.argmax(oof[i][covered], 1) == y[covered]).astype(float)
                oks[f"{lam:g}"] = ok
                r = measure.compare_votes(h[covered], n[covered], ok, m=10, repetitions=args.reps)
                if "ideal_phi" in r:
                    r["tu_app"] = r["ideal_phi"] / r["dh"]
                rows[f"{lam:g}"] = {"accuracy": float(ok.mean()), "angle": r,
                                    "group": group_turing(sdt, counts[covered], n[covered], oof[i][covered],
                                                          y[covered], args.reps)}
            rec["lambdas"] = rows
            rec["images"] = int(covered.sum())
            rec["paired"] = paired_differences(sdt, h[covered], oks, args.reps)
        per_seed[f"s{s}"] = rec

    # criteria
    with_folds = {k: v for k, v in per_seed.items() if "paired" in v}
    if with_folds:
        lams_pos = [lam for lam in next(iter(with_folds.values()))["paired"]]
        opened = [lam for lam in lams_pos if all(v["paired"][lam]["ci"][0] > 0 for v in with_folds.values())]
        h3a = verdict(bool(opened), lambdas_opening_in_every_seed=opened,
                      by_seed={k: v["paired"] for k, v in with_folds.items()},
                      rule="for some lambda > 0, phi(lambda) - phi(0) on the unseen half with a paired 95% "
                           "interval above zero, in every seed")
        phis = {k: {lam: r["angle"].get("phi") for lam, r in v["lambdas"].items()} for k, v in with_folds.items()}
        flat = [p for v in phis.values() for p in v.values() if p is not None]
        h3b = verdict(all(p < 90 for p in flat), phi_by_seed=phis, max_phi=max(flat) if flat else None,
                      rule="point estimate of phi below 90 degrees for every lambda and seed")
    else:
        h3a = verdict(None, reason="no fold files")
        h3b = verdict(None, reason="no fold files")
    summary = {}
    if with_folds:
        for lam in next(iter(with_folds.values()))["lambdas"]:
            rs = [v["lambdas"][lam] for v in with_folds.values()]
            summary[lam] = {"phi": med([r["angle"].get("phi") for r in rs]),
                            "accuracy": med([r["accuracy"] for r in rs]),
                            "tu_app": med([r["angle"].get("tu_app") for r in rs]),
                            "k": med([r["angle"].get("k") for r in rs]),
                            "turing_group": med([r["group"]["turing_group"] for r in rs]),
                            "acc_combined": med([r["group"]["acc_combined"] for r in rs])}
    write_json(args.out, {
        "experiment": "E3", "protocol": "PROTOCOL.md § 7 (v1.2)", "raw": Path(args.raw).name,
        "manifest": raw.read_manifest(args.raw),
        "turing_meter": {"repository": "https://github.com/paithon-it/turing-meter", "commit": tm_commit,
                         "function": "experiments/angle-between-errors/angle/measure.py: compare_votes, m = 10"},
        "cifar10h": {"file": Path(args.counts).name, "license": "CC BY-NC-SA 4.0, not redistributed",
                     **people},
        "combiner": "conditional logit on logit((c + 0.5)/(n + 1)) of each class's votes and the machine's log "
                    "probability of it; 5-fold cross-validation over images; group alone = plurality vote",
        "criteria": {"H3a": h3a, "H3b": h3b}, "by_lambda": summary, "seeds": per_seed,
        "minutes": timer.minutes(), "environment": environment()})
    print("H3a", h3a["outcome"], "H3b", h3b["outcome"], "minutes", timer.minutes())


if __name__ == "__main__":
    main()
