"""Sanity checks on the raw files (not the formal analysis): accuracies and raw correlations per condition.

    python gpu/sanity.py RESULTS_RAW [CIFAR10H_COUNTS]

Cues are standardised by their within-class standard deviation on validation, as in protocol section 3.
"""
import json
import sys
from pathlib import Path

import numpy as np


def within_sd(z, y):
    r = z.astype(np.float64).copy()
    for c in (0, 1):
        r[y == c] -= r[y == c].mean()
    return np.sqrt((r ** 2).mean())


def within_corr(u, v, y):
    u, v = u.astype(np.float64).copy(), v.astype(np.float64).copy()
    for c in np.unique(y):
        u[y == c] -= u[y == c].mean()
        v[y == c] -= v[y == c].mean()
    return float((u * v).sum() / np.sqrt((u * u).sum() * (v * v).sum()))


def acc(z, y):
    return float(((z > 0) == (y == 1)).mean())


def slope_on_a(z, a, y):
    """Within-class slope of z on a (class-specific intercepts)."""
    z, a = z.astype(np.float64).copy(), a.astype(np.float64).copy()
    for c in (0, 1):
        z[y == c] -= z[y == c].mean()
        a[y == c] -= a[y == c].mean()
    return float((z * a).sum() / (a * a).sum())


def bench(raw: Path):
    rows = {}
    for p in sorted((raw / "bench").glob("*.npz")):
        task, mu, s = p.stem.split("_")
        d = np.load(p)
        y, vy = d["y"], d["val_y"]
        sd = {r: within_sd(d[f"val_z_{r}"], vy) for r in ("plus", "minus", "plus2")}
        zs = {r: d[f"z_{r}"] / sd[r] for r in sd}
        rows.setdefault(float(mu[2:]), {}).setdefault(task, []).append({
            "plus": acc(zs["plus"], y), "minus": acc(zs["minus"], y), "plus2": acc(zs["plus2"], y),
            "sum": acc(zs["plus"] + zs["minus"], y), "sum_same": acc(zs["plus"] + zs["plus2"], y),
            "r_pm": within_corr(d["z_plus"], d["z_minus"], y), "r_pp": within_corr(d["z_plus"], d["z_plus2"], y),
            "slope_plus": slope_on_a(zs["plus"], d["a"], y), "slope_minus": slope_on_a(zs["minus"], d["a"], y)})
    keys = ("plus", "minus", "plus2", "sum", "sum_same", "r_pm", "r_pp", "slope_plus", "slope_minus")
    print("E1 bench: median over tasks of the mean over seeds (test, 5 presentations per image)")
    print("mu    n  " + " ".join(f"{k:>11}" for k in keys))
    for mu in sorted(rows):
        per_task = {t: {k: np.mean([u[k] for u in v]) for k in keys} for t, v in rows[mu].items()}
        n = sum(len(v) for v in rows[mu].values())
        print(f"{mu:<5g} {n:<3d}" + " ".join(f"{np.median([x[k] for x in per_task.values()]):11.3f}" for k in keys))
    print("per task, plus / minus / sum accuracy and r(plus, minus):")
    for mu in sorted(rows):
        print(f"  mu {mu:g}: " + "  ".join(
            f"{t} {np.mean([u['plus'] for u in v]):.3f}/{np.mean([u['minus'] for u in v]):.3f}/"
            f"{np.mean([u['sum'] for u in v]):.3f} r={np.mean([u['r_pm'] for u in v]):+.2f}"
            for t, v in sorted(rows[mu].items())))


def rephase(raw: Path):
    files = sorted((raw / "rephase").glob("*.npz"))
    if not files:
        return
    out = {}
    for p in files:
        d = np.load(p)
        y, vy, a = d["y"], d["val_y"], d["a"]
        zH = d["z_H"] / within_sd(d["val_z_H"], vy)
        zH2 = d["z_H2"] / within_sd(d["val_z_H2"], vy)
        rec = {"H": (acc(zH, y), slope_on_a(zH, a, y), 1.0), "H2": (acc(zH2, y), slope_on_a(zH2, a, y),
                                                                     within_corr(d["z_H"], d["z_H2"], y))}
        for i, lam in enumerate(d["lambdas"]):
            zm = d["z_M"][i] / within_sd(d["val_z_M"][i], vy)
            rec[f"M{lam:g}"] = (acc(zm, y), slope_on_a(zm, a, y), within_corr(d["z_H"], d["z_M"][i], y))
        out[p.stem] = rec
    print("\nE2 rephase: median over units (task x seed) of accuracy alone, within-class slope of the standardised "
          "cue on a, raw within-class r with H")
    print(f"{'judge':<6} {'acc':>7} {'slope':>8} {'r(H,.)':>8}   per-task acc")
    for k in next(iter(out.values())):
        v = np.array([u[k] for u in out.values()])
        print(f"{k:<6} {np.median(v[:, 0]):7.3f} {np.median(v[:, 1]):+8.3f} {np.median(v[:, 2]):+8.3f}   " +
              " ".join(f"{u[k][0]:.3f}" for u in out.values()))


def people(raw: Path, counts_path):
    folder = raw / "people"
    if not list(folder.glob("*.npz")):
        return
    print("\nE3 people")
    counts = np.load(counts_path) if counts_path else None
    for p in sorted(folder.glob("base_s*.npz")):
        d = np.load(p)
        print(f"  {p.stem}: test accuracy {(d['logits'].argmax(1) == d['y']).mean():.4f}")
    for p in sorted(folder.glob("fold*_s*.npz")):
        d = np.load(p)
        line = []
        for i, lam in enumerate(d["lambdas"]):
            lg = d["logits"][i].astype(np.float64)
            a = (lg.argmax(1) == d["y"]).mean()
            s = f"lambda {lam:g}: acc {a:.4f}"
            if counts is not None:
                y = d["y"].astype(int)
                c = counts[d["image"]]
                n, h = c.sum(1), c[np.arange(len(y)), y]
                cue_p = np.log((h + 0.5) / (n + 1 - h - 0.5))
                true = lg[np.arange(len(y)), y]
                oth = lg.copy(); oth[np.arange(len(y)), y] = -np.inf
                m = oth.max(1)
                cue_m = true - (m + np.log(np.exp(oth - m[:, None]).sum(1)))
                s += f", r(machine, people) {within_corr(cue_m, cue_p, y):+.3f}"
            line.append(s)
        print(f"  {p.stem}: " + "; ".join(line))


def main() -> None:
    raw = Path(sys.argv[1])
    counts = sys.argv[2] if len(sys.argv) > 2 else None
    if list((raw / "bench").glob("*.npz")):
        bench(raw)
    if (raw / "rephase").exists():
        rephase(raw)
    if (raw / "people").exists():
        people(raw, counts)


if __name__ == "__main__":
    main()
