"""E1 (protocol § 5): the bench, two opposite biases.

Reads results/raw/bench/*.npz and writes every measure of the protocol for each unit
(task, mu, seed) and each pair (opposite: plus with minus; same: plus with plus2), then
the outcome of H1a-H1f with the numbers that decide it.

    python analysis/bench.py [--raw results/raw/bench] [--out results/e1_bench.json]
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np

from common import ROOT, Timer, environment, med, run_pool, verdict, write_json
from cognitive_resonance import estimators as est
from cognitive_resonance import model, raw

JUDGES = ("plus", "minus", "plus2")
PAIRS = {"opposite": ("plus", "minus"), "same": ("plus", "plus2")}


def standardized(d, names, prefix="z_"):
    return {n: est.standardize(d[prefix + n], d["y"], d["val_" + prefix + n], d["val_y"]) for n in names}


def unit(job):
    path, reps, fixed = job
    d = raw.read_bench(path)
    y = d["y"]
    cues = standardized(d, JUDGES)
    design = est.Design(y, d["image"], cues, a=d["a"])
    pairs = {name: est.measure_pair(cues[h], cues[m], y, design.half) for name, (h, m) in PAIRS.items()}
    weights = {name: p["w_tuned_halves"] for name, p in pairs.items()} if fixed else None
    boot = est.bootstrap(design, PAIRS, reps=reps, slopes=True, fixed_weights=weights)
    for name in PAIRS:
        est.attach_intervals(pairs[name], boot, name)
    judges = {}
    for j in JUDGES:
        judges[j] = {"d_a": float(est.d_a(cues[j], y)), "d_a_ci": est.ci(boot[(j, "d_a")]),
                     "d_yesno": float(est.d_yesno(cues[j], y)),
                     "slope_on_a": float(est.slope_within(cues[j], d["a"], y)),
                     "slope_on_a_ci": est.ci(boot[(j, "slope")]),
                     "val_within_class_sd_raw": est.within_class_sd(d["val_z_" + j], d["val_y"]),
                     "test_within_class_sd_standardized": est.within_class_sd(cues[j], y)}
    return {"task": d["task"], "mu": d["mu"], "seed": d["seed"], "file": d["file"], "n": int(y.size),
            "n_images": design.n_images, "n_val": int(d["val_y"].size), "sd_a": float(np.std(d["a"])),
            "judges": judges, "pairs": pairs}


def _above90(p):
    return p.get("phi_ci") is not None and p["phi_ci"][0] > 90.0


def _need(n_tasks, of_five):
    return math.ceil(of_five / 5 * n_tasks)


def criteria(units) -> dict:
    tasks = sorted({u["task"] for u in units})
    seeds = sorted({u["seed"] for u in units})
    mus = sorted({u["mu"] for u in units})
    by = {(u["task"], u["mu"], u["seed"]): u for u in units}
    out = {}

    # H1a: rho falls with mu in the opposite pairs; phi above 90 at the two highest mu
    rank = {}
    for t in tasks:
        for s in seeds:
            xs = [(m, by[(t, m, s)]["pairs"]["opposite"]["rho"]) for m in mus if (t, m, s) in by]
            rank[f"{t} s{s}"] = est.spearman([x[0] for x in xs], [x[1] for x in xs])
    top = mus[-2:]
    per_task = {}
    for t in tasks:
        cells = [by[(t, m, s)]["pairs"]["opposite"] for m in top for s in seeds if (t, m, s) in by]
        per_task[t] = {"pairs": len(cells), "phi_interval_above_90": sum(_above90(p) for p in cells),
                       "all": bool(cells) and all(_above90(p) for p in cells)}
    med_rank = med(rank.values())
    n_ok = sum(v["all"] for v in per_task.values())
    out["H1a"] = verdict(med_rank is not None and med_rank <= -0.9 and n_ok >= _need(len(tasks), 4),
                         median_spearman_mu_rho=med_rank, spearman_by_task_seed=rank, top_mu=top,
                         tasks_with_phi_interval_above_90=n_ok, tasks_needed=_need(len(tasks), 4),
                         by_task=per_task,
                         rule="median Spearman(mu, rho) <= -0.9 and, at the two highest mu, the phi interval "
                              "above 90 degrees in every seed, in at least 4 tasks of 5")

    # the pairs beyond 90 degrees, any kind
    sel = [(u, kind, u["pairs"][kind]) for u in units for kind in PAIRS if _above90(u["pairs"][kind])]
    n_sel = len(sel)
    if n_sel:
        above = [p["tu_tuned_ci"][0] > p["pythagoras"] for _, _, p in sel]
        share = float(np.mean(above))
        out["H1b"] = verdict(share >= 0.8, pairs=n_sel, kinds={k: sum(1 for _, kk, _ in sel if kk == k) for k in PAIRS},
                             lower_end_above_pythagoras=int(sum(above)), share=share,
                             rule="lower end of the tuned turing's interval above sqrt(1 + k^2) in >= 80% of the pairs "
                                  "whose phi interval is above 90 degrees")
        ratio = [p["ratio_tuned_app"] for _, _, p in sel]
        mare_app = float(np.mean([abs(p["tu_app"] - p["tu_tuned"]) / p["tu_tuned"] for _, _, p in sel]))
        mare_90 = float(np.mean([abs(p["pythagoras"] - p["tu_tuned"]) / p["tu_tuned"] for _, _, p in sel]))
        mr = med(ratio)
        out["H1c"] = verdict(mr is not None and 0.9 <= mr <= 1.1 and mare_app < mare_90, pairs=n_sel,
                             median_ratio_tuned_over_tu_app=mr,
                             ratio_quartiles=[float(np.percentile(ratio, 25)), float(np.percentile(ratio, 75))],
                             mean_abs_rel_error_tu_app=mare_app, mean_abs_rel_error_pythagoras=mare_90,
                             rule="median tuned/Tu_app in [0.9, 1.1] and mean |pred - tuned|/tuned smaller for "
                                  "Tu_app than for sqrt(1 + k^2)")
        both = [p for _, _, p in sel if p["band_measured"]["log_width"] is not None
                and p["band_predicted"]["log_width"] is not None]
        lr = [p["band_measured"]["log_width"] / p["band_predicted"]["log_width"] for p in both]
        withw = [p for _, _, p in sel if p["band_measured"]["log_width"] is not None]
        rk = est.spearman([p["phi"] for p in withw], [p["band_measured"]["log_width"] for p in withw])
        mlr = med(lr)
        out["H1d"] = verdict(None if mlr is None or rk is None else (0.7 <= mlr <= 1.3 and rk <= -0.7),
                             pairs=n_sel, pairs_with_both_widths=len(both),
                             open_measured=n_sel - len(withw),
                             median_ratio_log_width_measured_over_predicted=mlr,
                             spearman_phi_log_width=rk,
                             rule="median ln(measured width)/ln(predicted width) in [0.7, 1.3] and "
                                  "Spearman(phi, measured width) <= -0.7")
    else:
        for h in ("H1b", "H1c", "H1d"):
            out[h] = verdict(None, pairs=0, reason="no pair with the phi interval above 90 degrees")

    # H1e: the same bias
    rank_s = {}
    for t in tasks:
        for s in seeds:
            xs = [(m, by[(t, m, s)]["pairs"]["same"]["rho"]) for m in mus if (t, m, s) in by]
            rank_s[f"{t} s{s}"] = est.spearman([x[0] for x in xs], [x[1] for x in xs])
    top_same = [u["pairs"]["same"]["tu_tuned"] for u in units if u["mu"] == mus[-1]]
    mrs, mts = med(rank_s.values()), med(top_same)
    out["H1e"] = verdict(None if mrs is None else (mrs >= 0.9 and mts <= 1.1),
                         median_spearman_mu_rho_same=mrs, spearman_by_task_seed=rank_s,
                         median_tuned_turing_same_at_top_mu=mts, top_mu=mus[-1],
                         rule="median Spearman(mu, rho) >= 0.9 in the equal pairs (threshold of H1a and H2a; the "
                              "protocol says only that rho rises) and median tuned turing <= 1.1 at the highest mu")

    # H1f: nothing is created
    ref = {}
    for t in tasks:
        vals = [by[(t, mus[0], s)]["pairs"]["opposite"]["d_pair_tuned"] for s in seeds if (t, mus[0], s) in by]
        ref[t] = float(np.mean(vals)) if vals else None
    per_mu = {}
    for m in mus:
        r = [by[(t, m, s)]["pairs"]["opposite"]["d_pair_tuned"] / ref[t]
             for t in tasks for s in seeds if (t, m, s) in by and ref[t]]
        per_mu[f"{m:g}"] = {"median_ratio": med(r), "max_ratio": float(np.max(r)) if r else None,
                            "above_1_05": int(sum(x > 1.05 for x in r)), "pairs": len(r)}
    out["H1f"] = verdict(all(v["median_ratio"] is not None and v["median_ratio"] <= 1.05 for v in per_mu.values()),
                         reference_mu=mus[0], reference_d_pair_by_task=ref, by_mu=per_mu,
                         rule="at every mu, median over tasks and seeds of d_pair_tuned(opposite) / d_pair_tuned at "
                              "mu = 0 (task mean over seeds) <= 1.05")
    return out


def by_mu(units) -> dict:
    """Medians over tasks and seeds, per mu and kind of pair."""
    keys = ("phi", "phi_tetrachoric", "phi_gap", "k", "k_yesno", "q", "tu_tuned", "tu_app", "pythagoras",
            "d_h", "d_m", "d_pair_tuned", "yield_on_machine", "ratio_tuned_app", "rho", "rho_tetrachoric")
    out = {}
    for kind in PAIRS:
        rows = {}
        for m in sorted({u["mu"] for u in units}):
            us = [u for u in units if u["mu"] == m]
            rows[f"{m:g}"] = {k: med([u["pairs"][kind].get(k) for u in us]) for k in keys}
            rows[f"{m:g}"]["n_pairs"] = len(us)
            rows[f"{m:g}"]["phi_interval_above_90"] = sum(_above90(u["pairs"][kind]) for u in us)
        out[kind] = rows
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default=str(ROOT / "results" / "raw" / "bench"))
    ap.add_argument("--out", default=str(ROOT / "results" / "e1_bench.json"))
    ap.add_argument("--reps", type=int, default=1000)
    ap.add_argument("--fixed-weight", action="store_true",
                    help="keep the weight chosen on the whole data in the resamples (§ 3 fallback)")
    ap.add_argument("--workers", type=int, default=3)
    args = ap.parse_args()
    timer = Timer()
    files = raw.list_units(args.raw, raw.BENCH_RE)
    if not files:
        raise SystemExit(f"no bench files in {args.raw}")
    units = run_pool(unit, [(f, args.reps, args.fixed_weight) for f in files], args.workers)
    units.sort(key=lambda u: (u["task"], u["mu"], u["seed"]))
    out = {"experiment": "E1", "protocol": "PROTOCOL.md § 5 (v1.2)", "raw": Path(args.raw).name,
           "manifest": raw.read_manifest(args.raw),
           "settings": {"resamples": args.reps, "grid": "49 weights, 1/64..64 even in ln w",
                        "bootstrap_weight": "fixed at the whole-data choice" if args.fixed_weight
                        else "chosen again in each resample on 13 weights",
                        "split_seed": est.SPLIT_SEED, "bootstrap_seed": est.BOOT_SEED},
           "criteria": criteria(units), "by_mu": by_mu(units), "units": units,
           "minutes": timer.minutes(), "environment": environment()}
    write_json(args.out, out)
    for h, v in out["criteria"].items():
        print(h, v["outcome"])
    print("minutes", out["minutes"])


if __name__ == "__main__":
    main()
