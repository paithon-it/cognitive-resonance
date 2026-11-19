"""Measures of a pair at a weight that nobody tunes, shared by E6 and E7 (protocol §§ 12-13).

The tuned turing of § 3 answers "how much can this pair be worth". Here the question is the
other one: how much is it worth at a weight fixed beforehand, and at which weight does it
peak. Two scales for the weight:

- standardized: `z_H / sd_H + w z_M / sd_M`, the scale of § 3;
- raw: `z_H + w z_M`, the log-odds as the networks give them. At `w = 1` it is the pair
  "at face value": the two opinions added as they are.
"""
from __future__ import annotations

import math

import numpy as np

from cognitive_resonance import estimators as est
from cognitive_resonance import model

STD_WEIGHTS = (0.5, 1.0, 2.0, 8.0)      # standardized weights at which every pair is read
PEAK_LEVEL = 0.9                        # the peak is the middle of the weights within 10% of the maximum of Tu^2 - 1


def std_key(w: float) -> str:
    return f"std{w:g}"


def scores_of(zh_raw, zm_raw, zh_std, zm_std) -> dict:
    """The scores of one pair that are read at a fixed weight, plus the machine alone."""
    out = {std_key(w): zh_std + w * zm_std for w in STD_WEIGHTS}
    out["face"] = np.asarray(zh_raw, float) + np.asarray(zm_raw, float)
    out["alone"] = np.asarray(zm_raw, float)
    return out


def boot_d_a(design: est.Design, scores: dict, reps: int, seed: int = est.BOOT_SEED, chunk: int = 100) -> dict:
    """d_a of every score in the cluster resamples of est.bootstrap: same generator, same seed,
    same chunks, so the draws are the same images and differences with its arrays are paired."""
    sorted_scores = {name: est.SortedScore(s, design.y) for name, s in scores.items()}
    gen = np.random.default_rng(seed)
    out = {name: [] for name in scores}
    done = 0
    while done < reps:
        bc = min(chunk, reps - done)
        W = est.resample_counts(gen, design.n_images, bc)[:, design.img_index]
        for name, s in sorted_scores.items():
            out[name].append(est.d_a_from_auc(*s.auc(W)))
        done += bc
    return {name: np.concatenate(v) for name, v in out.items()}


def raw_curve(zh_raw, zm_raw, y, grid=model.GRID) -> np.ndarray:
    """Tu(w) = d_a(z_H + w z_M) / d_a(z_H) with the raw log-odds."""
    return est.tu_curve(np.asarray(zh_raw, float), np.asarray(zm_raw, float), y, grid)[0]


def peak_centre(grid, tu_values, level: float = PEAK_LEVEL) -> dict:
    """Where a curve peaks. The maximum of a flat curve is a noisy place, so the peak is the
    geometric mean of the two weights at which Tu^2 - 1 crosses `level` times its maximum
    (linear interpolation in ln w). A crossing outside the grid leaves the centre open (None);
    `estimate` is then the weight of the maximum.
    """
    grid = np.asarray(grid, float)
    v = np.asarray(tu_values, float) ** 2 - 1.0
    lw = np.log(grid)
    i = int(np.argmax(v))
    out = {"argmax": float(grid[i]), "tu_max": float(tu_values[i]), "low": None, "high": None, "centre": None}
    if v[i] > 0:
        th = level * v[i]
        for j in range(i - 1, -1, -1):
            if v[j] < th:
                t = (th - v[j]) / (v[j + 1] - v[j])
                out["low"] = float(math.exp(lw[j] + t * (lw[j + 1] - lw[j])))
                break
        for j in range(i + 1, v.size):
            if v[j] < th:
                t = (v[j - 1] - th) / (v[j - 1] - v[j])
                out["high"] = float(math.exp(lw[j - 1] + t * (lw[j] - lw[j - 1])))
                break
        if out["low"] is not None and out["high"] is not None:
            out["centre"] = math.sqrt(out["low"] * out["high"])
    out["estimate"] = out["centre"] if out["centre"] is not None else out["argmax"]
    return out


def accuracy(z_raw, y) -> float:
    """Share of right answers when the answer is the sign of the raw log-odds."""
    return float(np.mean((np.asarray(z_raw, float) > 0) == (np.asarray(y) == 1)))


def fixed_measures(d: dict, cues: dict, design: est.Design, machines, reps: int):
    """For every machine: the pair at the fixed weights and at face value, the machine alone, the
    raw curve with its peak, and the shares of right answers (descriptive: no criterion uses
    them). Returns the point estimates and the bootstrap arrays, keyed (machine, score)."""
    y = d["y"]
    zh_raw = np.asarray(d["z_H"], float)
    names = d["machine_names"]
    scores, point = {}, {}
    dh_raw = float(est.d_a(zh_raw, y))
    for name in machines:
        zm_raw = np.asarray(d["z_M"][names.index(name)], float)
        sc = scores_of(zh_raw, zm_raw, cues["H"], cues[name])
        curve = raw_curve(zh_raw, zm_raw, y)
        point[name] = {"d_a": {k: float(est.d_a(s, y)) for k, s in sc.items()},
                       "accuracy": {"alone": accuracy(zm_raw, y), "face": accuracy(zh_raw + zm_raw, y),
                                    "person_alone": accuracy(zh_raw, y)},
                       "raw_curve": [float(x) for x in curve], "raw_peak": peak_centre(model.GRID, curve),
                       "sd_ratio_m_over_h": float(est.within_class_sd(d["val_z_M"][names.index(name)], d["val_y"])
                                                  / est.within_class_sd(d["val_z_H"], d["val_y"]))}
        for k, s in sc.items():
            scores[(name, k)] = s
    scores[("H", "alone")] = zh_raw
    boot = boot_d_a(design, scores, reps)
    return point, boot, dh_raw


def paired(point_a: float, point_b: float, boot_a, boot_b) -> dict:
    """a - b, with the percentile interval of the paired difference."""
    return {"diff": point_a - point_b, "ci": est.ci(np.asarray(boot_a) - np.asarray(boot_b))}


def lower_above_zero(x: dict) -> bool:
    return x["ci"] is not None and x["ci"][0] > 0


def upper_below_zero(x: dict) -> bool:
    return x["ci"] is not None and x["ci"][1] < 0
