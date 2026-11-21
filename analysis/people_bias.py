"""E13, the readable bias measured on people (protocol § 19).

Reads the files of the Confidence Database (analysis/download_confidence_db.py), keeps the two-answer tasks with a
confidence, and for every person measures the bias that the history of their trials lets one read, and what a
corrector that sees no stimulus gives back. The files are split into exploration and confirmation by the SHA-1 of
their name; the criteria P1 to P4 are read on the confirmation files only.

    python analysis/people_bias.py [--data data/confidence_db] [--only exploration|confirmation]
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import math
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ROOT, write_json  # noqa: E402

from cognitive_resonance import readable  # noqa: E402

COLUMNS = ("Subj_idx", "Stimulus", "Response", "Confidence")
MIN_TRIALS, MIN_PEOPLE = 400, 10
ACCURACY = (0.55, 0.95)
MIN_CLASS_SHARE = 0.20
MIN_CONFIDENCE_LEVELS = 3
RESAMPLES = 2000
SHRINK_GRID = (0.0, 30.0, 100.0, 300.0, 1000.0, 3000.0, 1e9)    # trials the group counts for; 1e9 is the group alone


def split_of(name: str) -> str:
    return "exploration" if hashlib.sha1(name.lower().encode()).hexdigest()[0] in "012" else "confirmation"


def _number(text: str) -> float | None:
    try:
        x = float(text)
    except (TypeError, ValueError):
        return None
    return x if np.isfinite(x) else None


def _label(text: str) -> str | None:
    """A stimulus or a response as a label: a number in one spelling, or the text as it is. None if missing."""
    x = _number(text)
    if x is not None:
        return repr(x)
    text = (text or "").strip().lower()
    return None if text in ("", "nan", "na", "n/a", "null", "none") else text


def read_file(path: Path) -> tuple[dict | None, str]:
    """People of a file as arrays in trial order, or None with the reason why the file stays out."""
    with open(path, newline="", encoding="utf-8-sig", errors="replace") as f:
        reader = csv.DictReader(f)
        names = {(c or "").strip().lower(): c for c in reader.fieldnames or []}
        if any(c.lower() not in names for c in COLUMNS):
            return None, "a required column is missing"
        subject, stimulus, response, confidence = (names[c.lower()] for c in COLUMNS)
        rows = {}
        for row in reader:
            who = (row[subject] or "").strip()
            values = (_label(row[stimulus]), _label(row[response]), _number(row[confidence]))
            if who == "" or any(v is None for v in values):
                continue
            rows.setdefault(who, []).append(values)
    if not rows:
        return None, "no valid row"
    table = [v for person in rows.values() for v in person]
    stimuli, responses = sorted({v[0] for v in table}), sorted({v[1] for v in table})
    if len(stimuli) != 2 or stimuli != responses:
        return None, "not a two-answer task"
    conf = np.array([v[2] for v in table])
    if np.unique(conf).size < MIN_CONFIDENCE_LEVELS:
        return None, "confidence with fewer than three values"
    correct = np.array([v[0] == v[1] for v in table])
    if not conf[correct].mean() > conf[~correct].mean():
        return None, "confidence not higher on correct responses"
    one = stimuli[1]
    return {k: {"stimulus": np.array([v[0] == one for v in person], int),
                "response": np.array([v[1] == one for v in person], int),
                "confidence": np.array([v[2] for v in person], float)} for k, person in rows.items()}, "read"


def included(p: dict) -> bool:
    n = p["stimulus"].size
    if n < MIN_TRIALS:
        return False
    accuracy = float((p["stimulus"] == p["response"]).mean())
    share = float(p["stimulus"].mean())
    return (ACCURACY[0] <= accuracy <= ACCURACY[1] and MIN_CLASS_SHARE <= share <= 1 - MIN_CLASS_SHARE
            and np.unique(p["confidence"]).size >= MIN_CONFIDENCE_LEVELS)


def measure_file(name: str, people: dict) -> list[dict]:
    """One record per included person: the bias, the two correctors, and the same with the trials shuffled."""
    keys = [k for k in people if included(people[k])]
    if len(keys) < MIN_PEOPLE:
        return []
    out, kept = [], []
    for k in keys:
        p = people[k]
        m = readable.person(p["stimulus"], p["response"], p["confidence"])
        seed = int(hashlib.sha1(f"{name}|{k}".encode()).hexdigest()[:12], 16)
        order = np.random.default_rng(seed).permutation(p["stimulus"].size)
        s = readable.person(p["stimulus"][order], p["response"][order], p["confidence"][order])
        out.append({"file": name, "person": len(out) + 1, "n": m["n"], "n_first": m["n_first"],
                    "accuracy": float((p["stimulus"] == p["response"]).mean()),
                    "b": m["b"], "b_first": m["b_first"], "gain_personal": m["gain_personal"],
                    "gain_moments": m["gain_moments"],
                    "d_second": m["d_second"], "b_shuffled": s["b"], "gain_shuffled": s["gain_personal"],
                    **{k: m[k] for k in ("accuracy_before", "accuracy_after", "changed", "changed_to_right",
                                         "changed_to_wrong")}})
        kept.append(m)
    xtx = sum(m["_moments_first"][0] for m in kept)
    xty = sum(m["_moments_first"][1] for m in kept)
    n_first = sum(m["n_first"] for m in kept)
    others = [(xtx - m["_moments_first"][0], xty - m["_moments_first"][1], n_first - m["n_first"]) for m in kept]
    for rec, m, o in zip(out, kept, others):
        rec["gain_everyone"] = readable.gain(*m["_second"], readable.slopes(o[0], o[1]))
        rec["gain_sq_predicted"] = readable.predicted_gain_sq(m["b_first"], m["n_first"])
    # added after the confirmation run: a corrector between the two, the slopes of the person pulled towards those
    # of the others. How hard is chosen file by file inside the first halves only (first quarter against second).
    def missed(weight: float) -> float:
        total = 0.0
        for m, o in zip(kept, others):
            (q1x, q1y, _, _), (q2x, q2y, q2g, _) = m["_quarters"]
            beta = readable.shrunk_slopes((q1x, q1y), *o, weight)
            total += q2g - 2.0 * float(beta @ q2y) + float(beta @ q2x @ beta)
        return total
    weight = min(SHRINK_GRID, key=missed)
    for rec, m, o in zip(out, kept, others):
        rec["gain_between"] = readable.gain(*m["_second"], readable.shrunk_slopes(m["_moments_first"], *o, weight))
        rec["weight_of_the_group"] = weight
    return out


def _by_file(records: list[dict], key: str) -> list[np.ndarray]:
    files = {}
    for r in records:
        if r.get(key) is not None:
            files.setdefault(r["file"], []).append(r[key])
    return [np.asarray(v, float) for _, v in sorted(files.items())]


def median_ci(records: list[dict], key: str, rng: np.random.Generator) -> dict:
    """Median over people, and the 95 per cent interval resampling the files."""
    groups = _by_file(records, key)
    if not groups:
        return {"median": None, "lo": None, "hi": None, "people": 0, "files": 0}
    pooled = np.concatenate(groups)
    draws = np.empty(RESAMPLES)
    for i in range(RESAMPLES):
        pick = rng.integers(0, len(groups), len(groups))
        draws[i] = np.median(np.concatenate([groups[j] for j in pick]))
    lo, hi = np.quantile(draws, [0.025, 0.975])
    return {"median": float(np.median(pooled)), "lo": float(lo), "hi": float(hi), "people": int(pooled.size),
            "files": len(groups), "p10": float(np.quantile(pooled, 0.10)), "p90": float(np.quantile(pooled, 0.90))}


def formula(records: list[dict], rng: np.random.Generator) -> dict:
    """P3: measured gain squared against (1 + B_1) / (1 + 3 / n_1), person by person."""
    rows = [r for r in records if r["gain_personal"] is not None]
    for r in rows:
        r["_gap"] = r["gain_personal"] ** 2 - r["gain_sq_predicted"]
    files = sorted({r["file"] for r in rows})
    by_file = {f: [(r["gain_personal"] ** 2, r["gain_sq_predicted"]) for r in rows if r["file"] == f] for f in files}
    measured, predicted = (np.array([r["gain_personal"] ** 2 for r in rows]),
                           np.array([r["gain_sq_predicted"] for r in rows]))
    rho = float(spearmanr(measured, predicted).statistic)
    draws = np.empty(RESAMPLES)
    for i in range(RESAMPLES):
        pairs = np.array([p for j in rng.integers(0, len(files), len(files)) for p in by_file[files[j]]])
        draws[i] = spearmanr(pairs[:, 0], pairs[:, 1]).statistic
    lo, hi = np.quantile(draws, [0.025, 0.975])
    gap = median_ci(rows, "_gap", rng)
    for r in rows:
        del r["_gap"]
    return {"gap": gap, "spearman": rho, "spearman_lo": float(lo), "spearman_hi": float(hi)}


def summary(records: list[dict], seed: int) -> dict:
    rng = np.random.default_rng(seed)
    if not records:
        return {"people": 0}
    b = np.array([r["b"] for r in records])
    both = [r for r in records if r["gain_personal"] is not None and r["gain_everyone"] is not None]
    for r in both:
        r["_edge"] = r["gain_personal"] - r["gain_everyone"]
    out = {"people": len(records), "files": len({r["file"] for r in records}),
           "trials_median": float(np.median([r["n"] for r in records])),
           "accuracy_median": float(np.median([r["accuracy"] for r in records])),
           "b": median_ci(records, "b", rng), "b_shuffled": median_ci(records, "b_shuffled", rng),
           "gain_personal": median_ci(records, "gain_personal", rng),
           "gain_moments": median_ci(records, "gain_moments", rng),
           "gain_everyone": median_ci(records, "gain_everyone", rng),
           "gain_shuffled": median_ci(records, "gain_shuffled", rng),
           "formula": formula(records, rng),
           "personal_minus_everyone": median_ci(both, "_edge", rng),
           "share_personal_above_everyone": float(np.mean([r["_edge"] > 0 for r in both])) if both else None,
           "share_b_above_0_05": float((b > 0.05).mean()), "share_b_above_0_2": float((b > 0.2).mean()),
           "share_gain_above_1": float(np.mean([r["gain_personal"] > 1 for r in both])) if both else None,
           "gain_not_computed": sum(r["gain_personal"] is None for r in records)}
    for r in both:
        del r["_edge"]
    out["corrector_angle_at_median_b"] = readable.corrector_angle(out["b"]["median"])
    # added after the confirmation run, descriptive: the same medians file by file (the file is the unit resampled)
    by_file = []
    for name in sorted({r["file"] for r in records}):
        rs = [r for r in records if r["file"] == name]
        gains = [r["gain_personal"] for r in rs if r["gain_personal"] is not None]
        by_file.append({"file": name, "people": len(rs), "b": float(np.median([r["b"] for r in rs])),
                        "gain_personal": float(np.median(gains)) if gains else None})
    out["by_file"] = by_file
    # added after the confirmation run, descriptive: the answers themselves, the corrector in between, the tails
    for key in ("accuracy_before", "accuracy_after", "changed", "changed_to_right", "changed_to_wrong", "gain_between"):
        out[key] = median_ci(records, key, rng)
    net = [r for r in records if r["accuracy_after"] is not None]
    for r in net:
        r["_net"] = r["accuracy_after"] - r["accuracy_before"]
    out["accuracy_change"] = median_ci(net, "_net", rng)
    out["accuracy_change_mean"] = float(np.mean([r["_net"] for r in net]))
    out["share_accuracy_up"] = float(np.mean([r["_net"] > 0 for r in net]))
    out["share_accuracy_down"] = float(np.mean([r["_net"] < 0 for r in net]))
    for r in net:
        del r["_net"]
    cut = float(np.quantile([r["b_first"] for r in records], 0.9))
    top = [r for r in records if r["b_first"] >= cut and r["gain_personal"] is not None]
    out["top_tenth_by_b_first"] = {"people": len(top), "b_first_at_least": cut,
                                   "b_first_median": float(np.median([r["b_first"] for r in top])),
                                   "gain_personal_median": float(np.median([r["gain_personal"] for r in top])),
                                   "gain_predicted_median": float(np.median([math.sqrt(max(r["gain_sq_predicted"], 0.0)) for r in top]))}
    out["weights_of_the_group"] = sorted({(r["file"], r["weight_of_the_group"]) for r in records})
    out["share_between_above_personal"] = float(np.mean([r["gain_between"] > r["gain_personal"] for r in both
                                                          if r.get("gain_between") is not None]))
    out["share_between_above_everyone"] = float(np.mean([r["gain_between"] > r["gain_everyone"] for r in both
                                                          if r.get("gain_between") is not None]))
    out["files_b_above_zero"] = sum(f["b"] > 0 for f in by_file)
    out["files_gain_above_1"] = sum(f["gain_personal"] is not None and f["gain_personal"] > 1 for f in by_file)
    out["b_median_of_files"] = float(np.median([f["b"] for f in by_file]))
    out["gain_median_of_files"] = float(np.median([f["gain_personal"] for f in by_file if f["gain_personal"] is not None]))
    return out


def criteria(s: dict) -> dict:
    """P1 to P3 of the protocol, read on the confirmation files."""
    if not s.get("people"):
        return {}
    b, sh, g, f = s["b"], s["b_shuffled"], s["gain_personal"], s["formula"]
    return {"P1_there": bool(b["median"] > 0 and b["lo"] > 0
                             and (sh["lo"] <= 0 <= sh["hi"] or abs(sh["median"]) < 0.002)),
            "P2_small": bool(g["median"] < 1.02),
            "P3_formula": bool(-0.01 <= f["gap"]["median"] <= 0.01 and f["spearman"] > 0 and f["spearman_lo"] > 0)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT / "data" / "confidence_db"))
    ap.add_argument("--only", choices=("exploration", "confirmation"))
    ap.add_argument("--out", default=str(ROOT / "results" / "e13_people_bias.json"))
    args = ap.parse_args()
    files, records = [], []
    for path in sorted(Path(args.data).glob("data_*.csv")):
        part = split_of(path.name)
        if args.only and part != args.only:
            continue
        people, status = read_file(path)
        recs = measure_file(path.name, people) if people else []
        if people and not recs:
            status = "fewer than ten people included"
        files.append({"file": path.name, "split": part, "status": status if not recs else "included",
                      "people_read": len(people) if people else 0, "people_included": len(recs)})
        for r in recs:
            r["split"] = part
        records += recs
    out = {"protocol": "§ 19, E13", "files": files, "people": records,
           "files_by_split": {part: {"read": sum(f["split"] == part for f in files),
                                     "included": sum(f["split"] == part and f["status"] == "included" for f in files)}
                              for part in ("exploration", "confirmation")}}
    for part, seed in (("exploration", 13), ("confirmation", 1313)):
        out[part] = summary([r for r in records if r["split"] == part], seed)
    out["criteria"] = criteria(out["confirmation"])
    write_json(args.out, out)
    for part in ("exploration", "confirmation"):
        s = out[part]
        if s.get("people"):
            print(part, s["people"], "people in", s["files"], "files; B", round(s["b"]["median"], 4),
                  "gain", round(s["gain_personal"]["median"], 4))
    print(out["criteria"])


if __name__ == "__main__":
    main()
